"""rank_mscomplex.py — ranking within plan на MSComplexTasks.

Отвечает на вопрос: "дано K уже выполненных подзадач из DAG,
какую из оставшихся recommend() поставит первой?"

Это inference-time ranking, а НЕ обучение переходов A→B.
assoc может быть random / warm-start на MS-LaTTE / обученным —
тест покажет, что важно.

Метрики:
    hit@1, hit@3, hit@5 — попала ли "истинная следующая" (по DAG) в топ
    MRR                 — Mean Reciprocal Rank, основной сигнал
    rank_of_true        — средний ранг истинной следующей

Baselines:
    random     — случайный порядок пула
    reverse    — обратный порядок (худшее, что может быть)
    alphabetical — по алфавиту
    seq_from_dag — "предсказать ровно следующую" (как в прошлом тесте)

Запуск:
    uv run rank_mscomplex.py
    uv run rank_mscomplex.py --assoc random
    uv run rank_mscomplex.py --assoc identity
    uv run rank_mscomplex.py --assoc latent --latent-dim 64 --latent-seed 42
    uv run rank_mscomplex.py --min-pool 5   # только шаги с >=5 оставшихся
"""
from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from pathlib import Path

import torch

from mind_engine import MindEngine, Concept, Thought, pick_device
from mscomplex_loader import load_mscomplex, build_sessions_from_mscomplex


# ============================================================
# 1. Encoder
# ============================================================

class Encoder:
    """text → signature через MiniLM + сопоставление с vocab.

    Для каждой подзадачи: эмбеддинг → top-k ближайших концептов в vocab
    через cosine. Если vocab = набор всех подзадач, это фактически
    "какая из остальных подзадач семантически ближе всего".
    """
    def __init__(self, concepts: list[Concept], *, model_name: str,
                 device: torch.device, top_k: int = 5, temperature: float = 0.3):
        from sentence_transformers import SentenceTransformer
        self.concepts = concepts
        self.concept_ids = [c.id for c in concepts]
        self.top_k = min(top_k, len(concepts))
        self.temperature = temperature
        self.device = device

        model = SentenceTransformer(model_name, device=str(device))
        self.concept_emb = torch.tensor(
            model.encode(self.concept_ids, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False),
            dtype=torch.float32, device=device,
        )
        self.model = model
        # warmup
        self.model.encode(["warmup"], normalize_embeddings=True)

    @torch.no_grad()
    def __call__(self, text: str) -> dict[str, float]:
        e = torch.tensor(
            self.model.encode([text], normalize_embeddings=True,
                              convert_to_numpy=True)[0],
            dtype=torch.float32, device=self.device,
        )
        sims = self.concept_emb @ e
        vals, idx = torch.topk(sims, self.top_k)
        weights = torch.softmax(vals / self.temperature, dim=0)
        return {
            self.concept_ids[i]: float(w)
            for i, w in zip(idx.tolist(), weights.tolist())
            if w > 0.05
        }


# ============================================================
# 2. Baseline strategies
# ============================================================

def rank_random(pool, rng):
    pool = list(pool)
    rng.shuffle(pool)
    return pool

def rank_reverse(pool):
    return list(reversed(list(pool)))

def rank_alphabetical(pool):
    return sorted(pool)


# ============================================================
# 3. MindEngine ranking
# ============================================================

def rank_with_mind(mind, encoder, history, pool, *, mode="resonance",
                   resonance_weight=0.5):
    """history = список уже выполненных задач (по порядку).
    pool = список оставшихся задач.
    Возвращает pool, отсортированный по score (best first).
    """
    if not history:
        # нет контекста — ранжирование случайно
        return list(pool)

    mind.reset()
    for task in history:
        sig = encoder(task)
        if sig:
            mind.think(Thought.make(content=task, signature=sig))

    if not pool:
        return []

    # Один recommend() по всему пулу
    library = []
    for t in pool:
        sig = encoder(t)
        if sig:
            library.append(Thought.make(content=t, signature=sig, id=t))

    if not library:
        return list(pool)

    recs = mind.recommend(
        library, mode=mode, top_k=len(library),
        resonance_weight=resonance_weight,
    )
    ranked = [r["thought"].id for r in recs]
    # добавить задачи, чья signature была пустой
    ranked_set = set(ranked)
    for t in pool:
        if t not in ranked_set:
            ranked.append(t)
    return ranked


# ============================================================
# 4. Evaluation
# ============================================================

def evaluate(
    sessions, encoder, *, mind=None, mode="resonance",
    resonance_weight=0.5, strategy="mind", min_pool=2, seed=42,
):
    """Для каждой сессии: итеративно "выполняем" шаги, на каждом шаге
    ранжируем оставшийся пул, проверяем, где истинная следующая."""
    rng = random.Random(seed)
    hits = {1: 0, 3: 0, 5: 0}
    mrr_sum = 0.0
    rank_sum = 0
    total = 0

    for session in sessions:
        if len(session) < 2:
            continue
        for k in range(len(session) - 1):
            history = session[:k + 1]
            true_next = session[k + 1]
            pool = session[k + 1:]   # сама true_next + все оставшиеся
            if len(pool) < min_pool:
                continue
            if true_next not in pool:
                continue

            if strategy == "mind":
                ranked = rank_with_mind(mind, encoder, history, pool,
                                        mode=mode,
                                        resonance_weight=resonance_weight)
            elif strategy == "random":
                ranked = rank_random(pool, rng)
            elif strategy == "reverse":
                ranked = rank_reverse(pool)
            elif strategy == "alphabetical":
                ranked = rank_alphabetical(pool)
            else:
                raise ValueError(strategy)

            rank = ranked.index(true_next) + 1 if true_next in ranked else len(ranked) + 1
            total += 1
            rank_sum += rank
            mrr_sum += 1.0 / rank
            for kk in (1, 3, 5):
                if rank <= kk:
                    hits[kk] += 1

    if total == 0:
        return {"hit@1": 0.0, "hit@3": 0.0, "hit@5": 0.0,
                "MRR": 0.0, "avg_rank": float("inf"), "n": 0}
    return {
        "hit@1": hits[1] / total,
        "hit@3": hits[3] / total,
        "hit@5": hits[5] / total,
        "MRR": mrr_sum / total,
        "avg_rank": rank_sum / total,
        "n": total,
    }


# ============================================================
# 5. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="task_subtask_dataset.json")
    parser.add_argument("--min-session-len", type=int, default=3)
    parser.add_argument("--max-session-len", type=int, default=15)
    parser.add_argument("--min-pool", type=int, default=2,
                        help="минимальный размер пула для оценки")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--mode", choices=["resonance", "homeostasis", "mixed"],
                        default="resonance")
    parser.add_argument("--resonance-weight", type=float, default=0.5)
    parser.add_argument("--forgetting", type=float, default=0.85)
    parser.add_argument("--encoder-top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-json", default="rank_mscomplex_results.json")
    args = parser.parse_args()

    device = pick_device(args.device)
    print("=" * 70)
    print("  rank_mscomplex.py — ranking within plan")
    print("=" * 70)
    print(f"  device: {device}  mode={args.mode}  forgetting={args.forgetting}")

    print("\n[1] MSComplexTasks...")
    entries = load_mscomplex(args.data)
    print(f"  {len(entries)} задач")

    print("\n[2] Сессии из DAG (топологическая сортировка)...")
    raw_sessions = build_sessions_from_mscomplex(
        entries,
        min_subtasks=args.min_session_len,
        max_subtasks=args.max_session_len,
        require_order=True,
        verbose=True,
    )
    # дедупликация
    seen = set()
    sessions = []
    for s in raw_sessions:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            sessions.append(s)
    print(f"  после дедупликации: {len(sessions)}")
    print("\n  пример сессии:")
    print(f"    {' → '.join(sessions[0][:8])}")

    print("\n[3] Split...")
    rng = random.Random(args.seed)
    shuffled = list(sessions)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * args.train_frac)
    train_sessions = shuffled[:n_train]
    test_sessions = shuffled[n_train:]
    print(f"  train={len(train_sessions)}  test={len(test_sessions)}")

    # Vocab: все уникальные названия подзадач, чтобы encoder мог найти
    # семантически близкие концепты
    print("\n[4] Vocab (все подзадачи как концепты)...")
    vocab = sorted({n for s in sessions for n in s})
    print(f"  уникальных подзадач: {len(vocab)}")
    concepts = [Concept(v) for v in vocab]

    print("\n[5] Encoder (MiniLM)...")
    t0 = time.perf_counter()
    encoder = Encoder(
        concepts, model_name=args.model, device=device,
        top_k=args.encoder_top_k,
    )
    print(f"  готов за {time.perf_counter()-t0:.1f}s  "
          f"(top_k={encoder.top_k})")

    print("\n[6] MindEngine (assoc = random — state всё равно несёт контекст)...")
    mind = MindEngine(
        concepts,
        associations="random", seed=args.seed,
        mode="dense" if len(concepts) < 500 else "low_rank",
        rank=128,
        forgetting=args.forgetting,
        device="auto",
    )
    print(f"  mode={mind.mode}  n={mind.n}")

    print(f"\n[7] Baseline strategies на test:")
    results = {}
    for strategy in ("random", "reverse", "alphabetical"):
        t0 = time.perf_counter()
        m = evaluate(test_sessions, encoder, strategy=strategy,
                     min_pool=args.min_pool, seed=args.seed)
        results[strategy] = m
        print(f"  {strategy:14s}  hit@1={m['hit@1']:.3f}  "
              f"hit@3={m['hit@3']:.3f}  hit@5={m['hit@5']:.3f}  "
              f"MRR={m['MRR']:.3f}  avg_rank={m['avg_rank']:.1f}  "
              f"n={m['n']}  ({time.perf_counter()-t0:.1f}s)")

    print(f"\n[8] MindEngine (mode={args.mode}, forgetting={args.forgetting}) на test:")
    t0 = time.perf_counter()
    m = evaluate(
        test_sessions, encoder, mind=mind, strategy="mind",
        mode=args.mode, resonance_weight=args.resonance_weight,
        min_pool=args.min_pool, seed=args.seed,
    )
    results["mind"] = m
    print(f"  {'mind':14s}  hit@1={m['hit@1']:.3f}  "
          f"hit@3={m['hit@3']:.3f}  hit@5={m['hit@5']:.3f}  "
          f"MRR={m['MRR']:.3f}  avg_rank={m['avg_rank']:.1f}  "
          f"n={m['n']}  ({time.perf_counter()-t0:.1f}s)")

    print("\n[9] Итог:")
    print(f"  {'strategy':14s}  {'hit@1':>7s}  {'hit@3':>7s}  {'hit@5':>7s}  "
          f"{'MRR':>7s}  {'avg_rank':>9s}")
    for name in ("random", "reverse", "alphabetical", "mind"):
        r = results[name]
        print(f"  {name:14s}  {r['hit@1']:>7.3f}  {r['hit@3']:>7.3f}  "
              f"{r['hit@5']:>7.3f}  {r['MRR']:>7.3f}  {r['avg_rank']:>9.1f}")

    # Δ MindEngine vs random
    delta = results["mind"]["MRR"] - results["random"]["MRR"]
    print(f"\n  Δ MRR (mind - random) = {delta:+.3f}")

    # Примеры: как MindEngine ранжирует на 5 случайных шагах
    print("\n[10] Примеры ранжирования (те же 5 шагов, разные стратегии):")
    sample_session = test_sessions[0] if test_sessions else None
    if sample_session and len(sample_session) >= 3:
        k = len(sample_session) // 2
        history = sample_session[:k]
        true_next = sample_session[k]
        pool = sample_session[k:]
        print(f"  История: {' → '.join(history)}")
        print(f"  Пул ({len(pool)}): {pool}")
        print(f"  Истинная следующая (по DAG): {true_next}")
        ranked = rank_with_mind(mind, encoder, history, pool,
                                 mode=args.mode,
                                 resonance_weight=args.resonance_weight)
        print(f"  MindEngine ranking: {ranked}")
        print(f"  Позиция истинной: {ranked.index(true_next)+1}/{len(ranked)}")

    # сохранение
    import json
    with Path(args.save_json).open("w", encoding="utf-8") as f:
        json.dump({k: {kk: (float(vv) if isinstance(vv, (int, float)) else vv)
                       for kk, vv in v.items()}
                   for k, v in results.items()},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[11] результаты сохранены в {args.save_json}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()