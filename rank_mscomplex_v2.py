"""rank_mscomplex_v2.py — правильный тест ранжирования на MSComplexTasks.

Отличия от v1:
    1. Distractor pool: в пул добавляются K случайных подзадач из
       ДРУГИХ сессий. Пул = 15-30 задач, имитирует "план + шум".
    2. Valid-next: правильны ВСЕ задачи, у которых все parents
       в history. DAG — это частичный порядок, не полный.
    3. Метрика: rank лучшей valid-задачи в пуле, MRR, hit@k.
    4. Отдельно проверяется, ставит ли модель "invalid" (с невыполненными
       parents) выше "valid" — это ключевое свойство понимания DAG.

Запуск:
    uv run rank_mscomplex_v2.py
    uv run rank_mscomplex_v2.py --n-distractors 30 --min-history 2
    uv run rank_mscomplex_v2.py --mode homeostasis --forgetting 0.95
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from mind_engine import MindEngine, Concept, Thought, pick_device
from mscomplex_loader import load_mscomplex, build_sessions_from_mscomplex, _topological_sort


# ============================================================
# 1. DAG helpers
# ============================================================

def load_dags(entries) -> dict[tuple, list[tuple]]:
    """sess_key -> список рёбер (parent_name, child_name) в нормализованных именах."""
    dags = {}
    for entry in entries:
        if str(entry.get("subtask_order_matters", "")).strip() != "Yes":
            continue
        subtasks = entry.get("subtasks") or []
        deps = entry.get("subtask_dependencies") or []
        if len(subtasks) < 2 or not deps:
            continue
        id_to_name = {
            str(s["id"]): s["name"].strip().lower()
            for s in subtasks if "id" in s and "name" in s
        }
        edges = []
        for d in deps:
            if "parent" in d and "child" in d:
                p = id_to_name.get(str(d["parent"]))
                c = id_to_name.get(str(d["child"]))
                if p and c and p != c:
                    edges.append((p, c))
        order = _topological_sort(list(id_to_name.keys()), [
            (str(d["parent"]), str(d["child"])) for d in deps
            if "parent" in d and "child" in d
        ])
        if order is None:
            continue
        session = tuple(id_to_name[sid] for sid in order if sid in id_to_name)
        if len(session) >= 2:
            dags[session] = edges
    return dags


def valid_nexts(session, dag_edges, history_set, remaining):
    """Все задачи из remaining, у которых все parents в history_set."""
    parents = defaultdict(set)
    for p, c in dag_edges:
        parents[c].add(p)
    return [t for t in remaining if parents[t].issubset(history_set)]


# ============================================================
# 2. Encoder
# ============================================================

class Encoder:
    def __init__(self, concepts, *, model_name, device, top_k=5, temperature=0.3):
        from sentence_transformers import SentenceTransformer
        self.ids = [c.id for c in concepts]
        self.top_k = min(top_k, len(concepts))
        self.temperature = temperature
        self.device = device
        model = SentenceTransformer(model_name, device=str(device))
        self.concept_emb = torch.tensor(
            model.encode(self.ids, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False),
            dtype=torch.float32, device=device,
        )
        self.model = model
        self.model.encode(["warmup"], normalize_embeddings=True)

    @torch.no_grad()
    def __call__(self, text):
        e = torch.tensor(
            self.model.encode([text], normalize_embeddings=True,
                              convert_to_numpy=True)[0],
            dtype=torch.float32, device=self.device,
        )
        sims = self.concept_emb @ e
        vals, idx = torch.topk(sims, self.top_k)
        w = torch.softmax(vals / self.temperature, dim=0)
        return {
            self.ids[i]: float(wi)
            for i, wi in zip(idx.tolist(), w.tolist())
            if wi > 0.05
        }


# ============================================================
# 3. One step = rank pool by mind + return best match
# ============================================================

def rank_pool(mind, encoder, history, pool, *, mode="resonance",
              resonance_weight=0.5):
    """history = list[str] выполненных задач.
    pool = list[str] всех кандидатов (in-task + distractors).
    Возвращает pool отсортированный по релевантности.
    """
    mind.reset()
    for t in history:
        sig = encoder(t)
        if sig:
            mind.think(Thought.make(content=t, signature=sig))

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
    ranked_set = set(ranked)
    for t in pool:
        if t not in ranked_set:
            ranked.append(t)
    return ranked


# ============================================================
# 4. Evaluation with distractor pool
# ============================================================

def evaluate_v2(
    sessions, dags, encoder, *,
    mind=None, strategy="mind", mode="resonance",
    resonance_weight=0.5, forgetting=0.85,
    n_distractors=20, min_history=1, min_pool=5,
    seed=42,
):
    """Для каждого шага: pool = remaining in-task + n_distractors случайных.

    Метрики:
        hit@1/3/5 — попала ли любая valid-задача в топ-k
        MRR       — 1/rank лучшей valid-задачи
        invalid_above_valid — доля шагов, где хотя бы одна invalid
                              стоит выше лучшей valid
    """
    rng = random.Random(seed)
    all_tasks = list({t for s in sessions for t in s})
    if len(all_tasks) < n_distractors + 5:
        n_distractors = max(2, len(all_tasks) // 2)

    hits = {1: 0, 3: 0, 5: 0}
    mrr_sum = 0.0
    rank_sum = 0
    total = 0
    invalid_above_valid = 0

    for session in sessions:
        if len(session) < min_history + 2:
            continue
        edges = dags.get(session, [])
        if not edges:
            continue
        in_task_set = set(session)

        for k in range(min_history, len(session) - 1):
            history = list(session[:k + 1])
            history_set = set(history)
            remaining = list(session[k + 1:])
            if not remaining:
                continue

            # валидные цели
            valid = set(valid_nexts(session, edges, history_set, remaining))
            if not valid:
                continue

            # distractors: случайные задачи из других сессий
            distractors = []
            for _ in range(n_distractors * 3):
                if len(distractors) >= n_distractors:
                    break
                cand = rng.choice(all_tasks)
                if cand not in in_task_set and cand not in distractors:
                    distractors.append(cand)

            pool = remaining + distractors
            if len(pool) < min_pool:
                continue
            rng.shuffle(pool)

            if strategy == "mind":
                ranked = rank_pool(mind, encoder, history, pool,
                                    mode=mode,
                                    resonance_weight=resonance_weight)
            elif strategy == "random":
                ranked = list(pool)
                rng.shuffle(ranked)
            elif strategy == "alphabetical":
                ranked = sorted(pool)
            else:
                raise ValueError(strategy)

            # позиции
            rank_of_valid = [
                ranked.index(v) + 1 for v in valid if v in ranked
            ]
            if not rank_of_valid:
                continue
            best_rank = min(rank_of_valid)

            total += 1
            rank_sum += best_rank
            mrr_sum += 1.0 / best_rank
            for kk in (1, 3, 5):
                if best_rank <= kk:
                    hits[kk] += 1

            # invalid above valid?
            best_valid_pos = best_rank
            inv_positions = [
                ranked.index(t) + 1 for t in pool
                if t not in valid and t not in history_set
                and t in in_task_set  # invalid только из этой задачи
            ]
            if any(p < best_valid_pos for p in inv_positions):
                invalid_above_valid += 1

    if total == 0:
        return {"hit@1": 0, "hit@3": 0, "hit@5": 0,
                "MRR": 0, "avg_rank": 0, "invalid_above_valid": 0, "n": 0}
    return {
        "hit@1": hits[1] / total,
        "hit@3": hits[3] / total,
        "hit@5": hits[5] / total,
        "MRR": mrr_sum / total,
        "avg_rank": rank_sum / total,
        "invalid_above_valid": invalid_above_valid / total,
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
    parser.add_argument("--min-history", type=int, default=1)
    parser.add_argument("--min-pool", type=int, default=5)
    parser.add_argument("--n-distractors", type=int, default=20)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--mode", choices=["resonance", "homeostasis", "mixed"],
                        default="resonance")
    parser.add_argument("--resonance-weight", type=float, default=0.5)
    parser.add_argument("--forgetting", type=float, default=0.85)
    parser.add_argument("--encoder-top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = pick_device(args.device)
    print("=" * 70)
    print("  rank_mscomplex_v2 — distractor pool + valid-next")
    print("=" * 70)
    print(f"  device: {device}  mode={args.mode}  forgetting={args.forgetting}  "
          f"distractors={args.n_distractors}")

    print("\n[1] MSComplexTasks...")
    entries = load_mscomplex(args.data)
    print(f"  {len(entries)} задач")

    print("\n[2] DAG-сессии...")
    dags = load_dags(entries)
    sessions = list(dags.keys())
    print(f"  сессий с DAG: {len(sessions)}")

    print("\n[3] Split...")
    rng = random.Random(args.seed)
    shuffled = list(sessions)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * args.train_frac)
    test_sessions = shuffled[n_train:]
    print(f"  train={n_train}  test={len(test_sessions)}")

    print("\n[4] Vocab + encoder...")
    vocab = sorted({t for s in sessions for t in s})
    print(f"  уникальных подзадач: {len(vocab)}")
    t0 = time.perf_counter()
    encoder = Encoder(
        [Concept(v) for v in vocab],
        model_name=args.model, device=device,
        top_k=args.encoder_top_k,
    )
    print(f"  encoder готов за {time.perf_counter()-t0:.1f}s")

    print("\n[5] MindEngine...")
    mind = MindEngine(
        [Concept(v) for v in vocab],
        associations="random", seed=args.seed,
        mode="low_rank", rank=128, forgetting=args.forgetting,
        device="auto",
    )
    print(f"  mode={mind.mode}  n={mind.n}")

    print(f"\n[6] Baselines на test (pool = in-task + {args.n_distractors} distractors):")
    results = {}
    for strategy in ("random", "alphabetical"):
        t0 = time.perf_counter()
        m = evaluate_v2(
            test_sessions, dags, encoder,
            strategy=strategy, mode=args.mode,
            resonance_weight=args.resonance_weight,
            forgetting=args.forgetting,
            n_distractors=args.n_distractors,
            min_history=args.min_history, min_pool=args.min_pool,
            seed=args.seed,
        )
        results[strategy] = m
        print(f"  {strategy:12s}  hit@1={m['hit@1']:.3f}  hit@3={m['hit@3']:.3f}  "
              f"hit@5={m['hit@5']:.3f}  MRR={m['MRR']:.3f}  "
              f"avg_rank={m['avg_rank']:.1f}  "
              f"inv>val={m['invalid_above_valid']:.3f}  n={m['n']}")

    print(f"\n[7] MindEngine (mode={args.mode}, forgetting={args.forgetting}):")
    t0 = time.perf_counter()
    m = evaluate_v2(
        test_sessions, dags, encoder,
        mind=mind, strategy="mind", mode=args.mode,
        resonance_weight=args.resonance_weight,
        forgetting=args.forgetting,
        n_distractors=args.n_distractors,
        min_history=args.min_history, min_pool=args.min_pool,
        seed=args.seed,
    )
    results["mind"] = m
    print(f"  {'mind':12s}  hit@1={m['hit@1']:.3f}  hit@3={m['hit@3']:.3f}  "
          f"hit@5={m['hit@5']:.3f}  MRR={m['MRR']:.3f}  "
          f"avg_rank={m['avg_rank']:.1f}  "
          f"inv>val={m['invalid_above_valid']:.3f}  n={m['n']}  "
          f"({time.perf_counter()-t0:.1f}s)")

    print("\n[8] Итог:")
    print(f"  {'strategy':12s}  {'hit@1':>7s}  {'hit@3':>7s}  {'hit@5':>7s}  "
          f"{'MRR':>7s}  {'avg_rank':>9s}  {'inv>val':>9s}")
    for name in ("random", "alphabetical", "mind"):
        r = results[name]
        print(f"  {name:12s}  {r['hit@1']:>7.3f}  {r['hit@3']:>7.3f}  "
              f"{r['hit@5']:>7.3f}  {r['MRR']:>7.3f}  "
              f"{r['avg_rank']:>9.1f}  {r['invalid_above_valid']:>9.3f}")

    print(f"\n  Δ MRR (mind - random) = "
          f"{results['mind']['MRR'] - results['random']['MRR']:+.3f}")
    print(f"  Δ inv>val (mind - random) = "
          f"{results['mind']['invalid_above_valid'] - results['random']['invalid_above_valid']:+.3f}")

    # Пример
    print("\n[9] Пример на одной сессии:")
    if test_sessions:
        s = test_sessions[0]
        edges = dags[s]
        k = min(2, len(s) - 2)
        history = list(s[:k + 1])
        remaining = list(s[k + 1:])
        valid = set(valid_nexts(s, edges, set(history), remaining))
        pool = remaining + random.Random(0).sample(
            [t for t in vocab if t not in set(s)],
            min(args.n_distractors, len(vocab) - len(set(s))),
        )
        print(f"  history ({len(history)}): {history}")
        print(f"  valid next ({len(valid)}): {list(valid)[:3]}")
        print(f"  pool size: {len(pool)}")
        ranked = rank_pool(mind, encoder, history, pool,
                            mode=args.mode,
                            resonance_weight=args.resonance_weight)
        print(f"  mind top-5: {ranked[:5]}")
        best = min(ranked.index(v) + 1 for v in valid if v in ranked)
        print(f"  лучшая valid на позиции: {best}/{len(pool)}")

    Path("rank_mscomplex_v2_results.json").write_text(
        json.dumps({k: {kk: float(vv) for kk, vv in v.items()}
                    for k, v in results.items()},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()