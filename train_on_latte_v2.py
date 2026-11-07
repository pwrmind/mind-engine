"""train_on_latte_v2.py — task-to-task co-occurrence + class-balanced context.

Изменения в v2.1 (после первого прогона):
    1. time_accuracy и loc_accuracy измеряются РАЗДЕЛЬНО — top-k
       считается только по 10 временам (или только по 3 локациям),
       а не по смеси 13 контекстов. Честные baseline: 1/10 и 1/3.
    2. class_balanced=True: loss перевешивает редкие цели (конкретное
       время) против частых (home). Без этого модель ставит home первым
       для ~90% задач, а правильное время уходит на rank 2-3.
    3. rank=128 по умолчанию (n=163, place for both objectives).
    4. Диагностика: для каждой задачи печатается место (rank),
       на котором модель поставила истинное время.
    5. λ_task=0.3 по умолчанию (context доминирует, task-to-task
       остаётся сильным, но не забивает контекст).

Запуск:
    uv run train_on_latte_v2.py
    uv run train_on_latte_v2.py --no-text --lambda-task 0.0   # = v1
    uv run train_on_latte_v2.py --class-balance 0.0            # без balance
    uv run train_on_latte_v2.py --rank 128 --epochs 500
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam

from mind_engine import MindEngine, Concept, pick_device


TIME_LABELS = [
    "wd-morning", "wd-afternoon", "wd-evening", "wd-night", "wd-anytime",
    "we-morning", "we-afternoon", "we-evening", "we-night", "we-anytime",
]
LOC_LABELS = ["home", "work", "public"]


# ============================================================
# 1. Загрузка
# ============================================================

def load_latte(path: str = "MS-LaTTE.json") -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} не найден")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for v in raw.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    raise ValueError(f"Не удалось найти список записей в {path}")


def _is_known(j: dict) -> bool:
    v = j.get("Known")
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("yes", "true", "1", "y")
    return bool(v)


def _as_labels(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [s.strip().lower() for s in x.split(",") if s.strip()]
    if isinstance(x, list):
        out = []
        for s in x:
            if isinstance(s, str):
                out.extend(t.strip().lower() for t in s.split(",") if t.strip())
        return out
    return []


def _as_judgements(x) -> list[dict]:
    if x is None:
        return []
    if isinstance(x, list):
        return [j for j in x if isinstance(j, dict)]
    if isinstance(x, dict):
        return [j for j in x.values() if isinstance(j, dict)]
    return []


def collect_annotations(tasks, *, top_n_tasks=150, min_known=1):
    counts = Counter(str(t.get("TaskTitle", "")).strip().lower() for t in tasks)
    counts.pop("", None)
    top_tasks = [t for t, _ in counts.most_common(top_n_tasks)]

    time_acc: dict[str, Counter] = defaultdict(Counter)
    loc_acc: dict[str, Counter] = defaultdict(Counter)

    for t in tasks:
        title = str(t.get("TaskTitle", "")).strip().lower()
        if title not in top_tasks:
            continue
        for j in _as_judgements(t.get("TimeJudgements")):
            if _is_known(j):
                for label in _as_labels(j.get("Times")):
                    if label in TIME_LABELS:
                        time_acc[title][label] += 1
        for j in _as_judgements(t.get("LocJudgements")):
            if _is_known(j):
                for label in _as_labels(j.get("Locations")):
                    if label in LOC_LABELS:
                        loc_acc[title][label] += 1

    time_dist, loc_dist = {}, {}
    for title in top_tasks:
        tc, lc = time_acc.get(title, Counter()), loc_acc.get(title, Counter())
        if sum(tc.values()) >= min_known:
            time_dist[title] = {k: v / sum(tc.values()) for k, v in tc.items()}
        if sum(lc.values()) >= min_known:
            loc_dist[title] = {k: v / sum(lc.values()) for k, v in lc.items()}
    selected = [t for t in top_tasks if t in time_dist or t in loc_dist]
    return selected, time_dist, loc_dist, counts


# ============================================================
# 2. Similarity
# ============================================================

def build_task_similarity(
    task_ids, time_dist, loc_dist,
    *, w_time=0.5, w_loc=0.1, w_text=0.4,
    device: torch.device = torch.device("cpu"),
    use_text: bool = True,
):
    n = len(task_ids)

    T = torch.zeros(n, len(TIME_LABELS))
    for i, t in enumerate(task_ids):
        for k, lbl in enumerate(TIME_LABELS):
            T[i, k] = time_dist.get(t, {}).get(lbl, 0.0)
    T = T / T.norm(dim=1, keepdim=True).clamp_min(1e-8)
    S_time = T @ T.T

    L = torch.zeros(n, len(LOC_LABELS))
    for i, t in enumerate(task_ids):
        for k, lbl in enumerate(LOC_LABELS):
            L[i, k] = loc_dist.get(t, {}).get(lbl, 0.0)
    L = L / L.norm(dim=1, keepdim=True).clamp_min(1e-8)
    S_loc = L @ L.T

    S_text = torch.zeros(n, n)
    if use_text:
        try:
            from sentence_transformers import SentenceTransformer
            print("  загрузка MiniLM...")
            model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device=str(device),
            )
            embs = torch.tensor(
                model.encode(task_ids, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False),
                dtype=torch.float32, device=device,
            )
            S_text = (embs @ embs.T).cpu()
            print(f"  MiniLM: {embs.shape[0]}×{embs.shape[1]}")
        except Exception as e:
            print(f"  [warn] text недоступен ({e}); только контекст")
            w_text = 0.0
            s = w_time + w_loc
            w_time, w_loc = w_time / s, w_loc / s

    S = w_time * S_time + w_loc * S_loc + w_text * S_text
    S.fill_diagonal_(0.0)
    return S.cpu(), S_time.cpu(), S_loc.cpu(), S_text.cpu()


# ============================================================
# 3. Targets
# ============================================================

def build_targets_v2(
    task_ids, time_dist, loc_dist, similarity, counts,
    *, temperature=0.1, task_task_weight=0.3, freq_weight=0.2,
    device: torch.device = torch.device("cpu"),
):
    n_tasks, n_times, n_locs = len(task_ids), len(TIME_LABELS), len(LOC_LABELS)
    n = n_tasks + n_times + n_locs
    concept_ids = (
        [f"task:{t}" for t in task_ids]
        + [f"time:{t}" for t in TIME_LABELS]
        + [f"loc:{l}" for l in LOC_LABELS]
    )
    time_idx = {t: n_tasks + i for i, t in enumerate(TIME_LABELS)}
    loc_idx = {l: n_tasks + n_times + i for i, l in enumerate(LOC_LABELS)}

    targets = torch.zeros(n, n, dtype=torch.float32)

    for i, task in enumerate(task_ids):
        for tl, p in time_dist.get(task, {}).items():
            targets[time_idx[tl], i] += p
        for ll, p in loc_dist.get(task, {}).items():
            targets[loc_idx[ll], i] += p

    freq = torch.tensor([counts[t] for t in task_ids], dtype=torch.float32)
    freq = freq / freq.sum()
    for i, task in enumerate(task_ids):
        for tl, p in time_dist.get(task, {}).items():
            targets[i, time_idx[tl]] += p * freq[i]
        for ll, p in loc_dist.get(task, {}).items():
            targets[i, loc_idx[ll]] += p * freq[i]

    if task_task_weight > 0:
        logits = similarity.float() / temperature
        logits = logits + freq_weight * torch.log(freq.clamp_min(1e-8)).unsqueeze(1)
        logits.fill_diagonal_(-float("inf"))
        probs = torch.softmax(logits, dim=0)
        targets[:n_tasks, :n_tasks] += task_task_weight * probs

    col_sums = targets.sum(dim=0, keepdim=True).clamp_min(1e-8)
    return concept_ids, (targets / col_sums).to(device)


# ============================================================
# 4. Обучение с class-balance
# ============================================================

def train_assoc_ce(
    mind, targets, *,
    epochs=300, lr=0.01, weight_decay=1e-4,
    class_balance: float = 0.5,   # 0 = off, 0.5 = sqrt inverse freq, 1 = full
    verbose=True,
):
    """Soft CE. class_balance=α: w[b] = 1 / freq[b]^α, потом нормируется
    так, чтобы среднее w было 1. При α>0 редкие цели (конкретное время)
    получают больший вес против частых (home)."""
    opt = Adam([mind.assoc_U, mind.assoc_V], lr=lr, weight_decay=weight_decay)
    hist: list[float] = []

    if class_balance > 0:
        freq = targets.sum(dim=1).clamp_min(1e-8)
        w = (1.0 / freq.pow(class_balance))
        w = w / w.mean()
        w = w.to(targets.device)
        if verbose:
            print(f"    class-balance α={class_balance}: "
                  f"w_range=[{w.min().item():.2f}, {w.max().item():.2f}]")
    else:
        w = None

    for epoch in range(epochs):
        logits = mind.assoc_U @ mind.assoc_V.T
        log_probs = F.log_softmax(logits, dim=0)
        if w is not None:
            loss = -(w.unsqueeze(1) * targets * log_probs).sum()
        else:
            loss = -(targets * log_probs).sum()

        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(loss.item())
        if verbose and (epoch == 0 or (epoch + 1) % 50 == 0):
            print(f"    epoch {epoch + 1:3d}  loss={loss.item():.4f}")
    return hist


@torch.no_grad()
def normalize_assoc(mind: MindEngine, target_scale: float = 0.5) -> float:
    A = mind.assoc_U @ mind.assoc_V.T
    m = A.abs().max().item()
    if m < 1e-8:
        return 1.0
    s = target_scale / m
    mind.assoc_U.data *= s
    return s


# ============================================================
# 5. Оценка (раздельно по категориям)
# ============================================================

@torch.no_grad()
def accuracy_by_category(
    mind, concept_ids, task_ids, time_dist, loc_dist,
    *, category: str, top_k: int,
) -> float:
    """time / loc / all. Для 'time' — кандидаты только 10 времён,
    для 'loc' — только 3 локации. Честные baseline: 1/10 и 1/3."""
    n_tasks = len(task_ids)
    n_times = len(TIME_LABELS)
    n_locs = len(LOC_LABELS)
    if category == "time":
        lo, hi = n_tasks, n_tasks + n_times
    elif category == "loc":
        lo, hi = n_tasks + n_times, n_tasks + n_times + n_locs
    elif category == "all":
        lo, hi = n_tasks, n_tasks + n_times + n_locs
    else:
        raise ValueError(f"unknown category: {category}")

    idx = {cid: i for i, cid in enumerate(concept_ids)}
    hits, total = 0, 0
    for task in task_ids:
        i = idx.get(f"task:{task}")
        if i is None:
            continue
        scores = (mind.assoc_U @ mind.assoc_V[i]).cpu().clone()
        # mask: только [lo, hi)
        mask = torch.ones(len(concept_ids), dtype=torch.bool)
        mask[lo:hi] = False
        scores[mask] = -float("inf")
        k = min(top_k, hi - lo)
        top = torch.topk(scores, k).indices.tolist()

        if category in ("time", "all") and task in time_dist:
            true_t = max(time_dist[task], key=time_dist[task].get)
            ti = idx.get(f"time:{true_t}")
            if ti is not None and ti < hi:
                hits += int(ti in top)
                total += 1
        if category == "loc" and task in loc_dist:
            true_l = max(loc_dist[task], key=loc_dist[task].get)
            li = idx.get(f"loc:{true_l}")
            if li is not None and li < hi:
                hits += int(li in top)
                total += 1
    return hits / max(total, 1)


@torch.no_grad()
def task_retrieval_score(mind, task_ids, similarity, *, k: int = 5) -> float:
    n = len(task_ids)
    hits, total = 0.0, 0
    for i in range(n):
        scores = (mind.assoc_U @ mind.assoc_V[i]).cpu()
        scores[i] = -float("inf")
        model_top = torch.topk(scores[:n], k).indices.tolist()
        true = similarity[i].clone()
        true[i] = -float("inf")
        true_top = torch.topk(true, k).indices.tolist()
        hits += len(set(model_top) & set(true_top)) / k
        total += 1
    return hits / max(total, 1)


@torch.no_grad()
def show_rank_diagnostics(
    mind, concept_ids, task_ids, time_dist, *, n_samples=10,
):
    """Для каждой задачи: где модель поставила истинное время (rank 1-10)."""
    n_tasks = len(task_ids)
    n_times = len(TIME_LABELS)
    idx = {cid: i for i, cid in enumerate(concept_ids)}
    print(f"\n  Диагностика времени (rank истинного time среди 10 времён):")
    ranks = []
    for task in task_ids[:n_samples]:
        i = idx[f"task:{task}"]
        scores = (mind.assoc_U @ mind.assoc_V[i]).cpu()
        time_scores = scores[n_tasks:n_tasks + n_times]
        true_t = max(time_dist[task], key=time_dist[task].get)
        true_i = idx[f"time:{true_t}"] - n_tasks
        rank = (time_scores > time_scores[true_i]).sum().item() + 1
        ranks.append(rank)
        top3 = torch.topk(time_scores, 3)
        pred3 = [TIME_LABELS[j] for j in top3.indices.tolist()]
        print(f"    {task[:36]:36s}  pred={pred3}  true={true_t}  rank={rank}")
    return sum(ranks) / max(len(ranks), 1)


# ============================================================
# 6. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="MS-LaTTE.json")
    parser.add_argument("--no-text", action="store_true")
    parser.add_argument("--top-tasks", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--lambda-task", type=float, default=0.3)
    parser.add_argument("--class-balance", type=float, default=0.5)
    parser.add_argument("--text-weight", type=float, default=0.3,
                        help="вес MiniLM в similarity (0..1)")
    parser.add_argument("--temp", type=float, default=0.1)
    args = parser.parse_args()

    print("=" * 70)
    print("  train_on_latte_v2.py — task co-occurrence + class-balanced")
    print("=" * 70)
    device = pick_device("auto")
    print(f"  device: {device}")
    print(f"  λ_task={args.lambda_task}  class_balance={args.class_balance}  "
          f"rank={args.rank}  epochs={args.epochs}  lr={args.lr}")

    print("\n[1] Загрузка...")
    t0 = time.perf_counter()
    tasks = load_latte(args.data)
    print(f"  {len(tasks)} задач  ({(time.perf_counter()-t0)*1000:.0f} ms)")

    print(f"\n[2] Аннотации (top-{args.top_tasks})...")
    task_ids, time_dist, loc_dist, counts = collect_annotations(
        tasks, top_n_tasks=args.top_tasks, min_known=1,
    )
    if not task_ids:
        print("  нет задач с аннотациями")
        return
    print(f"  задач: {len(task_ids)}  time: {len(time_dist)}  loc: {len(loc_dist)}")

    print("\n[3] Similarity...")
    t0 = time.perf_counter()
    w_time = 0.5
    w_loc = 0.1
    w_text = 0.0 if args.no_text else args.text_weight
    similarity, _, _, _ = build_task_similarity(
        task_ids, time_dist, loc_dist,
        w_time=w_time, w_loc=w_loc, w_text=w_text,
        device=device, use_text=not args.no_text,
    )
    print(f"  {(time.perf_counter()-t0)*1000:.0f} ms")

    print("\n[4] Targets...")
    concept_ids, targets = build_targets_v2(
        task_ids, time_dist, loc_dist, similarity, counts,
        temperature=args.temp, task_task_weight=args.lambda_task,
        device=torch.device("cpu"),
    )
    n = len(concept_ids)
    print(f"  концептов: {n}  non-zero cols: "
          f"{(targets.sum(0) > 0).sum().item()}/{n}")

    print("\n[5] MindEngine...")
    rank = min(args.rank, n)
    mind = MindEngine(
        [Concept(c) for c in concept_ids],
        associations="random", seed=42,
        mode="low_rank", rank=rank, device="auto",
    )
    print(f"  mode={mind.mode}  rank={mind.rank}  n={mind.n}")

    print("\n[6] Baseline (random assoc, честные 1/10 и 1/3):")
    for cat in ("time", "loc"):
        b1 = accuracy_by_category(mind, concept_ids, task_ids,
                                   time_dist, loc_dist, category=cat, top_k=1)
        b3 = accuracy_by_category(mind, concept_ids, task_ids,
                                   time_dist, loc_dist, category=cat, top_k=3)
        print(f"  {cat:4s}: acc@1={b1:.3f}  acc@3={b3:.3f}")

    print(f"\n[7] CE ({args.epochs} эпох, lr={args.lr}, "
          f"class_balance={args.class_balance})...")
    t0 = time.perf_counter()
    train_assoc_ce(
        mind, targets.to(mind.device),
        epochs=args.epochs, lr=args.lr,
        class_balance=args.class_balance,
    )
    print(f"  время: {time.perf_counter() - t0:.1f} s")
    scale = normalize_assoc(mind, 0.5)
    print(f"  normalize: × {scale:.4f}")

    print("\n[8] После обучения (честные метрики):")
    for cat in ("time", "loc"):
        ta1 = accuracy_by_category(mind, concept_ids, task_ids,
                                    time_dist, loc_dist, category=cat, top_k=1)
        ta3 = accuracy_by_category(mind, concept_ids, task_ids,
                                    time_dist, loc_dist, category=cat, top_k=3)
        print(f"  {cat:4s}: acc@1={ta1:.3f}  acc@3={ta3:.3f}")
    print(f"  task retrieval@5: {task_retrieval_score(mind, task_ids, similarity):.3f}")

    print("\n[9] Диагностика rank'а истинного времени:")
    show_rank_diagnostics(mind, concept_ids, task_ids, time_dist, n_samples=10)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()