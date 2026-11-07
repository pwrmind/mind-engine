"""train_on_latte.py — обучение ассоциаций на реальном MS-LaTTE.

Схема MS-LaTTE.json (проверена на реальном файле):
    entry: {ID, TaskTitle, ListTitle, LocJudgements, TimeJudgements}
    TimeJudgements: [{Known: "yes", Times: "WE-morning"}, ...,
                     {Known: "yes", Times: "WE-afternoon,WD-evening"}, ...]
    LocJudgements:  [{Known: "yes", Locations: "home", PublicLocations: ""}, ...]

Особенности:
    - Known: "yes" (lowercase)
    - Times: дефис, lowercase, может быть несколько через запятую
    - Locations: lowercase, может быть несколько через запятую
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam

from mind_engine import MindEngine, Concept, Thought, pick_device


# Ключи нормализованы в lowercase-через-дефис — ровно как в данных.
TIME_LABELS = [
    "wd-morning", "wd-afternoon", "wd-evening", "wd-night", "wd-anytime",
    "we-morning", "we-afternoon", "we-evening", "we-night", "we-anytime",
]
LOC_LABELS = ["home", "work", "public"]


# ============================================================
# 1. Устойчивый загрузчик
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
    """'WE-afternoon,WD-evening' → ['we-afternoon', 'wd-evening']."""
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
    """Judgements: list of dicts или dict of dicts."""
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
    print(f"    top-1 title: {top_tasks[0]!r}  (count={counts[top_tasks[0]]})")

    time_acc: dict[str, Counter] = defaultdict(Counter)
    loc_acc: dict[str, Counter] = defaultdict(Counter)

    for t in tasks:
        title = str(t.get("TaskTitle", "")).strip().lower()
        if title not in top_tasks:
            continue

        for j in _as_judgements(t.get("TimeJudgements")):
            if not _is_known(j):
                continue
            for label in _as_labels(j.get("Times")):
                if label in TIME_LABELS:
                    time_acc[title][label] += 1

        for j in _as_judgements(t.get("LocJudgements")):
            if not _is_known(j):
                continue
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
    return selected, time_dist, loc_dist


# ============================================================
# 2. Targets
# ============================================================

def build_targets(task_ids, time_dist, loc_dist, *, device):
    n_tasks, n_times, n_locs = len(task_ids), len(TIME_LABELS), len(LOC_LABELS)
    n = n_tasks + n_times + n_locs
    concept_ids = (
        [f"task:{t}" for t in task_ids]
        + [f"time:{t}" for t in TIME_LABELS]
        + [f"loc:{l}" for l in LOC_LABELS]
    )
    task_idx = {t: i for i, t in enumerate(task_ids)}
    time_idx = {t: n_tasks + i for i, t in enumerate(TIME_LABELS)}
    loc_idx = {l: n_tasks + n_times + i for i, l in enumerate(LOC_LABELS)}

    targets = torch.zeros(n, n, dtype=torch.float32)
    for task in task_ids:
        i = task_idx[task]
        for tl, p in time_dist.get(task, {}).items():
            targets[time_idx[tl], i] += p
            targets[i, time_idx[tl]] += p
        for ll, p in loc_dist.get(task, {}).items():
            targets[loc_idx[ll], i] += p
            targets[i, loc_idx[ll]] += p

    col_sums = targets.sum(dim=0, keepdim=True).clamp_min(1e-8)
    return concept_ids, (targets / col_sums).to(device)


# ============================================================
# 3. Обучение и оценка
# ============================================================

def train_assoc_ce(mind, targets, *, epochs=100, lr=0.02, weight_decay=1e-4, verbose=True):
    opt = Adam([mind.assoc_U, mind.assoc_V], lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        logits = mind.assoc_U @ mind.assoc_V.T
        log_probs = F.log_softmax(logits, dim=0)
        loss = -(targets * log_probs).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and (epoch == 0 or (epoch + 1) % 20 == 0):
            print(f"    epoch {epoch + 1:3d}  loss={loss.item():.4f}")


@torch.no_grad()
def normalize_assoc(mind, target_scale=0.5):
    A = mind.assoc_U @ mind.assoc_V.T
    m = A.abs().max().item()
    if m < 1e-8:
        return 1.0
    s = target_scale / m
    mind.assoc_U.data *= s
    return s


@torch.no_grad()
def context_accuracy(mind, concept_ids, task_ids, time_dist, loc_dist, top_k=3):
    idx = {cid: i for i, cid in enumerate(concept_ids)}
    th, tt, lh, lt = 0, 0, 0, 0
    for task in task_ids:
        i = idx.get(f"task:{task}")
        if i is None:
            continue
        scores = (mind.assoc_U @ mind.assoc_V[i]).cpu()
        scores[i] = -float("inf")
        top = torch.topk(scores, top_k).indices.tolist()
        if task in time_dist:
            ti = idx.get(f"time:{max(time_dist[task], key=time_dist[task].get)}")
            if ti is not None:
                th += int(ti in top); tt += 1
        if task in loc_dist:
            li = idx.get(f"loc:{max(loc_dist[task], key=loc_dist[task].get)}")
            if li is not None:
                lh += int(li in top); lt += 1
    return th / max(tt, 1), lh / max(lt, 1)


# ============================================================
# 4. Main
# ============================================================

def main():
    print("=" * 70)
    print("  train_on_latte.py — MS-LaTTE")
    print("=" * 70)
    print(f"  device: {pick_device('auto')}")

    print("\n[1] Загрузка...")
    t0 = time.perf_counter()
    tasks = load_latte("MS-LaTTE.json")
    print(f"  {len(tasks)} задач  ({(time.perf_counter()-t0)*1000:.0f} ms)")
    print(f"  поля: {list(tasks[0].keys())}")

    print("\n[2] Аннотации (top-150, min_known=1)...")
    t0 = time.perf_counter()
    task_ids, time_dist, loc_dist = collect_annotations(tasks, top_n_tasks=150, min_known=1)
    print(f"  задач: {len(task_ids)}  time: {len(time_dist)}  loc: {len(loc_dist)}  "
          f"({(time.perf_counter()-t0)*1000:.0f} ms)")
    if not task_ids:
        print("\n  ВСЁ ЕЩЁ 0 — пришлите вывод, буду копать глубже.")
        return

    print("\n  Примеры аннотаций:")
    for t in task_ids[:3]:
        print(f"    {t:25s}  time={dict(time_dist.get(t, {}))}  loc={dict(loc_dist.get(t, {}))}")

    print("\n[3] Targets...")
    concept_ids, targets = build_targets(task_ids, time_dist, loc_dist, device="cpu")
    n_nonzero = (targets.sum(0) > 0).sum().item()
    print(f"  концептов: {len(concept_ids)}  "
          f"({len(task_ids)} задач + {len(TIME_LABELS)} времён + {len(LOC_LABELS)} мест)")
    print(f"  non-zero columns: {n_nonzero}/{len(concept_ids)}")

    print("\n[4] MindEngine (low_rank)...")
    rank = min(64, max(16, len(concept_ids) // 2))
    mind = MindEngine(
        [Concept(c) for c in concept_ids],
        associations="random", seed=42,
        mode="low_rank", rank=rank, device="auto",
    )
    print(f"  mode={mind.mode}  rank={mind.rank}  n={mind.n}")

    print("\n[5] Baseline (random):")
    ta_b, la_b = context_accuracy(mind, concept_ids, task_ids, time_dist, loc_dist)
    print(f"  time acc@3={ta_b:.3f}   loc acc@3={la_b:.3f}")

    print("\n[6] CE-обучение (100 эпох, lr=0.02)...")
    t0 = time.perf_counter()
    train_assoc_ce(mind, targets.to(mind.device), epochs=100, lr=0.02)
    print(f"  время: {time.perf_counter() - t0:.1f} s")
    scale = normalize_assoc(mind, target_scale=0.5)
    print(f"  normalize: assoc × {scale:.4f}")

    print("\n[7] После обучения:")
    ta, la = context_accuracy(mind, concept_ids, task_ids, time_dist, loc_dist)
    print(f"  time acc@3={ta:.3f}  (+{(ta-ta_b)*100:.1f} pp)")
    print(f"  loc  acc@3={la:.3f}  (+{(la-la_b)*100:.1f} pp)")

    print("\n[8] Примеры предсказаний:")
    idx = {c: i for i, c in enumerate(concept_ids)}
    for task in task_ids[:8]:
        i = idx[f"task:{task}"]
        scores = (mind.assoc_U @ mind.assoc_V[i]).cpu()
        scores[i] = -float("inf")
        top = torch.topk(scores, 3).indices.tolist()
        predicted = [concept_ids[j].split(":", 1)[1] for j in top]
        true_t = max(time_dist[task], key=time_dist[task].get) if task in time_dist else "—"
        true_l = max(loc_dist[task], key=loc_dist[task].get) if task in loc_dist else "—"
        print(f"  {task[:32]:32s} → {predicted}")
        print(f"  {'':32s}   true: time={true_t}, loc={true_l}")

    print("\n" + "=" * 70)
    print("  готово.")
    print("=" * 70)


if __name__ == "__main__":
    main()