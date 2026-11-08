"""train_on_history.py — warm-start + fine-tune (v4, с MSComplexTasks).

Новые возможности:
    --session-source mscomplex — сессии из MSComplexTasks DAG.
        Топологическая сортировка даёт логически правильный порядок
        подзадач, а не просто соседство по ListTitle.

    --mscomplex-path — путь к task_subtask_dataset.json (default).

    --mscomplex-max-len — максимальная длина сессии (обрезка).

Запуск:
    uv run train_on_history.py --session-source mscomplex
    uv run train_on_history.py --session-source list_title
    uv run train_on_history.py --history-file mind_history.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam

from mind_engine import MindEngine, Concept, pick_device
from mscomplex_loader import (
    load_mscomplex,
    build_sessions_from_mscomplex,
    print_example_sessions,
)


TIME_LABELS = [
    "wd-morning", "wd-afternoon", "wd-evening", "wd-night", "wd-anytime",
    "we-morning", "we-afternoon", "we-evening", "we-night", "we-anytime",
]
LOC_LABELS = ["home", "work", "public"]


# ============================================================
# 1. MS-LaTTE loading
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
    time_acc = defaultdict(Counter)
    loc_acc = defaultdict(Counter)
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


def build_list_titles(tasks, task_set):
    list_acc = defaultdict(Counter)
    for t in tasks:
        title = str(t.get("TaskTitle", "")).strip().lower()
        if title not in task_set:
            continue
        lt = str(t.get("ListTitle", "")).strip().lower()
        if lt:
            list_acc[title][lt] += 1
    return {t: max(c, key=c.get) if c else None for t, c in list_acc.items()}


# ============================================================
# 2. Similarity
# ============================================================

def build_task_similarity(task_ids, time_dist, loc_dist, *,
                          w_time=0.5, w_loc=0.1, w_text=0.4,
                          device=torch.device("cpu"), use_text=True):
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
                "sentence-transformers/all-MiniLM-L6-v2", device=str(device),
            )
            embs = torch.tensor(
                model.encode(task_ids, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False),
                dtype=torch.float32, device=device,
            )
            S_text = (embs @ embs.T).cpu()
            print(f"  MiniLM: {embs.shape[0]}×{embs.shape[1]}")
        except Exception as e:
            print(f"  [warn] text недоступен ({e})")
            w_text = 0.0
            s = w_time + w_loc
            w_time, w_loc = w_time / s, w_loc / s
    S = w_time * S_time + w_loc * S_loc + w_text * S_text
    S.fill_diagonal_(0.0)
    return S.cpu()


# ============================================================
# 3. v2-targets (warm-start)
# ============================================================

def build_v2_targets(task_ids, time_dist, loc_dist, similarity, counts, *,
                     temperature=0.1, task_task_weight=0.3, freq_weight=0.2,
                     device=torch.device("cpu")):
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
# 4. Sessions
# ============================================================

def sessions_from_similarity(similarity, *, n_sessions, session_len,
                              top_k, temperature, seed):
    n = similarity.shape[0]
    k = min(top_k, n - 1)
    topk_vals, topk_idx = torch.topk(similarity, k, dim=1)
    topk_prob = torch.softmax(topk_vals / temperature, dim=1).tolist()
    topk_idx_l = topk_idx.tolist()
    rng = random.Random(seed)
    sessions = []
    for _ in range(n_sessions):
        start = rng.randrange(n)
        s = [start]
        for _step in range(session_len - 1):
            cur = s[-1]
            for _attempt in range(10):
                nxt = rng.choices(topk_idx_l[cur], weights=topk_prob[cur], k=1)[0]
                if nxt not in s:
                    break
            s.append(nxt)
        sessions.append(s)
    return sessions


def sessions_from_list_titles(task_list_title, task_ids, *,
                               n_sessions, session_len, seed):
    by_list = defaultdict(list)
    for i, t in enumerate(task_ids):
        lt = task_list_title.get(t)
        if lt:
            by_list[lt].append(i)
    eligible = [(lt, g) for lt, g in by_list.items() if len(g) >= 2]
    if not eligible:
        print("  [warn] нет ListTitle-групп с >=2 задачами")
        return []
    print(f"  ListTitle-групп (>=2): {len(eligible)}, "
          f"средний размер: {sum(len(g) for _, g in eligible)/len(eligible):.1f}")
    rng = random.Random(seed)
    sessions = []
    for _ in range(n_sessions):
        _, group = rng.choice(eligible)
        size = min(session_len, len(group))
        sessions.append(rng.sample(group, size))
    return sessions


def sessions_from_mscomplex_entries(
    entries: list[dict],
    task_ids: list[str],
    *,
    min_subtasks: int = 2,
    max_subtasks: int = 8,
    verbose: bool = True,
) -> list[list[int]]:
    """MSComplexTasks → сессии индексов task_ids.

    Подзадачи сопоставляются с task_ids по нормализованному названию
    (lowercase, strip). Если подзадача не найдена в task_ids — она
    пропускается, но если после фильтрации осталось >= min_subtasks,
    сессия сохраняется.
    """
    name_to_idx = {
        str(t).strip().lower(): i for i, t in enumerate(task_ids)
    }
    raw_sessions = build_sessions_from_mscomplex(
        entries,
        min_subtasks=min_subtasks,
        max_subtasks=max_subtasks,
        require_order=True,
        verbose=verbose,
    )
    sessions: list[list[int]] = []
    matched_subtasks = 0
    total_subtasks = 0
    for s in raw_sessions:
        total_subtasks += len(s)
        idx_session = []
        for name in s:
            key = str(name).strip().lower()
            if key in name_to_idx:
                idx = name_to_idx[key]
                if not idx_session or idx_session[-1] != idx:
                    idx_session.append(idx)
                matched_subtasks += 1
        if len(idx_session) >= min_subtasks:
            sessions.append(idx_session)

    if verbose:
        print(f"  MSComplexTasks → {len(sessions)} сессий после матчинга "
              f"({matched_subtasks}/{total_subtasks} подзадач найдено в task_ids)")
        if sessions:
            avg = sum(len(s) for s in sessions) / len(sessions)
            print(f"  средняя длина после матчинга: {avg:.1f}")
            print_example_sessions(
                [[task_ids[i] for i in s] for s in sessions], n=3,
            )
    return sessions


def pairs_from_sessions(sessions):
    pairs = []
    for s in sessions:
        for i in range(len(s) - 1):
            if s[i] != s[i + 1]:
                pairs.append((s[i], s[i + 1]))
    return pairs


def pairs_from_history_json(path, task_ids):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    history = data.get("history", [])
    if not history:
        raise ValueError(f"{path}: нет 'history'")
    idx = {tid: i for i, tid in enumerate(task_ids)}
    pairs = []
    prev = None
    for entry in history:
        thought = entry.get("thought")
        if not thought:
            continue
        sig = thought.get("signature") or {}
        if not sig:
            prev = None
            continue
        top = max(sig.items(), key=lambda kv: abs(kv[1]))[0]
        bare = top.removeprefix("task:")
        if bare in idx:
            cur = idx[bare]
            if prev is not None and prev != cur:
                pairs.append((prev, cur))
            prev = cur
        else:
            prev = None
    return pairs


def split_sessions(sessions, *, train_frac=0.8, val_frac=0.1, seed=42):
    rng = random.Random(seed)
    s = list(sessions)
    rng.shuffle(s)
    n = len(s)
    n_tr = int(n * train_frac)
    n_va = int(n * val_frac)
    return s[:n_tr], s[n_tr:n_tr + n_va], s[n_tr + n_va:]


# ============================================================
# 5. Session targets
# ============================================================

def build_session_targets(pairs, n_tasks, *, smoothing=1e-3):
    counts = torch.zeros(n_tasks, n_tasks, dtype=torch.float32)
    for a, b in pairs:
        counts[b, a] += 1.0
    counts = counts + smoothing
    col_sums = counts.sum(dim=0, keepdim=True).clamp_min(1e-8)
    return counts / col_sums


# ============================================================
# 6. Fine-targets: хирургический blend
# ============================================================

def build_fine_targets(v2_targets, session_targets, *, n_tasks, fine_lambda,
                       device=torch.device("cpu")):
    fine = v2_targets.clone()
    v2_block = v2_targets[:n_tasks, :n_tasks].clone()
    col_mass = v2_block.sum(dim=0, keepdim=True).clamp_min(1e-3)
    session_scaled = session_targets * col_mass
    blended = (1 - fine_lambda) * v2_block + fine_lambda * session_scaled
    fine[:n_tasks, :n_tasks] = blended
    return fine.to(device)


# ============================================================
# 7. Training
# ============================================================

def train_ce(mind, targets, *, epochs, lr, weight_decay=1e-4,
             class_balance=0.0, verbose=True, tag=""):
    opt = Adam([mind.assoc_U, mind.assoc_V], lr=lr, weight_decay=weight_decay)
    w = None
    if class_balance > 0:
        freq = targets.sum(dim=1).clamp_min(1e-8)
        w = 1.0 / freq.pow(class_balance)
        w = (w / w.mean()).to(targets.device)
    for epoch in range(epochs):
        logits = mind.assoc_U @ mind.assoc_V.T
        log_probs = F.log_softmax(logits, dim=0)
        loss = -(targets * log_probs)
        if w is not None:
            loss = loss * w.unsqueeze(1)
        loss = loss.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and (epoch == 0 or (epoch + 1) % max(1, epochs // 8) == 0):
            print(f"    {tag} epoch {epoch + 1:4d}  loss={loss.item():.4f}")


def train_ce_early_stop(mind, targets, *, epochs, lr, val_eval_fn,
                        patience=30, weight_decay=1e-4, class_balance=0.0,
                        verbose=True, tag=""):
    opt = Adam([mind.assoc_U, mind.assoc_V], lr=lr, weight_decay=weight_decay)
    w = None
    if class_balance > 0:
        freq = targets.sum(dim=1).clamp_min(1e-8)
        w = 1.0 / freq.pow(class_balance)
        w = (w / w.mean()).to(targets.device)

    best_score = -1.0
    best_state = (mind.assoc_U.detach().clone(), mind.assoc_V.detach().clone())
    bad_epochs = 0

    for epoch in range(epochs):
        logits = mind.assoc_U @ mind.assoc_V.T
        log_probs = F.log_softmax(logits, dim=0)
        loss = -(targets * log_probs)
        if w is not None:
            loss = loss * w.unsqueeze(1)
        loss = loss.sum()
        opt.zero_grad(); loss.backward(); opt.step()

        if (epoch + 1) % 5 == 0:
            score = val_eval_fn()
            if score > best_score + 1e-4:
                best_score = score
                best_state = (mind.assoc_U.detach().clone(),
                              mind.assoc_V.detach().clone())
                bad_epochs = 0
            else:
                bad_epochs += 5
                if bad_epochs >= patience:
                    if verbose:
                        print(f"    {tag} early stop at epoch {epoch+1} "
                              f"(best={best_score:.3f})")
                    break

        if verbose and (epoch == 0 or (epoch + 1) % max(1, epochs // 8) == 0):
            print(f"    {tag} epoch {epoch + 1:4d}  loss={loss.item():.4f}  "
                  f"best_val={best_score:.3f}")

    mind.assoc_U.data = best_state[0]
    mind.assoc_V.data = best_state[1]
    return best_score


@torch.no_grad()
def normalize_assoc(mind, target_scale=0.5):
    A = mind.assoc_U @ mind.assoc_V.T
    m = A.abs().max().item()
    if m < 1e-8:
        return 1.0
    s = target_scale / m
    mind.assoc_U.data *= s
    return s


# ============================================================
# 8. Evaluation
# ============================================================

@torch.no_grad()
def session_hit_at_k(mind, pairs, n_tasks, *, k=3):
    hits, total = 0, 0
    for a, b_true in pairs:
        scores = (mind.assoc_U @ mind.assoc_V[a]).cpu()
        scores[a] = -float("inf")
        scores[n_tasks:] = -float("inf")
        kk = min(k, n_tasks - 1)
        top = torch.topk(scores, kk).indices.tolist()
        hits += int(b_true in top)
        total += 1
    return hits / max(total, 1)


@torch.no_grad()
def context_accuracy(mind, concept_ids, task_ids, time_dist, loc_dist,
                     *, category, top_k):
    n_tasks = len(task_ids)
    n_times, n_locs = len(TIME_LABELS), len(LOC_LABELS)
    if category == "time":
        lo, hi = n_tasks, n_tasks + n_times
    elif category == "loc":
        lo, hi = n_tasks + n_times, n_tasks + n_times + n_locs
    else:
        lo, hi = n_tasks, n_tasks + n_times + n_locs
    idx = {cid: i for i, cid in enumerate(concept_ids)}
    hits, total = 0, 0
    for task in task_ids:
        i = idx.get(f"task:{task}")
        if i is None:
            continue
        scores = (mind.assoc_U @ mind.assoc_V[i]).cpu().clone()
        mask = torch.ones(len(concept_ids), dtype=torch.bool)
        mask[lo:hi] = False
        scores[mask] = -float("inf")
        top = torch.topk(scores, min(top_k, hi - lo)).indices.tolist()
        if category in ("time", "all") and task in time_dist:
            true_t = max(time_dist[task], key=time_dist[task].get)
            ti = idx.get(f"time:{true_t}")
            if ti is not None and lo <= ti < hi:
                hits += int(ti in top); total += 1
        if category == "loc" and task in loc_dist:
            true_l = max(loc_dist[task], key=loc_dist[task].get)
            li = idx.get(f"loc:{true_l}")
            if li is not None and lo <= li < hi:
                hits += int(li in top); total += 1
    return hits / max(total, 1)


@torch.no_grad()
def task_retrieval_score(mind, task_ids, similarity, *, k=5):
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
def snapshot_metrics(mind, concept_ids, task_ids, time_dist, loc_dist,
                     p_test, similarity, n_tasks):
    return {
        "hit@1": session_hit_at_k(mind, p_test, n_tasks, k=1) if p_test else 0.0,
        "hit@3": session_hit_at_k(mind, p_test, n_tasks, k=3) if p_test else 0.0,
        "hit@5": session_hit_at_k(mind, p_test, n_tasks, k=5) if p_test else 0.0,
        "time@1": context_accuracy(mind, concept_ids, task_ids, time_dist, loc_dist,
                                    category="time", top_k=1),
        "loc@1": context_accuracy(mind, concept_ids, task_ids, time_dist, loc_dist,
                                   category="loc", top_k=1),
        "task_ret": task_retrieval_score(mind, task_ids, similarity, k=5),
    }


# ============================================================
# 9. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="MS-LaTTE.json")
    parser.add_argument("--no-text", action="store_true")
    parser.add_argument("--top-tasks", type=int, default=150)
    parser.add_argument("--rank", type=int, default=128)

    parser.add_argument("--warm-epochs", type=int, default=300)
    parser.add_argument("--warm-lr", type=float, default=0.01)
    parser.add_argument("--lambda-task", type=float, default=0.3)

    parser.add_argument("--fine-epochs", type=int, default=200)
    parser.add_argument("--fine-lr", type=float, default=0.002)
    parser.add_argument("--fine-lambda", type=float, default=0.3)
    parser.add_argument("--fine-patience", type=int, default=30)
    parser.add_argument("--class-balance", type=float, default=0.5)

    parser.add_argument("--session-source",
                        choices=["similarity", "list_title", "mscomplex", "history"],
                        default="mscomplex",
                        help="mscomplex — реальный DAG из Wunderlist")
    parser.add_argument("--generate-sessions", type=int, default=3000)
    parser.add_argument("--session-len", type=int, default=3)
    parser.add_argument("--session-top-k", type=int, default=8)
    parser.add_argument("--session-temp", type=float, default=0.5)
    parser.add_argument("--history-file", default=None)
    parser.add_argument("--mscomplex-path", default="task_subtask_dataset.json")
    parser.add_argument("--mscomplex-min-len", type=int, default=2)
    parser.add_argument("--mscomplex-max-len", type=int, default=8)

    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--no-fine-tune", action="store_true")
    parser.add_argument("--save-model", default="mind_latte.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("  train_on_history.py — MSComplexTasks integration")
    print("=" * 70)
    device = pick_device("auto")
    print(f"  device: {device}")
    print(f"  session_source={args.session_source}  "
          f"fine_lambda={args.fine_lambda}")

    print("\n[1] MS-LaTTE...")
    tasks = load_latte(args.data)
    print(f"  {len(tasks)} задач")

    print(f"\n[2] Аннотации (top-{args.top_tasks})...")
    task_ids, time_dist, loc_dist, counts = collect_annotations(
        tasks, top_n_tasks=args.top_tasks, min_known=1,
    )
    task_list_title = build_list_titles(tasks, set(task_ids))
    print(f"  задач: {len(task_ids)}  "
          f"с ListTitle: {sum(1 for v in task_list_title.values() if v)}")

    print("\n[3] Similarity...")
    similarity = build_task_similarity(
        task_ids, time_dist, loc_dist, w_time=0.5, w_loc=0.1, w_text=0.3,
        device=device, use_text=not args.no_text,
    )

    print("\n[4] v2-targets (warm)...")
    concept_ids, v2_targets = build_v2_targets(
        task_ids, time_dist, loc_dist, similarity, counts,
        temperature=0.1, task_task_weight=args.lambda_task,
    )
    n = len(concept_ids)
    n_tasks = len(task_ids)
    print(f"  концептов: {n}")

    print("\n[5] MindEngine...")
    mind = MindEngine(
        [Concept(c) for c in concept_ids],
        associations="random", seed=args.seed,
        mode="low_rank", rank=min(args.rank, n), device="auto",
    )
    print(f"  mode={mind.mode}  rank={mind.rank}")

    print(f"\n[6] Сессии (source={args.session_source})...")
    if args.history_file:
        print(f"  из {args.history_file}")
        pairs_all = pairs_from_history_json(args.history_file, task_ids)
        sessions = [[a, b] for a, b in pairs_all]
    elif args.session_source == "mscomplex":
        print(f"  MSComplexTasks: {args.mscomplex_path}")
        entries = load_mscomplex(args.mscomplex_path)
        sessions = sessions_from_mscomplex_entries(
            entries, task_ids,
            min_subtasks=args.mscomplex_min_len,
            max_subtasks=args.mscomplex_max_len,
        )
    elif args.session_source == "list_title":
        print(f"  ListTitle-based ({args.generate_sessions})")
        sessions = sessions_from_list_titles(
            task_list_title, task_ids,
            n_sessions=args.generate_sessions,
            session_len=args.session_len, seed=args.seed,
        )
    else:
        print(f"  similarity walk ({args.generate_sessions}, "
              f"len={args.session_len})  [циркулярно — sanity check]")
        sessions = sessions_from_similarity(
            similarity,
            n_sessions=args.generate_sessions,
            session_len=args.session_len,
            top_k=args.session_top_k,
            temperature=args.session_temp, seed=args.seed,
        )
    if not sessions:
        print("  нет сессий — fine-tune пропущен")
        sessions_train, sessions_val, sessions_test = [], [], []
    else:
        sessions_train, sessions_val, sessions_test = split_sessions(
            sessions, seed=args.seed,
        )
    p_train = pairs_from_sessions(sessions_train)
    p_val = pairs_from_sessions(sessions_val)
    p_test = pairs_from_sessions(sessions_test)
    print(f"  split: train={len(p_train)}  val={len(p_val)}  test={len(p_test)}")

    print("\n[7] Baseline (random):")
    b = snapshot_metrics(mind, concept_ids, task_ids, time_dist, loc_dist,
                          p_test, similarity, n_tasks)
    print(f"  session hit@5={b['hit@5']:.3f}  time@1={b['time@1']:.3f}  "
          f"loc@1={b['loc@1']:.3f}  task_ret={b['task_ret']:.3f}")

    warm_metrics = b
    if not args.no_warm_start:
        print(f"\n[8] Warm-start ({args.warm_epochs} эпох, lr={args.warm_lr})...")
        t0 = time.perf_counter()
        train_ce(mind, v2_targets.to(mind.device),
                 epochs=args.warm_epochs, lr=args.warm_lr,
                 class_balance=0.5, verbose=True, tag="warm")
        print(f"  время: {time.perf_counter() - t0:.1f} s")
        normalize_assoc(mind, 0.5)
        warm_metrics = snapshot_metrics(mind, concept_ids, task_ids,
                                          time_dist, loc_dist, p_test,
                                          similarity, n_tasks)
        print(f"\n[9] После warm-start:")
        print(f"  session hit@5={warm_metrics['hit@5']:.3f}  "
              f"time@1={warm_metrics['time@1']:.3f}  "
              f"loc@1={warm_metrics['loc@1']:.3f}  "
              f"task_ret={warm_metrics['task_ret']:.3f}")

    fine_metrics = warm_metrics
    if not args.no_fine_tune and p_train:
        print(f"\n[10] Fine-tune ({args.fine_epochs} эпох, lr={args.fine_lr}, "
              f"λ_fine={args.fine_lambda})...")
        session_targets = build_session_targets(p_train, n_tasks, smoothing=1e-3)
        fine_targets = build_fine_targets(
            v2_targets, session_targets,
            n_tasks=n_tasks, fine_lambda=args.fine_lambda,
        )

        def val_hit3():
            return session_hit_at_k(mind, p_val, n_tasks, k=3) if p_val else 0.0

        t0 = time.perf_counter()
        best_val = train_ce_early_stop(
            mind, fine_targets.to(mind.device),
            epochs=args.fine_epochs, lr=args.fine_lr,
            val_eval_fn=val_hit3, patience=args.fine_patience,
            class_balance=args.class_balance,
            verbose=True, tag="fine",
        )
        print(f"  время: {time.perf_counter() - t0:.1f} s  "
              f"best val hit@3={best_val:.3f}")
        normalize_assoc(mind, 0.5)

        fine_metrics = snapshot_metrics(mind, concept_ids, task_ids,
                                         time_dist, loc_dist, p_test,
                                         similarity, n_tasks)
        print(f"\n[11] После fine-tune:")
        print(f"  session hit@5={fine_metrics['hit@5']:.3f}  "
              f"time@1={fine_metrics['time@1']:.3f}  "
              f"loc@1={fine_metrics['loc@1']:.3f}  "
              f"task_ret={fine_metrics['task_ret']:.3f}")

    print("\n[12] Итог:")
    print(f"  {'этап':20s} {'hit@5':>7s} {'time@1':>7s} {'loc@1':>7s} {'task_ret':>9s}")
    print(f"  {'baseline':20s} {b['hit@5']:>7.3f} {b['time@1']:>7.3f} "
          f"{b['loc@1']:>7.3f} {b['task_ret']:>9.3f}")
    if not args.no_warm_start:
        print(f"  {'warm-start':20s} {warm_metrics['hit@5']:>7.3f} "
              f"{warm_metrics['time@1']:>7.3f} "
              f"{warm_metrics['loc@1']:>7.3f} "
              f"{warm_metrics['task_ret']:>9.3f}")
    if not args.no_fine_tune and p_train:
        print(f"  {'fine-tune':20s} {fine_metrics['hit@5']:>7.3f} "
              f"{fine_metrics['time@1']:>7.3f} "
              f"{fine_metrics['loc@1']:>7.3f} "
              f"{fine_metrics['task_ret']:>9.3f}")
        print(f"  Δ fine vs warm:    "
              f"{fine_metrics['hit@5']-warm_metrics['hit@5']:+.3f}   "
              f"{fine_metrics['time@1']-warm_metrics['time@1']:+.3f}   "
              f"{fine_metrics['loc@1']-warm_metrics['loc@1']:+.3f}   "
              f"{fine_metrics['task_ret']-warm_metrics['task_ret']:+.3f}")

    if args.save_model:
        mind.save(args.save_model)
        print(f"\n[13] модель сохранена в {args.save_model}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()