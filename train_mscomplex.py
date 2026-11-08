"""train_mscomplex.py — обучение MindEngine на MSComplexTasks.

Самостоятельный эксперимент: только MSComplexTasks, без MS-LaTTE.
Отвечает на вопрос "выучивает ли MindEngine логику декомпозиции".

Архитектура:
    1. Концепты = топ-K частотных названий подзадач.
    2. Сессии = топологическая сортировка DAG для задач с
       subtask_order_matters == "Yes".
    3. Warm-start: CE на co-occurrence подзадач внутри одной задачи
       (source → все остальные subtasks той же задачи).
    4. Fine-tune: CE на последовательных парах A → B из DAG.
    5. Оценка: session_hit@k на held-out сессиях.

Две метрики:
    seq_hit@k   — предсказать следующую по порядку подзадачу.
    sibling_hit@k — предсказать любую подзадачу из той же сложной задачи
                    (аналог ListTitle, но по реальной декомпозиции).

Запуск:
    uv run train_mscomplex.py
    uv run train_mscomplex.py --top-subtasks 300 --rank 128
    uv run train_mscomplex.py --min-subtask-freq 2
"""
from __future__ import annotations

import argparse
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
)


# ============================================================
# 1. Построение словаря концептов
# ============================================================

def build_vocabulary(entries, *, top_k: int = 500, min_freq: int = 2):
    """Считает частоту названий подзадач и берёт top-k с частотой >= min_freq."""
    freq: Counter = Counter()
    for entry in entries:
        for s in entry.get("subtasks") or []:
            name = str(s.get("name", "")).strip().lower()
            if name:
                freq[name] += 1

    candidates = [(name, c) for name, c in freq.items() if c >= min_freq]
    candidates.sort(key=lambda kv: -kv[1])
    selected = [name for name, _ in candidates[:top_k]]

    print(f"  словарь: {len(freq)} уникальных подзадач, "
          f"{len(candidates)} с freq >= {min_freq}, "
          f"взято {len(selected)}")
    print(f"  топ-5: {[(n, freq[n]) for n in selected[:5]]}")
    return selected, freq


# ============================================================
# 2. Сессии → пары, split
# ============================================================

def sessions_to_indices(sessions, vocab, *, verbose=True):
    """Сопоставляет названия подзадач из сессий с индексами в vocab."""
    name_to_idx = {name: i for i, name in enumerate(vocab)}
    result = []
    dropped = 0
    for s in sessions:
        idx = [name_to_idx[n.strip().lower()] for n in s
               if n.strip().lower() in name_to_idx]
        # убрать повторы подряд
        clean = []
        for i in idx:
            if not clean or clean[-1] != i:
                clean.append(i)
        if len(clean) >= 2:
            result.append(clean)
        else:
            dropped += 1
    if verbose:
        print(f"  сессий после матчинга: {len(result)} "
              f"(отброшено: {dropped})")
        if result:
            avg = sum(len(s) for s in result) / len(result)
            print(f"  средняя длина: {avg:.1f}")
    return result


def pairs_from_sessions(sessions):
    pairs = []
    for s in sessions:
        for i in range(len(s) - 1):
            pairs.append((s[i], s[i + 1]))
    return pairs


def sibling_pairs_from_sessions(sessions):
    """Все пары внутри одной сессии (source, target), source != target.
    Используется для sibling_hit@k — аналог ListTitle, но по декомпозиции."""
    pairs = []
    for s in sessions:
        for i, a in enumerate(s):
            for j, b in enumerate(s):
                if i != j:
                    pairs.append((a, b))
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
# 3. Targets
# ============================================================

def build_pair_targets(pairs, n, *, smoothing=1e-3):
    """targets[b, a] = P(b следует за a)."""
    counts = torch.zeros(n, n, dtype=torch.float32)
    for a, b in pairs:
        counts[b, a] += 1.0
    counts = counts + smoothing
    col_sums = counts.sum(dim=0, keepdim=True).clamp_min(1e-8)
    return counts / col_sums


def build_sibling_targets(sessions, n, *, smoothing=1e-3):
    """targets[b, a] = P(b в той же задаче, что и a)."""
    counts = torch.zeros(n, n, dtype=torch.float32)
    for s in sessions:
        for a in s:
            for b in s:
                if a != b:
                    counts[b, a] += 1.0
    counts = counts + smoothing
    col_sums = counts.sum(dim=0, keepdim=True).clamp_min(1e-8)
    return counts / col_sums


def build_warm_targets(sessions, n, *, sibling_weight=0.5, seq_weight=0.5):
    """Комбинация sibling + seq — warm-start."""
    sib = build_sibling_targets(sessions, n)
    pairs = pairs_from_sessions(sessions)
    seq = build_pair_targets(pairs, n)
    combined = sibling_weight * sib + seq_weight * seq
    col_sums = combined.sum(dim=0, keepdim=True).clamp_min(1e-8)
    return combined / col_sums


# ============================================================
# 4. Training
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
# 5. Evaluation
# ============================================================

@torch.no_grad()
def session_hit_at_k(mind, pairs, *, k=3):
    hits, total = 0, 0
    for a, b_true in pairs:
        scores = (mind.assoc_U @ mind.assoc_V[a]).cpu()
        scores[a] = -float("inf")
        kk = min(k, mind.n - 1)
        top = torch.topk(scores, kk).indices.tolist()
        hits += int(b_true in top)
        total += 1
    return hits / max(total, 1)


@torch.no_grad()
def snapshot(mind, p_test_seq, p_test_sib):
    return {
        "seq@1": session_hit_at_k(mind, p_test_seq, k=1) if p_test_seq else 0.0,
        "seq@3": session_hit_at_k(mind, p_test_seq, k=3) if p_test_seq else 0.0,
        "seq@5": session_hit_at_k(mind, p_test_seq, k=5) if p_test_seq else 0.0,
        "sib@5": session_hit_at_k(mind, p_test_sib, k=5) if p_test_sib else 0.0,
    }


# ============================================================
# 6. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="task_subtask_dataset.json")
    parser.add_argument("--top-subtasks", type=int, default=500)
    parser.add_argument("--min-subtask-freq", type=int, default=2)
    parser.add_argument("--max-session-len", type=int, default=10)
    parser.add_argument("--min-session-len", type=int, default=2)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--mode", choices=["low_rank", "dense"], default="low_rank")
    parser.add_argument("--warm-epochs", type=int, default=300)
    parser.add_argument("--warm-lr", type=float, default=0.01)
    parser.add_argument("--fine-epochs", type=int, default=200)
    parser.add_argument("--fine-lr", type=float, default=0.002)
    parser.add_argument("--fine-patience", type=int, default=30)
    parser.add_argument("--class-balance", type=float, default=0.5)
    parser.add_argument("--sibling-weight", type=float, default=0.5)
    parser.add_argument("--save-model", default="mind_mscomplex.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("  train_mscomplex.py — pure MSComplexTasks experiment")
    print("=" * 70)
    device = pick_device("auto")
    print(f"  device: {device}")

    print("\n[1] MSComplexTasks...")
    entries = load_mscomplex(args.data)
    print(f"  {len(entries)} сложных задач загружено")

    print("\n[2] Словарь концептов...")
    vocab, freq = build_vocabulary(
        entries, top_k=args.top_subtasks,
        min_freq=args.min_subtask_freq,
    )
    n = len(vocab)
    if n < 10:
        print("  слишком мало концептов — увеличьте --top-subtasks или уменьшите --min-subtask-freq")
        return

    print("\n[3] Сессии из DAG...")
    raw_sessions = build_sessions_from_mscomplex(
        entries,
        min_subtasks=args.min_session_len,
        max_subtasks=args.max_session_len,
        require_order=True,
    )
    sessions = sessions_to_indices(raw_sessions, vocab)
    if len(sessions) < 20:
        print("  слишком мало сессий — попробуйте --min-subtask-freq 1")
        return

    print("\n[4] Split...")
    s_train, s_val, s_test = split_sessions(sessions, seed=args.seed)
    p_train_seq = pairs_from_sessions(s_train)
    p_val_seq = pairs_from_sessions(s_val)
    p_test_seq = pairs_from_sessions(s_test)
    p_test_sib = sibling_pairs_from_sessions(s_test)
    print(f"  train sessions={len(s_train)}  pairs={len(p_train_seq)}")
    print(f"  val sessions={len(s_val)}    pairs={len(p_val_seq)}")
    print(f"  test sessions={len(s_test)}   pairs={len(p_test_seq)}  "
          f"sibling_pairs={len(p_test_sib)}")

    print("\n[5] MindEngine...")
    mind = MindEngine(
        [Concept(name) for name in vocab],
        associations="random", seed=args.seed,
        mode=args.mode, rank=min(args.rank, n),
        device="auto",
    )
    print(f"  mode={mind.mode}  rank={getattr(mind, 'rank', '-')}  n={mind.n}")

    print("\n[6] Baseline (random):")
    b = snapshot(mind, p_test_seq, p_test_sib)
    print(f"  seq hit@1={b['seq@1']:.3f}  @3={b['seq@3']:.3f}  @5={b['seq@5']:.3f}  "
          f"sib hit@5={b['sib@5']:.3f}")

    print(f"\n[7] Warm-start ({args.warm_epochs} эпох, sibling_weight={args.sibling_weight})...")
    warm_targets = build_warm_targets(
        s_train, n, sibling_weight=args.sibling_weight, seq_weight=1 - args.sibling_weight,
    )
    t0 = time.perf_counter()
    train_ce(mind, warm_targets.to(mind.device),
             epochs=args.warm_epochs, lr=args.warm_lr,
             class_balance=args.class_balance, verbose=True, tag="warm")
    print(f"  время: {time.perf_counter() - t0:.1f} s")
    normalize_assoc(mind)
    warm_m = snapshot(mind, p_test_seq, p_test_sib)
    print(f"\n[8] После warm-start:")
    print(f"  seq hit@1={warm_m['seq@1']:.3f}  @3={warm_m['seq@3']:.3f}  "
          f"@5={warm_m['seq@5']:.3f}  sib hit@5={warm_m['sib@5']:.3f}")

    print(f"\n[9] Fine-tune на последовательных парах ({args.fine_epochs} эпох, "
          f"lr={args.fine_lr})...")
    fine_targets = build_pair_targets(p_train_seq, n)

    def val_hit3():
        return session_hit_at_k(mind, p_val_seq, k=3) if p_val_seq else 0.0

    t0 = time.perf_counter()
    best_val = train_ce_early_stop(
        mind, fine_targets.to(mind.device),
        epochs=args.fine_epochs, lr=args.fine_lr,
        val_eval_fn=val_hit3, patience=args.fine_patience,
        class_balance=args.class_balance,
        verbose=True, tag="fine",
    )
    print(f"  время: {time.perf_counter() - t0:.1f} s  best val hit@3={best_val:.3f}")
    normalize_assoc(mind)
    fine_m = snapshot(mind, p_test_seq, p_test_sib)
    print(f"\n[10] После fine-tune:")
    print(f"  seq hit@1={fine_m['seq@1']:.3f}  @3={fine_m['seq@3']:.3f}  "
          f"@5={fine_m['seq@5']:.3f}  sib hit@5={fine_m['sib@5']:.3f}")

    print("\n[11] Итог:")
    print(f"  {'этап':12s} {'seq@1':>7s} {'seq@3':>7s} {'seq@5':>7s} {'sib@5':>7s}")
    for tag, m in [("baseline", b), ("warm", warm_m), ("fine", fine_m)]:
        print(f"  {tag:12s} {m['seq@1']:>7.3f} {m['seq@3']:>7.3f} "
              f"{m['seq@5']:>7.3f} {m['sib@5']:>7.3f}")

    print("\n[12] Примеры предсказаний (test):")
    for a, b_true in p_test_seq[:8]:
        scores = (mind.assoc_U @ mind.assoc_V[a]).cpu()
        scores[a] = -float("inf")
        top = torch.topk(scores, 3).indices.tolist()
        mark = "✓" if b_true in top else "✗"
        print(f"  {mark} {vocab[a][:28]:28s} → {vocab[b_true][:28]:28s}")
        print(f"        pred: {[vocab[j] for j in top]}")

    if args.save_model:
        mind.save(args.save_model)
        print(f"\n[13] модель сохранена в {args.save_model}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()