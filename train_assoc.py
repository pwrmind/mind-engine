"""train_assoc.py — обучение матрицы ассоциаций.

Два метода:
    BPR  — парный loss «pos > neg». Быстро, но не различает силу
           внутри позитивов и насыщает логиты.
    CE   — soft cross-entropy против эмпирического распределения
           P(b | a) из train-пар. Сохраняет порядок сильных связей
           и не взрывает логиты.

После обучения — нормализация assoc, чтобы state оставался
в рабочем диапазоне движка (|assoc| ~ 0.5, а не ~12).

Запуск:
    uv run train_assoc.py
"""
from __future__ import annotations

import random
import time
from collections import defaultdict
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.optim import Adam

from mind_engine import MindEngine, Concept, Thought, pick_device


# ============================================================
# 1. Истинный граф (ground truth)
# ============================================================

def build_semantic_graph() -> dict[str, dict[str, float]]:
    return {
        "coffee":  {"morning": 0.8, "work": 0.3},
        "morning": {"work": 0.6, "coffee": 0.4, "walk": 0.2},
        "work":    {"focus": 0.7, "stress": 0.5, "fatigue": 0.3},
        "focus":   {"work": 0.6, "calm": 0.2},
        "stress":  {"rest": 0.7, "fatigue": 0.5, "sadness": 0.3},
        "rest":    {"calm": 0.6, "evening": 0.3},
        "calm":    {"walk": 0.5, "evening": 0.4},
        "walk":    {"joy": 0.6, "calm": 0.3},
        "joy":     {"friend": 0.7, "calm": 0.4},
        "friend":  {"joy": 0.5, "calm": 0.4, "rest": 0.2},
        "fatigue": {"rest": 0.8, "evening": 0.4},
        "hunger":  {"food": 0.9},
        "food":    {"joy": 0.3, "rest": 0.3},
        "evening": {"rest": 0.5, "calm": 0.3},
    }


# ============================================================
# 2. Генерация последовательностей
# ============================================================

def random_walk(graph, n_steps, start, rng):
    walk = [start]
    current = start
    for _ in range(n_steps - 1):
        nbrs = graph.get(current)
        if not nbrs:
            current = rng.choice(list(graph.keys()))
            walk.append(current)
            continue
        keys, weights = list(nbrs.keys()), list(nbrs.values())
        r = rng.random() * sum(weights)
        acc = 0.0
        for k, w in zip(keys, weights):
            acc += w
            if r <= acc:
                current = k
                break
        walk.append(current)
    return walk


def generate_sequences(graph, *, n_seqs=500, seq_len=6, seed=42):
    rng = random.Random(seed)
    starts = list(graph.keys())
    return [random_walk(graph, seq_len, rng.choice(starts), rng) for _ in range(n_seqs)]


def extract_pairs(sequences: Sequence[Sequence[str]], *, window: int = 1):
    pairs = []
    for seq in sequences:
        T = len(seq)
        for i in range(T):
            for j in range(i + 1, min(i + 1 + window, T)):
                if seq[i] != seq[j]:
                    pairs.append((seq[i], seq[j]))
    return pairs


# ============================================================
# 3. Данные для двух loss'ов
# ============================================================

def build_triplets(pairs, all_concepts, *, n_neg=4, seed=42):
    rng = random.Random(seed)
    pos_by_a = defaultdict(set)
    for a, b in pairs:
        pos_by_a[a].add(b)
    all_c = list(all_concepts)
    out = []
    for a, b in pairs:
        cand = [c for c in all_c if c != a and c not in pos_by_a[a]]
        if not cand:
            continue
        for _ in range(n_neg):
            out.append((a, b, rng.choice(cand)))
    return out


def build_empirical_targets(
    pairs: Sequence[tuple[str, str]],
    all_concepts: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    """targets[b, a] = P(b следует за a) по train-парам."""
    idx = {c: i for i, c in enumerate(all_concepts)}
    n = len(all_concepts)
    counts = torch.zeros(n, n)
    for a, b in pairs:
        counts[idx[b], idx[a]] += 1.0
    col_sums = counts.sum(dim=0, keepdim=True).clamp_min(1.0)
    return (counts / col_sums).to(device)


# ============================================================
# 4. Обучение: BPR (baseline) и CE (recommended)
# ============================================================

def train_assoc_bpr(mind, triplets, *, epochs=30, lr=0.05, batch_size=512,
                    weight_decay=1e-4, verbose=True):
    """BPR с weight decay. Всё ещё насыщает, но медленнее."""
    device = mind.device
    a_idx = torch.tensor([mind._idx[a] for a, _, _ in triplets], device=device)
    bp_idx = torch.tensor([mind._idx[b] for _, b, _ in triplets], device=device)
    bn_idx = torch.tensor([mind._idx[n] for _, _, n in triplets], device=device)
    N = len(triplets)

    opt = Adam([mind.assoc_U, mind.assoc_V], lr=lr, weight_decay=weight_decay)
    hist = {"loss": [], "acc": []}

    for epoch in range(epochs):
        perm = torch.randperm(N, device=device)
        tot, correct = 0.0, 0
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            a_b, bp_b, bn_b = a_idx[idx], bp_idx[idx], bn_idx[idx]
            s_pos = (mind.assoc_U[bp_b] * mind.assoc_V[a_b]).sum(dim=-1)
            s_neg = (mind.assoc_U[bn_b] * mind.assoc_V[a_b]).sum(dim=-1)
            diff = s_pos - s_neg
            loss = -F.logsigmoid(diff).mean()

            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
            correct += (diff > 0).sum().item()

        hist["loss"].append(tot / N)
        hist["acc"].append(correct / N)
        if verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
            print(f"    epoch {epoch + 1:3d}  loss={hist['loss'][-1]:.4f}  "
                  f"acc={hist['acc'][-1]:.3f}")
    return hist


def train_assoc_ce(mind, targets, *, epochs=100, lr=0.02, weight_decay=1e-4,
                   verbose=True):
    """Soft cross-entropy: для каждого a распределение по b,
    предсказанное softmax'ом assoc[b, a], приближается к
    эмпирическому P(b | a). Естественно сохраняет порядок сил.
    """
    opt = Adam([mind.assoc_U, mind.assoc_V], lr=lr, weight_decay=weight_decay)
    hist = {"loss": []}

    for epoch in range(epochs):
        logits = mind.assoc_U @ mind.assoc_V.T          # [n, n], logits[b, a]
        log_probs = F.log_softmax(logits, dim=0)         # softmax over b для каждого a
        loss = -(targets * log_probs).sum()

        opt.zero_grad(); loss.backward(); opt.step()
        hist["loss"].append(loss.item())
        if verbose and (epoch == 0 or (epoch + 1) % 20 == 0):
            print(f"    epoch {epoch + 1:3d}  loss={hist['loss'][-1]:.4f}")
    return hist


# ============================================================
# 5. Нормализация assoc под рабочий диапазон движка
# ============================================================

@torch.no_grad()
def normalize_assoc(mind: MindEngine, target_scale: float = 0.5) -> float:
    """Масштабирует assoc_U так, чтобы max|assoc| = target_scale.
    Направления сохраняются — hit@k не меняется, но state в _apply
    остаётся в [−target_scale, +target_scale] вместо [−15, +15].
    """
    A = mind.assoc_U @ mind.assoc_V.T
    max_abs = A.abs().max().item()
    if max_abs < 1e-8:
        return 1.0
    scale = target_scale / max_abs
    mind.assoc_U.data *= scale
    return scale


# ============================================================
# 6. Оценка
# ============================================================

@torch.no_grad()
def eval_hit_at_k(mind, sequences, *, k=3):
    hits, total = 0, 0
    for seq in sequences:
        for i in range(len(seq) - 1):
            a, b_next = seq[i], seq[i + 1]
            if a not in mind._idx or b_next not in mind._idx:
                continue
            a_i = mind._idx[a]
            scores = mind.assoc_U @ mind.assoc_V[a_i]
            scores[a_i] = -float("inf")
            top_k = torch.topk(scores, k).indices.cpu().tolist()
            if mind._idx[b_next] in top_k:
                hits += 1
            total += 1
    return hits / max(total, 1)


@torch.no_grad()
def compare_with_truth(mind, graph, *, top_n=3):
    print(f"\n  {'Concept':10s}  {'Learned top-' + str(top_n):40s}  {'True top-' + str(top_n):24s}  match")
    print(f"  {'-'*10}  {'-'*40}  {'-'*24}  -----")
    matched, checked = 0, 0
    per_source = []
    for c in sorted(graph.keys()):
        if c not in mind._idx:
            continue
        a_i = mind._idx[c]
        scores = mind.assoc_U @ mind.assoc_V[a_i]
        scores[a_i] = -float("inf")
        top = torch.topk(scores, top_n).indices.cpu().tolist()
        learned = [mind.ids[i] for i in top if i != a_i]

        true_sorted = sorted(graph[c].items(), key=lambda kv: -kv[1])
        true_names = [name for name, _ in true_sorted[:top_n]]

        learned_str = ", ".join(
            f"{name}({float(scores[mind._idx[name]]):+.2f})" for name in learned
        )
        true_str = ", ".join(true_names)
        m = sum(1 for name in learned if name in true_names)
        # hit@N считаем только по позициям, где порядок важен: совпал ли
        # именно top-1, и т.д. Здесь — просто доля попаданий в множество.
        matched += m
        checked += min(top_n, len(true_names))
        per_source.append(m / max(min(top_n, len(true_names)), 1))
        print(f"  {c:10s}  {learned_str:40s}  {true_str:24s}  {m}/{min(top_n, len(true_names))}")

    avg = sum(per_source) / max(len(per_source), 1)
    print(f"\n  средний hit@{top_n}: {avg:.3f}  ({matched}/{checked})")
    return avg, matched, checked


@torch.no_grad()
def ordering_accuracy(mind, graph) -> float:
    """Доля источников, у которых топ-1 выученного = топ-1 истинного.
    Именно это BPR не умеет, а CE должен уметь."""
    correct, total = 0, 0
    for c, succs in graph.items():
        if c not in mind._idx:
            continue
        a_i = mind._idx[c]
        scores = mind.assoc_U @ mind.assoc_V[a_i]
        scores[a_i] = -float("inf")
        top1 = mind.ids[torch.argmax(scores).item()]
        true_top1 = max(succs.items(), key=lambda kv: kv[1])[0]
        if top1 == true_top1:
            correct += 1
        total += 1
    return correct / max(total, 1)


# ============================================================
# 7. Демо
# ============================================================

def _fmt_focus(focus, n=4):
    items = focus["dominant"][:n]
    return ", ".join(f"{it['concept']}={it['activation']:+.3f}" for it in items) or "(пусто)"


def run_method(name, build_engine, train_fn, train_args, target_scale, test_seqs, graph):
    print(f"\n  ── {name} ──")
    mind = build_engine()
    train_fn(mind, **train_args)
    scale = normalize_assoc(mind, target_scale=target_scale)
    print(f"    normalize: assoc × {scale:.4f}  (max|assoc| → {target_scale})")

    hit = {k: eval_hit_at_k(mind, test_seqs, k=k) for k in (1, 3, 5)}
    print(f"    hit@1={hit[1]:.3f}  hit@3={hit[3]:.3f}  hit@5={hit[5]:.3f}")
    print(f"    ordering (top1 == true top1): {ordering_accuracy(mind, graph):.3f}")

    mind.reset()
    mind.think(Thought.make("кофе", {"coffee": 1.0}))
    print(f"    после think('coffee'): фокус {_fmt_focus(mind.current_focus(), 3)}")
    return mind, hit


def main():
    print("=" * 70)
    print("  train_assoc.py — BPR vs CE, с нормализацией")
    print("=" * 70)
    print(f"  device: {pick_device('auto')}")

    graph = build_semantic_graph()
    concept_ids = sorted({c for c in graph.keys()} | {t for s in graph.values() for t in s})
    print(f"  концептов: {len(concept_ids)}  источников: {len(graph)}")

    print("\n[1] Данные...")
    train_seqs = generate_sequences(graph, n_seqs=500, seq_len=6, seed=42)
    test_seqs = generate_sequences(graph, n_seqs=100, seq_len=6, seed=1234)
    train_pairs = extract_pairs(train_seqs, window=1)
    triplets = build_triplets(train_pairs, concept_ids, n_neg=4, seed=42)
    print(f"  пар: {len(train_pairs)}  триплетов: {len(triplets)}")

    def build_engine():
        return MindEngine([Concept(c) for c in concept_ids],
                          associations="random", seed=42,
                          mode="low_rank", rank=16, device="auto")

    # baseline
    print("\n[2] Baseline (random assoc):")
    base = build_engine()
    hit_b = {k: eval_hit_at_k(base, test_seqs, k=k) for k in (1, 3, 5)}
    print(f"  hit@1={hit_b[1]:.3f}  hit@3={hit_b[3]:.3f}  hit@5={hit_b[5]:.3f}")
    print(f"  ordering: {ordering_accuracy(base, graph):.3f}")

    # BPR
    mind_bpr, hit_bpr = run_method(
        "BPR (30 эпох, wd=1e-4)",
        build_engine,
        train_assoc_bpr,
        dict(triplets=triplets, epochs=30, lr=0.05, weight_decay=1e-4),
        target_scale=0.5,
        test_seqs=test_seqs, graph=graph,
    )

    # CE
    targets = build_empirical_targets(train_pairs, concept_ids, torch.device("cpu"))
    mind_ce, hit_ce = run_method(
        "CE (100 эпох, wd=1e-4)",
        build_engine,
        lambda m, **kw: train_assoc_ce(
            m, targets.to(m.device), **kw),
        dict(epochs=100, lr=0.02, weight_decay=1e-4),
        target_scale=0.5,
        test_seqs=test_seqs, graph=graph,
    )

    print("\n[3] Сравнение с истинным графом (CE):")
    compare_with_truth(mind_ce, graph, top_n=3)

    print("\n[4] Итог:")
    print(f"  {'':20s} {'hit@1':>7s} {'hit@3':>7s} {'hit@5':>7s} {'ordering':>9s}")
    print(f"  {'baseline':20s} {hit_b[1]:>7.3f} {hit_b[3]:>7.3f} {hit_b[5]:>7.3f} "
          f"{ordering_accuracy(base, graph):>9.3f}")
    print(f"  {'BPR (норм.)':20s} {hit_bpr[1]:>7.3f} {hit_bpr[3]:>7.3f} {hit_bpr[5]:>7.3f} "
          f"{ordering_accuracy(mind_bpr, graph):>9.3f}")
    print(f"  {'CE (норм.)':20s} {hit_ce[1]:>7.3f} {hit_ce[3]:>7.3f} {hit_ce[5]:>7.3f} "
          f"{ordering_accuracy(mind_ce, graph):>9.3f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()