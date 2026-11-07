"""MindEngine — cognitive engine with dense / sparse / low-rank scaling.

Three association representations:
    dense     — full [N, N] matrix. Best for N < ~500. Exact but O(N²) work.
    sparse    — CSR sparse matrix with avg_degree edges per node.
    low_rank  — U [N, r] @ V.T [r, N]. Best for N in thousands:
                O((N + nnz) * r) per thought.

Mode is auto-selected by concept count if not specified.

Semantics of the association matrix:
    assoc[b, a] = how strongly thinking `a` activates `b`.
    Hence (assoc @ e_a) yields the activation caused by thought `a`.
    reinforce(a, b) strengthens "a → b" by updating assoc[b, a].
"""
from __future__ import annotations

import re
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Iterator, Optional, Protocol, Sequence
import time
import uuid

import torch

torch.sparse.check_sparse_tensor_invariants.disable()
warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support.*")
warnings.filterwarnings("ignore", message=".*Sparse invariant checks.*")


# ============================================================
# Device
# ============================================================

def pick_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# Data
# ============================================================

@dataclass
class Concept:
    id: str
    label: str = ""
    description: str = ""
    target: float = 0.0
    keywords: list[str] = field(default_factory=list)


@dataclass
class Thought:
    id: str
    content: str = ""
    signature: dict[str, float] = field(default_factory=dict)
    weight: float = 1.0
    source: str = ""
    timestamp: float = field(default_factory=time.time)

    @staticmethod
    def make(content: str = "", signature: Optional[dict] = None, *, id: Optional[str] = None,
             weight: float = 1.0, source: str = "", timestamp: Optional[float] = None) -> "Thought":
        return Thought(
            id=id or str(uuid.uuid4())[:8],
            content=content,
            signature=dict(signature or {}),
            weight=weight,
            source=source,
            timestamp=timestamp if timestamp is not None else time.time(),
        )


# ============================================================
# Text -> signature encoders
# ============================================================

class TextEncoder(Protocol):
    def __call__(self, text: str) -> dict[str, float]: ...


_WORD_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)
_CLAUSE_SPLIT_RE = re.compile(r"[,.;:!?]")
_NEG_WORDS = frozenset(("не", "нет", "no", "not", "without", "без"))


class KeywordEncoder:
    """Token matcher with stem matching and clause-local negation.

    word == term              exact match, any length
    OR term in word           substring, only when len(term) >= 3
    """

    def __init__(self, concepts: Sequence[Concept]):
        self.concepts = list(concepts)

    def _first_match_pos(self, text: str, terms: set[str]) -> Optional[int]:
        for m in _WORD_RE.finditer(text):
            word = m.group().lower()
            for term in terms:
                if word == term or (len(term) >= 3 and term in word):
                    return m.start()
        return None

    def _negated_in_clause(self, text: str, pos: int) -> bool:
        clause = _CLAUSE_SPLIT_RE.split(text[:pos])[-1]
        return any(w in _NEG_WORDS for w in _WORD_RE.findall(clause.lower()))

    def __call__(self, text: str) -> dict[str, float]:
        if not text:
            return {}
        sig: dict[str, float] = {}
        for c in self.concepts:
            terms = {
                t for t in (
                    c.id.lower(), c.label.lower(),
                    *(k.lower() for k in c.keywords),
                ) if t
            }
            if not terms:
                continue
            pos = self._first_match_pos(text, terms)
            if pos is None:
                continue
            sig[c.id] = -1.0 if self._negated_in_clause(text, pos) else 1.0
        return sig


class EmbeddingEncoder:
    def __init__(
        self,
        concepts: Sequence[Concept],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        top_k: int = 5,
        temperature: float = 0.3,
        min_weight: float = 0.05,
        device: str = "auto",
        warmup: bool = True,
    ):
        from sentence_transformers import SentenceTransformer

        self.concepts = list(concepts)
        self.top_k = top_k
        self.temperature = temperature
        self.min_weight = min_weight
        self.device = pick_device(device)

        self.model = SentenceTransformer(model_name, device=str(self.device))
        texts = [f"{c.label or c.id}. {c.description}".strip() for c in self.concepts]
        self.concept_emb = torch.tensor(
            self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True),
            dtype=torch.float32, device=self.device,
        )
        if warmup:
            self.model.encode(["warmup"], normalize_embeddings=True)

    @torch.no_grad()
    def __call__(self, text: str) -> dict[str, float]:
        e = torch.tensor(
            self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0],
            dtype=torch.float32, device=self.device,
        )
        sims = self.concept_emb @ e
        k = min(self.top_k, len(self.concepts))
        top = torch.topk(sims, k)
        weights = torch.softmax(top.values / self.temperature, dim=0)
        return {self.concepts[i].id: float(w)
                for i, w in zip(top.indices.tolist(), weights.tolist()) if w >= self.min_weight}

    @torch.no_grad()
    def encode_batch(self, texts: Sequence[str], batch_size: int = 32) -> list[dict[str, float]]:
        embs = self.model.encode(list(texts), normalize_embeddings=True,
                                  convert_to_numpy=True, batch_size=batch_size)
        E = torch.tensor(embs, dtype=torch.float32, device=self.device)
        sims = E @ self.concept_emb.T
        k = min(self.top_k, len(self.concepts))
        vals, idxs = torch.topk(sims, k, dim=-1)
        weights = torch.softmax(vals / self.temperature, dim=-1)
        return [
            {self.concepts[i].id: w for i, w in zip(row_i, row_w) if w >= self.min_weight}
            for row_w, row_i in zip(weights.tolist(), idxs.tolist())
        ]


# ============================================================
# Engine
# ============================================================

class MindEngine:
    """Signed activation over N concepts, evolved by thoughts.

    Update rule per thought:
        s_new = f( gamma * s + (w + A @ w) )

    where w is the (sparse) thought signature, A is the association matrix
    with A[b, a] = strength of "a activates b", gamma is per-thought
    forgetting, f is identity or tanh.

    State is always detached after each update — it is data, not a parameter.
    Only assoc_U / assoc_V remain nn.Parameters (for reinforce / training).
    """

    def __init__(
        self,
        concepts: Sequence[Concept],
        associations: str | Sequence | torch.Tensor = "random",
        *,
        seed: Optional[int] = 42,
        forgetting: float = 0.85,
        ambient_weight: float = 0.5,
        focus_temperature: float = 1.0,
        bound: Optional[float] = None,
        nonlinearity: Optional[str] = None,
        beta: float = 1.0,
        device: str = "auto",
        mode: Optional[str] = None,
        rank: Optional[int] = None,
        avg_degree: int = 32,
    ):
        self.device = pick_device(device)
        self.concepts = list(concepts)
        self.ids = [c.id for c in self.concepts]
        self._idx = {cid: i for i, cid in enumerate(self.ids)}
        self.n = len(self.ids)
        if self.n == 0:
            raise ValueError("At least one concept required")

        if mode is None:
            mode = "dense" if self.n < 500 else "low_rank"
        if mode not in ("dense", "sparse", "low_rank"):
            raise ValueError(f"Unknown mode: {mode!r}")
        self.mode = mode

        self.rank = rank
        self.avg_degree = avg_degree
        self.targets = torch.tensor([c.target for c in self.concepts],
                                     dtype=torch.float32, device=self.device)

        if mode == "low_rank":
            self.rank = rank or max(16, min(128, self.n // 8))
            U, V = self._build_low_rank(associations, seed, self.rank)
            self.assoc_U = torch.nn.Parameter(U.to(self.device))
            self.assoc_V = torch.nn.Parameter(V.to(self.device))
        elif mode == "sparse":
            self.assoc = self._build_sparse(associations, seed, avg_degree).to(self.device)
        else:
            self.assoc = self._build_dense(associations, seed).to(self.device)

        self.forgetting = forgetting
        self.ambient_weight = ambient_weight
        self.temperature = focus_temperature
        self.bound = bound
        self.nonlinearity = nonlinearity
        self.beta = beta

        self.state = torch.zeros(self.n, device=self.device)
        self.ambient: Optional[Thought] = None
        self.history: list[tuple[Thought, Optional[Thought]]] = []

    # ---------- construction ----------

    def _build_dense(self, mode, seed) -> torch.Tensor:
        if isinstance(mode, torch.Tensor):
            if mode.shape != (self.n, self.n):
                raise ValueError(f"assoc shape {tuple(mode.shape)} != ({self.n},{self.n})")
            return mode.float().clone()
        if isinstance(mode, (list, tuple)):
            return torch.tensor(mode, dtype=torch.float32)
        if mode == "identity":
            return torch.eye(self.n)
        if mode == "random":
            g = torch.Generator().manual_seed(seed) if seed is not None else None
            m = torch.randn((self.n, self.n), generator=g) * 0.08
            m.fill_diagonal_(0.0)
            return m
        raise ValueError(f"Unknown associations mode: {mode!r}")

    def _build_sparse(self, mode, seed, avg_degree: int) -> torch.Tensor:
        if isinstance(mode, torch.Tensor):
            if mode.shape != (self.n, self.n):
                raise ValueError(f"assoc shape {tuple(mode.shape)} != ({self.n},{self.n})")
            return mode.float() if mode.is_sparse_csr else mode.float().to_sparse_csr()
        if isinstance(mode, (list, tuple)):
            return torch.tensor(mode, dtype=torch.float32).to_sparse_csr()
        if mode == "identity":
            return torch.eye(self.n).to_sparse_csr()
        if mode == "random":
            g = torch.Generator().manual_seed(seed) if seed is not None else None
            total = avg_degree * self.n
            rows = torch.randint(0, self.n, (total,), generator=g)
            cols = torch.randint(0, self.n, (total,), generator=g)
            vals = torch.randn(total, generator=g) * 0.08
            mask = rows != cols
            rows, cols, vals = rows[mask], cols[mask], vals[mask]
            A = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (self.n, self.n))
            return A.coalesce().to_sparse_csr()
        raise ValueError(f"Unknown associations mode: {mode!r}")

    def _build_low_rank(self, mode, seed, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(mode, (torch.Tensor, list, tuple)):
            A = (mode.float() if isinstance(mode, torch.Tensor)
                 else torch.tensor(mode, dtype=torch.float32))
            if A.shape != (self.n, self.n):
                raise ValueError(f"assoc shape {tuple(A.shape)} != ({self.n},{self.n})")
            if A.is_sparse_csr:
                A = A.to_dense()
            U, S, V = torch.svd_lowrank(A.cpu(), q=min(rank, self.n))
            return (U * S.unsqueeze(0)), V
        if mode == "identity":
            eye = torch.eye(rank)
            return eye.repeat(self.n // rank + 1, 1)[:self.n], eye.repeat(self.n // rank + 1, 1)[:self.n]
        if mode == "random":
            g = torch.Generator().manual_seed(seed) if seed is not None else None
            U = torch.randn(self.n, rank, generator=g) * 0.1
            V = torch.randn(self.n, rank, generator=g) * 0.1
            return U, V
        raise ValueError(f"Unknown associations mode: {mode!r}")

    # ---------- signature helpers ----------

    def _sig_sparse(self, t: Thought) -> tuple[torch.Tensor, torch.Tensor]:
        if not t.signature:
            return (torch.empty(0, dtype=torch.long, device=self.device),
                    torch.empty(0, dtype=torch.float32, device=self.device))
        idx, val = [], []
        for cid, w in t.signature.items():
            i = self._idx.get(cid)
            if i is not None:
                idx.append(i)
                val.append(w)
        if not idx:
            return (torch.empty(0, dtype=torch.long, device=self.device),
                    torch.empty(0, dtype=torch.float32, device=self.device))
        return (torch.tensor(idx, dtype=torch.long, device=self.device),
                torch.tensor(val, dtype=torch.float32, device=self.device) * t.weight)

    def _build_S_batched(self, chunk: Sequence[Thought]) -> torch.Tensor:
        T = len(chunk)
        rows, cols, vals = [], [], []
        for i, t in enumerate(chunk):
            idx, val = self._sig_sparse(t)
            if len(idx) > 0:
                rows.append(torch.full_like(idx, i))
                cols.append(idx)
                vals.append(val)
        S = torch.zeros(T, self.n, device=self.device)
        if rows:
            S[torch.cat(rows), torch.cat(cols)] = torch.cat(vals)
        return S

    # ---------- association matvec per mode ----------

    def _assoc_matvec(self, s: torch.Tensor) -> torch.Tensor:
        if self.mode == "low_rank":
            return self.assoc_U @ (self.assoc_V.T @ s)
        if self.mode == "sparse":
            return torch.sparse.mm(self.assoc, s.unsqueeze(1)).squeeze(1)
        return self.assoc @ s

    def _assoc_batch(self, S: torch.Tensor) -> torch.Tensor:
        if self.mode == "low_rank":
            return (S @ self.assoc_V) @ self.assoc_U.T
        if self.mode == "sparse":
            return torch.sparse.mm(self.assoc, S.T).T
        return S @ self.assoc.T

    # ---------- core update ----------

    def _nonlinear(self, x: torch.Tensor) -> torch.Tensor:
        if self.nonlinearity == "tanh":
            return torch.tanh(self.beta * x)
        if self.nonlinearity is None:
            if self.bound is not None:
                return torch.clamp(x, -self.bound, self.bound)
            return x
        raise ValueError(f"Unknown nonlinearity: {self.nonlinearity!r}")

    def _apply(self, thought: Thought, ambient: Optional[Thought]) -> None:
        with torch.no_grad():
            idx, val = self._sig_sparse(thought)
            s = torch.zeros(self.n, device=self.device)
            if len(idx) > 0:
                s[idx] = val
            if ambient is not None:
                a_idx, a_val = self._sig_sparse(ambient)
                if len(a_idx) > 0:
                    s.index_add_(0, a_idx, a_val * self.ambient_weight)
            raw = self.forgetting * self.state + s + self._assoc_matvec(s)
            self.state = self._nonlinear(raw).detach()

    def _advance_linear(self, C: torch.Tensor) -> None:
        T = C.shape[0]
        k = torch.arange(T, device=self.device, dtype=torch.float32)
        weights = self.forgetting ** ((T - 1) - k)
        with torch.no_grad():
            self.state = ((self.forgetting ** T) * self.state
                          + (weights.unsqueeze(1) * C).sum(dim=0)).detach()

    def _replay(self) -> None:
        self.state = torch.zeros(self.n, device=self.device)
        for t, a in self.history:
            self._apply(t, a)

    # ---------- mutations ----------

    def think(self, thought: Thought) -> dict:
        self._apply(thought, self.ambient)
        self.history.append((thought, self.ambient))
        return self._focus_light()

    def think_text(self, text: str, encoder: TextEncoder, *, source: str = "", weight: float = 1.0) -> dict:
        return self.think(Thought.make(content=text, signature=encoder(text), source=source, weight=weight))

    def think_many(self, thoughts: Sequence[Thought], *, chunk_size: int = 512) -> torch.Tensor:
        linear = self.nonlinearity is None and self.bound is None
        for start in range(0, len(thoughts), chunk_size):
            chunk = thoughts[start:start + chunk_size]
            S = self._build_S_batched(chunk)
            if self.ambient is not None:
                a_idx, a_val = self._sig_sparse(self.ambient)
                if len(a_idx) > 0:
                    S[:, a_idx] += (a_val * self.ambient_weight).unsqueeze(0)
            with torch.no_grad():
                C = S + self._assoc_batch(S)
            if linear:
                self._advance_linear(C)
            else:
                for k in range(len(chunk)):
                    with torch.no_grad():
                        self.state = self._nonlinear(
                            self.forgetting * self.state + C[k]
                        ).detach()
            self.history.extend((t, self.ambient) for t in chunk)
        return self.state

    def forget_last(self) -> None:
        if not self.history:
            return
        self.history.pop()
        self._replay()

    def edit_thought(self, index: int, new: Thought) -> None:
        if not 0 <= index < len(self.history):
            raise IndexError(index)
        _, ambient = self.history[index]
        self.history[index] = (new, ambient)
        self._replay()

    def set_ambient(self, ambient: Optional[Thought]) -> None:
        self.ambient = ambient

    def replay_with_ambient(self, ambient: Optional[Thought]) -> None:
        self.ambient = ambient
        self.history = [(t, ambient) for t, _ in self.history]
        self._replay()

    def set_associations(self, assoc: torch.Tensor) -> None:
        if assoc.shape != (self.n, self.n):
            raise ValueError(f"assoc shape {tuple(assoc.shape)} != ({self.n},{self.n})")
        if self.mode == "low_rank":
            A = assoc.float().cpu()
            if A.is_sparse_csr:
                A = A.to_dense()
            U, S, V = torch.svd_lowrank(A, q=self.rank)
            self.assoc_U = torch.nn.Parameter((U * S.unsqueeze(0)).to(self.device))
            self.assoc_V = torch.nn.Parameter(V.to(self.device))
        elif self.mode == "sparse":
            self.assoc = (assoc.float() if assoc.is_sparse_csr
                          else assoc.float().to_sparse_csr()).to(self.device)
        else:
            self.assoc = (assoc.float().to_dense() if assoc.is_sparse_csr
                          else assoc.float()).to(self.device)
        self._replay()

    def reinforce(self, a: Thought, b: Thought, lr: float = 0.01) -> None:
        """Hebbian step: strengthen "thinking a activates b".

        Matrix semantics: assoc[b, a] += lr * outer(b_val, a_val).
        The reverse direction (b → a) needs a separate reinforce(b, a).
        """
        a_idx, a_val = self._sig_sparse(a)
        b_idx, b_val = self._sig_sparse(b)
        if len(a_idx) == 0 or len(b_idx) == 0:
            return

        if self.mode == "dense":
            with torch.no_grad():
                update = lr * torch.outer(b_val, a_val)          # [nb, na]
                self.assoc[b_idx.unsqueeze(1), a_idx.unsqueeze(0)] += update

        elif self.mode == "sparse":
            ii = b_idx.repeat_interleave(len(a_idx))              # rows: b
            jj = a_idx.repeat(len(b_idx))                         # cols: a
            vv = lr * (b_val.repeat_interleave(len(a_idx)) * a_val.repeat(len(b_idx)))
            update = torch.sparse_coo_tensor(torch.stack([ii, jj]), vv, (self.n, self.n))
            with torch.no_grad():
                combined = (self.assoc.to_sparse_coo() + update).coalesce()
                self.assoc = combined.to_sparse_csr()

        elif self.mode == "low_rank":
            with torch.no_grad():
                U_sel = self.assoc_U[b_idx]                       # [nb, r]
                V_sel = self.assoc_V[a_idx]                       # [na, r]
                pred = U_sel @ V_sel.T                            # [nb, na]
                target = torch.outer(b_val, a_val)                # [nb, na], no lr
                err = pred - target
                grad_U = err @ V_sel                              # [nb, r]
                grad_V = err.T @ U_sel                            # [na, r]
                self.assoc_U.index_add_(0, b_idx, -lr * grad_U)
                self.assoc_V.index_add_(0, a_idx, -lr * grad_V)

    def reset(self) -> None:
        self.state = torch.zeros(self.n, device=self.device)
        self.history.clear()
        self.ambient = None

    # ---------- branching ----------

    def _snapshot(self):
        if self.mode == "low_rank":
            assoc_snap = (self.assoc_U.data.clone(), self.assoc_V.data.clone())
        else:
            assoc_snap = self.assoc.clone()
        return (self.state.clone(), list(self.history), self.ambient, assoc_snap)

    def _restore(self, snap) -> None:
        state, history, ambient, assoc_snap = snap
        self.state = state.clone()
        self.history = list(history)
        self.ambient = ambient
        if self.mode == "low_rank":
            self.assoc_U.data = assoc_snap[0].clone()
            self.assoc_V.data = assoc_snap[1].clone()
        else:
            self.assoc = assoc_snap.clone()

    @contextmanager
    def branched(self) -> Iterator["MindEngine"]:
        snap = self._snapshot()
        try:
            yield self
        finally:
            self._restore(snap)

    def probe(self, thought: Thought) -> dict:
        snap = self._snapshot()
        try:
            self.think(thought)
            return self._focus_light()
        finally:
            self._restore(snap)

    # ---------- read ----------

    def _focus_light(self, top_k: int = 5, min_abs: float = 1e-3) -> dict:
        with torch.no_grad():
            vals = self.state
            k = min(top_k, self.n)
            top = torch.topk(vals.abs(), k)
            top_idx = top.indices.cpu().tolist()
            top_vals = vals[top.indices].cpu().tolist()
        dominant = []
        for i, v in zip(top_idx, top_vals):
            if abs(v) < min_abs:
                continue
            dominant.append({"concept": self.ids[i], "activation": float(v)})
        return {"dominant": dominant}

    def current_focus(
        self,
        top_k: int = 5,
        min_abs: float = 1e-3,
        full: bool = False,
        max_full_n: int = 500,
    ) -> dict:
        result = self._focus_light(top_k, min_abs)
        if full or self.n <= max_full_n:
            with torch.no_grad():
                vec_list = self.state.cpu().tolist()
            result["vector"] = {cid: v for cid, v in zip(self.ids, vec_list)}
        return result

    def attention(self, signed: bool = False, top_k: Optional[int] = None) -> dict[str, float]:
        with torch.no_grad():
            x = self.state if signed else self.state.abs()
            w = torch.softmax(x / self.temperature, dim=0)
            if top_k is not None and top_k < self.n:
                top = torch.topk(w, top_k)
                idx = top.indices.cpu().tolist()
                vals = w[top.indices].cpu().tolist()
                return {self.ids[i]: float(v) for i, v in zip(idx, vals)}
            w_list = w.cpu().tolist()
        return {cid: v for cid, v in zip(self.ids, w_list)}

    def deficit(self) -> torch.Tensor:
        with torch.no_grad():
            return torch.clamp(self.targets - self.state, min=0.0)

    # ---------- recommend ----------

    def _sparse_library(self, library: Sequence[Thought]) -> Optional[torch.Tensor]:
        T = len(library)
        rows, cols, vals = [], [], []
        for i, t in enumerate(library):
            idx, val = self._sig_sparse(t)
            if len(idx) > 0:
                rows.append(torch.full((len(idx),), i, dtype=torch.long, device=self.device))
                cols.append(idx)
                vals.append(val)
        if not rows:
            return None
        return torch.sparse_coo_tensor(
            torch.stack([torch.cat(rows), torch.cat(cols)]),
            torch.cat(vals), (T, self.n),
        ).coalesce()

    def recommend(
        self,
        library: Sequence[Thought],
        *,
        mode: str = "resonance",
        top_k: int = 5,
        resonance_weight: float = 0.5,
    ) -> list[dict]:
        if not library:
            return []
        W = self._sparse_library(library)
        if W is None:
            return []

        with torch.no_grad():
            res = torch.sparse.mm(W, self.state.unsqueeze(1)).squeeze(1)
            hom = torch.sparse.mm(W, self.deficit().unsqueeze(1)).squeeze(1)

            if mode == "resonance":
                scores = res
            elif mode == "homeostasis":
                scores = hom
            elif mode == "mixed":
                scores = resonance_weight * _norm(res) + (1.0 - resonance_weight) * _norm(hom)
            else:
                raise ValueError(f"Unknown recommend mode: {mode!r}")

            k = min(top_k, len(library))
            top = torch.topk(scores, k)
            idx = top.indices.cpu().tolist()
            vals = scores[top.indices].cpu().tolist()

        return [{"thought": library[i], "score": float(v)} for i, v in zip(idx, vals)]

    def continue_thought(
        self,
        library: Sequence[Thought],
        *,
        avoid_recent: int = 0,
        **kw,
    ) -> Optional[dict]:
        """Auto-flow: think the single top recommendation.

        `avoid_recent=N` excludes the ids of the last N thoughts from the
        library, preventing trivial loops like coffee → coffee → coffee.
        Caller-supplied `top_k` is dropped — this method returns exactly one.
        """
        kw.pop("top_k", None)
        if avoid_recent > 0 and self.history:
            recent = {t.id for t, _ in self.history[-avoid_recent:]}
            library = [t for t in library if t.id not in recent]
        if not library:
            return None
        recs = self.recommend(library, top_k=1, **kw)
        if not recs:
            return None
        top = recs[0]
        self.think(top["thought"])
        return top

    # ---------- persistence ----------

    def _config_dict(self) -> dict:
        return {
            "concepts": [asdict(c) for c in self.concepts],
            "forgetting": self.forgetting,
            "ambient_weight": self.ambient_weight,
            "focus_temperature": self.temperature,
            "bound": self.bound,
            "nonlinearity": self.nonlinearity,
            "beta": self.beta,
            "mode": self.mode,
            "rank": self.rank,
            "avg_degree": self.avg_degree,
        }

    def to_dict(self) -> dict:
        base = {
            "config": self._config_dict(),
            "state": self.state.cpu().tolist(),
            "ambient": _t2d(self.ambient),
            "history": [{"thought": _t2d(t), "ambient": _t2d(a)} for t, a in self.history],
        }
        if self.mode == "low_rank":
            base["assoc_U"] = self.assoc_U.detach().cpu().tolist()
            base["assoc_V"] = self.assoc_V.detach().cpu().tolist()
        elif self.mode == "sparse":
            base["assoc_sparse"] = {
                "crow_indices": self.assoc.crow_indices().cpu().tolist(),
                "col_indices": self.assoc.col_indices().cpu().tolist(),
                "values": self.assoc.values().cpu().tolist(),
                "shape": list(self.assoc.shape),
            }
        else:
            base["assoc"] = self.assoc.cpu().tolist()
        return base

    @classmethod
    def from_dict(cls, data: dict, *, device: str = "auto") -> "MindEngine":
        cfg = data["config"]
        concepts = [_concept_from_dict(c) for c in cfg["concepts"]]
        mode = cfg.get("mode", "dense")
        avg_degree = cfg.get("avg_degree", 32)

        if mode == "low_rank":
            eng = cls(concepts, associations="random", seed=0,
                      forgetting=cfg["forgetting"], ambient_weight=cfg["ambient_weight"],
                      focus_temperature=cfg["focus_temperature"], bound=cfg["bound"],
                      nonlinearity=cfg["nonlinearity"], beta=cfg["beta"],
                      device=device, mode=mode, rank=cfg.get("rank"))
            eng.assoc_U = torch.nn.Parameter(
                torch.tensor(data["assoc_U"], dtype=torch.float32, device=eng.device))
            eng.assoc_V = torch.nn.Parameter(
                torch.tensor(data["assoc_V"], dtype=torch.float32, device=eng.device))
        elif mode == "sparse":
            eng = cls(concepts, associations="identity",
                      forgetting=cfg["forgetting"], ambient_weight=cfg["ambient_weight"],
                      focus_temperature=cfg["focus_temperature"], bound=cfg["bound"],
                      nonlinearity=cfg["nonlinearity"], beta=cfg["beta"],
                      device=device, mode=mode, avg_degree=avg_degree)
            sp = data["assoc_sparse"]
            eng.assoc = torch.sparse_csr_tensor(
                torch.tensor(sp["crow_indices"], dtype=torch.long),
                torch.tensor(sp["col_indices"], dtype=torch.long),
                torch.tensor(sp["values"], dtype=torch.float32),
                size=tuple(sp["shape"]),
            ).to(eng.device)
        else:
            eng = cls(concepts, associations=torch.tensor(data["assoc"], dtype=torch.float32),
                      forgetting=cfg["forgetting"], ambient_weight=cfg["ambient_weight"],
                      focus_temperature=cfg["focus_temperature"], bound=cfg["bound"],
                      nonlinearity=cfg["nonlinearity"], beta=cfg["beta"],
                      device=device, mode=mode)

        eng.state = torch.tensor(data["state"], dtype=torch.float32, device=eng.device)
        eng.ambient = _d2t(data["ambient"])
        eng.history = [(_d2t(h["thought"]), _d2t(h["ambient"])) for h in data["history"]]
        return eng

    def save(self, path: str) -> None:
        payload = {
            "config": self._config_dict(),
            "state": self.state.cpu(),
            "ambient": self.ambient,
            "history": self.history,
        }
        if self.mode == "low_rank":
            payload["assoc_U"] = self.assoc_U.detach().cpu()
            payload["assoc_V"] = self.assoc_V.detach().cpu()
        else:
            payload["assoc"] = self.assoc.cpu()
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, *, device: str = "auto") -> "MindEngine":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = payload["config"]
        concepts = [_concept_from_dict(c) for c in cfg["concepts"]]
        mode = cfg["mode"]
        avg_degree = cfg.get("avg_degree", 32)

        if mode == "low_rank":
            eng = cls(concepts, associations="random", seed=0,
                      forgetting=cfg["forgetting"], ambient_weight=cfg["ambient_weight"],
                      focus_temperature=cfg["focus_temperature"], bound=cfg["bound"],
                      nonlinearity=cfg["nonlinearity"], beta=cfg["beta"],
                      device=device, mode=mode, rank=cfg["rank"])
            eng.assoc_U = torch.nn.Parameter(payload["assoc_U"].to(eng.device))
            eng.assoc_V = torch.nn.Parameter(payload["assoc_V"].to(eng.device))
        elif mode == "sparse":
            eng = cls(concepts, associations="identity",
                      forgetting=cfg["forgetting"], ambient_weight=cfg["ambient_weight"],
                      focus_temperature=cfg["focus_temperature"], bound=cfg["bound"],
                      nonlinearity=cfg["nonlinearity"], beta=cfg["beta"],
                      device=device, mode=mode, avg_degree=avg_degree)
            a = payload["assoc"]
            eng.assoc = (a if a.is_sparse_csr else a.to_sparse_csr()).to(eng.device)
        else:
            eng = cls(concepts, associations=payload["assoc"],
                      forgetting=cfg["forgetting"], ambient_weight=cfg["ambient_weight"],
                      focus_temperature=cfg["focus_temperature"], bound=cfg["bound"],
                      nonlinearity=cfg["nonlinearity"], beta=cfg["beta"],
                      device=device, mode=mode)

        eng.state = payload["state"].to(eng.device)
        eng.ambient = payload["ambient"]
        eng.history = payload["history"]
        return eng


# ============================================================
# Helpers
# ============================================================

def _norm(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    lo, hi = x.min(), x.max()
    if (hi - lo).abs() < 1e-9:
        return torch.zeros_like(x)
    return (x - lo) / (hi - lo)


def _t2d(t: Optional[Thought]) -> Optional[dict]:
    return asdict(t) if t is not None else None


def _d2t(d: Optional[dict]) -> Optional[Thought]:
    return Thought(**d) if d is not None else None


def _concept_from_dict(c: dict) -> Concept:
    return Concept(
        id=c["id"],
        label=c.get("label", ""),
        description=c.get("description", ""),
        target=c.get("target", 0.0),
        keywords=list(c.get("keywords", [])),
    )