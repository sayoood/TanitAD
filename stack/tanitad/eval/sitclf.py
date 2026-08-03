"""Situation classifier — the causal-history head and the late-fusion combiner.

WHAT DEFECT THIS FIXES
----------------------
Two, both MEASURED on the banked gen-1 held-out scores
(``…/2026-07-26-situation-classifier/artifacts/heldout_frames.npz``):

1. **The deployed multimodal arm is worse than its own ego-only ablation** on
   all three situations — ``head_img_ego`` 0.04347 / 0.01237 / 0.10757 against
   ``head_ego`` 0.08699 / 0.02328 / 0.13494 (lane_change / roundabout /
   intersection). Adding the camera *removes* 50 % / 47 % / 20 % of the ego
   head's AP. The mechanism is in the source: ``sc_train.py:143`` fuses by
   ``np.concatenate([img, S["E"]], 1)`` — an early concat of a 16-dim PCA image
   block normalised by its own global mean-abs (``:132-133``) against a 3-dim
   ego block scaled by a hand-set ``EGO_SCALE = [10, 2, 0.5]`` (``:38``). The
   two normalisations are unrelated, so the modalities enter the shared
   ``Linear`` at an arbitrary relative scale and the wider block wins.
   :func:`late_fuse_scores` replaces that with score-level fusion, which cannot
   be swamped because each modality is reduced to one calibrated number first.

2. **The causal window is shorter than the label horizon.** ``sc_train.py:37``
   fixes ``WIN = 8`` (0.8 s) while the label asks whether the ego will execute a
   manoeuvre over the NEXT ~3 s (mean positive run MEASURED at 27-29 frames).
   :class:`CausalSitHead` is the same architecture with the window as a
   parameter, so the constant can be measured instead of assumed.

Nothing here re-selects episodes or touches the label definition, so every
number stays comparable to the banked arms.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

SITUATIONS: tuple[str, ...] = ("lane_change", "roundabout", "intersection")
#: ``sc_train.py:38`` — the banked ``ego`` array is ALREADY divided by this
#: (``sc_train.py:93``), so features taken from the bundle need no rescaling.
EGO_SCALE = np.array([10.0, 2.0, 0.5], dtype=np.float32)


# --------------------------------------------------------------------------- #
# clip geometry                                                               #
# --------------------------------------------------------------------------- #
def clip_runs(clip_id) -> tuple[np.ndarray, np.ndarray]:
    """Contiguous ``[start, end)`` runs of equal ``clip_id``.

    The banked bundle stores frames in clip order, so a run IS a clip. Asserting
    that here is what makes a *causal* window well defined: without it a window
    could silently reach across a clip boundary and read another drive's past.
    """
    c = np.asarray(clip_id).ravel()
    if c.size == 0:
        raise ValueError("clip_runs needs a non-empty clip id array")
    chg = np.flatnonzero(np.diff(c) != 0) + 1
    return np.concatenate([[0], chg]), np.concatenate([chg, [c.size]])


def causal_window(feats, starts, ends, win: int) -> tuple[np.ndarray, np.ndarray]:
    """Flat causal history ``[N, win*C]`` at offsets ``-(win-1)..0`` + validity.

    Row ``t`` holds ``feats[t-win+1 .. t]`` — strictly past-and-present, and
    never crossing a clip boundary. Rows without a full in-clip history are
    zero-filled and marked invalid rather than edge-padded: padding would make
    the first frames of every clip look like a stationary vehicle, which is
    exactly the pattern the classifier is meant to key on.
    """
    if win < 1:
        raise ValueError(f"win must be >= 1, got {win}")
    X = np.asarray(feats, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"feats must be [N, C], got {X.shape}")
    n, c = X.shape
    out = np.zeros((n, c * win), dtype=np.float32)
    ok = np.zeros(n, dtype=bool)
    off = np.arange(-(win - 1), 1)
    for a, b in zip(starts, ends):
        if b - a < win:
            continue
        centres = np.arange(a + win - 1, b)
        idx = centres[:, None] + off[None, :]
        out[centres] = X[idx].reshape(centres.size, c * win)
        ok[centres] = True
    return out, ok


def cluster_folds(clip_id, n_folds: int = 2, seed: int = 0) -> np.ndarray:
    """Fold assignment per ROW, split on whole CLUSTERS.

    Splitting on rows would put frames of one clip on both sides and leak; the
    unit of independence here is the clip cluster, the same unit the interval
    estimator resamples.
    """
    c = np.asarray(clip_id).ravel()
    uniq = np.unique(c)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    fold_of = {int(uniq[p]): i % n_folds for i, p in enumerate(perm)}
    return np.array([fold_of[int(x)] for x in c], dtype=np.int8)


# --------------------------------------------------------------------------- #
# the head — sc_train.py:59-74 verbatim, with WIN as a parameter               #
# --------------------------------------------------------------------------- #
class CausalSitHead(nn.Module):
    """``sc_train.py``'s ``SitHead`` with the window length made explicit.

    Architecture, widths, dropout and the attention-pooled readout are
    unchanged, so a WIN=8 fit reproduces the deployed arm's regime and any
    difference at another WIN is attributable to the window alone.
    """

    def __init__(self, in_dim: int, win: int, d: int = 128,
                 n_out: int = len(SITUATIONS), dropout: float = 0.2):
        super().__init__()
        self.win = int(win)
        self.inp = nn.Linear(in_dim, d)
        self.pos = nn.Parameter(torch.zeros(self.win, d))
        layer = nn.TransformerEncoderLayer(d, 4, d * 4, dropout=dropout,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.att = nn.Linear(d, 1)
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` ``[B, win, in_dim]`` -> ``[B, n_out]`` independent logits."""
        if x.dim() != 3 or x.shape[1] != self.win:
            raise ValueError(f"expected [B, {self.win}, in_dim], got {tuple(x.shape)}")
        h = self.enc(self.inp(x) + self.pos)
        a = torch.softmax(self.att(h), 1)
        return self.mlp((h * a).sum(1))


def train_sit_head(Xtr, Ytr, Vtr, *, win: int, in_dim: int, epochs: int = 8,
                   pos_weight: float = 20.0, d: int = 128, batch: int = 1024,
                   lr: float = 3e-4, wd: float = 0.01, seed: int = 0,
                   device: str = "cpu", log=None) -> CausalSitHead:
    """Fit :class:`CausalSitHead` with ``sc_train.run_fold``'s exact recipe:
    AdamW(3e-4, wd 0.01), masked ``BCEWithLogitsLoss(pos_weight)``, grad-clip 1.0.

    ``Xtr`` is the FLAT window ``[N, win*in_dim]`` from :func:`causal_window`;
    ``Vtr`` is the per-situation validity mask, and invalid entries contribute
    zero loss (never a zero *label*, which would train the head to say "no").
    """
    torch.manual_seed(seed)
    model = CausalSitHead(in_dim, win, d=d, n_out=Ytr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    pw = torch.full((Ytr.shape[1],), float(pos_weight), device=device)
    lossf = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)
    X = torch.from_numpy(np.ascontiguousarray(Xtr, dtype=np.float32))
    Y = torch.from_numpy(np.ascontiguousarray(Ytr, dtype=np.float32))
    V = torch.from_numpy(np.ascontiguousarray(Vtr, dtype=np.float32))
    n = X.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot, seen = 0.0, 0
        for b in range(0, n, batch):
            j = perm[b:b + batch]
            vb = V[j].to(device)
            if float(vb.sum()) == 0:
                continue
            xb = X[j].to(device).view(len(j), win, in_dim)
            loss = (lossf(model(xb), Y[j].to(device)) * vb).sum() / vb.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(j)
            seen += len(j)
        if log:
            log(f"[sitclf win={win}] epoch {ep+1}/{epochs} loss {tot/max(seen,1):.4f}")
    return model


# --------------------------------------------------------------------------- #
# MATCHED CAPACITY (BACKLOG B4)                                               #
# --------------------------------------------------------------------------- #
#: ``CausalSitHead`` fixes 4 attention heads, so a legal width is a multiple of
#: 4. Solving a parameter budget over ALL integers silently proposes widths the
#: module refuses to build ("embed_dim must be divisible by num_heads").
HEAD_N_HEADS = 4


def head_param_count(in_dim: int, win: int, d: int,
                     n_out: int = len(SITUATIONS)) -> int:
    """Exact parameter count of :class:`CausalSitHead` at width ``d``.

    Counted by CONSTRUCTING the module, never by a closed-form formula. A
    formula is what let ``head_img`` be quoted at "2.17 M" when the checkpoint
    holds **417,028** (`checkpoints/head_img.pt`) — an arithmetic claim about an
    artifact that was never checked against the artifact.
    """
    d = int(d)
    if d < 1 or d % HEAD_N_HEADS:
        raise ValueError(f"width must be a positive multiple of {HEAD_N_HEADS}, got {d}")
    m = CausalSitHead(int(in_dim), int(win), d=d, n_out=int(n_out))
    return int(sum(p.numel() for p in m.parameters()))


def width_for_param_budget(budget: int, in_dim: int, win: int,
                           n_out: int = len(SITUATIONS),
                           d_max: int = 1024) -> int:
    """Smallest legal width whose parameter count is >= ``budget``.

    This is what makes a capacity ladder *matched*: rungs are specified as
    parameter budgets and the width is solved for, so a change of ``in_dim``
    (PCA rank) or ``win`` cannot silently move the rungs. Comparing a PCA-16
    head against a PCA-64 head at the same *width* would compare different
    capacities and call it a representation effect.
    """
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")
    h = HEAD_N_HEADS
    lo, hi = 1, int(d_max) // h
    if head_param_count(in_dim, win, hi * h, n_out) < budget:
        raise ValueError(f"budget {budget} unreachable at d <= {hi * h}")
    while lo < hi:
        mid = (lo + hi) // 2
        if head_param_count(in_dim, win, mid * h, n_out) >= budget:
            hi = mid
        else:
            lo = mid + 1
    return int(lo * h)


def ridge_scores(Xtr, Ytr, Vtr, Xte, lam: float = 1.0) -> np.ndarray:
    """CLOSED-FORM ridge on +-1 targets, per situation -> ``[n_te, n_out]``.

    Ported verbatim from ``sc_train.ridge_fit_predict`` (the recipe that
    produced the banked ``ridge_img`` arm) so the ladder's LINEAR floor is the
    same estimator the banked table reports, not a re-derivation of it. Closed
    form on purpose: with no optimiser there is no optimiser to blame when the
    floor beats a transformer.

    ``Xtr``/``Xte`` are the FLAT windows from :func:`causal_window`;
    standardisation uses TRAIN rows only and the intercept is never penalised.
    """
    A0 = np.asarray(Xtr, dtype=np.float64)
    B0 = np.asarray(Xte, dtype=np.float64)
    Y = np.asarray(Ytr)
    V = np.asarray(Vtr).astype(bool)
    if A0.ndim != 2 or B0.ndim != 2 or A0.shape[1] != B0.shape[1]:
        raise ValueError(f"ridge needs aligned [N, D] windows: {A0.shape} vs {B0.shape}")
    if Y.shape[0] != A0.shape[0] or V.shape != Y.shape:
        raise ValueError("ridge needs Ytr/Vtr aligned with Xtr")
    mu = A0.mean(0, keepdims=True)
    sd = np.maximum(A0.std(0, keepdims=True), 1e-3)
    A = np.concatenate([(A0 - mu) / sd, np.ones((len(A0), 1))], 1)
    B = np.concatenate([(B0 - mu) / sd, np.ones((len(B0), 1))], 1)
    eye = np.eye(A.shape[1])
    eye[-1, -1] = 0.0                       # never penalise the intercept
    out = np.zeros((len(B0), Y.shape[1]), np.float64)
    for i in range(Y.shape[1]):
        m = V[:, i]
        if m.sum() < 50:
            continue
        Am = A[m]
        t = np.where(Y[m, i].astype(bool), 1.0, -1.0)
        w = np.linalg.solve(Am.T @ Am + lam * eye, Am.T @ t)
        out[:, i] = B @ w
    return out.astype(np.float32)


def ridge_param_count(flat_dim: int, n_out: int = len(SITUATIONS),
                      per_situation: bool = True) -> int:
    """Parameters of :func:`ridge_scores`: ``flat_dim + 1`` weights per output.

    ``per_situation=False`` returns the per-head figure the banked
    ``train_summary.json`` records (``ridge_img`` -> **129** = 16 PCA dims x
    win 8 + intercept), which is the number that belongs on a capacity axis
    beside a transformer's total.
    """
    n = int(flat_dim) + 1
    return n * int(n_out) if per_situation else n


@torch.no_grad()
def predict_sit_head(model: CausalSitHead, X, in_dim: int, *, device: str = "cpu",
                     batch: int = 8192) -> np.ndarray:
    """Sigmoid scores ``[N, n_out]`` for flat windows ``X`` ``[N, win*in_dim]``."""
    model.eval()
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    out = []
    for b in range(0, Xt.shape[0], batch):
        xb = Xt[b:b + batch].to(device).view(-1, model.win, in_dim)
        out.append(torch.sigmoid(model(xb)).float().cpu().numpy())
    return (np.concatenate(out) if out
            else np.zeros((0, model.mlp[-1].out_features), np.float32))


# --------------------------------------------------------------------------- #
# late fusion — the fix for defect (1)                                         #
# --------------------------------------------------------------------------- #
def late_fuse_scores(score_cols, y, valid, folds, *, l2: float = 1.0,
                     max_iter: int = 300) -> np.ndarray:
    """Out-of-fit late fusion of per-modality SCORES for ONE situation.

    ``score_cols`` ``[N, K]`` are the unimodal scores (e.g. the image head and
    the ego head). A logistic combiner is fitted on the rows of every OTHER fold
    and applied to this one, so the returned score for a row was produced by a
    combiner that never saw it. With K of order 2-4 the combiner has a handful
    of parameters and cannot itself be the source of a gain — which is exactly
    what makes it a fair test of whether *late* fusion beats *early* concat.

    Rows outside ``valid`` are returned as ``-inf`` so a caller that forgets to
    mask them ranks them last instead of scoring them as a confident negative.
    """
    from sklearn.linear_model import LogisticRegression      # noqa: PLC0415

    F = np.asarray(score_cols, dtype=np.float64)
    if F.ndim != 2:
        raise ValueError(f"score_cols must be [N, K], got {F.shape}")
    y = np.asarray(y).astype(np.int64).ravel()
    valid = np.asarray(valid).astype(bool).ravel()
    folds = np.asarray(folds).ravel()
    if not (F.shape[0] == y.size == valid.size == folds.size):
        raise ValueError("late_fuse_scores needs aligned [N] inputs")
    out = np.full(y.size, -np.inf, dtype=np.float64)
    for f in np.unique(folds):
        tr = valid & (folds != f)
        te = valid & (folds == f)
        if te.sum() == 0:
            continue
        if tr.sum() < 50 or y[tr].sum() < 2 or y[tr].sum() == tr.sum():
            # not enough signal to fit a combiner: fall back to the FIRST column
            # rather than emitting a constant, which would destroy the ranking.
            out[te] = F[te, 0]
            continue
        mu = F[tr].mean(0)
        sd = np.maximum(F[tr].std(0), 1e-9)
        lr = LogisticRegression(C=1.0 / l2, max_iter=max_iter, solver="lbfgs")
        lr.fit((F[tr] - mu) / sd, y[tr])
        out[te] = lr.decision_function((F[te] - mu) / sd)
    return out
