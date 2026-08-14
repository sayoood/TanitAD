"""P1+P2 — latent-state decodability & nuisance non-retention (WM_PHYSICS_PROOF.md).

P1 (physical-state decodability curve): LINEAR probes (closed-form ridge) on the
ENCODED latent z_{t+k} vs the PREDICTED latent ẑ_{t+k} (k = 5, 10, 15, 20) for the
driving-state targets: ego speed at t+k, yaw rate at t+k, local path curvature
around t+k (refb_labels.path_curvature), and lead-vehicle gap at t+k (from the
pod-built I1a/P8 join file). A WM that "predicts pixels" but loses the driving
state fails here; one that carries it passes.

P2 (nuisance non-retention — the "right part" half): the SAME probe protocol on
variables a driving abstraction should DISCARD: episode identity (multinomial
logistic) and a static-appearance proxy (mean brightness of the WINDOW-START
frame — a pure appearance scalar derivable with no extra labels). Weather /
illumination CLASS labels arrive with the VLM labeling pipeline
(VLM_STRATEGIC_LABELING.md) and are n/a today — reported as such, never silently
dropped. Together P1↑ and P2↓ are the information-bottleneck signature of
learning the RELEVANT physical state.

PRE-REGISTERED GATES (committed BEFORE any number; written to <out>/p12_gate.json):
  P1 (per driving target):
    (a) retention  — R²(ẑ, k=10) >= 0.85 x R²(z, k=10);
    (b) smoothness — no cliff: R²(ẑ) drop between CONSECUTIVE evaluated horizons
        (the k grid 5, 10, 15, 20) < 0.25 per step. *(The doc's "error grows
        monotonically and smoothly with k (no cliff)" made concrete — the
        operationalisation is stated in the JSON.)*
  P2 (episode identity, the doc's "clip-ID decodability from ẑ DROPS with k"
      made concrete — this operationalisation IS the pre-registration):
    episode-ID OOF accuracy from ẑ at k=20 is (i) BELOW its k=5 value and
    (ii) < 50 % of its k=5 value. Chance = 1/n_episodes is reported alongside;
    if the k=5 accuracy is already at chance (<= 1.1 x chance) the predictor
    never carried episode identity at all and the gate PASSES with branch
    "at_chance_from_start" (non-retention holds trivially — stated, not hidden).

PROTOCOL (the leakage rule, binding):
  * 5-fold cross-validation with folds split by EPISODE, never by window — a
    window and its near-duplicate neighbour may not straddle train/test (the
    REF-A I-JEPA lesson: ~80 % of val inside train made the number unusable).
  * ⚠️ EXCEPTION, forced by construction, stated loudly: the EPISODE-IDENTITY
    target cannot use episode-disjoint folds — a held-out episode is an UNSEEN
    CLASS and its accuracy is 0 by construction, which would make the P2 gate
    vacuous (0 < 0). For that ONE target the folds are within-episode TEMPORAL
    BLOCKS (each episode's windows split into 5 contiguous runs, block j ->
    fold j), consistent across k so the gate's k-trend is well-defined. The
    deviation and its reason are recorded in the JSON ("fold_scheme").
  * Per-target exclusions are per-target with n reported: windows without a
    lead vehicle (no candidate in the corridor) or without a join label are
    EXCLUDED from the lead-gap probe, never imputed; an empty agents list IS a
    label (road clear -> no lead -> excluded from this target), a missing record
    is NO_LABEL (join doc §4).

ESTIMATOR NOTE: headline numbers are pooled OUT-OF-FOLD R² / accuracy over the
canonical eval grid (episodes < 40, stride 8 — the 881-window grid). The
DECISION-grade interval for any registry claim is the episode-cluster bootstrap
over the 40 val episodes (taniteval/ci.py) — run pod-side before publishing.

TIER STAMP: P1/P2 are T0-DIAGNOSTIC BY DESIGN (they interrogate representations
— WM_PHYSICS_PROOF.md "Discipline"); NEVER a driving-performance number, and no
ADE / four-family table is produced here (this is one probe row of the
WM-physics battery, not a driving eval).

SEAMS (imported, never re-implemented — train_p8_occupancy is the template):
  * frozen trunk + corpus + geometry: ``eval_flagship_v4`` (``_eval_cfg`` /
    ``resolve_eval_frames`` / ``_plan`` / ``build_v2_val_episodes`` /
    ``load_v1_from_ck`` MODE A) — byte-identical guards;
  * latents: ``train_p8_occupancy.p8_latents`` — the shifted-window encode for
    z_{t+k} and the imagination/canary-family ``rollout_transitions`` roll under
    RECORDED actions for ẑ_{t+k} (``trans[k-1][1]``), one implementation;
  * lead labels: ``train_p8_occupancy.JoinFileReader`` (the I1a/P8 join file;
    ego-frame agents, +x fwd +y left) + the ``lead_state_gate`` corridor
    constants (LEAD_LAT_M = 2.0, LEAD_MAX_GAP_M = 80.0, lead_state_gate.py:86-87;
    gap = cx - l/2, the ``lead_frame`` definition at :239). ⚠️ The join-file
    schema carries NO label_class, so the lead candidate set here is
    class-agnostic — a join built with ``agents_at_time(classes=
    VEHICLE_CLASSES)`` restores class purity POD-SIDE; stated in the JSON.
  * curvature: ``refb_labels.path_curvature`` over the LOCAL arc around t+k
    (+-5 steps = +-0.5 s, clipped to the episode; mean signed kappa).

P9 REUSE: <out>/probe_arrays.pt dumps the per-k latents (fp16), every target,
masks, fold ids and the full-set ridge probe directions (w, b, lambda) so the
P9 probe-gradient saliency pass rebuilds nothing.

⚠️ POD-SIDE ONLY for the full path (GPU + v5f checkpoint + v2 val corpus + join
file). Runnable here: ``python -m py_compile`` and the CPU tests
(``stack/tests/test_probe_latent.py``).

Usage (pod5; PYTHONPATH=/workspace/TanitAD/stack):

  OMP_NUM_THREADS=6 PYTHONPATH=/workspace/TanitAD/stack \
  python3 scripts/probe_latent_state.py \
      --ckpt /workspace/experiments/flagship-v5f-w120-30k/ckpt_30k_final.pt \
      --v2-val-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
      --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
      --v2-subframe 176x624 \
      --join-file /workspace/data/p8_join/agents.jsonl \
      --out /workspace/experiments/p12-latent-probes
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import refb_labels  # noqa: E402  (scripts/refb_labels.py — curvature/wrap)
from train_p8_occupancy import (JoinFileReader, p8_latents,  # noqa: E402
                                window_frame)

try:                                     # lead corridor: the SAME constants the
    from lead_state_gate import (LEAD_LAT_M, LEAD_MAX_GAP_M,  # noqa: E402
                                 VEHICLE_CLASSES)
except ImportError:                      # I1a gate fixed before any number was
    LEAD_LAT_M = 2.0                     # read (lead_state_gate.py:86-87);
    LEAD_MAX_GAP_M = 80.0                # fallback for hosts without pandas.
    VEHICLE_CLASSES = ("automobile", "heavy_truck", "bus", "other_vehicle",
                       "trailer")        # lead_state_gate.py:88 verbatim

KS_DEFAULT = (5, 10, 15, 20)             # 0.5/1/1.5/2 s @10 Hz — the WP_STEPS grid
GATE_K = 10                              # P1 retention horizon (doc verbatim)
P1_RETENTION = 0.85                      # R2(pred) >= 0.85 x R2(enc) at k=10
P1_CLIFF_DROP = 0.25                     # max R2(pred) drop per consecutive grid k
P2_K_LO, P2_K_HI = 5, 20                 # P2 decay endpoints
P2_RATIO_MAX = 0.5                       # acc(k=20) < 0.5 x acc(k=5)
P2_CHANCE_SLACK = 1.1                    # acc(k=5) <= 1.1 x chance => at-chance
DT = refb_labels.DT_DEFAULT              # 0.1 s — the 10 Hz contract
CURV_HALF_STEPS = 5                      # local-arc half width (+-0.5 s)
RIDGE_LAMBDAS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3)
MIN_N_PER_TARGET = 50                    # below this a probe row is not-computable
DRIVING_TARGETS = ("speed", "yaw_rate", "curvature", "lead_gap")


# ============================================================================
# ridge regression — closed form via one SVD per design matrix (pure numpy)
# ============================================================================
class RidgeSVD:
    """Closed-form ridge on a FIXED design matrix, any lambda, any target.

    One economy SVD of the centered X is shared across every (target, lambda)
    — the probe battery fits 5 targets x 6 lambdas per fold off a single
    factorisation. Solution (centered data): w = V diag(s/(s^2+lam)) U^T y_c,
    b = mean(y) - mean(X) @ w — identical to (X_c^T X_c + lam I)^-1 X_c^T y_c
    for every lam > 0 (pinned against the direct normal-equation solve in
    tests/test_probe_latent.py).
    """

    def __init__(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be [n, d], got {X.shape}")
        self.n, self.d = X.shape
        self.mx = X.mean(axis=0)
        Xc = X - self.mx
        self.U, self.s, self.Vt = np.linalg.svd(Xc, full_matrices=False)

    def fit(self, y: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
        """(w [d], b) for ridge parameter ``lam`` (> 0)."""
        if lam <= 0:
            raise ValueError(f"lam must be > 0, got {lam}")
        y = np.asarray(y, dtype=np.float64)
        my = y.mean()
        Uty = self.U.T @ (y - my)
        w = self.Vt.T @ ((self.s / (self.s ** 2 + lam)) * Uty)
        return w, float(my - self.mx @ w)

    def gcv(self, y: np.ndarray, lam: float) -> float:
        """Generalized cross-validation score (Golub-Heath-Wahba): n ||r||^2 /
        (n - df)^2 with df = sum s^2/(s^2+lam). Train-fold-internal criterion
        for lambda ONLY — model selection never touches the held-out fold."""
        y = np.asarray(y, dtype=np.float64)
        yc = y - y.mean()
        Uty = self.U.T @ yc
        shrink = self.s ** 2 / (self.s ** 2 + lam)
        resid_sq = float(yc @ yc - Uty @ ((2 * shrink - shrink ** 2) * Uty))
        df = float(shrink.sum())
        denom = max(self.n - df, 1e-9)
        return self.n * max(resid_sq, 0.0) / denom ** 2

    def best_lambda(self, y: np.ndarray,
                    lambdas: tuple[float, ...] = RIDGE_LAMBDAS) -> float:
        return min(lambdas, key=lambda lam: self.gcv(y, lam))


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float,
              mode: str = "svd") -> tuple[np.ndarray, float]:
    """Thin closed-form entry point. ``mode``: ``svd`` (the production path) |
    ``primal`` ((X_c^T X_c + lam I)^-1 X_c^T y_c) | ``dual``
    (X_c^T (X_c X_c^T + lam I)^-1 y_c) — the three are algebraically identical;
    the test pins them equal so the SVD path cannot drift."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if mode == "svd":
        return RidgeSVD(X).fit(y, lam)
    mx, my = X.mean(axis=0), y.mean()
    Xc, yc = X - mx, y - my
    n, d = Xc.shape
    if mode == "primal":
        w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(d), Xc.T @ yc)
    elif mode == "dual":
        w = Xc.T @ np.linalg.solve(Xc @ Xc.T + lam * np.eye(n), yc)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return w, float(my - mx @ w)


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float | None:
    """1 - SS_res/SS_tot against the mean of ``y``; None when y has no
    variance (R2 undefined — reported as not-computable, never 0-filled)."""
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0.0:
        return None
    return float(1.0 - ((y - yhat) ** 2).sum() / ss_tot)


# ============================================================================
# folds — episode-disjoint (the leakage rule) + the episode-ID exception
# ============================================================================
def episode_disjoint_folds(episode_uids: np.ndarray, n_folds: int = 5
                           ) -> np.ndarray:
    """Fold id per window; folds partition EPISODES, never windows.

    Deterministic greedy balance: episodes sorted by (-window count, uid), each
    assigned to the currently lightest fold — no RNG, so the assignment is
    reproducible from the uid set alone. Raises when there are fewer episodes
    than folds (an episode straddling folds is never an option)."""
    uids = np.asarray(episode_uids)
    uniq, counts = np.unique(uids, return_counts=True)
    if uniq.size < n_folds:
        raise ValueError(f"{uniq.size} episodes < {n_folds} folds — "
                         f"episode-disjoint CV is not constructible")
    order = sorted(range(uniq.size), key=lambda i: (-int(counts[i]), int(uniq[i])))
    load = [0] * n_folds
    fold_of_uid: dict[int, int] = {}
    for i in order:
        f = int(np.argmin(load))
        fold_of_uid[int(uniq[i])] = f
        load[f] += int(counts[i])
    return np.asarray([fold_of_uid[int(u)] for u in uids], dtype=np.int64)


def within_episode_block_folds(episode_uids: np.ndarray, n_folds: int = 5
                               ) -> np.ndarray:
    """Fold id per window for the EPISODE-IDENTITY target ONLY.

    Episode-disjoint folds are DEGENERATE for this target (a held-out episode
    is an unseen class — accuracy 0 by construction, and the P2 decay gate
    becomes vacuous), so each episode's windows are split IN ARRAY ORDER
    (= temporal order on the eval grid) into ``n_folds`` contiguous blocks,
    block j -> fold j. Every fold sees every (sufficiently long) episode; the
    temporal blocking keeps adjacent near-duplicate windows on one side of the
    split. This deviation is recorded in the output JSON."""
    uids = np.asarray(episode_uids)
    out = np.empty(uids.shape[0], dtype=np.int64)
    for u in np.unique(uids):
        pos = np.flatnonzero(uids == u)
        for j, block in enumerate(np.array_split(pos, n_folds)):
            out[block] = j
    return out


def _standardize(Xtr: np.ndarray, Xte: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (Xtr - mu) / sd, (Xte - mu) / sd


# ============================================================================
# cross-validated probes (pure numpy/torch, CPU-testable)
# ============================================================================
def ridge_probe_cv(X: np.ndarray, y: np.ndarray, fold_ids: np.ndarray,
                   lambdas: tuple[float, ...] = RIDGE_LAMBDAS) -> dict:
    """Pooled out-of-fold ridge probe. Per fold: standardize by TRAIN stats,
    pick lambda by GCV on the TRAIN fold, closed-form fit, predict the held-out
    fold. Headline = R2 over the pooled OOF predictions (estimator note in the
    module docstring)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    fold_ids = np.asarray(fold_ids)
    yhat = np.full(y.shape[0], np.nan)
    per_fold, lam_by_fold = [], []
    for f in sorted(set(fold_ids.tolist())):
        te = fold_ids == f
        tr = ~te
        if not te.any() or not tr.any():
            continue
        Xtr, Xte = _standardize(X[tr], X[te])
        solver = RidgeSVD(Xtr)
        lam = solver.best_lambda(y[tr], lambdas)
        w, b = solver.fit(y[tr], lam)
        yhat[te] = Xte @ w + b
        per_fold.append(r2_score(y[te], yhat[te]))
        lam_by_fold.append(lam)
    ok = np.isfinite(yhat)
    return {"r2": r2_score(y[ok], yhat[ok]) if ok.any() else None,
            "per_fold_r2": per_fold, "lambda_by_fold": lam_by_fold,
            "n": int(ok.sum())}


def multinomial_logistic_fit(X: torch.Tensor, y: torch.Tensor, n_classes: int,
                             *, l2: float = 1e-3, iters: int = 400,
                             lr: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-batch multinomial logistic (softmax) regression, zero-init (the
    problem is convex, so zero init + fixed iterations is deterministic with no
    RNG). Returns (W [d, C], b [C])."""
    d = X.shape[1]
    W = torch.zeros(d, n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(X @ W + b, y) \
            + l2 * (W ** 2).sum()
        loss.backward()
        opt.step()
    return W.detach(), b.detach()


def logistic_probe_cv(X: np.ndarray, y_class: np.ndarray,
                      fold_ids: np.ndarray, *, l2: float = 1e-3,
                      iters: int = 400, lr: float = 0.05) -> dict:
    """Pooled OOF accuracy of a linear softmax probe. Class set = the FULL
    label set (so every fold scores against the same C); chance = 1/C and the
    majority-class rate are reported alongside — an accuracy without its chance
    line is not interpretable."""
    X = np.asarray(X, dtype=np.float32)
    y_class = np.asarray(y_class, dtype=np.int64)
    fold_ids = np.asarray(fold_ids)
    classes = np.unique(y_class)
    remap = {int(c): i for i, c in enumerate(classes)}
    y = np.asarray([remap[int(c)] for c in y_class], dtype=np.int64)
    n_classes = classes.size
    pred = np.full(y.shape[0], -1, dtype=np.int64)
    per_fold = []
    for f in sorted(set(fold_ids.tolist())):
        te = fold_ids == f
        tr = ~te
        if not te.any() or not tr.any():
            continue
        Xtr, Xte = _standardize(X[tr].astype(np.float64),
                                X[te].astype(np.float64))
        W, b = multinomial_logistic_fit(
            torch.from_numpy(Xtr.astype(np.float32)),
            torch.from_numpy(y[tr]), n_classes, l2=l2, iters=iters, lr=lr)
        logits = torch.from_numpy(Xte.astype(np.float32)) @ W + b
        pred[te] = logits.argmax(dim=1).numpy()
        per_fold.append(float((pred[te] == y[te]).mean()))
    ok = pred >= 0
    counts = np.bincount(y, minlength=n_classes)
    return {"accuracy": float((pred[ok] == y[ok]).mean()) if ok.any() else None,
            "per_fold_accuracy": per_fold, "n": int(ok.sum()),
            "n_classes": int(n_classes),
            "chance": 1.0 / float(n_classes),
            "majority_rate": float(counts.max() / max(y.shape[0], 1))}


# ============================================================================
# targets (pure, CPU-testable)
# ============================================================================
def lead_gap_from_agents(agents: np.ndarray | None,
                         lat_max: float = LEAD_LAT_M,
                         max_gap: float = LEAD_MAX_GAP_M,
                         classes=None) -> float | None:
    """Lead-vehicle gap (m) from ego-frame agents ``[A, 6] = (cx, cy, yaw, l,
    w, occ)`` — the ``lead_state_gate.lead_frame`` definition on the join-file
    schema: gap = cx - l/2 (bumper distance, :239), candidate iff gap in
    [0, LEAD_MAX_GAP_M] and |cy| < LEAD_LAT_M (:253); the lead is the NEAREST
    candidate. ``None`` = no lead (excluded per-target, n reported) — both for
    NO_LABEL (``agents is None``) and for a labelled-clear corridor."""
    if agents is None:
        return None
    ag = np.asarray(agents, dtype=np.float64)
    if ag.ndim != 2 or ag.shape[0] == 0:
        return None
    gap = ag[:, 0] - ag[:, 3] / 2.0
    cand = (gap >= 0.0) & (gap <= max_gap) & (np.abs(ag[:, 1]) < lat_max)
    if classes is not None:
        # the 2026-08-11 instrument fix: a pedestrian crossing the corridor is
        # not a LEAD VEHICLE — lead_state_gate.py:242's class gate, applied
        # here the moment the join carries classes. classes rows are aligned
        # with `agents` rows by JoinFileReader.lookup_classes.
        cls = np.asarray(classes, dtype=object)
        if cls.shape[0] != ag.shape[0]:
            raise ValueError(f"classes ({cls.shape[0]}) misaligned with "
                             f"agents ({ag.shape[0]})")
        cand &= np.isin(cls.astype(str), np.asarray(VEHICLE_CLASSES))
    if not cand.any():
        return None
    return float(gap[cand].min())


def yaw_rate_at_k(future_poses: torch.Tensor, k: int,
                  dt: float = DT) -> torch.Tensor:
    """Yaw rate (rad/s) at t+k from ``future_poses [B, H, 4]``: the wrapped
    one-step yaw delta ending AT t+k over dt (k >= 2 so both endpoints live in
    the future block; the probe grid 5/10/15/20 always satisfies this)."""
    if k < 2:
        raise ValueError(f"yaw_rate_at_k needs k >= 2, got {k}")
    dyaw = refb_labels.wrap_to_pi(future_poses[:, k - 1, 2]
                                  - future_poses[:, k - 2, 2])
    return dyaw / dt


def local_curvature(poses: torch.Tensor, center: int,
                    half_steps: int = CURV_HALF_STEPS) -> float | None:
    """Mean signed path curvature (1/m) of the LOCAL arc around pose index
    ``center``: ``refb_labels.path_curvature`` over poses[center-half ..
    center+half] clipped to the episode. None when fewer than 3 poses survive
    the clip (< 2 curvature samples — excluded, n reported)."""
    lo = max(0, center - half_steps)
    hi = min(poses.shape[0] - 1, center + half_steps)
    if hi - lo + 1 < 3:
        return None
    kappa = refb_labels.path_curvature(poses[lo:hi + 1])
    return float(kappa.mean())


# ============================================================================
# gates (pure; every branch CPU-tested)
# ============================================================================
def p1_gate_dict(table: dict, ks: tuple[int, ...], *, gate_k: int = GATE_K,
                 retention: float = P1_RETENTION,
                 cliff_drop: float = P1_CLIFF_DROP,
                 targets: tuple[str, ...] = DRIVING_TARGETS) -> dict:
    """P1 verdict from the per-target-per-k probe table.

    ``table[target][k]`` needs ``r2_enc``, ``r2_pred`` (float | None) and
    ``n``. Per target: (a) retention at ``gate_k`` — undefined (not-computable,
    never a fake verdict) when the row is missing, n < MIN_N_PER_TARGET, or
    R2_enc <= 0 (retention against a failed encoded probe is meaningless);
    (b) smoothness — every consecutive-k drop of R2_pred < ``cliff_drop``.
    Overall PASS = every COMPUTABLE target passes and >= 1 is computable;
    not-computable targets are listed with reasons, never silently dropped."""
    ks_sorted = tuple(sorted(ks))
    per_target: dict[str, dict] = {}
    computable_passes: list[bool] = []
    for tgt in targets:
        rows = table.get(tgt, {})
        row = rows.get(gate_k, rows.get(str(gate_k)))
        v: dict = {"gate_k": gate_k, "retention_threshold": retention,
                   "cliff_drop_threshold": cliff_drop}
        reason = None
        if row is None:
            reason = f"k={gate_k} not evaluated"
        elif row.get("n", 0) < MIN_N_PER_TARGET:
            reason = (f"n={row.get('n', 0)} < {MIN_N_PER_TARGET} at k={gate_k} "
                      f"— too few labelled windows")
        elif row.get("r2_enc") is None or row.get("r2_pred") is None:
            reason = f"probe R2 missing at k={gate_k}"
        elif row["r2_enc"] <= 0.0:
            reason = ("R2(enc) <= 0 at the gate horizon — the encoded-latent "
                      "probe itself failed; retention is undefined (fix the "
                      "probe/target before gating the predictor)")
        if reason is not None:
            v.update(computable=False, reason=reason, **{"pass": None})
            per_target[tgt] = v
            continue
        ratio = float(row["r2_pred"]) / float(row["r2_enc"])
        retention_ok = ratio >= retention
        drops, cliff_ok = [], True
        for a, b in zip(ks_sorted[:-1], ks_sorted[1:]):
            ra = rows.get(a, rows.get(str(a), {})).get("r2_pred")
            rb = rows.get(b, rows.get(str(b), {})).get("r2_pred")
            if ra is None or rb is None:
                drops.append(None)
                cliff_ok = False        # a hole in the curve cannot certify
                continue                # smoothness — conservative, stated
            d = float(ra) - float(rb)
            drops.append(round(d, 6))
            if d >= cliff_drop:
                cliff_ok = False
        ok = bool(retention_ok and cliff_ok)
        v.update(computable=True, reason=None,
                 r2_enc_at_gate_k=float(row["r2_enc"]),
                 r2_pred_at_gate_k=float(row["r2_pred"]),
                 retention_ratio=round(ratio, 6), retention_ok=retention_ok,
                 r2_pred_drops_per_step=drops, cliff_ok=cliff_ok,
                 n=int(row["n"]), **{"pass": ok})
        per_target[tgt] = v
        computable_passes.append(ok)
    overall = all(computable_passes) if computable_passes else None
    return {
        "rule": (f"per driving target: R2(z_hat, k={gate_k}) >= {retention} x "
                 f"R2(z_enc, k={gate_k}); AND R2(z_hat) drop between "
                 f"consecutive evaluated horizons (k grid {ks_sorted}) "
                 f"< {cliff_drop} (operationalisation of 'error grows smoothly"
                 f", no cliff')"),
        "per_target": per_target,
        "n_computable": len(computable_passes),
        "PASS": overall,
    }


def p2_gate_dict(acc_by_k: dict, chance: float, *, k_lo: int = P2_K_LO,
                 k_hi: int = P2_K_HI, ratio_max: float = P2_RATIO_MAX,
                 chance_slack: float = P2_CHANCE_SLACK) -> dict:
    """P2 verdict on episode-ID accuracy from the PREDICTED latent.

    Pre-registered operationalisation of the doc's "clip-ID decodability from
    z_hat DROPS with k" (stated in the JSON as the pre-registration):
      falls    — acc(k_hi) < acc(k_lo);
      ratio    — acc(k_hi) < ratio_max x acc(k_lo);
      at-chance branch — acc(k_lo) <= chance_slack x chance means the predictor
        never carried episode identity at all: non-retention holds TRIVIALLY,
        gate PASSES with branch "at_chance_from_start" (stated, not hidden).
    Missing k rows -> not computable (pass None), never a fake verdict."""
    gate = {"rule": (f"episode-ID OOF accuracy from z_hat at k={k_hi} is BELOW "
                     f"its k={k_lo} value AND < {ratio_max} x its k={k_lo} "
                     f"value; chance = 1/n_episodes reported alongside; "
                     f"acc(k={k_lo}) <= {chance_slack} x chance passes "
                     f"trivially (branch 'at_chance_from_start')"),
            "k_lo": k_lo, "k_hi": k_hi, "ratio_max": ratio_max,
            "chance": chance}
    a_lo = acc_by_k.get(k_lo, acc_by_k.get(str(k_lo)))
    a_hi = acc_by_k.get(k_hi, acc_by_k.get(str(k_hi)))
    if a_lo is None or a_hi is None:
        gate.update(computable=False,
                    reason=f"accuracy missing at k={k_lo} and/or k={k_hi}",
                    branch=None, **{"pass": None})
        return {"gate": gate, "PASS": None}
    a_lo, a_hi = float(a_lo), float(a_hi)
    if a_lo <= chance_slack * chance:
        gate.update(computable=True, reason=None, branch="at_chance_from_start",
                    acc_k_lo=a_lo, acc_k_hi=a_hi, ratio=None, **{"pass": True})
        return {"gate": gate, "PASS": True}
    ratio = a_hi / a_lo
    falls = a_hi < a_lo
    ok = bool(falls and ratio < ratio_max)
    gate.update(computable=True, reason=None, branch="decay_test",
                acc_k_lo=a_lo, acc_k_hi=a_hi, falls=falls,
                ratio=round(ratio, 6), **{"pass": ok})
    return {"gate": gate, "PASS": ok}


# ============================================================================
# collection (POD-SIDE: GPU + checkpoint + v2 val corpus + join file)
# ============================================================================
def collect_grid(world, ds_val, device, *, ks: tuple[int, ...], amp_on: bool,
                 join: JoinFileReader | None, episodes: int, stride: int,
                 batch: int) -> dict:
    """Latents + targets over the canonical eval grid (e < episodes, t % stride
    == 0 — the 881-window rule). Returns numpy arrays keyed for the probes."""
    from torch.utils.data import default_collate

    from train_flagship_v4 import _to_device
    sel = [i for i, (e, t) in enumerate(ds_val.index)
           if e < episodes and t % stride == 0]
    if not sel:
        raise SystemExit("[p12] grid selected 0 windows — check "
                         "--episodes/--stride against the val cache")
    ks_all = tuple(sorted(set(ks)))
    n = len(sel)
    S = world.state_dim
    z_enc = {k: np.empty((n, S), dtype=np.float32) for k in ks_all}
    z_hat = {k: np.empty((n, S), dtype=np.float32) for k in ks_all}
    speed = {k: np.empty(n, dtype=np.float64) for k in ks_all}
    yawrate = {k: np.empty(n, dtype=np.float64) for k in ks_all}
    curv = {k: np.full(n, np.nan) for k in ks_all}
    leadgap = {k: np.full(n, np.nan) for k in ks_all}
    lead_census = {k: {"no_label": 0, "labelled_no_lead": 0, "lead": 0}
                   for k in ks_all}
    bright = np.empty(n, dtype=np.float64)
    ep_uid = np.empty(n, dtype=np.int64)
    present = np.empty(n, dtype=np.int64)
    t0 = time.time()
    for b0 in range(0, n, batch):
        idx = sel[b0:b0 + batch]
        b = _to_device(default_collate([ds_val[i] for i in idx]), device)
        zt, ze, zh = p8_latents(world, b, ks_all, amp_on=amp_on,
                                want_pred=True, want_enc_k=True)
        del zt
        fp = b["future_poses"].float().cpu()
        bright_b = b["frames"][:, 0].float().mean(dim=(1, 2, 3)).cpu()
        for j, i in enumerate(idx):
            row = b0 + j
            eid, pf = window_frame(ds_val, i)
            ep_uid[row] = eid
            present[row] = pf
            bright[row] = float(bright_b[j])
            e_i, _t = ds_val.index[i]
            poses = torch.as_tensor(ds_val.episodes[e_i].poses).float()
            for k in ks_all:
                speed[k][row] = float(fp[j, k - 1, 3])
                yawrate[k][row] = float(yaw_rate_at_k(fp[j:j + 1], k)[0])
                c = local_curvature(poses, pf + k)
                curv[k][row] = np.nan if c is None else c
                if join is not None:
                    ag = join.lookup(eid, pf + k)
                    if ag is None:
                        lead_census[k]["no_label"] += 1
                    else:
                        g = lead_gap_from_agents(
                            ag, classes=join.lookup_classes(eid, pf + k))
                        if g is None:
                            lead_census[k]["labelled_no_lead"] += 1
                        else:
                            lead_census[k]["lead"] += 1
                            leadgap[k][row] = g
        for k in ks_all:
            z_enc[k][b0:b0 + len(idx)] = ze[k].cpu().numpy()
            z_hat[k][b0:b0 + len(idx)] = zh[k].cpu().numpy()
        if (b0 // batch) % 10 == 0:
            print(f"[p12] collected {b0 + len(idx)}/{n} windows "
                  f"({time.time() - t0:.0f} s)", flush=True)
    return {"z_enc": z_enc, "z_hat": z_hat, "speed": speed,
            "yaw_rate": yawrate, "curvature": curv, "lead_gap": leadgap,
            "lead_census": lead_census, "brightness": bright,
            "episode_uid": ep_uid, "present_frame": present,
            "grid_index": np.asarray(sel, dtype=np.int64),
            "n_grid_windows": n,
            "collect_wall_s": round(time.time() - t0, 1)}


def run_probes(col: dict, ks: tuple[int, ...], n_folds: int,
               join_available: bool) -> tuple[dict, dict, dict]:
    """All P1/P2 probes off the collected arrays (CPU, closed-form/deterministic).
    Returns (p1_table, p2_block, probe_fit_dump)."""
    ks_all = tuple(sorted(set(ks)))
    uids = col["episode_uid"]
    folds_ep = episode_disjoint_folds(uids, n_folds)
    folds_blk = within_episode_block_folds(uids, n_folds)

    target_vals = {"speed": col["speed"], "yaw_rate": col["yaw_rate"],
                   "curvature": col["curvature"], "lead_gap": col["lead_gap"]}
    p1_table: dict[str, dict] = {t: {} for t in DRIVING_TARGETS}
    fits: dict = {"ridge_probes": {}, "fold_ids_episode_disjoint": folds_ep,
                  "fold_ids_within_episode_blocks": folds_blk}
    for tgt in DRIVING_TARGETS:
        fits["ridge_probes"][tgt] = {}
        for k in ks_all:
            y = np.asarray(target_vals[tgt][k], dtype=np.float64)
            mask = np.isfinite(y)
            row: dict = {"n": int(mask.sum())}
            if tgt == "lead_gap" and not join_available:
                row.update(r2_enc=None, r2_pred=None, computable=False,
                           reason="no --join-file — lead labels unavailable")
                p1_table[tgt][k] = row
                continue
            if mask.sum() < MIN_N_PER_TARGET:
                row.update(r2_enc=None, r2_pred=None, computable=False,
                           reason=f"n={int(mask.sum())} < {MIN_N_PER_TARGET}")
                p1_table[tgt][k] = row
                continue
            fits["ridge_probes"][tgt][k] = {}
            for rep, feats in (("enc", col["z_enc"]), ("pred", col["z_hat"])):
                X = feats[k][mask]
                res = ridge_probe_cv(X, y[mask], folds_ep[mask])
                row[f"r2_{rep}"] = res["r2"]
                row[f"per_fold_r2_{rep}"] = res["per_fold_r2"]
                row[f"lambda_by_fold_{rep}"] = res["lambda_by_fold"]
                # full-set fit for the P9 saliency reuse (probe DIRECTION only —
                # never quoted as a held-out number)
                solver = RidgeSVD((X - X.mean(0)) / np.where(
                    X.std(0) < 1e-8, 1.0, X.std(0)))
                lam = solver.best_lambda(y[mask])
                w, b_ = solver.fit(y[mask], lam)
                fits["ridge_probes"][tgt][k][rep] = {
                    "w": torch.from_numpy(w.astype(np.float32)),
                    "b": float(b_), "lambda": float(lam),
                    "standardized_features": True}
            row["computable"] = True
            p1_table[tgt][k] = row

    # ---- P2: episode identity (block folds — documented deviation) ----------
    epid_acc: dict[str, dict] = {"enc": {}, "pred": {}}
    epid_detail: dict[str, dict] = {"enc": {}, "pred": {}}
    y_ep = uids
    for k in ks_all:
        for rep, feats in (("enc", col["z_enc"]), ("pred", col["z_hat"])):
            res = logistic_probe_cv(feats[k], y_ep, folds_blk)
            epid_acc[rep][k] = res["accuracy"]
            epid_detail[rep][k] = res
    chance = epid_detail["pred"][ks_all[0]]["chance"]

    # ---- P2: static-appearance proxy (brightness; ridge, disjoint folds) ----
    bright_r2: dict[str, dict] = {"enc": {}, "pred": {}}
    for k in ks_all:
        for rep, feats in (("enc", col["z_enc"]), ("pred", col["z_hat"])):
            res = ridge_probe_cv(feats[k], col["brightness"], folds_ep)
            bright_r2[rep][k] = res["r2"]

    p2 = {
        "episode_identity": {
            "probe": "multinomial logistic (linear softmax, zero-init, "
                     "full-batch, deterministic)",
            "fold_scheme": {
                "scheme": "within-episode temporal blocks (5 contiguous runs "
                          "per episode, block j -> fold j; consistent across k)",
                "deviation_from_episode_disjoint": True,
                "reason": "episode-disjoint folds are DEGENERATE for the "
                          "episode-identity target: a held-out episode is an "
                          "unseen class, accuracy is 0 by construction and the "
                          "decay gate becomes vacuous (0 < 0). All OTHER "
                          "targets use episode-disjoint folds (the leakage "
                          "rule)."},
            "accuracy_pred": {str(k): epid_acc["pred"][k] for k in ks_all},
            "accuracy_enc": {str(k): epid_acc["enc"][k] for k in ks_all},
            "detail": {rep: {str(k): epid_detail[rep][k] for k in ks_all}
                       for rep in ("enc", "pred")},
            "chance": chance,
        },
        "static_appearance_brightness": {
            "probe": "ridge (episode-disjoint folds) on mean WINDOW-START "
                     "frame brightness — a pure appearance scalar (no extra "
                     "labels needed)",
            "r2_pred": {str(k): bright_r2["pred"][k] for k in ks_all},
            "r2_enc": {str(k): bright_r2["enc"][k] for k in ks_all},
            "gated": False,
            "note": "reported, not gated — the P2 gate is on episode identity",
        },
        "weather_illumination": {
            "available": False,
            "reason": "weather/illumination CLASS labels arrive with the VLM "
                      "labeling pipeline (VLM_STRATEGIC_LABELING.md); n/a "
                      "today — reported per the discipline rule, not silently "
                      "dropped",
        },
    }
    return p1_table, p2, fits


# ============================================================================
# main
# ============================================================================
def build_args(argv=None):
    ap = argparse.ArgumentParser("probe_latent_state", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint with model+grounding keys (v5f/v4/v1 — "
                         "MODE A load, planner head unused)")
    ap.add_argument("--v2-val-cache", required=True, nargs="+",
                    help="v2 compressed VAL split dir(s) — probes run on the "
                         "val grid only; no train corpus is read")
    ap.add_argument("--v2-lru", type=int, default=64)
    ap.add_argument("--v2-subframe", default=None, metavar="HxW")
    ap.add_argument("--require-parity", action="store_true")
    from tanitad.geometry import add_geometry_args
    add_geometry_args(ap)      # --frame-h/--frame-w/--frame-hfov/--projection
    ap.add_argument("--join-file", default=None,
                    help="I1a/P8 agents jsonl (train_p8_occupancy.JoinFileReader "
                         "schema) for lead-gap labels; absent -> lead_gap "
                         "reported not-computable with reason, never faked")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--ks", default="5,10,15,20")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = build_args(argv)
    torch.manual_seed(a.seed)
    device = a.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[p12] WARNING: cuda unavailable, falling back to cpu", flush=True)
        device = "cpu"
    amp_on = (device == "cuda") and not a.no_amp
    ks = tuple(int(x) for x in str(a.ks).split(",") if x)
    os.makedirs(a.out, exist_ok=True)

    from eval_flagship_v4 import (_eval_cfg, _plan, build_v2_val_episodes,
                                  load_v1_from_ck, resolve_eval_frames)
    from train_flagship4b import FlagshipWindowDataset
    from train_v58f_unicycle_head import module_md5

    # ---- geometry FIRST (the W4/eval seam, not re-resolved here) ------------
    cfg = _eval_cfg()
    cache_frame, model_frame = resolve_eval_frames(a, cfg,
                                                   label="probe_latent_state")
    plan = _plan(cfg)
    if plan.max_horizon < max(ks):
        raise SystemExit(f"[p12] --ks max {max(ks)} > plan.max_horizon "
                         f"{plan.max_horizon} — future_frames cannot cover it")

    # ---- frozen trunk: MODE A (model + grounding; no planner head) ----------
    print(f"[p12] loading checkpoint {a.ckpt} ...", flush=True)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    # v6 checkpoints ({"stack": …}) rebuild a V6Stack and are wrapped in the
    # v5 trunk interface; v5 checkpoints take the byte-identical old path.
    # MEASURED 2026-08-14: without this the probe died on a v6 ckpt at
    # `predictor.act_emb.0.weight` — a v5 PARAMETER name (see v6_probe_trunk).
    from tanitad.eval.v6_probe_trunk import is_v6_checkpoint, load_trunk_auto
    is_v6 = is_v6_checkpoint(ck)
    world, _grounding, base_step = load_trunk_auto(
        ck, device, ckpt_path=a.ckpt, frame=model_frame)
    print(f"[p12] trunk generation: {'v6 (V6Stack)' if is_v6 else 'v5'}",
          flush=True)
    del ck
    assert not any(p.requires_grad for p in world.parameters())
    md5_before = module_md5(world)
    print(f"[p12] trunk frozen · base step {base_step} · "
          f"state_dim {world.state_dim} · md5 {md5_before[:12]}", flush=True)

    # ---- val data (the eval-side seam, imported) ----------------------------
    val_eps, val_prov = build_v2_val_episodes(
        a, cache_frame=cache_frame, train_frame=model_frame)
    # ⛔ The window is the TRUNK's, not the v5 eval default. MEASURED
    # 2026-08-14: v6's predictor is configured for window 6 while
    # `_eval_cfg()` says 8, and the mismatch surfaced as a ValueError deep
    # inside the predictor. Same class as the geometry seam above: a
    # property of the checkpoint must be read from the checkpoint.
    ds_val = FlagshipWindowDataset(val_eps,
                                   window=getattr(world, "window",
                                                  cfg.predictor.window),
                                   max_horizon=plan.max_horizon,
                                   maneuver_h=plan.maneuver_h,
                                   channels=cfg.encoder.in_channels)
    print(f"[p12] val {len(val_eps)} eps / {len(ds_val)} windows", flush=True)

    join = None
    if a.join_file:
        join = JoinFileReader(a.join_file)
        print(f"[p12] join file: {join.n_records} records, {join.n_clips} "
              f"clips", flush=True)
    else:
        print("[p12] no --join-file — lead_gap will be reported "
              "not-computable (reason recorded)", flush=True)

    # ---- collect latents + targets over the canonical grid ------------------
    col = collect_grid(world, ds_val, device, ks=ks, amp_on=amp_on, join=join,
                       episodes=a.episodes, stride=a.stride,
                       batch=a.eval_batch)
    md5_after = module_md5(world)

    # ---- probes (CPU, closed-form/deterministic) ----------------------------
    t0 = time.time()
    p1_table, p2, fits = run_probes(col, ks, a.folds,
                                    join_available=join is not None)
    p1_gate = p1_gate_dict(p1_table, ks)
    acc_pred = {k: p2["episode_identity"]["accuracy_pred"][str(k)]
                for k in sorted(set(ks))}
    p2_gate = p2_gate_dict(acc_pred, p2["episode_identity"]["chance"])

    summary = {
        "probe": "P1+P2 latent-state decodability & nuisance non-retention "
                 "(WM_PHYSICS_PROOF.md P1/P2)",
        "tier": "T0-diagnostic (representation probe — NEVER a driving-"
                "performance number; one row of the WM-physics battery, not a "
                "standalone driving eval; no ADE / four-family table is "
                "produced here by design)",
        "preregistration": {
            "p1": p1_gate["rule"],
            "p2": p2_gate["gate"]["rule"],
            "note": "gates committed in code before any number was produced; "
                    "the P2 operationalisation above IS the pre-registration "
                    "of the doc's 'drops with k'",
        },
        "p1": {
            "table": {t: {str(k): p1_table[t][k] for k in sorted(p1_table[t])}
                      for t in p1_table},
            "gate": p1_gate,
        },
        "p2": {**p2, "gate": p2_gate},
        "lead_label_provenance": {
            "source": "I1a/P8 join file (train_p8_occupancy.JoinFileReader)",
            "path": a.join_file,
            "corridor": {"lat_max_m": LEAD_LAT_M, "max_gap_m": LEAD_MAX_GAP_M,
                         "gap_def": "cx - l/2 (lead_state_gate.lead_frame:239)"},
            "class_filter": (
                {"vehicle_classes": list(VEHICLE_CLASSES),
                 "applied": "per-record (lookup_classes aligned rows)"}
                if getattr(join, "has_classes", False) else
                "NONE — this join file predates the `cls` field; rebuild with "
                "build_obstacle_join.py (2026-08 schema) for class purity"),
            "census_per_k": col["lead_census"],
        },
        "grid": {"episodes": a.episodes, "stride": a.stride,
                 "n_grid_windows": col["n_grid_windows"],
                 "n_episodes_on_grid": int(np.unique(
                     col["episode_uid"]).size)},
        "folds": {"n_folds": a.folds,
                  "scheme_driving_targets": "episode-disjoint (greedy "
                                            "count-balanced, deterministic)",
                  "scheme_episode_identity": "within-episode temporal blocks "
                                             "(documented deviation — see "
                                             "p2.episode_identity.fold_scheme)"},
        "trunk_frozen_proof": {"md5_before": md5_before,
                               "md5_after": md5_after,
                               "identical": md5_before == md5_after},
        "base_ckpt": a.ckpt, "base_step": base_step,
        "val_provenance": {kk: str(vv) for kk, vv in val_prov.items()},
        "probe_wall_s": round(time.time() - t0, 1),
        "collect_wall_s": col["collect_wall_s"],
        "_estimator_note": "headline numbers are pooled OUT-OF-FOLD R2 / "
                           "accuracy over the eval grid; the DECISION-grade "
                           "interval for any registry claim is the episode-"
                           "cluster bootstrap over the 40 val episodes "
                           "(taniteval/ci.py) — run it before publishing",
        "_evidence_class": "MEASURED (ours; artifact = this JSON)",
    }
    with open(os.path.join(a.out, "p12_gate.json"), "w") as gf:
        json.dump(summary, gf, indent=1)

    # ---- P9 reuse dump ------------------------------------------------------
    dump = {
        "z_enc": {int(k): torch.from_numpy(v).to(torch.float16)
                  for k, v in col["z_enc"].items()},
        "z_hat": {int(k): torch.from_numpy(v).to(torch.float16)
                  for k, v in col["z_hat"].items()},
        "targets": {
            "speed": {int(k): torch.from_numpy(v.astype(np.float32))
                      for k, v in col["speed"].items()},
            "yaw_rate": {int(k): torch.from_numpy(v.astype(np.float32))
                         for k, v in col["yaw_rate"].items()},
            "curvature": {int(k): torch.from_numpy(v.astype(np.float32))
                          for k, v in col["curvature"].items()},
            "lead_gap": {int(k): torch.from_numpy(v.astype(np.float32))
                         for k, v in col["lead_gap"].items()},
            "brightness": torch.from_numpy(
                col["brightness"].astype(np.float32)),
        },
        "episode_uid": torch.from_numpy(col["episode_uid"]),
        "present_frame": torch.from_numpy(col["present_frame"]),
        "grid_index": torch.from_numpy(col["grid_index"]),
        "fold_ids_episode_disjoint": torch.from_numpy(
            fits["fold_ids_episode_disjoint"]),
        "fold_ids_within_episode_blocks": torch.from_numpy(
            fits["fold_ids_within_episode_blocks"]),
        "ridge_probes": fits["ridge_probes"],
        "meta": {"ks": sorted(set(ks)), "base_ckpt": a.ckpt,
                 "base_step": base_step, "state_dim": world.state_dim,
                 "latent_dtype": "float16 (storage only; probes ran fp32)",
                 "nan_is_excluded": "NaN in a target = excluded window "
                                    "(NO_LABEL / no lead / clipped arc), "
                                    "never zero",
                 "purpose": "P9 probe-gradient saliency reuse "
                            "(WM_PHYSICS_PROOF.md P9)"},
    }
    torch.save(dump, os.path.join(a.out, "probe_arrays.pt"))

    print(f"\n[P12 SUMMARY] {json.dumps(summary, indent=1)}", flush=True)
    if not summary["trunk_frozen_proof"]["identical"]:
        raise SystemExit("⛔ TRUNK CHANGED DURING PROBING — run invalid")
    for name, g in (("P1", p1_gate), ("P2", p2_gate)):
        verdict = ("PASS" if g["PASS"] else
                   "FAIL" if g["PASS"] is not None else "NOT COMPUTABLE")
        print(f"[{name} GATE] {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
