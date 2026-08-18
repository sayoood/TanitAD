"""E-R1-0 — THE POOLING-RATIO LADDER (POOLING_BOTTLENECK_R1R2.md §7.1).

⛔ WHAT THIS IS. On ONE frozen banked checkpoint and ONE banked window set, run
the SAME linear readout at four spatial-averaging ratios that differ in NOTHING
but the pooling kernel over the SAME 16x40 encoder token grid:

    arm    kernel   units   tokens averaged      what it is
    p40    (4,10)      16          40            ⭐ THE DEPLOYED READOUT
    p10    (2,5)       64          10
    p4     (2,2)      160           4
    p1     (1,1)      640           1            ⭐ NO POOLING AT ALL

⛔ T0-DIAGNOSTIC. A frozen-latent linear readout is a WORLD-MODEL DIAGNOSTIC and
is NEVER driving performance. Nothing here is an ADE or a closed-loop number.

⛔ THE DIMENSION CONFOUND, AND WHY THE RANDOM PROJECTION IS NOT OPTIONAL.
Raw feature counts are 12 288 / 49 152 / 122 880 / 491 520 against n ~ 1 400
eval windows. A ridge on 491 520 features fits anything, and p1 would "win" for
reasons that have nothing to do with pooling. ⇒ EVERY arm is projected to
EXACTLY 2 048 features by a FIXED Gaussian random projection, and the whole
ladder is repeated over >= 5 projection seeds so the instrument reports its own
noise floor. (C92's trivial-proxy discipline, applied to dimensionality.)

⚠️ AND THE RP's OWN LIMIT, STATED RATHER THAN HIDDEN: a 2 048-dim projection of
a 491 520-dim space keeps a smaller *fraction* of its arm than a 2 048-dim
projection of a 12 288-dim space does. The RP therefore HANDICAPS the fine arms.
That makes a RISING ladder strong evidence and a FLAT ladder ambiguous — which
is exactly why the two planted positive controls below are mandatory, because
they measure how much a REAL pooling-destroyed signal survives THIS SAME RP.

⛔ C92 — THE INTERCEPT. `pc6_linear_readout.ridge_fit` penalises the intercept
by default, so a no-signal arm scores worse than a constant BY CONSTRUCTION.
Every fit here passes ``intercept_col=-1``, the argument is written into the
emitted JSON, and it is ASSERTED at startup (not remembered).

⛔ C97 — THE MIRROR-IMAGE DEFECT. Under the repaired ridge a fully shrunk fit is
the train MEAN while C-CONST is the train MEDIAN, so on a skewed target a pure
noise arm can PASS K1 while predicting a near-constant. ⇒ ``pred_sd/gt_sd`` is
emitted on EVERY row and any PASS with a ratio below ``--sd-ratio-floor`` is
stamped ``K1_DEGENERATE`` and is NOT quotable as a pass.

⭐ THE POSITIVE CONTROLS, AND WHY THERE ARE THREE (C79: D1 was withdrawn because
a probe failed its positive control). They are planted INTO THE REAL TOKENS, so
the background covariance is the real one:
  PC-DIST   the answer written into ALL 640 tokens (a global/distributed code).
            An unweighted mean PRESERVES a global code, so this must survive at
            every ratio. It proves the harness can read at every arm's n/p.
  PC-LOCAL  the answer written into ONE 2x2 token block that lies wholly inside
            ONE deployed 4x10 cell. Signal-to-noise falls as sqrt(k/K) under the
            mean, so this must DEGRADE monotonically with the pooling ratio. It
            calibrates HOW MUCH ladder slope a genuinely localised signal makes.
  PC-2OBJ   ⭐⭐ the sharpest one. TWO tokens INSIDE THE SAME 4x10 cell carry the
            answer with OPPOSITE SIGN. The cell mean annihilates it EXACTLY, and
            no other arm's cells contain both tokens. ⇒ the pre-registered
            prediction is a STEP: readable at 1:1, 4:1 and 10:1, dead at 40:1.
            This is the mechanism claim of §7.2 ("what a mean cannot do is
            separate two objects inside one cell") turned into an instrument.

CONTROLS ON EVERY ROW (a row without its floor is not a result)
  C-CONST   the probe-train median.               C-EPMEAN  own-episode LOO mean.
  C-V0      ⭐ the trivial-proxy arm: the SAME ridge on the ego-speed scalar
            alone. C92: v0 alone BEAT the whole 2 048-d latent on lead gap.
  r_partial ⭐ every correlation is ALSO reported with v0 partialled out.
  NULL      matched-random features per arm (same per-feature mu/sd).

⛔ NOTHING IS RE-IMPLEMENTED. The solve is `pc6_linear_readout.ridge_fit`; the
targets, the pose binding and the ladder rungs are `ll1_ladder`; the estimator
is `taniteval.ci.paired_episode_cluster_bootstrap`. `overlapping_holdout_se` is
never imported.

⛔ PARITY. This SELECTS NOTHING. It reads the banked window set of
`cache_tok11250` (the lead-enriched 130-clip probe set, `parity: False`,
inherited unchanged from the precedent) and a frozen checkpoint. No episode is
added, removed, reordered or re-hashed.
"""
from __future__ import annotations

# ⚠️ pyarrow BEFORE torch on this box (0xC0000005 segfault, proven A/B).
import pyarrow  # noqa: F401  # isort: skip

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve()
# ⛔ DEPTH-CORRECTED FOR THIS COPY. The ER10 original lives at
# .../Implementation/incoming/<dir>/code/ (6 levels under the repo root); this
# copy lives at .../Research/<dir>/code/ (5). Leaving parents[6] would resolve
# ABOVE the repo and every import would fail — or, worse on a machine where a
# sibling checkout exists, succeed against the WRONG tree.
_REPO = _HERE.parents[5]
assert (_REPO / "stack" / "tanitad").is_dir(),     f"repo root resolved to {_REPO}, which has no stack/tanitad" 
_INC = _REPO / "TanitAD Research Hub/Architecture & Inference/Implementation/incoming"
for _p in (_REPO / "taniteval", _REPO / "stack",
           _INC / "2026-08-17-probe-positive-control/code",
           _INC / "2026-08-17-slot-probe-parity/code",
           _INC / "2026-08-17-latent-linear-ladder/code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ll1_ladder as LL                                          # noqa: E402
from pc6_linear_readout import ridge_fit                         # noqa: E402
from taniteval.ci import (_draws,                                # noqa: E402
                          episode_cluster_bootstrap, episode_index,
                          paired_episode_cluster_bootstrap)

# --- the ladder of pooling ratios -----------------------------------------
# kernel (kh, kw) over the 16x40 token grid; ratio = kh*kw tokens per unit.
POOL_ARMS = {"p40": (4, 10), "p10": (2, 5), "p4": (2, 2), "p1": (1, 1), "s16": (4, 4)}
# ⛔ THE SEED KEY IS PINNED TO AN EXPLICIT TABLE, NOT TO sorted(POOL_ARMS).index().
# Adding the square-grid arm `s16` would otherwise SHIFT the sorted position of
# `p4`/`p40` and silently change the random projection of every previously
# published ER10 arm. This table reproduces sorted(['p1','p10','p4','p40'])
# exactly (asserted below) and gives the new arm a fresh, non-colliding index.
ARM_SEED_IDX = {"p1": 0, "p10": 1, "p4": 2, "p40": 3, "s16": 4}
assert [ARM_SEED_IDX[a] for a in ("p1", "p10", "p4", "p40")] == [0, 1, 2, 3]
assert sorted(a for a in POOL_ARMS if a != "s16") == ["p1", "p10", "p4", "p40"]
RP_DIM = 2048
RP_BLOCK_UNITS = 32          # P is generated in blocks of this many UNITS
PROJ_SEED_BASE = 20260818

# the four planted quantities, and the scale each is divided by before planting
ORACLE_TARGETS = (("ego_v0", 10.0), ("lead_gap", 15.0),
                  ("lead_closing", 2.0), ("ego_yawrate", 0.1))
# PC-LOCAL: a 2x2 token block wholly inside ONE deployed 4x10 cell (cell 1,1
# spans token rows 4..7 and token cols 10..19).
PC_LOCAL_TOKENS = ((6, 18), (6, 19), (7, 18), (7, 19))


def pc_local_tokens(th: int, tw: int, kernel) -> tuple:
    """⭐ PC-LOCAL, DERIVED FOR THIS GRID instead of hard-coded to 16x40.

    The hard-coded coords above are columns 18-19, which DO NOT EXIST on the
    16x16 square grid the REF-A-geometry arms produce — planting them would
    IndexError, or (worse, with negative wrap) plant somewhere else entirely.
    Returns the 2x2 token block at the bottom-right of cell (1,1), asserted to
    lie wholly inside ONE cell of `kernel`, which is what makes the control a
    test of pooling rather than of position.
    """
    kh, kw = kernel
    r1, c1 = 2 * kh - 1, 2 * kw - 1          # last row/col of cell (1,1)
    blk = ((r1 - 1, c1 - 1), (r1 - 1, c1), (r1, c1 - 1), (r1, c1))
    for (r, c) in blk:
        if not (0 <= r < th and 0 <= c < tw):
            raise SystemExit(f"[encloc] PC-LOCAL token ({r},{c}) outside the "
                             f"{th}x{tw} grid")
        if (r // kh, c // kw) != (1, 1):
            raise SystemExit(f"[encloc] PC-LOCAL token ({r},{c}) escapes cell "
                             f"(1,1) under kernel {kernel}")
    return blk
# PC-2OBJ: two tokens inside that SAME 4x10 cell, opposite sign. They fall in
# DIFFERENT cells for every other arm — asserted at runtime, never assumed.
PC_2OBJ_TOKENS = ((4, 10), (7, 19))


# ---------------------------------------------------------------------------
def _tok_index(r: int, c: int, tw: int) -> int:
    """readout.py:116 reshapes [B,N,D]->[B,D,th,tw]; token index is r*tw + c."""
    return r * tw + c


def assert_pc2obj_geometry(th: int, tw: int) -> dict:
    """⭐ PROVE the PC-2OBJ construction rather than asserting it in prose.

    The two planted tokens must share a cell under (4,10) and must NOT share a
    cell under any finer kernel — otherwise the pre-registered STEP is not what
    the control measures.
    """
    out = {}
    (r0, c0), (r1, c1) = PC_2OBJ_TOKENS
    for arm, (kh, kw) in POOL_ARMS.items():
        if max(r0, r1) >= th or max(c0, c1) >= tw:
            out[arm] = {"skipped": f"PC-2OBJ tokens outside this {th}x{tw} grid"}
            continue
        same = (r0 // kh == r1 // kh) and (c0 // kw == c1 // kw)
        out[arm] = {"same_cell": bool(same),
                    "cell_a": [r0 // kh, c0 // kw],
                    "cell_b": [r1 // kh, c1 // kw]}
    if not out["p40"].get("same_cell", True):
        raise SystemExit("[er10] ⛔ PC-2OBJ: the two tokens are NOT in the same "
                         "40:1 cell — the control does not test what it claims")
    for arm in ("p10", "p4", "p1"):
        if out[arm].get("same_cell"):
            raise SystemExit(f"[er10] ⛔ PC-2OBJ: the two tokens SHARE a cell at "
                             f"{arm} — the pre-registered step is not clean")
    return out


_ARM_NOW = ["p40"]   # set by build_features; PC-LOCAL is per-arm geometry


def plant_oracle(tok: torch.Tensor, u: np.ndarray, kind: str, amp: float,
                 th: int, tw: int, R: np.ndarray) -> torch.Tensor:
    """Add a planted code to the REAL tokens. ``tok`` [N, D] float32."""
    n_tok, d = tok.shape
    v = torch.from_numpy((R @ u).astype(np.float32))     # [D] the planted dir
    if kind == "dist":
        tok += amp * v.unsqueeze(0)
    elif kind == "local":
        for (r, c) in pc_local_tokens(th, tw, POOL_ARMS[_ARM_NOW[0]]):
            tok[_tok_index(r, c, tw)] += amp * v
    elif kind == "local2":
        (r0, c0), (r1, c1) = PC_2OBJ_TOKENS
        tok[_tok_index(r0, c0, tw)] += amp * v
        tok[_tok_index(r1, c1, tw)] -= amp * v
    else:
        raise SystemExit(f"[er10] unknown oracle {kind}")
    return tok


# ---------------------------------------------------------------------------
def make_projection(n_units: int, d_model: int, seed_key, device):
    """A FIXED Gaussian RP [n_units*d_model, RP_DIM], generated on the CPU in
    deterministic unit-blocks so the matrix is reproducible from the seed alone
    and never needs 4 GB of host RAM at once.

    ⚠️ Entries are N(0,1) and the output is NOT rescaled: the ladder z-scores
    every projected column with the probe-train mean/sd, so any global scale is
    normalised away and fp16 storage stays in its well-conditioned range.
    """
    rng = np.random.default_rng(seed_key)
    P = torch.empty((n_units * d_model, RP_DIM), dtype=torch.float16,
                    device=device)
    for s in range(0, n_units, RP_BLOCK_UNITS):
        e = min(n_units, s + RP_BLOCK_UNITS)
        blk = rng.standard_normal(((e - s) * d_model, RP_DIM), dtype=np.float32)
        P[s * d_model:e * d_model] = torch.from_numpy(blk).to(device).half()
    return P


def pool_tokens(tok: torch.Tensor, kernel, th: int, tw: int) -> torch.Tensor:
    """[C, N, D] -> [C, n_units*D], in the readout's own ordering.

    ⛔ THIS IS THE DEPLOYED OPERATOR, NOT A RE-DERIVATION.
    `readout.py:116-124` does exactly: transpose -> reshape(b,d,th,tw) ->
    AvgPool2d(kernel) -> flatten(2) -> transpose(1,2). Only the kernel varies.
    """
    c, n, d = tok.shape
    x = tok.transpose(1, 2).reshape(c, d, th, tw)
    x = F.avg_pool2d(x, kernel)
    x = x.flatten(2).transpose(1, 2)                     # [C, n_units, D]
    return x.reshape(c, -1)


# ---------------------------------------------------------------------------
def build_features(rows, arm, seeds, th, tw, d_model, device, oracle,
                   oracle_amp, oracle_u, oracle_R, chunk=48, block_diag=None):
    """-> {seed: X [n, RP_DIM] float64}. One pooling pass, all seeds."""
    kernel = POOL_ARMS[arm]
    _ARM_NOW[0] = arm
    n_units = (th // kernel[0]) * (tw // kernel[1])
    if th % kernel[0] or tw % kernel[1]:
        raise SystemExit(f"[encloc] ⛔ kernel {kernel} does not divide the "
                         f"{th}x{tw} token grid — avg_pool2d would SILENTLY "
                         f"drop the remainder rows/cols.")
    out = {}
    # ⛔ ONE projection matrix RESIDENT AT A TIME. For the widest arm it is
    # n_units*d_model*RP_DIM fp16 — 2.0 GB at d_model 768 and 6.0 GB at the
    # external encoder's 2304 — so holding every seed at once spills into host
    # memory over WDDM and turns a 2-minute arm into a long one.
    # ⚠️ Seeds are INDEPENDENT BY CONSTRUCTION (`make_projection` is seeded on
    # [BASE, seed, arm]), so this loop order changes SPEED and NOT ONE NUMBER.
    for s in seeds:
        if block_diag:
            # [d_model, D] shared by ALL cells -> effective output n_units*D.
            rng = np.random.default_rng([PROJ_SEED_BASE, s,
                                         ARM_SEED_IDX[arm], 977])
            Pc = torch.from_numpy(rng.standard_normal(
                (d_model, int(block_diag)), dtype=np.float32)).to(device).half()
            P = None
        else:
            P = make_projection(n_units, d_model,
                                [PROJ_SEED_BASE, s, ARM_SEED_IDX[arm]],
                                device)
        X = np.empty((len(rows),
                      n_units * int(block_diag) if block_diag else RP_DIM),
                     dtype=np.float64)
        for s0 in range(0, len(rows), chunk):
            sl = rows[s0:s0 + chunk]
            tk = torch.stack([r["tokens"].float() for r in sl])   # [C,N,D]
            if oracle is not None:
                for j, r in enumerate(sl):
                    tk[j] = plant_oracle(tk[j], oracle_u[s0 + j], oracle,
                                         oracle_amp, th, tw, oracle_R)
            tk = tk.to(device).half()
            pooled = pool_tokens(tk, kernel, th, tw)
            if block_diag:
                # [C, n_units, d_model] @ [d_model, D] -> [C, n_units*D]
                cellwise = pooled.reshape(pooled.shape[0], n_units, d_model)
                X[s0:s0 + len(sl)] = (cellwise @ Pc).reshape(
                    pooled.shape[0], -1).float().cpu().numpy()
            else:
                X[s0:s0 + len(sl)] = (pooled @ P).float().cpu().numpy()
        out[s] = X
        del P
        Pc = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out, n_units


# ---------------------------------------------------------------------------
def fit_one(Ztr, ytr, ctr, Zev, yev, eev, cev, alphas, inner_frac, seed,
            n_boot, v0ev, sd_floor, icol=-1):
    """pc6's fit with the C92 REPAIR (`intercept_col=-1`) and the C97 GUARD.

    ⛔ The design matrix's LAST column is the ones-column and it is EXCLUDED
    from the penalty. The argument is emitted, not remembered.
    ⚠️ ``icol=None`` reproduces the INCUMBENT BIASED solve and exists for ONE
    purpose: the reproduction gate against banked `ll_*.json`. It must never be
    used for a finding.
    """
    rng = np.random.default_rng(seed)
    clips = np.array(sorted(set(ctr.tolist())))
    rng.shuffle(clips)
    n_in = max(1, int(round(len(clips) * inner_frac)))
    inner = set(clips[:n_in].tolist())
    m_in = np.array([c in inner for c in ctr])
    best, best_mae, tried = None, np.inf, {}
    for al in alphas:
        w = ridge_fit(Ztr[~m_in], ytr[~m_in], al, intercept_col=icol)
        mae = float(np.abs(Ztr[m_in] @ w - ytr[m_in]).mean())
        tried[f"{al:g}"] = round(mae, 6)
        if mae < best_mae:
            best, best_mae = al, mae
    w = ridge_fit(Ztr, ytr, best, intercept_col=icol)
    pred = Zev @ w

    const_v = float(np.median(ytr))
    epmean = LL.loo_epmean(yev, cev, const_v)
    e_arm, e_con, e_ep = (np.abs(pred - yev), np.abs(const_v - yev),
                          np.abs(epmean - yev))
    arm_ci = episode_cluster_bootstrap(e_arm, eev, n_boot=n_boot)
    k1 = paired_episode_cluster_bootstrap(e_arm, e_con, eev, n_boot=n_boot)
    k5 = paired_episode_cluster_bootstrap(e_arm, e_ep, eev, n_boot=n_boot)
    sst = float(((yev - yev.mean()) ** 2).sum())
    r2 = float(1.0 - ((pred - yev) ** 2).sum() / sst) if sst > 0 else float("nan")
    psd, gsd = float(pred.std()), float(yev.std())
    ratio = psd / max(gsd, 1e-12)
    passes = bool(k1["separated"] and k1["delta"] < 0)
    out = {"alpha_chosen": best, "alpha_inner_mae": tried,
           "alpha_at_grid_edge": bool(best in (alphas[0], alphas[-1])),
           "err": round(float(e_arm.mean()), 4),
           "err_lo": arm_ci["lo"], "err_hi": arm_ci["hi"],
           "c_const_value": round(const_v, 4),
           "c_const_err": round(float(e_con.mean()), 4),
           "c_epmean_err": round(float(e_ep.mean()), 4),
           "R2": round(r2, 4),
           "K1_delta": k1["delta"], "K1_lo": k1["lo"], "K1_hi": k1["hi"],
           "K1_separated": k1["separated"], "K1_PASSES": passes,
           "K5_delta": k5["delta"], "K5_PASSES": bool(k5["separated"]
                                                      and k5["delta"] < 0),
           "corr": round(LL.corr(pred, yev), 4),
           "corr_within_ep": round(LL.corr(LL.demean_by(pred, cev),
                                           LL.demean_by(yev, cev)), 4),
           # ⭐ r^2 of the PREDICTION-TRUTH correlation: the ceiling an
           # optimally rescaled linear readout could reach. This is the ladder's
           # PRIMARY quantity, because it is invariant to the over/under
           # dispersion that C92/C97 both act on.
           "r2_ceiling": round(float(LL.corr(pred, yev) ** 2), 4),
           "pred_sd": round(psd, 4), "gt_sd": round(gsd, 4),
           # ⛔ C97 GUARD.
           "pred_sd_over_gt_sd": round(ratio, 4),
           "K1_DEGENERATE": bool(passes and ratio < sd_floor)}
    if v0ev is not None:
        out["corr_partial_v0"] = round(LL.partial_corr(pred, yev, v0ev), 4)
        out["r2_ceiling_partial_v0"] = round(
            float(LL.partial_corr(pred, yev, v0ev) ** 2), 4)
    return out, pred


# ---------------------------------------------------------------------------
def paired_delta_r2c(pred_a, pred_b, y, eid, n_boot, seed=0, alpha=0.05,
                     z=None):
    """⭐ THE PRE-REGISTERED STATISTIC: a PAIRED EPISODE-CLUSTER BOOTSTRAP on
    Δ r²_ceiling (arm_a − arm_b), i.e. corr(pred,y)² differenced between two
    arms scored on the SAME windows.

    ⛔ The resampling is `taniteval.ci`'s OWN — `episode_index` + `_draws`, the
    same functions `paired_episode_cluster_bootstrap` calls, at the same seed —
    so this is the module's estimator applied to a different statistic, NOT a
    second implementation of the bootstrap. `overlapping_holdout_se` is never
    imported.

    ``z`` (optional) partials a covariate (ego speed `v0`) out of BOTH the
    prediction and the truth INSIDE every draw, which is the only correct way
    to bootstrap a partial correlation.
    """
    def stat(sel):
        if z is None:
            ra = LL.corr(pred_a[sel], y[sel]) ** 2
            rb = LL.corr(pred_b[sel], y[sel]) ** 2
        else:
            ra = LL.partial_corr(pred_a[sel], y[sel], z[sel]) ** 2
            rb = LL.partial_corr(pred_b[sel], y[sel], z[sel]) ** 2
        return float(ra - rb)

    uniq, idx_by_ep = episode_index(eid)
    full = np.arange(len(y))
    point = stat(full)
    d = np.array([stat(sel) for sel in _draws(uniq, idx_by_ep, n_boot, seed)])
    d = d[np.isfinite(d)]
    lo, hi = (float(x) for x in np.percentile(d, [100 * alpha / 2,
                                                  100 * (1 - alpha / 2)]))
    return {"delta": round(point, 5), "lo": round(lo, 5), "hi": round(hi, 5),
            "p_delta_gt0": round(float((d > 0).mean()), 4),
            "separated": bool(lo > 0 or hi < 0),
            "n_windows": int(len(y)), "n_episodes": int(len(uniq)),
            "n_boot": int(len(d)),
            "estimator": "paired_episode_cluster_bootstrap (taniteval.ci "
                         "_draws/episode_index) on Δ corr²",
            "partialled": "v0" if z is not None else None}


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="a TOKEN-banked sp1 cache")
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--episodes-dir", required=True)
    ap.add_argument("--join-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--arms", nargs="+", default=list(POOL_ARMS) + ["cells"])
    ap.add_argument("--targets", nargs="+", default=LL.LADDER)
    ap.add_argument("--proj-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--ridge-seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    ap.add_argument("--inner-frac", type=float, default=0.25)
    ap.add_argument("--sd-ratio-floor", type=float, default=0.10)
    ap.add_argument("--oracle", choices=["dist", "local", "local2"], default=None)
    ap.add_argument("--oracle-amp", type=float, default=1.0,
                    help="planted amplitude in units of the REAL token sd")
    ap.add_argument("--randomise-features", type=int, default=None,
                    help="matched-random NULL: replace X with N(mu,sd) per "
                         "feature, mu/sd from the real projected features")
    ap.add_argument("--proxy-v0", action="store_true",
                    help="C-V0: replace every arm's features with the v0 scalar")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-rows", type=int, default=None,
                    help="SMOKE ONLY — strided subsample, never a finding")
    ap.add_argument("--legacy-penalised-intercept", action="store_true",
                    help="⚠️ REPRODUCTION GATE ONLY: run the INCUMBENT biased "
                         "solve (intercept_col=None). Never a finding.")
    ap.add_argument("--block-diag-proj", type=int, default=None,
                    metavar="D_READOUT",
                    help="⭐ E-ADAPT-0. Replace the DENSE random projection with "
                         "the SHAPE REF-A's adapter actually has: ONE shared "
                         "random [d_model -> D_READOUT] map applied to EVERY "
                         "pooled cell, concatenated (block-diagonal, weights "
                         "TIED across cells). The dense projection is strictly "
                         "MORE expressive, so the dense result is only an UPPER "
                         "BOUND on what `SpatialGridReadout`'s per-cell "
                         "Linear(d_model->d_readout) can preserve. This makes "
                         "the operator exact up to the weights being random "
                         "rather than trained.")
    ap.add_argument("--dump-preds", default=None,
                    help="⭐ pickle the eval-row PREDICTIONS + targets. The "
                         "within-run `deltas_vs_p40` cannot compare arms that "
                         "live in DIFFERENT caches (the geometry arms do), and "
                         "an unpaired difference of two bootstrap CIs is not a "
                         "paired test. Dumping predictions lets the paired "
                         "episode-cluster bootstrap run ACROSS caches, which is "
                         "the estimator the pre-registration commits to.")
    ap.add_argument("--gate-json", default=None,
                    help="a banked ll_*.json; asserts the `cells` arm at the "
                         "first proj seed reproduces its per-target numbers")
    a = ap.parse_args(argv)
    icol = None if a.legacy_penalised_intercept else -1

    # ⛔ ASSERT the C92 repair is reachable, at startup, from the SOURCE.
    import inspect
    if "intercept_col" not in inspect.signature(ridge_fit).parameters:
        raise SystemExit("[er10] ⛔ ridge_fit has no intercept_col — C92 repair "
                         "absent; refusing to run on the biased floor")
    _Xg = np.concatenate([np.random.default_rng(7).normal(size=(40, 3)),
                          np.ones((40, 1))], 1)
    _yg = np.random.default_rng(8).normal(size=40) + 5.0
    _wp = ridge_fit(_Xg, _yg, 1e9, intercept_col=None)
    _wu = ridge_fit(_Xg, _yg, 1e9, intercept_col=-1)
    intercept_gate = {"alpha": 1e9, "y_mean": round(float(_yg.mean()), 6),
                      "penalised_pred": round(float((_Xg @ _wp).mean()), 6),
                      "unpenalised_pred": round(float((_Xg @ _wu).mean()), 6)}
    if abs(float((_Xg @ _wu).mean()) - float(_yg.mean())) > 1e-4:
        raise SystemExit(f"[er10] ⛔ intercept gate FAILED: {intercept_gate}")

    dev = torch.device(a.device if (a.device == "cpu" or torch.cuda.is_available())
                       else "cpu")
    t0 = time.time()
    print(f"[er10] loading {a.cache}", flush=True)
    blob = torch.load(a.cache, map_location="cpu", weights_only=False)
    rows, meta = blob["rows"], blob["meta"]
    if a.limit_rows:
        rows = rows[::max(1, len(rows) // int(a.limit_rows))]
    if rows[0].get("tokens") is None:
        raise SystemExit("[er10] ⛔ this cache banked no tokens")
    th, tw = int(meta["token_grid"][0]), int(meta["token_grid"][1])
    d_model = int(rows[0]["tokens"].shape[-1])
    n_tok = int(rows[0]["tokens"].shape[0])
    if th * tw != n_tok:
        raise SystemExit(f"[er10] ⛔ token_grid {th}x{tw} != n_tokens {n_tok}")
    print(f"[er10] {len(rows)} rows  grid {th}x{tw}  d_model {d_model}  "
          f"{time.time()-t0:.0f} s", flush=True)

    pc2 = assert_pc2obj_geometry(th, tw)

    decl = json.loads(Path(a.split_json).read_text("utf-8"))
    ev_c, tr_c = set(decl["eval_clips"]), set(decl["train_clips"])
    clips_used = {r["clip_id"] for r in rows}
    need_ego = any(t in ("ego_accel", "ego_yawrate", "ego_curv")
                   for t in a.targets) or a.oracle is not None
    ego = LL.load_ego(Path(a.episodes_dir), Path(a.join_file),
                      clips_used) if need_ego else {}
    align = LL.bind_pose_grid(rows, ego) if need_ego else None
    pose_off = align["pose_index_offset"] if align else 0

    idx_tr = [i for i, r in enumerate(rows) if r["clip_id"] in tr_c]
    idx_ev = [i for i, r in enumerate(rows) if r["clip_id"] in ev_c]
    keep = idx_tr + idx_ev
    sub = [rows[i] for i in keep]
    pos_tr = np.arange(len(idx_tr))
    pos_ev = np.arange(len(idx_tr), len(keep))

    # --- the planted oracle, if any ----------------------------------------
    oracle_u = oracle_R = None
    token_sd = None
    if a.oracle is not None:
        smp = np.concatenate([sub[i]["tokens"].float().numpy().reshape(-1)
                              for i in range(0, len(sub), max(1, len(sub) // 64))])
        token_sd = float(smp.std())
        oracle_R = (np.random.default_rng([PROJ_SEED_BASE, 999])
                    .standard_normal((d_model, len(ORACLE_TARGETS)))
                    .astype(np.float32))
        oracle_R /= np.sqrt(len(ORACLE_TARGETS))
        oracle_u = np.zeros((len(sub), len(ORACLE_TARGETS)), dtype=np.float32)
        for j, r in enumerate(sub):
            for q, (tname, sc) in enumerate(ORACLE_TARGETS):
                v, ok = LL.target_of(r, ego, tname, pose_off)
                oracle_u[j, q] = (v / sc) if ok else 0.0
        amp = a.oracle_amp * token_sd
        print(f"[er10] ORACLE {a.oracle}  token_sd={token_sd:.6f}  "
              f"amp={amp:.6f}", flush=True)
    else:
        amp = 0.0

    res = {"_evidence_class":
           "MEASURED (ours; pooling-ratio ladder on a frozen banked checkpoint "
           "— a T0 world-model diagnostic, never driving performance)",
           "eval_tier": "T0-DIAGNOSTIC",
           "experiment": "E-R1-0 (POOLING_BOTTLENECK_R1R2.md §7.1)",
           "arm_label": a.label,
           "run_stamp": meta.get("run_stamp"), "step": meta.get("step"),
           "cache": str(a.cache), "token_grid": [th, tw], "d_model": d_model,
           "rp_dim": RP_DIM, "rp_block_units": RP_BLOCK_UNITS,
           "rp_seed_base": PROJ_SEED_BASE, "proj_seeds": a.proj_seeds,
           "ridge_seed": a.ridge_seed,
           "ridge_intercept_col": icol,
           "ridge_intercept_gate": intercept_gate,
           "legacy_penalised_intercept": bool(a.legacy_penalised_intercept),
           "solve_source": "pc6_linear_readout.ridge_fit (imported, not "
                           "re-implemented)",
           "estimator": "taniteval.ci.paired_episode_cluster_bootstrap",
           "forbidden": "overlapping_holdout_se",
           "n_boot": a.n_boot, "alphas": a.alphas,
           "sd_ratio_floor": a.sd_ratio_floor,
           "oracle": a.oracle, "oracle_amp_rel": a.oracle_amp,
           "oracle_token_sd": token_sd,
           "oracle_targets": [t for t, _ in ORACLE_TARGETS],
           "pc_local_tokens": list(PC_LOCAL_TOKENS),
           "pc_2obj_tokens": list(PC_2OBJ_TOKENS),
           "pc_2obj_cell_geometry": pc2,
           "randomise_features_seed": a.randomise_features,
           "proxy_v0": bool(a.proxy_v0),
           "n_train_windows_all": len(idx_tr),
           "n_eval_windows_all": len(idx_ev),
           "parity": "SELECTS NOTHING — banked window set read verbatim; "
                     f"cache meta parity={meta.get('parity', {}).get('train_parity','?')[:120]}",
           "device": str(dev), "arms": {}}

    ctr_all = np.array([sub[i]["clip_id"] for i in pos_tr])
    cev_all = np.array([sub[i]["clip_id"] for i in pos_ev])
    eev_all = np.array([sub[i]["episode_uid"] for i in pos_ev])
    v0_all = np.array([float(r["v0"]) for r in sub])

    # --- target values, computed ONCE ---------------------------------------
    tvals = {}
    for tname in a.targets:
        vv, ok = zip(*[LL.target_of(r, ego, tname, pose_off) for r in sub])
        tvals[tname] = (np.array(vv, dtype=np.float64), np.array(ok))

    preds: dict = {}
    scored: dict = {}
    for arm in a.arms:
        ta = time.time()
        if a.proxy_v0:
            feats = {a.proj_seeds[0]: v0_all.reshape(-1, 1)}
            n_units, n_raw = 1, 1
        elif arm == "cells":
            X = np.stack([r["cells"].numpy().reshape(-1).astype(np.float64)
                          for r in sub])
            feats = {a.proj_seeds[0]: X}
            n_units, n_raw = int(meta["n_cells"]), X.shape[1]
        else:
            feats, n_units = build_features(sub, arm, a.proj_seeds, th, tw,
                                            d_model, dev, a.oracle, amp,
                                            oracle_u, oracle_R,
                                            block_diag=a.block_diag_proj)
            n_raw = n_units * d_model
        if a.randomise_features is not None:
            g = np.random.default_rng(a.randomise_features)
            for s in list(feats):
                Xr = feats[s]
                mu_r, sd_r = Xr[pos_tr].mean(0), Xr[pos_tr].std(0)
                feats[s] = g.normal(mu_r, np.maximum(sd_r, 1e-12), Xr.shape)
        kh, kw = POOL_ARMS.get(arm, (0, 0))
        arec = {"pool_kernel": [kh, kw],
                "tokens_averaged_per_unit": kh * kw,
                "n_units": n_units, "n_raw_features": n_raw,
                "n_fit_features": int(next(iter(feats.values())).shape[1]) + 1,
                "projected": bool(arm in POOL_ARMS and not a.proxy_v0),
                "targets": {}}
        for tname in a.targets:
            y, ok = tvals[tname]
            mtr, mev = ok[pos_tr], ok[pos_ev]
            ytr, yev = y[pos_tr][mtr], y[pos_ev][mev]
            ctr, cev = ctr_all[mtr], cev_all[mev]
            eev = eev_all[mev]
            v0ev = None if tname == "ego_v0" else v0_all[pos_ev][mev]
            per_seed = {}
            for s in feats:
                X = feats[s]
                Xtr, Xev = X[pos_tr][mtr], X[pos_ev][mev]
                mu, sd = Xtr.mean(0), Xtr.std(0)
                sd = np.where(sd < 1e-12, 1.0, sd)
                Ztr = np.concatenate([(Xtr - mu) / sd,
                                      np.ones((Xtr.shape[0], 1))], 1)
                Zev = np.concatenate([(Xev - mu) / sd,
                                      np.ones((Xev.shape[0], 1))], 1)
                per_seed[str(s)], pr = fit_one(
                    Ztr, ytr, ctr, Zev, yev, eev, cev, a.alphas, a.inner_frac,
                    a.ridge_seed, a.n_boot, v0ev, a.sd_ratio_floor, icol)
                preds.setdefault(tname, {}).setdefault(arm, {})[str(s)] = pr
                scored[tname] = (yev, eev, v0ev)
            r2s = [per_seed[k]["r2_ceiling"] for k in per_seed]
            k1s = [per_seed[k]["K1_delta"] for k in per_seed]
            arec["targets"][tname] = {
                "unit": LL.UNITS[tname], "rung": LL.RUNG[tname],
                "n_train": int(mtr.sum()), "n_eval": int(mev.sum()),
                "n_eval_clusters": int(len(np.unique(eev))),
                "gt_mean": round(float(yev.mean()), 4),
                "gt_sd": round(float(yev.std()), 4),
                # ⭐ the PRIMARY quantity and its RP-seed spread = the
                # instrument's own noise floor at this arm.
                "r2_ceiling_mean": round(float(np.mean(r2s)), 5),
                "r2_ceiling_sd": round(float(np.std(r2s)), 5),
                "r2_ceiling_min": round(float(np.min(r2s)), 5),
                "r2_ceiling_max": round(float(np.max(r2s)), 5),
                "K1_delta_mean": round(float(np.mean(k1s)), 4),
                "K1_delta_range": round(float(np.max(k1s) - np.min(k1s)), 4),
                "n_seeds": len(per_seed),
                "per_seed": per_seed}
            p = per_seed[sorted(per_seed)[0]]
            print("  %-6s %-14s n=%4d/%2d r2c=%.5f+-%.5f K1=%+8.4f %-4s "
                  "r=%+.3f rpv0=%+.3f psd/gsd=%.3f%s"
                  % (arm, tname, int(mev.sum()), len(np.unique(eev)),
                     float(np.mean(r2s)), float(np.std(r2s)),
                     float(np.mean(k1s)),
                     "PASS" if p["K1_PASSES"] else "fail",
                     p["corr"], p.get("corr_partial_v0", float("nan")),
                     p["pred_sd_over_gt_sd"],
                     "  ⛔DEGENERATE" if p["K1_DEGENERATE"] else ""),
                  flush=True)
        arec["wall_s"] = round(time.time() - ta, 1)
        res["arms"][arm] = arec
        del feats

    # --- ⭐ THE PRE-REGISTERED CONTRAST: Δ against the DEPLOYED 40:1 arm ------
    # §7.1 commits to "the paired episode-cluster-bootstrap CI on Δr²(1:1−40:1)
    # excludes 0, AND that Δ survives partialling out v0". Both are computed
    # here, per projection seed, and the conservative read (ALL seeds separated)
    # is emitted so no single lucky seed can carry a verdict.
    ref = "p40"
    if ref in a.arms and not a.proxy_v0:
        res["deltas_vs_p40"] = {}
        for tname in a.targets:
            if tname not in preds or ref not in preds[tname]:
                continue
            yev, eev, v0ev = scored[tname]
            row = {}
            for arm in a.arms:
                if arm == ref or arm not in preds[tname]:
                    continue
                per_seed, per_seed_pv0, per_seed_mae = {}, {}, {}
                common = sorted(set(preds[tname][arm]) & set(preds[tname][ref]))
                for s in common:
                    pa, pb = preds[tname][arm][s], preds[tname][ref][s]
                    per_seed[s] = paired_delta_r2c(pa, pb, yev, eev, a.n_boot)
                    if v0ev is not None:
                        per_seed_pv0[s] = paired_delta_r2c(pa, pb, yev, eev,
                                                           a.n_boot, z=v0ev)
                    per_seed_mae[s] = paired_episode_cluster_bootstrap(
                        np.abs(pa - yev), np.abs(pb - yev), eev,
                        n_boot=a.n_boot)
                if not per_seed:
                    continue
                row[arm] = {
                    "delta_r2_ceiling_per_seed": per_seed,
                    "delta_r2_ceiling_partial_v0_per_seed": per_seed_pv0,
                    "delta_mae_per_seed": per_seed_mae,
                    "delta_r2c_mean": round(float(np.mean(
                        [v["delta"] for v in per_seed.values()])), 5),
                    "ALL_SEEDS_SEPARATED": bool(all(
                        v["separated"] for v in per_seed.values())),
                    "ALL_SEEDS_SEPARATED_AND_POSITIVE": bool(all(
                        v["separated"] and v["delta"] > 0
                        for v in per_seed.values())),
                    "ALL_SEEDS_SEPARATED_POSITIVE_PARTIAL_V0": (
                        bool(per_seed_pv0) and bool(all(
                            v["separated"] and v["delta"] > 0
                            for v in per_seed_pv0.values())))}
                print("  Δ%-4s-p40 %-14s r2c=%+.5f  allsep=%s  "
                      "allsep+v0partial=%s"
                      % (arm, tname, row[arm]["delta_r2c_mean"],
                         row[arm]["ALL_SEEDS_SEPARATED_AND_POSITIVE"],
                         row[arm]["ALL_SEEDS_SEPARATED_POSITIVE_PARTIAL_V0"]),
                      flush=True)
            res["deltas_vs_p40"][tname] = row

    # --- ⛔ J5/C94: the REPRODUCTION GATE against a banked ladder JSON --------
    # A fixture written by this file's own author proves only that this file
    # agrees with itself. This asserts against the PRODUCER's committed output.
    if a.gate_json:
        if not a.legacy_penalised_intercept:
            raise SystemExit("[er10] ⛔ --gate-json is only meaningful with "
                             "--legacy-penalised-intercept; the C92 repair "
                             "CHANGES the numbers on purpose.")
        want = json.loads(Path(a.gate_json).read_text("utf-8"))
        got = res["arms"]["cells"]["targets"]
        checks, bad = {}, []
        for tname, w in want["targets"].items():
            if tname not in got:
                continue
            wp = w["per_seed"][str(want["seeds"][0])]
            gp = got[tname]["per_seed"][sorted(got[tname]["per_seed"])[0]]
            for k, tol in (("err", 1e-4), ("c_const_err", 1e-4),
                           ("K1_delta", 5e-3), ("corr", 1e-3),
                           ("alpha_chosen", 0.0)):
                checks[f"{tname}.{k}"] = {"banked": wp[k], "ours": gp[k],
                                          "tol": tol}
                if abs(float(wp[k]) - float(gp[k])) > tol:
                    bad.append((tname, k, wp[k], gp[k]))
            checks[f"{tname}.n_eval"] = {"banked": w["n_eval"],
                                         "ours": got[tname]["n_eval"],
                                         "tol": 0.0}
            if int(w["n_eval"]) != int(got[tname]["n_eval"]):
                bad.append((tname, "n_eval", w["n_eval"], got[tname]["n_eval"]))
        res["reproduction_gate"] = {"against": str(a.gate_json),
                                    "checks": checks, "PASSED": not bad,
                                    "failures": bad}
        print(f"  [gate] banked reproduction: "
              f"{'PASS' if not bad else 'FAIL ' + repr(bad[:6])}", flush=True)

    res["wall_s"] = round(time.time() - t0, 1)
    if a.dump_preds:
        import pickle
        Path(a.dump_preds).parent.mkdir(parents=True, exist_ok=True)
        with open(a.dump_preds, "wb") as _fh:
            pickle.dump({"preds": preds, "scored": scored,
                         "label": a.label, "cache": str(a.cache),
                         "encloc_arm": meta.get("encloc_arm"),
                         "token_grid": [th, tw], "d_model": d_model,
                         "proj_seeds": list(a.proj_seeds)}, _fh)
        print(f"[encloc] dumped predictions -> {a.dump_preds}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), "utf-8")
    print(f"[er10] wrote {a.out}  {res['wall_s']} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
