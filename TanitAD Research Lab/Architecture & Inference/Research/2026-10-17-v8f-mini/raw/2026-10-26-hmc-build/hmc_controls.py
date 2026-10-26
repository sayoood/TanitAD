"""hmc_controls.py -- the tick-dump controls of PREREG_V8F_HACC_MC (C-1, C-3, the mechanism rows M1 / M2), shared by the reader and the smoke.  CPU only, read-only.

REUSE, NEVER RE-IMPLEMENT:
  * ``apply`` / ``inc`` (the along-ramp term's two blends) are AST-extracted from lcdiag's ``probe12_mc_trainfit.py`` snapshot;
  * ``replan_state`` / ``state_from_knots`` are AST-extracted from lcdiag's ``statekin.py`` snapshot and bound to THE TOOL TREE'S OWN ``hermite_frames`` /
    ``yaw_kin_from_track`` (statekin hard-codes C:/Users/Admin/t1k; here the tree is the one the runs used -- nothing else changes);
  * ``pair_err`` (M1 / M2's pair speeds and L2 shares) is AST-extracted from lcdiag's ``common.py`` snapshot.
"""
import ast
import importlib.util
import math
import os
import sys

import numpy as np

import hmc_lib as L

_ns = {"np": np, "torch": L.torch, "FF": L.FF, "DT": L.DT, "VMIN": L.VMIN, "math": math}


def _grab(src, names):
    for n in ast.parse(open(src, encoding="utf-8").read()).body:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            exec(compile(ast.Module([n], []), src, "exec"), _ns)


_grab(f"{L.REF}/lcdiag/probe12_mc_trainfit.py", ("apply", "inc"))
_grab(f"{L.REF}/lcdiag/common.py", ("pair_err",))
apply_w, inc_w, pair_err = _ns["apply"], _ns["inc"], _ns["pair_err"]


def bind_statekin(tree=None):
    """lcdiag's statekin functions, bound to ``tree``'s own tool (default READ_ROOT).  Returns (state_from_knots, st, T1, D)."""
    tree = tree or L.READ_ROOT
    for p in (f"{tree}/stack", f"{tree}/taniteval"):
        if p not in sys.path:
            sys.path.insert(0, p)
    from tanitad.data import v8f_mini_data as D
    spec = importlib.util.spec_from_file_location("t1tool_hmc", f"{tree}/stack/scripts/v8f_t1_eval.py")
    T1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T1)
    ns = {"np": np, "math": math, "T1": T1, "D": D}
    src = f"{L.REF}/lcdiag/statekin.py"
    for n in ast.parse(open(src, encoding="utf-8").read()).body:
        if isinstance(n, ast.FunctionDef) and n.name in ("replan_state", "state_from_knots"):
            exec(compile(ast.Module([n], []), src, "exec"), ns)
    sfk = ns["state_from_knots"]

    def st(A):                                                    # probe12's ``st``: kin3 = (v0, a0, kappa0) of a [30, 2] plan from the origin
        return sfk(np.vstack([np.zeros((1, 2)), A]), 0)[0]
    return sfk, st, T1, D


def load_ticks(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def table_from_ckpt(path):
    """The [30, 2] hacc_w buffer of a checkpoint FILE (float32 values, as float64) -- the table the loop actually ran."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["model"]["hacc_w"].double().numpy()


# ------------------------------------------------------------------------------------------------ C-1
def c1_reach(z, w, k_flat=12, bar=1e-6):
    """C-1 (reachability): on every tick, (plan - P)_along = w_along[k] * (tau - P)_along at knots 1..k_flat (for MC-H / MC-F w_along[k] is the constant c).  ``w`` is the float32 table the loop ran.
    PASS bar as written: max |d| <= 1e-6 m.  Reported beside: |d| in float32 ulps of the plan (the dumped ``plan`` is a float32 tensor, so |d| is bounded by ~half an ulp of |plan| -- 9.5e-7 m
    at 16-32 m, 1.9e-6 m at 32-64 m), and a float32 torch.lerp replica of the post-map (informative: it is NOT bit-reproducible, SMOKE max 2 ulp)."""
    import torch
    P, T, H = z["P"], z["tau"], z["plan"]
    wk = np.asarray(w, np.float64)[:k_flat, 0]
    d = (H[:, :k_flat, 0] - P[:, :k_flat, 0]) - wk[None] * (T[:, :k_flat, 0] - P[:, :k_flat, 0])
    ulp = np.spacing(np.abs(H[:, :k_flat, 0]).astype(np.float32)).astype(np.float64)
    i = np.unravel_index(int(np.abs(d).argmax()), d.shape)
    out = {"n_ticks": int(len(P)), "knots": k_flat, "max_abs_d_m": float(np.abs(d).max()), "bar_m": bar, "PASS": bool(np.abs(d).max() <= bar),
           "argmax_tick_knot": [int(i[0]), int(i[1] + 1)], "plan_along_at_argmax_m": float(H[i[0], i[1], 0]),
           "n_tick_knots_over_bar": int((np.abs(d) > bar).sum()), "frac_tick_knots_over_bar": float((np.abs(d) > bar).mean()),
           "max_d_over_ulp32_of_plan": float((np.abs(d) / ulp).max()), "max_abs_plan_along_m": float(np.abs(H[:, :k_flat, 0]).max())}
    # PREREG Amendment 2: the ulp at the blend's LARGEST OPERAND. Under cancellation (a near-stationary tick: P and tau of opposite sign, the result
    # smaller than both) the float32 roundings inside the blend are at the operands' scale, which a ulp at |plan| understates.
    M = np.maximum(np.maximum(np.abs(P[:, :k_flat, 0]), np.abs(T[:, :k_flat, 0])), np.abs(H[:, :k_flat, 0]))
    out["max_d_over_ulp32_of_max_operand"] = float((np.abs(d) / np.spacing(M.astype(np.float32)).astype(np.float64)).max())
    P32, T32 = torch.as_tensor(P.astype(np.float32)), torch.as_tensor(T.astype(np.float32))
    assert np.array_equal(P32.double().numpy(), P) and np.array_equal(T32.double().numpy(), T), "the dumped P / tau are not float32-exact"
    w32 = torch.as_tensor(np.asarray(w, np.float32))[None].expand_as(P32)
    rep = torch.lerp(P32, T32, w32).double().numpy()
    ulp_all = np.spacing(np.abs(H).astype(np.float32)).astype(np.float64)
    out["float32_lerp_replica_max_abs_vs_dumped_plan_m"] = float(np.abs(rep - H).max())              # INFORMATIVE ONLY: torch's CPU lerp is not bit-reproducible across buffer shapes
    out["float32_lerp_replica_max_over_ulp32_of_plan"] = float((np.abs(rep - H) / ulp_all).max())    # (the model blends a [B, 60] grid, the replica a [B, 30] one: SMOKE max = 2 ulp)
    return out


# ------------------------------------------------------------------------------------------------ C-3
def ramp_term(z, w, st, idx=None):
    """C-3's instrument, probe12's own: for each tick the replan STATE (v0, a0, kappa0) of H = P + w (tau - P) minus that of H with its ALONG axis re-blended in the increment domain
    (same w), both through lcdiag's statekin on the dumped P and tau.  Identically 0 for a w that is constant over every knot the state reads."""
    P, T = z["P"], z["tau"]
    idx = np.arange(len(P)) if idx is None else idx
    out = np.zeros((len(idx), 3))
    for r, i in enumerate(idx):
        Pi, Ti = P[i][None], T[i][None]
        hp = apply_w(Pi, Ti, w)[0]
        hi_ = hp.copy()
        hi_[:, 0] = inc_w(Pi, Ti, w)[0][:, 0]
        out[r] = st(hp) - st(hi_)
    return out


def c3_summary(R, bar=1e-9):
    a = np.abs(R)
    return {"n_ticks": int(len(R)), "max_abs_v0_mps": float(a[:, 0].max()), "max_abs_a0_mps2": float(a[:, 1].max()), "max_abs_kappa0_1pm": float(a[:, 2].max()),
            "median_a0_mps2": float(np.median(R[:, 1])), "frac_ticks_nonzero_a0": float((a[:, 1] > 1e-9).mean()), "bar": bar, "is_zero": bool(a.max() <= bar)}


def c3_train_known(st, s, wh):
    """HACC's TRAIN known value (probe12): median a0 ramp term on the TRAIN tick-0 plans with v0 >= 5.2 m/s, every 4th tick of {tick <= 1}, HACC's own table.  Bar: -0.1588 / -0.0726 +- 2e-3."""
    want = {0: -0.1588, 1: -0.0726}[s]
    z = np.load(f"C:/Users/Admin/hlat/dumps_train/ticks_HACC_s{s}_m10_train.npz")
    idx = np.nonzero(z["tick"] <= 1)[0][::4]
    idx = idx[(z["tick"][idx] == 0) & (z["v0"][idx] >= 5.2)]
    R = ramp_term({"P": z["P"], "tau": z["tau"]}, wh, st, idx)
    med = float(np.median(R[:, 1]))
    return {"n": int(len(idx)), "median_ramp_a0_mps2": med, "banked": want, "abs_diff": abs(med - want), "bar": 2e-3, "PASS": bool(abs(med - want) <= 2e-3 and len(idx) == 1747)}


# ------------------------------------------------------------------------------------------------ M1 / M2 (probe2's definitions, verbatim quantities)
def m_rows(X, F, G):
    """M1 = the arm-crawl fraction: GT-fast pairs (both tracks' pair valid, GT pair speed >= 5.2) whose OWN pair speed is < 5.2, pooled over all windows (probe2 ``b_frac_fast_pairs_arm_slow``;
    HACC 0.0536 / 0.0277, P10 0.0137).  M2 = the arm's L2 share on those pairs, as probe2's ``b_share_of_excess_on_pairs_where_arm_own_speed_lt_vmin``: per-window sum of the L2 share over the arm's
    crawl pairs minus P10's over P10's own crawl pairs, averaged over the windows with L2 defined for both arms (HACC +0.0202 / +0.0072)."""
    pa, pf = pair_err(X, G), pair_err(F, G)
    ok = (pa["c"] > 0) & (pf["c"] > 0)
    slow_a, slow_f = pa["fast"] & (pa["vX"] < L.VMIN), pf["fast"] & (pf["vX"] < L.VMIN)
    return {"M1": float(slow_a.sum() / pa["fast"].sum()), "M1_P10": float(slow_f.sum() / pf["fast"].sum()),
            "M2": float(np.where(slow_a[ok], pa["sh"][ok], 0).sum(1).mean() - np.where(slow_f[ok], pf["sh"][ok], 0).sum(1).mean()),
            "n_fast_pairs": int(pa["fast"].sum()), "n_arm_crawl_pairs": int(slow_a.sum()), "_slow_a": slow_a, "_fast_a": pa["fast"]}


def m1_diff_ci(slow_a, fast_a, slow_b, fast_b, eid, n_boot=2000, seed=0):
    """BESIDE (decides nothing): episode-cluster bootstrap of M1(a) - M1(b), a ratio of pooled pair counts, resampling episodes with the arms held fixed."""
    eids = np.unique(eid)
    sa, fa = np.array([slow_a[eid == e].sum() for e in eids]), np.array([fast_a[eid == e].sum() for e in eids])
    sb, fb = np.array([slow_b[eid == e].sum() for e in eids]), np.array([fast_b[eid == e].sum() for e in eids])
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        j = rng.integers(0, len(eids), len(eids))
        vals[b] = sa[j].sum() / max(fa[j].sum(), 1) - sb[j].sum() / max(fb[j].sum(), 1)
    pt = sa.sum() / fa.sum() - sb.sum() / fb.sum()
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"delta": float(pt), "lo": float(lo), "hi": float(hi), "separated": bool(lo > 0 or hi < 0), "estimator": "episode-cluster bootstrap of the pooled-pair-count ratio difference, B 2000, seed 0 (beside)"}
