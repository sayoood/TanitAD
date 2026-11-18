r"""headway_read.py -- HEADWAY (card PREREG_V8F_PROOF_ADD25, definitions HEADWAY_DEFS.md D1..D11).
Distance keeping in the T1-kin rollout (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance),
read post hoc from the banked dumps and the lead labels of the evaluation episodes.  CPU, zero training, no new rollout.  Evidence class: MEASURED once run.

  python headway_read.py controls --t1-root DIR[,DIR] [--arm HIER --seed 0]
  python headway_read.py read --t1-root DIR[,DIR] --pairs HIER:FLAT,G:HIER --seeds 0,1,2,3,4,5 --spread-arm HIER --spread-seeds 0,1,2,3,4,5 --out-dir DIR
  Q3: python headway_read.py read --t1-root ... --pairs LS:G,LSH:G,LSD:G --seeds 0,1 --spread-arm G --spread-seeds 0,1,2,3,4,5 --out-dir out_q3

Engine imported unchanged: spatial/q3/q3_read.py (cell_rule, find_dump) and, through it, hierrep_read.py (load_dump, episode_units, seed_arrays, read_cell).
Exit: 0 valid, 2 a control failed (no outcome named), 3 refused.
"""
import argparse
import itertools
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
Q3DIR = os.environ.get("HEADWAY_Q3DIR", "F:/TanitAD-scratch/v8proof/spatial/q3")
if Q3DIR not in sys.path:
    sys.path.insert(0, Q3DIR)
import numpy as np  # noqa: E402
import q3_read as Q  # noqa: E402

RH = Q.RH
TIER = "T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance)"
CARD = "PREREG_V8F_PROOF_ADD25.md"
RAW = "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/"
LABEL_JOIN = os.environ.get("HEADWAY_JOIN", RAW + "join_eval141.npz")
BANK = os.environ.get("HEADWAY_BANK", "F:/TanitAD-scratch/v8proof/res/cache/frames.npz")
DT = 0.2
K = 30
STEP_MAX_M = 3.0
GAP_MIN, GAP_MAX = 0.0, 60.0
MIN_LEAD_STEPS = 10
UNSAFE_GAP_M, UNSAFE_TAU_S, V_FLOOR = 2.0, 1.0, 0.5
F_BAR_M = 10.0
MIN_WINDOWS, MIN_EPISODES = 200, 20
H4_TOL_MPS = 0.05
RAY = 1000.0
CELLS = ("U", "C", "F", "bias_2.0", "bias_4.0", "bias_6.0", "abs_2.0", "abs_4.0", "abs_6.0", "tg_2.0", "tg_all")
CELL_TEXT = {"U": "U model-induced unsafe rate (primary)", "C": "C contact rate", "F": "F falls behind by more than 10 m at the last scored step",
             "bias_2.0": "gap bias 2.0 s (m, below 0 = closes in more than the driver)", "bias_4.0": "gap bias 4.0 s (m)", "bias_6.0": "gap bias 6.0 s (m)",
             "abs_2.0": "gap error 2.0 s (m)", "abs_4.0": "gap error 4.0 s (m)", "abs_6.0": "gap error 6.0 s (m)",
             "tg_2.0": "time-gap error 2.0 s (s)", "tg_all": "time-gap error over the scored steps (s)"}
FAMILY_LINE = ("LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; "
               "STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)")
LIMITS = ("The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. "
          "The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. "
          "One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.")
# pair -> (question, name when the first arm has the lower U, name when it has the higher U, name otherwise)
QUESTIONS = {("HIER", "FLAT"): ("Q1", "HIER-KEEPS-DISTANCE", "FLAT-KEEPS-DISTANCE", "DISTANCE-EQUAL"),
             ("G", "HIER"): ("Q2", "GRID-DISTANCE-BETTER", "GRID-DISTANCE-WORSE", "GRID-DISTANCE-EQUAL"),
             ("LS", "G"): ("Q3", "SUP-DISTANCE-BETTER", "SUP-DISTANCE-WORSE", "SUP-DISTANCE-EQUAL"),
             ("LSH", "G"): ("Q3", "SUP-DISTANCE-BETTER", "SUP-DISTANCE-WORSE", "SUP-DISTANCE-EQUAL"),
             ("LSD", "G"): ("Q3", "SUP-DISTANCE-BETTER", "SUP-DISTANCE-WORSE", "SUP-DISTANCE-EQUAL")}


# ============================================================================================ path geometry (D4)
def _path_frame(g):
    """g [N,30,2] -> vertices V [N,33,2] (back ray, P0..P30, forward ray), arc A [N,33], segment lengths L [N,32], unit directions U [N,32,2], non-degenerate nd [N,32]."""
    g = np.asarray(g, np.float64)
    n = len(g)
    P = np.concatenate([np.zeros((n, 1, 2)), g], axis=1)
    d = P[:, 1:] - P[:, :-1]
    ln = np.linalg.norm(d, axis=-1)
    nd = ln > 1e-9
    first = np.argmax(nd, axis=1)
    last = (d.shape[1] - 1) - np.argmax(nd[:, ::-1], axis=1)
    anyn = nd.any(1)
    ux = np.array([1.0, 0.0])
    ar = np.arange(n)
    u0 = np.where(anyn[:, None], d[ar, first] / np.maximum(ln[ar, first], 1e-300)[:, None], ux)
    u1 = np.where(anyn[:, None], d[ar, last] / np.maximum(ln[ar, last], 1e-300)[:, None], ux)
    V = np.concatenate([(P[:, 0] - RAY * u0)[:, None], P, (P[:, -1] + RAY * u1)[:, None]], axis=1)
    seg = V[:, 1:] - V[:, :-1]
    L = np.linalg.norm(seg, axis=-1)
    Lnd = L > 1e-9
    U = np.where(Lnd[..., None], seg / np.maximum(L, 1e-300)[..., None], 0.0)
    A = np.concatenate([np.full((n, 1), -RAY), np.zeros((n, 1)), np.cumsum(L[:, 1:-1], axis=1)], axis=1)       # arc of V[0] = -RAY, V[1] = P0 = 0, V[2+i] = P(i+1)
    A = np.concatenate([A, (A[:, -1] + L[:, -1])[:, None]], axis=1)
    return V, A, L, U, Lnd


def logged_arc(g):
    """a_l(k), k = 1..30: arc coordinate of the logged points (cumulative length of the logged path). [N,30]"""
    _, A, _, _, _ = _path_frame(g)
    return A[:, 2:32].copy()


def progress(g, pts):
    """a(q) for pts [N,M,2]: arc coordinate of the orthogonal projection on the nearest non-degenerate segment of the extended logged path. [N,M]"""
    V, A, L, U, nd = _path_frame(g)
    pts = np.asarray(pts, np.float64)
    a = V[:, None, :-1, :]                                   # [N,1,S,2]
    ab = (V[:, 1:] - V[:, :-1])[:, None]
    q = pts[:, :, None, :]
    l2 = (L ** 2)[:, None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.clip(((q - a) * ab).sum(-1) / np.where(l2 > 0, l2, 1.0), 0.0, 1.0)
    proj = a + t[..., None] * ab
    dist = np.linalg.norm(q - proj, axis=-1)
    dist = np.where(nd[:, None, :], dist, np.inf)
    j = np.argmin(dist, axis=-1)                              # [N,M]
    jj = j[..., None]
    arc = np.take_along_axis(np.broadcast_to(A[:, None, :-1], dist.shape), jj, -1)[..., 0]
    tt = np.take_along_axis(t, jj, -1)[..., 0]
    ll = np.take_along_axis(np.broadcast_to(L[:, None, :], dist.shape), jj, -1)[..., 0]
    return arc + tt * ll


def point_at_arc(g, s):
    """The point of the extended logged path at arc coordinate s [N,M] (inverse of `progress` on the path)."""
    V, A, L, U, nd = _path_frame(g)
    s = np.asarray(s, np.float64)
    As = A[:, None, :-1]
    cond = (As <= s[..., None]) & nd[:, None, :]
    S = cond.shape[-1]
    j = (S - 1) - np.argmax(cond[..., ::-1], axis=-1)
    jj = j[..., None]
    Vj = np.take_along_axis(np.broadcast_to(V[:, None, :-1, :], cond.shape + (2,)), jj[..., None], -2)[..., 0, :]
    Uj = np.take_along_axis(np.broadcast_to(U[:, None, :, :], cond.shape + (2,)), jj[..., None], -2)[..., 0, :]
    Aj = np.take_along_axis(np.broadcast_to(A[:, None, :-1], cond.shape), jj, -1)[..., 0]
    return Vj + (s - Aj)[..., None] * Uj


def advance_along_path(g, mps):
    """The logged trajectory advanced along the path by mps times the elapsed time (0.2 s per step): the planted regression of control h2."""
    a_l = logged_arc(g)
    s = a_l + float(mps) * DT * np.arange(1, K + 1)[None, :]
    return point_at_arc(g, s)


def step_speeds(traj):
    p = np.concatenate([np.zeros_like(np.asarray(traj, np.float64)[:, :1]), np.asarray(traj, np.float64)], axis=1)
    return np.linalg.norm(p[:, 1:] - p[:, :-1], axis=-1) / DT


# ============================================================================================ lead track (D3)
def valid_frames(lead_valid, gap):
    gap = np.asarray(gap, np.float64)
    with np.errstate(invalid="ignore"):
        return (np.asarray(lead_valid) == 1) & np.isfinite(gap) & (gap >= GAP_MIN) & (gap <= GAP_MAX)


def build_track(valid, gap, step_max=STEP_MAX_M, min_steps=MIN_LEAD_STEPS):
    """valid, gap: [N,31] (frames k = 0..30).  -> G [N,31] (gap on the track frames, NaN after the track ends), n_s [N] (scored steps, 0 if frame 0 not valid),
    lead [N] (valid at frame 0 and frames 1..min_steps in the track).  A frame ends the track when it is invalid or its step from the previous frame exceeds step_max."""
    gap = np.asarray(gap, np.float64)
    ok = np.asarray(valid, bool).copy()
    with np.errstate(invalid="ignore"):
        step = np.abs(np.diff(gap, axis=1))
    ok[:, 1:] &= (step <= step_max)
    track = np.logical_and.accumulate(ok, axis=1)
    ns = np.where(track[:, 0], track.sum(1) - 1, 0)
    lead = track[:, 0] & track[:, 1:min_steps + 1].all(1)
    return np.where(track, gap, np.nan), ns.astype(np.int64), lead


def load_join(path=LABEL_JOIN):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in ("clip", "off", "n", "lead_valid", "lead_gap")}


def label_windows(J, ci, ws):
    """D2 frame map: window frame S0 + k = join row off[ci] + ws + k.  -> dict G [N,31], ns, lead."""
    ci, ws = np.asarray(ci, np.int64), np.asarray(ws, np.int64)
    rows = J["off"][ci][:, None] + ws[:, None] + np.arange(K + 1)[None, :]
    inclip = (ws[:, None] + np.arange(K + 1)[None, :]) < J["n"][ci][:, None]
    gap = np.where(inclip, J["lead_gap"][rows].astype(np.float64), np.nan)
    val = valid_frames(np.where(inclip, J["lead_valid"][rows], 0), gap)
    G, ns, lead = build_track(val, gap)
    return {"G": G, "ns": ns, "lead": lead}


# ============================================================================================ cells (D5, D6)
def window_cells(g, cl, L):
    """g, cl [N,30,2]; L = label_windows output.  -> dict of per-window arrays (NaN where a cell is not defined) plus 'gm' [N,30] and 'logged' rates."""
    g, cl = np.asarray(g, np.float64), np.asarray(cl, np.float64)
    G, ns, lead = L["G"], L["ns"], L["lead"]
    n = len(g)
    kk = np.arange(1, K + 1)[None, :]
    M = lead[:, None] & (kk <= ns[:, None])
    gl = np.where(M, G[:, 1:], np.nan)
    a_l = logged_arc(g)
    a_m = progress(g, cl)
    gm = gl + a_l - a_m
    vm, vl = step_speeds(cl), step_speeds(g)
    tm = gm / np.maximum(vm, V_FLOOR)
    tl = gl / np.maximum(vl, V_FLOOR)
    with np.errstate(invalid="ignore"):
        un_m = M & ((gm < UNSAFE_GAP_M) | (tm < UNSAFE_TAU_S))
        un_l = M & ((gl < UNSAFE_GAP_M) | (tl < UNSAFE_TAU_S))
        ct_m = M & (gm <= 0.0)
        ct_l = M & (gl <= 0.0)
    nan = np.full(n, np.nan)
    ud = lead & ~un_l.any(1)
    out = {"U": np.where(ud, un_m.any(1).astype(np.float64), nan),
           "C": np.where(lead, ct_m.any(1).astype(np.float64), nan)}
    last = np.clip(ns, 1, K) - 1
    ar = np.arange(n)
    diff = gm - gl
    out["F"] = np.where(lead, (diff[ar, last] > F_BAR_M).astype(np.float64), nan)
    for sec, k in (("2.0", 10), ("4.0", 20), ("6.0", 30)):
        ok = lead & (ns >= k)
        out["bias_" + sec] = np.where(ok, diff[:, k - 1], nan)
        out["abs_" + sec] = np.where(ok, np.abs(diff[:, k - 1]), nan)
    out["tg_2.0"] = np.where(lead & (ns >= 10), np.abs(tm[:, 9] - tl[:, 9]), nan)
    d = np.where(M, np.abs(tm - tl), 0.0)
    cnt = M.sum(1)
    out["tg_all"] = np.where(lead & (cnt > 0), d.sum(1) / np.maximum(cnt, 1), nan)
    out["_gm"] = gm
    out["_gl"] = gl
    out["_logged_unsafe"] = np.where(lead, un_l.any(1).astype(np.float64), nan)
    out["_logged_contact"] = np.where(lead, ct_l.any(1).astype(np.float64), nan)
    return out


def level(x):
    x = np.asarray(x, np.float64)
    f = np.isfinite(x)
    return float(x[f].mean()) if f.any() else float("nan")


def name_outcome(pair, verdict, final, blocked=False, enough=True):
    """Outcome naming (D8).  pair = (A, B).  -> dict(question, name, final, note)."""
    if pair not in QUESTIONS:
        return {"question": None, "name": "NOT NAMED (no question of the card for this pair)", "final": False}
    q, better, worse, equal = QUESTIONS[pair]
    if blocked:
        return {"question": q, "name": "NOT NAMED (a control failed)", "final": False}
    if not enough:
        return {"question": q, "name": "NOT NAMED (fewer than %d lead windows or %d episodes)" % (MIN_WINDOWS, MIN_EPISODES), "final": False}
    nm = better if verdict == "BETTER" else worse if verdict == "WORSE" else equal
    return {"question": q, "name_by_rule": nm, "name": nm if final else "PRELIMINARY (" + nm + ")", "final": bool(final)}


# ============================================================================================ spread and pair cells
def spread_of(levels_by_seed):
    """levels_by_seed: list of arm-level cell means over the spread seeds.  -> dict(max, median, n_pairs, levels)."""
    diffs = [abs(x - y) for x, y in itertools.combinations(levels_by_seed, 2)]
    return {"max": float(max(diffs)) if diffs else float("nan"), "median": float(np.median(diffs)) if diffs else float("nan"),
            "n_pairs": len(diffs), "levels": [float(x) for x in levels_by_seed]}


def pair_cells(W, inv, n_ep, A, B, seeds, n_boot, label_n, spreads, same_sign=True):
    """W[(arm, seed)] -> window cells.  Cells of the paired window difference A minus B with the two-level interval and the rule."""
    out = {}
    ones = np.ones(len(inv), bool)
    for cell in CELLS:
        sE, cE = RH.seed_arrays([W[(A, s)][cell] for s in seeds], [W[(B, s)][cell] for s in seeds], ones, inv, n_ep)
        c = RH.read_cell(sE, cE, n_boot, label_n)
        sp = spreads.get(cell, {})
        c["spread_max"], c["spread_median"] = sp.get("max", float("nan")), sp.get("median", float("nan"))
        c["verdict"], c["outside_spread"] = Q.cell_rule(c, c["spread_max"], same_sign)
        out[cell] = c
    return out


def arm_level_cells(W, inv, n_ep, arm, seeds, n_boot):
    """Level of an arm: mean over seeds of the window mean, interval by the same engine against zero."""
    out = {}
    ones = np.ones(len(inv), bool)
    for cell in CELLS:
        xs = [W[(arm, s)][cell] for s in seeds]
        sE, cE = RH.seed_arrays(xs, [np.zeros(len(x)) for x in xs], ones, inv, n_ep)
        c = RH.read_cell(sE, cE, n_boot, len(seeds))
        out[cell] = {"per_seed": c["d_s"], "mean": c["D"], "lo": c["A_lo"], "hi": c["A_hi"], "n_windows_per_seed": c["n_windows_per_seed"]}
    return out


# ============================================================================================ controls (D9)
def check_h1(g, L, tol=1e-6):
    """h1 identity: the logged trajectory given as the model reads gap_m = gap_l (<= tol), U 0, gap bias 0."""
    c = window_cells(g, g, L)
    m = np.isfinite(c["_gl"])
    dev = float(np.nanmax(np.abs(c["_gm"] - c["_gl"]))) if m.any() else float("nan")
    u = c["U"][np.isfinite(c["U"])]
    bias = [float(np.nanmax(np.abs(c["bias_" + s]))) if np.isfinite(c["bias_" + s]).any() else float("nan") for s in ("2.0", "4.0", "6.0")]
    ok = bool(m.any() and dev <= tol and len(u) > 0 and np.all(u == 0.0) and all(np.isfinite(b) and b <= tol for b in bias))
    return {"PASS": ok, "max_abs_gap_diff_m": dev, "n_defined_U": int(len(u)), "U_mean": float(u.mean()) if len(u) else float("nan"), "max_abs_bias_m": bias}


def check_h2(g, L, tol=1e-3, mps=0.5, rule_check=None):
    """h2 planted regression: advance along the path by mps * elapsed time -> gap bias -1.0 m at 2.0 s and -3.0 m at 6.0 s (scaled by mps / 0.5).  rule_check(planted U) is evaluated separately."""
    pl = advance_along_path(g, mps)
    c = window_cells(g, pl, L)
    b2, b6 = c["bias_2.0"], c["bias_6.0"]
    e2 = float(np.nanmax(np.abs(b2 - (-2.0 * mps)))) if np.isfinite(b2).any() else float("nan")
    e6 = float(np.nanmax(np.abs(b6 - (-6.0 * mps)))) if np.isfinite(b6).any() else float("nan")
    ok = bool(np.isfinite(e2) and np.isfinite(e6) and e2 <= tol and e6 <= tol)
    return {"PASS": ok, "mps": mps, "max_dev_bias_2s_from_-1.0": e2 if mps == 0.5 else e2, "max_dev_bias_6s_from_-3.0": e6,
            "mean_bias_2s": level(b2), "mean_bias_6s": level(b6), "n_2s": int(np.isfinite(b2).sum()), "n_6s": int(np.isfinite(b6).sum())}, c


def check_h3(f_self, f_fw, f_sw, tol=1e-12):
    mx = max(abs(c["D"]) for c in f_self.values())
    dev = max(abs(f_sw[k]["D"] + f_fw[k]["D"]) for k in f_fw)
    return {"PASS": bool(mx == 0.0 and dev < tol), "max_abs_self_D": float(mx), "max_abs_swap_dev": float(dev)}


def check_h4(v0_dump, v_label, tol=H4_TOL_MPS):
    """v0_dump [n], v_label [n] (NaN where the label-side row is missing).  A window that cannot be checked counts as failed."""
    v0_dump, v_label = np.asarray(v0_dump, np.float64), np.asarray(v_label, np.float64)
    miss = int((~np.isfinite(v_label)).sum())
    d = np.abs(v0_dump - v_label)
    mx = float(np.nanmax(d)) if np.isfinite(d).any() else float("nan")
    n_bad = int((d > tol).sum())
    return {"PASS": bool(miss == 0 and n_bad == 0 and np.isfinite(mx)), "n_checked": int(np.isfinite(d).sum()), "n_unchecked": miss, "n_over_tol": n_bad, "max_abs_diff_mps": mx, "tol_mps": tol}


def check_h5(L, eid):
    lead = L["lead"]
    nw = int(lead.sum())
    ne = int(len(np.unique(np.asarray(eid)[lead]))) if nw else 0
    return {"PASS": True, "n_lead_windows": nw, "n_episodes_with_lead": ne, "enough": bool(nw >= MIN_WINDOWS and ne >= MIN_EPISODES),
            "min_windows": MIN_WINDOWS, "min_episodes": MIN_EPISODES}


def check_h6(dumps, keys=("g", "ws", "eid", "clip_index", "v0")):
    ref = dumps[0]
    bad = [i for i, d in enumerate(dumps) for k in keys if not np.array_equal(d[k], ref[k])]
    return {"PASS": not bad, "n_dumps": len(dumps), "n_differing": len(bad)}


def bank_speeds(path=BANK):
    with np.errstate(all="ignore"):
        Z = np.load(path)
        return {(str(e), int(g)): float(v) for e, g, v in zip(Z["beid"], Z["bg"], Z["E"][:, 0])}


def label_speed(bank, J, ci, ws, shift=0):
    return np.array([bank.get((str(J["clip"][c]), int(w) + shift), np.nan) for c, w in zip(ci, ws)], np.float64)


def label_crosscheck(J, bank_path=BANK):
    """J against the bank on their common rows: presence mismatches and the largest gap difference (printed, decides nothing)."""
    Z = np.load(bank_path)
    ci = {str(c): i for i, c in enumerate(J["clip"])}
    nb = nboth = 0
    mx = 0.0
    for e, g, v in zip(Z["beid"], Z["bg"], Z["lead_gap"]):
        j = J["off"][ci[str(e)]] + int(g)
        a = float(J["lead_gap"][j])
        if np.isfinite(v) != np.isfinite(a):
            nb += 1
        elif np.isfinite(v):
            nboth += 1
            mx = max(mx, abs(float(v) - a))
    return {"bank_rows": int(len(Z["beid"])), "presence_mismatch_rows": nb, "rows_finite_in_both": nboth, "max_abs_gap_diff_m": mx}


def run_controls(ref, L, J, bank, W_by, inv, n_ep, spread_levels, seeds, n_boot_control, pairs_for_h3, seeds_by_pair, spreads_for_h3):
    """All six controls, each with its regression form.  ref = reference dump dict; L = labels of its windows."""
    res = {}
    g = ref["g"]
    res["h5_counts"] = check_h5(L, ref["eid"])
    lead = L["lead"]
    # h4: alignment on the lead windows; the shifted mapping must FAIL
    ci, ws, v0 = ref["clip_index"][lead], ref["ws"][lead], ref["v0"][lead]
    h4 = check_h4(v0, label_speed(bank, J, ci, ws, 0))
    h4s = check_h4(v0, label_speed(bank, J, ci, ws, 5))
    h4["shift5_check"] = {"PASS_expected": False, "n_over_tol": h4s["n_over_tol"], "max_abs_diff_mps": h4s["max_abs_diff_mps"], "n_unchecked": h4s["n_unchecked"],
                          "check_fails_as_required": bool((h4s["n_over_tol"] > 0))}
    h4["PASS"] = bool(h4["PASS"] and h4["shift5_check"]["check_fails_as_required"])
    res["h4_alignment"] = h4
    # h6: same windows
    dumps = W_by["_dumps"]
    h6 = check_h6(dumps)
    d2 = dict(dumps[0])
    d2["ws"] = d2["ws"].copy()
    d2["ws"][0] += 1
    reg = check_h6([dumps[0], d2])
    h6["regression_form_fails"] = bool(not reg["PASS"])
    h6["PASS"] = bool(h6["PASS"] and h6["regression_form_fails"])
    res["h6_same_windows"] = h6
    # h1
    h1 = check_h1(g, L)
    adv = advance_along_path(g, 0.1)
    reg1 = window_cells(g, adv, L)
    reg_dev = float(np.nanmax(np.abs(reg1["_gm"] - reg1["_gl"]))) if np.isfinite(reg1["_gl"]).any() else float("nan")
    h1["regression_form_advance_0.1mps_max_abs_gap_diff_m"] = reg_dev
    h1["regression_form_fails"] = bool(reg_dev > 1e-6)
    h1["PASS"] = bool(h1["PASS"] and h1["regression_form_fails"])
    res["h1_identity"] = h1
    # h2: planted advance 0.5 m/s, then the rule machinery on the pair (planted, logged)
    h2, cp = check_h2(g, L)
    cl_ident = window_cells(g, g, L)
    ones = np.ones(len(inv), bool)
    Wp = {("PL", s): cp for s in seeds}
    Wp.update({("LG", s): cl_ident for s in seeds})
    sE, cE = RH.seed_arrays([Wp[("PL", s)]["U"] for s in seeds], [Wp[("LG", s)]["U"] for s in seeds], ones, inv, n_ep)
    c = RH.read_cell(sE, cE, n_boot_control, len(seeds))
    c["spread_max"] = spread_levels["max"]
    c["verdict"], c["outside_spread"] = Q.cell_rule(c, c["spread_max"], True)
    h2["planted_vs_logged_U"] = {"D": c["D"], "A_lo": c["A_lo"], "A_hi": c["A_hi"], "spread_max": c["spread_max"], "verdict": c["verdict"]}
    h2["named_worse_on_U"] = bool(c["verdict"] == "WORSE")
    zero_bias = float(np.nanmax(np.abs(window_cells(g, advance_along_path(g, 0.0), L)["bias_2.0"] - (-1.0))))
    h2["regression_form_zero_advance_dev_from_-1.0"] = zero_bias
    h2["regression_form_fails"] = bool(zero_bias > 1e-3)
    h2["PASS"] = bool(h2["PASS"] and h2["named_worse_on_U"] and h2["regression_form_fails"])
    res["h2_planted_regression"] = h2
    # h3: self and swap on every real pair with data, and the regression form
    h3s = []
    for (A, B) in pairs_for_h3:
        ps = [s for s in seeds_by_pair[(A, B)]]
        f_fw = pair_cells(W_by, inv, n_ep, A, B, ps, n_boot_control, len(ps), spreads_for_h3)
        f_self = pair_cells(W_by, inv, n_ep, A, A, ps, n_boot_control, len(ps), spreads_for_h3)
        f_sw = pair_cells(W_by, inv, n_ep, B, A, ps, n_boot_control, len(ps), spreads_for_h3)
        x = check_h3(f_self, f_fw, f_sw)
        bad = {k: dict(v) for k, v in f_sw.items()}
        bad["U"]["D"] += 1e-6
        x["regression_form_fails"] = bool(not check_h3(f_self, f_fw, bad)["PASS"])
        x["pair"] = "%s:%s" % (A, B)
        x["seeds"] = ps
        x["PASS"] = bool(x["PASS"] and x["regression_form_fails"])
        h3s.append(x)
    h3 = {"PASS": bool(h3s) and all(x["PASS"] for x in h3s), "pairs": h3s}
    res["h3_swap_self"] = h3
    res["ALL_PASS"] = bool(all(v["PASS"] for v in res.values() if isinstance(v, dict) and "PASS" in v))
    return res


# ============================================================================================ the read
def load_world(roots, need, expect_files):
    W_dump, missing, where = {}, [], {}
    for arm, s in dict.fromkeys(need):
        d = Q.find_dump(roots, arm, s)
        if d is None:
            missing.append("%s_s%d" % (arm, s))
            continue
        try:
            W_dump[(arm, s)] = RH.load_dump(d, expect_files)
        except ValueError as e:
            missing.append("%s_s%d (%s)" % (arm, s, e))
            continue
        where[(arm, s)] = d
    return W_dump, missing, where


def cmd_read(a, controls_only=False):
    t0 = time.time()
    roots = [r for r in a.t1_root.split(",") if r]
    seeds = [int(x) for x in a.seeds.split(",")]
    pairs = [tuple(p.split(":")) for p in a.pairs.split(",")]
    sp_req = [int(x) for x in a.spread_seeds.split(",")]
    need = [(arm, s) for A, B in pairs for s in seeds for arm in (A, B)] + [(a.spread_arm, s) for s in sp_req]
    Wd, missing, where = load_world(roots, need, a.expect_files)
    if not Wd:
        print("REFUSED: no dump found; missing %s" % missing)
        return 3
    dumps = list(Wd.values())
    ref = dumps[0]
    h6 = check_h6(dumps)
    if not h6["PASS"]:
        print("REFUSED: windows differ between dumps (h6): %s" % h6)
        return 3
    J = load_join()
    L = label_windows(J, ref["clip_index"], ref["ws"])
    uniq, inv = RH.episode_units(ref["eid"])
    n_ep = len(uniq)
    bank = bank_speeds()
    W = {k: window_cells(d["g"], d["cl"], L) for k, d in Wd.items()}
    W["_dumps"] = dumps
    sp_seeds = [s for s in sp_req if (a.spread_arm, s) in W]
    spreads = {cell: spread_of([level(W[(a.spread_arm, s)][cell]) for s in sp_seeds]) for cell in CELLS}
    seeds_by_pair = {(A, B): [s for s in seeds if (A, s) in W and (B, s) in W] for A, B in pairs}
    pairs_ok = [p for p in pairs if seeds_by_pair[p]]
    first = pairs_ok[0] if pairs_ok else None
    res = {"card": CARD, "tier": TIER, "evidence_class": "MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join')",
           "STRATEGIC": "NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)", "family_line": FAMILY_LINE, "limits": LIMITS, "label_join": LABEL_JOIN, "bank_for_h4": BANK,
           "estimator": "paired window differences A minus B, two-level bootstrap seeds then episodes (B %d, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed" % a.n_boot,
           "missing": missing, "dumps": {"%s_s%d" % k: v for k, v in sorted(where.items())}, "pairs": {}, "levels": {}, "controls": {}}
    res["counts"] = check_h5(L, ref["eid"])
    res["counts"]["n_windows_total"] = int(len(ref["g"]))
    res["counts"]["n_episodes_total"] = int(n_ep)
    res["label_crosscheck_join_vs_bank"] = label_crosscheck(J)
    if first is None:
        res["controls"] = {"ALL_PASS": False, "note": "no pair with data"}
        res["outcomes_named"] = "none: no pair has both arms on disk (missing: %s)" % ", ".join(missing)
        _write(res, a.out_dir)
        print("no pair with data; missing %s" % missing)
        return 2
    ctrl = run_controls(ref, L, J, bank, W, inv, n_ep, spreads["U"], [s for s in seeds if (first[0], s) in W and (first[1], s) in W], a.n_boot_control, pairs_ok, seeds_by_pair, spreads)
    res["controls"] = ctrl
    res["spread"] = {"arm": a.spread_arm, "seeds_present": sp_seeds, "seeds_requested": sp_req, "complete_six": bool(len(sp_seeds) >= 6)} if ctrl["ALL_PASS"] else \
        {"arm": a.spread_arm, "seeds_present": sp_seeds, "note": "cells not written: a control failed"}
    blocked = not ctrl["ALL_PASS"]
    enough = bool(res["counts"]["enough"])
    if blocked or controls_only:
        res["outcomes_named"] = "none: " + ("a control failed" if blocked else "controls-only run")
        _write(res, a.out_dir)
        print("controls %s | counts %s | missing %s" % ("PASS" if ctrl["ALL_PASS"] else "FAILED", res["counts"], missing))
        return 0 if ctrl["ALL_PASS"] else 2
    res["spread"]["cells"] = spreads
    for A, B in pairs:
        ps = [s for s in seeds if (A, s) in W and (B, s) in W]
        tag = "%s_minus_%s" % (A, B)
        if not ps:
            res["pairs"][tag] = {"seeds_read": [], "note": "no seed with both arms"}
            continue
        cells = pair_cells(W, inv, n_ep, A, B, ps, a.n_boot, len(seeds), spreads, True)
        final = (len(ps) == len(seeds)) and bool(len(sp_seeds) >= 6)
        out = {"seeds_read": ps, "seeds_requested": seeds, "cells": cells,
               "outcome": name_outcome((A, B), cells["U"]["verdict"], final, False, enough)}
        # sensitivity: spread of the second arm of the pair, if present for six seeds, beside and naming nothing
        if B != a.spread_arm and all((B, s) in W for s in sp_req) and len(sp_req) >= 6:
            spB = spread_of([level(W[(B, s)]["U"]) for s in sp_req])
            vB, _ = Q.cell_rule(cells["U"], spB["max"], True)
            out["U_with_spread_of_second_arm"] = {"arm": B, "spread_max": spB["max"], "verdict": vB}
        res["pairs"][tag] = out
    arms = sorted({k[0] for k in W if k != "_dumps" and isinstance(k, tuple)})
    for arm in arms:
        ss = [s for s in seeds + sp_req if (arm, s) in W]
        ss = sorted(set(ss))
        res["levels"][arm] = {"seeds": ss, "cells": arm_level_cells(W, inv, n_ep, arm, ss, a.n_boot)}
    lg_u, lg_c = level(W[next(k for k in W if k != "_dumps")]["_logged_unsafe"]), level(W[next(k for k in W if k != "_dumps")]["_logged_contact"])
    res["logged_driver"] = {"unsafe_rate_unrestricted": lg_u, "contact_rate_unrestricted": lg_c,
                            "U_by_construction": 0.0, "bias": 0.0, "n_lead_windows": res["counts"]["n_lead_windows"]}
    res["elapsed_s"] = round(time.time() - t0, 1)
    _write(res, a.out_dir)
    for tag, p in res["pairs"].items():
        if p.get("outcome"):
            print("%s: %s %s (seeds %s)" % (tag, p["outcome"]["question"], p["outcome"]["name"], p["seeds_read"]))
    print("controls PASS | counts %s | spread seeds %s | missing %s" % (res["counts"], sp_seeds, missing))
    return 0


def _write(res, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "RESULT_HEADWAY.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(os.path.join(out_dir, "VERDICT_HEADWAY.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(verdict(res))


def _f(x, nd=4, sign=True):
    if x is None or not np.isfinite(x):
        return "nan"
    return (("%+." if sign else "%.") + str(nd) + "f") % x


def verdict(res):
    L = ["# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)", "",
         "Card: %s. Tier: %s. Evidence class: %s." % (res["card"], res["tier"], res["evidence_class"]),
         "Estimator: %s." % res["estimator"], "Label source: %s. Alignment reference (ego speed): %s." % (res["label_join"], res["bank_for_h4"]), "",
         "## Controls (first)", ""]
    ct = res["controls"]
    for k, v in ct.items():
        if isinstance(v, dict) and "PASS" in v:
            L.append("* %s: %s  %s" % (k, "PASS" if v["PASS"] else "FAIL", json.dumps({x: y for x, y in v.items() if x != "PASS"}, default=str)))
    L.append("* ALL_PASS: %s" % ct.get("ALL_PASS"))
    c = res["counts"]
    L += ["", "## Counts (h5, before any comparison)", "",
          "Lead windows: %d of %d windows; episodes with at least one lead window: %d of %d. Bars of the card: at least %d windows and %d episodes; enough: %s." % (
              c["n_lead_windows"], c["n_windows_total"], c["n_episodes_with_lead"], c["n_episodes_total"], c["min_windows"], c["min_episodes"], c["enough"]),
          "Join against the bank on their common rows (decides nothing): %s." % json.dumps(res["label_crosscheck_join_vs_bank"])]
    if res.get("outcomes_named"):
        L += ["", "No outcome is named (%s). No arm table is written." % res["outcomes_named"], "", "Four-family line: " + res["family_line"], "", "Limits: " + res["limits"], ""]
        return "\n".join(L)
    sp = res["spread"]
    L += ["", "## Spread of the baseline", "", "Arm %s, seeds present %s of requested %s; complete for the card (six seeds): %s." % (sp["arm"], sp["seeds_present"], sp["seeds_requested"], sp["complete_six"])]
    if res["missing"]:
        L.append("Dumps not present: " + ", ".join(res["missing"]) + ".")
    L += ["", "## Names of section 3 of the card", ""]
    for tag, p in res["pairs"].items():
        if not p.get("seeds_read"):
            L.append("* %s: no seed with both arms; nothing named." % tag)
            continue
        o = p["outcome"]
        extra = ""
        if p.get("U_with_spread_of_second_arm"):
            e = p["U_with_spread_of_second_arm"]
            extra = " Sensitivity (names nothing): with the spread of %s (max %s) the rule on U reads %s." % (e["arm"], _f(e["spread_max"], 4, False), e["verdict"])
        L.append("* %s, cell U, %s over seeds %s of %s: **%s**%s%s" % (o.get("question"), tag.replace("_minus_", " minus "), p["seeds_read"], p["seeds_requested"], o["name"],
                 "" if o.get("final") else " (decides nothing)", extra))
    for tag, p in res["pairs"].items():
        if not p.get("seeds_read"):
            continue
        L += ["", "## " + tag.replace("_", " "), "", "Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.", "",
              "| cell | per seed | D | two-level interval | spread max (median) | rule |", "|---|---|---|---|---|---|"]
        for cell in CELLS:
            x = p["cells"][cell]
            L.append("| %s | %s | %s | [%s, %s] | %s (%s) | %s |" % (CELL_TEXT[cell], " / ".join(_f(v) for v in x["d_s"]), _f(x["D"]), _f(x["A_lo"]), _f(x["A_hi"]),
                     _f(x["spread_max"], 4, False), _f(x["spread_median"], 4, False), x["verdict"]))
    L += ["", "## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)", "",
          "| arm (seeds) | " + " | ".join(CELLS) + " |", "|---|" + "---|" * len(CELLS)]
    lg = res["logged_driver"]
    for arm, lv in res["levels"].items():
        row = []
        for cell in CELLS:
            x = lv["cells"][cell]
            row.append("%s [%s, %s]" % (_f(x["mean"], 4, False), _f(x["lo"], 4, False), _f(x["hi"], 4, False)))
        L.append("| %s (%s) | %s |" % (arm, ",".join(str(s) for s in lv["seeds"]), " | ".join(row)))
    L.append("| logged driver | U 0 by construction (windows where it has no unsafe step) | contact %s | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |" % _f(lg["contact_rate_unrestricted"], 4, False))
    L += ["", "Logged driver, unrestricted over the %d lead windows (same rule): unsafe-step share %s, contact share %s." % (lg["n_lead_windows"], _f(lg["unsafe_rate_unrestricted"], 4, False), _f(lg["contact_rate_unrestricted"], 4, False)),
          "", "Four-family line: " + res["family_line"], "", "Limits: " + res["limits"], ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("read", "controls"):
        r = sub.add_parser(name)
        r.add_argument("--t1-root", required=True)
        r.add_argument("--pairs", default="HIER:FLAT,G:HIER")
        r.add_argument("--seeds", default="0,1,2,3,4,5")
        r.add_argument("--spread-arm", default="HIER")
        r.add_argument("--spread-seeds", default="0,1,2,3,4,5")
        r.add_argument("--out-dir", default="out_controls" if name == "controls" else "out_q12")
        r.add_argument("--n-boot", type=int, default=2000)
        r.add_argument("--n-boot-control", type=int, default=300)
        r.add_argument("--expect-files", type=int, default=141)
    a = ap.parse_args(argv)
    return cmd_read(a, controls_only=(a.cmd == "controls"))


if __name__ == "__main__":
    sys.exit(main())
