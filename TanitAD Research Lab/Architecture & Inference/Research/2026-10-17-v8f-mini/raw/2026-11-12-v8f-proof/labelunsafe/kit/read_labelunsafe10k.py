#!/usr/bin/env python
"""read_labelunsafe10k.py -- read tool of card PREREG_V8F_PROOF_ADD135 (LABELUNSAFE10K): what are the unsafe windows of the lead rule when it is fed label inputs?
Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions, executor of record v8f_t1_eval_hb.py, heading bound 0.2 1/m); not driving performance. CPU only, banked runs, no new rollout.

Re-reads the dumps of the label form RLc (the subject) and, beside it, GEc and A90 at the model seeds 0 to 3. The per-window, per-step quantities (model gap, time gaps, scored steps, U)
come from the engine files (headway_read_r2.py, read_longroll10k.py and what they import), imported by path and not edited; this file holds no definition of the engine.
Measures M1 to M5 of section 3 with a cluster bootstrap over the evaluation episodes (2,000 draws, seed 135, 95 % interval); names of section 4; controls u0 to u4 of section 5.

  python read_labelunsafe10k.py controls [--t1-root D] [--out-dir D] --tests-out DEV_OUT --tests-out POD_OUT
  python read_labelunsafe10k.py names    [--t1-root D] [--out-dir D]

Stage controls prints only `ctl u<k> PASS|FAIL <reason>` lines (a reason holds no value) and `ZZ READ-LABELUNSAFE10K controls all_pass True|False`; it writes LABELUNSAFE10K_CONTROLS.json.
Stage names is refused (rc 3) unless the controls file has all five controls passing and every input file hashes as the controls stage hashed it; it writes LABELUNSAFE10K_RESULT.json and
LABELUNSAFE10K_VERDICT.md and prints `ZZ READ-LABELUNSAFE10K names done`. An output file that exists is never overwritten. Exit codes: 0 stage done, 3 refused, 1 error.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys

import numpy as np

CARD = "PREREG_V8F_PROOF_ADD135"
PRE = "LABELUNSAFE10K"
TIER_SHORT = "T1-kin/hb0.2"
TIER = "T1-kin/hb0.2 (kinematic rollout on the model's own actions, heading bound 0.2 1/m; not driving performance)"
SUBJECT = "RLc"
TAGS = {"RLc": "I10RLcKF5Hb20", "GEc": "I10RQ50KF5Hb20gGEc", "A90": "I10RQ50KF5Hb20gA90"}
CFG = {
    "tags": TAGS, "subject": SUBJECT, "seeds": (0, 1, 2, 3), "expect_files": 141, "expect_lead": 642, "tol": 1e-4,
    "expected": {"RLc": {"per_seed": [0.0391, 0.0338, 0.0391, 0.0356], "pooled": 0.0369}, "GEc": {"pooled": 0.0480}, "A90": {"pooled": 0.0365}},
    "n_boot": 2000, "boot_seed": 135,
}
M2_TAU, M3_TAU, M4_DELTA, M5_SEEDS = 0.9, 1.2, 0.2, 3
NAME_BAR = 0.5
PLANT_S = 0.35
GAP_BINS = ((0.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, 60.0))
POD_V = "/workspace/v8proof"
DEV_V = "F:/TanitAD-scratch/v8proof"
CONTROLS = tuple("u%d" % i for i in range(5))
CONTROLS_NAME = "LABELUNSAFE10K_CONTROLS.json"
RESULT_NAME = "LABELUNSAFE10K_RESULT.json"
VERDICT_NAME = "LABELUNSAFE10K_VERDICT.md"
MECH_THRESH = PRE + "-K0-THRESHOLD-NEAR-THE-LOGGED-DRIVER"
MECH_CLOSER = PRE + "-K0-RULE-FOLLOWS-CLOSER-THAN-LOGGED"
MECH_MIXED = PRE + "-K0-MIXED"
REP_SAME = PRE + "-K0-SAME-WINDOWS-ACROSS-SEEDS"
REP_DIFF = PRE + "-K0-WINDOWS-DIFFER-BY-SEED"
FAMILIES = [
    "LONGITUDINAL: read here (distance keeping: time gap and gap against the logged driver), %s, not driving performance" % TIER_SHORT,
    "LATERAL: not read by this card (the family reads of these banked runs are in rows 193 and 202); n = 0 here",
    "TACTICAL: not read by this card (the family reads of these banked runs are in rows 193 and 202); n = 0 here",
    "STRATEGIC: not read by this card (the family reads of these banked runs are in rows 193 and 202); n = 0 here",
]
LIMITS = ("%s only, no re-perception, the lead moves as logged; about 83 unsafe windows over four model seeds, so the intervals of the shares are wide; the lines of 0.9 s, 1.2 s and 0.2 s are choices "
          "of the card; four model seeds of one line; the label form is a reference with label values and not a candidate. The interval answers what another draw of episodes would say, "
          "not what another training run would say." % TIER_SHORT)


class LoadError(Exception):
    pass


# ============================================================================================ small helpers
def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def clean(o):
    """json-safe copy: numpy scalars and arrays to python, non-finite floats to None"""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    return o


def norm(p):
    return os.path.normcase(os.path.abspath(p))


def run_folder(root, tag, s):
    return os.path.join(root, "%s_s%d" % (tag, s))


def dump_folder(root, tag, s):
    return os.path.join(run_folder(root, tag, s), "dump_%s_s%d_m10_causal_heads" % (tag, s))


def run_keys(cfg):
    return [(arm, s) for arm in cfg["tags"] for s in cfg["seeds"]]


# ============================================================================================ engine (imported by path, unchanged)
def default_paths():
    pod_code = POD_V + "/podread/code"
    if os.path.isdir(pod_code):
        d = {"headway": pod_code + "/headway/headway_read_r2.py", "q3": pod_code + "/q3", "stageb": pod_code + "/stageB", "tev": pod_code + "/tev",
             "longroll": POD_V + "/podread/root/integ10k/read_longroll10k.py", "join": POD_V + "/podread/data/join_eval141.npz", "grid": POD_V + "/hierrep/pins/eval_grid_windows.json"}
    else:
        d = {"headway": DEV_V + "/headway/headway_read_r2.py", "q3": DEV_V + "/spatial/q3", "stageb": DEV_V + "/pod/stageB", "tev": os.environ.get("TANITEVAL_ROOT", ""),
             "longroll": DEV_V + "/integ10k/read_longroll10k.py", "join": os.environ.get("HEADWAY_JOIN", ""), "grid": ""}
    for k, env in (("headway", "LABELUNSAFE_HEADWAY"), ("q3", "LABELUNSAFE_Q3DIR"), ("stageb", "LABELUNSAFE_STAGEB"), ("tev", "LABELUNSAFE_TEV"), ("longroll", "LABELUNSAFE_LONGROLL"),
                   ("join", "LABELUNSAFE_JOIN"), ("grid", "LABELUNSAFE_GRID")):
        if os.environ.get(env):
            d[k] = os.environ[env]
    return d


class Engine(object):
    pass


def _load_module(name, path):
    p = os.path.abspath(path)
    d = os.path.dirname(p)
    if d not in sys.path:
        sys.path.insert(0, d)
    old = sys.modules.get(name)
    if old is not None and norm(getattr(old, "__file__", "") or "") == norm(p):
        return old
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def bind_engine(paths):
    """imports headway_read_r2.py and read_longroll10k.py by path (their own siblings resolve from the folders named here) and returns the engine handle"""
    for k in ("headway", "longroll"):
        if not paths.get(k) or not os.path.isfile(paths[k]):
            raise LoadError("engine file %s not found" % k)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    if paths.get("q3"):
        os.environ.setdefault("HEADWAY_Q3DIR", paths["q3"])
    if paths.get("stageb"):
        os.environ.setdefault("HIERREP_STAGEB", paths["stageb"])
    if paths.get("tev"):
        os.environ.setdefault("TANITEVAL_ROOT", paths["tev"])
    e = Engine()
    e.H = _load_module("headway_read_r2", paths["headway"])
    e.R = _load_module("read_longroll10k", paths["longroll"])
    e.RH = e.H.RH
    e.files = {"engine_headway": e.H.__file__, "engine_longroll": e.R.__file__, "engine_q3": e.H.Q.__file__, "engine_hierrep": e.RH.__file__}
    return e


# ============================================================================================ per-window quantities (all definitions are the engine's)
def step_arrays(E, g, cl, L):
    """per window and step, from the engine: model gap gm, logged gap gl (NaN outside the scored steps), model and logged time gaps, model speed, the unsafe step flags, and the engine's U.
    The time gaps are the engine's own formula (gap over the step speed with the floor) applied to the engine's gaps and speeds; control u0 compares the unsafe flags with the engine's U."""
    H = E.H
    c = H.window_cells(g, cl, L)
    gm, gl = np.asarray(c["_gm"], np.float64), np.asarray(c["_gl"], np.float64)
    vm, vl = H.step_speeds(cl), H.step_speeds(g)
    with np.errstate(invalid="ignore", divide="ignore"):
        tm = gm / np.maximum(vm, H.V_FLOOR)
        tl = gl / np.maximum(vl, H.V_FLOOR)
        un = ((gm < H.UNSAFE_GAP_M) | (tm < H.UNSAFE_TAU_S)) & np.isfinite(gm)
    U = np.asarray(c["U"], np.float64)
    return {"gm": gm, "gl": gl, "tm": tm, "tl": tl, "vm": vm, "un": un, "any": un.any(1), "U": U, "ud": np.isfinite(U), "gap_lim": float(H.UNSAFE_GAP_M), "tau_lim": float(H.UNSAFE_TAU_S)}


def summarize(A):
    """per-window summaries used by the measures and by control u1"""
    fin_m, fin_l = np.isfinite(A["tm"]), np.isfinite(A["tl"])
    min_tm = np.where(fin_m, A["tm"], np.inf).min(1)
    min_tl = np.where(fin_l, A["tl"], np.inf).min(1)
    min_gm = np.where(np.isfinite(A["gm"]), A["gm"], np.inf).min(1)
    for x in (min_tm, min_tl, min_gm):
        x[~np.isfinite(x)] = np.nan
    with np.errstate(invalid="ignore"):
        has_gap = (A["gm"] < A["gap_lim"]).any(1)
        has_tau = (A["tm"] < A["tau_lim"]).any(1)
    first0 = np.argmax(A["un"], axis=1)
    ar = np.arange(len(first0))
    first = np.where(A["any"], first0 + 1, 0)
    return {"sel": A["ud"] & A["any"], "ud": A["ud"], "min_tm": min_tm, "min_tl": min_tl, "min_gm": min_gm, "has_gap": has_gap, "has_tau": has_tau, "first": first,
            "gl_first": A["gl"][ar, first0], "vm_first": A["vm"][ar, first0]}


def second_routine(E, g, cl, L):
    """control u1: a second routine, window by window and step by step in scalar form: projection of the model point on the extended logged path, model gap, time gaps; returns the lowest model time
    gap, the lowest logged time gap, the lowest model gap and the first unsafe step (0 = none) of every lead window (NaN / 0 elsewhere)"""
    H = E.H
    n = len(g)
    out = {k: np.full(n, np.nan) for k in ("min_tm", "min_tl", "min_gm")}
    first = np.zeros(n, np.int64)
    ray, dt, floor = float(H.RAY), float(H.DT), float(H.V_FLOOR)
    for i in np.flatnonzero(L["lead"]):
        P = np.vstack([np.zeros((1, 2)), np.asarray(g[i], np.float64)])
        d = P[1:] - P[:-1]
        ln = np.hypot(d[:, 0], d[:, 1])
        nz = np.flatnonzero(ln > 1e-9)
        if len(nz):
            u0, u1 = d[nz[0]] / ln[nz[0]], d[nz[-1]] / ln[nz[-1]]
        else:
            u0 = u1 = np.array([1.0, 0.0])
        V = np.vstack([P[0] - ray * u0, P, P[-1] + ray * u1])
        seg = V[1:] - V[:-1]
        sl = np.hypot(seg[:, 0], seg[:, 1])
        arc = np.zeros(33)
        arc[0] = -sl[0]
        for j in range(2, 33):
            arc[j] = arc[j - 1] + sl[j - 1]
        ok = sl > 1e-9
        l2 = np.where(ok, sl * sl, 1.0)
        mt = ml = mg = np.inf
        fu = 0
        for k in range(1, int(L["ns"][i]) + 1):
            q = np.asarray(cl[i, k - 1], np.float64)
            t = np.clip(((q - V[:-1]) * seg).sum(1) / l2, 0.0, 1.0)
            pr = V[:-1] + t[:, None] * seg
            dist = np.hypot(q[0] - pr[:, 0], q[1] - pr[:, 1])
            dist[~ok] = np.inf
            j = int(np.argmin(dist))
            a_m = arc[j] + t[j] * sl[j]
            gl = float(L["G"][i, k])
            gm = gl + arc[k + 1] - a_m
            qp = np.zeros(2) if k == 1 else np.asarray(cl[i, k - 2], np.float64)
            vm = math.hypot(q[0] - qp[0], q[1] - qp[1]) / dt
            gp = np.zeros(2) if k == 1 else np.asarray(g[i, k - 2], np.float64)
            vl = math.hypot(g[i, k - 1][0] - gp[0], g[i, k - 1][1] - gp[1]) / dt
            tm, tl = gm / max(vm, floor), gl / max(vl, floor)
            mt, ml, mg = min(mt, tm), min(ml, tl), min(mg, gm)
            if fu == 0 and (gm < float(H.UNSAFE_GAP_M) or tm < float(H.UNSAFE_TAU_S)):
                fu = k
        if math.isfinite(mt):
            out["min_tm"][i], out["min_tl"][i], out["min_gm"][i] = mt, ml, mg
        first[i] = fu
    out["first"] = first
    return out


def routines_agree(S1, S2, lead, tol=1e-9):
    """the two routines agree on every lead window: lowest time gaps (model and logged), lowest model gap within tol, first unsafe step exactly"""
    for k in ("min_tm", "min_tl", "min_gm"):
        a, b = np.asarray(S1[k], np.float64)[lead], np.asarray(S2[k], np.float64)[lead]
        if not np.all((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= tol)):
            return False
    return bool(np.array_equal(np.asarray(S1["first"])[lead], np.asarray(S2["first"])[lead]))


# ============================================================================================ measures and bootstrap
def make_draws(n_ep, n_boot, seed):
    return np.random.default_rng(seed).integers(0, n_ep, size=(n_boot, n_ep))


def episode_sums(flag, mask, inv, n_ep):
    return np.bincount(inv, weights=(np.asarray(flag, bool) & np.asarray(mask, bool)).astype(np.float64), minlength=n_ep)


def share_read(numE, denE, idx):
    """share with the interval of the cluster bootstrap over episodes (the same draws for every measure): -> value, [lo, hi], n (the denominator)"""
    den = float(denE.sum())
    if den <= 0.0:
        return float("nan"), [float("nan"), float("nan")], 0
    v = float(numE.sum() / den)
    nb, db = numE[idx].sum(1), denE[idx].sum(1)
    ok = db > 0
    if not ok.any():
        return v, [float("nan"), float("nan")], int(round(den))
    lo, hi = np.percentile(nb[ok] / db[ok], [2.5, 97.5])
    return v, [float(lo), float(hi)], int(round(den))


def arm_measures(sums, inv, n_ep, idx):
    """M1 to M5 of one arm over the model seeds (pooled: a window of the grid counts once per model seed) and the counts beside them"""
    flags = {
        "M1": lambda s: ~s["has_gap"],
        "M2": lambda s: s["min_tm"] >= M2_TAU,
        "M3": lambda s: s["min_tl"] < M3_TAU,
        "M4": lambda s: s["min_tm"] < s["min_tl"] - M4_DELTA,
    }
    out, cnt = {}, {}
    with np.errstate(invalid="ignore"):
        for m, fn in flags.items():
            num = sum(episode_sums(fn(s), s["sel"], inv, n_ep) for s in sums)
            den = sum(episode_sums(np.ones(len(s["sel"]), bool), s["sel"], inv, n_ep) for s in sums)
            v, iv, n = share_read(num, den, idx)
            out[m] = {"value": v, "interval": iv, "n": n}
        # beside M3: the same share over all scored windows (U defined)
        numa = sum(episode_sums(flags["M3"](s), s["ud"], inv, n_ep) for s in sums)
        dena = sum(episode_sums(np.ones(len(s["ud"]), bool), s["ud"], inv, n_ep) for s in sums)
    va, iva, na = share_read(numa, dena, idx)
    stack = np.array([s["sel"] for s in sums])
    c = stack.sum(0)
    anyw, hiw = c >= 1, c >= M5_SEEDS
    v5, iv5, n5 = share_read(episode_sums(hiw, anyw, inv, n_ep), episode_sums(np.ones(len(anyw), bool), anyw, inv, n_ep), idx)
    out["M5"] = {"value": v5, "interval": iv5, "n": n5}
    allm = np.concatenate([s["min_tm"][s["sel"]] for s in sums])
    cnt["median_lowest_tau_m_s"] = float(np.median(allm)) if len(allm) else float("nan")
    cnt["M3_over_all_scored_windows"] = {"value": va, "interval": iva, "n": na}
    cnt["n_unsafe_windows_per_seed"] = [int(s["sel"].sum()) for s in sums]
    cnt["n_scored_windows_per_seed"] = [int(s["ud"].sum()) for s in sums]
    cnt["n_distinct_windows_unsafe_at_one_or_more_seeds"] = int(anyw.sum())
    cnt["distinct_windows_by_number_of_seeds_unsafe"] = {str(j): int((c == j).sum()) for j in range(1, len(sums) + 1)}
    return out, cnt


def first_step_reads(sums):
    """beside: step index of the first unsafe step, and the label gap and model speed at it by gap band, over the unsafe windows of the subject (pooled over seeds)"""
    first = np.concatenate([s["first"][s["sel"]] for s in sums])
    gl = np.concatenate([s["gl_first"][s["sel"]] for s in sums])
    vm = np.concatenate([s["vm_first"][s["sel"]] for s in sums])
    rd = {"tier": TIER_SHORT, "n": int(len(first))}
    if len(first):
        q = np.percentile(first, [25, 50, 75])
        rd["first_unsafe_step_index"] = {"min": int(first.min()), "q25": float(q[0]), "median": float(q[1]), "q75": float(q[2]), "max": int(first.max()),
                                         "count_by_step": [int((first == k).sum()) for k in range(1, 31)]}
    else:
        rd["first_unsafe_step_index"] = None
    bands = []
    for j, (lo, hi) in enumerate(GAP_BINS):
        m = (gl >= lo) & ((gl < hi) if j < len(GAP_BINS) - 1 else (gl <= hi))
        bands.append({"gap_band_m": [lo, hi], "n": int(m.sum()), "median_label_gap_m": float(np.median(gl[m])) if m.any() else None,
                      "median_model_speed_mps": float(np.median(vm[m])) if m.any() else None})
    rd["by_label_gap_at_first_unsafe_step"] = bands
    return rd


def overlap_with_subject(sums_arm, sums_subject):
    per, num, den = [], 0, 0
    for a, b in zip(sums_arm, sums_subject):
        n = int(a["sel"].sum())
        k = int((a["sel"] & b["sel"]).sum())
        per.append(k / n if n else float("nan"))
        num, den = num + k, den + n
    return {"per_seed": per, "pooled": (num / den) if den else float("nan"), "n_unsafe_windows_of_arm": den}


# ============================================================================================ names (two routines that must agree)
def mechanism_name(m):
    m3lo, (m4lo, m4hi) = m["M3"]["interval"][0], m["M4"]["interval"]
    if m3lo >= NAME_BAR and m4hi <= NAME_BAR:
        return MECH_THRESH
    if m4lo >= NAME_BAR:
        return MECH_CLOSER
    return MECH_MIXED


def mechanism_name_b(m):
    a, b, c = (float(m["M3"]["interval"][0]), float(m["M4"]["interval"][0]), float(m["M4"]["interval"][1]))
    fin = [math.isfinite(x) for x in (a, b, c)]
    one = fin[0] and fin[2] and (a - NAME_BAR >= 0.0) and (NAME_BAR - c >= 0.0)
    two = fin[1] and (b - NAME_BAR >= 0.0)
    return [MECH_MIXED, MECH_THRESH, MECH_CLOSER, MECH_CLOSER][int(one) + 2 * int((not one) and two)]


def repeat_name(m):
    v = m["M5"]["value"]
    return REP_SAME if (math.isfinite(v) and v >= NAME_BAR) else REP_DIFF


def repeat_name_b(m):
    v = float(m["M5"]["value"])
    return (REP_DIFF, REP_SAME)[int(math.isfinite(v) and v - NAME_BAR >= 0.0)]


def derive_names(m):
    a, b = (mechanism_name(m), repeat_name(m)), (mechanism_name_b(m), repeat_name_b(m))
    if a != b:
        raise SystemExit("REFUSED: the two routines of the names disagree")
    return list(a)


# ============================================================================================ world (the twelve runs, restricted to the grid)
class World(object):
    pass


def load_world(E, cfg, root, join_path, grid_path):
    w = World()
    w.problems, w.D, w.n_files, w.done, w.lead_counts = [], {}, {}, {}, {}
    J = E.H.load_join(join_path)
    grid = E.R.jl(grid_path) if grid_path else None
    if grid is None:
        raise LoadError("grid file cannot be read")
    keys = E.R.grid_keys(grid, [str(c) for c in J["clip"]])
    w.J = J
    for arm, s in run_keys(cfg):
        key = "%s_s%d" % (arm, s)
        tag = cfg["tags"][arm]
        w.done[key] = os.path.isfile(os.path.join(run_folder(root, tag, s), "T1_DONE"))
        raw = E.R.load_dump_dir(dump_folder(root, tag, s))
        if raw is None:
            w.problems.append("%s: no dump files" % key)
            continue
        w.n_files[key] = int(raw["n_files"])
        idx, rep = E.R.restrict_idx(raw["clip_index"], raw["ws"], keys)
        if rep["missing"] or rep["duplicated_windows"]:
            w.problems.append("%s: windows of the grid missing or repeated" % key)
            continue
        w.D[(arm, s)] = E.R.cut(raw, idx)
    ref = next(iter(w.D.values()), None)
    w.ref = ref
    if ref is not None:
        for k, D in w.D.items():
            if not all(np.array_equal(D[f], ref[f]) for f in ("g", "ws", "eid", "clip_index", "v0")):
                w.problems.append("%s_s%d: windows differ from the first run" % k)
        w.L = E.H.label_windows(J, ref["clip_index"], ref["ws"])
        for k, D in w.D.items():
            w.lead_counts["%s_s%d" % k] = int(E.H.label_windows(J, D["clip_index"], D["ws"])["lead"].sum())
        w.uniq, w.inv = E.RH.episode_units(ref["eid"])
        w.n_ep = len(w.uniq)
    w.complete = bool(ref is not None and len(w.D) == len(run_keys(cfg)))
    w.A, w.S = {}, {}
    if w.complete:
        for k, D in w.D.items():
            w.A[k] = step_arrays(E, D["g"], D["cl"], w.L)
            w.S[k] = summarize(w.A[k])
    return w


# ============================================================================================ controls
def ctl_u0(cfg, w):
    """identity: the unsafe rate per model seed and pooled reproduces row 202 within tol, from the engine's own per-window U; the unsafe flags of this tool equal the engine's U"""
    if not w.complete:
        return False, "the runs could not be read"
    for k, A in w.A.items():
        ud = A["ud"]
        if not np.array_equal(A["any"][ud], A["U"][ud] == 1.0):
            return False, "unsafe windows of this tool differ from the engine's U (%s_s%d)" % k
    for arm, e in cfg["expected"].items():
        lev = [float(np.mean(w.A[(arm, s)]["U"][w.A[(arm, s)]["ud"]])) if w.A[(arm, s)]["ud"].any() else float("nan") for s in cfg["seeds"]]
        if "per_seed" in e:
            for s, x, want in zip(cfg["seeds"], lev, e["per_seed"]):
                if not (abs(x - want) <= cfg["tol"]):
                    return False, "rate of %s at model seed %d differs from row 202" % (arm, s)
        if not (abs(float(np.mean(lev)) - e["pooled"]) <= cfg["tol"]):
            return False, "pooled rate of %s differs from row 202" % arm
    return True, ""


def ctl_u1(E, cfg, w):
    """second routine: lowest time gap and first unsafe step of every lead window, step by step, within 1e-9"""
    if not w.complete:
        return False, "the runs could not be read"
    for k, D in w.D.items():
        S2 = second_routine(E, D["g"], D["cl"], w.L)
        if not routines_agree(w.S[k], S2, w.L["lead"]):
            return False, "second routine differs from the first (%s_s%d)" % k
    return True, ""


def ctl_u2(E, w, plant_s=PLANT_S, ident_cl=None):
    """planted: model track = logged track gives no unsafe window and no window for M4; the logged track advanced along the path by plant_s seconds of travel at each step gives windows that
    all count for M4, and every window unsafe by the gap alone has a lowest model gap below 2.0 m by the second routine (a plant that makes no window unsafe fails)"""
    if not w.complete:
        return False, "the runs could not be read"
    H = E.H
    g, L = w.ref["g"], w.L
    A0 = step_arrays(E, g, g if ident_cl is None else ident_cl, L)
    S0 = summarize(A0)
    if S0["sel"].any() or bool(np.nansum(A0["U"][A0["ud"]]) != 0.0):
        return False, "the logged track read as the model track has an unsafe window"
    a_l, v_l = H.logged_arc(g), H.step_speeds(g)
    cl_p = H.point_at_arc(g, a_l + plant_s * v_l)
    Ap = step_arrays(E, g, cl_p, L)
    Sp = summarize(Ap)
    sel = Sp["sel"]
    if not sel.any():
        return False, "the planted track makes no window unsafe"
    with np.errstate(invalid="ignore"):
        m4 = Sp["min_tm"] < Sp["min_tl"] - M4_DELTA
    if not bool(m4[sel].all()):
        return False, "a window made unsafe by the planted track does not count for M4"
    S2 = second_routine(E, g, cl_p, L)
    alone = sel & Sp["has_gap"] & ~Sp["has_tau"]
    if alone.any() and not bool((S2["min_gm"][alone] < float(H.UNSAFE_GAP_M)).all()):
        return False, "a window unsafe by the gap alone has no model gap below 2.0 m"
    if not bool(np.array_equal(Sp["has_gap"][sel], S2["min_gm"][sel] < float(H.UNSAFE_GAP_M))):
        return False, "the step with a gap below 2.0 m differs between the two routines in a planted window"
    return True, ""


def pytest_ok(text):
    """the text of a pytest run: a summary line `N passed` (N > 0) with no failed or error count, and no FAILED or ERROR line"""
    lines = [x for x in (text or "").splitlines() if x.strip()]
    if not lines:
        return False
    last = lines[-1]
    m = re.search(r"(\d+) passed", last)
    if not m or int(m.group(1)) <= 0 or re.search(r"\d+ (failed|errors?)\b", last):
        return False
    return not any(re.match(r"\s*(FAILED|ERROR)\b", x) for x in lines)


def ctl_u3(paths):
    """tests: the two outputs (dev box, pod) of this tool's test file read N passed and no failure; they lie outside the kit folder"""
    if len(paths) != 2:
        return False, "two test outputs are needed (dev box and pod)"
    here = norm(os.path.dirname(os.path.abspath(__file__)))
    for p in paths:
        if not os.path.isfile(p):
            return False, "a test output is missing"
        parts = [x.lower() for x in re.split(r"[\\/]", os.path.abspath(p))]
        if "kit" in parts or norm(p).startswith(here + os.sep):
            return False, "a test output lies inside the kit folder"
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        if not pytest_ok(text) or "test_read_labelunsafe10k" not in text:
            return False, "a test output does not read N passed with no failure"
    return True, ""


def ctl_u4(cfg, w):
    """counts: 141 dump files and the done marker for each of the twelve runs; 642 lead windows per model seed"""
    for arm, s in run_keys(cfg):
        key = "%s_s%d" % (arm, s)
        if not w.done.get(key):
            return False, "done marker missing for %s" % key
        if w.n_files.get(key) != cfg["expect_files"]:
            return False, "dump file count of %s is not %d" % (key, cfg["expect_files"])
    if w.problems or not w.complete:
        return False, "windows of the twelve runs are missing or differ"
    for arm, s in run_keys(cfg):
        if w.lead_counts.get("%s_s%d" % (arm, s)) != cfg["expect_lead"]:
            return False, "lead window count of %s_s%d is not %d" % (arm, s, cfg["expect_lead"])
    return True, ""


def run_controls(E, cfg, root, paths, tests):
    """-> (dict u<k> -> (ok, reason), world or None)"""
    res = {}
    try:
        w = load_world(E, cfg, root, paths["join"], paths["grid"])
    except Exception as ex:                                                        # noqa: BLE001
        w = None
        why = "inputs could not be read (%s)" % type(ex).__name__
        for c in ("u0", "u1", "u2", "u4"):
            res[c] = (False, why)
    if w is not None:
        for c, fn in (("u0", lambda: ctl_u0(cfg, w)), ("u1", lambda: ctl_u1(E, cfg, w)), ("u2", lambda: ctl_u2(E, w)), ("u4", lambda: ctl_u4(cfg, w))):
            try:
                res[c] = fn()
            except Exception as ex:                                                # noqa: BLE001
                res[c] = (False, "the control raised %s" % type(ex).__name__)
    try:
        res["u3"] = ctl_u3(tests)
    except Exception as ex:                                                        # noqa: BLE001
        res["u3"] = (False, "the control raised %s" % type(ex).__name__)
    return res, w


# ============================================================================================ inputs (hashes) and files
def dump_aggregate(folder):
    h, n = hashlib.sha256(), 0
    if os.path.isdir(folder):
        for f in sorted(x for x in os.listdir(folder) if re.match(r"ep.*\.npz$", x)):
            h.update(("%s %s\n" % (f, sha_file(os.path.join(folder, f)))).encode("utf-8"))
            n += 1
    return {"path": os.path.abspath(folder), "sha256": h.hexdigest(), "n_files": n}


def collect_inputs(E, cfg, root, paths, tests):
    files = {"this_tool": os.path.abspath(__file__), "join": paths["join"], "grid": paths["grid"]}
    files.update(E.files)
    for i, t in enumerate(tests):
        files["tests_out_%d" % i] = t
    for arm, s in run_keys(cfg):
        files["done_%s_s%d" % (arm, s)] = os.path.join(run_folder(root, cfg["tags"][arm], s), "T1_DONE")
    inp = {"files": {lab: {"path": os.path.abspath(p), "sha256": sha_file(p) if os.path.isfile(p) else None} for lab, p in files.items() if p},
           "run_dumps": {"%s_s%d" % (arm, s): dump_aggregate(dump_folder(root, cfg["tags"][arm], s)) for arm, s in run_keys(cfg)}}
    return inp


def inputs_changed(inputs):
    fl, rd = (inputs or {}).get("files"), (inputs or {}).get("run_dumps")
    if not isinstance(fl, dict) or not fl or not isinstance(rd, dict) or not rd:
        return ["the controls file records no input hashes"]
    bad = []
    for lab, v in sorted(fl.items()):
        p, want = (v or {}).get("path"), (v or {}).get("sha256")
        got = sha_file(p) if p and os.path.isfile(p) else None
        if got != want:
            bad.append("%s differs from the controls stage" % lab)
    for lab, v in sorted(rd.items()):
        now = dump_aggregate(v.get("path", ""))
        if now["sha256"] != v.get("sha256") or now["n_files"] != v.get("n_files"):
            bad.append("dump files of %s differ from the controls stage" % lab)
    return bad


def write_new(path, text):
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        f.write(text)


# ============================================================================================ the read (names stage)
def read_all(cfg, w):
    n_ep = w.n_ep
    idx = make_draws(n_ep, cfg["n_boot"], cfg["boot_seed"])
    seeds = list(cfg["seeds"])
    sums = {arm: [w.S[(arm, s)] for s in seeds] for arm in cfg["tags"]}
    meas, cnt = {}, {}
    for arm in cfg["tags"]:
        meas[arm], cnt[arm] = arm_measures(sums[arm], w.inv, n_ep, idx)
    sub = cfg["subject"]
    beside = {"tier": TIER_SHORT, "counts": {a: cnt[a] for a in cfg["tags"]}, "first_unsafe": first_step_reads(sums[sub]),
              "arms_M1_to_M5": {a: meas[a] for a in cfg["tags"] if a != sub},
              "GEc_unsafe_windows_also_unsafe_for_RLc_same_model_seed": overlap_with_subject(sums["GEc"], sums[sub]) if "GEc" in sums else None,
              "n_evaluation_episodes": int(n_ep), "n_lead_windows_per_seed": [int(w.L["lead"].sum())] * len(seeds)}
    return meas[sub], beside, meas


def result_json(cfg, m, beside, controls_sha):
    names = derive_names(m)
    return {"card": CARD, "tier": TIER, "names": names, "measures": m, "beside": beside, "families": FAMILIES,
            "estimator": "cluster bootstrap over the evaluation episodes, %d draws, seed %d, 95 percent interval; the interval answers what another draw of episodes would say, not what another training run would say" % (cfg["n_boot"], cfg["boot_seed"]),
            "controls_sha256": controls_sha}


def _f(x, nd=4):
    return "nan" if x is None or not math.isfinite(x) else ("%." + str(nd) + "f") % x


def verdict_md(res):
    m = res["measures"]
    L = ["# VERDICT LABELUNSAFE10K (card %s)" % CARD, "", "Tier: %s. Not driving performance. Controls u0 to u4 passed before this file was written (controls file sha256 %s)." % (res["tier"], str(res["controls_sha256"])[:16]), "",
         "## Names", "", "* Mechanism: **%s**" % res["names"][0], "* Repeat: **%s**" % res["names"][1], "",
         "## Measures on the unsafe windows of RLc (pooled over the four model seeds; %s)" % TIER_SHORT, "", "| measure | value | 95 % interval | n |", "|---|---|---|---|"]
    txt = {"M1": "M1 unsafe by the time gap alone (no step with gap below 2.0 m)", "M2": "M2 lowest time gap at least 0.9 s", "M3": "M3 logged driver's lowest time gap below 1.2 s",
           "M4": "M4 lowest model time gap more than 0.2 s below the logged one", "M5": "M5 of distinct windows unsafe at one seed or more, share unsafe at three or four"}
    for k in ("M1", "M2", "M3", "M4", "M5"):
        L.append("| %s | %s | [%s, %s] | %d |" % (txt[k], _f(m[k]["value"]), _f(m[k]["interval"][0]), _f(m[k]["interval"][1]), m[k]["n"]))
    b = res["beside"]
    c = b["counts"][SUBJECT]
    L += ["", "## Beside (no name)", "", "* Median of the lowest model time gap: %s s. M3 over all scored windows: %s [%s, %s], n %d." % (
        _f(c["median_lowest_tau_m_s"]), _f(c["M3_over_all_scored_windows"]["value"]), _f(c["M3_over_all_scored_windows"]["interval"][0]), _f(c["M3_over_all_scored_windows"]["interval"][1]), c["M3_over_all_scored_windows"]["n"]),
          "* Unsafe windows per model seed: %s; scored windows per model seed: %s; distinct windows unsafe at one seed or more: %d; by number of seeds: %s." % (
              c["n_unsafe_windows_per_seed"], c["n_scored_windows_per_seed"], c["n_distinct_windows_unsafe_at_one_or_more_seeds"], c["distinct_windows_by_number_of_seeds_unsafe"])]
    fu = b["first_unsafe"]
    if fu["first_unsafe_step_index"]:
        s = fu["first_unsafe_step_index"]
        L.append("* Step index of the first unsafe step: median %s, quartiles %s to %s, range %d to %d." % (_f(s["median"], 1), _f(s["q25"], 1), _f(s["q75"], 1), s["min"], s["max"]))
    for x in fu["by_label_gap_at_first_unsafe_step"]:
        L.append("* Label gap %g to %g m at the first unsafe step: n %d, median label gap %s m, median model speed %s m/s." % (x["gap_band_m"][0], x["gap_band_m"][1], x["n"], _f(x["median_label_gap_m"], 2), _f(x["median_model_speed_mps"], 2)))
    for a, mm in b["arms_M1_to_M5"].items():
        L.append("* %s: " % a + "; ".join("%s %s [%s, %s] n %d" % (k, _f(mm[k]["value"]), _f(mm[k]["interval"][0]), _f(mm[k]["interval"][1]), mm[k]["n"]) for k in ("M1", "M2", "M3", "M4", "M5")))
    o = b["GEc_unsafe_windows_also_unsafe_for_RLc_same_model_seed"]
    if o:
        L.append("* Unsafe windows of GEc that are also unsafe windows of RLc at the same model seed: pooled %s (n %d), per seed %s." % (_f(o["pooled"]), o["n_unsafe_windows_of_arm"], [_f(x) for x in o["per_seed"]]))
    L += ["", "## Families", ""] + ["* " + x for x in res["families"]] + ["", "## Limits", "", LIMITS, ""]
    return "\n".join(L)


# ============================================================================================ stages
def stage_controls(a, cfg):
    out = os.path.join(a.out_dir, CONTROLS_NAME)
    if os.path.exists(out):
        print("REFUSED: %s exists" % out)
        return 3
    os.makedirs(a.out_dir, exist_ok=True)
    paths = paths_of(a)
    tests = list(a.tests_out or [])
    E = bind_engine(paths)
    res, _ = run_controls(E, cfg, a.t1_root, paths, tests)
    rec = {"card": CARD}
    for c in CONTROLS:
        rec["pass_" + c] = bool(res[c][0])
        print("ctl %s %s%s" % (c, "PASS" if res[c][0] else "FAIL", "" if res[c][0] else " " + res[c][1]))
    rec["all_pass"] = bool(all(res[c][0] for c in CONTROLS))
    rec["reasons"] = {c: res[c][1] for c in CONTROLS if not res[c][0]}
    rec["inputs"] = collect_inputs(E, cfg, a.t1_root, paths, tests)
    write_new(out, json.dumps(clean(rec), indent=1))
    print("ZZ READ-LABELUNSAFE10K controls all_pass %s" % rec["all_pass"])
    return 0


def refuse(why):
    print("REFUSED: %s" % why)
    return 3


def stage_names(a, cfg):
    cpath = os.path.join(a.out_dir, CONTROLS_NAME)
    if not os.path.isfile(cpath):
        return refuse("no controls file")
    try:
        with open(cpath, "r", encoding="utf-8") as f:
            rec = json.load(f)
    except ValueError:
        return refuse("the controls file cannot be read")
    if not isinstance(rec, dict) or rec.get("card") != CARD:
        return refuse("the controls file is not that of the card")
    for c in CONTROLS:
        if rec.get("pass_" + c) is not True:
            return refuse("control %s is not recorded as passed" % c)
    if rec.get("all_pass") is not True:
        return refuse("all_pass is not True")
    bad = inputs_changed(rec.get("inputs"))
    if bad:
        return refuse("an input differs from the controls stage: " + "; ".join(bad[:4]))
    rpath, vpath = os.path.join(a.out_dir, RESULT_NAME), os.path.join(a.out_dir, VERDICT_NAME)
    if os.path.exists(rpath) or os.path.exists(vpath):
        return refuse("an output file exists")
    paths = paths_of(a)
    E = bind_engine(paths)
    recorded = {norm(v["path"]) for v in rec["inputs"]["files"].values() if v.get("path")}
    for lab, p in E.files.items():
        if norm(p) not in recorded or rec["inputs"]["files"].get(lab, {}).get("path") is None or norm(rec["inputs"]["files"][lab]["path"]) != norm(p):
            return refuse("the engine file %s is not the one hashed by the controls stage" % lab)
    if norm(__file__) != norm(rec["inputs"]["files"]["this_tool"]["path"]):
        return refuse("this tool is not the one hashed by the controls stage")
    w = load_world(E, cfg, a.t1_root, paths["join"], paths["grid"])
    if not w.complete or w.problems:
        return refuse("the runs are incomplete")
    m, beside, _ = read_all(cfg, w)
    res = result_json(cfg, m, beside, sha_file(cpath))
    write_new(rpath, json.dumps(clean(res), indent=1))
    write_new(vpath, verdict_md(res))
    print("names: %s | %s" % tuple(res["names"]))
    print("ZZ READ-LABELUNSAFE10K names done")
    return 0


def paths_of(a):
    d = default_paths()
    for k in ("headway", "q3", "stageb", "tev", "longroll", "join", "grid"):
        v = getattr(a, k, None)
        if v:
            d[k] = v
    return d


def build_parser():
    d = default_paths()
    ap = argparse.ArgumentParser(description="read tool of card %s" % CARD)
    ap.add_argument("stage", choices=("controls", "names"))
    ap.add_argument("--t1-root", default=POD_V + "/integ10k/t1")
    ap.add_argument("--out-dir", default=POD_V + "/labelunsafe/read_out")
    ap.add_argument("--tests-out", action="append", default=None, help="pytest output of the dev box, then of the pod (outside the kit folder)")
    ap.add_argument("--headway", default=d["headway"])
    ap.add_argument("--q3", default=d["q3"])
    ap.add_argument("--stageb", default=d["stageb"])
    ap.add_argument("--tev", default=d["tev"])
    ap.add_argument("--longroll", default=d["longroll"])
    ap.add_argument("--join", default=d["join"])
    ap.add_argument("--grid", default=d["grid"])
    return ap


def main(argv=None, cfg=None):
    a = build_parser().parse_args(argv)
    cfg = cfg or CFG
    try:
        return stage_controls(a, cfg) if a.stage == "controls" else stage_names(a, cfg)
    except LoadError as ex:
        print("REFUSED: %s" % ex)
        return 3


if __name__ == "__main__":
    sys.exit(main())
