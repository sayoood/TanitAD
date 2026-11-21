#!/usr/bin/env python3
"""T1-kin cell rows beside the DAG read: DAG vs DAG-0 / DAG vs ORC head F1 (R-TLAT / R-TLON), per-class lat-head F1 (TURN_L / TURN_R and the rest),
and the PAIRED STRATEGIC rows (S checkpoint position / heading at 30 / 50 / 80 m) at T1-kin.

Driver only. Every statistic comes from the programme's own instruments, unchanged:
  * `t1hd_read` (PREREG_V8F_T1_HEADS reducer, imported as a module exactly as dag_read.py imports it): `load_ticks`, `Ctx`, `analyze_run` give, per run, the matched
    ticks' cells (true / pred) and the S checkpoint errors; `f1_red` / `E0._f1_encoded` / `pbrow` / `srow` are the reader's own row builders.
  * the head-F1 pairing is `dag_read.f1_pair_row` re-stated (matched cells = scored in BOTH runs; the T1-HEADS slot layout: 2 cells per window, ticks 1 and 2).
Per-class F1 uses `taniteval.v8f_consistency.macro_f1(...)["per_class_f1"]` (the function the macro-F1 is the nan-mean of) with the same encoded cells, the same
slot layout, and the SAME bootstrap draws (`taniteval.ci._draws`, seed 0, B = 2000); a draw in which a class has no labelled cell is excluded and COUNTED
(`n_draws_class_absent`). The class-F1 bootstrap is cross-checked against the standard estimator on macro-F1 (must reproduce R-TLAT / R-TLON exactly).

Tier: T1-kin = self-action OPEN loop (plain causal bucket, nav dead-reckoned, scene logged, replan every 10 steps). Never "closed loop".
CUDA is never used.
"""
import hashlib
import importlib.util
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np

ART = "E:/Projects/TanitAD-artifacts"
TREE = ART + "/dagt1_tree_b43c7a9a"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
PKG = TREE + "/" + PK
OUT = os.environ.get("DAGF_T1OUT", ART + "/dagfam/DAG_T1CELLS.json")
NB = int(os.environ.get("DAGF_NBOOT", "2000"))
os.environ["TANITEVAL_STACK_OVERRIDE"] = TREE + "/stack"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


T1HD = PKG + "/raw/2026-10-26-t1hdread/t1hd_read.py"
d = os.path.dirname(T1HD)
sys.path.insert(0, d)
T = load_module("t1hd_read", T1HD)
T.setup_imports(TREE)
from taniteval import ci as CI  # noqa: E402

READ_PATH = ART + "/v8f-dagger-t1/read/DAG_READ.json"
READ = json.load(open(READ_PATH, encoding="utf-8"))
RUNS = json.load(open(ART + "/v8f-dagger-t1/runs.json", encoding="utf-8"))
ORC_RUNS = json.load(open(ART + "/v8f-orecover-t1/runs.json", encoding="utf-8"))
SEEDS = (0, 1)
LAT_NAMES = ("LANE_KEEP", "TURN_L", "TURN_R", "LANE_CHANGE_L", "LANE_CHANGE_R")      # build_v9_labels.py:461 (LAT9); the join's lat_cls are v9 ids (v8f_mini_data.py:155)

ns = T.parse(["--mode", "real", "--tool-root", TREE, "--eval-cache", ART + "/sfresh_inputs/cache", "--tracks", ART + "/sfresh_inputs/inputs/native_tracks.pt",
              "--eval-grid", PKG + "/raw/eval_grid_windows.json", "--eval-join", PKG + "/raw/join_eval141.npz", "--banked-dir", ART + "/sfresh_inputs/eval",
              "--out", ART + "/dagfam/_unused.json"])
t_start = time.time()
ctx = T.Ctx(ns)
C = T.C
NCLS = T.NCLS

ENTRY = {}
for s in SEEDS:
    for arm in ("DAG", "DAG0"):
        ENTRY[(arm, s)] = RUNS["arms"][arm][str(s)]["t1"]
    ENTRY[("ORC", s)] = ORC_RUNS["arms"]["ORC"][str(s)]["t1"]

RUN = {}
for key, e in ENTRY.items():
    t0 = time.time()
    arm, s = key
    tt = T.load_ticks(e["ticks"])
    R = T.analyze_run(ctx, tt, ns, f"{arm}_s{s}", NB, e["dump_dir"])
    assert tt["N"] == 2399
    RUN[key] = (R, tt)
    print(f"[{arm} s{s}] analyzed in {time.time() - t0:.0f}s  x1 {R['x1']}", flush=True)

ci = RUN[("DAG", 0)][1]["ci"]
for key, (R, tt) in RUN.items():
    assert np.array_equal(tt["ci"], ci) and np.array_equal(tt["t0"], RUN[("DAG", 0)][1]["t0"]), f"{key}: windows differ"
eid2 = np.repeat(ci, 2)


def cells_ge1(R, hd):
    n = NCLS[hd]
    return T.E0._f1_encoded(R["cells"][hd]["true"], R["cells"][hd]["pred"], n)[:, 1:]            # [N, 2]: ticks 1, 2 (-1 = unscored)


def cells_t0(R, hd):
    n = NCLS[hd]
    return T.E0._f1_encoded(R["cells"][hd]["true"], R["cells"][hd]["pred"], n)[:, 0]


def matched(ea, eb):
    m = (ea >= 0) & (eb >= 0)
    return np.where(m, ea, -1), np.where(m, eb, -1)


def f1_row(Ra, Rb, hd, matched_cells=True):
    """dag_read.f1_pair_row, restated."""
    n = NCLS[hd]
    ea, eb = cells_ge1(Ra, hd), cells_ge1(Rb, hd)
    if matched_cells:
        ea, eb = matched(ea, eb)
    a, b = ea.reshape(-1), eb.reshape(-1)
    red = T.f1_red(n)
    row = T.pbrow(f"{hd} macro-F1 ticks>=1 {'matched' if matched_cells else 'own'}", a, b, eid2, NB, red, record=False, lower_is_better=False, head=hd)
    row.update(n_cells_a=int((a >= 0).sum()), n_cells_b=int((b >= 0).sum()), n_cells_matched_pairs=int(((ea >= 0) & (eb >= 0)).sum()), f1_a=float(red(a)), f1_b=float(red(b)))
    return row, (a, b, red)


def class_red(n, c):
    def red(v):
        v = np.rint(np.asarray(v, np.float64)).astype(np.int64)
        v = v[v >= 0]
        if not len(v):
            return float("nan")
        return float(C.macro_f1(v // (n + 1), v % (n + 1), n)["per_class_f1"][c])
    return red


def paired_boot_nan(a, b, red, eid):
    """The standard estimator's draws (`ci._draws`, seed 0) and percentile rule, with draws whose reducer is NaN (class absent) excluded and counted."""
    uniq, idx = CI.episode_index(eid)
    point = float(red(a) - red(b))
    dd = np.array([float(red(a[sel]) - red(b[sel])) for sel in CI._draws(uniq, idx, NB, 0)])
    ok = np.isfinite(dd)
    lo, hi = np.percentile(dd[ok], [2.5, 97.5])
    return {"delta": point, "lo": float(lo), "hi": float(hi), "separated": bool(lo > 0 or hi < 0), "n_boot": NB, "n_draws_class_absent": int((~ok).sum()),
            "n_episodes": int(len(uniq)), "estimator": "paired_episode_cluster_bootstrap (ci._draws seed 0; NaN draws excluded and counted)"}


def class_rows(Ra, Rb, la, lb, hd="lat"):
    """Per-class F1 (ticks >= 1, cells matched across the two runs): level of each side and the paired difference, for every class of the head."""
    n = NCLS[hd]
    ea, eb = matched(cells_ge1(Ra, hd), cells_ge1(Rb, hd))
    a, b = ea.reshape(-1), eb.reshape(-1)
    out = {}
    for c in range(n):
        red = class_red(n, c)
        sup = int(((a >= 0) & (a // (n + 1) == c)).sum())
        nm = LAT_NAMES[c] if hd == "lat" else f"lon_class_{c}"
        if sup == 0:
            out[nm] = {"class_index": c, "support_cells_matched": 0, "status": "not computed: class has no labelled cell among the matched ticks>=1 cells"}
            continue
        r = paired_boot_nan(a, b, red, eid2)
        r.update({"class_index": c, "support_cells_matched": sup, "f1_a": float(red(a)), "f1_b": float(red(b)),
                  "episodes_with_support": int(len(np.unique(eid2[(a >= 0) & (a // (n + 1) == c)])))})
        out[nm] = r
    return out, (a, b)


def class_levels(R, hd="lat"):
    """Unpaired per-run per-class F1 and support: tick 0 and ticks >= 1 (own scored cells), as T1-HEADS' `confusion` block reports them."""
    n = NCLS[hd]
    e0, eg = cells_t0(R, hd), cells_ge1(R, hd).reshape(-1)
    out = {}
    for lab, e in (("tick0", e0), ("ticks_ge1", eg)):
        m = C.macro_f1(np.where(e >= 0, e // (n + 1), -1), np.where(e >= 0, e % (n + 1), 0), n)
        out[lab] = {"per_class_f1": [None if not np.isfinite(x) else float(x) for x in m["per_class_f1"]], "support": [int(x) for x in m["support"]],
                    "macro_f1": float(m["macro_f1"]), "n_cells": int((e >= 0).sum())}
    return out


def err_arrays(R):
    """pos_err_m / head_err_deg per window, mean over the scored ticks >= 1 (the reader's `per_window`), [N, 3 arcs]."""
    P = {nm: T.per_window(R["err"][nm]["pos_err_m"])[1] for nm in ("S", "CTRV")}
    H = {nm: T.per_window(R["err"][nm]["head_err_deg"])[1] for nm in ("S", "CTRV")}
    return P, H


res = {"schema": "dag-t1cells-v1", "tool_tree": TREE, "n_boot": NB, "n_windows": 2399, "n_episodes": int(len(np.unique(ci))),
       "tier": "T1-kin (self-action OPEN loop; plain causal bucket; nav dead-reckoned; scene logged; replan every 10 steps)",
       "estimator": "taniteval.ci.paired_episode_cluster_bootstrap, B=%d, seed 0, over the 141 episodes; matched cells (scored in both runs), the T1-HEADS slot layout" % NB,
       "inputs": {"read_json_sha256": sha256(READ_PATH), "runs_json_sha256": sha256(ART + "/v8f-dagger-t1/runs.json"),
                  "orc_runs_json_sha256": sha256(ART + "/v8f-orecover-t1/runs.json"), "t1hd_read_sha256": sha256(T1HD),
                  "ticks": {f"{a}_s{s}": {"path": ENTRY[(a, s)]["ticks"], "sha256": sha256(ENTRY[(a, s)]["ticks"])} for (a, s) in ENTRY}},
       "lat_class_names": list(LAT_NAMES), "per_seed": {}, "controls": {}}
ctl = res["controls"]

for s in SEEDS:
    Rd, Rd0, Ro = RUN[("DAG", s)][0], RUN[("DAG0", s)][0], RUN[("ORC", s)][0]
    out = {}
    # -- 1. R-TLAT / R-TLON: DAG - DAG0 (reproduction) and DAG - ORC (the on-policy-vs-law comparison; the arms differ in more than the lever)
    for hd in ("lat", "lon"):
        for (nm, Rb, lb) in (("DAG_minus_DAG0", Rd0, "DAG0"), ("DAG_minus_ORC", Ro, "ORC")):
            row, items = f1_row(Rd, Rb, hd, True)
            row_own, _ = f1_row(Rd, Rb, hd, False)
            row["own_cells_beside"] = {k: row_own[k] for k in ("delta", "lo", "hi", "separated", "n_cells_a", "n_cells_b", "f1_a", "f1_b")}
            out.setdefault(f"R-T{hd.upper()}", {})[nm] = row
            if nm == "DAG_minus_DAG0":
                rr = READ["rows"][f"s{s}"][f"R-T{hd.upper()}"]
                ctl.setdefault("reproduces_reader_head_f1", {})[f"s{s}_{hd}"] = {
                    "here": {k: row[k] for k in ("delta", "lo", "hi", "separated")}, "reader": {k: rr[k] for k in ("delta", "lo", "hi", "separated")},
                    "n_matched_here_vs_reader": [row["n_cells_matched_pairs"], rr["n_cells_matched_pairs"]],
                    "f1_a_f1_b_here_vs_reader": [[row["f1_a"], row["f1_b"]], [rr["f1_a"], rr["f1_b"]]],
                    "PASS": bool(all(row[k] == rr[k] for k in ("delta", "lo", "hi", "separated")) and row["n_cells_matched_pairs"] == rr["n_cells_matched_pairs"]
                                 and abs(row["f1_a"] - rr["f1_a"]) < 1e-12 and abs(row["f1_b"] - rr["f1_b"]) < 1e-12)}
            # identity (a, a) must read exactly +0 [0, 0]
            a_, b_, red_ = items
            idr = T.PB(a_, a_, eid2, n_boot=NB, seed=0, reduce=red_)
            ctl.setdefault("identity_head_f1", {})[f"s{s}_{hd}_{nm}"] = bool(idr["delta"] == 0 and idr["lo"] == 0 and idr["hi"] == 0)
            if nm == "DAG_minus_DAG0":
                # the class-F1 bootstrap routine reproduces the standard estimator on macro-F1
                mine = paired_boot_nan(items[0], items[1], red_, eid2)
                ctl.setdefault("classboot_reproduces_standard_on_macro", {})[f"s{s}_{hd}"] = {
                    "mine": {k: round(mine[k], 4) if isinstance(mine[k], float) else mine[k] for k in ("delta", "lo", "hi", "separated")},
                    "standard": {k: row[k] for k in ("delta", "lo", "hi", "separated")},
                    "PASS": bool(round(mine["delta"], 4) == row["delta"] and round(mine["lo"], 4) == row["lo"] and round(mine["hi"], 4) == row["hi"] and mine["separated"] == row["separated"])}
    # -- 2. per-class lat-head F1 (ticks >= 1, matched cells): DAG - DAG0 (and the levels); DAG - ORC for completeness of the turn classes
    cr, (ca, cb) = class_rows(Rd, Rd0, "DAG", "DAG0", "lat")
    out["per_class_lat_F1_DAG_minus_DAG0_matched"] = cr
    cr2, _ = class_rows(Rd, Ro, "DAG", "ORC", "lat")
    out["per_class_lat_F1_DAG_minus_ORC_matched"] = cr2
    out["per_class_lat_F1_levels_unpaired"] = {"DAG": class_levels(Rd), "DAG0": class_levels(Rd0), "ORC": class_levels(Ro)}
    # identity for the class machinery
    c_t = class_red(NCLS["lat"], 1)
    idc = paired_boot_nan(ca, ca, c_t, eid2)
    ctl.setdefault("identity_class_f1", {})[f"s{s}"] = bool(idc["delta"] == 0 and idc["lo"] == 0 and idc["hi"] == 0)
    # reproduction of the per-class levels the reader's own `confusion` block would give: the macro of the per-class levels (over present classes) equals f1_ge1 of the run's own cells
    lv = out["per_class_lat_F1_levels_unpaired"]["DAG"]["ticks_ge1"]
    pc = [x for x in lv["per_class_f1"] if x is not None]
    own_a = cells_ge1(Rd, "lat").reshape(-1)
    ctl.setdefault("class_levels_macro_equals_run_macro", {})[f"s{s}"] = bool(abs(float(np.mean(pc)) - float(T.f1_red(NCLS['lat'])(own_a))) < 1e-12)
    # -- 3. paired STRATEGIC at T1-kin (DAG - DAG0): S checkpoint position / heading error at 30 / 50 / 80 m, mean over scored ticks >= 1, windows scored in both
    Pd, Hd = err_arrays(Rd)
    Pd0, Hd0 = err_arrays(Rd0)
    strat = {}
    for m, arc in enumerate(T.ARCS):
        strat[f"S-POS@{arc}"] = T.srow(f"S-POS@{arc} DAG - DAG0 (ticks>=1)", Pd["S"][:, m], Pd0["S"][:, m], ci, NB, record=False)
        strat[f"S-HEAD@{arc}"] = T.srow(f"S-HEAD@{arc} DAG - DAG0 (ticks>=1)", Hd["S"][:, m], Hd0["S"][:, m], ci, NB, record=False)
        # identity
        idr = T.srow("id", Pd["S"][:, m], Pd["S"][:, m], ci, NB, record=False)
        ctl.setdefault("identity_spos", {})[f"s{s}_{arc}"] = bool(idr["delta"] == 0 and idr["lo"] == 0 and idr["hi"] == 0)
    out["STRATEGIC_paired_DAG_minus_DAG0"] = strat
    # reproduce the reader's per-run S-POS vs CTRV (ticks >= 1) rows for DAG and DAG0 (the unpaired within-run numbers DAG_READ carries)
    rep = {}
    for lab, R_, P_ in (("DAG", Rd, Pd), ("DAG0", Rd0, Pd0)):
        rr = READ["beside"]["per_seed"][f"s{s}"]["per_run_t1heads"][lab]["S-POS vs CTRV (ticks>=1, within this run)"]
        for m, arc in enumerate(T.ARCS):
            r = T.srow("x", P_["S"][:, m], P_["CTRV"][:, m], ci, NB, record=False)
            q = rr[f"S-POS@{arc}"]
            rep[f"{lab}_S-POS@{arc}"] = {"here": {k: r[k] for k in ("delta", "lo", "hi", "separated")}, "reader": {k: q[k] for k in ("delta", "lo", "hi", "separated")},
                                         "PASS": bool(all(r[k] == q[k] for k in ("delta", "lo", "hi", "separated")))}
    ctl.setdefault("reproduces_reader_spos_vs_ctrv", {})[f"s{s}"] = rep
    # per-run S vs CTRV also kept as rows (this is the 'what exists' at T1-kin besides the paired rows)
    out["STRATEGIC_per_run_S_minus_CTRV"] = {lab: {f"S-POS@{arc}": {k: T.srow("x", P_["S"][:, m], P_["CTRV"][:, m], ci, NB, record=False)[k] for k in ("delta", "lo", "hi", "separated", "n_windows")}
                                                   for m, arc in enumerate(T.ARCS)} for lab, P_ in (("DAG", Pd), ("DAG0", Pd0))}
    out["n_scored_windows_ge1_DAG"] = {f"@{arc}": int(np.isfinite(Pd["S"][:, m]).sum()) for m, arc in enumerate(T.ARCS)}
    out["n_scored_windows_ge1_DAG0"] = {f"@{arc}": int(np.isfinite(Pd0["S"][:, m]).sum()) for m, arc in enumerate(T.ARCS)}
    out["x1_rebuild"] = {"DAG": Rd["x1"], "DAG0": Rd0["x1"], "ORC": Ro["x1"]}
    res["per_seed"][f"s{s}"] = out
    print(f"[s{s}] rows done", flush=True)

res["wall_s"] = round(time.time() - t_start, 1)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    json.dump(clean(res), f, indent=1)
print("wrote", OUT, "sha256", sha256(OUT), flush=True)
