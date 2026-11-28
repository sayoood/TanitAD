#!/usr/bin/env python3
"""speedsub_t1_read.py -- the dev-box T1-kin read of PREREG_V8F_SPEEDSUB_T1 (claim row H-V8F-SPEEDSUB-T1-1), modelled on the DAG-ORC reader (`dag_orc_read.py`).

NOTHING IS RE-DERIVED. The Read-A rows, the TAC / DK rows and the T1-HEADS pipeline are the DAG-ORC reader's own `Rig` (imported from `--orc-dir`, its `ref/` pins intact). This file adds only the
PREREG's composition, in two layers:

  PURE LAYER (no heavy import; every function below `# ---- pure` is unit-tested and mutation-tested)
     the bar (-F/2, inclusive on the point), HIT, the guards G-WORSE / G-NEW / G-ADE / G-VIOL, the decision table (REFUSED, HARMS, PASS, PARTIAL, NULL) as ONE function, section 6.1 Read A,
     section 6.2 the floor rule, the controls K-ID / K-ZERO / K-NONSPEED / K-REC / K-REACH / K-SHUF / K-FLOOR / K-SHA / K-REG, the four families (section 7).
  HEAVY LAYER (`Heavy`, `main_read`)
     the DAG-ORC `Rig` (A.pair_table, backend.analyze_t1) applied to the 4 runs x modes, plus the rule-violation row.

THE RULE-VIOLATION ROW (G-VIOL; the amendment is the Master Mind's): the T1-kin tool carries no violation measure (NOTES.md "G-VIOL"). `VIOL-ANY` = the T0 eval's own function chain
(`v8f_mini_eval.py:149-157`: constraints_from_v9 -> C.violations -> any_violated) applied to the `cl` arm (the executed path, 30 knots in the t0 ego frame) of each rollout; `VIOL-ANY-OL` the same on the
tick-0 plan `ol`. Per window 0/1, paired `egoc - identity` through the same estimator. G-VIOL fires if `VIOL-ANY` is separated-worse at >= 3 of the 4 runs. It is switched on by AMENDMENT_1_GVIOL and is
a guard like G-WORSE: a firing is HARMS. The PREREG pin below is the PROMOTED text; when Amendment 1 lands, move PREREG_SHA256 to the amended file and keep the promoted one as PREREG_SHA256_PROMOTED.

    python speedsub_t1_read.py --static-check
    python speedsub_t1_read.py --dry-run                      the banked stand-ins through the heavy layer (identity - P10 vs DAGOACC_READ.json, K-ZERO, VIOL row); no rollout output is read
    python speedsub_t1_read.py --manifest runs_manifest.json --out SPEEDSUB_T1_READ.json     THE READ (rc 2 REFUSED: nothing decisive written | rc 0 outcome written | rc 4 runs missing)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
ART = "E:/Projects/TanitAD-artifacts"
ORC_DIR = f"{ART}/dagorc_rebase/read"

# ---------------------------------------------------------------------------------------------------------------- PREREG_V8F_SPEEDSUB_T1 (committed numbers)
PREREG_ID = "Project Steering/PREREG_V8F_SPEEDSUB_T1.md @ 842ec2ac (promoted 2026-10-28)"
PREREG_SHA256 = "5edf6bc0acf86192e271721d064c835e4ed5f22a31a1ee213f7ac7fc248d95c2"      # LF-normalised bytes of the 842ec2ac blob
PREREG_SHA256_PROMOTED = PREREG_SHA256
CLAIM_ROW = "H-V8F-SPEEDSUB-T1-1"
AMENDMENT_1_GVIOL = True                 # the Master Mind's G-VIOL (coordinator message 2026-10-28); the PREREG text does not carry it until Amendment 1 lands
F_ADE6 = 0.1207
BAR = F_ADE6 / 2.0                       # 0.06035: a HIT needs d <= -BAR (inclusive) and CI upper end < 0
TWO_F = 0.2415                           # PREREG_V8F_DAG_REREAD section 2 D1, as printed
RUNS = ("CTRL_s0", "CTRL_s1", "REP_s0", "REP_s1")
CTRL_RUNS = ("CTRL_s0", "CTRL_s1")
SEEDS = (0, 1)
MODES = ("identity", "egoc", "egoshufc")
BESIDE_MODES = ("identity_l", "egol")
N_WINDOWS, N_EPISODES, N_TICKS = 2399, 141, 3
N_BOOT = 2000
GUARD_RUNS_MIN = 3                       # G-WORSE and G-VIOL fire at >= 3 of the 4 runs
SHUF_SEED, SHUF_N = 20261029, 2399
REG_PERM_SHA = "64785f8ad0797b02bdf9845cedf6cf425f8759a2993da889cf33146a0e37ab64"      # SPEEDSUB's banked perm_sha256 for default_rng(20261029).permutation(2399)
CODE_IDX = [46, 56, 57, 58, 59]
TOL_K_ID, TOL_K_REC = 1e-4, 1e-5
REACH_EGOC_MIN, REACH_SHUF_MIN = 0.99, 0.95
CKPT_PREFIX = {"CTRL_s0": "67265ec1", "CTRL_s1": "90b01474", "REP_s0": "b03fd6ef", "REP_s1": "77853e56"}       # PREREG section 1
CKPT_BANKED = {"CTRL_s0": f"{ART}/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt", "CTRL_s1": f"{ART}/v8f-dagoacc/M2b_s1_dagoacc/ckpt_last.pt",
               "REP_s0": f"{ART}/v8f-dag-orc-oacc/M2b_s0_dagoaccorcrep/ckpt_last.pt", "REP_s1": f"{ART}/v8f-dag-orc-oacc/M2b_s1_dagoaccorcrep/ckpt_last.pt"}
BANKED_DUMP = {"CTRL_s0": f"{ART}/v8f-dagoacc-t1/dump_DAGOACC_s0_m10_causal_heads", "CTRL_s1": f"{ART}/v8f-dagoacc-t1/dump_DAGOACC_s1_m10_causal_heads",
               "REP_s0": "F:/TanitAD-scratch/dagorc_t1_out/dump_DAGORCREP_s0_m10_causal_heads", "REP_s1": "F:/TanitAD-scratch/dagorc_t1_out/dump_DAGORCREP_s1_m10_causal_heads"}
BANKED_TICKS = {"CTRL_s0": f"{ART}/v8f-dagoacc-t1/ticks/ticks_DAGOACC_s0_m10_causal_heads.npz", "CTRL_s1": f"{ART}/v8f-dagoacc-t1/ticks/ticks_DAGOACC_s1_m10_causal_heads.npz",
                "REP_s0": "F:/TanitAD-scratch/dagorc_t1_out/ticks/ticks_DAGORCREP_s0_m10_causal_heads.npz", "REP_s1": "F:/TanitAD-scratch/dagorc_t1_out/ticks/ticks_DAGORCREP_s1_m10_causal_heads.npz"}
OACC_READ_JSON = f"{ART}/v8f-dagoacc-t1/read/DAGOACC_READ.json"

LABELS = ("ADE6", "speed MAE", "along-track MAE", "accel MAE", "DK-VIOL", "DK-HW", "DK-TG", "heading MAE", "yaw-rate MAE", "curvature MAE", "cross-track MAE",
          "L1 lateral-accel MAE", "L2 curvature MAE (GT>=5.2)", "TAC-LAT", "TAC-LON", "S-POS@30", "S-POS@50", "S-POS@80", "M1 crawl share")          # Read A's 19 rows (DAG-ORC section 2.1)
VIOL, VIOL_OL = "VIOL-ANY", "VIOL-ANY-OL"
DIRECTION = {l: "lower" for l in LABELS + (VIOL, VIOL_OL)}
DIRECTION.update({"TAC-LAT": "higher", "TAC-LON": "higher"})
FAMILY = {"ADE6": "PRIMARY", "speed MAE": "LONGITUDINAL", "along-track MAE": "LONGITUDINAL", "accel MAE": "LONGITUDINAL", "DK-VIOL": "LONGITUDINAL", "DK-HW": "LONGITUDINAL",
          "DK-TG": "LONGITUDINAL", "heading MAE": "LATERAL", "yaw-rate MAE": "LATERAL", "curvature MAE": "LATERAL", "cross-track MAE": "LATERAL", "L1 lateral-accel MAE": "LATERAL",
          "L2 curvature MAE (GT>=5.2)": "LATERAL", "TAC-LAT": "TACTICAL", "TAC-LON": "TACTICAL", "S-POS@30": "STRATEGIC", "S-POS@50": "STRATEGIC", "S-POS@80": "STRATEGIC",
          "M1 crawl share": "crawl", VIOL: "RULES", VIOL_OL: "RULES"}
FLOOR_ROWS = ("ADE6", "yaw-rate MAE", "curvature MAE")
TIER_T1 = "T1-kin (self-action OPEN loop; m=10, causal bucket, dead-reckoned nav, logged scene, periodic, executor plan). Never 'closed loop', never T2."
ESTIMATOR = ("taniteval.ci.paired_episode_cluster_bootstrap, B=2000, seed 0, 141 episodes / 2,399 windows, per run, paired. It answers 'would another draw of EPISODES say this?'. The training-run "
             "question is answered by the four runs (two seeds x two replicates); the planner is deterministic, so there is no inference-seed term.")
OUTCOME_NAMES = ("REFUSED", "HARMS", "PASS", "PARTIAL", "NULL")
COMMITTED_NEXT = {"REFUSED": "fix the control; no row is read",
                  "HARMS": "not adopted. Next: PREREG section 8 (b), the per-cell attribution at T1-kin, which finds the cell that harms",
                  "PASS": "adopt the read-out as the DAG-OACC base's speed path: build it as a forward_infer speed sub-head (buffers absent by default, every earlier checkpoint bit-identical; its own "
                          "launch-gate checks) and read section 6.1",
                  "PARTIAL": "not adopted, and not called a lever effect (H-ESTIM-SEED-1). Next: section 8 (a), the history-aware causal read-out, on the runs that missed",
                  "NULL": "next: section 8 (a), the history-aware causal read-out"}


# ---- pure ---------------------------------------------------------------------------------------------------------------------------------------------------------
class RowNotComputed(Exception):
    pass


def computed(row):
    return row is not None and row.get("status") != "NOT COMPUTED"


def worsening(label, row):
    """Positive = worse, in the row's own units: an error row (lower is better) worsens by +delta, an agreement row (higher is better) by -delta."""
    if not computed(row):
        raise RowNotComputed(label)
    d = float(row["delta"])
    return d if DIRECTION[label] == "lower" else -d


def sep_worse(label, row):
    return bool(computed(row) and row["separated"] and worsening(label, row) > 0)


def sep_better(label, row):
    return bool(computed(row) and row["separated"] and worsening(label, row) < 0)


def is_hit(row):
    """PREREG section 6: a HIT at X = separated-better (CI upper end < 0) AND d_X <= -F/2 (inclusive, on the point)."""
    return bool(computed(row) and float(row["hi"]) < 0.0 and float(row["delta"]) <= -BAR)


def hits_by_run(ade_ei):
    return {r: is_hit(ade_ei[r]) for r in RUNS}


def count_sep_worse(rows_by_run, label):
    return [r for r in RUNS if sep_worse(label, rows_by_run[r].get(label))]


def g_worse(rows_ei, labels=LABELS):
    """G-WORSE: no Read-A row (ADE6 included) has `egoc - identity` separated-worse at >= 3 of the 4 runs. -> (fires, per-row run lists)."""
    per = {l: count_sep_worse(rows_ei, l) for l in labels}
    return any(len(v) >= GUARD_RUNS_MIN for v in per.values()), {l: v for l, v in per.items() if v}


def g_new(rows_ep, rows_ip, labels=LABELS):
    """G-NEW: no row r for which `identity - P10` is not separated-worse at a CTRL seed becomes separated-worse for `egoc - P10` at BOTH CTRL seeds.
    READING (the PREREG sentence admits two; this is the stricter, fires more often): r fires iff `egoc - P10` is separated-worse at both CTRL seeds AND `identity - P10` is NOT separated-worse at
    >= 1 of the two CTRL seeds. The other reading (identity not separated-worse at BOTH) is returned beside as `fires_if_identity_clear_at_both` and decides nothing."""
    fires, both_clear, detail = [], [], {}
    for l in labels:
        e = [sep_worse(l, rows_ep[r].get(l)) for r in CTRL_RUNS]
        i = [sep_worse(l, rows_ip[r].get(l)) for r in CTRL_RUNS]
        if all(e):
            detail[l] = {"egoc_P10_sep_worse": e, "identity_P10_sep_worse": i}
            if not all(i):
                fires.append(l)
            if not any(i):
                both_clear.append(l)
    return bool(fires), {"rows": fires, "fires_if_identity_clear_at_both": both_clear, "detail": detail}


def g_ade(ade_ei):
    """G-ADE: `egoc - identity` ADE6 is not separated-worse at any run."""
    bad = [r for r in RUNS if sep_worse("ADE6", ade_ei[r])]
    return bool(bad), bad


def g_viol(viol_ei):
    """G-VIOL (Amendment 1, the Master Mind's text): VIOL-ANY `egoc - identity` separated-worse at >= 3 of the 4 runs."""
    bad = [r for r in RUNS if sep_worse(VIOL, viol_ei.get(r))]
    return bool(AMENDMENT_1_GVIOL and len(bad) >= GUARD_RUNS_MIN), bad


def decide_speedsub(controls_ok, runs_present, hits, guards):
    """The PREREG section 6 table as ONE function. REFUSED first, then HARMS, then PASS; PASS and HARMS cannot both hold.
    ``hits[r]`` bool per run, ``guards`` = {"G-WORSE": bool fires, "G-NEW": ..., "G-ADE": ..., "G-VIOL": ...}. -> {"OUTCOME", "label", "flags"}.
    PASS: no guard fires and a hit at ALL four runs. PARTIAL: no guard, hits at both CTRL seeds, not at all four. NULL: anything else (labels NULL-ONE / NULL-FLAT describe, decide nothing)."""
    if not controls_ok or not runs_present:
        return {"OUTCOME": "REFUSED", "label": None, "flags": {"controls_ok": bool(controls_ok), "runs_present": bool(runs_present)}}
    fired = [g for g, v in guards.items() if v]
    n_hit = sum(bool(hits[r]) for r in RUNS)
    ctrl_both = all(hits[r] for r in CTRL_RUNS)
    if fired:
        out = "HARMS"
    elif n_hit == len(RUNS):
        out = "PASS"
    elif ctrl_both:
        out = "PARTIAL"
    else:
        out = "NULL"
    label = None
    if out == "NULL":
        label = "NULL-FLAT" if n_hit == 0 else "NULL-ONE"
    return {"OUTCOME": out, "label": label, "flags": {"controls_ok": True, "guards_fired": fired, "hits": {r: bool(hits[r]) for r in RUNS}, "n_hit": n_hit, "hits_at_both_ctrl": ctrl_both}}


def read_a(rows_ep):
    """PREREG section 6.1: Read A (PREREG_V8F_DAG_REREAD section 5) on `egoc - P10` at both CTRL seeds. D1: ADE6 separated-better with point <= -2F; D2: no row separated-worse; D3: each of the four
    families has a separated-better row; crawl rows pass iff neither L2 nor M1 is separated-worse at any CTRL seed. -> {outcome, per_seed}.
    DOMINANT needs D1-D3 at both seeds AND the crawl rows (RESULT_V8F_DAG_OACC section 1, A1); the unamended table's label is returned beside."""
    per = {}
    for r in CTRL_RUNS:
        rows = rows_ep[r]
        d1 = bool(sep_better("ADE6", rows.get("ADE6")) and float(rows["ADE6"]["delta"]) <= -TWO_F)
        worse = [l for l in LABELS if sep_worse(l, rows.get(l))]
        fams = {f: any(sep_better(l, rows.get(l)) for l in LABELS if FAMILY[l] == f) for f in ("PRIMARY", "LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")}
        per[r] = {"D1": d1, "D2": not worse, "D2_worse_rows": worse, "D3": all(fams.values()), "D3_families": fams, "M1_sep_worse": sep_worse("M1 crawl share", rows.get("M1 crawl share")),
                  "L2_sep_worse": sep_worse("L2 curvature MAE (GT>=5.2)", rows.get("L2 curvature MAE (GT>=5.2)"))}
    d1 = all(per[r]["D1"] for r in CTRL_RUNS)
    d2 = all(per[r]["D2"] for r in CTRL_RUNS)
    d3 = all(per[r]["D3"] for r in CTRL_RUNS)
    crawl_pass = not any(per[r]["M1_sep_worse"] or per[r]["L2_sep_worse"] for r in CTRL_RUNS)
    if not d1:
        unamended = "NO MARGIN"
    elif not (d2 and d3):
        unamended = "MARGIN, NOT DOMINANT"
    elif not crawl_pass:
        unamended = "DOMINANT, CRAWL OPEN"
    else:
        unamended = "DOMINANT"
    outcome = "DOMINANT" if (d1 and d2 and d3 and crawl_pass) else ("NOT DOMINANT: " + unamended)
    return {"outcome": outcome, "unamended_label": unamended, "D1_both": d1, "D2_both": d2, "D3_both": d3, "crawl_rows_pass": crawl_pass, "per_seed": per,
            "next": ("the dominance claim at T1-kin: re-render the evidence videos on it, report to the PI, ask for compute to scale (PREREG 6.1)" if outcome == "DOMINANT"
                     else "the substituted base is the base, and the open rows' lever is next; on crawl rows that is LAUNCH-INTENT re-based on the substituted base by its own amendment (PREREG 6.1)")}


def floor_rule(g_id, g_c):
    """PREREG section 6.2. ``g_id[row]`` / ``g_c[row]`` = point of REP^identity - CTRL^identity / REP^egoc - CTRL^egoc (paired, same windows) on FLOOR_ROWS.
    COLLAPSE iff g_id(r) > 0 and g_c(r) <= 0.5 * g_id(r) on ALL three rows; otherwise NOT COLLAPSE (the floors stay DAG-ORC's)."""
    per = {r: bool(g_id[r] > 0.0 and g_c[r] <= 0.5 * g_id[r]) for r in FLOOR_ROWS}
    return {"verdict": "COLLAPSE" if all(per.values()) else "NOT COLLAPSE", "per_row": per, "g_id": {r: float(g_id[r]) for r in FLOOR_ROWS}, "g_c": {r: float(g_c[r]) for r in FLOOR_ROWS},
            "consequence": ("later arms on the substituted base take RF from the egoc replicate pair" if all(per.values()) else "the floors stay as DAG-ORC's")}


def is_exact_zero(rows):
    """K-ZERO: every computed row of `identity - identity` reads delta == lo == hi == 0.0 exactly."""
    bad = {l: (r.get("delta"), r.get("lo"), r.get("hi")) for l, r in rows.items() if computed(r) and not (r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0)}
    return not bad and any(computed(r) for r in rows.values()), bad


def k_reg(ade_si):
    """K-REG: `egoshufc - identity` ADE6 is separated-worse at all four runs."""
    return all(sep_worse("ADE6", ade_si[r]) for r in RUNS), {r: (None if not computed(ade_si[r]) else {k: ade_si[r][k] for k in ("delta", "lo", "hi", "separated")}) for r in RUNS}


def per_window_ade6(cl, g):
    return np.linalg.norm(np.asarray(cl, np.float64) - np.asarray(g, np.float64), axis=-1).mean(1)


def k_id(cl_run, cl_banked, g, tol=TOL_K_ID):
    """K-ID: the identity rollout's per-window ADE6 equals the banked own rollout's within `tol` m; max reported (plus the max over every `cl` coordinate)."""
    if np.shape(cl_run) != np.shape(cl_banked) or np.shape(g)[0] != np.shape(cl_banked)[0]:
        return False, {"shape_mismatch": [list(np.shape(cl_run)), list(np.shape(cl_banked)), list(np.shape(g))], "tol": tol}
    d = np.abs(per_window_ade6(cl_run, g) - per_window_ade6(cl_banked, g))
    dc = float(np.abs(np.asarray(cl_run, np.float64) - np.asarray(cl_banked, np.float64)).max())
    return bool(d.max() <= tol and len(d) == len(cl_banked)), {"max_abs_ade6_diff_m": float(d.max()), "max_abs_cl_diff_m": dc, "n": int(len(d)), "tol": tol}


def k_nonspeed(recs):
    """K-NONSPEED: at every (window, tick) of every rollout, the 55 non-speed code dims O read equal T's own: max |delta| = 0.0 exactly."""
    mx = {k: float(np.max(r["nonspeed_dev"])) if len(r["nonspeed_dev"]) else float("nan") for k, r in recs.items()}
    return bool(mx and all(v == 0.0 for v in mx.values())), mx


def k_rec(rec, predict_raw, tol=TOL_K_REC):
    """K-REC: the cells O read in `egoc` equal `mlp_ego_c` on the recorded ego block, standardised by the checkpoint's cons_mu / cons_sd, <= tol. ``predict_raw(ego) -> [R, 5]`` physical."""
    m = rec["meta"]
    mu, sd = np.asarray(m["cons_mu"], np.float64), np.asarray(m["cons_sd"], np.float64)
    want = ((predict_raw(rec["ego"]) - mu) / sd).astype(np.float32)
    d_used, d_o = float(np.abs(want - rec["used_std"]).max()), float(np.abs(want - rec["o_std"]).max())
    return bool(max(d_used, d_o) <= tol), {"max_abs_used": d_used, "max_abs_o_read": d_o, "tol": tol, "n_rows": int(len(want))}


def moved_fraction(a, b, tol=1e-6):
    return float((np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max(1) > tol).mean())


def k_reach(egoc, egoshufc):
    """K-REACH: `egoc` moves the cells away from T's own on >= 99 % of (window, tick); `egoshufc` differs from `egoc` on >= 95 %."""
    f1 = moved_fraction(egoc["o_std"], egoc["own_std"])
    f2 = moved_fraction(egoshufc["o_std"], egoc["o_std"])
    return bool(f1 >= REACH_EGOC_MIN and f2 >= REACH_SHUF_MIN), {"egoc_moved_fraction": f1, "egoshufc_differs_from_egoc_fraction": f2}


def perm_sha256(perm):
    return hashlib.sha256(np.asarray(perm).astype(np.int64).tobytes()).hexdigest()


def perm_fixed_ok(perm):
    """K-SHUF's fixed-point rule: at most max(3, 1 %) windows keep their own cells (the registered permutation has 0)."""
    n = len(perm)
    nfix = int((np.asarray(perm) == np.arange(n)).sum())
    return bool(nfix <= max(3, 0.01 * n)), nfix


def k_shuf(rec):
    """K-SHUF: the permutation recorded by the wrapper is the registered one (default_rng(20261029).permutation(2399), sha 64785f8a...), 2,399 long, fixed points <= max(3, 1 %), and every row read the
    cells of window perm[w] (src_win == perm[win])."""
    m = rec["meta"]
    perm = np.random.default_rng(SHUF_SEED).permutation(SHUF_N)
    fix_ok, nfix = perm_fixed_ok(perm)
    rows_ok = bool(np.array_equal(np.asarray(rec["src_win"]), perm[np.asarray(rec["win"])])) if len(rec["win"]) else False
    ok = bool(m.get("perm_sha256") == perm_sha256(perm) == REG_PERM_SHA and m.get("perm_seed") == SHUF_SEED and m.get("perm_n") == SHUF_N and fix_ok and rows_ok)
    return ok, {"recorded_sha256": m.get("perm_sha256"), "recomputed_sha256": perm_sha256(perm), "registered_sha256": REG_PERM_SHA, "n": m.get("perm_n"), "n_fixed": nfix, "rows_use_perm_of_window": rows_ok}


def k_floor(fl_run, fl_banked):
    """K-FLOOR: the P10 floor arm equals the banked floor arm bit for bit."""
    a, b = np.asarray(fl_run), np.asarray(fl_banked)
    return bool(a.shape == b.shape and a.tobytes() == b.tobytes()), {"shape": list(a.shape), "max_abs_diff": float(np.abs(a - b).max()) if a.shape == b.shape else None}


def k_sha(meta, run, pins):
    """K-SHA: one rollout's recorded checkpoint / read-out / tool / wrapper / instrument equal their pins (and the checkpoint starts with the PREREG's printed prefix)."""
    mode = meta["mode"]
    want = {"ckpt_sha256": pins["ckpts"][run]["sha256"], "tool_sha256": pins["tool_sha256"], "wrapper_sha256": pins["wrapper_sha256"], "instrument_sha256": pins["instrument_sha256"]}
    if mode in ("egoc", "egoshufc"):
        want["readout_sha256"] = pins["readout_c"]["sha256"]
    elif mode == "egol":
        want["readout_sha256"] = pins["readout_banked"]["sha256"]
    bad = {k: {"got": meta.get(k), "pin": v} for k, v in want.items() if meta.get(k) != v}
    if not str(meta.get("ckpt_sha256", "")).startswith(CKPT_PREFIX[run]):
        bad["ckpt_prefix"] = {"got": str(meta.get("ckpt_sha256", ""))[:8], "pin": CKPT_PREFIX[run]}
    return not bad, bad


def families_report(tabs):
    """PREREG section 7, per run and per contrast: the rows grouped by family. ``tabs`` = {contrast name: {run: {label: row}}}. STRATEGIC: S-POS within run (egoc - P10) only."""
    out = {}
    for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC", "RULES"):
        labs = [l for l in LABELS + (VIOL, VIOL_OL) if FAMILY[l] == fam]
        out[fam] = {"tier": TIER_T1, "estimator": "paired episode-cluster bootstrap (B=2000, seed 0)",
                    "rows": {l: {cn: {r: (tab[r][l] if tab.get(r) and computed(tab[r].get(l)) and not isinstance(tab[r][l], str) else "NOT COMPUTED") for r in RUNS} for cn, tab in tabs.items()} for l in labs}}
        if fam == "STRATEGIC":
            out[fam]["note"] = "S-POS is within run (S - CTRV, ticks >= 1); paired S-POS across arms is NOT COMPUTED (PREREG section 7). STRATEGIC is reported, not paused, for v8F."
        if fam == "TACTICAL":
            out[fam]["note"] = "goal-head F1 at T1-kin: NOT COMPUTABLE here (the paired tool reads eval dumps, not tick dumps); TAC-LAT / TAC-LON are the 3-way agreement of the executed track with the human's."
        if fam == "LONGITUDINAL":
            out[fam]["note"] = "the anti-echo block: NOT BUILT (the DAG-ORC reader carries none; PREREG section 7 allows 'NOT BUILT with the reason')."
    return out


def row_from_pair(res):
    """The estimator's dict -> the row schema the guards read."""
    return {"delta": float(res["delta"]), "lo": float(res["lo"]), "hi": float(res["hi"]), "separated": bool(res["separated"]), "n": f"{res.get('n_windows')}/{res.get('n_episodes')}"}


# ---- evidence of the inputs (static) ------------------------------------------------------------------------------------------------------------------------------
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def prereg_problems(path, pin=None):
    """The file is the pinned PREREG (LF-normalised bytes) and its decision table / hit rule / guard sentences are the ones coded here."""
    pin = pin or PREREG_SHA256
    raw = open(path, "rb").read().replace(b"\r\n", b"\n")
    bad = []
    if hashlib.sha256(raw).hexdigest() != pin:
        bad.append(f"{path} has sha256 {hashlib.sha256(raw).hexdigest()[:12]}..., the pin is {pin[:12]}...")
    txt = raw.decode("utf-8")
    for need in ("| REFUSED | a control fails, or a run is missing |", "| **HARMS** | a guard fires (G-ADE, G-WORSE or G-NEW) |", "| **PASS** | no guard; a hit at **all four** runs |",
                 "| **PARTIAL** | no guard; hits at both CTRL seeds, not at all four runs |", "separated-better (CI upper end < 0) **and** `d_X ≤ −0.06035`",
                 "**G-ADE:** `egoc − identity` ADE6 is not separated-worse at any run.", "separated-worse at ≥ 3 of the 4 runs"):
        if need not in txt:
            bad.append(f"the PREREG does not carry: {need!r}")
    return bad


# ---- heavy --------------------------------------------------------------------------------------------------------------------------------------------------------
def load_rec(path):
    z = np.load(path, allow_pickle=False)
    r = {k: z[k] for k in z.files if k != "meta_json"}
    r["meta"] = json.loads(str(z["meta_json"]))
    return r


class Heavy:
    """The DAG-ORC `Rig` applied to the 4 runs. Built once; every table goes through `A.pair_table` / `A.table` unchanged."""

    def __init__(self, orc_dir=ORC_DIR, work_dir=None, n_boot=N_BOOT):
        sys.path.insert(0, orc_dir)
        import dag_orc_read as O
        self.O = O
        a = O.default_args(work_dir=work_dir or f"{HERE}/_work", n_boot=n_boot)
        self.rig, self.a, self.n_boot = O.Rig(a), a, n_boot
        self.A, self.R = self.rig.A, self.rig.A.R
        self.viol_ctx = None

    # -- the rule-violation measure of the T0 eval, applied to a dump arm
    def viol_setup(self):
        if self.viol_ctx is None:
            tree = self.a.tool_root
            for p in (f"{tree}/stack", f"{tree}/taniteval"):
                if p not in sys.path:
                    sys.path.insert(0, p)
            from tanitad.data import v8f_mini_data as D
            from taniteval import v8f_consistency as C
            raw = os.path.dirname(self.a.eval_join)
            grid = json.load(open(self.a.eval_grid, encoding="utf-8"))
            J = D.load_join(self.a.eval_join)
            X = D.eval_context(grid, f"{raw}/eval141_kin.npz", J, kin_mode="yaw")
            self.viol_ctx = {"C": C, "gtc": C.constraints_from_v9(D.v9_label_dict(X), X["limit_ms"], X["lead_gap_h"], X["s_h"]), "ws": X["t"], "gt": X["gt"]}
        return self.viol_ctx

    def viol_any(self, plan_xy, ws):
        v = self.viol_setup()
        if not np.array_equal(np.asarray(ws), v["ws"]):
            raise ValueError("the dump's windows are not the eval_context's windows")
        return v["C"].violations(np.asarray(plan_xy, np.float64), v["gtc"])["any_violated"].astype(np.float64)

    def viol_row(self, dump_x, dump_y, arm="cl"):
        a, b = self.viol_any(dump_x[arm], dump_x["ws"]), self.viol_any(dump_y[arm], dump_y["ws"])
        res = self.R.paired(a, b, dump_x["eid"])
        r = row_from_pair(res)
        r.update({"mean_arm": float(a.mean()), "mean_ref": float(b.mean()), "kind": f"any-rule violation share of the `{arm}` arm (T0 eval's C.violations), paired per window"})
        return r

    # -- Read A tables for a family of two runs (seed 0, seed 1)
    def tables(self, dumps_by_seed, packs_by_seed, x, y, ref):
        """-> {seed: {label: row}} for X - Y; ``dumps_by_seed[s]`` = {mode name: dump}; Y = "P10" reads the `fl` arm. S-POS only within run (Y == P10)."""
        cj, tj, cache = {}, {}, {}
        for s in SEEDS:
            cache[s] = {"cx": self.A.tacdk_context(ref[s]), "P": {}}
            cj[s], tj[s] = self.rig.pair_table(dumps_by_seed[s], cache[s], x, y, ref[s])
        within = y == "P10"
        dag = {"beside": {"per_seed": {f"s{s}": {"per_run_t1heads": {x: packs_by_seed[s][x].extra}} for s in SEEDS}}} if within else {}
        return self.A.table(cj, tj, dag, x if within else None)


def make_viol_rows(heavy, dumps, x, y, arm="cl"):
    return heavy.viol_row(dumps[x], dumps[y], arm)


# ---- main read ----------------------------------------------------------------------------------------------------------------------------------------------------
def load_manifest(path):
    m = json.load(open(path, encoding="utf-8"))
    missing = []
    for r in RUNS:
        run = (m.get("runs") or {}).get(r)
        if not run:
            missing.append(f"{r}: no entry")
            continue
        for mode in MODES:
            e = (run.get("modes") or {}).get(mode)
            if not e:
                missing.append(f"{r}/{mode}: no entry")
                continue
            for k in ("dump_dir", "ticks", "json", "rec"):
                if not e.get(k) or not os.path.exists(e[k]):
                    missing.append(f"{r}/{mode}/{k}: {e.get(k)!r} is missing")
    return m, missing


def main_read(a):
    t_start = time.time()
    man, missing = load_manifest(a.manifest)
    if missing:
        print("SPEEDSUB-T1 READ: runs missing (rc 4, nothing decisive written):", *missing, sep="\n  ")
        return 4
    bad = prereg_problems(a.prereg, a.prereg_sha) if a.prereg else ["--prereg is required"]
    pins = json.load(open(man["pins"], encoding="utf-8"))
    if bad:
        print("SPEEDSUB-T1 READ REFUSED (nothing written): the PREREG pin:", *bad, sep="\n  ", file=sys.stderr)
        return 2
    H = Heavy(a.orc_dir, a.work_dir, a.n_boot)
    ctl, detail = {}, {}

    def add(name, ok, det=None):
        ctl[name] = bool(ok)
        detail[name] = det

    # ---- load everything
    recs, dumps, packs = {}, {}, {}
    for r in RUNS:
        dumps[r], packs[r], recs[r] = {}, {}, {}
        run = man["runs"][r]
        for mode in MODES:
            e = run["modes"][mode]
            dumps[r][mode] = H.A.dump(e["dump_dir"])
            recs[r][mode] = load_rec(e["rec"])
            packs[r][mode] = H.rig.backend.analyze_t1({"tag": e["tag"], "ticks": e["ticks"], "dump_dir": e["dump_dir"], "json": e["json"]}, e["tag"], int(run["seed"]))
        banked = H.A.dump(BANKED_DUMP[r])
        dumps[r]["banked"] = banked
        dumps[r]["P10"] = dumps[r]["identity"]
    # ---- controls on the rollouts
    for r in RUNS:
        for mode in MODES:
            ok, det = k_sha(recs[r][mode]["meta"], r, pins)
            add(f"K-SHA:{r}:{mode}", ok, det)
            add(f"K-ALIGN:{r}:{mode}", recs[r][mode]["meta"]["aligned"] and recs[r][mode]["meta"]["tool_rc"] == 0, recs[r][mode]["meta"]["alignment_with_tool_ticks"])
            for k, (okp, detp) in packs[r][mode].controls.items():
                add(f"{k}:{r}:{mode}", okp, detp)
            d = dumps[r][mode]
            ref = dumps[r]["banked"]
            same = all(np.array_equal(d[k], ref[k]) for k in ("clip_index", "ws", "eid", "g", "v0")) and len(d["eid"]) == N_WINDOWS and len(np.unique(d["eid"])) == N_EPISODES
            add(f"K-WIN:{r}:{mode}", same, "clip_index, ws, eid, g, v0 bit-equal to the banked rollout; 2399 windows / 141 episodes")
            add(f"K-FLOOR:{r}:{mode}", *k_floor(d["fl"], ref["fl"]))
        add(f"K-ID:{r}", *k_id(dumps[r]["identity"]["cl"], dumps[r]["banked"]["cl"], dumps[r]["banked"]["g"]))
    add("K-NONSPEED", *k_nonspeed({f"{r}:{m}": recs[r][m] for r in RUNS for m in MODES}))
    ro = sr_load_readout(H, pins["readout_c"]["path"], "mlp_ego_c", a.instr)
    for r in RUNS:
        add(f"K-REC:{r}", *k_rec(recs[r]["egoc"], lambda ego, ro=ro: ro.predict_raw(None, ego)))
        add(f"K-REACH:{r}", *k_reach(recs[r]["egoc"], recs[r]["egoshufc"]))
        add(f"K-SHUF:{r}", *k_shuf(recs[r]["egoshufc"]))
        d_plan = float((np.abs(dumps[r]["egoc"]["cl"] - dumps[r]["identity"]["cl"]).reshape(N_WINDOWS, -1).max(1) > 0).mean())
        add(f"K-REACH-PLAN:{r}", d_plan >= REACH_EGOC_MIN, {"egoc_cl_differs_from_identity_fraction": d_plan})
    # ---- tables
    fam = {"CTRL": CTRL_RUNS, "REP": ("REP_s0", "REP_s1")}
    tabs = {k: {} for k in ("egoc-identity", "egoc-P10", "identity-P10", "egoshufc-identity", "identity-identity")}
    for f, rr in fam.items():
        db = {s: dumps[rr[s]] for s in SEEDS}
        pk = {s: packs[rr[s]] for s in SEEDS}
        ref = {s: dumps[rr[s]]["identity"] for s in SEEDS}
        for name, (x, y) in {"egoc-identity": ("egoc", "identity"), "egoc-P10": ("egoc", "P10"), "identity-P10": ("identity", "P10"), "egoshufc-identity": ("egoshufc", "identity"),
                             "identity-identity": ("identity", "identity")}.items():
            t = H.tables(db, pk, x, y, ref)
            for s in SEEDS:
                tabs[name][rr[s]] = t[s]
    for r in RUNS:
        for cn, (x, y) in {"egoc-identity": ("egoc", "identity"), "egoc-P10": ("egoc", "P10"), "identity-P10": ("identity", "P10"), "egoshufc-identity": ("egoshufc", "identity"),
                           "identity-identity": ("identity", "identity")}.items():
            d = dumps[r]
            tabs[cn][r][VIOL] = H.viol_row(d[x] if x != "P10" else d["identity"], d[y] if y != "P10" else {**d["identity"], "cl": d["identity"]["fl"]})
            tabs[cn][r][VIOL_OL] = H.viol_row(d[x], d[y], "ol") if y != "P10" else {"status": "NOT COMPUTED"}
    for r in RUNS:
        ok, det = is_exact_zero(tabs["identity-identity"][r])
        add(f"K-ZERO:{r}", ok, det)
    add("K-REG", *k_reg({r: tabs["egoshufc-identity"][r].get("ADE6") for r in RUNS}))
    controls_ok = all(ctl.values())
    # ---- the decision
    ade_ei = {r: tabs["egoc-identity"][r]["ADE6"] for r in RUNS}
    viol_ei = {r: tabs["egoc-identity"][r][VIOL] for r in RUNS}
    gw, gw_d = g_worse(tabs["egoc-identity"])
    gn, gn_d = g_new(tabs["egoc-P10"], tabs["identity-P10"])
    ga, ga_d = g_ade(ade_ei)
    gv, gv_d = g_viol(viol_ei)
    guards = {"G-WORSE": gw, "G-NEW": gn, "G-ADE": ga, "G-VIOL": gv}
    hits = hits_by_run(ade_ei)
    dec = decide_speedsub(controls_ok, True, hits, guards)
    res = {"OUTCOME": dec["OUTCOME"], "label": dec["label"], "committed_next": COMMITTED_NEXT[dec["OUTCOME"]], "flags": dec["flags"], "claim_row": CLAIM_ROW, "tier": TIER_T1, "estimator": ESTIMATOR,
           "controls": ctl, "controls_detail": detail, "controls_PASS": controls_ok, "guards": {"G-WORSE": {"fires": gw, "rows_and_runs": gw_d}, "G-NEW": {"fires": gn, **gn_d}, "G-ADE": {"fires": ga, "runs": ga_d},
                                                                                                   "G-VIOL": {"fires": gv, "runs": gv_d, "active": AMENDMENT_1_GVIOL}},
           "d_ADE6_egoc_minus_identity": {r: {k: ade_ei[r][k] for k in ("delta", "lo", "hi", "separated")} for r in RUNS}, "bar": BAR, "hits": hits}
    if dec["OUTCOME"] == "PASS":
        res["read_A_on_CTRL"] = read_a(tabs["egoc-P10"])
    # ---- 6.2 the floor rule (beside)
    cross = {}
    for s in SEEDS:
        sd = {"REP_id": dumps[f"REP_s{s}"]["identity"], "CTRL_id": dumps[f"CTRL_s{s}"]["identity"], "REP_c": dumps[f"REP_s{s}"]["egoc"], "CTRL_c": dumps[f"CTRL_s{s}"]["egoc"]}
        ref = dumps[f"CTRL_s{s}"]["identity"]
        g_id, g_c = {}, {}
        for (x, y), tgt in ((("REP_id", "CTRL_id"), g_id), (("REP_c", "CTRL_c"), g_c)):
            cj0, _ = H.rig.pair_table(sd, {"cx": H.A.tacdk_context(ref), "P": {}}, x, y, ref)
            for lab in FLOOR_ROWS:
                tgt[lab] = float(cj0[lab]["delta"])
        cross[f"s{s}"] = floor_rule(g_id, g_c)
    res["floor_rule_H_V8F_SPEEDSUB_T1_2"] = {"s0 (decisive, beside)": cross["s0"], "s1 (reported)": cross["s1"]}
    res["tables"] = {cn: {r: {l: (row if not isinstance(row, dict) else {k: row.get(k) for k in ("delta", "lo", "hi", "separated", "n", "mean_arm", "mean_ref", "status")}) for l, row in tab[r].items()
                                if not str(l).startswith("_")} for r in RUNS} for cn, tab in tabs.items()}
    res["four_families"] = families_report({"egoc-identity": tabs["egoc-identity"], "egoc-P10": tabs["egoc-P10"]})
    res["provenance"] = {"prereg": PREREG_ID, "prereg_sha256": PREREG_SHA256, "reader_sha256": sha256_file(os.path.abspath(__file__)), "orc_reader": f"{a.orc_dir}/dag_orc_read.py",
                         "manifest_sha256": sha256_file(a.manifest), "pins_sha256": sha256_file(man["pins"]), "wall_s": round(time.time() - t_start, 1), "evidence_class": "MEASURED (ours; artifact paths in the manifest)"}
    if not controls_ok:
        print("SPEEDSUB-T1 READ REFUSED: failed controls", [k for k, v in ctl.items() if not v])
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"ZZSPEEDSUB-T1-READ-DONE {res['OUTCOME']} {res['label'] or ''} -> {a.out}", flush=True)
    return 2 if res["OUTCOME"] == "REFUSED" else 0


def sr_load_readout(heavy, path, key, instr):
    import importlib.util
    spec = importlib.util.spec_from_file_location("sr_instr_reader", instr)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.load_readouts(path)[key]


def main_dry(a):
    """The banked stand-ins through the heavy layer, on DAG-OACC's own banked rollouts (CTRL s0 / s1): (1) the heads pipeline `analyze_t1` on each banked run, (2) `identity - P10` through `Heavy.tables` against the
    banked DAGOACC_READ.json Read-A table (all 19 rows, tolerance 1e-4 = the 4 dp it prints), (3) K-ZERO (`identity - identity` exactly 0.0 on every row), (4) the VIOL row for the floor and the tick-0 plan.
    No rollout of this experiment is read."""
    t0 = time.time()
    H = Heavy(a.orc_dir, a.work_dir, a.n_boot)
    banked = json.load(open(OACC_READ_JSON, encoding="utf-8"))
    dumps = {s: H.A.dump(BANKED_DUMP[f"CTRL_s{s}"]) for s in SEEDS}
    packs = {s: {"identity": H.rig.backend.analyze_t1({"tag": f"DAGOACC_s{s}_m10_causal_heads", "ticks": BANKED_TICKS[f"CTRL_s{s}"], "dump_dir": BANKED_DUMP[f"CTRL_s{s}"],
                                                       "json": f"{ART}/v8f-dagoacc-t1/t1v8f_DAGOACC_s{s}_m10_causal_heads.json"}, f"DAGOACC_s{s}_m10_causal_heads", s)} for s in SEEDS}
    db = {s: {"identity": dumps[s], "P10": dumps[s]} for s in SEEDS}
    tab = H.tables(db, packs, "identity", "P10", dumps)
    zt = H.tables(db, packs, "identity", "identity", dumps)
    diffs, missing = {}, []
    for s in SEEDS:
        bt = banked["rows"][f"s{s}"]["READ_A"]["table_DAGOACC_minus_P10"]
        for l in LABELS:
            if not computed(tab[s].get(l)) or not computed(bt.get(l)):
                missing.append(f"s{s}:{l}")
                continue
            diffs[f"s{s}:{l}"] = max(abs(float(tab[s][l][k]) - float(bt[l][k])) for k in ("delta", "lo", "hi"))
            if bool(tab[s][l]["separated"]) != bool(bt[l]["separated"]):
                diffs[f"s{s}:{l}"] = float("inf")
    zero = {f"s{s}": is_exact_zero(zt[s]) for s in SEEDS}
    viol = {f"s{s}": {"floor_row": H.viol_row({**dumps[s], "cl": dumps[s]["fl"]}, dumps[s]), "viol_cl_mean": float(H.viol_any(dumps[s]["cl"], dumps[s]["ws"]).mean()),
                      "viol_ol_mean": float(H.viol_any(dumps[s]["ol"], dumps[s]["ws"]).mean()), "viol_fl_mean": float(H.viol_any(dumps[s]["fl"], dumps[s]["ws"]).mean())} for s in SEEDS}
    out = {"identity_minus_P10_vs_banked_max_abs_diff": max(diffs.values()) if diffs else None, "n_rows_compared": len(diffs), "rows_not_comparable": missing, "per_row": diffs,
           "k_zero_identity_identity": {k: v[0] for k, v in zero.items()}, "k_zero_nonzero_rows": {k: v[1] for k, v in zero.items()}, "viol": viol, "wall_s": round(time.time() - t0, 1),
           "banked_read_json": OACC_READ_JSON, "banked_read_json_sha256": sha256_file(OACC_READ_JSON)}
    with open(a.out, "w", encoding="utf-8", newline=chr(10)) as f:
        json.dump(out, f, indent=1, default=float)
    ok = out["n_rows_compared"] >= 15 and out["identity_minus_P10_vs_banked_max_abs_diff"] is not None and out["identity_minus_P10_vs_banked_max_abs_diff"] <= 1e-4 and all(out["k_zero_identity_identity"].values())
    print("ZZSPEEDSUB-T1-READ-DRY", "OK" if ok else "NOT-OK", json.dumps({k: v for k, v in out.items() if k not in ("per_row", "k_zero_nonzero_rows")}, default=float)[:1800], flush=True)
    return 0 if ok else 1


def load_dump_light(d):
    """A rollout's per-episode dumps concatenated in episode order (numpy only; the Rig is not needed for the light controls)."""
    import glob
    files = sorted(glob.glob(os.path.join(d, "ep*.npz")))
    arrs = {}
    for f in files:
        with np.load(f) as z:
            for k in ("g", "cl", "ol", "fl", "ha0", "v0", "eid", "clip_index", "ws"):
                arrs.setdefault(k, []).append(z[k])
    return {k: np.concatenate(v) for k, v in arrs.items()}


def light_controls(run, recs, dumps, pins, ro, need_banked=True):
    """The controls that need no Rig. ``recs`` / ``dumps`` = {mode: ...}. -> (ok, {name: {ok, detail}})."""
    ctl = {}

    def add(name, ok, det=None):
        ctl[name] = {"ok": bool(ok), "detail": det}

    add("K-NONSPEED", *k_nonspeed({f"{run}:{m}": r for m, r in recs.items()}))
    for m, r in recs.items():
        add(f"K-SHA:{m}", *k_sha(r["meta"], run, pins))
        add(f"K-ALIGN:{m}", r["meta"]["aligned"] and r["meta"]["tool_rc"] == 0, r["meta"]["alignment_with_tool_ticks"])
    if "egoc" in recs:
        add("K-REC", *k_rec(recs["egoc"], lambda ego: ro.predict_raw(None, ego)))
    if "egoc" in recs and "egoshufc" in recs:
        add("K-REACH", *k_reach(recs["egoc"], recs["egoshufc"]))
    elif "egoc" in recs:
        f = moved_fraction(recs["egoc"]["o_std"], recs["egoc"]["own_std"])
        add("K-REACH", f >= REACH_EGOC_MIN, {"egoc_moved_fraction": f, "note": "egoshufc not rolled (smoke)"})
    if "egoshufc" in recs:
        add("K-SHUF", *k_shuf(recs["egoshufc"]))
    if need_banked and "identity" in dumps:
        banked = load_dump_light(BANKED_DUMP[run])
        for m, d in dumps.items():
            add(f"K-FLOOR:{m}", *k_floor(d["fl"], banked["fl"]))
            add(f"K-WIN:{m}", all(np.array_equal(d[k], banked[k]) for k in ("clip_index", "ws", "eid", "g", "v0")), "windows and GT bit-equal to the banked rollout")
        add("K-ID", *k_id(dumps["identity"]["cl"], banked["cl"], banked["g"]))
    return all(v["ok"] for v in ctl.values()), ctl


def main_light(a, kind):
    """--preflight (CTRL s0, the full rollouts) or --smoke-check (the 200-window smoke: K-NONSPEED, K-REC, K-REACH, K-SHA only; K-ID / K-FLOOR need the full grid)."""
    pins = json.load(open(a.pins, encoding="utf-8"))
    ro = sr_load_readout(None, pins["readout_c"]["path"], "mlp_ego_c", a.instr)
    modes = ("identity", "egoc", "egoshufc") if kind == "preflight" else ("identity", "egoc")
    root = os.path.join(a.out_root, "" if kind == "preflight" else "SMOKE", a.run)
    recs, dumps = {}, {}
    for m in modes:
        tag = f"SST1_{'SMOKE_' if kind == 'smoke' else ''}{a.run}_{m}"
        p = os.path.join(root, m, "ticks", f"speedsub_{tag}.npz")
        if not os.path.isfile(p):
            print(f"ZZSPEEDSUB-T1-{kind.upper()}-STOP missing {p}")
            return 2
        recs[m] = load_rec(p)
        if kind == "preflight":
            dumps[m] = load_dump_light(os.path.join(root, m, f"dump_{tag}"))
    ok, ctl = light_controls(a.run, recs, dumps, pins, ro, need_banked=(kind == "preflight"))
    out = {"kind": kind, "run": a.run, "controls_PASS": ok, "controls": ctl, "reader_sha256": sha256_file(os.path.abspath(__file__))}
    with open(a.out, "w", encoding="utf-8", newline=chr(10)) as f:
        json.dump(out, f, indent=1, default=float)
    print(f"ZZSPEEDSUB-T1-{kind.upper()}-{'OK' if ok else 'STOP'}", json.dumps({k: v["ok"] for k, v in ctl.items()}), flush=True)
    return 0 if ok else 1


def make_manifest(a):
    """The standard layout OUT/<run>/<mode>/{dump_<tag>, ticks/ticks_<tag>.npz, ticks/speedsub_<tag>.npz, t1v8f_<tag>.json} -> runs_manifest.json (existence is checked by the read itself)."""
    man = {"schema": "speedsub-t1-manifest-v1", "pins": a.pins, "runs": {}}
    for r in RUNS:
        fam, seed = r.split("_s")[0], int(r.split("_s")[1])
        modes = {}
        for m in MODES + (BESIDE_MODES if fam == "CTRL" else ()):
            tag = f"SST1_{r}_{m}"
            d = f"{a.out_root}/{r}/{m}"
            modes[m] = {"tag": tag, "out_dir": d, "dump_dir": f"{d}/dump_{tag}", "ticks": f"{d}/ticks/ticks_{tag}.npz", "json": f"{d}/t1v8f_{tag}.json", "rec": f"{d}/ticks/speedsub_{tag}.npz"}
        man["runs"][r] = {"family": fam, "seed": seed, "banked": {"dump_dir": BANKED_DUMP[r], "ticks": BANKED_TICKS[r]}, "modes": modes}
    with open(a.out, "w", encoding="utf-8", newline=chr(10)) as f:
        json.dump(man, f, indent=1)
    print("manifest written", a.out)
    return 0


def main_beside(a):
    """The beside logged lane (CTRL only, decides nothing): `egol - identity_l` Read-A rows per CTRL seed through the same machinery. Bridges back to SPEEDSUB under the declared open channel
    (D-SPEEDBAND-EGO-ECHO-1); never a capability row."""
    man = json.load(open(a.manifest, encoding="utf-8"))
    H = Heavy(a.orc_dir, a.work_dir, a.n_boot)
    dumps = {s: {m: H.A.dump(man["runs"][f"CTRL_s{s}"]["modes"][m]["dump_dir"]) for m in BESIDE_MODES} for s in SEEDS}
    tab = H.tables(dumps, {s: {} for s in SEEDS}, "egol", "identity_l", {s: dumps[s]["identity_l"] for s in SEEDS})
    keep = ("delta", "lo", "hi", "separated", "mean_arm", "mean_ref", "status")
    out = {"lane": "--limit logged (declared OPEN channel: the logged speed bucket is the bucket of the ego's own realised future max speed, D-SPEEDBAND-EGO-ECHO-1)", "tier": TIER_T1,
           "role": "beside; decides nothing; never a capability row",
           "egol_minus_identity_l": {f"CTRL_s{s}": {l: {k: r.get(k) for k in keep} for l, r in tab[s].items() if not str(l).startswith("_")} for s in SEEDS}}
    with open(a.out, "w", encoding="utf-8", newline=chr(10)) as f:
        json.dump(out, f, indent=1, default=float)
    print("ZZSPEEDSUB-T1-BESIDE-DONE", a.out)
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--static-check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--out", default="SPEEDSUB_T1_READ.json")
    ap.add_argument("--prereg", default="")
    ap.add_argument("--prereg-sha", default="")
    ap.add_argument("--orc-dir", default=ORC_DIR)
    ap.add_argument("--work-dir", default="")
    ap.add_argument("--instr", default=f"{HERE}/../stack/scripts/v8f_mini_eval_speedread.py")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--preflight", action="store_true", help="light controls on the three CTRL s0 rollouts (K-NONSPEED K-REC K-REACH K-SHUF K-SHA K-FLOOR K-WIN K-ID)")
    ap.add_argument("--smoke-check", action="store_true", help="light controls on the 200-window smoke (K-NONSPEED K-REC K-REACH K-SHA)")
    ap.add_argument("--make-manifest", action="store_true")
    ap.add_argument("--beside", action="store_true", help="the beside logged lane: egol - identity_l on CTRL (decides nothing)")
    ap.add_argument("--run", default="CTRL_s0", choices=RUNS)
    ap.add_argument("--out-root", default="F:/TanitAD-scratch/speedsub_t1/out")
    ap.add_argument("--pins", default="")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.static_check:
        bad = prereg_problems(a.prereg, a.prereg_sha) if a.prereg else ["--prereg is required"]
        print(json.dumps({"problems": bad, "prereg_sha256": PREREG_SHA256, "amendment_1_gviol": AMENDMENT_1_GVIOL}, indent=1))
        print("SPEEDSUB-T1-READ-STATIC-OK" if not bad else "SPEEDSUB-T1-READ-STATIC-REFUSED")
        return 0 if not bad else 2
    if a.dry_run:
        return main_dry(a)
    if a.preflight or a.smoke_check:
        if not a.pins:
            raise SystemExit("--pins is required")
        return main_light(a, "preflight" if a.preflight else "smoke")
    if a.make_manifest:
        if not a.pins:
            raise SystemExit("--pins is required")
        return make_manifest(a)
    if a.beside:
        if not a.manifest:
            raise SystemExit("--manifest is required")
        return main_beside(a)
    if not a.manifest:
        raise SystemExit("--manifest is required (or --static-check / --dry-run)")
    return main_read(a)


if __name__ == "__main__":
    sys.exit(main())
