"""PREREG_V8F_SPEEDAVG reader, committed before its data: is T's speed error VARIANCE (averaging two seeds' speed cells helps) or BIAS (it does not)?

  python -I v8f_speedavg_read.py --eval-dir <speedavg evals> --kindx-dir-s0 <KINDX s0 evals> --kindx-dir-s1 <KINDX s1 evals> --expect-sha0 H --expect-sha1 H --out READ.json
  python -I v8f_speedavg_read.py ... --check-inputs        (every file, sha and structure refusal, NO number read)

ZERO TRAINING. The evals are `stack/scripts/v8f_mini_eval_speedavg.py`'s: per seed s, four modes on X_s (`own`, `self`, `gt`, `avg`; see the driver's docstring). X_s = `M2b-mixo_s<s>`.
The PREREG is binding and nothing here moves a cell set, the averaging rule, a bar or a routing (a test compares every committed string with `Project Steering/PREREG_V8F_SPEEDAVG.md`).

THE READ (T1-class open loop, eval-141, paired episode-cluster bootstrap B = 2,000, 141 episodes / 2,399 windows; F = 0.1207 m, committed):
  PRIMARY    own-code ADE6, SPEEDAVG_s - X_s  (avg - own)             <= -F AND separated, at BOTH seeds
  CAPTURE    (ADE6 X_s - SPEEDAVG_s) / gain_speed(X_s)                beside: the fraction of the speed oracle's headroom the average recovers
  SECONDARY  speed MAE, accel MAE, distance keeping, LEAD-VIOL: SPEEDAVG - X_s     not separated-worse at either seed
  beside     the four families; the speed cells' RMSE to the GT before and after averaging; the two seeds' speed-cell correlation          (descriptive)
CONTROLS (all before any number is read; a failed control is rc 2 and no outcome): 1 self-average == own (delta 0.0 per window); 2 `gt` == KINDX's speed-only oracle per window <= 1e-4 m (and the
driver's own code-level check == 0.0); 3 `own` == its banked eval per window <= 1e-4 m. Added guards: the hook leaves every non-speed cell alone; T's decisions and S are bit-equal to X_s's; the averaged cells
ARE the mean of the two seeds' cells; the two seeds' `avg` arms averaged the SAME pair; a live-probe check (the average moves O's plan on some window).

DECLARED INTERPRETATIONS (fixed before any SPEEDAVG number; none moves a bar):
  1. PRIMARY "separated": the paired CI excludes 0. "<= -F" is the UNROUNDED point (avg - own) <= -0.1207.
  2. "not separated-worse" (SECONDARY): worse = the paired CI lies ABOVE 0 (lo > 0) for a metric where more is worse: speed MAE, accel MAE; DISTANCE KEEPING = the arm's |metric - the human's own value| for
     headway / time-gap / min-TTC (`abs_error_vs_human`, K = 10 lead block): more distance to the human is worse; LEAD-VIOL = the share of lead windows violating (more is worse).
  3. PARTIAL = PRIMARY < 0 and separated at BOTH seeds, and the PRIMARY bar (<= -F at both) NOT met ("but above -F"). The row is silent on a pair like (-0.30, -0.05); this reader reads it as PARTIAL (the bar
     is missed at one seed) and records `partial_reading_dependent` when the other literal reading (BOTH seeds above -F) would call it MIXED. Both seeds <= -F with a separated-worse SECONDARY is MIXED.
  4. BIAS, NOT VARIANCE = PRIMARY not separated (the CI includes 0) at either seed, literally; a separated-WORSE PRIMARY is not "not separated" and falls to MIXED.
  5. gain_speed(X_s) is KINDX's (`eval_M2b-mixo_s<s>-kindx-own` - `-kindx-only-speed`, the quantity that made SPEED a carrier); the driver's own-minus-gt is reported beside it.
STRATEGIC is in force for v8F (PI-V8F-2026-10-17, retractions R72 / R73). S is X_s's own in every arm, so its block is identical to X_s's by construction; the reader CHECKS that and reports the numbers.
Tier stamp: "T1-class open loop (own held outputs; no recorded future action)" (the eval json's own `T0-open-loop...` string is a known mislabel and is not copied).
Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B = 2,000, over EPISODES: it answers "would another draw of EPISODES say this?" and is BLIND to training variance (H-ESTIM-SEED-1). The second
seed IS the replicate the PREREG asks for; a separated CI at two seeds is still necessary, not sufficient, for a lever claim.
Refusals (rc 2, one prefix): a missing file, an eval that is not the mode / seed / checkpoint it claims, a limited (smoke) run, a failed control, a window list that moved."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
PKG = HERE.parents[1]
RAW = PKG / "raw"
SCHEMA = "v8f-speedavg-read-v1"
TIER = "T1-class open loop (own held outputs; no recorded future action)"
F_BAR = 0.1207                                   # PREREG_V8F_SPEEDAVG: F, committed
B_BOOT = 2000
N_WINDOWS, N_EPISODES = 2399, 141
SEEDS = (0, 1)
MODES = ("own", "self", "gt", "avg")
TOL_M = 1e-4                                     # the controls' per-window bar (m)
TOL_CELLS = 1e-5                                 # the same model run in two processes: the cells agree to float noise
X0_SHA = "c6a628762790b4fc3f7634867fc15b8167b035692ff518203c9fc0b3c126250d"        # X_s0 = M2b-mixo_s0 (KINDX)
SPEED_CELLS = ("lon_v_target_ms", "v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms")
DK_KEYS = ("headway_min_m", "time_gap_min_s", "min_ttc_s")
K_DK = 10
OUTCOMES = ("VARIANCE PAYS", "PARTIAL", "BIAS, NOT VARIANCE", "MIXED")
#: the PREREG's outcome table, VERBATIM (a test compares every cell with the file): condition, committed next step
TABLE = {
    "VARIANCE PAYS": ("PRIMARY and SECONDARY at both seeds",
                      "the two-seed speed average joins `forward_infer`; next, distil it into one T (a train-time arm under its own PREREG) so the deployed model needs one T"),
    "PARTIAL": ("PRIMARY < 0 and separated at both seeds but above −F",
                "report CAPTURE; next: a temporal average of T's own speed cells over its last ticks (single model, zero training), under its own PREREG"),
    "BIAS, NOT VARIANCE": ("PRIMARY not separated at either seed",
                           "T's speed error is systematic; next: a T-side speed-head arm that targets bias (a re-weighted or class-balanced speed loss), under its own PREREG"),
    "MIXED": ("anything else", "reported per seed; seeds before windows"),
}


def refuse(msg):
    print(f"SPEEDAVG READ REFUSED: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _jl(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def _paths():
    for p in (HERE.parents[5] / "stack", HERE.parents[5] / "taniteval"):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pf_default():
    _paths()
    from taniteval.ci import paired_episode_cluster_bootstrap
    return paired_episode_cluster_bootstrap


def paired(pf, a, b, eid, n_boot=B_BOOT):
    """a - b, paired over EPISODES. `point` is the UNROUNDED difference (the library rounds `delta` to 4 dp; a size bar must not)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    r = dict(pf(a, b, eid, n_boot, 0))
    r["point"] = float(a.mean() - b.mean())
    return r


def dev(a, b):
    return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max())


def same_json(a, b) -> bool:
    """Two json-able blocks are equal (NaN == NaN): compared as canonical strings."""
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


# ------------------------------------------------------------------------------------------------ inputs
def load_eval(eval_dir, name, need_cells=False, need_json=True):
    pz, pj = os.path.join(eval_dir, f"eval_{name}.npz"), os.path.join(eval_dir, f"eval_{name}.json")
    if not os.path.exists(pz):
        refuse(f"eval_{name}.npz is missing from {eval_dir}")
    if need_json and not os.path.exists(pj):
        refuse(f"eval_{name}.json is missing from {eval_dir}")
    z = np.load(pz)
    for k in ("eid", "ade6", "pred"):
        if k not in z.files:
            refuse(f"eval_{name}.npz carries no {k}")
    ade = np.asarray(z["ade6"], np.float64)
    if not np.isfinite(ade).all():
        refuse(f"eval_{name}.npz holds a non-finite ade6")
    E = {"name": name, "eid": np.asarray(z["eid"]), "ade6": ade, "pred": np.asarray(z["pred"], np.float64), "path": pz,
         "ade2": np.asarray(z["ade2"], np.float64) if "ade2" in z.files else None, "fde6": np.asarray(z["fde6"], np.float64) if "fde6" in z.files else None,
         "json": _jl(pj) if os.path.exists(pj) else None, "cells": None}
    if need_cells:
        pc = os.path.join(eval_dir, f"speedavg_cells_{name}.npz")
        if not os.path.exists(pc):
            refuse(f"speedavg_cells_{name}.npz is missing from {eval_dir}")
        zc = np.load(pc)
        E["cells"] = {k: np.asarray(zc[k]) for k in zc.files}
    return E


def check_arm(E, seed, mode, sha0, sha1, allow_smoke=False):
    """One driver eval: the mode / seed / tag it claims, both checkpoints' sha256, the base checkpoint it scored, no `limit`, and the relabelled carrier."""
    j = E["json"]
    b = j.get("speedavg")
    if not isinstance(b, dict) or b.get("schema") != "v8f-speedavg-eval-v1":
        refuse(f"eval_{E['name']}.json carries no speedavg block: it is not a v8f_mini_eval_speedavg.py output")
    tag = f"M2b-mixo_s{seed}-speedavg-{mode}"
    if (b.get("mode"), b.get("seed"), b.get("tag"), j.get("tag")) != (mode, seed, tag, tag):
        refuse(f"eval_{E['name']}.json says mode/seed/tag {(b.get('mode'), b.get('seed'), b.get('tag'), j.get('tag'))}, expected {(mode, seed, tag, tag)}")
    if b.get("limit") and not allow_smoke:
        refuse(f"eval_{E['name']}.json is a LIMITED run (first {b['limit']} windows): a smoke, never a result")
    if (b.get("x0_sha256"), b.get("x1_sha256")) != (sha0, sha1):
        refuse(f"eval_{E['name']}.json scored checkpoints {(str(b.get('x0_sha256'))[:12], str(b.get('x1_sha256'))[:12])}, expected {(sha0[:12], sha1[:12])}")
    if str(j.get("ckpt")) != str(b.get("base_ckpt")):
        refuse(f"eval_{E['name']}.json: ckpt {j.get('ckpt')!r} is not the base checkpoint {b.get('base_ckpt')!r} of X_{seed}")
    if b.get("base_ckpt") != b.get(f"x{seed}_ckpt"):
        refuse(f"eval_{E['name']}.json: the base checkpoint is not X_{seed}'s")
    if mode != "own" and (j.get("tac") != f"speedavg-{mode}" or (j.get("tac_manipulation") or {}).get("mode") != f"speedavg-{mode}"):
        refuse(f"eval_{E['name']}.json: the carrier labels were not relabelled to speedavg-{mode}")
    if mode == "own" and (j.get("tac") != "pred" or j.get("tac_manipulation")):
        refuse(f"eval_{E['name']}.json is not X_{seed}'s own code (tac {j.get('tac')!r})")
    if mode != "own":
        h = b.get("hook") or {}
        if h.get("mode") != mode:
            refuse(f"eval_{E['name']}.json: the hook record is missing or of another mode")
    return b


def load_all(eval_dir, kindx_dirs, sha0, sha1, allow_smoke=False, first_n=None):
    """-> A[seed][mode] (the driver's evals + cells), K[seed] = {'own', 'speed'} (KINDX's), eid. Every file on ONE window list; the committed 2,399 windows over 141 episodes.
    SMOKE (`allow_smoke`, `first_n`): the driver evals hold the first n windows, KINDX's / banked ones the full list (compared on the first n); a missing KINDX speed-only eval is tolerated (control 2 is then NOT RUN)."""
    A, K = {}, {}
    for s in SEEDS:
        A[s] = {}
        for m in MODES:
            E = load_eval(eval_dir, f"M2b-mixo_s{s}-speedavg-{m}", need_cells=(m != "own"))
            E["block"] = check_arm(E, s, m, sha0, sha1, allow_smoke)
            A[s][m] = E
        K[s] = {}
        for nm, key in (("own", "own"), ("only-speed", "speed")):
            name = f"M2b-mixo_s{s}-kindx-{nm}"
            if allow_smoke and key == "speed" and not os.path.exists(os.path.join(kindx_dirs[s], f"eval_{name}.npz")):
                K[s][key] = None
                continue
            K[s][key] = load_eval(kindx_dirs[s], name, need_json=False)
    eid = A[0]["own"]["eid"]
    n = int(eid.size)
    for s in SEEDS:
        for E in list(A[s].values()):
            if not np.array_equal(E["eid"], eid):
                refuse(f"eval {E['name']} is not on one window list (eid differs): the window set or the episode list moved")
            if E["pred"].shape != A[s]["own"]["pred"].shape:
                refuse(f"eval {E['name']} has pred {E['pred'].shape}, the own eval has {A[s]['own']['pred'].shape}")
        for E in K[s].values():
            if E is None:
                continue
            if not np.array_equal(E["eid"][:n] if (allow_smoke and first_n) else E["eid"], eid):
                refuse(f"eval {E['name']} is not on the driver evals' window list (eid differs): the window set or the episode list moved")
    ne = int(np.unique(eid).size)
    if (n, ne) != (N_WINDOWS, N_EPISODES) and not allow_smoke:
        refuse(f"the evals hold {n} windows over {ne} episodes, the committed eval-141 grid has {N_WINDOWS} over {N_EPISODES}")
    return A, K, eid


# ------------------------------------------------------------------------------------------------ the controls
def controls(A, K, eid, pf, banked=None, tol=TOL_M, first_n=None, n_boot=B_BOOT, smoke=False):
    """Controls 1-3 and the added guards, per seed (`first_n`: a SMOKE compares the first n windows of the full-size banked npz). Returns {'PASS': bool, ...}; nothing is read from the arms' outcomes."""
    banked = banked or {}
    out = {"tolerance_m": tol, "seeds": {}}
    sl = (slice(0, first_n) if first_n else slice(None))
    for s in SEEDS:
        a, k = A[s], K[s]
        r = {}
        # CONTROL 1: X_s averaged with itself reproduces X_s's own ADE6 exactly, per window (and the plan, bit for bit)
        d1 = dev(a["self"]["ade6"], a["own"]["ade6"])
        p1 = paired(pf, a["self"]["ade6"], a["own"]["ade6"], eid, n_boot)
        r["control_1_self_average_is_identity"] = {"ade6_max_abs_dev_m": d1, "pred_equal": bool(np.array_equal(a["self"]["pred"], a["own"]["pred"])),
                                                   "paired_delta": [p1["delta"], p1["lo"], p1["hi"]], "bar": "exactly 0.0 per window (PREREG: delta = 0.0)"}
        r["control_1_self_average_is_identity"]["PASS"] = bool(d1 == 0.0 and r["control_1_self_average_is_identity"]["pred_equal"] and p1["delta"] == 0.0 and p1["lo"] == 0.0 and p1["hi"] == 0.0)
        # CONTROL 2: the GT in the five speed cells reproduces KINDX's speed-only oracle per window; and the driver's own code-level comparison with the eval's oracle code is exactly 0.0
        gtdev = a["gt"]["block"]["hook"].get("oracle_code_max_abs_dev")
        if k["speed"] is None:                                                   # a SMOKE with no banked speed-only oracle for this checkpoint
            r["control_2_gt_equals_kindx_speed_oracle"] = {"status": "NOT RUN: no banked speed-only oracle eval for this checkpoint on this machine (SMOKE)", "driver_code_max_abs_dev": gtdev,
                                                           "PASS": bool(gtdev == 0.0) if smoke else False}
        else:
            d2 = dev(a["gt"]["ade6"], k["speed"]["ade6"][sl])
            r["control_2_gt_equals_kindx_speed_oracle"] = {"ade6_max_abs_dev_m": d2, "pred_max_abs_dev_m": dev(a["gt"]["pred"], k["speed"]["pred"][sl]), "driver_code_max_abs_dev": gtdev,
                                                           "bar": f"ade6 <= {tol} m per window AND the driver's code == the eval's oracle code exactly (0.0)"}
            r["control_2_gt_equals_kindx_speed_oracle"]["PASS"] = bool(d2 <= tol and gtdev == 0.0)
        # CONTROL 3: X_s's own forward reproduces its banked eval per window
        refs = {"kindx_own": k["own"]["ade6"][sl]}
        if s in banked and banked[s] is not None:
            refs["banked_standard_eval"] = banked[s]["ade6"][sl]
        d3 = {nm: dev(a["own"]["ade6"], v) for nm, v in refs.items()}
        r["control_3_own_equals_banked"] = {"ade6_max_abs_dev_m": d3, "bar": f"<= {tol} m per window against each reference"}
        r["control_3_own_equals_banked"]["PASS"] = bool(all(v <= tol for v in d3.values()))
        # added guards
        g = {}
        g["hook_leaves_nonspeed_cells_alone"] = bool(all(a[m]["block"]["hook"].get("nonspeed_max_abs_dev") == 0.0 for m in ("self", "gt", "avg")))
        tact = same_json(a["own"]["json"]["raw"].get("tactical"), a["avg"]["json"]["raw"].get("tactical"))
        stra = same_json(a["own"]["json"]["raw"].get("strategic"), a["avg"]["json"]["raw"].get("strategic"))
        g["tactical_and_strategic_equal_to_X_s"] = bool(tact and stra)
        c = a["avg"]["cells"]
        g["avg_cells_are_the_mean_of_the_two_seeds_cells"] = bool(np.array_equal(c["used_std"], (c["own_std"] + c["partner_std"]) * np.float32(0.5)))
        cs = a["self"]["cells"]
        g["self_cells_equal_own_cells"] = bool(np.array_equal(cs["used_std"], cs["own_std"]) and np.array_equal(cs["partner_std"], cs["own_std"]))
        cg = a["gt"]["cells"]
        nh = ~cg["has"].astype(bool)
        g["gt_falls_back_to_own_where_no_v9_row"] = bool(np.array_equal(cg["used_std"][nh], cg["own_std"][nh]) and np.array_equal(a["gt"]["ade6"][nh], a["own"]["ade6"][nh]))
        g["avg_is_live_it_moves_the_plan"] = bool(np.abs(a["avg"]["pred"] - a["own"]["pred"]).max() > 0.0 and np.abs(c["used_std"] - c["own_std"]).max() > 0.0)
        g["window_counts"] = {"n_windows": int(eid.size), "n_episodes": int(np.unique(eid).size)}
        g["n_windows_without_v9_row"] = int(nh.sum())
        r["guards"] = g
        r["guards_PASS"] = bool(all(v for k_, v in g.items() if isinstance(v, bool)))
        r["PASS"] = bool(r["control_1_self_average_is_identity"]["PASS"] and r["control_2_gt_equals_kindx_speed_oracle"]["PASS"] and r["control_3_own_equals_banked"]["PASS"] and r["guards_PASS"])
        out["seeds"][f"s{s}"] = r
    # cross-seed: both avg arms averaged the SAME pair of models
    c0, c1 = A[0]["avg"]["cells"], A[1]["avg"]["cells"]
    x = {"used_std_max_abs_dev": dev(c0["used_std"], c1["used_std"]), "partner_s0_vs_own_s1": dev(c0["partner_std"], c1["own_std"]), "partner_s1_vs_own_s0": dev(c1["partner_std"], c0["own_std"]),
         "x0_differs_from_x1": bool(dev(c0["own_std"], c1["own_std"]) > 0.0), "bar": f"<= {TOL_CELLS} (the same model run in two processes)"}
    x["PASS"] = bool(x["used_std_max_abs_dev"] <= TOL_CELLS and x["partner_s0_vs_own_s1"] <= TOL_CELLS and x["partner_s1_vs_own_s0"] <= TOL_CELLS and x["x0_differs_from_x1"])
    out["cross_seed_same_pair"] = x
    out["PASS"] = bool(all(out["seeds"][f"s{s}"]["PASS"] for s in SEEDS) and x["PASS"])
    return out


# ------------------------------------------------------------------------------------------------ the reads
def primary_met(g, F=F_BAR) -> bool:
    """PRIMARY at one seed: avg - own <= -F (the UNROUNDED point) and separated (the paired CI excludes 0), on the improving side."""
    return bool(g["point"] <= -F and g["separated"] and g["point"] < 0)


def primary_negative_separated(g) -> bool:
    return bool(g["point"] < 0 and g["separated"])


def separated_worse(g) -> bool:
    """A paired block whose CI lies entirely above 0 (more is worse): speed MAE, accel MAE, |metric - human|, LEAD-VIOL."""
    return bool(isinstance(g, dict) and g.get("lo") is not None and g["lo"] > 0)


def secondary_of(fam, dk_pair):
    """SECONDARY at one seed: speed MAE, accel MAE (the paired LONGITUDINAL block), distance keeping (|arm - human| on headway / time-gap / min-TTC) and LEAD-VIOL. A missing block is REFUSED (never dropped)."""
    lon = (fam or {}).get("longitudinal") or {}
    rows = {}
    for nm, key in (("speed_mae", "speed_mae_mps"), ("accel_mae", "accel_mae_mps2")):
        blk = lon.get(key) if "paired_vs_own" not in lon else lon["paired_vs_own"].get(key)
        if not isinstance(blk, dict) or "lo" not in blk:
            refuse(f"the paired LONGITUDINAL block has no {key}: a missing family metric is refused, never dropped")
        rows[nm] = blk
    if not isinstance(dk_pair, dict):
        refuse("the distance-keeping read has no pair block: refused, never dropped")
    for kk in DK_KEYS:
        blk = (dk_pair.get("abs_error_vs_human") or {}).get(kk)
        if not isinstance(blk, dict):
            refuse(f"the distance-keeping pair has no abs_error_vs_human.{kk}: refused, never dropped")
        rows[f"distance_keeping_abs_error_vs_human.{kk}"] = blk
    lv = ((dk_pair.get("lead_viol") or {}).get("all_lead_windows"))
    if not isinstance(lv, dict) or "lo" not in lv:
        refuse("the distance-keeping pair has no lead_viol.all_lead_windows block: refused, never dropped")
    rows["lead_viol"] = lv
    worse = {k: separated_worse(v) for k, v in rows.items()}
    return {"rows": rows, "separated_worse": worse, "not_separated_worse": bool(not any(worse.values()))}


def decide(prim, sec_ok, F=F_BAR):
    """THE COMMITTED TABLE. prim[s] = the PRIMARY paired block, sec_ok[s] = SECONDARY not separated-worse. See the module docstring (interpretations 3 and 4)."""
    met = {s: primary_met(prim[s], F) for s in SEEDS}
    negsep = {s: primary_negative_separated(prim[s]) for s in SEEDS}
    sep = {s: bool(prim[s]["separated"]) for s in SEEDS}
    if all(met.values()) and all(sec_ok.values()):
        out = "VARIANCE PAYS"
    elif all(negsep.values()) and not all(met.values()):
        out = "PARTIAL"
    elif not any(sep.values()):
        out = "BIAS, NOT VARIANCE"
    else:
        out = "MIXED"
    one_above = [s for s in SEEDS if negsep[s] and not met[s]]
    # the other literal reading of PARTIAL ("above -F" at BOTH seeds): it differs only when one seed meets -F and the other does not
    lit2_partial = bool(all(negsep.values()) and not any(met.values()))
    return {"outcome": out, "primary_met": {f"s{s}": met[s] for s in SEEDS}, "primary_negative_and_separated": {f"s{s}": negsep[s] for s in SEEDS},
            "primary_separated": {f"s{s}": sep[s] for s in SEEDS}, "secondary_not_separated_worse": {f"s{s}": bool(sec_ok[s]) for s in SEEDS},
            "partial_reading_dependent": bool(out == "PARTIAL" and not lit2_partial), "seeds_above_minus_F": [f"s{s}" for s in one_above],
            "per_seed_row": {f"s{s}": ("PRIMARY met" if met[s] else ("PRIMARY < 0, separated, above -F" if negsep[s] else ("PRIMARY not separated" if not sep[s] else "PRIMARY separated WORSE"))) +
                             ("" if sec_ok[s] else "; SECONDARY separated-worse") for s in SEEDS},
            "F_m": F}


def capture(own, avg, g_own, g_speed, eid, n_boot=B_BOOT):
    """CAPTURE = (ADE6 X_s - ADE6 SPEEDAVG_s) / gain_speed(X_s). `g_own`, `g_speed` = KINDX's per-window ADE6 of X_s's own code / the speed-only oracle. The interval is DESCRIPTIVE: percentile of the ratio over the
    paired estimator's own episode draws (`ci._draws`, seed 0)."""
    _paths()
    from taniteval import ci
    num = float(np.mean(own) - np.mean(avg))
    den = float(np.mean(g_own) - np.mean(g_speed))
    if not den > 0:
        return {"status": "NOT-DEFINED", "reason": f"gain_speed(X_s) = {den:+.4f} <= 0: nothing to capture", "gain_speed_m": den, "recovered_m": num}
    uniq, idx = ci.episode_index(eid)
    vals = []
    for sel in ci._draws(uniq, idx, n_boot, 0):
        d = float(np.mean(g_own[sel]) - np.mean(g_speed[sel]))
        if d > 0:
            vals.append(float(np.mean(own[sel]) - np.mean(avg[sel])) / d)
    out = {"point": num / den, "recovered_m": num, "gain_speed_m": den, "n_draws": len(vals)}
    if len(vals) >= max(2, n_boot // 2):
        lo, hi = np.percentile(vals, [2.5, 97.5])
        out.update({"lo": float(lo), "hi": float(hi), "interval": "descriptive: percentile of the ratio over the paired estimator's own episode draws"})
    else:
        out["interval"] = "NONE: gain_speed > 0 on too few draws"
    return out


# ------------------------------------------------------------------------------------------------ descriptives (beside, decide nothing)
def _corr(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return None
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def descriptives(c0, c1):
    """The speed cells before and after averaging, in physical units (m/s). `c0` / `c1` = the seed-0 / seed-1 `avg` cells npz: c0.own_std = X_0's cells, c1.own_std = X_1's, used_std = their mean.
    RMSE to the GT on the windows that have a finite GT cell; the correlation of the two seeds' predictions (all windows) and of their ERRORS (GT windows); the RMSE ratio a pure-variance model
    (independent errors of equal variance, correlation rho_e) would predict: sqrt((1 + rho_e) / 2)."""
    mu, sd = np.asarray(c0["cons_mu"], np.float64), np.asarray(c0["cons_sd"], np.float64)
    ph = lambda x: np.asarray(x, np.float64) * sd + mu                                             # noqa: E731
    x0, x1, xa = ph(c0["own_std"]), ph(c1["own_std"]), ph(c0["used_std"])
    gt = np.asarray(c0["gt_phys"], np.float64)
    has = np.asarray(c0["has"], bool)[:, None] & np.isfinite(gt)
    rows = {}
    for i, nm in enumerate(SPEED_CELLS):
        m = has[:, i]
        e0, e1, ea = x0[m, i] - gt[m, i], x1[m, i] - gt[m, i], xa[m, i] - gt[m, i]
        rm = lambda e: float(np.sqrt(np.mean(e ** 2))) if e.size else None                          # noqa: E731
        r0, r1, ra = rm(e0), rm(e1), rm(ea)
        rho_e = _corr(e0, e1)
        rows[nm] = {"n_windows_with_gt": int(m.sum()), "rmse_to_gt_X0": r0, "rmse_to_gt_X1": r1, "rmse_to_gt_averaged": ra,
                    "corr_pred_X0_X1": _corr(x0[:, i], x1[:, i]), "corr_error_X0_X1": rho_e,
                    "rmse_ratio_measured": (ra / float(np.sqrt((r0 ** 2 + r1 ** 2) / 2)) if r0 and r1 and ra is not None else None),
                    "rmse_ratio_if_pure_variance": (float(np.sqrt((1.0 + rho_e) / 2.0)) if rho_e is not None else None)}
    m = has
    e0, e1, ea = (x0 - gt)[m], (x1 - gt)[m], (xa - gt)[m]
    pooled = {"n_cell_values": int(m.sum()), "rmse_to_gt_X0": float(np.sqrt(np.mean(e0 ** 2))), "rmse_to_gt_X1": float(np.sqrt(np.mean(e1 ** 2))), "rmse_to_gt_averaged": float(np.sqrt(np.mean(ea ** 2))),
              "corr_pred_X0_X1": _corr(x0.ravel(), x1.ravel()), "corr_error_X0_X1": _corr(e0, e1)}
    return {"unit": "m/s (standardised cells x cons_sd + cons_mu)", "per_cell": rows, "pooled": pooled,
            "reading": ("rmse_ratio_measured < rmse_ratio_if_pure_variance means the average removes MORE than independent noise would; > means LESS (errors shared between the seeds: bias-like). "
                        "Descriptive; it decides nothing.")}


# ------------------------------------------------------------------------------------------------ the real family / distance-keeping tools (injected in tests)
def real_tools(args, eid):
    """-> (famfn, dkfn): the programme's own instruments, IMPORTED (never copied): `v8f_pgt_read` brings the family block, the lead-block context and the K = 10 distance-keeping read."""
    _paths()
    PGT = _load("v8f_pgt_read_for_speedavg", HERE.parent / "v8f_pgt_read.py")
    PGT.refuse = refuse
    DK, WC = PGT._modules()
    ns = argparse.Namespace(eval_dir=args.eval_dir, runs_dir=None, out=None, ckpt_sha=None, n_boot=args.n_boot, dk_n_boot=args.n_boot, no_families=False, strata=None, no_strata=True,
                            n_windows=N_WINDOWS, n_episodes=N_EPISODES, lead_block=args.lead_block, p10_pred=None, eval_grid=str(RAW / "eval_grid_windows.json"),
                            eval_kin=str(RAW / "eval141_kin.npz"), eval_join=str(RAW / "join_eval141.npz"), gt_npz=None)
    famfn = PGT._famfn(ns)
    ctx = PGT.build_ctx(ns, DK, eid, need_p10=False)

    def dkfn(preds, saved, pair):
        try:
            return DK.read_dk(preds=preds, gt=ctx["G"]["gt"], eid=ctx["eid"], clip=ctx["G"]["clip"], frame=ctx["G"]["frame"], v0=ctx["G"]["v0"], block=ctx["blk10"], k=K_DK, dt=ctx["G"]["dt"],
                              n_boot=args.n_boot, seed=0, saved_ade6=saved, p10_floor_ade6=None, pairs=[pair])
        except DK.Refusal as e:
            refuse(f"v8f_dk_read: {e}")
    return famfn, dkfn


def family_reads(A_s, eid, famfn, dkfn, seed):
    """Per seed: the paired family blocks and the distance-keeping read of avg vs X_s's own."""
    na, nb = f"M2b-mixo_s{seed}-speedavg-avg", f"M2b-mixo_s{seed}-speedavg-own"
    pa, pb = A_s["avg"]["pred"], A_s["own"]["pred"]
    fam = famfn(pa, pb, eid)
    if not fam.get("longitudinal") or not fam.get("lateral"):
        refuse(f"the paired family block of seed {seed} has no longitudinal / lateral component: a missing family is refused")
    d = dkfn({na: pa, nb: pb}, {na: A_s["avg"]["ade6"], nb: A_s["own"]["ade6"]}, (na, nb))
    pair = d["read"]["pairs"].get(f"{na}__minus__{nb}")
    if not pair:
        refuse(f"the distance-keeping read has no pair block for {na} - {nb}")
    return fam, d["read"], pair


# ------------------------------------------------------------------------------------------------ the read
def build_result(A, K, eid, ctl, pf, fams, n_boot=B_BOOT, F=F_BAR):
    """The DECIDING block (PRIMARY, CAPTURE, SECONDARY, outcome). `fams[s]` = (fam, dk_read, dk_pair)."""
    prim, sec, cap = {}, {}, {}
    for s in SEEDS:
        a = A[s]
        prim[s] = paired(pf, a["avg"]["ade6"], a["own"]["ade6"], eid, n_boot)
        sec[s] = secondary_of(fams[s][0], fams[s][2])
        cap[s] = capture(a["own"]["ade6"], a["avg"]["ade6"], K[s]["own"]["ade6"], K[s]["speed"]["ade6"], eid, n_boot)
        cap[s]["driver_gain_speed_m"] = float(a["own"]["ade6"].mean() - a["gt"]["ade6"].mean())
    dec = decide(prim, {s: sec[s]["not_separated_worse"] for s in SEEDS}, F)
    out_name = dec["outcome"]
    res = {"schema": SCHEMA, "prereg": "PREREG_V8F_SPEEDAVG.md", "tier": TIER,
           "estimator": f"paired episode-cluster bootstrap (taniteval/ci.py), B={n_boot}, {int(np.unique(eid).size)} episodes / {int(eid.size)} windows; answers 'another draw of EPISODES' only; "
                        "BLIND to training variance (H-ESTIM-SEED-1): the second seed is the replicate; a separated CI at two seeds is necessary, not sufficient, for a lever claim",
           "arm": "SPEEDAVG_s = X_s with T's five speed cells replaced by the mean of X_0's and X_1's (X_s = M2b-mixo_s<s>)", "F_m": F, "speed_cells": list(SPEED_CELLS),
           "controls": {"PASS": ctl["PASS"], "detail": ctl},
           "ade6_m": {f"s{s}": {"X_own": float(A[s]["own"]["ade6"].mean()), "SPEEDAVG": float(A[s]["avg"]["ade6"].mean()), "self": float(A[s]["self"]["ade6"].mean()), "gt_speed_cells": float(A[s]["gt"]["ade6"].mean())} for s in SEEDS},
           "PRIMARY": {f"s{s}": {k: prim[s][k] for k in ("point", "delta", "lo", "hi", "separated", "n_windows", "n_episodes", "n_boot", "estimator")} for s in SEEDS},
           "CAPTURE": {f"s{s}": cap[s] for s in SEEDS},
           "SECONDARY": {f"s{s}": {"not_separated_worse": sec[s]["not_separated_worse"], "separated_worse": sec[s]["separated_worse"],
                                   "rows": {k: {x: v.get(x) for x in ("delta", "lo", "hi", "separated", "n_used", "n_windows") if x in v} for k, v in sec[s]["rows"].items()}} for s in SEEDS},
           "decision": dec, "OUTCOME": out_name,
           "outcome_table_verbatim": {o: {"condition": TABLE[o][0], "next": TABLE[o][1], "this_read": o == out_name} for o in OUTCOMES},
           "committed_next": TABLE[out_name][1]}
    res["reading"] = (f"PREREG_V8F_SPEEDAVG: {out_name}; PRIMARY (avg - own, ADE6 m): " + ", ".join(f"s{s} {prim[s]['point']:+.4f} [{prim[s]['lo']:+.4f}, {prim[s]['hi']:+.4f}]{'' if prim[s]['separated'] else ' ns'}" for s in SEEDS)
                      + f" vs -F = -{F}; CAPTURE " + ", ".join(f"s{s} " + (f"{cap[s]['point']:.2f}" if 'point' in cap[s] else "n/d") for s in SEEDS)
                      + "; SECONDARY not separated-worse: " + ", ".join(f"s{s} {sec[s]['not_separated_worse']}" for s in SEEDS) + ".")
    return res


def beside_block(A, K, eid, fams, pf, n_boot=B_BOOT):
    """Everything that decides nothing: the four families, the speed cells' RMSE before / after and the seeds' correlation, ADE2 / FDE6, S and T checked equal to X_s."""
    out = {"families": {}, "descriptives": descriptives(A[0]["avg"]["cells"], A[1]["avg"]["cells"]), "other_horizons": {}, "decides": "nothing"}
    for s in SEEDS:
        fam, dkr, pair = fams[s]
        j_own, j_avg = A[s]["own"]["json"]["raw"], A[s]["avg"]["json"]["raw"]
        out["families"][f"s{s}"] = {
            "LONGITUDINAL": {"paired_vs_own": fam.get("longitudinal"), "distance_keeping_k10": {"arm": dkr["arms"].get(f"M2b-mixo_s{s}-speedavg-avg"), "pair_vs_own": pair}},
            "LATERAL": {"paired_vs_own": fam.get("lateral")},
            "TACTICAL": {"status": "T's decisions are X_s's own in every arm: identical to X_s by construction (checked bit-equal)", "equal_to_X_s": same_json(j_own.get("tactical"), j_avg.get("tactical")),
                         "block": j_avg.get("tactical")},
            "STRATEGIC": {"status": "in force for v8F (PI-V8F-2026-10-17, R72 / R73). S is X_s's own: identical to X_s by construction (checked bit-equal)", "equal_to_X_s": same_json(j_own.get("strategic"), j_avg.get("strategic")),
                          "block": j_avg.get("strategic")}}
        out["other_horizons"][f"s{s}"] = {k: paired(pf, A[s]["avg"][k], A[s]["own"][k], eid, n_boot) for k in ("ade2", "fde6") if A[s]["avg"][k] is not None and A[s]["own"][k] is not None}
        out["other_horizons"][f"s{s}"]["self_minus_own_ade6"] = paired(pf, A[s]["self"]["ade6"], A[s]["own"]["ade6"], eid, n_boot)
        out["other_horizons"][f"s{s}"]["gt_speed_minus_own_ade6 (driver)"] = paired(pf, A[s]["gt"]["ade6"], A[s]["own"]["ade6"], eid, n_boot)
    return out


def md_of(res) -> str:
    L = [f"# SPEEDAVG_READ: PREREG_V8F_SPEEDAVG executed as written", "", f"**Outcome (the PREREG table applied verbatim): {res['OUTCOME']}.**", "",
         f"Tier: {res['tier']}. Estimator: {res['estimator']}.", "", f"Committed next step, quoted from the PREREG: *\"{res['committed_next']}\"*", "",
         "| read | seed 0 | seed 1 | bar |", "|---|---|---|---|"]
    P, C = res["PRIMARY"], res["CAPTURE"]
    f = lambda g: f"{g['point']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}]{' *' if g['separated'] else ''}"        # noqa: E731
    L.append(f"| PRIMARY ADE6 SPEEDAVG - X_s | {f(P['s0'])} | {f(P['s1'])} | <= -{res['F_m']} and separated at BOTH |")
    L.append("| CAPTURE | " + " | ".join((f"{C[f's{s}']['point']:.3f}" if "point" in C[f's{s}'] else "n/d") for s in SEEDS) + " | beside |")
    L.append("| SECONDARY not separated-worse | " + " | ".join(str(res["SECONDARY"][f"s{s}"]["not_separated_worse"]) for s in SEEDS) + " | both |")
    L += ["", "## The outcome table (verbatim from the PREREG)", "", "| outcome | condition | next (committed) | this read |", "|---|---|---|---|"]
    for o, v in res["outcome_table_verbatim"].items():
        L.append(f"| **{o}** | {v['condition']} | {v['next']} | {'**YES**' if v['this_read'] else ''} |")
    L += ["", "## Controls", "", f"PASS: {res['controls']['PASS']}"]
    for s in SEEDS:
        r = res["controls"]["detail"]["seeds"][f"s{s}"]
        L.append(f"* seed {s}: control 1 {r['control_1_self_average_is_identity']['PASS']} (max dev {r['control_1_self_average_is_identity']['ade6_max_abs_dev_m']}), "
                 f"control 2 {r['control_2_gt_equals_kindx_speed_oracle']['PASS']} (max dev {r['control_2_gt_equals_kindx_speed_oracle']['ade6_max_abs_dev_m']:.3g} m), "
                 f"control 3 {r['control_3_own_equals_banked']['PASS']} ({r['control_3_own_equals_banked']['ade6_max_abs_dev_m']}), guards {r['guards_PASS']}")
    L.append(f"* cross-seed (both avg arms averaged the same pair): {res['controls']['detail']['cross_seed_same_pair']['PASS']}")
    L += ["", f"Reading: {res['reading']}", ""]
    return "\n".join(L)


def check_inputs(args, A, K, eid):
    """Everything the read refuses on, no number read: every file present and on one window list, the shas, the modes, the lead block / ground truth for the family tools."""
    famfn, dkfn = real_tools(args, eid)
    return {"schema": "v8f-speedavg-check-inputs-v1", "n_windows": int(eid.size), "n_episodes": int(np.unique(eid).size), "seeds": list(SEEDS), "modes": list(MODES),
            "x0_sha256": args.expect_sha0, "x1_sha256": args.expect_sha1, "family_tools": "built", "decides": "nothing"}


def _dump(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--kindx-dir-s0", required=True)
    ap.add_argument("--kindx-dir-s1", required=True)
    ap.add_argument("--banked-own-s0", default=None, help="the banked STANDARD eval npz of X_s0 (control 3's second reference)")
    ap.add_argument("--banked-own-s1", default=None)
    ap.add_argument("--expect-sha0", default=X0_SHA)
    ap.add_argument("--expect-sha1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lead-block", default=None)
    ap.add_argument("--n-boot", type=int, default=B_BOOT)
    ap.add_argument("--no-beside", action="store_true")
    ap.add_argument("--check-inputs", action="store_true")
    ap.add_argument("--smoke-first-n", type=int, default=0, help="SMOKE ONLY: the evals are the first N windows (a driver --limit run); the outcome is NOT a result and is not written")
    args = ap.parse_args(argv)
    _paths()
    pf = pf_default()
    smoke = bool(args.smoke_first_n)
    if args.expect_sha0 == args.expect_sha1:
        refuse("--expect-sha0 == --expect-sha1: there is no second seed")
    A, K, eid = load_all(args.eval_dir, {0: args.kindx_dir_s0, 1: args.kindx_dir_s1}, args.expect_sha0, args.expect_sha1, allow_smoke=smoke, first_n=args.smoke_first_n or None)
    if smoke and int(eid.size) != args.smoke_first_n:
        refuse(f"--smoke-first-n {args.smoke_first_n} but the evals hold {eid.size} windows")
    banked = {}
    for s, p in ((0, args.banked_own_s0), (1, args.banked_own_s1)):
        if p:
            if not os.path.exists(p):
                refuse(f"--banked-own-s{s} {p} does not exist")
            banked[s] = load_eval(os.path.dirname(p), os.path.basename(p)[len("eval_"):-len(".npz")], need_json=False)
    if args.check_inputs:
        _dump(args.out, check_inputs(args, A, K, eid))
        print("SPEEDAVG INPUTS OK")
        return 0
    ctl = controls(A, K, eid, pf, banked, first_n=args.smoke_first_n or None, n_boot=min(args.n_boot, B_BOOT), smoke=smoke)
    _dump(os.path.splitext(args.out)[0] + "_controls.json", ctl)
    if not ctl["PASS"]:
        refuse("a control failed (see the *_controls.json): no outcome is read")
    if smoke:
        print("SMOKE: controls 1-3 and the guards PASS on the first", int(eid.size), "windows; no outcome is read from a smoke")
        return 0
    famfn, dkfn = real_tools(args, eid)
    fams = {s: family_reads(A[s], eid, famfn, dkfn, s) for s in SEEDS}
    res = build_result(A, K, eid, ctl, pf, fams, args.n_boot)
    _dump(args.out, res)                                          # the deciding block is on disk BEFORE any beside read is computed
    print(res["reading"])
    print("committed next:", res["committed_next"])
    if args.no_beside:
        return 0
    res["beside"] = beside_block(A, K, eid, fams, pf, args.n_boot)
    _dump(args.out, res)
    with open(os.path.splitext(args.out)[0] + ".md", "w", encoding="utf-8", newline="\n") as f:
        f.write(md_of(res))
    print("beside written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
