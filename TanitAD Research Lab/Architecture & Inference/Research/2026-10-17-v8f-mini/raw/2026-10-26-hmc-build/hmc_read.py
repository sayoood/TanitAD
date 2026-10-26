#!/usr/bin/env python3
"""hmc_read.py -- the READER of PREREG_V8F_HACC_MC (Project Steering/PREREG_V8F_HACC_MC.md @ 8d4b936d).  CPU only, read-only on the run.

  python hmc_read.py <TOOL_TREE> [--controls-only] [--smoke] [--run-dir DIR] [--out DIR] [--force-rows-under-refusal]
                     [--c1-bar literal|ulp32 --c1-amendment ID]  [--c4-clause2 literal|tail0 --c4-amendment ID]

ORDER (the PREREG's): provenance P-1, controls C-1..C-6 (C-7 = the suite, read from the chain's log) FIRST; the rows and the outcome word only if EVERY control passes
("REFUSED: any control fails -> no outcome row").  ``--controls-only`` never opens a row.  The two ``--c*-amendment`` switches exist ONLY so that a ratified PREREG amendment
can be applied without editing code: each REQUIRES an amendment id, which is printed into the report; without one the literal PREREG text governs.

REUSED, never re-implemented: comps / paired / ROWS (t1k_read_repo.py), lat_rows (t1klc_read.py), the along-ramp term (probe12's apply / inc + lcdiag's statekin bound to the tool
tree), pair_err (lcdiag common.py); see hmc_lib.py / hmc_controls.py.
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("tool_tree")
ap.add_argument("--run-dir", default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--controls-only", action="store_true")
ap.add_argument("--smoke", action="store_true", help="plumbing test on the 24-window SMOKE chain output (banked arrays subset to the same grid windows)")
ap.add_argument("--force-rows-under-refusal", action="store_true", help="compute the rows even if a control failed; written ONLY to rows_UNDER_REFUSAL.json, never an outcome")
ap.add_argument("--c1-bar", choices=("literal", "ulp32"), default="literal")
ap.add_argument("--c1-amendment", default="")
ap.add_argument("--c4-clause2", choices=("literal", "tail0"), default="literal")
ap.add_argument("--c4-amendment", default="")
A = ap.parse_args()
if (A.c1_bar != "literal" and not A.c1_amendment) or (A.c4_clause2 != "literal" and not A.c4_amendment):
    raise SystemExit("REFUSED: a non-literal bar needs the id of the ratified PREREG amendment (--c1-amendment / --c4-amendment)")
TREE = A.tool_tree.replace("\\", "/").rstrip("/")
os.environ["HMC_READ_ROOT"] = TREE
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Users/Admin/hmc")
import numpy as np  # noqa: E402

import hmc_lib as L  # noqa: E402
import hmc_controls as C  # noqa: E402

HMC = L.HMC
SMOKE = A.smoke
RUN = (A.run_dir or (f"{HMC}/smoke/chain" if SMOKE else f"{L.ART}/run")).replace("\\", "/")
LOGS = f"{HMC}/smoke/chain/logs" if SMOKE else f"{HMC}/logs"
OUT = (A.out or (RUN if SMOKE else f"{HMC}/read")).replace("\\", "/")
PFX = "SMOKE_" if SMOKE else ""
N_WIN = 24 if SMOKE else 2399
IDX = np.linspace(0, 2398, 24).round().astype(int) if SMOKE else np.arange(2399)
ARMS = ("MCH", "MCF", "STEEP")
NAME = {"MCH": "MC-H", "MCF": "MC-F", "STEEP": "STEEP"}
INPUTS = f"{HMC}/inputs"
FLOOR_OL_ADE6 = 2.48699212                                                  # DESIGN_T1_V8F.md section 8, T1KIN's C-2
t_start = time.time()
os.makedirs(OUT, exist_ok=True)
REP = {"prereg": "Project Steering/PREREG_V8F_HACC_MC.md @ 8d4b936d", "tier": "T1-kin (self-action OPEN loop; never 'closed loop')", "tool_tree": TREE,
       "SMOKE": SMOKE, "label": "SMOKE: plumbing only, not outcome-bearing" if SMOKE else "REAL",
       "estimator": "taniteval.ci.paired_episode_cluster_bootstrap, B 2000, seed 0, per seed, arm - P10 and arm - HACC on the same windows. It answers 'another draw of EPISODES'; it is blind to training "
                    "variance (H-ESTIM-SEED-1) and the two training seeds are the only replicate; no inference seed is varied (this planner is deterministic).",
       "amendments_applied": {"c1_bar": A.c1_bar, "c1_amendment": A.c1_amendment, "c4_clause2": A.c4_clause2, "c4_amendment": A.c4_amendment},
       "controls": {}}
BR = json.load(open(f"{HMC}/BAKE_REPORT_MC.json"))


def load_banked():
    f64 = {s: L.load_npz(f"{INPUTS}/banked_causal_f64/dumps_HACC_s{s}_m10_causal_f64.npz") for s in L.SEEDS}
    f32 = {s: L.load_npz(f"{INPUTS}/banked_causal/dumps_HACC_s{s}_m10_causal.npz") for s in L.SEEDS}
    sub = lambda d: {k: (v[IDX] if hasattr(v, "shape") and v.ndim >= 1 and len(v) == 2399 else v) for k, v in d.items()}
    return {s: sub(f64[s]) for s in L.SEEDS}, {s: sub(f32[s]) for s in L.SEEDS}


HB, HB32 = load_banked()
SFK, ST, T1, DDATA = C.bind_statekin(TREE)


def tagof(arm, s, m):
    return f"{PFX}{arm}_s{s}_m{m}"


def load_arm(arm, s, m):
    t = tagof(arm, s, m)
    d = L.load_run_dumps(RUN, t)
    d["json"] = json.load(open(f"{RUN}/t1v8f_{t}.json", encoding="utf-8"))
    d["ticks"] = C.load_ticks(f"{RUN}/ticks/ticks_{t}.npz")
    d["tag"] = t
    return d


DATA = {}
missing = []
for arm in ARMS:
    for s in L.SEEDS:
        for m in (10, 30):
            try:
                DATA[(arm, s, m)] = load_arm(arm, s, m)
            except (AssertionError, FileNotFoundError, OSError) as e:
                missing.append(f"{tagof(arm, s, m)}: {e}")
if missing:
    REP["controls"]["P-1"] = {"PASS": False, "missing_runs": missing}
    json.dump(REP, open(f"{OUT}/READ_CONTROLS.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("REFUSED: runs missing:", missing[:3])
    sys.exit(0)

CK = {(arm, s): BR["ckpts"][f"s{s}"]["arms"][NAME[arm]] for arm in ARMS for s in L.SEEDS}
TAB = {(arm, s): C.table_from_ckpt(CK[(arm, s)]["out"]) for arm in ARMS for s in L.SEEDS}      # the float32 hacc_w buffer each loop actually ran
W_HACC = {s: C.table_from_ckpt(f"{L.HACC_ART}/M2b_s{s}_hacc.pt") for s in L.SEEDS}
W_STEEP = {s: TAB[("STEEP", s)] for s in L.SEEDS}


# ================================================================================================ P-1 provenance (the runs are the pre-registered configuration of the baked checkpoints)
def p1():
    out, ok = {}, True
    for (arm, s, m), d in DATA.items():
        h, z = d["json"]["t1v8f"], d["ticks"]
        ck = CK[(arm, s)]
        want_limit = "causal"
        chk = {"replan_steps": h["replan_steps"] == m, "scene_logged": h["scene"] == "logged", "nav_deadreckon": h["nav"] == "deadreckon", "schedule_periodic": h["schedule"] == "periodic",
               "limit": h["limit"] == want_limit or (SMOKE and h["limit"] in ("causal", "logged")), "n_windows": h["n_windows"] == N_WIN,
               "causal_bucket_header": ("causal_bucket" in h) or (SMOKE and h["limit"] == "logged"), "ticks_split_eval": str(z["split"]) == "eval", "ticks_replan_steps": int(z["replan_steps"]) == m,
               "ticks_ckpt_is_the_baked_file": os.path.normpath(str(z["ckpt"])) == os.path.normpath(ck["out"]), "header_ckpt_is_the_baked_file": os.path.normpath(str(h["ckpt"])) == os.path.normpath(ck["out"]),
               "ckpt_sha256_is_the_reported_one": L.sha256_file(ck["out"]) == ck["sha256"], "n_ticks": len(z["tau"]) == N_WIN * (30 // m),
               "episode_files": d["n_files"] == (141 if not SMOKE else d["n_files"]), "no_hacc_gate": "hacc_gate_v" not in h,
               "table_in_loop_is_the_baked_table": bool(np.array_equal(TAB[(arm, s)].astype(np.float32), np.stack([
                   np.asarray(json.load(open(ck["table_file"]))["w_along_by_step"]), np.asarray(json.load(open(ck["table_file"]))["w_cross_by_step"])], 1).astype(np.float32))),
               "window_order_equals_banked": bool((d["clip_index"] == HB[s]["clip_index"]).all() and (d["ws"] == HB[s]["ws"]).all() and np.array_equal(d["g"], HB[s]["g"]))}
        out[d["tag"]] = {"checks": chk, "PASS": all(chk.values()), "limit": h["limit"]}
        ok &= out[d["tag"]]["PASS"]
    return {"PASS": bool(ok), "runs": out, "what": "each run is the pre-registered configuration of the right baked checkpoint (guards against a wrong ckpt / wrong flags reading as a result)"}


REP["controls"]["P-1"] = p1()


# ================================================================================================ C-1 reachability
def c1():
    res, ok = {}, True
    for arm in ("MCH", "MCF"):
        for s in L.SEEDS:
            for m in (10, 30):
                r = C.c1_reach(DATA[(arm, s, m)]["ticks"], TAB[(arm, s)])
                r["const_c_float32"] = float(TAB[(arm, s)][0, 0])
                r["table_constant_over_knots_1_12"] = bool(np.all(TAB[(arm, s)][:12, 0] == TAB[(arm, s)][0, 0]))
                r["PASS_literal_1e-6"] = r["PASS"]
                r["PASS_ulp32"] = bool(r["max_d_over_ulp32_of_plan"] <= 1.0)
                r["PASS"] = (r["PASS_literal_1e-6"] if A.c1_bar == "literal" else r["PASS_ulp32"]) and r["table_constant_over_knots_1_12"]
                res[f"{arm}_s{s}_m{m}"] = r
                ok &= r["PASS"]
    reg = {}
    for s in L.SEEDS:                                       # deliberate regression: the HACC table through the same check, on the MC-H ticks -> must FAIL
        r = C.c1_reach(DATA[("MCH", s, 10)]["ticks"], W_HACC[s])
        r["MUST_FAIL"] = True
        r["regression_fails_as_it_must"] = bool(not r["PASS"] and r["max_abs_d_m"] > 1e-3)
        reg[f"HACC_table_on_MCH_s{s}_m10"] = r
        ok &= r["regression_fails_as_it_must"]
    return {"PASS": bool(ok), "bar": "max|d| <= 1e-6 m (literal)" if A.c1_bar == "literal" else f"max|d| <= 1 ulp32 of |plan| (AMENDMENT {A.c1_amendment})",
            "runs": res, "deliberate_regression": reg}


REP["controls"]["C-1"] = c1()


# ================================================================================================ C-2 the post-map is in the loop
def c2():
    res, ok = {}, True
    for s in L.SEEDS:
        for arm in ARMS:
            v = float(np.abs(DATA[(arm, s, 10)]["cl"] - HB[s]["cl"]).max())
            res[f"{arm}_s{s}"] = {"max_abs_cl_minus_HACC_banked_m": v, "gating": arm == "MCH", "PASS": v > 1e-2}
            ok &= (res[f"{arm}_s{s}"]["PASS"] or arm != "MCH")
    return {"PASS": bool(ok), "bar": "max|MC-H - HACC| executed at m = 10 > 1e-2 m (MC-F, STEEP beside)", "runs": res}


REP["controls"]["C-2"] = c2()


# ================================================================================================ C-3 the along-ramp instrument
def nonzero(sm):
    return bool(sm["frac_ticks_nonzero_a0"] > 0.5 and sm["max_abs_a0_mps2"] > 1e-3)


def c3():
    res, reg, ok = {}, {}, True
    for arm in ("MCH", "MCF"):
        for s in L.SEEDS:
            for m in (10, 30):
                sm = C.c3_summary(C.ramp_term(DATA[(arm, s, m)]["ticks"], TAB[(arm, s)], ST))
                sm["PASS"] = sm["is_zero"]
                res[f"{arm}_s{s}_m{m}"] = sm
                ok &= sm["PASS"]
    for s in L.SEEDS:
        z10 = DATA[("MCH", s, 10)]["ticks"]
        for nm, z, w in ((f"MCH_ticks_HACC_table_s{s}", z10, W_HACC[s]), (f"MCH_ticks_STEEP_table_s{s}", z10, W_STEEP[s]), (f"STEEP_own_ticks_STEEP_table_s{s}", DATA[("STEEP", s, 10)]["ticks"], W_STEEP[s])):
            sm = C.c3_summary(C.ramp_term(z, w, ST))
            sm["MUST_BE_NONZERO"] = True
            sm["regression_fires_as_it_must"] = nonzero(sm)
            reg[nm] = sm
            ok &= sm["regression_fires_as_it_must"]
        kv = C.c3_train_known(ST, s, W_HACC[s])
        reg[f"HACC_TRAIN_known_value_s{s}"] = kv
        ok &= kv["PASS"]
    return {"PASS": bool(ok), "bar": "exactly 0.0 (<= 1e-9) on MC-H and MC-F ticks (v0, a0, kappa0)", "runs": res, "deliberate_regression": reg,
            "note": "HACC's own EVAL ticks are not in the pre-registered matrix: HACC's regression is its table through the same instrument on the MC-H ticks, plus its TRAIN known value (-0.1588 / -0.0726 +- 2e-3, n = 1,747)"}


REP["controls"]["C-3"] = c3()


# ================================================================================================ C-4 (from the bake report) + the TRAIN qualification bar
def c4():
    t = BR["tables"]
    out = {"clause1": {}, "clause2": {}, "ok": True}
    for s in L.SEEDS:
        c = t["seeds"][f"s{s}"]["C-4"]
        out["clause1"][f"s{s}"] = c["clause1_closed_form_vs_grid_argmin"]
        c2_ = c["clause2_regression_GT_is_P_gives_c_zero"]
        if A.c4_clause2 == "literal":
            out["clause2"][f"s{s}"] = {**c2_, "variant": "literal PREREG text"}
        else:
            v = c["beside_NOT_A_CONTROL"]["c_regression_with_held_tail_also_zeroed"]
            out["clause2"][f"s{s}"] = {"c": v, "PASS": bool(abs(v) <= 1e-9), "variant": f"tail0 (AMENDMENT {A.c4_amendment}): GT := P with the held tail also zeroed",
                                       "literal_value_for_the_record": c2_["c"]}
        out["ok"] &= bool(out["clause1"][f"s{s}"]["PASS"] and out["clause2"][f"s{s}"]["PASS"])
    bar = {f"s{s}": t["seeds"][f"s{s}"]["TRAIN_BAR_PASS"] for s in L.SEEDS}
    repro = bool(t["reproduces_probe12_and_prereg_values"])
    return {"PASS": bool(out["ok"] and all(bar.values()) and repro), "C-4": out, "TRAIN_qualification_bar_ade6_le_HACC_plus_F_half": bar, "tables_reproduce_probe12_and_PREREG_values": repro,
            "source": "BAKE_REPORT_MC.json (built before any eval run)"}


REP["controls"]["C-4"] = c4()


# ================================================================================================ C-5 the floor (T1KIN's identities)
def c5():
    import glob  # noqa: F401
    wins = DDATA.eval_grid_windows("C:/Users/Admin/hlat/inputs/eval_grid_windows.json", DDATA.load_join("C:/Users/Admin/hlat/inputs/join_eval141.npz"))     # md5-pinned by the chain
    gidx = {(int(w[0]), int(w[1])): i for i, w in enumerate(wins)}
    res, ok = {}, True
    for s in L.SEEDS:
        hbl = np.load(f"{INPUTS}/hblend_score_M2b_s{s}_perwindow.npz")
        bank = np.load(f"{INPUTS}/hacc_score_M2b_s{s}_perwindow.npz")
        for arm in ARMS:
            d30, d10 = DATA[(arm, s, 30)], DATA[(arm, s, 10)]
            gi = np.array([gidx[(int(c), int(w))] for c, w in zip(d30["clip_index"], d30["ws"])])
            r = {"max_abs_cl_minus_ol_m30_m": float(np.abs(d30["cl"] - d30["ol"]).max()),
                 "max_abs_floor_vs_banked_P10_plan_m": float(np.abs(d30["fl"] - hbl["P"][gi]).max()), "grid_indices_unique": bool(len(np.unique(gi)) == len(gi)), "n_windows": int(len(gi)),
                 "eid_equals_banked_grid": bool((d30["eid"] == bank["eid"][gi]).all()),
                 "floor_m30_ade6_m": float(np.linalg.norm(d30["fl"] - d30["g"], axis=-1).mean()),
                 "floor_m10_max_abs_vs_banked_HACC_dump_fl_m": float(np.abs(d10["fl"] - HB[s]["fl"]).max())}
            r["C1i_open_loop_identity"] = r["max_abs_cl_minus_ol_m30_m"] <= 1e-9
            r["C1ii_age_free_identity"] = bool(r["max_abs_floor_vs_banked_P10_plan_m"] <= 1e-4 and r["grid_indices_unique"] and r["eid_equals_banked_grid"] and (r["n_windows"] == 2399 or SMOKE))
            r["C2_floor_ade6"] = bool(abs(r["floor_m30_ade6_m"] - FLOOR_OL_ADE6) <= 1e-5) if not SMOKE else None
            r["P10_loop_m10_equals_banked"] = bool(r["floor_m10_max_abs_vs_banked_HACC_dump_fl_m"] <= 1e-9)
            r["PASS"] = bool(r["C1i_open_loop_identity"] and r["C1ii_age_free_identity"] and (r["C2_floor_ade6"] in (True, None)) and r["P10_loop_m10_equals_banked"])
            if SMOKE:
                r["SMOKE_NOTE"] = "24 windows in one batch: the loop amplifies float32 batch-composition noise ~1000x (MEASURED: batch 24 vs 12 moves fl by 6e-4 m), so the m10 floor identity is expected to fail here"
            res[f"{arm}_s{s}"] = r
            ok &= r["PASS"]
    return {"PASS": bool(ok), "bar": "T1KIN's C-1(i) <= 1e-9, C-1(ii) <= 1e-4 + unique grid + eid, C-2 floor ADE6 = 2.48699212 +- 1e-5 (m30), plus the m10 P10 loop = the banked HACC dump's floor <= 1e-9 "
                                    "(same windows, batch 32, 8 threads: a same-composition identity)", "runs": res}


REP["controls"]["C-5"] = c5()
REP["controls"]["C-6"] = L.c6_reader_control()
c6_64 = {}
for s in L.SEEDS:
    r = L.paired(L.lat_rows(HB[s]["cl"], HB[s]["g"])["L2_curvature_mae_fast_1pm"], L.lat_rows(HB[s]["fl"], HB[s]["g"])["L2_curvature_mae_fast_1pm"], HB[s]["eid"]) if not SMOKE else None
    c6_64[f"s{s}"] = r
REP["controls"]["C-6"]["beside_float64_originals_L2_HACC_minus_P10"] = c6_64


def c6b():
    """Reader check beside C-6 (a reproduction of the PREREG's own HACC numbers): M1 / M2 of HACC from the banked causal dumps (float32 bank, as probe2 computed them) at 4 dp."""
    want = {"M1": (0.0536, 0.0277), "M2": (0.0202, 0.0072), "M1_P10": (0.0137, 0.0137)}
    res, ok = {}, True
    for s in L.SEEDS:
        b = L.banked_hacc(s)                                  # float32, all 2,399 windows (the control is the full-set value even in SMOKE)
        mr = C.m_rows(b["cl"], b["fl"], b["g"])
        r = {k: mr[k] for k in want}
        r["matches_4dp"] = {k: bool(abs(mr[k] - want[k][s]) <= 5e-5) for k in want}
        res[f"s{s}"] = r
        ok &= all(r["matches_4dp"].values())
    return {"PASS": bool(ok), "what": "M1 / M2 / M1(P10) of HACC reproduce the PREREG's printed 0.0536 / 0.0277, +0.0202 / +0.0072, 0.0137", "runs": res}


REP["controls"]["C-6b"] = c6b()


def c7():
    p = f"{LOGS}/suite.ok"
    if SMOKE:
        return {"PASS": True, "note": "SMOKE: the suite is not run by the smoke chain"}
    if not os.path.exists(p):
        return {"PASS": False, "note": f"{p} missing: the chain's suite did not pass (or has not run); see {LOGS}/suite.out"}
    last = open(p).read().strip()
    return {"PASS": bool(" passed" in last and "failed" not in last and "error" not in last), "suite_last_line": last}


REP["controls"]["C-7"] = c7()

# ------------------------------------------------------------------------------------------------ the control table
order = ["P-1", "C-1", "C-2", "C-3", "C-4", "C-5", "C-6", "C-6b", "C-7"]
failed = [k for k in order if not REP["controls"][k]["PASS"]]
REP["controls_failed"] = failed
REP["controls_all_pass"] = not failed
print(("SMOKE " if SMOKE else "") + "CONTROLS: " + "  ".join(f"{k} {'PASS' if REP['controls'][k]['PASS'] else 'FAIL'}" for k in order))
for k in order:
    c = REP["controls"][k]
    if k == "C-1":
        for tg, r in c["runs"].items():
            print(f"   C-1 {tg:14s} max|d| {r['max_abs_d_m']:.3e} m  = {r['max_d_over_ulp32_of_plan']:.2f} ulp32  literal {r['PASS_literal_1e-6']}  ulp32 {r['PASS_ulp32']}  (max |plan| {r['max_abs_plan_along_m']:.1f} m)")
        for tg, r in c["deliberate_regression"].items():
            print(f"   C-1 REGRESSION {tg}: max|d| {r['max_abs_d_m']:.3e} m -> fails as it must: {r['regression_fails_as_it_must']}")
    if k == "C-4":
        for s in L.SEEDS:
            print(f"   C-4 s{s}: clause1 {c['C-4']['clause1'][f's{s}']['c_closed_form']:.4f} vs grid {c['C-4']['clause1'][f's{s}']['c_grid_argmin_201pt']:.3f} PASS {c['C-4']['clause1'][f's{s}']['PASS']} | "
                  f"clause2 ({c['C-4']['clause2'][f's{s}']['variant']}) c = {c['C-4']['clause2'][f's{s}']['c']:.4f} PASS {c['C-4']['clause2'][f's{s}']['PASS']}")
REP["seconds_controls"] = time.time() - t_start

JSONF = lambda o: o.tolist() if hasattr(o, "tolist") else (o.item() if hasattr(o, "item") else str(o))
if A.controls_only:
    json.dump(REP, open(f"{OUT}/READ_CONTROLS.json", "w"), indent=1, default=JSONF)
    print("CONTROLS-ONLY: " + ("ALL PASS" if not failed else "FAILED: " + ", ".join(failed)) + f"  ({REP['seconds_controls']:.0f}s)  -> {OUT}/READ_CONTROLS.json")
    sys.exit(0)

# ================================================================================================ rows (only past the controls)
if failed and not A.force_rows_under_refusal:
    REP["OUTCOME"] = {"outcome": "REFUSED", "refused_on": failed, "text": "any control fails -> no outcome row (PREREG section 6)"}
    json.dump(REP, open(f"{OUT}/HMC_READ.json", "w"), indent=1, default=JSONF)
    print("OUTCOME REFUSED on", failed)
    sys.exit(0)
ROW_FILE = f"{OUT}/HMC_READ.json" if not failed else f"{OUT}/rows_UNDER_REFUSAL.json"
ALLROWS = list(L.ROWS) + list(L.LROWS)
F_BARS = {"F_half_0.06035": L.F_HALF, "F_half_printed_0.0604": L.F_HALF_PRINTED, "F_half_hlat_0.06037": 0.06037}


def sepworse(r):
    return bool(r["separated"] and r["delta"] > 0)


def joint_mask_L2(X, Fl, G, eid):
    """lcdiag probe2 (b), lines verbatim: L2 on the JOINT own-speed mask (GT, the arm's own and P10's own pair speed all >= 5.2 m/s: the SAME pairs for both arms)."""
    pa, pf = C.pair_err(X, G), C.pair_err(Fl, G)
    jm = pa["fast"] & pf["fast"] & (pa["vX"] >= L.VMIN) & (pf["vX"] >= L.VMIN)
    cj = jm.sum(1)
    okj = cj > 0
    sa = np.where(jm, pa["e"], 0).sum(1) / np.maximum(cj, 1)
    sf = np.where(jm, pf["e"], 0).sum(1) / np.maximum(cj, 1)
    return L.paired(sa[okj], sf[okj], eid[okj]), float(jm.sum() / pa["fast"].sum())


ROWS_OUT = {"seeds": {}}
HACC_ROWS, HACC_M = {}, {}
for s in L.SEEDS:
    G = HB[s]["g"]
    HACC_ROWS[s] = L.rows_of(HB[s]["cl"], G)
    HACC_M[s] = C.m_rows(HB[s]["cl"], HB[s]["fl"], G)
    # reproduction control of the beside row: HACC's joint-mask L2 (diagnosis: +0.0018 / +0.0015)
    jm, kept = joint_mask_L2(HB[s]["cl"], HB[s]["fl"], G, HB[s]["eid"])
    ROWS_OUT.setdefault("HACC_reference", {})[f"s{s}"] = {"M1": HACC_M[s]["M1"], "M2": HACC_M[s]["M2"], "M1_P10": HACC_M[s]["M1_P10"], "joint_mask_L2_HACC_minus_P10": jm,
                                                         "joint_mask_matches_diagnosis_4dp": bool(abs(jm["delta"] - {0: 0.0018, 1: 0.0015}[s]) <= 1e-4)}

for s in L.SEEDS:
    G = HB[s]["g"]
    S = {"arms": {}}
    for arm in ARMS:
        d = DATA[(arm, s, 10)]
        Cx, Cf = L.rows_of(d["cl"], d["g"]), L.rows_of(d["fl"], d["g"])
        eid = d["eid"]
        vs_p10 = {r: L.paired(Cx[r], Cf[r], eid) for r in ALLROWS}
        vs_hacc = {r: L.paired(Cx[r], HACC_ROWS[s][r], eid) for r in ALLROWS}
        for tbl in (vs_p10, vs_hacc):
            for r in ALLROWS:
                tbl[r]["separated_worse"] = sepworse(tbl[r])
                tbl[r]["family"] = {"ade2_m": "ADE", "ade6_m": "ADE", "speed_mae_mps": "LONGITUDINAL", "along_mae_m": "LONGITUDINAL", "accel_mae_mps2": "LONGITUDINAL",
                                    "heading_mae_deg": "LATERAL", "yaw_rate_mae_degps": "LATERAL", "curvature_mae_1pm": "LATERAL", "cross_mae_m": "LATERAL"}.get(r, "LATERAL")
        mr = C.m_rows(d["cl"], d["fl"], d["g"])
        arm_out = {"rows_arm_minus_P10": vs_p10, "rows_arm_minus_HACC": vs_hacc, "M1": mr["M1"], "M2": mr["M2"], "M1_P10": mr["M1_P10"], "n_arm_crawl_pairs": mr["n_arm_crawl_pairs"],
                   "M1_minus_HACC_beside": C.m1_diff_ci(mr["_slow_a"], mr["_fast_a"], HACC_M[s]["_slow_a"], HACC_M[s]["_fast_a"], eid),
                   "mean_ade6_m": {"arm": float(Cx["ade6_m"].mean()), "P10": float(Cf["ade6_m"].mean()), "HACC": float(HACC_ROWS[s]["ade6_m"].mean())}}
        # beside: the joint own-speed mask, and a GT-defined launch subset (v0 < 6 m/s and GT speed at 6 s >= v0 + 1 m/s)
        jm, kept = joint_mask_L2(d["cl"], d["fl"], d["g"], eid)
        vG6 = np.linalg.norm(d["g"][:, 29] - d["g"][:, 28], axis=-1) / L.DT
        sub = (d["v0"] < 6.0) & (vG6 >= d["v0"] + 1.0)
        launch = {"n_windows": int(sub.sum()), "n_episodes": int(len(np.unique(eid[sub]))), "definition": "v0 (ego speed at t0) < 6 m/s and the GT chord speed over the last 0.2 s of the 6 s horizon >= v0 + 1 m/s"}
        if sub.sum() >= 5:
            for r in ("ade6_m", "L2_curvature_mae_fast_1pm", "accel_mae_mps2", "along_mae_m"):
                launch[f"{r}_arm_minus_P10"] = L.paired(Cx[r][sub], Cf[r][sub], eid[sub])
                launch[f"{r}_arm_minus_HACC"] = L.paired(Cx[r][sub], HACC_ROWS[s][r][sub], eid[sub])
        arm_out["beside"] = {"L2_joint_own_speed_mask_arm_minus_P10": jm, "joint_mask_fraction_of_fast_pairs_kept": kept, "launch_subset": launch,
                             "families_not_computed": {"TACTICAL": "NOT COMPUTED: the T1 dumps record no per-tick T code (work item)", "STRATEGIC": "PAUSED (PI_DECISION_QUEUE ITEM 25) and not computed: no per-tick S checkpoints in the T1 dumps",
                                                       "DISTANCE_KEEPING": "NOT COMPUTED: v8f_t1_eval passes no lead track to analyze()"}}
        # beside: m = 30 (the tick-0 plan executed whole), arm - P10
        d30 = DATA[(arm, s, 30)]
        C30, F30 = L.rows_of(d30["cl"], d30["g"]), L.rows_of(d30["fl"], d30["g"])
        arm_out["beside"]["m30_arm_minus_P10"] = {r: L.paired(C30[r], F30[r], d30["eid"]) for r in ALLROWS}
        # R-2 (informational, as T1KIN's): the reader's ADE6 delta equals the tool's own analyze() delta
        try:
            js6 = d["json"]["paired_decision_grade"]["paired_model_minus_floor_t1"]["ade_m"]
            arm_out["R2_reader_vs_tool_analyze_ade6_delta"] = {"reader": vs_p10["ade6_m"]["delta"], "tool": float(js6["delta"]), "abs_diff": abs(vs_p10["ade6_m"]["delta"] - float(js6["delta"]))}
        except (KeyError, TypeError):
            arm_out["R2_reader_vs_tool_analyze_ade6_delta"] = {"status": "tool json carries no paired_decision_grade"}
        S["arms"][NAME[arm]] = arm_out
    ROWS_OUT["seeds"][f"s{s}"] = S

# ------------------------------------------------------------------------------------------------ the outcome (PREREG section 6, exactly as the table words)
H = {s: ROWS_OUT["seeds"][f"s{s}"]["arms"]["MC-H"] for s in L.SEEDS}
ST_ = {s: ROWS_OUT["seeds"][f"s{s}"]["arms"]["STEEP"] for s in L.SEEDS}
l2_not_worse = {s: not H[s]["rows_arm_minus_P10"]["L2_curvature_mae_fast_1pm"]["separated_worse"] for s in L.SEEDS}
ade6_hi = {s: H[s]["rows_arm_minus_HACC"]["ade6_m"]["hi"] for s in L.SEEDS}
keep = {nm: {s: bool(ade6_hi[s] <= bar) for s in L.SEEDS} for nm, bar in F_BARS.items()}
m1_h_lt = {s: bool(H[s]["M1"] < HACC_M[s]["M1"]) for s in L.SEEDS}
m1_st_gt = {s: bool(ST_[s]["M1"] > HACC_M[s]["M1"]) for s in L.SEEDS}
both = lambda d: all(d.values())
cond = {
    "FIXED-KEEP": both(l2_not_worse) and both(keep["F_half_0.06035"]) and both(m1_h_lt),
    "FIXED-NOKEEP": both(l2_not_worse) and not both(keep["F_half_0.06035"]),
    "MECHANISM-NOT-ENOUGH": both(m1_h_lt) and both(m1_st_gt) and not both(l2_not_worse),
    "MECHANISM-REFUTED": (not both(m1_h_lt)) or (not both(m1_st_gt)),
}
matching = [k for k in ("FIXED-KEEP", "FIXED-NOKEEP", "MECHANISM-NOT-ENOUGH", "MECHANISM-REFUTED") if cond[k]]
word = matching[0] if matching else "NO ROW MATCHES (should be impossible: file a defect)"
flip = {nm: [k for k in ("FIXED-KEEP", "FIXED-NOKEEP") if (both(l2_not_worse) and ((both(keep[nm]) and both(m1_h_lt)) if k == "FIXED-KEEP" else (not both(keep[nm]))))] for nm in F_BARS}
NEXT = {"FIXED-KEEP": "adopt MC-H as the T1 post-map and re-read DOMINANCE (T1KIN rows) under its own read; then m = 5",
        "FIXED-NOKEEP": "a zero-training correction of P10's executed-horizon along bias: a linear (v0, a0) -> along-offset map over knots 1-12, fitted on TRAIN tick-0 samples, applied under MC-H, under its own PREREG",
        "MECHANISM-NOT-ENOUGH": "the box-2 TRAINING lever (O-RECOVER, PREREG section 7)",
        "MECHANISM-REFUTED": "the box-2 TRAINING lever (O-RECOVER, PREREG section 7); the diagnosis is re-opened on O's launch from the logged state"}
REP["rows"] = ROWS_OUT
REP["outcome_inputs"] = {"decisive_arm": "MC-H", "L2_MCH_minus_P10_not_separated_worse": l2_not_worse, "MCH_minus_HACC_ADE6_upper_bound_m": ade6_hi, "KEEP_by_F_half_reading": keep,
                         "M1": {"HACC": {s: HACC_M[s]["M1"] for s in L.SEEDS}, "MC-H": {s: H[s]["M1"] for s in L.SEEDS}, "STEEP": {s: ST_[s]["M1"] for s in L.SEEDS}, "MC-F": {s: ROWS_OUT["seeds"][f"s{s}"]["arms"]["MC-F"]["M1"] for s in L.SEEDS}},
                         "M1_MCH_lt_HACC": m1_h_lt, "M1_STEEP_gt_HACC": m1_st_gt, "F_half_used": L.F_HALF}
REP["OUTCOME"] = {"outcome": word if not failed else "REFUSED (rows below are NOT an outcome)", "matching_rows_in_table_order": matching, "first_match_reported": True,
                  "ambiguity": ("more than one table row's condition holds; the first in table order is the word, all are listed" if len(matching) > 1 else None),
                  "committed_next": NEXT.get(word), "KEEP_bar_sensitivity_rows_matched_if_bar_were": flip,
                  "note": "MC-F and m = 30 are beside: a MC-F win does not promote MC-F without its own read."}
json.dump(REP, open(ROW_FILE, "w"), indent=1, default=JSONF)
print(f"OUTCOME {REP['OUTCOME']['outcome']}  (matching rows: {matching})  -> {ROW_FILE}")
for s in L.SEEDS:
    for nm in ("MC-H", "MC-F", "STEEP"):
        a = ROWS_OUT["seeds"][f"s{s}"]["arms"][nm]
        r = a["rows_arm_minus_P10"]["L2_curvature_mae_fast_1pm"]
        rh = a["rows_arm_minus_HACC"]["ade6_m"]
        print(f"  s{s} {nm:6s} L2-P10 {r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] sep {r['separated']} | ADE6-HACC {rh['delta']:+.4f} hi {rh['hi']:+.4f} | M1 {a['M1']:.4f} (HACC {HACC_M[s]['M1']:.4f}) M2 {a['M2']:+.4f}")
