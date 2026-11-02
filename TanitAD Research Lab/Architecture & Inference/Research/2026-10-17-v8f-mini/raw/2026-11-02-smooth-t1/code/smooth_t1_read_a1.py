#!/usr/bin/env python3
"""smooth_t1_read_a1.py -- H-SMOOTH-T1-1 under AMENDMENT_1 (control-only, post-data, disclosed): the decision contrast is X - Y' (fresh --smooth off rollouts, 16 threads, same session), X - P10 uses the banked P10 rows,
the K1 / K2 / K5-floor tolerances are max(1e-4, the measured |Y' - Y''| maxima) from raw_a1/NONDET_BOUND.json.  Zero training, CPU only.

    python smooth_t1_read_a1.py --out-dir F:/TanitAD-scratch/smooth_t1/out_m16 --raw <package>/raw_a1

The outcome rule, rows, bars and estimator are the frozen card's: better / worse / read_a / read_xy / arm_outcome / the 19 rows (gapdecomp.decompose) are IMPORTED UNCHANGED from smooth_t1_read.py.
Order: gate hashes -> NONDET_BOUND.json must exist (written by a1_nondet_bound.py before this runs) -> extract -> controls (written to the JSON) -> rows -> outcomes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import smooth_t1_read as R0  # noqa: E402
import a1_nondet_bound as NB  # noqa: E402

PK = os.path.dirname(HERE)
X_RUNS = {"CTRL_s0": ("SMOOTH_DAGOACC_s0", "CTRL", 0), "CTRL_s1": ("SMOOTH_DAGOACC_s1", "CTRL", 1), "SOUP_s0": ("SMOOTH_SOUP_s0", "SOUP", 0), "SOUP_s1": ("SMOOTH_SOUP_s1", "SOUP", 1)}
YP_RUNS = {"CTRL_s0": ("K1OFF_DAGOACC_s0", "CTRL", 0), "CTRL_s1": ("OFF_DAGOACC_s1", "CTRL", 1), "SOUP_s0": ("OFF_SOUP_s0", "SOUP", 0), "SOUP_s1": ("OFF_SOUP_s1", "SOUP", 1)}
YPP = ("OFFREP_DAGOACC_s0", "CTRL", 0)
TOOL_COPY = f"{PK}/code/v8f_t1_eval.py"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    t_start = time.time()
    want = [ln.split()[2] for ln in open(f"{PK}/FROZEN_SHA.txt", encoding="utf-8") if ln.startswith("PREREG_FROZEN.md sha256")][0]
    want2 = [ln.split()[2] for ln in open(f"{PK}/AMENDMENT_1_SHA.txt", encoding="utf-8") if ln.startswith("AMENDMENT_1.md sha256")][0]
    got, got2 = R0.sha256_file(f"{PK}/PREREG_FROZEN.md"), R0.sha256_file(f"{PK}/AMENDMENT_1.md")
    if want != got or want2 != got2:
        print("REFUSED: card or amendment does not hash to its frozen sha256", file=sys.stderr)
        return 2
    if R0.sha256_file(R0.SOUP_READ) != R0.PIN_SOUP_READ:
        print("REFUSED: SOUP_READ.json is not the pinned banked read", file=sys.stderr)
        return 2
    nb_path = f"{a.raw}/NONDET_BOUND.json"
    if not os.path.exists(nb_path):
        print("REFUSED: raw_a1/NONDET_BOUND.json is missing -- the tolerance must be measured and written before any row is computed (AMENDMENT_1 section 3)", file=sys.stderr)
        return 2
    nb = json.load(open(nb_path, encoding="utf-8"))
    tol = nb["tol"]
    GD = R0.load_gd()
    arr_dir = f"{a.raw}/arrays"
    for tag, arm, seed in list(X_RUNS.values()) + list(YP_RUNS.values()) + [YPP]:
        NB.extract_run(tag, arm, seed, a.out_dir, arr_dir)
    man = json.load(open(R0.MANIFEST, encoding="utf-8"))
    banked = json.load(open(R0.SOUP_READ, encoding="utf-8"))["tables"]
    bk_tbl = {"CTRL": banked["CTRL_P10"], "SOUP": banked["SOUP_P10"]}
    res = {"hypothesis": "H-SMOOTH-T1-1 under AMENDMENT_1", "tier": "T1-kin", "evidence_class": "MEASURED (ours; artifacts in MANIFEST.txt)", "prereg_sha256": got, "amendment_sha256": got2,
           "nondet_bound_sha256": R0.sha256_file(nb_path), "tolerances": tol, "nondet_N": nb["N"],
           "estimator": f"paired episode-cluster bootstrap, B={a.n_boot}, seed 0 (taniteval.ci draw stream, via gapdecomp.decompose); answers another draw of EPISODES only",
           "run_clock": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "controls": {}, "arms": {}, "descriptive": {}}
    ctl = res["controls"]
    Z = {k: np.load(f"{arr_dir}/{v[0]}.npz", allow_pickle=True) for k, v in X_RUNS.items()}
    ZP = {k: np.load(f"{arr_dir}/{v[0]}.npz", allow_pickle=True) for k, v in YP_RUNS.items()}
    Y = {(arm, s): np.load(f"{R0.GD_DIR}/raw/arrays/{arm}_s{s}.npz", allow_pickle=True) for arm in ("CTRL", "SOUP") for s in (0, 1)}
    labels = tuple(str(x) for x in Y[("CTRL", 0)]["labels"])
    assert labels == GD.LABELS
    cls0, _m0 = GD.classify(Y[("CTRL", 0)]["v0"], Y[("CTRL", 0)]["g"])

    # ---- K1': Y' vs banked (DAG-OACC s0 is the card's K1 arm; the other three are descriptive)
    k1 = {}
    for k, (tag, arm, seed) in YP_RUNS.items():
        eb = man["arms"][arm][str(seed)]["t1"]
        n1, dk = R0.load_dump(f"{a.out_dir}/{tag}/dump_{tag}")
        n2, db = R0.load_dump(eb["dump_dir"])
        tk, tb = NB.load_ticks(f"{a.out_dir}/{tag}/ticks/ticks_{tag}.npz"), NB.load_ticks(eb["ticks"])
        kd, kt = NB.per_key_dev(dk, db), NB.per_key_dev(tk, tb)
        keyset_ok = bool(set(dk) == set(db) and set(tk) == set(tb) and n1 == n2)
        k0a, k0b = tk["tick"] == 0, tb["tick"] == 0
        k1[k] = {"tag": tag, "dump_max_abs_dev": NB.fmax(kd, False), "dump_nonnumeric_diff_keys": NB.nonnumeric(kd), "floor_fl_max_abs_dev": float(kd["fl"]), "ticks_max_abs_dev": NB.fmax(kt, False), "ticks_nonnumeric_diff_keys": NB.nonnumeric(kt), "tick0_plan_max_abs_dev": float(np.abs(tk["plan"][k0a] - tb["plan"][k0b]).max()),
                 "n_tick_float_fields_differing": sum(1 for v in kt.values() if v != 0.0), "keyset_and_filecount_equal": keyset_ok,
                 "n_ticks_plan_dev_gt_1e-4": int((np.abs(tk["plan"] - tb["plan"]).reshape(len(tk["plan"]), -1).max(1) > 1e-4).sum()), "n_ticks": int(len(tk["plan"])),
                 "bit_identical_dump": bool(all(v == 0.0 for v in kd.values())), "bit_identical_ticks": bool(all(v == 0.0 for v in kt.values())),
                 "within_tol_dump": bool(NB.fmax(kd, False) <= tol["tol_dump"]), "within_tol_floor": bool(float(kd["fl"]) <= tol["tol_floor"]), "within_tol_ticks": bool(NB.fmax(kt, False) <= tol["tol_ticks"])}
    kp = k1["CTRL_s0"]
    ctl["K1p_Yprime_vs_banked_DAGOACC_s0"] = {"by_arm_seed": k1, "tol_dump": tol["tol_dump"], "tol_ticks": tol["tol_ticks"], "threads": 16,
                                              "PASS": bool(kp["keyset_and_filecount_equal"] and kp["within_tol_dump"] and kp["within_tol_ticks"])}

    # ---- K2': (Y', Y banked) on the 19 rows (DAG-OACC s0 gated; the others descriptive) and (Y', Y'') from NONDET_BOUND.json
    k2 = {}
    for k in YP_RUNS:
        arm, s = k.split("_s")[0], int(k[-1])
        yb, zp = Y[(arm, s)], ZP[k]
        cls, _ = GD.classify(yb["v0"], yb["g"])
        rr = {}
        for i, lab in enumerate(labels):
            xa, ya = zp[f"a_{i}"], yb[f"a_{i}"]
            m = np.isfinite(xa) & np.isfinite(ya)
            r0 = GD.decompose(xa, ya, yb["eid"], cls, GD.POOL[lab], a.n_boot)
            rr[lab] = {"same_nan_pattern": bool(np.array_equal(np.isfinite(xa), np.isfinite(ya))), "row_delta": r0["delta"], "row_lo": r0["lo"], "row_hi": r0["hi"], "row_separated": r0["separated"],
                       "per_window_max_abs_diff": float(np.abs(xa[m] - ya[m]).max()) if m.any() else 0.0}
        k2[k] = {"rows": rr, "max_abs_row_delta": max(abs(v["row_delta"]) for v in rr.values()), "all_same_nan": all(v["same_nan_pattern"] for v in rr.values()),
                 "within_tol_row": bool(all(v["same_nan_pattern"] and abs(v["row_delta"]) <= tol["tol_row"] for v in rr.values()))}
    ctl["K2p_Yprime_minus_Ybanked_rows"] = {"by_arm_seed": k2, "tol_row": tol["tol_row"], "PASS": bool(k2["CTRL_s0"]["within_tol_row"])}
    rows_pp = nb["rows_Yp_minus_Ypp"]
    ctl["K2pp_Yprime_minus_Ydoubleprime_rows"] = {"max_abs_row_delta": max(abs(v["delta"]) for v in rows_pp.values()), "tol_row": tol["tol_row"], "all_same_nan": nb["same_nan_all_rows"],
                                                  "PASS": bool(nb["same_nan_all_rows"] and all(abs(v["delta"]) <= tol["tol_row"] for v in rows_pp.values()))}

    # ---- K3 / K4 as in the card (banked arrays only)
    k3, k4 = {}, {}
    for (arm, s), z in Y.items():
        cls, _ = GD.classify(z["v0"], z["g"])
        for i, lab in enumerate(labels):
            r = GD.decompose(z[f"a_{i}"], z[f"b_{i}"], z["eid"], cls, GD.POOL[lab], a.n_boot)
            bk = bk_tbl[arm][f"s{s}"][lab]
            k3[f"{arm}_s{s}:{lab}"] = bool(abs(r["delta"] - bk["delta"]) <= R0.TOL_ROW and abs(r["lo"] - bk["lo"]) <= R0.TOL_ROW and abs(r["hi"] - bk["hi"]) <= R0.TOL_ROW and r["separated"] == bk["separated"])
            zr = GD.decompose(z[f"a_{i}"], z[f"a_{i}"], z["eid"], cls, GD.POOL[lab], 200)
            k4[f"{arm}_s{s}:{lab}"] = bool(zr["delta"] == 0.0 and zr["lo"] == 0.0 and zr["hi"] == 0.0)
    ctl["K3_Y_minus_P10_reproduces_SOUP_READ"] = {"n_rows": len(k3), "n_pass": sum(k3.values()), "failed": [k for k, v in k3.items() if not v], "PASS": all(k3.values())}
    ctl["K4_self_minus_self_is_zero"] = {"n_rows": len(k4), "n_pass": sum(k4.values()), "PASS": all(k4.values())}

    # ---- K5' on the smoothed ticks
    from scipy.signal import savgol_filter
    k5 = {}
    for k, (tag, arm, seed) in X_RUNS.items():
        t = dict(np.load(f"{a.out_dir}/{tag}/ticks/ticks_{tag}.npz", allow_pickle=True))
        eb = man["arms"][arm][str(seed)]["t1"]
        tbk = dict(np.load(eb["ticks"], allow_pickle=True))
        typ = dict(np.load(f"{a.out_dir}/{YP_RUNS[k][0]}/ticks/ticks_{YP_RUNS[k][0]}.npz", allow_pickle=True))
        nb_, dbk = R0.load_dump(eb["dump_dir"])
        nx, dx = R0.load_dump(f"{a.out_dir}/{tag}/dump_{tag}")
        npp, dyp = R0.load_dump(f"{a.out_dir}/{YP_RUNS[k][0]}/dump_{YP_RUNS[k][0]}")
        rec = (t["P"] + savgol_filter(t["plan_raw"] - t["P"], 7, 2, axis=1, mode="interp")).astype(np.float32).astype(np.float64)
        k0, kb0, ky0 = t["tick"] == 0, tbk["tick"] == 0, typ["tick"] == 0
        d_raw_bank = float(np.abs(t["plan_raw"][k0] - tbk["plan"][kb0]).max())
        d_raw_yp = float(np.abs(t["plan_raw"][k0] - typ["plan"][ky0]).max())
        d_rec = float(np.abs(rec - t["plan"]).max())
        fl_bank, fl_yp = float(np.abs(dx["fl"] - dbk["fl"]).max()), float(np.abs(dx["fl"] - dyp["fl"]).max())
        s_active = float(np.abs(t["plan"] - t["plan_raw"]).max())
        win_eq = bool(all(np.array_equal(dx[q], dbk[q]) for q in ("g", "v0", "eid", "ws", "clip_index")))
        k5[k] = {"tick0_plan_raw_vs_Ybanked_plan_max_abs_m": d_raw_bank, "tick0_plan_raw_vs_Yprime_plan_max_abs_m": d_raw_yp, "plan_vs_P_plus_SG_max_abs_m": d_rec, "S_changed_plan_max_abs_m": s_active,
                 "floor_fl_vs_Ybanked_max_abs_dev_m": fl_bank, "floor_fl_vs_Yprime_max_abs_dev_m": fl_yp, "windows_equal_Ybanked": win_eq, "smooth_field": str(t.get("smooth")),
                 "plan_clause_PASS": bool(d_rec <= R0.TOL_DUMP and s_active > 1e-3 and win_eq), "tick0_clause_PASS": bool(d_raw_bank <= tol["tol_tick0"]), "floor_clause_PASS": bool(fl_bank <= tol["tol_floor"])}
    ctl["K5p_smoothed_ticks_identities"] = {"runs": k5, "tol_tick0": tol["tol_tick0"], "tol_floor": tol["tol_floor"], "plan_recompute_tol_m": R0.TOL_DUMP,
                                            "plan_clause_PASS": all(v["plan_clause_PASS"] for v in k5.values()), "tick0_clause_PASS": all(v["tick0_clause_PASS"] for v in k5.values()),
                                            "floor_clause_PASS": all(v["floor_clause_PASS"] for v in k5.values())}
    ctl["K5p_smoothed_ticks_identities"]["PASS"] = bool(ctl["K5p_smoothed_ticks_identities"]["plan_clause_PASS"] and ctl["K5p_smoothed_ticks_identities"]["tick0_clause_PASS"] and ctl["K5p_smoothed_ticks_identities"]["floor_clause_PASS"])

    # ---- K6: the tool is the patched tool of the card (identical sha256 in every done_*.txt and equal to code/v8f_t1_eval.py); the pytest logs are those of tests/ (no code changed)
    shas = {}
    for tag in [v[0] for v in X_RUNS.values()] + [v[0] for v in YP_RUNS.values()] + [YPP[0]]:
        shas[tag] = json.load(open(f"{a.out_dir}/done_{tag}.txt", encoding="utf-8")).get("tool_sha256")
    tool_sha = R0.sha256_file(TOOL_COPY)
    thr = {tag: json.load(open(f"{a.out_dir}/done_{tag}.txt", encoding="utf-8")).get("threads") for tag in shas}
    ctl["K6_tool_identity_and_threads"] = {"tool_copy_sha256": tool_sha, "all_runs_same_tool_sha256": bool(len(set(shas.values())) == 1 and tool_sha in shas.values()), "tool_sha256_by_run": shas, "threads_by_run": thr,
                                           "all_threads_16": bool(all(v == 16 for v in thr.values())), "pytest_logs": "tests/pytest_*.log (patched tool, unchanged)",
                                           "PASS": bool(len(set(shas.values())) == 1 and tool_sha in shas.values() and all(v == 16 for v in thr.values()))}
    if os.path.exists(R0.P10_PRED):
        p10 = np.load(R0.P10_PRED).astype(np.float64)
        t = dict(np.load(f"{a.out_dir}/{X_RUNS['CTRL_s0'][0]}/ticks/ticks_{X_RUNS['CTRL_s0'][0]}.npz", allow_pickle=True))
        k0 = t["tick"] == 0
        o = np.argsort(t["win"][k0], kind="stable")
        ctl["K7_tick0_P_vs_p10_eval_pred6_descriptive"] = {"max_abs_m": float(np.abs(t["P"][k0][o] - p10).max()) if t["P"][k0].shape == p10.shape else None}

    # ---- the status ladder of AMENDMENT_1 section 4
    hard = {"K2pp": ctl["K2pp_Yprime_minus_Ydoubleprime_rows"]["PASS"], "K3": ctl["K3_Y_minus_P10_reproduces_SOUP_READ"]["PASS"], "K4": ctl["K4_self_minus_self_is_zero"]["PASS"],
            "K5_plan_clause": ctl["K5p_smoothed_ticks_identities"]["plan_clause_PASS"], "K6": ctl["K6_tool_identity_and_threads"]["PASS"]}
    soft = {"K1p": ctl["K1p_Yprime_vs_banked_DAGOACC_s0"]["PASS"], "K2p": ctl["K2p_Yprime_minus_Ybanked_rows"]["PASS"], "K5_tick0_clause": ctl["K5p_smoothed_ticks_identities"]["tick0_clause_PASS"],
            "K5_floor_clause": ctl["K5p_smoothed_ticks_identities"]["floor_clause_PASS"]}
    res["controls_hard"], res["controls_soft_banked_reference"] = hard, soft
    if not all(hard.values()):
        res["outcome_status"] = "REFUSED: " + ", ".join(sorted(k for k, v in hard.items() if not v)) + " failed (a defect, not noise)"
    elif all(soft.values()):
        res["outcome_status"] = "QUOTABLE (K1', K2', K5', K3, K4 pass at the measured tolerances)"
    else:
        res["outcome_status"] = ("QUOTABLE ON Y' (banked-reference comparisons exceed the measured tolerance: " + ", ".join(sorted(k for k, v in soft.items() if not v))
                                 + "; the decision contrast is X - Y', X - P10 uses the banked rows; conditional on Y' being the configuration of the banked Y)")
    print("controls_hard:", json.dumps(hard))
    print("controls_soft:", json.dumps(soft))
    print("outcome_status:", res["outcome_status"])
    if res["outcome_status"].startswith("REFUSED"):
        res["wall_s"] = round(time.time() - t_start, 1)
        out = f"{a.raw}/SMOOTH_T1_A1.json"
        json.dump(res, open(out, "w", encoding="utf-8", newline="\n"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        print("READ REFUSED ->", out)
        return 1

    # ---- the rows: X - P10 (banked P10), X - Y' (decision), X - Y banked (beside), Y' - P10, Y - P10, Y' - Y
    import torch
    sys.path.insert(0, f"{R0.TREE}/taniteval")
    os.environ.setdefault("TANITEVAL_STACK_OVERRIDE", f"{R0.TREE}/stack".replace("/", os.sep))
    from taniteval import four_families as FF
    for arm in ("CTRL", "SOUP"):
        per_seed, per_seed_banked, rows_out = {}, {}, {}
        for s in (0, 1):
            xk = f"{arm}_s{s}"
            zx, zp, zy = Z[xk], ZP[xk], Y[(arm, s)]
            assert all(np.array_equal(zx[q], zy[q]) and np.array_equal(zp[q], zy[q]) for q in ("eid", "v0", "g", "ws", "clip_index"))
            assert all(np.array_equal(zx[f"b_{i}"], zy[f"b_{i}"], equal_nan=True) and np.array_equal(zp[f"b_{i}"], zy[f"b_{i}"], equal_nan=True) for i, l in enumerate(labels) if not l.startswith("S-POS"))
            cls, _masks = GD.classify(zy["v0"], zy["g"])
            tabs = {"XP": {}, "XYp": {}, "XY": {}, "YpP": {}, "YP": {}, "YpY": {}}
            for i, lab in enumerate(labels):
                pool = GD.POOL[lab]
                for nm, (u, v) in (("XP", (zx[f"a_{i}"], zx[f"b_{i}"])), ("XYp", (zx[f"a_{i}"], zp[f"a_{i}"])), ("XY", (zx[f"a_{i}"], zy[f"a_{i}"])),
                                   ("YpP", (zp[f"a_{i}"], zp[f"b_{i}"])), ("YP", (zy[f"a_{i}"], zy[f"b_{i}"])), ("YpY", (zp[f"a_{i}"], zy[f"a_{i}"]))):
                    r = GD.decompose(u, v, zy["eid"], cls, pool, a.n_boot)
                    tabs[nm][lab] = {"delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"], "n": f"{r['n_windows']}/{r['n_episodes_pool']}", "family": R0.FAMILY[lab],
                                     "direction": "higher" if lab in GD.HIGHER_BETTER else "lower", "classes": {c: {"n": r["classes"][c]["n"], "delta": r["classes"][c]["delta"], "contribution": r["classes"][c]["contribution"]} for c in GD.CLASSES}}
            A = R0.read_a(GD, tabs["XP"])
            per_seed[s] = {"A": A, "XY": R0.read_xy(GD, tabs["XYp"])}
            per_seed_banked[s] = {"A": A, "XY": R0.read_xy(GD, tabs["XY"])}
            rows_out[f"s{s}"] = tabs
            tag_x, tag_p = X_RUNS[xk][0], YP_RUNS[xk][0]
            eb = man["arms"][arm][str(s)]["t1"]
            _n, dx = R0.load_dump(f"{a.out_dir}/{tag_x}/dump_{tag_x}")
            tx = dict(np.load(f"{a.out_dir}/{tag_x}/ticks/ticks_{tag_x}.npz", allow_pickle=True))
            _n, dyp = R0.load_dump(f"{a.out_dir}/{tag_p}/dump_{tag_p}")
            typ = dict(np.load(f"{a.out_dir}/{tag_p}/ticks/ticks_{tag_p}.npz", allow_pickle=True))
            _n, dbk = R0.load_dump(eb["dump_dir"])
            tbk = dict(np.load(eb["ticks"], allow_pickle=True))
            res["descriptive"][xk] = {"X_smoothed": R0.curv_block(GD, dx, tx["plan"], tx["plan_raw"], tx["v0"], torch, FF), "Y_prime_off": R0.curv_block(GD, dyp, typ["plan"], None, typ["v0"], torch, FF),
                                      "Y_banked": R0.curv_block(GD, dbk, tbk["plan"], None, tbk["v0"], torch, FF)}
        res["arms"][arm] = {"name": R0.ARM_NAME[arm], "per_seed": {f"s{s}": per_seed[s] for s in (0, 1)}, "rows": rows_out, **R0.arm_outcome(per_seed),
                            "outcome_on_banked_Y_beside": R0.arm_outcome(per_seed_banked)}
        res["arms"][arm]["per_seed_banked_XY"] = {f"s{s}": per_seed_banked[s]["XY"] for s in (0, 1)}
        print(arm, res["arms"][arm]["outcome"], "| on banked Y:", res["arms"][arm]["outcome_on_banked_Y_beside"]["outcome"],
              {f"s{s}": {"D1": per_seed[s]["A"]["D1"], "D2": per_seed[s]["A"]["D2"], "D3": per_seed[s]["A"]["D3"], "MOVED": per_seed[s]["XY"]["MOVED_seed"], "HARMS": per_seed[s]["XY"]["HARMS_seed"]} for s in (0, 1)})
    res["wall_s"] = round(time.time() - t_start, 1)
    out = f"{a.raw}/SMOOTH_T1_A1.json"
    json.dump(res, open(out, "w", encoding="utf-8", newline="\n"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("READ OK ->", out, "sha256", R0.sha256_file(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
