"""run_mutations_tt.py -- the mutation runner of make_tab_tables.py (card ADD84). Each mutation changes ONE fragment of a control or formula line in a COPY of the tool inside tmp_mut/ (the real file is never
touched); test_tab_tables.py is run against the copy (TABTT_TOOL) and must go RED; the unmutated suite (control) must be GREEN before and after. Sequential, one pytest process at a time, CPU only.
Pytest base folders are new names inside tmp_pt/ (nothing is deleted or moved).
  PYCAP_MB=3000 bash ../ops/pycap.sh run_mutations_tt.py [--only m01,m02] [--out MUTATIONS_TT.json] [--check]
--check only verifies that every fragment occurs exactly once. Writes MUTATIONS_TT.json (red/total, each result, the two control runs). Exit 0 = every mutation red and both control runs green."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MUT = HERE / "tmp_mut"
PT = HERE / "tmp_pt"
TOOL = HERE / "make_tab_tables.py"

# name -> (what it breaks, exact text of ONE fragment of the tool, its replacement)
MUTATIONS = {
    "m01_run_min_two": ("a trailing run of two frames counts as a closing speed", "\nRUN_MIN = 3 ", "\nRUN_MIN = 2 "),
    "m02_run_condition_dropped": ("valued frames ignore the run length", '& (np.asarray(D["lab_run"], np.int64) >= RUN_MIN)', "& (np.asarray(D[\"lab_run\"], np.int64) >= 0)"),
    "m03_run_strictly_above": ("a run of exactly three frames carries no value", '>= RUN_MIN) & np.isfinite', '> RUN_MIN) & np.isfinite'),
    "m04_l100_value_zero": ("L100 carries 0 where valued", 'est[v] = np.asarray(D["lab_closing"], np.float64)[v]', 'est[v] = 0.0'),
    "m05_ar_coef": ("autoregression coefficient 0.8", "AR_COEF = 0.9", "AR_COEF = 0.8"),
    "m06_ar_innovation": ("innovation weight 1 - coef instead of sqrt(1 - coef^2)", "w = np.sqrt(1.0 - coef * coef)", "w = 1.0 - coef"),
    "m07_ar_no_restart": ("the autoregression is not restarted at a clip", "e[i] = zo[i] if (i == 0 or ci[i] != ci[i - 1]) else", "e[i] = zo[i] if (i == 0) else"),
    "m08_ar_row_order": ("the autoregression runs in row order, not frame order", "order = np.lexsort((np.asarray(frame, np.int64), np.asarray(clip_index, np.int64)))", "order = np.arange(len(z))"),
    "m09_no_projection": ("the noise is not made uncorrelated with L", "    if uncorrelated:\n", "    if False:\n"),
    "m10_no_scaling": ("the noise is not scaled to unit sd", "    return e / sd\n", "    return e\n"),
    "m11_w00_mean_not_removed": ("W00 noise keeps its drawn mean", "    else:\n        e = e - e.mean()\n", "    else:\n        pass\n"),
    "m12_ddof_one": ("the noise sd uses ddof 1", "    sd = e.std()\n", "    sd = e.std(ddof=1)\n"),
    "m13_rng_seed_without_k": ("the rng seed ignores the seed index", "default_rng(84000 + 10 * a + k)", "default_rng(84000 + 10 * a)"),
    "m14_arm_index_swapped": ("W50 and W70 swap their rng index", '"W50": 1, "W70": 2', '"W50": 2, "W70": 1'),
    "m15_r_instead_of_r2": ("the signal part uses r, not r^2", "m + r * r * (L - m) + s *", "m + r * (L - m) + s *"),
    "m16_noise_weight": ("the noise weight is sqrt(r (1 - r))", "np.sqrt(r * r * (1.0 - r * r))", "np.sqrt(r * (1.0 - r))"),
    "m17_w00_spread": ("W00 spread s instead of 0.5 s", "out[V] = m + W00_SD * s * e", "out[V] = m + s * e"),
    "m18_r_target_w90": ("W90 plants r 0.85", '"W90": 0.90', '"W90": 0.85'),
    "m19_std_ddof_one": ("m and s of the formula use ddof 1", "    m, s = float(L.mean()), float(L.std())\n    if arm", "    m, s = float(L.mean()), float(L.std(ddof=1))\n    if arm"),
    "m20_draw_reversed": ("the draw is used on the frames in reverse", "e = plant_noise(z[V], L,", "e = plant_noise(z[::-1][V], L,"),
    "m21_s70_without_ar": ("S70 skips the autoregression", '    if arm == "S70":\n        z = ar_along_clips', '    if False:\n        z = ar_along_clips'),
    "m22_a0_tolerance": ("a0 tolerance 1e-3", "A0_TOL = 1e-6", "A0_TOL = 1e-3"),
    "m23_a0_share": ("a0 share 0.99", "A0_SHARE = 0.999", "A0_SHARE = 0.99"),
    "m24_a0_wide_compare": ("a0 compares to within 1.0", "abs(tv - c) <= tol)\n                if not agree[i]:", "abs(tv - c) <= 1.0)\n                if not agree[i]:"),
    "m25_a0_zero_always_agrees": ("a0 accepts any table value where the log is 0", "agree[i] = bool((not np.isfinite(tv)) or tv == 0.0)", "agree[i] = True"),
    "m26_a0_nonzero_not_required": ("a0 passes a log without nonzero values", "good = bool(n > 0 and n_nz > 0 and share_all >= share_min)", "good = bool(n > 0 and share_all >= share_min)"),
    "m27_a0_present_only_judged": ("a0 judges the present looks only", "share_all = float(agree.mean()) if n else float(\"nan\")", "share_all = float(agree[pr].mean()) if pr.any() else float(\"nan\")"),
    "m28_a0_key_stride": ("a0 keys the table without the stride", "look = {int(a) * STRIDE + int(b): float(c)", "look = {int(a) + int(b): float(c)"),
    "m29_a0_frame_offset": ("a0 reads the table one frame late", "int(lci[i]) * STRIDE + int(lj[i]), float", "int(lci[i]) * STRIDE + int(lj[i]) + 1, float"),
    "m30_a1_r_tol": ("a1 r tolerance 0.05", "A1_R_TOL = 0.02", "A1_R_TOL = 0.05"),
    "m31_a1_r_tol_w00": ("a1 W00 r band 0.06", "A1_R_TOL_W00 = 0.03", "A1_R_TOL_W00 = 0.06"),
    "m32_a1_sd_band": ("a1 sd band 0.90 to 1.10", "A1_SD_BAND = (0.95, 1.05)", "A1_SD_BAND = (0.90, 1.10)"),
    "m33_a1_mean_tol": ("a1 mean tolerance 0.5", "A1_MEAN_TOL = 0.05", "A1_MEAN_TOL = 0.5"),
    "m34_a1_ac_s70": ("a1 S70 band 0.5 to 0.99", "A1_AC_S70 = (0.85, 0.95)", "A1_AC_S70 = (0.5, 0.99)"),
    "m35_a1_ac_w70": ("a1 W70 band -0.5 to 0.5", "A1_AC_W70 = (-0.05, 0.05)", "A1_AC_W70 = (-0.5, 0.5)"),
    "m36_a1_valued_set": ("a1 does not compare the valued frames with L100", "set_same = bool(rows_same and np.array_equal(np.isfinite(v), V))", "set_same = bool(rows_same)"),
    "m37_a1_sd_target": ("a1 sd target is s", "                sd_t = r_t * s\n", "                sd_t = s\n"),
    "m38_a1_noise_formula": ("a1 recovers the noise without removing the signal part", "noise = (v - m - r_t * r_t * (Lv - m)) /", "noise = (v - m) /"),
    "m39_a1_aggregate_or": ("a1 passes when any arm passes", "            ok = ok and good\n    return {\"PASS\": bool(ok), \"runs\": out}\n\n\n# ===", "            ok = ok or good\n    return {\"PASS\": bool(ok), \"runs\": out}\n\n\n# ==="),
    "m40_u4_h": ("a4 u4 uses H = 10", "U4_H = 25", "U4_H = 10"),
    "m41_stk_from_l100": ("STK is the map of the L100 table", 'stk = SU.surrogate_values(TK[k]["clip_index"], TK[k]["frame"], tk_est, ids, pi)', 'stk = SU.surrogate_values(TK[k]["clip_index"], TK[k]["frame"], l100, ids, pi)'),
    "m42_map_reversed_share": ("the map is built from the reversed share order", "pi = SU.compute_map(ids, share)", "pi = SU.compute_map(ids, -share)"),
    "m43_tk_perturbed": ("TK differs from the input table", 'arms["TK"] = tk_est.copy()', 'arms["TK"] = tk_est + 1e-9'),
    "m44_a0_table_is_tk": ("a0 reads the TK table instead of L100", 'a0 = check_a0({k: back[("L100", k)] for k in seeds}, logs)', 'a0 = check_a0({k: back[("TK", k)] for k in seeds}, logs)'),
    "m45_pass_ignores_a0": ("a0 does not enter the verdict", '(("a0", a0["PASS"]),', '(("a0", True),'),
    "m46_pass_ignores_a1": ("a1 does not enter the verdict", '("a1", a1["PASS"])', '("a1", True)'),
    "m47_pass_ignores_u1": ("a4_u1 does not enter the verdict", '("a4_u1", u1_ok)', '("a4_u1", True)'),
    "m48_pass_ignores_u4": ("a4_u4 does not enter the verdict", '("a4_u4", ok4)', '("a4_u4", True)'),
    "m49_label_columns_unchecked": ("differing label columns between the seeds are accepted", "            if not np.array_equal(D[seeds[0]][col], D[k][col]", "            if False and not np.array_equal(D[seeds[0]][col], D[k][col]"),
    "m50_tk_rows_unchecked": ("TK rows that differ from the dump rows are accepted", "        if not (np.array_equal(TK[k][\"clip_index\"].astype(np.int64)", "        if False and not (np.array_equal(TK[k][\"clip_index\"].astype(np.int64)"),
    "m51_existing_out_accepted": ("an existing out folder is accepted", '    if os.path.exists(out_dir):\n        raise SystemExit("REFUSED: %s exists" % out_dir)\n    SU', '    if False:\n        raise SystemExit("REFUSED: %s exists" % out_dir)\n    SU'),
    "m52_run_min_guard_dropped": ("a RUN_MIN that differs from closeread_fit is accepted", "    if int(CF.RUN_MIN) != RUN_MIN:", "    if False:"),
    "m53_planting_regressor_dropped": ("the projection regresses on the constant only", "Xm = np.stack([np.ones(len(L)), L], 1)", "Xm = np.stack([np.ones(len(L)), np.zeros(len(L))], 1)"),
    "m54_mid_log_name": ("the mid-look log is the replan log", 'for fn in ("rule_log_%s.npz" % key, "rule_log_mid_%s.npz" % key):', 'for fn in ("rule_log_%s.npz" % key, "rule_log_%s.npz" % key):'),
    "m55_u1_map_from_clips": ("u1 reads the map clips as the map", '"map": mapf["map"], "map_clips"', '"map": mapf["map_clips"], "map_clips"'),
    "m56_stk_not_from_dump_share": ("the share of the map is taken from the first seed's dump for every seed", "ids, share, counts, totals = SU.share_of(D[k])\n        pi = SU.compute_map", "ids, share, counts, totals = SU.share_of(D[seeds[0]])\n        pi = SU.compute_map"),
    "m57_rng_record_seed": ("the record's rng seeds omit the seed index", '84000 + 10 * ARM_INDEX[arm] + k', '84000 + 10 * ARM_INDEX[arm]'),
}


def apply(name, text):
    _, old, new = MUTATIONS[name]
    n = text.count(old)
    if n != 1:
        raise SystemExit("mutation %s: the fragment occurs %d times, expected exactly 1" % (name, n))
    return text.replace(old, new)


def run_suite(tag, tool, timeout=1800):
    env = dict(os.environ)
    env.update({"TABTT_TOOL": str(tool), "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "-1"})
    PT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", str(HERE / "test_tab_tables.py"), "--basetemp", str(PT / tag)], env=env, capture_output=True,
                       text=True, encoding="utf-8", timeout=timeout, cwd=str(HERE))
    out = r.stdout + r.stderr
    failed = re.findall(r"^FAILED (\S+)", out, re.M) + re.findall(r"^ERROR (\S+::\S+)", out, re.M)
    summ = re.findall(r"(\d+) (passed|failed|error)", out)
    return {"rc": r.returncode, "failed_tests": failed, "summary": " ".join("%s %s" % x for x in summ), "seconds": round(time.time() - t0, 1), "tail": out.strip().splitlines()[-3:]}


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(HERE / "MUTATIONS_TT.json"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    stamp = time.strftime("%H%M%S")
    names = [n for n in MUTATIONS if not a.only or n.split("_")[0] in a.only.split(",") or n in a.only.split(",")]
    base = TOOL.read_text(encoding="utf-8")
    if a.check:
        for n in names:
            apply(n, base)
        print("all %d fragments occur exactly once" % len(names))
        return 0
    MUT.mkdir(parents=True, exist_ok=True)
    res = {"tool": str(TOOL), "tool_sha256_16": hashlib.sha256(base.encode("utf-8")).hexdigest()[:16], "mutations": {}}
    res["control_before"] = run_suite("ctl_before_" + stamp, TOOL)
    print("control before:", res["control_before"]["summary"], flush=True)
    for n in names:
        p = MUT / ("tool_%s_%s.py" % (n, stamp))
        p.write_text(apply(n, base), encoding="utf-8", newline="\n")
        r = run_suite("%s_%s" % (n.split("_")[0], stamp), p)
        r["red"] = bool(r["rc"] != 0 and r["failed_tests"])
        r["what"] = MUTATIONS[n][0]
        res["mutations"][n] = r
        print("%s %s %s" % ("RED  " if r["red"] else "GREEN", n, (r["failed_tests"] or [""])[0][:90]), flush=True)
    res["control_after"] = run_suite("ctl_after_" + stamp, TOOL)
    print("control after:", res["control_after"]["summary"], flush=True)
    red = sum(1 for v in res["mutations"].values() if v["red"])
    res["red"], res["total"] = red, len(res["mutations"])
    res["controls_green"] = bool(res["control_before"]["rc"] == 0 and res["control_after"]["rc"] == 0)
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True), encoding="utf-8", newline="\n")
    print("ZZ MUTATIONS %d/%d red, controls green %s" % (red, len(res["mutations"]), res["controls_green"]))
    return 0 if (red == len(res["mutations"]) and res["controls_green"]) else 1


if __name__ == "__main__":
    sys.exit(main())
