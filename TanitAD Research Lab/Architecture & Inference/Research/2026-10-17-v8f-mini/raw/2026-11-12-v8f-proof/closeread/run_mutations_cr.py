"""run_mutations_cr.py -- the mutation runner of card PREREG_V8F_PROOF_ADD77 (tool checks). Each mutation is applied to a COPY of closeread_fit.py or closeread_dump.py inside tmp_mut/ (the real
files are never touched); test_closeread.py is run against the copy (CLOSEREAD_FIT_FILE / CLOSEREAD_DUMP_FILE) and must go RED; the unmutated suite (control) must be GREEN before and after.
Sequential, one pytest process at a time, CPU only. The pytest base folders are new names inside tmp_pt/ (nothing is deleted).

  python run_mutations_cr.py [--only m1,m2] [--out MUTATIONS_CR.json] [--tag NAME]
Exit 0 = every mutation red and both control runs green.
"""
from __future__ import annotations

import argparse
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

# name -> (file, what it breaks, exact text of ONE fragment, its replacement)
MUTATIONS = {
    "m1_heldout_rule_dropped": ("fit", "the held-out rule accepts every clip (the card's listed mutation: the rule dropped)", "% 10 == 0  # MUT-HELDOUT", "% 10 >= 0  # MUT-HELDOUT"),
    "m2_evaluation_clips_let_into_the_fit": ("fit", "the score rows are appended to the fit rows after the guard (the card's listed mutation)", "fit_rows = fit_rows  # MUT-FITROWS",
                                             'fit_rows = {"X": np.concatenate([fit_rows["X"], score_rows["X"]]), "y": np.concatenate([fit_rows["y"], score_rows["y"]]), "clip": np.concatenate([fit_rows["clip"].astype(object), score_rows["clip"].astype(object)])}  # MUT-FITROWS'),
    "m3_ridge_weight_chosen_on_the_score_set": ("fit", "the ridge weight is chosen on the score rows (the card's listed mutation)", "Xtr, ytr, Xva, yva = Xf[~va], yf[~va], Xf[va], yf[va]  # MUT-RIDGE-VAL",
                                                'Xtr, ytr, Xva, yva = Xf[~va], yf[~va], Xs, score_rows["y"]  # MUT-RIDGE-VAL'),
    "m4_label_of_another_frame_as_target": ("fit", "the target is the label three frames before the last frame (the card's listed mutation)", 'y = np.asarray(D["lab_closing"], np.float64)[sel]  # MUT-TARGET',
                                            'y = np.asarray(D["lab_closing"], np.float64)[np.maximum(sel - 3, 0)]  # MUT-TARGET'),
    "m5_sign_of_the_target_turned": ("fit", "the target is minus the label closing speed (the card's listed mutation)", 'y = np.asarray(D["lab_closing"], np.float64)[sel]  # MUT-TARGET',
                                     'y = -np.asarray(D["lab_closing"], np.float64)[sel]  # MUT-TARGET'),
    "m6_leak_guard_removed": ("fit", "the hard refusal of shared or non-held-out fit clips is replaced by nothing", "assert_no_leak(fit_rows[\"clip\"], score_rows[\"clip\"], train_ids)  # MUT-LEAKGUARD", "pass  # MUT-LEAKGUARD"),
    "m7_validation_split_by_row": ("fit", "the validation split of the fit set is by row, not by clip", "return np.asarray([str(c) in v for c in clips])", "return np.random.default_rng(seed).random(len(clips)) < frac"),
    "m8_per_seed_bar_dropped": ("fit", "the name rule does not ask r >= 0.30 in every reader seed", 'all(r >= BAR_R for r in s["r_per_seed"])', "True"),
    "m9_paired_difference_dropped": ("fit", "the name rule does not ask the paired absolute-error difference below 0", ' and s["diff_ci"][1] < 0)', ")"),
    "m10_window_validity_dropped": ("fit", "windows with frames that carry no read are used", "(cs[idx + 1] - cs[idx - H + 1] == 0) & ", ""),
    "m11_c3_gap_not_compared": ("fit", "control c3 compares presence only", "dp.max() <= tol and dg.max() <= tol)", "dp.max() <= tol)"),
    "m12_c3_map_not_applied": ("fit", "control c3 compares the gap read without the map of the run", 'qmap_apply(np.asarray(score_dump["gap"], np.float64)[rr], *qmap)', 'np.asarray(score_dump["gap"], np.float64)[rr]'),
    "m13_bootstrap_one_level": ("fit", "the bootstrap resamples episodes only, the reader seeds are fixed", "sd = rng.integers(0, S, (B, S))", "sd = np.tile(np.arange(S), (B, 1))"),
    "m14_permutation_dropped": ("fit", "the negative control fits on the unpermuted labels", "yf = np.random.default_rng(perm_seed).permutation(yf)", "yf = yf"),
    "m15_episode_map_ignored": ("fit", "every clip is its own episode whatever the map says", "e = np.asarray([str(ep_map.get(c, c)) if ep_map else c for c in cl])", "e = np.asarray([c for c in cl])"),
    "m16_gap_two_point_wrong_span": ("fit", "the two-point change divides by H frames instead of H - 1 intervals", "(X[:, -1, 0] - X[:, 0, 0]) / ((H - 1) * dt)", "(X[:, -1, 0] - X[:, 0, 0]) / (H * dt)"),
    "m17_c2_sign_turned": ("fit", "control c2 turns the sign of the label two-point change", "ch = two_point_label(D, 4, dt)", "ch = -two_point_label(D, 4, dt)"),
    "m18_c7_thresholds_dropped": ("fit", "control c7 passes whatever the counts", "nf >= FIT_MIN_FRAMES and ns >= SCORE_MIN_FRAMES and ne >= SCORE_MIN_EPISODES", "True"),
    "d1_label_run_min_two": ("dump", "a label closing speed is read from a run of two frames", "GAP_MAX_M, STEP_MAX_M, RUN_MAX, RUN_MIN = 60.0, 3.0, 5, 3", "GAP_MAX_M, STEP_MAX_M, RUN_MAX, RUN_MIN = 60.0, 3.0, 5, 2"),
    "d2_window_valid_one_frame_early": ("dump", "the window is marked valid one frame before the fifth frame exists", "valid[self.hist - 1:] = True", "valid[self.hist - 2:] = True"),
    "d3_closing_scale_one": ("dump", "the closing output is scaled by 1 instead of 10 m/s", "CLOS_SCALE_MS, GAP_SCALE_M = 10.0, 50.0", "CLOS_SCALE_MS, GAP_SCALE_M = 1.0, 50.0"),
    "d4_gap_scale_five": ("dump", "the gap read is scaled by 5 instead of 50 m", "CLOS_SCALE_MS, GAP_SCALE_M = 10.0, 50.0", "CLOS_SCALE_MS, GAP_SCALE_M = 10.0, 5.0"),
    "d5_heldout_rule_dropped": ("dump", "the dump tool's held-out rule accepts every clip", ".hexdigest()[:8], 16) % 10 == 0", ".hexdigest()[:8], 16) % 10 >= 0"),
    "d6_exclusion_ignored": ("dump", "the trainer's --exclude slice is not applied to the manifest", "excluded = set(sorted(manifest_ids, key=sha12)[ex[0]:ex[1]]) if ex else set()", "excluded = set()"),
    "d7_label_closing_sign": ("dump", "the label closing speed has the opposite sign", "clo[j] = float(", "clo[j] = -float("),
    "d8_label_step_bound_dropped": ("dump", "the label run ignores the 3 m step bound", " and abs(gap[f + 1] - gap[f]) <= STEP_MAX_M:", ":"),
}
INTENDED = {"m10_window_validity_dropped": "build_rows_window", "m13_bootstrap_one_level": "bootstrap_resamples", "m7_validation_split_by_row": "split_by_clip"}


def apply(name, text):
    _, _, old, new = MUTATIONS[name]
    n = text.count(old)
    if n != 1:
        raise SystemExit("mutation %s: the fragment occurs %d times, expected exactly 1" % (name, n))
    return text.replace(old, new)


def run_suite(tag, fit_file, dump_file, timeout=3000):
    env = dict(os.environ)
    env.update({"CLOSEREAD_FIT_FILE": str(fit_file), "CLOSEREAD_DUMP_FILE": str(dump_file), "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "-1"})
    PT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", str(HERE / "test_closeread.py"), "--basetemp", str(PT / tag)], env=env, capture_output=True,
                       text=True, encoding="utf-8", timeout=timeout, cwd=str(HERE))
    out = r.stdout + r.stderr
    failed = re.findall(r"^FAILED (\S+)", out, re.M) + re.findall(r"^ERROR (\S+::\S+)", out, re.M)
    summ = re.findall(r"(\d+) (passed|failed|error)", out)
    return {"rc": r.returncode, "failed_tests": failed, "summary": " ".join("%s %s" % x for x in summ), "seconds": round(time.time() - t0, 1), "tail": out.strip().splitlines()[-3:]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(HERE / "MUTATIONS_CR.json"))
    ap.add_argument("--tag", default="r%d" % int(time.time()))
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        print("REFUSED: output exists: " + a.out)
        return 2
    names = [n for n in MUTATIONS if not a.only or n in a.only.split(",")]
    MUT.mkdir(parents=True, exist_ok=True)
    fit_src = (HERE / "closeread_fit.py").read_text(encoding="utf-8")
    dump_src = (HERE / "closeread_dump.py").read_text(encoding="utf-8")
    res = {"tool_files": {"closeread_fit.py": str(HERE / "closeread_fit.py"), "closeread_dump.py": str(HERE / "closeread_dump.py")}, "control_before": None, "mutations": {}, "control_after": None}
    res["control_before"] = run_suite(a.tag + "_ctl_before", HERE / "closeread_fit.py", HERE / "closeread_dump.py")
    print("control before: rc %d %s" % (res["control_before"]["rc"], res["control_before"]["summary"]), flush=True)
    for n in names:
        f, what = MUTATIONS[n][0], MUTATIONS[n][1]
        ffile, dfile = HERE / "closeread_fit.py", HERE / "closeread_dump.py"
        if f == "fit":
            ffile = MUT / (a.tag + "_" + n + "_closeread_fit.py")
            ffile.write_text(apply(n, fit_src), encoding="utf-8", newline="\n")
        else:
            dfile = MUT / (a.tag + "_" + n + "_closeread_dump.py")
            dfile.write_text(apply(n, dump_src), encoding="utf-8", newline="\n")
        r = run_suite(a.tag + "_" + n, ffile, dfile)
        r["what"], r["file"], r["red"] = what, f, bool(r["rc"] != 0 and r["failed_tests"] and not any("collect" in t for t in r["tail"]))
        res["mutations"][n] = r
        print("%s: %s %s" % (n, "RED" if r["red"] else "NOT RED", r["failed_tests"][:2]), flush=True)
    res["control_after"] = run_suite(a.tag + "_ctl_after", HERE / "closeread_fit.py", HERE / "closeread_dump.py")
    print("control after: rc %d %s" % (res["control_after"]["rc"], res["control_after"]["summary"]), flush=True)
    n_red = sum(1 for v in res["mutations"].values() if v["red"])
    res["counts"] = {"red": n_red, "total": len(res["mutations"]), "control_before_green": res["control_before"]["rc"] == 0, "control_after_green": res["control_after"]["rc"] == 0}
    json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    ok = n_red == len(res["mutations"]) and res["counts"]["control_before_green"] and res["counts"]["control_after_green"]
    print("mutations red %d of %d; control before green %s, after green %s" % (n_red, len(res["mutations"]), res["counts"]["control_before_green"], res["counts"]["control_after_green"]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
