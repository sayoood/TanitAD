"""run_mutations_tc.py -- the mutation runner of card PREREG_V8F_PROOF_ADD81 (tool checks). Each mutation is applied to a COPY of tokclose_fit.py inside tmp_mut/ (the real file is never touched);
test_tokclose.py is run against the copy (TOKCLOSE_FILE) and must go RED; the unmutated suite (control) must be GREEN before and after. Sequential, one pytest process at a time, CPU only.
The pytest base folders are new names inside tmp_pt/ (nothing is deleted).

  python run_mutations_tc.py [--only m1,m2] [--out MUTATIONS_TC.json] [--tag NAME] [--skip-control]
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
SRC = HERE / "tokclose_fit.py"

# name -> (what it breaks, exact text of ONE fragment, its replacement)
MUTATIONS = {
    "m01_window_threshold_off_by_one": ("an eligible frame needs one frame more than the H reads (f >= H+4)", "(f >= H + READER_HIST - 2)", "(f >= H + READER_HIST - 1)"),
    "m02_run_minimum_two": ("a label from a run of two frames is used", "(np.asarray(run, np.int64) >= RUN_MIN)", "(np.asarray(run, np.int64) >= 2)"),
    "m03_finite_label_not_required": ("frames without a finite closing speed are rows", " & np.isfinite(np.asarray(clo, np.float64))", ""),
    "m04_label_of_another_frame": ("the target is the label three frames before the row", "clo[f].astype(np.float32))", "clo[np.maximum(f - 3, 0)].astype(np.float32))"),
    "m05_label_sign_turned": ("the target is minus the label closing speed", "clo[f].astype(np.float32))", "(-clo[f]).astype(np.float32))"),
    "m06_permutation_dropped": ("the planted arm keeps its labels (permutation dropped)", "ys = ys[np.random.default_rng(seed).permutation(len(ys))]", "ys = ys"),
    "m07_window_includes_next_frame": ("the window of frame f holds the embeddings f-H+1..f+1", "off = torch.arange(-self.H + 1, 1, device=E.device) + self.shift", "off = torch.arange(-self.H + 1, 2, device=E.device) + self.shift"),
    "m08_window_one_frame_short": ("the window holds H-1 embeddings", "off = torch.arange(-self.H + 1, 1, device=E.device) + self.shift", "off = torch.arange(-self.H + 2, 1, device=E.device) + self.shift"),
    "m09_first_state_read": ("the head reads the first recurrent state instead of the last", "return self.fc2(F.relu(self.fc1(o[:, -1])))[:, 0]", "return self.fc2(F.relu(self.fc1(o[:, 0])))[:, 0]"),
    "m10_t3_noise_starts_late": ("control t3 replaces the frames from t+2 on, not from t+1", "xn[t + 1:] = ", "xn[t + 2:] = "),
    "m11_t3_tolerance_loose": ("control t3 accepts a difference of 0.1 m/s", "T3_TOL = 1e-5", "T3_TOL = 1e-1"),
    "m12_better_needs_any_seed": ("BETTER-THAN-OUTPUTS asks r above the cell in one seed only", "d[2] < 0 and all(rx > rc", "d[2] < 0 and any(rx > rc"),
    "m13_better_without_seed_condition": ("BETTER-THAN-OUTPUTS does not ask r above the cell", ' and all(rx > rc for rx, rc in zip(s["r_per_seed"], c["r_per_seed"])))', ")"),
    "m14_readable_on_point_estimate": ("READABLE reads the point estimate, not the lower end of the interval", 's["r_ci"][0] > 0', 's["r_pooled"] > 0'),
    "m15_paired_difference_turned": ("the paired difference is cell minus X", 'st["sae"], st["say"] = sx["sae"], sc["sae"]', 'st["sae"], st["say"] = sc["sae"], sx["sae"]'),
    "m16_verdict_needs_all_four": ("the verdict asks all four arms BETTER", "n >= 1", "n >= 4"),
    "m17_t2_loss_comparison_turned": ("control t2 asks the stopping loss above the loss before", 'L["loss_at_stop"] < L["loss_before"]', 'L["loss_at_stop"] > L["loss_before"]'),
    "m18_t2_overlap_not_checked": ("control t2 does not check the overlap with the evaluation clips", 'L["overlap_fit_eval"] == 0 and L["overlap_stop_eval"] == 0', "True"),
    "m19_t2_hp_not_checked": ("control t2 accepts any hyperparameters", 'L["hp"] == HP[L["mode"]]', "True"),
    "m20_t1_band_dropped": ("control t1 passes whatever r the planted arm reads", "np.isfinite(r) and lo <= r <= hi", "np.isfinite(r)"),
    "m21_t0_r_tolerance_dropped": ("control t0 does not compare the reproduced r", "np.isfinite(r) and abs(r - want) <= T0_TOL_R and n_ok", "np.isfinite(r) and n_ok"),
    "m22_t0_frame_sets_not_compared": ("control t0 accepts any frame set", "ok = tool_sets[H] == dump_set", "ok = True"),
    "m23_t0_label_tolerance_dropped": ("control t0 does not compare the label values", "bool(same_mask and worst <= T0_TOL_LABEL)", "bool(same_mask)"),
    "m24_eval_overlap_guard_removed": ("the fit does not refuse a fit clip that is an evaluation clip", "if set(held) & set(eval_ids) or set(train) & set(eval_ids):", "if False:"),
    "m25_shuffle_flag_ignored": ("the planted arm is fitted on the true labels", 'rows_fit = permute_labels(rows_fit, fit_ids, HP["shuffle_seed"])', "rows_fit = rows_fit"),
    "m26_best_state_not_kept": ("the final state is kept, not the state of the best evaluation", "model.load_state_dict(best_state)", "pass"),
    "m27_patience_ignored": ("training never stops on patience", 'if bad >= hp["patience"]:', "if False:"),
    "m28_loss_before_after_a_pass": ("the loss before the first step is not measured before any step", "loss_before = eval_loss(model, st, load, rows, device, hp)", "loss_before = 1e9"),
    "m29_e_split_by_row": ("the one-fifth split of the E arms is by row", 'sp = CF.split_by_clip(held, HP["stop_frac"], HP["split_seed"])', "sp = np.random.default_rng(0).random(len(held)) < 0.2"),
    "m30_t_arm_fit_and_stop_swapped": ("the T arm fits on the held-out clips and stops on the training clips", "fit_ids, stop_ids = train, held", "fit_ids, stop_ids = held, train"),
    "m31_table_key_off_by_one": ("the estimate of frame f is read from frame f+1 of the table", "key = np.asarray(ci, np.int64) * 1_000_000 + np.asarray(fr, np.int64)", "key = np.asarray(ci, np.int64) * 1_000_000 + np.asarray(fr, np.int64) + 1"),
    "m32_cells_swapped": ("the cell of H 10 is the ridge, of H 25 the network", 'CELL_OF_H = {10: ("N10", "N_H10"), 25: ("L25", "L_H25")}', 'CELL_OF_H = {10: ("L25", "L_H25"), 25: ("N10", "N_H10")}'),
    "m33_output_files_overwritten": ("the fit overwrites existing output files", "if any(os.path.exists(p) for p in paths.values()):", "if False:"),
    "m34_score_folder_not_refused": ("the score does not refuse an existing output folder", "if os.path.exists(a.out_dir):", "if False:"),
    "m35_failed_control_still_writes": ("a failed control does not stop the names", "    if not allp:\n", "    if False:\n"),
    "m36_episode_map_ignored": ("every clip is its own episode whatever the map says", "rows_list = [CF.build_rows(dumps[k], H, ep_map) for k in ks]", "rows_list = [CF.build_rows(dumps[k], H, None) for k in ks]"),
    "m37_better_reads_lower_interval_end": ("BETTER-THAN-OUTPUTS reads the lower end of the paired interval", "better = bool(d[2] < 0 and", "better = bool(d[1] < 0 and"),
    "m38_training_order_from_stopping_clips": ("the training order is built from the stopping clips", "order = [tr[i] for i in gen.permutation(len(tr))]", "order = [st[i % len(st)] for i in range(len(tr))]"),
}


def apply(name, text):
    _, old, new = MUTATIONS[name]
    n = text.count(old)
    if n != 1:
        raise SystemExit("mutation %s: the fragment occurs %d times, expected exactly 1" % (name, n))
    return text.replace(old, new)


def run_suite(tag, fit_file, timeout=5400):
    env = dict(os.environ)
    env.update({"TOKCLOSE_FILE": str(fit_file), "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "-1", "PYCAP_MB": env.get("PYCAP_MB", "3000"),
                "PYTHONPATH": os.pathsep.join([str(HERE), str(HERE.parent / "closeread"), str(HERE.parent / "integ10k")]), "TOKCLOSE_BUILD_DIR": str(HERE.parent / "integ10k")})
    PT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", str(HERE / "test_tokclose.py"), "--basetemp", str(PT / tag)], env=env,
                       capture_output=True, text=True, encoding="utf-8", timeout=timeout, cwd=str(HERE))
    out = r.stdout + r.stderr
    failed = re.findall(r"^FAILED (\S+)", out, re.M) + re.findall(r"^ERROR (\S+::\S+)", out, re.M)
    summ = re.findall(r"(\d+) (passed|failed|error)", out)
    return {"rc": r.returncode, "failed_tests": failed, "summary": " ".join("%s %s" % x for x in summ), "seconds": round(time.time() - t0, 1), "tail": out.strip().splitlines()[-3:]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(HERE / "MUTATIONS_TC.json"))
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--skip-control", action="store_true")
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit("REFUSED: %s exists" % a.out)
    names = [n for n in MUTATIONS if not a.only or n in a.only.split(",") or n.split("_")[0] in a.only.split(",")]
    MUT.mkdir(parents=True, exist_ok=True)
    text = SRC.read_text(encoding="utf-8")
    res = {"card": "PREREG_V8F_PROOF_ADD81", "source_sha256_16": __import__("hashlib").sha256(SRC.read_bytes()).hexdigest()[:16], "control_before": None, "control_after": None, "mutations": {}}
    if not a.skip_control:
        res["control_before"] = run_suite("%s_ctl_before" % a.tag, SRC)
        print("control before:", res["control_before"]["summary"], flush=True)
        if res["control_before"]["rc"] != 0:
            json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
            print("the unmutated suite is not green: nothing mutated")
            return 1
    for n in names:
        f = MUT / ("%s_%s.py" % (a.tag, n))
        f.write_text(apply(n, text), encoding="utf-8", newline="\n")
        r = run_suite("%s_%s" % (a.tag, n), f)
        r["what"] = MUTATIONS[n][0]
        r["red"] = bool(r["rc"] != 0)
        res["mutations"][n] = r
        print("%s: %s (%s) %s" % (n, "RED" if r["red"] else "GREEN <-- not caught", r["summary"], (r["failed_tests"] or [""])[0]), flush=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    if not a.skip_control:
        res["control_after"] = run_suite("%s_ctl_after" % a.tag, SRC)
        print("control after:", res["control_after"]["summary"], flush=True)
    red = sum(1 for v in res["mutations"].values() if v["red"])
    res["red"], res["total"] = red, len(res["mutations"])
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    print("ZZ MUTATIONS-TC red %d of %d" % (red, len(res["mutations"])))
    ok = red == len(res["mutations"]) and (a.skip_control or (res["control_before"]["rc"] == 0 and res["control_after"]["rc"] == 0))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
