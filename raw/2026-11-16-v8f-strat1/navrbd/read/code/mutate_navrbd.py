"""mutate_navrbd.py -- the deliberate-regression arms of the NAVRBD read.  Each mutation is applied to a COPY of the study's files (a temp folder under mut_tmp, deleted afterwards; the frozen STRAT1 readers are
imported from their own folder and never mutated) and test_navrbd_read.py is run WHOLE against the copy: every mutation must turn its named test RED.
Usage: python mutate_navrbd.py -> out/MUTATIONS_NAVRBD.json, exit 1 if any mutation stays green or the expected test stays green."""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
FILES = [f for f in os.listdir(HERE) if f.startswith(("navrbd_", "test_navrbd")) and f.endswith(".py")]
BAR = "    return float(max(BAR_NAVRB2, f_ho, *others))"
MUTS = [
    ("the 5-way lat_cls accepted as the D1-HO label", "navrbd_read.py", "lab.max() <= 2", "lab.max() <= 4", "test_kinematic_label_accuracy_not_five_way"),
    ("the driver reads the 5-way lat_cls instead of read_otband.labels", "navrbd_read.py", 'gl = check_three_class(geom.labels(ref["g"])[0])', 'gl = np.asarray(ref["lat_cls"])', "test_driver_reads_the_kinematic_label"),
    ("the bar dropped to 0.0383", "navrbd_read.py", BAR, "    return float(BAR_NAVRB2)", "test_bar_is_the_larger_of_the_rig_floor_and_f_ho"),
    ("the bar dropped to 0.0383 (gain rule reads the rig floor)", "navrbd_read.py", '"bar": float(bar), "gain": bool(d >= bar and lo > 0)', '"bar": float(bar), "gain": bool(d >= 0.0383 and lo > 0)', "test_d1_ho_gain_rule_on_planted_windows"),
    ("N1D read against the wrong seed", "navrbd_read.py", 'return f"N1R_s{int(seed)}"', 'return f"N1R_s{1 - int(seed)}"', "test_pairing_is_same_seed"),
    ("gain without the positive lower bound", "navrbd_read.py", '"gain": bool(d >= bar and lo > 0)', '"gain": bool(d >= bar)', "test_d1_ho_gain_rule_on_planted_windows"),
    ("F_HO not absolute", "navrbd_read.py", "return float(abs(np.mean(np.asarray(acc_s1, np.float64)) - np.mean(np.asarray(acc_s0, np.float64))))", "return float(np.mean(np.asarray(acc_s1, np.float64)) - np.mean(np.asarray(acc_s0, np.float64)))", "test_f_ho_is_the_absolute_mean_difference_of_the_pair"),
    ("Z-EVAL exact-zero comparison dropped", "navrbd_read.py", "ok = ok and mx == 0.0 and nm == 0", "ok = ok", "test_z_eval_exact_zero_and_mask"),
    ("Z-EVAL mask ignored", "navrbd_read.py", "    m = np.asarray(mask, bool)\n    out, ok = {}, bool(m.any())", "    m = np.ones(len(mask), bool)\n    out, ok = {}, True", "test_z_eval_exact_zero_and_mask"),
    ("K-CONST compares against the wrong share", "navrbd_read.py", "return {\"majority_class\": CL[maj], \"majority_share\": share, \"constant_accuracy\": acc, \"pass\": bool(acc == share)}", "return {\"majority_class\": CL[maj], \"majority_share\": share, \"constant_accuracy\": acc, \"pass\": True}", "test_k_const_and_k_floor"),
    ("K-FLOOR vacuous", "navrbd_read.py", 'return {"live": float(f_live), "banked": float(f_banked), "pass": bool(float(f_live) == float(f_banked))}', 'return {"live": float(f_live), "banked": float(f_banked), "pass": True}', "test_k_const_and_k_floor"),
    ("K1 self-pair accepts a non-degenerate interval", "navrbd_read.py", 'float(r["lo"]) == 0.0 and float(r["hi"]) == 0.0', "True", "test_d1_ho_uses_the_bar_it_is_given_and_reads_a_self_pair_as_zero"),
    ("RBD-GAIN: N1Dx dropped", "navrbd_read.py", 'if gain_label and n1dx is not None and not n1dx["gain"]:', "if gain_label:", "test_outcome_gain_needs_every_condition"),
    ("RBD-GAIN: Z-USE dropped", "navrbd_read.py", "if all(gain) and zk and all(zs) and guard_known and not gc[0]", "if all(gain) and guard_known and not gc[0]", "test_outcome_gain_needs_every_condition"),
    ("RBD-GAIN: the guard at seed 1 dropped", "navrbd_read.py", "guard_known and not gc[0] and not gc[1]:", "guard_known and not gc[0]:", "test_outcome_gain_needs_every_condition"),
    ("N1Dx gain does not stand the labels down", "navrbd_read.py", 'if n1dx is not None and n1dx["gain"]:\n        labels = ["N1DX-REGRESSION-GAINS"]', "if False:\n        labels = [\"N1DX-REGRESSION-GAINS\"]", "test_outcome_gain_needs_every_condition"),
    ("RBD-ONE-SEED: reads at least one seed", "navrbd_read.py", "if sum(gain) == 1:", "if sum(gain) >= 1:", "test_outcome_gain_needs_every_condition"),
    ("RBD-NO-GAIN: Z-USE at one seed enough", "navrbd_read.py", "if zk and all(zs) and not any(gain):", "if zk and any(zs) and not any(gain):", "test_outcome_no_gain_unread_costs_floor_high"),
    ("RBD-UNREAD on one unread seed", "navrbd_read.py", "if zk and not any(zs):", "if zk and not all(zs):", "test_outcome_no_gain_unread_costs_floor_high"),
    ("RBD-COSTS: any guard cell, not the same cell at both seeds", "navrbd_read.py", "if all(cost) or (guard_known and both_cost):", "if all(cost) or (guard_known and (gc[0] | gc[1])):", "test_outcome_no_gain_unread_costs_floor_high"),
    ("RBD-FLOOR-HIGH without the rig-floor condition", "navrbd_read.py", "if f_ho > BAR_NAVRB2 and all(float(f_ho) > d for d in delta):", "if all(float(f_ho) > d for d in delta):", "test_outcome_floor_high_needs_f_above_the_rig_floor"),
    ("a missing input reads as a pass (no NOT-YET)", "navrbd_read.py", 'if any(d1.get(s) is None for s in (0, 1)):\n        return {"status": "NOT-YET"', 'if False:\n        return {"status": "NOT-YET"', "test_outcome_missing_input_is_not_yet_never_a_pass"),
    ("a missing N1Dx not listed", "navrbd_read.py", '    if n1dx is None:\n        miss.append("N1Dx_s0")', "    if False:\n        miss.append(\"N1Dx_s0\")", "test_outcome_missing_input_is_not_yet_never_a_pass"),
    ("the zero twin dumped with the channel on", "navrbd_dumps.py", 'return "zero" if tag.endswith(NL.Z) else "follow"', 'return "follow"', "test_t1_argv_follow_and_zero_twin"),
    ("the feat dropped from the dump argv", "navrbd_dumps.py", '"--nav-rb-feat", "exitdist", ', "", "test_t1_argv_follow_and_zero_twin"),
    ("the RB table pin not checked", "navrbd_dumps.py", "    if got != RB_SHA256:", "    if False:", "test_dump_wrapper_uses_the_rbd_script_and_refuses_a_wrong_rb_table"),
]


def main():
    res, bad = [], 0
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    for name, f, old, new, expect in MUTS:
        tmp = os.path.join(HERE, "mut_tmp", "m")
        shutil.rmtree(os.path.join(HERE, "mut_tmp"), ignore_errors=True)
        os.makedirs(tmp)
        for x in FILES:
            shutil.copy(os.path.join(HERE, x), tmp)
        p = os.path.join(tmp, f)
        s = open(p, encoding="utf-8").read()
        assert s.count(old) >= 1, (name, "pattern not found")
        open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
        r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", os.path.join(HERE, "mut_tmp", "bt"), "test_navrbd_read.py"], cwd=tmp, env=env, capture_output=True, encoding="utf-8")
        red = [ln.split("::")[-1].split(" ")[0] for ln in r.stdout.splitlines() if ln.startswith("FAILED")]
        ok = r.returncode != 0 and bool(red) and expect in red
        bad += (not ok)
        res.append({"mutation": name, "file": f, "red_tests": red, "expected_red": expect, "expected_in_red": expect in red, "went_red": bool(red)})
        print(("RED " if ok else "NOT-AS-EXPECTED(!) ") + name, "| expected:", expect, flush=True)
    shutil.rmtree(os.path.join(HERE, "mut_tmp"), ignore_errors=True)
    os.makedirs(HERE + "/out", exist_ok=True)
    json.dump(res, open(HERE + "/out/MUTATIONS_NAVRBD.json", "w"), indent=1)
    print(f"{len(res)} mutations, {len(res) - bad} as expected")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
