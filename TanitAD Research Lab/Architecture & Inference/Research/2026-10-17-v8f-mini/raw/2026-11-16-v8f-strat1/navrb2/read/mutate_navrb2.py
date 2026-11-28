"""mutate_navrb2.py -- the deliberate-regression arms of the NAVRB2 read.  Each mutation is applied to a COPY of the study's files (a temp folder under mut_tmp, deleted afterwards; the frozen STRAT1 readers are
imported from their own folder and are never mutated here) and test_navrb2_read.py is run WHOLE against the copy: every mutation must turn at least one named test RED.
Usage: python mutate_navrb2.py -> out/MUTATIONS_NAVRB2.json, exit 1 if any mutation stays green or the expected test stays green."""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
FILES = [f for f in os.listdir(HERE) if f.startswith(("navrb2_", "test_navrb2", "make_standin")) and f.endswith(".py")]
MUTS = [
    ("Z-EVAL: the exact-zero comparison dropped", "navrb2_read.py", "ok = ok and mx == 0.0 and nan_mismatch == 0", "ok = ok", "test_z_eval_one_window_in_mask_fails"),
    ("Z-EVAL: the zero-history mask ignored (reads every window)", "navrb2_read.py", "    m = np.asarray(mask, bool)\n    cells, ok = {}, bool(m.any())", "    m = np.ones(len(mask), bool)\n    cells, ok = {}, True", "test_z_eval_difference_outside_mask_is_not_read"),
    ("D1: the RB stratum ignored (reads all windows)", "navrb2_read.py", "np.asarray(arm[DECIDING])[m], np.asarray(ref[DECIDING])[m], np.asarray(eid)[m], True, floor, nb)",
     "np.asarray(arm[DECIDING]), np.asarray(ref[DECIDING]), np.asarray(eid), True, floor, nb)", "test_d1_reads_only_the_stratum"),
    ("RB-CLOSES: the guard dropped", "navrb2_read.py", 'if pair == "S1R-S1" and all(gain) and guard_ok and all(zuse.values()):', 'if pair == "S1R-S1" and all(gain) and all(zuse.values()):', "test_label_closes_needs_the_guard_zuse_and_the_s1_pair"),
    ("RB-CLOSES: Z-USE dropped", "navrb2_read.py", 'if pair == "S1R-S1" and all(gain) and guard_ok and all(zuse.values()):', 'if pair == "S1R-S1" and all(gain) and guard_ok:', "test_label_closes_needs_the_guard_zuse_and_the_s1_pair"),
    ("RB-UNREAD fires on one unread seed (reading r3 broken)", "navrb2_read.py", "    if not any(zuse.values()):\n        labels.append(\"RB-UNREAD\")", "    if not all(zuse.values()):\n        labels.append(\"RB-UNREAD\")", "test_label_read_no_gain_and_unread"),
    ("RB-NAV-ONLY: the S1R pair condition dropped", "navrb2_read.py", 'and not all(s[1]["d1_gain_by_seed"])', 'and True', "test_cross_pair_nav_only"),
    ("RB-SEEDS-DISAGREE: the one-seed gain test dropped", "navrb2_read.py", "if sum(gain) == 1 or one_cost:", "if one_cost:", "test_label_seeds_disagree_on_a_one_seed_gain_or_cost"),
    ("RB-COSTS: the D1 both-seed cost dropped", "navrb2_read.py", "if all(cost) or both_cost:", "if both_cost:", "test_label_costs_d1_both_seeds_or_a_guard_cell_at_both_seeds"),
    ("K-RE: a vacuous compare accepted", "navrb2_read.py", "return (n > 0 and not bad), bad, n", "return (not bad), bad, n", "test_k_re_equal_changed_and_vacuous"),
    ("strata: the history window crosses clips", "navrb2_lib.py", "lo = max(int(off[c]), r - int(hmax) + 1)", "lo = r - int(hmax) + 1", "test_window_state_history_does_not_cross_clips"),
    ("strata: the state read 5 rows AHEAD of NOW (the G-CLOCK-RB regression arm)", "navrb2_lib.py", "now[i] = x[r] > 0.5", "now[i] = x[min(r + 5, len(x) - 1)] > 0.5", "test_window_state_now_and_history_rule"),
    ("the nav_rb table pin not checked", "navrb2_lib.py", "    if got != RB_SHA256:", "    if False:", "test_rb_npz_pin_refuses_a_changed_file"),
    ("an R arm's bank commit not checked", "navrb2_lib.py", '    if man.get("commit") != COMMIT:', "    if False:", "test_refusal_accepts_a_good_bank_and_refuses_each_defect"),
    ("the zero twin dumped with the channel on", "navrb2_dumps.py", 'return "zero" if tag.endswith(NL.Z) else "follow"', 'return "follow"', "test_t1_argv_follow_and_zero_twin"),
    ("T0 donor pass without the named channel variant", "navrb2_t0.py", '"donor": ["--nav-shuffle", "--nav-rb-variant", "zero"]', '"donor": ["--nav-shuffle"]', "test_t0_variants_donor_holds_channel_zero_and_zero_is_withheld"),
    ("the view takes S1 / N1 from the banked passes", "navrb2_view.py", 'out.append((f"{t0}/eval_NAVRB2-{fam}_s{s}_{v}.npz"', 'out.append((f"{bt0}/eval_STRAT1-{name}_s{s}_{v}.npz"', "test_view_plan_names_and_refuses_missing"),
    ("the fog logit not masked", "navrb2_ctx_nofog.py", "        parts[W][:, FOG_ID] = -np.inf", "        pass", "test_nofog_masks_labels_and_logits_and_baselines"),
    ("ROUTE-COST on a single seed", "navrb2_strategic.py", "    if len(rows_by_seed) != 2:", "    if False:", "test_route_cost_cells_needs_both_seeds"),
]


def main():
    res, bad = [], 0
    env = dict(os.environ, NAVRB2_TREE=HERE + "/tree", NAVRB2_CKPTS=HERE + "/ckpts", CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
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
        r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", os.path.join(HERE, "mut_tmp", "bt"), "test_navrb2_read.py"], cwd=tmp, env=env, capture_output=True, encoding="utf-8")
        red = [ln.split("::")[-1].split(" ")[0] for ln in r.stdout.splitlines() if ln.startswith("FAILED")]
        ok = r.returncode != 0 and bool(red) and expect in red
        bad += (not ok)
        res.append({"mutation": name, "file": f, "red_tests": red, "expected_red": expect, "expected_in_red": expect in red, "went_red": bool(red)})
        print(("RED " if ok else "NOT-AS-EXPECTED(!) ") + name, red, flush=True)
    shutil.rmtree(os.path.join(HERE, "mut_tmp"), ignore_errors=True)
    os.makedirs(HERE + "/out", exist_ok=True)
    json.dump(res, open(HERE + "/out/MUTATIONS_NAVRB2.json", "w"), indent=1)
    print(f"{len(res)} mutations, {len(res) - bad} as expected")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
