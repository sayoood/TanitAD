"""mutate_check.py -- the deliberate-regression arms of the STRAT1 readers.  Each mutation is applied to a COPY of the reader files (a temp folder under mut_tmp, deleted afterwards) and
test_strat1_readers.py is run against the copy: every mutation must turn at least one named test RED.  Usage: python mutate_check.py   ->  out/MUTATIONS.json, exit 1 if any mutation stays green."""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
FILES = ["strat1_lib.py", "strat1_strategic.py", "strat1_ctx_read.py", "strat1_read.py", "run_dumps_strat1.py", "strat1_k4_eval.py", "strategic_cells.py", "test_strat1_readers.py", "READ_PINS.json"]
MUTS = [
    ("floor comparison flipped (obd_read.cell_delta)", "theory/obd/read/obd_read.py", "above = abs(pt) > floor", "above = abs(pt) < floor", "test_floor_rule_sub_floor_neither_gain_nor_cost"),
    ("ctx accuracy counts masked (-1) rows", "strat1_ctx_read.py", "    m = y >= 0\n    yy, pp, ee = y[m], pred[m], eid[m]", "    m = y >= -1\n    yy, pp, ee = y[m], pred[m], eid[m]", "test_ctx_masked_rows_not_scored"),
    ("route-following outcome reads zero nav instead of donor", "strat1_strategic.py", 'b["real_minus_donor"][DECIDE]', 'b["real_minus_zero"][DECIDE]', "test_route_following_outcomes"),
    ("a v9 arm may be given a text-nav file", "run_dumps_strat1.py", "        if npz or sha:\n            raise ValueError(f\"{arm}: a v9-nav", "        if False:\n            raise ValueError(f\"{arm}: a v9-nav", "test_nav_args_refusals"),
    ("twilight not masked", "strat1_ctx_read.py", "            ok = ok & (ids != TWILIGHT_ID)", "            ok = ok", "test_label_masking_twilight_flag_valid"),
    ("COSTS-NAV needs no N1 cost", "strat1_read.py", "nav = sorted(both & n0 & n1)", "nav = sorted(both)", "test_driving_costs_nav_heads_disagree"),
    ("K4 re-init skipped", "strat1_k4_eval.py", "                m.reset_parameters()", "                pass", "test_k4_reinit_changes_head_and_nothing_else"),
    ("misaligned label table accepted", "strat1_ctx_read.py", "        if not np.array_equal(tab[f\"eval_{nm}\"], join[nm]):", "        if False:", "test_misaligned_table_refused"),
]


def main():
    res, bad = [], 0
    env = dict(os.environ, STRAT1_TREE=HERE + "/tree", STRAT1_INPUTS=HERE + "/inputs", STRAT1_CKPTS=HERE + "/ckpts", CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    for name, f, old, new, expect in MUTS:
        tmp = os.path.join(HERE, "mut_tmp", "m")
        shutil.rmtree(os.path.join(HERE, "mut_tmp"), ignore_errors=True)
        os.makedirs(tmp)
        for x in FILES:
            shutil.copy(os.path.join(HERE, x), tmp)
        shutil.copytree(os.path.join(HERE, "theory"), tmp + "/theory")
        p = os.path.join(tmp, f)
        s = open(p, encoding="utf-8").read()
        assert s.count(old) >= 1, (name, "pattern not found")
        open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
        r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_strat1_readers.py"], cwd=tmp, env=env, capture_output=True, encoding="utf-8")
        red = [ln.split("::")[-1].split(" ")[0] for ln in r.stdout.splitlines() if ln.startswith("FAILED")]
        ok = r.returncode != 0 and bool(red)
        bad += (not ok)
        res.append({"mutation": name, "file": f, "red_tests": red, "expected_red": expect, "expected_in_red": expect in red, "went_red": ok})
        print(("RED " if ok else "GREEN(!) ") + name, red, flush=True)
    shutil.rmtree(os.path.join(HERE, "mut_tmp"), ignore_errors=True)
    json.dump(res, open(HERE + "/out/MUTATIONS.json", "w"), indent=1)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
