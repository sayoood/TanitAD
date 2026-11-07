"""Mutation check: for every rule, remove the single line that decides its verdict (marked `# MUT:<rule>`) in a COPY of
behaviour_score.py and require the rule's deliberate-regression test to go RED; the unmutated module must be GREEN on the same tests.

Two mutations per rule:
  neuter : the verdict line becomes `status = "PASS"`  (the rule silently stops detecting anything)
  delete : the verdict line is removed                  (the rule no longer computes its verdict at all)
f1 additionally gets  oldrule : the marked classifier call (# MUT:f1_cls) is swapped for the SUPERSEDED +-45 deg bearing rule
  (superseded_a10_bearing_rule.py, AMENDMENT_1 ruling A-10); the A-10 scenario test must go RED.
The original file is never edited (its sha256 is checked before and after) so 'restore it' is trivially true.
Usage: python mutation_check.py [--out mutation_check_result.json]     exit 0 only if every mutant is killed.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "behaviour_score.py"
TESTS = {
    "a1": "test_a_phantom_braking_fails_a1_and_a2",
    "a2": "test_a_phantom_braking_fails_a1_and_a2",
    "b1": "test_b_tailgating_at_08s_fails_b1_and_b2",
    "b2": "test_b_tailgating_at_08s_fails_b1_and_b2",
    "c1": "test_c1_stall_at_clear_junction_fails",
    "c2": "test_c2_stop_3m_past_line_on_red_fails",
    "c3": "test_c3_entering_zone_with_conflicting_agent_within_2s_fails",
    "c4": "test_c4_creep_stall_after_green_fails",
    "d1": "test_d1_harsh_braking_fails_and_matches_navsim_on_the_same_states",
    "d2": "test_d_oscillating_speed_fails_d2",
    "e1": "test_e1_lane_drift_08m_for_3s_fails",
    "e2": "test_e2_heading_bias_fails_and_curvature_ringing_fails_but_clean_passes",
    "f1": "test_f_overlap_with_a_stationary_agent_fails_f1_and_f2",
    "f2": "test_f_overlap_with_a_stationary_agent_fails_f1_and_f2",
}
OLDRULE_TEST = "test_f1_ego_stopped_is_speed_le_0p05_not_0p005"     # A-10: ego 0.04 m/s is stopped in NAVSIM source (the old rule's threshold was 0.005)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_tests(module_dir, names):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", BEHAV_VENDOR_DIR=str(HERE))
    if module_dir:
        env["BEHAV_MODULE_DIR"] = str(module_dir)
    p = subprocess.run([sys.executable, str(HERE / "test_behaviour_score.py"), "--only", ",".join(sorted(names))], env=env, capture_output=True, text=True)
    return p.returncode, p.stdout


def main():
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else HERE / "mutation_check_result.json"
    sha_before = sha(SRC)
    lines = SRC.read_text(encoding="utf-8").split("\n")
    result, ok = {}, True
    base_rc, base_out = run_tests(None, set(TESTS.values()))
    result["_baseline_green"] = base_rc == 0
    ok &= base_rc == 0
    tmp = Path(tempfile.mkdtemp(prefix="mut_"))
    try:
        for rule, test in TESTS.items():
            idx = [i for i, l in enumerate(lines) if re.search(rf"# MUT:{rule}$", l)]
            assert len(idx) == 1, (rule, idx)
            i = idx[0]
            indent = re.match(r"\s*", lines[i]).group(0)
            result[rule] = {"test": test, "line": i + 1, "original": lines[i].strip()}
            for kind in ("neuter", "delete"):
                m = list(lines)
                if kind == "neuter":
                    m[i] = f'{indent}status = "PASS"  # MUTATED'
                else:
                    del m[i]
                d = tmp / f"{rule}_{kind}"
                d.mkdir()
                (d / "behaviour_score.py").write_text("\n".join(m), encoding="utf-8")
                rc, so = run_tests(d, {test})
                killed = (rc != 0) and (f"FAIL {test}" in so) and (str(d) in so)     # RED, and it ran against the mutant
                result[rule][kind + "_killed"] = bool(killed)
                ok &= bool(killed)
            if rule == "f1":                                      # A-10 deliberate-regression arm: the old bearing rule swapped back in
                j = [k for k, l in enumerate(lines) if l.rstrip().endswith("# MUT:f1_cls")]
                assert len(j) == 1 and lines[j[0]].count("pdm_collision.get_collision_type(") == 1, j
                m = list(lines)
                m[j[0]] = lines[j[0]].replace("pdm_collision.get_collision_type(", '__import__("superseded_a10_bearing_rule").old_collision_type(')
                d = tmp / "f1_oldrule"
                d.mkdir()
                (d / "behaviour_score.py").write_text("\n".join(m), encoding="utf-8")
                rc, so = run_tests(d, {OLDRULE_TEST})
                killed = (rc != 0) and (f"FAIL {OLDRULE_TEST}" in so) and (str(d) in so)
                result[rule]["oldrule_test"], result[rule]["oldrule_killed"], ok = OLDRULE_TEST, bool(killed), ok and bool(killed)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    result["_original_sha256_before"], result["_original_sha256_after"] = sha_before, sha(SRC)
    result["_original_untouched"] = sha_before == sha(SRC)
    result["_all_killed"] = bool(ok and result["_original_untouched"])
    out.write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    for r in TESTS:
        print(r, "neuter:", "KILLED" if result[r]["neuter_killed"] else "SURVIVED", "| delete:", "KILLED" if result[r]["delete_killed"] else "SURVIVED",
              ("| oldrule (A-10 +-45 deg bearing swapped back in): " + ("KILLED" if result[r]["oldrule_killed"] else "SURVIVED")) if r == "f1" else "")
    print("baseline green:", result["_baseline_green"], "| original untouched:", result["_original_untouched"], "| ALL KILLED:", result["_all_killed"])
    return 0 if result["_all_killed"] else 1


if __name__ == "__main__":
    sys.exit(main())
