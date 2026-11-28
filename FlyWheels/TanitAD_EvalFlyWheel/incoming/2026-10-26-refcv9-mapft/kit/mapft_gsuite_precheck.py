"""mapft_gsuite_precheck.py -- Pre-check G-SUITE with the gate's OWN judge (launch_gate.judge_suite_verdict) on the MAPFT launch tree, before the gate.

A copy of /home/nvidia/refcv8_run/gsuite_precheck.py (the NT/refcv8 precheck) that reads PROFILES[<profile>] with the profile NAME AS AN ARGUMENT
(default refcv8-mapft) instead of the literal "refcv8".  The two profiles carry the SAME suite keys (suites, suite_mode, suite_env_failures, suite_pinned_skip_ok,
suite_skip_not_pass: asserted equal on 70211d6 on 2026-10-26), so the judgement is the one the gate will make.

usage: python mapft_gsuite_precheck.py <launch_tree> <suite_bound.json> <commit40> [profile]
prints a LAST LINE  `G-SUITE PRECHECK: PASS|FAIL`  (the caller reads that line from the captured file, not an exit code)."""
import sys
from pathlib import Path

tree, bound_p, commit = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
profile = sys.argv[4] if len(sys.argv) > 4 else "refcv8-mapft"
sys.path.insert(0, str(tree / "stack" / "scripts"))
sys.path.insert(0, str(tree / "stack"))
import launch_gate as L  # noqa: E402

prof = L.PROFILES[profile]
pin, prob = L.env_failure_pin(prof, tree)
print("profile:", profile, "| pin problem:", prob)
tree_sha = L.tree_digest(L.tree_manifest(tree))
reasons, det = L.judge_suite_verdict(L.read_json(bound_p), commit, tree_sha, pin or {}, tree)
print("launch tree sha", tree_sha[:16], "| bound cand tree", str(det.get("cand_tree_sha256"))[:16])
print("n_fail", det.get("n_fail"), "n_pinned", det.get("n_pinned"), "pinned_failing", det.get("n_pinned_failing"),
      "outside_pinned", det.get("n_outside_pinned"), "flaky_failing", len(det.get("flaky_failing") or []))
for x in (det.get("outside_pinned") or [])[:20]:
    print("  OUTSIDE:", x)
for r in reasons:
    print("  reason:", r[:300])
print("G-SUITE PRECHECK:", "FAIL" if reasons or prob else "PASS")
