# P6 G-SUITE exact pre-run on the pod (ship c1dc5d3), 2026-10-16 — PASS

P6's ship `c1dc5d3` (2026-10-16 ~01:00Z) had never run its G-SUITE on the pod: the last pod suite run (`/workspace/ops/p6suite.log`, 2026-10-14) was an older ship.

This pre-run used the gate's own code: `gate_checks.check_suite(PROFILES["refe_p6"])`, from `/workspace/refe-plan-p6`, in P6's stage-11 environment (`p6_start.env` + `teacher_env.sh`; see `p6_gsuite_exact.py`'s docstring).
- The GPU was hidden, because P2-s1 was training.
- The key was a throwaway outside the tree.

**Result (MEASURED):**
- `PASS`, no reasons;
- 4/4 scripts rc 0 with 0 skips (test_p2r_chain, test_launch_gate --static, test_p6_levers --fast, test_p6_chain);
- **1,766.5 s**.

⇒ P6's launch gate will not refuse on G-SUITE after S1c, unless its environment changes. About 20 of the 29 min are `test_p2r_chain.py`'s real-data fingerprinting (MEASURE_RERUN.md §5). That is fixed from f41eec8 on, but P6 keeps c1dc5d3.
