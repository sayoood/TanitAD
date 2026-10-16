# S1c pod G-SUITE pre-run FAILED on test_s1c.py: a repo-relative path indexed at import time (fixed)

**Master Mind, 2026-10-16.** The exact-env pre-run (`gate_checks.check_suite(PROFILES["refe_aux_s1c"])`, shipped tree 10d4fae, the env `s1aux_pod.sh` gives `launch_gate.py`) read FAIL.
- Five of six suites passed.
- `test_s1c.py` died after **0.2 s with rc 1**: `test_s1aux_chain.py:35` computed `PKGROOT.parents[3] / "FlyWheels"/…/navtest.json` at IMPORT time.
- On the pod, `PKGROOT` is `/workspace/refe-plan-s1aux`, which has 2 parents, so the import raised `IndexError: 3`. `test_s1c.py:435` built the same path eagerly.

**Why the dev box missed it.** The build's "pod-like tree" check ran at a deep dev-box path, where `parents[3]` exists. A pod-like CONTENT check is not a pod-like PATH check. This is the same class as the 2026-10-14 and 2026-10-15 G-SUITE failures: proven only on the dev box.

**The fix (2 lines).**
- On a shallow tree the repo copy cannot exist, so the path resolves to a NON-existent file. The sandbox end-to-end cases then SKIP on the pod, exactly as their docstring says (`test_s1c.py:603`).
- It deliberately does NOT fall back to `$AUX_NAVTEST`: that would silently switch the end-to-end case on, on the pod.

**Proof (MEASURED, `logs/podfix/`):**

| tree | path | result |
|---|---|---|
| fixed | deep dev-box | `test_s1aux_chain` 404, `test_s1c` 241 (unchanged) |
| fixed | shallow `C:/s1cfixA` (1 parent, pod-like) | `test_s1c` 219 + 2 skipped |
| UNFIXED | shallow `C:/s1cfixB` | `IndexError: 3`, the pod failure reproduced |

## Part 2: pre-run #2 (ship fd7e175) FAILED on a SKIP; the e2e scope is now DECLARED (fixed)
- After the import fix, every suite ran. `test_s1c.py` printed `ZZTEST_SKIP end_to_end`, and G-SUITE refuses ANY skip (`gate_checks.py:1301`, "a skip is not a pass").
- That case is dev-box-only by design: it runs the real chain in `test_s1aux_chain`'s sandbox, which needs the repo checkout. `test_s1aux_chain` itself is in no pod suite for the same reason.
- **The fix declares the scope, never hides it.**
  - `launch_gate.AUXS1C_SUITE` runs `test_s1c.py --no-e2e`. With that flag the case prints `ZZTEST_SCOPE` (neither a pass nor a skip).
  - A dev-box run without the flag still runs it.
  - A new check pins `args == ["--no-e2e"]` in the suite entry.
- Rejected alternatives:
  - turning off `suite_skips_fail` for the profile, which would weaken the gate for every script;
  - printing nothing for the case, which would hide it.

**Proof (MEASURED, `logs/podfix2/`):**

| run | result |
|---|---|
| deep, no flag | `test_s1c` **242** (241 + the pin; the e2e case RAN), 0 skips |
| shallow `C:/s1cfixC` (pod-like), `--no-e2e`, `AUX_NAVTEST` set as on the pod | **225**, **0 skips**, 1 declared scope line |
| gate static | `test_launch_gate --static` **645** (unchanged) |

The unfixed state is pre-run #2 itself (`logs/podfix/`).
