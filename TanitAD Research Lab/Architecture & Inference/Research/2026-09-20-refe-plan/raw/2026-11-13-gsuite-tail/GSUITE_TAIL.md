# G-SUITE failure tail (2026-11-13)

**What.** `check_suite` (refe/gate_checks.py) now adds `stdout_tail` (last 40 lines) and `stderr_tail` (last 20) to the dict of a FAILING suite script. Each line is cut to 300 chars; a line matching `hf_[A-Za-z0-9]{8,}` or containing `HF_TOKEN=` becomes `<redacted>`. Passing scripts get no new keys. Reasons, status, rc and token logic are untouched. Adds `import re` and a helper `suite_tail`. Diff: diffs/gate_checks.diff (+16 lines).
**Why.** HC-s0 failed G-SUITE on the pod with only "rc 1, token NOT seen"; the cause had to be re-derived on the dev box (~45 min per pod retry).
**Base.** tip c78dff15f98407e1a5d224a2d7b7d2db1754bfe3 (agent/arch-inf-20260803). gate_checks.py has no CR on the tip; hold files are LF.

## Measured
- refe/test_gsuite_tail.py: `ZZTEST_PASS 3` (evidence/test_gsuite_tail.out). Checks: exact tails, 40/20 limits, 300 cut, token redaction (also HF_TOKEN=), passing entry keys == old set, verdict/reasons/rc equal to the tail-less engine on 4 mixed entries, and a MUTANT (temp copy of gate_checks.py without the two tail lines, loaded by importlib) that FAILS the judge (so the test is not vacuous). It clears REFE_GATE_NESTED around its calls so it works inside a suite.
- Regression: all 52 refe/test_*.py that mention gate_checks or launch_gate, run one at a time on a pristine export and on the patched one (CUDA off): **52 of 52 give the identical rc and result line** (evidence/PRISTINE_VS_PATCHED.md, regress_*.jsonl).
- 7 scripts fail IDENTICALLY on both (environment, not the patch): test_carve_feed, test_launch_gate (FileNotFoundError D:\Projects\TanitAD\data\refe_consisten...), test_rls1_chain (no MODEL_REGISTRY.md above the package), test_s1z_preflight, test_s1zsm_preflight, test_scorer1p6_preflight, test_vdac_stage0_forward. Because the export holds only the package dir, these were NOT exercised on this dev box: UNVERIFIED for the patch. Skips (no CUDA, no nuPlan devkit) also identical.
- Pins: no pinned fingerprint or sha in a test moved (no test failed on patched that passed on pristine). BUT gate_token.py binds `gate_checks_sha256` (sha256 of gate_checks.py) into every minted token, so any existing PASS token or accepted binding is void once this lands; the pod needs a freshly minted token and a pod copy of the new gate_checks.py. test_gate_suite_prerun (144 checks) reads it live and passed.
- Not covered by the spec: a script that only fails on SKIPS (token seen, rc 0) gets no tail.

## Deletions (rule 1-4)
- F:/tanitad_scratch/gsuite_tail/export (837M), F:/tanitad_scratch/gsuite_tail/patched (837M): deleted by the agent; the MM confirmed the folder empty on 2026-11-13 ~05:55Z.
- Scratchpad helper files runall.py and list.txt (tiny) left in the session scratchpad.

## Integration (MM, 2026-11-13)
- Landed on the repo only. The running pod trees (`/workspace/refe-plan-p7`, `/workspace/refe-plan-vdac` at c78dff15) are NOT re-shipped: `gate_token.py` binds `gate_checks_sha256`, so a re-ship would void their stage markers and tokens mid-stage. The next ship of either tree carries the tail.
- `refe/p7_ship_list.json` records gate_checks.py as `0e6da3ee...`: that is P7's ship-time record (the tip was already `b9179307...` before this change), not a live pin.
