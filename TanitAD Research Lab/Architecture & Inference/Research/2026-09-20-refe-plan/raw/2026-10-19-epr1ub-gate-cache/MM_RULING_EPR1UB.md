# MM ruling on EPR1UB-GATE-CACHE (Master Mind, 2026-10-19)

**1. The EPR1UB fix is ACCEPTED and LANDED** (`raw/2026-10-19-ep-r1ub/code/epr1ub_gate.py`, `test_epr1ub_gate.py`).
- `load_verified_ub` no longer caches. Each load gets a unique module name, and a failed load removes its module.
- `_load_landed_gate` now requires the module's `__file__` to be the verified path.
- Mine passes 23 / 23 in all 7 orders (unittest default / first / last; pytest default / first / last / reversed).
- The tip fails the named test in 4 of the 7 orders, which confirms the order-dependence.
- The RED arms (second dir, same dir in a fresh interpreter, leftover module) catch the sha-only cache when it is put back.
- The staged bytes equal the tested copies (sha256 `a74129ae…` / `5840ef62…`).
- `CODE_SHA256_EPR1UB.json` and the EPR1UB gate token describe the pre-fix bytes and stay as the record. **The gate must be re-minted before any reuse.**

**2. The C2 edits are NOT landed.** They are kept as a record under `c2_not_landed/`; the live `2026-10-18-sel-rank/code/` files are unchanged.
- With them, `test_srttc_c2_gate.py` reads 42 / 43 against 39 / 39 on the tip.
- The error is `FileNotFoundError` in `srttc_c2.run_c2`. `R.__file__` (srttc_r1) points into a deleted temp dir that an earlier new poison test loaded.
- ⇒ The C2 chain resolves `srttc_r1` by a process-global module, not per verified code_dir. This is the SAME defect class the fix targets: a sibling module is not re-verified per directory.
- The audit's "no sibling import" reading of the C2 loaders therefore missed a path that the new test reached.
- The C2 line is VOID, so this is a BACKLOG item (`C2-GATE-SIBLING-LEAK`, fix before any reuse), not a blocker.
- ⛔ A red test is never landed to make a point.
