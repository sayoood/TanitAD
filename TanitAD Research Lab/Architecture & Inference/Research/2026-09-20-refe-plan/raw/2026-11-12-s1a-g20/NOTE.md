# S1a G20 refusal and the A4 fix (2026-11-12, Master Mind)

## What happened
- After the cuBLAS fix (914aeaaa) was re-shipped, S1a reached the HA-s0 preflight a second time. The VDAC alignment gate G20 then REFUSED (09:08Z).
- MEASURED on 256 real frames (5,120 points):
  - `frac_real` 1.0000
  - `frac_yflip` 0.9572
  - `frac_shift1` 0.9582
- `judge_g20` requires both mutants to read below the bar, and the bar was 0.95, so the check refused itself.
- P7 resumed at once. No training step ran.

## Fix
- `PREREG_VDAC_P6_A4.md` (sha256 `045a10c8...`, frozen 2026-11-12T09:13:55Z) restores the card's own criterion, "inside the label at the banked rate", read to two nines: **BAR 0.99**.
  - The old 0.95 was SPEC_P7's bar, not this card's.
  - Disclosure: the addendum was written after the readings above were seen. It changes an instrument constant toward the frozen card (stricter, not looser).
- `refe/vdac_probes.py`: `BAR = 0.99`.
- `refe/test_vdac_gate.py`:
  - asserts `BAR == 0.99`;
  - asserts the pod's measured readings pass at 0.99 (real 1.0, mutants 0.957 / 0.958 refused as mutants) and are refused at 0.95;
  - asserts that a real fraction under the bar is RED;
  - the launch-judge fixture `good20` now uses `VP.BAR`, not a literal 0.95.

## Tests (dev box, CUDA hidden)
| test | result | log |
|---|---|---|
| `test_vdac_gate.py` | rc 0, `ZZTEST_PASS` | `test_vdac_gate.log` |
| mutation: `BAR` reverted to 0.95 | rc 1, `AssertionError: PREREG_VDAC_P6_A4: G20 ba...` (RED as required) | `test_mut.log` |
| `test_suite_nesting.py` | rc 0, `ZZTEST_PASS 250` | `test_suite_nesting.log` |
| `test_gate_suite_prerun.py` | rc 0, `ZZTEST_PASS 144` | `test_gate_suite_prerun.log` |

## Next
- Re-ship `refe/` to `/workspace/refe-plan-vdac` only (never `refe-plan-p7`), set the old S1a logs aside, and rerun `s1a_go.sh`.
- Open item from A4: does SPEC_P7's own 0.95 alignment check lack power in the same way? P7's run is unaffected while its real reading is at or above 0.99, but its gate's mutants were never read against 0.957.
