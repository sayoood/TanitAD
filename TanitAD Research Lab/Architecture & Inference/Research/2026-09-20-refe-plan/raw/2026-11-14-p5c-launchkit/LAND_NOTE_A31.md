# LAND NOTE (MM, 2026-11-15 22:45Z): the kit land after P7, with A3.1

## 1 · What lands in `PK/refe/`
- The kit's `files/` with **two replacements from the A3.1 hold**:
  - `refe/p5c_pod.sh` → `p5c_a31/refe/p5c_pod.sh` (blob `11d657d9`). It replaces the kit's A2 copy, blob `cf82384478d4…`, sha256 `6b6e121f…`.
  - `refe/test_p5c_chain.py` → `p5c_a31/refe/test_p5c_chain.py` (blob `67fda327`). It replaces the kit's `e1b541f2`.
- The kit's G-SUITE entry for the chain test (`launch_gate.py:496`, token `ZZP5CCHAIN_PASS`) matches what the A3.1 version prints. No other kit file references either blob.

## 2 · Doc typo (no rule changes)
- A3 (lines 184 and 214) and A3.1 §1 abbreviate the A2 blob of `p5c_pod.sh` as `cf823843`. The blob is `cf82384478d4…`, so the correct 8-character form is **`cf823844`**.
- The sha256 they quote (`6b6e121f…`) is correct, and it is the binding pin.
- Record this in the kit land's GAC row; do not edit the frozen files.

## 3 · Pre-window check
MEASURED, `logs/mm_kitchk_2026-11-15/`, driver `kitchk.sh`.
- **Tree:** tip `f1f5159c` + kit `files/` + the two A3.1 files. Every blob was verified: launch_gate, train and gate_checks equal the kit; p5c_pod.sh is `11d657d9` and test_p5c_chain.py is `67fda327`.
- **Run:** serial, CPU, tanitad venv, TMP on F: scratch.
- **Results:** all 10 scripts returned rc 0.

| script | token |
|---|---|
| `test_p5c_chain` | 126 |
| `test_p5c_profile` | `--fast` 233; full 261 |
| `test_p5c_live --fast` | 49 |
| `test_p5c_postgate --fast` | 32 |
| `test_traj_label_table --fast` | 166 |
| `test_launch_gate --static --no-banked` | 1180 |
| `test_suite_nesting` | 286 |
| `test_p7_head_wiring --static` | 182 |
| `test_ego_yaw --fast` | 160 |

- Every count equals the HANDOFF_2 / A3.1 record, except `test_p5c_live --fast`: 49 here, against 48 in `TESTS.md`. No log of the 48 survives, so the delta is **unattributed**. Every section reads ok; it is a PASS, not a regression.
- **Kit land guard:** all 13 guarded tip blobs still equal `REBASE.md` on `f1f5159c`, and the 5 new files are still absent.

## 4 · Still to do at land time (unchanged)
1. Fill the 3 pins from the real objects:
   - `P5C_BASE_SHA256`: sha256 of P7's `model_final.pt`.
   - `P5C_LABEL_TABLE_SHA256`: what the runner prints.
   - `P5C_RANK_POLICY`: the table's `meta.rank_policy`, which is `rank0` under A3 §3.3.
2. Re-run the pin-sensitive tests, including all launch_gate-reading tests, because a pin edit moves profile fingerprints.
3. Re-derive `SHIP_MANIFEST.json` and the ship list (`--profile refe_p5c`).
4. Run `ls-remote` and re-check the land guard on the tip of that day.
