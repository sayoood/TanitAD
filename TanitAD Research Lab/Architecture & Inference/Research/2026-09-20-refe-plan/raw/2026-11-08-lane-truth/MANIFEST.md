# MANIFEST lane_truth (2026-11-08). Nothing staged, nothing pushed; the Master Mind lands. Every item exists in ONE place (this folder, E: drive) until landed.

| artifact | path (under `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/lane_truth/`) |
|---|---|
| result | `RESULT_LANE_SPEED.md` |
| numbers | `lane_speed_result.json`, `gl1_gates.json`, `gl1b.json`, `per_token_gap.npy` |
| pre-registration + amendments | `PREREG_LANE_SPEED.md`, `AMENDMENT_1.md`, `AMENDMENT_2.md` (+ .sha256) |
| code | `code/analyze_lane.py` (patched: intersection of arms, read label), `code/gl1_gates.py` (new), `code/gl1b.py` (new), `code/gl1_key.py`, build/chain scripts |
| label side files | `lane_labels/out_cruise.jsonl` (154 CRUISE tokens), `out_ctrl2_ph|real.jsonl` (34 each), `out_ctrl3_nocache|norepair.jsonl` (6 each), `out_ctrl_ph.jsonl` (24), queues `q_cruise|q_ctrl2_ph|q_ctrl2_real.jsonl` |
| logs | `logs/` |

## Deletion log (CLAUDE.md auto-delete rule)
- Deleted `F:/scratch/lane_truth_work/run/` (59,619,177 bytes) with `Remove-Item -LiteralPath`, after RESULT and the label side files were written. Content: copies of the label outputs and queues (banked above) plus `c3tab/table.npz` (re-creatable: C3 table, mirror tip e112911d blob 7e89d02c, sha256 `cf30519051bea03b...`). `q_other` (optional, never queued) was empty. Consequence: `analyze_lane.py` reads `W = F:/scratch/lane_truth_work/run`; to rerun, re-export table.npz and the side files there.
- No process was killed.
