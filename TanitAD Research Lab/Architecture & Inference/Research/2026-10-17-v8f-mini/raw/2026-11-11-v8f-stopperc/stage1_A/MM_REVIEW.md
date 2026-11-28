# Master Mind review of arm A's code (2026-11-12)

**Landed.**
* Commit `3e264bbf3fa2f6f17bea448de09aa972faaf7725` on `agent/v8f-stopaux-20261112`, child of `715ef8fc` (BAL-NULL's commit). Landed by `land_stopaux.sh`.
* Verified with `git ls-remote`; the commit changes exactly 5 paths, and each blob equals the file tested below (`git hash-object --no-filters`).

**What changed from the agent's package.**
1. Presence negatives (`AMENDMENT_3_A_READINGS.md` §1):
   * `stop_aux_labels` takes `lab["lon_cls"]` and keeps only moving windows (v9 lon 3..6) with no rest point as presence-0 rows.
   * An out-of-frame rest point, and HOLD / CREEP / unknown windows, are left out.
   * `test_label_masks` was rewritten to pin this.
   * The agent's originals are banked in `agent_v1/`.
2. Camera table (`AMENDMENT_3` §2):
   * `build_stopaux_extr_table.py` builds `extr/stopaux_extr_join4572.json` (4,572 / 4,572 clips, sha256 `77b6e8dc…e361`).
   * The repo's 2,400-clip table covers 187 of the 4,512 training clips, so the gate would have refused.
3. `land_stopaux.sh`: three re-measured pins (trainer `465a7f08…`, `v8f_stopaux.py` `fe80ac35…`, test `1278abd0…`); the model file is unchanged (`af428fb2…`); the commit trailer reads Opus.

**Tests (MEASURED, MM's own export of `715ef8fc` + the diff, CPU, `tanitad.__file__` inside the export).**
* `test_v8f_stopaux.py`: **38 passed** (145 s).
* The touched suites `test_launch_gate_v8f`, `test_v8f_mini`, `test_v8f_stopbal`, `test_v8f_lead_head`, `test_v8f_level_head`, `test_v8f_init_ckpt_o_interface`: **171 passed, 2 skipped** (the agent's per-file pass counts sum to the same 171).
* **Mutation:** the mask set back to the agent's reading (`neg = lab_ok & ~pos`) turns `test_label_masks` **red**; restoring the file brings back the pinned sha256.
* The agent's OFF-path bit-identity (same `ckpt_last.pt` sha256 on base and patched trainers, `58c085e4…`) was not re-run here. The change since then touches only `--stop-aux on` code: the label function is called only when the head is on.

**Open before any arm-A launch (none of these is a code defect).**
* **PI's stage-1 go.**
* **A launch-gate PASS per run.** G-DVB entries are listed in `RESULTS.md` "Gate list". The STOPBAL-style box-2 checker still needs those clauses added.
* **W.** Set from the real G-LIVE probe (`stop_aux_w_from_smoke.py`), not from a synthetic run.
* The real probe batch must hold at least one in-frame rest point.
  * The probe is the sampler's first batch (`--batch`, default 64), drawn from a few clips at a time by `ChunkedSampler`, and fixed by the seed.
  * About 4 % of training rows are STOP (13,116 / 306,566), so about 2.5 in-frame rest points are expected. A zero has a binomial chance of about 8 %, and clip clustering makes it more likely (ESTIMATED).
  * ⇒ The gate smoke reads `probe_n_pos` first. If it is 0, the refusal is named (`... cannot see the cell term`), and the fix is a STOP-stratified probe batch: a code change, landed before the launch.
