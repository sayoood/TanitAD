# HANDOFF_LFOBUILD: the logged-speed oracle (`--floor-vhat-source {heads,logged}`) in the LF tool, for arm LFO075 (A4, sha 8bfbc386). Done, staged, nothing committed.

LFO tool `lfo_build/tool/v8f_t1_eval_ri_lfo.py` sha256 `6744d3e4686e344f...` (`PINS.txt` key `patched_tool`); placed in the read tree `F:/TanitAD-scratch/motion_c/lfo_read/tree/stack/scripts/v8f_t1_eval_ri.py` beside the unchanged operator `level_floor.py` (`2781e23a...`). `code/apply_lfo_patch.py` regenerates the tool byte for byte from the LF tool 6166178a (checked in `PINS.txt`).

## Interface
* `--floor-vhat-source heads` (default under `--level-floor on`): the LF tool's behaviour bit for bit. `logged`: v_hat := the logged route speed of the tick (`logged_route_speed(gt_in_frame(...))`: path length of knots 1..6 of the tick's `gt` / 1.0 s, the reader's `logged_speed`), computed at the replan BEFORE F from the same `gt_in_frame` the tick dump uses; `gt_ok` False (or a non-finite knot) gives NaN, hence no floor. Presence and gap still come from the LLR heads. Refused without `--level-floor on`; an explicit `heads` with the floor off is refused too.
* Wiring: `rollout` hands the window states to the planner (`planner.states`), `model_planner` hands them to the floor, the floor's `oracle` (built in `cmd_run` from the same ClipStore and join) returns v_log.
* Header: `floor_vhat_source`; for `logged` also `"label": "CEILING"`, `level_floor_detail.vision_only` starting "NO.", an `open_channels` entry. Ticks file: `floor_vhat_source` (and `label` for logged) and the field `lf_vhat_heads` (what the heads said on the same tick); `lf_vhat` is then v_log.

## Known answers (MEASURED; `raw/`, `logs/`)
* KA-O1 PASS, full rollout RILF075_s0 (reindexed, 12 threads, CPU, 306 s) with `heads`: all 46 common tick fields (7,197 rows), 141 dump files, the analysis JSON and the 44 per-window metric arrays byte-identical to the banked RILF075_s0. Only differences: the ticks file gains `floor_vhat_source`; header keys `ticks_file`, `floor_vhat_source`, `level_floor_detail.floor_vhat_source`.
* KA-O2 PASS, full rollout RIC0_s0 with the floor off (317 s): same byte identity (35 common fields), only the header `ticks_file` differs.
* KA-O3 PASS: `tests/test_lfo.py` (7 tests, 4 of them `test_ka_o3_*`). Mutations (`logs/MUTATIONS_LFO.txt`): deleting the single oracle-assignment line fails 3 of 4 KA-O3 tests, dropping the gt_ok guard fails 1 of 4, 12 of 12 mutations RED. LF suite (`lf_build/tests`) on the LFO tool: 12 of 12 green.
* Smoke on real data (`raw/SMOKE_LFO.json`, 64 windows, bare and composite): `lf_vhat` = the reader's logged speed (max diff 9.4e-7 m/s, float32), NaN exactly where the route is incomplete (27 of 192 ticks), no floor there or below 3 m/s, v_f = 0.75 v_log where the cap did not bind.

## Notes
* About 14 % of ticks have no complete logged route (the clip ends within the 6 s horizon): the oracle arm has no floor there, as the card says. The reader reports the share.
* The base tool carries two inherited docstring sentences about the T2 wording (from the RI tool, not edited).

## Manifest
`code/` (apply_lfo_patch, ka_lfo, smoke_lfo, run_mutations_lfo, make_pins_lfo), `tests/test_lfo.py`, `tool/v8f_t1_eval_ri_lfo.py`, `raw/` (KA_LFO.json, KA_LFO_O1/O2.json, SMOKE_LFO.json), `logs/`, `PINS.txt`. Scratch: `F:/TanitAD-scratch/motion_c/lfo_build/` (`ka/`, `smoke/`, `mut/`, `pt/`), `F:/TanitAD-scratch/motion_c/lfo_read/tree/`.
