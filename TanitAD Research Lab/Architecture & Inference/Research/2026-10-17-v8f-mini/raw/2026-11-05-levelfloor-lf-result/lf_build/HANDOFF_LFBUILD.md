# HANDOFF_LFBUILD: floor operator F in the T1-kin RI tool (A3, sha 977a785e; A3_RULING_1 applied)

**State: done, staged, nothing committed.** Patched tool `F:/TanitAD-scratch/motion_c/lf_read/tree/stack/scripts/v8f_t1_eval_ri.py` sha256 `6166178a604edc53...` (`PINS.txt` key `patched_tool`); `level_floor.py` sits beside it.

## Interface (as fixed)
- Flags: `--level-floor {off,on}`, `--floor-kappa K` (finite, 0..1), `--floor-ckpt PATH` (cfg must have level_head AND lead_head on, same data interface as the planner). Refused: on without kappa/ckpt, off with either, kappa outside [0,1], `--histswap` with the floor.
- Placement (A3_RULING_1): F after S in every branch. `--smooth sg72` only = raw, S, F (bare arm); sg72 + kbox = raw, S, F, kbox; smooth off = raw, F.
- Ticks (`--dump-ticks`): `lf_bind`, `lf_cap` bool; `lf_vf` (NaN = no floor), `lf_vhat`, `lf_pi`, `lf_gap` (metres), `lf_lift` f32; `plan_raw` is kept whenever the floor is on. EXTRA: `lf_bind_k10` (a knot among the first 10, the 2 s executed, changed). `lf_bind` is literal (any knot); late-knot binds have `lf_lift` 0.
- JSON header `t1v8f`: `level_floor`, `floor_kappa`, `floor_ckpt`, `floor_ckpt_sha256`, `level_floor_detail` (counts, floor-model keys read, parity).
- Floor heads: second model instance, own encode, adapter + lvl_head + lead_head only (keys read: feats, frame_ok); first replan checked against its `forward_infer` outputs: max diff 0.0.

## Results (CPU; logs in `logs/`, raw in `raw/`)
- Unit tests 12 green. Mutations: 13 of 13 RED (8 operator, 5 tool placement), `logs/MUTATIONS_LF.txt`.
- KA-LF1 floor off, first 64 windows: A logged 16t vs C0_s0, B reindexed sg72 12t vs RIC0_s0, C sg72+kbox vs RIC0C_s0: max diff 0.0 m (192 rows each), dumps 0.0.
- KA-LF2 kappa 0 same three: 0.0 vs banked and bit-identical to the off runs.
- Smoke kappa 0.5, s0, 6 windows (18 replans), reindexed: composite bind 1/18, cap 0; bare sg72 bind 1/18, cap 0; smooth off bind 4/18. `plan` recomputed from the dumped fields: 0.0 m.
- Wall, 64 windows composite 12t: off 12.3-13.2 s, on 13.7-15.9 s (0.19 vs 0.21 s/window).

## Pins
`PINS.txt`: original tool 0417c838, patched tool, level_floor.py, model 5fd4baea, data 22672d87, trainer e808cf1a, card, checkpoints. The patch regenerates the tool byte for byte.

## Notes
- Last-segment direction: exactly-zero segments fall back to the last non-zero one, else +x (counted in header). A near-zero last segment makes the extrapolation direction noisy (`short_last_seg_binds` in the header).

## Manifest (all under `E:/Projects/TanitAD-artifacts/dagland/motion_c/lf_build/`)
`code/` (level_floor, apply_lf_patch, ka_lf, smoke_lf, run_mutations_lf, make_pins_lf, run_all_lf.sh); `tests/test_level_floor.py`; `tool/` (patched, `orig/`); `logs/`; `raw/`; `PINS.txt`. Scratch (65 MB): `F:/TanitAD-scratch/motion_c/lf_read/` (`tree/`, `smoke/`, `ka/`, `mut/`, `aside/`).
