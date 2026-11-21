# HANDOFF B (E5b: DK rows, S-POS, scene bootstrap, outcome logic), 2026-11-06, tip 24b97ba9

Evidence class: MEASURED (ours; logs in `logs/`: `mutations_e5b.log`, `mutations_e5b_part2.log`, `pytest_all_e5b.log`; hashes in `PINS_E5B.txt`). CPU only, no network, no box 2, nothing committed or staged.
Files: `code/t2v8f_lead.py` (new), `code/t2v8f_score.py` (extended), `tests/test_t2v8f_e5b.py`, `PINS_E5B.txt`, `HANDOFF_B.md`; `code/t2a_v8f_policy.py` changed ADDITIVELY (see Policy).

## Done
* **DK rows.** `lead_source.lead_block` (the function that built the banked block) from scene tracks. `sequence_tracks.json` is world frame; each sample goes to the rig frame at its own time (`obs_from_tracks`), `size_x` = cuboid length.
  Pin on 4 banked eval clips (all four states: LEAD 225, NO_LEAD 170, NOT_STRAIGHT 185, NO_LABEL 224): layer 1 bit for bit against the banked block and the reader's `join_block`; layer 2 (cuboids as world tracks through the adapter) same states, leads within 3e-14 m;
  layer 3 DK-VIOL / HW / TG on 68 banked RIC0_s0 windows: bit-identical (layer 1), 1e-14 (layer 2). 5 mutations RED.
* **S-POS@30/50/80.** `spos_errors` = the reader's match (float32 nearest, 3 m on-route), GT = v9 `rc` moved to the window origin, CTRV from the executed track, S from `s_ckpt`; row = (S - CTRV), ticks >= 1.
  GT from a scene's ego log by the v9 builder's own functions (`scene_rc_all`): bit-identical to the join's `rc` / `rc_ok` on 6 clips, 849 cells not reached (NaN path). Pin vs `t1hd_read.analyze_run` on the banked DAGOACC_s0 heads ticks (2399 windows): S and CTRV errors,
  kappa and gt_ok bit-identical; pooled -1.7428 / -4.5331 / -9.5624 with the banked CIs, n 2321 / 2313 / 2290. 5 mutations RED.
* **Bootstrap.** `x_contrasts` (X1, X2, BLIND), `eid` = scene, B 2000 seed 0 via the reader's `paired`; pairing guard on (scene, f0). Synthetic 42 launch + 20 guard scenes: X1 0.0488 (truth 0.05), X2 0.0304 (0.03), 42 clusters. 3 mutations RED.
* **outcome().** 19 named cases (boundaries incl.) + 1944-combination comparison with a restatement; 19 source mutations RED (every branch).
* **End to end.** Oracle records on 2 real downloaded scenes (car executes the logged track, S emits the v9 checkpoints): all 19 rows computed, error rows exactly 0, S-POS <= 0, counts 11 / 8 / 4 (route ends), 2 mutations RED.

## Policy (additive)
`forward_batch` extra now also holds `s_ckpt` [3,3] (S's checkpoints, S's tick frame) and `s_pose` [3] (float32 values). No existing key changed; G1 bit-for-bit tests green.

## Scorer input (for the entry script)
Per rollout record: `scene`, `f0`, `t0_step`, `cl`, `g`, `v0`, `track`, `extra[0..2]` with `s_ckpt`, `s_pose`, `a_s`, `step`. `SceneLog(gt_T, gt_ts_us, sequence_tracks)` per scene; `rows_for_records(recs, scenes)` gives the 19 rows; `pair_keys` / `pair_check` pair arms on (scene, f0).

## Open items (decisions for the card, before freeze)
1. BLIND sub-branch: card says "separated"; code needs BOTH seeds (b) or NEITHER (a); one seed only returns branch None plus a note. Card should state the rule.
2. UNDERPOWERED read as "no bound decides at either seed" (hi >= h and lo <= 0 at both); MIXED is every other case that is not 1 or 2. Card should say so.
3. S-POS matching runs on the 0.2 s grid anchored at the start tick; a scene's route ends within ~20 s, so S-POS@80 counts are small (n shown per row). Scorer must print n beside every S-POS delta.
4. The `empty` condition scores DK against the LOGGED agents (what the car sees changes, the scoring reference does not).

## Not done / blockers
None blocking. Not run: the rows on a real policy rollout (needs the entry script of agent A and box-2 GPU stage 1).
