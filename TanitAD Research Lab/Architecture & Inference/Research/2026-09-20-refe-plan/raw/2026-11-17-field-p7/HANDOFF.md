# HANDOFF: field-loss arm (D-AUX-4) on host P7 — 0-GPU prep, 2026-11-17

**Hold:** `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/field_p7/`. Nothing staged, committed, landed or pushed. No existing file edited. Mirror tip used: `6b789b7e`.

## Ready
* `PREREG_AUX_FIELD_LOSS_P7_DRAFT.md`: new finalised draft for H-P7 (F `--w-field 2.0` vs C `0.0`, seed 0; replicates seed 1; both outcomes and next arms committed; four families + STRATEGIC). Old draft untouched.
* **Reach: `--w-field` is wired in the trainer P7 used** (`train.py` @ `181c7a3b`, `:2227` argparse → `:1767` term → `:1907-1908` online / `:1925-1926` bank site → `.backward()`), no patch needed. Real P7 weights + real navtrain row + real SDF, CPU: contribution non-zero on traj_head, dec, queries, scene_proj, reg_compress and the 96 LoRA tensors, exactly 0.0 on the score head; `--w-field 0.0` bit-identical to no flag (max abs diff 0.0); deleting `train.py:1926` turns the check RED. `REACHABILITY_FIELDP7.md`, `raw/reach/`.
* **w_field = 2.0** from the committed balance (r(2) = 0.394; K2/K2b clear), `WEIGHT_FIELDP7.md`.
* S0 gate on the train label store: PASSED (`PK/raw/2026-10-12-aux-label-store/s0_gate_post.json`).
* Cost ESTIMATED ~28.7 L40-h per arm at 2,000 steps (P5-C recipe): pair ~57, four arms ~115.

## Blocks the launch
1. **No gate profiles** for a P7-graft field arm (G1-G2, G4-G6 in `LAUNCH_GATE_GAPS_FIELDP7.md`): `refe_field*` are P2 clones; `p5c_refusals` would refuse the field flags. Needs a launch_gate/profile edit + chain + tests (an MM/implementation task, 0 GPU).
2. **Label-store pin (G3):** `--field-labels` hard-pins the full store (4a17db67, 103,139 keys); the pod's P7 store was re-finalised (df2ddd1b, 92,993 keys) and would be refused. Fix: ship the full store beside it (HF tar, 353,802,240 B) or change the loader.
3. **MM ratifications:** (a) w=2.0 was measured on a dev-box proxy pool (local rank-0 navtrain rows, not P7's pod-only bank; 6 deviations listed), frozen now, gate may only veto; (b) P5-C's C0/C0r as the field arms' C / C-seed (saves ~28.7 L40-h each if the argv diff holds); (c) P7's `config.json` was not read (pod): argv taken from `p7_pod.sh` @ `181c7a3b`, UNVERIFIED against it.
4. The online-labeller loss site was traced and covered by the 2026-10-13 tiny proof, not executed here (needs pod workers); the pod gate's multiplier-at-the-graph row covers it.
5. Pod busy: P5-C C0/C1 until ~2026-11-19.

## The exact PI ask
Spend **~57 L40-h** (F + C, 2,000 steps each) on the **L40 pod**, queued after P5-C; or **~29 L40-h** (F only) if C0 is accepted as the control. Replicates (F-seed, +~29; C-seed = C0r, which needs its own approval) before any lever claim. No box 2, no Thor. Dev box does the eval.

## Next steps (for a fresh agent)
Write profiles/chain/tests per the gaps file (dev box, CPU); after P5-C ends, run the gate on the real bank (re-measure r(w), report); ship the full label store; then launch.

## Deletions logged (own scratch)
`raw/reach/{F,C,N,M}.pt` and `raw/reach_attempt1_cleanfan_row/{F,C,N,M}.pt`, 76,074,675 B each (8 files, 608.6 MB), listed in `raw/DELETED_LOG.txt`; re-creatable by `code/run_reach_cpu.sh` (the verdicts are kept in `COMPARE.json`). Scratch `F:/tanitad_scratch/fieldp7/` (31 MB at the time) removal logged below.
Removed 2026-11-17: `F:/tanitad_scratch/fieldp7/` (972 files, 31 MB: the tip and 181c7a3b git exports, bank copies, run dirs, logs). Re-creatable: exports by `git archive` of `PK/refe` at the two commits; the two bank files are kept in `code/data/` (`bank_local_…` = the local rank-0 bank filtered to the yaw table's coverage, all 1,600 rows; `bank_one_…` = row 934). The `code/*.sh` scripts name the scratch paths they expect.

## MM addendum (2026-11-17 ~03:00 local)
- **P7's config.json was read and verified (closes MM ratification (c)).**
  - Source: pod `/workspace/data/refe_runs/p7_reserve_head/config.json`, sha256 `fce271fe7f14e798…`, 6,199 B, copied to `raw/p7_config/`.
  - Its recorded argv has the same 34 flags as `refe/p7_pod.sh` `ARGS` at `181c7a3b`. Every literal value is equal; only the `$`-variables are filled (paths, `--yaw-wrap off`, `--label-workers 13`, `--label-chunks 2`). `--w-field 0.0` and `--w-aux-drv 0.1` are named, and `warm_start` is null.
  - ⇒ The INHERITED recipe is now MEASURED against the run's own record.
- **(a) w_field = 2.0, MM-ratified pre-data.**
  - It follows the draft's gradient-norm procedure on a proxy pool, with the 6 deviations listed in `WEIGHT_FIELDP7.md`. No PDMS was read.
  - It is frozen. The launch gate may VETO it on P7's real bank but never re-pick it.
- **(b) C0 / C0r as C / C-seed:** this goes to the PI with the spend decision, valid only iff argv(F) == argv(C0) except `--w-field`. Proposed, not ruled.
- **Not landed:** `code/data/bank_local_targets_rank0.jsonl` (5.4 MB), landed gzipped instead; `raw/p7_config/p7_pod_181c7a3b.sh`, which is in git at 181c7a3b.
