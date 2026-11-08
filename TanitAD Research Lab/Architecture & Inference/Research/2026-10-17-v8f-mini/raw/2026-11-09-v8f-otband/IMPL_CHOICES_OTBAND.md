# IMPL_CHOICES_OTBAND (stream B, 2026-11-08): what was built, the choices made, and what is unverified

Base: commit 1faa1a068f9f46dcb9d230c6b05efc20dce2fc1c (stack + taniteval). Deliverable: `otband.diff` (sha256 in `otband.diff.sha256`). Files changed: `stack/tanitad/models/v8f_mini.py`, `stack/scripts/v8f_mini_train.py`, `stack/tanitad/data/v8f_mini_data.py`; new: `stack/tests/test_v8f_otband.py`, `stack/tests/v8f_mini_tip_snapshot_otband.txt`. Nothing is trained, committed or pushed.

## 1. Design after the PI ruling (replaces the first design: no tactical trajectory head, no hand-over state, no continuity term)
S, T and O all read the measured ego state; no layer reads a state planned by another. T emits goals and tactical constraints that condition O through `e_code` (as built). O keeps ONE decoder with 30 knots. The band split exists only in the loss and the metrics.

## 2. Levers (all default off; off is bit-equal to the base, pinned against a frozen snapshot of the base model module, sha256 a6546109afa3297216b44431f9d69a2b3e4d355a54917e6e5f8ca50710e5108d)
| flag | meaning | choice |
|---|---|---|
| `--w-op W` | loss on knots 1-10 (0.2-2.0 s) | ADE/3 + 0.5 x FDE at knot 10 / 6 |
| `--w-tac-traj W` | loss on knots 11-30 (2.2-6.0 s) | ADE/3 + 0.5 x FDE at knot 30 / 6 |
| `--w-traj` | the 30-knot term | must be 0 when a band weight is on (else the knots count twice); the gate refuses |
| `--t-stok-dir D`, `--t-stok-shuffle SEED` | detached strategic token into T; shuffle control | section 5 |
| `--freeze S T adapter cond` | freezes groups (any subset) | section 4 |
| `--v10-md5 MD5` | accepts a v10 label release by md5 | section 6 |
| `--detach-upper` (existing, default False) | O's loss puts 0.0 gradient on T and S | named in EVERY arm argv (coordinator instruction); M2b ran without it |
Band scales (ESTIMATED choice): the ADE term is divided by 3 and the FDE term by 6 (FDE scale `BAND_FDE_SCALE`), so with weights 10/30 and 20/30 the ADE part equals the 30-knot ADE weighting. Band terms are computed only if a band weight is above 0 (a weight of 0 is not in the total and not computed).

## 3. Gate (G-DVB / G-LIVE)
Every new flag has a `FLAG_CHECKS` entry and a `bad_*` clause that compares the declared value with the BUILT model or data object (weights vs config, token dim vs built `e_stok` and store, shuffle derangement, `stok_attached`, freeze readback, v10 md5). `bad_oband` also refuses a band weight together with `--w-traj`, and `--wm-only` with a lever. `oband_probe` runs in the real caller on the real batch: the band contribution equals exactly its weight; with `detach_upper` O's op term gives exactly 0.0 gradient on S, T and `e_stok` while O's head receives gradient > 0; code edge and token edge change O's / T's outputs. New parameters are in `trainable_groups` ("STOK") so the dead-parameter check covers them. There is NO `launch_gate.py` profile for the mini trainer (AUDIT_MODEL F12): the supervisor cannot demand a PASS token for it. Named blocker, not built here.

## 4. Freeze
Groups S, T, adapter, cond (module sets in `FREEZE_GROUPS`). The arms freeze `S adapter cond`: S's output stays M2b's; `adapter` and `cond` are shared embeddings that would otherwise move S's input and O's conditioning; T and O train. A freeze identity test shows frozen weights equal the init checkpoint bit for bit after training steps. UNVERIFIED on the real cache: only the synthetic world was trained.

## 5. Strategic token (interface assumptions: UNVERIFIED until stream C names its dump)
* Directory `--t-stok-dir`: `<clip>.npz` with `stok` [T, D] and optional `ok` [T], plus `stok_meta.json` with `stok_dim`. A row count other than the clip's T is an error.
* The token is read at S's tick (sample-and-hold, `_at(b["stok"], last_s)`) and appended as one extra conditioning token to `run_t` through `e_stok = Linear(D+1, d)` (the +1 is the `ok` flag); detached; no re-expression into another frame (the token is assumed to be in the clip's cache frame).
* Shuffle control: `--t-stok-shuffle SEED` at training or `D.attach_stok(ds, dir, seed)` at eval = a seeded cyclic derangement of clips (no clip keeps its own token; tested).
* A token checkpoint must be fine-tuned with the same `stok_dim` (`INIT_CKPT_MONOTONE`); an off checkpoint (M2b) loads into a token build, `e_stok.*` being a fresh key (`NEW_LEVER_KEY_PREFIXES`).
* INTEGRATION ITEM for the Master Mind: `STOK_KEYS` ("stok", "stok_ok") are deliberately NOT in `INFERENCE_INPUT_KEYS` (pinned by other tests). `v8f_t1_eval.py` (its `MODEL_KEYS` and the dagger / ego key sets) must carry these keys when `stok_dim > 0`, or a token model cannot be evaluated there.

## 6. v10 md5
`ACCEPTED_V10_TRAIN_MD5` (in `v8f_mini_data.py`) is EMPTY until stream A names the release; `register_v10_md5` and `v10_md5_problems` are the mechanism. `--v10-md5` must equal `--v9-train-md5` and the md5 the loader hashed. The base release is the 4,572-clip one; nothing in the code fixes a clip count. The v10 schema is UNVERIFIED.

## 7. Top-K (eval only, built, cheap)
`forward_topk(batch, K)`: O is decoded once per top-K joint (lat, lon) manoeuvre code of T (`topk_codes`: one-hot lat and lon of the K most probable pairs, the rest of the code unchanged), returning K plans and their probabilities. No parameter, no training effect. Output of R-OUT ("trajectories, plural"); a trained mixture form is not built.

## 8. Retrain argvs (`argvs_otband.txt` / `.json`, from `make_argvs.py`; all 7 parse with the trainer's parser; box-2 paths from IMPL_CHOICES_LADDER section 2)
| run | init | seed | levers |
|---|---|---|---|
| OB-BAND-TOK_s0 / _s1 | M2b_s0 / _s1 | 0 / 1 | bands, token, detach, freeze |
| OB-BAND_s0 / _s1 | M2b_s0 / _s1 | 0 / 1 | bands, detach, freeze |
| OB-CTL_s0 / _s1 | M2b_s0 / _s1 | 0 / 1 | detach, freeze (today's loss) |
| OB-BAND_rep | M2b_s0 | 2 | as OB-BAND |
Common: `--freeze S adapter cond --detach-upper --steps-mult 2` (ESTIMATED 2 epochs; run time not measured). Band arms: `--w-traj 0 --w-op 0.3333333 --w-tac-traj 0.6666667`. The token directory in the OB-BAND-TOK argvs is the placeholder `D:\tanitad\data\stok\STREAM_C_DIR_UNVERIFIED`; replace it with stream C's directory before launch.

## 9. Evidence (MEASURED on the CPU synthetic world and the pinned suites, T0 / unit level; no real-data run)
Measured on CPU (CUDA hidden), synthetic world and the pinned suites; unit level, no real-data run.
* `tests/test_v8f_otband.py`: 46 passed (32.6 s on the final tree; log `logs/otband_tests_final.log`).
* Mutations (`mutate_otband.py`, `logs/mutations.json`): 15 of 15 RED, including the removal of the code detach (a) and of the checkpoint detach (b) with `--detach-upper` on. Mutation n (top-K ignores rank) first read GREEN; the test was too weak, a rank test (`test_topk_codes_follow_the_joint_rank`) was added and n is RED. j, k and o are RED by fixture ERRORS (the trained-arm fixture cannot build), not by an assertion failure: they catch the regression, with a blunter signal.
* Regression: the 52 test files that name v8f_mini* were run on this tree (`logs/regress_summary.json`); 11 files have failures, all also failing on the base tree (export-limited or pre-existing; the base tree has the same failures) except one, `test_v8f_flat::test_the_cut_block_is_exactly_the_text...`, which my first `_forward` edit broke (the pinned statement `tout = self.run_t(...)` was merged into a conditional expression). Fixed by restoring the statement in an if/else; the file re-run reads 1 failed, 40 passed, with no failure only on this tree (`logs/regress_flat.out`). The other 41 files pass. Files that were not re-run after the fix: none changed by it apart from this one.
* The unpinned remainder: the 11 base-tree failures are not mine and were not investigated.

## 10. Open items
1. Stream C's `stok` dump and `stok_meta.json` (blocks OB-BAND-TOK only).
2. `v8f_t1_eval.py` key sets for `stok` (section 5).
3. Stream A's v10 md5 into `ACCEPTED_V10_TRAIN_MD5`.
4. A `launch_gate.py` profile for the mini trainer (AUDIT F12).
5. Real-data smoke of the freeze and the probes on box 2 (only synthetic here).
