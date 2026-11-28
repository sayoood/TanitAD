# L_field wired into `refe/train.py` and the launch gate: patches, proofs, weight-rule tool (2026-10-13)

**Status: DELIVERED AS PATCHES, PROVEN ON A COPY (CPU, tiny scale). No original file was edited** (`cmp` of `train.py`, `gate_checks.py`, `launch_gate.py`, `test_launch_gate.py` against the pristine copy: identical at the end).
Evidence classes: **MEASURED** = ours, this session, the artifact named. Nothing below is INHERITED unless it says so.

## ⛔ Integration (for the Master Mind)

1. **Apply AFTER P6 ships.** Any change under `refe/` changes the code-tree digest a PASS token binds (`gate_token.tree_manifest`); P6's token would stop verifying. This is also why nothing was edited in place.
2. **Apply the two patches TOGETHER**, from the package root, `train` then `gate`:
   `patch -p1 < raw/2026-10-13-field-loss-wiring/PATCH_train.diff && patch -p1 < raw/2026-10-13-field-loss-wiring/PATCH_gate.diff`
   `PATCH_train` alone leaves `test_launch_gate.py` red on purpose (S5 pins `DVB_KINDS == parser dests`; 60 != 55: a new trainer flag without a G-DVB entry is refused by the gate itself). `PATCH_gate` alone makes `refe_field` refuse with `DECLARED, NOT BUILT`.
3. **Land `weight_rule.py` as `refe/weight_rule.py`** (new file, no patch; `--w-field`'s help names that path) and regenerate `SHIP_MANIFEST.json` (`launch_gate.py ship-manifest`).
4. `patch -p1 --dry-run` on the CURRENT originals: both apply cleanly (`patch_dry_run.txt`, with the md5 of the four originals).

## Deliverable manifest (all in `PKG/raw/2026-10-13-field-loss-wiring/`; the working copy `C:/Users/Admin/sx_fieldwire/` also holds the patched files, but the diffs here regenerate them exactly: ONE place = this directory)

| file | what |
|---|---|
| `PATCH_train.diff` | `refe/train.py`, 265 lines |
| `PATCH_gate.diff` | `refe/gate_checks.py`, `refe/launch_gate.py`, `refe/test_launch_gate.py`, 504 lines |
| `test_field_wiring.py` | the proofs; applies both patches to a temp copy of the current originals itself; `python test_field_wiring.py` (about 8-12 min CPU) |
| `weight_rule.py` | the committed weight rule as a tool (`decide()` is pure) |
| `test_field_wiring.out` | the final run: `ZZTEST_PASS 116` |
| `launch_gate_static_BEFORE.out` / `_AFTER.out`, `test_field_loss_on_patched_copy.out` | the ZZTEST lines below |
| `other_suites_orig_vs_patched.txt` | every other test that imports the trainer, run on both trees |
| `weight_rule_tiny_selftest.json` | the tool's output on the tiny bank (NOT A DECISION) |
| `real_bank_label_coverage.out` | label coverage of the real rank-0 bank by the real store |
| `patch_dry_run.txt` | `patch -p1 --dry-run` on the current originals |

## ZZTEST lines (MEASURED, CPU, this session)

| command | before (originals) | after (patched) |
|---|---|---|
| `test_launch_gate.py --static --no-banked` | `ZZLG_STATIC_PASS 538` | `ZZLG_STATIC_PASS 623` (+85) |
| `test_launch_gate.py --no-banked` (FULL: the real trainer in-process, every gate check and arm) | `ZZLG_FULL_PASS 569` | `ZZLG_FULL_PASS 654` |
| `test_field_loss.py` | `ZZTEST_PASS 42` | `ZZTEST_PASS 42` (module untouched) |
| `test_field_wiring.py` (new) | n/a | `ZZTEST_PASS 116` |
| `test_aux_s1_wiring.py` | `ZZTEST_PASS 20` | `ZZTEST_PASS 20` |
| `test_p6_levers.py --fast` | `ZZP6_PASS 123` | `ZZP6_PASS 123` |
| `test_p6_chain.py` | `ZZP6CHAIN_PASS 204` | `ZZP6CHAIN_PASS 204` |
| `test_rl_tlane_label.py` | `ZZTEST_PASS 36` | `ZZTEST_PASS 36` |
| `test_rl_fan_dump.py`, `test_rls1_chain.py` | FAIL (rc 1) | FAIL (rc 1), **identical**: my copy lacks `run_calibration` / the `eval/` tree. Pre-existing for the copy, not caused by the patch. UNVERIFIED on the full package |

(The FULL gate test needs `SHIP_MANIFEST.json` beside `refe/`; the copy got it from the package root. Without it `test_p2r_chain.py` fails in G-SUITE on the ORIGINAL too.)

## 1. `PATCH_train.diff`: what changed

| plan item | done |
|---|---|
| import | `from field_loss import field_loss`; the label reader (`aux_drivable_head`) is imported lazily, only by a `TargetBank` given `--field-labels` |
| flags | `--w-field` (0.0), `--field-margin` (0.25), `--field-interp` (`5hz`\|`navsim2hz`, `5hz`), `--field-labels` (None), `--field-shuffle-seed` (None) |
| resume identity | the five in `IDENTITY_ARGS`, defaults in `IDENTITY_DEFAULTS` (an old checkpoint resumes); plus `field_store_sha256` (the manifest sha256, i.e. the store's CONTENT) in `run_identity`, **only on a field run** and compared in `_identity_diff` |
| loader | `TargetBank(field_labels=, field_shuffle_seed=)`; with labels the item gets `(sdf int8 [160,128], valid bool)` as its LAST TWO elements (after the online row index); without labels the tuple and the collation are unchanged |
| batch unpack | field elements by negative index; `ridx` is `batch[9]` iff the batch is longer than 9 (+2 with labels) |
| the term | `field_loss(traj, sdf, margin, valid, interp, return_parts=True)` after `wta_loss`, **only when `--w-field > 0`** |
| loss sites | non-online `loss = (l_traj + l_score + w*l_field) / accum`; online deferred backward likewise (`l_field`/`fparts` ride in `cur`). The default path executes the same operations as before (the expression is split, not changed) |
| telemetry | `field_L` (unweighted), `field_w`, `field_viol`, `field_inext` on the logged rows; one `field` event (store sha256, key count, rows, rows with a label, shuffle seed) at start |
| refusals | `--w-field` not finite/negative; `--w-field > 0` without `--field-labels`; `--field-shuffle-seed` without labels; a store that is not the pinned one (`StoreRefused` -> `SystemExit` naming the flag) |

### Deviations from the plan (each deliberate)

1. **No `labels.i8` memmap.** The store is per-log shards read by `aux_drivable_head.AuxLabelSource` (pinned sha256 + key count + shard hashes + navtest-leak refusal); the task said reuse it. Key = `(log_name, token)` only (goal twins share a frame). The reader keeps every shard it touches per loader process (up to 351 MB compressed over a full epoch): **RAM per worker UNMEASURED**.
2. **A frame whose key is not in the store under `--field-shuffle-seed`** reads zeros + `valid = False` instead of raising `KeyError` from inside a worker (a crash loop under a supervisor). `AuxLabelSource.label` alone would raise.
3. `--w-field 0.0` **with** `--field-labels` is allowed (the matched control may carry the same argv as F but for the weight).

## 2. What the tests prove (`test_field_wiring.py`; setup self-made: 6-scene bank, 2 rank-1 twins, a REAL store written by `aux_label_store.write_shard`, SDF = a 2.5 m corridor around each scene's own logged path, scene `log5` has no label; real REFe ViT-S, 1 camera, `--weights none`, synthetic images, CPU, batch 2, accum 2, 2 optimiser steps)

**(a) Default path bit-identical.** Unpatched vs patched with no field flag, with `--w-field 0.0`, and with `--w-field 0.0 --field-labels` (label delivered, term off): per-micro-batch `l_traj`, the gradients before AND after the clip at every step, every final parameter, the logged rows, all by sha256 of dtype+shape+bytes (stricter than `torch.equal`). On the fixed-scorer path and the `--labeller online` (deferred backward) path. The loader items (8 items, the collated batch) are byte-identical; with labels they carry exactly two more, and the online row index stays at position 9. Halt (UNPATCHED) + resume (PATCHED) of an old checkpoint equals the uninterrupted unpatched run; a resume that changes `--w-field` is refused (`--w-field: 1.0 -> 2.0`); halt + resume of a FIELD run equals the uninterrupted field run; `--workers 2` equals `--workers 0` with the term on (the label source is pickled to loader workers).
*Controls:* unpatched vs unpatched is identical (the runs are deterministic, so any difference is the patch's); `--w-field 1.0` DIFFERS in parameters and gradients (the comparison can see a change).

**(b) Gradient reach.** L_field-only backward through the real model: non-zero on `traj_head`, `dec`, `queries`, `scene_proj`, `reg_compress`, `backbone.lora`; **`grad is None` on every parameter** of `score_q_mlp`, `score_dec`, `score_head` (no autograd path, stronger than zero); weight 0 reads exactly 0.0 on every group. Inside the real trainer (trajectory loss zeroed, `--score-w 0`): the same non-zero groups, `--w-field 0` exactly 0.0 everywhere and the term is not even called; the gradient arriving at the term is `w/accum` on all 4 micro-batches (a hook on the term's output); logged `field_L` equals `field_loss` recomputed from the recorded inputs, bitwise.

**(c) Mutations that go RED** (one line of the patched `train.py` edited in a copy; the proofs raise): MEASURED messages in `test_field_wiring.out`.

| mutation | red because |
|---|---|
| multiplier `0.5*w` (fixed and online sites) | the multiplier arriving at L_field is 2.0 (x accum), not w = 4.0 |
| term computed, not added (fixed, online) | the gradient reached L_field on 0 of 4 micro-batches |
| term added detached | same, 0 of 4 |
| label y-flipped in the loader | alignment 0.550 < 0.95 (`aux_label_store.target_probe` at the hand-off) |
| valid mask dropped | the valid mask was not passed |
| margin hard-coded 0.5 / interp ignored | `field_loss` received margin 0.5 / interp navsim2hz |

The clean tree passes the same proofs first, on both paths, so a RED is the mutation's.

## 3. `PATCH_gate.diff`

* `gate_checks.py`: the five flags in `DVB_KINDS`; `instrumented_run` wraps `train.field_loss` and records the margin / interp / valid it **received** and the **multiplier measured at the graph** (a hook on the term's output reads d(loss)/d(L_field); `w/accum` when added once per micro-batch, never fires when not added); `field_dvb_rows` (also the two `TargetBank` rows for every profile; the store's pinned sha256 / key count / row coverage >= 99.9 % / raster constants against `field_loss`'s); `field_reach` + `judge_field_reach` (G-LIVE: the reach probe, scorer groups `None`, weight-0 control exactly 0.0); arms `field_weight_off`, `field_term_detached` (G-DVB), `field_weight_zero` (G-LIVE).
* `launch_gate.py`: profiles **`refe_field`** (clone of `refe_p2`; `--w-field` named and **> 0**, `--field-margin 0.25`, `--field-interp 5hz`, `--field-labels`; live term `field_L`) and **`refe_field_ctrl`** (`--w-field` named and **exactly 0.0**, otherwise identical; F and C differ by that token alone); both put `test_field_loss.py` in their suite and `--field-labels` in `data_flags`; `--field-shuffle-seed` forbidden until a dated amendment adds that arm.
* `test_launch_gate.py`: `field_static()` (+85 checks, S5 count 55 -> 60) and an admissibility `grep` equivalent (`model.py`, `planner.py` contain neither `field_loss` nor `sdf`).
* End to end (part d, the real trainer, tiny scale): G-DVB and G-LIVE PASS clean (13 field rows, all ok); `field_weight_off` -> G-DVB FAIL ("multiplier measured at the graph"); `field_term_detached` -> G-DVB FAIL ("0 of 4"); `field_weight_zero` -> G-LIVE FAIL ("EXACTLY zero on traj_head"); `--w-field 0.0` against a profile declaring `field_L` -> G-LIVE FAIL ("ABSENT"); the registered 99.9 % coverage refuses the tiny bank (7 of 8 rows labelled): a built-in negative control. `--w-field` absent from the argv of an arm that means it: refused by the profile (static).

### Deviations from the plan

1. `test_field_loss.py` is in the **`refe_field`/`refe_field_ctrl` suites, not the base `SUITE`**: the base is shared with `refe_p2`/`refe_p6`/`refe_rehearsal`, whose gates should not change under an in-flight launch. One-line change if you want it global.
2. **`r(w)` is not computed by the gate** (the plan listed it under G-LIVE); `weight_rule.py` computes it. The gate reports label coverage, `field_inext`, `field_viol`.
3. The plan's G-LIVE **alignment / leakage** check is **not wired** (it is shared with the head's S0 gate). What exists: the reader refuses a manifest reporting a navtest intersection; this test's alignment check is a test-time instance on the tiny bank.
4. **The pod chain for arms F/C (a `p2r_pod.sh` analogue that names the flags, runs the weight rule, then the gate) does not exist.** The profile and the tool are the parts.

## 4. `weight_rule.py`

Runs the REAL `train.run(a)` on the arm's argv until the optimiser is built (captures the model and the `TargetBank` the trainer itself built, warm start and label store included; writes nothing; ignores `--resume` on purpose), draws 8 micro-batches (`manual_seed(20261012)`, `randperm`, without replacement), takes the gradient of `l_traj` and of `L_field` (w = 1) on `traj_head`, pools over the 8, and applies the rule: smallest w in {1/4..64} with r(w) >= 0.25; K2 if r(64) < 0.25; K2b if r(1/4) > 1.0; exit 0 / 5 / 4.
Proved on the tiny setup: `decide()` at its boundaries (r exactly 0.25 selects; r(1/4) exactly 1.0 is not K2b; NaN/negative -> ERROR); the seeded draw recomputed independently; reproducible; r exactly linear in w; **w * grad == grad of (w * L_field) bitwise at all 9 candidates**; no label store / too few rows -> refused; an all-road store -> L_field exactly 0 -> K2, exit 5, no weight. On the tiny bank: `r(1) = 0.471`, rule SELECT `w = 1.0` (`weight_rule_tiny_selftest.json`, stamped `NOT_A_DECISION`).

## UNVERIFIED / not done

* **Nothing ran on a GPU**, no A40 timing (K3 unread), no bf16 autocast run with the term (it sits outside autocast like `wta_loss`, but untested here). **Real images / real bank not on the dev box**: every trainer test is synthetic, tiny, CPU. The real store's pinned sha256/key count were not re-hashed shard by shard here.
* **Interpretation flagged for you:** the prereg's `r(w)` does not say whether the 8 micro-batch gradients are pooled before the norms. I pooled them (what an accumulation window gives the optimiser) and report the per-micro-batch ratios (min/median/max) as diagnostics. If the rule is meant per micro-batch (e.g. a median), that is one line in `measure()`.
* **Coverage, MEASURED on the only real bank on this box** (`D:/Projects/TanitAD-artifacts/_rlfan_work/bank_full/targets_rank0.jsonl`, 103,037 rank-0 rows): 103,037 of 103,037 have a stored SDF (100 %). The launch bank may differ; the gate measures it. Prereg blank says "expect 103,288 frames"; the pinned store has **103,139** keys (103,288 is the yaml token count, 149 absent): the blank should say 103,139.
* `--grow` with field labels: `_rebuild` passes the arguments, **not tested**. `--workers` > 0 tested at 2 on Windows spawn only.
* G-CKPT / G-EVAL with field flags at pod scale not run (tiny-scale G-CKPT and G-EVAL passed inside the FULL gate test on the P2 rehearsal fixture, which does not use field flags; halt + resume with field flags is proved in part (a)).
* `test_rl_fan_dump.py`, `test_rls1_chain.py`: cannot run in my copy (see table); equal before/after.
* Admissibility of `train.py` importing the label path: `model.py` and `planner.py` still contain 0 occurrences of `field_loss` / `sdf` (checked, and pinned by the new static test).
