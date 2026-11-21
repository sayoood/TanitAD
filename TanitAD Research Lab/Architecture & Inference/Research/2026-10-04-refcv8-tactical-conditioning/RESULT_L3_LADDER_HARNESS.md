# WP-B L3: the speed-only mode for V0, the VIS-1 × ego-strip fix, and the ladder harness (2026-10-05)

This package contains no ladder number. Every entry below is instrument or plumbing work and is stamped with its
evidence class.

## 1. L3: the repo delta, on tip 3cd48fd

L2 landed as a5c2a9b; all 37 of its files are verified equal to the tip blobs. L3 adds the following.

| file | what |
|---|---|
| `stack/scripts/refc_v3_train.py` (CRLF, owned) | **Speed-only mode.** Without `--refcv8`, the X3 speed flags are now legal. The v9 release is joined for its input column only, so the arm keeps its own labels as targets. The model-level speed generator is the same one V-R8 uses. `r8_spd_crc` is logged on every arm. **VIS-1 row re-key** (`vis1_rows_by_track`, §2). |
| `stack/tanitad/train/refcv8_train.py` (owned) | `speed_gen`, `speed_crc`, `speed_only_fwd`, all using the dedicated generator; the global stream is never consumed. |
| `stack/tanitad/refs/refc_v3.py` (owned) | The fed ceiling is honoured whenever `cfg.refcv8.speed_input` is set. This no longer depends on `r8_enabled`. |
| `stack/tanitad/eval/refcv7_loader.py` | The speed-only eval join (input column only). The eval clock is **unchanged**, per the MM ruling. |
| `stack/tests/test_refcv8_speed_input.py` | Covers: speed-only mode; byte-identity of the fed channel between V0 and V-R8 (same windows, same seed); CRC; the RC shuffle. |
| `stack/tests/test_refcv8_trainer_pin.py` | A speed-only argv is now valid. The refusal needle changed to `--max-speed-input-v6`. |
| `stack/tests/test_refcv8_vis1_ego_rekey.py` (NEW) | §2's pins. Values are literals. The RED arm reproduces the Thor refusal. |

## 2. MEASURED defect: WP-C I3's ego strip × refcv7 VIS-1. It would have killed every refcv8 arm.

* **Observed.** The Thor L1 full-size smoke (`/home/nvidia/refcv8_run/logs/smoke_L1_train.log`, 2026-10-05 ~00:58
  local) logged the cd rows at step 1. Then a DataLoader worker died with `[vis1] ... frame 75 row 19: an in-scope row with a 3-D label is
  ABSENT from the sidecar`. No checkpoint was written, and `p7_smoke_check` reads X4_rows FAIL.
* **Mechanism (MEASURED).**
  - That (clip, frame) appears in `refcv8_join_label_defects.json` among the ego-footprint frames. That clip has 116 listed frames, including 75.
  - `JoinDefectMasks.strip_ego_boxes` REMOVES the ego row at read time. Every later row then shifts up by one position.
  - The VIS-1 sidecar is keyed by the join's original row positions, so its guard refuses.
  - Scope: every arm that carries `--join-defect-masks` is exposed. That is R8_BUNDLE, which means V-R8 and every arm derived from it, plus the refcv8 smoke and launch argv. A crash comes on the first batch that touches one of the 693 listed frames.
  - V0 is not exposed (it carries no mask), and neither is the eval reader (unmasked by design).
* **Why no test saw it.** The two transforms are each correct and each pinned, but only in isolation. My CPU smokes on eval139
  never touched a listed frame (`n_ego_rows_removed 0`).
* **Fix.** It lives in my owned trainer; WP-C's file is untouched. On a LISTED frame only, the dataset rows are re-keyed to the join's original rows
  by TRACK ID, which is unique per frame. `vis1_block_for_rows` still re-checks the track id and the float32 centre. A
  different box is therefore still refused, and so is an unstored in-scope track (it gets a unique negative index). Pinned by 7 tests:
  - the re-key gives `[0, 2]`;
  - the stored `n_full` reads `[1000, 500]`;
  - RED arm: the position key on the same stripped rows gives the exact Thor refusal;
  - composition with the truncation permutation;
  - a moved centre is refused;
  - a duplicated sidecar track refuses the re-key.
* **Also found.** The smoke driver logged `trainer exit 0` for this crash because it read `$?` after a `$(date)` (the CLAUDE.md
  status-not-artifact trap). Fixed. The artifact check had already caught it.
* **Owed.** The L1 smoke re-runs after the landing: one lock slot of about 40 min. Its reads are still owed: the step-0 identity with emission off,
  G-LIVE for every `w_r8_*`, and s/step against 10.5.

## 3. The ladder harness (`code/ladder/`, package)

| file | role |
|---|---|
| `ladder_arms.py` | Builds the 16 arms (SPEC + A1) from refcv7-r101-s0's argv. It runs the one-variable audit on the ACTUAL argv. It also checks MM I-0 assertion 1, `no_v8_sidecar_anywhere`, and `pose_sync_on_every_arm` (MM 2026-10-05: the same sidecar on every arm, V0 included). |
| `ladder_eval.py` | One process per arm, covering all rows (`base, legal, rc_off, rc_shuf, vmax_off, vmax_shuf`) × sampler seeds. Window set: the EVAL-DIAG grid (K = 8 per episode). It runs `refcv7_loader.build_model` STRICT plus `build_eval_dataset`. The X10 clock comes from the arm's OWN config.json; a mismatch is refused. Per window it writes: plan, GT, fed speed and ceiling, tactical probabilities, the off-drivable rate (NavSim-DAC footprint, 0.5–4.0 s), 10 cm map inter/union for the headline cells, and v9 constraint predictions and targets. |
| `ladder_score.py` | Reports four families plus PLAN / SAFETY / PERCEPTION rows. Estimator: the paired episode-cluster bootstrap, B 2000, with seed-0 draws shared by every pass, as a ratio of sums. Also F, R, and every SPEC / A1 bar written as a literal. B-REG decides first. Refusals: without `I0.json` PASS, with mismatched window digests, and on an incomplete rung. `size-k` implements §4.2 and reproduces the SPEC's worked example (k = 51). |
| `ladder_i0.py` | Checks I0-a through I0-e, plus MM-1 (re-read from V0's config argv) and MM-2 (`r8_spd_crc` equal on every step both arms logged). |
| `ladder_chain.sh` | Thor. Order: S0 → N → arms → k → arms → I0 → score. Each job runs under the existing lock, with no queue jumping. A chunk is killed only right after a checkpoint, once the next save would pass 40 min. Disk at or above 22 GB is required before any job. After eval: a model-only `model_final.pt` with its md5; `ckpt.pt` and milestones are deleted. |
| `ladder_pull.sh` | Dev box. Pulls the finished arms over the LAN and verifies the md5 on the dev-box bytes. Only after that does it delete Thor's copy. |
| `ladder_stage_thor.sh` | Ships an ~18 MB minimal tree plus the harness, md5-checked on Thor. |
| `local_smoke.py` | A dev-box CPU smoke through the real `train()`. It uses a disjoint 8/4 split of eval139, because the in-run eval refuses an overlapping cache and has no override. |
| `test_ladder_score.py` | 18 tests, all passing. Targets are analytic or literal: a circle's curvature is 1/R; the 2 m/s speed gap; the §4.2 worked example. Mutation arms go RED: a regression arm that still drives makes the rung NOT QUOTABLE; a second difference in the argv; a live refcv8 flag on V0; V0 on the uncorrected clock. |

A DRY run of the chain on the dev box went S0 → all 16 arms → evals → I0. It stopped only at the scorer's Thor-only v9 train
path, which is expected (MEASURED, `scratchpad/dry_ladder/W/chain.log`).

**Stamps for every future ladder number:** TINY RUNG (lever claims only); OPEN-LOOP on logged eval139; nav, RC and labels
are ego-future derived; RC-A50 is PROVISIONAL.
