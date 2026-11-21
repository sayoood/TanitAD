# ERRCUT_DEFS -- kit of card PREREG_V8F_PROOF_ADD96 (ERRCUT10K)

Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s), UE_s0 and UE_s1. The card file is not edited. Everything is in `F:/TanitAD-scratch/v8proof/errcut/`; the kit of card ADD95 in `gaprule/` was read, not edited.

## Dump field semantics (evidence for the join row)
* `cid` indexes the dump's OWN array `clips`: `for n, c in enumerate(clips)` at `closeread/closeread_dump.py:165`, `("cid", np.full(T, n))` at `:180`, `out["clips"] = np.asarray(clips)` at `:184`. The clip order of the dump is the evaluation cache order sorted by the join order (`score_clips = sorted(eman, key=...)`, `:226`), so it need not equal the join's order.
* `clip_index` is the row of that clip in the evaluation join's `clip` array: `row = {c: i for i, c in enumerate(join_clips)}` at `:163`, `("clip_index", np.full(T, row[c]))` at `:180`, `join_clips = je` = the `clip` array of the join given to the dump tool (`:225-226`, `:233`). It does NOT index `clips`.
* So the join row of a dump frame is `off[index of the clip NAME clips[cid] in the join's clip] + frame`. The kit uses the NAME (as the card says) and control e0 additionally requires `clip_index` to equal that index on every frame (`n_clip_index_mismatch == 0`).
* `lab_gap` is the join's lead gap where `lead_valid == 1`, finite and `0 <= gap <= 60`, else NaN (`label_clip`, `:84-108`, test at `:88`; same test as `LabelSource._ok`, `distrule/v8f_t1_eval_rule.py:88-90`). `valid` is the reader's window validity (frames 0 to 3 are not valid, `closeread/CLOSEREAD_DEFS.md` section 1); `gap` is the raw read r before any map.

## Files
| file | role |
|---|---|
| make_mix_joins.py | `--eval-join NPZ --dump-dir DIR --qmap-dir DIR --out-dir NEWDIR`; `--selftest`; writes `K100_s0 ... K12_s1.npz` and `MIX_JOINS_RECORD.json`; functions `e0_check(rec)`, `e1_check(rec)`; own copy of `qmap_apply` (numpy only) |
| make_errcut_queue.py -> pod_errcut10k_queue.sh | 8 rollouts; counted edits of `gaprule/pod_gaprule10k_queue.sh` (sha256 prefix 4873249749264b43 asserted); the `rollout()` function is not edited (asserted, and a test compares it byte for byte) |
| read_errcut10k.py | reader, stages `controls` and `names`; `--selftest` |
| podread_errcut10k.sh | pod read script, stages `controls` and `names`; sets `CUDA_VISIBLE_DEVICES=-1` itself |
| test_errcut.py | control e4 tests, synthetic only |

## Choices (the card is silent)
* Frame sets. "Valid label gap" = the join's `lead_valid == 1`, finite, `0 <= L <= 60`. "Reader read" = dump `valid` true AND raw gap finite AND the frame lies inside the join's rows of its clip. The 3,986 of row 154 is the first set; the mixed frames are the intersection. A frame with a valid label gap that is absent from the dump counts as "no reader read".
* Reader seed s: dump `closeread/dump/both/reader_s<s>_score.npz`, map `QMAP_LR16_s<s>_q50.json`; model seed s uses reader seed s. The record holds both file names and sha256, and the map folder; the reader passes `--qmap-dir $W/qmap_sweep` (the folder the banked reader-fed queue takes its map from, `$W/qmap_sweep/QMAP_LR16_s${k}_q50.json`, `gaprule/make_gaprule_queue.py` line of the removed start check) and e1 compares it.
* The mixed value is computed in float64 and stored as float32 (the dtype of the join); the deviation of (M - L) from k (g - L) is computed back from the SAVED file, with g recomputed from the dump and map of the right reader seed by code separate from the builder, so a fault in the builder cannot hide in the check.
* e0 asks more than the card, never less: besides the 1e-4 m bound on every dump frame with a label gap it requires the dump's recorded seed to equal the reader seed, no duplicate (clip, frame) keys, no clip name missing from the join, `clip_index` consistent, and no dump label frame that the join marks invalid.
* e1 asks: 8 arms with the right k and seed; deviation <= 1e-4 m; frames with a valid label gap and no reader read <= 5 percent of the valid label frames (per arm file); all other arrays equal to the evaluation join (value, dtype, shape) and no frame changed outside the mixed frames; no mixed value below 0; dtype of `lead_gap` kept; map file name and sha256 recorded.
* Error statistics (row 154 definitions, `kinclose/kinclose6.py:49-85`): `mean` and `sd` over all mixed frames; `rms_about_clip_mean` (FAST), `sd_of_clip_means` (CONST) and the change between neighbouring frames (STEP) over the clips with at least 5 mixed frames (`MIN_CLIP_FRAMES`, `frame_set`, `:43-46`); neighbours = same clip, frame difference 1. Row 154 defines STEP as a root mean square; the record carries both `step_rms` (the row 154 form) and `step_sd` (standard deviation of the change). The card's "standard deviation of the change" is read as `step_rms` in the printed section and both are stored.
* The record is bound to the evaluation join by sha256 and to every join file by sha256; the queue admission reads `e0.PASS` and `e1.PASS` from the record and the hashes; the reader recomputes e0 and e1 from the record fields (not from the two stored booleans) and checks the files on disk.
* Ladder: arms ordered by cut 1, 2, 4, 8 (K100, K50, K25, K12). NOT-MONOTONE when some arm is WITHIN-HALF-THE-GAP and an arm of a larger cut is not WITHIN-HALF-THE-GAP (OUTSIDE or NOT-DECIDED). `read_errcut10k.ladder_check` re-derives the name by a second route and the names stage refuses on a mismatch.
* The names stage refuses a controls file unless `controls_all_pass` is true, the six controls e0 to e5 are all present and each reads PASS, and a payload exists; it binds the result by the sha256 of the controls file.
* Controls e3: round(C(RR), 4) == 0.0954 and round(C(RLc), 4) == 0.0000. e5: the engine's grid read ALL_PASS and the strategic grid `stage_A_pass` and `stage_B_pass`; the STRATEGIC pair reads (each arm against RR and RLc) are computed in the controls stage as `gaprule/read_gaprule_strat.py` does and stored in the payload under `strategic`; the LONGITUDINAL, LATERAL and TACTICAL readings are under `families`. With e5 failed no strategic reading is written.
* Queue: root `$GP` = `${GR_ROOT:-/workspace/v8proof/errcut10k}`; check folder `${GR_CHECK:-/workspace/v8proof/gaprule10kb/check_j2}` (the passed record of card ADD95; wrapper hash cbf7063d867c9f4e); joins `${GR_JOINS:-$GP/joins}`; small folder `$GP/small_ec`. One process per model seed with markers `START_ERRCUT10K_s<k>.txt`, `ERRCUT10K_DONE_s<k>.txt`, `ERRCUT10K_STOP_s<k>.txt`, log `queue_errcut_s<k>.log`; with `GR_SEED=ALL` the done marker is `ERRCUT10K_DONE.txt` and the pod read accepts either form. The admission checks all eight join files whatever the seed scope.
* DRY=1 prints each command and needs no pod paths for the file checks it skips; the tests run it on the dev box and compare each DRY command with the model queue's DRY command for the label gap arm after removing the arm suffix and the join path.
* Not built: `lead_pytest.out` (the run of the tests that the lead makes on the dev box, control e4); the pod copies of the dump, map and join files.

## Order of use
1. dev box: `python make_mix_joins.py --selftest`; `python read_errcut10k.py --selftest`; pytest (command in the head of `test_errcut.py`), output to `errcut/lead_pytest.out`; `python make_errcut_queue.py` (only to rebuild the queue).
2. pod: `make_mix_joins.py --eval-join /workspace/v8proof/hierrep/pins/join_eval141.npz --dump-dir /workspace/v8proof/closeread/dump/both --qmap-dir /workspace/v8proof/integ10k/qmap_sweep --out-dir /workspace/v8proof/errcut10k/joins` (one run; refuses an existing folder; last line `ZZ MIX-JOINS e0 PASS e1 PASS`).
3. pod: copy the kit to `$GP/kit`; `DRY=1 bash pod_errcut10k_queue.sh`; then `GR_SEED=0 bash pod_errcut10k_queue.sh` and `GR_SEED=1 ...` (or ALL).
4. pod read: `bash podread_errcut10k.sh controls`, then `bash podread_errcut10k.sh names`.

## Note N1 of the card (applied by the lead with apply_note_n1.py before any join was built)
The clause of e1 that no mixed value is below 0 is a recorded count: `n_mixed_below_0` per arm and reader seed in the record and in the reader's payload. Such a value is left as it is; the rule's label source takes a gap as a lead only from 0 to 60 m, so the rule sees no lead on that frame.
