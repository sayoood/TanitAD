# TOKCLOSE_DEFS (card PREREG_V8F_PROOF_ADD81): definitions, fixed hyperparameters, assumptions, commands

Tier T0 (reads on logged frames; no rollout). Written before any fit on data. Every hyperparameter below is fixed here with one line of reason and is not searched.
Tools: `tokclose_fit.py` (subcommands `fit`, `score`, `--selftest`), `run_tokclose.sh`, `test_tokclose.py`, `run_mutations_tc.py`.
Modules imported unchanged: `closeread_dump.py` (label function `label_clip`, `reader_lists`, `heldout_rule`), `closeread_fit.py` (`build_rows`, `episode_sums`, `boot_totals`, `pair_summary`, `cell_summary`, `pearson`, `split_by_clip`), `closest.py` (`ClosingEstimator`), `tok_reader.py` (`TokReader`, `frame_embed`).

## 1. Definitions of the card with their source

| item | definition | source |
|---|---|---|
| label closing speed | least-squares slope of the label gap over the trailing run of at most 5 frames (step bound 3 m, run of at least 3), the function whose full-run frames passed control c2 of card ADD78; the SAME function (`closeread_dump.label_clip`) labels fit and score frames | card ADD81 section 2; `closeread/closeread_dump.py:84-108` |
| frame period | 0.2 s | `closeread_dump.DT_S` |
| scored frame of card ADD78 for history H | frame f of an evaluation clip with a label that has a lead and a run of at least 3 frames (finite closing speed), and the H reads on the frames f-H+1..f all present; a read at frame j exists when frames j-4..j exist, so f >= H+3 | `closeread_fit.build_rows` (`closeread_fit.py:86-100`), `closeread_dump.py:134-141` |
| eligible frame of this tool (fit and score) | f >= H+3, label present, run >= 3, finite closing speed; test `test_rows_equal_build_rows` proves the set equals `build_rows` on a synthetic dump | `tokclose_fit.eligible_frames` |
| history H | 10 and 25 frames (2 s and 5 s) | card section 2 |
| head input | the head output at frame f uses the embeddings of frames f-H+1..f of the same clip and nothing else (window gather, a recurrent layer over H steps only; control t3 measures it) | card section 2 and control t3 |
| E arms | frozen frame embedding (64 values) of the lr16 reader of seed k: `TokReader.frame_embed` (per-frame part of the reader: layer norm, 1,024 to 128 with GELU, 128 to 16 per token, 40 tokens flattened to 640, 640 to 64) | `integ10k/tok_reader.py:102-108` |
| T arms | the same per-frame map with fresh weights (layer norm, 1,024 to 128, GELU, 128 to 16 per token, flatten 640, 640 to 64), trained end to end with the head | card section 2 ("token projection of the reader's form"); width 16 is the width of the reader (arm name TK16) |
| reader seed k = head seed k | seeds 0 and 1; E arm seed k reads the frozen reader TK16_s<k> | card section 2 |
| fit clips of E arms | the 432 clips held out of the reader's training: `int(sha256(id)[:8],16) % 10 == 0` after the trainer's `--exclude 540:600` slice, recomputed from the manifest and checked against `train_meta.json` (count and hash) | `closeread_dump.reader_lists` (`closeread_dump.py:61-80`) |
| training clips of T arms | the 4,080 clips of the same recomputation (`train_ids`) | same |
| evaluation clips | the 141 clips of the evaluation cache, join `join_eval141.npz` | `closeread_dump.py:223-226` |
| ADD78 cells for the comparison | N_H10 (H 10) and L_H25 (H 25): the saved estimators `closest_N10_s<k>.npz`, `closest_L25_s<k>.npz` applied by `ClosingEstimator.apply` to the X rows of `build_rows` | card section 3; `closeread/closest.py` |
| episode unit | the clip, through the episode map of card ADD78 (`episode_map.json`, 141 ids) | `closeread/CLOSEREAD_DEFS.md` addendum |

## 2. Fixed hyperparameters (one line of reason each)

| item | value | reason |
|---|---|---|
| temporal head | LayerNorm(64), GRU(64 to 128, one layer, batch first) over the H embeddings, last state, Linear(128, 64), ReLU, Linear(64, 1) | the card's "one recurrent layer of 128 units, then two layers to one output"; the layer norm puts frozen (E) and learned (T) embeddings on one scale without tuning |
| output | closing speed in m/s, no output scaling | the label unit; the same unit as the cells of ADD78 |
| loss | Huber, delta 1 m/s, mean over labelled eligible frames | the loss of the nets of card ADD78 (`closeread_fit.train_mlp`) |
| optimiser | Adam, lr 1e-3, no weight decay, constant lr, gradient norm clip 1.0 | the ADD78 nets used Adam 1e-3; a constant rate with early stopping needs no schedule; the clip guards the recurrent layer |
| batch | 8 clips per step; every eligible frame of those clips is a row | a clip is 101 frames, so a step holds a few hundred windows; clips are read once per step (disk bound for T) |
| order | clips shuffled each epoch by a generator seeded 1000 + seed; model weights by `torch.manual_seed(seed)` before construction | repeatable, per seed |
| E family stopping | evaluation of the stopping set at step 0 and after every epoch (345 clips, 44 steps per epoch), at most 100 epochs, patience 10 evaluations, state of the best evaluation kept | small fit set (432 clips), cheap evaluation |
| T family stopping | evaluation at step 0 and after every 510 steps (one pass over 4,080 clips), at most 8 evaluations (4,080 steps), patience 2 evaluations, state of the best evaluation kept | a pass costs about 17 GB of reads; eight passes bound the run time |
| E split | `closeread_fit.split_by_clip(clips, 0.2, 0)`: one fifth of the 432 held-out clips (ceil(0.2 x 432) = 87) stop the fit, 345 fit it | card: "one fifth", by clip; the generator of the split is the one of card ADD78 with frac 0.2 |
| T stopping set | all 432 held-out clips (none of them trained on, none of the 141) | card section 2 |
| stopping point | the best evaluation after at least one step; `loss_before` is the stopping-set loss at step 0, `loss_at_stop` the loss at the kept state; control t2 asks `loss_at_stop < loss_before` | control t2 |
| planted arm (control t1) | T10, seed 0, the finite closing labels permuted among the labelled eligible rows, separately within the training rows and within the stopping rows, generator seed 20261120; mask unchanged | control t1 |
| bootstrap | the two-level bootstrap of card ADD78 (reader seeds, then evaluation episodes), B = 2,000, percentile 2.5 / 97.5, seeds 1000 + H (point) as in `closeread_fit.run_all` | the same estimator as the cells it is compared with |
| t3 | 6 evaluation clips x 3 frames per head, frames after t replaced by Gaussian noise of the token standard deviation, CPU, tolerance 1e-5 m/s | card control t3 |
| t1 pooled r | the planted fit is one seed: its r over the evaluation frames of H 10 (reader seed 0 rows), interval not needed, bar [-0.10, +0.10] | card control t1 |
| constant read | the constant 0 m/s (as `mae_zero` of card ADD78) and, beside, the constant equal to the mean of the label on the scored frames | card section 3 "mean absolute error of a constant read" |

## 3. Names (as implemented)

* `TOKCLOSE-<X>-READABLE` when the lower end of the pooled r interval is above 0, else `TOKCLOSE-<X>-NOT-READABLE`; X in E10, E25, T10, T25.
* `TOKCLOSE-<X>-BETTER-THAN-OUTPUTS` when the upper end of the pooled paired interval (absolute error of X minus the cell of the same H) is below 0 AND r of X is above r of the cell in both seeds; else `-NOT-BETTER`.
* Verdict `TOKCLOSE-A-TOKEN-HEAD-READS-MORE` (with the count of four) when at least one arm is BETTER, else `TOKCLOSE-NO-GAIN-FROM-TOKENS`.
* Controls t0 to t4 are computed before any arm value is opened; if one fails the result file and stdout carry no name and no arm value.

## 4. Assumptions about the pod that I could not verify (paths, names, shapes)

1. Token caches: `<cache>/train/manifest.jsonl` (field `clip`), `<cache>/train/clips/<id>.pt` with key `pool4x10`, shape `[101, 40, 1024]`, float8 storage (`.to(float16).float()` exact); eval cache the same under `eval/`. Source: `closeread_dump.py:155-159`, `CLOSEREAD_DEFS.md` section 3 and the addendum (checked by the lead).
2. Joins `join_train.npz`, `join_eval141.npz` with `clip`, `off`, `n`, `lead_gap`, `lead_valid`. Source: addendum of `CLOSEREAD_DEFS.md`.
3. `train_meta.json` with `heldout.{rule,n_heldout,n_trained,heldout_sha256}` and `--exclude 540:600` in `trainer_argv`. Source: same.
4. Reader checkpoints `/workspace/v8proof/integ10k/lr16/TK16_s<k>/ckpt_last.pt` (kind `tokprobe`, history 5, 40 tokens, W 16). Source: `pod_estclose10k_queue.sh` and `CLOSEREAD_DEFS.md` section 4.
5. Dumps of card ADD78 in `/workspace/v8proof/closeread/dump/both/reader_s<k>_score.npz` and `reader_s<k>_lists.json`; ADD78 result `/workspace/v8proof/closeread/read2/CLOSEREAD_RESULT.json` with cells `N_H10`, `L_H25` carrying `r_per_seed` (list over seeds 0, 1); estimators in `/workspace/v8proof/closeread/est/`; episode map `/workspace/v8proof/closeread/episode_map.json`. Source: `estclose_read/run_estsave.sh`, `closeread/pod/`.
6. The build folder `/workspace/v8proof/integ10k/build` holds `tok_reader.py`; the tool files of card ADD78 (`closeread_dump.py`, `closeread_fit.py`, `closest.py`) are copied to the tools folder of this card (the runner checks they exist). Python imports resolve from the tools folder, then `--build-dir`.
7. The stream of the reader's own training (windows of the trainer's dataset) is NOT reused: T arms read the 4,080 training clips whole from the cache (every clip once per epoch, all eligible frames of it). Difference from the reader's training stream, stated for the lead.
8. cgroup files, `nvidia-smi` exist as in `pod_estclose10k_queue.sh`; the pod GPU has at least 8 GB free for one fit.

## 5. Commands

* Local: `PYCAP_MB=3000 bash ops/pycap.sh tokclose/tokclose_fit.py --selftest`; `PYCAP_MB=3000 bash ops/pycap.sh -m pytest -x -q -p no:cacheprovider --basetemp <new folder> tokclose/test_tokclose.py`; `PYCAP_MB=3000 bash ops/pycap.sh tokclose/run_mutations_tc.py --tag NAME`.
* Pod: `DRY=1 bash run_tokclose.sh` prints every command; `bash run_tokclose.sh` runs the chain (markers `START_TOKCLOSE.txt`, `TOKCLOSE_DONE.txt`, `TOKCLOSE_STOP.txt` in `/workspace/v8proof/tokclose`).
* One fit by hand: `tokclose_fit.py fit --arm T10 --seed 0 --train-cache C --eval-cache E --join-train J1 --join-eval J2 --train-meta M --build-dir B --out-dir OUT [--reader CKPT] [--shuffle-closing-labels]`.
* Score: `tokclose_fit.py score --fit-dir OUT --dump-dir D --est-dir EST --add78-result R --episode-map MAP --eval-cache E --join-eval J2 --build-dir B --out-dir NEW`.

## 6. Differences from the card that the lead decides

1. Control t2 on the planted fit: the card says "for every fit" the held-out loss at the stopping point is below the loss before the first step. A head fitted on permuted labels has nothing to learn, so its stopping loss can sit above its starting loss by chance. The tool judges the loss pair on the eight real fits and the counts on all nine; the planted fit's loss pair is written to the control record and not judged. `score --judge-planted-loss` judges it too (the form of the card as written).
2. Control t1 pools over the one planted fit (reader seed 0 rows at H 10); the card says "pooled r" for a single fit.
3. The T arms read the 4,080 training clips whole from the cache, eight clips per step (all eligible frames of those clips), not the window stream of the reader's own trainer.
4. Hyperparameters can be overridden only by the environment variable `TOKCLOSE_HP_OVERRIDE` (used by the synthetic tests); the log of a fit records the override and control t2 fails when any fit carries one (unless `--allow-hp-override` is given, which the tests use).
5. The estimate table of a fit holds every frame f >= H-1 of the 141 clips (the head needs only H embeddings); the scored frames are the subset of card ADD78 (f >= H+3 with a label).

## 7. Cost and memory (ESTIMATED, not measured on the pod)

* E fit: 432 clips read once for the embedding (numpy, about 1 to 3 minutes), then CPU/GPU head training on 11 MB of embeddings; at most 100 epochs of 44 steps; minutes. Resident memory: one clip batch plus the embeddings, under 2 GB.
* T fit: one pass over 4,080 clips is about 17 GB of reads (float8 on disk); at most 8 passes plus 9 stopping-set evaluations of 432 clips. Resident memory: three prefetched batches of 8 clips in float16 (about 0.2 GB each) plus the batch in float32 on the device; under 3 GB. GPU: under 2 GB (8 clips x 101 frames x 40 tokens x 1,024 floats = 0.13 GB plus activations).
* Run time: eight fits plus the planted fit are about 1 to 4 hours when the cache is in the page cache, up to about 8 hours when every pass reads from disk; the per-fit timeouts in `run_tokclose.sh` are 1 h (E) and 4 h (T).
