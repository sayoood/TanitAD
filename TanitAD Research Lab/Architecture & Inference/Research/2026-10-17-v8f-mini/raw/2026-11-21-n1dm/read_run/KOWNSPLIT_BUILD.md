# K-OWN-SPLIT build (PREREG_V8F_N1DM 2.3 iii) -- read-side diagnostic, no bar, decides no outcome

Nothing is committed or staged (the MM lands). All files are in `F:/TanitAD-scratch/n1dc/read_n1dm/`. Code of record: the N1DM tree in `read_n1dm/tree` (not edited).

## Source facts (stack/tanitad/models/v8f_mini.py of the tree, quoted by line)
- :1156 `_forward(b, p_gt, use_labels, gen, code_override)`; :1189 `own` all True; :1195-1197 under `code_mix == "whole"`: `use_t = (rand < p_gt) & lab_t.has`, `own = ~use_t`.
- :1214 `out["route_logits"], out["route_own"]`; :1397 the route CE mask is `route_ok & route_own`, so the head is trained on own-code rows only.
- :1121-1122 the branches read `p = softmax(route_head(h)).detach()`; h holds the GT code on teacher-forced rows.
- :317-318 `aux_route != off` requires `code_mix == whole`. Both N1DH (cond) and N1DM (modes) checkpoints carry `code_mix whole` (read from their cfg in the result JSON). The tool REFUSES any other code_mix.

## Design
- Probe batch: the pinned evaluator itself (`n1dm_t0route_lib` setup, `v8f_t1_eval_hb`, `--limit logged`, one tick). A wrapper after the lib's own capture keeps a deep clone of the collated batch of each forward_infer call INCLUDING `lab`, `lab_t`, `lab_s` (the lib's tensor-only clone drops the dicts; the evaluator's inference batch is key-guarded and carries no labels). Labels: the trainer's own `v8f_mini_train.route_stage` (the K-LABEL path), unedited, on that batch.
- Arms (model.eval(), no_grad, 64-row slices): OWN = `_forward(b, 0.0, True)`; GT = `_forward(b, GT_P=1.0, True)`; GT_TCODE_ONLY = `code_override` = GT code on has rows, own elsewhere (S untouched; at p_gt 1 the S-checkpoint substitution is also on, as in training, so this arm isolates the T-code effect); NATURAL = p_gt 0.2 with a seeded generator, split by `route_own`.
- Metric: route-head argmax accuracy vs `route_y` on rows with `route_ok & lab_t.has`. Paired difference GT minus OWN, `taniteval.ci.paired_episode_cluster_bootstrap`, B 2000, seed 0. This answers only "would another draw of EPISODES say this"; it does not speak to another training run or inference run.
- Refusals (never a number): no route head; `code_mix != whole`.
- Controls: K-ID, K-MASK, K-PERM, K-UNIF, K-INFER (OWN-arm logits equal forward_infer bit for bit); on real runs K-ALIGN (captured plans equal the dump) and K-LABEL-RUN (route_stage equals the dump's kinematic label).
- The probe is sliced at 64 rows: the first N1DM_s0 run on the whole 480-row batch ended in a native crash (rc 139) in the read phase; the sliced run is the one reported (the unsliced N1DM_s1 run gave identical numbers).

## Files
| file | what |
|---|---|
| n1dm_kownsplit.py | the tool (`read_split` core, `main` runner) |
| test_n1dm_kownsplit.py | 9 test cases (T1-T6), green |
| RESULT_N1DH_s0_KOWNSPLIT.json, RESULT_N1DH_s1_KOWNSPLIT.json | real runs, N1DH checkpoints |
| RESULT_N1DM_s0_KOWNSPLIT.json, RESULT_N1DM_s1_KOWNSPLIT.json | real runs, N1DM checkpoints (full records with controls) |
| RESULT_N1DM_KOWNSPLIT.json | N1DM s0+s1 combined summary (requested) |
| run_kownsplit_real.sh | driver for the four real runs (gated, one python at a time); log `kownsplit_real.log`, `kownsplit_real_N1DM.log` |
| run_n1dm_reads.sh (+ `.bk1`) | step `kownsplit_s0/s1` added after the K-LIM loop (before the eval-141 chain); `.cmd` calls the .sh, unchanged |
| dry_check_n1dm.sh (+ `.bk1`) | compile of both new .py, the pytest file, and a wiring check |
| dry_check_n1dm.log | last line `ZZ N1DM-READ-DRYCHECK PASS ZZ` (33 PASS lines, 0 FAIL) |

## Tests and mutations
T1/T2 the record is VALID on a built route model in `cond` and `modes`, controls read their known values. T3 refusals. T4 a stand-in model right only on GT-code rows reads own acc 0.0, GT acc 1.0, gap exactly +1.0, n rows = has rows. T5 mutations of the tool source: `GT_P` 1.0 to 0.0 and to 0.2 turn K-MASK RED (status INVALID), and dropping `& has` from the pair mask breaks T4's row count and gap. T6 slicing preserves argmax and row count.

## Results (MEASURED, ours; T0 training-format forward; paired episode-cluster bootstrap B 2000 seed 0; eval-141 probe windows, 480 windows, 472 with a T label, 139 episodes; NOT D1-HO)
All five controls PASS on all four runs. Class share of route_y on the pair rows: 0.708 / 0.097 / 0.195 (lane_keep / left / right). Accuracy is route-head argmax accuracy.

| arm | acc OWN | acc GT | gap GT-OWN [95 % CI] | T-code-only gap [CI] | natural p_gt 0.2: own rows / GT rows |
|---|---|---|---|---|---|
| N1DH_s0 | 0.8729 | 0.9068 | +0.0339 [0.0106, 0.0596] | +0.0212 [0.0000, 0.0446] | 0.875 (n 384) / 0.927 (n 96) |
| N1DH_s1 | 0.8792 | 0.9174 | +0.0381 [0.0085, 0.0687] | see JSON | see JSON |
| N1DM_s0 | 0.8771 | 0.9068 | +0.0297 [0.0085, 0.0520] | +0.0191 [-0.0041, 0.0423] | 0.880 (n 384) / 0.927 (n 96) |
| N1DM_s1 | 0.8750 | 0.9174 | +0.0424 [0.0107, 0.0746] | see JSON | see JSON |

Reading: the head is a few points more accurate on GT-code rows, in N1DH and N1DM alike, separated on the episode draw; about two thirds of it survives in the T-code-only arm (CI touching zero). The magnitude is small (3 to 4 points on a base of about 0.88) and the card commits no bar, so this document makes no call on whether the shortcut "fires". Limits: single probe set of 480 eval-141 windows; the interval answers only another draw of episodes, never another training run (the N1DH s0/s1 pair and the N1DM s0/s1 pair are the only seed spread available, 0.030 to 0.042).

## Not done / open
- The D1-HO profile (`--profile ho`) needs the held-out clip caches (about 10 GB), which are not on the dev box; the tool supports it through the `--ho-*` flags (same as K-LIM) and the dev-box CPU is enough once they are pulled. Nothing was staged on or run against a pod.
- The `run_n1dm_reads.sh` step reads `eval141/ckpts/N1DM_s$s/ckpt_last.pt` (the path K-LIM uses); the checkpoints on the box are at `ckpts/N1DM_s$s/ckpt_last.pt`. The real runs above used the latter via `run_kownsplit_real.sh`. Placing or linking them under `eval141/ckpts` is a step for the owner of `pull_n1dm.sh`.
- Temp deleted/moved are logged in `F:/TanitAD-scratch/n1dc/DELETIONS.log`; first-attempt results are kept in `read_n1dm/tmp/` (`*_unsliced.json`, attempt logs).
