# RESULT — refcv7 four-family milestone battery (EvalFlyWheel)

**Pre-registration:** `SPEC.md` (sha256 at registration in `raw/SPEC_SHA256_AT_REGISTRATION.txt`, written 05:29 Berlin BEFORE any refcv7 forward; AMENDMENT A1 05:39, before any refcv7 number: surface M = the S2 windows).
**Status of this file:** the pipeline-validation record (step 1,500) and the armed milestone. The step-5,000 section is APPENDED here by `code/chain_milestone.sh` when it finishes (its full artifacts land under `raw/step5000/`).

> ⛔ **Everything computed on the step-1,500 checkpoint is PIPELINE VALIDATION ONLY — not a result** (SPEC §6). No bar is evaluated on it.

## 1. The kit (MEASURED)
* refcv7 config `D:/refcv7_eval_kit/ckpt/config.json` md5 `e6512a01…` == Thor; launch `fec3a0dc`; the `ev7` code tree (`0c444082`) differs from the launch commit in 2 Training-Watch files only (`git diff --raw --abbrev=40`).
* Data: **290/290** eval-side inputs md5-identical to Thor (`D:/refcv7_eval_kit/kit_checks/eval_inputs_md5_compare.json`), so the refcv6 kit's data root is reused; nothing pulled or copied twice.
* Validation checkpoint: rolling `ckpt.pt` at step 1,500, md5 `c35966f799b1c724515ba1281736fd48` (Thor before == after == local; step key 1,500 read locally).
* Milestone file (from source): `ckpt_5000.pt` = `{model, step}`, written by `MILESTONES = (5000, 15000, 20000, 30000)` (`refc_v3_train.py:225`, `:9645-9647`) after the rolling `ckpt.pt` and before the step's eval row; never rotated. The rolling `ckpt.pt` is the fallback in [5,000, 5,500).

## 2. G0 on the validation checkpoint (step 1,500) — PIPELINE VALIDATION ONLY

Source: `raw/g0_step1500/g0.json` (8 inference seeds × the trainer's 128 in-run windows in its 8 fixed batches, micro-batched `2,2,3,3,3,3`; 1,396 s; peak 5.86 GiB), re-judged with zero GPU by `code/g0_rejudge.py` after SPEC AMENDMENT A2 was registered (`raw/g0_step1500/g0_rejudge.log`).

| verdict | value |
|---|---|
| **G0 as registered** | **FAIL** — exactly ONE term out: `eval_box3d_det_ap4_stroller_0_20` in-run 0.11111 vs 0.14286 on all 8 seeds (sd 0); its support is **n_pos = 1** |
| **G0-A2** (the milestone gate, SPEC A2) | **PASS** — 0 reasons; 238 low-support detection keys REPORTED (n < 30), 96 gating |
| strict load / `param_breakdown` / loader G-DVB | 0 missing, 0 unexpected / equal / 0 mismatches (221 registry entries) |
| SMOOTH (179) | median rel dev **3.6e-5** (bar 0.2 %); worst `eval_lon_tac` 0.17 % |
| STOCHASTIC (6) | all inside the 8-seed 99 % PI (`loss`, `traj`, `cascade`, `law`, `sel_v3`, `goal_score_absmean`) |
| MATCHED (29) / MAP10 counts (160) / MAP10 IoU (80) | median 4.6e-4 / 4.4e-5 / all inside abs 0.01 |
| DETECTION gating (96) | median abs dev 4.5e-5 (bar 0.005) |
| COUNT (210) / null keys (98) | exact / null on both sides |
| wrapper clause (P3 `fp32_det`) | **PASS**: max wrapper rel 6.4e-6 vs floor 1.4e-7 → bar 1e-5 held on every term; W1 and W2 both detected. (P1 as-run 2.8e-3 and P2 1.5e-3 are batch-composition numerics, reported, not gating — the refcv6 A2 finding reproduced) |
| **M1** trunk equalisation dropped (the FIX-3 defect; must FAIL) | **detected**, 1 term: `eval_lon_tac` 1.68761 → 1.64529 (rel 2.5 %). ⚠️ Thin power at step 1,500; the equalise call counter read 0 during the arm, so the mutation did take effect |
| M2 map rule `prior_corrected` → `raw` | detected, 40 terms |
| M4 residual prior OFF | detected, 4 terms (`anchor_acc`, `sel_v3`, `law`, `goal_score_absmean`) |
| requires_grad control (trained flags vs the loader's all-False, 2 windows, eps 0) | 23 of 444 terms differ; the 20 the artifact records (`differ_first20`) are all at **≤ 1.4e-6 relative** (slot heads, traj, cascade, law); the other 3 were not recorded (UNVERIFIED). The battery rolls in the loader's state; the difference is stated beside it |

⚠️ **Judge defect found and fixed (SPEC A2 item 2):** the mutation count first included terms already OUT under the unmutated reproduction (M1 read "2 terms", one being the stroller key). Both verdicts now count only terms a mutation MOVES outside.

## 3. VOID gates 5 and 6 (MEASURED, `raw/g0_step1500/void_gates_map.json`)
* **Gate 5** — the `/3` GT inside the old 60 m × ±16 m window equals the `/2` `fine_codes` byte for byte on **137/137 clips, 27,540 frames** (anchored column offset 140); the refcv6 0.5 m → 10 cm hook literal (one class-5 cell → exactly its 25 fine cells, `NO_PREDICTION` everywhere outside the old window, 1000 × 600) PASSES.
* **Gate 6** — perception is seed-invariant: **603/603** MAP10 / DETECTION / MATCHED terms have rel spread exactly 0 over G0's 8 inference seeds (the heads run before the DDIM planner), so ONE perception pass per checkpoint suffices.

## 4. The armed milestone (step 5,000)
* **Waiter:** `code/chain_step5000.sh wait` → `code/waiter_milestone.sh 5000`, launched detached 05:54:52 Berlin (MSYS pid 134504 / Windows pid 40712). Log `D:/refcv7_eval_kit/chain/waiter_5000.log`; pull log `D:/refcv7_eval_kit/chain/pull_5000.log`.
* **It waits for the EVENT:** Thor's `ckpt_5000.pt` present, size/mtime stable ≥ 20 s, and the step-5,000 eval row in `metrics.jsonl` (the trainer saves before that row). It then pulls read-only (Thor md5 before == after == local), reads the `step` key LOCALLY and REFUSES anything but 5,000. Fallback: the rolling `ckpt.pt` while the last step is in [5,000, 5,500). Poll 5 min; 1 min once the file exists.
* **Then** `code/chain_milestone.sh 5000`: GPU lock (held throughout) → G0 (8 seeds, M1/M2/M4, wrapper; STOP unless G0-A2 = PASS) → T1 rolls at inference seeds 0 and 1 on S2 → panels, four families, BAR-R7-1/2/3 → VOID gates 5/6 → perception pass (map 10 cm + box; refcv7 vs refcv6@38k vs positional prior) → BAR-M7-1..4, BAR-B7-1/2 → interim RESULT → refcv6@38k S2 rolls if the pre-stage has not banked them → re-read bars → `RESULT_step5000.json` + section appended here + a new LANDING_READY heading.
* **By hand** (if the waiter died): `sh D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code/chain_step5000.sh wait` (or `... run` once `D:/refcv7_eval_kit/ckpt/ckpt_5000.pt` and `D:/refcv7_eval_kit/chain/pull_5000_OK.json` exist).

## 5. Pipeline smoke on the step-1,500 checkpoint (PIPELINE VALIDATION ONLY — no number here is a result)
Banked sanitized under `raw/smoke_step1500/` (work dirs on the dev box: `D:/refcv7_eval_kit/smoke/`, `D:/refcv7_eval_kit/battery/smoke_step1500/`). Every stage of the milestone chain ran once, 06:23–06:58 Berlin, under one hold of the GPU lock:
1. **refcv6@38k perception dump** (82c2331 tree, the refcv6 package's loader), 40 S2 windows: 40/40 maps (9-channel 0.5 m logits), 40/40 box3d slot sets (100 queries).
2. **refcv7 perception pass**, the same 40 windows, THREE arms on every window (refcv7 prior-corrected 10 cm codes, refcv6@38k upsampled, positional prior fitted on 300 TRAIN clips / 60,286 frames, md5-verified): 40/40 map windows, 40/40 box3d packs for both models on refcv7's VIS-1 target blocks.
3. **map/box scorer**: `summarize` + `evaluate_bars_m7` + BAR-B7 bootstrap ran; both box controls read their known values (pooled AP == `detection_metrics.summarise` exactly, per-window census additive). Bars: NOT EVALUATED (validation). BAR-B7-2 exposed an undefined-F1 case (no confident slot at σ ≥ 0.5 at step 1,500), now reported NOT EVALUABLE with the census, never as a number.
4. **T1 battery**, 2 clips × 4 windows, seeds 0 and 1, reusing the step-1,500 G0 artifact (G0-A2 PASS; as registered FAIL, shown beside it): rolls, panels, VOID gates 1–4 PASS at both seeds, four-family analysis, cross-model paired cells, S6, tactical, replicate. BAR-R7-2 read ABSENT (no refcv6@38k dump yet), exactly as designed. The criteria registry check first failed to find `products/P7-TanitEval/CRITERIA_REGISTRY.json` in the ev7 tree; it was added to ev7 from the same commit by `git archive`, and the re-run reads 0 violations.
5. **refcv6@38k baseline harness** (the refcv6 package's `run_battery.py`, 82c2331 tree, 1 clip × 3 windows, no G0, smoke): dump written.

**Cost MEASURED here → SPEC A3:** 4.1 s per window for the full arm set (5.4 h per seed over S2), hence os-only at inference seed 1.

## RESULT — refcv7 step5000 (2026-09-28T14:19:53+0200)

* **G0 as registered: FAIL** — eval_traj [STOCHASTIC] in-run 0.64714 vs seeds mean 0.673517 (sd 0.00212)
* **G0-A2 (the gate, SPEC A2): FAIL**; mutation terms moved: M1 7 (detected), M2 60 (detected), M4 3 (detected); wrapper clause PASS; reasons: ['eval_traj [STOCHASTIC] in-run 0.64714 vs seeds mean 0.673517 (sd 0.00212)']
* **Tier:** T1 self-action OPEN LOOP (status UNRULED for an action-free model); never closed loop, never driving performance. **Estimator:** FULL-SET point estimate; paired episode-cluster bootstrap, n_boot 2000, seed 0.
* **Inference-seed floor:** None m ADE (training-seed floor NOT measured).

| bar | verdict | seed 0 | seed 1 |
|---|---|---|---|

| BAR-R7-N1 (NavSim) | not this battery | — | — |

**Four families, paired `os − ha0_ext` (seed 0):** 
* strategic: NOT APPLICABLE (n 0): strategic layer OFF

## 2026-10-04 — WHY step 5,000's G0 failed (EvalFlyWheel; diagnosis, SPEC AMENDMENT A5)

> Every number here is a **G0 loss-replay diagnostic** (the trainer's own in-run eval terms on its 128 fixed windows; `eval_traj` is the oracle-anchor L1), **not a battery number and not a tier claim**. Evidence class: MEASURED from `raw/step5000/g0.json` unless stated.

**The failure.** G0 as registered and G0-A2 FAIL on one term, `eval_traj` [STOCHASTIC]: in-run 0.64714 vs the 8-seed replay mean 0.673517, sd 0.00212 (12 sd). Every other term held (SMOOTH median 5.1e-4, MATCHED 6.4e-4, DETECTION 1.8e-4, COUNT exact, wrapper clause PASS under P3, M1/M2/M4 detected 7/60/3 terms).

**Ruled out from source** (launch blob `refc_v3_train.py` 203b437f = the ev7 copy, md5 0a6fb0d8…; file:line in SPEC A5):
* a step-keyed schedule in the eval path — the loop mutates one model attribute per step, `anchor_withheld_bank` (`:9410-9412`), constant `"fixed"` for this run (`config.json` withheld_bank mode fixed, warmup 0); the LR schedule (`:8969-8971`) is optimiser-only; the tactical-prior EMA is gated on `model.training` (`:4608-4609`); **nothing changes between steps 1,500 and 5,000 in the eval path**;
* EMA / averaged weights — none in the trainer; train/eval mode — `model.eval()` + `no_grad` (`:9658`, `:9672`), F1 random-t only in training (`refc.py:2800`);
* the saved weights — `ckpt.pt` / `ckpt_5000.pt` written at `:9634-9647` BEFORE the eval at `:9656`, same `model`, same step; `train.log`: `ckpt step 5000 -> ckpt.pt` then `[v3:eval] step 5000 … traj 0.6471`; one uninterrupted run (no restart in 1,474 log lines);
* the windows — identical `inrun_eval_perm` sha256; `V3Dataset.__getitem__` draws no random number (`:3808-3937`); `num_workers=0`, `shuffle=False` (`:8908-8911`);
* micro-batching — wrapper P3 max rel 1.5e-6 on `traj`;
* trunk numerics (`--trunk-compile` dropped, Thor vs RTX 4060, bf16) — **quantitatively**: M1 perturbs the trunk enough to move `lat` 1.4 %, `goal_tac` 3.7 %, `lon_tac` 2.6 %, yet moves `eval_traj` only +0.12 % (0.67416 vs seed-0 0.67337); the in-run row matches the replay on those heads to ≤ 0.16 %.

**What remains: the DDIM draw and the estimator built on it.** The eval decode draws `eps = torch.randn_like(x0_n)` (`refc.py:2802`) from the run's training RNG; the replay seeds it. The per-batch seed spread of `traj` (sd 0.0175–0.0333 per batch; 8 batches × 8 seeds) implies an sd of the 128-window mean of **0.00878** if the batches' draws are independent; the 8 seed means have sd **0.00212** (ratio 0.24; χ²₇ lower tail p = 2.8e-4). At step 1,500 the same ratio is 1.30 (sd 0.0153). Under the independence sd the in-run deviation (0.0264) is 3.0 sd.

**The discriminating test, pre-registered in SPEC A5 item 7 BEFORE any of its numbers** (`code/g0_diag_r7.py`, launched detached by `code/orchestrate_final.sh`): 16 more seeds (8..23) + `eps0` / `fp32_s0` / `loaderflags_s0` / `micro_alt_s0` + an `s0` control that must reproduce 0.67337. 24-seed sd ≥ 0.006 ⇒ estimator (A5 is the fix); ≤ 0.004 ⇒ mechanism, localised by the arms; between ⇒ INCONCLUSIVE. **RESULT of the discriminating test (MEASURED 2026-10-04 01:10–02:35 Berlin; `raw/g0diag_step5000/diag.json`, `raw/step5000/g0_A5_posthoc.json`):**

| arm (seed-0 replay unless named; 128 windows, 8 batches) | `eval_traj` | `eval_cascade` | `eval_sel_v3` |
|---|---|---|---|
| in-run (Thor, the run's own draw) | 0.64714 | 4.46348 | 1.85047 |
| `s0` — control, a fresh process through the A5 mmap path | **0.67337** (= G0 seed 0; 1,097/1,098 keys bit-equal, the other `eval_map_hires` rel 7e-9) | 4.71026 | 1.90450 |
| seeds 0..23 (8 from G0 + 16 new): mean / **sd** / min / max | 0.67161 / **0.01072** / **0.63596** / 0.68866 | 4.60576 / 0.08069 | 1.88176 / 0.05502 |
| `eps0` — DDIM draw zeroed | 0.65478 | 4.47285 | 1.82476 |
| `fp32_s0` — trunk fp32 + NCHW, cuDNN TF32 off | 0.67313 (−0.04 % vs `s0`) | 4.71075 | 1.90347 |
| `loaderflags_s0` — every `requires_grad` False | 0.67337 (identical to `s0`) | 4.71026 | 1.90450 |
| `micro_alt_s0` — micro 3,3,3,3,4 (⚠ a different micro split re-maps the eps stream, so this is another DRAW, not a numerics test) | 0.66036 | 4.54096 | 1.87656 |

* **Pre-registered branch reached: 24-seed sd 0.01072 ≥ 0.006 ⇒ the K = 8 sd was a small-sample under-estimate; the mechanism is the ESTIMATOR, and A5 is the fix.** Seeds 0..7 have sd 0.00212, seeds 8..23 sd 0.01308; with 24 seeds the 8 batch draws are independent (row-mean sd 0.01072 vs the independence value 0.01001; off-diagonal/trace +0.15 — the −0.94 seen on 8 seeds was the same sampling accident).
* The in-run value is an ordinary draw of the replay distribution: z = −2.28 on 24 seeds, and **seed 15 (0.63596) lies BELOW it**. `cascade` z −1.76, `loss` −1.51, `sel_v3` −0.57, `goal_score_absmean` −2.58 (below all 24 seeds; inside the A5 interval only through the registered 1 % term — noted, not hidden). These decoder terms move together because they share one draw; the in-run draw was a good one.
* Every other candidate is now measured, not argued: trunk precision 0.04 %, `requires_grad` state 0.0, a noise-free decode (`eps0`) sits between the in-run and the seed mean and does not reproduce it (0.65478 vs 0.64714).
* **Post-hoc A5 re-judge of step 5,000: PASS** (`raw/step5000/g0_A5_posthoc.json`; A5 PI for `eval_traj` [0.63418, 0.70903]; M1 / M2 / M4 still detected: 7 / 60 / 4 terms). ⛔ **POST HOC — A5 was written after the step-5,000 numbers were seen; this is REPORTED and is NOT step 5,000's gate result. Step 5,000's registered verdicts (as registered FAIL, A2 FAIL) stand.** No step-5,000 battery number exists.
* ⚠️ Cost note (MEASURED): an arm took 80–305 s instead of G0's 77 s; the desktop held 3.2–3.6 GB of the 8 GB card, so G0's 5.9 GiB peak spilled into shared memory whenever both were resident. Numbers are unaffected (`s0` bit-reproduced G0); wall time is not.

**SPEC AMENDMENT A5** (registered 2026-10-04T00:58:29+0200, sha256 `5582f092…` in `raw/SPEC_SHA256_AMENDMENT_A5.txt`, before any step-15k/20k/30k/50.4k G0 number): STOCHASTIC class on **K = 24 seeds**, t(0.995, 23) = 2.807, form and 1 % term unchanged — an exact 99 % PI, **stricter in expectation** (2.83 σ vs 3.58 σ); as registered (seeds 0..7) and A2 still reported first; the battery proceeds iff G0-A5 = PASS; A5 fixes an estimator, never a mechanism. Step 5,000's registered verdicts **stand** (FAIL).

## RESULT — refcv7 step30000 (2026-10-04T06:58:23+0200)

* **G0 as registered: FAIL** — eval_agent_det_ap2_bus_40_60 [DETECTION] in-run 0.1 vs seeds mean 0.125 (sd 0); eval_box3d_det_ap2_bus_20_40 [DETECTION] in-run 0.25 vs seeds mean 0.33333 (sd 0); eval_box3d_det_ap2_stroller_0_20 [DETECTION] in-run 0.16667 vs seeds mean 0.14286 (sd 0)
* **G0-A2 (seeds 0..7): FAIL** — eval_tacv6_goal_conf_bce [SMOOTH] in-run 0.15189 vs seeds mean 0.1542 (sd 0)
* **G0-A5 (THE GATE, SPEC A5; 24 inference seeds): FAIL**; mutation terms moved: M1 7 (detected), M2 59 (detected), M4 2 (detected); wrapper clause PASS; reasons: ['eval_tacv6_goal_conf_bce [SMOOTH] in-run 0.15189 vs seeds mean 0.1542 (sd 0)']
* **Tier:** T1 self-action OPEN LOOP (status UNRULED for an action-free model); never closed loop, never driving performance. **Estimator:** FULL-SET point estimate; paired episode-cluster bootstrap, n_boot 2000, seed 0.
* **Inference-seed floor:** None m ADE (training-seed floor NOT measured).

| bar | verdict | seed 0 | seed 1 |
|---|---|---|---|

| BAR-R7-N1 (NavSim) | not this battery | — | — |

**Four families, paired `os − ha0_ext` (seed 0):** 
* strategic: NOT APPLICABLE (n 0): strategic layer OFF

## 2026-10-04 G0 DIAGNOSIS AT STEP 30,000 — why `eval_tacv6_goal_conf_bce` is 1.5 % off, the A6 draft, the lock fix (G0-30k diagnosis agent; written ~08:15 Berlin)

**Verdicts that stand (unchanged, registered):** step 30,000 G0 as registered **FAIL**, G0-A2 **FAIL**, G0-A5 **FAIL** (`raw/step30000/g0.json`). Nothing below changes them. No step-30,000 battery number exists.

**The failing term.** `eval_tacv6_goal_conf_bce` [SMOOTH]: in-run 0.15189 vs 0.15420 on all 24 seeds (sd 0), rel **1.52 %** against the 1 % per-term bar (MEASURED, `raw/step30000/g0.json`). The as-registered DETECTION failures are all A2-exempt: support n_pos = 1–3 (REPORTED, `verdict.terms.*.support_n`).

### Mechanism (source + MEASURED; the 30,000-specific cell-level confirmation is ARMED, see below)
1. **The term is discontinuous in the numerics by construction.** Its target is a hard threshold of the model's own output: `correct = (sigmoid(goal_logits) >= 0.5) == goal_y` (`stack/tanitad/refs/refcv6_tactical.py:823-825`), then `BCE(goal_conf, correct)` weighted by `goal_w × class_mask`, divided by the batch's Σw (`:826-833`). On G0's 128 windows only **271 cells are supervised — 18 / 26 / 53 / 23 / 7 / 53 / 36 / 55 per batch** (MEASURED, `tacv6_n_supervised_goal_cells` per batch). One validity logit crossing 0 moves its batch by exactly ±c·w/Σw (`softplus(c) − softplus(−c) = c`), i.e. the mean by ±c/(8·n_b). The whole 30k gap (−0.00231 on the mean) is ONE flip with a confidence logit |c| of 0.13 (batch 4) … 1.02 (batch 7); in batch 4 any flip with |c| > 0.085 alone exceeds the 1 % bar (ESTIMATED from those counts). SPEC §2 names exactly this property as the reason for the DETECTION class ("a threshold … can flip on cross-hardware numerics"); the by-name table put this term in SMOOTH.
2. **The replay cannot reproduce the run's trunk numerics, and the gap grows with training.** The run used `--trunk-compile` (Inductor, bf16, Thor; `config.json` argv). The loader drops it (`refcv7_loader.py:144-146`, no Triton on Windows) and records "module tree and state_dict identical" — true, silent on numerics; the trunk's own docstring MEASURED on Thor that compiled bf16 differs from eager bf16 (`timm_trunk.py:223-226`). MEASURED drift of the replay vs the run, same 128 windows: SMOOTH class median **3.6e-5 (1,500) → 5.1e-4 (5,000) → 1.64e-3 (30,000)** against a 2e-3 bar; the wrapper clause's as-run batch-partition sensitivity **0.28 % → 0.78 % → 5.2 %** (P1 max rel) while fp32 stays ≤ 6.4e-6 (P3) (`raw/*/g0.json` `wrapper_control`). This term: exact at 1,500 (0.33698 = 0.33698), 0.36 % at 5,000, 1.52 % at 30,000 — the confidence head grows more confident (in-run BCE 0.48 → 0.15, `metrics_final_50400.jsonl`), so each flip costs more.
3. **The discriminating measurement (MEASURED, step 5,000, `raw/g0diag_step5000/diag.json`, an arm pre-registered in A5 item 7):** `fp32_s0` — the SAME weights, flags, windows and code, only the trunk in fp32 + NCHW with cuDNN TF32 off — moved this term **0.29403 → 0.29518 (+0.39 %), entirely in ONE batch** (batch 1, n = 26: 0.33497 → 0.34446; the other seven ≤ 0.0003): one flip with c ≈ 0.25. The continuous `eval_tacv6_goal_bce` moved 0.03 %. The in-run value **0.29509 sits 0.03 % from the fp32 arm.** Over the 13 SMOOTH keys with |x| ≥ 0.1 the median |fp32 − s0| = **5.6e-4** ≈ the median |in-run − s0| = **5.2e-4**: the replay's deviation from the run sits at the dev box's own numerics floor. `micro_alt_s0` (partition only) moved nothing (≤ 2.4e-6).

### Ruled out, with file:line (the brief's candidates a–d)
* **(a) state outside the state_dict.** The training loop mutates one model attribute per step, `anchor_withheld_bank` (`refc_v3_train.py:9410-9412`; `fixed`, warmup 0); the tactical-prior EMA is gated on `model.training` (`:4607`); every non-persistent buffer in `stack/tanitad/` is a construction-time constant (`timm_trunk.py:608-615`, `slot_query_select.py:169-170`, `refc.py:1761-1762`, `refcv7_heads.py:168/250-251`, `encoder.py:369-370`, `readout.py:107-110`, `v6.py:2492/5199-5201`); the tactical decoder is a pure function of parameters + (agent tokens, BEV tokens, nav/max-speed/v0/a0) (`refcv6_tactical.py:500-582`). The `requires_grad` flag state moves no tactical term on the dev box (`requires_grad_control`, 23/444 terms at ≤ 8.6e-7 rel, none `tacv6_*`).
* **(b) a training-time target.** The class mask and pos_weight are launch-time constants from the train split (`refc_v3_train.py:8490-8496`), read by the loader from the stamp (`refcv7_loader.py:472-491`); `tacv6_n_supervised_goal_cells` (Σ w·mask) reproduces EXACTLY at 1,500 / 5,000 / 30,000 (33.875). A static constant would have shown at 1,500, where the term reproduced exactly.
* **(c) a build departure.** Static ones (graft flags via the trainer's own pin, `refcv7_loader.py:360-362`; G-DVB 0 mismatches; `param_breakdown` equal) would show at 1,500. The one departure whose effect GROWS with training is the dropped `--trunk-compile` — mechanism 2.
* **(d) the milestone file.** `ckpt_30000.pt` is `model.state_dict()` written at `refc_v3_train.py:9645-9647`, after `ckpt.pt` (`:9634-9644`) and BEFORE the eval (`:9656`), same object, same step; strict load 0 missing / 0 unexpected. Key sets: the in-run row and the replay carry the same 1,098 keys at all three steps (the in-run's 11 extra are EXCLUDED `*_calib_*` + `eval_step`).

### The 30,000-specific confirmation — ARMED, outcomes committed before any number
* **CPU (no GPU, waits for ≥ 6 GB free commit and no chain marker; gives up 11:26 if never started):** `code/g0_cells_cpu_r7.py` → `D:/refcv7_eval_kit/battery/g0cells_step30000_cpu/cells.json`: G0's 128 windows with the trunk in fp32 on CPU, every batch's tactical cells captured; then `code/g0_cells_analyse.py` decomposes each batch's bf16-replay − fp32 difference and the in-run − fp32 difference into validity flips.
* **GPU (first stage of the armed waiter, ~25 min):** `g0_refcv7.py --seeds 0` on `ckpt_30000.pt` (seed 0 must reproduce the banked seed 0 within rel 1e-5, else REFUSED) + the `fp32_s0` arm + both captures → `code/g0_rejudge_a6.py` → the POST-HOC A6 re-judge, banked to `raw/g0a6supp_step30000/`.
* **Committed now:** CONFIRMED if the fp32 arm's difference from the bf16 replay is a sum of single-cell flips of near-threshold cells (residual at the continuous term's own floor) and the in-run value lies inside the A6 flip interval; NOT CONFIRMED if the fp32 arm moves the term by less than its continuous floor with no flip AND no near-threshold cell can carry ±0.0185 on the 8-batch sum — then the gap is UNEXPLAINED by numerics and the next lever is a seed-0 replay **on Thor with `--trunk-compile` kept** (the loader supports a Thor kit, `refcv7_loader.py:95-101`; Thor compute is the Master Mind's call).

### The fix: SPEC AMENDMENT A6 — DRAFTED, NOT REGISTERED (`raw/AMENDMENT_A6_DRAFT.md`, sha256 in `raw/AMENDMENT_A6_DRAFT.sha256`, written before any step-50,400 number and before any step-30,000 cell/fp32 number)
* **THRESHOLD_TARGET class** (by name): `eval_tacv6_goal_conf_bce`, judged by a **flip interval** — every supervised cell whose replay validity logit is within 3·δ of 0 (δ = the largest validity-logit move the numerics-only `fp32_s0` arm produced on that checkpoint) may flip; the in-run value must lie in the reachable interval ± the SMOOTH tolerance; two controls (cells reproduce the trainer's per-batch values; the interval centre equals the logged replay) or OUT.
* **SMOOTH floor:** tolerance max(registered, 3·φ_k), φ_k = |fp32_s0 − s0| measured on that checkpoint; class-median bar max(0.2 %, 3 × median φ/|x|). 3 = SPEC §2's wrapper-clause convention. Mutations judged with the same tolerances; **M1 undetected ⇒ VOID** (A6 can lose power, never invent it). Every rescued term is named.
* **Power check at 5,000 (POST HOC, partial — no cells existed there):** with the φ floors applied, M1 / M2 / M4 are still detected with the same counts (7 / 60 / 4); the THRESHOLD term fails closed for lack of cells, as designed.
* **Code** (`code/g0_refcv7.py`): every 24-seed G0 now runs `fp32_s0` + both captures (+~1 eval pass) and writes `verdict_A6` beside `verdict_A5`; **the gate switches to A6 only if `raw/SPEC_SHA256_AMENDMENT_A6.txt` was written BEFORE that G0 started** (`a6_registration`). Unregistered ⇒ A5 gates, A6 is reported as DRAFT. Re-judging the banked 1,500 / 5,000 / 30,000 artifacts with the modified judge reproduces every registered verdict, reason list, median and mutation count exactly (A5/A2/as-registered are untouched). Tests: `code/test_g0_a6.py` (literals, a brute-force flip enumeration, the REAL `tactical_behaviour_losses`, and the deliberate regression of a flip-SIGN error this author actually made first), `code/test_g0_main_a6_flow.py` (the real `main()` end to end on fakes: A6 DRAFT → A5 gates; registered-before-start → A6 gates; `--no-a6` = the old flow).
* ⚠️ **Why register it before 12:00 if at all:** under A5 alone the step-50,400 G0 is likely to fail the same way — the same discontinuous term, and the SMOOTH median at 82 % of its bar and rising.

### The lock gate (the 03:35 `ZZCHAINNOLOCK50400ZZ`) — FIXED and tested
`nvidia-smi` timed out once (60 s) under memory pressure; `subprocess.TimeoutExpired` escaped `gpu_gate.gate()` and `gpu_lock.acquire()`, the lock tool wrote no `--out`, the chain read a missing file and stopped (`D:/refcv7_eval_kit/chain/chain_50400.log`). Now a failed probe is a WAIT (fail closed, `gpu_gate.PROBE_ERRORS`, `probe_ok` False) retried every poll; `gpu_lock.py` always writes `--out` and releases a lock it created if anything unexpected raises; `with_gpu_lock.py --child-timeout-s`. `code/test_gpu_lock_retry.py`: 3 timeouts then success → acquired on poll 4; never acquires while the probe fails; never breaks a held lock; the deliberate regression (`PROBE_ERRORS = ()`) reproduces the 03:35 crash.

### Armed (nothing started on the GPU; the NavSim runner holds the lock)
`code/waiter_final.sh 50400 129600 120 disk` (detached; log `D:/refcv7_eval_kit/chain/waiter_final_50400.log`): GO iff the lock file is absent AND free disk on D: ≥ 10 GB AND free commit ≥ 6 GB (`code/final_waiter.py`, fail closed), polled every 120 s for ≤ 36 h → the step-30,000 A6 supplement + post-hoc re-judge → GO re-checked → `chain_milestone.sh 50400`.
⛔ **Named blocker: D: has 7.7 GB free (< 10 GB)** — MEASURED 08:03; until it reaches 10 GB the waiter will not start, by the brief's rule. One-command lever (the Master Mind's call): relaunch it with mode `auto` (`ram` when the disk rule fails): G0's batches then stay in RAM (`REFCV7_G0_BATCH_CACHE=ram`, the same tensors; the disk batch-cache path never runs) and GO needs ≥ 12 GB free commit instead.

### Exploratory (NOT a registered read; the registered check is `raw/PREREG_SEED_GROUP_CHECK_50400.md` on step 50,400)
Step 30,000 `eval_traj` seed means: seeds 0–7 sd **0.00399**, seeds 8–23 sd **0.00715**, one-sided F(15, 7) = **3.21, p = 0.063**; Levene p = 0.20 (MEASURED, `raw/step30000/g0.json`). The step-5,000 pattern (F 38.1) does not recur at this strength; it neither confirms nor refutes reading (A) or (B) — the registered test is the step-50,400 one.
* **Addendum 08:13 (operational, no outcome changed):** the CPU job first met its 6 GB commit bar at 08:11:52 and was stopped by its author before it allocated anything, because free commit was MEASURED swinging **1.57 → 5.86 GB within two 2-min polls** (another stream's ~4.3 GB burst) — a 6 GB bar does not protect that stream from a ~3 GB job. Relaunched with a **10 GB** bar, no start deadline (≤ 36 h), and it also waits while any battery GPU job (`refcv7-milestone-*`, `refcv7-g0*`) holds the lock. Follow-up `code/post_cells.sh` runs the flip decomposition and banks to `raw/g0cells_step30000_cpu/` when it ends.
