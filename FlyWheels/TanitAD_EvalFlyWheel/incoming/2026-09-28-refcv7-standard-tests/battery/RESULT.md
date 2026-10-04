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
