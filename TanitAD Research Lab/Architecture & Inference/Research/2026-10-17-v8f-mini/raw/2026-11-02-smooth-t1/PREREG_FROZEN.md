# PREREG_FROZEN: H-SMOOTH-T1-1, the SG(7, 2) residual smoother read inside the T1-kin rollout

Frozen before any smoothed T1 number exists (sha256 and wall clock in `FROZEN_SHA.txt`). Zero training, dev-box CPU. Tier **T1-kin** (own executed state, logged cameras; a self-action open loop). Evidence class of every number: **MEASURED (ours; artifacts in the manifest)**. The Master Mind registers the claim row; this card does not.

**Question.** Does the pre-registered operator of H-V8F-SMOOTH-1 (PASS at T0, 2026-10-20), applied to the plan at every replan inside the rollout, remove the plan curvature jitter that blocks v8F's DOMINANT read against P10?

**Operator (verbatim from `PREREG_V8F_SMOOTH.md` sha256 `a2f19e48…`; nothing tuned).** `S(plan) = P + savgol_filter(plan - P, 7, 2, axis=1, mode="interp")`, per coordinate along the 30 plan steps, in float64. `P` = `out["p10"]` of the very `forward_infer` call that produced the plan (the P10 prior at the EXECUTED state; at tick 0 it is the T0 base). The smoothed plan is what the executor commits. The P10 floor is never smoothed. Wired as `--smooth {off,sg72}` (default off) in `stack/scripts/v8f_t1_eval.py`; no window, order or target is a flag.

**Arms.** X = arm + S, Y = the same checkpoint without S (the BANKED dumps of `runs_soup.json`, sha256 `35833493…`; Y is not re-rolled for any X - Y number). DAG-OACC s0 / s1 (`v8f-dagoacc/M2b_s{0,1}_dagoacc`, ckpt sha256 `67265ec1…` / `90b01474…`), SOUP s0 / s1 (`dagland/soup/ckpt/M2b_s{0,1}_soup`, `238aab03…` / `81924b1a…`). Rollout settings exactly as banked: `--replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal --dump-ticks --dump-heads --device cpu`, batch 32, eval-141, 2,399 windows / 141 episodes, each arm in the source tree of its banked run.

**Reader.** The 19 rows of Read A by the landed gap-split machinery (`extract_arrays.py` on X's dump and ticks, context from the banked DAG-OACC dump; `gapdecomp.decompose` per row: row pool, direction, `separated`). Each row is a window mean of a paired per-window difference. Estimator: paired episode-cluster bootstrap, B = 2,000, seed 0, `taniteval.ci` draw stream; it answers *another draw of episodes* only. The planner is deterministic (no inference seed); each arm is one training run per seed. Rows are reported for **X - P10, X - Y and Y - P10**, with all four families (LONGITUDINAL, LATERAL, TACTICAL, STRATEGIC), PRIMARY (ADE6) and the M1 crawl row.

**Known answers, all before any smoothed number is read.**
* K1 patched tool, `--smooth off`, DAG-OACC s0 on the full grid: dump arrays and tick fields equal the banked ones bit for bit, else max abs deviation <= 1e-4 m (stated with the thread count). K2 (X_off, Y) reads 0 on all 19 rows (|delta| <= 1e-4).
* K3 Y - P10 reproduces the banked `SOUP_READ.json` rows within 6e-5 (as the gap split did). K4 an arm against itself reads exactly 0, interval [0, 0].
* K5 on the smoothed run, tick-0 `plan_raw` equals Y's banked tick-0 `plan` (<= 1e-4 m), and `plan` equals `P + SG(plan_raw - P)` recomputed offline (<= 1e-4 m); the floor `fl` equals Y's.
* K6 pytest: a planted noisy plan loses high-frequency curvature energy under S; a quadratic residual and a zero residual are unchanged; `off` returns the plan object; deleting the smoother call turns the test RED.
* K7 tick-0 `P` equals `p10_yaw4512/p10_eval_pred6.npy` on the grid (descriptive; the T0 base).

**Outcomes, committed now (per arm, both seeds; D1-D3 are Read A).**
* **DOMINANT** iff at BOTH seeds X - P10 meets: D1 ADE6 separated-better with point <= -0.2415 m; D2 no row of the 19 separated-worse (direction-aware); D3 each of the four families has a separated-better row (STRATEGIC is not paused for v8F).
* **HARMS** iff at EITHER seed ADE6 X - Y point > +0.0604 m, or a non-crawl row (the 15 rows other than yaw-rate MAE, curvature MAE, L2 curvature MAE, M1 crawl share) reads X - Y separated-worse.
* **MOVED** iff not DOMINANT and at BOTH seeds every one of the four crawl rows reads X - Y separated-better while ADE6 X - Y point <= +0.0604 m.
* **NO EFFECT** otherwise. Precedence DOMINANT, then HARMS, then MOVED, then NO EFFECT (an overlap of MOVED and HARMS is reported as HARMS, with the MOVED condition stated). Multiplicity: HARMS tests 15 rows x 2 seeds; the number of separated-worse rows is reported beside the verdict.
* The headline arm is DAG-OACC; SOUP is stated separately. Beside, decides nothing: plan mean |kappa| and curvature roughness (mean squared first difference of kappa) of the executed path in the v0 3-8 m/s band (Y reproduces the banked 0.1845 / 0.2621 for DAG-OACC), and of the issued plan before and after S.

**Prediction (ESTIMATED from the T0 effect, written before the read).** At T0, S cut yaw-rate MAE by 0.82-1.53 deg/s at no ADE cost. The T1 gap of DAG-OACC against P10 on yaw-rate MAE is +3.47 / +8.68 deg/s, so I expect lateral rows to move separated-better but not to close; the M1 crawl share (a low-speed step rule) may not move. DOMINANT <= 5 %, MOVED about 30 %, NO EFFECT about 55 %, HARMS about 10 %.

**Next arm per outcome (Rule Zero).** DOMINANT: re-read on a fresh seed or window set before adoption; the training lever is a smooth output basis (the PI's go). MOVED: name the rows still separated-worse vs P10 per seed; the cheapest next step is S composed with the curvature-bounded decode below. NO EFFECT or HARMS: a curvature-bounded decode (`feasible_decode.py` envelope, KAPPA_MAX 0.2) as the next zero-training arm, wired the same way as a flag of the same tool.

**Not decided here.** Adoption into any K-1 / WM-1 claim (it needs its own confirm on a fresh read, as H-V8F-SMOOTH-1 states). No operator, row, threshold or arm is changed after data.
