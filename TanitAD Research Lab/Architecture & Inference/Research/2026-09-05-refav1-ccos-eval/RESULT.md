# D-REFAV1-CCOS-EVAL — the centred-cosine fix MEASURED on the real implementation, and refav1 re-evaluated with it

**status: IN PROGRESS — done: predecessor's uncommitted ccos plumbing rescued, tests made true (62 green) and COMMITTED (`044eafc`); the REAL-implementation goal-term panel on the full 282-window grid (§1, both probes, controls); the weight-neutrality factor and compensated triple (§2); the `cos` arm re-analysed with the `ha0_ext` echo control (§4); criteria checker on the `cos` record: 0 violations / 0 work items / const0 OK; echo gate 1 on the `cos` arm: FAIL (§4); the cross-dump paired tool passes its known-value control (exactly 0.0000 [0, 0] on all 10 metrics × 2 strata). RUNNING: Thor chain (ccos naive → ccos compensated → chord, `/home/nvidia/refav1_ccos/`, launched 02:18Z, ~3.5 h per arm) and dev-box chain (chord → ccos compensated, `C:\Users\Admin\ccos_eval\devbox\`, ~2 h per arm). next: when `dump_ccos_naive` completes → analyse (`refav1_arm.py --analyze-only`) → shape panel (§1b–d) → suite + criteria → paired ccos-vs-cos on the same windows → echo gate → register rows.**

Arch+Inference FlyWheel · 2026-09-05 · checkpoint `refav1-b1-v72-ep3-speed/ckpt.pt` step **21,109** (strict load, `missing_keys: []`; md5 `1189bc020018c2c67ce03d566c390285`, 2,122,997,633 B — identical on Thor and the dev box) · **OPEN LOOP** (PI ruling 2026-09-02): every planner arm is **T1** (self-action open loop: the predictor consumes the planner's own actions); the WM diagnostic is T0. Nothing here is driving performance.

⚠️ **The brief named `/home/nvidia/experiments/refav1-b1-v72-1ep-21109` as the checkpoint. That directory is an EARLIER run: its `train_log.jsonl` stops at step 1,000, it carries a `DRIFT_ALARM`, and no `summary.json`.** The step-21,109 model is `refav1-b1-v72-ep3-speed/ckpt.pt` (`summary.json`: `"done": true, "final_step": 21109`; `MODEL_REGISTRY.md` §2.4; the banked `cos` eval's own `launch.sh`). Every number below is on that checkpoint.

**Two sample sizes, never mixed.** The panel (§1) and the arms (§3) are the **full 282 windows / 141 episode clusters** of the banked `refav1-21109-openloop` grid (v7.2 EVAL split, `--window-stride 40`, K = 10 @ 0.2 s, `--action-units kappa`, `--no-navshuf`). The pilot factor (§2) is the predecessor's **40-window / 20-clip** dev-box form comparison. Intervals: **episode-cluster bootstrap** (`taniteval/ci.py`), **paired** for two arms on the same windows; `overlapping_holdout_se` nowhere.

---

## 0. What is settled so far (evidence class beside each)

| finding | class |
|---|---|
| **The fix exists in code and is default-safe.** `COST_METRICS = ("cos", "chord", "ccos")`; `_goal_term(zt, g, "ccos", z_ref) = 1 − cos(z_t − z_ref, g − z_ref)` with `z_ref` the window's own zero-action terminal field from `plan()`'s `_zero_action_ref` (same predictor, same `z0`, same forward pass; refuses any other centring); `"cos"` bit-identical to the pre-change expression; `plan(cost_weights=)` carries the weight triple on the call and stamps it on the result. 62 tests green. | MEASURED (`stack/tests/test_cost_ccos.py`, `test_cost_chord.py`, `test_refav1_arm.py`; commit `044eafc`) |
| **On the REAL implementation, over all 282 windows: `cos` excludes 100.00 % of a 300-sample iCEM iteration-0 population before the world model is consulted (goal(cv) median 1.19e-07, max 5.58e-04 against a smallest penalty 0.094); `chord` — the uncentred, monotone-equivalent form, the deliberate regression — ALSO 100.00 %; `ccos` 1.67 % at the median window, 0.33 % at the most favourable.** | MEASURED (`raw/panel_282.json`, from `thor/box_panel_282.json`; second probe `thor/ccos_panel_probe.json` agrees on cos/chord to ≤ 2.2e-07) |
| **But `ccos` has a measured HOLD stratum on 133/282 = 47.2 % of the grid**: there the decoded goal's canonical controls are zero, the goal rollout IS the zero-action rollout (`‖g − z_ref‖/‖z_ref‖ < 1e-6`), the centred goal direction is the zero vector, and **every candidate reads exactly 1.0 (ptp 0.0 on 133/133)** — the term carries no information and the penalties alone decide, so the injected zero-penalty baselines win exactly as under `cos`. On the other 149 windows the term spans its range (box spread median 1.99 of [0, 2]) and the L/R decision is 193 % of the term's magnitude (vs 59 % for `cos` measured in float32 ulps, 30 % for `chord`). | MEASURED (`raw/panel_282_strata.json`) |
| **`ccos(cv)` is 1.0 exactly only when cv's field is bit-identical to `z_ref`.** In the planner's own batch the cv row is rolled beside other candidates while `z_ref` is rolled alone, so cv's centred vector is batch-composition noise and its cosine against `g − z_ref` is O(1/√D): on the 149 non-HOLD windows `ccos(cv)` = 1.0 ± 0.062 (median |dev|), range **0.777–1.177**. The planner's floor is therefore partly decided by rounding — a defect of the same family as the `cem`-label mislabel (RESULT 2026-09-04 §5), now on the value rather than the label. The smoke read `basecost_cv_cl` 0.959 / 1.033 on its two windows. | MEASURED (`thor/box_panel_282.json` `controls.ccos_cv`; `thor/dumps/dump_smoke/decisions`) |
| **Weight neutrality: `ccos` is NOT weight-neutral, by a decision-scale factor of 643× [p10 260, p90 1,560] on the non-HOLD stratum (n = 149; 129× over all 282 because the HOLD windows contribute 0; pilot 214× on n = 40) and a value-scale factor of 2,850× (n = 149) / 33,715× (pilot).** The compensating triple at the decision scale is **(12.859, 32.149, 64.297)**; the closed-form iteration-0 bound then reads **100.00 % excluded** again — compensation restores the flat plan by construction. That is the pre-registered prediction for the compensated arm (§3). | MEASURED (`raw/panel_282_strata.json`, `raw/ccos_scale_factor_devbox40.json`, `raw/arm2_weights_decision.json`) |
| **The `cos` arm against the echo control `ha0_ext` (constant (a0, κ0) of the measured t0 state, T1, added post hoc with three content checks):** ADE `cl − ha0_ext` = +0.0267 [−0.0285, +0.0813] straddles; LON speed MAE **+0.2706 [+0.2140, +0.3312] separated WORSE**; LAT cross-track **−0.1409 [−0.1844, −0.1015] separated BETTER**, heading −1.54° [−2.04, −1.07], yaw-rate −0.0409 [−0.0529, −0.0297] (both better); TAC lon correct **−0.1277 [−0.1879, −0.0674] separated WORSE**. ⇒ the straight-line plan beats the ego-extrapolation laterally only because `ha0_ext` holds the recorded κ0 (which drifts), and loses to it longitudinally — the same "wins by doing nothing" pattern as against `ha`. `cl − ha0` unchanged: +0.0158 [+0.0007, +0.0315]. | MEASURED (`thor/rec_cos_ext.json`, n = 282 / 141, paired episode-cluster bootstrap n_boot 2000) |
| **The smoke (ccos naive, 2 windows of clip 01be5919): the planner TURNED** — window 1 (decoded goal TURN_R) emitted a `cem` plan with |κ| up to 0.08 1/m of the correct sign; window 0 (goal BRAKE_TO) emitted the injected `decel_1.5`. First real re-ranking observed on this model. **n = 2; a smoke, not a result.** | MEASURED (`thor/dumps/dump_smoke`, `raw/shape_cos_and_smoke.json`) |

---

## 1. The panel — the REAL `_goal_term` under `cos` / `chord` / `ccos` on the same captured fields (n = 282 / 141)

Source: `thor/box_panel_282.json` (predecessor's `ccos_box_panel.py`, Thor, 443 s: one `plan(cost_metric=m)` capture PER metric so each metric's closure — including `z_ref`, which only the `ccos` closure builds — is the planner's own; 22 designed candidates: cv, decel_1.5, ±κ ∈ {0.2, 0.1, 0.05, 0.02, 0.01}, ±a ∈ {4, 2, 1.5, 0.5}, turnL/R_slow, rampL/R; float32 as the planner sees it) cross-checked by an independently written second probe (`tools/ccos_panel_probe.py`, 304 s) and by the float64 re-implementation banked beside both.

### 1a. Fraction of the iteration-0 population excluded before the world model is consulted

Closed-form bound (`analyse8_exclusion.py`'s rule, unchanged): the goal term is non-negative, so candidate n beats cv only if `penalty(n) < goal(cv) − goal(n) ≤ goal(cv)`. `penalty` is `W_JERK·jerk_raw + W_KAPPA·κ_raw` of the REAL 300-sample iteration-0 population (`colored_noise`, seed 0: jerk_raw min 4.711, median 18.01 ⇒ penalty median 0.362, max 1.344).

| metric | goal(cv) median [CI] | goal(cv) max | exactly 0 / exactly 1 | box spread (median) | L/R decision as % of term [CI] | **iter-0 EXCLUDED @median / @max window, shipped weights** |
|---|---|---|---|---|---|---|
| `cos` (shipped) | 1.19e-07 [5.96e-08, 1.79e-07] | 5.58e-04 | 95 / 0 | 2.42e-03 | 31.3 % [30.5, 38.4] *(ulp-quantised: the term is a handful of 2⁻²⁴ steps)* | **100.00 % / 100.00 %** |
| `chord` (deliberate regression) | 3.63e-04 [9.5e-08, 5.5e-04] | 3.34e-02 | 0 / 0 | 6.84e-02 | 8.5 % [0.8, 15.6] | **100.00 % / 100.00 %** ✅ *fails, as it must* |
| `ccos` | 1.0 [1, 1] | 1.177 | 0 / 133 | 1.95 | 189 % [0, 191] | **1.67 % / 0.33 %** |

By stratum (`raw/panel_282_strata.json`; HOLD = `‖g − z_ref‖/‖z_ref‖ < 1e-6`, i.e. the decoded goal's canonical controls are zero):

| stratum | n (win / clusters) | `ccos` goal(cv) | `ccos` box spread | `ccos` L/R share | `cos` L/R share | identity control (candidate = goal) |
|---|---|---|---|---|---|---|
| HOLD | 133 / 85 | **1.0 exactly on 133/133; every candidate 1.0, ptp 0.0** | 0.0 | 0 % | 18 % | `ccos` reads 1.0 (zero vector vs zero vector — no information, correct) |
| non-HOLD | 149 / 93 | 1.009 median, **0.777–1.177** (batch-noise, see §0) | 1.99 | **193 %** | 59 % | `cos` 2.4e-07, `chord` 0.0, `ccos` 0.0 |

**Controls, real function, real fields:** zero-model ptp (all candidates = the cv field) = **0.0 exactly** for all three metrics on 282/282; identity control `chord` = 0.0 exactly (Sterbenz), `cos` ≤ 2.38e-07 and `ccos` = 0.0 on non-HOLD windows (the float32 cosine of identical vectors, as `test_c` now states); float64 cross-check max |ccos32 − ccos64| = 2.49e-07; the second probe agrees on `cos`/`chord` to ≤ 2.2e-07 on every cell, on `ccos` to ≤ 8.8e-06 on every non-cv row, and differs on the **cv row by up to 0.031** — exactly the batch-composition residual (its box has a different size), which is the mechanism of §0's `ccos(cv)` finding, reproduced by two tools.

### 1b–1d. Shape of the emitted plans — `cos` banked; `ccos` / `chord` arms RUNNING

| arm (n = 282 / 141 unless stated) | L / R / straight | distinct plans | bit-exact to an injected baseline (zeros ∪ decel_1.5) | κ ≡ 0 | plan turns on the 38 TURN_L/R-goal windows (sign agrees) |
|---|---|---|---|---|---|
| `cos` @ shipped (banked 2026-09-04, re-read by content) | 0 / 0 / 282 | **2** | 270 + 12 = **282/282** | **1.0000** | 0/38 (—) |
| `ccos` @ shipped — NAIVE (smoke, n = 2 / 1) | 0 / 1 / 1 | 2 | 1/2 (decel) | 0.5000 | 1/1 (1/1) |
| `ccos` @ shipped — NAIVE, full grid | *running on Thor* | | | | |
| `ccos` @ compensated (12.859, 32.149, 64.297) | *queued (Thor arm 2 / dev box arm 2)* — **PRE-REGISTERED PREDICTION: κ ≡ 0 on ~100 %, 2 distinct plans** (100 % iteration-0 exclusion by the bound) | | | | |
| `chord` @ shipped (deliberate regression, full grid) | *running on the dev box* — **PRE-REGISTERED PREDICTION: κ ≡ 0 on 100 %** (24-window sweep S6 of 2026-09-04 read 100 %) | | | | |

---

## 2. Weight neutrality — the factor and the compensating triple

| quantity | pilot (float64 re-implementation, n = 40 / 20, dev box) | REAL implementation, all 282 | REAL, non-HOLD 149 / 93 |
|---|---|---|---|
| value leverage `ccos / cos` (median cell ratio) | 33,715× | 5,166× | 2,850× |
| **decision leverage** (median ratio of the term's spread over the candidate box) | **214.46×** | 128.6× [0, 318] | **643.0× [p10 260, p90 1,560]** |
| L/R-gap ratio | — | 123,650× | — |
| `chord / cos` decision leverage (for scale; chord cannot re-rank) | — | 28.2× [26.5, 29.9] | — |

**Compensating triple** = shipped × F. At the decision scale (what trades against the penalties) with F = 643.0 (the stratum where the term varies at all — on HOLD windows any factor gives the same flat plan): **`(W_JERK, W_KAPPA, W_VEND) = (12.859, 32.149, 64.297)`**, banked in `raw/arm2_weights_decision.json` and shipped to both chains as `arm2_weights.txt` BEFORE any compensated arm started; the pilot triple (4.289, 10.723, 21.446) is superseded and recorded. `W_VEND` is never charged on the eval path (`target_speed=None`). Under either triple the iteration-0 bound reads **100.00 % excluded** — so the compensated arm is predicted to reproduce the flat plan, which is what makes it the control: any difference between the naive and the compensated arm is the implicit re-weight, not the centring.

⚠️ The predecessor's `test_cost_ccos.py` docstring quotes a value leverage of "751,546×" from an unbanked `raw/ccos_scale_factor.json`; that file never existed on disk. The definitions above are stated and the numbers re-measured; the decision leverage matches the predecessor's 214.46 exactly on the same 40 windows.

---

## 3. The arms (RUNNING) — filled in as they land

Chains: Thor `/home/nvidia/refav1_ccos/thor_chain_ccos.sh` (smoke ✅ exit 0 → `ccos_naive` → `ccos_comp` → `chord_shipped`; each writes `dump_<arm>/`, `rec_<arm>.json`, `<arm>.log`, `<arm>.EXIT`); dev box `tools/devbox_chain.sh` (`chord_shipped` → `ccos_comp`). `--analyze-only <dump>` recovers any partial arm with zero GPU.

## 4. The `cos` arm with the echo control, and echo gate 1

`thor/rec_cos_ext.json` (suite `suite_cos_ext/`: 0 violations, 0 work items, const0 OK). **Echo gate 1** (`stack/tanitad/eval/echo_gate.py::echo_gate`, references `ha`, `ha0_ext` (required) + `ha0`, per horizon slot, paired episode-cluster bootstrap, `raw/echo_gate1_cos.json`): **FAIL** — at 1.0 s the arm (0.3679 m) is separated WORSE than `ha` (0.3219; Δ −0.0460 [−0.0805, −0.0102]), `ha0_ext` (0.3120; Δ −0.0559 [−0.0874, −0.0232]) and `ha0` (0.3561; Δ −0.0118 [−0.0219, −0.0021]); at 2.0 s every reference straddles (`ha` +0.0936 [−0.0899, +0.2877], `ha0_ext` +0.0422 [−0.1144, +0.2109], `ha0` −0.0354 [−0.0769, +0.0044]). Gates 2 / 2b need the model re-run under deranged inputs and are REFUSED from a dump (n = 0, reason recorded); `assert_not_echoing` is NOT called with a fabricated gate 2. The functional ablation (`tools/refav1_source_ablation.py`) is queued on the dev box after its chain.

Stratified paired reads on the SAME windows (`raw/paired_control_cos_vs_cosext.json`, n_boot 500; the control pair `cosext.cl − cos.cl` reads **exactly 0.0000 [0, 0]** on all 10 metrics in both strata): on the **149 non-HOLD** windows `cos.cl − ha0` ADE **+0.0299 [+0.0046, +0.0589] separated WORSE**, FDE +0.0669 [+0.0033, +0.1426], LON speed +0.0584 [+0.0149, +0.1059], LON along +0.0488 [+0.0166, +0.0855]; on the **133 HOLD** windows every `cl − ha0` delta is identically 0 (the plan IS `ha0` there). Against `ha0_ext`: LON speed +0.3411 [+0.2540, +0.4273] worse / LAT cross −0.1815 [−0.2630, −0.1110] better on non-HOLD; +0.1916 / −0.0955 on HOLD (both separated).

## 5. Provenance

| what | where |
|---|---|
| code | commit `044eafc` on `agent/arch-inf-20260803` (`refa_v1.py` `cost_weights`; `refav1_arm.py` `--cost-metric/--cost-weights`, `ha0_ext`, `basecost_*`; `refav1_add_floor.py`; `t1_eval.py` `ha0_ext: T1`; tests) |
| Thor tree | `/home/nvidia/refav1_ccos/repo/` = `evalstack_ccos.tgz` md5 `cfc43a35fe0289b7616586c47a459a7c` built from the mirror; `refa_v1.py` md5 `c3dab40241be4830c27c447f46e00946`, `refav1_arm.py` `4ac869ee0c92a5be49e4f5150dee5491`, `t1_eval.py` `d944a5fb4197acce9861c076ac42e47f` — verified on both ends |
| dev-box inputs | `C:\Users\Admin\refav1_eval_full\{fp8 (8.7 GB, 142 files), eps (5.1 GB, 141)}` pulled from Thor by tar stream; labels md5 `aa12c948f062181c3297265b51526ec5` (identical to the banked run) |
| estimator | episode-cluster / paired episode-cluster bootstrap, n_boot 2000 |
