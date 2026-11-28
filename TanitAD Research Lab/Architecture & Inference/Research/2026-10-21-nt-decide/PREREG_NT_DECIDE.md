# PREREG_NT_DECIDE: does a navtrain fine-tune of refc close the gap to REFe? (`H-NT-DECIDE-1`)

**Status: DRAFT, registered before any data exists.**
* Written 2026-10-21 by the Research Lab (Architecture & Inference).
* Becomes binding when the Master Mind files it in `GOALS_AND_CLAIMS.md`, **and** the PI authorises compute, **and** the PI's choice of the DATA line (§4.1) is written to `AMENDMENT_ACK_NT.txt`.
* No arm has been trained or read. No navtest number for A1 or A0 exists.

Tier: NAVSIM v1.1 PDMS, non-reactive pseudo-simulation of an open-loop plan against logged agents (**T1-family, not closed loop**). Rolls on Thor CUDA, scoring on the Thor CPU backend, one backend throughout (GROUNDWORK B6). REFe rows are mixed-backend: REFe's own rolls, scored on the dev box. This is stated wherever a REFe row is used.

## 1. Hypothesis and the committed question

`H-NT-DECIDE-1`: *refc's 16-point navtest gap to REFe is mostly a **data** gap.*

Test: fine-tune θ_DEC on NAVSIM navtrain with the refc architecture unchanged and the front 120° camera only (arm A1).

The hypothesis says nothing about cameras, backbone or the selection method; those are the residual. ⚠️ One measured residual is outside both of the PI's named buckets. On REFe itself (§14.2: 83.24 → 84.46 → 88.25 → 88.82 / 89.15):
* REFe's **learned PDM scorer** is worth about **4.5 points** (84.46 → 88.99);
* the **route fix** is worth a further **1.2** (83.24 → 84.46);
* every REFe row also carries a route-derived **goal-point input** that refc lacks. That is why §4 reports a method-matched share beside the PI's thresholds.

## 2. Arms (definitions in `DESIGN_AND_COSTING.md` §2–§3)

| arm | definition |
|---|---|
| **A0** | θ_DEC = `ckpt_wiseft_mod2_DEC.pt`, md5 `dad4cb5b5fb316922b4e9a7e85cee8f8`, no training |
| **A1** | θ_DEC + navtrain fine-tune. Everything trainable. 12,478 steps at batch 16 (2 epochs of the 99,824-token / 1,154-log pool). Seed 0. All refcv8 flags unchanged except the NavSim data inputs listed in DESIGN §3. Selection per §5. |
| **A1r** | A1 with `--seed 1`. Mandatory under §4.3; recommended otherwise. |
| A2 / E-FOV | OPTIONAL. Never deciding. Each needs its own go. |

Held out of all training: all **38** selection-scan logs (sample sha `9dd68698…`, 2,089 tokens / 30 logs with tokens) and every navtest log.

## 3. Metrics

**PRIMARY:** navtest PDMS ×100 (12,146 tokens / 136 logs), arm R8_LEGAL, mean over **inference seeds 0 and 1**.

**Reported beside, never deciding:**
* PDMS sub-scores: NC, DAC, TTC, EP, C.
* The straight-command and turn-command cells (the known −19.64 straight deficit, registry §16.3).
* The STOP, PRIOR_ha0p and CV floors on the same tokens.
* PhysicalAI eval139 in-run metrics as a **forgetting monitor**.

**The four families** (binding rule 2026-08-02), via the NavSim families instrument (`…/2026-10-21-refcv8-wiseft/families/`, 01a71dc). Pairs: A1 − A0 and A1 − PRIOR, both seeds, per family, never pooled:

| family | reported | not computable |
|---|---|---|
| LONGITUDINAL | speed MAE, along-track, accel; TAC longitudinal agreement | **distance keeping UNAVAILABLE** on the NavSim bridge (no lead track); n = 0 stated |
| LATERAL | heading, cross-track, **pair-masked** yaw rate, curvature (⛔ the unmasked yaw rate is inadmissible on NavSim, §16.7) | |
| TACTICAL | trajectory-derived lateral / longitudinal decision agreement | |
| STRATEGIC | — | **UNAVAILABLE**, n = 0 (`--no-strategic`; NavSim scores no strategic decision) |
| ADE | over the 8 NavSim poses | |

## 4. Committed outcomes (fixed now; nothing is chosen after a read)

Let **P** = A1's primary metric (§3). When §4.3 applies, P is the mean over training seeds of each seed's inference-seed mean.

### 4.1 The verdict

| verdict | condition |
|---|---|
| **DATA** | P ≥ **L_D**, **and** the paired A1 − A0 interval lies above 0 at log_name **and** at nuplan_drive |
| **SENSORS/BACKBONE** | P ≤ **75.0**, **and** the plateau rule (§5) holds |
| **BOTH** | 75.0 < P < L_D, **and** the plateau rule holds. The split is named by §4.2. |
| **UNDERTRAINED** | P < L_D, **and** the plateau rule fails. This is not a verdict about sensors. The next lever is extending to 4 epochs (PI compute). |

**L_D = 85.0**, the PI's line, **unless** the PI records **84.5** in `AMENDMENT_ACK_NT.txt` before the run. 84.5 is the method-matched line: REFe model_final + route fix, without a learned PDM scorer, reads 84.46 (§14.2). The ACK's blob sha is pinned by the launch script; absent an ACK the line is 85.0.

### 4.2 Naming the split (BOTH; also reported under every other verdict)

All three shares come from one paired three-arm log-cluster bootstrap: the same log draws for A0, A1 and REFe, B 2000, seed 0.

| share | formula | what it says |
|---|---|---|
| **D** | (P − P_A0) / (88.99 − P_A0) | share of the full REFe gap closed by data (88.99 = mean of P2-s0 88.82 and P2-s1 89.15; their per-token CSVs `refe_stage2_p2_full` / `refe_stage2_p2s1_full`) |
| **D_M** | (P − P_A0) / (84.46 − P_A0) | share of the **method-matched** gap closed by data (`refe_navtest_final_routefix` CSV) |
| residual | 88.99 − P | reported as **scorer/route-goal (MEASURED ≈ 4.5 on REFe) + cameras + backbone**, the last two not separated without A2 / E-FOV |

**Committed reading of the shares for refcv9:**
* **D_M ≥ 0.8:** data closes most of the method-matched gap. Keep the refc trunk and port the PDM scorer first.
* **D_M < 0.5:** cameras/backbone carry most of it. Adopt REFe's backbone and multi-camera input (architecture change; PI).
* **0.5 ≤ D_M < 0.8:** both. A2 / E-FOV is the next experiment before any backbone switch.

*Illustration only; the A0 read sets the real values.* If A0 reads 72.8 (θ_0.5's value), then:
* D_M ≥ 0.8 ⇔ P ≥ 82.1;
* D_M < 0.5 ⇔ P < 78.6.

### 4.3 Boundary band → replicate (H-ESTIM-SEED-1)

* If |P − L_D| ≤ 1.0 or |P − 75.0| ≤ 1.0, **A1r is trained, selected by §5 and read on navtest**, and §4.1 is applied to the two-seed mean with no band.
* The band ±1.0 is about 2× the larger of two floors:
  * REFe P2's MEASURED training-seed difference, 0.33;
  * TransFuser's PUBLISHED training-seed SD on navtest, ±0.56 (`2406.15349`, banked).

## 5. Selection and the plateau rule

* **Snapshots:** EMA and RAW at steps 1,560 / 3,120 / 6,239 / 9,359 / 12,478. All are read **after** training ends, on the registered navtrain sample (R8_LEGAL, seeds 0 and 1). The box never evals while it trains.
* **Choice:** the **EMA** snapshot with the highest mean sample PDMS over the two seeds. An exact tie goes to the earlier snapshot. RAW is reported beside and never chosen (PI rule 2026-10-16).
* **Inference floor:** F_inf = the mean over the 5 EMA snapshots of |PDMS_s0 − PDMS_s1|.
* **Plateau rule:** EMA(12,478) − EMA(6,239) ≤ max(2 × F_inf, 1.0) PDMS on the sample.
* **One navtest read** of the chosen A1 snapshot, and one of A0. Both happen after the PI's go and after every control in §6 has passed. No second navtest read of A1, except A1r under §4.3.

## 6. Controls (each must read its known value, or the run STOPS and no verdict is written)

| # | control | known value |
|---|---|---|
| **K0** | Warm-start identity. The step-0 EMA and model saved by the fine-tune trainer (state_dict tensor-equal to the `dad4cb5b…` file), read by the **unchanged** bridge on the registered sample. | R8_LEGAL rows **bit-identical** to the banked MOD2 rows; PDMS **76.36 / 75.68** (s0 / s1), equal to the banked value at 0.01 |
| **K1** | Train = inference feed. For 200 selection-sample tokens (built by the training writer, never trained on) and 200 training tokens, the dataset item is compared with the bridge's inputs: frames (`pack_rows`, ST), `pose_hist` (`ego_history_poses`), `nav_cmd` (`nav_input8`), the speed row, the lift grid (`hires_lift_for_rig`). | frame sha16 equal; poses ≤ 1e-6; nav equal; speed row `(0, 0)` literal; grid ≤ 1e-5. Mutation `wrong_token` in the writer ⇒ **RED** |
| **K2** | Independent targets and labels. | 2 Hz targets vs the devkit's `Scene.get_future_trajectory` at 0.5–4.0 s: max \|Δ\| ≤ 1e-3 m. Spline at the sample times: \|Δ\| = 0.0. Synthetic straight constant-speed track ⇒ LANE_KEEP / CRUISE. Synthetic left circle R = 20 m at 8 m/s ⇒ TURN_L. All literals. |
| **K3** | Exclusion. | train logs ∩ the 38 selection logs = **0**; ∩ navtest logs = **0**. A mutation adding one selection log ⇒ **REFUSE** |
| **K4** | Map off. | the map-head classifier grads are exactly **0.0** over a 20-step smoke; their final rel-L2 change is reported |
| **K5** | Deliberate regression (eval side). | A1's R8_BLIND on the sample scores **below STOP 67.57**. `wrong_token` on A1 lowers the turn-cell PDMS. |
| **K6** | Estimator. | self-pair A1 − A1 = **0.0** with CI **[0, 0]**; token-shuffled pairing widens the interval |
| **K7** | The trunk actually trained. | trunk rel-L2 drift from θ_DEC > 0, reported; the P-GRAD share read at steps 1,000 / 6,239 / 12,478 |
| **K8** | Launch gate. | a PASS token for the `refcv8-nt` profile before step 1 (PI 2026-09-26) |

## 7. Estimator

* **Paired log-cluster bootstrap:** `navsim_bars8.pair_record` → `taniteval/ci.paired_log_cluster_bootstrap`, `aggregation = AGG_SINGLE_STAGE`, **B 2000, seed 0**, cluster **log_name**, sensitivity **nuplan_drive**, at least 8 clusters (`[PKG]/navsim_bars8.py:309-320`).
* "Separated" means the interval excludes 0 at log_name **and** at nuplan_drive.
* **Three variances, each named:**
  * this interval answers *another draw of LOGS*;
  * the inference seeds 0 / 1 answer *another inference run* (F_inf);
  * A1r answers *another training run*.
* A separated A1 − A0 difference is necessary, not sufficient, for a lever claim (`H-ESTIM-SEED-1`).

## 8. What this prereg forbids

* Changing L_D, 75.0, the band, the snapshot set, the selection rule or the plateau rule after any A1 read.
* Reading navtest before every control in §6 passes and the PI's go is recorded.
* Training on any selection or navtest log.
* Quoting the navtrain sample as the decision. θ_DEC was **selected on it** (MOD2 / DECLAM), so A0's 76.36 / 75.68 carry winner's-curse bias. Only navtest decides.
* Calling A1 a refc deliverable. A1 is a diagnostic arm: its agent classes and map supervision differ from PhysicalAI's, and its PhysicalAI forgetting is reported, not gated.
