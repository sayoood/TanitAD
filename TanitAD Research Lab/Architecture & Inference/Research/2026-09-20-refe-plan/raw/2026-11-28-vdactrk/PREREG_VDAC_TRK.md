# PREREG_VDAC_TRK: a vision-predicted drivable-area veto on C1's TRACKED proposals (H-VDAC-TRK-1)

**Status: FROZEN by the Master Mind by sha256 (`PREREG_VDAC_TRK.sha256`) on 2026-11-28, before any VDAC-TRK code, head, dump or number exists.**
- It succeeds H-VDAC-P6-1 (`PK/raw/2026-11-09-scorer1p6-prereg/vdac/PREREG_VDAC_P6.md` + A1-A5). It reuses that card's instruments and changes the BASE, the GATE and the RULE.
- PI ruling 2026-11-28, "Vision-predicted map": **no HD map at inference, not even as an upper-bound arm**. The HD map stays admissible for labels.

## 0. What the MM has seen before this freeze (disclosure)
All of the following are MEASURED. The C1 descriptives are T1, on C1's banked harness-exact navtest table `full_C1_s*of2.jsonl` (`tr` = tracked sub-scores per proposal), landed under `PK/raw/2026-11-28-vetodiag/`. **No head-based navtest number exists for C1.**

**The C1 descriptives:**
- 336 zeroed ship picks (NC 138, DAC 198) with a passing alternative, regret 2.723.
- C1's own NC/DAC logits detect them at AUC 0.763. The SAFE-HEAD heads detect them at 0.766-0.769, and their OOF vetoes lose.
- Perfect tracked-DAC checker: +1.532 [1.147, 1.974], 198 better / 0 worse. Planned-proposal DAC checker: +0.538 [0.314, 0.781], 0.351 of that.
- i.i.d. checker-error sensitivity (`DAC_SENSITIVITY.md`):
  - false alarms are nearly free: e_fa 0.20 keeps +1.536;
  - misses cost more than linearly: e_miss 0.10 gives +0.926, 0.25 gives +0.573.

**The P6 predecessor** (`PREREG_VDAC_P6*`, GOALS rows H-VDAC-P6-1 / D-VDAC-*):
- **S0:** a true-raster pose-only hard veto, R 0.348 of O-DAC (navtrain, in-sample).
- **G0-TRK:** NAVSIM's LQR-tracked footprint reproduces the table's DAC on 777,344 / 777,344 proposals; tracked and raw verdicts differ on 4.34 %.
- **Heads on a frozen P6 trunk**, H48 perception:
  - HA: S1-Q 0.8332, IoU 0.7802;
  - HB (LoRA trunk copy): S1-Q 0.8376, IoU 0.8113;
  - HC: S1-Q 0.8346, IoU 0.7757;
  - true-field ceiling 0.9408.
- **S2 never ran:** S1-Q's 0.90 gate stopped every head.

**What this card changes:**
- The perception AUC no longer gates. Navtest PDMS decides, through a recall-oriented margin calibrated on navtrain. The rationale is the i.i.d. sensitivity above, derived from oracle errors and not from any head output.
- ⚠ Disclosed: the MM knew the P6 heads' S1-Q values when choosing this. That is why this is a NEW card with new bars, not a re-read of the old one.

## 1. Question and hypothesis
**Question:** does a drivable field predicted from C1's own cameras, checked on the TRACKED execution of each of C1's 64 proposals as a hard veto, raise C1's navtest PDMS beyond the selection-lever bar without a separated-worse sub-score?

**H-VDAC-TRK-1:** yes, because DAC failures are 198 of C1's 336 zeroed picks, and neither C1's scorer nor a fine-tuned scorer head detects them.

## 2. Admissibility (the binding rules, checked)
- **Inference inputs:** C1's cameras (vision) and C1's own proposals. The tracker uses NAVSIM's PDMSimulator (LQR + kinematic bicycle) and reads only the plan and the t0 ego status. `PREREG_TRKSCORE.md` declares that; it is INHERITED and was re-checked by `FACTS.md` §4.
- **No HD map and no map_api at inference.**
- **Labels:** the nuPlan-map SDF raster in `aux_drivable/label_store` (navtrain, 1,192 logs, no navtest overlap; `FACTS.md` §2), and NAVSIM's harness for PDMS. The HD map is allowed here, on the label side only.
- **Situation classifier:** the field never reads its output.

## 3. Arms

**Heads** (trained; the planner is NEVER changed: C1's weights, logits and proposals stay bit-identical):

| head | recipe | role |
|---|---|---|
| **HT-s0** | `PK/refe/aux_drivable_head.py` on C1's FROZEN `visual_ctx` [B,7680,256]; S1c recipe (10,000 updates, batch 8, visual tap, depth 2, feed off) with `--ckpt` = C1 (sha256 f09949e2...); seed 0 | primary |
| HT-s1 | same, seed 1 | replicate (training variance of the head) |
| HT-SHUF | seed 0, label keys deranged, 2,000 updates | deliberate regression |

- **Head training logs:** every `label_store` navtrain log except the calibration set **HC40** (section 5).
- **Frozen trunk, not a LoRA copy (the HA family):**
  - it is the cheapest that runs on the dev-box RTX 4060;
  - S1-Q was flat across HA/HB/HC (0.833/0.838/0.835);
  - HB's LoRA copy is the named next arm, under its own card.

**Selection arms** (CPU, no training; each changes only which of C1's proposals is picked):

| arm | rule | role |
|---|---|---|
| **VT** (HT-s0, margin m*_0) | veto k if d_k < m*; pick = argmax C1 agg32 over the survivors; all vetoed ⇒ C1's pick | **primary** |
| VT-r (HT-s1, its own m*_1) | same | replicate |
| VT-SHUF (HT-SHUF, m*_0) | same | deliberate regression: must NOT pass P1 |
| VT-0 (HT-s0, m = 0) | same, margin 0 | reported only |
| O-DAC-T | veto k iff `tr.dac[k] = 0` | reported ceiling (+1.532 known), never deployable |

**d_k** is the minimum, over the 41 tracked states (t = 0, 0.1, ..., 4.0 s) and the 4 ego footprint corners, of the head's signed distance read bilinearly. That is exactly `feed_scalars`' "min" in `aux_drivable_head.py`, with the ego footprint and the 0.1 s states of the G0-TRK instrument (`vdac/trk/trk_dump.py`). **agg32** is `refe/safehead_data.agg32` on C1's logits, unchanged.

## 4. Gates and controls (each can FAIL; any FAIL = INVALID, fix and re-run that step; no bar moves)
- **G0-TRK-C1:** `trk_dump.py` run on C1's 12,146 × 64 navtest proposals. NAVSIM's DAC rule on the tracked states must equal `tr.dac` on **777,344 / 777,344**. A second dump of the first 100 sorted tokens must be bit-identical.
- **CTRL-ID:** an all-drivable field changes 0 picks, with Delta exactly 0.0. With the veto off, the read reproduces C1's pick on 12,146 / 12,146 and the mean 92.5548 (±5e-5).
- **CTRL-PLAN:** C1's logits and proposals in the dump equal the banked TRK dump (`trk_c1.npz` `s_plan`, k_ship) bit-for-bit, so the head pass does not perturb the planner.
- **C-SHUF:** HT-SHUF's drivable IoU on HC40 (x in [0, 40) m, mask logit > 0) must be < 0.50, else INVALID.
- **C-FLIP:** HT-s0's field y-flipped must give a lower HC40 calibration gain than unflipped at m*_0, else INVALID.
- **CTRL-T (harness confirmation):** for VT's picks, seam rows are built from C1's planned-proposal dump (`stage2_p5c_c1.proposals.npz`). They are scored by the pinned harness, and the harness CSV must equal the table at VT's picks to 1e-6 on every token. Until that passes, every VT number is PROVISIONAL.
- **Launch gate:** every head training needs a `launch_gate.py` PASS token on its own profile (`refe_vdt_s0 / _s1 / _shuf`, with G-DVB entries for every lever: ckpt, seed, steps, label derangement). No token, no training.

## 5. Calibration of the margin (navtrain only; navtest is never read for it)
- **HC40:** 40 navtrain logs from `label_store`, absent from navtest, being the first 40 by sha256(log name). The list and its sha256 are written beside this card by the build, before any head trains. The heads never see HC40's labels.
- **On HC40:**
  - C1 runs (logits, proposals, `visual_ctx` → each head's field);
  - C1's proposals are tracked (trk_dump) and labelled by NAVSIM (the harness-exact per-proposal labeler `sh_label.py`).
- **m\*** = argmax over the fixed grid {-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0} m of HC40's mean PDMS gain of the VT rule vs C1's pick. Ties go to the smaller m. It is chosen per head (m*_0 for HT-s0, m*_1 for HT-s1).
- **No m with a positive point gain ⇒ outcome CAL-NULL** (no navtest read).
- **Reported, not deciding (HC40):** IoU, S1-Q-style AUC of -d_k for tracked DAC = 0 among each token's top-8 proposals by agg32, the miss and false-alarm rates at m*, and the HC40 gain curve over the grid.

## 6. The navtest read (E3: 12,146 tokens, 136 logs, T1-class open loop, PDMS × 100 by table lookup on `tr`; then CTRL-T)
**Estimator:** the paired log-cluster bootstrap (`eval/p5c/p5c_read.py` Boot), 10,000 draws, seed 20260927, 95 % percentile. It answers *another draw of logs*. The head's training variance is read by VT-r. C1's training variance is not resampled (C1 is a fixed planner). Inference is deterministic, which CTRL-PLAN and a second pass assert.

| id | metric (vs C1) | bar |
|---|---|---|
| **P1** | VT PDMS | CI-lo > 0 |
| **P2** | VT PDMS | point ≥ **0.2316** (= 2 · max(F, F_REFe 0.1158), the selection-lever bar) |
| **M** | the share of picks whose `tr.dac` = 0, VT | CI-hi < 0 (the mechanism) |
| **G** | NC, DAC, TTC, EP, C of the pick, VT | none separated worse (CI-hi < 0 ⇒ fails) |
| **R** | VT-r PDMS | point > 0, with no separated-worse PDMS or sub-score |
| **X** | VT-SHUF PDMS | must NOT meet P1 (else INVALID: the rule gains without map information) |

**Reported, not deciding:**
- VT-0;
- O-DAC-T and each arm's capture ratio;
- tokens changed, split by whether C1's pick had DAC 0 (recoveries vs false moves);
- the NC-zero pick share and EP of the pick.

**The four families (binding):** LONGITUDINAL, LATERAL, TACTICAL and STRATEGIC run on VT's and VT-r's seams through the `run_fam_chain_ds.sh`-style role mapping, SMOKE=50 first. They are required before ADOPT, each family with its own CI, never pooled.

## 7. Outcomes (committed, in precedence order; each non-ADOPT outcome names its next arm, per Rule Zero)
1. **INVALID:** any gate or control in section 4 fails, or X passes P1. Fix the instrument and re-run. No bar moves.
2. **CAL-NULL:** no positive HC40 margin. The field does not reach C1's failing proposals even on navtrain.
   - Next: HB (LoRA trunk copy for the head, P6 IoU +0.031) under its own card, and the agent-check line (D-VDAC-6) for NC.
3. **ADVERSE:** VT's CI-hi < 0, or G fails. False vetoes or correlated field errors dominate, against the i.i.d. sensitivity.
   - No margin is re-chosen on navtest.
   - Next: HB under its own card if HC40's gain at m* was positive; else the DAC-veto family closes on C1 and the agent check carries the line.
4. **NULL:** P1 fails, M fails, no harm. Next: HB, then the agent check.
5. **MECHANISM-SMALL:** M holds, but P1 or P2 fails, and G holds. The veto removes DAC-zero picks, but not enough PDMS.
   - Next: HB (a better field), with VT-r reported.
6. **SINGLE-SEED:** P1, P2, M and G hold, but R fails. The result is *separated, single head seed, not yet a lever claim*.
   - Next: HT-s2, read by these bars unchanged.
7. **ADOPT:** P1, P2, M, G, R and CTRL-T hold, X fails P1, and all four families are present with none separated worse on a binding family component.
   - VT enters C1's selection; deploying is the PI's call.
   - Next: goal status vs 94.55, then the agent check (NC) on top of VT.
- **Unnamed** if none matches; reported as such.

⛔ Every outcome is reported as written. A FAILED arm leaves its next arm above, and no bar, grid or log set moves after a number is seen.

## 8. K ledger
- Navtest table reads under this card: VT, VT-r, VT-SHUF and VT-0, so **K + 4**.
- O-DAC-T is already reported (`FILTER_ORACLE.json`) and adds none.

## 9. Build items (before any training; each with its own tests and a deliberate-regression check)
- **B1. Head trainer:** the S1c trainer with `--ckpt` C1 and HC40 excluded from the labels, plus three launch-gate profiles with G-DVB entries.
  - G-LIVE (a real-data smoke) must show a non-zero gradient on the head and exactly 0.0 on C1's trunk.
  - It must fit the RTX 4060's 8 GB at batch 8; if it does not, gradient accumulation to an effective 8 is an execution amendment, not a recipe change.
- **B2. `vdt_dump.py`:** one pass of C1 per token emitting logits, proposals and each head's field.
  - It runs on HC40 and on navtest, and CTRL-PLAN binds it.
  - C1's navtest `visual_ctx` is not banked anywhere (`FACTS.md` §6), so the head runs inline.
- **B3. Tracker dump:** `trk_dump.py` (G0-TRK instrument) on C1's proposals, 41 states at 0.1 s, plus G0-TRK-C1.
- **B4. HC40 labels:** `sh_label.py` (the harness-exact per-proposal labeler) on C1's HC40 proposals.
- **B5. `vdt_read.py`:** the calibration, the navtest read, every control above, and the outcome logic.
  - Tests include a synthetic all-drivable field (Delta 0), a y-flipped field and an always-veto field (falls back to C1).

## 10. Compute and data (needs the PI's go before B1's first training; no pods, PI 2026-11-22)
- **Dev-box RTX 4060 estimates** (ESTIMATED from the P6 runs: one 10k-update head ≈ 3.2 L40-h; this box assumed 3-4× slower):

  | step | estimate |
  |---|---|
  | HT-s0 + HT-s1 | ~20-26 h |
  | HT-SHUF | ~2-3 h |
  | `vdt_dump` on navtest (12,146 tokens) | ~4.3 h |
  | `vdt_dump` on HC40 | ~1-1.5 h |
  | **total** | **≈ 28-35 h of GPU**, queued after ENS and DATA-SCALE |

- **The open data question:** the head trains on `label_store` frames, and the dev box holds no navtrain camera blobs at that scale (x150 streams frames log by log and deletes them). B1's first step measures the frames and bytes the 10k × 8 draws need, and the stream cost. The PI then chooses between:
  - (a) streaming on the dev box;
  - (b) a short pod;
  - (c) restricting the training logs to frames already streamed. (c) is an amendment that must be frozen before training.

## 11. Registration
`Project Steering/GOALS_AND_CLAIMS.md` rows H-VDAC-TRK-1 (CARD FROZEN) and its correction, in the same commit as this card.
