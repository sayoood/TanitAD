# MM reading: DSCALE mixture arm UM_s0 (stage A, ONE seed): DILUTION-PARTIAL under UM-A1; BLOCKED as committed

**Card:** `CARD_DSCALE_MIX.md`, sha256 0bc241f5. **Amendment UM-A1** (K-TACSRC = exact replay equality), sha256 b2b2c4f0, landed 3fd4c797; the PI accepted it on 2026-11-18 ("Yes, read UM_s0").
**Read:** `read_mix/out/RESULT_DSCALE_MIX.json` (tool `ds_read_mix.py`), MEASURED on the dev-box CPU on 2026-11-18 (log `dscale_mix.log`).
**Tiers:**
- LONGITUDINAL, LATERAL and TACTICAL: T1-kin. That is a self-action open loop: m = 10, causal bucket, dead-reckoned nav, logged scene, deterministic planner.
- STRATEGIC: T0 at the window's first tick.
- ADE6 and goal F1: T0, beside, diagnostic only.

**Estimator:** paired episode-cluster bootstrap, B 2,000, seed 0, over 2,399 windows / 141 episodes. It answers "another draw of EPISODES" only. Training-run variance enters only through F_cell: the stage-A floor from ONE replicate pair (B_rep vs B_s0). The planner is deterministic, so no inference variance.

## 1. Label
* **As committed: BLOCKED (reach).** `tac_src_empty_batches` was 10, and the card's bar was 5 or fewer. The sampler replay reproduces 10 exactly, and the rows listed / other match too (1,223,523 / 2,708,437).
* **Read under UM-A1: DILUTION-PARTIAL.**
  * Deciding row 1, **TAC-LAT UM − UE: +0.0692 [+0.0283, +0.1092]**. It is separated better, but Δ ≤ F_cell 0.0767, so it is not a gain by the committed rule.
  * Deciding row 2, **TAC-LAT UM − BL: −0.0163 [−0.0433, +0.0121]**. It is not separated: the block-3 cost UE − BL −0.0855 is not visible between UM and BL.
* **Prediction H** (card section 1, written before any number): UM − UE in [+0.04, +0.10], with UM − BL not separated worse. **It held at one seed.** As the card said in advance (section 0 item 4), DILUTION-PARTIAL was the likely label under either hypothesis, because the effect is about 1.1 F.

## 2. Controls
- K1 self-pair: 0.0 [0, 0].
- K-RE (B3): 234 + 234 + 234 rows plus ADE6, mechanism and STRATEGIC rows, all reproduced exactly.
- K-EVALX: UM and the five references pass.
- Gate and post: 252/252 Adam state, banked checks.
- Problems: none.

## 3. Beside labels
* **STRAT-KEPT: false.** UM − BL STRATEGIC has no headline gain:
  * S1 pos_err@80 −0.6331 [−1.3560, +0.0080];
  * S2 skill@80 +0.0656 [−0.0453, +0.1702].
  * UM − UE S1 / S2 are not separated: +0.0915 [−0.2207, +0.4370] and −0.0533 [−0.1440, +0.0305].
  * UE's strategic gains over BL (S1 −0.7246, S2 +0.1189 in B3) are not carried over to UM − BL as gains at this seed.
* **TAC-LON-PRICE: false.** UM − UE TAC-LON is −0.0179 [−0.0521, +0.0158], not separated.
* **Other families:** by the committed rule (a family cost needs a headline cell), UM − UE has **no family cost** in LONGITUDINAL, LATERAL, TACTICAL or STRATEGIC.
  * ⚠️ However, five NON-headline cells in the **tactical band (knots 11-30, 2-6 s)** read COST in UM − UE. They are named here because they are the price signal:

| cell (UM − UE, T1-kin) | Δ [95 %] | F |
|---|---|---|
| tactical_2_6s heading_mae_deg | +1.0144 [+0.1091, +2.0890] | 0.6075 |
| tactical_2_6s yaw_rate_mae_degps | +3.3913 [+0.6367, +6.3063] | 1.2069 |
| tactical_2_6s curvature_mae_1pm | +0.0263 [+0.0043, +0.0511] | 0.0183 |
| tactical_2_6s accel_mae_mps2 | +0.2344 [+0.0233, +0.4352] | 0.0971 |
| tactical_2_6s dk_hw_abs_err_m (553 windows) | +0.2321 [+0.0272, +0.4571] | 0.0054 |

  Reading:
  - Removing the 10k rows' tactical supervision moved the 6-way TAC-LAT decision toward BL.
  - It also made UM's 2-6 s lateral geometry (heading, yaw rate, curvature) worse than UE's.
  - The operative band (0-2 s) is unchanged.
  - That is a lateral-geometry price of the rebalancing at one seed. It is outside the committed family rule, so it does not change the label.

## 4. The four families against the small-corpus base (UM − B_s0, beside)
UM keeps the data-scale gains that UE had over B_s0:
- LONGITUDINAL family gain: op speed −0.1010, op accel −0.4513, tactical along −0.5684 m.
- LATERAL headline cells: op yaw −0.3099 and op curvature −0.0049. Both are separated but inside their floors.
- Tactical-band LATERAL gains: heading −4.13 deg, yaw rate −14.59 deg/s, curvature −0.128 1/m.
- TACTICAL: TAC-LON +0.0413 is a gain. TAC-LAT +0.0654 [+0.0329, +0.0983] is separated but inside F.
- STRATEGIC: none.

## 5. STRATEGIC in full (PI 2026-11-16: the pause has ended)
- Route setting (RC-A, 6 cells) and route following (S1, S2) are printed per pair in `RESULT_DSCALE_MIX.md`.
- The only separated STRATEGIC gains are RC-A head_err@80 in UM − BL (−1.65 deg) and RC-A head_err@30 in UM − U_s0 (−0.32 deg). Neither is a headline cell.
- Detected information: n/a (no ctx head in this recipe).
- Derived goals: not in this arm.
- Strategic intent 6-30 s: not computable (no output beyond 80 m of arc).
- Fog: not scored in this card (weather is not a cell here).

## 6. Beside diagnostics (T0, never driving performance)
* ADE6: UM 1.977 m, UE 1.967 m, BL 2.063 m. UM − UE +0.0095 [−0.0467, +0.0653], not separated (F_A 0.0828).
* Goal macro F1:
  - VLM goals: UM 0.114, UE 0.125, BL 0.168.
  - Geometry goals: UM 0.582, UE 0.668, BL 0.565.
  - This matches the lateral-geometry price in section 3: without the 10k rows' tactical labels, UM's geometry-goal F1 falls from UE's level to near BL's.

## 7. Next lever (Rule Zero), as committed for DILUTION-PARTIAL
1. **The replicate pair UM_s1 + UE_s1** (seed 1), about 2 × 4.1 LoRA pod-h (E, from UE's MEASURED 0.2242 s/step).
   - This answers "would another TRAINING RUN say this" for UM − UE, which the episode bootstrap cannot (`H-ESTIM-SEED-1`).
   - The card launches nothing by itself, so this needs a short amendment and **the PI's go**.
2. The visit-form (sampler-weight) arm: **not triggered**, because UM − BL is not separated worse.
3. A lever the MM proposes beside it, for the PI to decide, NOT committed by this card: the **lateral-only weight** (card section 5.4, DILUTION-CONFIRMED item 3).
   - It applies λ to `lat` and the four lateral geometry bits only.
   - It aims to keep UE's 2-6 s lateral geometry while restoring TAC-LAT.
   - It would need its own amendment.

## 8. Banking
- UM's derived read outputs (T1-kin dump, T0 mechanism evals, T0 route passes, 183 files) are in private HF `Sayood/tanitad-v8f-dscale10k-runs`, `UM_s0/A/read_mix/um_read_outputs.tar`. Tar sha256 f8143482, equal to the hub's LFS sha256 (`BANKED_UM_READ.json`).
- The checkpoint is banked as UM_s0/A, sha256 39de0b05.
