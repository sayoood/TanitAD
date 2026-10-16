# PREREG_AUXVETO — can a drivable-area veto from the aux head's PREDICTED raster recover the picks the scorer is confidently wrong about?

> ⭐ **IN FORCE from its landing commit. Written by the Master Mind on 2026-10-16 (~09:00 Berlin), BEFORE any pick, ΔPDMS or veto outcome has been computed on any raster.** The feasibility study (`raw/2026-10-16-auxveto-feasibility/`) was barred from computing one, and its code was checked for it.
>
> After landing, nothing below may change except through a dated amendment appended at the end that says what it changes and why.

## 1. Why this arm exists (Rule Zero: the next lever after `D-REFE-SELRULE-FIT`)
- **MEASURED** (`raw/2026-10-16-selrule-dac/`): on B's picks with true DAC = 0, the u400 scorer's own p̂_DAC has a median of 0.93. No threshold on that head reaches them, and an oracle DAC veto is worth +2.13 PDMS on RL-val.
- **The candidate lever is an INDEPENDENT drivable-area signal.** The aux head predicts a BEV drivable-area raster. A geometric check of each proposal's ego footprint against that raster (geoDAC) can veto proposals whatever the scorer believes about them.
- **The instrument is validated** (MEASURED, `raw/2026-10-16-auxveto-feasibility/AGREEMENT.json`, 67,712 proposals). On the GT raster, pose-only geoDAC agrees with NAVSIM's DAC on 87.6 % [85.9, 89.4] of proposals (κ 0.713). The cap is the LQR tracker (27.8 % of NAVSIM's DAC failures exist only in the tracked states). The raster-only error is a ≤ 0.25 m boundary band.
- **This read answers two questions:** does the route work at all (GT raster), and does it work at today's head quality (predicted raster, held-out IoU 0.675)? Together they say what S1c's head must deliver, in PDMS rather than IoU.

## 2. Population, data and base (all exist; `INVENTORY.md`)
- **Tokens:** the 1,058 C3 tokens (`raw/2026-10-11-c3-selection-headroom/tokens.json`, sha256 `03072566…e2254d`), in two strata:
  - **Z (census):** ALL 529 navtest tokens whose u400 pick has NC × DAC = 0;
  - **N (sample):** 529 tokens drawn uniformly (seed 20261011) from the 11,617 non-zeroed tokens.
- **Proposals, scores, table:** u400 (SFT-4 arm B, 400 updates). The 64 proposals, the logits `[1058, 64, 7]` and the NAVSIM v1.1 per-proposal table `[1058, 64, 6]` come from `table.npz` (sha256 `cf30519051bea03b…`). Index by token string, never by row position.
- **B** = the u400 scorer's pick, `rl_val_eval.select(…, "navsim_v1_lane", space="logit")`. It must reproduce `pick` on 1,058 / 1,058, or the read is INVALID.
- **PDMS of any pick:** by table lookup (NAVSIM v1.1, non-reactive). Tier: the programme's NAVSIM tier.
- ⚠️ **Transfer caveat, stated now.** The base is u400, not P2 or P6. A result here is a statement about the u400 scorer's failure set. P2's own table covers a different 509 tokens.

## 3. The rasters (arms)
| arm | raster | role |
|---|---|---|
| **V-PRED** (PRIMARY) | the A2 probe head's held-out cross-fold prediction (10,000 updates, seed 0) | today's head quality |
| V-GT | the GT label-store raster (store sha256 `985a3bc7…`) | ceiling of the pose-only raster route |
| V-PRED-2k, V-PRED-400 | the same head at 2,000 and 400 updates (held-out) | reported: the head-quality curve |
| O-DAC | NAVSIM's own DAC subscore (oracle) | reported: the absolute ceiling of any DAC veto |
| V-GT-TRACKED | GT raster on NAVSIM's LQR-tracked states | reported only (see §7) |

**Which raster output is read.** The MASK, `logit > 0`, for every predicted arm.
- The rule needs a signed distance. For EVERY arm (GT and predicted alike) it is computed from the binary mask by a Euclidean distance transform, positive inside the drivable area, on the raster's own grid.
- The head's SDF channel is NOT read: it disagrees with the mask sign on 2.0 % of cells.
- **Identity check (must hold, else INVALID):** the mask read by the read script reproduces A2's held-out IoU [0, 40) **0.675218** (±1e-6) for V-PRED.

## 4. The rule (one rule, no fitted parameter)
For each token and arm:
1. A proposal is **VETOED** if any footprint sample over its horizon has a signed distance < **−0.25 m**, i.e. lies more than 0.25 m outside the arm's drivable area.
   - Footprint, pose sampling and frame are exactly those of `geodac.py`: pose-only, raw poses. The footprint is half-length 2.588 m, half-width 1.1485 m, centre offset 1.461 m.
   - **The 0.25 m margin is an INSTRUMENT property fixed before any outcome.** It is the band that holds all 3,148 GT-raster false alarms on tracked states (`AGREEMENT.json`). It is not tuned on any pick.
2. **The pick** is the proposal with the highest B score (the same `select` rule) among the NON-vetoed proposals.
3. **If every proposal is vetoed,** the pick is B's pick (no change).

**Reported, never gating:** the margin curve m ∈ {0, 0.5, 1.0, 2.0} m for V-PRED and V-GT.

## 5. The estimand and the estimator
- **Primary:** the navtest-population ΔPDMS of V-PRED vs B (×100), by the stratified estimator
  Δ = (529 / 12,146) · Δ_Z + (11,617 / 12,146) · Δ_N,
  where Δ_Z is the mean over all Z tokens and Δ_N the mean over the N tokens.
- **Interval:** paired log-cluster bootstrap over the 123 C3 logs, 10,000 resamples, seed 20260927.
  - In each resample, Δ_Z and Δ_N are recomputed over the tokens of the resampled logs, and the weights stay fixed.
  - 95 % percentile CI.
- **Which question the CI answers:** *another draw of logs*. ⛔ It is blind to the head's TRAINING variance: the head has one seed (H-ESTIM-SEED-1). It is also blind to raster inference variance, which is zero (deterministic).

## 6. Committed outcomes (decision list, read top-down, on V-PRED's primary Δ unrounded; V-GT's Δ by the same estimator)
| row | condition | reading | next arm (committed now) |
|---|---|---|---|
| **AV-GO** | V-PRED lower bound > 0 AND point ≥ +0.30 | the predicted-raster veto recovers enough confident DAC failures to matter at today's head quality | **"separated, single head seed, not yet a lever claim"**: (1) a seed-1 head replicate (dev-box GPU, ~1.6 h for 5 fold units per `COST.md`), read by this same rule; (2) the S1c head's rasters through this rule; (3) a wiring spec into REFe's selection at inference |
| **AV-PROMISING** | V-PRED lower bound > 0 AND point < +0.30 | real but small | S1c (a better raster) through this rule, plus the seed-1 replicate |
| **AV-HARM** | V-PRED upper bound < 0 | false vetoes on good picks cost more than the recoveries | read V-GT: if the GT ceiling is positive, the lever is raster quality (S1c), else AV-NULL's next arm |
| **AV-CEILING-ONLY** | V-PRED not separated AND V-GT lower bound > 0 | the route works with a good raster, and today's head is not good enough | S1c through this rule. The head-quality curve (400 / 2k / 10k) says how far the IoU must move |
| **AV-NULL** | V-PRED not separated AND V-GT not separated above 0 | a pose-only raster veto does not reach the scorer's DAC failures | the TRACKED variant, IF ruled admissible (§7), else the representational route (P6's trunk; P7.1's better-DAC-head step) |

## 7. Admissibility (vision-only rule; the goal / situation rules do not apply)
- V-PRED reads only the camera-derived raster and the proposals: **admissible at inference**.
- V-GT and O-DAC are oracles, never deployable.
- **V-GT-TRACKED** needs the initial ego state to run NAVSIM's tracker at inference. It is **reported only**. Whether a tracked veto is admissible at inference is a Master Mind and PI question, to be asked only if AV-NULL fires. NAVSIM's agent input carries ego velocity and acceleration, but the tracker also needs state it does not provide.

## 8. Reported with every read (four families; STRATEGIC is PAUSED, PI 2026-09-19)
- **LONGITUDINAL:** Δ EP, Δ TTC (stratified, paired CI).
- **LATERAL:** Δ DAC, Δ DDC (stratified, paired CI). Also: the Z-stratum recovery (how many of the 529 zeroed picks become non-zero), and the false-veto rate on N (B's pick vetoed although NAVSIM DAC = 1).
- **TACTICAL (selection quality):** random / B / best-of-64 / each arm, per stratum.
- **STRATEGIC:** PAUSED.
- Also, per arm: Δ NC, Δ comfort; the share of tokens where every proposal is vetoed; the per-city split (descriptive, never gating).

## 9. Validity checks (any failure ⇒ INVALID, fixed before the read is reported)
1. B reproduces `pick` 1,058 / 1,058.
2. V-PRED's mask reproduces IoU 0.675218.
3. **Zero control:** an all-drivable raster vetoes nothing, and Δ = 0.0 exactly.
4. **O-DAC sanity:** O-DAC's Z-stratum recovery is ≥ the recovery of every raster arm (the oracle bounds the rasters).
5. **Mutation:** a y-flipped V-GT raster must not read as well as V-GT. It is reported, and must fall below V-GT's point estimate.

## Amendment A1 (2026-10-16 ~09:30 Berlin, Master Mind; written BEFORE S1c has trained and before any arm of this prereg has been read): arm V-S1C
**What it adds.** One arm, **V-S1C**: the S1c head's (`eval/PREREG_S1C.md`, arm `H1u10k-s0`) navtest forward rasters on the 1,058 C3 tokens
(`aux_navtest_forward.py`).
- It is read through §3–§9 unchanged: the MASK (`logit > 0`), the distance transform, the −0.25 m margin, the same rule, estimator and validity checks.
- Its identity check replaces §9.2's: the read reproduces S1c's own C3 P1 IoU [0, 40) from `aux_s1_readout.py`.

**Its role.** S1c's head is trained on navtrain only, so V-S1C is the first **deployable** raster arm. V-PRED's head was trained on navtest
folds; it measures the route, not a deployable component.

**Decision rule.** V-S1C is read by the same §6 decision list, as its own row, after V-PRED's. It cannot change V-PRED's row.

**What it does not change.** Every other line above.
