# PREREG — E-P4-SCALE1: does encoder capacity buy the LATERAL channel?

`PRE-REGISTRATION, 2026-08-23, TanitAD_TrainingFlyWheel. Both outcomes committed
BEFORE the arm runs. Licensed by the champ30k decodability readout (Branch A,
V7_FULLSCALE_PLAN §0b).`

---

## 1. The question

champ30k (two-term + k1 + subspace32, 30k steps, **19.34 M params**) is the first
v6/v7 checkpoint to acquire environment information — `lead_gap_m` R²
**+0.1273 [0.0866, 0.1683]**, CI wholly above zero, against every v6F rung below
zero. But the acquisition is **narrow and longitudinal**:

| target | champ30k | verdict |
|---|---|---|
| `lead_gap_m` | +0.1273 [0.0866, 0.1683] | ✅ separated |
| `ego_speed` | +0.1823 [0.0847, 0.2686] | ✅ separated |
| `left_occupied` | AUC 0.5240 [0.464, 0.583] | ✗ chance |
| `right_occupied` | AUC 0.6144 [0.5423, 0.6838] | ~ weak |
| `vru_ahead` | AUC 0.4544 | ✗ below chance |
| `n_agents_log` | +0.1353 [−0.0292, 0.2582] | ✗ straddles 0 |
| `nearest_any_m` | +0.0144 [−0.054, 0.0658] | ✗ |
| `ego_yawrate` | **−0.4925 [−0.924, −0.2306]** | ⛔ anti-predictive |

⭐ **The encoder is 0.97 M parameters — 5 % of a model whose remaining deficit is
perceptual.** The predictor carries 7.57 M. If lateral/occupancy structure is
absent because the encoder lacks capacity to represent it, encoder scaling is the
intervention. If it is absent because the *objective* does not reward it, scaling
will raise nothing but the longitudinal channel it already has.

**E-P4-SCALE1 separates those two.**

## 2. The arm

Identical to champ30k in every respect except encoder capacity — the objective,
data, parity key, seed, schedule and every loss weight are unchanged.

| | champ30k | scale1 |
|---|---|---|
| `--enc-dim` | 128 | **256** |
| `--enc-depth` | 3 | **6** |
| `--enc-heads` | 4 | **8** |
| everything else | — | identical |

Parity: `physicalai-train-e438721ae894-w120-256x640cyl`, skip-hash `f09e44db`.
Seed 0. 30,000 steps. `--spectrum-accum 43` so the O6 gate CAN rule
(24 rows/call × 43 → ceiling 1031 ≥ 1024).

## 3. ⛔ Both outcomes, committed now

| result | reading | what we do |
|---|---|---|
| **LATERAL MOVES** — any of `left_occupied` / `right_occupied` / `vru_ahead` separates above chance and above the C-PIXEL floor, **and** `lead_gap_m` is retained (≥ champ30k's CI floor 0.0866) | capacity was the binding constraint on lateral perception | proceed to Scale-2 (≈150 M) — **gated on the held-out re-measurement P4-3b**, since every number here is in-sample |
| **LATERAL FLAT, LONGITUDINAL UP** — `lead_gap_m` improves but occupancy/VRU stay at chance | ⭐ **capacity is NOT the constraint — the OBJECTIVE is.** Scaling buys a better rangefinder, not a driver | ⛔ **STOP scaling this objective.** The lateral channel needs a loss term that rewards it (P4-3c), or the E-ENC-3WAY encoder question (P4-18). Do not spend Scale-2 compute |
| **NOTHING MOVES / lead_gap regresses** | more capacity under a fixed objective and fixed data does not help at all | stop; the binding constraint is data or objective, and Scale-2 is refuted in advance |
| **`ego_yawrate` rises out of the negative** | the anti-predictive lateral reading was a capacity artefact | note it — it would be the strongest single sign that lateral structure is learnable here |

## 4. Validity conditions

1. **The encoder is the only variable.** Any other diff invalidates the arm.
2. **Same probe, same 130 episodes, same folds** as champ30k — else the rungs are
   not comparable (falsifier 1 enforces this by refusing on a key mismatch).
3. ⛔ **Every number is IN-SAMPLE** (H-DEC-3: 100 % probe/train episode overlap).
   Comparisons are valid; absolutes are memorisation-permissive and are NOT
   quotable as held-out decodability.
4. **T0-DIAGNOSTIC.** Nothing here is a driving claim. E-TRUNK-3 §5 already
   established decodability does not translate into driving (REF-A had better
   decodability and drove 2.62 m worse).
5. Param budget 300 M enforced by `assert_param_budget`; X3 isolation strict.

## 5. Cost

champ30k: 14,774.7 s (4.1 h) at 19.34 M. scale1 is ≈3× the encoder and larger
overall. ⚠️ **Do not assume 3× wall-clock** — the trainer has ZERO DataLoader
workers, so throughput may be data-bound and barely move (SPEC §4.2, P4-6
unmeasured). Estimate 4–12 h on Thor, unmetered.
