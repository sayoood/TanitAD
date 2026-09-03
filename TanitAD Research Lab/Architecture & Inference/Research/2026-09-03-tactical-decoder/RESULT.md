# RESULT — the refav1 TACTICAL DECODER, and why a correctly-decoded turn still loses

**Package:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-tactical-decoder/`
**Date:** 2026-09-03 (Europe/Berlin; pods and logs are UTC) · **FlyWheel:** Architecture & Inference
**Register:** proposes `D-REFAV1-TAC-DECODER`, `D-REFAV1-SEED-GOAL-MISMATCH`,
`D-REFAV1-TAC-LABEL-SHARE`, `D-REFAV1-TAC-NAV-ECHO`, and one CORRECTION to the R28 brief.
**Pre-registration of the fix:** `Project Steering/PREREG_TACTICAL_DECODER.md` (`H-REFAV1-TAC-DECODER-1`).
**Compute:** dev-box RTX 4060 only — 2 arms × 140 windows × 2 nav conditions = 560 forward passes,
MEASURED wallclock **182.7 s + 188.0 s**. Everything else is 0 GPU. ⛔ **No pod, no Thor,
no training launched, no model file edited.**

---

## ⚠️ ESCALATE FIRST — three things the programme must act on today

1. **THE PLANNER CANNOT FAITHFULLY CHASE ANY MANOEUVRE EXCEPT "DO NOTHING" — QUANTIFIED.**
   ℹ️ **Credit where it is due: this residual was ESCALATED QUALITATIVELY hours before this
   package, by the steer-conversion agent in commit `4139203`** (*"a 2 s plan is judged against a
   6 s goal, so the planner's own seed cannot reproduce its own goal — a plausible reason a correct
   turn buys one float32 ULP"*). **What is new here is the measurement, and it is worse than
   "plausible".** The goal is rolled from the FULL 30-step canonical control subsampled onto the
   tactical grid (`_imagine_tactical_goal`, operative indices `[0,3,…,27]`) while the seed is
   TRUNCATED to `plan_steps = 10` (`refa_v1.py:1932`). **MEASURED: 62 of 64 (lat, lon) token pairs
   give the goal and the seed DIFFERENT tactical action sequences (48/64 differ in curvature
   alone).** The only two that agree — `(LANE_KEEP, CRUISE)` and `(ABORT_LC, CRUISE)` — are the
   all-zero control. `TURN_R` over-rotates its own goal by **82.5°**; `LANE_CHANGE_R` by
   **−27.1°**; `NUDGE_R`'s S-curve is stretched 2.5× in time so **8 of 10** tactical steps carry
   the wrong curvature; `LANE_KEEP × BRAKE_TO` differs on **9 of 10** accel steps and ends at
   **6.66×** the deceleration the goal ends on (−0.5811 vs −0.0872 m/s²).
   ⚠️ **AND THE `cost_time_grid` REPAIR IN `4139203` DOES NOT CLOSE IT**: under the new
   `cost_time_grid="tactical"` the mismatch is **still 62/64 (48/64 in curvature)**, because the
   cause is the TRUNCATION to `plan_steps`, not the regrid. Anyone reading "the cost's time grid is
   repaired" as closing this residual would be wrong.
   ⇒ **The cost is not measuring the wrong thing precisely. It is measuring the wrong pair.**
   It must be closed before any decoder or metric arm is spent.
2. **THE R28 BRIEF'S LOSS-WEIGHT PREMISE IS WRONG, AND THE CORRECTION MATTERS.** `loss_feat_tac`
   at 0.1–0.8 % of the loss is the tactical **FIELD PREDICTOR's** MSE, not the decoder's term.
   The decoder's term is `w_tac_label · mean(CE_lat, CE_lon)`, **MEASURED at 13.59 % (incumbent) /
   20.51 % (ep2)** of the label+feature sum on the eval windows. ⇒ **"the head is unsupervised" is
   REFUTED.** Raising `w_tac_label` is not the lever.
3. **NO TRAINING LOG IN THE PROGRAMME HAS EVER CARRIED THIS TERM.** `refa_v1_train.py:699-734`
   writes `loss_feat_op / loss_feat_tac / loss_feat_str` and **no `loss_lat_label` or
   `loss_lon_label` key**. Verified by CONTENT across all **nine** banked refav1 `train_log.jsonl` files (every row
   parsed, not grepped): **0 rows** carry any of `loss_lat_label`, `loss_lon_label`,
   `loss_route_label`, `loss_feat_str_ext`. The one term the PI made mandatory ("It must be trained with this data",
   2026-08-31) is the one term with no instrument on it. Adding two floats to that dict is free.

---

## 0. Scope, tier, and what is NOT claimed

| | |
|---|---|
| **TIER** | **T0** throughout — banked open-loop step-1,000 checkpoints, 140 windows / 20 episodes, stride 10. No simulator, no closed loop, no T1 driving claim. |
| **EVIDENCE CLASS** | **MEASURED** on every number below unless stamped otherwise. Two numbers are stamped **INFERRED** (§4, the train-corpus in-band fraction) and they say so in place. |
| **Estimator** | Episode-cluster bootstrap (percentile, **full-set** point estimate), `taniteval/taniteval/ci.py` construction, 10 000 draws over the 20 episodes. `overlapping_holdout_se` is never called. |
| **Nulls** | Label permutation, reported in TWO forms — `free` (row-independent) and `clustered` (episode blocks permuted whole). ⚠️ **The clustered one is quoted**: the v7.2 label is emitted once per CLIP, so an episode's in-band rows carry the SAME label and a free permutation is anti-conservative. |
| **NOT claimed** | Any statement about closed-loop driving; any statement about the training stream's own loss ratio (the eval-window ratio is what was measured); any claim that the tactical latent is unlearnable (see §6's function-class caveat). |

**Sources.** Model/consumers read at `stack/tanitad/refs/refa_v1.py`, `stack/tanitad/refs/refa_v1_plan.py`,
`stack/tanitad/models/fourbrain.py`, `stack/tanitad/models/vocab_v7.py`, `stack/tanitad/data/refav1_loader.py`,
`stack/scripts/refa_v1_train.py`. Artifacts re-read from
`.../Research/2026-09-03-refav1-step1000-read/` and `.../Research/2026-09-03-refav1-cost-surface/raw/`.
Labels: `_s2build/release/v72/s2_labels_v7.2_{train,eval}.jsonl.gz`, eval md5
`aa12c948f062181c3297265b51526ec5`.

---

## 1. (a) What the decoder outputs, and on what clock

| # | fact | class · source |
|---|---|---|
| 1 | **Two independent 8-way softmaxes**, not one. `out["lat_logits"] = lat_head(intent)`, `out["lon_logits"] = lon_head(intent)`. The legacy mixed 5-way head is emitted only as `legacy_mixed_maneuver_logits_DO_NOT_USE` and is consumed nowhere. | MEASURED — `refa_v1.py:1313-1318` |
| 2 | Each head is `LayerNorm(256) → Linear(256, 8)` — a **linear read-out of one 256-d vector**. | MEASURED — `refa_v1.py:1030-1033`; `d_intent = 256` (`config.py`, ckpt `tactical_cfg`) |
| 3 | That vector is `intent = intent_proj(norm(x[:, -1]))` — the tactical transformer's **last** state of the 4-frame observed window, FiLM-conditioned on the strategic `ctx`, plus `nav_to_intent(nav_emb)` when `nav_inject`. | MEASURED — `fourbrain.py:339-343`; `refa_v1.py:1153-1157` |
| 4 | Vocabulary v7.0, 8 lateral × 8 longitudinal: `LANE_KEEP, LANE_CHANGE_L, LANE_CHANGE_R, ABORT_LC, NUDGE_L, NUDGE_R, TURN_L, TURN_R` × `FOLLOW, CRUISE, YIELD_MERGE, BRAKE_TO, CREEP, HOLD, ADAPT_SPEED_FOR_CURVE, ACCELERATE`. | MEASURED — `vocab_v7.py:290,298`; verified live by the audit tool, which refuses on drift |
| 5 | **CLOCK: one (lat, lon) pair per MPC tick, argmax'd once and HELD CONSTANT over the whole 6.0 s goal.** `canonical_controls` expands the single token pair into a 30-step (a, κ) profile; the goal is one 10-step tactical rollout of it. | MEASURED — `refa_v1.py:1711-1761` |
| 6 | Nominal recompute cadence is `TacticalPolicyConfig.cadence = 5` operative ticks = **1.0 s** (strategic: 20 ticks = 4.0 s). In this eval every window is a fresh tick, so cadence never binds. | MEASURED — `config.py:124,146` |

⇒ **The decoder cannot express "straight, then turn."** One token covers 6 s. Any manoeuvre that
begins part-way through the horizon is representable only as a whole-horizon manoeuvre or not at all.

---

## 2. (b) Is it trained at all — and what share of the loss does it carry

**It is trained, and the term is an order of magnitude larger than the R28 brief assumed.**

| # | fact | class · source |
|---|---|---|
| 1 | The term exists: `loss_{lat,lon}_label = F.cross_entropy(out[key], lbl)`, added as `w_tac_label · mean(...)`. `-100` is the ignore marker; an all-ignored family is **skipped, not averaged** (an all-ignored CE is NaN). | MEASURED — `refa_v1.py:1578-1595` |
| 2 | `w_tac_label = 0.1` (`w_str_label = 0.1`) in **both** banked checkpoints' recorded config. | MEASURED — `t1_dump_manifest.json` `model.cfg`; `ckpt_ep2/config.json` `cfg` |
| 3 | The live ep2 run passed real labels: `--labels /home/nvidia/data/v72/labels/s2_labels_v7.2_train.jsonl.gz`, `allow_unlabelled: false`. The trainer **refuses** `--cache` without `--labels`. | MEASURED — `ckpt_ep2/config.json` `args`; `refa_v1_train.py:506-513` |
| 4 | The head has **moved off init**: lat `W_fro` **1.8043** (incumbent) / **1.8333** (ep2) against a fresh `Linear(256,8)` reference of **1.6332 ± 0.0167 sd** over 20 seeds — ≈ 10 sd. | MEASURED — `raw/intent_probe_*.json` `head_geometry`; reference banked at `raw/head_init_reference.json` |

### The share, MEASURED on the 140 eval windows (in-band n = 40)

| quantity | incumbent (fp32) | ep2 (EMA + bf16) |
|---|---|---|
| `CE_lat` (nats) | **0.8665** | **0.9420** |
| `CE_lon` (nats) | 1.3962 | 1.3970 |
| `w_tac_label · mean(CE_lat, CE_lon)` | **0.11313** | **0.11695** |
| `w_feat_op · loss_feat_op` | 0.47970 | 0.44863 |
| `w_feat_tac · loss_feat_tac` | 0.22582 | **0.00132** |
| `w_feat_str · loss_feat_str` | 0.01366 | 0.00329 |
| Σ feature terms | 0.71918 | 0.45324 |
| **tactical-label share of (label + feature)** | **13.59 %** | **20.51 %** |

⚠️ **THE CORRECTION.** The brief's "the live speed-channel epoch logs `loss_feat_tac` at 0.1–0.8 %
of the loss" is a **different term**: the tactical FIELD predictor's MSE. On ep2 it is indeed tiny
(`0.5 × 0.00263 = 0.00132`, 0.29 % of the sum) — and that is a *separate* finding about the EMA
target path, not about the decoder. The decoder's own term is 13.6–20.5 %.
⚠️ Scope: these are **eval-window** means, not the training stream. The training-stream ratio is
**UNVERIFIED and cannot be verified today** — the log has no key for it (§0 escalation 3) and Thor
is off-limits until ≈ 2026-09-04 02:00Z.

### The decisive likelihood read

Label entropy on the same in-band rows — the CE a head that emits the MARGINAL and nothing else
would attain — is **lat 1.0944 / lon 1.3047 nats**. Therefore:

* **lat: CE 0.8665 < H 1.0944.** The lateral head carries **0.228 nats (20.8 %) of information
  beyond the marginal** — *while its argmax is constant*. It ranks; it does not decide.
* **lon: CE 1.3962 > H 1.3047.** The longitudinal head is *worse* than the marginal in likelihood
  even though its argmax accuracy beats the constant control — it is confidently wrong on the rows
  it misses.
* Under `nav_zero` both collapse: lat CE **1.3390** (incumbent) / **1.3684** (ep2), i.e. **above**
  the marginal entropy. ⇒ **every scrap of the head's information comes from `nav_cmd`.**

---

## 3. (c) The label distribution it is fit to

`s2_labels_v7.2_train.jsonl.gz`, n = **4,572 records** (= exactly the 4,572/4,713 clip ids the
loader joins). One label per clip.

| lateral token | n | share |
|---|---:|---:|
| **LANE_KEEP** | 2,958 | **64.70 %** |
| NUDGE_R | 592 | 12.95 % |
| NUDGE_L | 488 | 10.67 % |
| TURN_L | 275 | 6.01 % |
| TURN_R | 259 | 5.66 % |
| LANE_CHANGE_L | **0** | 0 % |
| LANE_CHANGE_R | **0** | 0 % |
| ABORT_LC | **0** | 0 % |

**non-LANE_KEEP = 1,614 = 35.30 %.** Turns alone = 534 = **11.68 %**.
Longitudinal: CRUISE 1,243 (27.19 %), ACCELERATE 998 (21.83 %), ADAPT_SPEED_FOR_CURVE 995 (21.76 %),
BRAKE_TO 905 (19.79 %), FOLLOW 165 (3.61 %), CREEP 135 (2.95 %), HOLD 131 (2.87 %), **YIELD_MERGE 0**.
Eval (n = 147): LANE_KEEP 99 (67.35 %), NUDGE_R 21, NUDGE_L 14, TURN_R 8, TURN_L 5.

**Reading.** This is a **1.83 : 1** majority, not a 100 : 1 pathology. A 64.7 % base rate does not
by itself force a constant predictor — but **4 of 8 lateral and 1 of 8 longitudinal classes have
ZERO support**, so 5 of the head's 16 output units can only ever be pushed down.

---

## 4. How much gradient the head actually sees — the in-band rule

`RefAV1Windows` attaches the label only when the window's NOW falls inside the clip's tactical
band: `row[3] <= t*dt <= row[4]` (`refav1_loader.py:443`), band = `t0_s ± (tac_hi − tac_lo)/2`.
**MEASURED: `t0_s = 8.0` and `bands.tactical_s = [2.0, 6.0]` on 4,572 / 4,572 train records and
147 / 147 eval records** — so the band is `[6.0 s, 10.0 s]` for every clip, tolerance ±2.0 s.

| loader shape | reach | windows | in-band | fraction | P(batch of 8 has ZERO labelled rows) |
|---|---:|---:|---:|---:|---:|
| **train** (`str_ext_steps=2`) | 60 | 739 | 199 | **26.93 %** | **8.13 %** |
| eval (`str_ext_steps=0`) | 30 | 1,339 | 420 | 31.37 % | 4.92 % |

MEASURED on the 20 eval-slice episodes; the train shape's arithmetic is exact but the episode set
is the eval slice, so for the 4,713-episode training corpus this is **INFERRED**, not measured.
Confirmed independently by the loader's own banner in the probe log: `420/1339 windows in-band`.

⚠️ **Read this correctly.** `F.cross_entropy` uses `reduction='mean'` over non-ignored rows, so the
term's *magnitude* does not shrink — what shrinks is its *sample size*: at batch 8 the lateral CE is
estimated from **2.15 rows on average**, and on 8.13 % of steps from none at all (the term is
`continue`-skipped). **The gradient is noisy, not weak.** Anyone who reads "27 % of rows" as "the
loss is 27 % smaller" will pick the wrong lever.

---

## 5. (d) Does the decision reach the goal — PLUMBING

**YES, exactly. This hypothesis is REFUTED.**

| quantity | incumbent | ep2 |
|---|---|---|
| `goal_source == tactical_imagined` | 140 / 140 | 140 / 140 |
| `goal_lat == argmax(lat_head)` | **1.0000** | **1.0000** |
| `goal_lon == argmax(lon_head)` | **1.0000** | **1.0000** |
| same, under nav-shuffle | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| `goal_lat` histogram | LANE_KEEP 140 | LANE_KEEP 115, **TURN_R 25** |

Source: `refa_v1.py:1745-1761` — `lat_head(intent).argmax(-1)` feeds `canonical_controls`, and
`TURN_*` writes `κ = ±0.08` for the first 20 operative steps. **A decoded turn does become a curved
goal.** ep2 proves it end-to-end: 25 curved goals reached the cost.

---

## 6. (e) The class-recall table — BOTH banked checkpoints

n = 140 windows / 20 episodes; **in-band n = 40** (window indices {33, 43}, i.e. 2 of 7 per
episode — the stride-10 grid lands exactly twice inside `[6.0 s, 10.0 s]`).

### LATERAL — `nav_true`

| class | support | incumbent recall | ep2 recall |
|---|---:|---:|---:|
| LANE_KEEP | 26 | **1.000** | 0.885 |
| NUDGE_L | 6 | 0.000 | 0.000 |
| NUDGE_R | 2 | 0.000 | 0.000 |
| TURN_L | 2 | 0.000 | 0.000 |
| TURN_R | 4 | 0.000 | **1.000** |
| LANE_CHANGE_L / _R / ABORT_LC | 0 | — | — |
| **accuracy** | | **0.650** | 0.675 |
| **macro-recall (5 present)** | | **0.200** | 0.377 |
| argmax histogram, all 140 | | **LANE_KEEP 140** | LANE_KEEP 115, TURN_R 25 |

**CONTROLS (they read their known values EXACTLY — asserted in code, not hoped over):**

| control | value | incumbent | ep2 |
|---|---|---|---|
| constant-only accuracy | = base rate | **0.650** | 0.650 |
| constant-only macro-recall | = 1/K = 1/5 | **0.200** | 0.200 |
| shuffled-label (clustered) `p(macro ≥ observed)` | | **1.0000** | 0.0168 |
| shuffled-label (free) `p(macro ≥ observed)` | | 1.0000 | 0.0003 |

⇒ **The incumbent's lateral head is the majority-class predictor, to the digit**: same accuracy,
same macro-recall, `is_constant_predictor_ALL_windows = True`, and a permutation null it does not
beat (p = 1.0000). ep2 clears the null (clustered p = 0.0168) on the strength of `TURN_R` alone.

### LATERAL — nav controls (the C6 / nav-echo obligation, `refa_v1.py:395`)

| condition | incumbent argmax | acc | macro | ep2 argmax | acc | macro |
|---|---|---:|---:|---|---:|---:|
| `nav_true` | LANE_KEEP 140 | 0.650 | 0.200 | LK 115 / TURN_R 25 | 0.675 | 0.377 |
| `nav_shuffled` | LANE_KEEP 140 | 0.650 | 0.200 | LK 120 / TURN_R 20 | 0.575 | **0.177** |
| `nav_zero` | LANE_KEEP 140 | 0.650 | 0.200 | **LANE_KEEP 140** | 0.650 | 0.200 |

⇒ **ep2's turns are a nav echo.** Shuffling nav keeps 20 of the 25 turns but drives macro-recall
**below the constant control** (0.177 < 0.200); zeroing nav removes every turn.

### LONGITUDINAL — `nav_true` (identical on both arms, row for row)

| class | support | recall (both arms) |
|---|---:|---:|
| CRUISE | 18 | 0.889 |
| ADAPT_SPEED_FOR_CURVE | 6 | **1.000** |
| ACCELERATE | 12 | 0.000 |
| FOLLOW | 2 | 0.000 |
| CREEP | 2 | 0.000 |
| **accuracy** (const-only 0.450) | | **0.550** |
| **macro-recall** (const-only 0.200) | | **0.378** |
| shuffled-label clustered `p` | | **0.0098** |

⚠️ Under `nav_zero` the longitudinal head collapses to **CRUISE 140/140** (acc 0.450 = the base
rate, macro 0.200 = 1/K). Under `nav_shuffled` the histogram is unchanged (98/42) but accuracy falls
0.550 → 0.425 and macro 0.378 → 0.211. **The longitudinal head is a nav decoder, not a scene decoder.**

### The GEOMETRIC "human turns" stratum (`gt_turn_deg ≥ 5°` over the 2 s horizon)

| | incumbent | ep2 |
|---|---:|---:|
| n human-turn windows | 27 / 140 (10 episodes) | 27 / 140 |
| head asks TURN_* | **0** | **9** |
| recall [episode-cluster 95 %] | **0.000 [0.000, 0.000]** | **0.333 [0.000, 0.667]** |

⚠️ **A caveat the brief's "0 of the 27" needs.** The 27 is a GEOMETRIC stratum, not the label. Only
**8 of the 27** are in-band at all, and only **3 of those 8** carry a `TURN_*` v7.2 label. The
geometric criterion and the v7.2 lateral token are weakly concordant; do not read "recall 0.000 on
27 windows" as "27 labelled turns were missed."

### RANK vs ARGMAX — the head's own probabilities (AUC, no-information 0.500)

| score / stratum | incumbent | ep2 |
|---|---|---|
| `P(TURN_L)+P(TURN_R)` vs **label TURN**, in-band (n=40, n⁺=6) | **0.873 [0.686, 1.000]** | **0.922 [0.737, 1.000]** |
| same, `nav_zero` | 0.520 [0.000, 1.000] | **0.858 [0.667, 1.000]** |
| `P(TURN)` vs **geometric** turn (n=140, n⁺=27) | 0.606 [0.419, 0.834] | 0.616 [0.438, 0.809] |
| `1 − P(LANE_KEEP)` vs geometric turn | 0.681 [0.524, 0.825] | 0.700 [0.547, 0.842] |
| median `P(LANE_KEEP)` / median `P(TURN)` | 0.768 / 0.0142 | 0.735 / 0.0163 |

⇒ **The head RANKS turns while never EMITTING one.** On the incumbent that ranking is entirely
nav-borne (0.873 → 0.520 when nav is zeroed); on ep2 a vision-borne component survives
(0.858 [0.667, 1.000], CI excludes 0.500). The decision rule — an unweighted argmax against a
`P(LANE_KEEP)` prior of ~0.75 — is what discards it.

### The NAV-ONLY CEILING

A predictor reading **nothing but `nav_cmd`** (leave-one-episode-out, majority within nav class)
scores **0.684** on the same 38 scorable in-band rows — **above the model's 0.650 and above the
constant control's 0.650.** Per-nav label table: nav=0 (n=28) → LANE_KEEP 20 / NUDGE_L 6 / NUDGE_R 2;
nav=1 (n=2) → TURN_L 2; nav=2 (n=10) → LANE_KEEP 6 / TURN_R 4.
⚠️ `nav` is an **ORACLE (ego-future)** derivation, training-input only. The one signal the tactical
head has learned to read is the one that will not exist at deployment.

### The latent, by linear probe — REPORTED, NOT DECISIVE

Episode-disjoint leave-one-episode-out logistic probe on frozen `intent`, PCA-8 and standardiser fit
on the FIT fold only:

| panel | n | d | accuracy | constant-only | AUC |
|---|---:|---:|---:|---:|---:|
| geometric turn, binary | 140 | 256 | 0.786 | **0.807** | 0.512 (ep2 0.495) |
| v7.2 lateral, 5-class | 40 | 256 | 0.550 | **0.650** | — (minority recalls all 0.000) |

⛔ **`n ≪ d` BY CONSTRUCTION** on the label panel (40 rows, 256 dims, 8 PCA components fit on ~38).
And a negative from a LINEAR probe is not a negative about learnability — the function class is
`PCA-8 → multinomial logistic`. **This panel does not refute the representation hypothesis; the
better-powered read is the head's own AUC above**, which is a 256-d readout trained on 4,572 clips.

---

## 7. (b′) The instrument gap

Every row of all **nine** banked refav1 `train_log.jsonl` artifacts was PARSED (not grepped)
(`2026-09-02-refav1-ema-inflation/raw/{A_prime,B_prime,R7_resume,fit_probe}`,
`2026-09-02-refav1-target-space-collapse/raw/TSC-{A,B}`). The union of every key ever written is
`step, loss, precision, tf32, loss_feat_op, loss_feat_tac, loss_feat_str, grad_norm, adapter_std,
participation, loss_sigreg, loss_varfloor, tgt_std_{op,tac,str}, ema_decay, clip, skipped_steps,
tac_target_s, str_target_s, elapsed_s`. **`loss_lat_label` / `loss_lon_label` appear zero times**,
because `refa_v1_train.py:699-734` never writes them. Neither is
`loss_feat_str_ext` or `loss_route_label`. Consequence: the residual
`loss − (1.0·op + 0.5·tac + 0.25·str)` in any banked refav1 log conflates the strategic-extension
term, the tactical-label term and the route term, and cannot be attributed. **This is why §2's
number had to be recomputed from checkpoints rather than read from a log.**

---

## 8. ⭐ THE FINDING THAT REORDERS EVERYTHING — the seed is not the goal

`plan()` builds the goal and the candidate that chases it from the SAME `canonical_controls`
tensor, on **two different index sets**:

ℹ️ **Prior art, credited:** the steer-conversion agent escalated this as an open residual in
commit `4139203` — *"a 2 s plan is judged against a 6 s goal, so the planner's own seed cannot
reproduce its own goal"*. This section is the measurement of that residual.

`plan()` builds the goal and the candidate that chases it from the SAME `canonical_controls`
tensor, on **different index sets**. The candidate's regrid now has two modes
(`COST_TIME_GRIDS`, added in `4139203`); **`"dense"` is the DEFAULT and is what every BANKED refav1
number was produced under** — the parameter did not exist before that commit.

| feed | operative indices each of the 10 TACTICAL steps consumes | source |
|---|---|---|
| **GOAL** | `[0, 3, 6, 9, 12, 15, 18, 21, 24, 27]` | `_imagine_tactical_goal` — `acts[:, ::stride][:, :tac_steps]`, stride = `tac_dt/op_dt` = 3 |
| **SEED**, `cost_time_grid="dense"` **(DEFAULT, banked)** | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` | `refa_v1.py:1932` truncates to `cfg.plan_steps` = 10; no regrid |
| **SEED**, `cost_time_grid="tactical"` (the `4139203` repair) | `[0, 3, 6, 9, 9, 9, 9, 9, 9, 9]` | same truncation, then `tac_idx = min(j*stride, H−1)` |

**MEASURED (exact arithmetic on the shipped source, `raw/seed_goal_mismatch.json`, v0 = 10 m/s):
62 of 64 token pairs mismatch — UNDER BOTH GRIDS. 48 of 64 differ in curvature alone, under both.**
⚠️ So the `cost_time_grid` repair does **not** close this: the cause is the truncation to
`plan_steps`, not the regrid.

Under the DEFAULT `dense` grid:

| token pair | goal κ on the tactical grid | seed κ | steps differing | heading excess |
|---|---|---|---|---|
| `TURN_R × CRUISE` | −0.08 × 7, then 0 × 3 | −0.08 × 10 | 3 / 10 | **−82.506°** (ψ −3.360 → −4.800 rad) |
| `TURN_L × CRUISE` | +0.08 × 7, then 0 × 3 | +0.08 × 10 | 3 / 10 | **+82.506°** |
| `LANE_CHANGE_R × CRUISE` | −0.00875 × 4, **+0.00875 × 3**, 0 × 3 | −0.00875 × 10 | 6 / 10 | **−27.072°**, and the return arc's SIGN is gone |
| `NUDGE_R × CRUISE` | −0.01 × 2, +0.01 × 2, 0 × 6 | −0.01 × 5, +0.01 × 5 | **8 / 10** | 0.000° — the net heading matches, but the S-curve is **stretched 2.5× in time** and every interior step is wrong |
| `LANE_KEEP × BRAKE_TO` | a decays −1.500 → **−0.0872** | a decays −1.500 → **−0.5811** | **9 / 10** (accel) | — the seed ends at **6.66×** the goal's final decel |
| `LANE_KEEP × ACCELERATE` | 0.750 → **0.0436** | 0.750 → **0.2906** | **9 / 10** (accel) | — same 6.66× factor |

**The only two pairs that agree are `(LANE_KEEP, CRUISE)` and `(ABORT_LC, CRUISE)` — both the
all-zero control.** So: *the one manoeuvre the planner can faithfully chase is doing nothing*, which
is exactly the behaviour observed on 140/140 windows.
⚠️ `NUDGE_R` is the caution this section carries about itself: an earlier revision of
`seed_goal_mismatch.py` hard-coded the `"tactical"` grid and reported **+20.6°** for it. Under the
DEFAULT grid that number is **0.000°**. The mismatch is real either way — 8 of 10 steps — but a
heading-excess headline would have been wrong for the banked configuration. The tool now reads
`COST_TIME_GRIDS` and `plan()`'s default out of the source and reports every grid.

---

## 9. The refutation arm, run today at 0 GPU: does a curved goal even help?

ep2's 25 curved-goal windows are a ready-made natural experiment for the COST with the decoder held
fixed. `cost_surface_probe.py` already banked, per window, the full decomposition of the canonical
seed and of constant-velocity into `c_goal / c_jerk / c_kappa / c_vend / c_total`, so this needed no
forward pass (`raw/turn_decomposition_{A,B}.json`).

| quantity, over the 25 windows | convention A (legacy κ) | convention B (repaired steer) |
|---|---|---|
| goal advantage `cv_c_goal − turn_c_goal`, median | **0.000** | **−5.96e-08** |
| … mean / min / max | +1.07e-07 / −5.96e-08 / +1.967e-06 | +1.19e-08 / −1.19e-07 / +1.431e-06 |
| **n windows where the goal term prefers the turn** | **8 / 25** | **4 / 25** |
| n fp32-resolvable (\|adv\| > 2 ULP) | 15 / 25 | 21 / 25 |
| charge `0.05·κ²` | 3.200e-04 on all 25 | same |
| charge `0.02·jerk²` median / mean / max | 0.000 / 1.695e-04 / **3.218e-03** | same |
| n windows where jerk charge > κ charge | 2 / 25 | 2 / 25 |
| **n turns that WIN as shipped** | **0 / 25** | **0 / 25** |
| **n turns that WIN under the chord distance, same weights** | **1 / 25** | **0 / 25** |
| chord leverage gain, median | **5,792.6×** | 4,096.0× |

**Three readings, each load-bearing.**

1. **BACKLOG R29 is NECESSARY BUT NOWHERE NEAR SUFFICIENT.** `chord = √(2(1−cos))` is strictly
   monotone in `1−cos`, so it cannot change the goal term's own ordering; what it changes is the
   goal term's *leverage inside the sum*, by a **measured median 5,793×**. That flips **1 of 25**
   windows. ⚠️ And because the cost is a SUM, swapping the metric silently re-weights goal against
   penalty by that same 5,793× — **it is not a neutral reparameterisation and an arm that swaps it
   while holding `0.05` fixed is not a one-variable arm.**
2. **ON THE BANKED PANEL the direction is wrong, not only the magnitude.** On 17/25 (A) and 21/25
   (B) the canonical turn's goal term is no better than constant velocity. §8 gives one reason that
   survives every repair — the seed is a different manoeuvre from the goal — and the scope note
   below gives another that does NOT: the banked panel ran the half-applied units crossing, and
   `4139203`'s tiny-model evidence puts the sign right (**+5.364e-07**) once that is fixed on both
   sides. ⇒ **Do not carry "the goal term prefers the straight line" forward as settled.** What is
   settled is that the turn WINS 0/25 as shipped.
3. **The κ² penalty is not the only charge.** Its median dominance (3.2e-04 vs a median jerk of 0)
   hides a mean jerk charge of 1.70e-04 and a max of **3.218e-03** — an order of magnitude above the
   curvature penalty on 2 of 25 windows, produced by the canonical control's own longitudinal step.

⚠️ **SCOPE, AND A NUMBER THAT MOVED UNDER ME.** Every row above is a re-read of the BANKED
cost-surface panel, which was produced **before** the steer-conversion repair landed
(commit `4139203`, same day). That commit's own tiny-model evidence
(`…/2026-09-03-steer-conversion-completion/raw/evidence.json`, `P1_tiny_model_costs`) measures the
canonical turn's goal advantage over the straight line as **0.0** (units OFF, self-consistent),
**−2.384e-07** (the half-applied crossing that shipped), and **+5.364e-07** (crossing FIXED on both
sides). So on the units-FIXED path the sign is RIGHT — it was the half-applied crossing that made
it negative. That does not rescue the ranking on its own: **+5.364e-07 against the `0.05·κ²` charge
of 3.200e-04 is still 596× short.** ⚠️ But it is a TINY MODEL, not these checkpoints, and it is
not window-for-window comparable to the 25 banked windows — which is exactly why §8's E0 must
re-measure on the real checkpoints before any arm is bought.
⭐ **And one arithmetic consequence worth stating, because it changes what E0 might find:** applying
the chord to those same two tiny-model values gives
`√(2·5.960e-07) − √(2·5.960e-08) = 1.0918e-03 − 3.4526e-04 = 7.465e-04`, which **EXCEEDS the
3.200e-04 curvature charge by 2.33×**. On the banked (pre-repair) windows the chord flipped 1/25;
on the units-fixed tiny model the same transform would tip it. **The two readings disagree, and
neither is on the real checkpoints.** E0 is the experiment that decides between them, and this
package does not pick a side.

Cross-check against the banked R10 panel (independent instrument, same windows): with the penalty
removed and evaluated in **float64**, the surface's argmin is a turn on only **7.86 %** of windows
(`turn_frac_A_nopen_f64` 0.0786 [0.0286, 0.1357]) — the fp32 reading of 89.29 % was noise. The goal
term's κ-range is **1.63e-10 (f64)** against the penalty's **2.00e-03** across the same box: a
factor **1.2 × 10⁷**.

---

## 10. THE MECHANISM, in one paragraph

The four hypotheses separate cleanly, and **three of the four are refuted**. It is **not a plumbing
problem in the decoder**: `goal_lat` equals the head's argmax on 140/140 windows in both arms
(§5), and ep2's 25 decoded turns became 25 genuinely curved goals. It is **not an unsupervised
head**: the term is live at `w_tac_label = 0.1` and carries **13.6 % / 20.5 %** of the loss on these
windows, the weights sit ≈ 10 sd off a fresh init, and `CE_lat = 0.8665 < H(p) = 1.0944` proves the
head uses information the marginal does not have (§2) — so raising the weight is not the lever. It
is **not simply a representation problem either**, though it is partly one: the head's own
probabilities rank turns at **AUC 0.873 / 0.922** against the in-band label, so the signal reaches
the logits — but under `nav_zero` the incumbent's ranking collapses to **0.520** and a nav-only
predictor **beats the model outright (0.684 > 0.650)**, so *what the head has learned is the oracle
nav command, not the scene*. What remains, and what the numbers name, is **majority-class collapse
of the DECISION RULE sitting on top of a nav-borne latent**: with a 64.7 % base rate, 5 of 16 output
classes carrying zero support, a lateral CE estimated from ~2.15 rows per batch-8 step (26.93 % of
windows are in-band; 8.13 % of steps see none at all), and an unweighted argmax against a
`P(LANE_KEEP)` prior of ~0.75, a head that *ranks* turns correctly still *emits* `LANE_KEEP` on
140/140 windows and reads its accuracy and macro-recall EXACTLY off the constant control.
**And behind that decoder sits a second, independent defect that would swallow any decoder repair:
the planner's seed is not the goal's control (62/64 token pairs, `TURN` over-rotating by 82.5°), so
even ep2's correctly-decoded turns beat constant velocity on the goal term in only 8/25 windows and
win 0/25 — with the chord distance rescuing exactly 1.**

---

## 11. Lever ranking — by cost and evidence, not preference

| # | lever | cost | what the evidence says | verdict |
|---|---|---|---|---|
| **L0** | **Make the seed the goal's own control** (extend `plan_steps` to `op_steps`, or stop `tac_idx` saturating, or build the goal on the truncated-and-held profile) | ~3 lines, **0 GPU to verify** on banked ckpts | 62/64 token pairs mismatch; `TURN` over-rotates 82.5°; the only faithful manoeuvre is "do nothing" (§8) | **FIRST. Blocking.** |
| **L1** | **Log `loss_lat_label` / `loss_lon_label`** (and `loss_feat_str_ext`, `loss_route_label`) | 4 lines, 0 GPU | no run has ever carried them (§7) | **FIRST. Free.** |
| **L2** | **Class-balanced / focal loss** on the lateral head, or an explicit prior correction at decode | one arg + one line; one tiny-rig arm | the head ranks (AUC 0.873) and does not decide; acc = base rate EXACTLY; `P(LANE_KEEP)` ≈ 0.75 | **the decoder lever.** |
| **L3** | **Cost: `1−cos` → chord** (R29) **together with the goal/penalty balance** | ~2 lines + one weight; 0 GPU to verify | leverage +5,793× but flips only 1/25; goal κ-range 1.63e-10 vs penalty 2.00e-03 = 1.2e7 (§9) | **necessary, insufficient alone; must move `0.05` with it or it is not one-variable.** |
| **L4** | **The `0.05·κ²` weight** (and `0.02·jerk²` — mean charge 1.70e-04, max 3.218e-03) | one number | κ penalty is 3.2e-04 on every turn window; jerk exceeds it on 2/25 | **couple to L3, never alone.** |
| **L5** | **The label distribution itself** — the ±2 s band (26.93 % of windows), one token per 20 s clip, 5 of 16 classes with zero support, `t0_s ≡ 8.0` | a Data-FlyWheel re-emit; days | 64.7 % is not a pathological imbalance; the *sparsity in time* and the *dead classes* are the real defects | **highest ceiling, longest lead time. Open with the Data FlyWheel now.** |
| **L6** | **Raise `w_tac_label`** | one number | REFUTED as the lever: the term is already 13.6–20.5 % of the loss | **do not spend an arm on it.** |
| **L7** | **Remove or de-weight oracle `nav` into the tactical head** | one flag | a nav-only predictor beats the model (0.684 > 0.650); every head signal is nav-borne | **must be measured, but it will make T0 numbers WORSE before better — pre-register that.** |

**Which must land first, and why.** **L0 and L1, together, before any training arm.** L1 because a
term with no instrument cannot be tuned. L0 because until the seed *is* the goal's control, the T1
endpoint is blind to a decoder improvement: ep2 already emits 25 turns and wins 0 of them, so a
better decoder would still read as a null — **the exact failure mode of "the target-scale
instrument was blind to the collapse it was built to detect" (commit `e1352da`).** L3+L4 next,
because they are also verifiable at 0 GPU on banked checkpoints. **L2 is the only lever that needs a
training arm, and it should be the last one bought**, because L0/L3/L4 change what a decoder
improvement is worth. A decoder fix and a cost fix are INDEPENDENT and **both are necessary**: a
turn that is requested but unrankable is exactly as useless as one never requested — measured, 0/25.

---

## 12. Deliverables and reproduction

Every number above is regenerable from `tools/` against the banked artifacts:

```
decode_audit.py         0 GPU   class-recall + controls + plumbing + nav conditions
window_band_census.py   0 GPU   in-band fraction, train shape and eval shape
turn_decomposition.py   0 GPU   the refutation arm on ep2's 25 curved-goal windows
seed_goal_mismatch.py   0 GPU   the goal-vs-seed control sequences, all 64 token pairs
intent_logit_probe.py   GPU     loss share, rank-vs-argmax AUC, head geometry, latent probe
```

Runner: `raw/run_probe.sh`; logs `raw/probe_{incumbent,ep2}.log`, `raw/decode_audit.log`.
Mirror used for RUNNING only (`C:/Users/Admin/tanitad-wt`); every tool is authored in the repo.

**Open / UNVERIFIED**

* The **training-stream** loss ratio for the tactical-label term (Thor off-limits; no log key).
* Whether the 27-window geometric turn stratum and the v7.2 lateral token *should* agree — 3 of 8
  in-band geometric turns carry a `TURN_*` label. Data FlyWheel question.
* `intent_logit_probe.py` writes `NaN` for a degenerate AUC; Python round-trips it, strict JSON
  parsers will not.
