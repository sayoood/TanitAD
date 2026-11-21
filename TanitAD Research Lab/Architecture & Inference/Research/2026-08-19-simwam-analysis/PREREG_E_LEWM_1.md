# PRE-REGISTRATION — E-LEWM-1: which v6 deviation from LeWorldModel costs the representation?

`PRE-REGISTERED 2026-08-20, BEFORE THE HARNESS PRODUCED ANY NUMBER.`
**T0-DIAGNOSTIC.** Dev-box RTX 4060 — **Thor is untouched and finishes S-W
normally.**

---

## 1. Why

`LEWM_VS_OURS.md`: LeWorldModel (`2603.19312`) is **v6's architecture** —
encoder + action-conditioned next-latent predictor, jointly trained, SIGReg
against collapse, no teacher — and it reaches probe **r ≈ 0.90–0.99**,
competitive with frozen DINOv2. **v6's latent decodes nothing** (E-TRUNK-2), and
its SIGReg has **stalled at 7.83 since step 11 k — worse than rank-1 collapse —
while consuming 45 % of the loss.**

⇒ The recipe works; **our instance is misconfigured.** Four deviations are
candidates and **v6 costs 9+ days per experiment at 336 M params**, so the
question is answered at **LeWM's scale** (~15 M, hours) and the winner applied
once.

## 2. Arms — one-factor-at-a-time from the LeWM configuration

Baseline **`lewm`** = LeWM as published: `d_latent` 192 · SIGReg on **encoder z
AND predictor ẑ** · **no detach** · **2 loss terms** (MSE + λ·SIGReg, λ = 0.1).

Each arm flips **exactly one axis toward v6**:

| arm | flip | v6's value |
|---|---|---|
| `lewm` | — (baseline) | — |
| **`d2048`** | `d_latent` 192 → **2048** | v6 `d_op` = 16 × 128 |
| **`sigop`** | SIGReg on a pooled operative readout **only** | v6 applies it to `states` alone |
| **`detach`** | detach the target `z_{t+1}` | v6: *"detached by the caller"* |
| **`terms7`** | add v6-shaped auxiliary terms (near-field, masked-cell, rollout-consistency) | v6 runs 7 |

⚠️ **Identical everywhere else**: same frames, same clips, same seeds (3), same
steps, same optimiser, same episode-disjoint split.

## 3. Evaluation — LeWM's own metric

The **E-TRUNK-2 probe**, unchanged: linear (dual/Gram, scale-normalised) probes
for `lead_gap_m`, `left/right_occupied`, `nearest_any_m`, `n_agents_log`,
`ego_speed`; **episode-disjoint** folds; **episode-cluster bootstrap**. This is
LeWM §5.1's instrument and LeJEPA Fig. 1's model-selection criterion, so it is
their standard, not one imported to judge them.

**Reported alongside, per arm:** final `SIGReg` value, its **trajectory** (does
it converge or stall like v6's?), and **between/within-episode variance ratio**
(v6 = 4.56×, `dino_pooled` = 2.47×).

## 4. ⛔ Committed decision rule — written before any number

**Primary readout: `lead_gap_m` R² and `left/right_occupied` AUC, `lewm` vs each
flip, paired episode-cluster bootstrap.**

| observation | conclusion, committed now |
|---|---|
| `lewm` **decodes** (`lead_gap_m` CI excludes 0, occupancy AUC CI excludes .5) **and one flip kills it** | ⭐ **that flip is the defect.** If it is `sigop`, `detach` or `terms7` → **v6 IS REPAIRABLE IN PLACE** (all loss/config-only, SigReg has no state_dict keys). If it is `d2048` → **retrain required**, 70/573 tensors change shape. |
| `lewm` decodes and **no flip kills it** | ⛔ **none of the four is the mechanism.** The deficit is elsewhere — scale, corpus, or the encoder — and the next question is data, not configuration. |
| ⛔ **`lewm` itself does NOT decode** | ⛔ **THE HARNESS IS NOT A LEWM REPLICATION** and no arm may be interpreted. Report as a failed replication, state the difference from the paper, and **do not** claim anything about v6. |
| multiple flips each kill it | report **per axis with its interval**; the strongest claim is restricted to axes that separate individually. **No additivity is assumed.** |
| `lewm` decodes and a flip **improves** it | report it; an unexpected direction is a result, not a nuisance. |

⚠️ **`lewm` decoding is the GATE.** Without it the experiment measures nothing —
this is the built-in falsifier and it is the most likely failure mode.

## 5. ⛔ What this cannot settle

* **Not driving.** T0 throughout. No arm here licenses a T1 claim — that is C129's
  error and it is not repeated.
* **Not v6 itself.** A 15 M-param model on 130 clips is **not** v6 at 336 M on
  2,376 episodes. A flip that matters at small scale **may not** at full scale,
  and vice versa. This ranks candidates for one expensive confirmation; it does
  not replace it.
* **Not LeWM's numbers.** Push-T (their §5.1) is not driving. We are testing
  **relative** effects of configuration flips, never reproducing their absolute r.
* **Not the corpus question.** If nothing separates, "2,376 episodes is too few"
  remains live and untested here.
* **Linear probes only**, as in E-TRUNK-2. LeWM reports linear **and** MLP; ours
  are linear, so non-linearly encoded content reads as absent in every arm alike.

## 6. Falsifiers built in

1. **`lewm` must decode** — §4 row 3. Otherwise nothing is interpretable.
2. **`C-EGO` → `ego_speed` must read ≈ 1.0** (identity map); it caught the
   Gram-scaling defect and stays.
3. **`C-PIXEL` and `C-EGO` are checkpoint-independent** and must be identical
   across arms; drift ⇒ harness fault.
4. **Seeds reported individually**, never only the mean.

## 7. Manifest

| artifact | where |
|---|---|
| this pre-registration | `…/simwam-analysis/PREREG_E_LEWM_1.md` |
| harness | `…/simwam-analysis/code/e_lewm_ablate.py` |
| result (to be written) | `…/simwam-analysis/E_LEWM_1_ABLATION.md` + `raw/e_lewm_ablate.json` |
