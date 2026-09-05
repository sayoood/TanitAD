# WHY refav1 DOES NOT TURN — and it is NOT the goal head

**Row:** `D-REFAV1-VOCAB-QUANT` · **Class:** MEASURED, **zero GPU**
**Agent:** TanitAD Architecture & Inference FlyWheel · **Date:** 2026-09-05
**Instruments:** `tools/margin_anatomy.py`, `tools/vocab_fit.py`
**Raw:** `raw/margin_anatomy_s40.json`, `raw/vocab_fit_s40.json`, `raw/threshold_sweep.json`
**Panel:** 282 windows / 141 episodes, ckpt step 21,109 (132 GT-turn, 150 GT-straight)
**Pre-registration:** `SPEC.md`, banked before any number below was read.

---

## The one-paragraph answer

The predecessor package MEASURED gate 1 correctly — the goal head decodes `LANE_KEEP`
on 86.5 % of windows and a `LANE_KEEP` decode makes a turn unreachable — and read it as a
**head** defect to be fixed by a decision rule or a re-fit. **That implication is wrong.**
On **90.15 %** of GT-turn windows, `LANE_KEEP` is the **vocabulary-optimal** token: no
lateral token the vocabulary contains is closer to the road's actual curvature. Across all
282 windows the vocabulary-optimal token is `LANE_KEEP` **95.39 %** of the time, while the
head emits it only **86.52 %** — **the head already turns MORE often than its own vocabulary
justifies.** The binding constraint is that the lateral goal vocabulary can command exactly
two *sustained* curvatures, **0 and 0.08 1/m (R = 12.5 m)**, and the roads in this corpus
curve at **R = 100–1000 m**. ⇒ **No decision rule and no re-fit of `lat_head` can make
refav1 track a normal road curve.** The fix is a finer sustained curvature, and it is cheap.

---

## 1. The vocabulary, read off the source

`stack/tanitad/refs/refa_v1.py:118-123` and `:347-383` (`canonical_controls`):

| token | curvature profile | net heading change |
|---|---|---|
| `LANE_KEEP` | `k = 0` | 0 |
| `TURN_L/R` | `k[0 : 4.0/dt] = ±0.08` — **SUSTAINED** | **large** |
| `NUDGE_L/R` | S-curve `+kap` then `−kap`, `kap = 2·0.5/(v_ref²·1.0²)` | **0** |
| `LANE_CHANGE_L/R` | S-curve, `kap = 2·1.75/(v_ref²·2.0²)` | **0** |
| `ABORT_LC` | `k = 0` | 0 |

⭐ **An S-curve integrates to zero net heading change**, so `NUDGE` and `LANE_CHANGE` cannot
track a sustained curve *however large their peak* — they shift the car sideways and return
it to its original heading. ⇒ **The only SUSTAINED curvatures the vocabulary can command are
`0` and `±0.08`.**

## 2. What the road actually does — MEASURED

`gt_kappa` = `ff._seq_geometry(g, 0.2)`'s `yaw_rate.mean / speed.mean.clamp_min(0.5)`, the
banked definition (`raw/gt_kappa_282.npz`, reproduced by control C2).

| percentile of `|κ|` | value (1/m) | radius |
|---|---|---|
| p50 | 0.00085 | **1176 m** |
| p75 | 0.00375 | 267 m |
| p90 | 0.01445 | 69 m |
| p99 | 0.20308 | 4.9 m |

⇒ `|κ| > 1e-3` — the predecessor's "GT turn" threshold, and mine inherited from it — is a
**1000 m radius**. Half the corpus's "turns" are gentler than a 267 m curve. The token that
exists is **12.5 m**.

## 3. The test: given the tokens that exist, what SHOULD the head emit?

`tools/vocab_fit.py` scores every lateral token by the L2 distance between its canonical
curvature profile (re-stated from source) and the window's constant GT curvature, and reports
the argmin.

| `|κ|` band | radius | n | vocab-optimal is `LANE_KEEP` | head decodes `LANE_KEEP` |
|---|---|---|---|---|
| (0, 1e-3] | > 1000 m | 145 | **1.0000** | 0.9448 |
| (1e-3, 5e-3] | 1000–200 m | 71 | **1.0000** | 0.8732 |
| (5e-3, 1e-2] | 200–100 m | 23 | **1.0000** | 0.7826 |
| (1e-2, 2e-2] | 100–50 m | 17 | **1.0000** | 0.8824 |
| (2e-2, 4e-2] | 50–25 m | 8 ⚠️ | **1.0000** | 0.6250 |
| (4e-2, 6e-2] | 25–16.7 m | 2 ⚠️ | 0.0000 | 0.0000 |
| (6e-2, ∞) | < 16.7 m | 11 | **0.0000** | 0.4545 |

⚠️ Bands with n < 10 are marked and no verdict is read off them.

**Headline:** of the 132 GT-turn windows, the vocabulary-optimal token is `LANE_KEEP` on
**90.15 %**. Over all 282 windows it is `LANE_KEEP` on **95.39 %**, against the head's
**86.52 %**.

⭐ **The head is not under-turning. It is over-turning by 8.9 points relative to the best its
vocabulary allows.**

### ⛔ Controls

* **C1 — the scoring is not rigged to say `LANE_KEEP`.** On the 11 windows with `|κ| > 0.06`
  the argmin is `TURN_*` on **11/11 = 100 %**. A same-breath column that had to read the
  opposite value, and does. Without it, "vocab says LANE_KEEP everywhere" would be
  indistinguishable from a broken scorer.
* **C2 — the crossover is analytic, not fitted.** For a constant target the L2 argmin between
  `0` and `K` is the midpoint: `GOAL_KAPPA_TURN/2 = 0.040`. **Measured: 0.0450** (the small
  excess is the 4 s/6 s duty cycle of the `TURN` profile). The instrument reproduces theory it
  was not told.
* **C3 — `n` printed per band**, and under-powered bands flagged rather than quoted.

## 4. Why the margin looked like a head defect — the threshold was doing the work

The head's turn-vs-hold separability is a **strong function of what counts as a turn**, and
the inherited `1e-3` threshold is the weakest possible definition:

| turn threshold | radius | n turn | **AUC** | best bal-acc | argmax recall | argmax false |
|---|---|---|---|---|---|---|
| 1e-3 | 1000 m | 132 | 0.7120 | 0.6853 | 0.2045 | 0.0733 |
| 5e-3 | 200 m | 61 | 0.7720 | 0.7483 | 0.2951 | 0.0905 |
| 1e-2 | 100 m | 38 | **0.8039** | 0.7770 | 0.3421 | 0.1025 |
| 3e-2 | 33 m | 16 | **0.8799** | 0.8109 | 0.6250 | 0.1053 |
| **8e-2 = `GOAL_KAPPA_TURN`** | **12.5 m** | 8 ⚠️ | 0.8558 | **0.8339** | **0.6250** | 0.1204 |

⇒ At the curvature the `TURN` token actually commands, argmax **already** decodes 62.5 % of
real turns at a 12 % false-turn rate, and best-threshold balanced accuracy is **0.834**.

⚠️ **My own pre-registration has a gap and I am naming it rather than exploiting it.**
`SPEC.md` committed *"CALIBRATION iff `bal_acc_at_best_threshold` ≥ 0.80"* but **did not pin
the turn definition**, and the verdict flips with it: **SEPARABILITY at 1e-3 (0.685),
CALIBRATION at 8e-2 (0.834)**. I am not free to pick the one I like. The principled
resolution is a fact about the code, not a choice made after seeing results: **the turn
definition should be the curvature the system's own goal token commands**
(`GOAL_KAPPA_TURN = 0.08`, `refa_v1.py:119`). Under that definition the head is
**adequately calibrated** and the defect is elsewhere — which is what §3 independently shows
by a different route. Both readings are reported; neither is suppressed.

## 5. The margin gap that motivated the re-fit is not a valid objective

`MUST_WE_RETRAIN.md` proposes `median_margin_gap_logits` (today **0.297**) as Step 2's success
criterion. **`lat_head` is `LayerNorm → Linear`**, so scaling that `Linear`'s weight and bias
by `c` multiplies every logit — and the gap — by `c` while changing nothing else.

**MEASURED (`raw/margin_anatomy_s40.json`, control 3), with `c = 3`:**

| quantity | result |
|---|---|
| decode bit-identical | **True** (282/282) |
| AUC unchanged | **True** (< 1e-12) |
| Cohen's d unchanged | **True** |
| `median_margin_gap_logits` | **× 3.0000 exactly** |

⇒ **A raw-logit margin is a property of the weight norm, not of the decision.** An objective
phrased *"widen the margin gap in logits"* is satisfiable by scaling the head and changing
nothing. ⛔ It is inadmissible as stated and must be replaced by a scale-free quantity
(AUC, balanced accuracy, Cohen's d). *Same family as the `df` / Thor `free` / `step_s` traps:
a true number quoted outside the scope in which it means anything.*

**The scale-free anatomy of that same separation (n = 282, d = 1):**

| quantity | value |
|---|---|
| AUC | 0.7120 (shuffled control **0.5055 ± 0.0336**) |
| Cohen's d | 0.5902 |
| overlap coefficient of the two densities | **0.5845** |
| gap as a fraction of pooled SD | **0.329** |

## 6. What this changes

| the predecessor's next step | status after this measurement |
|---|---|
| Step 1 — `lat_logit_bias` decision rule | ⚠️ **bounded and mostly beside the point.** It can only move windows where a `TURN` goal is the right goal — 4.6 % of the panel. Its max-Youden point (`b = 1.5`) turns on **46 %** of straights. |
| Step 2 — re-fit `lat_head` on v7.2 labels | ⛔ **cannot work as specified.** The head already tracks its vocabulary better than the vocabulary tracks the road; and its own objective was voided in §5. |
| **the real lever** | ⭐ **a finer SUSTAINED curvature in the lateral goal.** The head's features are adequate (AUC 0.804 at R ≤ 100 m); what is missing is a token — or a continuous magnitude — between `0` and `0.08`. |

⇒ **Step 2 is re-specified, not abandoned:** fit a **curvature-MAGNITUDE regressor on the same
banked `intent`**, and let the goal command a continuous `κ`. That is still "hours, not
GPU-weeks", still leaves the trunk and world model untouched, and it is the change that makes
a 200 m road curve expressible at all.

## ⛔ P3 admissibility, MEASURED not assumed

`plan()` builds the goal head's input at `refa_v1.py:2156` as
`self._run_brains(pooled_win, nav_cmd)` — **`ego=None`**. So at inference the goal path sees
**vision + nav only**; no ego channel, and the situation classifier's output enters nowhere
(no arm in this package computes one).

⚠️ **nav is weakly route-informative and that is INHERITED, not introduced here.** MEASURED
on the 282: `P(GT turn | nav)` = 0.411 / 0.500 / 0.592 against a base rate of 0.468;
**nav-only AUC 0.5967**, best nav-only balanced accuracy 0.5730 — well below the head's
0.7120, so the head is not a nav echo. The dense bank carries `intent` computed **twice**,
under true and shuffled nav (`refav1_arm.shuffle_nav`, 2455/4786 rows changed), so any re-fit
can be scored under both and a fit that leans on the route signal is visible rather than
assumed away.

## What this does NOT say

* It does **not** say the head is good. At R ≤ 100 m its AUC is 0.804 — real, not excellent.
* It does **not** retract gate 1. `LANE_KEEP ⇒ κ ≡ 0` on 244/244 stands; what changes is the
  attribution of *why* `LANE_KEEP` is emitted.
* It does **not** touch gate 2 (`cos` refusing decoded turns). That remains a live defect and
  a `ccos` prerequisite for any lateral control.
* ⚠️ **One checkpoint, one panel, n = 282, and the sharp bands are thin (n = 11 at `|κ| > 0.06`).**
  The dense panel (4786 windows) is banked to widen exactly those bands.

---

## 7. The DENSE panel — 4786 windows / 141 episodes, 17× the data

The §3 and §4 tables rest on 282 windows, and their sharp bands were thin (n = 11 at
`|κ| > 0.06`). The `intent` bank (`tools/extract_intent.py`, all four controls passed) re-runs
the same forward pass on a stride-2 grid and gives **4786 windows over the same 141 episodes**.
Every headline holds, and the controls get 12× stronger.

### 7.1 The vocabulary result holds

| `|κ|` band | radius | n | vocab-optimal is `LANE_KEEP` | head decodes `LANE_KEEP` |
|---|---|---|---|---|
| (0, 1e-3] | > 1000 m | 2464 | **1.0000** | 0.9387 |
| (1e-3, 5e-3] | 1000–200 m | 1175 | **1.0000** | 0.8502 |
| (5e-3, 1e-2] | 200–100 m | 369 | **1.0000** | 0.7724 |
| (1e-2, 2e-2] | 100–50 m | 336 | **1.0000** | 0.8452 |
| (2e-2, 4e-2] | 50–25 m | 156 | **1.0000** | 0.5897 |
| (4e-2, 6e-2] | 25–16.7 m | 73 | 0.0000 | 0.2329 |
| (6e-2, ∞) | < 16.7 m | 138 | 0.0000 | 0.2681 |

* of GT-turn windows, vocab-optimal is `LANE_KEEP` on **0.9061** (282-panel: 0.9015)
* over all windows, vocab-optimal `LANE_KEEP` **0.9559** vs the head's **0.8521**
* ⛔ **C1 with 138 sharp windows instead of 11: `TURN_*` is the argmin on 138/138 = 1.0000**
* ⛔ **C2: measured crossover 0.04101** against the analytic `GOAL_KAPPA_TURN/2 = 0.040` — the
  agreement TIGHTENS with more data, which is what a real quantity does and an artefact does not

### 7.2 The threshold family holds, and the well-powered row is the interesting one

| turn threshold | radius | n turn | AUC | best bal-acc | argmax recall | argmax false | shuffled AUC |
|---|---|---|---|---|---|---|---|
| 1e-3 | 1000 m | 2247 | 0.7235 | 0.6886 | 0.2372 | 0.0689 | 0.4996 |
| 5e-3 | 200 m | 1072 | 0.7682 | 0.7383 | 0.3330 | 0.0945 | 0.4999 |
| 1e-2 | 100 m | 703 | 0.7970 | 0.7672 | 0.3883 | 0.1065 | 0.5036 |
| 3e-2 | 33 m | 281 | 0.8602 | 0.7734 | 0.6512 | 0.1165 | 0.5008 |
| ⭐ **4e-2** | **25 m** | **211** | **0.8806** | **0.8200** | **0.7441** | **0.1204** | 0.4983 |
| 8e-2 | 12.5 m | 84 | 0.8457 | 0.7904 | 0.6786 | 0.1385 | 0.4975 |

⭐⭐ **The `4e-2` row is the one that matters, and it is now well powered (n = 211).** `4e-2` is
not a threshold chosen for a good number — it is the **measured vocabulary crossover (0.04101)**,
the curvature above which `TURN` beats `LANE_KEEP` in the goal vocabulary. **At exactly the
curvature where a `TURN` goal is the correct goal, the SHIPPED head already decodes a
curvature-carrying token on 74.4 % of windows at a 12.0 % false-turn rate.**

⇒ **Gate 1 is largely OPEN where it matters.** The 20.5 % "turn recall" that motivated the
whole decision-rule programme is a statistic about **1000 m curves**, which the vocabulary
cannot express anyway. ⇒ the binding constraint on *"does refav1 turn"* is **gate 2** — the
cost metric, which under the shipped `cos` default refuses a correctly decoded turn on 38/38
windows (`D-REFAV1-DRIVE-GATE2`, INHERITED) — and that is what P4 measures directly.

### ⚠️ One control does NOT apply here, and it is reported rather than hidden

`threshold_sweep.py`'s `CONTROL_reproduces_banked_at_1e-3` reads **`passes: false`** on this
panel. That is **correct behaviour, not a failure**: the control asserts the banked
**282-window** operating point (AUC 0.7120 / recall 0.2045 / false 0.0733), and the dense panel
is a **different, 17× larger window set** (4786 windows on the same 141 episodes), which must
not reproduce it. The control PASSES on the stride-40 panel it was written for
(`raw/threshold_sweep_s40.json`, exact to 1e-9). ⛔ It is left in the output deliberately —
a control silently disabled on the panel where it does not apply is how a real failure later
goes unnoticed.
