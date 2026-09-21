# PREREG — `H-BOXCLS-1`: weighting the `cls` term recovers the box head's class, and size follows

**Status:** PRE-REGISTERED, GATE **PASSED**, NOT LAUNCHED. **Needs:** GPU ⇒ **the PI's call**.

> ⭐ **GATE RESULT, recorded after §3 was written and unchanged since.** The launch gate below was
> committed *before* the gate experiment produced a number — and it came within one arm of ruling
> this experiment out. Outcome: **LAUNCH.** Within a single scene, a **LINEAR** readout of
> `box_dec`'s own per-slot features separates a vulnerable road user from a vehicle at AUC
> **0.58317**, CI **[0.52416, 0.64653]** — separated from chance — while the label-shuffled control
> returns to **0.47057**, CI [0.36058, 0.57529], straddling 0.5 as it must. The head itself scores
> **exactly 0.5**: emitting one class for every slot, it has no within-scene ordering at all.
> ⇒ the signal is present **in the function class the head already is** (`cls` is one `nn.Linear`,
> `agent_slots.py:368`), so the collapse is an **objective** failure, not a representation one.
> Under an *unweighted* loss on a 577.5:1 target the constant genuinely IS the optimum — which is
> why re-weighting is the lever and why `A_ctrl` must reproduce the collapse.
> ⚠️ **Read the magnitude honestly: 0.583 is WEAK discrimination.** It establishes that signal
> EXISTS, not that a re-weighted head will classify well. §5's criteria are unchanged and are not
> to be softened by this. Artifact: `raw/vru_within.json`.
**Registers against:** `D-S1-DEP-BOX` (open), gates `S1-GATE-PRED`.
**Written:** 2026-09-21, before any arm ran. **Author:** Master Mind.

---

## 0. Which head — stated first, because getting this wrong cost a retraction today

Every quantity in this document is measured on **`perception.box_dec`**, a `Box3DSlotDecoder`
(`box3d_head.py:227`) emitting **100 slots**, read as `box_slots` at `s1_pass.py:307` and produced
at `refcv6_perception_branch.py:426`. **Not** `core.agent_head` (the 2-D head, 16 queries).

⛔ Any probe in this experiment selects the module **by name** and asserts, per window, that the
hooked tensor's slot count **equals** the `box_slots` count the same forward emitted. A width match
is forbidden as a selector: `Box3DSlotDecoder` subclasses `AgentSlotDecoder`, so the parent's width
still matches a different live module — the exact defect of
`RETR-2026-09-21-ALIGNMENT-MEASURED-ON-THE-WRONG-HEAD`.

## 1. The defect this arm addresses — MEASURED, with paths

| fact | value | artifact |
|---|---|---|
| distinct classes the head predicts | **1 of 10** (2,000/2,000 slots; 1,482/1,482 matched pairs) | `raw/cls_collapse.json` |
| top-1 accuracy | 0.77935 | same |
| majority-class baseline | **0.77935** — equal to 5 dp | same |
| mean top1−top2 softmax margin | 0.59198 (confidently wrong) | same |
| GT `person` → predicted `person` | **188 → 0** | same |
| GT `rider` → predicted `rider` | **50 → 0** | same |
| imbalance, majority : rarest | **577.5 : 1** | same |
| head `l` MAE vs a single global median | 1.02306 vs 1.03962 (CI straddles 0) | `raw/size_floor.json` |
| head `w` MAE vs a single global median | 0.31953 vs 0.32174 (CI straddles 0) | same |
| median keyed on the **TRUE** class | `l` **0.39653**, `w` **0.10100** | same |

## 2. The hypothesis, and why it is a loss hypothesis

`cls` carries weight **1.0** in `SLOT_LOSS_W` and A8 ran `--w-agent 1.0`, so the collapse is not a
down-weighting artifact. But the `cls` term is a plain `cross_entropy` with **no `weight=`**
(`agent_slots.py:591`), while presence receives `NO_OBJECT_W = 0.1` introduced with the reason
(`:230-231`):

> unmatched slots vastly outnumber matched ones, and an unweighted BCE simply learns "always empty"

**`H-BOXCLS-1`:** the same imbalance argument applies to `cls` at 577.5:1, and applying the same
remedy recovers the class — and size, which is **downstream** (§1: a true-class median beats the
head's own size regression 2.6× on `l`, 3.2× on `w`).

## 3. ⛔ LAUNCH GATE — a CPU result decides whether this arm is worth GPU at all

This arm is **NOT** authorised to run on the strength of §2 alone, because a representation defect
produces an identical collapse and would not be fixed by re-weighting. The gate is
`code/cls_probe.py`: a linear probe from `box_dec`'s own per-slot features to the GT class,
episode-disjoint, PCA basis fit on the FIT split only, with **both** arms present.

| gate outcome | meaning | action |
|---|---|---|
| **BALANCED arm clears `1/K` with a separated interval, UNBALANCED arm does not** | the signal is present and an unweighted estimator collapses on it — the head's failure reproduced *and removed* by the exact proposed remedy | ⭐ **LAUNCH** |
| both arms read `1/K` | the features do not linearly separate the classes | ⛔ **DO NOT LAUNCH** on re-weighting. Next arm is upstream (queries/capacity) or a nonlinear probe first |
| balanced arm clears `1/K` and unbalanced does too | the collapse is not an imbalance effect | ⚠️ re-open §2; the mechanism is mis-identified |

⚠️ **The unbalanced arm is mandatory, and it is the interesting one.** Measured today: an
**unweighted** one-vs-rest ridge on these features read balanced accuracy **0.11111** against a
constant-predictor control of **exactly 1/9 = 0.11111** — the probe collapsed the same way the head
does. ⛔ A probe that shares the defect it is checking for cannot separate *"no signal"* from *"my
estimator collapsed"*, and reading that number as a representation defect would have blamed the
trunk on an estimator artifact.

### 3b. ⛔ Three inadmissible attempts preceded the gate result — read this before designing the arm

Each was killed by a **control**, not by inspection, and each failure is a trap the launch panel can
repeat. None is a retry of the last; the design changed every time because a control refused it.

| attempt | statistic | what the control did | cause |
|---|---|---|---|
| `cls_nonlinear.py` | macro-recall over 9 classes | shuffled control read **0.15056** vs 1/K **0.11111** — *above* no-information | classes of n = 2, 3, 4 weigh as much as one of n = 1,141; one lucky hit adds 0.5 to the mean |
| `vru_probe.py` v1 | row-level AUC, VRU vs vehicle | shuffled **0.654**, tied with the real arm | *(first diagnosis — tie-ranking in `auc` — was **WRONG**: fixing ties moved it only to 0.647, and the wrong diagnosis is recorded as wrong)* |
| `vru_probe.py` v2 | + per-arm seeds, 3-seed averaging, TRAIN-AUC gate | gate passes (1.000 / 0.784 / 1.000) yet shuffled still generalises at **0.699**, *beating both real arms* | **episode confound**: VRU density is clustered by scene, so pooled AUC rewards any model tracking scene appearance regardless of its labels |

⭐ **The fix was to CONDITION on the confounder, not to correct for it**: compute AUC **within each
scored episode** that carries both classes, then average. A model that merely ranks scenes has no
within-scene ordering and scores 0.5 by construction. The shuffled control returned to **0.47057**
immediately. ⇒ **any arm reading this head's class quality must score WITHIN scene**; a pooled
number on this corpus is not interpretable, and a pooled improvement would not be evidence.

⚠️ And the one that generalises past this experiment: **a permuted-label arm that beats the real
arm is not noise — it is a map to the confound.** It says the signal being measured lives at a
level the labels were not varied at.

## 4. Arms — exactly one variable

| arm | `cls` criterion | everything else |
|---|---|---|
| **A_ctrl** (deliberate regression) | `cross_entropy`, **unweighted** — today's configuration | identical |
| **B_bal** | `cross_entropy(weight = 1/freq)`, frequencies counted on the **TRAIN** split only | identical |
| **B_bal_rep** | as `B_bal`, **different seed** | identical |

`held_constant`: corpus + parity hash, steps, batch, lr + schedule, geometry (416×1024), all other
entries of `SLOT_LOSS_W`, `NO_OBJECT_W`, `--agent-queries`, `--agent-pad`, the join, the eval
windows. ⛔ Preflight **diffs the launch commands** and refuses if they differ in more than the
`cls` criterion — intent is not evidence, and a row-bank arm once changed `n` *and* silently
multiplied λ.

⚠️ **`B_bal_rep` is not optional.** This is a **lever** claim — a change moved a metric — so
`H-ESTIM-SEED-1` binds: a separated episode-cluster CI is **necessary, not sufficient**, and the
effect must be read against this rig's own run-to-run noise floor. Without the replicate the result
is inadmissible as a lever effect no matter how wide the interval.

## 5. Criteria — committed in advance

**PRIMARY (`cls`).** Balanced accuracy = mean per-class recall on matched pairs, episode-clustered
bootstrap, episode-disjoint eval.

* **SUCCESS:** `B_bal` − `1/K` separated **above** zero, **and** `B_bal` − `A_ctrl` separated above
  zero, **and** the difference exceeds the `B_bal` ↔ `B_bal_rep` spread.
* **FAILURE:** `B_bal` indistinguishable from `1/K`, or not separated from `A_ctrl`.
* ⛔ **RAW ACCURACY IS NOT A CRITERION AND MAY NOT BE HEADLINED.** A fully collapsed head already
  scores **0.77935** on it. It is reported beside balanced accuracy, never instead of it.
* ⛔ **`A_ctrl` MUST REPRODUCE THE COLLAPSE** (1 distinct class, accuracy == majority baseline). If
  it does not, the rig is not the rig that produced §1 and **no arm may be read**.

**SECONDARY (size, and it is the one that matters for driving).** `l`/`w` MAE per component, never
pooled, against three fixed reference points from `raw/size_floor.json`:

| reference | `l` | `w` |
|---|---|---|
| floor — one global median | 1.03962 | 0.32174 |
| today's head | 1.02306 | 0.31953 |
| **ceiling — median on the TRUE class** | **0.39653** | **0.10100** |

* **SUCCESS:** `B_bal` beats today's head with a separated paired interval on **both** components.
* **Reported either way:** the fraction of the head→ceiling gap closed, per component.
* ⚠️ Size is **not** a free win even if `cls` succeeds: the ceiling is a lookup on the *true* class,
  so a head with imperfect class recovers only part of it. Stating the fraction closed prevents the
  ceiling being read as a target.

**SAFETY, reported unconditionally.** `person` and `rider` recall, with n. Today: **188 → 0** and
**50 → 0**. Any write-up that omits these two rows is incomplete.

**FOUR FAMILIES.** This is a perception arm, so the driving families are reported as *unchanged or
not*, with the reason per family — not silently dropped (`EVAL_DOCTRINE`, the binding four-family
rule). Tier stamp: perception readout, **not** a T1 driving number.

## 6. Controls the panel carries

* **constant-only** → balanced accuracy of a constant predictor is **exactly `1/K`**. A control that
  must read a **known value**, not an estimated one.
* **deliberate regression** → `A_ctrl`, which must reproduce the collapse. A gate that cannot fail
  the broken configuration proves nothing about the fixed one.
* **replicate** → `B_bal_rep`, the run-to-run floor (§4).
* **n and d printed**, per class and in total; any class with zero train support is **named**, never
  silently dropped.
* **the true-class ceiling** (`F_CLSGT`), so "better" is measured against what the label itself
  supports rather than against nothing.

## 7. What this arm does NOT claim

* It does not address **presence** (degenerate, and NOT misalignment — ratio 1.2517 on the correct
  head). Separate arm.
* It does not address **`occluded`**, which is a **decode** change with zero training cost
  (`6c5fb62`): the channel loses to a 2-parameter read of the head's own azimuth, CI
  [−0.1844, −0.0376].
* ⛔ It does not separate **loss design** from **training duration**. A8 is `ckpt_5000`; a collapse
  at 5,000 steps could be either. `A_ctrl` at the same step count controls for this **only if** it
  reproduces the collapse — which is why §5 makes that a hard precondition rather than a nicety.
