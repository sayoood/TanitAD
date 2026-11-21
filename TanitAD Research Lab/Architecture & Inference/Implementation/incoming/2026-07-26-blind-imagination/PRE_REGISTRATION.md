# E-IMAG — BLIND-IMAGINATION DRIVING: pre-registration

**Written 2026-07-26 (Europe/Berlin), BEFORE any rollout was executed on pod2.**
Every artifact in `artifacts/` and `perwindow/` carries a later mtime. The instrument
(`taniteval/blindimag.py` + `taniteval/tests/test_blindimag.py`, 22/22 green) existed before this
document; **no measurement did.**

**Host:** pod2 (A40, idle). pod1 (training v2corpus), pod3 and the eval pod (Bar-A) are untouched.
🔒 PhysicalAI-AV is gated-confidential: no clip UUID or raw content appears in any artifact here.

---

## 1. The PI's question, and the fact that reframes it

> *"assume the model is not consuming any more the front camera image — how long and how good is the
> model able to drive? How should we do to maximize this capability?"*

**The instrument already exists and I verified it in the source before writing anything.**
`tanitad/models/metric_dynamics.py::rollout_decode` advances its latent window by appending the
model's own prediction:

```python
win_s = torch.cat([win_s[:, 1:], z_hat.unsqueeze(1)], dim=1)     # metric_dynamics.py:241
```

**No frame is encoded after the initial window.** Therefore `taniteval.rollout.collect` (which
produces the program's headline `ade_0_2s`) and `canary_rollout` (`wm_canary_ade_2s`) are **already**
blind-imagination drives. The program has been measuring the PI's question all along and only ever
reading it at **k = 20 (2 s) under the expert's TRUE future actions**.

⇒ This stream does not build a new instrument. It **frees the two axes that were frozen** — horizon
and action source — and adds the control that turns a demonstration into an experiment.

---

## 2. Arms — pre-registered, and no arm may be added after a result is seen

**State source (what enters the latent window each step):**

| id | arm | what it is |
|---|---|---|
| **(a)** | `imagination` | the predictor's own `z_hat`. **The thing under test.** Bit-identical to `rollout_decode`. |
| **(b)** | `frozen_last` | the encoding of the **last real frame**, re-appended every step — "the world stopped". 🔴 **THE CRITICAL CONTROL.** |
| **(c)** | `full_obs` | the encoding of the **true next frame** every step. **The ceiling.** |
| **(d)** | `constant_velocity` | no model. **The floor.** (`baseline_waypoints['constant_velocity']`, extended densely.) |
| (c2) | `observed_pair` | diagnostic ceiling: decode `(z_true_t, z_true_{t+1})` — pure latent odometry, no prediction. Separates readout error from predictor error. Not one of the four arms. |
| (d2) | `hold_v0` | second floor: go straight at the current speed. Reported, not decisive. |

**(a) and (b) differ in exactly one tensor.** The predictor call, the step readout, the action stream
and the accumulation are byte-for-byte identical
(`test_blindimag.py::test_first_step_is_identical_across_state_sources` pins that they agree at
step 1 and `test_other_state_sources_are_NOT_rollout_decode` pins that they genuinely diverge after).

**Action source, reported separately and NEVER pooled:**

| id | source | status |
|---|---|---|
| **(i)** | `true_future` — the expert's logged actions | ⚠️ **A PRIVILEGED UPPER BOUND, not deployable capability.** This is what the existing canary uses. |
| **(ii)** | `own_kinematic` — the model's own action, derived from its OWN decoded motion | ⭐ **THE DEPLOYABLE CONDITION. The one that answers the PI.** |
| (ii-c) | `gt_kinematic` — the same inverse applied to the TRUE Δposes | **The convention control.** Without it an own-action penalty cannot be attributed to the model rather than to my action inverse. |
| (ii-0) | `hold_last` | zero-order hold of the last observed action — a no-policy floor. |

The action inverse is the **exact inverse of the corpus's own steer definition**
(`steer = atan(WHEELBASE · κ)`, `physicalai.py:412`, `WHEELBASE = 2.9`). The `accel` half is **not**
an exact inverse — the corpus uses the dataset's measured `ax`, deliberately not a finite difference
of speed — which is precisely why **(ii-c) is pre-registered as a required control, not an optional
extra.**

---

## 3. The measurement

**Corpus.** The **600-episode clean val** `physicalai-val-0c5f7dac3b11` on pod2 — a byte-verified,
**order-preserving superset** of the canonical 40 (`MODEL_REGISTRY §1.2a`). Read only.

**Arm.** **v1 = `flagship4b-speedjerk-30k` @ 29999** (the DEPLOYED model; *not* `flagship4b-phase0-30k`,
which is the no-speed ablation control). The brief's note that v1 imagines far better than v4 is
`INHERITED-UNVERIFIED`; it is not relied on — v1 is chosen because it is the deployed arm and it is
the arm with committed reference numbers to reproduce.

**Window set.** `starts = range(0, T − W − K_max, 8)` at **K_max = 185** — the harness's own rule
(`taniteval.rollout.collect`, `clhorizon.horizon_windows`). This yields **~1 window per episode** and
therefore **one fixed window set shared by every horizon and every arm**, so the whole curve is
paired. Windows are the wrong power unit anyway: stride 8→1 multiplies windows ×5.9 and clusters by
**exactly 0** (MEASURED, `…/2026-07-26-pod2-eval-host/`). **n will be quoted as windows AND episode
clusters, always.**

**Horizon grid (reporting):** N ∈ {5, 10, 20, 30, 45, 60, 90, 120, 185} = 0.5 s … 18.5 s @ 10 Hz.
`T_blind` is computed on the **full dense 1…185 grid**, not the reporting grid.

**Metrics.**

| symbol | definition |
|---|---|
| `de_N` | ‖pred_N − gt_N‖ at step N — **displacement error AT horizon N**. ⭐ **PRIMARY for `T_blind`** (a local statement about horizon N). |
| `ade_N` | mean over steps 1…N of ‖pred_j − gt_j‖ — cumulative. Secondary. |
| `ade_sparse_2s` | the program's `ade_0_2s`: mean over the 4 waypoints {0.5, 1, 1.5, 2 s}. Used ONLY for the reproduction check. |

**Estimator.** **Paired episode-cluster bootstrap**, `taniteval/ci.py`, **B = 2000, seed 0**,
resampling unit = **val episode**, on **identical windows**. ⛔ `overlapping_holdout_se` appears
nowhere. Single-arm intervals use the unpaired form of the same estimator with the **`full_set`**
point estimate.

---

## 4. ⭐ THE HEADLINE — `T_blind`, and how its interval is constructed

> **`T_blind` = the largest horizon N such that imagination (a) is separated-better than
> frozen-last-frame (b) on `de_N` at every N′ ≤ N.**

Contiguity from N = 1 is required so `T_blind` cannot be manufactured by an isolated significant
point in the middle of a noisy curve.

**Its interval is bootstrapped, not asserted.** Inside each of the B = 2000 episode-resamples the
*entire* Δ(N) = `de_N`(b) − `de_N`(a) curve is recomputed and `T_blind` re-derived; the reported CI
is the percentile interval of that statistic. **`T_blind` is reported as an interval in seconds, never
as a point.**

**Secondary:** `T_useful` = the largest N at which imagination's `de_N` stays below a stated bar.
Three bars, all fixed now:
* **2.0 m** — the program's own `miss_rate@2m` threshold (primary);
* **1.391 m** — the MEASURED corridor threshold (`BOOST_PROGRAM §5.4`);
* **1.0 m** — a lane-half-width-scale reference.

Plus `T_beats_cv` = the largest N at which (a) is separated-better than the CV floor (d).

---

## 5. ⭐ BOTH OUTCOMES, COMMITTED IN ADVANCE

> **OUTCOME A — imagination beats frozen-last-frame out to some `T_blind` > 0.**
> Then that is a **real budget for attention reallocation**, and it is stated in seconds with its
> interval. The efficiency claim gets an axis with genuine dynamic range (§7).

> **OUTCOME B — imagination never separates from frozen-last-frame.**
> Then **the world model's dynamics add nothing blind**, the efficiency idea cannot rest on
> imagination, and that is a **major, publishable negative.** ⛔ **It will not be softened, re-scoped,
> or re-described as "promising at short horizon".** It would also mean the program's own
> `wm_canary_ade_2s` / `ade_0_2s` measure something a frozen percept achieves equally well — which is
> a finding about the headline metric, not only about this stream.

> **OUTCOME U — UNPOWERED.** If (a) and (b) do not separate *and* the paired half-width at the
> relevant N is wider than the (a)-vs-(c) gap at the same N, the result is **UNPOWERED, not refuted**,
> and is reported as such. A 40-episode-style "not separated" has already flipped once in this program
> on power alone. Power is reported at every N regardless of outcome.

**Falsifier for the whole stream (M5.2 — a stream that cannot say what would end it is an activity):**
if `T_blind` ≤ 0.5 s (5 steps), blind imagination buys less than one control cycle of camera budget
and **the stream is closed**, with the negative recorded.

---

## 6. E-IMAG-2 — the diagnosis, pre-declared so it cannot be chosen after the fact

At every horizon: **lateral vs longitudinal decomposition** (`taniteval/lateral.py` on the dense
path — a `horizon_s = 0.4` reading on a sparse surface would mean stale code and the run would be
blocked), **drift vs variance** (mean signed error vs its spread, per axis), and the
**out-of-envelope fraction** via `taniteval/ood.py`.

⚠️ Pre-committed labelling: **the last horizon that is a genuine measurement is 0.4 s**; everything
beyond is labelled **EXTRAPOLATION**. The OOD **ratio** is NOT quoted — `sup(ratio_arr) = 1.298888`
makes the `≤1.30` test a tautology (C13). `ENV_YAW_MAX = 12°` was never measured; it is a grid
terminus (class C14).

Prior finding to test against (INHERITED, `MODEL_REGISTRY §1.2`): v1's 2 s error is ~89 % longitudinal.

---

## 7. E-IMAG-4 — the efficiency claim, and ⚠️ what would be DISAPPOINTING (§7.3, binding)

H2 MEASURED that the compute-saving framing is **information-free**: never-escalate saves 85.7 %, a
perfect oracle 85.6 %, the real gate 84.8 % — **the whole span between useless and perfect is 0.1 pp.**

`T_blind` supplies an axis that can actually move. A peek-every-`T′` policy costs `1/T′` of the
front-camera budget. Two policies at matched budget: **uniform** peeking and an **ORACLE** peek
(re-anchor only on ticks where the model's own per-step decode error already exceeds a bar — it reads
the true error, so it is privileged by construction and may never be quoted as deployable). The gap
between them is what a learned tactical trigger could win.

**Stated BEFORE the numbers exist, per §7.3:**

| quantity | ⚠️ what value would be DISAPPOINTING |
|---|---|
| duty-cycle saving at a fixed error bar | **< 2×** vs always-on. Below that the front camera is essentially always on and there is nothing to reallocate. |
| oracle-vs-uniform gap at matched duty cycle | **< 15 % relative error reduction.** Below that a learned trigger is worth less than the engineering to build it, and the H2 failure repeats one level up. |
| `T_blind` | **≤ 0.5 s.** Less than one control cycle ⇒ the stream closes (§5). |

**If the duty-cycle curve is flat between uniform and oracle, that is reported as the same
information-free result H2 found — not as a saving.**

---

## 8. Reproduction gate — no new number is quoted until a committed one is reproduced

Before any E-IMAG number is read, the new code must reproduce, on pod2:

1. **Bit-identity** with `metric_dynamics.rollout_decode` on the stub fixture — `test_blindimag.py`, **already green (22/22)**.
2. **Tensor identity** with the unmodified `taniteval.rollout.collect` on real val episodes at K = 20.
3. **v1 = `ade_0_2s` 0.4271** on the 40-episode deployment (881 windows / 40 clusters).
4. **v1 = `ade_0_2s` 0.4108 [0.3956, 0.4273]** on the 600-episode deployment (13,198 windows / 600 clusters).

⛔ These are **DIFFERENT DEPLOYMENTS** and are never substituted for one another. **If (2)–(4) do not
reproduce, this pre-registration is void and the run is reported as BLOCKED**, not adjusted.

---

## 9. Evidence tiers (M1)

Nothing here reaches a decision above **PROVISIONAL** without an independent reproduction path.
The headline `T_blind` targets **DECISION-GRADE**: CONFIRMED (two independent paths — the paired
bootstrap on `de_N` and the same verdict on `ade_N`, plus the reproduction gate) **+** pre-registered
(this document) **+** estimator named (§3) **+** falsifier stated (§5).

## 10. What is NOT done in this run

**No training is launched.** E-IMAG-3 is a **design** with pre-registered falsifiers and a ranking by
expected value per GPU-hour. Any intervention it recommends is a separate, separately-approved run.
