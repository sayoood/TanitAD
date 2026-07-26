# BLIND-IMAGINATION DRIVING — how long, how well, and what actually limits it

**Date:** 2026-07-26 (Europe/Berlin; pods log UTC). **Stream:** blind-imagination driving (new).
**Pre-registration:** `PRE_REGISTRATION.md`, this folder, written **before any rollout was executed**.
**Host:** pod2 (A40, idle). pod1 (training v2corpus), pod3 and the eval pod (Bar-A) were never touched.
🔒 PhysicalAI-AV is gated-confidential: no clip UUID and no raw content appears in this folder.

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) · `INHERITED` (another
agent/doc, **not** re-verified) · `ESTIMATED` · `HYPOTHESIS`.
**Tiers (M1):** `PROVISIONAL` (one path, unreproduced) · `CONFIRMED` (independent reproduction) ·
`DECISION-GRADE` (CONFIRMED + pre-registered + estimator named + falsifier stated).

*(Every table below is rendered from `artifacts/*.json` by `scripts/bi_report.py` and placed by
`scripts/bi_splice.py`. No number in this document is typed by hand.)*

---

# 0. VERDICT

<!-- VERDICT -->

---

# 1. The instrument already existed, and the program has been reading it at one horizon

## 1.1 The source fact, verified before anything was built

`tanitad/models/metric_dynamics.py::rollout_decode` advances its latent window by appending **the
model's own prediction**:

```python
z_hat = predictor(win_s, win_a)[1]                              # :236
dposes.append(step_readout(win_s[:, -1], z_hat))                # :237
win_s = torch.cat([win_s[:, 1:], z_hat.unsqueeze(1)], dim=1)    # :241  ← no frame is re-encoded
```

**No frame is encoded after the initial window.** So `taniteval.rollout.collect` — which produces the
program's headline `ade_0_2s` — and `train_flagship_v4.canary_rollout` / `train_flagship_v16.canary_rollout`
— which produce `wm_canary_ade_2s` — are **already blind-imagination drives**. `k` was always a free
parameter; only `k = 20` was ever read, and only under the expert's **true future actions**.

⇒ **`ade_0_2s = 0.4271` (40 eps) and `0.4108` (600 eps) ARE v1's blind-imagination numbers at 2 s
under privileged actions.** They are not open-loop-with-perception numbers. `MEASURED`, and it follows
from the code above rather than from a claim.

⚠️ **Retraction of an INHERITED premise in my own brief.** The brief cited *"a v1-line reference of
~0.452 in `canary_rollout`'s own docstring"* as the number to establish. It is the **`heldout`
split-mean 0.4522** of the very same quantity (`MODEL_REGISTRY §1.2`), i.e. the deprecated
`overlapping_holdout_se` central value — not an independent measurement. The `full_set` value is
**0.4271**. Class **C4** (inherited without re-verification) crossed with the `heldout`/`full_set`
confusion `CLAUDE.md` warns about.

## 1.2 ⭐ The fact that explains everything downstream: **the operative brain was trained to imagine 0.4 s**

`MEASURED`, from v1's **own** stored config and **own** run manifest on pod2 — not from prose:

| quantity | value | primary source |
|---|---|---|
| operative predictor JEPA heads | **horizons `[1, 2, 4]`** | `flagship4b-speedjerk-30k/config.json` (`cfg.predictor.horizons`) |
| recursive rollout in training | `rollout_k` = **4** | same `config.json` (`cfg.train.rollout_k`) |
| operative forward-metric-consistency rollout | `op_fwd_k` = **4** (trainer default) | `train_flagship4b.py:648`; **v1's run manifest `/workspace/ops/runs.d/flagship-speed.env` contains ZERO `fwd-k` overrides** (`grep -c 'fwd-k'` → **0**) |
| tactical / strategic forward rollout | `tac_fwd_k` = **16**, `str_fwd_k` = **20** | same defaults, same manifest |

> ### **The maximum horizon v1 was ever trained to imagine is 4 steps = 0.4 s.**
> `wm_canary_ade_2s` reads it at **20 steps — 5× beyond**. This sweep reads it to **185 — 46× beyond.**

This is not a criticism of the canary; it is the mechanism the canary has been measuring without
naming. And it produces a **zero-training lever** that nobody had noticed: `HierarchicalGrounding`
holds **three** step readouts and `flagship_losses.grounding_losses` trains **all three on the same
operative imagination rollout**, only over different lengths — so `step["str"]` is a **20-step-calibrated
decoder of exactly the latents `step["op"]` is being asked to decode at 20 steps.** Every grounded
number in this program uses `step["op"]`. See §5.1 and amendment **A2**.

## 1.3 Instrument certification (M3)

`taniteval/blindimag.py` **generalises** `rollout_decode` rather than replacing it: the same loop with
a switchable **state source** and **action source**. Certified three ways:

1. **Bit-identity.** `blind_rollout(state_source="imagination", action_source="true_future")` is
   `torch.equal`-identical to `rollout_decode`; `action_source="hold_last"` is identical to
   `rollout_decode(future_actions=None)`. `taniteval/tests/test_blindimag.py`, **22/22 green on the dev
   box AND on pod2** (host-compatibility check).
2. **The guards can FAIL.** Per M3 the suite feeds deliberately wrong inputs: the control arms are
   asserted to genuinely *differ* from `rollout_decode` (a control that silently equalled the arm
   under test would make the experiment vacuous); the steer/accel clamps are shown to fire; the
   oracle peek trigger is shown to fire at a zero bar and never at an infinite one; the path-deviation
   instrument is shown to return 0 on the logged path and to recover an injected 1.5 m offset.
3. **All four arms decode an identical step 1** — they only diverge from step 2, which is what makes
   the contrast attributable (`test_first_step_is_identical_across_state_sources`).

## 1.4 ⛔ The reproduction gate — passed before any new number was read

`PRE_REGISTRATION.md §8`: no E-IMAG number is quoted until two committed deployments reproduce.

<!-- TABLES:GATE -->
<!-- /TABLES:GATE -->

Both committed values reproduce, **CI bounds included** (`[0.3675, 0.4871]` at 40 and
`[0.3956, 0.4273]` at 600 — identical to `MODEL_REGISTRY §1.2a`). The 14 µm residual against the
unmodified harness is float-kernel noise from a different encode batch shape, not a code difference.
⛔ The two deployments are **different corpora** and are never substituted for one another.
`artifacts/gate_reproduction.json`.

---

# 2. E-IMAG-1 — the blind-driving horizon curve

## 2.1 Design (pre-registered; §2–§4 of `PRE_REGISTRATION.md`)

**Four arms**, differing in **exactly one tensor** — what enters the latent window each step:
**(a)** the predictor's own `z_hat` (*the thing under test*) · **(b)** the encoding of the **last real
frame**, re-appended every step — *"the world stopped"*, 🔴 **the critical control** · **(c)** the
encoding of the **true next frame** (*the ceiling*) · **(d)** constant velocity (*the floor*). Plus a
diagnostic **(c2)**: decode `(z_true_t, z_true_{t+1})` — pure latent odometry, no prediction at all.

**Action regimes, reported separately and never pooled:** **(i)** the expert's true future actions —
⚠️ **a privileged upper bound, not deployable capability**; **(ii)** the model's **own** actions,
derived from its **own** decoded motion by the exact inverse of the corpus's steer definition
(`steer = atan(2.9·κ)`, `physicalai.py:412`) — ⭐ **the deployable condition**; **(ii-0)** the last
observed action held (also deployable, no policy); and a **convention control** feeding the same
inverse the **true** motion, so an own-action penalty can be attributed to the model rather than to my
inverse.

## 2.2 The curve

<!-- TABLES:CURVE -->
<!-- /TABLES:CURVE -->

## 2.3 ⭐ `T_blind`

<!-- TABLES:TBLIND -->
<!-- /TABLES:TBLIND -->

## 2.4 The convention control — is the own-action penalty mine or the model's?

<!-- TABLES:CONTROL -->
<!-- /TABLES:CONTROL -->

## 2.5 Reading it

<!-- READING -->

---

# 3. E-IMAG-2 — what limits it

<!-- TABLES:DECOMP -->
<!-- /TABLES:DECOMP -->

<!-- DIAGNOSIS -->

---

# 4. E-IMAG-4 — the efficiency claim, with an axis that can actually move

H2 MEASURED that the compute-saving framing is **information-free**: against always-on-7,
never-escalating saves **85.7 %**, a perfect oracle **85.6 %**, the real gate **84.8 %** — the whole
span between useless and perfect is **0.1 pp** (`INHERITED`, `…/2026-07-26-h2-classifier/`). No compute
number can distinguish a good gate from a useless one.

**Pre-registered before the numbers existed (§7.3, binding):** a duty-cycle saving **< 2×** vs
always-on would be **DISAPPOINTING**; an oracle-vs-uniform gap **< 15 %** relative error reduction at
matched duty cycle would be **DISAPPOINTING** — a learned trigger would then be worth less than the
engineering to build it, and H2's failure would repeat one level up.

<!-- TABLES:DUTY -->
<!-- /TABLES:DUTY -->

<!-- DUTYREAD -->

---

# 5. E-IMAG-3 — how to maximise blind driving. **Design only; no training was launched.**

<!-- DESIGN -->

---

# 6. Limitations, stated plainly

<!-- LIMITS -->

---

# 7. What this unblocks (§7.4 requires this field)

<!-- UNBLOCKS -->

---

# 8. Amendments — recorded here, not by editing the pre-registration

| # | what changed | why, and what it can and cannot bias |
|---|---|---|
| **A1** | A **second peek base** (`hold_last`) was added beside the pre-registered `own_kinematic` base for E-IMAG-4 | An addition, not a substitution: the pre-registered base is still reported and still primary. Adding it costs one extra rollout per policy and lets the duty-cycle curve be read in **both** deployable action regimes, which matters because the two regimes do not give the same verdict (§2.5). |
| **A2** | Six **readout-level sensitivity arms** (`step["tac"]` / `step["str"]` instead of `step["op"]`) were added after the primary arm table was fixed and before any result was read | They exist because of §1.2, a fact read out of v1's own config — not out of a result. They are a **sensitivity, never part of the primary comparison**, they are named `__roTAC`/`__roSTR` everywhere, and `T_blind` for the four pre-registered regimes is computed without them. ⚠️ They are the one place in this report where an arm was added mid-run, and the reader should discount them accordingly until an independent re-run confirms them. |
| **A3** | `T_blind`'s interval is the percentile interval of `T_blind` **re-derived inside every episode resample**, not a read-off of where the point curve's CI crosses zero | Pre-registered in §4 as written; recorded here because the distinction matters: the second construction would understate the uncertainty of a *horizon*, which is a non-linear functional of the curve. |

---

# 9. Deliverable manifest

<!-- MANIFEST -->
