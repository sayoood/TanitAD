# PRE-REGISTERED DISCRIMINATOR — did the O5 EMA teacher MOVE BY GRADIENT?

**Written BEFORE any measurement was run.** ArchInf FlyWheel · 2026-09-06 · 0 GPU.

## The question

`apply_stage_freeze` un-freezes `ema_o5_enc.*` / `ema_o5_ro.*` (group `aux`, trained by stage S-W),
so the O5 EMA teacher is `requires_grad=True` and **in the optimizer**. My predecessor asserted the
teacher "never moved" because `_EmaCopy.forward` is under `@torch.no_grad`, so `p.grad is None` and
AdamW skips the parameter entirely.

⛔ **That is a CODE READING, not a measurement.** `in the optimizer` does not imply `received
updates`, and equally `no_grad in one call site` does not imply `no gradient ever reached it` —
`_EmaCopy` may be *called* from more than one place. This turn tests it against the banked bytes.

## Hypotheses

* **H_EMA** — the teacher moved by EMA only. `p.grad` stayed `None`; AdamW never collected it;
  **decoupled weight decay never applied.**
* **H_GRAD** — the teacher was stepped by AdamW. Then, whatever the gradient's magnitude,
  **decoupled weight decay applied on every step**, because torch's AdamW applies
  `p.mul_(1 - lr*wd)` to every parameter it *collects*, and it collects exactly those with
  `p.grad is not None`.

## D1 — the decoupled-weight-decay signature (PRIMARY)

For every tensor `T` present as both `ema_o5_enc.T` (teacher) and `encoder.T` (student), measure

    r(T) = ||teacher_T||_2 / ||student_T||_2

* **H_EMA predicts r ≈ 1.0** — the teacher is a lagged convex average of the student's own weights,
  so its norm tracks the student's.
* **H_GRAD predicts r ≤ prod_t (1 - lr_t * wd), systematically BELOW 1 on every tensor.**
  At the v7-tiny recipe's `lr 1e-4, wd 0.05` over `N = 30,000` steps that is
  `(1 - 5e-6)^30000 = 0.8607` ⇒ **a −13.9 % norm shrink, and that is a FLOOR** (a real gradient
  moves it further).

**Decision rule, committed in advance:** median `r` over the teacher's tensors.
`r > 0.99` ⇒ H_EMA. `r < 0.95` on a majority of tensors ⇒ H_GRAD. Anything between ⇒ INCONCLUSIVE.

## D2 — the frozen-tensor identity test (DECISIVE, ~10^5-10^6 separation)

Any tensor that is **frozen in the student** for the whole run is a CONSTANT `x`. For it:

* **H_EMA:** `ema <- d*ema + (1-d)*x` with `ema_0 = x` ⇒ the teacher stays at `x` up to float32
  rounding ⇒ relative deviation **<= ~1e-6**.
* **H_GRAD:** decoupled weight decay shrinks it ~13.9 % ⇒ relative deviation **~1.4e-1**.

Measure `max|teacher - student| / ||student||_inf` on the frozen set. The two hypotheses are
separated by roughly five to six orders of magnitude, so this test does not depend on a threshold
being tuned.

## D3 — teacher/student geometry (CORROBORATING ONLY, explicitly weaker)

`cos(teacher_T, student_T)` and `||teacher-student|| / ||student||`. Under pure EMA the teacher is a
convex combination of the student's *past* weights and must sit close to its trajectory.
⚠️ A gradient-trained teacher initialised FROM the student could also stay close, so a "close"
reading here supports but never establishes H_EMA. Reported as corroboration, never as the verdict.

## Controls — each MUST read a known value

| id | control | value it MUST read | what it proves if it fails |
|---|---|---|---|
| **C1** | `predictor_op.heads.1.weight` md5, across the two EMA arms | **DIFFERENT** (predecessor: 9 distinct fingerprints over 9 arms) | my reader is not reading real per-arm bytes |
| **C2** | `predictor_op.out_proj.weight` + `heads.2.weight` md5, across arms | **IDENTICAL** (predecessor: 3 values total over 9 arms; `rdw8p30k` bit-identical to a fresh seed-0 init after 30 k AdamW steps) | the instrument cannot detect "no gradient ⇒ no movement" at all, so a null result from it is meaningless |
| **C3** | `ema_o5_enc.*` keys present in the checkpoint | **PRESENT** | if ABSENT, D1-D3 cannot run ⇒ the answer is **INCONCLUSIVE**, never H_EMA and never H_GRAD |
| **C4** | the arm's own recorded argv / config | must contain **`--o5-target ema`** and state `lr`, `wd`, `steps`, `ema_decay` | ⛔ A GATE ROW CARRIES ITS ARM — three arm-substitutions have already been found in v7-land. A checkpoint whose argv does not say `ema` is the wrong arm and its reading is void. |

## The three admissible outcomes, committed in advance

1. **The teacher moved by gradient** (D1 median r < 0.95 and/or D2 ~1e-1) ⇒ the v7 recipe was never
   tested as specified; scope which claims are affected.
2. **The teacher moved by EMA only** (D1 r ~ 1.0 and D2 <= ~1e-6) ⇒ the un-freeze is a
   **budget/accounting defect**, real but **not** an algorithmic one. Say so and do **not** inflate it.
3. **INCONCLUSIVE** — the banked artifacts cannot distinguish these (e.g. C3 fails, or D1 lands in
   the 0.95-0.99 band). ⭐ A legitimate answer; reported as INCONCLUSIVE, never as either other.

⛔ No threshold, control or outcome below this line was changed after seeing data.
