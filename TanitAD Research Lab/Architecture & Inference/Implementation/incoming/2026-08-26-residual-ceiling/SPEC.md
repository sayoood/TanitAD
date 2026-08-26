# SPEC — E-DEC-63: what is the CEILING on predicting Δz beyond drift?

**Written** 2026-08-26 · **Author** Master Mind · **Rig** probe over banked latents
(no training run) · **Tier** T0-DIAGNOSTIC

```yaml
hypothesis: E-DEC-63
one_variable: what_the_probe_is_given      # the ONLY difference between arms
held_constant: [target, split, lambda_selection_protocol, corpus, windows, k]
success: "ORACLE-t clears the measured null (|t| >= 2.9) on the post-drift residual
          by a margin that leaves real headroom over our own predictor"
failure: "ORACLE-t is INSIDE the measured null, i.e. indistinguishable from the
          constant control -- the residual is not predictable from time-t
          observation at all"
controls: [constant_only, raw_input_floor, time_shuffled, positive_control,
           deliberate_regression]
splits: {fit: "FIT clips", val: "carved from FIT only -- lambda and PCA basis",
         test: "scored, never tuned on"}
```

---

## 1. ⭐⭐⭐ Why this is the FIRST thing v7 needs, before any training run

**What the trail established:** Δz is **64 % drift** — predictable from `z_t` alone
(r 0.674, t 134.84) — and the drift-removed residual carries **the action in 0 of 8
arms**. `splitp30k`'s predictor **adds nothing** over `z_t` (−0.0008 / −0.0320 /
−0.0158 at k=1/3/6, t −3.69 / −5.62 / −6.26).

⛔ **What it did NOT establish, and what nobody has ever measured: whether that
residual is predictable BY ANYTHING.** Fourteen rows in `GOALS_AND_CLAIMS.md`
mention the residual; **none measures its ceiling.** So the programme cannot
currently distinguish two worlds that demand opposite v7 designs:

| | if TRUE | the v7 consequence |
|---|---|---|
| **H-A: the residual is largely UNPREDICTABLE** | our predictor is already at ceiling | ⛔ **Stop optimising the predictor — we have been chasing noise.** It also *confirms* E-DEC-7's `z = (u, η)`: the residual IS the noise dimensions, unpredictable by construction. The fix is **representational**, not predictive |
| **H-B: the residual IS predictable, we just miss it** | our predictor is underpowered or mis-targeted | ✅ The loss/architecture is the lever; a teacher-target or richer predictor is warranted |

⚠️ **This is why the obvious v7 proposal was withdrawn before it was written.** The
tempting move — *"replace the self-generated target with a frozen/EMA teacher"* — is
**already refuted by an arm we have run**: `splitp30k`'s encoder is FROZEN, so its
target is exactly non-self-generated, and its predictor **still degenerated**.
⇒ **Non-self-generation is not the missing property.** Committing GPU to a teacher
recipe before this measurement would repeat the C164 pattern: a plausible lever
adopted without the crossed cell that tests it.

---

## 2. The target

For each window, the residual any predictor must beat:

```
drift_hat = OLS(Δz ~ z_t)      # fit on FIT only
r         = Δz − drift_hat     # the post-drift residual — THE TARGET
```

Scored as the C149 nrmse form, `‖r̂ − r‖ / ‖r − mean(r)‖`, so **1.0 is exactly the
constant-predictor floor** and the no-information value is unambiguous.

---

## 3. Arms — the ONE variable is what the probe is given at time t

| arm | given | must read |
|---|---|---|
| **A0 constant-only** | nothing | ⛔ **EXACTLY** the no-information value. Off it ⇒ discard the panel |
| **A1 positive control** | `z_t`, target = the **drift** component (not the residual) | ⛔ **≈ 0.674 (t 134.84)** — a KNOWN value. Off it ⇒ the rig is wrong, read nothing else |
| **A2 our predictor** | the arm's own `zhat` | where we actually are |
| **A3 raw-pixel floor** | pixels at t | a learned representation below this added nothing |
| **A4 ⭐ ORACLE-t** | the **full pre-pool token field** (16×40) at t + pixels — far more than the predictor gets | **the ceiling** |
| **A5 time-shuffled** | A4's inputs, shuffled across time | ⛔ structure surviving the shuffle is **leakage**, not dynamics |
| **A6 deliberate regression** | `z_t` with its **drift already removed** (target-orthogonal by construction) | ⛔ **MUST FAIL.** If the gate passes it, a pass on A4 means nothing |

⛔ **A6 is the skill's requirement and it is not a formality.** It re-introduces the
defect the gate exists to catch: an input from which the target is provably absent.

---

## 4. ⚠️ The traps this panel is built to survive

Each has already cost this programme a published wrong number:

1. ⛔ **`n ≪ d` is underpowered BY CONSTRUCTION, not a negative.** The token field is
   16×40×d — vastly more features than windows. **PCA basis fitted on FIT only**, and
   **`n` and `d` printed in the table.** *(2026-08-22: 2,050 features on ~700 rows made
   everything read +0.0000 and a panel concluded "the latent carries no dynamics".)*
2. ⛔ **λ selected on the scored split picks maximal regularisation and reads EXACTLY
   the floor**, beating every noisy positive estimate. λ is chosen on a val **carved
   from FIT**.
3. ⛔ **Never normalise by the target's raw energy** when it has a large constant —
   that made the latent, `[z, dz]` and raw pixels all read "+0.54" identically.
4. ⚠️ **A negative from a LINEAR probe is not a negative about learnability.** A4 runs
   **both** a linear and a nonlinear head; a linear-only null is reported as
   *"not linearly predictable"*, never as *"unpredictable"*.
5. ⛔ **The bar is the MEASURED null, |t| ≈ 2.9** (`taniteval/taniteval/null_calibration.py`,
   104 draws), **not 2.0**, and p is floored at 1/N.

---

## 5. Outcomes, committed in advance

| reading | verdict | what v7 does next |
|---|---|---|
| **A4 inside the null** (and A1/A0 correct, A6 failed) | **H-A** — the post-drift residual is not predictable from time-t observation | ⛔ **Stop optimising the predictor.** v7's world-model target changes: predict a **coarser, structured** quantity (occupancy / free-space evolution) rather than the full latent, and the encoder — not the predictor — becomes the object of work |
| **A4 clears the null with headroom over A2** | **H-B** — the residual is predictable and we miss it | ✅ The predictor/loss is the lever. The teacher-target and richer-predictor designs become admissible **and must then be run as crossed cells** |
| **A4 clears but A2 ≈ A4** | our predictor is already at the achievable ceiling | Same action as H-A, different reason — and a much stronger statement about the current arm |
| **A1 off 0.674, or A0 off its floor, or A6 passes** | ⛔ **rig invalid** | Read nothing. Fix and re-run |

⚠️ **A5 (time-shuffled) clearing the null invalidates the whole panel** regardless of
the others — it would mean the probe is reading leakage.

---

## 6. Cost, and why it is run before anything else

**No training run.** A probe over already-banked latents on the dev box, hours not
GPU-days, and it **decides whether the next v7 training arm is about the predictor or
about the encoder.** Per the operating standard (rule 5): the cheapest discriminating
experiment, pre-registered with **both outcomes committed in advance**.

⛔ **The data-blocked variant, stated so it is not silently dropped:** an **ORACLE-priv**
arm given `obstacle.offline` agent positions and velocities at t would be the strongest
form — *if a probe that knows exactly where every agent is and how fast cannot predict
how the scene latent changes, the latent's residual is not about the scene at all.* It
needs the parquet staged (Data FlyWheel, item A5/B3 in `NEXT_ARMS.md`) and is **deferred,
not abandoned**.

---

## 7. Close-the-loop obligations (same turn as the result)

- `GOALS_AND_CLAIMS.md` — E-DEC-63 status + evidence path
- `RESULT.md` here, with evidence class and tier stamps
- raw JSON under `raw/`
- ⛔ **if any control misreads, log the class in `RETRACTION_LOG.md`** — this campaign's
  eleven defects were all caught by controls, most after publication
