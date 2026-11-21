# PRE-REGISTRATION — O13-EGO: action-conditioning on the ego's own dynamics

**Registered** 2026-08-26 00:05 Europe/Berlin, **before any O13 arm is trained.**
**Author** Master Mind · **Tier** T0-DIAGNOSTIC throughout
**Implementation** `stack/scripts/train_v6_staged.py` (`o13_ego_dynamics_loss`),
tests `stack/tests/test_o13_ego_dynamics.py` (9, passing), wiring smoke
`o13_smoke.sh` (2 arms, passing).

---

## 1. Why this term exists, in two measurements

**E-DEC-48b** — against a positive control at t 8.5–14.3, the action's *marginal*
contribution to predicting the **future scene** is **zero or negative**
(−0.1678, t −3.50 on `n_agents`). Nine objective terms (O1, O2, O3, O7, O8, O9,
O10, O11, PSG) all asked the action to move the scene latent. **They asked the
model to extract information observational driving data does not contain.**

**E-DEC-50** — against an IDENTITY control reading **+0.9337 (t 23.74)**, the
action **does** determine the **ego's own** dynamics: Δspeed **t 2.56**, Δyaw
**t 4.57**. And the encoder already carries the ego *levels* (speed 2.07,
yaw-rate 2.76, accel 3.10) while carrying neither *change*.

⇒ **The substrate exists, the information exists, and no objective has ever
connected them.**

---

## 2. ⛔ The design constraint, and the oracle that imposed it

The obvious form is a head on `(z_t, action_t)` → Δ(speed, yaw). **E-DEC-51
measured that this would be degenerate before any GPU was spent:**

| arm | latent adds to Δspeed | latent adds to Δyaw |
|---|---|---|
| `rdw8p30k` | **−0.0065 (t −0.06)** | **−0.0153 (t −0.12)** |
| `splitp30k` | +0.0541 (t +0.45) | (n/s) |

⇒ Such a head **learns to read the two action scalars and ignore the 2048-d
latent**; the loss falls, the metric looks excellent, and the world model learns
nothing. That is O11's degeneracy in a new costume.

**Therefore the readout sees ONLY `zhat_{t+k}`** — not the action, not `z_t`. The
action's only path to this loss is *through the predictor*. Excluding `z_t`
additionally forbids the passthrough solution. The projection is a **frozen,
parameter-free random map regenerated from a fixed seed**, so it cannot adapt to
become easy to hit (ActSWM's guard, arXiv 2607.26712, applied to a target with
measured information behind it rather than to a manufacturable separation score).

**The floor is arithmetic and exact: 1.0.** Targets are standardised per batch, so
a zero prediction scores exactly 1.0 and any constant `c` scores `1 + c² ≥ 1`.

---

## 3. The arm

```
--w-o13-ego <w>  --o13-k 4   (+ the otherwise-identical S-W recipe and seed)
```

Matched pair against the same recipe with `--w-o13-ego 0`, which the smoke has
verified is a **true no-op** (0 of 6 step rows carry any `o13_*` key).

**Read at:** step 30,000, on the **held-out** 20-clip lead-matched panel, using
the **existing** instrument `egostate.py` — an instrument built and banked
*before* this term existed, so it cannot have been shaped to flatter it.

---

## 4. ⭐ The outcomes, committed in advance

**PRIMARY read — `egostate.py`, `zhat` column, Δspeed and Δyaw.**
Incumbent (`rdw8p30k`): **−0.0445 (t −0.72)** and **+0.0670 (t 1.06)**.

| outcome | criterion | what we conclude, and what we do |
|---|---|---|
| ⭐ **CONFIRMED** | `zhat` on **either** change target reaches **t > 2** on held-out clips, AND `o5` degrades by **< 10 %** | The transition can be made to carry ego dynamics. **O13 enters the recommended config** in `PROVEN_TRAINING_SETUP.md` and the v7-full recipe. |
| 🔶 **PARTIAL** | `o13_excess > 0` in training but held-out `zhat` stays t ≤ 2 | The term is learnable but does not generalise. **Report as a training-set-only effect; O13 does NOT enter the recipe.** |
| ⛔ **DEGENERATE** | `o13_excess > 0` while `o5` degrades **≥ 10 %**, or `o13_beats_passthrough ≤ 0` at the end of training | The separation-without-accuracy failure, third occurrence. **O13 is abandoned, not retuned**, and logged as the tenth failed objective. |
| ⛔ **REFUTED** | `o13_excess ≤ 0` at step 30,000 | The predicted latent cannot be made to carry the ego's own dynamics even with a direct, controlled, frozen-readout objective. ⇒ **The remaining lever is INTERVENTIONAL DATA** (the same scene paired with different actions — simulation / AlpaSim), which is a **PI provisioning decision, not a loss change.** |

⚠️ **The REFUTED branch is the consequential one and is committed here so it
cannot be renegotiated later.** If a term aimed at the one target the action
demonstrably determines, through a readout the action cannot reach, still fails —
then the problem was never objective design, and no tenth objective should be
attempted on this corpus.

---

## 5. Guards that travel with the run

- **`o13_shuffled`** — the same loss with targets permuted across the batch, every
  step, free. Must sit near **1.0**. If it drops with the real loss, the term is
  fitting something other than the pairing.
- **`o13_on_z_t` / `o13_beats_passthrough`** — the frozen readout applied to the
  TRUE present latent. If `zhat` never beats `z_t`, the term is not earning its
  place, and that is visible **live** rather than at eval.
- **`o5`** is reported beside `o13_excess` in every row. ⚠️ **A run that improves
  O13 while O5 degrades is the degenerate solution** — the same guard O11 carries,
  for the same reason.
- ⛔ **The abort criterion is fixed now:** if at step **12,800** `o13_excess ≤ 0`
  **and** `o13_beats_passthrough ≤ 0` on a 200-step median, the arm is **killed**
  and reported as REFUTED-EARLY. *(O3 was aborted on exactly this kind of
  pre-committed criterion at step 20,400; committing it before the outcome is
  known is what makes the abort honest rather than a rationalisation.)*

⚠️ **Multiplicity:** two change targets × two arms = four primary cells. The
CONFIRMED criterion requires only one to clear t 2, so the honest report must
state **which** cleared and that the other did not.

⚠️ **Scope:** every number here is **T0**. A CONFIRMED outcome says the world
model's transition carries ego dynamics — **it says nothing about driving.**
T1 remains the primary tier and no arm in this campaign has been evaluated there.
