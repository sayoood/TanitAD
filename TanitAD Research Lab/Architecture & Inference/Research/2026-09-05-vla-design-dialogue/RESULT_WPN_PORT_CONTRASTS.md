# WP-N — the port's effect IS scene-specific, and WP-M's "89 % generic" is retracted

**TanitAD_TrainingFlyWheel · 2026-09-06 · Tier T0, NON-PARITY pilot.**
Evidence class **MEASURED (ours)**. `n = 1,360` windows · 34 episodes · K = 128 ·
`refc-base-30k` · verified join · paired episode-cluster bootstrap, 4,000 iterations ·
cross-episode and norm-matched arms averaged over 20 draws.
Code `code/wpn_port_contrasts.py` · raw `raw/wpn_port_contrasts.json`, `raw/wpn_full2.log`.
**Supersedes `RESULT_WPM_PORT_INFLUENCE.md` §3.**

---

## 1. ⛔ THE RETRACTION — WP-M's permuted control was not a control

WP-M concluded that *"≈89 % of everything the port achieves is achieved by a prior
belonging to some other scene"*, and from that, that the port carries almost no
scene-specific information.

**That control was `roll_control(energy, shift=1)` — a roll by ONE ROW.** The rows are
accumulated episode by episode, so row *i−1* is almost always the same episode and usually
the adjacent timestep.

> **MEASURED: roll-1 pairs a window with the SAME EPISODE in 97.5 % of rows.**

A prior from 0.1 s earlier in the same clip is not a wrong-scene prior — it is very nearly
the right one. ⇒ **the control was ~97.5 % a no-op**, the real prior's small margin over it
(+0.0012 m) measured nothing about scene-specificity, and the 89 % figure is **RETRACTED**.

⚠️ Same family as every other specification defect this campaign: **two things compared
without the affordance that was supposed to differ actually differing.** Here the affordance
was *"a different scene"*, and the control did not deliver it.

---

## 2. The honest control, and every contrast with an interval

`cross-episode` draws each window's prior from a **different episode**, by per-row rejection
sampling with an assertion that no same-episode pair survives — so a silent failure to
mispair cannot pass as a control, which is exactly how `roll_control` failed here.

| arm | ADE (m) |
|---|---:|
| base (port zeroed) | 0.6636 |
| **REAL prior** | **0.6521** |
| roll-1 prior — *diagnostic only, 97.5 % same episode* | 0.6534 |
| **cross-episode prior — the control** | **0.6776** |
| norm-matched random prior | 0.7474 |

| paired contrast | delta (m) | CI95 | |
|---|---:|---|---|
| real − base | −0.0113 | [−0.0416, +0.0175] | overlaps 0 |
| cross-episode − base | +0.0141 | [−0.0097, +0.0363] | overlaps 0 |
| **real − cross-episode  ⭐ SCENE-SPECIFIC** | **−0.0254** | **[−0.0425, −0.0093]** | **SEPARATED** |
| real − roll-1 *(WP-M's number)* | −0.0012 | [−0.0098, +0.0068] | overlaps 0 |
| norm-matched − base | +0.0838 | [+0.0600, +0.1087] | SEPARATED |

---

## 3. What this says, stated exactly

⭐ **The port's effect DEPENDS ON THE PRIOR MATCHING THE SCENE.** A matched prior and a
mismatched one differ by **0.0254 m** with a **separated** interval. The model's selection
is genuinely a function of *which scene the prior came from* — the channel carries content,
not just magnitude.

⛔ **But "the port improves on no prior at all" is NOT established.** `real − base` is
−0.0113 m with CI [−0.0416, +0.0175], straddling zero. The separated result is a
**matched-vs-mismatched** contrast, not a **prior-vs-no-prior** one, and the two must not be
conflated.

⚠️ **A wrong-scene prior appears to HURT** (+0.0141 m vs base, not separated) and a random
prior hurts a lot (+0.0838 m, separated). So the ordering is
`real < base < cross-episode < norm-matched` — i.e. **most of the separated
`real − cross-episode` gap comes from the mismatched prior being harmful**, not from the
matched prior being helpful. That is still evidence of scene-dependence, and it is a weaker
claim than "the port helps".

⛔ **The "generic share" ratio is not reportable.** The run prints −121.7 %, but its
denominator (`real − base`) is not separated from zero, so the ratio is undefined in any
useful sense. It is retained in the raw JSON only as a diagnostic and must not be quoted.

⚠️ **One asymmetry to state:** the cross-episode and norm-matched arms are averaged over 20
draws while the real and base arms are single realisations. That is the right estimand — we
want *expected* ADE under a wrong prior — and it narrows those arms' noise legitimately, but
it means the contrast is "one real draw vs an expectation", not "draw vs draw".

---

## 4. What this changes in the design

| | |
|---|---|
| ⭐ **`DIALOGUE_07` §1–§2 premise** | **Strengthened, not weakened.** The re-ranking channel is demonstrably scene-dependent. WP-M's Amendment 5 §A5.3 read the opposite from a broken control. |
| ⛔ **The bar for TanitLang** | Still the **cross-episode** control, not the no-port arm — but now a real bar rather than a no-op: a reasoner must beat a prior drawn from another episode, and that gap is measurable at **0.0254 m** for a 5-class manoeuvre prior. |
| ⚠️ **The port-helps claim** | Remains unestablished. The existing prior's benefit over no prior is inside the noise at n = 1,360 / 34 episodes. |
| ⭐ **Instrument** | `roll_control` is correct for a batch whose rows are independent, and **wrong for rows grouped by episode**. Its docstring now says so; a cross-episode variant is the right control whenever the grouping exists. |

---

## 5. Scope

* T0, NON-PARITY pilot, one checkpoint, one inference seed, one port
  (`maneuver_to_anchor`; `lat_to_anchor` / `lon_to_anchor` / `lan_gate` are refcv3+ and
  absent from this checkpoint).
* Not a T1 number, no metric family. A capability claim needs both.
* ⛔ **A separated episode-cluster CI on a one-seed arm is necessary and not sufficient**
  (`H-ESTIM-SEED-1`). The clean replicate here is a **second inference seed**, and it has
  not been run.
* The tensors are banked (`wpn_port_tensors.pt`), so any further contrast on this arm costs
  no GPU.


---

# ADDENDUM - WP-O: the contrast REPLICATES, and it is broad-based (2026-09-06, same night)

Code `code/wpo_replicate.py` - raw `raw/wpo_replicate.json`, `raw/wpo.log`.

**Which replicate is even available here, established before one was run.** `CLAUDE.md`
names three variances the episode bootstrap is blind to. Two do not apply:

* **INFERENCE seed: NOT APPLICABLE.** `refc.py:2091` - *"noise only in training
  (deterministic at eval so decoding is reproducible)"*. Re-running the same windows
  reproduces S0 and S1 bit-for-bit. Claiming an inference-seed replicate here would be
  theatre, so none was claimed.
* **TRAINING seed:** out of scope - one checkpoint exists.
* **WINDOW SAMPLE: this is the live one.** WP-N drew 40 of ~180 available windows per
  episode under `default_rng(1234)`; the episode bootstrap resamples EPISODES and never
  asks whether a different draw of windows WITHIN them would agree.

| window seed | n | base | real | cross-episode | delta | CI95 | |
|---|---:|---:|---:|---:|---:|---|---|
| 1234 | 1360 | 0.6636 | 0.6521 | 0.6776 | **-0.0255** | [-0.0425, -0.0093] | **SEPARATED** |
| 4321 | 1360 | 0.6742 | 0.6493 | 0.6868 | **-0.0375** | [-0.0521, -0.0236] | **SEPARATED** |

=> **REPLICATES: separated with the same sign under both window draws.**

**And it is not a few episodes.** Per-episode breakdown:

| seed | episodes favouring the real prior | median per-episode effect | worst | best |
|---|---|---:|---:|---:|
| 1234 | **23 / 34** | **-0.0251** | +0.0693 | -0.1497 |
| 4321 | **26 / 34** | **-0.0253** | +0.0184 | -0.1344 |

The two medians agree to the fourth decimal across independent window draws, and a clear
majority of episodes points the same way in both. => the pooled contrast is **broad-based**,
not driven by outliers - a distinction the bootstrap's interval alone cannot make.

!! Scope unchanged: T0, NON-PARITY pilot, ONE checkpoint. This replicates the WINDOW draw,
not the TRAINING run. `H-ESTIM-SEED-1`'s training-seed caveat still stands and cannot be
discharged without a second checkpoint.
