# E-LEWM-1 — INTERIM. The capacity control settles WHERE the fault is; no fix is validated yet

`MEASURED (ours; dev-box RTX 4060)` · **T0-DIAGNOSTIC** · pre-registered in
`PREREG_E_LEWM_1.md` (**e4d58be, committed before any number**) · scored on the
**same 5,617 keys, same order, same episode-disjoint folds** as E-TRUNK-2, so
every row sits in the same table as `v6_cells` and `dino_pooled` ·
**Thor untouched throughout.**

⛔ **STATUS: THE PRE-REGISTERED GATE HAS NOT PASSED.** No arm carrying the LeWM
objective decodes, so **the four-way ablation has not been run and no claim about
v6 follows from it.** What IS established is where the fault is *not*.

---

## 1. Results so far

| arm | between/within | SIGReg | `lead_gap_m` R² | `left_occ` | `right_occ` |
|---|---|---|---|---|---|
| `lewm` (tick = 1 frame) | 3.22 | 17.07 → 0.551 | −0.0130 | .5371 | .4112 |
| `lewm` (tick = 10 frames) | **16.25** | 30.51 → 1.636 | −0.0257 | .4803 | .5385 |
| **`wsig`** (within-episode SIGReg) | **2.99** ⭐ | 18.20 → 2.041 | −0.0068 | .4858 | .5118 |
| ⭐ **supervised CONTROL** | **1.58** | — | **+0.9934** [.9908,.9952] | **.9941** | **.9958** |
| *reference: `dino_pooled`* | *2.47* | *—* | *+0.3792* | *.8312* | *.8236* |
| *reference: `v6_cells`* | *4.56* | *7.83 (stalled)* | *−0.0176* | *.5321* | *.5890* |

## 2. ⭐⭐ The capacity control — the one unambiguous result

**Same encoder (5.44 M), same frames, same steps, same optimiser, same
episode-disjoint folds**, trained with direct supervision on the probe targets:
`lead_gap_m` **R² 0.9934**, `ego_speed` **0.9758**, occupancy **AUC 0.994/0.996**.

⇒ ⛔ **THE HARNESS IS NOT THE LIMIT.** 5.4 M parameters on 26 k frames of real
driving decode lead-vehicle distance at **R² 0.99 from a single frame**, on
held-out episodes. **Not scale. Not data. Not the encoder. Not the probe.**
The fault is in **what the objective asks for.**

⚠️ **This is a CEILING, not an arm.** A supervised encoder is not a world model
and 0.99 is never a WM result. It bounds what this harness could show.

## 3. Two hypotheses tested, both MEASURED, neither sufficient

### 3.1 The tick — a real defect, and fixing it made things worse

MEASURED: at k=1 (0.1 s) the latent moves **1.12 %** of its magnitude, so the
identity map explains **98.9 %** of the target. `k=10 → 0.157 (×14)`,
`k=60 → 0.542 (×48)`. Pixel control agrees (|Δ| 5.45 → 21.23 /255).
**Next-frame prediction at 10 Hz is nearly trivial in driving** — LeWM's Push-T
and Reacher move materially per step; a car at 0.1 s does not.

⇒ Ticking at 1.0 s with a 3-step autoregressive roll raised the prediction loss
6× (0.0118 → 0.070) — a real task at last. ⛔ **And between/within went
3.22 → 16.25.** Predicting 3 s ahead is *harder*, so the cheapest way to succeed
is to encode only what does **not** change over 3 s: episode identity. **The fix
strengthened the degenerate solution.**

### 3.2 Within-episode SIGReg — the mechanism fix works, decodability does not follow

Pre-stated in `LEJEPA_VS_OURS` §5(1) **before** these runs: LeJEPA's optimality is
for the marginal over the samples SIGReg is computed across, and isotropy **over
episodes** is satisfied by encoding *which* episode. `wsig` applies SIGReg to the
**within-episode residual** (clip-grouped batches, per-clip mean removed).

⭐ **It works on its target: between/within 16.25 → 2.99**, a **5.4× reduction**,
landing beside `dino_pooled`'s 2.47 and near the supervised control's 1.58.

⛔ **And decodability did not follow** — `lead_gap_m` −0.0068, occupancy at
chance. ⇒ **Reducing between-episode dominance is NECESSARY BUT NOT SUFFICIENT.**

## 4. ⚠️ The hypothesis this leaves, explicitly NOT yet tested

The prediction target in a **forward driving camera** is dominated by
**ego-motion-induced optical flow**. To predict the next latent given
(steering, accel) you must model how the scene sweeps past — and *other agents*
are a small, partly stochastic residual on top of that. In Push-T the action
moves **the agent**, and the agent and block **are** what changes, so the
objective cannot succeed without encoding them.

⇒ **The same objective may reward different content in the two domains.** That is
a real hypothesis with a mechanism — and it is **exactly the family of claim I
have had refuted four times tonight**, so it is recorded as untested, not
concluded. *(Refuted so far: collapse-onto-ego, "2.3 of 2048 dims" (C128),
"collapse" as framing, "freeze the encoder" (C129), "the self-target is the
problem", and the Diaconis–Freedman power argument.)*

⏳ **Running:** `wsig` and `lewm` at **20,000 steps** (4× the current 5,000, ≈24
epochs) on seed 1. Undertraining is the cheapest remaining confound and must be
excluded before §4 is even proposed.

## 5. ⛔ Why v6.5f is NOT built yet

The PI asked for the fixes to be **validated**, then v6.5f built as the result.
**No fix is validated.** What is validated is a *diagnosis* (the objective, not
the harness) and one *partial mechanism* repair (between/within). Building a v6.5f
on that would repeat exactly tonight's error pattern — acting on a hypothesis
before the next measurement refutes it.

**What v6.5f would need, in order:**
1. an arm carrying the **LeWM objective** that **decodes** — the pre-registered
   gate, still open;
2. the four-way ablation actually run against it;
3. the winning axis mapped to v6 as **repairable in place** (SIGReg placement,
   detach, loss-term count — all loss/config-only, SigReg has no state_dict keys)
   or as a **retrain** (`d_op`, 70/573 tensors change shape).

## 6. Recorded defects in this harness

* ⛔ **`train` overwrote `e_lewm_train.json`** instead of appending, silently
  dropping two trained arms from the scoring list. Fixed; records rebuilt from
  the logs and the on-disk latents.
* ⚠️ **`d2048` carries 7.37 M params vs 5.44 M** — a latent cannot be widened
  without widening what touches it. The direction favours interpretation (more
  capacity, so a loss is not for want of it), and it is declared, not hidden.
* ⚠️ **Single seed so far.** The pre-registration requires 3 and per-seed
  reporting; the 20 k runs are on seed 1 precisely so a second seed exists.
