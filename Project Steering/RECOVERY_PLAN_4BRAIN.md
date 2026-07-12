# Recovery Plan — Full 4-Brain Flagship + Grounding-Done-Right (Sayed escalation, 2026-07-12)

**Mandate (Sayed):** full implementation and wiring of ALL FOUR brains for the flagship — no
compromise, no toys — plus an evidence-based explanation of why the grounding fine-tune brought
nothing. This is the Phase-0 core (the 4B thesis) and was aligned as Phase-0 scope.

## A. Honest state (what exists vs what functions)
| Brain | Exists | Trained? | Wired into flagship fwd? | Functional? |
|---|---|---|---|---|
| Operative | yes (action-conditioned predictor, k∈{1,2,4}) | yes | yes | imagines; metric trajectory FAILS (D1) |
| Tactical | `tactical_pred` (k∈{8,16}) + `TacticalSelector` imagine-and-select + maneuver vocab + subgoals | predictor yes; **no trained maneuver/goal policy head** | selector runs offline, not a trained head | maneuver select 0.25 (poor; bounded by rollout fidelity) |
| Strategic | `StrategicGraph` (k-means VQ + Dijkstra) | **NO — non-parametric stub** | **NO** | not evaluable — Phase-0 gap |
| Fallback | imagination-error + OOD + rule-barrier monitors | n/a (logic) | partial | exists |

**Verdict:** the *mechanisms/scaffolds* exist as designed, but a **trained, wired, functioning
4-brain hierarchy does not** — the tactical policy and the strategic brain are not trained neural
components in the flagship. Recovery = build + train + wire them, and fix the grounding.

## B. Why the grounding fine-tune brought nothing — evidence-based
Numbers: probe oracle ceiling 1.65 → **1.60** (negligible), held-out MLP 3.89 → 3.63, rollout-decode
ade_0_2s **6.0**, straight 2.56 vs CV 0.18 (does NOT beat CV). Four mechanisms, evidence for each:
1. **The representation was NOT reshaped — only read out.** The *oracle in-distribution ceiling*
   (best possible decode of the frozen fine-tuned latent) barely moved (1.65→1.60). If grounding had
   reshaped the encoder, the oracle would drop; its flatness is direct evidence the fine-tune left
   the representation's metric content essentially unchanged and the metric heads just read the same
   limited signal.
2. **Late fine-tune with the SSL losses still dominant.** 8k steps from a converged 27k checkpoint,
   with JEPA + SIGReg UNCHANGED and at full weight; the auxiliary grounding (λ_invdyn 2, λ_fwd 1) is
   too weak/late to overcome losses that keep pulling the representation to its isotropic-predictable
   form. Grounding must be **in the objective from the start**, not bolted on late.
3. **SIGReg ↔ metric-position tension (mechanistic).** Metric ego-position lives in a low-dimensional
   structured subspace; SIGReg drives the embedding toward an isotropic Gaussian (Cramér–Wold),
   actively spreading/whitening exactly that structure (this is the step-21k "regression" mechanism).
   The two objectives partially cancel. **Remedy + test:** exempt an ego-motion subspace from SIGReg,
   or down-weight SIGReg on the grounded dimensions; re-measure the oracle ceiling.
4. **Forward grounding built on poor rollouts.** L_fwc decodes displacement from the predictor's
   *recursive rollout*, whose fidelity is itself weak (D3). Grounding a readout on degraded rollouts
   caps the achievable metric accuracy.
5. **Geometry confound.** The fine-tune ran on the wrong-zoom f-theta data (1.6×, 60% of the mix) —
   the grounding fought a geometric inconsistency throughout.

**Conclusion (pre-registered falsifier honored):** grounding as a *late fine-tune on broken geometry
with SSL dominant* cannot reshape the representation. The definitive test is grounding **co-trained
from scratch, on corrected geometry, with the SIGReg-position tension resolved** — run inside the
full 4-brain flagship below. If it *still* fails, the bottleneck is encoder capacity/data and we
escalate there.

## C. The recovery build — full 4-brain flagship (no toys)
All brains parametric, trained, wired, and grounded at their level (H18). Reuse the validated REF-B
rev2 tactical/strategic head designs (they exist and are budget-proven) — ported into the
world-model (they condition the predictor + are grounded, not directly BC-supervised).

**Brain 1 — Operative** (upgrade): action-conditioned predictor (steer,accel)→future latent +
metric Δpose head; grounded (H18 operative). Keep multi-horizon {1,2,4}.

**Brain 2 — Tactical** (BUILD trained policy): a trained head state→(a) maneuver distribution over
the vocabulary {lane_keep, turn_L, turn_R, accel, brake_stop, lane_change_L/R, follow}, (b) a tactical
GOAL = 2 s sub-waypoint in metric ego-space + target latent. Conditioned by the strategic context
(FiLM); emits an intent token that FiLM-conditions the operative predictor — closing the hierarchy.
Trained: maneuver CE (kinematic pseudo-labels, class-weighted) + tactical goal grounding (the 2 s
trajectory consequence, H18 tactical). Imagine-and-select becomes: the trained head PROPOSES,
the grounded rollout SCORES.

**Brain 3 — Strategic** (BUILD trained layer): port REF-B rev2's strategic transformer (d384×4,
route-heading head route_L/straight/R over 5–30 s, context token). Parametric, trained with route CE
(long-horizon heading pseudo-labels) + strategic grounding (place-to-place consequence, H18 strategic).
Its context token conditions the tactical layer. Keep `StrategicGraph` as an AUXILIARY memory (Phase 1),
not the brain.

**Brain 4 — Fallback** (wire): imagination-error + H15 σ + confidence + rule-barrier → deterministic
MRM. Consume the trained brains' uncertainty, not heuristics.

**Wiring (hierarchy):** strategic context ⟶(FiLM)⟶ tactical ⟶(intent FiLM)⟶ operative ⟶ actions +
metric trajectory; every level grounded (H18); fallback monitors all. Frequencies: operative every
step, tactical every N_tac, strategic every N_str.

**Training (the definitive flagship):** joint, **from scratch**, on **corrected f-theta geometry**,
with L = JEPA(all levels) + hierarchical metric grounding (op+tac+strat) + maneuver CE + route CE +
SIGReg(**position-subspace-relaxed**) + imagination(H15). Budget-matched ~261 M. This is the run
that tests the whole thesis honestly.

## D. Sequencing (realistic — this is a multi-day build, not overnight)
1. **Now:** architecture build — wire the trained tactical policy + strategic transformer + hierarchical
   grounding + SIGReg-position-relaxation into `fourbrain.py` + the trainer, with tests (agent).
2. **Cheap decisive pre-checks (in parallel, existing compute):** (i) SIGReg-relaxation probe — does
   exempting an ego-subspace drop the oracle ceiling? (ii) grounding-from-scratch short run on
   corrected comma-only to see if the oracle moves when co-trained (isolates late-fine-tune vs
   objective). These tell us B.2/B.3 are the real levers before the full run.
3. **Definitive flagship run** on corrected geometry once the 4-brain trainer passes tests + the
   pre-checks are green; REF-A/REF-B mirror the tactical/strategic wiring for the fair comparison.
4. Gate on the revised Phase-0 exit (beat CV on straight+curve, closed-loop completion) — only then
   multi-cam / H-stack.

**Owner:** MVP loop builds; Architecture agent reviews; every step pre-registered with falsifiers.
Registered: **H18** (hierarchical grounding) now spans the build; the 4-brain wiring is the D-030
recovery scope.
