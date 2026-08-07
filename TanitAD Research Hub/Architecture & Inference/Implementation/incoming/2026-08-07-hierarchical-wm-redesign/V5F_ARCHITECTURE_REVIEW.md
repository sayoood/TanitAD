# v5f architecture review — against the redesign axes

**2026-08-07, PI request.** Reviewed: goal propagation, hierarchical thinking, time
horizon, prediction/planning, self-action conditioning. Sources: `flagship_v4.py`,
`flagship_v15.py`/`refc.py` (AnchoredDiffusionDecoder), `config.py` (ImaginationConfig),
registry §1.8, and the live training log (MEASURED, step 21,250).

## 0. Training health (MEASURED at step 21,250 / 30,000 — 71 %, ETA ≈ 2 days)

- **Gradients flowing** on all groups (trunk 9.75, encoder 7.22, predictor 0.15); lr on
  cosine at 2.2e-5; **19.2 s/step** (961 s accumulated ÷ 50 — the log-every trap avoided).
- **Val, last 6 blocks (18.5k–21k):** `ade@2s` oscillates 0.42–0.48 (no trend);
  **`oracle_ade@2s` 0.205–0.227** — down from 0.53 at step 4k, still improving;
  `sel_gap` 0.20–0.27 (early-run was 0.27–0.49); `miss@2m` ~0.16.
- ⚠️ `canary_ade@2s` drifted 1.05 → ~1.3–1.5 over these blocks while val ade is flat —
  controller says "ok", but the canary/val divergence deserves one focused look before any
  30k claim (work item, not an alarm; the 13:00-alarm lesson applies).
- **The headline defect is unchanged in kind and now precise in size: the selector
  captures barely half the fan's quality** — plan 0.42–0.48 vs oracle 0.21. The fan is
  excellent; the choosing is not.

## 1. Goal propagation — absent where it matters

What v5f has: factorised LAT×LON×DIST **grafts on the candidate RANKING** (zero-init,
norm-clamped) and the ego-history conditioning of the head. What it does **not** have: any
goal, route, or nav conditioning of candidate **generation** — the anchors are noised and
refined unconditionally with respect to intent, and intent enters only as a scoring prior.
Consequences: (a) in an intersection the fan must happen to contain the correct branch —
nothing *steers* generation toward it; (b) the navigation capability the programme needs
("finding the right lane in difficult situations") has no input path at all. **Redesign
mapping:** E5's goal-conditioned decode applies to this head directly — `g_tac` as a
generation-conditioning token, not another ranking graft. The admissibility rule travels
(geometric goal only; never the situation classifier's output).

## 2. Hierarchical thinking — one level deep, plus a spatial (not temporal) abstraction

v5f's "hierarchy" today: the trunk's declared policy heads (unchanged, with the measured
0.55 s dwell defect) and the **H15 imagination field** — a belief over *spatially
unobserved* areas read at the tactical horizons. That is a genuinely valuable mechanism
(occlusion handling — something the redesign's temporal hierarchy does not provide) but it
is **orthogonal to hierarchical thinking in time**: there is one predictor, one latent
clock (10 Hz), one abstraction level. No tactical/strategic predictors, no goal fans above
the operative level, no slow-clock state. **Redesign mapping:** v5f is the natural
**monolithic reference arm (M)** for the §3.6 efficiency ladder — the comparison the
pillar's claim needs; and its imagination field should be *kept* and read by the future
tactical layer (the two mechanisms compose: spatial belief + temporal abstraction).

## 3. Time horizon — 2 s, short of every planning-grade anchor

Dense operative anchors at steps 1..20 @ 0.1 s = **2 s**, vs Alpamayo 6.4 s, UniAD 6 s,
nuPlan/Waymo 8 s (PUBLISHED). The 120° FOV makes the *spatial* context wide while the
*temporal* reach stays short — an odd asymmetry for a planner. **Redesign mapping:** E-H1's
verdict transfers only partially (different geometry/trunk); a w120 variant of E-H1 should
run on the 30k checkpoint before any v5f-lineage horizon decision.

## 4. Prediction & planning — the right topology with two known defects

The fan-and-select topology is sound and *measured to work at the fan level* (oracle 0.21).
Defects: (1) **the selector** — sel_gap stuck at ~50 % of plan error across 17k steps of
co-training; ranking supervision against hindsight-oracle outcomes (redesign §3.3) is the
concrete fix, and `sel_gap` must stay a first-class metric (E8.1 generalises the
instrument). (2) **anchors are free waypoints** — on REF-C-XL's fan, 72 % of candidates
were not physically flyable (v5f's own fan: UNMEASURED — the registry escalation to dump
fan geometry at 30k stands). The unicycle action-space lesson (v1.6/v1.7, and Alpamayo's
identical choice) applies: candidates parameterised as (accel, curvature) sequences are
feasible by construction and would also make the fan directly consumable by the §1.12
closed-loop pipeline and the MPC cost terms.

## 5. Conditioning on own generated actions — the §1.12 exposure, unmeasured here

v5f trains and validates **teacher-forced**: the predictor rolls under GT actions; the
emitted candidates never feed back into the roll. Every v5f val number is therefore a
**T0 number** in the new doctrine. Given what T1 did to v1.6/v1.7 (lateral collapse,
S-rate 0.98→0.05), v5f's true driving competence is **unknown until E1.4 runs its T1
eval at 30k**. Its co-trained trunk *might* be more action-controllable than v1arch's
(the planner gradients flow into the predictor — a structural difference), or less — that
is exactly what the stage-A controllability probes will measure on both trunks.

## 6. Verdict and the five concrete asks

v5f is a healthy, well-instrumented **monolithic** arm with one genuinely novel mechanism
(spatial imagination) and two measured/known defects (selector, feasibility). It is not —
and does not claim to be — the hierarchy; its role in the redesign is the reference arm
and a donor of mechanisms.

1. **T1 closed-loop eval at 30k** (E1.4) — before any capability claim.
2. **Fan-geometry + feasibility dump at 30k** (standing registry escalation) — decides the
   unicycle-anchor retrofit.
3. **Selector retrofit experiment**: ranking loss vs current, same fan — the cheapest
   attack on the 2× sel_gap.
4. **w120 E-H1** horizon probe on the 30k trunk.
5. **Canary/val divergence check** before the 30k registry row.

*(All five are post-30k items; nothing here touches the running trainer.)*
