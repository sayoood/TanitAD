# EDIT_LOG: TanitAD living paper v1.5.1 to v1.6 (2026-11-08)

Work copy `F:/TanitAD-scratch/hwm_bands/paper/TANITAD_PAPER.md` (from `TANITAD_PAPER.tip.md`, 6,796 lines). Line numbers are those in the work copy at the moment of each edit, in order applied, so they drift by the size of earlier insertions (SEC_3_14 inserted at entry 1 adds 64 lines). `old` and `new` are truncated at about 240 / 560 characters; the work copy is the full record. Reason prefixes: P1 to P5 are the priorities of the brief.

SEC_3_14.md was inserted verbatim, so it keeps its own en dashes (ranges such as knots 1-10 written with U+2013) and its text is not edited. All other new text has no em or en dash and none of the banned words (scanned by diff against the tip).

## A. Paper edits

**A1** line 888 (insert). Reason: P1: insert SEC_3_14.md verbatim (plus one blank line) before section 4
- old: `(insertion before: ## 4. Why less data suffices: theoretical grounding
)`
- new: `### 3.14 The three bands and the top-down cascade, formally (added v1.6, PI rulings of 2026-11-08)  This section states the v8F design against the PI's definition (`REQ_HORIZONS.md`, D-HWM-BANDS-1). W ...(6950 chars)`

**A2** line 3679 (insert). Reason: P1: new section 7.22, the PLANTEST first planning read (T0, oracle goal)
- old: `(insertion before: ## 8. Discussion: self-supervision, the two-stage question, )`
- new: `### 7.22 The redesign's first planning read (2026-11-08)  **Tier T0, oracle goal, zero training. This is a world-model diagnostic and never a driving result.** It asks whether the hierarchical latent  ...(4995 chars)`

**A3** line 349 (edit). Reason: P2 §3.3: 0.1-0.5 s and 0.8-5 s horizons conflict with the PI's bands (REQ_HORIZONS R-OP, R-TAC, R-ABS); record kept, scoped pointer to 3.14 added
- old: `the 207 M operative path is the only component at full rate.
`
- new: `the 207 M operative path is the only component at full rate. *(Scope, v1.6: the horizons and rates in this
paragraph record the generation it was written for. The definition in force is §3.14: operative up to 2 s,
tactical up to 6 s, strategic 6 to 30 s, and a level whose horizon is 0.8 s is not a long-term abstract level.)*
`

**A4** line 593 (edit). Reason: P2 3.9: strategic horizon 6-15 s and 'goals condition downward' as predictor inputs conflict with R-STR and SPEC 9; record kept, scoped pointer
- old: `are the defect.

**Control-response post-training (stage A), the mathematics.**`
- new: `are the defect. *(Scope, v1.6: this table and the 6–15 s strategic horizon record the design of 2026-08-07. The strategic band
is 6 to 30 s and the downward interface is the tactical action u_t, both stated in §3.14.)*

**Control-response post-training (stage A), the mathematics.**`

**A5** line 727 (edit). Reason: P2 3.11: equations put goals inside predictors, table dims/rates differ from as-built (AUDIT_MODEL 1, 2; SPEC 9.1; REQ R-EGO R-T2O)
- old: `  ẑ_O(t+j) = P_O( z_O(t), (a, κ)_t │ g_tac )
`
- new: `  ẑ_O(t+j) = P_O( z_O(t), (a, κ)_t │ g_tac )

*Scope (v1.6).* These equations, the table above and the dimensions in it are the v6 design. The v8F stack as built differs in four
ways, stated in §3.14: (i) the world-model predictors are goal-free at every level, so the downward conditioning lives in the
planner cost and in the reader cascade and not inside P_ℓ; (ii) the tactical layer emits goals **and** constraints, that is the
vocabulary code u_t over the horizon up to 6 s, and u_t is the action of the operative layer; (iii) every layer reads the curren ...`

**A6** line 741 (edit). Reason: P3 arithmetic 3.11(1): 'three steps' holds only for tau_S = 5 s (0.2 Hz); table gives 0.2-0.5 Hz
- old: `A strategic roll over 15 s is
three steps of a 256-dimensional model — the hierarchy`
- new: `At the 0.2 Hz end of its clock range a strategic roll over 15 s is
three steps of a 256-dimensional model (at 0.5 Hz it is about seven, and a 30 s roll is six steps at 0.2 Hz) — the hierarchy`

**A7** line 761 (edit). Reason: P2/P3 3.11: isolation claim stated as by-construction; as-built detach_upper default False (v8f_mini.py:215); also G direction covers only upper->lower
- old: `because isolation is an *architecture* property, not a training-state property; and a zero-init
head produces a zero gradient that also looks like isolation, which a dedicated test pins.
`
- new: `because isolation is an *architecture* property, not a training-state property; and a zero-init
head produces a zero gradient that also looks like isolation, which a dedicated test pins.

*Scope of the isolation claim (v1.6).* G as written blocks the **upper** level's loss from the lower level's parameters (the
upward links are `sg[·]`). It says nothing about the converse edge, the **lower** level's loss reaching the upper level's
parameters, which is the edge the PI's rule needs at 0.0 (`∂L_O/∂θ_T = 0`, `∂L_T/∂θ_S = 0`). In the v8F stack that edge is th ...`

**A8** line 784 (edit). Reason: P2 3.11: horizon-spec paragraph, add as-built single-decoder statement per SPEC 9.1
- old: `The 0–2 s
band carries operative control authority and the 2–6 s band is shaped by g_tac conditioning, but
they share one integrator, so the 2 s seam is **discontinuity-free by construction** and the seam
metrics verify rather than repair it.`
- new: `The 0–2 s
band carries operative control authority and the 2–6 s band is shaped by g_tac conditioning, but
they share one integrator, so the 2 s seam is **discontinuity-free by construction** and the seam
metrics verify rather than repair it. *(As built, one decoder of the operative layer outputs all 30 knots at 0.2 s, conditioned
on the tactical action u_t; the band split is in the loss and the metrics, §3.14.)*`

**A9** line 734 (edit). Reason: P2 3.11 scope sentence: separate band-design strides from M2b strides, as in 3.14
- old: `with steps 0.2, 0.6 and
3.0 s and readers run as sample-and-hold at about 10, 2 and 1 Hz (not 10, 1–2 and 0.2–0.5 Hz).`
- new: `with world-model steps of 0.2, 0.6 and
3.0 s in the band design (0.2, 0.4 and 0.8 s in the M2b generation) and readers run as sample-and-hold at about 10, 2 and 1 Hz
(not 10, 1–2 and 0.2–0.5 Hz).`

**A10** line 4262 (edit). Reason: P2 11.4: opening requirement sentence conflicts with REQ_HORIZONS (R-OP, R-TAC, R-STR, R-ABS); scoped sentence
- old: `with the
strategic level predicting on its own subspace. A single-rate rollout cannot serve all three.
`
- new: `with the
strategic level predicting on its own subspace. A single-rate rollout cannot serve all three. *(Scope, v1.6: this is the
REF-A v1 requirement as it stood. The PI's definition in force is operative up to 2 s, tactical up to 6 s and strategic 6 to 30 s,
with the trajectory the 6 s combination of operative and tactical (§3.14). The table below records REF-A v1 as built: only its
tactical row is a band, and its operative and strategic rows reaching 6.0 s do not meet R-OP and R-STR.)*
`

**A11** line 4271 (edit). Reason: P3 11.4 math: formula is 0/0 at lambda = 1 although the text allows lambda >= 1; limit K*eps added
- old: `&nbsp;&nbsp;&nbsp;&nbsp;E(K) ≈ ε · (λ^K − 1)/(λ − 1)
`
- new: `&nbsp;&nbsp;&nbsp;&nbsp;E(K) ≈ ε · (λ^K − 1)/(λ − 1)&nbsp;&nbsp; (equal to K·ε in the limit λ → 1)
`

**A12** line 4282 (edit). Reason: P3 11.4 math: saving from K=4 vs K=30 assumes equal eps per step across levels; narrowed
- old: `but the strategic level pays only K = 4 of
compounding to get there rather than K = 30.`
- new: `but the strategic level pays only K = 4 of
compounding to get there rather than K = 30 (if the per-step error ε did not grow with Δt; a coarser step carries a larger ε, so
this is an upper bound on the saving, not a measured one).`

**A13** line 4346 (edit). Reason: P3 11.6 arithmetic: 300 x 30 x 0.160 s = 1,440 s, not 325 s; derived (not measured) figure, flagged UNVERIFIED rather than silently replaced
- old: `DINO-WM's published configuration (300 samples × 30
iterations) would therefore cost **325 s per MPC tick**.`
- new: `DINO-WM's published configuration (300 samples × 30
iterations = 9,000 candidate rollouts, which is 1,440 s at 160 ms each) would therefore cost on the order of a thousand seconds per MPC tick (the **325 s** first printed here does not reproduce from the stated factors, and its derivation is UNVERIFIED).`

**A14** line 822 (edit). Reason: P3 3.12: 'selection degrades as fan deepens' not supported by cited K-sweep (w7_gate_k8/32/64.json: 0.577245/0.517281/0.531908, net improvement 8->64); evidence reassigned to fixed-N rank and stratum; Spearman vs Gaussian-copula rho mapping added
- old: `*better* on a deeper fan. **Our measurement therefore refutes Model A.** Selection degrades as
the fan deepens (frozen trunk K = 8/32/64 → 0.5772/0.5173/0.5319; repaired trunk N = 256 →
3.3348; different trunks, same direction, §7.16), and  ...`
- new: `*better* on a deeper fan. **Our measurement therefore refutes Model A at the tail.** The fan-depth series is *not* the evidence: the
frozen-trunk K-sweep reads 0.5772/0.5173/0.5319 m at K = 8/32/64 (one seed, no interval; `w7_gate_k{8,32,64}.json`, `w7_selected_ade`),
which improves from K = 8 to 32, worsens from 32 to 64, and ends 0.0453 m *better* at 64 than at 8, so it neither confirms nor
refutes the model; and the repaired trunk at N = 256 (3.3348) is a different trunk and not a point on that curve (§7.16). The evidence
is at fixed N = 256: the argm ...`

**A15** line 861 (edit). Reason: P3 3.12 Model B: 'monotone divergence is what we observe' not supported by the non-monotone K-sweep
- old: `This is the
predicted monotone divergence, and it is what we observe. It also predicts`
- new: `This is the
predicted divergence in N; what we have observed is its fixed-N face (the flat low-cost stratum above), and a single-trunk sweep over N
that would show the divergence directly has not been run (the frozen-trunk K-sweep is non-monotone). It also predicts`

**A16** line 2737 (edit). Reason: P3 7.16 record: same unsupported degradation claim; narrowed in place with pointer to 3.12
- old: `numbers *do* settle is a model comparison stated in §3.12: under a homoscedastic Gaussian-copula
model of a noisy-but-honest cost, deepening the fan would make the argmin *better*
(E[error | argmin] scales with −ρ·√(2 ln N)); the measured d ...`
- new: `numbers *do* settle is a model comparison stated in §3.12: under a homoscedastic Gaussian-copula
model of a noisy-but-honest cost, deepening the fan would make the argmin *better*
(E[error | argmin] scales with −ρ·√(2 ln N)); the measured tail behaviour at fixed N (mean error-rank 132.3 of 256, flat low-cost stratum) therefore **refutes that
model** (the K-sweep itself is non-monotone, 0.5772 / 0.5173 / 0.5319, net better at K = 64 than at K = 8, and does not carry the argument; v1.6)`

**A17** line 2736 (edit). Reason: P3 7.16: 'the direction agrees' false against w7_gate_k*.json (0.5772 -> 0.5319 improves)
- old: `curve** — the direction agrees, the series does not, and we do not draw it as one.`
- new: `curve** — the frozen-trunk series is non-monotone and ends better at K = 64 than at K = 8, so the two do not even agree in direction, and we do not draw them as one.`

**A18** line 857 (edit). Reason: P3 3.12 Model B: limit statement needs the stated separation condition between cost on D_delta and off it; added (OVERSTATED as written)
- old: `positive mass near 𝒟_δ,

  P( i\* ∈ 𝒟_δ ) → 1 as N → ∞,`
- new: `positive mass near 𝒟_δ **and the cost on 𝒟_δ lies below its infimum off 𝒟_δ** (an assumption, not a derived fact: a
candidate the world model reproduces well can also be self-consistent away from 𝒟_δ, and then only the weaker statement that the
argmin is *drawn toward* 𝒟_δ holds),

  P( i\* ∈ 𝒟_δ ) → 1 as N → ∞,`

**A19** line 2734 (edit). Reason: P3 7.16 heading asserted worsening with depth; the K-sweep is non-monotone
- old: `**Why the fan gets *worse* to minimise over as it deepens, and the model this refutes.**`
- new: `**Why a deeper fan does not help to minimise over, and the model this refutes.**`

**A20** line 885 (edit). Reason: P3 3.12: generalisation sentence outran Model A/B; scoped
- old: `and widening is exactly what makes a degenerate or tail-uninformative cost worse while the
reported oracle improves.`
- new: `and under the degenerate-minimiser model (Model B) widening is exactly what makes a degenerate or tail-uninformative cost worse while the
reported oracle improves; under Model A it helps, and which holds is an empirical question per cost.`

**A21** line 345 (edit). Reason: P3 3.3: 'strictly smaller bound at each level' does not give a smaller total under a linear-in-T bound; product->sum needs independence of the level components
- old: `regret grows linearly with horizon T (Cui et al. Thm 4.2), so factorizing one long horizon into
per-level short horizons pays a strictly smaller bound at each level; (ii) hierarchical
recombination of maneuvers and places converts sample co ...`
- new: `regret grows linearly with horizon T (Cui et al. Thm 4.2), so each per-level short horizon carries a smaller bound than the one
long horizon; but if the bound is linear in T the per-level bounds add back to the undivided one at equal constants, so on its own
the factorisation buys no bound, and any gain has to come from a smaller constant at a coarser level or from (ii) (the theorem was
not re-read against its primary in this revision); (ii) hierarchical
recombination of maneuvers and places converts sample complexity from the product to the sum of
level ...`

**A22** line 987 (edit). Reason: P3 4(1): 'win precisely when' (iff) and 'degenerate k=n endpoint' outran a sufficient-condition bound for rank-k factorisation
- old: `Latent models win precisely when moderate k captures task dynamics while
   pixels carry nuisance — the driving regime by construction. Pixel-level prediction is the
   degenerate k = n endpoint paying maximal sample complexity.`
- new: `The bound favours latent models when moderate k captures task dynamics while
   pixels carry nuisance, which is the driving regime we argue for (an argument about the bound's trade-off, not a measurement that
   it holds here). Pixel-level prediction can be read as the k = n endpoint of the same trade-off, with the largest estimation term;
   the cited theory is stated for the rank-k factorisation, not for a pixel decoder, so that reading is an analogy.`

**A23** line 992 (edit). Reason: P3 4(2): the cited 0.05-0.11 is the frame-change fraction (per 3.2), not the action's share of the change; wording narrowed
- old: `the action's consequence is a dominant, predictable fraction of frame change (measured
   0.05–0.11 per step on our corpora).`
- new: `the action's consequence is taken to be the dominant, predictable part of frame change. What was measured
   is the per-step frame-change fraction, 0.05–0.11 on our corpora (A8, §3.2): most of each frame is static, which is what makes a
   change-weighted loss concentrate gradient. That the changing part is the action's consequence is the premise, and the causal
   panel of §7.3 is the evidence for it, not this number.`

**A24** line 4628 (edit). Reason: P3 13.1: calling (u, eta) an 'optimum' outran the argument: a predictable scene code lowers O5; narrowed to 'solution that satisfies every term'
- old: `**So \((u,\eta)\) — ego motion plus noise — is a non-collapsed optimum carrying
no scene information.** This is not a hypothetical:`
- new: `**So \((u,\eta)\) — ego motion plus noise — is a non-collapsed solution that satisfies every
term and carries no scene information.** It is not shown to be the global minimiser: a predictable, scene-encoding \(\eta\) would
lower the O5 term by removing its irreducible noise cost, so the argument establishes that the objective does not *force* content
and that optimisation can settle here, which the measurements say it did. This is not a hypothetical:`

**A25** line 5525 (edit). Reason: P3 14.1: keep wording consistent with the narrowed 13.1
- old: `(§13.1's \((u,\eta)\) optimum is one construction)`
- new: `(§13.1's \((u,\eta)\) solution is one construction)`

**A26** line 5759 (edit). Reason: P3 14.7 arithmetic: 12.5/1 = 12.5 and 12.5/0.1 = 125, not 10-100x; the reference fractions are UNVERIFIED
- old: `**12.5 % of the run** — proportionally 10–100× above every published operating
point (data2vec's start ≈ 1 % of run; BYOL's constant-0.99 ≈ 0.1 %)`
- new: `**12.5 % of the run** — proportionally about 12–125× above the published operating
points it is compared with (12.5 % against data2vec's start ≈ 1 % of run, and against BYOL's constant-0.99 ≈ 0.1 %; those two fractions are not re-derived here)`

**A27** line 205 (edit). Reason: P2 abstract: scope note on horizons/hierarchy (3.14), parameter envelope (v1 261/263.4 M vs v8F <= 800 M, Plan 13.1), encoder, vision-only scope
- old: `has not yet trained.

---

## 1. Introduction`
- new: `has not yet trained.

*Scope note (v1.6, 2026-11-08):* the rounds above report the generation each was measured on, and no number in them moves. For the v8F redesign the definition in force is §3.14. The operative level covers 0 to 2 s, the tactical level up to 6 s with the tactical vocabulary, and the strategic level 6 to 30 s with route checkpoints, detected context and derived goals; the trajectory is the 6 s combination of operative and tactical. The tactical layer emits goals and constraints that are the operative layer's action, and all three layer ...`

**A28** line 235 (edit). Reason: P2 §1 contribution 1: horizons and parameter-envelope scope
- old: `operative/tactical/strategic/fallback decomposition, trained end-to-end with a single
   anti-collapse mechanism and no teacher-student heuristics (§3).`
- new: `operative/tactical/strategic/fallback decomposition, trained end-to-end with a single
   anti-collapse mechanism and no teacher-student heuristics (§3). *(Scope, v1.6: the bands are operative up to 2 s, tactical up
   to 6 s and strategic 6 to 30 s, §3.14. The ≈ 261 M envelope is the v1 generation's; v8F is budgeted at most 800 M since 2026-10-17.)*`

**A29** line 3813 (edit). Reason: P2 §9: horizons/hierarchy scope pointer
- old: `## 9. v3 design: hierarchical goal-vocabulary planning over the world model
`
- new: `## 9. v3 design: hierarchical goal-vocabulary planning over the world model

*Scope, v1.6: this section records the v3 design of 2026-07-19 and its gates. The bands in force are operative up to 2 s, tactical up to 6 s and strategic 6 to 30 s (§3.14); where a table below reads a strategic ADE over 8 to 30 s or a 0.8 s tactical reach, it records the design as it stood.*
`

**A30** line 3927 (edit). Reason: P2 §10: scope pointer for horizons, hierarchy and the parameter envelope
- old: `## 10. Roadmap
`
- new: `## 10. Roadmap

*Scope, v1.6: the roadmap below is the v1.5 ordering. The v8F redesign adds the work of §3.14 (band-split world model, band labels and HF release, strategic heads, the top-down planner with its controls); its first read is §7.22. The ≤ 300 M envelope quoted in item 11 below is the v1 invariant; v8F is budgeted at most 800 M (PI, 2026-10-17).*
`

**A31** line 4134 (edit). Reason: P2 §10.1 item 6: the 'PI's call' on the envelope was made 2026-10-17 for v8F; state scope, keep record
- old: `   no longer holds. *(A related scope error is worth recording beside it:`
- new: `   no longer holds. *(Scope, v1.6: for v8F the PI set the envelope at most 800 M on 2026-10-17, `Project Steering/PLAN_V8F_REFAV1_HJEPA_2026-10-17.md` §13.1; the 336.5 M run measured here sits inside it, and the sub-300 M sentence above stays as the record of the earlier envelope.)* *(A related scope error is worth recording beside it:`

**A32** line 4908 (edit). Reason: P2 vision-only sweep (13.x PSG): 'inference remains vision-only' broader than the 2026-08-03 rule's scope
- old: `model, and it is training-time only, so inference remains vision-only.`
- new: `model, and it is training-time only, so inference of the latent stays image-driven. (Scope, v1.6: the planner layers of v8F also read measured ego state, §3.14; the PI's vision-only rule of 2026-08-03 concerns the scenario classifier and the labels.)`

**A33** line 2657 (edit). Reason: P2 vision-only sweep (9.x label pipeline): scoped to the pipeline
- old: `**inference stays
  vision-only** (nothing from this pipeline is ever an inference-time input)`
- new: `**inference stays
  vision-only for this pipeline** (nothing from it is ever an inference-time input; the planner layers separately read measured ego state, §3.14)`

**A34** line 3 (edit). Reason: P4 status line to v1.6 (2026-11-08)
- old: `**Status:** living paper, **v1.5.1 (2026-10-18)**.`
- new: `**Status:** living paper, **v1.6 (2026-11-08)**.`

**A35** line 6925 (insert). Reason: P4 changelog entry v1.6
- old: `(append at end of Changelog)`
- new: `- **v1.6 (2026-11-08): the three bands defined, the top-down cascade stated, the first planning read, and a mathematics audit.** **New §3.14**: the PI's definition of the horizons (operative up to 2 s, tactical up to 6 s with the tactical vocabulary, strategic 6 to 30 s with route checkpoints, detec ...`

**A36** line 6925 (edit). Reason: P4 wording: remove banned phrase from own changelog text
- old: `§13.1 no longer calls (u, eta) a proven optimum;`
- new: `§13.1 states (u, eta) as a solution that satisfies every term, not a proven optimum;`

**A37** line 598 (edit). Reason: P4 style: no en dash in new text
- old: `this table and the 6–15 s strategic horizon record`
- new: `this table and the 6 to 15 s strategic horizon record`

**A38** line 741 (edit). Reason: P4 style: no en dash in new text
- old: `(not 10, 1–2 and 0.2–0.5 Hz).`
- new: `(not 10, 1 to 2 and 0.2 to 0.5 Hz).`

**A39** line 996 (edit). Reason: P4 style: no en dash in new text
- old: `is the per-step frame-change fraction, 0.05–0.11 on our corpora`
- new: `is the per-step frame-change fraction, 0.05 to 0.11 on our corpora`

**A40** line 5766 (edit). Reason: P4 style: no en dash in new text
- old: `proportionally about 12–125× above`
- new: `proportionally about 12 to 125 times above`


## B. Report text edits (`E:/Projects/TanitAD-artifacts/dagland/report_v45/sec_750_horizon_bands.md`; `edits_v45.md` needed no change: it carries no knots, hand-over or vision-only statement)

**B1** line 77. Reason: P5 design table: T emits vocabulary and constraints conditioning O, O outputs all 6 s (REQ R-T2O, R-OUT; SPEC 9.1)
- old: `| Outputs | trajectory knots 1 to 10 | tactical vocabulary and trajectory knots 11 to 30 | time-indexed checkpoints at 6 to 30 s, detected context, derived goals |`
- new: `| Outputs | the whole 6 s plan, 30 knots (one trajectory, or one per tactical hypothesis); band split in the loss and the metrics (knots 1 to 10, knots 11 to 30) | vocabulary (goals, lateral and longitudinal classes) and constraints, up to 6 s, which condition O; no trajectory head, no knots | time-indexed checkpoints at 6 to 30 s, detected context, derived goals |`

**B2** line 79. Reason: P5 line 79: no knots from T, O outputs the whole 6 s, every layer reads current ego state (SPEC 9.1, R-EGO)
- old: `a 6 s trajectory made of the operative knots 1 to 10 and the tactical knots 11 to 30 with a continuity term,`
- new: `a 6 s plan emitted whole by the operative level (30 knots, scored per band on knots 1 to 10 and knots 11 to 30, with no continuity term because one decoder emits every knot) and conditioned on the tactical level's vocabulary and constraints, which the tactical level emits without a trajectory head, every layer reading the current measured ego state,`

**B3** line 19. Reason: P5/P2 vision-only claim broader than the 2026-08-03 rule's scope
- old: `Standing rules that continue to apply: inference is vision-only, labels may use ego state and metadata (PI, 2026-08-03);`
- new: `Standing rules that continue to apply: for the scenario classifier and the labels, inference is vision-only and labels may use ego state and metadata (PI, 2026-08-03), while the planner layers read the current measured ego state (R-EGO, below);`

**B4** line 17. Reason: P5: rows for the PI's ruling of 2026-11-08 that the design table now implements
- old: `| R-PLAN | The multi-hierarchy world-model planner is the baseline of the redesign and is compared with other approaches (the reactive v8F planner, a flat world-model planner, the kinematic floor P10)`
- new: `| R-PLAN | The multi-hierarchy world-model planner is the baseline of the redesign and is compared with other approaches (the reactive v8F planner, a flat world-model planner, the kinematic floor P10). |
| R-EGO | Strategic, tactical and operative each read the current measured ego state; no layer reads a state planned by another layer. |
| R-T2O | The tactical layer emits goals and tactical const`

**B5** line 84. Reason: P5: record the H-JEPA consistency review in two sentences
- old: `**7.50.5 What §7.50 establishes`
- new: `The design was reviewed against H-JEPA (arXiv 2610.06805) in `horizon_review/REVIEW_HJEPA_CONSISTENCY.md` [M, code and paper read; no planner result]: per-level encoders and predictors, SIGReg and inverse dynamics at every level and the planner's Eqs. 4 to 6 match; the frozen DINOv3 encoder and the added k-step loss are declared deviations; and nine gaps (G1 to G9) were found, among them the plann`


Check: `python -I check_numbers_v45.py` exit 0, 391 checks, 0 failed. No number was added, so the checker was not extended.
