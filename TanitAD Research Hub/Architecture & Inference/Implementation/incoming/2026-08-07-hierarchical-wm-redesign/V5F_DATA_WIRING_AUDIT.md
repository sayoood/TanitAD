# v5f data-wiring audit — PI assurance request, MEASURED off the run's own config

**2026-08-09.** Every ✅ below is read from `flagship-v5f-w120-30k/config.json` on pod5
(the artifact the trainer actually runs), not from code defaults. Requirements from the
PI verbatim: complete input data incl. resolution and past measurements; predictor
actioned by planner output; hierarchical wiring; strategic/tactical data diets.

## R1 — input data completeness & resolution: ✅ VERIFIED (with two stated capability limits)

| item | config value | verdict |
|---|---|---|
| geometry | `256x640f305.5775cyl`, achieved HFOV **120.0°** (requested 120.0), patch 16 | ✅ exact |
| corpus parity | `physicalai-train-e438721ae894-w120-256x640cyl`, skip-hash `f09e44db`, `checked: true` (trainer-verified sha) | ✅ |
| past visual context | window = **8 frames = 0.8 s @ 10 Hz**, n_windows 8 | ✅ as designed |
| past measurements | `action_dim: 3` = steer (atan(wb·curvature)) + **measured** longitudinal accel (dataset `ax`, not finite-diff) + speed channel; ego conditioning with `ego_dropout 0.5` and a learned **ego_null_row** (P5b) | ✅ |
| goal inputs | `cond_vtarget: true` (tactical set-speed) · `cond_route: true` (strategic class) · `goal_dropout: 0.5` | ✅ wired + anti-shortcut |
| imagination | `cond_imagination: true`, reads at steps **[5,10,15,20] = 0.5–2.0 s**, 256 anchors | ✅ |

**Capability limits (design decisions, not wiring faults):** single front 120° view (no
rear/side cameras — cut-ins from behind are structurally invisible, unlike Alpamayo's 6-cam
ring); visual past of 0.8 s (Alpamayo: 4 frames but a 16-step ego-motion history). Both are
v6-class levers, recorded here so they are chosen, not discovered.

**⚠️ Review corrections this audit forces (already applied to `V5F_ARCHITECTURE_REVIEW.md`):**
(1) §3: THIS RUN's horizons are dense 1..20 (2 s) with imagination reads at 0.5–2 s — the
"5 s tactical knots" applied to the v4-code default, not this run's config. (2) §0: the
canary/val divergence flag is **resolved benign** — the run's own config documents that
from-scratch training makes the canary controller **inert by construction** (its baseline
is the untrained step-0 canary), so canary drift carries no signal here.

## R2 — "predictor actioned by the planner's output": ⛔ NOT TRUE of v5f as running — and must not be patched mid-run

The predictor trains teacher-forced on GT actions; emitted candidates never re-enter the
roll. This is the measured §1.12 exposure class, and the run is at 95 % — changing it now
would invalidate 28k steps of attribution. The committed path (unchanged): post-30k
controllability probes → stage-A self-supervised post-training (`L_ctrl`/`L_scene`, the
only sound off-policy piece) → T1 closed-loop eval → Diffusion-MPC L1 where the planner's
own actions drive per-candidate rolls. Full on-policy training needs the T2 simulator
(counterfactual-supervision wall, redesign §2.1).

## R3/R4 — hierarchical wiring and layer data diets: v5f carries the SIGNALS, v1.8 builds the LAYERS

What v5f verifiably wires today: the tactical signal (vtarget set-speed) and strategic
signal (route class) as gated generation conditioning — the proto-hierarchy. The full
layers (own predictors, goal fans, selectors per level) are the v1.8 build (E4/E7), and
their data diets are specified and SOURCED:

| layer | receives (spec) | data source status |
|---|---|---|
| tactical (1 Hz) | pooled z_op window + hindsight goal labels (pose@t+2..6 s) + 3-axis severity manoeuvres + Alpamayo meta-action teacher | poses ✅ in corpus · labeler = E4.1 · teacher dataset **accumulating now** (~2k clips) |
| strategic (0.2 Hz) | pooled z_tac + 25 s corridor labels (`nav_command`) + road-class strata | corridor labeler ✅ exists · road classes ✅ MEASURED (1,884 urban / 1,241 intersection / 384 highway / 83 unstructured) |

Assurance statement: nothing in the v1.8 layer designs requires data the programme does
not already possess or is not already producing; the single external dependency (T2 sim)
gates only the on-policy stage, not the hierarchy.
