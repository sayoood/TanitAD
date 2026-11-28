# Cheap proof: imagination-in-the-loop vs single-shot open-loop (closed-loop drift)

**Date:** 2026-07-22 · **Evidence class:** MEASURED (`incoming/2026-07-22-imagination-closedloop-proof/closedloop_flagship-30k_imagination-proof.json`) · **Model:** flagship v1 (`flagship4b-speedjerk-30k`, ckpt step 29999, HF `Sayood/tanitad-flagship-4b-speedjerk`) · **Compute:** NVIDIA GeForce RTX 4060 (local, 8-batch), wall 32.4s.

**n = 265 windows / 12 held-out val episodes** (physicalai-val-0c5f7dac3b11 ep_00000..ep_00011 — a CHEAP 12-ep subset, not the full 40). No renderer: the flagship world model is its own neural simulator (`taniteval/taniteval/closedloop.py`).

## Thesis under test

The 2026-07-22 closed-loop synthesis (`Architecture & Inference/Research/2026-07-22-closed-loop-robustness-and-imagination.md`, section 3/5) argues that a planner that rolls candidate actions through a world model and re-plans on the imagined consequence should accumulate LESS closed-loop error than an open-loop planner that commits to a single-shot trajectory (the REF-C ablation that fails ~half the AlpaSim scenes). This is the cheap, no-renderer, on-pod analogue of the AlpaSim REF-C-vs-imagination comparison, runnable NOW.

## Method — the two arms (paired, identical at tick 0)

Both arms use the SAME trained strategic->tactical head, the SAME pure-pursuit + kinematic-bicycle controller, and the SAME held-out windows; they are byte-identical at tick 0. The ONLY difference is what happens after:

- **(A) single-shot open-loop** (`open_plan_bike`, the arm ADDED this run): plan the 2 s tactical trajectory ONCE on the REAL encoded window, then TRACK that FROZEN plan open-loop — the world-model predictor is NEVER stepped (REF-C-style single-shot).

- **(B) imagination-in-the-loop** (`closed_bike`, the harness headline closed loop): re-plan every 0.1 s on the model's OWN imagined latent (operative predictor rolls the latent forward under the executed action each tick).

So `closed_bike - open_plan_bike` isolates exactly imagination-in-the-loop. Decision-grade interval: the **paired episode-cluster bootstrap** over the val episodes (`taniteval/ci.py`), not the deprecated overlapping-holdout.

## Pre-registration (locked before the measured run; both outcomes committed)

- **(B) materially < (A), CI-separated** -> early SUPPORT for the imagination thesis (and v4's value).

- **(B) ties (A)** (CI not separated) -> imagination-as-implemented does NOT buy closed-loop robustness at this fidelity -> flag for closed-loop synthesis section 6 (closed-loop-aware training).

- **(B) materially > (A), CI-separated** -> imagination over THIS world model DEGRADES closed-loop -> the world model is unfaithful under self-rollout (synthesis section 4: imagination needs a healthy WM); v4.1 lower-lr_trunk becomes the prerequisite.

## Result (MEASURED)

| arm | closed-loop ADE@2s (paired episode-cluster bootstrap 95%) | divergence >5 m @2s |
|---|---|---|
| (A) single-shot open-loop `open_plan_bike` | 1.933 m [1.632, 2.275] | 39.2% |
| (B) imagination-in-the-loop `closed_bike` | 1.720 m [1.444, 2.040] | 22.3% |
| _reference: raw single-shot plan, no executor_ | 3.147 m [2.242, 4.170] | — |

**Paired delta (B − A) ADE@2s = -0.213 m [-0.341, -0.053], separated = True, P(B>A) = 0.0045.** FDE@2s delta (B − A) = -0.418 m [-0.857, 0.112], separated = False.

Per-horizon closed_bike ADE (deprecated overlapping-holdout CI, context only): 0.5s=0.240, 1s=0.614, 1.5s=1.127, 2s=1.732 m. open_grnd (TRUE-action WM-fidelity reference) ADE@2s = 0.318 m; CV baseline = 0.955 m.

## Verdict: IMAGINATION_HELPS

B (imagination-in-the-loop) drifts LESS than (A) by 0.213 m at 2 s (CI-separated). Early SUPPORT for the imagination thesis: even over the current flagship world model, re-planning on imagined consequences reduces closed-loop drift vs a single-shot open-loop plan.

## Honest caveats

- **This is the WEAK form of the v4 thesis.** Arm (B) re-plans on the imagined latent but does NOT sample+select candidates by imagined consequence (no CEM/imagine-and-select). This is the cheap no-renderer proving ground, not the AlpaSim imagine-and-select test (synthesis section 5) — that remains the decision-grade closed-loop test.

- **Self-referential.** The world model is BOTH the planner's state estimator and the simulator; failures it cannot imagine are invisible. A drift/stability closed loop, NOT a collision/PDM safety loop (no HD map / agent boxes in our data).

- **Cheap subset.** 12 val episodes / 265 windows on the local 4060, not the full 40-episode suite — early evidence, wider CIs than a full run. The arms are paired on identical windows, so the DELTA is the robust part.

- The pure-pursuit + bicycle is a HARNESS controller, shared by both arms, so its fidelity floor cancels in the paired delta.

## Deliverable manifest

| artifact | location |
|---|---|
| measured result JSON | `repo: .../incoming/2026-07-22-imagination-closedloop-proof/closedloop_flagship-30k_imagination-proof.json` |
| this note | `repo: .../incoming/2026-07-22-imagination-closedloop-proof/README.md` |
| provenance | `repo: .../incoming/2026-07-22-imagination-closedloop-proof/provenance.json` |
| arm (A) added to harness | `repo: taniteval/taniteval/closedloop.py` (open_loop_plan_rollout, densify_plan, imagination_comparison block) |
| local driver | `repo: .../incoming/2026-07-22-imagination-closedloop-proof/run_proof_local.py` |

_All staged (git add), NOT committed, NOT pushed (Agent Operating Standard)._
