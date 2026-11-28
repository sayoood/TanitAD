# Training WM-driven driving policies in closed loop — RL, or something cheaper? The verdict

**2026-07-23, orchestrator synthesis** of a 3-angle cited literature sweep (staged raw in
`Benchmarks & Eval/Implementation/incoming/2026-07-23-closed-loop-wm-research/angle{1,2,3}_*.md`,
~40 sources) + two in-house MEASURED results. Sayed's question: *how do we particularly train
WM-driven driving models in closed loop — combine with RL, or other techniques?*

Evidence classes: **PUBLISHED** (arXiv id) for external facts · **MEASURED** (ours, artifact path) ·
**HYPOTHESIS** for our inferences. Every external magnitude below is single-to-few-study — directional.

---

## TL;DR verdict

For **our** regime — logged data (comma2k19 + PhysicalAI-AV), **no fast simulator**, safety-critical,
a **differentiable** world model + anchored-diffusion planner that already has a MEASURED closed-loop
win from imagination re-planning — **reward-driven RL is the LAST lever, not the first.** Build the
**free inference-time floor** and run one near-zero-cost experiment that can close the question outright.
When a *learned* closed-loop lever is warranted, the highest-ceiling one is **analytic-gradient
closed-loop imitation through our differentiable WM + a BC anchor + a pessimism guard** — which is the
same mathematical object as "Dreamer-style RL" for continuous control, and has a direct logged-driving
precedent. The single crux across every learned lever is **world-model exploitation**, bounded by WM
fidelity — i.e. the v4.2 line and this line are the same bet.

## The problem, precisely

Open-loop imitation optimizes single-shot prediction vs GT; the planner never sees its own errors
compound → covariate shift → **off-road departure under its own drift** — our MEASURED failure, and
(MEASURED, scenario suite) **concentrated off-highway** (flagship ties REF-C on highway, goes off-road
6/8 on non-highway). The field agrees this is the core issue and that **open-loop ADE is misaligned with
closed-loop driving** (nuPlan "misconceptions", CoRL'23; Waymo "Imitation Is Not Enough", arXiv:2212.11419).

## The ladder — cheapest & safest first (build 1–3 regardless)

| # | lever | cost | reward? | precedent / note |
|---|---|---|---|---|
| 1 | **Safety filter / shield** — road-boundary CBF / RSS / reachability over the WM | ~free | no | production pattern (Mobileye RSS, NVIDIA SFF); bounds catastrophe, can't game the WM |
| 2 | **Cost-guided diffusion at inference** — drivable-area/collision energy in the denoiser | ~free | no | Diffusion-Planner (ICLR'25, 2501.15564); **drop-in to our anchored-diffusion head; the energy IS our off-road failure** |
| 3 | **WM-planning / MPC** — upgrade the −0.213 re-planner to MPPI/CEM + a learned terminal value | low | no | TD-MPC2; value decouples short (error-safe) horizon from long-horizon consequence — our exact ADE⊥closed-loop pathology. Add **test-time-only first**, no retraining |
| 4 | **Closed-loop-aware IL** — RoaD / CAT-K / model-based DAgger | med | no | CAT-K (CVPR'25, 2412.05334): 7M CL-tuned > 102M open-loop. **RoaD (2512.01993) tunes our exact AlpaSim+VLM stack → +41% score / −54% collisions**, and "one-off rollouts ≈ continuous" eases the no-fast-sim objection |
| 5 | **Offline RL** — CQL / IQL | med | yes | logs-native; MEASURED head-to-head (arXiv:2508.07029): **CQL 3.2× success / 7.4× fewer collisions vs BC** — but OOD-brittle, needs conservatism |
| 6 | **Analytic-gradient closed-loop imitation through the diff-WM** (= "Dreamer-RL" for continuous control) | 10s of GPU-days | shaped cost | **Nachkov (2409.07965): off-road 0.028 vs BC 0.27 on 500k logged Waymo** — fixes our exact failure. But they differentiate an *exact* sim; ours is *learned* → exploitable |

**Rows 1–3 are inference-time, add no training, and cannot learn to game the WM — build them regardless.**
They attack the off-highway road-keeping weakness directly and ship in the deployed planner.

## The insight that reorganizes the "RL" question

For a **continuous** trajectory, **DreamerV3's actor gradient is ρ=0 — pure dynamics-backpropagation
(analytic backprop through the differentiable model)**; REINFORCE (ρ=1) is only its discrete-Atari
setting (arXiv:2301.04104). So **"imagined-rollout RL" and "analytic gradients through our WM" are the
same object** in our regime — the scary "RL rung" is really **closed-loop-aware training through a
differentiable model**, the family that already works on logged driving (Nachkov; Urban Driver; Waymo
MGAIL, arXiv:2210.09539). The recurring *deployed* recipe is **closed-loop-through-a-model + an open-loop
BC anchor.** We can borrow DreamerV3's numerical stack (symlog, two-hot value, percentile-return
normalization, KL-balance) for stability **with no online loop and no environment reward**.

## Where reward-driven RL genuinely wins (and why not yet for us)

- The **sparse safety-critical tail**: Waymo "Imitation Is Not Enough" — **>38% fewer failures** on the
  hardest slice with an added RL objective.
- **Large-scale planning IF you can roll out**: CaRL (300–500M sim samples, 8×A100·week); Raw2Drive
  (40–64 H800-days). Both need an interactive sim we don't have — and even best-scaled RL only **ties**
  rule-based PDM-Closed on nuPlan. **Nobody has shown RL is *necessary*.**
- **WM-as-RL-environment has essentially no driving track record**: GAIA-1/2, Vista, DriveDreamer,
  DriveArena use the WM for data-gen / scoring / *evaluation*, never as an RL training env. The one
  exception, **Think2Drive** (DreamerV3-in-a-WM, 100% CARLA-v2, ECCV'24), runs on **privileged BEV state
  in live CARLA** — not pixels, not logs.

## The crux, everywhere: world-model exploitation

Every *learned* right-half lever optimizes the planner against a **learned** WM, which it will exploit
precisely **off-highway where our data is thin and the WM is ~3× OOD** — the exact place we fail, and
invisible to open-loop ADE. **MEASURED corroboration (ours):** the DAgger proof on the no-renderer
kinematic harness came back **DAGGER_HURTS** (closed-loop ADE +0.266 [0.008,0.550], off-road proxy +0.548)
— because the "on-policy states" were the WM's own imagined off-manifold latents and the head learned to
over-correct (`…/incoming/2026-07-23-dagger-closedloop-aware/`). The lesson: **closed-loop training needs
FAITHFUL rollouts (AlpaSim), not free imagination**, and the mitigation bundle is non-optional:
ensemble-disagreement **pessimism** (MOPO/MOReL/COMBO), **short truncated horizons grown with measured WM
fidelity**, a **BC/log anchor + curriculum resets**, and **gate on closed-loop drivability, never
open-loop ADE.** This is why v4.2's healthy WM is the prerequisite for the entire right half of the ladder.

## Pre-registered experiments (both/all outcomes committed in advance)

**Gate 0 — near-zero cost, can close the question with no training.** Add training-free
drivable-area/collision **guidance + a road-boundary safety clamp** to the *existing* anchored-diffusion
planner; measure **off-road departure on the 40 val episodes** (paired episode-cluster bootstrap,
`taniteval/ci.py`). *If off-road collapses to ~0 → RL is unnecessary, decided for free.* If a residual
survives (esp. off-highway) → Gate 1.

**Gate 1 — only on a surviving residual; still no live sim.** Fine-tune the same planner two ways over
**identical imagined WM rollouts**, differing **only in a reward/objective flag**: (Arm A) closed-loop-aware
IL (RoaD/CAT-K, imitation-divergence) vs (Arm B) analytic-gradient through the diff-WM with a shaped
drivability cost + pessimism. Paired **off-highway off-road** metric. *RL/analytic wins only if it beats
IL-CL on the tail beyond the CI.* **A gain that appears in imagination but not on held-out episodes is the
WM-gaming signature — caught before it costs GPU-weeks.** (Angle-1's Arm A "value+MPPI test-time only" is a
free precursor to fold into Gate 0/rung 3.)

**⚠️ Rollout-fidelity caveat:** RoaD reports sim2sim degradation 0.75→0.58; with NuRec's ~3× OOD, the
"rollouts ≈ good demonstrations" assumption is the thing most likely to break off-highway. The
inference-time floor (rows 1–3) is safest precisely because it can't learn to game the model.

## Recommendation

1. **Build the free inference-time floor (rungs 1–3)** — it ships in the deployed anchored-diffusion
   planner (which v4.2 uses), targets the measured off-highway failure, and adds a permanent safety
   override. **Run Gate 0 first** (~hours) — it may close the RL question outright.
2. **Let v4.2 land** — a faithful WM is the prerequisite for any learned right-half lever, and its healthy
   canary is the exploitation guard.
3. **The AlpaSim scenario scale-up (running) is the faithful-rollout source** the IL rung (#4) needs; the
   self-referential harness cannot substitute for it (MEASURED, DAgger).
4. **Only then, if an off-highway residual survives Gate 0**, run Gate 1 — and note the "RL" arm is
   analytic closed-loop imitation through the diff-WM + BC anchor + pessimism, not PPO-in-a-sim.

**One-line answer to Sayed:** don't reach for RL yet — build the free guidance+safety floor and run Gate 0;
if a residual survives, the right "RL" is analytic closed-loop imitation *through our own world model*,
gated on drivability and guarded against WM-exploitation — and it stands or falls on WM fidelity (v4.2).

---

## MEASURED UPDATE (2026-07-23, Gate 0 + 0b) — the free floor is RULED OUT for junction off-road; Gate-1 is warranted

We ran the free inference-time floor (rungs 1–2) on REF-C's anchored-diffusion planner over the balanced
38-scene AlpaSim suite, WITH vs WITHOUT, **zero training**, cost geometry validated first (a `world_to_nre`
frame bug was caught + fixed — the null is trustworthy). Both forms FAIL to fix junction off-road:
- **Gate 0 (cost-guided SELECTION):** ΔOFFROAD **+0.00 [−0.08, +0.08]** overall, junctions unchanged.
  Cause = the **anchor-vocabulary ceiling** — selection can only pick among 128 straight-biased comma2k19
  anchors; **~21% of off-road junction moments have NO on-road anchor** to pick.
- **Gate 0b (per-denoise GRADIENT NUDGE = cost-guided diffusion proper):** synthesizes trajectories
  OUTSIDE the anchor set (validated: escapes the ceiling) — yet junction off-road is still **0.73 → 0.73 [0,0]**.

⭐ **The decisive mechanism:** with the gradient nudge, **0 planned trajectories are off-road** (vs 96 under
selection) — *the plan is on-road at every junction* — **yet the ego still departs the road.** So the junction
failure is **not plan-representability, it is the EXECUTED path** (controller tracking + covariate drift over a
20 s junction crossing). Both inference-time levers are ruled out; only closed-loop-aware **training** shapes the
executed trajectory. **⇒ Gate-1 (rung 4: RoaD/CAT-K closed-loop IL, or analytic-gradient through the diff-WM) is
warranted for junction off-road — now MEASURED-justified, not merely argued.**

**Also banked:** the gradient-nudge floor is a strictly-better FREE safety override — **deploy it** (intersection
at-fault collisions 0.71→0.43, plan-deviation 0.34 vs selection's 0.76, no roundabout regression). Raw:
`Benchmarks & Eval/Implementation/incoming/2026-07-23-gate0-freefloor/` (Gate 0 + 0b NOTEs, results,
cost/gradient validation JSONs).

**Rung 3 (WM-planning / MPC) — TIE, MEASURED** (`…/incoming/2026-07-23-freefloor-rung3-wm-mpc/`). MPPI/CEM
planning over the WM does NOT beat the single-step re-plan: Δ **+0.005 to +0.011** across configs, **none
CI-separated**; the only separated result is *worse*, and lateral off-road is separated-worse (**+0.136**,
from stochastic-selection jitter). The single-step re-plan already captures the imagination benefit;
sampling-based MPC adds compute + slightly worse road-keeping, no gain. Compute is **feasible** (batch-1 tick
~50 ms at K8, under the 100 ms budget) — feasibility isn't the blocker, it just doesn't help. Deferred (not
refuted) to an AlpaSim non-self-referential test.

**⇒ The free inference floor is now FULLY characterized: none of the three rungs (guidance-selection,
gradient-synthesis, WM-MPC) fixes junction off-road.** Three independent lines — the literature verdict,
Gate 0/0b, and rung 3 — converge on the same answer: **inference-time levers are exhausted for junction
off-road; Gate-1 (closed-loop-aware training) is the lever.** Ship rungs 1–2 as a free safety override
(collisions down); the junction fix is Gate-1.

**Gate-1 rollout prep (2026-07-23, MEASURED, `…/incoming/2026-07-22-alpasim-closedloop-evalpod/GATE1_ROLLOUTS_NOTE.md`)
REFINES the mechanism — and supersedes the "execution failure" framing above.** On-policy junction rollouts
(15 scenes, 675 steps) decompose the divergence: the ego **tracks its own plan tightly (0.49 m)**, but the
**plan itself degrades on-policy** (plan cross-track 0.57 m → 12.98 m as the ego reaches off-distribution
states). So the junction failure is **on-policy PLANNER covariate-shift, faithfully executed** — the *planner*,
not tracking. The conclusion is unchanged and strengthened: **Gate-1 = retrain the planner on its own on-policy
states with GT-recovery targets (now collected: GT path 0.5–2 s ahead per visited state = drop-in DAgger/CAT-K
labels).** ⚠️ Real caveat now measured: the on-policy states are NuRec reconstructions (~3.2× OOD), so a fine-tune
on them partly targets *reconstruction*-OOD — **sufficient to prototype/de-risk, not to train a robust policy**
until a lower-OOD (real-footage) closed-loop source exists.
