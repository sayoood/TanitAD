# Ranked reference systems — the external work TanitAD should measure itself against

**Requested by the PI 2026-07-29:** *"a ranked list of the best approaches from AD and AI, robotics
research which we could use as reference."* This is the piece the two deep-research reports did not
deliver — they gave findings and transfers, not a reference frame.

Source: `…/incoming/2026-07-29-deep-research-sota/` (215 agents, 12.9 M tokens, 3-vote adversarial
verification). Ranked by **usefulness to us as a reference**, not by the paper's own prestige.

## ⛔ Read this first — what "refuted" means here

**A claim refuted 0-3 does NOT make the system a bad reference.** Verification removed our right to
quote that paper's *numbers* as decision inputs. It did not show the system is uninteresting, and in
several cases the verifiers simply ran out of search budget. Three tiers are used below:

| tier | meaning | may we quote its numbers? |
|---|---|---|
| **A — VERIFIED** | survived 3-vote adversarial verification against the primary | **YES**, with the caveats attached |
| **B — REFERENCE ONLY** | the system is a legitimate reference; its specific claims were refuted or unverified | ⛔ **NO** — design experiments from it, never decisions |
| **C — INSTRUMENT** | a benchmark/harness we should run against, not a model to copy | its protocol, yes; our score, only once measured |

⚠️ Under CLAUDE.md rule 1, quoting a tier-B number to decide a GPU-day would be an **INHERITED claim
deciding a GPU-day** — forbidden. Tier B exists so we stop re-discovering these systems, not so we
can cite them.

---

# PART 1 — AUTONOMOUS DRIVING

### R1 · **NAVSIM v2 / EPDMS** — the scoring surface we should adopt · TIER A (instrument)
`github.com/autonomousvision/navsim`

The single most useful AD reference we have, because it names the structural property our scoring
surface **lacks**: a **multiplicative safety gate** that a good average cannot buy back.

```
EPDMS = (∏ NC, DAC, DDC, TLC) × (Σ w·filter_m / Σ w)     w: EP 5, TTC 5, LK 2, HC 2, EC 2
filter_m(agent,human) = 1.0 if m(human)==0 else m(agent)
```

⭐ **What to take:** the gate structure **and** the `filter_m` human-reference semantics. Implementing
from the plain formula yields PDMS-v1 and **over-penalises arms** — this correction is the reason to
read the devkit rather than a summary. Our `taniteval/registry.py:272` already carries PDM-Closed
EPDMS 51.3 as an external baseline.
⚠️ Full EPDMS needs two-stage pseudo-closed-loop, reactive traffic, traffic-light state and a
**human-reference rollout** ⇒ a lane graph is **necessary, not sufficient**. LK is disabled at
intersections because centreline annotations don't match real markings.

### R2 · **Bench2Drive** — closed-loop harness, and honest about its own limits · TIER C
`github.com/Thinklab-SJTU/Bench2Drive` (NeurIPS 2024 D&B)

Reference for *how a benchmark should behave*: its maintainers publish against-interest testimony
telling the field to stop reporting nuScenes open-loop planning. Use it as the closed-loop target if
we ever run CARLA.
⚠️ Its own known pathology (tier B): reusing training scenarios at test time, so a high score can
reflect memorisation. The claim that this is fatal was refuted **1-2** — treat as a caution, not a
finding.

### R3 · **PlanT 2.0** — the shortcut-probe methodology · TIER A (method) / TIER B (numbers)
Gerstenecker, Geiger, Renz — Tübingen; code, data, checkpoints released.

⭐ **Take the METHOD, not the model.** Its §5.4 "Positional shortcuts" probe — sweep ego rotation,
watch the speed output for a **step** — is the cheapest diagnostic in either report, and **we can run
a strictly cleaner version than they can** (our `pseudosim` `dyaw` is an exact homography on real
footage; theirs moves world pose and object input together).
⚠️ Privileged planner (GT perception), CARLA-specific root cause. Its Town13 data-composition
ablation was refuted **0-3** — do not quote those numbers.

### R4 · **Fail2Drive** — the generalisation-failure taxonomy · TIER B
Same lab, 2026-04-09; open toolbox released.

Reference for *what to test for*: policies that ignore obstacles present in their own sensor returns,
and collapse when a texture cue is removed. Motivates our free/occupied linear probe.
⚠️ Per-scenario n unpublished (~5), no CIs, and "0.00 HM" is partly a harmonic-mean floor artifact.
**Direction, not effect size.**

### R5 · **Hydra-MDP / anchored trajectory vocabularies** — TIER B, and our top re-verification target
⭐ **The single most valuable follow-up.** The **factorised path × velocity vocabulary** (paths and
speed profiles enumerated separately rather than whole trajectories) maps **one-to-one** onto our
diagnosed *"5-way maneuver softmax mixes LAT+LON"* mechanism and our **88.7 % longitudinal** oracle
gap. Both the anchor-scaling result (1024→16384, EPDMS 85.02→87.35) and the factorisation went
**0-3**. ⛔ Not admissible now — **re-verify this before anything else in a future survey.**

### R6 · **Waymo long-tail E2E benchmark (RFS)** — TIER B
Human-preference scoring instead of displacement. Two refuted-but-important leads: ADE and RFS
diverge across 19 submissions; and a **36 M** trajectory model reportedly beat a **3 B** MLLM. Both
**0-3** — if true, the second directly supports our sub-300 M thesis, which is exactly why we may
not lean on it unverified.

### R7 · **DriveVLN / map-free navigation** — TIER B
The only reference touching our hardest constraint: **no map, no lane graph, no GNSS**. Nothing
survived verification, so this is a reading list for the strategic-brain problem, not guidance.

---

# PART 2 — WORLD MODELS, JEPA, ROBOTICS

### R8 · **SkyJEPA** — the drift instrument · TIER A ⭐ HIGHEST-VALUE REFERENCE IN EITHER REPORT
arXiv 2606.23444 (NYU ARPL + Brown; Rao, Zhang, Balestriero, LeCun, Loianno)

```
CR_k = e_k,rollout / e_k,teacher-forced      ER_k = E[e_k − e_{k−1}]
```

⭐ This is the reference that **caught an error in our own headline** (C61). Any latent-rollout claim
we make from here on should carry `CR_k`. They train rollout on T=20 (1.0 s) yet measure to k=60
(**3× the trained horizon**) — a protocol worth copying directly.
⚠️ Quadrotor state, low-dimensional, physics-structured prober. **The metric design transfers; the
1.4/2.4 magnitudes do not.** Their CR is still *rising* at k=60.

### R9 · **V-JEPA 2 / V-JEPA 2-AC** — the closest published analogue to our design · TIER A
arXiv 2506.09985 (Meta FAIR; MIT-licensed weights, `vjepa2_ac_vit_giant`)

The strongest external support for **latent-space prediction over pixel reconstruction**, and for
action-conditioned heads being **data-cheap**: <62 h of DROID video, no reward, zero-shot on Franka
arms in two unseen labs (80/80/80/50 %, n=10 each).
⭐ Also the honest reference on horizon: **planning horizon = 1 (~0.25 s at 4 fps)**, receding-horizon,
*because* of error accumulation — stated against their own interest.
⚠️ Human-provided sub-goals; **16 seconds per action** ⇒ no evidence for real-time control.
⛔ Its frozen ViT-g must **not** be used to argue against our from-scratch encoder — that promotion
was refuted **1-2**.

### R10 · **HorizonDrive** — the long-horizon mechanism · TIER A (mechanism) / TIER B (headline)
arXiv 2605.11596 (Horizon Robotics)

The only driving-specific quantitative long horizon: **~20 s**, 10× our cap, with a bounded rolling
context whose per-step cost is flat in horizon.
⭐ **Take the mechanism — scheduled rollout recovery (train on your own prediction-corrupted
histories) — not the number.** It rolls out in a **pixel-reconstructive VAE latent** scored with
video metrics; not like-for-like with our decoder-free predictive latent.
⚠️ Its "minute-scale" headline is **qualitative only**, appendix, no metric — the authors say so.

### R11 · **MoP-JEPA** — the executor-masking warning · TIER B, but the warning is load-bearing
arXiv 2607.05238

⭐ Authorial concession worth internalising: *"With 10 % false edges, replanning raises success from
0.40 to 1.00."* **An online-replanning executor masks transition-model error.** This is now our
reporting protocol: open-loop fidelity and closed-loop success are separate decision inputs.
Also a reference for extending planning *distance* by **graph search over real dataset states**
rather than deeper imagination — relevant if the strategic brain can't get topology.

### R12 · **Koopman Dreamer** — spectral constraint on the latent transition · TIER B, LOW confidence
arXiv 2607.19719 (7 days old at survey time)

Interesting mechanism (bounded spectral radius on the transition), **wrong domain** — proprioceptive
state only, **no pixel/ViT observations anywhere**, seed count unstated, and the headline −89.4 %
latent MSE is plausibly a units artifact (relay only −23.2 % observation-space).
⛔ Its stated trade-off — lowering the radius **erases persistent task information** — directly
threatens the speed/scale magnitude our speed fix bought (R² 0.965). **Sequence after E-CR, never before.**

### R13 · **Sub-JEPA** — collapse control via low-dimensional subspaces · TIER B (has public code)
arXiv 2605.09241. Impose the isotropic prior inside **random low-dimensional subspaces** rather than
the full ambient embedding — a cheap swap against our existing `SigRegConfig` (`n_slices=512`,
weight 0.1). Refuted 1-2; **experiment design only.**

### R14 · **Delta-JEPA** — decoder input form over capacity · TIER B
arXiv 2606.31232. An inverse-dynamics decoder should take the latent **displacement** `z_{t+1}−z_t`
rather than concatenated endpoints (+12.60 on Push-T). A **one-line** change to our IDM, and it
asserts that **input form, not capacity, is the lever** — which speaks directly to our
smaller-is-better IDM ladder (0.86 M > 2.90 M > 19.98 M). Refuted 1-2.

### R15 · **DINO-WM / LeWM (FF-JEPA)** — the frozen-vs-trained axis · TIER B
arXiv 2606.09311. An 18 M task-trained encoder reportedly beating frozen DINOv2 at long horizon
(91.80 % vs 61.0 %) — this would **corroborate our measured ceiling**, and it was refuted **0-3**.
⭐ **Consequence to internalise: our frozen-encoder position rests ENTIRELY on MEASURED-ours.** All
seven candidate claims failed in **both** directions. Absence of a verified contradiction is not
corroboration.

---

## The gap nobody filled — and it is a real finding

**Four of six requested areas in Report 2 and five of eight in Report 1 produced ZERO admissible
evidence.** No verified reference exists in this survey for:

- **VLAs / world-action models** — π-0 / π-0.5, GR00T N1.x, RT-2 successors, OpenVLA, LAPO,
  action chunking, flow-matching vs autoregressive heads
- **Robotics transfer** — cross-embodiment, Open-X/DROID scale, sim-to-real, failure recovery
- **LLM reasoning** — test-time compute, distillation, structured latent reasoning for a strategic planner
- **Small-model scaling laws** — which is exactly where our anchor-count knee and IDM ladder live
- **Genie 3, DreamerV4, Cosmos** — appear in **no surviving claim**

⇒ These are **not** areas where the field is silent; they are areas where **this survey failed**.
They are the brief for the next one, and the ranked re-verification order is:
**(1) factorised lat/lon vocabulary [R5] · (2) frozen-video vs from-scratch encoders [R9/R15] ·
(3) small-model scaling laws · (4) VLA action heads.**

## The one experiment nobody has run

⭐ **Frozen VIDEO-pretrained encoder vs from-scratch, on a metric driving task.** V-JEPA 2's frozen
encoder was **video/motion-pretrained on 1 M+ hours**; DINOv2 and I-JEPA — the two we measured our
ceiling against — are **image**-pretrained. Nobody has run the video-pretrained arm on driving.
Cheapest form: freeze a V-JEPA 2 ViT-L, train only our predictor head, same corpus and steps.
This is the highest-value open question in the reference set, and it is *ours to answer*.
