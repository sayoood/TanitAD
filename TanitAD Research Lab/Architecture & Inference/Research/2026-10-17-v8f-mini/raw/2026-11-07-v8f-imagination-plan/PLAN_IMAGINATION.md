# Validating and improving what the v8F world model imagines: the plan (Master Mind, 2026-11-07)

Inputs: `GAPS.md` (G1–G9, MEASURED on M2b s0 / s1, tier T0), `A/REPORT_A.md` (12 validation families across JEPA application areas, 26 papers banked), `B/REPORT_B.md` (driving and hierarchical world models, 25 papers banked). Every row below that names a number cites one of those three files. ⛔ T0 numbers are world-model diagnostics, never driving performance. Only E9 (the reliance arm) can produce a driving claim, at T1-kin and T2.

## 1. What "the imagination is validated" means for a latent, unsupervised model

The latent of a JEPA has no pixels to compare against, so no single test proves a prediction right. The literature (REPORT_A, short answer) combines four kinds of evidence. The hierarchy adds two more:

| # | question | what answers it | the control that makes it mean something |
|---|---|---|---|
| Q1 | Is the latent healthy? | effective rank, retrieval against exact chance | Graph-JEPA 2608.20516: probe 0.871 and rank 18–47 on a latent that recovers 0.00 of 14.4 bits. A gate, never a result. |
| Q2 | What does each level carry? | probes on ENCODED latents, in NMSE units (1 = the marginal mean) | random-init readers, control tasks (selectivity), our permutation reference |
| Q3 | Is the prediction right? | the same probe, fitted on encoded latents only, applied to PREDICTED latents: excess NMSE per horizon | Copy-Last floor (the present latent used as the prediction) and Oracle ceiling (the encoded true future) |
| Q4 | Does it answer the action, with the right sign and size? | matched-intervention audit (gain, direction cosine, sign), inverse-probe 2×2, action-bank retrieval | zero-effect, persistence and oracle baselines; a closed-form kinematic null; capture first: propagation is readable only where Q2 passes (Intervention Gap 2608.29998) |
| Q5 | Do the levels differ the way a hierarchy should? | slow/fast split across levels; L2 one step vs L1 two steps; true-subgoal control; event alignment of upper-level surprise | Clockwork VAE 2102.09532, H-JEPA 2610.06805, Mind the Gap 2607.12547, THICK |
| Q6 | Does the planner gain from it? | the reliance arm at T1-kin and T2 | the same planner without the world model, paired, two seeds |

⭐ Decoders (pictures) answer none of Q1–Q6. They show what a level holds. Every decoder below is trained on ENCODED latents only (V-JEPA 2 B.3), and every picture carries a shuffled-condition decoder beside it.

## 2. The instrument: the Imagination Audit (IA), one battery for every level, horizon and seed

Run per level (L1 / L2 / L3 at 0.2 / 0.4 / 0.8 s steps), per horizon (1, 2, 3 steps of that level, so up to 2.4 s at L3), per model seed (s0, s1), episode-cluster bootstrap B 2,000 plus the permutation reference (the two-part rule from LATSEM2B).

| module | method (source) | targets | what it reports |
|---|---|---|---|
| IA-0 health | RankMe / effective rank on encoded and predicted latents (cap 128); retrieval rank of the true future among same-clip windows vs exact chance | — | a pass / fail gate per level |
| IA-1 capture | 2-layer MLP and ridge probes, 3 probe seeds, NMSE (H-JEPA template) | the four families: LON speed, accel, lead gap, TTC, closing speed; LAT yaw rate, curvature; TAC 7-class manoeuvre, lead present; SCENE agents within 30 m, cuboid occupancy | NMSE per target per level; random-init row |
| IA-2 probe transfer | SG-JEPA excess NMSE; UWM-JEPA ΔR²; DINO-Foresight Oracle / Copy-Last | the same targets at the future step | IMAGINES / COPIES / UNREADABLE per row (bar below) |
| IA-3 action audit | Intervention Gap capture-gated propagation; ATM inverse-probe 2×2; IWM action-bank MRR; iKCE kinematic null | speed, yaw rate, lead gap under logged action contrasts | gain, direction cosine, sign; MRR vs exact chance |
| IA-4 surprise | violation of expectation (Garrido 2502.11831; LeWM; 2606.28383) | log-built impossibilities (below) | pairwise accuracy per level with CI; event-tag AP |
| IA-5 decoders (look only) | cuboid BEV occupancy decoder per level (OccWorld, MILE); nearest-frame retrieval | 10-class agent occupancy, ego frame | per-horizon IoU vs Copy&Paste and Oracle; pictures |
| IA-6 hierarchy | timescale index; L2 vs L1×2 slope; true-subgoal planning; upper-level surprise vs events | — | does each level do a different job |

**Bars (to be frozen per card, not here):**
* A row is **UNREADABLE** when the probe on the encoded TRUE future is not separated from NMSE 1. No imagination claim is possible on it; the lever is capture, not prediction.
* A row **IMAGINES** when the predicted-latent NMSE is separated below the Copy-Last NMSE (two-part rule, both seeds).
* A row **COPIES** when it is readable but not separated from Copy-Last.
* An action row **ANSWERS** when the gain is in [0.5, 1.5], the direction cosine is above 0.7, the sign is right, and it is separated from zero effect, at both seeds.

## 3. The gaps, the lever for each, and how the lever is validated

| gap | measured state | lever (method source) | validated by | cost |
|---|---|---|---|---|
| G1 no motion in any state | a, w, Δv2, Δψ2 not encoded at any level | **MRATE** W1: K = 3 states per level at its own spacing (pre-registered `50667a79`). Literature: DINO-WM history 1→3 is the biggest gain WITH a causal mask (0.76 → 0.92) and a collapse without it (0.76 → 0.08); DOME's biggest jump is 1 → 2 frames | IA-1 (a, w), IA-2; the causal-mask check is already pinned by MRATE's causality tests | 4 runs on box 2 (card) |
| G2 L1 overshoots ×2 | shrink c ≈ 0.5, SPEED-SKILL not separated | measure first: IA-3 splits the overshoot into action gain vs dynamics gain ("excess gain", LeWM / Intervention Gap); then a residual (Δ on Copy-Last) parameterisation at L1 if the gain is in the dynamics term | IA-3 gain within [0.5, 1.5]; IA-6 L2-vs-L1×2 slope near 1 | zero training, then one arm |
| G3 steering barely imagined, L1 wrong sign | w+ vs w− L3 +0.012 / +0.062 of 0.5 rad/s | **action-diverse resampling** of the world-model loss (DOME: 87 % of nuScenes training scenes go straight, and the model ignored the trajectory until resampled; +36 % mIoU). Count turning vs straight windows first. | IA-3 lateral gain and sign; action-bank MRR above chance | count: minutes; arm: box 2 |
| G4 no slow/fast split | top level holds less speed than the bottom | measure the timescale index (IA-6); then a level-factor sweep (stride 2 vs 4, Clockwork VAE) | IA-6 ordering of fast vs slow targets across levels | zero training, then 2 arms |
| G5 lead gap encoded nowhere | S1 not separated at both seeds | **capture ladder first**: probe the gap from full-resolution DINOv3 patch tokens → pool 8 × 20 → pool 4 × 10 (ours) → readers → WM levels. The rung where it is lost picks the lever: the encoder (LoRA L), the pool (unpooled tokens), or the WM loss (a scene-target auxiliary) | IA-1 on the gap at each rung | one DINOv3 forward pass on the val frames (GPU, about an hour) |
| G6 no scene imagination | S2 at most one seed separated | gated by G5 (capture first). After G5's lever: IA-2 on gap and occupancy, IA-5 BEV decoder | IA-2 IMAGINES on gap; IoU above Copy&Paste | follows G5 |
| G7 the imagined scene answers the action the wrong way | S3-PARTIAL +0.72 to +3.30 m, wrong sign | stratify IA-3 by scene cluster (is it a scene prior?); then the same resampling as G3 plus matched counterfactual pairs (same scene type, opposite action) | IA-3 gap sign within scene strata | zero training, then shared with G3 |
| G8 the retrieval decoder has no headroom | even the true future not separated on R2 | replace it as the yardstick: IA-2 probe transfer and IA-5 decoders; keep retrieval only with hard banks (same clip, other actions) and exact chance | IA-0 / IA-2 | in E1 |
| G9 the planner reads no world-model output | by design | **reliance arm**: the planner reads the predicted L1/L2/L3 states (Epona: a shared latent lifts PDMS 78.1 → 86.2; WoTE: scoring candidate trajectories with world-model rollouts, PDMS 84.0 → 85.6) | Q6: paired at T1-kin and T2 against the same planner without them | box 2 arm + T2 pass |

## 4. The experiments, ranked by information per cost

| id | what | training | compute | gaps | status |
|---|---|---|---|---|---|
| **E1 IMAG1** | IA-0, IA-1, IA-2, IA-3, IA-6 (timescale, L2 vs L1×2) on M2b s0 / s1 | none | dev-box CPU, hours | G1–G4, G7, G8 | card frozen with this plan |
| **E2 VOE1** | IA-4 surprise on log-built impossibilities: teleport (skip 1.6 s), freeze (repeat a frame while the actions say the car moves), reverse, splice (switch clips), action swap (real frames, another window's actions). Controls: random-init WM, shuffled scores, constant-latent null, norm-matched token noise as the visual-only perturbation. Event alignment of per-level surprise with braking onset and cut-ins from the cuboids | none | dev-box CPU, hours | G1, G3, G4 (does each level notice what it should) | card frozen with this plan |
| E3 CAPLADDER | the G5 capture ladder | none | one DINOv3-L pass at full resolution (GPU; the dev-box GPU is held by another session's jobs) | G5, G6 | card after E1 |
| E4 SUBGOAL | H-JEPA's own gradient planner recovers the logged action given (a) the true future latent, (b) L2's predicted subgoal, (c) a random goal (Mind the Gap's control; H-JEPA Eqs. 4–6) | none | CPU | G4, G9 | card after E1 |
| E5 BEVDEC | per-level cuboid occupancy decoder, trained on encoded latents only, applied to predicted; per-horizon IoU vs Copy&Paste and Oracle; shuffled-condition decoder | decoder only | hours | G6, G8; the pictures | after E3 |
| E6 MRATE | W1 multi-rate history | 4 runs | box 2 | G1, G2, G4 | card frozen; lands with the trainer after the suite |
| E7 DIVRS | action-diverse resampling of the world-model loss | arms | box 2 | G3, G7 | card after E1's counts |
| E8 SCNAUX or UNPOOL | a scene-target auxiliary (gap, TTC, occupancy from the cuboids; labels may use them, inference stays vision-only) OR unpooled tokens, chosen by E3's rung | arms | box 2 | G5, G6 | card after E3 |
| E9 REL | the reliance arm | arms + T2 | box 2 | G9 | card after E1 |

⭐ Order: E1 and E2 run now (zero training, both seeds, hours). E3 waits only for a GPU. E7–E9 are designed from E1's numbers, so their cards are written after E1 and before any of their code. E6 waits for the full-suite pass of the MRATE trainer change, which is running.

## 5. What the PI will see (visualisations, all from val windows; raw frames stay local)

1. **Imagination strip**, per window and level: the real frame now and at +0.8 / +1.6 / +2.4 s (local only), beside the decoded predicted BEV occupancy, the true occupancy, and the Copy&Paste occupancy; decoded speed and yaw rate against the truth.
2. **Action fan**: decoded imagined speed and lead gap for a sweep of longitudinal actions from one state, beside the kinematic null and the real outcome (V-JEPA 2-AC's energy sweep, made legible).
3. **Surprise trace**: per-level surprise along a drive, with braking onsets and cut-ins marked, beside the impossibility clips.
4. **Timescale spectrum**: NMSE per target per level as a heatmap. A working hierarchy shows fast targets fading upward and slow targets staying.
5. **Probe-transfer curves**: NMSE vs horizon per level, with the Copy-Last and Oracle bands.
6. **Latent distance maps** (H-JEPA Fig. 7–8): distance to the goal state along the logged path, per level.

## 6. What is new here (paper candidates, HYPOTHESIS until measured)
* No paper reports probe transfer from encoded to predicted latents per horizon **for a hierarchy** (REPORT_A, "does not settle"). IA-2 per level is that table.
* No paper sweeps history spacing against the decodability of acceleration or yaw rate (REPORT_B §3). MRATE is that sweep.
* Driving JEPA validation is thin (one modest nuPlan-mini signal, 2606.28383). IA-4 on PhysicalAI with four controls would be the first per-level surprise read on real driving video.

## 7. Process rules carried in
* Two-part separation (bootstrap CI AND the R = 199 permutation reference) for every per-row control (LATSEM2 run 1 lesson).
* Probes are fitted on encoded latents only; a probe fitted on predicted latents relearns the state from the copied context (REPORT_A F4).
* Probe capacity is reported (ridge and MLP both), because it can decide the answer (LeWM: linear 0.187 vs MLP 0.021 MSE).
* Capture before propagation: a gap that is not encoded cannot be imagined, so G6 is read only after G5's lever.
