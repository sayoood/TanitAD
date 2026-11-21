# v7F under the PI's R1–R6 directive — what a compliant first run is, what blocks it, and the one decision

**2026-09-27 · TrainingFlyWheel · read from tip source (`b3f7ea6f`; `v6.py` and `train_v6_staged.py` unchanged at
`546f34c9`) · STAGED, not committed.** PI: *"close P3, solve the rest"* + requirements R1–R6 (memory
`pi-requirements-refav1-v7f-2026-09-27`).

## 1. The stage plan the requirements imply
`v6.py:4567` `STAGE_GROUPS`: **S-W** = `encoder, readout, predictor_op, aux` (the world model) · **S-T** =
`layer_tac, planner` · **S-S** = `layer_str` · **S-J** = everything trainable.
⇒ **R6 (strategic off) = S-W → S-T, S-S skipped**, and any joint phase must exclude `layer_str`.
⛔ The pre-registered v7F run (`PREREG_V7F.md` §9) is **S-W only**: it trains none of R3/R4/R5, so it is not a
compliant first run under R1–R6 whatever its outcome.

## 2. What S-T needs that does not exist yet (from source)
| req | state in the v7 trainer | to build |
|---|---|---|
| R1 max speed input + cap | 0 max-speed hits in `train_v6_staged.py` | the per-clip limit (`refcv6_max_speed`, same as refav1's R1) as a tactical INPUT, and the cap on the emitted plan |
| R2 nav → tactical + operative | ✅ `--nav-cond`, all layers | — |
| R3 ALL tactical labels train the tactical layer | no tactical-label loss; `D-TLIGHT-1`: the traffic-light goals "never reach training" (`test_tactical_label_reach.py` pins it) | a tactical label loss over the full v7.2/v8 goal vocabulary (`--goal-factored` / `--goal-cat-args` heads exist) |
| R4 tactical conditions the operative plan | `--tac-goal-cond` builds the **strategic→tactical** port, not tactical→operative | verify/build the tactical-goal → operative-planner conditioning |
| R5 combined 6 s trajectory | the planner emits a 60-step (6 s at 0.1 s) plan (`--dump-seam-plan`) | train it in S-T |
| launch gate | ⛔ `launch_gate.py` absent; 232 flags, 36 loss/model levers default OFF (`raw/flag_inventory_train_v6_staged.*`) | G-DVB/G-LIVE entries for every S-T lever |

## 3′. ⛔ CORRECTION (2026-09-27 ~14:55, from source) — §3 and §4 below misstate what gates S-T
**What I wrote:** "S-T's planner chooses by imagining each candidate's outcome with the world model", so P2 (action-
deafness) blocks S-T. **What the code says** (`v6.py` at `c36b6ddd`): S-T's `planner` group emits a goal-conditioned
FAN of 60 × (a, κ) candidates (`DiffusionProposalGenerator`, `:3129`, conditioned on plan features + the tactical goal
embedding + v0, trained winner-take-all) and SELECTS with a learned scorer over the emitted waypoints — goal distance
(`GoalDistanceScorer`) or its capacity control (`MLPCandidateScorer`), `:5800-5818` — optionally MPC-refined and
re-scored by goal distance ONLY (`MpcRefiner`, `:3314`). The world model's roll-consistency enters only as an MPC
regularizer (`mpc_w_consist` default 0.0) and as the fallback trigger's MONITORED uncertainty signal; as a SELECTOR
it was already MEASURED **+5.98 m worse** than the goal selector (`v6.py:3946-3950`) and is refuted.
⇒ **P2 does not block S-T. What blocks it is `SEL-1`:** any selector launch is REFUSED
(`scripts/v6_chain.py:501 assert_selector_admissible`) because E-WC2 (2026-08-16) measured σ/ADE **9.99
[7.45, 13.51]** against a pre-registered 3.0 refusal line — and a 0-parameter constant-yaw-rate goal beat the ridge
from latents: *"THESE LATENTS ARE THE WRONG SURFACE"* (measured on REF-C's surface; S-W latents have never been
dumped). Its committed unblock is the E-WC2-SW dump at the S-W → S-T boundary, which needs an S-W-trained v7F.
⭐ **The same finding, independently, on refav1 today:** neither refav1's global-mean state nor a 4 × 10 region pooling
lets a readout beat t0 kinematics (`…/2026-09-27-refav1-trunk-probe/`). Two architectures, one pattern.
**What survives of §4:** option B (ONE trajectory, decoded from the tactical goal + nav + max speed + t0 kinematics as
a residual over a kinematic prior) needs no selector, so it sidesteps SEL-1 — the recommendation C (B first) stands,
for this reason and not the P2 one. Option A as written ("solve P2 first") is the wrong prerequisite for S-T; the
right one is SEL-1's admission (the S-W latent dump). P2 remains a genuine research problem for imagination-based
PLANNING, and for the world model's own quality.
⭐ **And option B is already in the code — a configuration, not a build.** The S-T emission (`v6.py:5766-5795`) is a
goal-conditioned decoder: `plan_proj(z_op) + cand_queries ⊕ g_tac_embed → UnicycleEmission(·, v0)` → N × 60 × (a, κ)
(6 s at 0.1 s), its final layer zero-initialised to the constant-velocity prior; `squeeze_candidates` collapses N = 1.
The trainer exposes `--n-candidates` (default 8), `--proposals query|diffusion` and `--selector none|goal|mlp`
(`train_v6_staged.py:9271-9298`). ⇒ **B = `--n-candidates 1 --proposals query --selector none`**: ONE trajectory,
conditioned on the tactical goal (R4), 6 s (R5), no selector — so `assert_selector_admissible` does not apply
(`v6_chain.py:512`: selector `"none"` ⇒ `applies: False`). What B still needs is exactly R1 (max-speed input + cap)
and R3 (all tactical labels → loss) — the two streams building now — plus two design upgrades the refav1 evidence
argues for: a kinematic PRIOR better than constant velocity (`kd_x` / a learned kinematic readout, −0.23 to −0.31 m
vs `ha0_ext` on refav1's grid), and an operative input richer than the pooled `z_op` (refav1's pooled state carried
nothing beyond kinematics).

## 3. (SUPERSEDED by 3′ — kept as written) The blocker under all of it: P2/P5 — the world model ignores its actions
S-T's planner chooses by **imagining** each candidate's outcome with the world model. MEASURED: actions move the
prediction 0.4–0.6 % as much as the scene; the predictor adds nothing over `z_t` on any arm (`D-V7F-L3-RULED`);
feeding the true future actions and holding the last one give the **same** rollout (`D-V7F-L3-ACTION-DEAF`). An
action-deaf world model scores every candidate the same, so an imagination planner cannot choose.
Ranked causes (`V7_LAUNCH_GATE.md` P2): **(d) teacher-forced targets** admit an action-invariant solution — the only
cause with a published mechanism (UWM-JEPA 2605.25313: action sensitivity requires counterfactual targets); our
simulator-free version, O11, was **positive by construction** (`H-LEAK-6`) and its re-run needs a same-clip
negative sampler (a batch-construction change, possibly a parity question). **(b)** the "action" is realised motion,
never a command — **untestable on PhysicalAI**; comma2k19 carries real CAN steering. **(c)** latent geometry, untested.

## 4. ⭐ THE DECISION (PI) — how the operative plan is produced in the first compliant v7F
| option | what it is | cost | risk |
|---|---|---|---|
| **A — imagination planner (the programme's thesis)** | solve P2 first: O11 re-run with same-clip negatives (sampler build + 2 tiny arms + replicate), then S-T | days before S-T can start; P2 may not yield on logged data | P2 stays open → no S-T |
| **B — direct tactical-goal-conditioned decoder at the operative level** | the operative layer DECODES the 6 s trajectory from scene latent + tactical goal + nav + max speed (refcv6's proven pattern); the world model stays for representation and for SCORING, not for choosing | ~S-T build only; P2 can be pursued in parallel | departs from "every planner predicts via imagination" until P2 is solved |
| **C — both, sequenced** | B for the first positive result; A as the parallel research line that replaces B's selection once P2 is solved | B's cost now, A's later | none beyond A and B |
**Recommendation: C.** It gives the PI a compliant, trainable v7F now without waiting on P2, and keeps the thesis
alive as a measured research line instead of a blocker.

### 4b. ⭐ Evidence from refav1 TODAY that bears on this decision (MEASURED 2026-09-27, T1, 141 clusters)
- **An imagination planner over an action-deaf world model cannot steer — measured on the sister model.** refav1's
  world model contributes 1.63e-10 of the lateral cost; its pre-registered full-grid test FAILED (`loncomb3 − ha0_ext`
  +0.0896 / +0.1515 m ADE at both inference seeds), and every zero-training planner lever is eliminated
  (`…/2026-09-27-refav1-fullgrid-loncomb3/RESULT.md`). This bears on the THESIS (imagination-based planning), not on
  v7F's S-T, which already selects without imagination (§3′). It strengthens **C (B first)**: B emits one trajectory
  and needs neither an action-sensitive world model nor an admitted selector.
- **The floor option B must beat is higher than `ha0_ext`.** A planner-free damped hold (`damp50`) beats `ha0_ext`
  by −0.158; the damped path re-timed to the held acceleration (`kd_x`) by −0.23 to −0.26; and a RIDGE READOUT on 8
  t0 kinematic features beats `kd_x` by a further −0.056 on held-out episodes (`…/2026-09-27-refav1-trunk-probe/`).
  ⇒ option B's decoder should predict a RESIDUAL over a kinematic prior, and its bar is the LEARNED kinematic
  readout (same inputs minus vision), not the undamped hold.
- **A decoder must read the token field, and even that is not free.** On refav1, neither the global-mean state
  (what its heads read) nor a 4 × 10 region pooling carried trajectory information a linear readout could use beyond
  kinematics when fitted on ~113 episodes; the 600-train-clip test (power fix + nonlinear readout) is running. The same
  probe is the cheapest pre-flight for v7F's trunk before option B is trained: if v7F's S-W trunk fails it too, B's
  decoder has nothing to decode.

## 5. The other hold items under R6
- **P1** (no v7 arm beats hold-action at T1): closes only with an S-T-trained v7F read at T1 against hold-action **and
  the damped hold** (a model-free damped hold beats `ha`/`ha0` by ~0.15 m ADE on the refav1 grid — that is the honest
  "do nothing" bar now). Needs S-T (§4).
- **P4** (the horizon ladder is short of the ~12.5 s label events): a STRATEGIC-horizon problem. Under R6
  (strategic off) and R5 (6 s), the first experiments target 6 s; P4 returns with the strategic phase — **stated here
  so it is not silently dropped.**
- **P3** CLOSED (PI, 2026-09-27; `D-V7F-DRIFT-NULL`).

## 6″. ⭐ What refav1's trunk verdict (16:45) means for v7F option B
refav1's frozen trunk failed to beat t0 kinematics under every readout, and a learned KINEMATIC MLP scores 6 s ADE 2.93 m
(vs `damp50` 3.45). v7's own E-WC2 found the same on REF-C latents. ⇒ (1) option B's decoder must be read against the
kinematic MLP (same inputs minus vision), not against `ha0_ext` — beating 5.36 m at 6 s is not evidence of driving;
(2) before S-T is funded, run the trunk probe on the S-W trunk (the same instrument, `…/2026-09-27-refav1-trunk-probe/
code/`): if the S-W latents also add nothing to the kinematic MLP, option B would converge to kinematics and the S-W
objective, not the decoder, is the lever; (3) the kinematic MLP is the natural residual prior for option B's decoder
(its zero-init currently starts at constant velocity).

## 6′. State at ~15:25 (supersedes §6's list below)
| item | state |
|---|---|
| R1 max speed (input + cap) | ✅ built + verified (`v7f_r1r4/`; `--max-speed-input-v6`, `--speed-max-sidecar-v6`, `--plan-vmax-cap`) |
| R2 nav | ✅ **three launch-blocking defects fixed** (`v7f_merge/`: build-time nav keys, step-1 overwrite, ungrouped nav params; 4/4 mutants) |
| R3 all tactical labels → tactical loss | building (`v7f_r3/`); at the tip NO family reaches any loss and S-T never loads the labels |
| R4 tactical → operative | ✅ goal path present at the tip; behaviour path built (`--tac-op-cond`; recommend `detached`) |
| R5 one 6 s trajectory | ✅ exists (60 × 0.1 s); first run = option B `--n-candidates 1 --proposals query --selector none` |
| R6 strategic off | S-S skipped by the stage plan; `--strategic-off` (zero strategic conditioning into the tactical goal heads; refuses `--tac-goal-cond`, which the chain's S-T surface currently turns ON) built in `v7f_merge/` |
| launch gate | a `v7f` profile is being built (`v7f_gate/`); no v7F launch is possible without it |
| S-W trunk | the pre-registered S-W run (`PREREG_V7F` §9) — needs the hold released for S-W → S-T under R1–R6, D1 (the DINOv3 wrap), and a Thor slot (Thor's GPU was idle at 14:30) |

## 6. What runs first (dev box, no Thor)
1. The v7F launch gate: G-DVB entries from the inventory, G-LIVE smoke on S-T (every declared loss finite, gradient
   into `layer_tac`/`planner`), with the R1–R6 checks as gate items (a strategic group with grad > 0 in S-T FAILS).
2. The tactical label loss over the full vocabulary (R3), pinned by extending `test_tactical_label_reach.py` so the
   traffic-light goals must now REACH training.
3. Option B's decoder or option A's sampler — per the PI's answer to §4.
