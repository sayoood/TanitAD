# RESULT — E2: refcv4b on the OFFICIAL NavSim v2 scorer (warmup_two_stage, STAGE 2)

**Stream** EvalFlyWheel E2 · **2026-09-19** · **Checkpoint** `refcv4b-b1-v72-40k` (registry §4.6),
`ckpt_40284_FINAL.pt` md5 `99b573e8277d94a5e3bfbf630cb4d751` (verified this run, step 40,284,
strict load) · **Harness** `autonomousvision/navsim@0a380a9` + 3 pre-existing local patches + E1's
Windows loader patch (`navsim_win.py --patch-loader`), E1's verified C: mirror · **Pre-registered**
`SPEC.md` (blob `cb1d11de…`, amended once before any score → `3ca367ca…`; `raw/SPEC_PREREG_HASH.txt`).

Every number below is **MEASURED** by this stream unless marked; the artifact is named beside it.
Tier **T1-family**; stage-1 loop **OPEN**, stage-2 loop **UNRULED**; background vehicles
**IDM-reactive** in both stages; ⛔ never closed loop. **No interval**: 7 log groups < the RG-14
floor of 8 (`estimator.interval = UNAVAILABLE`). Statistic **S2-EPDMS-u** = the devkit's stage-2
aggregation of the official per-scene `score` with uniform within-group weights (SPEC §2) —
⛔ **not** a two-stage EPDMS.

## 1. Findings first

1. **BAR-E2-1 PASSES as pre-registered.** A1 (frames + t0 ego + NavSim command) scores
   **S2-EPDMS-u 0.4670** vs the devkit CV agent's **0.3971** on the identical 204 stage-2 scenes
   (**+0.0699**; paired per-scene W/T/L **51 / 76 / 77**; A1 ahead on **5 of 7** logs).
   `raw/scores_summary.json` → `BAR_E2_1`, `pairs.A1_ego_cmd__minus__CV_official`.
2. ⛔ **A do-nothing plan beats the bar arm.** The vision-pure arm A2 collapses to a
   near-standstill plan (202 / 204 scenes move < 1 m in 4 s) and scores **0.5211** — above A1,
   the echo control and CV. The explicit all-zero STOP plan (POST-HOC) scores **0.5212** (A2 − STOP **−0.0001**: A2 *is* a
   stop), and its **official two-stage EPDMS is 0.3009** — against CV's **0.1854** (= the HF
   warmup leaderboard's 18.54). **Standing still out-scores constant velocity by 11.6 EPDMS
   points on the official warmup protocol.** A1 − STOP = **−0.0542** (W/T/L 108 / 40 / 56: A1
   wins more scenes but its losses are zero-score collisions / off-road). **So the
   pre-registered bar was necessary and insufficient: on warmup stage 2 this statistic rewards
   not moving.** A1's win over CV is carried by the safety multipliers (NC **+0.118**, TTC
   **+0.113**, DAC **+0.054**, DDC **+0.047**) while progress and cross-frame comfort fall (EP
   **−0.152**, EC **−0.186**): A1 travels a median **0.55×** CV's 4 s distance on moving scenes.
3. **Why stopping is cheap here — a protocol property (PUBLISHED-CODE + MEASURED).** EP is the
   agent's progress divided by the best rule-compliant proposal's, and is set to **1 for every
   proposal when that best progress ≤ 5 m** (`pdm_scorer.py:231-236`); warmup's stage-2 starts
   are slow (median v0 **4.14 m/s**, **36 / 204** below 1 m/s; `raw/navsim_agent_inputs.json`).
   ⇒ **Every warmup NavSim row must be read against STOP, not only CV** (integration ask 5).
4. **A1 does beat the echo of its own ego inputs** (POST-HOC diagnostic, `ECHO_ha0_ext` =
   constant measured a0 and κ0 via the programme's `refc_v3.kinematic_goal_extrapolation`):
   **0.4670 vs 0.4287, +0.0383**, W/T/L 55 / 78 / 71 — won on DAC **+0.083** and DDC **+0.081**
   (road geometry, i.e. perception) and lost on EC **−0.451** (the plan changes more between
   consecutive frames than a kinematic echo does). The echo itself beats CV (+0.0316), and its
   **official two-stage EPDMS 0.2248** exceeds CV's **0.1854** (both genuine: neither needs frames).
5. **The paired ego-status contribution (the Lab's rec. 3) is NEGATIVE here:** A1 − A2 =
   **−0.0541** (W/T/L 95 / 39 / 70). Because A2 degenerates to a stop, this pair measures
   "moving vs stopping" under a statistic that rewards stopping — not perception with vs without
   ego. It is reported, not interpreted as "ego hurts".
6. **Vision is strongly load-bearing through the bridge; history construction and command are
   not the lever.** Frames-blind A4 collapses to **0.0950** (it leaves the drivable area on
   **86.3 %** of scenes; official two-stage EPDMS **0.0**): **A1 − A4 = +0.372**, W/T/L
   **111 / 81 / 12**, won on DAC **+0.632** and DDC **+0.512** — the model reads the NavSim road
   through the stitched frames. A1NT (nearest-time 2 Hz history) **0.4511** (A1 − A1NT
   **+0.0159**, static history no worse); A2NT **0.5207** (≈ A2, both stop). Command: A3 (nav
   withheld) **0.4546** (A1 − A3 **+0.0124**, W/T/L 68 / 84 / 52). Nav-compliance of the plan
   (NavSim's own 20 m / ±2 m rule): left-commanded (n 11) **0.364** with nav vs **0.273**
   without; straight (n 193) **0.959** vs **0.953** — ⚠️ only 18 % of left plans even reach 20 m.
7. ⛔ **No refcv4b two-stage EPDMS exists on this box, and none is claimed.** The 16 stage-1
   scenes have **0 / 192** camera jpgs (every warmup jpg is a 17-hex synthetic render); they set
   both the stage-1 factor and the stage-2 kernel weights. CV's official numbers on the same
   harness: stage 1 **0.4603**, stage 2 **0.3341**, combined **0.1854** — bit-identical to E1's
   independent CV run (223 rows, max |Δ| **0.0**).

## 2. All arms (stage 2, n = 204 scenes, same tokens; `raw/scores_summary.json`)

| arm | inputs | S2-EPDMS-u | scene mean | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **A1** `ego+cmd` (bar) | frames ST + ego t0 + cmd | **0.4670** | 0.4449 | 0.799 | 0.770 | 0.858 | 0.961 | 0.566 | 0.794 | 0.559 | 0.971 | 0.402 |
| **A2** `vision-pure` | frames ST only | **0.5211** | 0.5054 | 0.926 | 0.926 | 0.983 | 0.980 | 0.365 | 0.926 | 0.593 | 0.662 | 0.235 |
| **A3** `ego, no cmd` | frames ST + ego t0 | **0.4546** | 0.4345 | 0.799 | 0.765 | 0.860 | 0.956 | 0.564 | 0.794 | 0.549 | 0.971 | 0.431 |
| A1NT (sensitivity) | frames NT + ego + cmd | 0.4511 | 0.4327 | 0.799 | 0.750 | 0.843 | 0.961 | 0.565 | 0.789 | 0.564 | 0.907 | 0.451 |
| A2NT (sensitivity) | frames NT only | 0.5207 | 0.5043 | 0.922 | 0.922 | 0.980 | 0.980 | 0.375 | 0.922 | 0.598 | 0.662 | 0.235 |
| A4 `frames-blind` (diagnostic) | constant grey + ego + cmd | **0.0950** | 0.0874 | 0.618 | 0.137 | 0.346 | 0.966 | 0.860 | 0.574 | 0.441 | 0.882 | 0.049 |
| **CV** (reference, devkit) | ego_velocity[t0] | **0.3971** | 0.3756 | 0.681 | 0.716 | 0.811 | 0.946 | 0.718 | 0.681 | 0.564 | 0.995 | 0.588 |
| ECHO `ha0_ext` (post-hoc) | ego t0 (a0, κ0) | 0.4287 | 0.4145 | 0.765 | 0.686 | 0.777 | 0.961 | 0.584 | 0.755 | 0.529 | 0.995 | 0.853 |
| STOP (post-hoc) | nothing (all-zero plan) | **0.5212** | 0.5063 | 0.926 | 0.926 | 0.983 | 0.980 | 0.359 | 0.926 | 0.593 | 0.662 | 0.255 |

Official two-stage EPDMS exists only for arms that need no frames: **CV 0.1854** (stage 1 0.4603 /
stage 2 0.3341), **ECHO 0.2248** (0.3895 / 0.3998), **A4 0.0000** (0.0547 / 0.0950), **STOP 0.3009** (0.5777 / 0.5212). The
camera arms' `extended_pdm_score_*` rows are HYBRID (stage 1 = devkit CV stand-in) and are not
reported as their EPDMS (`score_<arm>.csv` keeps them, labelled in `scores_summary.json`).

## 3. Bar verdict and the Rule-Zero decomposition

**BAR-E2-1: PASS** (0.4670 > 0.3971). Decomposed (A1 vs CV, `raw/…` pairs + per-scene analysis):

* zero-score scenes (any multiplier failed): **A1 72 vs CV 100**; multiplier flips CV-fail → A1-pass
  vs the reverse: NC **27 vs 3**, DAC **22 vs 11**, DDC **24 vs 10**, TLC **3 vs 0**;
* by t0 speed: v0 < 1 m/s (n 36) **+0.031**; 1–4 (n 64) **+0.100**; 4–8 (n 59) **+0.151**;
  **≥ 8 m/s (n 45) −0.052** — A1 is WORSE than CV at speed, where braking costs EP and buys less;
* by command: left (n 11) **+0.152**; straight (n 193) **+0.065**; no right / unknown at t0.

**Rule Zero.** The bar passed, but the STOP floor says the bar did not measure driving. The
pre-declared ladder was walked in the same run: time construction (A1NT) — no gain; command
(A3) — +0.0124 only; vision (A4) — +0.372, i.e. vision is the part that WORKS. The deficit vs a stop is the multipliers (A1 fails NC on 20.1 % of
scenes, DAC 23.0 %, vs A2's 7.4 % / 7.4 %): the model moves into collisions and off the drivable
area on renders from a camera rig it never trained on. Every lever that remains is blocked
beyond this stream's authority (named in §8): the fix for "moves into hazards" is perception /
planning quality under the domain shift, i.e. NavSim-side training (navtrain) or a planner
change (e.g. a hazard-aware stop fallback) — a training / architecture decision, not a cheap input
fix; and the headline protocol needs frames this box does not hold.

## 4. Controls (each must read a known value — all read it)

| id | control | result | artifact |
|---|---|---|---|
| K1 | frame integrity | **204 / 204** bank sha256[:16] == provenance, u8 (4,256,640,3), checked per scene before use | `raw/seam_*.manifest.json` (`frame_sha16`) |
| K2 | GT round trip (human futures → same conversion) | knots reproduced to **7.3e-15 m** max (pass < 1e-3) over **5,093** paths; interpolation 2.5 s mean **0.0085 m** (p95 0.025, max 0.089), 3.5 s mean **0.015 m** (p95 0.047, max 0.141); heading (tangent vs logged yaw, moving n 3,495) mean **0.54°**, p95 1.77° | `raw/K2_K3_K9_roundtrip.json` |
| K3 | frame + lateral sign (`navsim.py::verify_frame`) | axis OK (rel err 0.051); y-sign agreement **0.9838** on **1,052** turning windows | same |
| KF | TanitAD training GT frame (`refb_labels.ego_frame`) vs devkit local poses | **3.6e-15 m**, heading 0.0, n 204 | same |
| K9 | NavSim command index order | argmax 0 → +y on **99.6 %** of 283 turning windows; argmax 2 → −y on **100 %** of 283 | same |
| KC | curvature channel κ = ay / max(v0,4)² | sign agreement **0.969**, Pearson **0.958**, median ratio **1.024** vs realised curvature (n 712) | same |
| K4 | seam transparency | devkit CV poses through the seam agent ≡ official CV run: **220 tokens × 20 columns, max \|Δ\| 0.0** | `raw/scores_summary.json` `K4_seam_transparency` |
| K5 | undeclared ego fields randomised | A1 **20 / 20**, A3 **20 / 20**, A2 (all ego) **20 / 20** byte-identical | `raw/K5_K6_ego_mutation.json` |
| K6 | declared field mutated (v0 × 1.5) | A1 changed on **20 / 20** (max 11.49 m) — the probe can fail | same |
| K0 | inference determinism | 20 / 20 byte-identical repeat forwards | same |
| K7 | my packer vs the trainer's window (real PhysicalAI frames) | **6 / 6** windows byte-equal; one-frame shift detected **6 / 6** | `raw/K7_packer_vs_trainer.json` |
| K8 | empty-set guard | every reported run: log `successful 220`, `failed 0`, CSV 220 valid rows, agent calls == seam declaration; it REFUSED the two STOP attempts that E1's RAM guard aborted mid-run (91 and 25 calls) | `raw/score_*.counts.json`, `raw/score_STOP_zero.attempt{1,2}_ram_guard_abort.*` |
| KX | cross-stream replication | my CV run ≡ E1's CV run, 223 rows, max \|Δ\| 0.0 | this file §1.7 |
| KB | frame builder (for navhard) | rebuilt 6 warmup scenes **bit-exact** vs the DataFlyWheel bank | `raw/frame_builder_reproduction.json` |

## 5. Ego-status enforcement (gate `navsim.ego_enforcement`)

Mechanism: the **declared-input seam** — `code/tanitad_navsim_bridge.py::declare` copies ONLY an
arm's declared t0 fields out of the devkit's own `AgentInput` export; the model call receives
that dict and the frames, nothing else. Evidence: (i) per-arm manifests (`raw/seam_<arm>.manifest.json`:
declared list, 13–16 withheld fields, per-token values fed); (ii) K5 byte-identity under
randomisation of every undeclared field, with K6 proving the probe can fail; (iii) A2 invariant
to ALL ego fields. The NavSim-side agent is a lookup keyed by the scorer token that re-checks the
AgentInput fingerprint (a byte-identical ego history is shared by 8 groups / 19 tokens of distinct
renders, so an AgentInput key was refused by measurement). `tools/criteria_check.py` over each
artifact: **0 violations** (15–18 reasoned work items) on all 6 arm artifacts; the four
BLOCKING NavSim gates — ⛔ which `tools/criteria_check.py` (registry v2.9.0) does **not evaluate at
all** (0 occurrences of `navsim` in its report) — were self-checked from the registry's own gate
keys: `navsim.ego_enforcement` **PASS**, `navsim.modality_label` **PASS**, `navsim.estimator_unit`
**PASS** (`{UNAVAILABLE, reason, n}`), `navsim.cross_protocol` **PASS** (`EPDMS_v2_warmup_two_stage`
+ devkit SHA) on every arm (`raw/artifact_<arm>.json` → `navsim_gate_selfcheck`,
`raw/criteria_check_<arm>.{txt,json}`). ⚠️ `route_leak_check` = **UNVERIFIED** (stream E3).

## 6. Four metric families

Our instruments need the plan AND a GT future on the same scene; warmup has none for a camera arm
(stage 2: frames, no future; stage 1: future, no frames) ⇒ LONGITUDINAL / LATERAL / TACTICAL
**UNAVAILABLE, reason + n = 0, per criterion** in every camera artifact. STRATEGIC
nav-compliance needs no future and IS computed (plan's offset where its arc reaches 20 m, NavSim's
own ±2 m rule; paired vs the nav-withheld A3): overall **0.926** vs **0.917** (paired **+0.010**,
n 204); left-commanded (n 11) **0.364 vs 0.273**; straight (n 193) **0.959 vs 0.953**; no
right-commanded scene exists at t0. ⚠️ Weak readout: only 18 % of left-commanded plans (26 % of
straight) reach 20 m of arc — the model's slow plans often end before the command's decision
point. The nav-SHUFFLE control was not run (reasoned refusal in the artifact). Stage-1 families (n = 16) for the two
arms that can run stage 1 — A4 (frames-blind) and CV — are in `raw/four_families_stage1.json` (T1, n 16, no CI; neither is a
refcv4b camera arm): CV speed MAE **0.893 m/s**, along **1.267 m**, cross **1.066 m**, heading
**7.48°**, curvature **0.0155 1/m**, yaw-rate **4.47°/s**; A4 **1.879 / 2.702 / 2.249 / 11.58° /
0.0230 / 8.31°/s** — worse than CV on every family, consistent with its EPDMS.

## 7. Proposed register rows (not edited here — `GOALS_AND_CLAIMS.md` is staged by others)

* **D-NAVSIM-E2-1** (MEASURED): refcv4b zero-shot, warmup stage 2 (204 scenes), S2-EPDMS-u
  A1 0.4670 > CV 0.3971 (BAR-E2-1 PASS) — **but the all-zero STOP plan scores 0.5212 (official two-stage EPDMS 0.3009 vs CV 0.1854) and
  the vision-pure arm 0.5211**; the stage-2-uniform warmup statistic rewards standing still. Not a two-stage EPDMS.
* **D-NAVSIM-E2-STOPFLOOR** (PUBLISHED-CODE + MEASURED): EP ≡ 1 for every proposal when the best
  rule-compliant progress ≤ 5 m (`pdm_scorer.py:231-236`); every NavSim row must carry a STOP
  floor beside CV.
* **D-NAVSIM-E2-STAGE1** (MEASURED): warmup_two_stage's 16 stage-1 scenes have 0 / 192 camera jpgs
  on the box; a camera agent's warmup two-stage EPDMS needs the OpenScene-test camera archives.
* **C-DATAFLYWHEEL-NAVSIM-COVERAGE** (retraction class *coverage stated for a whole from one of its
  parts*): "204 / 204 scenes" is the stage-2 count; stage 1 was never built.

## 8. Levers and blockers (Rule Zero, point 3)

**Done-condition (Rule Zero 3):** the committed bar is CLEARED (BAR-E2-1 PASS), and the remaining
levers are **blocked** — each named with what unblocks it:

| # | lever | blocked on | unblock |
|---|---|---|---|
| 1 | the OFFICIAL two-stage EPDMS for refcv4b on warmup | stage-1 camera frames: **0 / 192** on the box | PI: download the 7 warmup logs' OpenScene-test camera frames (inside `openscene_sensor_test_camera_{0..31}.tgz`), then `code/build_frames.py --stage 1` + `run_bridge.py` (stage-1 frame path) — or go to #2 |
| 2 | the official column: **navhard_two_stage** | E1's navhard metric cache + the PI-authorised sensor download (in flight) + frames for 450 + 5,462 scenes | `export_agent_inputs.py` (`E2_SPLIT=navhard_two_stage`), `build_frames.py --split navhard_two_stage` (proven bit-exact on warmup), `run_bridge.py`, `score_arm.py --split navhard_two_stage`; ⚠️ verify first that navhard's curr/hist archives carry stage-1 originals |
| 3 | beat STOP (the real floor), i.e. stop moving into hazards | a TRAINING decision (NavSim/navtrain fine-tune, forbidden on the test splits) or a PLANNER change (hazard-aware stop fallback) | PI / Master Mind — outside this stream's authority; cheapest first step is the navhard number, which says whether STOP also dominates the official protocol |
| 4 | a decision-grade interval | the NavSim cluster unit (E3) — and warmup's 7 logs < RG-14 8 regardless | E3's estimator registration; navhard |
| 5 | nav-shuffle control; A1 vs ECHO / STOP as PRE-REGISTERED bars | nothing — budget; cheap on navhard | pre-register `BAR = A1 > max(CV, STOP, ECHO)` for navhard |

**Stamps.** Tier **T1-family**; stage-1 loop **OPEN** (PI 2026-09-02); stage-2 loop **UNRULED**;
background vehicles **IDM-reactive**; ⛔ never closed loop. Protocol `EPDMS_v2_warmup_two_stage`,
**stage-2 rows only** for camera arms; devkit `0a380a9` + the 3 pre-existing local patches E1
recorded + E1's loader patch. **Zero-shot** (PhysicalAI-AV B1 → nuPlan synthetic renders),
**3-camera stitch → 256×640 cylindrical**, perception-free, CPU inference, non-parity corpus.

## 9. Where everything is

Code `code/` (bridge, seam agent, scorer driver, parser, artifact builder, frame builder, control
seams), tests `tests/` (K2/K3/K9/KF/KC round trip, K5/K6/K0 ego mutation, K7 packer), raw evidence
`raw/` (export, seams + declared-input manifests, official CSVs + logs + counts + agent call logs +
E1-wrapper manifests per arm, `scores_summary.json`, artifacts + criteria_check per arm, controls).
All in the repo under this package; nothing exists only on the dev box except the E1-built metric
cache (E1's deliverable) and the DataFlyWheel frame bank (bit-reproducible, §4 KB).
