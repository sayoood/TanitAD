# RESULT — NavSim v2 camera-only lane: the ≤300M field, and what an entry needs beyond PDMS

`Research Lab daily pass 2026-08-28, Opponent Analysis (serves Benchmarks & Evals
too — EvalFlyWheel handoff inside). Literature-only (0 GPU). Seed: enumerate
camera-only entries at/under ~300M with PDMS and inputs (reference: Drive-JEPA
89.0 PDMS @307M); name what an entry needs beyond PDMS. All numbers below from
banked primaries (Drive-JEPA lib 2601.22032 Tab. 1–3/6; DrivoR lib 2601.05083
Tab. 3; TOAD lib 2606.07170 Tab. 1–2/5–6).`

## Findings first

**F1 — Our reference number decoded: "Drive-JEPA 89.0 PDMS @307M" is the
PERCEPTION-FREE arm on NAVSIM v1 navtest** (V-JEPA-2-init ViT-L 307M encoder +
lightweight transformer planner, single front camera, 330 h curated video).
The full Drive-JEPA framework scores **93.7 PDMS (v1)** and **87.8 EPDMS (v2,
navtest-protocol)**. [PUBLISHED, lib `2601.22032`] Perception-free peers: LAW
83.8 (21M), World4Drive 85.1 (21M), Epona 86.1 (1.1B — over cap).

**F2 — ⛔ There are TWO "NavSim v2 EPDMS" protocols in the wild and they differ
by ~30 points; cross-quoting them would manufacture a fake 30-point gap.**
[PUBLISHED, all banked]
- *Papers' v2 protocol* (navtest-style EPDMS): Transfuser 76.7 → HydraMDP++ 81.4
  → DriveSuprim 87.1 → Drive-JEPA 87.8.
- *Challenge protocol* (navhard-two-stage, stage-2 = 3DGS-synthesized
  counterfactual starts): privileged PDM-Closed 56.6; best camera-only
  **DrivoR + TOAD test-time search 56.3**; best purely-learned DriveFuture 55.5;
  GTRS-D (V2-99) 45.0; iPad 34.7 (+TOAD 49.8); RAP-DINO ViT-H 39.6.
  DrivoR's own table carries a further trap: computed **after an official
  harness bug fix**, excluding pre-fix rows — leaderboard numbers move with
  harness versions.
- ⚠️ Drive-JEPA reports NO navhard-two-stage number at all ("navhard" absent
  from the paper) — our reference model is UNPLACED on the actual challenge
  protocol. [MEASURED absence, 2 probes: full-text regex + table scan]

**F3 — The ≤300M camera-only field (best-per-family, with inputs):** [PUBLISHED]

| entry | trained params / backbone | inputs | v1 PDMS (navtest) | v2 EPDMS navhard-2-stage |
|---|---|---|---|---|
| DrivoR (CVPR26, Valeo) | "roughly 40M" total (register bottleneck 0.6M); pretrained-ViT-based | multi-cam + **ego status** (poses, vel, accel, command added to trajectory queries) | 94.6 | ~54.6 base; **56.3 + TOAD** |
| DriveFuture | UNVERIFIED params | UNVERIFIED inputs (learned, no TTO) | — | 55.5 |
| Drive-JEPA | 307M encoder + light planner | front cam + **ego status** (command, speed, accel into proposal queries) | 93.7 (89.0 perception-free) | not reported |
| DriveSuprim | ViT/L | camera | 93.5 | (87.1 on papers' v2 protocol only) |
| iPad | ViT/L | camera | 91.7 | 34.7 (+TOAD 49.8) |
| GTRS-A/D (v2-challenge winner family) | V2-99 backbone | camera (LiDAR use UNVERIFIED) | 90.4 (GTRS) | 45.0–53.2 |
| RAP-DINO | ViT-H (over 300M cap) | camera | 93.8 | 39.6 |

Two structural reads: (a) **~40M total is enough to lead the camera-only lane**
(DrivoR) — parameter budget is NOT the binding constraint on this leaderboard,
which supports our sub-300M thesis outright; (b) the navhard ranking INVERTS
chunks of the navtest ranking (RAP-DINO 93.8 v1 → 39.6 navhard; iPad 91.7 →
34.7) — navtest skill does not transfer to re-rendered OOD starts, which is the
content of the v2 lane.

**F4 — Every camera-only leader consumes EGO STATUS at inference.** DrivoR
encodes poses/velocities/accelerations/driving command into trajectory queries;
Drive-JEPA injects command/speed/accel into proposal queries. [PUBLISHED, both
banked] The benchmark permits it; our vision-only-at-inference doctrine
(2026-08-03, binding) forbids it. This is a REAL entry blocker and a PI
decision, not an engineering task — options in the recommendation.

**F5 — Opponent-measured support for our encoder thesis:** Drive-JEPA's own
frozen-encoder ablation (perception-free navtest v1): DINOv2 ViT-L **76.1** vs
V-JEPA 2 ViT-L **86.1** vs their driving-video-continued V-JEPA **89.0** — a
+10-PDMS gap from video-predictive pretraining over image-SSL at matched
architecture, and +2.9 more from in-domain continuation. [PUBLISHED, lib
`2601.22032` Tab. 1/6] Directly relevant to our frozen-encoder and O14 threads:
the planning readout rewards predictive video features, not appearance features.

**F6 — What an entry needs beyond PDMS** (the seed's second half), assembled
from the banked primaries + challenge docs: (1) **two-stage robustness** — stage
2 scores the agent from 3DGS-re-rendered counterfactual starts; domain-gap
robustness to synthesized views is scored, and our NuRec/gsplat pseudo-sim work
(H-EVAL-1) is the matching in-house asset; (2) **per-stage sub-metric vector**
(NC, DAC, DDC, TLC, EP, TTC, LK, HC, EC — reported per stage, then aggregated
to EPDMS) — an entry must emit and track all of them, which maps cleanly onto
our four-families doctrine (LK/DDC lateral, EP/TTC/NC longitudinal-interaction,
HC/EC comfort); (3) **harness-version pinning** (DrivoR's post-bug-fix note);
(4) **a sanctioned validation split** — warmup-two-stage intersects navhard and
required author validation before use; (5) the perception interface: NavSim
serves the nuPlan multi-cam pinhole rig, not our cylindrical 256×640 — the
scene-token estimator gap is a RIG adapter problem, and DrivoR publishes the
mechanism: camera-aware register tokens, **0.6M params, 250× fewer tokens,
"nearly reaches the performances of the no-compression model"** — converging
with Alpamayo-R1's Flex result (20× compression, metrics maintained,
LAB-RUN-001). [PUBLISHED]

## What this changes for TanitAD (≤3 recommendations)

1. **EvalFlyWheel: pin the target as navhard-two-stage EPDMS with harness
   version + split stamped on every number**; treat papers' navtest-EPDMS as a
   separate, non-comparable column (F2). Our criteria-registry gains the 9
   sub-metrics as PRESENT/REFUSED/ABSENT rows mapped to the four families. The
   competitive bar to state in the registry: camera-only learned 55.5 / with
   test-time search 56.3 / privileged ceiling 56.6.
2. **Build the NavSim entry's perception interface as a register-token scene
   estimator** (DrivoR's 0.6M/250× mechanism) rather than a bespoke rig
   adapter: it simultaneously (a) closes the scene-token estimator gap, (b)
   makes the entry rig-agnostic (their multi-cam → our cylindrical), and (c) is
   the published validation of our compact-scene-token thesis at 40M total —
   the sub-300M claim is already community-demonstrated, our differentiator
   must be the hierarchy/world-model, not size alone.
3. **Put the ego-status question to the PI as a two-arm design, not a waiver:**
   enter with the benchmark-standard ego-status input AND report the paired
   vision-pure arm as our scientific contribution (no leader publishes that
   pair; F4/F5 make it novel). This keeps the doctrine intact (the vision-pure
   arm remains the programme's claim-bearing arm) while making the entry
   comparable. Blocked on PI; blocks nothing else.

## Searches that came up empty
- DriveFuture params/inputs and GTRS LiDAR usage: not resolvable from this
  pass's sources — marked UNVERIFIED in F3 rather than guessed.
- Any navhard-two-stage number for Drive-JEPA (our reference): absent from the
  paper by two probes (regex over full text; table scan). If we benchmark
  against Drive-JEPA, it must be on the papers' protocol, or we reproduce it
  ourselves on navhard.

## Evidence table
| # | claim | class | source |
|---|---|---|---|
| F1 | 89.0 = perception-free v1; full 93.7 v1 / 87.8 v2-navtest | PUBLISHED | lib `2601.22032` (abstract, Tab. 1–3) |
| F2 | two v2 protocols ~30 pts apart; bug-fix note | PUBLISHED | libs `2601.22032` Tab. 3, `2601.05083` Tab. 3, `2606.07170` Tab. 2 |
| F3 | table rows as printed | PUBLISHED (UNVERIFIED where marked) | same three libs |
| F4 | DrivoR + Drive-JEPA ego-status mechanisms | PUBLISHED | libs `2601.05083`, `2601.22032` |
| F5 | DINOv2 76.1 / V-JEPA2 86.1 / ours 89.0 | PUBLISHED | lib `2601.22032` Tab. 6 |
| F6 | registers 0.6M / 250× / near-lossless; warmup-val provenance | PUBLISHED | lib `2601.05083` |
