# BRIEF — refcv6 standard test battery + NavSim suite (Master Mind → EvalFlyWheel)

**Requested by the PI, 2026-09-23 (verbatim):** *"contact the eval fly wheel agent to prepare the
required standard tests inclduing the navsimn suite for refcv6"*.
**Delivered to** the session "Start TanitAD_EvalFlyWheel: NavSim/nuScenes + leaderboard" by
cross-session message on 2026-09-23 ~22:30 Berlin — **queued**: that session was offline (Remote
Control, another machine) and receives it when it reconnects. This file is the durable copy.
**Reply to:** the Master Mind session "TanitAD project kickoff (fork 8)", and bank `PLAN.md` beside
this file.

## Operating rules (binding)

1. STAGE, NEVER PUSH. `git add` every deliverable into the working tree when you finish.
   Do NOT `git commit`, do NOT `git push`, do NOT switch branches. Never leave work that
   took real effort living ONLY on a pod or ONLY in a worktree — copy it into the repo and
   stage it. If you believe something should not be staged, say why in your report.
2. END WITH A DELIVERABLE MANIFEST — a table of every artifact you produced and WHERE it
   lives: `repo:<path>` / `<pod>:<path>` / `worktree:<name>`. Mark anything that exists in
   only ONE place. This table is not optional.
3. ESCALATE INTEGRATION. If your work needs merging, wiring in, or a decision, say so in
   your report's headline. Do NOT write "please merge this" into a README.
4. QUOTE ONLY PRIMARY SOURCES. Model facts come from `Project Steering/MODEL_REGISTRY.md`
   or raw eval JSON — never from a summary, changelog, or weekly report. If a doc conflicts
   with the registry, the registry wins; report the conflict.
5. FAIL LOUD, REPORT HONESTLY. If something is uncertain, mark it UNVERIFIED rather than
   guessing. A flagged gap is far better than a confident wrong answer. If you could not do
   part of the task, say so plainly — do not quietly narrow the scope.

## The run

* **refcv6-r101-s0** trains on Thor (`tanitad-thor-wifi`) since **2026-09-23 20:42 Berlin**:
  commit `284393c`, then `287d72e` from step 500 (logging-only switch), branch
  `agent/arch-inf-20260803`; batch 16, **50,400 steps** (the pre-registered `full`), ~6.5 s/step,
  finish ≈ **2026-09-27** afternoon.
* Run dir `/home/nvidia/refcv6_run/runs/refcv6-r101-s0`; `ckpt.pt` is **overwritten every 500
  steps**; `config.json` carries the full argv and seams.
* The arm: resnet101.a1_in1k at **416 × 1024, cylindrical front-wide 120°**, 8 window rows of 3-frame
  D-015 stacks · nav from the v7 token (follow / left / right) · ego history (GRU over 8 steps;
  measured v0 at t0 is admissible, ruling 2026-09-02) · v6 max-speed input (4-way one-hot, sidecar) ·
  tactical decoder v6 (22-token goals + lat / lon) · SAM3 BEV map + agents + 3-D boxes supervision ·
  BEV coupling · DDIM diffusion over 117 v0-conditioned anchors, 6 s horizon.
* Launch line and levers: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-23-refcv6-fixes/LAUNCH_READINESS_FIXES.md`
  §5–§11 and `code/run_refcv6.sh`. Live training dashboard: the "refcv6 Training Watch" artifact
  (built by `taniteval/tools/training_watch/build_watch_refcv6.py`).

## Asks, in priority order

1. **The standard held-out eval (TanitEval)** — the four metric families (LONGITUDINAL, LATERAL,
   TACTICAL, STRATEGIC) + ADE on the 139-clip eval split at 416 × 1024 (Thor:
   `/home/nvidia/data/refcv6-b1-416x1024-eval139`; the dev box may hold the §10.6 copy); tier stamps
   (T0 vs T1 = self-action open loop, per the open-vs-closed-loop ruling); the paired
   episode-cluster bootstrap against refcv5-v2 and refcv4b (registry rows) AND controls that must
   read known values (hold-action, constant velocity, STOP); bars pre-registered before any score.
   STRATEGIC is not applicable to this arm (strategic layer OFF, SPEC_REFCV6_V2) — say so per family.
2. **The NavSim suite** — extend the E2 bridge
   (`FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-refcv4b-bridge/`) to refcv6's inputs:
   416 × 1024 cylindrical frames built from NavSim's cameras (state the projection), the
   nav-command mapping, the ego history, and the max-speed input (nuPlan map speed limits are
   admissible, PI ruling 2026-09-19); keep the control set (CV, STOP — warmup rewards stopping —
   echo, frames-blind, nav-withheld); add NavSim v1 **navtest PDMS** (the community headline;
   DiffusionDrive reports 88.1) if its data can be provisioned — name the storage and compute;
   anything metered (pod, HF) is a PI decision.
3. **The other charter benchmarks** (nuScenes open-loop L2 / collision and peers): feasibility for a
   front-camera-only arm, and cost.
4. **Which checkpoint steps you need** (e.g. 10k / 25k / 50,400): the Master Mind snapshots them,
   md5-verified, to the dev box, because `ckpt.pt` is overwritten.

## Constraints

⛔ Never eval on Thor while it trains (no GPU / RAM load on a training box). Use the dev box
RTX 4060 (the PI's desktop: gate at 4,300 MiB GPU used and ≥ 8 GB free host RAM, one GPU job at a
time) or ask the PI for a pod. Four families, never ADE alone; every estimator named; no interval
without its estimator; every NavSim warmup row read against STOP, not only CV.
