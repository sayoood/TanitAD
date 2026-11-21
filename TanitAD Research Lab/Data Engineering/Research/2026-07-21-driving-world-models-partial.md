# PARTIAL / INTERRUPTED — Non-NVIDIA driving world models & photorealistic synthetic data

**Status: PARTIAL. Survey aborted before any source was fetched.**
Date: 2026-07-21. Author: research subagent (survey arm).
Interrupted by: weekly API cap (resets **2026-07-26, 00:00 Berlin**).

---

## 0. Read this first — what this file is and is NOT

This file contains **NO survey findings**. Not one model was fetched, not one number was
read from a primary source. The run died during setup.

It is banked anyway because the *setup* is reusable and the *dead ends* are real: a next
agent picking this up after Jul 26 should not re-derive the scope, re-discover the process
failure, or re-spawn the fan-out that burned the budget.

**Do not quote anything in this file as a model fact.** There are no model facts here.
Sections 1 and 2 are process and constraint records. Section 3 is a worklist of things
*not* done. Section 4 is the question schema. Per `CLAUDE.md`, prose that looks like a
finding but is not sourced is exactly the failure mode this project has been burned by —
so this document deliberately carries no model claims at all.

---

## 1. Established

Nothing MEASURED (ours). Nothing PUBLISHED (cited). Everything below is process or
instruction, tagged as such.

| # | Item | Tag | Note |
|---|------|-----|------|
| 1 | **Waymax and Waymo Open Dataset = `refuse`.** Their terms are held to follow the trained *weights* into vehicle operation, not merely the data. Excluded from any recommendation. | **INSTRUCTION (coordinator)** — *not independently verified* | Next agent: verify against the actual Waymo Open Dataset License Agreement and the Waymax licence text, quote the operative clause, and record it. Until then this is a directive, not a citation. |
| 2 | Fan-out (parallel subagents) is what exhausted the API budget. Next pass must be **serial**. | PROCESS (ours) | Binding for the resumed run. |
| 3 | Spawning a second `sonnet` subagent failed outright: *"claude-sonnet-5[1m] is temporarily unavailable, so auto mode cannot determine the safety of Agent right now."* | PROCESS (ours) | Sonnet fan-out was unavailable at the time of the run; do not build a plan that depends on it. |
| 4 | One subagent (3DGS / neural-reconstruction cluster) was launched and **never reported back** — no completion notification arrived before the cap. Its findings, if any, were not harvested. | PROCESS (ours) | Its transcript lived in the session scratch/tasks dir, which is **ephemeral** — assume it is gone. Do not plan around recovering it. |

### Working assumption carried forward (NOT yet verified)
- nuScenes and Waymo are non-commercial datasets, and nearly every AV video generator in
  this space is trained and/or evaluated on one of them — so licence contamination is
  expected to flow from dataset → model → outputs. **UNVERIFIED as stated.** This is the
  hypothesis the licence half of the survey exists to test, not a result.

---

## 2. Ruled out

| Ruled out | Reason | Tag |
|-----------|--------|-----|
| Waymax / Waymo Open Dataset | Licence gate — terms follow the weights into vehicle operation (§1.1) | INSTRUCTION (coordinator) |
| Parallel subagent fan-out for this survey | Exhausted the weekly API cap; also hit a model-availability failure | PROCESS (ours) |

**Nothing was ruled out on technical evidence.** No technical evidence was gathered.
In particular: no model was eliminated for lack of action-conditioning, lack of public
weights, or licence — those checks were never run.

---

## 3. Not yet reached — the full worklist

All of it. Listed so the resumed run can go straight to work. Priority reflects how much
each one would move the decision.

### Cluster A — action-conditioned generative world models (highest value)
| Model | Org | Priority | The question it must answer |
|-------|-----|----------|------------------------------|
| GAIA-1 | Wayve | High | Action-conditioning details; availability |
| GAIA-2 | Wayve | **Highest** | Is it open at all? Any API? Explicitly action-conditioned — get licence + availability |
| Vista | OpenDriveLab | **Highest** | Action-controllability claims vs reality; short-horizon (~25 frame) limit; weights on HF |
| Epona | — | High | Autoregressive diffusion WM, 2025 |
| DrivingGPT | — | Medium | |
| Doe-1 | — | Medium | Closed-loop large world model |
| GenAD / OpenDV-2K | OpenDriveLab | Medium | |
| Copilot4D | Waabi | Medium | LiDAR world model, ICLR 2024 — action-conditioned? |
| World4Drive | — | Medium | |

### Cluster B — layout/BEV-conditioned multi-view video generators
| Model | Priority | Note |
|-------|----------|------|
| MagicDrive / -V2 / 3D / DiT | High | Multi-view + BEV-conditioned; the DiT variant is the long-clip one |
| OpenDWM (SenseTime) | High | Open toolkit — which base models, what conditioning |
| Panacea / Panacea+ | Medium | |
| DiVE, UniMLVG, MiLA | Medium | Multi-view / long-horizon |
| DriveDreamer / -2 / -4D | Medium | |
| ReconDreamer / ++ | Medium | |
| Delphi, InfinityDrive, DrivingSphere | Medium | Long-video claims |

### Cluster C — neural reconstruction / 3DGS replay from real logs
Launched as a subagent; **never returned**. Fully unaddressed.
UniSim (Waabi) · NeuRAD (Zenseact) · OmniRe · Street Gaussians · DrivingGaussian ·
HUGSIM · R3D2 · plus any 2025–2026 successors (SplatAD, StreetCrafter, FreeSim, DriveX,
STORM, DeSiRe-GS — names unconfirmed, search fresh).

**The single most valuable unfetched number in the whole survey:** published
lateral-shift ablations — PSNR/FID/LPIPS as the virtual camera moves 0m / 1m / 2m / 3m
off the recorded ego path. UniSim and NeuRAD are both believed to contain lane-shift
experiments. **UNVERIFIED — nobody has looked.** This number decides whether the
reconstruct-and-replay line can serve true counterfactual ego trajectories at all.

### Cluster D — 2026 work
Not searched. Run fresh queries for 2026 arXiv: *action-conditioned driving video
generation*, *long-horizon / minute-level driving video*, *closed-loop 3DGS driving sim*.

### Special-focus questions — all three unanswered
1. Which models produce a **counterfactual rollout** (same past, different ego action) at
   training-useful fidelity? Ranked, sceptically.
2. **Longest coherent clip** any of them produces, and the evidence for temporal drift
   beyond it.
3. Which have **permissive licences** covering commercial use of *both* model and outputs.

---

## 4. Question schema to reuse (the actual reusable asset)

Per model, answer exactly these. Every number carries a URL; tag **PUBLISHED (cited)** or
**UNVERIFIED**.

- **(a)** Release date · arXiv URL · official repo URL
- **(b)** **Are weights actually public?** — `WEIGHTS PUBLIC` / `CODE ONLY` /
  `PAPER ONLY` / `UNVERIFIED`. Check the repo's model-zoo/checkpoint section and HF, not
  the abstract. Many AV world models are paper-only; this axis is decisive.
- **(c)** **Consumes**: text? reference image/frames? HD map or BEV layout render? 3D
  boxes? ego trajectory / action sequence? camera intrinsics+extrinsics?
- **(d)** **Emits**: resolution · fps · max length (frames *and* seconds) · single- or
  multi-view (how many cameras)
- **(e)** **Action-conditioning / counterfactuals** — the key question. Quote the paper's
  own wording. Then classify:
  - **(i) true action input** — the dynamics model consumes an ego action/trajectory, so
    the past can be held fixed and the action varied; or
  - **(ii) layout condition** — a BEV/HD-map render or 3D-box sequence that merely encodes
    where the ego *already* went. Changing the ego pose requires re-rendering a consistent
    layout, which presupposes the answer. **This is not a dynamics model.**
  Note whether the paper shows explicit steering/speed counterfactual figures.
- **(f)** **Licence**: exact name + URL of the actual `LICENSE` file (not the README
  badge). Commercial use? Outputs redistributable? Base-model licence inheritance (SVD,
  CogVideoX, Wan, Open-Sora, SD each drag their own) and dataset contamination.
- **(g)** **Hardware + throughput**: VRAM, inference GPU, training GPU-days, wall-clock
  seconds per clip. Real published numbers only.
- **(h)** FVD / FID **with dataset and exact protocol** — nuScenes FVD is not comparable
  across papers; note protocol differences where the paper does.

---

## 5. Traps for the next agent

- **"Controllable" ≠ action-conditioned.** Most such claims are about the layout
  condition. Apply the (e)(i)/(e)(ii) split to every single model.
- **Weights-public vs paper-only** is the axis that decides usability. Do not gloss it.
- **Open the LICENSE file.** A GitHub repo implies nothing about commercial rights.
- **Licence is a gate, not a footnote.** Waymax/Waymo Open are `refuse` (§1.1).
- Primary sources only: arXiv abs/HTML, official repo READMEs, HF model cards. No blog
  summaries for numbers — per `CLAUDE.md`, a number copied from prose is how this project
  propagated three errors for days.
- **Go serial.** Fan-out killed this run.

---

## 6. Pickup instructions

1. Wait for the cap reset — **2026-07-26, 00:00 Berlin**.
2. Work **serially**, one cluster at a time, in priority order:
   **Cluster A (GAIA-2, Vista first) → Cluster C lateral-shift numbers → Cluster B → Cluster D.**
3. Append findings to this file and retitle it from PARTIAL when a cluster completes, or
   supersede it with a dated successor and link back.
4. Stage, never push.

**Deliverable manifest for this run:** this file only —
`TanitAD Research Hub/Data Engineering/Research/2026-07-21-driving-world-models-partial.md`
(repo working tree, staged, uncommitted). No other artifact was produced anywhere. No pod
was touched.
