# TanitAD programme report — 2026-09-13 13:45 (Europe/Berlin)

**Master Mind.** Requested by the PI: *"a general report where we are and our possible next steps and options to
move forward with TanitAD"*. Every number carries its evidence class; driving numbers are **T1 = self-action OPEN
loop** (PI ruling 2026-09-02); perception numbers carry no driving tier.

---

## 0. The short version

1. ⛔ **22 days to the first final evaluation (5 Oct 2026, Mission Plan P7) — and Phase 0 is not done.** Two
   defining gaps: **no trained driving arm beats doing nothing** at T1, and **no closed-loop result exists** for the
   current model line.
2. **Driving (REF-C line, ~108 M params):** refcv4b **ties** the do-nothing controls; refcv5-v2 **fails both bars
   and is separably worse than refcv4b**. The do-nothing controls win speed, target-speed and along-track accuracy. The one clear gain is lateral
   (curvature error 0.51× the straight-line floor). *(MEASURED, `Benchmarks & Eval/LEADERBOARD.md` §1e.)*
3. **refcv6 — every DiffusionDrive V1/V2 piece + max-speed input + tactical goal — is built, gated, and stopped**
   at step 250 by the 2026-09-11 pivot. ~47 h per arm on Thor. *(MEASURED rate 4.09 s/step.)*
4. **Perception pivot (since 09-11):** the Qwen-Drive teacher works on our data after an input fix; its **map is
   real but coarse**, its **occupancy is weak**; SAM3 extracts **lane lines** but over-calls crosswalks; LiDAR BEV
   ground truth is built for 134 clips; a BEV transformer on the **frozen** trunk **fails its bars** (interim).
   ⇒ **our trunk carries little spatial scene structure — the planner is not grounded in the environment.**
5. **Data:** we train on ~26 h (4,729 clips) of PhysicalAI's ~1,701 h *(MEASURED, registry)*. nuScenes (5.5 h,
   real maps) is one download-OK away.

---

## 1. Status by stream

| stream | state | best measured result | evidence |
|---|---|---|---|
| **Driving model** (P1) | refcv4b / refcv5-v2 evaluated; refcv6 staged, stopped | refcv5-v2 `os − ha0_ext` **+0.0205 [+0.0043, +0.0390]** m ADE (wrong way) · speed MAE 0.292 vs hold-action **0.254** m/s · curvature **0.512×** floor · anchor selection **61.7×** chance · route acc 0.771 (oracle nav) | LEADERBOARD §1e |
| **RL on the planner** (P1/P4) | ≥GT truncation (DiffusionDrive V2) reachable | **2.24× more reward-efficient** than an equally small update, p = 0.0079 (T0, training-side); driving effect **untested** | commit a4ef511 |
| **Longitudinal input** | max-speed ceiling built, 100 % coverage, floor 20 km/h | arm pre-registered (`H-E16-1`), **not run** | PREREG_E16 |
| **Qwen-Drive teacher** (P2) | v2 packing (virtual nuPlan rig) | detection recall **0.18 → 0.40**, precision 0.25 → 0.43 · map edges on real curbs **51 %** (true map 75 %) · occupancy covers **53 %** of obstacles (in-distribution 96 %) | review package §6, §12 |
| **SAM3 road paint** (P2) | runs on Thor, lifted to BEV on LiDAR ground | vs a true map: P **0.47** / R **0.60** (Qwen in-distribution 0.99/0.99) · our data: **no admissible number** (reference failed its controls) · crosswalks over-segmented | review package §13 |
| **LiDAR BEV** (P2/P1) | corpus GT built (139-clip eval join, 5 quarantined); frozen-trunk head panel run | head AP **0.414** vs prior 0.377 / shuffled 0.380 / pixels 0.385 — **fails all 4 bars**; overfits by step 1,000; levers L4 (more clips) and L1 (stride-16) running | `…/2026-09-13-bev-lidar-corpus-and-head/RESULT.md` ⚠️ INHERITED, agent still running |
| **Eval** (P7) | T1 four-family panel, paired episode bootstrap, do-nothing bars; LiDAR-calibrated map metrics (new) | closed loop **not provisioned** for the current line | LEADERBOARD; review §12 |
| **Deploy** (P6) | Thor runs Qwen-Drive 4B perception | ~4.1 s/frame (teacher, offline) · our 108 M model not profiled this cycle | review §6 |
| **Reconstruction / TanitScena / DataSetCreator** (P3/P5/P8) | no active work this cycle | — | — |

---

## 2. Position against the Mission Plan

**Phase 0** = *running architecture with the key hypotheses (4B hierarchy, world-model approach), real data, open AND
closed loop, single front camera + reasoning + tactical MoE.*

| Phase-0 element | state |
|---|---|
| hierarchical architecture on real data | ✅ built (REF-C line; tactical 22-token vocab; strategic route head) |
| open-loop evaluation with honest controls | ✅ exists — ⛔ **the bar is not cleared** |
| closed loop | ⛔ **missing** for the current line |
| single front camera | ✅ (vision-only inference) |
| reasoning / tactical MoE steering extra sensors | ⚠️ tactical tokens yes; reasoning not integrated |

**The edges** (grade: A proven with controls · B supported · C pieces only · D not demonstrated):

| edge | grade | why |
|---|---|---|
| planning quality | **D** | ties / loses to hold-action at T1; no closed loop |
| inference efficiency | **C** | Thor runs pieces; no end-to-end latency for the driving stack this cycle |
| data efficiency (1000× less) | **D** | not measured against a baseline trained on more data |
| safety by design | **D** | no safety metric in the loop yet |

---

## 3. Options to move forward

| option | what it is | cost | what it buys | main risk |
|---|---|---|---|---|
| **A — relaunch refcv6** | the DiffusionDrive-complete planner, arms V0 → V0b → D | ~47 h/arm on Thor (~6 days for 3) | the first arm with every DiffusionDrive piece, read against the do-nothing bars | the longitudinal failure is not addressed by any refcv6 piece; the trunk stays ungrounded |
| **B — longitudinal lever first** | run the pre-registered max-speed arm; RL ≥GT fine-tune of refcv4b | ~2–3 days Thor | a direct test of the **largest measured gap** (the do-nothing controls win speed, target-speed and along-track accuracy) | the registered residual (slow ego ⇒ low limit) must be shown not to be the win |
| **C — grounded perception (the 09-11 pivot)** | LiDAR-supervised BEV + agent/map queries trained **with** the planner (DiffusionDrive-style spatial cross-attention), maps from Qwen + LiDAR (+ SAM3 lines) | 1–2 weeks to a trained integrated arm | the missing environment grounding — the frozen-trunk result says it must be **trained in**, not read out | not ready for 5 Oct; token grid loses ~2/3 of localisation (WP-A) |
| **D — closed-loop harness for 5 Oct** | AlpaSim / NuRec closed loop on Thor for the best current arm | ~3–5 days engineering | the Goal-1 number the evaluation needs; honest closed-loop failure modes | sim-to-real gap; shares Thor with training |
| **E — more data** | scale beyond 26 h (PhysicalAI has ~1,701 h); Qwen map labels; nuScenes maps | bandwidth (~12.6 MB/s dev-box link, MEASURED) + Thor time | the most reliable quality lever for a 108 M model | tension with the "1000× less data" edge; label pipeline time |

### ⭐ Recommendation for the 22 days (single Thor, dev-box RTX 4060)

1. **Now → 20 Sep:** **B on Thor** (shortest path to beating the do-nothing bar where it fails worst) while the dev box
   builds **D's** closed-loop harness plumbing; the BEV stream (C) continues on the dev box with its data lever.
2. **20 → 27 Sep:** relaunch **refcv6 V0 with the winning longitudinal lever folded in** (A+B); fold the BEV head in only
   if a lever makes it clear its bars.
3. **27 Sep → 5 Oct:** freeze the best arm; closed-loop run + T1 four-family panel + videos; write the evaluation.
4. **C (grounded perception) becomes the October programme**, with Qwen-map + LiDAR (+ SAM3-line) labels on the
   139-clip slice as its first data.

---

## 4. Decisions needed from Sayed (each with its default)

| # | decision | default if silent |
|---|---|---|
| 1 | Do the Qwen-Drive v2 videos pass your validation? (queue item 14a) | no corpus-scale augmentation |
| 2 | Augmentation scope — 139-clip slice (~9.9 GB + ~6.3 h Thor) or parity (~170 GB + ~108 h) (14b) | 139-clip slice after (1) |
| 3 | Download nuScenes metadata **0.46 GB** (+ CAN bus 0.78 GB) from the official bucket | not downloaded |
| 4 | Which track for the next 22 days (A–E above) | the 09-11 pivot stands: perception only, Thor not used for planner training |
| 5 | Must 5 Oct include a closed-loop result? | T1 open-loop panel + videos only |
| 6 | Older open items: 15-token strategic vocabulary (corpus gap) · `Sayood/tanitad-refc-v3` is public · g_str ~40k retrain · 50-frame traffic-light spot-check · WP-C authorisation | unchanged (see `PI_DECISION_QUEUE.md`) |

---

## 5. What went wrong this cycle (honestly)

* The Qwen-Drive validation video you rejected was **my renderer** (decoded from a guessed schema) plus an input
  outside the model's camera geometry — fixed, measured, logged as retraction classes C and D.
* Caught before publishing: a confounded control (A2), a "has / has not" map claim a count refuted, 4 plaintext clip
  ids in a staged artifact (redacted), a sparse-LiDAR ground failure and a pooled average hiding the urban clips.
* SAM3 on our data: my LiDAR paint reference **failed its own controls** — no SAM3-vs-Qwen claim on our data.
* 11 Sep: refcv5-v2 landed as a FAIL, and a column I had called "refcv4b" was the hold-action control (retracted).

## 6. In flight right now

* **Dev box RTX 4060:** the LiDAR-BEV agent (corpus GT done; frozen-trunk panel failed its bars; data lever L4 running).
* **Thor:** idle (Qwen-Drive and SAM3 runs finished) — available for the track you choose.
