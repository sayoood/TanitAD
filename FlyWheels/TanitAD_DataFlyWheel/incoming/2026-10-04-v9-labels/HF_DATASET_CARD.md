---
license: other
tags: [autonomous-driving, labels, tactical-planning, physicalai-av, tanitad]
---

# TanitAD refcv8 — v9 label release (PRIVATE)

The corrected **label layer** of the TanitAD refcv training corpus (PhysicalAI-AV front-camera clips; 4,369 train +
139 held-out eval clips). The images and poses are unchanged; what changes is how the corpus is LABELLED and which
labels are used. Research use only (the source corpus's research licence applies to these derived labels).

**Why it exists.** The refcv7 audit (2026-10-04) found the v8 labels were applied wrongly in time: one record per clip,
anchored at a single instant, so tactical labels existed on only **23.22 %** of training windows, the nav input was
wrong on **74.3 %** of left/right windows, and the max-speed input was a future-speed oracle. The PI required, for
refcv8: tactical goals + actions **per camera frame** over the next 2–8 s with constraints (distance, time); a
**per-frame** nav command; and a **route checkpoint** input. This release delivers them.

## What every frame now carries (one row per cached 10 Hz frame, key `(sid, k)` on the trainer's own clock)
* **Lateral action** over [NOW+2 s, NOW+8 s]: `LANE_KEEP` / `TURN_L` / `TURN_R` (turn = |Δyaw| ≥ 30° on a sustained-yaw
  segment; variant a = junction turns, default; variant b = any ≥ 30°), with turn start/end time, distance, angle, radius.
  `NUDGE` is not emitted (it labelled road curves). Lane change could not be measured reliably → never-positive.
* **Longitudinal action** over the same band: `HOLD`, `CREEP`, `STOP`, `FOLLOW`, `DECELERATE`, `ACCELERATE` (±1.5 m/s),
  `KEEP`, with target speed + time/distance to reach it, stop distance/time/point, lead gap and time gap.
* **Goals** (the 22-token v7 family) per frame: geometry goals dense; VLM goals re-timed to the VLM's own window;
  negatives inside the release (no sidecar, no module state); red-light reaction propagated over its stop episode;
  `SPEED_BAND` replaced by a continuous speed target.
* **Nav** per frame from a whole-recording turn table: `TURN_L/R` while a turn is in progress or within
  max(30 m, 6 s × current speed), else `FOLLOW`; with distance to the turn start/end, angle and validity.
* **Route checkpoint (an INPUT)**: the next route point in the vehicle frame from the smoothed driven path — RC-A at
  30 / 50 / 80 m and RC-B (next turn end). Recommended: **RC-A50 with training noise** (σ along 2 m, lateral 0.75 m) and
  ≥ 0.3 dropout.
* Speed-limit proxies N2 / N3 from PAST ego speed only.

## Validation (MEASURED; full report `docs/RESULT.md`)
| check | train | eval139 |
|---|---|---|
| frames with tactical labels (refcv7 v8: 23.22 %) | **95.09 %** | 94.57 % (95.58 % on tactically scored clips) |
| TURN side recall / precision vs an independent derivation | 0.980 / 0.988 | 0.943 / 0.966 (0.992 / 0.982 excl. 3 reversing clips) |
| STOP recall / precision | 0.991 / 0.994 | 0.999 / 0.998 |
| realised announced turns get the matching nav token | 99.55 % | 99.34 % |
| route checkpoint RC-A50 noised, future-speed leak vs a road-level route (bar ≤ 0.01) | −0.035 (PASS) | — |

Reported as FAILED as registered: eval coverage (0.4 pp short), eval TURN recall before the reversing mask, lane
change (never-positive), nav distance ±2 m (0.76 / 0.80), the 6-s nav bar (the token announces approaches early by
design), "every turn window labelled" (881 / 113,946 unlabelled).

## Files
* `release/v9_labels_train.npz` (md5 `f63ece41…`), `release/v9_labels_eval139.npz` (md5 `6b5c7f20…`) + manifests.
* `inputs/` — the lead-vehicle and lane-offset tables the builder read.
* `docs/` — SPEC (definitions), RESULT (validation), V9_SCHEMA (all 132 fields), INTEGRATION (the trainer contract),
  the two pre-registered addenda.
* `corpus/refcv8_corpus_manifest.json` — the 693-frame ego-as-agent-box MASK (applied) and an 8-clip DROP list
  (recording partners of eval clips + duplicate videos) **pending the PI's decision 8 (default: keep all clips)**;
  `corpus/refcv8_join_label_defects.json` — the agent-label hygiene masks (ego boxes, track-id switches).
* `MD5SUMS.txt`.

Reader: `stack/tanitad/data/v9_labels.py` in github.com/sayoood/TanitAD, branch `agent/arch-inf-20260803`, commit
`c367e7b` (builder `stack/scripts/build_v9_labels.py`). Clip identities are sha12 / stable ids only.

**Pending PI decisions:** 8 (drop the 8 clips?) and 11 (road geometry admissible inside the route input? the RC-A50
recommendation rests on "yes"). NavSim leaderboard-legal use: no route checkpoint and no turn distance (bare command).
