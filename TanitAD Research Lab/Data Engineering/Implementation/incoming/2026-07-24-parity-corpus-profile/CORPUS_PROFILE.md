# Parity WM-training corpus — exact profile (data-enlargement decision input)

**Date:** 2026-07-24 · **Discipline:** Data Engineering · **Slug:** `2026-07-24-parity-corpus-profile`
**Corpus:** `physicalai-train-e438721ae894` (skip-hash `f09e44db`) — the canonical
sub-300M WM training set. **READ-ONLY profile; the parity set was not modified.**

**Evidence class:** every headline number is **MEASURED** by loading each cached
episode's `poses[T,4]` and the pre-computed per-timestep `maneuvers[T]` on
`tanitad-pod3` (CPU-only, frames left disk-backed via mmap, ~5 min single process).
Artifacts: `corpus_profile.json` (machine-readable) + `corpus_profile_probe.py`
(this dir) + raw at `tanitad-pod3:/workspace/tmp/corpus_profile.json`. Cross-checks
against `Project Steering/MODEL_REGISTRY.md` and code are cited inline.

> **One-line answer for Sayed:** the parity corpus is **13.13 driving hours** /
> 472,627 frames / 2,376 clips of **~20 s each**, and its scenario mix is
> **kinematically 60% lane-keep, ~14% turns, ~26% accel/brake**, **~46% highway /
> ~46% city / ~8% stopped**. It is **turn- and stop-*event*-poor at the clip level**
> (57% of clips contain no turn; 62% no junction-scale turn) and **has zero semantic
> labels** (lights, roundabouts, pedestrians, merges are invisible). Enlargement
> should buy **junction turns, low-speed urban/stop events, and semantic scenarios**.

---

## 1. Exact size — confirms the ~13 h estimate ✅ (MEASURED)

| Quantity | Value | Source |
|---|---|---|
| Usable clips (after skip-hash `f09e44db`) | **2,376** | `ep_*.pt` count; 24 `skip_*` present |
| Total frames @ 10 Hz | **472,627** | sum of per-clip `T` |
| **Total driving hours** | **13.13 h** (787.7 min) | 472,627 / 10 / 3600 |
| Clip length — mean / median | **198.9 / 199 frames** = 19.89 / 19.9 s | |
| Clip length — min / max | 188 / 205 frames = 18.8 / 20.5 s | very uniform ~20 s clips |

The nominal "2376 × 20 s = 13.2 h" is essentially exact here because clips are
tightly clustered at ~19.9 s. **Real hours = 13.13 h** (the correct figure to quote).

## 2. Epochs — 30k steps ≈ **4.73 epochs** (MEASURED frames → derived)

- **Window count per clip** = `max(0, T − window − max_horizon)` (`refb_train.py:115`,
  the shared `build_window_index`).
- **`max_horizon` is 20, not 16.** The flagship dataset windowing horizon is
  `plan.max_horizon = max(goal_h=20, maneuver_h=20, tac_h=16, op_h=4)`, and
  `goal_h = max(waypoint_horizons = [5,10,15,20]) = 20`
  (`tanitad/train/flagship_losses.py:horizon_plan`). The "16" in the brief is the
  tactical **predictor's** farthest horizon; the **dataset** must supply the 2 s
  (20-step) waypoint/maneuver target, so it windows at 20.
- **Unique windows = 406,099.** Since every clip ≥ 188 frames, no clip clamps, so
  `472,627 − 2,376×(8+20) = 406,099` — which **exactly reproduces the registry's
  run-log count** ("2,376 eps / 406,099 windows", MODEL_REGISTRY §0.1/§4). This is an
  independent confirmation that the frame count is exact.
- **Presentations** = 30,000 steps × **effective batch 64** (batch 16 × accum 4; one
  optimizer step consumes `accum` micro-batches, `train_flagship4b.py:444`) =
  **1,920,000**.
- **Epochs = 1,920,000 / 406,099 = 4.73.**
  *(If windowing had used horizon 16 → 415,603 windows → 4.62 epochs. The operative
  figure is 4.73.)*

So the deployed flagship saw the corpus **fewer than 5 times**. Enlargement trades
directly against this: e.g. 2× the data at fixed 30k steps → ~2.4 epochs.

## 3. Maneuver histogram — the headline scenario distribution (MEASURED)

Per-timestep, over all **425,107 valid** (non-sentinel) timesteps. **Headline = the
STORED v1 kinematic labeler** (`refb_labels.maneuver_labels`, 2 s horizon) — the exact
per-timestep label persisted in the parity cache. 47,520 tail timesteps (last 20/clip)
are sentinel `-1` and excluded.

| Class | Count | **Fraction** |
|---|---|---|
| 0 lane_keep | 253,542 | **59.64%** |
| 1 turn_left | 29,150 | **6.86%** |
| 2 turn_right | 31,418 | **7.39%** |
| 3 accelerate | 56,231 | **13.23%** |
| 4 brake_stop | 54,766 | **12.88%** |

Grouped: **lane-keep 59.6%** · **lateral (turns L+R) 14.25%** · **longitudinal
(accel+brake) 26.11%**.

**Labeler-sensitivity cross-check (v2 curvature-gated recompute, same 2 s horizon):**
lane_keep **61.18%**, turn_left 5.90%, turn_right 5.66% (**turns 11.56%**, ~2.7 pp
fewer — gentle highway curves reclassified to lane-keep), accelerate 13.96%,
brake_stop 13.29%. The trainer's per-window maneuver label uses this v2 gate, so the
model effectively sees **~11.6% turns**, even fewer than the stored 14.25%.

*Corroborates prior work* (`incoming/2026-07-18-curve-rebalance`, a 500-clip subset,
different `|net-yaw@2 s|` taxonomy): PhysicalAI urban measured 56.0% straight / 20.6%
"sharp". Consistent order of magnitude on the full 2,376 set.

## 4. Strategic / nav histogram (MEASURED)

**v2.1 adaptive-horizon route** (`route_from_future_v21`, the current best labeler;
sampled every 1 s = 47,520 samples; **76.3% judgeable**):

| Route | Fraction (all samples) |
|---|---|
| straight / road-following | **52.79%** |
| left | **12.03%** |
| right | **11.47%** |
| unknown (masked, unjudgeable) | 23.71% |

Corroborates the registry's REF-C label-coverage line (left 0.121 / straight 0.5645 /
right 0.115 / unknown 0.1995); my unknown is a touch higher because I sample whole
clips including the low-future tails.

**Legacy v1 nav** (`nav_command`, 15–25 s future) for reference: over its judgeable
windows — follow **76.4%**, left 11.9%, right 11.8% — **but coverage is only 24.6%**.
v1 needs 15 s of future and clips are ~20 s, so only the first ~5 s of each clip is
judgeable (the documented "D1 coverage collapse"). **Use v2.1, not v1, for strategy on
this corpus.**

## 5. Speed distribution (MEASURED, 472,627 timesteps, m/s)

| | p10 | p25 | p50 | p75 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| v (m/s) | 1.81 | 5.59 | **11.04** | 19.33 | 27.70 | 31.73 | 35.96 | 46.65 |

Mean **13.04 m/s (≈ 47 km/h)**; max 46.6 m/s (≈ 168 km/h). Regime split:

| Regime | Fraction |
|---|---|
| **stopped** (< 1 m/s) | **7.76%** |
| **city** (1–12 m/s) | **45.92%** |
| **highway** (> 12 m/s) | **46.32%** |

Speed coverage is broad and well-balanced across city and highway — this axis is
**not** a scarcity problem (unlike turns/stops). Standstill is the thin tail (7.8%).

## 6. Turn / stop / junction rarity — per-clip (MEASURED)

| Event (≥ 1 occurrence in the clip) | Clips | Fraction |
|---|---|---|
| Any turn (v1 turn_left/right) | 1,012 / 2,376 | **42.6%** |
| — turn_left | 659 | 27.7% |
| — turn_right | 706 | 29.7% |
| Any brake_stop | 1,366 | **57.5%** |
| Any accelerate | 1,410 | 59.3% |
| **v2.1 junction-scale turn** (tight+transient) | 895 | **37.7%** |
| Net heading change > 45° | 594 | **25.0%** |
| Net heading change > 90° (≈ sharp/roundabout) | 248 | **10.4%** |

Per-clip net heading change: median **10.8°**, mean 29.7°, p90 90.2°, max 178.9°.

**Reading:** **57% of clips contain no turn at all**, and **62% contain no
junction-scale turn**. Turns are not just a minority of *timesteps* (14%) — they are
absent from the majority of *clips*. Sharp/roundabout geometry (>90° net) appears in
only ~10% of clips.

## 7. Under-representation ranking → what an enlargement should prioritize

Ranked by scarcity (rarest first). **Priority order** for data acquisition:

1. **Junction / route turns — the scarcest maneuver class.** turn_left is the rarest
   at **6.86%** of timesteps; 57% of clips are turn-free; only 37.7% have a
   junction-scale turn. Left turns marginally rarer than right. This is the #1
   kinematic gap and the one most tied to the program's driving-capability risk (a
   mostly-straight corpus is satisfiable by the "keep going at v0" shortcut).
2. **Stop events & low-speed urban.** Standstill is 7.8% of time; stop *events*
   (moving→stopped) live inside the 12.9% brake_stop but the *stopped-state* coverage
   is thin. Urban creep / queue / stop-and-go is under-sampled vs the 46% highway.
3. **Semantic scenarios — the biggest gap, and entirely unmeasured here (§8).**
   Signalized intersections, roundabouts, pedestrian/cyclist interaction, merges,
   yields. **0% label coverage** — cannot be balanced against this corpus at all.
4. **Sharp / high-curvature geometry** (>90° net heading): only ~10% of clips —
   roundabouts and tight urban turns are rare.

Speed regime and straight/lane-keep are **well-covered** — enlargement should *not*
spend budget there.

## 8. Honest gap — this is a KINEMATIC profile, not a semantic one

Every label above is derived **purely from ego poses** (yaw + speed). The corpus has
**no semantic scenario labels**: traffic lights, stop/yield signs, roundabouts,
pedestrians, cyclists, lane merges/splits, right-of-way, weather, or lighting are all
**invisible** to this profile. "Intersection presence" is proxied only kinematically
(a junction-scale heading change), which **misses every intersection the ego crosses
straight through on green** — likely the majority. Consequently:

- The scenario distribution is a distribution over **ego-kinematic regimes**, not
  driving *situations*.
- A data-enlargement targeting "more intersections" or "pedestrian interactions"
  **cannot be measured or balanced** against this corpus until semantic labels exist
  (the VLM semantic-labeling track, or map/nav ground truth).

This is the single most important caveat for the enlargement decision: **the corpus is
kinematically profiled and semantically blind.**

---

## Deliverable manifest

| Artifact | Location | Notes |
|---|---|---|
| `CORPUS_PROFILE.md` (this file) | `repo:TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-24-parity-corpus-profile/` | staged |
| `corpus_profile.json` | same dir | machine-readable; epochs block corrected to operative `max_horizon=20` |
| `corpus_profile_probe.py` | same dir | the profiling script (read-only, CPU) |
| raw run output | `tanitad-pod3:/workspace/tmp/corpus_profile.json` + `corpus_profile.log` | pod copy (also mirrored into the repo JSON) |

**Escalation / integration:** none required — this is a measurement deliverable, not a
code change to `stack/`. The one correction worth surfacing to the registry owner: the
**dataset windowing horizon is 20** (goal_h), so "30k steps = 4.73 epochs" on this
corpus; a doc that says "max_horizon 16" is describing the tactical predictor, not the
windowing. Numbers here are all MEASURED (poses/labels on pod3) or derived from the
MEASURED frame count; the 406,099-window reconciliation with the registry is exact.
