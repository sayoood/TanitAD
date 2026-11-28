# v9 labels — what every camera frame now carries (report for the PI)

*Master Mind, 2026-10-04 evening. Built by the Data FlyWheel (WP-A); every number MEASURED on the refcv training corpus
(4,369 clips, 746,946 training windows) and the 139 held-out eval clips. Full definitions: `SPEC.md`; validation:
`RESULT.md`; all 132 fields: `V9_SCHEMA.md`; trainer contract: `INTEGRATION.md` (same folder).*

## 1. The change in one line

**Before (v8):** one label record per clip, anchored at a single instant (8.0 s into a 20-s clip). Tactical labels
existed on **23 %** of training frames; the nav command was one token for the whole clip (wrong on **74 %** of
left/right frames); the speed-limit input was the car's own future speed.
**Now (v9):** every 10 Hz camera frame has its own tactical actions and goals for the next **2–8 s**, with distance and
time constraints, its own nav command, and a route checkpoint. Tactical labels on **95.1 %** of training frames.

## 2. Tactical ACTIONS per frame (band: NOW+2 s … NOW+8 s)

| | classes | rule (literal) | constraints carried |
|---|---|---|---|
| **Lateral** | `LANE_KEEP`, `TURN_L`, `TURN_R` | a turn = heading change ≥ 30° on a sustained-yaw segment (≥ 6°/s for ≥ 0.8 s); default variant = junction turns (a road bend is LANE_KEEP with `curve = 1`), variant b (any ≥ 30°) also shipped | turn start and end time (s from NOW), distance, angle, radius |
| **Longitudinal** | `HOLD` (≤ 0.5 m/s), `CREEP` (≤ 2.0 m/s), `STOP` (comes to rest), `FOLLOW` (a lead governs: ≤ 1.75 m from the ego's own path, time gap ≤ 3 s, ≤ 100 m), `DECELERATE` / `ACCELERATE` (±1.5 m/s), `KEEP` | the documented bars (refcv7's builder had silently used +1.0 m/s) | target speed + time and distance to reach it; stop distance, time and point; lead gap and time gap |

* **NUDGE is gone** — the audit showed it labelled road curves (only 10–12 % agreed with the car's motion).
* **Lane change ships never-positive** — measured against the VLM's lane-change text it was right on only 14 of 101
  clips; it needs a better lane-line detector before it can be a label.
* "Nothing happens" labels (LANE_KEEP, KEEP, HOLD) require the full 8-s future to be observed; otherwise the frame
  carries a PARTIAL label (the set of classes the evidence allows) rather than a guess.
* Frames where the car reverses are masked (v9 has no reverse class).

## 3. Tactical GOALS per frame (the 22 goal tokens)

* Geometry goals (turns, stop points, follow-lane) are dense — every frame.
* VLM-derived goals (traffic-light reaction, yield, evade, overtake, merge, exits) are re-timed to the window the VLM
  actually described, with your 2026-09-16 ruling's negatives built into the release; **red-light reaction is propagated
  over its whole stop episode** (22,274 positive frames, v8: 14,022).
* `SPEED_BAND` is replaced by a continuous speed target.
* No label state lives in module memory any more (the defect that trained "lane change left" as a negative everywhere
  is fixed and pinned by a test).

## 4. NAV per frame

* Built from a turn table over the WHOLE recording (fixing three builder defects: suppressed turns leaking through, a
  30-s cap, and curves hiding the next real turn).
* Token: `TURN_L` / `TURN_R` while a turn is in progress or when it is within **max(30 m, 6 s × current speed)**, else
  `FOLLOW` — announced by DISTANCE, like a navigation system ("turn left in 80 m"), with the distance to the turn start
  and end and its angle.
* Realised announced turns get the matching token on **99.6 %** of frames. A time-based token was rejected: it leaks
  future speed (+0.089 linear / +0.267 nonlinear vs +0.039 / +0.152 for the distance token).

## 5. ROUTE CHECKPOINT (your R8-3 — an input, not a label)

* The next route point in the vehicle frame (x forward, y left), read from the ego's driven path smoothed over 8 m — no
  speed enters it. Variants: 30 / 50 / 80 m ahead (RC-A) and the end of the next turn (RC-B).
* **It is powerful:** a trivial planner that just aims at it already turns the right way on 0.92–0.97 of turn frames vs
  refcv7's 0.84, and holds heading within 15° on up to 0.65 vs 0.51.
* **Leak study:** every variant carries some future-speed information, and the source is the ROAD CURVATURE AHEAD —
  a heavily smoothed route with no lane-level detail carries 80–90 % of it. A car's navigation map carries exactly that.
* **Recommendation:** RC-A50 with training noise (2 m along the road, 0.75 m sideways) and ≥ 30 % dropout; its leak
  relative to a road-level route then reads −0.035 (bar ≤ 0.01, PASS); the clean point would fail (+0.021).
* NavSim: a leaderboard-legal agent receives neither the checkpoint nor a turn distance, only the bare
  left/straight/right command — refcv8 is trained to work without them, and every NavSim number gets a legal row and a
  privileged (diagnostic) row.

## 6. Validation (independent derivation: heading from the position path, distance from integrated speed, radius from
the provider's curvature — nothing imported from the builder)

| check | train | held-out eval |
|---|---|---|
| frames with tactical labels (bar ≥ 95 %) | **95.09 %** ✔ | 94.57 % ✘ (95.58 % on tactically scored clips) |
| TURN side recall / precision (bar ≥ 0.95) | **0.980 / 0.988** ✔ | 0.943 / 0.966 ✘ → 0.992 / 0.982 once 3 reversing clips are masked (post hoc) |
| STOP recall / precision | **0.991 / 0.994** ✔ | **0.999 / 0.998** ✔ |
| turn timing, angle, radius, stop time and distance within tolerance | 0.91–1.00 ✔ | 0.90–1.00 ✔ |
| realised turns get the matching nav token | **99.55 %** ✔ | **99.34 %** ✔ |
| "commanded turn but no turn within 6 s ≤ 5 %" | 22.2 % ✘ (by design: the distance token announces slowing approaches early; refcv7: 74.3 %) | 30.4 % ✘ |
| nav distance within ±2 m | 0.76 ✘ (median relative error 0.09 %) | 0.80 ✘ |

## 7. Status and what it feeds

* Built, validated, landed (reader `stack/tanitad/data/v9_labels.py`, builder, tests; commit `c367e7b`), and wired into
  the refcv8 trainer by WP-B (join through the trainer's own clock: 23,772 / 23,772 eval frames, residual 0 s; route
  checkpoint with its noise and dropout; nav-argument dropout; a NavSim-legal mode).
* **Pushed** to the PRIVATE HF dataset `Sayood/tanitad-refcv8-v9-labels`, revision `130c0b94` (2026-10-04 21:07
  Berlin): labels 254 MB + 8 MB, input tables, docs, corpus manifest, dataset card — 18 / 18 files verified after
  upload (the 6 large files by sha256 against the local bytes, the 12 small ones by size), repo confirmed private.
  The first attempt (xet storage path) failed on a network error; the classic LFS path succeeded.
* **Your decisions:** (11) is road geometry admissible inside the route input — the RC-A50 recommendation rests on
  "yes"; (8) drop the 8 clips that share recordings with eval or duplicate a video — default keep (the 693 frames where
  the ego was labelled as another vehicle are masked either way).
