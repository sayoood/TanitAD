# Option (c), motion from vision: design note and the admissibility rule (MM, 2026-11-05, before any (c) data)

The PI's decision on DECISION_K2 (verbatim, 2026-11-05): *"Do c then a, contibue autonmously until scalabiliry readiness is achieved"*.
This note fixes how (c) is built and read on the allocated resources (box 2, dev-box CPU) before any (c) number exists.

## 1. The leak the decision item did not state

T1-kin integrates the model's own actions but feeds the LOGGED camera frames on the human's clock (the frame-freeze HANDOFF
discloses this: "the current frame still depicts the human's position at tick j"). The frozen DINOv3-L features carry no motion
the trunk uses (§7.24, §7.31, §7.33), so no arm so far could exploit that. An arm that CAN read motion from vision would read the
human's speed off the logged frames. In the M1 crawl (own speed < half the logged speed while the logged speed is >= 3 m/s) that
is the logged-speed oracle HLVL (§7.35: closes 0.599 to 0.633) arriving through the camera: a leak of the evaluation target, not a
capability. On a real car the camera shows the car's own motion, which the measured ego history already carries (and the
history at the executed speed, HCV, made M1 WORSE, §7.35).

DECISION_K2 says (c) "supplies the speed level the history oracle shows O needs, from vision". Under T1-kin on logged frames
that sentence describes the leak. It is kept as written in the landed item; this note scopes what (c) can legitimately add.

## 2. What (c) can legitimately add

Scene dynamics that the ego history does not carry: other agents' motion (a lead vehicle pulling away or closing, crossing
traffic moving or stopped) and the static scene's rate of approach consistent with the OWN progress. The ego's own speed read
from vision adds nothing on a real car beyond the measured history.

## 3. The instrument that makes a (c) read admissible: `--scene reindexed` (RI)

Frames are re-indexed to the model's OWN along-track progress on the logged path: for a row at time i after t0, the frame is the
logged frame where the human was at the model's own along-track position, never a frame from after time i (tau(i) = min(i, tau*)),
non-decreasing; rows of the measured past stay logged. This is a longitudinal-only approximate re-perception: static geometry is
consistent with the own position, dynamic agents are shown at the human's time, lateral offset is ignored. It is not T2.
Build: `motion_c/ri/` (tool `v8f_t1_eval_ri.py`, patch + tests + mutation arms, after the frame-freeze tool's pattern).
First zero-training use: C0 at s0 / s1 under RI, to measure how much the base depends on the logged frame clock.

## 4. The rule for reading any (c) arm (committed here, before data; enters the (c) PREREG verbatim)

* Each (c) arm and its base are read three ways: `logged`, `hold_now` (frame freeze), `reindexed`.
* **The deciding read is `reindexed`.** A DOMINANT or a closure on `logged` that `reindexed` does not reproduce is reported as
  the camera-clock leak, never as a result.
* A (c) arm whose `logged` closure exceeds its `reindexed` closure by more than half of the `logged` closure is labelled
  LEAK-BORNE in its register row, whatever else it reads.

## 5. The arm, gated by the motion probe (`motion_c/probe/`, pre-registered, TRAIN clips only)

* MOTION PRESENT in the cached 4 x 10 (or 8 x 20) grids: (c-feat), a small temporal adapter over the cached token stack
  (differences of consecutive frames per cell) feeding O, trained on box 2 from cached features (the cheapest (c)).
* ABSENT or WEAK: (c-pix), a small CNN on low-resolution frame differences decoded from the cached JPEGs, if box 2 can carry
  it; encoder adaptation (LoRA on DINOv3-L, AR-2 infrastructure PASS) is projected at 470 h per seed on box 2 (§7.9) and needs
  the A40 or other compute the PI has not allocated: a named blocker, asked, not worked around.

## 6. Then (a)

(a), the level-from-image auxiliary of row K2, is read under the same three scene modes (its single-frame input also depicts the
human's position at tick j), with `reindexed` deciding.
