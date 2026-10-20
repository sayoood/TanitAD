# H-V8F-LON-1: the seed-1 re-read (2026-10-20)

**What was read:**
* The same command as the seed-0 note (`code/lon_rough_diag.py`), run on all four runs plus the nav-shuffle control from box 2's full chain.
* Raw: `lon_rough_full.json`. Tier: T0 open loop, eval-141.
* The seed-0 note committed: *"a material change withdraws it."* What follows is that test, applied part by part.

| run | ADE@6s | oracle per-window speed | one global scale | final arc bias (m) | mean \|accel\| (human 0.751) |
|---|---|---|---|---|---|
| P10 | 2.4870 | 2.0868 | 2.4828 | −0.305 | 0.826 |
| M1 s0 | 2.2466 | 2.0493 | 2.8202 | −2.884 | 1.547 |
| M1 s1 | 2.0244 | 1.6695 | 2.1328 | −1.047 | 1.224 |
| M2b s0 | 2.1753 | 1.8880 | 2.5916 | −2.392 | 1.488 |
| M2b s1 | 2.0546 | 1.6684 | 2.0853 | −0.535 | 1.206 |

**What REPLICATES at seed 1 (kept):**
* **A per-window speed choice is the lever.**
  * The oracle per-window speed scale is worth 0.197 / 0.287 (seed 0) and 0.355 / 0.386 m (seed 1).
  * One global scale makes every run WORSE.
* **Launch from standstill** (v0 < 1, dv > 3) is worse than P10 in all four runs: 2.78–3.06 m vs 2.46.
* **Accelerating windows** (dv ≥ 3) under-progress by −6.5 to −7.9 m, as P10 does.
* **The paths are rougher than the human's.**
  * Mean |accel| is 1.21–1.55 m/s² vs 0.75.
  * Acceleration MAE is worse than P10, separated, in all four runs (`RESULT_FULL.md`).

**What does NOT replicate (WITHDRAWN as seed-0 artefacts):**
* **"Slow in cruise".** At seed 1 the cruise arc bias is −0.93 / −0.62 m, about P10's −1.05.
* **"Strong braking".** At seed 1 the arms overshoot stops by +6.6 / +7.0 m (seed 0: +2.3 m; P10: +11.4).
* **"Yaw rate worse than P10".** At seed 1 it is not separated (+0.24 / +0.13 deg/s).

**What the two seeds suggest (HYPOTHESIS, not tested):**
* Seed 0 landed cautious: slower overall, better at stopping.
* Seed 1 landed assertive: closer in cruise, worse at stopping.
* Both seeds share the launch weakness.
* A single regression output that averages a "keep going" mode and a "brake" mode would place each seed at a different point of that trade-off. A multimodal output with winner-takes-all selection is the test.
* That is a training-side arm: the PI's go and a box-2 slot.
