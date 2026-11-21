**Verdicts, as committed (Amendment 4, before the data):**
* **H-MINI-K-1: PASS on M2b.**
  * M2b − P10-yaw-4512 at 6 s is −0.3117 (seed 0) and −0.4324 (seed 1). Both are separated and both clear 2F = 0.2415.
  * **M1 FAILS on its own floor.** It is separated at both seeds (−0.2404 / −0.4626), but seed 0's |Δ| is below 2F = 0.4444.
  * Two arms were read, and K-1 passes iff ≥ 1 passes (the stated multiplicity).
  * M2b's 6 s ADE is 2.1753 / 2.0546 m against P10's 2.4870: a 13–17 % lower open-loop 6 s error than the learned kinematic floor, at T0 on eval-141.
* **H-MINI-WM-1: FAIL.**
  * M2b − M1 is −0.0713 [−0.1548, +0.0100] at seed 0 and **+0.0302** [−0.0244, +0.0846] at seed 1. Neither is separated, and seed 1 has the wrong sign.
  * It fails under ANY seed floor, not only the committed 2F = 0.4444.
  * ⇒ The H-JEPA WM + IDM auxiliary is not a v8F-mini trajectory lever at this scale. Per the committed branch, the WM's remaining route is **rung β** (the WM as a planner), which the box-2 queue starts next.

**The seed floor is the result's most consequential number (MEASURED).**
* The two seeds of one arm differ by **0.2222 m** (M1) and **0.1207 m** (M2b) at 6 s.
* The runs differ ONLY in the seed: the argv differs in `--seed` and `--out` alone (`raw/full/runs/*/argv.json`).
* Seed 1's training ADE is also lower (last 10 %: M1 1.679 vs 1.746, M2b 1.710 vs 1.775), so this is run-to-run training variance, not an eval artefact.
* It is **common to both arms**: seed 1 is better in both, by 0.12–0.22 m, because the arms share seeds.
* The seed-paired M2b − M1 contrast therefore moves much less between seeds (−0.071 → +0.030, a spread of 0.10) than either arm does.
* ⇒ On this rig, two seeds cannot resolve a lever smaller than about 0.25–0.45 m through a within-arm floor. The cheapest next experiment for any lever claim here is **more seeds, not more windows** (`H-ESTIM-SEED-1`).
* A future pre-registration may justify a seed-PAIRED floor. ⛔ This one's bar is read as written.

**The four families** (tables above; MEASURED, T0):
* **LONGITUDINAL.**
  * Acceleration MAE is **worse than P10 in all four runs** (+0.29 to +0.61 m/s², separated). This is the one family defect that replicates.
  * At seed 1, along-track MAE beats P10 (−0.21 m, separated, both arms); speed MAE does too, separated for M2b only (−0.069; M1 −0.055 is not). At seed 0 the arms are slower (final along-track bias −2.2 to −2.8 m vs P10 +0.3).
  * Distance keeping is UNAVAILABLE (no lead track wired into this eval; a work item, not a pass).
* **LATERAL.**
  * Heading (−1.46 to −1.76 deg) and cross-track (−0.32 to −0.43 m) beat P10 in all four runs, separated.
  * Yaw rate is worse than P10 at seed 0 only (+0.93 / +1.60). At seed 1 it is not separated (+0.24 / +0.13).
* **TACTICAL.**
  * Path-derived lateral decision accuracy is 0.83–0.85 vs P10's 0.77. Longitudinal decision accuracy is 0.56–0.59 vs 0.53: the weak family, kappa 0.33–0.37.
  * Tactical-head macro-F1: lat 0.82–0.84 (majority 0.317); lon 0.47–0.53 (majority 0.071).
  * Under nav shuffle, lat collapses to 0.30. The lateral head reads the route.
* **STRATEGIC** (un-paused for v8F-mini).
  * Checkpoint error at 50 m is 7.9–8.1 m (M1-navshuf: 12.3).
  * Nav-used, seed 0 (M1-navshuf − M1): +4.25 [+3.02, +5.57] m at 50 m and +9.86 at 80 m, separated. The strategic layer uses its route input.
* **CONSISTENCY (a):** the emitted plans violate active GT constraints at 0.155–0.210, against P10's 0.311.

**Riders read on the same files, with no bar on this result:**
* `PREREG_V8F_SMOOTH` (sealed `e58534a`) reads **SMOOTH-PASS for both arms**: `raw/2026-10-20-smooth/RESULT.md`.
* The seed-1 re-read of `H-V8F-LON-1` is in `raw/2026-10-20-lon-rough/NOTE_S1.md`. The per-window speed lever and the launch weakness replicate; the seed-0 "slow in cruise" and "strong braking" do not.

**Next (Rule Zero):**
1. Rung β runs now on box 2 (the WM as a planner: WM-1's committed branch).
2. K-1 PASS names the controls M2a / M1-navonly / M1-nocons as next. Each is the PI's go and a box-2 slot.
3. The largest measured lever is the **longitudinal speed decision**: an oracle per-window speed scale is worth 0.20–0.39 m across the four runs (M1 s0 0.197, M2b s0 0.287, M1 s1 0.355, M2b s1 0.386). It is training-side, so the PI's go.
