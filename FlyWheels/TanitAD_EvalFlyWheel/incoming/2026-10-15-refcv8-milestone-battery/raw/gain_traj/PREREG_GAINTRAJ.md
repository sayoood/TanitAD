# PREREG_GAINTRAJ -- does refcv8's best arc-length RE-TIMING gain move toward 1 across the banked checkpoints?

Written 2026-10-19 15:08 local, **before any per-tree gain `g*` (or per-horizon re-timing gain) of any tree other than the 30k EMA's pre-banked frozen table was computed.**
Zero GPU, CPU only, banked panels only. Instrument: NEW `code/gain_traj_r8.py` (imports `a0k0_headroom_retime_r8` = RT and `compliant_reread_r8` = CR, edits neither).
Tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**. Trees: `C:/Users/Admin/refcv8_eval_archive/battery/<tree>/`.
Single training run (one training seed), checkpoints of ONE trajectory of that run.

## DISCLOSURE (binding)
* The brief states the lever (`s'(t) = s_prior(t) + g_t (s_os(t) - s_prior(t))`, train-fit gains S2 ~ [0.967, 0.619, 0.551, 0.531], flat g = 0.5 nearly matches). That is the motivation; it is not a result of this registration.
* **Pre-seen, one tree only:** `raw/a0k0_retime/FROZEN_GAINS_RETIME_ema30k.json` holds RT's per-horizon re-timing gains fitted in-sample on EMA-30k with both inference seeds POOLED
  (S2 compliant `[1.0372, 0.6356, 0.4906, 0.4358]`, S2 all `[0.9968, 0.5834, 0.4260, 0.3825]`; S6 compliant `[0.6463, 0.4482, 0.4820, 0.5163, 0.5339, 0.5562]`). I read that file while
  designing this registration, so **the 30k-EMA per-horizon (secondary) reading is not blind**. It is NOT the primary estimand (the flat arc-length least-squares `g*` below was not computed on any tree before this text was frozen),
  and it is used as a known-value control for the data path (C4), not as evidence.
* No per-tree `g*` of the five EMA trees or the five raw trees, at any seed, has been seen.

## 1. Estimands (arrays exactly as RT builds them)
Per (tree, surface in {S2, S6}, inference seed in {0, 1}), the **ceiling-compliant** windows only (mask = `CR.prepare_seed(...)["comp"][rows]`, asserted: `mask_sha256 == 4ebc6b8604f064f69e9e94c15e099062f68339a8ef38d666e8a29d29cb1c4129`
on every tree and seed; counts asserted equal across trees and equal to the brief's 30k values S2 4,159 / S6 3,251). Plans `O` (= os), prior `HP` (= `ha0_ext_pose`), human GT `G`
come from `RT.surface_inputs` (`X["O"]`, `X["HP"]`, `X["G"]`), positions at the surface's S-grid instants (S2: 0.5/1.0/1.5/2.0 s; S6: 1..6 s), float64.
Arc length per instant, for any of the three paths: cumulative chord length of the polyline through (0,0) and its grid points = `RT.polyline_geometry(x)["L"][:, 1:]`
(this is RT's own `s_os` and `s_prior = RT.prior_arc(HP)`; `s_human` is the same function on `G`).

* **PRIMARY.** `d = s_os - s_prior`, `h = s_human - s_prior` (both `[n_windows, T]`).
  `g* = argmin_g  sum_{windows, instants} (s_prior + g * d - s_human)^2 = sum(d*h) / sum(d*d)`, pooled over ALL compliant windows of the surface and ALL grid instants of the surface. No clip, no box.
  `sum(d*d) == 0` refuses (no estimate). g* = 1: the os arc length equals the human's on average (the defect is absent); g* < 1: os departs from the prior along-track by more than the human does.
* **SECONDARY (diagnostic, in-sample, no role in the label).** (a) Per-instant gains from RT's re-timing fitter with its DEFAULTS: `RT.retime_fit(O, HP, G)` (coarse grid 0.05 over [0, 2] + bounded Brent; it minimises mean position error ALONG the os polyline,
  so it is not the arc-length LS above). **Deviation from the brief, stated up front:** the brief names `fit_gains` (~line 147); that function is the Amendment-2 position-BLEND weight fitter (w on os vs prior per axis), not the re-timing gain
  fitter, and the quoted S2 numbers are re-timing gains, so `retime_fit` (the re-timing fitter) is used. (b) Per-instant arc-length LS gains `g_t* = sum_w d_t h_t / sum_w d_t^2` (the primary estimator restricted to one instant).
  Fit on the same compliant windows, one fit per (tree, surface, seed), no CI.

## 2. Interval (episode-cluster bootstrap; names its question)
For each (surface, seed) the compliant windows' episode ids (`eid`, taken as `str`) define `n_ep` clusters; per-episode sufficient statistics `A_e = sum d*h`, `B_e = sum d*d`.
`rng = numpy.random.default_rng(0)`; `picks = rng.integers(0, n_ep, size=(2000, n_ep))`; replicate `b`: `g*_b = sum_e c_be A_e / sum_e c_be B_e` with `c_be` the multiplicity of episode e in draw b.
95 % interval = percentiles [2.5, 97.5] of `g*_b`. **The same `picks` are used for every tree of that (surface, seed)** (episode sets and window orders asserted identical across trees), so the paired difference
`Delta = g*(30k) - g*(5k)` is the percentile interval of `g*_b(30k) - g*_b(5k)` on the same draws; `separated > 0` means its lower bound > 0 (decided on the unrounded bounds).
**What the interval answers: "would another draw of EPISODES say this?" ONLY** -- models held fixed. It does NOT answer "would another TRAINING run say this?" (one training seed; the trees are checkpoints of one run) and it does NOT cover
an inference-sampling redraw (seed 1 is the replicate beside seed 0: the seed-0 vs seed-1 difference of `g*` is reported per tree as the rig's inference-noise reading, and the label demands agreement of both seeds).

## 3. Outcome rule (EMA trees, S2, both inference seeds; raw trees and S6 are reported BESIDE, never in the label)
EMA series = steps 5,000 / 10,500 / 15,000 / 20,000 / 30,000 (`step{N}ema_unc`).
* **CONVERGING** = `Delta = g*(30k) - g*(5k)` separated > 0 at BOTH seeds AND the five EMA point estimates are non-decreasing (`g*_{i+1} >= g*_i` for each of the four consecutive steps, literal, no tolerance) at BOTH seeds.
* **STRUCTURAL** = NOT CONVERGING AND the 30k `g*` interval upper bound < 0.8 at BOTH seeds.
* **MIXED** = otherwise.
The label does not distinguish a flat `g*` from a falling one (a falling `g*` that fails CONVERGING with 30k upper < 0.8 is also STRUCTURAL); the sign and CI of `Delta` and all four consecutive differences are reported next to the label so a reader sees which.
Meaning of the labels: CONVERGING = the along-track over-scaling shrinks with training in this run (not yet evidence that it vanishes by the end); STRUCTURAL = it does not close over the sampled range and 30k is still far from 1, so it must be addressed in refcv9's
training; MIXED = no lever read. **No extrapolation to the final step 48,852 (nor to any step beyond 30,000) is made or implied; 40,000 and 48,852 trees exist in the archive but are outside this registration.**

## 4. Known-value controls (all must PASS before any real number is written; a failure writes `raw/gain_traj/WITHHELD.json` and stops). Literals, on EMA-30k S2 compliant arrays, BOTH inference seeds, through the same estimator function as the real estimate:
* **C1** `s_os := s_human` -> `g* == 1` within 1e-9 (and every bootstrap replicate == 1 within 1e-9: interval [1, 1]).
* **C2** `s_os := s_prior + 2 (s_human - s_prior)` -> `g* == 0.5` within 1e-9.
* **C3** `s_os := s_prior + 4 (s_human - s_prior)` -> `g* == 0.25` within 1e-9 (added: C2 is invariant under the prior/os swap used by the mutation, 0.5 -> 0.5, so C2 alone cannot detect it).
* **C-MUT** deliberate regression: the estimator with prior and os SWAPPED (`s_os` as the base, `s_prior` as the direction) must read **RED on C1** (expected 0.0 vs 1) and on **C3** (expected 0.75 vs 0.25); the mutation is a defect on a copy, not applied to the real estimate. If the swapped estimator passes C1 the control withholds.
* **C4** data-path: `RT.retime_fit` on the 30k-EMA compliant arrays with both seeds POOLED reproduces `FROZEN_GAINS_RETIME_ema30k.json` `surfaces.<S>.compliant_info_only.RETIME` within 1e-6 (S2 and S6); and `n_windows` equals the asserted compliant counts.
* **C5** plumbing: the PRIMARY `g*` recomputed by an independent plain-loop (per-window python sums, no shared helper) equals the vectorised value within 1e-9 on EMA-30k S2/S6 seed 0.

## 5. Outputs
`raw/gain_traj/GAINTRAJ.json` (per tree / surface / seed: `g*`, CI, n_windows, n_episodes, per-instant gains (a) and (b); the paired `Delta`s; controls; the label with its inputs), `GAINTRAJ.md` (table + label + caveats),
`PREREG_GAINTRAJ.sha256` (written immediately after this file, before the script runs). The script verifies this file's sha256 at start and records its own sha256.
Caveats carried into every table: single training seed; T1 open loop on eval139 frames; per-tree per-instant fits are in-sample; eval episodes only; interval = episode draw only.

## 6. Optional (priority 4): TRAIN-clip flat `g*`
If the train panels `E:/refcv8_eval_kit/trainfit/step30000ema_train139/{panel_s0,s6_s0,...}` are readable through `retime_trainfit_r8._load_train`, the same PRIMARY `g*` (compliant windows, and all windows beside) with the same bootstrap is fitted on the 139
TRAIN-clip rolls of EMA-30k, both seeds, and put beside the eval EMA-30k `g*`. The train-vs-eval difference uses an UNPAIRED (two different clip sets) bootstrap. Reported beside; no role in the label. Skipped with the reason if unreadable.
