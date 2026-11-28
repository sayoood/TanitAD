# PREREG_V8F_VOE1: does each level notice an impossible future? Violation of expectation on real driving video (Master Mind, 2026-11-07, written before any VOE1 code or statistic exists)

## Why
A latent world model can be checked without any decoder: show it a future that cannot happen and measure whether its prediction error jumps. V-JEPA reaches 98 % pairwise accuracy on IntPhys this way (Garrido 2502.11831), and LeWM shows a teleportation spike in three environments (REPORT_A F6). Driving has only one modest signal so far (2606.28383, nuPlan mini, structured states). This card is the first per-level read on real driving video (`research_imagination/PLAN_IMAGINATION.md` E2). Tier **T0**.

⚠️ A model can be "surprised" only because the frame changed a lot. So the deciding comparison is always against the **Copy-Last null** (surprise = the distance between consecutive encoded latents), and two manipulations are built so that the null cannot see them at all.

## Data and models
* eval-141 windows (`theory/wmact/latents.pt` frame bank and windows; 1,976 windows, 141 clips). Manipulations act on the cached per-frame tokens and the logged action sequence. No image is edited or rendered.
* Models: M2b s0 and s1, zero training. A **random-init world model** (the same architecture and config, seeds 0 and 1, untrained; the encoder side is the trained M2b chassis) is the control.

## Manipulations (applied from the onset frame t0 onward; each window yields one manipulated copy per type, paired with its unmodified copy)
1. **TELEPORT:** the frames after t0 are replaced by the frames 1.6 s later in the same clip. The actions stay logged.
2. **FREEZE:** frame t0 is repeated for 0.8 s, only in windows whose logged speed is above 3 m/s. The Copy-Last surprise is ZERO here.
3. **REVERSE:** the 1.6 s after t0 play backwards.
4. **SPLICE:** the frames after t0 come from another clip (seed 0 draw).
5. **ACTION-SWAP:** the frames are real, and the actions after t0 come from another window whose mean longitudinal action differs by at least 1.5 m/s² or whose mean yaw rate differs by at least 0.15 rad/s. The Copy-Last null cannot see this one.

## The statistic
* Surprise at level l and step t is ‖prediction − encoded latent‖², divided by that level's median surprise on unmodified windows.
* The window score is the **maximum** over the level's steps that fall within 1.6 s after t0 (max beats mean, Garrido).
* Pairwise accuracy = the share of pairs where the manipulated copy scores above its unmodified copy.
* Interval: episode-cluster bootstrap B 2,000, seed 0. Two-part rule: the R = 199 permutation reference swaps the two copies' labels within whole episodes.

## Committed outcomes, per manipulation, level and seed
* **NOTICES:** accuracy is two-part separated above 0.5, AND (TELEPORT, REVERSE and SPLICE only) separated above the Copy-Last null's accuracy, paired over the same pairs, at both seeds.
* **NULL-ONLY:** above 0.5 but not above the Copy-Last null. The level sees a large change, not an impossible one.
* **BLIND:** not separated above 0.5.
* ⭐ FREEZE and ACTION-SWAP are the decisive rows: the null reads ≤ 0.5 on them by construction, so NOTICES there needs a world model that predicts motion (FREEZE) or uses the action (ACTION-SWAP).

## Event alignment (IA-6d, THICK's question)
* On unmodified windows, each level's surprise trace is scored against three event types built from labels (labels may use ego and cuboids):
  * braking onset: acceleration first below −2 m/s²;
  * turn onset: |yaw rate| first above 0.1 rad/s;
  * a lead change with the gap closing.
* Score: average precision (AP) of the surprise peaks within ±0.4 s of an event, against the event rate (chance) and against the Copy-Last null's AP.
* **ALIGNS** per level: AP − chance is separated above 0 and above the null's at both seeds.
* The hierarchy hypothesis (HYPOTHESIS, Clockwork VAE / THICK): L3 aligns at least as well as L1. It is reported, with no bar.

## What each outcome triggers (committed now)
* ACTION-SWAP **BLIND** at every level → the world model ignores the action after t0 in its error. This joins IMAG1's IA-3; if IA-3 is also INERT or WEAK, the lever is **E7 DIVRS** plus a counterfactual-target term (UWM-JEPA: teacher-forced targets leave the action term inert).
* FREEZE **BLIND** at every level → the predictor does not carry motion. This joins G1, and **MRATE** is the lever (already carded).
* TELEPORT / SPLICE **NULL-ONLY** at every level → surprise is a frame-change detector. The next lever is IMAG1's IA-2 (does the prediction beat Copy-Last at all?) before any new arm.
* NOTICES on FREEZE or ACTION-SWAP at any level → the first positive "imagination is used" evidence on video. It goes into the paper with the per-level table and the surprise-trace figure.

## Controls that must read known values (the run is INVALID if any fails)
1. **Real-vs-real:** two unmodified copies of the same window, the second from the window starting 0.2 s later. Accuracy must not be two-part separated from 0.5 at any level.
2. **Random-init world model:** on FREEZE and ACTION-SWAP it must not be NOTICES at any level.
3. Calibration: 20 extra permutation draws (seeds 1001–1020) scored as if real; at most 5 % of rows separated per family.
4. A synthetic test (a toy latent world model that predicts constant velocity, given clips with a planted freeze and a planted teleport) must give NOTICES on both before the real run. An action-blind toy must give BLIND on ACTION-SWAP.

## Implementation choices
The implementer writes `theory/voe1/IMPL_CHOICES_VOE1.md` (t0, window lengths per level, how each level's steps map to frames after t0, the event-label sources, the draw seeds) and records its sha256 **before any statistic is computed**.

## Log
* 2026-11-07 14:50:03 frozen: prefix 5822 bytes, sha256 555c5a3896d1ff0d745763b48a14ed7294b4ef814fab0fd5a31d44e2ccfbf69a (everything above this line). No VOE1 code or statistic existed at this time.
* 2026-11-07 15:30:33 VOE1 log (MM): run INVALID as committed, control 1 (RI_s0/L3 real-vs-real 0.469 separated below 0.5) and control 2 (RI_s1/ACTION-SWAP/L3 0.550 NOTICES) fail; controls 3 and 4 pass; IMPL_CHOICES 848a2c8c; RESULT_VOE1.json 79cfeb6f. No M2b manipulation row read. Next arm: PREREG_V8F_VOE1B.md (prefix sha256 aa317fe3f046e9d0522701d4b35c9d029b1fd55a4a4590823c02ff60449493c3).
