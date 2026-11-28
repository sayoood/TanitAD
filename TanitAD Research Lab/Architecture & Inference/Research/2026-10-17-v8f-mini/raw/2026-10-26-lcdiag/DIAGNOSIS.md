# DIAGNOSIS: HACC's T1-kin curvature excess "at speed" (L2) is a LONGITUDINAL launch-crawl, scored as lateral

**Tier T1 (T1-kin).** This is self-action open loop at m = 10, on eval-141 (2,399 windows; 2,106 windows and 135 episodes carry an L2 value for both arms). It is not closed loop. Zero GPU.

**Type.** DESCRIPTIVE. Every number below is MEASURED from banked dumps, unless a line says otherwise. The four-family rows come from the banked row functions: `comps` (`t1k_read.py`) and `lat_rows` (`t1klc_read.py`), exec'd from source by `common.py`. No row is re-defined. The decomposition reproduces `lat_rows`' L2 per window to ≤ 8.9e-16 in every configuration (`probe1.log`).

**Intervals.** `paired_episode_cluster_bootstrap`, B = 2,000. It answers *another draw of EPISODES*. The two training seeds are the replicate. Medians are descriptive and carry n, not CIs.

**Decisive configuration** (the headline): causal bucket, `consolidated_c/dumps_HACC_s{0,1}_m10_causal.npz`.

## Root cause, in one paragraph

**88–96 % of the L2 excess sits on step pairs where the GT is moving at ≥ 5.2 m/s but HACC's own executed car is below 5.2 m/s.**
* That is +0.0202 of +0.0210 at s0 and +0.0072 of +0.0082 at s1 (1/m).
* These are 5.4 % / 2.8 % of the GT-fast pairs, against 1.4 % for P10.
* They lie in 332 / 220 windows from 61 / 47 episodes: low-speed **launches**.
  * v0 median 5.7 / 4.7 m/s, against 11.2 m/s overall.
  * The human accelerates from 5.7 to 6.8 m/s; HACC's loop decelerates from 5.75 to 4.84 m/s, and to near-standstill in the worst windows.

**At the same own speed, HACC's curvature error equals P10's.** In every bin with ≥ 60 P10 pairs the two are within −19 % / +19 %. The error is a steep function of chord length: 0.0044 1/m above 11.8 m/s, 0.10 at 3.5–5.2 m/s, 1.1 at 1–2 m/s.
* ⇒ The "lateral deficit at speed" is HACC **crawling where the human launches**. The curvature row, masked on the **GT's** speed only, charges the jitter of a crawling track to lateral control.
* With both arms' own speeds also ≥ 5.2 m/s (the same pairs for both arms), the excess is +0.0018 [+0.0002, +0.0036] at s0 and +0.0015 [−0.0001, +0.0033] at s1, n.s.

**What pushes the loop into crawl.** The leading measured mechanism has not yet been intervened on.
* HACC's executed knots are H = P + w(k)·(O − P). The step-varying along weight (w_along 0.14 → 0.32 across knots 8–11) puts a term **(dw/dk)·(O − P)** into the plan's velocity, and that term is in **neither** plan.
* The T1 replan state reads exactly knots 8–11 (one-frame differences of the Hermite track). So at every replan it receives a deceleration that P10 then extrapolates as a constant control.
* At eval tick 1 (exact, logged state) in the launch windows, this term is **79 % / 74 %** of HACC's acceleration gap to P10: −1.06 of −1.35 and −0.52 of −0.70 m/s². O's own (slow) launch intent supplies the rest.

## The hypotheses, each with its discriminating measurement

| | measurement (file) | s0 | s1 | verdict |
|---|---|---|---|---|
| **H1 seams** | L2 share of the seam pairs 9, 19 (exact decomposition, `probe1.log`) | +0.0015 [+0.0004, +0.0031] of +0.0210 (7 %) | +0.0003 [+0.0001, +0.0006] of +0.0082 (4 %) | **REFUTED as the cause.** The seams are real but minor. |
| **H2 state distribution** | L2 excess on GT-fast pairs where the arm's OWN speed < 5.2 m/s (`probe2.log` b) | +0.0202 of +0.0210 | +0.0072 of +0.0082 | **SUPPORTED, in a specific form:** the loop visits *crawl* states in launch windows. |
| | the same pairs, joint own-speed mask (`probe2.log` b) | +0.0018 [+0.0002, +0.0036] | +0.0015 [−0.0001, +0.0033] n.s. | |
| | curvature error per own-speed bin, HACC vs P10 (`probe9.log`) | 2–3.5 m/s: 0.356 vs 0.437; 3.5–5.2: 0.099 vs 0.102 | 0.410 vs 0.437; 0.098 vs 0.102 | equal at equal own speed |
| **H2b (added) lateral state feedback** (κ0 re-issued as a constant arc) | next plan's mean curvature regressed on the end pair of the previous plan (`probe7.log`) | slope negative for HACC, B and P10 (−0.03 to −0.43); GT +0.60 to +0.80 | same | **REFUTED** as the channel. The state channel that matters is longitudinal (H3′). |
| **H3 the post-map's LATERAL shape** | plain M2b B (no post-map) − P10, L2, logged bucket (`probe1.log`) | +0.0497 [+0.0308, +0.0732] | +0.0343 [+0.0211, +0.0508] | **REFUTED.** Without the post-map the excess is 2.5× / 3.9× HACC's own logged-bucket excess (+0.0195 / +0.0087). |
| | the cross axis's w-ramp κ0 injection, median \|Δκ0\| HACC vs ramp-free (`probe10.log`) | 0.0124 vs 0.0130 | 0.0248 vs 0.0248 | none |
| **H3′ (new) the post-map's LONGITUDINAL step variation** | the (dw/dk)·D share of HACC's tick-1 a0 gap to P10, launch windows (`probe10.log`) | −1.06 of −1.35 m/s² (79 %) | −0.52 of −0.70 (74 %) | **LEADING MECHANISM** for the crawl. Descriptive, not intervened. |
| | the increment-domain ramp term, launch windows / other windows | −0.54 / −0.15 m/s² median | −0.29 / −0.12 | |
| **H4 O's high-frequency residual** | the jointly-moving remainder, HACC (B) | +0.0018 (B +0.0092) | +0.0015 (B +0.0089) | **MINOR.** At most 9 % / 18 % of L2; the post-map already removes about 80 % of O's. |
| **H5 (added) timing / along-track misalignment** | arc-length-aligned curvature error (`probe2.log` c) | +0.0161 [+0.0071, +0.0280] | +0.0060 [+0.0002, +0.0137] | **REFUTED.** Aligning by distance travelled removes ≤ 27 %. |
| **H6 (added) the camera viewpoint** | L2 with `--scene frozen` (`probe5.log`) | +0.0170 | +0.0102 | **REFUTED.** Freezing the cameras at t0 keeps 81 % / 124 %. |
| **H7 (added) is it in the open-loop plan?** | L2 at m = 30 (the tick-0 plan executed whole) (`probe5.log`) | +0.0003 [−0.0010, +0.0016] | +0.0009 [−0.0008, +0.0029] | **NO.** It is created by the loop. |

## The supporting measurements

### 1. Where the excess is (`probe1.log`, exact additive split of the paired L2 delta)

| | s0 | s1 |
|---|---|---|
| plan 0 (logged state) | **−0.0005** (HACC better) | **−0.0005** (HACC better) |
| plan 1 | +0.0014 | about 0 |
| the seams | +0.0015 | +0.0003 |
| **plan 2 (pairs 20–28, t = 4–6 s)** | **+0.0186 (89 %)** | **+0.0085 (104 %)** |
| the last pair alone | 27 % | 34 % |

⇒ The excess grows with time spent in the loop.

### 2. The windows (`probe3.log`, `probe4.log`, `probe8.log`)

* **The excess outside those windows** is +0.0001 / +0.0003. All of it is inside them.
* **The top 30 windows** (1.4 % of 2,106), from 12 / 18 episodes, carry +0.0138 / +0.0080 (66 % / 98 %).
* **They are launches.** Example: eid 69, v0 2.96 m/s.
  * The GT goes from 2.9 to 5.5 m/s.
  * HACC decelerates to 0.75 m/s inside plan 0 and crawls at 0.2–0.4 m/s through plan 1.
  * Its curvatures reach 12–52 1/m on centimetre chords.
  * P10 accelerates to 14.5 m/s.
* **HACC lags the GT** by a median 0.26, 2.66 and 7.89 m at 2, 4 and 6 s (s0). P10's along offset stays between −0.3 and +1.0 m.
* **HACC's executed heading oscillates**, with a sign-flip rate of 0.64 / 0.60 against the GT's 0.15. These are real centimetre-to-decimetre deviations on short chords (median \|Δκ\|·ds² = 5 cm), about 10× the 5 mm noise floor at the same chord length. They are not millimetres.

### 3. The replan state (`probe6.log`, `probe10.log`, `statekin.py` = the tool's own `hermite_frames` and `yaw_kin_from_track`)

**Controls (both seeds):**
* S-1: the rebuilt frame-2j pose equals the dumped tick pose, max \|d\| = 0.0.
* S-2: the checkpoint's own P10 prior on the rebuilt kin3 equals the dumped P to ≤ 1.2e-3 m.
* The P10 tick-0 plan recovered from HACC = P + w(O − P) equals `fl[:10]` to ≤ 4.9e-6 m.
* The blend of the recovered P and O reproduces HACC's knots 1–12 to 1.4e-14 m.

**Eval tick 1, launch windows (n = 332 / 220):**

| rebuilt state | s0 | s1 |
|---|---|---|
| HACC a0 − GT (m/s²) | −0.68 | −0.12 |
| P10 a0 − GT | +0.67 | +0.58 |
| HACC v0 − GT (m/s) | −0.47 | −0.53 |
| O's own v0 − GT | −1.78 | −0.93 |

**Train ticks (own state):** the ramp term is −0.13 to −0.40 m/s² median in every speed band, at every replan.

**The per-plan probe confounded state with plan.** From the SAME state, the own speed is the same. Since the error is a function of own speed, the planner does not matter there. That is why the earlier per-plan probe found HACC ≤ P10, and why HACC-LAT arm B (cross = P10 below v\*) moved nothing (`RESULT_V8F_HACC_LAT.md`).

## How much it accounts for

| part of the L2 excess (1/m) | s0 | s1 |
|---|---|---|
| on arm-crawl pairs (own < 5.2, GT ≥ 5.2) | **+0.0202 (96 %)** | **+0.0072 (88 %)** |
| jointly moving remainder (the real lateral-at-speed part) | +0.0018 [+0.0002, +0.0036] | +0.0015 [−0.0001, +0.0033] n.s. |

**The crawl's sources** (eval tick 1 in launch windows; a share of HACC's a0 gap to P10, not of L2):

| source | s0 | s1 |
|---|---|---|
| the post-map's along step variation | ~79 % | ~74 % |
| O's own slow launch intent | ~21 % | ~26 % |

⇒ **ROOT CAUSE: a launch-window speed collapse of HACC's loop.**
* The collapse is pumped by the post-map's step-varying along weight acting through the one-frame replan state, on top of O's slow launch intent.
* The curvature row conditions on the GT's speed only, so it scores the collapse as lateral.

## Instrument finding (for `RETRACTION_LOG` class review; nothing is re-scored here)

This is the class of `RETR-2026-09-26-YAWMASK` again: a lateral rate row scoring standstill jitter.
* There, the jitter entered through the **GT's** speed.
* Here, it enters through the **arm's**. L2's mask keeps a pair when the GT moves, whatever the arm does.
* ⛔ The pre-registered L2 result stands as written: CONDITIONED LATERAL DEFICIT.
* A conditioned row on the joint mask can only be read **beside** a future arm, because HACC's value on it is now known (+0.0018 / +0.0015).

## Beside: a hypothesis that decides nothing

**HACC's T1 margin may partly ride on the same term.** In non-launch windows, HACC's tick-1 a0 is 0.55 / 0.48 m/s² below P10's, and 48 % / 44 % of that is the step-variation term. It offsets P10's own +0.88 / +0.91 m/s² replan-acceleration overshoot (`probe10.log`).

⇒ A ramp-free post-map may trade the launch crawl for P10-like overshoot. `PREREG_DRAFT.md` commits that outcome as a row.

## Files (all in `C:/Users/Admin/lcdiag/`, 1.6 MB)

| file | what it is |
|---|---|
| `common.py`, `statekin.py` | shared loaders; the replan-state rebuild through the tool's own functions |
| `probe1_pair_profile.py` … `probe12b_mch.py` | the probes |
| `probe*.log`, `probe*.json`, `probe6_*_s{0,1}.npz` | their logs and outputs |
| `probe11`, `probe12`, `probe12b` | TRAIN only: fits and the lever's train-side checks. No eval array is opened. |
