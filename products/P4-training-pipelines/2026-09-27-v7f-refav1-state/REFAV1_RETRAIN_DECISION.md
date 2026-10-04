# refav1 under R1–R6 — what a retrain is, what it costs, and the one decision it needs

**2026-09-27, TrainingFlyWheel. FINAL — the trunk probe decided it (`…/2026-09-27-refav1-trunk-probe/RESULT.md` §3);
staged, never committed.**

## 0. ⭐ The answer: H is NOT the lever — T is, and a learned kinematic prior is free today
Pre-registered and run on 600 TRAIN episodes, scored on the 141 EVAL episodes (A2 + A3, validity controls OK):
no readout of the FROZEN refav1 trunk — linear or MLP, global or 4 × 10 spatial — beats the same readout on the 8 t0
kinematic features alone; adding the trunk to the MLP makes it separated-WORSE (`P9 − P10` +0.0054 m at 2 s, +0.0987 m
at 6 s). ⇒ a heads-only head (H) would learn kinematics and nothing more. ⭐ The same test's positive: the kinematic
MLP itself (`P10`) — 6 s ADE **2.93 m** vs `damp50` 3.45, `kd_x` 3.56, the linear readout 3.43 (−0.50, separated),
`ha0_ext` 5.36 — is the new bar every learned head must beat and the best measured residual prior.
**⇒ Recommendation: option T** (the adapter/world model retrained WITH the trajectory objective — the R5 head trained
jointly, prior = the learned kinematic MLP), behind a `refav1` launch-gate profile; the storage/compute question below
is the only thing it needs from you, plus the four R1–R6 decisions the build stream raised (κ0 source; the cot-absence
ruling for v7.2; SPEED_BAND as target while max speed is an input; the default prior).

## 1. Why a retrain is the only lever left on refav1
- The pre-registered full-grid test of the best zero-training configuration FAILED at both inference seeds
  (`loncomb3 − ha0_ext` +0.0896 / +0.1515 m ADE, T1, 141 clusters), and every zero-training PLANNER lever is eliminated
  on checkpoint 21,109: the planner adds nothing over t0 kinematics on either axis, the world model's lateral cost share
  is 1.63e-10, and the proposal head was never trained (`…/2026-09-27-refav1-fullgrid-loncomb3/RESULT.md`).
- PI requirements R1 (max-speed INPUT), R3 (all tactical labels), R5 (one 6 s trajectory) and R6 (strategic off
  without losing nav) cannot be met without code + training; the code is being built now (stream `refav1_r1r6/`,
  default-OFF flags, bit-identical when off).

## 2. The two retrains, and what decides between them
| | **H — heads-only** | **T — trunk + heads** |
|---|---|---|
| what trains | the R5 trajectory head (residual over a damped kinematic prior, reading REGION-pooled adapter tokens + t0 kinematics + nav + max speed + the tactical decision) and the R3 goal head; DINOv3 and the adapter/world model FROZEN | the adapter / world models AND the heads jointly, with the imitation trajectory loss as a first-class objective beside feature prediction |
| input it needs | the frozen trunk's region-pooled field per window: **~80 KB/window** (40 regions × 1,024 × fp16); ~16 GB for 4,572 train clips at stride 2 | the full fp8 feature cache: **~303 GB** (4,572 × ~66.2 MB) — or DINOv3 re-encoded on the fly |
| where it runs | the dev box end to end, STREAMED so no cache accumulates: per clip pull → encode → frozen trunk → region-pool → keep ~80 KB/window, discard the fp8 and the v2ep; 4,572 clips at ~4.6 s/clip ≈ 5.8 h on the 4060 (read-only pulls from Thor's B1 sources; the encode is BIT-EXACT to the shipped cache — verified), then head training | nowhere today: D: 92 GB free, C: ~70 GB after the probe, Thor 96 GB ⇒ a new disk, a pod volume, or on-the-fly encoding — **a PI storage/compute decision** |
| decided by | A2 POSITIVE (a readout fitted on TRAIN beats the kinematic readout on EVAL) | A2 NEGATIVE (the frozen trunk carries nothing decodable beyond kinematics, linear or MLP) |
| launch gate | a `refav1-heads` profile (G-DVB for every new flag, G-LIVE: gradient reaches the head only) | a `refav1` profile covering all 40 + new flags |

## 3. Bars, committed before any retrain data (both variants)
T1, 141 eval episodes, paired episode-cluster bootstrap, four families, **2 s AND 6 s**, and read against:
(a) the damped floors `damp50` / `kd_x`; (b) **the NONLINEAR kinematic readout** — the same head with the vision inputs
removed, trained on the same data (the "same inputs minus vision" floor; on the probe the kinematic MLP beats `kd_x`
by 0.086 m at 2 s and 0.63 m at 6 s, and beats every trunk readout — it is the bar, not the linear readout);
(c) GT-defined event strata (turn / accelerate / hard windows) as well as pooled. PRIMARY: `retrained − kinematic
readout` ADE@6 s < 0, CI excluding 0, at two training seeds, with the difference larger than twice the training-seed
replicate. R1: 0 plan exceedances of the clip's limit on every window. R6: no strategic parameter is UPDATED (bit-identical
before/after, the v7f gate's rule; whether `.grad` is populated is reported, not gated).

## 4. What I need from the PI
**T's storage/compute:** ~303 GB of fp8 train features placed somewhere (D: 92 GB free, C: ~70 GB, Thor 96 GB — none
fits), OR train ON THOR with DINOv3 encoded on the fly from the v2ep cache already there (4,713 episodes; the encode is
bit-exact to the shipped cache — verified today; cost per step to be measured before any launch), plus a slot. H is
withdrawn by the probe's committed outcome.
