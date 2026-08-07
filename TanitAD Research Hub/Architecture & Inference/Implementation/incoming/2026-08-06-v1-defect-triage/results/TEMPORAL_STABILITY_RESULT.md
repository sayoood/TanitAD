# Temporal stability — both of Sayed's observations, measured

**MEASURED 2026-08-06** · `flagship-v1arch-v2bal-30k` @ step **29999** · **6,834 windows /
6,794 consecutive pairs** over 40 PhysicalAI OOD-val q90 episodes · idle A40 ·
`results/temporal_stability_v1arch.json.xz` · evidence class **MEASURED (ours)**.

⛔ **The window grid is STRIDE 1** — `corpus_overlay.episode_rollouts` is *"stride-1 grounded
k=20 rollout for every frame"* (`starts = range(0, T - WINDOW - K)`, step 1). **Consecutive
windows are 0.1 s apart.** Every number below is a frame-to-frame quantity, and reading it as
if the windows were 0.8 s apart would understate it eightfold.

| | **flagship v1** | **GT floor** | ratio |
|---|---|---|---|
| replan shift, mean (m) | 0.0947 | **0.0** | — |
| replan shift, p90 (m) | 0.2022 | 0.0 | — |
| replan shift, **max** (m) | **1.0722** | 0.0 | — |
| **replan accel jump, mean (m/s²)** | **1.1021** | **0.0001** | **~11,000×** |
| replan accel jump, p90 (m/s²) | 2.8553 | 0.0003 | — |
| intra-plan jerk RMS (m/s³) | **52.2148** | **1.7066** | **30.6×** |
| manoeuvre toggle rate (per 0.1 s pair) | **0.1759** | — | — |
| manoeuvre mean dwell (windows) | **5.5336** → **0.55 s** | — | — |

⭐ **The GT floor is zero by construction and it measured as zero** (0.0 m, 0.0001 m/s²). The
human's future from `t + 0.1 s` is literally a suffix of its future from `t`, so a perfect
replanner scores 0. The floor is not an argument — it is a measured control that came back
where the maths says it must, which is what makes every flagship number below pure
self-inconsistency rather than a modelling difference.

---

## 1. "Its trajectory is jumping sometimes between the frames" — CONFIRMED, and the mechanism is not where it looks

⚠️ **In POSITION the replan is nearly fine.** 9.5 cm mean, 20 cm p90 over a 0.1 s gap — small.
An eyeball metric on the path would have said "stable".

⭐ **In CONTROL it is not.** The commanded acceleration at the **same absolute instant** changes
by **1.1021 m/s² on average** — and the human's *entire* acceleration RMS is **0.8048 m/s²**.
**Every 0.1 s the model revises its command by more than the whole magnitude of human
acceleration.** p90 is 2.8553 m/s².

⇒ **A small position shift hides a large acceleration change**, which is exactly what a
passenger feels and exactly what no position-space metric can see. This is why the instrument
reports control jump separately, and it is the single most actionable number in this document.

⚠️ **"Sometimes" is the right word and the tail confirms it:** mean 0.095 m, p90 0.202 m, but
**max 1.0722 m**. The jumps are a tail phenomenon on top of a continuously noisy command.

⚠️ **Intra-plan jerk 52.21 vs a 1.71 floor (30.6×)**, on 6,834 windows — independently
consistent with the 39-clip Alpamayo-comparison read (64.30 vs 1.80, 35.8×) at 175× the
sample. ⇒ **The roughness is inside a single plan first.** Smoothing across frames without
fixing that treats a symptom.

## 2. "The tactical manoeuvre is toggling the whole time" — CONFIRMED, literally

Mean dwell **5.53 windows at 0.1 s = 0.55 s**. **The declared manoeuvre changes roughly twice
a second**, and 17.59 % of all consecutive frame pairs see a class change.

⚠️ Read this next to the gate-swept panel: `man~traj κ = 0.5787` at 0.15 falling to **0.2038**
at 0.01, and `kappa_turn_subset = 0.2005` *on* the threshold. **The head is coarse-scale
coherent and fine-scale unstable** — it agrees with the trajectory about big turns while
changing its mind twice a second about everything else.

---

## What this changes in the recommendations (`../V1_DEFECT_TRIAGE.md`)

| recommendation | status after this measurement |
|---|---|
| **Jerk barrier on the current head** (§2 #1) | ⭐ **PROMOTED to first action.** 30.6× on 6,834 windows, ~5 lines, runs against the checkpoint we have. |
| **Unicycle action space** (§1) | ⭐ **STRENGTHENED.** The defect is in *control space*, which is the space the unicycle makes the model's output. A 1.1 m/s² frame-to-frame revision is directly penalisable there and structurally invisible in a free-waypoint head. |
| **Inference-time EMA over controls** (§2 #3) | ⭐ **PROMOTED.** With a measured control jump of 1.1 m/s² per frame and a position shift of only 9.5 cm, an EMA on the *controls* is precisely targeted — and it is free. Still illegal on positions, still requires the unicycle first. |
| **Factorised manoeuvre head** (§3 #1) | ⭐ **STRENGTHENED.** 0.55 s dwell is not a smoothing problem. |
| **Hysteresis diagnostic** (§3 #2) | **Now has its baseline.** Toggle rate 0.1759, dwell 5.5336 — a fix must move these, and without them "it looks steadier" would have been unfalsifiable. |
| **Train-time temporal-consistency loss** (§2 #5) | **Justified but not yet needed.** Try the three cheap fixes first and re-measure; this loss optimises exactly the numbers above, so its value is now testable rather than assumed. |

## Caveats

⚠️ **Unweighted over 6,794 pairs, no CI.** Pairs within an episode are strongly dependent, so an
i.i.d. interval would be badly optimistic. The decision-grade form is the episode-cluster
bootstrap over the 40 episodes — **not run here** and a work item. It would not change the
sign: the GT floor is 0.0 and the arm is at 1.1 m/s².

⚠️ **No second arm.** These are flagship-only; Alpamayo cannot be run stride-1 across an episode
at 30 s/sample. The GT floor is the reference, and for a *self*-consistency metric it is the
correct one.

⚠️ **Jerk and control jump are finite differences** and so amplify discretisation noise — but
the GT goes through the identical instrument and returns 1.7066 and 0.0001.
