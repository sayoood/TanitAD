# D9 is not a dead end — it is **two named reward defects**, one per seed, each measured and separated

**Date:** 2026-09-17 · **Evidence class: MEASURED (ours)** · **Tier: T1** (self-action open loop) ·
**Estimator:** `paired_episode_cluster_bootstrap`, `n_boot` 2000, seed 0, clustered by episode.
**Zero GPU** — every number here is a re-read of dumps the Stage A run already banked.

**PI directive, 2026-09-17: *"dont park D9, solve it and prove it."*** This package is the
diagnosis half. It does not yet claim a fix works; it claims we now know **what to fix, and why**,
with the failure localised separately for each seed.

## What was known, and what was missing

`H-DDV2RL-2` closed **FAIL-HARM** on both seeds: T1 `ade_m` **0.2994 → 0.5502 / 0.5561**, **43×**
the two-seed floor. The banked note said the two seeds *"damage different families"* but no
mechanism was named — so the standing option was to park D9.

⛔ **A refutation is a waypoint, not a deliverable.** The mechanism was findable from source and
testable on artifacts already on disk.

## ⭐ The defect, read from source

`tanitad/rl/pdm_proxy.py`:

```
pdms = NC × DAC × (w_ep·EP + w_ttc·TTC + w_c·C) / (w_ep + w_ttc + w_c)
```

**DAC is a binary multiplier**, and the module's own docstring records the rest:

> `| DAC | footprint in non-drivable area at any tick | dac_from_drivable from SAM3 map GT when a map exists; **1 when it does not** | MISSING on the RL-train split |`

The RL-train split had no SAM3 maps, so **DAC ≡ 1 for every candidate in every anchor group**.
GRPO's advantage is computed **within** a group (`intra_anchor_advantage`), so a term with **zero
within-group variance contributes exactly zero to the gradient**.

⭐ **This is algebraic, not a matter of degree.** The road-boundary constraint was not weak — it was
**absent**. The policy was optimised on `NC × (5·EP + 5·TTC + w_c·C)` alone.

## ⭐ Test 1 — seed 0 drives off the road, and it is separated

Off-road waypoint rate: an ego-centre point test against SAM3 map GT, on the **identical windows**,
37 of 41 held-out episodes, 1,265 windows. *(A point test, not the footprint, so no yaw derivation
can corrupt it — therefore a **lower bound** on off-road contact.)*

| arm | off-road rate |
|---|---|
| **human** (gold control) | **0.0171** |
| `ha0` (straight ahead at v0) | 0.0202 |
| **base** — the cold start | **0.0165** |
| **rl-s0** | **0.0312** |
| **rl-s1** | **0.0176** |

| paired comparison | Δ | CI | |
|---|---|---|---|
| **rl-s0 − base** | **+0.0151** | [+0.0057, +0.0261] | ⭐ **SEPARATED** |
| rl-s1 − base | +0.0015 | [−0.0037, +0.0068] | not separated |
| **base − human** | −0.0009 | [−0.0040, +0.0018] | not separated |
| `ha0` − human | +0.0038 | [−0.0059, +0.0140] | not separated |

⭐ **The cold start drives on drivable ground indistinguishably from the human** (`base − human` not
separated). ⭐ **rl-s0 nearly doubles the off-road rate, separated.** ⚠️ **rl-s1 does not**, so
DAC-blindness cannot be seed 1's cause.

## ⭐ Test 2 — seed 1 does not leave the road, it drives too fast

The four-family paired read localises the damage differently per seed: **s0 in LATERAL**
(`LAT_heading_mae_deg` −0.9919), **s1 in LONGITUDINAL** (`LON_speed_mae_mps` −0.2795,
`LON_accel_mae_mps2` −0.3114). EP rewards **progress along the human's own future path**, and
nothing in the proxy penalises exceeding the speed the situation warrants.

Mean plan speed (waypoint separations over `dt` = 0.5 s), 1,402 windows, 41 episodes:

| arm | mean speed | signed bias vs human |
|---|---|---|
| human | 11.503 m/s | — |
| **base** | 11.548 | **+0.045** |
| **rl-s0** | 11.513 | **+0.010** |
| **rl-s1** | **11.920** | **+0.417** |

| paired comparison | Δ | CI | |
|---|---|---|---|
| rl-s0 − base | −0.0353 | [−0.0925, +0.0249] | not separated |
| **rl-s1 − base** | **+0.3718** | [+0.2708, +0.4777] | ⭐ **SEPARATED** |

## ⭐⭐ Why this is a discriminating result and not a story fitted to noise

**Each seed is separated on its OWN failure and NOT separated on the other's.**

| | off-road | over-speed |
|---|---|---|
| **rl-s0** | ⭐ **SEPARATED** +0.0151 | not separated −0.0353 |
| **rl-s1** | not separated +0.0015 | ⭐ **SEPARATED** +0.3718 |

A single confound acting on both arms could not produce that pattern. ⇒ **Two distinct failure
modes, one per seed, each traceable to a reward term that carried no information.**

| seed | measured failure | the blind reward term |
|---|---|---|
| **rl-s0** | off-road rate nearly doubles; LATERAL heading damage | **DAC ≡ 1** — no drivable-area constraint at all |
| **rl-s1** | over-speeds +0.372 m/s vs base; LONGITUDINAL damage; **not** more off-road | **EP** rewards raw progress with no speed-appropriateness term |

## ⛔ Controls, and the one that failed first

* **The human's own driven path is the gold control** and it had to read small. **On the first run
  it read 0.1741** — and that correctly voided the whole comparison. The cause was mine:
  `gt_future_ext` is in the **episode** frame while `os`/`ha0` are **ego frame at t0**
  (MEASURED: `gt[0] − pose_last` = `v0 × 0.1` m). Transformed, it reads **0.0171**, a **10×**
  difference. ⇒ The arm deltas were only read **after** the control passed; the verdict is now
  gated on it in code (`CONTROL_BAR = 0.05`) so a broken instrument cannot print a verdict.
* **Every index convention was pinned by measurement, never assumed:** `ha0` reads exactly `v0·t`
  with `y ≡ 0` ⇒ metres, ego frame, +x forward / +y **left**, matching the SAM3 grid's declared
  convention; `pose_last[w] == ep_poses[ws[w]]` at residual **0.000000** ⇒ `ws` **is** the NOW row
  (an earlier `+7` guess was wrong); `ep_poses == poses[2:]` at residual **0.000000** ⇒ raw frame
  = `ws + (n_stack − 1)`.
* **The episode → clip mapping is self-verifying twice:** the payload's own poses must reproduce
  `ep_poses`, and `ClipMapGT.check_pose_alignment` must return `ALIGNED`. **4 of 41 episodes were
  REFUSED** by those checks and excluded — 3 `INCONCLUSIVE` (an ego that barely moves cannot
  identify the frame axis) and 1 `TimeMisalignment` at 0.086 m > the 0.05 m tolerance. Worst
  surviving residual: **0.0055 m**.

## ⛔ What this does NOT establish

* ⛔ **No fix is claimed to work.** This is a diagnosis. The repaired arm is a **new
  pre-registration**, not a re-run, and it has not been run.
* ⚠️ **n = 2 seeds.** A separated CI answers *"would another draw of EPISODES say this?"* — never
  *"would another TRAINING RUN say this?"*. Each mechanism rests on **one** seed showing it.
* ⚠️ The EP → over-speeding link is **the mechanism most consistent with the measurement**, not a
  measured causal chain: what is MEASURED is that s1 over-speeds and is not more off-road.
* ⚠️ The off-road test is a **point test**, a lower bound; the true DAC is a footprint test.

## What this changes

D9 moves from *"refuted, park it"* to **two named defects with two named fixes**, and both inputs
now exist: **SAM3 maps cover 135 of the 139 eval clips** (measured today) and the corpus run
finishes ≈2026-09-22, so **DAC can be made live**. A speed-appropriateness term is a reward change,
not a corpus dependency.

## Artifacts

`raw/dac_attrib.json` · `raw/speed_attrib.json` · `code/dac_attrib.py` · `code/speed_attrib.py`.
Clip identifiers appear only as **sha12**.

## ⛔⭐ ADDENDUM, same day — the root cause is SHARPER, and it changes the schedule

**The claim above said DAC was ≡ 1 *because the RL-train split had no SAM3 maps*. That is
true about the data and WRONG about the binding constraint.**

`score_candidates` takes DAC as **optional keyword arguments**:

```python
def score_candidates(cand_states, human_states, agents, route, *,
                     dac_cand=None, dac_human=None, cfg=PROXY):
    ...
    dac = ones.clone()
    if dac_cand is not None:  dac[1:] = dac_cand
    if dac_human is not None: dac[0]  = dac_human
```

⛔ **And no caller anywhere supplies them.** Both call sites in `ddv2_rl_refcv5.py`
(:231, :300) pass exactly four positional arguments — `(states, human, tracks, route)`.
A repo-wide search finds `dac_cand` / `dac_human` **only inside `pdm_proxy.py`'s own
signature**, and `dac_from_drivable` called **only by `pdm_proxy.py` itself and its test**:
⭐ **it has never had a production caller.** The RL trainer contains **no map machinery at
all** (`semantic_map_gt` / `perception_targets` / `sam3`: zero hits across
`tanitad/rl/` and `scripts/ddv2*.py`, with `score_candidates` reading non-zero in the
same breath as the positive control).

⇒ **Maps alone would NOT have fixed it.** On a fully mapped corpus, this code still
computes `dac = ones` and the road-boundary term is still algebraically absent.

⚠️ **And the damage is wider than the final multiplier.** `multi = nc * dac` feeds
`raw = ego_progress(...) * multi`, so DAC ≡ 1 also removes drivability from the **EP
normalisation**: a candidate that leaves the road without colliding receives **full
progress credit**. The missing term was gating progress, not just scaling the total.

### What changes

* ✅ **The measurements stand unchanged** — rl-s0's separated off-road increase and rl-s1's
  separated over-speeding are observations about the arms, not about why DAC was constant.
* ⭐ **The schedule changes.** `H-DDV2RL-3` was gated on SAM3 covering the **RL-train**
  split (≈2026-09-22). That gate is **necessary but NOT sufficient**: the plumbing must be
  built, and it can be built and tested **now** against the eval-139 maps, which already
  exist (135/139).
* **ROOT-CAUSE CLASS (mine):** *a docstring's conditional read as this run's cause.* The
  module says DAC is *"1 when no map exists"*; I checked that no map existed and stopped,
  without checking whether the caller ever passes it when one does. Same family as the
  entry logged hours earlier the same day — **a number read as a mechanism** — with the
  object swapped for a documented default.

<!-- D9-ADDENDUM-NO-PRODUCTION-CALLER-2026-09-17 -->
