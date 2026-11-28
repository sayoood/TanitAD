# The window-alignment defect has a mechanism — and the fix is NOT to move our window

`MEASURED (ours)` · 12 clips (every clip holding BOTH a local `v2ep` pose file
and an Alpamayo taxonomy row) · ego speed read from poses, never from a label ·
artifacts `window_align.json`, `window_align_profiles.json`, `ego_tier_demo.json`.

⚠️ **n = 12.** Every number here is a direction to test at scale, not a corpus
statistic. The 130-clip ego cache and the 4,729-row taxonomy are near-disjoint
subsets — **exact and 8-char-prefix intersections are both 12**, so this is a
real sample limit and not an id-format bug (probed twice, per the absence rule).

---

## 1. The mechanism, which was previously only a mismatch count

Our clips are **T = 201 frames = 20.1 s at 10 Hz**. The payload's window opens at
`mid = 100` (**t ≈ 10 s**) and runs to **t = 16 s**. Alpamayo's `longitudinal`
magnitude describes **the clip**, and for decelerate-to-stop clips the entire
deceleration is **over before our window opens**:

| clip | Alpamayo | v(t) sampled every 1.6 s |
|---|---|---|
| `1f57d8ad` | Strong Deceleration | 5.89 → 6.20 → 6.71 → 6.13 → 5.34 → 2.43 → 1.14 → **0.07 → 0.0 → 0.0 → 0.0** → 0.12 → 0.72 |
| `20070bb6` | Gentle Deceleration | 9.84 → 7.92 → 6.61 → 5.25 → 3.81 → 2.44 → 1.47 → **0.58 → 0.0 → 0.0 → 0.0 → 0.0 → 0.0** |
| `c568fffd` | Strong Deceleration | 11.62 → 10.27 → 9.02 → 7.97 → 5.92 → 2.56 → 0.71 → **0.03 → 0.0 → 0.0 → 0.0 → 0.0 → 0.0** |

⇒ By `mid` these vehicles are **already stopped**. Our window sees the
aftermath; Alpamayo's label describes the approach. **The two are not
disagreeing about the same interval** — they are describing different ones.

## 2. Sign agreement across five candidate windows

Ego `dv` sign vs the magnitude's implied sign. **The dead-band is an estimator
choice and it moves the numbers, so both readings are published:**

| window | agree, dead-band ‖dv‖ < 1.0 m/s | agree, strict sign |
|---|---|---|
| **first half** (0 → mid) | **7/12** | **7/12** |
| first 6 s (0 → 60) | **7/12** | — |
| centred (mid−3 s → mid+3 s) | 5/12 | — |
| full clip (0 → T) | 4/12 | 6/12 |
| ⛔ **our window** (mid → mid+6 s) | **2/12** | **4/12** |

⭐ The **ranking is identical under both**: *first half ≫ full clip ≫ our
window*. Our tactical window is the **worst** of the five. The strict-sign column
reproduces the previously banked 7/12 · 6/12 · 4/12 exactly.

## 3. ⛔ Why re-aligning our window would be the WRONG fix

The obvious move — slide our label window back to the first half, where
agreement is 7/12 — **would label a different interval than the one the model is
asked to predict**. The tactical layer predicts `mid → mid+6 s`. Moving the
label to `0 → mid` buys agreement by measuring the wrong thing, which is the
same class of error as scoring a decoder on its marginal (C6).

⇒ **The defect is not that our window is misplaced. It is that Alpamayo's
magnitude is not a label for our window at all.**

## 4. What this does to the ego tier — and to the pending PI decision

`lon_from_ego` **fired 0 times on 12 clips**, and the reason is now measured
rather than assumed. Kinematics proposed `HOLD` on **3** clips; the magnitude
gate (`lon_is_admissible`) refused **all 3**:

| clip | Alpamayo | v0 | v_end | kinematics | gate |
|---|---|---|---|---|---|
| `1f57d8ad` | Strong Deceleration | 0.75 | **0.0** | HOLD | ⛔ refused |
| `20070bb6` | Gentle Deceleration | 1.26 | **0.0** | HOLD | ⛔ refused |
| `c568fffd` | Strong Deceleration | 0.46 | **0.0** | HOLD | ⛔ refused |

In all three the ego is **measurably at rest for the whole window**, and the
magnitude that vetoes it was measured over an **earlier interval**. The gate is
behaving exactly as written; **the input it gates on is out of scope**.

⇒ **Recommendation for the open PI decision** ("should ego override Alpamayo's
magnitude veto when they contradict?"): **on the magnitude axis, for our window,
yes** — and not as a preference but because the two quantities have different
supports. The clean form is not an override at all:

* **Alpamayo** keeps the **class** and the **reason/referent** (`cot`), which is
  what it is good at and what §3 of the confirmation memo measured.
* **the magnitude** for our window comes from **ego over that window**, where it
  is directly measurable.
* `lon_is_admissible` then gates on an **ego-derived magnitude on the same
  interval**, so it stops vetoing correct labels.

## 5. ⛔ The residual, which no window fixes

**4 of 12 clips agree with the poses on NO window tested:**

| clip | Alpamayo | dv full | dv our window |
|---|---|---|---|
| `4d58621f` | Stop | **+1.30** | +1.06 |
| `6c5c503d` | Gentle Deceleration | **+7.20** | +2.21 |
| `8dc5d14d` | Gentle Acceleration | **−8.07** | −4.41 |
| `bb41e3b8` | Gentle Acceleration | **−6.75** | −3.76 |

These are **outright sign contradictions against measured poses**, not alignment
artifacts. `bb41e3b8` is labelled *Gentle Acceleration* while decelerating
6.75 → 0.51 m/s. ⇒ Re-alignment closes part of the gap; **a residual
contradiction rate survives and must be carried as a known label-noise floor**,
not silently composed.

⚠️ Whether this rate is ~33 % (4/12 here) or the ~6 % measured corpus-wide on
`cot`-implied actions is **unresolved** — the two are computed over different
populations and different fields. **Do not quote either as "the" contradiction
rate.**

## 6. What must happen before the 4,522-clip run

1. **Re-derive the magnitude from ego on the labelled window**, and demote
   Alpamayo's magnitude from *gate* to *cross-check* (§4). This is a code change
   in `lon_from_ego` / `lon_is_admissible`, not a re-extraction.
2. **Measure the contradiction rate at scale** on a population where both ego
   and taxonomy exist — the current 12-clip overlap cannot settle §5.
3. **Keep the contradiction flag in the output.** A refused label with a reason
   is an asset; a silently composed one is not.

## 7. Manifest

| artifact | where |
|---|---|
| `window_align.json` (5 windows × 12 clips, both estimators) | scratchpad |
| `window_align_profiles.json` (per-clip speed profile) | scratchpad |
| `ego_tier_demo.json` (ego tier over the 12) | scratchpad |
| this document | `…/2026-08-19-alpamayo-screening/WINDOW_ALIGNMENT_DEFECT.md` |
