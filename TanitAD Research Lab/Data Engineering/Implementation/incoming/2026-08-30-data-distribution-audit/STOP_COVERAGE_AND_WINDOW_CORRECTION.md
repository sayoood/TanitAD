# ⛔ Correction to D-INERTIA-B1 — I measured the wrong window — plus stop-launch coverage

**Date** 2026-08-30 · **Owner** DataFlyWheel · Corrects **D-INERTIA-B1**
(committed `d415bddb5`) · Blob `ee44875916ae7c0ac002c6716b9658ea`

## The error

**The egomotion parquet spans a median 139.3 s. The camera clip the model sees is
20 s. I computed the inertia statistic over 7× more driving than the model is
ever shown.** Every number in D-INERTIA-B1 describes a window that does not exist
in training.

MEASURED: egomotion rows median 2,828, span median **139.3 s** (min 20.2, max
140.4) against a **20 s** camera clip.

**ROOT-CAUSE CLASS — this is C82 again** (*price the artifact the CONSUMER
reads*), in its fourth costume tonight: I read the file I had rather than the
window the trainer consumes. The recognition signal was available and I walked
past it — I had already noted the parquet was ~3,131 rows while a 20 s clip at
10 Hz is ~200.

## Corrected numbers — on the 20 s trainable window

| statistic | published (139 s, WRONG) | **corrected (20 s)** |
|---|---|---|
| stopped-frame fraction, mean | 5.6 % | **6.8 %** |
| stopped-frame fraction, median | 0.0 % | 0.0 % |
| clips with **NO** stopped frames | 61.3 % | **76.1 %** |
| any stop | 38.7 % | **23.9 %** |
| clips > 50 % stopped | 1.7 % | 3.0 % |

⇒ **The inertia conclusion STRENGTHENS.** Codevilla's failure needs
over-represented stopped frames to build a spurious *low-speed → no-acceleration*
correlation. On the window the model actually sees, **76.1 % of clips never stop
at all** — even less mass for it to form on than I reported. **The direction was
right; the magnitude was wrong in our favour.**

## ⭐ Stop-launch coverage — and my flip-side warning was RIGHT, more so than I said

The Master Mind asked for the TAIL, not the mean: how many clips contain a full
stop-and-launch cycle at all — the competence we would be unable to learn.
Measured with the programme's own detector (`ego_manoeuvre.stop_episodes`,
`V_STOP_MS = 0.5`, `V_CRUISE_MS = 5.0`), on the 20 s window:

| competence | clips | share |
|---|---|---|
| any stop | 1,129 | 23.9 % |
| ⭐ **full STOP → LAUNCH cycle** | **558** | **11.8 %** |
| **stop-and-go (≥ 2 stops)** | **115** | **2.4 %** |

*(Over the full 139 s egomotion these read 33.3 % and 12.4 % — which is why the
window matters: measured on the wrong span, stop-launch coverage looks
comfortable. It is not.)*

⇒ **We hold 558 clips that contain a complete stop-and-launch, and 115 with
stop-and-go.** That is thin for a competence every urban driver exercises daily,
and it is the axis where "add stop-rich clips" is the fix — the OPPOSITE
direction from what a Codevilla reading would suggest.

## ⚠️ What this does NOT establish

Our records carry **"0/881 accelerate"** and **"no arm beats hold-v0 at
cruising"**. Thin stop-launch coverage is a *plausible* second explanation for
those symptoms, and it is **independent** of the inertia mechanism, which is
measured absent.

⛔ **It is not evidence of that link, and I am not asserting one.** A coverage
number and a symptom are not a cause. Establishing it needs its own experiment —
the obvious one being an equal-size, equal-compute arm enriched for stop-launch
clips, scored on longitudinal control. That is exactly the shape of the DDS
validation already planned, so it costs one extra arm rather than a new
programme.

## Deliverable manifest

| artifact | where |
|---|---|
| `raw/b1_stop_coverage_20s.parquet` — per-clip, 20 s window (the admissible one) | repo |
| `raw/b1_stop_coverage.parquet` — per-clip, full egomotion span (retained to show the delta) | repo |
| `code/measure_stop_coverage.py` | repo |
