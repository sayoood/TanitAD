# Situation labels — lane change + intersection, emitted. No objects.

**PI 2026-07-29:** *"the classifier should detect the situation of lane change not necessarily based
on objects — only on situational labels, so you don't need objects. You need the detection of lane
changes and an intersection label."*

⇒ `stack/scripts/emit_situation_labels.py` produces exactly those two labels. **No object features,
no `obstacle.offline`, no detections** — I had proposed that and it was over-engineering.
Roundabout is computed but **not emitted** (26 held-out clusters = UNPOWERED).

## 1. The discipline that makes these usable — the PI's rule, written into the script

> **Future ego motion is legitimate for GENERATING and VALIDATING labels, and illegitimate as a
> model INPUT.**

The detectors read the whole pose track *including the future* — that is what makes them ground
truth. **A classifier trained on these must take the CAMERA** (plus the ego's own instantaneous
state, which a vehicle genuinely has at inference) and must **never** be fed the pose track the
labels came from. The script's docstring carries this so it cannot be lost.

The target is **ANTICIPATION, not recognition**: `y(t) = 1` iff an onset falls in `(t, t+lead]`, and
frames *inside* an ongoing situation are masked via `valid_*`. A classifier therefore cannot score
by noticing a manoeuvre already under way.

## 2. First emission — 40-episode val subset (`situ_val40.npz`)

| situation | episodes with | base rate | positive frames | scorable frames |
|---|---|---|---|---|
| **lane change** | 5 / 40 (**12.5 %**) | **0.01824** | 120 | 6,580 |
| **intersection** | 7 / 40 (**17.5 %**) | **0.03704** | 237 | 6,399 |

7,964 frames total; the gap to "scorable" is the in-situation masking.

## 3. ✅ THE DISCREPANCY IS RESOLVED — two detector GENERATIONS, not a broken promotion

The first emission (40 val episodes) gave lane change **12.5 %** / intersection **17.5 %** against a
spec table (`H2_SUBSTRATE_AND_LABELING.md` §E.2) reporting **39.07 %** / **28.20 %**. I named two
candidate causes — a manoeuvre-poor subset, or a promotion that changed behaviour — and ran the
discriminating test: **400 TRAIN episodes**.

| | val 40 | **train 400** | spec §E.2 |
|---|---|---|---|
| lane change | 12.5 % | **9.25 %** (37/400) | 39.07 % |
| intersection | 17.5 % | **20.25 %** (81/400) | 28.20 % |

**Neither candidate was right.**
- **NOT a subset effect:** at n=400 the intervals are ~[6.6 %, 12.5 %] and ~[16.5 %, 24.5 %]; the
  spec values lie outside both. The train corpus itself does not produce 39 %.
- **NOT a broken promotion:** the constants themselves differ between the two sources —

| threshold | spec §E.2 prose | pre-registered module (promoted) |
|---|---|---|
| net \|Δψ\| | ≤ 3° | `LC_DPSI_MAX = 8.0°` |
| lateral band | 2.5–5.0 m | `LC_LAT_MIN/MAX = 2.4 / 5.5 m` |
| yaw lobes | ≥ 2° | `LC_LOBE_DEG = 1.5°` |

⇒ **§E.2 is an EARLIER detector generation** (`situ_full.py`); `sc_situations.py` is the later one
whose constants `PRE_REGISTRATION.md` §2 **froze** before the study ran. The promotion reproduces the
**pre-registered** detector faithfully — which is the one the measured A− verdicts belong to.

⛔ **CONSEQUENCE, and it is the reusable part: §E.2's 39.07 % / 28.20 % MUST NOT be quoted as the
prevalence of these labels.** They describe a detector that no longer defines the situations.
Quoting them beside the classifier's AP numbers would mix generations — the same class of error as
mixing estimators. **The prevalence of the pre-registered labels is 9.25 % / 20.25 % of episodes
(n=400 train), base rates 0.01439 / 0.03558 per scorable frame.**

⚠️ **This also revises the power picture.** The study's own held-out cluster counts (153 lane change,
264 intersection) stand — they were computed from these detectors. But anyone sizing a NEW experiment
off §E.2 would over-estimate available positives by ~4× for lane change.

## 4. Provenance

`stack/scripts/emit_situation_labels.py` · `stack/tanitad/data/situations.py` (promoted, 9 tests
passing + 2 xfail fixtures) · `raw_situ_val40_summary.json`.
