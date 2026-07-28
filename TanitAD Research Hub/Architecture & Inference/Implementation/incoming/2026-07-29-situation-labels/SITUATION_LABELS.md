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

## 3. ⚠️ A DISCREPANCY that must be resolved before these are trusted at scale

The corpus-wide spec (`H2_SUBSTRATE_AND_LABELING.md` §E.2) reports **lane change 39.07 %** of
episodes and **intersection 28.20 %**. This emission gives **12.5 %** and **17.5 %**.

With n=40 and 5 hits the 95 % interval on the lane-change rate is roughly **[4 %, 27 %]** — **39 %
lies outside it**, so this is **not** small-sample noise.

**Two candidate causes, neither yet tested:**
1. this 40-episode val subset is genuinely manoeuvre-poor relative to the full corpus, or
2. the module promoted into `stack/` differs in behaviour from the `situ_full.py` that produced the
   corpus figures.

⛔ **Cause (2) would matter a great deal** — it would mean the promotion was not behaviour-preserving
and the measured study's counts do not describe these labels. **The cheap discriminating test:** run
the emitter over a few hundred TRAIN episodes and compare the rate against 39.07 % / 28.20 %. If the
rates match there, the subset explains it; if they do not, the module does. **Do this before any
classifier is trained on these labels.**

## 4. Provenance

`stack/scripts/emit_situation_labels.py` · `stack/tanitad/data/situations.py` (promoted, 9 tests
passing + 2 xfail fixtures) · `raw_situ_val40_summary.json`.
