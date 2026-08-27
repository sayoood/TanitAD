# P7 TanitEval + P9 TanitResim — the VISION

`Drafted 2026-08-23 by TanitAD_EvalFlyWheel at the PI's request ("analyze the current state of
TanitEval, TanitResim and develop the vision together"). ⚠️ This is a PROPOSAL. §6 lists the
decisions that are the PI's, not mine. Everything else is already being implemented.`

---

## 0. The one-sentence vision

> **TanitEval is the instrument that decides whether TanitAD drives — and TanitResim is how a
> human sees why it does or doesn't.** Together they are the programme's truth layer: nothing
> becomes a claim without passing through them, and nothing passes through them without stating
> its tier, its estimator, its scope, and its gaps.

The two are one system. An eval that produces a number nobody can see the behaviour behind
produces folklore; a video that is not tied to a scored artifact produces a demo. TanitEval emits
the artifact; TanitResim renders **that same artifact**, never a re-run.

---

## 1. Where we actually are (MEASURED 2026-08-23, not asserted)

The honest surprise of this audit: **the instruments are far better than the programme believed,
and the reporting layer is far worse.**

| | state |
|---|---|
| **Four families** | ✅ ALL implemented. LONGITUDINAL incl. headway/time-gap/min-TTC with a strictly-causal lead selector and a three-state LEAD/NO_LEAD/**NO_LABEL** contract. LATERAL incl. heading, curvature+bias, yaw-rate, cross-track. TACTICAL incl. κ, per-class confusion, factored lat/lon, `never_predicted`. STRATEGIC implemented but corpus-blocked. |
| **Estimator** | ✅ `ci.py` is sound: `episode_cluster_bootstrap`, paired form, B=2000, `full_set` point estimate. |
| **Emission** | ⛔ **This is the real gap.** 93 of 135 in-scope artifacts carry NO tier stamp. Only 4 are T1. Half the LON/LAT components carry no interval. |
| **Reporting** | ⛔ Two of my own claims this session were config errors that read as programme gaps (C133, C133-b). |
| **Benchmarks** | ⛔ None runnable. NavSim + nuScenes protocols now banked and specified; adapter in progress. |
| **TanitResim** | ✅ Exists as a tested web app (`stack/tanitad/resim/`) + 3 renderer families. ⛔ Renders a GT-derived input in the strategic *prediction* slot. |

**The measurement that should worry us most** is not a missing metric. It is this: on T1, the
`stage-a-repaired · cl` arm reads **9.3697 m** against its own hold-action control at **0.4246 m** —
the trivial control beats the model **22×**, on the same windows and the same checkpoint whose T0
number is 0.3659. And the tactical head predicts the longitudinal axis at **κ = 0.0405 — chance**,
while the collapsed 5-way softmax (κ 0.1404) reports neither axis.

⇒ **We have been reading a world-model fidelity diagnostic as driving skill.** The instruments were
telling us this; nothing was aggregating it into a verdict. That is exactly what P7 must become.

---

## 2. What TanitEval becomes

### 2.1 The principle: an eval is a CONTRACT, not a script
Every eval run emits an artifact that is self-describing and machine-checkable: tier, scope,
estimator, n, the four families, what it refused and why. `tools/criteria_check.py` already
enforces this. The pipeline must be unable to emit an artifact that fails its own check.

### 2.2 The four layers

**L1 — The harness** (exists, needs completion). One entry point, `--tier` mandatory, sparse and
dense surfaces both handled and *recorded*, intervals on every reported component. → in progress.

**L2 — The criteria registry** (exists, v2.1.0). The binding criteria as DATA, not prose, validated
against its own emitters. Extended per benchmark. → done, keeps growing.

**L3 — The Release Gate** (in progress). One command, one verdict, per-family, PASS/FAIL/
**CANNOT-RULE**. ⭐ The CANNOT-RULE state is not a nicety: this programme has shipped a gate that
could never rule (O6 rank, spectrum n=24 vs ceiling 1024) and read its silence as a pass.

**L4 — The Leaderboard** (rebuilt). Our arms + external stacks, every row carrying tier, estimator,
interval, and protocol. External rows use *their* published protocol, named — never ours relabelled.

### 2.3 Community benchmarks — adopt selectively, and say why
- **NavSim**: adopt. Read `score`, never `pdm_score`. ⚠️ The estimator does not transfer (scene
  token ≠ episode) — this is an open design question, not a detail to paper over.
- **nuScenes detection / tracking / prediction**: adopt unreservedly. Devkit-enforced, labels are
  not functions of inference inputs.
- **nuScenes open-loop planning**: ⛔ **never as a criterion or gate.** External-comparability row
  only, and only with its GT-trajectory control published alongside. VAD's "high-level command" is
  the ground-truth future thresholded at ±2 m and fed back as input — our own route-echo defect,
  published as SOTA. Adopting that protocol as a target would be adopting the defect.

⚠️ **Consequence the PI must see coming**: our vision-only rule means we *structurally cannot*
produce the configuration that generates the published headline numbers. Our NavSim/nuScenes rows
will look worse than UniAD's 0.46. The correct comparison is against ego-free rows (UniAD 1.03,
VAD 1.25, BEV-Planner 0.55). This must be stated on the leaderboard itself, or our own row will be
misread as underperformance — including by us.

---

## 3. What TanitResim becomes

### 3.1 The principle: render the ARTIFACT, never a re-run
TanitResim takes a scored eval artifact and shows the behaviour behind the number. If the picture
and the number can disagree, the picture is decoration. One consequence: TanitResim's input
contract is TanitEval's output contract — the same seam, in both directions.

### 3.2 The viz standard as an enforced contract, not a convention
Camera overlay + metric BEV inset + decoded tactical manoeuvre + strategic route/goal (+ADE),
together — with a BEV-only fallback when calibration is unrecoverable. Enforced in code: a frame
with a silently missing element must not render. An unavailable element is DRAWN as
`unavailable: <reason>`. This is the per-family reporting discipline applied to pixels.

⛔ **And a prediction slot may only ever show a prediction.** The current SPA renders `nav_cmd` — a
GT-derived input — in the strategic slot. Two fields, always: `route (model)` and
`route (logged input)`.

### 3.3 The three modes
- **CLI** — batch render for a campaign, consumed by the release gate; the artifact of record.
- **UI** — scrub an episode, compare two arms side by side, jump to worst-ADE windows, filter by
  tactical class. ⭐ *Jump to the windows where the model and the control disagree most* is the
  highest-value view we do not have: it is how a human sees a 22× control gap instead of reading it.
- **Embedded** — the same renderer inside eval reports and the paper, so figures cannot drift from
  artifacts.

### 3.4 Consolidation
Three renderer families implement one contract three times (PIL, JS/canvas, cv2), each strongest in
a different place: `corpus_overlay` has the only working BEV-only fallback, `overlay_video` the only
true f-theta camera model and in-frame lat/lon families, `render_openloop_video` the only portable
packaging. **The core is assembled from three; no single file is the merge target.** Extract the
shared contract first, migrate one renderer, prove it, then the others.

---

## 4. How the two fit together

```
  checkpoint ──▶ L1 harness ──▶ ARTIFACT (tier, families, estimator, refusals)
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
              L3 release gate   L4 leaderboard   P9 TanitResim
              PASS/FAIL/        row w/ tier +    the behaviour behind
              CANNOT-RULE       estimator        the number
```
One artifact, three consumers, no re-runs, no hand-copied numbers.

---

## 5. Sequence (my proposal)

| # | step | why now | status |
|---|---|---|---|
| 1 | intervals on every component; `--tier` mandatory | half the families are un-quotable without it | in progress |
| 2 | release gate + regression arms | the PI asked for exactly this | in progress |
| 3 | TanitResim: fix the prediction-slot defect; land the viz contract | a correctness bug outranks features | in progress |
| 4 | NavSim adapter (synthetic-tested) | first external comparability | in progress |
| 5 | close the forbidden estimator, incl. the unnamed clone | a live biased estimator under the guard's radar | in progress |
| 6 | re-run every banked arm through the completed pipeline | turn 93 unstamped artifacts into quotable ones | next |
| 7 | TanitResim consolidation + the disagreement view | after the contract holds | next |
| 8 | first real NavSim run | needs a PI data decision (§6) | blocked |

---

## 6. ⭐ DECISIONS THAT ARE YOURS, NOT MINE

1. **NavSim data.** A real run needs the dataset (100s of GB) and the estimator question settled.
   Download and where? I will not start a metered or large transfer without your word.
2. **nuScenes licence conflict.** Paper says CC BY-NC-SA 4.0 (non-commercial); AWS Open Data says
   "Commercial". I am treating it as RESEARCH-ONLY until you obtain written terms.
3. **The 68 stranded MP4s** on one consumer disk, unrecoverable from git. Tracked text manifest, or
   force-add the binaries, or an external store?
4. **Does the release gate BLOCK a release, or advise?** My recommendation: blocking on
   completeness/tier/estimator/leak-guards (cheap, objective, and each earned by a real failure),
   advisory on regression until we have two clean releases to compare.
5. **Do we publish a NavSim/nuScenes row that looks worse than SOTA** because we refuse ego inputs?
   My recommendation: yes, loudly, with the ego-free comparison shown — it is a methodological
   contribution, not a weakness. But it is your call how it is framed publicly.

---

## 7. What I got wrong today, kept here on purpose

I twice reported an eval gap that was a defect in my own instrument (C133: measured one directory,
stated it repo-wide; C133-b: named a key the emitter never writes). Both reached the PI before they
were caught. The lesson is now machinery, not memory: the census prints its scope, and refuses to
run without first checking that every criterion resolves somewhere.

⚠️ It is worth stating plainly, because it shapes this vision: **the programme's eval problem was
never mainly missing instruments. It was that nothing checked the instruments against reality, and
nothing aggregated their output into a verdict.** That is what P7 is now being built to be.
