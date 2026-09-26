# PRE-REGISTRATION — refcv4b on NAVSIM v2 `navhard_two_stage`

**Written 2026-09-20 16:53 Europe/Berlin, while the run is at scene ~300/5,462 of A1's INFERENCE and
⛔ BEFORE ANY ARM HAS BEEN SCORED.** No `scores/*.csv` exists in the run directory at the time of
writing; the only navhard scores on this box are the banked **floors** run
(`…/20260920T082848Z-navsim_v2-none-06e257`), whose CV/STOP values are quoted below *as the prior*.

⭐ Why this file exists: `CLAUDE.md` rule 5 — *settle conflicts with experiments, pre-registered with
BOTH outcomes committed in advance, not a scoped-down goal.* E2 named this bar for navhard in its own
"next steps" table (row 5) after measuring the warmup result; it is adopted here verbatim rather than
invented after seeing data.

---

## 1. The bar

> **BAR-W7-1:** on `navhard_two_stage`, refcv4b's **A1** arm (`frames ST + measured t0 ego + NavSim
> driving_command`) beats **every** floor on the **stage-2 statistic S2-EPDMS-u**, computed on the
> IDENTICAL 5,462 stage-2 tokens:
>
> **`A1 > max(CV, STOP, ECHO)`**

**Statistic.** `S2_EPDMS_u` = the mean over the split-yaml mapping groups of the uniform within-group
mean of the official stage-2 `score` column (`summarize.py::s2_group_uniform`). ⛔ **It is NOT a
two-stage EPDMS** and must never be quoted as one. The headline column is `score`; ⛔ `pdm_score` is
forbidden.

**Why the stage-2 statistic and not EPDMS.** Because `raw/stage_frame_census.json` establishes, by
count and with a same-breath control, that **0 of 5,400** stage-1 camera jpgs are UNPACKED anywhere
this run can read them, while **65,544 of 65,544** stage-2 jpgs are. A camera arm's stage 1 is
therefore a declared CV stand-in and its official two-stage EPDMS is **UNDEFINED for this run**. That
fact was established BEFORE this bar was written, and it is the reason the bar names a stage-2
statistic.

> ⚠️ **SCOPE AMENDMENT (2026-09-20, after the bar was written, and it does NOT move the bar).** The
> frames are on this box **inside the sha256-verified OpenScene test camera tarballs** (32 shards,
> 127,882,665,618 B): shard 6 carries **120/120** of the stage-1 files for both navhard logs it
> holds, complete per log, with a stage-2 control reading **0/1,185**
> (`raw/stage1_in_archives_shard6.json`). ⛔ They are **not extracted**, so the arm being scored in
> THIS run still has no stage-1 frames and its two-stage EPDMS is still UNDEFINED — the bar, the
> statistic and the tokens are unchanged. ⭐ Recording the amendment here rather than quietly editing
> the paragraph is the point: a pre-registration that is edited after the fact is not one.

**Interval.** The settled `navsim_log_cluster_bootstrap` (cluster = `log_name`, B = 2000, seed 0,
RG-14 floor of 8 clusters), **paired** for two arms on the same tokens. ⛔ `overlapping_holdout_se` is
forbidden. If the pre-CSV frame is not captured, the interval is
`{status: UNAVAILABLE, reason, n}` — never a substitute estimator.

---

## 2. Both outcomes, committed now

| outcome | what is written | what happens next, in the SAME turn |
|---|---|---|
| **PASS** — `A1 > max(CV, STOP, ECHO)`, separated | "refcv4b beats every floor on navhard stage 2" + the decomposition that says *why* | publish the row; name the weakest sub-metric as the next lever |
| **FAIL-A** — `A1 > CV` but `A1 <= STOP` | ⛔ **"refcv4b does not beat a stopped car on navhard."** Reported plainly, in the headline, not in a footnote | RULE ZERO: run `code/decompose.py` and report WHERE it loses — sub-metric, stage, speed band, log, and the zero-attribution over the five multipliers |
| **FAIL-B** — `A1 <= CV` | ⛔ "refcv4b does not beat constant velocity on navhard" — a **worse** result than warmup's | same decomposition, plus the A1−ECHO contrast (does the model add anything over an echo of its own ego inputs?) |
| **UNDEFINED** — an arm fails to score | reported as FAILED with its counts; ⛔ never as a missing row | re-run only the failed arm's cause; a completed-but-unaggregated run is re-aggregated, never re-scored |

⛔ **The bar is not moved after seeing the data, no control is dropped, and a post-hoc finding gets
its own pre-registration or it is not promoted.**

---

## 3. The prior — what a like-for-like result would look like (INHERITED, stated as such)

**E2 on `warmup_two_stage`, 204 stage-2 scenes** (INHERITED from
`…/2026-09-19-navsim-refcv4b-bridge/RESULT.md`, **not** re-verified by me):

| arm | S2-EPDMS-u (warmup) |
|---|---|
| STOP | **0.5212** |
| A2 vision-pure | 0.5211 |
| A1 `ego+cmd` | **0.4670** |
| CV | 0.3971 |

⇒ On warmup, **A1 beat CV (+0.0699) and LOST to STOP (−0.0542)**. **FAIL-A is the outcome this
pre-registration EXPECTS**, and saying so in advance is the point: a bar you expect to fail is the
only kind whose failure teaches anything.

**navhard floors, MEASURED, banked run `…-06e257`** (the arms my run re-scores on identical tokens):

| arm | official two-stage EPDMS | stage-2 scene mean |
|---|---|---|
| **STOP** | **0.2985** | 0.4702 |
| **CV** | **0.1148** (= the official leaderboard's 11.4816 — reproduced exactly) | 0.3294 |

⚠️ **STOP beats CV on both generations** (warmup EPDMS 0.3009 vs 0.1854; navhard 0.2985 vs 0.1148).
**A stopped car is not a weak baseline on this benchmark**, and an arm that only beats CV has cleared
the easier of the two floors.

---

## 4. Controls that must read a known value (checked whatever the outcome)

1. **CV must reproduce the official leaderboard.** navhard CV = `0.11481608441648`; the run's
   `reference_check.external.reproduced` must be **true**. A CV that drifts invalidates the
   *harness*, not the model, and nothing else in the run is quotable until it is explained.
2. **CV and STOP must reproduce the banked floors run bit-for-bit** on the same tokens
   (`controls.reference_check`). They are scored again only so the paired comparison is
   within-run; if they disagree with the bank, the two runs are not one surface.
3. **C4 formula control** (the recomputed EPDMS equals the devkit's, `max_abs_diff <= 1e-9`).
4. **Stand-in count read from the SEAM ARTIFACT, not from a report** — A1 must show exactly **450**
   `cv_standin` rows and **5,462** model rows, and the agent-call log must match the seam
   declaration. A mismatch is a scoring FAIL, not a warning.
5. **`criteria/A1.txt` must read 0 violations.** If a gate fires, the run is fixed, not the gate.

---

## 5. What this number will NOT be, whatever it says

* ⛔ **Not a two-stage EPDMS** (stage 1 is a stand-in — §1).
* ⛔ **Not closed loop.** NavSim is T1-family: stage 1 **OPEN**, stage 2 **UNRULED** (a PI ruling is
  pending and is ⛔ not guessed here); background vehicles are IDM-reactive, the ego is not.
* ⛔ **Not evidence of route or strategic skill.** A1 consumes `driving_command`, which W2/E3
  measured to be a **ROUTE-LEVEL ORACLE** (reproduced 1,902/1,902 by OpenScene's own function; on
  stage 2 it is COPIED from the expert's own frame, 5,462/5,462 on navhard). A command-conditioned
  NavSim number is *driving with an oracle route* — comparable within NavSim, never a route claim.
* ⛔ **Not zero-shot-adjusted.** refcv4b has **no NavSim training**; this is a transfer number, and
  no correction for that is applied or implied.
* ⚠️ **The interval answers ONE question: "would another draw of SCENES/LOGS say this?"** It is
  structurally blind to training variance (`H-ESTIM-SEED-1`). The **inference**-variance question is
  closed by construction for this arm: `refc.py:1599` sets the decoder noise to `torch.zeros_like`
  outside training, so refcv4b is deterministic at inference (`MODEL_REGISTRY.md` §4.6).
  A separated CI here is **necessary, not sufficient**, for any claim that a LEVER moved a metric —
  and this run moves no lever, it places one arm against three floors.
