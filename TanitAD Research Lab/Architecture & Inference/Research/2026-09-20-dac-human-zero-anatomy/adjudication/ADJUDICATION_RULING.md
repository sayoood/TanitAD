<title>DAC adjudication ruling</title>

# The adjudication is INCONCLUSIVE FOR ADOPTION — and the reason is not that the rules failed

`Architecture & Inference · 2026-09-20 · Master Mind ruling on H-DAC-DEF-1, after the key was released · MEASURED, 0 GPU`
`Re-derived independently of the DataFlyWheel's scorer: code/rescore.py → the table below. Their numbers reproduce exactly.`

## 0 · Findings first

| # | finding | class |
|---|---|---|
| **1** | **The scoring reproduces.** Independently re-derived from the key and the landed labels: V0 agrees **10/60 (16.7 %)**, P1 **30/60 (50.0 %)**, P2 **50/60 (83.3 %)**; false-passes **0** for all three. Same as the DataFlyWheel's script, computed by different code. | MEASURED |
| **2** | ⛔ **VERDICT: INCONCLUSIVE FOR ADOPTION — not "no candidate clears".** §5.4's ≥ 95 % bar was written for a *representative* sample; §5.1 specifies a *disagreement-enriched* one. That is a contradiction **inside the pre-registration**, so the bar is unreachable by construction for any rule that ever fires. Reporting "no candidate clears" as a result would quote a bar outside its scope. | ruling |
| **3** | ⛔ **And the post-hoc reweighted column does not govern either.** Adopting on a column computed after the labels, chosen because it is the one that adopts, is the thing pre-registration exists to prevent. It is recorded, labelled post-hoc, and **adopts nothing**. | ruling |
| **4** | ⭐ **What holds under either reading:** 60 windows spanning every disagreement region produced **zero over-boundary**. Every firing of every candidate in this sample is a false alarm. | MEASURED |
| **5** | ⛔⭐ **NEW — the sample cannot rank the candidates at all, because it contains NO POSITIVES.** With 0 over-boundary labels, false-pass is **0/0** for every rule. ⇒ **a rule that simply always answers "on-surface" scores 60/60 = 100 % here**, beating P2's 83.3 %. The design enriched for *rule firing*, not for *actual departures*. | MEASURED (arithmetic on the table) |
| **6** | ⛔⭐ **NEW — P2's evidence base is 3 clips, not 10 windows.** Stratum C is **10 windows from 3 distinct clips**, one contributing **6 (60 %)**. Under the programme's own estimator doctrine the unit is the episode cluster. So the candidate the count flatters rests on the **smallest and most clustered** stratum. | MEASURED |

## 1 · The table, re-derived

| candidate | fires | false-fire | false-pass | agree | agree % |
|---|---|---|---|---|---|
| **V0** | 50 | 50 | 0 | 10 | **16.7 %** |
| **P1** | 30 | 30 | 0 | 30 | **50.0 %** |
| **P2** | 10 | 10 | 0 | 50 | **83.3 %** |
| *(a rule that never fires)* | *0* | *0* | *0* | *60* | ***100 %*** |

Label cells, kept apart and never folded: **A** 20/0/0 · **B** 20/0/0 · **C** 10/0/0 · **D** 10/0/0.

⭐ **The last row of that table is the ruling.** It is not a rhetorical device — it is arithmetic on the sample as drawn, and it says plainly that agreement on this pack is a measure of how *rarely a rule fires*, not of whether it is *right*.

## 2 · Clip clustering — the read the draw does not account for

The draw sampled **windows** (`random.Random(20260920).sample`), so *n* windows is not *n* independent observations.

| stratum | windows | distinct clips | largest cluster |
|---|---|---|---|
| A | 20 | 16 | 3 (15 %) |
| B | 20 | 13 | 3 (15 %) |
| **C** | **10** | **3** | **6 (60 %)** |
| D | 10 | 10 | 1 (10 %) |

**P2's 10 false fires are 3 clips** — one contributing 6, one 3, one 1. ⇒ P2 is **not** better-supported than its count suggests; per independent observation it is the **least**-supported of the three. ⚠️ This is the `overlapping_holdout_se` family again with the object swapped: the estimator is fine, the **unit** is wrong.

## 3 · ⚠️ An inconsistency in my own landed labels, and what I am NOT doing about it

`RESULT_MM_LABELS.md` reports **10** BORDERLINE rows; the CSV carries **9** — W18, W19, W25, W28, W43, W44, W50, W51, W60. The DataFlyWheel found it. Cause: my count came from a `grep -c` that also matched the **header line** describing the convention, and **W46** is named in the RESULT's list but its note says *"STATIONARY at a give-way line…"* without the token.

⛔ **I am not editing the landed CSV, because the key is now out.** Touching a label file after seeing the answers — even an annotation — destroys the property that made it worth landing first. The record is: **9 flagged in the CSV, plus W46 identified in prose**, and the PI's spot-check should cover all **10**. ⭐ Note where they fall: **6 of the 9 are stratum B**, where P1 fires and P2 does not — exactly the boundary an adoption decision turns on.

## 4 · What the next pack must do differently

1. ⛔ **Enrich for POSITIVES, not for firing.** Until the sample contains real over-boundary events, no candidate can be ranked on the error that matters for safety — missing a departure.
2. ⛔ **Find those candidates by a mechanism INDEPENDENT of the rules under test.** *"Windows where V0 fires hardest"* is the check-shares-the-defect trap in a new costume. Admissible: a physical cue (a kerb strike in the ego dynamics), a human sweep of video, or windows where **all three** candidates and the ego's own future pose agree by a wide margin — never the rule being adjudicated.
3. **Spend the next samples in stratum D** (408 windows, 10 sampled) — that is where a false-pass would live, and it is the only stratum whose clips are unclustered.
4. **Draw by CLIP, then by window within clip**, and report the cluster count beside every n.

## 5 · ⭐ What the DataFlyWheel did right, recorded because it should be repeated

It reported a defect **in its own pre-registration** and refused to score past it; it computed the reweighted column, labelled it post-hoc everywhere, and **declined to choose the reading that would adopt a rule**, referring that up instead. It also carried all three of my caveats into the write-up without paraphrasing them away. ⇒ That is what made this adjudication worth ruling on rather than re-running.
