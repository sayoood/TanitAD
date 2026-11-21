<title>The navhard split stamp — counts partition, scoring basis does not</title>

# navhard vs navtest: the counts partition cleanly, the scoring basis still does not

`TanitAD Research Lab · Benchmarks & Evals · 2026-09-10 · LAB-RUN-011`
`Serves: FS9-2 (blocking) · register debt D-10 · LAB_BACKLOG row 32 (efficiency wedge) · guideline T-4 · row 3 (benchmark portfolio)`
`Search log: raw/search_log.md · Frontier scan F2`

---

## THE PRE-COMMITTED READ, AND WHICH BRANCH FIRED

FS9-2 committed in advance:

> *"if counts partition cleanly into ~5.9k navhard and ~12k navtest, D-10 CLOSES and every number gets a stamp; if any paper reports ~12k AND calls it navhard, the split names themselves are unreliable and we stop using published EPDMS for external comparison entirely."*

⇒ **The first branch fired on the counts. Neither branch fired on the scoring basis, and that is where the residue is.**

---

## FINDINGS

**1. ⭐⭐⭐ navhard's size, from the maintainers' own primary.**
`PUBLISHED` · *Pseudo-Simulation for Autonomous Driving* (arXiv **2506.04218**, CoRL '25 — the NAVSIM v2 authors), verbatim:

> *"Our public NAVSIM v2 leaderboard features challenging driving scenarios… It uses a subset of nuPlan that we refer to as **navhard**, involving **450 Stage 1 and 5462 Stage 2** observations."*

A second subset used for their correlation study: *"244 initial observations (Stage 1) and 4164 synthetic observations (Stage 2)."* ⚠️ **That 244/4164 subset is NOT navhard** and must never be reported as such.

**2. ⭐⭐ navtest's size.** ≈ **12,000 samples** (`PUBLISHED-SECONDARY`, consistent across multiple 2026 method papers; ~136 test logs). ⚠️ Our own prior assignment of "12k" to navtest was an *inference from the count*; it is now corroborated but still wants a maintainers' primary.

**3. ⭐⭐⭐ THE SPLIT STAMP — the two populations are real and separable by count.**

| population | size | published anchors |
|---|---|---|
| **navhard** (NAVSIM v2, two-stage, pseudo-closed-loop) | 450 S1 / **5,462** S2 | PDM-Closed **51.3** · Latent TransFuser **23.1** · MLP **12.7** · Constant Velocity **10.9** |
| **navtest** (scored with the v2 EPDMS metric) | **≈12,000** | the 84.8 / 85.1 / 86.1 / **86.4** / 89.3 cluster |

⇒ **The 84.8–89.3 cluster is navtest-EPDMS and may not be called navhard.** Six of the numbers FS9-2 listed are now stamped. Drive-HWM's **86.4 EPDMS** (arXiv `2609.03572`, read today) falls in the navtest cluster and is stamped with it; its **93.8 PDMS** is **NAVSIM v1** and a different metric entirely.

**4. ⛔ THE RESIDUE — one privileged baseline, two values, one split name.**
Our records carry **PDM-Closed = 56.6 on navhard** (LAB-RUN-008, the source for row 32's *"DrivoR 56.3 is within 0.3 of privileged PDM-Closed"*). The maintainers' primary reports **PDM-Closed = 51.3 on navhard**. **A 5.3-point spread on the same privileged planner on the same named split.**

⇒ **The split name does not pin the scoring basis.** That was D-10's original suspicion and it survives the count check. Candidate causes, none yet verified: a NAVSIM version bump between the CoRL '25 paper and the 2026 tables; a two-stage weighting change; a sub-metric added to EPDMS (DDC / TLC / LK / HC / EC).

**5. ⚠️ A CORRECTION TO OUR OWN RECORD.** The 2026-09-05 pass wrote *"three independent tables cap navhard at ≤ 45.0"*. The maintainers' primary puts PDM-Closed at **51.3** on navhard. **The ≤45.0 cap is superseded and must not be re-quoted.** Recorded here rather than silently dropped.

**6. Where 55.5 / 56.3 land.** DriveFuture **55.5** and DrivoR **56.3** match **neither** cluster's centre: too high for the navhard anchors as published in CoRL '25, far too low for navtest. ⭐ **The benign reading is the likely one** — a year of leaderboard progress past PDM-Closed's 51.3 puts them in the navhard population. But that is an inference, and it is exactly the inference F4's 51.3-vs-56.6 spread says we cannot safely make.

**7. Metric definitions, banked for the four-families work (D-EPDMS-FAM, row 12).**
`PDMS = NC · DAC · (5·EP + 5·TTC + 2·Comf) / 12`. **EPDMS** adds **DDC** (driving-direction compliance), **TLC** (traffic-light compliance), **LK** (lane keeping) and replaces Comf with **HC** (history comfort) and **EC** (extended comfort). Range **[0, 1]**. ⭐ Note for row 12: **none of the eleven sub-metrics maps to `tac.manoeuvre_decision` or `strat.route_goal`** — the four-families gap D-EPDMS-FAM names is confirmed at the level of the metric's own definition, not merely suspected.

---

## THE FIVE DIMENSIONS (deep-read item: `2506.04218`)

1. **RELEVANCE ⭐⭐⭐** — D-10 blocks row 32 and guideline T-4 bars six numbers from every comparability table. G3 (beat SOTA on community benchmarks) has no lane while this is open.
2. **CONSEQUENCE** — six numbers gain a split stamp; the ≤45.0 cap is retired; row 12's four-families gap is confirmed from the metric definition; **row 32 stays BARRED** on the scoring-basis residue.
3. **COMBINATION** — with 09-09's two-population mechanism this completes the diagnosis: the populations are real, separable by count, and *still* not sufficient to make a number comparable, because the same baseline reads two ways inside one population.
4. **CHANCES / RISKS** — *Chance:* a single leaderboard submission of our own would resolve the basis question by construction, since the leaderboard scores every entry identically. *Risk:* stamping the six numbers may read as "D-10 is closed" and unbar row 32 prematurely. **It is not closed. Say so in the same sentence as the stamp, every time.**
5. **EXPERIMENT (0 GPU)** — resolve the 51.3-vs-56.6 spread by provenance: for each of the two PDM-Closed navhard values, extract the NAVSIM version/commit and the scoring-basis era from its source. ⛔ **Committed in advance: if the two values come from different NAVSIM versions, the basis is a VERSION variable, every external EPDMS carries a version stamp from now on, and D-10 closes; if they come from the same version, the split name is unreliable within a version and we stop using published EPDMS for external comparison entirely — FS9-2's second branch, fired late.**

---

## WHAT THIS CHANGES FOR TANITAD — ≤3 recommendations

1. ⭐⭐⭐ **Stamp the six navtest numbers now, and keep row 32 BARRED.** The stamp resolves *which population*, not *which basis*. Both halves ship together or neither does.
2. ⭐⭐ **Extend guideline T-4: every external EPDMS carries split name AND NAVSIM version/commit.** The 51.3-vs-56.6 spread is what a missing version stamp looks like.
3. ⚠️ **Retire the "navhard ≤ 45.0" cap from the A5 ledger and anywhere it was quoted.** Superseded by the maintainers' own 51.3.

## Escalations

- **To the Master Mind:** D-10 moves to `COUNTS RESOLVED · BASIS OPEN`. Row 32 stays barred. The ≤45.0 correction needs propagating wherever it was cited.
- **To the PI:** the cheapest route to closing the basis question is a single navhard leaderboard submission of our own, which is a row-3 portfolio decision already awaiting PI approval.
