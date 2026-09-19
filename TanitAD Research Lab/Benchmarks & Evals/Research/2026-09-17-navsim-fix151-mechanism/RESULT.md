<title>NAVSIM Issue #151 read at the primary — the mechanism confirms, the committed partition does NOT</title>

# `E-BE-FIX151-1`: #151 is a `human_penalty_filter` inconsistency in the **weighted** metrics array — which contains **EP**. The pre-committed branch fires: the stamp table carries a caveat

**2026-09-17 · Research Lab (LAB-RUN-014) · Benchmarks & Evals · serves the 09-15 package's pre-registered `E-BE-FIX151-1`; touches row 32, row 3, row 12 and the A5 ledger**
⛔ **Tier:** external NAVSIM-v2 EPDMS (pseudo-closed-loop). Not a TanitAD tier; not comparable to T0/T1. The four-families gap (row 12) stands.

---

## 0 · Findings first

| # | finding | class |
|---|---|---|
| **F1** | ✅ **Mechanism CONFIRMED at the primary.** Issue #151's title is *"Question Regarding the human_penalty_filter Mechanism: Discrepancy Between Reported Sub-score and Final Score Calculation"*, opened **2025-09-03**. The defect: when the human driver fails a metric (0.0), the agent's score for that metric is exempted and overwritten to **1.0 in the reported sub-scores**, while the **final total still used the original 0.0** — the exemption modified individual metric scores but **not the underlying `weighted_metrics_array`**. The demonstration in the issue: all individual metrics reported 1.0, final score **0.875**. ⇒ 09-15's F1 (*"the failure of filtering out human driver errors"*, taken from DrivoR's caption) describes the right mechanism. | PUBLISHED-ISSUE, github.com/autonomousvision/navsim/issues/151, retrieved 2026-09-17 |
| **F2** | ⛔⭐⭐ **The pre-committed branch that fires is the SECOND one — the fix CAN touch EP.** EPDMS splits into **four multiplicative penalty gates — NC, DAC, DDC, TLC** — and **five weighted metrics — EP(5), TTC(5), LK(2), HC(2), EC(2)**. The array named in the issue is the **`weighted_metrics_array`**, and **EP is its highest-weighted member**. `E-BE-FIX151-1` committed: *"if the fix touches EP or DAC, the 7/9 pattern has an unexplained component and the stamp table carries a caveat."* ⇒ **Caveat applies.** | PUBLISHED (EPDMS composition, cross-confirmed at two independent sources) + F1 |
| **F3** | ⚠️⭐ **And the complement is the sharper half: the multiplicative gates are NOT in that array.** NC, DAC, DDC and TLC are penalty multipliers, not members of `weighted_metrics_array`. ⇒ **#151 cannot, by construction, move NC / DAC / DDC / TLC directly.** 09-13 observed **7 of 9 Stage-2 subscores moving** between substrates. If any of those seven are multiplicative gates, that movement has a source other than #151, and the single-cause story is incomplete. | MEASURED (composition) + INHERITED (09-13's 7/9 observation) |
| **F4** | ⚠️ **The fix itself is still UNVERIFIED.** The issue is **closed with no linked fix commit or pull request visible** on the issue page. ⇒ the *mechanism* is PUBLISHED-ISSUE, but *"the change that produced the 08/2025 → 03/2026 substrate shift is this fix"* remains an **inference over a caption**, not a read of a diff. The C2-class debt named on 09-15 is **not** discharged; it changes shape, from *"read the issue"* to *"find the commit"*. | MEASURED (absence on the issue page, one probe — see §4) |
| **F5** | ⛔⭐⭐ **A live comparability catch from today's frontier scan, banked here because it belongs to B&E:** **WA-JEPA's headline 91.7 EPDMS is `navtest`, not `navhard`** — its Table 1 caption reads *"Comparison with state-of-the-art methods on NAVSIM-v2 navtest"* (baselines: SparseDriveV2 90.1, Discrete-WAM 90.4). Our entire stamp table is **navhard** (48.3–56.6). ⇒ **91.7 must never enter a navhard comparability table.** A reader meeting "EPDMS 91.7" and "EPDMS 56.6" without the split stamp would conclude the field moved 35 points in a year. | PUBLISHED lib `2608.20974` Tab. 1 |

---

## 1 · The EPDMS partition, stated once so the next pass does not re-derive it

| kind | members | can #151 move it? |
|---|---|---|
| **multiplicative penalty gates** | NC · DAC · DDC · TLC | ⛔ **no** — not in `weighted_metrics_array` |
| **weighted metrics** (weights) | **EP (5)** · **TTC (5)** · LK (2) · HC (2) · EC (2) | ✅ yes — this is the array the issue names |

The issue demonstrates the defect on **lane_keeping**, but states the filter applies to *any* metric
where `human_pdm_result[column].iloc[0] == 0`. ⇒ **LK is the example, not the scope.**

## 2 · Consequence for the 09-15 stamp table

The stamp table stands — **every external navhard number still carries `pre-#151` / `post-#151`** —
with one added line:

> ⚠️ **#151 is a necessary part of the substrate story, not demonstrably the whole of it.** It acts on
> the five weighted metrics (EP, TTC, LK, HC, EC) and cannot directly move the four multiplicative
> gates. Any Stage-2 movement observed on NC / DAC / DDC / TLC has another cause, still unnamed.

⛔ **Nothing is retracted.** 09-15's F1 identified the right mechanism; what today's primary changes is
the *scope* of what that mechanism can explain, which is exactly what the experiment was registered
to test.

## 3 · Five-dimension analysis

| dim | |
|---|---|
| **RELEVANCE** ⭐⭐ | G3 comparability; **row 3** (benchmark portfolio — we would submit under this scorer); **row 32** (efficiency wedge target); **row 12** (D-EPDMS-FAM); the A5 ledger's stamps. |
| **CONSEQUENCE** | The stamp question *"pre or post #151?"* remains the right first question, but it is **not sufficient** to explain a cross-substrate table. A second, unnamed change sits behind any movement in the multiplicative gates — and those four are precisely the **safety** terms (collision, drivable area, direction, traffic light), i.e. the ones a safety case would lean on. ⭐ F5 adds a **split** stamp alongside the **fix** stamp: an external EPDMS now needs *both* `navtest\|navhard` *and* `pre\|post-#151` before it is quotable. |
| **COMBINATION** | ⭐ This is the **V-5 / T-5 unit-error family for the third consecutive pass**, and the object keeps moving up a level: 09-09 split *inference vs training params*; 09-15 split *base vs +134 k-data variant*; today splits *which benchmark SPLIT* (F5) and *which metric SUBSET a fix can reach* (F2/F3). ⭐ It is also the **`df` / `step_s` family**: a true statement (*"#151 fixed the human-error filter"*) quoted outside its scope (*"therefore all nine subscores are explained"*). |
| **CHANCES / RISKS** | **Upside:** two cheap stamps now make every external EPDMS admissible or refuse it, and the partition table above means neither has to be re-derived. **Risks:** (a) F4 — no diff was read, so the link from #151 to the substrate shift is still an inference; (b) the EPDMS composition is PUBLISHED from papers describing NAVSIM, not from NAVSIM's own source, so a version drift in the weights (5/5/2/2/2) would not be caught here; (c) F3's force depends on 09-13's 7/9 identification, which is INHERITED and not re-verified today. |
| **EXPERIMENT** | **`E-BE-FIX151-2` (0 GPU):** find the **fix commit** — search the NAVSIM repo history and release notes for `human_penalty_filter` / `weighted_metrics_array` between 2025-09-03 and 2026-03, and read the diff. **Committed in advance:** if the diff touches only `weighted_metrics_array`, F2/F3 are confirmed at the source and the caveat in §2 becomes a permanent line in the stamp table; if it also changes the multiplicative gates or the aggregation, then 09-15's single-cause account is **superseded** and the stamp table needs a third axis. ⚠️ **And name which of 09-13's 7/9 moved subscores are multiplicative** — that is a re-read of a banked artifact, 0 GPU, and it decides whether F3's "unexplained component" is real or an artefact of the earlier identification. |

## 4 · What this changes (≤3)

1. ⛔ **Add the caveat line of §2 to the stamp table**, and record the EPDMS partition (§1) where the next pass will find it.
2. ⛔⭐ **An external EPDMS now needs TWO stamps to be quotable: the SPLIT (`navtest` / `navhard`) and the FIX (`pre-` / `post-#151`).** ⛔ **WA-JEPA 91.7 is `navtest` and is excluded from every navhard table** (F5).
3. ⚠️ **F4 keeps the C2 debt open, re-scoped**: the fix commit is unread, so the causal link between #151 and the substrate shift stays an inference over DrivoR's caption.

## 5 · Stopping condition (Rule Zero)

**(3a) partially / (3b) for the remainder.** The registered experiment ran and its committed branch fired — that is a cleared bar, not a refutation stop. The follow-on `E-BE-FIX151-2` is **0 GPU and unblocked**; it was not run in this pass because the four-package budget was spent, and it is proposed to the backlog in the same turn rather than left as a sentence in a report.

## 6 · Manifest

| artifact | location |
|---|---|
| RESULT | `repo:TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-17-navsim-fix151-mechanism/RESULT.md` |
| Search log | `repo:…/raw/search_log.md` |
| Primary (issue) | `github.com/autonomousvision/navsim/issues/151`, retrieved 2026-09-17 — ⚠️ a web page, not bankable as a PDF; recorded with its retrieval date per charter §3 |
| Primary (banked today) | lib `2608.20974` (WA-JEPA, F5's source) |
