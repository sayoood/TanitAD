<title>PDM-Closed 51.3 vs 56.6 — two versions of one paper; D-10 closes</title>

# The 51.3-vs-56.6 PDM-Closed spread is two versions of the maintainers' own paper — D-10 CLOSES on its committed branch

**2026-09-13 · Research Lab (LAB-RUN-012) · Benchmarks & Evals · serves BE10-1 (⛔ blocking) and debt D-10**
⛔ **Tier:** external NAVSIM-v2 EPDMS (pseudo-closed-loop, two-stage) — **not** a TanitAD tier, not comparable to T0/T1. EPDMS measures compliance and outcome; the four-families gap (D-EPDMS-FAM, row 12) stands.

---

## 0 · Findings first

| # | finding | class |
|---|---|---|
| **F1** | ⭐⭐⭐ **The two numbers are two VERSIONS of one maintainers' paper.** *Pseudo-Simulation for Autonomous Driving* `2506.04218` **v2 (2025-08-27)** Table 2: PDM-Closed navhard two-stage **51.3**. The **same paper, v3 (2026-03-20)**, Table 2 — captioned *"navhard leaderboard. Snapshot from 03/2026"* — PDM-Closed **56.6**, *"closely followed by recent scalable architectures like DrivoR"*. Both versions state **450 Stage-1 / 5,462 Stage-2** observations. | PUBLISHED — v3 read locally from the banked PDF (PyMuPDF, sha256 `a8431697…`); v2 read via arXiv HTML |
| **F2** | ⭐⭐⭐ **Stage 1 is identical in all 9 subscores across v2, v3 and TOAD:** NC 94.4 · DAC 98.8 · DDC 100 · TLC 99.5 · EP 100 · TTC 93.5 · LK 99.3 · HC 87.7 · EC 36.0. PDM-Closed is deterministic ⇒ identical Stage-1 scenes and scoring. | PUBLISHED (v3 PDF + v2 HTML + `2606.07170` HTML) |
| **F3** | ⭐⭐⭐ **Stage 2 moved in 7 of 9 subscores between v2 and v3** — NC 88.1 → **90.5**, TTC 83.1 → **86.6**, EC 25.4 → **29.7**, LK 73.7 → 74.2, HC 91.5 → 91.9, DDC 96.3 → 95.4, TLC 98.5 → 98.4 — and v3's Stage-2 values are **exactly TOAD `2606.07170`'s** (cross-checked in the v3 PDF text: 90.5 / 86.6 / 29.7 / 74.2 present; 51.3 / 88.1 / 83.1 / 25.4 / 73.7 absent). ⇒ **the Stage-2 substrate changed between the Aug-2025 and Mar-2026 leaderboard snapshots**; the planner, Stage 1 and the `s1·s2` aggregation did not. | PUBLISHED |
| **F4** | ⭐⭐⭐ **BE10-1's committed branch FIRES as written: "different versions ⇒ the basis is a VERSION variable, every external EPDMS carries a version stamp, and D-10 CLOSES."** The version stamp that exists is **paper version + leaderboard snapshot date**, not a devkit commit — GitHub releases stop at v2.1.2 (28 Apr) and name no later Stage-2 change. | PUBLISHED + PUBLISHED-RELEASE-NOTE |
| **F5** | ⛔⛔ **SELF-CORRECTION — the 2026-09-10 package manufactured this contradiction by reading an older version than the one we hold.** It quoted *"the maintainers' primary reports PDM-Closed = 51.3"*; the Library has held **v3** (56.6) since **2026-08-23**. **The same failure LI10-1 was written for, one day after it was proposed** — an arXiv id is not a document. | MEASURED (banked PDF version string vs 09-10 RESULT text) |
| **F6** | ⚠️ **TOAD cites PDM-Closed to Dauner et al., CoRL 2023** — which predates navhard and EPDMS; its 56.6 is the 03/2026 leaderboard value, not that paper's. A citation hygiene defect, not a numbers defect. | PUBLISHED `2606.07170` ref [8] |

## 1 · Consequences for our records

| record | motion |
|---|---|
| **D-10** (navhard scoring basis) | ✅ **CLOSED** — counts (09-10) + mechanism (Stage-2 substrate, today) + provenance (v2 = 08/2025 snapshot, v3 = 03/2026 snapshot). |
| **BE10-1** | ✅ **DONE** on its first committed branch. |
| **row 32** (efficiency wedge vs DrivoR) | ⭐ **UNBARRED for the same-snapshot comparison:** DrivoR + TOAD **56.3** vs PDM-Closed **56.6** are both on the **03/2026 substrate** (TOAD Table 2 ≡ v3 Table 2 for PDM-C). ⛔ Still barred: mixing either with any 08/2025-substrate number, and the second reason recorded 09-09 (104 M at inference vs the ~40 M figure) is **untouched by this**. |
| **"navhard ≤ 45.0 cap"** (BE10-2) | superseded twice over — retire. |
| **`LEDGER_A5_benchmarks.md`** 09-10 residue | dated correction entry (appended today): the spread was two paper versions. |

## 2 · Five-dimension analysis

| dim | |
|---|---|
| **RELEVANCE** ⭐⭐ | G3 external comparability; unblocks row 32's same-table sentence; informs row 3 (benchmark portfolio). |
| **CONSEQUENCE** | Every external navhard EPDMS now needs a **substrate stamp** (08/2025 or 03/2026 snapshot, or later). A leaderboard that re-scores Stage 2 between snapshots moves *every* entry, not just new ones. |
| **COMBINATION** | ⭐ **The cheap discriminator that settled it is the Stage-1 fingerprint** — a shared deterministic baseline whose Stage 1 must match. It is `H-ESTIM-SEED-1`'s logic in benchmark form: name which substrate your number answers. And it is LI10-1 twice in four days (Kairos v1/v3 on 09-10; Pseudo-Sim v2/v3 today). |
| **CHANCES / RISKS** | Upside: D-10 closes at 0 GPU; a reusable admission test. Risk: the leaderboard may re-score again — any number taken from a live board carries its snapshot date or it is unquotable. |
| **EXPERIMENT** | **`E-BE-S1S2-1` (0 GPU):** re-fingerprint every external EPDMS we hold (55.5 DriveFuture, 48.3 CLOVER, 36.9 RAP-DINO, 54.6 DrivoR base) by locating a PDM-Closed row in the *same* table and checking its Stage 2 against the 08/2025 vs 03/2026 clusters. **Committed:** a row whose table has no shared deterministic baseline is **un-stampable and not quoted**; a row matching neither cluster reveals a third substrate and is quoted only with its own snapshot. |

## 3 · What this changes (≤3)

1. ⭐⭐⭐ **Close D-10 and BE10-1; stamp every external EPDMS with its leaderboard snapshot** (paper version + snapshot date).
2. ⭐⭐ **Adopt LI10-1 (version-stamp rule) now** — it has two measured near-misses in four days, the second of which reached a ⛔-blocking backlog row.
3. ⭐ **Run `E-BE-S1S2-1`** before the row-3 portfolio plans any submission.

## 4 · Manifest

| artifact | location |
|---|---|
| RESULT | `repo:TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-13-pdmc-stage2-provenance/RESULT.md` |
| Stage-wise extracts (v2 vs v3/TOAD) | `repo:…/raw/stage_subscores.md` |
| Search log | `repo:…/raw/search_log.md` |
| Primaries | `repo:TanitAD Research Lab/Library/papers/2506.04218_Pseudo-Simulation-for-Autonomous-Driving.pdf` (**v3**), `…/2606.07170_Test-Time-Trajectory-Optimization-for-Autonomous-Driving.pdf` — both banked earlier; cited-by updated today. ⚠️ **v2 is NOT banked** — its 51.3 is read from arXiv HTML only. |
