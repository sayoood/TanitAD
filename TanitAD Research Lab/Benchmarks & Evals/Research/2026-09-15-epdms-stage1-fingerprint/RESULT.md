<title>EPDMS Stage-1 fingerprint — the substrate shift was a bug fix, and DrivoR's 54.6 is a data-augmented variant</title>

# `E-BE-S1S2-1`: the navhard Stage-2 shift is NAVSIM's official metric bug fix; DrivoR's 54.6 carries +134 k synthetic labels; two headline rows are un-stampable

**2026-09-15 · Research Lab (LAB-RUN-013) · Benchmarks & Evals · serves BE13-1 (pre-committed rotation item 2) and row 32**
⛔ **Tier:** external NAVSIM-v2 navhard-two-stage EPDMS (pseudo-closed-loop). Not a TanitAD tier, not comparable to T0/T1; the four-families gap (row 12) stands.

---

## 0 · Findings first

| # | finding | class |
|---|---|---|
| **F1** | ⭐⭐⭐ **The 08/2025 → 03/2026 Stage-2 substrate change has a NAME: the official NAVSIM metric bug fix (GitHub Issue #151, "related to the failure of filtering out human driver errors").** DrivoR's supplementary prints both tables, captioned *"Results before official metric bug fixing"* and *"Results after official metric bug fixing"*. Its pre-fix PDM-Closed row carries every v2 Stage-2 token (88.1 · 96.3 · 98.5 · 83.1 · 73.7 · 91.5 · 25.4 → **51.3**). ⇒ 09-13's "substrate changed between snapshots" was a correct mechanism missing its cause. **A score is pre-fix or post-fix; any cross-fix comparison is invalid.** | PUBLISHED lib `2601.05083` v2 (full text, Tab. 3 + Tab. 14) |
| **F2** | ⛔⭐⭐ **"DrivoR 54.6" is not the ~40 M base model on NAVSIM data. It is `DrivoR (+134k SimScale data, ViT-S)`, trained on synthetic scenes with PDM-Closed pseudo-annotations.** In the same post-fix table: base `DrivoR (ViT-S)` **48.3**, `+65k SimScale` **52.3**, `+134k` **54.6**; base pre-fix **45.3**. TOAD's `DrivoR 54.6 → +TOAD 56.3` inherits the +134 k variant. | PUBLISHED lib `2601.05083` v2 Tab. 3; `2606.07170` v1 Tab. 2 |
| **F3** | ⛔ **RAP-DINO 36.9 is a PRE-fix number; post-fix it is 39.6.** Our A5 ledger (09-05) carries 36.9 as a navhard SOTA claim with no substrate. RAP's own paper reports **navtest** tables only (its PDM-Closed row is NAVSIM v1: 89.1 PDMS). | PUBLISHED lib `2601.05083` Tab. 14 (both rows); `2606.07170` Tab. 2 (39.6); `2510.04333` v2 Tab. 1 |
| **F4** | ⚠️ **Under the committed rule, two headline rows are UN-STAMPABLE and are not quoted.** **DriveFuture 55.5**: its paper has one PDM-Closed mention and **0 / 23** fingerprint tokens near it, so no navhard table with a shared deterministic baseline exists. **CLOVER 48.3**: **0** PDM-Closed mentions. ⚠️ **Indirect evidence (does not stamp under the rule):** DriveFuture's Fig. 1 plots DrivoR 54.6 and SimScale 53.2, both post-fix values, and claims rank 1 *"as of April 2026"*. CLOVER's 48.3 *"matching the strongest reported result"* equals DrivoR base post-fix 48.3. Both are *consistent with* post-fix. | MEASURED `raw/s1s2_fingerprint.json` + PUBLISHED text |
| **F5** | ✅ **The instrument reads known values on its positive controls.** TOAD: Stage-1 **6/7**, v3 Stage-2 **7/8**, v2 **2/8**. Pseudo-Sim v3: Stage-1 **5/7**, v3 **4/8**, v2 **0/8**. DrivoR (pre-fix table): v2 **8/8**, v3 **1/8**. | MEASURED |
| **F6** | ⚠️ **A transcription discrepancy in the deterministic baseline itself:** PDM-Closed Stage-1 **HC = 97.7** in DrivoR's table vs **87.7** in TOAD and Pseudo-Sim v3. The other 8 Stage-1 subscores agree. One of the two is a typo; the headline EPDMS is unaffected. | PUBLISHED, cross-paper |

## 1 · The stamp table (replaces every unstamped navhard number we hold)

| row | value | substrate | how stamped | quotable? |
|---|---|---|---|---|
| PDM-Closed | 51.3 | **pre-fix** | own table, 8/8 tokens | ✅ with stamp |
| PDM-Closed | 56.6 | **post-fix** | TOAD / Pseudo-Sim v3 fingerprint | ✅ with stamp |
| DrivoR (ViT-S) base | 45.3 / **48.3** | pre / post | DrivoR Tab. 14 / Tab. 3 | ✅ |
| DrivoR +134k SimScale | **54.6** | post | DrivoR Tab. 3 | ✅ **only with "+134k synthetic PDM-C-labelled data"** |
| DrivoR +134k + TOAD | **56.3** | post | TOAD Tab. 2 ≡ PDM-C 56.6 table | ✅ same caveat |
| RAP-DINO | 36.9 / **39.6** | pre / post | DrivoR Tab. 14 / TOAD Tab. 2 | ✅ with stamp; ⛔ **36.9 retired as "SOTA"** |
| DriveFuture | 55.5 | **UN-STAMPABLE** | no shared baseline | ⛔ not quoted (plus CW-1 open) |
| CLOVER | 48.3 | **UN-STAMPABLE** | no PDM-Closed row | ⛔ not quoted |
| GuideFlow | 23.1–45.0 (old) · **27.1** in TOAD Tab. 6 | old range unknown · 27.1 post | TOAD Tab. 6 | ✅ 27.1 only; ⚠️ lib entry has **no PDF file** (not fingerprinted) |

## 2 · Five-dimension analysis

| dim | |
|---|---|
| **RELEVANCE** ⭐⭐ | G3 comparability; **row 32** (efficiency wedge vs DrivoR); row 3 (benchmark portfolio); the A5 ledger's own numbers. |
| **CONSEQUENCE** | ⛔⭐ **Row 32 changes shape again.** The number the wedge was aimed at (54.6 / 56.3) is a **data-scaled** result: +134 k synthetic scenes labelled by a privileged planner. Any "we beat DrivoR at N params" sentence must now name *which* DrivoR. The parity-data comparison is base **48.3 post-fix**. That is a params-and-data-matched target we could meet on our fixed corpus without synthetic augmentation. |
| **COMBINATION** | ⭐ **This is the T-5 / V-5 unit-error class again, one level up**: 09-09 split *inference vs training* params; today splits *base vs data-augmented* variant. Same failure family as the version-stamp rule (LI10-1): a model name, like an arXiv id, is a token standing in for several artifacts. It also sharpens **GS-5 / row 18**: DrivoR's own scaling rows (+65 k → +3.9, +134 k → +2.3 EPDMS) are the benchmark's clearest data-scaling curve, and they use **PDM-Closed as teacher**, the privileged planner at 56.6. |
| **CHANCES / RISKS** | **Upside:** F1 lets every external navhard number be stamped by one question ("pre or post #151?") instead of a snapshot hunt; the post-fix base target is 48.3, not 56.3. **Risks:** (a) F4's indirect reads look persuasive and are exactly the kind of evidence the committed rule excludes; (b) Issue #151 was read through DrivoR's description, not the GitHub issue itself (C2 debt); (c) the leaderboard may fix again. |
| **EXPERIMENT** | **`E-BE-FIX151-1` (0 GPU):** read NAVSIM Issue #151 and the fix commit, and state **which Stage-2 subscores the fix can move** (it filters human-driver errors). **Committed:** if EP and DAC are untouched by construction while NC/TTC/EC/LK/HC/DDC/TLC move, that matches 09-13's observed 7/9 and F1 is mechanism-confirmed; if the fix touches EP or DAC, the 7/9 pattern has an unexplained component and the stamp table carries a caveat. |

## 3 · What this changes (≤3)

1. ⭐⭐⭐ **Every external navhard EPDMS carries `pre-#151` / `post-#151`**, replacing 09-13's snapshot-date stamp. Correct the A5 ledger's 36.9 (dated entry appended today).
2. ⛔⭐ **Row 32: name the DrivoR variant.** Parity-data target = **48.3 post-fix (base ViT-S)**; 54.6 / 56.3 = **+134 k synthetic PDM-Closed-labelled data**. A wedge claim against the latter without that clause is a V-5 error.
3. ⚠️ **Hold DriveFuture 55.5 and CLOVER 48.3 out of every comparability table** until their authors publish a table with a PDM-Closed row (or the board lists them with a date).

## 4 · Manifest

| artifact | location |
|---|---|
| RESULT | `repo:TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-15-epdms-stage1-fingerprint/RESULT.md` |
| Code | `repo:…/code/s1s2_fingerprint.py` |
| Raw | `repo:…/raw/s1s2_fingerprint.json` (per-paper token hits + best window), `repo:…/raw/search_log.md` |
| Primaries | banked today: `2605.15120` (CLOVER), `2510.04333` (RAP); cited-by updated: `2601.05083`, `2605.09701`, `2606.07170`, `2506.04218` |
