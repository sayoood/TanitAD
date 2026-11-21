# REGISTRATION DIFF: PREREG_R0E_PICK.draft.md -> PREREG_R0E_PICK.md (2026-10-18)

- Draft sha256: `d481ffc7078553311043485ec4803e2bb50ef664280e29006f3171e8bbe27a25`.
- Registered sha256: `1be7a0301b454a21a0adcfc209ce73f63b7802a72d3155008a90ca616d5cccd0` (also in `PREREG_R0E_PICK.sha256`).
- Decisions applied: the PI's answers of 2026-10-18 ~17:30 Berlin, `Project Steering/PI_DECISION_QUEUE.md` (section "DECIDED 2026-10-18 ~17:30 Berlin"). The PI's labels 1a-1f are the draft `NOTES.md` decisions 1-6: 1a A40 spend and Stage 2 only on O1; 1b T = 0.01; 1c R0e-id to u1000 (B4 dropped); 1d O2 earns no Stage 2; 1e DEPLOYABLE-SAFE vs-BASE line report-only; 1f DDC pre-authorised lever.
- Applying the diff below to the draft with `patch` reproduces the registered file byte for byte (checked: sha256 equal).
- Decisions 1a, 1b and 1e needed no text change beyond the header block and the decisions list (H2, H5): the draft already carried them.
- Not touched: every bar (S, G, L, E, K0-K6), every outcome row's condition and meaning, the estimator, the argv, D1-D5, the four metric families, §2.3 and the other build-item rows.

## Justification per hunk (draft line = first draft line of the change)

| hunk | draft line | PI decision | justification |
|---|---|---|---|
| H1 | 1 | FROZEN marker | Title marker DRAFT becomes FROZEN (instruction: every DRAFT marker becomes FROZEN). |
| H2 | 3 | registration header; all six (1a-1f) | Header block: registration date, the six PI decisions applied, and the draft's sha256. Replaces the draft's own status line ("It is not registered"), which no longer holds. |
| H3 | 30 | 1c | R0e-id runs to u1000, not u250; stated as a full same-seed replicate of R0d at all four checks. |
| H4 | 34 | 1c | Headline cost recomputed: R0e-id 1.27 -> 3.68 (+2.41), so Stage 1 4.95 -> 7.36, maximum 12.31 -> 14.72, u500-kill Stage 1 3.34 -> 5.75 (3.68 + 2.07). Arithmetic in §7. |
| H5 | 39 | 1a, 1b, 1c, 1d, 1e, 1f | The draft's list of PI decisions to take becomes the record of the PI decisions taken, one line each. |
| H6 | 134 | 1c | Removes the dependence on build item B4: R0e-id runs R0d's --max-updates 1000 to completion instead of being stopped by PID after the u250 dump. The warning that --max-updates 250 is not an identity argv is kept. |
| H7 | 198 | 1d | Replaces the pointer to an open PI question ("the PI question is in NOTES") with the decision: an O2 outcome, safe but below G, does not earn Stage 2. Bar G itself is unchanged. |
| H8 | 215 | 1c | R0e-id control row: runs to u1000; the gate reading stays at the first check; later checks are banked as further floors; cost 1.27 -> 3.68. |
| H8b | 215 | 1c | R0e-id control row cost 1.27 -> 3.68 A40-h (same arithmetic as §7). |
| H9 | 235 | 1c | Consequence of the full replicate: the paired B_id - B_R0d and A_id - A_R0d floors are also banked at u500 / u750 / u1000. Descriptive; the ID-FAIL / ID-DIVERGED readings stay at step0 and u250. No bar, outcome row or estimator changes. |
| H10 | 248 | 1f | Row P-O4-PICK: DDC rider appended to the "next arm" cell only; the row's condition, meaning and existing next arm are unchanged. |
| H11 | 249 | 1f | Row P-O4-WEDGE: DDC rider appended to the "next arm" cell only; the row's condition, meaning and existing next arm are unchanged. |
| H12 | 251 | 1f | Row O1: DDC rider appended to the "next arm" cell only; the row's condition, meaning and existing next arm are unchanged. |
| H13 | 252 | 1f | Row O2: DDC rider appended to the "next arm" cell only; the row's condition, meaning and existing next arm are unchanged. |
| H14 | 253 | 1f | Row O3-a: DDC rider appended to the "next arm" cell only; the row's condition, meaning and existing next arm are unchanged. |
| H15 | 254 | 1f | Row O3-b: DDC rider appended to the "next arm" cell only; the row's condition, meaning and existing next arm are unchanged. |
| H16 | 255 | 1f | Row O3 SLACK: DDC rider appended to the "next arm" cell only; the row's condition, meaning and existing next arm are unchanged. |
| H17 | 284 | 1c | Cost table: R0e-id now runs to u1000: 0.34 + 12,007 / 3600 = 3.68 (the same arithmetic as R0e-pick's row). Was 0.34 + (467 + 2,425 + 460) / 3600 = 1.27. |
| H18 | 288 | 1c | Stage 1 total: 1.27 + 3.68 = 4.95 -> 3.68 + 3.68 = 7.36; the u500-kill case 3.34 -> 3.68 + 2.07 = 5.75. |
| H19 | 290 | 1c | Maximum: 12.31 -> 7.36 + 7.36 = 14.72 (arithmetic now shown; Stage 2 row unchanged at 2 x 3.68 = 7.36). |
| H20 | 291 | 1c | The draft's option row (R0e-id to u1000, +3.68 - 1.27 = +2.41, marked PI) becomes the adopted decision. |
| H21 | 293 | 1c | States how the totals are rounded (sums of the rounded rows, as the draft did) so the recomputed figures can be re-derived. |
| H22 | 339 | 1c | Build item B4 (stop-after-u250 supervisor, or the PI accepting u1000) is dropped: the PI took the u1000 branch. Row kept so the B-numbering the registers cite does not shift. |

Hunks H10-H16 are the seven §6 rows that received the DDC rider: P-O4-PICK, P-O4-WEDGE, O1, O2, O3-a, O3-b, O3 SLACK. Rows O6, O7, P-O5 and O4+O5 / SLACK did not (see RESULT.md, item 1 of "NEEDS THE MASTER MIND"). H8 and H8b are one table row in two edits.

## Unified diff

```diff
--- PREREG_R0E_PICK.draft.md
+++ PREREG_R0E_PICK.md
@@ -1,6 +1,16 @@
-# PREREG R0e-pick (DRAFT): constrained EPO with a pick-aligned constraint temperature, `--epo-con-temp 0.01`
+# PREREG R0e-pick (FROZEN): constrained EPO with a pick-aligned constraint temperature, `--epo-con-temp 0.01`
 
-**DRAFT, written 2026-10-18 by a Research Lab design agent for the Master Mind. It is not registered: the Master Mind freezes and hashes it.**
+**FROZEN, registered 2026-10-18. Drafted 2026-10-18 by a Research Lab design agent; landed by the Master Mind.**
+- **Registered with the PI decisions of 2026-10-18 ~17:30 Berlin** (verbatim in `Project Steering/PI_DECISION_QUEUE.md`, section "DECIDED 2026-10-18 ~17:30 Berlin"). The PI's questions 1a-1f are decisions 1-6 of the draft's `NOTES.md`:
+  - **1a** Stage 1 A40 spend AUTHORISED, queued after P6. Stage 2 runs only on outcome O1, as drafted.
+  - **1b** T = 0.01, as drafted. The disclosed departure from A2R2's T\* = 0.02 is kept (§1.3).
+  - **1c** R0e-id runs to **u1000**, not u250. The cost table is recomputed (§7). Build item B4 is dropped.
+  - **1d** An O2 outcome does NOT earn Stage 2, as drafted.
+  - **1e** The DEPLOYABLE-SAFE vs-BASE line stays report-only, as drafted. No E clause.
+  - **1f** PRE-AUTHORISED: if DDC separates adverse again, g5 + DDC as a third constrained term is the committed next lever. It is added to the "next arm" text of the §6 rows only.
+- **Draft sha256** (`PREREG_R0E_PICK.draft.md`, git blob `304b49d2…`): `d481ffc7078553311043485ec4803e2bb50ef664280e29006f3171e8bbe27a25`.
+- **This file** is hashed in `PREREG_R0E_PICK.sha256`. The unified diff draft → this file, with one justification per hunk, is `REGISTRATION_DIFF.md` in this directory.
+- **Everything not named in that diff is the draft, unchanged.** In particular the build-item status lines in §2.3 and §10 are as drafted (mirror tip `8bb47ad`). Their current status lives in BACKLOG row `R0E-PICK`, not here.
 - No R0e number exists. Nothing has run: no R0e training step, no trainer edit, no pod or GPU access.
 - Sources: mirror tip `8bb47ad`, plus the SO-PICKCON package being landed (`C:/Users/Admin/pickcon/out/`, trainer blob `869b1862…`).
 - PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.
@@ -27,19 +37,23 @@
   - it is a second lever;
   - its feed half does not exist in the trainer (§2.3).
 - **Arms, in order:**
-  1. **R0e-id**: the identity control, T = 0.1 named, to u250.
+  1. **R0e-id**: the identity control, T = 0.1 named, run to **u1000** (PI 1c): a full same-seed replicate of R0d at all four checks (u250 / u500 / u750 / u1000).
   2. **R0e-pick**: seed 0.
   3. Stage 2, only if R0e-pick reads O1: **R1e-pick** (seed 1, same flags), then **R1o** (seed 1, constraint off, as registered in SAFEOBJ).
 - **Bars:** S, G, L, E and K0–K6 are **unchanged** (§4). FREEPOINT read FP-NONE, so the navtest bar adds nothing.
-- **Cost (ESTIMATED, §7):**
-  - Stage 1 is ≤ **4.95 A40-h**: 1.27 for R0e-id and 3.68 for R0e-pick.
+- **Cost (ESTIMATED, §7; recomputed for PI 1c):**
+  - Stage 1 is ≤ **7.36 A40-h**: 3.68 for R0e-id (to u1000) and 3.68 for R0e-pick.
   - Stage 2 adds ≤ **7.36 A40-h**.
-  - The maximum is **12.31 A40-h**.
-  - If R0e-pick is K3-killed at u500, Stage 1 costs 3.34.
-- **PI decisions:**
-  1. A40 spend.
-  2. T = 0.01 versus SO-PICKCON's T* = 0.02 (§1.3).
-  3. Stage 2 under the same rule as SAFEOBJ ("conditional").
+  - The maximum is **14.72 A40-h**.
+  - If R0e-pick is K3-killed at u500, Stage 1 costs 5.75 (3.68 + 2.07).
+  - The draft's 4.95 / 12.31 / 3.34 assumed R0e-id stopped at u250 (1.27). The difference is the +2.41 A40-h of PI 1c.
+- **PI decisions, taken 2026-10-18 ~17:30 Berlin (header block):**
+  1. A40 spend: AUTHORISED for Stage 1, queued after P6 (1a).
+  2. T = 0.01, the disclosed departure from SO-PICKCON's T* = 0.02 (§1.3) (1b).
+  3. Stage 2 under the same rule as SAFEOBJ ("conditional": only on O1). An O2 outcome does not earn it (1a, 1d).
+  4. R0e-id runs to u1000; build item B4 is dropped (1c).
+  5. The DEPLOYABLE-SAFE vs-BASE line beside E is report-only (1e).
+  6. If DDC separates adverse again, g5 + DDC as a third constrained term is the pre-authorised next lever (1f; §6 rows).
 
 ## 1. Why this lever, and why T = 0.01
 
@@ -131,8 +145,9 @@
   * The same holds for R0e-id with `--epo-con-temp 0.1`.
   * `lever_tokens_not_in_R0d` must be exactly `{--epo-con-temp, <T>, own paths}`.
 * ⛔ **`--max-updates 250` is NOT an identity argv.** The cosine schedule runs over `total = min(total, a.max_updates)` (patched `:1661-1662`), so the learning rate at u ≤ 250 would differ from R0d's.
-  * R0e-id therefore runs R0d's `--max-updates 1000`.
-  * It is **stopped by PID after its u250 score dump is banked**. That is build item B4.
+  * R0e-id therefore runs R0d's `--max-updates 1000`, **to completion (PI 1c)**.
+  * It is **not stopped at u250**. Build item B4 (supervisor stop-after-dump) is dropped; no stop-after-dump feature is built.
+  * R0e-id is thus a full same-seed replicate of R0d at all four checks.
 
 ### 2.2 The hop trace (SO-PICKCON, MEASURED on the CPU rig; patched trainer `869b1862…`)
 
@@ -195,7 +210,7 @@
 | bar | text (PREREG_SAFEOBJ §5 + addendum R1/R4) | change for R0e-pick |
 |---|---|---|
 | **S** | TTC and NC point deltas (B − own A, RL-val) ≥ −0.5 at **every** check u250–u1000; g3 at every check (pick / fan-mean path ≥ 0.894, shortest-bin share ≤ 0.55) | **none** |
-| **G** | at u1000: the PDMS lower 95 % bound > 0 **and** the point ≥ G(t) = max(+0.60, F(t)), where F interpolates dose.json's vertices at full precision (addendum R1) | **none.** ⚠️ FREEPOINT measured the frontier's α-0.25 point M25 on navtest at PDMS −0.47 [−1.03, +0.24] vs BASE (D-REFE-FREEPOINT-1, INHERITED), so G's premise ("a free deployable point exists") is weaker on the endpoint than on RL-val. G is kept as written; the PI question is in NOTES |
+| **G** | at u1000: the PDMS lower 95 % bound > 0 **and** the point ≥ G(t) = max(+0.60, F(t)), where F interpolates dose.json's vertices at full precision (addendum R1) | **none.** ⚠️ FREEPOINT measured the frontier's α-0.25 point M25 on navtest at PDMS −0.47 [−1.03, +0.24] vs BASE (D-REFE-FREEPOINT-1, INHERITED), so G's premise ("a free deployable point exists") is weaker on the endpoint than on RL-val. G is kept as written (PI 1d: an O2 outcome, safe but below G, does not earn Stage 2) |
 | **L** | TTC only (A-L1): the constraint's effect (R·e-pick B−A) − (unconstrained B−A) > 0 at u1000 on both seeds, and > 2 × F_seed | **none.** The unconstrained partners are **banked R0** (seed 0) and **R1o** (seed 1, Stage 2). F_seed = max(\|ΔTTC(R0 vs R1o)\|, \|ΔTTC(R0e-pick vs R1e-pick)\|) |
 | **E** | both pairs (R0e-pick, R1e-pick) on navtest: each ≥ +0.25 and mean ≥ +0.5; pooled lower bound > 0; mean > 2 × max(seed floor, I-rep \|Δ\|); per pair, the TTC and NC upper bounds ≥ 0 (A-E2); every sub-score reported | **none.** `PREREG_FREEPOINT.md`: *"FP-NONE: neither is DEPLOYABLE-SAFE ⇒ the safety-objective arm proceeds exactly as pre-registered"*, and the read was FP-NONE. So "beats R0B against BASE" is **not** added |
 | **K0–K6** | `PREREG_RL_S1.md` §5. K3/K4 vs own A. K3 = any g3/g5 breach at two consecutive checks | **none.** R0e-id and R1o run K3 **observe-only**; K2 and K6 stay live |
@@ -212,7 +227,7 @@
 | control | what it answers | cost |
 |---|---|---|
 | **Launch-gate PASS token (K0)** | each of R0e-id, R0e-pick, R1e-pick and R1o starts only on a token from `refe/launch_gate.py` bound to commit, argv sha256 and data-manifest sha256s; `RLS1D_SUITE` + `test_rl_pickcon.py ZZTEST_PASS 20`; G13 incl. `temp_reach`; G-DVB `epo_con_temp` built == registered | in §7's preflight line |
-| **R0e-id, the T = 0.1 identity, vs banked R0d at the first check (u250)** | (i) the PICKCON bytes, as shipped, run R0d's configuration on the A40; (ii) **new information**: the "same run again" floor for **arm B** of a constrained EPO arm (training non-determinism at the same seed: bf16 autocast and the atomics-based memory-efficient SDPA backward. Patched `:1369-1373` records two head-less A40 runs differing in all 78 scorer tensors, max \|d\| 1.2e-4; the main loop runs without strict determinism). It has never been measured | 1.27 A40-h |
+| **R0e-id, the T = 0.1 identity, run to u1000 (PI 1c), vs banked R0d. The gate reading is at the first check (u250); u500 / u750 / u1000 are banked as further floors** | (i) the PICKCON bytes, as shipped, run R0d's configuration on the A40; (ii) **new information**: the "same run again" floor for **arm B** of a constrained EPO arm (training non-determinism at the same seed: bf16 autocast and the atomics-based memory-efficient SDPA backward. Patched `:1369-1373` records two head-less A40 runs differing in all 78 scorer tensors, max \|d\| 1.2e-4; the main loop runs without strict determinism). It has never been measured | 3.68 A40-h |
 | **Free "same run again" floor for arm A** | R0e-pick-A vs R0d-A, same seed and flags; A never sees the lever. In SO-PICKCON's CPU test, a change of T leaves arm A bit-identical. Precedent R0-A vs C0-A: PDMS +0.01 [−0.13, +0.14], TTC 0.00 [−0.13, +0.13] (INHERITED, PREREG_SAFEOBJ §7) | 0 |
 | **Second seed R1e-pick** (Stage 2, iff R0e-pick reads O1) | the training-seed floor of the constrained arm at T = 0.01; replication of S and G | 3.68 A40-h |
 | **R1o** (Stage 2, after R1e-pick passes S) | bar L's seed-1 unconstrained partner, and the programme's first seed replicate of the unconstrained R trade | 3.68 A40-h |
@@ -233,6 +248,7 @@
    Meaning: the rig's same-run floor is as large as S's tolerance, so a one-seed S read is uninterpretable. Next: a PI decision between two seeds as the unit of S (both must pass) and stopping.
 3. **ID-PASS.** Otherwise. R0e-pick launches.
    * Banked as floors, quoted beside every R0e-pick vs R0d contrast: the paired B_id − B_R0d and A_id − A_R0d on RL-val at u250 (PDMS, TTC, NC with CIs), and the contrast-level |Δ|.
+   * Because R0e-id runs to u1000 (PI 1c), the same floors are banked at u500, u750 and u1000 and quoted beside the R0e-pick vs R0d contrast at the matching check. The ID-FAIL and ID-DIVERGED readings above stay at step0 and u250.
    * A **separated** paired difference here is reported as a finding (the H-ESTIM-SEED-1 family) and **does not block**. With 6 cells at ≥ 5 % per cell, a chance separation is expected about 26 % of the time, and the tiny rig measured 9.5 % per cell.
 
 ## 6. Committed outcomes, each with its next arm (Rule Zero)
@@ -245,14 +261,14 @@
 | **O6 INERT** | λ_TTC = λ_NC = 0 at every logged update | the constraint never acted, which contradicts R0d (λ > 0 on 941 / 1,000 updates); a mechanism failure | audit `con_*` logging against the A2 harness on the run's own dumps; no new arm before that |
 | **O7 OTHER KILL** | K2 (non-finite loss), K4 or K6; or g3 alone (addendum R2) | K2 at T = 0.01 implicates the 1/T gradient; g3 alone means TTC was bought with short plans | K2 → **R0e-pick at T = 0.02** (A2R2's T\*, half the 1/T scale). g3 → g3's quantity as a third constrained term. Both PI |
 | **P-O5 DUAL SATURATED / LAGGING** | breached AND (λ reached λ_max, OR the in-sample window mean g > 0.001) | the dual could not hold even its own quantity | the PID dual or projected-gradient constraint (SAFEOBJ O5). CPU unit proof first; arm PI |
-| **P-O4-PICK HELD IN-SAMPLE, VAL BREACHED** | breached AND g held (≤ 0.001 per term) AND λ > 0 on ≥ 50 % of post-ramp updates AND the **in-run wedge \|D1\| ≤ 0.10 points** | the pick held in-sample and did not transfer: a genuine generalisation gap for a **constrained** policy, which SO-CARVE could not measure on R0 states | **R0e-carve**: the dual fed from a log-disjoint carve of RL-train, never RL-val, at T = 0.01. Blocked on the carve build (§2.3; 0 GPU, ~1 day ESTIMATED; started in parallel) and the PI |
-| **P-O4-WEDGE HELD, PICK NOT ALIGNED IN-RUN** | breached AND g held AND **\|D1\| > 0.10** | T = 0.01 aligned on R0 states but not along the constrained trajectory: the softmax still is not the pick | a **hard-pick constraint**: a hinge on the pick's TTC through a straight-through or sampled estimator (SO-CARVE's named fallback). CPU unit proof with an identity at T → 0 first; arm PI |
+| **P-O4-PICK HELD IN-SAMPLE, VAL BREACHED** | breached AND g held (≤ 0.001 per term) AND λ > 0 on ≥ 50 % of post-ramp updates AND the **in-run wedge \|D1\| ≤ 0.10 points** | the pick held in-sample and did not transfer: a genuine generalisation gap for a **constrained** policy, which SO-CARVE could not measure on R0 states | **R0e-carve**: the dual fed from a log-disjoint carve of RL-train, never RL-val, at T = 0.01. Blocked on the carve build (§2.3; 0 GPU, ~1 day ESTIMATED; started in parallel) and the PI. **DDC rider (PI 1f, pre-authorised):** if D5 reads DDC separated adverse again (B − own A, 95 % CI entirely below 0, as R0d's −0.92 [−1.45, −0.48]), **g5 + DDC as a third constrained term is the committed next lever**, in addition to the lever above. |
+| **P-O4-WEDGE HELD, PICK NOT ALIGNED IN-RUN** | breached AND g held AND **\|D1\| > 0.10** | T = 0.01 aligned on R0 states but not along the constrained trajectory: the softmax still is not the pick | a **hard-pick constraint**: a hinge on the pick's TTC through a straight-through or sampled estimator (SO-CARVE's named fallback). CPU unit proof with an identity at T → 0 first; arm PI. **DDC rider (PI 1f, pre-authorised):** if D5 reads DDC separated adverse again (B − own A, 95 % CI entirely below 0, as R0d's −0.92 [−1.45, −0.48]), **g5 + DDC as a third constrained term is the committed next lever**, in addition to the lever above. |
 | O4+O5 / SLACK | as addendum R2 | as addendum R2 | both first steps (0 GPU), then the PI |
-| **O1 SAFE + ABOVE FRONTIER** | S and G | the pick-aligned dual repairs the trade at a gain above the free frontier | **Stage 2**: R1e-pick, then R1o, then bar L, then navtest E plus the I-rep. ADOPT puts the constrained EPO at T = 0.01 into P6's scorer loss |
-| **O2 SAFE, BELOW FRONTIER** | S holds; PDMS lower bound > 0; point < G(t) | the pick-aligned dual over-pays | if the S margin (min over checks of TTC point + 0.5) ≥ 0.15: **R0e-pick at T = 0.02**. That is the lighter dual, which understates the pick by about 0.1 more at u1000 (A2R2): it spends measured margin on gain. Otherwise there is no margin to spend: SAFEOBJ O2(ii), the fixed λ at this run's measured mean λ over u250–u1000. Both PI |
-| **O3-a SAFE, FLAT: GAIN CROWDED OUT** | S holds; PDMS lower bound ≤ 0; λ > 0 on ≥ 50 % post-ramp (**the dual binds, TTC holds on RL-val, the PDMS gain vanishes**); AND **D2 J-crowding**: R0e-pick's in-sample `epo_J` mean over u751–1000 has risen from its u25 row by less than half of R0d's rise (R0d: 0.8171 → 0.8766, rise +0.0595, MEASURED, `sft.jsonl`) | the constraint's ~1/T gradient took the clipped update budget (`clip_grad_norm_ 1.0`, `:1693`); EPO never built its gain | **R0e-pick at T = 0.02**: half the gradient scale, still aligned on A2R2's committed bar. 0 GPU first: the D2 numbers. Arm PI |
-| **O3-b SAFE, FLAT: GAIN BUILT, NOT TRANSFERRED** | as O3-a, but D2 is not crowded | EPO built its in-sample gain and the pick-safe part of it did not reach RL-val. SO-CARVE already MEASURED EP's in-sample excess for R0 (+1.665 [+0.17, +3.32] at u1000). Re-ranking a fixed fan has no TTC-neutral gain at this representation | the **representational / proposal-side lever**: P6 trunk with hard negatives (P6-HN step 0, `8bb47ad`: expert-proximate hard negatives in 33.7 % of held-out sets; reach ESTIMATED 0.29-0.32 PDMS). The logit mixture is **not** a fallback: it was read FP-NONE on navtest. PI |
-| **O3 SLACK** | S holds; PDMS lower bound ≤ 0; λ > 0 on < 50 % post-ramp | S held while the dual was mostly idle: the arm is an R0-like draw, not a lever reading | report, with the seed floor beside it; no lever claim. Next: Stage 2's R1e-pick, for replication only (PI) |
+| **O1 SAFE + ABOVE FRONTIER** | S and G | the pick-aligned dual repairs the trade at a gain above the free frontier | **Stage 2**: R1e-pick, then R1o, then bar L, then navtest E plus the I-rep. ADOPT puts the constrained EPO at T = 0.01 into P6's scorer loss. **DDC rider (PI 1f, pre-authorised):** if D5 reads DDC separated adverse again (B − own A, 95 % CI entirely below 0, as R0d's −0.92 [−1.45, −0.48]), **g5 + DDC as a third constrained term is the committed next lever**, in addition to the lever above. |
+| **O2 SAFE, BELOW FRONTIER** | S holds; PDMS lower bound > 0; point < G(t) | the pick-aligned dual over-pays | if the S margin (min over checks of TTC point + 0.5) ≥ 0.15: **R0e-pick at T = 0.02**. That is the lighter dual, which understates the pick by about 0.1 more at u1000 (A2R2): it spends measured margin on gain. Otherwise there is no margin to spend: SAFEOBJ O2(ii), the fixed λ at this run's measured mean λ over u250–u1000. Both PI. **DDC rider (PI 1f, pre-authorised):** if D5 reads DDC separated adverse again (B − own A, 95 % CI entirely below 0, as R0d's −0.92 [−1.45, −0.48]), **g5 + DDC as a third constrained term is the committed next lever**, in addition to the lever above. |
+| **O3-a SAFE, FLAT: GAIN CROWDED OUT** | S holds; PDMS lower bound ≤ 0; λ > 0 on ≥ 50 % post-ramp (**the dual binds, TTC holds on RL-val, the PDMS gain vanishes**); AND **D2 J-crowding**: R0e-pick's in-sample `epo_J` mean over u751–1000 has risen from its u25 row by less than half of R0d's rise (R0d: 0.8171 → 0.8766, rise +0.0595, MEASURED, `sft.jsonl`) | the constraint's ~1/T gradient took the clipped update budget (`clip_grad_norm_ 1.0`, `:1693`); EPO never built its gain | **R0e-pick at T = 0.02**: half the gradient scale, still aligned on A2R2's committed bar. 0 GPU first: the D2 numbers. Arm PI. **DDC rider (PI 1f, pre-authorised):** if D5 reads DDC separated adverse again (B − own A, 95 % CI entirely below 0, as R0d's −0.92 [−1.45, −0.48]), **g5 + DDC as a third constrained term is the committed next lever**, in addition to the lever above. |
+| **O3-b SAFE, FLAT: GAIN BUILT, NOT TRANSFERRED** | as O3-a, but D2 is not crowded | EPO built its in-sample gain and the pick-safe part of it did not reach RL-val. SO-CARVE already MEASURED EP's in-sample excess for R0 (+1.665 [+0.17, +3.32] at u1000). Re-ranking a fixed fan has no TTC-neutral gain at this representation | the **representational / proposal-side lever**: P6 trunk with hard negatives (P6-HN step 0, `8bb47ad`: expert-proximate hard negatives in 33.7 % of held-out sets; reach ESTIMATED 0.29-0.32 PDMS). The logit mixture is **not** a fallback: it was read FP-NONE on navtest. PI. **DDC rider (PI 1f, pre-authorised):** if D5 reads DDC separated adverse again (B − own A, 95 % CI entirely below 0, as R0d's −0.92 [−1.45, −0.48]), **g5 + DDC as a third constrained term is the committed next lever**, in addition to the lever above. |
+| **O3 SLACK** | S holds; PDMS lower bound ≤ 0; λ > 0 on < 50 % post-ramp | S held while the dual was mostly idle: the arm is an R0-like draw, not a lever reading | report, with the seed floor beside it; no lever claim. Next: Stage 2's R1e-pick, for replication only (PI). **DDC rider (PI 1f, pre-authorised):** if D5 reads DDC separated adverse again (B − own A, 95 % CI entirely below 0, as R0d's −0.92 [−1.45, −0.48]), **g5 + DDC as a third constrained term is the committed next lever**, in addition to the lever above. |
 
 * **R1e-pick** uses the same table.
   * R0e-pick O1 with R1e-pick not O1 reads **NOT REPLICATED**: no lever claim, R1o does not run, and the next step is O2's.
@@ -281,15 +297,16 @@
 
 | item | arithmetic | A40-h |
 |---|---|---|
-| R0e-id to u250, then stop | 0.34 + (467 + 2,425 + 460) / 3600 | **1.27** |
+| R0e-id to u1000 (PI 1c) | 0.34 + 12,007 / 3600 | **3.68** |
 | R0e-pick to u1000 | 0.34 + 12,007 / 3600 | **3.68** |
 | R0e-pick K3-killed at u500 | 0.34 + (467 + 4,850 + 920) / 3600 | 2.07 |
 | R0e-pick K3-killed at u750 | 0.34 + (467 + 7,275 + 1,380) / 3600 | 2.87 |
-| **Stage 1** (R0e-id + R0e-pick) | 1.27 + 3.68 | **≤ 4.95** (3.34 if killed at u500) |
+| **Stage 1** (R0e-id + R0e-pick) | 3.68 + 3.68 | **≤ 7.36** (5.75 if R0e-pick is killed at u500: 3.68 + 2.07) |
 | Stage 2 (R1e-pick + R1o) | 2 × 3.68 | ≤ 7.36 |
-| **maximum** | | **12.31** |
-| option: R0e-id run to u1000 (a full same-run replicate of R0d at all four checks) | +3.68 − 1.27 | +2.41 (PI) |
+| **maximum** | 7.36 + 7.36 | **14.72** |
+| adopted (PI 1c), included above: R0e-id run to u1000 instead of the draft's u250 stop, a full same-seed replicate of R0d at all four checks | +3.68 − 1.27 | +2.41 |
 
+* The totals are sums of the rounded rows, as in the draft. Unrounded: Stage 1 7.35, maximum 14.70.
 * Not in these totals:
   * The I-rep. INHERITED: PLAN_IREP proposes the A40, about 5 h of mostly idle GPU.
   * The navtest E reads (PLAN_E). Both run only after Stage 2.
@@ -336,7 +353,7 @@
 | B1 | SO-PICKCON landed (trainer `869b1862…`, gate, `test_rl_pickcon.py`); `PK/SHIP_MANIFEST.json` regenerated | Master Mind (in progress) |
 | B2 | the profile registers R0e-id (0.1), R0e-pick (0.01), R1e-pick (0.01, seed 1) and R1o, with must-name lists including `--epo-con-temp`; `eval/rls1e_arms.json` with the §2.1 self-check | build agent |
 | B3 | `safeobj_outcome.py` per-arm registered constraint T, plus its mutation tests (§4) | build agent |
-| B4 | the R0e-id stop-after-u250-dump (supervisor, kill by explicit PID, never `pkill -f`) with a test; **or** the PI accepts R0e-id to u1000 (+2.41 A40-h) | build agent / PI |
+| B4 | **DROPPED (PI 1c, 2026-10-18):** R0e-id runs to u1000, so no supervisor stop-after-u250-dump is built | none |
 | B5 | `safeobj-gate --dirs` re-pointed to R0e-pick / R1e-pick for R1o | build agent |
 | B6 | the D1-D3 reader, in the outcome package | build agent |
 | B7 | the D4 read (NC alignment at 0.01) | Research Lab, CPU |
```
