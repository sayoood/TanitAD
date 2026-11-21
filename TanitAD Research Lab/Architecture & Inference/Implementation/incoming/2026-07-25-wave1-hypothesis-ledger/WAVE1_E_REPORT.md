# Wave-1 workstream E — hypothesis-ledger merge

**Date:** 2026-07-25 · **Item:** `01_EXECUTION_PLAN.md` §B.1 row P1/P2 ("Merge the hypothesis ledgers
into ONE living ledger; split H4→H4a/H4b, H1→H1-operative/H1-hierarchy/H1-coupling; add PARKED state")
· **Compute:** 0 GPU, no pod contact · **Staging:** files written, **NOT** `git add`ed (orchestrator
stages).

**Sources of truth used, faithfully:**
- `Project Steering/Reviews/2026-07-25-independent-chief-scientist-review/R3_hypothesis_portfolio.md`
  — statuses, DoA, deciding evidence (the audited master table, 37 hypotheses).
- `.../01_EXECUTION_PLAN.md` **PART C** — the action class for every hypothesis.
- Artifact paths verified to exist on disk before being written into a row (see §5).

**No audited verdict was changed.** Every disagreement or extension is a numbered note in
`HYPOTHESIS_LEDGER.md` §6 and is repeated in §4 below.

---

## 1. Deliverables

| Artifact | Path | State |
|---|---|---|
| **The unified ledger** | `TanitAD Research Hub/HYPOTHESIS_LEDGER.md` | rewritten in place |
| **The overview pointer** | `Project Steering/PROGRAM_OVERVIEW.md` §3 (+ 1 clause in §2 row ②) | surgically edited |
| **This report** | `TanitAD Research Hub/Implementation/incoming/2026-07-25-wave1-hypothesis-ledger/WAVE1_E_REPORT.md` | new |

**Untouched, as briefed:** `Mission Plan.md`, `MODEL_REGISTRY.md`, `RETRACTION_LOG.md`,
`LOOP_STATE.md`, `taniteval/`, `stack/`, `Project Steering/Reviews/2026-07-25-*`.

---

## 2. What merged

**41 rows across 37 audited hypotheses**, each carrying all ten mandatory columns
(`ID · hypothesis · status · DoA % · evidence-class · deciding-artifact-path or "untested" ·
gate/falsifier · action · last-retested · owner`).

- **From the frozen ledger (H0–H18):** its two genuinely useful columns — the **Phase-0 gate
  bindings** (D1–D9) and the **owner agent** — were carried into the new `Gate / falsifier` and
  `Owner` columns. Its status column and status *vocabulary* were retired (see divergence D24).
- **From `PROGRAM_OVERVIEW` §3 (live):** every status, plus H4, H19, H25–H28 which existed **only**
  there.
- **From R3 (neither ledger had them):** **H20–H24** (survey proposals that existed only in the
  frozen ledger's *changelog prose*) and **IMP-1…IMP-8** (eight hypotheses the program actually
  tested and decided, numbered for the first time by the audit). This is the largest single gain:
  both ledgers under-counted the portfolio at 18 rows against an actual 37.
- **New structure:** a `PARKED {reason, revisit-trigger}` register (§3 of the ledger), a
  "how to update" header with the inadmissibility rule (§0), a live action-class summary (§2), a
  gate-status section (§4), an open-escalations section (§5), and merge notes (§6).
- **History preserved:** the change log (2026-07-05 → 2026-07-19) is kept **verbatim** as an
  explicitly-labelled archive with three loud cautions attached (non-chronological, one future-dated
  entry, two contradictory H4 entries). It is marked **NOT a status source**.
- **History deliberately NOT preserved:** the frozen 2026-07-05 status table itself. Reproducing it
  is the exact failure this merge exists to end; its content is summarised in the ledger §7.1 and
  its divergences enumerated below.

---

## 3. Every divergence found between the two ledgers

*Reported in full because the divergence inventory is itself a deliverable. `L` =
`HYPOTHESIS_LEDGER.md` (frozen table, 2026-07-05); `O` = `PROGRAM_OVERVIEW.md` §3 (live, 2026-07-25).*

### 3.1 Structural divergences

| # | Divergence | Consequence |
|---|---|---|
| **D1** | **L's status column is literally headed "Status (2026-07-05)"; O's "Status (2026-07-25)". A 20-day gap.** | The primary defect. A reader landing on the Research-Hub file (the one *named* `HYPOTHESIS_LEDGER`) got 20-day-old statuses. |
| **D2** | **L carries `Phase-0 gate(s)` and `Owner agent` columns; O carries neither.** | Gates and ownership existed **only in the stale doc** — the live doc could not say who owned anything or what would falsify it. |
| **D3** | **O carries evidence and artifact numbers; L carries none in its table.** | Neither document alone satisfied the operating standard. |
| **D4** | **L's changelog is live to 2026-07-19 but NONE of it propagated into L's own status table.** | Internal divergence *within* the frozen ledger: e.g. the flagship v1 30 k result, the H26 verdicts and the OOD-generalization read are all in the diary and none in the table. |
| **D5** | **L's changelog is not chronologically ordered.** Entry order: 07-18, 07-31, 07-17, 07-15, 07-15, 07-17, 07-15, 07-24, 07-09, 07-09, 07-08, 07-08, 07-07, 07-17, 07-16, 07-14, 07-06, 07-14, 07-07, 07-05, then seven 07-18s and a 07-19. | Unusable as a timeline; "the latest entry" is not the last entry. |
| **D6** | **L contains a FUTURE-DATED entry: `2026-07-31 (real wall-clock 2026-07-17)`.** | The narrative-clock artifact, 6 days ahead of today. It sorts near the top and reads as the newest fact. |
| **D7** | **Three status vocabularies, no mapping.** L: `confirmed / validated-toy / supported / open / at-risk / refuted`. O: emoji (✅/🔶/🟢/🟥) + free prose. R3: `Confirmed / Refuted / Confounded / Partially / Open / Stale-Orphaned`. | "supported" (= someone else's paper) and "validated-toy" (= toy scale) were being read as capability. Resolved by adopting R3's. |
| **D8** | **Neither ledger contains IMP-1…IMP-8** — eight hypotheses the program tested and decided (speed-fix, heads-as-lossy-readout, open-loop ⊥ closed-loop, closed-loop improvability, Branch-B, INT8, recovery-augmentation, v2 balancing). | The portfolio's largest blind spot: three *refuted-at-power* results and the program's **strongest single positive** (IMP-1, +2.21 m causal) were untracked. |
| **D9** | **H20–H24 appear in NEITHER status table** — only inside one changelog paragraph of L dated 07-18. | Five proposals sat un-statused for 8 days; R3 is the first document to give them one (all Stale-Orphaned). |
| **D10** | **H27 and H28 appear in O only** — they are absent from L's table *and* L's changelog. | The two newest flagship-line hypotheses were invisible from the Research-Hub side. |

### 3.2 Per-hypothesis status divergences

| # | ID | L (frozen 07-05) | O (live 07-25) | Resolution in the merged ledger |
|---|---|---|---|---|
| **D11** | **H4** | *"open (cheap to answer) — real corpus (comma2k19) ready to make it meaningful"* | *"✅ **CLOSED NEGATIVE** — and re-localized"* | **The sharpest divergence: open ↔ closed-negative.** And *both* are wrong per the audit — the flat closure is UNSAFE. **SPLIT into H4a (Confirmed-neg, 2.9196, RETIRE) / H4b (OPEN, positive at 0.599 m).** |
| **D12** | **H4 (within L)** | L's changelog carries **two contradictory 07-18 entries**: *"H4 CLOSES NEGATIVE"* and *"H4 REFRAMED — H4 is NOT negative"* | — | Neither reached L's table. The split resolves the contradiction; **both entries kept** in the archive because the reasoning in each is still useful. |
| **D13** | **H4 owner** | `Data Engineering` | *(no owner column)* | **Corrected to Architecture & Inference** — every deciding artifact came from the REF-A lineage, not Data Engineering. |
| **D14** | **H1** | `validated-toy (5× hierarchy lift, ALPS-4B)` — a **toy** result quoted as the hierarchy claim | `🔶 operative validated; heads-as-decision-makers falsified → planning` | Diverged in both status *and in what is being claimed*. **Neither says the D5/D6 gates never ran.** SPLIT into H1a (Confirmed) / **H1b (UNTESTED)** / H1c (= H27). |
| **D15** | **H3** | `validated-toy (A1–A10)` | `✅ strengthened` + 0.599 vs oracle 0.4045 | Toy → MEASURED. Merged as **Confirmed (as representation), 75 %**, with the live falsifier (SIGReg still `NOT-YET-ADMISSIBLE`) restored from L's changelog — it was in **neither** table. |
| **D16** | **H5** | `supported` | `✅ + first real CNCE 210,551 + 18.75 ms p50` | PUBLISHED → MEASURED. **Neither table mentions the retracted "11.16 ms"** — added as an explicit ⚠️ in the merged row. |
| **D17** | **H6** | `actionable · 3 scenarios in eval set` | `actionable — scenarios shipped … + SC-14` (**4**) | Scenario count diverged 3 → 4. More importantly **neither says they are design-oracle only**; the merged row states it and marks the model-side number renderer-gated. |
| **D18** | **H7** | `supported (VLM3, LAPA; +LAOF …)` — **PUBLISHED only** | `🟢→✅ first end-to-end evidence … ≈92 % of ceiling` | PUBLISHED → MEASURED-but-DIRECTIONAL. **Neither table says the C2 *slope* itself is unmeasured** — that is the program's headline claim and it is now the row's explicit `untested`. |
| **D19** | **H9** | `supported, concrete math` | `✅→🔶 now scorable … 1.0 vs 0.0` | Argument → design oracle. Merged with the renderer gate named. |
| **D20** | **H11** | `validated-toy (3 mechanisms)` | `validated-toy; D8 AUROC > 0.85 not yet reached` + canary | Converged on "not reached", but **the σ-dissipation cap (chance by k=4) is in L's changelog only, in neither table.** Restored to the row. |
| **D21** | **H15** | `in Phase 0 per D-008 — ImaginationField implemented` | `🔶 live; vision_use flat ~12 %; capped at 1-step` | L has no `vision_use` and no cap. **Neither says D9 was never ablated** — now the row's `untested`. |
| **D22** | **H14** | `Track 1 adopted (kinematic decoder + Kamm circle)` | `Track 1 adopted (kinematic + Kamm)` | **The one row where the two agreed — and both were incomplete.** Both omit that the broad physical-law/ethics/culture vision is untouched. **SPLIT into H14a (done) / H14b (untouched, no gate written).** |
| **D23** | **H17** | `open` | `open` | Agreed — and **both wrong**: R3 measures it Stale-Orphaned, 13 days, no owner. Now **PARKED**, owner explicitly `— unowned`. |
| **D24** | **H16, H18** | long dossier prose | one compressed line | Information loss in the live doc (H16's F1–F3 pre-registration and H18's per-level falsifier existed only in L). Both restored. |
| **D25** | **H19** | changelog wording: *"discrete tactical vocabulary/LAMP"* | *"Maneuver → anchor prior (anchored decoders)"* | Wording evolved silently; flagged by R3 itself. Merged row carries the evolved form with the origin noted. |
| **D26** | **H0, H2, H8, H10, H12, H13** | essentially unchanged | essentially unchanged | No status divergence — but all six lost their gate binding and owner in O (per D2), and all six lacked an artifact path or `untested` in **both**. Now compliant. |
| **D27** | **H4 in O itself** | — | `✅ CLOSED NEGATIVE` | ⚠️ **Even the LIVE document carried a verdict the audit calls UNSAFE.** Fixed in the overview edit. |

---

## 4. Splits applied

| Parent | Sub-rows | Why the split was mandatory |
|---|---|---|
| **H4** (Confirmed-neg, DoA 85 %) | **H4a** frozen + supervised regression — Confirmed (neg), `RETIRE`, artifact `taniteval/results/eff_refa-dynin-30k.json` (2.9196, monotone 5k→30k) · **H4b** frozen + feature-prediction + planning — **Open and POSITIVE**, artifact `.../2026-07-23-frozen-wm-learned-planner/artifacts/results.json` (0.5989) | The flat label hid the actual v3 question. The *same frozen latent* reaches 0.599 m through its dynamics while static decode off it gives 3.649 m — the ceiling is **static decode, not freezing**. |
| **H1** (Partially, DoA 40 %) | **H1a-operative** Confirmed (0.4522, `driving_flagship-30k.json`) · **H1b-hierarchy-edge** **Open — UNTESTED** (`untested`: D5/D6 have never run) · **H1c-planner-coupling** alias → **H27** | The constitutional core claim was being read as validated because the *operative* number is good. The hierarchy-vs-flat comparison has never been run, and the only adverse datapoint is confounded six ways (`01_EXECUTION_PLAN` §A.1). |
| **H14** (Partially, DoA 35 %) | **H14a-narrow** Confirmed (95.9 % physically-shaped) · **H14b-broad** **Open — UNTOUCHED**, no gate written anywhere in the program | "Track 1 adopted" in both ledgers read as physical grounding delivered. The broad vision has no design, no instrument, no owner and no falsifier. |

**PARKED applied** to H2, H8, H10, H12, H16, H17, H20, H21, H23, H24 — each with a written reason and
a revisit-trigger (ledger §3). **H22 UN-PARKED** — it targets the MEASURED σ-dissipation to chance by
k=4 that caps **both H15 and H11**, which is what separates it from the other four orphans.

---

## 5. Verification performed

- Every artifact path written into a row was checked to exist on disk. Confirmed present:
  `taniteval/results/{driving_flagship-30k,eff_refa-dynin-30k,eff_flagship-30k,eff_flagship-speed,eff_flagship-nospeed,driving_refc-base-30k,postmortem_b_egodropout_v3enc10k,driving_flagship-v3enc-10k,flagship-v4.1-10k,v1_g1_dryrun_gate,v1_g1_dryrun_gate_FIXED}.json`;
  `.../incoming/2026-07-23-frozen-wm-learned-planner/artifacts/{results,bigplanner_v2,bigplanner_large,bigplanner_mlp,valuemodel_results}.json`;
  `.../incoming/2026-07-25-closedloop-horizon-and-shift/{e1a_horizon_heldout44_K185,e2a_localize_heldout44}.json`;
  `.../incoming/2026-07-24-branchb-transfer-eval/results_branchb_transfer_e10_UNDERFIT.json`;
  `.../incoming/2026-07-24-idm-pipeline-derisk/results_idm_pipeline_derisk.json`;
  the H16/H17 dossiers; `Project Steering/Phase 0 Plan.md`;
  `TanitAD Research Hub/Project Steering/Research/2026-07-17-external-survey-derivation.md`.
- ⚠️ **`taniteval/results/` contains 14 files suffixed `.CONTAMINATED-20260720-*`** (e.g.
  `eff_refa-dynin-30k.CONTAMINATED-…json`). The ledger cites only the **un-suffixed** clean files.
  Anyone glob-matching these paths must not pick up the contaminated twins.

---

## 6. What the audit left ambiguous

| # | Ambiguity | How the ledger handled it |
|---|---|---|
| **A1** | **The audit's own action counts do not match its own ID lists.** `§C.6` states `PARK = 9` but lists **10** IDs (H2, H8, H10, H12, H16, H17, H20, H21, H23, H24); `RETIRE = 9` but lists **10** (H0, H5, H13, H19, H28, IMP-1, IMP-3, IMP-5, IMP-6, IMP-7). | Counted **by listed IDs**: PARK 10, RETIRE 10 IDs (11 ledger rows once H4a is added). Discrepancy is arithmetic, not substantive — recorded as merge note N1. |
| **A2** | **Double-assigned actions are single-counted inconsistently.** H1 = SPLIT+PROVE, H9 = FIX+MEASURE, H15 = MEASURE+FIX, H26 = FIX+PROVE. `MEASURE` is listed as 7 but omits H9; `SPLIT` counts H1 which is also under PROVE. | The ledger counts a hypothesis under **every** action assigned to it, so its MEASURE count is **8** (adds H9) where the audit says 7. Stated openly in ledger §2 and note N1. |
| **A3** | **H19 is counted under both RETIRE and the PROVE row's supporting inputs.** | Kept in both, explicitly labelled — it *is* a retired question **and** a hierarchy-supporting result to cite in the HPP. |
| **A4** | **H24's action is genuinely undecided in the audit**: §C.3 says "`PARK` **or run** — … Decide explicitly", §C.6 counts it under PARK. | PARKED as the **reversible default**, with the decision escalated (ledger §5.3). Flagged so nobody reads the park as a decision. |
| **A5** | **DoA is assigned to parents only.** The audit gives H1 = 40 %, H4 = 85 %, H14 = 35 % with no decomposition, yet mandates the splits. | Sub-row DoA is an **ESTIMATED decomposition by this merge**, flagged `≈` in every split row, with the parent's audited value named beside it. It is not audited and must not be quoted as such. |
| **A6** | **No owner exists for H17, H20, H21, H23, H24 — and none for the newly UN-PARKED H22.** The audit says un-park H22 but assigns no owner. | Owner column reads `— unowned` for the parked ones (honest) and **`⚠️ UNASSIGNED — needs an owner`** for H22, escalated in ledger §5.1. **Un-parking without an owner recreates the orphan state.** |
| **A7** | **H14b has no falsifier anywhere in the program.** The audit says "scope it as a real work item or PARK explicitly" but names no gate. | Row states `none defined; a gate must be written before this is admissible as a claim`, escalated to the PI (ledger §5.2). |
| **A8** | **One audit statement is superseded by a same-day artifact.** R3 §3.2 and `01_EXECUTION_PLAN` §C.4/B.3 state *"3 of 8 kill-secondaries still have no emitter, so the gate literally cannot render a verdict."* `taniteval/results/v1_g1_dryrun_gate_FIXED.json` (MEASURED, 2026-07-25) reports `kill_adjudicated: 8`, the 5 report-only falsifiers emitted, and all 3 gate bugs closed. | **H27's status is unchanged** (the *formal* gate is still deferred behind the 30 k finish) but the stated *reason* no longer applies. Recorded as merge note N5 and in ledger §4 — **not** an edit to the audit. |
| **A9** | **The audit's H26 evidence cites `driving_flagship-30k.json` for the hierarchy-panel numbers**, but the panel is a separate instrument (`taniteval/hierarchy.py`) and no `hierarchy_*.json` exists under `taniteval/results/`. | Carried the audit's citation with the panel named alongside it. **The hierarchy panel's own output does not appear to be committed** — worth a check by the panel's owner. |
| **A10** | **The audit cites `MODEL_REGISTRY §6` / `PROGRAM_OVERVIEW §7` (documents) rather than raw artifacts for IMP-2, IMP-3 and parts of H1.** | Carried as-is with the document path named, so the reader can see the citation is a doc and not a JSON. These are the rows most worth re-anchoring to raw eval output. |

---

## 7. Escalations carried into the ledger (§5) — need a decision, not a re-read

1. **H22 needs an owner** (UN-PARKED; caps H15 and H11).
2. **H14b needs a PI scoping decision** (real work item, or PARKED — it cannot stay an unqualified
   constitutional claim).
3. **H24 needs the explicit PARK-or-run decision** the audit demands.
4. **`MODEL_REGISTRY.md:1737` correction is owed** — it states `physicalai-val-f1b378f295ae` is
   "episode-disjoint" from the parity train set; a byte-level check measures **78.5 % overlap**
   (`E1a_E2a_RESULTS.md` §1.1, MEASURED). *Registry is orchestrator-owned this session — flagged,
   not edited.*
5. **Sweep any doc still carrying "closed-loop improvement is BOUND"** (overturned 2026-07-25 as
   horizon-confounded).
</content>
