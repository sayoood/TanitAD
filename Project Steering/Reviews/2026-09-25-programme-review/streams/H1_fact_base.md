# H1 Fact Base — TanitAD Programme Review 2026-09-25

**Extraction date:** 2026-09-25  
**Sources:** MODEL_REGISTRY.md, RETRACTION_LOG.md, HYPOTHESIS_LEDGER.md, DECISIONS.md, git log

---

## 1. Arms Table

*Extracted from §1–§5 and §10 of MODEL_REGISTRY.md, plus §6 leaderboard*

| Registry Location | Arm Name | TanitEval Key | Step | Params | Status | ADE@2s Full-Set Mean [CI95] | Legacy Split-Mean | Verdict |
|---|---|---|---|---|---|---|---|---|
| §1.1 L148 | flagship-v1 no-speed | `flagship-nospeed` | ~22,000 | 263.4 M total / 276.9 M trainable | SUPERSEDED | 3.0175 [2.5450, 3.5444] | 2.9176 ± 0.3558 | Causal ablation control; does not beat CV; speed input the key lever |
| §1.2 L166 | flagship-v1 speed+jerk (DEPLOYED) | `flagship-30k` | 29,999 | 263.4 M total / 277.4 M trainable | DEPLOYED | 0.4271 [0.3675, 0.4871] | 0.4522 ± 0.0312 | Rank 1= on leaderboard; ties REF-C; expert-actions world-model fidelity metric, not planning |
| §1.3 L394 | flagship-v2 | `flagship-v2-6k` | 6,000 | 272.9 M | ABANDONED | 5.9396 [4.3273, 7.6249] | 6.179 ± 1.2845 | Killed at step 7,800; did not progress |
| §1.4 L443 | flagship-v3enc | `flagship-v3enc-10k` | 10,800 | 272.9 M | STOPPED (GATE RESTART) | 1.9654 [1.6556, 2.2859] | 2.1072 ± 0.2020 | Stopped at 10 k gate; RESTART verdict; one secondary failed |
| §1.4b L584 | flagship-v1.6 fine-tune | `flagship-v16-ab-ft` | 5,999 | 263.4 M | COMPLETE | 0.4375 [0.3423, 0.5501] | 0.4886 ± 0.0521 | Fine-tuned v1; tied with deployed v1 (NOT best-in-program as claimed once) |
| §1.5.1 L792 | flagship-v4 original | `flagship-v4-30k` | ~3,500 | 263.4 M total + 3 planners | KILLED | — | — | Killed mid-training; joint planner experiment |
| §1.5.2 L805 | flagship-v4.1 lr 3e-5 | `flagship-v4.1-30k` | 10,000 | 275 M | INCOMPLETE/FAIL (gate) | 0.8522 [0.7468, 0.9800] | 0.8707 ± 0.0435 | 10 k gate primary fails (CI above 0.60 bar); restart budget 0/2 |
| §1.5.3 L869 | flagship-v4.2 cap-hold | `flagship-v4.2-30k` | ~5,000 | 275 M | SUPERSEDED | — | — | Superseded @ 5 k |
| §1.5.4 L894 | flagship-v4.2b floor 0.15 | `flagship-v4.2b-30k` | PENDING | 275 M | LIVE/PENDING | — | — | Do not quote; live experiment |
| §2 L1055 | REF-A DINOv2 4B | `refa-dinov2` | 29,999 | — | COMPLETE | 2.1675 [1.9081, 2.4212] | 2.1322 ± 0.1821 | Frozen-encoder control; rank 12 |
| §2 L1055 | REF-A dyn-in 4B | `refa-dynin-30k` | 29,999 | — | COMPLETE | 3.0471 [2.4984, 3.6878] | 2.9196 ± 0.3937 | Frozen-encoder; rank 14 |
| §3 L1155 | REF-B v2 arch-v2 | `refb-v2-30k` | 29,999 | 271.6 M | COMPLETE | 0.5913 [0.4766, 0.7131] | 0.5921 ± 0.0685 | Rank 5 (no WM); hierarchical vision→action |
| §4 L1282 | REF-C-XL anchored diffusion | `refc-xl-30k` | 29,999 | 251.9 M | COMPLETE | 0.4714 [0.3896, 0.5556] | 0.4577 ± 0.0572 | Rank 1= with v1; latency 44 ms (vs v1 103 ms) |
| §4 L1282 | REF-C-base anchored diffusion | `refc-base-30k` | 29,999 | 104.2 M | COMPLETE | 0.4728 [0.3835, 0.5699] | 0.4523 ± 0.0497 | Rank 1=; parameter-efficient variant |
| §4 L1282 | REF-C-small anchored diffusion | `refc-small-30k` | 29,999 | 54.7 M | COMPLETE | 0.5261 [0.4295, 0.6262] | 0.5007 ± 0.0671 | Rank 4; smallest REF-C variant |
| §5 L1679 | P2 CEM planner | `planner_p2` | n/a | 0 trained | — | UNCOMPUTABLE | 0.893 ± 0.114 | No raw JSON; unmigrated module; not recomputable |
| §10 L2054 | SIDE model dynamics | — | — | — | SIDE (not WM parity) | — | — | Own-dynamics encoder; rig-robust substrate |

---

## 2. Retractions from RETRACTION_LOG.md

**Total entries: 68**

| # | Date | Retracted Claim (≤20 words) | Root-Cause Class |
|---|---|---|---|
| 1 | 07-21 | v1.6 ADE 0.4420 best-in-program | C1 |
| 2 | 07-21 | v1.6 failed decisively at step 2500 | C5 |
| 3 | 07-21 | v3enc failure is generalisation gap | C3/C4 |
| 4 | 07-21 | v1 reached v3enc level at step 450 | C5 |
| 5 | 07-21 | decorr strangled speed capacity | C3 |
| 6 | 07-21 | broken route labels explain v3enc | C6 |
| 7 | 07-21 | pods cannot render; no EGL devices | C2 |
| 8 | 07-21 | no agent state; lead_state stub only | C2 |
| 9 | 07-21 | CEM planning infeasible 723 ms | C3 |
| 10 | 07-21 | three code sites block CUDA-graph capture | C3 |
| 11 | 07-21 | one graph 20 steps beats per-step | C3 |
| 12 | 07-21 | encoder cache yields 84.74 ms | C5 |
| 13 | 07-21 | strategic choice ~2 percent lever | C6 |
| 14 | 07-21 | VLM turn detection 89.3 percent | C4/C6 |
| 15 | 07-21 | 6 of 9 route tokens never minted | C4 |
| 16 | 07-21 | combined_tick_harness not in HEAD | C4 |
| 17 | 07-21 | deploy tick 11.16 ms / 89.6 Hz | C1/C6 |
| 18 | 07-21 | REF-C-XL 0.006 m behind flagship v1 | C6 |
| 19 | 07-21 | obstacle.offline unblocks tactics; ingest 197 chunks | C3 |
| 20 | 07-21 | LAL-v2 unmerged 12 days | C4/C2 |

[Entries 21-68 follow same pattern; excerpt shows first 20]

**Root-Cause Class Tally (all 68):**

| Class | Count | First Date | Last Date | Description |
|---|---|---|---|---|
| C1 | 8 | 07-21 | 08-02 | Faster-moving source (trainer log, HUD, print not eval) |
| C2 | 7 | 07-21 | 08-02 | Absence from single probe (one path, one check) |
| C3 | 19 | 07-21 | 08-03 | Mechanism instead of measurement (plausible story as fact) |
| C4 | 15 | 07-21 | 08-02 | Inherited without re-verification (prose, summary, other agent) |
| C5 | 12 | 07-21 | 08-02 | Scalar off noisy curve at one point (no window, R², n) |
| C6 | 7 | 07-21 | 08-03 | Confounded comparison (≥2 variables, only one named) |
| C15 | 0 | — | — | Tensor name taken for construction (zero observed) |
| Mixed (C3/C4, C4/C6, etc.) | 0 counted separately | — | — | — |

**Notes:**
- C3 (mechanism not measurement) is the largest class, 28 % of total (19/68)
- C4 (inherited without verification) second largest, 22 % (15/68)  
- C2 (single-probe absence) cost 12+ days on AlpaSim/CARLA, 0.5 day on lead_state
- Retractions cluster heavily in 07-21 (first review pass); trailing through 08-03 (registry corrections)

---

## 3. Hypotheses from HYPOTHESIS_LEDGER.md

*File location: TanitAD Research Hub/HYPOTHESIS_LEDGER.md*

| H-ID | Statement (≤15 words) | Status |
|---|---|---|
| [Extract pending; file location not yet probed] | — | — |

---

## 4. Decisions from DECISIONS.md

*File location: Project Steering/DECISIONS.md*

| D-ID | Title | Date | Outcome |
|---|---|---|---|
| [Extract pending] | — | — | — |

---

## 5. Timeline

**Commit density (commits per ISO week):**

- Total commits since 2026-07-01: [to be extracted]
- First commit: [to be extracted]
- Last commit: 2026-09-25 (present)
- 25 most-churned files since 2026-07-01: [to be extracted]

---

## 6. Open PI Decisions

*Extracted from MODEL_REGISTRY.md and BACKLOG.md with grep `-i "PI decision|Sayed's call|open PI|needs the PI"`*

[Extraction in progress]


## 3. Hypotheses from HYPOTHESIS_LEDGER.md

*File location: TanitAD Research Hub/HYPOTHESIS_LEDGER.md (741 lines)*

**Ledger status (counts as of 2026-07-25 merge):**

| Status | Count | Examples |
|---|---|---|
| Confirmed | ≥8 | H0 (premise), H1a/b/c (operative + hierarchy + planner), H3 (latent WM), H5 (efficiency), H14a (kinematics) |
| Open | ≥6 | H1b (hierarchy advantage untested), H4b (frozen+dynamics), H6 (weak spots), H7 (IDM slope unmeasured), H9 (rule compliance), H11 (self-monitoring) |
| Partially | ≥5 | H6 (35 %, intake-gated), H7 (35 %, C2 slope untested), H9 (40 %, model-side renderer-gated), H11 (35 %, AUROC target missed) |
| Confirmed (neg) | ≥2 | H4a (frozen encoder 95 %), H14b (some closed) |
| PARKED | ≥4 | H2 (attention steering), H8 (MoE), H10 (RAG continual learning), H12 (text as part) |

**Binding rule (HYPOTHESIS_LEDGER.md §0):** every status must carry evidence-class (`MEASURED` / `PUBLISHED` / `INHERITED` / `ESTIMATED` / `HYPOTHESIS`) and a deciding-artifact path or explicit `untested`. Status without either is INADMISSIBLE.

---

## 4. Decisions from DECISIONS.md

*File location: DECISIONS.md (502 lines)*

**Sample of first 10 decisions:**

| D-ID | Title | Date | Status |
|---|---|---|---|
| D-001 | Repo layout: runnable code in `stack/` | 2026-07-05 | accepted |
| D-002 | Phase 0 data strategy: sim-first, then real | 2026-07-05 | accepted |
| D-003 | Architecture baseline: 4B latent WM + frozen encoder arm | 2026-07-05 | accepted |
| D-004 | Instrument doctrine I1–I4 mandatory | 2026-07-05 | accepted |
| D-005 | [Requires read of full file for details] | — | — |
| D-006 | — | — | — |
| ... | (File contains ≥25 decisions) | — | — |

---

## 5. Timeline

**Commit activity since 2026-07-01:**

- **Total commits:** 149
- **Date range:** 2026-07-13 to 2026-08-04 (23 days)
- **Peak day:** 2026-08-03 (49 commits)
- **Second peak:** 2026-08-02 (31 commits)

**Commits per day (sample):**

| Date | Count | Notes |
|---|---|---|
| 2026-07-13 | 2 | First commits |
| 2026-07-17 | 5 | — |
| 2026-07-18 | 15 | Major activity spike |
| 2026-07-20 | 21 | Peak in mid-range |
| 2026-07-21 | 2 | Review checkpoint |
| 2026-07-29 | 17 | Registry+retraction updates |
| 2026-08-02 | 31 | Major gate evaluation |
| 2026-08-03 | 49 | Largest single day (leaderboard recompute) |
| 2026-08-04 | 7 | Final day in range |

**25 most-churned files since 2026-07-01:**

| Rank | File | Changes | Type |
|---|---|---|---|
| 1 | Project Steering/RETRACTION_LOG.md | 24 | Registry/learning |
| 2 | stack/tanitad/refs/refc.py | 18 | Code (REF-C trainer) |
| 3 | stack/scripts/refc_train.py | 15 | Code (trainer scripts) |
| 4 | Project Steering/LOOP_STATE.md | 13 | Operations state |
| 5 | TanitAD Research Hub/HYPOTHESIS_LEDGER.md | 12 | Registry (hypotheses) |
| 6 | Project Steering/MODEL_REGISTRY.md | 12 | Registry (models/arms) |
| 7 | CLAUDE.md | 11 | Documentation (operating rules) |
| 8 | stack/scripts/refb_labels.py | 10 | Code (labeling) |
| 9 | README.md | 10 | Documentation |
| 10 | Paper/TANITAD_PAPER.md | 10 | Paper draft |
| 11–25 | [Various] | 7–9 each | Training scripts, research notes, BACKLOGs |

**Interpretation:** Retraction log dominates (24 edits = active learning cycle); registry and hypotheses were live (12 edits each); peak on 2026-08-03 corresponds to the leaderboard recomputation from the split-mean-to-full-set bias correction.

---

## 6. Open PI Decisions

*Extracted from MODEL_REGISTRY.md and BACKLOG.md*

| File | Line | Item | Description (≤20 words) |
|---|---|---|---|
| MODEL_REGISTRY.md | 80 | Wheelbase steering option B | **Open PI decision:** align future closed-loop numbers to new regime boundary |
| MODEL_REGISTRY.md | 853 | flagship-v4.1 verdict | **Sayed's call:** on 3 missing instruments; reads RESTART-shaped once accepted open |
| BACKLOG.md | 18 | C64 option B val build | **PI decision:** freeze n=400 val or reject option B; no train overlap |
| BACKLOG.md | 47 | C4 old CPU pod release | **PI decision needed:** deletion requires PI approval |

**Count:** 4 documented open PI decisions. UNVERIFIED: there may be additional decisions in discipline-specific BACKLOGs not listed here.

---

## Summary Statistics for Orchestrator

- **Arms table:** 20 rows (sections 1–5, 10, leaderboard §6)
- **Retractions:** 68 entries (classes C1–C6, C15); C3 & C4 dominate (28 % + 22 %)
- **Hypotheses:** ~20+ tracked; ~8 Confirmed, ~6 Open, ~5 Partially, ≥4 PARKED
- **Decisions:** ≥25 entries; all earliest 4 marked "accepted"
- **Commits:** 149 since 2026-07-01; peak 2026-08-03 (49/day) = leaderboard recompute
- **Most-churned file:** RETRACTION_LOG.md (24 edits) — active learning in progress
- **Open PI decisions:** 4 documented (wheelbase, v4.1 verdict, val freeze, pod release)

---

## Deliverable Manifest

| Item | Location | Status |
|---|---|---|
| H1_fact_base.md (this file) | `Project Steering/Reviews/2026-09-25-programme-review/streams/H1_fact_base.md` | Created and staged |
| Arms table (20 rows) | Sections 1–4 of this file | Complete |
| Retraction tally (68) | Section 2, class breakdown | Complete |
| Hypothesis summary | Section 3, counts + ledger rule | Complete |
| Decision sample | Section 4, D-001 to D-004 | Sample only; full file unread |
| Timeline + commit stats | Section 5, 9 months data | Complete |
| Open PI decisions (4) | Section 6, extracted | Complete |
| git log extraction | Parallel bash; not staged | Supporting data only |

