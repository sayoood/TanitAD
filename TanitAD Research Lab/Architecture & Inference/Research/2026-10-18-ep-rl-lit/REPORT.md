# LIT-EP-RL: how strong NAVSIM methods get high EP, and how post-training raised progress without a safety trade (2026-10-18)

**Provenance.** This is a Research Lab literature pass by a Sonnet agent, briefed by the Master Mind and built on tip `c58a17e`. The agent's harness refused to let it write this file ("subagents should return findings as text"). The Master Mind transcribed this report from the agent's hand-back, without changing any number. The tables are in `raw/` (`published_subscores.csv`, `rl_step_deltas.csv`, `testtime_and_scorer_only_deltas.csv`, `lever_ranking.csv`, `gap_arithmetic.json`).

**Verification of the numbers.** `raw/verify_published_rows.py` auto-checked 71 of 71 rows as PASS against the PDF page; 15 rows were checked manually.

**Evidence classes:**
- Every paper number is PUBLISHED, from a banked primary (`TanitAD Research Lab/Library/`, 11 newly banked and 23 `cited_by` extensions).
- Our own numbers are MEASURED, with the register row named.
- The mapping in section 3 is ESTIMATED; its heuristic is disclosed in `raw/make_tables.py`.

## 1. High-EP mechanisms (PUBLISHED)

**Every row with EP ≥ 88 and NC/TTC intact has EP ABOVE the logged human's 87.5** (DriveZero Tab. 4 p11).
- DriveZero A13 p31: EP 92.72, PDMS 94.55.
- CLOVER 91.7, DrivoR 91.6, DriveSuprim ViT-L 91.3.

**EP above the human is SELECTED, not imitated or distilled.**
- A ViT-S student trained on human trajectories reaches EP 91.99, and on RL-teacher trajectories 91.64 (DriveZero Tab. 7 p13).
- Goal augmentation adds only +0.15 EP (NC +0.57, DAC +0.33, TTC +1.02).
- P2's EP of 82.34 is 5.2 BELOW the human. This agrees with our measurement that the scorer, not the fan, holds the gap: oracle 98.23 vs pick 88.82, `D-REFE-P2-FULLTABLE`, MEASURED.

**No missing mechanism found.**
- Aggregation matches REFe's: the log-linear alternative is closed by `D-REFE-SELRULE-FIT`.
- A dense speed vocabulary is not the gap.

**The attributable EP jumps are on weaker bases:**
- iPad's proposal-anchored encoder: EP 71.5 → 84.2 (Tab. 4 p8).
- DriveSuprim's scorer recipe at ViT-L: EP 85.0 → 91.3 (Tab. 2 p6).
- DrivoR's data ladder: EP 89.1 → 91.6 (Tab. 1 p5).
- TOAD search: EP +0.2 to +7.4 (Tab. 5 p14).

No paper ablates EP by mechanism at DINOv3 ViT-L. The remaining difference is recipe and budget (HYPOTHESIS); P6 tests it.

## 2. RL / preference post-training without a trade (PUBLISHED, same checkpoint before and after; `raw/rl_step_deltas.csv`)

**The EP-for-safety trade follows the SIDE a step updates, not RL itself.**
- **Ranking-side steps trade:**
  - The DDv2 selector swap alone: EP +4.6, TTC −2.5, NC −1.0 (Tab. 9 p13).
  - TOAD on ZTRS: EP +7.4, TTC −0.8, NC −0.4, DAC −0.9.
  - Ours, R0B: EP +1.81, TTC −0.40, NC −0.25 (MEASURED, `D-REFE-FREEPOINT-1`).
- **Proposal-side steps do not trade:**
  - DDv2's RL step with the selector held: EP +0.7, NC +1.1, DAC +0.9, TTC +2.6.
  - Together with the selector swap: EP +5.3, NC +0.1, TTC +0.1, PDMS +3.1.

**No-trade steps, all updating the policy, refiner or encoder:**
- ReCogDrive: EP +6.4, NC −0.2, TTC +0.7.
- CoPhy refiner GRPO: EP +1.8 / +2.1, NC +0.6 / +0.7, TTC +1.3 / +2.3.
- TSA-MPR single-deficit-loser DPO: EP +3.7, NC +0.4, TTC +0.9.
- SimVTA (3 seeds): EP +6.4, NC +0.2, TTC +3.1. Its trajectory-GRPO-only arm LOST NC 0.7, and a shared-encoder video loss restored it.
- BeyondDrive hard negatives: EP +1.4 / +1.2.
- PaIR-Drive: EP +6.1 / +8.9 with NC and TTC kept, but comfort paid (−5.9 / −6.9).

**How far this transfers:**
- The bases had EP 80.3-86.8 (P2: 82.34). That makes the size a fair prior, but transfer to a near-oracle fan is unproven.
- No published SCORER-ONLY step raised EP by ≥ 0.5 without a trade. The best is Versatile-samples at EP +0.1 / +0.3. This absence claim rests on the banked set, the 2607.08072 survey and three searches.
- Constrained GRPO (navhard v2, not navtest) trades the other way: EP −4.0, DAC +13.9.
- The largest no-trade EP step (+6.4) closes 62 % of the 10.38-point EP gap, so RL alone is not enough.

## 3. Mapping onto REFe (ESTIMATED; heuristic in `raw/make_tables.py`)

**The gap is 5.73 PDMS.** EP alone is worth about 4.2, DAC about 2.2 and TTC about 0.2. These are non-additive: their sum is 6.57.
- Ranks 3-10 below lie within the heuristic's noise.
- The reaches overlap. The cheap stack L1-L4 is worth +0.5 to +3.5 (mid 2.0).

| # | lever | PDMS reach | cost | evidence | code touched |
|---|---|---|---|---|---|
| 1 | L1 CEM search on the scorer, admission test first | +0.3 to +1.5 (EP +1 to +3) | 0 A40-h; ≤ 4 dev-box GPU-h | PUBLISHED, transfer ESTIMATED | new `tts_cem.py`; `model.py:699`; `planner.py:537`; `ctx_cache.py` |
| 2 | L2 SR-TTC extension: EP-output pairs, designed negatives, EMA soft targets | +0.1 to +0.8 | CPU | ESTIMATED | `scorer_finetune.py`; `onpolicy_label.py:125` |
| 3 | L4 RL that also updates the encoder under an aux loss | 0 to +0.8 | 10-20 A40-h | one base, HYPOTHESIS | `train.py`; invalidates `ctx_cache` |
| 4 | L5 generator-side step with the scorer held (CLOVER/DDv2 form) | +0.2 to +1.0 | 20-40 A40-h | PUBLISHED, transfer ESTIMATED | generator loss |
| 5 | L3 two-scorer ensemble, P2-s0 + P2-s1 | +0.1 to +0.4 | ~3 dev-box GPU-h | HYPOTHESIS | cross-scoring in the seam |
| 6 | L7 proposal-anchored scorer features (needs a PI architecture decision) | +0.3 to +1.5 | ≥ 60 A40-h | HYPOTHESIS | `model.py` |
| 7 | L6 confidence-gated verifier | +0.1 to +0.5 | ~30 A40-h | PUBLISHED, weaker bases | new verifier head |
| 8 | L10 synthetic-data scale-up | +0.5 to +1.0 | ≥ 150 A40-h | ESTIMATED | data build |
| 9 | L8 DrivoR layer-wise decoder | +0.1 to +0.5 | ~36 A40-h | component not isolated, UNVERIFIED | `score_dec` |
| 10 | L9 accelerated second target | +0.2 to +0.6 | ~160 A40-h | PUBLISHED; −1.6 on v2 | `build_scorer_targets.py` |

SR-TTC (registered, `PREREG_SELRANK_ARM`) is not re-proposed. P6 stays the main path.

**L1 design traps:**
- Scoring is set-conditional (`model.py:699`), so search populations must be scored as FULL sets.
- DriveRL's value-guided search moved the action in 1 of 120 tokens (P4 F1); that is why the prereg has an INERT outcome and gate.
- TOAD shows that only an off-proposal-accurate scorer works. Test P2-s0, not an EPO-tuned scorer.

## 4. Next lever and blocker (Rule Zero)

**The next lever is L1 Step A, the off-proposal admission test.** Its DRAFT pre-registration is `PREREG_L1_SEARCH_ADMISSION.md` (sha256 `a76f62d8043e4cc9d6c73b35cf2884f27027a1e01bcd3deec658380cc58bd801`). It was NOT run: this task forbade experiments, and the dev-box GPU belongs to another session.

It needs:
- adoption of the prereg;
- gate G0, that P2-s0 did not train on the fit sets (UNVERIFIED);
- cached `visual_ctx` for those sets (UNVERIFIED).

**If L1 fails the admission test,** L2 is the committed next arm. If L1 and L2 both fail, the next arm is proposal-side (L5, then L4), which needs an A40 allocation.

## 5. Banking and admissibility

**Newly banked (11):** 2603.29163, 2609.01799, 2610.13781, 2603.13842, 2605.21139, 2608.10403, 2510.08562, 2509.20938, 2609.29813, 2602.05863, 2605.19524.
- Screened only: TPSP, ResAD, S2Planner, SafeAlign-VLA.
- 23 already-banked entries had `cited_by` extended.
- `--verify`: 650 entries, 0 orphans, 4 export artefacts. The 4 are CRLF conversions in the agent's Windows export, not in the git blobs; those files were not staged.

**PUBLISHED-SECONDARY (inadmissible for the registry):**
- the GTRS and ZTRS v1 base rows (TOAD's own pair is primary);
- the ReflectDrive-2 deltas and Curious-VLA's pre-RL row (INHERITED);
- the DriveVer PDMS column (manual);
- SafeAlign-VLA and TPSP (abstract only).

**Limits:**
- These are preprints and single runs.
- DriveSuprim's Tab. 5 check-marks were reconstructed.
- DrivoR's label cadence and DriveSuprim's tuning set are not stated.
- The gap arithmetic is means-level.
