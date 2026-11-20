# PREREG · SCORER-FT: does a within-token ranking term, trained through the representation, raise C1's tracked PDMS?

Written 2026-11-20 by the SCORER-FT design agent for the Master Mind. Status: **FROZEN 2026-11-20 ~13:40Z by the MM** (sha256 in
LOOP_STATE and the register row), **before any arm, weight-rule run or SCORER-FT number exists** (MEASURED at freeze: the pod GPU runs
the field arm F; no `scorerft_*` output directory exists on the pod). ⛔ Nothing is launched by this document. The PI's go is recorded in
§11 (*"Follow your recommendations, acclerate progress."*, 2026-11-20 ~10:35Z): base C1, N = 1,000, Stage 2 conditional. The design it registers is `DESIGN_SCORER_FT.md` (same folder; line numbers at `e17d040e`). Proposed register key:
**`H-SCORERFT-1`**. `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.

## 1 · Question and hypothesis
**Q.** Does fine-tuning C1 with `L_rank` (pairwise within each 64-proposal set: EP among safe proposals, plus gate pairs on the
scorer's own top-8; `DESIGN` §2.2), with the gradient reaching the trunk (`--scorer-sees-trunk`), raise **tracked PDMS** over an
identical fine-tune without it?

**H-SCORERFT-1 (HYPOTHESIS).** T − C tracked PDMS > 0 on full navtest, replicated across a second training seed and above that seed
pair's own floor, with no NC / DAC / EP / TTC / C sub-score separated worse.

## 2 · Basis (every number: class, artifact, tier, estimator)
All **MEASURED**, T1 (NAVSIM v1.1 ego pseudo-sim, LQR-tracked), full navtest 12,146 tokens / 136 logs, one training seed per arm:
* **C1 92.55 [91.80, 93.26]**, human 94.55, C0 91.38: paired log-cluster bootstrap 10,000 draws, seed 20260927 (`D-REFE-P5C-READ-1`,
  `PK/raw/2026-11-20-p5c-read/read.json`). Verdict GUARD-FAIL (TM+C), not adopted.
* **Regret 6.80 = gate 2.72 (357 tokens) + EP 2.63 + TTC 1.38 + C 0.00 + other 0.06.** Oracle-of-64 99.41 on EVAL logs
  (`PK/raw/2026-11-20-p5c-next/propscore/REGRET_DECOMP.json`; descriptive; computed on all navtest tokens, a **SNOOPED** population).
* Within-token heads: AUC NC 0.929, DAC 0.962, TTC 0.900, C 0.996. **EP Spearman over safe proposals 0.413 [0.389, 0.434] vs path
  length 0.828.** On rescuable zeroed picks the DAC output is above the passing median on 70.2 % of 198 (`HEAD_AUC.json`,
  `EP_GEOM_CHECK.json`; log bootstrap 2,000 draws; SNOOPED).
* No fixed reweighting of the heads helps (`D-REFE-SRULE-1`: Δ −0.17 [−0.32, −0.01]). Learned re-rankers on frozen outputs: NO SIGNAL
  (`D-REFE-SRERANK-1`).
* The heads-only ranking family on P6 is swamped / saturated (`H-SCORER1P6-1` O5, `H-S1Z-P6-1` Z5, `H-S1ZSM-P6-1` SM5,
  `H-REP-P6-1` REP-B, `H-DOSE-P6-1` DOSE-C). Training floor on that rig: F = 0.380, s_p 0.179. ⛔ **Not imported** here (different
  rig).
* **INHERITED** (MM brief, not re-verified): ~51 s/step at the P5-C recipe on the L40.
* **ESTIMATED:** the effect size. No prior result predicts it. Recovering 20-30 % of the ep + gate regret (5.35) would give +1.1 to
  +1.6 points. That is a HYPOTHESIS, not a bar.

## 3 · Arms (same init, same bank, same steps; the only differences are listed)
| arm | profile | init | seed | argv vs C |
|---|---|---|---|---|
| **C** | `refe_scorerft_c` | C1 final (`SCORERFT_BASE_SHA256`, MM to measure and bank) | 0 | C1's argv (`p5c_argv("C1")`) + `--init-from <C1 final>` + `--steps N` + `--rank-w 0.0` (named) + own `--out` |
| **T** | `refe_scorerft_t` | same | 0 | + `--rank-w w*` `--rank-heads ep,gate` `--rank-ep-gap 0.02` `--rank-gate-topk 8` |
| **Cs** | `refe_scorerft_c_s1` | same | 1 | C with `--seed 1` |
| **Ts** | `refe_scorerft_t_s1` | same | 1 | T with `--seed 1` |

* **N = 1,000 steps** (PI go 2026-11-20; 2,000 = C1 parity was the alternative). Cosine to 0 over N, lr 5e-5, batch 4 × accum 64, fresh AdamW.
* Every arm keeps C1's trajectory label table (`3b198314…`, rank0). Bank `train_grow_cmd_3c_p7` (150,603 tuples, content
  `92af8d66…`): **parity, no re-selection**. Navtest is never trained on.
* `w*` comes from the committed weight rule (`DESIGN` §2.5), measured on navtrain at C1's init **before** T launches. **K2 or K2b
  ⇒ T is not launched** and the card returns to the MM. The gate may only veto `w*`, never retune it.
* **Staging:** Stage 1 = C, then T. Stage 2 = Cs, Ts, run **only if P1 and G1 hold** (§5): **conditional** (PI go, the recommended form).
* **Queue (PI go):** right after the field arm F; if F is refuted (its K4), SCORER-FT replaces the field Cs/Fs replicates. On the pod:
  F's training ends -> the weight rule (navtrain, C1's init) -> C -> T. F's navtest read runs on the dev box at the same time and
  feeds nothing in this card (every argv here is fixed above), so the order cannot bias either read.

## 4 · Admissibility before minute 0
1. A launch-gate PASS token per arm (`DESIGN` §3), including G-DVB `rank_*` rows, G-LIVE L1-L5 + (f) + (h), G-SUITE 0 regressions,
   and every deliberate-regression arm R-DROP … R-CTRL turning the gate RED on the dev box.
2. `SCORERFT_BASE_SHA256` registered, and C1's `model_final.pt` banked (HF, sha256 verified). It must not be the only copy.
   MEASURED before freeze: `f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4` (the pod's sha file == the verified HF
   pull `Sayood/tanitad-refe-p5c` == the `s2_c1` read preflight); the pod copy `/workspace/data/refe_runs/p5c_c1/model_final.pt`
   (1,288,515,000 B) is present 2026-11-20 13:16Z.
3. The weight-rule JSON banked, with both per-head pair counts > 0 (G-LIVE L2).
4. After training, per arm: the post-training G-ADAM gate (`p5c_postgate.py`) PASS. Adam state exists on every trainable parameter;
   for T, non-zero state on `score_head` rows NC/DAC/EP is a reach check.

## 5 · Primary metric, estimator, bar
**Metric:** tracked PDMS ×100 (NAVSIM v1.1, LQR-tracked; **tier T1**), full navtest 12,146 tokens.
**Estimator:** paired log-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile. ⚠️ It answers *another draw of LOGS* only.
* Training variance is answered by the seed pair (§8).
* Inference variance is **zero by measurement** on C1 (`PK/raw/2026-11-20-p5c-read/seam_determinism_c1.json`: max |Δlogit| 0.0,
  pick agreement 1.0). The read re-checks it on T (CTRL-DET).

| clause | definition | pass |
|---|---|---|
| **P1** | Δ₀ = T − C | **CI-lo > 0** |
| **G1** | each of NC, DAC, EP, TTC, C (×100), T − C | **none separated adverse** (CI-hi < 0) |
| **P2** | Δ₁ = Ts − Cs, and G1 on Ts − Cs | CI-lo > 0, none adverse |
| **P3** | F = \|Cs − C\| (this rig's own control floor); d = (Δ₀ + Δ₁)/2 | **d > 2F** (s_p reported beside it) |
| **ADOPT** | P1 ∧ G1 ∧ P2 ∧ P3 | as a scorer-training lever on the C1 recipe |

* DDC is reported, not a guard (`navsim_v1` ignores it).
* GOAL: human 94.55. Any arm below it is reported **BELOW GOAL**.
* ⚠️ ADOPT does **not** clear C1's own GUARD-FAIL (TM+C: transparent comfort, the longitudinal speed bias). Those stay open on the
  lineage.

**Read controls** (any FAIL ⇒ the read is INVALID, never a verdict):
* **CTRL-T:** the per-proposal table's pick == the harness CSV row to 1e-9 on every token, every arm (`p5c_prop_read.py`).
* **CTRL-SET:** identical 12,146 tokens across the 4 arms.
* **CTRL-ARGV:** the argv difference from the tokens == §3 exactly.
* **CTRL-DET:** T's seam re-run on 200 tokens is identical.
* **CTRL-W0:** C's step-0 digest == C1.

**Comfort-guard form (open PI item, 2026-11-20).** It enters **no clause** of this card: G1's C is NAVSIM's tracked comfort sub-score
as the harness computes it. C1's own transparent comfort guard (GUARD-FAIL, TM+C) stays open on the lineage. The read reports that
transparent comfort in **both** forms (the 20-pose plan and the 8 submitted poses), descriptively.

## 6 · The four metric families (binding; per family, never pooled; T − C and Ts − Cs, same tokens, same paired estimator)
| family | metrics (instrument) | notes |
|---|---|---|
| **LONGITUDINAL** | speed_bias, speed_mae, along_final_bias, accel_mae, progress_ratio, under_progress_rate, target speed within 0.5 / 1 / 2 m/s, **min_ttc_s**; **headway / time-gap to the lead agent** (`p5c_read` family code, tracked-executed frame) | headway / time-gap / TTC to the lead: the `families` step of `eval/p7_read_chain.py` (seam -> plans8 -> `transp_score.py` -> `families_run.py --per-window` -> `fam_ci.py` paired), verified in source at `b3545f08` (`p7_read_chain.py:19`, `families_run.py:9`: computed where a lead exists with its n, refused with the reason where none does). The S1Z failure mode (slower picks) is read here |
| **LATERAL** | heading error, **curvature error, yaw-rate error**, cross_mae, cross_final_mae | as P5-C |
| **TACTICAL** | (a) manoeuvre decision: turn-active command accuracy + κ (`tactical.json` path); (b) **selection quality from each arm's 64-proposal table** (`p5c_prop_score.py` → `regret_decomp.py`, `head_auc.py`, `ep_geom_check.py`): regret (oracle − pick) and its split gate / EP / TTC, **top-1 rate** (pick PDMS == oracle PDMS within 1e-9), zeroed-pick rate, TTC-0 pick with a TTC-1 alternative count, within-token AUC NC / DAC / TTC, EP Spearman over safe proposals, DAC / NC boundary share; (c) tactical goal-setting | (c) **NOT COMPUTABLE**: REFe has no tactical goal head (n = 0, reason stated) |
| **STRATEGIC** | S_DEC all-token accuracy and the P5-C read's strategic block (`strategic_family_*.json`), **in full** (PI unpaused 2026-11-16) | ⚠️ the P5-C generator still stamps PAUSED (`eval/p5c/p5c_read.py:38`, `p5c_tactical.py:39`, `p5c_families.py:6`, with tests pinning it). A **zero-GPU work item before the read**: the read reports the strategic block in full and never prints PAUSED |

A separated adverse cell in any family is **stated in the verdict line as a qualifier**. Only G1 blocks ADOPT. That was decided
here, before data.

## 7 · Mechanism reads (decide only WHICH next arm, never ADOPT; T − C, log bootstrap 10,000, seed 20260927)
* **M1 EP ranking:** EP within-token Spearman over safe proposals (`head_auc.py:53-56` definition). *Moved* iff Δ ≥ +0.10 **and**
  CI-lo > 0. (+0.10 is a quarter of the measured 0.41 → 0.83 gap to path length.)
* **M2 boundary:** zeroed-pick rate, and the DAC / NC boundary share on rescuable zeroed picks.
* **M3 regret:** oracle − pick, and its gate / EP / TTC split. *Moved* iff the CI-hi < 0.
* **M4 generation guard:** oracle-of-64. *Damaged* iff separated worse (CI-hi < 0).
* **Mechanism moved** := M1 moved ∨ M3 moved.

## 8 · Seed replicate plan (`H-ESTIM-SEED-1`)
The bootstrap is blind to training variance, and this rig's floor is **unmeasured**: C0r was never triggered, and P6's F = 0.380 is
another rig. So:
* **ADOPT requires the seed-1 pair (P2) and the floor clause (P3).** Cs and Ts differ from C and T only by `--seed 1` (data order and
  augmentation draw; same init).
* Inference seed: the planner is a deterministic argmax, and inference variance is measured at 0.0 (§5).
* The field chain's Cs − C (P7 base, 2,000 steps, no table) is **reported as context, never used as F**.
* If the outcome is O2 (§9), the cheapest next experiment is **more seeds, not more windows** (CLAUDE.md).

## 9 · Outcomes (both directions committed now; evaluated in this order; exhaustive and exclusive) and the NEXT arm for each (Rule Zero)
| # | condition | verdict | next arm, committed now (cost ESTIMATED) |
|---|---|---|---|
| **O1** | P1 ∧ G1 ∧ P2 ∧ P3 | **ADOPT** (BELOW GOAL unless ≥ 94.55) | **SCORER-FT-2:** add `ttc` to the heads (the 1.38 term), from T's final, same design, own card (~14.2 L40-h + control). `L_rank` carries into the next generator arm (TEACHER-N7 / comfort) |
| **O2** | P1 ∧ G1, but ¬(P2 ∧ P3) | **SWAMPED, NOT PROVEN** | one more seed pair (T s2, C s2; ~28.3 L40-h): seeds, not windows. Still swamped ⇒ O4's next |
| **O5** | P1 ∧ ¬G1 | **ADVERSE, NOT ADOPTED** | drop the head behind the adverse sub-score: EP adverse ⇒ `gate` only; NC / DAC / TTC adverse ⇒ `ep` only. **C is reused** (same init / seed / steps), so ~14.2 L40-h |
| **O6** | ¬P1 ∧ M4 damaged | **FAIL: the trunk path hurt generation** | `L_rank` on a second scorer pass over **detached** `visual_ctx` (rank stops at the scorer decoder; BCE path unchanged), new flag `--rank-detach-ctx`, own card; C reused, ~14.2 L40-h |
| **O3** | ¬P1 ∧ ¬M4 damaged ∧ mechanism moved | **FAIL: the heads rank, the pick does not** | decision-level term: **pick-anchored pairwise on the `navsim_v1` aggregate** (every j with PDMS_j > PDMS_pick vs the current pick, weight PDMS_j − PDMS_pick), a new head key `pick` in the same `rank_loss.py`; C reused, ~14.2 L40-h |
| **O4** | otherwise (¬P1, mechanism not moved) | **FAIL: the representation path does not carry the margin at this dose** | the selection line closes. Generation **TEACHER-N7** (`PREREG_P5C_NEXT` §5 rank 3), **or** explicit proposal-geometry inputs to the EP head (architecture: PI). Both are **named blockers on the PI and the L40** |

A FAIL at Stage 1 (O3 / O4 / O6) is accepted on one seed. This was decided before data, to spend the replicate only where an effect
could be real. The verdict states that training variance was not measured for that arm.

## 10 · Multiplicity, snooping, ledger
* **One primary test (P1)** plus its replicate (P2); five guards (G1); families and M1-M4 are descriptive with CIs and do not adopt.
* **Snooping disclosure:** the lever, its heads and its sizes were chosen from C1's navtest diagnostics (`HEAD_AUC`, `REGRET_DECOMP`,
  all 12,146 tokens, labelled SNOOPED). The arms train on navtrain only. Constants are fixed in this card without a navtest look;
  `w*` comes from a navtrain-only rule. The navtest read of these arms is a test of trained models, not of a rule fit on navtest.
* **Ledger:** no navtest look ledger exists in the repo (searched at `b3545f08`; control: 2,808 files under PK mention PDMS). This
  card's Stage-1 read (T, C) is **look 1 of `H-SCORERFT-1`**, and the Stage-2 read (Ts, Cs) is **look 2**. Multiplicity across cards
  is not corrected, and the verdict says so.

## 11 · Cost (ESTIMATED) and the three PI decisions
* Stage 1: 2 × 1,000 × 51 s = **28.3 L40-h**. Stage 2: **28.3 L40-h**. Gate / weight-rule smokes: about 0.5 L40-h each.
* Reads run on the dev box (not L40; cost UNVERIFIED, as P5-C's read).
* At 2,000 steps, double everything (≈ 115 L40-h).

**PI decisions, RECORDED (PI, 2026-11-20 ~10:35Z, verbatim: *"Follow your recommendations, acclerate progress."*, on the card that
carried these three recommendations):**
1. **Compute and queue:** Stage 1 (~28.3 L40-h) approved, queued right after F (replacing the field Cs/Fs replicates if F is refuted);
   Stage 2 **conditional** on P1 ∧ G1 (+28.3 L40-h).
2. **Base:** **C1**.
3. **Dose:** **N = 1,000 steps**.
Spend beyond Stage 1 + the conditional Stage 2 (O2's extra seed pair, any next arm in §9) needs a new PI go.

## 12 · Proposed register row (the MM writes it from the tip, in the same turn it freezes this card)
`H-SCORERFT-1, pre-registered` (2026-11-20, MM; `PK/raw/2026-11-20-scorer-ft/PREREG_SCORER_FT.md`, sha256 in LOOP_STATE) | Does `L_rank` (safe-only EP pairs +
top-8 gate pairs, harness-weighted, through the trunk) raise C1's tracked PDMS over a matched fine-tune? | OPEN. Frozen before any
arm. Arms C, T (seed 0) and conditional Cs, Ts (seed 1), N steps from C1. Bar: P1 CI-lo > 0, G1 no adverse sub-score, P2 replicate,
P3 d > 2F. Outcomes O1-O6, each with its next arm. Source finding (step 1): **no ranking term in `train.py`; EP is unconditional
pairwise-vs-PDM-Closed EP; the online labeller, not `onpolicy_label_v4.py`, labels P5-C** (`DESIGN_SCORER_FT.md` §1).
