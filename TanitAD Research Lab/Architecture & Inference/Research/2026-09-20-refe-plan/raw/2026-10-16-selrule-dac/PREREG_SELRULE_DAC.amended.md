# PREREG: a drivable-area/collision-aware SELECTION RULE on P2's own scorer (0 GPU; fit on RL-val, confirm once on navtest)

**Status.** IN FORCE 2026-10-15 ~23:30 Berlin. Master Mind, written **before any number below exists**:
- P2's full-navtest 64-proposal table is still being built (`raw/2026-10-16-p2-full-table/`).
- No RL-val curve for these rules has been computed.

**Amendments.** Instrument facts (paths, n, field names) may be added by a dated amendment **before** the confirm read. Nothing in sections 2–5 may change after any confirm number is seen.

## 1. Why (MEASURED)

**The zero tail is a selection failure.**
- `D-REFE-GAP-1`: zeroed tokens (NC × DAC = 0) are 4.15 of the 6.30 PDMS between u400 and 94.55.
- `D-REFE-ZEROTAIL-2`, MEASURED on P2's own table: of P2's 509 zero tokens, **445 (87.4 %) had a non-zero proposal among the 64**. Oracle-on-SEL ceiling: **+3.54 PDMS**.
- 74 % of the zeros are DAC-only and 25 % NC-only.

**So P2's scorer ranks drivable-area and collision violations too low.** The cheapest test is to ask the scorer's OWN sub-score heads to weigh them harder **at selection**: no new information, no training, no GPU.

**Prior: expect a modest effect.**
- The scorer's DAC ranking AUC on C3 is 0.787 (`D-REFE-S1-READ`).
- Per-head calibration did not rescue TTC on RL-val (`D-REFE-RLS1-O2-CALIB`).
- A label-only re-ranker recovered +1.12 (REVIEW_7 §0.5).

A null result is informative. It moves the zero-tail lever entirely to representation: P6's trunk, then P7.1.

## 2. Rules (selection only; the scorer is FIXED: P2 `model_final`, sha256 `ea31db22…`)
Notation: p̂_X = sigmoid of P2's own head for sub-score X on a proposal. `agg` = the registered `navsim_v1` aggregate of the p̂'s (P2's rule).

| rule | definition | grid (fixed now) |
|---|---|---|
| **B** (control) | argmax `agg` | — |
| **V(τ)**, DAC veto | argmax `agg` over proposals with p̂_DAC ≥ τ; if none qualifies, fall back to B | τ ∈ {0.5, 0.6, 0.7, 0.8, 0.9} |
| **K(k)**, gate sharpening | argmax of `agg` with its DAC and NC gate factors raised to the power k | k ∈ {2, 3, 5, 8} |

**Oracle controls.** They are reported, never selected or deployed. Each is B with one predicted gate replaced by the TRUE binary sub-score of that proposal:
- **O-DAC:** the true DAC replaces p̂_DAC.
- **O-NC:** the true NC replaces p̂_NC.
- **O-both:** both are replaced.

Each oracle is the **selection ceiling of a perfect head for that gate**. It is the number that sizes P7.1's prize, without training anything.

## 3. Fit (RL-val), then ONE confirm (navtest)
**Fit set: RL-val**, the RL-S1 selection tier: 3,000 sets over 75 navtrain-held-out logs.
- The inputs are P2's base-scorer dumps (`scores_*_base.npz`) and the true per-proposal PDMS/sub-score labels (`rl_val_eval`).
- **Choose ONE rule and parameter** from V(τ) ∪ K(k): the one with the highest mean true PDMS on RL-val.
  - **Tie rule:** a tie within 0.05 PDMS goes to the smaller change (lower τ, lower k). V and K tie toward V.
- Report the full RL-val curve. If no candidate beats B on RL-val, the confirm still runs on the RL-val argmax, and the outcome reads as it falls.

**Confirm set:** P2's full navtest table, 12,146 tokens over 136 logs, NAVSIM v1.1 (T1).
- The chosen rule is applied **once**.
- **Disjointness check, run first:** RL-val logs ∩ navtest logs = ∅. A non-empty intersection STOPS the study.

**Estimator.** Paired log-cluster bootstrap over 136 logs, 10,000 resamples, seed 20260927, chosen rule against B on identical tokens.
- It answers "another draw of LOGS" only. P2 is one training seed. The rule is deterministic, so there is no inference variance.

## 4. Committed outcomes

| outcome | condition (navtest confirm) | next lever |
|---|---|---|
| **ADOPT** | the lower bound of ΔPDMS > 0, AND no PDMS sub-score adverse (its paired CI entirely < 0), AND no longitudinal/lateral family component adverse (the P2 R2 GO rule) | deploy as P2's selection rule (0 GPU); carry it into P6's read as a SECOND, separately registered seam beside P6's `navsim_v1` read, never replacing it |
| **NOT PROVEN** | the CI straddles 0 | the scorer's own DAC/NC heads lack the ranking signal. The zero-tail lever is REPRESENTATIONAL (P6's trained trunk, then P7.1), sized by the oracle controls |
| **REFUTED** | the upper bound of ΔPDMS < 0 | re-weighting costs more EP than it saves in zeros. Same lever as NOT PROVEN, more strongly |

**Reported in every outcome** (never gating):
- the zero-token count, DAC = 0 and NC = 0 counts, B → chosen;
- the EP and TTC change, which is the expected cost;
- the oracle controls' ΔPDMS, and the share of the +3.54 SEL ceiling each one recovers;
- per command (left / straight / right), since turning tokens carry 54.8 % of the zeros.

**The four families.**
- LONGITUDINAL (EP, speed, along-track) and LATERAL (DAC, heading, cross-track) are reported with paired CIs.
- TACTICAL: the selection quality as random / actual / best-of-64 on the same tokens.
- STRATEGIC: **PAUSED** (PI 2026-09-19). Never a pass.

## 5. What this does NOT answer
- It does not test a better head. The oracle controls only bound one.
- It answers nothing about P6, which trains its trunk and will be read on its own registered seam.
- RL-val is a navtrain-held-out selection tier, not navtest. That is why the confirm is a single, separate read.


---

## Amendment 1 (2026-10-15 evening, written by the fit agent BEFORE any RL-val pick, PDMS, curve or oracle was computed)

Sections 1-5 above are byte-identical to the original (`eval/PREREG_SELRULE_DAC.md` at origin `159c862`, sha256 `48a6940f717e0e264930d788d4b33b3a71e8c063d2fc5957e17c5e6952b2c9e1`). Only instrument facts and operational definitions are added. What I had looked at when I wrote this: the shapes, dtypes and unique VALUES of the RL-val label columns, the file shas and the code paths named below. No pick, no PDMS at a pick, no curve.

### A1. PREMISE CORRECTION (the one that matters): the RL-val score dumps are NOT P2's scorer
Section 3 says the fit inputs are "P2's base-scorer dumps (`scores_*_base.npz`)". MEASURED from source and from the file:
- The only `*_base.npz` in `Sayood/tanitad-refe-rls1-c0/final/` is `scores_step0_base.npz` (sha256 `6332fcc6175805349dd84766d6e38eac79f3ab194588b541755c1cf6281fef63`, byte-equal to `scores_step0_A.npz` and `scores_step0_B.npz`, and to `r0b/final/scores_step0_base.npz`). `base` = the frozen reference scorer `sx0` of `refe/scorer_finetune.py` (`ARMS = ("base","A","B")`, L141, L847); `step0` = before any RL update. So it IS the untouched scorer of the RL-S1 run, which is how it is identified.
- That scorer is **SFT-4 arm B u400** (`model_sft4_B_u400.pt`, sha256 `8b4dba48...`), NOT P2: `refe/rls1_pod.sh:37` `BASE=...model_sft4_B_u400.pt`; `eval/PREREG_RL_S1.md` row OPEN-BASE "DECIDED: A"; the dump is `scores [3000, 64, 7]` (7 outputs = 6 PDM heads + the teacher-lane head), while P2 has 6 outputs ("No lane head in P2", `PREREG_RL_S1` line 20; P2's own table logits are `[N, 64, 6]`).
- It scores the 64 proposals of the PRE-P2 `model_final.pt` (sha256 `4246a320...`, `rls1_checks.py` PINS; `ckpt_step` 10075 in the dump), not of P2's `model_final.pt` (sha256 `ea31db22...`). `PREREG_RL_S1` line 22: "P2 changed the proposals ... must be re-dumped and re-labelled"; variant B (P2 as base) was not chosen, so **no P2 RL-val dump or label set exists**.

CONSEQUENCE, stated so nobody reads the fit as more than it is. The RL-val FIT selects a rule from the u400 scorer's heads on the pre-P2 fan. Its transfer to P2 on navtest is an ASSUMPTION: that the DAC/NC heads of two scorers of one lineage (both start from `model_final`'s scorer) are calibrated alike enough for a probability threshold tau and a power k to carry over. u400's DAC AUC on RL-val is 0.9045 (`c0/final/sft.jsonl` step0 row); P2's DAC AUC on C3 is 0.787 (`D-REFE-S1-READ`), so the calibrations are known to differ. The confirm is still a clean pre-registered test of the rule on P2; but **a fit on the wrong scorer is a weaker prior for the chosen parameter than the prereg intended**. The decision to run, or to demand a P2 RL-val dump first, is the Master Mind's. The fit proceeds as ordered because it is 0 GPU and its input is the only RL-val dump that exists.

### A2. What "B" and "agg" are on a 7-output scorer
Section 2 defines `agg` as the registered `navsim_v1` aggregate (P2's rule, 6 outputs). The u400 dump has a 7th output `p_lane` that u400's deployed rule multiplies in (`navsim_v1_lane`).
- **Primary (the only one that can be selected):** `agg = NC * DAC * (5 EP + 5 TTC + 2 C) / 12` over columns 0-5, ignoring column 6. That is the rule P2 deploys and the rule the confirm table uses.
- **Secondary (reported, NEVER selected, does not touch the frozen choice):** the same grid with the lane factor kept in `agg` (`navsim_v1_lane`, u400's own deployed rule), to show how much the choice depends on dropping the lane head.
- Identity (hard gate, FIT refuses on failure): primary B equals `rl_val_eval.select(.., "navsim_v1")` on all 3,000 sets AND `rl_val_eval.run` on the same inputs returns the same mean PDMS; secondary B equals `select(.., "navsim_v1_lane")`.

### A3. Oracle gate values
The labels have NC in {0, 0.5, 1} (3 levels), DAC in {0, 1}. "TRUE binary sub-score" is read as "the true sub-score value of that proposal, as the harness scored it": the oracle replaces p-hat_NC by the true NC in {0, 0.5, 1} and p-hat_DAC by the true DAC in {0, 1}. Nothing is binarised (binarising 0.5 would move the ceiling).

### A4. Units and the tie rule, exactly
- PDMS is reported x100 (points). "within 0.05 PDMS" = 0.05 points on that scale.
- Candidates are the 9 grid points V(0.5), V(0.6), V(0.7), V(0.8), V(0.9), K(2), K(3), K(5), K(8). **B is not a candidate.**
- M* = the largest candidate mean PDMS. The tied set is {c : M* - mean(c) <= 0.05}. The chosen rule is the FIRST member of the tied set in the order V(0.5) < V(0.6) < V(0.7) < V(0.8) < V(0.9) < K(2) < K(3) < K(5) < K(8) ("smaller change; V and K tie toward V"). If the tied set is a singleton, that is the argmax.
- The RL-val curve is computed at float64 from `sigmoid(logits)`; a proposal qualifies for V(tau) iff `p_DAC >= tau`; ties inside an argmax resolve to the lowest proposal index (`np.argmax`, = `rl_val_eval.select`).
- K(1) = B and V(0) = B are identities and are tested.

### A5. Estimator for the fit and the confirm
`proxy_eval.boot`'s resampling stream (per-iteration `rng.integers(0, n_logs, n_logs)` from `default_rng(20260927)`, token-mean of the drawn logs' tokens, 10,000 resamples, percentile 2.5 / 97.5), computed by cluster sums; the test pins it to `proxy_eval.boot` itself to 1e-9. Fit: 75 logs. Confirm: 136 logs. It answers "another draw of LOGS" only; the chosen scorer-rule is deterministic; P2 and u400 are one training run each.

### A6. Confirm instrument facts
- Tokens and logs: `raw/p2_r2_instrument/baseline/navtest_full_tokens.json` (12,146 tokens, 136 logs). Command: argmax of `driving_command` in `navtest_inputs.json.gz` (0 left, 1 straight, 2 right) if present; else the per-command split is reported "NOT COMPUTED".
- The table is P2's `table.npz` format (`token, pdms [N,64], sub [N,64,6] in nc,dac,ep,ttc,c,ddc order, logits [N,64,6], proposals, pick, valid`). PDMS deltas are on the harness `pdms` column; sub-score deltas on `sub`.
- Gates before any delta is computed: token set == the 12,146, all `valid`, all finite, B reproduces the table `pick` and the mean PDMS at the table pick is P2's banked 88.82 +- 0.01 (x100); otherwise the confirm STOPS.
- FIT.json is read-only after the fit; `confirm.py` refuses a FIT.json whose sha256 differs from `FIT.json.sha256`.
- **Family leg of ADOPT.** The longitudinal/lateral components come from `families6.py` on the two executed-pick seams (B's pick, the chosen rule's pick, poses = `proposals[n, pick]`, the seam schema of `stage3_full.compute_families`), compared by `proxy_eval.fam_adverse` (unpaired interval non-overlap, conservative). The P2 R2 GO rule gated on the six sub-scores only and REPORTED the families; section 4 here says ADOPT needs neither adverse. The LITERAL reading of section 4 governs: ADOPT needs both legs. If `families6.py` cannot run, the outcome is `UNRESOLVED (family leg not computed)`, never ADOPT. The JSON also states the sub-score-only reading beside it.
- The EP/TTC "cost", the zero / DAC=0 / NC=0 counts, the oracle share of the +3.54 ceiling, the per-command splits and random / actual / best-of-64 are reported, never gating (section 4).

### A7. A cell the committed outcome table does not contain (written before the confirm, with the fit still unrun)
Section 4 has three rows. A navtest confirm with a lower bound of DeltaPDMS > 0 AND an adverse sub-score (or family component) is none of them: it is not ADOPT, not NOT PROVEN (the CI does not straddle 0) and not REFUTED. It is read as written, `NOT ADOPTED (adverse component: <names>)`, with the component named, and the decision on it escalates to the Master Mind. It is never silently mapped to NOT PROVEN or to ADOPT. Edge conventions: "lower bound > 0" and "upper bound < 0" are strict; a bound equal to 0 straddles.
