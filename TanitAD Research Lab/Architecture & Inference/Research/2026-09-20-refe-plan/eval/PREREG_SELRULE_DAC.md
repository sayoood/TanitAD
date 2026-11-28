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
