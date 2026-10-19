# AMENDMENT 2 to PREREG PICKRULE-AUDIT-NC (Research Lab agent, 2026-10-19): the C-ID3 reach clause is mis-specified; it failed on a property of P2's picks

**Parent:** `PREREG_PICKRULE_AUDIT_NC.md` (sha256 `28947063…53ea7`) + `PREREG_PICKRULE_AUDIT_NC_AMENDMENT1.md` (sha256 `28e349b8…96522`).
**Status at writing:** `prepare` ran (C-JOIN PASS). The first `analyze` attempt stopped at the GATE: C-ID1, C-PDMS, C-COL (x4) and C-ID2 (both seeds) PASS; **C-ID3 FAIL**. As designed, no Q1, Q2 or Q3 value and no decision-rule label was computed or read: `result.json` does not exist. The failed gate report is kept as `raw/results/gate_report_attempt1_C-ID3_FAIL.json`.

## A2.1 What failed, and why it is a data fact, not a code defect

C-ID3 (parent section 8) required, among other things, that "veto at tau = 0.5 changes the pick on at least one set and on fewer than all". On P2's own rows it changed **0 of 15,808** sets (fallback sets also 0). The identity clauses passed (tau = 0 reproduces the unvetoed picks on 100 %; tau = 1.01 falls back on every set and reproduces them; every non-fallback vetoed pick has NC sigmoid >= tau).

The label-free reason, measured after the failure and before amending (`sigmoid(NC logit)` at P2's picks, no label involved): **the minimum over all 15,808 sets is 0.6286**; the number of P2 picks with NC sigmoid below tau is **0 / 0 / 1 / 62** for tau = 0.3 / 0.5 / 0.7 / 0.9. So P2's rule never selects a proposal its NC row puts below 0.63: a veto at tau <= 0.5 CANNOT change P2's pick. The clause presumed P2's picks include some with NC sigmoid < 0.5; they do not. The veto code is not at fault: it reaches on planted and random data in `test_pickrule.py` (20 tests, RED mutants caught), and the identity clauses above passed on the real rows.

## A2.2 What this amendment changes (only the control; no quantity, rule, grid, estimator or next arm)

1. **The registered reach clause is NOT deleted.** It stays in the gate report as an INFORMATIONAL row (`C-ID3-registered-reach-clause`, result FAIL, not gating) and in RESULT.md. It was amended AFTER it failed: it cannot be called a pre-registered pass, and the result must say so.
2. **C-ID3 (gating) keeps the identity clauses**: tau = 0 identity; tau = 1.01 identity with every set in fallback; every non-fallback vetoed pick at tau = 0.5 has NC sigmoid >= 0.5.
3. **New gating clause C-ID3b (reach, data-independent):** on P2's rows with the NC logit column shifted by **-3.0** (a planted, label-free perturbation; no real label enters), the veto at tau = 0.5 changes the pick on at least one set and on fewer than all, and every non-fallback vetoed pick has NC sigmoid >= 0.5. A veto that is a no-op, or a switch, fails it (a RED mutant in the test-suite).
4. The per-tau counts of picks the veto changes on the REAL rows (P2 and the EP-led composite) are reported in Q3's `n_changed_vs_unvetoed` column, as the parent already specified; they are the veto's reach on the real rows and are not a gate.

## A2.3 Consequences stated now

- **Q2 is determined by the gate's number:** P2's picks all have NC sigmoid >= 0.6286, so Q2 (the share of P2's own NC-unsafe picks whose NC sigmoid is < 0.5) is **exactly 0** whenever P2 has at least one NC-unsafe pick. I learned this from the failed gate before amending. Q2 is the parent's reference quantity and has no authority over the decision, which rests on Q1 alone. The matched Q2 and the Q1 - Q2 contrast inherit that and carry no authority either.
- **The veto at tau in {0.3, 0.5} is inert on P2's own rows** (0 changed picks); P2 with veto at those two tau is P2.
- **If the Master Mind rules that the registered clause stands**, the parent's text ("each must hold or the read is INVALID") makes the read INVALID; the numbers are produced regardless and are all in `result.json`, so nothing needs re-running to apply that ruling.
