# PREREG_SAFEOBJ — ADDENDUM 1 (Master Mind, 2026-10-17 ~13:20 CEST): the outcome rules made mechanical, BEFORE DATA

**Status when written:** R0d has NOT been launched. No arm of this pre-registration has produced a number. `PREREG_SAFEOBJ.md` (sha256 `b404c2fb…`)
is unchanged. This addendum resolves the ambiguities the outcome-reader build found (`outcome/OUTCOME_BUILD.md` §4: every one was listed, none was
resolved by the tool). Every resolution below is chosen by the bar's stated meaning or toward the stricter reading. None was chosen by its effect on an
arm, because no arm exists. No threshold, kill or arm changes.

## R1 — Bar G (A-G1, A-G2, A-G3; build escalation 1)

* **F uses dose.json's own Pareto points at full precision.** §5 defines F as "the linear interpolation of dose.json's Pareto points". The parenthesised
  numbers are their 2-decimal renderings. Source: `raw/2026-10-14-rls1b-attribution/dose.json`, sha256 `735a6d909bc0ef41…`, key `full[alpha].{TTC,PDMS}.mean` (×100):
  α 0.25 (−0.033333, +0.598750) · α 0.375 (−0.266667, +0.707195) · α 0.5 (−0.466667, +0.840554). F is held flat beyond the last vertex.
  The floor of G(t) = max(+0.60, F(t)) stays **+0.60 exactly**: it is a stated constant (52 % of R0's +1.15), not a dose.json number.
* The comparison stays **≥** as written. A measured arm ties the frontier with probability ~0. A tie reads PASS, as the text says.
* Consequence, stated so the reader's selftest is right: the α-0.25 mixture point (+0.598750) sits **0.00125 below** the +0.60 floor, so it reads
  **G FAIL**. The α-0.375 and α-0.5 vertices sit exactly on F and read **G PASS** by the ≥. The IMPL_PLAN selftest ("α 0.25 sits exactly on G's
  boundary") was wrong; it is replaced by the vertex test.
* **"The PDMS lower 95 % bound"** = the lower end of the two-sided 95 % percentile interval of the §4 estimator (paired log-cluster bootstrap,
  10,000 resamples, seed 20260927). It is taken on the **u1000 B − own-A PDMS contrast on RL-val** (3,000 sets / 75 logs). **t** = that read's TTC point delta.
* **This read is not written by the chain** (its second `rl_val_eval` read is against the seed-matched C). It is produced after the arm by ONE
  `rl_val_eval.py` call with `--base A` on the arm's own `scores/scores_update1000_{A,B}.npz` (command: `outcome/OUTCOME_BUILD.md` §6). The output
  records both inputs' sha256. ⛔ The arm's u1000 score dumps must therefore be kept until this read is banked.
* **A-G2 (own A vs C0-A):** kept as written. G judges B − own A against F, which was measured vs C0-A. The own-A vs C0-A difference is reported beside G
  as a disclosure. It is §7's "same run again" floor, whose precedent is R0-A vs C0-A PDMS +0.01 [−0.13, +0.14].

## R2 — The outcome table: a fixed precedence (A-O1, A-O2, A-O3, A-O9)

The rows are read **in this order; the first that matches is the outcome**:

1. **O6 INERT.** λ_TTC = λ_NC = 0.0 at every logged update. It **wins over every other row**: if the constraint never acted, nothing can be attributed
   to it, and the run is an R0 replicate as O6 says (A-O1).
2. **O7 OTHER KILL / g3 ALONE.**
   * Any kill whose recorded rule is not K3 (K2, K4, K6, or any other code) is O7.
   * A kill written by the live monitor (`K-MON`) **applying the K3 rule** counts as **K3**, judged by its recorded reason, not by its code (A-O9).
   * **"g3 alone"** = S fails, and every recorded breach at every check is a g3 breach (no TTC or NC g5 breach at any check); any K3 kill was triggered
     by g3 breaches only (A-O2). This is O7, not O4/O5: the dual constrains TTC and NC, so a g3-only failure says nothing about whether it held them.
3. **Breached** (K3 kill, or S fails with at least one g5 breach): O4 / O5 as written, with the two holes closed (A-O3):
   * exactly one of O4, O5 matches → that row;
   * **both match** (held in-sample, λ > 0 on ≥ 50 %, AND λ reached λ_max) → **O4+O5**. Both rows' first steps are 0-A40 (O4: the train-vs-carve gap
     on banked R0 states with the A2 harness; O5: the CPU unit proof). Both run, and the one-arm decision that both rows already give to the PI takes
     both results;
   * **neither matches** (held in-sample, λ > 0 on < 50 %, never at λ_max) → **O4, flagged SLACK**. O5's two clauses are both false, so "the dual could
     not hold even in-sample" is false. What remains is O4's meaning (held on train, did not transfer), with the dual mostly idle.
4. **Not breached** (S holds, no kill): O1 if G holds; O2 if the PDMS lower bound is > 0 and the point is < G(t); O3 if the lower bound is ≤ 0.
   Given S, these three are exhaustive and exclusive.

## R3 — The quantities inside O4 / O5 / O6 (A-O4, A-O5, A-O6, A-O7, A-O8)

* **In-sample violation (A-O4):** per term k ∈ {TTC, NC}, the **signed** mean over the window of the per-update constraint value g_k that the dual
  step consumes, λ_k ← clip(λ_k + η·g_k, 0, λ_max) (SPEC §1). It is the signed value because the dual integrates the signed value, and a positive
  part would read sampling noise as violation.
  **Held** ⇔ both terms' window means are ≤ 0.001. **Not held** ⇔ either is > 0.001. The tool names the logged column it reads.
* **The window (A-O5):** the 100 logged updates immediately before the first breached check. For a K3 kill, that is the first of the two consecutive
  breaches. If the window overlaps the λ_pg ramp, the overlap is disclosed as a flag; the window does not move.
* **Post-ramp share and λ_max (A-O6, A-O7):** post-ramp = updates with λ_pg = 1, counted from the end of the ramp **through the first breached check**.
  "λ reached λ_max" = λ_TTC or λ_NC ≥ λ_max at any update from 0 through the first breached check. Both clauses diagnose the dual before the breach;
  updates after the breach do not bear on that. Neither is read for a non-breached arm.
* **O6's "every update" (A-O8):** through the last logged update. A killed arm's unlogged tail (fewer than 25 updates) is disclosed.

## R4 — Replication (A-O10), lever claim (A-L1, A-L2), endpoint (A-E1 … A-E4)

* **A-O10:** "R1d O1 while R0d is not" is unreachable under this protocol. Stage 2 runs only if R0d reads O1 (PI, 2026-10-17: *"Yes, conditional
  (Recommended)"*). If it ever arises outside the protocol, it reads NOT_APPLICABLE: no lever claim.
* **A-L1:** **L is decided on TTC only.** The PDMS line is the same contrast, reported as the lever's cost, with the same 2 × F_seed yardstick printed
  beside it. It is not a pass condition. **A-L2:** as written.
* **A-E1 / A-E4:** the pooled contrast = the mean of the two pairs' navtest PDMS B − A contrasts on the same 12,146 tokens. Its interval comes from the §4
  bootstrap, resampling the 136 navtest logs **jointly** for both pairs (one index set per draw, applied to both). "The mean contrast" is this mean.
* **A-E2 (clause 4):** **per pair**. For EACH pair, the TTC and NC upper 95 % bounds are ≥ 0: four bounds. This is the stricter reading of the RR1 clause.
* **A-E3 (seed floor at E):** the larger of §7's RL-val training-seed floor and the navtest seed floor |(R0d B − A) − (R1d B − A)|. That is the
  conservative reading, because E reads navtest.

**What this addendum obliges, before R0d's result is read:**
1. `eval/safeobj_outcome.py` applies R1–R4 mechanically, and still lists any reading this addendum does not settle.
2. The G read of R1 is added to the pod sequence (BUILD.md §4) after each arm.
3. The arms' u1000 dumps are kept until it is banked.
