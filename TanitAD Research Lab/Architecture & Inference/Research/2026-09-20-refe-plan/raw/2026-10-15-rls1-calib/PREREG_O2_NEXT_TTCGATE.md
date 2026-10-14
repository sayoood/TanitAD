# PRE-REGISTRATION (package-local, exploratory): the selection-time TTC gate, the next zero-A40 lever after section 11's FAIL

> Written 2026-10-15 by the O2(i) agent AFTER `calib_read.json` read FAIL and BEFORE any pick under the rule below was computed. It lives in the package, not in
> `eval/PREREG_RL_S1.md`, because this agent's brief allowed only section 11 there; the Master Mind may transcribe it as section 12. Its sha256 is recorded in
> `ttcgate_read.json` (`prereg_sha256`), which the script refuses to run without. **Exploratory: a PASS here names a lever, it does not adopt one**, because it changes the
> deployed selection rule (SPEC_NAVTEST Amendment 5 territory: Master Mind / PI).

## Why this lever (RULE ZERO item 4: the cheapest experiment that could still work)

Section 11 failed for a measured reason: the TTC / NC heads are ALREADY calibrated on RL-val (held-out ECE 0.004-0.026, Platt slope 0.90-1.02, `calib_read.json` D1 / D4), so a
monotone map has nothing to correct, and the TTC loss is a TRADE defect: the rule lets EP outbid TTC (5/12 each, additive). The reward-side analogue of a harder TTC gate is
candidate (c1) of `raw/2026-10-14-rls1b-attribution/` (TTC / NC hard multiplier); this is the same idea at SELECTION time, zero training, minutes of CPU.

## The rule (one rule, no grid)

`T = navsim_v1_lane x p_TTC` = `NC * DAC * TTC * (5 EP + 5 TTC + 2 C) / 12 * LANE` over the sigmoid of the raw logits (no calibration). TTC enters the multiplicative gate as NC
and DAC already do. Ties -> lowest index.

## Bar, reads, outcomes (section 11.3 unchanged, only the rule differs, applied identically to every dump)

* Pairs: R0-B vs C0-A at u500 and u1000; R0b-B vs C0-A AND vs R0b-A at u250 and u500; both sides under rule T.
* B1: TTC and NC point deltas (x100) >= -0.5. B2: PDMS 95 % lower bound > 0. Paired log-cluster bootstrap, 10,000, seed 20260927. An arm passes iff B1 and B2 hold at both reads.
* Reported, deciding nothing: T's own effect on each control (C0 under T minus C0 under `navsim_v1_lane`, every sub-score); R under T minus C0 under `navsim_v1_lane`.
* Control: the rule `navsim_v1_lane x p_TTC^0` must reproduce the raw picks exactly (0 changed); the T rule must change picks on > 0 sets of every dump.
* **PASS (R0 or R0b):** names a lever, nothing more: a section-12 pre-registration of "RL + TTC-gated selection" as a NEW pair (R0 / R0b stay KILLED), with a decision by the
  Master Mind / PI on changing the deployed rule, then the full-navtest read of both arms under T. Also reported: what T costs C0 itself (if T lowers C0's PDMS, the rule is a trade, not a fix).
* **FAIL:** every zero-A40 lever on the banked dumps is spent (reward side: no unsafe preference, 10.1; dose: refuted by R0b; calibration: section 11; selection gate: this).
  The remaining lever is O2 (ii), a separate safety head: architecture + A40, PI decision.
* H-ESTIM-SEED-1: one training seed per arm, RL-val selection tier, the interval answers another draw of logs only.
