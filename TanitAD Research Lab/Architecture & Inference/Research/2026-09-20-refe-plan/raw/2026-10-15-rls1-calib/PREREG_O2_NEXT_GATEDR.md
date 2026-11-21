# PRE-REGISTRATION (package-local, exploratory): RL + a cross-fitted TTC gate on the R side, vs the control AS DEPLOYED

> Written 2026-10-15 by the O2(i) agent AFTER `ttcgate_read.json` read FAIL and BEFORE any pick of the read below was computed. Package-local for the same reason as
> `PREREG_O2_NEXT_TTCGATE.md`. **Disclosed post-hoc origin:** `ttcgate_read.json` `R_under_T_minus_rawC0` (reported, deciding nothing) showed R under the gate vs the
> un-gated control at TTC -0.47 / -0.63 (R0 u500 / u1000) and -0.27 / -0.17 (R0b u250 / u500). That observation is NOT admissible. This read makes the gate's strength a
> cross-fitted SELECTION (never chosen on the logs it is scored on), and the only admissible confirmation is a later full-navtest read. RL-val has now been used for three
> lever reads in this package: it is a selection tier, and nothing here is an endpoint.

## Question

Is "R + a TTC-gated selection rule" a better DEPLOYED system than the control as deployed (`navsim_v1_lane`), within g5? This is a deployment comparison, not a claim that
the RL scorer is better: the gate is applied to R only. C0's own gate trade is reported beside it (`ttcgate_read.json`: under T, C0 gains TTC +0.63 to +0.77 and loses
PDMS -0.08 to -0.39), so a reader sees what the control could buy with the same knob.

## Procedure (one grid, one selection rule, fixed before reading)

* System `R_k`: R's raw logits under `navsim_v1_lane x p_TTC^k`, k in **{0, 0.5, 1, 2, 4}**. Control: the control dump under `navsim_v1_lane` (k = 0), unchanged.
* **Cross-fitted choice of k**, per pair: the 5 log folds of section 11 (seed 20261015). For each held-out fold, on the OTHER four folds' sets compute the point deltas of
  `R_k` minus the control for every k; choose the k with the **largest PDMS delta among those whose TTC and NC deltas are both >= -0.25** (half the g5 bound: the value rule
  of section 10.2); if no k qualifies, the k with the largest TTC delta (ties -> smaller k). Apply that k to the held-out fold. A set's own log never chooses its k.
* Pairs and bar exactly as section 11.3: R0-B vs C0-A at u500 and u1000; R0b-B vs C0-A AND vs R0b-A at u250 and u500. B1: TTC and NC point deltas >= -0.5; B2: PDMS 95 %
  lower bound > 0 (paired log-cluster bootstrap, 10,000, seed 20260927). An arm passes iff B1 and B2 hold at both reads (both controls for R0b).
* Reported, deciding nothing: the k chosen per fold; every sub-score delta; the frontier (TTC, EP, PDMS of every k on all sets) of R and of the control under the same gate.
* Control that must read a known value: with the grid restricted to {0}, the cross-fitted picks equal the raw picks exactly.

## Outcomes

* **PASS (R0 or R0b):** names the lever "RL + gated selection" as a NEW system (R0 / R0b stay KILLED: K3 judged the arm under the deployed rule). Next: a section-12
  pre-registration with k frozen by the same rule on ALL 75 logs, then a full-navtest read on the dev box of R + gate vs C0 as deployed (with C0 + the same gate reported),
  and a Master Mind / PI decision, because it changes the deployed selection rule (SPEC_NAVTEST Amendment 5 territory).
* **FAIL:** the zero-A40 levers on the banked dumps are spent. Remaining lever: O2 (ii), a separate safety head (architecture + A40, PI decision).
* H-ESTIM-SEED-1: one training seed per arm; RL-val selection tier; the interval answers another draw of logs only, with the per-fold k held fixed.
