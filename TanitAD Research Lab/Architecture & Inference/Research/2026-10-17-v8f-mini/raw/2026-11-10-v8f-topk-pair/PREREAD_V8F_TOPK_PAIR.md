# PREREAD_V8F_TOPK_PAIR: is the best-of-3 headroom on the tactical band reachable by choosing the manoeuvre? (Master Mind, 2026-11-10, written before any number of this read exists)

Status: rules fixed before the run. Zero training, CPU. Instrument `theory/topk/topk_pair_read.py`, a copy of the OTBAND read's `theory/otband/read/topk_read.py` loop with one added decode; the sha256 of this card and of the script go in `PREREAD_V8F_TOPK_PAIR.sha256` before the run.

## Why
* **The trigger fired (OTBAND, MEASURED, one-shot plan at the first tick of the T1-kin windows, where the self-action history equals the logged one):** best-of-3 minus top-1 on tactical ADE (2.2-6.0 s) is -0.35 to -0.39 m in all five OB runs, above the 0.29 m replicate floor; the logged (lat, lon) pair is among T's top 3 in 90 % of windows against 57-59 % at top 1 (`raw/2026-11-09-v8f-otband/read/out/TOPK_FULL.json`).
* **Two facts say the headroom may not be a manoeuvre choice.** (1) In the same JSON, the plan closest to the logged one is T's rank 1, 2 or 3 about equally often (OB-CTL s0: 0.31 / 0.32 / 0.36). (2) H-V8F-TAC-2 (MEASURED, T0, M2b, 2026-10-21): with T's own constraints kept, the GT decisions alone carry 0.07-0.12 of the full tactical-code oracle (-0.08 to -0.14 m at 6 s, below 2F), and the continuous constraints carry 0.88-0.90. A minimum over three different plans is lower than any one of them even when no plan is systematically better.
* A trained selection head over T's top K is the next card only if choosing the right manoeuvre is what buys the headroom. This read measures that on the adopted base, before any head is designed.

## Arms (checkpoints as banked; no training)
* `PARITY_DAGOACC_s0`, `PARITY_DAGOACC_s1`: the adopted base, DAG-OACC (the question's object).
* `OB-CTL_s0`: method control only; its top-1 and best-of-3 tactical ADE must reproduce `TOPK_FULL.json` to 1e-9 m.

## Plans read per window (eval-141, 2,399 windows per arm, first tick; same batch build as `topk_read.py`)
* **TOP1:** `forward_infer`'s plan on T's own code (the plan a T1-kin rollout executes at that tick).
* **BEST3:** per-window minimum over the K = 3 plans of `forward_topk` (the oracle of the OTBAND read).
* **PAIR-SEL:** the K-plan whose (lat, lon) pair equals the logged pair when the logged pair is among the three; TOP1 otherwise. A perfect selector over T's top 3.
* **GT-PAIR:** O decoded (`forward_diag`) on T's own code with the lat and lon parts replaced by the one-hot of the logged pair. A perfect manoeuvre classifier with T's own goals and constraints.
* Label: `lab_t` of the window's T tick, `lat_cls` and `lon_cls` both labelled (the classes T's heads train on and `topk_codes` ranks). Every comparison runs on the labelled windows only; n is reported.

## Rows and estimator
Per band (operative knots 1-10, tactical 11-30, full 1-30): ADE of each plan; paired differences PAIR-SEL - TOP1, GT-PAIR - TOP1, BEST3 - TOP1 with `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0 (answers "another draw of episodes" only; no training run is involved, and the planner is deterministic). Beside: the captured share S = (PAIR-SEL - TOP1) / (BEST3 - TOP1), a ratio of means (ESTIMATED from MEASURED means, no interval). The four families do not apply to an oracle bound: ADE is the decision row, and FDE plus the longitudinal and lateral rows of `read_lib.band_rows` are reported beside it for PAIR-SEL and GT-PAIR. STRATEGIC: PAUSED (PI item 25).

## Committed outcomes (tactical band, DAG-OACC, per seed; both seeds must agree, else SEEDS-DISAGREE)
* **SELECTABLE:** PAIR-SEL - TOP1 separated on the better side and S >= 0.5. Trigger: the top-K card trains a manoeuvre selector over T's top K. Its bar is half of the measured PAIR-SEL gain, above a same-recipe replicate floor.
* **DIVERSITY:** PAIR-SEL - TOP1 not separated, or S < 0.25. Trigger: no manoeuvre selector is built. The best-of-3 gap is recorded as plan diversity, and the next tactical lever is the continuous constraints. The card reads T's constraint error on the adopted base: the H-V8F-TAC-2/-3 line, re-measured on DAG-OACC at T1-kin.
* **MIXED:** separated and 0.25 <= S < 0.5. Trigger: the top-K card carries a selector arm and a constraint arm side by side.
* **Beside, no bar:** GT-PAIR - TOP1 gives the ceiling of a perfect manoeuvre classifier on this base. If it is not separated, O's tactical plan barely responds to the manoeuvre code, and that is stated as the reason for any outcome.

## Controls (the read is INVALID if one fails)
1. Each checkpoint's sha256 equals its bank manifest, before the read.
2. TOP1 equals the arm's T1-kin dump `ol` at tick 0 (max |diff| <= 1e-4 m, as `topk_read.py`), and the window set equals the dump's.
3. OB-CTL_s0's TOP1 and BEST3 tactical ADE means equal `TOPK_FULL.json` to 1e-9 m.
4. PAIR-SEL equals TOP1 exactly on every window where the logged pair is not among the three, and equals one of the three K-plans where it is. GT-PAIR equals the K-plan of that pair, to 1e-5 m, on every window where the logged pair is among the three.
5. Memory: the read waits for 10 GB of free physical memory before each arm and 7.5 GB before each chunk, so the planner gates' 6 GB guard keeps its margin.

## Log
* 2026-11-10 written before the run.
