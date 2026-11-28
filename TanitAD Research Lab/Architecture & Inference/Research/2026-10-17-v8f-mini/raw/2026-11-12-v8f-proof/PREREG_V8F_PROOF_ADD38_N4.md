# PREREG_V8F_PROOF_ADD38_N4 : LATVIS-2, the token-level read at equal calibration on two new seeds (2, 3) of G and RR

Note to card `PREREG_V8F_PROOF_ADD38.md` and its notes N1, N2, N3. Frozen before any token dump of G_s2, G_s3, RR_s2 or RR_s3 exists and before the k5 probe of these arms is run. The card, the notes and the names of row H-V8F-PROOF-70 stay as written. Tier of every read: T0 (a world-model diagnostic on logged frames, never driving performance). STRATEGIC: PAUSED (PI item 25).

## 1. Why (MEASURED today)

* Row H-V8F-PROOF-70 named, on seeds 0 and 1: both arms `TOKENS-FORECAST-FROM-IMAGINATION` and `ROLLOUT-BEYOND-LINEAR`, and the pair `RR-FORECASTS-TOKENS-BETTER` (C1 of RR minus C1 of G +0.44 [+0.34, +0.55]).
* The four-seed lead read of RR (notes ADD18_N2 and N3, `spatial/q2/pool4/read_rr4`) reads `MOTION-ADDS-NOTHING-4`: what the rollout of RR added to the present state on seeds 0 and 1 (+0.046 and +0.040) is +0.006 and +0.010 on the new seeds 2 and 3, and the interval of the new seeds includes 0. Seeds 0 and 1 are therefore not a safe base for a claim about RR, and a separated interval over episodes at two seeds is necessary, not sufficient.
* The checkpoints, the pod state files and the banked rollouts (context 3 from the state of frame t) of G_s2, G_s3, RR_s2 and RR_s3 exist. No token of these arms has been read.

## 2. What is run

The tool of row H-V8F-PROOF-70 unchanged (`latvis2/lv2_pod_tokens.py`, sha256 f863d50e061d69016107e23e5d6b2ba8b34e061e435b273819a9df45f3c43a8c on the pod), with `--forecast`, on the pod CPU for G_s2, G_s3, RR_s2, RR_s3: same windows (1,948, 139 episodes), same folds, same penalty grid, same definition of C1 and C2 on the moving cells. The k5 probe of note N2 unchanged for each new arm. Readers unchanged (`lv2_forecast_read.py`, `lv2_read.py`).

Two scopes per arm, each read by the unchanged reader: the new seeds alone (2, 3), and four seeds (0, 1, 2, 3; seeds 0 and 1 are known and declared as such).

## 3. Names (added by this note; the new seeds carry the confirmation)

"The reader gives a name in a scope" means what note N3 fixed: every seed of the scope above 0 and the lower bound of the two-level interval above 0, all controls of the scope passing.

* `TOKENS-FORECAST-FROM-IMAGINATION-4` for an arm: the reader gives `TOKENS-FORECAST-FROM-IMAGINATION` in the scope of the new seeds alone and in the scope of four seeds. Else `TOKENS-FORECAST-FLAT-4`.
* `ROLLOUT-BEYOND-LINEAR-4` for an arm: the same for `ROLLOUT-BEYOND-LINEAR`. Else `ROLLOUT-WITHIN-LINEAR-4`.
* `RR-FORECASTS-TOKENS-BETTER-4`: the pair reader gives `RR-FORECASTS-TOKENS-BETTER` in both scopes. Else nothing is named for the pair, and it is stated that the pair name of row H-V8F-PROOF-70 did not hold on the new seeds.
* Reported beside, naming nothing: C1 and C2 per seed, the share of the ceiling, all cells and the four rows, the held-out skills of the forecasts, and the first read (D against the copy) in both scopes.

## 4. Controls (all must pass in a scope; else no name of that scope)

* f1 to f5 of note N3 on the new dumps, with k5 in parts (k5a to k5e of note N2) from the probe run on each new arm.
* g1: the sha256 of the tool on the pod equals the one of row H-V8F-PROOF-70 (printed by the pod script).
* g2: in the four-seed scope the per-seed values of seeds 0 and 1 equal those of `latvis2/read_n3` (largest absolute difference stated; the old dumps are read, not made again).

## 5. What follows (committed before the data)

* `TOKENS-FORECAST-FROM-IMAGINATION-4` for RR with `RR-FORECASTS-TOKENS-BETTER-4`: the rollout of RR transports scene layout at the token grid beyond the present state on four seeds, while its lead read by a linear probe adds nothing; the claim for RR is limited to that, and the lead line goes on with the supervised head on rolled-out states (cards ADD36, ADD37).
* The name for RR without the pair name: both arms transport a small part; the residual form with the rollout loss is not shown to be the better form at the token level.
* `TOKENS-FORECAST-FLAT-4` for RR: the token-level result of row H-V8F-PROOF-70 holds for seeds 0 and 1 only and is not quoted as a property of the arm; the lever of card ADD18 for a flat motion read stays (explicit object-motion target, second seed of PF).
* In every case the first read (D) is not quoted as transport (note N3).

## 6. Limits

The statistic and the reader were fixed on seeds 0 and 1; the confirmation rests on seeds 2 and 3, and the four-seed interval includes the two seeds that chose the read. One training run per seed; linear maps and forecasts; a pooled 4x10 grid; one corpus.
