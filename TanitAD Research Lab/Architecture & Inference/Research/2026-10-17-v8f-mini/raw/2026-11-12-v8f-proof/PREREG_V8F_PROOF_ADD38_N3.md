# PREREG_V8F_PROOF_ADD38_N3 : LATVIS-2, a second token-level read at equal calibration (is the gain over the copy specific to the window?)

Note to card `PREREG_V8F_PROOF_ADD38.md` with notes N1 and N2. Written after the first token-level read was seen (both arms `TOKENS-IMAGINED` under k5 in parts: D on the moving cells +4.82 for G and +4.57 for RR, about half of the headroom) and frozen before the read defined here has been run on any arm. It adds a read with its own names; it changes no name, bar or control of the card. Tier T0. LONGITUDINAL, LATERAL, TACTICAL: not measured. STRATEGIC: PAUSED (PI item 25).

## 1. Why

D compares two states read through one map fitted for "state of a frame to tokens of the same frame". A lower squared error than the copy can have two sources: the imagined state carries the change of this window, or it is a smoother state nearer to the average future (any forecast that moves toward the average gains squared error against a sharp copy). D does not separate the two. The read below gives every state the same chance: each is the input of its own linear forecast of the true future tokens, fitted out of fold.

## 2. The read

Arms G and RR, seeds 0 and 1, the same 1,948 windows, folds, token targets (64 principal components per cell, fitted on the training-fold frames) and rollout as the first read. Target of every forecast: the true token coordinates of frame t + 2.0 s. For each held-out fold a ridge map is fitted on the windows of the four other folds from each of five inputs:
* P: the present state (frame t);
* PA: the present state and the 10 logged ego actions of the 2.0 s (20 numbers);
* I: the imagined state at 2.0 s;
* PAI: the present state, the actions and the imagined state;
* T: the true future state (ceiling, known positive).
Penalty: per input, the value of {0.01, 0.1, 1, 10, 100} times the mean diagonal of X'X with the lowest squared error in a cross-validation over the four training folds (the held-out fold is not used for it). Error E of an input: squared error per window and cell of its forecast on the held-out fold.

Statistics (per window the mean over the moving cells of the first read, named; over all cells, reported): C1 = E(P) minus E(I); C2 = E(PA) minus E(PAI). Bootstrap as the card (episode-cluster per seed; two-level for the pooled value; B 2,000; seed 0). Beside: the share of E(P) minus E(T) that C1 reaches, C1 per grid row, the chosen penalties.

## 3. Names (each needs both seeds of the arm; "above 0" as note N1: each seed above 0 and the lower bound of the two-level interval above 0)

* `TOKENS-FORECAST-FROM-IMAGINATION` if C1 on the moving cells is above 0; else `TOKENS-FORECAST-FLAT`.
* `ROLLOUT-BEYOND-LINEAR` if C2 on the moving cells is above 0; else `ROLLOUT-WITHIN-LINEAR`.
* For the pair: `RR-FORECASTS-TOKENS-BETTER` if C1 of RR minus C1 of G, paired on the same windows, is above 0 at both seeds with the paired two-level interval above 0; else nothing is named.

## 4. Controls (all must pass for an arm; else none of its names of this note is given)

* f1 (known zero): with the present state given as the imagined state, C1 is exactly 0.
* f2 (known positive): E(P) minus E(T) on the moving cells is above 0 at both seeds with the two-level interval above 0.
* f3: with the imagined states permuted across windows, the held-out skill of the forecast from I is at most 0.02 in every fold.
* f4: for every input and fold the skill on the training windows is above the held-out skill, and the held-out skill of P is above 0.
* f5: k6 of the card, and k5 in parts of note N2 on the new dump (the same rollout: the dump's k5 value equals the probe's to 1e-6).
* In the tests: a world where the imagined state is only the present state moved toward the average reads `TOKENS-FORECAST-FLAT`; a world where it carries the future reads `TOKENS-FORECAST-FROM-IMAGINATION`.

## 5. What follows (committed before the data)

* C1 and C2 both above 0: the imagined state holds window-specific information about the future tokens that a linear map of the present state and the ego actions does not give. The token level is the read for layout transport, as section 7 of the card says.
* C1 above 0, C2 not: the imagined state forecasts the future tokens better than the present state, and a linear map of the present state and the ego actions does as well. Reported as: at 4x10 tokens the rollout transports what the ego's own motion implies, linearly available. The lever for more is the object level (the lead reads of cards ADD36 and ADD37) and a finer token grid (a decision of the PI).
* C1 not above 0: `TOKENS-IMAGINED` is reported together with this, as a gain against the copy that is not specific to the window at equal calibration; no claim of layout transport is made from it.

## 6. Limits

Linear forecasts on about 1,560 training windows per fold for 2,560 targets; the penalty is chosen on the training folds. The input I is a function of the present state and the actions, so C1 includes what the actions tell; C2 is the stricter read. Two seeds per arm, one corpus, 139 episodes.
