# DSCALE_PRES_READ_DEFS: definitions of the reader of card PREREG_V8F_PROOF_ADD126 (DSCALE-PRES-T0)

Tier T0: a probe on cached tokens of logged frames; it says what the tokens allow, not what a model does with them. Not driving performance.

## M4 in the banked chain (what is reused)
* File and function: `F:/TanitAD-scratch/v8proof/spatial/leadsup/read/leadsup_read_n1.py`, `eval_read` (last line) calling `auroc_row`; `tokprobe_read.read_arm` stores it as `R["M4"]`, the readers print it as `P[label]["R"]["M4"]["auroc"]`.
* Score: `presence_logit`, the first raw output of the head (`head_out_units`), taken at frame S0 of each window (`window_outputs`).
* Label: `ctx["pres0"]`, the logged lead-present flag of the window at S0 (`P["pres0"]` of the pack).
* Windows: ALL windows of the pack (1,948 on the real pack), with finite label and finite score and an episode index of at least 0. It is NOT the scored subset (429 windows, `ctx["valid"]`) that the gap measures use.
* Episodes and draw: `ctx["ep_all"]`, `ctx["n_all"]`, `ctx["C_all"]` (the chain's `boot_counts(n_all, B, seed 0)`). The gap measures use `ep10`, `n10`, `C10`; when every episode has a scored window the two are the same array. Control p5 records whether `C10` equals `C_all`.
* Point value: `RP.auc_pooled` (rank sum). Interval: `RP.boot_auc` over the draw (episode counts as case weights, ties ignored).

## Weighted AUROC of this tool (`auc_weighted`)
* Sum over pairs (a window with a lead, a window with none) of the product of their episode weights times 1 if the lead window scores higher, 1/2 on a tie, 0 below, divided by the sum of the weight products.
* The weights of a draw are the episode counts of one row of the draw matrix, so the AUROC is computed again on every draw. The point value uses weight 1 everywhere.
* It equals the chain's rank form when no two scores are equal (a test asserts it). With ties it differs from the chain's value, which breaks ties by window order; the number of tied windows per probe is written into control p1.

## Measure A and the beside measures
* A: per seed K, AUROC of `P16E_sK` minus AUROC of `P16C_sK` on the same windows; the interval is the chain's `two_level` (rng 0, seed indices then episodes, shared episode draws) applied to the four point differences and to the four vectors of per-draw differences.
* Beside: the same form for `P16Q_sK` minus `P16C_sK` and `P16U_sK` minus `P16C_sK`.
* Beside, descriptive, per probe: with the threshold 0.5 on the presence probability, which is logit 0 (the stored score is a logit, read in `head_out_units`): the share of windows with a lead scored below it in all, and by the label gap at S0 (`ctx["G15"][:, S0F]`) in the bins 0 to under 20 m, 20 to under 40 m, 40 m and more; the share of windows with no lead scored at or above it. Lead windows with no finite gap label are counted apart (`lead_windows_without_a_finite_gap_label`).

## Names (card section 4)
SCALES: interval of A above 0, at least 3 of 4 seeds above 0, mean at least 0.020. SMALL: the same with a mean below 0.020. WORSE: interval below 0 and at least 3 of 4 seeds below 0. FLAT: none of the three.

## Controls
* p0: `DSCALE_EQP_RESULT.json` is found, its sha256 begins 927b06ff7e4a69e9, it has `input_hashes`, every input hash this tool reads (done files, state records, batch logs, precheck records, train logs; computed by the imported reader's `input_hashes`, without the test output) equals the hash stored under the same path, none is MISSING, and the result holds at most one input this tool does not read (the earlier test output). `--eqp-prefix-map OLD=NEW` renames path prefixes when the earlier read ran under other paths. Nothing else of that file is read or printed.
* p1: for each of the 18 probes this tool's AUROC point equals the chain's `R["M4"]["auroc"]` within 1e-9; all 18 are read. Only the largest absolute difference and pass or fail are reported.
* p2: on the labels, episodes and draw of M4, with synthetic scores (no probe value is touched): a pair against itself gives mean exactly 0 and the interval [0, 0]; a swapped pair gives the negative mean and the mirrored interval; scores replaced by the labels give AUROC 1 on the point and on every draw; a fixed permutation of scores with signal gives an episode interval that includes 0.5.
* p3: at least 50 windows with a lead, 50 with none and 20 episodes, for the labels and for every probe's finite scores; counts printed.
* p4: the imported `tests_clause`: the pytest output file exists, ends with a passing summary, has no failure text, and is older than `--test-before`.
* p5: all 18 probes read; same state ids, order and count (imported `q6_one_evalset`); the windows of M4 come from 48 episodes and the scored windows come from 48 episodes.

## Output
`DSCALE_PRES_CONTROLS.json`, `DSCALE_PRES_RESULT.json`, `DSCALE_PRES_VERDICT.md`; never overwritten (a second call writes `..._2`). Exit 0 all pass, 2 a control failed, 3 refused. In controls mode no value of A, no beside value and no share is computed or printed.
