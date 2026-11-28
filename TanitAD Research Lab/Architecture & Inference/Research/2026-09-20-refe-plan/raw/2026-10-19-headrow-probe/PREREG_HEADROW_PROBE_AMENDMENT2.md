# Amendment 2 to PREREG_HEADROW_PROBE: C2x, a POST-HOC, UNREGISTERED, DESCRIPTIVE control (it decides nothing)

**Parent:** `PREREG_HEADROW_PROBE.md` sha256 `1800705ef49333dffe0162ea5e51eb72fa1fe8fba57a0ea2f2869bf156762c55`; Amendment 1 sha256 `f98e24deafdefe3501faecaa09a4fa328e2d00ef8658458678e09f6a034fd7f3`.

**Written 2026-10-19 ~09:29, AFTER the outcome tables were read (first read of any outcome: ~09:22 for the per-row call line of a premature `metrics` run, ~09:27 for the full tables), and after the C2x fits were launched but BEFORE any C2x number exists (the first cell needs several minutes).** This amendment is therefore NOT a pre-registration of a result: it adds a control whose reading rule is committed before its own numbers exist, and whose result may NOT change, rescue or void any per-row call, the outcome row or the next arm. Sections 2 to 7 of the parent are unchanged.

## Why it exists

The registered C2 (cold-start probes trained on labels permuted WITHIN each set, scored OOF against the real labels) read **RED in all 8 (row, arm) cells**, with (concordance - 0.5) between +0.00 and +0.32, against a measured per-cell false-RED rate of 13 to 17 % under its null (family rate 0.72). Eight of eight is not chance. Two readings exist:
* (a) the within-set permutation is not a null for THIS data: it leaves each set's label mean intact and learnable, and P2's features (in-sample for P2's trunk) encode each proposal's own label, so a probe that predicts the set mean from proposal features also ranks proposals by their real labels. The pipeline is clean.
* (b) the pipeline (folds, scoring, bootstrap, features-to-labels alignment) manufactures concordance from labels that carry no information.
The registered consequence of a RED C2 is only "CAPACITY void", and no row is called CAPACITY, so no call depends on it. But reading (b) would put every number of this probe in doubt, so it must be told apart from (a).

## C2x (exact)

* Same fits as C2 (cold start: skip = 0, no stored logit, M's hidden branch as in M with W2 = b2 = 0; Adam lr 1e-3, weight decay 1e-4, 20 epochs, 256 sets per batch, 5 folds grouped by log, training seed 0), but the training labels are the labels of a RANDOM PERMUTATION OF THE SETS: `y_train[i] = y[pi(i)]` with `pi = numpy default_rng(20261019 + 77 + col).permutation(15808)` (global, across sets and logs), the valid mask kept as the set's own. This destroys the feature-to-label relation at BOTH levels (within-set and between-set). Scoring: OOF logit against the REAL labels and the real valid mask, all-64 concordance, the same estimator.
* 1 replicate per (row, arm): 8 fits-of-5-folds. Statistic per cell: concordance - 0.5 with the 95 % paired log-cluster interval; RED iff its lower bound is > 0 (the C2 rule). Also M - L (paired) per row.
* **Reading rule, committed now:** reading (a) is SUPPORTED iff at most 1 of the 8 cells is RED AND every cell's point estimate of (concordance - 0.5) lies within +-0.01. Expected number of chance REDs with a per-replicate false-RED rate of about 3 % is 0.24. Otherwise reading (a) is NOT supported, and the registered C2 REDs must be read as a possible pipeline artefact that all FEATURE-LIMITED calls inherit as a caveat.
* Nothing here is used to change a call.

Files: `code/c2x.py` (new), outputs `raw/f10/c2x.json` and the per-cell tables on disk.
