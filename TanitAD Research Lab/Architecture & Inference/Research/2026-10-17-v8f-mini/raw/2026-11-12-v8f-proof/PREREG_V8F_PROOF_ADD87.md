# PREREG_V8F_PROOF_ADD87 (TOKGRID): the token head on the finer 8 x 20 token grid that the caches already hold

Tier T0 (reads on logged frames; no rollout; a diagnostic of what the features carry, never a statement on driving performance). Evidence class of every value it will give: MEASURED. Written while the fits of card ADD86 run and before any value of card ADD86 is read; before any tool or fit of this card exists. No new encode and no camera frame: the clip files of the training cache (4,572 of 4,572) and of the evaluation cache (141 of 141) carry the key `pool8x20` (160 tokens per frame) beside `pool4x10` (40 tokens), read from the two manifests on 2026-11-20.

## 1. What is asked

The cost card on the closing read has two ways to a better read: a finer encode of the scene, or another signal. Before a new encode is put to the PI, the grid that is four times finer and already on disk says what finer tokens buy at T0. Asked: does the token head of card ADD81, on the 8 x 20 grid and with the fit of card ADD86, read the label closing speed better than on the 4 x 10 grid, and how far is it from the requirement of r 0.9 of row 147?

## 2. Arms and fits

* T10G, T25G: the arms T10 and T25 with the tokens of key `pool8x20`; the fit setting of card ADD86 (at most 24 evaluations, patience 4, through `TOKCLOSE_HP_OVERRIDE`), so that the difference to T10L, T25L of card ADD86 is the grid alone; the same clips, labels, optimiser, batch, loss and head form (the frame projection takes 160 x 16 values in place of 40 x 16); seeds 0 and 1. Four fits.
* Planted fit of control t1: T10 on the 8 x 20 grid with closing labels permuted, seed 0 (a new fit: the planted control has to sit on the grid of the arms).
* The four E fits of card ADD81 are taken over as files (they read the frozen frame embedding of the 4 x 10 reader and do not change).
* Tool: `tokgrid/tokclose_fit_g8.py`, written from `tokclose/tokclose_fit.py` 84e82ee9184289c4 by a generator with counted edits: the grid key of the T arms comes from the environment variable `TOKCLOSE_T_GRID` (`pool4x10` when unset), is written into the fit log, and every later load of tokens for a head uses the grid of that head's log. Nothing of the scorer's arithmetic changes.

## 3. Read and names

The scorer of card ADD81 in the generated tool, on the 141 evaluation clips and the scored frames of card ADD78, with `--allow-hp-override`. Per arm X in T10G, T25G: pooled r with its interval, per seed r, mean absolute error, paired difference of the absolute error against the ADD78 cell.

* `TOKGRID-<X>-READABLE` / `-NOT-READABLE`, `TOKGRID-<X>-BETTER-THAN-OUTPUTS` / `-NOT-BETTER`: the rules of card ADD81 section 3 as the scorer writes them.
* Grid against grid, by the reader of this card from the saved tables of the fits: paired difference of the mean absolute error, T25G minus T25L of card ADD86, on the scored frames of H 25, two-level bootstrap (reader seeds, then evaluation episodes), 2,000 draws, percentile 2.5 / 97.5. `TOKGRID-FINER-GRID-READS-MORE` when the upper end is below zero, `TOKGRID-FINER-GRID-READS-LESS` when the lower end is above zero, `TOKGRID-FINER-GRID-NOT-SEPARATED` otherwise. The same pair for H 10 is reported with no name.
* Against the requirement: `TOKGRID-BELOW-THE-REQUIREMENT` when the upper end of the interval of the pooled r is below 0.9 for both arms; `TOKGRID-REACHES-THE-REQUIREMENT` when the lower end of T25G is at or above 0.9; `TOKGRID-REQUIREMENT-NOT-DECIDED` otherwise.
* Reported with no name: stop loss before and at the stop, best step, steps, stop reason per fit; the standard deviation and the lag-1 autocorrelation of the error of T25G per seed.

## 4. Controls (all have to pass before a name or a value is opened)

* g0 (the generated tool is the tool on the 4 x 10 path): the scorer of the generated tool, run on the fit folder of card ADD86, writes the arm values of the result file of card ADD86 exactly (every r, interval, error and paired difference equal as numbers).
* t0, t1 (on the planted fit of this card), t3, t4 as the scorer computes them; t2 in the form of note N1 of card ADD81.
* c0: the twelve files taken over (four E fits, each with head, log and table) are equal in sha256 to the files of card ADD81.
* c1: the logs of the five T fits name the grid `pool8x20`, 160 tokens and the override (24 evaluations, patience 4); the logs of the E fits name no override.
* Card ADD86 has to be read with its controls passed before the pair of section 3 is computed; if card ADD86 has a control that did not pass, the pair has no name.

A control that does not pass: the card is reported with that control failed and has no name for what the control guards; no form of a control is changed inside the card (rule of row 136).

## 5. What follows (committed before the data)

* `FINER-GRID-READS-MORE`: finer tokens carry more of the closing speed; the cost card names the 8 x 20 grid as the first step with no new encode and gives the r reached beside the 0.9 required. If T25G is also BETTER-THAN-OUTPUTS it is tested in the rule with the matched surrogate of card ADD85 beside it, in a card of its own.
* `FINER-GRID-NOT-SEPARATED` or `-READS-LESS`: four times the tokens do not move the read; the cost card says that a finer grid of the same encoder is not the lever and puts the question to the signal (card ADD88) and to a crop of the lead region.
* `REACHES-THE-REQUIREMENT`: a rule test of T25G is the next card.

## 6. Limits known in advance

T0 only; two seeds; one head form and one fit setting, no search; the head on 160 tokens has four times the input of the frame projection with the same 4,080 training clips, so a worse read can come from the fit and not from the tokens; chosen after rows 143 to 147 on the same 141 clips; the label bounds every r from above by an amount this card does not measure.
