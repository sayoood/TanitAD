# PREREG_V8F_PROOF_ADD121_N2 — note 2 to card ADD121 (REPL10K): definitions of the reader

Written before the two further models exist and before any rollout of the main stage. The card (e87dc5b05f6848b5) and note N1 (ddf3556bf0617825) are unchanged. Every definition below was fixed with only banked runs of the first pair on disk; the full text is `repl/read/REPL_READ_DEFS.md` (4f0a7723cd965858).

## 1. The read of the new pair alone

The engine's grid reader takes the spread run at its seed label 0 as the reference of the windows; with the seed set (2, 3) it stops. The read of the new pair therefore runs with the engine's labels (0, 1) and with the engine's run name of the card's own runs shifted by 2: label 0 is model seed 2, label 1 is model seed 3. The engine's files are unchanged. The reads of the first pair and of four seeds are unshifted. The estimator is the one the card names (two-level bootstrap over the two model seeds and the evaluation episodes, 2,000 draws).

## 2. The strategic grid at the new seeds

The engine's strategic grid needs, at every model seed, a run called `R0` for its own controls s3 (read at model seed 0) and s5 (the compare of first ticks); `R0` enters no pair. The card queues the spread run `I10R0Hb20` at the seeds 2 and 3 and no run called `R0`. For a model seed of 2 or 3 the tick file of `R0` is taken from `I10R0Hb20`; the grid runs once over four seeds and the pairs are read on the new pair.

## 3. Controls

* r0 counts 26 new runs (24 arm runs and the two spread runs): the first line of each names a checkpoint hash that agrees with the record.
* r2 as in note N1, section 3; in addition a new line that names the checkpoint hash of the banked run of model seed m - 2 fails. The share of rule steps with a lead, which the reader of card ADD117 had in its arguments control, is not part of r2 here and is printed only.
* r6: the number of windows with a lead comes from the label windows of the engine and does not depend on the model; the part of r6 that depends on the runs is the done marker and the 141 dump files of each.
* r7 covers the 12 new runs with a gap file (GEc and A90); the reference rows are those of the RR run of the same setting and the same model seed.

## 4. Names

`REPL10K-<X>-REPLICATED`: the interval on the new pair is wholly on the claimed side (an end at 0 is not). `-OPPOSITE`: wholly on the other side. Else `-NOT-SEPARATED-ON-THE-NEW-PAIR`. C-K0 and C-CONV read `not a claim` when the interval recomputed on the first pair is not on the claimed side.

## 5. What was run before this note, all on banked runs

* The identity stage of the queue (note N1, section 2): done 2026-11-22T19:32:31Z, the three runs made again through the wrapper 28bc31f21ecb4ccb equal their banked runs.
* A rehearsal of the reader (`repl/reh/rehearse_repl_read.sh`): a stand-in second pair made of links to the banked runs in swapped order (seed 2 = banked seed 1, seed 3 = banked seed 0), first lines rewritten to the form this queue writes. All eight controls and the names stage ran; the pooled levels of the stand-in pair equal those of the first pair, the levels per seed come in the other order, and the first pair recomputes as banked. The rehearsal outputs are not results of the card; no run of the card existed.

## 6. Limit added

With reader seeds and gap files held, the second pair reads the training variance of the model. The variance of the reader fit and of the gap head is not read by this card.
