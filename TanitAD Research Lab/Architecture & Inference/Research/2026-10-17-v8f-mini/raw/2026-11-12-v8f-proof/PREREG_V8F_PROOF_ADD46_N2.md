# Note N2 to card PREREG_V8F_PROOF_ADD46 (INTEG10K-PRE): the train-log part of control k8 failed in the form the tool implements; the measured cause; how the card goes on

Note of the v8F proof study, written by the coordinator fork. **It is written after the train logs of the two reader arms were seen**, and after a replay of their batch order. It is frozen before any reader row, quantile map, reader control or rollout of the card exists; no outcome difference of the card exists at this time (time of the freeze: the line of this note in `TIMES_NOTE.md`; hash in the sidecar file). Tier as in the card.

## 1. What happened (MEASURED; files in `integ10k/pull/small/`)

* The reader arms `LR16_s0` and `LR16_s1` (tool names `TK16_s0`, `TK16_s1`) ran to 19,164 steps in 2,475 s. Lead term, mean of the interval means of the first 1,000 steps against the last 1,000 steps: 0.4152 to 0.1950 (`LR16_s0`) and 0.4053 to 0.2033 (`LR16_s1`).
* The train-log check of the tool (the code of card ADD43) has three parts: the lead term falls; the log is complete; the smallest count of gap-labelled frames over every training step is above 0 (read through the minimum of each log interval). The first two hold on both arms. The third reads 0 on both arms: 26 of the 385 log intervals hold a training step with no gap-labelled frame. The tool returned a non-zero code and the queue stopped (`STOP TRAIN-RC`, 2026-11-19T12:34:32Z) before its second step.
* **In the form the tool implements, the train-log part of k8 FAILED on both reader arms.** The row of the card reports this as it is.

## 2. The cause (MEASURED; no model involved; tools `integ10k/diag_tail.py`, `diag_tail2.py`, `diag_tail3.py`, result on the pod in `integ10k/diag/out/DIAG_TAIL3.json`)

The trainer's sampler takes 48 clips at a time, shuffles their windows and cuts them into batches of 64 rows; the last batch of such a pool holds what is left over. The replay builds the data and the sampler exactly as the reader training did and counts the gap-labelled frames with the mask of the lead term (items built as the trainer builds them).

* The replay is the trained order: the hash over the first 200 batches equals the one the training wrote (17ff3aa249537f89).
* The exact count is the trainer's count: it equals the logged count at the first eight log records (257, 266, 202, 192, 165, 268, 190, 200), and the exact minimum of an interval equals the logged minimum on five intervals whose minimum is above 0 (140, 161, 169, 132, 84).
* All 1,300 steps of the 26 flagged intervals were counted exactly. 26 steps have no gap-labelled frame, one in each flagged interval. Every one of them is a last batch of a pool with 1 to 4 rows (14 with one row, 5 with two, 3 with three, 4 with four; 49 rows in all). The 1,274 full batches of these intervals hold at least 77 gap-labelled frames each. Outside the flagged intervals the log's own interval minimum is above 0.
* 26 of 19,164 steps is 0.136 %; the 49 rows are 0.004 % of the window rows drawn in the training. The reader training holds 355 batches with fewer than 64 rows.
* On such a step the gap part of the lead term is 0 by its definition (a mean over an empty mask, guarded), the presence part acts as on any step, and no non-finite value occurred (the tool stops on one).
* The arms of card ADD43 passed the same part by a small margin of the same mechanism: their smallest count over all training steps was 6 on all 24 arms (pool size 41, no held-out clips; `finegrid_r/out_pod/train_A_TRAIN_DONE.json`, `train_B_TRAIN_DONE.json`). The held-out split of this card changes which windows fall into the last batch of a pool.

## 3. The wording of the cards against the tool

Card ADD27, to which card ADD43 and through it control k8 refer, writes: "the count of gap-labelled frames above 0 at every logged step". At the 385 log records the count is at least 59 on both arms, so in this wording the part holds. The tool reads every training step, which is stricter than the wording. Both readings are printed in the row of the card; neither is dropped.

## 4. Decision of the coordinator (under the PI's delegation of 2026-11-19; the PI can overrule it)

* The card goes on with the two trained reader arms. There is no retraining: a sampler without short last batches would be another training recipe than "the token probe of card ADD43" that the card names, and the steps concerned hold 0.004 % of the window rows.
* For the names of the card, the train-log part of k8 is read as k8-N2 (tool `integ10k/check_k8_n2.py`, output `K8_N2.json`); all of the following must hold:
  * a. the lead term of the last 1,000 steps below that of the first 1,000 steps, and the log complete at 19,164 steps, on both arms (as in the tool);
  * b. gap-labelled frames above 0 at every log record (the wording of card ADD27);
  * c. the replay is the trained batch order (hash of the first 200 batches), and its exact count reproduces the logged interval minimum on five intervals whose minimum is above 0;
  * d. the log intervals with a zero minimum are the same in both arms and in the replay, every step of them is counted exactly, each holds a step with an exact count of 0, and every step with an exact count of 0 is a batch with fewer than 64 rows and at most 8 rows;
  * e. the steps with no gap-labelled frame are at most 1 % of the training steps.
* **Parts d and e are written knowing their measured values (26 steps, 1 to 4 rows, 0.136 %). They state what was accepted. They are not blind controls, and the bounds of 8 rows and 1 % were chosen with the values in view.** The blind checks of the reader stay k1 (gap error and presence on the reference windows), k6, k7, k9 and k10 of the card; none of them has run.
* The counts part of k8 (2,399 windows and the same lead windows in every dump) is unchanged.
* Reporting: beside any name of the card the row states "train-log check of the tool (every training step): FAIL on both reader arms; wording of card ADD27 (every log record): holds; k8 read by note N2". If the PI overrules this note, the card reads NOT NAMED (k8), its outcome differences stay withheld, and the next arm is a reader retrained under a card of its own.

## 5. Mechanics

* Continuation queue `integ10k/pod_integ10k_queue_n2.sh`, written from the first queue by `integ10k/make_n2.py` (asserted text edits; sha256 of the queue 110a19218d3c4880, of the maker 4bf204ddeab8a492). It trains nothing. It checks the two reader checkpoints against the sha256 that the pull of the first queue recorded (`LR16_s0` 6fa4ab461fc254aa, `LR16_s1` ac23087554a6ceff), runs the tool check and records its output without stopping on it, runs `check_k8_n2.py` (sha256 7230aee61fc1b40d) and stops if it fails, and uses its own start, stop and done files. Every other step (reader rows on the held-out clips, quantile maps, reader controls k1, k6, k7, the twelve rollouts) is the text of the first queue. The stop file of the first queue stays where it is.
* Reader `integ10k/read_i10k_n2.py` (same maker, sha256 b5ffd1dc268cdd3d): takes the small files of the continuation and the train-log part of k8 from `K8_N2.json`; the tool check is printed beside.
* `check_k8_n2.py --selftest`: 11 cases; the good case passes and ten planted faults fail (other batch order, a known-value interval that differs, a zero step with 20 rows, a zero step in a full batch, a flagged interval without a zero step, flagged intervals that differ, too many zero steps, a log record without gap labels, a rising lead term, flagged steps not all counted).

## 6. What does not change

Everything else of the card and of its note N1: parts, runs, reads E1 and E2, names, the other controls, what follows.
