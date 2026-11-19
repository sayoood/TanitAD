# PREREG_V8F_PROOF_ADD43_N1 : what the build of card ADD43 found before any run (sampler chunk, bootstrap form, name guard, budget)

Note to card `PREREG_V8F_PROOF_ADD43.md` (FINEGRID-R). Frozen before any run of the card: no arm is trained, no head output of a new arm exists, the changed tool has not run on the pod (time of the freeze: the line of this note in `TIMES_NOTE.md`; hash in the sidecar file). Tier T0 (a diagnostic on logged frames, never driving performance). STRATEGIC: NOT COMPUTED (a token probe reads no strategic quantity; n = 0).

## 1. The sampler chunk depends on the grid (finding of the build; read from source)

* The trainer sizes its clip store by the token count (`v8f_mini_train.py:2736`: 174 clips at 40 tokens per frame, 43 at 160 on this data; the pilot's `train_meta.json` records 43) and takes the sampler chunk as `max(4, min(48, store - 2))` (`:2851`): **48 on the 4 by 10 grid, 41 on the 8 by 20 grid**. The chunk sets the pools in which the windows are shuffled.
* So with the unchanged tool and the same argument line the two processes of card section 2 would NOT see the same windows in the same order; control r1 would fail by construction.
* Change (tool `finegrid_r/tok_probe_gr.py`, made by asserted edits): a lever `--sampler-chunk-like GRID`. The 4 by 10 process and the 4 by 10 control arm run with `--sampler-chunk-like pool8x20` (chunk 41). Without the flag the tool takes the original expression (control g1 on the changed tool, on the pod, before any arm). The 8 by 20 side is not changed: a chunk of 48 at 160 tokens would need a larger clip store than the trainer takes.
* Control r1 is read from the two batch logs (`r1_batches.json` of each process: for the first 200 batches a hash of the window indices and a hash of the full items, in order). It must show equal sequences; a planted difference of one batch must fail it (in the pod checks).

Consequences, stated before the data:

* The new 4 by 10 arms differ from the banked probe of card ADD27 in the seed AND in the sampler chunk (41 against 48). Card section 4, control r2, is read with this sentence added.
* The pilot of card ADD41 (row H-V8F-PROOF-87) compared 8 by 20 arms at chunk 41 with banked 4 by 10 arms at chunk 48, so on different window orders. That is a limit of the pilot and is written into its row from this hold on. The pilot named nothing.

## 2. What the eight seeds vary

The arm seed sets the probe initialisation. The batch order comes from the trainer seed and is shared by all arms of a process and, by r1, by both processes. So the paired difference at seed index K is paired on the evaluation windows and on the training batch order, and the interval over seeds carries the initialisation variance of the probe, not the variance of the batch order. Added to the limits of card section 7.

## 3. Form of the two-level bootstrap

Crossed: per replicate one draw of seed indices with replacement and ONE draw of episodes with replacement, shared by all drawn seeds, because every seed is scored on the same 48 evaluation episodes. B 2,000, rng 0. The form with an own episode draw per drawn seed is printed beside and names nothing.

## 4. Name guard

Note N1 of card ADD41 applies to this card: a name from the pod run is given only if every comparison that decides it is farther than 1e-4 from its bar (each interval end from 0, the gap MAE ratio from 0.90). The count rule (at least six of eight seed indices) has no such guard. The reader prints the distances; the lead applies the guard when the row is written; a name closer than that is written "not named yet" and waits for a run of the reader on the dev box.

## 5. Control p5

The parts of the lead term (presence, gap, closing) are already in the train log of the pilot's tool; no logging change. For an arm with permuted closing labels the tool reads p5 on presence plus gap and reports the closing part beside (card section 4).

## 6. Checks before the run (pod CPU, `finegrid_r/run_checks_pod.py`)

Control g1 on the changed tool, the r1 log and its planted difference, the chunk lever, p5 in the tool, g3, g4. The run starts only with 0 failed checks; the checks also write the controls file the reader needs.

## 7. Budget

ESTIMATED 3 to 5 pod-h on one GPU slot (the card said 2 to 3), about 1.5 GB of pod volume (evaluation states only). Never a fourth GPU job; 28 GB of the container memory limit free before every GPU step.

## 8. Unchanged

The arms, the steps, the read, the statistic apart from section 3, the outcomes and what follows (card sections 2, 3, 5, 6).
