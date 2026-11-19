# PREREG_V8F_PROOF_ADD43 (FINEGRID-R) : a replicate that can separate what one step of grid resolution does to the lead read (probe only)

Frozen before any run of this card. No model is trained: token probes on frozen cached tokens. Tier T0 (a diagnostic on logged frames, never driving performance). LONGITUDINAL (distance keeping) is the family this lever serves and it is not measured by this read; LATERAL, TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (a token probe reads no strategic quantity; n = 0).

## 1. Why this card

* The pilot of card ADD41 (row H-V8F-PROOF-87) ended without a name: the train-log check p5 failed on the arm with permuted closing labels, and the mean gains of the gap R2 (0.13 and 0.16) lie under the largest seed spread inside one arm type (0.211, two seeds per type); 429 scored windows of 48 evaluation episodes (MEASURED, T0).
* The other v8f session read the same question on another quantity and did not separate it either (RAW8 minus RAW4 +0.0071; INHERITED from its state file).
* Coordination decision D1 (`COORD_V8F_2026-11-19.md`): no 8 by 20 base model is trained for either session before a read that can separate the grid effect. This card is that read. It decides whether a shared 8 by 20 base is carded for the PI.

## 2. Arms (the probe of card ADD27 section 2, with only the grid and the seed changed; the tool `tok_probe_g.py` of card ADD41)

* Eight new seed indices K = 2 to 9. Per K three probe arms: `TK16_sK` (grid 4 by 10, per-token width 16), `TK64_sK` (grid 4 by 10, width 64), `F16_sK` (grid 8 by 20, width 16). 24 arms.
* Control arms: `TK16SH_s2` and `F16SH_s2` (permuted closing labels, 2,000 steps).
* Two processes, one per grid, each with all its arms on the same batches; both with the trainer argument line and trainer seed of card ADD27, so both grids see the same windows in the same order (control r1). Data set, exclusion, window layout, labels, loss, batch size, 19,164 steps, optimiser, schedule and `--input-norm ln` of card ADD27. The arms of the pilot at seeds 0 and 1 are not retrained; they are reported beside and are not pooled with the new seeds.
* One pod GPU job at a time, started only with fewer than three GPU jobs and 28 GB of the container memory limit free (never a fourth GPU job). Budget ESTIMATED from the pilot (MEASURED 73 min for five arms on the 8 by 20 grid in one slot): 2 to 3 pod-h, under 8 GB of pod volume.

## 3. Read (the reader of card ADD24 with note N1; the 429 scored windows of 48 evaluation episodes; head outputs only)

* Per arm: M2 gap R2, the gap MAE in metres, M1 closing-speed R2, M4 presence AUROC.
* Per seed index K, paired on the same windows: G16 = `F16_sK` minus `TK16_sK` (same per-token width), G16w = `F16_sK` minus `TK64_sK` (same width per frame), on M2, on the gap MAE and on M1.
* Statistic: the mean over the eight seed indices of the paired difference, with a two-level bootstrap (seed indices with replacement, then episodes with replacement; B 2,000; rng 0). The eight per-seed values are listed. This interval carries the run-to-run variance of the probe, which the pilot could only bound by a floor.

## 4. Controls (all must pass before any outcome is named)

* g1 to g7 of card ADD41 with its note N1 (on the pod the reproduction of the banked 4 by 10 reads within 1e-5), with one change stated here, before the data: **p5 for an arm with permuted closing labels** is read on the parts of the lead term whose labels are intact (gap and presence), and the permuted part is reported beside; for every other arm p5 is the check of card ADD27 as written (the lead term of the last 1,000 steps below that of the first 1,000 steps; gap-labelled frames above 0 at every logged step). If the train log does not carry the parts, the tool is changed to log them, and control g1 (bit-identical probe weights on the default path) is run again on the changed tool.
* g6 for both control arms: closing R2 at most +0.02.
* r1 the two processes saw the same windows in the same order: the sequence of window indices of the first 200 batches is equal between the two grids (logged by the tool; a logging change only).
* r2 the new 4 by 10 arms are the probe of card ADD27: `TK16` and `TK64` at the banked seeds 0 and 1 are not retrained; control g7 covers the reader; for the new seeds the checkpoint of every arm records its grid, token count and width (g5).
* r3 planted difference for the two-level reader (synthetic head outputs): a planted gain of 0.10 in R2 at every seed is read as separated above; a planted gain of 0.10 at two of eight seeds and 0 at six is not; eight identical pairs read exactly 0. A mutation that resamples episodes only (seeds held fixed) makes a test fail.

## 5. Outcomes (committed before the data)

* `FINER-GRID-READS-GAP-R`: the two-level intervals of G16 and of G16w on M2 lie above 0, at least six of the eight seed indices are above 0 in each, and the mean gap MAE of F16 over the seeds is at most 0.90 times the lower of the mean gap MAE of TK16 and of TK64.
* `FINER-GRID-WORSE-R`: the two-level intervals of G16 and of G16w on M2 lie below 0.
* Else `FINER-GRID-FLAT-R`.
* `FINER-GRID-READS-CLOSING-R` (the positive side only): the rule of the first bullet on M1, without the MAE clause.
* Beside, naming nothing: the pilot's seeds 0 and 1; M4; the gap error by gap range (below 20 m, 20 to 40 m, above 40 m); the standard deviation over seeds of M2 per arm type.

## 6. What follows (committed before the data)

* `FINER-GRID-READS-GAP-R`: the pooling of the tokens limits the lead read. A card goes to the PI for one shared 8 by 20 base for both v8f sessions (two seeds, from scratch, cost taken from the measured log of the existing 8 by 20 base of the other session), with the lead-head arm, the family reads, the distance rule and the planning tool on it.
* `FINER-GRID-FLAT-R`: one step of grid resolution does not change the lead read at this data size, measured with the probe's own run-to-run variance. No 8 by 20 base is trained on this ground, in either session; the vision lever of the other session waits with it. The levers that remain for the lead read are data scale (the 10k corpus) and a second encoder pass over a crop of the lead region (needs the frames; a data step named with its cost for the PI).
* `FINER-GRID-WORSE-R`: reported as written; no weight-decay or width sweep after the fact.
* A control fails: no name; the failing control is repaired under a note before any repeat, and the outcome differences stay withheld.

## 7. Limits

A probe on the tokens says what the tokens allow, not what a model does with them; 429 scored windows of 48 episodes (the evaluation set is not enlarged by this card); one corpus; tokens stored as float8; the grid before pooling is not read; eight seeds bound the probe's run-to-run variance and not the variance of the encoder (one frozen encoder).
