# PREREG_V8F_PROOF_ADD41: FINEGRID. Does a finer grid of the frozen camera tokens read the lead gap better? The probe of card ADD27 on the cached 8 by 20 grid, beside its arms on the 4 by 10 grid. No training of the model, no camera frame.

Frozen before any number of this card is computed and before the tool of this card exists. Addendum to `PREREG_V8F_PROOF.md`, to card ADD27 (TOKPROBE, row H-V8F-PROOF-56) and its note N1. Go of the PI on 2026-11-18 ("Yes, run pilot") for a small pilot of the finer lead read. Tier T0 (a diagnostic on logged frames, never driving performance). Evidence class of every result: MEASURED once run. LONGITUDINAL (distance keeping) is the family this lever serves and it is not measured by this read; LATERAL, TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (a token probe reads no strategic quantity; n = 0).

## 1. Why, and what was found before the freeze

* Rows 58, 59 and 81 (MEASURED): every vision-fed source of the distance rule leaves about twice the unsafe-headway rate of the label source, and the head reads the lead gap with an error of 5 to 6 m. Row 56: a probe on the pooled 4 by 10 token grid reads the lead no better than the head on the 640-wide state. The finer lead read was named there as a decision of the PI, with the re-encoding of the camera frames as its cost.
* Found on 2026-11-18 while sizing the pilot (MEASURED, `data/cache_pod/train/meta.json` on the pod): the token cache of the pod stores TWO grids per frame, 4 by 10 (40 tokens) and 8 by 20 (160 tokens), both from the frozen DINOv3 ViT-L/16 encoder (pipeline `ra1a-pooled-v1`, stored as float8), for all 4,572 train clips and the evaluation clips; the trainer and the clip store accept `--grid pool8x20`. The camera frames themselves are not on the pod (`data/podwork` holds the ledger only). So one step of resolution (four times the tokens) can be read with no frame and no re-encoding; the grid before any pooling (16 by 40 for a 256 by 640 frame, to be confirmed from the encoder script) needs the frames and is NOT part of this card.
* The earlier probe run (four arms, 19,164 steps, one process) took 25 minutes on the pod GPU (file times of `runs/tokprobe/train`).

## 2. Arms (the probe of card ADD27, section 2, with only the grid changed)

* F16_s0, F16_s1: grid 8 by 20, per-token width W = 16 (2,560 per frame). F64_s0, F64_s1: grid 8 by 20, W = 64 (10,240 per frame). Per-token network 1,024 to 128 (GELU) to W, the head `LeadSupHead` on five frames, token normalisation `--input-norm ln` (the form that names, note ADD27_N1), the data set, exclusion, window layout, labels, loss, batch size, 19,164 steps, optimiser and schedule of card ADD27, all four arms on the same batches in one process.
* Reference arms, not retrained: TK16_s0, TK16_s1, TK64_s0, TK64_s1 of card ADD27 (grid 4 by 10, main form), with their banked head outputs.
* One pod GPU job, started only when fewer than three GPU jobs run and 28 GB of the container memory limit are free (never a fourth GPU job). Budget ESTIMATED: 1 to 2 A40-h (data-bound; four times the token volume per batch), under 8 GB of pod volume.

## 3. Read (the reader of card ADD24 with note N1, the 429 scored windows of 48 evaluation episodes; head outputs only)

* Per arm: M2 gap R2, the gap MAE in metres (mean absolute error of the head's gap output on the scored windows with a lead and a finite label), M1 closing-speed R2, M4 presence AUROC; the same on the train probe clips (in-sample, naming nothing).
* Paired differences on the same windows, same seed index K (episode-cluster bootstrap, B 2,000, rng 0): G16 = F16_sK minus TK16_sK (same per-token width), G16w = F16_sK minus TK64_sK (same width per frame), G64 = F64_sK minus TK64_sK, each on M2, on the gap MAE and on M1.
* Floor: the largest absolute difference of M2 between the two seeds of one arm type, over the four arm types (TK16, TK64, F16, F64); the same for M1.

## 4. Controls (all must pass before any outcome is named)

* g1 the default path is unchanged: the tool of this card with the grid 4 by 10 gives bit-identical probe weights to the tool of card ADD27 for the same seed and steps (dev box, synthetic data).
* g2 the two cached grids belong together: on 200 frames of 20 train clips the 2 by 2 mean of the 8 by 20 tokens against the stored 4 by 10 tokens: correlation at least 0.99 (reported with the median absolute difference). If it is lower, the 8 by 20 grid is not a refinement of the 4 by 10 grid; the read is then reported as "another cached grid" and no name of section 5 is given.
* g3 planted tokens at 160 tokens: a synthetic token set that linearly encodes a known gap is read at R2 above 0.9 on held-out clips (control p2 of card ADD27 at the new grid).
* g4 the probe reaches its parameters at 160 tokens: non-zero gradient on the per-token network and the head, exactly 0.0 on a control parameter outside the graph; a mutation that removes the loss line makes a test fail; a mutation that feeds the 4 by 10 grid under the 8 by 20 flag makes a test fail.
* g5 declared against built: the checkpoint of every F arm records 160 tokens and the batches it was trained on have the shape [batch, frames, 160, 1024]; the run refuses otherwise.
* g6 no label and no ego input in the forward path (the signature, and a run with noise labels at read time giving identical outputs); the train log check p5 of card ADD27 for every arm; control p3 (permuted closing labels, 2,000 steps, arm F16SH_s0): closing R2 at most +0.02.
* g7 the controls of the reader (k1, k2, k3, k4 of card ADD24 with note N1) for every F arm, and the reader gives for the TK arms the M1, M2, M4 that row 56 banked (max absolute difference 0.0 on the banked outputs).

## 5. Outcomes (all committed before the data; both seeds needed)

* `FINER-GRID-READS-GAP`: G16 and G16w on M2 are above 0 with intervals that exclude 0 at both seeds, each mean is larger than the floor, and the gap MAE of F16 is at least 10 % below the lower of the gap MAE of TK16 and TK64 of the same seed index, at both seeds.
* `FINER-GRID-WORSE`: G16 and G16w on M2 are below 0 with intervals that exclude 0 at both seeds.
* Else `FINER-GRID-FLAT`.
* `FINER-GRID-READS-CLOSING` (the positive side only): the rule of the first bullet on M1, without the MAE clause. A flat closing read names nothing (control p2b of note ADD27_N1 did not pass for the closing speed).
* Beside, naming nothing: G64 with the same statistics (the capacity form); M4; the in-sample reads; the gap error by gap range (below 20 m, 20 to 40 m, above 40 m) for F16 and TK64.

## 6. What follows (committed before the data)

* `FINER-GRID-READS-GAP`: the pooling of the tokens is a limit of the lead read. Next, own card and a go of the PI for the compute: model arms on the cached 8 by 20 grid (`--grid pool8x20`, 160 tokens; the lead-head arm and the rollout-trained base, two seeds, with the family reads and the distance rule and planning tool on them). The grid before pooling (frames to be brought to the pod and re-encoded) is asked for only if those arms confirm the gain.
* `FINER-GRID-FLAT`: one step of resolution does not change the read; the pooled grid is not the limit at this scale and the 8 by 20 model arms are not trained on this ground. The levers that remain are a second encoder pass over a crop of the lead region (needs the frames; a data step named with its cost for the PI) and, inside the present tokens, the trained low-quantile gap head (card ADD31 section 8).
* `FINER-GRID-WORSE`: reported as written (a probe with four times the inputs at this data size); no weight-decay or width sweep is run after the fact.

## 7. Limits

A probe on the tokens says what the tokens allow, not what the model does; 429 scored windows of 48 episodes; one corpus; two initialisation seeds per arm type; the reference arms come from an earlier run of the same tool (control g1 covers the code path, not the run); tokens stored as float8; the grid before pooling is not read.
