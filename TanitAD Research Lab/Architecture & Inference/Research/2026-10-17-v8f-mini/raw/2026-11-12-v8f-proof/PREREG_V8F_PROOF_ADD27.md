# PREREG_V8F_PROOF_ADD27: TOKPROBE. Where is the limit of reading the closing speed of the lead: in the pooled frozen camera tokens, or in the 640-wide state the world model makes of them? A projection of the pooled tokens trained for the lead labels alone, read with the head and the reader of card ADD24. No training of the model.

Frozen before any number of this card is computed. Addendum to `PREREG_V8F_PROOF.md`, to card ADD24 and its note N1. Tier T0 (a diagnostic on logged frames, never driving performance). Evidence class of every result: MEASURED once run. LONGITUDINAL (distance keeping) is the family this lever serves and it is not measured by this read; LATERAL, TACTICAL: not measured; STRATEGIC: PAUSED (PI item 25).

## 1. What is known before it

* Card ADD23 (MEASURED, `spatial/leadhead_b/VERDICT_LEADHEAD_B.md`): a ridge head on five frames of the grid state, fitted on 30,865 rows, reads the closing speed on the held-out windows at R2 +0.10 to +0.18, with a learning curve that still rises.
* Card ADD24, preliminary seed-0 run of its reader (`spatial/leadsup/read/out_prelim_s0/`, a run that names nothing; the both-seed names are not yet read): the supervised head trained on all 4,512 clips reads the closing speed at R2 +0.155 (weight 1), +0.092 (weight 10), +0.178 (detached state) on the held-out windows and at +0.20 to +0.21 on training clips; the gap at R2 0.60 to 0.72; presence AUROC 0.90 to 0.92. A head that reaches 0.21 on clips it was trained on is limited by its input, not by the row count.
* Two inputs can be the limit: the 640-wide state (40 cells of width 16, shaped mainly by the world-model term), or the pooled 4 by 10 grid of frozen camera tokens it is made from. The token cache of the pod holds the pooled tokens; tokens before pooling are not cached.

## 2. The probe (pod GPU, one job; the model and its checkpoints are not touched)

* Input: the cached pooled camera tokens of a frame, 40 tokens of width 1,024, frozen. A per-token network 1,024 to 128 (GELU) to W gives a vector of 40 * W per frame; on five consecutive frames the head of card ADD24 (`LeadSupHead`: per-frame linear to 64, concatenation, 320 to 128 to 3) reads presence, gap and closing speed.
* Arm types: TK16 with W = 16 (640 per frame, the width of the state) and TK64 with W = 64 (2,560 per frame). Two initialisation seeds each (0 and 1): TK16_s0, TK16_s1, TK64_s0, TK64_s1, trained on the same batches in one process.
* Training: the data set, exclusion, window layout, labels (`lead_sup_label_arrays`), loss (`lead_sup_terms`), batch size, number of steps (19,164), optimiser and learning-rate schedule of the arms of card ADD24, read from the argv record of arm LSD_s0; only the probe is trained, on the lead term alone.
* Outputs per arm: the projected vectors of the evaluation episodes and of the train probe clips in the layout of the state files, the head tensors and the head outputs in the layout of `pod_lead_apply.py`, so that the reader of card ADD24 reads a probe arm exactly as it reads a model arm.

## 3. Read (the reader of card ADD24 with note N1, the 429 scored windows of 48 evaluation episodes)

M1 closing-speed R2, M2 gap R2, M4 presence AUROC, the slot increment DELTA of note N1, on the evaluation windows and on the train probe clips, for the four probe arms beside LSD_s0 and LSD_s1 (the same head on the state of the model, trained on the same labels and clips). Paired difference of M1 between a probe arm and the LSD arm of the same seed index over the scored windows (episode-cluster bootstrap B 2,000, rng 0); floor = the absolute difference of M1 between LSD_s0 and LSD_s1.

## 4. Controls (all must pass before any outcome is named)

* p1 the probe reaches its parameters: non-zero gradient on the per-token network and the head, and exactly 0.0 on a control parameter that is not in the graph; a mutation that removes the loss line makes a test fail.
* p2 planted tokens: with a synthetic token set that linearly encodes a known gap, the trained probe reads gap and closing speed at R2 above 0.9 on held-out clips.
* p3 shuffled labels: trained with the closing-speed labels permuted across rows, the held-out closing-speed R2 is at most +0.02 (short run of 2,000 steps).
* p4 no label and no ego input in the forward path: the probe input is the token array alone (asserted by the function signature and by a run with the labels replaced by noise at read time giving identical outputs).
* p5 the train log: the lead term at the last 1,000 steps below the first 1,000 steps for every arm; the count of gap-labelled frames above 0 at every logged step.
* p6 the controls of the reader (k1, k2, k3-closing, k3-slot of note N1, k4) for every probe arm.

## 5. Outcomes (both committed), per arm type, both seeds needed

* `TOKENS-HOLD-CLOSING` if M1 on the evaluation windows is above 0 with an interval that excludes 0 at both seeds, AND the paired difference to LSD of the same seed index is above 0 with an interval that excludes 0 at both seeds, AND its mean is larger than the floor. Else `TOKENS-FLAT`.
* Reported, naming nothing: M1 on the train probe clips (the in-sample ceiling), M2, M4, DELTA.

## 6. Committed next steps

* `TOKENS-HOLD-CLOSING` at TK16: a projection of the state's own width holds the closing speed when it is trained for it, so the world-model term crowds it out of the state. Next arm, own card: a lead slot beside the state, a small projection of the tokens trained by the lead term alone and given to the head and to the distance rule (stage 1 of card ADD20 on a dedicated slot), two seeds on the pod.
* `TOKENS-HOLD-CLOSING` at TK64 only: the width is the limit. Next arm, own card: the grid state with a wider cell and the lead term.
* `TOKENS-FLAT` at both: the pooled frozen tokens do not carry the closing speed at this precision. The lever is then the cue itself: tokens before pooling, a finer grid or a crop of the lead region, or an adapted encoder. Each needs the camera frames re-encoded and stored (the pod holds the pooled tokens only); that is a data and compute step the PI decides, and it is named as a blocker with its cost, not run.

## 7. Cost and limits

* One GPU job on the pod of about one hour after the rollouts of card ADD26; no training of the model.
* The two seeds of a probe arm differ in initialisation only (same batches); the floor is taken from the two LSD arms, which differ in the whole training run.
* A probe on the tokens says what the pooled tokens allow, not what the model does; 429 scored windows of 48 episodes; one corpus.
