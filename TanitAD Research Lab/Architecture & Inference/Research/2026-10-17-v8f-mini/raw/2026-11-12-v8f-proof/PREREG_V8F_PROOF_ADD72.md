# PREREG_V8F_PROOF_ADD72 (AGEDOSE10K): the cost of rollout age at 1, 2 and 4 s of own driving, split into the carried pose and the own motion state; the label-fed candidate's hold on the unsafe rate at those ages

Card of the coordinator fork (v8F proof study). Written after the reads of card ADD70 (rows 128, 129) and card ADD71 (parts A and B, notes N1 and N2) and before any rollout of this card exists (time: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). No training. Tier of every number: T1-kin/hb0.2 under the study wrapper of record of card ADD70 (kinematic rollout on the model's own actions with the heading bound 0.2 1/m; logged camera frames; the lead does not react); never driving performance. Models: the two 10k line checkpoints `UE_s0`, `UE_s1` of the other session (code cdbff971), INHERITED.

## 1. Question and what is known

Known (MEASURED, rows 129 and card ADD71): after 6 s of own driving under readout K the tactical track error is higher by 14.6 m than in the same window started fresh; card ADD71 part B splits that into 7.29 m carried pose (the fresh plan moved to the own pose) and 7.33 m own motion state (the plan the model makes after its own driving against the fresh plan at the same pose). On the along-track error the state part is the larger one (5.89 against 4.24 m), on the heading error the pose part (4.95 against 2.69 deg); the speed error (+1.54 m/s) is the state part alone, by identity.

Not known: whether the state part is there from the first second of own driving (the model reads a history it rolled itself differently from a logged one: a cost of the input regime) or grows with the age (a compounding cost). The two cases ask for different things from the training line of decision D16 C and decide how old a window acceptance check C5 has to be.

Second question: card ADD70 as re-read measured the label-fed candidate's unsafe rate at 6 s of own driving (not separated from fresh, contact 0). Whether that holds at 1, 2 and 4 s is not measured.

## 2. Runs (12 rollouts, pod of this fork, one GPU job at a time; about 55 minutes)

All on the P = 30 grid of card ADD70 (`longroll/grid_p30.json`, 1,271 windows, 280 scored lead windows; every window has its pre-roll inside the clip for every P up to 30), 2.0 s cycle, `--kappa-max 0.2`, model seeds 0 and 1, wrapper of record of card ADD70 (unchanged), queue `integ10k/pod_agedose10k_queue.sh` written by `integ10k/make_agedose_queue.py` from the queue of card ADD71 (helper functions, start checks and the rollout function are its lines).

* K block (first): `I10R0KP5Hb20`, `I10R0KP10Hb20`, `I10R0KP20Hb20` (readout K: `--readout causal_g --acc-gain 0 --path-exec`; pre-roll of 5, 10 and 20 steps = 1, 2 and 4 s).
* Candidate block (second): `I10RLcKF5P5Hb20`, `I10RLcKF5P10Hb20`, `I10RLcKF5P20Hb20` (label-fed lead rule with closing speed under readout K, second look `--rule-every 5`).
* Banked and used as they are: `I10R0KHb20`, `I10RLcKF5Hb20` (fresh), `I10R0KP30Hb20`, `I10RLcKF5P30Hb20` (6 s), `I10R0Hb20` (spread run of the family reader).

If the queue stops after the K block, the K block is read alone and the candidate block is stated as not run.

## 3. Readings and names

Estimators as in card ADD71 with its notes: headway (U unsafe rate on the scored lead windows, C contact, F falling behind) by the paired episode-cluster bootstrap pooled over the two model seeds; family cells by the family reader of card ADD46 (two-level bootstrap, B 2,000; BETTER or WORSE only when the interval excludes 0 and the difference is outside the spread of `R0Hb20` over the two seeds on the same windows); STRATEGIC by the instrument of card ADD66 (route goal at 30, 50, 80 m at the third tick; a decision between routes is not scored, n = 0). Reader of record for pre-rolled runs: the repaired reader of notes N4 and N5 of card ADD70 (step-derived cells on the track moved to its own start point).

**3a. AGE pairs, aged minus fresh on the same windows**, for P in 5, 10, 20: AGE-K-P<P> (`R0KP<P>Hb20` minus `R0KHb20`) and AGE-LABEL-CAND-P<P> (`RLcKF5P<P>Hb20` minus `RLcKF5Hb20`). Names `AGEDOSE10K-<PAIR>-<FAMILY>-BETTER|WORSE|MIXED|NOT-SEPARATED` for the four families and `AGEDOSE10K-<PAIR>-U-LOWER|HIGHER|NOT-SEPARATED` (LOWER or HIGHER needs the pooled interval to exclude 0 and both seeds on that side). The pairs at P = 30 are re-read as identity pairs and name nothing here.

**3b. Split of the cost of age under readout K, per age P in 5, 10, 20, 30** (tool of card ADD71 part B with its note N2, the same rule at every age): POSE(P) = the fresh track moved rigidly to the own pose of the aged run at t0 (offset from the tick file of the aged run at that P) minus the fresh track; STATE(P) = the aged track minus that moved track; on the windows where the cell has a value in all three tracks. By identity POSE is 0 for the speed, yaw rate and curvature error.

**3c. Shape of the state part**, for four cells: TRACK (`TACTICAL:ade_tac_m`), ALONG (`LONGITUDINAL:along_mae_m`), SPEED (`LONGITUDINAL:speed_mae_mps`), HEADING (`LATERAL:heading_mae_deg`). With S(P) the STATE part of the cell at age P, read on the windows where the cell has a value at both ages, two-level bootstrap as above:

* `AGEDOSE10K-STATE-<CELL>-ONSET` when the interval of S(5) minus one half of S(30) excludes 0 above;
* `AGEDOSE10K-STATE-<CELL>-GROWS` when the interval of S(5) minus one third of S(30) excludes 0 below;
* `AGEDOSE10K-STATE-<CELL>-BETWEEN` otherwise;
* `AGEDOSE10K-STATE-<CELL>-NO-STATE-COST` when S(30) is not above 0 with its interval excluding 0 (then no shape is read).

Why these two bars, fixed now: a cost that grows in proportion to the age has S(5) at one sixth of S(30); a cost that is complete after the first second has S(5) equal to S(30). ONSET asks for at least one half, GROWS for at most one third (twice the proportional share). The yaw rate error gets no shape name (its state part at 6 s is not separated from 0). Reported without a bar: S(10), S(20), POSE(P), the total and the share of POSE at every age, and the mean and median pose offset at t0 per age.

## 4. Controls (a control that does not pass: no name is written, the fault is logged before a repair)

* Queue: every rollout ends with its dump, its tick file and its pre-roll log; the wrapper and the pinned tool have the hashes of card ADD70.
* Reader: the controls of the reader of card ADD71 as they apply (e1 environment identity, window counts, pre-roll logs with the P of the run name, readout logs, hashes, reader identity, headway and family reader controls on the grid, strategic rows), n1 to n3 of note N4 in the form of note N5, and a1: the pairs AGE-K and AGE-LABEL-CAND at P = 30 read here equal the re-read of card ADD70 to 1e-9.
* Split: b1 to b3 and b5, b6 of card ADD71 at every age; b4 at P = 30: the total equals the banked total of the pair AGE-K of the re-read of card ADD70 and POSE plus STATE equals it, both to 1e-9; b4 at P = 5, 10, 20: POSE plus STATE equals the total to 1e-9 and the total equals the cell of the pair AGE-K-P<P> of the reader of 3a to 1e-9 on the cells without a missing value (a cross-tool identity); planted start offset of note N4.
* d1 (shape statistic): on planted per-window arrays the statistic returns ONSET for a state part equal at both ages, GROWS for a state part of one sixth, BETWEEN for a state part of 0.4, and NO-STATE-COST for a state part of 0 at the older age.

## 5. What follows, fixed now (the card changes no convention and adopts nothing)

* ONSET on TRACK: the state part of the cost of age is a cost of the input regime; it is there once the model reads a history it rolled itself. Then (i) acceptance check C5 of the candidate specification can be read at 1 s of own driving as well as at 6 s, and a training arm on own rolled history needs short rolled histories first; (ii) a study arm on the hand-over of the own motion state to the model at inference is worth its card (which input of the state carries the cost), and it is the next card of this fork.
* GROWS on TRACK: the state part compounds; training has to roll histories as long as the age it is to be used at, C5 stays at 6 s, and no inference-side arm on the hand-over is opened by this card.
* BETWEEN: both are reported, C5 stays at 6 s, and the next card is chosen from the per-age table with nothing claimed about the shape.
* The names on ALONG, SPEED and HEADING are reported beside the TRACK name and decide nothing by themselves.
* Candidate block: when U of AGE-LABEL-CAND-P<P> is NOT-SEPARATED or LOWER at 5, 10 and 20 steps and the contact share of the aged candidate is 0 at both seeds at each age, the sentence of record is "the label-fed rule's hold on the unsafe rate and on contact is measured at 1, 2, 4 and 6 s of own driving"; an age where U is HIGHER or contact is above 0 is named as the limit of the hold.
* A family WORSE in an AGE pair is the cost of age at that age and is reported as written.

## 6. Limits stated now

* Two model seeds are not a replicate study; a separated interval answers "another draw of episodes" and is necessary, not sufficient. The readout and the rule do not sample; no inference seed is involved.
* At 5 steps of own driving the model's input history can still hold logged steps; the card reads the age, not the share of own steps in the history, and says so with an ONSET or GROWS name.
* The split moves the fresh plan rigidly; it does not re-plan at the own pose with logged state. The POSE part is the cost of where the window starts while the camera frames stay the logged ones, which a kinematic rollout cannot repair or fairly score.
* The label-fed rule reads the lead from labels; the reader-fed form is not run at these ages (card ADD70 as re-read holds its 6 s reading).
* STRATEGIC is read in 3a and not split in 3b.
