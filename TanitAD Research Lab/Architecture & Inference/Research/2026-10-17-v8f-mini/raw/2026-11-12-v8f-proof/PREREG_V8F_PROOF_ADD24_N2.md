# PREREG_V8F_PROOF_ADD24_N2: second note to card ADD24 (LEADSUP). Control k5 fails as written for all six arms. Frozen before the six-arm lead read is run.

Tier T0 for the lead read, T1-kin (kinematic rollout on the model's own actions) for the cost read; never driving performance. STRATEGIC: PAUSED (PI item 25).

## 1. What the card says and what was measured

Control k5 of the card: in the train log of every arm `lead_sup_n_gap` is above 0 at every logged step, and the mean of `lead_sup` over the last 1,000 steps is below its mean over the first 1,000 steps. "An arm that fails k5 is not read."

MEASURED on the pod train logs (384 logged rows per arm, 19,164 steps; script `pod_k56.sh`), mean of `lead_sup` over the first 1,000 steps / the last 1,000 steps:

| arm | first 1,000 | last 1,000 | minimum of `lead_sup_n_gap` | k5 as written |
|---|---|---|---|---|
| LS_s0 | 0.3027 | 0.3741 | 51 | FAIL |
| LS_s1 | 0.2867 | 0.3540 | 5 | FAIL |
| LSD_s0 | 0.3798 | 0.4337 | 51 | FAIL |
| LSD_s1 | 0.3710 | 0.4066 | 5 | FAIL |
| LSH_s0 | 0.2119 | 0.3190 | 51 | FAIL |
| LSH_s1 | 0.2088 | 0.2897 | 5 | FAIL |

**By the card as written, k5 fails for all six arms and card ADD24 names no outcome.** This is reported first wherever a result of these arms is stated.

## 2. Cause, measured (binned means of the logged terms)

The lead term falls within the first 100 to 300 steps (first logged value at step 50: 0.49 to 0.72) and its presence part (binary cross-entropy) then rises again and stays higher: LSD_s0 0.286 (steps 100 to 300), 0.318 (300 to 1,000), 0.401 (1,000 to 4,000), 0.405 (last 1,000); LS_s0 0.178, 0.252, 0.366, 0.347; LSH_s0 0.069, 0.173, 0.338, 0.292. The gap part falls over the run (LS_s0 0.0124 at steps 100 to 300, 0.0090 at the end; LSD_s1 0.0098, 0.0095). The closing-speed part is flat (0.012 to 0.023 throughout). So the window "first 1,000 steps" of k5 already holds the lowest values of the run, and k5 as worded measures the later rise of the presence part, not whether the head was trained.

The rise is in the detached arms too, where the head cannot change the state. It is an observation, not a result: the state the head reads changes while the other terms train it, or the batches differ early. It is followed up in section 5.

## 3. What was seen before this note

The preliminary seed-0 lead read (three arms, registered in row H-V8F-PROOF-51), the cost read of all six arms (T1-kin, `spatial/leadsup/read/out_cost/`: no pair has a cell that is worse under the rule), the log table and the binned means above, control k6 (PASS for all six: the rollout dump of each arm names the checkpoint file of the arm, and the checkpoint is older than the dump). NOT seen: any lead-read number of a seed-1 arm.

## 4. Control k5-N2 (trained, not only built), defined here

All parts must hold for an arm to be read under this note:

* a. `lead_sup_n_gap` above 0 at every logged step (unchanged from the card).
* b. The first logged value of `lead_sup` (step 50) is above its mean over the last 1,000 steps. Known to hold on the logs for all six arms (seen).
* c. The mean of the gap part `lead_sup_gap` over the last 1,000 steps is below its mean over the first 1,000 steps. Seen in bins for all six arms; computed exactly before the read.
* d. Function, on the 429 scored windows the head never saw: gap R2 above 0 with an interval that excludes 0, and presence AUROC above 0.5 with an interval that excludes 0.5. A head that was built and not trained cannot pass d. Seen for the three seed-0 arms; not seen for seed 1.

Parts b and c were written after the logs were seen, so they add little; part d carries the control for seed 1.

## 5. Naming under this note, and the follow-up of the rise

* Every outcome of card ADD24 read from these arms is stated as "under notes N1 and N2" and carries the mark N2 in the register; the card-form results (k5 FAIL for all arms; the card-form slot control k3) are stated first.
* Follow-up of the rise of the presence part, committed here: (1) card ADD27 trains a projection of the tokens for the lead term alone on the same data path; its logged presence, gap and closing-speed parts over the last 1,000 steps are set beside those of LSD_s0 and LSD_s1 (reported, names nothing); (2) if the token projection ends clearly lower, the next arm is the lead slot beside the state of card ADD27 section 6; (3) a read of the state at several training steps needs checkpoints that were not kept, so it is part of the next training card, not of this one.
