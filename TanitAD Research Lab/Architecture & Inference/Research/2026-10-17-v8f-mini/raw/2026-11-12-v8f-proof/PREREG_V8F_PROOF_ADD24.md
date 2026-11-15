# PREREG_V8F_PROOF_ADD24: LEADSUP. A supervised head on lead presence, gap and closing speed, trained together with the grid-state arm. Does the level-1 state hold the motion of the lead vehicle once it is asked for it, and does asking cost anything in the rollout on the model's own actions?

Frozen before any arm of this card is trained and before the result of card ADD23 was read (its reader was still running at the freeze; this card is blind to it). Addendum to `PREREG_V8F_PROOF.md`, to `PREREG_V8F_PROOF_ADD20.md` (the object slot) and to `PREREG_V8F_PROOF_ADD21.md` / `ADD23.md` (heads fitted after training). Tiers: T0 for the lead read (a world-model diagnostic on logged frames), T1-kin (kinematic rollout on the model's own actions, logged frames) for the cost read; neither is driving performance. Evidence class of every result: MEASURED once run. STRATEGIC: PAUSED (PI item 25).

## 1. What is known before it (MEASURED)

* Card ADD20 (`spatial/objslot/VERDICT_OBJSLOT_S0.md`): with the label gap and closing speed, the kinematic rule R2 predicts the gap 2.0 s ahead with skill +0.750 over the copy (`KIN-CEILING-HOLDS`); with the gap read by a one-frame probe from the state, the same rule has no skill (`SLOT-FLAT`).
* Card ADD21 (`spatial/leadhead/VERDICT_LEADHEAD_S0.md`): heads fitted after training on at most 1,926 labelled rows read the closing speed at held-out R2 -0.06 to -0.11 (`HEAD-CLOSING-FLAT`, `HEAD-SLOT-FLAT`, `STATE-LACKS-FUTURE`), with the row count named as the limit.
* The lever of this card is the one the committed next steps of cards ADD21 and ADD23 name in two of three outcomes: supervise the state during training, on all training clips (4,512 clips, 461,518 label-join rows, of which 26.7 % carry a gap label inside 0 to 60 m on the pod join).
* The loss scale of the base arm (pod, `runs/spatial/G_s0/train_log.jsonl`, last step): total 14.1, of which the world-model term (117.4 before its weight) is the largest part. A lead term of order 0.3 to 0.7 at weight 1 is a few percent of the total, so the card carries a second weight (10) as a dose arm.

## 2. Arms (pod, two seeds each, the argv, steps, data and exclusion of arm G of card ADD12)

| arm | added argv | what it asks |
|---|---|---|
| LS_s0, LS_s1 | `--lead-sup on --w-lead-sup 1.0` | the head trains with the model, its gradient reaches the state projection and the adapter |
| LSH_s0, LSH_s1 | `--lead-sup on --w-lead-sup 10.0` | the same at ten times the weight (dose) |
| LSD_s0, LSD_s1 | `--lead-sup on --w-lead-sup 1.0 --lead-sup-detach on` | the head reads the detached state: its term reaches no parameter outside the head. The control for the gradient question, and a head trained on all clips on a state that was never asked |

Baseline: G_s0 to G_s5 (card ADD12 and card ADD22). Head: per-frame linear 640 to 64 over the level-1 grid states of the last 5 frames, concatenated, 320 to 128 to 3 (presence logit, gap / 50 m, closing speed / 10 m/s), +82,499 parameters, built last so that the initial weights of every other tensor equal those of G. Loss: BCE(presence) + Huber(gap, 5 m) + Huber(closing, 2 m/s), each the mean over its own labelled frames, at every window end frame whose 5 frames are real. Labels (training only, privileged, from the label join): presence = `lead_valid`; gap inside 0 to 60 m; closing speed = least-squares slope of the gap over the trailing run of at most 5 labelled frames, a step above 3.0 m breaks the run, at least 3 frames (positive = the gap opens). The head is not read by the planner and not used at inference; the inputs of the model at inference are unchanged. Definitions, decisions and pins: `spatial/leadsup/LEADSUP_DEFS.md`, `PINS.txt`.

Gate: no arm trains without the dry gate PASS of its chain (declared against built levers, a real-data smoke in which `lead_sup` and its three parts are finite and every trainable group receives a gradient, pins by sha256). Pre-launch evidence (`spatial/leadsup/checks/`, CPU): flag off = the base tree bit for bit on argv G, P and PE (run before the last edit, which touches only the synthetic selfcheck path with the flag on; the final tree passes the 103 tests); the term alone reaches the head, the state projection and the adapter and reads exactly 0.0 on every other group; with detach on nothing outside the head is non-zero; a fully masked batch reads exactly 0.0; three mutations go red.

## 3. Reads

### 3a. Lead read, T0 (the 429 scored windows of 48 eval episodes of cards ADD20, ADD21, ADD23; estimator: episode-cluster bootstrap B 2,000, rng 0)

The trained head of the arm is applied to the level-1 states of the arm at the frames S0-4 to S0 of each scored window (no frame after S0).

* M1 closing speed: R2 and RMSE of the head output against the label closing speed at S0.
* M2 gap: R2 and RMSE of the head output against the label gap at S0, beside the one-frame probe of the same arm.
* M3 slot: rule R3 of card ADD20 with the gap and closing speed of the head; skill over the copy of the same head (the head gap at S0 as the prediction for S0 + 2.0 s); level R2 beside the banked rollout read.
* M4 presence: AUROC of the presence output over all 1,948 windows at S0. Reported, names nothing.
* The same four numbers for the head of each arm applied to the train probe clips (held-in for the model and the head): the generalisation gap. Reported, names nothing.

### 3b. Cost read, T1-kin (141 eval episodes, 2,399 windows, the chain dumps of `v8f_t1_eval.py`, which stays pinned)

`spatial/q3/q3_read.py read --same-sign`, pairs LS:G, LSH:G, LSD:G on seeds 0 and 1, spread arm G with seeds 0 to 5, the rule of card ADD11: an effect needs the same sign in both seeds, a two-level interval (seeds, then episodes) that excludes 0, and a size outside the six-seed spread of G for the cell. Primary cell: ADE over 6 s on all windows. The four families are reported for every pair: LONGITUDINAL (speed error, along-track error; headway and time gap are not in the dumps and are stated as not measured), LATERAL (cross-track, heading, curvature and yaw-rate error), TACTICAL (mode error rates, tactical-band ADE), STRATEGIC PAUSED.

## 4. Controls (all must pass before any outcome is named)

* k1 label path: the label gap and closing speed of the scored windows through the rule path reproduce 0.750 within 1e-6.
* k2 head forward: the head applied on the dev box to pulled states equals `lead_dump.py` run on the pod on the same windows within 1e-4 (m and m/s) for at least 64 windows per arm.
* k3 shuffled states (deliberate-regression arm): with the states permuted across windows, closing-speed R2 and rule R3 skill are at most +0.02.
* k4 no future input: replacing every state after S0 by noise changes no output.
* k5 trained, not only built: in the train log of every arm `lead_sup_n_gap` is above 0 at every logged step, and the mean of `lead_sup` over the last 1,000 steps is below its mean over the first 1,000 steps. An arm that fails k5 is not read.
* k6 checkpoint identity: the sha256 of the checkpoint the head was taken from equals the one the T1-kin dump of the arm used.
* k7 the reader controls of `q3_read.py` (same windows, self is zero, swap is minus, planted regression read back) for the cost read.

## 5. Outcomes (both committed), per arm type, both seeds needed for a name

* Closing speed: `SUP-READS-CLOSING` if M1 R2 is above 0 with an interval that excludes 0 at both seeds; else `SUP-CLOSING-FLAT`.
* Slot: `SUP-SLOT-IMAGINES` if the R3 skill over the copy of the same head is above 0 with an interval that excludes 0 at both seeds (in addition `SUP-SLOT-BEATS-ROLLOUT` if its level R2 minus the banked rollout read is above 0 with an interval that excludes 0 at both seeds); else `SUP-SLOT-FLAT`.
* Gradient question (LS against LSD, and LSH against LSD, same seed, paired over episodes): `SUP-GRADIENT-HELPS` if the difference of the closing-speed R2 is positive in both seeds, each paired interval excludes 0, and the mean difference is larger than the run-to-run floor, taken as the absolute difference between the two seeds of LSD; `SUP-GRADIENT-HURTS` with the signs reversed; else `SUP-GRADIENT-EQUAL`.
* Cost: `SUP-NO-COST` if no cell of the cost read is WORSE under the rule; `SUP-COSTS` with the family named if one is; `SUP-IMPROVES` if the primary cell is better under the rule.
* Reading with the earlier cards: flat in cards ADD21 and ADD23 and `SUP-READS-CLOSING` at LSD = the state of G held the motion and the earlier heads were the limit; flat at LSD and reading at LS or LSH = the state holds it only when asked during training; flat in all three = the pooled 4 by 10 grid of frozen camera tokens does not carry the motion of the lead at this precision.

## 6. Committed next steps

* `SUP-SLOT-IMAGINES` with `SUP-NO-COST` (any arm type): stage 1 of card ADD20, own card: the slot (head outputs and the fixed kinematic transition) becomes a state the tactical level and the safety check read, two seeds, read at T0 with this reader and at T1-kin with the four families, then planning by imagination on the slot. If the name holds at LSD, stage 1 starts from the banked G checkpoints with a head fitted on the detached state, with no retraining of the model.
* `SUP-READS-CLOSING` with `SUP-SLOT-FLAT`: the head is given the gap 2.0 s ahead as a direct target beside the rule, own card.
* `SUP-CLOSING-FLAT` in all three arm types: the cue is the limit. The token read before pooling of card ADD23 (zero training), then an arm with a finer grid or a crop of the lead region, each its own card. A change of the frozen encoder is a PI decision and is named as a blocker, not run.
* `SUP-COSTS`: the weight is lowered or the detached head becomes the default; reported as written.
* A dose response (LSH above LS on M1 beyond the floor) with no cost: the weight is raised once more in the next card.

## 7. Cost and limits

* Six arms of 19,164 steps, three at a time on the pod (about 6 to 7 GPU-hours on the NVIDIA A40 the pod reports, about 2.5 hours of wall time), after the arms of card ADD22 are done; the container memory is read before the launch and 20 GB stay free.
* Two training runs per arm type: a name from two seeds is stated with that limit, and the interval of the bootstrap answers the question about episodes, not about training runs; the gradient question carries its own floor from the two LSD seeds.
* One corpus, one evaluation set; 429 scored windows of 48 episodes for the lead read. The labels come from the offline obstacle tracks through the label join; presence is the validity flag of the join, not a range rule.
* A subset of the label rows (the closing speed needs three consecutive labelled frames) trains the closing output: about 25 % of rows.
* The weight values 1 and 10 are fixed here and are not tuned on any read of this card.
