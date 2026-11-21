# PREREG_V8F_PROOF_ADD75 (CLOSRC10K): which input of the lead rule carries the contact of the reader-fed form at 6 s of own driving

Frozen by the coordinator fork before any rollout of this card exists (time: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). Tier of every number of this card: T1-kin/hb0.2 (kinematic rollout on the model's own actions, heading bound 0.2 1/m, executor of record `v8f_t1_eval_hb.py`), two model seeds (UE_s0, UE_s1), frames logged. None of it is a statement about driving on a road.

## 1. Known and not known

Known (MEASURED, row 130, `integ10k/read_agerule/pod/READ_AGERULE10K.json`, 325 windows with a lead in 46 episodes of the P = 30 grid, model seed 0 / seed 1): after 6 s of own driving under readout K with the second look, the contact rate is 0.000 / 0.000 for the label-fed candidate, 0.092 / 0.098 for the reader-fed candidate and 0.194 / 0.182 with no rule. The reader-fed form is the one a car could run (tokens only); the label-fed form reads labels.

The two forms differ in two inputs of the same rule: where presence and gap come from (labels, or the token reader through the q50 map) and the closing speed (labels, or none: the reader-fed form feeds 0).

Not known: which of the two inputs carries the difference in contact, and whether the reader's own closing output (third output of the same token reader, tokens only) removes any of it.

## 2. Runs

P = 30 grid of card ADD70 (1,271 windows), pre-roll 30 steps, 2.0 s cycle, readout K (`--readout causal_g --acc-gain 0 --path-exec`), `--kappa-max 0.2`, the rule at the replan ticks (no second look), model seeds 0 and 1, reader seed = model seed, q50 map, wrapper of record `integ10k/v8f_t1_eval_reader_rule_rzgfhq.py` (sha256 prefix 92a8034cc0f7b0d2), forms that exist in it:

| run | form | presence and gap | closing speed |
|---|---|---|---|
| `I10RQ50KP30Hb20` | RR | reader, q50 map | 0 |
| `I10RQ50cKP30Hb20` | RRc | reader, q50 map | labels |
| `I10RQ50hKP30Hb20` | RRh | reader, q50 map | third output of the reader |
| `I10RLbKP30Hb20` | RLb | labels | 0 |
| `I10RLcKP30Hb20` | RLc | labels | labels |

Banked: `I10R0KP30Hb20` (no rule at age), `I10R0KHb20` (fresh), `I10R0Hb20` (spread), `I10R0` (control of the strategic read). The pre-roll of each run is driven under its own form.

No second look: the wrapper of record refuses a second look for the forms RRc and RRh. A second look for these forms is a new wrapper generation and not part of this card.

Queue `integ10k/pod_closrc10k_queue.sh` (written by `integ10k/make_closrc_queue.py` from the queue of card ADD71: the same helper functions and rollout function, with the forms RRc and RRh given the reader and the map as RR is).

## 3. Readings and names

Contact C, unsafe rate U and falling behind F on the windows with a lead by the headway reader (reader of record for pre-rolled tracks, notes N3 and N4 of card ADD70), paired episode-cluster bootstrap over the two model seeds; the family cells by the family reader. A share is LOWER / HIGHER when both seeds have that sign and the interval excludes 0, else NOT-SEPARATED (the rule of card ADD71 for U, used for C too).

* The gap to explain, pair RR minus RLc on C: `CLOSRC10K-GAP-PRESENT` when HIGHER, else `CLOSRC10K-NO-GAP-AT-TICK-LEVEL`.
* Closing speed with the reader's presence and gap, pair RRc minus RR: `CLOSRC10K-LABEL-CLOSING-C-LOWER|NOT-SEPARATED|HIGHER`.
* Source of presence and gap with closing 0, pair RLb minus RR: `CLOSRC10K-LABEL-SOURCE-C-LOWER|NOT-SEPARATED|HIGHER`.
* What a car could run, pair RRh minus RR: `CLOSRC10K-READER-CLOSING-C-LOWER|NOT-SEPARATED|HIGHER`.
* Which input carries the gap (only when GAP-PRESENT), on the windows with a lead: `CLOSRC10K-CLOSING-CARRIES` when the interval of (C_RR minus C_RRc) minus one half of (C_RR minus C_RLc) excludes 0 above; `CLOSRC10K-SOURCE-CARRIES` when the interval of (C_RR minus C_RLb) minus one half of (C_RR minus C_RLc) excludes 0 above; both may hold; when neither holds `CLOSRC10K-NEITHER-INPUT-ALONE`.
* For each of the four pairs against RR, and for RR against no rule at age: U by the same rule and the four families (LONGITUDINAL with the headway read, LATERAL, TACTICAL, STRATEGIC by the instrument of card ADD66): `CLOSRC10K-<PAIR>-U-…`, `CLOSRC10K-<PAIR>-<FAMILY>-BETTER|WORSE|MIXED|NOT-SEPARATED`.
* Reported, no bar: the levels of C, U, F per run and seed; the removed share of the gap per input.

## 4. Controls

* The engine controls of the grid read (same windows in every dump, headway controls, self pair 0, swap, planted regression) and stage A and B of the strategic read.
* a1: the pair `I10R0KP30Hb20` minus `I10R0KHb20` read in this grid read equals the pair AGE-K of the banked re-read of card ADD70 on every headway share and family cell to 1e-9.
* r1: the run text of each new run names its form, readout causal_g, heading bound 0.2, pre-roll 30, the rule at every 10 steps, return code 0, and the dump holds every clip of the grid.
* x1, the input reached the rule: from the rule logs of the runs, the closing speed fed to the rule is exactly 0 on every row of the forms RR and RLb, and differs from 0 on at least one row with a lead present for each of RRc, RRh and RLc, both seeds.
* i1: on the differences of the means, each statistic of the attribution equals its two pairs combined (1e-9).
* h1: planted arrays give CARRIES for an input that removes 0.8 of a gap and not for one that removes 0.2.
A control that does not pass: no name and no value of this card is written; the cause is logged before any repair.

## 5. Consequences, fixed before the runs

* CLOSING-CARRIES and READER-CLOSING-C-LOWER with no family WORSE and U not HIGHER for the pair RRh minus RR: the reader's closing output is a candidate addition to the reader-fed form. Next arm: the second look for that form (a new wrapper generation with its pod check) at age and on fresh windows; the result goes to the PI and the other session as a candidate change. Nothing is adopted by this card.
* CLOSING-CARRIES and READER-CLOSING-C not LOWER: the missing piece is a closing-speed reading of enough quality from tokens. Next lever: a closing reader trained on the pod cache (a probe, no model training), on a card of its own.
* SOURCE-CARRIES: the lever is the reading of presence and gap at age. Next arm: the q35 map and the presence threshold at age.
* NO-GAP-AT-TICK-LEVEL: the difference of row 130 comes with the second look; the next arm is the second look for the forms of this card (new wrapper generation).
* The forms RRc, RLb and RLc read labels: they are attribution arms and not forms a car could run.

## 6. Limits

T1-kin with logged frames; two model seeds; 325 windows with a lead in 46 episodes, so a difference of a few windows moves a rate; the rule acts at the replan ticks only; the pre-roll of each run is driven under its own form, so the pose and state at t0 differ between runs and the pairs read the whole 12 s of own driving of each form.
