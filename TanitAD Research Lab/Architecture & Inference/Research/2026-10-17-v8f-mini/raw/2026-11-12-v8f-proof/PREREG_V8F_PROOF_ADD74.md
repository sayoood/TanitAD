# PREREG_V8F_PROOF_ADD74 (HANDSRC60): which input of the state hand-over carries the state part of the cost of age at 12 s of own driving

Frozen by the coordinator fork while the rollouts of card ADD73 are still running and before any dump of card ADD73 or of this card is opened (time: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). Tier of every number of this card: T1-kin/hb0.2 (kinematic rollout on the model's own actions, heading bound 0.2 1/m, executor of record `v8f_t1_eval_hb.py`), two model seeds (UE_s0, UE_s1), frames logged. None of it is a statement about driving on a road.

## 1. Known and not known

Known (MEASURED, row 131): on the 284 windows of the P = 60 grid the state part of the track cost of age under readout K is 8.58 m at 6 s and 9.81 m at 12 s of own driving (ratio 1.14, named PLATEAU), while the carried pose grows from 9.60 m to 24.99 m. Card ADD73 asks which input group of the state hand-over at the first scored tick carries the state part at 6 s (speed group, path group, or all of the logged state); its values are not known when this card is frozen.

Not known: which group carries the state part at 12 s, where it has stopped growing, and whether the shares of the groups are those of 6 s.

## 2. Runs

P = 60 grid of card ADD70 (284 windows), 2.0 s cycle, readout K (`--readout causal_g --acc-gain 0 --path-exec`), form R0 (no rule), `--kappa-max 0.2`, `--preroll-steps 60`, model seeds 0 and 1, wrapper `integ10k/v8f_t1_eval_reader_rule_rzgfhqs.py` (sha256 prefix cd4b043dcf8b219a, the wrapper of card ADD73, pod check of note N1 of that card):

* `I10R0KP60LgHb20` with `--t0-state logged`, `I10R0KP60SlHb20` with `--t0-state speed_logged`, `I10R0KP60PlHb20` with `--t0-state path_logged`.
* Banked: `I10R0KP60Hb20` (own), `I10R0KHb20` (fresh, cut to the 284 windows), `I10R0Hb20` (spread run, cut), `I10R0` (control of the strategic read).

Queue `integ10k/pod_handsrc60_queue.sh` (written by `integ10k/make_handsrc60_queue.py` from the queue of card ADD73). It starts only when the read of card ADD73 exists with every control of its part 1 passed: c0, c1 and c2 of that read are the check of the wrapper on the full P = 30 grid.

## 3. Readings and names

The readings, the windows, the name rule and the bar of one half are those of card ADD73 section 3 under its note N2, on the runs of this card, reader `integ10k/read_handsrc_decomp.py --p 60` (one file for both cards; its hash is logged in `TIMES_NOTE.md` before it reads any data).

* Named, for M in LOGGED, SPEED-LOGGED, PATH-LOGGED and the cells TRACK (`TACTICAL:ade_tac_m`), ALONG (`LONGITUDINAL:along_mae_m`), SPEED (`LONGITUDINAL:speed_mae_mps`), HEADING (`LATERAL:heading_mae_deg`): `HANDSRC60-<M>-<CELL>-CARRIES|PART|WORSE|NOT-SEPARATED|NO-STATE-COST`.
* Reported, no bar: cross-track, yaw-rate and the decision error rates per mode; the removed share per mode; the first replan segment; the removed share at 12 s beside that of card ADD73 at 6 s (the two are read on different window sets, 284 and 1,271, and the row says so).
* The four families for the pairs mode minus own at 12 s by part 2 of the reader (note N2 of card ADD73): `HANDSRC60-<M>-VS-OWN-<FAMILY>-BETTER|WORSE|MIXED|NOT-SEPARATED`, `HANDSRC60-<M>-VS-OWN-U-LOWER|HIGHER|NOT-SEPARATED`; if a control of part 2 does not pass, the row says per family what is missing.

## 4. Controls

As card ADD73 under its notes N1 and N2, on the 284 windows: c0 (under `logged` the first replan segment of the aged track equals the moved fresh track, 1e-3 m, both seeds), c1 (offsets of every mode equal those of the own run, exactly 0), c2 (pre-roll logs name the mode and the groups), b1 to b3, b5, b6 and the identity b4 per run, the planted start offset of note N4 of card ADD70, h1 (planted name rule), q1 (identities of the cell blocks, 1e-9).
Bank control of the own split: POSE, STATE and TOTAL of `I10R0KP60Hb20` read here equal the parts at P60 of the banked read of card ADD72 Amendment 1 (`READ_AGEDOSE_DECOMP.json`) on the differences of the means to 1e-9.
Part 2: the engine controls of the grid read, stage A and B of the strategic read, and a1 against the pair AGE-K@P60 of the banked re-read of card ADD70 (1e-9).
A control of part 1 that does not pass: no name and no value of part 1. A control of part 2 that does not pass: no family name.

## 5. Consequences, fixed before the runs

* A mode that is CARRIES on TRACK at 6 s (card ADD73) and at 12 s: that input group is named in the candidate specification as the state input a training arm on own rolled history has to cover at both ages, and the other session is told (its card ROLLHIST).
* The group named at 12 s differs from that at 6 s, or a mode is CARRIES at one age and not at the other: reported per age as read; the specification states the group per age and draws no single statement.
* NOT-SEPARATED on 284 windows is weaker than on the 1,271 windows of card ADD73 (wider intervals); the row says so and does not read it as "no effect".
* Nothing in this card is a deployable readout: the logged state is not available to a car that drives. No convention, readout or rule is changed by it.

## 6. Limits

T1-kin with logged frames; two model seeds; one pre-roll per window; the P = 60 grid holds 284 windows of the eval clips with 12 s of history; the fresh and spread runs are cut to those windows.
