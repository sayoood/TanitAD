# PREREG_V8F_PROOF_ADD114 — PRESATTR10K: does the unsafe rate of the candidate form sit in the presence input or in the gap value?

Frozen before any tool of this card is written and before any value of it exists. Tier **T1-kin/hb0.2** (kinematic rollout on the model's own actions, executor of record `v8f_t1_eval_hb.py` with the heading bound 0.2 1/m). Six rollouts of existing forms on the two existing 10k models `UE_s0`, `UE_s1`; no training, no fit, no encode.

## 1. Why this card

Card ADD112 (row 178) reads `GAPMAP-T0-MAP-DOES-NOT-CUT-SLOW-PART`, and its committed reading sends the fork from the gap value to the presence input. Card ADD113 (row 179) reads the candidate form (reader's own presence, camera gap, no label at inference) with contact at the level of the label form at every setting, and with an unsafe rate above the label form: at readout K with no pre-roll 0.0498 against 0.0365 (+0.0133 [+0.0036, +0.0250]), at the convention 0.0569 against 0.0320 (+0.0249 [+0.0072, +0.0464]). Rows 176 and 177 have one more cell at the pre-roll of 30 steps (6 s): with the label gap on every label row and the reader's own presence the unsafe rate stays at 0.0839, the level of the candidate, against 0.0714 for the label form. The fourth cell of that two-by-two, label presence with the camera gap, does not exist with the mean gap of the three large fits, and at readout K with no pre-roll (642 windows with a lead, twice the number at 30 steps) two cells are missing.

Stated guess before data: the presence input carries the unsafe rate; the camera gap with label presence is at the level of the label form.

## 2. Cells and runs

Two inputs of the lead rule, each from the label or from the camera: presence (label: the label's marking of frames with a lead; camera: the present reader's presence output) and gap (label gap; camera: base gap file `VGAPLE.npz` of card ADD106, sha256 begins 55de2f7c). The closing input is the slope of the gap the rule receives in every cell.

| setting | presence label, gap label | presence camera, gap label | presence label, gap camera | presence camera, gap camera |
|---|---|---|---|---|
| K0 (readout K, no pre-roll, pinned grid) | `I10RLcKF5Hb20` (banked) | **new `I10RQ50KF5Hb20aLA`** | **new `I10RLcKF5Hb20pLE`** | `I10RQ50KF5Hb20gGEc` (card ADD113) |
| K30 (readout K, pre-roll of 30 steps (6 s)) | `I10RLcKF5P30Hb20` (banked) | `I10RQ50KF5P30Hb20aLA` (card ADD110) | **new `I10RLcKF5P30Hb20pLE`** | `I10RQ50KF5P30Hb20gGEc` (banked) |

* LA at K0: the reference run RR of K0 with `--vis-gap-file` = the banked gap file `VGAPA_LA.npz` of card ADD110 and `--vis-closing`, exactly as the banked LA run at K30 was called.
* PL (suffix `pLE`): the label form RLc of the setting with `--rule-join` = a planted join `JLE.npz`: the evaluation join with `lead_gap` replaced by the base gap of `VGAPLE.npz` on every row where the label gap is valid and the base file has a finite output, the label gap kept on the label rows with no base output, every other array unchanged. The rule reads its lead from the planted join; the scoring keeps the evaluation join (as cards ADD94 and ADD101).
* Wrapper `v8f_t1_eval_reader_rule_rzgfhqetjv.py` (28bc31f21ecb4ccb), unchanged, for all six new runs.

Cells with a label input are attribution arms and never candidates.

## 3. Measures and estimator

Per setting on its windows with a lead: contact C, unsafe U, falling behind by more than 10 m F for the four cells and RR; levels pooled and per model seed with window counts. Pairs per setting, paired on the same windows, two-level bootstrap over model seeds and evaluation episodes, 2,000 draws:

* presence contrasts: LA − RLc (at the label gap) and GEc − PL (at the camera gap);
* gap contrasts: PL − RLc (at label presence) and GEc − LA (at camera presence);
* GEc − RLc (the whole difference).

Four families per setting as in card ADD113 (engine family reads of each pair; the strategic grid at T1-kin for each pair). The interval answers what another draw of episodes and of these two seeds would say, not what another training run would say.

## 4. Names

Per setting S in {K0, K30} and measure M in {UNSAFE, CONTACT, FALLING-BEHIND}, for each of the four contrasts: `PRESATTR10K-<S>-<contrast>-RAISES-<M>` / `-LOWERS-<M>` / `-<M>-NOT-SEPARATED`, with the contrast written `PRESENCE-AT-LABEL-GAP`, `PRESENCE-AT-CAMERA-GAP`, `GAP-AT-LABEL-PRESENCE`, `GAP-AT-CAMERA-PRESENCE` (camera minus label in each).

One name per setting on the unsafe rate:

* `PRESATTR10K-<S>-PRESENCE-CARRIES-UNSAFE` — at least one presence contrast raises the unsafe rate and no gap contrast does;
* `PRESATTR10K-<S>-GAP-CARRIES-UNSAFE` — at least one gap contrast raises it and no presence contrast does;
* `PRESATTR10K-<S>-BOTH-CARRY-UNSAFE` — at least one of each kind raises it;
* `PRESATTR10K-<S>-UNSAFE-NOT-ATTRIBUTED` — none raises it.

## 5. Controls (all have to pass before any name or value is opened)

* **e0 planted join** — `JLE.npz` differs from the evaluation join in `lead_gap` only, and only on the rows named in section 2, where it equals the base gap within 1e-4 m; a second routine (a plain loop row by row) gives the same array; the record holds the hashes of the join, the base file and the evaluation join.
* **e1 wrapper** — the wrapper has the hash named above and its banked check files are in place.
* **e2 arguments** — for each new run, the first line of its run log against that of its reference run (RR of K0 for LA; RLc of the setting for PL) under the rules of note N1 of card ADD113.
* **e3 identity** — the reader reproduces within 1e-4 the banked levels at K30 of GEc, RR, RLc and LA (rows 172, 176) and at K0 of GEc, RR and RLc (row 179).
* **e4 tests** — the queue kit's test passes in the lead's run before the queue start; the reader kit's test passes in the lead's run before the controls stage; outputs outside the kit folders.
* **e5 engine** — the engine's own controls of each grid read and of the strategic grid pass.
* **e6 counts** — every new run has its done marker and 141 dump files.
* **e7 rule logs** — LA at K0: the rule log agrees with the gap file `VGAPA_LA.npz` in the way control b2 of card ADD111 checks it. PL runs: on every logged step with a lead the gap the rule received equals the gap of `JLE.npz` at that clip and frame within 1e-3 m, and the logged steps with a lead are the steps where the lead of the planted join is valid by the rule's own test for a join (marking 1, gap finite and from 0 to 60 m). The number of rows of `JLE.npz` whose camera gap lies above 60 m (a lead for the label, no lead for the rule in this cell) is printed with no name.

This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

Order of work, stated before data: the queue kit (with the join tool) is built, tested and dry-run by the lead, and the rollouts then start; the reader kit is built while they run. No value of a new run is opened before the reader kit's tests pass and the controls are logged.

## 6. Committed readings

* **PRESENCE-CARRIES-UNSAFE** at K0 (the setting with more windows; K30 is reported beside it) → the next lever for the candidate is the presence input: a presence head fitted on the existing tokens of the train clips (own card with its own stop rule), read against the present reader's presence on the evaluation frames at T0 first.
* **GAP-CARRIES-UNSAFE** → with rows 177 and 178 (smoothing and a map of the gap value do not help) the gap input needs the reader's own input, a new encode and a PI decision; the candidate's unsafe rate is carried in the spec as a stated limit.
* **BOTH-CARRY-UNSAFE** → the presence head is the next card (it needs no encode); the gap part is stated as under GAP-CARRIES.
* **UNSAFE-NOT-ATTRIBUTED** → two model seeds do not resolve the two-by-two; a finer read needs more model seeds of the 10k line, which is a PI decision, put to the PI with its cost. No further attribution card on two seeds.

## 7. Cost and limits

ESTIMATED 1.0 pod-h for six rollouts one after another (four on the pinned grid, two on the grid of 30 steps); stop rule 3 pod-h from the queue start. Limits: T1-kin only, no re-perception; the lead moves as logged whatever the ego does; two model seeds of one line, and a separated interval from two model seeds is necessary, not sufficient; the unsafe differences in question are 7 to 14 windows; the label marking and the label gap are references and not ground truth.
