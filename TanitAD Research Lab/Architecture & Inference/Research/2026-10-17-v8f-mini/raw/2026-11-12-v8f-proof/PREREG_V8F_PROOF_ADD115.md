# PREREG_V8F_PROOF_ADD115 — PRESATTRC10K: the two-by-two of presence and gap at the convention of the programme's reports

Frozen before any tool of this card is written, before any value of it exists, and before any value of card ADD114 is opened (its rollouts are running; only markers and timing lines of it have been read). Tier **T1-kin/hb0.2** (kinematic rollout on the model's own actions, executor of record `v8f_t1_eval_hb.py` with the heading bound 0.2 1/m). Four rollouts of existing forms on the two existing 10k models `UE_s0`, `UE_s1`; no training, no fit, no encode.

## 1. Why this card

Row 179 (card ADD113) reads the candidate form of the lead rule with an unsafe rate above the label form, and the largest such difference is at the convention (no pre-roll, pinned readout, pinned grid, 642 windows with a lead per model seed): 0.0569 against 0.0320, +0.0249 [+0.0072, +0.0464]. Card ADD114 splits that difference into presence and gap at readout K with no pre-roll and at a pre-roll of 30 steps (6 s). The convention is the setting of record of decision D16-A, and it has neither of the two mixed cells. This card runs them, so that the reading of card ADD114 stands or falls at the setting the programme reports under, and so that the lever that follows (section 6) has its bar at that setting.

Stated guess before data: as in card ADD114, the presence input carries the unsafe rate.

## 2. Cells and runs

| setting | presence label, gap label | presence camera, gap label | presence label, gap camera | presence camera, gap camera |
|---|---|---|---|---|
| CONV (pinned readout, no pre-roll, pinned grid) | `I10RLcHb20` (banked) | **new `I10RQ50Hb20aLA`** | **new `I10RLcHb20pLE`** | `I10RQ50Hb20gGEc` (card ADD113) |

* LA at CONV: the reference run RR of CONV (`I10RQ50Hb20`) with `--vis-gap-file` = the banked gap file `VGAPA_LA.npz` of card ADD110 and `--vis-closing`.
* PL at CONV: the label form RLc of CONV with `--rule-join` = the planted join `JLE.npz` that the queue of card ADD114 built and admitted (control e0 of that card: pass, four clauses true; 3,841 label rows carry the camera gap, 145 keep the label gap, none above 60 m). The file is used as it is and is not built again; the queue admits it by its record (hash of the join, of the base file, of the evaluation join).
* Wrapper `v8f_t1_eval_reader_rule_rzgfhqetjv.py` (28bc31f21ecb4ccb), unchanged. The rule is called every 10 steps as in every run of CONV.

Cells with a label input are attribution arms and never candidates.

## 3. Measures and estimator

On the 642 windows with a lead per model seed: contact C, unsafe U, falling behind by more than 10 m F for the four cells and RR; levels pooled and per model seed with window counts. Pairs, paired on the same windows, two-level bootstrap over model seeds and evaluation episodes, 2,000 draws:

* presence contrasts: LA − RLc (at the label gap) and GEc − PL (at the camera gap);
* gap contrasts: PL − RLc (at label presence) and GEc − LA (at camera presence);
* GEc − RLc (the whole difference; identity with row 179).

Four families as in card ADD113: LONGITUDINAL (the measures above and the engine read of each pair), LATERAL and TACTICAL (engine family reads of each pair), STRATEGIC (the strategic grid at T1-kin for each pair, position error at the marks of 30, 50 and 80 m). The interval answers what another draw of episodes and of these two seeds would say, not what another training run would say.

## 4. Names

For measure M in {UNSAFE, CONTACT, FALLING-BEHIND} and each of the four contrasts: `PRESATTRC10K-CONV-<contrast>-RAISES-<M>` / `-LOWERS-<M>` / `-<M>-NOT-SEPARATED`, the contrast written `PRESENCE-AT-LABEL-GAP`, `PRESENCE-AT-CAMERA-GAP`, `GAP-AT-LABEL-PRESENCE`, `GAP-AT-CAMERA-PRESENCE` (camera minus label in each).

One name on the unsafe rate, by the rule of card ADD114:

* `PRESATTRC10K-CONV-PRESENCE-CARRIES-UNSAFE` — at least one presence contrast raises the unsafe rate and no gap contrast does;
* `PRESATTRC10K-CONV-GAP-CARRIES-UNSAFE` — at least one gap contrast raises it and no presence contrast does;
* `PRESATTRC10K-CONV-BOTH-CARRY-UNSAFE` — at least one of each kind raises it;
* `PRESATTRC10K-CONV-UNSAFE-NOT-ATTRIBUTED` — none raises it.

One name across settings, given only when card ADD114 has passed its controls and has its name at K0:

* `PRESATTRC10K-ATTRIBUTION-AS-AT-K0` — the name on the unsafe rate at CONV and the one at K0 are of the same kind, and that kind is not NOT-ATTRIBUTED;
* `PRESATTRC10K-ATTRIBUTION-NOT-AS-AT-K0` — both settings are attributed, to different kinds;
* `PRESATTRC10K-ONE-SETTING-NOT-ATTRIBUTED` — exactly one of the two is NOT-ATTRIBUTED;
* `PRESATTRC10K-NEITHER-SETTING-ATTRIBUTED` — both are NOT-ATTRIBUTED.

## 5. Controls (all have to pass before any name or value is opened)

* **f0 planted join** — the join the PL runs received is the file of card ADD114: its hash equals the one in `JLE_RECORD.json`, the record has pass_e0 true, and the queue log has the admission line.
* **f1 wrapper** — the wrapper has the hash named above and its banked check files are in place.
* **f2 arguments** — for each new run, the first line of its run log against that of its reference run (RR of CONV for LA; RLc of CONV for PL) under the rules of note N1 of card ADD113.
* **f3 identity** — the reader reproduces within 1e-4 the banked levels at CONV of GEc, RR and RLc (row 179: 0.0016 / 0.0569 / 0.2329, 0.0280 / 0.0783 / 0.2593, 0.0016 / 0.0320 / 0.2243).
* **f4 tests** — the queue kit's test passes in the lead's run before the queue start; the reader kit's test passes in the lead's run before the controls stage; outputs outside the kit folders.
* **f5 engine** — the engine's own controls of the grid read and of the strategic grid pass.
* **f6 counts** — every new run has its done marker and 141 dump files.
* **f7 rule logs** — LA: the rule log agrees with the gap file `VGAPA_LA.npz` in the way control b2 of card ADD111 checks it. PL: on every logged step with a lead the gap the rule received equals the gap of `JLE.npz` at that clip and frame within 1e-3 m, and the logged steps with a lead are the steps where the lead of the planted join is valid by the rule's own test for a join (marking 1, gap finite and from 0 to 60 m). A run of CONV writes one rule log (the rule is called every 10 steps); the loader takes the files that exist, as the reader of card ADD113 does.

This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

Order of work, stated before data: the queue is generated from the queue of card ADD114 by asserted substitutions, tested and dry-run by the lead, and placed on the pod to start when the queue of card ADD114 has written its done marker (a pod-side wait; a stop marker of card ADD114 starts nothing). The reader is the reader kit of card ADD114 with the setting CONV, built and tested while the rollouts run. No value of a new run is opened before the reader's tests pass and the controls are logged.

## 6. Committed readings

* **PRESENCE-CARRIES-UNSAFE or BOTH-CARRY-UNSAFE at CONV** → the next lever is the presence input (the presence head of card ADD114's reading: fitted on the existing tokens of the train clips, own card, T0 read first), and its bar is set at the convention.
* **GAP-CARRIES-UNSAFE at CONV** → the gap input needs the reader's own input, a new encode and a PI decision; stated as a limit in the spec for the convention.
* **UNSAFE-NOT-ATTRIBUTED at CONV** → with NEITHER-SETTING-ATTRIBUTED: two model seeds do not resolve the two-by-two anywhere; more model seeds of the 10k line go to the PI with their cost, and no further attribution card is written on two seeds. With ONE-SETTING-NOT-ATTRIBUTED: the lever of the attributed setting stands, and the other setting is carried as not resolved.
* **ATTRIBUTION-NOT-AS-AT-K0** → reported as written; the lever card then has to state for which readout it is built, and both settings are carried in its bar.

## 7. Cost and limits

ESTIMATED 0.7 pod-h for four rollouts on the pinned grid, one after another; stop rule 2 pod-h from the queue start. Limits: T1-kin only, no re-perception; the lead moves as logged whatever the ego does; two model seeds of one line, and a separated interval from two model seeds is necessary, not sufficient; the unsafe difference in question is about 28 windows of 1,124 scored; the label marking and the label gap are references and not ground truth; one setting.
