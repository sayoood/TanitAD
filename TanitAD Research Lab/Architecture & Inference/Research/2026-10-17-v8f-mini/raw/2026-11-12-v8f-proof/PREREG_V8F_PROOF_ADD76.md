# PREREG_V8F_PROOF_ADD76 (MIDLOOK10K): the lead rule with a closing-speed input and a second look in the reader-fed form, at age and fresh

Written by the coordinator fork before any rollout of this card and before any value of card ADD75 was opened (its queue is still running; its read has not been started). Time of freezing: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file. No training. Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions, heading bound 0.2 1/m).

## 1. Question

The reader-fed form of the candidate (presence and gap of the lead from the token reader through the q50 map, closing speed 0) keeps a contact share at age that the label-fed form does not have (rows 129 and 130: with the second look, contact share label-fed 0.000 and 0.000, reader-fed 0.092 and 0.098, no rule 0.194 and 0.182 for the two model seeds). The second look (the rule looks again 1.0 s into the 2.0 s cycle) was so far available for forms without a closing input from the reader side only. The second-look generation of the study wrapper (`integ10k/v8f_t1_eval_reader_rule_rzgfhqc.py`, sha256 8732329b4476347d; pod check PASS in the form of record, `integ10k/CHECK_MIDLOOK_CLOSING_chk1n1.txt`, note `integ10k/MIDLOOK_CLOSING_CHECK_NOTE_N1.md`) admits it for the two reader-fed forms with a closing input:

* RRh: presence and gap from the reader, closing speed from the third output of the same reader (times 10 m/s), at the replan tick and at the later look from the tokens of that frame. This is a form a car could run: it reads tokens only.
* RRc: presence and gap from the reader, closing speed from the labels. A study form (the label closing is not available to a car); it bounds what a closing reader could give.

Does a closing input bring the reader-fed form with the second look to the contact share of the label-fed form, at age and fresh, without a cost on the four families?

## 2. Runs

Eight rollouts, model seeds 0 and 1 (line models `UE_s0`, `UE_s1`, INHERITED from the other session), readout K (`--readout causal_g --acc-gain 0 --path-exec`), `--kappa-max 0.2`, `--rule-every 5`, replan every 10 steps, q50 map of the reader seed, wrapper as above:

| run | form | setting |
|---|---|---|
| `I10RQ50hKF5P30Hb20` | RRh | at age: P = 30 grid of card ADD70 (1,271 windows), pre-roll 30 steps |
| `I10RQ50cKF5P30Hb20` | RRc | at age |
| `I10RQ50hKF5Hb20` | RRh | fresh: pinned grid, no pre-roll |
| `I10RQ50cKF5Hb20` | RRc | fresh |

Banked runs used: at age `I10RQ50KF5P30Hb20` (RR with the second look), `I10RLcKF5P30Hb20` (RLc with the second look), `I10R0KP30Hb20` (no rule); fresh `I10RQ50KF5Hb20`, `I10RLcKF5Hb20`, `I10R0KHb20`; `I10R0Hb20` (spread run of the family engine) and `I10R0`. Queue `integ10k/pod_midlook10k_queue.sh` written by `integ10k/make_midlook_queue.py ADD76 RRc,RRh` from the queue of card ADD75; it starts only if the marker of the wrapper check says PASS and the wrapper on the pod carries the checked hash. It may run beside the queue of card ADD75 (never more than three GPU jobs; each rollout waits for its slot).

## 3. Read

Reader `integ10k/read_midlook10k.py`, written while the rollouts run and before any dump of this card is opened; its hash is logged in `TIMES_NOTE.md` before the read. Engines as in card ADD75: headway instrument (unsafe rate U, contact share C, falling-behind share F), family engine (two-level bootstrap, B 2,000), strategic read at the third tick (instrument of card ADD66). A cell is LOWER or HIGHER when both model seeds have the same sign and the interval excludes 0, else NOT-SEPARATED. Pairs A minus B, each in the setting AGE and in the setting FRESH:

| pair | A | B | what it says |
|---|---|---|---|
| READER-CLOSING | RRh, second look | RR, second look | the reader's own closing output added |
| LABEL-CLOSING | RRc, second look | RR, second look | the label closing added (bound) |
| RRH-VS-LABEL-FED | RRh, second look | RLc, second look | what is left to the label-fed form |
| RRC-VS-LABEL-FED | RRc, second look | RLc, second look | what the reader's presence and gap cost when the closing is the label's |
| RRH-VS-NO-RULE | RRh, second look | no rule under readout K | the form against no rule |

Names per pair and setting: `MIDLOOK10K-<PAIR>-<SETTING>-C-<LOWER|HIGHER|NOT-SEPARATED>`, the same for U, and one name per family (LONGITUDINAL, LATERAL, TACTICAL, STRATEGIC: BETTER, WORSE, MIXED, NOT-SEPARATED by the reading rule of the family engine). The four families are reported for every pair; the headway side is reported only where the headway instrument is qualified by its own bar (200 windows with a lead and 20 episodes; the P = 30 grid has 325 in 46).

Verdict names, fixed here:

* `MIDLOOK10K-RRH-CHANGE-SUPPORTED`: pair READER-CLOSING at AGE reads C LOWER, U not HIGHER and no family WORSE; and the same pair FRESH reads C not HIGHER, U not HIGHER and no family WORSE. Otherwise `MIDLOOK10K-RRH-CHANGE-NOT-SUPPORTED`.
* `MIDLOOK10K-LABEL-CLOSING-LOWERS-CONTACT`: pair LABEL-CLOSING at AGE reads C LOWER. Otherwise `MIDLOOK10K-LABEL-CLOSING-DOES-NOT-LOWER-CONTACT`.
* `MIDLOOK10K-RRH-REACHES-LABEL-FED` when the pair RRH-VS-LABEL-FED at AGE reads C NOT-SEPARATED and U NOT-SEPARATED or LOWER; `MIDLOOK10K-RRH-GAP-TO-LABEL-FED` when its C or U reads HIGHER.

## 4. Controls (all have to pass before a name or a value is opened)

* g: the engine controls of the grid read (both settings).
* s: the strategic rows, three ticks per window.
* r1: every run text names its form, rule every 5, readout causal_g, kappa 0.2, the pre-roll of its setting, the wrapper hash 8732329b4476347d and `rc 0`; dump counts equal those of the grid.
* x1: the closing fed to the rule, from the rule log (replan ticks) and from the later-look log: not 0 on at least one row with a lead present for RRh and RRc, in both logs; for RRh the fed closing equals the logged reader closing (`c_read`) on the present rows of the later-look log (difference 0); the banked RR run feeds 0 on every row.
* m1: every new run has later-look rows, and their count per run equals the count of the banked RR run of the same setting and seed where the windows are the same (same grid, same pre-roll; the looks depend on the grid, not on the form) or the difference is reported with its cause before any value is opened.
* a1: the pair RR with the second look minus no rule at AGE, read here, equals the banked read of card ADD71 (`READ_AGERULE10K.json`) on its headway cells and family cells (1e-9) where that file holds the pair; where it does not, the levels of the two banked runs equal the banked levels (1e-9).
* i1: on the contact levels, (RRh minus RLc) equals (RRh minus RR) plus (RR minus RLc) per seed (1e-12).
* h1: a planted name-rule arm (a pair with a known sign reads the expected name; a pair with seeds of opposite sign reads NOT-SEPARATED).

A control that does not pass is logged in `TIMES_NOTE.md` before any repair; the repair is a note to this card and the re-read goes into a new folder.

## 5. Fixed consequences

* `RRH-CHANGE-SUPPORTED`: the reader-fed form with the reader's own closing output and the second look goes to the PI and the other session as a proposed change of the reader-fed candidate form (it is not adopted here); the candidate specification names it as a proposal with these values.
* `RRH-CHANGE-NOT-SUPPORTED` and `LABEL-CLOSING-LOWERS-CONTACT`: the closing input is worth having and the reader's closing output is not good enough to carry it: a closing-reader probe on the pod cache gets its own card (what it costs and whether it needs a go is named there).
* `RRH-CHANGE-NOT-SUPPORTED` and `LABEL-CLOSING-DOES-NOT-LOWER-CONTACT`: with the second look the contact of the reader-fed form is not a matter of the closing input; the next arm is on presence and gap (the q35 map or the presence threshold) at age with the second look, own card.
* Card ADD75 keeps its own names and consequences; where its consequence asks for a second look for one of these forms, this card is that arm.

## 6. Limits known in advance

Readout K is a study device, not a readout a car could run. RRc uses a label input. Two model seeds are not a replicate study; a separated interval answers another draw of episodes and is necessary, not sufficient. T1-kin with logged frames: at age the tokens show the motion of the logged car. One reader seed per model seed. The readout does not sample.
