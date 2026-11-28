# PREREG_V8F_PROOF_ADD113 — CANDSET10K: the camera-fed candidate form at the convention and at two more rollout ages

Frozen before any tool of this card is written and before any value of it exists. Tier **T1-kin/hb0.2** (kinematic rollout on the model's own actions, executor of record `v8f_t1_eval_hb.py` with the heading bound 0.2 1/m). Six rollouts of an existing form on the two existing 10k models `UE_s0`, `UE_s1`; no training, no fit, no encode.

## 1. Why this card

Decision D16-A of the PI asks for the integrated 10k candidate with the lead rule reported under the convention (no pre-roll, pinned readout) and under readout K. The candidate's lead rule with no label at inference (form VD5c of card ADD102, fed by the mean gap of the three large fits of card ADD106, run name suffix `gGEc`) is banked at one setting only: readout K, rule every 5 steps, pre-roll of 30 steps (6 s) (rows 172, 176, 177: contact 0.0062 against 0.0954 for the reader-fed form RR). The reader-fed form RR, the label form RLc and the form with no rule R0 are banked at three more settings. This card runs the candidate form there, so that its contact cut is read at every setting of the programme's reports and not at one.

Stated guess before data: the cut holds at readout K with no pre-roll and at the pre-roll of 60 steps (12 s); at the convention the contact rate of RR is itself small and the setting may have little contact to cut.

## 2. Settings and runs

| setting | grid | arguments (as the banked reference runs of that setting) | new run (both model seeds) | banked references |
|---|---|---|---|---|
| CONV | pinned grid | no pre-roll, `--readout pinned`, `--kappa-max 0.2` | `I10RQ50Hb20gGEc` | `I10RQ50Hb20` (RR), `I10RLcHb20` (RLc), `I10R0Hb20` (R0) |
| K0 | pinned grid | no pre-roll, readout K (`--readout causal_g --acc-gain 0 --path-exec`), `--rule-every 5`, `--kappa-max 0.2` | `I10RQ50KF5Hb20gGEc` | `I10RQ50KF5Hb20`, `I10RLcKF5Hb20`, `I10R0KHb20` |
| K60 | `grid_p60.json` (284 windows) | pre-roll of 60 steps (12 s), readout K, `--rule-every 5`, `--kappa-max 0.2` | `I10RQ50KF5P60Hb20gGEc` | `I10RQ50KF5P60Hb20`, `I10RLcKF5P60Hb20`, `I10R0KP60Hb20` |
| K30 (banked, read again as identity) | `grid_p30.json` (1,271 windows) | pre-roll of 30 steps (6 s), readout K, `--rule-every 5` | none | `I10RQ50KF5P30Hb20gGEc`, `I10RQ50KF5P30Hb20`, `I10RLcKF5P30Hb20` |

Each new run is the reference run RR of its setting (same reader checkpoint of the model seed, same q-map file, same arguments) with two additions and nothing else: `--vis-gap-file /workspace/v8proof/cropscale/gapfiles/VGAPLE.npz` (sha256 begins 55de2f7c) and `--vis-closing`, through the unchanged wrapper `v8f_t1_eval_reader_rule_rzgfhqetjv.py` (28bc31f21ecb4ccb), exactly as the banked K30 run was called. No label enters a new run at inference.

## 3. Measures and estimator

Per setting, on the windows with a lead of that setting's grid (the engine's own definition, as in rows 172 to 177): contact C, unsafe U, falling behind by more than 10 m F; levels pooled over both model seeds and per seed, with the number of windows printed. Pairs GEc − RR, GEc − RLc and RR − RLc per setting: paired on the same windows, two-level bootstrap over model seeds and evaluation episodes, 2,000 draws. The interval answers what another draw of episodes and of these two seeds would say, not what another training run would say.

Four families per setting. LONGITUDINAL: the measures above and the engine's longitudinal read. LATERAL and TACTICAL: the engine's pair read of GEc against RR and against RLc. STRATEGIC: the T1-kin read of the strategic grid (position error at the marks of 30, 50 and 80 m), GEc against RR and against RLc, where the engine reads that setting; where it does not (stated per setting with the reason and n), the family is reported as not read at that setting, not as passed.

## 4. Names

Per setting S in {CONV, K0, K60}:

* contact against RR: `CANDSET10K-<S>-NO-CONTACT-TO-CUT` if the pooled contact rate of RR at S is below 0.01; else `CANDSET10K-<S>-CONTACT-CUT-HOLDS` if the interval of C(GEc) − C(RR) lies below 0, `-CONTACT-RAISED` if it lies above 0, else `-CONTACT-NOT-SEPARATED`;
* unsafe against RR: `CANDSET10K-<S>-UNSAFE-CUT-HOLDS` / `-UNSAFE-RAISED` / `-UNSAFE-NOT-SEPARATED` by the interval of U(GEc) − U(RR);
* price against the label form: `CANDSET10K-<S>-FALLS-BEHIND-MORE` if the interval of F(GEc) − F(RLc) lies above 0, else `-NO-SEPARATED-PRICE`;
* against the label form on contact: `CANDSET10K-<S>-CONTACT-ABOVE-LABEL-FORM` if the interval of C(GEc) − C(RLc) lies above 0, else `-CONTACT-AT-LABEL-FORM`.

One name over the settings: `CANDSET10K-CUT-HOLDS-AT-EVERY-SETTING-WITH-CONTACT` if every setting of {CONV, K0, K60} that is not NO-CONTACT-TO-CUT reads CONTACT-CUT-HOLDS and at least one such setting exists; `CANDSET10K-CUT-RAISED-AT-A-SETTING` if any reads CONTACT-RAISED; else `CANDSET10K-CUT-NOT-AT-EVERY-SETTING`.

## 5. Controls (all have to pass before any name or value is opened)

* **d0 gap file** — the gap file on disk has the sha256 of card ADD111's record (`base.sha256`), and the rule log of each new run agrees with that file in the way control b2 of card ADD111 checks it.
* **d1 wrapper** — the wrapper file has the hash named above and its banked check files are in place.
* **d2 arguments** — for each new run, every field that the first line of its run log and that of the reference run RR of its setting both carry (form, replan steps, rule every, readout, kappa, pre-roll, nav, grid, model seed, reader seed, checkpoint hash) is equal, with readout, kappa, pre-roll, grid and model seed present in both; the wrapper hash is left out of this comparison (the reference runs were made with the wrapper of their own card) and is checked by d1. The rule log of each new run exists and shows the rule acting on steps with a lead (a share above 0). The q-map file of each new run is the one of its reference run: read by the lead from the printed commands of the dry run and logged before the queue start.
* **d3 identity** — the reader reproduces the banked levels of GEc, RR and RLc at K30 (rows 172 and 177) within 1e-4.
* **d4 tests** — the test file of the reader kit passes in the lead's run before the controls stage of the read (a line with a count of passed tests and no failed test; the lead's output is written outside the kit folder). The queue kit's test passes in the lead's run before the queue start.
* **d5 engine** — the engine's own controls of each grid read and of the strategic grid pass.
* **d6 counts** — every new run has its done marker and as many dump files as its grid has clips.

This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

Order of work, stated before data: the queue kit is built, tested and dry-run by the lead, and the rollouts then start; the reader kit is built while they run. No value of a new run is opened before the reader kit's tests pass and the controls are logged. The marker lines of the queue keep the format of the queue of card ADD111, which the reader's driver of that card parses.

## 6. Committed readings

* **CUT-HOLDS-AT-EVERY-SETTING-WITH-CONTACT** → the candidate spec gets the levels of the candidate form at each setting as its table under decision D16-A; the form is unchanged.
* **CUT-NOT-AT-EVERY-SETTING** → the spec states at which setting the cut is not separated, with the number of windows with a lead there; if that setting is K60 (284 windows on the grid) the next step is the count of windows a separated read needs, and no change of the form.
* **CUT-RAISED-AT-A-SETTING** → the candidate form is not proposed at that setting; the rule logs of that setting are read (which input, gap or slope, acts) in a card of its own before anything else on the gap input.
* FALLS-BEHIND-MORE at a setting is reported as the price of the gap input there (rows 172 to 177 at K30) and changes nothing by itself.

## 7. Cost and limits

ESTIMATED 1.0 pod-h for six rollouts one after another; stop rule 3 pod-h from the queue start. Limits: T1-kin only, no re-perception; the lead moves as logged whatever the ego does; two model seeds of one line, and a separated interval from two model seeds is necessary, not sufficient; the K60 grid has 284 windows; the gap file is the mean of three fits of one head recipe.
