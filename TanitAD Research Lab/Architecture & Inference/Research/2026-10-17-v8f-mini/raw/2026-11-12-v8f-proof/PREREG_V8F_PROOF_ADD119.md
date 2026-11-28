# PREREG_V8F_PROOF_ADD119 — GAPMARGINAGE10K: the margin on the camera gap at the pre-rolls of 30 and 60 steps

Frozen before any tool of this card is written and before any value of it exists. Open at this time: card ADD117 (row 183 in writing; `GAPMARGIN10K-K0-MARGIN-CLOSES-UNSAFE-AT-A-PRICE` with the closing arm A90, `GAPMARGIN10K-CONV-MARGIN-DOES-NOT-CLOSE-UNSAFE`); no value of card ADD118 is open (its fits are running). Tier **T1-kin/hb0.2** (kinematic rollout on the model's own actions, executor of record `v8f_t1_eval_hb.py` with the heading bound 0.2 1/m). Eight rollouts of the candidate form with a scaled gap file on the two existing 10k models `UE_s0`, `UE_s1`; no training, no fit, no encode.

## 1. Why this card

The committed reading of card ADD117 for a margin that closes the unsafe rate under readout K with no pre-roll: the next card reads the margin at the two aged settings, where falling behind is largest (label form 0.4908 at a pre-roll of 30 steps (6 s), 0.6301 at a pre-roll of 60 steps (12 s)). The margin of 10 % is the named one; the margin of 20 % is read beside it as the second point of the dose, because at the convention only 20 % lowered the unsafe rate.

Stated guess before data: at the pre-roll of 30 steps the margin of 10 % does not separate on the unsafe rate (280 scored windows per seed) and raises falling behind; 20 % lowers the unsafe rate and raises falling behind by more than 0.03. At the pre-roll of 60 steps (73 windows with a lead per seed) nothing separates.

## 2. Arms and runs

The candidate form GEc (reader's own presence, `--vis-gap-file`, `--vis-closing`, no label at inference) with the scaled gap files of card ADD117, unchanged and not rebuilt: `VGAPLE_A90.npz` (arm A90) and `VGAPLE_A80.npz` (arm A80), admitted by their record of card ADD117.

| setting | A90 | A80 | banked cells |
|---|---|---|---|
| K30 (readout K, pre-roll of 30 steps (6 s), grid_p30, rule every 5 steps) | **new `I10RQ50KF5P30Hb20gA90`** | **new `I10RQ50KF5P30Hb20gA80`** | GEc `I10RQ50KF5P30Hb20gGEc`, RLc `I10RLcKF5P30Hb20`, RR `I10RQ50KF5P30Hb20` |
| K60 (readout K, pre-roll of 60 steps (12 s), grid_p60, rule every 5 steps) | **new `I10RQ50KF5P60Hb20gA90`** | **new `I10RQ50KF5P60Hb20gA80`** | GEc `I10RQ50KF5P60Hb20gGEc`, RLc `I10RLcKF5P60Hb20`, RR `I10RQ50KF5P60Hb20` |

Wrapper `v8f_t1_eval_reader_rule_rzgfhqetjv.py` (28bc31f21ecb4ccb), unchanged; each new run is called exactly as the banked GEc run of its setting with the gap file exchanged. A margin on the gap is a parameter of the rule's input: a study arm, measured here; its adoption is a design decision for the PI and the other session.

## 3. Measures and estimator

Per setting on its windows with a lead (325 per model seed at K30, 73 at K60): contact C, unsafe U, falling behind by more than 10 m F for A90, A80, GEc, RLc, RR; levels pooled and per model seed. Pairs per setting, paired on the same windows, two-level bootstrap over model seeds and evaluation episodes, 2,000 draws: A90 − GEc, A80 − GEc, A90 − RLc, A80 − RLc, and GEc − RLc (identity with rows 179 and 181).

Four families per setting as in card ADD117: LONGITUDINAL (the measures above and the engine read of each pair), LATERAL and TACTICAL (engine family reads of each pair), STRATEGIC (the strategic grid at T1-kin for each pair, position error at the marks of 30, 50 and 80 m). The interval answers what another draw of episodes and of these two seeds would say, not what another training run would say.

## 4. Names

The names of card ADD117 with the prefix `GAPMARGINAGE10K` and S in {K30, K60}, by the same reader logic: per arm A in {A90, A80} and measure M in {UNSAFE, CONTACT, FALLING-BEHIND} `GAPMARGINAGE10K-<S>-<A>-LOWERS-<M>` / `-RAISES-<M>` / `-<M>-NOT-SEPARATED` (interval of A − GEc); `GAPMARGINAGE10K-<S>-<A>-UNSAFE-ABOVE-LABEL-FORM` / `-UNSAFE-BELOW-LABEL-FORM` / `-UNSAFE-AT-LABEL-FORM` (interval of A − RLc). An arm closes the unsafe rate at S when it lowers it against the candidate and its pooled unsafe rate is at most that of the label form plus 0.0045 (the bound of card ADD117, kept). One name per setting: `GAPMARGINAGE10K-<S>-MARGIN-CLOSES-UNSAFE-AT-NO-SEPARATED-PRICE` / `-MARGIN-CLOSES-UNSAFE-AT-A-PRICE` / `-MARGIN-DOES-NOT-CLOSE-UNSAFE`.

Stated before data: at these settings the candidate's unsafe rate is not separated from the label form's to begin with (row 181 at K30: +0.0125 [-0.0051, +0.0332]), so the name per setting describes whether a margin moves it and at what price; it cannot show a closing that the instrument could not see open.

## 5. Controls (all have to pass before any name or value is opened)

* **m0 gap files** — the record of card ADD117 for the two scaled files passes as in its control k0, and the files on disk have the hashes of that record.
* **m1 wrapper** — the wrapper has the hash named above and its banked check files are in place.
* **m2 arguments** — for each new run, the first line of its run log against that of the banked GEc run of its setting: every field both carry is equal (form, replan steps, rule every, readout, heading bound, pre-roll, exec, nav, grid, model seed, reader seed, checkpoint).
* **m3 identity** — the reader reproduces within 1e-4 the banked levels of GEc, RR and RLc at K30 (0.0062 / 0.0839 / 0.5154, 0.0954 / 0.1339 / 0.4954, 0.0000 / 0.0714 / 0.4908) and at K60 (0.0137 / 0.0738 / 0.6438, 0.1096 / 0.1230 / 0.6575, 0.0137 / 0.0410 / 0.6301).
* **m4 tests** — the queue kit's test passes in the lead's run before the queue start; the reader kit's test passes in the lead's run before the controls stage; outputs outside the kit folders.
* **m5 engine** — the engine's own controls of each grid read and of the strategic grid pass.
* **m6 counts** — every new run has its done marker and 141 dump files.
* **m7 rule logs** — for each new run the rule log names the hash of its scaled file and the closing switch, and agrees with that file as in control k7 of card ADD117.

This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

Order of work, stated before data: the queue kit and the reader kit are generated from the kits of card ADD117 (tables and run lines only), tested by the lead on the pod, and the queue is started beside the fits of card ADD118 (the queue starts a rollout only with fewer than three GPU processes and more than 28 GB free). No value of a new run is opened before the reader kit's tests pass and the controls are logged.

## 6. Committed readings

* A margin lowers the unsafe rate at K30 with falling behind raised by at most 0.03 → the price table of the margin in the candidate spec gains the aged settings; the design decision put to the PI and the other session is stated with all four settings.
* A margin raises falling behind at K30 by more than 0.03, or the strategic read is WORSE → stated as the price of that margin at age; the spec keeps the margin as a study-arm option for readout K with no pre-roll only, with that limit beside it.
* Nothing separates at a setting → reported as not separated for that setting, with the number of windows; no claim for or against the margin there.
* K60 has 73 windows with a lead per seed, below the instrument's bar of 200: its names are reported with that statement and decide nothing alone.

## 7. Cost and limits

ESTIMATED 1.0 pod-h for eight rollouts, one after another, beside the fits of card ADD118; stop rule 3 pod-h from the queue start. Limits: T1-kin only, no re-perception; the lead moves as logged whatever the ego does; two model seeds of one line, and a separated interval from two model seeds is necessary, not sufficient; few windows at K60; two margins only; a margin does not make the gap read more exact; the label form is a reference with label values and not a candidate.
