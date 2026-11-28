# PREREG_V8F_PROOF_ADD117 — GAPMARGIN10K: does a margin on the camera gap close the unsafe rate of the candidate, and at what price?

Frozen before any tool of this card is written and before any value of it exists. Open at this time: card ADD114 (row 181 in writing) and card ADD116 (row 180); no value of card ADD115 is open (its rollouts are running). Tier **T1-kin/hb0.2** (kinematic rollout on the model's own actions, executor of record `v8f_t1_eval_hb.py` with the heading bound 0.2 1/m). Eight rollouts of the candidate form with a scaled gap file on the two existing 10k models `UE_s0`, `UE_s1`; no training, no fit, no encode.

## 1. Why this card

Card ADD114 reads `PRESATTR10K-K0-GAP-CARRIES-UNSAFE`: at readout K with no pre-roll the unsafe rate of the candidate above the label form (0.0498 against 0.0365) comes with the camera gap value and not with the reader's presence (label presence with the camera gap 0.0489; the reader's presence with the label gap 0.0365, equal to the label form). Rows 172 to 178 have tried to make the camera gap more exact with what exists: more clips (the error falls, no driving measure moves), windows at the head's direction (not separated), smoothing (pays in safety), a map of the gap value (does not cut the slow part). A more exact read needs the reader's own input, a new encode, which is the PI's decision.

One lever needs neither a fit nor an encode nor a label: the rule can be handed a gap that is deliberately short. A gap estimate with a known error of a few metres is not an exact gap, and a rule that treats it as exact follows closer than the label form whenever the estimate is long. This card measures what a fixed margin buys and what it costs, so that the unsafe rate of the candidate is either closed at a known price or stays a stated limit of the gap input.

Stated guess before data: a margin of 10 % lowers the unsafe rate but does not reach the label form; 20 % reaches it at readout K with no pre-roll, with a separated price in falling behind.

## 2. Arms and runs

The candidate form GEc (reader's own presence, `--vis-gap-file`, `--vis-closing`, no label at inference) with the gap file replaced by a scaled copy of the base file `VGAPLE.npz` of card ADD106 (sha256 begins 55de2f7c): `gap' = a * gap` on every row with an output, every other array unchanged. Two margins: a = 0.90 (arm A90) and a = 0.80 (arm A80). The closing input is the slope of the gap the rule receives (so it is scaled with it). The rule's own test of a camera gap (finite, 0 to 60 m) is applied to the scaled value, so a row whose base gap lies between 60 m and 60/a m becomes a lead for the rule; their number is printed with no name.

| setting | A90 | A80 | banked cells |
|---|---|---|---|
| K0 (readout K, no pre-roll, pinned grid, rule every 5 steps) | **new `I10RQ50KF5Hb20gA90`** | **new `I10RQ50KF5Hb20gA80`** | GEc `I10RQ50KF5Hb20gGEc`, RLc `I10RLcKF5Hb20`, RR `I10RQ50KF5Hb20` |
| CONV (pinned readout, no pre-roll, pinned grid) | **new `I10RQ50Hb20gA90`** | **new `I10RQ50Hb20gA80`** | GEc `I10RQ50Hb20gGEc`, RLc `I10RLcHb20`, RR `I10RQ50Hb20` |

Wrapper `v8f_t1_eval_reader_rule_rzgfhqetjv.py` (28bc31f21ecb4ccb), unchanged; each new run is called exactly as the banked GEc run of its setting with the gap file exchanged. A margin on the gap is a parameter of the rule's input: as a study arm it is measured here; its adoption is a design decision for the PI and the other session.

## 3. Measures and estimator

Per setting on its windows with a lead (642 per model seed): contact C, unsafe U, falling behind by more than 10 m F for A90, A80, GEc, RLc, RR; levels pooled and per model seed. Pairs per setting, paired on the same windows, two-level bootstrap over model seeds and evaluation episodes, 2,000 draws: A90 − GEc, A80 − GEc, A90 − RLc, A80 − RLc, and GEc − RLc (identity with rows 179 and 181).

Four families per setting as in card ADD113: LONGITUDINAL (the measures above and the engine read of each pair), LATERAL and TACTICAL (engine family reads of each pair), STRATEGIC (the strategic grid at T1-kin for each pair, position error at the marks of 30, 50 and 80 m). The interval answers what another draw of episodes and of these two seeds would say, not what another training run would say.

## 4. Names

Per setting S in {K0, CONV} and arm A in {A90, A80}:

* against the candidate, for measure M in {UNSAFE, CONTACT, FALLING-BEHIND}: `GAPMARGIN10K-<S>-<A>-LOWERS-<M>` / `-RAISES-<M>` / `-<M>-NOT-SEPARATED` (interval of A − GEc below 0 / above 0 / else);
* against the label form on the unsafe rate: `GAPMARGIN10K-<S>-<A>-UNSAFE-ABOVE-LABEL-FORM` / `-UNSAFE-BELOW-LABEL-FORM` / `-UNSAFE-AT-LABEL-FORM` (interval of A − RLc above 0 / below 0 / else).

An arm **closes** the unsafe rate at S when it lowers the unsafe rate against the candidate (interval below 0) and its pooled unsafe rate is at most that of the label form plus 0.0045 (a third of the difference of 0.0133 that row 181 reads at K0). One name per setting:

* `GAPMARGIN10K-<S>-MARGIN-CLOSES-UNSAFE-AT-NO-SEPARATED-PRICE` — an arm closes it, and that arm's falling behind is not separated from the candidate's (of two such arms the one with the larger a is named);
* `GAPMARGIN10K-<S>-MARGIN-CLOSES-UNSAFE-AT-A-PRICE` — an arm closes it, and every arm that closes it raises falling behind against the candidate;
* `GAPMARGIN10K-<S>-MARGIN-DOES-NOT-CLOSE-UNSAFE` — no arm closes it.

## 5. Controls (all have to pass before any name or value is opened)

* **k0 gap files** — each scaled file differs from the base file in `gap` only; on every row with an output it equals a times the base gap within 1e-9 m and is not a number where the base has no output; the same tool at a = 1 gives the base arrays exactly; the record holds the hashes of the base file and of both scaled files, and the count of rows whose base gap lies between 60 m and 60/a m.
* **k1 wrapper** — the wrapper has the hash named above and its banked check files are in place.
* **k2 arguments** — for each new run, the first line of its run log against that of the banked GEc run of its setting: every field both carry is equal (form, replan steps, rule every, readout, heading bound, pre-roll, exec, nav, grid, model seed, reader seed, checkpoint).
* **k3 identity** — the reader reproduces within 1e-4 the banked levels of GEc, RR and RLc at K0 (0.0000 / 0.0498 / 0.1651, 0.0296 / 0.0738 / 0.1682, 0.0000 / 0.0365 / 0.1550) and at CONV (0.0016 / 0.0569 / 0.2329, 0.0280 / 0.0783 / 0.2593, 0.0016 / 0.0320 / 0.2243).
* **k4 tests** — the queue kit's test passes in the lead's run before the queue start; the reader kit's test passes in the lead's run before the controls stage; outputs outside the kit folders.
* **k5 engine** — the engine's own controls of each grid read and of the strategic grid pass.
* **k6 counts** — every new run has its done marker and 141 dump files.
* **k7 rule logs** — for each new run the rule log names the hash of its scaled file and the closing switch, and agrees with that file in the way control b2 of card ADD111 checks it (the gap the rule received at steps with an output equals the file's gap within 1e-3 m; at steps with no output it is the reader's own, as in the reader-fed run of the setting).

This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

Order of work, stated before data: the queue kit (with the gap-file tool) is built, tested and dry-run by the lead and placed on the pod to start when the queue of card ADD115 has written its done marker (a stop marker of that queue starts nothing); the reader kit is built while the rollouts run. No value of a new run is opened before the reader kit's tests pass and the controls are logged.

## 6. Committed readings

* **MARGIN-CLOSES-UNSAFE (either form) at K0** → the margin of the named arm enters the candidate spec as a study-arm option with its measured price at both settings, and goes to the PI and the other session as a design decision (adopt the margin, or carry the unsafe rate as a limit). The next card then reads the named margin at the two aged settings (pre-roll of 30 and of 60 steps), where the price in falling behind is largest.
* **MARGIN-DOES-NOT-CLOSE-UNSAFE at K0** → a margin of up to 20 % is not the lever; the unsafe rate of the candidate stays a stated limit of the gap input, and what remains is a more exact read of the gap (a new encode, the PI's decision, put to the PI with its cost and with what rows 171 to 178 measured).
* The two settings disagree → reported as written; the spec carries the reading of each setting.

## 7. Cost and limits

ESTIMATED 1.0 pod-h for eight rollouts on the pinned grid, one after another; stop rule 3 pod-h from the queue start. Limits: T1-kin only, no re-perception; the lead moves as logged whatever the ego does; two model seeds of one line, and a separated interval from two model seeds is necessary, not sufficient; the unsafe differences in question are about 15 windows of 1,124 scored; a margin moves the rule along a trade between following close and falling behind and does not make the gap read more exact; two margins only; the label form is a reference with label values and not a candidate.
