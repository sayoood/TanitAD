# PREREG_V8F_PROOF_ADD134 — GAPSPREAD10K: the margin placed by the spread of the three gap fits in the lead rule, against the flat margin

Frozen before any tool of this card is written and before any value of it exists. PI decision of 2026-11-23 (verbatim: "follow your recommendation", in answer to "Spend about 0.6 pod-hours on a rollout card for the spread-placed margin?", the recommendation being a card first, then four rollouts). Open at this time: row 201 (cards ADD132 and ADD133, landing). Tier **T1-kin/hb0.2** (kinematic rollout on the model's own actions, executor of record `v8f_t1_eval_hb.py` with the heading bound 0.2 1/m). Four rollouts on the four existing 10k models `UE_s0` to `UE_s3`; no training, no fit, no encode, no detector.

## 1. Why this card

Row 201 reads at T0 that the spread of the candidate's three gap fits marks the frames where the gap reads long, and names a 10 % margin placed by it `PLACED-NOT-SEPARATED` from the flat margin on the long part; its committed reading spends no rollout on it. Beside that name the short part of the placed margin was 0.8010 [0.7400, 0.8613] of the flat margin's, and below 20 m its long part was smaller. This card is the departure from that committed reading which the PI approved: it is a hypothesis formed after the data of row 201, and it gets its own form and its own run here. The question: does the placed margin keep the unsafe rate of the flat margin A90 (which the PI adopted for readout K, rows 183 and 193) at less falling behind?

Stated guess before data: the label rows receive a mean factor of 0.9215, a weaker margin where a lead is; the unsafe rate of the placed margin lies above that of A90 by a small amount that is not separated, falling behind lies below by about half of the margin's price (+0.0062), and the name is `PLACED-NOT-SEPARATED-FROM-FLAT`.

## 2. Arm and runs

Arm **SP90**: the candidate form GEc (reader's own presence, `--vis-gap-file`, `--vis-closing`, no label at inference) with the gap file `VGAPSP90.npz`: on every row with an output the base gap of `VGAPLE.npz` (55de2f7c) times max(0.60, 1 - k s), s the relative spread of the three fit files of card ADD106 and k the one number of card ADD133 (mean factor 0.90 over all rows with an output; no label enters); every other array as in the base file. The closing input is the slope of the gap the rule receives. Setting K0 (readout K, no pre-roll, pinned grid, rule every 5 steps), model seeds 0 to 3: new runs `I10RQ50KF5Hb20gSP90_s0` to `_s3`. Banked cells on the same four seeds: A90 `I10RQ50KF5Hb20gA90`, GEc `I10RQ50KF5Hb20gGEc`, RLc `I10RLcKF5Hb20`, RR `I10RQ50KF5Hb20`. Wrapper `v8f_t1_eval_reader_rule_rzgfhqetjv.py` (28bc31f21ecb4ccb), unchanged; each new run is called exactly as the banked A90 run of its model seed with the gap file exchanged. A margin placed by the spread is a parameter of the rule's input: measured here as a study arm; its adoption is a design decision for the PI and the other session.

## 3. Measures and estimator

On the windows with a lead: contact C, unsafe U, falling behind by more than 10 m F for SP90, A90, GEc, RLc, RR; levels per model seed and pooled over the four. Pairs, paired on the same windows, two-level bootstrap over the four model seeds and the evaluation episodes, 2,000 draws: SP90 - A90 (the pair of the name), SP90 - GEc, SP90 - RLc, and A90 - GEc (identity with row 193). Four families for SP90 - A90 and SP90 - GEc as in card ADD121: LONGITUDINAL (the measures above and the engine read), LATERAL and TACTICAL (engine family reads), STRATEGIC (the strategic grid at T1-kin, position error at the marks of 30, 50 and 80 m). The interval answers what another draw of episodes and of these four model seeds would say; the four seeds are two pairs of training runs of one line.

## 4. Names

For pair P in {SP90 - A90 (`VS-FLAT`), SP90 - GEc (`VS-CANDIDATE`)} and measure M in {UNSAFE, CONTACT, FALLING-BEHIND}: `GAPSPREAD10K-K0-<P>-LOWERS-<M>` / `-RAISES-<M>` / `-<M>-NOT-SEPARATED` (interval below 0 / above 0 / else).

One name on the question, from SP90 - A90, the first clause that holds:

1. `GAPSPREAD10K-K0-PLACED-GIVES-UP-SAFETY` — the interval of the unsafe rate or of the contact rate lies above 0;
2. `GAPSPREAD10K-K0-PLACED-SAME-SAFETY-AT-A-LOWER-PRICE` — the upper end of the interval of the unsafe rate is at most +0.0058 (half of the 0.0116 by which the flat margin lowers the unsafe rate on four seeds), the interval of the contact rate does not lie above 0, and the interval of falling behind lies below 0;
3. `GAPSPREAD10K-K0-PLACED-SAFER-AT-NO-HIGHER-PRICE` — the interval of the unsafe rate lies below 0 and the interval of falling behind does not lie above 0;
4. `GAPSPREAD10K-K0-PLACED-NOT-SEPARATED-FROM-FLAT` — else.

## 5. Controls (all have to pass before any name or value is opened)

* **p0 gap file** — `VGAPSP90.npz` differs from the base file in `gap` only; it is not a number exactly where the base has no output; on every row with an output it equals the base gap times the factor within 1e-9 m by a second routine; k equals the k of row 201 (0.6033) within 1e-4 and the mean factor is 0.90 within 1e-6; on the label rows of row 201 the long and the short part of the file reproduce row 201 (0.492 m and 3.105 m) within 1e-3 m; the tools of cards ADD132 and ADD133 are imported unchanged; the record holds the hashes of the base file, the three fit files and the new file, and the count of rows whose gap leaves the rule's range of 0 to 60 m.
* **p1 wrapper** — the wrapper has the hash named above and its banked check files are in place.
* **p2 arguments** — for each new run, the first line of its run log against that of the banked A90 run of the same model seed: every field both carry is equal (form, replan steps, rule every, readout, heading bound, pre-roll, exec, nav, grid, model seed, reader seed, checkpoint).
* **p3 identity** — the reader reproduces within 1e-4 the banked levels on four seeds at K0 of row 193 (GEc 0.0000 / 0.0480 / 0.1589, A90 0.0000 / 0.0365 / 0.1651, RLc 0.0004 / 0.0369 / 0.1460, RR 0.0300 / 0.0703 / 0.1589) and the pair A90 - GEc on four seeds (unsafe -0.0116 [-0.0175, -0.0063], falling behind +0.0062 [+0.0019, +0.0112]).
* **p4 tests** — the queue kit's test passes in the lead's run before the queue start; the reader kit's test passes in the lead's run before the controls stage; outputs outside the kit folders.
* **p5 engine** — the engine's own controls of each grid read and of the strategic grid pass.
* **p6 counts** — every new run has its done marker and 141 dump files.
* **p7 rule logs** — for each new run the rule log names the hash of `VGAPSP90.npz` and the closing switch; the gap the rule received at steps with an output equals the file's gap within 1e-3 m, and at steps with no output it is the reader's own, as in the reader-fed run of the same model seed.

This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

## 6. Committed readings

* **SAME-SAFETY-AT-A-LOWER-PRICE or SAFER-AT-NO-HIGHER-PRICE** → the placed margin enters the candidate spec as a study-arm option beside the flat margin, with its measured price, and goes to the PI and the other session as a design decision. The next card reads it at the pre-roll of 30 steps (6 s) under readout K, where the price in falling behind is largest.
* **GIVES-UP-SAFETY** → the mark the spread carries at T0 does not carry into the rule; the flat margin stays the form under readout K.
* **NOT-SEPARATED-FROM-FLAT** → the flat margin stays: of two forms that read alike the one with no further input is kept.
* After either of the last two the gap line is read out as far as existing files go, and the next card turns to the unsafe rate of the label form itself (0.0369 on four seeds), on banked runs with no new rollout.

## 7. Cost and limits

ESTIMATED 0.6 pod-hours for four rollouts on the fork pod, one after another (a rollout took about 9 minutes in card ADD131); stop rule 1.0 pod-hours from the queue start, above it back to the PI. The gap file and the read are CPU work of minutes. Limits: T1-kin only, no re-perception; the lead moves as logged whatever the ego does; readout K with no pre-roll alone (the margin is a setting of readout K); four model seeds of one line; one dose, one floor; the hypothesis was formed after the data of row 201 and the form of SP90 was fixed before it; the label form is a reference with label values and not a candidate.
