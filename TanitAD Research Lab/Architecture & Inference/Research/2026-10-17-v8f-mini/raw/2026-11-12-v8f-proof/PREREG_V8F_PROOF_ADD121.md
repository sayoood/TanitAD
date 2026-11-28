# PREREG_V8F_PROOF_ADD121 — REPL10K: the lead-rule candidate set on two further model seeds of the 10k line

Frozen before the two further models exist and before any tool of this card is written. PI decision of 2026-11-22 (verbatim: "Yes, two more seeds"). Open at this time: rows 179 to 186. Tier **T1-kin/hb0.2** (kinematic rollout on the model's own actions, executor of record `v8f_t1_eval_hb.py` with the heading bound 0.2 1/m). The two models `UE_s2`, `UE_s3` are trained by the other session with the recipe of `UE_s0`, `UE_s1` and only the seed changed, under its launch gate; this fork trains nothing. Here: 26 rollouts on the two new models, no fit, no encode.

## 1. Why this card

Every separated interval of rows 179 to 186 rests on the two model seeds 0 and 1. The paired bootstrap resamples episodes and those two seeds; a separated interval from one pair of training runs is necessary, not sufficient. This card reads the same arms on a second pair of training runs and asks, claim by claim, whether the direction that was separated on the first pair is separated again on the second pair alone. The margin of 10 per cent on the camera gap, adopted by the PI for readout K on 2026-11-22, is the first of these claims.

Stated guess before data: the margin's cut of the unsafe rate under readout K with no pre-roll is separated again; at the pre-roll of 30 steps (6 s) it has the same sign and may not separate; the candidate's distance to the label form at the convention is separated again; under readout K with no pre-roll it may not separate.

## 2. Models, arms and runs

Model seed m in {2, 3}. The token reader and its map read frozen tokens and do not depend on the model: model seed 2 runs with reader seed 0 and its map, model seed 3 with reader seed 1 and its map, so that against the first pair only the training run of the model changes. Gap files, wrapper (`v8f_t1_eval_reader_rule_rzgfhqetjv.py`, 28bc31f21ecb4ccb), grids and evaluation join are those of the banked runs.

Arms: RLc (label form, a reference with label values, never a candidate), RR (reader-fed form), GEc (the candidate: reader's presence, camera gap file `VGAPLE.npz`, closing switch), A90 (the candidate with `VGAPLE_A90.npz` of card ADD117), and the spread run R0 (no rule) that the reader needs.

| setting | runs per model seed (names as banked, with the new seed) |
|---|---|
| CONV (pinned readout, no pre-roll, pinned grid, rule every 10 steps) | `I10RLcHb20`, `I10RQ50Hb20`, `I10RQ50Hb20gGEc`, `I10RQ50Hb20gA90`, and `I10R0Hb20` |
| K0 (readout K, no pre-roll, pinned grid, rule every 5 steps) | `I10RLcKF5Hb20`, `I10RQ50KF5Hb20`, `I10RQ50KF5Hb20gGEc`, `I10RQ50KF5Hb20gA90` |
| K30 (readout K, pre-roll of 30 steps (6 s), grid_p30, rule every 5 steps) | `I10RLcKF5P30Hb20`, `I10RQ50KF5P30Hb20`, `I10RQ50KF5P30Hb20gGEc`, `I10RQ50KF5P30Hb20gA90` |

Each new run is called exactly as the banked run of the same name at model seed m - 2, with the checkpoint exchanged. No label enters at inference in RR, GEc and A90.

## 3. Measures and estimator

Per setting on its windows with a lead: contact C, unsafe U, falling behind by more than 10 m F; levels per model seed for all four seeds and pooled per pair of seeds. Pairs of arms per setting, paired on the same windows: A90 - GEc, GEc - RLc, A90 - RLc, GEc - RR. Two estimates of each: on the new pair alone (two-level bootstrap over the model seeds 2 and 3 and the evaluation episodes, 2,000 draws), and on all four seeds (the same bootstrap over four seeds). The first pair is recomputed by the same code as identity.

Four families per setting on the new pair, as in card ADD117: LONGITUDINAL (the measures above and the engine read of each pair), LATERAL and TACTICAL (engine family reads), STRATEGIC (strategic grid at T1-kin, position error at the marks of 30, 50 and 80 m). The interval on the new pair answers what another draw of episodes and of these two training runs would say; set beside the first pair it is the first read of this rig against a second pair of training runs.

Printed with no name: for each arm and setting the difference of the pooled level of the new pair to that of the first pair, as the spread between pairs of training runs.

## 4. Claims and names

Each claim has the side that was separated on the first pair:

| claim | pair and measure | setting | side | first pair |
|---|---|---|---|---|
| M-K0 | A90 - GEc, unsafe | K0 | below 0 | -0.0125 [-0.0203, -0.0057] (row 183) |
| M-K30 | A90 - GEc, unsafe | K30 | below 0 | -0.0268 [-0.0499, -0.0092] (row 185) |
| P-K0 | A90 - GEc, falling behind | K0 | above 0 | +0.0078 [+0.0016, +0.0153] (row 183) |
| D-CONV | GEc - RLc, unsafe | CONV | above 0 | +0.0249 [+0.0072, +0.0464] (row 182) |
| D-K0 | GEc - RLc, unsafe | K0 | above 0 | +0.0133 [+0.0036, +0.0250] (row 181) |
| C-K0 | GEc - RR, contact | K0 | below 0 | levels 0.0000 against 0.0296; the interval is recomputed here |
| C-CONV | GEc - RR, contact | CONV | below 0 | levels 0.0016 against 0.0280; the interval is recomputed here |

Per claim X, from the interval on the new pair alone: `REPL10K-<X>-REPLICATED` (on the claimed side), `REPL10K-<X>-NOT-SEPARATED-ON-THE-NEW-PAIR` (it includes 0), `REPL10K-<X>-OPPOSITE` (on the other side). A claim of the last two rows counts only if its recomputed interval on the first pair is on the claimed side; else it is reported as not a claim.

For the margin: `REPL10K-MARGIN-REPLICATED` when M-K0 and M-K30 are both replicated; `REPL10K-MARGIN-REPLICATED-AT-K0-ONLY` or `REPL10K-MARGIN-REPLICATED-AT-K30-ONLY` when one is; `REPL10K-MARGIN-NOT-REPLICATED` when none is. Beside it, with no name: whether the pooled unsafe rate of A90 on the new pair is at most that of the label form plus 0.0045 at K0 (the bound of card ADD117).

The estimates on four seeds are reported for every pair and carry no name.

## 5. Controls (all have to pass before any name or value is opened)

* **r0 models** — for each new model the checkpoint on the fork pod has the sha256 that the other session's arm record names; that record states a launch-gate pass and the recipe of `UE_s0`, `UE_s1` (code commit, arguments and data manifests) with the seed as the only difference. The lead writes the record's place and hashes into a note to this card before the queue start. A model with a changed recipe is not admitted and the card is not run on it.
* **r1 wrapper and gap files** — hashes as in the banked runs; the record of card ADD117 for the scaled file passes.
* **r2 arguments** — for each new run, the first line of its run log against that of the banked run of the same name at model seed m - 2: every field both carry is equal but the model seed and the checkpoint hash; the reader seed is m - 2.
* **r3 identity** — the reader reproduces within 1e-4 the banked pooled levels of the first pair: CONV GEc 0.0016 / 0.0569 / 0.2329, A90 0.0008 / 0.0463 / 0.2539, RLc 0.0016 / 0.0320 / 0.2243, RR 0.0280 / 0.0783 / 0.2593; K0 GEc 0.0000 / 0.0498 / 0.1651, A90 0.0000 / 0.0374 / 0.1729, RLc 0.0000 / 0.0365 / 0.1550, RR 0.0296 / 0.0738 / 0.1682; K30 GEc 0.0062 / 0.0839 / 0.5154, A90 0.0031 / 0.0571 / 0.5246, RLc 0.0000 / 0.0714 / 0.4908, RR 0.0954 / 0.1339 / 0.4954.
* **r4 tests** — the queue kit's test passes in the lead's run before the queue start; the reader kit's test passes in the lead's run before the controls stage; outputs outside the kit folders.
* **r5 engine** — the engine's own controls of each grid read and of the strategic grid pass.
* **r6 counts** — every new run has its done marker and 141 dump files; per setting the number of windows with a lead per new model seed equals that of the first pair (642, 642, 325).
* **r7 rule logs** — for each new run with a gap file the rule log names the hash of its gap file and the closing switch and agrees with the file as in control k7 of card ADD117.

This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

Order of work, stated before data: the queue kit and the reader kit are generated after this freeze from the kits of cards ADD117 and ADD119 (tables, run lines, the seeds, and the estimate on four seeds); they may be built and tested while the models are trained. The queue starts only after control r0's note is written. No value of a new run is opened before the reader kit's tests pass and the controls are logged.

## 6. Committed readings

* **MARGIN-REPLICATED** → the PI's adoption of the margin for readout K stands on a second pair of training runs; the price table of the candidate spec is restated on four seeds.
* **MARGIN-REPLICATED at one setting only, or MARGIN-NOT-REPLICATED** → reported as written to the PI and the other session with the estimates on four seeds: the adoption of 2026-11-22 rests on the first pair where the second does not separate, and whether it stays is the PI's decision. The fork changes nothing in the spec but the table.
* **D-CONV or D-K0 not separated on the new pair** → the stated limit of the candidate against the label form is rewritten as measured on four seeds; no claim that the distance is closed.
* **C-K0 and C-CONV** → whatever they read, the cut of the contact rate by the candidate against the reader-fed form is restated with both pairs.
* **Any claim OPPOSITE** → reported first; the spread between the two pairs is then stated as the floor of this rig, and no lever claim below it is made from one pair.
* The second pair differs from the first in the training run only (same reader seeds), so what does not replicate is attributed to training variance of the model and not to the reader.

## 7. Cost and limits

ESTIMATED 4 pod-h for 26 rollouts on the fork pod, one after another; stop rule 7 pod-h from the queue start. The training of the two models is the other session's arm and cost. Limits: T1-kin only, no re-perception; the lead moves as logged whatever the ego does; two pairs of training runs of one line are still a small number; reader seeds and gap files are those of the first pair, so reader variance is not read here; the windows are the same 141 clips; the label form is a reference with label values and not a candidate.
