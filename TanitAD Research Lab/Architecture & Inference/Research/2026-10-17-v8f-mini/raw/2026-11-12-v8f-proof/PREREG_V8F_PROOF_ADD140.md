# PREREG_V8F_PROOF_ADD140 — RHK10K: is the adopted recipe (RH under readout K) better than the line model under readout K, fresh and after 6 s of own driving?

Frozen before any tool of this card is written and before any value of it exists. Open at this time: the read of card ADD139 with its amendments 1 to 3 (`rhacc10k/read/out/READ_RHACC10K.json`, sha256 d58f1017ab2f1f43, all controls passed, `RHACC10K-NOT-YET`; not yet landed). Tier **T1-kin/hb0.2** (kinematic rollout on the model's own actions; logged camera frames; the lead does not react). Never driving performance.

## 1. Why this card

Card ADD139 read every acceptance check of the specification as failed for RH: with the causal readout and its own acceleration, RH reads worse than readout K on every family but STRATEGIC (C1), worse than the line candidate under readout K (C2), and the 6 s rollout-age cost stays (C5). So readout K cannot be dropped. The PI adopted RH under readout K ("Adopt RH", 2026-11-24; specification section 5b). Card ADD139 did not run RH under readout K with the rule. This card does: it asks whether the recipe as adopted is a better candidate than the line candidate, and whether it lowers the cost of 6 s of own driving.

Stated guess before data: fresh, RH under K is not separated from UE under K on every family, with U not separated (the other session read RH against B0 under readout K as not separated on headway); after 6 s of own driving, RH reads BETTER on TACTICAL and LONGITUDINAL (it was trained on its own rolled history). Name guessed: `RHK10K-FRESH-EQUAL`, `RHK10K-AGED-BETTER`.

## 2. Models and environment

RH_s0..s2 and UE_s0..s2 as in card ADD139 and its amendment 1 (sha256 in `rhacc10k/inputs/INPUTS.json`); readers as amendment 2 (seed 2 uses reader seed 0). Box 2, the same tree, pins, eval cache and safe mode as card ADD139. The runs `U_RLcK_s<m>` and `U_RQ50K_s<m>` of card ADD139 (done on box 2, controls passed) are reused as the fresh partners; they are not run again.

## 3. New runs (per model seed m in {0, 1, 2}; order of record m = 0, 1, 2, each i, j, k, l)

All with `--replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal --kappa-max 0.2`, readout K (`--readout causal_g --acc-gain 0 --path-exec`), the rule every 5 steps.

| run | model | form | grid | wrapper |
|---|---|---|---|---|
| i `H_RLcK` | RH_s<m> | RLc | eval-141 (2,399 windows) | `v8f_t1_eval_reader_rule_rzgfh.py` |
| j `H_RQ50K` | RH_s<m> | RR at map level 0.50 | eval-141 | `v8f_t1_eval_reader_rule_rzgfh.py` |
| k `H_RLcK_P30` | RH_s<m> | RLc | grid of record of card ADD70 (1,271 windows), pre-roll 30 steps (6 s) | `v8f_t1_eval_reader_rule_rzgfhq.py` |
| l `U_RLcK_P30` | UE_s<m> | RLc | the same, pre-roll 30 steps | `v8f_t1_eval_reader_rule_rzgfhq.py` |

12 runs. This card has no map of arms to controls beyond section 4.

## 4. Controls (all must pass before any value is opened; a failed control stops the card, no value is read, a successor card is frozen)

* q1 complete: done files; 141 tick files (i, j) or the count of the ADD70 grid (k, l); no non-finite value in a dumped plan; a pre-roll log for k and l.
* q2 models and argv: the checkpoint of each run equals its recorded sha256; each run's argv carries exactly its row's flags and wrapper.
* q3 reused partners: the folders `U_RLcK_s<m>`, `U_RQ50K_s<m>` are byte-identical to the ones read in card ADD139 (sha256 of every file listed before this card's read).
* q4 identity: the dumps of i and of `U_RLcK` are not byte-identical for any seed; k and l use the same windows (same (clip, t0) list).
* The reader controls of the landed headway, family, route-goal and pre-roll readers pass as in card ADD139.

Names opened after the controls pass, in this order.

## 5. Reads (landed readers unchanged; estimators as card ADD139: headway pooled paired episode-cluster bootstrap over the three seeds; families with the two-level reader of card ADD46, bar the spread of `U_RLcK` over the seeds; route goal with the reader of card ADD66 at the third tick; pre-rolled pairs with `integ10k/read_longroll10k_n4.py`, bar the spread of `U_RLcK_P30`)

* R1 fresh, label form: i minus `U_RLcK`. R2 fresh, reader form: j minus `U_RQ50K`. Per family BETTER / WORSE / NOT-SEPARATED / MIXED, U, C, F, route goal at 30, 50, 80 m.
* R3 aged, label form: k minus l (both after 6 s of own driving, same windows).
* Names: `RHK10K-FRESH-BETTER` if R1 and R2 read no family WORSE, U not higher (separated) and at least one family BETTER; `RHK10K-FRESH-EQUAL` if no family WORSE or BETTER and U not separated; `RHK10K-FRESH-WORSE` if any family WORSE in R1 or R2; else `RHK10K-FRESH-MIXED`. `RHK10K-AGED-BETTER` if R3 reads TACTICAL or LONGITUDINAL BETTER and no family WORSE; `RHK10K-AGED-EQUAL` if no family separated; `RHK10K-AGED-WORSE` if any family WORSE without one BETTER; else `RHK10K-AGED-MIXED`.
* Consequence stated in advance: FRESH-BETTER or FRESH-EQUAL with AGED-BETTER makes RH under K the candidate of record in place of UE under K (specification section 1, model row); FRESH-WORSE keeps the line candidate and sends the recipe question back to the other session with the failing cells.

## 6. Cost and limits

Box 2, one job at a time, safe mode, telemetry and the USB temperature rule, busy marker. 12 runs at about 2.5 to 3.5 min each, about 40 min. No training, no encode. If a run takes more than 15 min the queue stops and the card goes back with its cost.
