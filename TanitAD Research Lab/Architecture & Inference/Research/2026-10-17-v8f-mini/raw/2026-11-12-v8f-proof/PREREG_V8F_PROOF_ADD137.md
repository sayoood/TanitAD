# PREREG_V8F_PROOF_ADD137 — RULELOG10K: in the unsafe windows of the label form, is it the rule's forecast of the lead or the rollout's execution that loses the gap?

Frozen before any tool of this card is written and before any value of it exists. Open at this time: row 203 (card ADD135, landed). Tier **T1-kin/hb0.2** (kinematic rollout on the model's own actions, executor of record `v8f_t1_eval_hb.py` with the heading bound 0.2 1/m). CPU only, on banked runs and their banked rule logs; no new rollout, no training, no fit.

## 1. Why this card

Row 203: with label inputs the lead rule keeps an unsafe rate of 0.0369 at readout K with no pre-roll; its 83 unsafe windows over four model seeds (31 distinct) are slow, close following in which the rollout ends more than 0.2 s closer than the logged driver. The rule (`dist_rule.py`: distance D 3.0 m, target time gap 1.2 s, braking limit 6.0 m/s2) is applied every 5 steps (1.0 s) under readout K; at a tick it caps the plan's progress so that the gap to a lead moving on at the constant speed v_L stays at or above max(D, 1.2 s times the ego speed of the step before). A window can then lose the gap in two ways: the lead moves on less than the constant-speed forecast says, or the rollout moves on more than the cap allows. They call for different levers (the rule's forecast and rate; the path from the plan to the executed motion). This card splits the loss on the banked logs.

Stated guess before data: the rule is on at the tick before the first unsafe step in nine of ten windows; the braking limit does not bind; the lead's forecast carries the larger part (pooled share of the loss between 0.6 and 0.8), name `LEAD-FORECAST-CARRIES`.

## 2. Windows and quantities

The unsafe windows of the label form RLc `I10RLcKF5Hb20` at the model seeds 0 to 3, exactly those of card ADD135 (its tool imported unchanged), each with its first unsafe step k*. From the engine per step k: the model's arc a_m(k), the model gap g_m(k), the model step speed v_m(k). From the banked rule log of the run, per rule tick: the step k of the tick, `active`, `present`, `sat_exec`, the gap g0 the rule used and the lead speed v_L it assumed. The tick of a window: the last rule tick k_t with k_t < k*; i = k* - k_t steps.

* Lead position along the path: P(k) = a_m(k) + g_m(k).
* Loss against the rule's target at the first unsafe step: S = max(3.0 m, 1.2 s x v_m(k* - 1)) - g_m(k*).
* Forecast part: e_L = [P(k_t) + v_L x 0.2 s x i] - P(k*): how far short of the constant-speed forecast the lead is.
* Execution part: e_X = [a_m(k*) - a_m(k_t)] - [g_m(k_t) + v_L x 0.2 s x i - max(3.0 m, 1.2 s x v_m(k* - 1))]: how far the rollout is beyond the cap, with the cap written with the realised speed of the step before.

By these definitions S = e_L + e_X in every window.

## 3. Measures and estimator

Over the unsafe windows pooled over four model seeds:

* **R1** — share with the rule on at the tick (`active` and `present` both true).
* **R2** — share with `sat_exec` above 0 at the tick (the braking limit binds within the executed steps).
* **R3** — the pooled share of the loss carried by the forecast: the sum of e_L over the sum of S, over the windows with the rule on at the tick.
* **R4** — share of those windows in which e_L is larger than e_X.

Each with a cluster bootstrap over the evaluation episodes, 2,000 draws, seed 137, 95 % interval. Beside, no name: the medians of S, e_L, e_X and i; the share of windows in which the lead's mean speed over the interval lies more than 0.5 m/s below v_L; the share with a negative e_X; R1 to R4 per model seed; the same four measures on the candidate GEc `I10RQ50KF5Hb20gGEc` (there g0 is the camera gap and P is read with the label gap; its loss has a third part, g0 minus the label gap at the tick, printed beside). The interval answers what another draw of episodes would say, not what another training run would say.

Families: LONGITUDINAL (distance keeping) is the subject. LATERAL, TACTICAL and STRATEGIC are not read by this card (n = 0 here; the family reads of these banked runs are in rows 193 and 202).

## 4. Names

Split, one name: `RULELOG10K-K0-LEAD-FORECAST-CARRIES` (the lower end of the interval of R3 is at least 0.5), `RULELOG10K-K0-EXECUTION-CARRIES` (its upper end is at most 0.5), `RULELOG10K-K0-MIXED` (else).

Rule state, one name: `RULELOG10K-K0-RULE-ON` (the lower end of the interval of R1 is at least 0.9), else `RULELOG10K-K0-RULE-OFF-IN-PART`. Braking limit, one name: `RULELOG10K-K0-BRAKING-LIMIT-BINDS` (the lower end of the interval of R2 is at least 0.5), else `RULELOG10K-K0-BRAKING-LIMIT-DOES-NOT-BIND`.

## 5. Controls (all have to pass before any name or value is opened)

* **w0 windows** — the unsafe windows and their first unsafe steps are those of the tool of card ADD135 (imported unchanged, hash recorded): 22, 19, 22 and 20 windows at the four model seeds.
* **w1 log join** — every unsafe window has its rule ticks in the banked rule log of its run, at the steps the rule rate gives; at each tick the logged g0 equals the engine's g_m at that step within 1e-3 m and the logged arc of the model equals the engine's a_m within 1e-3 m.
* **w2 identity** — S equals e_L plus e_X within 1e-6 m in every window, with S taken from the engine's gap and speed, e_L from the logged v_L and the lead position, e_X from the model's arc, the logged g0 and v_L.
* **w3 the forecast part reads the forecast alone** — with v_L replaced by the lead's realised mean speed over the interval, e_L is 0 within 1e-6 m in every window and e_X equals S; with v_L raised by 1.0 m/s, e_L rises by 0.2 m x i in every window.
* **w4 tests** — the tool's test file passes in the lead's run on the dev box and on the pod before the run of record, with the name of the test file in the output; outputs outside the kit folder.
* **w5 counts** — 141 dump files, the done marker and one rule log for each of the eight runs.

This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

## 6. Committed readings

* **LEAD-FORECAST-CARRIES** → the lever is the rule's forecast of the lead and its rate: a rule that is applied at every step, or that forecasts the lead with its measured change of speed, would act on the larger part. Both are changes of the rule: a study arm with a rollout cost goes to the PI and the other session as a card.
* **EXECUTION-CARRIES** → the rule asks for the gap and the rollout does not keep it: the lever lies between the plan and the executed motion (readout, path execution), which is the other session's line; the finding goes to it with the windows' numbers.
* **MIXED** → both parts are stated with their shares, and the study arm of the larger part is named first.
* `RULE-OFF-IN-PART` or `BRAKING-LIMIT-BINDS` beside any of them is stated as a further part with its share.

## 7. Cost and limits

ESTIMATED under 15 minutes of CPU on the fork pod, no GPU time. Limits: T1-kin only, no re-perception, the lead moves as logged; 83 windows over four model seeds of one line, 31 distinct; the cap is written with the realised speed of the step before, the rule itself uses the planned one; the split reads the last tick before the first unsafe step alone, a loss built up over earlier ticks sits in g_m(k_t); the label form is a reference with label values and not a candidate.
