# Note N7 to card PREREG_V8F_PROOF_ADD40: the trained low-quantile gap head as the present read of the planner

Note of the v8F proof study, written by the coordinator fork. Frozen before the tool option exists and before any run of this note (time of the freeze: the line of this note in `TIMES_NOTE.md`; hash in the sidecar file). Tier of every rate and family below: T1-kin (kinematic rollout on the model's own actions; the lead does not react); never driving performance.

## 1. Why (MEASURED, with the artifact)

* Row H-V8F-PROOF-95 (`imagplan/read_i9/RLA3/READ_I9_RLA3.json`): the planner fed by the full imagined rollout reads `IMAG-PLAN-CUTS-UNSAFE` with `IMAG-PLAN-EQUALS-EXTRAPOLATION`. Section 6 of the card, second bullet, committed for this outcome: "the next lever inside the present encoder is the trained low-quantile gap head (card ADD31 section 8) read through the same tool".
* Row H-V8F-PROOF-94 (`imagplan/read_cvroom/`): at 2.0 s the read of the lead ranks before the forecast of its motion; inside the read the gap part is the larger one.
* Row H-V8F-PROOF-90: in the distance rule the trained 0.20-quantile gap head (`g_used = min(g_head, q_head)`) reaches the label-fed rule on the unsafe rate on both arm types.

## 2. The lever (inference only; no weight of any model changes)

A new source of the planner tool, `cand-cvq`: source `cand-cv` of the card with the present gap replaced by `g_used = min(g_head, q_head)`; the closing speed, the presence gate, the candidate family (decel), the floor and the fallback are those of `cand-cv`, unchanged. `q_head` is the quantile head of card ADD42 fitted for the same checkpoint with all training clips (the fit form of note ADD42_N2; the banked fits of note ADD42_N3 for the LS arms and of note ADD42_N2 for the LSD arms; no new fit). A head whose coverage on its check clips lies outside [0.15, 0.25] is not read (band of card ADD42, unchanged).

## 3. Arms and runs

* First pair: LS_s0, LS_s1 (lead-supervised arms, weight 1; the arm type proposed for the integrated tree). Runs per checkpoint, decel family, suffix q: `IPo` (no planner), `IPc` (`cand-cv`), `IPl` (`cand-label`), `IPcq` (`cand-cvq`), and the two control runs of section 5.
* Second pair (the replicate on two further training runs): LSD_s0, LSD_s1 with their own banked heads, the same runs. It is run whatever the first pair reads.
* No training. Budget (ESTIMATED from the planner runs of row 95: six runs in about 25 min beside two trainings): under 2 pod-h for both pairs, one run at a time behind the three-job limit and the memory guard of the sequence script.

## 4. Reads and names (per pair; U = model-induced unsafe rate of the headway read on the 642 lead windows of 56 episodes; paired differences, two-level bootstrap over the two checkpoints, then episodes, B 2,000)

* `QPLAN-CUTS-MORE`: U(`cand-cvq`) minus U(`cand-cv`) below 0 at both checkpoints with the interval excluding 0. `QPLAN-CUTS-LESS`: above 0 at both with the interval excluding 0. Else `QPLAN-EQUAL`.
* `QPLAN-REACHES-ORACLE`: U(`cand-cvq`) minus U(`cand-label`) at most +0.010 at both checkpoints and the interval does not lie above 0.
* Price against `cand-cv` (what the head adds to the plain planner): the family read `cand-cvq` minus `cand-cv` by the rule of card ADD12 Q3 with the same-sign rule of ADD13: `QPLAN-NO-COST` or `QPLAN-COSTS-<family>`; the 6 s position error difference in metres with its interval. Beside: the same against no planner and against label-fed; the contact rate against label-fed.
* A name of the first pair is a candidate. It is stated as a result with the suffix `-CONFIRMED` only if the second pair reads the same name. If the pairs differ, both are reported and nothing is confirmed.
* The four metric families are reported for `cand-cvq` against `cand-cv` and against no planner. STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and the reader has no strategic instrument).

## 5. Controls (all must pass for a pair; else nothing is named for it)

* q1 (identity): with the quantile head replaced by a constant of 1e9 m the dumps of `cand-cvq` equal those of `cand-cv` bit for bit.
* q2 (firing, planted): with the quantile head replaced by a constant of 0 m the 6 s position error rises against `cand-cv` (interval excluding 0): the tool shows a too-short gap as a price.
* q3 (the head is the banked one): sha256 of the head file equals the banked record of its fit; its check-clip coverage is in the band.
* q4 (known positive): `cand-label` cuts U against no planner at both checkpoints with the interval excluding 0 (p2 of the card).
* q5 (counts): 2,399 windows, 642 lead windows of 56 episodes; the active ticks of `cand-cv` and `cand-cvq` are the same set.
* q6: the controls h1 to h6 of the headway read and q1 to q5 of the family read.
* Tool checks before any run: unit tests of the new source (a hand-computed case of the min; the identity; refusal without a head file) and a counted mutation list (the min dropped; the head applied to another source; the head read at the wrong tick) that goes red.

## 6. What follows (committed before the data)

* `QPLAN-CUTS-MORE-CONFIRMED` (with or without `-REACHES-ORACLE`): the planner with the trained gap head is the planner form read beside the distance rule on the integrated candidate (plan step 3); its price is stated beside the price of the rule (row 90).
* `QPLAN-EQUAL` or `-CUTS-LESS` on both pairs: reported as written; the conservative gap does not help candidate selection where it helps the rule. The next lever is the outcome of the grid replicate (card ADD43) and, if the finer grid leaves the closing-speed read flat, a probe that reads the lead from two frames (own card).
* Any `QPLAN-COSTS-<family>` against `cand-cv`: reported beside the cut; the head is then offered to the integrated candidate only in the rule.

## 7. Limits

Two arm types with two training runs each, one corpus; T1-kin (the lead does not react); the head was fitted for use in the rule, not for candidate selection; a quantile gap is conservative by construction, so a cut with a cost is the expected form and only the size of each decides its use.
