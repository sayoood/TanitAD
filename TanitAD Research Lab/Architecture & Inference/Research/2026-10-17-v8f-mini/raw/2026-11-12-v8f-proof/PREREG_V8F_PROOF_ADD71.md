# Card PREREG_V8F_PROOF_ADD71 (AGERULE10K): at a rollout age of 6 to 12 s, the candidate against the convention with the rule, the own-acceleration term, and what part of the cost of age is the carried pose; no training

Card of the v8F proof study, written by the coordinator fork after the read of card ADD70 under its note N3 (all controls pass; names in the line of that read in `TIMES_NOTE.md`) and before any run or tool of this card exists (time of the freeze: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). Tier of every number: T1-kin/hb0.2 (kinematic rollout on the model's own actions with the heading bound 0.2 1/m, study wrapper; logged camera frames; the lead does not react); never driving performance.

## 1. What is known (MEASURED, card ADD70, 1,271 windows of 141 episodes, 280 scored lead windows, two model seeds)

* A window that starts after 6 s of the model's own driving reads worse than the same window started fresh in every setting: the unsafe rate is higher by 0.47 to 0.50 and the four families are WORSE for the convention, for readout K, for both candidate forms and with the logged route. At the start of the scored window the own pose is off the logged pose by 6.1 to 6.5 m and 7.0 to 7.3 deg on average under readout K (8.1 to 8.6 m and 12.5 to 13.0 deg in the convention).
* At that age both candidate forms still read BETTER on four families than the convention without a rule (U lower for the label-fed form, not separated for the reader-fed form).
* Not measured: (a) the candidate against the convention WITH the rule at that age, which is the pair the candidate specification reports; (b) the own-acceleration term at that age (the partner of acceptance check C1); (c) how much of the cost of age is the carried pose (where the window starts) and how much is the own motion state the model reads after 6 s of own driving. Training on the model's own rolled history (recommendation C of decision card D16) can act on the second part only.

## 2. Part A: six rollouts on the pod (P = 30 grid, pre-roll of 30 steps, 2.0 s cycle, wrapper of record of card ADD70, `--kappa-max 0.2`, model seeds 0 and 1)

| run | form | readout | rule interval |
|---|---|---|---|
| `RLcP30Hb20` | label-fed rule `RLc` | pinned | every replan |
| `RQ50P30Hb20` | reader-fed rule `RR` at map level 0.50 | pinned | every replan |
| `R0C10XP30Hb20` | no rule | causal readout on the executed path with the own acceleration (`--readout causal_g --acc-gain 1 --path-exec`) | - |

Fresh partners and aged partners are banked runs (cards ADD67, ADD68, ADD70), restricted in memory to the windows of the P = 30 grid by the reader of card ADD70.

Named pairs (A minus B):

| pair | A | B |
|---|---|---|
| LABEL-CAND-VS-RULE-AT-AGE | `RLcKF5P30Hb20` | `RLcP30Hb20` |
| READER-CAND-VS-RULE-AT-AGE | `RQ50KF5P30Hb20` | `RQ50P30Hb20` |
| ACC-AT-AGE | `R0C10XP30Hb20` | `R0KP30Hb20` |
| AGE-LABEL-RULE | `RLcP30Hb20` | `RLcHb20` |
| AGE-READER-RULE | `RQ50P30Hb20` | `RQ50Hb20` |
| AGE-ACC | `R0C10XP30Hb20` | `R0C10XHb20` |

Beside, naming nothing: the three first pairs on fresh windows of the same grid (`RLcKF5Hb20` minus `RLcHb20`, `RQ50KF5Hb20` minus `RQ50Hb20`, `R0C10XHb20` minus `R0KHb20`) with the difference of the two differences; the rule in the convention at age (`RLcP30Hb20` and `RQ50P30Hb20` minus `R0P30Hb20`, headway shares).

Estimators and readings as in card ADD70 with notes N2 and N3: headway by the paired episode-cluster bootstrap pooled over the model seeds, family cells by the family reader of card ADD46 with the spread of `R0Hb20` over the two seeds on the same windows as the bar, STRATEGIC by the instrument of card ADD66 (route goal at 30, 50, 80 m at the third tick; a decision between routes is not scored, n = 0). Names: `AGERULE10K-<PAIR>-<FAMILY>-BETTER|WORSE|MIXED|NOT-SEPARATED` and `AGERULE10K-<PAIR>-U-LOWER|HIGHER|NOT-SEPARATED`.

What follows, fixed now:

* LABEL-CAND-VS-RULE-AT-AGE or READER-CAND-VS-RULE-AT-AGE with a family BETTER, none WORSE and U not higher: that candidate form keeps its reading against the convention with the rule at an age of 6 to 12 s; the specification says so. A family WORSE or U HIGHER: the specification states that limit at age. Anything else: reported as written.
* ACC-AT-AGE with a family WORSE: the cost of handing the model its own acceleration holds at age; its values are the line-model reference of acceptance check C1 at age. No family WORSE: at age the term does not separate from the drift on these windows; C1 is read on fresh windows only.
* AGE-LABEL-RULE, AGE-READER-RULE, AGE-ACC: named and reported as written; they decide nothing by themselves.

## 3. Part B: the carried pose against the own motion state (CPU, no rollout; banked runs `R0KNlHb20` and `R0KNlP30Hb20` of card ADD70, both seeds)

In these two runs there is no rule and the route input is the logged one, so no input of the planner and no step of the executor depends on the absolute pose: a window started at the carried pose with the LOGGED motion state would drive the fresh track, displaced rigidly by the carried offset. Per window and seed three tracks are scored with the family reader's cells against the logged track: the fresh track f, the fresh track displaced by the offset of the aged run at the start of the scored window d (offset = own pose minus logged pose at t0 from the tick file of the aged run, in the logged frame; the speeds of d are those of f), and the aged track a. POSE part = d minus f. STATE part = a minus d. Their sum is the pair AGE-K-NAVLOGGED of card ADD70.

Per named cell of LONGITUDINAL, LATERAL and TACTICAL: the two parts with the two-level bootstrap of the family reader. Names per family: `AGERULE10K-DECOMP-<FAMILY>-STATE-LARGER` when on every named cell of the family the STATE part is above 0 with its interval excluding 0 and larger than the POSE part with the interval of (STATE minus POSE) excluding 0; `-POSE-LARGER` for the mirror case; `-BOTH` when both parts exclude 0 on a named cell and neither is larger by that rule; else `-NOT-SEPARATED`. STRATEGIC is not decomposed here (its instrument reads tick files; a displaced tick file is a larger tool) and is stated as not decomposed.

What follows, fixed now: STATE-LARGER on a family: the larger part of the cost of age there is the motion state the model reads after its own driving; training on the model's own rolled history is the lever, and acceptance check C5 is read on the whole AGE pair. POSE-LARGER: the larger part there is where the window starts while the camera frames stay the logged ones; a kinematic rollout cannot repair or fairly score that, a tier with re-rendered or re-perceived frames would be needed (a PI decision), and C5 for that family is read on the STATE part. BOTH or NOT-SEPARATED: both parts are reported and C5 is read on both.

## 4. Controls (all must pass; else nothing is named)

Part A: the controls of the reader of card ADD70 with note N3 on the six new runs (window counts, pre-roll logs, readout logs, first-tick offset term, hashes of wrapper and checkpoints, the reader controls on the P = 30 grid and on all windows, strategic rows), and a1: the pair `R0KP30Hb20` minus `R0KHb20` read by the tool of this card equals the banked result of card ADD70 (pair AGE-K) on every headway share and every family cell to 1e-9. e1: the pod read reproduces the bank (the file of the first read of card ADD70 is reused only if this read runs in the same pod session; else e1 runs again).

Part B: b1 with a zero offset d equals f on every cell exactly; b2 a planted offset of 1 m ahead on a straight synthetic track changes the along-track error by 1 m and no lateral cell; b3 a planted heading offset of 90 deg turns an along-track run into a cross-track one of the same length; b4 the sum of the two parts equals the AGE-K-NAVLOGGED cells of card ADD70 to 1e-9 on the differences of the means; b5 the first executed point of the aged track lies within 0.5 m plus one step of travel of the offset position for at least 99 percent of the windows (the offset read from the tick file is the one the track starts from); b6 window identity of the three tracks.

## 5. Limits

T1-kin/hb0.2 with logged camera frames and a lead that does not react; two model seeds are not a replicate study; a separated interval is necessary, not sufficient. The P = 30 grid holds 280 scored lead windows in 46 episodes. Part B decomposes one setting (readout K, no rule, logged route); with a rule or with the dead-reckoned route the displaced window is not the displaced fresh track, and no decomposition is claimed there. The displaced track d is a counterfactual computed from a banked run, not a rollout. Nothing is adopted by this card: a rollout age in the convention, a re-perception tier and any training arm are decisions of the PI and the other session.
