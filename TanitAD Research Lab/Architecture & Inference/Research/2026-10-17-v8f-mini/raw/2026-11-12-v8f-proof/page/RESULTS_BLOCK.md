**World-model yardstick (H-V8F-PROOF-3: WMCEIL, WMSTACK, JUMPCURVE), MEASURED, tier T0, a world-model diagnostic.** Sources `wm/VERDICT_WMCEIL.md`, `wm/VERDICT_WMSTACK.md`; eval-141, 1,976 windows, 141 episodes, 5 folds by episode, two seeds (s0 / s1); cards frozen before the data. Controls: the banked world-model skill is reproduced to 0.000000; a shuffled target reads at most +0.0027; a stack fed only the learned prediction reproduces it exactly.
* Outcomes by the frozen cards: **WM-BELOW-LINEAR**, **JUMP-DOMINATES**, **STACK-ADDS**, **JUMP-MONOTONE**.
* Skill against "nothing changes" (learned network / closed-form ridge / closed-form stack of both):
* 0.2 s: +0.0010 / +0.0222 / **+0.0402** (seed 1: -0.0010 / +0.0215 / +0.0400).
* 0.4 s: +0.0295 / +0.0272 / **+0.0619** (seed 1: +0.0264 / +0.0265 / +0.0585).
* 0.8 s: +0.1316 / +0.0752 / **+0.1531** (seed 1: +0.1351 / +0.0752 / +0.1561).
* 2.0 s: -0.1500 / +0.1225 / **+0.1710** (seed 1: -0.1724 / +0.1292 / +0.1641).
* The stack is written in closed form, with no gradient step. It is better than the better single predictor at all four horizons at both seeds (+0.018 to +0.049, every interval excluding 0). At 2 s it moves the world model from -0.15 / -0.17 to +0.17 / +0.16.
* Theorem T1 along the horizon: the direct k-step error over k times the one-step error reads 0.743, 0.573, 0.521, 0.416 [0.389, 0.449] at 0.4, 0.8, 1.0 and 2.0 s (seed 1: 0.745, 0.586, 0.539, 0.433). The criterion holds and strengthens with horizon. The realised penalty of composing a well-fitted linear step is small (composed over direct error 1.07 / 1.06 at 2 s); it is the learned network whose ten-step rollout fails.
* Limits. The latent changes slowly (lag correlation 0.975 at 0.2 s, 0.883 at 2 s) and the actions add little (ridge without actions: +0.0218 at 0.2 s, +0.089 at 2 s), so even the stack explains 4 % to 17 % of the change: the target space limits imagination, not only the predictor. Nothing here is an effect on driving; the world model is a training auxiliary today. The bootstrap answers "another draw of episodes"; the two seeds agree.

**Rule shield on real v8F plans (addendum 3, RULESHIELD), MEASURED, tier T1-kin on logged frames, a diagnostic.** Source `rule/VERDICT_RULESHIELD.md`; zero training; first-tick plans of the two DAG-OACC-recipe seeds (s0 / s1); 2,399 windows, 131 stop windows; paired episode-cluster bootstrap B 2,000. The shield code is the solver-checked algorithm (VC-C1, VC-C2), with 7 properties tested on 20,000 random cases each.
* Without a shield the plan drives past the logged rest point in 91.6 % / 84.7 % of the stop windows, by 9.7 / 9.1 m on average.
* With the logged rest distance as the cue (ORACLE, privileged: a ceiling, not a deployable arm) the overrun is exactly 0 on every feasible window (128 / 129 of 131; 3 / 2 windows cannot stop at 3 m/s2 from their start state and are reported apart). Outcome **SHIELD-HOLDS-AND-GAINS**.
* On the stop windows: 2-6 s ADE -3.51 m [-4.38, -2.62] / -3.37 m [-4.33, -2.34]; 0-6 s ADE -2.36 / -2.27 m; final displacement -8.47 / -8.05 m; speed error -1.25 / -1.17 m/s; cross-track -0.51 / -0.55 m; heading -1.40 / -1.00 deg; curvature and yaw rate not separated. Over all 2,399 windows: 0-6 s ADE -0.129 / -0.124 m. No cost cell at both seeds.
* Controls: the mutation without the braking fallback overruns on 117 of 120 / 109 of 111 windows; the path is unchanged to 1.8e-15 m; without a cue the plan is bit-identical.
* With the cue the model itself emits (T-CUE) nothing is gained: the stop flag fires on 19 / 33 windows, 10 / 18 of them stop windows, and its stop position is off by 11.8 / 9.6 m (median). Outcome SHIELD-HOLDS-NO-GAIN.
* Reading: the rule mechanism is worth about 3.4 m on stop windows and costs nothing measurable; all of it waits on the cue. No replicate arm exists, so the interval answers another draw of episodes only; the effect is about 12 times the tactical replicate floor (0.287 m). The plan has 30 knots at 0.2 s, so the shield ran at dt 0.2 s (the card assumed 0.1 s; recorded as a deviation).

**Bias edit, arithmetic preview only (addendum 4, BIASEDIT; the weight-edit run is the deciding read and has not run).** Source `bias/QUICK_OFFSET.md`, banked first-tick reads, T1-kin. Adding one offset to the stop-goal logit of the unweighted model reproduces the false-positive rate of the trained weighting doses at offsets 1.09 and 2.76 (seed 0; ln W is 1.52 and 3.04), and the hit rate there is within 0.009 and 0.026 of the trained arms, inside the replicate gap of 0.068.

**Weight-space read of a lever (H-V8F-PROOF-2 TASKVEC), MEASURED, banked checkpoints, one replicate pair, no interval.** Source `wsp/TASKVEC.md`. Outcome **DIFFUSE**.
* The weight change caused by the stop weighting is smaller than the change caused by another seed in every module group (0.42 over all trainable tensors, 0.11 in the world model).
* Doubling the dose does not scale the change (norm ratio 0.98; cosine 0.50 between the two doses). Most of what separates a lever arm from its control in weight space is training chaos, not the lever.
* The only tensors where the lever clearly exceeds seed noise are two head biases (`h_lon.bias` 3.81, `h_goal.bias` 1.55). That is theorem T4 seen in the weights: a class weight is a logit offset. Follow-up frozen (addendum 4, BIASEDIT): add ln W to one bias scalar and compare with the trained dose.

**The existing hard rule code (`project_speed`), MEASURED, code property check.** Source `fv/CODE_PROPS_VERDICT.md`; 20,000 random paths per family; spec frozen first.
* The speed cap holds for every output: proved for the construction and 100 % in the test; removing the cap line turns the test red.
* The grip (Kamm) term is not a projection. Read by its own formula, its output exceeds the bound on 42 % to 99.9 % of the test paths by family (median 1.2 to 6.3 times the bound among failing paths), applying it twice moves the plan again, and a turn at the first step is never slowed.
* Not measured: how often the grip term is active in the evaluation, so no reported number is called into question by this alone.

**Literature, now primary.** Six cited papers were added to the Library (sha256 verified) and nine already banked ones were linked. The two bounds the first pass left unsure are read from the PDFs: MBPO Theorem 4.1 and the 2018 hierarchy sub-optimality bound (`kb/BOUNDS_VERBATIM.md`).

**Not read yet:** the weight-average test (SOUP) and the cached-token resolution probe (RESPROBE). Both dev-box jobs stopped on out-of-memory errors at 00:26 and 00:28 on 13 November (the machine was shared with other jobs); no number was produced. They move to the pod, with the bias-edit run behind SOUP.
