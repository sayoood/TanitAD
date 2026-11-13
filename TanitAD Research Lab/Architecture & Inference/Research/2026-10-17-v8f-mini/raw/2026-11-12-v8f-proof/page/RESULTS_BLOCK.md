**Standing against the three goals, 13 Nov 08:05.** Every measured value below is a T0 or T1-kin diagnostic on logged frames, never driving performance. The strategic family is paused (PI item 25).

**Goal 1, prove dominance of the hierarchy: NOT ESTABLISHED yet. The deciding experiment is frozen and in preparation on the pod.**
* Proved by the solver (47 conditions, 23 mutated twins refuted): what a hierarchy can gain over a flat model and through which term (T1, T2), and why a commit rule cannot bank that gain at the present decision quality (group J).
* Measured on the banked rollouts: the hierarchy is ahead of its flat control on selection-critical windows in all four cells (-0.48, -0.16, -0.18, -0.12 m on 6 s ADE; three of the four intervals exclude 0). On common-case windows it is ahead in one cell and behind in three. On all windows the verdict of the programme stays NOT ESTABLISHED: one training run per arm per seed.
* Deciding experiment (addendum 11, frozen 13 Nov 07:03 before its arms): 24 arms, six seeds, hierarchy against flat in both training protocols, seeds as the unit. A class reads HIER only when all six seed differences agree in sign and the two-level interval excludes 0.

**Goal 2, validate the design: three parts hold, two gaps are located.**
* Holds: the jump-ratio and linear-yardstick statements for the world model (T1, T6); the rule shield, with conformal coverage as proved (T3); a two-member weight average realises 0.74 of the ensemble gain (T9).
* Gap A, the physical world beyond ego motion: the frozen camera tokens carry lead presence, lead range, nearest agent and agent count beyond ego motion, but the latents of the model do not expose them, and no stop cue is found in the pooled tokens. A finer grid carries more lead content under a full-kernel probe, and the probes are still rising at 111 episodes. The direct read on un-pooled and higher-resolution tokens (addendum 9) gives the interface a small part: un-pooling adds 0.05 R2 on lead gap, a higher-resolution centre crop adds 0.07 AUROC on lead presence, the full higher-resolution grid adds nothing, and no stop cue appears beyond ego motion. Pictures decoded from the world-model state (addendum 10) show the scene gist and no object.
* Gap B, decisions: the tactical confidence tells right from wrong decisions at AUROC 0.64; a commit rule needs about 0.9 to bank half of the oracle gain.

**Goal 3, start scaling: started 13 Nov.**
* The pod (A40) is rebuilding the training token cache from the frames (about 750 clips per hour, 4,713 clips); the gated chain for the 24 arms waits behind it.
* Committed arms after the replicate, by its outcome: the data-efficiency pair (a quarter, a half and all of the clips), the finer-interface pair, training weighted toward selection-critical windows, and a third replicate for the three-member weight average.

**Resolution probe (H-V8F-PROOF-4, RESPROBE), MEASURED, tier T0, a perception diagnostic. Outcome RES-REFUTED: a prediction of this page failed.** Source `res/VERDICT_RESPROBE.md`; frozen-feature probes (ridge on 256 principal components), 9,321 frames, 139 episodes (lead gap: 56), 5 folds by episode, episode-cluster bootstrap B 2,000.
* Predicted from the camera geometry: the 4x10 token grid cannot carry far range, and the 8x20 grid would do better. Measured: neither.
* The 4x10 tokens carry range beyond ego motion at 12.5 m and more: lead-gap R2 gain over the ego-only control +0.707 [+0.353, +1.178] in that stratum (bar 0.20). Both stratum R2 are below 0 (ego -0.846, tokens -0.139): the tokens find the range band, not the range inside it.
* The 8x20 grid adds nothing measurable over 4x10: lead gap +0.0111 R2 [-0.0233, +0.0479], nearest agent +0.0019 [-0.0085, +0.0129], agent count +0.0026 [-0.0066, +0.0262], lead present +0.0178 AUROC [+0.0005, +0.0355].
* Vision adds over ego motion on all four targets (4x10 with ego minus ego alone): lead gap +0.2927 R2 [+0.1082, +0.5512], nearest agent +0.1491 [+0.0220, +0.2661], agent count +0.1637 [+0.0320, +0.3689], lead present +0.1801 AUROC [+0.0899, +0.2673]. Absolute levels are low: lead gap R2 0.31, lead present AUROC 0.79.
* Reading: the input tokens carry scene content beyond ego motion that the latents of the model do not expose. The gap is between input and reader, not at the pooling grid, as far as these probes can see.
* One control missed its written bar (shuffled-label AUROC pooled over folds reads 0.43 to 0.50). Measured cause: the lead-present rate differs by fold (0.24 to 0.32), so a constant per-fold score reads 0.467 pooled; per fold the control reads 0.485 to 0.513. The outcome rule uses R2 cells only, whose shuffled control reads at most 0. Limit: a finer grid may need more components or data to show (next card).

**Where the hierarchy differs from its flat control (addenda 6 and 6b, HIERWHERE), MEASURED, tier T1-kin on logged frames, a diagnostic; zero training, banked rollouts.** Sources `hier/VERDICT_HIERWHERE.md`, `hier/pair2/VERDICT_HIERWHERE2.md`. The matched flat control was already run by the programme (H-V8F-FLATDAG-1, NOT ESTABLISHED: -0.3451 m [-0.4421, -0.2506] at seed 0, -0.0544 m [-0.1605, +0.0547] at seed 1). One run per arm per seed, no replicate arm.
* Windows are split by the logged future with thresholds fixed before the read: 1,305 selection-critical (the future is not keep-speed-and-straight), 1,094 bulk.
* Selection-critical, hierarchy minus flat: -0.4795 [-0.6222, -0.3445] / -0.1589 [-0.3043, -0.0117] m. Bulk: -0.1847 [-0.2933, -0.0866] / +0.0702 [-0.0610, +0.2201] m. Outcome CRIT-CONCENTRATED.
* Second banked pair (the same fine-tunes without own-state supervision), outcome CRIT-PARTIAL: selection-critical -0.1817 [-0.3054, -0.0686] / -0.1196 [-0.2619, +0.0243] m; bulk +0.3180 [-0.0079, +0.7169] / +0.4928 [+0.2488, +0.7552] m. There the hierarchy is worse than flat on the common case.
* Critical minus bulk is negative and separated in all four cells (-0.2948, -0.2291, -0.4997, -0.6124 m).
* The exact decomposition did not confirm the theory as read (outcome MIXED): at seed 0 of the first pair -0.2594 of -0.3451 m sits on windows where neither arm has the logged mode, -0.0536 where exactly one has it, -0.0320 where both have it. Both arms miss the 12-cell logged mode on 53 % to 71 % of windows.
* Commit rule (addendum 8, GATEMIX, `gate/VERDICT_GATEMIX.md`, all four controls pass), outcome **GATE-COSTS**: letting the hierarchy drive only where its own tactical output is confident of a decision that is not the common case beats both models in no cell and is worse than one of them in two of four. That confidence tells selection-critical from bulk windows with AUROC 0.64 only. A gate that knows the logged window class (a privileged ceiling) is ahead of the flat model in all four cells (-0.26, -0.09, -0.10, -0.07 m). The information to commit only where it pays is not in the tactical output today.
* Why, proved (group J, 9 conditions with Z3, written after the read): a gate beats the flat model only if its hit rate on selection-critical windows exceeds s times its false-alarm rate on bulk windows, where s is the bulk cost per unit of critical gain (0.37, 1.47 and 3.45 in the three cells where the two window groups disagree). ESTIMATED from the measured group means with a binormal gate: at AUROC 0.64 a gate banks at most 7 % of what the class oracle banks; half of it needs AUROC 0.88 to 0.94.
* Reading: the hierarchy gains where the future is not the common case and can cost where it is. These reads locate the difference; they do not establish dominance. A label-swap control missed its written single-seed bar once; over 20 swap seeds it reads 19 / 18 of 20 (first pair) and 18 / 15 of 20 (second pair, bar 15).

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

**Stop cue from the frozen tokens (addendum 5, CUEPROBE), MEASURED, T0 for the probes and T1-kin on logged frames for the shield, a diagnostic.** Source `cue/VERDICT_CUEPROBE.md`; closed-form probes cross-fitted by episode on the 2,399 windows of the rule-shield read (131 stop windows); zero training.
* Stop flag: ego kinematics alone AUROC 0.810 [0.744, 0.868]; adding the 8x20 tokens gives 0.799 (difference -0.011 [-0.061, +0.043]). Outcome **VISION-FLAT**.
* Stop distance: ego kinematics reach 3.82 / 3.97 m median error; the tokens reach 10.5 to 12.5 m, no better than a constant (10.3 m). Outcome **CUE-ABSENT**.
* Shield with this cue: no gain on stop windows (+0.09 m [-0.53, +0.64] / +0.09 [-0.45, +0.61]) and +0.052 / +0.049 m cost elsewhere. Outcome **CUE-PROBE-COSTS**. The flag hits 27 % of stop windows at a 2.2 % false-positive rate.
* The conformal margin does what the theory says: coverage 0.908 (119 of 131) for a nominal 0.9, and with the true stop windows the overrun share falls from 0.916 / 0.847 to 0.107 / 0.092. The margin is about 7.4 m, so the plan stops short and the error does not improve.
* Reading: for lead range the pooled tokens carry coarse content; for the stop cue they carry nothing these probes can find. The rule is safe by construction; its usefulness waits on a sharp cue.
* One control missed its bar as implemented: the leak-regression arm raises the flag AUROC on the two vision-only sets and not on the two sets with ego. Limit: 131 stop windows, one frame, a closed-form probe.

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

**Probe rank, data and selection curves (addendum 7, RESCURVE), MEASURED, T0, a perception diagnostic.** Source `curve/VERDICT_RESCURVE.md`; 9,321 frames, 139 episodes; controls c1 to c3 pass.
* Rank, outcome **RANK-HIDES**: the earlier finding that the 8x20 grid adds nothing held for the rank-256 probe only. With the full kernel the finer grid is ahead on lead gap (+0.076 R2 [+0.001, +0.170]) and on lead present (+0.043 AUROC [+0.022, +0.066]). The gap gain is thin (its interval touches 0); nearest agent and agent count stay flat.
* Data: the probes are still rising at the full 111 training episodes, about 37 minutes of driving: lead present AUROC 0.73, 0.78, 0.84 at 28, 56, 111 episodes. Outcome DATA-LIMITED for lead present, UNDECIDED for lead gap.
* Selection, outcome **SELECT-FLAT**: choosing a quarter or a half of the episodes by coverage or by leverage in token space is inside the range of random draws on every target.

**Weight average of two replicates (main card, SOUP), MEASURED, T1-kin, a diagnostic.** Source `wsp/VERDICT_WSP.md`; eval-141; controls c1 to c3 pass.
* Outcome **SOUP-REALISES**: averaging the weights of two same-init replicates gives -0.075 m [-0.103, -0.047] on tactical ADE against the mean of the two, 0.74 [0.48, 0.99] of what running both models and averaging their plans would give, at the cost of one model.
* No family pays for it: longitudinal and lateral cells are equal or better, tactical decisions are unchanged.
* Limit: the average is not ahead of the better member (3.254 against 3.238 m); it removes the risk of drawing the worse one (3.420 m). One pair, so no run-to-run interval.
* Dose by weight interpolation (P2b): outcome DOSE-NONLINEAR, missed on the false-positive rate by 0.0004.

**Un-pooled and higher-resolution tokens (addendum 9, FULLRES), MEASURED, T0, a perception diagnostic.** Source `full/VERDICT_FULLRES.md`; 139 episodes; controls f1 to f5 pass.
* Lead gap R2 with ego motion: ego alone 0.13, 4x10 grid 0.40, 8x20 0.43, un-pooled 16x40 0.46, centre crop at twice the resolution 0.45, the full higher-resolution grid 0.39.
* Outcomes: **UNPOOL-HELPS** (small: +0.027 R2 [+0.000, +0.056] over 8x20), **RES-FLAT** (the full higher-resolution grid is worse under this probe), **INTERFACE-PARTIAL** (+0.055 R2 [+0.000, +0.111] over 4x10, under the 0.10 bar).
* Lead present: the centre crop at twice the resolution is best, AUROC 0.84 against 0.77 for 4x10 (+0.066 [+0.033, +0.103], the same under both probes).
* Stop cue: **CUE-ABSENT-FULLRES**. No token set beats ego motion alone on the stop flag or the stop distance (131 stop windows).
* Control: a planted one-token object is found at AUROC 1.00 in un-pooled tokens and at 0.50 after 4x10 pooling, as theorem T5 predicts.

**One-scalar weight edit against a trained lever (addendum 4, BIASEDIT), MEASURED, T1-kin first tick, a diagnostic.** Source `bias/VERDICT_BIASEDIT.md`; eval-141, 117 stop-point windows; controls e1 to e3 pass.
* Outcome **BIAS-MATCHES-ROC**: adding one number to the stop-goal bias of the unweighted model reproduces what a full training run with a stop class weight does to the stop decision. At the offset that matches the trained false-positive rate, the hit rate is within the replicate gap in all four cells (two doses, two seeds).
* The weighted training runs moved the threshold along one ROC; the stop-goal AUROC stays 0.82 and 0.85. A dose now costs a scalar edit and seconds instead of a training run.
* What it cannot do: improve the ROC. A better stop decision needs a better cue.

**Pictures decoded from the world-model state (addendum 10, LATVIS), MEASURED, T0, a diagnostic.** Source `latvis/VERDICT_LATVIS.md`; 1,948 windows of 139 episodes; controls v1 to v3 pass. The pictures hold camera content and are kept private.
* What is retained, outcome **Z-BELOW-G**: the 128-number state of the world model decodes at skill 0.36; the 4x10 token grid it comes from decodes at 0.78. The state shows the scene gist (sky, road, light level) and no object; the token grid shows a truck, close cars and a street light as blobs.
* What is imagined at 2.0 s: the learned rollout is not closer to the true future frame than the decoded present; the stacked predictor is, at both seeds, by 0.6 to 0.7 of the available headroom (**IMAGINE-PIXELS** for the stack). The effect is small, because a mean-pooled state barely changes its picture in 2 s.
* Committed next lever: a spatial world-model state (the token grid, not its mean), with this read as its acceptance test.

**All thirteen reads of the programme so far are in.** Open: the six-seed hierarchy against flat replicate on the pod (addendum 11).
