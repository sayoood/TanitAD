# HIER_DEFS (HIERWHERE, PREREG_V8F_PROOF_ADD6 sha256 787f925f8853...; MEASURED, T1-kin on logged frames, CPU, zero training and zero rollouts)
Code: F:/TanitAD-scratch/v8proof/hier/hierwhere.py (line numbers below). Tests: test_hierwhere.py (14 tests, OK).

## 1 Data and trajectory
* Dumps (read-only): E:/Projects/TanitAD-artifacts/v8f-dagger-t1/dump_DAG_s{0,1}_m10_causal_heads and v8f-flatdag-t1/dump_FLATDAG_s{0,1}_m10_causal_heads, 141 ep*.npz each, loaded in sorted file order (hierwhere.py:44). Keys per file: g, ws, eid, clip_index, v0, cl, ol, fl, ha0.
* Trajectory behind the banked ADE6: `cl` [n,30,2], the executed T1-kin track, ego-frame metres, 30 knots at dt 0.2 s (knot k at t = (k+1) x 0.2 s, so 0.2 to 6.0 s). `g` = logged future, same grid. ADE6 = mean over the 30 knots of ||cl - g|| (t1k_read.py:49 `comps`, copied at hierwhere.py:109 and checked by h1).
* Window pairing: DAG and FLAT-DAG share clip_index, ws, eid, g, v0 bit for bit (asserted); 2,399 windows, 141 episodes per seed. g and the window set are identical across the two seeds.
* Estimator: taniteval.ci `_draws` (episodes with replacement, B 2,000, seed 0) via `boot` (hierwhere.py:130), the same draws for every statistic of a call; h1 also runs `paired_episode_cluster_bootstrap` itself.

## 2 Mode function (one function, hierwhere.py:58 `kinematics`, :77 `classify`; applied alike to g and to each arm's cl)
* Speed: step displacement with the origin prepended, speed[k] = ||p_k - p_{k-1}|| / 0.2 (four_families._seq_geometry; control: max abs difference 0.0). v(0) = speed[0], v(6 s) = speed[29].
* STOP: speed < 0.5 at some knot with t >= 2.0 s (index >= 9) and below 0.5 at every later knot (implemented as an all-suffix test; it reduces to speed[29] < 0.5). Else DECEL if v(6)-v(0) <= -2.0, ACCEL if >= +2.0, else KEEP.
* Heading change: heading of each step with ||step|| > 0.1 m (four_families MIN_DS_MPS 0.5 x 0.2 s), unwrapped over those valid steps, last minus first, in degrees (positive = counter-clockwise in the dump frame). LEFT >= +10, RIGHT <= -10, else STRAIGHT; fewer than 2 valid steps = 0.
* Joint cell = lon x lat (12 cells); SELECTION-CRITICAL = not (KEEP, STRAIGHT); c_a(w) = arm joint cell equals the logged joint cell. 2-6 s ADE = knots index 10 to 29 (t = 2.2 to 6.0 s).
* h5 scales all three thresholds (0.5 m/s, 2.0 m/s, 10 deg) by 0.75 and 1.25; the 0.1 m step-validity gate is not a threshold of the card and is not scaled.

## 3 Counts (logged, identical at both seeds)
SELECTION-CRITICAL 1,305 windows (110 episodes), BULK 1,094 (122). Longitudinal STOP 141, DECEL 401, ACCEL 457, KEEP 1,400. Lateral LEFT 219, RIGHT 429, STRAIGHT 1,751. Joint cells in RESULT_HIERWHERE.json modes.s0.logged_joint_counts.

## 4 Stop-window label (control h3)
RULE_DEFS item b: `stop_hit` in E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/stopbal/read/out/stopdec/stopdec_BAL-NULL_s{0,1}.npz (rows asserted equal to the dump's clip_index and ws; 131 label windows). Its source rule: g_tac_geom.py stop_point (tree F:/TanitAD-scratch/v8proof/res/tree/stack/tanitad/data/g_tac_geom.py:433-473).

## 5 Deviations and reading choices (none moves a threshold, bar or outcome rule)
* R1 Tool tree: the pinned tree E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a (the FLATDAG read's own tool root; ci.py sha 38db571e, four_families.py sha 96839329 per v8f-dagger-t1/PINS.txt) is imported, not F:/.../res/tree, whose ci.py (sha 528276c5) differs by added functions only. No tree edited. h1 reads 0.0 deviation.
* R2 The banked flatdag_read.py (783 lines, pins and a pinned chain of readers) is not run whole; ADE6 and the family components use a copy of t1k_read.comps on the same dump arrays, and the interval comes from taniteval.ci itself. h1 reproduces both seeds to 0.00e+00 m (point and both bounds, 4 dp).
* R3 v(0) is the first-segment speed of the trajectory (not the dump v0), so the rule needs no input beyond the trajectory; on the logged future it differs from dump v0 by 0.098 m/s on average (mean absolute).
* R4 The card does not fix the heading-change estimator or the knot-time convention; section 2 fixes them before any arm trajectory was looked at (arm trajectories entered only through the stated function).
* R5 h3 agreement is defined as the Jaccard overlap of the two STOP sets (accuracy over all windows is printed too, it is inflated by the 94 % non-STOP majority). It reads 0.314, below 0.90: flagged as a definition gap and R1 is also given on the existing label (VERDICT section h3). Measured reasons, on the logged future only: (a) the label code excludes an ego already at rest when the band opens (g_tac_geom.py:459-466) while the card rule keeps it (63 % of the mode-STOP windows that the label lacks are already below 0.5 m/s over the step ending at t = 2.0 s); (b) 66 label STOP windows still move at 6 s on g (speed[29] 0.53 to 5.7 m/s), and the label file's own v_end_gt on label windows has median 0.73 and maximum 6.54 m/s, so the label is not a pure at-rest-at-6-s function. The cause of (b) was not traced further.
* R6 h4: the card says "seeded" without a seed. Swap seed 0 is the control as run. It FAILS at seed s0 (SELECTION interval [+0.0015, +0.0523] excludes 0; chance rejection, since a random-sign swap makes the point a random draw and the interval does not know that). 19 further swap seeds are printed as information only: they do not replace the control. By the card the read is then not valid until h4 reads its known value; the exit code is 2.
* R7 "At the seed where D is separated (seed 0 by the banked read)" is taken as seed 0, confirmed by the h1 reproduction (s1 D is not separated).
* R8 R1's paired difference of the two groups is the statistic mean_crit(d) - mean_bulk(d), resampled over episodes (the two groups hold disjoint windows of the same episodes).
* R9 Family rows with a NaN mask (heading, curvature, yaw rate on windows with fewer than 2 valid steps) average over the windows that have a value; n windows is the finite count.
* R10 Window classes are defined on the logged future; the BOTH-MISS, SELECTION and WITHIN sets are also defined by each arm's own rollout, so they are post-hoc partitions (card limits).

## 6 Controls: values in VERDICT_HIERWHERE.md and RESULT_HIERWHERE.json
h1 PASS both seeds; h2 PASS (residual 0.0); h3 flagged (R5); h4 FAIL at s0 under swap seed 0 (R6); h5 declared, printed.
