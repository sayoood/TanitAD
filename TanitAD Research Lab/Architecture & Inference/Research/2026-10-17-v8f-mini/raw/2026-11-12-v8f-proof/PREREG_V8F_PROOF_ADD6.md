# PREREG_V8F_PROOF addendum 6: HIERWHERE, where the hierarchy differs from its flat control, and whether the difference is selection or regression (frozen 2026-11-13 01:07:29 dev-box clock, before its data)

Theorem T2 (V8_THEORY.md; VC-B1 to VC-B4 in fv/FV_SPEC_V1.md) splits the risk of a planner into a within-mode term and a selection term, and its corollary says a hierarchy cannot beat a flat model of equal capacity through the within-mode term alone. The programme already holds the matched comparison: H-V8F-FLATDAG-1, DAG against FLAT-DAG (same recipe, O reads constants in place of the upper codes), outcome NOT ESTABLISHED: ADE6 difference -0.3451 [-0.4421, -0.2506] m at seed 0 and -0.0544 [-0.1605, +0.0547] m at seed 1 (MEASURED, T1-kin, RESULT_V8F_FLATDAG.md). This card reads the SAME banked rollouts by window class and by an exact decomposition. Zero training, zero new rollouts. Tier T1-kin on logged frames: a diagnostic, never driving performance. STRATEGIC: PAUSED (PI item 25). This read cannot change the outcome of H-V8F-FLATDAG-1; it locates the difference and tests the prediction of T2.

## Data (banked, dev box, read-only)
* DAG: E:/Projects/TanitAD-artifacts/v8f-dagger-t1/dump_DAG_s{0,1}_m10_causal_heads. FLAT-DAG: E:/Projects/TanitAD-artifacts/v8f-flatdag-t1/dump_FLATDAG_s{0,1}_m10_causal_heads. 141 episodes, 2,399 windows, paired by window. The banked read: raw/2026-10-27-flatdag/read/FLATDAG_READ.json and its reader raw/2026-10-27-flatdag/t1read/flatdag_read.py (the trajectory and the ADE6 definition are taken from that reader, not re-invented).
* Estimator: taniteval.ci paired episode-cluster bootstrap, B 2,000, seed 0, per training seed.

## Modes (pure geometry, one function, applied alike to the logged future and to each arm's trajectory; thresholds fixed now, before any arm trajectory was looked at)
* Longitudinal: STOP if the speed is below 0.5 m/s at some time in [2 s, 6 s] and stays below it to 6 s; else DECEL if v(6 s) - v(0) <= -2.0 m/s; ACCEL if >= +2.0 m/s; else KEEP.
* Lateral: LEFT if the heading change over 6 s >= +10 deg; RIGHT if <= -10 deg; else STRAIGHT.
* Joint mode = (longitudinal, lateral), 12 cells. A window is SELECTION-CRITICAL when the logged joint mode is not (KEEP, STRAIGHT); the rest is BULK.
* c_a(w) = 1 when arm a's trajectory has the logged joint mode on window w.

## Reads
* R1 WHERE: ADE6(DAG) - ADE6(FLAT-DAG) on SELECTION-CRITICAL and on BULK windows, their difference, and per logged class (four longitudinal, three lateral), each with n windows, n episodes and the paired interval.
* R2 SELECTION RATES: 1 - mean c_a per arm (longitudinal, lateral, joint) and the paired difference DAG - FLAT-DAG.
* R3 EXACT DECOMPOSITION of the ADE6 difference D = mean_w (e_DAG - e_FLAT) over all N windows into three sums that add to D exactly: WITHIN = (1/N) sum over windows where both arms have the logged mode; SELECTION = (1/N) sum over windows where exactly one arm has it; BOTH-MISS = (1/N) sum over windows where neither has it. Each with its paired interval. Also the same three terms for the 2-6 s ADE.
* Four families beside R1 on the two window groups: LONGITUDINAL (speed error, along-track), LATERAL (cross-track, heading, curvature, yaw rate), TACTICAL (R2 is the tactical read), STRATEGIC PAUSED.

## Prediction and outcomes (fixed now)
T2 predicts that the hierarchy's margin sits in the SELECTION term.
* SELECTION-CARRIES: at the seed where D is separated (seed 0 by the banked read), SELECTION < 0 with its interval excluding 0 and |SELECTION| >= |WITHIN|; and at seed 1 SELECTION <= WITHIN (point values).
* WITHIN-CARRIES: at seed 0, WITHIN < 0 with its interval excluding 0 and |WITHIN| > |SELECTION|. Then T2's reading of this comparison is not supported and section T2 of the theory must say so.
* MIXED: anything else, reported term by term.
* R1: CRIT-CONCENTRATED if, at seed 0, the difference on SELECTION-CRITICAL windows is more negative than on BULK with the paired difference of the two excluding 0; CRIT-FLAT otherwise.
No outcome here is a claim that the hierarchy beats flat: one training run per arm per seed, no replicate arm, so a separated interval is necessary and not sufficient. The seed floor F = 0.12075 m (the banked bar was 2F) and the tactical replicate floor 0.287 m are printed beside every term.

## Controls (each must read its known value, or the read is not valid)
* h1: the reader reproduces the banked ADE6 difference at both seeds, point and interval (-0.3451 [-0.4421, -0.2506]; -0.0544 [-0.1605, +0.0547]) to 1e-4 m.
* h2: WITHIN + SELECTION + BOTH-MISS - D, absolute value <= 1e-9 m.
* h3: the mode function on the logged future: its STOP class against the existing stop-window label (131 windows); agreement is reported; below 0.90 it is flagged as a definition gap, and R1 is then also given on the existing label.
* h4: swapping the arm labels at random per window (seeded) gives SELECTION and WITHIN intervals that include 0.
* h5 sensitivity, declared now: all thresholds times 0.75 and times 1.25. The outcome is decided at the stated thresholds; the two repeats are printed beside it and a sign change of SELECTION or WITHIN is named.

## What each outcome commits
* SELECTION-CARRIES: the dominance test is run where selection matters: evaluation and training weighted toward SELECTION-CRITICAL windows (theorem T8), with replicate arms, and the stop shield as the selection mechanism whose cue is the open item (addendum 5).
* WITHIN-CARRIES: the next read is which upper input carries regression information (code ablation at inference on the banked DAG arm), and T2's section is rewritten.
* MIXED or CRIT-FLAT: both follow-ups are sized by the measured terms.

## Limits stated in advance
A post-hoc partition of a banked comparison. Mode thresholds are a choice; h5 shows how much rides on it. The interval answers another draw of episodes. Deviations are recorded in the verdict, never by editing this card.
