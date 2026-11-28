# PREREG_FROZEN: H-KAPPA-T1-1, the lever against the residual plan-curvature excess after S (gate G-KAPPA, then one T1-kin arm)

Frozen before any gate number exists (sha256 and wall clock in `FROZEN_SHA.txt`). Zero training, dev-box CPU. Tier **T1-kin** (own executed state, logged cameras; a self-action open loop). Evidence class of every number: **MEASURED (ours; artifacts in `MANIFEST.txt`)** unless marked ESTIMATED. The Master Mind registers the claim row; this card does not.

**Question.** After S (SG(7, 2) on the residual over P10, H-SMOOTH-T1-1), v8F still reads curvature MAE, L2 curvature MAE and M1 crawl share separated-worse against P10 at T1-kin. Which lever removes that excess, and does it reach DOMINANT?

## Part 1. Gate G-KAPPA (offline, no rollout; decides the lever)

**Data.** The four matched-thread (16) S runs `F:/TanitAD-scratch/smooth_t1/out_m16/SMOOTH_{DAGOACC,SOUP}_s{0,1}` (dump `cl` = X's executed path, `fl` = the P10 floor's executed path, `g` = GT; 2,399 windows / 141 episodes). Read-only; the gate copies nothing into `smooth_t1/`.

**Row.** Curvature MAE per window exactly as Read A: `_seq_geometry(., dt=0.2)` (taniteval `four_families`), mean of |kappa_path - kappa_GT| over the steps where path and GT are both pair-valid (step length > 0.1 m). E = window mean of (row_X - row_P10) over windows finite on both = the Read A X - P10 point.

**Exact additive partition of E** (per window w, step k: c_wk = m_Xk |e_Xk| / n_X - m_Pk |e_Pk| / n_P; sum over k = Delta_w). Every (w, k) goes to exactly one class, by X's own path, in this precedence:
* **DIP**: all steps of a window whose X path has a step speed < 0.5 m/s while the GT path has none (an unwarranted dip).
* **HIGHK**: otherwise, a step where X is pair-valid and |kappa_X| > KAPPA_MAX = 0.2 1/m (`feasible_decode.KAPPA_MAX_1PM`).
* **REST**: every other step (|kappa_X| <= 0.2, or X not pair-valid there).
Shares s_C = E_C / E.
**Frequency.** f_LF = E_lo / E, where E_lo is E with X's executed path replaced by its low-pass `savgol_filter(cl, 7, 2, axis=1, mode="interp")` (the S operator's window and order, applied to the whole executed path, across the replan seams; P10 and GT untouched).

**Decision, on DAG-OACC (the headline arm), committed now; first match wins.**
* **(a)** s_HIGHK >= 0.5 at both seeds => the curvature-bounded decode: `project_feasible` verbatim (box only, `mu=None`, `a_max` 4.0, `kappa_max` 0.2, its default margin and stop epsilons, `clamp_entry=False`) on the origin-prepended plan at the plan's own grid dt = 0.2 s, composed after S.
* **(b)** s_DIP >= 0.5 at both seeds => the accel-bounded longitudinal decode from the same envelope: `project_feasible(kappa_max=inf, a_max=4.0, mu=None, clamp_entry=True, v0 = executed speed at the tick)`, composed after S.
* **(c)** s_REST >= 0.5 and f_LF >= 0.5 at both seeds (a low-frequency overshoot in ordinary steps) => a lateral residual shrink `P + diag(1, alpha)(S(plan) - P)` in the tick frame, alpha chosen on TRAIN windows only (never on eval-141); if the tool cannot roll out TRAIN windows, that is the named blocker and no eval number is produced for (c).
* **(d)** otherwise => the lever of the larger of mean(s_HIGHK) and mean(s_DIP) over the two seeds ((a) on a tie); (c) if both are <= 0.
* If E <= 0 at a DAG-OACC seed, the gate is REFUSED (there is no excess to split) and nothing is launched.

**Gate known answers (before the decision is read).** KG1 the row computed here for the BANKED Y and P10 equals the landed per-window arrays `dagland/gapdecomp/raw/arrays/CTRL_s0.npz` (`a_9`, `b_9`) to <= 1e-9. KG2 the partition sums to E to <= 1e-12 at every run. KG3 an X path against itself gives E = 0 and every share term 0.
**Beside, decides nothing:** the same partition for yaw-rate MAE; per-class window counts; the 3-8 m/s band split; the share of HIGHK carried by steps under 2 m/s; rho_a = the fraction of E removed by clipping kappa_X to +-0.2 on the executed path (no re-integration); SOUP's shares.

## Part 2. The arm (the selected lever only)

**Tool.** The selected lever as a new flag of the SMOOTH-T1 tool copy (`--feasible {off,box,lon}` for (a) / (b); `--lat-shrink` for (c)), default off, bit-identical when off (returns the plan object), applied to the issued plan at every replan after S; the P10 floor is never touched. A test goes RED when the lever call is deleted.

**Arms.** X = S + lever, Y = S (rerun fresh, same tool file, same 16 threads, one run at a time). DAG-OACC s0 / s1 (`v8f-dagoacc/M2b_s{0,1}_dagoacc`, ckpt sha256 `67265ec1…` / `90b01474…`) and SOUP s0 / s1 (`dagland/soup/ckpt/M2b_s{0,1}_soup`, `238aab03…` / `81924b1a…`), each in the source tree of its banked run. Settings as SMOOTH-T1: `--replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal --dump-ticks --dump-heads --device cpu --smooth sg72`, batch 32, eval-141.

**Reader and estimator.** The SMOOTH-T1 reader's pure layer (`read_a`, `read_xy`, `arm_outcome`, imported read-only from `dagland/smooth_t1/code/smooth_t1_read.py`) on the 19 rows of `extract_smooth.py`, with Y = the fresh S run. Paired episode-cluster bootstrap, B = 2,000, seed 0 (`gapdecomp.decompose`); it answers *another draw of episodes* only. The planner is deterministic; each arm is one training run per seed. Rows reported for X - P10, X - Y, Y - P10, all four families (STRATEGIC is not paused for v8F), PRIMARY ADE6 and the M1 crawl row.

**Known answers.** KA1 Y's dump and ticks equal `out_m16/SMOOTH_*` bit for bit (Y reproduces S's X at 16 threads); else max abs deviation stated, and the read is REFUSED above 1e-4 m. KA2 on X's ticks, `plan` equals lever(P + SG(plan_raw - P)) recomputed offline (<= 1e-4 m); the floor `fl` equals Y's. KA3 pytest: off returns the same object; a feasible plan round-trips (< 1e-5 m); a planted high-curvature plan comes back inside the envelope in the projection's own convention; deleting the call turns the test RED.

**Outcomes (per arm, both seeds; precedence DOMINANT, HARMS, MOVED, NO EFFECT).** DOMINANT iff at BOTH seeds X - P10: D1 ADE6 separated-better with point <= -0.2415 m; D2 no row of 19 separated-worse (direction-aware); D3 a separated-better row in each of the four families. HARMS iff at EITHER seed ADE6 X - Y point > +0.0604 m or one of the 15 non-crawl rows reads X - Y separated-worse. MOVED iff not DOMINANT and at BOTH seeds every one of the four crawl rows (yaw-rate MAE, curvature MAE, L2 curvature MAE, M1 crawl share) reads X - Y separated-better with ADE6 X - Y point <= +0.0604 m. NO EFFECT otherwise. Headline arm DAG-OACC; SOUP stated separately.

## Predictions (ESTIMATED, written before any number) and the next arm per outcome (Rule Zero)

**Gate.** From the heavy window tail (top 10 % of 3-8 m/s windows carry 0.55 / 0.49 of the |kappa| sum, H-GAPDECOMP-M-1) and the post-S per-step |kappa| of 0.1-0.33 at steps 9-29 (SEAM_DIAG): (a) 55 %, (b) 25 %, (c) 10 %, (d) 10 %.
**Arm, if (a).** The clamp caps the tail, not the ordinary steps: curvature MAE X - Y separated-better at both seeds; M1 and L2 move less. DOMINANT 10 %, MOVED 40 %, NO EFFECT 30 %, HARMS 20 % (re-integration moves positions).
* DOMINANT: re-read on a fresh seed or window set before adoption; the training lever is the envelope as the decoder's output layer (the PI's go).
* MOVED: name the rows still separated-worse vs P10 per seed; next = the gate's next-ranked class lever composed on top ((b) for DIP, (c) for REST).
* NO EFFECT: next = (c), with alpha on TRAIN windows (or its named blocker).
* HARMS: next = the kappa-only projection (`a_max` = inf) to attribute the harm to the accel or the curvature clamp.
**If (b), (c) or (d) is selected,** the same four outcomes map to: DOMINANT as above; MOVED and NO EFFECT => the gate's next-ranked lever; HARMS => the lever without `clamp_entry` (for (b)) or alpha one step nearer 1 on the TRAIN grid (for (c)).

**Not decided here.** Adoption into any K-1 / WM-1 claim; the thread-count question of SMOOTH-T1 (this card runs everything at 16 threads and compares only runs made here or in `out_m16`). No class, threshold, operator, row or arm changes after data.
