# IMPL_CHOICES_LATSEM2: exact fields, units, NaN rules, probes and control wiring, fixed BEFORE any LATSEM2 statistic exists (agent, dev box, 2026-11-07)
Card: `../PREREG_V8F_LATSEM2.md`, frozen prefix 3,769 bytes, sha256 `1325f21c902967f3c86fdd2f9b56acf3ee28de01648d97496545cf6a130b1751` (verified before writing this file). Nothing here edits the card or moves a bar. Tier T0 throughout: a world-model diagnostic, never a driving claim.
What exists when this file is frozen: the label module `labels_latsem2.py` and its outputs (label counts only). No probe, rollout, R², SKILL or S3 value has been computed.

## 1. Inputs (sha256 of each file read)
| input | path | sha256 |
|---|---|---|
| level-1 latents, windows | `theory/wmact/latents.pt` | `7077ec717f633e0ef0814fec0a5fb43609e714bd8f521ccf6eb5e64ad02333a4` |
| L1 shrink factors c_k (TRAIN) | `theory/wmact/RESULT_WMACT_SHRINK.json` | `bb6df3a324cb0f3ac432ab3ca4604d19f7ea7f840591d4c66f377b768fd9d4eb` |
| label join, eval-141 (`lon_cls`, `has`, `lead_valid`, `off`, `n`) | `sfresh_inputs/code/.../2026-10-17-v8f-mini/raw/join_eval141.npz` | `f0ea722f59c1a4d8ba81ce7f9b204dd3a28950fc1c16778f63c2962786e5206a` |
| lead file, eval (keys `sha12, k, has, lead, gap, tg, s, v`; 26,394 rows, 139 clips, raw frame k = 0..199) | `v9labels/inputs/lead_eval139.npz` | `d59e1c74654f6279e9c578a5a2940dc6e0420e3c57f2431d06273e844022546f` |
| obstacle join, B1-eval agents, 3-D (26,394 lines, 139 clips, 905,512 cuboids; the output of `build_obstacle_join.py` plus `build_b1_agent_join_3d.py`) | `b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz` | `c863ee749acd86a4bbed5e3907f6bc5d5a559a9909eac16368eca71c022b8649` |
| native tracks (heading, row 2g) | `sfresh_inputs/inputs/native_tracks.pt` | `1db431f112b1bd7c426073c43c7df5fc75b2a032736c1e2020f02ffb118669e7` |
| code export (pinned tip, `tanitad.__file__` asserted under it) | `scratchpad/tip_1246c8dd` | commit 1246c8dd |
| label module / label arrays | `labels_latsem2.py` / `labels_latsem2.npz` | `92b673d1253d5c4aede3ea767003461b234e9b1a7fb25ac04935095ad69904fb` / `ff49e0350a59bf37db801f885630f7c4e545f7e3accb1277060fff3c92f53eed` |
Models: M2b s0 / s1 (`v8f-mini-hacc/M2b_s{0,1}_hacc.pt`), as LATSEM. The obstacle join and the lead file exist for eval-141's 139 episodes with a frame file; the 2 clips without a frame file are outside the bank as in LATSEM. Nothing is "not buildable".

## 2. Frame bank, windows, folds (LATSEM's rules, copied)
* Bank = every distinct (episode, cache step g) with frame_ok and g >= 0 among windows of the 139 episodes (g = t - 15 + f): **9,460 frames, 139 episodes**. Level 2 / 3 per-frame latents z2 = E2(z1), z3 = E3(z2), pointwise. Windows with all 16 frames real: **1,948**.
* Folds: `int(sha256(episode)[:8], 16) % 2`. Every probe is fitted on one fold and predicts the other, both ways (cross-fitted).
* Estimator: episode-cluster bootstrap, `wmact_lib.boot_weights` (B 2,000, seed 0) over the 139 episodes, ratios of per-episode sums; paired where two quantities share frames or windows. Episodes with no labelled frame for a target contribute zero to every sum.

## 3. Targets: exact fields (cache step g reads join row `off[clip] + g`, raw frame 2g)
Frame time: a cache step is 0.2 s, raw rows are 10 Hz, raw frame = 2g in the lead file (`k`), the agent join (`frame`, the RAW v2ep index) and `native_tracks` (row 2g). Alignment check, labels only: over 6,102 frames the in-corridor agent distance (agents ahead with |cy| < 1.5 m, `cx - l/2`) against the lead gap correlates 0.6525 at a 0-frame offset, 0.649 at +1 and 0.648 at -1 (peak at 0; no evidence of an offset beyond one raw frame). The agent join's key `frame` is raw (`frame_idx` = frame - 2 is not used).
| target | kind | definition (units) | NaN rule |
|---|---|---|---|
| `lead_present` | binary | the trainer's `lead_near` from `D.lead_label_arrays(J, lead_eval139.npz)`: `lead == 1`, gap finite, gap <= `LEAD_RANGE_M` (40 m). Not the release's `lead_valid`, which marks leads at any range | NaN where the lead file has no row for raw frame 2g (frames after raw 199): NO_LABEL, never "no lead" |
| `lead_gap` | continuous, m | `40 * lead_y` = the lead file's gap clipped to [0, 40], on frames with `lead_near` | NaN where there is no near lead |
| `lead_closing` | continuous, m/s | `(gap(g-2) - gap(g)) / 0.4` (positive = closing; backward difference over 2 cache steps = 0.4 s), both frames `lead_near` | NaN if g < 2, either frame not near, or **abs value > 20 m/s** (the lead file carries no track id, so a switch of lead appears as a jump; the label distribution has sd 7.31 m/s with them and 2.72 m/s without, 53 of 2,364 frames above 20 m/s) |
| `agents30` | continuous (count), agents | number of cuboids of the join at (clip, raw frame 2g) with centre distance `hypot(cx, cy)` <= 30 m in the ego frame of that frame, **all 11 class strings the join carries, any azimuth, any `occ`** (the card says 10 dynamic classes; the join has `automobile, person, rider, heavy_truck, bus, trailer, protruding_object, other_vehicle, stroller, animal, train_or_tram_car`, none dropped) | NaN where the join has no line for (clip, 2g) (frames after raw 199). An empty `agents` list is a label: count 0 |
| `nearest` | continuous, m | min over the same cuboids of `hypot(cx, cy)`, no cap, no transform | NaN where there is no line or no cuboid (count 0 has no nearest) |
| `lon7` | 7 classes | the v9 release `lon_cls` row of the join (LON9 index: 0 HOLD, 1 CREEP, 2 STOP, 3 FOLLOW, 4 DECELERATE, 5 ACCELERATE, 6 KEEP), the training label `lab_t` of the TOWN read | NaN where `has` is false or the class is the IGNORE value -100 |
| `turn` | binary | `abs(wrap(psi[2g + 40] - psi[2g])) > 0.3 rad`, psi = `native_tracks` column 2, 40 rows = 4 s | NaN where row 2g + 40 is past the track or psi is not finite |
Beside, decide nothing: `lead_closing_uncapped` (no 20 m/s cap) and `agents30_view` (the same count over cuboids with `occ == 0`, i.e. inside the 120 degree front-camera frustum: agents behind the car cannot be seen).
Label sources for the lead and the agents are the trainer's own arrays (`lead_has / lead_near / lead_y`) and the programme's obstacle join; the card's `lead_gap_h` and `s_h` are windows over the same files and are not used.

### n per target (labelled frames of the 9,460-frame bank; episodes with at least one labelled frame; fold 0 / fold 1)
| target | n frames | n episodes | by fold |
|---|---|---|---|
| lead_present | 8,874 (2,489 positive, 6,385 negative) | 139 | 4,057 / 4,817 |
| lead_gap | 2,489 | 56 | 922 / 1,567 |
| lead_closing | 2,311 | 54 | 859 / 1,452 |
| agents30 | 8,874 | 139 | 4,057 / 4,817 |
| nearest | 8,636 | 139 | 3,869 / 4,767 |
| lon7 | 8,834 (HOLD 95, CREEP 183, STOP 510, FOLLOW 1,720, DECELERATE 1,574, ACCELERATE 1,861, KEEP 2,891) | 136 | 4,079 / 4,755 |
| turn | 9,460 (1,414 positive, 8,046 negative) | 139 | 4,356 / 5,104 |
| beside: lead_closing_uncapped | 2,364 | 55 | 875 / 1,489 |
| beside: agents30_view | 8,874 | 139 | 4,057 / 4,817 |
Windows with all 16 frames real: 1,948. Of these, **505 (53 episodes) have a near lead at frame 0**: the S3 windows. Disagreement of the two lead flags on the 8,874 labelled frames (release `lead_valid` against the trainer's `near`): both 0: 4,650; valid 0, near 1: 70; valid 1, near 0: 1,735 (leads beyond 40 m); both 1: 2,419.
Label distributions (quantiles, labels only; `labels_latsem2.report.json`): lead_gap median 17.9 m, p5 3.5, p95 37.0; agents30 median 4, p95 30, max 145; nearest median 10.8 m, p95 70.4, max 190.0.

## 4. Probes (fixed now)
* Features: the level's per-frame latent (`initial_latents`), standardised on the FIT fold (mean, sd + 1e-8).
* Continuous targets (including the `agents30` count): ridge, lambda 10, intercept, exactly LATSEM's `ridge_fit`.
* Binary and 7-class targets: `sklearn.linear_model.LogisticRegression(C=0.1, solver="lbfgs", max_iter=5000, tol=1e-4)`, L2, intercept, no class weights, multinomial for `lon7`; prediction = `predict` (argmax). Non-convergence is counted and reported, never tuned.
* A fold in which a class has no fit frames predicts only the classes it has. No other choice is made after data is seen.

## 5. S1 (what a level ENCODES), per (seed, level l in 1..3, target)
* Continuous: R² = 1 - sum (pred - y)² / sum (y - fit-fold mean)², per-episode sums, bootstrapped as LATSEM (ratio of sums).
* Binary and 7-class: balanced accuracy minus the fit-fold majority baseline's. Balanced accuracy = mean over classes present among the evaluated labels of recall_c = TP_c / N_c, from per-episode TP_c and N_c sums (per bootstrap draw the classes with N_c > 0 in that draw). The baseline predicts the fit-fold's most frequent class on every frame of the other fold; it goes through the same function. The difference is paired (same frames).
* ENCODES (q, l): point > 0 and lower bound > 0 at both seeds.

## 6. S2 (what a level IMAGINES), per (seed, l, k), q
* Windows: the 1,948 with all frames real; for a target the windows whose label at the FUTURE frame is finite (n reported per row). The start-frame label is not required.
* Future frame f = k * stride(l) with stride {1: 1, 2: 2, 3: 4} frames and k in L1 {1, 5, 10}, L2 {1, 3}, L3 {1, 3}; the label is the bank frame (episode, G[window, f]). Longest horizon per level: L1 k10 (2.0 s), L2 k3 (1.2 s), L3 k3 (2.4 s).
* Query: `HJEPAStack.rollout` from the start state under the TRUE actions (`level_pack`, as LATSEM). **Decision queries: L1 shrunk `z0 + c_k (zhat - z0)` with `c_k` = `RESULT_WMACT_SHRINK.json` `rows["{tag}/L1/k{k}"]["c_train"]` (fitted on TRAIN), L2 and L3 raw.** Beside: L1 raw, the oracle (true future latent), and the start state.
* Decode with the probe fitted on the OTHER fold of the window's episode (`decode` of LATSEM), the same probe for the rollout and the start state.
* SKILL = 1 - sum err(query) / sum err(start state), err = squared error for continuous targets, **0-1 error for binary and 7-class targets** (the card names "binary"; the 7-class label takes the same 0-1 error). IMAGINES (q, l, k): SKILL > 0, lower bound > 0, both seeds.

## 7. S3 (does the imagined scene FOLLOW the action)
* Windows: the 505 with a near lead at frame 0. Constant raw actions over the window: accelerate (+1.5, 0), brake (-3.0, 0), plus brake rolled out a second time (control 2). Rolled out per level from the true start state (macro-actions of the same sequence above L1), read at the longest horizon (L1 k10, L2 k3, L3 k3), decoded by the cross-fitted `lead_gap` ridge probe.
* S3_l = mean over windows of [gap_hat(accelerate) - gap_hat(brake)], per-episode sums, bootstrapped as LATSEM `CF_LON`. **L1 decides with the shrunk rollout; L1 raw is beside.** (Both give the same sign and the same separation: the probe is linear, so shrinking multiplies the difference by c_k > 0.)
* ACTION-SEMANTIC-LEAD at l: point < 0 and upper bound < 0 at both seeds.

## 8. Controls (each must read its value, or the read is INVALID; nothing is tuned to pass)
1. **Start-state query reads SKILL exactly 0.0:** the start-state query's error is computed through the same function as the rollout query's, with the start-state decode as the query; every S2 row must give point exactly 0.0.
2. **Two identical action sequences give S3 exactly 0.0:** brake against its second rollout, per seed and level: point exactly 0.0, interval [0, 0].
3. **Permuted-label probes:** for every S1 row (7 decision targets plus the 2 beside targets, 3 levels, 2 seeds), a probe of the same kind fitted on labels permuted across episodes (LATSEM `perm_labels`: the fit-fold episodes with a finite label are put in a random order, each takes the previous episode's label sequence cycled to its length; no episode keeps its own), scored with the same metric on the other fold: its lower bound must not be above 0 on any row. One draw per (seed, level, target, fold) from `numpy.random.default_rng(0)` created once per seed and consumed in the order level 1..3, targets in the order of section 3 (decision targets, then beside), fold 0, 1. Beside, deciding nothing: the same permuted probes decoded through the S2 queries (rows reported as `S2_permuted`; any row separated above 0 is flagged beside the outcome that uses that cell), and, only if control 3 fails, 20 further draws per row (rng seeds 1..20) to show the control's own false-separation rate.
4. **Label-shuffled regression arm of S3:** the `lead_gap` labels of the fit-fold lead frames are permuted across frames (`default_rng(0)`, one draw per (seed, fold)) before the ridge fit; S3 is then computed as in section 7: its upper bound must not be below 0 at any (seed, level). Beside: 20 further draws (rng seeds 1..20), reported only.
5. Not controls, recorded: the label alignment check in section 3; the C1 / C2 checks of LATSEM (copies of one frame agree to 1e-5, per-frame z2 / z3 equal `HJEPAStack.encode` to 1e-5) are re-run on the same inputs and reported.
If any control fails the outcomes read `INVALID (control N failed)`; the numbers stay in the json under `uncontrolled` and are not decisions. If a committed value turns out to be mis-specified (for example float rounding), the run stops and the evidence goes to the handoff for the Master Mind to amend.

## 9. Outcomes (from the card, per level and target; no single verdict)
* SCENE-AWARE at l: ENCODES `lead_gap` OR ENCODES `agents30`, at both seeds; the result names which. Every other target's ENCODES is tabulated beside.
* SCENE-IMAGINING at l: IMAGINES `lead_gap` at l's longest horizon, both seeds. Every other (target, k) is tabulated beside.
* ACTION-SEMANTIC-LEAD at l: section 7.
* Next lever, as the card says: no level SCENE-AWARE: MRATE W1 then a scene-target auxiliary (own card); a level SCENE-AWARE but not SCENE-IMAGINING: the reliance arm on that level; MRATE W0 / W1 do not exist yet (`theory/mrate/` holds an empty `tree`), so no beside rows on them.

## 10. Pictures (decide nothing; seed s0; windows chosen by labels only)
* Retrieval for the S3 windows CLOSE (smallest lead gap at frame 0) and FAR (lead gap at frame 0 nearest the median over the 505): per level, now, and the nearest bank frame from OTHER episodes (squared L2 in the level's latent) of the imagined state under accelerate and under brake at the longest horizon (L1 shrunk); each cell titled with that frame's lead gap and `agents30` from the labels.
* Per-level PCA maps (levels 1 to 3) of the 9,460 frames: coloured by `lead_gap` (frames with a near lead) and by `agents30` (clipped at its 98th percentile).
* The strips embed downscaled camera frames: they stay on this disk and off the public repo; the PCA map holds no frame. The result lists each file's sha256.
* Outputs: `RESULT_LATSEM2.json`, `RESULT_LATSEM2.md` (tables generated from the json), `run_latsem2.py`, `run_latsem2.run*.log`, `fig/`.
