# PREREG_V8F_CKPT120 (FROZEN 2026-11-16 by the MM, NOT RUN): S sets and follows the route at 120 m, the strategic layer's first output in the 6-30 s band

Status: drafted 2026-11-16 by the Architecture agent for the MM, before any arm is built or trained and before any 120 m output exists. Nothing here is launched. FROZEN by the MM 2026-11-16 ~22:55Z with the rulings in `MM_RULINGS_CKPT120.md` (binding, same weight as this text; ruling 9 reorders the arm priority of section 5). sha256 in `PREREG_V8F_CKPT120.sha256`. It is STRAT1's committed next step on **S-PARTIAL** (`PREREG_V8F_STRAT1.md` section 6, decision 1: "The 6-30 s band (a 120 m checkpoint) is the next card and an architecture change, so it needs a PI go"). PI decisions: `QUESTIONS_CKPT120.md`.
Evidence classes: MEASURED (M, artifact named), PUBLISHED (P), INHERITED (I), ESTIMATED (E), HYPOTHESIS (H), UNVERIFIED (U). Tiers: T0 (diagnostic), T1-kin (primary).
Design artifacts (this folder, `F:/TanitAD-scratch/strat1/cards/ckpt120/`): `CKPT120_COVERAGE.json`, `CKPT120_POWER.json`, `STRAT1_80M_STRATUM_SPLIT.json`, `STRAT1_80M_STRATUM_CI.json`, `eval141_grid_truth.npz`, and the four scripts that wrote them (section 11).

## 0. Decisions for the MM (escalated) and what the design measurement changed

1. **One lever per arm, as a two-rung ladder.** H1 = S1 + a 120 m checkpoint (an S OUTPUT). H2 = H1 + S reads the route text's next turn at any distance (an S INPUT). Each arm differs from its comparator by one lever (H1 vs S1, H2 vs H1).
2. **30/50/80 m stay unchanged** (same head, same loss, same weight). The 120 m checkpoint is read by the loss and the evaluator only. It is **not** an input of T or O in this card; feeding it to T is a later card.
3. **H2 runs in the same chain, unconditionally, after H1.** The design measurement (section 2) predicts H1 falls short for a structural reason that only H2 addresses. Waiting for H1's read to launch H2 would spend a PI round-trip on a predicted outcome (RULE ZERO).
4. **The deciding constant contrast changes form, before any 120 m number:** balanced level − 0.5 separated AND the STRAT1 raw contrast's point estimate > 0. The STRAT1 raw form (real − best constant, separated) is reported as written beside, and is the STRONG label. Reason: at n = 27 episodes with a 79 / 262 side split, the raw form separates a 90 %-accurate follower with probability 0.26-0.38 (E, section 2.4). STRAT1's outcome at 80 m stands as written; nothing in this card re-reads it.
5. **Floor F = the larger of two replicate pairs**, |H1_rep − H1_s0| and |S1_s10 − S1_s0| (S1_s10 is in rep's form, `strat1_check.py:36-37` at `a4357e79`, M; training now). STRATEGIC 120 m cells have only the first pair.
6. **Label build on the train release of record** (md5 `83e30bd3...`, I from `sfresh_inputs/inputs/v9/MD5SUMS.txt`; not on the dev box). The coverage below is measured on the 2026-10-04 release (md5 `f63ece41...`, 4,369 clips); see section 2.2.
7. **Gate route:** the STRAT1 study checker extended with kinds H1, H2, R1, R2, minting `ckpt120-gate-token-v1` bound to commit, argv sha256 and every data sha256 including the RC-A120 table, with the runner refusing without a matching token (the PI rule of 2026-09-26, as OBD, ACTSENS and STRAT1).
8. **For the register (escalated):** STRAT1's pre-written S-PARTIAL reason ("inside 80 m the scene already shows the turn") is not what the stratum split shows (section 2.3). The outcome stands as written; the mechanism row should carry this card's beside diagnostic.

## 1. Question

Fine-tuned on the STRAT1 S1 recipe, does S, given a 120 m route checkpoint (H1) and then the route text's next turn at any distance (H2), place its 120 m checkpoint on the announced side of the route beyond what any constant does, set the 120 m point closer to the driven route than its own 80 m checkpoint carried forward, and keep the driving of all four families?

## 2. What the design measurement found (CPU, labels and logs only, zero training, no 120 m model output exists)

### 2.1 RC-A at 120 m reproduces the label builder exactly
* RC-A at arc L is `rc_point` on the 8 m-smoothed egomotion path, valid where `s_now + L <= s_hi_valid` and the chord sagitta over `[s_now − 24, s_now + L + 24]` m is at most 0.25 m (`build_v9_labels.py:400-422` at `a4357e79`, git blob `edf5424d...`). The RC-A120 label is the same function at L = 120 m. Labels may use the ego future (PI 2026-08-03).
* Recomputing 30/50/80 m with this function reproduces the eval join's `rc_ok` with **0 mismatches** over 2,365 labelled windows (max |Δxy| 3.8e-06 m, max |Δψ| 2.3e-05 deg), and the 2026-10-04 train release's `rc_A*_valid` with **0 mismatches** over 436,667 labelled steps (Δ exactly 0.0). The vectorised sagitta equals the builder's `chord_sagitta_max` on all 14,190 eval queries (M, `CKPT120_COVERAGE.json` `CONTROL_R1_eval`, `CONTROL_R1_train`, `CONTROL_R2_sagitta`).
* The 80 m scorable set reproduces the STRAT1 reader: **244 windows / 25 episodes, best constant 0.7664** (M, `CONTROL_R3_reader_80m`). The STRAT1 card's 214 / 24 and 0.794 were INHERITED; the reader's and this count agree.

### 2.2 Coverage

| | 80 m | 100 m | 120 m | 150 m |
|---|---|---|---|---|
| eval: RC valid / labelled windows | 0.970 | 0.960 | **0.956** (2,261 windows / 135 episodes) | 0.944 |
| eval: S_CKPT-scorable windows / episodes | 244 / 25 | 296 / 25 | **341 / 27** | 383 / 29 |
| eval scorable: left / right | 57 / 187 | 70 / 226 | **79 / 262** | 94 / 289 |
| eval scorable: side reaches S (tactical token on) | 209 | 218 | **219** | 219 |
| train: RC valid / labelled steps | 0.956 | 0.947 | **0.938** (409,786) | 0.924 |
| train: \|ψ\| >= 25 deg among RC-valid steps | 0.271 | 0.302 | **0.328** | 0.361 |
| train text-turn scorable steps (left / right) | 55,409 (26,847 / 28,562) | 62,719 | **69,236 (33,962 / 35,274)** | 77,676 |
| train scorable: side reaches S | 46,595 | 47,367 | **47,137** | 46,332 |

(M, `CKPT120_COVERAGE.json`. Eval = the 2,399-window eval-141 grid, 34 of its windows have no v9 row. Train = the 4,572-clip cache joined to the 2026-10-04 release: 203 clips are absent from it, U for those; `--exclude 540:600` moves the 120 m rate by 0.0004.)
* The 120 m point is a 6-30 s target: 120 m / v_now lies in 6-30 s on 72.8 % of labelled eval windows (median 11.75 s) and on **304 of the 341 scorable windows (27 episodes)**; 0 lie below 6 s and 37 (10 episodes) above 30 s (M). The future log covers 15 s on 97.6 % (eval) / 96.7 % (train) of steps and 30 s on 95.2 % / 91.2 % (M). A time-based target is therefore not needed for this card; the 120 / v split is reported beside.
* **Denser windows do not add episodes.** Every labelled eval step instead of the grid gives 1,943 scorable steps but only 32 episodes at 120 m (M, `dense_all_labelled_steps`). The grid stays the read basis; no n is inflated.

### 2.3 Where STRAT1's 80 m shortfall sits (beside diagnostic, re-read of the banked STRAT1 T0 passes, no bar)
* The nav text token carries the side only while `d <= max(30 m, 6 s x v)` (`BUILD_SPEC.md` section 5 step 5, I; on eval the largest announced d is 86.3 m, M). Beyond that S sees FOLLOW with a distance.
* S1 at 80 m, T0, real nav (M, `STRAT1_80M_STRATUM_SPLIT.json`; levels reproduce `RESULT_STRAT1.md` exactly, e.g. S1_s0 0.8443):

| arm | side reaches S: n 209 (constant 0.761) | side does not reach S: n 35 (constant 0.800) | zero nav, all 244 |
|---|---|---|---|
| S1_s0 | **0.943** | **0.257** | 0.180 |
| S1_s1 | **0.933** | **0.400** | 0.197 |
| N1_s0 / N1_s1 | 0.947 / 0.933 | 0.257 / 0.400 | 0.193 / 0.189 |

* On the 209 windows where the side reaches S, S1 real − best constant is **+0.182 [+0.018, +0.380]** (s0) and **+0.172 [+0.012, +0.369]** (s1), separated at both seeds (M, `STRAT1_80M_STRATUM_CI.json`, paired episode-cluster bootstrap B 2,000 seed 0). This is a post-hoc stratum on seen data: it is beside evidence for ranking the levers, never a result.
* Reading: the shortfall is in the 35 windows where the side never reaches S, and with zero nav S turns on 18-20 % of windows. Inside 80 m, vision alone does not set S's side. The binding term is the nav announcement horizon, not the camera's horizon (H, consistent with every row above).

### 2.4 Can the bar be passed at 120 m? (labels only)
* **K-PERFECT** (a planted follower that is right on every scorable window): real − best constant **+0.232 [+0.090, +0.406]**, separated (M, `K_PERFECT_minus_best_constant`). The bar is passable at n = 27.
* **K-NAVINF** (right exactly where the tactical token carries the side, straight elsewhere: the ceiling of a perfect follower of S1's own nav input): **−0.126 [−0.301, +0.080]**; balanced − 0.5 **+0.148 [−0.014, +0.300]**, neither separated (M). Of the 97 scorable windows with d > 60 m, the token carries the side on **10** (M). ⇒ H1 is predicted to fall short of the bar for this reason (H).
* **Power** (planted follower correct with probability p per window, independent or episode-clustered, 100 simulations x B 500; E on the M window structure, `CKPT120_POWER.json`): at 120 m, P(separated) for the raw form is 0.26-0.38 at p = 0.90 and 0.84-1.00 at p = 0.95; for balanced − 0.5 it is 1.00 at every p >= 0.80. At 80 m the raw form separates a 90 % follower with probability 0.14-0.26. ⇒ the raw form decides only for a near-perfect follower at this n (decision 4).

## 3. Admissibility

* **Goal-input rule (CLAUDE.md 2026-08-03).** The 120 m checkpoint reaches neither T, O nor any goal path (test T-ISO-FAR: perturbing `s_far_out` leaves every T and O output bit-identical). H2's strategic token is route text (GPS matched to OSM routing, I), built offline from the text and odometry like STRAT1's nav; it carries no situation-classifier output, and `ctx_logits` never reach S (STRAT1's T-ISO, I).
* **Labels may use ego; inference is vision-only (PI 2026-08-03).** RC-A120 is derived from the ego future (labels). The far head adds no input: it reads S's hidden state, whose inputs are the recipe of record (adapted camera tokens, measured ego motion per the PI yaw-rate ruling, nav, limit). H2 adds the route text, which is the nav input (PI: nav is an input at training and inference).
* **Optimistic by construction.** The text is computed along the drive's own trace; every STRATEGIC number is worded "with the route supplied along the driven path". H2's "next turn at any distance" is the same text with no announcement cut, so the same wording applies.
* Parity: corpus `train4572 --exclude 540:600` and eval-141, unchanged (as STRAT1).

## 4. The levers (built default off; every flag has a G-DVB entry; defaults off = the `a4357e79` build bit for bit)

| flag | default | arm value | what it builds |
|---|---|---|---|
| `--s-far` | off | on | `s_far_out = Linear(512, 3)` on the hidden layer of `head_s` (1,539 params), last layer zero-initialised so the far point starts at the nominal (120, 0, 0). Output `ckpt_far` = (120, 0, 0) + o x `ck_scale`, in S's tick frame. Key prefix `s_far_out.` joins `NEW_LEVER_KEY_PREFIXES`, so an M2b init loads with exactly these keys fresh. |
| `--w-ckpt-far` | 0.0 | 0.3333 | `T["ckpt_far"]` = the masked mean of \|Δx\|/10 + \|Δy\|/10 + \|Δψ\|/0.5 against RC-A120, mask `rc120_ok & has` (the form of `T["ckpt"]`, `v8f_mini.py:1203-1206`). 0.3333 = `w_ckpt / 3`: `T["ckpt"]` is a mean over three arcs, so the fourth arc gets one arc's weight. Fixed here, not tuned. G-DVB refuses `--s-far on` with weight 0 outside the reachability control. |
| `--s-far-lr-mult` | 1 | 10 | the fresh layer's lr (1e-3 inside the lr-1e-4 fine-tune): STRAT1's fresh-head convention (`--ctx-lr-mult 10`). Fixed, not tuned. |
| `--rc120-npz`, `--rc120-sha256` | "" | the table | RC-A120 per cache step (train and eval-141), built by the builder's own `rc_point` + sagitta rule at L = 120 m (section 2.1). |
| `--s-far-label-perm` | off | clip (R1 only) | RC-A120 of each train clip replaced by a fixed derangement of clips (seed 0), steps beyond the donor's length masked. The deliberate regression of the new head. |
| `--nav-strat` | off | on (H2) | S reads ONE extra conditioning token `e_nav_s(nav_s)`, `Linear(5, d)`, key prefix `e_nav_s.`. `nav_s` = STRAT1's text rule (`BUILD_SPEC.md` section 5) with step 5 set to d_ann = ∞: the text's next TURN-class side at any distance, countdown distance, ch3 = clip(d / 100, 0, 3), ch4 = 1. T and O never read it (key guard). In T1 it is dead-reckoned like the tactical channel. |
| `--nav-strat-variant` | real | donor (R2 only) | at TRAIN, S's `nav_s` comes from a fixed derangement of clips (seed 0); T, O and the tactical nav keep the real text. |
| `--s-adapter-detach` | off | on (COSTS branch only) | S's reader reads the adapter tokens detached, so no S-side loss moves the shared adapter. |

Unchanged: `CKPT_ARCS` (30, 50, 80), `head_s` (9 outputs), `e_ck`, the p_gt teacher forcing of the three checkpoints into T, `--detach-upper`, the ctx heads, every DAgger setting.

## 5. Arms (LoRA pod, one job at a time, 3,000 steps each from M2b, matched to the base)

Base of record = **STRAT1 S1** (text nav countdown, ctx heads on, `w_ctx 0.5`, ctx lr x10) on the OBD-CTL recipe (DAG-OACC 3,000 steps from M2b, lr 1e-4, warmup 100, p_gt 0.2, DAgger p 0.1, `--w-acc 49.6969`, `--detach-upper`), commit `a4357e79` on build base `cdbff971` (I, `PREREG_V8F_STRAT1.md` section 3). The new build branches from `a4357e79`. The banked S1_s0 and S1_s1 are the base; they are not retrained.

| priority | arm | = | init / DAgger bank / seed |
|---|---|---|---|
| base | S1_s0, S1_s1 | banked (STRAT1 chain) | M2b_s0 / s0 / 0; M2b_s1 / s1 / 1 |
| base floor | S1_s10 | banked when the SEEDS arms land | M2b_s0 / s0 / 10 (rep's form) |
| 1 | **H1_s0** | S1 + `--s-far on --w-ckpt-far 0.3333 --s-far-lr-mult 10 --rc120-npz <t> --rc120-sha256 <s>` | M2b_s0 / s0 / 0 |
| 2 | **H1_s1** | as H1_s0 | M2b_s1 / s1 / 1 |
| 3 | **H1_rep** | as H1_s0, only `--seed 2` | M2b_s0 / s0 / dagger 0, seed 2 (the training-run floor) |
| 4 | **R1_s0** | H1_s0 + `--s-far-label-perm clip` | M2b_s0 / s0 / 0 (must fail) |
| 5 | **H2_s0** | H1_s0 + `--nav-strat on` | M2b_s0 / s0 / 0 |
| 6 | **H2_s1** | as H2_s0 | M2b_s1 / s1 / 1 |
| 7 | R2_s0 | H2_s0 + `--nav-strat-variant donor` | M2b_s0 / s0 / 0 (must fail) |

* Priority order, not a dependency chain: a chain stopped after arm 3 still yields the H1 read with its floor; after arm 6, the H2 read.
* **Base-explicit rebase rules** (amendments, fixed now): (a) if STRAT1's SEEDS-DISAGREE extension reads COSTS-HEADS (S1 − C0 at >= 3 of 4 seeds), the base becomes N1 (banked at s0/s1) and H1/H2 re-run on it; (b) if the data-scale card reads DATA-HELPS and the PI rebases (data-scale Q3), H1/H2 re-run on the union base, which needs RC-A120 on the new clips (they need v9 rows first: the data-scale label-coverage question). Each read keeps its own base.

## 6. Read

**Tiers.** STRATEGIC route cells: **T0 at the window's first tick** (fresh ages, as STRAT1), equal to T1 tick 0 under K9b; the T1 per-tick row is reported beside. LONGITUDINAL, LATERAL, TACTICAL: **T1-kin** (self-action open loop, m = 10, causal bucket, dead-reckoned nav and `nav_s`, logged scene), eval-141, 2,399 windows / 141 episodes, per seed. Detected info: T0.

**Estimator.** `taniteval.ci.paired_episode_cluster_bootstrap`, B 2,000, seed 0 (`taniteval/taniteval/ci.py` at `a4357e79`, blob `4f17fe11...`). The balanced cell uses the same episode draws (`ci._draws`) on the balanced statistic. Three questions, named: another draw of EPISODES (the bootstrap); another TRAINING RUN (floor F); another INFERENCE RUN does not apply (deterministic planner; every arm dumped in one environment at one thread count).

**Floor.** F per cell = max(|H1_rep − H1_s0|, |S1_s10 − S1_s0|) at seed 0, applied at both seeds. If S1_s10 is not banked at read time, F = |H1_rep − H1_s0| and the read says so. STRATEGIC 120 m cells: F120 = |H1_rep − H1_s0| on that cell.

**Cost / gain.** As STRAT1: a cost is separated on the worse side with |Δ| > F; a gain is separated on the better side with |Δ| > F.

### 6.1 STRATEGIC route following at 120 m: S_CKPT120 (deciding; H1 and H2 each, both seeds)
* S_CKPT120 = 1 when the far checkpoint's heading is on the announced side by >= theta (25 deg; SLIGHT 10 deg). Scorable where the text announces a TURN-class side at the window's tick (flag 0), d_text (the navtext countdown) + 20 m <= 120 m, and RC-A120 is valid: **341 windows / 27 episodes, left 79 / right 262** (M). COMPLETE_M 20 and the thetas are STRAT1's (`strategic_cells.py ckpt_cell`, I).
* Per arm and seed: real, donor (`donor_indices` seed 0; tactical nav and `nav_s` swapped together) and zero (both zeroed).
* Contrasts: **(a) real − donor** (paired); **(b) balanced − 0.5**, balanced = the mean of the left and right levels (any constant reads exactly 0.5); **(c) raw = real − best constant** (STRAT1's form; best constant = always the majority side among the scorable windows = right, 262/341 = 0.768; right is also the train majority at 120 m, 35,274 vs 33,962, M).
* Strata, fixed now, reported with n and no bar: side reaches S by the tactical token (219 / 24) or not (122 / 22); near d <= 60 m (244 / 25) or far d in (60, 100] m (97 / 20); 120 / v in 6-30 s (304 / 27) or above 30 s (37 / 10). Per-side levels are reported for every row.
* Beside: S_CKPT at 30/50/80 m for H1 and H2 in all three forms (continuity with STRAT1); COMPLETE_M 10 and 30 sensitivity rows; the raw text distance row.

### 6.2 STRATEGIC route setting
* **At 120 m:** position error (m) and heading error (deg) of the far checkpoint against RC-A120, on every window with RC-A120 valid: **2,261 windows / 135 episodes** (M). Paired baselines: **EXTRAP** = the same arm's 80 m checkpoint carried 40 m straight along its own heading (what S already said, taken to 120 m); STRAIGHT = (120, 0, 0). The cells are H − EXTRAP. S1's EXTRAP error is reported as the base row (S1 has no far output).
* **At 30/50/80 m:** H1 − S1 and H2 − H1, position and heading (6 cells per seed), the cost rule. Text nav shares no source with RC-A, so these cells carry no echo.

### 6.3 Driving (24 cells per seed; H1 − S1 and H2 − H1)
The STRAT1 table (`RESULT_STRAT1.md` section 1): per band (operative knots 1-10, tactical 11-30): ADE, FDE; LONGITUDINAL speed, acceleration, along-track, headway, time-gap MAE; LATERAL heading, curvature, yaw-rate, cross-track MAE; TACTICAL TAC-LAT and TAC-LON accuracy. DK-VIOL, TTC, the tactical confusion and goal selection are STRAT1 work items (its section 4); they are reported here if the reader has them by then, else named absent with the reason and n.

### 6.4 Detected information (beside)
H − S1 accuracy per variable (country, weather, surface, illumination, period of day) with the cost rule: the far head and `e_nav_s` share the adapter with the ctx head.

### 6.5 STRATEGIC members not in this card (reported, not omitted)
Derived goals (later card, goal-input rule); the 120 m point as an input of T (later card); intent beyond 120 m of arc; visibility (not built, STRAT1 section 4.5).

## 7. Controls (each with the value it must read)

| id | control | must read |
|---|---|---|
| K1 | reader self-pair H1_s0 vs H1_s0 | 0.0, CI [0, 0] on every cell |
| K7 | S_CKPT120 on HOLD (ψ = 0) / announced side x 40 deg / mirrored | 0.0 / 1.0 / 0.0 |
| K8 | S_CKPT120 donor on a synthetic arm that copies its nav side | real 1.0; donor = the donors' side agreement computed from labels |
| K9b | T1 tick 0 at `--limit logged --batch 64` vs T0 (logged, batch 64), S's 30/50/80 and far checkpoints, full grid, every new arm | max \|Δ\| <= 1e-6 m (STRAT1's K9b read 0.0 m, M via I). The `--limit causal` form is reported beside with STRAT1's limit-mode explanation (`k9diag/`) |
| K-PERFECT | planted perfect follower (labels) | raw separated: **+0.232 [+0.090, +0.406]** (M, done) |
| K-NAVINF | planted follower of the tactical token (labels) | raw −0.126 and balanced +0.148, neither separated (M, done) |
| K-LABEL | recomputed RC-A 30/50/80 vs the train release of record | 0 validity mismatches, \|Δ\| <= 1e-5 (M on eval and the 2026-10-04 release; repeated at build on the release of record) |
| K-REACH-a | `--s-far off --nav-strat off` | state dict keys and every output bit-identical to the `a4357e79` S1 build |
| K-REACH-b | step-0 gradient on `s_far_out.*` with weight 0.3333; on `e_nav_s.*` with `--nav-strat on` | non-zero; and **exactly 0.0** on `s_far_out.*` with `--w-ckpt-far 0` (the probe's zero control) |
| K-REACH-c | mutations: delete the `w_ckpt_far * T["ckpt_far"]` term; join RC-A80 in place of RC-A120; delete the `nav_s` assignment | each named test RED (G-LIVE, label identity, G-LIVE) |
| K-REACH-d | argv | names every lever with its value; G-DVB refuses `--s-far on` with weight 0 |
| K-REACH-e | post-training zerograd gate (`zerograd_gate.py`, Adam-state method) | every `s_far_out.*` and `e_nav_s.*` param has optimizer state; else **HEAD-DEAD** and nothing is read |
| T-ISO-FAR | perturb `s_far_out` | T and O outputs bit-identical; T and O never read `nav_s` (key guard) |
| G-CLOCK-NAV-S | `nav_s` countdown vs v9 `nav_d` where both valid, as STRAT1's G-CLOCK-NAV | countdown error <= raw error; the `+ travelled` arm FAILS |
| R1 | label-permuted arm | NOT S120-FOLLOWS, and H1_s0 − R1_s0 on far position error separated negative |
| R2 | H2 trained with donor `nav_s` | real − donor at 120 m on the stratum where the tactical token is off NOT separated, and H2_s0 − R2_s0 on S_CKPT120 real there separated positive |

If R1 or R2 passes its must-fail condition, or K1/K7/K8/K9b fails, the read is broken and nothing from this card is quoted until it is fixed.

## 8. Outcomes, committed before any number (each names what runs next)

**STRATEGIC route following at 120 m** (judged for H1 and for H2 separately, both seeds; every effect also > F120)

| outcome | condition | triggers |
|---|---|---|
| **S120-FOLLOWS** | real − donor separated, balanced − 0.5 separated, and raw point > 0, at both seeds | the T-reads-far card (S's 120 m point as a fourth checkpoint token of T; PI go) and the derived-goals card. Labelled **S120-FOLLOWS-STRONG** when raw is also separated at both seeds |
| **S120-PARTIAL** | real − donor separated at both seeds, another condition fails at >= 1 seed | routed by stratum: shortfall where the side does not reach S → for H1, H2 (already in the chain); for H2, the far stratum's read decides the next lever (below). Shortfall where the side reaches S → the zero-training probe on the arm's frozen S hidden (fit on train once, scored on eval once; a negative probe is not a negative about learnability); positive → a 6,000-step arm; negative → S reader depth 2 |
| **S120-IGNORES** | real − donor not separated at >= 1 seed | zero-training check: far point under zero vs real nav on RC-A120 error (does the head read nav?); then the probe above |
| **HEAD-DEAD** | K-REACH-e fails | build defect: fix, rerun the arm; no read |

For H2 under S120-PARTIAL with the shortfall in the far stratum (d > 60 m), the next lever is the strategic token's distance resolution (ch3 clipped at 300 m today; a log-distance channel), a card amendment.

**Route setting at 120 m:** **FAR-SETS** = H − EXTRAP on position error separated negative at both seeds with |Δ| > F120 (the head adds route information beyond S's 80 m point); otherwise **FAR-NO-GAIN**, reported as written and read together with the route-following outcome. **ROUTE-COST** = a 30/50/80 cell costs at both seeds → H1s, the far point from its own trunk (`Linear(1024, 512) GELU Linear(512, 3)`, 526,339 params), so 30/50/80 keep their head layer to themselves.

**Driving (24 cells; H1 − S1 and H2 − H1 separately)**

| outcome | condition | triggers |
|---|---|---|
| **KEEPS** | no cost on any cell at either seed | the arm is the strategic-recipe candidate; adoption also needs STRAT1's SEEDS verdict on S1 − C0 |
| **ADDS** | KEEPS and a gain on the same cell at both seeds | as KEEPS; the gain is named |
| **COSTS-FAR** | a cell costs at both seeds | the same arm with `--s-adapter-detach on` (built, default off), same seeds and reads; no adoption |
| **SEEDS-DISAGREE** | a cost at exactly one seed | the arm at seeds 10 (init 0) and 11 (init 1), rep's form; a cell is a cost at >= 3 of 4 seeds with one per init |

**Detected info (beside):** a variable that DETECTS in S1 and costs at both seeds in H − S1 triggers the `--s-adapter-detach` arm, as COSTS-FAR.

**Written before data (H):** H1 reads S120-PARTIAL with the shortfall where the side does not reach S (K-NAVINF); H2 closes most of it. If H1 reads S120-FOLLOWS instead, S has learned a side prior from vision or distance beyond the token, and that is reported as found.

Bars, controls, arms and outcomes do not change after any number is seen. A bar that fails is reported as written, then the triggered arm runs (RULE ZERO).

## 9. Cost and the GPU queue

* **Training (LoRA pod, RTX 4000 Ada, one job at a time).** 0.19 s/step for a v8f-mini fine-tune (I: the brief, from the STRAT1 pilots; `pilot_summary.json` lives on the pod and was not read here). 3,000 steps = 9.5 min, about 15 min per arm with mint and gate (E). Arms 1-4: **about 1.0 h**; all 7: **about 1.75 h**; plus a 200-step pilot of H1 and H2 (about 5 min). Conditional arms (COSTS-FAR, SEEDS-DISAGREE, H1s): about 15 min each (E).
* **G-SUITE on the new commit:** pod CPU, duration U (STRAT1's ran in shards). Run it only with memory headroom: the pod's container cap is 50 GB and a gate beside trainers has been OOM-killed (memory note); read `memory.max` and anon first.
* **Stop rule:** a projected chain above 4 h stops and goes back to the MM.
* **Queue (PI, Q3):** the pod carries the data-scale blocks (about 15-26 GPU-h, one job at a time, I) and STRAT1's four SEEDS arms (in flight). Recommended slot: between two data-scale blocks, after the SEEDS arms, so CKPT120 reads within a day instead of after the whole data-scale run.
* **Labels (dev box CPU):** the RC-A120 table for train and eval-141 is the coverage pass with a writer: **64 s** for 4,572 train clips plus eval (M, this pass), plus moving the 254 MB release of record from box 2 or the pod.
* **Reads (dev box CPU):** T1 dumps about 10 min per arm (I, OBD README) = about 1.2 h for 7 arms; T0 STRATEGIC passes, 3 nav variants x 9 arms (7 + S1 s0/s1 re-scored for EXTRAP from the banked T0 arrays) about 2.5 h (E, scaled from STRAT1's 2 h for 21 passes); K9b about 10 min per arm; reads under 30 min (E).
* **Build:** about half a Sonnet agent-day (E): one layer, one token, 9 flags with G-DVB entries, the label writer, evaluator fields (far point, `nav_s` dead-reckoning and its donor/zero), reader cells (balanced, strata, EXTRAP), about 20 tests.
* **Disk:** at most 1.5 GB per banked arm (OBD bound, I) = at most 10.5 GB; label tables about 10 MB.

## 10. Known risks

* A zero-initialised layer inside a 3,000-step fine-tune may under-train (the lr x10 is the hedge; the probe branch of S120-PARTIAL reads it).
* The far layer shares `head_s`'s hidden layer with 30/50/80, so H1 can move them (ROUTE-COST → H1s).
* The DAgger banks were rolled by M2b under the v9 nav; base and arms share them, so they do not confound the contrasts, but they are not on-policy for a text-fed model (as STRAT1).
* Eval sides are 79 / 262 at 120 m while train is 33,962 / 35,274. The eval constant is strong, and the raw form is underpowered at 27 episodes (section 2.4). A larger strategic eval set would need held-out clips outside train; the PI's data-scale ruling keeps that pool to train, so it is not proposed here.
* `nav_s` with no announcement cut reads text rows up to p90 15.45 s old (I, BUILD_SPEC section 5); the countdown handles it and G-CLOCK-NAV-S gates it.
* 203 cache clips are absent from the 2026-10-04 release, so train coverage on them is U until the label build runs on the release of record.
* Pod preconditions (cache format, shipped inputs, memory cap) are U; the checker asserts each before arm 1, as in STRAT1.

## 11. Design artifacts (this folder; single copies until the MM lands them)

| file | what | sha256 |
|---|---|---|
| `measure_ckpt120_coverage.py` | coverage, controls R1-R3, K-PERFECT / K-NAVINF; pins the builder blob `edf5424d...` and ci.py blob `4f17fe11...` | `a1d052fe...` |
| `CKPT120_COVERAGE.json` | its output (inputs' sha256 match STRAT1's reader pins) | `759d8c21...` |
| `eval141_grid_truth.npz` | per-window ground truth (side, d_text, token on, RC valid at 30-150 m, v) | `da9529a1...` |
| `power_ckpt120.py` / `CKPT120_POWER.json` | power simulation (E) | `e2776e80...` / `689da936...` |
| `strat1_80m_stratum_split.py` / `STRAT1_80M_STRATUM_SPLIT.json` | the 80 m stratum re-read of the banked STRAT1 T0 passes | `b78bce99...` / `7777bb5f...` |
| `strat1_80m_stratum_ci.py` / `STRAT1_80M_STRATUM_CI.json` | its bootstrap rows | `3af15532...` / `2449f1cb...` |

Re-run: `python -I measure_ckpt120_coverage.py --builder <git show a4357e79:stack/scripts/build_v9_labels.py> --ci <git show a4357e79:taniteval/taniteval/ci.py> --out CKPT120_COVERAGE.json --allow-release-md5 f63ece410b725febb8a5242cf2b01d3c --dump-npz eval141_grid_truth.npz` (tanitad venv; about 1 min).

## Log
* 2026-11-16, Architecture agent: drafted for the MM. Not frozen, not built, not run.
* 2026-11-16 ~22:55Z, MM: FROZEN. Decisions 1-8 accepted as drafted. Added: arm priority H1_s0, H1_s1, H2_s0, H2_s1, H1_rep, R1_s0, R2_s0 (numbers PROVISIONAL until R1 and R2 read); H2 route following worded as following the supplied route text, the route-setting claim rests on FAR-SETS; the optimism wording; the queue slot after the PI's go. Text in `MM_RULINGS_CKPT120.md`.
