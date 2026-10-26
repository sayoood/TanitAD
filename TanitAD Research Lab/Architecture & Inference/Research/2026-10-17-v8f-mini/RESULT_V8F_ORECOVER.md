# PREREG_V8F_ORECOVER: result, both seeds (ORC vs ORC-0; trained on box 2, T1-kin read on the dev-box CPU; read 2026-10-26)

**Outcome, as committed: HARMS.** The pre-registered reader (`orc_read.py`, sha256 `91a9e97176968b5f…`) matched exactly one row of the outcome table, `HARMS`, on its condition *"R-ADE separated-worse at ≥ 1 seed, or R-T0 above +F/2"*. Two guards fired: R-ADE is separated-worse at seed 0 (+0.1100 m), and R-T0 is above +F/2 at both seeds (seed 0 +0.1103 m, seed 1 +0.0849 m, against 0.0604 m).

* **Committed next, verbatim from the PREREG's HARMS row:** *"ORC is not adopted. Reported with the rows; next is NO EFFECT's arm at p = 0.1"*.
* **Evidence class:** MEASURED (ours; artifact paths in `runs` and the manifest). Every number below is copied from `ORC_READ.json` (the reader; sha256 `a50dbba9…`), from `ORC_FAMILIES.json` / `PAIRFAM_T0_s0.json` / `PAIRFAM_T0_s1.json` (section 3; produced by this draft's driver and by the unchanged programme CLI), or from the run records they point at. The key path of every number is in `numbers_manifest.json`; `verify_numbers.py` re-reads all of them.
* **Tier, as the reader states it:** T1 (T1-kin: self-action OPEN loop, plain causal bucket, nav dead-reckoned; never 'closed loop') for R-TLAT / R-TLON / R-ADE; T0 (teacher-forced own-code, eval-141) for R-T0. The families in section 3 carry their own tier label per table. Nothing here is a driving claim, and "closed loop" does not occur in this document except in that negation.
* **Estimator of every interval:** taniteval.ci.paired_episode_cluster_bootstrap, B=2000, seed 0, over episodes; ORC - ORC-0 compares two NETWORKS per seed on the same windows (F1 rows: the T0 eval's F1 reducer on the matched cells; ADE rows: per-window mean). It answers 'would another draw of EPISODES say this?'; the training-seed question is answered only by the two seeds (H-ESTIM-SEED-1); a single ORC-0 per seed is the control for '3,000 more steps', not a replicate of the lever's noise.
* **Pre-registration:** Project Steering/PREREG_V8F_ORECOVER.md @28e28c88 + Amendment 1 (A1-2026-10-26) + Amendment 2 (A2-2026-10-26) + Amendment 3 (A3-2026-10-26) + Amendment 4 (A4-2026-10-26). Claim row `H-V8F-ORECOVER-1`. Code under test and gate commit: `853f2fc3330d29fbd73f44b4e758b31bd4ce8572`.
* **Status of this file:** a draft for the Master Mind. Nothing was staged into the repository, committed or pushed by it.

---

## 1. The outcome as committed

The PREREG's outcome table (section 4, with Amendment 2's wording of NO EFFECT), quoted:

| outcome | condition | committed next |
|---|---|---|
| ROBUST | R-TLAT and R-TLON separated-better and >= 2F, at both seeds; R-ADE not separated-worse; R-T0 within +F/2 | adopt ORC as the base. Re-read T1 dominance over P10 and the FLAT control on it |
| PARTIAL | at least one F1 row separated-better and >= 2F at both seeds, the other not, with the guards held | adopt for the head that moved; the next arm targets the other head's inputs |
| NO EFFECT | no F1 row separated-better and >= 2F at both seeds, guards held | on-policy states (DAgger form), under its own PREREG |
| **HARMS** | **R-ADE separated-worse at >= 1 seed, or R-T0 above +F/2** | **ORC is not adopted. Reported with the rows; next is NO EFFECT's arm at p = 0.1** |
| REFUSED | any control fails | fix and re-run |

The reader evaluates every row's predicate literally and lists every match in table order: `HARMS` only. The guards-held flag reads `no`. Because HARMS tests the guards, the F1 rows below cannot change the outcome in either direction.

### 1.1 The four rows, per seed (ORC - ORC-0)

**Estimator:** paired episode-cluster bootstrap, B = 2,000, 141 episodes, 2,399 windows. F1 rows: T1-kin, ticks >= 1, *matched* cells (cells scored in both runs; the reader's adopted reading, Amendment 2). ADE rows: per-window mean. Bars: 2F = 0.0248 (lat) / 0.0146 (lon), with F = 0.0124 / 0.0073 (the T0 fresh-age values the PREREG names); the R-ADE guard is *not separated-worse*; the R-T0 guard is point <= 0.0604 m (F/2, with F = 0.1207 m).

| row | seed | tier | ORC (level) | ORC-0 (level) | ORC - ORC-0 point [95 % interval] | separated | bar | bar met |
|---|---|---|---|---|---|---|---|---|
| R-TLAT: lat macro-F1, ticks >= 1 | s0 | T1-kin | 0.5360 | 0.5029 | **+0.0331 [-0.0079, +0.0730]** | no | separated-better and >= 2F (0.0248) | no |
| R-TLON: lon macro-F1, ticks >= 1 | s0 | T1-kin | 0.2747 | 0.2961 | **-0.0214 [-0.0497, +0.0131]** | no | separated-better and >= 2F (0.0146) | no |
| R-ADE: T1-kin plain ADE6, m | s0 | T1-kin | 2.9221 | 2.8121 | **+0.1100 [+0.0083, +0.2283]** | yes | guard: not separated-worse | separated-worse: **yes** |
| R-T0: T0 own-code ADE6, m | s0 | T0 | 2.0899 | 1.9796 | **+0.1103 [+0.0443, +0.1856]** (unrounded point +0.110267) | yes | guard: point <= 0.0604 | above the bar: **yes** |
| R-TLAT: lat macro-F1, ticks >= 1 | s1 | T1-kin | 0.5523 | 0.5172 | **+0.0351 [-0.0004, +0.0671]** | no | separated-better and >= 2F (0.0248) | no |
| R-TLON: lon macro-F1, ticks >= 1 | s1 | T1-kin | 0.2930 | 0.2618 | **+0.0312 [+0.0063, +0.0542]** | yes | separated-better and >= 2F (0.0146) | yes |
| R-ADE: T1-kin plain ADE6, m | s1 | T1-kin | 2.8828 | 2.8473 | **+0.0355 [-0.0363, +0.1047]** | no | guard: not separated-worse | separated-worse: **no** |
| R-T0: T0 own-code ADE6, m | s1 | T0 | 2.1365 | 2.0517 | **+0.0849 [+0.0378, +0.1324]** (unrounded point +0.084853) | yes | guard: point <= 0.0604 | above the bar: **yes** |

Cell counts behind the F1 rows (matched pairs): lat s0 3,831, s1 3,842; lon s0 3,726, s1 3,732. All four rows are over 141 episodes.

### 1.2 Which guard fired

* **R-ADE (T1-kin ADE6), separated-worse:** seed 0 **yes** (+0.1100 m, interval excludes 0), seed 1 **no** (+0.0355 [-0.0363, +0.1047] m, not separated). HARMS fires at any seed, so seed 0 alone fires it.
* **R-T0 (T0 ADE6) above +F/2:** seed 0 **yes** (+0.1103 m), seed 1 **yes** (+0.0849 m); the bar is 0.0604 m. Both intervals exclude 0 (s0 [+0.0443, +0.1856], s1 [+0.0378, +0.1324]). This second clause fires at both seeds independently of the first.
* **Size against F:** the T0 points are 0.91 F (s0) and 0.70 F (s1), with F = 0.1207 m the only run-to-run information on this rig (two M2b seeds). Two M2b training runs differ by F on this metric, so this rig alone cannot tell a difference of that size from training noise (H-ESTIM-SEED-1); see section 4.

### 1.3 The F1 rows, which the HARMS reading does not use (reported as the PREREG requires)

* **R-TLAT (lat macro-F1):** the points are +0.0331 (s0) and +0.0351 (s1), both above 2F = 0.0248; neither interval is separated (s0 lower bound -0.0079, s1 lower bound -0.0004). So the bar "separated-better and >= 2F" is met at neither seed (separated: s0 no, s1 no).
* **R-TLON (lon macro-F1):** s0 -0.0214 [-0.0497, +0.0131], not separated; s1 +0.0312 [+0.0063, +0.0542], separated-better and above 2F = 0.0146 (meets: yes). The reader's flag `separated_better_and_ge_2F_per_seed.lon` reads s0 no, s1 yes; for lat both read no. The two seeds disagree in sign on the lon row.
* **Own-cells version beside (cells scored in each run on its own, not the matched ones):** lat s0 +0.0360 [-0.0061, +0.0762] (separated: no), lat s1 +0.0350 [+0.0004, +0.0662] (separated: yes, meets 2F: yes); lon s0 -0.0182 [-0.0458, +0.0136], lon s1 +0.0289 [+0.0032, +0.0532]. Disclosure: on the own-cells definition the lat row at s1 has its interval just above 0. The decisive definition is the matched one (the PREREG adopts the reader's choice in Amendment 2), and at s0 the lat row is not separated under either definition, so no definition meets the lat bar at both seeds.
* **The reader's other flags:** `head_moved_at_both_seeds` lat no, lon no; `sub_bar_separated_better_at_both_seeds` lat no, lon no.

---

## 2. The controls: all PASS

The reader ran 104 controls and every one passed; a failed control would have produced REFUSED. The count is len(`controls_PASS`) and the all-true test is re-run in `verify_numbers.py`. By family (the second column counts how many of the controls are in the family; the third says what it checks; `s0`/`s1`/`ORC`/`ORC0` instances are counted separately):

| control family | instances | what it checks |
|---|---|---|
| `PREREG-TEXT` | 1 | every cell of the outcome table is verbatim in the PREREG file |
| `M2B-CKPT` | 2 | the M2b init checkpoint's sha256 equals the one banked in the PREREG (`601cb435…` / `6104fb15…`) |
| `ARGV` | 4 | each run's argv names the registered fine-tune |
| `SUMMARY` | 4 | the run's summary record (argv sha, 3,000 steps) |
| `INIT-CKPT` | 4 | the trainer's recorded `--init-ckpt` sha256 equals the M2b checkpoint's |
| `PERTURB-REACHED` | 4 | the perturbation reached training: ORC's perturbed share of eligible samples is 0.3 +/- 0.02; ORC-0's count is exactly 0 |
| `GATE-TOKEN` | 4 | a launch-gate PASS token binding commit, argv sha256 and data sha256 (the HMAC is not verified by the reader: key on the launch host; the tokens record `hmac: absent`) |
| `T0-CKPT` | 4 | box 2's standard T0 eval names this run's checkpoint |
| `T0F-CKPT` | 4 | the fresh-age T0 eval names this run's checkpoint (Amendment 4) |
| `T0-AGES` | 4 | the standard eval says ages `cycle`, the fresh eval ages `fresh`, so neither stands in for the other |
| `PAIR` | 2 | ORC and ORC-0 share init, steps, data and seed; argv differs only in the perturbation flags and `--out` |
| `K1` | 4 | T1-HEADS K1 (instrument identity on the run) |
| `K2` | 4 | T1-HEADS K2, causal run |
| `K3` | 4 | T1-HEADS K3, causal run |
| `K5` | 4 | T1-HEADS K5 (mirror control) |
| `K6` | 4 | T1-HEADS K6 |
| `K7` | 4 | the executed track rebuilt from the dumped plans equals the dumped track (Amendment 2's new K7) |
| `GRID` | 4 | the causal run is on the 2,399-window grid |
| `T1-CONFIG` | 4 | the causal run's configuration is the registered one (causal bucket, m = 10, periodic, logged scene, dead-reckoned nav, heads dumped) |
| `K2-logged` | 4 | K2 on the `--limit logged` run |
| `K3-logged` | 4 | K3 on the `--limit logged` run |
| `GRID-logged` | 4 | the logged run is on the grid |
| `T1-CONFIG-logged` | 4 | the logged run's configuration |
| `K4` | 4 | tick 0 of the logged run equals the arm's own fresh-age T0 eval to 1e-3 (Amendment 4) |
| `DUMP-WINDOWS` | 4 | dump windows equal tick windows, window for window |
| `GRID-COUNTS` | 4 | 2,399 windows / 141 episodes |
| `R2-ADE6` | 4 | the reader's ADE6 delta equals the run's own analysis delta |
| `SAME-WINDOWS` | 2 | ORC and ORC-0 are on identical windows (ticks, dumps, T0 eid) |
| `K1-ORC-ROWS` | 1 | the reader's own K1 on its output rows |

Control values worth quoting (all from `controls.*.detail`):

| run | perturbed / eligible samples | share | steps | K4: max abs position diff vs own T0 (3 horizons), m | K4: lat / lon F1 diff | K7: max track diff, m |
|---|---|---|---|---|---|---|
| ORC_s0 | 24,720 / 82,457 | 0.2998 | 3,000 | 4.8e-05, 3.9e-05, 2.0e-05 | 0.0000 / 0.0000 | 0.0000 |
| ORC_s1 | 24,830 / 82,166 | 0.3022 | 3,000 | 2.7e-05, 5.0e-05, 4.4e-05 | 0.0000 / 0.0000 | 0.0000 |
| ORC0_s0 | 0 (exactly 0) | n/a | 3,000 | 4.4e-05, 9.2e-06, 2.0e-05 | 0.0000 / 0.0000 | 0.0000 |
| ORC0_s1 | 0 (exactly 0) | n/a | 3,000 | 4.4e-05, 3.1e-06, 2.1e-05 | 0.0000 / 0.0000 | 0.0000 |

* **Deliberate regression:** not spent as a GPU arm, as committed: the PREREG records it as proven at build time (deleting the perturbation line turns the lever's tests RED). Not re-verified here (INHERITED).
* **Gate token HMAC:** the four tokens record `hmac: absent`; the reader checks the token's binding to argv, commit and data, and does not verify an HMAC. So "launch-gate PASS" here means the binding was checked, not that the signature was.
* **Provenance pins:** M2b init checkpoints s0 `601cb43526731786…`, s1 `6104fb15dc21d524…`; manifest sha256 `27917c78538ac78e…`; the reader read 197.6 s.

---

## 3. The four metric families (binding), per family, never pooled

The reader's rows are two F1 rows and two ADE6 rows, and HARMS is decided by the two ADE6 rows alone. The binding rule (CLAUDE.md, "every eval reports four metric families") asks for LONGITUDINAL, LATERAL, TACTICAL and STRATEGIC beside the ADE. The reader computes only part of that (the lat / lon head F1 at T1-kin, per-run S-POS against CTRV, the tick-0 and goal F1s). The rest was computed for this draft, ORC - ORC-0 per seed on the same windows, with the programme's own instruments:

* `stack/scripts/v8f_paired_families.py` (unchanged CLI, commit `07dcb513`): TACTICAL (plan-derived decisions, goal point, lat / lon head F1, goal head F1) and STRATEGIC (S's checkpoint position and heading error at 30 / 50 / 80 m), on the **T0 eval npz** of each arm. It reads only an eval `run` dump, so it cannot read the T1 tick dumps (reason stated per row).
* `taniteval.v8f_beta_eval.family_windows` + `paired_row` (the paired per-window family instrument that `v8f_kscore.py` and the OACC reader use) for LONGITUDINAL speed / accel / along-track, LATERAL heading / curvature / yaw-rate / cross-track and the plan-derived decision agreement; and `v8f_dk_read.read_dk` (the distance-keeping reader, with its own identity control) for headway / time gap / TTC / LEAD-VIOL. These run on both tiers: T0 on box 2's standard eval `pred`, T1-kin on the `cl` (executed) track of the causal dump. Driver: `orc_families.py`.
* **Same estimator everywhere:** paired episode-cluster bootstrap, B = 2,000, seed 0, over 141 episodes (2,399 windows; rows restricted to windows with a valid value say their n).

**Controls on this section (all must hold):**

* **Reproduction of the reader.** The driver's ADE6 ORC - ORC-0 equals the reader's, to the estimator's own four decimals, interval and separation included: T0 s0 +0.1103 vs +0.1103, T0 s1 +0.0849 vs +0.0849, T1-kin s0 +0.1100 vs +0.1100, T1-kin s1 +0.0355 vs +0.0355. PASS flags: s0 yes, s1 yes. So the families below are on the reader's windows and the reader's tracks.
* **Identity.** Every paired family row run on ORC against ORC reads exactly +0 [0, 0], and `read_dk`'s own identity control (headway, time gap, TTC, LEAD-VIOL) reads +0 [0, 0] (flags in `ORC_FAMILIES.json`, `identity_all_exactly_zero` and `distance_keeping.identity_control`; asserted again in `verify_numbers.py`).
* **Grid.** At T1-kin the dump's windows, episode ids, `2 * ws` frames and `v0` equal the T0 grid's for both arms and seeds; the dump GT differs from the grid GT by at most 7.6e-06 m (float32 round trip). The lead block's sha256 is `7a879baa2b6f4c14…`, the registered one.
* **Cross-instrument.** At T0 the plan-derived lateral / longitudinal agreement from `v8f_paired_families.py` equals the `family_windows` rows below (asserted in `verify_numbers.py`).

**Tier labels used in the tables.** `T0` = the PREREG's name for box 2's standard eval (R-T0's source). The eval JSON stamps itself `raw.tier = "T0-open-loop-own-held-outputs (trunk-probe grid)"` and the paired-families tool stamps its output "T1-class open loop (own held outputs; no recorded future action)"; these are three labels for one open-loop scoring of the logged future, and I have not reconciled them (the same open item as in `RESULT_V8F_OACC.md`). I use the PREREG's names. `T1-kin` = the causal-bucket dev-box rollout whose executed track `cl` the reader scores; the lead follows its LOGGED future and does not react to the ego.

Column key: each cell is `ORC - ORC-0` point [95 % interval], then whether the interval excludes 0 and which way is better. Directions are labelled only where the metric has one (error: lower is better; agreement, F1: higher; LEAD-VIOL share: lower). Levels (each arm's mean on the paired windows) are in `ORC_FAMILIES.json` (`mean_a_on_pairs`, `mean_b_on_pairs`).

### 3.1 LONGITUDINAL

**Anti-echo caveat, per tier.** At T0 the eval JSONs of all four arms carry `anti_echo.status`: ORC_s0 UNAVAILABLE, ORC_s1 UNAVAILABLE, ORC0_s0 UNAVAILABLE, ORC0_s1 UNAVAILABLE (no ego speed at t0 in the window dict), and `distance_keeping.status` is UNAVAILABLE in the eval JSON itself; the distance-keeping rows below come from the lead-block join (`read_dk`), not from the eval JSON. At T1-kin the anti-echo block is computed and reads, for each of the four runs: 

* `ORC_s0`: status OK, `longitudinal_claim_admissible` **no**, flagged yes; holdv0=LOSES_TO_HOLDV0 · copy_detector=CLEAN (echo_index 0.0000 vs GT 0.0992) · shuffle=UNAVAILABLE.
* `ORC_s1`: status OK, `longitudinal_claim_admissible` **no**, flagged yes; holdv0=LOSES_TO_HOLDV0 · copy_detector=CLEAN (echo_index 0.0000 vs GT 0.0992) · shuffle=UNAVAILABLE.
* `ORC0_s0`: status OK, `longitudinal_claim_admissible` **no**, flagged yes; holdv0=LOSES_TO_HOLDV0 · copy_detector=CLEAN (echo_index 0.0000 vs GT 0.0992) · shuffle=UNAVAILABLE.
* `ORC0_s1`: status OK, `longitudinal_claim_admissible` **no**, flagged yes; holdv0=LOSES_TO_HOLDV0 · copy_detector=CLEAN (echo_index 0.0000 vs GT 0.0992) · shuffle=UNAVAILABLE.

⇒ **Every LONGITUDINAL number in this document, for ORC and for ORC-0 alike, is a paired contrast of two arms that the programme's own block does not admit as a longitudinal capability claim.** The contrast ORC - ORC-0 is still a valid description of what the perturbation changed on these windows; it is not a statement that either arm sets the right speed.

| cell (ORC - ORC-0) | T0 s0 | T0 s1 | T1-kin s0 | T1-kin s1 |
|---|---|---|---|---|
| **speed MAE, m/s** | +0.0590 [+0.0375, +0.0811], separated, ORC WORSE (n 2,399 / 141) | +0.0212 [+0.0036, +0.0379], separated, ORC WORSE (n 2,399 / 141) | +0.0074 [-0.0606, +0.0693], not separated (n 2,399 / 141) | -0.0386 [-0.1048, +0.0241], not separated (n 2,399 / 141) |
| **accel MAE, m/s^2** | +0.1312 [+0.1075, +0.1568], separated, ORC WORSE (n 2,399 / 141) | +0.0547 [+0.0297, +0.0793], separated, ORC WORSE (n 2,399 / 141) | +0.2552 [+0.1080, +0.3945], separated, ORC WORSE (n 2,399 / 141) | +0.1162 [-0.0345, +0.2651], not separated (n 2,399 / 141) |
| along-track MAE, m | +0.1126 [+0.0483, +0.1879], separated, ORC WORSE (n 2,399 / 141) | +0.0751 [+0.0296, +0.1209], separated, ORC WORSE (n 2,399 / 141) | +0.0937 [-0.0190, +0.2287], not separated (n 2,399 / 141) | +0.0309 [-0.0508, +0.1127], not separated (n 2,399 / 141) |

**Distance keeping, K = 30 (6 s), where a lead exists** (`read_dk`; the lead block's windows: LEAD 743, NO_LEAD 552, NOT_STRAIGHT 931, NO_LABEL 173; only LEAD windows are scored, the others are counted and never scored as free flow; the lead follows its logged future). Differences are ORC - ORC-0; n shown as windows used / episodes.

| cell (ORC - ORC-0) | T0 s0 | T0 s1 | T1-kin s0 | T1-kin s1 |
|---|---|---|---|---|
| min headway, m | -0.1841 [-0.4446, +0.0492], not separated (n 724 / 73) | -0.0965 [-0.3122, +0.1312], not separated (n 722 / 74) | -0.2871 [-0.5297, -0.0842], separated (no direction label) (n 714 / 73) | -0.3141 [-0.5636, -0.0763], separated (no direction label) (n 721 / 74) |
| min time gap, s | -0.0976 [-0.2038, -0.0199], separated (no direction label) (n 663 / 73) | -0.0427 [-0.0814, -0.0109], separated (no direction label) (n 661 / 73) | -0.1113 [-0.2174, -0.0330], separated (no direction label) (n 660 / 73) | -0.0341 [-0.0809, +0.0171], not separated (n 665 / 74) |
| min TTC, s (censored at a 30 s cap) | -0.7703 [-1.2558, -0.3826], separated (no direction label) (n 724 / 73) | +0.1002 [-0.2151, +0.4301], not separated (n 722 / 74) | -1.0032 [-1.4538, -0.6020], separated (no direction label) (n 714 / 73) | -0.1953 [-0.6651, +0.2620], not separated (n 721 / 74) |
| \|headway error\| vs the human, m | +0.1108 [-0.0640, +0.3148], not separated (n 717 / 72) | +0.0088 [-0.1422, +0.1995], not separated (n 716 / 73) | +0.0367 [-0.1232, +0.1906], not separated (n 707 / 72) | -0.0771 [-0.2530, +0.0747], not separated (n 712 / 72) |
| \|time-gap error\| vs the human, s | +0.0325 [-0.0419, +0.1055], not separated (n 656 / 72) | -0.0084 [-0.0337, +0.0142], not separated (n 655 / 72) | +0.0091 [-0.0487, +0.0732], not separated (n 653 / 72) | +0.0190 [-0.0118, +0.0576], not separated (n 656 / 72) |
| \|min-TTC error\| vs the human, s | +0.2990 [+0.0564, +0.5580], separated, ORC WORSE (n 717 / 72) | -0.1696 [-0.4966, +0.1163], not separated (n 716 / 73) | +0.1875 [-0.1110, +0.5104], not separated (n 707 / 72) | +0.1307 [-0.2321, +0.5090], not separated (n 712 / 72) |
| LEAD-VIOL, at risk (violating share) | +0.0067 [-0.0018, +0.0162], not separated (n 595 / 70) | +0.0017 [-0.0036, +0.0078], not separated (n 595 / 70) | +0.0151 [-0.0017, +0.0348], not separated (n 595 / 70) | +0.0000 [-0.0106, +0.0104], not separated (n 595 / 70) |
| LEAD-VIOL, all lead windows | +0.0054 [-0.0015, +0.0130], not separated (n 743 / 77) | +0.0013 [-0.0029, +0.0064], not separated (n 743 / 77) | +0.0121 [-0.0014, +0.0276], not separated (n 743 / 77) | +0.0000 [-0.0081, +0.0092], not separated (n 743 / 77) |
| lead-progress ratio | +0.0461 [+0.0240, +0.0726], separated (no direction label) (n 595 / 70) | +0.0221 [+0.0104, +0.0365], separated (no direction label) (n 595 / 70) | +0.0518 [-0.0939, +0.2121], not separated (n 595 / 70) | +0.1140 [+0.0228, +0.2364], separated (no direction label) (n 595 / 70) |

LEAD-VIOL counts (violations ORC / ORC-0 among the at-risk windows):

| tier, seed | at risk | violations ORC | violations ORC-0 |
|---|---|---|---|
| T0 s0 | 595 | 37 | 33 |
| T0 s1 | 595 | 32 | 31 |
| T1-kin s0 | 595 | 34 | 25 |
| T1-kin s1 | 595 | 16 | 16 |

Min-TTC means are over censored data: the closing-window counts per arm are in `ORC_FAMILIES.json` (`distance_keeping.arms.*.metrics.min_ttc_s.n_closing`). `headway`, `time gap` and `TTC` carry no direction label from the instrument (more distance kept is not automatically better), so they are tagged "no direction label" when separated.

### 3.2 LATERAL

| cell (ORC - ORC-0) | T0 s0 | T0 s1 | T1-kin s0 | T1-kin s1 |
|---|---|---|---|---|
| **heading MAE, deg** | -0.3253 [-0.8306, +0.0894], not separated (n 2,344 / 141) | +0.1115 [-0.0018, +0.2328], not separated (n 2,347 / 141) | -1.3094 [-1.8920, -0.7948], separated, ORC better (n 2,373 / 141) | -1.0031 [-1.4493, -0.6149], separated, ORC better (n 2,373 / 141) |
| **curvature MAE, 1/m** | -0.0161 [-0.0291, -0.0059], separated, ORC better (n 2,329 / 141) | -0.0009 [-0.0087, +0.0082], not separated (n 2,329 / 141) | -0.0537 [-0.0768, -0.0330], separated, ORC better (n 2,366 / 141) | -0.0220 [-0.0348, -0.0097], separated, ORC better (n 2,365 / 141) |
| **yaw-rate MAE, deg/s** | -0.6539 [-1.0441, -0.3325], separated, ORC better (n 2,329 / 141) | +0.0179 [-0.3278, +0.3557], not separated (n 2,329 / 141) | -6.5892 [-9.7183, -3.9884], separated, ORC better (n 2,366 / 141) | -3.1579 [-4.7949, -1.6844], separated, ORC better (n 2,365 / 141) |
| cross-track MAE, m | +0.0109 [-0.0091, +0.0299], not separated (n 2,399 / 141) | +0.0215 [+0.0013, +0.0435], separated, ORC WORSE (n 2,399 / 141) | +0.0208 [-0.0166, +0.0572], not separated (n 2,399 / 141) | +0.0150 [-0.0226, +0.0542], not separated (n 2,399 / 141) |

Heading, curvature and yaw-rate are means of per-window components over the windows where both paths move (so their n is below the full grid); the n of every cell is printed. Cross-track is an offset at a matched time index, not a distance to the GT path, and is uninterpretable alone when the along-track error is large (the instrument's own caveat).

### 3.3 TACTICAL

**Plan-derived decisions** (the manoeuvre read off each arm's executed plan against the one read off the human path; both tiers; `family_windows`):

| cell (ORC - ORC-0) | T0 s0 | T0 s1 | T1-kin s0 | T1-kin s1 |
|---|---|---|---|---|
| lateral decision agreement | +0.0188 [+0.0017, +0.0358], separated, ORC better (n 2,399 / 141) | -0.0108 [-0.0196, -0.0029], separated, ORC WORSE (n 2,399 / 141) | +0.0221 [+0.0029, +0.0430], separated, ORC better (n 2,399 / 141) | +0.0379 [+0.0171, +0.0587], separated, ORC better (n 2,399 / 141) |
| longitudinal decision agreement | -0.0079 [-0.0225, +0.0071], not separated (n 2,399 / 141) | +0.0038 [-0.0088, +0.0183], not separated (n 2,399 / 141) | +0.0017 [-0.0183, +0.0225], not separated (n 2,399 / 141) | +0.0346 [+0.0083, +0.0626], separated, ORC better (n 2,399 / 141) |

**Head-based decisions and goal setting.** The decisive R-TLAT / R-TLON are in section 1.1 (T1-kin, ticks >= 1). The T0 versions below are the head argmax at the window, against the v9 labels, from `v8f_paired_families.py` (the head F1 is re-computed on every bootstrap resample). The paired-families tool cannot read the T1 tick dumps (it requires an eval `run` dump with `pred`, `lat_logits`, `lon_logits`, `goal_prob`, `ckpt`, `ck_pos_err`, `ade6` per window), so the T1-kin head rows are the reader's R-TLAT / R-TLON and the per-run values beside.

| cell (ORC - ORC-0) | T0 s0 | T0 s1 | T1-kin s0 | T1-kin s1 |
|---|---|---|---|---|
| lat head macro-F1 (T0: window argmax; T1-kin: ticks >= 1 matched cells = R-TLAT) | +0.0005 [-0.0184, +0.0235], not separated (n 2,107 / 139) | +0.0120 [-0.0091, +0.0306], not separated (n 2,107 / 139) | +0.0331 [-0.0079, +0.0730], not separated (matched cells 3,831) | +0.0351 [-0.0004, +0.0671], not separated (matched cells 3,842) |
| lon head macro-F1 (T0: window argmax; T1-kin: ticks >= 1 matched cells = R-TLON) | -0.0403 [-0.0912, +0.0293], not separated (n 2,240 / 136) | +0.0046 [-0.0582, +0.0589], not separated (n 2,240 / 136) | -0.0214 [-0.0497, +0.0131], not separated (matched cells 3,726) | +0.0312 [+0.0063, +0.0542], separated, ORC better (matched cells 3,732) |
| lat head accuracy | -0.0009 [-0.0076, +0.0048], not separated (n 2,107 / 139) | +0.0024 [-0.0015, +0.0066], not separated (n 2,107 / 139) | n/c (see text) | n/c (see text) |
| lon head accuracy | -0.0138 [-0.0303, +0.0027], not separated (n 2,240 / 136) | +0.0089 [-0.0097, +0.0284], not separated (n 2,240 / 136) | n/c (see text) | n/c (see text) |
| goal head macro-F1 (22 tokens) | -0.0296 [-0.0571, -0.0049], separated, ORC WORSE (n 2,365 / 139) | -0.0215 [-0.0439, +0.0061], not separated (n 2,365 / 139) | n/c (see text) | n/c (see text) |
| 5-way collapsed manoeuvre accuracy (plan-derived) | +0.0021 [-0.0159, +0.0192], not separated (n 2,399 / 141) | -0.0008 [-0.0146, +0.0158], not separated (n 2,399 / 141) | n/c (see text) | n/c (see text) |
| lateral decision accuracy (plan-derived, tool's own) | +0.0188 [+0.0017, +0.0358], separated, ORC better (n 2,399 / 141) | -0.0108 [-0.0196, -0.0029], separated, ORC WORSE (n 2,399 / 141) | n/c (see text) | n/c (see text) |
| longitudinal decision accuracy (plan-derived, tool's own) | -0.0079 [-0.0225, +0.0071], not separated (n 2,399 / 141) | +0.0038 [-0.0088, +0.0183], not separated (n 2,399 / 141) | n/c (see text) | n/c (see text) |
| goal-point error (FDE6), m | +0.0947 [-0.0311, +0.2287], not separated (n 2,399 / 141) | +0.0785 [-0.0265, +0.1814], not separated (n 2,399 / 141) | n/c (see text) | n/c (see text) |
| goal bearing MAE, deg | -0.0139 [-0.5931, +0.4731], not separated (n 2,367 / 141) | +0.3021 [+0.0258, +0.6524], separated, ORC WORSE (n 2,367 / 141) | n/c (see text) | n/c (see text) |

**Per run, beside, not paired** (the reader's `beside.per_seed`; point values with no interval, so a difference between two of them is a description and not a finding). Tick 0 is the logged-state rollout step; ticks >= 1 are the model's own executed states. The PREREG's reason for tick 0: it shows whether ORC lost logged-state skill.

| seed | head | tick-0 F1 ORC | tick-0 F1 ORC-0 | ORC - ORC-0 (tick 0) | goal macro-F1, ticks >= 1: ORC | ORC-0 | goal macro-F1, tick 0: ORC | ORC-0 |
|---|---|---|---|---|---|---|---|---|
| s0 | lat | 0.8432 | 0.8556 | -0.0124 | 0.1688 | 0.1756 | 0.2490 | 0.2750 |
| s0 | lon | 0.4258 | 0.4784 | -0.0525 | (same head-independent value as the lat row) | | |  |
| s1 | lat | 0.8445 | 0.8585 | -0.0140 | 0.1782 | 0.1732 | 0.2631 | 0.2938 |
| s1 | lon | 0.4568 | 0.4847 | -0.0279 | (same head-independent value as the lat row) | | |  |

(Tick-0 F1 is over 2,107 lat cells and 2,240 lon cells, the same for every run. The goal macro-F1 is one value per run, so it is printed on the lat row only.)

### 3.4 STRATEGIC

**Status: NOT PAUSED for v8F (PI ruling 2026-10-17, R73): reported BESIDE as S-POS@30/50/80 vs CTRV per run (unpaired); decides nothing.** It is reported, not paused, and it decides nothing.

**T0, paired** (S's checkpoint error against the v9 route checkpoints at 30 / 50 / 80 m of arc; `v8f_paired_families.py`, ORC - ORC-0; lower is better; windows with a valid checkpoint label):

| cell (ORC - ORC-0) | T0 s0 | T0 s1 | T1-kin s0 | T1-kin s1 |
|---|---|---|---|---|
| checkpoint position error @30 m, m | +0.0225 [-0.0087, +0.0637], not separated (n 2,322 / 137) | +0.0102 [-0.0264, +0.0488], not separated (n 2,322 / 137) | n/c (paired instrument reads eval dumps only) | n/c (same reason) |
| checkpoint position error @50 m, m | +0.0077 [-0.0545, +0.0757], not separated (n 2,319 / 137) | -0.0022 [-0.0813, +0.0781], not separated (n 2,319 / 137) | n/c (paired instrument reads eval dumps only) | n/c (same reason) |
| checkpoint position error @80 m, m | -0.0363 [-0.1967, +0.1121], not separated (n 2,294 / 136) | -0.0788 [-0.3270, +0.1337], not separated (n 2,294 / 136) | n/c (paired instrument reads eval dumps only) | n/c (same reason) |
| checkpoint heading error @30 m, deg | -0.0101 [-0.0989, +0.0756], not separated (n 2,322 / 137) | -0.0017 [-0.2482, +0.1878], not separated (n 2,322 / 137) | n/c (same reason) | n/c (same reason) |
| checkpoint heading error @50 m, deg | -0.0546 [-0.2258, +0.1034], not separated (n 2,319 / 137) | -0.2312 [-0.7013, +0.1262], not separated (n 2,319 / 137) | n/c (same reason) | n/c (same reason) |
| checkpoint heading error @80 m, deg | -0.1435 [-0.6418, +0.2594], not separated (n 2,294 / 136) | -0.1192 [-0.6331, +0.3461], not separated (n 2,294 / 136) | n/c (same reason) | n/c (same reason) |

**T1-kin, per run, unpaired** (the reader's S-POS against a constant-turn-rate-and-velocity (CTRV) rollout from the same pose, ticks >= 1, within each run; delta = S - CTRV position error, so negative means S is nearer the route checkpoint; each is a within-run contrast with its own interval, not an ORC vs ORC-0 contrast):

| seed | arc | ORC: S - CTRV | ORC-0: S - CTRV |
|---|---|---|---|
| s0 | 30 m | -0.2174 [-0.6438, +0.2074], not separated | -0.0699 [-0.4545, +0.2895], not separated |
| s0 | 50 m | -1.6218 [-2.7553, -0.5431], separated, S nearer than CTRV | -1.3533 [-2.3551, -0.4411], separated, S nearer than CTRV |
| s0 | 80 m | -4.8894 [-7.2380, -2.5920], separated, S nearer than CTRV | -4.9005 [-7.0365, -2.8924], separated, S nearer than CTRV |
| s1 | 30 m | +0.5466 [+0.1125, +1.0024], separated, S farther than CTRV | +0.8183 [+0.3656, +1.2547], separated, S farther than CTRV |
| s1 | 50 m | -0.2676 [-1.3028, +0.7607], not separated | +0.2848 [-0.7789, +1.2839], not separated |
| s1 | 80 m | -3.1151 [-5.3261, -0.8864], separated, S nearer than CTRV | -2.2420 [-4.5161, -0.0389], separated, S nearer than CTRV |

**Not computed:** a PAIRED ORC - ORC-0 contrast on STRATEGIC at T1-kin. Reason: the only paired STRATEGIC instrument used here (`v8f_paired_families.py`) needs an eval `run` dump, and the T1 tick dump is a different object; building the paired S-POS from the ticks is a work item (the per-run unpaired values above are what exists). n for that cell: 0 (not attempted), UNVERIFIED either way.

Crawl fraction (ticks >= 1, beside): ORC s0 0.1263, ORC-0 s0 0.1332, ORC s1 0.1355, ORC-0 s1 0.1507 (the reader stores one value for both heads; unpaired points, no interval).

### 3.5 What the four families say, descriptively (no interpretation yet)

* **LONGITUDINAL.** At T0 the three kinematic rows (speed, accel, along-track MAE) are separated-worse for ORC at both seeds. At T1-kin accel MAE is separated-worse at seed 0 only, and speed and along-track are not separated at either seed.
* **LATERAL.** At T1-kin heading, curvature and yaw-rate MAE are separated-better for ORC at both seeds. At T0 curvature and yaw-rate are separated-better at seed 0 only; cross-track is not better at either tier.
* **TACTICAL.** Plan-derived lateral agreement is separated-better at T1-kin at both seeds; at T0 it is separated-better at s0 and separated-worse at s1. The T1-kin head F1 rows are those of section 1.1 (lat n.s. at both seeds, lon n.s. at s0 and separated-better at s1). At T0 the lat and lon head macro-F1 rows are not separated at either seed, and the goal head macro-F1 is separated-worse at s0 and not separated at s1.
* **DISTANCE KEEPING (a LONGITUDINAL cell group).** The LEAD-VIOL rate (at risk and over all lead windows) is not separated at either tier or seed. At T1-kin ORC's minimum headway is separated-lower than ORC-0's at both seeds and its minimum time gap at seed 0; at T0 the minimum time gap is separated-lower at both seeds and the minimum headway is not separated. The instrument gives these cells no direction label.
* **STRATEGIC.** At T0 no checkpoint position or heading cell (30 / 50 / 80 m) is separated at either seed. No paired T1-kin contrast was computed (stated above); the per-run unpaired S-POS rows are what exists at that tier.

### 3.6 Per-family gaps: what was not computed, and why

* **LONGITUDINAL:** the target-speed accuracy bands and the ego-progress ratio are in each arm's eval JSON (`raw.families.longitudinal`) as unpaired levels and were not paired here. The anti-echo condition is undischarged at T0 and flagged at T1-kin (section 3.1). Nothing else missing: speed, accel, along-track and distance keeping are paired at both tiers.
* **LATERAL:** none missing at either tier.
* **TACTICAL:** at T1-kin the paired head accuracy, the goal-head F1 and the 5-way collapsed accuracy were not computed, because `v8f_paired_families.py` reads eval `run` dumps and not the T1 tick dumps; the reader's R-TLAT / R-TLON and the unpaired per-run goal F1 stand in. Anchor selection and selected-vs-executed decisions are UNAVAILABLE by design (the tool's own `unavailable_by_design` block: a trajectory dump carries one path per window and no tactical selection).
* **STRATEGIC:** the paired T1-kin contrast was not computed (section 3.4). The T0 paired cells exist. The eval JSON's own strategic block reads `status: see H-MINI-STRAT-1`, `paused: false`, i.e. it points elsewhere and carries no number.

These bullets restate the table cells in words. `verify_numbers.py` re-derives each directional sentence from the JSONs (the tags in the table are computed, not typed).

---

## 4. Tier, estimator, and the three variance questions

| item | value |
|---|---|
| tier of R-TLAT / R-TLON / R-ADE | T1-kin: self-action OPEN loop, plain causal bucket, nav dead-reckoned, scene logged, replan every 10 steps; the model conditions on its own executed states and the lead follows its logged future. Not a driving claim |
| tier of R-T0 | T0: teacher-forced own-code eval-141 (box 2's standard eval, ages cycle) |
| estimator | taniteval.ci.paired_episode_cluster_bootstrap, B=2000, seed 0, over episodes; ORC - ORC-0 compares two NETWORKS per seed on the same windows (F1 rows: the T0 eval's F1 reducer on the matched cells; ADE rows: per-window mean). It answers 'would another draw of EPISODES say this?'; the training-seed question is answered only by the two seeds (H-ESTIM-SEED-1); a single ORC-0 per seed is the control for '3,000 more steps', not a replicate of the lever's noise. |
| windows / episodes / B | 2,399 / 141 / 2,000 |

**Which of the three variances each interval answers.**

1. **Another draw of EPISODES** (the models held fixed): answered by every interval here. This is all the bootstrap answers.
2. **Another TRAINING run:** answered only by the two seeds (the reader's own wording: "the training-seed question is answered only by the two seeds (H-ESTIM-SEED-1); a single ORC-0 per seed is the control for '3,000 more steps', not a replicate of the lever's noise"). There is no replicate of ORC at a given seed. Consequences: (a) the R-T0 guard is broken at both seeds with separated intervals, so the sign agrees across the two training runs, but two runs are two runs; (b) the R-ADE guard is broken at s0 only, and its s0 size (+0.1100 m) is below F = 0.1207 m; (c) the lon F1 row changes sign across seeds (-0.0214, +0.0312). A separated interval from one training seed is necessary, not sufficient, for a lever effect.
3. **Another INFERENCE run:** the reader's own text is SILENT on whether the T1-kin planner is deterministic (`ORC_READ.json` and `NOTES.md` contain no such statement; UNVERIFIED as a stated property). What the reader's controls show instead: K4 reproduces each arm's own T0 eval from an independent T1 rollout's tick 0 to at most 5.0e-05 m with 2,399 of 2,399 identical argmax decisions, and K7 rebuilds the executed track from the dumped plans to 0.0000 m. Both are consistent with a deterministic planner (there is no sampling step in this rollout's configuration as the reader records it) but they do not test a second inference run of the same checkpoint. I therefore do not claim the inference question is closed; it is probably not binding here, and it is not measured.

---

## 5. Interpretation (POST-DATA; separate from the outcome; HYPOTHESIS class unless a cell is cited)

The outcome above was fixed by the committed table before and independent of anything in this section. What follows was written after all the numbers were in, so it can only generate hypotheses.

1. **Lat F1 rose by about +0.034 at both seeds, but not separated.** The matched-cells points are +0.0331 (s0) and +0.0351 (s1): same sign, same size, both above 2F = 0.0248, and neither interval excludes 0 (s1's lower bound is -0.0004). Two same-sign, same-size points are what a real small effect would leave, and noise of this size can also produce them; the data do not say which. Per the committed outcome table it is not an effect, and it does not rescue ORC from HARMS.
2. **The T0 head rows do not move where the T1-kin ones do.** At T0 (logged states) the lat head macro-F1 difference is +0.0005 (s0) and +0.0120 (s1), neither separated, against +0.0331 / +0.0351 at T1-kin ticks >= 1 (the model's own executed states). HYPOTHESIS: the lever, which perturbs the ego inputs, acts where the model reads its own motion and not on logged states. This is a description of two different protocols (window argmax at T0, matched ticks at T1-kin) and not a paired contrast between them.
3. **T0 ADE worsened at both seeds, and the T0 longitudinal kinematics worsened with it.** R-T0 is +0.1103 / +0.0849 m (separated both); speed, accel and along-track MAE are separated-worse at T0 at both seeds (section 3.1). HYPOTHESIS: T0 scores the model on logged ego states, and a fine-tune on states pulled off the logged trajectory spends some accuracy at the logged states. The tick-0 F1 beside agrees in direction: ORC's tick-0 lat F1 is below ORC-0's at both seeds (-0.0124, -0.0140) and so is the lon F1 (-0.0525, -0.0279); those are point values with no interval, so they are a description and not a finding.
4. **The families split by axis.** At T1-kin the executed LATERAL kinematics improve under the perturbation (heading, curvature, yaw-rate MAE separated-better at both seeds) and so does plan-derived lateral agreement, while the LONGITUDINAL rows do not improve (accel MAE separated-worse at s0) and the T0 longitudinal rows are separated-worse at both seeds. HYPOTHESIS: the yaw-rate component of the perturbation is learnable and helps the executed lateral track, and the speed / acceleration component costs longitudinal accuracy at the logged states. ⛔ The PREREG states "one lever with two components: a separated effect is attributed to the perturbation as a whole, never to its lateral or longitudinal half; splitting them is a follow-up arm under its own read." This pattern is the reason such an arm could be worth a PREREG; it is not an attribution, and it is one training seed per arm.
5. **The lon F1 row is not a stable signal.** It is negative and not separated at s0 and positive, separated and above 2F at s1; the T0 fresh-age floor F = 0.0073 that the bar uses is far smaller than that sign flip, which is evidence that the floor is the wrong scale for the T1-kin F1 rows (a T0 floor is not a T1-kin training-run replicate). I do not re-score anything against a different floor; the committed bar stands.
6. **The anti-echo flags** (section 3.1) mean none of the longitudinal contrasts supports a claim that ORC or ORC-0 sets the right speed; they describe a change between two arms.

---

## 6. What this does and does not establish

* **Established (MEASURED):** under the committed table, ORC (3,000 steps of fine-tune with 30 % of samples perturbed) is HARMS relative to ORC-0, because R-ADE is separated-worse at s0 and R-T0 exceeds the bar at both seeds. All 104 controls pass.
* **Not established:** that the perturbation helps the T head (no F1 row meets its bar at both seeds); that it hurts the T1-kin planner at both seeds (R-ADE is not separated at s1); that the ADE cost is the lever and not seed noise (one seed per arm; the cost is of the size of F); any attribution to the lateral or the longitudinal half.
* **Committed next (verbatim, not re-decided here):** *"ORC is not adopted. Reported with the rows; next is NO EFFECT's arm at p = 0.1"*. Reading: "NO EFFECT's arm" is the on-policy-states (DAgger form) arm named in the NO EFFECT row, at p = 0.1. Its pre-registration was drafted and banked BEFORE this read: `raw/2026-10-26-dagger-draft/PREREG_V8F_DAGGER.md` (commit `ac139f71`). Its own §7 maps O-RECOVER HARMS to p = 0.1. It is not yet in `Project Steering/`.
* **Rule Zero:** HARMS is a waypoint, and the committed next is the lever. Unblocked and 0 GPU: building that arm's code and reader (in progress). Needs the Master Mind or the PI: launching it on box 2 (compute). Not committed, for the Master Mind to weigh: section 5 item 4 points at the follow-up the ORC PREREG itself names ("splitting them is a follow-up arm under its own read"), a lateral-only perturbation arm.

## 7. Where every number comes from

| artifact | path | sha256 |
|---|---|---|
| reader output (primary) | `E:/Projects/TanitAD-artifacts/v8f-orecover-t1/read/ORC_READ.json` | `a50dbba9fee93828b17b8b1b71335b2db73e249fd7ab9e78c9b4ea4e3577afd6` (checked by `verify_numbers.py`) |
| families, ORC - ORC-0, both tiers | `C:/Users/Admin/results_orc/ORC_FAMILIES.json` (driver `orc_families.py`) | in `bank_list.txt` |
| TACTICAL heads + STRATEGIC, T0 | `C:/Users/Admin/results_orc/PAIRFAM_T0_s0.json`, `PAIRFAM_T0_s1.json` (CLI `v8f_paired_families.py` at `07dcb513`) | in `bank_list.txt` |
| T0 evals | `E:/Projects/TanitAD-artifacts/v8f-orecover/M2b_s{0,1}_{orc,orc0}/eval_M2b-<arm>_s<S>.{json,npz}` | `ids` in `ORC_FAMILIES.json` (T0) |
| T1-kin runs | `E:/Projects/TanitAD-artifacts/v8f-orecover-t1/t1v8f_<ARM>_s<S>_m10_causal_heads.json` and `dump_<ARM>_s<S>_m10_causal_heads/` | `runs.json` carries the ticks sha256 |
| number manifest | `numbers_manifest.json`: every number, its source file and key path | |

