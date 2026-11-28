# LEVER SHEET, TanitAD v8 proof study (2026-11-19)

For the session that owns the repository (D3 of `dagland/COORD_V8F_2026-11-19.md`: this study never edits `stack/`; the owner lands).
Notation: rN = register row H-V8F-PROOF-N. Evidence class of every number: MEASURED (row id given). Tiers: T0 = diagnostic on logged frames; T1-kin = kinematic rollout on the model's own actions, the lead does not react. Folders are under `F:/TanitAD-scratch/v8proof/`. STRATEGIC family: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and the readers have no strategic instrument), r90 and r88; stated once, true for every block below.

## 1. Levers

| lever | acts as | flag(s), default | status | rows |
|---|---|---|---|---|
| LEADSUP, supervised lead head | training loss term + head (train-time label only; driving path never reads it) | `--lead-sup {off,on}` off; `--w-lead-sup` 0.0; `--lead-hist` 5; `--lead-sup-detach {off,on}` off | READY WITH CAVEAT | r52 (outside 56-91, same register), r68, r77 |
| ACTCONS, action consistency, low weight | training loss term on the predictor | `--w-act-cons` 0.0 (off); `--act-cons-max` 1.5 | READY WITH CAVEAT | r80, r86, r88 (r76 for LS bases) |
| DISTRULE with gap safety (margin / quantile map) | inference-time rule on the plan | `--dist-rule {off,label,head,never,push,plumb}` off; `--gap-margin-m`, `--gap-map`, `--head-p-min` (each required by its wrapper) | READY WITH CAVEAT | r57, r58, r59, r62 |
| QHEAD, trained low-quantile gap head | head fitted on frozen states, used inside the rule | `--qhead <file>` (empty = none), `--q-equals-head` (off) | READY WITH CAVEAT | r89, r90 |
| ACTLEVEL, level term | training loss term | `--w-act-level` | NOT READY (no confirmed result in rows 56-91) | r86 (pre-registered only) |
| Rule in the training target / on own plan | training target / loss term | none handed | NOT FOR INTEGRATION | r61, r63, r65, r64, r66 |
| Lead read as plan-head input | plan-head input | none handed | NOT FOR INTEGRATION | r71, r74, r75 |
| Planner on imagined rollouts | planner | none handed | NOT FOR INTEGRATION | r81, r91 |
| MOTION (RR), TARGET (PF), finer grid | training form / target / probe | none handed | NOT READY | r60, r67, r72, r82, r87 |

## 2. Blocks

### 2.1 LEADSUP (prerequisite of the vision-fed rule: its presence and gap reads)
* What: a small head (per-frame Linear 640->64, 5 frames concatenated, Linear 320->128, GELU, Linear 128->3: presence, gap/50 m, closing/10 m/s; +82,499 parameters on the tiny model) reads the level-1 grid states and is trained on privileged labels from the label join (`spatial/leadsup/HANDOFF.md`).
* Flags (from `spatial/leadsup/HANDOFF.md`, `LEADSUP.patch`): `--lead-sup` default `off` (off builds nothing); `--w-lead-sup` default `0.0` (on with 0 is refused); `--lead-hist` default 5; `--lead-sup-detach` default `off`. Arm types: LS = grid + `--lead-sup on --w-lead-sup 1.0`; LSD = LS + detach on (`ARMS_LEADSUP.txt`).
* Port: `LEADSUP.patch`; `PINS.txt` 11 lines (3 changed: `v8f_mini.py`, `v8f_mini_train.py`, `v8f_mini_data.py`; 2 new: test file, `lead_dump.py`; 6 unchanged pinned). Needs `join_train.npz` with lead arrays.
* Tests: 16 new, 103 passed in `tests.log`. Mutation list: not established in the rows or the folder (no mutation file); `checks/t3_checks.json` has reach checks, t1 identity max diff 0.0 (ran before the last synthetic edit, per HANDOFF).
* Read on: LS, LSH (weight 10), LSD, seeds 0 and 1, 19,164 steps (r52). Names: `SUP-CLOSING-FLAT`, `SUP-SLOT-FLAT-N1`, `SUP-GRADIENT-EQUAL`, `SUP-NO-COST` (r52, T0 and T1-kin). Gap R2 LS 0.724 / 0.580, presence AUROC 0.904 to 0.922; no cell worse in the three families (r52).
* Caveat carried: control k5 of card ADD24 FAILED for all six arms, so by the card as written no name is given; names come under notes N1 and N2 (r52). Closing speed R2 intervals include 0 (r52). Two seeds per type.
* D3: (a) names under notes, replicate = seed 1; (b) 16 tests, mutation list not established; (c) four flags need entries, owner's work (the patch carries its own trainer-side G-DVB check `bad_lead_sup`).

### 2.2 ACTCONS at weight 0.0001
* What: a kinematic term: the predictor is rolled 10 steps under logged actions with and without a random acceleration offset, and the imagined lead gap must move by the kinematic amount. Without it the imagined gap moves the wrong way (slope -0.468 pooled, LS, r68; -0.895 RLFT, r80).
* Flags (`actcons/ACTCONS.patch`, `ARMS_ACTCONS.txt`): `--w-act-cons` default `0.0` = off; `--act-cons-max` default 1.5 (non-default needs `--w-act-cons` above 0). Named dose: 0.0001 (arm RLAm). Fine-tune argv: 6,000 steps from the base checkpoint.
* Port: `ACTCONS.patch`; `PINS.txt` 20 lines (1 changed: `v8f_mini_train.py`; 2 new: `models/act_cons.py`, `tests/test_v8f_act_cons.py`; 17 unchanged pinned).
* Tests: 14 new, 173 passed in `tests.log`. Mutation list: HANDOFF names a3 planted checks and a4 mutations red; a counted list is not established in the rows.
* Read on: rollout-trained bases RL_s0, RL_s1 (r80) and RL_s2, RL_s3 trained from scratch (r88), one run per base. Names: `LOW-WEIGHT-CONFIRMED` = `LOW-WEIGHT-RESPONDS-CONFIRMED` + `LOW-WEIGHT-KEEPS-LEVEL-CONFIRMED` + `LOW-WEIGHT-NO-COST` (r88). Slope RLAm +0.820 [+0.766, +0.878] against RLFT -0.746 (bar 0.5 per base; 0.835 and 0.805), bases 0 to 3: 0.805, 0.817, 0.835, 0.805 (r88, T0).
* Costs (r88, T1-kin, against RLFT): LONGITUDINAL speed -0.001 m/s [-0.054, +0.051], along-track -0.020 m, unsafe rate +0.006 [-0.010, +0.021]; LATERAL cross-track +0.025 m [-0.001, +0.051], heading +0.41 deg [+0.11, +0.76] inside spread 4.56 (stated); TACTICAL lateral decision error rate +0.024 [+0.012, +0.040] inside spread 0.104 (stated), joint +0.001.
* Caveats carried: the level bar (ratio 1.15) is met narrowly: ratios 1.077, 1.094, 1.141, 1.059, base 2 passes by 0.009 (r88). Weights 0.001 and 0.01 respond but miss the level (ratios up to 2.580): `DOSE-BREAKS-AT-0.001`, `DOSE-BREAKS-AT-0.01`, `DOSE-NO-GAIN-ABOVE-LOW` (r86, r88). Weight 1.0 loses the imagination: `ACT-CONS-FLAT-RL`, `ACT-CONS-LOSES-IMAGINATION-RL` (r80). Response is read around the logged actions; on the model's own plans RLAm reads 0.495 and 0.479 against bar 0.5 (r80, r81). Response about 0.82 of the kinematic size, not 1. Position error over 6 s of a control rerun differs +0.030 m [+0.002, +0.057] (run-to-run spread, r80).
* D3: (a) met, four bases; (b) 14 tests, mutation list not counted; (c) `--w-act-cons`, `--act-cons-max` need entries.

### 2.3 DISTRULE with gap safety (inference-time)
* What: a rule slows the plan along its own path against an imagined constant-speed lead; the gap comes from the LEADSUP head, lowered by a safety margin. Constants are module constants of `dist_rule.py` (D 3.0 m, TAU 1.2 s, B_MAX 6.0 m/s2), no flag (`distrule/DISTRULE_DEFS.md`, `dist_rule.py`). Wrappers import the pinned `v8f_t1_eval.py` (sha256 `61e140ed...`) and replace two module globals; the tool file is unchanged. Closing speed is set to zero (`ZERO-CLOSING-EQUAL`, four arms, r62), so two reads are used: presence and gap.
* Flags: `--dist-rule` default `off` (installs nothing); `v8f_t1_eval_rule_margin.py` `--gap-margin-m`; `v8f_t1_eval_rule_qcal.py` `--gap-map`; both `--head-p-min`; `--floor {card,cap-only}` default card; `v8f_t1_eval_rule_mix.py` `--mix-presence`, `--mix-gap`, `--mix-closing {label,head,zero}`.
* Port: `dist_rule.py`, the three wrappers, `margin_calib.py`, `qcal_fit.py`, `qcal_map.py`, `cmp_dumps.py`. No PINS file in the folder: not established. Tests: `test_dist_rule.py` 22 tests (HANDOFF: 22 passed). Mutation list: HANDOFF names three mutations red, no counted list.
* Read on LS and LSD, seeds 0 and 1 (r58, r59), 642 lead windows (distance), 2,399 windows (cost), T1-kin. Names: `MARGIN-REACHES-ORACLE`, `MARGIN-CUTS-MORE`, `MARGIN-COSTS` (r58); `QCAL-REACHES-ORACLE`, `QCAL-U-EQUAL` against the fixed margin, `QCAL-CHEAPER` for LS (-0.13 m, inside spread 0.4002), `QCAL-COST-EQUAL` for LSD, `QCAL-COSTS` (r59). Unsafe rate vs no rule: margin LS -0.0712 [-0.1118, -0.0326], LSD -0.0827 [-0.1202, -0.0483] (r58); map LS -0.0596, LSD -0.0685 (r59). Gap read is about 80 percent of the shortfall (r57).
* Costs vs no rule (r58, r59): LONGITUDINAL and TACTICAL worse; 6 s position error +0.945 / +0.858 m (margin), +0.811 / +0.877 m (map), against +0.18 m for the label-fed rule; no LATERAL cell worse. Contact: the map lets more through than the fixed margin (LS +0.0125 [+0.0049, +0.0213], r59); presence threshold alone does nothing (r58).
* Caveats: the rule has a price against no rule; the lead does not react, so braking for a lead read too near is not rewarded; calibration from train probe clips the heads were trained on; two seeds per type (r58, r59).
* D3: (a) met on two seeds per type; (b) tests yes, mutation list not counted; (c) eval-time wrappers, not trainer flags: whether the owner wants gate entries is not established in the rows.

### 2.4 QHEAD, trained low-quantile gap head (replaces the hand-built map)
* What: a head on frozen states trained with a pinball loss at 0.20, early stopping on 200 select clips, a constant shift fitted there, used as `min(g_head, q_head)` in the rule (r90). First run without selection failed coverage on held-out clips (0.42 to 0.48) and named nothing (r89).
* Flags/tools (`qhead_n1/QHEAD_N1_DEFS.md`): fit `qhead_fit_n1.py` with `--fit-clips --select-clips --check-clips --select-every`; rollout wrapper `qhead/v8f_t1_eval_rule_qhead.py` `--qhead` (empty default), `--q-equals-head`, `--head-p-min`. Port: `qhead_model.py` (sha256 `f6ff755e...`), `qhead_fit_n1.py`, `qhead_h3_n1.py`, names scripts. No PINS file: not established.
* Tests: n1 files 14 + 15 + 8 + 5 test functions (counted from the files); first run 16 + 11 + 8. Mutation lists: `MUTATIONS_QHEAD.json` 19 of 19 red (r89 h6), `MUTATIONS_QHEAD_N1.json` 7 of 7 red.
* Read on: two checkpoints per type (LS, LSD), T0 for fits, T1-kin for rollouts (r90). Names `QHEAD-U-EQUAL-QE`, `QHEAD-COST-EQUAL-QE`, `QHEAD-CUTS-MORE-QF` (narrow), `QHEAD-U-EQUAL-QF`, `QHEAD-CHEAPER-QF`, `QHEAD-COSTS`. LS first fit: unsafe rate vs no rule -0.0658 [-0.1055, -0.0269]; LSD dose step -0.0783 [-0.1237, -0.0388], 6 s error vs map -0.199 m [-0.391, -0.017]; LS dose step vs map -0.0151 [-0.0297, -0.0006] (r90).
* Costs: LONGITUDINAL and TACTICAL worse vs no rule (`COST-LONGITUDINAL`, `COST-TACTICAL`); LATERAL none named (r90).
* Misses carried (r90): contact rate above the label-fed rule in all three reads (+0.0117 to +0.0132); coverage on evaluation windows 0.10 to 0.15 against 0.20 on check clips; LSD not named at the first fit (check coverage 0.1086); LS dose step run once under its own names; a rule without cost against no rule is not shown.
* D3: (a) two checkpoints per type; (b) yes, with lists; (c) not established, as 2.3.

## 3. Not handed over
* Rule inside the training target: `RULE-TRAINED-FLAT`, `-6K`, weight 4 flat (r61, r63, r65); no unsafe-rate move.
* Rule on the model's own plan as a loss: lowers unsafe rate only by slowing everywhere, +2.67 m (squared), +2.12 m (linear); weight 4 +3.86 m and +3.29 m, LATERAL cells worse for squared (r64, r66).
* Lead read as plan-head input: `LEAD-INPUT-FLAT` after fine-tune (r71); `LEAD-INPUT-SCRATCH-UNUSED` from scratch (r74); `PLAN-ERROR-NOT-LEAD` (r75).
* Planner on imagined rollouts: cuts unsafe rate but is not better than extrapolation, worse at new bases (+0.013 [+0.004, +0.024] on four bases, r91; r81).
* Level term: pre-registered (r86); no arm read in rows 56-91.
* Consistency weights 0.001 and 0.01 (r86), weight 1.0 (r80); also LS-base fine-tunes lose the level at every weight 0.0001 to 1.0 (r76).
* MOTION RR: lead information not confirmed on four seeds (`MOTION-ADDS-NOTHING-4`, r72); one-step gain holds, flags not established. TARGET PF: `TARGET-FLAT`, position at seed 0 only (r82). Finer grid: NOT NAMED (r87). TOKPROBE: no outcome (r56). Fixed margin alone: kept as a dose reference, price above the map for LS (r58, r59).

## 4. Re-measure in the integrated candidate
* A lead that reacts (every T1-kin number here has a passive lead; r58, r59).
* Gap read quality: the limit of the rule (r57); contact and coverage misses of QHEAD (r90).
* More seeds: one training run per base for ACTCONS (r88); two checkpoints per type for the rule and QHEAD.
* Response of ACTCONS on the model's own plans, bar 0.5 missed by 0.005 (r80, r81, r91).
* The 10k corpus, and the STRATEGIC family (NOT COMPUTED here).
* Cost against no rule of every distance-rule form (r58, r59, r90).
