# PREREG -- BRAKE LOCUS: where is refcv8's cruise false-brake DECIDED?  (HEAD / SELECTION / CANDIDATE)

Sealed 2026-10-19 BEFORE `code/brake_locus_r8.py` computed any locus number.  Seal: `PREREG_BRAKE_LOCUS.sha256`; the tool REFUSES to run if the file's sha256 differs.
Tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**.  Evidence class: MEASURED (ours, banked dumps), CPU only, no model forward.
Estimator: paired / one-sample **episode-cluster bootstrap** (`pair_read_r8.paired_cell` -> `taniteval.ci`), B 2000, seed as `pair_read_r8.SEED`; answers "another draw of EPISODES" only.  It is NOT another TRAINING run (single training seed per arm, `H-ESTIM-SEED-1`) and NOT another INFERENCE run (inference seeds 0 and 1 are reported side by side, never pooled).

## 0. Honest limits of this pre-registration (read first)

* **Pre-seal look (disclosed).**  Before sealing I ran an inventory pass (scratch, not banked) that printed head-class counts and false-brake rates conditional on head class, for refcv8 and refcv7, both seeds.  The thresholds below (0.6 / 0.5 / 0.3) are the CALLER'S, from the brief.  The choices made here -- the decel-class set D, the paired reading of clause 2, the statistic for X, the precedence rule -- were fixed from code semantics (`refcv8_conditioning.py` LON6 tags) and the brief's wording, not tuned; but they were fixed AFTER that look, so **the HEAD clause-1 / clause-2 outcomes were not blind.**  The pre-registration protects the rule and the controls, not the surprise.
* **The refined fan is NOT in the banked dumps** (inventory in section 1).  Therefore (ii) and (iii) of the brief are **NOT MEASURABLE here**, and the 4-way label (HEAD / SELECTION / CANDIDATE / MIXED) cannot be issued from banked data.  What this run measures is the HEAD test, plus descriptive proxies that are explicitly NOT part of the rule.
* **(iii) is degenerate as worded.**  On the false-brake set every window's selected plan brakes by definition, so "the selected plan brakes although such a candidate exists" is the same event as (ii).  It is implemented as ONE number E (section 4) with the cuts `E < 0.3` -> CANDIDATE, `E >= 0.5` -> SELECTION, in between -> neither.  The caller should confirm this reading.
* **In refcv8 HEAD and SELECTION are not orthogonal by design.**  The tactical lon head reaches the planner THREE ways (`refc.py` `_r8_extend`, `_rank`): (a) it allocates extra candidates in proportion to its posterior (fan = 117 base + 32 appended = 149, from the banked shapes `n_anchors` 117 and `fan_term_heading` [W,149]); (b) its log-posterior enters the ranking score as `beta_lon * log p(lon6 of the candidate)` (`r8_sel`); (c) its constraint outputs enter the same score.  So "the head is wrong" and "selection picks braking" can be one mechanism.  Only an INTERVENTION separates them (design in `BRAKE_LOCUS.md`); this run is association only.

## 1. Inventory (what the banked dumps carry) -- facts, verified from the files, not rules

Banked tree `C:/Users/Admin/refcv8_eval_archive/battery/step30000ema_unc/` (refcv8 EMA-30k), S6 windows (3,668 / 139 episodes, instants 1..6 s) in `s6_s{0,1}/`, S2 windows (4,754) in `dump_s{0,1}/`.

| item | banked? | where / shape |
|---|---|---|
| refined fan, all candidates, full trajectories (`out["anchor_traj"]`) | **NO** | only `fan_term_heading_*` [W,149] = terminal HEADING of every refined candidate (no longitudinal content), and `reach_keep_*` [W,149] |
| selection scores (`out["sel_score"]`, per candidate) | **NO** | only `sel_score_max` [W] (the max) |
| candidate tags (`r8_lon6`, `r8_alloc`, `r8_kind`) | **NO** | -- |
| selected index | yes | `decisions/sel_idx`, `sel_idx_base` (pre-graft core pick), `sel_idx_{cond}` |
| GT-nearest BASE-bank anchor | yes | `decisions/a_star` (index into the 117-anchor decoded bank), `sel_agrees_oracle` |
| selected plan, all 8 model slots | yes | `decisions/plan_full_nav_true` [W,8,2]; S6 grid = slots [1,3,4,5,6,7] |
| tactical lon head (8-way logits) | yes | `dump_s{0,1}/refcv6_extras.npz`: `nav_true.tacv6_lon_logits` (the v6 tactical decoder head, `refcv6_tactical.py:635`, which feeds the planner via `hypothesis_posteriors` / `planner_feeds`) and `nav_true.lon_logits_tac` (legacy z_tac head, feeds E7/E9 only; = `decisions/lon_pred_nav_true`).  **The two heads DISAGREE in 22.6 % of windows** (argmax), so "the tactical lon head" must name which one.  PRIMARY here = `tacv6` (the one that reaches the planner and the one `H-R8-S6-LON-1`'s 109 % used). |
| head CONSTRAINT outputs (`cons_lon8`) | **NO** | they enter the ranking score (c) |
| `oracle_sel` (refined GT-nearest-anchor plan) | S2 grid only (0-2 s, `dump_s0/*.npz`) | absent from the S6 dumps |
| `sel_bank_*` (selected anchor's UNREFINED path) | **inadmissible** | row count != window count (e.g. 29 rows for 35 windows) and != number of unique selected indices: not window-aligned |
| ablated arms | seed 0 only | `refcv6_extras.npz`: `filter_off.*`, `vmax_zero.*`, `nav_{shuffled,flipped,zero}.*`, `obey30.*`; seed 1 carries `nav_true` and `obey30` only |

## 2. Populations and definitions (the instrument of `H-R8-S6-LON-1` / `L2_HEADROOM`, reused)

* Windows: S6 **compliant** windows of `step30000ema_unc`, mask = `S6_compliant` in `raw/retime_deploy/DEPLOY8_POS_step30000ema_unc_s{s}.npz` (3,251 windows / 138 episodes).  Seeds 0 and 1 SEPARATELY.
* Plan class (`s6_lon_decomp.plan_lon_class`): `v_end = |P6 - P5| / 1.0 s`, `dv = v_end - v0`; decel if `dv <= -1.5`, accel if `>= +1.5`, else cruise.  GT-cruise = plan class of `g` is cruise (1,856 windows / 135 episodes among compliant).
* **False-brake window of arm A** = compliant AND GT-cruise AND plan class of A's `os` is decel.
* Arms: **r8** = refcv8 EMA-30k `os`; **r7** = refcv7-50,400 = the embedded `b_refcv7_50400_s{s}` (bit-equal to `D:/refcv7_eval_kit/battery/step50400_a6text/s6_s{s}` -- a control).  NOT the `r7on_step50400_unc` archive (a different re-roll, differs by up to 18 m).
* **Head** = argmax of `nav_true.tacv6_lon_logits` over the 8-way v7 lon vocabulary (FOLLOW 0, CRUISE 1, YIELD_MERGE 2, BRAKE_TO 3, CREEP 4, HOLD 5, ADAPT_SPEED_FOR_CURVE 6, ACCELERATE 7), each arm's OWN head, joined on (episode, ws) [r8] / (clip_id, ws) [r7].
* **Decel-class set D = {2 YIELD_MERGE, 3 BRAKE_TO, 4 CREEP, 5 HOLD, 6 ADAPT_SPEED_FOR_CURVE}** -- the lon classes whose meaning is slowing.  FOLLOW (steady gap keeping) is NOT in D; sensitivity D+ = D u {FOLLOW} is reported.  (refcv8 masks classes 2 and 6 out of the planner's hypothesis space, `LON6_V7_IDS`; for r8 the two choices coincide if r8 never argmaxes 2 or 6 -- reported, not assumed.)

## 3. The HEAD test (the only part of the rule that banked data can answer)

* **(i)** = share of r8 false-brake windows whose r8 head argmax is in D.  Point estimate + CI per seed.
* **Clause 2** = on the windows {compliant AND GT-cruise AND r8 head NOT in D} ("head says cruise"), the r8 false-brake rate is `<=` the r7 false-brake rate **on the same windows** (paired).  Literal reading: the PAIRED POINT ESTIMATE `delta = r8 - r7 <= 0`, at BOTH seeds.  The "not separated worse" reading (`lo <= 0`) and the unpaired reading (each arm's own head-says-cruise set) are reported as sensitivities, not as the rule.
* **HEAD-test PASS** iff `(i) >= 0.6` at both seeds AND clause 2 holds at both seeds.  Otherwise **FAIL**.
* **Mandatory control P1 (permutation null for (i)):** permute the r8 head argmax across the GT-cruise compliant windows (1,000 permutations, rng seed 0); (i) is read as an association only if the observed (i) exceeds the 97.5th percentile of the permuted (i).  If not, the verdict is stamped BASE-RATE and the HEAD test is void.

## 4. (ii) / (iii) -- DEFINED NOW, NOT MEASURABLE FROM BANKED DATA (re-roll design in `BRAKE_LOCUS.md`)

For the future fan dump `F[w]` = refined fan, `[149, 8, 2]` -> S6 grid slots [1,3,4,5,6,7], over reach-surviving candidates (`reach_keep == 1`; sensitivity: all 149):
* candidate is **non-braking** iff its plan class (same rule, section 2) is not decel;
* candidate is **good** iff non-braking AND `|F[w,c,6 s] - G[w,6 s]| <= X`, **X = 2.91 m** = the median over refcv7's GT-cruise compliant windows (both seeds pooled, 3,712 values) of refcv7's 6-s final displacement error.  (The brief suggested the median ADE; the median ADE is 1.21 m and is carried as sensitivity `X_ade = 1.21 m` applied to candidate ADE.  FDE at 6 s is used as primary because the quantity thresholded is a 6-s distance.)
* **E** = share of r8 false-brake windows (by the re-rolled N arm's own plan class) with at least one good candidate.  `E_ctrl` = the same on a random sample of GT-cruise non-false-brake windows.
* **(ii) = E; (iii) = E** (degenerate, see section 0).  `E < 0.3` -> CANDIDATE fires; `E >= 0.5` -> SELECTION fires; `0.3 <= E < 0.5` -> neither fires.

## 5. Label rule (committed)

Let T_HEAD = HEAD-test PASS (section 3), T_SEL = (E >= 0.5), T_CAND = (E < 0.3).  Label = the single one that fires; **MIXED** if none or more than one fires.  **From banked data alone only T_HEAD is evaluable, so the banked-data label is `UNDETERMINED`**, reported together with the HEAD-test verdict and the sentence "SELECTION vs CANDIDATE not measurable without the fan".  No goalpost is moved after the numbers: a HEAD-test FAIL is reported as FAIL.

## 6. Descriptive supplements (pre-registered as DESCRIPTIVE; they do not enter the rule)

* S1 head-decel share on GT-cruise compliant windows, r8 vs r7 (each own head).
* S2 coupling `P(plan brakes | own head in D)` on GT-cruise compliant windows, r8 vs r7.
* S3 paired same-window decomposition of the r8 - r7 false-brake excess by r8's head state (in D / not in D); the strata contributions must sum to the total (identity control).
* S4 selection provenance on r8 false-brake windows vs GT-cruise non-false-brake windows: share selected from the ALLOCATED group (`sel_idx >= 117`), `sel_idx == a_star`, `sel_idx != sel_idx_base`; r7 `sel_idx == a_star`.
* S5 seed-0 banked ablations: false-decel rate with the ceiling filter OFF (`filter_off.traj`) and the speed label zeroed (`vmax_zero.traj`) vs native.
* S6 sensitivities: legacy head `lon_logits_tac`; D+ (FOLLOW added); head == CRUISE only; unpaired and "not separated worse" readings of clause 2.

## 7. Controls (the tool exits non-zero if any fails)

* C1 literals (typed from `L2_HEADROOM.json`, not derived): compliant 3,251 / 138; GT-cruise 1,856 / 135; r8 false-decel **0.265625 (s0) / 0.257543 (s1)**; r7 **0.091056 (s0) / 0.093750 (s1)**.
* C2 alignment: `ws` and `os` bit-equal the DEPLOY8 npz `S6_ws` / `S6_os`; compliant mask equal across seeds; r7 embedded arm bit-equal to the `step50400_a6text` S6 `os`; **head join**: extras `nav_true.traj[:, slots]` bit-equal `os` (r8) and bit-equal the r7 arm (r7) -- a wrong join cannot pass this; extras `nav_true.sel_idx` == decisions `sel_idx`.
* C3 identities: legacy-head argmax == `decisions/lon_pred_nav_true` on every window; S3 strata sum to the total.
* C4 decision-function arms through the rule (answers fixed by construction): an oracle head (argmax in D exactly on r8 false-brake windows) -> clause 1 passes; a never-decel head -> clause 1 fails; a rule-logic table of fake cells -> PASS only when both clauses hold at both seeds.
* C5 mutations of the instrument, each must go RED (some control or literal fails): decel threshold 1.5 -> 1.0; head join shifted by one window; CRUISE and BRAKE_TO logit columns swapped.
* C6 self-test of the section-4 fan function on a synthetic fan with known E, plus a mutation (X -> 0) that must go RED.
