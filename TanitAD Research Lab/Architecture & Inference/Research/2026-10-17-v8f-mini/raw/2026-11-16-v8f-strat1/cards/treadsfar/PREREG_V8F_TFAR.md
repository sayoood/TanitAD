# PREREG_V8F_TFAR (FROZEN 2026-11-18 by its author for the MM, NOT BUILT, NOT RUN): T reads S's 120 m point as a fourth checkpoint

Status: written 2026-11-18 by the pre-registration designer for the MM, after the CKPT120 read (landed 6cfcdf6b) and before any build, label table or arm of this card. Nothing here is built or launched. sha256 in `PREREG_V8F_TFAR.sha256`. It is the committed next step of CKPT120's **S120-FOLLOWS** row: *"the T-reads-far card (S's 120 m point as a fourth checkpoint token of T; PI go)"*. PI decisions: `QUESTIONS_TFAR_DGOAL.md`. Build list: `BUILD_GAPS_TFAR.md`.
Evidence classes: MEASURED (M, artifact named), PUBLISHED (P), INHERITED (I), ESTIMATED (E), HYPOTHESIS (H), UNVERIFIED (U). Tiers: T0 (diagnostic), T1-kin (primary). Design artifacts: this folder (section 12).

## 0. Decisions for the MM (escalated)

1. **Form of the fourth checkpoint.** T's three checkpoints already enter T as ONE token, `e_ck = Linear(12, 256)` over (x/50, y/50, ψ, ok) of 30/50/80 m, and that layer is SHARED with O (`v8f_mini.py:887`, `_ck_cond` :977-979, called by `run_t` :1046 and `run_o`). The far point joins T's token through a new zero-initialised `e_ck_far = Linear(4, 256)` (1,280 params), ADDED inside `run_t` only. This is a widening of T's `e_ck` from 12 to 16 inputs with the new columns at zero, without touching O's copy. Step 0 is then bit-identical to H2 (x + 0.0 = x), so the arm differs from H2 by the lever alone. The appended-token form (a fifth token, the OTBAND `e_stok` precedent) changes the step-0 forward and is kept as a named branch (section 8).
2. **Teacher forcing follows the three checkpoints.** With `--detach-upper` (recipe), T reads S's far point detached. With probability p_gt = 0.2 it reads RC-A120 instead, on the SAME per-sample draw as 30/50/80 (no extra RNG draw), masked by `rc120_ok & has`. Eval never teacher-forces.
3. **The design measurement (section 2) predicts a small lever.** Rule 5 of RULE ZERO asks for the largest measured lever. This card is still worth running: it costs about 1.6 pod-h, it is the card's committed next step, and its ORACLE row (T fed RC-A120 at T0) says which lever comes next whatever the outcome. The prediction is written in section 8 before any number.
4. **Deciding cells are named, few and longitudinal.** The labels show the far point carries information T lacks mainly about DECELERATE/STOP ahead of a bend beyond 80 m, not about the lateral side (section 2). Rig noise is 5 of 48 single cells between same-flag replicates (C0REP, M). So four deciding cells are fixed now, all on one stratum, and the 24-cell table is a keep/cost table.
5. **Attribution arm TN.** T reads the raw route text `nav_s` through the same zero-init additive form. TF − TN separates "S's processed far point" (the hierarchy) from "the route text reaching T" (flat). This lifts CKPT120's isolation rule "T never reads nav_s" for that one arm only: a PI item (Q3).
6. **Base and branch.** The base is H2, banked at `aedf8716` on `agent/v8f-ckpt120-20261117`. The CKPT120 levers are absent at `4dedac45` and `f95de67a` (`grep -c s_far_out` = 0 at both, 15 at `aedf8716`, M). The build branches from `aedf8716` (`BUILD_GAPS_TFAR.md`).
7. **Gate route.** As CKPT120: a study checker `tfar_check.py` mints `tfar-gate-token-v1`, bound to commit, argv sha256 and every data sha256. The runner refuses to start without a matching token (PI rule 2026-09-26).
8. **Init: matched from M2b, not continued from H2.** TF trains 3,000 steps from M2b exactly as H2 did, so it differs from H2 by the lever alone. The cost of this choice is that S's far head is learned in the same run that T learns to read it. The alternative (init from H2's checkpoint) needs a matched continuation control H2c, and is fixed now as the TFc / H2c branch, not the primary arm.

## 1. Question

Fine-tuned on the H2 recipe, does T, reading S's 120 m point as a fourth checkpoint, set its longitudinal tactical decision better ahead of bends that lie beyond 80 m? Does it read the point at all? And does it keep all four families: H2's STRATEGIC route following and route setting, and the LONGITUDINAL, LATERAL and TACTICAL driving cells?

## 2. Design measurement (labels and banked H2 T0 arrays only, zero training)

### 2.1 Where the far point carries information T does not already read (M, `TFAR_LABELS.json`, `measure_tfar_labels.py`)
* T reads the tactical nav token, which carries the side only while d <= max(30 m, 6 s x v), and S's 30/50/80 m checkpoints. The 120 m point adds two things: the side beyond the token, and the road geometry between 80 and 120 m.
* **Side, on the 122 scorable 120 m windows (22 episodes) where the token does not carry it** (control: 341 / 219 / 122 reproduce the CKPT120 card exactly):
  * T's lateral label is LANE_KEEP on 88, TURN_L on 3, TURN_R on 17, and ignored on 14. There is no lane change.
  * Of the 20 windows with a sided label, 12 match the far side and 8 are opposite.
  * ⇒ A planted T that knows the far side can gain TAC-LAT on at most **12 of the 2,399 grid windows (0.5 %)**.
  * Across all 2,261 RC-A120-valid windows, the eval label carries **0 LANE_CHANGE** labels (LANE_KEEP 1,839, TURN 191, ignored 231).
* **Longitudinal, same 122 windows:** DECELERATE 62 and STOP 5, so 67 of 122 (0.549) decelerate-or-stop.
* **Geometry stratum FB (far bend only): |ψ_RC120| >= 25° and |ψ_RC80| < 25°,** with `has`, RC-A80 and RC-A120 valid.
  * Size: **202 windows / 48 episodes**. The token is on in 13 of them. 201 / 47 have a valid lon label.
  * Decelerate-or-stop share: **0.332 (67/202)**, against **0.139 (208/1,498, 115 ep)** where the road does not bend by 120 m, and 0.389 (230/592) where it already bends by 80 m.
  * Lateral label: LANE_KEEP on 168 of the 202.
  * Control: RC-A120 validity identity with the grid truth, 0 mismatches.
* ⇒ FB is the stratum where the 120 m point holds longitudinal information that the 80 m checkpoint does not. It has 48 episodes, against the 27 of the S_CKPT120 set.

### 2.2 Does H2's PREDICTED far point carry it? (M, `H2_FAR_PROBE.json`, `probe_h2_far_info.py`)
* **What T would read at inference.** Logistic probe, episode-grouped 2-fold split, on the banked H2 T0 arrays (real nav). Target: lon label in {DECELERATE, STOP}. The base features are T's own lon softmax (no far input yet), S's predicted 80 m heading and v. The far features are |ψ_far| and |y_far| / 100.
* **Probe AUC, base → base + far:**

  | stratum | H2_s0 | H2_s1 |
  |---|---|---|
  | all RC-A120-valid (2,155 / 132) | 0.742 → 0.743 | 0.753 → 0.753 |
  | FB (201 / 47) | 0.699 → **0.729** | 0.682 → 0.686 |
  | side announced, token off (323 / 34) | 0.789 → 0.796 | 0.665 → 0.663 |

* This is a post-hoc probe on the evaluation set. It is beside evidence for ranking levers, never a result. ⚠️ A negative from a linear probe is not a negative about learnability (TRAPS).
* **Consistent with CKPT120's route setting** (M, `RESULT_CKPT120.md`): H2's far point is 15.15 m / 15.42 m from RC-A120, only 0.53 / 0.57 m closer than its own 80 m point carried forward (EXTRAP). Most of its geometry is already in the 80 m checkpoint that T reads. Its side, by contrast, is strong: S_CKPT120 real is 0.885 / 0.811 on the token-off stratum.

### 2.3 What this predicts (H)
* TAC-LAT has almost nothing to gain (section 2.1).
* The longitudinal lever exists in the labels on FB. S's predicted point carries little of it at one seed and some at the other.
* The ORACLE row (T fed RC-A120 at T0) bounds what a better S far point would give. It routes the next lever (section 8).

## 3. Admissibility and the leak test

* **What the new input is computed from.**
  * At inference: S's far point, `s_far_out` on `head_s`'s hidden layer (`v8f_mini.py:1026-1031`), read from S's inputs. Those are the adapted camera tokens, the measured ego motion `p10f` (PI yaw-rate ruling), the tactical nav token, the limit, and `nav_s` (the route text's next turn at any distance).
  * At train: the same point, or RC-A120 with p 0.2 (teacher forcing).
  * Nothing else reaches `e_ck_far`. A source pin and test **T-SRC-FAR** check this.
* **Is it the situation classifier's output, or derived from it? No.**
  * v8f-mini has no situation classifier. The inference read set is pinned by `assert_no_situation_classifier_input` (`v8f_mini.py:1543`).
  * The H2 base has no detected-info head (argv `--ctx-heads off`, M, `ckpts/H2_s0/argv.json`).
  * S reads no output of T. T's tactical classes (TURN / LANE_CHANGE, the semantic neighbours of the situation detectors) are downstream of the far point, never upstream.
  * Information-disjointness is pinned by **T-ISO-UP**: the gradient of T's loss terms on every S parameter is **exactly 0.0** under `--detach-upper`.
* **Is it computed from anything the measured quantity also produces?**
  * The measured quantities are T's lon/lat decisions, O's trajectory, and S's far point. All three are scored against labels built from the ego's future path.
  * At inference the far token is a model output, not a label. RC-A120 is never read by `forward_infer` (key guard **K-KEYS**; `forward_infer` runs with p_gt 0.0 and `use_labels` False, `v8f_mini.py:1174-1182`).
  * **The shared source is the route text.** `nav_s` is computed along the driven path, so the far point carries the driven route's next turn. T's TURN labels and the far side then share one source, the driven path. This is the same optimism T's own tactical nav token already carries.
  * ⇒ Every TACTICAL and LATERAL number is worded **"with the route supplied along the driven path"**. T's agreement with the `nav_s` side is printed beside every TACTICAL row (the echo share).
  * The longitudinal deciding cells are scored on FB. FB is defined from geometry, so it is not a side echo. The text still tells S that a turn exists, so FB cells carry the same wording.
* **Labels may use ego; inference is vision-only (PI 2026-08-03).** RC-A120 is a label, used only for teacher forcing and scoring. T reads nothing new from outside the model; the far point derives from the recipe-of-record inputs. Nav is an input at training and inference (PI).
* **TN arm.** T reads `nav_s` directly. It is admissible for the same reasons (route text, no classifier output) and has the same optimism.
* **Parity:** corpus `train4572 --exclude 540:600` and eval-141, unchanged.

## 4. Levers (built default off; every flag has a G-DVB entry; all defaults = the `aedf8716` build, bit for bit)

| flag | default | arm value | what it builds |
|---|---|---|---|
| `--t-far` | off | on | `e_ck_far = Linear(4, 256)`, zero-initialised (no RNG draw), key prefix `e_ck_far.` in `NEW_LEVER_KEY_PREFIXES`. Inside `run_t`: T's checkpoint token = `e_ck(x12) + e_ck_far([x/50, y/50, ψ, ok])` of the far point in T's tick frame. O's `_ck_cond` call is untouched. G-DVB refuses `--t-far on` with `--s-far off` (nothing to read) or without `--detach-upper`. |
| `--t-far-lr-mult` | 1 | 10 | the fresh layer's lr (1e-3 inside the 1e-4 fine-tune), the `--s-far-lr-mult` convention. Fixed, not tuned. |
| `--t-far-input` | pred | zero (TZ only) | zero: the four features are 0.0 (ok = 0) at train and eval. The layer is built and trained, but receives no information. |
| `--t-far-variant` | real | roll (R3 only) | at TRAIN only, T's far features are rolled by one sample along the batch after teacher forcing (a within-batch derangement). Eval feeds the real point. The deliberate regression. |
| `--t-far-tf` | same | same; off (branch only) | same: teacher-forced with RC-A120 on the three checkpoints' own per-sample draw (decision 2). off: T always reads S's predicted point. |
| `--nav-strat-t` | off | on (TN only) | `e_nav_s_t = Linear(5, 256)`, zero-init (1,536 params), ADDED to T's nav token: `e_nav(nav) + e_nav_s_t(nav_s)`, dead-reckoned in T1 like `nav_s`. Lifts CKPT120's T-side `nav_s` key guard for this flag only; O never reads `nav_s`. |
| `--nav-strat-t-lr-mult` | 1 | 10 (TN only) | as `--t-far-lr-mult`. |

Unchanged: `CKPT_ARCS` (30, 50, 80), `head_s`, `s_far_out`, `e_nav_s`, `w_ckpt_far` 0.3333, `--s-far-lr-mult 10`, `e_ck` (shared with O), p_gt 0.2, `--detach-upper`, every DAgger setting, the N1 base flags.

## 5. Arms (LoRA pod, one job at a time, 3,000 steps each from M2b, matched to H2)

Base of record: **H2** = N1 (text nav, no ctx heads, OBD-CTL recipe, rebase rule (a) of CKPT120) + `--s-far on --w-ckpt-far 0.3333 --s-far-lr-mult 10 --rc120-npz/sha256 --nav-strat on --nav-strat-npz/sha256 --nav-strat-variant real` (M, `ckpts/H2_s0/argv.json`). H2_s0 and H2_s1 are banked (HF private `Sayood/tanitad-v8f-strat1-runs`, ckpt sha256 `26a65fd4…` / `af6d5eaa…`, M) and are not retrained.

| priority | arm | = | init / DAgger bank / seed |
|---|---|---|---|
| 0 | H2_s10, H2_s11 | CKPT120's own SEEDS-DISAGREE arms (its card section 8, under its go). Needed here as the 4-seed comparators | M2b_s0 / s0 / 10; M2b_s1 / s1 / 11 |
| 1 | **TF_s0** | H2_s0 + `--t-far on --t-far-lr-mult 10 --t-far-input pred --t-far-variant real --t-far-tf same` | M2b_s0 / s0 / 0 |
| 2 | **TF_s1** | as TF_s0 | M2b_s1 / s1 / 1 |
| 3 | **TF_rep** | as TF_s0, only `--seed 2` (H1_rep's form): the training-run floor | M2b_s0 / s0 / dagger 0, seed 2 |
| 4 | **R3_s0** | TF_s0 + `--t-far-variant roll` (must fail) | M2b_s0 / s0 / 0 |
| 5 | **TZ_s0** | TF_s0 + `--t-far-input zero` (token off: path built, no information) | M2b_s0 / s0 / 0 |
| 6 | **TN_s0** | H2_s0 + `--nav-strat-t on --nav-strat-t-lr-mult 10` (attribution, PI Q3) | M2b_s0 / s0 / 0 |

* Priority order, not a dependency chain. A chain stopped after arm 2 still yields the two-seed read with the CKPT120 floor; after arm 5, the full read.
* **Conditional arms**, each only on its committed trigger (section 8), about 16 min each (E): TF_s10 / TF_s11 (rep's form); TN_s1; TFnt_s0 / s1 (`--t-far-tf off`); TFc_s0 / s1 with H2c_s0 / s1 (init from the banked H2 checkpoint, + 3,000 steps with and without the lever: matched continuation, 4 arms); TFa_s0 / s1 (appended-token form); TFd_s0 / s1 (`--s-adapter-detach on`).
* **Rebase rules, fixed now:**
  * (a) If CKPT120's SEEDS extension reads H2's tactical 2-6 s accel MAE as a cost at >= 3 of 4 seeds (its COSTS-FAR), the base becomes H2 with `--s-adapter-detach on`, and TF re-runs on it.
  * (b) If NAVRB2 is adopted (N1R base), TF re-runs on that base in a separate card. That rebase needs the merge of `f95de67a` (section BUILD_GAPS 1).
  * Each read keeps its own base.

## 6. Read

* **Tiers.** TACTICAL, LONGITUDINAL and LATERAL: **T1-kin** (self-action open loop, m = 10, causal bucket, dead-reckoned nav and `nav_s`, logged scene; S's far point held and transformed per tick like the three checkpoints). Grid: eval-141, 2,399 windows / 141 episodes, per seed. STRATEGIC (S's route following and route setting): **T0 at the window's first tick**, as CKPT120. The T0 far-token variants are T0 diagnostics, never a capability claim.
* **Far-token variants** (per TF / R3 / TZ arm, zero training):
  * **real**: S's own far point.
  * **donor**: S's far point taken from the donor window (`donor_indices` seed 0, the CKPT120 donor map). This swaps T's far token only; S's nav is untouched.
  * **zero**: features 0, ok 0.
  * **ORACLE**: RC-A120 in T's frame where valid, else S's point with the window flagged. ORACLE is T0 only and is never quoted as driving performance.
* **Estimator.** `taniteval.ci.paired_episode_cluster_bootstrap`, B 2,000, seed 0, the CKPT120 reader's `ci.py` (same sha256). Three questions are named:
  * another draw of EPISODES: the bootstrap;
  * another TRAINING RUN: floor F, then the 4-seed rule;
  * another INFERENCE RUN does not apply: deterministic planner; every arm dumped in one environment at one thread count (memory: T1 dumps drift at other thread counts).
* **Floor.** F per cell = max(|TF_rep − TF_s0|, |H1_rep − H1_s0|), the latter from the banked CKPT120 dumps. For STRATEGIC 120 m cells, F120 = the CKPT120 F120 (rd 0.0059, bal 0.0803, raw 0.0147, position 0.0856 m, M) or |TF_rep − TF_s0|, whichever is larger.
* **Cost / gain.** A cost is separated on the worse side with |Δ| > F. A gain is separated on the better side with |Δ| > F. Any adoption claim needs the **4-seed rule**: >= 3 of 4 seeds, one per init (STRAT1's COST4 / GAIN4).

### 6.1 Deciding cells (TF − H2, per seed, T1-kin, stratum FB = 202 windows / 48 episodes, fixed from labels in section 2.1)
| id | family | cell |
|---|---|---|
| D-LON-1 | LONGITUDINAL | tactical band (knots 11-30, 2-6 s) speed MAE on FB |
| D-LON-2 | LONGITUDINAL | tactical band acceleration MAE on FB |
| D-TAC-1 | TACTICAL | TAC-LON accuracy on FB (201 / 47 with a valid lon label) |
| D-TAC-2 | TACTICAL | DECELERATE-or-STOP recall of T's lon decision on FB (67 positives) |

* **T-READS** (does T use the token?): T0, real − donor on D-TAC-1, per seed. The share of FB windows where T's lon argmax changes between real and zero is printed beside.
* LATERAL and STRATEGIC have **no deciding gain cell**. LATERAL is predicted flat (section 2.1); STRATEGIC is unchanged by construction under `--detach-upper`, except through the shared adapter. Both are KEEP families with the cost rule. A LATERAL gain is reported if it occurs, never decided on.

### 6.2 The four families, reported in full (per seed; TF − H2, R3 − H2, TZ − H2, TN − H2, TF − TN)
* **LONGITUDINAL, LATERAL, TACTICAL:** the 24-cell CKPT120 / STRAT1 table on all 2,399 windows. Per band (operative knots 1-10, tactical 11-30): ADE, FDE; speed, acceleration, along-track, headway and time-gap MAE; heading, curvature, yaw-rate and cross-track MAE; TAC-LAT and TAC-LON accuracy. The same table is printed on FB beside.
  * DK-VIOL, TTC, the tactical confusion matrix and goal selection are reported if the reader has them by then (CKPT120 work item W-FAM-STRAT1-TTC-TACCONF). Otherwise each is named absent with its reason and n.
* **STRATEGIC:**
  * route following S_CKPT120 (real − donor, balanced − 0.5, raw; 341 / 27) and the 30/50/80 S_CKPT;
  * route setting at 120 m (far point − EXTRAP, 2,261 / 135) and at 30/50/80 m;
  * detected information: not computable (no ctx head in the base; reason and n printed);
  * derived goals: not in this card (`PREREG_V8F_DGOAL.md`, scoping).
  * Every STRATEGIC number is worded "with the route supplied along the driven path".
* **Beside, no bar:** ORACLE − real on D-* cells, the echo share, T's lon confusion on FB, and the K9b per-tick row.

## 7. Controls (each with the value it must read)

| id | control | must read |
|---|---|---|
| K1 | reader self-pair TF_s0 vs TF_s0 | 0.0, CI [0, 0] on every cell |
| K-LABEL | FB recomputed from the label tables at read time; S_CKPT120 set | 202 / 48 (201 / 47 lon-valid) and 341 / 27, matching section 2.1 |
| K9b | T1 tick 0 vs T0 (logged limit, batch 64) on T's lon/lat logits and S's four checkpoints, every new arm | max \|Δ\| <= 1e-6 |
| K-REACH-a | `--t-far off --nav-strat-t off` | state-dict keys and every output bit-identical to the `aedf8716` H2 build |
| K-REACH-a2 | `--t-far on` at step 0 (zero-init), and `--nav-strat-t on` at step 0 | every output bit-identical to H2 at step 0 |
| K-REACH-b | step-0 gradient on `e_ck_far.weight` / `e_nav_s_t.weight` | non-zero; **exactly 0.0** on `e_ck_far.weight` under `--t-far-input zero`, with `e_ck_far.bias` non-zero (the probe's zero control) |
| T-ISO-UP | gradient of T's loss terms alone on `s_far_out.*`, `head_s.*`, `read_s.*`, `e_nav_s.*` | **exactly 0.0** (`--detach-upper`) |
| T-ISO-O | perturb the far point | O's checkpoint-token input bit-identical (O reads the far point only through T's code); O never reads `nav_s` |
| K-KEYS | `forward_infer` read set | INFERENCE_INPUT_KEYS + NAV_S_KEYS, unchanged; `rc120` never read |
| G-CLOCK-TFAR | far point re-expressed from S's tick frame into T's (planted poses) | equals `transform_ckpt` to 1e-6; an arm that skips the transform FAILS |
| K-RNG | teacher-forcing draw count with `--t-far on` | equal to H2's (one `torch.rand(B)` draw); a second draw FAILS |
| K-REACH-c | mutations: delete the `+ e_ck_far(...)` term; feed RC-A80 in place of the far point; skip the transform; draw a second Bernoulli | each named test RED (G-LIVE, label identity, G-CLOCK-TFAR, K-RNG) |
| K-REACH-d | argv | names every lever with its value; G-DVB refuses `--t-far on` with `--s-far off` or without `--detach-upper` |
| K-REACH-e | post-training zerograd gate (Adam-state method, `ckpt120_reach.py` extended) | every `e_ck_far.*` (TF, R3, TZ: bias at least) and `e_nav_s_t.*` (TN) param has optimizer state, plus CKPT120's `s_far_out.*` / `e_nav_s.*`; else **HEAD-DEAD**, nothing read |
| R3 | rolled far token at train | T-READS **not** separated at s0, and no D-* cell gains vs H2 |
| TZ | token off | no D-* cell gains vs H2; \|TZ − H2\| on D-* printed against F |
| K-STRAT | S's outputs in TF vs H2 | STRATEGIC cells inside F120 unless the adapter moved them (reported as a cost if separated) |

If R3 or TZ passes its must-fail condition, or K1 / K9b / T-ISO-UP / K-KEYS fails, the read is broken, and nothing from this card is quoted until it is fixed.

## 8. Outcomes, committed before any number (each names what runs next)

**Use and gain (TF, both seeds; every effect also > F)**

| outcome | condition | triggers |
|---|---|---|
| **TFAR-GAINS** | T-READS separated positive at both seeds, AND >= 1 D-* cell gains at both seeds | PROVISIONAL until GAIN4: TF_s10 / TF_s11 against H2_s10 / H2_s11. On GAIN4, T-reads-far is the strategic-recipe candidate, TN_s1 runs for the attribution at two seeds, and the DGOAL geometric-goal rung builds on TF |
| **TFAR-READS-NO-GAIN** | T-READS separated at both seeds, no D-* cell gains at both | the ORACLE row decides. ORACLE − real separated positive on a D-* cell at both seeds ⇒ the lever is S's far-point quality: the CKPT120 H1s branch (the far point from its own trunk, 526,339 params), then TF on it. ORACLE − real not separated ⇒ TFc − H2c (T reads an already-trained far point from step 0); if TFc is still flat against H2c, the information is not usable at T's 6 s band, and the next card is an S-level longitudinal output in the 6-30 s band |
| **TFAR-IGNORES** | T-READS not separated at >= 1 seed | zero-training check: \|\|e_ck_far.weight\|\| and the real-vs-zero argmax change. Then TFc − H2c; if still ignored, the appended-token form TFa |
| **HEAD-DEAD** | K-REACH-e fails | build defect: fix, rerun the arm, no read |

**Driving and STRATEGIC keep (24 cells + STRATEGIC cells; TF − H2)**

| outcome | condition | triggers |
|---|---|---|
| **KEEPS** | no cost on any cell at either seed | as the use/gain row |
| **COSTS** | a cell costs at both seeds | TZ decides attribution. If TZ − H2 costs on the same cell, the cost is the new path or lr: TF with `--t-far-lr-mult 1`. If TZ keeps, the information costs (likely exposure to RC-A120 under teacher forcing): TFnt (`--t-far-tf off`). A STRATEGIC cost: TFd (`--s-adapter-detach on`). No adoption |
| **SEEDS-DISAGREE** | a cost at exactly one seed | TF_s10 / TF_s11; a cell is a cost at >= 3 of 4 seeds with one per init (the C0REP noise of 5 / 48 single cells, M) |

**Attribution (beside; one seed unless TFAR-GAINS):**
* TF − TN on D-* separated in TF's favour ⇒ **HIER-ADDS**: S's processing of the route adds beyond the raw text.
* Not separated ⇒ **FLAT-EQUAL**, reported as written: the hierarchy claim for this lever rests on TF alone.

**Written before data (H):** TF reads **TFAR-READS-NO-GAIN** or **TFAR-IGNORES** on D-*. ORACLE − real is positive on D-TAC-2 on FB. The basis is section 2: H2's predicted far point adds probe AUC +0.030 / +0.004 on FB; it sits only 0.53 / 0.57 m closer to the route than EXTRAP; and TAC-LAT has at most 12 windows to gain. If TF reads TFAR-GAINS instead, T has learned a non-linear use of the far point that the probe cannot see, and that is reported as found.

Bars, controls, arms and outcomes do not change after any number is seen. A bar that fails is reported as written, then the triggered arm runs (RULE ZERO).

## 9. Launch-gate checks this card needs (PI rule 2026-09-26)

* **G-HYG:** the config dataclass refuses undeclared attributes: the new fields `t_far`, `t_far_input`, `t_far_variant`, `t_far_tf`, `nav_strat_t` are declared and validated.
* **G-DVB:** every new argv lever equals the built value. `--t-far` → `built_levers()["t_far"]` and `e_ck_far` exists iff on; `--t-far-lr-mult` → the optimiser group `tfar` / `tfar_nd` lr; `--t-far-input` / `--t-far-variant` / `--t-far-tf` → recorded levers; `--nav-strat-t` → `e_nav_s_t` exists iff on. Refusals: `--t-far on` with `--s-far off`; without `--detach-upper`; a variant other than real with the layer off.
* **G-LIVE:** a real-data smoke. Every declared loss term is finite (no new term: T's own terms train the layer). The gradient on the new token path is non-zero (`e_ck_far.weight`, `e_nav_s_t.weight`), with exact-zero controls on `e_ck_far.weight` under `--t-far-input zero` and on every S parameter from T's loss (T-ISO-UP).
* **G-CLOCK:** G-CLOCK-TFAR (S frame → T frame), plus CKPT120's G-CLOCK-NAV-S for `nav_s`.
* **G-EVAL:** the eval loader builds the identical model. `g_box_overfit._build_model` and `refcv7_loader` mirror the new attachment (memory: replays must mirror the trainer build).
* **G-CKPT:** checkpoint round-trip with `e_ck_far.*` / `e_nav_s_t.*`.
* **G-SUITE:** clean-tree tests with 0 regressions on the pod. Watch memory headroom: the container cap is 50 GB (memory note).
* **Post-training:** K-REACH-e on `e_ck_far.*` / `e_nav_s_t.*`, with its planted HEAD-DEAD controls.
* Each check has a deliberate-regression arm that must make the gate FAIL (K-REACH-c).

## 10. Cost and the GPU queue

* **Measured arm wall time.** A 3,000-step v8f-mini arm on the LoRA pod (RTX 4000 Ada), including mint, gate, post and reach, takes **about 15-16 min**. MEASURED from the MM's pod-log timestamps (LOOP_STATE_2026-11-02, 2026-11-17):
  * H1_s0 minted 14:09Z and the chain reached H2_s0's mint at 14:39Z, so 2 arms took 30 min.
  * The fix3 chain minted H2_s0 at 20:03Z and was seen done by 21:22Z (10-min poll), so 5 arms took 79 min or less.
  * The C0REP pair that followed ended at 21:53Z.
  * These times are read from the MM's log, not from the pod's `train_log.jsonl`; treat them as INHERITED.
* **This card (E, on that rate):**
  * arms 1-6: **about 1.6 pod-h**;
  * H2_s10 / H2_s11, if not already run under CKPT120: about 0.5 pod-h;
  * a 200-step pilot of TF and TN: about 5 min;
  * conditional arms: about 16 min each, so the GAIN4 pair is about 0.5 pod-h and the TFc / H2c branch (4 arms) about 1.1 pod-h.
* **Stop rule:** a projected chain above 4 h stops and goes back to the MM.
* **G-SUITE** on pod CPU: the S1d suite took 23 min (M, INHERITED from the CKPT120 kit record), and the CKPT120 kit expected 45-60 min.
* **Reads (dev box):** the CKPT120 read ran in about 1.8 h for 7 arms (M: arm pulls 22:23-22:24, `RESULT_CKPT120.md` at 00:11). This card adds the ORACLE variant and the FB table, so about 2.5 h for 8 arms (E). It runs under the dev-box GPU lock (memory).
* **Build:** about half a Sonnet agent-day (E), `BUILD_GAPS_TFAR.md`.
* **Disk:** about 87 MB per banked checkpoint (M, H2_s0 `ckpt_last.pt`), so under 1 GB for the card.
* **Queue:** the pod runs DSCALE block 3 (to about 05:00Z, E, LOOP_STATE), then NAVRB2. TF follows NAVRB2.

## 11. Known risks

* A zero-initialised layer inside 3,000 steps may under-train. The lr x10 is the hedge. S's far head and `e_nav_s` are also trained from zero in the same 3,000 steps, so the far token is near its nominal value early on; teacher forcing gives an informative token on 20 % of samples from step 0, and TFc / H2c is the branch (decision 8).
* Teacher forcing feeds RC-A120 (exact) 20 % of the time, while S's point is about 15 m off at inference. T may learn to trust a point that is wrong at inference. TFnt is the branch, and the ORACLE row shows the gap.
* FB is defined from GT geometry. At inference T does not know which windows are FB, so the cell scores T where the information exists, by design. The all-window table guards against a cost elsewhere.
* The DAgger banks were rolled by M2b and are not on-policy for TF (as CKPT120). Base and arms share them, so they do not confound the contrasts.
* TN lifts the T-side `nav_s` guard for one arm. Its reach and key guards must make that the only exception.
* If the training rig is not deterministic at a fixed seed on the pod (M, C0REP), TF_rep sets the floor, and a single-seed effect is never quoted as a lever effect.

## 12. Design artifacts (this folder, single copies until the MM lands them)

| file | what | sha256 |
|---|---|---|
| `measure_tfar_labels.py` | section 2.1 (pins the grid truth `da9529a1…` and RC-A120 table `0f122570…`) | `92ead9d7…` |
| `TFAR_LABELS.json` | its output (join_eval141 sha256 `f0ea722f…`) | `e970e3fe…` |
| `probe_h2_far_info.py` | section 2.2 | `832590aa…` |
| `H2_FAR_PROBE.json` | its output (H2 T0 inputs: `b7dd1428…`, `4194fae5…`, `2e6487cc…`, `a991390a…`) | `48b419b4…` |

Re-run (tanitad venv, about 5 s each): `python -I measure_tfar_labels.py --out TFAR_LABELS.json`; `python -I probe_h2_far_info.py --out H2_FAR_PROBE.json`.

## Log
* 2026-11-18, pre-registration designer: written and frozen for the MM (sha256 beside). Not built, not run. MM rulings, if any, go in `MM_RULINGS_TFAR.md` beside and bind like this text.
