# AUDIT — do the v7.2 TACTICAL labels reach the v7F (v6-staged) training loss?

PI requirement R3 (BINDING, 2026-09-27): *"All our tactical labels must be used to train the tactical
layer to estimate and choose the right tactical behaviors and goals which MUST condition the operative
planning."*

Scope: trainer `stack/scripts/train_v6_staged.py`, model `stack/tanitad/models/v6.py`, label consumer
`stack/tanitad/data/v7_labels.py`, all read at the origin tip **c36b6ddd** (snapshot
`C:/Users/Admin/tipsnap/c36b6ddd/`, CRLF copy of the LF git blobs — content-identical, verified by byte
comparison after CRLF->LF normalisation). Every `file:line` below is a line of that snapshot.

Evidence class: **MEASURED** (read from source + executed census). Census numbers are from the canonical
v7.2 TRAIN blob, md5 `0ff902130ce76886b8a925eceed9e3a5` (= `intrain_eval.V72["train"]["md5"]`, verified),
4,572 records, read-only from `C:/Users/Admin/tanitad-wt/_s2build/release/v72/`; raw output:
`raw/audit_census_train_blob.txt`.

## 0. Verdict (one paragraph)

**At c36b6ddd, NO tactical label family reaches any loss of the v6-staged trainer.** The lat/lon action
ids are joined into the batch only on the S-S/S-J path and are read by no loss; the 22-token goal SET
(including every traffic-light colour), every per-goal argument (SPEED_BAND `v_lo_ms`/`v_hi_ms` ...), the
action arguments and the goal anchor are never even projected into a batch key. Worse, in **S-T — the
only non-joint stage that trains `layer_tac` — the label file is never loaded at all**
(`train_v6_staged.py:6662` loads `--s2-labels` only when `w_s2_goal` is in force, and
`V6LossWeights.for_stage("S-T")` zeroes `w_s2_goal`, `:368-379`; preflight `:10790-10799` refuses
`--w-s2-goal` in S-T and `:10808-10814` refuses `--s2-labels` without it). The existing D-TACGOAL-1
closure (`V7Label.tac_goals`, `v7_labels.tactical_goal_targets`, `tanitad.refs.tac_goal_head`) is wired
ONLY into the refcv3 trainer (`refc_v3_train.py:4931-4968`, `:8216-8240`); the v6 trainer contains zero
occurrences of `tactical_goal_targets`, `TacGoalEmitter`, `tac_goals`, `tac_goal_meta`, `tac_anchor`,
`oracle_max_speed` (two probes: `grep`, and a substring count over the file — both 0).

## 1. The label record (layer 2) — what the release carries

`s2-geom-v7` record, per clip (one record per clip, anchored at `t0_s`; tactical band `bands.tactical_s`
= [2, 6] s). Extraction of the traffic-light COLOUR: `tanitad/data/alpamayo_semantics.py:42-44` (red /
yellow|amber / green CoT regexes) -> `:231-236` (`TRAFFIC_LIGHT_REACT` + `state`) -> frozen coloured tokens
`vocab_v7.py:108-111`; perception-only, never geometry-emittable (`vocab_v7.py:258-268`).

MEASURED on the train blob (4,572 records):

| family | field in record | census |
|---|---|---|
| lateral action | `a_tac.lat` (8-way `TACTICAL_LAT_ACTIONS_V7`, `vocab_v7.py:320-323`) | LANE_KEEP 2958, NUDGE_R 592, NUDGE_L 488, TURN_L 275, TURN_R 259; LANE_CHANGE_L/R + ABORT_LC absent |
| longitudinal action | `a_tac.lon` (8-way `TACTICAL_LON_ACTIONS_V7`, `vocab_v7.py:328-331`) | CRUISE 1243, ACCELERATE 998, ADAPT_SPEED_FOR_CURVE 995, BRAKE_TO 905, FOLLOW 165, CREEP 135, HOLD 131; YIELD_MERGE absent |
| action args | `a_tac.lat_args` {`within_m`, `lat_peak_m`}; `a_tac.lon_args` {`v_target_ms`, `within_m`} | on every record (eval 147/147) |
| goal SET | `g_tac.goals.<TOKEN>` over the 22 `TACTICAL_GOAL_TOKENS_V7` (`vocab_v7.py:95-113`) | 2-7 tokens/record, mean 2.751 |
| traffic light, colour | `g_tac.goals.TRAFFIC_LIGHT_REACT_{RED,GREEN,YELLOW}` / colourless `TRAFFIC_LIGHT_REACT` | RED 376, GREEN 363, YELLOW 22, colourless 18, any 779 |
| SPEED_BAND | `g_tac.goals.SPEED_BAND` {`v_lo_ms`, `v_hi_ms`, `band_s`, `held`} | present on **4,572/4,572** (a constant token); v_lo 0.0-37.31, v_hi 0.0-37.8 m/s |
| other goals | FOLLOW_LANE 3629, CORRIDOR_OFFSET 860, YIELD 609, GAP_TARGET 368, REACT_ON_ONCOMING 333, STOP_POINT 327, TURN_L 275, TURN_R 259, EVADE_IN_CORRIDOR 240, TAKE_EXIT_R 128, MERGE 79, LANE_CHANGE_L 23, TAKE_EXIT_L 21, YIELD_FOR_TURN_L 21, OVERTAKE_VEHICLE 20, YIELD_FOR_TURN_R 20, LANE_CHANGE_R 15 | all 22 tokens have >= 15 positives |
| per-goal args | e.g. STOP_POINT {`within_m`, `hold_for_s`}; TURN_* {`within_m`, `by_time_s`, `radius_m`, `dyaw_deg`}; GAP_TARGET {`time_gap_s`, `agent_slot`}; CORRIDOR_OFFSET {`side`}; TL {`state`, `object_kind`} | (eval-blob key census; train not enumerated) |
| goal anchor | `g_tac.anchor` {`goal_x_m`, `goal_y_m`, `t_reach_s`, `band_s`} | every record |
| max-speed INPUT | `speed_max_input` (oracle, ego-future) | 0/4,572 on v7.2 (`v7_labels.py:641-642`) |

## 2. Layer 3 — loader -> batch -> heads -> loss, per family (the hops)

`L` = `stack/tanitad/data/v7_labels.py`, `T` = `stack/scripts/train_v6_staged.py`,
`M` = `stack/tanitad/models/v6.py`.

| family | loader (`L`) | trainer batch (`T`) | model head (`M`) | loss in `T` | REACHES A LOSS? |
|---|---|---|---|---|---|
| lat action | `V7Label.tac_lat` `L:382`; `tactical_class_ids` `L:701-710` (band rule `L:678-698`) | `V72WindowSupervision.batch` -> `tac_lat_id`,`tac_valid` `T:2594-2611`; merged only via `s2_sup` `T:7410-7414`; `s2_sup` built only `if a.s2_labels and w_stage.w_s2_goal` `T:6662` | `act_head_lat` `M:5155`, logits `M:5939` (8-way, v7.0 vocab via `tac_vocab_version` default `M:3890`, `M:5084-5085`) | **none** — `V72_TACTICAL_BATCH_KEYS` comment `T:2410-2415` "No loss term reads them yet"; report `T:2589-2591` "tactical_consumer: NONE"; `a_lat` logits read only by the `t5_plan_switch_rate` METRIC `T:4693-4695` | **NO** (and not even loaded in S-T) |
| lon action | `V7Label.tac_lon` `L:382` | `tac_lon_id` `T:2598-2610` (same S-S/S-J-only join) | `act_head_lon` `M:5156`, `M:5940` | none | **NO** |
| action args | `V7Label.audit["a_tac_args"]` `L:399-400` — audit is "NEVER a training input" `L:206` | none | `a_lat/a_lon["args"]` emitted `M:2669`, consumed only by `vocab_a_*.encode` `M:5958-5960` | none | **NO** |
| goal SET (22) incl. **TL colours** | `V7Label.tac_goals` `L:386` (D-TACGOAL-1); targets `tactical_goal_targets` `L:917-1047`; emitter `TacGoalEmitter` `L:1149-1238` | **no key** — 0 occurrences of `tactical_goal_targets`/`TacGoalEmitter`/`tac_goals` in `T` | `goal_head_tac` = `GoalHead(vocab_tac, d_tac, d_cond)` `M:5153-5154`, 22-wide under v7.0 (`M:5078-5079`, `M:376-379`); `g_tac` `M:5938` | none — `out["g_tac"]` is read by no term of `v6_loss_step` `T:4160-4726` (terms: o1,o14,o5,o11,o13,o2,o3,o6,seam,t1,t2,s1,s1_multi,plan,select,anchor,t5,s2) | **NO** — this is `D-TLIGHT-1` for v7F |
| SPEED_BAND args (v_lo, v_hi) | `V7Label.tac_goal_meta["SPEED_BAND"]` `L:387-389` | none | `g_tac["args"]` (8 physical-unit slots, `M:415-418`, `M:2609`) | none | **NO** |
| other per-goal args | `tac_goal_meta` `L:387-389` | none | same `g_tac["args"]` / cat channel | none | **NO** |
| goal anchor | `V7Label.tac_anchor` `L:385` | none (0 occurrences in `T`) | `AnchorGoalHead` (`--anchor-goal`, planner group) is supervised by `plan_target[:, -1]` (ego future at the plan horizon), NOT the record's anchor `T:4639-4658`; refused without a plan-horizon anchor table `T:10642-10651` ("NO ADMISSIBLE TABLE EXISTS TODAY") | only indirectly: ego future via `lambda_plan`/`w_anchor` | **NO** (record field); the same geometry reaches the plan loss via `plan_target` |
| strategic goal/action | `V7Label.str_goal/str_action` `L:383-384` | rows `T:2785-2829` -> `g_str_id`/`a_str_id` | `goal_head_str`/`act_head_str` | `s2_goal_loss` `T:2310-2386` under `--w-s2-goal`, S-S/S-J only | yes — strategic (out of R3 scope; strategic layer OFF per the PI) |
| nav command | `_oracle["nav_command"]` `L:408`, `NavEmitter` `L:1244-1338` | `nav_token`/`nav_args` `T:7314-7317` | `NavConditioner` INPUT to all three layers | n/a (an INPUT, not a target) | n/a |
| audit fields (`violations`, `serves_goals`, `truncated`, `alpamayo.*`, `scene.*`, `cot_*`, `strata`) | `audit` `L:393-407` or not read | none | none | none | by design (audit-only) |

## 3. What the existing heads/flags are (and are not)

| flag (`T` line) | what it builds (`M`) | supervised by a LABEL? |
|---|---|---|
| `--goal-factored` `T:9420` | `goal_head_tac_lat`/`_lon` `M:5230-5251`; `e_g_tac` = mean of the pair `M:5954-5955`. ⚠️ the LAT vocab is the **v6** partition `TACTICAL_GOAL_TOKENS_LAT` (ANCHOR_GOAL, CORRIDOR_OFFSET, EVADE_IN_CORRIDOR, LAT_UNCONSTRAINED — `M:446-448`), NOT versioned, so no v7 label can populate 2 of its 4 classes; the LON vocab under v7.0 is `tactical_lon_goals("v7.0")` = ALL 22 v7 tokens (`M:492-493`: every v7 token is a key of `GOAL_ADMISSIBLE_LON`) — i.e. the "factored" pair is not a LAT/LON factoring of the v7 goal vocabulary | **no** |
| `--goal-multilabel` `T:9425` | 0 params: `g_tac["gates"] = sigmoid(logits)` `M:2636-2652`, `M:2671-2672`; `_encode_goal` then conditions on the gates instead of the softmax `M:5650` | **no** |
| `--goal-cat-args` `T:9430` | typed categorical arg channel `M:5252-5261` (anchor_id, lat_bin, agent_slot, reason, state) | **no** |
| `--tac-goal-cond` `T:9434` | the **strategic -> tactical** port `cond_tac_dyn` `M:5286-5291`, `M:5985-5986` (NOT tactical -> operative) | n/a |
| `--w-s2-goal` + `--s2-labels` `T:9917`, `T:9943` | CE on `g_str`/`a_str` | yes — strategic only; S-S/S-J only |

The ONLY gradients `goal_head_tac` / `act_head_lat` / `act_head_lon` receive today are latent/plan
objectives flowing back through the conditioning: `seam_op` (`T:4511-4521`, via `e_g_tac` ->
`predictor_op` intent), `lambda_plan`/`w_select` (via `e_g_tac` -> `emit`), and `t1_latent`
(`T:4524-4531`, via `e_a_tac` -> `predictor_tac`). None of them is a label.

## 4. Does the tactical layer's output condition the OPERATIVE planning? (context for R3/R4)

* tactical **GOAL** embedding `e_g_tac` -> operative predictor `intent` `M:5996-5998` and the seam copy
  `M:6018-6021`; -> the 6 s emission/fan `emit(z_plan, e_g_tac, v0)` `M:6035`, `M:5775-5776`;
  -> selector/anchor head `M:5796-5820`. **YES.** With `--goal-multilabel` it is built from the sigmoid
  gates (the BCE-supervisable quantity), otherwise from `softmax(logits)`.
* tactical **BEHAVIOURS** `a_lat`/`a_lon` -> `e_a_tac` `M:5958-5960` -> **`predictor_tac` only**
  `M:5984-5989`. They do NOT reach `predictor_op` or `emit`. ⚠️ So the PI's "behaviors ... MUST condition
  the operative planning" is **structurally unmet** at c36b6ddd for the behaviour half — an architecture
  change (R4's territory), not a loss change. Recorded as an open item in `RESULT.md`; NOT changed here.

## 5. Consequences for any label loss (MEASURED on the train blob, `raw/audit_census_train_blob.txt`)

Goal-set supervision per negatives policy (`v7_labels.goal_supervision_census` +
`tac_goal_head.mask_report`, the refcv3 recipe):

* `measured` (the module's default): **17/22 classes trainable**. Masked (positives, ZERO supervised
  negatives — the logit could only be pushed to 1): YIELD, SPEED_BAND, CORRIDOR_OFFSET, GAP_TARGET,
  REACT_ON_ONCOMING. The traffic-light colours ARE trainable (RED 376 pos / 403 neg, GREEN 363 / 416,
  YELLOW 22 / 757, colourless 18 / 761 — the negatives are ENTAILED by the other colours through
  `TACTICAL_GOAL_EXCLUSIVE`, `vocab_v7.py:157-162`; 3,793 no-light clips are IGNORED, not negatives).
* `all` / `cot-absence-negative` (PI ruling 2026-09-16): **21/22** trainable; only SPEED_BAND stays
  masked, because it is present on 4,572/4,572 records — its information is ONLY in its args
  (`v_lo_ms`, `v_hi_ms`). ⇒ SPEED_BAND can reach a loss only through an ARG term.
* ⚠️ The only banked cot-absence sidecar,
  `TanitAD Research Lab/Data Engineering/Research/2026-09-16-flywheel-negatives/cot_absence_negative_v8.0_train.json.gz`
  (file md5 `fc2ee151620d2c7dbc2416f7f17481f1`), declares `source_blob_md5 = fa89ea55dfce68403eb30300e57852ab`
  — NOT the canonical v7.2 train pin `0ff902130ce76886b8a925eceed9e3a5` that `T:2657-2672` enforces.
  `load_cot_negative_sidecar` refuses exactly that mismatch (`L:480-486`), so the PI-ruled
  absence-as-negative policy is **not launchable on the canonical v7.2 blob today** without a sidecar
  rebuilt over it (or a switch to the v8.0 blob). Which blob `fa89ea55` is: UNVERIFIED (the file name
  says v8.0).

Band coverage: a record describes only windows whose NOW is within +-(hi-lo)/2 = +-2.0 s of `t0_s`
(`L:678-698`), so only a fraction of windows per clip is supervisable (MEASURED on the eval join in
`raw/smoke_real_eval_labels.log`).

## 6. Hop-by-hop summary of what must change (implemented in `code/fix/`, see `RESULT.md`)

1. `T:6662` — load the v7.2 labels in S-T when the new tactical term is in force (not only for S2).
2. `V72WindowSupervision` `T:2528-2611` — project the goal SET `(y, w)` (via
   `v7_labels.tactical_goal_targets`, the ONE rule, with the chosen negatives policy) and the SPEED_BAND
   args onto the SAME windows/band as `tac_lat_id`/`tac_lon_id`.
3. `v6_loss_step` `T:4160-4726` — a new term reading `out["a_lat"]`, `out["a_lon"]`, `out["g_tac"]`
   (logits + args) against those keys; heads only (planner cut `M:5927-5930`), in force where
   `layer_tac` trains (S-T/S-J).
4. Preflight `T:10298-10874` — the `--s2-labels`-without-`--w-s2-goal` refusal `T:10808` must not fire
   when the tactical term is the consumer; refuse the term where it cannot train or cannot condition
   the planner.
