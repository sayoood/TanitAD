# RULETRAIN_DEFS: what `--rule-relabel on` does, and every loss term whose target comes from the logged future (card PREREG_V8F_PROOF_ADD32)

Tree: `C:/Users/Admin/v8proof_c/ruletrain/tree_patch` (LEADSUP tree + this lever). Line numbers are of that tree. Tier of any later read: T1-kin; evidence class of this file: MEASURED by reading source, plus the checks in `checks/`.

## The decision, in one place

* The loss reads ONE array for the executed plan's target: `gt, gok = batch["gt"], batch["gt_ok"]` (`tanitad/models/v8f_mini.py:1319`). `gt` is `[B, 30, 2]` knots at 0.2 s in the ego frame of the sample's current frame, origin not in the array (`gt_future`, `v8f_mini_data.py:458`: `fk.gt_waypoints(pose, t0, 1..30)`), the same frame and layout as `out["traj30"]` (the loss takes their difference) and as the plan `distance_rule` expects (its docstring: N knots at 0.2 s, ego frame of the tick, origin not in the array; the rollout wrapper builds its logged path from the same `st.d["gt"][t0]`). So no frame change is needed: the target and the rule's input are the same object.
* The trained output is `traj30 = subsample_02(traj60)` from `_forward` (`v8f_mini.py:1217`). `traj60` has NO term of its own in `loss` (only `traj30` enters ade, fde, acc, launch_along, the penalties). The HACC post-map (`_hacc_post_map`) is inference only ("NEVER in `_forward`: training and the loss read O's raw plan", `v8f_mini.py:1245`). So the card's "same output before its post-map" has no separate term.
* The relabel replaces `batch["gt"]` by `dist_rule.distance_rule(gt, v0, present, gap, gap, v_L, m=10, floor_lifts=False)`. The model code is unchanged (`v8f_mini.py`, `hjepa.py`, `v8f_beta.py`, `v8f_t1_eval.py` byte for byte). The relabel is a trainer stage: `rule_relabel.RuleRelabel.stage(batch)` is called in the loop on the COLLATED, LOGGED batch before the ego stages, and the loop line `b["gt"] = rr_gt` attaches the result after them (the pattern of `launch_stage` / `lead_stage`).
* Inputs of the rule, and where they come from (the same quantities the rollout wrapper `distrule/v8f_t1_eval_rule.py:150-216` uses):
  * `v0 = d["p10f"][t][0]` read RAW from the clip store (wrapper: `batch["p10f"][:, -1, 0]`; the batch tensor equals it where finite); on a logged state this is the logged ego speed channel of the current frame.
  * `v_logged = d["v"][t]` read RAW from the clip store = cache speed row of the current frame (wrapper: `st.logged_v[j]`, the same `base["v"]` array). Non-finite speed rule: a row whose `v_logged` or `v0` is not finite is NOT relabelled and is not counted (it keeps its target exactly), as the wrapper's `src_bad = present and not finite(g_src, c_src, v_l)` makes that tick inactive; the batch tensors cannot show it (`nan_to_num` turns NaN into 0.0), hence the raw read.
  * `gap`, `closing`: `D.lead_sup_label_arrays(J)` rows `off[ci] + t` (`J["lead_valid"]`, `J["lead_gap"]`: presence 1, finite gap in 0..60 m; closing = least-squares slope over the trailing run of at most 5 labelled frames, step <= 3 m, at least 3 frames). The wrapper's `LabelSource.read` uses the same rule (DISTRULE_DEFS, label source).
  * `v_L = max(0, v_logged + closing)`; `g0 = gap` (on a logged state a_l = a_m = 0 so the wrapper's `g0 = g_src + a_l - a_m` is `g_src`); `present` = True for an eligible row.
  * `m = 10`: it only feeds the `n_sat_exec` log of `distance_rule` and does not change a plan.
  * Constants `D_M 3.0, TAU_S 1.2, B_MAX 6.0, DT 0.2` are the module constants of `dist_rule.py` (copy byte-identical to `distrule/dist_rule.py`, sha256 `40b5dd9173aeaf8c7d7a7d4bfbb52484a8f30b9a8480d3356f8fa0aee65746dc`; the trainer records it in `info.rule_relabel.dist_rule_sha256`). `floor_lifts=False` is the card's cap-only form.
* Eligible row: label presence 1, finite gap, finite closing label, `gt_ok` True (a row without a GT future has an all-zero target that the loss masks out), `frame_ok[:, -1]` True, finite raw `v0` and `v_logged`. Every other row keeps its target bit for bit.
* The labels come from the join the trainer already loads (`--join-train`), through `lead_sup_label_arrays`; no lead head is built (`--lead-sup off` stays), no `--lead-train` file is read, the dataset carries no new key.

## Table: loss terms whose target comes from the logged future of the sample

| term | file:line | target key | decision | reason |
|---|---|---|---|---|
| `ade`, `fde` (`traj_terms`) | v8f_mini.py:1320 (def 551) | `batch["gt"]` | RELABELLED | the card's lever: the executed output's target. They read the relabelled `gt`. |
| `acc` (`accel_term`) | v8f_mini.py:1321 (def 565) | `batch["gt"]` | RELABELLED by sharing the key | logged always; in the total only when `w_acc > 0` (arm G: 0.0). If an arm sets `w_acc > 0` the term compares the plan's second difference with the relabelled target's: consistent. |
| `launch_along` | v8f_mini.py:1325 (def 590) | `batch["gt"]` | RELABELLED by sharing the key (only if `launch_intent_w > 0`; arm G: 0) | the launch MASK is built from the logged gt BEFORE the stage (`launch_stage`), the along-track error is read against the relabelled gt. |
| `pen_*` hinges vs GT constraints (`consistency_penalties(traj30, gt_constraints_t(lab, lim, lead_gap_h, s_h))`) | v8f_mini.py:1358-1359 (def 664, 746) | `lab` (cons pack of the logged future), `lead_gap_h` (lead gap along the future), `s_h` (the human's arc) | NOT relabelled | card: leaves the other terms unchanged. SPEED / STOP / TURN targets are label values. The LEAD hinge (`lead_gap_h + s_h - s_plan`, time gap against the driver's minimum time gap, weight 0.5 x 0.5) keeps the HUMAN arc `s_h`, not the relabelled arc. OPEN QUESTION (see hand-back): it pushes the plan toward larger time gaps, the same direction as the rule, so it does not oppose the relabel. |
| `pens_*` self-consistency hinges | v8f_mini.py:1360-1361 | the model's own detached predictions | NOT relabelled | not from the logged future. |
| `ckpt` (S's route checkpoints) | v8f_mini.py:1301 | `lab_s["rc"]` | NOT relabelled | card: world-model / hierarchy terms unchanged; a route checkpoint is a path point of the logged driver. |
| `goal`, `lat`, `lon`, `cons` (T, the 6 s tactical target) | v8f_mini.py:1307-1317 | `lab_t` goal_y, lat_al, lon_al, cons | NOT relabelled | card: the 6 s tactical target unchanged. |
| `wm` (level-1..3 predictors, SIGReg, futtok if on) | v8f_mini.py:1372 | the cached future latents and the logged `act` | NOT relabelled | card: world-model terms unchanged; they predict latents, not the plan. |
| `lead_sup` (LEADSUP head) | v8f_mini.py:1376 | labels | not present in arm G | `--lead-sup off`; card: the lead head unchanged. |
| `level`, `lead`, `stop_aux` | v8f_mini.py:1328-1356 | logged speed, lead file, stop labels | not present in arm G | off. |

Not a loss term but read from the logged future: the `launch` mask (above) and the DAgger / recover / ego-perturb stages (refused with the flag, below).

## Gate and refusals (G-DVB, G-LIVE)

* `FLAG_CHECKS["rule_relabel"] = "data:rule_relabel"`; `bad_rule_relabel(a, info)` reads the BUILT stage record `info["rule_relabel"]` (mode, label array sha256 and counts, rule file sha256, constants). Off with a stage built, on with none built, label counts 0, and the combinations with `--ego-perturb-p`, `--dagger-p`, `--recover-p` (they rewrite the logged state or target the rule is defined on) or `--encoder lora` are refused.
* G-LIVE: the probe batch is relabelled by the loop's own call; `info.rule_relabel_n` is its count and the launch is REFUSED when it is 0. `rule_relabel_probe`: the loss and the gradient of the total differ between the logged and the relabelled target when the rule changed at least one target (`rule_relabel_changed > 0`); the same batch twice and an all-False mask read exactly 0.0.
* Loop guard: after `to_dev` the batch's `gt` must equal the relabelled target, else exit `ZZRULE-NOT-APPLIEDZZ` (a deleted attach line is caught at step 0).
* Train log (when on): `rule_relabel_n` (relabelled rows since the last log line), `rule_relabel_changed` (of those, targets the rule moved), `rule_relabel_rows`. `summary.json` and the checkpoint carry the run totals (`rule_relabel_totals`, a top-level key of the checkpoint blob only when on; the model `cfg` gets NO new key).

## Selfcheck strings for the chain gate

`levers.rule_relabel=on` and `info.rule_relabel_n=<N>` (N read from the first real-data selfcheck, not predeclared; synthetic probe: 7 to 50). A flag-off arm has no `rule_relabel` key in `levers` / `info` / the log.

## Limits

* How often the cache `v` or `p10f` is non-finite at a labelled frame is not measured here (the rule above excludes such rows).
* The share of real logged futures that the cap-only rule actually moves is not measured here (no real cache on this box); on the synthetic world 13 to 16 of 64 rows moved. The G-LIVE probe prints `rule_relabel_changed` on the pod.
