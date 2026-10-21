# PREREG R8-ORACLE: is refcv8's tactical headroom in its CONTINUOUS constraints or in its CLASS decisions, on refcv8's OWN planner?

**Written 2026-10-21, clock read `2026-10-21T20:09:26+0200` (dev box `date +%FT%T%z`, Berlin) BEFORE any forced forward of refcv8 on any eval window, CPU smoke included.** The sha256 of this file is written to `PREREG_R8_ORACLE.sha256` immediately after it is saved; `code/r8oracle_roll.py` and `code/r8oracle_analyze.py` refuse to run unless this file still hashes to that value.

Author: EvalFlyWheel agent (Opus). Owner of the question: Master Mind brief 2026-10-21. Evidence classes are stated per line below.

## 0. Why

* INHERITED (another model, not re-run): the v8F-mini T0 oracle (`RESULT_V8F_TAC_A1.md`) put **0.88-0.90** of its headroom in the constraints and **0.07-0.12** in the decisions. That is a different architecture.
* MEASURED today (`C:/Users/Admin/refcv8_pf/tacconstr/RESULT.md`, `TACCONSTR_R8*.json`, EMA-48,852, per-frame nav, 4,754 windows): refcv8's planner reads the 4 probe constraints per hypothesis (`out_r8_cons_lat/_lon` [B,8,2]: psi_term, t_onset | prog6, v_end), never the 26-dim v9 vectors; the trained selector weights `gamma_head +0.4562`, `gamma_prog +0.1219`, `beta_lat -0.0430`, `beta_lon -0.0210`; the speed/lonspeed v9 heads do not beat a (v0, a0) ridge (head - ridge -0.0075 [-0.0200, +0.0059]); v_end residual 1.73 m/s. **No refcv8-planner oracle exists**: the headroom in metres has never been measured on refcv8 itself.

## 1. The seam, read from source (launch tree `C:/lgs/Lfb47a58376`; blob equality with commit fb47a58 VERIFIED for `refc_v3.py` 68ff839c, `refcv8_conditioning.py` 316b467b, `refcv8_train.py` a95745a1 by `git rev-parse fb47a58:<p>` == `git hash-object <p>`; `refc.py` byte-identical by sha256 bd8f1c42... == `git show fb47a58:<p>`, size 335,805 == blob size)

1. `RefCV3Model.set_r8_force(lat3, lon6, cons, min_alloc)` (`refc_v3.py:2081-2092`) stores a ONE-SHOT dict `{lat3, lon6, cons, min}`; `min` defaults to `cfg.refcv8.n_alloc` (= 32 in this checkpoint's config). Refused in training. `lat3` / `lon6` are SCALARS (one class for every row of the batch); `cons` is [B, 4] normalised (psi_term/pi, t_onset/6, log1p(prog6)/log1p(120), v_end/30). The battery rolls ONE window per forward (rows = that window's nav conditions), so a per-window force is expressible without editing the seam.
2. Where it acts: inside the scene hook, AFTER the tactical decoder `tac_decoder_v6` has produced `lat_logits`, `lon_logits`, `cons_lat`, `cons_lon` and AFTER the ATTACHED raw outputs were written to the loss cache (`refc_v3.py:1981-2000`), `out, r8_force = self._r8_apply_force(out, b, dev)` (`:2001`) runs. `_r8_apply_force` (`:2094-2129`):
   * `lat3` given -> `lat_logits` replaced by -20 everywhere and +20 at `LAT3_V7_IDS[lat3]` (a ONE-HOT posterior, not a soft one); `lon6` likewise with `LON6_V7_IDS` (`refcv8_conditioning.py:61,64`).
   * the posterior argmax hypothesis (hl, ho) is recomputed from the (possibly forced) logits; `cons` (NaN entry = keep the head's own value) is written into the constraint tables `cons_lat[:, LAT3_V7_IDS[hl]]`, `cons_lon[:, LON6_V7_IDS[ho]]`, i.e. into the rows of the **argmax hypothesis**, not of a GT hypothesis.
   * class-only force (`cons` None): returns no force dict; the one-hot posterior alone drives allocation and the factorised score.
   * `cons` given: also returns a force dict `{hyp = joint_id(hl, ho), cons, tf = True, min}`, which `:2016-2018` puts in the planner feed as `force_hyp / force_cons / force_tf / force_min`.
3. What the forced tensors reach (all DOWNSTREAM of the decoder; the trunk, BEV, nav and speed inputs are untouched):
   * `hypothesis_posteriors` -> `r8_feed.logp_lat3 / logp_lon6` (`:2002-2003`) -> ALLOCATION `r8c.allocate(joint, n_alloc=32, top_k=4, min_each=2, force=force_hyp, force_min)` (`refc.py:2510-2514`, `refcv8_conditioning.py:395-431`) and the FACTORISED SELECTION score `beta_lat log p + beta_lon log p` (`refc.py:3834`) and phi;
   * `r8_feed.cons_lat8 / cons_lon8` (`:2004-2005`) -> per-candidate constraints `candidate_constraints` (`refc.py:2503`, `:2517`) -> the selection's `gamma_prog / gamma_head` satisfaction terms (`refc.py:3834`) and phi (CONDITIONING of the allocated candidates' decoder); `force_cons` overrides the phi constraints of candidates of the forced joint hypothesis (`refc.py:2520-2523`);
   * `v6tac.planner_feeds(out)` (`refc_v3.py:2020`) -> `lat_prior / lon_prior / behaviour_term` (the tac8 prior of the refcv6 path) read the forced logits too.
   * So **a class force changes allocation, conditioning (phi) and selection; a constraint force changes conditioning (phi) and selection, and changes allocation ONLY through `force_min`.**
4. ⛔ The default `min_alloc` = `n_alloc` = 32 hands ALL 32 allocated candidates to the argmax hypothesis whenever `cons` is forced: an allocation collapse that is NOT a constraint intervention. **Every constraint force in this study passes `min_alloc = cfg.refcv8.alloc_min_each` (= 2).** Source-literal consequence (`allocate`, `refcv8_conditioning.py:409-416`): the forced hypothesis is the argmax joint, which is already first in the top-k order, so `sel` and every floor are unchanged and the allocation is identical to the unforced one. This is a CLAIM about the code that control C3 below must confirm bit-exactly, or the constraint arms are void.
5. The eval draws (allocated candidates' source-anchor choice, sampler eps) come from a fresh generator seeded from the device's global RNG STATE, read not consumed (`refcv8_conditioning.py:173-189`); a forked / re-seeded global RNG therefore reproduces a forward bit for bit. Every forced arm of a window runs from the SAME pre-call global RNG state as that window's OWN call (common random numbers), inside `torch.random.fork_rng`, so the battery's own sampler stream is left exactly where OWN left it.

## 2. Arms (all T0)

Checkpoint EMA-48,852 (`E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt`, md5 b4364029b68f29a33eeb4aebbbbd002b), config `E:/refcv8_eval_kit/ckpt/config.json` (md5 4ce2c77230dc4e802e78b39ce05af916), per-frame nav feed (`perframe_full`, overlay sha256 98293dd7...), standalone pose clock (`R8_NO_POSE_SYNC=1`), the SHIP-R8 battery's windows (`windows_s{seed}.json` of `C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf/`: 4,754 windows / 139 episodes), inference seeds **0 and 1**, the battery's `--os-only` roll (one model call per window, the `os` arm + model-free controls), dev-box CUDA.

GT, per window, derived from the window's own `item` (`pose_last`, `future_poses_ext`, `future_valid_ext`) with the TRAINER's literal code (`refc_v3_train.py:5419-5421` -> `refcv8_train.r8_before_forward`, `refcv8_train.py:544-551`): `traj_tgt = refb_labels.waypoint_targets(pose_last, fut, horizons)`, `slot_valid = fut_valid[:, h-1]`, `(gl, go) = r8c.tag_paths(traj_tgt, v0, slot_t)`, set to -1 where `slot_valid[:, -1]` is False; `ct = r8c.constraint_targets(...)`, `(lat_t, lat_v, lon_t, lon_v) = r8c.normalise_constraints(ct)`. A GT constraint entry is FORCED only where the trainer's LOSS mask is on (`lat_v & gl >= 0`, `lon_v & go >= 0`); elsewhere it is NaN (= the head keeps its own value). A window with `gl < 0` or `go < 0` gets no class force for that axis (counted per arm).

| arm | force | role |
|---|---|---|
| **OWN** | none: the battery's own call, its own dump | reference |
| **GT_CONS** | `set_r8_force(cons = GT targets (NaN where masked), min_alloc = 2)` -> the GT constraint values go to the planner's OWN argmax hypothesis | constraint oracle |
| **GT_HYP** | `set_r8_force(lat3 = gl, lon6 = go)` (class only; one-hot) | decision oracle |
| **GT_BOTH** | `set_r8_force(lat3 = gl, lon6 = go, cons = GT, min_alloc = 2)` -> argmax == GT, GT constraints on the GT rows | full tactical oracle |
| GT_SPEED_ONLY (beside) | `set_r8_force(cons = (NaN, NaN, NaN, v_end GT), min_alloc = 2)` | the speed constraint alone |
| OWN_ARGMAX (beside) | `set_r8_force(lat3 = own argmax, lon6 = own argmax)` | the one-hot SHARPENING the class force applies even when the class is right; splits GT_HYP into sharpening + correction |
| SELF_NAN (control, every 10th episode by file index: 14 episodes) | `set_r8_force(cons = (NaN x4), min_alloc = 2)` | the constraint path at self: must equal OWN bit-exactly (C3) |

Interpretation constraints stated in advance: (i) GT_CONS hands the GT path's constraints to whatever hypothesis the planner ranks first; where its argmax joint differs from the GT tag, the true values sit on a wrong-class hypothesis, and that mismatch is part of the interaction term, not hidden. The fraction of windows where OWN's argmax joint equals the GT tag is reported per seed. (ii) GT_HYP is a ONE-HOT force; OWN_ARGMAX measures what one-hot sharpening alone does.

## 3. Metrics (every arm, both seeds; per family, never pooled)

* **ADE**: S2 = model slots [0,1,2,3] (0.5-2.0 s) on all 4,754 windows; S6 = slots [1,3,4,5,6,7] (1-6 s) on the battery's 3,668 S6 windows; GT = the banked panels' `g` (`panel_s{seed}` / `s6_s{seed}`) keyed by (clip, ws); per-window ADE and every other component via the battery's family code `pair_read_r8.components(ra, P, G, dt)` (dt 0.5 / 1.0).
* **E1** = S2 ADE(arm) - S2 ADE(`ha0_ext_pose`) on the S2 ceiling-compliant windows (4,159 / 138), **E3** the same on S6 compliant (3,251 / 138); `ha0_ext_pose` = `S{2,6}_prior8` of the SHIP-R8 battery's banked `DEPLOY8_POS_step48852ema_pf_s{seed}.npz` (the 8-slot prior SHIP-R8 scores against, AMENDMENT 2), compliant masks from the same file. Raw planner output (no re-timing): the oracle intervenes on the planner, the RETIME stage is downstream and is not part of this question.
* **False-decel on GT cruise** (an3's definition, `.../2026-10-21-refcv8-why-poor/code/an3.py`): on S6, `v_end = |P[5]-P[4]| / 1 s`, `dv = v_end - v0`, class decel if `dv <= -1.5`, accel if `>= +1.5`, else cruise; rate = P(plan decel | GT cruise); beside, missed-decel = P(plan not decel | GT decel).
* **Four families**: every `pair_read_r8.METRICS` key on S2 and S6: ADE (ade, fde), LONGITUDINAL (speed MAE, along-track MAE, |along| final, speed within 0.5, accel MAE), LATERAL (heading, curvature masked, yaw-rate masked, cross-track, |cross| final), TACTICAL (`TAC_traj_lat_correct`, `TAC_traj_lon_correct`). **STRATEGIC: NOT APPLICABLE, n 0** (SHIP-R8 clause (6): `--no-strategic`; the battery's strategic block reads NOT APPLICABLE; the route-checkpoint gate is also off, `--r8-no-rc`), stated, not dropped.
* Beside: per-arm head readback (constraint fed vs GT at the argmax rows, class agreement), allocation match level.

## 4. Estimator

Paired episode-cluster bootstrap (`taniteval.ci.paired_episode_cluster_bootstrap` through `pair_read_r8.paired_cell`), B 2000, seed 0, over the episodes present; models FIXED. It answers *another draw of EPISODES* only. Not another TRAINING run (one training seed, H-ESTIM-SEED-1). The INFERENCE question is answered by running every arm at seeds 0 and 1 and by the floor below. Every arm is paired to OWN of the same seed on the same windows with the same sampler draw.

**Inference-seed floor** `F` (per metric): the paired CI of OWN(s0) - OWN(s1) on the same windows; `F = max(|lo|, |hi|)`.

## 5. Decomposition and the COMMITTED readings

For a lower-is-better metric m, per seed: `H = m(OWN) - m(GT_BOTH)`, `c = m(OWN) - m(GT_CONS)`, `d = m(OWN) - m(GT_HYP)`, `I = H - c - d`; `share_CONS = c / H`, `share_HYP = d / H`, `share_INT = I / H`. Shares and their CIs from the SAME bootstrap draws (ratio of resampled means). Beside: `s = m(OWN) - m(GT_SPEED_ONLY)`, `share_SPEED = s / H`; `sharp = m(OWN) - m(OWN_ARGMAX)`, `correction = d - sharp`.

**PRIMARY metric: S6 ADE, all 3,668 S6 windows. The reading must hold at BOTH inference seeds.**

1. **NO-HEADROOM** if at either seed `H` is not separated above 0 (CI lo <= 0) or `H <= F`. (The tactical layer, even made perfect at T0, does not buy a separated, above-floor S6 ADE gain; the next lever is below the tactical layer: decoder / selection.)
2. Otherwise **REDUNDANT** if `share_CONS >= 0.6` AND `share_HYP >= 0.6` at both seeds (either oracle alone recovers most: the two levers substitute).
3. Otherwise **CONSTRAINT-DOMINANT** if `share_CONS >= 0.6` at both seeds.
4. Otherwise **DECISION-DOMINANT** if `share_HYP >= 0.6` at both seeds.
5. Otherwise **MIXED** (shares, interaction and CIs reported; no lever named dominant).

Point estimates decide the share thresholds; the share CIs are printed beside every reading. The same rule is applied to S2 ADE (4,754 windows) and printed BESIDE; S6 decides. Beside, not deciding: `share_SPEED >= 0.6 * share_CONS` at both seeds -> "the v_end constraint carries most of the constraint share".

Comparison to v8F-mini (INHERITED 0.88-0.90 / 0.07-0.12) is printed as a comparison of two DIFFERENT models, never as a replication.

## 6. Controls (each must read its known value; a failed control is reported, never dropped)

* **C1 OWN reproduces the battery.** Seed 1: the roll's dumped `os` and `plan_full_nav_true` vs the banked `dump_s1` (same argv `--os-only`, same device class): **max |diff| <= 1e-5 m**. Seed 0: the banked `dump_s0` came from the FULL arm-set roll (3 fed rows + a nav-null call per window: a different batch shape and sampler-stream consumption), so OWN s0 is a fresh inference draw; it is compared with the banked s0 and judged against the banked inference floor (os s0 - os s1 S2 ADE -0.0029 [-0.0064, +0.0006], `inference_seed_replicate.json`), not against 1e-5. A C1 failure at seed 1 stamps the run NOT-BATTERY-ANCHORED (the within-run pairing stays valid; the verdict is printed as INCONCLUSIVE-CONTROL).
* **C2 in-run identity:** the OWN trajectory captured by the wrapper equals the dumped `plan_full_nav_true` bit-exactly on every window.
* **C3 seam identity at self:** SELF_NAN == OWN bit-exactly (trajectory, `sel_idx`) on every control window. **If C3 fails, GT_CONS / GT_BOTH / GT_SPEED_ONLY are VOID** (the constraint path perturbs allocation) and the verdict is INCONCLUSIVE-CONTROL.
* **C4 GT-as-prediction readback:** in GT_CONS / GT_BOTH / GT_SPEED_ONLY, the constraint the planner received at the argmax rows equals the GT target on every forced entry: max |error| **exactly 0**; in GT_HYP / GT_BOTH the post-force argmax equals the GT class on **100 %** of forced windows; the analysis's own GT-vs-GT constraint MAE reads **0**.
* **C5 force consumed:** `model._r8_force is None` after every forced forward (the roll raises otherwise).
* **C6 prior invariance + parity:** the emitted `residual_prior_path` is identical across arms per window, and equals the banked `prior8` to <= 1e-3 m (SHIP-R8 clause-7 tolerance).
* **C7 GT parity:** the roll's dumped `g` equals the banked panels' `g` bit-exactly on every window.
* **C8 calibration (literals from `FORWARD_DEPLOY8_step48852ema_pf.json`, banked):** the analysis applied to the BANKED trees' raw `os` must reproduce `os - ha0_ext_pose`: S2 compliant s0 **-0.0069 [-0.0246, +0.0112]**, s1 **-0.0048 [-0.0210, +0.0124]** (4,159 / 138); S6 compliant s0 **-0.7285 [-1.0809, -0.3758]**, s1 **-0.7287 [-1.0635, -0.3826]** (3,251 / 138), to 4 dp.
* **C9 mutation:** the decision function, fed OWN in place of GT_BOTH, must read `H = 0` and NO-HEADROOM; fed GT_CONS and GT_HYP swapped, CONSTRAINT- and DECISION-DOMINANT must swap. Pinned in `code/r8oracle_analyze.py --selftest`, run before any real number.

## 7. Tier and what this is not

**T0**: GT enters at inference (the tactical decision and/or its constraints). This is a WM/planner DIAGNOSTIC of where the tactical headroom sits; it is **never driving performance** and no row of it may be quoted as one. Open loop on logged eval139 frames. One training seed. The oracle measures the headroom available to a perfect tactical layer through THIS planner, not what any trainable head will reach.
