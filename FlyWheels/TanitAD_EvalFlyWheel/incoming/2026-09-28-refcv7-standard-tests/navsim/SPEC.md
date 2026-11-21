# SPEC — refcv7 on the NavSim suite (warmup_two_stage · navhard_two_stage · navtest v1.1) — PRE-REGISTERED

**Written 2026-09-28 ~05:20 Berlin (03:20Z), BEFORE any refcv7 NavSim score exists.** At this moment
no refcv7 checkpoint has been run through any NavSim input, and no scorer has seen a refcv7 plan.
The sha256 of this file is recorded in `raw/SPEC_PREREG_HASH.txt` the moment it is written, before
the pipeline-validation run (§6). Any later change is a dated amendment below §12, made before the
scores it touches, with its own hash line.

**Authority.** `Project Steering/SPEC_REFCV7.md` (§3 BAR-R7-N1; §4 "battery + NavSim at 5k / 15k /
30k / final"; §7 A2 "filter ON and filter OFF on the same windows"; §10 A5 the prior) and
`Project Steering/PREREG_REFCV7.md`. **Template (ported, not re-derived):** the refcv6 NavSim suite,
`../../2026-09-23-refcv6-standard-tests/navsim/` (its SPEC §0–§12 and amendments A1–A4 bind here
wherever this file does not say otherwise). Code under test is the clean tree
`0c444082a08efdcbeef36585f1b3baabbfafc1fc` (extracted with `git archive` to
`C:/Users/Admin/ev7nav`; the refcv6 package's code there is byte-identical, EOL-normalised, to the
D: copy — checked 2026-09-28 05:15 Berlin).

## 0. Facts this SPEC is built on (MEASURED 2026-09-28 unless marked)

| fact | evidence |
|---|---|
| The run: `refcv7-r101-s0` on Thor, 50,400 steps, b16; at 05:11 Berlin its `train.log` read step 1,500. `config.json` pulled read-only by the battery stream: `D:/refcv7_eval_kit/ckpt/config.json`, md5 `e6512a01b9c70f0e4a4dac581621984a` | file |
| Milestones: the trainer saves a PERSISTENT, never-rotated `ckpt_{step}.pt` at `MILESTONES = (5000, 15000, 20000, 30000)` (`stack/scripts/refc_v3_train.py:225`, `:9645`) beside the rolling `ckpt.pt`; the battery stream's waiter pulls each to `D:/refcv7_eval_kit/ckpt/ckpt_<N>.pt` with `MD5SUMS` | source read; Master Mind 05:10 + correction |
| Loader: `stack/tanitad/eval/refcv7_loader.py` (blob `a8eb3e7e`), replaying `train()` block by block, STRICT load, G-DVB on the built model, `config.json` stamp checks. G-EVAL on Thor: strict 0/0 over 1,131 keys, 188/188 outputs | INHERITED (brief), not re-verified here beyond: this suite builds only through that loader and records its record |
| What changed vs refcv6 AT THE NAVSIM SEAM (from the run's argv): `--residual-prior ha0_ext_pose` (NEW-1); `--map-hires on` + `--bev-source map_hires_pool` (A6: the forward's ONE lift is stride 8 at 0.25 m over x 0–100 m, y ±30 m — `--w-map 0`, the stride-16 lift is not built); `--speed-ceiling-filter` (A2, inference only); `--graft-nav-compliance` (τ 0.18063741505146028, train-derived file); `--graft-tac8-prior`; box/agent heads 300 queries, `learned_ref`; trunk equalisation of the bottom 43 rows now BUILT (FIX-3); strategic layer OFF | `config.json` argv |
| refcv7's forward is fed by `compute_losses_v3` (`refc_v3_train.py:4350`) as: `model.core.set_ego_window(pose_hist, 8, actions=None)` (ha0_ext_pose reads no actions), `(perception_grid, perception_valid) = model._lift_bank_hires.for_episodes(map_ep)`, then `model(frames, nav_cmd, v0, steps, lan, ego_state, nav_args, v_max_ms, v_max_valid, agent_gt, perception_grid, perception_valid)` | source read |
| The prior `ha0_ext_pose` (`tanitad/models/kinematic_prior.py::prior_controls`): `a0 = (v[t0] − v[t0−0.1 s]) / 0.1`, `κ0 = clip(ω0 / max(v0, 2.0), ±0.3)` with `ω0 = wrap(yaw[t0] − yaw[t0−0.1 s]) / 0.1` — the ego-history encoder's own backward differences on the pose window; `P` = constant (a0, κ0) rolled from v0 by `rollout_unicycle` at 0.1 s | source read |
| The ceiling the filter reads is the containing-window bin's limit of the fed one-hot (`refc_v3.py:1782-1790`): `valid = 0` → `+inf` (no filter) | source read |
| The selection's reachability clamp is ON for this run (`config.json` `selection.sel_reach_clamp true`, `sel_accel_max 2.5`) | file |

**Consequences for the seam, stated before any score:**
1. **The NavSim ego window is the refcv6 suite's construction, unchanged**: 8 × (x, y, yaw, v) at
   t0−0.7 … t0, LINEAR in time between the 2 Hz states at t0−1.0 / −0.5 / 0 (A1 of the refcv6 SPEC
   for dropped frames). ⇒ **On NavSim the prior's a0 and ω0 are the MEAN acceleration and yaw rate
   over [t0−0.5 s, t0]**, where training read a 0.1 s backward difference at 10 Hz. That is the
   honest content of 2 Hz data; it makes NavSim's prior SMOOTHER than training's, and it is a
   zero-shot difference of the SAME kind as the frame rate itself. Stated, not corrected.
2. **The prior is a model-free function of the declared inputs**, so this suite scores it as its own
   control (`PRIOR_ha0p`, §3) — the reading of what the learned residual adds on NavSim.
3. **The lift geometry is the stitch's virtual camera** (refcv6 `rig6.py`: rotation = the stitch's
   `M`, centre = CAM_F0's mount, height = F0.z − road plane −0.3557 m) **built with the MODEL'S OWN
   0.25 m bank parameters** (stride, heights, grid, 43-row observed mask read off
   `model._lift_bank_hires`), never retyped.

## 1. Question

For a refcv7-r101-s0 milestone checkpoint, run **zero-shot** on NavSim with every input it was
trained on built from NavSim's own data, **does it drive better than doing nothing (STOP), constant
velocity (CV), the kinematic echo (ECHO) and its own kinematic prior (PRIOR) on the official
scorer** — above all on the **FULL navtest split (BAR-R7-N1)** — and what do its inputs and its
declared selection mechanisms contribute, read against its own inference-seed floor?

## 2. The inputs — every one an explicit function in `code/refcv7_bridge.py`

| input | construction | test (literal / analytic; the arm that must go RED) |
|---|---|---|
| frames, nav, v0, ego history, max speed (map) | the refcv6 suite's functions, IMPORTED unchanged (`refcv6_bridge.declare6` / `ego_history_poses` / `nav_input` / `v0_of` / `max_speed_input`, banks `Bank416` / `BankNavtest416`, `pack_rows`, `knots_to_navsim`) under refcv7 arm specs | re-run here: 2.0 m/s² literal, −0.2 rad/s literal, ±π unwrap, dropped-frame A1, refusals, declared-input seam, wrong-grid arm RED |
| **residual prior window** | the ego window above, handed to `set_ego_window(poses, 8, actions=None)` exactly as `compute_losses_v3` does | the prior of a constant-accel history reads **a0 = 2.0 m/s²** and κ0 = 0; constant yaw rate −0.2 rad/s at 8 m/s reads **κ0 = −0.025 1/m**; at 1 m/s the floor gives **κ0 = −0.1**; a rigid transform of the history leaves (a0, κ0) unchanged; **arm RED**: the 2 Hz states packed as consecutive 0.1 s samples read a0 = 10 m/s² |
| **prior path P** | `kinematic_prior.prior_controls` + `prior_path` (the model's own functions) on the declared window, horizons read from the build | a0 = κ0 = 0 at 12 m/s → x = 12 t exactly at the 8 knots; the model's EMITTED `residual_prior_path` equals the bridge's model-free P on every row (control **KPR**, ≤ 1e-4 m) |
| **lift (0.25 m)** | `build_lift_geometry(virtual camera, frame=FRAME_416x1024, stride, heights_m, grid, observed)` with every parameter read off `model._lift_bank_hires` | **KL-lift**: for PhysicalAI clips in the model's own bank, this function on the bank's camera reproduces `bank.geometry(ep)` **bit-exactly**; analytic road-point row; **arms RED**: stride 16 (the refcv6 lift) and the road offset ignored |
| **max-speed ceiling** | the fed one-hot's bin limit (`limit_ms_of_bin`), applied by the model's own filter | literal: 25 mph → bin 50 km/h → ceiling 13.888… m/s; 35 mph → 100 km/h; unknown → +inf |
| **forward contract** | `forward_kwargs7`: exactly the keyword set `compute_losses_v3` passes to `model(...)` | an AST read of `compute_losses_v3` must yield the SAME literal keyword list (a trainer that adds a keyword turns it RED) |
| output | `out["traj"]` 8 knots (0.5…6 s) → E2's `knots_to_navsim` (imported) | as refcv6 (CV line exact, circle heading, wrong-knot-times RED) |

## 3. Arms

| arm | frames | nav | max speed | ceiling filter | seed | role | splits |
|---|---|---|---|---|---|---|---|
| **R7_A1** | ST | cmd | map | **ON** (as configured) | 0 | **PRIMARY — the bar arm** | all three |
| **R7_A1_s1** | ST | cmd | map | ON | 1 | **inference-seed replicate** (the floor every read is held against) | all three (navtest FULL) |
| R7_FILTOFF | ST | cmd | map | **OFF** | 0 | SPEC_REFCV7 A2's sensitivity reading, a real second forward | warmup; navtest 200-token subset |
| R7_FILTOFF_d | = R7_A1's forward, the argmax WITHOUT the ceiling mask | | | OFF | 0 | the A2 reading on the full splits, DERIVED — **admissible only if control KF reads bit-identical plans vs R7_FILTOFF on every warmup scene** | navtest full, navhard |
| R7_BLIND | constant grey (bank mean) | cmd | map | ON | 0 | frames-blind: the registered deliberate regression | warmup |
| R7_NAVOFF | ST | withheld (`nav_cmd=None`) | map | ON | 0 | nav contribution (reported REFUSED if the build refuses it) | warmup |
| R7_VMAXOFF | ST | cmd | withheld (`valid=0`: also no ceiling) | — | 0 | max-speed contribution | warmup; navtest subset |
| R7_A1NT | NT | cmd | map | ON | 0 | time-construction sensitivity | warmup |
| ⛔ R7_VMAXORACLE | ST | cmd | the human's realised max over [t0+2, t0+6] s (refcv6 `vmax_oracle.py`) | ON | 0 | **PRIVILEGED DIAGNOSTIC** — prices the train/deploy mismatch; never a result, never in a bar | navtest subset |
| **PRIOR_ha0p** | — | — | — | — | — | the model's own prior P alone (model-free; Δ = 0) | all three |
| CV_official / STOP_zero / ECHO_ha0_ext | — | — | — | — | — | the banked model-free floors (refcv6 suite / E2 / W3 / W7) | as banked |
| HUMAN (navtest) | — | — | — | — | — | W3's banked human-log reference (94.5514) — context, not a floor | navtest |

Stage-1 tokens: as the refcv6 suite (navhard: all 450 real; warmup camera arms: the devkit CV agent
is the DECLARED stand-in and their official two-stage EPDMS is HYBRID).

## 4. Statistics, estimators, stamps

* **warmup_two_stage** — S2-EPDMS-u over the 204 stage-2 tokens; paired per-scene W/T/L. ⛔ **No
  interval** (7 logs < the 8-cluster floor).
* **navhard_two_stage** — the official two-stage EPDMS (`extended_pdm_score_combined`), plus
  S2-EPDMS-u. Interval: `taniteval/adapters/navsim_ci.py` log-cluster bootstrap, PAIRED variant for
  every difference (76 log clusters, B 2000, seed 0).
* **navtest v1.1** — PDMS ×100 over the FULL 12,146 tokens (W3's harness); paired log-cluster
  bootstrap (136 logs, W2's registered estimator). Floors W3's banked STOP 61.8202, CV 20.6517,
  HUMAN 94.5514 — each re-read against this package's own harness control (§7 KH).
* ⭐ **Every interval names its question**: the log-cluster bootstrap answers *"another draw of
  LOGS?"* only. **Inference variance** is read from R7_A1_s1 on every split. **Training variance**
  is UNTESTED (one run, one training seed — `H-ESTIM-SEED-1`), so per SPEC_REFCV7 §3 a margin
  within **2× the inference-seed floor** is reported **NOT PROVEN**, whatever its interval says.
* **Tier stamps on every row**: NavSim **open-loop benchmark** (T1-family: the planner's own plan is
  scored; stage-1 loop OPEN; v2 background IDM-reactive, v1 non-reactive/logged); ⛔ **never closed
  loop**; **zero-shot** (PhysicalAI-AV B1 → nuPlan cameras, 3-camera stitch, camera ~0.57 m higher
  than the training rigs' median); **non-parity** data; device + precision of every refcv7 row (one
  device per split, refcv6 amendment A3); the model-as-trained stamp of the checkpoint (§9).

## 5. The bars (committed, both outcomes) — milestone checkpoints only (step ≥ 5,000)

* ⭐ **BAR-R7-N1 (programme bar, SPEC_REFCV7 §3, verbatim):** *"NavSim navtest PDMS > STOP, with the
  paired log-cluster interval excluding 0, on the FULL split."* Read on **R7_A1 (filter ON)** over
  all 12,146 tokens. **PASS** iff PDMS(R7_A1) > 61.8202 AND the paired interval of R7_A1 − STOP
  excludes 0 AND the margin exceeds 2× the full-split inference-seed floor |PDMS(R7_A1) −
  PDMS(R7_A1_s1)|. Interval excludes 0 but the margin is inside 2× the floor → **NOT PROVEN**.
  Otherwise **FAILED**. Beside it, never instead of it: R7_A1 − CV, R7_A1 − PRIOR_ha0p (what the
  residual adds), R7_FILTOFF_d (A2), the stop fraction, and the per-command / per-speed-band
  decomposition.
* **BAR-R7-NW1 (suite bar, carried from BAR-R6-W1):** S2-EPDMS-u(R7_A1) > max(CV, STOP, ECHO) on
  the 204 warmup tokens, margin above 2× the warmup seed floor; no interval exists.
* **BAR-R7-NH1 (suite bar, carried from BAR-R6-H1):** official two-stage EPDMS(R7_A1) > max(STOP, CV,
  ECHO) on navhard, paired interval vs STOP excluding 0, margin above 2× the navhard seed floor.
* ⚠️ **Stop-gaming** (W3: a command-gated stop cleared a STOP bar): every bar row carries the stop
  fraction (4 s endpoint < 1 m). An admissibility clause against stop-gaming is a PI decision, not
  assumed here.
* **Lever reads (secondary, never bars)**, each paired and read against the seed floor: A1 − PRIOR
  (the residual), A1 − FILTOFF (the ceiling filter), A1 − BLIND (vision), A1 − NAVOFF (nav), A1 −
  VMAXOFF (max speed), A1 − A1NT (time construction), ORACLE − A1 (PRIVILEGED; the mismatch).
* **FAIL ⇒ Rule Zero**: decomposed automatically (multipliers, per command, per speed band — the
  refcv6 ladder `decompose6.py`, imported) and the next lever named in the RESULT.

## 6. Pipeline validation — the pre-milestone checkpoint, NEVER A RESULT

The most recent complete checkpoint available before step 5,000 (pulled ONCE from Thor's rolling
`ckpt.pt` with size/mtime stable, md5 on both sides, its step read locally and the file named by
that step) runs **warmup only**, every warmup arm of §3 plus PRIOR_ha0p and R7_FILTOFF_d, through the
whole path: bridge → seams → official scorer with count guards → statistics → bars file. Its purpose:
the count guards, the seam, the input reach, the controls KPR / KF / K0 / KD / KI on real scenes,
the seed floor's size, and the device/throughput needed to schedule the milestone. ⛔ Its scores are
stamped **VALIDATION ONLY** in every artifact, no bar is evaluated on it, and it is never quoted as
refcv7's NavSim performance.

## 7. Controls (each must read a known value, or the numbers it guards are void)

| id | control | pass |
|---|---|---|
| KH | harness known values through THIS package's drivers: warmup CV, STOP, ECHO vs the banked E2 CSVs; navtest STOP vs W3; navhard CV vs W7 | every per-token cell \|Δ\| ≤ 1e-9 and the headline statistic equal; a split's refcv7 result is admissible only beside its KH |
| KPR | the model's emitted `residual_prior_path` == the bridge's model-free P | max \|Δ\| ≤ 1e-4 m on every refcv7 row (checked per row, recorded) |
| KF | R7_FILTOFF_d (derived argmax) == R7_FILTOFF (real forward), and the derived FILTER-ON argmax == the model's `sel_idx` | bit-identical plans on every warmup scene; the per-row consistency check on every row of every split |
| KL-lift | the bridge's 0.25 m geometry == `model._lift_bank_hires.geometry(ep)` on PhysicalAI clips | bit-exact (grid and valid) |
| K0 | determinism | same scene + seed ⇒ bit-identical plan |
| KD | exact-duplicate dedup | == the trunk's native path (max \|Δ\| 0.0 on CPU; re-measured on CUDA before use) |
| KI | input reach | frames, nav, v0 and history each move ≥ 1 of 4 plans; the max-speed one-hot reaches the tactical logits |
| K5/K6 | declared-input seam | undeclared randomised ⇒ identical inputs; declared mutated ⇒ changed |
| K8 | count guards | log successful == expected, failed == 0, CSV rows valid, agent calls == seam declaration |
| KX | seam transparency | the unchanged E2 seam agent (refcv6 KX holds) |

## 8. Four metric families

As the refcv6 suite (`families6.py`, imported): LONGITUDINAL / LATERAL / TACTICAL are computable
where a logged human future exists — **navhard stage 1 (n = 450)** and **navtest (n = 12,146)** —
through `taniteval/adapters/navsim.py` (`scenes_to_win` → `four_families_block`, log-cluster CI),
for R7_A1 and for PRIOR_ha0p on the same scenes. Warmup and navhard stage 2 carry no GT future ⇒
those three families are **UNAVAILABLE there, reason + n = 0**. **STRATEGIC: not applicable** —
the strategic layer is OFF (`--no-strategic`) and NavSim scores no strategic decision; stated per
family, never dropped.

## 9. Compute, order, stamps

* Dev-box GPU (RTX 4060, 8 GB), ONE job at a time: the exclusive lock
  `C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock` (job name + pid; removed after) AND no other
  python compute app in `nvidia-smi`. At a milestone the four-family battery goes FIRST; this suite
  waits for the battery's lock release. CUDA → the trunk's recorded levers (bf16 + NHWC) as trained;
  CPU → fp32. One device per split, recorded on every row (refcv6 amendment A3).
* Order per milestone: warmup → navtest (R7_A1, R7_A1_s1 full; subset diagnostics) → navhard. The
  scorer (CPU) runs as each seam lands, RAM-gated.
* The model-as-trained stamp is read from the run's record at each checkpoint (argv sha and
  `config.json` md5); a mid-run change of the recipe is flagged on every comparison that straddles it.

## 10. What this SPEC does NOT claim

1. ⛔ No closed-loop claim. NavSim here is an open-loop benchmark on logged/synthetic scenes.
2. ⛔ No comparison with published NavSim leaderboards as a bar: refcv7 is camera-only, zero-shot,
   single front stitch; published numbers are context, cited from their primaries only.
3. ⛔ No training-variance claim (one training seed).
4. The max-speed map input is the refcv6 suite's deployable stand-in for a channel TRAINED on the
   ego-future oracle (refcv6 SPEC A4); the gap is priced by R7_VMAXORACLE, never hidden.
5. The ceiling filter acts on that map input at inference (A2) — the primary reading is the model
   AS CONFIGURED; R7_FILTOFF(_d) is the sensitivity reading.

## 11. Amendment A1 — 2026-09-28 ~05:47 Berlin (03:47Z), BEFORE any refcv7 NavSim score: the speed-ceiling filter never reaches refcv7's EMITTED plan; R7_FILTOFF_d is withdrawn, R7_CEILDECL_d is added as a diagnostic

**What was found (MEASURED, not inferred).** The step-1,500 pipeline-validation BRIDGE (warmup, CUDA,
trajectories only — no scorer has run on any refcv7 plan; `raw/validation/step1500/bridge_warmup/`)
emitted **bit-identical plans for R7_A1 (ceiling filter ON) and R7_FILTOFF (OFF) on 204/204 stage-2
scenes**, although the ceiling masked at least one candidate in 97 of the 126 scenes with a finite
ceiling, and the emitted plan lay OVER the ceiling in 2 of them. The derived filter-off argmax of §3
(the decoder's `sel_score` under the reach mask) reproduced the model's `sel_idx` on only 199/204.
The cause, read at source: `RefCV3Model.forward` RE-SELECTS over the fan after the decoder — E9 goal
selection, `stack/tanitad/refs/refc_v3.py:2285-2298`: `blended = apply_seam_clamp(sel_score,
goal_gate · scorer)`, `rank = blended` masked by `reach_keep` ONLY, `argmax` → `out["traj"]`; the
decoder's ceiling-filtered pick (`refc.py:3605-3610`) is demoted to `sel_idx_base`. The additive
selection terms (nav compliance, behaviour) survive inside `sel_score`; the ceiling, a MASK, does not.
⇒ **SPEC_REFCV7 A2's "the fed set-speed filters the ARGMAX" is BUILT in the decoder and INERT on the
emitted plan** — the declared-but-not-reaching class of refcv6's COMMS escalation 9, surviving FIX-4
because G-DVB checks the filter is built and G-LIVE that the mask is active in an eval step, and
neither asserts the EMITTED plan obeys it. Escalated (COMMS, report headline).

**Changes (no bar, primary arm, statistic or floor changes; R7_A1 remains the model AS BUILT):**
1. **R7_FILTOFF_d is WITHDRAWN.** SPEC_REFCV7 A2's filter-OFF reading on the full splits IS R7_A1's
   emitted plan (the filter cannot reach it); it is reported as "identical to R7_A1 by construction",
   supported by control KF (below), and not scored separately (identical seams score cell-identically,
   refcv6 KDET).
2. **Control KF is redefined:** (a) R7_FILTOFF (a real second forward) vs R7_A1 — the count of
   bit-identical emitted plans, on warmup (all) and the navtest 200-token subset; (b) on EVERY
   refcv7 row, the bridge's re-derivation `argmax(sel_score_v3 | reach_keep)` must reproduce the
   model's `sel_idx` (same device, dtype and kernel), else the row carries no derived reading.
3. **R7_CEILDECL_d is ADDED — ⛔ DIAGNOSTIC, never a bar arm, never "refcv7's number":** from
   R7_A1's own forward, the E9 rank ALSO masked by the captured ceiling keep (an empty intersection
   keeps the emitted pick — the filter's own empty-row rule), i.e. the ceiling AS DECLARED. It prices
   the zero-training fix (apply the mask at the E9 argmax); whether to evaluate or ship that is a
   PI / Arch decision, not taken here. Scored on warmup and on the FULL navtest split (it costs no
   forward), paired against R7_A1 with the same estimators.

**Why it cannot have been tuned on a score.** No refcv7 NavSim score exists at this moment: the only
refcv7 artefacts are the step-1,500 validation trajectories (no scorer run) and the model-seam
tests. The finding is about which code path emits the plan, read from source and from plan identity.
