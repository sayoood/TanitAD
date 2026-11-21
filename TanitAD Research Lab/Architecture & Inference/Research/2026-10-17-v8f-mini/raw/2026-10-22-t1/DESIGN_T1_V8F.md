# DESIGN — a T1 (self-action) eval for v8F-mini

Written BEFORE the code. Tip = mirror `f135c49`. Paths are repo-relative at that tip. Evidence classes: MEASURED = ours, artifact in `probes/`; READ = read from source at the cited line.

## 0. Headline, and one conflict the brief must hear

**The doctrine does not say what the brief says.** `Project Steering/EVAL_DOCTRINE.md:3-16` (PI amendment 2026-09-02): *"T0 and T1 are both OPEN LOOP. T2 is the only closed loop"*; `:29` T1 = *"self-action OPEN loop: predictor consumes the decoder/planner's own actions; perception context fixed at t0 ... NEVER 'closed-loop'"*; `:35-39` T1 cannot support "the model drives". The brief's "action-closed loop" is the pre-amendment wording. `taniteval/tools/t1_eval.py:1-12` still carries that stale wording too (title "ACTION-CLOSED-LOOP", "a capability claim requires T1 or better") and its `analyze()` output still prints `_tier_doctrine` with it (`t1_eval.py:479`); `v8f_t1_eval.py` overwrites that field in its own output. **Doctrine wins; I follow the amended text.**

What this tool is, in doctrine vocabulary: **T1-kin — self-action OPEN loop with a closed KINEMATIC state.** The ego state the model reads (speed, accel, yaw rate, odometry, route progress) is the model's OWN executed motion, rebuilt at every replan. The camera stream is NOT re-rendered (that is T2). It is stamped `T1`, never "closed loop", and supports only open-loop trajectory-quality claims, not "drives".

Why it still earns its keep: v8F-mini's existing eval (`stack/scripts/v8f_mini_eval.py`, `cmd_run` L231) is ONE shot from a logged t0 state; every kinematic input is the human's. The `H-MINI-K-1` margin over the P10 floor (0.31 / 0.43 m at 6 s) is therefore a single-plan margin. T1-kin asks whether the margin survives when the model has to live on its own motion for 6 s.

## 1. What the model reads at inference (READ, `v8f_mini.py`)

`INFERENCE_INPUT_KEYS` (`:100-101`) = `feats, frame_ok, p10f, act, nav, pose, limit, limit_ok, base, a_t, a_s, pose_now, pose_t, pose_s`. `forward_infer` (`:627-632`) wraps the batch in `_KeyGuard` and records every key read. By reading `_forward` (`:587-625`), `run_s/run_t/run_o` (`:541-581`):

| key | where read | content | ego-derived? |
|---|---|---|---|
| `feats`, `frame_ok` | `_scene` `:513`, `_encode` `:583` | pooled DINOv3-L tokens of `hist=4` frames ending at the tick frame | camera stream (see §4) |
| `p10f` [F,8] | `_ego(_at(p10f, i))` (`:506`; called `:544,:555,:571`) and the prior (`:579`) | `kin_features(v0, a0, kappa0)` at the tick frame; kappa0 = clamp(omega0/max(v0,2), +-0.3) from the MEASURED 10 Hz yaw (`v8f_mini_data.py:285-298`) | **YES — replace** |
| `base` [30,2] | prior `:579` | kd_x path integrated from `kin3[t]` (`v8f_mini_data.py:325`) | **YES — replace** |
| `pose_now/_t/_s` | `transform_ckpt` `:606,:621` | world (x,y,yaw) at NOW / T tick / S tick; only RELATIVE poses matter | **YES — replace** |
| `nav` [F,5] | `e_nav(_at(nav,i))` `:544,:555,:571` | token one-hot (FOLLOW/L/R), `d_next/100` clipped to [0,3], valid | route; **contains logged progress** (§4) |
| `limit`, `limit_ok` | `e_lim` `:545,:556,:572` | per-clip speed bucket one-hot | declared oracle (§4) |
| `a_t`, `a_s` | `run_o` ages `:570` | sample-and-hold ages (cache steps) | schedule, no ego info |
| `act`, `pose` | **not read** by `forward_infer` (`act` only at `:691`, inside `loss`) | | dropped from the T1 batch |

The T1 loop hands the model **only** the keys it reads (allowlist). A model that tried `act`/`pose` would raise `KeyError` — loud, not silent.

## 2. What gets replaced, what stays logged

**Replaced by the model's own executed motion** (all from ONE executed 10 Hz pose track, §3): `p10f`, `base`, `pose_now/_t/_s`, and (unread) `act`/`pose`/`a`/`ch1`.
**Stay logged**: `feats`, `frame_ok` (camera: T1 is not re-perception), `limit`, `limit_ok`, `a_t`/`a_s` (from a schedule), labels (never read).
**Replaced by a dead-reckoned route (default)**: `nav` rows after t0 (logged mode kept as `--nav logged`, flagged non-strict).

The model's weights, config and `forward_infer` are untouched; the P10 prior is the checkpoint's own (inside `ck["model"]`).

## 3. The loop, and the frame bookkeeping

Grid: cache step = 0.2 s = raw frames 2j (`v8f_mini_data.py:7-9`); the native track is 10 Hz (raw frame k = 0.1 s). Window = (clip, t0). Tick n has NOW = `t_n = t0 + n*m`; `m = --replan-steps` (default 1 = 5 Hz; must divide 30). Horizon 30 steps = 6 s (the same 30 GT waypoints as the open-loop eval, `gt_waypoints`).

Per tick, for a batch of windows:
1. Build items with the UNCHANGED `V8FMiniWindows.item(ci, t_n, a_t, a_s)` (`v8f_mini_data.py:471-498`) over a per-window VIEW of the clip whose ego arrays (`pose, kin3, p10f, base, a, ch1, v`) are: logged for steps <= t0, **NaN beyond t_n**, own-executed in between. So the item builder cannot see the logged ego after t0 even if it tried.
2. Patch `nav` (§4), optionally `limit`, optionally `feats` (frozen scene); filter to the allowlist; `forward_infer`.
3. Take `traj30` (the model's plan at 0.2 s, ego frame at NOW). World = `p_n + R(yaw_n) * plan`.
4. **Execute the first m knots** with perfect tracking (no actuator model, no feasibility projection; stated idealisation).
5. Rebuild the 10 Hz executed track for the new frames. **New math, stated so it can be attacked:** the plan is on a 0.2 s grid and `traj60`'s odd points are linear midpoints of the prior (`upsample_prior`, `v8f_mini.py:244-251`). One-frame backward differences on those midpoints give `a0 = omega0 = 0` at every tick frame (a sawtooth artefact). So the 0.1 s frames are filled by **cubic Hermite with Catmull-Rom tangents** over the plan's knots (previous executed knot as the left neighbour). Position, heading (tangent angle, held below 1 mm/interval) and speed (|H'|) per frame; then `kin3`, `p10f`, `base` come from the project's ONE spelling `yaw_kin_from_track` / `kin_features` / `kdx_base` on the appended slice. Quadratic paths are exact (tested against analytic v, a); a constant-curvature arc is recovered to <1 %.
6. Record the executed knot positions in the t0 ego frame (the `cl` arm).

**Ages / sample-and-hold.** Default `--schedule periodic`: T ticks every 3 steps, S every 6 (`a_t = n mod 3`, `a_s = n mod 6`; both inside the trained support `max_age_t=2, max_age_s=6`, `V8FMiniConfig:137-138`). `--schedule fresh`: a_t=a_s=0 each tick. Because `forward_infer` recomputes S and T at their tick frames from that frame's inputs, the executed past is automatically what they read. Tick 0 is always fresh, so tick 0 == open-loop `--ages fresh` (NOT the evaluator's default `cycle`, `v8f_mini_eval.py:289,:627`).

**Arms in one pass** (each its own executed track): `cl` = the model (T1); `fl` = the P10 floor driven the same way (prior only: `prior(p10f[:, -1], base)`, same call as `run_o:579`; T1); `ol` = tick-0 plan (== open-loop eval, stamped T0 as `t1_eval.DEFAULT_TIERS` does); `ha0` = constant speed at the measured t0 speed (T1 control, `fk.ha0_paths`).

## 4. The leak test — every inference input, one question each

Question: *could this contain the logged future, or the logged ego history after t0?*

| input | verdict | handling |
|---|---|---|
| `p10f`, `base`, `pose_*` | contain logged ego if built from the logged track | **rebuilt from the executed track; logged rows beyond t0 are NaN; test goes RED if a logged row is read** |
| `nav` | **YES, in logged mode.** `d_next` is the distance to the next manoeuvre from the HUMAN's position (`build_label_join` `:149-176`, nav block): at tick n it counts down the human's progress, and later rows announce the human's next manoeuvre. | default **dead-reckoned**: token and valid held from t0, `d_n = max(d_0 - executed arc since t0, 0)`. MEASURED on the real eval join (141 clips, `probes/nav_probe2.py`, `nav_probe4.py`): against the logged `d_next`, 9,474 token-held (window,step) pairs, median error 0.0002 m, 98.35 % within 1 m; FOLLOW-with-valid-d 27,534 pairs, 100 % within 1 m; 2,376 token changes, **100 % occur at dead-reckoned distance 0**. What dead-reckoning cannot know is the NEXT manoeuvre (not knowable at t0 without the human's path). Dwell at d=0 before a token change: median 17.5 steps (n=26 clean runs, `nav_probe3.py`), so holding the token for <= 6 s is inside the data's behaviour. `--nav logged` exists, is stamped NON-STRICT, and is what the exact-reproduction control uses. |
| `limit`, `limit_ok` | **YES, by declaration.** The bucket comes from `SPEED_BAND.v_hi_ms`, *"the max of the ego's realised future speed (R^2 0.97 recoverable) => an oracle the PI chose to feed WITH declaration (D-SPEEDBAND-EGO-ECHO-1)"* (`Project Steering/Handover/2026-10-15-refa-refd-flagship/D_PROGRAMME.md:109`). INHERITED, not re-verified here. | kept (removing it is a model-input change, forbidden by the brief); identical in open loop and T1, so it cannot differentially help T1; stamped on every output. `--limit off` zeroes it (sensitivity arm). |
| `feats` | **YES, unquantified.** Camera frames after t0 are the HUMAN's view: a 4-frame stack encodes the human's ego-motion and the scene he reached. This is the doctrine's "perception fixed at t0" clause being relaxed. | `--scene logged` (default; the brief's reading) vs `--scene frozen` (doctrine-literal: every tick reads the t0 stack). The two bracket the camera channel; frozen has its own defect (a still scene next to a moving state). NOT claimed leak-free. |
| `a_t`, `a_s` | no ego info | schedule |
| evaluator keys (`gt`, `s_h`, `lead_gap_h`, `lab*`) | GT future | never reach the model (allowlist) |

Net: the strictly-closed parts are the kinematics and, by default, the route. Two channels remain open and are named, not hidden: the camera stream and the speed bucket. **A T1-kin number is quotable only with `scene`, `nav`, `limit` and `m` beside it.**

## 5. What `t1_eval.py` gives, and what is new

**Reused unchanged:** `analyze()` (`t1_eval.py:336`): four families through `taniteval.four_families`, S-curve rate, lag, event response, paired episode-cluster bootstrap, tier stamps (`resolve_tiers` refuses an unstamped arm), `dt` is a parameter (0.2 here). I write per-episode `ep*.npz` dumps in its schema (`g`, arms, `eid`, `clip_index`, `v0`) and call it. **Not reused:** its roll (`run_rollout*`, `roll_closed`) — those drive a world model from `epcache` frames and a unicycle head; v8F-mini has no latent roll and its inputs are cached pooled features.
**New:** `stack/scripts/v8f_t1_eval.py` — the per-window executed-track state, Hermite fill, dead-reckoned nav, the view/allowlist machinery, the rollout driver, dump writer; `stack/tests/test_v8f_t1.py`.
Extra outputs `analyze()` does not give: executed-vs-logged pose deviation per step, executed accel / jerk extremes, and the sweep over `m` (m=30 must equal open loop — §6).

## 6. Controls the tests pin

* **C1 exact reproduction.** Executor = the logged motion, `--nav logged`, `--scene logged`: every key the model READS is bit-identical to the open-loop batch for window (ci, t_n) at every tick, and `traj30` equals it (max abs diff reported).
* **C2 m = 30.** One tick, the whole plan executed: the executed knots ARE `traj30` -> `cl` == `ol` exactly.
* **C3 perturbation.** Add a fixed offset to the model's executed plan -> the next tick's `p10f`, `base`, `pose_*` change (and a `nav` d decrements by the executed arc) -> the next plan changes. The loop is closed on state.
* **C4 leak guard.** Poison every logged ego array beyond t0 with NaN / garbage -> T1 output unchanged bit-for-bit, while the OPEN-LOOP item builder fed the same poisoned arrays changes. Mutation: a deliberately leaky executor (reads logged `kin3[t_n]`) must make the guard FAIL.
* **K known-value.** Hermite on a constant-acceleration straight path and a constant-curvature arc recovers analytic (v0, a0, kappa0).

## 7. Built and CPU-tested (added after the code)

Files: `tree/stack/scripts/v8f_t1_eval.py` (rollout, per-window state, Hermite fill, dead-reckoned nav, envelope, dumps, `run` / `analyze` CLI) and `tree/stack/tests/test_v8f_t1.py` (23 tests, ~18 s, CPU, synthetic world; `PYTHONPATH=stack;taniteval;stack/scripts`).

**Test result: 23 passed.** Pinned: K (Hermite recovers analytic v, a0, kappa0 on a constant-acceleration line and a constant-curvature arc, holds heading at standstill); C1 (30 ticks, 6 windows, every key the model reads bit-identical to the open-loop batch of the same windows, plan max-abs-diff 0.0); C2 (m=30 executed path == tick-0 plan to 1e-9); C3 (a 0.5 m/0.6 m shift of the tick-0 executed plan leaves tick-0 inputs untouched and changes tick-1 `p10f`, `base`, `pose_now`, `nav` and the tick-1 / tick-4 plans); C4 (NaN and 1e3-garbage in every logged ego row after t0, store and track: output bit-identical and finite; the open-loop item builder over the same poison changes); allowlist (exactly `MODEL_KEYS`, `act`/`pose` unread, a read of a dropped key is a `KeyError`); dead-reckoned nav against hand values (2 m/step countdown, clamp at 0, token held, invalid args frozen); floor planner == `forward_infer`'s `p10`; the scene-frozen and limit-off knobs; ages schedule inside the trained support; `m` must divide 30; CLI end to end through `t1_eval.analyze` (tier stamps `cl` T1 / `fl` T1 / `ol` T0, paired blocks, four-family blocks, strict-flag header); a `kin_mode steer` checkpoint is REFUSED; the real `M2b_s0` checkpoint (hmax 16, WM aux b) loads and rolls finite on the synthetic world.

**Mutations (source edited, tests must go RED; `mutation_results.json`):**

| mutation | tests that went red |
|---|---|
| M1 keep the logged future in the view | `test_window_state_has_no_logged_ego_after_t0` (latent hazard; no output change because every row is overwritten before it is read) |
| M2 ignore the model's plan, execute a fixed one | C2, C3, `test_executing_m_knots_commits_exactly_those_knots` |
| M3 drop the nav patch | C3, `test_c1_is_sensitive...` |
| M4 allowlist includes `act` | C1, `test_model_gets_exactly_the_keys_it_reads` |
| M5 nav never decremented | dead-reckoning hand values, C3 |
| M6 `p10f` not rebuilt from the executed track | C1, C1-sensitivity, C3 |
| leaky state class (reads logged `p10f`/`base`/`pose`/`kin3` back in) | `_leak_guard` returns False (`test_c4_mutation...`), while the clean class returns True |

Real-checkpoint plumbing smoke on the synthetic world (`smoke_out/`, random features, NUMBERS MEAN NOTHING): the CLI runs M2b_s0 at m = 1 / 5 / 30 in 14 / 10 / 9 s for 24 windows on CPU; m=30 `cl` == `ol` ADE.

## 8. Real-data, camera-free measurements — the floor arm on the 141 real eval clips (MEASURED)

The floor needs no image feature, so it runs on the real native tracks (`raw/eval141_kin.npz`, `raw/eval_grid_windows.json`, 2,399 windows) with the frozen P10 inside `M2b_s0`'s checkpoint. Scripts and outputs: `probes/floor_real_probe.py` (+ `.log`, `.json`), `probes/floor_sensitivity_probe.py`.

* **Open-loop equivalence on real data.** m=30 floor ADE6 = 2.48699212 m; the banked `floors_perwindow.npz::P10_refit__ade6` mean = 2.48699212 m (n=2,399). The T1 pipeline reproduces the banked open-loop floor.
* **The floor in its own loop is not stable at short replan periods.**

| m (steps) | replan | floor ADE6 (m) | ADE2 (m) | windows hitting the 60 m/s envelope |
|---|---|---|---|---|
| 30 | one shot | 2.487 | 0.222 | 0 |
| 10 | 0.4 Hz | 3.443 | 0.222 | 178 / 2399 |
| 5 | 1 Hz | 5.407 | 0.274 | 648 |
| 2 | 2.5 Hz | 23.33 | 1.794 | 2,067 |
| 1 | 5 Hz | 21.72 | 3.096 | 2,397 |

* **It is the prior's feedback, not the state rebuild.** Oracle run (execute the LOGGED future, m=1; 5,502 tick-cells): rebuilt-minus-logged |error| median / p90 = v0 0.065 / 0.152 m/s (logged std 7.7), a0 0.055 / 0.202 m/s^2 (std 0.78, corr 0.955), kappa0 0.00025 / 0.0024 (std 0.018, corr 0.89). And the floor's one-shot ADE6 from the rebuilt state is 2.458 m against 2.541 m from the logged state (ratio 0.9675, n=5,895): the rebuild noise does not hurt the map. (`probes/floor_real_fid.json`, `probes/floor_sensitivity.json`.)
* Reading (HYPOTHESIS, not tested): P10 is fitted on the human's state distribution; iterating it on its own output compounds a small early-knot bias and the a0 / v*a features amplify it. The 60 m/s envelope exists precisely so this stays finite and COUNTED.

## 9. What this means, and the next levers (RULE ZERO)

1. **T1-kin is implementable and built.** What it cannot yet deliver is the model-vs-floor T1 number, because the pooled DINOv3 eval cache is not on this machine (`C:/Users/Admin/refav1_hj_cache/eval` absent) and box 2 is busy (`BOX2_BUSY.txt`: "Do NOT launch box 2 jobs").
2. **Operating point.** At m <= 5 the floor itself runs away, so T1 numbers there are runaway-dominated for BOTH arms; the informative sweep for the `H-MINI-K-1` margin is m in {30, 10} (+ 5 as a stress point). m=1 is not a usable headline for this prior.
3. **Cheapest next experiments, in order:** (a) run `v8f_t1_eval.py run` for M2b_s0 / s1 and M1_s0 / s1 at m = 30 / 10 / 5 on the eval cache (measured on CPU with synthetic features: 14 s for 24 windows at m=1, model + floor arms, i.e. ~25 min for 2,399 windows plus cache I/O; faster at larger m) and read `paired_model_minus_floor_t1`: does the residual stabilise or worsen the floor's loop? (b) `--scene frozen` and `--limit off` at m=10 to bracket the two open channels; (c) a feasibility-limited executor (acceleration / jerk / Kamm circle, the `project_speed` envelope) if (a) shows the model is also unstable — a vehicle model is a design decision, so it is NOT added silently.
4. **Escalate:** this tool and its tests need landing (`stack/scripts/v8f_t1_eval.py`, `stack/tests/test_v8f_t1.py`); the stale T1 wording in `taniteval/tools/t1_eval.py` (header, `_tier_doctrine`) should be corrected by its owner.

## 10. UNVERIFIED / not done here

* No local pooled eval cache (`C:/Users/Admin/refav1_hj_cache` has no `eval/`); no real-data run. The real M2b_s0 checkpoint loads and rolls on the synthetic world (shape/plumbing only).
* How much the camera channel and the speed bucket flatter the T1 number is unmeasured; `--scene frozen` and `--limit off` are the instruments, to be run on box 2 / a GPU after the training chain.
* Real measured v / yaw are noisier than the Hermite-smooth executed track: measured for the FLOOR map only (§8: one-shot ADE6 ratio 0.9675 rebuilt vs logged state, rebuilt a0 / kappa0 errors p90 0.20 m/s^2 / 0.0024); for the full model (camera-conditioned) it is unmeasured.
* Perfect tracking of the plan (no actuator lag, no jerk limit, no `project_speed`) is an idealisation. The only bound is the 60 m/s step envelope (`V_ENV_MS`), counted per window in the output header.
* The Hermite construction assumes the plan's heading is the path tangent (true to sideslip).

## 11. How to run it on the box that has the eval cache

```
set CUDA_VISIBLE_DEVICES=            (or a GPU)
python stack/scripts/v8f_t1_eval.py run --ckpt <ckpt_last.pt> --tag M2b_s0_m10 --tracks <native_tracks.pt> ^
   --eval-cache <cache with eval/clips> --eval-grid <eval_grid_windows.json> --eval-join <join_eval141.npz> ^
   --out-dir <dir> --replan-steps 10 [--scene frozen] [--limit off] [--nav logged]
```
Output: `dump_<tag>/ep*.npz` (t1_eval schema: `g`, `cl`, `ol`, `fl`, `ha0`, `eid`, `clip_index`, `v0`) and `t1v8f_<tag>.json` = `t1_eval.analyze` (four families, S-curve, lag, response, paired model-minus-floor and T1-minus-open) plus the `t1v8f` header (tier, m, schedule, scene, nav, limit, keys read, strict flags, open channels, envelope-clipped windows). A `--kin-mode steer` checkpoint is refused.
