# T2 sizing for v8F-mini's M1 crawl read (2026-11-06, tip c59e7b85)

Classes: **M** MEASURED (artifact) · **I** INHERITED (doc, not re-run) · **E** ESTIMATED · **U** UNVERIFIED. Tier proposed: T2 (re-perception rollout: camera follows the car's pose; nothing reacts to the ego; no map).
Paths: `BOX2` = `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-17-t2a-closedloop-box2/`, `PREP` = same folder with `-prep`, `H` = `E:/Projects/TanitAD-artifacts/dagland/motion_c/t2_sizing/`.

## Verdict
* **Stage 1, instrument check on the 2 scenes on hand: feasible on the allocated box 2 with a PI go per job, no download.** The T2-A kit already ran there on 2026-10-17. It cannot decide the lever: one scene holds an M1 situation, once, in one clip.
* **Stage 2, decision-grade read: feasible only with a download (PI approval).** About 42 M1-bearing scenes (E), 26 to 80 GB, 3 to 7 box-2 hours (E). No A40.
* Engineering: 6 to 8 days (E), no PI input.
* `C_STATUS.md` calls T2 "not provisioned". T2-A is provisioned for 2 scenes (M); the gap is scenes and actors, not the box.

## 1. Blockers today
One T2-A run exists: commit `c61c2a6c` (2026-10-17), `BOX2/RESULT.md`, `BOX2/raw/preflight.json`, `BOX2/raw/t2a_run2.log` (M). A search of the tip for later `preflight_box2` / `t2a_run` use finds only `GOALS_AND_CLAIMS.md` row `H-T2A-REFCV7-1`, which cites that run (M).

| # | status | evidence |
|---|---|---|
| B1 box unreachable | resolved | run done (M). Kit still at `C:/Users/Admin/t2a_scratch/box2_kit` (M); box-2 copy `D:\t2a\box2_kit` not re-checked (U) |
| B2 gsplat on Windows | resolved, fragile | 40/40 preflight, RTX 4060, torch 2.11.0+cu128, nvcc 12.8, MSVC 14.38 (M). Two edits sit in box 2's `D:\tanitad\venv` (torch header, gsplat `_backend.py`) (M); still there: U |
| B3 VRAM | resolved | renderer 1.355 GiB; renderer + refcv7 2.127 of 8.00 GiB (M, preflight.json) |
| B4 two scenes | **holds** | section 3 |
| B5 ceiling / nav | ceiling inert for refcv7 (M); **v8F nav and speed bucket open** | section 2 |
| B6 render gap 3.21x | holds (I); gap on frozen DINOv3 tokens never measured (U) | `PREP/BLOCKERS.md` |
| B7 one training seed | resolved for v8F | C0 has s0, s1 (I, H-RI-C0-1). `forward_infer` has no sampler (M, `v8f_mini.py:914`); renderer repeat diff 0 (M): inference-seed floor 0 |
| B8, B9 no collision / off-road; steer clamp | hold, irrelevant to a speed row | actuator also clamps accel at 3 m/s2, 0.5 s time constant (M, `closedloop_drive.py:482`) |
| B10 integration | resolved | `EXTRA_ARMS`, `floors_kin.py` at the tip (M) |
| **B11** new | kit code tree is launch commit `fec3a0dc`, cannot build v8F | tip tree + harness overlay needed (M, `PREP/RESULT.md`) |
| **B12** new | `objects` (actors drawn) never ran on box 2 | banked run was `empty`: no vehicle in any frame (M). Needs `extracted/sequence_tracks.json`, absent from the kit (M) |
| **B13** new | in-loop DINOv3-L | `load_encoder` reads a hardcoded G: Keys path (M, `dinov3_fp8_encode_ship.py:91`); use the local-snapshot route of `pull_encode_pooled.py:473`. Weights on box 2: probable (423 clips encoded there, I, `D3_SCOPING.md`), U now |

## 2. Inputs
v8F-mini reads exactly `feats, frame_ok, p10f, nav, pose_now, pose_t, pose_s, limit, limit_ok, base, a_t, a_s` (M, `v8f_mini.py:169`; `act`, `pose` unread, `DESIGN_T1_V8F.md` section 1).

| input | v8F consumes | T2-A supplies | gap |
|---|---|---|---|
| camera | **one**: front-wide 120 deg, 256x640 cylindrical, frozen DINOv3-L/16 bf16, 16x40 patches pooled to 40 tokens x 1024, 4 frames on a 0.2 s grid (M, `pull_encode_pooled.py`) | front-wide f-theta 1920x1080 at 10 Hz, canon to 416x1024 for refcv7 (M); rig has 5 more cameras, unused | encoder in the loop (B13). 416 to 256 resample exists; pooled residue 2.5 to 2.8 % rel-L2 vs the banked cache (M, `RESULT_S4B.md`); which route built the training cache: U |
| ego | `p10f` (8 kinematic features, yaw rate = one-frame backward difference of the 10 Hz track), `base`, odometry (M, `v8f_mini_data.py`) | `set_history`: 16 rows of (x, y, yaw, v) at 10 Hz (M) | none; executed pose, as in T1-kin |
| nav | token + `d_next/100` + valid | discrete command from the log route, optimistic (M, B5) | no v9 join for the 2 scenes; both junction-free, so FOLLOW with a capped distance. Equivalence: **U** |
| speed bucket | `limit`; v8F's T1 statement uses the **causal** bucket (4 s past speed) (I, PI queue 2026-10-26) | none | `causal` mode exists in `v8f_t1_eval.py` (M); reuse |
| other agents | via pixels | `objects` replays NuRec tracks; `empty` draws none | B12 |
| output | 6 s plan, 30 x 2 | actuator follows the 0.5 s waypoint (accel clamp 3) | T1-kin assumes perfect plan tracking: a playback executor is needed, or the tiers differ in more than perception |

## 3. Scene content
Release: 1,606 to 1,607 scenes, 2.89 TB, about 1.8 GB mean (I, `ALPASIM_STATE.md`, `LOOP_STATE.md:1078`); 79 local scene folders, 2 renderable (M `ls`; I, `CLOSED_LOOP_FEASIBILITY_2026-09-03.md`).

| | scene `00040136` | scene `7c72937c` |
|---|---|---|
| logged speed | 12.1 to 20.4 m/s (M, `PREP/raw/scene_info.json`) | 0 to 15.2 m/s; 71 % of frames at first-second speed >= 5.2 (M, `H/m1_ticks_in_t2a.json`) |
| vehicle in ego corridor, 0 to 40 m | 0 of 202 ticks (M, `H/scene_leads.json`) | 136 of 202 (M); `actor_map.py` documents 135 close-following rows (I) |
| lead at the launch | none | 9.2 to 13.3 m ahead, accelerating 0.9 to 5.7 m/s while the ego goes 0 to 5.2: it pulls away (M, `H/scene2_launch_lead.json`) |
| banked M1-like ticks (logged >= 5.2, own < 5.2), of 450 | refcv7 0, P10 0 | refcv7 s0 62, s1 42, P10 29, P10-held 31; all from starts 34, 51, 102 (M, `H/m1_ticks_in_t2a.json`) |

**Only scene `7c72937c` contains the M1 situation.** The banked images were `empty`, so the lead was never drawn.
**Decision-grade needs** (E, `H/m1_power.json`): the T1-kin interval (+0.0549 [+0.0356, +0.0775], 141 episodes) implies a per-episode SD of about 0.127. Then 42 scene-clusters give 80 % power at a 0.055 effect, 82 give a CI half-width of 0.0275, 168 resolve half the effect. Scene kind: a launch or slow start behind a lead that pulls away, found by screening rig and track logs (about 4 MB per scene, E), plus about 20 non-M1 scenes for the "no other row separated-worse" guard. Drop the 16 corpus-overlap ids (I, `T2_FEASIBILITY.md`).
**Download** (M sizes: volume 0.60 to 0.64 GB, full archive 1.84 to 1.96 GB): 42 scenes = 26 GB by Range pulls of members (I: stored uncompressed, 39/39 verified, `STRATEGIC_FAMILY.md`) or 80 GB as full archives; screening 170 scenes about 0.7 GB (E). Repo access was granted 2026-07-21 (I, `LOOP_STATE.md:1078`) but `PREP/BLOCKERS.md` B4 calls terms acceptance the PI's call: PI approval either way. Box-2 free disk: U.

## 4. Compute on box 2 (RTX 4060, 8 GiB)
* **VRAM:** renderer 1.355 (M) + DINOv3-L bf16 0.57 + v8F 0.03 + activations 0.4 + actors 0.3 (E) = **2.65 GiB, 33 % of the card**.
* **Per 0.2 s step:** 2 renders x 20 ms (M) + DINOv3 29 ms (I, `dinov3_fp8_encode_ship.py`: 2.9 s per ~100-frame episode) + canon/resize 15 + forward 15 (E) = **99 ms** (low 64). refcv7's plan alone took 314 ms per 0.1 s tick (M).
* **Arm-scene (10 starts x 25 steps + 15 s load):** 31 / 40 / 77 s low / central / contended (E). The 2.5x contention factor is measured: one refcv7 arm planned at 314 ms on scene 1 and 800 ms on scene 2 (M, `H/banked_timing.json`).
* **Stage 1:** 7 arms x 2 conditions x 2 scenes = 28 runs: **14 to 36 min (central 19) plus 15 min of gates** (E).
* **Stage 2, 42 scenes x 8 arms = 336 runs:** **2.9 / 3.7 / 7.2 h**; 82 scenes: 5.6 / 7.2 / 14 h (E). Needs an exclusive slot. Helper: `H/sizing_arith.py`, `H/T2_SIZING_V8F.json`.
* The alternative costs about 100+ GPU-h per seed on the A40 (I, PI queue 2026-10-26).

## 5. Engineering and design
| step | files | effort (E) |
|---|---|---|
| E1 `V8FMiniT2Policy`: checkpoint, 4-frame feature deque, items via the T1 tool's `WindowState` / `V8FMiniWindows.item`, HACC post-map, P10 arm from the same forward's `out["p10"]` | new `t2a_v8f_policy.py` | 2 to 3 d |
| E2 in-loop encoder, offline weights, canon, resize to 256x640, pool 4x10 | same file | 1 d |
| E3 playback executor beside the harness actuator | `closedloop_drive.py` hook | 1 d |
| E4 kit: `sequence_tracks.json`, tip tree, actor falsifier on box 2 | `make_kit.py` | 0.5 d |
| E5 scorer: M1 tick share, four families, STRATEGIC PAUSED | `t2a_score.py`; prototype `H/m1_ticks_in_t2a.py` | 1.5 d |
| E6 reachability: exact reproduction of a T1-kin window's plan, camera-blind mutation must move M1, zero control exact 0.0 | tests | 1.5 d |
| E7 stage 2 only: log screening, download script | new | 1 d |

**Cheapest discriminating experiment.**
* **Scenes:** stage 1 on the 2 on hand; stage 2 on about 42 M1 + 20 guard scenes.
* **Arms:** C0 s0, C0 s1, base (which checkpoint: U), P10 prior, zero control, camera-blind control, render-replay control (logged poses and ego: prices the renderer). Deciding: `objects`, playback executor, causal bucket. Beside: `empty`, harness actuator, logged bucket.
* **Metric:** M1 crawl share = ticks with logged first-second speed >= 5.2 and own < 5.2, over ticks with logged >= 5.2 (pin the denominator against the T1 reader before data: U). Four families beside; STRATEGIC PAUSED (PI item 25).
* **Estimator:** paired scene-cluster bootstrap, B = 2000, seed 0: "another draw of scenes" only. Training-seed question: |C0 s1 - C0 s0|.
* **What the interval answers.** Stage 1 (9 starts, 3 carry an M1 tick, starts overlap about 66 %): the banked refcv7 - P10 contrast reads +0.073, CI [0.000, 0.173], 10 % of resamples exactly 0; refcv7's own inference-seed replicate moves the share by -0.044 (M, `H/m1_power.json`). Nothing under about 0.1 is readable, against a T1-kin excess of 0.053 / 0.060 (I, `RESULT_RI_C0.md`). Stage 1 tests plumbing and existence, not the hypothesis.

**Pre-registration outline (DRAFT, not frozen; outcomes and next levers fixed before data).** Question: with the camera re-rendered at the car's own pose and time and the lead replayed, does C0's M1 excess over P10 persist?
* **A, CRAWL-PERSISTS:** at both seeds the CI lower bound > 0, the point estimate >= half the T1-kin reindexed excess (0.027 / 0.030) and > the seed floor. Not a logged-camera artefact. Sub-branch on C0's `objects - empty`: ~0 means frozen tokens do not carry the cue, so the perception lever is justified and the A40 request carries this tier as its evaluation; clearly negative means the cue is used, so train with leads and recovery data first.
* **B, CRAWL-CLOSES:** at both seeds the CI upper bound < half the T1-kin excess. The T1-kin crawl is largely a camera-clock artefact. No A40 on this evidence; T2 becomes the longitudinal tier; next lever is the other D2 rows.
* **Otherwise UNDERPOWERED:** add scenes; not a result.
* **Gates first:** preflight with encoder resident; zero control exact; render-replay C0 against the scene's reference video on pooled-token cosine (the render gap on the frozen encoder, unmeasured); camera-blind arm moves M1.

## 6. Remaining blockers and what unblocks them
1. **Scenes (B4):** 1 M1 clip on hand. Unblock: PI approval of a ~26 GB download (80 GB worst case) and of box-2 disk; screening first.
2. **Box-2 slots:** a PI go per job (stage 1, actor falsifier, stage 2), exclusive use (contention 2.5x).
3. **Engineering E1 to E7:** 6 to 8 days, no PI input.
4. **Open (U):** nav fill for label-free scenes, causal bucket over a 5 s rollout, training-cache route, DINOv3 weights and patched venv still on box 2.
5. **Executor choice** (playback vs actuator): a design call inside the pre-registration.
6. **Interaction stays open:** actors replay, nothing yields; the read says "perception closed", never "safe".
