# POINTING THE CLOSED-LOOP HARNESS AT THE REFERENCE ARMS (2026-09-03)

**Author:** FlyWheel agent (Benchmarks & Evals), 2026-09-03
**Binding doctrine this serves:** PI ruling 2026-09-02 — *"Closed loop means that the trajectory is
controlling the vehicle in the simulation, this can be done e.g. in AlpaSim or in a real test
vehicle."*
**Status:** investigation only. **Nothing was launched, no GPU was used, no training was disturbed.**
Every host probe was read-only (`ls`, `ps`, `tail`, one 64 MiB `dd` write test on Thor's own home).
⛔ **No render was started on Thor.**

---

## ⭐ VERDICT

**We already have a true closed-loop harness, it has already produced published numbers, and the
question is not "can we" but "what does it take to point it at refcv3 and refav1".**
`stack/experiments/alpasim-gsplat/closedloop_drive.py:518-533` steps a kinematic bicycle from the
model's own control and then **renders the next observation from the resulting ego pose**, feeding it
back into `policy.plan` — the PI's definition, satisfied. It has run for real on the Jetson Thor:
9 starts × 50 ticks, **437 paired windows**, all four metric families with episode-cluster bootstrap
CIs (`results/closedloop-hq-render/`). **For refcv3 the gap is ≈1.5 engineer-days and ~1 GPU-hour:**
a geometry swap onto the already-existing `calib.cylindrical_rectify` (the harness canonicalises to
**256×256 f-theta**; refcv3 trains at **256×640 cylindrical**), a thin `RefCV3Policy`, and a
428 MB checkpoint hop. **Three of the five pieces already exist:** the tracking controller
(`wp_to_control`), the 10 Hz re-plan, and — staged by another agent today — a strict,
cross-checked refcv3 loader at `taniteval/tools/refcv3_arm.py:561`. **And the horizon grids already
align**: refcv3's first four waypoints are 0.5/1.0/1.5/2.0 s, exactly the harness's own. **For refav1
the answer is no this week**, on three independent blockers, the worst of which is not engineering:
its action-unit contract is **live and broken** (a steer angle consumed as `kappa`, 2.9× inflated, on
the run training right now). ⛔ **The hard constraint is that Thor is the only renderer host we have
and it is training refav1 until ≈00:50 UTC** (MEASURED: step 15650/21109 at 3.500 s/step ⇒ 5.31 h
from 19:30 UTC — an independent arithmetic that lands on the coordinator's figure). And the sentence
that must travel with every number this harness ever produces: **the loop is closed on PERCEPTION,
not on INTERACTION** — see §4, which is the most important section of this document.

### ⚠️ Two corrections I have to make before anything else

1. ⛔ **"ADE does not separate at all" is the RETRACTED reading, and it was repeated to me in my own
   brief.** `PROGRAM_OVERVIEW.md:45` carries the retraction **R-2026-08-03-C**; I verified it against
   the primary artifact rather than the prose. On the **current** panel
   (`results/closedloop-hq-render/HQ_flagship_vs_refc_empty.json`, n = 437) **`ade_0_2s` = +7.1642
   [+5.2654, +8.9661] — it separates**, as do both longitudinal metrics and strategic corridor
   departure (+0.5057 [+0.3822, +0.6289]). The "ADE saw nothing / entirely lateral" table is the
   **pre-retraction** panel (`results/metrics_empty.json`, ADE +0.7885 [−0.8653, +2.7282]). The
   mechanism is stated at `PROGRAM_OVERVIEW.md:45`: under the improved render **flagship v1's driven
   path moves 9.05 m mean while REF-C's moves 0.43 m (21×)**.
   ⚠️ **`PROGRAM_OVERVIEW.md` §5.0.1 still prints the superseded table** while line 45 carries the
   retraction, so the same document says both things. **That is a doc fix somebody owes.**
2. ⛔ **The video path cited in `PROGRAM_OVERVIEW.md` §5.0.1 does not exist.** It says
   `TanitAD Research Lab/Evaluation/Videos/alpasim-closedloop/`; that directory has **0 tracked
   files**. The real location is
   **`TanitAD Research Lab/Benchmarks & Evals/_evaluation/Videos/alpasim-closedloop-thor-2026-08-03/`**
   (plus an `-archive-2026-07-22/` sibling), under a tree with 200 tracked files. Probe: `git ls-files`
   on the index, both spellings. This is almost certainly Research-Hub→Research-Lab rename drift.

---

## 1 · WHAT THE HARNESS IS, AND WHERE ITS SEAMS ARE

### 1.1 · Entry point

**`stack/experiments/alpasim-gsplat/closedloop_drive.py`** (35 705 B, 712 lines). CLI at `:548`:

```
python closedloop_drive.py --scene-dir <NuRec scene> --arm <name> --ckpt <path>
       --condition empty|objects --starts 0,17,34,… --steps 50 --out <dir>
       [--addr host:port]            # gRPC renderer; omit for in-process
       [--all-dynamic-layers] [--cull-scale-quantile 0.95] [--sky-gain 0.3] [--rolling-shutter]
```

Runner: `run_panel_hq.sh` (the protocol the current numbers were produced under). Scoring:
`cl_metrics.py` (60 461 B, 1 055 lines). Matched open-loop control arm: `openloop_drive.py`.
Renderer: `gsplat_renderer.py` (50 118 B, 1 011 lines). AlpaSim wire server:
`sensorsim_gsplat_server.py`.

### 1.2 · The model-loading seam — exactly three things to touch

| # | seam | file:line | what it does |
|---|---|---|---|
| **S1** | `_BasePolicy` | `closedloop_drive.py:181-213` | Base class. Owns `canon()` — the **canonicalisation**, which is where the geometry is fixed and where the `f_eff` self-check lives. Declares `consumes_nav_known`. |
| **S2** | the arm's policy class | `FlagshipV1Policy` `:218-256`, `RefCPolicy` `:259-291` | Constructor loads the checkpoint; `plan(frames, intr, v0, nav_cmd, nav_known) -> (traj [4,2] float64, extra dict)`. **That signature is the entire model interface.** |
| **S3** | arm registration | `:551-552` (`choices=[…]`), `:660-663` (the dispatch) | Adds the `--arm` name. |

**The interface a new arm must satisfy, in full:**
* **IN:** `frames` — a list of ≥ `NEED_FRAMES = WINDOW + STACK − 1 = 10` native renderer images
  (`:44-48`); `intr` — the renderer's f-theta intrinsics; `v0` — ego speed **m/s**; `nav_cmd` — int
  in `("follow","left","right","straight")` (`:53`); `nav_known` — optional companion bit, fed **only
  if** the class sets `consumes_nav_known` (`:187-192, :492`).
* **OUT:** `traj` — **[4, 2] ego-frame waypoints in metres, x forward / y left**, at
  **0.5 / 1.0 / 1.5 / 2.0 s** (`HORIZON_S`/`WP_STEPS`, `:50-52`), plus an `extra` dict of any ≤16-wide
  vectors, which the driver banks verbatim into the rollout JSON.
* The loop then does everything else: `wp_to_control` → bicycle → `GroundFollower` → re-render.

⇒ **To accept a new checkpoint the harness needs a ~40-line subclass and one line in `choices`.**
The expensive part is not the seam; it is that **refcv3 and refav1 are not 256×256 f-theta models.**

### 1.3 · Where the loop actually closes

`closedloop_drive.py:518-533`:

```python
dyaw  = v / WHEELBASE * math.tan(steer) * DT     # :519   the model's own control
T_new = T_ego @ D                                # :522
T_ego, _ = gf.correct(T_new)                     # :523   ground-follow (harness choice)
img = transport.render(T_ego @ Ts_cam, t_us, …)  # :529 ⭐ next observation from THAT pose
frames.append(img)                               # :532   fed back to policy.plan
```

Rate **10 Hz** (`DT = 0.1`, `:46`); re-plan **every tick** (`policy.plan` inside `for k in
range(n_steps)`, `:485-491`).

---

## 2 · WHAT IT TAKES TO DRIVE refcv3 — 5 gaps, ≈1.5 engineer-days (G1 ~1 d · G2 ~2 h · G3 0 · G4 ~1 h · G5 0)

refcv3 is one-shot: `RefCV3Model.forward(frames, nav_cmd, v0, steps, lan, nav_known)` →
`out["traj"] [B,S,2]` (`stack/tanitad/refs/refc_v3.py:480`, `:528`). No action input, no rollout, no
per-step decode.

| # | gap | effort | detail |
|---|---|---|---|
| **G1** | **Geometry — the only real work.** `_BasePolicy.canon` builds `[1,8,9,256,256]` via `ftheta_crop_resize(vid, intr, 256, center="principal")` and **refuses** unless `\|f_eff − 266.0\| < 8` (`closedloop_drive.py:199-213`; `calib.py:38 F_REF = 266.0`). refcv3 trains at **`--image-hw 256 640`** (MEASURED off the live A40 process). | **~1 day** | Swap to **`calib.cylindrical_rectify`** (`stack/tanitad/data/calib.py:960`) with **`PHYSICALAI_WIDE120_256x640 = CanonicalFrame(height=256, width=640, f_ref=305.5774907364391, projection="cylindrical")`** (`calib.py:1247-1248`). Re-point the self-check at `cylindrical_rectify.last_f_eff` (≈ 305.577) and **also print `last_observed_frac`**. ⚠️ **`cylindrical_rectify` requires a PER-CLIP principal point and raises without one** (`calib.py:981`); the renderer serves one — MEASURED in `results/contract_test.json`, the rig is **1920×1080 f-theta, cx 958.4379, cy 751.1790**, reached via `GrpcTransport._fetch_camera` (`:110-118`). **Wire it; do not default it.** **Keep the refuse-on-mismatch behaviour**: a silent geometry mismatch is exactly what that assertion exists to stop. ⭐ The rig is `camera_front_wide_120fov` and the target frame is 120° cylindrical, so `last_observed_frac` should come back **≈1.0** — if it does not, the crop box is wrong and the run must be refused, not scaled. |
| **G2** | **Policy class.** `RefCPolicy` loads REF-C **v1.2** via `from refc_v12_cache import load_frozen` (`:263`); `--arm` offers only `flagship-v1 / refc-base / refc-xl` (`:551-552`). | **~2 h — smaller than it looks** | ⭐ **The hard half already exists and is staged right now**: `taniteval/tools/refcv3_arm.py` (2 127 lines) has **`load_model(ckpt_path, config_path, device, allow_nonstrict) -> (model, cfg, train_args, provenance)`** at **`:561`**, with `rebuild_config` (`:485`) and `cross_check_config` (`:527`) verifying `arm` / `image_hw` / `tac_vocab_version` / `horizons` against the checkpoint's own `config.json`, and a **STRICT** load that refuses on any shape or key disagreement (*"fix the config, never the weights"*). ⇒ `RefCV3Policy` is **`from refcv3_arm import load_model`** plus a `plan()` wrapper around `RefCPolicy.plan` (`:277-291`) — whose existing call `self.model(fw, nav_cmd=navt, v0=v0t, steps=2, nav_known=nk)` differs from refcv3's forward by **one argument (`lan`)**. ⚠️ That file is another agent's live staged work — **coordinate before editing it**; importing it is free, forking it is not. |
| **G3** | **Tracking controller + re-plan rate.** | **0 days — and the grids ALIGN** | `wp_to_control` (`closedloop_drive.py:316-328`) is the programme's published pure-pursuit + P-speed: `κ = 2y/l_d²`, `steer = clip(atan(L·κ), ±0.05)`, `accel = clip((v_target − v)/0.5, ±3.0)`, `l_d²` floored at 0.25. Re-plan is already every tick at 10 Hz. ⭐ **And the horizon grids match** — MEASURED from the refcv3 fixture manifest (`…/2026-09-03-refcv3-arm/raw/fixture_dump/manifest.json`): refcv3's `horizons` are **[5, 10, 15, 20, 30, 40, 50, 60]** steps at 10 Hz = 0.5/1.0/1.5/2.0/**3.0/4.0/5.0/6.0** s, whose **first four are exactly** the harness's `WP_STEPS = (5, 10, 15, 20)` / `HORIZON_S = (0.5, 1.0, 1.5, 2.0)`. So `LOOKAHEAD_IDX = 0` **is** the 0.5 s waypoint for refcv3 too. ⚠️ **Assert it, do not assume it** — read the loaded `cfg`'s horizons and fail loudly if the first four are not `(5,10,15,20)`; a checkpoint built with a different grid would otherwise steer to the wrong horizon in silence. Bonus: refcv3 reaches **6 s**, so the harness could score a longer horizon than it currently does. |
| **G4** | **Checkpoint transfer.** No refcv3 on Thor (`/home/nvidia/models/` = flagship-v1-speedjerk, flagship-v4.2b, refc-base, refc-base-e1b-clsft, refc-xl, rollout-recovery, v5f). | **~1 h** | `tanitad-a40:/workspace/experiments/refcv3-b1-v72-30k/ckpt_30000.pt` = **428 519 790 B**, mtime 2026-09-03 11:04 — **already on disk, no need to wait for training**. ⚠️ Direct pod→Thor SSH is **UNVERIFIED**; probe `ssh -n … 'echo OK'` on the source's own `$RUNPOD_PUBLIC_IP:$RUNPOD_TCP_PORT_22` before building a transfer on it, and fall back to the HF relay. **md5 both ends.** |
| **G5** | **Wheelbase** — see §5. | **0 days (declare it)** | refcv3 takes no action input, so the harness's 2.7 m cancels exactly inside its own integrator. **State it; do not change it silently.** |

**Also required, and it is not optional:** a **leak check** of the NuRec scene against the
`physicalai-b1-w120-256x640cyl` episode list — see §7 item 2. It has not been done.

---

## 3 · WHAT IT TAKES TO DRIVE refav1 — 6 gaps, ≥1–2 weeks, one of them a PI decision

refav1's gap is the **inverse** of refcv3's: `plan()` emits **controls**, so no tracking controller is
needed — but it needs a **live encoder** and its planner is an **MPC**.

| # | gap | effort | detail |
|---|---|---|---|
| **A1** | ⛔ **The action-unit contract is live and broken.** `stack/tanitad/data/refav1_loader.py:252` reads `o["actions"][:, 0]` and `:264` emits it **as `kap`** — but `physicalai.py:620` wrote a **steer angle** `arctan(2.9·κ)` into that channel. Every curvature is inflated ~2.9×, and **that is the training input of the run on Thor right now.** | **PI DECISION — not an engineering estimate** | `…/Benchmarks & Evals/Research/2026-09-03-refav1-kinematic-contract-lateral/RESULT.md` (MEASURED, n = 140 windows / 20 episodes, episode-cluster bootstrap). Three incompatible conventions are live at once. ⛔ **A closed-loop number produced before this is settled banks a 2.9×-wrong lateral actuation.** |
| **A2** | **DINOv3 must run INSIDE the loop.** `RefAV1.plan(feats, …)` (`refa_v1.py:1925`) needs ViT-L/16 patch tokens — `d_enc 1024`, `n_tokens 640`, grid 16×40, 120° HFOV, 256×640, **no CLS** (`refa_v1.py:75-84`). The programme only ever computes these **offline**. | **~1 day** | New per-tick stage: render → cylindrical 256×640 → the processor's ImageNet mean/std (`dinov3_fp8_encode_ship.py:99`) → DINOv3 forward → `[1, 640, 1024]`. ⚠️ The banked cache is **fp8**; a live bf16/fp16 encode is a **different numeric path** — it needs a matched-frame agreement check against the cache before any number is quoted. |
| **A3** | **DINOv3 weights are not on Thor.** | **~1 h + 1.21 GB** | Two probes, different path-binding: (a) `find /home/nvidia -maxdepth 5 -iname "*dinov3*" -not -path "*/data/*"` → **empty**; (b) `ls ~/.cache/huggingface/hub` → 7 repos, **no** `models--facebook--dinov3-*`. Only the derived caches: `dinov3-b1-fp8-w120-256x640cyl` **291 G**, `dinov3-val600-…` **37 G**. The weights **are** on the dev box — `model.safetensors` **1 212 559 808 B**, verified by size not presence. Model is **gated** (`dinov3_fp8_encode_ship.py:61,:99-100`) ⇒ token needed. **Read it in place; never print it or pass it as an argument.** |
| **A4** | **MPC cost is unmeasured.** `PlanConfig(n_samples=300, n_iters=30, n_elites=30, decay=1.25, min_samples=32, cost_chunk=64)` (`refa_v1_plan.py:79-96`). | ⛔ **UNVERIFIED — must be measured before it is priced** | **ESTIMATED** candidates/tick: `Σ_{i=0}^{29} max(300/1.25^i, 32)` = 1 371.15 (i ≤ 10) + 19×32 = **≈1 979**, each `horizon = 10` steps through the tactical predictor ⇒ ≈31 batched passes/tick; ≈**99 000** candidate rollouts per 50-tick rollout. Feasible in principle — **but per-tick wall-clock has never been timed and could turn a 5 s rollout into minutes.** |
| **A5** | **The planner→model crossing must be DECLARED.** `model_action_units` defaults to `"kappa"`, which `refa_v1.py:1938-1946` documents as **wrong** (a `GOAL_KAPPA_TURN = 0.08` candidate, R 12.5 m, is imagined by the model at R ≈ 36 m). `cost_time_grid` and `goal_time_grid` carry the same "legacy default is wrong, kept for reproducibility" warning. | **~0.5 day** | Pass `model_action_units="steer"` (converts at `STEER_WHEELBASE_M = 2.9` — correct) and **state all three grid choices in the result**. |
| **A6** | **Controls → vehicle.** `PlanResult.controls [H,2]` is `(a, kappa)` on a **0.2 s** grid (`refa_v1_plan.py:113`, `PlanConfig.dt = 0.2`); the harness bicycle steps at **0.1 s** through a *steer* integrator. | **~0.5 day** | Integrate refav1's **κ directly** (`dyaw = v·κ·dt`, which needs no wheelbase at all) rather than round-tripping through the harness's 2.7 m steer path, and re-sample 0.2 s → 0.1 s explicitly. Plus **G1** (same geometry swap) and the same Thor scheduling constraint. |

---

## 4 · ⭐⭐ THE HONEST SCOPE LIMIT — the sentence that must travel with every number

> **The loop is closed on PERCEPTION, not on INTERACTION. The ego's own trajectory determines every
> future observation — that is real, and it is what the PI asked for — but *nothing in the scene
> responds to the ego*. Other agents are replayed or scripted, there is no map and no drivable-area
> polygon, and the vehicle model has no collision response. ⇒ The harness measures HOW WELL THE MODEL
> DRIVES ITSELF. It cannot measure WHAT HAPPENS WHEN IT DRIVES BADLY.**

That is the whole distinction, and it is exactly the phase-0 "safety-grade half" gap. Concretely:

**✅ CAN produce — paired, all four families, episode-cluster bootstrap CIs**
(keys verified in `results/closedloop-hq-render/HQ_flagship_vs_refc_empty.json`, n = 437):

| family | metrics it really emits |
|---|---|
| **ADE** | `ade_0_2s`, `lateral_ade_m` |
| **LONGITUDINAL** | `abs_target_speed_err_ms`, `along_track_ade_m`, `executed_speed_err_ms`, `abs_executed_speed_err_ms`, `real_lead_headway_m`, `real_lead_time_gap_s`, `real_lead_frac_tg_below_1s`, `real_lead_ttc_s_when_closing` |
| **LATERAL** | `cross_track_abs_m` (= `dist_to_gt_traj_m`), `heading_err_rad`, `curvature_err_1pm`, `yawrate_err_rads` |
| **TACTICAL** | `manoeuvre_plan_eq_logged`, `manoeuvre_head_eq_logged`, `manoeuvre_exec_eq_plan`, `confusion_planned_x_executed`, `head_class_share` |
| **STRATEGIC** | `route_head_eq_logged`, `route_corridor_departure_rate`, `route_label_valid_rate` |

**⛔ CANNOT produce — and must never be implied**

| not available | why, from source |
|---|---|
| **Off-road / drivable-area departure rate** | There is **no map in the harness at all** — `grep -iE "xodr\|off_road\|offroad\|drivable"` over `closedloop_drive.py`, `cl_metrics.py`, `actor_map.py` returns **zero hits**. (NuRec scenes ship a `map.xodr`, and a research package read it — but **it is not wired into this harness**.) `route_corridor_departure_rate` is a **corridor around the logged path**, not a lane or a road edge — verified at source: `cl_metrics.py:386` is literally `int(abs(ct) > CORRIDOR_M)` on the **cross-track distance to the logged trajectory**. A model that departs the corridor may be perfectly on-road, and one that stays inside it may be off-road; the metric cannot tell. |
| **A real collision rate** | The only collision metric is `synth_lead_collision_rate_inlane` (`cl_metrics.py:634`) — a **geometric headway ≤ 0 test, lane-gated**, against a **scripted** lead. Its ungated sibling is literally named `synth_lead_collision_rate_UNGATED_DO_NOT_QUOTE` (`:632`) because 13 of flagship-v1/cutin's "collisions" had **median \|y\| 13.795 m**, i.e. 0 % in-lane (`:409-414`). The bicycle has **no collision response**: it drives straight through everything. |
| **Anything involving a REACTIVE agent** | Actors are **replayed** from the log or **synthesised** on a fixed script (`actor_map.py`, `synth_actor.py`). Nothing brakes, yields, or swerves because of us. So no right-of-way, no negotiation, no merge, no induced-collision measurement. |
| **Absolute rates of any kind** | ⛔ **WITHIN-SIM RELATIVE**, stated in the artifact's own `within_sim_note`: REF-C open-loop ADE **1.5157** on these reconstructions vs **0.4728** on real footage — **3.21× OOD**. Orderings survive; absolute rates do not. |
| **Strategic accuracy** | **Degenerate on a junction-free 20 s clip**: flagship `route_head_eq_logged` 1.0000 is a **constant-predictor tie**, `route_label_valid_rate` 0.3778/0.3867, the rest gray-zone. **A junction scene is required before any strategic-accuracy claim.** |
| **A 40-episode-style interval** | The bootstrap clusters are **disjoint segments of ONE clip**, not independent episodes — stated in the JSON's own `estimator_note` so it is never confused with the n=40 val bootstrap. |
| **`synth_lead_*` on the empty-road panel** | `n = 0`, reason `"no jointly finite windows"` — reported as UNAVAILABLE with its reason, per the four-families rule, never dropped. |

⚠️ **And one more honest bound, from the harness's own docstring** (`closedloop_drive.py:25-28`):
the bicycle is **planar** and the z/roll/pitch ground-following is a **harness choice, not a physics
engine**. Say so in any write-up.

---

## 5 · THE STEER/CURVATURE CONTRACT — where the conversion must live

**The contract (MEASURED, and it holds).** `stack/tanitad/data/physicalai.py:620` writes
`steer = np.arctan(float(wheelbase) * curv)` into `actions[:,0]` with `physicalai.py:63
WHEELBASE = 2.9`. `stack/tanitad/models/kinematic.py:36-51` states the rule explicitly —
***`STEER_WHEELBASE_M` is an ENCODING constant, not a vehicle property*** — recovered by inverting
the producer at **L_enc = 2.9000000 (spread 3e-8)**, against true per-clip wheelbases
{2.73, 3.135, 3.165, 3.216}, **none of which is 2.9**. Using a real wheelbase would *inject* an error
the encoding never had.

**Where it must live: at the MODEL BOUNDARY and nowhere else.** Today that is
`kinematic.kappa_of_steer` / `steer_of_kappa` (`kinematic.py:57-65`), `refa_v1_plan.py:294`, and
`RefAV1.plan(model_action_units="steer") → _model_actions` — all defaulting to **2.9**, all correct.

**⛔ Where a different constant is in use — flagged, not changed:**

| site | constant | binds a model? | verdict |
|---|---|---|---|
| `taniteval/taniteval/closedloop.py:118` | **2.7** | **YES** — `build_action` (`:188`) feeds `steer` to the predictor | ⚠️ **Train/serve skew, already documented and priced at `:104-118`**: +7.41 % action skew; open-loop cost **ΔADE +0.0026 [−0.0006, +0.0062]** (paired episode-cluster bootstrap) — **not separated from zero**. Escalated as a PI decision 2026-07-26 and **still open**. |
| `stack/experiments/alpasim-gsplat/closedloop_drive.py:316` | **2.7** | **NO for refcv3**; **would for refav1** | For refcv3 it is harmless and **cancels exactly**: `wp_to_control`'s `atan(L·κ)` and the bicycle's `v/L·tan(steer)` use the **same** L, so **no closed-loop path number in the programme is wrong because of 2.7**. For refav1, do not round-trip — integrate κ directly (**A6**). |
| `stack/tanitad/data/refav1_loader.py:252,:264` | — (a **unit** error, not a constant error) | **YES** | ⛔ The live defect **A1**. |

**⚠️ Why the earlier probe cleared A1 and today's did not — the general lesson.**
`refav1_loader.py:17-23` **still carries** a MEASURED claim that `actions[:, 0]` "correlates
r = 0.995 with pose-derived curvature ⇒ the STORED order is (kappa, accel-like)". The correlation is
real and the conclusion is wrong: at our steering magnitudes `atan(2.9·κ) ≈ 2.9·κ`, so steer and
curvature are **collinear to r > 0.999** and a correlation **cannot** separate them — the information
lives entirely in the **slope**, which reads ≈ L. Today's package settles it by **algebraic inversion
against the producer's own input** instead (that RESULT's §B4 names exactly this). ⇒ **A correlation
is not a units check.** Same family as the `df` / Thor `free` / cgroup `usage_in_bytes` traps: a probe
that answers a *different question* than the one asked, read as an answer. ⛔ **The corrected
docstring is PROPOSED, not applied** — the refuted claim is still in the working tree today.

**Recommendation (escalated, not applied):** leave both 2.7 constants alone **for now**. Changing them
is a one-line, no-retrain fix, but it **moves every closed-loop number in the programme** and breaks
comparability with the published n = 12 suites (`taniteval/taniteval/closedloop.py:114-117`). Change
them **once**, in the same turn as the A1 unit repair, under one PI decision, and re-run the affected
suites — not piecemeal inside a new harness.

---

## 6 · COMPUTE — the hard constraint, and what the alternatives cost

⛔ **Thor renders, and Thor is training.** `nvidia-smi` on `tanitad-thor-wifi` reads
`NVIDIA Thor, [N/A], [N/A], 97 %` (the memory columns are the documented Thor probe trap; only
`torch.cuda.max_memory_allocated()` is admissible there), and `ps` shows
`refa_v1_train.py … --out /home/nvidia/experiments/refav1-b1-v72-ep3-speed`. **No GPU load was added
and none should be.**

**ETA, MEASURED and independently arrived at:** `train_log.jsonl` last row is step **15650** of
**21109** at `elapsed_s` **54 778.7** ⇒ **3.5003 s/step**; 5 459 steps remain ⇒ **19 108 s = 5.31 h**
from 19:30 UTC ⇒ **≈00:48 UTC**. That lands on the coordinator's independently-derived ≈00:50 UTC,
which is the kind of agreement that makes a number trustworthy.

**Renderer host options:**

| host | verdict | evidence / cost |
|---|---|---|
| **Thor** ⭐ | **The only ready host.** gsplat 1.5.3 + torch 2.13.0+cu130, CUDA available; `alpasim` 1.6 G; `nurec_scenes` 7.8 G; `tanitad_cl` 14 M code checkout; `dd` write 1.9 GB/s. | **Free at ≈00:48 UTC.** A refcv3 panel is **~1 GPU-hour** (a 50-tick rollout MEASURED at **5.577 s wall / 38.08 ms per render**; generous headroom allowed because 256×640 is 6.25× the pixels of 256×256). **Cost: zero — hardware we own.** |
| **`tanitad-a40`** (69.30.85.211) | ❌ **Not viable now.** | Alive, but it **is** the refcv3 trainer: A40 **44 729 / 46 068 MiB** used. And it carries **no AlpaSim, no NuRec, no gsplat** (two path-bindings: name glob → empty, and a full `du -sh /workspace/*` listing showing only training assets). Would need the 640 MB volume + a gsplat build + the 14 MB code — on a GPU with no room. |
| **`tanitad-eval`** | ❌ **DEAD.** | `ssh` → `connect to host 69.30.85.106 port 22073: Connection refused`. This was the A40 that ran the 2026-07-22 AlpaSim panel. The old note is stale. |
| **Dev box (RTX 4060, 8 GB)** | ⚠️ **UNVERIFIED, probably not.** | Would need a gsplat CUDA-extension build on Windows (this box has no Triton and torch.compile is unusable here — a related but not identical blocker), plus a 640 MB scene download. 3.1 M gaussians in 8 GB is plausible but **untested**. Do not plan on it without a build probe. |
| **A new rented x86 GPU** | ⛔ **PI DECISION — priced and STOPPED.** | The only path that *requires* it is AlpaSim's native runtime (route B), whose stock `nre-ga` renderer is a **closed, amd64-only container** (`gsplat_renderer.py:5-7`) and therefore cannot run on Thor's aarch64. Marginal need is **~1 GPU-hour on an A40-class box** — order-of-$1 at the rates this programme has historically used, **ESTIMATED and not checked against a current price list**. **Nothing was provisioned, nothing was bought, and this document does not request it** — Thor freeing in ~5 h makes it unnecessary. |

**Other costs:** scene assets already resident (7.8 G); a panel writes ~10 MB of JSON; checkpoint hop
428 MB (~1 h with md5). **HF quota is untouched** if the checkpoint moves by direct SSH — if the HF
relay is used instead, **check remaining quota first**; it is a hard ceiling. **Licences: none
blocking** — AlpaSim source is resident with its `LICENSE`, `volume.nurec` is open gzip+MessagePack,
gsplat 1.5.3 is installed, and research-licensed assets are sanctioned for this programme.

⚠️ **The real ceiling is not compute — it is scenes.** Only **2 renderable NuRec volumes exist**:
`00040136-…/volume.nurec` **639 393 135 B** and `7c72937c-…/extracted/volume.nurec` **603 238 216 B**.
A third `volume.nurec` is a **110-byte stub**, and ~78 of the 80 scene directories hold only a
`camera_front_wide_120fov.mp4`. **n = 2 scenes, one of them junction-free**, bounds the tactical and
strategic families hard. Acquiring more NuRec volumes is the highest-leverage item here, and it is a
provisioning decision — therefore the PI's.

---

## 7 · ARE THE 2026-08-03 NUMBERS STILL QUOTABLE?

**Yes for flagship v1 vs REF-C base — with four conditions, and only from the right panel.**

1. ⛔ **Quote `results/closedloop-hq-render/`, NOT `results/metrics_empty.json`.** The morning panel
   was measured on `background,road` with no sky; the render-quality pass then raised grad-NCC
   0.2774 → 0.3424 (+23.4 %) and the shipped videos were re-rendered with it. **Changing the render
   changes what the policy sees, so the old driving numbers do not transfer** — `run_panel_hq.sh`
   exists precisely to say so. The retraction **R-2026-08-03-C** is the consequence.
2. ✅ **The panel carries its own controls, and they hold — verified against the primary artifacts,
   not the prose.** `CONTROL_refc-base_repro_vs_morning.json`: `ade_0_2s`, `cross_track_abs_m`,
   `heading_err_rad`, `abs_target_speed_err_ms` **all read `delta = 0.0` with a ZERO-WIDTH CI
   [0.0, 0.0]** over **437 windows / 9 episodes**, `paired_episode_cluster_bootstrap`, 2 000
   resamples. *(PROGRAM_OVERVIEW says "450/450"; both are right — 9 × 50 = 450 raw ticks, 437 paired
   windows after the warm-up. Quote the denominator you mean.)* Also present:
   `CONTROL_*_objmorn_vs_morning`, `ABLATE_sky_*`, `ABLATE_cull_*`, `RENDER_AB_empty.json`.
   `contract_test.json` confirms the wire contract — `get_version` `tanitad-gsplat-sensorsim-1.0`,
   `camera_param_kind` `ftheta_param`, **`negative_control_pass = True`** (grad-NCC **0.3664** on the
   correct frame vs **0.2052** best wrong, margin 0.1612), `canon_pass = True`.
   ⚠️ Its top-level `ALL_PASS` is **False** — solely because `transport_bit_identical` is False, which
   §5.0.1 explains is **expected**: the renderer is a step function of pose, and *"bit-identical is
   the wrong acceptance criterion for a splat renderer."* **Do not read `ALL_PASS: false` as a broken
   harness.**
3. ⚠️ **Every bound in §4 travels with them** — within-sim relative (3.21× OOD), one-clip clusters,
   degenerate strategic, no reactive agents.
4. ⚠️ **`manoeuvre_exec_eq_plan` is NOT comparable across these two arms.** REF-C's executed class is
   `lane_keep` on **270/270** windows, so the metric degenerates to *"how often was the plan
   lane_keep"* = 236/270 = **0.8741 exactly** — a **constant-predictor tie**, not evidence REF-C
   executes well. Verified against `confusion_planned_x_executed` in the primary artifact.

**Nothing since 2026-08-03 invalidates them.** Checked: the `overlapping_holdout_se` correction is
**2026-07-25**, i.e. *before* the panel, and the panel already uses the episode-cluster bootstrap
(its own `estimator_note` says so). The 2.7 m wheelbase **cancels** for these arms (§5). The refav1
unit defect (**A1**) does not touch flagship v1 or REF-C base.

⛔ **But two DOC-level defects must be fixed before anyone quotes §5.0.1 from the prose:** the
superseded ADE table still printed there, and the non-existent video path — §Verdict, corrections 1
and 2. **Quote the JSON, not the section.**

---

## 8 · THE CHEAPEST EXPERIMENT THAT PRODUCES ONE HONEST CLOSED-LOOP NUMBER FOR ONE ARM

**Arm:** refcv3 (`ckpt_30000.pt`, already on disk). **Host:** Thor, **after ≈00:50 UTC**.
**Cost:** ≈1.5 engineer-days of code + **≈1 GPU-hour**. **No purchase, no new hardware.**

1. **G1 — port the canonicalisation** to `cylindrical_rectify` + `PHYSICALAI_WIDE120_256x640`, fed the
   renderer's per-clip principal point; assert `last_f_eff ≈ 305.577` and print
   `last_observed_frac`; keep refuse-on-mismatch.
2. **G2 — add `RefCV3Policy`** by **importing `refcv3_arm.load_model`** (do not fork that file — it
   is another agent's live staged work) + `--arm refcv3`; **G3 — assert** that the loaded config's
   first four horizons are `(5, 10, 15, 20)` rather than assuming it, then keep `LOOKAHEAD_IDX = 0`.
3. **G4 — ship the checkpoint**, md5 both ends, probing the direct SSH mapping first.
4. **Run `run_panel_hq.sh`'s exact protocol** on scene `00040136-…`, `--condition empty`,
   **9 starts × 50 ticks**, same render flags (`--cull-scale-quantile 0.95 --sky-gain 0.3`, rolling
   shutter OFF) — so the contrast is a clean A/B against the banked `HQ_refc-base_empty`.
   **⇒ THE ONE NUMBER: paired closed-loop `ade_0_2s` and `route_corridor_departure_rate` for refcv3
   vs refc-base on identical pixels**, paired episode-cluster bootstrap — never
   `overlapping_holdout_se`.
5. **Report all four families** with per-family `n`, `UNAVAILABLE + reason` where `n = 0`, the tier
   stamp, and the §4 scope sentence in the same paragraph as the headline.

**⛔ The positive control that makes it admissible.** Re-run **`refc-base` unchanged** through the
**new** cylindrical path *and* the **old** 256×256 path. **If refc-base's number moves between them,
the contrast is measuring the geometry swap, not refcv3.** Without this control the experiment
changes two things and attributes the difference to one — the exact failure the ridge-probe rule
exists to prevent, and the exact mechanism behind R-2026-08-03-C.

**Pre-registered falsifier.** If refcv3 is **not** distinguishable from refc-base under the paired
bootstrap, the honest verdict is *"the arms are indistinguishable in closed loop on n = 2 scenes"* —
**not** *"refcv3 is better but underpowered"*.

---

## 9 · WHAT I COULD NOT VERIFY (unverified, not assumed)

1. **Per-tick latency of `RefAV1.plan`** — never timed anywhere I could find. The ≈1 979
   candidates/tick figure is **ESTIMATED arithmetic** from `PlanConfig` defaults, not a measurement.
2. ⛔ **Whether the 2 NuRec scenes overlap the training corpus.** Neither UUID appears in
   `alpamayo_IN_parity_train_EXCLUDE_FROM_EVAL.txt` (201 clips) — **but neither appears in
   `alpamayo_clip_ids.txt` (4 729 clips) either**, so the id namespaces are simply different and the
   check proves **nothing**. **A leak check against the actual `physicalai-b1-w120-256x640cyl` episode
   list is REQUIRED before any refcv3/refav1 closed-loop number is quoted.** *(It is not required for
   re-quoting the flagship/REF-C panel, whose bounds are already published.)*
3. **Direct A40 → Thor SSH.** Untested here; the env-advertised mapping has been dead on healthy pods
   before.
4. **Whether AlpaSim's own runtime (route B) has ever run end-to-end.** `drive_smoke.py` proves only
   the *plugin contract* — by its own docstring, *"Not a sim run"*. Thor's `alpasim_setup2.log` ends
   mid-setup at `sourcing setup_local_env.sh` with no success marker. **Treat route B as unproven.**
5. **Whether a live bf16/fp16 DINOv3 encode matches the banked fp8 cache** numerically. Unmeasured;
   **A2** requires an agreement check.
6. **Exact current refcv3 training step.** I limited the probe to a directory listing because the pod
   is training. `ckpt_30000.pt` proves ≥ 30 000 of 40 284 steps.
7. **Whether the 640 MB volume renders well at 256×640 without re-tuning** `--cull-scale-quantile` /
   `--sky-gain`, which were tuned for the 256×256 crop. If they need re-tuning, the §8 positive
   control becomes **mandatory rather than merely good practice**.
8. **Whether `map.xodr` could be wired in cheaply** to unlock off-road rate. The file exists in the
   NuRec scenes and a 2026-08-02 research package read it, but I did not scope the integration.

---

## 10 · ESCALATIONS (headline, per the operating standard)

1. ⛔ **`PROGRAM_OVERVIEW.md` contradicts itself on the headline closed-loop result.** Line 45 carries
   retraction **R-2026-08-03-C** ("ADE separates at +7.164"); §5.0.1 still prints the superseded
   "ADE saw nothing" table. **The stale reading is actively propagating — it reached me inside my own
   brief.** Fix §5.0.1 to the `closedloop-hq-render` panel, or mark the old table RETRACTED in place.
2. ⛔ **§5.0.1's video path does not exist** (`TanitAD Research Lab/Evaluation/Videos/…`, 0 tracked
   files). Real path: `TanitAD Research Lab/Benchmarks & Evals/_evaluation/Videos/alpasim-closedloop-thor-2026-08-03/`.
   Looks like Research-Hub→Research-Lab rename drift; worth a repo-wide sweep for the same pattern.
3. ⛔ **refav1 closed loop is blocked on the A1 unit decision**, already awaiting a PI/Master-Mind
   call. ⚠️ **Independently of that decision, `stack/tanitad/data/refav1_loader.py:17-23` should be
   fixed today** — its docstring still asserts the refuted "(kappa, accel-like)" reading as MEASURED,
   so the file actively misinforms anyone who opens it. The corrected text is already drafted in that
   package. **Zero-risk doc fix; does not touch the live run.**
4. ⚠️ **The 2.7 vs 2.9 wheelbase decision has been open since 2026-07-26.** Settle it **together with
   A1**, in one turn, and re-run the affected suites — not piecemeal.
5. ⚠️ **`EVAL_DOCTRINE.md`'s "T2 not provisioned" line is stale** (doc 2026-08-09, panel 2026-08-03)
   — and the same doc's tier vocabulary now needs the PI's 2026-09-02 ruling folded in, since what it
   calls "T1 closed loop" (`taniteval/taniteval/closedloop.py`, `taniteval/tools/t1_eval.py`) is
   **action-closed, world-open**: the model's trajectory never moves a vehicle. **This is a
   vocabulary correction, not a re-measurement** — no banked number changes, but every "T1" stamp
   needs re-reading against the new definition.
6. ⚠️ **`n = 2` renderable scenes is the real ceiling on this capability**, and one of them is
   junction-free. If the PI wants a closed-loop *result* rather than a closed-loop *demonstration*,
   acquiring more NuRec volumes — especially a junction scene — is the highest-leverage item. It is a
   provisioning decision, so it is the PI's.

---

## 11 · DELIVERABLE MANIFEST

| artifact | where it lives | only one place? |
|---|---|---|
| This document | `repo:Project Steering/CLOSED_LOOP_FEASIBILITY_2026-09-03.md` — **committed on `agent/arch-inf-20260803`** via `stack/scripts/scoped_commit.py`, and **verified scoped: `1 file changed`**, no sibling work swept. Not pushed; not on `main`. | no |

⚠️ **One operational note worth recording, because it will recur on this mount.**
The first two `scoped_commit.py` attempts died at `git read-tree HEAD -> 128` on
`fatal: Unable to create '.git/scoped-commit-index.lock': File exists` — and **both `rm` and `mv` on
that 0-byte lock failed with `Device or resource busy`** while **no `scoped_commit` process was
alive**: a handle held by one of the **59 orphaned `git.exe` processes** on this box. That is the
poisoned-file class CLAUDE.md documents for `.git/COMMIT_EDITMSG`, reproduced on the scratch-index
lock. It cleared on its own; the third attempt succeeded after **~13 minutes** of `read-tree` over
8 952 files on the `G:` mount. ⇒ **Budget minutes, not seconds, for `scoped_commit` here, and do not
kill git processes to "unblock" it** — that is the intervention that caused the previous index
corruption on this mount.
⛔ **A pathspec-free `git commit -F` would NOT have been an acceptable fallback**: the shared index
held **19 entries, 18 of them another agent's live work** (`…/2026-09-03-cost-repair/`,
`stack/tanitad/refs/refa_v1.py`, `Project Steering/MODEL_REGISTRY.md`,
`taniteval/tools/ref{a,c}v3_arm.py`, …). Committing the whole index would have swept them under this
message — the exact failure CLAUDE.md records having happened twice. **`scoped_commit` is the right
tool precisely because it refuses that.**
⚠️ **HEAD moved under me mid-turn** (another agent landed `76143cd` while I worked), which is the
documented normal state here — re-verified staging at the end of the turn rather than trusting it.
| Host probe outputs | quoted inline in §§1-9; every probe is read-only and reproducible from the commands named there | n/a — no host artifacts created |
| Code changes | **none.** No file was modified; no harness was patched; no render was started. | n/a |

**Nothing was stranded.** The only write to any host was a 64 MiB `dd` probe on Thor's own home,
deleted in the same command.
