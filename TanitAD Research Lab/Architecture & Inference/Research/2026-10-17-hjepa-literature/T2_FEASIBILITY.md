# T2 (closed loop) feasibility for the refcv family — 2026-10-17 (stream S7, plan §7)

**Question.** G1 ("a model that drives") needs T2: the trajectory *controls the vehicle* and the next observation is
re-perceived from the resulting pose (`EVAL_DOCTRINE.md`, D-LOOP-1). Which simulator gives the cheapest honest first read?
**Tags:** `P:<key>` = PUBLISHED, primary banked in `TanitAD Research Lab/Library` · `P-sec` = published, read from a web page
summary, NOT banked (not admissible for registry/paper) · `M` = MEASURED this session · `I` = INHERITED (earlier note, not re-verified) ·
`E` = ESTIMATED · `U` = UNVERIFIED.

## Verdict (read this first)
1. **We already own a T2 harness**, so the cheapest first read is *not* a new simulator: `stack/experiments/alpasim-gsplat/closedloop_drive.py`
   (kinematic bicycle → re-render from the new pose via gsplat on NuRec scenes; I: `Project Steering/CLOSED_LOOP_FEASIBILITY_2026-09-03.md` §1.3, §8).
   It was removed from Thor on 2026-09-14 but is **intact on the dev box** (M: `E:\thor_archive\2026-09-14\{alpasim,nurec_scenes}` listed today; 79 scene dirs, 2 renderable `volume.nurec`, as in `I`).
2. **Recommended T2-A (first read):** refcv7-50,400 (and refcv8 if it clears) vs the *same-pixel controls* — hold-action/echo, P10 constant-kinematics floor, and
   `refc-base` re-run through the new path (must reproduce its banked numbers to a zero-width CI). Build the adapter now (0 GPU); run it on Thor in a **gap between v8F-α arms** — the plan books Thor for ~7 arms × 4.5 h × 2 seeds from ≈ 2026-10-20 (plan §0 item 11), and a render job must never co-run with a training arm. Running it on refcv7-50,400 first gives a pre-v8F T2 baseline to difference against (plan §4 says "first T2 read after γ"; this moves the baseline earlier, the v8F read stays after γ). Cost **E: ≈ 2 engineer-days + ≈ 1 GPU-hour**, no purchase.
3. **Recommended T2-B (the one that can carry a G1 claim):** AlpaSim native with a TanitAD gRPC driver on an x86-64 Linux GPU, `public_2601` suite subset, minus the 17 clip ids that overlap our corpus. Needs a PI decision (x86 GPU time + gated NuRec download). **Do not start HUGSIM** (reasons below).
4. **Honest scope of T2-A (must travel with every number):** loop closed on *perception*, not *interaction* — replayed actors, no map, no collision response, n = 2 scenes, clusters are segments of one clip (`I`: `CLOSED_LOOP_FEASIBILITY` §4). T2-A can say "the trajectory drives and recovers better/worse than the floors"; it cannot say "safe".

## Comparison
| | **HUGSIM** (as used by WA-JEPA) | **NAVSIM v2 pseudo-sim** | **AlpaSim + NuRec** | **In-house gsplat harness** |
|---|---|---|---|---|
| Loop | real closed loop, 3DGS re-render; gym env | **not** closed: 4 s trajectory is *committed* (bicycle + LQR, 10 Hz), no feedback; stage 2 = ~12 pre-rendered 3DGS views per scene `P:2506.04218` | closed loop, 3DGS re-render when ego deviates; MPC tracks trajectory, bicycle dynamics `P:2511.00088` §6.4 | closed on perception only (`I`) |
| Camera input | per-method rig: LTF 3 front cams, UniAD/VAD 6, WA-JEPA 4 `P:2608.20974` App. A; camera yaml per dataset `P:hugsim-readme` | multi-view cameras + ego status (velocity, history) + discrete command (L/S/R) `P:2506.04218` §3 | any configured camera set; NuRec scenes carry front-wide 120°, front-tele 30°, cross L/R 120°, rear L/R 70° (P-sec: HF dataset card). Alpamayo 1.5 gets newest 4 frames/camera + 1.5 s ego history `P:alpasim-tutorial` | 1 camera, `camera_front_wide_120fov` → cylindrical crop (`I`) |
| Ego state in | ego info provided to client `P:2412.01718` §4.2 | yes (`P:2506.04218`) | `submit_egomotion_observation` (pose + rig-frame velocities/accel), optional route waypoints, `drive(time_now_us, time_query_us)` `P:alpasim-egodriver-proto` | `v0`, nav cmd (`I`) |
| Output expected | waypoints **or** (δ, ȧ) sequence "for the next several seconds"; waypoints → LQR `P:2412.01718` §4.2 | 4 s waypoints in ego frame `P:2506.04218` §3 | `DriveResponse.trajectory` = local→rig_est transforms; horizon/rate not fixed by the proto (U) | 4 waypoints @ 0.5–2.0 s, re-plan 10 Hz (`I`) |
| Other agents | replay / IDM (nuScenes only; needs HD map) / aggressive attackers (hard, extreme) `P:2412.01718` §4.3 | IDM-reactive boxes; pedestrians replay `P:2506.04218` | replay of recorded trajectories `P:2511.00088`; `trafficsim=catk` option exists `P:alpasim-tutorial` | none (`I`) |
| Score | HD-Score = Rc × mean_t[NC·DAC·(TTC,COM)] `P:2412.01718` Eq. 20–21; WA-JEPA 0.4462 vs LTF 0.2310, DrivoR 0.3252, UniAD 0.3124, VAD 0.1393, 436 scenarios, commit `ead17f2` `P:2608.20974` T.2/App. A | EPDMS two-stage; R² 0.8 vs nuPlan CLS; ~6× fewer inferences than 8 s sequential `P:2506.04218` | close-encounter, off-road, "km between events" (AlpaSim score); AR1-10B 0.72 vs 0.5B 0.35 on 920 scenarios `P:2511.00088` T.10; events with > 4 m deviation from log are excluded | our 4 families + corridor departure (`I`) |
| Compute | scene render 89 FPS on RTX 3090, 4.45M Gaussians (Table 5; views/resolution not stated in extracted text, U) `P:2412.01718`; scripts take separate `sim_cuda` / `ad_cuda` `P:hugsim-readme`; full-loop throughput U | pre-rendered; planner inferences only | Docker Compose, `topology=1gpu`; Alpamayo needs ~40 GB VRAM, small drivers (VaVAM, LTFv6) unspecified `P:alpasim-tutorial`; stock renderer is a closed amd64 container (`I`) so not Thor | 38 ms/render @256², 50-tick rollout 5.6 s on Thor (`I`) → E: render ~0.25 s/tick @416×1024 (38 ms × 6.5× pixels), model forward extra |
| Licence | repo MIT (P-sec, GitHub); scenes from KITTI-360/Waymo/nuScenes/PandaSet (research datasets, exact terms U) | repo Apache-2.0 (P-sec); data keeps nuPlan terms | code Apache-2.0 `P:alpasim-readme`; NuRec data: NVIDIA NuRec Dataset License, internal AV/ADAS development, 12-month term, HF gated (P-sec) | ours |

## How a camera-only 6 s planner plugs in
* **Waypoint output (refcv):** refcv3's horizons are 0.5…6 s (`I`); the harness uses the first waypoint as pure-pursuit lookahead and re-plans every tick, so 2–10 Hz both work (hold the last trajectory between plans). Assert the first four horizons rather than assuming them (`I`).
* **Controls output ((a, κ), refav1/v8F imagination planner):** integrate κ directly (`dyaw = v·κ·dt`) and re-sample 0.2 s → harness dt; never round-trip through a wheelbase. AlpaSim needs *poses*, so the driver service integrates (a, κ) to transforms itself; HUGSIM wants (δ, ȧ) or waypoints, where δ = atan(Lκ) must use the **2.9 m steer-encoding constant**, not a vehicle length (`I`: `CLOSED_LOOP_FEASIBILITY` §5).
* **Geometry:** refcv8's chassis is 416×1024 cylindrical (plan §4 table, row "Chassis") vs the harness's 256×256 f-theta; the swap is `calib.cylindrical_rectify` with per-clip principal point and a refuse-on-mismatch check (`I`). NuRec scenes are PhysicalAI-AV clips, so the 120° front-wide camera is *native* — the reason AlpaSim/NuRec fits us and HUGSIM does not.

## Why not HUGSIM now
Our trunk expects a 120° front-wide image; HUGSIM scenes come from nuScenes/Waymo/KITTI-360/PandaSet rigs with much narrower front views (U: from memory, not banked). A cylindrical 120° frame would be mostly empty unless a virtual wide camera is rendered from the splats (3DGS can, but extrapolated-view fidelity at 120° is U). It is also zero-shot cross-dataset for a PhysicalAI-only model, so a floor-level HD-Score would diagnose the domain gap, not the driving. Reserve it as an *external comparator* at rung δ.

## Hygiene findings that bear on the plan
* **WA-JEPA 91.7 and AD-E2E-JEPA 84.1/86.5 EPDMS are `navtest` stage-1 numbers** (`P:2608.20974` T.1; `P:2609.34085` Table 2: "navtest-stage-1"). The pseudo-simulation authors explicitly discourage reporting EPDMS on `navtest` without two-stage evaluation `P:2506.04218` §4.2; the official leaderboard is `navhard` (450 stage-1 + 5,462 stage-2 observations). Our two-stage `navhard` numbers are therefore **not comparable** to those competitor headlines.
* **NAVSIM v2 is T1-family** (open-loop plan scoring) by our doctrine; it cannot satisfy G1.
* **Leak control for T2-B:** NuRec 26.04 = 1,607 PhysicalAI clips; 17 ids intersect our sets (16/4,572 v7.2-train, 1/147 eval) — M: `TanitAD Research Lab/Data Engineering/Research/2026-09-13-posted-speed-limit-supplier/RESULT.md` F1/F5, exclusion list `raw/nurec_corpus_overlap.json`. Overlap with the 2,376-episode refcv corpus: U (expected ⊆ the 16 if that corpus is a subset of v7.2-train).

## Blockers / decisions
| item | owner | blocks |
|---|---|---|
| a ~1 h Thor gap between α arms (plan §0 item 11 books ~63 GPU-h of α arms); alternative host U (dev-box RTX 4060 untested, A40 pod has no gsplat build, `I`) | scheduler / PI | T2-A run |
| ~2 eng-days: `RefCV7Policy` via the strict loader + 416×1024 canonicalisation + controls (hold, P10, refc-base re-run) | stream owner | T2-A |
| x86-64 Linux GPU + HF/NuRec licence acceptance; ~600 MB per scene (`I`), E: has 4.3 GB free (M `df`) so scenes go to a pod volume | **PI** (spend) | T2-B |
| AlpaSim driver servicer (6 RPCs, `P:alpasim-egodriver-proto`) ≈ 2–3 days (E) | stream owner | T2-B |
| refav1 κ-vs-steer unit contract (`I`: `CLOSED_LOOP_FEASIBILITY` A1) | existing PI decision | any refav1 T2 number |
