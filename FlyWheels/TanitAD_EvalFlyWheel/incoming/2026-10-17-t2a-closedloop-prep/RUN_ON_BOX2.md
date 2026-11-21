# RUN_ON_BOX2 — T2 closed-loop evaluation of refcv7-50,400 (T2-A)

**Tier T2** (perception-closed loop, in-house NuRec/gsplat harness, NOT AlpaSim). **The loop is closed on PERCEPTION, not on
INTERACTION**: nothing in the scene responds to the ego, there is no map, the vehicle model has no collision response. This
measures how well a model drives itself; it cannot measure what happens when it drives badly, and it cannot say "safe".

Nothing here has run on a GPU yet. Every box-2 number below is `UNVERIFIED` until `preflight_box2.py` says otherwise.
Box 2 = Windows PC, 8 GB NVIDIA GPU, on the LAN. The dev box never touched it; no contact with Thor is needed or made.

## 0. What the run is

2 scenes × 9 arms (+1 optional control) × 9 start frames × 50 ticks at 10 Hz, `empty` condition, the SAME render flags as every
banked Thor closed-loop number (`--cull-scale-quantile 0.95 --sky-gain 0.3`, rolling shutter off).

| tag | what it is | role |
|---|---|---|
| `cl_ha0` | constant-velocity straight line | **zero control**: commanded steer and accel must be EXACTLY 0.0 (checked automatically) |
| `cl_ha0_ext` | the `ha0_ext` echo, state held from t0 (harness arm) | floor |
| `cl_ha0_ext_live` | the same echo re-estimated from the live pose history every tick | floor (per-tick driver) |
| `cl_p10_t0` | P10 (kd_x + MLP residual), held from t0 | floor, like-for-like with `cl_ha0_ext` |
| `cl_p10` | P10 re-estimated every tick | floor (per-tick driver) |
| `refcv7_vmax_off_s0` | refcv7-50,400, ceiling WITHHELD (`v_max_valid=0`), inference seed 0 | **the model under test (deployable form)** |
| `refcv7_vmax_off_s1` | as `_off_s0` with inference seed 1 | **inference-seed replicate**: the floor an effect must clear |
| `refcv7_vmax_log_s0` | refcv7 on the LAUNCH tree, ceiling = max logged speed over [t0+2 s, t0+6 s], quantised to 30/50/100/120 km/h | **ORACLE, low-yield, run last.** SPEC_REFCV7 §26.1 (INHERITED): on the launch tree the ceiling never reaches the emitted plan, so a null pair vs `off_s0` is the expected result. Scene 1's ceiling is the 100 km/h bin (27.8 m/s), which cannot bind a ≤ 20 m/s plan by construction; scene 2 may bind at some starts |
| `refcv7_r4code_vmax_log_s0` | the SAME weights on the R4 inference code (`tree_r4/`, commit `0fcb1c1c`, the SPEC 26.1 fix), same ceiling | **ORACLE**, same caveats, runs only if `tree_r4/` is in the kit. Never the headline: the ceiling is derived from the logged future |
| `refc-base` (`--with-control`) | refc-base through THIS box's render | cross-host repro control against the banked Thor panel |

Every steer **angle** that is integrated passes `action_units="steer"` (recorded channel 1 is `arctan(2.9·κ)`, not a curvature;
`"kappa"` over-rotates ≈2.9× — RETRACTION_LOG R46). The package test pins it with an analytic circle (ratio of the two readings 2.7-3.1×).
Per-tick echo floors are unit-gain feedback loops, so a bias integrates; that is why the held and the live variants both run.

## 1. Files to copy (exact)

Source on the dev box: `C:/Users/Admin/t2a_scratch/box2_kit/` — **2.96 GB (≈ 2,958,400,000 B; `KIT_MANIFEST.json` lists every file exactly)**, built by `code/make_kit.py`, self-contained
(its own copy of the test-suite passes from inside it: `raw/pytest_fast_from_kit.log`, 21 passed / 1 slow skipped). `KIT_MANIFEST.json` in the
kit root lists every item with size and md5. Copy the whole folder to e.g. `D:\t2a\kit` on box 2 (needs ≥ 4 GB free for the kit + results).

| path inside the kit | bytes | what |
|---|---:|---|
| `ckpt/ckpt_50400.pt` | 402,708,019 | refcv7-50,400 checkpoint (md5 `b418d0fc4a92a6848c246a6a7c50207b`) |
| `ckpt/config.json` | 66,500 | its run config |
| `scenes/00040136-e651-4abd-991d-0655ccda9430/volume.nurec` | 639,393,135 | scene 1 |
| `scenes/00040136-e651-4abd-991d-0655ccda9430/rig_trajectories.json` | 2,155,480 | scene 1 rig |
| `scenes/7c72937c-c620-4776-9555-d57222c0081f/volume.nurec` | 603,238,216 | scene 2 |
| `scenes/7c72937c-c620-4776-9555-d57222c0081f/rig_trajectories.json` | 2,136,806 | scene 2 rig |
| `p10/p10_S2.pt` (and a copy in `assets/`) | 21,011 | P10 floor weights |
| `assets/p10_known_value_fixture.{npz,meta.json}` | 43,660 + 283 | known-value fixture for the P10 test |
| `refcv6_eval_kit/data/anchors/refc_anchors_6s_v0cond_alat_117.pt` | 13,477 | decoder anchors |
| `refcv6_eval_kit/data/refcv7/map_hires_class_weights_train_100x30.json` | 3,046 | map class weights |
| `refcv6_eval_kit/data/refcv7/nav_compliance_tau_train.json` | 2,410 | nav tau |
| `refcv6_eval_kit/data/refcv6_train_eval139_extrinsics.json` | 3,367,838 | extrinsics the original lift bank is built from |
| `models/refc-base/ckpt.pt` | 1,250,838,325 | OPTIONAL control checkpoint |
| `models/refc-base/config.json` | 3,174 | OPTIONAL |
| `banked/HQ_refc-base_empty.json` | 620,303 | OPTIONAL banked Thor panel the control is paired against |
| `tree/` (1,048 files, ≈ 27.6 MB) | | `git archive fec3a0dc stack taniteval` (the launch commit `refcv7-r101-s0` trained on: `config.json` has `--conflict-every 10`, i.e. the pre-restart run), pruned, + overlay of the patched `closedloop_drive.py` (64,277 B) and `floors_kin.py` (17,885 B) |
| `tree_r4/` (1,125 files, ≈ 27 MB) | | `git archive 0fcb1c1c` (R4 restart commit, a descendant of `fec3a0dc`) + the same overlay: inference code on which the ceiling reaches the plan. Strict-loads the same checkpoint (`build_model` raises on any mismatch and did not) |
| `t2a/*.py`, `t2a/tests/` (11 files) | ≈ 0.12 MB | the package code |

Without the optional control the assets are 1.65 GB; the 1.25 GB refc-base checkpoint is only for `--with-control`.

```powershell
# from the dev box (adjust the share), or put the folder on a USB drive
robocopy C:\Users\Admin\t2a_scratch\box2_kit \\BOX2\share\t2a_kit /E /R:2 /W:2 /MT:4
```

## 2. Environment on box 2 (once)

Python ≥ 3.10 (the dev box ran 3.13.5, torch 2.11.0+cu128 for every dry test). In an **x64 Native Tools Command Prompt for VS** (so `cl` is on PATH):

```bat
python -m venv D:\t2a\venv && D:\t2a\venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128        :: CUDA build matching the driver
pip install numpy scipy pyyaml pillow safetensors huggingface_hub msgpack ninja pytest
pip install gsplat==1.5.3                                                   :: JIT-compiles CUDA kernels on first use: needs nvcc + cl + ninja
python -c "import torch;print(torch.cuda.get_device_capability())"          :: then: set TORCH_CUDA_ARCH_LIST=<major>.<minor>
```

Needed: torch(+CUDA), numpy, msgpack, gsplat==1.5.3 (the version every banked Thor number used). NOT needed on this in-process path: cv2, grpc
(MEASURED by `grep`: they are imported only inside helpers the run does not call). `raw/devbox_pip_freeze.txt` is the dev-box venv for reference.
UNVERIFIED: that gsplat 1.5.3 builds on Windows with this torch; if it does not, see BLOCKERS.md B2.

## 3. Preflight (≈ 2-10 min)

```bat
cd D:\t2a\kit
python t2a\preflight_box2.py --kit . --out preflight.json
```

Checks: software, every kit file by size (and md5 for the checkpoint, scenes and config), GPU name/VRAM, then **the VRAM question**: peak GPU
memory of (a) the renderer alone, (b+c) refcv7 built on the same device with the scene still resident and two real closed-loop ticks. It also
renders frame 30 twice (non-trivial, deterministic) and prints `render_ms` and `plan_ms`.
On the dev box the same script returns `26/29 ok` with exactly the three expected failures (msgpack, gsplat, no visible GPU): `raw/preflight_devbox_expected_3_hard_failures.json`.

| preflight result | meaning | do |
|---|---|---|
| 0 HARD failures | go | step 4 |
| `import gsplat` / JIT build fails | toolchain | BLOCKERS B2 |
| `(c) ... fit in VRAM` FAIL / CUDA OOM | the 8 GB card cannot hold both | BLOCKERS B3. Do NOT change the render flags for some arms only — one render config for the whole grid or the pairs are void |
| render not deterministic (soft) | pixel noise between identical calls | run on, but note it: it widens every paired interval's true uncertainty |

## 4. Run

Smoke first (cheap, validates render + loop end to end; ≈ 10 min ESTIMATED):

```bat
python t2a\run_on_box2.py --kit . --out results --scenes 00040136 --arms cl_ha0,cl_p10_t0
```

`cl_ha0` must print `[ok]` — the runner reads back every commanded steer/accel and marks the arm FAILED unless both are exactly 0.0
(deliberate-regression arm: a 1e-9 steer or accel makes the check go RED — `tests/test_t2a_dry.py::test_box2_runner_zero_control_check_reads_exact_zero_and_goes_red_on_a_1e_minus_9_steer`). Then the whole grid (resumable — an arm whose rollouts file exists is skipped):

```bat
python t2a\run_on_box2.py --kit . --out results            :: add --with-control for the refc-base cross-host control
```

Arms run in value order: floors first (they validate the render path), then refcv7 off/s0, off/s1, the launch-tree ceiling control, the R4-code ceiling arm. Per arm and scene the runner writes
`results\<scene8>\<tag>\rollouts_<arm>_empty.json` + `run.log`; at the end, per-arm `T2A_<scene8>_<tag>.json` (every file stamped `"tier": "T2"`
with the scope sentence), paired `T2A_PAIR_*.json`, `T2A_SUMMARY.md/.json`, and `PROGRESS.json`.

**Runtime = ESTIMATED, decide from the preflight numbers:** wall ≈ 9,720 × t_render + 3,600 × t_plan (18 arm-scenes × ≈ 540 renders incl. the 10-frame warm-up;
4 refcv7 arms × 2 scenes × 450 plan calls). With a guess of t_render 0.4 s and t_plan 1.5 s that is ≈ 2.6 h (drop `refcv7_vmax_log_s0` with `--arms` to save ≈ 0.4 h: it is a control with a predicted zero); the CPU plan is 165-201 s per tick
(MEASURED), so the CPU is never an option. Host RAM for the 3.1 M-gaussian scene decode: UNVERIFIED, plan for ≥ 16 GB.

## 5. What to bring back (≈ tens of MB)

`results\` (without `*.png` if any), `preflight.json`, and `PROGRESS.json`. Drop them in a new
`FlyWheels/TanitAD_EvalFlyWheel/incoming/<date>-t2a-closedloop-box2/raw/` and score nothing by hand: the JSONs are already scored.

## 6. How to read it — committed BEFORE any T2 number exists

Gates, in order. A failed gate voids what follows it, and the failure is the finding.

1. Preflight: 0 HARD failures and (c) fits.
2. Zero control exact 0.0 (automatic).
3. `--with-control` only: `T2A_PAIR_00040136_refc-base_THISBOX_minus_BANKED_THOR.json`. If its `ade_0_2s` paired CI separates from 0, box 2's render path is not
   Thor's; within-box-2 pairs stay valid, any cross-host statement is void.
4. The **inference-seed floor** = `T2A_PAIR_*_refcv7_vmax_off_s1_minus_refcv7_vmax_off_s0.json` (same checkpoint, same flags, only the DDIM seed differs). An effect
   smaller than this floor is not an effect. The episode-cluster bootstrap in every pair file answers ONE question only — *would another draw of rollout starts
   say this?* — and is structurally blind to training-seed variance (one checkpoint here: UNMEASURED) and, separately, inference-seed variance (this arm).
   Every statement names which of the three it rests on.

The question: **does refcv7-50,400 drive better than its own kinematic floors in the perception-closed loop?**
Bar (per scene, per family, never pooled): `refcv7_vmax_off_s0 − {cl_ha0_ext, cl_p10_t0, cl_p10, cl_ha0_ext_live}` has a paired CI wholly on the side of refcv7 AND |Δ| exceeds the
inference-seed floor. A floor that beats refcv7 on a family is a FAIL on that family, reported as written; the bar does not move after the data.
`log − off_s0` (launch tree) and `r4code_log − off_s0` price the oracle ceiling: they are levers, so they must clear the same floor, and an exact 0 is the EXPECTED reading on scene 1 (the 100 km/h bin cannot bind) and possible on scene 2 (the empty-survivor fallback keeps the whole fan when no candidate is compliant, `refcv6_selection.py:187-192`). Whether the ceiling can move the emitted plan at all is UNVERIFIED on both trees (BLOCKERS B5); `cl_p10_t0 − cl_ha0_ext` and `cl_p10 − cl_ha0_ext_live` price P10's residual over the echo.

If it fails (Rule Zero — a refutation is a waypoint), the next levers in order of expected effect, each cheap and already wired:
1. longitudinal vs lateral split: re-run refcv7 **on the R4 tree** (`T2A_TREE=<kit>	ree_r4`; on the launch tree the ceiling is inert, see the arm table) with `--t2a-vmax-mode fixed --t2a-vmax-ms <scene logged speed>` (separates "wrong speed" from "bad steering");
2. the steer clamp (0.05 rad ⇒ κ ≤ 0.0185 1/m) — compare planned vs executed curvature in the LATERAL family;
3. the render domain gap (REF-C open-loop ADE 1.5157 on reconstructions vs 0.4728 on real footage, 3.21× OOD): if refc-base also loses to the floors, the gap is the lever, not refcv7;
4. held vs live floors: how much per-tick closure alone moves the floor.

Four families + the extras, all in each per-arm JSON: ADE · LONGITUDINAL · LATERAL · TACTICAL (degenerate on these straight scenes) · **STRATEGIC: PAUSED** (PI ruling
2026-09-19, item 25 — raw values "for reference, NOT A RESULT", never a pass) · progress · **COLLISION and OFF-ROAD UNAVAILABLE with the reason** (no actors in `empty`, no map).
n = 2 scenes, clusters are rollout starts of ONE clip each; both scenes are unseen by refcv7 (`raw/leak_check.json`).

## 7. If something breaks

`results\<scene>\<tag>\run.log` has the full traceback. Re-run the same command: finished arms are skipped. `--no-resume` redoes everything. `--dry` prints every command without running.
