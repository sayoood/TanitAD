# SHIP-R8 sub-reads: AMENDMENT 4 (H2cn stage) and AMENDMENT 5 (factored decode)

EvalFlyWheel, 2026-10-26. This package runs the registered SHIP-R8 battery, PhysicalAI part only (no NavSim), on two refcv8 variants. Each variant is
defined by a sealed amendment:

| amendment | file | sha256 (seal) | scored arm |
|---|---|---|---|
| 4 | `SHIP_R8_AMENDMENT_4_H2CN.md` | `57b377162b99fc9e9f33307e35fd2cd6211494f5e174d6cc8566dddeb4d04634` | `H2CN_DEPLOY8` |
| 5 | `SHIP_R8_AMENDMENT_5_FACTORED.md` | `b100c470d65ef99131f833191321a3c3194a3875eb126b20e5ce266ee8d991bc` | `FACTORED_DEPLOY8` |

Every tool re-hashes both amendments at start and **refuses** to run if either one differs from its `.sha256` seal. A seal that cannot be read counts
as INCONCLUSIVE and is also refused, never treated as a match.

The registered battery code is **imported and monkeypatched, never edited**. `pf_lane.configure` re-points it at the REGISTERED per-frame table, as
the registered lane did. `deploy8_forward_r8`'s four-hook device then replaces only the ARM POSITIONS. All scoring is `RT.forward`
(`a0k0_headroom_retime_r8.py`), the TAC rows are `tac_retime_r8.run`, and the clauses are the registered verdict's own functions
(`ship_r8_verdict_pf` + its pinned base copy).

Tier and estimator for every number these tools write: **T1 self-action open loop on logged eval139 frames**. Intervals are a **paired
episode-cluster bootstrap, B 2000, seed 0, over EPISODES**. That interval answers only *"would another draw of episodes say this?"*. It does not
answer "another training run" (one training seed) or "another inference run" (the two inference seeds 0 and 1 are reported side by side, not
pooled).

## Environment (CPU only)

```
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 R8_NO_POSE_SYNC=1 PYTHONIOENCODING=utf-8 \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe          # torch 2.x CPU; the registered battery's venv
```
`run_ship_r8_sub.sh` sets all of these. Launch tree: `C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58` (`R8_LAUNCH_TREE`). The eval tree is
`step48852ema_pf` (the registered per-frame battery tree, `C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf`).

## Commands, in the order they must run

```
cd C:/Users/Admin/refcv8_pf/ship_r8_sub
# 1. Amendment 4 s1 -- freeze the H2cn stage (fresh rows only).  Refuses if H2CN_STAGE_FROZEN.pt already exists.  NO eval-window output.
$PY -X utf8 h2cn_freeze.py
#    -> H2CN_STAGE_FROZEN.pt / .sha256 / H2CN_FREEZE_INFO.json.  STOP: the MM registers the sha256 (REGISTERED_H2CN_STAGE.sha256 + GOALS_AND_CLAIMS).
# 2. Amendment 4 read (ONLY after the MM's registration go): forward -> TAC rows (H2CN_DEPLOY8, H2CN_NEAR, RETIME_DEPLOY8) -> verdict
bash run_ship_r8_sub.sh h2cn
$PY -X utf8 ship_r8_sub_forward.py --mode h2cn --tree step48852ema_pf --mutate stage      # deliberate regression: K-PERSURF must read RED
# 3. Amendment 5 identity / control arms (allowed before the seed-1 replicate): forward -> TAC (RETIME_DEPLOY8, OSONLY_DEPLOY8) -> identity record
bash run_ship_r8_sub.sh factored
$PY -X utf8 ship_r8_sub_forward.py --mode factored --tree step48852ema_pf --mutate draw    # K-SSOS must read RED
$PY -X utf8 ship_r8_sub_forward.py --mode factored --tree step48852ema_pf --mutate knot    # K-RETIME must read RED
# 4. Amendment 5 SCORED read -- the MM's one-liner, gated on F:/refcv9_selfsharp/trace_selfsharp_s1.npz final + SSGUARD s1 REPLICATED:
bash C:/Users/Admin/refcv8_pf/ship_r8_sub/run_ship_r8_sub.sh factored_scored
```
Every step is judged on its **artifact**: the JSON exists and carries the field. It is never judged on an exit code read through a pipe. The runner
writes `ZZRUN<step>ZZ` markers to `logs/run_<mode>.log`. If a forward control fails, the run writes `out/forward/WITHHELD_<mode>_<tree>.json`
naming the control, writes no FORWARD file, and exits rc 4.

## Files

| file | role |
|---|---|
| `ship_r8_sub_common.py` | seal check, import paths, fresh-clip disjointness (independent sources + mutation), stage write/load/predict |
| `h2cn_freeze.py` | Amendment 4 s1: per-surface H2cn fit exactly as `l1c_h2cn_confirm` (`l1_fit_score.h2n_fit`, seed 0, 5-fold episode CV), frozen to a file |
| `ship_r8_sub_forward.py` | the deploy8 device with arm positions replaced per amendment (+ all controls, beside arms, `--mutate`) |
| `ship_r8_sub_tac.py` | clause-5 TAC rows of one arm through the registered `tac_retime_r8` device (`TAC.DEP` / `TAC.SCORED_ARM` re-pointed) |
| `ship_r8_sub_verdict.py` | clauses (1)-(7) + (1b) mechanically; clause 1 replaced as each amendment writes it; OVERALL line first |
| `run_ship_r8_sub.sh` | the one-command read per mode, with artifact assertions |

## Controls -- what each one proves

**Freeze (`H2CN_FREEZE_INFO.json`)**
* **K-SEAL**: both amendments equal their seals.
* **K-DISJ**: the fresh fit clips are disjoint from eval139 and from train4572. This is checked as sha256-digest set intersections from INDEPENDENT
  sources: the fresh id list, the fresh tree dump, the v9 eval139/train4572 label npz (md5-pinned to the refcv8 config argv), the 4,711-id
  extrinsics file, the eval tree dump, and the programme digest lists. All intersections must read 0. A **mutation arm** injects one eval139 clip and
  must read RED (it reads 1 overlap). This shows the check can see an overlap.
* **K-FIT**: wd / epochs / n_params / n_fresh equal `H2CN_RESULT.json`, and the CV L1 sum is reproduced exactly. The frozen nets are therefore the
  confirmed nets, not a refit that happens to look similar.
* **K-RELOAD**: the reloaded file's predictor is bit-equal to the in-memory closure on every fresh row.
* **K-DETERM**: the serialised bytes are identical across three serialisations. The payload is serialised in memory: `torch.save` to a file names
  the zip root after the file, so file-based saves are not byte-stable across names.

**Forward (`out/forward/FORWARD_<MODE>_<tree>.json`, `controls`)**
* deploy8_forward_r8's own (a)-(f):
  * RT.forward ALL_PASS, including every arm's on-path geometry.
  * CONST_G1 − os is exactly 0 on every row.
  * The 8-slot known values: g≡1 is bit-equal to plan8, and g≡0 equals the independent walker.
  * RETIME_DEPLOY_PERSURFACE S2 equals the registered FORWARD_TRAINFIT exactly.
  * Row alignment holds.
  * The deploy table equals the REGISTERED per-frame train gains at its knots.
* **K-RETIME**: RETIME_DEPLOY8, RETIME_DEPLOY_PERSURFACE and CONST_G1, re-read through THIS code path, equal the registered
  `FORWARD_DEPLOY8_step48852ema_pf.json` arm blocks and bars EXACTLY (6,760 leaves). This proves the substitution device scores an unchanged arm
  exactly as registered, so any difference in the scored arm comes from the arm, not the harness.
* h2cn **K-REG**:
  * the stage file's sha256 equals `REGISTERED_H2CN_STAGE.sha256`;
  * the registration gate record (`REGISTRATION_GATE_H2CN.json`) is written BEFORE the first eval-side stage output.
* h2cn **K-DISJ**: the freeze's disjointness is re-asserted at forward time.
* h2cn **K-PERSURF**: H2CN_PERSURFACE S2 compliant (H2cn − H0) ADE equals l1c_h2cn_confirm's N1 −0.0130 / −0.0136 (seed 0 / 1) within 1e-4. Beside
  this, the S6 positions are bit-equal to the l1c `arms_S6_s{seed}.npz` arm_H2cn. This proves the frozen file reproduces the confirmed stage on the
  eval windows.
* h2cn **K-ATPROG**: the per-surface positions equal `progress_headroom.at_progress` bit for bit.
* factored **K-SSOS**: FACTORED with P_ss := P_os equals OSONLY_DEPLOY8 bit for bit (composed plan, float32 and float64 positions, and every cell). This
  proves the composition `eval_polyline(geo(P_os), arc(P_ss))` is an identity when both draws are the same plan, so a FACTORED effect cannot be a
  composition artifact.
* factored **K-TRACE**: each input trace is final, has 4,754 windows, carries its hook counters, and has its sha256 recorded.
* factored **K-FEED**: the os-only rolls' own manifests read `perframe_full`. When a roll's dump dir was not kept, the check falls back to the roll
  json's `pf.census.manifest_nav_feed`.

**TAC (`out/arms/<tag>/<ARM>/TAC_RECORD_<tree>.json`)**: for RETIME_DEPLOY8, the clause-5 cells, status and every boolean control equal the
REGISTERED `TAC_step48852ema_pf.json` exactly.

**Verdict**:
* Clause 1 for Amendment 4 is the conjunction the amendment writes: registered sha, registration-before-output, disjointness, the PERSURFACE
  identity, and the RETIME identity against the registered SHIP_R8 cells (clauses 2-5 evidence + TAC).
* Clause 1 for Amendment 5 is `clause1_pf` + K-SSOS + K-TRACE + the RETIME identity + the trace sha recheck.
* Clauses 6/7 re-check the registered PAIRED_READ / PRIOR_PARITY sha256 before reading them.

**Mutations (`out/mutation/MUTATION_<mode>_<kind>_<tree>.json`, `mutation_detected` must be true)**
* `stage`: adds +0.05 m to every S2 output of the frozen S2 net. K-PERSURF must go RED.
* `knot`: adds +0.01 on deploy-table knot 1 (`deploy8_forward_r8.mutate_table`). K-RETIME must go RED.
* `draw`: sets P_ss := the battery tree's own plan, which reproduces the MAPSEL K_STAGE draw mismatch. K-SSOS must go RED.

## Pinned inputs (sha256 measured 2026-10-26)

| file | sha256 | role |
|---|---|---|
| `C:/Users/Admin/qland/hold_r8pf/REGISTERED_PF_TABLE.sha256` | `3e77cd12e2e584e8be5da469da9e5bb4e51e4eda02ec048c7d3470eb7310b7ae` | registered per-frame table seal |
| `C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final/pfpkg/raw/retime_deploy/refcv8_retime_deploy_gains_pf.json` | `3319b003fecc83577430f07dad762bce0a9a0b8594e9748e9a61bc4d264136b7` | registered per-frame RETIME deploy table |
| `C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final/pfpkg/raw/retime_trainfit/FROZEN_GAINS_RETIME_train.json` | `2aacc9ff32d3d2a5a18dda2d72ab5ce6c42d1c1268e9186912c5151bbe4d6d80` | registered frozen train gains |
| `C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final/pfpkg/raw/retime_deploy/FORWARD_DEPLOY8_step48852ema_pf.json` | `939d3da1257a59ad935781d33bf5810e6ee213502aed40e09c7a885271c64aec` | registered FORWARD_DEPLOY8 (K-RETIME reference) |
| `C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final/pfpkg/raw/retime_deploy/FORWARD_DEPLOY_step48852ema_pf.json` | `cd429c449c28b1559a6f74a4def52dda15f626616a7183ced9302d596e029f21` | registered FORWARD_DEPLOY (TAC K6) |
| `C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final/pfpkg/raw/retime_trainfit/FORWARD_TRAINFIT_step48852ema_pf.json` | `30138287283841149d7df678d049d91f209a89dac4f9e109397f9b2972a76f2d` | registered FORWARD_TRAINFIT (control e) |
| `C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final/pfpkg/raw/retime_deploy/TAC_step48852ema_pf.json` | `32f17fb09ebffd35756845e938b48308ab7d93eded1e7292e5be28be59265ceb` | registered TAC (RETIME_DEPLOY8 TAC identity) |
| `C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final/pfpkg/raw/retime_deploy/PRIOR_PARITY.json` | `a439adc3cf14e0422c68226cc4c8e308de0b78446dd0875e9e78808e50480038` | registered PRIOR_PARITY (clause 7) |
| `C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final/raw/ship_r8/SHIP_R8_step48852ema_pf.json` | `893b69a8cd8da312b89c31206db2bf637232468d89711e8bd00e68e34cc48c39` | registered SHIP-R8 verdict (RETIME identity reference) |
| `C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final/code/pf_lane.py` | `86868bcb62a4a64ed6c227d2ad5c4df6b4d87de4060550f145c4baa7a3d32ec9` | pf_lane (imported) |
| `C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final/code/ship_r8_verdict_pf.py` | `61095bc603adf1db4ae1b295c7e7f73962e62903974976c2444bf186d8b1e769` | registered verdict (imported) |
| `C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final/code/_ship_r8_verdict_base.py` | `7543c585d9cc099942862c7f644c5fcae797272cd9b4dfec93708450224660f2` | registered verdict base copy (imported, pinned) |
| `C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code/deploy8_forward_r8.py` | `70036113273de6492fd3498e86569212d81b216ab4d8bef015c5f168369153a0` | battery deploy8 device (imported) |
| `C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code/a0k0_headroom_retime_r8.py` | `842889130005dc108c10f53ad8454f6e9314646ae7bc0488c3519da452d78a67` | battery RT.forward (imported) |
| `C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code/tac_retime_r8.py` | `e3e49cdb809374aa3bfebeced470cfe4cbe70f183373f0a0ae8c69b8d52ab3d7` | battery TAC device (imported) |
| `C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code/compliant_reread_r8.py` | `ab5f29d5bb427fc9127266a83269c7d8214a0328a78513459d48afd4fedc0dfc` | battery compliance sets (imported) |
| `C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code/prior_parity_r8.py` | `b079ccaa03ec7b020aa8e994508a0d234a5e91d681fb075efe33d7df1e78b0f7` | battery prior parity (imported) |
| `C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code/retime_trainfit_r8.py` | `17b05b7ee48f1fe477f95f920e472caecd902e917f324a24aa7d96ab3c2526d4` | battery trainfit (imported) |
| `C:/lgs/retime_stage_scratch/stack/tanitad/retime_stage.py` | `b4bc7b79aa9a0ac02a170b4e633da0de4cd6026f76923c8d2706a0b2244f08ee` | retime_stage (imported) |
| `C:/Users/Admin/refcv8_pf/refcv9_l1/code/l1_fit_score.py` | `bf1a50962ca7b55347336642ff1a2d72ad5f0e565ffa73e5e2acaa34fc272b11` | h2n_fit / mlp_inputs (imported) |
| `C:/Users/Admin/refcv8_pf/refcv9_l1c/code/l1c_h2cn_confirm.py` | `4120e2ced9807e3dedcf5e625a671e5b52ad4ccdb994150a278564870b023736` | l1c confirm (fit recipe reference) |
| `C:/Users/Admin/refcv8_pf/refcv9_l1c/result_h2cn/H2CN_RESULT.json` | `48c38b5bc06fdf4206f581ce415f350decbb1f40aea235aff114360067bffd1c` | l1c H2CN_RESULT (K-FIT reference) |
| `C:/Users/Admin/refcv8_pf/refcv9_l1c/result_h2cn/arms_S6_s0.npz` | `a79556b0fa50075c32ef22613ca3a2fbfedcd378eefc9dbe1b88ef4520947a62` | l1c S6 arm s0 (beside S6 identity) |
| `C:/Users/Admin/refcv8_pf/refcv9_l1c/result_h2cn/arms_S6_s1.npz` | `d1cf787a287f134af06a8ba0fbf3eaa3c7e3e3b28d32ebea0d5f9caaa90f4671` | l1c S6 arm s1 (beside S6 identity) |
| `F:/l1c/bank/ztac_fresh.npz` | `c07c7849c7ba897aea29ee0068d970ae5d511a13f88f4113f25c033a3fa20755` | fresh-clip z_tac bank (fit rows) |
| `F:/l1c/select/final_ids_fresh.json` | `04ff350eb37e00e8ccde395498a70ac737fb3c269fe584c81dfe5bcb626e628a` | fresh clip list |
| `E:/refcv8_eval_kit/data/refcv8_v9labels/release/v9_labels_eval139.npz` | `1ce7ea2d227fb3358e9b61bd68ba2f1fbe4b06b3f8c8f256eb6d7f594df02532` | v9 eval139 labels (disjointness) |
| `E:/refcv8_eval_kit/data/refcv8_v9labels/release_train4572/v9_labels_train.npz` | `b2efe86452ed935bf07bc0b913aa92ff37fd25362d90f1d7f7424447aeb0c383` | v9 train4572 labels (disjointness) |
| `E:/refcv8_eval_kit/data/refcv8_train4572_eval139_extrinsics.json` | `933c779b52a0fbb20e37445a933d49d8fa7a630307d3b5eceaeb1a751cc3eb1a` | 4,711 clip extrinsics (disjointness x-check) |
| `C:/Users/Admin/refcv8_pf/refcv9_l1/bank/ztac_eval_s0.npz` | `e807c0d7d43ece24e39b6b23d1d5504162222bddca2e24b86c7e7b0d48bd1586` | L1 os-only eval bank s0 (factored) |
| `C:/Users/Admin/refcv8_pf/refcv9_l1/bank/ztac_eval_s1.npz` | `a1de32c6236b2335790b9a322301814e4923bca9c73d2e9abeebc10bd9f5ac5d` | L1 os-only eval bank s1 (factored) |
| `C:/Users/Admin/refcv8_pf/ship_r8_sub/H2CN_STAGE_FROZEN.pt` | `f293b7bdc06aaeec19a10fd8cac9af993c4116401034a9d4ff9c28649e256f32` | the frozen H2cn stage (REGISTERED by the MM 2026-10-26T04:24:43+0100, REG-R8-H2CN-STAGE-1) |

## Measured wall times (dev box, CPU, 4 threads; other agents' jobs were running concurrently)

| step | wall | note |
|---|---|---|
| `h2cn_freeze.py` | **658.1 s** | S2 fit 392.1 s + S6 fit 240.9 s + checks; run alone |
| `run_ship_r8_sub.sh h2cn` | **557 s** total | forward 427 s, TAC H2CN_DEPLOY8 37 s / H2CN_NEAR 44 s / RETIME_DEPLOY8 48 s, verdict ~1 s; ran concurrently with the factored read |
| `run_ship_r8_sub.sh factored` | **435 s** total | forward 364 s, TAC RETIME_DEPLOY8 36 s / OSONLY_DEPLOY8 34 s, identity record <1 s; ran concurrently with the h2cn read |
| `--mode h2cn --mutate stage` | **672 s** | ran concurrently with the factored mutations |
| `--mode factored --mutate draw` | **457 s** | |
| `--mode factored --mutate knot` | **586 s** | |
| `ship_r8_sub_verdict.py` (re-run, label fix) | <2 s | pure json |

## Control results of the 2026-10-26 runs (all read from the artifacts)

| control | artifact | result |
|---|---|---|
| freeze K-SEAL / K-DISJ (+ mutation RED, 1 overlap) / K-FIT (CV L1 exact) / K-RELOAD (bit-equal) / K-DETERM | `H2CN_FREEZE_INFO.json` | all PASS |
| h2cn forward: D8 (a)-(f), K-RETIME (6,760 leaves, 0 mismatches), K-REG, K-DISJ, K-PERSURF (absdiff 0.0 at both seeds), K-ATPROG | `out/forward/FORWARD_H2CN_step48852ema_pf.json` | ALL_PASS |
| factored forward: D8 (a)-(f), K-RETIME (6,760 / 0), K-SSOS (2,256 cell leaves, 0 mismatches; positions bit-equal), K-TRACE, K-FEED | `out/forward/FORWARD_FACTORED_step48852ema_pf.json` | ALL_PASS |
| RETIME_DEPLOY8 TAC == registered TAC (cells, status, controls) | `out/arms/{h2cn,factored}/RETIME_DEPLOY8/TAC_RECORD_step48852ema_pf.json` | 0 mismatches, both |
| RETIME_DEPLOY8 clauses 2-5 == registered SHIP_R8 evidence exactly | `out/ship_r8/IDENTITY_factored_step48852ema_pf.json`, clause 1 of `SHIP_R8_H2CN_DEPLOY8.json` | 0 mismatches |
| mutation `stage` -> K-PERSURF | `out/mutation/MUTATION_h2cn_stage_step48852ema_pf.json` | **RED** (S2 H2cn − H0 +0.0015 / +0.0010 vs −0.0130 / −0.0136); every other control stayed green |
| mutation `knot` -> K-RETIME | `out/mutation/MUTATION_factored_knot_step48852ema_pf.json` | **RED** (330 of 6,760 leaves differ, both bars differ) |
| mutation `draw` -> K-SSOS | `out/mutation/MUTATION_factored_draw_step48852ema_pf.json` | **RED** (537 of 2,256 cell leaves differ). Only seed 0 can see it: at seed 1 the L1 os-only bank's plan equals the battery tree's plan on every window (0 of 4,754 S2 / 0 of 3,668 S6 differ, vs 4,630 / 3,607 at seed 0), so P_ss := tree plan is a no-op there |

The registration gate record (`REGISTRATION_GATE_H2CN.json`) is written into the forward's scratch work dir and removed with it. Its full content
(registered sha256, registration-file mtime 04:24:43+0100, gate passed 04:27:08+0100, first eval-side output 04:27:09+0100) is carried inline in
`FORWARD_H2CN_step48852ema_pf.json` -> `controls.K_REG_detail`.

## Known deviations / flags for the MM

1. **The factored identity/control arms were run on inference seeds 0 AND 1, not "on seed 0" only.** A seed-0-only read is impossible on the
   registered device. `RT.forward` computes every arm's flip floor from both seeds unconditionally
   (`a0k0_headroom_retime_r8.py:853`, `fl = {surf: abs(g(0, surf)[...] - g(1, surf)[...])}`), and the bars carry no seed parameter. The
   identity/control arms only use the L1 os-only banks `ztac_eval_s{0,1}.npz`, which were final for both seeds. So the run read both seeds and **did
   not read the seed-1 SELFSHARP trace and did not compute any FACTORED_DEPLOY8 cell**. `FORWARD_FACTORED_*.json` carries
   `selfsharp.not_read` for both seeds.
2. **FZR-LAT is read "minus_os" against the battery panel's os** (`a0k0_headroom_retime_r8.py:863`, `acc_lat({... ["minus_os"][k] ...})`).
   OSONLY_DEPLOY8 / FACTORED decode from the L1 os-only roll, which is a different inference draw of the same checkpoint than the battery panel's
   os. So FZR-LAT on those arms includes the os-vs-os draw difference. This is kept verbatim because the registered code computes it this way; it is
   flagged, not re-interpreted.
3. **No external sha256 pin of the input traces was found.** Clause 1 (A5) therefore re-hashes each trace against the forward's own record, plus an
   optional `--trace-pins <json>` (`{"s0": {"l1_base": sha, "selfsharp": sha}, "s1": {...}}`) if the MM holds independent pins.
4. **The SELFSHARP s0 dump dir (`F:/refcv9_selfsharp/dump_selfsharp_s0`) was not kept.** K-FEED falls back to the roll json's
   `pf.census.manifest_nav_feed` (the roll's own manifest census), and the record names which source it read.
5. **H2CN_PERSURFACE is not an 8-slot arm** (S2 slots 0-3 and S6 slots 1,3-7 separately), so it gets no TAC package. Its beside clause 5 reads
   INCOMPLETE by construction. Its role is the K-PERSURF identity, not a verdict.
6. Concurrency: the h2cn and factored reads ran at the same time on the CPU (separate work/arms/log dirs, no shared output path), while other agents'
   jobs were also running. The walls below include that contention.
7. **The verdict wrapper corrects two labels after the first h2cn verdict; no status or cell changes.** (a) The registered base's clause-5 PASS
   reason is a hard-coded string naming RETIME_DEPLOY8 (`_ship_r8_verdict_base.py:198`). The base also requires the TAC file's clause-5 arm to
   equal the scored arm (`:184`), so the cells judged were H2CN_DEPLOY8's. The wrapper substitutes the arm name in the reason and keeps the registered
   text verbatim in `evidence.reason_text_registered_verbatim`. (b) The `inputs` record let the registered verdict's `tac` key overwrite the arm's
   TAC path. Now `tac_arm` is the arm's file and `tac_registered_RETIME_DEPLOY8` is the registered one. The verdict was re-run (pure json, <2 s);
   OVERALL and every clause status are unchanged.
8. **Beside finding (no clause reads it): the H2cn stage raises fed-speed-ceiling exceedance.** Compliance row (A-R8-CLAMP-2 s2.3), full plan
   0-6 s, compliant set, H2CN_DEPLOY8 vs RETIME_DEPLOY8:
   * S2: 0.0163 vs 0.0055 (s0) and 0.0163 vs 0.0060 (s1);
   * S6: 0.0169 vs 0.0049 (s0) and 0.0169 vs 0.0055 (s1).

   These are episode-cluster rates, not a paired difference. H2CN_NEAR sits in between (≈0.009). This is a PI-relevant fact for the capped refcv8,
   reported beside the PASS, not folded into it.

## AMENDMENT 4b -- `H2CN_CAP8` (the H2cn stage clamped to the FED ceiling), EvalFlyWheel 2026-10-26

Sealed text: `SHIP_R8_AMENDMENT_4B_H2CN_CAP.md` (sha256 `339363f505393c268298f058a7b3ff96130667aab424c52e519eeec348b4b70b`, seal file beside it). **Everything writes to `out_4b/`; the landed `out/` is never touched**
(`--cap8` refuses `--out-root out/`; the verdict refuses a cap8 forward with `--out-dir out/ship_r8`). Amendments 4 / 5 flows are unchanged: `--cap8` is the only switch.

```
bash run_ship_r8_sub.sh h2cn_cap8        # forward --mode h2cn --cap8 -> TAC (H2CN_CAP8, H2CN_DEPLOY8, H2CN_NEAR, RETIME_DEPLOY8) -> verdicts H2CN_CAP8 + H2CN_DEPLOY8 (reproduction)  ~640 s
$PY -X utf8 ship_r8_sub_forward.py --mode h2cn --cap8 --mutate cap105 --out-root out_4b      # deliberate regression: clamp with lim*1.05 -> K-CAP-COMPLY must read RED
$PY -X utf8 read_4b.py                   # REPRO_H2CN_DEPLOY8_vs_landed.json (exact leaf comparison vs the landed out/) + SUMMARY_4B.json
$PY -X utf8 test_ship_r8_sub_cap8.py     # the clamp against analytic literals
```
* Arm: `cap_progress(s8, lim)` in `ship_r8_sub_forward.py` -- `s_cap[k] = min(s8[k], s_cap[k-1] + lim*(t[k]-t[k-1])*(1-1e-6))`, `lim` = `cur["lim"][rows]` widened float32 -> float64 exactly; non-finite `lim` = unclamped.
* Controls (in `controls`, all in `required`, so a failure WITHHOLDS): `K_CAP_ID` (lim := +inf -> progress, float64 and cast positions bit-equal H2CN_DEPLOY8; non-finite-lim windows bit-equal), `K_CAP_COMPLY`
  (compliance row `exceed_rate_strict` mean 0 and n_exceed 0 on 16 cells: 2 surfaces x 2 seeds x {panel grid, full plan} x {all_windows, compliant_set}), `K_CAP_MONO`. Clamp-touched counts: `per_window_extra[*].cap8`.
* Verdict clause 1 for `H2CN_CAP8` = Amendment 4's clause-1 function (unedited) + the 4b seal re-hashed now == the forward's record + the three K-CAP controls.
