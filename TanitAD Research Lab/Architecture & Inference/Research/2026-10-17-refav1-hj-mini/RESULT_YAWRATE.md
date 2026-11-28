# RA-1 / v8F-mini: the MEASURED YAW RATE replaces the steer angle (PI ruling `D-PI-YAWRATE-1`, 2026-10-17)

PI, verbatim: *"the model should get measured ego motion and its history, instead of steering angle which represents a sort of action, the model should get the measured yaw rate"*.
Spec: `Project Steering/PREREG_HJ_STAGE0.md` Amendment 3 and `PREREG_V8F_MINI.md` Amendment 3 (both written before any compliant-run data). Nothing was launched; CPU only.

## What changed (`kin_mode`: `steer` = legacy, `yaw` = compliant; the trainers and evaluators default to `yaw`)
| input | steer mode (retired) | yaw mode |
|---|---|---|
| P10 features `(v0, a0, k0)` | `a0` 0.2 s, `k0 = actions[2t, 0]` | `p10_pose`'s triple: `a0` one frame, `k0 = clamp(ω0 / max(v0, 2), ±0.3)`, `ω0` = measured yaw rate |
| `kd_x` base | integrated as `tan(k)/2.9` | integrated as a curvature |
| action channel 1 (RA-1 WM, IDM targets, SENS; v8F-mini WM) | `steer` | `yaw_rate` (RA-1: `("a", "yaw_rate", "v")`) |
| the spelling | — | `ego_channels_from_poses`' `dyaw`, called, never re-implemented |

* The S4 cache holds only the even frames, so the one-frame yaw difference reads `native_tracks.pt` (poses only; `stack/scripts/build_native_tracks.py`). The loader REFUSES a track that does not reproduce the clip's `pose_grid`. On the 3 real cache clips available locally all 301 even rows agree exactly and `κ0` correlates 0.99 with the recorded curvature.
* Why not the cache's own 0.2 s grid: MEASURED `raw/p10_yaw_dt_ablation_540.json`, 2.5944 m vs 2.5627 m at 6 s, a gap larger than the P10 training-seed floor (0.0213 m).

## P10-yaw reference numbers (MEASURED, eval-141, 2,399 windows; `raw/p10_yaw540/p10_fit.json`, `…/2026-10-17-v8f-mini/raw/p10_yaw/p10_fit.json`, `raw/p10_yaw_vs_steer.json`)
| P10 | ADE @ 6 s | ADE @ 2 s | holdout (clips 541–600) ADE @ 6 s |
|---|---|---|---|
| steer-540 (retired) | 2.5560 | 0.2169 | 3.0469 |
| **yaw-540 (RA-1)** | **2.5627** | **0.2185** | **3.0529** |
| steer-600 (retired) | 2.5744 | 0.2187 | — |
| **yaw-600 (v8F-mini)** | **2.5817** | **0.2201** | — |

Paired `yaw − steer` (episode-cluster bootstrap, B 2,000; episode variance only): 540 clips +0.0067 m [-0.0154, 0.0302] at 6 s; 600 clips +0.0074 m [-0.0140, 0.0306]; neither separated. The pipeline control (steer refit, same script) reproduces the banked 2.5560 and 2.5744. The compliant floor is slightly LOOSER (disclosed in both amendments).

## Guards (`stack/tests/test_yaw_rate_ruling.py`, 26 tests, plus the parametrised `test_v8f_mini.py` end-to-end)
* spelling = `dyaw` on 141 real clips ≤ 1e-6, a one-frame shift fails; causal (a later pose never changes row `2j`); `window_kin(yaw)` never opens `actions`;
* a twin cache with garbage `steer / kin3 / p10 / yaw_rate` gives bit-identical v8F-mini windows and RA-1 clips/losses in `yaw`, different ones in `steer` (control);
* deliberate-regression arms (loader re-reads steer) go RED; **11 SOURCE mutations** all turn a guard RED (`raw/mutate_yaw_guards.json`, rerun: `code/mutate_yaw_guards.py`);
* refusals: wrong-units prior, missing `--tracks`, a foreign track, mixing steer and yaw in one `verdict`; the banked P10-yaw artifacts equal what the loader computes (≤ 1e-3 m).

## Run it (box 2)
Copy `native_tracks.pt` next to `D:\tanitad\cache` (sha256 `1db431f112b1bd7c426073c43c7df5fc75b2a032736c1e2020f02ffb118669e7`), then `code\ra1_box2_queue.cmd gate` (writes to `D:\tanitad\runs\ra1_yaw`, never the steer runs' directory). v8F-mini: `v8f_mini_train.py --tracks <native_tracks.pt>` (`--kin-mode yaw` is the default; `--p10` defaults to `raw/p10_yaw`).
The 2026-10-17 22:35 steer runs are NON-DECIDING (representation-equivalence check only); `verdict` refuses to mix them.
