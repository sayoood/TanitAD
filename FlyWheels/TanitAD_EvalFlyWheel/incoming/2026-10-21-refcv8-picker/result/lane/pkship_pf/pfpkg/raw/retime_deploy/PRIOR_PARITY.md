# PRIOR PARITY -- SHIP-R8 AMENDMENT 2 clause (7)

Generated 2026-10-22T06:21:48 by `code/prior_parity_r8.py`. CPU only, no GPU lock, no network forward. Launch stack `C:\lgs\Lfb47a58376\stack` (fb47a58).

**READING: PASS (model-path parity)** -- max |model path - eval rebuilt| = 0.000e+00 m over all 8 slots, 9508 window rows in 2 panels (>= 4754 per panel; the bar is 0.001 m on >= 200).

## 1. Does `residual_prior_path` depend on the model's WEIGHTS?  No -- inputs only

INPUT-ONLY: residual_prior_path = prior_path(a0, kappa0, v, anchor_horizons, anchor_dt) with (a0, kappa0) from prior_controls(ego_poses, ego_n_past) and v = the v0 argument; every other quantity is a configuration scalar or list (EgoHistoryConfig.dt, anchor_dt, anchor_horizons, anchor_withheld_bank). No nn.Module, parameter or buffer is read; the lateral dropout is training-only. => it can be computed with the model's own methods WITHOUT the network.

| file:line | quote | meaning | verified |
|---|---|---|---|
| `stack/tanitad/refs/refc.py:5136` | `_rp_in = self._residual_prior_inputs(ego_poses, n_past,` | RefCModel.forward hands the prior ONLY forward ARGUMENTS: ego_poses, n_past (= ego_n_past), ego_actions, v0 | yes |
| `stack/tanitad/refs/refc.py:4530` | `a0, k0 = _kp.prior_controls(mode, ego_poses, n_past, acts,` | (a0, kappa0) are a pure function of the pose window; dt = self.ego_hist.cfg.dt, a CONFIG scalar (EgoHistoryConfig.dt = 0.1) | yes |
| `stack/tanitad/models/kinematic_prior.py:311` | `ch = ego_channels_from_poses(poses, n, dt=float(dt))` | backward differences of the pose window -- no module, no parameter | yes |
| `stack/tanitad/models/kinematic_prior.py:341` | `kappa0 = (omega0 / v0.clamp_min(POSE_KAPPA_V_FLOOR)).clamp(` | kappa0 = omega0 / max(v0, 2), clamped to +-0.3: arithmetic on the window | yes |
| `stack/tanitad/refs/refc.py:3437` | `_rp = self._residual_prior(residual_prior, v_ms, ego_keep, b,` | the decoder composes (a0, kappa0, v) once per forward | yes |
| `stack/tanitad/refs/refc.py:2345` | `v = self._prior_speed(v_ms, ego_keep, withheld_speed)` | v = the v0 ARGUMENT (kept rows); a withheld row (ego_keep False) would be rolled at the bank's withheld speed -- at eval ego_keep is None | yes |
| `stack/tanitad/refs/refc.py:2346` | `a0, k0 = _kp.withhold(a0, k0, ego_keep)` | identity when ego_keep is None | yes |
| `stack/tanitad/refs/refc.py:2350` | `if (self.r8_cfg is not None and self.training` | the lateral prior dropout is TRAINING-only (`self.training`) | yes |
| `stack/tanitad/refs/refc.py:3952` | `out["residual_prior_path"] = _kp.prior_path(` | the emitted tensor is prior_path(a0, kappa0, v, anchor_horizons, tick=anchor_dt) | yes |
| `stack/tanitad/refs/refc.py:3954` | `tick=self.anchor_dt).to(x.dtype).detach()` | cast to the fan's dtype (float32 at eval: the banked plans are full float32, see plan_dtype_evidence), detached | yes |
| `stack/tanitad/models/kinematic_prior.py:387` | `path = rollout_unicycle(state0, ctrl, dt=float(tick))[..., :2]` | a constant-control unicycle roll from a zero state with v in slot 3 | yes |
| `stack/tanitad/refs/refc.py:1748` | `self.anchor_dt = 0.1` | a constructor constant, not a parameter or buffer | yes |
| `stack/tanitad/refs/refc.py:1769` | `self.anchor_horizons: tuple[int, ...] = tuple(int(k) for k in _h)` | a plain attribute (refc.py:1764-1768 says so): the configured horizons, here [5, 10, 15, 20, 30, 40, 50, 60] | yes |
| `stack/tanitad/models/kinematic.py:260` | `x = x + v * torch.cos(yaw) * dt` | rollout_unicycle: elementwise torch arithmetic only (kinematic.py:220-266) | yes |
| `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code/r7_roll.py:154` | `ph = it["pose_hist"].to(dev)` | the battery roll feeds the model ego_poses = the dataset item's pose_hist ... | yes |
| `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code/r7_roll.py:162` | `"ego_n_past": int(ph.shape[0]),` | ... and n_past = its full length | yes |
| `stack/scripts/refc_v3_train.py:4533` | `item["pose_hist"] = ep.poses[t:t + w]` | pose_hist = ep.poses[t : t+w] when pose-sync is OFF -- the banked decisions' `ep_poses` IS ep.poses (taniteval/tools/refcv3_arm.py:2582) | yes |
| `stack/scripts/refc_v3_train.py:4506` | `item["pose_hist"] = pv[t:t + w]` | with pose-sync ON (corrected clock) pose_hist is the RE-SAMPLED pv, which `ep_poses` does not carry (see caveats) | yes |
| `taniteval/tools/refcv3_arm.py:2582` | `ep_poses=ep.poses.float().cpu().numpy()` | the banked window source is the raw provider poses | yes |

## 2. Known-value controls (literals typed by hand)

A 10 m/s straight-line history must give x = 5, 10, 15, 20, 30, 40, 50, 60 m; a standing history zeros; a history ending 9.9 -> 10.0 m/s (a0 = +1 m/s^2) x = [5.1, 10.45, 16.05, 21.9, 34.35, 47.8, 62.25, 77.7].  Tolerance 0.001 m.

| path | max abs error (m) | PASS |
|---|---|---|
| model_path | 6.561e-05 | True |
| rebuilt | 6.561e-05 | True |
| independent_f64 | 6.752e-05 | True |

Mutations (each must FAIL):

| mutation | max abs error (m) | detected |
|---|---|---|
| model_path_wrong_tick_0p2 | 113 | True |
| rebuilt_wrong_horizons | 7.85 | True |
| independent_wrong_tick_0p2 | 113 | True |

## 3. Parity on banked eval windows

model path = the model's own methods (stub self, no weights); rebuilt = the eval's `references()` construction; indep = a float64 numpy derivation.

| tree | seed | n | max model-vs-rebuilt (m) | mean (m) | bit-equal windows | max model-vs-indep f64 (m) | vs DEPLOY8 npz prior8 max (m) |
|---|---|---|---|---|---|---|---|
| pkship_pf | 0 | 4754 | 0.000e+00 | 0.000e+00 | 4754 | 2.134e-03 | 0.000e+00 |
| pkship_pf | 1 | 4754 | 0.000e+00 | 0.000e+00 | 4754 | 2.134e-03 | 0.000e+00 |

The float64 column is a cross-check of the ALGORITHM, not part of the clause: float32 accumulation over 60 steps differs from exact arithmetic by up to ~2e-3 m at 6 s, which is common to both float32 paths (same op order) and therefore absent from the model-vs-rebuilt column.
Per-slot max (m), pkship_pf s0: 0.5s 0.00e+00, 1s 0.00e+00, 1.5s 0.00e+00, 2s 0.00e+00, 3s 0.00e+00, 4s 0.00e+00, 5s 0.00e+00, 6s 0.00e+00

## 4. Caveats (what this does NOT show)

- SCOPE: the model's own methods were run on CPU float32 against the banked windows; the tensor `out['residual_prior_path']` of an actual forward was NOT observed (the banked dumps carry no residual_prior_* key). What that leaves open: (i) GPU-vs-CPU float32 cos/sin: both devices run the SAME float32 op sequence (x = x + v*cos(yaw)*dt ...), only the libm cos/sin may differ by an ulp; one float32 ulp of x at 60 m is 3.8e-6 m, so even a 1-ulp flip on EVERY one of 60 steps is <= 2.3e-4 m < 1e-3 m (an ANALYTIC bound, not a measurement), (ii) that the roll fed the model exactly ep.poses[t:t+w] (source-traced above, and the window / v0 / pose_last consistency checks pass).
- CLOCK: parity holds on a tree rolled with pose-sync OFF (the `_unc` trees: R8_NO_POSE_SYNC=1; battery_summary.json r8.pose_sync_policy). With pose-sync ON (corrected clock, `_corr` trees) the model's window is the re-sampled pv (refc_v3_train.py:4506) while the banked decisions carry the raw ep.poses, so the eval-rebuilt prior would NOT equal the model's -- clause (7) must not be read across clocks.
- TO CLOSE (i)+(ii) AT 48,852: have the roll dump out['residual_prior_path'][:, slots] beside `os` (a one-line extra in the dump sidecar) and compare to the rebuilt array; or run one CPU forward on >= 200 windows (needs the checkpoint + the eval frames, ~6 GB commit).
