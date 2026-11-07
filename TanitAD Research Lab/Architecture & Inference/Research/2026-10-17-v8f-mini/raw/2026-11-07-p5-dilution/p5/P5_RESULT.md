# P5 result: regression dilution of the acceleration input (THEORY_CRAWL_AMEND_3.md)
Written 2026-11-07 03:32:51 (dev-box clock). Card sha256 `dcb82ae94c22ffbeb1353704ae6951f1fce2143eebe72d7b5544a4a1fce7f394` verified before and after; the card is not edited. Tip `1f6a0f1f2dc8`. Dev-box CPU only, zero training, no eval-141 window read, no checkpoint loaded.
Evidence tags: [M] measured here (artifact under `dagland/motion_c/theory/p5/`), [E] estimated, [H] hypothesis; [M-inh] a measured number inherited from `wsp_read/WSP_READ.md`, not re-measured here.

## Verdict: **P5 FAILS**
| clause of the card's section 3 | reading | met |
|---|---|---|
| a0 gate (max abs diff <= 1e-4 m/s^2 on >= 1,000 windows) | 0.0 on 1,500 windows from 1,077 clips; 0.0 on all 87,942 in-band windows [M] | yes |
| >= 500 episodes | 2,586 [M] | yes |
| upper bound of the lambda interval < 0.90 | lambda = **0.9680**, 95 % interval [0.9659, 0.9698] [M] | **no** (0.9698 >= 0.90) |
| gain ratio 0.666 (s0) within lambda +- 0.10 | |0.666 - 0.968| = 0.302 [M] | **no** |
| gain ratio 0.586 (s1) within lambda +- 0.10 | |0.586 - 0.968| = 0.382 [M] | **no** |

P5 FAILS on all three clauses. The input noise is too small to make the shrinkage: the 0.1 s difference carries 3.2 % of the (residualised) variance of a0 as error vs a +-0.5 s fit of the same log. Under classical errors-in-variables the base gain 1.899 would fall to 1.838 [E: lambda x 1.899]; the weights show 1.265 (s0) and 1.112 (s1) [M-inh], so dilution accounts for 0.061 of a shortfall of 0.634 (s0) / 0.787 (s1) in ds/da0, i.e. 10 % / 8 % [E: arithmetic on the numbers above]. Every sensitivity below gives the same word.

## 1. The gate: how v8F computes a0 (source quotes), and a finding about the card's wording
* The C0 arms ran `--kin-mode yaw --tracks native_tracks.pt` (`v8f-recover/M2b_s0_recft_C0/argv.json`; `summary.json` levers.kin_mode "yaw") [M].
* In that mode the data path is `stack/tanitad/data/v8f_mini_data.py` (tip): `ClipStore.get`, line 474: `kin3, omega = yaw_kin_from_track(self.tracks[clip])`; line 498: `p10[1:] = _fk().kin_features(kin3[1:])` (the 8 model inputs: v, a, k, v^2, v*a, v*k, v^2*k, a*k, `p10_prior.py:142`); `yaw_kin_from_track` (line 333) takes `ego_channels_native(track)[0::2]` (line 323), which calls `ego_history.ego_channels_from_poses(..., dt=DT_NATIVE)` with `DT_NATIVE = 0.1` (line 320); `ego_history.py:98`: `dv[:, 1:] = (v[:, 1:] - v[:, :-1]) / float(dt)`. The model reads it at `v8f_mini.py:698` (`e_ego = Linear(8, d)`) and `:738` (`_ego(f8) = e_ego(prior.standardise(f8))`).
* **So the model's a0 is the ONE-frame backward difference of the native 10 Hz speed column, i.e. a 0.1 s difference (v at raw frame 2t minus v at 2t-1, over 0.1 s). It is not a 0.2 s difference.** The 0.2 s form `(v[2t] - v[2t-2]) / 0.2` is the `steer`-mode cache `kin3` (`window_kin`, same file), which the C0 arms do not use. The card's section 1 ("a0 is the 0.2 s backward difference", copied from `WSP_READ.md` line 31) and section 2 ("a 0.2 s difference") carry the 0.2 s wording; the gate clause ("reproduce the model's own a0") is what decides, and it passes on the 0.1 s definition only. The tip and the dagt1 tree (the source that trained C0) differ in this file only by the lead-label arrays, `ClipStore.get_kin` and the `EgoPerturb` stream/composition code (off in the C0 arms: `--ego-perturb-p 0`); `yaw_kin_from_track`, `ClipStore.get` and `V8FMiniWindows.item` are identical (all 15 diff hunks read), and `ego_history.py`, `kinematic_prior.py`, `p10_prior.py` are identical modulo line endings [M].
* Logged sampling rate: the native track is 10 Hz (median displacement / (speed x 0.1 s) = 1.0067 over 598 clips, 5-95 % [1.0050, 1.0095]); the training cache keeps every second frame (5 Hz). `native_tracks.pt` sha256 `1db431f112b1bd7c...` [M].
* Reproduction (`p5_gate.py`, `raw/P5_GATE.json`): the real `ClipStore(kin_mode="yaw")` + `V8FMiniWindows.item` on 1,500 seeded windows (seed 1, 1,077 clips) against a fresh float64 re-implementation from the track's speed column:
| candidate a0 definition | max abs diff vs the model's a0 (m/s^2) | gate |
|---|---|---|
| one native frame, 0.1 s (what the source computes) | 0 | **pass** |
| two native frames, 0.2 s (the card's wording) | 0.663 | fail |
| the cache's steer-mode `kin3` (0.2 s on the 5 Hz grid) | 0.663 | fail |
| forward difference (mutant) | 2.106 | fail |
(`kin3[1]` equals `p10f[-1][1]` bit for bit; the 0.0 is exact because the track is float32 and a difference of two nearby float32 values times 10 is representable [M].)

## 2. The window set
TRAIN windows of the base's 4,512 clips (the banked 306,566-window list, sha256 `e144adcd9b6273f2...` rebuilt and asserted; `--exclude 540:600`, window_start 3), v0 (= `kin3[0]`) in [3, 8): 87,942 windows from 2,687 clips [M, equals wsp_edit's pool 87,942]. 2,223 of them sit at cache steps t = 3, 4 where the +-0.8 s fit would reach before the track starts (the first two track rows are NaN); they are dropped so that the three fit widths run on identical windows, leaving a pool of 85,719. **A seeded sample of 20,000 windows from 2,586 episodes**: rule `np.sort(np.random.default_rng(0).choice(pool, 20000, replace=False))`, seed 0, list digest sha256 `27cab227116e2e1e98c9ab24f719fd6670774569fd99db7e0c8d1fd12912a466` (`raw/p5_window_ids.txt`, one `clip:t` per line, LF) [M]. Mean v0 5.719 m/s. Eval-141 is not read [M].

## 3. The estimator of a (label side, future allowed)
Local quadratic (Savitzky-Golay, order 2) fit of the logged speed over +-0.5 s (11 native frames) around the present raw frame 2t, derivative at the centre: `a = sum_k k v[2t+k] / (0.1 x sum k^2)`. Checked against `np.polyfit` per window and `scipy.signal.savgol_filter(deriv=1)` to 1e-9 (`test_p5.py`). +-0.3 s (7 frames) and +-0.8 s (17 frames) run beside.

## 4. lambda
lambda = 1 - Var(e_perp) / Var(a0_perp), e = a0 - a, perp = OLS residual on **[1, v0, kappa0]**: the two other entries of the model's ego triple at the present tick (the 8 p10 features are functions of v0, a0, kappa0; the curvature is `kappa0 = clamp(omega0 / max(v0, 2), +-0.3)` from the measured yaw rate, the PI ruling that replaced the steer channel). Var(a0_perp) = 0.9766, Var(e_perp) = 0.0313, Var(a_perp) = 0.9148, corr(e_perp, a_perp) = 0.090 [M]. **lambda = 0.9680 [0.9659, 0.9698]** (episode-cluster bootstrap over 2,586 episodes, B 2,000, seed 0, `ci.py` `stat_cluster_bootstrap`; a second, Gram-matrix implementation with the same draws gives the identical interval). This interval answers "would another draw of episodes say this?"; no model is trained or run, so there is no training or inference variance in it. Another 20,000-window draw from the same pool moves the point by sd 0.0007 (8 seeds, range 0.9672 to 0.9694) [M].

## 5. Sensitivities (decide nothing)
| variant | lambda | 95 % interval | windows / episodes | word it would give |
|---|---|---|---|---|
| primary: +-0.5 s, [1, v0, kappa0] | 0.9680 | [0.9659, 0.9698] | 20,000 / 2,586 | FAILS |
| +-0.3 s fit | 0.9784 | [0.9769, 0.9797] | 20,000 / 2,586 | FAILS |
| +-0.8 s fit | 0.9568 | [0.9542, 0.9592] | 20,000 / 2,586 | FAILS |
| perp on every p10 feature without a0 (v, k, v^2, v*k, v^2*k) | 0.9680 | [0.9659, 0.9698] | 20,000 / 2,586 | FAILS |
| all 85,719 valid in-band windows (no sampling) | 0.9678 | [0.9662, 0.9694] | 85,719 / 2,667 | FAILS |
| v0 in [3, 5.5) | 0.9712 | [0.9689, 0.9734] | 8,664 / 1,898 | FAILS |
| v0 in [5.5, 8) | 0.9646 | [0.9615, 0.9675] | 11,336 / 2,182 | FAILS |
| a0 as the card words it (0.2 s difference; NOT the model input) | 0.9740 | [0.9723, 0.9757] | 20,000 / 2,586 | FAILS |
| MUTANT: no residualisation | 0.9679 | [0.9659, 0.9698] | 20,000 / 2,586 | FAILS |
| MUTANT: forward-difference a0 | 0.9660 | [0.9637, 0.9681] | 20,000 / 2,586 | FAILS |
Across every reading lambda lies in [0.957, 0.978] [M]. A wider fit window lowers it (0.978, 0.968, 0.957 at +-0.3, 0.5, 0.8 s), because the target is smoother, and every reading stays far above the 0.686 that a HOLDS would need at the top of the window both seeds allow.

## 6. Controls (`p5_controls.py`, `raw/P5_CONTROLS.json`; 11 tests in `test_p5.py`, all pass)
1. **Known noise returns its known lambda.** Synthetic series (20,000 one-window episodes; speed linear in time plus white noise sigma 0.06 m/s; true acceleration correlated with v0 so the residualisation matters; population lambda from the noise coefficient vectors, no simulation): expected 0.4593, observed 0.4593 [0.4492, 0.4688], abs error 0.00005, expected value inside the interval [M].
2. **a0 replaced by a gives lambda = 1**: synthetic 1.000000000000, real sample 1.000000000000 [M].
3. **Deliberate-regression mutants change the number and a test catches them.** On the synthetic world, against the known value 0.4593: no residualisation 0.6137 (miss 0.154, caught), forward-difference a0 0.4887 (miss 0.029, caught, and caught by the gate), 0.2 s a0 0.7855 (miss 0.326, caught, and caught by the gate); a +-0.3 s instead of +-0.5 s fit moves it by 0.008, inside the 0.015 tolerance, so that one is NOT caught by the known-value check [M]. On the real sample the mutants move lambda by -0.00003 (no residualisation: v0 and kappa0 explain almost none of the variance of a0 here), -0.0019 (forward), +0.0061 (0.2 s) and +0.0104 (+-0.3 s), and none of them changes the word. The gate catches the forward and 0.2 s definitions with max abs diffs of 3.01 and 1.46 m/s^2 over all 87,942 in-band windows [M].
4. The Gram-matrix bootstrap equals `ci.py`'s on identical draws (1e-9), the verdict function is tested at its boundaries, and the SG estimator does not wrap a negative index at the clip start (NaN instead) [M].

## 7. What the card's section 4 sends next
**FAILS:** dilution does not explain the shrinkage. The card commits: the remaining explanation inside the weights is a learned response to the data, intent or the frontier, and the perception lever (LoRA, P4) carries the crawl alone, as T2 already ranks it. The "M2b with a denoised acceleration input" arm is not written. Nothing in this read asks for it: the error variance a denoised input could remove is 3.2 % of Var(a0_perp).

**Beside, post hoc, decides nothing, the cheapest discriminating measurement for the remaining explanations (`p5_next_lever.py`, `raw/P5_NEXT_LEVER.json`):** the data's own least-squares gain of the realised plan speed s_gt (secant of the GT future, knots 8 to 12) on a0, given v0 and kappa0, on the same 20,000 TRAIN windows: **ds/da0 = 1.384 [1.360, 1.408]** and ds/dv0 = 0.936 [0.919, 0.952] (R^2 0.783) [M]. The non-learned base applies 1.899; the data's optimum is 1.38; the learned model sits at 1.265 / 1.112 [M-inh]. Reading [H]: most of the shrinkage (1.899 to about 1.38) is the data's own mean reversion of acceleration (the unresolved-intent side of the card's remaining explanations), and only the last 0.12 (s0) / 0.27 (s1) is the model's additional under-response. Caveat: the OLS gain is a population linear fit on train windows and the model's gains are medians of local Jacobians on 586 held-out windows, two different estimands; this is a pointer for the next card, not a test.

**Beside, decides nothing: R^2 of a on the frozen pooled tokens** (`p5_ridge.py`, `raw/P5_RIDGE.json`; the train cache's `pool4x10` tokens at the present frame, ridge fit on train folds only, 5 episode-grouped folds, out of fold; tokens are on the dev box for all 20,000 windows): mean token (1,024-d) R^2 = 0.112 [0.097, 0.128], four row-means of the 4 x 10 grid (4,096-d) R^2 = 0.158 [0.143, 0.174]; for reference R^2 of a0 itself 0.098 / 0.135; the part of a that a0 does not carry (a - 0.952 a0) 0.012 [0.008, 0.016] / 0.022 [0.017, 0.027]; shuffled-target control -0.0004 / -0.0009 (must be about 0) [M]. A single frame carries a little of a (0.11 to 0.16) and almost nothing that a0 lacks (0.01 to 0.02), consistent with T3. The R^2 depends on the penalty rule: a first RidgeCV (GCV, alphas 1 to 1e5) attempt on the mean token read 0.086 for a and 0.072 for a0 before it was stopped for speed (`logs/ridge_attempt1_killed.log`); the reported numbers use an inner grouped hold-out with alphas up to 1e6. The row-mean grouping assumes token index = row x 10 + column (UNVERIFIED against the cache builder; the mean-token numbers do not depend on it).

## 8. Scope and what this read cannot see
* lambda compares a 0.1 s difference with a +-0.5 s fit of the SAME logged speed. An error common to both (for example a lag or smoothing built into the egomotion speed estimate) cancels in e and is invisible here [H]. The widest fit tried, +-0.8 s, still gives lambda 0.957; the P5 clauses are read as written.
* The gain ratios 0.666 / 0.586 are [M-inh] from `wsp_read/WSP_READ.md` lines 31-44 (held-out eval-141, v0 in 3-8 m/s); lambda here is on TRAIN windows, as the card specifies.
* The card's "0.2 s" wording (section 1 above): if the 0.2 s difference had been the input, lambda would be 0.974 and the word the same.

## 9. Deliverable manifest (nothing staged, committed or pushed; everything exists in ONE place, `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/p5/`)
See the table in the hand-back and `MANIFEST.txt` (sha256 of every file).
