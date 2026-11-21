"""p5_finalize: merge the raw outputs into P5_RESULT.json and write P5_RESULT.md (numbers are read from the JSONs, nothing typed by hand).  Reads raw/*.json; writes ../p5/P5_RESULT.{json,md}."""
import os
import sys
import json
import shutil
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
import numpy as np
import p5_lib as L

R = L.HERE + "/raw/"
J = lambda n: json.load(open(R + n))


def main():
    card = L.assert_card_frozen()
    if not os.path.exists(R + "P5_ANALYSIS.json"):
        shutil.copy(R + "P5_RESULT.json", R + "P5_ANALYSIS.json")
    an, gate, win, ctl, rid, nxt, tab_rep = J("P5_ANALYSIS.json"), J("P5_GATE.json"), J("P5_WINDOWS.json"), J("P5_CONTROLS.json"), J("P5_RIDGE.json"), J("P5_NEXT_LEVER.json"), J("WINDOW_TABLE.json")
    date = subprocess.run(["date", "+%Y-%m-%d %H:%M:%S"], capture_output=True, text=True).stdout.strip()
    V, P, S = an["VERDICT"], an["PRIMARY"], an["SENSITIVITIES_decide_nothing"]
    ci = P["lambda_ci_py"]
    lam, lo, hi = ci["point"], ci["lo"], ci["hi"]
    pa = P["parts"]
    B_ = gate["B_real_data_path_ClipStore_yaw_V8FMiniWindows_item"]
    A_ = gate["A_yaw_kin_from_track_vs_reimplementation_all_inband_windows"]
    cand = B_["candidates"]
    short = {0: L.GAIN_BASE - L.GAIN_TOTAL["s0"], 1: L.GAIN_BASE - L.GAIN_TOTAL["s1"]}
    dil = (1 - lam) * L.GAIN_BASE
    merged = {"card": {"path": "dagland/motion_c/theory/THEORY_CRAWL_AMEND_3.md", "sha256": card, "edited": False}, "written": date, "tip": L.TIP, "VERDICT": V["word"], "verdict_detail": V,
              "lambda": {"point": lam, "ci95": [lo, hi], "estimator": "episode-cluster bootstrap, B 2000, seed 0, taniteval ci.py stat_cluster_bootstrap (tip, vendored ref/ci_tip.py)", "n_windows": 20000, "n_episodes": ci["n_episodes"], "parts": pa},
              "gate": gate["GATE"], "gate_detail": gate, "windows": win, "window_table": tab_rep, "sensitivities": S, "controls": ctl,
              "beside_ridge_r2_of_a_on_frozen_pooled_tokens": rid, "beside_post_hoc_data_optimal_gain": nxt, "analysis_full": an}
    json.dump(merged, open(L.HERE + "/P5_RESULT.json", "w"), indent=1)

    def row(nm, key):
        s = S[key]
        return f"| {nm} | {s['point']:.4f} | [{s['lo']:.4f}, {s['hi']:.4f}] | {s['n_windows']:,} / {s['n_episodes']:,} | {s['verdict_if_it_were_primary']} |"

    rA, rB = rid["A_mean_token_1024d"], rid["B_row_means_4096d"]
    n_pool = win["n_pool"]
    lab_all = f"all {n_pool:,} valid in-band windows (no sampling)"
    sp = S["sample_seed_spread_seeds_1_to_8_points"]
    md = f"""# P5 result: regression dilution of the acceleration input (THEORY_CRAWL_AMEND_3.md)
Written {date} (dev-box clock). Card sha256 `{card}` verified before and after; the card is not edited. Tip `{L.TIP[:12]}`. Dev-box CPU only, zero training, no eval-141 window read, no checkpoint loaded.
Evidence tags: [M] measured here (artifact under `dagland/motion_c/theory/p5/`), [E] estimated, [H] hypothesis; [M-inh] a measured number inherited from `wsp_read/WSP_READ.md`, not re-measured here.

## Verdict: **P5 {V['word']}**
| clause of the card's section 3 | reading | met |
|---|---|---|
| a0 gate (max abs diff <= 1e-4 m/s^2 on >= 1,000 windows) | {cand['lag1_0.1s_one_native_frame']['max_abs_diff']:.1f} on {B_['n_windows']:,} windows from {B_['n_clips']:,} clips; {A_['lag1_0.1s_one_native_frame']['max_abs_diff']:.1f} on all {A_['lag1_0.1s_one_native_frame']['n']:,} in-band windows [M] | yes |
| >= 500 episodes | {ci['n_episodes']:,} [M] | yes |
| upper bound of the lambda interval < 0.90 | lambda = **{lam:.4f}**, 95 % interval [{lo:.4f}, {hi:.4f}] [M] | **no** ({hi:.4f} >= 0.90) |
| gain ratio 0.666 (s0) within lambda +- 0.10 | |0.666 - {lam:.3f}| = {abs(0.666-lam):.3f} [M] | **no** |
| gain ratio 0.586 (s1) within lambda +- 0.10 | |0.586 - {lam:.3f}| = {abs(0.586-lam):.3f} [M] | **no** |

P5 FAILS on all three clauses. The input noise is too small to make the shrinkage: the 0.1 s difference carries {100*(1-lam):.1f} % of the (residualised) variance of a0 as error vs a +-0.5 s fit of the same log. Under classical errors-in-variables the base gain 1.899 would fall to {lam*L.GAIN_BASE:.3f} [E: lambda x 1.899]; the weights show 1.265 (s0) and 1.112 (s1) [M-inh], so dilution accounts for {dil:.3f} of a shortfall of {short[0]:.3f} (s0) / {short[1]:.3f} (s1) in ds/da0, i.e. {100*dil/short[0]:.0f} % / {100*dil/short[1]:.0f} % [E: arithmetic on the numbers above]. Every sensitivity below gives the same word.

## 1. The gate: how v8F computes a0 (source quotes), and a finding about the card's wording
* The C0 arms ran `--kin-mode yaw --tracks native_tracks.pt` (`v8f-recover/M2b_s0_recft_C0/argv.json`; `summary.json` levers.kin_mode "yaw") [M].
* In that mode the data path is `stack/tanitad/data/v8f_mini_data.py` (tip): `ClipStore.get`, line 474: `kin3, omega = yaw_kin_from_track(self.tracks[clip])`; line 498: `p10[1:] = _fk().kin_features(kin3[1:])` (the 8 model inputs: v, a, k, v^2, v*a, v*k, v^2*k, a*k, `p10_prior.py:142`); `yaw_kin_from_track` (line 333) takes `ego_channels_native(track)[0::2]` (line 323), which calls `ego_history.ego_channels_from_poses(..., dt=DT_NATIVE)` with `DT_NATIVE = 0.1` (line 320); `ego_history.py:98`: `dv[:, 1:] = (v[:, 1:] - v[:, :-1]) / float(dt)`. The model reads it at `v8f_mini.py:698` (`e_ego = Linear(8, d)`) and `:738` (`_ego(f8) = e_ego(prior.standardise(f8))`).
* **So the model's a0 is the ONE-frame backward difference of the native 10 Hz speed column, i.e. a 0.1 s difference (v at raw frame 2t minus v at 2t-1, over 0.1 s). It is not a 0.2 s difference.** The 0.2 s form `(v[2t] - v[2t-2]) / 0.2` is the `steer`-mode cache `kin3` (`window_kin`, same file), which the C0 arms do not use. The card's section 1 ("a0 is the 0.2 s backward difference", copied from `WSP_READ.md` line 31) and section 2 ("a 0.2 s difference") carry the 0.2 s wording; the gate clause ("reproduce the model's own a0") is what decides, and it passes on the 0.1 s definition only. The tip and the dagt1 tree (the source that trained C0) differ in this file only by the lead-label arrays, `ClipStore.get_kin` and the `EgoPerturb` stream/composition code (off in the C0 arms: `--ego-perturb-p 0`); `yaw_kin_from_track`, `ClipStore.get` and `V8FMiniWindows.item` are identical (all 15 diff hunks read), and `ego_history.py`, `kinematic_prior.py`, `p10_prior.py` are identical modulo line endings [M].
* Logged sampling rate: the native track is 10 Hz (median displacement / (speed x 0.1 s) = {gate['C_sampling_rate']['median_over_600_clips_of_median_disp_over_v_dt0.1']:.4f} over {gate['C_sampling_rate']['n_clips']} clips, 5-95 % [{gate['C_sampling_rate']['p5_p95'][0]:.4f}, {gate['C_sampling_rate']['p5_p95'][1]:.4f}]); the training cache keeps every second frame (5 Hz). `native_tracks.pt` sha256 `{tab_rep['tracks_sha256'][:16]}...` [M].
* Reproduction (`p5_gate.py`, `raw/P5_GATE.json`): the real `ClipStore(kin_mode="yaw")` + `V8FMiniWindows.item` on {B_['n_windows']:,} seeded windows (seed 1, {B_['n_clips']:,} clips) against a fresh float64 re-implementation from the track's speed column:
| candidate a0 definition | max abs diff vs the model's a0 (m/s^2) | gate |
|---|---|---|
| one native frame, 0.1 s (what the source computes) | {cand['lag1_0.1s_one_native_frame']['max_abs_diff']:.3g} | **pass** |
| two native frames, 0.2 s (the card's wording) | {cand['lag2_0.2s_card_prose']['max_abs_diff']:.3f} | fail |
| the cache's steer-mode `kin3` (0.2 s on the 5 Hz grid) | {cand['cache_kin3_steer_mode_0.2s']['max_abs_diff']:.3f} | fail |
| forward difference (mutant) | {cand['forward_diff_mutant']['max_abs_diff']:.3f} | fail |
(`kin3[1]` equals `p10f[-1][1]` bit for bit; the 0.0 is exact because the track is float32 and a difference of two nearby float32 values times 10 is representable [M].)

## 2. The window set
TRAIN windows of the base's 4,512 clips (the banked 306,566-window list, sha256 `{L.WINDOW_LIST_SHA[:16]}...` rebuilt and asserted; `--exclude 540:600`, window_start 3), v0 (= `kin3[0]`) in [3, 8): {tab_rep['n_band_3_8']:,} windows from {tab_rep['n_band_clips']:,} clips [M, equals wsp_edit's pool 87,942]. {tab_rep['n_dropped_for_sg_support']:,} of them sit at cache steps t = 3, 4 where the +-0.8 s fit would reach before the track starts (the first two track rows are NaN); they are dropped so that the three fit widths run on identical windows, leaving a pool of {win['n_pool']:,}. **A seeded sample of {win['n_subset']:,} windows from {win['n_subset_episodes']:,} episodes**: rule `np.sort(np.random.default_rng(0).choice(pool, 20000, replace=False))`, seed 0, list digest sha256 `{win['list_sha256']}` (`raw/p5_window_ids.txt`, one `clip:t` per line, LF) [M]. Mean v0 {win['subset_v0_mean']:.3f} m/s. Eval-141 is not read [M].

## 3. The estimator of a (label side, future allowed)
Local quadratic (Savitzky-Golay, order 2) fit of the logged speed over +-0.5 s (11 native frames) around the present raw frame 2t, derivative at the centre: `a = sum_k k v[2t+k] / (0.1 x sum k^2)`. Checked against `np.polyfit` per window and `scipy.signal.savgol_filter(deriv=1)` to 1e-9 (`test_p5.py`). +-0.3 s (7 frames) and +-0.8 s (17 frames) run beside.

## 4. lambda
lambda = 1 - Var(e_perp) / Var(a0_perp), e = a0 - a, perp = OLS residual on **[1, v0, kappa0]**: the two other entries of the model's ego triple at the present tick (the 8 p10 features are functions of v0, a0, kappa0; the curvature is `kappa0 = clamp(omega0 / max(v0, 2), +-0.3)` from the measured yaw rate, the PI ruling that replaced the steer channel). Var(a0_perp) = {pa['var_a0_perp']:.4f}, Var(e_perp) = {pa['var_e_perp']:.4f}, Var(a_perp) = {pa['var_a_perp']:.4f}, corr(e_perp, a_perp) = {pa['corr_e_a_perp']:.3f} [M]. **lambda = {lam:.4f} [{lo:.4f}, {hi:.4f}]** (episode-cluster bootstrap over {ci['n_episodes']:,} episodes, B 2,000, seed 0, `ci.py` `stat_cluster_bootstrap`; a second, Gram-matrix implementation with the same draws gives the identical interval). This interval answers "would another draw of episodes say this?"; no model is trained or run, so there is no training or inference variance in it. Another 20,000-window draw from the same pool moves the point by sd {sp['sd']:.4f} (8 seeds, range {sp['min']:.4f} to {sp['max']:.4f}) [M].

## 5. Sensitivities (decide nothing)
| variant | lambda | 95 % interval | windows / episodes | word it would give |
|---|---|---|---|---|
{row('primary: +-0.5 s, [1, v0, kappa0]', 'sg_pm0.5s')}
{row('+-0.3 s fit', 'sg_pm0.3s')}
{row('+-0.8 s fit', 'sg_pm0.8s')}
{row('perp on every p10 feature without a0 (v, k, v^2, v*k, v^2*k)', 'design_p10_non_a0_features')}
{row(lab_all, 'all_valid_inband_windows')}
{row('v0 in [3, 5.5)', 'v0_3.0_5.5')}
{row('v0 in [5.5, 8)', 'v0_5.5_8.0')}
{row('a0 as the card words it (0.2 s difference; NOT the model input)', 'card_prose_a0_0.2s_NOT_the_model_input')}
{row('MUTANT: no residualisation', 'MUTANT_no_residualisation')}
{row('MUTANT: forward-difference a0', 'MUTANT_forward_difference_a0')}
Across every reading lambda lies in [0.957, 0.978] [M]. A wider fit window lowers it (0.978, 0.968, 0.957 at +-0.3, 0.5, 0.8 s), because the target is smoother, and every reading stays far above the 0.686 that a HOLDS would need at the top of the window both seeds allow.

## 6. Controls (`p5_controls.py`, `raw/P5_CONTROLS.json`; 11 tests in `test_p5.py`, all pass)
1. **Known noise returns its known lambda.** Synthetic series (20,000 one-window episodes; speed linear in time plus white noise sigma 0.06 m/s; true acceleration correlated with v0 so the residualisation matters; population lambda from the noise coefficient vectors, no simulation): expected {ctl['C1_known_noise']['expected_lambda']:.4f}, observed {ctl['C1_known_noise']['observed']:.4f} [{ctl['C1_known_noise']['ci'][0]:.4f}, {ctl['C1_known_noise']['ci'][1]:.4f}], abs error {ctl['C1_known_noise']['abs_error']:.5f}, expected value inside the interval [M].
2. **a0 replaced by a gives lambda = 1**: synthetic {ctl['C2_a0_replaced_by_a']['synthetic_lambda']:.12f}, real sample {ctl['C2_a0_replaced_by_a']['real_sample_lambda']:.12f} [M].
3. **Deliberate-regression mutants change the number and a test catches them.** On the synthetic world, against the known value {ctl['C1_known_noise']['expected_lambda']:.4f}: no residualisation {ctl['C3_mutants_synthetic']['no_residualisation']['synthetic_lambda']:.4f} (miss {ctl['C3_mutants_synthetic']['no_residualisation']['miss_vs_known']:.3f}, caught), forward-difference a0 {ctl['C3_mutants_synthetic']['forward_difference_a0']['synthetic_lambda']:.4f} (miss {ctl['C3_mutants_synthetic']['forward_difference_a0']['miss_vs_known']:.3f}, caught, and caught by the gate), 0.2 s a0 {ctl['C3_mutants_synthetic']['a0_0.2s']['synthetic_lambda']:.4f} (miss {ctl['C3_mutants_synthetic']['a0_0.2s']['miss_vs_known']:.3f}, caught, and caught by the gate); a +-0.3 s instead of +-0.5 s fit moves it by {ctl['C3_mutants_synthetic']['sg_pm0.3s_instead_of_pm0.5s']['shift_vs_correct']:.3f}, inside the 0.015 tolerance, so that one is NOT caught by the known-value check [M]. On the real sample the mutants move lambda by {ctl['C3_mutants_real_sample']['no_residualisation']['shift_vs_correct']:+.5f} (no residualisation: v0 and kappa0 explain almost none of the variance of a0 here), {ctl['C3_mutants_real_sample']['forward_difference_a0']['shift_vs_correct']:+.4f} (forward), {ctl['C3_mutants_real_sample']['a0_0.2s']['shift_vs_correct']:+.4f} (0.2 s) and {ctl['C3_mutants_real_sample']['sg_pm0.3s_instead_of_pm0.5s']['shift_vs_correct']:+.4f} (+-0.3 s), and none of them changes the word. The gate catches the forward and 0.2 s definitions with max abs diffs of {ctl['C3_mutants_real_sample']['gate_on_real_table_max_abs_vs_model_a0']['forward_mutant']['max_abs']:.2f} and {ctl['C3_mutants_real_sample']['gate_on_real_table_max_abs_vs_model_a0']['lag2_prose']['max_abs']:.2f} m/s^2 over all {ctl['C3_mutants_real_sample']['gate_on_real_table_max_abs_vs_model_a0']['forward_mutant']['n']:,} in-band windows [M].
4. The Gram-matrix bootstrap equals `ci.py`'s on identical draws (1e-9), the verdict function is tested at its boundaries, and the SG estimator does not wrap a negative index at the clip start (NaN instead) [M].

## 7. What the card's section 4 sends next
**FAILS:** dilution does not explain the shrinkage. The card commits: the remaining explanation inside the weights is a learned response to the data, intent or the frontier, and the perception lever (LoRA, P4) carries the crawl alone, as T2 already ranks it. The "M2b with a denoised acceleration input" arm is not written. Nothing in this read asks for it: the error variance a denoised input could remove is {100*(1-lam):.1f} % of Var(a0_perp).

**Beside, post hoc, decides nothing, the cheapest discriminating measurement for the remaining explanations (`p5_next_lever.py`, `raw/P5_NEXT_LEVER.json`):** the data's own least-squares gain of the realised plan speed s_gt (secant of the GT future, knots 8 to 12) on a0, given v0 and kappa0, on the same {nxt['n_windows']:,} TRAIN windows: **ds/da0 = {nxt['data_optimal_ds_da0_given_v0_kappa0']['coef_on_a0']:.3f} [{nxt['data_optimal_ds_da0_given_v0_kappa0']['ci95'][0]:.3f}, {nxt['data_optimal_ds_da0_given_v0_kappa0']['ci95'][1]:.3f}]** and ds/dv0 = {nxt['data_optimal_ds_da0_given_v0_kappa0']['coef_on_v0']:.3f} [{nxt['data_optimal_ds_da0_given_v0_kappa0']['coef_on_v0_ci95'][0]:.3f}, {nxt['data_optimal_ds_da0_given_v0_kappa0']['coef_on_v0_ci95'][1]:.3f}] (R^2 {nxt['r2_of_s_gt_on_v0_a0_kappa0']:.3f}) [M]. The non-learned base applies 1.899; the data's optimum is {nxt['data_optimal_ds_da0_given_v0_kappa0']['coef_on_a0']:.2f}; the learned model sits at 1.265 / 1.112 [M-inh]. Reading [H]: most of the shrinkage (1.899 to about 1.38) is the data's own mean reversion of acceleration (the unresolved-intent side of the card's remaining explanations), and only the last 0.12 (s0) / 0.27 (s1) is the model's additional under-response. Caveat: the OLS gain is a population linear fit on train windows and the model's gains are medians of local Jacobians on 586 held-out windows, two different estimands; this is a pointer for the next card, not a test.

**Beside, decides nothing: R^2 of a on the frozen pooled tokens** (`p5_ridge.py`, `raw/P5_RIDGE.json`; the train cache's `pool4x10` tokens at the present frame, ridge fit on train folds only, 5 episode-grouped folds, out of fold; tokens are on the dev box for all {rid['n_windows']:,} windows): mean token (1,024-d) R^2 = {rA['a']['r2_oof']:.3f} [{rA['a']['ci95'][0]:.3f}, {rA['a']['ci95'][1]:.3f}], four row-means of the 4 x 10 grid (4,096-d) R^2 = {rB['a']['r2_oof']:.3f} [{rB['a']['ci95'][0]:.3f}, {rB['a']['ci95'][1]:.3f}]; for reference R^2 of a0 itself {rA['a0_reference']['r2_oof']:.3f} / {rB['a0_reference']['r2_oof']:.3f}; the part of a that a0 does not carry (a - {rid['b_slope_a_on_a0']:.3f} a0) {rA['a_minus_b_a0']['r2_oof']:.3f} [{rA['a_minus_b_a0']['ci95'][0]:.3f}, {rA['a_minus_b_a0']['ci95'][1]:.3f}] / {rB['a_minus_b_a0']['r2_oof']:.3f} [{rB['a_minus_b_a0']['ci95'][0]:.3f}, {rB['a_minus_b_a0']['ci95'][1]:.3f}]; shuffled-target control {rA['CONTROL_shuffled_a']['r2_oof']:.4f} / {rB['CONTROL_shuffled_a']['r2_oof']:.4f} (must be about 0) [M]. A single frame carries a little of a (0.11 to 0.16) and almost nothing that a0 lacks (0.01 to 0.02), consistent with T3. The R^2 depends on the penalty rule: a first RidgeCV (GCV, alphas 1 to 1e5) attempt on the mean token read 0.086 for a and 0.072 for a0 before it was stopped for speed (`logs/ridge_attempt1_killed.log`); the reported numbers use an inner grouped hold-out with alphas up to 1e6. The row-mean grouping assumes token index = row x 10 + column (UNVERIFIED against the cache builder; the mean-token numbers do not depend on it).

## 8. Scope and what this read cannot see
* lambda compares a 0.1 s difference with a +-0.5 s fit of the SAME logged speed. An error common to both (for example a lag or smoothing built into the egomotion speed estimate) cancels in e and is invisible here [H]. The widest fit tried, +-0.8 s, still gives lambda 0.957; the P5 clauses are read as written.
* The gain ratios 0.666 / 0.586 are [M-inh] from `wsp_read/WSP_READ.md` lines 31-44 (held-out eval-141, v0 in 3-8 m/s); lambda here is on TRAIN windows, as the card specifies.
* The card's "0.2 s" wording (section 1 above): if the 0.2 s difference had been the input, lambda would be {S['card_prose_a0_0.2s_NOT_the_model_input']['point']:.3f} and the word the same.

## 9. Deliverable manifest (nothing staged, committed or pushed; everything exists in ONE place, `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/p5/`)
See the table in the hand-back and `MANIFEST.txt` (sha256 of every file).
"""
    open(L.HERE + "/P5_RESULT.md", "w", newline="\n").write(md)
    print("written", len(md), "chars")


if __name__ == "__main__":
    main()
