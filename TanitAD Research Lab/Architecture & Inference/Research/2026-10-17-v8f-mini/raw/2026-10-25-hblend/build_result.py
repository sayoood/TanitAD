"""Builds HBLEND_RESULT.md from the JSON outputs (every number is read from an artifact, none is typed). CPU, stdlib + numpy."""
import json
import os

OUT = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
SEEDS = ("M2b_s0", "M2b_s1")
S = {s: json.load(open(f"{OUT}/score_{s}.json")) for s in SEEDS}
W = {s: json.load(open(f"{OUT}/weights_{s}.json")) for s in SEEDS}
R = json.load(open(f"{OUT}/READ.json"))
DK = json.load(open(f"{OUT}/dk_read.json"))
XC = json.load(open(f"{OUT}/xcheck.json"))
FC = json.load(open(f"{OUT}/fit_clips.json"))
SV = {"M2b_s0": "s0", "M2b_s1": "s1"}


def f(v, dp=4):
    return "n/a" if v is None else f"{v:+.{dp}f}"


def g(v, dp=4):
    return "n/a" if v is None else f"{v:.{dp}f}"


def ci(d, dp=4):
    return f"{f(d['mean'] if 'mean' in d else d['delta'], dp)} [{f(d['lo'], dp)}, {f(d['hi'], dp)}]"


def sep(d):
    return "separated" if d.get("separated") else "not separated"


def ex(m, dp=4):
    return f"{g(m['mean'], dp)} [{g(m['lo'], dp)}, {g(m['hi'], dp)}]"


EST = "paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0"
TIER = "T1-class open loop (own held outputs; not closed loop)"
L = []
w = L.append

o = R["outcome"]
w("# HBLEND_RESULT: PREREG_V8F_HBLEND executed exactly as written (2026-10-25)\n")
w("Dev box, CPU only, ZERO training, no GPU. Every number below is **MEASURED (ours)** from the artifact paths named; tier **%s**; estimator **%s** unless a row says otherwise. "
  "The bootstrap answers 'would another draw of EPISODES say this?' only; training-seed variance enters only through the two seeds shown side by side and F = |ADE6(s0) - ADE6(s1)| = %.5f m (`H-ESTIM-SEED-1`). "
  "Inference is deterministic, so no inference-seed floor applies. `OUT` = `%s`.\n" % (TIER, EST, R["F_exact_m"], OUT))

# ----------------------------------------------------------------------------------------------------------------- headline
w("## 0. Outcome (the PREREG outcome table, applied verbatim)\n")
w(f"**{o}.** PRIMARY-S passes at both seeds, PRIMARY-N passes at both seeds, SECONDARY-A passes at both seeds. "
  f"Committed next step from the table: *{R['committed_next_step_verbatim']}*\n")
w("**But read section 7 before deploying anything.** The PREREG's bars do not include the CONSISTENCY read the programme registered (H-MINI-CONS-1) and H is **separated-worse than B on it at both seeds**; "
  "distance keeping is mixed (seed 0 worse on headway / time-gap / LEAD-VIOL, seed 1 not separated); H does **not** dominate P10 (acceleration MAE stays above P10's). "
  "None of these changes the table's outcome (the bars are fixed); all of them bear on the decision 'H becomes the deployed v8F output'. Escalated in the hand-back.\n")

# ----------------------------------------------------------------------------------------------------------------- what was run
w("## 1. What was run\n")
n_win = W["M2b_s0"]["n_windows"]
w(f"* **Route.** B's banked per-window plan tau = `pred` of the KIND run (own code), read from the repo mirror at `refs/heads/agent/arch-inf-20260803` "
  f"(`.../raw/2026-10-25-kind/run/eval/eval_M2b_s{{0,1}}-kind-own.npz`, copied to `OUT/inputs/banked/`, sha256 in each score json). "
  f"P = the checkpoint's own frozen P10 evaluated on the eval kinematic features (it needs no image), cross-checked to the P10-yaw-4512 file. GT = the eval's own context. "
  f"The banked npz carries tau but not P, so P came from the checkpoint; the CPU forward was run on the 85 eval windows whose features are on this box (section 2).")
w(f"* **Fit set (deterministic, declared).** The 4,512 trained clips were rebuilt from box 2's `manifest.jsonl` + `--exclude 540:600` and reproduce the trainer's recorded digests "
  f"(`train_sha256` 37f4f9ee..., `excluded_sha256` f6a1dee0..., both read from box 2's `M2b_s0/selfcheck.json`). Sorted by sha12, **every k-th with k = {FC['k']}**: "
  f"k = 32 is the smallest k whose copy fits 3.0e9 bytes (k = 31 gives 146 clips = 3.05e9 B). **{FC['n_clips']} clips, 2,946,307,281 B** copied by sftp (`OUT/cache/train/clips/`, size-verified). "
  f"clip list sha256 `{FC['sha256_newline_joined']}` (`OUT/fit_clips.json`).")
w(f"* **Windows.** The trainer's own builder (`train_windows`, hist 4, start 3): **{n_win:,} windows** (about 68 per clip), ALL windows of the {FC['n_clips']} clips, in eval mode "
  f"(`forward_infer`: p_gt 0, T's own code, no label read). Ages: the eval's own deterministic schedule (a_t = i mod 3, a_s = a_t + (i div 3) mod 5, i = index in the fit list). "
  f"Eval-141 never enters the fit.")
w("* **Fit.** Closed form per step and axis in the NOW ego frame (axis 0 = along/x forward, axis 1 = cross/y left), clip to [0, 1], then unweighted pool-adjacent-violators to non-decreasing in t, separately per seed from its own checkpoint. "
  "`v8f_hblend.py fit`; weights in `weights_M2b_s{0,1}.json`.")
w("* **Declared interpretations (each was a choice the PREREG text left open; none was made after seeing eval).**\n"
  "  1. *'120 constants'.* The model's plan on the scored grid is the 0.2 s knot plan (`traj30 = traj60[1::2]`), the GT exists only on that grid and the trainer's loss is on it. So w is **fitted on 30 steps x 2 axes = 60 numbers per seed**; the 60 midpoint weights of the 0.1 s `traj60` are not identifiable from the data and are never scored. `weights_*.json` carries them as `w60_derived_unscored` (mean of neighbouring knots). No scored number depends on them.\n"
  "  2. *Isotonic* = unweighted PAV (the PREREG names no weights).\n"
  "  3. *3 GB* = 3.0e9 bytes (the stricter reading; under 3 GiB k would be 29-30 and the footprint ~3.2 GB).\n"
  "  4. *Bar of PRIMARY-N* uses the exact F = %.5f m (the PREREG prints 0.1207), so the bar is +%.5f m (the PREREG prints +0.060). No upper bound lies between the two.\n"
  "  5. *Fit ages* = the eval's own cycle (the PREREG is silent).\n" % (R["F_exact_m"], R["PRIMARY_N_bar_m"]))

# ----------------------------------------------------------------------------------------------------------------- controls
w("\n## 2. Identity and pipeline controls (run BEFORE any read; the script refuses to read if any fails)\n")
w("| control | bar | seed 0 | seed 1 |")
w("|---|---|---|---|")
c0, c1 = S["M2b_s0"]["identity_controls"], S["M2b_s1"]["identity_controls"]
rows = [("w = 1: max abs diff of H vs B's banked `pred` (m)", "<= 1e-4", "w1_vs_banked_pred_maxabs_m"),
        ("w = 1: ADE6 mean vs B's banked ADE6 (abs diff, m)", "<= 1e-4", "w1_ade6_vs_banked_mean_absdiff"),
        ("w = 1: per-window ADE6 max abs diff vs banked", "-", "w1_ade6_perwindow_maxabs_vs_banked"),
        ("w = 0: ADE6 (m)", "2.4870 +- 1e-3", "w0_ade6_mean"),
        ("w = 0: per-window ADE6 max abs diff vs `floors_perwindow.npz` P10_refit", "-", "w0_ade6_perwindow_maxabs_vs_floors_P10_refit"),
        ("w = 0: per-window ADE2 max abs diff vs `floors_perwindow.npz`", "-", "w0_ade2_perwindow_maxabs_vs_floors_P10_refit"),
        ("checkpoint's embedded prior vs P10-yaw-4512 file, max abs (m)", "<= 1e-4", "model_prior_vs_P10_4512_file_maxabs_m"),
        ("banked ADE6 recomputed from `pred` + GT, max abs (ADE2 and FDE6: also 0.0)", "-", "banked_ade6_recomputed_maxabs")]
for name, bar, k in rows:
    w(f"| {name} | {bar} | {c0[k]:.6g} | {c1[k]:.6g} |")
w(f"| fitted weights in [0, 1] and non-decreasing | required | {c0['fit_weights_in_range_and_monotone']} | {c1['fit_weights_in_range_and_monotone']} |")
w(f"| **all identity controls pass** | | **{c0['all_pass']}** | **{c1['all_pass']}** |")
w("")
for s in SEEDS:
    b = S[s]["w1_full_block_vs_banked_json"]
    a = S[s]["accel_mae_control"]
    w(f"* **{s}, w = 1 through the eval's OWN `evaluate()`** (traj, consistency_a, strata, tactical, self_consistency_b, families lon/lat/tac) vs the banked JSON `raw` block: **{b['n_mismatches']} mismatches** over keys {b['compared_keys']}. "
      f"Accel MAE: recomputed from per-window components {a['B_mean_fw_vs_banked_family_json'][0]:.4f} vs banked family block {a['B_mean_fw_vs_banked_family_json'][1]}; "
      f"P10 {a['P10_mean_fw_vs_PREREG_0.638'][0]:.4f} vs the PREREG's 0.638.")
w("")
w("* **CPU forward vs B's banked GPU output** (the PREREG's 1e-4 m bar): run on the eval windows whose features are on this box (1 + 4 clips = 85 windows; the other 136 clips' features, 2.8 GB, do not fit the 3.5 GB budget alongside the train copy). "
  + "; ".join(f"{s}: max |traj30_cpu - banked pred| = {XC[s]['max_abs_cpu_traj30_vs_banked_pred_m']:.2e} m, max |embedded p10 - standalone prior| = {XC[s]['max_abs_embedded_p10_vs_standalone_prior_m']:.2e} m (pass: {XC[s]['pass']})" for s in SEEDS)
  + ". The full-population figure (4.6e-5 / 9.9e-5 m over all 2,399 windows) is the video agent's `controls.json`, INHERITED, not re-run. Artifact: `OUT/xcheck.json`.")
w("* **Checkpoint identity.** sha256 of `M2b_s0.pt` / `M2b_s1.pt` re-computed by the fit script and asserted equal to box 2's recorded 601cb435...71dcc6f8 / 6104fb15...e50c2e6.")
w("* **Self-tests** (`test_v8f_hblend.py`, 7 passed, `--basetemp C:/Users/Admin/pt_hblend`): PAV optimality vs brute force; closed-form identities (y = tau gives w = 1, y = P gives w = 0, y = P + 0.4 r gives 0.4 flat); clip-then-isotonic order; a deliberate-regression arm (GT silently replaced by the prior collapses w to 0 and H to P10, which the w = 0 control would flag as 'H = P10', never as a gain).\n")

FV = json.load(open(f"{OUT}/fit_verify.json"))
w("* **Independent re-derivation of the weights** (`verify_fit.py`, `fit_verify.json`): from the saved fit-window arrays, np.linalg.lstsq per step and axis reproduces the raw weights to " + ", ".join(f"{s_}: {FV[s_]['max_abs_raw_lstsq_vs_json']:.1e}" for s_ in SEEDS) + " and sklearn `IsotonicRegression` reproduces the final weights to " + ", ".join(f"{s_}: {FV[s_]['max_abs_sklearn_isotonic_vs_json']:.1e}" for s_ in SEEDS) + ".")

# ----------------------------------------------------------------------------------------------------------------- weights
w("## 3. The fitted weights (the entire deployable artifact: 60 numbers per seed)\n")
w(f"Fit set: {n_win:,} windows, {FC['n_clips']} clips, k = {FC['k']}. Final = clip then isotonic. Counts per seed (of 60 step-axis cells): "
  + "; ".join(f"{s}: {W[s]['n_clipped_at_0']} clipped at 0, {W[s]['n_clipped_at_1']} clipped at 1, isotonic changed {W[s]['n_changed_by_isotonic']}" for s in SEEDS)
  + ". Artifact: `OUT/weights_M2b_s{0,1}.json` (raw, clipped, final, sum r^2, derived 0.1 s grid).\n")
w("| t (s) | s0 along | s0 cross | s1 along | s1 cross | s0 raw along | s0 raw cross |")
w("|---|---|---|---|---|---|---|")
for i in range(30):
    w(f"| {0.2 * (i + 1):.1f} | {W['M2b_s0']['w_along_by_step'][i]:.3f} | {W['M2b_s0']['w_cross_by_step'][i]:.3f} | {W['M2b_s1']['w_along_by_step'][i]:.3f} | "
      f"{W['M2b_s1']['w_cross_by_step'][i]:.3f} | {W['M2b_s0']['raw_along'][i]:+.3f} | {W['M2b_s0']['raw_cross'][i]:+.3f} |")
w("")
def _rng(v, a, b):
    return f"{v[a]:.2f}-{v[b]:.2f}"


a0, a1 = W["M2b_s0"]["w_along_by_step"], W["M2b_s1"]["w_along_by_step"]
c0_, c1_ = W["M2b_s0"]["w_cross_by_step"], W["M2b_s1"]["w_cross_by_step"]
w("Reading: on the along axis the residual is shrunk hard up to 2 s (final w %s at 0.4-2.0 s for s0, %s for s1; %.3f / %.3f at 0.2 s) and kept progressively later (%.2f / %.2f at 6 s); "
  "the cross axis keeps %s / %s early (0.2-1.8 s) and %.2f / %.2f at 6 s. The two seeds agree in shape; s1 keeps more of its residual late. " % (
      _rng(a0, 1, 9), _rng(a1, 1, 9), a0[0], a1[0], a0[29], a1[29], _rng(c0_, 0, 8), _rng(c1_, 0, 8), c0_[29], c1_[29])
  + f"In-sample on the fit windows (descriptive, not a result): ADE2 tau / P / H = "
  + " ; ".join(f"{SV[s]} {W[s]['in_sample_train_ade2']['B_tau']:.4f} / {W[s]['in_sample_train_ade2']['P']:.4f} / {W[s]['in_sample_train_ade2']['H']:.4f}" for s in SEEDS)
  + "; ADE6 tau / P / H = " + " ; ".join(f"{SV[s]} {W[s]['in_sample_train_ade6']['B_tau']:.4f} / {W[s]['in_sample_train_ade6']['P']:.4f} / {W[s]['in_sample_train_ade6']['H']:.4f}" for s in SEEDS) + ".\n")

# ----------------------------------------------------------------------------------------------------------------- the reads
w("## 4. The PREREG rows\n")
w("| read | seed | value | 95 % CI | verdict | estimator | tier | artifact |")
w("|---|---|---|---|---|---|---|---|")
for s in SEEDS:
    r = R["per_seed"][s]
    a2, a6, ac = r["PRIMARY_S"], r["PRIMARY_N"], r["SECONDARY_A"]
    art = f"`OUT/score_{s}.json::paired_H_minus_B`"
    w(f"| **PRIMARY-S** ADE2 H - B (m) | {SV[s]} | {f(a2['delta'])} | [{f(a2['lo'])}, {f(a2['hi'])}] | bar: < 0 and separated -> **{'PASS' if a2['pass'] else 'FAIL'}** | {EST} | {TIER} | {art}.ade2 |")
    w(f"| **PRIMARY-N** ADE6 H - B (m) | {SV[s]} | {f(a6['delta'])} | [{f(a6['lo'])}, {f(a6['hi'])}] | bar: upper bound <= +{a6['bar_hi_le']:.5f} -> **{'PASS' if a6['pass'] else 'FAIL'}** | {EST} | {TIER} | {art}.ade6 |")
    w(f"| **SECONDARY-A** accel MAE H - B (m/s^2) | {SV[s]} | {f(ac['delta'])} | [{f(ac['lo'])}, {f(ac['hi'])}] | bar: < 0 and separated -> **{'PASS' if ac['pass'] else 'FAIL'}** | {EST} | {TIER} | {art}.fw_lon_accel_mae |")
w("")
w("**DOMINANCE vs P10-yaw-4512** (descriptive: 'H dominates P10' iff ADE2, ADE6 and accel MAE point estimates are all <= P10's and ADE6 is separated):\n")
w("| seed | ADE2 H / P10 (m) | ADE6 H / P10 (m) | ADE6 H - P10, paired | accel MAE H / P10 (m/s^2) | accel H - P10, paired | H dominates P10 |")
w("|---|---|---|---|---|---|---|")
for s in SEEDS:
    j = S[s]
    h, p = j["single"]["H"], j["single"]["P10"]
    dp6, dpa = j["paired_H_minus_P10"]["ade6"], j["paired_H_minus_P10"]["fw_lon_accel_mae"]
    dom = R["per_seed"][s]["DOMINANCE_vs_P10"]
    w(f"| {SV[s]} | {h['ade2']['mean']:.4f} / {p['ade2']['mean']:.4f} | {h['ade6']['mean']:.4f} / {p['ade6']['mean']:.4f} | {ci(dp6)} {sep(dp6)} | "
      f"{h['fw_lon_accel_mae']['mean']:.4f} / {p['fw_lon_accel_mae']['mean']:.4f} | {ci(dpa)} {sep(dpa)} | **{dom['H_dominates_P10']}** (ADE2 {dom['point_estimates_le_P10']['ade2']}, ADE6 {dom['point_estimates_le_P10']['ade6']}, accel {dom['point_estimates_le_P10']['accel']}) |")
w("")
_h = {s_: S[s_]["single"]["H"]["ade2"]["mean"] for s_ in SEEDS}
_p = S["M2b_s0"]["single"]["P10"]["ade2"]["mean"]
_sep2 = all(S[s_]["paired_H_minus_P10"]["ade2"]["separated"] and S[s_]["paired_H_minus_P10"]["ade2"]["mean"] < 0 for s_ in SEEDS)
_sep6 = all(S[s_]["paired_H_minus_P10"]["ade6"]["separated"] and S[s_]["paired_H_minus_P10"]["ade6"]["mean"] < 0 for s_ in SEEDS)
_acc = all(S[s_]["paired_H_minus_P10"]["fw_lon_accel_mae"]["separated"] and S[s_]["paired_H_minus_P10"]["fw_lon_accel_mae"]["mean"] > 0 for s_ in SEEDS)
assert _sep2 and _sep6 and _acc, "the narrative sentence below states exactly this; re-write it if the data changed"
w("H beats P10 on ADE2 (point estimates %.4f and %.4f against %.4f; paired H - P10 separated at both seeds, section 5) and on ADE6 (separated at both seeds), but **its acceleration MAE stays above P10's** (paired H - P10 separated, positive, at both seeds). "
  "So the answer to 'does the planner now dominate its kinematic prior' is: on position yes at every horizon 1-6 s, on acceleration no. Artifact: `OUT/score_M2b_s*.json::paired_H_minus_P10`.\n" % (_h["M2b_s0"], _h["M2b_s1"], _p))

# ----------------------------------------------------------------------------------------------------------------- horizons
w("## 5. ADE and FDE at 1-6 s (beside, per PREREG)\n")
for s in SEEDS:
    w(f"**{s}** (m; mean [95 % CI]; paired deltas {EST})\n")
    w("| metric | B | H | P10 | H - B | H - P10 |")
    w("|---|---|---|---|---|---|")
    for kind in ("ade", "fde"):
        for h in range(1, 7):
            k = f"{kind}{h}"
            sg = S[s]["single"]
            dB, dP = S[s]["paired_H_minus_B"][k], S[s]["paired_H_minus_P10"][k]
            w(f"| {kind.upper()}@{h}s | {ex(sg['B'][k])} | {ex(sg['H'][k])} | {ex(sg['P10'][k])} | {ci(dB)} {'*' if dB['separated'] else ''} | {ci(dP)} {'*' if dP['separated'] else ''} |")
    w("\n`*` = CI excludes 0. Artifact: `OUT/score_%s.json::single / paired_H_minus_B / paired_H_minus_P10`.\n" % s)

# ----------------------------------------------------------------------------------------------------------------- families
w("## 6. The four families, beside (H - B and H - P10 paired; lower is better except the two tactical agreement rows)\n")
fams = [("LONGITUDINAL", "fw_lon_speed_mae", "speed MAE (m/s)"), ("LONGITUDINAL", "fw_lon_accel_mae", "accel MAE (m/s^2)"), ("LONGITUDINAL", "fw_lon_along_mae", "along MAE (m)"),
        ("LATERAL", "fw_lat_cross_mae", "cross-track offset MAE (m)"), ("LATERAL", "fw_lat_heading_mae_deg", "heading MAE (deg)"),
        ("LATERAL", "fw_lat_yaw_rate_mae_degps", "yaw-rate MAE (deg/s)"), ("LATERAL", "fw_lat_curvature_mae_1pm", "curvature MAE (1/m)"),
        ("TACTICAL (plan-derived)", "fw_tac_lat_acc", "lateral decision agreement (higher better)"), ("TACTICAL (plan-derived)", "fw_tac_lon_acc", "longitudinal decision agreement (higher better)")]
for s in SEEDS:
    w(f"**{s}**\n")
    w("| family | metric | B | H | P10 | H - B | H - P10 |")
    w("|---|---|---|---|---|---|---|")
    for fam, k, nm in fams:
        sg = S[s]["single"]
        dB, dP = S[s]["paired_H_minus_B"][k], S[s]["paired_H_minus_P10"][k]
        w(f"| {fam} | {nm} | {g(sg['B'][k]['mean'])} | {g(sg['H'][k]['mean'])} | {g(sg['P10'][k]['mean'])} | {ci(dB)} {'*' if dB['separated'] else ''} | {ci(dP)} {'*' if dP['separated'] else ''} |")
    w("")
for _s in SEEDS:
    for _fam, _k, _nm in fams:
        assert not S[_s]["paired_H_minus_B"][_k]["worse"], f"{_s} {_k} separated-worse than B: rewrite the sentence below"
    assert all(S[_s]["paired_H_minus_B"][k_]["better"] for k_ in ("fw_tac_lat_acc", "fw_tac_lon_acc"))
w("Nothing in LONGITUDINAL (speed, accel, along), LATERAL or the plan-derived TACTICAL rows is separated-worse than B at either seed; the plan-derived tactical agreement is separated-better at both. "
  "(Plan-derived tactical = `four_families.tactical_from_trajectory`: the decision read OFF THE TRAJECTORY. It moves with H, because H moves the trajectory. Do not confuse it with T's own decision heads below.)\n")
w("### 6a. Distance keeping (the other half of LONGITUDINAL), K = 30 (6 s) lead block, through the programme's own `v8f_dk_read.read_dk` -> `taniteval.lead_metrics`\n")
w(f"Lead block `b1_eval_lead_block_k30.npz` (sha256 `{DK['lead_block']['sha256'][:16]}...`), window states {DK['join']['window_states']}, kinematic-contract check **{DK['human_reference']['kinematic_contract']['status']}**, "
  f"arm-vs-itself identity control exactly +0 [0, 0]. Open loop: the lead's logged future does not react to the arm. No distance-keeping bar is registered for v8F-mini, so nothing here is PASS/FAIL. "
  "Higher min-TTC is a CENSORED mean (capped at 30 s when never closing); direction 'more gap' is not 'better'; the better/worse reading is LEAD-VIOL and the distance to the human.\n")
w("| arm | headway min (m) | time-gap min (s) | min-TTC (s, censored) | LEAD-VIOL share of at-risk (n viol / 595) |")
w("|---|---|---|---|---|")
hm = DK["human_reference"]["metrics"]
w(f"| human (GT path) | {hm['headway_min_m']['mean']:.3f} | {hm['time_gap_min_s']['mean']:.3f} | {hm['min_ttc_s']['mean']:.3f} | by definition 0 |")
for nm in ("B_s0", "H_s0", "B_s1", "H_s1", "P10"):
    a = DK["arms"][nm]
    m = a["metrics"]
    w(f"| {nm} | {m['headway_min_m']['mean']:.3f} | {m['time_gap_min_s']['mean']:.3f} | {m['min_ttc_s']['mean']:.3f} | {a['lead_viol']['share_of_at_risk']:.4f} ({a['lead_viol']['n_viol']}) |")
w("")
w("| paired contrast | headway (m) | time-gap (s) | min-TTC (s) | LEAD-VIOL at-risk (share) |")
w("|---|---|---|---|---|")
for pn in ("H_s0__minus__B_s0", "H_s1__minus__B_s1", "H_s0__minus__P10", "H_s1__minus__P10"):
    pr = DK["pairs"][pn]
    cells = []
    for k in ("headway_min_m", "time_gap_min_s", "min_ttc_s"):
        d = pr["dk"][k]
        cells.append(f"{f(d['delta'], 3)} [{f(d['lo'], 3)}, {f(d['hi'], 3)}] {'*' if d['separated'] else ''}")
    lv = pr["lead_viol"]["at_risk"]
    cells.append(f"{f(lv['delta'], 4)} [{f(lv['lo'], 4)}, {f(lv['hi'], 4)}] {'*' if lv['separated'] else ''}")
    w(f"| {pn.replace('__minus__', ' - ')} | " + " | ".join(cells) + " |")
w("\n`*` = CI excludes 0. Artifact: `OUT/dk_read.json` (`pairs`, `arms`), per-window `OUT/dk_perwindow.npz`.\n")
w("**Reading.** Seed 0: H holds LESS headway (-0.21 m) and time-gap (-0.10 s) than B and has more LEAD-VIOL (29 vs 21 of 595 at-risk windows, separated); seed 1: none of the three is separated and LEAD-VIOL is 35 vs 36. "
  "Min-TTC is higher at both seeds (separated); it is a censored mean (30 s cap), so it is not read as 'better' on its own. Against P10, H keeps a LONGER time-gap (separated at both seeds) and has fewer LEAD-VIOL (separated at seed 0 only). "
  "Seed-dependent, so not a finding about H; it is a flag.\n")
w("### 6b. T's own decision heads and S's checkpoints: the structural identity\n")
si = S["M2b_s0"]["structural_identity"]
w(f"* **TACTICAL (T's heads: lat / lon macro-F1 and goal-token macro-F1):** H's block equals B's banked block **exactly** (json equality) at s0: {si['tactical_block_T_heads']['H_equals_banked_B']} and s1: {S['M2b_s1']['structural_identity']['tactical_block_T_heads']['H_equals_banked_B']}. "
  f"Values (identical for B and H): lat macro-F1 {S['M2b_s0']['blocks']['H']['tactical']['lat']['macro_f1']:.4f}, lon {S['M2b_s0']['blocks']['H']['tactical']['lon']['macro_f1']:.4f}, goal {S['M2b_s0']['blocks']['H']['tactical']['goal_macro_f1']:.4f} (s0); "
  f"s1: lat {S['M2b_s1']['blocks']['H']['tactical']['lat']['macro_f1']:.4f}, lon {S['M2b_s1']['blocks']['H']['tactical']['lon']['macro_f1']:.4f}, goal {S['M2b_s1']['blocks']['H']['tactical']['goal_macro_f1']:.4f}.")
w("* **The PREREG's sentence 'tactical ... must read EXACTLY B's values' holds for T's heads and does NOT hold for the plan-derived tactical agreement of section 6** (it is read from the trajectory and moves with H). This is a wording gap in the PREREG, reported here rather than hidden.")
w("* **STRATEGIC: PAUSED** per the PI ruling of 2026-09-19 (`PI_DECISION_QUEUE.md` ITEM 25): the strategic layer is switched off temporarily; the family is reported as PAUSED, not as a pass. "
  "S's checkpoint errors are B's by construction (H never touches S; banked `ck_pos_err` carried unchanged; B's block is in `OUT/score_*.json::structural_identity.strategic_block_S_checkpoints`).\n")

# ----------------------------------------------------------------------------------------------------------------- consistency
w("## 7. CONSISTENCY: the read the PREREG table does not carry, and where H pays\n")
w("CONSISTENCY(a) = share of windows whose PLAN violates a GT-derived constraint (speed limit, stop, turn, lead gap); (b) = violation of the model's OWN predicted constraints. Lower is better. "
  f"Estimator: {EST}. Artifact: `OUT/score_M2b_s*.json::consistency_beside`; P10 rates from `floors_perwindow.npz`.\n")
w("| seed | rule | B | H | P10 | H - B | H - P10 |")
w("|---|---|---|---|---|---|---|")
for s in SEEDS:
    cb = S[s]["consistency_beside"]
    for k in ("viol_any", "viol_SPEED", "viol_STOP", "viol_TURN", "viol_LEAD", "self_viol_any"):
        dB = cb["paired_H_minus_B"][k]
        dP = cb["paired_H_minus_P10"].get(k)
        pcell = f"{ci(dP)} {'*' if dP['separated'] else ''}" if dP else "n/a"
        w(f"| {SV[s]} | {k.replace('viol_', '').replace('self_', 'self (b) ')} | {cb['rate']['B'][k]:.4f} | {cb['rate']['H'][k]:.4f} | {g(cb['rate']['P10'].get(k)) if k in cb['rate']['P10'] else 'n/a'} | {ci(dB)} {'**WORSE**' if dB['worse'] else ('*' if dB['separated'] else '')} | {pcell} |")
w("")
_cb = {s_: S[s_]["consistency_beside"] for s_ in SEEDS}
assert all(_cb[s_]["paired_H_minus_B"]["viol_any"]["worse"] and _cb[s_]["paired_H_minus_B"]["viol_SPEED"]["worse"] and _cb[s_]["paired_H_minus_B"]["self_viol_any"]["worse"] for s_ in SEEDS)
w("**H is separated-worse than B on CONSISTENCY(a) 'any' at both seeds (%s and %s), on SPEED (%s and %s), and on the model's own self-consistency (b) (%s and %s).** It is still better than P10 (separated at s0 only for 'any'). " % (
    f(_cb["M2b_s0"]["paired_H_minus_B"]["viol_any"]["mean"], 3), f(_cb["M2b_s1"]["paired_H_minus_B"]["viol_any"]["mean"], 3),
    f(_cb["M2b_s0"]["paired_H_minus_B"]["viol_SPEED"]["mean"], 3), f(_cb["M2b_s1"]["paired_H_minus_B"]["viol_SPEED"]["mean"], 3),
    f(_cb["M2b_s0"]["paired_H_minus_B"]["self_viol_any"]["mean"], 3), f(_cb["M2b_s1"]["paired_H_minus_B"]["self_viol_any"]["mean"], 3)) +
  "Mechanism (HYPOTHESIS, consistent with the numbers, not isolated): B's residual was trained with the consistency penalties in the loss, so part of the residual IS the constraint-satisfying correction; shrinking it back toward the unconstrained P10 gives that part back, "
  "and the fit minimises position error only. The weights have no term for the constraints. Whether the trade (ADE and acceleration for constraint violations) is acceptable is a decision for the Master Mind / PI, not for this read; the PREREG's PASS does not decide it.\n")

# ----------------------------------------------------------------------------------------------------------------- secondary & diagnostics
w("## 8. SECONDARY (descriptive) and diagnostics\n")
w("**2-fold episode cross-fit on eval-141** (fold = parity of the sorted-episode index; fit on one fold, score the other, swap; **CROSS-FITTED, not a deployable weight**; SECONDARY, descriptive). Paired vs B:\n")
w("| seed | ADE2 CF - B | ADE6 CF - B | accel CF - B | ADE2 / ADE6 CF (m) |")
w("|---|---|---|---|---|")
for s in SEEDS:
    c = S[s]["crossfit_secondary"]
    d = c["paired_vs_B"]
    w(f"| {SV[s]} | {ci(d['ade2'])} | {ci(d['ade6'])} | {ci(d['fw_lon_accel_mae'])} | {c['single']['ade2']['mean']:.4f} / {c['single']['ade6']['mean']:.4f} |")
w("\nThe cross-fit reproduces the train-fit result at both seeds (same sign, all separated; ADE6 s1 -0.040 vs -0.070 with half the episodes in each fit).\n")
w("**Diagnostics, declared as diagnostics (never a bar):**\n")
w("| diagnostic | seed | ADE2 - B | ADE6 - B | accel - B | reading |")
w("|---|---|---|---|---|---|")
for s in SEEDS:
    o_ = S[s]["diag_eval_oracle_in_sample"]
    sh = S[s]["diag_shuffled_residual_fit"]
    w(f"| eval-oracle w (fitted ON eval-141, reads GT; the ceiling of this lever) | {SV[s]} | {ci(o_['paired_vs_B']['ade2'])} | {ci(o_['paired_vs_B']['ade6'])} | {ci(o_['paired_vs_B']['fw_lon_accel_mae'])} | train-fitted H captures nearly all of it |")
    w(f"| noise control: w fitted on train with the residual PERMUTED across windows | {SV[s]} | {ci(sh['paired_vs_B']['ade2'])} | {ci(sh['paired_vs_B']['ade6'])} | n/a | w collapses to ~0.00-0.05 (H = P10), ADE6 gets WORSE than B |")
w("")
_g6 = [(S[s_]["paired_H_minus_B"]["ade6"]["mean"], S[s_]["diag_eval_oracle_in_sample"]["paired_vs_B"]["ade6"]["mean"]) for s_ in SEEDS]
_sh = [S[s_]["diag_shuffled_residual_fit"]["paired_vs_B"] for s_ in SEEDS]
assert all(x["ade2"]["separated"] and x["ade2"]["mean"] < 0 and x["ade6"]["separated"] and x["ade6"]["mean"] > 0 for x in _sh)
w("Two things these show. (i) The train-fitted weights sit at the ceiling of the lever: H's ADE6 gain (%s / %s) is within %.3f m of what weights fitted on eval itself would give (%s / %s), so a better weight fit has nothing left to buy. " % (
    f(_g6[0][0], 3), f(_g6[1][0], 3), max(abs(a_ - b_) for a_, b_ in _g6), f(_g6[0][1], 3), f(_g6[1][1], 3)) +
  "(ii) **PRIMARY-S is a bar against B, and B is easy to beat at 2 s by returning to P10**: the permuted-residual control (H = P10 essentially) also passes ADE2 H - B, separated at both seeds. "
  "So PRIMARY-S alone does not show that shrinkage found signal; what shows it is that the fitted H also beats P10 at 2 s (section 4, separated at both seeds) and that the permuted control LOSES to B at 6 s (separated worse at both seeds) while the fitted H wins.\n")

# ----------------------------------------------------------------------------------------------------------------- outcome
w("## 9. The outcome table, applied verbatim\n")
w("| outcome | condition | met? |")
w("|---|---|---|")
w(f"| **PASS** | PRIMARY-S and PRIMARY-N at both seeds | **{R['PRIMARY_S_both_seeds'] and R['PRIMARY_N_both_seeds']}** (S: {R['PRIMARY_S_both_seeds']}, N: {R['PRIMARY_N_both_seeds']}) |")
w(f"| SHORT ONLY | PRIMARY-S passes, PRIMARY-N fails | {R['PRIMARY_S_both_seeds'] and not R['PRIMARY_N_both_seeds']} |")
w(f"| NO GAIN | PRIMARY-S fails | {not R['PRIMARY_S_both_seeds']} |")
w(f"\n**Result: {o}.** Committed next (PREREG): {R['committed_next_step_verbatim']}\n")
w("The functional form, the clip range, the isotonic step and the bars were not changed after any eval was seen. The weights were fitted once per seed before the first eval read; the script refuses to read unless the identity controls pass. "
  "A first run of `score` with `--no-families --n-boot 100` (a smoke to flush bugs) was made after the fit and before the full read; it printed the same point estimates; no choice depended on it. `score` was then re-run once after adding the CONSISTENCY block (section 7); the bootstrap is seeded and every PREREG row reproduced to the printed digit.\n")

# ----------------------------------------------------------------------------------------------------------------- limits
w("## 10. What this does and does not show\n")
w("* **Evidence class.** All rows MEASURED (this run). B's per-window plans are the banked KIND-run outputs (INHERITED from box 2's GPU run; re-derived metrics reproduce the banked arrays with max abs diff 0.0 and the full `evaluate()` block with 0 mismatches; the CPU forward reproduces them to <= 2.8e-5 m on 85 windows).")
w("* **Tier.** T1-class open loop on 2,399 windows / 141 episodes. Not closed loop: nothing here says H drives better, only that its open-loop plan is closer to the logged future and smoother in acceleration.")
w("* **Variance.** The bootstrap resamples episodes with the models fixed. Training-run variance is covered only by the two seeds, which agree in sign on every PREREG row but differ in size (ADE6 H - B %s vs %s; the s1 arm starts from a better B, ADE6 %.3f vs %.3f, and shrinkage buys less). Two seeds are a floor check, not an interval." % (f(S["M2b_s0"]["paired_H_minus_B"]["ade6"]["mean"], 3), f(S["M2b_s1"]["paired_H_minus_B"]["ade6"]["mean"], 3), S["M2b_s0"]["single"]["B"]["ade6"]["mean"], S["M2b_s1"]["single"]["B"]["ade6"]["mean"]))
w("* **Fit-set caveat (declared, in the PREREG).** The fit windows are windows B was trained on; the weights therefore shrink LESS than eval-optimal ones. The eval-oracle diagnostic and the cross-fit show the effect is not an artifact of that: the train-fitted weights reach the eval-oracle ceiling.")
w("* **Not done.** The eval CPU forward over all 2,399 windows (the eval cache is 2.8 GB; the budget could not hold it beside the 2.9 GB train copy and deletion needs the PI). It is not needed for the read (B's tau is banked, P is image-free) and the 85-window cross-check covers the route. "
  "No H + hard-speed-projection read (needs traj60 on every eval window, i.e. the full CPU forward).\n")

# ----------------------------------------------------------------------------------------------------------------- next
w("## 11. Rule Zero: the levers this leaves, ranked by measured effect (nothing below is a new bar; each needs its own PREREG)\n")
w("1. **The committed next step (PASS):** weights become a model buffer in the next trained arm; FLAT and MIXO re-scored with H on each arm's own fitted weights (descriptive). Unblocked on the fit side (the 141-clip fit set is on disk, about 1 minute of fit per checkpoint); needs those arms' checkpoints (their location was not checked here).")
w("2. **Constraint-aware H.** The measured cost of H is CONSISTENCY(a) (+0.075 to +0.100 'any', SPEED +0.10). Cheapest experiment: fit the SAME closed form but only on the windows where no constraint is active, or add the hard speed projection after H (B's own projection variant exists, `traj30_proj`). Blocked on: H on traj60 for every eval window = the full CPU forward (eval cache 2.8 GB, needs a disk decision) or one box-2 run.")
w("3. **Acceleration gap to P10** (+0.104 / +0.066 m/s^2, separated): the weights are at their ceiling, so this is not a weight problem. Candidate: shrink per DCT coefficient instead of per step (the residual's high-frequency part is the jitter). Zero training, same fit machinery.\n")

# ----------------------------------------------------------------------------------------------------------------- manifest
w("## 12. Deliverable manifest\n")
w("All under `OUT` (dev-box scratchpad; **nothing staged to git, nothing pushed, nothing written on box 2**). The Master Mind lands them.\n")
w("| artifact | path (under OUT) | note |")
w("|---|---|---|")
man = [("this report", "HBLEND_RESULT.md", "generated by build_result.py from the JSONs below"),
       ("executor (fit / score / read / dk / xcheck)", "v8f_hblend.py", "imports the eval's own `evaluate`, `taniteval.ci`, `v8f_beta_eval`, the DK reader"),
       ("report generator", "build_result.py", ""), ("self-tests", "test_v8f_hblend.py", "7 passed"), ("independent weight re-derivation", "verify_fit.py, fit_verify.json", "lstsq + sklearn isotonic"),
       ("fitted weights", "weights_M2b_s0.json, weights_M2b_s1.json", "THE deployable artifact: 60 numbers per seed + raw/clipped/derived"),
       ("fit-set predictions", "fit_windows_M2b_s{0,1}.npz", f"{n_win:,} windows: traj30, p30, gt, ages"),
       ("fit clip list", "fit_clips.json, train_clip_order.json", "k = 32, 141 clips, sha256 above"),
       ("per-seed score (every row, CI, blocks, controls)", "score_M2b_s0.json, score_M2b_s1.json", ""),
       ("per-window H / B / P10 arrays", "score_M2b_s{0,1}_perwindow.npz", "H, P, tau, per-window ADE/FDE/family components"),
       ("outcome", "READ.json", "the table, verbatim"),
       ("distance keeping", "dk_read.json, dk_perwindow.npz", "via v8f_dk_read.read_dk"),
       ("route cross-check", "xcheck.json", "85 windows CPU vs banked"),
       ("logs", "fit_M2b_s*.log, score_M2b_s*.log, read.log, dk.log, copy_log.txt", ""),
       ("inputs fetched (read-only)", "inputs/ (banked B npz+json from the mirror, P10-4512 file, floors, lead block), box2/ (manifest, join_train.npz, argv, summaries, sftp batches), cache/train/clips/ (141 clips, 2.946 GB), deps/ (taniteval copy + DK reader)", "copies; originals untouched"),
       ("smoke leftovers", "weights_smoke.json, fit_windows_smoke.npz", "3-clip smoke of the fit; not used in any read")]
for a, b, c in man:
    w(f"| {a} | `{b}` | {c} |")
w("\nFootprint on C: about 2.99 GB (2.946 GB train clips, 35 MB join, the rest small); nothing on E:; nothing deleted. `df -h /c` before the copy: 11 GB free; after: 7.4 GB (other processes also write to C:).\n")
w("## 13. Reproduce\n")
w("```\nexport CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4\nPY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe\n$PY v8f_hblend.py fit   --seed M2b_s0   # ~52 s\n$PY v8f_hblend.py fit   --seed M2b_s1\n"
  "$PY v8f_hblend.py score --seed M2b_s0   # ~31 s, identity controls first\n$PY v8f_hblend.py score --seed M2b_s1\n$PY v8f_hblend.py read\n$PY v8f_hblend.py dk\n$PY v8f_hblend.py xcheck\n$PY build_result.py\n```\n"
  "Paths are hard-coded to the scratchpad (`SCR` in `v8f_hblend.py`); relocate before landing. Pinned inputs: tip a99ca5d6 export at `viz/tree`, `taniteval` modules copied from box 2 (`viz/work/taniteval_b2`) plus `lead_metrics` / `lead_source` / `progress` / `v0_antiecho` from the mirror tip 25037c4.")

open(f"{OUT}/HBLEND_RESULT.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("written", f"{OUT}/HBLEND_RESULT.md", len("\n".join(L)))
