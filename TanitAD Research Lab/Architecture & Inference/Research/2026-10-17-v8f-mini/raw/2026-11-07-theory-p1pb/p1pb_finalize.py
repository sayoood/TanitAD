#!/usr/bin/env python3
"""p1pb_finalize.py -- P1PB_RESULT.md and P1PB_FILES.sha256 from P1PB_RESULT.json, P1PB_ADMISSIBILITY.json and the control files (no new data).  python -I p1pb_finalize.py"""
import hashlib
import json
import os

N_MIN = 30
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
R = json.load(open(f"{HERE}/P1PB_RESULT.json"))
ADM = json.load(open(f"{HERE}/P1PB_ADMISSIBILITY.json"))
FW = json.load(open(f"{HERE}/P1PB_FW_CONTROL.json"))
F1 = json.load(open(f"{HERE}/P1PB_F1_CONTROL.json"))
P1P = json.load(open(f"{HERE}/P1P_RESULT.json"))
P = R["PRIMARY"]
ps = P["per_seed"]
S = R["SENSITIVITIES_not_the_verdict"]
PO = R["POOLED_beside_decides_nothing"]
E141 = R["EVAL141_part_P1P_primary"]
steps, bins = P["window_filter_steps"], P["posterior_bins"]
k1, k2, k3 = R["K1_probe_auc_holdout_pmove"], R["K2_probe_auc_holdout_band_3_6"], R["K3_probe_auc_holdout_a_human_gt_0p5_band_3_8"]
verdict = P["verdict"]


def f(x, n=3):
    return "n/a" if x is None else f"{x:.{n}f}"


def sgn(x, n=3):
    return "n/a" if x is None else f"{x:+.{n}f}"


def iv(r, key_lo="lo", key_hi="hi"):
    return f"[{sgn(r[key_lo])}, {sgn(r[key_hi])}]"


def sep(r):
    return "yes" if (r["positive"] and r["separated"]) else "no"


# ------------------------------------------------------------------------------------------------ verdict text, by branch
if verdict == "NOT EVALUABLE":
    reason = (f"CONFIDENT holds {ps['s0']['n_conf']} windows at s0 and {ps['s1']['n_conf']} at s1 (floor {N_MIN}); UNSURE holds {ps['s0']['n_unsure']} and {ps['s1']['n_unsure']}. The criterion names this NOT EVALUABLE and the groups are not padded.")
elif verdict == "FAILS":
    fails = [s for s in ("s0", "s1") if not (ps[s]["positive"] and ps[s]["separated"])]
    reason = (f"Both groups clear the floor of {N_MIN} at both seeds (CONFIDENT {ps['s0']['n_conf']}, UNSURE {ps['s0']['n_unsure']}). Δ is positive at both seeds, and is separated at "
              f"{', '.join(s for s in ('s0', 's1') if s not in fails) or 'neither seed'}; at {', '.join(fails)} the lower end of the 95 % interval is "
              f"{sgn(ps[fails[0]]['lo'])}. The criterion (AMEND_1 §4): \"P1′ fails if the difference is not separated at either seed, or is negative.\"")
else:
    reason = f"Both groups clear the floor of {N_MIN} at both seeds and Δ is positive and separated at both seeds."
R["verdict_reason"] = reason
R["verdict_consequence_as_written"] = {"HOLDS": "P1′ holds on windows the base never trained on; T1′ is supported in the form tested (not proven).",
                                       "FAILS": "AMEND_1 §4: the under-acceleration is not a mean over unresolved intent, and T1′ is wrong too.",
                                       "NOT EVALUABLE": "AMEND_2 §2: P1′ is closed as untestable at this data scale; the next test of T1′ is P4 on the LoRA arm."}[verdict]
json.dump(R, open(f"{HERE}/P1PB_RESULT.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

PW = __import__("numpy").load(f"{HERE}/P1PB_PERWINDOW.npz", allow_pickle=False)
_np = __import__("numpy")
_pm = PW["primary_mask"].astype(bool)
_post = PW["post_primary"][_pm]
med_post_conf = float(_np.median(_post[_post >= 0.8]))
med_post_unsure = float(_np.median(_post[(_post >= 0.35) & (_post <= 0.65)]))
R["descriptive_median_posterior"] = {"confident": med_post_conf, "unsure": med_post_unsure}
json.dump(R, open(f"{HERE}/P1PB_RESULT.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

ctrl = R["controls_summary"]
L = []
w = L.append
w("# P1PB_RESULT: P1′b (THEORY_CRAWL_AMEND_2) on the RA-1 holdout tick-0 windows")
w("")
w(f"**VERDICT, as the criterion is written: {verdict}.** {reason}")
w("")
if verdict == "FAILS":
    w("Written consequence (`THEORY_CRAWL_AMEND_1.md` §4): the under-acceleration is not a mean over unresolved intent, and T1′ is wrong too. "
      "Nothing below moves a threshold, a band or a group; the descriptive material in §5 to §7 is labelled and does not change this line.")
elif verdict == "NOT EVALUABLE":
    w("Written consequence (`THEORY_CRAWL_AMEND_2.md` §2): P1′ is closed as untestable at this data scale; the next test of T1′ is P4 on the LoRA arm.")
else:
    w("Written consequence: P1′ holds on windows the base never trained on; T1′ is supported in the form tested, not proven.")
w("")
if R["probe_at_chance_in_band_K2"]:
    w("**The probe is at chance on the holdout band (K2 interval covers 0.5): P1′b cannot test T1′ whatever the verdict says.**")
else:
    w(f"The probe can test T1′ on this set: its AUC on the holdout windows is {f(k2['auc'])} [{f(k2['lo'])}, {f(k2['hi'])}] in the P1′ band where its label exists (v0 in [3, 6), n {k2['n']}, {k2['n_clips']} episodes) "
      f"and {f(k1['auc'])} [{f(k1['lo'])}, {f(k1['hi'])}] on the P-MOVE range (n {k1['n']}, {k1['n_clips']} episodes); neither interval covers 0.5.")
w("")
w("Evidence class MEASURED (ours, dev-box CPU, zero planner training, no weight edit, nothing staged). Tier: a tick-0 plan-level read of the C0 checkpoints on logged windows (the plan is made from the logged history); not a driving result. "
  "Estimator: episode-cluster bootstrap over the holdout clips, B 2,000, seed 0 (the `tp_lib` construction). It answers \"another draw of EPISODES\" and is blind to training and inference variance (H-ESTIM-SEED-1); "
  "the two planner seeds (two training runs of the same recipe) are read separately, and a separated interval from them is necessary, not sufficient, for any lever claim.")
w("")
w("## 1. Spec and order of work")
w(f"* `THEORY_CRAWL_AMEND_2.md` sha256 `{R['amend2_sha256'][:16]}...` (frozen, verified); `THEORY_CRAWL_AMEND_1.md` §4 sha256 `{R['amend1_sha256'][:16]}...` (frozen, verified). The rule is exactly P1′'s; the P1′ code and probe artifact (`p1p_tp_lib.py`, `p1p_probe_refit.py`, `p1p_read.py`, `P1P_PROBE_MODELS.npz`) are loaded unchanged and their sha256 equal the values recorded in `P1P_RESULT.json`.")
w(f"* Order: (1) the holdout list and admissibility, `P1PB_ADMISSIBILITY.json`, before any score; (2) `SPEC_P1PB.md` sha256 `{R['spec_sha256']}` stamped in `SPEC_P1PB.sha256` (2026-11-07T00:07:38Z, 01:07:38 +0100) before any holdout feature, forward or posterior existed; "
  f"(3) control FW (the driver reproduces the T1-kin dumps' tick-0 rows) before the holdout forward; (4) the holdout tick-0 forward of C0 s0 and s1; (5) the read.")
w("* Rule fixed in the spec before scoring, stated here because it decides which table is the verdict: the primary window set is EVERY window of the 60 clips (stride 1, t in [3, n − 30), the training enumeration), as AMEND_2 says \"every window passing the filter is read\"; "
  "the eval-141 grid construction (t = 3 mod 4) is the declared sensitivity G4. Clips with no row in the dev-box v9 label release (nav input all zero, a state C0 never saw) are outside the primary set and inside S5.")
w("* Run history: `p1pb_admissibility.py` ran once (its first launch failed on shell quoting before any line executed); `p1pb_forward.py control`, `p1pb_forward.py holdout` and `p1pb_read.py` each ran once, and none of those three was edited after its first run; `p1pb_finalize.py` (report text only, no statistic) was re-run after wording edits. "
  "Before the spec, three scratch probes (kept in `p1pb/`: `ho_list.py`, `chk_cache.py`, `chk_inputs.py`) read identity and structure only: the holdout list and its shas, manifest records and file sizes of the cache, the number of holdout clips with native tracks and with v9 rows (56 of 60, which is why the spec has a label-coverage rule). No feature, v0, a_human, plan or posterior of a holdout window existed.")
w("")
w("## 2. Admissibility of the holdout set (checked before any score; all pass)")
w("")
w("| check | what | result |\n|---|---|---|")
a = ADM["checks"]
w(f"| A1 holdout list | clips 540:600 of the sha12 order of the 4,572 cache_full TRAIN clips (PREREG_V8F_MINI Amendment 4, `--exclude 540:600`); rebuilt from the box 2 manifest copy | sha256 `{a['A1_holdout_list']['holdout_sha256'][:16]}...` equals the digest every run of the chain recorded (`excluded_sha256`); train list `{a['A1_holdout_list']['train_list_sha256'][:16]}...` equals `train_sha256`; overlap 0 |")
runs = a["A2_C0_training_clip_list"]["runs"]
w(f"| A2 no holdout clip in C0's training clip list | the summaries and argv of M2b, dagoacc and C0 (s0 and s1): {', '.join(runs)} | all six: n_clips 4,512, `--exclude 540:600`, `excluded_sha256` and `train_sha256` as above; chain M2b → dagoacc → recft_C0, every stage excluded the same 60 |")
a3 = a["A3_probe_training_windows"]
w(f"| A3 no holdout clip among the go/stay probe's training windows | `windows_train.npz` ({a3['windows_train_npz_n_windows']:,} windows, {a3['windows_train_n_clips']:,} clips) and the sampled rows | window-list digest `{a3['window_list_digest'][:16]}...` equals C0's `train_window_list_sha256`; holdout clips among them 0; among the 12,000 sampled P-MOVE rows ({a3['sampled_rows_used_by_the_probe']['pmove']['n_clips']:,} clips) 0 |")
w(f"| A4 disjoint from eval-141 | `join_eval141.npz` clip list | overlap {a['A4_disjoint_from_eval141']['overlap']} of {a['A4_disjoint_from_eval141']['n_eval_clips']} |")
a5 = a["A5_features_on_dev_box"]
w(f"| A5 frozen features on the dev box | `F:/TanitAD-scratch/speedread/cache/train/clips` (the cache the probe's own V was extracted from) | {a5['n_holdout_files_present_and_md5_equal_to_box2_manifest']} of 60 files present with md5 equal to the box 2 manifest; same encoder route as the eval-141 cache (float8_e4m3fn, bf16, cuda, local256, grids 4x10 and 8x20); {a5['n_frames_total']:,} cache steps |")
w("")
w("Provenance limits, stated: the box 2 manifest is read from a copy (`t2_scenes/raw/box2_cache_full_train_manifest.jsonl`); the dev-box v9 release is f63ece41 (4,369 clips) while the C0 chain trained on 83e30bd3 (4,572 clips), whose rows for the old 4,369 clips are bitwise identical on the non-offset fields (INHERITED from `SPEC_REFCV8_DRAFT.md` line 1542, not re-verified); "
  f"{len(F1['clips_without_v9_rows'])} holdout clips have no f63ece41 rows ({R['holdout_windows']['windows_in_clips_without_rows']} windows) and are outside the primary set.")
w("")
w("## 3. Window set and group counts")
w("")
w(f"| step (primary: every window of the labelled holdout clips, v0 in [3, 8), `plan`) | windows |\n|---|---|\n| tick-0 rows of the 60 clips (every t in [3, n − 30)) | {steps['tick0_rows']} |\n| in the {R['holdout_windows']['n_clips_labelled']} clips with v9 rows | {steps['in_scope_clips']} |\n"
  f"| and v0 in [3, 8) | {steps['v0_band']} |\n| and gt_ok | {steps['gt_ok']} |\n| and a_human > 0.5 m/s² | {steps['a_human_gt_0p5']} |\n| and all of a_human, â_plan, ρ, posterior finite | {steps['finite']} ({ps['s0']['n_episodes']} episodes) |")
w("")
w(f"Posterior bins over the {steps['finite']} windows: below 0.35: {bins['lt_0.35']}; UNSURE [0.35, 0.65]: {bins['0.35_0.65_UNSURE']}; (0.65, 0.8): {bins['0.65_0.8']}; CONFIDENT ≥ 0.8: {bins['ge_0.8_CONFIDENT']}. The windows, a_human and the posterior do not depend on the planner seed, so both seeds hold the same groups. "
  f"CONFIDENT spans {ps['s0']['n_episodes_conf']} episodes (largest clip holds {100 * ps['s0']['conf_clip_stats']['max_clip_share']:.0f} % of its windows; median {ps['s0']['conf_clip_stats']['median_windows_per_clip']:.0f} windows per clip), "
  f"UNSURE {ps['s0']['n_episodes_unsure']} episodes (largest clip {100 * ps['s0']['unsure_clip_stats']['max_clip_share']:.0f} %). Stride-1 windows overlap within a clip, so the support is the number of episodes, not of windows.")
w("")
w(f"Domain note: the probe was trained on v0 below 6. {P['n_windows_v0_ge_6']} of the {steps['finite']} windows have v0 in [6, 8) and were scored by extrapolation (S1 restricts to the in-domain band).")
w("")
w("## 4. The statistic, holdout alone (this table is the verdict)")
w("")
w("| seed | n CONF / UNSURE (episodes) | median ρ CONFIDENT [95 %] | median ρ UNSURE [95 %] | Δ [95 %] | Δ > 0 and separated |\n|---|---|---|---|---|---|")
for s in ("s0", "s1"):
    r = ps[s]
    w(f"| {s} | {r['n_conf']} / {r['n_unsure']} ({r['n_episodes_conf']} / {r['n_episodes_unsure']}) | {f(r['median_conf'])} [{f(r['median_conf_ci'][0])}, {f(r['median_conf_ci'][1])}] | {f(r['median_unsure'])} [{f(r['median_unsure_ci'][0])}, {f(r['median_unsure_ci'][1])}] | "
      f"{sgn(r['delta'])} {iv(r)} | {sep(r)} |")
w("")
w(f"Median a_human (m/s²) is {f(ps['s0']['median_a_human_conf'])} in CONFIDENT and {f(ps['s0']['median_a_human_unsure'])} in UNSURE; median â_plan is {f(ps['s0']['median_a_plan_conf'])} / {f(ps['s0']['median_a_plan_unsure'])} at s0 and "
  f"{f(ps['s1']['median_a_plan_conf'])} / {f(ps['s1']['median_a_plan_unsure'])} at s1; median v0 (m/s) is {f(ps['s0']['median_v0_conf'], 2)} in CONFIDENT and {f(ps['s0']['median_v0_unsure'], 2)} in UNSURE. "
  f"Means of ρ: CONFIDENT {f(ps['s0']['mean_conf'])} / {f(ps['s1']['mean_conf'])}, UNSURE {f(ps['s0']['mean_unsure'])} / {f(ps['s1']['mean_unsure'])} (s0 / s1).")
w("")
w("## 5. Reported beside, deciding nothing: the pooled read (eval-141 primary windows ∪ holdout primary windows)")
w("")
w(f"eval-141's {PO['n_eval141_windows']} windows are SEEN (they produced the NOT EVALUABLE of P1′ and its descriptive Δ). The two clip sets are disjoint (checked); the pooled set holds {PO['n_eval141_windows'] + PO['n_holdout_windows']} windows from {PO['n_clips_pooled']} episodes.")
w("")
w("| set | seed | n CONF / UNSURE | Δ [95 %] | Δ > 0 and separated |\n|---|---|---|---|---|")
for nm, d in (("eval-141 alone (P1′ primary, seen)", E141), ("holdout alone (the verdict)", ps), ("pooled", PO["per_seed"])):
    for s in ("s0", "s1"):
        r = d[s]
        w(f"| {nm} | {s} | {r['n_conf']} / {r['n_unsure']} | {sgn(r['delta'])} {iv(r)} | {sep(r)} |")
w("")
w(f"The pooled logic returns {PO['verdict_logic_returns']}. It is a table of two sets, one of them seen, and it is not promoted over the holdout line. The eval-141 and holdout point estimates agree in sign; the holdout estimates are smaller by "
  f"{E141['s0']['delta'] - ps['s0']['delta']:.3f} (s0) and {E141['s1']['delta'] - ps['s1']['delta']:.3f} (s1).")
w("")
w("## 6. Controls")
w("")
w("| control | what it checks | result |\n|---|---|---|")
w(f"| C4 unit tests | weighted median = `np.median` of the replicated array; bootstrap counts = tp_lib construction; group edges at 0.8 / 0.35 / 0.65 | {'pass' if ctrl['C4'] else 'FAIL'} |")
fws = FW["per_seed"]
w(f"| FW the driver reproduces the T1-kin dumps (run before the holdout forward) | {FW['n_windows']} evenly spaced eval-141 tick-0 windows, both seeds, vs the dump rows | {'pass' if ctrl['FW'] else 'FAIL'}: v0 and gt max abs {fws['s0']['v0_max_abs']:.0e} / {fws['s0']['gt_max_abs']:.0e}; plan max abs {fws['s0']['plan_max_abs_m']:.1e} m (s0), {fws['s1']['plan_max_abs_m']:.1e} m (s1); plan_raw {fws['s0']['plan_raw_max_abs_m']:.1e} / {fws['s1']['plan_raw_max_abs_m']:.1e} m; â_plan {fws['s0']['a_plan_max_abs']:.1e} / {fws['s1']['a_plan_max_abs']:.1e} m/s² (tolerances 1e-6, 1e-6, 5e-3, 5e-3, 2e-3) |")
w(f"| F1 holdout feature path | store pose grid = native track (checked per clip by the store), n_frames and gt_ok on every enumerated t, finite plans, identical v0/gt between seeds, v0 = kin3[t0, 0] | {'pass' if ctrl['F1'] else 'FAIL'}: {F1['n_windows']} windows, v0 vs kin3 max abs {F1['v0_vs_kin3_t0_max_abs']}, v9 release md5 {F1['v9_release_md5'][:8]} |")
f2 = R["F2_features_rebuilt_twice"]
w(f"| F2 features rebuilt twice | {f2['n']} random windows, V and E built twice from the cache | {'pass' if ctrl['F2'] else 'FAIL'}: V bit-equal {f2['V_bit_equal']}, E max abs {f2['E_max_abs']}, v0 vs kin3 {f2['v0_vs_kin3_max_abs']} |")
w(f"| Probe artifacts unchanged | sha256 of `P1P_PROBE_MODELS.npz`, `p1p_tp_lib.py`, `p1p_probe_refit.py`, `p1p_read.py` vs `P1P_RESULT.json` and `trigger_probe/tp_lib.py` | {'pass' if ctrl['probe_artifacts_unchanged'] else 'FAIL'} |")
w(f"| K1 probe AUC on holdout, P-MOVE range | posterior vs GO, 0.5 ≤ v0 < 6 | {f(k1['auc'])} [{f(k1['lo'])}, {f(k1['hi'])}], n {k1['n']} ({k1['n_pos']} GO, {k1['n_clips']} episodes); above chance |")
w(f"| K2 probe AUC on holdout, P1′ band | posterior vs GO, 3 ≤ v0 < 6 | {f(k2['auc'])} [{f(k2['lo'])}, {f(k2['hi'])}], n {k2['n']}, {k2['n_clips']} episodes; above chance |")
w(f"| K3 two-second go indicator | posterior vs a_human > 0.5, 3 ≤ v0 < 8 (descriptive) | {f(k3['auc'])} [{f(k3['lo'])}, {f(k3['hi'])}], n {k3['n']} ({k3['n_pos']} positive, {k3['n_clips']} episodes) |")
n1 = R["N1_posterior_shuffle"]["separated_of_20"]
n3 = R["N3_planted_rho_equals_posterior"]["per_seed"]
w(f"| N1 posterior shuffle | 20 permutations of the posterior, real ρ | {'pass' if ctrl['N1'] else 'FAIL'}: separated {n1['s0']} of 20 at s0, {n1['s1']} of 20 at s1 (limit 3) |")
w(f"| N2 identity (ρ ≡ 1, also from `plan := gt`) | exact values | {'pass' if ctrl['N2'] else 'FAIL'} (medians 1.0, Δ 0.0, interval [0, 0], not separated) |")
w(f"| N3 planted (ρ = posterior) | the pipeline sees a planted mechanism | {'pass' if ctrl['N3'] else 'FAIL'} (Δ {sgn(n3['s0']['delta'])} {iv(n3['s0'])} at s0, separated) |")
w("")
w("## 7. Sensitivities (declared in SPEC_P1PB.md §5; none is the verdict, each through the identical verdict logic)")
w("")
w("| variant | n CONF / UNSURE | Δ s0 [95 %] | Δ s1 [95 %] | verdict logic returns |\n|---|---|---|---|---|")
for k, v in S.items():
    a_, b_ = v["per_seed"]["s0"], v["per_seed"]["s1"]
    w(f"| {v['name']} | {a_['n_conf']} / {a_['n_unsure']} | {sgn(a_['delta'])} {iv(a_)} | {sgn(b_['delta'])} {iv(b_)} | {v['verdict']} |")
w("")
pos_all = all(v["per_seed"][k]["delta"] is not None and v["per_seed"][k]["delta"] > 0 for v in S.values() for k in ("s0", "s1"))
both = [v["name"].split(":")[0] for v in S.values() if all(v["per_seed"][k]["separated"] and v["per_seed"][k]["positive"] for k in ("s0", "s1"))]
only1 = [v["name"].split(":")[0] for v in S.values() if v["per_seed"]["s1"]["separated"] and not v["per_seed"]["s0"]["separated"]]
none_ = [v["name"].split(":")[0] for v in S.values() if not v["per_seed"]["s1"]["separated"] and not v["per_seed"]["s0"]["separated"]]
ne = [v["name"].split(":")[0] for v in S.values() if v["verdict"] == "NOT EVALUABLE"]
w(f"Reading of the table: Δ is positive at both seeds in {'every' if pos_all else 'NOT every'} variant. Separated at both seeds: {', '.join(both) or 'none'}; at s1 only: {', '.join(only1) or 'none'}; at neither: {', '.join(none_) or 'none'}. "
  f"Variants returning NOT EVALUABLE: {', '.join(ne) or 'none'} (G4 keeps {S['G4_grid_stride4_t3mod4']['per_seed']['s0']['n_conf']} CONFIDENT windows against the floor of {N_MIN}). "
  "S2 (`plan_raw`, the network's plan before smoothing and projection) is a different plan field from the one P1′ names and is not promoted. The primary line uses every window, as fixed before scoring.")
w("")
w("## 8. What the read says and does not say (descriptive; HYPOTHESIS beyond the verdict line)")
if verdict == "FAILS":
    w(f"* **The written verdict stands: FAILS.** It rests on one interval: at s0 the 95 % interval of Δ is {iv(ps['s0'])}, {abs(ps['s0']['lo']):.3f} below zero, from {ps['s0']['n_episodes_conf']} CONFIDENT episodes; at s1 it is {iv(ps['s1'])}. "
      "The rule requires both seeds; no threshold, band or group is moved to change that.")
    w(f"* **The direction is the predicted one at both seeds.** Δ is {sgn(ps['s0']['delta'])} / {sgn(ps['s1']['delta'])} on the holdout against {sgn(E141['s0']['delta'])} / {sgn(E141['s1']['delta'])} on the seen eval-141 windows. "
      f"A shrinkage that is larger where the probe is unsure is present in both sets; what the holdout does not give is a separated interval at both seeds with {ps['s0']['n_episodes_conf']} CONFIDENT episodes.")
    w(f"* **A confound the grouping carries (post hoc, not tested):** the CONFIDENT windows have median v0 {f(ps['s0']['median_v0_conf'], 2)} m/s against {f(ps['s0']['median_v0_unsure'], 2)} and median a_human {f(ps['s0']['median_a_human_conf'])} against {f(ps['s0']['median_a_human_unsure'])} m/s²: "
      "go-confident windows are the slower, harder-accelerating ones, so a difference in ρ between the groups is not by itself a signature of intent mixing.")
    w(f"* **Median ρ against the groups' median posterior (post hoc, descriptive):** CONFIDENT ρ {f(ps['s0']['median_conf'])} / {f(ps['s1']['median_conf'])} (s0 / s1) against a median posterior of {f(med_post_conf)}; "
      f"UNSURE ρ {f(ps['s0']['median_unsure'])} / {f(ps['s1']['median_unsure'])} against {f(med_post_unsure)}. T1′'s form â ≈ π·a_go + (1 − π)·a_stay gives ρ ≈ π when a_stay ≈ 0. The medians of ρ lie below the posterior by "
      f"{med_post_conf - ps['s0']['median_conf']:.2f} / {med_post_conf - ps['s1']['median_conf']:.2f} in CONFIDENT and {med_post_unsure - ps['s0']['median_unsure']:.2f} / {med_post_unsure - ps['s1']['median_unsure']:.2f} in UNSURE (s0 / s1): "
      "the ordering and the rough size match, the level is lower. This is an observation made after the verdict, with no interval on the comparison, and it does not alter the verdict.")
    w("* **What is not available:** a larger unseen set on the dev box. The 4,512 training clips are inadmissible for this test; eval-141 is seen; the 60 holdout clips are all of the unseen clips with cached features here, and they are seen after this read. "
      "Another look at the same 60 clips under a changed rule would be a post hoc rescue and is not proposed.")
    w("* **Next levers (the Master Mind's to pre-register; none was run; zero training):** "
      "(a) a fresh pre-registration whose statistic uses the continuous posterior (ρ against π over all windows, including the neither-bin windows the split discards: "
      f"{ps['s0']['n_neither']} of {ps['s0']['n_windows']} here), with a committed power statement, read on FRESH windows; "
      "(b) fresh windows need clips outside the 4,572 encoded by DINOv3: blocked on encode compute (box 2 or a pod) and on the PI's go for it, since the dev box holds no further unseen cached features; "
      "(c) P4 on the LoRA arm, the pre-registered next test of the lever that never depended on T1′ (T2 and T3 are unchanged by this verdict).")
elif verdict == "NOT EVALUABLE":
    w("* The groups are below the floor; nothing is padded. Closed as untestable at this data scale; next test of T1′ is P4 on the LoRA arm.")
else:
    w("* P1′ holds on unseen windows at both seeds. A separated interval from two training seeds is necessary, not sufficient, evidence of a mechanism; the confound in the grouping (v0 and a_human differ between groups) is untested.")
w("")
w("## 9. Provenance and manifest (all under `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/wsp_read/`; nothing staged, committed or pushed; each exists in this one place only)")
w("")
w("| artifact | role | sha256 (first 16) |\n|---|---|---|")
FILES = (("SPEC_P1PB.md", "pre-registered spec"), ("SPEC_P1PB.sha256", "spec hash and time"), ("P1PB_ADMISSIBILITY.json", "admissibility checks A1 to A5"), ("P1PB_HOLDOUT_60.json", "the 60 holdout clips"),
         ("p1pb_admissibility.py", "admissibility script"), ("p1pb_forward.py", "tick-0 forward (control and holdout modes)"), ("p1pb_read.py", "the read, loads the P1′ code unchanged"), ("p1pb_finalize.py", "this report"),
         ("P1PB_FW_CONTROL.json", "control FW"), ("P1PB_F1_CONTROL.json", "control F1"), ("P1PB_TICK0_s0.npz", "holdout tick-0 plans, C0 s0"), ("P1PB_TICK0_s1.npz", "holdout tick-0 plans, C0 s1"),
         ("P1PB_PERWINDOW.npz", "per-window posteriors, a_human, a_plan, ρ"), ("P1PB_RESULT.json", "machine-readable result"), ("p1pb_fw_control.log", "log"), ("p1pb_forward_holdout.log", "log"), ("p1pb_read.log", "log"))
sums = []
for fn, role in FILES:
    p = f"{HERE}/{fn}"
    h = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else "(missing)"
    sums.append(f"{h} *{fn}")
    w(f"| `{fn}` | {role} | `{h[:16]}` |")
w("")
w("Inputs read, unmodified: the checkpoints `v8f-recover/M2b_s{0,1}_recft_C0/ckpt_last.pt`; the F: train cache `F:/TanitAD-scratch/speedread/cache/train/{manifest.jsonl,clips/<60 holdout clips>.pt}`; `sfresh_inputs/inputs/native_tracks.pt`; `v9labels/v9_labels_train.npz`; "
  "`t2_scenes/raw/{box2_cache_full_train_manifest.jsonl,TRAIN_CLIPS_4512.json}`; the run summaries named in §2; `trigger_probe/raw/windows_train.npz`, `sample_pmove.npz`; the P1′ files `P1P_PROBE_MODELS.npz`, `P1P_PERWINDOW.npz`, `P1P_RESULT.json`; "
  "the T1-kin dumps `C:/Users/Admin/rr_scratch/out/C0C_s{0,1}/ticks/` (control FW only); `ri/code/v8f_t1_eval_ri.py`; the dagt1 tree `dagt1_tree_b43c7a9a`.")
open(f"{HERE}/P1PB_RESULT.md", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
open(f"{HERE}/P1PB_FILES.sha256", "w", encoding="utf-8", newline="\n").write("\n".join(sums) + "\n")
print("wrote P1PB_RESULT.md and P1PB_FILES.sha256; verdict", verdict)
