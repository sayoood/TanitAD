"""Merge the pod measurement with the RAW-LOG references and emit the
decision-grade results JSON for post-mortem experiment B (ego-dropout zero-fill,
mask forced OFF vs ON at inference).

Stage 2 of 2. Stage 1 is ``postmortem_b_egodropout.py``, which runs on the pod
that holds the checkpoint and writes ``exp_b_{bf16,fp32,bf16_perm}.json``, plus
``postmortem_b_v0_stats.py`` -> ``v0_stats.json``. Copy those four next to each
other and point ``--raw-dir`` at them.

References are recomputed here from ``taniteval/results/trainlogs/*.jsonl``
(primary source), never copied from the post-mortem's prose.

  python taniteval/postmortem_b_analyze.py --raw-dir <dir with the pod JSONs>
"""
from __future__ import annotations
import argparse
import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "taniteval/results/trainlogs"
_ap = argparse.ArgumentParser()
_ap.add_argument("--raw-dir", required=True,
                 help="directory holding exp_b_bf16.json / exp_b_fp32.json / "
                      "exp_b_bf16_perm.json / v0_stats.json from the pod")
_ap.add_argument("--out", default=str(
    REPO / "taniteval/results/postmortem_b_egodropout_v3enc10k.json"))
_args = _ap.parse_args()
SP = Path(_args.raw_dir)
OUT = Path(_args.out)

KEYS = ["g_op_fwd_ade_m", "g_op_mid_de_m", "g_tac_fwd_ade_m", "g_tac_mid_de_m",
        "g_str_fwd_ade_m", "g_str_mid_de_m", "inv"]
ARMS = {"v1_speedjerk": "v1-speedjerk_train_log.jsonl",
        "v3enc": "v3enc_train_log.jsonl",
        "nospeed_phase0": "nospeed-phase0_train_log.jsonl"}
LO, HI = 8000, 10000


def bucket(fn):
    bystep = {}
    for line in (LOGS / fn).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            bystep[int(r["step"])] = r          # dedupe resumes: keep LAST
    sel = [r for s, r in bystep.items() if LO <= s < HI]
    out = {"n_log_rows": len(sel)}
    for k in KEYS:
        v = [r[k] for r in sel if k in r]
        if v:
            out[k] = round(st.fmean(v), 4)
    return out


ref = {a: bucket(f) for a, f in ARMS.items()}
for a, f in ARMS.items():
    cfg = json.loads((LOGS / f.replace("_train_log.jsonl", "_config.json"))
                     .read_text())
    ref[a]["action_dim"] = json.loads(cfg["cfg"])["predictor"]["action_dim"]

meas = {p: json.loads((SP / f"exp_b_{p}.json").read_text())
        for p in ("bf16", "fp32") if (SP / f"exp_b_{p}.json").exists()}
_pf = SP / "exp_b_bf16_perm.json"
PERM = json.loads(_pf.read_text()) if _pf.exists() else None
_vf = SP / "v0_stats.json"
V0 = json.loads(_vf.read_text()) if _vf.exists() else None

PRIM = "bf16"          # the training-faithful precision (trainer ran autocast bf16)
m = meas[PRIM]["means"]
v1 = ref["v1_speedjerk"]
ns = ref["nospeed_phase0"]
v3 = ref["v3enc"]

derived = {}
for lvl in ("op", "tac", "str"):
    k = f"g_{lvl}_fwd_ade_m"
    on_a, on_r, off, zero = (m["on25_analytic"][k], m["on25"][k],
                             m["off"][k], m["zero"][k])
    d = {
        "measured_mask_ON_analytic_p0.25": on_a,
        "measured_mask_ON_realised_draw": on_r,
        "measured_mask_OFF": off,
        "measured_all_rows_ZEROED": zero,
        "v1_ref_8-10k_bucket": v1.get(k),
        "v3enc_train_log_8-10k_bucket": v3.get(k),
        "nospeed_ref_8-10k_bucket": ns.get(k),
        # (i) assumption-free, internal to this checkpoint: what share of the
        #     MASKED metric is the mask itself?
        "artifact_share_of_masked_metric": round((on_a - off) / on_a, 4),
        "artifact_share_realised_draw": round((on_r - off) / on_r, 4),
        # (ii) the headline: how much of the ON-vs-v1 gap does mask-off close?
        "recovered_fraction_of_gap_vs_v1": round((on_a - off) / (on_a - v1[k]), 4)
        if v1.get(k) else None,
        "recovered_fraction_using_realised_draw": round(
            (on_r - off) / (on_r - v1[k]), 4) if v1.get(k) else None,
        # (iii) transported onto the LOGGED 8-10k buckets. MULTIPLICATIVE is the
        #     primary transport (the ckpt is one step, the bucket is an average
        #     over evolving weights, so a ratio travels better than a delta).
        "logged_gap_v3enc_minus_v1": round(v3[k] - v1[k], 4) if v1.get(k) else None,
        "corrected_v3enc_bucket_multiplicative": round(v3[k] * off / on_a, 4),
        "corrected_v3enc_bucket_additive": round(v3[k] - (on_a - off), 4),
        "recovered_fraction_of_LOGGED_gap_multiplicative": round(
            (v3[k] - v3[k] * off / on_a) / (v3[k] - v1[k]), 4) if v1.get(k) else None,
        "recovered_fraction_of_LOGGED_gap_additive": round(
            (on_a - off) / (v3[k] - v1[k]), 4) if v1.get(k) else None,
        "logged_matched_step_ratio_to_v1": round(v3[k] / v1[k], 3)
        if v1.get(k) else None,
        "corrected_matched_step_ratio_multiplicative": round(
            v3[k] * off / on_a / v1[k], 3) if v1.get(k) else None,
        "corrected_matched_step_ratio_additive": round(
            (v3[k] - (on_a - off)) / v1[k], 3) if v1.get(k) else None,
        "ratio_to_v1_measured_mask_ON": round(on_a / v1[k], 3) if v1.get(k) else None,
        "ratio_to_v1_measured_mask_OFF": round(off / v1[k], 3) if v1.get(k) else None,
        # the post-mortem's level-free statistic (nospeed - arm)/nospeed
        "speed_benefit_recovered_mask_ON": round((ns[k] - on_a) / ns[k], 4)
        if ns.get(k) else None,
        "speed_benefit_recovered_mask_OFF": round((ns[k] - off) / ns[k], 4)
        if ns.get(k) else None,
        "speed_benefit_recovered_v1": round((ns[k] - v1[k]) / ns[k], 4)
        if ns.get(k) and v1.get(k) else None,
        # the like-for-like version: v3enc's LOGGED bucket rescaled by off/on,
        # compared against the no-speed arm's LOGGED bucket (both log-side)
        "speed_benefit_recovered_v3enc_corrected_bucket": round(
            (ns[k] - v3[k] * off / on_a) / ns[k], 4) if ns.get(k) else None,
        "speed_benefit_as_pct_of_v1_logged": round(
            (ns[k] - v3[k]) / (ns[k] - v1[k]), 4) if ns.get(k) else None,
        "speed_benefit_as_pct_of_v1_corrected": round(
            (ns[k] - v3[k] * off / on_a) / (ns[k] - v1[k]), 4)
        if ns.get(k) else None,
        "paired_ON_minus_OFF": meas[PRIM]["paired"]["on25_analytic_minus_off"][k],
        "paired_ZERO_minus_OFF": meas[PRIM]["paired"]["zero_minus_off"][k],
        "v0_sensitivity_x2_minus_off": meas[PRIM]["paired"]["x2_minus_off"][k],
        "fp32_replicate": ({"off": meas["fp32"]["means"]["off"][k],
                            "zero": meas["fp32"]["means"]["zero"][k],
                            "on25_analytic":
                                meas["fp32"]["means"]["on25_analytic"][k],
                            "artifact_share": round(
                                (meas["fp32"]["means"]["on25_analytic"][k]
                                 - meas["fp32"]["means"]["off"][k])
                                / meas["fp32"]["means"]["on25_analytic"][k], 4)}
                           if "fp32" in meas else None),
    }
    if PERM is not None and V0 is not None:
        rz = V0["perturbation_rms_scaled"]["zero_fill  RMS(v-0)"]
        rp = V0["perturbation_rms_scaled"]["within_batch_perm RMS(v_i-v_j)"]
        dz = PERM["means"]["zero"][k] - PERM["means"]["off"][k]
        dp = PERM["means"]["perm"][k] - PERM["means"]["off"][k]
        d["supplementary_run_with_v0_PERMUTED"] = {
            "off": PERM["means"]["off"][k], "zero": PERM["means"]["zero"][k],
            "perm": PERM["means"]["perm"][k],
            "paired_perm_minus_off": PERM["paired"]["perm_minus_off"][k],
            "input_perturbation_rms_scaled": {"zero": rz, "perm": rp},
            "damage_per_unit_rms_perturbation": {
                "zero": round(dz / rz, 4), "perm": round(dp / rp, 4),
                "perm_over_zero": round((dp / rp) / (dz / rz), 3)},
            "reading": "the zero-fill is a LARGER input perturbation than the "
                       "permutation yet does LESS damage per unit -- the model "
                       "has a special-case behaviour for v0 == 0, i.e. it built "
                       "an implicit (bad) null embedding for the mask value."}
        if k == "g_op_fwd_ade_m":
            lit = V0["literal_following_ADE_m_over_op_fwd_k4"]
            base = PERM["means"]["off"][k]
            d["realised_fraction_of_a_literal_speed_follower"] = {
                "definition": "excess ADE over the correctly-conditioned run, "
                              "divided by the ADE a rollout would incur if it "
                              "tracked the FED speed exactly (0.25 * E|v_fed - "
                              "v_true| over op fwd_k=4 at 0.1 s/step). 1.0 = the "
                              "model swallows the lie whole; 0.0 = it ignores it.",
                "zero": round((PERM["means"]["zero"][k] - base) / lit["zero"], 4),
                "perm": round((PERM["means"]["perm"][k] - base) / lit["perm"], 4),
                "x2": round((PERM["means"]["x2"][k] - base) / lit["x2"], 4),
                "literal_ade_m": lit,
                "reading": "the model realises only ~21 % of a v0=0 error but "
                           "67-72 % of any OTHER v0 error -- a 3.2-3.4x "
                           "asymmetry at exactly the mask's sentinel value."}
    derived[k] = d

# ---- encoder-grounding term: NULL BY CONSTRUCTION -------------------------
derived["encoder_grounding_terms"] = {
    "note": "g_*_mid_de_m reads ONLY (z_t, fut_states) and NO actions "
            "(metric_dynamics.py:362-372). It is EXACTLY invariant to the mask; "
            "the harness verified max|delta| == 0.0 for all three levels. This "
            "is a structural identity, not an empirical null.",
    "measured": {f"g_{l}_mid_de_m": m["off"][f"g_{l}_mid_de_m"]
                 for l in ("op", "tac", "str")},
    "max_abs_delta_across_conditions":
        meas[PRIM]["harness_checks"]["mask_invariance_max_abs_delta"],
    "v1_ref_8-10k_bucket": {f"g_{l}_mid_de_m": v1[f"g_{l}_mid_de_m"]
                            for l in ("op", "tac", "str")},
    "v3enc_train_log_8-10k_bucket": {f"g_{l}_mid_de_m": v3[f"g_{l}_mid_de_m"]
                                     for l in ("op", "tac", "str")},
}

# ---- inv ------------------------------------------------------------------
derived["inv"] = {
    "measured_mask_ON_analytic_p0.25": m["on25_analytic"]["inv"],
    "measured_mask_ON_realised_draw": m["on25"]["inv"],
    "measured_mask_OFF": m["off"]["inv"],
    "measured_all_rows_ZEROED": m["zero"]["inv"],
    "v1_ref_8-10k_bucket": v1["inv"],
    "v3enc_train_log_8-10k_bucket": v3["inv"],
    "nospeed_ref_8-10k_bucket": ns["inv"],
    "recovered_fraction_of_gap_vs_v1": round(
        (m["on25_analytic"]["inv"] - m["off"]["inv"])
        / (m["on25_analytic"]["inv"] - v1["inv"]), 4),
    "paired_ON_minus_OFF": meas[PRIM]["paired"]["on25_analytic_minus_off"]["inv"],
    "paired_ZERO_minus_OFF": meas[PRIM]["paired"]["zero_minus_off"]["inv"],
    "paired_v0channel_ZERO_minus_OFF":
        meas[PRIM]["paired"]["zero_minus_off"]["inv_ch2"],
    "fp32_replicate": ({"off": meas["fp32"]["means"]["off"]["inv"],
                        "on25_analytic":
                            meas["fp32"]["means"]["on25_analytic"]["inv"]}
                       if "fp32" in meas else None),
    "per_channel_mask_OFF": {"steer": m["off"]["inv_ch0"],
                             "accel": m["off"]["inv_ch1"],
                             "v0": m["off"]["inv_ch2"]},
    "per_channel_all_ZEROED": {"steer": m["zero"]["inv_ch0"],
                               "accel": m["zero"]["inv_ch1"],
                               "v0": m["zero"]["inv_ch2"]},
    "steer_accel_only_2ch_mean_mask_OFF": round(
        (m["off"]["inv_ch0"] + m["off"]["inv_ch1"]) / 2, 5),
    "COMPARABILITY_DEFECT": {
        "finding": "the no-speed control (flagship4b-phase0-30k) has "
                   "predictor.action_dim = 2, so ITS `inv` is a mean over "
                   "(steer, accel) ONLY. v1 and v3enc have action_dim = 3 and "
                   "average a THIRD channel (v0). The post-mortem's line "
                   "'v3enc 0.378 vs no-speed 0.364' compares a 3-channel mean "
                   "to a 2-channel mean -- they are not the same statistic.",
        "action_dim": {a: ref[a]["action_dim"] for a in ARMS},
        "like_for_like_2ch": {
            "v3enc@10k_mask_OFF": round(
                (m["off"]["inv_ch0"] + m["off"]["inv_ch1"]) / 2, 5),
            "nospeed_8-10k_bucket_2ch": ns["inv"]},
    },
}
if V0 is not None:
    f0 = V0["v0_mps"]["frac_below_0.5mps"]
    shown = 0.25 + 0.75 * f0
    derived["v0_distribution_and_the_ambiguity_of_0.0"] = {
        **V0,
        "genuinely_stopped_frac_v0_below_0.5mps": f0,
        "frac_of_windows_PRESENTED_as_v0_zero_under_p0.25": round(shown, 4),
        "of_those_genuine": round(f0 / shown, 4),
        "of_those_a_LIE": round(1 - f0 / shown, 4),
        "reading": "0.0 m/s is not a rare corner: 6.45 % of train windows are "
                   "genuinely stopped. The p=0.25 mask raises the share of "
                   "windows shown v0=0 to 29.8 %, of which 78.4 % are lies. "
                   "That is the quantitative form of the post-mortem's "
                   "'0.0 is an in-distribution value' claim.",
    }
derived["a5_v0_readout"] = {
    "pred_r2_on_train_batches": meas[PRIM]["derived"].get(
        "a5_v0_pred_r2_trainbatches"),
    "note": "R2 of the A5 inverse-dynamics head's own v0 output against the "
            "true v0/10 over the sampled train windows. NOT a fitted probe -- "
            "it is the model's own prediction, so no in-sample fit is involved; "
            "it is however TRAIN data, not held-out.",
}

res = {
    "experiment": "post-mortem experiment B - ego-dropout zero-fill, "
                  "mask forced OFF vs ON at inference",
    "spec": "2026-07-21-flagship-v3enc-postmortem.md sec.9 row B",
    "date": "2026-07-21",
    "arm": "flagship4b-v3enc-30k @ ckpt_step10000.pt (read-only)",
    "no_training": True, "optimizer_steps": 0,
    "primary_precision": PRIM,
    "supplementary_perm_run": ({k: PERM[k] for k in
                                ("n_windows", "n_episodes", "precision",
                                 "conditions", "means", "paired")}
                               if PERM else None),
    "measurement": {p: {k: meas[p][k] for k in
                        ("n_windows", "n_episodes", "batch_size", "n_batches",
                         "corpus", "precision", "seed", "realised_keep_frac",
                         "estimator", "harness_checks",
                         "xcheck_per_window_vs_grounding_losses", "means",
                         "paired", "ci_module_md5")}
                    for p in meas},
    "reference_bucket_means_8-10k": {
        "source": "taniteval/results/trainlogs/*_train_log.jsonl (RAW), rows "
                  "deduped on `step` keeping the LAST occurrence (v1 and the "
                  "no-speed arm replay steps after a resume), arithmetic mean "
                  "over the log rows in [8000, 10000).",
        **ref},
    "derived": derived,
}
OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
print(f"wrote {OUT}")
for k in ("g_op_fwd_ade_m", "g_tac_fwd_ade_m", "g_str_fwd_ade_m"):
    print(f"\n{k}: " + json.dumps(
        {kk: vv for kk, vv in derived[k].items()
         if not isinstance(vv, dict)}, indent=1))
print("\ninv: " + json.dumps({kk: vv for kk, vv in derived["inv"].items()
                              if not isinstance(vv, dict)}, indent=1))
