#!/usr/bin/env python3
"""PREREG_V8F_SFRESH analysis (zero training, CPU). Reads the banked --ages cycle evals, the re-run control, the --ages fresh evals and the
nav-only kinematic baseline, all on the SAME 2,399 windows / 141 episodes, and applies the PREREG outcome table verbatim.

  controls   banked-side identity checks that need no new eval (banked standard vs KIND-own; banked Q-LAYER vs compare.json)
  final      Q-STALE, Q-LAYER, beside rows, outcome table -> READ.json (+ numbers for SFRESH_RESULT.md)

Estimator: taniteval.ci.paired_episode_cluster_bootstrap via v8f_mini_eval.paired, B = 2000, seed 0, resampling the 141 EPISODES with the
models held fixed (it does NOT answer 'would another training run say this?': the two seeds are shown side by side for that).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(OUT, "code")
for p in (os.path.join(CODE, "stack"), os.path.join(CODE, "taniteval"), os.path.join(CODE, "stack", "scripts")):
    sys.path.insert(0, p)
PKG = os.path.join(CODE, "TanitAD Research Lab", "Architecture & Inference", "Research", "2026-10-17-v8f-mini")
RAW = os.path.join(PKG, "raw")
BANK = os.path.join(RAW, "full", "eval")
KIND = os.path.join(RAW, "2026-10-25-kind", "run", "eval")
EVD = os.path.join(OUT, "eval")                       # the new evals (fresh + control)
NB = 2000

import v8f_mini_eval as E  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from taniteval import v8f_consistency as C  # noqa: E402

ARCS = (30, 50, 80)


def ns():
    return argparse.Namespace(eval_grid=os.path.join(RAW, "eval_grid_windows.json"), eval_kin=os.path.join(RAW, "eval141_kin.npz"),
                              eval_join=os.path.join(RAW, "join_eval141.npz"), v9_eval=None, lead_eval=None, speed_eval=None,
                              v9_eval_md5=None, n_boot=NB, kin_mode="yaw")


def load_ctx():
    grid, J, X = E.ctx_from_args(ns(), "yaw")
    return grid, J, X


def jd(o, path):
    E.jdump(o, path)


def pr(c):                                           # a paired() result, compact
    return {"delta": c["mean"], "lo": c["lo"], "hi": c["hi"], "separated": bool(c["separated"]), "n_windows": c["n_windows"]}


def paired_masked(a, b, X, m):
    ok = X["rc_ok"][:, m]
    r = E.paired(a[ok, m], b[ok, m], X["eid"][ok], NB)
    r["n_episodes"] = int(len(np.unique(X["eid"][ok])))
    return r


def banked_cycle_npz(seed):
    return np.load(os.path.join(BANK, f"eval_M2b_s{seed}.npz"))


def kind_own_npz(seed):
    return np.load(os.path.join(KIND, f"eval_M2b_s{seed}-kind-own.npz"))


def baseline_errors(X):
    fl = np.load(os.path.join(BANK, "floors_perwindow.npz"))
    nb = fl["nav_baseline_ckpt"]
    return C.checkpoint_errors(nb, X["rc"], X["rc_ok"]), nb


def cmd_controls(a):
    grid, J, X = load_ctx()
    base, nb = baseline_errors(X)
    out = {"n_windows": int(len(X["eid"])), "n_episodes": int(len(np.unique(X["eid"])))}
    cmp_ = json.load(open(os.path.join(BANK, "compare.json")))
    out["banked_floors_nav_only_baseline_pos_err_mean"] = {f"@{arc}": float(np.nanmean(base["pos_err_m"][:, m])) for m, arc in enumerate(ARCS)}
    out["banked_standard_vs_kind_own"] = {}
    out["banked_qlayer_reproduction"] = {}
    for s in (0, 1):
        z, k = banked_cycle_npz(s), kind_own_npz(s)
        d = float(np.nanmax(np.abs(z["ck_pos_err"] - k["ck_pos_err"])))
        out["banked_standard_vs_kind_own"][f"s{s}"] = {"max_abs_diff_ck_pos_err": d, "max_abs_diff_pred": float(np.abs(z["pred"] - k["pred"]).max()),
                                                        "max_abs_diff_ckpt": float(np.abs(z["ckpt"] - k["ckpt"]).max())}
        for m, arc in ((1, 50), (2, 80)):
            r = paired_masked(z["ck_pos_err"], base["pos_err_m"], X, m)
            ref = cmp_["arms"]["M2b"]["strat"][f"vs_nav_only_baseline@{arc}"][s]
            out["banked_qlayer_reproduction"][f"s{s}@{arc}"] = {"recomputed": pr(r), "compare_json": {k2: ref[k2] for k2 in ("mean", "lo", "hi", "separated", "n_windows")},
                                                                 "match": bool(abs(r["mean"] - ref["mean"]) < 5e-4 and abs(r["lo"] - ref["lo"]) < 5e-4
                                                                               and abs(r["hi"] - ref["hi"]) < 5e-4 and r["n_windows"] == ref["n_windows"])}
    jd(out, os.path.join(OUT, "controls_banked.json"))
    print(json.dumps(out, indent=1))
    return 0


def heading_err(ck_now, X):
    return C.checkpoint_errors(ck_now, X["rc"], X["rc_ok"])["head_err_deg"]


def ds_for(seed):
    """the eval dataset exactly as cmd_run builds it (needed only to re-express the BANKED cycle checkpoints in NOW for heading errors)."""
    import torch
    from tanitad.models import v8f_mini as M
    ck = torch.load(os.path.join(OUT, "inputs", f"M2b_s{seed}", "ckpt_last.pt"), map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    tracks = D.load_native_tracks(os.path.join(OUT, "inputs", "native_tracks.pt"))
    ej = D.load_join(os.path.join(RAW, "join_eval141.npz"))
    store = D.ClipStore(os.path.join(OUT, "cache"), "eval", lru=32, grid=ck.get("grid", "pool4x10"), kin_mode=cfg.kin_mode, tracks=tracks)
    wins = D.eval_grid_windows(os.path.join(RAW, "eval_grid_windows.json"), ej)
    ds = D.V8FMiniWindows(store, ej, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    return ds, wins


def fam_digest(r):
    """the four families (+ ADE) from one eval json's `raw` block, as flat numbers."""
    f = r["families"]
    lon, lat, tac = f["longitudinal"], f["lateral"], f["tactical_from_trajectory"]
    d = {"ade6_m": r["traj"]["ade6_m"], "ade2_m": r["traj"]["ade2_m"], "fde6_m": r["traj"]["fde6_m"],
         "LONG": {k: lon.get(k) for k in ("speed_mae_mps", "speed_bias_mps", "along_mae_m", "along_final_bias_m", "accel_mae_mps2")},
         "LONG_target_speed_within_1.0_mps": (lon.get("target_speed_acc") or {}).get("within_1.0_mps"),
         "LAT": {k: lat.get(k) for k in ("heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm", "cross_mae_m", "cross_final_mae_m")},
         "TAC_from_trajectory": {"lateral_decision_accuracy": (tac.get("lateral_decision") or {}).get("accuracy"),
                                 "lateral_decision_kappa": (tac.get("lateral_decision") or {}).get("kappa")},
         "TAC_head": {"lat_macro_f1": r["tactical"]["lat"]["macro_f1"], "lon_macro_f1": r["tactical"]["lon"]["macro_f1"],
                      "goal_macro_f1": r["tactical"]["goal_macro_f1"]},
         "CONSISTENCY_a_any_rate": r["consistency_a"]["any"]["rate_all"],
         "CONSISTENCY_a_rates": {k: r["consistency_a"][k]["rate_all"] for k in ("SPEED", "STOP", "TURN", "LEAD") if k in r["consistency_a"]},
         "self_consistency_b": r["self_consistency_b"]["self_consistency"],
         "STRATEGIC_S_ckpt": {k: v["mean"] for k, v in r["strategic"].items() if isinstance(v, dict) and "mean" in v}}
    for k in ("longitudinal", "lateral", "tactical_from_trajectory"):
        if f[k].get("status") == "ERROR":
            d[k + "_ERROR"] = f[k]
    return d


def cmd_final(a):
    grid, J, X = load_ctx()
    base, nb = baseline_errors(X)
    n = len(X["eid"])
    res = {"prereg": "PREREG_V8F_SFRESH", "tip": "583cfc02", "n_windows": int(n), "n_episodes": int(len(np.unique(X["eid"]))), "n_boot": NB,
           "estimator": "paired episode-cluster bootstrap, B=2000, seed 0, 141 episodes (taniteval/ci.py); answers 'another draw of EPISODES'",
           "tier": "T1-class open loop, DIAGNOSTIC: fresh ages are not the deployed schedule", "seeds": {}}
    for s in (0, 1):
        cyc = banked_cycle_npz(s)                               # the banked --ages cycle eval (reference)
        fr = np.load(os.path.join(EVD, f"eval_M2b_s{s}-sfresh-fresh.npz"))
        ctl_p = os.path.join(EVD, f"eval_M2b_s{s}-sfresh-cycle.npz")
        S = {}
        # ---- Q-STALE / Q-LAYER at 30, 50, 80
        S["Q_STALE_fresh_minus_cycle_pos_err"] = {f"@{arc}": pr(paired_masked(fr["ck_pos_err"], cyc["ck_pos_err"], X, m)) for m, arc in enumerate(ARCS)}
        S["Q_LAYER_fresh_S_minus_navonly_pos_err"] = {f"@{arc}": pr(paired_masked(fr["ck_pos_err"], base["pos_err_m"], X, m)) for m, arc in enumerate(ARCS)}
        S["banked_cycle_S_minus_navonly_pos_err"] = {f"@{arc}": pr(paired_masked(cyc["ck_pos_err"], base["pos_err_m"], X, m)) for m, arc in enumerate(ARCS)}
        # ---- level numbers (single-arm means, episode-cluster CI) for the three arms
        lv = {}
        for nm, arr in (("cycle_banked", cyc["ck_pos_err"]), ("fresh", fr["ck_pos_err"]), ("navonly_baseline", base["pos_err_m"])):
            lv[nm] = {}
            for m, arc in enumerate(ARCS):
                ok = X["rc_ok"][:, m]
                c = E.episode_cluster_bootstrap(arr[ok, m], X["eid"][ok], n_boot=NB)
                lv[nm][f"@{arc}"] = {"mean": c["mean"], "lo": c["lo"], "hi": c["hi"], "n_windows": c["n_windows"], "n_episodes": c["n_episodes"]}
        S["levels_pos_err_m"] = lv
        # ---- heading (beside): banked cycle re-expressed in NOW with the evaluator's own function; fresh is already NOW
        ds, wins = ds_for(s)
        ck_cyc_now = E._ckpt_to_now(cyc["ckpt"], ds, wins, "cycle")
        chk = C.checkpoint_errors(ck_cyc_now, X["rc"], X["rc_ok"])["pos_err_m"]
        S["helper_check_cycle_pos_err_vs_banked_max_abs"] = float(np.nanmax(np.abs(chk - cyc["ck_pos_err"])))
        h_cyc = heading_err(ck_cyc_now, X)
        h_fr = heading_err(fr["ckpt"], X)                       # fresh: S tick frame == NOW frame (ages 0)
        h_base = C.checkpoint_errors(nb, X["rc"], X["rc_ok"])["head_err_deg"]
        S["heading_deg"] = {"STALE_fresh_minus_cycle": {f"@{arc}": pr(paired_masked(h_fr, h_cyc, X, m)) for m, arc in enumerate(ARCS)},
                            "LAYER_fresh_minus_navonly": {f"@{arc}": pr(paired_masked(h_fr, h_base, X, m)) for m, arc in enumerate(ARCS)},
                            "levels": {nm: {f"@{arc}": float(np.nanmean(h[X["rc_ok"][:, m], m])) for m, arc in enumerate(ARCS)}
                                       for nm, h in (("cycle_banked", h_cyc), ("fresh", h_fr), ("navonly_baseline", h_base))}}
        # ---- the four families under fresh ages (descriptive) vs the banked cycle json
        jf = json.load(open(os.path.join(EVD, f"eval_M2b_s{s}-sfresh-fresh.json")))
        jc = json.load(open(os.path.join(BANK, f"eval_M2b_s{s}.json")))
        assert jf["ages"] == "fresh" and jc["ages"] == "cycle"
        S["families_fresh"] = fam_digest(jf["raw"])
        S["families_cycle_banked"] = fam_digest(jc["raw"])
        # ---- the re-run control, if present
        if os.path.exists(ctl_p):
            ct = np.load(ctl_p)
            jct = json.load(open(os.path.join(EVD, f"eval_M2b_s{s}-sfresh-cycle.json")))
            S["control_cycle_rerun_cpu"] = {
                "max_abs_diff_ck_pos_err_vs_banked_m": float(np.nanmax(np.abs(ct["ck_pos_err"] - cyc["ck_pos_err"]))),
                "max_abs_diff_pred_vs_banked_m": float(np.abs(ct["pred"] - cyc["pred"]).max()),
                "max_abs_diff_ckpt_vs_banked": float(np.abs(ct["ckpt"] - cyc["ckpt"]).max()),
                "pos_err_mean_rerun": {f"@{arc}": float(np.nanmean(ct["ck_pos_err"][X["rc_ok"][:, m], m])) for m, arc in enumerate(ARCS)},
                "pos_err_mean_banked": {f"@{arc}": float(np.nanmean(cyc["ck_pos_err"][X["rc_ok"][:, m], m])) for m, arc in enumerate(ARCS)},
                "ade6_rerun": jct["raw"]["traj"]["ade6_m"]["mean"], "ade6_banked": jc["raw"]["traj"]["ade6_m"]["mean"],
                "ade2_rerun": jct["raw"]["traj"]["ade2_m"]["mean"], "ade2_banked": jc["raw"]["traj"]["ade2_m"]["mean"],
                "paired_rerun_minus_banked_pos_err@50": pr(paired_masked(ct["ck_pos_err"], cyc["ck_pos_err"], X, 1))}
        # fresh ages really are ages 0: the saved windows' tick poses coincide
        res["seeds"][f"s{s}"] = S
    # ---- the outcome table, VERBATIM
    q = {s: (res["seeds"][s]["Q_STALE_fresh_minus_cycle_pos_err"]["@50"], res["seeds"][s]["Q_LAYER_fresh_S_minus_navonly_pos_err"]["@50"]) for s in ("s0", "s1")}
    qs_neg_sep = {s: bool(q[s][0]["delta"] < 0 and q[s][0]["hi"] < 0) for s in q}
    ql_le0 = {s: bool(q[s][1]["delta"] <= 0) for s in q}
    ql_gt0_sep = {s: bool(q[s][1]["delta"] > 0 and q[s][1]["lo"] > 0) for s in q}
    ql_lt0_sep = {s: bool(q[s][1]["delta"] < 0 and q[s][1]["hi"] < 0) for s in q}
    artefact = all(ql_le0.values()) and all(qs_neg_sep.values())
    worse = all(ql_gt0_sep.values())
    if artefact:
        outcome = "STALENESS ARTEFACT"
        clause1 = "PASSES on fresh ages" if all(ql_lt0_sep.values()) else "does NOT pass on fresh ages (Q-LAYER < 0 separated is required at both seeds)"
    elif worse:
        outcome, clause1 = "S GENUINELY WORSE", None
    else:
        outcome, clause1 = "NOT ESTABLISHED", None
    res["outcome_table"] = {"Q_STALE_50m": {s: q[s][0] for s in q}, "Q_LAYER_50m": {s: q[s][1] for s in q},
                            "Q_STALE_lt0_separated": qs_neg_sep, "Q_LAYER_le0_point": ql_le0, "Q_LAYER_gt0_separated": ql_gt0_sep,
                            "Q_LAYER_lt0_separated": ql_lt0_sep, "STALENESS_ARTEFACT_condition": artefact, "S_GENUINELY_WORSE_condition": worse,
                            "OUTCOME": outcome, "clause1_reread_on_fresh_ages": clause1}
    nxt = {"STALENESS ARTEFACT": "STRAT-1's failure was the metric. Clause 1 is re-read on fresh ages: it PASSES iff Q-LAYER < 0 separated at both seeds. Then the deployed-schedule cost (Q-STALE) becomes the strategic lever: a faster S tick or a re-anchored checkpoint",
           "S GENUINELY WORSE": "the S layer as built adds nothing over nav-only kinematics at 50 m. The PREREG clause 'if the scene adds nothing, the nav-only layer ships' goes to the PI as a decision, with a pre-registered arm 'S as a residual on the nav-only baseline' (an architecture change: needs the PI's go)",
           "NOT ESTABLISHED": "report it as such. Seeds come before windows (H-ESTIM-SEED-1)"}
    res["outcome_table"]["committed_next_step_quoted_from_PREREG"] = nxt[outcome]
    jd(res, os.path.join(OUT, "sfresh_numbers.json"))
    print(json.dumps(res["outcome_table"], indent=1))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("controls", "final"))
    a = ap.parse_args()
    sys.exit({"controls": cmd_controls, "final": cmd_final}[a.mode](a))
