#!/usr/bin/env python3
"""v8F-mini evaluation on the refav1 trunk-probe grid (141 episodes, 2,399 windows; 2 s and 6 s; T1-equivalent open
loop, tier T0/T1 per `EVAL_DOCTRINE`: the model is conditioned on its OWN held upper-layer outputs, never on GT).

Modes
  join     build the eval label join (v9 eval139 release + lead + speed sidecar) for the 141 grid clips
  floors   NO image feature needed: refit-P10 (correct unit), kd_x, the GT-plan CONTROL, the nav-only checkpoint
           baseline and CONSISTENCY of every floor -- the numbers the arms are held against
  run      evaluate one checkpoint on the eval cache -> eval_<tag>.npz / .json (per-window predictions + metrics)
  compare  PREREG bars from several `run` outputs (paired episode-cluster bootstrap, B = 2000; seeds floor F)

Every row names the variance it answers: the bootstrap resamples EPISODES with the models fixed. The training-seed
variance is answered ONLY by F = |arm_s0 - arm_s1| (two seeds), never by the interval.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in (os.path.join(REPO, "stack"), os.path.join(REPO, "taniteval")):
    if p not in sys.path:
        sys.path.insert(0, p)

from tanitad.data import v8f_mini_data as D  # noqa: E402
from taniteval import floors_kin as fk  # noqa: E402
from taniteval import v8f_consistency as C  # noqa: E402
from taniteval.ci import episode_cluster_bootstrap, paired_episode_cluster_bootstrap  # noqa: E402

PKG = os.path.join(REPO, "TanitAD Research Lab", "Architecture & Inference", "Research", "2026-10-17-v8f-mini")
RAW = os.path.join(PKG, "raw")
B_BOOT = 2000


# ------------------------------------------------------------------------------------------------ inputs
def build_eval_join(grid, release_path, lead_path, speed_path, kin_npz, expect_md5=None):
    from tanitad.data.v9_labels import load_v9_release
    rel = load_v9_release(release_path, expect_md5=expect_md5)
    z = np.load(kin_npz)
    clips = [(c, (int(n) + 1) // 2) for c, n in zip(z["clip"].tolist(), z["n"].tolist())]
    assert [c for c, _ in clips] == list(grid["order"])
    J = D.build_label_join(clips, rel, lead_npz=lead_path, speed=D.load_speed_sidecar(speed_path))
    return J, rel


def ctx_from_args(a):
    grid = json.load(open(a.eval_grid))
    if os.path.exists(a.eval_join):
        J = D.load_join(a.eval_join)
    else:
        J, _ = build_eval_join(grid, a.v9_eval, a.lead_eval, a.speed_eval, a.eval_kin, a.v9_eval_md5)
        D.save_join(J, a.eval_join)
    return grid, J, D.eval_context(grid, a.eval_kin, J)


# ------------------------------------------------------------------------------------------------ metrics
def traj_metrics(pred, X, n_boot=B_BOOT, seed=0):
    gt = X["gt"]
    d = np.linalg.norm(pred - gt, axis=-1)
    ade6, fde6, ade2 = d.mean(1), d[:, -1], d[:, :10].mean(1)
    ci = lambda v: episode_cluster_bootstrap(v, X["eid"], n_boot=n_boot, seed=seed)
    return {"n": int(len(d)), "ade6_m": ci(ade6), "fde6_m": ci(fde6), "ade2_m": ci(ade2),
            "acts": fk.acts_check(pred, D.DT), "_pw": {"ade6": ade6, "fde6": fde6, "ade2": ade2}}


def families(pred, X, n_boot=B_BOOT, seed=0):
    """The binding LONGITUDINAL / LATERAL / TACTICAL families via `taniteval.four_families` (no re-implementation).
    A family that raises is recorded with its error -- never dropped. Lead-track (distance keeping) is reported by the
    CONSISTENCY LEAD rule here; the family's own lead-track input is not built (UNAVAILABLE, a work item)."""
    import torch
    from taniteval import four_families as ff
    P, G = torch.as_tensor(pred, dtype=torch.float32), torch.as_tensor(X["gt"], dtype=torch.float32)
    out = {}
    for name, fn in (("longitudinal", lambda: ff.longitudinal(P, G, C.DT, eid=X["eid"], n_boot=n_boot, seed=seed)),
                     ("lateral", lambda: ff.lateral(P, G, C.DT, eid=X["eid"], n_boot=n_boot, seed=seed)),
                     ("tactical_from_trajectory", lambda: ff.tactical_from_trajectory(P, G, C.DT, eid=X["eid"],
                                                                                        n_boot=n_boot, seed=seed))):
        try:
            out[name] = fn()
        except Exception as e:                                           # a work item, not a pass
            out[name] = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"}
    out["strategic"] = {"status": "see H-MINI-STRAT-1 (checkpoint error)", "paused": False}
    return out


def gt_constraints(X):
    L = D.v9_label_dict(X)
    return C.constraints_from_v9(L, X["limit_ms"], X["lead_gap_h"], X["s_h"])


def event_strata(X, V):
    """GT turn onset / stop / lead braking windows (reporting only, never decisive)."""
    v0 = X["kin"][:, 0]
    v6 = np.linalg.norm(X["gt"][:, -1] - X["gt"][:, -2], axis=-1) / C.DT
    return {"turn": V["TURN"]["active"], "stop": V["STOP"]["active"], "lead_brake": V["LEAD"]["active"] & ((v6 - v0) < -1.0)}


def checkpoint_metrics(ck_pred, X, n_boot=B_BOOT):
    """Strategic checkpoint error at 30/50/80 m (m, deg) with episode-cluster CIs, GT = v9 RC-A in the NOW frame."""
    e = C.checkpoint_errors(ck_pred, X["rc"], X["rc_ok"])
    out = {"n_valid": [int(X["rc_ok"][:, m].sum()) for m in range(3)], "_pw": e}
    for m, arc in enumerate(C.RULES["ckpt_arcs_m"]):
        ok = X["rc_ok"][:, m]
        out[f"pos_err_m@{int(arc)}"] = episode_cluster_bootstrap(e["pos_err_m"][ok, m], X["eid"][ok], n_boot=n_boot)
        out[f"head_err_deg@{int(arc)}"] = episode_cluster_bootstrap(e["head_err_deg"][ok, m], X["eid"][ok], n_boot=n_boot)
    return out


def tactical_metrics(goal_prob, lat_logits, lon_logits, X):
    """Macro-F1 over labelled cells (IGNORE masked) for lat / lon classes and the goal tokens; majority baselines."""
    has = X["has"]
    lat_t = np.where(has & (X["lat_cls"] >= 0), X["lat_cls"], D.IGNORE)
    lon_t = np.where(has & (X["lon_cls"] >= 0), X["lon_cls"], D.IGNORE)
    out = {"lat": C.macro_f1(lat_t, lat_logits.argmax(-1), 5), "lon": C.macro_f1(lon_t, lon_logits.argmax(-1), 7),
           "lat_majority": C.majority_baseline(lat_t, 5), "lon_majority": C.majority_baseline(lon_t, 7)}
    gy = ((X["goal_y"].astype(np.int64)[:, None] >> np.arange(22)) & 1).astype(bool)
    gw = ((X["goal_w"].astype(np.int64)[:, None] >> np.arange(22)) & 1).astype(bool) & has[:, None]
    f1 = []
    for g in range(22):
        m = gw[:, g]
        if m.sum() == 0 or gy[m, g].sum() == 0:
            f1.append(np.nan)
            continue
        p = goal_prob[m, g] > 0.5
        tp, fp, fn = int((p & gy[m, g]).sum()), int((p & ~gy[m, g]).sum()), int((~p & gy[m, g]).sum())
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    out["goal_f1_per_token"] = {D.GOAL22[g]: (None if np.isnan(f1[g]) else float(f1[g])) for g in range(22)}
    out["goal_macro_f1"] = float(np.nanmean(f1)) if np.isfinite(f1).any() else float("nan")
    out["goal_majority_macro_f1"] = 0.0          # all-negative: F1 0 on every positive-support token
    return out


def evaluate(pred, X, extras=None, n_boot=B_BOOT, seed=0, with_families=True):
    """One arm's full result dict. `extras`: optional model outputs {ckpt, goal_prob, lat_logits, lon_logits, cons}."""
    res = {"tier": "T0-open-loop-own-held-outputs (trunk-probe grid)", "estimator": "episode-cluster bootstrap, B=%d" % n_boot}
    tm = traj_metrics(pred, X, n_boot, seed)
    res["traj"] = {k: v for k, v in tm.items() if k != "_pw"}
    gtc = gt_constraints(X)
    cons = C.consistency(pred, gtc, eid=X["eid"], n_boot=n_boot, seed=seed)
    res["consistency_a"] = {k: v for k, v in cons.items() if k != "_per_window"}
    V = C.violations(pred, gtc)
    res["strata_ade6"] = {k: {"n": int(m.sum()), "ade6_m": float(tm["_pw"]["ade6"][m].mean()) if m.any() else None}
                          for k, m in event_strata(X, V).items()}
    pw = {"ade6": tm["_pw"]["ade6"], "fde6": tm["_pw"]["fde6"], "ade2": tm["_pw"]["ade2"],
          "viol_any": cons["_per_window"]["any_violated"], **{f"viol_{r}": cons["_per_window"][r] for r in C.RULE_NAMES}}
    if extras:
        if "ckpt" in extras:
            cm = checkpoint_metrics(extras["ckpt"], X, n_boot)
            pw["ck_pos_err"] = cm["_pw"]["pos_err_m"]
            res["strategic"] = {k: v for k, v in cm.items() if k != "_pw"}
        if "goal_prob" in extras:
            res["tactical"] = tactical_metrics(extras["goal_prob"], extras["lat_logits"], extras["lon_logits"], X)
            pc = C.constraints_from_pred({"goal_prob": extras["goal_prob"], "lon_logits": extras["lon_logits"],
                                          "cons": extras["cons"]}, X["limit_ms"], X["lead_gap_h"], X["s_h"])
            sc = C.consistency(pred, pc, eid=X["eid"], n_boot=n_boot, seed=seed)
            res["self_consistency_b"] = {"violation": {k: v for k, v in sc.items() if k != "_per_window"},
                                         "self_consistency": 1.0 - sc["any"]["rate_all"]}
            pw["self_viol_any"] = sc["_per_window"]["any_violated"]
    if with_families:
        res["families"] = families(pred, X, n_boot, seed)
    return res, pw


def jdump(o, path):
    def enc(x):
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(type(x))
    json.dump(o, open(path, "w"), indent=1, default=enc)


# ------------------------------------------------------------------------------------------------ floors
def cmd_floors(a):
    t0 = time.time()
    grid, J, X = ctx_from_args(a)
    outdir = a.out_dir
    os.makedirs(outdir, exist_ok=True)
    p10 = np.load(a.p10_pred)
    assert p10.shape == X["gt"].shape, "P10 predictions are not aligned with the grid"
    floors = {"P10_refit": p10, "kd_x_steer": X["base"]}
    ha0 = fk.ha0_paths(X["kin"][:, 0], D.DT, D.HORIZON)
    hx = fk.ha0_ext_paths(X["kin"], D.DT, D.HORIZON, action_units="steer")
    floors["damp50_steer"] = fk.damp50(ha0, hx)
    floors["ha0_const_velocity"] = ha0
    # the GT-plan control: the human's own future must violate (almost) nothing
    floors["GT_control"] = X["gt"]
    R, PW = {}, {}
    for name, pred in floors.items():
        R[name], PW[name] = evaluate(np.asarray(pred, np.float64), X, n_boot=a.n_boot, with_families=a.families)
    # nav-only kinematic checkpoint baseline vs the GT RC-A
    nb = C.nav_only_checkpoints(p10, np.where(X["has"], X["nav_tok"], 0), np.where(X["has"], X["nav_d"], np.nan))
    cm = checkpoint_metrics(nb, X, a.n_boot)
    nav_stats = {"n_turn_token_windows": int(((X["nav_tok"] > 0) & X["has"]).sum()),
                 "n_turn_token_with_distance": int(((X["nav_tok"] > 0) & X["has"] & np.isfinite(X["nav_d"])).sum())}
    # label coverage
    cov = D.join_report(J)
    cov["windows_total"] = int(len(X["eid"]))
    cov["windows_with_v9_row"] = int(X["has"].sum())
    cov["windows_without_v9_row"] = int((~X["has"]).sum())
    out = {"grid": {"n_windows": int(len(X["eid"])), "n_episodes": int(len(grid["order"]))}, "coverage": cov,
           "floors": R, "nav_only_baseline": {k: v for k, v in cm.items() if k != "_pw"}, "nav_stats": nav_stats,
           "rules": C.RULES, "seconds": time.time() - t0}
    np.savez_compressed(os.path.join(outdir, "floors_perwindow.npz"),
                        **{f"{n}__{k}": v for n, d in PW.items() for k, v in d.items()},
                        nav_baseline_ckpt=nb, eid=X["eid"])
    jdump(out, os.path.join(outdir, "floors_report.json"))
    for n, r in R.items():
        print(f"{n:22s} ADE6 {r['traj']['ade6_m']['mean']:.4f}  ADE2 {r['traj']['ade2_m']['mean']:.4f}  "
              f"CONS(a) any {r['consistency_a']['any']['rate_all']:.4f} "
              f"[SPEED {r['consistency_a']['SPEED']['rate_all']:.3f} STOP {r['consistency_a']['STOP']['rate_all']:.3f} "
              f"TURN {r['consistency_a']['TURN']['rate_all']:.3f} LEAD {r['consistency_a']['LEAD']['rate_all']:.3f}]")
    print("nav-only baseline pos err m:", {k: round(v["mean"], 3) for k, v in out["nav_only_baseline"].items()
                                           if k.startswith("pos_err")})
    return 0


# ------------------------------------------------------------------------------------------------ run
def cmd_run(a):
    import torch
    from tanitad.models import v8f_mini as M
    grid, J, X = ctx_from_args(a)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    # the checkpoint is self-contained: the frozen prior's weights and the constraint statistics are in its state dict
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    dev = torch.device(a.device if (a.device != "auto") else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev).eval()
    tot_hmax = cfg.hmax
    ej = D.load_join(a.eval_join)
    store = D.ClipStore(a.eval_cache, "eval", lru=a.lru, grid=ck.get("grid", "pool4x10"))
    wins = D.eval_grid_windows(a.eval_grid, ej)
    ds = D.V8FMiniWindows(store, ej, wins, hist=cfg.hist, hmax=tot_hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s)
    N = len(wins)
    perm = None
    if a.nav_shuffle:
        from tanitad.models.hjepa import donor_indices
        perm = donor_indices(N, episode_ids=torch.as_tensor(X["eid"]), seed=0).numpy()
    outs = {"traj30": [], "ckpt": [], "goal_prob": [], "lat_logits": [], "lon_logits": [], "cons_pred": [],
            "traj30_proj": []}
    with torch.no_grad():
        for b0 in range(0, N, a.batch):
            items = []
            for i in range(b0, min(b0 + a.batch, N)):
                ci, t, _, _ = wins[i]
                if a.ages == "cycle":
                    a_t = i % 3
                    a_s = a_t + (i // 3) % 5
                else:
                    a_t = a_s = 0
                it = ds.item(ci, t, a_t, a_s)
                if perm is not None:
                    dci, dt_, _, _ = wins[int(perm[i])]
                    it["nav"] = ds.item(dci, dt_, a_t, a_s)["nav"]
                items.append(it)
            b = D.collate(items)
            b = {k: ({kk: vv.to(dev) for kk, vv in v.items()} if isinstance(v, dict) else v.to(dev)) for k, v in b.items()}
            o = model.forward_infer(b)
            outs["traj30"].append(o["traj30"].float().cpu().numpy())
            outs["ckpt"].append(o["ckpt"].float().cpu().numpy())
            outs["goal_prob"].append(o["goal_prob"].float().cpu().numpy())
            outs["lat_logits"].append(o["lat_logits"].float().cpu().numpy())
            outs["lon_logits"].append(o["lon_logits"].float().cpu().numpy())
            outs["cons_pred"].append(o["cons_pred"].float().cpu().numpy())
            if a.project:
                lim = M.limit_ms_from_onehot(b["limit"], b["limit_ok"])
                cap = torch.where(torch.isfinite(lim), lim, torch.full_like(lim, 1e9))
                outs["traj30_proj"].append(M.project_speed(o["traj60"].float(), cap)[:, 1::2].cpu().numpy())
    O = {k: np.concatenate(v) for k, v in outs.items() if v}
    cons_dec = {n: O["cons_pred"][:, D.CONS_IDX[n]] for n in D.CONS_FIELDS}
    # the ckpt of a window is expressed in S's TICK frame; re-express in NOW with the measured odometry for scoring
    # (the model already does this for its own downstream use -- here it is the evaluator, against the NOW-frame GT)
    ck_now = _ckpt_to_now(O["ckpt"], ds, wins, a.ages)
    extras = {"ckpt": ck_now, "goal_prob": O["goal_prob"], "lat_logits": O["lat_logits"],
              "lon_logits": O["lon_logits"], "cons": cons_dec}
    tag = a.tag
    res = {"tag": tag, "ckpt": a.ckpt, "ages": a.ages, "nav_shuffle": bool(a.nav_shuffle)}
    r, pw = evaluate(O["traj30"].astype(np.float64), X, extras, a.n_boot, with_families=a.families)
    res["raw"] = r
    if a.project:
        r2, pw2 = evaluate(O["traj30_proj"].astype(np.float64), X, None, a.n_boot, with_families=False)
        res["projected_r1_kamm"] = r2
        os.makedirs(a.out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(a.out_dir, f"eval_{tag}_proj.npz"), pred=O["traj30_proj"], **pw2)
    os.makedirs(a.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(a.out_dir, f"eval_{tag}.npz"), pred=O["traj30"], **O, **pw, eid=X["eid"])
    jdump(res, os.path.join(a.out_dir, f"eval_{tag}.json"))
    print(tag, "ADE6", r["traj"]["ade6_m"]["mean"], "ADE2", r["traj"]["ade2_m"]["mean"],
          "CONS(a)", r["consistency_a"]["any"]["rate_all"])
    return 0


def _ckpt_to_now(ck, ds, wins, ages):
    """Checkpoints are predicted in S's tick frame; with ages 'fresh' that IS the NOW frame. With 'cycle' ages we
    re-express them in NOW using the measured poses (evaluator-side; the model's own use is internal)."""
    if ages != "cycle":
        return ck
    import torch
    from tanitad.models.v8f_mini import transform_ckpt
    out = np.empty_like(ck)
    for i, (ci, t, _, _) in enumerate(wins):
        a_t = i % 3
        a_s = a_t + (i // 3) % 5
        it = ds.item(ci, t, a_t, a_s)
        out[i] = transform_ckpt(torch.as_tensor(ck[i:i + 1]), torch.as_tensor(it["pose_s"])[None],
                                torch.as_tensor(it["pose_now"])[None])[0].numpy()
    return out


# ------------------------------------------------------------------------------------------------ compare
def paired(a_pw, b_pw, eid, n_boot=B_BOOT):
    """CI on mean(a) - mean(b) with the SAME resampled episodes each draw (`taniteval.ci`)."""
    r = paired_episode_cluster_bootstrap(a_pw, b_pw, eid, n_boot=n_boot, seed=0)
    return {"mean": r["delta"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"], "n_windows": r["n_windows"],
            "estimator": r["estimator"]}


def _f1_encoded(true, pred, n):
    return np.where(true >= 0, true * (n + 1) + pred, -1)


def _f1_reducer(n):
    def red(v):
        v = np.asarray(v)
        v = v[v >= 0]
        return C.macro_f1(v // (n + 1), v % (n + 1), n)["macro_f1"] if len(v) else float("nan")
    return red


def cmd_compare(a):
    """PREREG bars from several `run` outputs. `--runs tag=eval_<tag>.npz ...`; seeds are named `<arm>_s<k>` (two per arm);
    the nav-shuffle control is its own arm name `M1-navshuf_s<k>`. Baselines come from `floors`.
    H-MINI-K-1   ADE6 arm - P10 (refit), paired, separated at BOTH seeds and |delta| > 2F (F = seed floor);
    H-MINI-CONS-1 CONSISTENCY(a) rate vs P10's, paired; self-consistency (b) >= 0.90;
    H-MINI-STRAT-1 checkpoint error vs the nav-only baseline (50, 80 m), navshuf worse than M1, M1 better than navonly;
    H-MINI-TAC-1 lat / lon macro-F1 vs the majority class and M1-navonly (paired episode-cluster bootstrap)."""
    grid, J, X = ctx_from_args(a)
    R = {}
    for spec in a.runs:
        tag, path = spec.split("=", 1)
        R[tag] = np.load(path)
    fl = np.load(a.floors_npz)
    eid, p10 = X["eid"], fl["P10_refit__ade6"]
    arm_of = lambda t: t.rsplit("_s", 1)[0]
    arms = sorted({arm_of(t) for t in R})
    seeds = {arm: [R[t] for t in sorted(R) if arm_of(t) == arm] for arm in arms}
    out = {"estimator": "paired episode-cluster bootstrap, B=%d (answers: another draw of EPISODES); training-seed variance "
                        "is answered ONLY by F = |arm_s0 - arm_s1|" % a.n_boot, "arms": {}}
    base_err = C.checkpoint_errors(fl["nav_baseline_ckpt"], X["rc"], X["rc_ok"])["pos_err_m"]
    for arm in arms:
        sd = seeds[arm]
        d = {"n_seeds": len(sd)}
        ci = [paired(s["ade6"], p10, eid, a.n_boot) for s in sd]
        d["K1_ade6_minus_P10"] = [{"mean": c["mean"], "lo": c["lo"], "hi": c["hi"], "separated": c["separated"]} for c in ci]
        if len(sd) >= 2:
            F = float(abs(sd[0]["ade6"].mean() - sd[1]["ade6"].mean()))
            d["seed_floor_F_m"] = F
            d["K1_pass"] = bool(all(c["separated"] and c["mean"] < 0 and abs(c["mean"]) > 2 * F for c in ci))
        d["consistency_a_rate"] = [float(s["viol_any"].mean()) for s in sd]
        d["consistency_a_vs_P10"] = [paired(s["viol_any"].astype(float), fl["P10_refit__viol_any"].astype(float), eid, a.n_boot)
                                     for s in sd]
        if "self_viol_any" in sd[0].files:
            d["self_consistency_b"] = [float(1 - s["self_viol_any"].mean()) for s in sd]
        if "ck_pos_err" in sd[0].files:
            d["strat"] = {}
            for m, arc in ((1, 50), (2, 80)):
                ok = X["rc_ok"][:, m]
                d["strat"][f"vs_nav_only_baseline@{arc}"] = [paired(s["ck_pos_err"][ok, m], base_err[ok, m], eid[ok], a.n_boot)
                                                          for s in sd]
        out["arms"][arm] = d
    if "M1" in seeds and "M1-navshuf" in seeds:
        out["nav_used_M1navshuf_minus_M1_pos_err"] = {
            f"@{arc}": paired(seeds["M1-navshuf"][0]["ck_pos_err"][X["rc_ok"][:, m], m], seeds["M1"][0]["ck_pos_err"][X["rc_ok"][:, m], m],
                              eid[X["rc_ok"][:, m]], a.n_boot) for m, arc in ((1, 50), (2, 80))}
    if "M1" in seeds and "M1-navonly" in seeds:
        out["vision_helps_M1_minus_navonly_pos_err"] = {
            f"@{arc}": paired(seeds["M1"][0]["ck_pos_err"][X["rc_ok"][:, m], m], seeds["M1-navonly"][0]["ck_pos_err"][X["rc_ok"][:, m], m],
                              eid[X["rc_ok"][:, m]], a.n_boot) for m, arc in ((1, 50), (2, 80))}
    # H-MINI-TAC-1: lat / lon macro-F1, paired (arm - M1-navonly) and vs the majority class (point estimates)
    for kind, n, tcol in (("lat", 5, "lat_cls"), ("lon", 7, "lon_cls")):
        true = np.where(X["has"] & (X[tcol] >= 0), X[tcol], -1)
        red = _f1_reducer(n)
        tac = {}
        for arm in arms:
            sd = seeds[arm]
            enc = [_f1_encoded(true, s[f"{kind}_logits"].argmax(-1), n) for s in sd]
            tac[arm] = {"macro_f1": [red(e) for e in enc]}
            if "M1-navonly" in seeds and arm != "M1-navonly":
                ref = _f1_encoded(true, seeds["M1-navonly"][0][f"{kind}_logits"].argmax(-1), n)
                r = paired_episode_cluster_bootstrap(enc[0], ref, eid, n_boot=a.n_boot, seed=0, reduce=red)
                tac[arm]["vs_navonly"] = {"mean": r["delta"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"]}
        maj = C.majority_baseline(true, n)
        tac["_majority"] = {"macro_f1": maj["macro_f1"], "class": maj["majority_class"]}
        out[f"tactical_{kind}"] = tac
    out["P10_consistency_a_rate"] = float(fl["P10_refit__viol_any"].mean())
    jdump(out, a.out)
    print(json.dumps({k: (v if k != "arms" else {n: {kk: vv for kk, vv in d.items() if kk in ("seed_floor_F_m", "K1_pass", "consistency_a_rate", "self_consistency_b")}
                                                   for n, d in v.items()}) for k, v in out.items()
                      if k in ("arms", "P10_consistency_a_rate")}, indent=1, default=float)[:3000])
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    def common(p):
        p.add_argument("--eval-grid", default=os.path.join(RAW, "eval_grid_windows.json"))
        p.add_argument("--eval-kin", default=os.path.join(RAW, "eval141_kin.npz"))
        p.add_argument("--eval-join", default=os.path.join(RAW, "join_eval141.npz"))
        p.add_argument("--v9-eval", default="C:/Users/Admin/qland/work/refcv8_hf_v9/release/v9_labels_eval139.npz")
        p.add_argument("--v9-eval-md5", default="6b5c7f207cffc3b7eb3cd527fd433599")
        p.add_argument("--lead-eval", default="D:/Projects/TanitAD-artifacts/v9labels/inputs/lead_eval139.npz")
        p.add_argument("--speed-eval", default="C:/Users/Admin/qland/landed_hold_datacov/raw/refcv6_speed_max_v8_eval.jsonl")
        p.add_argument("--n-boot", type=int, default=B_BOOT)
        p.add_argument("--families", action="store_true", help="also run taniteval.four_families (slower)")
        p.add_argument("--p10", default=os.path.join(RAW, "p10", "p10_prior.pt"))
        p.add_argument("--out-dir", default=os.path.join(RAW, "eval"))
    p = sub.add_parser("join"); common(p)
    p = sub.add_parser("floors"); common(p)
    p.add_argument("--p10-pred", default=os.path.join(RAW, "p10", "p10_eval_pred6.npy"))
    p = sub.add_parser("run"); common(p)
    p.add_argument("--ckpt", required=True); p.add_argument("--tag", required=True)
    p.add_argument("--eval-cache", default="C:/Users/Admin/refav1_hj_cache")
    p.add_argument("--ages", choices=("fresh", "cycle"), default="cycle")
    p.add_argument("--nav-shuffle", action="store_true"); p.add_argument("--project", action="store_true")
    p.add_argument("--batch", type=int, default=64); p.add_argument("--lru", type=int, default=32)
    p.add_argument("--device", default="auto")
    p = sub.add_parser("compare"); common(p)
    p.add_argument("--runs", nargs="+", required=True); p.add_argument("--floors-npz", default=os.path.join(RAW, "eval", "floors_perwindow.npz"))
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.mode == "join":
        grid, J, X = ctx_from_args(a)
        print(json.dumps(D.join_report(J), indent=1)[:3000])
        return 0
    return {"floors": cmd_floors, "run": cmd_run, "compare": cmd_compare}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
