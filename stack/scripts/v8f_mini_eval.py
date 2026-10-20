#!/usr/bin/env python3
"""v8F-mini evaluation on the refav1 trunk-probe grid (141 episodes, 2,399 windows; 2 s and 6 s; T1-equivalent open
loop, tier T0/T1 per `EVAL_DOCTRINE`: the model is conditioned on its OWN held upper-layer outputs, never on GT).

Modes
  join     build the eval label join (v9 eval139 release + lead + speed sidecar) for the 141 grid clips
  floors   NO image feature needed: refit-P10 (correct unit), kd_x, the GT-plan CONTROL, the nav-only checkpoint
           baseline and CONSISTENCY of every floor -- the numbers the arms are held against
  run      evaluate one checkpoint on the eval cache -> eval_<tag>.npz / .json (per-window predictions + metrics);
           with `--plan beta` ALSO the rung-beta gate (PREREG_V8F_BETA.md): the checkpoint's own world model ranks the 117-candidate
           fan against a predicted goal latent -> eval_<tag>_beta.npz and the `plan_beta` block of eval_<tag>.json. ZERO TRAINING.
  compare  PREREG bars from several `run` outputs (paired episode-cluster bootstrap, B = 2000; seeds floor F)
  compare-beta  the rung-beta verdict from two seeds' `run --plan beta` JSONs (PREREG_V8F_BETA.md §3)

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
from taniteval import v8f_beta_eval as BE  # noqa: E402
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


def ctx_from_args(a, kin_mode=None):
    kin_mode = kin_mode or a.kin_mode
    grid = json.load(open(a.eval_grid))
    if os.path.exists(a.eval_join):
        J = D.load_join(a.eval_join)
    else:
        J, _ = build_eval_join(grid, a.v9_eval, a.lead_eval, a.speed_eval, a.eval_kin, a.v9_eval_md5)
        D.save_join(J, a.eval_join)
    return grid, J, D.eval_context(grid, a.eval_kin, J, kin_mode=kin_mode)


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
        if isinstance(x, (np.floating, np.integer, np.bool_)):
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
    tag = "steer" if a.kin_mode == "steer" else "yaw"       # the floor names say which representation they are
    floors = {"P10_refit": p10, f"kd_x_{tag}": X["base"]}
    ha0 = fk.ha0_paths(X["kin"][:, 0], D.DT, D.HORIZON)
    hx = fk.ha0_ext_paths(X["kin"], D.DT, D.HORIZON, action_units=D.KIN_UNITS[a.kin_mode])
    floors[f"damp50_{tag}"] = fk.damp50(ha0, hx)
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
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])                 # a checkpoint written before the ruling has no kin_mode => steer
    if a.plan == "beta":
        if cfg.wm == "off":
            raise SystemExit("REFUSED: --plan beta scores with the checkpoint's OWN world model; this one was trained with wm='off' (arm M1). Use M2a / M2b.")
        if cfg.kin_mode != "yaw":
            raise SystemExit("REFUSED: --plan beta runs only on a --kin-mode yaw checkpoint (PI ruling D-PI-YAWRATE-1: no steering angle at inference).")
    if a.tac != "pred" and (a.plan == "beta" or a.nav_shuffle):
        raise SystemExit("REFUSED: --tac shuffle|oracle is ONE intervention on T -> O; run it without --plan beta and without --nav-shuffle")
    grid, J, X = ctx_from_args(a, cfg.kin_mode)       # the scoring context is built in the CHECKPOINT's representation
    tracks = None
    if cfg.kin_mode == "yaw":
        if not a.tracks:
            raise SystemExit("REFUSED: this checkpoint was trained with --kin-mode yaw; run needs --tracks native_tracks.pt")
        tracks = D.load_native_tracks(a.tracks)
    # the checkpoint is self-contained: the frozen prior's weights and the constraint statistics are in its state dict
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    dev = torch.device(a.device if (a.device != "auto") else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev).eval()
    tot_hmax = cfg.hmax
    ej = D.load_join(a.eval_join)
    store = D.ClipStore(a.eval_cache, "eval", lru=a.lru, grid=ck.get("grid", "pool4x10"), kin_mode=cfg.kin_mode, tracks=tracks)
    wins = D.eval_grid_windows(a.eval_grid, ej)
    ds = D.V8FMiniWindows(store, ej, wins, hist=cfg.hist, hmax=tot_hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s)
    N = len(wins)
    perm = None
    if a.nav_shuffle:
        from tanitad.models.hjepa import donor_indices
        perm = donor_indices(N, episode_ids=torch.as_tensor(X["eid"]), seed=0).numpy()
    tperm = None
    if a.tac == "shuffle":
        from tanitad.models.hjepa import donor_indices
        tperm = donor_indices(N, episode_ids=torch.as_tensor(X["eid"]), seed=1).numpy()
    tac_rows = {"shift": [], "oracle_fallback": []}
    outs = {"traj30": [], "ckpt": [], "goal_prob": [], "lat_logits": [], "lon_logits": [], "cons_pred": [],
            "traj30_proj": []}
    beta_rows = []                                     # rung beta: the planner's per-window inputs (only with --plan beta)
    if a.plan == "beta":
        from tanitad.models import v8f_beta as BETA
    with torch.no_grad():
        for b0 in range(0, N, a.batch):
            items, ditems = [], []
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
                if tperm is not None:                  # the donor window, at the SAME ages: its T runs in its own episode
                    dci, dt_, _, _ = wins[int(tperm[i])]
                    ditems.append(ds.item(dci, dt_, a_t, a_s))
            todev = lambda bb: {k: ({kk: vv.to(dev) for kk, vv in v.items()} if isinstance(v, dict) else v.to(dev))   # noqa: E731
                                for k, v in bb.items()}
            b = todev(D.collate(items))
            o = model.forward_infer(b)
            if a.tac != "pred":                      # PREREG_V8F_TAC: O reads another code; S, T and the scene are untouched
                code0 = o["code"]
                if a.tac == "shuffle":
                    code = model.forward_infer(todev(D.collate(ditems)))["code"]
                    fb = torch.zeros(code.shape[0], dtype=torch.bool, device=dev)
                else:                                    # oracle: the GT code where the window has a v9 row (a T0 diagnostic)
                    has = b["lab_t"]["has"].bool()
                    code = torch.where(has[:, None], model._code_gt(b["lab_t"]).to(code0.dtype), code0)
                    fb = ~has
                o = model.forward_diag(b, code)
                tac_rows["shift"].append((o["code"] - code0).abs().mean(-1).float().cpu().numpy())
                tac_rows["oracle_fallback"].append(fb.cpu().numpy())
            if a.plan == "beta":
                beta_rows.append(BETA.collect_beta_inputs(model, b, o))
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
    res = {"tag": tag, "ckpt": a.ckpt, "ages": a.ages, "nav_shuffle": bool(a.nav_shuffle), "kin_mode": cfg.kin_mode, "plan": a.plan,
           "tac": a.tac}
    tac_npz = {}
    if a.tac != "pred":
        shift, fb = np.concatenate(tac_rows["shift"]), np.concatenate(tac_rows["oracle_fallback"])
        res["tac_manipulation"] = {"mode": a.tac, "code_shift_mean": float(shift.mean()), "frac_code_changed": float((shift > 1e-6).mean()),
                                   "n_oracle_fallback_no_v9_row": int(fb.sum()), "n_windows": int(shift.size),
                                   "donor_seed": 1 if a.tac == "shuffle" else None}
        tac_npz = {"tac_code_shift": shift, "tac_oracle_fallback": fb}
    r, pw = evaluate(O["traj30"].astype(np.float64), X, extras, a.n_boot, with_families=a.families)
    res["raw"] = r
    if a.plan == "beta":
        res["plan_beta"], beta_npz = beta_block(model, beta_rows, X, O["traj30"], pw, extras, a)
        os.makedirs(a.out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(a.out_dir, f"eval_{tag}_beta.npz"), **beta_npz)
    if a.project:
        r2, pw2 = evaluate(O["traj30_proj"].astype(np.float64), X, None, a.n_boot, with_families=False)
        res["projected_r1_kamm"] = r2
        os.makedirs(a.out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(a.out_dir, f"eval_{tag}_proj.npz"), pred=O["traj30_proj"], **pw2)
    os.makedirs(a.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(a.out_dir, f"eval_{tag}.npz"), pred=O["traj30"], **O, **pw, **tac_npz, eid=X["eid"])
    jdump(res, os.path.join(a.out_dir, f"eval_{tag}.json"))
    print(tag, "ADE6", r["traj"]["ade6_m"]["mean"], "ADE2", r["traj"]["ade2_m"]["mean"],
          "CONS(a)", r["consistency_a"]["any"]["rate_all"])
    if a.plan == "beta":
        pb = res["plan_beta"]
        print(tag, "BETA ADE6", pb["arms"]["beta"]["traj"]["ade6_m"]["mean"], "replaced", round(pb["wm_diag"]["replaced_frac"], 4),
              "beta-M2 ADE6", pb["paired_vs_m2"]["ade6"].get("mean"), "scorer calls", pb["score_calls"])
    return 0


# ------------------------------------------------------------------------------------------------ rung beta (PREREG_V8F_BETA.md)
def _ade(pred, gt, k=None):
    d = np.linalg.norm(np.asarray(pred, np.float64) - np.asarray(gt, np.float64), axis=-1)
    return d.mean(1) if k is None else d[:, :k].mean(1)


def beta_block(model, rows, X, pred_m2, pw_m2, extras, a):
    """The rung-beta gate for one checkpoint. ``rows`` = the per-batch :func:`v8f_beta.collect_beta_inputs` dicts gathered in the `run`
    loop (so the planner saw exactly the windows, the held-output ages and the inference keys the M2 evaluation used); ``pred_m2`` /
    ``pw_m2`` = M2's own 6 s plan and per-window results. Returns ``(json block, npz arrays)``. Reads the ground truth ONLY for the
    scoring and for the T0 diagnostic arms, never for the result arm ``beta``."""
    import torch
    from tanitad.models import v8f_beta as BETA
    from tanitad.models import v8f_mini as M
    BETA.reset_counters()
    if a.fan_artifact:
        ctrl, fan = BETA.load_fan_controls(a.fan_artifact)
    else:
        ctrl, fan = None, {"source": "pinned grid (stack/tests/test_anchor_flyability.py:11-22); NOT checked against the shipped artifact",
                           "max_abs_diff_vs_pinned_grid": 0.0}
    inp = BETA.cat_inputs(rows)
    n, gt, eid, nb = len(inp), np.asarray(X["gt"], np.float64), X["eid"], a.n_boot
    if n != len(eid):
        raise AssertionError(f"beta saw {n} windows, the scoring context has {len(eid)}")
    if not np.array_equal(inp.traj60[:, 1::2].numpy(), np.asarray(pred_m2, np.float32)):
        raise AssertionError("IDENTITY BROKEN: the incumbent beta carries is not the plan M2 emitted")
    arms = BETA.run_beta(model.wm, inp, gt=torch.as_tensor(gt), eid=eid, controls=ctrl, chunk=a.beta_chunk, seed=0)
    R, PW, PRED = {}, {}, {"m2": np.asarray(pred_m2, np.float64)}
    for name, arm in arms.items():
        PRED[name] = arm["traj30"].double().numpy()
        R[name], PW[name] = evaluate(PRED[name], X, extras if name == "beta" else None, nb, with_families=a.families and name == "beta")
    PW["m2"] = pw_m2
    # per-window family components (m2 and every arm), on the same windows
    FW = {}
    for name, p in PRED.items():
        fw = BE.family_windows(p, gt, C.DT)
        ra = BE.route_adherence(p, X["rc"], X["rc_ok"])
        for m, arc in enumerate(BE.CKPT_ARCS_M):
            fw[f"strat_route_err_{int(arc)}m"] = ra[:, m]
        fw["consistency_viol_any"] = np.asarray(PW[name]["viol_any"], np.float64)
        FW[name] = fw
    fam = {c: BE.paired_row(FW["beta"][c], FW["m2"][c], eid, nb, lower_is_better=lib) for c, (_, lib) in BE.COMPONENTS.items()}
    d6 = BE.paired_row(PW["beta"]["ade6"], pw_m2["ade6"], eid, nb)
    out = {
        "spec": {"goal": "S's route checkpoints (NOW frame) x the frozen P10 prior's pace -> the WM's level-2 latent at +6 s "
                         "(v8f_beta.route_goal_xy / goal_latents)", "level": BETA.WM_LEVEL, "last_k": 1, "context": 1, "lam_subgoal": 0.0,
                 "candidates": "117-fan + M2's own plan as the incumbent (strict improvement replaces it)", "tuned_parameters": 0,
                 "ages": a.ages, "nav_shuffle": bool(a.nav_shuffle)},
        "fan": fan, "score_calls": dict(BETA.SCORE_CALLS), "n_windows": n,
        "arms": R, "tier": "T0/T1 open loop, own held outputs; beta_oracle / fan_oracle read the GROUND TRUTH (T0 diagnostics, never a result)",
        "paired_vs_m2": {"ade6": d6, "ade2": BE.paired_row(PW["beta"]["ade2"], pw_m2["ade2"], eid, nb),
                         "fde6": BE.paired_row(PW["beta"]["fde6"], pw_m2["fde6"], eid, nb), "families": fam,
                         "estimator": "paired episode-cluster bootstrap, B=%d, episodes only" % nb},
    }
    shuf = BE.paired_row(PW["beta_shuf"]["ade6"], PW["beta"]["ade6"], eid, nb) if "beta_shuf" in PW else {"status": "UNAVAILABLE"}
    ade_orc_fan = _ade(arms["beta_oracle"]["traj30_fan"].double().numpy(), gt)
    rand, ceil = float(PW["fan_random"]["ade6"].mean()), float(PW["fan_oracle"]["ade6"].mean())
    sig = {"oracle_fanonly_ade6": float(ade_orc_fan.mean()), "random_ade6": rand, "ceiling_ade6": ceil,
           "recovery": BE.signal_recovery(rand, float(ade_orc_fan.mean()), ceil),
           "oracle_minus_random": BE.paired_row(ade_orc_fan, PW["fan_random"]["ade6"], eid, nb)}
    rep = arms["beta"]["replaced"].numpy()
    out["controls"] = {"ade6_shuf_minus_beta": shuf}
    out["diagnostics_T0"] = {"signal": sig,
                             "beta_fanonly_ade6": float(_ade(arms["beta"]["traj30_fan"].double().numpy(), gt).mean())}
    out["wm_diag"] = {"replaced_frac": float(rep.mean()), "fan_endpoint_spread_ratio": BETA.endpoint_spread_ratio(arms["beta"]),
                      "oracle_fan_endpoint_spread_ratio": BETA.endpoint_spread_ratio(arms["beta_oracle"])}
    out["admissibility"] = {"goal_computed_from": ["S checkpoints (ck_now)", "frozen P10 prior (p30)", "the WM's own z1 at NOW"],
                            "never_read_by_the_goal": ["goal_prob", "lat_logits", "lon_logits", "cons_pred", "any v9 label", "ground truth"],
                            "inference_keys_allowed": sorted(M.INFERENCE_INPUT_KEYS)}
    out["decision_inputs"] = {"ade6_beta_minus_m2": d6, "ade6_shuf_minus_beta": shuf, "families": fam,
                              "replaced_frac": out["wm_diag"]["replaced_frac"], "signal": sig}
    npz = {"eid": eid, "beta__replaced": rep, "beta__fan_idx": arms["beta"]["fan_idx"].numpy()}
    for name, p in PRED.items():
        npz[f"{name}__pred"] = p
        for k, v in (PW[name] if name != "m2" else {}).items():
            npz[f"{name}__{k}"] = v
        for k, v in FW[name].items():
            npz[f"{name}__fw_{k}"] = v
    return out, npz


def cmd_compare_beta(a):
    """The PREREG_V8F_BETA.md §3 verdict from the two training seeds' `run --plan beta` JSONs (`--runs s0=eval_X_s0.json s1=eval_X_s1.json`)."""
    recs, m2, tags = [], [], []
    for spec in a.runs:
        tag, path = spec.split("=", 1)
        with open(path, encoding="utf-8") as fh:
            j = json.load(fh)
        if j.get("plan") != "beta" or "plan_beta" not in j:
            raise SystemExit(f"REFUSED: {path} is not a `run --plan beta` output")
        recs.append(j["plan_beta"]["decision_inputs"])
        m2.append(float(j["raw"]["traj"]["ade6_m"]["mean"]))
        tags.append({"tag": tag, "ckpt": j.get("ckpt"), "n_windows": j["plan_beta"]["n_windows"]})
    if len({t["ckpt"] for t in tags}) < len(tags):
        raise SystemExit("REFUSED: two `--runs` name the same checkpoint -- the seed floor needs two TRAINING seeds")
    out = BE.decide_beta(recs, m2)
    out.update({"runs": tags, "m2_ade6_means": m2,
                "estimator": "paired episode-cluster bootstrap per seed (episodes only); training-seed variance enters ONLY through F = |M2_s0 - M2_s1|"})
    jdump(out, a.out)
    print(json.dumps({k: out[k] for k in ("verdict", "next", "seed_floor_F_m", "threshold_2F_m", "P1_ade6_beats_m2_separated_beyond_2F",
                                          "P2_shuffled_goal_control_loses", "P3_no_family_separated_worse", "families_separated_worse",
                                          "inert", "replaced_frac") if k in out}, indent=1, default=float))
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
        p.add_argument("--kin-mode", choices=D.KIN_MODES, default="yaw",
                       help="PI ruling D-PI-YAWRATE-1: yaw (default; the floors are built on the MEASURED yaw rate, curvature unit) | steer (legacy)")
        p.add_argument("--p10", default=None, help="default = the prior for --kin-mode (raw/p10_yaw | raw/p10)")
        p.add_argument("--out-dir", default=os.path.join(RAW, "eval"))
    p = sub.add_parser("join"); common(p)
    p = sub.add_parser("floors"); common(p)
    p.add_argument("--p10-pred", default=None, help="default = raw/p10_yaw/p10_eval_pred6.npy (yaw) | raw/p10/p10_eval_pred6.npy (steer)")
    p = sub.add_parser("run"); common(p)
    p.add_argument("--ckpt", required=True); p.add_argument("--tag", required=True)
    p.add_argument("--tracks", default=None, help="native_tracks.pt: REQUIRED for a checkpoint trained with --kin-mode yaw")
    p.add_argument("--eval-cache", default="C:/Users/Admin/refav1_hj_cache")
    p.add_argument("--ages", choices=("fresh", "cycle"), default="cycle")
    p.add_argument("--nav-shuffle", action="store_true"); p.add_argument("--project", action="store_true")
    p.add_argument("--tac", choices=("pred", "shuffle", "oracle"), default="pred",
                   help="PREREG_V8F_TAC: pred (DEFAULT) = O reads T's own held code. shuffle = O reads a DONOR window's predicted code (another "
                        "episode, seed 1): does O use T at all? oracle = O reads the GT v9 code where the window has a row (T0 diagnostic: what "
                        "a perfect tactical layer would buy). One intervention per run.")
    p.add_argument("--batch", type=int, default=64); p.add_argument("--lru", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--plan", choices=("none", "beta"), default="none",
                   help="none (DEFAULT) = M2's own output only. beta = ALSO run the rung-beta gate: the checkpoint's world model ranks the "
                        "117-candidate fan against a predicted goal latent (PREREG_V8F_BETA.md). A default that disables a mechanism must be "
                        "NAMED in the arm's argv, or the arm did not test it.")
    p.add_argument("--fan-artifact", default=None,
                   help="anchors_117_alat_declared.pt (declared 'alat' controls); default = the pinned 13 x 9 grid, which is then NOT checked against the shipped file")
    p.add_argument("--beta-chunk", type=int, default=16, help="windows per world-model scoring call (x 118 candidates)")
    p = sub.add_parser("compare"); common(p)
    p.add_argument("--runs", nargs="+", required=True); p.add_argument("--floors-npz", default=os.path.join(RAW, "eval", "floors_perwindow.npz"))
    p.add_argument("--out", required=True)
    p = sub.add_parser("compare-beta"); common(p)
    p.add_argument("--runs", nargs=2, required=True, metavar="TAG=eval_X.json", help="the two training seeds' `run --plan beta` JSONs")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sub_p10 = "p10_yaw" if a.kin_mode == "yaw" else "p10"
    if getattr(a, "p10", "x") is None:
        a.p10 = os.path.join(RAW, sub_p10, "p10_prior.pt")
    if getattr(a, "p10_pred", "x") is None:
        a.p10_pred = os.path.join(RAW, sub_p10, "p10_eval_pred6.npy")
    if a.mode == "join":
        grid, J, X = ctx_from_args(a)
        print(json.dumps(D.join_report(J), indent=1)[:3000])
        return 0
    return {"floors": cmd_floors, "run": cmd_run, "compare": cmd_compare, "compare-beta": cmd_compare_beta}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
