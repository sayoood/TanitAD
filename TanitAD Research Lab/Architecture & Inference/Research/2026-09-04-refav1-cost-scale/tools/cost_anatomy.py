"""D-REFAV1-COST-ANATOMY — what the refav1 planner cost ACTUALLY is, in units.

⭐ WHAT THIS DOES AND WHY IT IS CHEAP. `RefAV1.plan` builds its cost as a CLOSURE
(`_cost_chunk`) over the encoded window, the imagined goal and the search
predictor. This probe captures that EXACT closure — by wrapping
`refa_v1.icem_plan`, which `plan` calls with it — and then evaluates it on a
DESIGNED candidate box instead of paying iCEM's 2,100 rollouts. So one window
costs ~|box| model rollouts, not 2,100, and the full 282-window eval grid is
affordable.

⛔ NOTHING IS RE-DERIVED. The cost expression, the goal, the rollout, the metric
and the weights are the shipped ones; the probe only records the terms and the
raw (zt, g) fields the shipped `_goal_term` was handed.

RECORDS, per window and per candidate:
  cost32       the shipped total (float32, as the planner sees it)
  goal32       the shipped goal term  1 - F.cosine_similarity(zt, g)   [float32]
  goal64       the SAME quantity recomputed in float64 from the same fields
  chord64      ||x̂ - ŷ||  (the monotone-equivalent chord), float64
  jerk_raw     mean(((a_{t+1}-a_t)/dt)^2)      — unweighted
  kap_raw      mean(kappa^2)                   — unweighted
plus the field geometry that explains the magnitudes.

⇒ Any (W_JERK, W_KAPPA, metric) setting can then be SCORED OFFLINE on the same
box with zero further GPU, which is the screen that says whether a setting can
possibly change the plan before an iCEM sweep is paid for.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch


def _load_arm(repo: str):
    sys.path.insert(0, os.path.join(repo, "stack"))
    sys.path.insert(0, os.path.join(repo, "taniteval"))
    import importlib.util
    p = os.path.join(repo, "taniteval", "tools", "refav1_arm.py")
    spec = importlib.util.spec_from_file_location("refav1_arm", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["refav1_arm"] = m
    spec.loader.exec_module(m)
    return m


class _Captured(Exception):
    pass


def build_box(H: int, dev, seed_pool, proposal, kappa_max: float, a_max: float):
    """The candidate box. Rows are NAMED so an argmin is interpretable."""
    names, rows = [], []

    def add(nm, a_val, k_val):
        c = torch.zeros(H, 2, device=dev)
        c[:, 0] = a_val
        c[:, 1] = k_val
        names.append(nm)
        rows.append(c)

    add("cv", 0.0, 0.0)                       # == hold_v0, the incumbent
    add("decel_1.5", -1.5, 0.0)               # the injected brake baseline
    for k in (0.2, 0.1, 0.05, 0.02, 0.01):
        add(f"kap+{k}", 0.0, +k)
        add(f"kap-{k}", 0.0, -k)
    for aa in (4.0, 2.0, 1.5, 0.5):
        add(f"acc+{aa}", +aa, 0.0)
        add(f"acc-{aa}", -aa, 0.0)
    add("turnL_slow", -1.0, +0.1)             # a realistic cornering plan
    add("turnR_slow", -1.0, -0.1)
    # a ramped curvature (what a real turn-in looks like)
    for sgn, nm in ((+1.0, "rampL"), (-1.0, "rampR")):
        c = torch.zeros(H, 2, device=dev)
        c[:, 1] = sgn * torch.linspace(0.0, 0.15, H, device=dev)
        names.append(nm)
        rows.append(c)
    if seed_pool is not None and seed_pool.numel():
        for i in range(seed_pool.shape[0]):
            names.append(f"seed{i}")           # the imagined goal's own controls
            rows.append(seed_pool[i].to(dev))
    if proposal is not None:
        names.append("proposal")
        rows.append(proposal.to(dev))
    return names, torch.stack(rows, 0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--nav", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--window-stride", type=int, default=40)
    ap.add_argument("--episodes-n", type=int, default=0)
    ap.add_argument("--lru", type=int, default=8)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--out", required=True)
    a0 = ap.parse_args(argv)

    arm = _load_arm(a0.repo)
    from tanitad.refs import refa_v1 as R
    import torch.nn.functional as F

    dev = a0.device
    model, cfg, prov = arm.load_model(a0.ckpt, a0.config, dev, False)
    print(f"[model] step={prov['step']} params={prov['trainable_parameters']:,} "
          f"target_space={cfg.target_space}", flush=True)

    class _A:
        pass
    a = _A()
    a.cache, a.episodes, a.labels, a.nav = a0.cache, a0.episodes, a0.labels, a0.nav
    a.lru, a.plan_seed = a0.lru, 0
    a.plan_n_samples = a.plan_n_iters = a.plan_n_elites = None
    names_ep = arm.episode_names(a0.cache)
    if a0.episodes_n:
        names_ep = names_ep[: a0.episodes_n]
    k = arm.K_TRAJ_DEFAULT if hasattr(arm, "K_TRAJ_DEFAULT") else 10
    k_wm = int(cfg.op_steps)
    ld = arm.build_loader(a, cfg, max(k, k_wm), names_ep)
    pc = arm._plan_cfg(cfg, a)
    print(f"[plan_cfg] {pc}", flush=True)

    stride = max(1, int(a0.window_stride))
    sel = [(wi, ei, t) for wi, (ei, t) in enumerate(ld.windows)
           if (t - (ld.W - 1)) % stride == 0]
    if a0.max_windows:
        sel = sel[: a0.max_windows]
    print(f"[grid] windows={len(sel)} episodes={len(ld.names)}", flush=True)

    nav_true = np.zeros(len(sel), dtype=np.int64)
    nav_valid = np.zeros(len(sel), dtype=bool)
    if ld._nav_on:
        for i, (wi, ei, t) in enumerate(sel):
            nid = ld._nav_id.get(ld.clip_id[ld.names[ei]])
            nav_true[i] = 0 if nid is None else int(nid)
            nav_valid[i] = nid is not None

    # ---- instrumentation ------------------------------------------------- #
    HOLD = {}
    real_icem = R.icem_plan

    def _icem_capture(cost_fn, **kw):
        HOLD["cost_fn"] = cost_fn
        HOLD["seed_pool"] = kw.get("seed_pool")
        HOLD["proposal"] = kw.get("proposal")
        raise _Captured()

    REC = []
    real_goal = R._goal_term

    def _goal_rec(zt, g, metric="cos"):
        out = real_goal(zt, g, metric)
        x = zt.flatten(1).double()
        y = g.flatten(1).double()
        xn = x / x.norm(dim=-1, keepdim=True)
        yn = y / y.norm(dim=-1, keepdim=True)
        cos64 = (xn * yn).sum(-1)
        REC.append({
            "goal32": out.detach().float().cpu().numpy().copy(),
            "cos64": cos64.cpu().numpy().copy(),
            "goal64": (1.0 - cos64).cpu().numpy().copy(),
            "chord64": (xn - yn).norm(dim=-1).cpu().numpy().copy(),
            "xnorm": x.norm(dim=-1).cpu().numpy().copy(),
            "ynorm": y.norm(dim=-1).cpu().numpy().copy(),
            "x": x,          # kept only for the current window
            "y": y[:1],
        })
        return out

    R.icem_plan = _icem_capture
    R._goal_term = _goal_rec

    rows = []
    t0 = time.time()
    by_ep = {}
    for i, (wi, ei, t) in enumerate(sel):
        by_ep.setdefault(ei, []).append((i, wi, t))

    box_names_ref = None
    for fi, ei in enumerate(sorted(by_ep)):
        nm = ld.names[ei]
        F_ep, v_ep, kap_ep = ld._episode(nm)
        for (i, wi, t) in by_ep[ei]:
            ld._order, ld._cursor = [wi], 0
            b = ld.batch(1)
            feats = b["feats"].to(dev)
            v0 = float(v_ep[2 * t])
            nav_t = (torch.tensor([int(nav_true[i])], device=dev)
                     if ld._nav_on else None)
            HOLD.clear()
            REC.clear()
            with torch.no_grad():
                try:
                    model.plan(feats, v0=v0, nav_cmd=nav_t, plan_cfg=pc,
                               model_action_units="kappa")
                except _Captured:
                    pass
            cost_fn = HOLD.get("cost_fn")
            if cost_fn is None:
                raise RuntimeError("cost_fn was not captured — plan() path changed")
            bn, box = build_box(pc.horizon, dev, HOLD.get("seed_pool"),
                                HOLD.get("proposal"), pc.kappa_max, pc.a_max)
            if box_names_ref is None:
                box_names_ref = bn
            REC.clear()
            with torch.no_grad():
                cost32 = cost_fn(box).detach().float().cpu().numpy().copy()
            # concatenate the per-chunk goal records in candidate order
            goal32 = np.concatenate([r["goal32"] for r in REC])
            goal64 = np.concatenate([r["goal64"] for r in REC])
            cos64 = np.concatenate([r["cos64"] for r in REC])
            chord64 = np.concatenate([r["chord64"] for r in REC])
            xnorm = np.concatenate([r["xnorm"] for r in REC])
            ynorm = float(REC[0]["ynorm"][0])
            X = torch.cat([r["x"] for r in REC], 0)          # [n, D] float64
            Y = REC[0]["y"][0]                               # [D] float64
            xbar = X.mean(0)
            dev_rel = ((X - xbar).norm(dim=-1) / xbar.norm()).cpu().numpy()
            # DC structure of the field: how much of the norm is the token mean
            n_tok = None
            try:
                n_tok = int(cfg.tac_queries)
                Xt = X.view(X.shape[0], n_tok, -1)
                tok_mean = Xt.mean(1, keepdim=True)
                dc_frac = (tok_mean.expand_as(Xt).reshape(X.shape[0], -1).norm(dim=-1)
                           / X.norm(dim=-1)).cpu().numpy()
            except Exception:
                dc_frac = np.full(X.shape[0], np.nan)
            # cos between candidates themselves (vs cv, row 0)
            xn = X / X.norm(dim=-1, keepdim=True)
            cos_vs_cv = (xn * xn[0:1]).sum(-1).cpu().numpy()

            ctrl = box.detach().cpu().numpy()
            jerk_raw = (((ctrl[:, 1:, 0] - ctrl[:, :-1, 0]) / pc.dt) ** 2).mean(-1)
            kap_raw = (ctrl[..., 1] ** 2).mean(-1)

            rows.append({
                "w": int(i), "ep": int(ei), "t": int(t), "clip": ld.clip_id[nm],
                "v0": v0, "nav": int(nav_true[i]),
                "cost32": cost32.tolist(),
                "goal32": goal32.tolist(),
                "goal64": goal64.tolist(),
                "cos64": cos64.tolist(),
                "chord64": chord64.tolist(),
                "jerk_raw": jerk_raw.tolist(),
                "kap_raw": kap_raw.tolist(),
                "xnorm": xnorm.tolist(),
                "ynorm": ynorm,
                "dev_rel": dev_rel.tolist(),
                "dc_frac": dc_frac.tolist(),
                "cos_vs_cv": cos_vs_cv.tolist(),
                "box": ctrl[:, 0, :].tolist(),   # (a, kappa) at step 0
            })
            del X, Y, xn, xbar
            REC.clear()
            if len(rows) % 20 == 0:
                el = time.time() - t0
                print(f"[{len(rows)}/{len(sel)}] {el:.0f}s "
                      f"({el/len(rows):.2f}s/window)", flush=True)

    R.icem_plan = real_icem
    R._goal_term = real_goal
    out = {
        "tool": "cost_anatomy.py",
        "model": prov,
        "plan_cfg": {kk: getattr(pc, kk) for kk in
                     ("n_samples", "n_iters", "n_elites", "horizon", "dt",
                      "a_max", "kappa_max", "beta", "inject_baselines", "seed")},
        "weights_shipped": {"W_JERK": R.W_JERK, "W_KAPPA": R.W_KAPPA,
                            "W_VEND": R.W_VEND},
        "cost_metric": "cos", "cost_time_grid": "dense", "goal_time_grid": "full",
        "target_speed": None,
        "box_names": box_names_ref,
        "n_windows": len(rows),
        "wallclock_s": time.time() - t0,
        "rows": rows,
    }
    with open(a0.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print(f"[done] {len(rows)} windows -> {a0.out} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
