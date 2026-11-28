"""D-REFAV1-COST-FORMS — five candidate FORMS of the goal term, scored on the
SAME rollouts, so "re-weight vs re-form" is decided by measurement.

Same capture trick as `cost_anatomy.py` (wrap `refa_v1.icem_plan` to steal the
shipped `_cost_chunk` closure, then evaluate a designed box), so one window costs
|box| rollouts, not iCEM's 2,100.

FORMS (all computed in float64 from the SAME (zt, g) the shipped `_goal_term`
was handed, so nothing is re-derived and nothing is re-rolled):

  cos       1 - cos(x, y)                       the SHIPPED term
  chord     ||x_hat - y_hat||                   already implemented, removes the SQUARING
  ccos      1 - cos(x - x_cv, y - x_cv)         CENTRED on the zero-action field:
                                                the term becomes the ACTION-INDUCED
                                                change, which is what a planner should
                                                be ranking. Removes the common mode.
  cchord    ||(x-x_cv)_hat - (y-x_cv)_hat||     centred AND unsquared
  rel_l2    ||x - y|| / ||y||                   scale-free absolute distance
  abs_l2    ||x - y||                           unnormalised (the "chord in metres" form)

⚠️ `x_cv` is the terminal field under ZERO controls — always present in the box as
row 0 — so the centring uses no privileged information and costs no extra rollout.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, sys, time
import numpy as np
import torch


def _load_arm(repo):
    sys.path.insert(0, os.path.join(repo, "stack"))
    sys.path.insert(0, os.path.join(repo, "taniteval"))
    p = os.path.join(repo, "taniteval", "tools", "refav1_arm.py")
    spec = importlib.util.spec_from_file_location("refav1_arm", p)
    m = importlib.util.module_from_spec(spec); sys.modules["refav1_arm"] = m
    spec.loader.exec_module(m)
    return m


class _Captured(Exception):
    pass


def build_box(H, dev, seed_pool):
    names, rows = [], []
    def add(nm, a, k):
        c = torch.zeros(H, 2, device=dev); c[:, 0] = a; c[:, 1] = k
        names.append(nm); rows.append(c)
    add("cv", 0.0, 0.0)                       # MUST be row 0 (the centring ref)
    add("decel_1.5", -1.5, 0.0)
    for k in (0.2, 0.1, 0.05, 0.02, 0.01):
        add(f"kap+{k}", 0.0, +k); add(f"kap-{k}", 0.0, -k)
    for a in (4.0, 2.0, 1.5, 0.5):
        add(f"acc+{a}", +a, 0.0); add(f"acc-{a}", -a, 0.0)
    if seed_pool is not None and seed_pool.numel():
        for i in range(seed_pool.shape[0]):
            names.append(f"seed{i}"); rows.append(seed_pool[i].to(dev))
    return names, torch.stack(rows, 0)


def forms(X, Y):
    """X [n,D] float64 candidate terminal fields (row 0 = cv), Y [D] the goal."""
    x, y = X, Y[None]
    xh = x / x.norm(dim=-1, keepdim=True)
    yh = y / y.norm(dim=-1, keepdim=True)
    out = {}
    out["cos"] = (1.0 - (xh * yh).sum(-1))
    out["chord"] = (xh - yh).norm(dim=-1)
    ref = X[0:1]
    xc, yc = x - ref, y - ref
    nxc = xc.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    nyc = yc.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    xch, ych = xc / nxc, yc / nyc
    out["ccos"] = (1.0 - (xch * ych).sum(-1))
    out["cchord"] = (xch - ych).norm(dim=-1)
    out["rel_l2"] = (x - y).norm(dim=-1) / y.norm(dim=-1)
    out["abs_l2"] = (x - y).norm(dim=-1)
    # ⭐ THE TWO WELL-CONDITIONED CENTRED FORMS. `ccos`/`cchord` above divide by
    # ||x - x_cv||, which is EXACTLY ZERO for the cv row and near zero whenever
    # the goal is "hold" -- ill-conditioned on the majority stratum. These
    # normalise the DISTANCE by an action-response SCALE instead, so nothing is
    # divided by its own (possibly zero) length.
    #   boxnorm : ||x - y|| / median_i ||x_i - x_cv||   (the box's own scale)
    #   goalnorm: ||x - y|| / (||y - x_cv|| + 1e-3||x_cv||)  (the GOAL's own
    #             action-induced displacement; when the goal IS "hold", the
    #             denominator collapses and any action is correctly expensive)
    resp = (x - ref).norm(dim=-1)
    s_box = resp.median().clamp_min(1e-30)
    out["boxnorm"] = (x - y).norm(dim=-1) / s_box
    s_goal = (y - ref).norm(dim=-1) + 1e-3 * ref.norm(dim=-1)
    out["goalnorm"] = (x - y).norm(dim=-1) / s_goal.clamp_min(1e-30)
    out["_resp"] = resp                                  # ||x_i - x_cv||
    out["_goalresp"] = (y - ref).norm(dim=-1).expand(x.shape[0])
    out["_refnorm"] = ref.norm(dim=-1).expand(x.shape[0])
    return {k: v.cpu().numpy().copy() for k, v in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    for f in ("repo", "ckpt", "cache", "episodes", "out"):
        ap.add_argument(f"--{f}", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--nav", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--window-stride", type=int, default=40)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--lru", type=int, default=8)
    a0 = ap.parse_args(argv)

    arm = _load_arm(a0.repo)
    from tanitad.refs import refa_v1 as R
    dev = a0.device
    model, cfg, prov = arm.load_model(a0.ckpt, a0.config, dev, False)
    print(f"[model] step={prov['step']}", flush=True)

    class _A: pass
    a = _A()
    a.cache, a.episodes, a.labels, a.nav = a0.cache, a0.episodes, a0.labels, a0.nav
    a.lru, a.plan_seed = a0.lru, 0
    a.plan_n_samples = a.plan_n_iters = a.plan_n_elites = None
    ld = arm.build_loader(a, cfg, max(10, int(cfg.op_steps)),
                          arm.episode_names(a0.cache))
    pc = arm._plan_cfg(cfg, a)
    stride = max(1, int(a0.window_stride))
    sel = [(wi, ei, t) for wi, (ei, t) in enumerate(ld.windows)
           if (t - (ld.W - 1)) % stride == 0]
    if a0.max_windows:
        sel = sel[: a0.max_windows]
    nav_true = np.zeros(len(sel), dtype=np.int64)
    if ld._nav_on:
        for i, (wi, ei, t) in enumerate(sel):
            nid = ld._nav_id.get(ld.clip_id[ld.names[ei]])
            nav_true[i] = 0 if nid is None else int(nid)
    print(f"[grid] windows={len(sel)}", flush=True)

    HOLD, REC = {}, []
    real_icem, real_goal = R.icem_plan, R._goal_term

    def _cap(cost_fn, **kw):
        HOLD["cost_fn"] = cost_fn; HOLD["seed_pool"] = kw.get("seed_pool")
        raise _Captured()

    def _rec(zt, g, metric="cos"):
        out = real_goal(zt, g, metric)
        REC.append((zt.flatten(1).double(), g.flatten(1).double()))
        return out

    R.icem_plan, R._goal_term = _cap, _rec
    by_ep = {}
    for i, (wi, ei, t) in enumerate(sel):
        by_ep.setdefault(ei, []).append((i, wi, t))
    rows, t0, bn_ref = [], time.time(), None
    for ei in sorted(by_ep):
        nm = ld.names[ei]
        F_ep, v_ep, kap_ep = ld._episode(nm)
        for (i, wi, t) in by_ep[ei]:
            ld._order, ld._cursor = [wi], 0
            b = ld.batch(1)
            feats = b["feats"].to(dev); v0 = float(v_ep[2 * t])
            nav_t = (torch.tensor([int(nav_true[i])], device=dev)
                     if ld._nav_on else None)
            HOLD.clear(); REC.clear()
            with torch.no_grad():
                try:
                    model.plan(feats, v0=v0, nav_cmd=nav_t, plan_cfg=pc,
                               model_action_units="kappa")
                except _Captured:
                    pass
            bn, box = build_box(pc.horizon, dev, HOLD.get("seed_pool"))
            bn_ref = bn_ref or bn
            REC.clear()
            with torch.no_grad():
                HOLD["cost_fn"](box)
            X = torch.cat([r[0] for r in REC], 0)
            Y = REC[0][1][0]
            f = forms(X, Y)
            ctrl = box.detach().cpu().numpy()
            rows.append({"w": int(i), "ep": int(ei), "t": int(t),
                         "clip": ld.clip_id[nm], "v0": v0,
                         "jerk_raw": (((ctrl[:, 1:, 0] - ctrl[:, :-1, 0])
                                       / pc.dt) ** 2).mean(-1).tolist(),
                         "kap_raw": (ctrl[..., 1] ** 2).mean(-1).tolist(),
                         **{k: v.tolist() for k, v in f.items()}})
            del X, Y
            REC.clear()
            if len(rows) % 40 == 0:
                print(f"[{len(rows)}/{len(sel)}] {time.time()-t0:.0f}s", flush=True)
    R.icem_plan, R._goal_term = real_icem, real_goal
    json.dump({"tool": "cost_forms.py", "model": prov, "box_names": bn_ref,
               "forms": ["cos", "chord", "ccos", "cchord", "rel_l2", "abs_l2",
                         "boxnorm", "goalnorm"],
               "n_windows": len(rows), "wallclock_s": time.time() - t0,
               "rows": rows}, open(a0.out, "w", encoding="utf-8"))
    print(f"[done] {len(rows)} -> {a0.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
