"""D-REFAV1-CCOS-EVAL step 1 — the goal term of the REAL implementation, under all three
metrics, on the SAME captured fields, for every window of the banked 282-window grid.

Same capture trick as the cost-scale package's `cost_forms.py`: wrap `refa_v1.icem_plan` to
steal the shipped `_cost_chunk` closure, then evaluate a designed candidate box through it.
THE DIFFERENCE from `cost_forms.py` (which RE-IMPLEMENTED the forms in float64): here the
values are produced by `refa_v1._goal_term` itself — the ONE site the planner's goal
distance is computed — in float32, with `z_ref` taken from the planner's OWN
`_zero_action_ref` (the cv rollout of the same window, same predictor, same z0), by calling
`plan(..., cost_metric="ccos")` so that reference is actually built and handed over.
A float64 re-implementation is banked BESIDE it as the cross-check, never instead of it.

Per window it banks, per candidate of the box:
    cos32 / chord32 / ccos32   the REAL `_goal_term` values (float32)
    cos64 / chord64 / ccos64   the float64 re-implementation from the same (x, y, z_ref)
    jerk_raw / kap_raw         the penalties' raw terms for the candidate
and per window: ||z_ref||, ||x_cv - z_ref|| (the batch-composition residual: the cv row of
the box is rolled in a batch of |box| while z_ref is rolled alone — how far from the zero
vector cv's centred field really is), ||g - z_ref|| / ||z_ref|| (the goal's own action-induced
displacement: float rounding when the goal is "hold"), the decoded goal tokens, nav.

Zero extra rollouts beyond |box| per window. n = 282 windows / 141 episodes on Thor: ~5 min.
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
    add("cv", 0.0, 0.0)                       # row 0 = the do-nothing candidate
    add("decel_1.5", -1.5, 0.0)
    for k in (0.2, 0.1, 0.05, 0.02, 0.01):
        add(f"kap+{k}", 0.0, +k); add(f"kap-{k}", 0.0, -k)
    for a in (4.0, 2.0, 1.5, 0.5):
        add(f"acc+{a}", +a, 0.0); add(f"acc-{a}", -a, 0.0)
    n_designed = len(names)
    if seed_pool is not None and seed_pool.numel():
        for i in range(seed_pool.shape[0]):
            names.append(f"seed{i}"); rows.append(seed_pool[i].to(dev))
    return names, n_designed, torch.stack(rows, 0)


def forms64(X, Y, Rf):
    x, y, r = X, Y[None], Rf[None]
    xh = x / x.norm(dim=-1, keepdim=True); yh = y / y.norm(dim=-1, keepdim=True)
    xc, yc = x - r, y - r
    xch = xc / xc.norm(dim=-1, keepdim=True).clamp_min(1e-300)
    ych = yc / yc.norm(dim=-1, keepdim=True).clamp_min(1e-300)
    return {"cos64": 1.0 - (xh * yh).sum(-1), "chord64": (xh - yh).norm(dim=-1),
            "ccos64": 1.0 - (xch * ych).sum(-1)}


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
    assert "ccos" in R.COST_METRICS, f"STALE STACK: {R.COST_METRICS} from {R.__file__}"
    dev = a0.device
    model, cfg, prov = arm.load_model(a0.ckpt, a0.config, dev, False)
    print(f"[model] step={prov['step']} from {R.__file__}", flush=True)

    class _A:
        pass
    a = _A()
    a.cache, a.episodes, a.labels, a.nav = a0.cache, a0.episodes, a0.labels, a0.nav
    a.lru, a.plan_seed = a0.lru, 0
    a.plan_n_samples = a.plan_n_iters = a.plan_n_elites = None
    ld = arm.build_loader(a, cfg, max(10, int(cfg.op_steps)), arm.episode_names(a0.cache))
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
    print(f"[grid] windows={len(sel)} stride={stride}", flush=True)

    HOLD, REC = {}, []
    real_icem, real_goal = R.icem_plan, R._goal_term

    def _cap(cost_fn, **kw):
        HOLD["cost_fn"] = cost_fn; HOLD["seed_pool"] = kw.get("seed_pool")
        raise _Captured()

    def _rec(zt, g, metric="cos", z_ref=None):
        # the REAL function, called by the REAL closure; we only look.
        REC.append((zt.detach().flatten(1), g.detach().flatten(1),
                    None if z_ref is None else z_ref.detach().flatten(1), metric))
        return real_goal(zt, g, metric, z_ref)

    R.icem_plan, R._goal_term = _cap, _rec
    by_ep = {}
    for i, (wi, ei, t) in enumerate(sel):
        by_ep.setdefault(ei, []).append((i, wi, t))
    rows, t0, bn_ref, nd_ref = [], time.time(), None, None
    for ei in sorted(by_ep):
        nm = ld.names[ei]
        F_ep, v_ep, kap_ep = ld._episode(nm)
        for (i, wi, t) in by_ep[ei]:
            ld._order, ld._cursor = [wi], 0
            b = ld.batch(1)
            feats = b["feats"].to(dev); v0 = float(v_ep[2 * t])
            nav_t = (torch.tensor([int(nav_true[i])], device=dev) if ld._nav_on else None)
            HOLD.clear(); REC.clear()
            res_goal = {}
            with torch.no_grad():
                try:
                    # ccos so that plan() builds its own z_ref and hands it to _goal_term
                    model.plan(feats, v0=v0, nav_cmd=nav_t, plan_cfg=pc,
                               model_action_units="kappa", cost_metric="ccos")
                except _Captured:
                    pass
            bn, nd, box = build_box(pc.horizon, dev, HOLD.get("seed_pool"))
            bn_ref, nd_ref = bn_ref or bn, nd_ref or nd
            REC.clear()
            with torch.no_grad():
                HOLD["cost_fn"](box)              # the REAL closure, metric = ccos
            assert REC and all(r[3] == "ccos" and r[2] is not None for r in REC), \
                "the closure did not call _goal_term with ccos + z_ref"
            X = torch.cat([r[0] for r in REC], 0)            # [n, D] float32
            Y = REC[0][1][0]                                 # [D]
            Z = REC[0][2][0]                                 # [D]  the planner's OWN z_ref
            with torch.no_grad():
                v32 = {m: real_goal(X[:, None, :], Y[None, None, :].expand(X.shape[0], 1, -1),
                                    m, Z[None, None, :]).float().cpu().numpy()
                       for m in ("cos", "chord", "ccos")}
                f64 = forms64(X.double(), Y.double(), Z.double())
            ctrl = box.detach().cpu().numpy()
            xcv = X[0]
            row = {"w": int(i), "ep": int(ei), "t": int(t), "clip": ld.clip_id[nm], "v0": v0,
                   "nav": int(nav_true[i]),
                   "jerk_raw": (((ctrl[:, 1:, 0] - ctrl[:, :-1, 0]) / pc.dt) ** 2).mean(-1).tolist(),
                   "kap_raw": (ctrl[..., 1] ** 2).mean(-1).tolist(),
                   "cos": v32["cos"].tolist(), "chord": v32["chord"].tolist(),
                   "ccos": v32["ccos"].tolist(),
                   "cos64": f64["cos64"].cpu().numpy().tolist(),
                   "chord64": f64["chord64"].cpu().numpy().tolist(),
                   "ccos64": f64["ccos64"].cpu().numpy().tolist(),
                   "norm_zref": float(Z.norm()), "norm_goal": float(Y.norm()),
                   "cv_minus_zref_norm": float((xcv - Z).norm()),
                   "goal_minus_zref_rel": float((Y - Z).norm() / Z.norm().clamp_min(1e-30)),
                   "resp_rel": ((X - Z[None]).norm(dim=-1) / Z.norm().clamp_min(1e-30)).cpu().numpy().tolist()}
            rows.append(row)
            del X, Y, Z
            REC.clear()
            if len(rows) % 40 == 0:
                print(f"[{len(rows)}/{len(sel)}] {time.time()-t0:.0f}s", flush=True)
    R.icem_plan, R._goal_term = real_icem, real_goal
    json.dump({"tool": "ccos_panel_probe.py", "model": prov, "box_names": bn_ref,
               "n_designed": nd_ref, "forms": ["cos", "chord", "ccos"],
               "tag": "REAL refa_v1._goal_term (float32) on captured fields, z_ref = plan()'s own "
                      "_zero_action_ref; float64 re-implementation banked beside it",
               "n_windows": len(rows), "wallclock_s": time.time() - t0, "rows": rows},
              open(a0.out, "w", encoding="utf-8"))
    print(f"[done] {len(rows)} -> {a0.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
