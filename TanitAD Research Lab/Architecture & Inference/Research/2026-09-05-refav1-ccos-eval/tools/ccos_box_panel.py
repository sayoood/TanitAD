"""D-REFAV1-CCOS-EVAL — the goal term under cos / chord / ccos on the SAME captured fields,
through the REAL `refa_v1._goal_term`, on the full 282-window eval grid.

Same capture discipline as `2026-09-04-refav1-cost-scale/tools/cost_anatomy.py`: wrap
`refa_v1.icem_plan` to steal the shipped `_cost_chunk` closure that `plan()` builds for THIS
window, evaluate a DESIGNED candidate box through it (|box| rollouts instead of iCEM's 2,100),
and record what `_goal_term` was handed. One capture per metric per window, so the goal term
under each metric is produced by the closure `plan(cost_metric=...)` itself would use —
including `z_ref`, which only the `ccos` closure computes.

RECORDS per window: for each metric m in (cos, chord, ccos): `goal_<m>` [n_box] (the REAL
`_goal_term`, float32, as the planner sees it) and `cost_<m>` [n_box] (the closure's TOTAL
cost at SHIPPED weights); the penalties `jerk_raw`, `kap_raw`; the CONTROLS: zero-model ptp
(all rows = the cv field), ccos(cv), identity (candidate = goal), and the float64 cross-check
`ccos64` recomputed from the captured (zt, g, z_ref).
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


def build_box(H, dev, seed_pool, proposal):
    """The 26-row 'anatomy' box of cost_anatomy.py (+ seeds, + proposal), row 0 = cv."""
    names, rows = [], []

    def add(nm, a_val, k_val):
        c = torch.zeros(H, 2, device=dev); c[:, 0] = a_val; c[:, 1] = k_val
        names.append(nm); rows.append(c)
    add("cv", 0.0, 0.0)
    add("decel_1.5", -1.5, 0.0)
    for k in (0.2, 0.1, 0.05, 0.02, 0.01):
        add(f"kap+{k}", 0.0, +k); add(f"kap-{k}", 0.0, -k)
    for aa in (4.0, 2.0, 1.5, 0.5):
        add(f"acc+{aa}", +aa, 0.0); add(f"acc-{aa}", -aa, 0.0)
    add("turnL_slow", -1.0, +0.1); add("turnR_slow", -1.0, -0.1)
    for sgn, nm in ((+1.0, "rampL"), (-1.0, "rampR")):
        c = torch.zeros(H, 2, device=dev)
        c[:, 1] = sgn * torch.linspace(0.0, 0.15, H, device=dev)
        names.append(nm); rows.append(c)
    if seed_pool is not None and seed_pool.numel():
        for i in range(seed_pool.shape[0]):
            names.append(f"seed{i}"); rows.append(seed_pool[i].to(dev))
    if proposal is not None:
        names.append("proposal"); rows.append(proposal.to(dev))
    return names, torch.stack(rows, 0)


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
    if tuple(R.COST_METRICS) != ("cos", "chord", "ccos"):
        raise SystemExit(f"STALE STACK: COST_METRICS={R.COST_METRICS} from {R.__file__}")
    print(f"[verify] COST_METRICS={R.COST_METRICS} from {R.__file__}", flush=True)
    dev = a0.device
    model, cfg, prov = arm.load_model(a0.ckpt, a0.config, dev, False)
    print(f"[model] step={prov['step']} target_space={cfg.target_space}", flush=True)

    class _A: pass
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
    print(f"[grid] windows={len(sel)} episodes={len(ld.names)} plan={pc}", flush=True)

    HOLD, REC = {}, []
    real_icem, real_goal = R.icem_plan, R._goal_term

    def _cap(cost_fn, **kw):
        HOLD["cost_fn"] = cost_fn; HOLD["seed_pool"] = kw.get("seed_pool")
        HOLD["proposal"] = kw.get("proposal")
        raise _Captured()

    def _rec(zt, g, metric="cos", z_ref=None):
        out = real_goal(zt, g, metric, z_ref)
        REC.append((zt.detach().flatten(1).double(), g.detach().flatten(1).double(),
                    None if z_ref is None else z_ref.detach().flatten(1).double(),
                    out.detach().float().cpu().numpy().copy()))
        return out

    R.icem_plan, R._goal_term = _cap, _rec
    by_ep = {}
    for i, (wi, ei, t) in enumerate(sel):
        by_ep.setdefault(ei, []).append((i, wi, t))
    rows, t0, bn_ref = [], time.time(), None
    W = {"W_JERK": float(R.W_JERK), "W_KAPPA": float(R.W_KAPPA), "W_VEND": float(R.W_VEND)}
    for ei in sorted(by_ep):
        nm = ld.names[ei]
        F_ep, v_ep, kap_ep = ld._episode(nm)
        for (i, wi, t) in by_ep[ei]:
            ld._order, ld._cursor = [wi], 0
            b = ld.batch(1)
            feats = b["feats"].to(dev); v0 = float(v_ep[2 * t])
            nav_t = (torch.tensor([int(nav_true[i])], device=dev) if ld._nav_on else None)
            row = {"w": int(i), "ep": int(ei), "t": int(t), "clip": ld.clip_id[nm],
                   "v0": v0, "nav": int(nav_true[i])}
            box = None
            for metric in ("cos", "chord", "ccos"):
                HOLD.clear(); REC.clear()
                with torch.no_grad():
                    try:
                        res = model.plan(feats, v0=v0, nav_cmd=nav_t, plan_cfg=pc,
                                         model_action_units="kappa", cost_metric=metric)
                    except _Captured:
                        pass
                if box is None:
                    bn, box = build_box(pc.horizon, dev, HOLD.get("seed_pool"),
                                        HOLD.get("proposal"))
                    bn_ref = bn_ref or bn
                    ctrl = box.detach().cpu().numpy()
                    row["jerk_raw"] = (((ctrl[:, 1:, 0] - ctrl[:, :-1, 0]) / pc.dt) ** 2
                                       ).mean(-1).tolist()
                    row["kap_raw"] = (ctrl[..., 1] ** 2).mean(-1).tolist()
                    row["box_a0"] = ctrl[:, 0, 0].tolist()
                    row["box_k0"] = ctrl[:, 0, 1].tolist()
                    row["box_kmean"] = ctrl[..., 1].mean(-1).tolist()
                REC.clear()
                with torch.no_grad():
                    cost = HOLD["cost_fn"](box).detach().float().cpu().numpy()
                goal = np.concatenate([r[3] for r in REC])
                row[f"goal_{metric}"] = goal.tolist()
                row[f"cost_{metric}"] = cost.tolist()
                if metric == "ccos":
                    X = torch.cat([r[0] for r in REC], 0)          # [n, D] f64
                    Y = REC[0][1][0]                                 # [D]
                    Rf = REC[0][2][0]                                # [D]
                    xc, yc = X - Rf[None], (Y - Rf)[None]
                    nx = xc.norm(dim=-1, keepdim=True).clamp_min(1e-300)
                    ny = yc.norm(dim=-1, keepdim=True).clamp_min(1e-300)
                    row["ccos64"] = (1.0 - ((xc / nx) * (yc / ny)).sum(-1)).cpu().numpy().tolist()
                    row["resp_rel"] = (xc.norm(dim=-1) / Rf.norm()).cpu().numpy().tolist()
                    row["goalresp_rel"] = float((Y - Rf).norm() / Rf.norm())
                    row["refnorm"] = float(Rf.norm())
                    # ---- CONTROLS on the real fields, real function ------------
                    zt_cv = REC[0][0][0:1].float().reshape(1, *REC[0][1].shape[1:]) \
                        if False else None
                    # zero-model: every candidate = the cv field (row 0). The
                    # captured tensors are flattened; rebuild [n, Q, d] shapes.
                    Q, D = int(cfg.tac_queries), int(cfg.d_state)
                    x_cv = X[0].float().reshape(1, Q, D).to(dev)
                    g_t = Y.float().reshape(1, Q, D).to(dev)
                    r_t = Rf.float().reshape(1, Q, D).to(dev)
                    same = x_cv.expand(9, Q, D)
                    ctl = {}
                    with torch.no_grad():
                        for m2 in ("cos", "chord", "ccos"):
                            v = real_goal(same, g_t.expand(9, Q, D), m2,
                                          r_t if m2 == "ccos" else None)
                            ctl[f"zero_model_ptp_{m2}"] = float(v.max() - v.min())
                            ctl[f"zero_model_value_{m2}"] = float(v[0])
                        ctl["ccos_cv"] = float(real_goal(x_cv, g_t, "ccos", r_t)[0])
                        ctl["ident_cos"] = float(real_goal(g_t, g_t, "cos")[0])
                        ctl["ident_ccos"] = float(real_goal(g_t, g_t, "ccos", r_t)[0])
                        ctl["ident_chord"] = float(real_goal(g_t, g_t, "chord")[0])
                    row["controls"] = ctl
                    row["ccos_res"] = None
            rows.append(row)
            REC.clear()
            if len(rows) % 20 == 0:
                el = time.time() - t0
                print(f"[{len(rows)}/{len(sel)}] {el:.0f}s ({el/len(rows):.2f}s/window)",
                      flush=True)
    R.icem_plan, R._goal_term = real_icem, real_goal
    out = {"tool": "ccos_box_panel.py", "model": prov, "box_names": bn_ref,
           "metrics": ["cos", "chord", "ccos"], "weights_shipped": W,
           "plan_cfg": {kk: getattr(pc, kk) for kk in ("n_samples", "n_iters", "n_elites",
                                                       "horizon", "dt", "a_max", "kappa_max",
                                                       "beta", "inject_baselines", "seed")},
           "goal_dim": [int(cfg.tac_queries), int(cfg.d_state)],
           "n_windows": len(rows), "wallclock_s": time.time() - t0, "rows": rows}
    with open(a0.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print(f"[done] {len(rows)} windows -> {a0.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
