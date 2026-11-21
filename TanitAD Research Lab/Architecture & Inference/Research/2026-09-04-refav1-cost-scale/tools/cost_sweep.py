"""D-REFAV1-COST-SWEEP — re-run the SHIPPED iCEM planner at inference under a
grid over (cost_metric, W_JERK, W_KAPPA). NO RETRAINING: these are cost weights
read by `RefAV1.plan`'s `_cost_chunk` closure at plan time.

⭐ THE PROBES THAT DETECT PLANNING (not ADE):
   * fraction of windows whose plan has kappa IDENTICALLY 0
   * number of DISTINCT plans emitted (bit-exact over the [H,2] control block)
⛔ `baseline_won_frac` is NOT recorded as truth: both tie-break repairs are gated
on the plan's LABEL. This tool banks the CONTROLS and derives triviality from
CONTENT (== zeros / == the decel_1.5 block), alongside the label, so the two can
be compared.

Also banks, per window, the shipped tie-break diagnostic: the baseline stack's
cost evaluated ALONE vs inside a padded batch, which is the mechanism that lets a
plan byte-identical to a baseline be labelled "cem".
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch


def _load_arm(repo: str):
    sys.path.insert(0, os.path.join(repo, "stack"))
    sys.path.insert(0, os.path.join(repo, "taniteval"))
    p = os.path.join(repo, "taniteval", "tools", "refav1_arm.py")
    spec = importlib.util.spec_from_file_location("refav1_arm", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["refav1_arm"] = m
    spec.loader.exec_module(m)
    return m


# (label, cost_metric, W_JERK, W_KAPPA)
SETTINGS = [
    ("S0_shipped",      "cos",   0.02,   0.05),
    ("S1_kap_1e-2x",    "cos",   0.02,   5e-4),
    ("S2_kap_1e-3x",    "cos",   0.02,   5e-5),
    ("S3_both_zero",    "cos",   0.0,    0.0),
    ("S4_jerk_zero",    "cos",   0.0,    0.05),
    ("S5_kap_zero",     "cos",   0.02,   0.0),
    ("S6_chord_shipw",  "chord", 0.02,   0.05),
    ("S7_chord_rescal", "chord", 2e-5,   5e-5),
    # the PROPORTIONATE re-balance: both weights at their measured break-even
    # (cost_anatomy: W_JERK 9.85e-05 for the goal seed, W_KAPPA 2.04e-04 for the
    # best constant-curvature candidate) -- the "re-weight, do not re-form" arm.
    ("S8_rebalance",    "cos",   1e-4,   2e-4),
]


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
    ap.add_argument("--every", type=int, default=7,
                    help="keep every Nth window of the eval grid (stratified)")
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--settings", default="",
                    help="comma-separated subset of setting labels")
    ap.add_argument("--lru", type=int, default=8)
    ap.add_argument("--out", required=True)
    a0 = ap.parse_args(argv)

    arm = _load_arm(a0.repo)
    from tanitad.refs import refa_v1 as R

    dev = a0.device
    model, cfg, prov = arm.load_model(a0.ckpt, a0.config, dev, False)
    print(f"[model] step={prov['step']} target_space={cfg.target_space}", flush=True)

    class _A:
        pass
    a = _A()
    a.cache, a.episodes, a.labels, a.nav = a0.cache, a0.episodes, a0.labels, a0.nav
    a.lru, a.plan_seed = a0.lru, 0
    a.plan_n_samples = a.plan_n_iters = a.plan_n_elites = None
    names_ep = arm.episode_names(a0.cache)
    k, k_wm = 10, int(cfg.op_steps)
    ld = arm.build_loader(a, cfg, max(k, k_wm), names_ep)
    pc = arm._plan_cfg(cfg, a)

    stride = max(1, int(a0.window_stride))
    sel_all = [(wi, ei, t) for wi, (ei, t) in enumerate(ld.windows)
               if (t - (ld.W - 1)) % stride == 0]
    sel = [(gi, s) for gi, s in enumerate(sel_all) if gi % max(1, a0.every) == 0]
    if a0.max_windows:
        sel = sel[: a0.max_windows]
    print(f"[grid] eval-grid windows={len(sel_all)}  swept={len(sel)} "
          f"(every {a0.every}th)  episodes={len(ld.names)}", flush=True)

    nav_true = np.zeros(len(sel_all), dtype=np.int64)
    if ld._nav_on:
        for i, (wi, ei, t) in enumerate(sel_all):
            nid = ld._nav_id.get(ld.clip_id[ld.names[ei]])
            nav_true[i] = 0 if nid is None else int(nid)

    setts = SETTINGS
    if a0.settings:
        want = set(a0.settings.split(","))
        setts = [s for s in SETTINGS if s[0] in want]
    print(f"[settings] {[s[0] for s in setts]}", flush=True)

    W_J0, W_K0 = R.W_JERK, R.W_KAPPA
    rows = []
    t0 = time.time()
    by_ep = {}
    for gi, (wi, ei, t) in sel:
        by_ep.setdefault(ei, []).append((gi, wi, t))

    n_done = 0
    for ei in sorted(by_ep):
        nm = ld.names[ei]
        F_ep, v_ep, kap_ep = ld._episode(nm)
        for (gi, wi, t) in by_ep[ei]:
            ld._order, ld._cursor = [wi], 0
            b = ld.batch(1)
            feats = b["feats"].to(dev)
            v0 = float(v_ep[2 * t])
            nav_t = (torch.tensor([int(nav_true[gi])], device=dev)
                     if ld._nav_on else None)
            rec = {"gi": int(gi), "ep": int(ei), "t": int(t),
                   "clip": ld.clip_id[nm], "v0": v0, "nav": int(nav_true[gi]),
                   "plans": {}}
            for (lbl, metric, wj, wk) in setts:
                R.W_JERK, R.W_KAPPA = wj, wk
                tp = time.time()
                with torch.no_grad():
                    res = model.plan(feats, v0=v0, nav_cmd=nav_t, plan_cfg=pc,
                                     model_action_units="kappa",
                                     cost_metric=metric)
                ctrl = res.controls.detach().float().cpu().numpy()
                ga = res.goal_action or {}
                rec["plans"][lbl] = {
                    "controls": ctrl.tolist(),
                    "source": res.source,
                    "cost": float(res.cost),
                    "baseline_costs": {kk: float(vv) for kk, vv
                                       in (res.baseline_costs or {}).items()},
                    "fine_costs": {kk: float(vv) for kk, vv
                                   in (res.fine_costs or {}).items()},
                    "coarse_fine_agree": res.coarse_fine_agree,
                    "goal_source": res.goal_source,
                    "goal_lat": (str(ga.get("lat")) if ga else None),
                    "goal_lon": (str(ga.get("lon")) if ga else None),
                    "goal_controls": (ga["controls"].detach().float().cpu()
                                      .numpy().tolist() if ga else None),
                    "n_evaluated": int(res.n_evaluated),
                    "plan_s": time.time() - tp,
                }
            R.W_JERK, R.W_KAPPA = W_J0, W_K0
            rows.append(rec)
            n_done += 1
            el = time.time() - t0
            print(f"[{n_done}/{len(sel)}] {el:.0f}s  ({el/n_done:.1f}s/window, "
                  f"{el/n_done/max(1,len(setts)):.1f}s/plan)", flush=True)
            with open(a0.out, "w", encoding="utf-8") as fh:
                json.dump({"tool": "cost_sweep.py", "model": prov,
                           "plan_cfg": {kk: getattr(pc, kk) for kk in
                                        ("n_samples", "n_iters", "n_elites",
                                         "horizon", "dt", "a_max", "kappa_max",
                                         "beta", "inject_baselines", "seed")},
                           "settings": [{"label": s[0], "cost_metric": s[1],
                                         "W_JERK": s[2], "W_KAPPA": s[3]}
                                        for s in setts],
                           "eval_grid_windows": len(sel_all),
                           "every": a0.every, "n_windows": len(rows),
                           "wallclock_s": time.time() - t0,
                           "rows": rows}, fh)
    print(f"[done] {len(rows)} windows x {len(setts)} settings -> {a0.out} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
