#!/usr/bin/env python
"""L3 + L4 — the CHORD metric and the weight it forces, measured JOINTLY.

Pre-registered in `./SPEC.md` (written before this file ran) under
`Project Steering/PREREG_TACTICAL_DECODER.md` §5 (*"L3+L4 jointly, never L3
alone"*), §7 S3 and §8.2 Q2.

WHAT IT MEASURES, per window, on the planner's OWN candidate box
----------------------------------------------------------------
The full 2 x 2 x 2 cell set ``{cos, chord} x {kappa, steer} x {full, plan}``:

  * **the kappa-marginal** — 21 CONSTANT candidates at ``a = 0`` on the same
    kappa axis the banked cost surface used (`build_grid`, zero forced exactly
    on-grid), the goal term evaluated by the SHIPPED `refa_v1._goal_term` in
    float32 and, on the same float32 fields, in float64. Jerk is identically 0
    on constant candidates, so ``total(k) - total(0)`` is the goal response
    PLUS ``W_KAPPA*k^2`` and nothing else;
  * **the named candidates** the planner actually ranks — `cv`, `hold_v0`,
    `proposal`, `decel_1.5` and `goal_canonical` (the decoded turn's own seed);
  * **the representable-step read** the claim rests on: ``n_distinct_f32``,
    the REALISED float32 quantum (smallest positive difference actually taken
    on the grid), and how many float32 steps the TRUE (f64) kappa response
    spans at that metric's own operating point.

CONTROLS THAT MUST READ KNOWN VALUES (a failure VOIDS the panel; SPEC §3.7)
--------------------------------------------------------------------------
  X0  ZERO-MODEL      the predictor is fed the ZERO action for every candidate,
                      so it cannot see kappa -> the goal term's peak-to-peak
                      along kappa must be EXACTLY 0.0 under BOTH metrics and
                      ``total(k) - total(0)`` must recover ``W_KAPPA*k^2``
                      EXACTLY. This isolates the penalty, exactly.
  X1  CONSTANT-COST   goal term dropped and all weights 0 -> c_total == 0.0
                      everywhere and the argmin is the first grid index.
  X2  HARNESS GATE    (a) our re-scoring == the SHIPPED `_cost_chunk` via
                      `PlanResult.baseline_costs` (<= 1e-6 rel), and our own
                      rollout+metric == `cost_surface_probe`'s `c_goal`
                      BIT-IDENTICALLY at units="kappa";
                      (b) a real `plan()` re-run reproduces the BANKED `cl`
                      trajectory to < 3.8e-6 m — the precedent gate.
                      ⛔ Run BEFORE any conclusion is drawn.
  X3  REGRESSION      ``sqrt(2*(1-cos))`` computed FROM the float32 cosine must
                      recover NO resolution. If it does, the panel cannot see
                      the defect it exists to catch.
  X6  n PRINTED       windows / EPISODES / distinct decoded tokens, every time.

⛔ READ-ONLY DEPENDENCIES. `taniteval/tools/cost_surface_probe.py` is IMPORTED
and never modified: its `WindowContext` mirrors `plan()` statement for
statement and is C1-gated against the shipped cost, which is the only reason a
re-scoring is admissible at all. Its md5 is banked in the output.

TIER: T0 (open-loop, banked checkpoints; a cost/WM diagnostic, never a driving
claim). EVIDENCE CLASS: MEASURED. GPU: forward-only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

KAPPA_MAX = 0.2                  #: PlanConfig.kappa_max — the candidate box
COS32_QUANTUM = 5.9604644775390625e-08   #: spacing(1f)/2 — `1-cos`'s own step
BANKED_GATE_M = 3.8e-6           #: the precedent trajectory-reproduction gate

METRICS = ("cos", "chord")
UNITS = (("kappa", "A"), ("steer", "B"))
GOALS = ("full", "plan")         #: goal_time_grid — L0 OFF / L0 ON


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def spacing32(x: float) -> float:
    x = abs(float(x))
    if x == 0.0:
        return float(np.spacing(np.float32(0.0)))
    return float(np.spacing(np.float32(x)))


#: ⛔ THE STRUCTURAL float32 FLOOR OF EACH METRIC, and why `spacing32(value)`
#: alone is the WRONG denominator (found on this tool's first smoke run, where
#: it made the shipped cosine look like it resolved 1e7 steps).
#:   * ``"cos"``  -- the term is ``1.0 - c`` with ``c`` in [0.5, 1), whose
#:     spacing is ``2**-24``. The result is therefore an integer multiple of
#:     5.96e-08 NO MATTER HOW SMALL IT IS, so its resolution does NOT shrink
#:     with its value; ``spacing32(1e-07)`` is 1e-14 and is a fiction.
#:   * ``"chord"`` -- a norm of a difference: it IS resolved at its own scale,
#:     so `spacing32(value)` is the honest floor and there is no structural
#:     term to add.
STRUCTURAL_Q32 = {"cos": COS32_QUANTUM, "chord": 0.0}


def resolution32(metric: str, value: float) -> float:
    """The float32 resolution a metric ACTUALLY delivers at `value`."""
    return max(spacing32(value), STRUCTURAL_Q32[metric])


def quantum(vals) -> dict:
    """The REALISED float32 quantum of a marginal: the smallest positive
    difference the metric actually takes on the grid. `None` when the metric
    collapsed the whole box to one value — which is itself the finding."""
    u = np.unique(np.asarray(vals, dtype=np.float32))
    d = np.diff(u)
    d = d[d > 0]
    return {"n_distinct_f32": int(u.size),
            "q_realised_f32": float(d.min()) if d.size else None,
            "q_ulp_at_median": spacing32(float(np.median(u)))}


def _q(v) -> dict | None:
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not v.size:
        return None
    return {"n": int(v.size), "median": float(np.median(v)),
            "mean": float(v.mean()), "min": float(v.min()),
            "max": float(v.max()), "p25": float(np.percentile(v, 25)),
            "p75": float(np.percentile(v, 75))}


def _cluster_ci(vals, eps, n_boot: int = 10000, seed: int = 0) -> dict | None:
    """Episode-CLUSTER bootstrap on the MEDIAN. The clusters are EPISODES, not
    windows: an episode's windows share a decoded token and are not
    independent (the anti-conservative free permutation lesson)."""
    v = np.asarray(vals, float)
    e = np.asarray(eps)
    ok = np.isfinite(v)
    v, e = v[ok], e[ok]
    if not v.size:
        return None
    ue = np.unique(e)
    rng = np.random.default_rng(seed)
    idx = {u: np.flatnonzero(e == u) for u in ue}
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(ue, size=ue.size, replace=True)
        out[b] = np.median(np.concatenate([v[idx[u]] for u in pick]))
    return {"median": float(np.median(v)), "n": int(v.size),
            "n_clusters": int(ue.size),
            "ci95": [float(np.percentile(out, 2.5)),
                     float(np.percentile(out, 97.5))]}


# --------------------------------------------------------------------------- #
def collect(a) -> dict:
    import torch

    sys.path.insert(0, a.arm_tool_dir)
    CS = _load(a.cost_surface_probe, "cost_surface_probe_ro")
    import refav1_arm as ARM                                     # noqa: N812
    import tanitad.refs.refa_v1 as R
    from tanitad.refs.refa_v1 import _goal_term

    dev = a.device
    model, cfg, prov = ARM.load_model(a.ckpt, a.config, dev, False)
    H = int(cfg.plan_steps)
    pc = ARM._plan_cfg(cfg, argparse.Namespace(
        plan_seed=0, plan_n_samples=None, plan_n_iters=None,
        plan_n_elites=None))
    names = ARM.episode_names(a.cache)
    if a.episodes_n:
        names = names[:int(a.episodes_n)]
    k_loader = max(int(a.horizon_k), int(cfg.op_steps))
    ld = ARM.build_loader(argparse.Namespace(
        cache=a.cache, episodes=a.episodes, lru=a.lru, labels=a.labels,
        nav=a.nav), cfg, k_loader, names)
    sel = [(wi, ei, t) for wi, (ei, t) in enumerate(ld.windows)
           if (t - (ld.W - 1)) % int(a.window_stride) == 0]
    nav_true = np.zeros(len(sel), dtype=np.int64)
    if ld._nav_on:
        for i, (wi, ei, t) in enumerate(sel):
            nid = ld._nav_id.get(ld.clip_id[ld.names[ei]])
            nav_true[i] = 0 if nid is None else int(nid)

    # the kappa-marginal: n_accel = 1 puts `a` EXACTLY on 0 (build_grid forces
    # the zero on-grid, which is why C3/C5 read their known values at all)
    grid, a_axis, k_axis = CS.build_grid(H, pc.kappa_max, pc.a_max,
                                         int(a.n_kappa), 1, dev, torch.float32)
    assert float(np.abs(a_axis).max()) == 0.0, "the accel axis is not exactly 0"
    zero_ctrl = torch.zeros_like(grid)

    print(f"[cost-repair] model={a.ckpt} step={prov.get('step')} "
          f"plan_level={cfg.plan_level} H={H} kappa_max={pc.kappa_max} "
          f"W_KAPPA={R.W_KAPPA} W_JERK={R.W_JERK}", flush=True)
    print(f"[grid] {grid.shape[0]} constant candidates, a == 0 exactly, "
          f"kappa in +-{pc.kappa_max}", flush=True)
    print(f"[pop] {len(sel)} windows / {len({e for _, e, _ in sel})} episodes "
          f"stride={a.window_stride} device={dev}", flush=True)

    by_ep: dict[int, list] = {}
    for i, (wi, ei, t) in enumerate(sel):
        by_ep.setdefault(ei, []).append((i, wi, t))

    rows: list[dict] = []
    gates = {"X2a_c1_shipped_cost": [], "X2a_bit_identity": [],
             "X2b_banked_winner": []}
    t0 = time.time()
    for fi, ei in enumerate(sorted(by_ep)):
        nm = ld.names[ei]
        # ⭐ the SIGNED ground-truth heading change over the plan horizon.
        # `cost_surface_probe._gt_turn_deg` returns the ABSOLUTE value, which
        # cannot answer "does the cost turn the way the human turned"; the
        # frame indexing (2*(t+j)) is `refav1_arm.gt_waypoints:436-441`'s and
        # is copied here rather than re-derived. GROUND TRUTH: it may only
        # SELECT and DESCRIBE windows, never enter a cost.
        _o = torch.load(ld.episode_dir / f"{nm}.v2ep.pt", map_location="cpu",
                        weights_only=False)
        poses = _o["poses"].float()

        def gt_signed(t: int, k: int) -> float | None:
            f0, f1 = 2 * t, 2 * (t + k)
            if f1 >= poses.shape[0]:
                return None
            y0, y1 = float(poses[f0, 2]), float(poses[f1, 2])
            return math.degrees(math.atan2(math.sin(y1 - y0),
                                           math.cos(y1 - y0)))

        for (i, wi, t) in by_ep[ei]:
            ld._order, ld._cursor = [wi], 0
            b = ld.batch(1)
            feats = b["feats"].to(dev)
            v0 = float(b["v0"][0])
            v0_t = torch.tensor([v0], dtype=torch.float32, device=dev)
            nav_t = (torch.tensor([int(nav_true[i])], device=dev)
                     if ld._nav_on else None)

            with torch.no_grad():
                ctx = CS.WindowContext(model, cfg, feats, v0, nav_t, pc)
                if ctx.goal_t is None:
                    raise SystemExit("no hierarchy on this checkpoint — this "
                                     "probe is a question about the GOAL term")
                named = CS.named_candidates(pc, v0, ctx.proposal,
                                            ctx.goal_controls, dev)
                nkeys = list(named)
                ncand = torch.stack([named[k] for k in nkeys])

                def roll(controls, units):
                    n = controls.shape[0]
                    acts = model._model_actions(controls, v0_t.expand(n),
                                                units)
                    zk = ctx.pred.rollout(ctx.z0.expand(n, -1, -1), acts,
                                          intent=ctx.intent, last_only=True)
                    return (zk if ctx.pred is model.tactical
                            else model._tac_field(zk))

                zt_grid = {u: roll(grid, u) for u, _ in UNITS}
                zt_named = {u: roll(ncand, u) for u, _ in UNITS}
                zt_zero = ctx.pred.rollout(
                    ctx.z0.expand(grid.shape[0], -1, -1),
                    torch.zeros_like(model._model_actions(
                        grid, v0_t.expand(grid.shape[0]), "kappa")),
                    intent=ctx.intent, last_only=True)
                if ctx.pred is not model.tactical:
                    zt_zero = model._tac_field(zt_zero)

                gi = nkeys.index("goal_canonical") if "goal_canonical" in nkeys \
                    else None
                goals = {}
                for u, _ in UNITS:
                    goals[(u, "full")] = ctx.goal_t
                    # L0: the goal re-rolled from the SEED's own feed --
                    # `refa_v1.py` goal_time_grid="plan". The seed IS
                    # `goal_canonical`, so this costs no extra rollout.
                    goals[(u, "plan")] = (zt_named[u][gi:gi + 1]
                                          if gi is not None else ctx.goal_t)

                def terms(zt, g, metric, dtype):
                    n = zt.shape[0]
                    gg = g.expand(n, *g.shape[1:])
                    return _goal_term(zt.to(dtype), gg.to(dtype),
                                      metric).detach().cpu().numpy().tolist()

                # --- the explicit charges, on the CONTROLS (never the feed) --
                def charges(c):
                    j = ((c[:, 1:, 0] - c[:, :-1, 0]) / pc.dt).pow(2).mean(-1)
                    k2 = c[..., 1].pow(2).mean(-1)
                    return (j.detach().cpu().numpy().tolist(),
                            k2.detach().cpu().numpy().tolist())

                j_grid, k2_grid = charges(grid)
                j_named, k2_named = charges(ncand)

                row = {"ep_index": fi, "ep": nm, "t": int(t), "v0": v0,
                       "nav": int(nav_true[i]),
                       "lat": ctx.goal_lat, "lon": ctx.goal_lon,
                       "zero_goal": bool(ctx.zero_goal),
                       "goal_kappa_max": (
                           float(ctx.goal_controls[:, 1].abs().max())
                           if ctx.goal_controls is not None else 0.0),
                       "named": nkeys, "jerk2_named": j_named,
                       "kappa2_named": k2_named,
                       "jerk2_grid_max": float(np.max(j_grid)),
                       "kappa2_grid": k2_grid,
                       "gt_turn_deg_signed": gt_signed(int(t),
                                                       int(a.horizon_k)),
                       "cells": {}}
                for metric in METRICS:
                    for u, conv in UNITS:
                        for gname in GOALS:
                            g = goals[(u, gname)]
                            key = f"{metric}__{conv}__{gname}"
                            row["cells"][key] = {
                                "kmarg_f32": terms(zt_grid[u], g, metric,
                                                   torch.float32),
                                "kmarg_f64": terms(zt_grid[u], g, metric,
                                                   torch.float64),
                                "named_f32": terms(zt_named[u], g, metric,
                                                   torch.float32),
                                "named_f64": terms(zt_named[u], g, metric,
                                                   torch.float64),
                            }
                # ---- X0 ZERO-MODEL: the predictor cannot see kappa --------- #
                row["X0_zero_model"] = {}
                for metric in METRICS:
                    g = goals[("kappa", "full")]
                    v32 = np.asarray(terms(zt_zero, g, metric, torch.float32))
                    tot = v32 + np.asarray(j_grid) * R.W_JERK + \
                        np.asarray(k2_grid) * R.W_KAPPA
                    k0 = int(np.argmin(np.abs(k_axis)))
                    rec = tot - tot[k0] - R.W_KAPPA * (
                        np.asarray(k2_grid) - k2_grid[k0])
                    row["X0_zero_model"][metric] = {
                        "goal_ptp": float(v32.max() - v32.min()),
                        "penalty_recovery_max_abs": float(np.abs(rec).max()),
                        # ⚠️ the recovery is a float64 SUBTRACTION of two ~1e-3
                        # numbers, so its own last bit is 1 f64 ULP of the
                        # total. Reporting the residue in ULPs is what makes
                        # "exactly" checkable rather than metric-dependent:
                        # a term of a different magnitude carries a different
                        # absolute ULP for the SAME exact arithmetic.
                        "penalty_recovery_in_f64_ulps": (
                            float(np.abs(rec).max()
                                  / np.spacing(float(np.abs(tot).max())))
                            if float(np.abs(tot).max()) > 0 else 0.0),
                        "goal_value_at_k0": float(v32[k0]),
                        "total_absmax": float(np.abs(tot).max()),
                        "argmin_kappa": float(k_axis[int(np.argmin(tot))]),
                    }
                # ---- X3 the deliberate regression -------------------------- #
                gA = goals[("kappa", "full")]
                c32 = np.asarray(terms(zt_grid["kappa"], gA, "cos",
                                       torch.float32), dtype=np.float32)
                row["X3_sqrt_of_1mcos_f32_kmarg"] = np.sqrt(
                    2.0 * np.clip(c32, 0, None)).astype(np.float32).tolist()
                rows.append(row)

                # ---- X2 the harness gates, on the first `gate_n` windows --- #
                if len(gates["X2b_banked_winner"]) < int(a.gate_n):
                    res, g1 = CS._c1_gate(
                        ctx, model, feats, v0, nav_t, pc, ARM, dev,
                        goal_field=None, label="imagined",
                        wheelbase=a.wheelbase, chunk=64, target_speed=None)
                    g1.update({"ep": nm, "t": int(t)})
                    gates["X2a_c1_shipped_cost"].append(g1)
                    mine = np.asarray(terms(zt_grid["kappa"], gA, "cos",
                                            torch.float32), dtype=np.float32)
                    theirs = ctx.score(grid, units="kappa",
                                       wheelbase=a.wheelbase,
                                       chunk=64)["c_goal"]
                    theirs = theirs.detach().float().cpu().numpy()
                    gates["X2a_bit_identity"].append({
                        "ep": nm, "t": int(t),
                        "max_abs_diff": float(np.abs(mine - theirs).max()),
                        "bit_identical": bool(np.array_equal(mine, theirs)),
                        "note": ("units='kappa' -> as_command returns the "
                                 "input object unchanged, so our rollout and "
                                 "cost_surface_probe's must be bit-identical"),
                    })
                    if a.banked_dump:
                        bg = CS._banked_gate(a.banked_dump, fi, int(t), res,
                                             v0, pc.dt, int(a.horizon_k), ARM)
                        if bg is not None:
                            bg["ep"] = nm
                            gates["X2b_banked_winner"].append(bg)
        print(f"  ep{fi:03d} {nm} ({time.time() - t0:.1f}s, "
              f"{len(rows)} windows)", flush=True)

    src = {p: _md5(p) for p in (a.cost_surface_probe,
                                os.path.join(os.path.dirname(R.__file__),
                                             "refa_v1.py"))}
    return {"rows": rows, "gates": gates,
            "meta": {"ckpt": a.ckpt, "config": a.config,
                     "step": prov.get("step"),
                     "config_source": prov.get("config_source"),
                     "n_windows": len(rows), "n_episodes": len(by_ep),
                     "window_stride": int(a.window_stride),
                     "kappa_axis": [float(x) for x in k_axis],
                     "accel_axis": [float(x) for x in a_axis],
                     "kappa_max": float(pc.kappa_max),
                     "plan_steps": H, "plan_level": cfg.plan_level,
                     "tac_vocab_version": getattr(cfg, "tac_vocab_version",
                                                  None),
                     "goal_dim": int(cfg.tac_queries * cfg.d_state),
                     "W_JERK": R.W_JERK, "W_KAPPA": R.W_KAPPA,
                     "W_VEND": R.W_VEND,
                     "cost_time_grid": "dense (held constant)",
                     "wheelbase": a.wheelbase,
                     "source_md5": src,
                     "wallclock_s": round(time.time() - t0, 1)}}


# --------------------------------------------------------------------------- #
def analyse(d: dict, banked: str | None, gt_turn_min: float) -> dict:
    rows, meta = d["rows"], d["meta"]
    k_axis = np.asarray(meta["kappa_axis"], float)
    k0 = int(np.argmin(np.abs(k_axis)))
    kmax2 = float(meta["kappa_max"]) ** 2
    wk, wj = float(meta["W_KAPPA"]), float(meta["W_JERK"])

    gt = {}
    if banked and os.path.exists(banked):
        with open(banked, encoding="utf-8") as f:
            bj = json.load(f)
        for w in bj["rows"]:
            gt[(str(w["ep_name"]), int(w["t"]))] = w
    for r in rows:
        w = gt.get((str(r["ep"]), int(r["t"])))
        r["gt_turn_deg"] = (float(w["gt_turn_deg"]) if w is not None
                            and w.get("gt_turn_deg") is not None else None)

    curved = [r for r in rows if r["goal_kappa_max"] > 0.0]
    turners = [r for r in rows
               if r["gt_turn_deg"] is not None
               and r["gt_turn_deg"] >= gt_turn_min]

    def pop(rs, label):
        return {"label": label, "n_windows": len(rs),
                "n_episodes": len({r["ep"] for r in rs}),
                "n_distinct_tokens": len({(r["lat"], r["lon"]) for r in rs}),
                "tokens": sorted({f"{r['lat']} x {r['lon']}" for r in rs}),
                "episodes": sorted({r["ep"] for r in rs})}

    out = {"populations": {
        "all": pop(rows, "every scored window"),
        "curved_goal": pop(curved, "the DECODER asked for curvature "
                                   "(goal_kappa_max > 0)"),
        f"gt_turn_ge_{gt_turn_min:g}deg": pop(
            turners, "the CAR actually turns — a DIFFERENT denominator, "
                     "never mixed with the curved-goal stratum"),
    }, "controls": {}, "cells": {}, "weights": {}}

    # ---------------- X0 / X1 / X3 -------------------------------------- #
    x0 = {m: {"goal_ptp_max": max(r["X0_zero_model"][m]["goal_ptp"]
                                  for r in rows),
              "penalty_recovery_max_abs": max(
                  r["X0_zero_model"][m]["penalty_recovery_max_abs"]
                  for r in rows),
              "penalty_recovery_max_f64_ulps": max(
                  r["X0_zero_model"][m].get("penalty_recovery_in_f64_ulps",
                                            float("nan")) for r in rows),
              "n_argmin_at_kappa_0": int(sum(
                  1 for r in rows
                  if r["X0_zero_model"][m]["argmin_kappa"] == 0.0)),
              "n": len(rows)} for m in METRICS}
    for m in METRICS:
        x0[m]["pass"] = bool(x0[m]["goal_ptp_max"] == 0.0
                             and x0[m]["penalty_recovery_max_f64_ulps"] <= 4.0
                             and x0[m]["n_argmin_at_kappa_0"] == len(rows))
    out["controls"]["X0_zero_model"] = x0
    out["controls"]["X0_note"] = (
        "the predictor is fed the ZERO action for every candidate, so it "
        "cannot see kappa: the goal term MUST be flat (peak-to-peak EXACTLY "
        "0.0 — asserted, not toleranced) and total(k)-total(0) MUST recover "
        "W_KAPPA*k^2. This isolates the penalty exactly. ⚠️ the recovery "
        "residue is quoted in float64 ULPs of the total, because the residue "
        "is a property of THIS analysis's subtraction and its absolute size "
        "therefore scales with the metric's magnitude: `cos` reads 0.0 and "
        "`chord` reads ~1 ULP for the SAME exact arithmetic.")

    # X1 constant-cost: arithmetic, but it must still be executed and read
    zc = np.zeros(k_axis.size)
    out["controls"]["X1_constant_cost"] = {
        "c_total_max_abs": float(np.abs(zc).max()),
        "argmin_index": int(np.argmin(zc)),
        "pass": bool(np.abs(zc).max() == 0.0 and int(np.argmin(zc)) == 0),
        "note": "goal term dropped and every weight 0 -> c_total == 0 "
                "everywhere; the argmin is the FIRST index, a deterministic "
                "tie, so the argmin machinery cannot manufacture structure"}

    # ---------------- the per-cell reads -------------------------------- #
    def cellq(rs, key, field):
        return [quantum(r["cells"][key][field]) for r in rs]

    for metric in METRICS:
        for _u, conv in UNITS:
            for gname in GOALS:
                key = f"{metric}__{conv}__{gname}"
                nd, qr, st_f32, st_true, ptp32, ptp64, argk = (
                    [], [], [], [], [], [], [])
                for r in rows:
                    c = r["cells"][key]
                    v32 = np.asarray(c["kmarg_f32"], dtype=np.float32)
                    v64 = np.asarray(c["kmarg_f64"], float)
                    q = quantum(v32)
                    nd.append(q["n_distinct_f32"])
                    qr.append(q["q_realised_f32"])
                    p32 = float(v32.max() - v32.min())
                    p64 = float(v64.max() - v64.min())
                    ptp32.append(p32)
                    ptp64.append(p64)
                    st_f32.append(p32 / q["q_realised_f32"]
                                  if q["q_realised_f32"] else 0.0)
                    # ⛔ the denominator is the metric's DELIVERED float32
                    # resolution, not `spacing32(value)`: `1 - cos` is an
                    # integer multiple of 5.96e-08 however small it is.
                    st_true.append(p64 / resolution32(
                        metric, float(np.median(v32))))
                    argk.append(float(k_axis[int(np.argmin(v32))]))
                out["cells"][key] = {
                    "n_windows": len(rows),
                    "n_distinct_f32": _q(nd),
                    "q_realised_f32": _q(qr),
                    "goal_ptp_along_kappa_f32": _q(ptp32),
                    "goal_ptp_along_kappa_f64": _q(ptp64),
                    "steps_f32_ptp_over_realised_quantum": _q(st_f32),
                    "steps_true_f64ptp_over_f32ulp": _q(st_true),
                    "n_goal_argmin_at_kappa_0": int(sum(
                        1 for x in argk if x == 0.0)),
                }

    # the representable-step GAIN, paired per window, chord vs cos
    for _u, conv in UNITS:
        for gname in GOALS:
            kc = f"cos__{conv}__{gname}"
            kh = f"chord__{conv}__{gname}"
            gain_nd, gain_true, eps = [], [], []
            for r in rows:
                a32 = quantum(np.asarray(r["cells"][kc]["kmarg_f32"],
                                         dtype=np.float32))
                b32 = quantum(np.asarray(r["cells"][kh]["kmarg_f32"],
                                         dtype=np.float32))
                gain_nd.append(b32["n_distinct_f32"] / a32["n_distinct_f32"])
                pa = np.asarray(r["cells"][kc]["kmarg_f64"], float)
                pb = np.asarray(r["cells"][kh]["kmarg_f64"], float)
                va = np.asarray(r["cells"][kc]["kmarg_f32"], dtype=np.float32)
                vb = np.asarray(r["cells"][kh]["kmarg_f32"], dtype=np.float32)
                ta = (pa.max() - pa.min()) / resolution32(
                    "cos", float(np.median(va)))
                tb = (pb.max() - pb.min()) / resolution32(
                    "chord", float(np.median(vb)))
                gain_true.append(tb / ta if ta > 0 else float("nan"))
                eps.append(r["ep"])
            out["cells"][f"GAIN__{conv}__{gname}"] = {
                "n_distinct_f32_ratio": _q(gain_nd),
                "n_distinct_f32_ratio_cluster_ci": _cluster_ci(gain_nd, eps),
                "steps_true_ratio": _q(gain_true),
                "steps_true_ratio_cluster_ci": _cluster_ci(gain_true, eps),
                "note": "paired per window; the clusters are EPISODES",
            }

    # X3 the deliberate regression, against the cos arm it must not beat
    nd_cos, nd_sqrt, nd_ch = [], [], []
    for r in rows:
        nd_cos.append(quantum(np.asarray(r["cells"]["cos__A__full"]
                                         ["kmarg_f32"],
                                         dtype=np.float32))["n_distinct_f32"])
        nd_sqrt.append(quantum(np.asarray(r["X3_sqrt_of_1mcos_f32_kmarg"],
                                          dtype=np.float32))
                       ["n_distinct_f32"])
        nd_ch.append(quantum(np.asarray(r["cells"]["chord__A__full"]
                                        ["kmarg_f32"],
                                        dtype=np.float32))["n_distinct_f32"])
    out["controls"]["X3_deliberate_regression"] = {
        "n_distinct_cos": _q(nd_cos), "n_distinct_sqrt_of_1mcos_f32":
            _q(nd_sqrt), "n_distinct_chord": _q(nd_ch),
        "n_windows_where_sqrt_beats_cos": int(sum(
            1 for x, y in zip(nd_sqrt, nd_cos) if x > y)),
        "pass": bool(all(x <= y for x, y in zip(nd_sqrt, nd_cos))
                     and all(z > x for z, x in zip(nd_ch, nd_sqrt))),
        "note": "sqrt(2*(1-cos)) in f32 is the SAME ALGEBRA as the chord and "
                "must recover NOTHING; if it does, the panel cannot see the "
                "defect it exists to catch",
    }

    # ---------------- L4: the weights, derived ------------------------- #
    for metric in METRICS:
        for _u, conv in UNITS:
            for gname in GOALS:
                key = f"{metric}__{conv}__{gname}"
                tip, comm, konly, wins, adv, ded, eps = [], [], [], [], [], [], []
                for r in curved:
                    c = r["cells"][key]
                    nk = r["named"]
                    if "goal_canonical" not in nk or "cv" not in nk:
                        continue
                    gi, ci = nk.index("goal_canonical"), nk.index("cv")
                    a64 = c["named_f64"][ci] - c["named_f64"][gi]
                    k2 = r["kappa2_named"][gi] - r["kappa2_named"][ci]
                    j2 = r["jerk2_named"][gi] - r["jerk2_named"][ci]
                    charge = wk * k2 + wj * j2
                    adv.append(a64)
                    wins.append(1.0 if a64 > charge else 0.0)
                    base = k2 + j2
                    tip.append(wk * (a64 / base) if base > 0 else float("nan"))
                    konly.append((a64 - wj * j2) / k2 if k2 > 0
                                 else float("nan"))
                    v64 = np.asarray(c["kmarg_f64"], float)
                    comm.append(float(v64.max() - v64.min()) / kmax2)
                    q = quantum(np.asarray(c["kmarg_f32"], dtype=np.float32))
                    qq = q["q_realised_f32"] or q["q_ulp_at_median"]
                    w_tip = wk * (a64 / base) if base > 0 else float("nan")
                    ded.append(1.0 if (np.isfinite(w_tip)
                                       and w_tip * kmax2 < qq) else 0.0)
                    eps.append(r["ep"])
                out["weights"][key] = {
                    "n_windows": len(adv), "n_episodes": len(set(eps)),
                    "n_distinct_tokens": len({(r["lat"], r["lon"])
                                              for r in curved}),
                    "goal_advantage_f64": _q(adv),
                    "n_goal_advantage_positive": int(sum(1 for x in adv
                                                         if x > 0)),
                    "n_turn_total_cost_wins_at_shipped_w": int(sum(wins)),
                    "w_kappa_tip_scale_both_weights": _q(tip),
                    "w_kappa_commensurable_ptp_over_kmax2": _q(comm),
                    "w_kappa_only_holding_w_jerk": _q(konly),
                    "n_tipping_weight_is_a_deletion": int(sum(ded)),
                    "deletion_rule": ("w is a DELETION iff w*kappa_max^2 < the "
                                      "metric's own realised float32 quantum: "
                                      "the whole penalty over the whole box is "
                                      "then smaller than one representable "
                                      "step of the term it trades against"),
                }

    # ---------------- ⭐ THE ARGMIN: does the cost want a turn? ---------- #
    # The banked fact is that the shipped argmin is (a=0, kappa=0) on 139/140
    # (`D-REFAV1-COST-SURFACE`). A repair that never moves that argmin has not
    # repaired anything a planner can act on, whatever the win counts say.
    # Constant candidates -> jerk == 0, so total(k) = goal(k) + w*k^2 exactly.
    out["argmin"] = {"note": (
        "the kappa-marginal argmin of the TOTAL cost at a == 0. The SHIPPED-w "
        "column is the DECISION number; the derived-w column is a SIZING "
        "number and is labelled as such — at a weight chosen from these same "
        "windows roughly half of them move BY CONSTRUCTION. "
        "`sign_matches_human` is scored ONLY on the geometric turn stratum "
        "and uses the SIGNED ground-truth heading change, which is ground "
        "truth and only ever describes a window."), "by_cell": {}}
    for metric in METRICS:
        for _u, conv in UNITS:
            for gname in GOALS:
                key = f"{metric}__{conv}__{gname}"
                w_der = (out["weights"].get(key, {})
                         .get("w_kappa_commensurable_ptp_over_kmax2") or {})
                w_der = float(w_der.get("median", float("nan")))
                off_ship = off_der = 0
                sgn_ship = sgn_der = sgn_n = 0
                aks, akd = [], []
                for r in rows:
                    v = np.asarray(r["cells"][key]["kmarg_f64"], float)
                    k2 = np.asarray(r["kappa2_grid"], float)
                    a_ship = float(k_axis[int(np.argmin(v + wk * k2))])
                    aks.append(a_ship)
                    off_ship += int(a_ship != 0.0)
                    if np.isfinite(w_der):
                        a_der = float(k_axis[int(np.argmin(v + w_der * k2))])
                        akd.append(a_der)
                        off_der += int(a_der != 0.0)
                    g = r.get("gt_turn_deg_signed")
                    if g is not None and abs(g) >= gt_turn_min:
                        sgn_n += 1
                        sgn_ship += int(np.sign(a_ship) == np.sign(g)
                                        and a_ship != 0.0)
                        if akd:
                            sgn_der += int(np.sign(akd[-1]) == np.sign(g)
                                           and akd[-1] != 0.0)
                out["argmin"]["by_cell"][key] = {
                    "n_windows": len(rows),
                    "w_shipped": wk,
                    "n_argmin_off_zero_at_shipped_w": off_ship,
                    "argmin_kappa_at_shipped_w": _q(aks),
                    "w_derived_commensurable_median": w_der,
                    "n_argmin_off_zero_at_derived_w": off_der,
                    "n_gt_turners": sgn_n,
                    "n_sign_matches_human_at_shipped_w": sgn_ship,
                    "n_sign_matches_human_at_derived_w": sgn_der,
                }

    # the SAME win counts on the geometric turn stratum (a DIFFERENT n)
    out["strata"] = {"note": ("the curved-goal stratum is what the DECODER "
                              "asked for; the gt_turn stratum is where the CAR "
                              "turns. DIFFERENT denominators; never mixed."),
                     "by_cell": {}}
    for metric in METRICS:
        for _u, conv in UNITS:
            for gname in GOALS:
                key = f"{metric}__{conv}__{gname}"
                w2, a2 = [], []
                for r in turners:
                    nk = r["named"]
                    if "goal_canonical" not in nk or "cv" not in nk:
                        continue
                    gi, ci = nk.index("goal_canonical"), nk.index("cv")
                    c = r["cells"][key]
                    a64 = c["named_f64"][ci] - c["named_f64"][gi]
                    charge = wk * (r["kappa2_named"][gi]
                                   - r["kappa2_named"][ci]) + \
                        wj * (r["jerk2_named"][gi] - r["jerk2_named"][ci])
                    a2.append(a64)
                    w2.append(1.0 if a64 > charge else 0.0)
                out["strata"]["by_cell"][key] = {
                    "n_windows": len(w2),
                    "n_episodes": len({r["ep"] for r in turners}),
                    "n_turn_total_cost_wins_at_shipped_w": int(sum(w2)),
                    "goal_advantage_f64": _q(a2)}

    g = d["gates"]
    out["controls"]["X2a_c1_shipped_cost"] = {
        "n": len(g["X2a_c1_shipped_cost"]),
        "worst_rel_err": (max(x["rel_err"] for x in g["X2a_c1_shipped_cost"])
                          if g["X2a_c1_shipped_cost"] else None),
        "pass": bool(g["X2a_c1_shipped_cost"]
                     and all(x["pass"] for x in g["X2a_c1_shipped_cost"]))}
    out["controls"]["X2a_bit_identity"] = {
        "n": len(g["X2a_bit_identity"]),
        "all_bit_identical": bool(g["X2a_bit_identity"]
                                  and all(x["bit_identical"]
                                          for x in g["X2a_bit_identity"])),
        "max_abs_diff": (max(x["max_abs_diff"] for x in g["X2a_bit_identity"])
                         if g["X2a_bit_identity"] else None)}
    out["controls"]["X2b_banked_winner"] = {
        "n": len(g["X2b_banked_winner"]), "gate_m": BANKED_GATE_M,
        "max_abs_m": (max(x["max_abs_m"] for x in g["X2b_banked_winner"])
                      if g["X2b_banked_winner"] else None),
        "pass": bool(g["X2b_banked_winner"]
                     and all(x["pass"] for x in g["X2b_banked_winner"])),
        "detail": g["X2b_banked_winner"]}
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--config")
    ap.add_argument("--cache")
    ap.add_argument("--episodes")
    ap.add_argument("--labels")
    ap.add_argument("--nav")
    ap.add_argument("--name", required=True)
    ap.add_argument("--arm-tool-dir")
    ap.add_argument("--cost-surface-probe",
                    help="taniteval/tools/cost_surface_probe.py — IMPORTED, "
                         "never modified")
    ap.add_argument("--banked-dump", default=None,
                    help="t1_dump dir for the < 3.8e-6 m banked-winner gate")
    ap.add_argument("--banked-cost-surface", default=None,
                    help="cost_surface_<arm>.json — the source of gt_turn_deg")
    ap.add_argument("--episodes-n", type=int, default=20)
    ap.add_argument("--window-stride", type=int, default=10)
    ap.add_argument("--horizon-k", type=int, default=10)
    ap.add_argument("--n-kappa", type=int, default=21)
    ap.add_argument("--gate-n", type=int, default=5)
    ap.add_argument("--gt-turn-min", type=float, default=5.0)
    ap.add_argument("--wheelbase", type=float, default=2.9)
    ap.add_argument("--lru", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw-out", default=None)
    ap.add_argument("--from-rows", default=None,
                    help="⭐ RE-ANALYSE banked rows with ZERO GPU — the "
                         "compute is already paid for and an analysis fix "
                         "must never re-roll it")
    a = ap.parse_args()

    if a.from_rows:
        with open(a.from_rows, encoding="utf-8") as f:
            d = json.load(f)
        print(f"[cost-repair] re-analysing {len(d['rows'])} banked rows — "
              f"no GPU", flush=True)
    else:
        for req in ("ckpt", "cache", "episodes", "arm_tool_dir",
                    "cost_surface_probe"):
            if getattr(a, req) is None:
                raise SystemExit(f"--{req.replace('_', '-')} is required "
                                 f"unless --from-rows is given")
        d = collect(a)
    res = analyse(d, a.banked_cost_surface, a.gt_turn_min)
    res.update({"tool": "cost_repair_probe.py", "tier": "T0",
                "tier_note": "a COST/WM diagnostic anchored at t0; every win "
                             "count is a property of the COST, never of the "
                             "car. No driving claim without the T1 adapter.",
                "evidence_class": "MEASURED", "arm": a.name,
                "meta": d["meta"], "gates_raw": d["gates"],
                "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime())})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    if a.raw_out:
        with open(a.raw_out, "w", encoding="utf-8") as f:
            json.dump(d, f)
    print(json.dumps({"populations": res["populations"],
                      "controls": res["controls"]}, indent=1)[:4000])
    for k in ("cos__A__full", "chord__A__full", "chord__B__plan"):
        print(f"--- cell {k}\n" + json.dumps(res["cells"][k], indent=1))
        print(f"--- weights {k}\n" + json.dumps(res["weights"].get(k),
                                                indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
