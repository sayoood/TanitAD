#!/usr/bin/env python3
"""refav1_kin_contract_probe.py — WHY does the RECORDED (a, kappa) miss the human laterally?

⛔ WHAT THIS ANSWERS. The open-loop arm ``ol`` of ``refav1_arm.py`` replays the CORPUS'S OWN
recorded actions through the adapter's unicycle from the measured ``v0``. It is the
KINEMATIC-CONTRACT control: if the contract holds, that replay reproduces the human's path.
MEASURED 2026-09-03 on the banked step-1,000 T1 dumps it does NOT — mean ``|y_ol - y_gt|`` is
**0.716 m** on the 72 windows where the human's own path curves, WORSE than a straight line's
0.496 m there (`…/Architecture & Inference/Research/2026-09-03-refav1-step1000-read/RESULT.md` §7).
Every LATERAL row of every refav1 T1 read is suspended until that is resolved. This probe resolves it.

⭐ HOW. Two stages, both 0 GPU, both CPU-only:

  A. REPRODUCE — off the banked ``ep*.npz`` alone: per arm, the straight-plan / constant-speed
     fractions and the lateral+longitudinal error split by whether the HUMAN's path curves.
     This must reproduce the four amendment numbers or the rest of the run is void.

  B. REPLAY under ONE-VARIABLE HYPOTHESES — re-mint the actions from the v2ep episodes through
     the EXACT path the ``ol`` arm uses (``refav1_loader._kin_actions`` ->
     ``refav1_arm.paths_from_controls`` -> ``refa_v1_plan.unicycle_paths`` ->
     ``kinematic.rollout_unicycle``), change exactly one thing, and score BOTH channels: the
     lateral error on the curved windows AND the longitudinal error, because a repair that
     fixes ``y`` by breaking ``x`` is not a repair. Hypotheses and their EXPECTED readings are
     pre-registered in the package SPEC.md **before** this tool is run.

⛔ THE CONTROLS THAT MUST READ A KNOWN VALUE (the CLAUDE.md probe rule; without them a panel
   like this manufactures results):
     * ``ref_ol``      — my replay of the SHIPPED path must equal the banked ``ol`` to < 1e-3 m.
                        The tool REFUSES to report anything else if it does not.
     * ``ref_gt``      — my recomputed GT must equal the banked ``g`` to < 1e-3 m.
     * ``ref_floor``   — integrating from the TRUE pose yaw + TRUE pose speed must read ~0 on the
                        human-straight windows. If it does not, the geometry is wrong before any
                        hypothesis is asked and no kappa hypothesis means anything.
     * ``ref_line``    — the constant-velocity straight line must reproduce 0.4960 / 0.0350.

ESTIMATOR: full-set pooled means over windows; intervals = episode-cluster bootstrap
(``taniteval.ci``), paired against the shipped arm on the same windows. ``overlapping_holdout_se``
is never used. Every block prints its n.

USAGE (dev box, off-Drive mirror, PYTHONPATH=<wt>/stack;<wt>/colab;<wt>/taniteval):

    python taniteval/tools/refav1_kin_contract_probe.py \
        --dump-dir C:/Users/Admin/refav1_eval_slice/t1_dump \
        --episodes C:/Users/Admin/refav1_eval_slice/eps \
        --cache    C:/Users/Admin/refav1_eval_slice/fp8 \
        --out      raw/kin_contract.json --csv-dir raw/
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import math
import os
import sys

import numpy as np

#: ``max_k |y_gt[k]|`` at or above this is a CURVED window (the amendment's split).
CURVED_M = 0.3
#: ``physicalai.WHEELBASE`` — the ``const2p9`` build regime's constant. Imported at run time
#: from the module that MINTED the channel, never re-typed (a second constant is a second
#: convention, and two conventions is a retraction).
_WHEELBASE_FALLBACK = 2.9


# --------------------------------------------------------------------------- #
# imports of the code under test — by FILE PATH, so the probe can never score a
# different copy of the arm than the one on disk beside it                      #
# --------------------------------------------------------------------------- #
def _load_arm(path: str | None = None):
    p = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "refav1_arm.py")
    if not os.path.exists(p):
        raise SystemExit(f"[kin-probe] cannot find refav1_arm.py at {p}")
    spec = importlib.util.spec_from_file_location("_refav1_arm_probe", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_refav1_arm_probe"] = mod
    spec.loader.exec_module(mod)
    return mod, p


def _wheelbase() -> float:
    try:
        from tanitad.data.physicalai import WHEELBASE
        return float(WHEELBASE)
    except Exception:
        return _WHEELBASE_FALLBACK


# --------------------------------------------------------------------------- #
# the integrator, re-expressed in numpy so the ORDER can be a variable          #
# --------------------------------------------------------------------------- #
def integrate(a: np.ndarray, kap: np.ndarray, v0: float, dt: float,
              order: str = "shipped") -> np.ndarray:
    """``[K]`` accel + ``[K]`` curvature -> ``[K, 2]`` ego-frame (x, y).

    ``order="shipped"`` is ``kinematic.rollout_unicycle`` line for line
    (position on the START-of-step yaw, yaw after, v last, v clamped at 0) — and the
    tool ASSERTS that it reproduces the real integrator before any hypothesis is scored.
    ``yaw_first`` applies the yaw update before the position step; ``midpoint`` advances the
    position on the half-updated yaw. Those two ARE hypothesis (d).
    """
    x = y = yaw = 0.0
    v = float(v0)
    out = np.zeros((len(a), 2), dtype=np.float64)
    for k in range(len(a)):
        if order == "shipped":
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
            yaw += v * float(kap[k]) * dt
        elif order == "yaw_first":
            yaw += v * float(kap[k]) * dt
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
        elif order == "midpoint":
            ym = yaw + 0.5 * v * float(kap[k]) * dt
            x += v * math.cos(ym) * dt
            y += v * math.sin(ym) * dt
            yaw += v * float(kap[k]) * dt
        else:
            raise ValueError(f"unknown integration order {order!r}")
        v = max(0.0, v + float(a[k]) * dt)
        out[k] = (x, y)
    return out


def integrate_true_yaw(v_true: np.ndarray, yaw_rel: np.ndarray, dt: float) -> np.ndarray:
    """THE FLOOR: positions from the TRUE pose speed and TRUE pose heading, no action channel.

    ``v_true[k]`` / ``yaw_rel[k]`` are the pose speed and the pose yaw RELATIVE to t0 at the
    START of step k. Its residual against GT is discretisation + side-slip: the best any
    ``(a, kappa)`` contract could ever achieve on this corpus.
    """
    x = y = 0.0
    out = np.zeros((len(v_true), 2), dtype=np.float64)
    for k in range(len(v_true)):
        x += v_true[k] * math.cos(yaw_rel[k]) * dt
        y += v_true[k] * math.sin(yaw_rel[k]) * dt
        out[k] = (x, y)
    return out


def _wrap(a):
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi


# --------------------------------------------------------------------------- #
# stage A — reproduce the amendment off the banked dumps                        #
# --------------------------------------------------------------------------- #
def stage_a(dump_dir: str, arms: list[str]) -> tuple[list[dict], dict]:
    files = sorted(glob.glob(os.path.join(dump_dir, "ep*.npz")))
    if not files:
        raise SystemExit(f"[kin-probe] no ep*.npz under {dump_dir}")
    rows: list[dict] = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        g = z["g"]
        have = [a for a in arms if a in z.files]
        for i in range(g.shape[0]):
            exc = float(np.max(np.abs(g[i, :, 1])))
            r = {"file": os.path.basename(f), "eid": int(z["eid"][0]), "w": i,
                 "t": int(z["ws"][i]), "v0": float(z["v0"][i]),
                 "gt_excursion_m": exc, "curved": int(exc >= CURVED_M),
                 "lat_line": float(np.mean(np.abs(g[i, :, 1]))),
                 "lon_line": float("nan")}
            for a in have:
                tr = z[a]
                r[f"lat_{a}"] = float(np.mean(np.abs(tr[i, :, 1] - g[i, :, 1])))
                r[f"lon_{a}"] = float(np.mean(np.abs(tr[i, :, 0] - g[i, :, 0])))
                r[f"straight_{a}"] = int(np.max(np.abs(tr[i, :, 1])) < 1e-6)
                steps = np.linalg.norm(
                    np.diff(np.vstack([[0.0, 0.0], tr[i]]), axis=0), axis=1)
                r[f"const_speed_{a}"] = int(np.ptp(steps) < 1e-4)
            rows.append(r)
    n = len(rows)
    cur = [r for r in rows if r["curved"]]
    sti = [r for r in rows if not r["curved"]]
    rep = {"n_windows": n, "n_curved": len(cur), "n_straight": len(sti),
           "curved_rule": f"max_k |y_gt[k]| >= {CURVED_M} m", "arms": {}}
    for a in arms:
        if f"lat_{a}" not in rows[0]:
            continue
        rep["arms"][a] = {
            "straight_plans": sum(r[f"straight_{a}"] for r in rows),
            "const_speed_plans": sum(r[f"const_speed_{a}"] for r in rows),
            "lat_curved": float(np.mean([r[f"lat_{a}"] for r in cur])),
            "lat_straight": float(np.mean([r[f"lat_{a}"] for r in sti])),
            "lon_all": float(np.mean([r[f"lon_{a}"] for r in rows])),
            "lon_curved": float(np.mean([r[f"lon_{a}"] for r in cur]))}
    rep["arms"]["_line_y0"] = {
        "straight_plans": n, "const_speed_plans": n,
        "lat_curved": float(np.mean([r["lat_line"] for r in cur])),
        "lat_straight": float(np.mean([r["lat_line"] for r in sti])),
        "lon_all": None, "lon_curved": None,
        "note": "y == 0 exactly: the human's own lateral excursion"}
    return rows, rep


# --------------------------------------------------------------------------- #
# stage B — the one-variable hypothesis panel                                   #
# --------------------------------------------------------------------------- #
#: (id, description) — the pre-registered panel. Order is the SPEC's order.
HYPOTHESES = [
    ("ref_ol", "REF: the shipped path, replayed here (must equal the banked ol)"),
    ("ref_line", "REF: constant velocity, a = 0, kappa = 0 at the measured v0"),
    ("ref_floor", "REF: positions from the TRUE pose yaw + TRUE pose speed (no action channel)"),
    ("ref_kapose", "REF: kappa = dyaw_pose / (v * 0.2) on the cache grid, a as shipped"),
    ("ref_floor_mid", "REF: the same floor under the MIDPOINT quadrature -- the shipped Euler "
                      "order is itself part of ref_floor, so these two BRACKET the geometric floor"),
    ("h_a_sign", "(a) kappa -> -kappa"),
    ("h_b1_shift_m1", "(b) kappa = kap[2j-2] (the 'closes at t0' half-step)"),
    ("h_b2_shift_p1", "(b) kappa = kap[2j+2]"),
    ("h_b3_midframe", "(b) kappa = kap[2j+1] (the odd 10 Hz frame the cache skips)"),
    ("h_c0_half", "(c) kappa -> kap[2j] / 2 (per-frame vs per-second on a 2-frame grid)"),
    ("h_c1_wb", "(c) kappa = tan(kap[2j]) / L, L = physicalai.WHEELBASE: the channel is a STEER angle"),
    ("h_c2_wbfit", "(c) kappa = tan(kap[2j]) / L_hat_ep, L_hat fit per episode -- ORACLE CEILING"),
    ("h_d1_yawfirst", "(d) yaw updated BEFORE the position step"),
    ("h_d2_midpoint", "(d) position advanced on yaw + 0.5*v*kappa*dt"),
    ("h_c1_d2", "COMBINATION (labelled): h_c1_wb + h_d2_midpoint"),
]


def _episode_L(v: np.ndarray, yaw: np.ndarray, steer: np.ndarray,
               v_min: float = 3.0) -> tuple[float, float, int]:
    """Least-squares ``tan(steer) ~ L * kappa_pose`` over one episode at 10 Hz.

    ⚠️ ORACLE. This is fit on the very episodes the panel scores, so ``h_c2_wbfit`` is a
    CEILING, never a shippable repair. Reported with its r and n so it cannot be mistaken.
    """
    dyaw = _wrap(np.diff(yaw))
    vv = v[:-1]
    m = vv > v_min
    if m.sum() < 10:
        return float("nan"), float("nan"), int(m.sum())
    kap = dyaw[m] / (vv[m] * 0.1)
    tst = np.tan(steer[:-1][m])
    L = float((tst @ kap) / max(float(kap @ kap), 1e-12))
    r = float(np.corrcoef(tst, kap)[0, 1]) if kap.std() > 0 else float("nan")
    return L, r, int(m.sum())


def stage_b(dump_dir: str, episode_dir: str, cache_dir: str, k: int,
            arm_mod, atol: float) -> tuple[list[dict], dict, dict]:
    import torch
    from tanitad.data.refav1_loader import RefAV1Windows

    files = sorted(glob.glob(os.path.join(dump_dir, "ep*.npz")))
    man = json.load(open(os.path.join(dump_dir, "manifest.json")))
    names = {int(e["file_index"]): e["name"] for e in man["episodes"]}
    dt = float(man["grid"]["dt_s"])
    cfgd = man["model"]["cfg"]
    L_const = _wheelbase()

    # the REAL loader object, so `_kin_actions` is the code under test and not a copy
    ld = RefAV1Windows(cache_dir, episode_dir,
                       op_window=int(cfgd["op_window"]), op_steps=int(cfgd["op_steps"]),
                       op_dt=float(cfgd["op_dt"]), str_dt=float(cfgd["str_dt"]),
                       str_ext_steps=int(cfgd["str_steps"]),
                       episodes=sorted(names.values()), lru=1)

    rows: list[dict] = []
    Lfit: dict[str, dict] = {}
    checks = {"ol_max_abs_diff_m": 0.0, "gt_max_abs_diff_m": 0.0,
              "n_windows": 0, "n_episodes": 0}
    for fi, f in enumerate(files):
        z = np.load(f, allow_pickle=True)
        nm = names[fi]
        o = torch.load(os.path.join(episode_dir, f"{nm}.v2ep.pt"),
                       map_location="cpu", weights_only=False)
        poses = o["poses"].float()
        v_t = poses[:, 3]
        kap_t = o["actions"][:, 0].float()          # refav1_loader.py:252 — the SAME read
        v = v_t.numpy().astype(np.float64)
        yawp = poses[:, 2].numpy().astype(np.float64)
        steer = kap_t.numpy().astype(np.float64)
        T = v.shape[0]
        L_hat, r_hat, n_hat = _episode_L(v, yawp, steer)
        Lfit[nm] = {"L_ls": L_hat, "r": r_hat, "n": n_hat}
        checks["n_episodes"] += 1

        g_bank = z["g"]
        for i in range(g_bank.shape[0]):
            t = int(z["ws"][i])
            v0 = float(z["v0"][i])
            # ---- GT, recomputed through the arm's own path, checked against the bank
            g = arm_mod.gt_waypoints(poses, t, k)[0].numpy().astype(np.float64)
            dg = float(np.abs(g - g_bank[i]).max())
            checks["gt_max_abs_diff_m"] = max(checks["gt_max_abs_diff_m"], dg)
            # ---- the SHIPPED actions, through the REAL loader method
            act = ld._kin_actions(v_t, kap_t, t, k).numpy().astype(np.float64)
            a_s, k_s = act[:, 0], act[:, 1]
            # ---- the shipped arm through the REAL integrator, for the harness check
            ol_real = arm_mod.paths_from_controls(
                torch.from_numpy(act).float(), v0, dt, k)[0].numpy().astype(np.float64)
            checks["ol_max_abs_diff_m"] = max(
                checks["ol_max_abs_diff_m"], float(np.abs(ol_real - z["ol"][i]).max()))
            # ---- the numpy integrator must reproduce the real one, or nothing below counts
            mine = integrate(a_s, k_s, v0, dt, "shipped")
            d_np = float(np.abs(mine - ol_real).max())
            if d_np > atol:
                raise SystemExit(
                    f"[kin-probe] numpy integrator != rollout_unicycle by {d_np:.2e} m "
                    f"({nm} t={t}) — the panel would be measuring my re-implementation, "
                    f"refusing")

            # ---- per-hypothesis controls -----------------------------------
            j = np.arange(t, t + k)
            f0 = j * 2
            def _kap_at(fr):
                return steer[np.clip(fr, 0, T - 1)]
            kap_m1 = _kap_at(f0 - 2)
            kap_p1 = _kap_at(f0 + 2)
            kap_mf = _kap_at(f0 + 1)
            # true pose heading / speed at the START of each step, relative to t0
            yaw_rel = _wrap(yawp[np.clip(f0, 0, T - 1)] - yawp[2 * t])
            v_true = v[np.clip(f0, 0, T - 1)]
            yaw_next = _wrap(yawp[np.clip(f0 + 2, 0, T - 1)] - yawp[2 * t])
            kap_pose = _wrap(yaw_next - yaw_rel) / np.maximum(v_true * dt, 1e-6)

            traj = {
                "ref_ol": mine,
                "ref_line": integrate(np.zeros(k), np.zeros(k), v0, dt, "shipped"),
                "ref_floor": integrate_true_yaw(v_true, yaw_rel, dt),
                "ref_floor_mid": integrate_true_yaw(
                    v_true, _wrap(yawp[np.clip(f0 + 1, 0, T - 1)] - yawp[2 * t]), dt),
                "ref_kapose": integrate(a_s, kap_pose, v0, dt, "shipped"),
                "h_a_sign": integrate(a_s, -k_s, v0, dt, "shipped"),
                "h_b1_shift_m1": integrate(a_s, kap_m1, v0, dt, "shipped"),
                "h_b2_shift_p1": integrate(a_s, kap_p1, v0, dt, "shipped"),
                "h_b3_midframe": integrate(a_s, kap_mf, v0, dt, "shipped"),
                "h_c0_half": integrate(a_s, k_s / 2.0, v0, dt, "shipped"),
                "h_c1_wb": integrate(a_s, np.tan(k_s) / L_const, v0, dt, "shipped"),
                "h_c2_wbfit": integrate(a_s, np.tan(k_s) / (L_hat if np.isfinite(L_hat)
                                                            and L_hat > 0.5 else L_const),
                                        v0, dt, "shipped"),
                "h_d1_yawfirst": integrate(a_s, k_s, v0, dt, "yaw_first"),
                "h_d2_midpoint": integrate(a_s, k_s, v0, dt, "midpoint"),
                "h_c1_d2": integrate(a_s, np.tan(k_s) / L_const, v0, dt, "midpoint"),
            }
            exc = float(np.max(np.abs(g[:, 1])))
            r = {"episode": nm, "file": os.path.basename(f), "eid": int(z["eid"][0]),
                 "w": i, "t": t, "v0": v0, "gt_excursion_m": exc,
                 "curved": int(exc >= CURVED_M), "L_hat_ep": L_hat, "r_L_hat": r_hat}
            for hid, tr in traj.items():
                r[f"lat_{hid}"] = float(np.mean(np.abs(tr[:, 1] - g[:, 1])))
                r[f"lon_{hid}"] = float(np.mean(np.abs(tr[:, 0] - g[:, 0])))
            # ---- (e) the GT FRAME, scored against the shipped arm ------------
            r["lat_e1_gt_ysign"] = float(np.mean(np.abs(mine[:, 1] + g[:, 1])))
            d0 = poses[2 * t + 2, :2].numpy() - poses[2 * t, :2].numpy()
            yaw0v = math.atan2(float(d0[1]), float(d0[0]))
            dyaw0 = float(_wrap(np.array([yaw0v - yawp[2 * t]]))[0])
            c, s = math.cos(-dyaw0), math.sin(-dyaw0)
            g_vel = np.stack([g[:, 0] * c - g[:, 1] * s,
                              g[:, 0] * s + g[:, 1] * c], axis=-1)
            r["lat_e2_gt_velframe"] = float(np.mean(np.abs(mine[:, 1] - g_vel[:, 1])))
            r["lon_e2_gt_velframe"] = float(np.mean(np.abs(mine[:, 0] - g_vel[:, 0])))
            r["e2_frame_delta_deg"] = math.degrees(dyaw0)
            # ---- (f) SLIP: pose yaw vs the kappa-integrated yaw --------------
            dyaw_pose = float(_wrap(np.array([yawp[min(2 * (t + k), T - 1)]
                                              - yawp[2 * t]]))[0])
            vk = v0
            yi_ship = yi_rep = yi_half = 0.0
            for kk in range(k):
                yi_ship += vk * k_s[kk] * dt
                yi_rep += vk * (math.tan(k_s[kk]) / L_const) * dt
                yi_half += vk * (k_s[kk] / 2.0) * dt
                vk = max(0.0, vk + a_s[kk] * dt)
            r["dyaw_pose_rad"] = dyaw_pose
            r["dyaw_int_shipped_rad"] = yi_ship
            r["dyaw_int_h_c1_rad"] = yi_rep
            r["dyaw_int_h_c0_rad"] = yi_half
            rows.append(r)
            checks["n_windows"] += 1
    return rows, checks, {"L_fit": Lfit, "L_const": L_const, "dt": dt, "k": k}


# --------------------------------------------------------------------------- #
def _boot(vals, eids, ref=None):
    try:
        from taniteval.ci import (episode_cluster_bootstrap,
                                  paired_episode_cluster_bootstrap)
    except Exception as exc:                                  # pragma: no cover
        return {"mean": float(np.mean(vals)), "ci": f"UNAVAILABLE ({exc})",
                "n_windows": len(vals)}
    out = episode_cluster_bootstrap(np.asarray(vals), list(eids))
    if ref is not None:
        out["paired_vs_shipped"] = paired_episode_cluster_bootstrap(
            np.asarray(vals), np.asarray(ref), list(eids))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump-dir", required=True, help="the banked T1 dump (ep*.npz + manifest)")
    ap.add_argument("--episodes", required=True, help="the *.v2ep.pt directory")
    ap.add_argument("--cache", required=True, help="the stage-1 feature cache (loader init only)")
    ap.add_argument("--arm-file", default=None, help="refav1_arm.py (default: beside this file)")
    ap.add_argument("--k", type=int, default=10, help="horizon steps (default 10 = 2.0 s)")
    ap.add_argument("--atol", type=float, default=1e-4,
                    help="max |replay - banked| accepted for the harness checks [m]")
    ap.add_argument("--out", default=None, help="JSON record")
    ap.add_argument("--csv-dir", default=None, help="directory for the per-window CSVs")
    ap.add_argument("--stage-a-only", action="store_true")
    a = ap.parse_args(argv)

    arms = ["cl", "ha", "ol", "cl_navshuf", "cl_nonav", "cl_oraclegoal", "ha0"]
    rows_a, rep_a = stage_a(a.dump_dir, arms)
    print(f"\n=== STAGE A — the banked dumps ({rep_a['n_windows']} windows, "
          f"{rep_a['n_curved']} curved / {rep_a['n_straight']} straight; "
          f"curved = {rep_a['curved_rule']}) ===")
    print(f"{'arm':16s} {'straight':>10s} {'const-v':>9s} {'LAT curved':>11s} "
          f"{'LAT straight':>13s} {'LON all':>9s}")
    for k_, vv in rep_a["arms"].items():
        lon = "-" if vv["lon_all"] is None else f"{vv['lon_all']:9.4f}"
        print(f"{k_:16s} {str(vv['straight_plans']) + '/' + str(rep_a['n_windows']):>10s} "
              f"{str(vv['const_speed_plans']) + '/' + str(rep_a['n_windows']):>9s} "
              f"{vv['lat_curved']:11.4f} {vv['lat_straight']:13.4f} {lon}")

    rec = {"tool": "taniteval/tools/refav1_kin_contract_probe.py",
           "dump_dir": os.path.abspath(a.dump_dir), "stage_a": rep_a,
           "curved_threshold_m": CURVED_M}
    rows_b: list[dict] = []
    if not a.stage_a_only:
        arm_mod, arm_path = _load_arm(a.arm_file)
        rows_b, checks, prov = stage_b(a.dump_dir, a.episodes, a.cache, a.k,
                                       arm_mod, a.atol)
        print(f"\n=== HARNESS CHECKS (these gate everything below) ===")
        print(f"  replayed ol   vs banked ol : max |d| = {checks['ol_max_abs_diff_m']:.3e} m")
        print(f"  recomputed GT vs banked g  : max |d| = {checks['gt_max_abs_diff_m']:.3e} m")
        bad = [nm for nm, val in (("ol", checks["ol_max_abs_diff_m"]),
                                  ("gt", checks["gt_max_abs_diff_m"])) if val > a.atol]
        if bad:
            raise SystemExit(f"[kin-probe] harness check FAILED for {bad} at atol {a.atol} — "
                             f"refusing to report a hypothesis panel from a replay that does "
                             f"not reproduce the arm it claims to replay")
        print(f"  numpy integrator vs rollout_unicycle : verified per window (< {a.atol:g} m)")
        print(f"  n = {checks['n_windows']} windows over {checks['n_episodes']} episodes")

        cur = [r for r in rows_b if r["curved"]]
        eid_all = [r["eid"] for r in rows_b]
        eid_cur = [r["eid"] for r in cur]
        ship_lat_c = [r["lat_ref_ol"] for r in cur]
        line_c = float(np.mean([r["lat_ref_line"] for r in cur]))
        print(f"\n=== STAGE B — one-variable hypotheses (n curved = {len(cur)}, "
              f"n all = {len(rows_b)}); bar = the straight line's {line_c:.4f} m ===")
        print(f"{'id':16s} {'LATcurved':>10s} {'[95% CI]':>18s} {'LATstraight':>12s} "
              f"{'LONall':>8s} {'vs shipped (paired)':>24s}  verdict")
        panel = {}
        for hid, desc in HYPOTHESES:
            lat_c = [r[f"lat_{hid}"] for r in cur]
            lat_s = [r[f"lat_{hid}"] for r in rows_b if not r["curved"]]
            lon_a = [r[f"lon_{hid}"] for r in rows_b]
            b = _boot(lat_c, eid_cur, ref=ship_lat_c if hid != "ref_ol" else None)
            bl = _boot(lon_a, eid_all)
            mc, ms, ml = float(np.mean(lat_c)), float(np.mean(lat_s)), float(np.mean(lon_a))
            ok = (mc < line_c) and (ml <= 0.26) and (ms <= 0.069 + 1e-9)
            pd = b.get("paired_vs_shipped")
            ps = ("-" if pd is None else
                  f"{pd['delta']:+.3f} [{pd['lo']:+.3f},{pd['hi']:+.3f}]"
                  + ("*" if pd.get("separated") else " "))
            verdict = "RESOLVES" if ok else ("worse" if mc > 0.7156 else "partial")
            if hid.startswith("ref"):
                verdict = "control"
            print(f"{hid:16s} {mc:10.4f} {'[' + format(b['lo'], '.4f') + ',' + format(b['hi'], '.4f') + ']':>18s} "
                  f"{ms:12.4f} {ml:8.4f} {ps:>24s}  {verdict}")
            panel[hid] = {"desc": desc, "lat_curved": mc, "lat_curved_ci": b,
                          "lat_straight": ms, "lon_all": ml, "lon_ci": bl,
                          "n_curved": len(lat_c), "n_straight": len(lat_s),
                          "n_all": len(lon_a), "verdict": verdict,
                          "passes_committed_rule": bool(ok)}
        # (e) frame + (f) slip
        e1 = float(np.mean([r["lat_e1_gt_ysign"] for r in cur]))
        e2 = float(np.mean([r["lat_e2_gt_velframe"] for r in cur]))
        e2l = float(np.mean([r["lon_e2_gt_velframe"] for r in rows_b]))
        fdeg = float(np.mean([abs(r["e2_frame_delta_deg"]) for r in rows_b]))
        print(f"\n(e) FRAME  h_e1 GT y sign flipped: LAT curved {e1:.4f} m  "
              f"(shipped {float(np.mean(ship_lat_c)):.4f})")
        print(f"(e) FRAME  h_e2 GT in the t0 VELOCITY frame: LAT curved {e2:.4f} m, "
              f"LON all {e2l:.4f} m; mean |heading - velocity| angle {fdeg:.3f} deg")
        turn = [r for r in rows_b if abs(r["dyaw_pose_rad"]) > 0.02]
        if turn:
            rs = float(np.median([r["dyaw_int_shipped_rad"] / r["dyaw_pose_rad"]
                                  for r in turn]))
            rr = float(np.median([r["dyaw_int_h_c1_rad"] / r["dyaw_pose_rad"]
                                  for r in turn]))
            rh = float(np.median([r["dyaw_int_h_c0_rad"] / r["dyaw_pose_rad"]
                                  for r in turn]))
            es = math.degrees(float(np.sqrt(np.mean(
                [(r["dyaw_int_shipped_rad"] - r["dyaw_pose_rad"]) ** 2 for r in rows_b]))))
            er = math.degrees(float(np.sqrt(np.mean(
                [(r["dyaw_int_h_c1_rad"] - r["dyaw_pose_rad"]) ** 2 for r in rows_b]))))
            print(f"(f) SLIP   yaw over {a.k} steps, integrated / pose  (n turning = "
                  f"{len(turn)}, |dyaw_pose| > 0.02 rad): shipped x{rs:.3f}, "
                  f"h_c0_half x{rh:.3f}, h_c1_wb x{rr:.3f}")
            print(f"(f) SLIP   yaw RMSE vs the pose: shipped {es:.3f} deg, "
                  f"h_c1_wb {er:.3f} deg  (n = {len(rows_b)})")
            eh = math.degrees(float(np.sqrt(np.mean(
                [(r["dyaw_int_h_c0_rad"] - r["dyaw_pose_rad"]) ** 2 for r in rows_b]))))
            print(f"(f) SLIP   h_c0_half yaw RMSE {eh:.3f} deg — a HALF is not the gain")
            rec["slip"] = {"n_turning": len(turn), "ratio_shipped": rs,
                           "ratio_h_c0": rh, "ratio_h_c1": rr,
                           "yaw_rmse_deg_shipped": es, "yaw_rmse_deg_h_c0": eh,
                           "yaw_rmse_deg_h_c1": er}
        worst = sorted(cur, key=lambda r: -r["lat_h_c1_wb"])[:10]
        print(f"\nh_c1_wb — the 10 largest per-window LATERAL residuals of "
              f"{len(cur)} curved:")
        print(f"  {'episode':10s} {'t':>4s} {'v0':>6s} {'|y_gt|max':>10s} {'LAT':>7s} "
              f"{'LON':>7s} {'floor':>7s}")
        for r in worst:
            print(f"  {r['episode'][:8]:10s} {r['t']:4d} {r['v0']:6.2f} "
                  f"{r['gt_excursion_m']:10.3f} {r['lat_h_c1_wb']:7.3f} "
                  f"{r['lon_h_c1_wb']:7.3f} {r['lat_ref_floor']:7.3f}")
        rec["h_c1_worst_windows"] = [
            {kk: r[kk] for kk in ("episode", "t", "v0", "gt_excursion_m",
                                  "lat_h_c1_wb", "lon_h_c1_wb", "lat_ref_floor",
                                  "lat_ref_ol", "L_hat_ep")} for r in worst]
        Ls = [x["L_ls"] for x in prov["L_fit"].values() if np.isfinite(x["L_ls"])]
        print(f"\nimplied wheelbase L (LS fit, ORACLE): n {len(Ls)} eps, "
              f"min {min(Ls):.3f} max {max(Ls):.3f} median {float(np.median(Ls)):.3f}; "
              f"repair constant physicalai.WHEELBASE = {prov['L_const']}")
        rec.update({"harness_checks": checks, "panel": panel,
                    "frame": {"h_e1_gt_ysign_lat_curved": e1,
                              "h_e2_gt_velframe_lat_curved": e2,
                              "h_e2_gt_velframe_lon_all": e2l,
                              "mean_abs_heading_minus_velocity_deg": fdeg},
                    "wheelbase": prov, "arm_file": arm_path,
                    "families": {"longitudinal": "PRESENT (along-track x)",
                                 "lateral": "PRESENT (cross-track y + yaw agreement)",
                                 "tactical": "REFUSED — a replay of the human's own recorded "
                                             "actions declares no manoeuvre",
                                 "strategic": "REFUSED — and no route"}})

    if a.csv_dir:
        os.makedirs(a.csv_dir, exist_ok=True)
        for nm, rr in (("per_window_dump_arms.csv", rows_a),
                       ("per_window_hypotheses.csv", rows_b)):
            if not rr:
                continue
            p = os.path.join(a.csv_dir, nm)
            keys = list(rr[0].keys())
            with open(p, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(rr)
            print(f"[kin-probe] wrote {p} ({len(rr)} rows)")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1, default=float)
        print(f"[kin-probe] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
