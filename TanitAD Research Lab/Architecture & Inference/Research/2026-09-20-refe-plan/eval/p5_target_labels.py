#!/usr/bin/env python3
"""P5 (eval/PREREG_P5_TRACKER_AWARE_TARGETS.md): tracker-aware and margin-aware targets, held-out label test. 0 GPU.

On D1-navtrain's 1,018 held-out rank-0 targets (11 logs), every reference is an 8-pose NAVSIM plan, executed by NAVSIM's
own LQR + kinematic bicycle (refe/navsim_dac.simulated_states), and scored on its EXECUTED trajectory:
  T      to_navsim(t20), the plain teacher label                         baseline
  Tc20   t20 pre-distorted on its own 0.2 s grid (prereg §2), to_navsim   primary arm (ii)
  Tc8    p4_tracker_comp.compensate(T)                                   descriptive
  Tw     the wrong-sign iteration, last iterate                          negative control
  ident  the iteration with the error forced to 0                        identity control
  H      the logged human future (teacher_lane_check.human_p8)           reference
  Hc     compensate(H)                                                   descriptive

Definitions fixed here, in the commit that lands the prereg, before any number exists:
  * learnability on a 20-pose label q (origin prepended, dt 0.2 s): max |acceleration| = max over the 19 interior
    points of || second difference of xy || / dt^2 (vector: longitudinal AND lateral, so a pre-distortion wiggle
    shows); max |heading rate| = max |diff(unwrap([0, h]))| / dt, in deg/s.
  * executed vs intended: d = exec_xy - intended_xy at the 8 NAVSIM times; along/cross = d projected on the INTENDED
    heading (cos h, sin h) / (-sin h, cos h); max-xy = max ||d||; heading MAE = mean |wrap(h_exec - h_int)| in deg;
    path-length ratio = executed polyline length / intended (origin included), only where intended > 1 m.
  * margin: p4_dac_margin.margin on at_10hz(T) (the plan's own footprint, no tracker), metres, positive inside.

  python eval/run_p4.py --log <log> --script p5_target_labels.py -- --out <json> [--npz <npz>] [--limit N]
         [--controls-only] [--mutate knot|ident|d1|r7|margin|batch] [--interp linear|cubic|pchip]
Each --mutate value breaks exactly one control (C2, C1, C4, C5, C6, C7); with it, ZZP5_CONTROLS must read FAIL.
Prints ZZP5_DONE, or ZZP5_GATE_FAIL with the failing control (no verdict is then read).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG, "code"))
sys.path.insert(0, os.path.join(PKG, "refe"))
sys.path.insert(0, HERE)

HELD = os.path.join(PKG, "raw", "2026-10-04-lane-discipline", "teacher_check_heldout.jsonl")
D1 = os.path.join(PKG, "raw", "2026-10-05-teacher-navtest", "d1_navtrain.json")
BANK = "E:/Projects/TanitAD/data/refe_navtrain10"
TRAINVAL = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/trainval"
ITERS = 3
DT = 0.2
TK = np.arange(9) / 2.0                     # knots: t = 0, 0.5, ..., 4.0 s (0 = the origin, error 0)
TQ = np.arange(1, 21) / 5.0                 # queries: t = 0.2, ..., 4.0 s (the label grid)
KNOT_Q, KNOT_E8 = [4, 9, 14, 19], [1, 3, 5, 7]
REFS = ["T", "Tc8", "Tc20", "Tw", "H", "Hc", "ident"]
MARGIN_M = 0.6
SWEEP = [0.3, 0.45, 0.6, 0.75]
N_BOOT, SEED = 10_000, 20260927


def human_p8(sc, step, ego):
    """verbatim refe/teacher_lane_check.human_p8 (inlined: that module's top level parses a CLI)"""
    from nuplan.common.geometry.convert import absolute_to_relative_poses
    fut = list(sc.get_ego_future_trajectory(iteration=step, time_horizon=4.0, num_samples=8))
    rel = absolute_to_relative_poses([ego.rear_axle] + [s.rear_axle for s in fut])[1:]
    return np.array([[p.x, p.y, p.heading] for p in rel], np.float64)


def ego_frame(sim, ego):
    """[P, 41, S] world-frame rear-axle states -> the 8 executed NAVSIM-time poses, ego frame [P, 8, 3] (= tracked8)."""
    from p4_tracker_comp import wrap
    s = sim[:, 5::5][:, :8, :3]
    ra = ego.rear_axle
    c, sn = math.cos(-ra.heading), math.sin(-ra.heading)
    dx, dy = s[..., 0] - ra.x, s[..., 1] - ra.y
    return np.stack([dx * c - dy * sn, dx * sn + dy * c, wrap(s[..., 2] - ra.heading)], -1)


def interp20(tk, e8z, mode):
    """Spread the knot error [0, e8] onto the 0.2 s label grid. `linear` is prereg §2; `cubic` / `pchip` are the P5-S
    amendment (prereg §9, 2026-10-11)."""
    if mode == "linear":
        return np.stack([np.interp(TQ, tk, e8z[:, c]) for c in range(3)], -1)
    from scipy.interpolate import CubicSpline, PchipInterpolator
    if mode == "cubic":
        return CubicSpline(tk, e8z, axis=0, bc_type="natural")(TQ)
    return PchipInterpolator(tk, e8z, axis=0)(TQ)


def iterate20(ND, t20, T, ego, ctl, mutate=None, interp="linear"):
    """prereg §2: the 0.2 s-grid iteration, batched as [Tc20 (+), Tw (-), ident (0)]. Returns the three 20-pose labels
    and the per-k max-xy deviation traces. `interp` selects the P5-S interpolant; `linear` is the landed P5 run."""
    from p4_tracker_comp import tracked8, wrap
    r = np.stack([t20.copy(), t20.copy(), t20.copy()])
    sign = np.array([1.0, -1.0, 1e-9 if mutate == "ident" else 0.0])
    tk = TK + (0.05 if mutate == "knot" else 0.0)
    devs = [[], [], []]
    best, best_dev = t20.copy(), None
    for k in range(ITERS + 1):
        s = tracked8(ND, np.stack([ND.to_navsim(x) for x in r]), ego)               # [3, 8, 3]
        for j in range(3):
            devs[j].append(float(np.linalg.norm(s[j, :, :2] - T[:, :2], axis=1).max()))
        if best_dev is None or devs[0][-1] < best_dev:
            best, best_dev = r[0].copy(), devs[0][-1]
        if k == ITERS:
            break
        for j in range(3):
            e8 = np.concatenate([T[:, :2] - s[j, :, :2], wrap(T[:, 2] - s[j, :, 2])[:, None]], -1)
            e8z = np.concatenate([np.zeros((1, 3)), e8], 0)
            e20 = interp20(tk, e8z, interp)
            if interp == "linear":
                if not np.array_equal(e20[KNOT_Q], e8[KNOT_E8]):
                    ctl["knot_exact"] = False
            else:                                                   # C2': a spline is exact only to rounding
                dev = float(np.abs(e20[KNOT_Q] - e8[KNOT_E8]).max())
                ctl["knot_maxdev"] = max(ctl.get("knot_maxdev", 0.0), dev)
                if not dev <= 1e-9:
                    ctl["knot_exact"] = False
            ctl["knot_checks"] += 1
            r[j] = r[j] + sign[j] * e20
    return best, r[1].copy(), r[2].copy(), devs


def exec_vs_intended(ex, it):
    d = ex[:, :2] - it[:, :2]
    ch, sh = np.cos(it[:, 2]), np.sin(it[:, 2])
    along, cross = d[:, 0] * ch + d[:, 1] * sh, -d[:, 0] * sh + d[:, 1] * ch
    from p4_tracker_comp import wrap

    def plen(p):
        q = np.concatenate([np.zeros((1, 2)), p[:, :2]], 0)
        return float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
    li = plen(it)
    return {"max_xy": float(np.linalg.norm(d, axis=1).max()), "along_4s": float(along[-1]),
            "mean_abs_along": float(np.abs(along).mean()), "mean_abs_cross": float(np.abs(cross).mean()),
            "heading_mae_deg": float(np.degrees(np.abs(wrap(ex[:, 2] - it[:, 2]))).mean()),
            "path_len_ratio": (plen(ex) / li) if li > 1.0 else None}


def learnability(q20):
    q = np.concatenate([np.zeros((1, 3)), q20], 0)
    acc = np.linalg.norm(np.diff(q[:, :2], n=2, axis=0), axis=1) / DT ** 2
    hr = np.abs(np.diff(np.unwrap(q[:, 2]))) / DT
    return float(acc.max()), float(np.degrees(hr.max()))


def boot_weights(logs, n_boot=N_BOOT, seed=SEED):
    """[B, n_logs] multinomial cluster counts; row weight = count of its log."""
    u = sorted(set(logs))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(n_boot, len(u)))
    W = np.stack([np.bincount(i, minlength=len(u)) for i in idx]).astype(np.float64)
    li = np.array([u.index(g) for g in logs])
    return W, li


def paired_ci(a, b, W, li, rate):
    """b - a, paired, log-cluster bootstrap. rate: ratio estimator; else difference of cluster-weighted means."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    w = W[:, li]                                                            # [B, n]
    point = float(b.mean() - a.mean())
    den = w.sum(1)
    bs = (w @ b - w @ a) / den
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"delta": point, "lo": float(lo), "hi": float(hi), "kind": "ratio" if rate else "weighted_mean"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--npz", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--controls-only", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--mutate", choices=("knot", "ident", "d1", "r7", "margin", "batch"), default=None,
                    help="deliberate-regression arm: breaks ONE control, which must then FAIL (controls-only)")
    ap.add_argument("--interp", choices=("linear", "cubic", "pchip"), default="linear",
                    help="P5-S (prereg §9): interpolant of the knot error; linear reproduces the landed P5 run")
    a = ap.parse_args()
    if a.mutate and not a.controls_only:
        ap.error("--mutate is a control arm: it requires --controls-only")
    import navsim_dac as ND
    import navsim_lane as NL
    import navtrain_scenarios as NS
    from p4_tracker_comp import compensate
    from p4_dac_margin import margin
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from navsim_pdm.pdm_array_representation import state_array_to_coords_array
    from navsim_pdm.pdm_enums import BBCoordsIndex
    import glob

    t0 = time.time()
    d1 = {(r["log"], r["token"]): r for r in json.load(open(D1, encoding="utf-8"))["rows"]}
    held = {}
    for line in open(HELD, encoding="utf-8"):
        r = json.loads(line)
        if int(r.get("rank", 0)) == 0 and int(r.get("step", 0)) == 0 and "teacher" in r:
            held[(r["log"], r["token"])] = r
    bank = {}
    for fp in glob.glob(f"{BANK}/**/targets_rank0.jsonl", recursive=True):
        for line in open(fp, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = (r.get("log_name"), r.get("token"))
            if k in held and int(r.get("sim_hz", 0)) == 10 and int(r.get("step", 0)) == 0:
                bank[k] = r["traj"]
    keys = sorted(bank)
    if a.limit:
        keys = keys[:a.limit]
    by: dict = {}
    for lg, tok in keys:
        by.setdefault(lg, []).append(tok)
    dbs = NS.index_dbs(db_root=os.path.dirname(TRAINVAL))
    pp = get_pacifica_parameters()

    ctl = {"knot_exact": True, "knot_checks": 0, "ident_label_bit_exact": True, "ident_p8_bit_exact": True,
           "ident_sim8_eq_T": True, "batch_indep_max_m": 0.0, "margin_neg_not_interp8": 0, "d1_sim8_mismatch": 0,
           "d1_missing": 0, "r7_dac_T": 0, "r7_dac_H": 0, "r7_ddc_T": 0, "r7_ddc_H": 0, "r7_lk_T": 0, "r7_lk_H": 0,
           "r7_n": 0}
    rows, ex_all, it_T, it_H, logs_, toks_, v0_, lab20 = [], {k: [] for k in REFS}, [], [], [], [], [], []
    errors = []
    for li_, (lg, toks) in enumerate(sorted(by.items())):
        db = dbs.get(lg, "")
        if not os.path.exists(db):
            errors += [{"log": lg, "token": t, "error": "no local DB"} for t in toks]
            continue
        for sc in NS.build_scenarios_for_log(db, toks):
            tok = sc._initial_lidar_token
            try:
                ego = sc.get_ego_state_at_iteration(0)
                ra = ego.rear_axle
                polys = ND.drivable_polygons(sc.map_api, ego)
                t20 = np.asarray(bank[(lg, tok)], np.float64)
                T = ND.to_navsim(t20)
                tc20, tw20, id20, devs = iterate20(ND, t20, T, ego, ctl, a.mutate, a.interp)
                Tc8, devs8 = compensate(ND, T, ego)
                H = human_p8(sc, 0, ego)
                Hc, devsH = compensate(ND, H, ego)
                p8 = {"T": T, "Tc8": Tc8, "Tc20": ND.to_navsim(tc20), "Tw": ND.to_navsim(tw20), "H": H, "Hc": Hc,
                      "ident": ND.to_navsim(id20)}
                P = np.stack([p8[k] for k in REFS])
                sim = ND.simulated_states(P, ego)
                coords = state_array_to_coords_array(sim, pp)
                ci = [i for i in range(coords.shape[2]) if i != int(BBCoordsIndex.CENTER)]
                dac = ND.violations(polys, coords[:, :, ci])
                comf = ND.comfortable_from_states(sim)
                ids, _diag = NL.corrected_route_ids(sc.map_api, ra, list(sc.get_route_roadblock_ids()))
                ddc, lk = NL.navsim_ddc_and_lane(sim, sc.map_api, ego, ids)
                interp = ND.violations(polys, np.stack([ND.corners_global(ND.at_10hz(p8[k]), ra.x, ra.y, ra.heading)
                                                        for k in REFS]))
                ex = ego_frame(sim, ego)
                alone = ND.simulated_states(T[None], ego)
                ctl["batch_indep_max_m"] = max(ctl["batch_indep_max_m"],
                                               float(np.abs(alone[0, :, :2] - sim[1 if a.mutate == "batch" else 0, :, :2]).max()))
                if not np.array_equal(id20, t20):
                    ctl["ident_label_bit_exact"] = False
                if not np.array_equal(p8["ident"], T):
                    ctl["ident_p8_bit_exact"] = False
                ri = {k: i for i, k in enumerate(REFS)}
                if bool(dac[ri["ident"]]) != bool(dac[ri["T"]]):
                    ctl["ident_sim8_eq_T"] = False
                mT = margin(polys, ND.corners_global(ND.at_10hz(T), ra.x, ra.y, ra.heading))
                mH = margin(polys, ND.corners_global(ND.at_10hz(H), ra.x, ra.y, ra.heading))
                if a.mutate == "margin":
                    mT = -abs(mT) - 1e-3
                if mT < 0 and not bool(interp[ri["T"]]):
                    ctl["margin_neg_not_interp8"] += 1
                if mH < 0 and not bool(interp[ri["H"]]):
                    ctl["margin_neg_not_interp8"] += 1
                d = d1.get((lg, tok))
                if d is None:
                    ctl["d1_missing"] += 1
                elif bool(d["sim8"]) != (bool(dac[ri["T"]]) ^ (a.mutate == "d1" and len(rows) == 0)):
                    ctl["d1_sim8_mismatch"] += 1
                h = held[(lg, tok)]
                ctl["r7_n"] += 1
                for ref, src in (("T", "teacher"), ("H", "human")):
                    flip = a.mutate == "r7" and ref == "T" and len(rows) == 0
                    ctl[f"r7_dac_{ref}"] += int((bool(dac[ri[ref]]) ^ flip) == bool(h[src]["dac_viol"]))
                    ctl[f"r7_ddc_{ref}"] += int(abs(float(ddc[ri[ref]]) - float(h[src]["ddc"])) < 1e-9)
                    ctl[f"r7_lk_{ref}"] += int(abs(float(lk[ri[ref]]) - float(h[src]["lk10"])) < 1e-9)
                v0 = float(np.hypot(ego.dynamic_car_state.rear_axle_velocity_2d.x,
                                    ego.dynamic_car_state.rear_axle_velocity_2d.y))
                rec = {"log": lg, "token": tok, "v0": v0, "d1_class": None if d is None else (
                    "A_teacher_path" if d["raw20"] else "B_subsampling" if d["interp8"] else
                    "C_navsim_tracker" if d["sim8"] else "pass"),
                       "margin_T": mT, "margin_H": mH, "devs_Tc20": devs[0], "devs_Tw": devs[1],
                       "devs_ident": devs[2], "devs_Tc8": devs8, "devs_Hc": devsH,
                       "tc20_minus_tc8_max_m": float(np.abs(p8["Tc20"][:, :2] - Tc8[:, :2]).max()),
                       "learn": {"t20": learnability(t20), "Tc20": learnability(tc20), "Tw": learnability(tw20)}}
                for k in REFS:
                    i = ri[k]
                    it = H if k in ("H", "Hc") else T
                    rec[k] = {"sim8": bool(dac[i]), "interp8": bool(interp[i]), "ddc": float(ddc[i]),
                              "lk10": float(lk[i]), "comfort": float(comf[i]), "ev": exec_vs_intended(ex[i], it)}
                    if k in ("T", "Tc20", "Tc8"):
                        rec[k]["ev_vs_H"] = exec_vs_intended(ex[i], H)
                    ex_all[k].append(ex[i])
                rows.append(rec)
                it_T.append(T)
                it_H.append(H)
                logs_.append(lg)
                toks_.append(tok)
                v0_.append(v0)
                lab20.append(np.stack([t20, tc20, tw20]))
            except Exception as e:                                               # noqa: BLE001
                errors.append({"log": lg, "token": tok, "error": f"{type(e).__name__}: {str(e)[:160]}"})
        print(f"  [{li_ + 1}/{len(by)}] {len(rows)} rows, {len(errors)} errors, {time.time() - t0:.0f}s", flush=True)

    n = len(rows)
    r7n = max(ctl["r7_n"], 1)
    want_keys = set(d1) if not a.limit else set(keys)
    got_keys = {(r["log"], r["token"]) for r in rows}
    gates = {
        "C1_identity": ctl["ident_label_bit_exact"] and ctl["ident_p8_bit_exact"] and ctl["ident_sim8_eq_T"],
        "C2_knot_exact": ctl["knot_exact"] and ctl["knot_checks"] > 0,
        "C4_d1_join": ctl["d1_sim8_mismatch"] == 0 and ctl["d1_missing"] == 0 and got_keys == want_keys
        and (bool(a.limit) or n == len(d1)),
        "C5_review7": ctl["r7_dac_T"] == r7n and ctl["r7_dac_H"] == r7n
        and min(ctl["r7_ddc_T"], ctl["r7_ddc_H"], ctl["r7_lk_T"], ctl["r7_lk_H"]) / r7n >= 0.995,
        "C6_margin_direction": ctl["margin_neg_not_interp8"] == 0,
        "C7_batch_independence": ctl["batch_indep_max_m"] <= 1e-6,
    }
    res = {"prereg": "eval/PREREG_P5_TRACKER_AWARE_TARGETS.md", "n": n, "n_logs": len(set(logs_)),
           "n_errors": len(errors), "errors": errors, "controls_raw": ctl, "gates": gates,
           "controls_only": a.controls_only, "limit": a.limit, "mutate": a.mutate, "seconds": None}
    if a.interp != "linear":                                        # linear's JSON stays byte-comparable to P5's
        res["interp"] = a.interp
    if a.controls_only:
        res["seconds"] = round(time.time() - t0, 1)
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
        print(json.dumps({"n": n, "n_logs": res["n_logs"], "n_errors": len(errors), "gates": gates,
                          "controls_raw": ctl}, indent=1))
        print("ZZP5_CONTROLS " + ("PASS" if all(gates.values()) else "FAIL"))
        return 0 if all(gates.values()) else 1

    # ---------------- metrics (full run) ----------------
    W, li = boot_weights(logs_)

    def col(ref, key, sub=None):
        if sub:
            return np.array([float(r[ref]["ev"][sub]) for r in rows])
        return np.array([float(r[ref][key]) for r in rows])

    per_ref = {}
    for k in REFS:
        evk = {m: col(k, None, m) for m in ("max_xy", "along_4s", "mean_abs_along", "mean_abs_cross", "heading_mae_deg")}
        pl = [r[k]["ev"]["path_len_ratio"] for r in rows if r[k]["ev"]["path_len_ratio"] is not None]
        per_ref[k] = {"n": n, "n_logs": len(set(logs_)), "intended": "H" if k in ("H", "Hc") else "T",
                      "sim8_rate": float(col(k, "sim8").mean()), "interp8_rate": float(col(k, "interp8").mean()),
                      "ddc_mean": float(col(k, "ddc").mean()), "ddc_lt1_rate": float((col(k, "ddc") < 1).mean()),
                      "lk10_rate": float((col(k, "lk10") > 0).mean()), "comfort_rate": float(col(k, "comfort").mean()),
                      "exec_vs_intended": {m: {"mean": float(v.mean()), "median": float(np.median(v)),
                                               "p95": float(np.percentile(v, 95))} for m, v in evk.items()},
                      "path_len_ratio": {"n": len(pl), "median": float(np.median(pl)) if pl else None}}
        if k in ("T", "Tc20", "Tc8"):
            per_ref[k]["exec_vs_H"] = {m: float(np.mean([r[k]["ev_vs_H"][m] for r in rows]))
                                       for m in ("max_xy", "mean_abs_along", "mean_abs_cross", "heading_mae_deg")}

    def arm_ii(arm):
        cis = {"sim8": paired_ci(col("T", "sim8"), col(arm, "sim8"), W, li, True),
               "ddc": paired_ci(col("T", "ddc"), col(arm, "ddc"), W, li, True),
               "comfort": paired_ci(col("T", "comfort"), col(arm, "comfort"), W, li, True),
               "lk10": paired_ci((col("T", "lk10") > 0), (col(arm, "lk10") > 0), W, li, True)}
        for m in ("mean_abs_along", "mean_abs_cross", "max_xy", "heading_mae_deg"):
            cis[m] = paired_ci(col("T", None, m), col(arm, None, m), W, li, False)
        adverse = [g for g in ("ddc", "comfort") if cis[g]["hi"] < 0]
        adverse += [g for g in ("lk10", "mean_abs_along", "mean_abs_cross", "max_xy", "heading_mae_deg")
                    if cis[g]["lo"] > 0]
        if cis["sim8"]["hi"] < 0 and not adverse:
            v = "CONFIRMED"
        elif cis["sim8"]["lo"] > 0:
            v = "REFUTED"
        else:
            v = "NOT CONFIRMED"
        return {"verdict": v, "adverse_guards": adverse, "cis": cis}

    ii = arm_ii("Tc20")
    tw = arm_ii("Tw")
    lt = np.array([r["learn"]["t20"] for r in rows])
    lc = np.array([r["learn"]["Tc20"] for r in rows])
    lw = np.array([r["learn"]["Tw"] for r in rows])
    learn = {lab: {"max_acc_mps2": {"median": float(np.median(x[:, 0])), "p95": float(np.percentile(x[:, 0], 95))},
                   "max_heading_rate_dps": {"median": float(np.median(x[:, 1])),
                                            "p95": float(np.percentile(x[:, 1], 95))}}
             for lab, x in (("t20", lt), ("Tc20", lc), ("Tw", lw))}
    risk = (np.median(lc[:, 1]) > 2 * np.median(lt[:, 1])) or (np.median(lc[:, 0]) > 2 * np.median(lt[:, 0]))
    ii["learnability_flag"] = "LEARNABILITY RISK" if risk else "none"
    gates["C3_wrong_sign"] = (np.median(col("Tw", None, "max_xy")) > np.median(col("T", None, "max_xy"))
                              and tw["verdict"] != "CONFIRMED")

    # repairs / breaks by D1 class
    cls = {}
    for r in rows:
        c = cls.setdefault(r["d1_class"], {"n": 0, "T_fail": 0, "Tc20_fail": 0, "repaired": 0, "broken": 0})
        c["n"] += 1
        c["T_fail"] += r["T"]["sim8"]
        c["Tc20_fail"] += r["Tc20"]["sim8"]
        c["repaired"] += r["T"]["sim8"] and not r["Tc20"]["sim8"]
        c["broken"] += (not r["T"]["sim8"]) and r["Tc20"]["sim8"]

    # arm (i): margin flag
    mT = np.array([r["margin_T"] for r in rows])
    fT = col("T", "sim8").astype(bool)
    nfail = int(fT.sum())

    def flag(th):
        fl = mT < th
        return {"threshold_m": th, "flagged_share": float(fl.mean()), "flagged": int(fl.sum()),
                "recall": float((fl & fT).sum() / max(nfail, 1)), "caught": int((fl & fT).sum()),
                "precision": float((fl & fT).sum() / max(int(fl.sum()), 1))}
    m06 = flag(MARGIN_M)
    arm_i = {"at_0.6": m06, "verdict": "VIABLE" if (m06["flagged_share"] <= 0.25 and m06["recall"] >= 0.80)
             else "NOT VIABLE", "sweep_descriptive": [flag(t) for t in SWEEP], "n_sim8_fail_T": nfail,
             "drop_if_sim8": {"flagged_share": float(fT.mean()), "recall": 1.0},
             "margin_T": {"median": float(np.median(mT)), "p05": float(np.percentile(mT, 5))},
             "margin_H": {"median": float(np.median([r["margin_H"] for r in rows]))}}
    trace = {k: np.median(np.array([r[f"devs_{k}"] for r in rows]), 0).tolist() for k in ("Tc20", "Tw", "ident", "Tc8")}
    res.update({"per_ref": per_ref, "arm_ii": ii, "wrong_sign_Tw": tw, "learnability": learn, "d1_class_breakdown": cls,
                "arm_i": arm_i, "iteration_trace_median_max_xy": trace,
                "tc20_minus_tc8_max_m": {"median": float(np.median([r["tc20_minus_tc8_max_m"] for r in rows])),
                                         "max": float(np.max([r["tc20_minus_tc8_max_m"] for r in rows]))},
                "gates": gates, "rows": rows, "seconds": round(time.time() - t0, 1)})
    ok = all(gates.values())
    if not ok:
        res["arm_ii"]["verdict"] = "GATE_FAIL"
        res["arm_i"]["verdict"] = "GATE_FAIL"
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    if a.npz:
        np.savez(a.npz, **{f"exec_{k}": np.stack(v) for k, v in ex_all.items()}, intended_T=np.stack(it_T),
                 intended_H=np.stack(it_H), labels20_t20_tc20_tw=np.stack(lab20), logs=np.array(logs_),
                 tokens=np.array(toks_), v0=np.array(v0_))
    print(json.dumps({k: res[k] for k in ("n", "n_logs", "n_errors", "gates", "arm_i", "learnability",
                                          "iteration_trace_median_max_xy", "d1_class_breakdown")}, indent=1))
    print(json.dumps({"arm_ii": ii, "per_ref_sim8": {k: per_ref[k]["sim8_rate"] for k in REFS}}, indent=1))
    if not ok:
        print("ZZP5_GATE_FAIL " + ",".join(g for g, v in gates.items() if not v))
        return 1
    print(f"ZZP5_DONE ii={ii['verdict']} i={arm_i['verdict']} learn={ii['learnability_flag']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
