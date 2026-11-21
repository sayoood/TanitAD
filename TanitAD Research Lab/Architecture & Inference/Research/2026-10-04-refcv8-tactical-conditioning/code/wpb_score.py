"""SPEC_WPB (REGISTERED c952d4b4..., 2026-10-04T12:33:09Z) -- scoring (CPU; Thor). Writes <arms>/score_wpb.json.

Every arm is paired against T0 on the SAME windows and the SAME sampler seed (R1's episode-cluster bootstrap, B = 2000,
seed-0 draws -- imported from `r1_score.py`, not copied). Bars and the reading rule are SPEC_WPB sec. 5-6 verbatim. Four
families per family (STRATEGIC: N/A, OFF by PI ruling R5; distance keeping UNAVAILABLE in this harness). sha12 only.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC_SHA256 = "c952d4b45a55dfd422c920264f900057e4876780d81d01b859b6695c2d60e04f"
GATE_DIR, GATE_H15, GATE_STRAIGHT_PT, GATE_STRAIGHT_UB = 0.95, 0.70, 0.05, 0.10
TAC_LAT3, TAC_PROG, TAC_HEAD_DEG = 0.95, 0.0674, 15.0
CTRL, CONS, COST = 0.95, 0.95, 10.5
STOP_D = (10.0, 20.0)
PROG_LOG_SCALE = math.log1p(120.0)
REGRESSION_OF = {"T1d": "T1", "T2d": "T2"}
ORDER = ("T1", "T2", "X1h", "T2s", "X2a", "X2b")


def lat3_of_v7(ids):
    ids = np.asarray(ids)
    return np.where(ids == 6, 1, np.where(ids == 7, 2, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1-code", default="/home/nvidia/refcv8_r1/code")
    ap.add_argument("--cache", default="/home/nvidia/refcv8_r1/cache")
    ap.add_argument("--arms", default="/home/nvidia/refcv8_wpb/arms")
    a = ap.parse_args()
    if hashlib.sha256((HERE.parent / "SPEC_WPB.md").read_bytes()).hexdigest() != SPEC_SHA256:
        raise SystemExit("[wpb-score] SPEC_WPB.md differs from the REGISTERED one")
    sys.path.insert(0, a.r1_code)
    import torch
    import r1_lib as L1
    from r1_arms import load_split, dense_labels
    from r1_score import rate, paired, window_metrics, boot_ratio, ci
    from taniteval import four_families as ff

    D, _sel = load_split(a.cache, "eval")
    n = len(D["win_sha12"])
    ep = np.array(D["win_sha12"])
    E = len(np.unique(ep))
    draws = np.random.default_rng(0).integers(0, E, (2000, E))
    cls = np.array(D["_cls"])
    flags = D["_flags"]
    scored = np.array([("turn" in f) or ("sample" in f) for f in flags])
    gt, gv = D["gt"].numpy().astype(np.float64), D["gt_valid"].numpy()
    turn = scored & np.isin(cls, ["turnL", "turnR"])
    straight = scored & (cls == "straight")
    classified = scored & np.isin(cls, ["turnL", "turnR", "straight", "gentle"])
    lat_d, lon_d = dense_labels(D)
    v0 = D["b_pose_last"][:, 3].numpy().astype(np.float64)
    prog_gt = L1.path_length(gt)
    th_gt = L1.terminal_heading(gt)
    # TRAIN-median constant controls for the constraint heads (SPEC sec. 4)
    Dt, _ = load_split(a.cache, "train")
    gtt = Dt["gt"].numpy().astype(np.float64)
    okt = Dt["gt_valid"].numpy()[:, -1] & (L1.path_length(gtt) > 0.5)
    const_prog = float(np.median(L1.path_length(gtt)[okt]))
    lat_t, _lon_t = dense_labels(Dt)
    tt = okt & np.isin(lat_t, (6, 7))
    const_head = float(np.median(np.abs(L1.terminal_heading(gtt)[tt]))) if tt.any() else 0.0

    arms = {}
    for p in sorted(glob.glob(str(Path(a.arms) / "*" / "eval_s*.pt"))):
        r = torch.load(p, map_location="cpu", weights_only=False)
        if [int(x) for x in r["b_r1_i"]] != [int(x) for x in D["b_r1_i"]][: len(r["b_r1_i"])]:
            raise SystemExit(f"[wpb-score] {p}: window order differs from the cache")
        if r.get("spec_sha256") != SPEC_SHA256:
            raise SystemExit(f"[wpb-score] {p}: produced under another SPEC")
        arms.setdefault(r["arm"], {})[int(r["seed"])] = r
    if "T0" not in arms:
        raise SystemExit("[wpb-score] T0 has not been evaluated")
    res = {"spec_sha256": SPEC_SHA256, "n_eval_windows": n, "n_scored": int(scored.sum()), "n_turn": int(turn.sum()),
           "n_turn_episodes": int(len(set(ep[turn]))), "const_controls": {"prog6_train_median_m": const_prog,
                                                                        "abs_terminal_heading_train_median_rad":
                                                                        const_head}, "arms": {}}
    timing = Path(a.arms) / "timing_wpb.json"
    res["timing"] = json.loads(timing.read_text()) if timing.exists() else None
    base = {s: window_metrics(arms["T0"][s]["traj"].numpy().astype(np.float64), gt, gv, ff) for s in arms["T0"]}

    def route(m):
        return ((m["dir"] == m["side_g"]).astype(float), (m["herr"] <= math.radians(15.0)).astype(float))

    floor = {}
    if "T0r" in arms:
        for s in arms["T0r"]:
            if s in base:
                mr = window_metrics(arms["T0r"][s]["traj"].numpy().astype(np.float64), gt, gv, ff)
                d0, h0 = route(base[s])
                dr, hr = route(mr)
                floor[s] = {"dir": abs(float(dr[turn].mean() - d0[turn].mean())),
                            "h15": abs(float(hr[turn].mean() - h0[turn].mean()))}
    res["T0r_floor"] = floor
    for arm, by_seed in sorted(arms.items()):
        out = {}
        for s, r in sorted(by_seed.items()):
            if s not in base:
                continue
            m = window_metrics(r["traj"].numpy().astype(np.float64), gt, gv, ff)
            dir_ok, h15 = route(m)
            b_dir, b_h15 = route(base[s])
            o = {"turn_dir_correct": rate(dir_ok, turn, ep, draws), "turn_heading15": rate(h15, turn, ep, draws),
                 "turn_dir_gain_vs_T0": paired(dir_ok, b_dir, turn, ep, draws),
                 "turn_h15_gain_vs_T0": paired(h15, b_h15, turn, ep, draws),
                 "dADE_straight_vs_T0": paired(m["ade"], base[s]["ade"], straight & np.isfinite(m["ade"]), ep, draws),
                 "dADE_turn_vs_T0": paired(m["ade"], base[s]["ade"], turn & np.isfinite(m["ade"]), ep, draws),
                 "dADE_scored_vs_T0": paired(m["ade"], base[s]["ade"], scored & np.isfinite(m["ade"]), ep, draws)}
            fam = {}
            for cname, cm in (("turn", turn), ("straight", straight), ("scored", scored)):
                fam[cname] = {k: rate(np.nan_to_num(v, nan=0.0), cm & np.isfinite(v), ep, draws)["pt"]
                              for k, v in m.items() if k not in ("dir", "side_g")}
                prog_err = np.abs(L1.path_length(r["traj"].numpy().astype(np.float64)) / np.maximum(prog_gt, 1e-6) - 1)
                fam[cname]["prog_rel_err_pick"] = rate(np.nan_to_num(prog_err), cm & gv[:, -1] & (prog_gt > 0.5),
                                                       ep, draws)["pt"]
            o["four_families_point"] = fam
            o["LONGITUDINAL_distance_keeping"] = "UNAVAILABLE: no lead tracks in this harness"
            o["STRATEGIC"] = "N/A: strategic layer OFF (PI ruling R5, 2026-09-27)"
            # ---- TACTICAL (R8-4 i) ----
            pl = lat3_of_v7(r["p_lat"].numpy().argmax(-1))
            gl = lat3_of_v7(lat_d)
            okl = lat_d != L1.IGNORE_ID
            tac = {"lat3_acc_turn": rate((pl == gl).astype(float), turn & okl, ep, draws),
                   "lat3_majority_turn": float(np.mean(gl[turn & okl] == np.bincount(gl[turn & okl]).argmax()))
                   if (turn & okl).any() else None}
            f1s = []
            for c in (0, 1, 2):
                mm = classified & okl
                tp = ((pl == c) & (gl == c) & mm).sum()
                fp_ = ((pl == c) & (gl != c) & mm).sum()
                fn = ((pl != c) & (gl == c) & mm).sum()
                if (gl[mm] == c).any():
                    f1s.append(2 * tp / max(2 * tp + fp_ + fn, 1))
            tac["lat3_macro_f1_classified"] = float(np.mean(f1s)) if f1s else None
            po = r["p_lon"].numpy().argmax(-1)
            oko = lon_d != L1.IGNORE_ID
            tac["lon_acc_scored"] = rate((po == lon_d).astype(float), scored & oko, ep, draws)
            tac["lon_majority_scored"] = float(np.mean(lon_d[scored & oko] == np.bincount(lon_d[scored & oko]).argmax()))
            if "cons_lat" in r:
                cl, co = r["cons_lat"].numpy(), r["cons_lon"].numpy()
                ar = np.arange(len(lat_d))
                lrow = np.where(okl, lat_d, 0)
                orow = np.where(oko, lon_d, 1)
                p_hat = np.expm1(co[ar, orow, 0] * PROG_LOG_SCALE)
                pm = classified & oko & gv[:, -1] & (prog_gt > 0.5)
                rel = np.abs(p_hat / np.maximum(prog_gt, 1e-6) - 1)
                rel_c = np.abs(const_prog / np.maximum(prog_gt, 1e-6) - 1)
                tac["prog_median_rel_err"] = float(np.median(rel[pm])) if pm.any() else None
                tac["prog_median_rel_err_const_control"] = float(np.median(rel_c[pm])) if pm.any() else None
                psi = cl[ar, lrow, 0] * math.pi
                hm = turn & okl & gv[:, -1]
                err = np.abs(L1.wrap(psi - th_gt))
                errc = np.abs(L1.wrap(np.sign(th_gt) * const_head - th_gt))
                tac["head_rms_deg_turn"] = float(math.degrees(np.sqrt(np.mean(err[hm] ** 2)))) if hm.any() else None
                tac["head_rms_deg_turn_const_control"] = (float(math.degrees(np.sqrt(np.mean(errc[hm] ** 2))))
                                                          if hm.any() else None)
            o["TACTICAL"] = tac
            # ---- controllability (R8-4 ii) ----
            if "ctrl_dir" in r:
                cd = r["ctrl_dir"].numpy()
                cm = cd[:, 0] != -9
                ctrl = {}
                for k, (nm, want) in enumerate((("TURN_L", 1), ("TURN_R", -1), ("LANE_KEEP", 0))):
                    ctrl[nm] = rate((cd[:, k] == want).astype(float), cm, ep, draws)
                    sh = r["ctrl_alloc_share"].numpy()[:, k]
                    ctrl[nm + "_alloc_share_dir"] = float(np.nanmean(sh[cm])) if np.isfinite(sh[cm]).any() else None
                    if "ctrl_alloc_share_lat3" in r:
                        sh3 = r["ctrl_alloc_share_lat3"].numpy()[:, k]
                        ctrl[nm + "_alloc_share_lat3"] = (float(np.nanmean(sh3[cm])) if np.isfinite(sh3[cm]).any()
                                                          else None)
                ctrl["pooled_follow"] = np.stack([(cd[:, 0] == 1), (cd[:, 1] == -1), (cd[:, 2] == 0)], 1).mean(1)
                se = r["stop_d_err"].numpy()
                for k, d in enumerate(STOP_D):
                    elig = cm & (v0 >= 3.0) & (v0 * v0 / (2 * d) <= 4.0)
                    okk = np.isfinite(se[:, k]) & (np.abs(np.nan_to_num(se[:, k], nan=1e9)) <= max(2.0, 0.1 * d))
                    ctrl[f"STOP_at_{int(d)}m"] = rate(okk.astype(float), elig, ep, draws) if elig.any() else None
                o["CONTROLLABILITY"] = ctrl
            # ---- consistency (R8-4 iii): allocated candidates vs their tag ----
            if "kind" in r and isinstance(r["kind"], list):
                shares = np.full(n, np.nan)
                for j in range(min(n, len(r["kind"]))):
                    kd = r["kind"][j].reshape(-1).numpy()
                    if (kd == 1).any():
                        fa = r["fan16"][j][0][kd == 1].float().numpy().astype(np.float64)
                        tag = r["lat3"][j].reshape(-1).numpy()[kd == 1]
                        c3 = lat3_of_v7(L1.dense_lat(fa, np.ones(fa.shape[:2], bool)))
                        shares[j] = float((c3 == tag).mean())
                mm = classified & np.isfinite(shares)
                o["CONSISTENCY_alloc"] = rate(np.nan_to_num(shares), mm, ep, draws) if mm.any() else None
            out[s] = o
        res["arms"][arm] = out

    # ---- bars ----------------------------------------------------------------------------------------------- #
    def bar_route(arm):
        oks = {}
        for s, o in res["arms"].get(arm, {}).items():
            fl = res["T0r_floor"].get(s)
            c = (o["turn_dir_correct"]["pt"] >= GATE_DIR and o["turn_dir_gain_vs_T0"]["ci"][0] is not None
                 and o["turn_dir_gain_vs_T0"]["ci"][0] > 0 and o["turn_heading15"]["pt"] >= GATE_H15
                 and o["dADE_straight_vs_T0"]["pt"] <= GATE_STRAIGHT_PT
                 and o["dADE_straight_vs_T0"]["ci"][1] is not None
                 and o["dADE_straight_vs_T0"]["ci"][1] <= GATE_STRAIGHT_UB
                 and fl is not None and o["turn_dir_gain_vs_T0"]["pt"] > fl["dir"]
                 and o["turn_h15_gain_vs_T0"]["pt"] > fl["h15"])
            oks[s] = bool(c)
        return bool(oks) and set(oks) >= {0, 1} and all(oks.values())

    def bar_tac(arm):
        r = {}
        for s, o in res["arms"].get(arm, {}).items():
            t = o["TACTICAL"]
            r[s] = {"lat3_turn": t["lat3_acc_turn"]["pt"] >= TAC_LAT3,
                    "prog": (t.get("prog_median_rel_err") is not None and t["prog_median_rel_err"] <= TAC_PROG),
                    "head": (t.get("head_rms_deg_turn") is not None and t["head_rms_deg_turn"] <= TAC_HEAD_DEG)}
        return r

    def bar_ctrl(arm):
        res_s = {}
        for s, o in res["arms"].get(arm, {}).items():
            c = o.get("CONTROLLABILITY")
            if c is None:
                res_s[s] = False
                continue
            ok = all(c[nm]["pt"] >= CTRL for nm in ("TURN_L", "TURN_R", "LANE_KEEP"))
            for d in STOP_D:
                st = c.get(f"STOP_at_{int(d)}m")
                ok = ok and (st is not None and st["pt"] >= CTRL)
            if arm.startswith("T2"):
                ok = ok and all((c.get(nm + "_alloc_share_dir") or 0.0) >= CTRL
                                for nm in ("TURN_L", "TURN_R", "LANE_KEEP"))
            res_s[s] = bool(ok)
        return res_s

    bars = {}
    for arm in res["arms"]:
        bars[arm] = {"B_ROUTE": bar_route(arm), "B_TAC": bar_tac(arm), "B_CTRL": bar_ctrl(arm)}
        if arm.startswith("T2"):
            bars[arm]["B_CONS"] = {s: (o.get("CONSISTENCY_alloc") or {}).get("pt", 0.0) >= CONS
                                   for s, o in res["arms"][arm].items()}
        if res["timing"] is not None and arm in res["timing"]["projection"]:
            bars[arm]["B_COST"] = res["timing"]["projection"][arm]["B_COST_PASS"]
    # regression arms must FAIL B-CTRL and B-ROUTE, and sit below their arm on the pooled follow rate (separated)
    reg = {}
    for rg, arm in REGRESSION_OF.items():
        if rg in res["arms"] and arm in res["arms"]:
            seps = {}
            for s in res["arms"][rg]:
                if s in res["arms"][arm] and "CONTROLLABILITY" in res["arms"][rg][s]:
                    pr = res["arms"][rg][s]["CONTROLLABILITY"]["pooled_follow"]
                    pa = res["arms"][arm][s]["CONTROLLABILITY"]["pooled_follow"]
                    cm = arms[rg][s]["ctrl_dir"].numpy()[:, 0] != -9
                    seps[s] = paired(pa.astype(float), pr.astype(float), cm, ep, draws)
            reg[rg] = {"of": arm, "B_ROUTE_failed": not bars[rg]["B_ROUTE"],
                       "B_CTRL_failed_all_seeds": not any(bars[rg]["B_CTRL"].values()),
                       "pooled_follow_arm_minus_regression": seps}
    res["bars"] = bars
    res["regression_checks"] = reg
    for arm in list(res["arms"]):
        for s in res["arms"][arm]:
            c = res["arms"][arm][s].get("CONTROLLABILITY")
            if c is not None and "pooled_follow" in c:
                c["pooled_follow_mean"] = float(np.mean(c.pop("pooled_follow")))
    # ---- reading rule (SPEC sec. 6) -------------------------------------------------------------------------- #
    broken = any((not v["B_ROUTE_failed"]) or (not v["B_CTRL_failed_all_seeds"]) for v in reg.values())
    winner = None
    for arm in ORDER:
        if arm in bars and bars[arm]["B_ROUTE"]:
            rg = {"T1": "T1d", "T2": "T2d"}.get(arm)
            if rg is None or (rg in reg and reg[rg]["B_ROUTE_failed"] and reg[rg]["B_CTRL_failed_all_seeds"]):
                winner = arm
                break
    res["reading"] = {"instrument_broken": broken, "first_clearing_arm": winner,
                      "note": "SPEC_WPB sec. 6; T2 over T1 and the X/T2s adoptions need their own separated gains "
                              "(read from the paired blocks above)"}
    out = Path(a.arms) / "score_wpb.json"
    out.write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)),
                   encoding="utf-8")
    print(json.dumps({"reading": res["reading"], "bars": bars}, indent=1, default=str))


if __name__ == "__main__":
    main()
