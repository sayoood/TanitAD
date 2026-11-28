"""R1 scoring (CPU; Thor or dev box) -- SPEC_R1 sec. 5-7 on the arms' eval outputs. Writes arms/score.json.

Every arm is compared with H0 on the SAME windows and the SAME sampler seed (paired episode-cluster bootstrap, B = 2000,
seed-0 draws over the eval episodes -- the route package's `boot_ratio` construction, copied below). Four families per
family (STRATEGIC: N/A, OFF by PI ruling R5). Bars and reading rule are SPEC_R1 sec. 6-7 verbatim. sha12 only.
Needs the launch tree's taniteval on PYTHONPATH for `four_families._seq_geometry` (the route package's 0-2 s rates).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r1_lib as L1  # noqa: E402
from r1_arms import load_split, dense_labels  # noqa: E402

B = 2000
SEED = 0
GATE_DIR, GATE_H15, GATE_STRAIGHT_PT, GATE_STRAIGHT_UB = 0.95, 0.70, 0.05, 0.10
CTRL_BAR = 0.95


def boot_ratio(num, den, ep, draws):
    eps, inv = np.unique(np.asarray(ep).astype(str), return_inverse=True)
    E = len(eps)
    N = np.bincount(inv, weights=np.asarray(num, np.float64), minlength=E)
    D = np.bincount(inv, weights=np.asarray(den, np.float64), minlength=E)
    cnt = np.stack([np.bincount(d, minlength=E) for d in draws])
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = (cnt @ N) / (cnt @ D)
    pt = N.sum() / D.sum() if D.sum() > 0 else float("nan")
    return float(pt), bs


def ci(v):
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v.size else [None, None]


def rate(x, m, ep, draws):
    pt, bs = boot_ratio(np.where(m, x, 0.0), m.astype(float), ep, draws)
    return {"pt": pt, "ci": ci(bs), "n": int(m.sum())}


def paired(x, y, m, ep, draws):
    """mean(x - y) over windows m, episode-cluster bootstrap."""
    d = np.where(m, np.asarray(x, np.float64) - np.asarray(y, np.float64), 0.0)
    pt, bs = boot_ratio(d, m.astype(float), ep, draws)
    return {"pt": pt, "ci": ci(bs), "n": int(m.sum())}


def window_metrics(traj, gt, gv, ff):
    th_p = L1.terminal_heading(traj)
    th_g = L1.terminal_heading(gt)
    side_g = np.sign(th_g)
    dirc = L1.dir_class(th_p)
    ade = L1.ade(traj, gt, gv)
    ok60 = gv[:, -1]
    fde = np.where(ok60, np.linalg.norm(traj[:, -1] - gt[:, -1], axis=-1), np.nan)
    herr = np.abs(L1.wrap(th_p - th_g))
    # along / cross at 6 s in the GT terminal-tangent frame
    t = np.stack([np.cos(th_g), np.sin(th_g)], -1)
    nrm = np.stack([-t[:, 1], t[:, 0]], -1)
    e = traj[:, -1] - gt[:, -1]
    along = np.where(ok60, (e * t).sum(-1), np.nan)
    cross = np.where(ok60, np.abs((e * nrm).sum(-1)), np.nan)
    import torch
    gp = ff._seq_geometry(torch.as_tensor(traj[:, :4], dtype=torch.float32), dt=0.5)
    gg = ff._seq_geometry(torch.as_tensor(gt[:, :4], dtype=torch.float32), dt=0.5)

    def mae(k):
        """speed: every 0.5-s step 0-2 s; heading-derived rates: only where BOTH paths have a tangent
        (`pair_valid`, four_families.py:181) -- a crawling step has no meaningful curvature."""
        a, b = gp[k].numpy(), gg[k].numpy()
        vm = np.isfinite(a) & np.isfinite(b)
        if k in ("yaw_rate", "curvature"):
            vm &= gp["pair_valid"].numpy() & gg["pair_valid"].numpy()
        return np.where(vm.any(-1), np.where(vm, np.abs(a - b), 0).sum(-1) / np.maximum(vm.sum(-1), 1), np.nan)
    out = {"ade": ade, "fde": fde, "dir": dirc, "side_g": side_g, "herr": herr, "along": along, "cross": cross,
           "speed_mae02": mae("speed")}
    for k in ("curvature", "yaw_rate"):
        if k in gp:
            out[f"{k}_mae02"] = mae(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/home/nvidia/refcv8_r1/cache")
    ap.add_argument("--arms", default="/home/nvidia/refcv8_r1/arms")
    a = ap.parse_args()
    import torch
    from taniteval import four_families as ff
    D, sel = load_split(a.cache, "eval")
    n = len(D["win_sha12"])
    ep = np.array(D["win_sha12"])
    draws = np.random.default_rng(SEED).integers(0, len(np.unique(ep)), (B, len(np.unique(ep))))
    cls = np.array(D["_cls"])
    flags = D["_flags"]
    scored = np.array([("turn" in f) or ("sample" in f) for f in flags])
    grid = np.array(["grid" in f for f in flags])
    gt, gv = D["gt"].numpy().astype(np.float64), D["gt_valid"].numpy()
    turn = scored & np.isin(cls, ["turnL", "turnR"])
    straight = scored & (cls == "straight")
    classified = scored & np.isin(cls, ["turnL", "turnR", "straight", "gentle"])
    lat_d, lon_d = dense_labels(D)
    res = {"n_eval_windows": n, "n_scored": int(scored.sum()), "n_turn": int(turn.sum()), "n_straight": int(straight.sum()),
           "n_turn_episodes": int(len(set(ep[turn]))), "arms": {}}
    # ---- label controls (SPEC_R1 sec. 6 as amended by SPEC_R1_A1, sha256 cf3caf0b...) -------------------------- #
    cov = {}
    for c in ("turnL", "turnR", "straight", "gentle", "unclassified"):
        m = scored & (cls == c)
        cov[c] = {"n": int(m.sum()), "coverage": float((lat_d[m] != L1.IGNORE_ID).mean()) if m.any() else None}
    side_d = np.where(lat_d == L1.LAT_V7.index("TURN_L"), 1, np.where(lat_d == L1.LAT_V7.index("TURN_R"), -1, 0))
    side_g = np.where(cls == "turnL", 1, -1)
    lc = {"A1_coverage_per_class": cov,
          "registered_coverage_all_scored_(superseded_by_A1)": float((lat_d[scored] != L1.IGNORE_ID).mean()),
          "A1_PASS": bool(all(cov[c]["coverage"] is not None and cov[c]["coverage"] >= 0.95
                              for c in ("turnL", "turnR", "straight", "gentle"))),
          "validity_turn_side": float((side_d[turn] == side_g[turn]).mean())}
    lc["validity_PASS"] = bool(lc["validity_turn_side"] >= 0.95)
    # H2s: its TRAINING labels (the same seeded within-clip permutation r1_arms used) vs the GT turn class
    T, _st = load_split(a.cache, "train")
    lat_t, _lon_t = dense_labels(T)
    perm, frac = L1.shuffle_within_clip(np.arange(len(lat_t)), T["win_sha12"], seed=0)
    cls_t = np.array(T["_cls"])
    tt = np.isin(cls_t, ["turnL", "turnR"])
    gs_t = np.where(cls_t == "turnL", 1, -1)
    sd = lambda x: np.where(x == 6, 1, np.where(x == 7, -1, 0))  # noqa: E731
    lc["H2s_train_turn_agreement_shuffled"] = float((sd(lat_t[perm])[tt] == gs_t[tt]).mean())
    lc["H2_train_turn_agreement_unshuffled"] = float((sd(lat_t)[tt] == gs_t[tt]).mean())
    lc["H2s_windows_moved_frac"] = frac
    lc["H2s_control_PASS_(must_fall_below_0.95)"] = bool(lc["H2s_train_turn_agreement_shuffled"] < 0.95)
    res["label_controls"] = lc
    del T
    arms = {}
    for p in sorted(glob.glob(str(Path(a.arms) / "*" / "eval_s*.pt"))):
        r = torch.load(p, map_location="cpu", weights_only=False)
        if [int(x) for x in r["b_r1_i"]] != [int(x) for x in D["b_r1_i"]]:
            raise SystemExit(f"[r1-score] {p}: window order differs from the cache")
        arms.setdefault(r["arm"], {})[int(r["seed"])] = r
    if "H0" not in arms:
        raise SystemExit("[r1-score] H0 has not been evaluated")
    # ---- I-D: H0 replayed from disk vs the capture's in-process STORE outputs (seed 0) ------------------ #
    h0 = arms["H0"][0]
    res["I_D"] = {"n_sel_differ": int((h0["sel"].numpy() != D["store_sel"].numpy()).sum()),
                  "n_selb_differ": int((h0["selb"].numpy() != D["store_selb"].numpy()).sum()),
                  "maxabs_s_e9": float((h0["s_e9"] - D["store_s_e9"]).abs().max()),
                  "maxabs_p_lat": float((h0["p_lat"] - D["store_p_lat"]).abs().max())}
    res["I_D"]["PASS"] = bool(res["I_D"]["n_sel_differ"] == 0 and res["I_D"]["maxabs_s_e9"] == 0.0)
    # ---- K-NAV: announced nav on the A5 GRID (all 1,112, incl. the grid-only shards) reproduces A6's counts ---- #
    # (raw/a6_g1_consequences.json: nav_ann active on 117 of 1,112 grid windows, 47 of the 107 GT-turn ones). Computed
    # from the selection, the cached `_now_s` and the table alone -- no model -- with the arms' own `nav_ann`.
    from r1_arms import nav_ann
    allg = {"win_sha12": [], "b_r1_t_now": [], "cls": [], "flags": []}
    winfo = {w["i"]: w for w in sel["windows"]}
    for sp in (sorted(glob.glob(str(Path(a.cache) / "eval" / "shard_*.pt")))
               + sorted(glob.glob(str(Path(a.cache) / "eval" / "gridonly_*.pt")))):
        r = torch.load(sp, map_location="cpu", weights_only=False)
        allg["win_sha12"] += r["win_sha12"]
        allg["b_r1_t_now"].append(r["b_r1_t_now"])
        allg["cls"] += [winfo[int(i)]["cls"] for i in r["b_r1_i"]]
        allg["flags"] += [winfo[int(i)]["flags"] for i in r["b_r1_i"]]
    allg["b_r1_t_now"] = torch.cat(allg["b_r1_t_now"])
    table = json.loads((HERE.parent / "raw" / "r1_nav_ann_table.json").read_text())
    nv, _ = nav_ann(allg, table)
    g_ = np.array(["grid" in f for f in allg["flags"]])
    gt_ = g_ & np.isin(np.array(allg["cls"]), ["turnL", "turnR"])
    k = {"grid_active": int(((nv != 0) & g_).sum()), "grid_n": int(g_.sum()),
         "grid_turn_active": int(((nv != 0) & gt_).sum()), "grid_turn_n": int(gt_.sum()),
         "expected_A6": {"grid_active": 117, "grid_n": 1112, "grid_turn_active": 47, "grid_turn_n": 107}}
    k["PASS"] = bool(k["grid_active"] == 117 and k["grid_n"] == 1112 and k["grid_turn_active"] == 47
                     and k["grid_turn_n"] == 107)
    res["K_NAV"] = k
    base = {s: window_metrics(arms["H0"][s]["traj"].numpy().astype(np.float64), gt, gv, ff) for s in arms["H0"]}
    for arm, by_seed in sorted(arms.items()):
        out = {}
        for s, r in sorted(by_seed.items()):
            m = window_metrics(r["traj"].numpy().astype(np.float64), gt, gv, ff)
            b0 = base[s]
            dir_ok = (m["dir"] == m["side_g"]).astype(float)
            h15 = (m["herr"] <= math.radians(15.0)).astype(float)
            b_dir_ok = (b0["dir"] == b0["side_g"]).astype(float)
            fan = r["fan16"].numpy().astype(np.float64)
            reach = r["reach"].numpy()
            fan_dir = L1.dir_class(L1.terminal_heading(fan))
            fan_ok = (fan_dir == m["side_g"][:, None]) & reach
            fan_ade = L1.ade(fan, gt, gv)
            o = {"turn_dir_correct": rate(dir_ok, turn, ep, draws),
                 "turn_heading15": rate(h15, turn, ep, draws),
                 "turn_dir_gain_vs_H0": paired(dir_ok, b_dir_ok, turn, ep, draws),
                 "dADE_straight_vs_H0": paired(m["ade"], b0["ade"], straight & np.isfinite(m["ade"]), ep, draws),
                 "dADE_turn_vs_H0": paired(m["ade"], b0["ade"], turn & np.isfinite(m["ade"]), ep, draws),
                 "dADE_scored_vs_H0": paired(m["ade"], b0["ade"], scored & np.isfinite(m["ade"]), ep, draws),
                 "fan_contains_correct_turn": rate(fan_ok.any(1).astype(float), turn, ep, draws),
                 "oracle117_ADE_turn": rate(np.nanmin(np.where(reach, fan_ade, np.nan), 1), turn, ep, draws),
                 "random_in_reach_ADE_turn": rate(np.nanmean(np.where(reach, fan_ade, np.nan), 1), turn, ep, draws)}
            fam = {}
            for cname, cm in (("turn", turn), ("straight", straight), ("scored", scored)):
                fam[cname] = {k: rate(np.nan_to_num(v, nan=0.0), cm & np.isfinite(v), ep, draws)["pt"]
                              for k, v in m.items() if k not in ("dir", "side_g")}
                fam[cname]["along_signed"] = fam[cname].pop("along")
                fam[cname]["abs_along"] = rate(np.nan_to_num(np.abs(m["along"])), cm & np.isfinite(m["along"]), ep,
                                               draws)["pt"]
            o["four_families_point"] = fam
            o["LONGITUDINAL_distance_keeping"] = "UNAVAILABLE: no lead tracks in this harness"
            o["STRATEGIC"] = "N/A: strategic layer OFF (PI ruling R5, 2026-09-27)"
            # TACTICAL against the dense labels (3-way lateral, 5-way longitudinal) on every scored window
            tac = {}
            for head, pk_lat, pk_lon in (("behaviour_decoder", "p_lat", "p_lon"), ("z_tac", "lat_tac", "lon_tac")):
                pl = r[pk_lat].numpy().argmax(-1)
                po = r[pk_lon].numpy().argmax(-1)
                for nm, pred, lab, classes in (("lat", pl, lat_d, (0, 6, 7)), ("lon", po, lon_d, (5, 4, 7, 3, 1))):
                    ok = scored & (lab != L1.IGNORE_ID)
                    f1s, sup = [], {}
                    for c in classes:
                        tp = ((pred == c) & (lab == c) & ok).sum()
                        fp_ = ((pred == c) & (lab != c) & ok).sum()
                        fn = ((pred != c) & (lab == c) & ok).sum()
                        f1s.append(2 * tp / max(2 * tp + fp_ + fn, 1))
                        sup[(L1.LAT_V7 if nm == "lat" else L1.LON_V7)[c]] = int(((lab == c) & ok).sum())
                    tac[f"{head}_{nm}"] = {"acc": rate((pred == lab).astype(float), ok, ep, draws)["pt"],
                                           "macro_f1": float(np.mean(f1s)), "support": sup}
            o["TACTICAL_vs_dense"] = tac
            if "ctrl_sel_dir" in r:
                cs = r["ctrl_sel_dir"].numpy()
                sh = r["ctrl_fan_dir_share"].numpy()
                cm_ = classified & (cs[:, 0] != -9)
                lk_m = cm_ & (np.isin(cls, ["straight", "turnL", "turnR"]))
                o["controllability"] = {
                    "TURN_L": rate((cs[:, 0] == 1).astype(float), cm_, ep, draws),
                    "TURN_R": rate((cs[:, 1] == -1).astype(float), cm_, ep, draws),
                    "LANE_KEEP": rate((cs[:, 2] == 0).astype(float), lk_m, ep, draws),
                    "pooled": rate(np.concatenate([(cs[:, 0] == 1), (cs[:, 1] == -1)]).astype(float),
                                   np.concatenate([cm_, cm_]), np.concatenate([ep, ep]), draws),
                    "fan_share_consistent": {k: float(np.nanmean(sh[cm_, i])) for i, k in
                                             enumerate(("TURN_L", "TURN_R", "LANE_KEEP"))}}
            out[str(s)] = o
        # ---- the bar (SPEC_R1 sec. 6) on BOTH sampler seeds ------------------------------------------------ #
        def clears(o):
            c1 = o["turn_dir_correct"]["pt"] >= GATE_DIR and (o["turn_dir_gain_vs_H0"]["ci"][0] or -1) > 0
            c2 = o["turn_heading15"]["pt"] >= GATE_H15
            st = o["dADE_straight_vs_H0"]
            c3 = st["pt"] <= GATE_STRAIGHT_PT and (st["ci"][1] if st["ci"][1] is not None else 9) <= GATE_STRAIGHT_UB
            return {"c1_dir": bool(c1), "c2_h15": bool(c2), "c3_straight": bool(c3), "all": bool(c1 and c2 and c3)}
        if arm != "H0":
            out["bar"] = {s: clears(out[s]) for s in out if s in ("0", "1")}
            out["CLEARS_R1_GATE"] = bool(len(out["bar"]) == 2 and all(v["all"] for v in out["bar"].values()))
        if "0" in out and "controllability" in out["0"]:
            def ctrl_ok(o):
                c = o["controllability"]
                return all(c[k]["pt"] >= CTRL_BAR for k in ("TURN_L", "TURN_R", "LANE_KEEP"))
            out["CONTROLLABILITY_BAR"] = {s: bool(ctrl_ok(out[s])) for s in ("0", "1") if s in out}
        res["arms"][arm] = out
    p = Path(a.arms) / "score.json"
    p.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    brief = {arm: {"clears": v.get("CLEARS_R1_GATE"), "ctrl": v.get("CONTROLLABILITY_BAR"),
                   **{f"s{s}": {k: (v[s][k]["pt"] if isinstance(v[s].get(k), dict) else None)
                                for k in ("turn_dir_correct", "turn_heading15", "dADE_straight_vs_H0")}
                      for s in ("0", "1") if s in v}} for arm, v in res["arms"].items()}
    print(json.dumps({"I_D": res.get("I_D"), "K_NAV": res.get("K_NAV"), "label_controls": res.get("label_controls"),
                      "brief": brief}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
