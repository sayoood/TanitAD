"""perc8_box_read.py -- eval-grade BOX read of arm A (refcv8) vs arm B (refcv7 banked), episode-cluster bootstrap, marginal + paired.

    python perc8_box_read.py --a-pkl <refcv8 perc_det.pkl> --a-rows <rows.npz> --b-pkl <refcv7 perc_det.pkl> --b-rows <rows.npz> \
        --out <json> [--n-boot 1000] [--workers 8] \
        [--a-gates g.json --a-nms n.json --a-rows-nms rows_nms.npz  --b-gates ... --b-nms ... --b-rows-nms ...]

Estimator: episode-cluster bootstrap over the clips (the same draws as `taniteval.ci._draws`, seed 0), PAIRED = same draws for both
arms.  It answers "would another draw of EPISODES say this?" -- ONE training seed per arm, so it says nothing about run-to-run
training variance (H-ESTIM-SEED-1) or inference variance.  See perc8_score.py for the controls C1-C3 (recorded in the output).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time

import numpy as np

import perc8_score as S

_G = {}


def _init_worker(spec):
    """Load both arms' sorted rows + slot tables in the worker (works under fork and spawn)."""
    from tanitad.eval import detection_metrics as dm  # noqa: F401  (import check)
    _G.update(spec_load(spec))


def spec_load(spec):
    out = {"H": S.harmonic_table(int(spec["h_max"]))}
    clip_of_common = np.asarray(spec["clip_of_common"], np.int64)
    for arm in ("a", "b"):
        a = spec[arm]
        out[arm] = {"rows": S.Rows(a["rows"], np.asarray(a["win_map"], np.int64), clip_of_common),
                    "slots": slot_table(a["pkl"], np.asarray(a["win_map"], np.int64), clip_of_common)}
        if a.get("rows_nms"):
            out[arm]["rows_nms"] = S.Rows(a["rows_nms"], np.asarray(a["win_map"], np.int64), clip_of_common)
    out["mult_all"] = np.load(spec["mult_npz"])["mult"]
    out["npos_draw"] = {arm: np.load(spec["mult_npz"])[f"npos_{arm}"] for arm in ("a", "b")}
    out["n_clips"] = int(spec["n_clips"])
    return out


def slot_table(pkl, win_map, clip_of_common):
    """Non-exempt slots: presence prob, matched, objectness, clip -- for the weighted AUROC."""
    from tanitad.eval import detection_metrics as dm
    det = pickle.load(open(pkl, "rb"))
    recs = [r["box3d"][0] for r in det if r.get("box3d")]
    P, M, O, C = [], [], [], []
    for wi, pk in enumerate(recs):
        cm = win_map[wi]
        if cm < 0:
            continue
        keep = ~pk["exempt"]
        p = 1.0 / (1.0 + np.exp(-pk["logit"].astype(np.float64)))
        P.append(p[keep])
        M.append(pk["matched"][keep])
        pos_xy = pk["gt_xy"][pk["pos"]]
        if len(pos_xy):
            d = np.sqrt(((pk["xy"][:, None, :].astype(np.float64) - pos_xy[None].astype(np.float64)) ** 2).sum(-1))
            obj = d.min(1) <= dm.OBJECTNESS_RADIUS_M
        else:
            obj = np.zeros(len(p), bool)
        O.append(obj[keep])
        C.append(np.full(int(keep.sum()), clip_of_common[cm], np.int64))
    P, M, O, C = np.concatenate(P), np.concatenate(M), np.concatenate(O), np.concatenate(C)
    uq, inv = np.unique(P, return_inverse=True)
    return {"P": P, "M": M.astype(np.float64), "O": O.astype(np.float64), "clip": C, "inv": inv, "ng": len(uq)}


def draw_metrics(arm, d, with_nms):
    """All the AP/AUROC metrics of ONE arm for ONE draw (multiplicities mult_all[d])."""
    G = _G
    mult = G["mult_all"][d].astype(np.int64)
    npos = G["npos_draw"][arm][d]                                    # [11, 4]
    H = G["H"]
    rows = G[arm]["rows"]
    out = {}
    for ti in range(4):
        out[f"ap{S.THR_KEY[ti]}_all"] = S.pooled_ap_all(rows, ti, mult, int(npos[0, 0]), H)
        aps = S.class_aps(rows, ti, mult, [int(npos[1 + c, 0]) for c in range(10)], H)
        out[f"map{S.THR_KEY[ti]}"] = S.map_mean(aps, [int(npos[1 + c, 0]) for c in range(10)])
        if ti == 2:
            for c in range(10):
                out[f"ap2_cls{c}"] = aps[c]
    for b in range(3):
        out[f"ap2_band{b}"] = S.pooled_ap_all(rows, 2, mult, int(npos[0, 1 + b]), H, band=b)
    if with_nms and "rows_nms" in G[arm]:
        rn = G[arm]["rows_nms"]
        out["nms_ap2_all"] = S.pooled_ap_all(rn, 2, mult, int(npos[0, 0]), H)
        aps = S.class_aps(rn, 2, mult, [int(npos[1 + c, 0]) for c in range(10)], H)
        out["nms_map2"] = S.map_mean(aps, [int(npos[1 + c, 0]) for c in range(10)])
    sl = G[arm]["slots"]
    w = mult[sl["clip"]].astype(np.float64)
    out["auroc_matched"] = S.weighted_auc(sl["inv"], sl["ng"], w, sl["M"])
    out["auroc_objectness"] = S.weighted_auc(sl["inv"], sl["ng"], w, sl["O"])
    return out


def _task(span):
    d0, d1 = span
    with_nms = _G.get("with_nms", False)
    res = []
    for d in range(d0, d1):
        res.append({arm: draw_metrics(arm, d, with_nms) for arm in ("a", "b")})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-pkl", required=True)
    ap.add_argument("--a-rows", required=True)
    ap.add_argument("--b-pkl", required=True)
    ap.add_argument("--b-rows", required=True)
    ap.add_argument("--a-name", default="refcv8_ema48852")
    ap.add_argument("--b-name", default="refcv7_50400")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--a-gates")
    ap.add_argument("--a-nms")
    ap.add_argument("--a-rows-nms")
    ap.add_argument("--b-gates")
    ap.add_argument("--b-nms")
    ap.add_argument("--b-rows-nms")
    ap.add_argument("--skip-controls", action="store_true")
    a = ap.parse_args()
    from tanitad.eval import detection_metrics as dm
    from tanitad.eval import detection_nms as nms_mod
    from taniteval import ci
    t_all = time.time()
    recs = {}
    for arm, pkl in (("a", a.a_pkl), ("b", a.b_pkl)):
        det = pickle.load(open(pkl, "rb"))
        recs[arm] = [(r["sha12"], int(r["ws"]), r["box3d"][0]) for r in det if r.get("box3d")]
    ka = {(s, w): i for i, (s, w, _) in enumerate(recs["a"])}
    kb = {(s, w): i for i, (s, w, _) in enumerate(recs["b"])}
    common = [k for k in ((s, w) for s, w, _ in recs["a"]) if k in kb]
    sets_equal = (set(ka) == set(kb))
    cidx = {k: i for i, k in enumerate(common)}
    win_map = {arm: np.array([cidx.get((s, w), -1) for s, w, _ in recs[arm]], np.int64) for arm in ("a", "b")}
    for arm in ("a", "b"):                                            # monotone, so filtered arrays are in common order
        m = win_map[arm][win_map[arm] >= 0]
        assert np.all(np.diff(m) == 1) and m[0] == 0, f"arm {arm}: common windows are not in common order"
    eid = [s for (s, _) in common]
    uniq = np.unique(np.asarray(eid))
    clip_of_common = np.searchsorted(uniq, np.asarray(eid)).astype(np.int64)
    n_clips, Wc = len(uniq), len(common)
    print(f"[read] common windows {Wc}, clips {n_clips}, sets_equal {sets_equal} "
          f"(A {len(recs['a'])}, B {len(recs['b'])})", flush=True)
    # ---- draws (the programme's own rng stream) ------------------------------------------------------------ #
    rng = np.random.default_rng(a.seed)
    mult_all = np.zeros((a.n_boot, n_clips), np.int16)
    for d in range(a.n_boot):
        pick = rng.choice(uniq, size=n_clips, replace=True)
        mult_all[d] = np.bincount(np.searchsorted(uniq, pick), minlength=n_clips)
    clip_npos = {}
    npos_win = {}
    for arm, rows_path in (("a", a.a_rows), ("b", a.b_rows)):
        z = np.load(rows_path, allow_pickle=False)
        npw = z["npos"][win_map[arm] >= 0]                            # [Wc, 11, 4]
        npos_win[arm] = npw
        cs = np.zeros((n_clips, 11, 4), np.int64)
        np.add.at(cs, clip_of_common, npw)
        clip_npos[arm] = cs
    npos_draw = {arm: np.tensordot(mult_all.astype(np.int64), clip_npos[arm], axes=(1, 0)) for arm in ("a", "b")}
    mult_npz = a.out + ".mult.npz"
    np.savez(mult_npz, mult=mult_all, npos_a=npos_draw["a"], npos_b=npos_draw["b"])
    h_max = int(2.5 * 10815075)
    spec = {"h_max": h_max, "clip_of_common": clip_of_common.tolist(), "n_clips": n_clips, "mult_npz": mult_npz,
            "a": {"rows": a.a_rows, "pkl": a.a_pkl, "win_map": win_map["a"].tolist(), "rows_nms": a.a_rows_nms},
            "b": {"rows": a.b_rows, "pkl": a.b_pkl, "win_map": win_map["b"].tolist(), "rows_nms": a.b_rows_nms}}
    with_nms = bool(a.a_rows_nms and a.b_rows_nms)
    # ---- point estimates (weights = 1) + CONTROLS ---------------------------------------------------------- #
    t0 = time.time()
    _G.update(spec_load(spec))
    _G["with_nms"] = with_nms
    ones = np.ones((1, n_clips), np.int16)
    _G["mult_all"] = ones
    _G["npos_draw"] = {arm: clip_npos[arm].sum(0)[None] for arm in ("a", "b")}
    point = {arm: draw_metrics(arm, 0, with_nms) for arm in ("a", "b")}
    print(f"[read] point estimates {time.time() - t0:.0f}s", flush=True)
    controls = {}
    if not a.skip_controls:
        # C1: weights=1 AP == the programme's pooled AP on the same windows (all-class, 4 thresholds)
        c1 = {}
        for arm in ("a", "b"):
            pk_list = [pk for (s, w, pk) in recs[arm] if (s, w) in cidx]
            n_pos = int(sum(int(pk["pos"].sum()) for pk in pk_list))
            for ti, t in enumerate(S.THR):
                rows = [r for pk in pk_list for r in dm.greedy_rows(pk, t, None)]
                ref = dm._ap(rows, n_pos)
                c1[f"{arm}_t{t}"] = {"mine": point[arm][f"ap{S.THR_KEY[ti]}_all"], "programme": ref,
                                     "abs_diff": abs(point[arm][f"ap{S.THR_KEY[ti]}_all"] - ref)}
        controls["C1_weighted_ap_equals_programme_ap"] = {"pass": bool(all(v["abs_diff"] < 1e-9 for v in c1.values())),
                                                          "cells": c1}
        # C3: weighted AUROC at weights=1 == dm.auroc on the pooled slots
        c3 = {}
        for arm in ("a", "b"):
            sl = _G[arm]["slots"]
            c3[arm] = {"mine_matched": point[arm]["auroc_matched"],
                       "programme_matched": dm.auroc(sl["P"], sl["M"].astype(bool)),
                       "mine_obj": point[arm]["auroc_objectness"],
                       "programme_obj": dm.auroc(sl["P"], sl["O"].astype(bool))}
        controls["C3_weighted_auroc_equals_programme_auroc"] = {
            "pass": bool(all(abs(v["mine_matched"] - v["programme_matched"]) < 1e-9 and
                             abs(v["mine_obj"] - v["programme_obj"]) < 1e-9 for v in c3.values())), "cells": c3}
    # ---- bootstrap: AP/AUROC in worker processes ----------------------------------------------------------- #
    t0 = time.time()
    _G["mult_all"] = mult_all
    _G["npos_draw"] = npos_draw
    spans = [(i, min(i + 25, a.n_boot)) for i in range(0, a.n_boot, 25)]
    if a.workers <= 1:
        parts = [_task(s) for s in spans]
    else:
        from multiprocessing import Pool
        with Pool(a.workers, initializer=_init_worker_ap, initargs=(spec, with_nms)) as pool:
            parts = []
            for k, r in enumerate(pool.imap(_task, spans)):
                parts.append(r)
                if k % 4 == 0:
                    print(f"[boot] {k + 1}/{len(spans)} spans {time.time() - t0:.0f}s", flush=True)
    draws = [x for p in parts for x in p]
    assert len(draws) == a.n_boot
    boots = {arm: {k: np.array([d[arm][k] for d in draws]) for k in draws[0][arm]} for arm in ("a", "b")}
    print(f"[boot] AP/AUROC draws done {time.time() - t0:.0f}s", flush=True)
    # ---- census variants (gate / NMS) from the packs; bootstrap via clip sums ------------------------------ #
    variants = {"declared_gate_0.5": {"a": (float(dm._gate()), None), "b": (float(dm._gate()), None)}}
    if a.a_gates and a.b_gates:
        ga, _ = dm.load_head_gates(a.a_gates)
        gb, _ = dm.load_head_gates(a.b_gates)
        variants["train_PR_gate_noNMS"] = {"a": (ga["box3d"], None), "b": (gb["box3d"], None)}
    if a.a_nms and a.b_nms:
        na, _ = nms_mod.load_head_nms(a.a_nms)
        nb, _ = nms_mod.load_head_nms(a.b_nms)
        variants["train_NMS_plus_gate"] = {"a": (na["box3d"]["gate"], na["box3d"]), "b": (nb["box3d"]["gate"], nb["box3d"])}
    cen = {}
    for vn, v in variants.items():
        cen[vn] = {}
        for arm in ("a", "b"):
            gate, ncfg = v[arm]
            pk_list = [(s, w, pk) for (s, w, pk) in recs[arm] if (s, w) in cidx]
            per, tpc = S.census_table(pk_list, gate, ncfg)
            cs = S.ClipSums(per, clip_of_common, n_clips)
            tcs = S.ClipSums(tpc, clip_of_common, n_clips)
            cen[vn][arm] = {"gate": gate, "nms": ncfg, "per_window": per, "cs": cs, "tcs": tcs, "tpc_total": tpc.sum(0)}
    cbo = {}
    for vn in cen:
        cbo[vn] = {}
        for arm in ("a", "b"):
            tot_all = cen[vn][arm]["cs"].sum.sum(0)
            pt = S.ratios(tot_all)
            totd = mult_all.astype(np.float64) @ cen[vn][arm]["cs"].sum        # [n_boot, K]
            bt = [S.ratios(totd[d]) for d in range(a.n_boot)]
            cbo[vn][arm] = {"point": pt, "boots": {k: np.array([b[k] for b in bt]) for k in
                                                   ("prec", "rec", "f1", "conf_ratio", "boxes_per_object",
                                                    "frac_objects_ge2", "cls_acc_tp")}}
    # C2: draw machinery == ci._draws : recall@gate paired CI here == ci.paired_episode_cluster_bootstrap
    if not a.skip_controls:
        vn = "declared_gate_0.5"
        pa, pb = cen[vn]["a"]["per_window"], cen[vn]["b"]["per_window"]

        def rec_red(which):
            def red(idx):
                idx = np.asarray(idx, np.int64)
                tot = which[idx].sum(0)
                return float(tot[0] / tot[2]) if tot[2] else float("nan")
            red.__name__ = "pooled_recall"
            return red
        # a/b windows indices into the SAME array: stack so the reducer indexes per-arm arrays
        stack = np.concatenate([pa, pb])
        W = len(pa)

        def red_stack(idx):
            idx = np.asarray(idx, np.int64)
            tot = stack[idx].sum(0)
            return float(tot[0] / tot[2]) if tot[2] else float("nan")
        red_stack.__name__ = "pooled_recall_stack"
        ai = np.arange(W, dtype=np.float64)
        ref = ci.paired_episode_cluster_bootstrap(ai, ai + W, eid, n_boot=min(a.n_boot, 200), seed=a.seed, reduce=red_stack)
        mine = S.cell_paired(cbo[vn]["a"]["point"]["rec"], cbo[vn]["b"]["point"]["rec"],
                             cbo[vn]["a"]["boots"]["rec"][:min(a.n_boot, 200)],
                             cbo[vn]["b"]["boots"]["rec"][:min(a.n_boot, 200)])
        controls["C2_draws_equal_ci_draws"] = {
            "pass": bool(abs(ref["delta"] - mine["delta"]) < 1e-3 and abs(ref["lo"] - mine["lo"]) < 1e-3
                         and abs(ref["hi"] - mine["hi"]) < 1e-3),
            "ci_py": {k: ref[k] for k in ("delta", "lo", "hi", "n_boot")}, "mine": mine,
            "note": "ci rounds to 4 dp; compared at 1e-3 (the reducer path: the same rng.choice stream, seed 0, first 200 draws)"}
    # ---- assemble ------------------------------------------------------------------------------------------ #
    cells = {}
    for k in point["a"]:
        ca = S.cell_marginal(point["a"][k], boots["a"][k])
        cb = S.cell_marginal(point["b"][k], boots["b"][k])
        cells[k] = {a.a_name: ca, a.b_name: cb,
                    "paired_a_minus_b": S.cell_paired(point["a"][k], point["b"][k], boots["a"][k], boots["b"][k])}
    within = {a.a_name: {}, a.b_name: {}}
    if with_nms:
        for arm, nm in (("a", a.a_name), ("b", a.b_name)):
            for kn, kr in (("nms_ap2_all", "ap2_all"), ("nms_map2", "map2")):
                within[nm][f"{kn}_minus_{kr}"] = S.cell_paired(point[arm][kn], point[arm][kr], boots[arm][kn], boots[arm][kr])
    cen_out = {}
    for vn in cbo:
        cen_out[vn] = {"gate": {"a": cen[vn]["a"]["gate"], "b": cen[vn]["b"]["gate"]},
                       "nms": {"a": cen[vn]["a"]["nms"], "b": cen[vn]["b"]["nms"]}, "cells": {}}
        for k in cbo[vn]["a"]["boots"]:
            pa_, pb_ = cbo[vn]["a"]["point"][k], cbo[vn]["b"]["point"][k]
            cen_out[vn]["cells"][k] = {a.a_name: S.cell_marginal(pa_, cbo[vn]["a"]["boots"][k]),
                                       a.b_name: S.cell_marginal(pb_, cbo[vn]["b"]["boots"][k]),
                                       "paired_a_minus_b": S.cell_paired(pa_, pb_, cbo[vn]["a"]["boots"][k],
                                                                         cbo[vn]["b"]["boots"][k])}
        cen_out[vn]["counts"] = {a.a_name: {k: cbo[vn]["a"]["point"][k] for k in ("tp", "n_conf", "n_pos", "n_detected_objects")},
                                 a.b_name: {k: cbo[vn]["b"]["point"][k] for k in ("tp", "n_conf", "n_pos", "n_detected_objects")}}
        cen_out[vn]["recall_by_class_at_gate"] = {
            a.a_name: [float(x) / float(max(npos_win['a'][:, 1 + c, 0].sum(), 1)) for c, x in enumerate(cen[vn]["a"]["tpc_total"])],
            a.b_name: [float(x) / float(max(npos_win['b'][:, 1 + c, 0].sum(), 1)) for c, x in enumerate(cen[vn]["b"]["tpc_total"])]}
    for vn in cbo:
        if vn == "declared_gate_0.5":
            continue
        for arm, nm in (("a", a.a_name), ("b", a.b_name)):
            for k in ("prec", "rec", "f1", "conf_ratio", "boxes_per_object", "frac_objects_ge2"):
                within[nm][f"{vn}_minus_declared__{k}"] = S.cell_paired(
                    cbo[vn][arm]["point"][k], cbo["declared_gate_0.5"][arm]["point"][k],
                    cbo[vn][arm]["boots"][k], cbo["declared_gate_0.5"][arm]["boots"][k])
    # centre error p50 (matched pairs), point only
    cep = {arm: float(np.median(np.concatenate([pk["pair_err"] for (s, w, pk) in recs[arm] if (s, w) in cidx])))
           for arm in ("a", "b")}
    out = {"tool": "perc8_box_read.py", "arms": {"a": a.a_name, "b": a.b_name},
           "estimator": f"episode-cluster bootstrap over clips (ci._draws rng stream, seed {a.seed}), n_boot {a.n_boot}; "
                        f"paired = the same draws for both arms; percentile 95 % interval",
           "what_the_ci_answers": "would another draw of EPISODES say this? ONE training seed per arm: it says nothing "
                                  "about run-to-run training variance or inference variance",
           "n_windows_common": Wc, "n_clips": n_clips, "window_sets_equal": sets_equal,
           "n_windows_arm": {"a": len(recs["a"]), "b": len(recs["b"])},
           "n_pos": {a.a_name: int(npos_win["a"][:, 0, 0].sum()), a.b_name: int(npos_win["b"][:, 0, 0].sum())},
           "controls": controls, "ap_auroc_cells": cells, "census_variants": cen_out, "within_arm_effects": within,
           "centre_err_p50_matched": {a.a_name: cep["a"], a.b_name: cep["b"]},
           "classes": list(_classes()), "bands": list(S.BAND_NAMES), "thresholds_m": list(S.THR),
           "wall_s": round(time.time() - t_all, 1)}
    json.dump(out, open(a.out, "w"), indent=1, default=lambda o: None if isinstance(o, float) and o != o else float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else str(o))
    print(f"[read] DONE {a.out} wall {out['wall_s']}s; controls: "
          f"{ {k: v['pass'] for k, v in controls.items()} }", flush=True)


def _classes():
    from tanitad.models.agent_slots import AGENT_CLASSES
    return tuple(AGENT_CLASSES)


def _init_worker_ap(spec, with_nms):
    from tanitad.eval import detection_metrics as dm  # noqa: F401
    _G.update(spec_load(spec))
    _G["with_nms"] = with_nms


if __name__ == "__main__":
    sys.exit(main())
