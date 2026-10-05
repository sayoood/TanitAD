"""D6 P4 -- the registered drivable gate (SPEC_P4_DRIVABLE_GATE.md s2), applied to the banked P1' fan.   NAVSIM VENV (devkit geometry + metric cache).

    PYTHONPATH="C:/Users/Admin/navsim-crun/devkit;C:/Users/Admin/navsim-crun/nuplan-devkit" C:/Users/Admin/navsim-crun/venv/Scripts/python.exe \
        d6_p4_gate.py --mode {g0plans|controls|orc|g1|der|frame} [--shard k/n]

  g0plans  raw/p4_plans_G0.npz: the deployed pick's plan for all 5,912 tokens = the OFFICIAL 50,400 seam poses (the G0 arm).
  controls D0: the deployed rule (argmax over reach_keep of sel_score_v3) reproduces sel_idx on every fan record; the fan's pick vs the
           official seam plan; the check-point geometry vs the devkit's own state_array_to_coords_array.
  orc      G-ORC: GT drivable bits at every candidate check point from the metric cache's drivable_area_map (the 4 layers the DAC uses),
           per shard -> raw/p4_orc/gt_s<k>.jsonl; then (``--finalize``) the G-ORC picks + plans.
  g1       G1: the model's own mask (raw/p4_mask) -> picks + plans.
  der      G1-DER: the mask of the deranged donor scene (seed 20261004) -> picks + plans.
  frame    K-FRAME: agreement of the predicted bit with the GT bit at the check points, identity vs y-mirrored transform.

Pick rule (raw/p4_impl_choices.md s4): argmax over {reach_keep AND gate_pass} of sel_score_v3; empty -> the deployed pick.
Outputs raw/p4_picks_<ARM>.json (per token: new idx, changed, pass counts, unchecked shares; summary) and raw/p4_plans_<ARM>.npz (CHANGED
tokens only: their new plan = the fan's cands_poses[new]; unchanged tokens reuse the G0 row).
"""
from __future__ import annotations

import argparse
import base64
import json
import lzma
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d6_p4_common as C                                             # noqa: E402

RAW = C.RAW


def devkit_points(poses: np.ndarray) -> np.ndarray:
    """[N, 8, 3] rear-axle poses -> [N, 8, 5, 2] via the devkit's own function and vehicle."""
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_coords_array
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    p = np.asarray(poses, np.float64)
    st = np.zeros(p.shape[:2] + (StateIndex.size(),), np.float64)
    st[..., StateIndex.X] = p[..., 0]
    st[..., StateIndex.Y] = p[..., 1]
    st[..., StateIndex.HEADING] = p[..., 2]
    return state_array_to_coords_array(st, get_pacifica_parameters())


def inputs_logs() -> dict:
    d = json.load(open(C.INPUTS, encoding="utf-8"))
    return {t: r["log_name"] for t, r in d["tokens"].items()}


def write_plans(arm: str, toks: list, poses: list, logs: dict) -> str:
    p = os.path.join(RAW, f"p4_plans_{arm}.npz")
    np.savez(p, token=np.asarray(toks), log_name=np.asarray([logs[t] for t in toks]),
             poses=(np.stack(poses).astype(np.float32) if poses else np.zeros((0, 8, 3), np.float32)), arm=np.asarray(arm))
    return p


def pick_and_write(arm: str, fan: dict, pass_of: dict, extra: dict, logs: dict) -> dict:
    """pass_of: token -> (pass [117] bool, inside [117,8,5] bool). Applies the registered pick rule and writes picks + plans."""
    out, plans_t, plans_p = {}, [], []
    n_changed = n_none = n_pass_only_masked = 0
    unc_all, unc_pick = [], []
    for t in sorted(fan):
        f = fan[t]
        ok, inside = pass_of[t]
        new = C.deployed_rank(f["s9"], f["rk"], ok)
        none = new < 0
        if none:
            new = f["sel_idx"]
            n_none += 1
            if ok.any():
                n_pass_only_masked += 1
        ch = new != f["sel_idx"]
        n_changed += int(ch)
        u_all = float(1.0 - inside.mean())
        u_pick = float(1.0 - inside[new].mean())
        unc_all.append(u_all)
        unc_pick.append(u_pick)
        out[t] = {"sel": f["sel_idx"], "new": int(new), "changed": bool(ch), "n_pass": int(ok.sum()),
                  "n_pass_rk": int((ok & (f["rk"] > 0)).sum()), "n_rk": int((f["rk"] > 0).sum()), "fallback": bool(none),
                  "deployed_passes": bool(ok[f["sel_idx"]]), "unchecked_all": u_all, "unchecked_pick": u_pick,
                  **{k: v[t] for k, v in extra.items() if t in v}}
        if ch:
            plans_t.append(t)
            plans_p.append(f["poses"][new])
    summ = {"arm": arm, "n": len(out), "n_changed": n_changed, "changed_share": n_changed / len(out),
            "n_fallback_none_pass": n_none, "n_fallback_but_some_pass_outside_reach_keep": n_pass_only_masked,
            "deployed_pick_passes_share": float(np.mean([r["deployed_passes"] for r in out.values()])),
            "unchecked_share_all_candidates_points": float(np.mean(unc_all)),
            "unchecked_share_picked_plan_points": float(np.mean(unc_pick)),
            "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump({"summary": summ, "picks": out}, open(os.path.join(RAW, f"p4_picks_{arm}.json"), "w"), indent=0)
    summ["plans"] = write_plans(arm, plans_t, plans_p, logs)
    print(json.dumps(summ), flush=True)
    return summ


def load_fan_checked() -> dict:
    lf = C.load_fan()
    fan = lf["fan"]
    if len(fan) != C.N_TOK:
        raise SystemExit(f"fan has {len(fan)} tokens, not {C.N_TOK}")
    return fan


def mode_g0plans():
    seam = C.load_seam()
    logs = inputs_logs()
    toks = sorted(seam)
    p = write_plans("G0", toks, [seam[t]["poses"] for t in toks], logs)
    print("G0 plans ->", p, len(toks))


def mode_controls():
    fan = load_fan_checked()
    seam = C.load_seam()
    bad_rank, dmax, dpts = 0, 0.0, 0.0
    n_le = 0
    rng = np.random.default_rng(0)
    sample = set(rng.choice(sorted(fan), 200, replace=False).tolist())
    for t, f in fan.items():
        if C.deployed_rank(f["s9"], f["rk"]) != f["sel_idx"]:
            bad_rank += 1
        d = float(np.abs(f["poses"][f["sel_idx"]].astype(np.float64) - seam[t]["poses"].astype(np.float64)).max())
        dmax = max(dmax, d)
        n_le += int(d <= 1e-4)
        if t in sample:
            dpts = max(dpts, float(np.abs(devkit_points(f["poses"]) - C.check_points(f["poses"])).max()))
    res = {"D0_deployed_rule_reproduces_sel_idx": {"n": len(fan), "mismatches": bad_rank},
           "D0_fan_pick_vs_official_seam_plan": {"max_abs_m": dmax, "share_le_1e-4": n_le / len(fan)},
           "D0_checkpoints_numpy_vs_devkit_max_abs_m_on_200": dpts}
    json.dump(res, open(os.path.join(RAW, "p4_controls_D0.json"), "w"), indent=1)
    print(json.dumps(res))


def mode_orc(shard: str):
    import d6_rescore as R
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    import shapely.vectorized
    k, n = (int(x) for x in shard.split("/"))
    fan = load_fan_checked()
    logs = inputs_logs()
    os.makedirs(os.path.join(RAW, "p4_orc"), exist_ok=True)
    outp = os.path.join(RAW, "p4_orc", f"gt_s{k}.jsonl")
    done = set()
    if os.path.exists(outp):
        for ln in open(outp, encoding="utf-8"):
            try:
                done.add(json.loads(ln)["token"])
            except Exception:                                       # noqa: BLE001
                pass
    with open(outp, "a", encoding="utf-8") as fo:
        for i, t in enumerate(sorted(fan)):
            if i % n != k or t in done:
                continue
            t0 = time.time()
            with lzma.open(R.cache_path(logs[t], t), "rb") as fh:
                mc = pickle.load(fh)
            dam = mc.drivable_area_map
            idcs = dam.get_indices_of_map_type([SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION,
                                                SemanticMapLayer.DRIVABLE_AREA, SemanticMapLayer.CARPARK_AREA])
            pts = devkit_points(fan[t]["poses"])                       # [117, 8, 5, 2] ego frame
            ra = mc.ego_state.rear_axle
            g = C.ego_to_global(pts, ra.x, ra.y, ra.heading).reshape(-1, 2)
            inside_any = np.zeros(len(g), bool)
            geoms = dam._geometries
            for j in idcs:
                inside_any |= shapely.vectorized.contains(geoms[j], g[:, 0], g[:, 1])
            fo.write(json.dumps({"token": t, "n_poly": len(idcs),
                                 "gt_b64": base64.b64encode(np.packbits(inside_any).tobytes()).decode("ascii"),
                                 "s": round(time.time() - t0, 3)}) + "\n")
            fo.flush()
    print("ORC shard done", k, n)


def load_orc_bits() -> dict:
    import glob
    out = {}
    for p in sorted(glob.glob(os.path.join(RAW, "p4_orc", "gt_s*.jsonl"))):
        for ln in open(p, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            b = np.unpackbits(np.frombuffer(base64.b64decode(r["gt_b64"]), np.uint8))[: C.N_FAN * 40]
            out[r["token"]] = b.reshape(C.N_FAN, 8, 5).astype(bool)
    return out


def mode_orc_finalize():
    fan = load_fan_checked()
    logs = inputs_logs()
    gt = load_orc_bits()
    if len(gt) != C.N_TOK:
        raise SystemExit(f"GT bits for {len(gt)} tokens, not {C.N_TOK}")
    pass_of, extra_all = {}, {}
    for t, f in fan.items():
        _, _, inside = C.cell_index(devkit_points(f["poses"]))
        ok = np.where(inside, gt[t], True).reshape(C.N_FAN, -1).all(axis=1)
        pass_of[t] = (ok, inside)
        ok_all = gt[t].reshape(C.N_FAN, -1).all(axis=1)
        extra_all[t] = {"n_pass_allpoints": int(ok_all.sum()),
                        "new_allpoints": int(C.deployed_rank(f["s9"], f["rk"], ok_all))}
    pick_and_write("GORC", fan, pass_of, {"sens": extra_all}, logs)


def mode_mask(arm: str):
    ctlp = os.path.join(RAW, "p4_controls_EXPORT.json")
    if not os.path.exists(ctlp):
        raise SystemExit("EXPORT control not run: no mask may be used before it (run --mode exportctl)")
    ctl = json.load(open(ctlp, encoding="utf-8"))
    if not ctl.get("PASS_exact"):
        raise SystemExit(f"EXPORT control did not pass exactly ({ctl.get('n_equal')}): refusing to use the masks")
    fan = load_fan_checked()
    logs = inputs_logs()
    ms = C.MaskStore(os.path.join(RAW, "p4_mask"))
    toks = sorted(fan)
    if sorted(ms.idx) != toks:
        raise SystemExit(f"mask store has {len(ms.idx)} tokens; the fan has {len(toks)} -- refusing a partial gate")
    donor = {t: t for t in toks}
    meta = {}
    if arm == "G1DER":
        perm, draws = C.derangement(len(toks), 20261004)
        donor = {toks[i]: toks[int(perm[i])] for i in range(len(toks))}
        meta = {"derangement_draws": draws, "fixed_points": int(sum(1 for t in toks if donor[t] == t))}
    pass_of, extra = {}, {}
    for t in toks:
        f = fan[t]
        m = ms.get(donor[t], what=("m_hat",))["m_hat"]
        ok, inside, drv = C.gate_from_mask(m, devkit_points(f["poses"]))
        pass_of[t] = (ok, inside)
        extra[t] = {"donor_sha12": C.sha12(donor[t]) if arm == "G1DER" else None}
    s = pick_and_write(arm, fan, pass_of, {"d": extra}, logs)
    s.update(meta)
    json.dump(s, open(os.path.join(RAW, f"p4_picks_{arm}_summary.json"), "w"), indent=1)


def mode_frame():
    """K-FRAME: at every inside-window check point of every candidate, P(pred bit == GT bit) under identity vs y-mirror, and the
    painted-class share of predicted-non-drivable points that GT calls drivable (diagnostic)."""
    fan = load_fan_checked()
    gt = load_orc_bits()
    ms = C.MaskStore(os.path.join(RAW, "p4_mask"))
    agree = {"identity": [0, 0], "mirror_y": [0, 0]}
    cls_fail = np.zeros(256, np.int64)
    XB = (5.0, 10.0, 20.0, 40.0)
    band = [[0, 0, 0, 0] for _ in range(len(XB) + 1)]
    per_tok = []
    for t in sorted(fan):
        rec = ms.get(t, what=("m_hat", "cls"))
        pts = devkit_points(fan[t]["poses"])
        g = gt[t]
        a_t = {}
        for name, mir in (("identity", False), ("mirror_y", True)):
            i, j, inside = C.cell_index(pts, mirror_y=mir)
            pred = rec["m_hat"][i[inside], j[inside]]
            agree[name][0] += int((pred == g[inside]).sum())
            agree[name][1] += int(inside.sum())
            a_t[name] = float((pred == g[inside]).mean()) if inside.any() else float("nan")
            if not mir:
                bad = (~pred) & g[inside]
                np.add.at(cls_fail, rec["cls"][i[inside], j[inside]][bad], 1)
                xb = np.digitize(pts[..., 0][inside], XB)
                for b in range(len(XB) + 1):
                    sel = (xb == b)
                    band[b][0] += int((pred & g[inside] & sel).sum())       # predicted drivable where GT drivable
                    band[b][1] += int((g[inside] & sel).sum())
                    band[b][2] += int((~pred & ~g[inside] & sel).sum())     # predicted non-drivable where GT non-drivable
                    band[b][3] += int((~g[inside] & sel).sum())
        per_tok.append(a_t["identity"] - a_t["mirror_y"])
    names = ("nocls", "drivable", "lane", "crosswalk", "arrow", "edge", "hatched", "sidewalk")
    res = {"agreement": {k: v[0] / max(v[1], 1) for k, v in agree.items()},
           "n_points": {k: v[1] for k, v in agree.items()},
           "per_token_identity_minus_mirror": {"mean": float(np.nanmean(per_tok)), "share_positive": float(np.nanmean(np.asarray(per_tok) > 0)),
                                               "share_negative": float(np.nanmean(np.asarray(per_tok) < 0))},
           "pred_nondrivable_but_gt_drivable_by_pred_class": {names[c]: int(cls_fail[c]) for c in range(8)},
           "by_x_band_m": {lab: {"recall_drivable": (b[0] / b[1] if b[1] else None), "n_gt_drivable": b[1],
                                 "recall_nondrivable": (b[2] / b[3] if b[3] else None), "n_gt_nondrivable": b[3]}
                           for lab, b in zip(("<5", "5-10", "10-20", "20-40", ">=40"), band)},
           "PASS": None}
    res["PASS"] = bool(res["agreement"]["identity"] > res["agreement"]["mirror_y"])
    json.dump(res, open(os.path.join(RAW, "p4_controls_KFRAME.json"), "w"), indent=1)
    print(json.dumps(res))


def mode_exportctl():
    """EXPORT control (asserted BEFORE any mask is used): the P4 export's plans and ranking equal the banked P1' fan, and its emitted plan
    equals the official seam, on every token; the mask index covers every token and records the same sel_idx."""
    old = load_fan_checked()
    new = C.load_fan(os.path.join(RAW, "p4_fan", "fan_R7_A1.jsonl"))["fan"]
    seam = C.load_seam()
    ms = C.MaskStore(os.path.join(RAW, "p4_mask"))
    res = {"n_old": len(old), "n_new": len(new), "n_mask": len(ms.idx), "mask_torn_lines": ms.torn}
    nb = {"poses_bitwise": 0, "s9_bitwise": 0, "rk_bitwise": 0, "sel_idx": 0, "emitted_vs_seam_bitwise": 0, "mask_sel_idx": 0,
          "order_equal": 0}
    mx = {"poses": 0.0, "s9": 0.0}
    for t, f in old.items():
        g = new.get(t)
        if g is None:
            continue
        nb["poses_bitwise"] += int(np.array_equal(f["poses"], g["poses"]))
        nb["s9_bitwise"] += int(np.array_equal(f["s9"], g["s9"]))
        nb["rk_bitwise"] += int(np.array_equal(f["rk"], g["rk"]))
        nb["sel_idx"] += int(f["sel_idx"] == g["sel_idx"])
        nb["emitted_vs_seam_bitwise"] += int(np.array_equal(g["poses"][g["sel_idx"]], seam[t]["poses"]))
        nb["order_equal"] += int(np.array_equal(np.argsort(-np.where(f["rk"] > 0, f["s9"], -np.inf), kind="stable"),
                                                np.argsort(-np.where(g["rk"] > 0, g["s9"], -np.inf), kind="stable")))
        nb["mask_sel_idx"] += int(t in ms.idx and int(ms.idx[t]["sel_idx"]) == f["sel_idx"])
        mx["poses"] = max(mx["poses"], float(np.abs(f["poses"].astype(np.float64) - g["poses"]).max()))
        mx["s9"] = max(mx["s9"], float(np.abs(f["s9"].astype(np.float64) - g["s9"]).max()))
    res["n_equal"] = nb
    res["max_abs"] = mx
    res["PASS_exact"] = bool(len(new) == len(old) == len(ms.idx) == C.N_TOK and all(v == C.N_TOK for v in nb.values()))
    json.dump(res, open(os.path.join(RAW, "p4_controls_EXPORT.json"), "w"), indent=1)
    print(json.dumps(res))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("g0plans", "controls", "orc", "orc_finalize", "g1", "der", "frame", "exportctl"))
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    if a.mode == "g0plans":
        mode_g0plans()
    elif a.mode == "controls":
        mode_controls()
    elif a.mode == "orc":
        mode_orc(a.shard)
    elif a.mode == "orc_finalize":
        mode_orc_finalize()
    elif a.mode == "g1":
        mode_mask("G1")
    elif a.mode == "der":
        mode_mask("G1DER")
    elif a.mode == "frame":
        mode_frame()
    elif a.mode == "exportctl":
        mode_exportctl()
    return 0


if __name__ == "__main__":
    sys.exit(main())
