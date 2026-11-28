#!/usr/bin/env python3
"""D3 check 3 -- agent-box GT plausibility on the 3-D join the trainer reads (b1_train_plus_eval_agents_3d.jsonl.xz).

One streaming pass.  Rig frame: +x forward, +y LEFT, +z UP, origin rear axle on the road plane (join3d meta
convention_verdict).  Every threshold is a literal from vehicle / pedestrian geometry.  Only clips that are in the
trained caches (4,369 train + 139 eval) are audited; other clips in the join file are counted and skipped.
Ids are sha12 only.

Controls (computed in-script, must read known values):
  K1 world-frame registration : for tracks, world displacement per 0.1 s = |p_world(t+1) - p_world(t)| must be ~0 for the
     many parked/stopped objects.  MUTATION arm: the same statistic recomputed with the ego yaw DROPPED must be several m/s,
     i.e. the statistic can see a mis-registration.
  K2 synthetic duplicate      : a frame with two same-class boxes 0.4 m apart and one 5 m apart must flag exactly 1 pair
     (the pairwise routine is run on the synthetic frame before the real data).
"""
from __future__ import annotations
import collections, hashlib, json, lzma, math, os, sys, time
import numpy as np
import torch

OUT = "/home/nvidia/refcv8_audit/D3"
JOIN3D = "/home/nvidia/data/join3d/b1_train_plus_eval_agents_3d.jsonl.xz"
JOIN2D = "/home/nvidia/data/joins/b1_train_plus_eval_agents.jsonl.xz"
MAN_TR = "/home/nvidia/data/refcv6-b1-416x1024-train/_v2manifest.pt"
MAN_EV = "/home/nvidia/data/refcv6-b1-416x1024-eval139/_v2manifest.pt"
sha12 = lambda s: hashlib.sha256(s.encode()).hexdigest()[:12]

CLASSES = ["automobile", "person", "rider", "heavy_truck", "trailer", "bus", "protruding_object", "other_vehicle",
           "stroller", "train_or_tram_car", "animal"]
CID = {c: i for i, c in enumerate(CLASSES)}
# literal physical ranges  (l_min,l_max, w_min,w_max, h_min,h_max) [m]
RANGE = {
    "automobile": (2.5, 7.0, 1.4, 2.6, 1.0, 2.6),
    "person": (0.2, 1.2, 0.2, 1.2, 0.8, 2.3),
    "rider": (0.8, 2.8, 0.3, 1.2, 1.0, 2.5),
    "heavy_truck": (4.0, 25.0, 2.0, 3.2, 2.0, 4.6),
    "trailer": (4.0, 20.0, 1.5, 3.2, 1.0, 4.6),
    "bus": (6.0, 20.0, 2.2, 3.2, 2.4, 4.4),
    "other_vehicle": (1.0, 25.0, 0.8, 3.5, 0.8, 4.6),
    "train_or_tram_car": (8.0, 40.0, 2.2, 3.6, 2.8, 4.8),
    "stroller": (0.4, 1.6, 0.3, 1.2, 0.5, 1.5),
    "animal": (0.2, 3.0, 0.1, 1.2, 0.2, 2.2),
    "protruding_object": (0.0, 1e9, 0.0, 1e9, 0.0, 1e9),     # unconstrained (positivity only)
}
DUP_M = {"vehicle": 1.0, "person": 0.3, "other": 0.5}        # same-class centre distance threshold
VEHICLE = {"automobile", "heavy_truck", "trailer", "bus", "other_vehicle", "train_or_tram_car"}
EGO_X0, EGO_X1, EGO_Y = -1.0, 4.0, 1.0   # ego footprint rectangle for a box CENTRE (rear axle origin)
BASE_TOL = 1.5      # |cz - h/2| for ground-standing classes
JUMP_M = {"vehicle": 5.0, "person": 2.0, "other": 3.0}      # world displacement per 0.1 s step
GROUND_STANDING = {"automobile", "person", "rider", "heavy_truck", "trailer", "bus", "other_vehicle", "stroller", "train_or_tram_car"}


def hq(h, q):
    """quantile (m/s) from the 0.1 m/s world-speed histogram"""
    c = np.cumsum(h)
    if c[-1] == 0:
        return float("nan")
    return float((np.searchsorted(c, q * c[-1]) + 0.5) / 10.0)


def kind(c):
    return "vehicle" if c in VEHICLE else ("person" if c in ("person", "stroller") else "other")


def dup_pairs(cx, cy, cls_i, thr_by_cls):
    """number of unordered same-class pairs closer than the class threshold; also indices of boxes involved."""
    n = len(cx)
    if n < 2:
        return 0, set()
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d = np.hypot(dx, dy)
    same = cls_i[:, None] == cls_i[None, :]
    thr = np.minimum(thr_by_cls[cls_i][:, None], thr_by_cls[cls_i][None, :])
    m = np.triu(same & (d < thr), k=1)
    ii, jj = np.where(m)
    inv = set(ii.tolist()) | set(jj.tolist())
    return int(len(ii)), inv


THR = np.array([DUP_M[kind(c)] for c in CLASSES])


def main():
    t0 = time.time()
    # ---- K2 synthetic duplicate control ---------------------------------------------------------------
    k2 = dup_pairs(np.array([10.0, 10.3, 15.0]), np.array([0.0, 0.27, 0.0]), np.array([0, 0, 0]), THR)
    assert k2[0] == 1, ("K2 control failed", k2)
    k2b = dup_pairs(np.array([10.0, 10.3]), np.array([0.0, 0.27]), np.array([0, 1]), THR)   # different class -> 0
    assert k2b[0] == 0
    # ---- poses -----------------------------------------------------------------------------------------
    mt = torch.load(MAN_TR, map_location="cpu", weights_only=False)
    me = torch.load(MAN_EV, map_location="cpu", weights_only=False)
    poses, split = {}, {}
    for man, sp in ((mt, "train"), (me, "eval")):
        for cid, P in zip(man["clip_id"], man["poses"]):
            poses[cid] = P.numpy().astype(np.float64); split[cid] = sp
    print("poses", len(poses), flush=True)

    agg = {sp: dict(n_boxes=0, n_frames=0, n_empty_frames=0, cls=collections.Counter(), occ=collections.Counter(),
                    out_of_range=collections.Counter(), in_range=collections.Counter(), nan=0,
                    dup_pairs=collections.Counter(), dup_boxes=collections.Counter(), dup_frames=0, dup_same_track_pairs=0,
                    dup_frames_by_cls=collections.Counter(), tid_dup_frames=0, tid_dup_boxes=0,
                    ego_overlap=collections.Counter(), base_bad=collections.Counter(), base_n=collections.Counter(),
                    jumps=collections.Counter(), jump_n=collections.Counter(), clips=set(), clips_with_dup=set(), clips_with_jump=set(),
                    clips_with_ego=set(), agents_per_frame=[], far_gt150=0, behind=0, no_zh=0,
                    dup_examples=[], jump_examples=[], ego_examples=[], dim_examples=collections.defaultdict(list),
                    hist_h={c: np.zeros(60, np.int64) for c in CLASSES}, hist_l={c: np.zeros(60, np.int64) for c in CLASSES},
                    hist_cz_base=np.zeros(80, np.int64)) for sp in ("train", "eval")}
    WB = 500   # world-speed histogram: 0..50 m/s in 0.1 m/s bins
    world_hist = {"train": np.zeros(WB, np.int64), "eval": np.zeros(WB, np.int64)}
    mut_hist = np.zeros(WB, np.int64)
    skipped_lines = 0
    prev = {}          # per clip: (frame_idx, {track_id: (X, Y, kind)}, {track_id: (X_mut, Y_mut)})
    last_clip = None
    seen_clip_order = []
    clip_noncontig = 0
    n_lines = 0
    with lzma.open(JOIN3D, "rt", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            r = json.loads(line)
            cid = r["clip_id"]
            sp = split.get(cid)
            if sp is None:
                skipped_lines += 1
                continue
            A = agg[sp]
            if cid != last_clip:
                if cid in A["clips"]:
                    clip_noncontig += 1
                A["clips"].add(cid)
                prev = {}
                last_clip = cid
            fi = r["frame_idx"]
            ag = r["agents"]
            n = len(ag)
            A["n_frames"] += 1
            A["n_boxes"] += n
            if len(A["agents_per_frame"]) < 200000 and (n_lines % 4 == 0):
                A["agents_per_frame"].append(n)
            if n == 0:
                A["n_empty_frames"] += 1
                prev = {}
                continue
            cx = np.fromiter((a["cx"] for a in ag), np.float64, n)
            cy = np.fromiter((a["cy"] for a in ag), np.float64, n)
            yw = np.fromiter((a["yaw"] for a in ag), np.float64, n)
            L = np.fromiter((a["l"] for a in ag), np.float64, n)
            W = np.fromiter((a["w"] for a in ag), np.float64, n)
            H = np.fromiter((a.get("h", np.nan) if a.get("h") is not None else np.nan for a in ag), np.float64, n)
            CZ = np.fromiter((a.get("cz", np.nan) if a.get("cz") is not None else np.nan for a in ag), np.float64, n)
            ci = np.fromiter((CID.get(a["cls"], -1) for a in ag), np.int64, n)
            occ = np.fromiter((a["occ"] for a in ag), np.int64, n)
            tid = [a["track_id"] for a in ag]
            if (ci < 0).any():
                A["cls"]["__UNKNOWN__"] += int((ci < 0).sum())
                ci = np.where(ci < 0, CID["protruding_object"], ci)
            for c_i, cnt in zip(*np.unique(ci, return_counts=True)):
                A["cls"][CLASSES[c_i]] += int(cnt)
            for o, cnt in zip(*np.unique(occ, return_counts=True)):
                A["occ"][int(o)] += int(cnt)
            bad = ~(np.isfinite(cx) & np.isfinite(cy) & np.isfinite(L) & np.isfinite(W))
            A["nan"] += int(bad.sum())
            A["no_zh"] += int((~np.isfinite(H) | ~np.isfinite(CZ)).sum())
            A["far_gt150"] += int((np.hypot(cx, cy) > 150).sum())
            A["behind"] += int((cx < 0).sum())
            # ---- dimension plausibility per class ----
            for c_i in np.unique(ci):
                cname = CLASSES[c_i]
                m = ci == c_i
                lo = RANGE[cname]
                okl = (L[m] >= lo[0]) & (L[m] <= lo[1])
                okw = (W[m] >= lo[2]) & (W[m] <= lo[3])
                okh = np.isfinite(H[m]) & (H[m] >= lo[4]) & (H[m] <= lo[5])
                A["in_range"][cname] += int(m.sum())
                A["out_of_range"][(cname, "l")] += int((~okl).sum())
                A["out_of_range"][(cname, "w")] += int((~okw).sum())
                A["out_of_range"][(cname, "h")] += int((~okh).sum())
                A["out_of_range"][(cname, "any")] += int((~(okl & okw & okh)).sum())
                A["hist_h"][cname] += np.histogram(np.nan_to_num(H[m], nan=-1), bins=60, range=(0, 6))[0]
                A["hist_l"][cname] += np.histogram(L[m], bins=60, range=(0, 30))[0]
                if (~(okl & okw & okh)).any() and len(A["dim_examples"][cname]) < 3:
                    A["dim_examples"][cname].append({"clip": sha12(cid), "frame_idx": fi, "l": round(float(L[m][~(okl & okw & okh)][0]), 2)})
            # ---- base face (cz - h/2) for ground-standing classes ----
            gs = np.array([CLASSES[c] in GROUND_STANDING for c in ci]) & np.isfinite(H) & np.isfinite(CZ)
            base = CZ - H / 2
            A["hist_cz_base"] += np.histogram(base[gs], bins=80, range=(-4, 4))[0]
            for c_i in np.unique(ci[gs]):
                m = gs & (ci == c_i)
                A["base_n"][CLASSES[c_i]] += int(m.sum())
                A["base_bad"][CLASSES[c_i]] += int((np.abs(base[m]) > BASE_TOL).sum())
            # ---- duplicates ----
            npairs, inv = dup_pairs(cx, cy, ci, THR)
            if npairs:
                A["dup_frames"] += 1
                A["clips_with_dup"].add(cid)
                A["dup_pairs"]["all"] += npairs
                for i in inv:
                    A["dup_boxes"][CLASSES[ci[i]]] += 1
                # same-track-id pairs among them
                tarr = np.array(tid, dtype=object)
                d = np.hypot(cx[:, None] - cx[None, :], cy[:, None] - cy[None, :])
                same = (ci[:, None] == ci[None, :]) & (d < np.minimum(THR[ci][:, None], THR[ci][None, :]))
                same = np.triu(same, 1)
                ii, jj = np.where(same)
                A["dup_same_track_pairs"] += int(sum(1 for a_, b_ in zip(ii, jj) if tid[a_] == tid[b_]))
                if len(A["dup_examples"]) < 3:
                    A["dup_examples"].append({"clip": sha12(cid), "frame_idx": fi, "n_pairs": npairs, "cls": CLASSES[ci[ii[0]]],
                                              "dist_m": round(float(d[ii[0], jj[0]]), 3)})
            # identity duplicates (same track id twice in a frame)
            if len(set(tid)) != n:
                A["tid_dup_frames"] += 1
                A["tid_dup_boxes"] += n - len(set(tid))
            # ---- ego footprint overlap (box CENTRE inside the ego rectangle) ----
            eg = (cx > EGO_X0) & (cx < EGO_X1) & (np.abs(cy) < EGO_Y)
            if eg.any():
                for c_i in ci[eg]:
                    A["ego_overlap"][CLASSES[c_i]] += 1
                A["clips_with_ego"].add(cid)
                if len(A["ego_examples"]) < 3:
                    k = int(np.where(eg)[0][0])
                    A["ego_examples"].append({"clip": sha12(cid), "frame_idx": fi, "cls": CLASSES[ci[k]], "cx": round(float(cx[k]), 2), "cy": round(float(cy[k]), 2),
                                              "l": round(float(L[k]), 2), "w": round(float(W[k]), 2)})
            # ---- track continuity in the WORLD frame (ego-motion compensated) ----
            P = poses.get(cid)
            if P is not None and 0 <= fi < len(P):
                x0, y0, ya = P[fi, 0], P[fi, 1], P[fi, 2]
                cs, sn = math.cos(ya), math.sin(ya)
                X = x0 + cx * cs - cy * sn
                Y = y0 + cx * sn + cy * cs
                Xm = x0 + cx - 0 * cy          # MUTATION: ego yaw dropped
                Ym = y0 + cy
                cur = {}
                for k_ in range(n):
                    t_ = tid[k_]
                    if t_ in cur:
                        continue
                    cur[t_] = (X[k_], Y[k_], kind(CLASSES[ci[k_]]), Xm[k_], Ym[k_])
                pf = prev.get("fi")
                if pf is not None and pf == fi - 1:
                    dds, dms = [], []
                    for t_, (px, py, kd, pxm, pym) in prev["tr"].items():
                        c_ = cur.get(t_)
                        if c_ is None:
                            continue
                        dd = math.hypot(c_[0] - px, c_[1] - py)
                        A["jump_n"][kd] += 1
                        if dd > JUMP_M[kd]:
                            A["jumps"][kd] += 1
                            A["clips_with_jump"].add(cid)
                            if len(A["jump_examples"]) < 3:
                                A["jump_examples"].append({"clip": sha12(cid), "frame_idx": fi, "kind": kd, "jump_m": round(dd, 2)})
                        dds.append(dd / 0.1007)
                        dms.append(math.hypot(c_[3] - pxm, c_[4] - pym) / 0.1007)
                    if dds:
                        world_hist[sp] += np.bincount(np.minimum((np.array(dds) * 10).astype(np.int64), WB - 1), minlength=WB)
                        if n_lines % 20 == 0:
                            mut_hist += np.bincount(np.minimum((np.array(dms) * 10).astype(np.int64), WB - 1), minlength=WB)
                prev = {"fi": fi, "tr": cur}
            else:
                prev = {}
            if n_lines % 100000 == 0:
                print("lines", n_lines, round(time.time() - t0, 1), "s", flush=True)

    out = {"n_lines_total": n_lines, "n_lines_skipped_not_in_cache": skipped_lines, "clips_noncontiguous": clip_noncontig,
           "thresholds_literal": {"RANGE": RANGE, "DUP_M": DUP_M, "EGO_RECT_x": [EGO_X0, EGO_X1], "EGO_RECT_halfwidth_y": EGO_Y,
                                  "BASE_TOL": BASE_TOL, "JUMP_M": JUMP_M}, "K2_control_pairs_expected_1": k2[0]}
    for sp, A in agg.items():
        ap = np.array(A["agents_per_frame"]) if A["agents_per_frame"] else np.array([0])
        wh = world_hist[sp]
        d = {
            "n_clips": len(A["clips"]), "n_frames": A["n_frames"], "n_boxes": A["n_boxes"], "n_empty_frames": A["n_empty_frames"],
            "agents_per_frame": {"mean": float(ap.mean()), "q50": float(np.quantile(ap, .5)), "q99": float(np.quantile(ap, .99)), "max": int(ap.max())},
            "class_hist": dict(A["cls"]), "occ_hist": dict(A["occ"]), "nonfinite_boxes": A["nan"], "boxes_without_z_or_h": A["no_zh"],
            "boxes_beyond_150m": A["far_gt150"], "boxes_behind_ego_cx_lt_0": A["behind"],
            "dim_out_of_range": {f"{c}:{k}": v for (c, k), v in sorted(A["out_of_range"].items())},
            "dim_n_per_class": dict(A["in_range"]), "dim_examples": dict(A["dim_examples"]),
            "base_face_gt1.5m": {c: [A["base_bad"][c], A["base_n"][c]] for c in A["base_n"]},
            "duplicates": {"frames_with_dup_pair": A["dup_frames"], "pct_frames": round(100 * A["dup_frames"] / max(A["n_frames"], 1), 4),
                           "pairs": A["dup_pairs"]["all"], "boxes_involved_by_class": dict(A["dup_boxes"]),
                           "pairs_with_same_track_id": A["dup_same_track_pairs"], "clips_affected": len(A["clips_with_dup"]),
                           "frames_with_repeated_track_id": A["tid_dup_frames"], "boxes_repeated_track_id": A["tid_dup_boxes"],
                           "examples": A["dup_examples"]},
            "ego_footprint_overlap_boxes": dict(A["ego_overlap"]), "ego_overlap_clips": len(A["clips_with_ego"]), "ego_examples": A["ego_examples"],
            "track_jumps": {"jumps": dict(A["jumps"]), "steps_checked": dict(A["jump_n"]), "clips_affected": len(A["clips_with_jump"]), "examples": A["jump_examples"]},
            "world_speed_of_tracks_ms": {"n": int(wh.sum()), **{f"p{q}": hq(wh, q / 100) for q in (10, 25, 50, 75, 90, 99)}},
            "hist_h": {c: A["hist_h"][c].tolist() for c in CLASSES}, "hist_l": {c: A["hist_l"][c].tolist() for c in CLASSES},
            "hist_base_face": A["hist_cz_base"].tolist(),
        }
        out[sp] = d
    out["K1_registration"] = {"world_speed_p10_real_train": out["train"]["world_speed_of_tracks_ms"]["p10"],
                              "world_speed_p25_real_train": out["train"]["world_speed_of_tracks_ms"]["p25"],
                              "world_speed_p10_MUTATED_yaw_dropped": hq(mut_hist, .10),
                              "world_speed_p25_MUTATED_yaw_dropped": hq(mut_hist, .25), "n_mut": int(mut_hist.sum())}
    out["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(out, open(f"{OUT}/c3_agents_result.json", "w"), indent=1, default=str)
    print("done", out["elapsed_s"], flush=True)


if __name__ == "__main__":
    main()
