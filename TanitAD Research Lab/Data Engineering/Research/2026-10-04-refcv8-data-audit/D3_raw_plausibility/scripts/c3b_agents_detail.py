#!/usr/bin/env python3
"""D3 check 3b -- follow-ups the c3 pass raised (run on the dev box on the local copy of the 3-D join; single process, < 1 GB).

 (1) ego-footprint boxes: list EVERY box whose centre lies in x in [-1,4] m, |y| < 1 m (rig frame, rear-axle origin) -> are they the ego vehicle itself?
 (2) track-jump events: for each frame with >= 1 world-frame jump (> 5 m vehicles / > 2 m persons / > 3 m other per 0.1 s) record how many tracks were
     compared, how many jumped, and the MEDIAN world displacement of ALL compared tracks.  A pose glitch moves EVERY track by the same vector (median ~ jump);
     a tracking/ID defect moves only some (median ~ 0).
 (3) K1 registration control with a REAL mutation: world speed of tracks recomputed with the pose of the NEXT row (+1 row = +0.1 s) used for the box -> static
     objects must then appear to move at ~ the ego speed (p50 >> the unmutated p50).
 (4) base face (cz - h/2) by range band (near < 30 m, mid 30-100, far > 100): if z is a calibrated ground-plane height the near-field spread must be tight.
 (5) duplicate pairs: do they PERSIST (the same two tracks closer than the class threshold for >= 5 consecutive frames) or flicker?
Ids sha12 only.
"""
import collections, hashlib, json, lzma, math, os, sys, time
import numpy as np
import torch

JOIN = "D:/refcv6_eval_kit/data/join3d/b1_train_plus_eval_agents_3d.jsonl.xz"
OUT = sys.argv[2]
sha12 = lambda s: hashlib.sha256(s.encode()).hexdigest()[:12]
VEH = {"automobile", "heavy_truck", "trailer", "bus", "other_vehicle", "train_or_tram_car"}
PER = {"person", "stroller"}
GS = VEH | PER | {"rider"}
JUMP = {"v": 5.0, "p": 2.0, "o": 3.0}
DUP = {"v": 1.0, "p": 0.3, "o": 0.5}
kd = lambda c: "v" if c in VEH else ("p" if c in PER else "o")
WB = 400


def main():
    scr_train_man = sys.argv[1]
    mt = torch.load(scr_train_man, map_location="cpu", weights_only=False)
    me = torch.load("D:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139/_v2manifest.pt", map_location="cpu", weights_only=False)
    poses, split = {}, {}
    for man, sp in ((mt, "train"), (me, "eval")):
        for c, P in zip(man["clip_id"], man["poses"]):
            poses[c] = P.numpy().astype(np.float64); split[c] = sp
    maxlines = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
    t0 = time.time()
    ego_boxes = []
    ego_clip_frames = collections.defaultdict(set)
    jump_events = []
    wh = np.zeros(WB, np.int64); wh_mut = np.zeros(WB, np.int64)
    base_hist = {b: np.zeros(80, np.int64) for b in ("near<30", "mid30-100", "far>100")}
    base_gs_n = collections.Counter()
    w_hist = {}
    dup_track = {}      # (clip, tidA, tidB) -> consecutive count
    dup_runs = collections.Counter()
    prev = None; last_clip = None
    nl = 0
    cov = {}      # clip -> [min_fi, max_fi, n_records, n_empty]
    with lzma.open(JOIN, "rt", encoding="utf-8") as f:
        for line in f:
            nl += 1
            if nl > maxlines:
                break
            r = json.loads(line)
            cid = r["clip_id"]
            if cid not in poses:
                continue
            fi = r["frame_idx"]; ag = r["agents"]; n = len(ag)
            cv = cov.setdefault(cid, [10 ** 9, -10 ** 9, 0, 0]); cv[0] = min(cv[0], fi); cv[1] = max(cv[1], fi); cv[2] += 1; cv[3] += int(n == 0)
            if cid != last_clip:
                prev = None; last_clip = cid; dup_track = {}
            if n == 0:
                prev = None; continue
            P = poses[cid]
            cx = np.fromiter((a["cx"] for a in ag), np.float64, n); cy = np.fromiter((a["cy"] for a in ag), np.float64, n)
            W = np.fromiter((a["w"] for a in ag), np.float64, n); H = np.fromiter((a["h"] for a in ag), np.float64, n); CZ = np.fromiter((a["cz"] for a in ag), np.float64, n)
            cls = [a["cls"] for a in ag]; tid = [a["track_id"] for a in ag]
            # (1) ego footprint
            eg = np.where((cx > -1.0) & (cx < 4.0) & (np.abs(cy) < 1.0))[0]
            for k in eg:
                a = ag[k]
                if len(ego_boxes) < 20000:
                    ego_boxes.append((sha12(cid), split[cid], fi, a["cls"], a["cx"], a["cy"], a["l"], a["w"], a["h"], a["yaw"], a["occ"], str(a["track_id"])))
                ego_clip_frames[cid].add(fi)
            # (4) base face by range
            rr = np.hypot(cx, cy)
            gsm = np.array([c in GS for c in cls])
            base = CZ - H / 2
            for name, m in (("near<30", rr < 30), ("mid30-100", (rr >= 30) & (rr < 100)), ("far>100", rr >= 100)):
                mm = m & gsm
                base_hist[name] += np.histogram(base[mm], bins=80, range=(-4, 4))[0]
            for c in set(cls):
                m = np.array([x == c for x in cls])
                h_ = w_hist.setdefault(c, np.zeros(40, np.int64)); h_ += np.histogram(W[m], bins=40, range=(0, 4))[0]
            # (5) duplicate persistence
            ci = np.array([hash(c) % 1000 for c in cls])
            if n >= 2:
                d = np.hypot(cx[:, None] - cx[None, :], cy[:, None] - cy[None, :])
                thr = np.array([DUP[kd(c)] for c in cls])
                same = (ci[:, None] == ci[None, :]) & (d < np.minimum(thr[:, None], thr[None, :]))
                ii, jj = np.where(np.triu(same, 1))
                cur = {}
                for a_, b_ in zip(ii, jj):
                    key = (tid[a_], tid[b_]) if str(tid[a_]) < str(tid[b_]) else (tid[b_], tid[a_])
                    cur[key] = dup_track.get(key, 0) + 1
                for key, v in dup_track.items():
                    if key not in cur:
                        dup_runs[min(v, 10)] += 1
                dup_track = cur
            # (2),(3) world-frame continuity
            if 0 <= fi < len(P) - 1:
                x0, y0, ya = P[fi, 0], P[fi, 1], P[fi, 2]
                c_, s_ = math.cos(ya), math.sin(ya)
                X = x0 + cx * c_ - cy * s_; Y = y0 + cx * s_ + cy * c_
                x1, y1, y1a = P[fi + 1, 0], P[fi + 1, 1], P[fi + 1, 2]
                c1, s1 = math.cos(y1a), math.sin(y1a)
                Xm = x1 + cx * c1 - cy * s1; Ym = y1 + cx * s1 + cy * c1       # MUTATION: next row's pose
                cur = {}
                for k in range(n):
                    if tid[k] not in cur:
                        cur[tid[k]] = (X[k], Y[k], kd(cls[k]), Xm[k], Ym[k])
                if prev is not None and prev[0] == fi - 1:
                    dds, dms, jl = [], [], []
                    for t_, (px, py, k_, pxm, pym) in prev[1].items():
                        q = cur.get(t_)
                        if q is None:
                            continue
                        dx_, dy_ = q[0] - px, q[1] - py
                        dd = math.hypot(dx_, dy_)
                        dds.append(dd / 0.1007); dms.append(math.hypot(q[3] - pxm, q[4] - pym) / 0.1007)
                        if dd > JUMP[k_]:
                            jl.append((dx_, dy_))
                    if dds:
                        wh += np.bincount(np.minimum((np.array(dds) * 10).astype(np.int64), WB - 1), minlength=WB)
                        wh_mut += np.bincount(np.minimum((np.array(dms) * 10).astype(np.int64), WB - 1), minlength=WB)
                    if jl:
                        allv = [(q[0] - prev[1][t_][0], q[1] - prev[1][t_][1]) for t_, q in cur.items() if t_ in prev[1]]
                        mdx = float(np.median([a for a, b in allv])); mdy = float(np.median([b for a, b in allv]))
                        ego_step = float(math.hypot(P[fi, 0] - P[fi - 1, 0], P[fi, 1] - P[fi - 1, 1])) if fi >= 1 else float("nan")
                        jump_events.append({"clip": sha12(cid), "split": split[cid], "frame_idx": fi, "n_compared": len(allv), "n_jump": len(jl),
                                            "median_all_disp_m": round(math.hypot(mdx, mdy), 2), "max_jump_m": round(max(math.hypot(a, b) for a, b in jl), 1), "ego_step_m": round(ego_step, 2)})
                prev = (fi, cur)
            else:
                prev = None
            if nl % 50000 == 0:
                print("lines", nl, round(time.time() - t0, 1), "s", flush=True)
    # mutated world-speed histogram requires a second statistic stored above; recompute quickly on a fresh sub-pass (first 60k lines)
    out = {"lines_read": nl}
    # ---- coverage of agent records vs pose rows (the 'labelled windows' denominator) ----
    miss_tail, miss_head, win_unl = [], [], {"train": [0, 0], "eval": [0, 0]}
    for c, (lo, hi, nrec, nemp) in cov.items():
        T = len(poses[c])
        miss_head.append(max(lo, 0)); miss_tail.append(T - 1 - hi)
        # windows t in range(T-28): now = t+7 ; labelled iff a non-empty record exists at frame_idx == now  (approximation: record present)
    out["record_coverage"] = {"n_clips": len(cov), "first_frame_idx_hist": dict(collections.Counter(min(x, 9) for x in miss_head)),
                              "rows_after_last_record_hist(0..)": dict(collections.Counter(min(x, 30) for x in miss_tail)),
                              "median_rows_after_last_record": float(np.median(miss_tail)), "max_rows_after_last_record": int(max(miss_tail)),
                              "empty_record_frames": int(sum(v[3] for v in cov.values())), "records": int(sum(v[2] for v in cov.values())),
                              "clips_with_any_empty_frame": int(sum(1 for v in cov.values() if v[3] > 0)),
                              "clips_with_zero_agents_ever": int(sum(1 for v in cov.values() if v[3] == v[2]))}
    eb = np.array([(b[4], b[5], b[6], b[7]) for b in ego_boxes]) if ego_boxes else np.zeros((0, 4))
    out["ego_footprint"] = {"n_boxes_listed": len(ego_boxes), "n_clips": len(ego_clip_frames), "frames_total": int(sum(len(v) for v in ego_clip_frames.values())),
                            "split_counts": dict(collections.Counter(b[1] for b in ego_boxes)), "class_counts": dict(collections.Counter(b[3] for b in ego_boxes)),
                            "cx_q": [round(float(x), 2) for x in np.quantile(eb[:, 0], [0.05, .25, .5, .75, .95])] if len(eb) else None,
                            "cy_q": [round(float(x), 2) for x in np.quantile(eb[:, 1], [0.05, .25, .5, .75, .95])] if len(eb) else None,
                            "l_q": [round(float(x), 2) for x in np.quantile(eb[:, 2], [0.05, .5, .95])] if len(eb) else None,
                            "w_q": [round(float(x), 2) for x in np.quantile(eb[:, 3], [0.05, .5, .95])] if len(eb) else None,
                            "frac_within_0.5m_of_(1.4,0)": float(((np.hypot(eb[:, 0] - 1.4, eb[:, 1]) < 0.5).mean())) if len(eb) else None,
                            "per_clip_frames": sorted([len(v) for v in ego_clip_frames.values()], reverse=True)[:20],
                            "clip_sha12": sorted({b[0] for b in ego_boxes})[:20]}
    je = jump_events
    out["jump_events"] = {"n_event_frames": len(je), "n_clips": len({e["clip"] for e in je}),
                          "pose_glitch_like(median_all_disp>=0.5*max_jump)": int(sum(e["median_all_disp_m"] >= 0.5 * e["max_jump_m"] for e in je)),
                          "single_track_like(n_jump==1)": int(sum(e["n_jump"] == 1 for e in je)),
                          "frac_of_compared_tracks_jumping_median": float(np.median([e["n_jump"] / e["n_compared"] for e in je])) if je else None,
                          "max_jump_m_quantiles": [round(float(x), 1) for x in np.quantile([e["max_jump_m"] for e in je], [.1, .5, .9, .99])] if je else None,
                          "top_by_clip_count": collections.Counter(e["clip"] for e in je).most_common(8),
                          "examples": je[:6]}
    out["K1_world_speed_real_hist_p"] = {f"p{q}": float((np.searchsorted(np.cumsum(wh), q / 100 * wh.sum()) + 0.5) / 10) for q in (10, 25, 50, 75, 90)}
    out["K1_world_speed_MUTATED_next_row_pose_hist_p"] = {f"p{q}": float((np.searchsorted(np.cumsum(wh_mut), q / 100 * wh_mut.sum()) + 0.5) / 10) for q in (10, 25, 50, 75, 90)}
    out["base_face_by_range"] = {}
    for name, h in base_hist.items():
        e = np.linspace(-4, 4, 81); cs = np.cumsum(h) / max(h.sum(), 1)
        out["base_face_by_range"][name] = {"n": int(h.sum()), "q05": float(e[np.searchsorted(cs, .05)]), "q50": float(e[np.searchsorted(cs, .5)]), "q95": float(e[np.searchsorted(cs, .95)]),
                                           "frac_abs_gt_1.5": float((h[:25].sum() + h[55:].sum()) / max(h.sum(), 1)), "frac_abs_gt_0.5": float((h[:35].sum() + h[45:].sum()) / max(h.sum(), 1))}
    out["w_hist_q"] = {}
    for c, h in w_hist.items():
        e = np.linspace(0, 4, 41); cs = np.cumsum(h) / h.sum()
        out["w_hist_q"][c] = {"n": int(h.sum()), "q001": float(e[np.searchsorted(cs, .001)]), "q50": float(e[np.searchsorted(cs, .5)]), "q999": float(e[min(np.searchsorted(cs, .999) + 1, 40)])}
    out["dup_persistence_runs"] = {"completed_runs_by_len(1..10+)": {str(k): v for k, v in sorted(dup_runs.items())},
                                   "frac_runs_ge_5": float(sum(v for k, v in dup_runs.items() if k >= 5) / max(sum(dup_runs.values()), 1))}
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1)[:6000])
    print("elapsed", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
