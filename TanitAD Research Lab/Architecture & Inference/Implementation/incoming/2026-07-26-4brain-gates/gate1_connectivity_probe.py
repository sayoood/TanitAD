#!/usr/bin/env python3
"""
GATE 1 v2 -- corrected. Two independent probes + the S1 decision-point miner.

v1 bug (fixed here): "junction entry within 15-60 m" was measured to the end of the ego's
CURRENT lane. MADS lane segments are short (~10-20 m), so the window was almost never
satisfiable. The correct measure walks FORWARD along the lane graph through unique-successor
chains, accumulating arclength, until a lane with |succ|>=2 is reached. That branch point is
the junction entry and its successor set is the S1 option set.

v1 bug 2 (fixed here): probe B crashed on 47/51 scenes on a null 'objects' field.

Read-only w.r.t. the scenes. No GPU. Writes json.
"""
import sys, os, io, glob, json, zipfile, collections, traceback, time
sys.path.insert(0, "/workspace/alpa-invest/alpasim/src/runtime")
import numpy as np

OUT = "/workspace/gate1_v2.json"
OUT_S1 = "/workspace/s1_decision_points.json"
SSROOT = "/workspace/alpa-invest/alpasim/data/nre-artifacts/scenesets"
SENTINEL = "-1"
D_MIN, D_MAX = 15.0, 60.0
WALK_MAX = 90.0


def poly_cum_len(pts):
    d = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def dist_pts_to_polyline(q, pts):
    p0 = pts[:-1]; p1 = pts[1:]
    seg = p1 - p0
    L2 = np.maximum((seg ** 2).sum(-1), 1e-12)
    d = q[:, None, :] - p0[None, :, :]
    t = np.clip((d * seg[None]).sum(-1) / L2[None], 0.0, 1.0)
    proj = p0[None] + t[..., None] * seg[None]
    dist = np.linalg.norm(q[:, None, :] - proj, axis=-1)
    k = dist.argmin(1)
    cum = poly_cum_len(pts)
    s = cum[k] + t[np.arange(len(q)), k] * np.sqrt(L2[k])
    return dist[np.arange(len(q)), k], s


def scc(adj, nodes):
    index = {}; low = {}; onstack = {}; stack = []; out = []; c = [0]
    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(adj.get(root, ())))]
        index[root] = low[root] = c[0]; c[0] += 1
        stack.append(root); onstack[root] = True
        while work:
            v, it = work[-1]; adv = False
            for w in it:
                if w not in nodes:
                    continue
                if w not in index:
                    index[w] = low[w] = c[0]; c[0] += 1
                    stack.append(w); onstack[w] = True
                    work.append((w, iter(adj.get(w, ())))); adv = True; break
                elif onstack.get(w):
                    low[v] = min(low[v], index[w])
            if not adv:
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[v])
                if low[v] == index[v]:
                    comp = []
                    while True:
                        w = stack.pop(); onstack[w] = False; comp.append(w)
                        if w == v:
                            break
                    out.append(comp)
    return out


# ------------------------------------------------------------------ PROBE B (raw archive)
def probe_b(usdz_path):
    import pandas as pd
    rec = {"usdz": os.path.basename(os.path.realpath(usdz_path))}
    z = zipfile.ZipFile(usdz_path)
    names = z.namelist()
    rec["archive_entries"] = len(names)

    def find(base):
        m = [n for n in names if n.endswith("/" + base) or n == base]
        return m[0] if m else None

    a = find("association.parquet")
    if a:
        df = pd.read_parquet(io.BytesIO(z.read(a)))
        kinds = collections.Counter()
        rel = {k: collections.defaultdict(set) for k in
               ("NEXT_LANE", "PREVIOUS_LANE", "LEFT_LANE", "RIGHT_LANE")}
        skipped = 0
        for key, assoc in zip(df["key"], df["association"]):
            try:
                kind = key["kind"]
                kinds[kind] += 1
                if kind not in rel:
                    continue
                subs = assoc["subjects"]; objs = assoc["objects"]
                if subs is None or objs is None:          # <-- v1 crashed here
                    skipped += 1; continue
                objs = [o for o in list(objs) if o != SENTINEL]
                for s in list(subs):
                    rel[kind][s] |= set(objs)
            except Exception:
                skipped += 1
        rec["assoc_rows"] = int(df.shape[0])
        rec["assoc_rows_skipped"] = skipped
        rec["assoc_kind_counts"] = dict(kinds.most_common())
        succ = rel["NEXT_LANE"]
        rec["B_n_subjects_with_succ"] = sum(1 for v in succ.values() if v)
        rec["B_n_subjects_succ_ge2"] = sum(1 for v in succ.values() if len(v) >= 2)
        rec["B_succ_hist"] = dict(sorted(collections.Counter(len(v) for v in succ.values()).items()))
        rec["B_n_real_left"] = sum(1 for v in rel["LEFT_LANE"].values() if v)
        rec["B_n_real_right"] = sum(1 for v in rel["RIGHT_LANE"].values() if v)
    for base, key in (("lane.parquet", "B_n_lanes"),
                      ("intersection_area.parquet", "B_n_intersection_areas"),
                      ("road_island.parquet", "B_n_road_islands"),
                      ("traffic_light.parquet", "B_n_traffic_light_rows"),
                      ("obstacle.parquet", "B_n_obstacle_rows")):
        n = find(base)
        if n:
            try:
                rec[key] = int(pd.read_parquet(io.BytesIO(z.read(n))).shape[0])
            except Exception as e:
                rec[key + "_err"] = repr(e)[:100]
    xo = [n for n in names if n.endswith(".xodr")]
    if xo:
        try:
            txt = z.read(xo[0]).decode("utf-8", "ignore")
            rec["B_xodr_junctions"] = txt.count("<junction ")
            rec["B_xodr_roads"] = txt.count("<road ")
            rec["B_xodr_connections"] = txt.count("<connection ")
        except Exception as e:
            rec["B_xodr_err"] = repr(e)[:100]
    return rec


# ------------------------------------------------------------------ PROBE A + S1 miner
def probe_a(ds, sid):
    from trajdata.maps.vec_map_elements import MapElementType
    rec = {}
    vm = ds.map
    lanes = vm.elements[MapElementType.ROAD_LANE]
    rec["n_lanes"] = len(lanes)
    rec["element_counts"] = {str(int(k)): len(v) for k, v in vm.elements.items()}

    def clean(s):
        return {x for x in s if x != SENTINEL} if s else set()

    nxt = {k: clean(L.next_lanes) & set(lanes) for k, L in lanes.items()}
    prv = {k: clean(L.prev_lanes) & set(lanes) for k, L in lanes.items()}
    adjL = {k: clean(L.adj_lanes_left) for k, L in lanes.items()}
    adjR = {k: clean(L.adj_lanes_right) for k, L in lanes.items()}
    cen = {k: L.center.points[:, :2] for k, L in lanes.items()}
    ln = {k: float(poly_cum_len(c)[-1]) for k, c in cen.items()}

    rec["succ_hist"] = dict(sorted(collections.Counter(len(v) for v in nxt.values()).items()))
    rec["n_lanes_with_succ"] = sum(1 for v in nxt.values() if v)
    rec["n_lanes_succ_ge2"] = sum(1 for v in nxt.values() if len(v) >= 2)
    rec["n_lanes_with_any_parallel"] = sum(1 for k in lanes if adjL[k] or adjR[k])
    rec["lane_len_m_med"] = round(float(np.median(list(ln.values()))), 2)
    rec["lane_len_m_p90"] = round(float(np.percentile(list(ln.values()), 90)), 2)
    rec["connectivity_readable"] = bool(rec["n_lanes_with_succ"] > 0)

    # ---- ego
    traj = ds.rig.trajectory
    xyz = np.asarray(traj.positions, dtype=np.float64)
    q = xyz[:, :2]
    rec["ego_n_poses"] = int(len(q))
    rec["ego_path_len_m"] = round(float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum()), 1)
    try:
        ts = np.asarray(traj.timestamps_us, dtype=np.float64)
        rec["ego_duration_s"] = round(float((ts[-1] - ts[0]) / 1e6), 2)
    except Exception:
        ts = np.arange(len(q)) * 1e5
        rec["ego_duration_s"] = None
    try:
        yaws = np.asarray(traj.yaws, dtype=np.float64).ravel()
    except Exception:
        yaws = np.arctan2(*np.gradient(q, axis=0)[:, ::-1].T[::-1])
    try:
        vel = np.asarray(traj.velocities, dtype=np.float64)
        spd = np.linalg.norm(vel[:, :2], axis=1)
    except Exception:
        spd = np.concatenate([[0.0], np.linalg.norm(np.diff(q, axis=0), axis=1) / 0.1])

    # ---- ego -> lane matching
    ids = list(lanes.keys())
    D = np.full((len(q), len(ids)), np.inf); HW = np.zeros(len(ids))
    for j, k in enumerate(ids):
        c = cen[k]
        if c.shape[0] < 2:
            continue
        d, _ = dist_pts_to_polyline(q, c)
        D[:, j] = d
        L = lanes[k]
        if L.left_edge is not None and L.right_edge is not None:
            le = L.left_edge.points[:, :2]; re_ = L.right_edge.points[:, :2]
            n = min(len(le), len(re_))
            HW[j] = float(np.median(np.linalg.norm(le[:n] - re_[:n], axis=1))) / 2.0
        else:
            HW[j] = 1.75
    best = D.argmin(1); bestd = D[np.arange(len(q)), best]
    matched = np.array([ids[j] for j in best], dtype=object)
    inside = bestd <= np.maximum(HW[best], 1.0)
    rec["ego_lane_match_rate"] = round(float(inside.mean()), 4)
    rec["ego_lane_match_med_dist_m"] = round(float(np.median(bestd)), 3)
    rec["lane_match_works"] = bool(rec["ego_lane_match_rate"] >= 0.5)

    # realised lane sequence (ordered, deduped)
    seq = []
    for t in range(len(q)):
        if inside[t] and (not seq or seq[-1] != matched[t]):
            seq.append(matched[t])
    rec["ego_lane_seq_len"] = len(seq)

    # ---- forward lane-graph walk to the first branch point  (THE v1 FIX)
    def walk(lane_id, s_on):
        d = ln[lane_id] - s_on
        cur = lane_id; chain = [lane_id]
        for _ in range(60):
            su = nxt.get(cur, set())
            if len(su) >= 2:
                return d, cur, sorted(su), chain
            if len(su) != 1:
                return None
            nx = next(iter(su))
            if nx in chain:
                return None
            cur = nx; chain.append(cur); d += ln[cur]
            if d > WALK_MAX:
                return None
        return None

    dps = {}
    for t in range(len(q)):
        if not inside[t]:
            continue
        k = matched[t]
        _, s = dist_pts_to_polyline(q[t:t + 1], cen[k])
        w = walk(k, float(s[0]))
        if w is None:
            continue
        d_j, jlane, opts, chain = w
        if not (D_MIN <= d_j <= D_MAX):
            continue
        # keep the LAST qualifying frame per junction-entry lane (per spec)
        dps[jlane] = {"t_dp": int(t), "ego_lane": str(k), "junction_lane": str(jlane),
                      "options": [str(o) for o in opts], "n_options": len(opts),
                      "d_to_junction_m": round(float(d_j), 1),
                      "chain_len": len(chain)}

    # ---- non-circular target: which branch did the ego ACTUALLY take
    out_dps = []
    for jlane, d in dps.items():
        t = d["t_dp"]; opts = d["options"]
        tgt = None; tgt_t = None; via = None
        # descendants of each option (2 hops) to survive short segments
        desc = {}
        for i, o in enumerate(opts):
            s1 = set(nxt.get(o, set()))
            s2 = set().union(*[nxt.get(x, set()) for x in s1]) if s1 else set()
            desc[i] = {o} | s1 | s2
        for u in range(t + 1, len(q)):
            if not inside[u]:
                continue
            m = matched[u]
            hit = [i for i in range(len(opts)) if m in desc[i]]
            if len(hit) == 1:
                tgt = hit[0]; tgt_t = u; via = str(m); break
            if len(hit) > 1:
                continue
        d["target_branch"] = tgt
        d["t_enter"] = tgt_t
        d["entered_lane"] = via
        d["horizon_s"] = None if tgt_t is None else round(float((ts[tgt_t] - ts[t]) / 1e6), 2)
        d["v0_mps"] = round(float(spd[t]), 2)
        d["ego_xy"] = [round(float(q[t][0]), 2), round(float(q[t][1]), 2)]
        d["ego_yaw"] = round(float(yaws[t]), 4)
        # option geometry in the EGO frame at t_dp: bearing of each option's far end
        c0, s0 = np.cos(-yaws[t]), np.sin(-yaws[t])
        geo = []
        for o in opts:
            p = cen[o][-1] - q[t]
            xr = c0 * p[0] - s0 * p[1]; yr = s0 * p[0] + c0 * p[1]
            geo.append({"bearing_rad": round(float(np.arctan2(yr, xr)), 4),
                        "dist_m": round(float(np.hypot(xr, yr)), 1),
                        "heading_delta_rad": round(float(np.arctan2(
                            *( (cen[o][-1] - cen[o][0])[::-1] )) - yaws[t]), 4)})
        d["option_geom_egoframe"] = geo
        # realised future path in the ego frame (the S1 supervision surface)
        fut = q[t:] - q[t]
        d["future_xy_egoframe"] = [[round(float(c0 * p[0] - s0 * p[1]), 2),
                                    round(float(s0 * p[0] + c0 * p[1]), 2)] for p in fut[::5][:60]]
        d["future_path_len_m"] = round(float(np.linalg.norm(np.diff(q[t:], axis=0), axis=1).sum()), 1)
        out_dps.append(d)

    rec["n_s1_dp"] = len(out_dps)
    rec["n_s1_dp_with_target"] = sum(1 for d in out_dps if d["target_branch"] is not None)
    rec["s1_arity_hist"] = dict(sorted(collections.Counter(d["n_options"] for d in out_dps).items()))

    # ---- S2: parallel lanes on the ego's approach
    appr = {m for m, ok in zip(matched, inside) if ok}
    rec["n_ego_lanes"] = len(appr)
    rec["n_ego_lanes_with_parallel"] = sum(1 for k in appr if adjL.get(k) or adjR.get(k))
    rec["s2_option_sets_exist"] = bool(rec["n_ego_lanes_with_parallel"] > 0)

    # ---- S4: roundabout rings
    comps = [c for c in scc(nxt, set(lanes)) if len(c) >= 3]
    rings = []
    for c in comps:
        cs = set(c)
        pts = np.concatenate([cen[k] for k in c], axis=0)
        ctr = pts.mean(0); r = np.linalg.norm(pts - ctr, axis=1)
        cv = float(r.std() / max(r.mean(), 1e-6))
        exits = sorted({x for k in c for x in nxt[k] if x not in cs})
        entries = sorted({x for k in c for x in prv[k] if x not in cs})
        ego_on_ring = bool(appr & cs)
        rings.append({"n_ring_lanes": len(c), "radius_m": round(float(r.mean()), 1),
                      "radial_cv": round(cv, 3), "n_exits": len(exits), "n_entries": len(entries),
                      "annular": bool(cv < 0.35 and 4.0 < r.mean() < 60.0),
                      "ego_traverses": ego_on_ring})
    rings.sort(key=lambda x: (-int(x["ego_traverses"]), -x["n_ring_lanes"]))
    rec["rings"] = rings[:4]
    rec["roundabout_candidate"] = bool(any(x["annular"] and x["n_exits"] >= 3 for x in rings))
    rec["roundabout_ego_traverses"] = bool(any(x["annular"] and x["n_exits"] >= 3 and x["ego_traverses"] for x in rings))
    return rec, out_dps


def main():
    from alpasim_runtime.scene_loader import ArtifactSceneProvider
    setdirs = sorted(d for d in glob.glob(SSROOT + "/*") if os.path.isdir(d))
    scenes = {}
    for sd in setdirs:
        try:
            prov = ArtifactSceneProvider.from_path(sd, smooth_trajectories=True)
        except Exception as e:
            print("PROVIDER FAIL", sd, repr(e)[:120]); continue
        for sid in sorted(prov.scene_ids):
            scenes.setdefault(sid, (os.path.basename(sd), prov))
    print("TOTAL UNIQUE SCENES:", len(scenes), flush=True)

    results = {}; all_dp = []
    t0 = time.time()
    for i, (sid, (ssname, prov)) in enumerate(sorted(scenes.items()), 1):
        rec = {"scene_id": sid, "sceneset": ssname}
        try:
            ds = prov.get_data_source(sid)
            rec["usdz_source"] = os.path.realpath(ds.source) if getattr(ds, "source", None) else None
            a, dps = probe_a(ds, sid)
            rec["A"] = a
            for d in dps:
                d["scene_id"] = sid
                all_dp.append(d)
        except Exception as e:
            rec["A_err"] = repr(e)[:250]; rec["A_tb"] = traceback.format_exc()[-500:]
        try:
            if rec.get("usdz_source") and os.path.exists(rec["usdz_source"]):
                rec["B"] = probe_b(rec["usdz_source"])
        except Exception as e:
            rec["B_err"] = repr(e)[:250]
        results[sid] = rec
        a = rec.get("A", {}); b = rec.get("B", {})
        print("[%2d/%2d] %s A:succ2=%-3s dp=%-2s tgt=%-2s par=%-3s ring=%-5s | B:succ2=%-3s xodrJ=%-3s  %.0fs"
              % (i, len(scenes), sid[8:20], a.get("n_lanes_succ_ge2"), a.get("n_s1_dp"),
                 a.get("n_s1_dp_with_target"), a.get("n_ego_lanes_with_parallel"),
                 a.get("roundabout_ego_traverses"), b.get("B_n_subjects_succ_ge2"),
                 b.get("B_xodr_junctions"), time.time() - t0), flush=True)
        try:
            ds.clear_cache()
        except Exception:
            pass

    json.dump(results, open(OUT, "w"), indent=1, default=str)
    json.dump(all_dp, open(OUT_S1, "w"), indent=1, default=str)
    ok = [r for r in results.values() if "A" in r]
    okb = [r for r in results.values() if "B" in r and "B_n_subjects_succ_ge2" in r["B"]]
    print("=" * 78)
    print("PROBE A ok: %d/%d   PROBE B ok: %d/%d" % (len(ok), len(results), len(okb), len(results)))
    print("A: connectivity readable        :", sum(1 for r in ok if r["A"]["connectivity_readable"]), "/", len(ok))
    print("A: scenes with >=1 |succ|>=2    :", sum(1 for r in ok if r["A"]["n_lanes_succ_ge2"] > 0))
    print("A: scenes with parallel lanes   :", sum(1 for r in ok if r["A"]["s2_option_sets_exist"]))
    print("A: scenes with >=1 S1 DP        :", sum(1 for r in ok if r["A"]["n_s1_dp"] > 0))
    print("A: scenes with >=1 RESOLVED tgt :", sum(1 for r in ok if r["A"]["n_s1_dp_with_target"] > 0))
    print("A: TOTAL S1 DPs                 :", sum(r["A"]["n_s1_dp"] for r in ok))
    print("A: TOTAL resolved targets       :", sum(r["A"]["n_s1_dp_with_target"] for r in ok))
    print("A: roundabout (ego traverses)   :", sum(1 for r in ok if r["A"]["roundabout_ego_traverses"]))
    print("A: median lane segment length m :", round(float(np.median([r["A"]["lane_len_m_med"] for r in ok])), 2))
    print("A: mean ego lane-match rate     :", round(float(np.mean([r["A"]["ego_lane_match_rate"] for r in ok])), 4))
    both = [r for r in results.values() if "A" in r and "B" in r and "B_n_subjects_succ_ge2" in r["B"]]
    ag = sum(1 for r in both if r["A"]["n_lanes_succ_ge2"] == r["B"]["B_n_subjects_succ_ge2"])
    print("A-vs-B |succ|>=2 EXACT agreement: %d/%d" % (ag, len(both)))
    dis = [(r["scene_id"], r["A"]["n_lanes_succ_ge2"], r["B"]["B_n_subjects_succ_ge2"])
           for r in both if r["A"]["n_lanes_succ_ge2"] != r["B"]["B_n_subjects_succ_ge2"]]
    if dis:
        print("  disagreements (scene, A, B):", dis[:8])
    print("B: scenes with xodr junctions>0 :", sum(1 for r in okb if r["B"].get("B_xodr_junctions", 0) > 0))
    print("B: scenes with intersection_area:", sum(1 for r in okb if r["B"].get("B_n_intersection_areas", 0) > 0))
    print("arity hist over all DPs:", dict(sorted(collections.Counter(d["n_options"] for d in all_dp).items())))
    print("WROTE", OUT, "and", OUT_S1)


if __name__ == "__main__":
    main()
