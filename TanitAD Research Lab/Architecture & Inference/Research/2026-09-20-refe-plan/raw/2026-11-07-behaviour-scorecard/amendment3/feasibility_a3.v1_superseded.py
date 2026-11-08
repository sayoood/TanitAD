#!/usr/bin/env python
"""AMENDMENT_3 feasibility pass (stream BEHAV-A3, 2026-11-07). DESCRIPTIVE ONLY: pool sizes, no token is selected.

Implements the draft predicates of AMENDMENT_3_DRAFT.md section 2-4 on the navtest tokens, CPU only:
  map    : F:/Archive/devbox-C/navsim/data/maps/<city>/<ver>/map.gpkg  (lane_connectors, gen_lane_connectors_scaled_width_polygons,
           intersections, stop_polygons)
  logs   : OpenScene navsim_logs/test/*.pkl, frames located by LIST INDEX (navtest_behaviour_mine lesson), token = frames[i+3]
  tokens : navtest.yaml (12,146)
Approximations vs the frozen selector are listed in the draft section 6 (human-safety TTC is an ego-frame strip proxy).
"""
import os, sys, re, json, math, glob, pickle, hashlib, argparse, collections, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MINE = r"E:/Projects/TanitAD-artifacts/navtest_behaviour_mine_20261107/code"
sys.path.insert(0, MINE)
import mine_features as MF  # LOG_DIR, YAML, MAPS, parse_yaml, to_frame, wrap  (pinned in that package's CODE_PIN.json)

NB, NA_ = 4, 30            # window [t0-2 s, t0+15 s] at 2 Hz: 4 frames before t0, 30 after
RA2C, RA2F, EGO_L, EGO_W = 1.461, 4.049, 5.176, 2.297
COVER_TOL = 1.5            # m, connector baseline must lie within this of the human path end to end
TURN_L, TURN_S = 45.0, 20.0
JUNC_TYPES = {0, 1, 2}     # IntersectionType DEFAULT, TRAFFIC_LIGHT, STOP_SIGN (maps_datatypes.py:85-95)
TL_STOP_TYPE = 2           # StopLineType.TRAFFIC_LIGHT (maps_datatypes.py:61-70)
STOPPED_V, MOVING_V = 0.3, 1.0
CROSS_LO, CROSS_HI, ONCOMING = 60.0, 120.0, 135.0
ALIGN = 45.0
MIN_OVERLAP_M2 = 0.5
STOP_CAP = 6.0
TLS_MIN_STOP = [3.0, 5.0, 8.0]
LHT_CITIES = {"sg-one-north"}  # left-hand traffic


def wrapd(a):
    return (a + 180.0) % 360.0 - 180.0


class Map:
    def __init__(self, city):
        import pyogrio, shapely
        from shapely.strtree import STRtree
        self.sh = shapely
        g = glob.glob(f"{MF.MAPS}/{city}/*/map.gpkg")[0]
        meta = pyogrio.read_dataframe(g, layer="meta")
        proj = meta[meta["key"] == "projectedCoordSystem"]["value"].iloc[0]
        lc = pyogrio.read_dataframe(g, layer="lane_connectors", fid_as_index=True).to_crs(proj)
        gen = pyogrio.read_dataframe(g, layer="gen_lane_connectors_scaled_width_polygons").to_crs(proj)
        inter = pyogrio.read_dataframe(g, layer="intersections", fid_as_index=True, read_geometry=False)
        sp = pyogrio.read_dataframe(g, layer="stop_polygons", fid_as_index=True).to_crs(proj)
        itype = {int(k): int(v) for k, v in inter["intersection_type_fid"].items()}
        poly = {int(a): b for a, b in zip(gen["lane_connector_fid"], gen.geometry)}
        self.C = {}
        for fid, r in lc.iterrows():
            fid = int(fid)
            if fid not in poly:
                continue
            ln = self.sh.LineString(np.asarray(r.geometry.coords)[:, :2]); L = ln.length
            if L < 1.0:
                continue
            p = lambda d: np.asarray(ln.interpolate(d).coords[0])
            h0 = math.degrees(math.atan2(*(p(min(2, L)) - p(0))[::-1]))
            h1 = math.degrees(math.atan2(*(p(L) - p(max(0, L - 2)))[::-1]))
            dpsi = wrapd(h1 - h0)
            geo = "U" if abs(dpsi) >= 150 else "L" if dpsi >= TURN_L else "R" if dpsi <= -TURN_L else "S" if abs(dpsi) <= TURN_S else "bend"
            tt = int(r.turn_type_fid) if r.turn_type_fid == r.turn_type_fid else -1
            agree = {"S": 0, "L": 1, "R": 2}.get(geo, -9) == tt
            # stop line for this connector: devkit NuPlanLaneConnector.stop_lines (lane_connector.py:159-192), type-2 only
            sl_pt, sl_src = p(0), "connector_entry"
            ids = [s for s in str(r.traffic_light_stop_line_fids or "").replace(" ", "").split(",") if s]
            cands = [sp.loc[int(s)] for s in ids if int(s) in sp.index and int(sp.loc[int(s)].stop_polygon_type_fid) == TL_STOP_TYPE]
            if cands:
                hit = [c for c in cands if c.geometry.intersects(ln)]
                if hit:
                    best = None
                    for c in hit:
                        x = c.geometry.intersection(ln)
                        for q in getattr(x, "geoms", [x]):
                            for xy in np.asarray(q.coords)[:, :2]:
                                st = ln.project(self.sh.Point(xy))
                                if best is None or st < best[0]:
                                    best = (st, xy)
                    sl_pt, sl_src = np.asarray(best[1]), "stop_polygons:on_baseline"
                else:
                    d = [self.sh.Point(p(0)).distance(c.geometry) for c in cands]
                    c = cands[int(np.argmin(d))]
                    from shapely.ops import nearest_points
                    sl_pt, sl_src = np.asarray(nearest_points(c.geometry, self.sh.Point(p(0)))[0].coords[0]), "stop_polygons:nearest"
            self.C[fid] = dict(ln=ln, L=L, poly=poly[fid], h0=h0, h1=h1, dpsi=dpsi, geo=geo, tt=tt, agree=agree,
                               itype=itype.get(int(r.intersection_fid), -1) if r.intersection_fid == r.intersection_fid else -1,
                               exit_lane=r.exit_lane_fid, entry_lane=r.entry_lane_fid, lgc=int(r.lane_group_connector_fid),
                               sl_pt=sl_pt, sl_src=sl_src, start=p(0))
        self.ids = np.array(list(self.C))
        self.tree = STRtree([self.C[i]["poly"] for i in self.ids])
        self._conf = {}

    def conflicts(self, c):
        """crossing / merging connectors of c: polygon overlap >= MIN_OVERLAP_M2, not c, not a sibling (same incoming lane)."""
        if c in self._conf:
            return self._conf[c]
        C0 = self.C[c]; out = []
        for k in self.tree.query(C0["poly"], predicate="intersects"):
            o = int(self.ids[k])
            if o == c or self.C[o]["exit_lane"] == C0["exit_lane"]:
                continue
            part = C0["poly"].intersection(self.C[o]["poly"])
            if part.area < MIN_OVERLAP_M2:
                continue
            rel = abs(wrapd(self.C[o]["h0"] - C0["h0"]))
            out.append(dict(id=o, part=part, merge=self.C[o]["entry_lane"] == C0["entry_lane"], rel=rel,
                            oncoming=rel >= ONCOMING, perpendicular=CROSS_LO <= rel <= CROSS_HI))
        self._conf[c] = out
        return out


_M = {}


def get_map(city):
    if city not in _M:
        _M[city] = Map(city)
    return _M[city]


def frame_globals(f):
    R = np.asarray(f["ego2global"], float)
    x, y, h = R[0, 3], R[1, 3], math.atan2(R[1, 0], R[0, 0])
    eds = f["ego_dynamic_state"]
    a = f["anns"]
    box = np.asarray(a["gt_boxes"], float).reshape(-1, 7)
    vel = np.asarray(a["gt_velocity_3d"], float).reshape(-1, 3)
    c, s = math.cos(h), math.sin(h)
    gx = x + c * box[:, 0] - s * box[:, 1]; gy = y + s * box[:, 0] + c * box[:, 1]
    return dict(x=x, y=y, h=h, v=math.hypot(eds[0], eds[1]), names=np.asarray(a["gt_names"]), box=box, vel=vel,
                gx=gx, gy=gy, gh=box[:, 6] + h, spd=np.hypot(vel[:, 0], vel[:, 1]),
                trk=list(a["track_tokens"]), tl={int(i): bool(b) for i, b in f["traffic_lights"]},
                rb=set(int(i) for i in f["roadblock_ids"]), ts=int(f["timestamp"]))


def box_poly(sh, x, y, h, L, W):
    c, s = math.cos(h), math.sin(h)
    dx = np.array([L, L, -L, -L]) / 2; dy = np.array([W, -W, -W, W]) / 2
    return sh.Polygon(np.c_[x + c * dx - s * dy, y + s * dx + c * dy])


def human_safe(sh, W):
    """APPROXIMATE (draft section 6): no ego-box overlap with any agent box; ego-frame strip TTC >= 1.0 s."""
    min_ttc = math.inf
    for F in W:
        ex, ey = F["x"] + RA2C * math.cos(F["h"]), F["y"] + RA2C * math.sin(F["h"])
        near = np.where(np.hypot(F["gx"] - ex, F["gy"] - ey) < 12.0)[0]
        if near.size:
            ego = box_poly(sh, ex, ey, F["h"], EGO_L, EGO_W)
            for i in near:
                b = F["box"][i]
                if ego.intersects(box_poly(sh, F["gx"][i], F["gy"][i], F["gh"][i], b[3], b[4])):
                    return False, 0.0
        b = F["box"]; veh = F["names"] == "vehicle"
        hx = 0.5 * (b[:, 3] * np.abs(np.cos(b[:, 6])) + b[:, 4] * np.abs(np.sin(b[:, 6])))
        hy = 0.5 * (b[:, 3] * np.abs(np.sin(b[:, 6])) + b[:, 4] * np.abs(np.cos(b[:, 6])))
        gap = b[:, 0] - hx - RA2F
        m = veh & (np.abs(b[:, 1]) < 1.5 + hy) & (gap > 0) & (gap < 80)
        vr = F["v"] - F["vel"][:, 0]
        m &= vr > 0.1
        if m.any():
            min_ttc = min(min_ttc, float(np.min(gap[m] / vr[m])))
    return min_ttc >= 1.0, min_ttc


def densify(P, step=0.5):
    seg = np.hypot(*np.diff(P, axis=0).T); s = np.r_[0, np.cumsum(seg)]
    if s[-1] < 1e-6:
        return P[:1], s[:1]
    q = np.arange(0, s[-1] + 1e-9, step)
    return np.c_[np.interp(q, s, P[:, 0]), np.interp(q, s, P[:, 1])], q


def station(sh, path_ln, s_end, end_xy, end_h, pt):
    """station of pt on the rear-axle path, straight extension beyond the end (A-6)."""
    st = path_ln.project(sh.Point(pt))
    if st >= s_end - 1e-6:
        d = (pt[0] - end_xy[0]) * math.cos(end_h) + (pt[1] - end_xy[1]) * math.sin(end_h)
        return s_end + max(d, 0.0)
    return st


def token_row(M, W, token, log, city):
    sh = M.sh
    P = np.array([[F["x"], F["y"]] for F in W]); V = np.array([F["v"] for F in W])
    D, sD = densify(P)
    r = dict(token=token, log=log, city=city)
    if len(D) < 2:
        path_ln = None
    else:
        path_ln = sh.LineString(D)
    s_fr = np.array([path_ln.project(sh.Point(p)) for p in P]) if path_ln is not None else np.zeros(len(P))
    s_end = float(sD[-1]); end_h = W[-1]["h"]
    # ---- connectors traversed end to end, ordered by entry station
    trav = []
    if path_ln is not None and s_end > 2.0:
        for k in M.tree.query(path_ln.buffer(COVER_TOL), predicate="intersects"):
            c = int(M.ids[k]); C = M.C[c]
            pts = sh.points(np.asarray([C["ln"].interpolate(d).coords[0] for d in np.arange(0, C["L"] + 1e-9, 1.0)]))
            if float(np.max(sh.distance(pts, path_ln))) > COVER_TOL:
                continue
            s0 = path_ln.project(sh.Point(C["start"])); s1 = path_ln.project(sh.Point(np.asarray(C["ln"].coords[-1])))
            if s1 > s0:
                trav.append((s0, s1, c))
    trav.sort()
    r["n_trav"] = len(trav)
    # ---- junction slots
    for slot in ["J-UL", "J-R", "J-S", "J-UL_geo_only", "J-R_geo_only", "J-S_geo_only"]:
        r[slot] = False
    lht = city in LHT_CITIES
    far, near = ("R", "L") if lht else ("L", "R")
    for s0, s1, c in trav:
        C = M.C[c]
        if C["itype"] not in JUNC_TYPES or C["geo"] not in ("L", "R", "S"):
            continue
        # legality: the connector is not red at the frame the front bumper reaches its entry
        k_in = int(np.argmax(s_fr + RA2F >= s0)) if np.any(s_fr + RA2F >= s0) else None
        if k_in is None or W[k_in]["tl"].get(c) is True:
            continue
        kind = "far" if C["geo"] == far else "near" if C["geo"] == near else "S"
        conf = M.conflicts(c)
        if kind == "far":
            want = [o for o in conf if o["oncoming"] and not o["merge"]]
        elif kind == "S":
            want = [o for o in conf if o["perpendicular"] and not o["merge"]]
        else:
            want = list(conf)
        hit = False
        for o in want:
            O = M.C[o["id"]]
            for F in W:
                m = np.isin(F["names"], ["vehicle", "bicycle"]) & (F["spd"] >= MOVING_V)
                m &= np.hypot(F["gx"] - o["part"].centroid.x, F["gy"] - o["part"].centroid.y) < 20.0
                for i in np.where(m)[0]:
                    b = F["box"][i]
                    if not box_poly(sh, F["gx"][i], F["gy"][i], F["gh"][i], b[3], b[4]).intersects(o["part"]):
                        continue
                    st = O["ln"].project(sh.Point(F["gx"][i], F["gy"][i]))
                    a = np.asarray(O["ln"].interpolate(max(st - 1, 0)).coords[0]); bb = np.asarray(O["ln"].interpolate(min(st + 1, O["L"])).coords[0])
                    hh = math.degrees(math.atan2(bb[1] - a[1], bb[0] - a[0]))
                    if abs(wrapd(math.degrees(F["gh"][i]) - hh)) <= ALIGN:
                        hit = True; break
                if hit:
                    break
            if hit:
                break
        # unprotected (far turn): some oncoming crossing connector is not red at entry
        unprot = any(W[k_in]["tl"].get(o["id"]) is not True for o in want) if kind == "far" else True
        tag = {"far": "J-UL", "near": "J-R", "S": "J-S"}[kind]
        if C["agree"]:
            r[tag + "_geo_only"] = True
            if kind == "near":
                r[tag] = True
                r["J-R_with_conflict_agent"] = r.get("J-R_with_conflict_agent", False) or hit
            elif hit and unprot:
                r[tag] = True
        r[f"{tag}_any_agree"] = r.get(f"{tag}_any_agree", False) or bool(C["agree"])
    # ---- governing connector
    listed = set().union(*[set(F["tl"]) for F in W])
    gov, gsrc = [], None
    for s0, s1, c in trav:
        if c in listed:
            gov, gsrc = [c], "traversed"; break
    if not gov:
        rb = W[NB]["rb"]
        for c in listed:
            C = M.C.get(c)
            if C is None or C["lgc"] not in rb:
                continue
            dx, dy = C["start"][0] - P[-1, 0], C["start"][1] - P[-1, 1]
            lon = dx * math.cos(end_h) + dy * math.sin(end_h); lat = -dx * math.sin(end_h) + dy * math.cos(end_h)
            if 0.0 <= lon <= 40.0 and abs(lat) <= COVER_TOL and abs(wrapd(C["h0"] - math.degrees(end_h))) <= 30.0:
                gov.append(c)
        gsrc = "route_ahead" if gov else None
    r["gov_src"] = gsrc
    r["TL-G"] = False
    for t in TLS_MIN_STOP:
        r[f"TL-S_{t:g}s"] = False
    r["TL-G_queued30"] = False; r["TL-S_5s_queued30"] = False
    if gov:
        state = np.array([(1 if any(F["tl"].get(g) is True for g in gov) else 3 if any(F["tl"].get(g) is False for g in gov) else 0) for F in W])
        sl = min(station(sh, path_ln, s_end, P[-1], end_h, M.C[g]["sl_pt"]) for g in gov) if path_ln is not None else math.nan
        r["sl_src"] = M.C[gov[0]]["sl_src"]
        front = s_fr + RA2F
        before = front <= sl + 1e-6
        cross_red = np.any((state == 1) & ~before)
        stopped = (V <= STOPPED_V) & before & (state == 1)
        r["cross_on_red"] = bool(cross_red)
        if stopped.any() and not cross_red:
            k1 = int(np.argmax(stopped)); gap = sl - front[k1]
            for cap, sfx in [(STOP_CAP, ""), (30.0, "_queued30")]:
                if gap > cap:
                    continue
                g2 = np.where((np.arange(len(W)) > k1) & (state == 3))[0]
                if g2.size:
                    k2 = int(g2[0])
                    if np.any((np.arange(len(W)) >= k2) & ~before) and V[k2:].max() > MOVING_V:
                        r["TL-G" + sfx] = True
                else:
                    # still red (or unlisted) to the window end, never crosses; stop duration = stopped frames before the line
                    if before[k1:].all() and not np.any(state[k1:] == 3):
                        run = 0; best = 0
                        for k in range(k1, len(W)):
                            run = run + 1 if V[k] <= STOPPED_V else 0; best = max(best, run)
                        dur = (best - 1) * 0.5 if best else 0.0
                        if sfx == "":
                            for t in TLS_MIN_STOP:
                                r[f"TL-S_{t:g}s"] = dur >= t
                        elif dur >= 5.0:
                            r["TL-S_5s_queued30"] = True
    return r


def process_log(args):
    log, toks = args
    out = dict(log=log, rows=[], skipped=collections.Counter(), status="ok")
    try:
        frames = pickle.load(open(f"{MF.LOG_DIR}/{log}.pkl", "rb"))
        city = frames[0]["map_location"]; M = get_map(city)
        idx = {f["token"]: i for i, f in enumerate(frames)}
        G = {}
        for tok in toks:
            j = idx.get(tok)
            if j is None:
                out["skipped"]["not_found"] += 1; continue
            if j - NB < 0 or j + NA_ >= len(frames):
                out["skipped"]["window_out_of_log"] += 1; continue
            ts = np.array([frames[k]["timestamp"] for k in range(j - NB, j + NA_ + 1)]) / 1e6
            if np.any(np.abs(np.diff(ts) - 0.5) > 0.05):
                out["skipped"]["window_dt_gap"] += 1; continue
            for k in range(j - NB, j + NA_ + 1):
                if k not in G:
                    G[k] = frame_globals(frames[k])
            W = [G[k] for k in range(j - NB, j + NA_ + 1)]
            r = token_row(M, W, tok, log, city)
            safe, ttc = human_safe(M.sh, W)
            r["human_safe"] = bool(safe); r["human_min_ttc_proxy"] = ttc
            out["rows"].append(r)
    except Exception as e:  # loud: reported per log
        import traceback
        out["status"] = f"error: {e!r}"; out["tb"] = traceback.format_exc()[-1500:]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=12); ap.add_argument("--limit-logs", type=int, default=0)
    a = ap.parse_args()
    logs, tokens = MF.parse_yaml()
    by = collections.defaultdict(list)
    import pandas as pd
    feat = pd.read_parquet(r"E:/Projects/TanitAD-artifacts/navtest_behaviour_mine_20261107/features.parquet", columns=["token", "log_name"])
    for t, l in zip(feat.token, feat.log_name):
        by[l].append(t)
    jobs = sorted(by.items())
    if a.limit_logs:
        jobs = jobs[:a.limit_logs]
    t0 = time.time(); res = []
    import multiprocessing as mp
    with mp.Pool(a.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(process_log, jobs), 1):
            res.append(r); print(f"[{n}/{len(jobs)}] {r['log']} {r['status']} rows={len(r['rows'])} {time.time()-t0:.0f}s", flush=True)
    rows = [x for r in res for x in r["rows"]]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "feasibility_rows.csv.gz"), index=False)
    skipped = collections.Counter(); [skipped.update(r["skipped"]) for r in res]
    rep = dict(n_tokens_in=sum(len(t) for _, t in jobs), n_rows=len(df), skipped=dict(skipped),
               log_errors={r["log"]: r["status"] for r in res if r["status"] != "ok"}, n_logs=len(jobs),
               human_safe=int(df.human_safe.sum()), seconds=round(time.time() - t0, 1), slots={})
    cols = [c for c in df.columns if c.startswith(("J-", "TL-"))]
    for c in sorted(cols):
        m = df[c].fillna(False).astype(bool) & df.human_safe
        rep["slots"][c] = dict(tokens=int(m.sum()), logs=int(df.loc[m, "log"].nunique()),
                               tokens_no_safety=int(df[c].fillna(False).astype(bool).sum()),
                               by_city=df.loc[m].groupby("city").size().to_dict())
    rep["gov_src"] = df.gov_src.fillna("none").value_counts().to_dict()
    rep["sl_src_among_TL"] = df.loc[df["TL-G"].fillna(False).astype(bool) | df["TL-S_5s"].fillna(False).astype(bool), "sl_src"].value_counts().to_dict()
    json.dump(rep, open(os.path.join(HERE, "feasibility_report.json"), "w"), indent=1, sort_keys=True, default=str)
    print(json.dumps(rep, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
