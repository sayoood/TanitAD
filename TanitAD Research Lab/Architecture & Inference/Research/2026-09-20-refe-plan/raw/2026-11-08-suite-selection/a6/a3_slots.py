"""AMENDMENT_3 slots J-UL, J-R, J-S, TL-G, TL-S for the suite selector (built from amendment3/feasibility_a3.py v2, sha256 in SUITE_PIN_A3.json).

Predicates are the feasibility script's, unchanged, on the 2 Hz OpenScene logs (AMENDMENT_3 section 2). Additions: (1) the ego / governing connector and the section 5 map
fields (stop_line_xy, conflict zone hull, conflict track tokens, signal heads and governs flags) are recorded per passing slot; (2) token list and logs are read here, no pandas;
(3) `run_a3(tokens)` is called by select_suite_tokens.py, which joins the DB-based human-safe flag. A3_MUTATE=invert_red is the deliberate-regression arm.
"""
import os, sys, re, json, math, glob, pickle, hashlib, argparse, collections, time
import numpy as np
SCORER_DIR = "E:/Projects/TanitAD-artifacts/refe_behaviour_20261107/code"
sys.path.insert(0, SCORER_DIR)
os.environ.setdefault("BEHAV_VENDOR_DIR", SCORER_DIR)
import behaviour_score as B

LOG_DIR = r"F:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test"
MAPS = r"F:/Archive/devbox-C/navsim/data/maps"

NB, NA_ = 4, 30            # window [t0-2 s, t0+15 s] at 2 Hz: 4 frames before t0, 30 after
RA2C, RA2F, EGO_L, EGO_W = 1.461, 4.049, 5.176, 2.297
COVER_TOL = 1.5            # m, connector baseline must lie within this of the human path end to end
TURN_L, TURN_S = 45.0, 20.0
JUNC_TYPES = {0, 1, 2}     # IntersectionType DEFAULT, TRAFFIC_LIGHT, STOP_SIGN (maps_datatypes.py:85-95)
TL_STOP_TYPE = 2           # StopLineType.TRAFFIC_LIGHT (maps_datatypes.py:61-70)
STOP_C2, STOP_C4, C4_LAUNCH, MOVING_V, ROLL_V = 0.1, 1.0, 1.0, 1.0, 3.0   # P[STOPPED_V], P[C4_STOP_V], P[C4_LAUNCH_V] of behaviour_score.py
GOV_TIE_M = 3.0   # connectors entered within 3 m of the first listed one are a tie, resolved by mean distance to the path
EXT_FRAMES = 60   # governing-connector search continues 30 s past the window
CROSS_LO, CROSS_HI, ONCOMING = 60.0, 120.0, 135.0
ALIGN = 45.0
MIN_OVERLAP_M2 = 0.5
STOP_CAP = 6.0
TLS_MIN_STOP = [3.0, 5.0, 8.0]
MUT_INVERT = os.environ.get("A3_MUTATE", "") == "invert_red"   # deliberate-regression arm only
LHT_CITIES = {"sg-one-north"}  # left-hand traffic


def wrapd(a):
    return (a + 180.0) % 360.0 - 180.0


class Map:
    def __init__(self, city):
        import pyogrio, shapely
        from shapely.strtree import STRtree
        self.sh = shapely
        g = glob.glob(f"{MAPS}/{city}/*/map.gpkg")[0]
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
            # AMENDMENT_6 ruling 1: the stop line is the START of the governing lane connector (first point of its baseline), for every token; no polygon, no exceptions
            sl_pt, sl_src = p(0), "connector_start"
            ids = [x for x in str(r.traffic_light_stop_line_fids or "").replace(" ", "").split(",") if x]
            cands = [sp.loc[int(x)] for x in ids if int(x) in sp.index and int(sp.loc[int(x)].stop_polygon_type_fid) == TL_STOP_TYPE]   # diagnostics only (n_tl_type2)
            self.C[fid] = dict(ln=ln, L=L, poly=poly[fid], h0=h0, h1=h1, dpsi=dpsi, geo=geo, tt=tt, agree=agree,
                               itype=itype.get(int(r.intersection_fid), -1) if r.intersection_fid == r.intersection_fid else -1,
                               exit_lane=r.exit_lane_fid, entry_lane=r.entry_lane_fid, lgc=int(r.lane_group_connector_fid),
                               sl_pt=sl_pt, sl_src=sl_src, start=p(0),
                               pts=self.sh.points(np.asarray([ln.interpolate(d).coords[0] for d in np.arange(0, L + 1e-9, 1.0)])),
                               n_tl_ids=len(ids), n_tl_type2=len(cands))
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


def frame_globals(f):  # (also keeps the 2 Hz track tokens in F['trk'])
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
                trk=list(a["track_tokens"]), tl={int(i): (bool(b) if not MUT_INVERT else not bool(b)) for i, b in f["traffic_lights"]},
                rb=set(int(i) for i in f["roadblock_ids"]), ts=int(f["timestamp"]))


def shapely_union(parts):
    import shapely
    return shapely.union_all(parts)


def box_poly(sh, x, y, h, L, W):
    c, s = math.cos(h), math.sin(h)
    dx = np.array([L, L, -L, -L]) / 2; dy = np.array([W, -W, -W, W]) / 2
    return sh.Polygon(np.c_[x + c * dx - s * dy, y + s * dx + c * dy])


def densify(P, step=0.5):
    seg = np.hypot(*np.diff(P, axis=0).T); s = np.r_[0, np.cumsum(seg)]
    if s[-1] < 1e-6:
        return P[:1], s[:1]
    q = np.arange(0, s[-1] + 1e-9, step)
    return np.c_[np.interp(q, s, P[:, 0]), np.interp(q, s, P[:, 1])], q


def ext_dir(path_xy, v, end_h):
    """AMENDMENT_6 ruling 2: direction of the straight end extension = chord over the last 5.0 m of arc length of the path's MOVING part (v >= 0.5), via the scorer's own
    function (behaviour_score._moving_chord_dir), never a stopped human's last heading.  Falls back to end_h only when the moving part has no length."""
    d = B._moving_chord_dir(path_xy[:, 0], path_xy[:, 1], v)
    return np.array([math.cos(end_h), math.sin(end_h)]) if d is None else d


def station(sh, path_xy, d_ext, pt):
    """station of pt on the rear-axle path, extended straight by 300 m beyond its end along d_ext (scorer A-6 as amended by AMENDMENT_6 ruling 2)."""
    tail = path_xy[-1] + 300.0 * np.asarray(d_ext)
    return float(sh.LineString(np.vstack([path_xy, tail])).project(sh.Point(pt)))


def traversals(M, path_ln):
    """connectors whose baseline lies within COVER_TOL of the path end to end; (s0, s1, c, mean distance), ordered by entry station."""
    sh = M.sh; out = []
    for k in M.tree.query(path_ln.buffer(COVER_TOL), predicate="intersects"):
        c = int(M.ids[k]); C = M.C[c]
        d = sh.distance(C["pts"], path_ln)
        if float(np.max(d)) > COVER_TOL:
            continue
        s0 = path_ln.project(sh.Point(C["start"])); s1 = path_ln.project(sh.Point(np.asarray(C["ln"].coords[-1])))
        if s1 > s0:
            out.append((s0, s1, c, float(np.mean(d))))
    out.sort(key=lambda z: (z[0], z[3], z[2]))
    return out


def token_row(M, W, Pext, token, log, city):
    sh = M.sh
    P = np.array([[F["x"], F["y"]] for F in W]); V = np.array([F["v"] for F in W])
    D, sD = densify(P)
    r = dict(token=token, log=log, city=city)
    path_ln = sh.LineString(D) if len(D) >= 2 else None
    s_fr = np.array([path_ln.project(sh.Point(p)) for p in P]) if path_ln is not None else np.zeros(len(P))
    s_end = float(sD[-1]); end_h = W[-1]["h"]
    trav = traversals(M, path_ln) if path_ln is not None and s_end > 2.0 else []
    r["n_trav"] = len(trav)
    # ---- junction slots
    lht = city in LHT_CITIES
    far, near = ("R", "L") if lht else ("L", "R")
    tags = {"far": "J-UL", "near": "J-R", "S": "J-S"}
    for t in tags.values():
        r[t] = False; r[t + "_c3_clean"] = False; r[t + "_conf_agent"] = False; r[t + "_geo"] = False; r[t + "_wide"] = False
    listed = set().union(*[set(F["tl"]) for F in W])
    for s0, s1, c, _ in trav:
        C = M.C[c]
        if C["itype"] not in JUNC_TYPES or C["geo"] not in ("L", "R", "S") or not C["agree"]:
            continue
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
        wide = [o for o in conf if not o["merge"] and o["rel"] >= CROSS_LO] if kind == "S" else want
        def _hits(lst, ids_out=None):
            f = np.zeros(len(W), bool)
            for o in lst:
                O = M.C[o["id"]]
                for fi, F in enumerate(W):
                    m = np.isin(F["names"], ["vehicle", "bicycle"]) & (F["spd"] >= MOVING_V)
                    m &= np.hypot(F["gx"] - o["part"].centroid.x, F["gy"] - o["part"].centroid.y) < 25.0
                    for i in np.where(m)[0]:
                        b = F["box"][i]
                        if not box_poly(sh, F["gx"][i], F["gy"][i], F["gh"][i], b[3], b[4]).intersects(o["part"]):
                            continue
                        st = O["ln"].project(sh.Point(F["gx"][i], F["gy"][i]))
                        a = np.asarray(O["ln"].interpolate(max(st - 1, 0)).coords[0]); bb = np.asarray(O["ln"].interpolate(min(st + 1, O["L"])).coords[0])
                        hh = math.degrees(math.atan2(bb[1] - a[1], bb[0] - a[0]))
                        if abs(wrapd(math.degrees(F["gh"][i]) - hh)) <= ALIGN:
                            f[fi] = True
                            if ids_out is not None:
                                ids_out.add(str(F["trk"][i]))
            return f
        hf = _hits(want)
        wf = _hits(wide) if kind == "S" else hf
        hit = bool(hf.any())
        tag = tags[kind]
        unprot = any(W[k_in]["tl"].get(o["id"]) is not True for o in want) if kind == "far" else True
        if kind == "far":
            r["J-UL_hit"] = r.get("J-UL_hit", False) or hit
            r["J-UL_unprot"] = r.get("J-UL_unprot", False) or unprot
            nm = [o for o in conf if not o["merge"]]
            onc120 = [o for o in conf if not o["merge"] and o["rel"] >= 120.0]
            if nm:
                h2 = bool(_hits(nm).any())
                u2 = any(W[k_in]["tl"].get(o["id"]) is not True for o in nm)
                r["J-UL_altA_anycross"] = r.get("J-UL_altA_anycross", False) or (h2 and u2)
            if onc120:
                h3 = bool(_hits(onc120).any())
                u3 = any(W[k_in]["tl"].get(o["id"]) is not True for o in onc120)
                r["J-UL_altB_rel120"] = r.get("J-UL_altB_rel120", False) or (h3 and u3)
        # human c3 at 2 Hz (ego entering the zone while a conflict agent is in it within 2 s)
        zone = shapely_union([o["part"] for o in want]) if want else None
        clean = True
        if zone is not None and hit:
            ego_in = [box_poly(sh, F["x"] + RA2C * math.cos(F["h"]), F["y"] + RA2C * math.sin(F["h"]), F["h"], EGO_L, EGO_W).intersects(zone) for F in W]
            for e in range(1, len(W)):
                if ego_in[e] and not ego_in[e - 1] and hf[e:e + 5].any():
                    clean = False
            r[tag + "_zone_parts"] = max(r.get(tag + "_zone_parts", 0), len(getattr(zone, "geoms", [zone])))
        r[tag + "_geo"] = True
        if (kind == "near" or (hit and unprot)) and (tag + "_cid") not in r:
            r[tag + "_cid"] = int(c)
            # section 5: S = X_o (J-UL), X_p (J-S), all of X(c) (J-R); X(c) excludes merges (section 2)
            S_set = want if kind != "near" else [o for o in conf if not o["merge"]]
            cids = set()
            Sf = _hits(S_set, cids) if S_set else np.zeros(len(W), bool)
            zone_xy = []
            if S_set and Sf.any():
                hull = shapely_union([C["poly"].intersection(o["part"]) for o in S_set]).convex_hull
                if hull.geom_type == "Polygon" and hull.area > 0:
                    from shapely.geometry import polygon as _sp
                    xy = np.asarray(_sp.orient(hull, 1.0).exterior.coords)[:-1]
                    zone_xy = [[float(x), float(y)] for x, y in xy]
                else:
                    zone_xy = None   # degenerate hull: reported, not silently emitted
            heads = sorted(int(h) for h in listed)
            r["F_" + tag] = dict(connector=int(c), stop_line_xy=[float(C["sl_pt"][0]), float(C["sl_pt"][1])], stop_line_source=C["sl_src"],
                                 conflict_set=[int(o["id"]) for o in S_set], conflict_zone_xy=zone_xy,
                                 conflict_agent_ids=(sorted(cids) if zone_xy else []), signal_heads=heads,
                                 signal_governs_ego=[bool(h == c) for h in heads], ego_connector_listed=bool(c in listed))
        if kind == "near":
            r[tag] = True; r[tag + "_conf_agent"] = r[tag + "_conf_agent"] or hit
            if hit and clean:
                r[tag + "_c3_clean"] = True
            r[tag + "_itype"] = C["itype"]; r[tag + "_sig"] = bool(c in listed)
        elif hit and unprot:
            r[tag] = True; r[tag + "_itype"] = C["itype"]; r[tag + "_sig"] = bool(c in listed)
            if clean:
                r[tag + "_c3_clean"] = True
        if kind == "S" and bool(wf.any()):
            r[tag + "_wide"] = True
    # ---- governing connector: first listed connector traversed on the human path, extended up to 30 s past the window
    Dx, sx = densify(Pext)
    ext_ln = sh.LineString(Dx) if len(Dx) >= 2 else None
    tx = traversals(M, ext_ln) if ext_ln is not None and float(sx[-1]) > 2.0 else []
    cand = [t for t in tx if t[2] in listed]
    gov, gsrc = [], None
    if cand:
        s_first = cand[0][0]
        pool = [t for t in cand if t[0] <= s_first + GOV_TIE_M]
        pool.sort(key=lambda z: (round(z[3], 3), z[2]))
        gov, gsrc = [pool[0][2]], "traversed_window" if any(t[2] == pool[0][2] for t in trav) else "traversed_ext"
        r["gov_tied"] = len(pool)
    else:
        rb = W[NB]["rb"]
        for c in sorted(listed):
            C = M.C.get(c)
            if C is None or C["lgc"] not in rb:
                continue
            dx, dy = C["start"][0] - P[-1, 0], C["start"][1] - P[-1, 1]
            lon = dx * math.cos(end_h) + dy * math.sin(end_h); lat = -dx * math.sin(end_h) + dy * math.cos(end_h)
            if 0.0 <= lon <= 40.0 and abs(lat) <= COVER_TOL and abs(wrapd(C["h0"] - math.degrees(end_h))) <= 30.0:
                gov.append(c)
        gsrc = "route_ahead" if gov else None
    r["gov_src"] = gsrc
    r["gov_id"] = gov[0] if gov else -1
    if gov:
        heads = sorted(int(h) for h in listed)
        r["F_TL"] = dict(connector=int(gov[0]), stop_line_xy=[float(M.C[gov[0]]["sl_pt"][0]), float(M.C[gov[0]]["sl_pt"][1])], stop_line_source=M.C[gov[0]]["sl_src"],
                         conflict_set=[], conflict_zone_xy=[], conflict_agent_ids=[], signal_heads=heads, signal_governs_ego=[bool(h in gov) for h in heads],
                         governing_source=gsrc)
    r["TL-G_c2code"] = False
    names = ["TL-G", "TL-G_strict", "TL-G_queued30", "TL-G_roll"] + [f"TL-S_{t:g}s" for t in TLS_MIN_STOP] + [f"TL-S_{t:g}s_roll" for t in TLS_MIN_STOP] + ["TL-S_5s_queued30", "TL-S_5s_red90", "TL-G_route", "TL-S_5s_route"]
    for n in names:
        r[n] = False
    if gov and path_ln is not None:
        state = np.array([(1 if any(F["tl"].get(g) is True for g in gov) else 3 if any(F["tl"].get(g) is False for g in gov) else 0) for F in W])
        # A6: the TL stations use the path through the TRUE last window pose (densify() can stop up to 0.5 m short of it); the junction slots keep their path untouched
        Dtl = np.vstack([D, P[-1]]) if float(np.hypot(*(D[-1] - P[-1]))) > 1e-6 else D
        tl_ln = sh.LineString(Dtl)
        s_fr_tl = np.array([tl_ln.project(sh.Point(p)) for p in P])
        dx_ = ext_dir(P, V, end_h)
        r["ext_dir"] = [float(dx_[0]), float(dx_[1])]
        sl = min(station(sh, Dtl, dx_, M.C[g]["sl_pt"]) for g in gov)
        r["sl_src"] = M.C[gov[0]]["sl_src"]
        front = s_fr_tl + RA2F
        before = front <= sl + 1e-6
        cross_red = bool(np.any(before[:-1] & ~before[1:] & (state[1:] == 1)))   # crossing event at a red frame
        past_red = bool(np.any((state == 1) & ~before))                            # c2 as currently coded: any red frame beyond the line
        r["cross_on_red"] = cross_red; r["past_line_red"] = past_red
        stopped = (V <= STOP_C2) & before & (state == 1)
        idx = np.arange(len(W))
        route = gsrc == "route_ahead"
        r["tl_stage"] = "cross_red" if cross_red else ("no_stop_at_red" if not stopped.any() else "stopped")
        if stopped.any() and not cross_red:
            k1 = int(np.argmax(stopped)); gap = float(sl - front[k1]); r["stop_gap_m"] = gap
            r["tls_k1"] = k1
            roll = bool(V[:k1 + 1].max() >= ROLL_V)
            for cap, sfx in [(STOP_CAP, ""), (30.0, "_queued30")]:
                if gap > cap:
                    continue
                g2 = np.where((idx > k1) & (state == 3))[0]
                if sfx == "":
                    r["tl_stage"] = "green_later" if g2.size else ("tls_cand" if before[k1:].all() else "not_before_after_k1")
                    if not g2.size:
                        r["tls_frac"] = float(np.mean(state[k1:] == 1)); r["tls_absent"] = float(np.mean(state[k1:] == 0))
                if g2.size:
                    k2 = int(g2[0])
                    j = k2 - 1; run = 0
                    while j >= 0 and V[j] <= STOP_C4:
                        run += 1; j -= 1
                    held = run * 0.5 >= 1.0
                    proceed = bool(np.any(~before[k2:])) and float(V[k2:].max()) > C4_LAUNCH
                    if proceed and held:
                        if route:
                            r["TL-G_route"] = True; continue
                        r["TL-G" + sfx] = True
                        if sfx == "" and not past_red:
                            r["TL-G_c2code"] = True
                        if sfx == "" and np.all(state[k1:k2] == 1):
                            r["TL-G_strict"] = True
                        if sfx == "" and roll:
                            r["TL-G_roll"] = True
                elif before[k1:].all():
                    st = state[k1:]
                    frac = float(np.mean(st == 1))
                    run = best = 0
                    for k in range(k1, len(W)):
                        run = run + 1 if V[k] <= STOP_C2 else 0; best = max(best, run)
                    dur = (best - 1) * 0.5 if best else 0.0
                    if sfx == "":
                        r["tls_dur"] = dur
                    if route:
                        if sfx == "" and frac >= 0.9 and dur >= 5.0:
                            r["TL-S_5s_route"] = True
                        continue
                    if sfx == "":
                        for t in TLS_MIN_STOP:
                            if frac == 1.0 and dur >= t:
                                r[f"TL-S_{t:g}s"] = True
                                if roll:
                                    r[f"TL-S_{t:g}s_roll"] = True
                        if frac >= 0.9 and dur >= 5.0:
                            r["TL-S_5s_red90"] = True
                    elif frac == 1.0 and dur >= 5.0:
                        r["TL-S_5s_queued30"] = True
    return r


def process_log(args):
    log, tokset = args
    out = dict(log=log, rows=[], skipped=collections.Counter(), status="ok")
    try:
        frames = pickle.load(open(f"{LOG_DIR}/{log}.pkl", "rb"))
        city = frames[0]["map_location"]; M = get_map(city)
        idx = {f["token"]: i for i, f in enumerate(frames)}
        toks = sorted(t for t in tokset if t in idx)
        G = {}
        for tok in toks:
            j = idx[tok]
            if j - NB < 0 or j + NA_ >= len(frames):
                out["skipped"]["window_out_of_log"] += 1; continue
            ts = np.array([frames[k]["timestamp"] for k in range(j - NB, j + NA_ + 1)]) / 1e6
            if np.any(np.abs(np.diff(ts) - 0.5) > 0.05):
                out["skipped"]["window_dt_gap"] += 1; continue
            for k in range(j - NB, j + NA_ + 1):
                if k not in G:
                    G[k] = frame_globals(frames[k])
            W = [G[k] for k in range(j - NB, j + NA_ + 1)]
            last = j + NA_
            while last + 1 < len(frames) and last + 1 <= j + NA_ + EXT_FRAMES and abs((frames[last + 1]["timestamp"] - frames[last]["timestamp"]) / 1e6 - 0.5) <= 0.05:
                last += 1
            Pext = np.array([[frames[k]["ego2global"][0][3], frames[k]["ego2global"][1][3]] for k in range(j - NB, last + 1)], float)
            r = token_row(M, W, Pext, tok, log, city)
            r["openscene_list_idx"] = int(j)
            out["rows"].append(r)
    except Exception as e:  # loud: reported per log
        import traceback
        out["status"] = f"error: {e!r}"; out["tb"] = traceback.format_exc()[-1500:]
    return out


# slot -> (row flag predicate, field key)
SLOTS_A3 = ["J-UL", "J-R", "J-S", "TL-G", "TL-S"]


def slot_pass(r, slot):
    """AMENDMENT_3 section 3-4 predicates as the feasibility script computed them (human safe is joined by the caller)."""
    if slot == "TL-G":
        return bool(r.get("TL-G")) and bool(r.get("TL-G_c2code"))        # c2 as coded today (the human passes c2): the section 4 'no red frame past the line' clause
    if slot == "TL-S":
        return bool(r.get("TL-S_5s_roll"))                                  # 5.0 s run, all red from k1, rolling approach
    return bool(r.get(slot))


def slot_fields(r, slot):
    return r.get("F_TL") if slot.startswith("TL") else r.get("F_" + slot)


def run_a3(tokset, workers=12):
    """-> (rows, report). rows carry the slot flags and the section 5 fields; human safe is NOT joined here."""
    import multiprocessing as mp
    by_log = sorted(f[:-4] for f in os.listdir(LOG_DIR) if f.endswith(".pkl"))
    jobs = [(l, tokset) for l in by_log]
    t0 = time.time(); res = []
    with mp.Pool(workers) as pool:
        for n, r in enumerate(pool.imap_unordered(process_log, jobs), 1):
            res.append(r); print(f"[a3 {n}/{len(jobs)}] {r['log']} {r['status']} rows={len(r['rows'])} {time.time() - t0:.0f}s", flush=True)
    rows = sorted((x for r in res for x in r["rows"]), key=lambda x: x["token"])
    skipped = collections.Counter(); [skipped.update(r["skipped"]) for r in res]
    errs = {r["log"]: r["status"] for r in res if r["status"] != "ok"}
    return rows, dict(n_rows=len(rows), skipped=dict(skipped), log_errors=errs, seconds=round(time.time() - t0, 1), mutate=os.environ.get("A3_MUTATE", ""))
