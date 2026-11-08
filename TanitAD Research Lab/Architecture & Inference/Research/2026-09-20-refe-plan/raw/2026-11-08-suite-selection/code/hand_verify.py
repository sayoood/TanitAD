#!/usr/bin/env python3
"""Hand-verification of ONE candidate from the RAW nuPlan DB with an INDEPENDENT implementation (shapely geometry + plain sqlite rows; no code shared with
select_suite_tokens.py's geometry, projection, SAT or kinematics).  It prints the numbers a reader can check by eye and, last, compares the lead id and gap at all
171 ticks against select_suite_tokens.analyze (the only place the two meet).

    python hand_verify.py --token <t> --slot cruise|steady_lead|lead_braking|stop_go|merge_cutin [--alt]
"""
import argparse, math, os, sqlite3, sys, json, pickle
import numpy as np
from shapely.geometry import LineString, Polygon, Point

DB_DIR = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test"
FRONT, REAR, EGO_W = 4.049, 1.127, 2.297
HALF = 1.5


def box_poly(x, y, yaw, w, l):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for sl, sw in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        pts.append((x + sl * l / 2 * c - sw * w / 2 * s, y + sl * l / 2 * s + sw * w / 2 * c))
    return Polygon(pts), pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", required=True)
    ap.add_argument("--slot", required=True)
    ap.add_argument("--alt", action="store_true")
    ap.add_argument("--results", default="E:/Projects/TanitAD-artifacts/tsf_bundle2_20261107/run2/per_token_results.pkl")
    ap.add_argument("--no-compare", action="store_true")
    a = ap.parse_args()
    res = pickle.load(open(a.results, "rb"))["rows"]
    row = next(r for r in res if r["token"] == a.token)
    log = row["log"]
    con = sqlite3.connect(f"file:{DB_DIR}/{log}.db?mode=ro", uri=True)
    cur = con.cursor()
    lp = cur.execute("select hex(token), timestamp, hex(ego_pose_token), token from lidar_pc order by timestamp").fetchall()
    r0 = [i for i, r in enumerate(lp) if r[0].lower() == a.token][0]
    idx = [r0 + 2 * k for k in range(-20, 151)]
    T = np.array([lp[i][1] for i in idx]) / 1e6
    ego = []
    for i in idx:
        x, y, qw, qx, qy, qz, vx, vy = cur.execute("select x,y,qw,qx,qy,qz,vx,vy from ego_pose where hex(token)=?", (lp[i][2],)).fetchone()
        ego.append((x, y, math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)), math.hypot(vx, vy)))
    ego = np.array(ego)
    print(f"token {a.token} log {log} lidar_pc rank {r0}  window ticks {len(idx)}  t_span {T[-1]-T[0]:.3f} s")
    cat = {bytes(t): n for t, n in cur.execute("select track.token, category.name from track join category on track.category_token=category.token")}
    toks = [lp[i][3] for i in idx]
    boxes = {}
    for k, tk in enumerate(toks):
        boxes[k] = cur.execute("select track_token, x, y, yaw, width, length from lidar_box where lidar_pc_token=?", (tk,)).fetchall()

    P = [(e[0], e[1]) for e in ego]
    end_dir = (math.cos(ego[-1, 2]), math.sin(ego[-1, 2]))
    path = LineString(P + [(P[-1][0] + end_dir[0] * 200, P[-1][1] + end_dir[1] * 200)])
    corridor = path.buffer(HALF, cap_style="flat")
    s_ego = np.array([path.project(Point(p)) for p in P])
    front = s_ego + FRONT

    # track positions by tick
    pos = {}
    for k in range(171):
        for (tt, x, y, yaw, w, l) in boxes[k]:
            pos.setdefault(bytes(tt), {})[k] = (x, y)

    def spd(trk, k):
        a_, b_ = max(0, k - 5), min(170, k + 5)
        pa, pb = pos[trk].get(a_), pos[trk].get(b_)
        if pa is None or pb is None or a_ == b_:
            return float("nan")
        return math.hypot(pb[0] - pa[0], pb[1] - pa[1]) / (T[b_] - T[a_])

    def acc(trk, k):
        if k < 5 or k > 165:
            return float("nan")
        return (spd(trk, k + 5) - spd(trk, k - 5)) / (T[k + 5] - T[k - 5])

    lead = []
    for k in range(171):
        best = None
        for (tt, x, y, yaw, w, l) in boxes[k]:
            if cat.get(bytes(tt)) != "vehicle":
                continue
            poly, pts = box_poly(x, y, yaw, w, l)
            if not poly.intersects(corridor) or poly.intersection(corridor).area < 1e-6:
                continue
            ss = [path.project(Point(p)) for p in pts]
            if max(ss) <= front[k]:
                continue
            gap = max(0.0, min(ss) - front[k])
            if best is None or gap < best[0]:
                best = (gap, bytes(tt), path.project(Point(x, y)))
        lead.append(best)

    def fmt(k):
        L = lead[k]
        if L is None:
            return f"t={(k-20)*0.1:+5.1f}s ego v={ego[k,3]:5.2f}  no lead"
        v = spd(L[1], k)
        return f"t={(k-20)*0.1:+5.1f}s ego v={ego[k,3]:5.2f}  lead {L[1].hex()[:8]} gap={L[0]:6.2f} m  lead v={v:5.2f} a={acc(L[1],k):+5.2f}"

    S = row[a.slot + ("_alt" if a.alt and a.slot != "cruise" else "")]
    print("stored predicate values (main script):", json.dumps(S))
    ticks_print = list(range(0, 171, 10))
    sl = a.slot
    if sl == "cruise":
        v = ego[:, 3]
        med = np.median(v)
        print(f"ego speed every 1 s: {np.round(v[::10], 2).tolist()}")
        print(f"median v={med:.3f}; max |v/med-1| over ALL 171 ticks = {np.max(np.abs(v/med-1)):.4f} (rule <= 0.15)")
        n_lead = sum(L is not None for L in lead)
        print(f"ticks with ANY vehicle lead in the 1.5 m corridor (independent shapely): {n_lead} of 171 (cruise needs free road at every tick)")
        far = [(k, lead[k][0]) for k in range(171) if lead[k] is not None]
        print("  leads (tick, gap):", far[:6])
        yaw = np.unwrap(ego[:, 2])
        kap = [(yaw[min(170, k + 5)] - yaw[max(0, k - 5)]) / (s_ego[min(170, k + 5)] - s_ego[max(0, k - 5)]) for k in range(171)]
        print(f"|curvature| max over ticks = {np.nanmax(np.abs(kap)):.5f} (rule <= 0.02); omega_z/v cross-check not available (ego_pose angular rate unused)")
        print(f"min TTC: no lead => infinite; in cruise list: {row['cruise']['token_in_cruise_list']}")
    else:
        for k in ticks_print:
            print("  " + fmt(k))
    if sl == "steady_lead":
        tr = S["lead_track"]
        runs = [k for k in range(171) if lead[k] is not None and lead[k][1].hex() and (not np.isnan(acc(lead[k][1], k)))]
        a0 = int(round(S["run_start_rel_t0_s"] * 10)) + 20
        b0 = int(round(S["run_end_rel_t0_s"] * 10)) + 20
        ok = all(lead[k] is not None and lead[k][1] == lead[a0][1] and lead[k][0] <= 60 and spd(lead[k][1], k) >= 3 and abs(acc(lead[k][1], k)) <= 1.0 for k in range(a0, b0 + 1))
        print(f"independent re-check of the reported run ticks {a0}..{b0} ({(b0-a0)*0.1:.1f} s): same lead id, gap<=60, v>=3, |a|<=1 at every tick -> {ok}")
        print("  gap range %.2f..%.2f  lead v range %.2f..%.2f  max|a| %.3f" % (min(lead[k][0] for k in range(a0, b0 + 1)), max(lead[k][0] for k in range(a0, b0 + 1)),
              min(spd(lead[k][1], k) for k in range(a0, b0 + 1)), max(spd(lead[k][1], k) for k in range(a0, b0 + 1)), max(abs(acc(lead[k][1], k)) for k in range(a0, b0 + 1))))
    if sl == "lead_braking":
        k0 = int(round(S["onset_rel_t0_s"] * 10)) + 20
        k1 = int(round(S["t_end_rel_t0_s"] * 10)) + 20
        L = lead[k0]
        trk = L[1]
        print(f"onset tick {k0} (t={S['onset_rel_t0_s']}): lead {trk.hex()[:8]} gap {L[0]:.2f} (<=50)  v_onset={spd(trk,k0):.2f}  v_end(tick {k1})={spd(trk,k1):.2f}  drop={spd(trk,k0)-spd(trk,k1):.2f} (>=4) over {(k1-k0)*0.1:.1f} s (<=6)")
        print("  lead speed each 0.5 s from onset:", [round(spd(trk, k), 2) for k in range(k0, min(171, k1 + 6), 5)])
        print("  peak decel over [onset,end]: %.3f (>=1.5)" % (-np.nanmin([acc(trk, k) for k in range(k0, k1 + 1)])), " ego v0 =", round(ego[20, 3], 2), "(>=5)")
    if sl == "stop_go":
        f = int(round(S["t_fall_rel_t0_s"] * 10)) + 20
        g = int(round(S["t_go_rel_t0_s"] * 10)) + 20
        trk = lead[f][1]
        print(f"lead {trk.hex()[:8]}: v(f-1)={spd(trk,f-1):.2f} (>=0.5), v(f)={spd(trk,f):.2f} (<0.5) at t={S['t_fall_rel_t0_s']}; v(go)={spd(trk,g):.2f} (>2) at t={S['t_go_rel_t0_s']}")
        same = all(lead[k] is not None and lead[k][1] == trk for k in range(f - 1, g + 1))
        print("  same vehicle is THE lead at every tick from f-1 to go:", same, "  lead speed each 1 s from f-1:", [round(spd(trk, k), 2) for k in range(f - 1, g + 1, 10)])
    if sl == "merge_cutin":
        def signed_lat(x, y):
            sc = path.project(Point(x, y))
            pp = path.interpolate(sc)
            pq = path.interpolate(min(sc + 0.5, path.length))
            sign = 1.0 if ((pq.x - pp.x) * (y - pp.y) - (pq.y - pp.y) * (x - pp.x)) >= 0 else -1.0
            return sign * pp.distance(Point(x, y))

        found = []
        for trk, series in pos.items():
            if cat.get(trk) != "vehicle" or 20 not in series:
                continue
            l0 = signed_lat(*series[20])
            if abs(l0) < 2.0:
                continue
            for k in range(21, 171):
                if k not in series:
                    continue
                if abs(signed_lat(*series[k])) > 0.5:
                    continue
                bx = [b for b in boxes[k] if bytes(b[0]) == trk][0]
                _, pts = box_poly(bx[1], bx[2], bx[3], bx[4], bx[5])
                ss = [path.project(Point(p)) for p in pts]
                gap = max(0.0, min(ss) - front[k])
                if max(ss) > front[k] and gap <= 30:
                    found.append((k, trk, l0, signed_lat(*series[k]), gap))
                    break
        found.sort()
        print(f"independent search: {len(found)} vehicle(s) with |lateral(t0)|>=2.0 that later enter |lateral|<=0.5 at bumper gap<=30 m")
        for (k, trk, l0, le, gap) in found[:3]:
            print(f"  vehicle {trk.hex()[:8]}: lateral at t0 {l0:+.2f} m -> {le:+.2f} m at t={(k-20)*0.1:+.1f}s, bumper gap {gap:.2f} m, speed {spd(trk,k):.2f} m/s")
        print(f"  stored: entry t={S['t_entry_rel_t0_s']}, gap {S['gap_entry_m']}, lat_t0 {S['lat_t0_m']}, lat_entry {S['lat_entry_m']}, heading diff {S['heading_diff_to_path_deg']} deg")
    # safety
    ttcs = []
    for k in range(171):
        L = lead[k]
        if L is None:
            continue
        sc = L[2]
        v = spd(L[1], k)
        # along-path velocity component
        a_, b_ = max(0, k - 5), min(170, k + 5)
        pa, pb = pos[L[1]].get(a_), pos[L[1]].get(b_)
        if pa is None or pb is None:
            continue
        vv = ((pb[0] - pa[0]) / (T[b_] - T[a_]), (pb[1] - pa[1]) / (T[b_] - T[a_]))
        p0 = path.interpolate(sc)
        p1 = path.interpolate(min(sc + 0.5, path.length))
        tg = (p1.x - p0.x, p1.y - p0.y)
        n = math.hypot(*tg)
        u = (vv[0] * tg[0] + vv[1] * tg[1]) / n
        vrel = ego[k, 3] - u
        ttcs.append(L[0] / vrel if vrel > 0.1 else float("inf"))
    print(f"independent human min TTC over the window = {min(ttcs) if ttcs else float('inf'):.2f} s (stored {row['min_ttc_s']}; rule >= 1.0)")
    # ego-box overlap with any tracked object (shapely)
    ov = 0
    for k in range(171):
        c, s = math.cos(ego[k, 2]), math.sin(ego[k, 2])
        cx, cy = ego[k, 0] + c * (FRONT - REAR) / 2, ego[k, 1] + s * (FRONT - REAR) / 2
        ep, _ = box_poly(cx, cy, ego[k, 2], EGO_W, FRONT + REAR)
        for (tt, x, y, yaw, w, l) in boxes[k]:
            if math.hypot(x - cx, y - cy) > 15:
                continue
            bp, _ = box_poly(x, y, yaw, w, l)
            if ep.intersection(bp).area > 0:
                ov += 1
                break
    print(f"independent overlap ticks (ego box vs any tracked object) = {ov} (stored {row['overlap_ticks']}; rule 0)")

    if not a.no_compare:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import select_suite_tokens as M
        ctx = M.load_log(f"{DB_DIR}/{log}.db")
        dbg = {}
        M.analyze(ctx, ctx.tok2rank[a.token], a.token in set(json.load(open(M.CRUISE_JSON))["tokens"]), dbg)
        n_same_id = n_both = 0
        gdiff = []
        for k in range(171):
            mi = dbg["lead_trk"][k]
            mh = None if mi < 0 else bytes.fromhex(dbg["track_hex"][mi])
            ih = None if lead[k] is None else lead[k][1]
            if mh == ih:
                n_same_id += 1
                if ih is not None:
                    n_both += 1
                    gdiff.append(abs(dbg["lead_gap"][k] - lead[k][0]))
        print(f"MAIN vs INDEPENDENT over 171 ticks: same lead id at {n_same_id}/171 ticks; where both have a lead ({n_both}): gap |diff| median {np.median(gdiff) if gdiff else 0:.3f} m, max {max(gdiff) if gdiff else 0:.3f} m")
        print(f"  ego speed |diff| max {np.max(np.abs(dbg['ev'] - ego[:, 3])):.2e}; s_ego max |diff| {np.max(np.abs(dbg['s_ego'] - s_ego)):.3f} m")


if __name__ == "__main__":
    main()
