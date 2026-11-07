#!/usr/bin/env python
"""MINER: per-token behaviour feature table over the NAVSIM navtest split (CPU only).

Run:  C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I mine_features.py [--limit-logs N] [--workers W]
Reads : OpenScene metadata pkls (navsim_logs/test) + navtest.yaml (136 logs, 12,146 tokens) + nuplan maps.
Writes: features.parquet, features.csv.gz, run_report.json  (into OUT).
Builds FEATURES only. It does not select tokens. Definitions: see SCHEMA.md.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys, re, json, pickle, math, time, glob, hashlib, argparse, traceback
import numpy as np
import pandas as pd

LOG_DIR = r"F:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test"
YAML = (r"F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896/navsim/planning/"
        r"script/config/common/train_test_split/scene_filter/navtest.yaml")
MAPS = r"F:/Archive/devbox-C/navsim/data/maps"
OUT = r"E:/Projects/TanitAD-artifacts/navtest_behaviour_mine_20261107"

# ---- constants (all documented in SCHEMA.md) -------------------------------------------------------
NH, NF = 4, 10            # navtest.yaml: num_history_frames, num_future_frames (token = frame index NH-1 of a 14-frame window)
H = 8                     # evaluation horizon in frames = 4.0 s at 2 Hz (NAVSIM trajectory: 8 poses at 0.5 s)
DT = 0.5
EGO_FRONT = 4.049         # m, rear axle -> front bumper (nuplan get_pacifica_parameters().front_length)
CORR_HALF_W = 1.8         # m, corridor half width (ego frame |y|)
CORR_MAX_X = 60.0         # m, corridor length
STOP_V = 0.5              # m/s
START_V = 1.5             # m/s
ONCOMING_VX = -1.0        # m/s, agent velocity along ego x below this => oncoming
CUTIN_MIN_SPD = 1.0       # m/s
CUTIN_SIDE_MAX_Y = 8.0    # m
VRU_MAX_X = 30.0          # m
TL_AHEAD_MAX_X = 60.0
TL_AHEAD_MAX_Y = 3.5
TL_AHEAD_MAX_DHEAD = math.radians(35.0)
JUNC_PROBE_STEP = 5.0
JUNC_PROBE_MAX = 60.0
CITIES = ["us-nv-las-vegas-strip", "us-ma-boston", "us-pa-pittsburgh-hazelwood", "sg-one-north"]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def parse_yaml():
    logs, tokens, sec = [], [], None
    for line in open(YAML, "r", encoding="utf-8"):
        s = line.rstrip("\n")
        if s.startswith("log_names:"):
            sec = "log"; continue
        if s.startswith("tokens:"):
            sec = "tok"; continue
        m = re.match(r"^\s+-\s+'([^']+)'\s*$", s)
        if m and sec == "log":
            logs.append(m.group(1))
        elif m and sec == "tok":
            tokens.append(m.group(1))
    return logs, tokens


# ---- map ---------------------------------------------------------------------------------------------
class CityMap:
    def __init__(self, city):
        import pyogrio, shapely
        from shapely.strtree import STRtree
        g = glob.glob(f"{MAPS}/{city}/*/map.gpkg")[0]
        meta = pyogrio.read_dataframe(g, layer="meta")
        proj = meta[meta["key"] == "projectedCoordSystem"]["value"].iloc[0]
        inter = pyogrio.read_dataframe(g, layer="intersections", fid_as_index=True).to_crs(proj)
        lc = pyogrio.read_dataframe(g, layer="lane_connectors", fid_as_index=True).to_crs(proj)
        gen = pyogrio.read_dataframe(g, layer="gen_lane_connectors_scaled_width_polygons").to_crs(proj)
        self.shapely = shapely
        self.inter_n = len(inter)
        self.junc_geoms = np.array(list(inter.geometry) + list(gen.geometry), dtype=object)
        self.junc_tree = STRtree(self.junc_geoms)
        # lane-connector polygons keyed by connector id (== lane_connectors fid == traffic_lights id)
        self.conn_poly_ids = gen["lane_connector_fid"].astype(int).to_numpy()
        self.conn_polys = np.array(list(gen.geometry), dtype=object)
        self.conn_tree = STRtree(self.conn_polys)
        # entry point + entry heading of each connector (baseline start, heading over first 5 m)
        self.entry = {}
        for fid, geom in zip(lc.index.astype(int), lc.geometry):
            try:
                p0 = geom.interpolate(0.0)
                p1 = geom.interpolate(min(5.0, geom.length))
                self.entry[int(fid)] = (p0.x, p0.y, math.atan2(p1.y - p0.y, p1.x - p0.x))
            except Exception:
                pass

    def in_junction(self, xs, ys):
        pts = self.shapely.points(np.asarray(xs), np.asarray(ys))
        hit = self.junc_tree.query(pts, predicate="within")
        out = np.zeros(len(pts), bool)
        out[np.unique(hit[0])] = True
        return out

    def connectors_hit(self, xs, ys):
        pts = self.shapely.points(np.asarray(xs), np.asarray(ys))
        hit = self.conn_tree.query(pts, predicate="within")
        return set(int(self.conn_poly_ids[i]) for i in np.unique(hit[1]))


# ---- per-frame precompute ----------------------------------------------------------------------------
def frame_info(f):
    """Everything derived from one frame, in that frame's own ego frame (rear axle origin)."""
    R = np.asarray(f["ego2global"], float)
    pose = (R[0, 3], R[1, 3], math.atan2(R[1, 0], R[0, 0]))
    eds = f["ego_dynamic_state"]
    vx, vy, ax, ay = float(eds[0]), float(eds[1]), float(eds[2]), float(eds[3])
    a = f["anns"]
    names = np.asarray(a["gt_names"])
    box = np.asarray(a["gt_boxes"], float).reshape(-1, 7)
    vel = np.asarray(a["gt_velocity_3d"], float).reshape(-1, 3)
    trk = a["track_tokens"]
    veh = {}
    lead = None
    oncoming = False
    for i in np.where(names == "vehicle")[0]:
        x, y, _, L, W, _, h = box[i]
        c, s = math.cos(h), math.sin(h)
        hx = 0.5 * (L * abs(c) + W * abs(s))
        gap = x - hx - EGO_FRONT
        spd = math.hypot(vel[i, 0], vel[i, 1])
        cor = (abs(y) < CORR_HALF_W) and (0.0 < x < CORR_MAX_X)
        d = dict(x=x, y=y, h=h, L=L, W=W, gap=gap, spd=spd, vxe=vel[i, 0], cor=cor, trk=trk[i])
        veh[trk[i]] = d
        if cor and gap > 0.0:
            if vel[i, 0] < ONCOMING_VX:
                oncoming = True
            elif lead is None or gap < lead["gap"]:
                lead = d
    vru = False
    m = (names == "pedestrian") | (names == "bicycle")
    if m.any():
        bx = box[m]
        vru = bool(np.any((np.abs(bx[:, 1]) < CORR_HALF_W) & (bx[:, 0] > 0.0) & (bx[:, 0] < VRU_MAX_X)))
    return dict(pose=pose, vx=vx, vy=vy, v=math.hypot(vx, vy), ax=ax, ay=ay, veh=veh, lead=lead,
                oncoming=oncoming, vru=vru, ts=int(f["timestamp"]))


def to_frame(pose_from, pose_to, x, y):
    """point (x,y) in ego frame `pose_from` -> ego frame `pose_to`."""
    xf, yf, hf = pose_from
    xt, yt, ht = pose_to
    gx = xf + math.cos(hf) * x - math.sin(hf) * y
    gy = yf + math.sin(hf) * x + math.cos(hf) * y
    dx, dy = gx - xt, gy - yt
    return math.cos(ht) * dx + math.sin(ht) * dy, -math.sin(ht) * dx + math.cos(ht) * dy


def dist_to_polyline(pt, P):
    a, b = P[:-1], P[1:]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", pt - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9), 0, 1)
    proj = a + t[:, None] * ab
    return float(np.min(np.hypot(*(pt - proj).T)))


NAN = float("nan")


def nanfn(fn, arr):
    arr = np.asarray(arr, float)
    return float(fn(arr)) if np.isfinite(arr).any() else NAN


def build_row(info, frames, j, cmap):
    """info: list of frame_info; frames: raw frames (for meta); j: index of t0."""
    f0 = frames[j]
    I = info
    r = {}
    r["token"] = f0["token"]; r["log_name"] = f0["log_name"]; r["map_location"] = f0["map_location"]
    r["scene_token"] = f0.get("scene_token", ""); r["frame_idx"] = int(f0.get("frame_idx", -1)); r["list_idx"] = int(j)
    r["timestamp_t0_us"] = int(f0["timestamp"])
    ts = np.array([I[k]["ts"] for k in range(j - NH + 1, j + H + 1)], float) * 1e-6
    r["dt_ok"] = bool(np.all(np.abs(np.diff(ts) - DT) < 0.05))
    cmd = np.asarray(f0["driving_command"]).astype(int)
    r["cmd0"], r["cmd1"], r["cmd2"], r["cmd3"] = (int(c) for c in cmd[:4])
    r["cmd_idx"] = int(np.argmax(cmd)) if cmd.sum() == 1 else -1
    r["n_roadblocks"] = len(f0["roadblock_ids"])

    # ---------------- ego ----------------
    fut = range(j + 1, j + H + 1)
    hist = range(j - NH + 1, j + 1)
    win = range(j, j + H + 1)
    v = lambda ks: np.array([I[k]["v"] for k in ks])
    v0 = I[j]["v"]
    vf, vh, vw = v(fut), v(hist), v(win)
    r["ego_v0"] = v0
    r["ego_v_min_fut"] = float(vf.min()); r["ego_v_max_fut"] = float(vf.max()); r["ego_v_end"] = float(vf[-1])
    r["ego_v_hist_min"] = float(vh.min()); r["ego_v_hist_max"] = float(vh.max())
    axw = np.array([I[k]["ax"] for k in win])
    r["ego_ax_min_mps2"] = float(axw.min()); r["ego_ax_max_mps2"] = float(axw.max())
    r["ego_decel_max_ax_mps2"] = float(max(0.0, -axw.min())); r["ego_accel_max_ax_mps2"] = float(max(0.0, axw.max()))
    dv = np.diff(vw) / DT
    r["ego_decel_max_fd_mps2"] = float(max(0.0, -dv.min())); r["ego_accel_max_fd_mps2"] = float(max(0.0, dv.max()))
    r["ego_stopped_t0"] = bool(v0 < STOP_V)
    r["ego_stops_in_future"] = bool(v0 >= STOP_V and vf.min() < STOP_V)
    r["ego_stationary_window"] = bool(np.all(np.concatenate([vh, vf]) < STOP_V))
    r["ego_starts_from_stop"] = bool(v0 < STOP_V and vf.max() > START_V)
    r["ego_started_in_hist"] = bool(vh.min() < STOP_V and v0 >= STOP_V)
    P = np.array([to_frame(I[k]["pose"], I[j]["pose"], 0.0, 0.0) for k in win])
    r["ego_dist_fut_m"] = float(np.sum(np.hypot(*np.diff(P, axis=0).T)))
    r["ego_dyaw_fut_rad"] = float(wrap(I[j + H]["pose"][2] - I[j]["pose"][2]))
    r["ego_x_end_m"] = float(P[-1, 0]); r["ego_y_end_m"] = float(P[-1, 1])

    # ---------------- lead ----------------
    L0 = I[j]["lead"]
    r["lead_t0"] = L0 is not None
    if L0 is not None:
        closing = I[j]["vx"] - L0["vxe"]
        r["lead_trk_t0"] = L0["trk"]
        r["lead_gap_m"] = float(L0["gap"]); r["lead_speed_mps"] = float(L0["spd"])
        r["lead_vx_ego_mps"] = float(L0["vxe"]); r["lead_closing_mps"] = float(closing)
        r["lead_time_gap_s"] = float(L0["gap"] / I[j]["v"]) if I[j]["v"] > STOP_V else NAN
        r["lead_ttc_s"] = float(L0["gap"] / closing) if closing > 0.1 else NAN
        r["lead_stationary"] = bool(L0["spd"] < STOP_V)
    else:
        for c in ["lead_gap_m", "lead_speed_mps", "lead_vx_ego_mps", "lead_closing_mps", "lead_time_gap_s", "lead_ttc_s"]:
            r[c] = NAN
        r["lead_trk_t0"] = ""; r["lead_stationary"] = False
    leads = [I[k]["lead"] for k in win]
    has = np.array([ld is not None for ld in leads])
    r["lead_present_frac_win"] = float(has.mean()); r["lead_present_all_win"] = bool(has.all())
    r["lead_same_track_all_win"] = bool(L0 is not None and all(ld is not None and ld["trk"] == L0["trk"] for ld in leads))
    r["lead_appears_win"] = bool((not has[0]) and has[1:].any())
    gaps, ttcs, tgaps = [], [], []
    for k, ld in zip(win, leads):
        if ld is None:
            continue
        cl = I[k]["vx"] - ld["vxe"]
        gaps.append(ld["gap"])
        if cl > 0.1:
            ttcs.append(ld["gap"] / cl)
        if I[k]["v"] > STOP_V:
            tgaps.append(ld["gap"] / I[k]["v"])
    r["lead_gap_min_win_m"] = nanfn(np.min, gaps) if gaps else NAN
    r["lead_ttc_min_win_s"] = nanfn(np.min, ttcs) if ttcs else NAN
    r["lead_time_gap_min_win_s"] = nanfn(np.min, tgaps) if tgaps else NAN
    # the t0 lead tracked by track token through the future
    if L0 is not None:
        sp = np.array([I[k]["veh"][L0["trk"]]["spd"] if L0["trk"] in I[k]["veh"] else NAN for k in win])
        r["lead_track_found_frac"] = float(np.isfinite(sp).mean())
        if np.isfinite(sp[1:]).any():
            r["lead_speed_min_fut"] = float(np.nanmin(sp[1:]))
            r["lead_speed_drop_fut"] = float(sp[0] - np.nanmin(sp[1:]))
            steps = [(sp[a] - sp[a + 1]) / DT for a in range(H) if np.isfinite(sp[a]) and np.isfinite(sp[a + 1])]
            r["lead_decel_max_mps2"] = float(max(0.0, max(steps))) if steps else NAN
            r["lead_stops_fut"] = bool(sp[0] >= 1.0 and np.nanmin(sp[1:]) < STOP_V)
        else:
            r["lead_speed_min_fut"] = r["lead_speed_drop_fut"] = r["lead_decel_max_mps2"] = NAN
            r["lead_stops_fut"] = False
    else:
        r["lead_track_found_frac"] = NAN
        r["lead_speed_min_fut"] = r["lead_speed_drop_fut"] = r["lead_decel_max_mps2"] = NAN
        r["lead_stops_fut"] = False
    r["oncoming_corridor_t0"] = bool(I[j]["oncoming"])
    r["vru_corridor_t0"] = bool(I[j]["vru"]); r["vru_corridor_win"] = bool(any(I[k]["vru"] for k in win))

    # ---------------- cut-in ----------------
    cin = dict(k=-1, gap=NAN, clos=NAN, ttc=NAN, spd=NAN)
    cinp = -1
    cands = [d for d in I[j]["veh"].values() if 0.0 < d["x"] < CORR_MAX_X and CORR_HALF_W <= abs(d["y"]) <= CUTIN_SIDE_MAX_Y]
    best = None
    for cd in cands:
        for k in fut:
            dk = I[k]["veh"].get(cd["trk"])
            if dk is not None and dk["cor"] and dk["gap"] > 0.0 and dk["spd"] >= CUTIN_MIN_SPD and dk["vxe"] >= ONCOMING_VX:
                if best is None or (k - j, dk["gap"]) < (best[0], best[1]["gap"]):
                    best = (k - j, dk)
                break
        # path-based variant: agent centre within CORR_HALF_W of the ego's realised path (t0 frame)
        if cinp < 0:
            d0 = dist_to_polyline(np.array(to_frame(I[j]["pose"], I[j]["pose"], cd["x"], cd["y"])), P)
            if d0 >= CORR_HALF_W:
                for k in fut:
                    dk = I[k]["veh"].get(cd["trk"])
                    if dk is None or dk["spd"] < CUTIN_MIN_SPD or dk["x"] <= 0.0:
                        continue
                    pt = np.array(to_frame(I[k]["pose"], I[j]["pose"], dk["x"], dk["y"]))
                    if dist_to_polyline(pt, P) < CORR_HALF_W:
                        cinp = k - j
                        break
    if best is not None:
        k, dk = best
        cl = I[j + k]["vx"] - dk["vxe"]
        cin = dict(k=k, gap=dk["gap"], clos=cl, ttc=(dk["gap"] / cl if cl > 0.1 else NAN), spd=dk["spd"])
    r["cutin"] = bool(cin["k"] > 0); r["cutin_entry_step"] = int(cin["k"])
    r["cutin_gap_m"] = float(cin["gap"]); r["cutin_closing_mps"] = float(cin["clos"])
    r["cutin_ttc_s"] = float(cin["ttc"]); r["cutin_agent_speed_mps"] = float(cin["spd"])
    r["cutin_path"] = bool(cinp > 0); r["cutin_path_entry_step"] = int(cinp)

    # ---------------- traffic lights (list as given, frame t0) ----------------
    tl = list(f0["traffic_lights"])
    r["tl_n"] = len(tl); r["tl_present"] = len(tl) > 0
    r["tl_n_red"] = int(sum(1 for _, b in tl if b)); r["tl_any_red"] = r["tl_n_red"] > 0
    # ---------------- map-based ----------------
    if cmap is None:
        r["map_ok"] = False
        for c in ["junction_t0", "junction_in_win"]:
            r[c] = False
        r["junction_dist_ahead_m"] = NAN
        r["tl_ctrl_ahead"] = r["tl_red_ahead"] = r["tl_green_ahead"] = False
        r["tl_red_ahead_dist_m"] = NAN
        r["tl_ctrl_on_path"] = r["tl_red_on_path"] = False
    else:
        r["map_ok"] = True
        gx = np.array([I[k]["pose"][0] for k in win]); gy = np.array([I[k]["pose"][1] for k in win])
        jn = cmap.in_junction(gx, gy)
        r["junction_t0"] = bool(jn[0]); r["junction_in_win"] = bool(jn.any())
        x0, y0, h0 = I[j]["pose"]
        ds = np.arange(0.0, JUNC_PROBE_MAX + 1e-6, JUNC_PROBE_STEP)
        jp = cmap.in_junction(x0 + ds * math.cos(h0), y0 + ds * math.sin(h0))
        r["junction_dist_ahead_m"] = float(ds[np.argmax(jp)]) if jp.any() else NAN
        red_ids = {int(i) for i, b in tl if b}
        all_ids = {int(i) for i, _ in tl}
        hit = cmap.connectors_hit(gx, gy)
        r["tl_ctrl_on_path"] = bool(hit & all_ids); r["tl_red_on_path"] = bool(hit & red_ids)
        ctrl = red = green = False
        red_d = []
        for i, b in tl:
            e = cmap.entry.get(int(i))
            if e is None:
                continue
            ex, ey = to_frame((0.0, 0.0, 0.0), (x0, y0, h0), e[0], e[1])
            if 0.0 < ex < TL_AHEAD_MAX_X and abs(ey) < TL_AHEAD_MAX_Y and abs(wrap(e[2] - h0)) < TL_AHEAD_MAX_DHEAD:
                ctrl = True
                if b:
                    red = True; red_d.append(ex)
                else:
                    green = True
        r["tl_ctrl_ahead"] = ctrl; r["tl_red_ahead"] = red; r["tl_green_ahead"] = green
        r["tl_red_ahead_dist_m"] = float(min(red_d)) if red_d else NAN
    return r


# ---- worker ------------------------------------------------------------------------------------------
_MAPS = {}


def get_map(city):
    if city not in CITIES:
        return None
    if city not in _MAPS:
        _MAPS[city] = CityMap(city)
    return _MAPS[city]


def process_log(args):
    log, tokens = args
    t0 = time.time()
    path = f"{LOG_DIR}/{log}.pkl"
    res = dict(log=log, rows=[], status="ok", err="", n_frames=0, n_windows=0, skip_has_route=[], n_tokens_wanted=len(tokens))
    if not os.path.exists(path):
        res["status"] = "missing_pkl"; return res
    try:
        frames = pickle.load(open(path, "rb"))
        assert isinstance(frames, list) and len(frames) > 0
    except Exception as e:
        res["status"] = "unpickle_failed"; res["err"] = repr(e)[:300]; return res
    res["n_frames"] = len(frames)
    try:
        city = frames[0]["map_location"]
        cmap = get_map(city)
        res["map_status"] = "ok" if cmap is not None else f"no_map_for_{city}"
        info = [None] * len(frames)
        found = set()
        for i in range(0, len(frames) - (NH + NF) + 1):
            j = i + NH - 1
            tok = frames[j]["token"]
            if tok not in tokens:
                continue
            res["n_windows"] += 1
            if len(frames[j]["roadblock_ids"]) == 0:      # SceneFilter.has_route: true (dataloader.py:49)
                res["skip_has_route"].append(tok); continue
            for k in range(j - NH + 1, j + H + 1):
                if info[k] is None:
                    info[k] = frame_info(frames[k])
            res["rows"].append(build_row(info, frames, j, cmap))
            found.add(tok)
        res["found"] = sorted(found)
        res["has_route_skipped"] = res.pop("skip_has_route")
    except Exception as e:
        res["status"] = "extract_failed"; res["err"] = traceback.format_exc()[-600:]
    res["secs"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-logs", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    logs, toks = parse_yaml()
    tokset = set(toks)
    print(f"yaml: {len(logs)} log_names, {len(toks)} tokens ({len(tokset)} unique)", flush=True)
    present = set(os.path.basename(p)[:-4] for p in glob.glob(f"{LOG_DIR}/*.pkl"))
    todo = [l for l in logs]
    if a.limit_logs:
        todo = todo[: a.limit_logs]
    jobs = [(l, tokset) for l in todo]
    results = []
    t0 = time.time()
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for n, r in enumerate(pool.imap_unordered(process_log, jobs), 1):
                results.append(r)
                print(f"[{n}/{len(jobs)}] {r['log']} {r['status']} rows={len(r['rows'])} frames={r['n_frames']} {r.get('secs')}s", flush=True)
    else:
        for n, jb in enumerate(jobs, 1):
            r = process_log(jb); results.append(r)
            print(f"[{n}/{len(jobs)}] {r['log']} {r['status']} rows={len(r['rows'])} frames={r['n_frames']} {r.get('secs')}s", flush=True)
    rows = [row for r in results for row in r["rows"]]
    df = pd.DataFrame(rows)
    seen = df["token"].tolist() if len(df) else []
    dup = len(seen) - len(set(seen))
    df = df.drop_duplicates("token").sort_values(["log_name", "frame_idx"]).reset_index(drop=True)
    os.makedirs(a.out, exist_ok=True)
    suffix = "" if not a.limit_logs else f"_limit{a.limit_logs}"
    df.to_parquet(f"{a.out}/features{suffix}.parquet", index=False)
    df.to_csv(f"{a.out}/features{suffix}.csv.gz", index=False, compression="gzip")
    got = set(df["token"])
    wanted_in_scope = set(t for t in toks)
    rep = dict(
        yaml_logs=len(logs), yaml_tokens=len(toks), yaml_tokens_unique=len(tokset),
        yaml_tokens_sha256=hashlib.sha256("\n".join(sorted(toks)).encode()).hexdigest(),
        pkl_files_in_dir=len(present), pkl_not_in_navtest_logs=sorted(present - set(logs)),
        logs_processed=len(results), logs_by_status={s: sum(1 for r in results if r["status"] == s) for s in set(r["status"] for r in results)},
        logs_failed=[dict(log=r["log"], status=r["status"], err=r["err"]) for r in results if r["status"] != "ok"],
        n_tokens_read=int(len(df)), duplicate_tokens_dropped=int(dup),
        n_skipped_has_route=sum(len(r.get("has_route_skipped", [])) for r in results),
        has_route_skipped_tokens=[t for r in results for t in r.get("has_route_skipped", [])][:50],
        n_tokens_not_found=int(len(wanted_in_scope - got)) if not a.limit_logs else None,
        secs=round(time.time() - t0, 1), n_cols=int(df.shape[1]),
        map_status={r["log"]: r.get("map_status") for r in results if r.get("map_status") != "ok"},
    )
    json.dump(rep, open(f"{a.out}/run_report{suffix}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("map_status",)}, indent=1)[:3000], flush=True)


if __name__ == "__main__":
    main()
