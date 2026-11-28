"""Fetch + screening logic for SCREEN_RULE.md (sha in SCREEN_RULE_SHA.txt). No model, no pixels, no map, no volume is read."""
import json, struct, zlib
import numpy as np
import hfc

REV = "5fe7a149b40ee83a2d3e09f7fe5dc2f217cabe11"
REL = "sample_set/26.04_release"
CAM = "camera_front_wide_120fov"
VEH = ("automobile", "heavy_truck", "bus", "other_vehicle", "trailer")
V_UP, V_LOW, T_BACK, T_HEAD, T_TAIL, LEAD_X, LEAD_Y, GUARD_SHARE = 5.2, 3.0, 5.0, 2.0, 6.0, 30.0, 2.0, 0.5
TOL_US = 1.5e5


def usdz_url(sc):
    return f"https://huggingface.co/datasets/{hfc.REPO}/resolve/{REV}/{REL}/{sc}/{sc}.usdz"


def fetch_range(sc, start, end):
    r = hfc.get(usdz_url(sc), headers={"Range": f"bytes={start}-{end}"})
    if r.status_code not in (200, 206):
        raise RuntimeError(f"HTTP {r.status_code}")
    return r.content


def fetch_member(sc, name, ent):
    """ent = [file_size, crc, header_offset, compress_type] from the usdz central directory. Returns bytes, CRC-checked."""
    size, crc, ho, ct = ent
    if ct != 0:
        raise RuntimeError("member is not STORED")
    head = fetch_range(sc, ho, ho + 30 + len(name) + 512 - 1)
    sig, = struct.unpack("<I", head[:4])
    if sig != 0x04034B50:
        raise RuntimeError("bad local header")
    fnl, exl = struct.unpack("<HH", head[26:30])
    data_off = ho + 30 + fnl + exl
    body = fetch_range(sc, data_off, data_off + size - 1)
    if len(body) != size or (zlib.crc32(body) & 0xFFFFFFFF) != crc:
        raise RuntimeError(f"size/crc mismatch for {name}")
    return body


def ego_from_rig(rig_bytes):
    d = json.loads(rig_bytes)
    e = d["rig_trajectories"][0]
    key = f"{CAM}@{e['sequence_id']}"
    Ts, ts = e["cameras_frame_T_rig_worlds"][key], e["cameras_frame_timestamps_us"][key]
    dt = ts[1][1] - ts[0][1]
    stride = int(round(1e5 / dt))
    idx = range(0, len(Ts), stride)
    T = np.stack([np.array(Ts[i][1], np.float64) for i in idx])
    tus = np.array([ts[i][1] for i in idx], np.int64)
    return T, tus, stride, dt


def tracks_from_json(b):
    d = json.loads(b)
    td = d[next(iter(d))]["tracks_data"]
    return ([str(x) for x in td["tracks_id"]], [str(x) for x in td["tracks_label_class"]],
            [np.asarray(p, np.float64) for p in td["tracks_poses"]], [np.asarray(t, np.float64) for t in td["tracks_timestamps_us"]])


def speed_series(xy):
    """v[k] = path length over ticks k-5..k+5 (ten intervals) / 1.0 s; NaN where the window is incomplete."""
    n = len(xy)
    v = np.full(n, np.nan)
    if n < 11:
        return v
    cs = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    k = np.arange(5, n - 5)
    v[k] = cs[k + 5] - cs[k - 5]
    return v


def lead_at(T_m, t_us, tracks):
    ids, labs, poses, tss = tracks
    Ti = np.linalg.inv(T_m)
    best, n_veh = None, 0
    for i, lab in enumerate(labs):
        if lab not in VEH:
            continue
        ts = tss[i]
        if ts.size == 0 or t_us < ts[0] - TOL_US or t_us > ts[-1] + TOL_US:
            continue
        k = int(np.argmin(np.abs(ts - t_us)))
        if abs(ts[k] - t_us) > TOL_US:
            continue
        n_veh += 1
        p = Ti[:3, :3] @ poses[i][k][:3] + Ti[:3, 3]
        x, y = float(p[0]), float(p[1])
        if 0 < x <= LEAD_X and abs(y) <= LEAD_Y and (best is None or x < best["x_m"]):
            best = {"track_id": ids[i], "label": lab, "x_m": round(x, 2), "y_m": round(y, 2)}
    return best, n_veh


def analyse(T, tus, tracks):
    """Per-scene record fields: launch events, guard share, span. A pure function of the ego and agent tracks."""
    n = len(tus)
    xy = T[:, :2, 3]
    v = speed_series(xy)
    t0, t1 = int(tus[0]), int(tus[-1])
    events = []
    for k in range(6, n - 5):
        if not (v[k - 1] < V_UP <= v[k]):
            continue
        tc = int(tus[k])
        back = [m for m in range(0, k) if tus[m] >= tc - int(T_BACK * 1e6) and not np.isnan(v[m]) and v[m] <= V_LOW]
        c1, c2, c3 = bool(back), tc >= t0 + int(T_HEAD * 1e6), tc <= t1 - int(T_TAIL * 1e6)
        if not (c1 and c2 and c3):
            continue
        m = back[-1]
        lead, n_veh = lead_at(T[m], int(tus[m]), tracks)
        events.append({"t_c_s": round((tc - t0) / 1e6, 2), "v_before": round(float(v[k - 1]), 3), "v_at": round(float(v[k]), 3),
                       "t_s_s": round((int(tus[m]) - t0) / 1e6, 2), "v_at_t_s": round(float(v[m]), 3),
                       "lead": lead is not None, "lead_detail": lead, "n_vehicle_tracks_at_t_s": n_veh})
    share = float(np.sum(v >= V_UP) / n)          # NaN >= x is False
    dts = np.diff(tus) / 1e6
    fin = np.isfinite(v).any()
    return {"n_ticks": int(n), "span_s": round((t1 - t0) / 1e6, 2), "t_first_us": t0, "t_last_us": t1,
            "max_tick_dt_dev_s": round(float(np.abs(dts - 0.1).max()), 4) if len(dts) else None,
            "v_min": round(float(np.nanmin(v)), 3) if fin else None, "v_max": round(float(np.nanmax(v)), 3) if fin else None,
            "launch_events": events, "is_launch": bool(events), "guard_share": round(share, 4),
            "is_guard": (not events) and share >= GUARD_SHARE}
