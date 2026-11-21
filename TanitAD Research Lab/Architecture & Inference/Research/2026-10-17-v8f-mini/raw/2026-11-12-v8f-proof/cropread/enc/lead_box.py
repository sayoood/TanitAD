#!/usr/bin/env python3
"""Lead boxes and control p1 (card section 7).

  python lead_box.py boxes --list <ids> --root <stage root> --out <box dir>
  python lead_box.py p1 --eval-list L --train-final F --join-eval J --join-train J --box-dir D --out ctl/P1.json

PER CLIP AND CACHE FRAME j (episode frame k = 2j, clock t = t_query[k], the builder's own frame times):
  1. the obstacle.offline rows of the source (stage: obs/<clip>.parquet) are turned into agents of the frame by the join build's own
     composition (scripts/build_obstacle_join.py: world_agents_at -> rig_to_world at the sample's own time -> ego_frame_agents at the
     frame pose; the poses are physicalai.signals_at at t_query, exactly the builder's), with the 3-D columns (center_z, size_z) carried;
  2. the join's OWN lead rule (scripts/build_v9_labels.py:lead_for_frame, called unchanged) gives has_lead and the gap; lead_index() below is
     the same selection restated only to learn WHICH agent it chose (its gap is asserted equal to lead_for_frame's);
  3. the box of that agent goes to the camera frame with the front-wide mount (calibration/sensor_extrinsics, FrontWideExtrinsics,
     physicalai.py:~250: cam->vehicle rotation + translation; camera axes +x right, +y down, +z boresight as the builder's projection).
     Angles are the rectified frame's own: azimuth = atan2(-x, z) (LEFT positive, linear in the column), elevation = atan2(-y, hypot(x, z))
     (UP positive; the cylinder's rows are linear in tan(elevation)); elevation 0 is the optical axis (the builder applies no mount pitch).
Output box/<clip>.npz, one row per cache frame j (n_cache = ceil(n_ep / 2)), see KEYS.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enc_common import Layout, frame_plan, read_ids, setup_paths   # noqa: E402

setup_paths()

KEYS = ("has_lead", "gap_re", "ctr_az", "ctr_el", "bot_az", "bot_el", "ang_w", "ang_h", "rng", "ctr_ego", "size_lwh", "yaw_rel",
        "kappa_ahead", "v_ego", "t_query_s", "gap_rule_mismatch")
KAPPA_LOOK_M = 30.0


# ---------------------------------------------------------------------------------------------- agents
def tracks_3d(obs):
    """build_obstacle_join.clip_tracks (lines 364-403) with the 3-D columns carried: cz, h. Same track order, same sample order."""
    import build_obstacle_join as O
    ts = np.asarray(obs["timestamp_us"], dtype=np.float64) / 1e6
    ref = np.asarray(obs["reference_frame"]).astype(str)
    if (ref != "rig").any():
        raise ValueError("reference_frame != rig on some rows")
    tid = np.asarray(obs["track_id"]).astype(str)
    cx, cy, cz = (np.asarray(obs[c], dtype=np.float64) for c in ("center_x", "center_y", "center_z"))
    sx, sy, sz = (np.asarray(obs[c], dtype=np.float64) for c in ("size_x", "size_y", "size_z"))
    yaw = np.asarray(O.yaw_from_quaternion(obs["orientation_x"], obs["orientation_y"], obs["orientation_z"], obs["orientation_w"]))
    cls = np.asarray(obs["label_class"]).astype(str)
    out = []
    for track in np.unique(tid):
        m = tid == track
        o = np.argsort(ts[m], kind="stable")
        out.append({"tid": str(track), "cls": str(cls[m][o[0]]), "t": ts[m][o], "cx": cx[m][o], "cy": cy[m][o], "yaw": yaw[m][o],
                    "l": sx[m][o], "w": sy[m][o], "cz": cz[m][o], "h": sz[m][o]})
    return out


def agents_at_frame(tracks, t_abs_s, ego_track, pose_xyyaw, tol_s):
    """build_obstacle_join.world_agents_at (lines 406-432) + ego_frame_agents, 3-D columns carried -> list of agent dicts in the
    ego (rig) frame of the frame pose, keys cx cy yaw l w cls cz h tid."""
    import build_obstacle_join as O
    from tanitad.data.bev_raster import ego_frame_agents
    rows, samp_t, meta = [], [], []
    for tr in tracks:
        j = int(np.argmin(np.abs(tr["t"] - float(t_abs_s))))
        if abs(float(tr["t"][j]) - float(t_abs_s)) > float(tol_s):
            continue
        rows.append((float(tr["cx"][j]), float(tr["cy"][j]), float(tr["yaw"][j]), float(tr["l"][j]), float(tr["w"][j]), -1.0))
        samp_t.append(float(tr["t"][j]))
        meta.append((tr["tid"], tr["cls"], float(tr["cz"][j]), float(tr["h"][j])))
    if not rows:
        return []
    ag_w = O.rig_to_world(np.asarray(rows, dtype=np.float64), ego_track.at(np.asarray(samp_t)))
    ag_e = ego_frame_agents(ag_w, pose_xyyaw)
    return [{"cx": float(a[0]), "cy": float(a[1]), "yaw": float(a[2]), "l": float(a[3]), "w": float(a[4]),
             "tid": m[0], "cls": m[1], "cz": m[2], "h": m[3]} for a, m in zip(ag_e, meta)]


def lead_index(V, log, tr, path0, t, agents):
    """build_v9_labels.lead_for_frame (lines 501-532) restated ONLY to return the chosen agent's index in `agents`
    (None when no lead). Returns (index|None, gap)."""
    import numpy as _np
    s_k = float(V.s_at(tr, t))
    x0, y0, psi0, _v = (float(a) for a in V.sample(log, t))
    rng = V.LITERALS["lead_range_m"]
    m = (path0.s >= s_k) & (path0.s <= s_k + rng)
    if m.sum() < 2:
        return None, _np.nan
    px, py = V.to_now_frame(path0.x[m], path0.y[m], x0, y0, psi0)
    ps = path0.s[m] - s_k
    xlo, xhi, yab = V.LITERALS["ego_footprint"]
    cls_ok = V.LITERALS["lead_classes"]
    sel = [i for i, a in enumerate(agents) if a.get("cls") in cls_ok]
    if not sel:
        return None, _np.nan
    cx = _np.array([float(agents[i]["cx"]) for i in sel])
    cy = _np.array([float(agents[i]["cy"]) for i in sel])
    ln = _np.array([float(agents[i].get("l", 4.5)) for i in sel])
    keep = ~((cx > xlo) & (cx < xhi) & (_np.abs(cy) < yab))
    if not keep.any():
        return None, _np.nan
    sel = [i for i, kp in zip(sel, keep) if kp]
    cx, cy, ln = cx[keep], cy[keep], ln[keep]
    d2 = (cx[:, None] - px[None, :]) ** 2 + (cy[:, None] - py[None, :]) ** 2
    j = d2.argmin(1)
    dmin = d2[_np.arange(len(cx)), j]
    sa = ps[j]
    ok = (dmin <= V.LITERALS["lead_corridor_m"] ** 2) & (sa >= V.LITERALS["lead_min_s_m"]) & (sa <= rng)
    if not ok.any():
        return None, _np.nan
    b = int(_np.argmin(_np.where(ok, sa, _np.inf)))
    return sel[b], float(sa[b]) - 0.5 * float(ln[b]) - V.LITERALS["lead_front_offset_m"]


# ---------------------------------------------------------------------------------------------- camera geometry
def cam_angles(p_rig, R_c2v, t_cam):
    """Points in the rig frame [N,3] -> (az_left_deg, el_up_deg, range_m, z_cam) in the rectified frame's coordinates."""
    p = (np.asarray(p_rig, dtype=np.float64) - np.asarray(t_cam, dtype=np.float64)) @ np.asarray(R_c2v, dtype=np.float64)   # = R^T (p - t)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    az = np.degrees(np.arctan2(-x, z))
    el = np.degrees(np.arctan2(-y, np.hypot(x, z)))
    return az, el, np.sqrt((p ** 2).sum(1)), z


def box_geometry(a, R_c2v, t_cam):
    """One agent dict -> dict of centre / rear-face-bottom directions and angular extents in the rectified frame."""
    cx, cy, cz, l, w, h, yaw = a["cx"], a["cy"], a["cz"], a["l"], a["w"], a["h"], a["yaw"]
    ctr = np.array([[cx, cy, cz]])
    az_c, el_c, rng, _ = cam_angles(ctr, R_c2v, t_cam)
    # bottom of the box = bottom-face centre of the END of the long axis nearest the camera (the face the image shows)
    d = np.array([math.cos(yaw), math.sin(yaw)])
    ends = [np.array([cx, cy]) + s * 0.5 * l * d for s in (-1.0, 1.0)]
    e = min(ends, key=lambda q: (q[0] - t_cam[0]) ** 2 + (q[1] - t_cam[1]) ** 2)
    az_b, el_b, _, _ = cam_angles(np.array([[e[0], e[1], cz - 0.5 * h]]), R_c2v, t_cam)
    c, s = math.cos(yaw), math.sin(yaw)
    corners = []
    for sx in (-0.5, 0.5):
        for sy in (-0.5, 0.5):
            for sz in (-0.5, 0.5):
                corners.append([cx + c * sx * l - s * sy * w, cy + s * sx * l + c * sy * w, cz + sz * h])
    az_k, el_k, _, zc = cam_angles(np.array(corners), R_c2v, t_cam)
    if (zc > 0).all():
        ang_w, ang_h = float(az_k.max() - az_k.min()), float(el_k.max() - el_k.min())
    else:
        ang_w = ang_h = float("nan")
    return {"ctr_az": float(az_c[0]), "ctr_el": float(el_c[0]), "bot_az": float(az_b[0]), "bot_el": float(el_b[0]),
            "ang_w": ang_w, "ang_h": ang_h, "rng": float(rng[0])}


def kappa_ahead(poses, look=KAPPA_LOOK_M):
    """Ego path curvature at each episode frame from the ego's own track: heading change over the next `look` metres of arc divided by
    `look` (left positive; yaw is the track's unwrapped heading). NaN where the track does not extend `look` metres ahead."""
    x, y = poses[:, 0].astype(np.float64), poses[:, 1].astype(np.float64)
    yaw = np.unwrap(poses[:, 2].astype(np.float64))
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    tgt = s + look
    ok = tgt <= s[-1] + 1e-9
    yaw_t = np.interp(np.minimum(tgt, s[-1]), s, yaw)
    return np.where(ok, (yaw_t - yaw) / look, np.nan)


# ---------------------------------------------------------------------------------------------- one clip
def compute_clip(clip, L):
    import pandas as pd
    import build_obstacle_join as O
    import build_v9_labels as V
    from tanitad.data.physicalai import signals_at, FrontWideExtrinsics
    plan = frame_plan(L.ts(clip))
    n_ep, n = plan["n_ep"], plan["n_cache"]
    ego_df = pd.read_parquet(L.ego(clip))
    _, poses = signals_at(ego_df, plan["t_query"])                   # [n_ep,4] x y yaw v: the builder's own poses
    ext = pd.read_csv(L.ext_csv, dtype={"clip_id": str}).set_index("clip_id").loc[clip]
    fw = FrontWideExtrinsics(qx=float(ext["qx"]), qy=float(ext["qy"]), qz=float(ext["qz"]), qw=float(ext["qw"]),
                             x=float(ext["x"]), y=float(ext["y"]), z=float(ext["z"]))
    R_c2v, t_cam = fw.rotation_cam_to_vehicle(), np.array([fw.x, fw.y, fw.z])
    out = {k: np.full(n, np.nan, np.float64) for k in ("gap_re", "ctr_az", "ctr_el", "bot_az", "bot_el", "ang_w", "ang_h", "rng", "yaw_rel")}
    out["has_lead"] = np.zeros(n, bool)
    out["gap_rule_mismatch"] = np.zeros(n, bool)
    out["ctr_ego"] = np.full((n, 3), np.nan)
    out["size_lwh"] = np.full((n, 3), np.nan)
    kap = kappa_ahead(poses)
    ks = 2 * np.arange(n)
    out["kappa_ahead"], out["v_ego"] = kap[ks], poses[ks, 3].astype(np.float64)
    out["t_query_s"] = plan["t_query"][ks] / 1e6
    has_obs = os.path.exists(L.obs(clip))
    if has_obs:
        obs = pd.read_parquet(L.obs(clip))
        tracks = tracks_3d(obs)
        ego_track = O.EgoTrack(ego_df)
        log = V.load_log(L.ego(clip))
        ts0 = float(ego_df["timestamp"].to_numpy(np.float64)[0]) / 1e6
        trk = V.track10(log)
        p0 = V.smooth_path(trk, 0.0)
        for j, k in enumerate(ks):
            t_abs = plan["t_query"][k] / 1e6
            t_log = t_abs - ts0
            if t_log < 0 or t_log > log.t_end:
                continue
            agents = agents_at_frame(tracks, t_abs, ego_track, poses[k, :3].astype(np.float64), O.DEFAULT_TOL_S)
            lead, gap, _tg, _s, _v = V.lead_for_frame(log, trk, p0, t_log, agents)
            idx, gap2 = lead_index(V, log, trk, p0, t_log, agents)
            if bool(lead) != (idx is not None) or (lead and abs(gap - gap2) > 1e-9):
                out["gap_rule_mismatch"][j] = True
            if not lead or idx is None:
                continue
            a = agents[idx]
            g = box_geometry(a, R_c2v, t_cam)
            out["has_lead"][j] = True
            out["gap_re"][j] = gap
            for kk in ("ctr_az", "ctr_el", "bot_az", "bot_el", "ang_w", "ang_h", "rng"):
                out[kk][j] = g[kk]
            out["ctr_ego"][j] = (a["cx"], a["cy"], a["cz"])
            out["size_lwh"][j] = (a["l"], a["w"], a["h"])
            out["yaw_rel"][j] = a["yaw"]
    return out, {"clip": clip, "n_ep": n_ep, "n_cache": n, "has_obs": has_obs, "n_lead": int(out["has_lead"].sum()),
                 "n_gap_rule_mismatch": int(out["gap_rule_mismatch"].sum()),
                 "cam_xyz": [float(v) for v in t_cam], "optical_axis_pitch_deg_down": math.degrees(fw.optical_axis_pitch_rad())}


def cmd_boxes(a):
    os.makedirs(a.out, exist_ok=True)
    L = Layout(a.root)
    clips = read_ids(a.list)
    led = open(os.path.join(a.out, "ledger.jsonl"), "a", buffering=1, encoding="utf-8")
    t_all = time.time()
    n_done = 0
    for c in clips:
        p = os.path.join(a.out, f"{c}.npz")
        if os.path.exists(p):
            continue
        t0 = time.time()
        try:
            arrs, st = compute_clip(c, L)
        except Exception as e:                                       # noqa: BLE001
            led.write(json.dumps({"clip": c, "error": f"{type(e).__name__}: {str(e)[:200]}"}) + "\n")
            print(f"[boxes] FAIL {c[:8]} {type(e).__name__}: {str(e)[:160]}", flush=True)
            continue
        np.savez_compressed(p, **arrs)
        st["secs"] = round(time.time() - t0, 2)
        led.write(json.dumps(st) + "\n")
        n_done += 1
        if n_done <= 3 or n_done % 50 == 0:
            print(f"[boxes] {n_done} done, last {c[:8]} n={st['n_cache']} leads={st['n_lead']} mismatch={st['n_gap_rule_mismatch']} {st['secs']}s", flush=True)
    print(f"[boxes] done {n_done} new clips in {time.time() - t_all:.0f}s", flush=True)
    print("ZZ BOXES DONE ZZ", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------- control p1
def p1_from_arrays(items, tol_m=0.5, agree_min=0.75, n_min=200, below_min=0.98, match_min=0.98):
    """items: list of dict(clip, role, box (dict of arrays | None), gap_join, valid [bool per frame]) -> the P1 record (pure)."""
    n_v = n_m = 0
    diffs = []
    n2 = n2_ok = 0
    n3 = n3_ok = 0
    missing = []
    bins = {}
    for it in items:
        valid, gj, bx = it["valid"], np.asarray(it["gap_join"], dtype=np.float64), it["box"]
        if bx is None or len(bx["has_lead"]) != len(valid):
            n_v += int(valid.sum())
            n3 += int((valid & (gj < 20.0)).sum())
            missing.append(it["clip"])
            continue
        has, gre = bx["has_lead"].astype(bool), np.asarray(bx["gap_re"], dtype=np.float64)
        ok_i = valid & has & (np.abs(np.nan_to_num(gre, nan=1e9) - np.nan_to_num(gj, nan=-1e9)) <= tol_m)
        n_v += int(valid.sum())
        n_m += int(ok_i.sum())
        diffs.append(np.abs(gre - gj)[valid & has])
        az, kap, bel = np.asarray(bx["ctr_az"], np.float64), np.asarray(bx["kappa_ahead"], np.float64), np.asarray(bx["bot_el"], np.float64)
        m2 = valid & has & (gre >= 15.0) & (np.abs(az) >= 3.0) & (np.abs(kap) >= 1.0 / 300.0)
        n2 += int(m2.sum())
        agree = np.sign(az) == np.sign(kap)
        n2_ok += int((m2 & agree).sum())
        m3 = valid & (gj < 20.0)
        n3 += int(m3.sum())
        n3_ok += int((m3 & has & (bel < 0.0)).sum())
        for lo, hi in ((15, 25), (25, 40), (40, 60)):
            mm = m2 & (gre >= lo) & (gre < hi)
            b = bins.setdefault(f"{lo}-{hi}", [0, 0])
            b[0] += int(mm.sum())
            b[1] += int((mm & agree).sum())
    f1 = n_m / n_v if n_v else float("nan")
    f2 = n2_ok / n2 if n2 else float("nan")
    f3 = n3_ok / n3 if n3 else float("nan")
    rec = {"clause_i": {"what": "recomputed gap within 0.5 m of the join's lead gap, frames with a valid lead", "n_frames": n_v,
                        "n_ok": n_m, "frac": f1, "bar": match_min, "abs_diff_m_median_max": ([float(np.median(d)), float(d.max())] if (d := (np.concatenate(diffs) if diffs else np.zeros(0))).size else None), "pass": bool(n_v and f1 >= match_min)},
           "clause_ii": {"what": "sign(azimuth of box centre) == sign(ego path curvature); gap>=15, |az|>=3 deg, |curvature|>=1/300",
                         "n_frames": n2, "n_agree": n2_ok, "frac": f2, "bar": agree_min, "n_min": n_min,
                         "by_gap_bin_n_agree": bins, "pass": bool(n2 >= n_min and f2 >= agree_min)},
           "clause_iii": {"what": "elevation of the bottom of the box < 0, valid lead and gap < 20 m", "n_frames": n3, "n_ok": n3_ok,
                          "frac": f3, "bar": below_min, "pass": bool(n3 and f3 >= below_min)},
           "clips_without_box_file_or_length_mismatch": missing[:50], "n_clips_without_box": len(missing)}
    rec["pass_all"] = bool(rec["clause_i"]["pass"] and rec["clause_ii"]["pass"] and rec["clause_iii"]["pass"])
    return rec


def cmd_p1(a):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from make_lists import valid_lead_mask
    items = []
    for lst, jpath, role_default in ((a.eval_list, a.join_eval, "eval"), (a.train_final, a.join_train, None)):
        J = dict(np.load(jpath, allow_pickle=True))
        row = {str(c): i for i, c in enumerate(J["clip"])}
        for ln in open(lst, encoding="utf-8"):
            t = ln.split()
            if not t:
                continue
            c = t[0]
            if c not in row:
                continue
            i = row[c]
            o, n = int(J["off"][i]), int(J["n"][i])
            gj = np.asarray(J["lead_gap"][o:o + n], dtype=np.float64)
            valid = valid_lead_mask(J["lead_valid"][o:o + n], gj)
            p = os.path.join(a.box_dir, f"{c}.npz")
            bx = dict(np.load(p)) if os.path.exists(p) else None
            items.append({"clip": c, "role": role_default or (t[1] if len(t) > 1 else "train"), "box": bx, "gap_join": gj, "valid": valid})
    rec = p1_from_arrays(items)
    rec["n_clips"] = len(items)
    rec["clip_list_sources"] = {"eval": a.eval_list, "train_final": a.train_final}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps({k: (v if k.startswith("pass") else {kk: vv for kk, vv in v.items() if kk in ("n_frames", "frac", "pass", "n_ok", "n_agree")})
                      for k, v in rec.items() if k.startswith("clause") or k == "pass_all"}))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("boxes")
    b.add_argument("--list", required=True)
    b.add_argument("--root", required=True)
    b.add_argument("--out", required=True)
    p = sub.add_parser("p1")
    p.add_argument("--eval-list", required=True)
    p.add_argument("--train-final", required=True)
    p.add_argument("--join-eval", default="/workspace/v8proof/hierrep/pins/join_eval141.npz")
    p.add_argument("--join-train", default="/workspace/v8proof/data/inputs/join_train.npz")
    p.add_argument("--box-dir", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"boxes": cmd_boxes, "p1": cmd_p1}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
