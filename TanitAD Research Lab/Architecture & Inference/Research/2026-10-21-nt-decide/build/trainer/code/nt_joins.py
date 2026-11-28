#!/usr/bin/env python3
"""nt_joins -- E3 of H-NT-DECIDE-1: OpenScene ``anns`` -> the trainer's agent / 3-D / VIS-1 / rig inputs (CPU).

``build`` (per call, a set of logs): for every token of the token table (``nt_labels.py``'s ``token_table_nt.json``)

* **2-D agent join** (``--agent-join``, ``train_p8_occupancy.JoinFileReader``): one record per 2 Hz log frame at
  episode index ``frame_idx`` = 2 / 7 / 12 (t0 - 0.5 s / t0 / t0 + 0.5 s; NOW = provider row 7). The two
  neighbours exist ONLY so ``track_rates_from_join`` has a central difference at NOW (over 1.0 s, read from
  ``t_s`` -- never assumed); a neighbour whose spacing is not 0.5 +- 0.05 s is omitted (counted) and the rate
  degrades to one-sided / masked exactly as the reader defines. Agents are in THAT FRAME's ego frame (OpenScene
  ``gt_boxes`` are lidar = ego, ``lidar2ego`` identity), as PhysicalAI's join is (``_agent_item`` re-expresses
  future frames through the poses itself).
  - class map (DECLARED, PREREG / DESIGN sec. 3): vehicle -> automobile, pedestrian -> person, bicycle -> rider;
    every other OpenScene class is DROPPED (traffic_cone, barrier, czone_sign, generic_object). truck / bus /
    trailer therefore get no positives.
  - ``track_id`` = ``int(track_token[:15], 16)`` as a digit string (the VIS-1 sidecar keys tracks int64); a
    collision inside one frame REFUSES.
  - ``occ`` = ``|atan2(cy, cx)| > 60 deg`` on the box centre: the IDENTITY PhysicalAI's flag was measured to be
    (``agent_slots.OCC_HALF_ANGLE_RAD``, agreement 1.0000), recomputed here, not invented.
  - ego-as-agent: a box whose centre lies in the ego footprint (``join_label_hygiene.is_ego_footprint``) is
    DROPPED at build time and counted (the PhysicalAI D3 defect class; nuPlan's tracker should not emit the ego).
* **3-D join** (``--join3d``, ``agent_cuboid_gt.AgentJoin3D``): the SAME records + ``cz`` (rig frame: road z = 0,
  ``cz = z_ego - road_z``) and ``h``, keyed by the RAW frame ``frame = frame_idx + n_stack - 1``.
* **VIS-1 rows** at NOW (f = 7): ``vis1_zbuffer.zbuffer`` (vendored verbatim) on the trainer's float32 target rows,
  camera = the stitch's virtual camera (R = ``frames416.virtual_rotation`` M, t = CAM_F0's mount with the road
  plane removed), observed mask = the stitch's ``src >= 0`` (pixels no camera sees are not image).
* **rig table** (``--agent-rig-extrinsics``): ``{clip_id: {qx, qy, qz, qw, x, y, z}}`` with ``q = quat(M)`` (checked:
  ``FrontWideExtrinsics(q).rotation_cam_to_vehicle()`` == M to 1e-12) and ``(x, y, z) = (F0.x, F0.y, F0.z - road_z)``
  = ``rig6.rig_camera`` (the bridge's lift camera, K1).

``road_z`` = the bridge's constant (``nt_common.ROAD_PLANE_MEDIAN_M``); a per-log plane is NOT measured here (no
LiDAR read) -- the K1 lift parity requires the SAME constant the bridge feeds.

``merge``: NavSim parts + the PhysicalAI eval139 entries of the refcv8 artifacts (the trainer reads ONE file for
train and the in-run eval139 forgetting monitor) -> the final ``.jsonl.xz`` joins (+ ``join_meta`` sidecar), the
VIS-1 ``.npz`` (``vis1.write_sidecar``) and the rig table.
"""
from __future__ import annotations

import argparse
import json
import os
import lzma
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nt_common as C  # noqa: E402

NT_TREE = Path(os.environ.get("NT_TREE") or (HERE.parent / "nt_tree"))  # Thor: the launch tree (thor_build_a1.sh exports it)
FRAMES416_DIR = NT_TREE / "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/code"
OCC_HALF_ANGLE_RAD = math.radians(60.0)          # agent_slots.OCC_HALF_ANGLE_RAD (pinned equal by the test)
NEIGHBOUR_DT_S = (0.45, 0.55)                    # INTERFACE.md sec. 3: |dt - 0.5| > 0.05 s is a dropped frame
FRAME_NOW = 7                                    # episode index of NOW (t + w - 1 at t = 0)


def _stack():
    import os
    os.environ["TANITAD_REPO"] = str(NT_TREE)
    st = str(NT_TREE / "stack")
    if st not in sys.path:
        sys.path.insert(0, st)
    import tanitad
    if not str(Path(tanitad.__file__).resolve()).lower().startswith(str(NT_TREE.resolve()).lower()):
        raise C.NTRefused(f"[nt-joins] tanitad imported from {tanitad.__file__}, not the NT tree")
    return st


def _frames416():
    _stack()
    if str(FRAMES416_DIR) not in sys.path:
        sys.path.insert(0, str(FRAMES416_DIR))
    import frames416 as F
    from tanitad.models.trunk_shapes import FRAME_416x1024
    F.configure(FRAME_416x1024)
    return F


def quat_of_R(M) -> tuple:
    """Rotation matrix -> (qx, qy, qz, qw), Shepherd's method (numerically stable on every branch)."""
    m = np.asarray(M, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        qw, qx, qy, qz = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        qw, qx, qy, qz = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        qw, qx, qy, qz = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        qw, qx, qy, qz = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    return qx / n, qy / n, qz / n, qw / n


def rig_entry(rig: dict, road_z: float) -> dict:
    """The ``--agent-rig-extrinsics`` row of one token, checked against the matrix it encodes."""
    _stack()
    from tanitad.data.physicalai import FrontWideExtrinsics
    M = np.asarray(rig["virtual_R_cam_to_lidar"], dtype=np.float64)
    qx, qy, qz, qw = quat_of_R(M)
    back = FrontWideExtrinsics(qx=qx, qy=qy, qz=qz, qw=qw).rotation_cam_to_vehicle()
    err = float(np.abs(back - M).max())
    if err > 1e-12:
        raise C.NTRefused(f"[nt-joins] quaternion round trip {err:.3e} > 1e-12 for rig {rig.get('rig_key')}")
    t = np.asarray(rig["f0_sensor2lidar_translation"], dtype=np.float64)
    return {"qx": qx, "qy": qy, "qz": qz, "qw": qw, "x": float(t[0]), "y": float(t[1]),
            "z": float(t[2] - float(road_z)), "rig_key": rig["rig_key"], "quat_roundtrip_max_abs": err,
            "source": "frames416.rig_record (virtual camera M, CAM_F0 mount) - road_z"}


def _ego_fp():
    _stack()
    from tanitad.data.join_label_hygiene import is_ego_footprint
    return is_ego_footprint


def agents_of_frame(frame: dict, road_z: float, census: Counter, is_ego) -> list:
    """OpenScene ``anns`` of one 2 Hz frame -> the join's agent dicts (+ cz, h), in the source order."""
    an = frame.get("anns") or {}
    boxes = np.asarray(an.get("gt_boxes", np.zeros((0, 7))), dtype=np.float64).reshape(-1, 7)
    names = [str(x) for x in an.get("gt_names", [])]
    toks = [str(x) for x in an.get("track_tokens", [])]
    if not (len(boxes) == len(names) == len(toks)):
        raise C.NTRefused(f"[nt-joins] anns length mismatch: boxes {len(boxes)} names {len(names)} "
                          f"tracks {len(toks)} (token {frame.get('token')})")
    out, seen = [], set()
    for b, nm, tk in zip(boxes, names, toks):
        census[f"src:{nm}"] += 1
        cls = C.CLASS_MAP.get(nm)
        if cls is None:
            census["dropped_class"] += 1
            continue
        cx, cy, cz, l, w, h, yaw = (float(v) for v in b)
        if not all(math.isfinite(v) for v in (cx, cy, cz, l, w, h, yaw)):
            census["dropped_nonfinite"] += 1
            continue
        if is_ego(cx, cy):
            census["dropped_ego_footprint"] += 1
            continue
        tid = C.track_int(tk)
        if tid in seen:
            raise C.NTRefused(f"[nt-joins] track id collision (60-bit prefix) inside one frame: {tk}")
        seen.add(tid)
        out.append({"cx": round(cx, 4), "cy": round(cy, 4), "yaw": round(yaw, 5), "l": round(l, 3),
                    "w": round(w, 3), "occ": int(abs(math.atan2(round(cy, 4), round(cx, 4))) > OCC_HALF_ANGLE_RAD),
                    "track_id": str(tid), "cls": cls, "cz": round(cz - float(road_z), 4), "h": round(h, 3)})
        census[f"kept:{cls}"] += 1
    return out


def vis1_rows_now(agents: list, rig: dict, obs: np.ndarray, road_z: float) -> dict:
    """The VIS-1 numbers of one NOW record: every row WITH a 3-D label (all of ours) is an occluder; rows inside
    ``STORE_SCOPE`` are stored, keyed by their position in the record. Inputs are float32 as the trainer's target
    block carries them (``targets_from_join`` -> float32)."""
    _stack()
    from tanitad.data.vis1 import in_store_scope
    from tanitad.data.vis1_zbuffer import zbuffer
    f32 = lambda v: float(np.float32(v))  # noqa: E731
    bx = np.array([[f32(a["cx"]), f32(a["cy"]), f32(a["cz"]), f32(a["l"]), f32(a["w"]), f32(a["h"]),
                    f32(a["yaw"])] for a in agents]) if agents else np.zeros((0, 7))
    R = np.asarray(rig["virtual_R_cam_to_lidar"], dtype=np.float64)
    t0 = np.asarray(rig["f0_sensor2lidar_translation"], dtype=np.float64)
    t = np.array([t0[0], t0[1], t0[2] - float(road_z)])
    nf, ni, nv, vr, _ = zbuffer(bx, R, t, obs)
    keep = in_store_scope(bx[:, 0], bx[:, 1]) if len(bx) else np.zeros(0, bool)
    rows = [j for j in range(len(agents)) if keep[j]]
    return {"row": np.asarray(rows, np.int16), "track": np.asarray([int(agents[j]["track_id"]) for j in rows], np.int64),
            "n_full": np.asarray([nf[j] for j in rows], np.int32), "n_img": np.asarray([ni[j] for j in rows], np.int32),
            "n_vis": np.asarray([nv[j] for j in rows], np.int32), "vis_rows": np.asarray([vr[j] for j in rows], np.int32),
            "cx": np.asarray([np.float32(agents[j]["cx"]) for j in rows], np.float32),
            "cy": np.asarray([np.float32(agents[j]["cy"]) for j in rows], np.float32),
            "n_zh": len(agents), "n_valid": len(agents)}


def cmd_build(a) -> int:
    t_all = time.time()
    F = _frames416()
    is_ego = _ego_fp()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    road_z = float(a.road_z)
    tt = json.load(open(a.token_table, encoding="utf-8"))["rows"]
    if a.logs:
        tt = [r for r in tt if r["log_name"] in set(a.logs)]
    if a.limit:
        tt = tt[: a.limit]
    by_log: dict = {}
    for r in tt:
        by_log.setdefault(r["log_name"], []).append(r)
    shard = None
    if a.shard:
        k, n = (int(x) for x in str(a.shard).split("/"))
        if not (0 <= k < n):
            raise C.NTRefused(f"[nt-joins] --shard {a.shard}: need 0 <= k < N")
        keep = sorted(by_log)[k::n]
        by_log = {ln: by_log[ln] for ln in keep}
        tt = [r for r in tt if r["log_name"] in by_log]
        shard = {"k": k, "n": n, "n_logs": len(keep)}
    census, gaps = Counter(), Counter()
    rigs, rig_cache, rig_of_token = {}, {}, {}
    vis = {}
    n_rec = 0
    f2 = open(out / "nt_agents.jsonl", "w", encoding="utf-8", newline="\n")
    f3 = open(out / "nt_agents_3d.jsonl", "w", encoding="utf-8", newline="\n")
    try:
        for ln in sorted(by_log):
            fr = C.load_log(Path(a.logs_dir) / f"{ln}.pkl")
            ts = np.array([int(f["timestamp"]) for f in fr], dtype=np.int64)
            for r in sorted(by_log[ln], key=lambda r: r["frame_idx"]):
                i = int(r["frame_idx"])
                if str(fr[i]["token"]) != r["token"]:
                    raise C.NTRefused(f"[nt-joins] token table row {r['token']} != log frame {fr[i]['token']}")
                cams = fr[i]["cams"]
                rk = F.E2BF.rig_key(cams)
                if rk not in rig_cache:
                    rec = F.rig_record(cams)
                    _g, src, M = F.E2BF.build_map(cams, (int(a.img_h), int(a.img_w)))
                    if float(np.abs(np.asarray(rec["virtual_R_cam_to_lidar"]) - M).max()) != 0.0:
                        raise C.NTRefused("[nt-joins] frames416.virtual_rotation != build_map's M")
                    rig_cache[rk] = (rec, src >= 0)
                    rigs[rk] = rig_entry(rec, road_z)
                rec_rig, obs = rig_cache[rk]
                rig_of_token[r["clip_id"]] = rk
                t0 = int(ts[i])
                for j, fidx in ((i - 1, FRAME_NOW - 5), (i, FRAME_NOW), (i + 1, FRAME_NOW + 5)):
                    if j < 0 or j >= len(fr):
                        gaps["neighbour_absent_log_edge"] += 1
                        continue
                    if j != i:
                        dt = abs(int(ts[j]) - t0) / 1e6
                        if not (NEIGHBOUR_DT_S[0] <= dt <= NEIGHBOUR_DT_S[1]):
                            gaps["neighbour_dropped_dt"] += 1
                            continue
                    ags = agents_of_frame(fr[j], road_z, census, is_ego)
                    t_s = round(C.episode_now_s() + (int(ts[j]) - t0) / 1e6, 6)
                    base = {"clip_id": r["clip_id"], "frame": fidx + C.N_STACK - 1, "frame_idx": fidx, "t_s": t_s}
                    two = [{k: v for k, v in g.items() if k not in ("cz", "h")} for g in ags]
                    f2.write(json.dumps(dict(base, agents=two), separators=(",", ":")) + "\n")
                    f3.write(json.dumps(dict(base, agents=ags), separators=(",", ":")) + "\n")
                    n_rec += 1
                    if fidx == FRAME_NOW:
                        vis[r["clip_id"]] = vis1_rows_now(ags, rec_rig, obs, road_z)
                        census["now_frames"] += 1
                        census["now_agents"] += len(ags)
    finally:
        f2.close()
        f3.close()
    table = {cid: {k: v for k, v in rigs[rk].items()} for cid, rk in rig_of_token.items()}
    C.json_dump(table, out / "nt_rig_extrinsics.json")
    C.json_dump({rk: {"rig_record": rig_cache[rk][0], "observed_frac": float(rig_cache[rk][1].mean())}
                 for rk in rig_cache}, out / "nt_rigs.json")
    for rk, (_r, obs) in rig_cache.items():
        np.save(out / f"nt_obs_{rk}.npy", obs)
    # VIS-1 per-clip parts (merged into one sidecar by `merge`)
    order = sorted(vis, key=C.sha12)
    parts = {"clip_id": np.asarray(order), "sha12": np.asarray([C.sha12(c) for c in order], "<U12")}
    for k in ("row", "track", "n_full", "n_img", "n_vis", "vis_rows", "cx", "cy"):
        parts[k] = np.concatenate([vis[c][k] for c in order]) if order else np.zeros(0)
    parts["n_rows_per_clip"] = np.asarray([len(vis[c]["row"]) for c in order], np.int64)
    parts["n_zh"] = np.asarray([vis[c]["n_zh"] for c in order], np.int32)
    parts["n_valid"] = np.asarray([vis[c]["n_valid"] for c in order], np.int32)
    np.savez(out / "nt_vis1_parts.npz", **parts)
    rep = {"what": "H-NT-DECIDE-1 E3 NavSim joins (parts)", "shard": shard, "n_tokens": len(tt), "n_logs": len(by_log),
           "n_records": n_rec, "n_rigs": len(rig_cache), "road_z_m": road_z,
           "road_plane_source": {"md5": C.ROAD_PLANE_FILE_MD5, "median_m": C.ROAD_PLANE_MEDIAN_M},
           "class_map": C.CLASS_MAP, "census": dict(sorted(census.items())), "neighbours": dict(gaps),
           "vis1_rows": int(parts["n_rows_per_clip"].sum()) if order else 0,
           "rig_cam_height_m": sorted({round(v["z"], 4) for v in rigs.values()}),
           "img_hw_for_build_map": [int(a.img_h), int(a.img_w)], "wall_s": round(time.time() - t_all, 1),
           "code": {"nt_joins.py": C.md5_file(__file__), "nt_common.py": C.md5_file(C.__file__)}}
    C.json_dump(rep, out / "JOINS_REPORT.json")
    print(json.dumps({k: rep[k] for k in ("n_tokens", "n_records", "n_rigs", "vis1_rows", "neighbours", "wall_s")}))
    print(json.dumps(rep["census"]))
    return 0


# ------------------------------------------------------------------------------------------------------- merge
def _iter_jsonl(path):
    op = lzma.open if str(path).endswith(".xz") else open
    with op(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield line if line.endswith("\n") else line + "\n"


def _eval_clip_ids(eval_cache: str) -> list:
    return sorted(p.name[: -len(".v2ep.pt")] for p in Path(eval_cache).glob("*.v2ep.pt"))


def merge_join(nt_part: str, pai_join: str, eval_ids: set, out_xz: Path) -> dict:
    """NavSim records first (each clip contiguous, frames increasing), then the eval139 clips' PhysicalAI records
    VERBATIM (the bytes of their lines are not re-serialised)."""
    n_nt = n_pai = 0
    seen_pai = set()
    with lzma.open(out_xz, "wt", encoding="utf-8", preset=6) as fo:
        for line in _iter_jsonl(nt_part):
            fo.write(line)
            n_nt += 1
        for line in _iter_jsonl(pai_join):
            cid = line.split('"', 4)[3] if line.startswith('{"clip_id":"') else json.loads(line)["clip_id"]
            if cid in eval_ids:
                fo.write(line)
                n_pai += 1
                seen_pai.add(cid)
    return {"path": str(out_xz), "n_navsim_records": n_nt, "n_eval139_records": n_pai,
            "n_eval139_clips": len(seen_pai), "eval139_missing": sorted(set(eval_ids) - seen_pai)[:5]}


def _npz_eager(path) -> dict:
    """Every array of an ``.npz`` read ONCE into a plain dict. ``np.load(...)[k]`` on an ``NpzFile`` decompresses the
    whole array on EVERY access; indexing it inside a per-clip loop is O(clips x array) in time, and keeping slices of
    it alive is O(clips x array) in MEMORY (each slice pins its own full parent)."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def merge_vis1(parts_path: str, pai_sidecar: str, eval_sha12: set, out: Path, zb_meta: dict) -> dict:
    _stack()
    from tanitad.data import vis1 as V
    P = _npz_eager(parts_path)                       # each array read ONCE (see cmd_gather: the 122.9 GB OOM)
    S = V.VIS1Sidecar(pai_sidecar)
    a = S.a
    per = {}
    # NavSim clips: one frame (f = 7) each
    ptr = 0
    for ci, s12 in enumerate(P["sha12"].tolist()):
        n = int(P["n_rows_per_clip"][ci])
        sl = slice(ptr, ptr + n)
        ptr += n
        per[s12] = {"split": "nt", "mask": True, "frame_f": np.asarray([FRAME_NOW], np.int32),
                    "frame_n_zh": np.asarray([int(P["n_zh"][ci])], np.int32),
                    "frame_n_valid": np.asarray([int(P["n_valid"][ci])], np.int32),
                    "frame_cnt": [n], **{k: P[k][sl] for k in ("row", "track", "n_full", "n_img", "n_vis",
                                                                "vis_rows", "cx", "cy")}}
    # eval139 clips: copied from the refcv8 sidecar
    for s12 in sorted(eval_sha12):
        ci = S.clip_index.get(s12)
        if ci is None:
            raise C.NTRefused(f"[nt-joins] eval139 clip {s12} absent from {pai_sidecar}")
        if s12 in per:
            raise C.NTRefused(f"[nt-joins] sha12 collision NavSim / eval139: {s12}")
        f0, f1 = int(a["clip_frame_ptr"][ci]), int(a["clip_frame_ptr"][ci + 1])
        r0, r1 = int(a["frame_ptr"][f0]), int(a["frame_ptr"][f1])
        per[s12] = {"split": str(a["clip_split"][ci]), "mask": bool(a["clip_mask_ftheta"][ci]),
                    "frame_f": a["frame_f"][f0:f1], "frame_n_zh": a["frame_n_zh"][f0:f1],
                    "frame_n_valid": a["frame_n_valid"][f0:f1],
                    "frame_cnt": [int(a["frame_ptr"][k + 1] - a["frame_ptr"][k]) for k in range(f0, f1)],
                    **{k: a[k][r0:r1] for k in ("row", "track", "n_full", "n_img", "n_vis", "vis_rows", "cx", "cy")}}
    order = sorted(per)
    A = {k: [] for k in ("row", "track", "n_full", "n_img", "n_vis", "vis_rows", "cx", "cy")}
    fr = {k: [] for k in ("frame_clip", "frame_f", "frame_n_zh", "frame_n_valid")}
    frame_ptr, clip_frame_ptr, n_rows = [0], [0], 0
    for ci, s12 in enumerate(order):
        c = per[s12]
        for k in A:
            A[k].append(np.asarray(c[k]))
        nf = len(c["frame_f"])
        fr["frame_clip"].append(np.full(nf, ci, np.int32))
        for k in ("frame_f", "frame_n_zh", "frame_n_valid"):
            fr[k].append(np.asarray(c[k], np.int32))
        for cnt in c["frame_cnt"]:
            n_rows += int(cnt)
            frame_ptr.append(n_rows)
        clip_frame_ptr.append(clip_frame_ptr[-1] + nf)
    arrays = {k: np.concatenate(v) for k, v in A.items()}
    for k, dt in (("row", np.int16), ("track", np.int64), ("n_full", np.int32), ("n_img", np.int32),
                  ("n_vis", np.int32), ("vis_rows", np.int32), ("cx", np.float32), ("cy", np.float32)):
        arrays[k] = arrays[k].astype(dt)
    for k, v in fr.items():
        arrays[k] = np.concatenate(v).astype(np.int32)
    arrays["frame_ptr"] = np.asarray(frame_ptr, np.int64)
    arrays["clip_frame_ptr"] = np.asarray(clip_frame_ptr, np.int64)
    arrays["clip_sha12"] = np.asarray(order, dtype="<U12")
    arrays["clip_mask_ftheta"] = np.asarray([bool(per[s]["mask"]) for s in order], np.bool_)
    arrays["clip_split"] = np.asarray([per[s]["split"] for s in order], dtype="<U5")
    import hashlib
    content = hashlib.sha256()
    for k in sorted(arrays):
        content.update(k.encode())
        content.update(np.ascontiguousarray(arrays[k]).tobytes())
    meta = {"splits": {"nt": int(sum(1 for s in order if per[s]["split"] == "nt")),
                       "eval139": len(eval_sha12)},
            "sources": {"navsim_parts": [str(parts_path), C.md5_file(parts_path)],
                        "eval139_from": [str(pai_sidecar), S.sha256]},
            "zbuffer": zb_meta, "store_scope": V.STORE_SCOPE, "window": int(S.meta.get("window", 8)),
            "n_stack": int(S.meta.get("n_stack", 3)),
            "frame_key": "the 2-D agent join's frame_idx (V3Dataset._agent_item's f = t + W - 1)",
            "row_key": "the agent's position in the join record (the padded block's row, pre-truncation)",
            "occluders": "every valid row with a 3-D label",
            "navsim_observed_mask": "NavSim clips (split 'nt'): the stitch's src >= 0 map (frames416 / E2 build_map), "
                                    "stored under clip_mask_ftheta=True -- the clip's OWN observed mask, not image bounds",
            "n_clips": len(order), "n_frames": int(len(arrays["frame_f"])), "n_rows": int(len(arrays["row"])),
            "n_clips_bounds_only": int((~arrays["clip_mask_ftheta"]).sum()),
            "allow_bounds_only_mask": bool(S.meta.get("allow_bounds_only_mask", False)),
            "content_sha256_of_arrays": content.hexdigest()}
    dig = V.write_sidecar(out, arrays, meta)
    back = V.VIS1Sidecar(out)
    return {"path": str(out), **dig, "n_clips": meta["n_clips"], "n_rows": meta["n_rows"], "stamp": back.stamp()}


def cmd_merge(a) -> int:
    t0 = time.time()
    _stack()
    from tanitad.data import vis1_zbuffer as VZ
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    eval_ids = set(_eval_clip_ids(a.eval_cache))
    if not eval_ids:
        raise C.NTRefused(f"[nt-joins] no *.v2ep.pt in {a.eval_cache}")
    parts = Path(a.parts)
    rep = {"eval139_clips": len(eval_ids)}
    rep["agent_join"] = merge_join(str(parts / "nt_agents.jsonl"), a.pai_join, eval_ids,
                                   out / f"{a.tag}_agents.jsonl.xz")
    rep["join3d"] = merge_join(str(parts / "nt_agents_3d.jsonl"), a.pai_join3d, eval_ids,
                               out / f"{a.tag}_agents_3d.jsonl.xz")
    from tanitad.data import join_meta as JM
    for key in ("agent_join", "join3d"):
        p = rep[key]["path"]
        # the builder DECLARES which artifact its digest covers (M18: the compressed .xz), so the trainer's
        # `_verify_agent_join` checks it instead of warning "no sidecar"
        side = {"built_by": "nt_joins.py merge (H-NT-DECIDE-1 E3)", "builder_md5": C.md5_file(__file__),
                "navsim_records": rep[key]["n_navsim_records"], "eval139_records": rep[key]["n_eval139_records"],
                "eval139_from": a.pai_join if key == "agent_join" else a.pai_join3d}
        side = JM.attach(side, C.md5_file(p), scope="compressed", filename=Path(p).name, algo="md5",
                         declared_by="nt_joins.py")
        C.json_dump(side, p + ".meta.json")
        ev = JM.verify(p, side, where=p + ".meta.json")          # the trainer's own check must accept it
        rep[key]["meta"] = {"sidecar": p + ".meta.json", "verified": ev}
    zb = {"module_md5": C.md5_file(VZ.__file__), "audit_source": VZ.AUDIT_SOURCE,
          "function_sha256": VZ.function_sha256(Path(VZ.__file__).read_text(encoding="utf-8"), "zbuffer")}
    rep["vis1"] = merge_vis1(str(parts / "nt_vis1_parts.npz"), a.pai_vis1,
                             {C.sha12(c) for c in eval_ids}, out / f"{a.tag}_vis1_sidecar.npz", zb)
    nt_tab = json.load(open(parts / "nt_rig_extrinsics.json", encoding="utf-8"))
    pai_tab = json.load(open(a.pai_extrinsics, encoding="utf-8"))
    miss = sorted(eval_ids - set(pai_tab))
    if miss:
        raise C.NTRefused(f"[nt-joins] {len(miss)} eval139 clips absent from {a.pai_extrinsics}")
    clash = set(nt_tab) & set(pai_tab)
    if clash:
        raise C.NTRefused(f"[nt-joins] clip ids in both rig tables: {sorted(clash)[:3]}")
    merged = dict(nt_tab)
    merged.update({c: pai_tab[c] for c in sorted(eval_ids)})
    C.json_dump(merged, out / f"{a.tag}_extrinsics.json")
    rep["extrinsics"] = {"path": str(out / f"{a.tag}_extrinsics.json"), "n_navsim": len(nt_tab),
                         "n_eval139": len(eval_ids)}
    for k in ("agent_join", "join3d"):
        rep[k]["md5"] = C.md5_file(rep[k]["path"])
    rep["extrinsics"]["md5"] = C.md5_file(rep["extrinsics"]["path"])
    rep["wall_s"] = round(time.time() - t0, 1)
    C.json_dump(rep, out / "MERGE_REPORT.json")
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "stamp"})
                      for k, v in rep.items()}, indent=1, default=str)[:4000])
    return 0


VIS_ROW_KEYS = ("row", "track", "n_full", "n_img", "n_vis", "vis_rows", "cx", "cy")


def cmd_gather(a) -> int:
    """Concatenate ``build --shard k/N`` part dirs into one part dir. REFUSES a clip id in two shards, a rig key whose
    records differ between shards, and a shard whose JOINS_REPORT says another N -- and asserts every shard 0..N-1 is
    present exactly once (a missing shard is a silently missing ~1/N of the corpus)."""
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reps = [json.load(open(Path(d) / "JOINS_REPORT.json", encoding="utf-8")) for d in a.shards]
    ks = sorted(int((r.get("shard") or {}).get("k", -1)) for r in reps)
    ns = {int((r.get("shard") or {}).get("n", -1)) for r in reps}
    if len(ns) != 1 or ks != list(range(next(iter(ns)))):
        raise C.NTRefused(f"[nt-joins] gather: shards k={ks} of N={sorted(ns)} -- every shard 0..N-1 exactly once")
    tab, rigs, seen = {}, {}, set()
    vis = {}
    n2 = n3 = 0
    with open(out / "nt_agents.jsonl", "w", encoding="utf-8", newline="\n") as f2, \
            open(out / "nt_agents_3d.jsonl", "w", encoding="utf-8", newline="\n") as f3:
        for d in a.shards:
            d = Path(d)
            t = json.load(open(d / "nt_rig_extrinsics.json", encoding="utf-8"))
            dup = set(t) & set(tab)
            if dup:
                raise C.NTRefused(f"[nt-joins] gather: clip ids in two shards: {sorted(dup)[:3]}")
            tab.update(t)
            for rk, v in json.load(open(d / "nt_rigs.json", encoding="utf-8")).items():
                if rk in rigs and rigs[rk] != v:
                    raise C.NTRefused(f"[nt-joins] gather: rig {rk} differs between shards")
                rigs[rk] = v
                o = d / f"nt_obs_{rk}.npy"
                if not (out / o.name).exists():
                    np.save(out / o.name, np.load(o))
            for src, dst in (("nt_agents.jsonl", f2), ("nt_agents_3d.jsonl", f3)):
                with open(d / src, encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            dst.write(line if line.endswith("\n") else line + "\n")
                            if src == "nt_agents.jsonl":
                                n2 += 1
                            else:
                                n3 += 1
            # ⛔ EVERY array is read from the npz EXACTLY ONCE per shard. ``NpzFile[k]`` decompresses the WHOLE array on
            # each access and a slice of it is a VIEW that pins that full copy: the per-clip ``z[k][lo:hi]`` this
            # replaces held clips x keys full copies alive and was OOM-killed by the kernel at 122.9 GB anon RSS on
            # Thor (2026-10-21 22:50:38, global OOM; 99,824 clips). Pinned by tests/test_nt_code.py (npz read count).
            z = _npz_eager(d / "nt_vis1_parts.npz")
            off = np.concatenate([[0], np.cumsum(z["n_rows_per_clip"])]).astype(np.int64)
            for i, cid in enumerate(z["clip_id"].tolist()):
                if cid in seen:
                    raise C.NTRefused(f"[nt-joins] gather: VIS-1 clip {cid} in two shards")
                seen.add(cid)
                lo, hi = int(off[i]), int(off[i + 1])
                vis[cid] = {k: z[k][lo:hi] for k in VIS_ROW_KEYS}
                vis[cid].update(n_zh=int(z["n_zh"][i]), n_valid=int(z["n_valid"][i]), sha12=str(z["sha12"][i]))
    C.json_dump(tab, out / "nt_rig_extrinsics.json")
    C.json_dump(rigs, out / "nt_rigs.json")
    order = sorted(vis, key=lambda c: vis[c]["sha12"])
    parts = {"clip_id": np.asarray(order), "sha12": np.asarray([vis[c]["sha12"] for c in order], "<U12")}
    for k in VIS_ROW_KEYS:
        parts[k] = np.concatenate([vis[c][k] for c in order]) if order else np.zeros(0)
    parts["n_rows_per_clip"] = np.asarray([len(vis[c]["row"]) for c in order], np.int64)
    parts["n_zh"] = np.asarray([vis[c]["n_zh"] for c in order], np.int32)
    parts["n_valid"] = np.asarray([vis[c]["n_valid"] for c in order], np.int32)
    np.savez(out / "nt_vis1_parts.npz", **parts)
    census: Counter = Counter()
    for r in reps:
        census.update(r.get("census") or {})
    rep = {"what": "H-NT-DECIDE-1 E3 NavSim joins (gathered parts)", "shards": [str(x) for x in a.shards],
           "n_tokens": sum(int(r["n_tokens"]) for r in reps), "n_logs": sum(int(r["n_logs"]) for r in reps),
           "n_records": n2, "n_records_3d": n3, "n_rigs": len(rigs), "n_vis1_clips": len(order),
           "road_z_m": reps[0]["road_z_m"], "road_plane_source": reps[0]["road_plane_source"],
           "class_map": reps[0]["class_map"], "census": dict(sorted(census.items())),
           "wall_s": round(time.time() - t0, 1), "code": {"nt_joins.py": C.md5_file(__file__)}}
    if n2 != n3 or n2 != sum(int(r["n_records"]) for r in reps):
        raise C.NTRefused(f"[nt-joins] gather: record counts disagree ({n2} / {n3} / shard sum)")
    C.json_dump(rep, out / "JOINS_REPORT.json")
    print(json.dumps({k: rep[k] for k in ("n_tokens", "n_logs", "n_records", "n_rigs", "n_vis1_clips", "wall_s")}))
    return 0

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--logs-dir", required=True)
    b.add_argument("--token-table", required=True)
    b.add_argument("--logs", nargs="*", default=None)
    b.add_argument("--limit", type=int, default=0)
    b.add_argument("--road-z", type=float, default=C.ROAD_PLANE_MEDIAN_M)
    b.add_argument("--img-h", type=int, default=1080, help="native camera height for build_map (nuPlan 1080)")
    b.add_argument("--img-w", type=int, default=1920, help="native camera width for build_map (nuPlan 1920)")
    b.add_argument("--shard", default=None, help="k/N: this process builds logs[k::N] of the sorted log list")
    b.add_argument("--out", required=True)
    g = sub.add_parser("gather")
    g.add_argument("--shards", nargs="+", required=True, help="the `build --shard k/N` out dirs, all N of them")
    g.add_argument("--out", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--parts", required=True, help="a `build` --out directory")
    m.add_argument("--eval-cache", required=True, help="the eval139 v2 cache (clip ids from its listing)")
    m.add_argument("--pai-join", required=True)
    m.add_argument("--pai-join3d", required=True)
    m.add_argument("--pai-vis1", required=True)
    m.add_argument("--pai-extrinsics", required=True)
    m.add_argument("--tag", default="nt_train_plus_eval139")
    m.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"build": cmd_build, "gather": cmd_gather, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
