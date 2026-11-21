"""Front-camera SAM3 map PRODUCTION for the v7 training corpus (PI 2026-09-14: "After this we will start the production for the whole
training corpus"). One process, one model build; per clip:
  1 SEQUENCE   native f-theta front frames at 5 Hz from the clip's front-wide mp4 (build_front_seq_native's recipe), f-theta
               calibration from the chunk parquets, egomotion poses from the B1 bundle, ground from the ego path (no LiDAR) --
               written to a per-clip scratch root
  2 MAP        the approved per-frame configuration (env flags as sam3map_front_fast.py, default = spdF4a) + the approved
               pipeline measures (ego masks on the loaded model in fp32; refine / consensus / renderer v5m / compose unmodified,
               in the background) -> the composed map r of the clip
  3 EXPORT     per-frame BEV ground truth on the v2ep EPISODE grid the BEV head trains on (camera timestamps -> linspace at 10 Hz,
               exactly v2_compressed._resampled), the rig pose at each frame's exposure instant interpolated from egomotion, the
               same grids / soft class fractions / content checks as sam3map_export_gt.py -> <out>/<sha12>.sam3mapgt.npz + report
  4 LEDGER     <out>/manifest.jsonl, one line per clip (status, timings, checks); a clip already OK in the ledger is skipped
Inputs present locally for every corpus clip: B1 bundle egomotion + camera timestamps + index. Per clip the front mp4 and the chunk's
calibration parquets must be in SRC_CAM / SRC_CAL (the production fetch step fills them; nothing here downloads).
The pipeline's own per-clip working files (sequence, intermediate npz) are removed after that clip's export passed its checks.
Corpus run (2026-09-15, all env, defaults = the validated behaviour): PROD_WAIT_S waits for the input feeder to place a clip's mp4
(atomic rename, so existence = complete); PROD_MAX_FAILS gives a clip up after that many FAIL rows (FAIL-EXPORT-CHECKS after one: the map
is deterministic, C-REPRO); <out>/skip.txt lists clips the supervisor took out of a crash loop; PROD_KEEP_FAILED_WORK keeps the working
files of only that many failed clips. A failed sequence build no longer poisons every later clip (the next prebuild is submitted before
the result is read); a CPU-stage exception is a FAIL row, not a dead process; 3 GPU-stage failures in a row or an input timeout end
the process with code 3 / 4 so the supervisor restarts it with a fresh CUDA context; nothing to do prints ZZPROD-NOTHING-TO-DO.
Usage: sam3map_prod.py <clip id list file> <out dir> [max clips]"""
import collections, hashlib
import concurrent.futures as cf
import io, json, math, os, shutil, subprocess, sys, tarfile, time, traceback
from pathlib import Path
CAM = "CAM_F0"
os.environ["SAM3MAP_CAM"] = CAM; os.environ["SAM3MAP_VIEWS"] = CAM
SCR = Path(os.environ.get("PROD_SCRATCH", "/home/nvidia/sam3map/prod_scratch")); SCR.mkdir(parents=True, exist_ok=True)
os.environ["SAM3MAP_ROOT"] = str(SCR)
for k, v in (("MODE", "fast"), ("HALF", "fp16enc"), ("BATCH", "19"), ("ASYNC", "2")):
    os.environ.setdefault(k, v)
sys.path.insert(0, "/home/nvidia/sam3map"); sys.path.insert(0, "/home/nvidia/sam3map/eval"); sys.path.insert(0, "/home/nvidia/sam3paint")
import numpy as np
import pandas as pd
import av
import torch
from PIL import Image
import sam3map_front_fast as FF
import sam3_smoke
import sam3map_refine_v6 as R
import sam3map_export_gt as X
from ftheta_pinhole import quat_to_R
from build_front_seq import path_ground, pose_at

SM = Path("/home/nvidia/sam3map"); EV = SM / "eval"; PY = sys.executable
B1 = Path("/home/nvidia/data/b1-bundle")
SRC_CAM = Path(os.environ.get("SRC_CAM", "/home/nvidia/sam3map/data/frontwide"))
SRC_CAL = Path(os.environ.get("SRC_CAL", "/home/nvidia/sam3map/data/calib"))
FEAT = "camera_front_wide_120fov"; TARGET_HZ = 10.0; TAG = "prod"
PP = "/home/nvidia/sam3vendor:/home/nvidia/sam3paint:/home/nvidia/sam3map"
IDX = pd.read_parquet(B1 / "index" / "clip_to_chunk.parquet")
WAIT_S = float(os.environ.get("PROD_WAIT_S", "0"))
MAX_FAILS = int(os.environ.get("PROD_MAX_FAILS", "0"))                         # 0 = a failed clip is always retried
KEEP_FAILED_WORK = int(os.environ.get("PROD_KEEP_FAILED_WORK", "-1"))          # -1 = keep every failed clip's working files


class InputTimeout(Exception):
    pass


def sha12(clip):
    return hashlib.sha256(clip.encode()).hexdigest()[:12]


def given_up(rows_, out):
    """sha12s not to attempt again: FAIL-EXPORT-CHECKS once, any FAIL MAX_FAILS times, or listed in <out>/skip.txt (same rule in corpus_feeder.py)"""
    fails, checks = collections.Counter(), set()
    for r_ in rows_:
        s = str(r_.get("status", ""))
        if s.startswith("FAIL"):
            fails[r_["clip_sha12"]] += 1
            if s == "FAIL-EXPORT-CHECKS":
                checks.add(r_["clip_sha12"])
    gone = set((Path(out) / "skip.txt").read_text().split()) if (Path(out) / "skip.txt").exists() else set()
    if MAX_FAILS:
        gone |= checks | {s for s, n in fails.items() if n >= MAX_FAILS}
    return gone


def tar_member(tar_path, name):
    with tarfile.open(tar_path) as t:
        return pd.read_parquet(io.BytesIO(t.extractfile(name).read()))


def poses_interp(ego, t_us):
    """rig poses at t_us: linear position, quaternion nlerp (egomotion samples bracket every camera frame)."""
    ts = ego["timestamp"].to_numpy(np.float64)
    i = np.clip(np.searchsorted(ts, t_us), 1, len(ts) - 1); a = ((t_us - ts[i - 1]) / np.maximum(ts[i] - ts[i - 1], 1.0)).clip(0, 1)
    P = ego[["x", "y", "z"]].to_numpy(np.float64); Q = ego[["qw", "qx", "qy", "qz"]].to_numpy(np.float64)
    p = P[i - 1] * (1 - a)[:, None] + P[i] * a[:, None]
    q0, q1 = Q[i - 1], Q[i]; q1 = np.where((np.sum(q0 * q1, axis=1) < 0)[:, None], -q1, q1)
    q = q0 * (1 - a)[:, None] + q1 * a[:, None]; q /= np.linalg.norm(q, axis=1, keepdims=True)
    Ts = np.repeat(np.eye(4)[None], len(t_us), axis=0)
    for n in range(len(t_us)):
        Ts[n, :3, :3] = quat_to_R(q[n, 1], q[n, 2], q[n, 3], q[n, 0]); Ts[n, :3, 3] = p[n]
    return Ts


def build_sequence(clip):
    """build_front_seq_native.build with explicit sources; returns (c8, sha12, ego, t_cam)"""
    t_wait = time.time()
    while WAIT_S > 0 and not (SRC_CAM / f"{clip}.mp4").exists():               # the feeder places the file by atomic rename
        if time.time() - t_wait > WAIT_S:
            raise InputTimeout(f"{sha12(clip)}.mp4 absent after {WAIT_S:.0f}s")
        time.sleep(10)
    c8 = clip[:8]; sha = sha12(clip); chunk = int(IDX.loc[clip, "chunk"])
    idf = pd.read_parquet(SRC_CAL / f"camera_intrinsics.chunk_{chunk:04d}.parquet").reset_index()
    edf = pd.read_parquet(SRC_CAL / f"sensor_extrinsics.chunk_{chunk:04d}.parquet").reset_index()
    r = idf[(idf["clip_id"] == clip) & (idf["camera_name"] == FEAT)].iloc[0]
    poly = np.array([float(r[f"fw_poly_{i}"]) for i in range(5)])
    e = edf[(edf["clip_id"] == clip) & (edf["sensor_name"] == FEAT)].iloc[0]
    R_s = quat_to_R(float(e["qx"]), float(e["qy"]), float(e["qz"]), float(e["qw"])); t_s = np.array([float(e["x"]), float(e["y"]), float(e["z"])])
    ego = tar_member(B1 / "egomotion" / "egomotion_alpamayo.tar", f"{clip}.parquet").sort_values("timestamp").reset_index(drop=True)
    times = tar_member(B1 / "timestamps" / "timestamps.tar", f"{clip}.timestamps.parquet")["timestamp"].to_numpy(np.float64)
    t_refs = np.arange(times[0] + 0.5e6, times[-1] - 0.5e6, 0.2e6)
    want = {}
    for t in t_refs:
        want.setdefault(int(np.argmin(np.abs(times - t))), []).append(float(t))
    sd = SCR / f"seq_{c8}"
    if sd.exists():
        shutil.rmtree(sd)                                                     # a previous interrupted attempt's working files
    sd.mkdir(parents=True)
    # frame poses exactly as the validated builder (build_front_seq_native: nearest 100 Hz egomotion sample). MEASURED 2026-09-14 on
    # 1f1f05ca011d: interpolated poses (<= 5 cm different) change the map at cell level (thin-class IoU ~0.5, BEV-grid fractions within
    # 0.002-0.008) -- production must reproduce the validated map, so interpolation is used only by the export
    poses = {}
    with av.open(str(SRC_CAM / f"{clip}.mp4")) as cont:
        st = cont.streams.video[0]; st.thread_type = "AUTO"
        for i, fr in enumerate(cont.decode(st)):
            if i not in want:
                continue
            img = fr.to_ndarray(format="rgb24")
            for t in want[i]:
                tok = f"{c8}_t{int(round((t - times[0]) / 1e3)):05d}"
                d = sd / tok; (d / "images").mkdir(parents=True, exist_ok=True)
                Image.fromarray(img).save(d / "images" / f"{CAM}.jpg", quality=95)
                np.savez(d / "calib.npz", ftheta_poly=poly[None], ftheta_cx=np.array([float(r["cx"])]), ftheta_cy=np.array([float(r["cy"])]),
                         sensor2lidar_rotation=R_s[None], sensor2lidar_translation=t_s[None], image_wh=np.array([img.shape[1], img.shape[0]]))
                (d / "frame.json").write_text(json.dumps({"dataset_type": "native", "camera_model": "ftheta", "cam_order": [CAM]}), encoding="utf-8")
                T = pose_at(ego, times[i])
                np.save(d / "lidar.npy", path_ground(ego, T, times[i]))
                (d / "meta.json").write_text(json.dumps({"token": tok, "clip_sha12": sha, "t_ref_us": t, "t_frame_us": float(times[i]), "builder": "sam3map_prod",
                                                         "ground": "ego path -3..+6 s spread +-6 m (no LiDAR)", "chunk": chunk}), encoding="utf-8")
                poses[tok] = {"T_world_rig": T.tolist()}
            if i >= max(want):
                break
    (sd / "poses.json").write_text(json.dumps(poses), encoding="utf-8")
    return {"clip": clip, "c8": c8, "sha": sha, "ego": ego, "t_cam": times, "frames": len(poses), "split": str(IDX.loc[clip, "split"])}


def export_v2ep(info, wm_path, out_dir):
    """sam3map_export_gt.py's grids and checks, on the v2ep episode grid with interpolated poses."""
    t_cam = info["t_cam"]; span = t_cam[-1] - t_cam[0]
    n_target = max(int(span / 1e6 * TARGET_HZ), 4)
    t_query = np.linspace(t_cam[0], t_cam[-1], n_target); fidx = np.searchsorted(t_cam, t_query).clip(0, len(t_cam) - 1)
    t_img = t_cam[fidx]
    Ts = poses_interp(info["ego"], t_img.astype(np.float64))
    wm = dict(np.load(wm_path, allow_pickle=True))
    grids = {"cart": (X.cart_points(X.CART), X.CART), "polar": (X.polar_points(X.POLAR), X.POLAR), "polar48": (X.polar_points(X.POLAR48), X.POLAR48)}
    fx = (np.arange(X.FINE["shape"][0]) + 0.5) * X.FINE["cell_m"]; fy = -X.FINE["y_half_m"] + (np.arange(X.FINE["shape"][1]) + 0.5) * X.FINE["cell_m"]
    FXg, FYg = np.meshgrid(fx, fy, indexing="ij"); fine_xy = np.c_[FXg.ravel(), FYg.ravel()]
    arr = {f"{g}_frac": [] for g in grids}; arr["fine_codes"] = []
    for T in Ts:
        Rm, t = T[:2, :2], T[:2, 3]
        for g, ((pts, n_cells), spec) in grids.items():
            arr[f"{g}_frac"].append(X.fractions(X.lookup(wm, pts @ Rm.T + t), n_cells).reshape(9, *spec["shape"]))
        arr["fine_codes"].append(X.lookup(wm, fine_xy @ Rm.T + t).reshape(X.FINE["shape"]))
    out = Path(out_dir) / f"{info['sha']}.sam3mapgt.npz"
    meta = {"schema": "tanitad.sam3_map_gt/2", "frame": "rig", "frame_convention": "+x forward, +y LEFT, +z UP; origin rear axle on the road plane",
            "cartesian": {**X.CART, "row0": "x in [0, cell_m); col0 is y = -y_half (RIGHT), +y is LEFT"}, "polar": {**X.POLAR, "col0": "+hfov/2 = LEFT"},
            "polar48": {**X.POLAR48, "col0": "+hfov/2 = LEFT"}, "fine": {**X.FINE, "codes": "0-7 class, 255 not seen"}, "channels": X.CLASS_NAMES,
            "sub_samples_per_axis": X.SUB, "fraction_scale": 255,
            "time_grid": "v2ep EPISODE grid: t_query = linspace(t_cam[0], t_cam[-1], int(span_s * 10)), frame = first camera frame at or after "
                         "t_query (v2_compressed._resampled); axis0 = raw v2ep frame index; pose = egomotion interpolated at that frame's t_img",
            "source": {"map": "SAM3 front-camera map r (v6 extractor + v6s stripes + refine + consensus + renderer v5m + compose map-r options)",
                       "speed_config": {k: os.environ.get(k) for k in ("HALF", "BATCH", "BB_HALF", "ASYNC")}, "ground": "ego path (no LiDAR)",
                       "world_map_res_m": float(wm["res"]), "clip_sha12": info["sha"], "split": info["split"]},
            "non_causal": "labels use every frame of the clip; inference must never read this file", "label_only": True}
    np.savez_compressed(out, meta_json=json.dumps(meta), t_query_us=t_query, t_img_us=t_img.astype(np.int64), cam_frame_idx=fidx.astype(np.int32),
                        T_world_rig=Ts, **{k: np.stack(v) for k, v in arr.items()})
    cfr = np.stack(arr["cart_frac"]).astype(np.float64); seen = 1 - cfr[:, 8].mean() / 255; drivable = cfr[:, 1].mean() / 255
    bad_sum = int((np.abs(cfr.sum(axis=1) - 255) > 5).sum())
    fine = np.stack(arr["fine_codes"]); hit = {"real": [0, 0], "mirror": [0, 0]}
    for n, T in enumerate(Ts):
        fut = Ts[n + 1: n + 31, :2, 3]
        if not len(fut):
            continue
        q = (fut - T[:2, 3]) @ T[:2, :2]
        for name, sgn in (("real", 1.0), ("mirror", -1.0)):
            i = np.floor(q[:, 0] / X.FINE["cell_m"]).astype(int); j = np.floor((sgn * q[:, 1] + X.FINE["y_half_m"]) / X.FINE["cell_m"]).astype(int)
            ok = (i >= 0) & (i < X.FINE["shape"][0]) & (j >= 0) & (j < X.FINE["shape"][1]) & (q[:, 0] >= 2.0)
            c_all = fine[n, i[ok], j[ok]]; keep = c_all != 255; c = c_all[keep]
            hit[name][0] += int(np.isin(c, (1, 2, 3, 4, 6)).sum()); hit[name][1] += len(c)
            if name == "real":                                                # lateral points only: a straight path is its own mirror
                lat = np.abs(q[ok][keep][:, 1]) >= 0.75
                hit.setdefault("real_lateral", [0, 0]); hit["real_lateral"][0] += int(np.isin(c[lat], (1, 2, 3, 4, 6)).sum()); hit["real_lateral"][1] += int(lat.sum())
            else:
                lat = np.abs(q[ok][keep][:, 1]) >= 0.75
                hit.setdefault("mirror_lateral", [0, 0]); hit["mirror_lateral"][0] += int(np.isin(c[lat], (1, 2, 3, 4, 6)).sum()); hit["mirror_lateral"][1] += int(lat.sum())
    path_on_road = {k: (round(v[0] / v[1], 4) if v[1] else None) for k, v in hit.items()}
    n_lat = hit.get("real_lateral", [0, 0])[1]
    rl, ml = (path_on_road.get("real_lateral") or 0.0), (path_on_road.get("mirror_lateral") or 0.0)
    orientation = ("untestable: path too straight" if n_lat < 30 else "ok" if rl > ml + 0.02 else
                   "undecided: symmetric surroundings" if abs(rl - ml) <= 0.02 else "MIRRORED?")     # MEASURED: ties at 1.0 / 1.0 in wide drivable areas
    rep = {"frames": len(Ts), "cart_seen_share": round(float(seen), 4), "cart_drivable_share": round(float(drivable), 4), "cells_bad_sum": bad_sum,
           "ego_future_path_on_drivable": path_on_road, "lateral_path_points": n_lat, "orientation": orientation, "bytes": out.stat().st_size}
    ok = seen > 0.2 and drivable > 0.05 and bad_sum == 0 and path_on_road["real"] is not None and path_on_road["real"] >= 0.9 \
        and orientation != "MIRRORED?"
    (Path(out_dir) / f"{info['sha']}.sam3mapgt.report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return ok, rep


def cpu_stages(info, ego_path, out_dir):
    c8 = info["c8"]; t = {}
    env = dict(os.environ, SAM3MAP_ROOT=str(SCR), SAM3MAP_VIEWS=CAM, SAM3MAP_TAG=TAG, PYTHONPATH=PP, HF_HUB_OFFLINE="1", OMP_NUM_THREADS="4")
    steps = [("refine", [PY, str(EV / "refine_with_ego.py"), c8, str(ego_path)], SM, {}),
             ("consensus", [PY, "sam3map_consensus.py", c8, str(SM / f"{c8}_{TAG}"), str(SCR / f"{c8}_{TAG}c")], EV, {"CONSENSUS_NO_PROMOTE": "1"}),
             ("render", [PY, "sam3map_render_v5m.py", c8, str(SCR / f"{c8}_{TAG}c"), str(SCR / f"render5_{c8}_{TAG}m")], EV,
              {"SAM3MAP_COMPOSITE": "majority_v65", "SURFACE_MODE": "5", "PAINT_SHARE": "0.2", "PAINT_RULE": "p95", "CROSSWALK_STRIPES": "1", "NO_FRAMES": "1", "FIELDS": "1"}),
             ("compose", [PY, "compose.py", str(SCR / f"render5_{c8}_{TAG}m"), str(SCR / f"render5_{c8}_{TAG}r")], EV,
              {"LINE_EDGE_GAP": "1", "LINE_MIN_LEN_M": "1.0", "WLK_ISLAND_M2": "3", "EDGE_OBS": "near", "XWALK": "dirclose", "XWALK_LEN": "11"})]
    logf = Path(out_dir) / "logs" / f"{info['sha']}.log"; logf.parent.mkdir(exist_ok=True)
    with open(logf, "w") as fh:
        for name, cmd, cwd, extra in steps:
            a = time.time(); rc = subprocess.run(cmd, cwd=cwd, env=dict(env, **extra), stdout=fh, stderr=subprocess.STDOUT).returncode
            t[name] = round(time.time() - a, 1)
            if rc != 0:
                tidy(info, ego_path, out_dir, False)
                return {"status": f"FAIL-{name}", "stage_s": t}
            if name == "refine" and (SM / f"refine_{TAG}_{c8}.json").exists():    # refine's summary: into the clip's log folder
                os.replace(SM / f"refine_{TAG}_{c8}.json", logf.parent / f"{info['sha']}.refine.json")
            if name == "consensus":                                            # consensus writes its summary + atlas beside its output
                for src, suffix in ((SCR / f"consensus_{c8}_{TAG}c.json", "consensus.json"), (SCR / f"consensus_atlas_{c8}_{TAG}c.png", "consensus_atlas.png")):
                    if src.exists():
                        os.replace(src, logf.parent / f"{info['sha']}.{suffix}")
    wm = SCR / f"render5_{c8}_{TAG}r" / "worldmap.npz"
    a = time.time(); ok, rep = export_v2ep(info, wm, out_dir); t["export"] = round(time.time() - a, 1)
    shutil.copy2(wm, Path(out_dir) / f"{info['sha']}.worldmap.npz")
    tidy(info, ego_path, out_dir, ok)
    return {"status": "OK" if ok else "FAIL-EXPORT-CHECKS", "stage_s": t, "export": rep}


def tidy(info, ego_path, out_dir, ok):
    """the clip's working files: removed after its checks passed; a failed clip's are kept for the first PROD_KEEP_FAILED_WORK
    failed clips (-1 = always), the outputs and logs of every clip stay"""
    c8 = info["c8"]
    if not ok:
        kept = Path(out_dir) / "kept_failed_work.txt"
        n_kept = len(kept.read_text().split()) if kept.exists() else 0
        if KEEP_FAILED_WORK < 0 or n_kept < KEEP_FAILED_WORK:
            if KEEP_FAILED_WORK >= 0:
                with open(kept, "a") as fh:
                    fh.write(info["sha"] + "\n")
            return
    for p in (SCR / f"seq_{c8}", SM / f"{c8}_{TAG}raw", SM / f"{c8}_{TAG}", SCR / f"{c8}_{TAG}c", SCR / f"render5_{c8}_{TAG}m", SCR / f"render5_{c8}_{TAG}r", ego_path):
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()


def cpu_stages_safe(info, ego_path, out_dir):
    try:
        return cpu_stages(info, ego_path, out_dir)
    except Exception as e:                                                     # a FAIL row, never a dead production process
        traceback.print_exc()
        return {"status": "FAIL-CPU-STAGE", "error": f"{type(e).__name__}: {str(e)[:300]}"}


def write_row(ledger, row):
    row["t_row"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(ledger, "a") as fh:
        fh.write(json.dumps(row) + "\n")


def main():
    clips = [l.strip() for l in open(sys.argv[1]) if l.strip()]; out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else len(clips)
    ledger = out / "manifest.jsonl"
    rows_ = [json.loads(l) for l in open(ledger)] if ledger.exists() else []
    done = {r_["clip_sha12"] for r_ in rows_ if r_.get("status") == "OK"}      # failed clips are retried; the last line per clip wins
    gone = given_up(rows_, out) - done
    todo = [c for c in clips if sha12(c) not in done and sha12(c) not in gone][:limit]
    (out / "todo_head.txt").write_text("".join(sha12(c) + "\n" for c in todo[:2]))
    if not todo:
        print(f"ZZPROD-NOTHING-TO-DO {len(done)} OK, {len(gone)} given up", flush=True)
        return
    T0 = time.time(); proc, fix = FF.E.S.build(conf=0.25); fast = FF.FastSAM3(proc, FF.FLAGS)
    sam3_smoke.build = lambda conf=0.25: (proc, fix)
    print(f"model ready {time.time() - T0:.1f}s; {len(todo)} clips to do, {len(done)} already in the ledger, {len(gone)} given up", flush=True)
    bg = cf.ThreadPoolExecutor(max_workers=1); pre = cf.ThreadPoolExecutor(max_workers=1); post_pool = cf.ThreadPoolExecutor(max_workers=int(FF.FLAGS["ASYNC"]))
    jobs = []; gpu_fail_streak = 0

    def drain(wait):
        for r_, j_ in [x for x in jobs if wait or x[1].done()]:
            r_.update(j_.result()); jobs.remove((r_, j_)); write_row(ledger, r_)

    nxt = pre.submit(build_sequence, todo[0])
    for k, clip in enumerate(todo):
        tc = time.time(); rec = {"clip_sha12": sha12(clip), "t_start": time.strftime("%Y-%m-%dT%H:%M:%S")}
        try:
            fut = nxt                                                          # the next prebuild is queued BEFORE this result is read:
            nxt = pre.submit(build_sequence, todo[k + 1]) if k + 1 < len(todo) else None     # a failed build cannot poison later clips
            info = fut.result(); t_seq = time.time() - tc
            c8 = info["c8"]; sd = SCR / f"seq_{c8}"; toks = sorted(p.name for p in sd.iterdir() if p.is_dir())
            poses = json.loads((sd / "poses.json").read_text()); dst = SM / f"{c8}_{TAG}raw"
            if dst.exists():
                shutil.rmtree(dst)
            dst.mkdir(parents=True)
            ta = time.time(); futs = []
            for j in range(len(toks)):
                fi = FF.frame_inputs(c8, j, toks, sd, poses)
                st = fast.encode(fi["img"]); inst = fast.instances(st, FF.CLS_PT + FF.STRIPE_PT)

                def cpu(fi=fi, inst=inst, j=j):
                    cls, s_, evid = FF.CLASSIFY_INJECTED(FF._Shim(inst), fi["img"], (fi["camm"], fi["sgrid"]), True)
                    FF.post(fi, cls, s_, evid, inst["crosswalk stripe"], inst["white stripe on road"], dst, j, TAG)
                futs.append(post_pool.submit(cpu))
            for f in futs:
                f.result()
            t_ext = time.time() - ta
            saved = {n: fast.m.__dict__.pop(n) for n in list(FF.FastSAM3.STAGES.values()) if n in fast.m.__dict__}
            try:
                with torch.inference_mode():
                    ego, einfo = R.ego_masks(sd, toks)
            finally:
                for n, fn in saved.items():
                    setattr(fast.m, n, fn)
            ego_path = SCR / f"ego_{c8}.npz"
            np.savez_compressed(ego_path, _info=json.dumps(dict(einfo, _n_toks=len(toks))), **{kk: v for kk, v in ego.items() if kk != "_spread_m"})
            rec.update({"frames": len(toks), "split": info["split"], "seq_s": round(t_seq, 1), "extract_s": round(t_ext, 1), "s_per_frame": round(t_ext / len(toks), 3),
                        "cuda_alloc_gb": round(torch.cuda.memory_allocated() / 1e9, 2), "cuda_max_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
            jobs.append((rec, bg.submit(cpu_stages_safe, info, ego_path, out)))
            gpu_fail_streak = 0
        except InputTimeout as e:                                              # the feeder is behind (HF or network down): not a clip failure
            rec.update({"status": "WAIT-INPUT-TIMEOUT", "error": str(e)}); write_row(ledger, rec); drain(True)
            print(f"ZZPROD-INPUT-TIMEOUT {e}", flush=True); os._exit(4)            # os._exit: the prebuild thread may be inside its wait
        except Exception as e:
            rec.update({"status": "FAIL-GPU-STAGE", "error": f"{type(e).__name__}: {str(e)[:300]}"})
            traceback.print_exc()
            write_row(ledger, rec); gpu_fail_streak += 1
            if gpu_fail_streak >= 3:                                           # e.g. a poisoned CUDA context: restart, do not burn the list
                drain(True); print("ZZPROD-GPU-FAIL-STREAK", flush=True); os._exit(3)
        drain(False)
        print(f"[{k + 1}/{len(todo)}] {rec['clip_sha12']} {rec.get('frames')} frames extract {rec.get('extract_s')}s  elapsed {time.time() - T0:.0f}s", flush=True)
    drain(True)
    print(f"ZZPROD-DONE {len(todo)} clips {time.time() - T0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
