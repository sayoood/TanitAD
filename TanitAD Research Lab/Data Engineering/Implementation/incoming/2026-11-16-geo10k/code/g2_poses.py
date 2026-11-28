"""GEO10K step 2: rebuild each clip's v2 MANIFEST poses exactly as physicalai.build_episode does (signals_at on the linspace
t_query of the camera timestamps), for the 10,000 new clips and the 20 control clips; compare the controls with the
EXISTING manifest (train4572_v2manifest.pt). Also writes the flat ego dir (symlinks) the label builders read, and the
native-tracks file for the new clips (D.full_from_manifest, the builder build_native_tracks.py uses)."""
import glob, hashlib, io, json, os, sys
import numpy as np
import pandas as pd
import torch
from multiprocessing import Pool
sys.path.insert(0, "/root/geo10k/tree/stack")
from tanitad.data import physicalai as P
from tanitad.data import v8f_mini_data as D

IN = "/root/geo10k/inputs"
OUT = "/root/geo10k/out"
FLAT = "/root/geo10k/ego_flat"
os.makedirs(OUT, exist_ok=True)
os.makedirs(FLAT, exist_ok=True)
N_STACK = 3
TARGET_HZ = P.TARGET_HZ


def ego_paths():
    m = {}
    for p in glob.glob("/root/datascale/keep10k/*/ego_members/*.egomotion.parquet"):
        m[os.path.basename(p).split(".")[0]] = p
    ctl = {}
    for p in glob.glob(f"{IN}/ctl_ego/*.egomotion.parquet"):
        ctl[os.path.basename(p).split(".")[0]] = p
    return m, ctl


def poses_for(args):
    cid, ego_p = args
    ts = pd.read_parquet(f"{IN}/ts/{cid}.timestamps.parquet")
    tcol = next(c for c in ts.columns if "time" in c.lower()) if any("time" in c.lower() for c in ts.columns) else None
    if tcol is None:                                   # the timestamp is the index name (frame_index) + a column 'timestamp'
        raise RuntimeError("no time column")
    t_frames = ts[tcol].to_numpy(np.float64)
    ego = pd.read_parquet(ego_p)
    span = t_frames[-1] - t_frames[0]
    unit = 1.0
    for cand in (1e9, 1e6, 1e3):
        if span / cand > 1.0:
            unit = cand
            break
    n_target = max(int(span / unit * TARGET_HZ), N_STACK + 1)
    t_query = np.linspace(t_frames[0], t_frames[-1], n_target)
    _, poses = P.signals_at(ego, t_query, wheelbase=P.WHEELBASE)
    n = poses.shape[0]
    return cid, np.ascontiguousarray(poses[N_STACK - 1:n]), int(len(t_frames)), float(t_frames[0]), float(t_frames[-1])


def main():
    new_ego, ctl_ego = ego_paths()
    sel = [l.strip() for l in open("/root/datascale/selection/clips_10k.txt") if l.strip()]
    ctl = [l.strip() for l in open(f"{IN}/control_clips20.txt") if l.strip()]
    assert len(new_ego) == 10000 + 0 or set(sel) <= set(new_ego), (len(new_ego),)
    for c in sel:                                                     # flat ego dir: <clip>.parquet -> the banked parquet (read only)
        d = f"{FLAT}/{c}.parquet"
        if not os.path.lexists(d):
            os.symlink(new_ego[c], d)
    for c in ctl:
        d = f"{FLAT}/{c}.parquet"
        if not os.path.lexists(d):
            os.symlink(ctl_ego[c], d)
    jobs = [(c, new_ego[c]) for c in sel] + [(c, ctl_ego[c]) for c in ctl]
    with Pool(24) as pool:
        res = pool.map(poses_for, jobs, chunksize=20)
    by = {r[0]: (r[0], torch.from_numpy(r[1]), r[2], r[3], r[4]) for r in res}
    # ---- control: compare with the existing manifest
    m = torch.load(f"{IN}/train4572_v2manifest.pt", weights_only=False)
    mi = {c: i for i, c in enumerate(m["clip_id"])}
    rows, worst = [], 0.0
    for c in ctl:
        ref = m["poses"][mi[c]].double().numpy()
        mine = by[c][1].double().numpy()
        same_shape = ref.shape == mine.shape
        d = np.abs(ref - mine).max(0).tolist() if same_shape else None
        eq = bool(same_shape and np.array_equal(m["poses"][mi[c]].numpy(), by[c][1].numpy()))
        rows.append({"clip": c, "shape_ref": list(ref.shape), "shape_mine": list(mine.shape), "bitwise_equal": eq, "max_abs_diff_xyyawv": d})
        if d:
            worst = max(worst, max(d))
    ctl_rep = {"n": len(ctl), "n_bitwise_equal": sum(r["bitwise_equal"] for r in rows), "worst_max_abs_diff": worst, "rows": rows}
    json.dump(ctl_rep, open(f"{OUT}/control_poses_report.json", "w"), indent=1)
    print("[g2] control poses: bitwise equal", ctl_rep["n_bitwise_equal"], "/", len(ctl), "worst abs diff", worst, flush=True)
    # ---- regression arm: one control clip pointed at ANOTHER clip's egomotion must read RED
    c0, c1 = ctl[0], ctl[1]
    wrong = poses_for((c0, ctl_ego[c1]))[1]
    ref0 = m["poses"][mi[c0]].numpy()
    red = (wrong.shape != ref0.shape) or (not np.array_equal(wrong, ref0))
    wd = float(np.abs(wrong - ref0).max()) if wrong.shape == ref0.shape else None
    json.dump({"clip": c0, "wrong_ego_from": c1, "control_went_red": bool(red), "max_abs_diff": wd}, open(f"{OUT}/regression_poses.json", "w"))
    print("[g2] regression arm (wrong ego) control RED:", red, "max abs diff", wd, flush=True)
    # ---- manifests
    def save(clips, path):
        man = {"version": 3, "files": [f"{c}.v2ep.pt" for c in clips], "poses": [by[c][1] for c in clips],
               "clip_id": list(clips), "n_stack": [N_STACK] * len(clips), "T_out": [int(by[c][1].shape[0]) for c in clips],
               "ts_n_frames": [by[c][2] for c in clips], "ts_t0": [by[c][3] for c in clips], "ts_t1": [by[c][4] for c in clips],
               "provenance": "GEO10K g2_poses.py: physicalai.signals_at on linspace(t_frames[0], t_frames[-1]) of the camera timestamps; cdbff971 tree"}
        torch.save(man, path)
    save(sel, f"{OUT}/new10k_v2manifest.pt")
    save(ctl, f"{OUT}/ctl20_v2manifest.pt")
    tr = {}
    for c in sel + ctl:
        Pn, _ = D.full_from_manifest(by[c][1].numpy(), np.zeros((len(by[c][1]), 2)))
        tr[c] = torch.as_tensor(Pn, dtype=torch.float32)
    torch.save({"tracks": {c: tr[c] for c in sel}, "meta": {"schema": "native-tracks-v1", "n_train": len(sel), "n_eval": 0, "columns": "x,y,yaw,v",
                "train_rows": "raw frame i+2 of the v2 manifest; rows 0-1 are NaN", "eval_rows": "none", "corpus": "datascale-10k"}}, f"{OUT}/native_tracks_new10k.pt")
    # native tracks control: equal to the existing native_tracks.pt
    nt = torch.load(f"{IN}/native_tracks_existing.pt", weights_only=True)["tracks"]
    eqn = [bool(torch.equal(nt[c], tr[c])) or bool(np.array_equal(nt[c].numpy(), tr[c].numpy(), equal_nan=True)) for c in ctl]
    json.dump({"n": len(ctl), "n_equal": sum(eqn)}, open(f"{OUT}/control_native_tracks.json", "w"))
    print("[g2] control native tracks equal:", sum(eqn), "/", len(ctl), flush=True)
    lens = [int(by[c][1].shape[0]) for c in sel]
    print("[g2] new10k T: min", min(lens), "max", max(lens), "n", len(lens))
    print("ZZ GEO10K-G2 DONE ZZ")


main()
