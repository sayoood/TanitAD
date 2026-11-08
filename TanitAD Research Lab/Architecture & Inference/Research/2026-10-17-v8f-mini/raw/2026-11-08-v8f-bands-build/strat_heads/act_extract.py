"""Per-clip sidecar for the strategic-head trainer: act [Te, 2] (accel m/s^2, yaw rate rad/s), v [Te], nav [T, Dn], all on the 0.2 s cache grid.

act and v come from the NATIVE 10 Hz pose track (`ego_channels_native`, cache step j = raw frame 2 j), so they run to the end of the egomotion log (median about 140 s), well past the 20 s of video: the
logged macro-actions for the 6-30 s rollouts exist. nav is the label join's row for each cache step (`join_train.npz` / `join_eval141.npz`, key `nav`).
`--check-cache` compares the first T accel values with the cache file's own `a` channel (the one the world model was trained on) and writes ACT_CHECK.json (UNVERIFIED until it is run on box 2).
CPU only. Resumable (a clip whose sidecar exists is skipped). Reads no camera frame.
  python act_extract.py --code-root D:\tanitad\repo --tracks D:\tanitad\cache_full\native_tracks.pt --join D:\tanitad\runs\v8f_mini_full\join_train.npz --zcache D:\tanitad\zcache\M2b_s0\train --out D:\tanitad\strat_side\train [--cache-dir D:\tanitad\cache_full\train\clips --check-cache]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-root", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--join", required=True)
    ap.add_argument("--zcache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default=None, help="dir of <clip>.pt cache files, for --check-cache")
    ap.add_argument("--check-cache", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    for p in (a.code_root + "/stack", a.code_root + "/taniteval"):
        sys.path.insert(0, p)
    from tanitad.data import v8f_mini_data as D
    tracks = D.load_native_tracks(a.tracks)
    J = D.load_join(a.join)
    row = {str(c): i for i, c in enumerate(J["clip"])}
    os.makedirs(a.out, exist_ok=True)
    diffs, n_done = {}, 0
    for f in sorted(glob.glob(os.path.join(a.zcache, "*.npz"))):
        cid = os.path.basename(f)[:-4]
        out = os.path.join(a.out, cid + ".npz")
        if cid not in tracks or cid not in row:
            print(f"SKIP {cid}: no track or no join row", flush=True)
            continue
        T = int(np.load(f)["zmean"].shape[0])
        ch = D.ego_channels_native(tracks[cid])[0::2]
        act = np.nan_to_num(np.stack([ch[:, 1], ch[:, 2]], 1)).astype(np.float32)
        v = np.nan_to_num(ch[:, 0]).astype(np.float32)
        ci = row[cid]
        nav = np.nan_to_num(np.asarray(J["nav"][int(J["off"][ci]):int(J["off"][ci]) + int(J["n"][ci])], np.float32))
        if len(nav) < T:
            nav = np.concatenate([nav, np.repeat(nav[-1:], T - len(nav), 0)], 0)
        if a.check_cache and a.cache_dir:
            import torch
            o = torch.load(os.path.join(a.cache_dir, cid + ".pt"), map_location="cpu", weights_only=True, mmap=True)
            ca = np.nan_to_num(o["a"].numpy())
            m = min(len(ca), T, len(act))
            diffs[cid] = float(np.abs(ca[:m] - act[:m, 0]).max())
        tmp = out + ".tmp.npz"
        np.savez(tmp, act=act, v=v, nav=nav)
        os.replace(tmp, out)
        n_done += 1
        if a.limit and n_done >= a.limit:
            break
    if diffs:
        json.dump({"max_abs_diff_accel_track_vs_cache": max(diffs.values()), "per_clip": diffs}, open(os.path.join(a.out, "ACT_CHECK.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    print(f"wrote {n_done} sidecars to {a.out}", flush=True)


if __name__ == "__main__":
    main()
