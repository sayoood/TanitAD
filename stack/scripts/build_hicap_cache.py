#!/usr/bin/env python3
"""POD-SIDE builder for the HiCAP cache and vocabulary (paper sec:training-data). STATUS: written, NOT RUN (needs a GPU pod, the
corpus mp4s, torch + Transformers `main`, and the v9 release). Three subcommands:

  vocab   -- farthest-point factored vocabulary from TRAIN windows only  -> vocab.npz
  vocab-ego -- the same vocabulary on CPU, poses taken straight from the corpus egomotion tar on the v9 clock (no cache needed; the
             clock rule now_s + t_ego[0] is MEASURED in supervision.py)
  text    -- cached text-tower embeddings of node definitions            -> text.npz   (see backbone.py)
  cache   -- per-clip cache: poses + 64x160 frames from the SAME epcache episodes the REF-C line trains on, plus the Edge taps at
             a 2 Hz refresh -> <cache>/<sha12>.npz   (assertions fail loud: 160 visual tokens, tap unit/indices stored in manifest)

Episodes come from `tanitad.data.physicalai.build_episode` (cylindrical 256x640 front-wide, per-clip cy crop -- DO NOT re-implement the
decode); v9 row j == episode row j. The clip->sha12 map is sha256(clip_id)[:12]. NEVER run on a pod that is training.
"""
import argparse, hashlib, json, os, sys
import numpy as np

def sha12(cid): return hashlib.sha256(cid.encode()).hexdigest()[:12]

def cmd_vocab(a):
    from tanitad.hicap import vocab as V
    from tanitad.hicap.data import V9Rows, to_ego
    rows = V9Rows(a.v9_train)
    paths, lat, dvs, lon, v0s = [], [], [], [], []
    for s in rows.sha12:
        f = os.path.join(a.cache, s + ".npz")
        if not os.path.exists(f): continue
        c = np.load(f); P = c["poses"]; m = rows.clip(s)
        for j in range(3, len(P) - V.T_STEPS - 1, a.stride):
            r = m["rows"].start + j
            la, lo = int(rows["lat_cls_a"][r]), int(rows["lon_cls"][r])
            if la < 0 or lo < 0: continue
            p, dv = V.extract_path_profile(to_ego(P, j), float(P[j, 3])); paths.append(p); dvs.append(dv); lat.append(la); lon.append(lo); v0s.append(float(P[j, 3]))
    voc = V.build_factored_vocab(np.array(paths), lat, np.array(dvs), lon, v0s, K_P=a.k_p, K_V=a.k_v)
    np.savez(a.out, paths=voc.paths, path_cls=voc.path_cls, profiles=voc.profiles, prof_cls=voc.prof_cls, standstill=voc.standstill)
    print("vocab", voc.K_P, voc.K_V, "N=", voc.N)

def cmd_vocab_ego(a):
    """Train-split FactoredVocab without the GPU cache: episode-row poses (x, y, yaw, v) at t = grid_start + (k0 + j) dt + t_ego[0]."""
    import io, tarfile
    import pandas as pd
    from tanitad.hicap import vocab as V
    from tanitad.hicap import supervision as S
    from tanitad.hicap.data import V9Rows, to_ego
    rows = V9Rows(a.v9_train); Z = np.load(a.v9_train, allow_pickle=True)
    uu = {}
    for f in a.clip_index:
        for u in json.load(open(f))["clips"]: uu[sha12(u)] = u
    idx_path = a.out + ".ego_index.json"
    if os.path.exists(idx_path): idx = json.load(open(idx_path))
    else:
        with tarfile.open(a.ego_tar) as tf: idx = {m.name.split(".")[0]: [m.offset_data, m.size] for m in tf.getmembers() if m.isfile()}
        json.dump(idx, open(idx_path, "w"))
    fh = open(a.ego_tar, "rb")
    paths, lat, dvs, lon, v0s, n_clips = [], [], [], [], [], 0
    for i, s in enumerate(str(x) for x in Z["clip__sha12"]):
        u = uu.get(s)
        if u not in idx: continue
        o, n = idx[u]; fh.seek(o); ego = S.EgoTrack.from_columns(pd.read_parquet(io.BytesIO(fh.read(n))))
        k0, nr = int(Z["clip__k0"][i]), int(Z["clip__n_rows"][i]); m = rows.clip(s)
        tg = S.clip_clock(ego, float(Z["clip__grid_start_s"][i]), float(Z["clip__dt_s"][i]), k0 + nr + V.T_STEPS + 2)[k0:]
        if tg[-1] > ego.t[-1]: tg = tg[tg <= ego.t[-1]]
        x, y, yaw, v = ego.at(tg); P = np.stack([x, y, yaw, v], 1); n_clips += 1
        for j in range(3, min(nr, len(P) - V.T_STEPS - 1), a.stride):
            r = m["rows"].start + j
            la, lo = int(rows["lat_cls_a"][r]), int(rows["lon_cls"][r])
            if la < 0 or lo < 0: continue
            p, dv = V.extract_path_profile(to_ego(P, j), float(P[j, 3])); paths.append(p); dvs.append(dv); lat.append(la); lon.append(lo); v0s.append(float(P[j, 3]))
    voc = V.build_factored_vocab(np.array(paths), lat, np.array(dvs), lon, v0s, K_P=a.k_p, K_V=a.k_v)
    np.savez(a.out, paths=voc.paths, path_cls=voc.path_cls, profiles=voc.profiles, prof_cls=voc.prof_cls, standstill=voc.standstill)
    meta = dict(split="train", n_clips=n_clips, n_windows=len(paths), stride=a.stride, K_P=int(voc.K_P), K_V=int(voc.K_V), N=int(voc.N),
                source="corpus egomotion tar + v9 train labels", clock="t = grid_start_s + (k0 + j) dt_s + t_ego[0]")
    json.dump(meta, open(a.out + ".json", "w"), indent=1); print(meta)


def cmd_cache(a):
    import torch
    from tanitad.data import physicalai as pai
    from tanitad.hicap.backbone import FrozenTapBackbone, TapSpec
    from PIL import Image
    bb = FrozenTapBackbone(TapSpec(), device=a.device)
    os.makedirs(a.out, exist_ok=True)
    json.dump(bb.manifest(), open(os.path.join(a.out, "backbone_manifest.json"), "w"))
    clips = pai.discover_r0_clips(a.root)
    keep = set(json.load(open(a.clip_list))) if a.clip_list else None
    for cl in clips:
        sh = sha12(cl["clip_id"])
        if keep is not None and sh not in keep: continue
        f = os.path.join(a.out, sh + ".npz")
        if os.path.exists(f): continue
        ep = pai.build_episode(cl)                                       # frames [T,9,H,W] u8 stack of 3 (latest = last 3 channels)
        T = len(ep.poses); last = ep.frames[:, 6:9]                       # latest RGB frame of each stack
        small = torch.nn.functional.interpolate(last.float(), size=(64, 160), mode="area").clamp(0, 255).byte().numpy()
        tick = np.arange(0, T, 5)                                         # 2 Hz refresh
        vl, la = [], []
        for t in tick:
            img = Image.fromarray(last[t].permute(1, 2, 0).numpy())
            o = bb.extract(img); vl.append(o["vlm"].numpy()); la.append(o["last"].numpy())
        np.savez(f, sid=np.int64(0), poses=ep.poses.numpy(), frames=small, tick_k=tick, vlm=np.stack(vl), last=np.stack(la))
        print("cached", sh, T, len(tick), flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    v = sp.add_parser("vocab"); v.add_argument("--cache", required=True); v.add_argument("--v9-train", required=True); v.add_argument("--out", required=True)
    v.add_argument("--k-p", type=int, default=1024); v.add_argument("--k-v", type=int, default=256); v.add_argument("--stride", type=int, default=10)
    ve = sp.add_parser("vocab-ego"); ve.add_argument("--v9-train", required=True); ve.add_argument("--clip-index", nargs="+", required=True)
    ve.add_argument("--ego-tar", required=True); ve.add_argument("--out", required=True)
    ve.add_argument("--k-p", type=int, default=1024); ve.add_argument("--k-v", type=int, default=256); ve.add_argument("--stride", type=int, default=10)
    c = sp.add_parser("cache"); c.add_argument("--root", required=True); c.add_argument("--out", required=True); c.add_argument("--clip-list"); c.add_argument("--device", default="cuda")
    a = ap.parse_args()
    {"vocab": cmd_vocab, "vocab-ego": cmd_vocab_ego, "cache": cmd_cache}[a.cmd](a)
