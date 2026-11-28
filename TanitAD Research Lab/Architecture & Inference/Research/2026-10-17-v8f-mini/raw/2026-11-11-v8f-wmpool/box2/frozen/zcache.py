"""ZCACHE: per-frame cache of the frozen adapter's mean token of v8F-mini (M2b s0 / s1), BEFORE WMAux.proj.

zmean[t] = (model.adapter(feats[t].float()) + model.pos).mean(token axis)   -- exactly the tensor ``WMAux.z1`` projects
(v8f_mini.py WMAux.z1: ``self.proj(adapt.mean(2))``, with ``adapt = adapter(feats.float()) + pos[None, None]`` from ``V8FMini._encode``).
Inference only. Modes:
  build  : write <out>/M2b_s{k}/{split}/<clip>.npz  (zmean f32 [T, D], ok bool [T], t_s f32 [T]); per-clip tmp + rename; resumable.
  check  : checks 1 and 2 on the clips of --clips-file (see CHECKS below), writes <out>/CHECK_12.json.
  manifest : MANIFEST.json (counts, D, T distribution, adapter sha256, code commit, sha256 per file).
Code path: ``ClipStore.get_kin`` (tracks, ok) and the ``pool4x10`` grid as ``ClipStore.get`` reads it (float8 -> float16), the model built as wmact_lib.load_model does.
"""
from __future__ import annotations

import argparse, glob, hashlib, json, os, sys, time
import numpy as np
import torch

REPO = os.environ.get("ZC_REPO", "D:/tanitad/repo")
for p in (REPO + "/stack", REPO + "/taniteval"):
    if p not in sys.path:
        sys.path.insert(0, p)
from tanitad.data import v8f_mini_data as D   # noqa: E402
from tanitad.models import v8f_mini as M      # noqa: E402

CH = int(os.environ.get("ZC_CHUNK", "16"))    # frames per adapter call (the live path uses F = 16 per window)
GRID = "pool4x10"


def sha_file(p, blk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(blk), b""):
            h.update(b)
    return h.hexdigest()


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    r = model.load_state_dict(ck["model"], strict=False)
    assert not r.missing_keys and set(r.unexpected_keys) <= {"hacc_w"}, (r.missing_keys, r.unexpected_keys)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, ck


def adapter_sha(ck):
    h = hashlib.sha256()
    names = sorted(k for k in ck["model"] if k.startswith("adapter.") or k == "pos")
    for k in names:
        t = ck["model"][k].detach().cpu().contiguous()
        h.update(k.encode() + str(tuple(t.shape)).encode() + str(t.dtype).encode() + t.numpy().tobytes())
    return h.hexdigest(), names


@torch.no_grad()
def zmean_of(model, feats16: torch.Tensor) -> torch.Tensor:
    """feats16 [T, n_tok, d_feat] (float16) -> [T, d] float32; frames go through ``model.adapter`` in chunks of CH, as windows of F = CH."""
    out = []
    for s in range(0, feats16.shape[0], CH):
        f = feats16[s:s + CH].float()[None]                                   # [1, F, n_tok, d_feat]
        adapt = model.adapter(f) + model.pos[None, None]                       # V8FMini._encode
        out.append(adapt.mean(2)[0])                                           # WMAux.z1's argument
    return torch.cat(out).float()


def clip_inputs(root, split, clip, tracks):
    """(feats16 [T,40,1024], ok bool [T], t_s f32 [T]) for one clip, memory-mapped (only the 4x10 grid is read)."""
    path = f"{root}/{split}/clips/{clip}.pt"
    store = D.ClipStore(root, split, lru=1, grid=GRID, kin_mode="yaw", tracks=tracks)
    kin = store.get_kin(clip)                                                  # track check + yaw_kin_from_track, as ClipStore.get
    kin3 = kin["kin3"]
    T = len(kin["pose"])
    p10 = np.full((T, 8), np.nan, np.float32)
    p10[1:] = D._fk().kin_features(kin3[1:]).astype(np.float32)                # the line ClipStore.get runs
    idx = np.arange(T)
    ok = np.isfinite(p10).all(1) | (idx < 1)                                   # V8FMiniWindows.item's frame_ok, for the frame itself
    try:
        o = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (RuntimeError, TypeError):
        o = torch.load(path, map_location="cpu", weights_only=True)
    feats = o[GRID].to(torch.float16).clone()
    del o
    assert feats.shape[0] == T, (clip, feats.shape, T)
    return feats, ok, (idx * 0.2).astype(np.float32)


def live_zmean(model, root, split, clip, tracks, ts):
    """The LIVE path: ClipStore.get (full load) -> 16-frame windows (index clamp as V8FMiniWindows.item) -> V8FMini._encode -> mean(2).
    Returns {t: [16, d]} for the window ends ``ts`` and the frame indices used."""
    store = D.ClipStore(root, split, lru=1, grid=GRID, kin_mode="yaw", tracks=tracks)
    d = store.get(clip)
    T = len(d["pose"])
    res = {}
    with torch.no_grad():
        for t in ts:
            if t >= T:
                continue
            idx = np.arange(t - 16 + 1, t + 1)
            idc = np.clip(idx, 0, T - 1)
            f = torch.from_numpy(d["feats"][idc])[None]                        # [1, 16, 40, 1024] float16, as the dataset batch
            res[t] = (idc, model._encode({"feats": f}).mean(2)[0])
    return res


def list_clips(root, split):
    return sorted(os.path.basename(p)[:-3] for p in glob.glob(f"{root}/{split}/clips/*.pt"))


def cmd_build(a):
    tracks = D.load_native_tracks(a.tracks)
    models = {}
    for kv in a.ckpt:
        k, p = kv.split("=", 1)
        models[int(k)] = load_model(p)[0]
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    for split in a.splits:
        clips = list_clips(a.root, split)
        if a.clips_file:
            keep = set(json.load(open(a.clips_file))[split]) if split in json.load(open(a.clips_file)) else set()
            clips = [c for c in clips if c in keep]
        if a.limit:
            clips = clips[:a.limit]
        for k in models:
            os.makedirs(f"{a.out}/M2b_s{k}/{split}", exist_ok=True)
        t0 = time.time(); n_new = 0
        for i, clip in enumerate(clips):
            todo = [k for k in models if not os.path.exists(f"{a.out}/M2b_s{k}/{split}/{clip}.npz")]
            if not todo:
                continue
            feats, ok, t_s = clip_inputs(a.root, split, clip, tracks)
            for k in todo:
                z = zmean_of(models[k], feats).numpy().astype(np.float32)
                assert np.isfinite(z).all(), (clip, k)
                dst = f"{a.out}/M2b_s{k}/{split}/{clip}.npz"
                tmp = dst + ".tmp"
                with open(tmp, "wb") as fh:
                    np.savez(fh, zmean=z, ok=ok, t_s=t_s)
                os.replace(tmp, dst)
            n_new += 1
            if n_new % 25 == 0 or i == len(clips) - 1:
                print(f"{time.strftime('%H:%M:%S')} {split} {i + 1}/{len(clips)} new={n_new} {(time.time() - t0) / n_new:.2f} s/clip", flush=True)
        print(f"ZZSPLIT-DONE {split} clips={len(clips)} new={n_new} sec={time.time() - t0:.1f}", flush=True)
    print("ZZBUILD-OK", flush=True)


def cmd_check(a):
    tracks = D.load_native_tracks(a.tracks)
    sel = json.load(open(a.clips_file))
    mods = {int(kv.split("=")[0]): load_model(kv.split("=", 1)[1])[0] for kv in a.ckpt}
    res = {"tol": 1e-6, "clips": {}, "ts": [0, 3, 15, 40, 100]}
    ts = res["ts"]
    for split, clips in sel.items():
        for clip in clips:
            feats, ok, t_s = clip_inputs(a.root, split, clip, tracks)
            T = feats.shape[0]
            for k, m in mods.items():
                z = zmean_of(m, feats)                                           # cached value (the build's expression)
                for tag, mm in (("own", m), ("wrongseed", mods[1 - k])):
                    live = live_zmean(mm, a.root, split, clip, tracks, ts)
                    md = 0.0
                    for t, (idc, lz) in live.items():
                        md = max(md, float((z[torch.as_tensor(idc)] - lz).abs().max()))
                    res["clips"].setdefault(f"{split}/{clip}", {})[f"s{k}_{tag}"] = {"max_abs_diff": md, "T": T}
    own = [v[f] for v in res["clips"].values() for f in v if f.endswith("_own")]
    wrong = [v[f] for v in res["clips"].values() for f in v if f.endswith("_wrongseed")]
    res["check1_max_abs_diff"] = max(x["max_abs_diff"] for x in own)
    res["check1_pass"] = res["check1_max_abs_diff"] <= res["tol"]
    res["check2_min_abs_diff_wrongseed"] = min(x["max_abs_diff"] for x in wrong)
    res["check2_wrongseed_fails_check1"] = res["check2_min_abs_diff_wrongseed"] > res["tol"]
    json.dump(res, open(a.out_json, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "clips"}), flush=True)
    print("ZZCHECK-OK" if res["check1_pass"] and res["check2_wrongseed_fails_check1"] else "ZZCHECK-FAIL", flush=True)


def cmd_manifest(a):
    files, Ts, dims = {}, [], set()
    ckinfo = {}
    for kv in a.ckpt:
        k, p = kv.split("=", 1)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        s, names = adapter_sha(ck)
        ckinfo[f"M2b_s{k}"] = {"ckpt_file_sha256": sha_file(p), "adapter_state_sha256": s, "adapter_keys": names,
                               "adapter_state_note": "sha256 over name+shape+dtype+bytes of the sorted keys adapter.* and pos in ckpt['model']"}
    for sd in sorted(glob.glob(f"{a.out}/M2b_s*")):
        tag = os.path.basename(sd)
        for split in ("train", "eval"):
            fs = sorted(glob.glob(f"{sd}/{split}/*.npz"))
            for f in fs:
                z = np.load(f)
                dims.add(int(z["zmean"].shape[1])); Ts.append(int(z["zmean"].shape[0]))
                files[f"{tag}/{split}/{os.path.basename(f)}"] = sha_file(f)
    T = np.array(Ts)
    man = {"clip_count_files": len(files), "per_seed_split": {t: {s: len(glob.glob(f"{a.out}/{t}/{s}/*.npz")) for s in ("train", "eval")}
                                                         for t in sorted(os.path.basename(x) for x in glob.glob(f"{a.out}/M2b_s*"))},
           "D": sorted(dims), "T_min": int(T.min()), "T_max": int(T.max()), "T_mean": float(T.mean()),
           "T_hist": {str(int(v)): int(c) for v, c in zip(*np.unique(T, return_counts=True))},
           "adapter": ckinfo, "code_commit": a.commit, "code_files_sha256": (json.load(open(a.code_sha)) if os.path.exists(a.code_sha) else json.loads(a.code_sha)),
           "dtype": {"zmean": "float32", "ok": "bool", "t_s": "float32"}, "grid_dt_s": 0.2, "source_cache": a.root,
           "files_sha256": files}
    json.dump(man, open(f"{a.out}/MANIFEST.json", "w"), indent=1)
    print("ZZMANIFEST-OK", len(files), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "check", "manifest"])
    ap.add_argument("--root", default="D:/tanitad/cache_full")
    ap.add_argument("--tracks", default="D:/tanitad/cache_full/native_tracks.pt")
    ap.add_argument("--ckpt", nargs="+", required=True, help="k=path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["eval", "train"])
    ap.add_argument("--clips-file")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-json", default="CHECK_12.json")
    ap.add_argument("--commit", default="26fb0bc1b86bd290c9913ecee0b3676ba7684f05")
    ap.add_argument("--code-sha", default="{}")
    a = ap.parse_args()
    {"build": cmd_build, "check": cmd_check, "manifest": cmd_manifest}[a.mode](a)
