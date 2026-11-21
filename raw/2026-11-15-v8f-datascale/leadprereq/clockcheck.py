# clockcheck.py -- lead-file `v` (ego speed at raw frame k) vs the ego speed the trainer sees at cache step j (cache `v[j]` = raw frame 2j; native track col 3 at 2j).
# CPU, one clip at a time (mmap), ~60 clips per file. 0 GPU. Residual r_s = lead.v[k = 2j + s] - cache.v[j], pooled over clips and j; shifts s in -2..2 raw frames (s != 0 are regression arms).
import json, sys, hashlib, numpy as np, torch
torch.set_num_threads(1)
D = "/root/dscale10k"; C = "/root/data/cache_union_dscale/train/"
files = {"tenk": D + "/in/labels/lead_datascale10k.npz", "corpus": "/root/strat1/inputs/data/v9/lead_train4572.npz"}
man = {m["sha12"]: m for m in map(json.loads, open(D + "/merged/manifest.jsonl"))}
nt = torch.load(D + "/merged/native_tracks_union.pt", map_location="cpu", weights_only=False)
tr = nt.get("tracks", nt) if isinstance(nt, dict) else None
out = {}
rng = np.random.RandomState(0)
for name, p in files.items():
    z = np.load(p, allow_pickle=True)
    sh, kk, vv = np.asarray(z["sha12"]).astype(str), np.asarray(z["k"], np.int64), np.asarray(z["v"], np.float64)
    uniq = np.unique(sh)
    cand = sorted(u for u in uniq if u in man)
    pick = [cand[i] for i in sorted(rng.choice(len(cand), 60, replace=False))]
    rec = {s: [] for s in (-2, -1, 0, 1, 2)}; recn = {s: [] for s in (-2, -1, 0, 1, 2)}
    kmin, kmax, nrows, ntrk = [], [], 0, 0
    for u in pick:
        m = man[u]; sel = sh == u
        k, v = kk[sel], vv[sel]
        lut = np.full(int(k.max()) + 6, np.nan); lut[k] = v
        kmin.append(int(k.min())); kmax.append(int(k.max())); nrows += len(k)
        o = torch.load(C + m["file"], map_location="cpu", weights_only=True, mmap=True)
        cv = o["v"].numpy().astype(np.float64)
        tk = tr.get(m["clip"]) if tr is not None else None
        nv = None
        if tk is not None:
            a = np.asarray(tk, np.float64); nv = a[:, 3] if a.ndim == 2 else None
        for s in rec:
            for j in range(len(cv)):
                kq = 2 * j + s
                if 0 <= kq < len(lut) and np.isfinite(lut[kq]):
                    rec[s].append(lut[kq] - cv[j])
                    if nv is not None and 2 * j < len(nv) and np.isfinite(nv[2 * j]): recn[s].append(lut[kq] - nv[2 * j])
        ntrk += tk is not None
        del o
    def st(x):
        x = np.abs(np.asarray(x)); return {"n": int(len(x)), "median_abs": float(np.median(x)), "p95_abs": float(np.percentile(x, 95)), "max_abs": float(x.max()), "mean_signed": None}
    res = {}
    for s in rec:
        r = st(rec[s]); r["mean_signed"] = float(np.mean(rec[s])); res[str(s)] = {"vs_cache_v": r}
        if recn[s]:
            r2 = st(recn[s]); r2["mean_signed"] = float(np.mean(recn[s])); res[str(s)]["vs_native_track_v"] = r2
    best = min(res, key=lambda s: res[s]["vs_cache_v"]["median_abs"])
    out[name] = {"file": p, "n_clips_in_file": int(len(uniq)), "n_clips_in_union_cache": int(len(cand)), "n_clips_checked": len(pick), "rows_checked": nrows, "k_min_over_clips": [min(kmin), max(kmin)], "k_max_over_clips": [min(kmax), max(kmax)],
                "native_track_clips": ntrk, "by_shift": res, "best_shift_by_median": best}
    print(name, "best", best, json.dumps({s: res[s]["vs_cache_v"] for s in res}), flush=True)
json.dump(out, open("/root/dscale10k/leadreach/clockcheck.json", "w"), indent=1)
print("DONE")
