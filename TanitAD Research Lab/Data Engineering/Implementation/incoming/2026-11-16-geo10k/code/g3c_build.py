"""GEO10K step 3c: run build_v9_labels.cmd_build (cdbff971, unedited) in parallel parts, then merge the parts.
modes:  ctlA  20 control clips, WITH the real label records + existing lead/lc inputs  (must equal the release bitwise)
        ctlB  the same 20 clips WITHOUT records / lead / lc  (= the new-clip path; shows what the records carry)
        new   the 10,000 new clips: no records, my lead pass, no lc
        wrong ctlA but one clip pointed at another clip's egomotion  (regression arm; must read RED)"""
import argparse, gzip, json, os, sys, time, shutil
import numpy as np
import torch
from multiprocessing import Pool
sys.path.insert(0, "/root/geo10k/code")
import geo_common as G
G.point_ego_lookup()
G.ES._index()
import build_v9_labels as B

REF = "/root/geo10k/ref/tanitad-refcv8-v9-labels"
MODE = sys.argv[1]
NPARTS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
W = f"{G.OUT}/build_{MODE}"
os.makedirs(W, exist_ok=True)
man_path = {"new": "new10k_v2manifest.pt"}.get(MODE, "ctl20_v2manifest.pt")
M = torch.load(f"{G.OUT}/{man_path}", weights_only=False)
EMPTY = f"{W}/empty_labels.jsonl.gz"
with gzip.open(EMPTY, "wt") as f:
    pass
ego_fmt = f"{G.FLAT}/{{}}.parquet"
if MODE == "wrong":                                    # one clip reads ANOTHER clip's egomotion: flat dir copy with one swapped symlink
    wd = f"{W}/ego_wrong"
    os.makedirs(wd, exist_ok=True)
    for c in M["clip_id"]:
        d = f"{wd}/{c}.parquet"
        if os.path.lexists(d):
            os.unlink(d)
        os.symlink(os.path.realpath(f"{G.FLAT}/{c}.parquet"), d)
    c0, c1 = M["clip_id"][0], M["clip_id"][1]
    os.unlink(f"{wd}/{c0}.parquet")
    os.symlink(os.path.realpath(f"{G.FLAT}/{c1}.parquet"), f"{wd}/{c0}.parquet")
    ego_fmt = f"{wd}/{{}}.parquet"
LABELS = {"ctlA": f"/root/geo10k/ref/tanitad-v7-training-corpus/labels/s2_labels_v8_train.jsonl.gz", "wrong": f"/root/geo10k/ref/tanitad-v7-training-corpus/labels/s2_labels_v8_train.jsonl.gz"}.get(MODE, EMPTY)
LEAD = {"ctlA": f"{REF}/inputs/lead_train.npz", "ctlC": f"{REF}/inputs/lead_train.npz", "wrong": f"{REF}/inputs/lead_train.npz", "new": f"{G.OUT}/lead_new10k.npz"}.get(MODE)
LC = {"ctlA": f"{REF}/inputs/lc_train.npz", "wrong": f"{REF}/inputs/lc_train.npz"}.get(MODE)
def filt(src, keep, dst):
    z = np.load(src, allow_pickle=True)
    m = np.isin(z["sha12"].astype(str), list(keep))
    np.savez(dst, **{k: (z[k][m] if z[k].shape and z[k].shape[0] == len(m) else z[k]) for k in z.files})
    return dst


if MODE in ("ctlA", "wrong", "ctlC"):                         # the builder's per-clip scan of the full lead / lc files is quadratic: hand it the control clips' rows only (values untouched)
    _keep = {B.sha12(c) for c in M["clip_id"]}
    LEAD = filt(LEAD, _keep, f"{W}/lead_ctl_filtered.npz")
    LC = filt(LC, _keep, f"{W}/lc_ctl_filtered.npz") if LC else None
SIDE = f"{G.OUT}/clock_new10k.jsonl" if MODE == "new" else f"{G.OUT}/clock_ctl20.jsonl"


def part(i):
    cl = M["clip_id"]
    lo, hi = i * len(cl) // NPARTS, (i + 1) * len(cl) // NPARTS
    mp = f"{W}/man_{i:02d}.pt"
    sub = {k: (v[lo:hi] if isinstance(v, list) else v) for k, v in M.items()}
    torch.save(sub, mp)
    a = argparse.Namespace(split="train", manifest=mp, sidecar=SIDE, labels=LABELS, lead=LEAD, lc=LC, ego=ego_fmt,
                           out_dir=f"{W}/part_{i:02d}", limit=0, base_commit="cdbff971", lc_labels="off")
    B.cmd_build(a)
    return i


def merge():
    parts = [np.load(f"{W}/part_{i:02d}/v9_labels_train.npz", allow_pickle=False) for i in range(NPARTS)]
    names = sorted(set().union(*[set(p.files) for p in parts]))
    out, off = {}, 0
    nrows = [int(p["clip__n_rows"].sum()) for p in parts]
    for nm in names:
        pieces = []
        for p, nr in zip(parts, nrows):
            if nm in p.files:
                pieces.append(p[nm])
            else:
                ref = next(q[nm] for q in parts if nm in q.files)
                nclip = len(p["clip__sid"])
                n = nclip if nm.startswith("clip__") else nr
                fill = np.nan if np.issubdtype(ref.dtype, np.floating) else (B.IGNORE if ref.dtype == np.int16 else 0)
                pieces.append(np.full(n, fill, dtype=ref.dtype))
        if nm == "clip__row0":
            o, fixed = 0, []
            for p, nr in zip(parts, nrows):
                fixed.append(p["clip__row0"] + o)
                o += nr
            pieces = fixed
        out[nm] = np.concatenate(pieces)
    np.savez_compressed(f"{G.OUT}/v9_labels_{MODE}.npz", **out)
    man = json.load(open(f"{W}/part_00/v9_labels_train.manifest.json", encoding="utf-8"))
    man.update({"n_clips": int(len(out["clip__sid"])), "n_rows": int(sum(nrows)), "fields": sorted(out), "npz_md5": B.file_md5(f"{G.OUT}/v9_labels_{MODE}.npz"),
                "geo10k": {"mode": MODE, "n_parts": NPARTS, "records": os.path.basename(LABELS) if LABELS != EMPTY else "none", "lead": os.path.basename(LEAD) if LEAD else None,
                           "lc": os.path.basename(LC) if LC else None, "ego_lookup": "EGO_FLAT -> flat dir of symlinks to the banked parquets (configuration, no library edit)",
                           "tree": "git archive cdbff971 stack/", "manifest_poses": man_path, "clock_sidecar": os.path.basename(SIDE)}})
    man["sources"].update({"manifest": [man_path, B.file_md5(f"{G.OUT}/{man_path}")], "sidecar": [os.path.basename(SIDE), B.file_md5(SIDE)],
                           "labels": [os.path.basename(LABELS), B.file_md5(LABELS)], "lead": [os.path.basename(LEAD), B.file_md5(LEAD)] if LEAD else None,
                           "lc": [os.path.basename(LC), B.file_md5(LC)] if LC else None})
    json.dump(man, open(f"{G.OUT}/v9_labels_{MODE}.manifest.json", "w"), default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    return len(out["clip__sid"]), int(sum(nrows))


if __name__ == "__main__":
    t0 = time.time()
    with Pool(NPARTS) as pool:
        pool.map(part, range(NPARTS), chunksize=1)
    nc, nr = merge()
    print(f"[g3c {MODE}] merged {nc} clips {nr} rows {time.time()-t0:.0f}s -> {G.OUT}/v9_labels_{MODE}.npz", flush=True)
    for i in range(NPARTS):
        shutil.rmtree(f"{W}/part_{i:02d}", ignore_errors=True)
    print(f"ZZ GEO10K-G3C-{MODE} DONE ZZ")
