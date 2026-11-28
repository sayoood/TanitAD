#!/usr/bin/env python3
"""b3_setup_slices.py -- build the control-b3 slice caches (card PREREG_V8F_LEVER3_V2 (sha256 707c5225..) section 3 step 4).
Two slice caches of the same 20 clips (10 corpus, dev-box encode; 10 from s001_smoke20, pod encode):
  <B3>/orig/train/{clips/*.pt, manifest.jsonl}    original format (pool4x10 AND pool8x20), copies of the sources
  <B3>/repack/train/{clips/*.pt, manifest.jsonl}  repacked (pool8x20 dropped, every other key byte-equal; repack_bytes of l3_build.py, copied here with its b1 control)
  <B3>/labels/native_tracks_slice.pt              the 20 clips' native tracks (corpus file + 10k file; the file the tracks flag reads)
Copies, never links. Refuses to overwrite an existing slice dir (delete by literal path first).
usage: b3_setup_slices.py [--b3 E:/TanitAD-scratch/lever3/b3]"""
import argparse, hashlib, io, json, os, sys, tarfile
import torch

DROP = "pool8x20"
TAR = "{b3}/dl/cache/s001_smoke20.cache.tar"
CORPUS = "F:/TanitAD-scratch/speedread/cache/train"
UNION_MAN = "C:/Users/Admin/lever3/merged/manifest.jsonl"
TRK_CORPUS = "E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/native_tracks.pt"
TRK_10K = "{b3}/dl/labels/native_tracks_datascale10k.pt"


def tbytes(t):
    t = t.detach().contiguous().reshape(-1)
    return t.view(torch.uint8) if t.numel() else t


def same(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype
                and tuple(a.shape) == tuple(b.shape) and bool(torch.equal(tbytes(a), tbytes(b))))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def repack_bytes(raw, dst):
    src = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    kept = {k: v for k, v in src.items() if k != DROP}
    tmp = dst + ".tmp"
    torch.save(kept, tmp)
    back = torch.load(tmp, map_location="cpu", weights_only=True)
    if set(back) != set(kept) or DROP in back or not all(same(back[k], src[k]) for k in kept):
        raise RuntimeError(f"{dst}: b1 control failed")
    os.replace(tmp, dst)
    out = open(dst, "rb").read()
    return {"bytes": len(out), "md5": hashlib.md5(out).hexdigest(), "had_pool8x20": DROP in src}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b3", default="E:/TanitAD-scratch/lever3/b3")
    ap.add_argument("--n-corpus", type=int, default=10)
    ap.add_argument("--n-tenk", type=int, default=10)
    a = ap.parse_args()
    B3 = a.b3
    for d in ("orig", "repack"):
        if os.path.exists(f"{B3}/{d}"):
            sys.exit(f"REFUSED: {B3}/{d} exists (delete it by literal path first)")
    for d in ("orig", "repack"):
        os.makedirs(f"{B3}/{d}/train/clips")
    os.makedirs(f"{B3}/labels", exist_ok=True)
    union = {}
    for ln in open(UNION_MAN, encoding="utf-8"):
        if ln.strip():
            r = json.loads(ln)
            union[r["clip"]] = r
    rows = []   # (clip, part, manifest_row_of_source, raw_bytes)
    # corpus part: the first N rows of the dev-box manifest that are in the union (the 60 holdout clips are not)
    n = 0
    for ln in open(f"{CORPUS}/manifest.jsonl", encoding="utf-8"):
        if not ln.strip():
            continue
        r = json.loads(ln)
        if r["clip"] in union and r["clip"] not in {x[0] for x in rows}:
            rows.append((r["clip"], "corpus", r, open(f"{CORPUS}/clips/{r['clip']}.pt", "rb").read()))
            n += 1
            if n == a.n_corpus:
                break
    # 10k part: first N rows of the s001 shard's manifest
    tf = tarfile.open(TAR.format(b3=B3))
    man = [json.loads(x) for x in tf.extractfile("train/manifest.jsonl").read().decode().splitlines() if x.strip()]
    for r in man[:a.n_tenk]:
        rows.append((r["clip"], "10k", r, tf.extractfile(f"train/clips/{r['clip']}.pt").read()))
    assert len(rows) == a.n_corpus + a.n_tenk and len({x[0] for x in rows}) == len(rows), "clip count"
    mo, mr, report = [], [], []
    for clip, part, r, raw in rows:
        open(f"{B3}/orig/train/clips/{clip}.pt", "wb").write(raw)
        mo.append(json.dumps(r))
        rp = repack_bytes(raw, f"{B3}/repack/train/clips/{clip}.pt")
        r2 = dict(r)
        r2.update({"bytes": rp["bytes"], "md5": rp["md5"]})
        for k in ("content_md5",):
            r2.pop(k, None)
        r2["grids"] = ["4x10"]
        mr.append(json.dumps(r2))
        report.append({"clip": clip, "part": part, "src_bytes": len(raw), "src_md5": hashlib.md5(raw).hexdigest(), **rp})
    open(f"{B3}/orig/train/manifest.jsonl", "w", encoding="utf-8").write("\n".join(mo) + "\n")
    open(f"{B3}/repack/train/manifest.jsonl", "w", encoding="utf-8").write("\n".join(mr) + "\n")
    # tracks slice
    t1 = torch.load(TRK_CORPUS, map_location="cpu", weights_only=True)["tracks"]
    t2 = torch.load(TRK_10K.format(b3=B3), map_location="cpu", weights_only=True)["tracks"]
    sel = {}
    for clip, part, _, _ in rows:
        src = t1 if part == "corpus" else t2
        if clip not in src:
            sys.exit(f"no native track for {part} clip {clip}")
        sel[clip] = src[clip]
    torch.save({"tracks": sel}, f"{B3}/labels/native_tracks_slice.pt")
    # speed slice
    sp = {}
    for ln in open("F:/TanitAD-scratch/rollhist/lever3/rec/x/dscale10k/merged/speed_union.jsonl", encoding="utf-8"):
        if ln.strip():
            d = json.loads(ln)
            sp[d["clip_id"]] = ln
    open(f"{B3}/labels/speed_slice.jsonl", "w", encoding="utf-8").write("".join(sp[c[0]] for c in rows if c[0] in sp))
    json.dump({"clips": report, "n_speed_rows": sum(1 for c in rows if c[0] in sp)}, open(f"{B3}/SLICES.json", "w"), indent=1)
    print("OK", len(rows), "clips;", sum(1 for x in report if x["had_pool8x20"]), "had pool8x20; speed rows", sum(1 for c in rows if c[0] in sp))


main()
