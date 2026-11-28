"""Card PREREG_V8F_LEVER3 (4c019b01), section 3: write the union train/manifest.jsonl and meta.json for the REPACKED cache.

For every clip of the pod union manifest, in its order:
  * the repack row (state_tenk/*.rows.jsonl or state_corpus/corpus.rows.jsonl) exists, and the file on disk has that row's
    bytes and md5 (every file re-read);
  * the base row is the clip's own source-part manifest row (10k: the shard's train/manifest.jsonl; corpus: the dev-box
    cache manifest), and its n_frames equals the pod union row's (a different n_frames would change the window count).
The written row = the base row with file/bytes/md5 of the repacked file, grids ["4x10"], shapes without pool8x20, and the
source identity under "repack" (content_md5 moves there: it describes the source file, pool8x20 included).
Exit 1 on any failure; nothing is written then."""
import glob, hashlib, json, os, sys

B = "F:/TanitAD-scratch/rollhist/lever3/build"
POD_UNION = "F:/TanitAD-scratch/rollhist/lever3/rec/x/dscale10k/merged/manifest.jsonl"
DEV_CORPUS = "F:/TanitAD-scratch/speedread/cache/train/manifest.jsonl"
CACHE = "C:/Users/Admin/lever3/cache_union/train"


def jl(p):
    out = {}
    for ln in open(p, encoding="utf-8"):
        if ln.strip():
            d = json.loads(ln)
            out[d["clip"]] = d
    return out


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


pod_rows = [json.loads(x) for x in open(POD_UNION, encoding="utf-8") if x.strip()]
order = [r["clip"] for r in pod_rows]
pod = {r["clip"]: r for r in pod_rows}
assert len(order) == len(set(order)) == 14512, len(order)
rep = {}
for p in sorted(glob.glob(B + "/state_tenk/*.rows.jsonl")) + [B + "/state_corpus/corpus.rows.jsonl"]:
    for c, r in jl(p).items():
        if c in rep and rep[c]["src_md5"] != r["src_md5"]:
            sys.exit(f"FAIL: {c} repacked twice from different sources")
        rep[c] = r
base = {}
for p in sorted(glob.glob(B + "/state_tenk/*.meta/train__manifest.jsonl")):
    base.update({c: dict(r, _part="10k") for c, r in jl(p).items()})
dev = jl(DEV_CORPUS)
bad, rows, nf_bad = [], [], []
for i, c in enumerate(order):
    r = rep.get(c)
    if r is None:
        bad.append(f"{c}: no repack row"); continue
    b = base.get(c) if r["part"] == "10k" else (dict(dev[c], _part="corpus") if c in dev else None)
    if b is None:
        bad.append(f"{c}: no source-part manifest row"); continue
    if int(b["n_frames"]) != int(pod[c]["n_frames"]):
        nf_bad.append((c, b["n_frames"], pod[c]["n_frames"]))
    f = f"{CACHE}/clips/{c}.pt"
    if not os.path.isfile(f) or os.path.getsize(f) != r["bytes"]:
        bad.append(f"{c}: file missing or size differs"); continue
    if md5(f) != r["md5"]:
        bad.append(f"{c}: md5 differs from its repack row"); continue
    row = {k: v for k, v in b.items() if k not in ("_part", "content_md5", "bytes", "md5", "file", "grids", "shapes")}
    row.update({"clip": c, "file": f"clips/{c}.pt", "bytes": r["bytes"], "md5": r["md5"], "grids": ["4x10"],
                "shapes": {k: v for k, v in (b.get("shapes") or {}).items() if k != "pool8x20"},
                "repack": {"part": r["part"], "src_bytes": r["src_bytes"], "src_md5": r["src_md5"],
                           "src_content_md5": b.get("content_md5"), "dropped": r["dropped"],
                           "src_shard": r.get("shard"), "src_file": r.get("src")}})
    rows.append(row)
    if (i + 1) % 2000 == 0:
        print(i + 1, "checked", flush=True)
rec = {"n_union": len(order), "n_written": len(rows), "n_bad": len(bad), "bad_first": bad[:20],
       "n_frames_mismatch": len(nf_bad), "n_frames_mismatch_first": nf_bad[:20],
       "parts": {"10k": sum(1 for r in rows if r["repack"]["part"] == "10k"), "corpus": sum(1 for r in rows if r["repack"]["part"] == "corpus")}}
if bad or nf_bad or len(rows) != 14512:
    json.dump(rec, open(B + "/MANIFEST_RECORD.json", "w"), indent=1)
    sys.exit(f"FAIL: {json.dumps(rec)[:600]}")
out = CACHE + "/manifest.jsonl"
with open(out + ".tmp", "w", encoding="utf-8", newline="\n") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
os.replace(out + ".tmp", out)
meta = {"cache": "lever3 union, repacked to pool4x10 only (card PREREG_V8F_LEVER3 4c019b01)",
        "n_clips": 14512, "grids": ["4x10"], "dropped_key": "pool8x20",
        "encoders": {"10k": "encoder_module_sha256 308f36c949979495... (HF Sayood/tanitad-datascale-10k shards)",
                     "corpus": "encoder_module_sha256 361bc287c306a8f4... (dev-box cache F:/TanitAD-scratch/speedread/cache/train)"},
        "mixed_encode": True, "repack_tool": "F:/TanitAD-scratch/rollhist/lever3/build/l3_build.py",
        "repack_tool_sha256": hashlib.sha256(open(B + "/l3_build.py", "rb").read()).hexdigest()}
json.dump(meta, open(CACHE + "/meta.json", "w", encoding="utf-8"), indent=1)
rec.update({"manifest": out, "manifest_sha256": hashlib.sha256(open(out, "rb").read()).hexdigest(),
            "manifest_md5": md5(out), "meta_sha256": hashlib.sha256(open(CACHE + "/meta.json", "rb").read()).hexdigest()})
json.dump(rec, open(B + "/MANIFEST_RECORD.json", "w"), indent=1)
print("PASS", json.dumps(rec))
