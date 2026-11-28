"""xview.py -- the cache view of the X set (the 10,000 data-scale clips alone): /root/data/cache_x_dscale/train/{clips -> symlinks into the shard folders of /root/data/cache_10k, manifest.jsonl, meta.json}.
Why a view: the trainer reads <cache-dir>/train/manifest.jsonl and <cache-dir>/train/clips/<id>.pt; /root/data/cache_10k holds the 10k clips in shard folders s*/train/clips with NO top-level train/ (union_view.py builds the union the same way).
Manifest = in/manifest_10k.jsonl (10,000 rows, file = clips/<id>.pt), meta.json = the corpus cache's (the union view copies the same file).  Pod, CPU, 10,000 symlinks and two small copies; idempotent.
DS_FAKE_ROOT redirects every path to a fake tree (dry run)."""
import glob
import json
import os
import shutil
import sys

R = os.environ.get("DS_FAKE_ROOT", "")
V = R + "/root/data/cache_x_dscale/train"
os.makedirs(V + "/clips", exist_ok=True)
src = {}
for d in sorted(glob.glob(R + "/root/data/cache_10k/s*/train/clips")):
    for f in os.listdir(d):
        assert f.endswith(".pt"), f
        c = f[:-3]
        assert c not in src, ("clip in two shards", c)
        src[c] = d + "/" + f
rows = [json.loads(l) for l in open(R + "/root/dscale10k/in/manifest_10k.jsonl") if l.strip()]
miss = [r["clip"] for r in rows if r["clip"] not in src]
extra = sorted(set(src) - {r["clip"] for r in rows})
print("shard files", len(src), "manifest rows", len(rows), "missing", len(miss), "in shards but not in the manifest", len(extra))
assert not miss, ("manifest clips without a file", len(miss), miss[:3])
for r in rows:
    assert r["file"] == "clips/%s.pt" % r["clip"], r["file"]
    p = V + "/" + r["file"]
    if os.path.lexists(p):
        os.remove(p)
    os.symlink(src[r["clip"]], p)
shutil.copyfile(R + "/root/dscale10k/in/manifest_10k.jsonl", V + "/manifest.jsonl")
shutil.copyfile(R + "/root/data/cache_pod/train/meta.json", V + "/meta.json")
print("links", len(rows), "files in clips", len(os.listdir(V + "/clips")))
if len(rows) != 10000 and not R:
    sys.exit("X set is not 10,000 clips")
print("ZZ XVIEW DONE ZZ")
