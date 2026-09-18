import hashlib, json, os, glob
import tanitad
from tanitad.data import parity

CACHE = r"D:\Projects\TanitAD-artifacts\v2ep-eval139-416x1024cyl"
MAPS = r"D:\Projects\TanitAD-artifacts\sam3-maps-eval"


def sha12(s):
    return hashlib.sha256(str(s).encode()).hexdigest()[:12]


ids = sorted({os.path.basename(p)[: -len(".v2ep.pt")]
              for p in glob.glob(os.path.join(CACHE, "*.v2ep.pt"))})
maps = sorted({os.path.basename(p).split(".")[0]
               for p in glob.glob(os.path.join(MAPS, "*.npz"))})
d = {sha12(c): c for c in ids}
hit = sum(1 for m in maps if m in d)
print("tanitad:", tanitad.__file__)
print("cache", len(ids), "maps", len(maps), "sha12-join hits", hit)
if hit == 0:
    print("sha12(first cache id) =", sha12(ids[0]), " sample map =", maps[0])
    raise SystemExit(0)
have = {d[m] for m in maps if m in d}
in_train = set(parity.clips_in_parity_train(ids))
clean = [c for c in ids if c not in in_train]
clean_map = [c for c in clean if c in have]
mapless = [c for c in ids if c not in have]
out = {
    "n_cache": len(ids), "n_maps": len(maps), "n_with_map": len(have),
    "n_in_parity_train": len(in_train), "n_clean": len(clean),
    "n_clean_with_map": len(clean_map), "n_mapless": len(mapless),
    "mapless_that_are_also_parity": len([c for c in mapless if c in in_train]),
    "mapless_sha12": sorted(sha12(c) for c in mapless),
    "parity_sha12": sorted(sha12(c) for c in in_train),
}
print(json.dumps(out, indent=1))
json.dump(out, open("clean_split.json", "w"), indent=1)
