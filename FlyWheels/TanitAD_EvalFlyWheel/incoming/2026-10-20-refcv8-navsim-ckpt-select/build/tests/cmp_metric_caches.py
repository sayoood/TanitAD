"""(NAVSIM VENV helper for tests/test_navtrain_cache_score.py -- not a test.)  Compare two metric caches token by token, FIELD by field.

    python cmp_metric_caches.py <cache_a> <cache_b> <v11_tree> <maps_root>

Every ``metric_cache.pkl`` of A is unpickled with the devkit and so is B's twin (same ``<log>/<type>/<token>/`` path); every dataclass field
except ``file_path`` (the cache's own location, embedded in the pickle) must pickle to the same bytes.  Prints one ``ZZCMP {json}`` line.
"""
import dataclasses
import glob
import json
import lzma
import os
import pickle
import sys

a, b, tree, maps = sys.argv[1:5]
sys.path.insert(0, tree)
os.environ.update(NUPLAN_MAPS_ROOT=maps, NUPLAN_MAP_VERSION="nuplan-maps-v1.0", OPENSCENE_DATA_ROOT="x", NAVSIM_EXP_ROOT="x", NAVSIM_DEVKIT_ROOT=tree)


def load(p):
    with lzma.open(p, "rb") as f:
        return pickle.load(f)


pa = sorted(glob.glob(a + "/*/*/*/metric_cache.pkl"))
n_eq, diff, flds = 0, {}, []
for p in pa:
    q = os.path.join(b, *p.replace("\\", "/").split("/")[-4:])
    if not os.path.isfile(q):
        diff[os.path.basename(os.path.dirname(p))] = ["TWIN_ABSENT"]
        continue
    x, y = load(p), load(q)
    flds = [f.name for f in dataclasses.fields(type(x))]
    d = [n for n in flds if n != "file_path" and pickle.dumps(getattr(x, n)) != pickle.dumps(getattr(y, n))]
    if d:
        diff[os.path.basename(os.path.dirname(p))] = d
    else:
        n_eq += 1
print("ZZCMP " + json.dumps({"equal": n_eq, "total": len(pa), "fields": flds, "diff": diff}))
