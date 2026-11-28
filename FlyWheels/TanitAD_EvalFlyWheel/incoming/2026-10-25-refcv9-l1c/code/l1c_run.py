"""l1c_run.py -- run a launch-tree script (runpy, __main__) with egomotion_source repointed at the fetched flat store.
    L1C_EGO_FLAT=F:/l1c/pai_root/labels/egomotion_flat python l1c_run.py <script.py> [args...]
The Alpamayo augmentation is absent for fresh clips: the loader's STAMPED opt-out is set (records file deliberately nonexistent)."""
import os
import runpy
import sys

TREE = "C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58"
for p in (TREE + "/stack", TREE + "/stack/scripts", TREE + "/taniteval"):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["TANITAD_ALPAMAYO_RECORDS"] = "F:/l1c/work/__no_alpamayo_records_for_fresh_clips__.parquet"
os.environ["TANITAD_ALPAMAYO_RECORDS_OPTIONAL"] = "1"
flat = os.environ.get("L1C_EGO_FLAT")
if not flat:
    raise SystemExit("[l1c_run] L1C_EGO_FLAT not set")
from tanitad.data import egomotion_source as ES  # noqa: E402

ES.EGO_FLAT = flat.rstrip("/\\") + "/*.parquet"
ES.EGO_GLOB = flat.rstrip("/\\") + "/__no_zips__/*.zip"
ES._index.cache_clear()
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
