"""Push fused_aug120/ to Sayood/tanitad-ph0-aug120 and verify FAR-SIDE.

Far-side verify = fresh listing + byte-level round-trip of one batch's
records, never the push log (POD_HANDOVER_2026-08-13 §4b silent-failure class).
"""
import hashlib
import json
import os
import re
import sys

import truststore

truststore.inject_into_ssl()
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

KEYS = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\Keys.txt"
TOKEN = re.search(r"hf_[A-Za-z0-9]+",
                  open(KEYS, encoding="utf-8", errors="replace").read()).group(0)
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

api = HfApi(token=TOKEN)
DS = "Sayood/tanitad-ph0-aug120"
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "fused_aug120")

local = sorted(os.listdir(SRC))
print(f"local fused files: {len(local)}")

if "--push" in sys.argv:
    api.upload_folder(folder_path=SRC, path_in_repo="fused_aug120",
                      repo_id=DS, repo_type="dataset",
                      commit_message="PH1 fusion of the aug120 batches: 201 "
                                     "records (88 corroborations, 10 "
                                     "conflicts, 201 with Alpamayo, 115 "
                                     "named sam3-absent partials)")
    print("upload_folder returned (NOT trusted; verifying far-side)")

# ---- far-side listing ----------------------------------------------------- #
info = api.dataset_info(DS, files_metadata=True)
far = {f.rfilename[len("fused_aug120/"):]: f.size for f in info.siblings
       if f.rfilename.startswith("fused_aug120/")}
print(f"far-side fused_aug120/ files: {len(far)}")
missing = [f for f in local if f not in far]
extra = [f for f in far if f not in local]
size_bad = [f for f in local if f in far
            and far[f] != os.path.getsize(os.path.join(SRC, f))]
print(f"missing far-side: {len(missing)} {missing[:5]}")
print(f"extra far-side: {len(extra)} {extra[:5]}")
print(f"size mismatches: {len(size_bad)} {size_bad[:5]}")

# ---- far-side round-trip of one batch (batch_00008's attributed clips) ---- #
srcs = json.load(open(os.path.join(SRC, "_label_sources.json")))["sources"]
b8 = sorted(c for c, s in srcs.items() if s["v2"] == "batch_00008")
sample = b8 + ["_summary.json", "_batch_accounting.json"]
n_ok = 0
for name in sample:
    fn = f"{name}.json" if not name.startswith("_") else name
    p = hf_hub_download(DS, f"fused_aug120/{fn}", repo_type="dataset",
                        token=TOKEN, force_download=True)
    a = open(p, "rb").read()
    b = open(os.path.join(SRC, fn), "rb").read()
    same = hashlib.md5(a).hexdigest() == hashlib.md5(b).hexdigest()
    n_ok += same
    if not same:
        print(f"ROUND-TRIP MISMATCH: {fn}")
    if not name.startswith("_"):
        r = json.loads(a.decode("utf-8"))
        assert r["schema_version"] == "ph1-fused-v1" and r["clip_id"] == name
        assert set(r["inference_admissible"]) == {"perception", "semantics"}
print(f"round-trip: {n_ok}/{len(sample)} byte-identical "
      f"({len(b8)} records of batch_00008 + 2 meta)")
ok = (not missing and not extra and not size_bad and n_ok == len(sample))
print("FARSIDE_VERIFY", "PASS" if ok else "FAIL")
