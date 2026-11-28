"""Pack the per-episode T1 dump dirs of the DAG-HACC rollouts into one npz per run (the layout the Read-A readers consume), recording the sha256 of the packed file
AND of every per-episode file (pack_dumps.py of PREREG_V8F_DAG_REREAD, arm swapped; rr_lib's own loader/packer)."""
import glob
import hashlib
import json
import os
import sys

os.environ.setdefault("RR_READERS", "E:/Projects/TanitAD-artifacts/dagreread/reader_src")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/dagreread/code")
import rr_lib as R  # noqa: E402

PK = "E:/Projects/TanitAD-artifacts/daghacc"
os.makedirs(f"{PK}/packed", exist_ok=True)
SRC = {}
for s in (0, 1):
    SRC[f"dumps_DAGHACC_s{s}_m10_causal"] = f"{PK}/t1/dump_DAGHACC_s{s}_m10_causal_heads"
    SRC[f"dumps_DAGHACC_s{s}_m10_logged"] = f"{PK}/t1/dump_DAGHACC_s{s}_m10_logged_heads"
    SRC[f"dumps_DAGHACCSTRICT_s{s}_m10_frozen_causal"] = f"{PK}/t1/dump_DAGHACCSTRICT_s{s}_m10_frozen_causal_heads"
man_p = f"{PK}/packed/PACK_MANIFEST.json"
man = json.load(open(man_p)) if os.path.exists(man_p) else {}
for name, d in SRC.items():
    if not os.path.isdir(d) or len(glob.glob(d + "/ep*.npz")) != 141:
        print("skip (not ready)", name)
        continue
    dst = f"{PK}/packed/{name}.npz"
    per_ep = {os.path.basename(f): R.sha256_file(f) for f in sorted(glob.glob(d + "/ep*.npz"))}
    h = hashlib.sha256("".join(f"{k} {v}\n" for k, v in per_ep.items()).encode()).hexdigest()
    sha = R.pack(d, dst)
    man[name] = {"src_dir": d, "n_files": len(per_ep), "per_episode_files_digest_sha256": h, "packed_sha256": sha, "packed_bytes": os.path.getsize(dst)}
    print("packed", name, sha[:16], os.path.getsize(dst), flush=True)
json.dump(man, open(man_p, "w"), indent=1)
print("ZZPACK-DONE", len(man))
