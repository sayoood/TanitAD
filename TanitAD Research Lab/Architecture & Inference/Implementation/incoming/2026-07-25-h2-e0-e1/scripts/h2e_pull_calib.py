"""Pull calibration/ (intrinsics+extrinsics) for every local obstacle.offline chunk that lacks it.

~40 KB + ~60 KB per chunk => ~2 MB total. Dev box only, read-only HF pull, no pod touched.
Enlarges the E1 held-out set from 2 chunks to as many as the pull succeeds for; the held-out set
is pre-committed as "all local obstacle chunks except 0036/0170", so this cannot be a selection.
"""
import glob, os, re, sys

import truststore
truststore.inject_into_ssl()          # certifi fails behind this box's TLS proxy (MEMORY)
from huggingface_hub import hf_hub_download

DR = r"C:\Users\Admin\tanitad-data\physicalai"
REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
KEYS = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\Keys.txt"

tok = re.search(r"hf_[A-Za-z0-9]+", open(KEYS, encoding="utf-8", errors="ignore").read()).group(0)


def chunks_of(pat):
    return {os.path.basename(p).split("_")[-1].split(".")[0] for p in glob.glob(pat)}


ob = chunks_of(DR + r"\labels\obstacle.offline\*.zip")
have = chunks_of(DR + r"\calibration\camera_intrinsics\*.parquet") & \
       chunks_of(DR + r"\calibration\sensor_extrinsics\*.parquet")
todo = sorted(ob - have)
print(f"obstacle chunks {len(ob)} | already calibrated {len(ob & have)} | to pull {len(todo)}",
      flush=True)

ok, bad = [], []
for ch in todo:
    for kind in ("camera_intrinsics", "sensor_extrinsics"):
        rel = f"calibration/{kind}/{kind}.chunk_{ch}.parquet"
        dst = os.path.join(DR, "calibration", kind, f"{kind}.chunk_{ch}.parquet")
        if os.path.exists(dst):
            continue
        try:
            p = hf_hub_download(REPO, rel, repo_type="dataset", token=tok,
                                local_dir=os.path.join(DR, "_hfpull"))
            os.replace(p, dst)
            print(f"  OK  {rel}  {os.path.getsize(dst)/1024:.0f} KB", flush=True)
        except Exception as e:                                    # noqa: BLE001
            bad.append((ch, kind, type(e).__name__, str(e)[:140]))
            print(f"  FAIL {rel}: {type(e).__name__}: {str(e)[:140]}", flush=True)
            break
    else:
        ok.append(ch)

print(f"\npulled {len(ok)} chunks: {ok}")
if bad:
    print(f"failed {len(bad)}:")
    for b in bad[:10]:
        print("  ", b)
