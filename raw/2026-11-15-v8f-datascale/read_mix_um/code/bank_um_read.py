"""MM 2026-11-18: bank the UM_s0 read's own derived outputs (its T1-kin dump, its T0 mechanism evals, its T0 route passes) to
private HF datasets/Sayood/tanitad-v8f-dscale10k-runs under UM_s0/A/read_mix/ in one tar; verify hub LFS sha256 == local;
write read_mix/out/BANKED_UM_READ.json. The reference arms' dumps are banked already (read / read3)."""
import glob
import hashlib
import json
import os
import re
import tarfile

os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import HfApi  # noqa: E402

R = "F:/TanitAD-scratch/datascale/read_mix"
OUT = "C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/um_read_outputs.tar"
REPO, RT, PATH = "Sayood/tanitad-v8f-dscale10k-runs", "dataset", "UM_s0/A/read_mix/um_read_outputs.tar"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


files = []
for root, _, fs in os.walk(f"{R}/t1"):
    files += [os.path.join(root, f) for f in fs]
files += glob.glob(f"{R}/out/t0mech/*DSUMs0A*")
files += [p for p in glob.glob(f"{R}/out/t0route/*") if "UM_s0" in os.path.basename(p)]
files = sorted(p.replace(os.sep, "/") for p in files)
assert files, "nothing to bank"
members = {os.path.relpath(p, R).replace(os.sep, "/"): sha(p) for p in files}
with tarfile.open(OUT, "w") as tf:
    for p in files:
        tf.add(p, arcname="um_read/" + os.path.relpath(p, R).replace(os.sep, "/"))
local = sha(OUT)
tok = re.search(r"hf_[A-Za-z0-9]+", open("E:/Projects/TanitAD/Keys.txt", encoding="utf-8", errors="replace").read()).group(0)
api = HfApi(token=tok)
ci = api.upload_file(path_or_fileobj=OUT, path_in_repo=PATH, repo_id=REPO, repo_type=RT,
                     commit_message="DSCALE mixture read: UM_s0 stage-A derived outputs (T1-kin dump, T0 mech evals, T0 route passes)")
info = api.get_paths_info(REPO, [PATH], repo_type=RT, expand=True)[0]
hub = info.lfs.sha256 if info.lfs else None
rec = {"repo": REPO, "path": PATH, "commit": getattr(ci, "oid", None), "tar_sha256": local, "hub_lfs_sha256": hub, "equal": hub == local,
       "size": os.path.getsize(OUT), "n_members": len(members), "member_sha256": members}
json.dump(rec, open(f"{R}/out/BANKED_UM_READ.json", "w", encoding="utf-8"), indent=1)
os.remove(OUT)
print(json.dumps({k: v for k, v in rec.items() if k != "member_sha256"}, indent=1))
print("ZZ BANK UMREAD", "OK" if hub == local else "MISMATCH", "ZZ")
