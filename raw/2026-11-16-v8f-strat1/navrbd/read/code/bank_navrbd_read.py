"""MM 2026-11-19: bank the NAVRBD read's derived outputs (held-out T1 dumps pulled sha256-verified from the pod, the eval-141 T1 dumps,
the T0 real / donor / zero evals) to private HF datasets/Sayood/tanitad-v8f-strat1-runs under navrbd_read/ in one tar; verify the
hub LFS sha256 == local; write read/out/BANKED_NAVRBD_READ.json; delete the local tar. The arms' checkpoints are banked by the pod banker."""
import hashlib
import json
import os
import re
import tarfile

os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import HfApi  # noqa: E402

R = "F:/TanitAD-scratch/strat1/navrbd/read"
OUT = R + "/navrbd_read_outputs.tar"
REPO, RT, PATH = "Sayood/tanitad-v8f-strat1-runs", "dataset", "navrbd_read/navrbd_read_outputs.tar"
SRC = ["dumps", "dumps141", "out/t0"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


files = []
for s in SRC:
    for root, _, fs in os.walk(f"{R}/{s}"):
        files += [os.path.join(root, f) for f in fs]
for f in ("RESULT_NAVRBD.json", "RESULT_NAVRBD.md", "NAVRBD_STRATEGIC_view.json", "NAVRBD_STRATEGIC_contrast.json", "NAVRBD_CTX_view.json", "NAVRBD_CTX_nofog.json", "navrbd_read_chain.log"):
    files.append(f"{R}/out/{f}")
files = sorted(p.replace(os.sep, "/") for p in files)
assert len(files) > 2000, len(files)
members = {os.path.relpath(p, R).replace(os.sep, "/"): sha(p) for p in files}
with tarfile.open(OUT, "w") as tf:
    for p in files:
        tf.add(p, arcname="navrbd_read/" + os.path.relpath(p, R).replace(os.sep, "/"))
local = sha(OUT)
tok = re.search(r"hf_[A-Za-z0-9]+", open("E:/Projects/TanitAD/Keys.txt", encoding="utf-8", errors="replace").read()).group(0)
api = HfApi(token=tok)
assert api.repo_info(REPO, repo_type=RT).private is True
ci = api.upload_file(path_or_fileobj=OUT, path_in_repo=PATH, repo_id=REPO, repo_type=RT,
                     commit_message="NAVRBD read: held-out + eval-141 T1 dumps, T0 evals, result files (card 881d7f3d)")
info = api.get_paths_info(REPO, [PATH], repo_type=RT, expand=True)[0]
hub = info.lfs.sha256 if info.lfs else None
rec = {"repo": REPO, "path": PATH, "commit": getattr(ci, "oid", None), "tar_sha256": local, "hub_lfs_sha256": hub, "equal": hub == local,
       "size": os.path.getsize(OUT), "n_members": len(members), "member_sha256": members}
json.dump(rec, open(f"{R}/out/BANKED_NAVRBD_READ.json", "w", encoding="utf-8"), indent=1)
sz = os.path.getsize(OUT)
os.remove(OUT)
print(json.dumps({k: v for k, v in rec.items() if k != "member_sha256"}, indent=1))
print(f"deleted {OUT} ({sz} bytes)")
print("ZZ BANK NAVRBDREAD", "OK" if hub == local else "MISMATCH", "ZZ")
