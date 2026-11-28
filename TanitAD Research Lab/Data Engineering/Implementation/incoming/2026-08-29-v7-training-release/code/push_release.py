"""Push the release to HF PRIVATE and verify by content re-download.

Standing rules honoured: private repo on the PI's account; token read in place
from G: (never echoed); verification is a re-download sha256 of MANIFEST.json
plus a full file listing compared against the local tree — never exit codes.
On any 403/quota signal: abort and report, never escalate spend.
"""
import truststore; truststore.inject_into_ssl()

import hashlib
import json
import re
import time
from pathlib import Path

import huggingface_hub as H

KEYS = "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/Keys.txt"
SRC = Path("C:/Users/Admin/tanitad-wt/_s2build/release/tanitad-v7-training-corpus")
REPO = "Sayood/tanitad-v7-training-corpus"


def read_keys(tries=8):
    for i in range(tries):
        try:
            return open(KEYS, encoding="utf-8", errors="ignore").read()
        except OSError:
            time.sleep(4 * (i + 1))
    raise RuntimeError("Keys.txt unreadable")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


tok = re.search(r"hf_[A-Za-z0-9]+", read_keys()).group(0)
api = H.HfApi(token=tok)

man = json.loads((SRC / "MANIFEST.json").read_text(encoding="utf-8"))
assert man["per_clip_cy"]["status"] == "INCLUDED", \
    f"cy table not included: {man['per_clip_cy']['status']!r} — refuse to push incomplete"
local = {k: v["sha256"] for k, v in man["files"].items()}
local["MANIFEST.json"] = sha(SRC / "MANIFEST.json")
tot = sum(v["bytes"] for v in man["files"].values())
print(f"pushing {len(local)} files, {tot/1e6:.0f} MB -> {REPO} (PRIVATE)")

api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
info = api.repo_info(REPO, repo_type="dataset")
assert info.private, "repo is NOT private — aborting before any upload"
print("repo exists and is PRIVATE")

t0 = time.time()
api.upload_folder(folder_path=str(SRC), repo_id=REPO, repo_type="dataset",
                  commit_message="v7 training corpus — PI-authorized release "
                                 "2026-08-29 (baseline blob 121a8d93, corpus "
                                 f"{man['corpus']['corpus_id_sha256_16']})")
print(f"upload done in {time.time()-t0:.0f}s")

# --- verify by content ------------------------------------------------------
remote = set(api.list_repo_files(REPO, repo_type="dataset"))
missing = [f for f in local if f not in remote]
print(f"remote files: {len(remote)} | expected present: {len(local)} | missing: {missing}")
assert not missing, "remote listing incomplete"

p = H.hf_hub_download(REPO, "MANIFEST.json", repo_type="dataset", token=tok,
                      local_dir="C:/Users/Admin/tanitad-wt/_s2build/release/_verify",
                      force_download=True)
back = sha(Path(p))
print("MANIFEST sha local :", local["MANIFEST.json"][:20])
print("MANIFEST sha remote:", back[:20])
assert back == local["MANIFEST.json"], "MANIFEST round-trip sha mismatch"
p2 = H.hf_hub_download(REPO, "labels/s2_labels_v7.jsonl.gz", repo_type="dataset",
                       token=tok,
                       local_dir="C:/Users/Admin/tanitad-wt/_s2build/release/_verify",
                       force_download=True)
back2 = sha(Path(p2))
assert back2 == man["files"]["labels/s2_labels_v7.jsonl.gz"]["sha256"], \
    "labels round-trip sha mismatch"
print("VERIFIED: listing complete, MANIFEST + labels round-trip sha256 match")
print(f"RELEASE URL: https://huggingface.co/datasets/{REPO} (private)")
