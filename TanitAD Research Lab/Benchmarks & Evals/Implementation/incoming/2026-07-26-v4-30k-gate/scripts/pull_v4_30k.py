"""Eval-pod pull of the just-verified flagship-v4-fromscratch 30k ckpt from gated HF.

Round-trips the md5 recorded at push time on pod2, so a match here is an
end-to-end (pod2 disk -> HF -> eval pod disk) byte proof, not just an upload proof.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
DEST = "/workspace/_v4gate/flagship-v4-fromscratch-30k"
os.environ.setdefault("HF_HOME", "/workspace/_v4gate/hfcache")
from huggingface_hub import hf_hub_download  # noqa: E402

REPO = "Sayood/flagship-v4-fromscratch"
EXPECT = {
    "ckpt.pt": ("8771c1d9d3da696dcde2a745d628f6a8", 3243109310),
    "config.json": ("57223e6f58d71c02eefd6c92e0b3364a", 8233),
    "metrics.json": ("3c62a52cb44d39c773490c2f2ff19143", 464),
}

tokp = "/dev/shm/hf_tok"
with open(tokp) as f:
    tok = f.read().strip()
if not tok.startswith("hf_"):
    print("FATAL no token", flush=True)
    sys.exit(3)

os.makedirs(DEST, exist_ok=True)
res = {}
for fn, (md5_exp, size_exp) in EXPECT.items():
    p = hf_hub_download(repo_id=REPO, filename=fn, repo_type="model",
                        local_dir=DEST, token=tok)
    h = hashlib.md5()
    n = 0
    with open(p, "rb") as f:
        while True:
            b = f.read(16 << 20)
            if not b:
                break
            h.update(b)
            n += len(b)
    got = h.hexdigest()
    ok = (got == md5_exp and n == size_exp)
    res[fn] = {"path": p, "md5": got, "md5_expected": md5_exp,
               "bytes": n, "bytes_expected": size_exp, "roundtrip_md5_match": ok}
    print(f"PULLED {fn} bytes={n} md5={got} MATCH={ok}", flush=True)

os.remove(tokp)
allok = all(v["roundtrip_md5_match"] for v in res.values())
with open("/workspace/_v4gate/pull_receipt.json", "w") as f:
    json.dump({"repo": REPO, "dest": DEST, "all_roundtrip_verified": allok,
               "files": res}, f, indent=2)
print("PULL_RESULT " + ("ALL_ROUNDTRIP_VERIFIED" if allok else "MISMATCH"), flush=True)
sys.exit(0 if allok else 5)
