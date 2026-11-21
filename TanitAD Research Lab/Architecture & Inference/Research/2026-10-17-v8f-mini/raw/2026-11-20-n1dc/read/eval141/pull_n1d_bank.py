"""pull_n1d_bank.py OUT -- fetch navrbd_read/navrbd_read_outputs.tar (private HF dataset), check sha256 143f3b2a..., extract ONLY dumps141/ (N1D eval-141 old-executor dumps + zero twins + ticks) and out/t0/ (N1D T0 passes) into OUT. Token from HF_TOKEN env, never printed. Tar is deleted by the caller after extraction (logged)."""
import hashlib, os, sys, tarfile
from huggingface_hub import hf_hub_download
OUT = sys.argv[1]
WANT = "143f3b2ac35b17d7313804a0490e98aad300da4d297bde85cc674b70edca721a"
os.environ["HF_HUB_DISABLE_XET"] = "1"
p = hf_hub_download("Sayood/tanitad-v8f-strat1-runs", "navrbd_read/navrbd_read_outputs.tar", repo_type="dataset", local_dir=OUT + "/dl")
h = hashlib.sha256()
with open(p, "rb") as f:
    for b in iter(lambda: f.read(1 << 22), b""):
        h.update(b)
print("tar", os.path.getsize(p), "sha256", h.hexdigest()[:16], "equal", h.hexdigest() == WANT, flush=True)
if h.hexdigest() != WANT:
    sys.exit(3)
n = 0
with tarfile.open(p) as t:
    for m in t.getmembers():
        parts = m.name.replace("\\", "/").split("/")[1:]
        if m.isfile() and parts[0] in ("dumps141", "out") and ".." not in parts and (parts[0] == "dumps141" or parts[1] == "t0"):
            m.name = "/".join(parts); t.extract(m, OUT + "/x"); n += 1
print("extracted", n)
