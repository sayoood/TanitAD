"""pull_n1d141.py -- fetch navrbd_read/navrbd_read_outputs.tar (private HF dataset Sayood/tanitad-v8f-strat1-runs), check its sha256 against the
bank record (143f3b2a...), and extract only the N1D eval-141 dump folders (dump_NAVRBD-N1D_s{0,1}_m10_causal_heads) into OUT. The token comes from
the HF_TOKEN environment variable and is never printed."""
import hashlib
import os
import sys
import tarfile

from huggingface_hub import hf_hub_download

OUT = sys.argv[1]
WANT = "143f3b2ac35b17d7313804a0490e98aad300da4d297bde85cc674b70edca721a"
os.makedirs(OUT, exist_ok=True)
p = hf_hub_download("Sayood/tanitad-v8f-strat1-runs", "navrbd_read/navrbd_read_outputs.tar", repo_type="dataset", local_dir=OUT + "/dl")
h = hashlib.sha256()
with open(p, "rb") as f:
    for b in iter(lambda: f.read(1 << 22), b""):
        h.update(b)
got = h.hexdigest()
print("tar", os.path.getsize(p), "bytes sha256", got[:16], "equal", got == WANT, flush=True)
if got != WANT:
    sys.exit(3)
keep = ("dump_NAVRBD-N1D_s0_m10_causal_heads", "dump_NAVRBD-N1D_s1_m10_causal_heads")
n = 0
with tarfile.open(p) as t:
    names = t.getnames()
    for m in t.getmembers():
        parts = m.name.replace("\\", "/").split("/")
        if any(k in parts for k in keep) and m.isfile() and ".." not in parts and not m.name.startswith("/"):
            i = max(parts.index(k) for k in keep if k in parts)
            m.name = "/".join(parts[i:])
            t.extract(m, OUT + "/n1d")
            n += 1
print("extracted", n, "files; folders:", sorted(os.listdir(OUT + "/n1d")) if os.path.isdir(OUT + "/n1d") else [], flush=True)
if n == 0:
    print("tar members (first 20):", names[:20])
    sys.exit(4)
