"""Exact-offset tar tail fill: compute the byte offset of the first missing
member from the sizes of already-extracted members (tar order = sorted names),
verify the ustar magic via a ranged 512-byte probe, then stream-extract."""
import os
import subprocess
import sys
from pathlib import Path

URL = ("https://huggingface.co/datasets/Sayood/tanitad-comma2k19-episodes/"
       "resolve/main/comma_train.tar")
D = Path("/workspace/data/comma2k19-train-b40a21eb5216")

files = sorted(D.glob("ep_*.pt"))
names = [f.name for f in files]
# contiguity check: ep_00000 .. ep_{n-1} with no gaps
expect = [f"ep_{i:05d}.pt" for i in range(len(files))]
if names != expect:
    missing = sorted(set(expect) - set(names))[:5]
    sys.exit(f"NON_CONTIGUOUS: first gaps {missing} — exact offset unsafe")

pad = lambda n: (n + 511) // 512 * 512
base = sum(512 + pad(f.stat().st_size) for f in files)

def probe(off):
    r = subprocess.run(["curl", "-sL", "-r", f"{off}-{off+511}", URL],
                       capture_output=True, timeout=60)
    b = r.stdout
    if len(b) == 512 and b[257:262] == b"ustar":
        return b[:100].rstrip(b"\0").decode(errors="replace")
    return None

for delta in (0, 512, 1024, 1536, 2048, -512):
    off = base + delta
    name = probe(off)
    print(f"offset {off} (delta {delta}): {name!r}", flush=True)
    if name and "ep_" in name:
        print(f"ALIGNED at {off} -> member {name}", flush=True)
        cmd = (f"curl -sL --retry 5 -r {off}- '{URL}' | "
               f"tar -x --skip-old-files -f - -C /workspace/data")
        rc = os.system(cmd)
        n = len(list(D.glob("ep_*.pt")))
        print(f"EXACTFILL_DONE rc={rc} files={n}", flush=True)
        sys.exit(0 if n >= 400 else 3)
sys.exit("NO_ALIGNMENT_FOUND")
