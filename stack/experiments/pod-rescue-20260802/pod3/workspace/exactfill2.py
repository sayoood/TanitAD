"""Surgical tar fill v2: header-chaining via HTTP ranges.

Resolves exact member offsets by walking tar headers (512-byte ranged reads),
then fetches ONLY the missing members' byte ranges and pipes each slice into
tar (every slice starts at a valid header, so no alignment scanning needed).
Handles the two gap sets: the mid-hole (ep_00192..196) and the tail (>=359).
"""
import subprocess
import sys
from pathlib import Path

URL = ("https://huggingface.co/datasets/Sayood/tanitad-comma2k19-episodes/"
       "resolve/main/comma_train.tar")
D = Path("/workspace/data/comma2k19-train-b40a21eb5216")
TOTAL = 400
pad = lambda n: (n + 511) // 512 * 512


def rng(a, b):
    r = subprocess.run(["curl", "-sL", "--retry", "5", "-r", f"{a}-{b}", URL],
                       capture_output=True, timeout=120)
    return r.stdout


def hdr(off):
    b = rng(off, off + 511)
    if len(b) != 512 or b[257:262] != b"ustar":
        return None, None
    name = b[:100].rstrip(b"\0").decode(errors="replace")
    size = int(b[124:136].rstrip(b"\0 ").decode() or "0", 8)
    return name, size


def extract_range(a, b):
    curl = subprocess.Popen(["curl", "-sL", "--retry", "5", "-r", f"{a}-{b}",
                             URL], stdout=subprocess.PIPE)
    tar = subprocess.run(["tar", "-x", "--skip-old-files", "-f", "-",
                          "-C", "/workspace/data"], stdin=curl.stdout)
    curl.wait()
    return tar.returncode


have = {f.name: f.stat().st_size for f in D.glob("ep_*.pt")}
prefix = []                                  # contiguous known-size prefix
for i in range(TOTAL):
    n = f"ep_{i:05d}.pt"
    if n not in have:
        first_missing = i
        break
    prefix.append(have[n])
else:
    sys.exit(print("ALREADY_COMPLETE"))

base = sum(512 + pad(s) for s in prefix)
start = None
for delta in (0, 512, 1024, -512, 2048):
    name, size = hdr(base + delta)
    if name and name.split("/")[-1] == f"ep_{first_missing:05d}.pt":
        start = base + delta
        print(f"aligned: {name} at {start} (delta {delta}, size {size})",
              flush=True)
        break
if start is None:
    sys.exit("NO_ALIGNMENT")

# Walk headers from first_missing; fetch missing members; skip known ones
# arithmetically once their sizes are on disk.
off = start
idx = first_missing
while idx < TOTAL:
    n = f"ep_{idx:05d}.pt"
    if n in have:
        off += 512 + pad(have[n])
        idx += 1
        continue
    name, size = hdr(off)
    if name is None:
        sys.exit(f"HEADER_LOST at {off} (idx {idx})")
    short = name.split("/")[-1]
    if short != n:
        sys.exit(f"ORDER_MISMATCH at {off}: expected {n}, saw {short}")
    end = off + 512 + pad(size) - 1
    print(f"fetch {n}: bytes {off}-{end} ({size / 1e6:.0f} MB)", flush=True)
    rc = extract_range(off, end)
    if rc != 0 or not (D / n).exists():
        sys.exit(f"EXTRACT_FAIL {n} rc={rc}")
    have[n] = (D / n).stat().st_size
    off = end + 1
    idx += 1

n = len(list(D.glob("ep_*.pt")))
print(f"EXACTFILL2_DONE files={n}", flush=True)
sys.exit(0 if n >= TOTAL else 4)
