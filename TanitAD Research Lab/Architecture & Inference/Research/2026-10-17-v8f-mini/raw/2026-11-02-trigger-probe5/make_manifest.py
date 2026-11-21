"""make_manifest.py -- MANIFEST.txt: sha256 + size of every deliverable under the package (code, cards, raw/ results); big arrays on F: are listed by path + size + sha256 of FEAT_SHA256.txt.  python -I make_manifest.py"""
import hashlib
import os
import time

ROOT = "E:/Projects/TanitAD-artifacts/dagland/trigger_probe5"
SKIP_DIR = {"__pycache__", ".pytest_cache"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


lines = [f"# MANIFEST -- trigger_probe5 (sha256 of every deliverable; generated {time.strftime('%Y-%m-%dT%H:%M:%S%z')})", f"# root: {ROOT}/"]
for d, ds, fs in os.walk(ROOT):
    ds[:] = sorted(x for x in ds if x not in SKIP_DIR)
    for f in sorted(fs):
        if f == "MANIFEST.txt" or f.endswith(".pyc"):
            continue
        p = os.path.join(d, f).replace("\\", "/")
        lines.append(f"{sha(p)}  {p[len(ROOT) + 1:]}  ({os.path.getsize(p)} bytes)")
SC = "F:/TanitAD-scratch/trigger_probe5_scratch"
lines.append(f"# big arrays (F:, single copies unless noted): {SC}/shards/*.zip (staged JPEG crops; sha256 in {SC}/local_sha.txt), {SC}/out/feat_shard_*.npy (box-2 features; sha256 in raw/FEAT_SHA256.txt; box 2 holds the second copy under D:/tanitad/jobs/probe5/out), {SC}/FEAT_all.npy, {SC}/pcacomp_*_all.npz")
open(f"{ROOT}/MANIFEST.txt", "w", newline="\n").write("\n".join(lines) + "\n")
print(len(lines) - 3, "files")
