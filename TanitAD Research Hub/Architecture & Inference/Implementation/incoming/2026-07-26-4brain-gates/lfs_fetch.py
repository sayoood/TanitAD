#!/usr/bin/env python3
"""Fetch git-lfs objects WITHOUT the git-lfs binary, via the LFS batch API over plain HTTPS.
Public repo (NVlabs/alpasim, Apache-2.0) -> anonymous access.
Verifies sha256 against the pointer OID before replacing the pointer file."""
import os, sys, json, hashlib, urllib.request, glob

REPO = "https://github.com/NVlabs/alpasim.git"
ROOT = "/workspace/alpa-invest/alpasim/data/trafficsim-models"
BATCH = REPO + "/info/lfs/objects/batch"


def parse_pointer(p):
    try:
        txt = open(p, "rb").read(400).decode("utf-8", "ignore")
    except Exception:
        return None
    if "git-lfs.github.com/spec" not in txt:
        return None
    oid = size = None
    for line in txt.splitlines():
        if line.startswith("oid sha256:"):
            oid = line.split("sha256:")[1].strip()
        elif line.startswith("size "):
            size = int(line.split()[1])
    return (oid, size) if oid and size else None


files = []
for p in glob.glob(ROOT + "/**/*", recursive=True):
    if os.path.isfile(p):
        ptr = parse_pointer(p)
        if ptr:
            files.append((p, ptr[0], ptr[1]))
print("pointer files found:", len(files))
for p, o, s in files:
    print("   %-70s %12d  %s" % (os.path.relpath(p, ROOT), s, o[:16]))
if not files:
    print("NOTHING TO DO"); sys.exit(0)

body = json.dumps({
    "operation": "download",
    "transfers": ["basic"],
    "objects": [{"oid": o, "size": s} for _, o, s in files],
    "hash_algo": "sha256",
}).encode()
req = urllib.request.Request(BATCH, data=body, method="POST", headers={
    "Accept": "application/vnd.git-lfs+json",
    "Content-Type": "application/vnd.git-lfs+json",
    "User-Agent": "git-lfs/3.0.0",
})
try:
    resp = json.load(urllib.request.urlopen(req, timeout=120))
except Exception as e:
    print("BATCH API FAILED:", repr(e)[:300]); sys.exit(2)

hrefs = {}
for obj in resp.get("objects", []):
    if "error" in obj:
        print("  OBJ ERROR", obj["oid"][:16], obj["error"]); continue
    act = obj.get("actions", {}).get("download")
    if act:
        hrefs[obj["oid"]] = (act["href"], act.get("header", {}))
print("download URLs resolved:", len(hrefs), "/", len(files))

ok = 0
for p, o, s in files:
    if o not in hrefs:
        print("NO URL for", p); continue
    href, hdr = hrefs[o]
    r = urllib.request.Request(href, headers=hdr)
    tmp = p + ".dl"
    h = hashlib.sha256(); n = 0
    with urllib.request.urlopen(r, timeout=600) as f, open(tmp, "wb") as w:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            w.write(b); h.update(b); n += len(b)
    if h.hexdigest() == o and n == s:
        os.replace(tmp, p)
        print("OK  %-60s %d bytes sha256-verified" % (os.path.basename(p), n))
        ok += 1
    else:
        os.remove(tmp)
        print("MISMATCH %s got %d/%s want %d/%s" % (p, n, h.hexdigest()[:16], s, o[:16]))
print("FETCHED %d/%d" % (ok, len(files)))
