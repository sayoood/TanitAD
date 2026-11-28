# verify_cache.py : runs ON THE POD (staging of card PREREG_V8F_PROOF_ADD46). Checks a copied token cache against its own manifest:
# the sha256 of manifest.jsonl against the expected value, then size and md5 of every clip file against its manifest line.
# Usage: python verify_cache.py <cache split dir with manifest.jsonl> <expected sha256 of manifest.jsonl>
import hashlib
import json
import os
import sys

d, want = sys.argv[1], sys.argv[2]
mp = os.path.join(d, "manifest.jsonl")
raw = open(mp, "rb").read()
got = hashlib.sha256(raw).hexdigest()
print("manifest sha256", got[:16], "MANIFEST-OK" if got == want else "MANIFEST-MISMATCH")
rows = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
bad = []
for r in rows:
    p = os.path.join(d, r["file"])
    if not os.path.isfile(p) or os.path.getsize(p) != r["bytes"]:
        bad.append((r["clip"], "missing or size"))
        continue
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    if h.hexdigest() != r["md5"]:
        bad.append((r["clip"], "md5"))
print("clips", len(rows), "bad", len(bad), bad[:3])
print("ZZ VERIFY-CACHE DONE" if (got == want and not bad and len(rows) == 141) else "ZZ VERIFY-CACHE STOP")
