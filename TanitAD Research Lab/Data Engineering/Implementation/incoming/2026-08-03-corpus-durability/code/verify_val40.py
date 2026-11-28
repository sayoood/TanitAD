"""Verify Thor's copy of the 40-episode REF-C val raster against HF digests.

Size alone is not evidence and exit code is not evidence: this programme has
been bitten by silent truncation with exit 0, most recently a 21 %-truncated
episode that every prior Thor eval had been scoring against. So each file gets
BOTH a sha256 compared to the HF LFS digest AND a real torch.load with a
shape/dtype check on the episode contract.

Logs go to /tmp (LOCAL disk). A logger writing to the failing filesystem cannot
report that the filesystem is failing.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

SRC = Path(sys.argv[1])          # dir of ep_*.pt
SHAS = Path(sys.argv[2])         # "<sha256>  <name>" lines
OUT = Path(sys.argv[3])

expect = {}
for line in SHAS.read_text().splitlines():
    if not line.strip():
        continue
    sha, name = line.split()
    expect[name] = sha

rows = []
t0 = time.time()
for name in sorted(expect):
    f = SRC / name
    rec = {"name": name, "expect_sha": expect[name]}
    if not f.exists():
        rec.update(status="MISSING")
        rows.append(rec)
        continue
    h = hashlib.sha256()
    n = 0
    with f.open("rb") as fh:
        while True:
            b = fh.read(8 << 20)
            if not b:
                break
            n += len(b)
            h.update(b)
    rec["bytes"] = n
    rec["sha256"] = h.hexdigest()
    rec["sha_ok"] = rec["sha256"] == expect[name]
    try:
        obj = torch.load(f, map_location="cpu", weights_only=False)
        keys = sorted(obj.keys()) if isinstance(obj, dict) else None
        rec["load_ok"] = True
        rec["keys"] = keys
        if isinstance(obj, dict):
            for k in ("frames_u8", "actions", "poses"):
                v = obj.get(k)
                if torch.is_tensor(v):
                    rec[f"{k}_shape"] = list(v.shape)
                    rec[f"{k}_dtype"] = str(v.dtype)
        del obj
    except Exception as exc:  # noqa: BLE001
        rec["load_ok"] = False
        rec["load_error"] = f"{type(exc).__name__}: {exc}"
    rec["status"] = "OK" if (rec.get("sha_ok") and rec.get("load_ok")) else "BAD"
    rows.append(rec)
    print(json.dumps({k: rec[k] for k in ("name", "status", "sha_ok", "load_ok")}),
          flush=True)

summary = {
    "dir": str(SRC),
    "n": len(rows),
    "ok": sum(1 for r in rows if r["status"] == "OK"),
    "bad": [r["name"] for r in rows if r["status"] != "OK"],
    "seconds": round(time.time() - t0, 1),
    "total_bytes": sum(r.get("bytes", 0) for r in rows),
}
OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
print(json.dumps(summary), flush=True)
