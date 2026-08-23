"""Pull the RAW parity VAL epcache (600 eps) from the HF mirror onto Thor,
then VERIFY every file by sha256 against the HF LFS digest and torch.load a
sample.

Size alone is not evidence and exit code is not evidence. This programme has
been bitten by silent truncation with exit 0 -- most recently a 21 %-truncated
episode that every prior Thor eval had been scoring against. So the pull is not
"done" until the digests match.

The token is read in place from ~/.hf_token and never printed or placed in argv.
Logs go to /tmp = LOCAL disk.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import torch

REPO = "Sayood/tanitad-physicalai-w120-256x640cyl"
SUB = "epcache-256px-phase0/physicalai-val-0c5f7dac3b11"
DEST = Path.home() / "epcache"
SHAS = Path("/tmp/hf_val600_sha.txt")
OUT = Path("/tmp/pull_val600_verify.json")

tok = Path.home().joinpath(".hf_token").read_text().strip()
os.environ["HF_TOKEN"] = tok
from huggingface_hub import snapshot_download  # noqa: E402

t0 = time.time()
print(json.dumps({"stage": "download_start", "repo": REPO, "subdir": SUB,
                  "dest": str(DEST),
                  "t": time.strftime("%FT%TZ", time.gmtime())}), flush=True)

snapshot_download(REPO, repo_type="dataset", local_dir=str(DEST),
                  allow_patterns=[SUB + "/*"], max_workers=8, token=tok)

d = DEST / SUB
eps = sorted(d.glob("ep_*.pt"))
tot = sum(f.stat().st_size for f in eps)
dt = time.time() - t0
print(json.dumps({"stage": "download_done", "n_ep": len(eps), "bytes": tot,
                  "seconds": round(dt, 1),
                  "MB_per_s": round(tot / 1e6 / max(dt, 1e-9), 1)}), flush=True)

expect = {}
for line in SHAS.read_text().splitlines():
    if line.strip():
        sha, name = line.split()
        expect[name] = sha

rows = []
t1 = time.time()
for i, name in enumerate(sorted(expect)):
    f = d / name
    rec = {"name": name, "expect_sha": expect[name]}
    if not f.exists():
        rec["status"] = "MISSING"
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
    # torch.load every 25th file: a full 600-file load is ~2 min of pure CPU
    # for little marginal evidence once the digest matches bit-for-bit.
    if i % 25 == 0:
        try:
            obj = torch.load(f, map_location="cpu", weights_only=False)
            rec["load_ok"] = True
            rec["keys"] = sorted(obj.keys()) if isinstance(obj, dict) else None
            del obj
        except Exception as exc:  # noqa: BLE001
            rec["load_ok"] = False
            rec["load_error"] = f"{type(exc).__name__}: {exc}"
    rec["status"] = "OK" if rec.get("sha_ok") and rec.get("load_ok", True) else "BAD"
    rows.append(rec)

summary = {
    "dir": str(d),
    "n": len(rows),
    "ok": sum(1 for r in rows if r["status"] == "OK"),
    "bad": [r["name"] for r in rows if r["status"] != "OK"],
    "loaded": sum(1 for r in rows if r.get("load_ok")),
    "verify_seconds": round(time.time() - t1, 1),
    "total_bytes": sum(r.get("bytes", 0) for r in rows),
    "expected_bytes": 70389845888,
    "expected_n": 600,
}
OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
print(json.dumps(summary), flush=True)
