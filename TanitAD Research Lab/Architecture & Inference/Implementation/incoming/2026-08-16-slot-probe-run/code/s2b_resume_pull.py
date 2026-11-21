"""S2b — RESUMABLE range download of one HF file.

⛔ WHY NOT hf_hub_download: two concurrent calls for the same file deadlocked on
the shared .lock and the .incomplete stalled at 268 MB (MEASURED tonight); each
retry also restarted from byte 0. This does explicit HTTP Range resume against
the resolved CDN URL, appends to a .part, and verifies the final size.
"""
from __future__ import annotations
import os
import re
import sys
import time
from pathlib import Path

import truststore
truststore.inject_into_ssl()
import urllib.request  # noqa: E402
from huggingface_hub import get_hf_file_metadata, hf_hub_url  # noqa: E402

TOK = re.search(r"hf_[A-Za-z0-9]+", Path(
    r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\Keys.txt"
).read_text(errors="ignore")).group(0)

repo, rel, dest = sys.argv[1], sys.argv[2], Path(sys.argv[3])
dest.parent.mkdir(parents=True, exist_ok=True)
part = dest.with_suffix(dest.suffix + ".part")

url = hf_hub_url(repo, rel, repo_type="model")
meta = get_hf_file_metadata(url, token=TOK)
total = int(meta.size)
print(f"[s2b] {rel}: {total/1e9:.3f} GB -> {dest}", flush=True)

hdrs_base = {"Authorization": f"Bearer {TOK}"}
t0 = time.time()
stall = 0
while True:
    have = part.stat().st_size if part.exists() else 0
    if have >= total:
        break
    h = dict(hdrs_base, Range=f"bytes={have}-")
    try:
        req = urllib.request.Request(meta.location or url, headers=h)
        with urllib.request.urlopen(req, timeout=120) as r:
            with open(part, "ab") as fh:
                last = time.time()
                while True:
                    blk = r.read(1 << 20)
                    if not blk:
                        break
                    fh.write(blk)
                    now = time.time()
                    if now - last > 20:
                        n = fh.tell()
                        print(f"[s2b] {n/1e6:8.1f}/{total/1e6:.1f} MB "
                              f"({100*n/total:5.1f} %)  "
                              f"{n/1e6/max(now-t0,1e-9):.2f} MB/s", flush=True)
                        last = now
    except Exception as ex:  # noqa: BLE001
        got = part.stat().st_size if part.exists() else 0
        print(f"[s2b] resume after {type(ex).__name__} at {got/1e6:.1f} MB: "
              f"{str(ex)[:120]}", flush=True)
        if got <= have:
            stall += 1
            if stall > 40:
                raise SystemExit("[s2b] no forward progress in 40 retries")
            time.sleep(min(5 * stall, 30))
        else:
            stall = 0

n = part.stat().st_size
if n != total:
    raise SystemExit(f"[s2b] size mismatch {n} != {total}")
part.replace(dest)
print(f"[s2b] OK {dest} {n/1e9:.3f} GB in {(time.time()-t0)/60:.1f} min",
      flush=True)
