#!/usr/bin/env python3
"""SimScale camera probe: which CAM_* folders does a round-4 history scene hold?  (PREPARED 2026-10-12; NOT RUN -- it downloads, so it waits
for the PI's permission.)

The 0.9 MB index cannot answer it (it names no sensor; simscale_r4_index_check.json).  This streams the FIRST `--max-mb` megabytes (default 24)
of ONE history shard through an HTTP Range request, decompresses them in memory with tarfile's streaming mode, and records the member NAMES only.
Nothing is written to disk except the JSON of names.  A gzip stream can be read from its start, so the first scene's folders appear within the
first few MB.

Run (dev box):  python -I simscale_camera_probe.py --out simscale_r4_camera_probe.json [--shard 0] [--max-mb 24]
"""
import argparse
import collections
import io
import json
import re
import sys
import tarfile

REPO, REV = "OpenDriveLab/SimScale", "06a4fc88f48d8c3ac7c16cc1df05f3caab6397ae"
SHARD = "SimScale_data/synthetic_reaction_pdm_v1.0-4/simscale_pdm_v1.0-4_sensor_blobs_hist/simscale_pdm_v1.0-4_sensor_blobs_hist_{}.tar.gz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--max-mb", type=int, default=24)
    a = ap.parse_args()
    import truststore
    truststore.inject_into_ssl()
    import requests
    from huggingface_hub import hf_hub_url
    url = hf_hub_url(REPO, SHARD.format(a.shard), repo_type="dataset", revision=REV)
    n = a.max_mb * 1024 * 1024
    r = requests.get(url, headers={"Range": f"bytes=0-{n - 1}"}, timeout=120, allow_redirects=True)
    r.raise_for_status()
    data = r.content[:n]
    names, err = [], None
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r|gz") as tf:
            for m in tf:
                names.append([m.name, int(m.size)])
    except Exception as e:                                       # noqa: BLE001 -- the stream is truncated ON PURPOSE; the end raises
        err = f"{type(e).__name__}: {str(e)[:120]}"
    scenes = collections.defaultdict(set)
    for nm, _ in names:
        m = re.search(r"([0-9a-f]{16}-\d{3})/([A-Z0-9_]+)/", nm)
        if m:
            scenes[m.group(1)].add(m.group(2))
    res = {"url_file": SHARD.format(a.shard), "revision": REV, "http_status": r.status_code, "bytes_read": len(data),
           "members_listed": len(names), "stream_end": err, "first_members": names[:40],
           "sensor_folders_per_scene": {k: sorted(v) for k, v in list(scenes.items())[:5]},
           "evidence": "MEASURED: member names from the first bytes of one shard; no file content kept"}
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: res[k] for k in ("http_status", "bytes_read", "members_listed", "sensor_folders_per_scene")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
