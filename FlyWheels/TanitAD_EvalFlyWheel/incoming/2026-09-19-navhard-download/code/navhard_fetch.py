#!/usr/bin/env python3
"""Parallel, resumable, CONTENT-VERIFIED fetch of the NavSim navhard_two_stage archives.

PI authorisation, in chat, 2026-09-19: "Pickles + sensors" — the three archives of
`navsim-v2/navsim_v2.2_navhard_two_stage_{scene_pickles,curr_sensors,hist_sensors}.tar.gz`
from the public dataset `huggingface.co/datasets/OpenDriveLab/OpenScene`.
Sizes MEASURED by HEAD the same day (`X-Linked-Size`): 211,118,643 · 12,804,089,546 ·
25,521,966,184 B = 38.54 GB.

WHY THIS SHAPE (the dev-box link has cost time before):
* single-connection throughput on this box was 1.3-1.4 MB/s on 2026-08-29
  (`C:/Users/Admin/navsim/dl_warmup.log`) and plain curl RESETS mid-transfer — so a
  38.5 GB single pass is ~8 h with many restarts. The CDN honours byte ranges, so the file
  is split into N ranges fetched concurrently, each resumable from its own part-file size.
* ⛔ "the command finished" is not evidence. Completion is asserted on CONTENT:
  every part must reach its exact byte count, the joined file must equal X-Linked-Size,
  and its sha256 must equal HF's X-Linked-ETag (the content hash HF serves for the file).
  A mismatch is a FAILURE, never a warning.
* TLS on this box needs `truststore` (certifi fails behind the proxy and the failure reads
  exactly like an outage).

Public dataset bytes only: no inference, no Space, no metered HF compute.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = "https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/navsim-v2/"
FILES = {
    "scene_pickles": "navsim_v2.2_navhard_two_stage_scene_pickles.tar.gz",
    "curr_sensors": "navsim_v2.2_navhard_two_stage_curr_sensors.tar.gz",
    "hist_sensors": "navsim_v2.2_navhard_two_stage_hist_sensors.tar.gz",
}
CHUNK = 1 << 20


def _ssl() -> None:
    import truststore
    truststore.inject_into_ssl()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):  # noqa: D401 - keep the 302 visible
        return None


def head_meta(url: str) -> tuple[int, str]:
    """(size, sha256) from the HF resolve endpoint's 302 headers."""
    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(url, method="HEAD")
    try:
        resp = opener.open(req, timeout=60)
        hdr = resp.headers
    except urllib.error.HTTPError as e:  # the 302 surfaces here with NoRedirect
        hdr = e.headers
    size = int(hdr["X-Linked-Size"])
    etag = hdr["X-Linked-ETag"].strip('"')
    return size, etag


def fetch_range(url: str, part: str, lo: int, hi: int, log, stop: threading.Event,
                tries: int = 200) -> None:
    """Fetch bytes [lo, hi] into `part`, resuming from the part file's current size."""
    want = hi - lo + 1
    for attempt in range(1, tries + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have == want:
            return
        if have > want:
            raise RuntimeError(f"{part}: {have} B > expected {want} B — refusing to guess")
        req = urllib.request.Request(url, headers={"Range": f"bytes={lo + have}-{hi}"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab") as f:
                if r.status != 206:
                    raise RuntimeError(f"expected HTTP 206, got {r.status}")
                while not stop.is_set():
                    buf = r.read(CHUNK)
                    if not buf:
                        break
                    f.write(buf)
        except Exception as e:  # resets are EXPECTED on this link; the part keeps progress
            log(f"  retry {os.path.basename(part)} #{attempt}: {type(e).__name__}: {str(e)[:100]}")
            time.sleep(min(2 * attempt, 30))
        if stop.is_set():
            return
    raise RuntimeError(f"{part}: gave up after {tries} attempts")


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(8 * CHUNK), b""):
            h.update(buf)
    return h.hexdigest()


def fetch_file(key: str, dest: str, n_parts: int, receipt: dict, log) -> bool:
    name = FILES[key]
    url = BASE + name
    out = os.path.join(dest, name)
    size, etag = head_meta(url)
    rec = {"file": name, "url": url, "x_linked_size": size, "x_linked_etag": etag,
           "n_parts": n_parts, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    receipt[key] = rec
    if os.path.exists(out) and os.path.getsize(out) == size and sha256_of(out) == etag:
        rec.update(status="ALREADY_COMPLETE_VERIFIED", sha256=etag)
        log(f"{name}: already complete and sha256-verified")
        return True
    parts_dir = out + ".parts"
    os.makedirs(parts_dir, exist_ok=True)
    step = -(-size // n_parts)
    ranges = [(i, i * step, min(size, (i + 1) * step) - 1) for i in range(n_parts)
              if i * step < size]
    stop = threading.Event()
    errors: list[str] = []

    def worker(i, lo, hi):
        try:
            fetch_range(url, os.path.join(parts_dir, f"{i:03d}"), lo, hi, log, stop)
        except Exception as e:
            errors.append(f"part {i}: {e}")

    t0 = time.time()
    threads = [threading.Thread(target=worker, args=r, daemon=True) for r in ranges]
    for t in threads:
        t.start()
    last = -1
    while any(t.is_alive() for t in threads):
        time.sleep(30)
        got = sum(os.path.getsize(os.path.join(parts_dir, f"{i:03d}"))
                  for i, _, _ in ranges if os.path.exists(os.path.join(parts_dir, f"{i:03d}")))
        rate = got / max(time.time() - t0, 1) / 1e6
        if got != last:
            log(f"{name}: {got / 1e9:.2f}/{size / 1e9:.2f} GB ({100 * got / size:.1f} %), "
                f"{rate:.1f} MB/s avg")
            last = got
    if errors:
        rec.update(status="FAILED", errors=errors)
        log(f"⛔ {name}: {errors}")
        return False
    # join, then verify by CONTENT
    for i, lo, hi in ranges:
        p = os.path.join(parts_dir, f"{i:03d}")
        if os.path.getsize(p) != hi - lo + 1:
            rec.update(status="FAILED", errors=[f"part {i} size {os.path.getsize(p)} != {hi - lo + 1}"])
            return False
    tmp = out + ".joining"
    with open(tmp, "wb") as fo:
        for i, _, _ in ranges:
            with open(os.path.join(parts_dir, f"{i:03d}"), "rb") as fi:
                for buf in iter(lambda: fi.read(8 * CHUNK), b""):
                    fo.write(buf)
    got_size = os.path.getsize(tmp)
    got_sha = sha256_of(tmp)
    rec.update(bytes=got_size, sha256=got_sha, wall_s=round(time.time() - t0, 1),
               mean_MBps=round(got_size / max(time.time() - t0, 1) / 1e6, 2))
    if got_size != size or got_sha != etag:
        rec.update(status="FAILED_VERIFICATION")
        log(f"⛔ {name}: size {got_size} vs {size}, sha256 {got_sha} vs etag {etag}")
        return False
    os.replace(tmp, out)
    for i, _, _ in ranges:
        os.remove(os.path.join(parts_dir, f"{i:03d}"))
    os.rmdir(parts_dir)
    rec["status"] = "COMPLETE_VERIFIED"
    log(f"✅ {name}: {got_size} B, sha256 == X-Linked-ETag ({etag[:16]}…)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="+", choices=sorted(FILES))
    ap.add_argument("--dest", default="C:/Users/Admin/navsim/data/navsim-v2")
    ap.add_argument("--parts", type=int, default=8)
    ap.add_argument("--receipt", required=True)
    a = ap.parse_args()
    _ssl()
    os.makedirs(a.dest, exist_ok=True)
    receipt: dict = {"authorised": "PI in chat 2026-09-19 ('Pickles + sensors')",
                     "dest": a.dest}
    lock = threading.Lock()

    def log(msg: str) -> None:
        with lock:
            print(time.strftime("%H:%M:%S"), msg, flush=True)

    ok = True
    for k in a.keys:
        ok &= fetch_file(k, a.dest, a.parts, receipt, log)
        with open(a.receipt, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=1)
    log("ALL_VERIFIED" if ok else "⛔ NOT ALL VERIFIED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
