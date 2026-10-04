#!/usr/bin/env python3
"""Fetch chosen members of a PUBLIC remote zip over HTTP ranges, inflate them, and CRC-verify every one (VM side).

    python3 zip_member_fetch.py --manifest db_manifest_sub200.json --out <dir> [--streams 8] [--report fetch.json]

The manifest (written on the dev box from its copy of the SAME public zip, see build_sub200_bundle.py) lists per
member: name, local-header offset `lho`, compressed / uncompressed size, CRC-32, method. Per member this requests
[lho, lho + 30 + len(name) + 1024 + csz) in ONE range, checks the local header's signature AND name (a layout that
differs from the manifest fails loudly instead of inflating the wrong bytes), then stream-inflates exactly `csz`
bytes to <out>/<basename>.part, checks bytes == usz and CRC-32 == crc, and renames. Nothing is trusted unverified:
a member without a CRC match is a FAILURE (retried up to 3x), never a warning. Resumable (a finished file whose size
matches is re-CRC'd and kept). Persists the report after every member. stdlib only.
Prints ZZFETCH_OK / ZZFETCH_FAIL with counts, bytes and the aggregate compressed MB/s (the multi-stream rate).
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import threading
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

CHUNK = 1 << 20


def crc_file(p):
    c = 0
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            c = zlib.crc32(b, c)
    return c & 0xFFFFFFFF


def fetch_one(url, key, m, out_dir, tries=3):
    base = os.path.basename(m["name"])
    dst = os.path.join(out_dir, m.get("dest_name", base))
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) == m["usz"] and crc_file(dst) == m["crc"]:
        return {"key": key, "ok": True, "skipped": "present and CRC-verified", "csz": m["csz"], "s": 0.0}
    last = None
    for attempt in range(1, tries + 1):
        t0 = time.time()
        try:
            end = m["lho"] + 30 + len(m["name"].encode()) + 1024 + m["csz"] - 1
            req = urllib.request.Request(url, headers={"Range": f"bytes={m['lho']}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 206:
                    raise RuntimeError(f"HTTP {r.status}, expected 206")
                hdr = r.read(30)
                sig, _v, _f, meth, _t, _d, _crc, _cs, _us, nlen, elen = struct.unpack("<IHHHHHIIIHH", hdr)
                if sig != 0x04034B50:
                    raise RuntimeError(f"no local header at lho {m['lho']}")
                name = r.read(nlen).decode("utf-8", "replace")
                if name != m["name"]:
                    raise RuntimeError(f"local header names {name!r}, manifest {m['name']!r}")
                r.read(elen)
                if meth not in (0, 8):
                    raise RuntimeError(f"method {meth}")
                dec = zlib.decompressobj(-15) if meth == 8 else None
                left, crc, n = m["csz"], 0, 0
                with open(dst + ".part", "wb") as f:
                    while left > 0:
                        b = r.read(min(CHUNK, left))
                        if not b:
                            raise RuntimeError(f"short read, {left} B left")
                        left -= len(b)
                        o = dec.decompress(b) if dec else b
                        crc = zlib.crc32(o, crc)
                        n += len(o)
                        f.write(o)
                    if dec:
                        o = dec.flush()
                        crc = zlib.crc32(o, crc)
                        n += len(o)
                        f.write(o)
            crc &= 0xFFFFFFFF
            if n != m["usz"] or crc != m["crc"]:
                raise RuntimeError(f"verify failed: bytes {n} vs {m['usz']}, crc {crc:08x} vs {m['crc']:08x}")
            os.replace(dst + ".part", dst)
            dt = time.time() - t0
            return {"key": key, "ok": True, "csz": m["csz"], "usz": n, "crc_ok": True, "s": round(dt, 1),
                    "attempt": attempt, "mb_s": round(m["csz"] / 1e6 / max(dt, 1e-6), 1)}
        except Exception as e:  # noqa: BLE001 -- retried, then reported per member
            last = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(2 * attempt)
    return {"key": key, "ok": False, "error": last, "csz": m["csz"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--report", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    os.makedirs(a.out, exist_ok=True)
    mem = man["members"]
    keys = [k for k in mem if a.only is None or k in a.only]
    rep = {"url": man["url"], "streams": a.streams, "n": len(keys), "members": {},
           "t_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    lock = threading.Lock()
    t0 = time.time()

    def save():
        if a.report:
            with open(a.report + ".tmp", "w") as f:
                json.dump(rep, f, indent=1)
            os.replace(a.report + ".tmp", a.report)

    def job(k):
        r = fetch_one(man["url"], k, mem[k], a.out)
        with lock:
            rep["members"][k] = r
            save()
        return r

    # biggest first, so the tail of the run is not one large member on one stream
    order = sorted(keys, key=lambda k: -mem[k]["csz"])
    with ThreadPoolExecutor(max_workers=a.streams) as ex:
        res = list(ex.map(job, order))
    wall = time.time() - t0
    fetched = [r for r in res if r["ok"] and not r.get("skipped")]
    rep.update(wall_s=round(wall, 1), n_ok=sum(1 for r in res if r["ok"]), n_fail=sum(1 for r in res if not r["ok"]),
               csz_fetched=sum(r["csz"] for r in fetched),
               aggregate_mb_s=round(sum(r["csz"] for r in fetched) / 1e6 / max(wall, 1e-6), 1),
               t_end_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    save()
    ok = rep["n_fail"] == 0
    print(("ZZFETCH_OK " if ok else "ZZFETCH_FAIL ") + json.dumps({k: rep[k] for k in (
        "n", "n_ok", "n_fail", "csz_fetched", "wall_s", "aggregate_mb_s", "streams")}), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
