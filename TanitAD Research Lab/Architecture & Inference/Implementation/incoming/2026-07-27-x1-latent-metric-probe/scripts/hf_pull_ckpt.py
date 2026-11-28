#!/usr/bin/env python3
"""Pull a gated HF checkpoint to the dev box with parallel range requests.

Dev-box facts this encodes (each cost time before):
  * certifi fails behind the local TLS proxy -> ``truststore.inject_into_ssl()``.
  * the token lives in ``Keys.txt`` (git-ignored) and is read IN PLACE; it is
    never printed, never written to a file, and never passed as an argv value.
  * single-stream throughput measured 3.96 MB/s; 8 ranged streams are used.

Usage:
  python hf_pull_ckpt.py --repo Sayood/tanitad-flagship-4b-speedjerk \
      --file ckpt.pt --out C:/Users/Admin/tanitad-data/eval/v1_speedjerk_ckpt.pt
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time
import urllib.request

import truststore

truststore.inject_into_ssl()

KEYS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..",
                    "..", "Keys.txt")


def token() -> str:
    with open(os.path.abspath(KEYS), "r", encoding="utf-8",
              errors="replace") as fh:
        m = re.search(r"hf_[A-Za-z0-9]+", fh.read())
    if not m:
        raise SystemExit("no hf token found in Keys.txt")
    return m.group(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--file", default="ckpt.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    tok = token()
    url = f"https://huggingface.co/{a.repo}/resolve/main/{a.file}"
    hdr = {"Authorization": "Bearer " + tok}

    r = urllib.request.urlopen(
        urllib.request.Request(url, headers={**hdr, "Range": "bytes=0-0"}),
        timeout=120)
    total = int(r.headers["Content-Range"].split("/")[-1])
    print(f"{a.repo}/{a.file}  {total/1e9:.3f} GB", flush=True)

    if os.path.exists(a.out) and os.path.getsize(a.out) == total:
        print("already complete:", a.out)
        return

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "wb") as fh:            # pre-allocate
        fh.truncate(total)

    n = a.threads
    chunk = (total + n - 1) // n
    done = [0] * n
    err: list[BaseException] = []

    def worker(i):
        lo = i * chunk
        hi = min(total - 1, lo + chunk - 1)
        if lo > hi:
            return
        try:
            rq = urllib.request.Request(
                url, headers={**hdr, "Range": f"bytes={lo}-{hi}"})
            with urllib.request.urlopen(rq, timeout=600) as resp, \
                    open(a.out, "r+b") as fh:
                fh.seek(lo)
                while True:
                    b = resp.read(4 << 20)
                    if not b:
                        break
                    fh.write(b)
                    done[i] += len(b)
        except BaseException as e:               # noqa: BLE001
            err.append(e)

    t0 = time.time()
    ths = [threading.Thread(target=worker, args=(i,), daemon=True)
           for i in range(n)]
    for t in ths:
        t.start()
    while any(t.is_alive() for t in ths):
        time.sleep(10)
        g = sum(done)
        print(f"  {g/1e9:.2f}/{total/1e9:.2f} GB  "
              f"{g/1e6/max(1e-9, time.time()-t0):.1f} MB/s", flush=True)
    for t in ths:
        t.join()
    if err:
        raise SystemExit(f"download failed: {err[0]!r}")
    got = os.path.getsize(a.out)
    print(f"done {got} bytes in {time.time()-t0:.0f}s -> {a.out}")
    if got != total:
        sys.exit(f"SIZE MISMATCH {got} != {total}")


if __name__ == "__main__":
    main()
