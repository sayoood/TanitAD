"""Measure the dev box's UPLOAD bandwidth — small, non-destructive, no corpus data.

Payload: ``os.urandom`` bytes (incompressible, so a proxy cannot flatter the rate by
compressing, and they carry no information). Endpoint: Cloudflare's public speed-test
upload sink, which discards what it receives. ⚠️ It is a NEUTRAL THIRD-PARTY endpoint, not one
we control — no pod is running and no HF repo may be written — so this measures the box's
uplink, not the path to HF or to a pod specifically.

    python uplink_probe.py <out.json>
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import json
import os
import statistics as st
import sys
import time

import truststore

truststore.inject_into_ssl()
import requests  # noqa: E402

URL = "https://speed.cloudflare.com/__up"
MB = 1_000_000


def post(sess: requests.Session, nbytes: int) -> dict:
    body = os.urandom(nbytes)
    t0 = time.perf_counter()
    r = sess.post(URL, data=body, timeout=600,
                  headers={"Content-Type": "application/octet-stream"})
    s = time.perf_counter() - t0
    return {"bytes": nbytes, "s": round(s, 3), "status": r.status_code,
            "MBps": round(nbytes / s / MB, 3)}


def main(out_path: str) -> int:
    rep = {"endpoint": URL, "payload": "os.urandom", "started_local": dt.datetime.now().isoformat(timespec="seconds"),
           "trials": {}}
    with requests.Session() as s:
        rep["trials"]["latency_1kB"] = [post(s, 1000) for _ in range(3)]
        if any(t["status"] != 200 for t in rep["trials"]["latency_1kB"]):
            raise SystemExit(f"endpoint refused: {rep['trials']['latency_1kB']}")
        rep["trials"]["single_10MB"] = [post(s, 10 * MB) for _ in range(2)]
        rep["trials"]["single_25MB"] = [post(s, 25 * MB) for _ in range(3)]

    def one(n):
        with requests.Session() as s:
            return post(s, n)

    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(4) as ex:
        par = list(ex.map(one, [25 * MB] * 4))
    wall = time.perf_counter() - t0
    rep["trials"]["parallel_4x25MB"] = {"streams": par, "wall_s": round(wall, 3),
                                         "aggregate_MBps": round(100 * MB / wall / MB, 3)}
    single = [t["MBps"] for t in rep["trials"]["single_25MB"] if t["status"] == 200]
    rep["summary"] = {
        "single_stream_25MB_MBps_median": st.median(single),
        "single_stream_25MB_MBps_range": [min(single), max(single)],
        "parallel_4_streams_aggregate_MBps": rep["trials"]["parallel_4x25MB"]["aggregate_MBps"],
        "latency_1kB_s_median": st.median(t["s"] for t in rep["trials"]["latency_1kB"]),
        "total_bytes_sent": 3 * 1000 + 2 * 10 * MB + 3 * 25 * MB + 4 * 25 * MB,
        "all_status_200": all(t["status"] == 200 for k, v in rep["trials"].items()
                              for t in (v["streams"] if isinstance(v, dict) else v)),
    }
    rep["finished_local"] = dt.datetime.now().isoformat(timespec="seconds")
    open(out_path, "w", encoding="utf-8", newline="\n").write(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
