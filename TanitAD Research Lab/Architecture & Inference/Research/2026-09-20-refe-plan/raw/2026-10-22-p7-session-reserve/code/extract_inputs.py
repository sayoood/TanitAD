#!/usr/bin/env python3
"""Derive the small, pinned inputs of the P7 session-reserve draw from the large source artifacts (CPU, read-only, no network).

The draw (`draw_reserve.py`) reads only the files this script writes into `raw/src/`, so the draw and its tests run standalone and the
provenance of every input is one sha256 away. Sources are the artifacts as exported from the local mirror at commit f32ed6b with
`git show`; pass the exported copies (or the repo paths) on the command line.

    python extract_inputs.py --navtrain-yaml PK/splits/navtrain.yaml \
        --navtest-plan PK/raw/2026-10-19-search-p2-confirm/c2/result/navtest_plan.json \
        --navtest-cluster-map FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-estimator-and-route-leak/raw/cluster_maps/navtest.json \
        --aux-log-map PK/raw/2026-10-12-aux-label-store/log_map.json --out raw/src

PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.

Writes  navtrain_logs.txt   the 1,192 log names of navtrain.yaml (sorted), one per line
        navtest_tokens.txt.gz  the 12,146 navtest tokens (sorted; gzip mtime 0), for the token-level disjointness test
        navtest_logs.json   {"logs": [...136...], "tokens": 12146, "sources": {sha256 of both sources}}; REFUSES if the two navtest
                            sources disagree on the token set or the log set (two independent harness artifacts must agree)
        vehicle_city.json   {vehicle: nuPlan map name}; REFUSES if any vehicle spans two maps (MEASURED 2026-10-22: each of the 16 vehicles
                            drives in exactly one city), from the aux label store's `log_map.json` (462/462 DB agreement, INHERITED)
        SOURCES.json        sha256 + byte size of every source and of the already-present token table
"""
import argparse
import gzip
import hashlib
import json
import os
import re
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def yaml_logs(path):
    txt = open(path, encoding="utf-8").read()
    m = re.search(r"\nlog_names:\n((?:[ \t]+- .*\n)+)", txt)
    if not m:
        raise SystemExit("navtrain.yaml: no log_names block")
    logs = [ln.strip()[2:].strip().strip("'\"") for ln in m.group(1).splitlines() if ln.strip()]
    if len(logs) != len(set(logs)):
        raise SystemExit("navtrain.yaml: duplicate log names")
    return sorted(logs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--navtrain-yaml", required=True)
    ap.add_argument("--navtest-plan", required=True)
    ap.add_argument("--navtest-cluster-map", required=True)
    ap.add_argument("--aux-log-map", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    logs = yaml_logs(a.navtrain_yaml)
    open(os.path.join(a.out, "navtrain_logs.txt"), "w", newline="\n").write("\n".join(logs) + "\n")

    plan = json.load(open(a.navtest_plan, encoding="utf-8"))
    cmap = json.load(open(a.navtest_cluster_map, encoding="utf-8"))
    t1, t2 = set(plan["tokens"]), set(cmap["token_to_log_name"])
    l1, l2 = set(plan["token_log"].values()), set(cmap["token_to_log_name"].values())
    if t1 != t2 or l1 != l2:
        raise SystemExit("the two navtest sources DISAGREE: tokens equal=%s logs equal=%s" % (t1 == t2, l1 == l2))
    with gzip.GzipFile(os.path.join(a.out, "navtest_tokens.txt.gz"), "wb", mtime=0) as g:     # mtime=0: a byte-stable file
        g.write(("\n".join(sorted(t1)) + "\n").encode())
    srcs = {
        "navtest_plan.json": {"sha256": sha256(a.navtest_plan), "bytes": os.path.getsize(a.navtest_plan)},
        "navtest_cluster_map.json": {"sha256": sha256(a.navtest_cluster_map), "bytes": os.path.getsize(a.navtest_cluster_map)},
    }
    json.dump({"schema": "p7-navtest-logs-v1", "tokens": len(t1), "logs": sorted(l1), "sources": srcs,
               "note": "navtest = NAVSIM v1.1 navtest, 12,146 tokens; the devkit's own navtest scene-filter yaml is NOT in the repo (UNVERIFIED); "
                       "this set is the intersection-checked content of two independent harness artifacts"},
              open(os.path.join(a.out, "navtest_logs.json"), "w", newline="\n"), indent=1)

    lm = json.load(open(a.aux_log_map, encoding="utf-8"))["logs"]
    vc = {}
    for lg in logs:
        v = lg.rsplit("_", 2)[0].split("_")[1]
        m = lm[lg]["map"]
        if vc.setdefault(v, m) != m:
            raise SystemExit("vehicle %s spans two maps (%s, %s)" % (v, vc[v], m))
    json.dump({"schema": "p7-vehicle-city-v1", "vehicle_city": dict(sorted(vc.items())),
               "source": {"aux_log_map.json": {"sha256": sha256(a.aux_log_map), "bytes": os.path.getsize(a.aux_log_map)}}},
              open(os.path.join(a.out, "vehicle_city.json"), "w", newline="\n"), indent=1)

    sources = {
        "aux_log_map.json": {"sha256": sha256(a.aux_log_map), "bytes": os.path.getsize(a.aux_log_map)},
        "navtrain.yaml": {"sha256": sha256(a.navtrain_yaml), "bytes": os.path.getsize(a.navtrain_yaml), "log_names": len(logs)},
        **srcs,
    }
    tt = os.path.join(a.out, "navtrain_token_table.tsv.gz")
    if os.path.exists(tt):
        sources["navtrain_token_table.tsv.gz"] = {"sha256": sha256(tt), "bytes": os.path.getsize(tt)}
    json.dump(sources, open(os.path.join(a.out, "SOURCES.json"), "w", newline="\n"), indent=1)
    print("wrote navtrain_logs.txt (%d), navtest_logs.json (%d logs, %d tokens), SOURCES.json" % (len(logs), len(l1), len(t1)))


if __name__ == "__main__":
    sys.exit(main())
