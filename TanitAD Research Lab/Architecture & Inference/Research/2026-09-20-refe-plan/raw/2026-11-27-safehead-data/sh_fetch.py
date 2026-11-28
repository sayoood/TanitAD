#!/usr/bin/env python3
"""Frame extraction for the SAFE-HEAD cache (stdlib only).

    sh_fetch.py archive <name>      stream one public OpenScene trainval camera archive (curl | tar), extract ONLY the frames in plan/need_<name>.json
                                    to W/frames/<log>/<CAM>/<hash>.jpg, stop reading as soon as all are found, write plan/fetch_<name>.json.
                                    The archive itself is never stored (peak disk = the extracted frames). Exit 0 only if every needed frame was found.
    sh_fetch.py local <log>         extract the needed frames of a locally complete log from its camera zips (nuplan-camera/scenarios4, mini).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sh_lib as L  # noqa: E402

LOCAL_DIRS = ["E:/Projects/TanitAD/data/nuplan-camera/scenarios4", "E:/Projects/TanitAD/data/nuplan-camera/mini"]


def write_atomic(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    for att in range(12):                              # Windows: an indexer/AV scan can hold the fresh .part for a moment (WinError 32 seen 2026-11-26)
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.5 * (att + 1))
    os.replace(tmp, path)


def fetch_archive(name):
    plan = json.load(open(L.wp("plan", "PLAN.json")))
    arch = next(a for a in plan["archives"] if a["name"] == name)
    need = json.load(open(L.wp("plan", f"need_{name}.json")))            # rel -> log
    todo = {r for r in need if not os.path.exists(L.wp("frames", r))}
    found, t0, nbytes = 0, time.time(), 0
    if todo:
        # no auth needed (public dataset: HEAD returns 302 -> 200 anonymously); curl follows the CDN redirect
        p = subprocess.Popen(["curl", "-sL", "--fail", "--retry", "3", "--retry-delay", "5", arch["url"]], stdout=subprocess.PIPE, bufsize=1 << 22)
        try:
            tf = tarfile.open(fileobj=p.stdout, mode="r|gz")
            for m in tf:
                if not m.isfile() or not m.name.startswith(L.PREFIX):
                    continue
                rel = m.name[len(L.PREFIX):]
                if rel in todo:
                    data = tf.extractfile(m).read()
                    write_atomic(L.wp("frames", rel), data)
                    todo.discard(rel)
                    found += 1
                    nbytes += len(data)
                    if not todo:
                        break
        except (tarfile.TarError, OSError, EOFError) as e:
            print(f"STREAM ERROR {type(e).__name__}: {e}", flush=True)
        finally:
            p.kill()
            p.wait()
    miss = sorted(todo)
    by_log = {}
    for r in miss:
        by_log.setdefault(need[r], []).append(r)
    rep = {"archive": name, "bytes_archive": arch["bytes"], "needed": len(need), "found_now": found, "missing": len(miss), "missing_by_log": {k: len(v) for k, v in by_log.items()},
           "extracted_bytes": nbytes, "seconds": round(time.time() - t0, 1)}
    json.dump(rep, open(L.wp("plan", f"fetch_{name}.json"), "w"), indent=1)
    print("ZZFETCH " + json.dumps(rep), flush=True)
    return 0 if not miss else 3


def fetch_local(log):
    plan = json.load(open(L.wp("plan", "PLAN.json")))
    ent = next(x for x in plan["logs"] if x["log"] == log)
    fn = L.frame_names(ent["db"], ent["tokens"])
    rels = sorted({r for v in fn.values() if v for r in v})
    zips = {}
    n = miss = 0
    for rel in rels:
        dst = L.wp("frames", rel)
        if os.path.exists(dst):
            continue
        _lg, cam, base = rel.split("/")
        if cam not in zips:
            zp = next((f"{d}/{log}_{cam}.zip" for d in LOCAL_DIRS if os.path.exists(f"{d}/{log}_{cam}.zip")), None)
            zips[cam] = zipfile.ZipFile(zp) if zp else None
        z = zips[cam]
        try:
            write_atomic(dst, z.read(base))
            n += 1
        except (KeyError, AttributeError):
            miss += 1
    print("ZZFETCH " + json.dumps({"log": log, "needed": len(rels), "extracted": n, "missing": miss}), flush=True)
    return 0 if miss == 0 else 3


if __name__ == "__main__":
    mode, arg = sys.argv[1], sys.argv[2]
    sys.exit(fetch_archive(arg) if mode == "archive" else fetch_local(arg))
