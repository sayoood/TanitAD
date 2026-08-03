#!/usr/bin/env python3
"""Survey the 1607-scene HF NuRec release for a JUNCTION TRAVERSAL WITH A CHOICE.

Why this exists
---------------
MEASURED on the night scene 00040136 (``results/junction_00040136.json``): the
ego is inside a junction for 46 of its 202 poses, so "the clip contains no
junction" is FALSE — but every one of its four traversals offers exactly ONE
lane-level continuation and turns the car by at most 2.58 deg.  The scarce
thing is therefore not a junction, it is a **branch the ego could have taken
differently**.  This tool searches for that.

How it reads 1607 x ~2 GB scenes without downloading 3 TB
---------------------------------------------------------
A USDZ is a zip in which every member is STORED, never deflated (verified on
00040136: 39/39 members ``compress_type=0``).  So an HTTP ``Range`` request
against the HF ``resolve`` URL can pull one member out of the archive.  We
mount the remote file as a seekable file object and hand it to the stdlib
``zipfile``, which then reads the End-Of-Central-Directory and the member we
name — about 32 KB of traffic per scene in stage 1 instead of 2 GB.

Two stages
----------
stage 1 (all scenes, ~32 KB each): ``clipgt/egomotion_estimate.parquet`` (the
        202 ego poses) + ``clipgt/intersection_area.parquet`` (NVIDIA's own
        labelled intersection polygons).  Both live in ONE clip frame, so no
        alignment is needed and no transform can be blamed.  Emits, per
        traversed polygon: poses inside, arc length inside, heading change
        through, and how much clip remains after the exit.
stage 2 (shortlist, ~750 KB each): + ``map.xodr`` + ``pose_record.json`` +
        ``clipgt/lane.parquet``, then the full lane-level option count from
        ``junction_probe.py``.

Auth: reads ``HF_TOKEN`` from the environment.  Never pass it on argv.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from junction_probe import (point_in_poly, polyline_dist,  # noqa: E402
                            signed_poly_dist, wrap_deg)

REPO = "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
RELEASE = "sample_set/26.04_release"

STAGE1_MEMBERS = ["clipgt/egomotion_estimate.parquet",
                  "clipgt/intersection_area.parquet"]
STAGE2_MEMBERS = ["map.xodr", "pose_record.json", "clipgt/lane.parquet",
                  "clipgt/wait_line.parquet", "data_info.json"]


def _headers():
    tok = os.environ.get("HF_TOKEN", "")
    h = {"User-Agent": "tanitad-nurec-survey/1"}
    if tok:
        h["Authorization"] = "Bearer " + tok
    return h


# --------------------------------------------------------------------------
# a seekable file object backed by HTTP Range
# --------------------------------------------------------------------------
class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file.

    Over-reads by ``block`` bytes and caches, because ``zipfile`` issues many
    tiny reads while walking the central directory and one request per read
    would make the survey network-bound on latency.
    """

    def __init__(self, url, headers=None, block=1 << 18, retries=4):
        self.url = url
        self.h = dict(headers or {})
        self.block = block
        self.retries = retries
        self._pos = 0
        self._cache = {}          # block index -> bytes
        self.bytes_fetched = 0
        self.n_requests = 0
        self.size = self._probe_size()

    def _probe_size(self):
        req = urllib.request.Request(self.url, headers={**self.h, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            cr = r.headers.get("Content-Range", "")
            self.n_requests += 1
            if "/" in cr:
                return int(cr.rsplit("/", 1)[1])
            cl = r.headers.get("Content-Length")
            return int(cl) if cl else 0

    def _fetch(self, start, end):
        last = None
        for i in range(self.retries):
            try:
                req = urllib.request.Request(
                    self.url, headers={**self.h, "Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    b = r.read()
                self.n_requests += 1
                self.bytes_fetched += len(b)
                return b
            except (urllib.error.URLError, TimeoutError, OSError) as e:  # noqa: PERF203
                last = e
                time.sleep(1.0 + 2.0 * i)
        raise last

    # -- io api ----------------------------------------------------------
    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, off, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self._pos = off
        elif whence == io.SEEK_CUR:
            self._pos += off
        else:
            self._pos = self.size + off
        return self._pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self._pos
        n = max(0, min(n, self.size - self._pos))
        if n == 0:
            return b""
        out = bytearray()
        pos = self._pos
        remaining = n
        while remaining > 0:
            bi = pos // self.block
            if bi not in self._cache:
                s = bi * self.block
                e = min(s + self.block, self.size) - 1
                self._cache[bi] = self._fetch(s, e)
                if len(self._cache) > 64:                     # bound memory
                    for k in list(self._cache)[:32]:
                        if k != bi:
                            del self._cache[k]
            blk = self._cache[bi]
            off = pos - bi * self.block
            take = min(remaining, len(blk) - off)
            if take <= 0:
                break
            out += blk[off:off + take]
            pos += take
            remaining -= take
        self._pos = pos
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def open_scene_members(scene_id, members, release=RELEASE, block=1 << 18):
    """{member -> bytes} pulled by HTTP Range out of the remote usdz."""
    url = f"{RESOLVE}{release}/{scene_id}/{scene_id}.usdz"
    f = HttpRangeFile(url, _headers(), block=block)
    z = zipfile.ZipFile(f)
    names = set(z.namelist())
    out = {}
    for m in members:
        if m in names:
            out[m] = z.read(m)
    return out, {"usdz_bytes": f.size, "fetched_bytes": f.bytes_fetched,
                 "n_requests": f.n_requests, "members_present": sorted(names)}


# --------------------------------------------------------------------------
# parquet from bytes
# --------------------------------------------------------------------------
def pq_rows(raw, name):
    import pyarrow.parquet as pq
    t = pq.read_table(io.BytesIO(raw))
    keys = t.column("key").to_pylist() if "key" in t.schema.names else None
    pay = t.column(name).to_pylist() if name in t.schema.names else None
    return keys, pay


def _xyz(seq):
    return np.array([[p["x"], p["y"], p["z"]] for p in seq], float)


# --------------------------------------------------------------------------
# stage 1 scoring
# --------------------------------------------------------------------------
def score_stage1(scene_id, blobs):
    _, ego = pq_rows(blobs["clipgt/egomotion_estimate.parquet"], "egomotion_estimate")
    P = np.array([[d["location"]["x"], d["location"]["y"]] for d in ego], float)
    Q = np.array([[d["orientation"]["w"], d["orientation"]["x"],
                   d["orientation"]["y"], d["orientation"]["z"]] for d in ego], float)
    w, xq, yq, zq = Q.T
    yaw = np.degrees(np.arctan2(2 * (w * zq + xq * yq), 1 - 2 * (yq * yq + zq * zq)))
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    n = len(P)

    rec = {
        "scene_id": scene_id,
        "n_poses": int(n),
        "path_length_m": round(float(seg.sum()), 2),
        "heading_net_change_deg": round(float(wrap_deg(yaw[-1] - yaw[0])), 2),
        "heading_total_change_deg": round(float(np.sum(np.abs(wrap_deg(np.diff(yaw))))), 2),
        "n_intersection_areas": 0,
        "traversals": [],
    }

    raw_ia = blobs.get("clipgt/intersection_area.parquet")
    if raw_ia is None:
        rec["error"] = "no intersection_area member"
        return rec
    _, ia = pq_rows(raw_ia, "intersection_area")
    polys = []
    for i, d in enumerate(ia or []):
        loc = d.get("location") or []
        if len(loc) < 3 or loc[0].get("x") is None:
            continue
        polys.append((i, d.get("category"), bool(d.get("is_complete")),
                      _xyz(loc)[:, :2]))
    rec["n_intersection_areas"] = len(polys)
    if not polys:
        return rec

    dmins = []
    for i, cat, comp, poly in polys:
        sd = signed_poly_dist(P, poly)
        dmins.append(sd)
        ins = sd < 0
        if not ins.any():
            continue
        idx = np.flatnonzero(ins)
        j0, j1 = int(idx[0]), int(idx[-1])
        rec["traversals"].append({
            "poly_idx": i, "category": cat, "is_complete": comp,
            "pose_span": [j0, j1],
            "n_poses_inside": int(ins.sum()),
            "arc_len_inside_m": round(float(arc[j1] - arc[j0]), 2),
            "heading_change_through_deg": round(float(wrap_deg(yaw[j1] - yaw[j0])), 2),
            "poses_before_entry": j0,
            "poses_after_exit": int(n - 1 - j1),
            "entered_at_frac": round(j0 / max(n - 1, 1), 3),
        })
    dn = np.stack(dmins).min(0)
    rec["dist_to_nearest_intersection_m"] = {
        "min": round(float(dn.min()), 2),
        "median": round(float(np.median(dn)), 2),
        "max": round(float(dn.max()), 2),
        "n_poses_inside": int((dn < 0).sum()),
    }
    # headline ranking signals
    tr = rec["traversals"]
    rec["best_turn_deg"] = (round(max(abs(t["heading_change_through_deg"]) for t in tr), 2)
                            if tr else 0.0)
    rec["has_complete_traversal"] = bool(any(
        t["poses_before_entry"] >= 10 and t["poses_after_exit"] >= 10 for t in tr))
    rec["best_complete_turn_deg"] = (round(max(
        [abs(t["heading_change_through_deg"]) for t in tr
         if t["poses_before_entry"] >= 10 and t["poses_after_exit"] >= 10] or [0.0]), 2))
    return rec


# --------------------------------------------------------------------------
def list_scenes(cache: Path, release=RELEASE):
    if cache.exists():
        return json.loads(cache.read_text())
    out, cursor = [], None
    while True:
        u = f"{API}{release}?limit=1000" + (f"&cursor={cursor}" if cursor else "")
        req = urllib.request.Request(u, headers=_headers())
        with urllib.request.urlopen(req, timeout=120) as r:
            page = json.loads(r.read())
            link = r.headers.get("Link", "")
        out += [x["path"].split("/")[-1] for x in page if x["type"] == "directory"]
        if 'rel="next"' not in link:
            break
        cursor = link.split("cursor=")[1].split(">")[0].split("&")[0]
    out = sorted(set(out))
    cache.write_text(json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/nurec_survey")
    ap.add_argument("--stage", choices=["list", "1", "2"], default="1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--block", type=int, default=1 << 16)
    ap.add_argument("--scenes", default=None, help="comma list, stage 2")
    ap.add_argument("--download-dir", default=None,
                    help="stage 2: also write the pulled members to disk")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scenes = list_scenes(out / "scenes.json")
    print(f"{len(scenes)} scenes in {RELEASE}", flush=True)
    if a.stage == "list":
        return

    if a.stage == "1":
        jl = out / "stage1.jsonl"
        done = set()
        if jl.exists():
            for line in jl.read_text().splitlines():
                try:
                    done.add(json.loads(line)["scene_id"])
                except Exception:
                    pass
        todo = [s for s in scenes[a.offset:] if s not in done]
        if a.limit:
            todo = todo[:a.limit]
        print(f"{len(done)} already done, {len(todo)} to go", flush=True)
        lock = threading.Lock()
        t0 = time.time()
        counter = {"n": 0, "bytes": 0}

        def work(sid):
            try:
                blobs, meta = open_scene_members(sid, STAGE1_MEMBERS, block=a.block)
                rec = score_stage1(sid, blobs)
                rec["usdz_bytes"] = meta["usdz_bytes"]
                rec["fetched_bytes"] = meta["fetched_bytes"]
            except Exception as e:                                   # noqa: BLE001
                rec = {"scene_id": sid, "error": f"{type(e).__name__}: {e}"}
            with lock:
                with jl.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                counter["n"] += 1
                counter["bytes"] += rec.get("fetched_bytes", 0)
                if counter["n"] % 25 == 0:
                    el = time.time() - t0
                    print(f"  {counter['n']}/{len(todo)}  {el:.0f}s  "
                          f"{counter['bytes']/1e6:.1f} MB  "
                          f"{counter['n']/max(el,1e-9)*3600:.0f} scenes/h", flush=True)

        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(work, todo))
        print(f"stage1 done: {counter['n']} scenes, {counter['bytes']/1e6:.1f} MB, "
              f"{time.time()-t0:.0f}s", flush=True)
        return

    # ---- stage 2
    ids = [s.strip() for s in (a.scenes or "").split(",") if s.strip()]
    dd = Path(a.download_dir) if a.download_dir else None
    res = []
    for sid in ids:
        blobs, meta = open_scene_members(sid, STAGE1_MEMBERS + STAGE2_MEMBERS,
                                         block=a.block)
        if dd:
            d = dd / sid
            (d / "clipgt").mkdir(parents=True, exist_ok=True)
            for k, v in blobs.items():
                (d / k).write_bytes(v)
            print(f"{sid}: wrote {len(blobs)} members to {d} "
                  f"({meta['fetched_bytes']/1e6:.2f} MB fetched)", flush=True)
        rec = score_stage1(sid, blobs)
        rec["members_pulled"] = sorted(blobs)
        rec["fetched_bytes"] = meta["fetched_bytes"]
        res.append(rec)
    (out / "stage2.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
