"""Range-fetch ONLY the navtrain nuPlan DBs out of the train city zips -- never the whole zips.

⛔ WHY. navtrain needs 1,192 nuPlan logs. We hold 214 (they live in nuPlan **val**); the other 978
live in nuPlan **train**, which is published ONLY as city zips. MEASURED by HEAD 2026-09-23:

    train_boston 35.5 GiB · train_pittsburgh 28.5 · train_singapore 32.6 · train_vegas_1..6 850.8
    => ~947 GiB for all of train

The pod plan had carried "978 DBs ~ 68.5 GiB" -- which silently assumed per-file download. There
is no per-file download. What there IS: these are REAL ZIPs (magic `PK\\x03\\x04`, Zip64 EOCD) with a
tiny central directory (train_boston: **0.23 MiB for 1,649 entries**), every `.db` a separately
addressable DEFLATE member. So each wanted DB can be range-fetched and inflated on its own.
⚠️ Checked, not assumed: nuPlan's CAMERA "zips" are TARs with no central directory (R14). These are
not -- the same filename extension meant opposite things one directory apart.

Usage:
  python fetch_navtrain_dbs.py --plan                 # read 9 central directories only; exact sizes
  python fetch_navtrain_dbs.py --out <dir>            # fetch + inflate + CRC-verify, resumable
  python fetch_navtrain_dbs.py --out <dir> --limit 1  # one DB, to prove the path end to end
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "refe"))
import fetch_front_camera as F                     # rng / size_of / central_directory / entries

BASE = "https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1"
# ⛔ val IS INCLUDED. navtrain draws on nuPlan train AND val: the 214 logs the dev box holds came
# from `val.zip`. The first version searched train only, because it subtracted what the DEV BOX
# held -- simulated as an empty pod it reported "214 wanted logs are in NO train zip" and would
# have fetched only 978 of 1,192.
ZIPS = ["train_boston", "train_pittsburgh", "train_singapore"] + \
       [f"train_vegas_{i}" for i in range(1, 7)] + ["val"]
GiB = 1024 ** 3


def wanted_logs(held: set[str]) -> set[str]:
    import navtrain_scenarios as NS
    logs, _ = NS.load_split()
    return set(logs) - held


def plan(want: set[str]):
    """{log: (url, entry)} for every wanted log found in a train/val zip, plus per-zip totals."""
    found, per_zip = {}, {}
    for z in ZIPS:
        url = f"{BASE}/nuplan-v1.1_{z}.zip"
        total = F.size_of(url)
        cd = F.central_directory(url, total)
        n = csz = usz = 0
        for e in F.entries(cd):
            if not e["name"].endswith(".db"):
                continue
            log = os.path.basename(e["name"])[:-3]
            if log in want and log not in found:
                found[log] = (url, e)
                n += 1; csz += e["csz"]; usz += e["usz"]
        per_zip[z] = (n, csz, usz, total)
    return found, per_zip


def fetch_one(url: str, e: dict, out_path: str) -> int:
    """Range-fetch one DEFLATE member, inflate, verify size AND crc32, write atomically."""
    lh = F.rng(url, e["lho"], e["lho"] + 29)
    if lh[:4] != b"PK\x03\x04":
        raise RuntimeError(f"no local header at {e['lho']}")
    nlen, elen = struct.unpack("<HH", lh[26:30])
    start = e["lho"] + 30 + nlen + elen
    comp = F.rng(url, start, start + e["csz"] - 1)
    if len(comp) != e["csz"]:
        raise RuntimeError(f"short read {len(comp)} of {e['csz']}")
    if e["method"] == 8:
        data = zlib.decompress(comp, -15)
    elif e["method"] == 0:
        data = comp
    else:
        raise RuntimeError(f"unsupported method {e['method']}")
    # ⛔ Size AND CRC. A truncated inflate can still produce plausible SQLite pages; only the CRC
    # proves the bytes are the ones the archive holds.
    if len(data) != e["usz"]:
        raise RuntimeError(f"inflated {len(data)} != usz {e['usz']}")
    if (zlib.crc32(data) & 0xFFFFFFFF) != e["crc"]:
        raise RuntimeError("CRC32 mismatch")
    tmp = out_path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, out_path)
    return len(comp)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--logs-file", default=None,
                    help="fetch only these logs (a machine's share of a split data prep)")
    ap.add_argument("--jobs", type=int, default=8,
                    help="parallel member fetches. MEASURED 2026-09-23 on the first REFe pod: ONE "
                         "stream from the Tokyo-hosted nuPlan bucket ran at 17.3 MB/s, i.e. ~1.7 h "
                         "for the 97.6 GiB; S3 scales with connections, not with one stream")
    ap.add_argument("--held-root", default="D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits")
    a = ap.parse_args(argv)

    import navtrain_scenarios as NS
    held = set(NS.index_dbs(a.held_root)) if os.path.isdir(a.held_root) else set()
    want = wanted_logs(held)
    if a.logs_file:
        only = {l.strip() for l in open(a.logs_file, encoding="utf-8") if l.strip()}
        want = want & only
        print(f"  --logs-file: restricted to {len(want)} logs")
    print(f"  navtrain logs wanted (not held locally): {len(want)}")
    found, per_zip = plan(want)
    tc = sum(v[1] for v in per_zip.values()); tu = sum(v[2] for v in per_zip.values())
    tz = sum(v[3] for v in per_zip.values())
    for z, (n, c, u, t) in per_zip.items():
        print(f"    {z:18s} zip {t/GiB:6.1f} GiB | wanted DBs {n:4d} | "
              f"fetch {c/GiB:6.2f} GiB -> {u/GiB:6.2f} GiB on disk")
    print(f"  FOUND {len(found)} / {len(want)} wanted logs in the train + val zips")
    print(f"  TO FETCH: {tc/GiB:.2f} GiB compressed -> {tu/GiB:.2f} GiB inflated "
          f"(vs {tz/GiB:.1f} GiB for the whole zips = {tz/max(tc,1):.1f}x more)")
    missing = sorted(want - set(found))
    if missing:
        print(f"  ⚠️ {len(missing)} wanted logs are in NO train zip: {missing[:3]}")
    if a.plan:
        return 0
    if not a.out:
        print("  --out required unless --plan"); return 2
    os.makedirs(a.out, exist_ok=True)
    todo = [(lg, u, e) for lg, (u, e) in sorted(found.items())
            if not (os.path.exists(os.path.join(a.out, lg + ".db"))
                    and os.path.getsize(os.path.join(a.out, lg + ".db")) == e["usz"])]
    if a.limit:
        todo = todo[:a.limit]
    print(f"  fetching {len(todo)} (resume skips complete files)")
    import concurrent.futures as cf

    def job(item):
        lg, u, e = item
        err = None
        for attempt in range(3):              # a transient S3 error must not cost the whole stage
            try:
                return lg, fetch_one(u, e, os.path.join(a.out, lg + ".db")), None
            except Exception as ex:           # noqa: BLE001 -- recorded and re-raised as a count
                err = f"{type(ex).__name__}: {str(ex)[:100]}"
                time.sleep(2 * (attempt + 1))
        return lg, 0, err

    t0, got, k, bad = time.time(), 0, 0, []
    with cf.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        futs = [pool.submit(job, it) for it in todo]
        for fut in cf.as_completed(futs):
            lg, nb, err = fut.result()
            k += 1
            got += nb
            if err:
                bad.append((lg, err))
            if k % 10 == 0 or k == len(todo):
                dt = time.time() - t0
                print(f"    {k}/{len(todo)}  {got/GiB:.2f} GiB  {got/max(dt,1e-6)/1e6:.1f} MB/s  "
                      f"failed {len(bad)}  ({a.jobs} parallel)", flush=True)
    if bad:
        # ⛔ every member is size- AND crc-checked inside fetch_one; a failure is reported, never
        # papered over -- a re-run resumes, skipping the complete files
        print(f"  FAILED {len(bad)} after 3 attempts each, e.g. {bad[:3]}")
        return 1
    print("NAVTRAIN_DBS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
