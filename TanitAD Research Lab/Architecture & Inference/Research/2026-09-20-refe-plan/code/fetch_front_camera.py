"""Download a CHOSEN SUBSET of cameras out of nuPlan's sensor-blob zips, over HTTP ranges.

⚠️ THE FILE NAME IS HISTORICAL. This fetched CAM_F0 only until 2026-09-21, when the PI extended
REFe to the paper's FOUR cameras. `--cameras` now takes any comma-separated channel list and the
default is still CAM_F0, so every earlier invocation keeps its exact meaning.

Why this exists (MEASURED 2026-09-20 on nuplan-v1.1_mini_camera_0.zip):
  the zip is 48.63 GB and bundles all 8 cameras; CAM_F0 is 5.95 GB = 12.2% of it, in 30,297 files
  that occupy only **7 contiguous byte-runs** (one per log). Four cameras is ~4x that and still
  well under half the archive, so ranged fetching stays worth its complexity -- a whole-zip pull
  would waste roughly half of 1.83 TB across mini+test+val instead of ~88 %.
  Front-camera-only totals, extrapolated at the measured 12.2%:  mini ~51 GB, test ~73 GB, val ~99 GB.

How: S3 serves `Accept-Ranges: bytes`. Read the ZIP64 End-Of-Central-Directory from the last 128 KB,
range-fetch the ~40 MB central directory, keep the CAM_F0 entries, merge them into contiguous runs,
and fetch each run with one request. Local file headers inside a run are parsed to split the blob.

⚠️ Verified before use, not assumed: entries must be STORED (method 0). JPEGs are already compressed
so MOST are stored -- but MEASURED: CAM_F0 carries BOTH method 0 and method 8, so deflate is
handled and every written file is CRC-verified against the central directory. A blanket 'assume
STORED' would have silently dropped or corrupted part of the set.

Usage:
  python fetch_front_camera.py --list  <zip-url>          # index only, no payload
  python fetch_front_camera.py --probe <zip-url>          # fetch the FIRST run only, verify JPEGs
  python fetch_front_camera.py        <zip-url> <out-dir> # full extraction
  ... --cameras CAM_F0,CAM_L0,CAM_R0,CAM_B0               # the paper's four-camera rig
"""
from __future__ import annotations
import glob, io, json, os, struct, subprocess, sys, zlib

CURL = ["curl", "--ssl-no-revoke", "-s", "--retry", "10", "--retry-all-errors"]


def rng(url: str, a: int, b: int) -> bytes:
    p = subprocess.run(CURL + ["-r", f"{a}-{b}", url], capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"curl exit {p.returncode}: {p.stderr[:200]!r}")
    return p.stdout


def size_of(url: str) -> int:
    p = subprocess.run(CURL + ["-I", url], capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.lower().startswith("content-length"):
            return int(line.split(":")[1].strip())
    raise RuntimeError("no content-length")


CD_CACHE = os.environ.get("REFE_CD_CACHE", "D:/Projects/TanitAD/data/nuplan-camera/_cdcache")


def central_directory(url: str, total: int):
    """Range-fetch the zip's central directory, CACHED on disk.

    Each CD is ~40 MB and a survey over the 9 mini archives costs 360 MB every time. The cache key
    is the url plus the total size, so a changed object invalidates it rather than silently serving
    a stale index."""
    import hashlib
    os.makedirs(CD_CACHE, exist_ok=True)
    key = hashlib.sha256(f"{url}|{total}".encode()).hexdigest()[:32]
    cp = os.path.join(CD_CACHE, key + ".cd")
    if os.path.exists(cp):
        with open(cp, "rb") as f:
            return f.read()
    cd = _central_directory_remote(url, total)
    with open(cp, "wb") as f:
        f.write(cd)
    return cd


def _central_directory_remote(url: str, total: int):
    tail = rng(url, max(0, total - 131072), total - 1)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise RuntimeError("no EOCD")
    cdsz = struct.unpack("<I", tail[i + 12:i + 16])[0]
    cdoff = struct.unpack("<I", tail[i + 16:i + 20])[0]
    j = tail.rfind(b"PK\x06\x06")
    if j >= 0:
        cdsz = struct.unpack("<Q", tail[j + 40:j + 48])[0]
        cdoff = struct.unpack("<Q", tail[j + 48:j + 56])[0]
    return rng(url, cdoff, cdoff + cdsz - 1)


def entries(cd: bytes):
    out, off = [], 0
    while off + 46 <= len(cd) and cd[off:off + 4] == b"PK\x01\x02":
        (_, vm, vn, flg, mth, tm, dt, crc, csz, usz, nlen, elen, clen,
         dsk, iat, eat, lho) = struct.unpack("<IHHHHHHIIIHHHHHII", cd[off:off + 46])
        name = cd[off + 46:off + 46 + nlen].decode("utf-8", "replace")
        extra = cd[off + 46 + nlen:off + 46 + nlen + elen]
        if (usz == 0xFFFFFFFF or csz == 0xFFFFFFFF or lho == 0xFFFFFFFF) and extra:
            p = 0
            while p + 4 <= len(extra):
                hid, hsz = struct.unpack("<HH", extra[p:p + 4]); body = extra[p + 4:p + 4 + hsz]; q = 0
                if hid == 0x0001:
                    if usz == 0xFFFFFFFF and q + 8 <= len(body): usz = struct.unpack("<Q", body[q:q + 8])[0]; q += 8
                    if csz == 0xFFFFFFFF and q + 8 <= len(body): csz = struct.unpack("<Q", body[q:q + 8])[0]; q += 8
                    if lho == 0xFFFFFFFF and q + 8 <= len(body): lho = struct.unpack("<Q", body[q:q + 8])[0]; q += 8
                p += 4 + hsz
        out.append({"name": name, "method": mth, "csz": csz, "usz": usz, "lho": lho, "crc": crc})
        off += 46 + nlen + elen + clen
    return out


def runs(sel, gap=1 << 20):
    sel = sorted(sel, key=lambda e: e["lho"])
    out, cur = [], []
    for e in sel:
        if cur and e["lho"] > cur[-1]["lho"] + cur[-1]["csz"] + 4096 + gap:
            out.append(cur); cur = []
        cur.append(e)
    if cur:
        out.append(cur)
    return out


def write_run_container(blob: bytes, base: int, group, outdir: str) -> int:
    """Write one ZIP PER LOG instead of loose files.

    MEASURED 2026-09-20: D: is exFAT with a 1 MiB allocation unit, so each 204 KB JPEG occupies a
    whole megabyte -- 26,060 frames = 5.08 GB of data but 25 GB ON DISK, a 5.0x inflation. Projected
    over front-camera-only mini+test+val that is 223 GB of data costing 1,158 GB of disk, which does
    NOT fit beside the 481 GB database pull. The loss is per-FILE, so one container per log removes
    it entirely. JPEGs are stored uncompressed here because they are already compressed.
    """
    import zipfile
    # ⭐ ONE CONTAINER PER (LOG, CAMERA), not per log. The frames could safely share a container --
    # nuPlan names each one by a per-image hash, so basenames do not collide across cameras -- but
    # the entries inside a container are stored as bare basenames, which makes the CONTAINER NAME
    # the only place the channel survives. `build_targets.index_images` reads the camera off that
    # name. Merging four cameras into one zip would erase it.
    by_log: dict[tuple[str, str], list] = {}
    for e in group:
        if e["name"].endswith("/"):
            continue
        parts = e["name"].split("/")
        log = next((p for p in parts if p.startswith("20")), "misc")
        cam = next((p for p in parts if p.startswith("CAM_")), "CAM_F0")
        by_log.setdefault((log, cam), []).append(e)
    n = 0
    for (log, cam), entries in by_log.items():
        os.makedirs(outdir, exist_ok=True)
        zp = os.path.join(outdir, f"{log}_{cam}.zip")
        with zipfile.ZipFile(zp, "a", zipfile.ZIP_STORED) as z:
            existing = set(z.namelist())
            for e in entries:
                o = e["lho"] - base
                if o < 0 or o + 30 > len(blob) or blob[o:o + 4] != b"PK":
                    raise RuntimeError(f"local header missing for {e['name']}")
                nlen, elen = struct.unpack("<HH", blob[o + 26:o + 30])
                s = o + 30 + nlen + elen
                data = blob[s:s + e["csz"]]
                if e["method"] == 8:
                    data = zlib.decompress(data, -15)
                elif e["method"] != 0:
                    raise RuntimeError(f"{e['name']} method={e['method']}")
                if zlib.crc32(data) != e["crc"]:
                    raise RuntimeError(f"CRC mismatch for {e['name']}")
                arc = os.path.basename(e["name"])
                if arc not in existing:
                    z.writestr(arc, data)
                n += 1
    return n


def write_run(blob: bytes, base: int, group, outdir: str) -> int:
    n = 0
    for e in group:
        if e["name"].endswith("/"):
            continue
        o = e["lho"] - base
        if o < 0 or o + 30 > len(blob) or blob[o:o + 4] != b"PK\x03\x04":
            raise RuntimeError(f"local header missing for {e['name']}")
        nlen, elen = struct.unpack("<HH", blob[o + 26:o + 30])
        s = o + 30 + nlen + elen
        data = blob[s:s + e["csz"]]
        if e["method"] == 8:
            data = zlib.decompress(data, -15)          # raw deflate, no zlib header
        elif e["method"] != 0:
            raise RuntimeError(f"{e['name']} is method={e['method']} -- refusing to write garbage")
        if zlib.crc32(data) != e["crc"]:
            raise RuntimeError(f"CRC mismatch for {e['name']} -- refusing to write a corrupt image")
        dst = os.path.join(outdir, *e["name"].split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(data)
        n += 1
    return n


def main(argv):
    mode = "full"
    only_logs = None
    if "--only-logs" in argv:
        i = argv.index("--only-logs")
        only_logs = set(argv[i + 1].split(","))
        argv = argv[:i] + argv[i + 2:]
    # ⛔ DERIVE THE LOG LIST FROM THE BANK, NEVER HAND-WRITE IT. MEASURED 2026-09-21: a hand-written
    # six-log list fetched 6 of the **7** logs the target bank actually references, leaving 109 of
    # 1,091 tuples (10.0 %) without side/rear cameras. The missed log was
    # `2021.06.09.14.58.55_veh-35_01095_01484` -- a different SEGMENT of the same recording session
    # as one that WAS in the list (`..._01894_02311`), so the two look near-identical at a glance
    # and the omission survived review. A list that is a copy of a fact drifts from it silently;
    # the bank is the fact.
    if "--logs-from-bank" in argv:
        import json as _json
        i = argv.index("--logs-from-bank")
        bank = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
        paths = ([bank] if bank.endswith(".jsonl")
                 else sorted(p for p in glob.glob(os.path.join(bank, "*.jsonl"))
                             if "stats" not in os.path.basename(p)))
        derived = set()
        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        derived.add(_json.loads(line)["log_name"])
        if not derived:
            print(f"  --logs-from-bank read NO logs from {bank!r} -- refusing to fetch nothing")
            return 4
        print(f"  log list DERIVED from {len(paths)} bank file(s): {len(derived)} logs")
        only_logs = derived if only_logs is None else (only_logs | derived)
    cams = ["CAM_F0"]
    if "--cameras" in argv:
        i = argv.index("--cameras")
        cams = [c.strip() for c in argv[i + 1].split(",") if c.strip()]
        argv = argv[:i] + argv[i + 2:]
        if not cams:
            print("  --cameras was given an empty list"); return 2
    container = "--container" in argv
    if container:
        argv = [a for a in argv if a != "--container"]
    if argv[0] in ("--list", "--probe", "--logs"):
        mode, argv = argv[0][2:], argv[1:]
    url = argv[0]
    out = argv[1] if len(argv) > 1 else None
    total = size_of(url)
    cd = central_directory(url, total)
    es = entries(cd)
    # MEASURED: the archive carries DIRECTORY entries too (name ends "/", zero-length). Opening one
    # as a file raises FileNotFoundError and, behind a pipe, the wrapper still reports exit 0 --
    # the empty output directory is the admissible evidence, not the status code.
    f0 = [e for e in es
          if not e["name"].endswith("/") and any("/" + c + "/" in e["name"] for c in cams)]
    if only_logs is not None:
        f0 = [e for e in f0 if any("/" + lg + "/" in e["name"] for lg in only_logs)]
    # ⛔ A CHANNEL THAT MATCHED NOTHING IS A TYPO, NOT AN EMPTY ARCHIVE. Without this the fetch
    # would report FETCH_OK over zero bytes and the missing camera would surface days later as a
    # dropped-tuple statistic. Per-channel counts are printed for the same reason.
    per_cam = {c: sum(1 for e in f0 if "/" + c + "/" in e["name"]) for c in cams}
    rs = runs(f0)
    gb = sum(e["csz"] for e in f0) / 2**30
    tag = "+".join(cams)
    print(f"  zip {total/2**30:.2f} GB | {len(es):,} entries | {tag} {len(f0):,} files "
          f"{gb:.2f} GB ({100*gb*2**30/total:.1f}%) in {len(rs)} runs")
    print(f"  per channel: " + "  ".join(f"{c}={n:,}" for c, n in per_cam.items()))
    empty = [c for c, n in per_cam.items() if n == 0]
    if empty and only_logs is None and len(empty) == len(cams):
        print(f"  NO_SUCH_CHANNEL {','.join(empty)} -- refusing")
        return 3
    methods = {e["method"] for e in f0}
    print(f"  compression methods present in {tag}: {sorted(methods)}  (0 = STORED)")
    if mode == "logs":
        import collections
        by = collections.Counter()
        for e in f0:
            parts = e["name"].split("/")
            lg = next((p for p in parts if p.startswith("20")), "?")
            by[lg] += e["csz"]
        for lg, v in sorted(by.items()):
            print(f"    {lg:42s} {v/2**30:6.2f} GB")
        return 0
    if mode == "list":
        return 0
    if out is None:
        print("  need an output dir"); return 2
    os.makedirs(out, exist_ok=True)
    todo = rs[:1] if mode == "probe" else rs
    got = 0
    for k, g in enumerate(todo, 1):
        a, b = g[0]["lho"], g[-1]["lho"] + g[-1]["csz"] + 4096
        blob = rng(url, a, min(b, total - 1))
        got += (write_run_container if container else write_run)(blob, a, g, out)
        print(f"  run {k}/{len(todo)}: {len(g):,} files, {(b-a)/2**30:.2f} GB -> {got:,} written")
    print("PROBE_OK" if mode == "probe" else "FETCH_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
