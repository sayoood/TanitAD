"""S4 verifier. Run with `python -I` (it reads untrusted downloaded bytes; PIL needed -> the tanitad venv).

verify_jpg(data, exp_size, exp_crc32=None) checks, per jpg:
  size_ok   len(data) == the index's size
  crc_ok    zlib.crc32(data) == the index's CRC32 (None when the index has none, e.g. a tar member: then
            crc_ok is None and counted as NOT verified, never as a pass)
  soi_ok    starts with FF D8
  eoi_ok    ends with FF D9 (trailing NULs tolerated; JPEG streams from cameras may be padded)
  decode_ok PIL opens and fully decodes it (Image.load) with width/height > 0
  sha256    of the bytes
  ok        every check True; with require_crc (the default) a missing reference CRC32 FAILS closed

CLI:  verify_s4.py --index raw/s2_index.json --frames-dir C:/Users/Admin/nuplan_u/frames --out raw/s4_manifest.json
      [--allow-subset]   (only verify containers present; for a RED scratch dir)
Containers are `{log}_{cam}.zip` with bare-basename stored members, as fetch_front_camera.py makes them.
Exit 0 only if every indexed jpg is present and verifies; a manifest is written either way.
"""
import argparse, hashlib, io, json, os, sys, zipfile, zlib


def verify_jpg(data: bytes, exp_size, exp_crc32=None, require_crc=True) -> dict:
    r = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    r["size_ok"] = (exp_size is None) or (len(data) == exp_size)
    r["crc_ok"] = None if exp_crc32 is None else (zlib.crc32(data) & 0xFFFFFFFF) == exp_crc32
    r["soi_ok"] = data[:2] == b"\xff\xd8"
    tail = data.rstrip(b"\x00")
    r["eoi_ok"] = tail[-2:] == b"\xff\xd9"
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(data))
        im.load()
        r["decode_ok"] = im.width > 0 and im.height > 0
        r["wh"] = [im.width, im.height]
    except Exception as e:  # noqa: BLE001
        r["decode_ok"] = False
        r["decode_err"] = f"{type(e).__name__}: {str(e)[:120]}"
    checks = [r["size_ok"], r["soi_ok"], r["eoi_ok"], r["decode_ok"]]
    if exp_crc32 is not None:
        checks.append(r["crc_ok"])
    elif require_crc:
        checks.append(False)   # FAIL CLOSED: no reference CRC32 means a decodable bit-flip cannot be caught
    r["ok"] = all(checks)
    r["crc_checked"] = exp_crc32 is not None
    return r


def run(index_path, frames_dir, out_path, allow_subset=False, allow_no_crc=False):
    idx = json.load(open(index_path, encoding="utf-8"))
    ents = idx["entries"]
    rows, n_missing = [], 0
    zcache = {}
    for e in ents:
        log, cam, base = e["path"].split("/")
        zp = os.path.join(frames_dir, f"{log}_{cam}.zip")
        if not os.path.exists(zp):
            if allow_subset:
                continue
            rows.append({"path": e["path"], "present": False, "ok": False}); n_missing += 1
            continue
        z = zcache.get(zp) or zcache.setdefault(zp, zipfile.ZipFile(zp))
        if base not in z.namelist():
            if allow_subset:
                continue
            rows.append({"path": e["path"], "present": False, "ok": False}); n_missing += 1
            continue
        data = z.read(base)
        v = verify_jpg(data, e.get("usz"), e.get("crc32"), require_crc=not allow_no_crc)
        rows.append({"path": e["path"], "present": True, **v})
    n_ok = sum(1 for r in rows if r["ok"])
    man = {"index": os.path.abspath(index_path), "frames_dir": frames_dir,
           "n_indexed": len(ents), "n_rows": len(rows), "n_present": sum(1 for r in rows if r["present"]),
           "n_verified_ok": n_ok, "n_failed": len(rows) - n_ok, "n_missing": n_missing,
           "total_bytes": sum(r.get("bytes", 0) for r in rows), "rows": rows}
    json.dump(man, open(out_path, "w", encoding="utf-8"), indent=1)
    print(f"indexed {len(ents)} | present {man['n_present']} | verified ok {n_ok} | failed {man['n_failed']} | missing {n_missing}")
    return man


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True); ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--allow-subset", action="store_true")
    ap.add_argument("--allow-no-crc", action="store_true", help="index has no CRC32 (tar source); weakens the check, say so in the report")
    a = ap.parse_args()
    m = run(a.index, a.frames_dir, a.out, a.allow_subset, a.allow_no_crc)
    if m["n_indexed"] == 0:
        print("NOTHING_TO_VERIFY: the index has 0 entries; this is NOT a pass"); sys.exit(3)
    sys.exit(0 if m["n_failed"] == 0 and m["n_missing"] == 0 and m["n_rows"] > 0 else 1)
