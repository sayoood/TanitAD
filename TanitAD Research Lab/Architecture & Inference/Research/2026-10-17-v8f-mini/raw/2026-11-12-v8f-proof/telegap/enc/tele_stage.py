#!/usr/bin/env python3
"""Stage the FRONT TELE clips of a clip list by range reads out of the source dataset's chunk archives (card ADD122 section 5, t1).
Pod only; the HF token is read in process from /root/.hf_token and is never printed or put in an argument, a log line or an error text.

Source: HF dataset nvidia/PhysicalAI-Autonomous-Vehicles. clip_index.parquet gives the chunk of a clip; the tele clips of a chunk are in
camera/camera_front_tele_30fov/camera_front_tele_30fov.chunk_%04d.zip (stored, about 1.1 GB, 300 members). The archive is NEVER downloaded
whole: one HfFileSystem file object per chunk is opened, the zip directory is read and only the clip's members are read (zipfile checks the
CRC of a member on read). Clips are grouped by chunk, one archive open per chunk, a few worker threads over chunks.

Layout under --root (the data kit's Layout, so enc_common.Layout / the stack's helpers work on it unchanged):
  r0/camera/<clip>.mp4                                  the tele video
  ts/<clip>.timestamps.parquet                          the tele timestamps (int64 microseconds, column 'timestamp')
  calibration/physicalai_front_wide_intrinsics.csv      per-clip table of the TELE camera (the file NAME is the one the stack's
                                                        intrinsics_for_clip reads; its content is camera_front_tele_30fov: clip_id,
                                                        width,height,cx,cy,bw_poly_0..4,fw_poly_0..4)
  calibration/front_tele_extrinsics.csv                 clip_id,qx,qy,qz,qw,x,y,z of camera_front_tele_30fov
  ledger.jsonl                                          one line per clip and attempt
  records/stage_<time_ns>.json                            the record of a run: per clip chunk, bytes, width, height, n_frames, crc_ok, status
The blurred_boxes member is not read. Resume: a clip with an ok ledger row, both files at the ledger's size and a calibration row is skipped.
Nothing is deleted; a file is written whole under a .part name and renamed once.
"""
import argparse
import contextlib
import json
import os
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.environ.get("TG_ADD99_KIT", ""), "/workspace/v8proof/cropread/kit/enc", os.path.join(HERE, "..", "..", "cropread", "enc")):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
from enc_common import Layout, read_ids   # noqa: E402

REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
TELE = "camera_front_tele_30fov"
CHUNK_ZIP = f"camera/{TELE}/{TELE}.chunk_{{:04d}}.zip"
INTR_PQ = "calibration/camera_intrinsics/camera_intrinsics.chunk_{:04d}.parquet"
EXTR_PQ = "calibration/sensor_extrinsics/sensor_extrinsics.chunk_{:04d}.parquet"
SUFFIX = {"mp4": f"{TELE}.mp4", "ts": f"{TELE}.timestamps.parquet"}
INTR_COLS = ["clip_id", "width", "height", "cx", "cy"] + [f"bw_poly_{i}" for i in range(5)] + [f"fw_poly_{i}" for i in range(5)]
EXTR_COLS = ["clip_id", "qx", "qy", "qz", "qw", "x", "y", "z"]
_TOK = None
_LOCK = threading.Lock()


def tok():
    global _TOK
    if _TOK is None:
        _TOK = open(os.environ.get("TG_TOKEN_FILE", "/root/.hf_token"), encoding="utf-8").read().strip()
    return _TOK


def scrub(msg):
    """An error text with the token removed (a library may echo a header)."""
    s = str(msg)
    try:
        t = tok()
        if t:
            s = s.replace(t, "***")
    except Exception:                                                # noqa: BLE001
        pass
    return s


def log(m):
    print(f"[tele-stage {time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------------------------------------ source access (patched in tests)
def hf_dl(path, local_dir):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, path, repo_type="dataset", token=tok(), local_dir=local_dir)


@contextlib.contextmanager
def open_chunk_zip(chunk):
    """One open archive of one chunk by range reads: yields a zipfile.ZipFile on an HfFileSystem file object."""
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(token=tok())
    with fs.open(f"datasets/{REPO}/{CHUNK_ZIP.format(int(chunk))}", "rb", block_size=2 ** 22) as f:
        yield zipfile.ZipFile(f)


def chunk_index(hf_dir):
    """clip_index.parquet -> DataFrame indexed by clip_id (columns include chunk)."""
    import pandas as pd
    df = pd.read_parquet(hf_dl("clip_index.parquet", hf_dir))
    if "clip_id" in df.columns:
        df = df.set_index("clip_id")
    return df


def probe_mp4(path):
    """-> (width, height) of the first video stream."""
    import av
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        return int(s.width), int(s.height)


# ------------------------------------------------------------------------------------------------ pure helpers
def group_by_chunk(clips, index_df):
    """-> ({chunk: [clips in the order given]}, [clips with no row in the index])."""
    groups, missing = {}, []
    for c in clips:
        if c in index_df.index:
            groups.setdefault(int(index_df.loc[c, "chunk"]), []).append(c)
        else:
            missing.append(c)
    return groups, missing


def select_members(names, clip):
    """The mp4 and the timestamps member of `clip` among the archive's member names: names that contain the clip id and end in the tele
    suffix. Exactly one of each, else ValueError (a missing member and an ambiguous one are both refused)."""
    out = {}
    for kind, suf in SUFFIX.items():
        hit = [n for n in names if clip in n and n.endswith(suf)]
        if len(hit) != 1:
            raise ValueError(f"{len(hit)} members for {kind} (need exactly 1)")
        out[kind] = hit[0]
    return out


def _flat(df):
    import pandas as pd
    if isinstance(df.index, pd.MultiIndex) or df.index.name is not None:
        df = df.reset_index()
    return df


def calib_rows(intr_df, ext_df, clips):
    """Rows of the tele camera for `clips` from the chunk's two calibration tables -> (intr list of dict, ext list of dict).
    A clip with no tele row is simply absent from the result."""
    a = _flat(intr_df)
    a = a[(a["camera_name"].astype(str) == TELE) & (a["clip_id"].isin(set(clips)))]
    b = _flat(ext_df)
    b = b[(b["sensor_name"].astype(str) == TELE) & (b["clip_id"].isin(set(clips)))]
    miss = [c for c in INTR_COLS if c not in a.columns] + [c for c in EXTR_COLS if c not in b.columns]
    if miss:
        raise KeyError(f"calibration columns missing: {miss}")
    return a[INTR_COLS].to_dict("records"), b[EXTR_COLS].to_dict("records")


def read_ledger_ok(path):
    ok = {}
    if os.path.exists(path):
        for ln in open(path, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                try:
                    d = json.loads(ln)
                    if d.get("status") == "ok":
                        ok[d["clip"]] = d
                except Exception:                                    # noqa: BLE001
                    pass
    return ok


def _csv_clips(path):
    import pandas as pd
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, dtype={"clip_id": str})
    return {r["clip_id"]: r for r in df.to_dict("records")}


def _write_csv(path, rows_by_clip, cols):
    import pandas as pd
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    pd.DataFrame([rows_by_clip[c] for c in sorted(rows_by_clip)], columns=cols).to_csv(tmp, index=False)
    os.replace(tmp, path)


def is_complete(L, c, ledger_ok, intr_have):
    r = ledger_ok.get(c)
    return bool(r and c in intr_have and os.path.exists(L.mp4(c)) and os.path.exists(L.ts(c))
                and os.path.getsize(L.mp4(c)) == r.get("bytes"))


# ------------------------------------------------------------------------------------------------ staging
def write_whole(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def stage_chunk(chunk, clips, L, hf_dir, tries=3):
    """Stage the clips of one chunk with one open archive. -> (rows [ledger rows], intr rows, ext rows)."""
    rows = []
    last = None
    for k in range(tries):
        try:
            with open_chunk_zip(chunk) as z:
                names = z.namelist()
                for c in clips:
                    rows.append(_stage_clip(z, names, c, chunk, L))
            break
        except Exception as e:                                       # noqa: BLE001
            last = f"archive {type(e).__name__}: {scrub(e)[:160]}"
            rows = [r for r in rows if r["status"] == "ok"]
            clips = [c for c in clips if c not in {r["clip"] for r in rows}]
            time.sleep(0 if os.environ.get("TG_NO_SLEEP") else 2 * (k + 1))
    else:
        for c in clips:
            rows.append({"clip": c, "chunk": chunk, "bytes": 0, "width": None, "height": None, "n_frames": None, "crc_ok": False,
                         "status": f"fail: {last}", "t": time.strftime("%F %T")})
    intr, ext = [], []
    ok = [r["clip"] for r in rows if r["status"] == "ok"]
    if ok:
        try:
            import pandas as pd
            intr, ext = calib_rows(pd.read_parquet(hf_dl(INTR_PQ.format(chunk), hf_dir)),
                                   pd.read_parquet(hf_dl(EXTR_PQ.format(chunk), hf_dir)), ok)
            have = {r["clip_id"] for r in intr} & {r["clip_id"] for r in ext}
            for r in rows:
                if r["status"] == "ok" and r["clip"] not in have:
                    r["status"] = "fail: no tele calibration row"
        except Exception as e:                                       # noqa: BLE001
            for r in rows:
                if r["status"] == "ok":
                    r["status"] = f"fail: calibration {type(e).__name__}: {scrub(e)[:120]}"
    return rows, intr, ext


def _stage_clip(z, names, c, chunk, L):
    import pandas as pd
    row = {"clip": c, "chunk": chunk, "bytes": 0, "width": None, "height": None, "n_frames": None, "crc_ok": False,
           "status": "fail", "t": time.strftime("%F %T")}
    try:
        mem = select_members(names, c)
    except ValueError as e:
        row["status"] = f"fail: {e}"
        return row
    try:
        mp4 = z.read(mem["mp4"])
        ts = z.read(mem["ts"])
        row["crc_ok"] = True
    except (zipfile.BadZipFile, OSError, EOFError) as e:
        row["status"] = f"fail: read {type(e).__name__}: {scrub(e)[:120]}"
        return row
    write_whole(L.mp4(c), mp4)
    write_whole(L.ts(c), ts)
    row["bytes"] = os.path.getsize(L.mp4(c))
    try:
        row["n_frames"] = int(len(pd.read_parquet(L.ts(c))))
        row["width"], row["height"] = probe_mp4(L.mp4(c))
    except Exception as e:                                           # noqa: BLE001
        row["status"] = f"fail: probe {type(e).__name__}: {scrub(e)[:120]}"
        return row
    row["status"] = "ok"
    return row


def run(clips, root, hf_dir, workers=3, min_free_gb=30.0):
    """-> rc. Stages `clips` (a list of ids) under root; writes the ledger, the calibration tables and records/stage_<time_ns>.json."""
    import shutil
    L = Layout(root)
    os.makedirs(L.root, exist_ok=True)
    os.makedirs(hf_dir, exist_ok=True)
    free = shutil.disk_usage(root).free / 1e9
    if free < min_free_gb:
        log(f"STOP: disk free {free:.1f} GB < {min_free_gb}")
        return 7
    t0 = time.time()
    intr_have = _csv_clips(L.intr_csv)
    ext_have = _csv_clips(f"{L.root}/calibration/front_tele_extrinsics.csv")
    ledger_ok = read_ledger_ok(L.ledger)
    skipped = [c for c in clips if is_complete(L, c, ledger_ok, intr_have) and c in ext_have]
    todo = [c for c in clips if c not in set(skipped)]
    log(f"{len(clips)} clips, {len(skipped)} already staged, {len(todo)} to do")
    result_rows = [dict(ledger_ok[c], status="skipped_complete") for c in skipped]
    index = chunk_index(hf_dir)
    groups, missing = group_by_chunk(todo, index)
    for c in missing:
        result_rows.append({"clip": c, "chunk": None, "bytes": 0, "width": None, "height": None, "n_frames": None, "crc_ok": False,
                            "status": "fail: clip not in clip_index", "t": time.strftime("%F %T")})
    n_bad, n_ok = len(missing), 0

    def one(item):
        return item[0], stage_chunk(item[0], item[1], L, hf_dir)

    with open(L.ledger, "a", buffering=1, encoding="utf-8") as lf, ThreadPoolExecutor(max_workers=workers) as ex:
        for chunk, (rows, intr, ext) in ex.map(one, sorted(groups.items())):
            with _LOCK:
                for r in intr:
                    intr_have[r["clip_id"]] = r
                for r in ext:
                    ext_have[r["clip_id"]] = r
                if intr:
                    _write_csv(L.intr_csv, intr_have, INTR_COLS)
                    _write_csv(f"{L.root}/calibration/front_tele_extrinsics.csv", ext_have, EXTR_COLS)
                for r in rows:
                    lf.write(json.dumps(r) + "\n")
                    result_rows.append(r)
                    if r["status"] == "ok":
                        n_ok += 1
                    else:
                        n_bad += 1
                        log(f"NOT STAGED {r['clip'][:8]} chunk {chunk}: {r['status'][:120]}")
            if (n_ok + n_bad) % 25 == 0:
                log(f"progress ok={n_ok} bad={n_bad} {(time.time() - t0) / max(1, n_ok + n_bad):.1f} s/clip wall")
    rec = {"started": time.strftime("%FT%T", time.gmtime(t0)) + "Z", "finished": time.strftime("%FT%TZ", time.gmtime()), "root": root,
           "n_clips": len(clips), "n_skipped_complete": len(skipped), "n_ok": n_ok, "n_failed": n_bad, "workers": workers,
           "clips": result_rows}
    os.makedirs(f"{L.root}/records", exist_ok=True)
    with open(f"{L.root}/records/stage_{time.time_ns()}.json", "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    log(f"done: ok={n_ok} not_staged={n_bad} skipped={len(skipped)} wall={time.time() - t0:.0f}s")
    print("ZZ TELE STAGE DONE ZZ", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", required=True, help="file of clip ids (first token of each line)")
    ap.add_argument("--root", required=True, help="NEW tele stage root")
    ap.add_argument("--hf-dir", default="/workspace/v8proof/cropread/hfdl")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--min-free-gb", type=float, default=30.0)
    a = ap.parse_args(argv)
    clips = read_ids(a.list)
    if a.first:
        clips = clips[:a.first]
    return run(clips, a.root, a.hf_dir, a.workers, a.min_free_gb)


if __name__ == "__main__":
    sys.exit(main())
