"""l1c_fetch.py -- fetch ONLY the selected clips' members from the dataset's chunk zips by HTTP range reads (no chunk zip is downloaded).

For each candidate (in the sha256 order l1c_select.py wrote): camera mp4 + timestamps (camera_front_wide_120fov chunk zip), the egomotion parquet (labels/egomotion
chunk zip), the chunk's sensor_extrinsics parquet.  Layout written under F:/l1c/pai_root = exactly what v2_compressed.py build reuses locally:
   r0/camera_front_wide/<cid>.camera_front_wide_120fov.{mp4,timestamps.parquet}   labels/egomotion_flat/<cid>.parquet   calibration/{camera_intrinsics,sensor_extrinsics}/
Zip members are STORED (compress_type 0, measured), CRC verified by zipfile on read.  Records r0 gate status per clip (A1.1: recorded, not filtered).
Raw ids stay on F:; the sha12-only manifest goes to refcv9_l1c/select/.
    python l1c_fetch.py --first 0 --count 330 --threads 6
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hfenv  # noqa: E402

REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
ROOT = Path("F:/l1c/pai_root")
SEL = Path("F:/l1c/select")
PUB = HERE.parent / "select"
CAM = "camera_front_wide_120fov"
TREE = "C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58"


def load_r0():
    spec = importlib.util.spec_from_file_location("r0mod", TREE + "/stack/scripts/physicalai_r0.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def member_bytes(fs, path, name_suffix, retries=4):
    last = None
    for k in range(retries):
        try:
            with fs.open(path, "rb", block_size=8 * 1024 * 1024) as f, zipfile.ZipFile(f) as z:
                nm = [i for i in z.infolist() if i.filename.endswith(name_suffix)]
                if len(nm) != 1:
                    raise RuntimeError(f"{len(nm)} members end with {name_suffix} in {path.split('/')[-1]}")
                if nm[0].compress_type != 0:
                    raise RuntimeError("member is not STORED")
                with z.open(nm[0]) as m:
                    return m.read()
        except Exception as e:      # noqa: BLE001
            last = e
            time.sleep(2 + 3 * k)
    raise RuntimeError(f"{type(last).__name__}: {str(last)[:160]}")


def fetch_one(c: dict) -> dict:
    from huggingface_hub import HfFileSystem, hf_hub_download
    fs = HfFileSystem()
    cid, ch = c["clip_id"], int(c["chunk"])
    t0 = time.time()
    rec = {"sha12": c["sha12"], "chunk": ch, "rig": c["rig"]}
    camdir = ROOT / "r0" / "camera_front_wide"
    egodir = ROOT / "labels" / "egomotion_flat"
    camdir.mkdir(parents=True, exist_ok=True)
    egodir.mkdir(parents=True, exist_ok=True)
    mp4p = camdir / f"{cid}.{CAM}.mp4"
    tsp = camdir / f"{cid}.{CAM}.timestamps.parquet"
    egp = egodir / f"{cid}.parquet"
    try:
        zc = f"datasets/{REPO}/camera/{CAM}/{CAM}.chunk_{ch:04d}.zip"
        ze = f"datasets/{REPO}/labels/egomotion/egomotion.chunk_{ch:04d}.zip"
        if not (mp4p.exists() and mp4p.stat().st_size > 0):
            b = member_bytes(fs, zc, f"{cid}.{CAM}.mp4")
            tmp = mp4p.with_suffix(".tmp")
            tmp.write_bytes(b)
            os.replace(tmp, mp4p)
        rec["mp4_bytes"] = mp4p.stat().st_size
        rec["mp4_sha256"] = hashlib.sha256(mp4p.read_bytes()).hexdigest()
        if not (tsp.exists() and tsp.stat().st_size > 0):
            tsp.write_bytes(member_bytes(fs, zc, f"{cid}.{CAM}.timestamps.parquet"))
        if not (egp.exists() and egp.stat().st_size > 0):
            egp.write_bytes(member_bytes(fs, ze, f"{cid}.egomotion.parquet"))
        rec["ego_bytes"] = egp.stat().st_size
        hf_hub_download(REPO, f"calibration/sensor_extrinsics/sensor_extrinsics.chunk_{ch:04d}.parquet", repo_type="dataset", local_dir=str(ROOT))
        ego = pd.read_parquet(egp)
        s = load_r0()._urban_score_from_egomotion(ego)
        rec.update({"r0_mean_speed": s["mean_speed"], "r0_stop_frac": s["stop_frac"], "r0_distance": s["distance"], "r0_pass": bool(s["urban_score"] > 0),
                    "ego_rows": int(len(ego)), "ego_span_s": float((ego["timestamp"].iloc[-1] - ego["timestamp"].iloc[0]) / 1e6)})
        ts = pd.read_parquet(tsp)
        rec["n_cam_frames"] = int(len(ts))
        rec["ok"] = True
    except Exception as e:      # noqa: BLE001
        rec["ok"] = False
        rec["why"] = f"{type(e).__name__}: {str(e)[:200]}"
    rec["secs"] = round(time.time() - t0, 1)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--count", type=int, default=330)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    hfenv.init()
    cands = [json.loads(l) for l in open(SEL / "candidates.jsonl", encoding="utf-8")][a.first:a.first + a.count]
    out = SEL / "fetch_manifest.jsonl"
    done = {}
    if out.exists():
        for l in open(out, encoding="utf-8"):
            r = json.loads(l)
            if r.get("ok"):
                done[r["sha12"]] = r
    todo = [c for c in cands if c["sha12"] not in done]
    print(f"[fetch] {len(cands)} in range, {len(done)} already fetched, {len(todo)} to go", flush=True)
    t0 = time.time()
    nb = 0
    with ThreadPoolExecutor(a.threads) as ex, open(out, "a", encoding="utf-8") as fo:
        futs = {ex.submit(fetch_one, c): c for c in todo}
        for i, fu in enumerate(as_completed(futs)):
            r = fu.result()
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            nb += r.get("mp4_bytes", 0) + r.get("ego_bytes", 0)
            if (i + 1) % 10 == 0 or not r["ok"]:
                print(f"[fetch] {i+1}/{len(todo)} {r['sha12']} ok={r['ok']} {r.get('why','')} {r['secs']}s  total {nb/1e6:.0f} MB {time.time()-t0:.0f}s", flush=True)
    allr = {}
    for l in open(out, encoding="utf-8"):
        r = json.loads(l)
        allr[r["sha12"]] = r if r.get("ok") or r["sha12"] not in allr else allr[r["sha12"]]
    ok = [r for r in allr.values() if r.get("ok")]
    (PUB / "fetch_manifest_sha12.json").write_text(json.dumps({"n_fetched_ok": len(ok), "n_failed": sum(1 for r in allr.values() if not r.get("ok")),
                                                              "rows": sorted(allr.values(), key=lambda r: r["sha12"])}, indent=1), encoding="utf-8")
    print(f"[fetch] done ok={len(ok)} failed={len(allr)-len(ok)} {time.time()-t0:.0f}s  ZZL1CFETCHDONEZZ", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
