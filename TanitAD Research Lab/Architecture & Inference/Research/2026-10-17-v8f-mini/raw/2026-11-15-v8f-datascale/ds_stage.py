#!/usr/bin/env python3
"""DATASCALE pod-side STAGE: NVIDIA PhysicalAI-AV -> the staging root layout the production builder reads, by RANGE READS of single zip members
(one clip costs its own bytes, not a 1.5 GB chunk). Resumable (stage ledger, last line per clip wins), threaded over chunks.
Layout written (same as v2ep_pod.py / pod_build_b1_epcache._ctx):
  <ROOT>/r0/camera_front_wide/<clip>.mp4 , <clip>.timestamps.parquet
  <ROOT>/labels/egomotion/egomotion_all.zip      (STORED, members <clip>.egomotion.parquet)
  <ROOT>/calibration/physicalai_front_wide_intrinsics.csv
  <ROOT>/obsroot/labels/obstacle.offline/obstacle.offline.chunk_<cccc>.zip   (mini zips: only the selected members)
No address, port or pod id in this file; the HF token is read in place and never printed.
"""
import argparse
import hashlib
import io
import json
import os
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["HF_HUB_DISABLE_XET"] = "1"
import pandas as pd

R = "nvidia/PhysicalAI-Autonomous-Vehicles"
BASE = "/root/datascale"
ROOT = f"{BASE}/stage"
META = f"{BASE}/meta"
LEDGER = f"{BASE}/ledger_stage.jsonl"
LOCK = threading.Lock()
CAMP = "camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_{c:04d}.zip"
EGOP = "labels/egomotion/egomotion.chunk_{c:04d}.zip"
OBSP = "labels/obstacle.offline/obstacle.offline.chunk_{c:04d}.zip"
INTP = "calibration/camera_intrinsics/camera_intrinsics.chunk_{c:04d}.parquet"
ICOLS = ["cx", "cy", "width", "height"] + [f"fw_poly_{k}" for k in range(5)]


def log(m):
    print(f"[stage {time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def fs():
    from huggingface_hub import HfFileSystem
    return HfFileSystem(token=open("/root/.hf_token").read().strip())


def member_bytes(f, z, suffix):
    names = [n for n in z.namelist() if n.endswith(suffix)]
    if len(names) != 1:
        raise RuntimeError(f"{len(names)} members end with {suffix}")
    return z.read(names[0])


def do_chunk(chunk, clips, has_obs):
    """Stage every clip of one chunk. Returns a list of ledger rows. Never raises."""
    rows = []
    last = None
    for attempt in range(3):
        try:
            t0 = time.time()
            F = fs()
            out = {c: {"clip": c, "chunk": int(chunk)} for c in clips}
            # camera zip: one open (directory ~2 s), then one range read per member
            with F.open(f"datasets/{R}/" + CAMP.format(c=chunk), "rb") as f:
                z = zipfile.ZipFile(f)
                nm = set(z.namelist())
                for c in clips:
                    t1 = time.time()
                    mp4 = f"{c}.camera_front_wide_120fov.mp4"
                    tsn = f"{c}.camera_front_wide_120fov.timestamps.parquet"
                    if mp4 not in nm or tsn not in nm:
                        out[c]["err"] = "camera member missing"
                        continue
                    b = z.read(mp4)
                    ts = z.read(tsn)
                    os.makedirs(f"{ROOT}/r0/camera_front_wide", exist_ok=True)
                    for suf, data in ((".mp4", b), (".timestamps.parquet", ts)):
                        p = f"{ROOT}/r0/camera_front_wide/{c}{suf}"
                        with open(p + ".part", "wb") as fh:
                            fh.write(data)
                        os.replace(p + ".part", p)
                    out[c].update(mp4_bytes=len(b), mp4_sha256=sha(b), cam_s=round(time.time() - t1, 2))
            t2 = time.time()
            with F.open(f"datasets/{R}/" + EGOP.format(c=chunk), "rb") as f:
                z = zipfile.ZipFile(f)
                for c in clips:
                    if "err" in out[c]:
                        continue
                    b = member_bytes(f, z, f"{c}.egomotion.parquet")
                    os.makedirs(f"{ROOT}/ego_members", exist_ok=True)
                    open(f"{ROOT}/ego_members/{c}.egomotion.parquet", "wb").write(b)
                    out[c].update(ego_bytes=len(b))
            ego_s = time.time() - t2
            t3 = time.time()
            with F.open(f"datasets/{R}/" + INTP.format(c=chunk), "rb") as f:
                ci = pd.read_parquet(io.BytesIO(f.read()))
            ci = ci.reset_index()
            ci = ci[ci["camera_name"].astype(str) == "camera_front_wide_120fov"].drop_duplicates("clip_id").set_index("clip_id")
            for c in clips:
                if "err" in out[c]:
                    continue
                if c not in ci.index:
                    out[c]["err"] = "intrinsics row missing"
                    continue
                out[c]["intr"] = {k: float(ci.loc[c, k]) for k in ICOLS}
            intr_s = time.time() - t3
            t4 = time.time()
            obs = [c for c in clips if has_obs.get(c) and "err" not in out[c]]
            if obs:
                with F.open(f"datasets/{R}/" + OBSP.format(c=chunk), "rb") as f:
                    z = zipfile.ZipFile(f)
                    os.makedirs(f"{ROOT}/obs_members", exist_ok=True)
                    for c in obs:
                        b = member_bytes(f, z, f"{c}.obstacle.offline.parquet")
                        open(f"{ROOT}/obs_members/{c}.obstacle.offline.parquet", "wb").write(b)
                        out[c]["obs_bytes"] = len(b)
            obs_s = time.time() - t4
            n = max(len(clips), 1)
            for c in clips:
                out[c].update(chunk_secs=round(time.time() - t0, 2), n_in_chunk=len(clips), ego_s=round(ego_s / n, 2),
                              intr_s=round(intr_s / n, 2), obs_s=round(obs_s / n, 2))
                out[c]["status"] = "FAIL" if "err" in out[c] else "OK"
                rows.append(out[c])
            return rows
        except Exception as e:                                          # noqa: BLE001
            last = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(3 * (attempt + 1))
    return [{"clip": c, "chunk": int(chunk), "status": "FAIL", "err": "chunk: " + str(last)} for c in clips]


def done_set():
    d = {}
    if os.path.exists(LEDGER):
        for ln in open(LEDGER, encoding="utf-8"):
            try:
                r = json.loads(ln)
                d[r["clip"]] = r
            except Exception:                                          # noqa: BLE001
                pass
    return d


def finish():
    """Assemble egomotion_all.zip, the intrinsics csv and the obstacle mini zips from the ledger + members (idempotent)."""
    led = done_set()
    ok = {c: r for c, r in led.items() if r.get("status") == "OK"}
    os.makedirs(f"{ROOT}/labels/egomotion", exist_ok=True)
    z = f"{ROOT}/labels/egomotion/egomotion_all.zip"
    with zipfile.ZipFile(z + ".tmp", "w", zipfile.ZIP_STORED) as zf:
        for c in sorted(ok):
            zf.write(f"{ROOT}/ego_members/{c}.egomotion.parquet", f"{c}.egomotion.parquet")
    os.replace(z + ".tmp", z)
    os.makedirs(f"{ROOT}/calibration", exist_ok=True)
    rows = [dict(clip_id=c, **ok[c]["intr"]) for c in sorted(ok)]
    pd.DataFrame(rows)[["clip_id"] + ICOLS].to_csv(f"{ROOT}/calibration/physicalai_front_wide_intrinsics.csv", index=False)
    byc = {}
    for c, r in ok.items():
        if r.get("obs_bytes"):
            byc.setdefault(r["chunk"], []).append(c)
    for ch, cs in byc.items():
        d = f"{ROOT}/obsroot/labels/obstacle.offline"
        os.makedirs(d, exist_ok=True)
        p = f"{d}/obstacle.offline.chunk_{ch:04d}.zip"
        with zipfile.ZipFile(p + ".tmp", "w", zipfile.ZIP_STORED) as zf:
            for c in sorted(cs):
                zf.write(f"{ROOT}/obs_members/{c}.obstacle.offline.parquet", f"{c}.obstacle.offline.parquet")
        os.replace(p + ".tmp", p)
    log(f"finish: {len(ok)} clips ok, {len(byc)} obstacle mini zips")
    return len(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists", nargs="+", required=True, help="clip list files (pilot + controls)")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    clips = []
    for p in a.lists:
        clips += [ln.strip() for ln in open(p, encoding="utf-8") if ln.strip()]
    clips = list(dict.fromkeys(clips))
    ci = pd.read_parquet(f"{META}/clip_index.parquet")
    fp = pd.read_parquet(f"{META}/metadata/feature_presence.parquet")
    chunk_of = {c: int(ci.loc[c, "chunk"]) for c in clips}
    has_obs = {c: bool(fp.loc[c, "obstacle.offline"]) for c in clips}
    done = done_set()
    todo = [c for c in clips if done.get(c, {}).get("status") != "OK"
            or not os.path.exists(f"{ROOT}/r0/camera_front_wide/{c}.mp4")]
    byc = {}
    for c in todo:
        byc.setdefault(chunk_of[c], []).append(c)
    log(f"{len(clips)} clips, {len(todo)} to stage in {len(byc)} chunks, {a.threads} threads")
    t0 = time.time()
    n = 0
    with ThreadPoolExecutor(a.threads) as ex, open(LEDGER, "a", encoding="utf-8") as led:
        futs = [ex.submit(do_chunk, ch, cs, has_obs) for ch, cs in byc.items()]
        for fu in as_completed(futs):
            for r in fu.result():
                r["t"] = time.strftime("%F %T")
                with LOCK:
                    led.write(json.dumps(r) + "\n")
                    led.flush()
                n += 1
                if n % 20 == 0:
                    log(f"{n}/{len(todo)} staged, {n / (time.time() - t0) * 3600:.0f} clips/h")
    wall = time.time() - t0
    nok = finish()
    log(f"STAGE wall {wall:.0f}s for {len(todo)} clips -> {len(todo) / max(wall, 1e-9) * 3600:.0f} clips/h; ok in ledger {nok}/{len(clips)}")
    log("ZZ DS-STAGE DONE ZZ" if nok >= len(clips) - 0 else f"ZZ DS-STAGE PARTIAL ZZ {nok}/{len(clips)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
