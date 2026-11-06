"""Download the 41 stage-2 scenes' `camera_front_wide_120fov.mp4` (AGENT J, the ONE approved download of this brief) and verify each against `lfs_sha256` in RELEASE_LISTING.json.

    python -I download_refs.py [--workers 3] [--only <scene-prefix>,...]

* Source: nvidia/PhysicalAI-Autonomous-Vehicles-NuRec at revision 5fe7a149 (the listing's own revision), `sample_set/26.04_release/<scene>/camera_front_wide_120fov.mp4`, plain HTTPS
  with `HF_HUB_DISABLE_XET=1`. The 42nd scene, 7c72937c (the kit scene), is NOT downloaded: stage 1 already holds its mp4, verified against the same listing.
* Token: read in place from Keys.txt, held in a variable, sent only as an Authorization header to huggingface.co (never to the CDN hop), never printed, logged or written.
* Placement: into each scene's OWN folder (SCENE_SOURCES.json: F:/TanitAD-scratch/t2_scenes/<id>/ or the C: twin where that scene lives). A drive is never taken below 10 GB free.
* Resume: `<mp4>.part` is continued with a Range request; a finished file is verified (size, sha256) and kept; a file that fails is moved to `.bad` (never deleted) and fetched again.
* Output: DOWNLOAD_RESULT.json beside the extraction output (per scene: bytes, sha256, the listing's sha256, ok, attempts, seconds), one log line per event.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time

os.environ["HF_HUB_DISABLE_XET"] = "1"
import requests  # noqa: E402

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
T2 = os.path.dirname(HERE)
MC = os.path.dirname(T2)
KEYS = "E:/Projects/TanitAD/Keys.txt"
REPO = "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"
REL = "sample_set/26.04_release"
MP4 = "camera_front_wide_120fov.mp4"
KIT_SCENE = "7c72937c-c620-4776-9555-d57222c0081f"
LISTING = f"{MC}/t2_scenes/raw/RELEASE_LISTING.json"
SOURCES = f"{T2}/stage2/SCENE_SOURCES.json"
OUT = f"{T2}/stage2/ref_frames"
FLOOR = 10 * 10**9
NPZ_RESERVE = 24 * 10**6                                                                 # the derived ref_frames.npz lands beside the mp4 (3 frames, 8 to 17 MB measured on stage 1)
STAGE_C = "C:/Users/Admin/TanitAD-scratch/t2_refs"
CHUNK = 1 << 20
lock = threading.Lock()
inflight = {}
_TOKS = None


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)


def tokens():
    global _TOKS
    if _TOKS is None:
        seen = []
        for m in re.findall(r"hf_[A-Za-z0-9]+", open(KEYS, encoding="utf-8", errors="ignore").read()):
            if m not in seen:
                seen.append(m)
        _TOKS = seen
    return _TOKS


_ACTIVE = [None]


def get(url, rng, rev):
    """GET with the bearer token on the huggingface.co hop ONLY; redirects are followed by hand. -> a streaming response with the final 200/206 body."""
    cands = [_ACTIVE[0]] if _ACTIVE[0] else tokens()
    last = None
    for tok in cands:
        h = {"User-Agent": "tanitad-t2v8f-refs/1", "Range": rng}
        r = requests.get(url, headers={**h, "Authorization": "Bearer " + tok}, allow_redirects=False, timeout=120, stream=True)
        hops = 0
        while r.status_code in (301, 302, 303, 307, 308) and hops < 6:
            loc = r.headers["Location"]
            if loc.startswith("/"):
                loc = "https://huggingface.co" + loc
            r.close()
            hh = dict(h)
            if loc.split("/")[2].endswith("huggingface.co"):
                hh["Authorization"] = "Bearer " + tok
            r = requests.get(loc, headers=hh, allow_redirects=False, timeout=120, stream=True)
            hops += 1
        if r.status_code in (200, 206):
            _ACTIVE[0] = tok
            return r
        last = r.status_code
        r.close()
        if last not in (401, 403):
            break
    raise RuntimeError(f"HTTP {last}")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(sid, dest, size, want_sha, rev):
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{rev}/{REL}/{sid}/{MP4}"
    drive = os.path.splitdrive(dest)[0]
    t0, attempts = time.time(), 0
    if os.path.isfile(dest) and os.path.getsize(dest) == size and sha256_file(dest) == want_sha:
        return {"bytes": size, "sha256": want_sha, "ok": True, "attempts": 0, "seconds": 0.0, "note": "already present and verified"}
    part = dest + ".part"
    while attempts < 6:
        attempts += 1
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have > size:
            os.replace(part, part + ".oversize")
            have = 0
        with lock:
            free = shutil.disk_usage(os.path.dirname(dest)).free
            need = size - have
            if free - inflight.get(drive, 0) - need < FLOOR:
                raise RuntimeError(f"DISK FLOOR: {drive} free {free} would drop below 10 GB for {sid[:8]}")
            inflight[drive] = inflight.get(drive, 0) + need
        taken = 0
        try:
            if have < size:
                r = get(url, f"bytes={have}-{size - 1}", rev)
                with open(part, "ab") as f:
                    for b in r.iter_content(CHUNK):
                        if b:
                            f.write(b)
                            taken += len(b)
                            with lock:
                                inflight[drive] -= len(b)
                r.close()
        except Exception as e:                                                          # noqa: BLE001
            log(f"{sid[:8]} attempt {attempts}: {type(e).__name__} {str(e)[:100]}")
            time.sleep(min(30, 3 * attempts))
            continue
        finally:
            with lock:
                inflight[drive] = max(0, inflight.get(drive, 0) - max(0, (size - have) - taken))
        got = os.path.getsize(part)
        if got == size and sha256_file(part) == want_sha:
            os.replace(part, dest)
            return {"bytes": size, "sha256": want_sha, "ok": True, "attempts": attempts, "seconds": round(time.time() - t0, 1)}
        log(f"{sid[:8]} attempt {attempts}: size {got}/{size} or sha256 mismatch: the part is moved to .bad")
        os.replace(part, f"{part}.bad{attempts}")
    return {"bytes": os.path.getsize(part) if os.path.exists(part) else 0, "sha256": None, "ok": False, "attempts": attempts, "seconds": round(time.time() - t0, 1)}


def scene_dirs():
    src = json.load(open(SOURCES, encoding="utf-8"))
    out = {}
    for e in src:
        if e["rel"] == "volume.nurec":
            out[e["scene"]] = os.path.dirname(e["source"]).replace("\\", "/")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    lst = json.load(open(LISTING, encoding="utf-8"))
    rev = lst["revision"]
    assert rev.startswith("5fe7a149") and lst["repo"] == REPO, (rev, lst["repo"])
    dirs = scene_dirs()
    todo = {s: d for s, d in dirs.items() if s != KIT_SCENE and (not a.only or any(s.startswith(p) for p in a.only.split(",")))}
    assert len(todo) == 41 or a.only, f"{len(todo)} scenes to fetch, expected 41"
    os.makedirs(OUT, exist_ok=True)
    log(f"revision {rev[:8]}; {len(todo)} scenes; {sum(lst['scenes'][s][MP4]['size'] for s in todo)} bytes; workers {a.workers}; the token is not logged")
    res = {}
    # PLACEMENT: the scene's own folder (the brief), unless that drive would end below the 10 GB floor (the standing rule: "never leave any drive below 10 GB free"). F: was at 7.97 GB free before
    # the first byte, so its scenes are STAGED on C: in a folder of their own that holds only this mp4 and its derived npz; the choice is logged per scene and recorded in the result.
    place, planned = {}, {}
    for sid in sorted(todo):
        own, size = todo[sid], lst["scenes"][sid][MP4]["size"] + NPZ_RESERVE
        drv = os.path.splitdrive(own)[0]
        if shutil.disk_usage(own).free - planned.get(drv, 0) - size >= FLOOR:
            place[sid] = (own, "own scene folder")
            planned[drv] = planned.get(drv, 0) + size
        else:
            alt = f"{STAGE_C}/{sid}"
            assert shutil.disk_usage("C:/").free - planned.get("C:", 0) - size >= FLOOR, f"C: cannot take {sid[:8]} above the 10 GB floor"
            os.makedirs(alt, exist_ok=True)
            place[sid] = (alt, f"staged on C: ({drv} free {shutil.disk_usage(own).free / 1e9:.2f} GB is below the 10 GB floor)")
            planned["C:"] = planned.get("C:", 0) + size
    log("placement: " + str({w: sum(1 for v in place.values() if v[1].startswith(w)) for w in ("own scene folder", "staged on C:")}))

    def one(sid):
        m = lst["scenes"][sid][MP4]
        d = f"{place[sid][0]}/{MP4}"
        out = fetch(sid, d, m["size"], m["lfs_sha256"], rev)
        out.update(path=d, placement=place[sid][1], listing_lfs_sha256=m["lfs_sha256"], listing_size=m["size"])
        log(f"{sid[:8]} {'OK' if out['ok'] else 'FAILED'} {out['bytes']} B attempts {out['attempts']} {out['seconds']} s")
        return sid, out
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for fut in cf.as_completed([ex.submit(one, s) for s in sorted(todo)]):
            try:
                sid, out = fut.result()
            except Exception as e:                                                      # noqa: BLE001
                log(f"worker error: {type(e).__name__} {str(e)[:150]}")
                continue
            res[sid] = out
    ok = [s for s, o in res.items() if o["ok"] and o["sha256"] == o["listing_lfs_sha256"]]
    summary = {"revision": rev, "n_scenes": len(todo), "n_ok": len(ok), "n_failed": len(todo) - len(ok), "total_bytes": sum(o["bytes"] for o in res.values()), "scenes": res,
               "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    name = f"{OUT}/DOWNLOAD_RESULT.json" if not a.only else f"{OUT}/DOWNLOAD_RESULT_only.json"
    with open(name, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=1)
    log(f"DONE {len(ok)}/{len(todo)} verified against lfs_sha256; {name}")
    return 0 if len(ok) == len(todo) else 1


if __name__ == "__main__":
    sys.exit(main())
