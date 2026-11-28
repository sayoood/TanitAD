"""Stage 3, item 2 (AGENT K, the ONE approved download of this brief): the 20 guard scenes' `camera_front_wide_120fov.mp4` (about 0.85 GB), each verified against `lfs_sha256` in RELEASE_LISTING.json.

    python -I download_refs3.py [--workers 3] [--only <scene-prefix>,...]

Everything is `download_refs.py`'s (stage 2's, AGENT J): the same `fetch` (Range resume, sha256 check, `.bad` instead of delete, 10 GB floor per drive), the same token handling (read in place from
Keys.txt, held in a variable, sent only as an Authorization header to huggingface.co, never printed or written), `HF_HUB_DISABLE_XET=1`, revision 5fe7a149. Only the scene list and the output differ:
the 20 guard slots of DOWNLOAD_PLAN.json, each into its OWN scene folder (`<root>/<id>/`, the root the plan names: C: for all 20). A scene whose drive would end below the floor is refused, not moved.
Output: stage3/ref_frames/DOWNLOAD_RESULT3.json (per scene: bytes, sha256, the listing's sha256, ok, attempts, seconds, path).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import sys
import time

os.environ["HF_HUB_DISABLE_XET"] = "1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import download_refs as DR  # noqa: E402

T2 = DR.T2
OUT = f"{T2}/stage3/ref_frames"
PLAN = f"{DR.MC}/t2_scenes/raw/DOWNLOAD_PLAN.json"


def guard_dirs():
    """{scene id: its own scene folder} for guard_1 .. guard_20, in slot order."""
    p = json.load(open(PLAN, encoding="utf-8"))["scenes"]
    g = [s for s in p if str(s["slot"]).startswith("guard")]
    assert [s["slot"] for s in g] == [f"guard_{k}" for k in range(1, 21)], "the plan's guard slots are not guard_1..guard_20"
    return {s["scene"]: f"{s['root']}/{s['scene']}" for s in g}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    lst = json.load(open(DR.LISTING, encoding="utf-8"))
    rev = lst["revision"]
    assert rev.startswith("5fe7a149") and lst["repo"] == DR.REPO, (rev, lst["repo"])
    dirs = guard_dirs()
    todo = {s: d for s, d in dirs.items() if not a.only or any(s.startswith(p) for p in a.only.split(","))}
    assert len(todo) == 20 or a.only, f"{len(todo)} scenes to fetch, expected 20"
    for s, d in todo.items():
        assert os.path.isdir(d), f"{d}: the scene folder is missing"
    os.makedirs(OUT, exist_ok=True)
    DR.log(f"revision {rev[:8]}; {len(todo)} guard scenes; {sum(lst['scenes'][s][DR.MP4]['size'] for s in todo)} bytes; workers {a.workers}; the token is not logged")
    planned = {}
    for sid in sorted(todo):                                                   # own folder, or refuse: the floor is checked per drive with the bytes already planned
        size, drv = lst["scenes"][sid][DR.MP4]["size"] + DR.NPZ_RESERVE, os.path.splitdrive(todo[sid])[0]
        free = shutil.disk_usage(todo[sid]).free - planned.get(drv, 0)
        assert free - size >= DR.FLOOR, f"{drv} would end below the 10 GB floor at {sid[:8]} (free {free / 1e9:.2f} GB)"
        planned[drv] = planned.get(drv, 0) + size
    res = {}

    def one(sid):
        m = lst["scenes"][sid][DR.MP4]
        d = f"{todo[sid]}/{DR.MP4}"
        out = DR.fetch(sid, d, m["size"], m["lfs_sha256"], rev)
        out.update(path=d, placement="own scene folder", listing_lfs_sha256=m["lfs_sha256"], listing_size=m["size"])
        DR.log(f"{sid[:8]} {'OK' if out['ok'] else 'FAILED'} {out['bytes']} B attempts {out['attempts']} {out['seconds']} s")
        return sid, out
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for fut in cf.as_completed([ex.submit(one, s) for s in sorted(todo)]):
            try:
                sid, out = fut.result()
            except Exception as e:                                             # noqa: BLE001
                DR.log(f"worker error: {type(e).__name__} {str(e)[:150]}")
                continue
            res[sid] = out
    ok = [s for s, o in res.items() if o["ok"] and o["sha256"] == o["listing_lfs_sha256"]]
    summary = {"revision": rev, "n_scenes": len(todo), "n_ok": len(ok), "n_failed": len(todo) - len(ok), "total_bytes": sum(o["bytes"] for o in res.values()), "scenes": res,
               "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    name = f"{OUT}/DOWNLOAD_RESULT3.json" if not a.only else f"{OUT}/DOWNLOAD_RESULT3_only.json"
    with open(name, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=1)
    DR.log(f"DONE {len(ok)}/{len(todo)} verified against lfs_sha256; {name}")
    return 0 if len(ok) == len(todo) else 1


if __name__ == "__main__":
    sys.exit(main())
