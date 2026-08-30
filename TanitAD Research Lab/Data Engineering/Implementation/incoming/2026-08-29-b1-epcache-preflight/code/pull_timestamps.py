"""Pull the per-clip `.timestamps.parquet` the epcache build cannot start without.

MEASURED 2026-08-30 on Thor: `discover_r0_clips` yielded ZERO clips and the build
could not start. Reading its source (`physicalai.py:460`) gives the contract the
bundle did not satisfy:

    root/r0/camera_front_wide/<clip>.mp4                       <- dir name, no _120fov
    root/r0/camera_front_wide/<clip>.timestamps.parquet        <- ABSENT from the bundle
    root/labels/egomotion/egomotion.chunk_NNNN.zip             <- we shipped one .tar

⚠️ It returns an empty list SILENTLY (`if ts.exists() and ego_zip.exists()`), so a
layout miss reads as "0 clips", not as an error naming the missing file. The
preflight passed because it checks CALIBRATION, not DISCOVERY — two different
questions, and I had only instrumented one.

The timestamps live in the SAME camera chunk zips as the mp4s (verified: chunk
1409 holds 98 mp4 + 98 timestamps + 98 blurred_boxes), so this reuses the
HTTP-range reader — small parquets, no 2 GB chunk downloads.

Banked as `<clip>.timestamps.parquet` to match our `<clip>.mp4` naming, because
discovery derives one from the other by `name.replace(".mp4", ".timestamps.parquet")`.
"""
import os
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, r"C:\Users\Admin\tanitad-wt\_s2build\dl")
import pull_camera as P  # noqa: E402  (HTTPRangeFile + BASE + clip_index)

OUT = P.OUT                                    # beside the mp4s
SUFFIX = ".timestamps.parquet"
need = sorted({x.split(".")[0] for x in os.listdir(OUT) if x.endswith(".mp4")})
have = {x[:-len(SUFFIX)] for x in os.listdir(OUT) if x.endswith(SUFFIX)}
todo = [c for c in need if c not in have]
print(f"clips {len(need)} | timestamps present {len(need)-len(todo)} | to pull {len(todo)}",
      flush=True)

ci = pd.read_parquet(f"{P.ROOT}/clip_index.parquet")
by_chunk: dict[int, list[str]] = {}
for cid in todo:
    if cid in ci.index:
        by_chunk.setdefault(int(ci.loc[cid, "chunk"]), []).append(cid)
print(f"chunks to visit: {len(by_chunk)}", flush=True)

lock, done, errs = threading.Lock(), [0], []
t0 = time.time()


def one(chunk: int, ids: list[str]):
    url = f"{P.BASE}camera_front_wide_120fov.chunk_{chunk:04d}.zip"
    n = 0
    try:
        with zipfile.ZipFile(P.HTTPRangeFile(url)) as z:
            names = [m for m in z.namelist() if m.endswith(SUFFIX)]
            for cid in ids:
                m = next((x for x in names if x.split("/")[-1].startswith(cid)), None)
                if not m:
                    continue
                data = z.read(m)
                # CONTENT check: a short range read still "unzips". Parquet's
                # magic is PAR1 at both ends — a truncated read fails the tail.
                if data[:4] != b"PAR1" or data[-4:] != b"PAR1":
                    with lock:
                        errs.append(f"{cid[:8]}: not a parquet (magic) — refused")
                    continue
                p = f"{OUT}/{cid}{SUFFIX}"
                with open(p + ".part", "wb") as fh:
                    fh.write(data)
                os.replace(p + ".part", p)
                n += 1
    except Exception as e:                                  # noqa: BLE001
        with lock:
            errs.append(f"chunk {chunk}: {type(e).__name__}: {str(e)[:90]}")
    with lock:
        done[0] += n
        if done[0] % 250 < n:
            el = time.time() - t0
            print(f"  [{done[0]}/{len(todo)}] {done[0]/max(el,1e-9)*60:.0f}/min "
                  f"{el/60:.1f} min errs={len(errs)}", flush=True)


with ThreadPoolExecutor(max_workers=6) as ex:
    for f in as_completed({ex.submit(one, c, ids) for c, ids in sorted(by_chunk.items())}):
        f.result()

final = len({x[:-len(SUFFIX)] for x in os.listdir(OUT) if x.endswith(SUFFIX)} & set(need))
print(f"TIMESTAMPS DONE: {final}/{len(need)} clips, {len(errs)} errors, "
      f"{(time.time()-t0)/60:.0f} min", flush=True)
for e in errs[:8]:
    print("  ERR", e)
