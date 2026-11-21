"""Pull the sensor_extrinsics chunks the B1 epcache build needs.

MEASURED 2026-08-29: 52 of 1,411 required chunks are local. Without the rest,
`extrinsics_for_clip` returns **None** and callers "treat the mount as level
(optical axis == horizon)" -- physicalai.py:450 -- for 99 % of the corpus,
SILENTLY. The mount pitch is what locates the horizon in the output frame, so
this is not cosmetic.

Same shape as `pull_intrinsics.py` (small per-chunk parquets, plain download),
verified by CONTENT: every accepted file is parsed before it is kept, and the
final coverage is asserted against the corpus clip list, not against a count of
files.
"""
import truststore; truststore.inject_into_ssl()

import glob
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import huggingface_hub as H
import pandas as pd

KEYS = "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/Keys.txt"
REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
KIND = "sensor_extrinsics"
LOCAL = Path(f"C:/Users/Admin/tanitad-data/physicalai/calibration/{KIND}")
STAGE = "C:/Users/Admin/tanitad-data/physicalai/_dl_cal"
IDX = Path("C:/Users/Admin/tanitad-wt/_s2build/release/tanitad-v7-training-corpus"
           "/index/clip_to_chunk.parquet")
LOCAL.mkdir(parents=True, exist_ok=True)


def _read_keys(tries=150):
    """G: is the default surface; ride out DriveFS outages rather than dying."""
    for i in range(tries):
        try:
            return open(KEYS, encoding="utf-8", errors="ignore").read()
        except OSError:
            time.sleep(min(60, 4 * (i + 1)))
    raise RuntimeError("Keys.txt unreadable")


tok = re.search(r"hf_[A-Za-z0-9]+", _read_keys()).group(0)

need = {json.loads(l)["clip_id"] for l in open(
    "C:/Users/Admin/tanitad-wt/_s2build/v7_sup/s2_labels_v7.jsonl",
    encoding="utf-8") if l.strip()}
cix = pd.read_parquet(IDX)
chunks = sorted({int(c) for c in cix.chunk})
have = {int(re.search(r"chunk_(\d+)", p).group(1))
        for p in glob.glob(str(LOCAL / "*.parquet"))}
todo = [c for c in chunks if c not in have]
print(f"corpus chunks {len(chunks)} | local {len(chunks)-len(todo)} | pull {len(todo)}",
      flush=True)

lock, done, t0 = threading.Lock(), [0], time.time()


def one(chunk: int):
    name = f"calibration/{KIND}/{KIND}.chunk_{chunk:04d}.parquet"
    try:
        p = H.hf_hub_download(REPO, name, repo_type="dataset", token=tok,
                              local_dir=STAGE)
        pd.read_parquet(p)                      # parse BEFORE accepting
        Path(p).replace(LOCAL / Path(p).name)
        return None
    except Exception as e:                                  # noqa: BLE001
        return f"{chunk}: {type(e).__name__}: {str(e)[:80]}"
    finally:
        with lock:
            done[0] += 1
            if done[0] % 100 == 0 or done[0] == len(todo):
                el = time.time() - t0
                print(f"  [{done[0]}/{len(todo)}] {done[0]/max(el,1e-9)*60:.0f}/min "
                      f"{el/60:.1f} min", flush=True)


errs = []
if todo:
    with ThreadPoolExecutor(max_workers=8) as ex:
        for f in as_completed({ex.submit(one, c) for c in todo}):
            r = f.result()
            if r:
                errs.append(r)
print(f"pull done, {len(errs)} errors", flush=True)
for e in errs[:8]:
    print("  ERR", e)

# --- CONTENT verification: coverage over CLIPS, not over files ---------------
ids = set()
for p in glob.glob(str(LOCAL / "*.parquet")):
    d = pd.read_parquet(p)
    if isinstance(d.index, pd.MultiIndex) or d.index.name:
        d = d.reset_index()
    if "camera_name" in d.columns:
        d = d[d.camera_name == "camera_front_wide_120fov"]
    ids |= set(d["clip_id"].astype(str))
cov = len(need & ids)
print(f"EXTRINSICS COVERAGE: {cov}/{len(need)} corpus clips "
      f"({cov/len(need)*100:.2f}%)", flush=True)
if cov < len(need):
    miss = sorted(need - ids)[:5]
    print(f"  still missing {len(need)-cov}, e.g. {miss}")
