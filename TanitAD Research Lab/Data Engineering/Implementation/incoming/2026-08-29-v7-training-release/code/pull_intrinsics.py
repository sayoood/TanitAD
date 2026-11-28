"""Pull camera_intrinsics chunks for the corpus and build the full cy table.

Small parquets (one per chunk), so plain hf_hub_download in parallel — no
range tricks needed. Token read in place from G: Keys.txt, never echoed.
Verified by CONTENT: final table must cover 4,719/4,719 clips and the rig
clusters must match the known A≈542 / B≈753 split.
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
import numpy as np
import pandas as pd

KEYS = "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/Keys.txt"
REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
LOCAL = Path("C:/Users/Admin/tanitad-data/physicalai/calibration/camera_intrinsics")
OUT = Path("C:/Users/Admin/tanitad-wt/_s2build/release/front_wide_cy.parquet")
IDX = Path("C:/Users/Admin/tanitad-wt/_s2build/release/tanitad-v7-training-corpus/index/clip_to_chunk.parquet")

def _read_keys(tries=8):
    # G: default per the PI; retry through residual DriveFS flaps.
    for i in range(tries):
        try:
            return open(KEYS, encoding="utf-8", errors="ignore").read()
        except OSError:
            time.sleep(4 * (i + 1))
    raise RuntimeError("Keys.txt unreadable after retries")

tok = re.search(r"hf_[A-Za-z0-9]+", _read_keys()).group(0)

rows = [json.loads(l) for l in open(
    "C:/Users/Admin/tanitad-wt/_s2build/v7_sup/s2_labels_v7.jsonl",
    encoding="utf-8") if l.strip()]
need = {r["clip_id"] for r in rows}
cix = pd.read_parquet(IDX)
chunks = sorted(set(int(c) for c in cix.chunk))
have_chunks = {int(re.search(r"chunk_(\d+)", p).group(1))
               for p in glob.glob(str(LOCAL / "*.parquet"))}
todo = [c for c in chunks if c not in have_chunks]
print(f"corpus chunks {len(chunks)} | local {len(chunks)-len(todo)} | pull {len(todo)}")

lock = threading.Lock()
done = [0]
t0 = time.time()


def one(chunk: int):
    name = f"calibration/camera_intrinsics/camera_intrinsics.chunk_{chunk:04d}.parquet"
    try:
        p = H.hf_hub_download(REPO, name, repo_type="dataset", token=tok,
                              local_dir="C:/Users/Admin/tanitad-data/physicalai/_dl_cal")
        pd.read_parquet(p)          # parse BEFORE accepting
        Path(p).replace(LOCAL / Path(p).name)
        return None
    except Exception as e:                                  # noqa: BLE001
        return f"{chunk}: {type(e).__name__}: {str(e)[:80]}"
    finally:
        with lock:
            done[0] += 1
            if done[0] % 100 == 0 or done[0] == len(todo):
                el = time.time() - t0
                print(f"  [{done[0]}/{len(todo)}] {done[0]/max(el,1e-9)*60:.0f}/min",
                      flush=True)


errs = []
if todo:
    with ThreadPoolExecutor(max_workers=12) as ex:
        for f in as_completed({ex.submit(one, c) for c in todo}):
            r = f.result()
            if r:
                errs.append(r)
print(f"pull done, {len(errs)} errors")
for e in errs[:8]:
    print("  ERR", e)

parts = []
for p in glob.glob(str(LOCAL / "*.parquet")):
    d = pd.read_parquet(p).reset_index()
    parts.append(d[d.camera_name == "camera_front_wide_120fov"]
                 [["clip_id", "width", "height", "cx", "cy"]])
cal = pd.concat(parts).drop_duplicates("clip_id")
sub = cal[cal.clip_id.isin(need)].copy()
sub["rig"] = np.where(sub.cy < 650, "A", "B")
print(f"COVERAGE: {len(sub)}/{len(need)}")
print("rig split:", sub.rig.value_counts().to_dict())
print("clusters:", sub.groupby("rig").cy.agg(["min", "median", "max"]).round(1).to_dict())
a_ok = abs(sub[sub.rig == "A"].cy.median() - 542) < 15
b_ok = abs(sub[sub.rig == "B"].cy.median() - 753) < 15
assert len(sub) == len(need), "coverage incomplete"
assert a_ok and b_ok, "rig clusters do not match the known A~542/B~753 split"
sub.to_parquet(OUT)
print("WROTE", OUT, "— content-verified (coverage + cluster match)")
