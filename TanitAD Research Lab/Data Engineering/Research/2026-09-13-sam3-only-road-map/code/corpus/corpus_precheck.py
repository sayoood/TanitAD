"""Pre-launch checks for the corpus SAM3 map production: order file vs B1 index, c8 uniqueness (scratch names use clip[:8]),
calibration coverage (a front-wide intrinsics AND extrinsics row per clip), HF camera sha256 table fetched at a pinned revision,
and the validated inputs on Thor (data/frontwide, dev-box copies) hashed against that table."""
import collections, hashlib, json, time
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download
C = Path("/home/nvidia/sam3map/corpus"); REPO = "Sayood/tanitad-v7-training-corpus"; REV = "a0cf20dfb4eafa29b0ac1f3c05337f002bc33ca0"
FEAT = "camera_front_wide_120fov"
clips = [l.strip() for l in open(C / "production_order.txt") if l.strip()]
IDX = pd.read_parquet("/home/nvidia/data/b1-bundle/index/clip_to_chunk.parquet")
print("clips", len(clips), "| unique", len(set(clips)), "| unique c8", len({c[:8] for c in clips}), "| not in B1 index", sum(c not in IDX.index for c in clips))
by_chunk = collections.defaultdict(list)
for c in clips:
    by_chunk[int(IDX.loc[c, "chunk"])].append(c)
lack = []
for k, cs in sorted(by_chunk.items()):
    fi, fe = C / "calib" / f"camera_intrinsics.chunk_{k:04d}.parquet", C / "calib" / f"sensor_extrinsics.chunk_{k:04d}.parquet"
    if not (fi.exists() and fe.exists()):
        lack += cs; continue
    idf = pd.read_parquet(fi).reset_index(); edf = pd.read_parquet(fe).reset_index()
    ci = set(idf.loc[idf["camera_name"] == FEAT, "clip_id"]); ce = set(edf.loc[edf["sensor_name"] == FEAT, "clip_id"])
    lack += [c for c in cs if c not in ci or c not in ce]
print("chunks", len(by_chunk), "| clips lacking front-wide calibration", len(lack))
p = hf_hub_download(REPO, "camera/camera_sha256.json", repo_type="dataset", revision=REV, local_dir=str(C / "hfmeta"))
tab = json.load(open(p)); tab = tab.get("sha256", tab) if isinstance(tab, dict) else tab
keys = list(tab)[:2]; print("sha table entries", len(tab), "| key form", keys)
def key_of(clip):
    for k in (clip, f"{clip}.mp4", f"camera/{clip}.mp4"):
        if k in tab:
            return k
print("order clips in sha table", sum(key_of(c) is not None for c in clips))
a = time.time(); n = ok = 0
for f in sorted(Path("/home/nvidia/sam3map/data/frontwide").glob("*.mp4")):
    k = key_of(f.stem); n += 1
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    v = tab[k] if k else None; v = v.get("sha256") if isinstance(v, dict) else v
    ok += int(h == v)
print(f"validated-input mp4s hashed {n} | equal to HF table {ok} | {time.time() - a:.0f}s")
