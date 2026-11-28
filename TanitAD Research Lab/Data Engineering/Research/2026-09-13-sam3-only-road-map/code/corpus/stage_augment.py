"""Stage the dev-box augmentation components for the private HF corpus repo (PI 2026-09-15: "the data set should include all
augmentations"), in repo layout under STAGE, each with its own manifest:
  calibration/camera_intrinsics.parquet, calibration/sensor_extrinsics.parquet   upstream rows (all cameras / all sensors) for the corpus clips
  lidar_bev_gt/<clip_id>.bevgt.npz (+ LIDAR_BEV_GT_MANIFEST.json)               the 315 LiDAR BEV ground-truth clips (schema tanitad.lidar_bev_gt/2)
  agents/obstacle_offline/obstacle_offline_<NNN>.parquet (+ index.parquet, AGENTS_MANIFEST.json)   upstream obstacle.offline tracks, 100 clips/shard,
                                                                                one row group per clip, a clip_id column added, rows unchanged
Checks: every BEV file equals its builder record (bytes), the D: mirror equals the C: build cache (sha256), no quarantined clip ships;
every obstacle shard re-reads to the source parquet row for row (DataFrame equality per clip)."""
import collections, hashlib, json, shutil, sys, time
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SP = Path(r"<scratchpad>")
STAGE = Path(r"D:/Projects/TanitAD-artifacts/hf-corpus-aug-20260915/stage")
DATA = Path(r"C:/Users/Admin/tanitad-data/physicalai")
IDX = pd.read_parquet(SP / "corpus/hf/index/clip_to_chunk.parquet")
CLIPS = sorted(IDX.index); SHA = {hashlib.sha256(c.encode()).hexdigest()[:12]: c for c in CLIPS}
assert len(CLIPS) == 4719 and len(SHA) == 4719
T0 = time.strftime("%Y-%m-%dT%H:%M:%S")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def calibration():
    out = STAGE / "calibration"; out.mkdir(parents=True, exist_ok=True)
    ints, exts = [], []
    for chunk, grp in IDX.groupby("chunk"):
        want = set(grp.index)
        i = pd.read_parquet(DATA / "calibration/camera_intrinsics" / f"camera_intrinsics.chunk_{chunk:04d}.parquet").reset_index()
        e = pd.read_parquet(DATA / "calibration/sensor_extrinsics" / f"sensor_extrinsics.chunk_{chunk:04d}.parquet").reset_index()
        ints.append(i[i["clip_id"].isin(want)].assign(chunk=chunk)); exts.append(e[e["clip_id"].isin(want)].assign(chunk=chunk))
    I = pd.concat(ints).sort_values(["clip_id", "camera_name"]).reset_index(drop=True); E = pd.concat(exts).sort_values(["clip_id", "sensor_name"]).reset_index(drop=True)
    fw = I[I["camera_name"] == "camera_front_wide_120fov"]; fe = E[E["sensor_name"] == "camera_front_wide_120fov"]
    assert fw["clip_id"].nunique() == 4719 and len(fw) == 4719 and fe["clip_id"].nunique() == 4719 and len(fe) == 4719, (len(fw), len(fe))
    I.to_parquet(out / "camera_intrinsics.parquet", index=False); E.to_parquet(out / "sensor_extrinsics.parquet", index=False)
    return {"camera_intrinsics.parquet": {"rows": len(I), "clips": int(I["clip_id"].nunique()), "cameras": sorted(I["camera_name"].unique().tolist())},
            "sensor_extrinsics.parquet": {"rows": len(E), "clips": int(E["clip_id"].nunique()), "sensors": sorted(E["sensor_name"].unique().tolist())}}


def lidar_bev():
    out = STAGE / "lidar_bev_gt"; out.mkdir(parents=True, exist_ok=True)
    subsets = {"b1eval": (Path(r"D:/Projects/TanitAD-artifacts/bev-lidar-gt-b1eval-20260913"), Path(r"C:/Users/Admin/tanitad-caches/bevhead-20260913/bev_gt")),
               "b1train200": (Path(r"D:/Projects/TanitAD-artifacts/bev-lidar-gt-b1train200-20260913"), Path(r"C:/Users/Admin/tanitad-caches/bevhead-20260913/bev_gt_extra"))}
    clips = {}; schema_meta = None
    for name, (mirror, cache) in subsets.items():
        rec = {json.loads(l)["clip_sha12"]: json.loads(l) for l in open(cache / "manifest.jsonl")}
        ok = {s for s, r in rec.items() if r.get("ok")}
        files = sorted(mirror.glob("*.bevgt.npz")); got = {f.name.split(".")[0] for f in files}
        quarantined = {f.name.split(".")[0] for f in (cache / "quarantine").glob("*.bevgt.npz")}
        assert got == ok and not (got & quarantined), (name, len(got), len(ok), len(got & quarantined))
        for f in files:
            s = f.name.split(".")[0]; r = rec[s]; c = SHA[s]
            h = sha256(f); assert f.stat().st_size == r["artifact_bytes"] and h == sha256(cache / f.name), (name, s)
            shutil.copy2(f, out / f"{c}.bevgt.npz")
            clips[c] = {"sha12": s, "subset": name, "split": str(IDX.loc[c, "split"]), "sha256": h, "bytes": f.stat().st_size,
                        "builder_record": {k: v for k, v in r.items() if k not in ("fetch", "protected", "parquet_deleted", "t_unix", "decode_s", "raster_s", "build_s", "frames_s")}}
            if schema_meta is None:
                import numpy as np
                schema_meta = json.loads(str(np.load(f, allow_pickle=True)["meta_json"]))
    man = {"schema": "tanitad.lidar_bev_gt_manifest/1", "component": "lidar_bev_gt", "status": "COMPLETE", "generated": T0,
           "what": "per-frame LiDAR BEV ground truth (occupancy / observed / camera-visible grids) on the v2ep EPISODE grid, one file per clip",
           "label_only": "built from the top LiDAR: a training TARGET and evaluation reference, never an inference input (inference is vision-only)",
           "clips": len(clips), "by_subset": dict(collections.Counter(v["subset"] for v in clips.values())), "by_split": dict(collections.Counter(v["split"] for v in clips.values())),
           "builder": "TanitAD repo, TanitAD Research Lab/Architecture & Inference/Research/2026-09-13-bev-lidar-corpus-and-head (code/lidar_bev.py; loader code/bev_gt_loader.py)",
           "checks_at_publish": "file bytes == builder record; D: mirror sha256 == C: build cache sha256; builder ok == shipped set; 0 quarantined clips shipped",
           "file_schema_example": schema_meta, "files": clips}
    (out / "LIDAR_BEV_GT_MANIFEST.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return {"clips": len(clips)}


def agents():
    out = STAGE / "agents" / "obstacle_offline"; out.mkdir(parents=True, exist_ok=True)
    src = {}
    for sub in ("obstacle_offline_b1train", "obstacle_offline_b1eval"):
        for f in (DATA / "labels" / sub).glob("*.parquet"):
            assert f.stem not in src; src[f.stem] = (f, sub)
    have = [c for c in CLIPS if c in src]; missing = [c for c in CLIPS if c not in src]
    # MEASURED 2026-09-15: a clip with 0 upstream cuboids is stored with null-typed string columns and float32 numbers, so the shard schema
    # comes from a non-empty source file (native parquet types, not a pandas round trip) and empty clips are cast to it
    ref = next(pq.read_schema(src[c][0]) for c in have if pq.read_metadata(src[c][0]).num_rows > 0).remove_metadata()
    schema = pa.schema([pa.field("clip_id", pa.string())] + list(ref))
    index = []; shards = {}; empty = []
    for k in range(0, len(have), 100):
        name = f"obstacle_offline_{k // 100:03d}.parquet"
        with pq.ParquetWriter(out / name, schema) as w:
            for c in have[k:k + 100]:
                f, sub = src[c]; t = pq.read_table(f).replace_schema_metadata(None)
                t = t.cast(ref) if t.num_rows else ref.empty_table()
                empty += [c] if t.num_rows == 0 else []
                w.write_table(t.add_column(0, schema.field("clip_id"), pa.array([c] * t.num_rows, pa.string())))
                index.append({"clip_id": c, "shard": name, "rows": t.num_rows, "source_subset": sub, "source_bytes": f.stat().st_size, "source_sha256": sha256(f), "split": str(IDX.loc[c, "split"])})
        back = pd.read_parquet(out / name)                                      # re-read: rows unchanged per clip
        for c in have[k:k + 100]:
            a = pd.read_parquet(src[c][0]).reset_index(drop=True); b = back[back["clip_id"] == c].drop(columns="clip_id").reset_index(drop=True)
            pd.testing.assert_frame_equal(a, b, check_dtype=False)
        shards[name] = {"clips": len(have[k:k + 100]), "rows": int(len(back))}
        print("shard", name, shards[name], flush=True)
    pd.DataFrame(index).to_parquet(out / "index.parquet", index=False)
    classes = collections.Counter()
    for name in shards:
        classes.update(pd.read_parquet(out / name, columns=["label_class"])["label_class"].value_counts().to_dict())
    man = {"schema": "tanitad.agents_manifest/1", "component": "agents/obstacle_offline", "status": "COMPLETE", "generated": T0,
           "what": "UPSTREAM obstacle.offline 3D cuboid tracks (nvidia/PhysicalAI-Autonomous-Vehicles labels/obstacle.offline, source 'scene:obstacles:autolabels:v2'), "
                   "per clip, rows unchanged; a clip_id column is added and each clip is one parquet row group (filter on clip_id)",
           "not_an_augmentation": "published upstream labels, shipped here so the corpus is self-contained for agent / BEV work",
           "clips_with_source_file": len(have), "clips_with_zero_cuboids": len(empty), "clips_with_zero_cuboids_list": empty,
           "clips_without_source_file": len(missing), "clips_without_source_file_list": missing,
           "rows": int(sum(v["rows"] for v in shards.values())), "label_class_counts": dict(classes), "shards": shards,
           "columns": list(pd.read_parquet(out / next(iter(shards))).columns), "index": "agents/obstacle_offline/index.parquet (clip_id, shard, rows, source sha256)",
           "frames": "reference_frame 'rig' at reference_frame_timestamp_us; timestamps in microseconds on the clip timeline (same clock as egomotion / camera timestamps)",
           "checks_at_publish": "every shard re-read and compared to its source parquets clip by clip (pandas assert_frame_equal)"}
    (STAGE / "agents" / "AGENTS_MANIFEST.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return {"clips": len(have), "zero_cuboids": len(empty), "missing": len(missing), "shards": len(shards)}


if __name__ == "__main__":
    STAGE.mkdir(parents=True, exist_ok=True)
    todo = sys.argv[1:] or ["calibration", "lidar_bev", "agents"]
    res = {k: globals()[k]() for k in todo}
    files = {str(p.relative_to(STAGE)).replace("\\", "/"): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(STAGE.rglob("*")) if p.is_file() and p.name != "stage_files.json"}
    (STAGE / "stage_files.json").write_text(json.dumps(files, indent=1), encoding="utf-8")
    print("ZZSTAGE", json.dumps(res), "| files", len(files), "| GB", round(sum(v["bytes"] for v in files.values()) / 1e9, 3))
