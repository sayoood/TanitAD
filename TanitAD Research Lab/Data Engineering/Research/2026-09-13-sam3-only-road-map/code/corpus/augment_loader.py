"""Loader + verifier for the augmentation components of Sayood/tanitad-v7-training-corpus (added 2026-09-15).

Components (all keyed by clip_id, all on the clip's own clock):
  semantic_maps/gt/<clip_id>.sam3mapgt.npz      per-frame BEV semantic GT from the SAM3 front-camera map r (schema tanitad.sam3_map_gt/2)
  semantic_maps/gt_flagged/<clip_id>.sam3mapgt.npz   same, for clips that failed the gate only where nothing contradicts the map (a parked /
                                                stopped ego, or near-field road left unlabelled); every other check passed -- opt in
  semantic_maps/worldmap/<clip_id>.worldmap.npz the clip's world-frame map r (0.1 m) the GT is sampled from
  lidar_bev_gt/<clip_id>.bevgt.npz              per-frame LiDAR BEV GT, 315 clips (schema tanitad.lidar_bev_gt/2)
  agents/obstacle_offline/*.parquet             upstream obstacle.offline tracks, one row group per clip (+ index.parquet)
  calibration/camera_intrinsics.parquet, calibration/sensor_extrinsics.parquet   upstream calibration rows of the corpus clips

LABEL-ONLY. The maps are non-causal (built from every frame of a clip) and the LiDAR GT reads a sensor the deployed model does not have:
they are training targets and evaluation references, never inference inputs (programme rule: inference is vision-only).

Usage:
  python augment_loader.py --root <local snapshot> verify semantic_maps|lidar_bev_gt|agents|calibration [--limit N]
  python augment_loader.py --root <local snapshot> show <clip_id>
A local snapshot: huggingface_hub.snapshot_download("Sayood/tanitad-v7-training-corpus", repo_type="dataset",
                  allow_patterns=["semantic_maps/*", "lidar_bev_gt/*", "agents/*", "calibration/*"], local_dir=<root>)"""
import argparse, hashlib, json
from pathlib import Path

import numpy as np

SEMANTIC_CHANNELS = ["seen, no map class", "drivable", "lane / road line", "crosswalk", "arrow / text", "non-drivable edge", "hatched area",
                     "sidewalk / verge", "not seen"]                          # cart/polar *_frac channel order; fine_codes 0-7 = channels 0-7, 255 = not seen


def load_semantic_map(root, clip_id, as_float=True, allow_flagged=False):
    """dict: tier ("validated" | "flagged"; the reason is in SEMANTIC_MAPS_MANIFEST.json:clips_flagged), meta, t_query_us [T], t_img_us [T], cam_frame_idx [T], T_world_rig [T,4,4], cart [T,9,120,64], polar [T,9,24,20],
    polar48 [T,9,48,40] (fractions of each cell; float32 in [0,1] when as_float, else uint8 x/255), fine_codes [T,600,320].
    Frame: rig, +x forward, +y LEFT; cart row 0 = x in [0, 0.5 m), col 0 = y = -16 m (RIGHT)."""
    f = Path(root) / "semantic_maps" / "gt" / f"{clip_id}.sam3mapgt.npz"; tier = "validated"
    if not f.exists() and allow_flagged and (Path(root) / "semantic_maps" / "gt_flagged" / f"{clip_id}.sam3mapgt.npz").exists():
        f = Path(root) / "semantic_maps" / "gt_flagged" / f"{clip_id}.sam3mapgt.npz"; tier = "flagged"
    z = np.load(f, allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    assert meta["schema"] == "tanitad.sam3_map_gt/2", meta["schema"]
    assert meta["source"]["clip_sha12"] == _sha12(clip_id), f"{clip_id}: file belongs to clip sha12 {meta['source']['clip_sha12']}"
    conv = (lambda a: a.astype(np.float32) / meta["fraction_scale"]) if as_float else (lambda a: a)
    return {"tier": tier, "meta": meta, "t_query_us": z["t_query_us"], "t_img_us": z["t_img_us"], "cam_frame_idx": z["cam_frame_idx"], "T_world_rig": z["T_world_rig"],
            "cart": conv(z["cart_frac"]), "polar": conv(z["polar_frac"]), "polar48": conv(z["polar48_frac"]), "fine_codes": z["fine_codes"]}


def load_worldmap(root, clip_id):
    """dict: cls [H,W] uint8 class codes, rng [H,W] float16, origin (x0, y0) m in the clip world frame, res m/cell, options."""
    z = np.load(Path(root) / "semantic_maps" / "worldmap" / f"{clip_id}.worldmap.npz", allow_pickle=True)
    out = {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}
    assert str(out["clip_sha12"]) == _sha12(clip_id), f"{clip_id}: worldmap belongs to clip sha12 {out['clip_sha12']}"
    return out


def load_lidar_bev(root, clip_id):
    """dict of the LiDAR BEV GT arrays (cart/polar24/polar48 occupancy, observed, camera-visible, label_valid per frame) + meta."""
    z = np.load(Path(root) / "lidar_bev_gt" / f"{clip_id}.bevgt.npz", allow_pickle=True)
    out = {k: z[k] for k in z.files if k != "meta_json"}; out["meta"] = json.loads(str(z["meta_json"]))
    assert out["meta"]["schema"] == "tanitad.lidar_bev_gt/2", out["meta"]["schema"]
    s = out["meta"].get("clip_sha12") or out["meta"].get("source", {}).get("clip_sha12")
    assert s == _sha12(clip_id), f"{clip_id}: LiDAR BEV GT belongs to clip sha12 {s}"
    return out


def load_agents(root, clip_id):
    """pandas DataFrame of the clip's obstacle.offline cuboids (empty when the clip has no upstream tracks)."""
    import pandas as pd
    import pyarrow.parquet as pq
    idx = pd.read_parquet(Path(root) / "agents" / "obstacle_offline" / "index.parquet")
    row = idx[idx["clip_id"] == clip_id]
    if row.empty:
        return pd.DataFrame()
    t = pq.read_table(Path(root) / "agents" / "obstacle_offline" / row.iloc[0]["shard"], filters=[("clip_id", "=", clip_id)])
    return t.to_pandas().drop(columns="clip_id")


def load_calibration(root, clip_id, sensor="camera_front_wide_120fov"):
    """(intrinsics row or None, extrinsics row). Intrinsics: width, height, cx, cy, f-theta fw_poly_0..4 / bw_poly_0..4.
    Extrinsics: sensor pose in the rig frame, quaternion qx qy qz qw + translation x y z (m)."""
    import pandas as pd
    i = pd.read_parquet(Path(root) / "calibration" / "camera_intrinsics.parquet", filters=[("clip_id", "=", clip_id)])
    e = pd.read_parquet(Path(root) / "calibration" / "sensor_extrinsics.parquet", filters=[("clip_id", "=", clip_id)])
    ii = i[i["camera_name"] == sensor]; ee = e[e["sensor_name"] == sensor]
    return (ii.iloc[0] if len(ii) else None), ee.iloc[0]


def _sha12(clip_id):
    """every file stores sha256(clip_id)[:12]; the loaders refuse a file whose name and content disagree"""
    return hashlib.sha256(clip_id.encode()).hexdigest()[:12]


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def verify(root, component, limit=None):
    """recompute sha256 of every local file of a component against its manifest; returns (n_ok, n_bad, n_missing)"""
    root = Path(root)
    if component == "semantic_maps":
        man = json.loads((root / "semantic_maps" / "SEMANTIC_MAPS_MANIFEST.json").read_text())
        files = {p: v for block in ("clips", "clips_flagged") for e in man.get(block, {}).values() for p, v in e["files"].items()}
    elif component == "lidar_bev_gt":
        man = json.loads((root / "lidar_bev_gt" / "LIDAR_BEV_GT_MANIFEST.json").read_text())
        files = {f"lidar_bev_gt/{c}.bevgt.npz": v for c, v in man["files"].items()}
    else:
        man = json.loads((root / "AUG_2026_09_MANIFEST.json").read_text())
        files = {p: v for p, v in man["files"].items() if p.startswith(component + "/")}
    n_ok = n_bad = n_missing = 0
    for p, v in list(files.items())[:limit]:
        f = root / p
        if not f.exists():
            n_missing += 1
        elif f.stat().st_size == v["bytes"] and _sha256(f) == v["sha256"]:
            n_ok += 1
        else:
            n_bad += 1; print("MISMATCH", p)
    print(f"{component}: {n_ok} ok, {n_bad} mismatched, {n_missing} not downloaded (of {len(files)} listed)")
    return n_ok, n_bad, n_missing


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify"); v.add_argument("component"); v.add_argument("--limit", type=int)
    s = sub.add_parser("show"); s.add_argument("clip_id")
    a = ap.parse_args()
    if a.cmd == "verify":
        raise SystemExit(1 if verify(a.root, a.component, a.limit)[1] else 0)
    m = load_semantic_map(a.root, a.clip_id)
    seen = 1 - m["cart"][:, 8].mean(); print(f"semantic map: {len(m['t_query_us'])} frames, cart seen {seen:.3f}, drivable {m['cart'][:, 1].mean():.3f}")
    for name, fn in (("lidar bev", load_lidar_bev), ("agents", load_agents)):
        try:
            r = fn(a.root, a.clip_id); print(name, ":", (len(r) if hasattr(r, "__len__") else r))
        except FileNotFoundError:
            print(name, ": not available for this clip")
