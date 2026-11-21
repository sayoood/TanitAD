#!/usr/bin/env python3
"""build_stopaux_extr_table.py (MM, 2026-11-12): the per-clip front-wide mount table that arm A's `--stop-aux-extr` reads, over the v8F-mini
TRAIN JOIN's clips (box 2 D:\\tanitad\\runs\\v8f_mini_full\\join_train.npz, 4,572 clips, sorted-id sha256 e9bb9ef8...).

Why not stack/scripts/build_rig_extrinsics_table.py as it stands: its clip -> chunk map is the union of r0/phase0_selection.parquet and
r0/r0_selection.parquet (3,405 clips), which holds only 261 of the join's 4,572 clips (MEASURED 2026-11-12). This wrapper keeps the builder's
row reader (`_front_wide_rows`, imported from the builder itself) and its output format, and takes the chunk map from the corpus-wide
clip_index.parquet (306,152 clips) instead. Same source (PhysicalAI-AV calibration/sensor_extrinsics parquets), nothing fitted or assumed.

Refuses unless EVERY requested clip resolves (a partial table would let the trainer's 99 % coverage gate pass on a silent hole).

    python build_stopaux_extr_table.py --root C:/Users/Admin/tanitad-data/physicalai --builder-dir <stack/scripts of the tree> \
        --clips-txt b2_clips.txt --out stopaux_extr_join4572.json
"""
import argparse
import hashlib
import json
import math
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--builder-dir", required=True, help="the stack/scripts folder holding build_rig_extrinsics_table.py")
    ap.add_argument("--clips-txt", required=True, help="lines 'C <clip_id>' (the join's clip array, in join order)")
    ap.add_argument("--expect-sha", default="e9bb9ef87be20a4e606f21edbfc0165107521d654eb54b0271771241b4cdd18c")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, a.builder_dir)
    import pandas as pd
    import build_rig_extrinsics_table as B

    want = sorted({l[2:].strip() for l in open(a.clips_txt, encoding="utf-8") if l.startswith("C ")})
    digest = hashlib.sha256("\n".join(want).encode("utf-8")).hexdigest()
    if digest != a.expect_sha:
        raise SystemExit(f"[stopaux-extr] REFUSING: clip set digest {digest} != expected {a.expect_sha} (not the train join's clip set)")
    ci = pd.read_parquet(os.path.join(a.root, "clip_index.parquet"))
    if "clip_id" not in ci.columns:
        ci = ci.reset_index()
    chunk_of = dict(zip(ci["clip_id"].astype(str), ci["chunk"].astype(int)))
    by_chunk = {}
    for c in want:
        if c not in chunk_of:
            raise SystemExit(f"[stopaux-extr] REFUSING: clip {c} is not in clip_index.parquet")
        by_chunk.setdefault(chunk_of[c], []).append(c)
    out, missing = {}, []
    cal = os.path.join(a.root, "calibration", "sensor_extrinsics")
    for k in sorted(by_chunk):
        p = os.path.join(cal, "sensor_extrinsics.chunk_%04d.parquet" % k)
        if not os.path.exists(p):
            missing.extend(by_chunk[k])
            continue
        have = {}
        for r in B._front_wide_rows(pd.read_parquet(p)).itertuples(index=False):
            have[str(r.clip_id)] = dict(qx=float(r.qx), qy=float(r.qy), qz=float(r.qz), qw=float(r.qw), x=float(r.x), y=float(r.y), z=float(r.z))
        for c in by_chunk[k]:
            (out.__setitem__(c, have[c]) if c in have else missing.append(c))
    if missing:
        raise SystemExit(f"[stopaux-extr] REFUSING: {len(missing)} clips have no front-wide row (e.g. {missing[:3]})")
    bad = [c for c, v in out.items() if not all(math.isfinite(v[q]) for q in v)]
    if bad:
        raise SystemExit(f"[stopaux-extr] REFUSING: non-finite pose for {len(bad)} clips (e.g. {bad[:3]})")
    body = json.dumps(out, indent=1)
    open(a.out, "w", encoding="utf-8", newline="\n").write(body)
    zs = sorted(v["z"] for v in out.values())
    xs = sorted(v["x"] for v in out.values())
    stats = {"n_clips_requested": len(want), "n_clips_written": len(out), "n_clips_missing": 0, "n_chunks": len(by_chunk),
             "clip_id_sha256_sorted": digest, "join": "box 2 D:\\tanitad\\runs\\v8f_mini_full\\join_train.npz (4,572 clips; train = minus --exclude 540:600 by sha12 = 4,512)",
             "height_m": {"min": zs[0], "median": zs[len(zs) // 2], "max": zs[-1], "n_distinct": len(set(round(z, 6) for z in zs))},
             "forward_offset_m": {"min": xs[0], "max": xs[-1]}, "sensor": B.SENSOR,
             "source": "PhysicalAI-AV calibration/sensor_extrinsics parquets; chunk map from clip_index.parquet",
             "table_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest()}
    open(a.out + ".stats.json", "w", encoding="utf-8", newline="\n").write(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
