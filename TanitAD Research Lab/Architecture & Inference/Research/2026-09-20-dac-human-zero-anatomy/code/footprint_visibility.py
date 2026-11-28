"""How often can a sheet show the surface under the car at all? — the S53 failure, quantified.

S53 came back `cannot-tell` because the footprint was entirely BELOW the frame, so panels A and B
carried no overlay. That is an evidence property of the pack, not a one-off: a stationary or
crawling manoeuvre keeps every footprint within a few metres of the bumper, and the camera cannot
see its own wheels. This measures it across both packs so the next design can price the fix.

⛔ Geometry only — no frames are decoded and no map is read.

    python footprint_visibility.py --keys <key.json> [--keys ...] --out <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dac_anatomy import DEFAULTS, _import_harness
from dac_project import controls


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k, default=v)
    ap.add_argument("--keys", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extrinsics",
                    default="D:/Projects/TanitAD-artifacts/refcv5v2_final/extrinsics141.json")
    a = ap.parse_args()
    S1, P, SMG = _import_harness()
    import refc_v3_train as V3
    from tanitad.data.rig_projection import RigCamera
    frame = V3._agent_cam_frames()[(416, 1024)]
    _, table = V3._read_rig_extrinsics(a.extrinsics)
    H, W = frame.height, frame.width

    want: dict = {}
    for kp in a.keys:
        key = json.loads(Path(kp).read_text(encoding="utf-8"))
        tag = Path(kp).stem
        for r in key["rows"]:
            want[(r["sha12"], r["t0"])] = {"pack": tag, "blind_id": r["blind_id"],
                                           "source": r.get("source", r.get("stratum", ""))}
    corp = S1.Corpus(a.config, a.cache, a.labels, a.agents, a.maps, lru=2)
    rows, ctrl = [], None
    for wi in S1.trainer_windows(corp.ds, 1000):
        if corp.eligibility(wi) is not None:
            continue
        e_i, t = corp.ds.index[wi]
        cid = str(corp.clip_ids[e_i])
        k = (S1.sha12(cid), int(t + corp.W - 1))
        if k not in want:
            continue
        it = corp.light_item(wi)
        corners = P._ego_boxes(it["human"][None], P.PROXY)[0].numpy()
        cam = RigCamera.from_extrinsics(table[cid], frame)
        if ctrl is None:
            ctrl = controls(cam, frame, corners[0])
            if not ctrl["all_pass"]:
                raise SystemExit(f"⛔ a projection control missed: {ctrl}")
        pts = torch.tensor(np.concatenate([corners, np.zeros(corners.shape[:-1] + (1,))], -1),
                           dtype=torch.float64).reshape(-1, 3)
        col, row, _ = cam.project(pts)
        zc = cam.to_cam(pts)[..., 2].numpy()
        sh = corners.shape[:-1]
        col, row = col.numpy().reshape(sh), row.numpy().reshape(sh)
        front = (zc > 0).reshape(sh)
        vis_corner = front & (col > 0) & (col < W) & (row > 0) & (row < H)
        inside = vis_corner.all(axis=1)
        # ⚠️ two different failures: no COMPLETE footprint (the box is not drawn) and no corner
        # at all (the sheet carries no overlay whatsoever — S53's case).
        rows.append({**want[k], "sha12": k[0], "t0": k[1],
                     "ticks_any_corner_visible": int(vis_corner.any(axis=1).sum()),
                     "ticks_fully_visible": int(inside.sum()),
                     "first_visible_tick": int(np.argmax(inside)) if inside.any() else None,
                     "path_length_m": round(float(corners[..., 0].max() - corners[..., 0].min()), 2),
                     "min_row_of_nearest_box": round(float(row[0].min()), 1),
                     "image_height": H})
    by_pack: dict = {}
    for r in rows:
        d = by_pack.setdefault(r["pack"], {"n": 0, "no_visible_footprint": 0, "ids_blank": [],
                                           "path_len_when_blank": []})
        d["n"] += 1
        if r["ticks_fully_visible"] == 0:
            d["no_visible_footprint"] += 1
            d["ids_blank"].append(r["blind_id"])
            d["path_len_when_blank"].append(r["path_length_m"])
    blank = [r for r in rows if r["ticks_any_corner_visible"] == 0]
    short = [r for r in rows if r["path_length_m"] < 5.0]
    out = {"_what": "can a sheet show the surface under the car? ticks whose whole footprint "
                    "falls inside the frame",
           "_evidence_class": "MEASURED (ours; geometry only)",
           "windows": len(rows), "by_pack": by_pack,
           "no_visible_footprint_total": sum(1 for r in rows if r["ticks_fully_visible"] == 0),
           "no_corner_at_all_total": {"n": len(blank),
                                      "ids": [r["blind_id"] for r in blank],
                                      "path_length_m": [r["path_length_m"] for r in blank]},
           "short_paths_under_5_m": {"n": len(short),
                                     "of_which_no_visible_footprint":
                                         sum(1 for r in short if r["ticks_fully_visible"] == 0)},
           "median_ticks_fully_visible": float(np.median([r["ticks_fully_visible"] for r in rows])),
           "projection_controls": ctrl,
           "rows": sorted(rows, key=lambda r: r["ticks_fully_visible"])}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: out[k] for k in ("windows", "by_pack", "no_visible_footprint_total",
                                          "short_paths_under_5_m",
                                          "median_ticks_fully_visible")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
