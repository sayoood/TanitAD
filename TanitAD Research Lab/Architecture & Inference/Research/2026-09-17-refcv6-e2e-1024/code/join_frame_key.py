"""WHICH index space is ``AgentJoin3D`` keyed on -- RAW v2ep, or EPISODE?

The refcv6 box head reaches its z/h through ``agent_cuboid_gt.zh_for_frame
(clip_id, frame, track_ids)``. Two index spaces exist on this corpus and they
differ by ``n_stack - 1 = 2`` frames (~0.2 s):

* the **RAW** v2ep index ``i in [0, n_target)`` -- what ``MapGTStore.raw_frames``
  returns, and what ``build_b1_agent_join.py``'s docstring line 1 says the
  ``frame`` key is;
* the **EPISODE** index ``i - (n_stack - 1)`` -- what the same builder emits
  under the DIFFERENT name ``frame_idx``, and what
  ``refc_v3_train.py:2172`` passes to the trainer's own 2-D join reader
  (``self._agent_item(ep, t + w - 1)``).

⛔ A docstring is a claim; this is a measurement. The discriminator is
INDEPENDENT of both readers: the obstacle PARQUET carries ``center_z`` itself
and is reached by TIME (``cuboids_at_time(obs, t_s)``), never by a frame index.
So for each candidate offset we ask *"does the join's cz agree with the
parquet's cz for the same tracks?"* and read off the minimum.

⚠️ The 0-offset residual is NOT zero (0.0123 m): the two files pick their
nearest sample by slightly different rounding, which the 3-D builder's own C7
control measures. The claim is the SEPARATION, not a zero.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

WT = Path(os.environ.get("TANITAD_WT",
                         Path(__file__).resolve().parents[5]))
sys.path.insert(0, str(WT / "stack"))

from tanitad.data.agent_cuboid_gt import (  # noqa: E402
    cuboids_at_time, open_join3d, read_clip_cuboids, track_ids_at_time,
    zh_for_frame,
)
from tanitad.data.perception_targets import MapGTStore  # noqa: E402
from tanitad.data.semantic_map_gt import sha12  # noqa: E402

MAP_DIR = os.environ.get("TANITAD_SAM3_MAP_DIR",
                         "D:/Projects/TanitAD-artifacts/sam3-maps-eval")
JOIN = os.environ.get("TANITAD_AGENT_JOIN3D",
                      "D:/Projects/TanitAD-artifacts/b1-agent-join-3d-"
                      "20260917/b1eval_agents_3d.jsonl.xz")
OBST = os.environ.get("TANITAD_OBSTACLE_DIR",
                      "C:/Users/Admin/tanitad-data/physicalai/labels/"
                      "obstacle_offline_b1eval")
OFFSETS = range(-3, 4)
WINDOWS = (7, 30, 60, 90, 120, 150)


def main(n_clips: int = 12):
    maps = sorted(Path(MAP_DIR).glob("*.sam3mapgt.npz"))
    parq = {p.stem: p for p in Path(OBST).glob("*.parquet")}
    clips = [c for c in parq if sha12(c) in {m.name[:-len(".sam3mapgt.npz")]
                                             for m in maps}][:n_clips]
    if len(clips) < 3:
        raise SystemExit("need >= 3 clips joining map+parquet")
    j3 = open_join3d(JOIN, clips=clips)
    if j3 is None:
        raise SystemExit("no 3-D join at %s" % JOIN)
    store = MapGTStore(MAP_DIR, max_open=2)
    per_off = {o: [] for o in OFFSETS}
    n_pairs = n_masked = 0
    for cid in clips:
        obs = read_clip_cuboids(str(parq[cid]))
        for w in WINDOWS:
            mf = store.frames_for_windows(cid, [w], n_stack=3)
            raw = int(mf.frame_idx[0])
            t_s = float(mf.t_img_us[0]) / 1e6
            cub = cuboids_at_time(obs, t_s, clip_id=cid)
            tids = track_ids_at_time(obs, t_s)
            if cub.shape[0] == 0:
                continue
            n_pairs += 1
            for o in OFFSETS:
                cz, _h, m = zh_for_frame(cid, raw + o, tids, join3d=j3)
                if m.sum() == 0:
                    continue
                per_off[o].append(
                    float(np.abs(np.asarray(cz)[m] - cub[m, 6]).mean()))
                if o == 0:
                    n_masked += int((~m).sum())
    res = {
        "evidence_class": "MEASURED",
        "instrument": "join cz vs PARQUET center_z, parquet reached by TIME",
        "n_clips": len(clips), "clip_sha12": [sha12(c) for c in clips],
        "n_window_probes": n_pairs,
        "window_indices": list(WINDOWS),
        "mean_abs_dcz_by_offset_from_RAW": {
            str(o): (float(np.mean(v)) if v else None) for o, v in
            per_off.items()},
        "n_samples_by_offset": {str(o): len(v) for o, v in per_off.items()},
        "n_tracks_masked_at_offset0": n_masked,
    }
    got = {o: (np.mean(v) if v else np.inf) for o, v in per_off.items()}
    best = min(got, key=got.get)
    res["argmin_offset"] = int(best)
    res["separation_x"] = float(
        min(v for o, v in got.items() if o != best) / got[best])
    res["verdict"] = ("AgentJoin3D is keyed on the RAW v2ep frame index"
                      if best == 0 else
                      "AgentJoin3D is keyed on RAW%+d -- NOT what the "
                      "builder docstring says" % best)
    out = Path(__file__).resolve().parents[1] / "raw" / "join_frame_key.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
