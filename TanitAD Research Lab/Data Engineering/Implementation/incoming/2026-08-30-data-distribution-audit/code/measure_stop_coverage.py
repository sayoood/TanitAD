"""Stop-behaviour coverage on the window the model ACTUALLY sees.

⛔ THE WINDOW IS THE WHOLE POINT. The egomotion parquet spans a median 139.3 s;
the camera clip is 20 s. Measuring stop behaviour over the parquet's full span
counts ~7x more driving than training ever shows, and it makes stop-launch
coverage look comfortable (33.3 %) when on the trainable window it is 11.8 %.

Uses the programme's OWN detector and constants (ego_manoeuvre.stop_episodes,
V_STOP_MS, V_CRUISE_MS) rather than reinventing thresholds.
"""
import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack")
from tanitad.data.ego_manoeuvre import (V_CRUISE_MS, V_STOP_MS,  # noqa: E402
                                        stop_episodes)

REL = Path("C:/Users/Admin/tanitad-wt/_s2build/release/tanitad-v7-training-corpus")
OUT = Path(__file__).resolve().parent
CLIP_S = 20.0                      # the camera clip, i.e. the trainable window


def scan(window: bool):
    tf = tarfile.open(REL / "egomotion" / "egomotion_alpamayo.tar")
    rows = []
    for m in tf.getmembers():
        if not m.isfile():
            continue
        d = pd.read_parquet(io.BytesIO(tf.extractfile(m).read()),
                            columns=["vx", "vy", "timestamp"])
        t = d.timestamp.to_numpy() / 1e6
        v = np.hypot(d.vx.to_numpy(), d.vy.to_numpy())
        if window:
            v = v[(t >= 0.0) & (t <= CLIP_S)]
        if len(v) < 10:
            continue
        eps = stop_episodes(v)
        launch = any(j + 1 < len(v) and v[j + 1:].max() >= V_CRUISE_MS for i, j in eps)
        rows.append({"clip": m.name.split(".")[0], "n": len(v),
                     "stopped_frac": float(np.mean(v < V_STOP_MS)),
                     "has_stop": len(eps) > 0, "has_stop_launch": launch,
                     "stop_and_go": len(eps) >= 2})
    return pd.DataFrame(rows)


for window, label in ((True, f"0-{CLIP_S:.0f}s TRAINABLE WINDOW"),
                      (False, "FULL egomotion span (NOT what the model sees)")):
    df = scan(window)
    print(f"\n=== {label} — n={len(df)} ===")
    print(f"  stopped-frame mean {df.stopped_frac.mean()*100:5.1f}%  "
          f"median {df.stopped_frac.median()*100:.1f}%")
    print(f"  NO stopped frames  {(df.stopped_frac==0).mean()*100:5.1f}%")
    print(f"  any stop           {df.has_stop.mean()*100:5.1f}%")
    print(f"  STOP->LAUNCH       {df.has_stop_launch.mean()*100:5.1f}%  "
          f"({df.has_stop_launch.sum()} clips)")
    print(f"  stop-and-go        {df.stop_and_go.mean()*100:5.1f}%  "
          f"({df.stop_and_go.sum()} clips)")
    df.to_parquet(OUT / ("b1_stop_coverage_20s.parquet" if window
                         else "b1_stop_coverage.parquet"), index=False)
