"""Per-frame ego pose (clip-local world <- rig, 4x4) for every sequence frame, from egomotion chunk 768.

Written as <seq>/poses.json keyed by frame token, for the temporal-consistency filter. Nearest egomotion
sample to the frame's t_ref (egomotion runs at ~100 Hz on the same microsecond clock as the cameras);
the |dt| actually used is recorded per frame so a stale pose is visible, not silent.
"""
import io, json, sys, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ZIP = Path(r"C:/Users/Admin/tanitad-data/physicalai/labels/egomotion/egomotion.chunk_0768.zip")


def quat_R(qx, qy, qz, qw):
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw); qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


z = zipfile.ZipFile(ZIP)
for seq in sys.argv[1:]:
    sd = Path(seq)
    toks = sorted(p.name for p in sd.iterdir() if p.is_dir())
    meta0 = json.loads((sd / toks[0] / "meta.json").read_text(encoding="utf-8"))
    clip = meta0["clip_id"]
    df = pd.read_parquet(io.BytesIO(z.read(next(n for n in z.namelist() if clip in n))))
    ts = df["timestamp"].to_numpy(np.float64)
    out, dts = {}, []
    for tok in toks:
        t = json.loads((sd / tok / "meta.json").read_text(encoding="utf-8"))["t_ref_us"]
        i = int(np.argmin(np.abs(ts - t))); dts.append(abs(ts[i] - t) / 1e3)
        r = df.iloc[i]
        T = np.eye(4); T[:3, :3] = quat_R(r.qx, r.qy, r.qz, r.qw); T[:3, 3] = [r.x, r.y, r.z]
        out[tok] = {"T_world_rig": T.tolist(), "dt_ms": round(abs(ts[i] - t) / 1e3, 3)}
    (sd / "poses.json").write_text(json.dumps(out), encoding="utf-8")
    print(sd.name, "frames", len(out), "max |dt| ms", round(max(dts), 2),
          "path length m", round(float(sum(np.linalg.norm(np.diff(np.array([v["T_world_rig"] for v in out.values()])[:, :3, 3], axis=0), axis=1))), 1))
