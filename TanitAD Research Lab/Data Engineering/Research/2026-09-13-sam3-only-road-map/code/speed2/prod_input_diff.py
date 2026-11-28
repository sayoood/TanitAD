"""Why does the production map of a validation clip differ (thin classes IoU ~0.5) from the validation map? Compare the INPUTS the two
sequence builders used for the same clip, source by source: camera timestamps (B1 bundle tar vs data/frontwide parquet), egomotion
(B1 bundle tar vs data/egomotion_alpamayo parquet), and the resulting per-token t_frame and pose (production build_sequence with
interpolation vs the front_native sequence built by build_front_seq_native with nearest-sample poses). CPU only; writes nothing
outside /dev/shm. Usage: prod_input_diff.py <clip id>"""
import io, json, sys, tarfile
from pathlib import Path
import numpy as np
import pandas as pd

clip = sys.argv[1]; c8 = clip[:8]
B1 = Path("/home/nvidia/data/b1-bundle"); D = Path("/home/nvidia/sam3map/data")


def tar_member(tar_path, name):
    with tarfile.open(tar_path) as t:
        return pd.read_parquet(io.BytesIO(t.extractfile(name).read()))


ts_b1 = tar_member(B1 / "timestamps" / "timestamps.tar", f"{clip}.timestamps.parquet")
ts_loc = pd.read_parquet(D / "frontwide" / f"{clip}.timestamps.parquet")
print("timestamps  B1:", ts_b1.shape, list(ts_b1.columns), " local:", ts_loc.shape, list(ts_loc.columns))
a = ts_b1["timestamp"].to_numpy(np.float64); b = ts_loc["timestamp"].to_numpy(np.float64)
n = min(len(a), len(b))
print(f"  first {a[0]:.0f} vs {b[0]:.0f}; identical arrays: {len(a) == len(b) and np.array_equal(a, b)}; median |diff| over {n}: {np.median(np.abs(a[:n] - b[:n])):.1f} us")
eg_b1 = tar_member(B1 / "egomotion" / "egomotion_alpamayo.tar", f"{clip}.parquet").sort_values("timestamp").reset_index(drop=True)
eg_loc = pd.read_parquet(D / "egomotion_alpamayo" / f"{clip}.parquet").sort_values("timestamp").reset_index(drop=True)
print("egomotion   B1:", eg_b1.shape, " local:", eg_loc.shape, " same columns:", list(eg_b1.columns) == list(eg_loc.columns))
m = min(len(eg_b1), len(eg_loc))
for col in ("timestamp", "x", "y", "z", "qz", "qw"):
    if col in eg_b1 and col in eg_loc:
        d = np.abs(eg_b1[col].to_numpy(np.float64)[:m] - eg_loc[col].to_numpy(np.float64)[:m])
        print(f"  {col:9s} max |diff| {d.max():.6g}  median {np.median(d):.6g}")
seq = Path(f"/home/nvidia/sam3map/front_native/seq_{c8}")
poses = json.loads((seq / "poses.json").read_text())
toks = sorted(poses)
metas = [json.loads((seq / t / "meta.json").read_text()) for t in toks]
t_frame_loc = np.array([m_["t_frame_us"] for m_ in metas])
print("front_native tokens:", len(toks), " first token", toks[0], " t_frame first", t_frame_loc[0])
# what production's build_sequence would pick with the B1 timestamps
t_refs = np.arange(a[0] + 0.5e6, a[-1] - 0.5e6, 0.2e6)
pick = np.array([a[int(np.argmin(np.abs(a - t)))] for t in t_refs])
print("production tokens:", len(t_refs), " t_frame first", pick[0], " |t_frame prod - front_native| median/max us:",
      np.median(np.abs(pick[:len(t_frame_loc)] - t_frame_loc[:len(pick)])), np.max(np.abs(pick[:len(t_frame_loc)] - t_frame_loc[:len(pick)])))
# pose difference at the same token instants: front_native pose vs pose from B1 egomotion at the front_native t_frame (nearest, as its builder)
ts_e = eg_b1["timestamp"].to_numpy(np.float64)
dpos = []
for k, t in enumerate(toks):
    T_loc = np.array(poses[t]["T_world_rig"]); tf = t_frame_loc[k]
    i = int(np.argmin(np.abs(ts_e - tf))); p_b1 = eg_b1.loc[i, ["x", "y", "z"]].to_numpy(np.float64)
    dpos.append(np.linalg.norm(T_loc[:3, 3] - p_b1))
print(f"pose position front_native vs B1 egomotion at the same instant: median {np.median(dpos):.3f} m, max {np.max(dpos):.3f} m")
print("ZZINPUTDIFF-DONEZZ")
