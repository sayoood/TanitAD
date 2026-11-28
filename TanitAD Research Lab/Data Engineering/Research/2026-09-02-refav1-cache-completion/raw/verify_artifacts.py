"""Header + content verification of (a) the Sep-1 kept-local fp8 artifact,
(b) one fp16 gate-cache entry, (c) the pulled v2ep source. Read-only."""
import json
import sys
import zipfile
from pathlib import Path

import torch

SHIP = Path("C:/Users/Admin/refav1_probe/ship")
CACHE = Path("C:/Users/Admin/refav1_probe/dinov3cache")
REB = Path("C:/Users/Admin/refav1_probe/rebuild")
CLIP = "16d325e9-dbfe-439c-a0ed-4ff8850ad2e2"
rep = {}

# (a) the kept-local fp8 artifact from the original GPU build (Sep 1 03:17)
p = SHIP / f"{CLIP}.pt"
a = {"path": str(p), "bytes": p.stat().st_size}
try:
    z = zipfile.ZipFile(p)
    a["zip_entries"] = [(i.filename, i.file_size) for i in z.infolist()]
    a["testzip"] = z.testzip()
except Exception as e:  # noqa: BLE001
    a["zip_error"] = f"{type(e).__name__}: {e}"
x = torch.load(p, map_location="cpu", weights_only=True)
a["dtype"] = str(x.dtype)
a["shape"] = list(x.shape)
xf = x.float()
a["finite_all"] = bool(torch.isfinite(xf).all())
a["mean_abs"] = float(xf.abs().mean())
pf = xf.abs().mean(dim=(1, 2))
a["per_frame_mean_abs_min"] = float(pf.min())
a["per_frame_mean_abs_max"] = float(pf.max())
a["frac_exact_zero"] = float((xf == 0).float().mean())
a["payload_bytes_expected"] = int(x.numel())
rep["sep01_local_fp8"] = a
print("[a] sep-01 local fp8:", json.dumps(a, indent=1))

# (b) one fp16 gate-cache entry (what fp8_l2_gate.py reads as the fp16 arm)
names = sorted(q for q in CACHE.glob("*.pt"))
q = names[0]
y = torch.load(q, map_location="cpu", weights_only=True, mmap=True)
b = {"path": str(q), "dtype": str(y.dtype), "shape": list(y.shape),
     "n_entries_pt": len(names), "index_json": (CACHE / "index.json").exists()}
rep["gate_cache_entry"] = b
print("[b] gate cache entry:", json.dumps(b, indent=1))

# (c) the v2ep source header (mmap: pages in only the small tensors)
ep = REB / f"{CLIP}.v2ep.pt"
d = torch.load(ep, map_location="cpu", weights_only=False, mmap=True)
c = {"path": str(ep), "bytes": ep.stat().st_size, "keys": sorted(d.keys())}
for k in ("codec", "n_stack", "clip_id", "image_size", "image_h", "image_w",
          "quality", "episode_id", "projection_mode", "frame", "skip_hash"):
    if k in d:
        v = d[k]
        c[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
c["T_poses"] = int(d["poses"].shape[0])
c["poses_shape"] = list(d["poses"].shape)
c["actions_shape"] = list(d["actions"].shape)
c["n_frames_encoded"] = int(d["jpeg_len"].shape[0])
c["jpeg_len_sum"] = int(d["jpeg_len"].to(torch.int64).sum())
c["jpeg_buf_bytes"] = int(d["jpeg_buf"].numel())
c["jpeg_len_min"] = int(d["jpeg_len"].min())
c["jpeg_len_max"] = int(d["jpeg_len"].max())
c["png_magic_first_frame"] = d["jpeg_buf"][:8].tolist()
c["T_cache_expected_ceil_T_over_2"] = -(-c["T_poses"] // 2)
c["T_cache_expected_builder_range_0_T_2"] = len(range(0, c["n_frames_encoded"], 2))
c["poses_finite"] = bool(torch.isfinite(d["poses"]).all())
c["actions_finite"] = bool(torch.isfinite(d["actions"]).all())
c["speed_min_max"] = [float(d["poses"][:, 3].min()), float(d["poses"][:, 3].max())]
rep["v2ep_header"] = c
print("[c] v2ep header:", json.dumps(c, indent=1))

(REB / "verify_artifacts.json").write_text(json.dumps(rep, indent=1))
print("banked ->", REB / "verify_artifacts.json")
