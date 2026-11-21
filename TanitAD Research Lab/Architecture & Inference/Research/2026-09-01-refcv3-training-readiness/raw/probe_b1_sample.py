"""Probe one pulled B1 *.v2ep.pt sample: keys, codec, n_stack, geometry,
poses/actions shapes, frame dict — MEASURED evidence for the readiness audit."""
import json
import torch

P = r"C:/Users/Admin/refav1_probe/eps/001d413e-2803-4c6b-a11a-8f7793a3534e.v2ep.pt"
d = torch.load(P, map_location="cpu", weights_only=False, mmap=True)
out = {"keys": sorted(d.keys())}
for k in ("n_stack", "codec", "image_size", "image_h", "image_w", "clip_id",
          "episode_id"):
    if k in d:
        v = d[k]
        out[k] = v if isinstance(v, (str, int, float)) else repr(v)
out["frame"] = d.get("frame")
for k in ("poses", "actions", "jpeg_len", "jpeg_buf"):
    if k in d:
        t = d[k]
        out[f"{k}.shape"] = list(t.shape)
        out[f"{k}.dtype"] = str(t.dtype)
out["n_frames_encoded"] = int(d["jpeg_len"].numel())
out["T_poses"] = int(d["poses"].shape[0])

# decode frame 0 to confirm codec + true pixel geometry
import torchvision.io as tvio
buf = d["jpeg_buf"]
ln = d["jpeg_len"]
b0 = buf[: int(ln[0])]
dec = tvio.decode_png if str(d.get("codec", "jpeg")) == "png" else tvio.decode_jpeg
f0 = dec(b0, mode=tvio.ImageReadMode.RGB)
out["frame0.shape"] = list(f0.shape)
out["frame0.dtype"] = str(f0.dtype)
out["frame0.mean"] = round(float(f0.float().mean()), 3)
out["frame0.nonzero"] = bool((f0 > 0).any())
out["magic_bytes"] = [int(x) for x in b0[:4]]

# stacked contract check via the actual loader
import sys
sys.path.insert(0, r"C:/Users/Admin/tanitad-wt/stack")
from tanitad.data.v2_dataset import build_v2_providers
provs = build_v2_providers([r"C:/Users/Admin/refav1_probe/eps"], lru_size=2,
                           verbose=False)
out["n_providers"] = len(provs)
ep = provs[0]
out["ep.frames.shape"] = list(ep.frames.shape)
out["ep.poses.shape"] = list(ep.poses.shape)
out["ep.actions.shape"] = list(ep.actions.shape)
w = ep.frames[0:2]
out["ep.frames[0:2].shape"] = list(w.shape)
out["ep.frames[0:2].dtype"] = str(w.dtype)
out["ep.frames[0:2].mean"] = round(float(w.float().mean()), 3)
print(json.dumps(out, indent=1, default=str))
