import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
sys.path.insert(0, "E:/Projects/TanitAD/stack")
import numpy as np, torch, av
import torch.nn.functional as F
from tanitad.data.physicalai import intrinsics_for_clip
from tanitad.data.calib import CanonicalFrame, cylindrical_rectify, cylindrical_grid
torch.set_num_threads(1)
C = "01be5919-6e61-45ba-a193-5f447efb4b51"
os.environ["TANITAD_PAI_INTRINSICS"] = "C:/Users/Admin/tanitad-data/physicalai/calibration/physicalai_front_wide_intrinsics.csv"
intr = intrinsics_for_clip(C, None); frame = CanonicalFrame.from_hfov(120.0, 256, 640, "cylindrical")
with av.open(f"C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov/{C}.mp4") as c:
    vf = next(c.decode(c.streams.video[0]))
    t = torch.from_numpy(vf.to_ndarray(format="rgb24")).permute(2, 0, 1)[None]
o32 = cylindrical_rectify(t, intr, frame)[0].int()
grid, mask = cylindrical_grid(intr, 1080, 1920, frame, device=t.device)
o64 = F.grid_sample(t.double(), grid.double(), mode="bilinear", padding_mode="zeros", align_corners=False) * mask.double()
o64 = o64.clamp(0, 255).floor().to(torch.uint8)[0].int()
d = (o32 - o64).abs()
print("float32 vs float64 grid_sample: max", int(d.max()), "frac_nonzero", float((d > 0).float().mean()))
