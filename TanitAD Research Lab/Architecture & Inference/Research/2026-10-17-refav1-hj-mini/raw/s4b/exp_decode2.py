import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
sys.path.insert(0, "E:/Projects/TanitAD/stack"); sys.path.insert(0, "E:/Projects/TanitAD/stack/scripts")
import numpy as np, torch, av
import torch.nn.functional as F
import torchvision.io as tvio
torch.set_num_threads(1)
from tanitad.data.physicalai import intrinsics_for_clip
from tanitad.data.calib import CanonicalFrame, cylindrical_rectify

C = "01be5919-6e61-45ba-a193-5f447efb4b51"
os.environ["TANITAD_PAI_INTRINSICS"] = "C:/Users/Admin/tanitad-data/physicalai/calibration/physicalai_front_wide_intrinsics.csv"
intr = intrinsics_for_clip(C, None)
frame = CanonicalFrame.from_hfov(120.0, 256, 640, "cylindrical")
mp4 = f"C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov/{C}.mp4"
banked = torch.load(f"C:/Users/Admin/tanitad-data/refav1-eval141/eps/{C}.v2ep.pt", map_location="cpu", weights_only=False)
lens = banked["jpeg_len"]; offs = torch.cat([torch.zeros(1, dtype=lens.dtype), lens.cumsum(0)])
b0 = tvio.decode_png(banked["jpeg_buf"][offs[0]:offs[1]].clone()).int()

with av.open(mp4) as c:
    st = c.streams.video[0]; st.thread_count = 1
    vf = next(c.decode(st))
    pl = vf.to_ndarray()                     # yuv420p planar [H*3/2, W]
    H, W = vf.height, vf.width
    Y = torch.from_numpy(pl[:H]).float()
    U = torch.from_numpy(pl[H:H + H // 4].reshape(H // 2, W // 2)).float()
    V = torch.from_numpy(pl[H + H // 4:].reshape(H // 2, W // 2)).float()
    sw = vf.to_ndarray(format="rgb24")

def conv(Y, U, V, M="601"):
    y = (Y - 16.0) * (255.0 / 219.0)
    u = (U - 128.0) * (255.0 / 224.0); v = (V - 128.0) * (255.0 / 224.0)
    if M == "601":
        kr, kb = 0.299, 0.114
    else:
        kr, kb = 0.2126, 0.0722
    kg = 1 - kr - kb
    R = y + 2 * (1 - kr) * v
    B = y + 2 * (1 - kb) * u
    G = (y - kr * R - kb * B) / kg
    return torch.stack([R, G, B]).clamp(0, 255)

def up(P, how):
    p = P[None, None]
    if how == "nearest":
        return F.interpolate(p, scale_factor=2, mode="nearest")[0, 0]
    if how == "bilinear_center":
        return F.interpolate(p, scale_factor=2, mode="bilinear", align_corners=False)[0, 0]
    if how == "bicubic_center":
        return F.interpolate(p, scale_factor=2, mode="bicubic", align_corners=False)[0, 0].clamp(0, 255)
    if how == "bilinear_left":      # horizontally co-sited with even luma column, vertically centered
        q = F.pad(p, (0, 1, 0, 0), mode="replicate")
        xs = F.interpolate(q, scale_factor=(1, 2), mode="bilinear", align_corners=True)[..., :W]
        xs = xs[..., :W]
        return F.interpolate(xs, size=(H, W), mode="bilinear", align_corners=False)[0, 0] if False else xs[0, 0].repeat_interleave(2, 0)
    raise ValueError(how)

def rect(rgb_f, rnd):
    t = (rgb_f.round() if rnd == "round" else rgb_f.floor()).clamp(0, 255).to(torch.uint8)[None]
    return cylindrical_rectify(t, intr, frame)[0].int()

# reference: swscale rgb24 as shipped
swt = torch.from_numpy(np.ascontiguousarray(sw)).permute(2, 0, 1)
print("swscale vs float601/bilinear_center RGB (raw, pre-rectify):")
raw = conv(Y, up(U, "bilinear_center"), up(V, "bilinear_center"))
print("  raw diff max", float((raw.round() - swt.float()).abs().max()), "mean", float((raw.round() - swt.float()).abs().mean()))
d = (rect(swt.float(), "round") - b0).abs(); print(f"swscale default            max={int(d.max())} mean={float(d.float().mean()):.4f}")
for M in ("601", "709"):
    for how in ("nearest", "bilinear_center", "bicubic_center", "bilinear_left"):
        rgb = conv(Y, up(U, how), up(V, how), M)
        for rnd in ("round", "floor"):
            d = (rect(rgb, rnd) - b0).abs()
            print(f"M={M} chroma={how:16s} {rnd:5s} max={int(d.max())} mean={float(d.float().mean()):.4f} frac_nz={float((d>0).float().mean()):.4f}", flush=True)
