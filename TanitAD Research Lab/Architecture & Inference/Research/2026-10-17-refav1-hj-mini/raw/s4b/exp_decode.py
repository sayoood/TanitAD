import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys, time
sys.path.insert(0, "E:/Projects/TanitAD/stack"); sys.path.insert(0, "E:/Projects/TanitAD/stack/scripts")
import numpy as np, torch, av
import torchvision.io as tvio
torch.set_num_threads(1)
from tanitad.data.physicalai import intrinsics_for_clip
from tanitad.data.calib import CanonicalFrame, cylindrical_rectify
print("av", av.__version__, "torch", torch.__version__)
try:
    print("libs", av.library_versions)
except Exception as e:
    print("libs?", e)

C = "01be5919-6e61-45ba-a193-5f447efb4b51"
os.environ["TANITAD_PAI_INTRINSICS"] = "C:/Users/Admin/tanitad-data/physicalai/calibration/physicalai_front_wide_intrinsics.csv"
intr = intrinsics_for_clip(C, None)
frame = CanonicalFrame.from_hfov(120.0, 256, 640, "cylindrical")
mp4 = f"C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov/{C}.mp4"
banked = torch.load(f"C:/Users/Admin/tanitad-data/refav1-eval141/eps/{C}.v2ep.pt", map_location="cpu", weights_only=False)
lens = banked["jpeg_len"]; offs = torch.cat([torch.zeros(1, dtype=lens.dtype), lens.cumsum(0)])
b0 = tvio.decode_png(banked["jpeg_buf"][offs[0]:offs[1]].clone()).int()
print("banked f0", b0.shape, int(b0.max()))

with av.open(mp4) as c:
    st = c.streams.video[0]
    st.thread_count = 1
    vf = next(c.decode(st))
    print("frame fmt", vf.format.name, vf.width, vf.height, "colorspace", vf.colorspace, "range", vf.color_range)
    variants = {}
    variants["default_rgb24"] = lambda: vf.to_ndarray(format="rgb24")
    for interp in ("BILINEAR", "BICUBIC", "POINT", "AREA", "FAST_BILINEAR", "LANCZOS", "SPLINE", "GAUSS", "X"):
        variants[f"rgb24_{interp}"] = (lambda i=interp: vf.reformat(format="rgb24", interpolation=i).to_ndarray())
    for cs in ("ITU601", "ITU709", "SMPTE240M", "DEFAULT"):
        variants[f"rgb24_cs_{cs}"] = (lambda s=cs: vf.reformat(format="rgb24", src_colorspace=s).to_ndarray())
    variants["rgb24_range_mpeg"] = lambda: vf.reformat(format="rgb24", src_color_range="MPEG").to_ndarray()
    variants["rgb24_range_jpeg"] = lambda: vf.reformat(format="rgb24", src_color_range="JPEG").to_ndarray()
    variants["bgr24_flip"] = lambda: vf.to_ndarray(format="bgr24")[..., ::-1].copy()
    res = {}
    for name, fn in variants.items():
        try:
            arr = fn()
        except Exception as e:
            print(f"{name:22s} ERR {type(e).__name__}: {str(e)[:60]}")
            continue
        t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)[None]
        out = cylindrical_rectify(t, intr, frame)[0].int()
        d = (out - b0).abs()
        print(f"{name:22s} max={int(d.max())} mean={float(d.float().mean()):.4f} frac_nz={float((d>0).float().mean()):.4f}", flush=True)
