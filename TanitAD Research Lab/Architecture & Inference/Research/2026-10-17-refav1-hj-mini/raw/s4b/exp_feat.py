import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys, json, time
sys.path.insert(0, "E:/Projects/TanitAD/stack"); sys.path.insert(0, "E:/Projects/TanitAD/stack/scripts")
import numpy as np, pandas as pd, torch, av
import torch.nn.functional as F
import torchvision.io as tvio
torch.set_num_threads(10)
assert not torch.cuda.is_available()
from tanitad.data.physicalai import intrinsics_for_clip, TARGET_HZ
from tanitad.data.calib import CanonicalFrame, cylindrical_rectify
from transformers import AutoModel, AutoImageProcessor

MID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
proc = AutoImageProcessor.from_pretrained(MID, local_files_only=True)
model = AutoModel.from_pretrained(MID, local_files_only=True, dtype=torch.float32).eval()
mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1); std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
os.environ["TANITAD_PAI_INTRINSICS"] = "C:/Users/Admin/tanitad-data/physicalai/calibration/physicalai_front_wide_intrinsics.csv"
frame = CanonicalFrame.from_hfov(120.0, 256, 640, "cylindrical")
ST = "C:/Users/Admin/tanitad-data/refav1-train600-256_stage/_valroot/out"
EV = "C:/Users/Admin/tanitad-data/refav1-eval141/eps"
CL = ["01be5919-6e61-45ba-a193-5f447efb4b51", "060531e1-9935-482e-8c25-2abf37bdc41b"]
IDX = [0, 40, 100, 160, 200]


def pngs(d, idx):
    lens = d["jpeg_len"]; offs = torch.cat([torch.zeros(1, dtype=lens.dtype), lens.cumsum(0)])
    return torch.stack([tvio.decode_png(d["jpeg_buf"][offs[j]:offs[j + 1]].clone()) for j in idx])


def feats(x_u8):
    with torch.inference_mode():
        x = x_u8.float() / 255.0
        h = model(pixel_values=(x - mean) / std).last_hidden_state[:, 5:, :]
    return h


def pool(f, gh, gw):
    T, N, D = f.shape
    return f.reshape(T, gh, 16 // gh, gw, 40 // gw, D).mean(dim=(2, 4)).reshape(T, gh * gw, D)


def metrics(a, b):
    cos = F.cosine_similarity(a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1]), dim=-1).mean().item()
    rl2 = ((a - b).norm() / b.norm()).item()
    return cos, rl2


def conv601_nearest(Y, U, V):
    y = (Y - 16.0) * (255.0 / 219.0)
    u = (U - 128.0) * (255.0 / 224.0); v = (V - 128.0) * (255.0 / 224.0)
    u = u.repeat_interleave(2, 0).repeat_interleave(2, 1); v = v.repeat_interleave(2, 0).repeat_interleave(2, 1)
    kr, kb = 0.299, 0.114; kg = 1 - kr - kb
    R = y + 2 * (1 - kr) * v; B = y + 2 * (1 - kb) * u; G = (y - kr * R - kb * B) / kg
    return torch.stack([R, G, B]).clamp(0, 255)


def emulate(clip, idx_ep):
    mp4 = f"C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov/{clip}.mp4"
    ts = pd.read_parquet(mp4.replace(".mp4", ".timestamps.parquet"))
    tcol = next(c for c in ts.columns if "time" in c.lower()); tf = ts[tcol].to_numpy(np.float64)
    span = tf[-1] - tf[0]; unit = 1.0
    for cand in (1e9, 1e6, 1e3):
        if span / cand > 1.0:
            unit = cand; break
    nt = max(int(span / unit * TARGET_HZ), 4)
    tq = np.linspace(tf[0], tf[-1], nt)
    fi = np.searchsorted(tf, tq).clip(0, len(tf) - 1)
    need = {int(fi[j]): j for j in idx_ep}
    intr = intrinsics_for_clip(clip, None)
    out = {}
    with av.open(mp4) as c:
        st = c.streams.video[0]; st.thread_count = 2
        for n, vf in enumerate(c.decode(st)):
            if n in need:
                pl = vf.to_ndarray(); H, W = vf.height, vf.width
                Y = torch.from_numpy(pl[:H]).float()
                U = torch.from_numpy(pl[H:H + H // 4].reshape(H // 2, W // 2)).float()
                V = torch.from_numpy(pl[H + H // 4:].reshape(H // 2, W // 2)).float()
                rgb = conv601_nearest(Y, U, V).floor().clamp(0, 255).to(torch.uint8)[None]
                out[need[n]] = cylindrical_rectify(rgb, intr, frame)[0]
                if len(out) == len(need):
                    break
    return torch.stack([out[j] for j in idx_ep])


res = {}
for clip in CL:
    banked = torch.load(f"{EV}/{clip}.v2ep.pt", map_location="cpu", weights_only=False)
    mine = torch.load(f"{ST}/{clip}.v2ep.pt", map_location="cpu", weights_only=False)
    fb, fm = pngs(banked, IDX), pngs(mine, IDX)
    t0 = time.time(); fe = emulate(clip, IDX); print("emulate", time.time() - t0, flush=True)
    vars_ = {"swscale18(builder)": fm, "bt601_nearest_floor": fe}
    Fb = feats(fb)
    print(f"{clip[:8]} banked feats done", flush=True)
    for name, x in vars_.items():
        px = (x.int() - fb.int()).abs()
        Fx = feats(x)
        c_full, r_full = metrics(Fx, Fb)
        c_p, r_p = metrics(pool(Fx, 4, 10), pool(Fb, 4, 10))
        res[f"{clip[:8]}/{name}"] = {"pixel_mae": float(px.float().mean()), "pixel_max": int(px.max()),
                                     "full_cos": c_full, "full_relL2": r_full, "pool4x10_cos": c_p, "pool4x10_relL2": r_p}
        print(clip[:8], name, json.dumps(res[f"{clip[:8]}/{name}"]), flush=True)
json.dump(res, open("C:/Users/Admin/tanitad-data/refav1-train600-256_stage/feat_residue.json", "w"), indent=1)
print("ZZFEAT-DONEZZ")
