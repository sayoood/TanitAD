"""Stage 1: render N frames of the NuRec scene WITHOUT any ISP and bank the raw
linear RGB + alpha + the matching reference frames to a .npz, so the ISP identification
(stage 2) is a 0-GPU analysis that can be iterated without re-rendering."""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "6")
sys.path.insert(0, "/home/nvidia/nurec_work")
import numpy as np, torch, cv2
from gsplat import rasterization
from nurec_loader import (NuRecScene, RigTrajectories, K_for, ftheta_coeffs_for,
                          read_volume_nurec, quat_layout_selftest)

SD = "/home/nvidia/nurec_scenes/sample_set/26.04_release/00040136-e651-4abd-991d-0655ccda9430"
REF = SD + "/camera_front_wide_120fov.mp4"
CAM = "camera_front_wide_120fov"
FRAMES = [0, 150, 300, 450]
OUT = "/home/nvidia/nurec_work/linear_dump.npz"

t0 = time.time()
nre = read_volume_nurec(SD + "/volume.nurec")
print(f"[load] {time.time()-t0:.1f}s version={nre['version']}", flush=True)
scene = NuRecScene(nre, quat_layout="wxyz")
rig = RigTrajectories(SD + "/rig_trajectories.json")
cam = rig.camera(CAM)
W, H = cam.width, cam.height
K = torch.from_numpy(K_for(cam)[None]).cuda()
coeffs = ftheta_coeffs_for(cam)

t_lo = int(scene.sd[".gaussians_nodes.background.time_embed._extra_state"]["timestamps_us_min"])
t_hi = int(scene.sd[".gaussians_nodes.background.time_embed._extra_state"]["timestamps_us_max"])

cap = cv2.VideoCapture(REF)
def ref_frame(i):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, im = cap.read()
    assert ok, i
    im = im[:, :, ::-1].astype(np.float32) / 255.0
    if im.shape[:2] != (H, W):
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
    return im

renders, alphas, refs, taus = [], [], [], []
for fi in FRAMES:
    ts0, ts1 = rig.frame_timestamps_us(CAM, fi)
    tau = (ts1 - t_lo) / float(t_hi - t_lo)
    m, q, s, o, sh = [], [], [], [], []
    for L in ("background", "road"):
        F = scene.fourier_dim(L)
        b = np.zeros(F, np.float32); b[0] = 1.0          # f0 basis (as validated)
        g = scene.gaussians(L, time_basis=b)
        m.append(g.means); q.append(g.quats); s.append(g.scales)
        o.append(g.opacities); sh.append(g.sh)
    cat = lambda xs: torch.from_numpy(np.concatenate(xs)).cuda()
    vm = torch.from_numpy(rig.viewmat(CAM, fi, 1)[None].astype(np.float32)).cuda()
    torch.cuda.synchronize(); t1 = time.time()
    colors, alpha, meta = rasterization(
        means=cat(m), quats=cat(q), scales=cat(s), opacities=cat(o), colors=cat(sh),
        viewmats=vm, Ks=K, width=W, height=H, sh_degree=3, packed=False,
        with_ut=True, with_eval3d=True, camera_model="ftheta", ftheta_coeffs=coeffs,
        near_plane=0.05, far_plane=2000.0)
    torch.cuda.synchronize()
    img = colors[0].cpu().numpy().astype(np.float32)      # UNCLAMPED linear radiance
    a = alpha[0, ..., 0].cpu().numpy().astype(np.float32)
    print(f"[rend] f{fi} tau={tau:.4f} {time.time()-t1:.2f}s  "
          f"linear min={img.min():.4f} max={img.max():.4f} mean={img.mean():.4f} "
          f"alpha_mean={a.mean():.4f}", flush=True)
    renders.append(img); alphas.append(a); refs.append(ref_frame(fi)); taus.append(tau)
cap.release()

np.savez_compressed(OUT, frames=np.array(FRAMES), taus=np.array(taus),
                    render=np.stack(renders), alpha=np.stack(alphas), ref=np.stack(refs))
print(f"[save] {OUT}  {os.path.getsize(OUT)/1e6:.1f} MB  total {time.time()-t0:.1f}s")
