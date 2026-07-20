"""Quantify the true-horizon row for comma vs physicalai canonical (256px) frames.

For ~N straight-driving val windows per corpus:
  * extract the anchor (latest) RGB frame [256,256,3] from the epcache,
  * estimate the ground vanishing-point row two independent ways:
      (VP)  road-edge vanishing point via a pure-numpy Hough + pairwise
            line intersection (restricted to straight-driving frames),
      (SG)  sky/ground boundary via row-wise structure (Otsu split),
  * the model (to_image_plane) ASSUMES horizon = h/2 = 128; report the offset
    of the measured VP row from 128.
Saves annotated PNGs (GT fan + measured horizon + v=128) for a sample.
"""
import glob, io, json, math, os, sys
import numpy as np
import torch
from PIL import Image, ImageDraw

# ---- model projection (mirror of tanitad.replay.rr_log.to_image_plane) -------
F_EFF_256, CAM_H, X_CLIP = 266.0, 1.22, 2.0
def to_image_plane(xy, h, w):
    x = np.clip(xy[:, 0], X_CLIP, None)
    f = F_EFF_256 * (h / 256.0)
    u = w / 2 - f * (xy[:, 1] / x)
    v = h / 2 + f * (CAM_H / x)
    return np.stack([u, v], axis=1)

WAYPOINT_STEPS = (5, 10, 15, 20)

def ego_frame(dxy, yaw):
    c, s = math.cos(-yaw), math.sin(-yaw)
    return np.array([dxy[0]*c - dxy[1]*s, dxy[0]*s + dxy[1]*c])

def gt_fan(poses, last):
    yaw0 = float(poses[last, 2]); p0 = poses[last, :2].numpy().astype(float)
    wps = [ego_frame(poses[last+k, :2].numpy().astype(float) - p0, yaw0)
           for k in WAYPOINT_STEPS]
    return np.array(wps)

# ---- pure-numpy image ops ----------------------------------------------------
def conv3(a, k):
    ap = np.pad(a, 1, mode='edge'); out = np.zeros_like(a, dtype=float)
    for i in range(3):
        for j in range(3):
            out += k[i, j] * ap[i:i+a.shape[0], j:j+a.shape[1]]
    return out
KX = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], float); KY = KX.T

def gray(rgb):
    return rgb.astype(float) @ np.array([0.299, 0.587, 0.114])

def vp_row(rgb):
    """Road vanishing-point row via numpy Hough + pairwise intersection.
    Returns (v_vp, u_vp, n_votes) or (nan,nan,0)."""
    g = gray(rgb); H, W = g.shape
    gx = conv3(g, KX); gy = conv3(g, KY)
    mag = np.hypot(gx, gy); ori = np.arctan2(gy, gx)
    band = np.zeros_like(mag, bool)
    band[int(0.35*H):int(0.95*H), :] = True            # road band, drop sky/hood
    thr = np.percentile(mag[band], 88)
    ys, xs = np.where(band & (mag > thr))
    if len(xs) < 30:
        return math.nan, math.nan, 0
    phi = ori[ys, xs] + math.pi/2.0                    # line dir _|_ gradient
    slope = np.tan(phi)
    keep = (np.abs(slope) > 0.18) & (np.abs(slope) < 6.0)  # diagonal road edges
    ys, xs, phi = ys[keep], xs[keep], phi[keep]
    if len(xs) < 20:
        return math.nan, math.nan, 0
    cph, sph = np.cos(phi), np.sin(phi)
    dxdy = cph / np.where(np.abs(sph) < 1e-6, 1e-6, sph)   # dx per +1 row
    left = dxdy < 0; right = dxdy > 0                  # converging edge families
    def samp(m, n=90):
        idx = np.where(m)[0]
        if len(idx) > n:
            order = np.argsort(-mag[ys[idx], xs[idx]]); idx = idx[order[:n]]
        return idx
    li, ri = samp(left), samp(right)
    if len(li) < 3 or len(ri) < 3:
        return math.nan, math.nan, 0
    inter = []
    for a in li:
        p1 = np.array([xs[a], ys[a]], float); d1 = np.array([cph[a], sph[a]])
        for b in ri:
            p2 = np.array([xs[b], ys[b]], float); d2 = np.array([cph[b], sph[b]])
            det = d1[0]*(-d2[1]) - (-d2[0])*d1[1]
            if abs(det) < 1e-6:
                continue
            bb = p2 - p1
            t = (bb[0]*(-d2[1]) - (-d2[0])*bb[1]) / det
            pt = p1 + t*d1
            if 0.15*W < pt[0] < 0.85*W and 0.05*H < pt[1] < 0.85*H:
                inter.append(pt)
    if len(inter) < 10:
        return math.nan, math.nan, 0
    inter = np.array(inter)
    return float(np.median(inter[:, 1])), float(np.median(inter[:, 0])), len(inter)

def skyground_row(rgb):
    """Horizon as sky/ground split by row-structure Otsu (secondary check)."""
    g = gray(rgb); H, W = g.shape
    gy = conv3(g, KY)
    struct = np.abs(gy).mean(1) + g.std(1)*0.3         # low in sky, high on ground
    s = np.convolve(struct, np.ones(7)/7, mode='same')
    # Otsu over rows on the structure profile
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-6:
        return math.nan
    t = (s - lo)/(hi - lo)
    best, bi = -1, H//2
    for r in range(int(0.15*H), int(0.85*H)):
        top, bot = t[:r], t[r:]
        if len(top) < 3 or len(bot) < 3:
            continue
        var = (top.mean()-bot.mean())**2 * len(top)*len(bot)
        if var > best:
            best, bi = var, r
    return float(bi)

def anchor_rgb(frames_u8, last):
    f = frames_u8[last]                                 # [9,256,256]
    return f[-3:].permute(1, 2, 0).contiguous().numpy().astype(np.uint8)

def straight_moving(poses, last, need=20):
    if last + need >= poses.shape[0]:
        return False
    fan = gt_fan(poses, last)
    speed = float(poses[last, 3])
    lat = abs(float(fan[-1, 1])); fwd = float(fan[-1, 0])
    return speed > 3.0 and fwd > 6.0 and lat < 1.6      # moving, straight

def draw(rgb, fan, v_vp, save):
    im = Image.fromarray(rgb).resize((512, 512), Image.NEAREST)
    d = ImageDraw.Draw(im); sc = 512/256.0
    px = to_image_plane(np.vstack([[0., 0.], fan]), 256, 256) * sc
    d.line([tuple(p) for p in px], fill=(245, 179, 1), width=3)     # GT fan (amber)
    for p in px:
        d.ellipse([p[0]-3, p[1]-3, p[0]+3, p[1]+3], fill=(245, 179, 1))
    d.line([(0, 128*sc), (512, 128*sc)], fill=(80, 200, 255), width=2)   # model h/2
    if not math.isnan(v_vp):
        d.line([(0, v_vp*sc), (512, v_vp*sc)], fill=(255, 60, 60), width=2)  # measured
    im.save(save)

def run(cache_dir, tag, n_eps, out_dir, n_png):
    eps = sorted(glob.glob(os.path.join(cache_dir, 'ep_*.pt')))[:n_eps]
    vps, sgs, off, saved = [], [], [], 0
    for ei, p in enumerate(eps):
        d = torch.load(p, map_location='cpu', weights_only=False)
        frames, poses = d['frames_u8'], d['poses']
        T = frames.shape[0]
        cand = [t for t in range(15, T-21, 12) if straight_moving(poses, t)]
        for last in cand[:2]:                           # up to 2 frames per ep
            rgb = anchor_rgb(frames, last)
            v, u, nv = vp_row(rgb)
            sg = skyground_row(rgb)
            if not math.isnan(v):
                vps.append(v); off.append(v - 128.0)
            if not math.isnan(sg):
                sgs.append(sg)
            if saved < n_png and not math.isnan(v):
                draw(rgb, gt_fan(poses, last),
                     v, os.path.join(out_dir, f'{tag}_{saved}.png'))
                saved += 1
        if len(vps) >= 22:
            break
    def ms(a):
        a = np.array(a, float)
        return (round(float(a.mean()), 1), round(float(a.std()), 1), len(a)) if len(a) else (math.nan, math.nan, 0)
    return {'tag': tag, 'vp_mean_sd_n': ms(vps), 'vp_offset_from128': ms(off),
            'skyground_mean_sd_n': ms(sgs)}

if __name__ == '__main__':
    out = '/workspace/gt_check2'; os.makedirs(out, exist_ok=True)
    comma = '/workspace/data/comma2k19/_epcache/comma2k19-val-61c46fca8f7f'
    phys = '/workspace/data/physicalai/_epcache/physicalai-val-8c0d3047924e'
    res = {}
    res['comma'] = run(comma, 'comma', 40, out, 3)
    res['physicalai'] = run(phys, 'physicalai', 40, out, 3)
    print(json.dumps(res, indent=2))
