"""Is the zebra period recoverable from the vote fields? For each detected crossing (1.5 m closing of cells with any crosswalk
vote, on road, >= 2 m^2): the 2-D spectrum of three evidence fields inside the crossing (Hann-weighted, mean removed) -- SAM3 stripe
votes (vk3 + vraw3) / obs, image bright share vpf / obs, and their sum of z-scores -- and the strongest peak with a period of
0.6-1.6 m: period, bar direction, and its height over the median of that frequency ring (a noise-only field reads ~1-3).
Also a SHUFFLED control: the same crossing cells with their values permuted must show no peak.
Usage: stripe_fft_probe.py <fields render dir> [<fields render dir> ...]"""
import sys
from pathlib import Path
import numpy as np
import cv2

rng = np.random.default_rng(0)
for arg in sys.argv[1:]:
    F = np.load(Path(arg) / "worldmap_fields.npz", allow_pickle=True)
    RES = float(F["res"]); eps = 1e-6
    vk = {k: F[f"vk{k}"].astype(np.float32) for k in (1, 2, 3, 4, 6, 7)}
    obs = F["vobs"].astype(np.float32)
    dw = vk[1] + vk[2] + vk[3] + vk[4] + vk[6]
    drv = (dw > 0) & (dw >= vk[7])
    xv = (vk[3] + F["vraw3"].astype(np.float32)) / (obs + eps)
    br = F["vpf"].astype(np.float32) / (obs + eps)
    x_any = (vk[3] > 0) | (F["vraw3"].astype(np.float32) > 0)
    cross = (cv2.morphologyEx(x_any.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0) & drv
    n, lab, st, cen = cv2.connectedComponentsWithStats(cross.astype(np.uint8), connectivity=8)
    print("===", arg)
    for k in np.flatnonzero(st[:, cv2.CC_STAT_AREA] >= 200)[1:]:
        x_, y_, w_, h_ = st[k, 0], st[k, 1], st[k, 2], st[k, 3]
        comp = lab[y_:y_ + h_, x_:x_ + w_] == k
        N = int(2 ** np.ceil(np.log2(max(w_, h_) + 8)))
        rows = []
        for name, fld in (("votes", xv), ("bright", br), ("both", None), ("shuffled votes", "shuf")):
            if fld is None:
                a = xv[y_:y_ + h_, x_:x_ + w_]; b = br[y_:y_ + h_, x_:x_ + w_]
                sub = (a - a[comp].mean()) / (a[comp].std() + eps) + (b - b[comp].mean()) / (b[comp].std() + eps)
            elif isinstance(fld, str):
                sub = xv[y_:y_ + h_, x_:x_ + w_].copy(); vals = sub[comp].copy(); rng.shuffle(vals); sub[comp] = vals
            else:
                sub = fld[y_:y_ + h_, x_:x_ + w_].copy()
            z = np.zeros((N, N), np.float32)
            win = np.outer(np.hanning(h_ + 2)[1:-1], np.hanning(w_ + 2)[1:-1]).astype(np.float32)
            vals = np.where(comp, sub - sub[comp].mean(), 0) * win
            z[:h_, :w_] = vals
            S = np.abs(np.fft.fftshift(np.fft.fft2(z)))
            fy, fx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(N, d=RES)), np.fft.fftshift(np.fft.fftfreq(N, d=RES)), indexing="ij")
            fr = np.hypot(fx, fy)
            ring = (fr >= 1 / 1.6) & (fr <= 1 / 0.6)
            if not ring.any():
                continue
            i_pk = np.argmax(np.where(ring, S, -1))
            pk = S.ravel()[i_pk]; f_pk = fr.ravel()[i_pk]
            snr = pk / (np.median(S[ring]) + eps)
            ang = np.degrees(np.arctan2(fy.ravel()[i_pk], fx.ravel()[i_pk]))          # wave vector direction (across the bars)
            rows.append(f"{name}: period {1 / f_pk:.2f} m, across-bar dir {ang:6.1f} deg, peak/median {snr:5.1f}")
        cy, cx = cen[k][1], cen[k][0]
        print(f"crossing {k}: {st[k, cv2.CC_STAT_AREA] * RES * RES:5.1f} m2 at cell ({cy:.0f},{cx:.0f}) | " + " | ".join(rows))
print("ZZFFT-DONEZZ")
