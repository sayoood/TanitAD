"""Offline composition of the SAM3 world map from the renderer's saved vote fields (sam3map_render_v5m.py FIELDS=1).
With no options it reproduces the delivered map d0 (asserted cell for cell by the caller). Options, for the PI's review of
2026-09-14 ("why is the Sperrflaeche marked as red? why is the separation between yellow and red so noisy? it was better in the
past. The stripes of the crosswalk are also not well enough separated"):
  LINE_EDGE_GAP=g      a lane-line cell within g cells (0.1 m each) of a non-drivable cell (seen-no-class, sidewalk-verge) is road:
                       MEASURED at the day island, SAM3 calls the bright kerb face a line in some views and the adaptive paint
                       threshold keeps it, so yellow dots sit inside the red kerb edge
  LINE_MIN_LEN_M=L     a lane-line component shorter than L m (longest side of its rotated bounding box) is road
  WLK_ISLAND_M2=a      a sidewalk-verge component smaller than a m^2 whose 1-cell ring is >= 70 % drivable is road (before edges):
                       no red rings around sidewalk fragments on the road surface
  EDGE_OBS=0           observed-edge cells (>= 25 % of their near observations) are not drawn; EDGE_OBS=near keeps only those
                       within 1 cell of a non-drivable cell. Boundary edges (sidewalk-verge touching road) are unchanged.
  XWALK=image          crosswalk stripes from the IMAGE evidence inside each detected crossing: the crossing = 1.5 m closing of cells
                       with any crosswalk vote (after or before trimming), on road; per crossing the stripe direction comes from
                       the structure tensor of the mean top-hat / threshold field; that field is smoothed along the stripes
                       (0.9 m x 0.1 m line kernel) and split at the crossing's Otsu threshold; stripes >= 0.1 m^2 are crosswalk,
                       the rest of the crossing is road.
  XWALK=fft            crosswalk stripes as a FITTED periodic bar pattern per crossing. MEASURED (stripe_fft_probe.py): the large
                       crossings carry a spectral peak of 0.8-1.6 m period at 8-43x the ring median (shuffled control 2-3.6x), and
                       SAM3 stripe votes and image bright share agree on most. Per crossing (>= 5 m^2): the evidence (z-scored
                       votes + z-scored bright share) gives the peak wave vector; every cell's phase along it is binned (20 bins)
                       and the bins above the profile's mid level are bars. Kept only if peak/median >= 6 and the bar-gap evidence
                       contrast >= 0.5 z; otherwise the crossing keeps the vote rendering.
  XWALK=fft_tiles      as fft, but fitted in overlapping 4.5 m windows (stride 1.5 m, spectrum zero-padded to 256 for a finer
                       period), each window with its own guards (peak/median >= 5, contrast >= 0.5 z, >= 60 % of the window in the
                       crossing); a cell is a bar when more fitted windows call it bar than gap; cells no window fitted keep the
                       vote rendering. A crossing area that also holds a lane line or two crossings at an angle is split this way.
  XWALK=dirclose       keep the vote-rendered stripe cells (they sit on the paint: bar/gap image top-hat 1.54 night, 2.41 day,
                       xwalk_reproj_contrast.py) and join their fragments ALONG the local stripe direction only: per cell the
                       direction is the normal of the structure tensor of the stripe-vote field (Gaussian 1 m), quantised to 8
                       bins; each bin closes the stripe mask with a line element of XWALK_LEN cells at that angle; closed cells
                       inside the crossing and on road become crosswalk. MEASURED first: the fitted periodic bars (fft /
                       fft_tiles) looked clean but did not sit on the paint (bar/gap 1.07 / 1.12 night, 0.95 / 1.31 day).
Usage: compose.py <fields render dir> <out render dir>   (options by environment)"""
import json, os, sys
from pathlib import Path
import numpy as np
import cv2

fd, od = Path(sys.argv[1]), Path(sys.argv[2]); od.mkdir(parents=True, exist_ok=True)
F = np.load(fd / "worldmap_fields.npz", allow_pickle=True)
W0 = np.load(fd / "worldmap.npz", allow_pickle=True)
RES = float(F["res"]); SURFACE_MODE = int(os.environ.get("SURFACE_MODE", "5")); PAINT_SHARE = float(os.environ.get("PAINT_SHARE", "0.2"))
LINE_EDGE_GAP = int(os.environ.get("LINE_EDGE_GAP", "0")); LINE_MIN_LEN_M = float(os.environ.get("LINE_MIN_LEN_M", "0"))
WLK_ISLAND_M2 = float(os.environ.get("WLK_ISLAND_M2", "0")); EDGE_OBS = os.environ.get("EDGE_OBS", "1"); XWALK = os.environ.get("XWALK", "votes")
vk = {k: F[f"vk{k}"].astype(np.float32) for k in (1, 2, 3, 4, 6, 7)}
seen = F["seen"].astype(bool)
vedge, vnear = F["vedge"].astype(np.float32), F["vnear"].astype(np.float32)
stats = {}
# ---- surface and paint exactly as renderer v5d (majority_v65, SURFACE_MODE, PAINT_SHARE, crosswalk share 0.15)
dw = vk[1] + vk[2] + vk[3] + vk[4] + vk[6]; sw = vk[7]
out = np.where(seen, 0, 255).astype(np.uint8)
drv = (dw > 0) & (dw >= sw); wlk = (sw > 0) & (sw > dw)
if SURFACE_MODE >= 3:
    surf = np.where(drv, 1, np.where(wlk, 2, 0)).astype(np.uint8)
    sums = [cv2.boxFilter(((surf == k_) & seen).astype(np.float32), -1, (SURFACE_MODE, SURFACE_MODE), normalize=False) for k_ in (0, 1, 2)]
    surf2 = np.argmax(np.stack(sums), axis=0).astype(np.uint8)
    has = (dw + sw) > 0
    drv = seen & has & (surf2 == 1); wlk = seen & has & (surf2 == 2)
out[drv] = 1; out[wlk] = 7
stack = np.stack([vk[2], vk[3], vk[4], vk[6]])
best_p = np.argmax(stack, axis=0); best_w = np.max(stack, axis=0)
paint_k = np.array([2, 3, 4, 6], np.uint8)[best_p]
thr_p = np.where(paint_k == 3, 0.15, PAINT_SHARE)
pm = drv & (best_w >= thr_p * dw)
out[pm] = paint_k[pm]
# ---- options before edges
if WLK_ISLAND_M2 > 0:
    n_w, lab_w, st_w, _ = cv2.connectedComponentsWithStats(wlk.astype(np.uint8), connectivity=8)
    conv = 0
    for k_ in np.flatnonzero(st_w[:, cv2.CC_STAT_AREA] < WLK_ISLAND_M2 / RES ** 2)[1:]:
        x_, y_, w_, h_ = st_w[k_, 0], st_w[k_, 1], st_w[k_, 2], st_w[k_, 3]
        i0, i1, j0, j1 = max(y_ - 1, 0), min(y_ + h_ + 1, out.shape[0]), max(x_ - 1, 0), min(x_ + w_ + 1, out.shape[1])
        comp = lab_w[i0:i1, j0:j1] == k_
        ring = (cv2.dilate(comp.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~comp
        if ring.any() and float(drv[i0:i1, j0:j1][ring].mean()) >= 0.7:
            sub = out[i0:i1, j0:j1]; sub[comp] = 1
            wsub = wlk[i0:i1, j0:j1]; wsub[comp] = False
            dsub = drv[i0:i1, j0:j1]; dsub[comp] = True
            conv += int(comp.sum())
    stats["cells_sidewalk_islands_to_road"] = conv
S_votes = (out == 3).copy()
if XWALK in ("image", "image_shape"):
    E = F["vth"].astype(np.float32) / (F["vobs"].astype(np.float32) + 1e-6)
    x_any = (vk[3] > 0) | (F["vraw3"].astype(np.float32) > 0)
    cross = (cv2.morphologyEx(x_any.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0) & drv
    out[(out == 3)] = 1
    n_x, lab_x, st_x, _ = cv2.connectedComponentsWithStats(cross.astype(np.uint8), connectivity=8)
    stripes_total, n_cross, angles = 0, 0, []
    for k_ in np.flatnonzero(st_x[:, cv2.CC_STAT_AREA] >= 200)[1:]:
        x_, y_, w_, h_ = st_x[k_, 0], st_x[k_, 1], st_x[k_, 2], st_x[k_, 3]
        m = 12
        i0, i1, j0, j1 = max(y_ - m, 0), min(y_ + h_ + m, out.shape[0]), max(x_ - m, 0), min(x_ + w_ + m, out.shape[1])
        comp = lab_x[i0:i1, j0:j1] == k_
        e = E[i0:i1, j0:j1].copy()
        med = float(np.median(e[comp])); e[~comp] = med
        gx = cv2.Sobel(e, cv2.CV_32F, 0, 1, ksize=3); gy = cv2.Sobel(e, cv2.CV_32F, 1, 0, ksize=3)     # d/drow (axis 0), d/dcol (axis 1)
        jxx, jyy, jxy = float((gx * gx)[comp].sum()), float((gy * gy)[comp].sum()), float((gx * gy)[comp].sum())
        ang_grad = 0.5 * np.arctan2(2 * jxy, jxx - jyy)                 # dominant gradient direction: (cos, sin) in (row, col)
        ang_stripe = ang_grad + np.pi / 2
        L = 9
        ker = np.zeros((L + 2, L + 2), np.float32); c0 = (L + 1) / 2
        for t in np.linspace(-(L - 1) / 2, (L - 1) / 2, 4 * L):
            ker[int(round(c0 + t * np.cos(ang_stripe))), int(round(c0 + t * np.sin(ang_stripe)))] = 1.0
        ker /= ker.sum()
        es = cv2.filter2D(e, -1, ker, borderType=cv2.BORDER_REPLICATE)
        vals = es[comp]
        lo, hi = float(np.percentile(vals, 2)), float(np.percentile(vals, 98))
        q = np.clip((vals - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        t_otsu, _ = cv2.threshold(q.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = lo + t_otsu / 255.0 * (hi - lo)
        stripe = comp & (es >= thr)
        n_s, lab_s, st_s, _ = cv2.connectedComponentsWithStats(stripe.astype(np.uint8), connectivity=8)
        stripe = np.isin(lab_s, np.flatnonzero(st_s[:, cv2.CC_STAT_AREA] >= 10)[1:]) if n_s > 1 else stripe
        if XWALK == "image_shape":                                       # bar-shaped pieces only: <= 0.9 m wide, within 30 deg of the stripes
            keep_s = np.zeros_like(stripe)
            n_s2, lab_s2, st_s2, _ = cv2.connectedComponentsWithStats(stripe.astype(np.uint8), connectivity=8)
            for q_ in range(1, n_s2):
                ys2, xs2 = np.nonzero(lab_s2 == q_)
                if len(ys2) < 10:
                    continue
                (_, _), (rw2, rh2), ra2 = cv2.minAreaRect(np.c_[xs2, ys2].astype(np.float32))
                long_side, short_side = max(rw2, rh2), min(rw2, rh2)
                d_long = np.array([np.cos(np.radians(ra2)), np.sin(np.radians(ra2))]) if rw2 >= rh2 else np.array([-np.sin(np.radians(ra2)), np.cos(np.radians(ra2))])
                d_str = np.array([np.sin(ang_stripe), np.cos(ang_stripe)])   # (x=col, y=row) of the stripe direction
                cosang = abs(float(d_long @ d_str)) / (np.linalg.norm(d_long) * np.linalg.norm(d_str) + 1e-9)
                if short_side * RES <= 0.9 and (long_side < 1.5 * short_side or cosang >= np.cos(np.radians(30))):
                    keep_s[ys2, xs2] = True
            stripe = keep_s
        sub = out[i0:i1, j0:j1]
        sub[stripe & (sub == 1)] = 3                                          # lines and other paint inside the crossing are kept
        stripes_total += int(stripe.sum()); n_cross += 1; angles.append(round(float(np.degrees(ang_stripe)), 1))
    stats["crossings"] = n_cross; stats["crosswalk_stripe_cells"] = stripes_total; stats["stripe_angles_deg"] = angles
    if XWALK == "image_shape":
        back = S_votes & (out == 1)
        out[back] = 3                                                   # the vote-rendered stripe cells stay (they sit on the paint)
        stats["vote_stripe_cells_restored"] = int(back.sum())
if XWALK == "fft":
    eps_ = 1e-6
    obs_ = F["vobs"].astype(np.float32)
    xv_ = (vk[3] + F["vraw3"].astype(np.float32)) / (obs_ + eps_)
    br_ = F["vpf"].astype(np.float32) / (obs_ + eps_)
    x_any = (vk[3] > 0) | (F["vraw3"].astype(np.float32) > 0)
    cross = (cv2.morphologyEx(x_any.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0) & drv
    n_x, lab_x, st_x, _ = cv2.connectedComponentsWithStats(cross.astype(np.uint8), connectivity=8)
    fits = []
    for k_ in np.flatnonzero(st_x[:, cv2.CC_STAT_AREA] >= 500)[1:]:
        x_, y_, w_, h_ = st_x[k_, 0], st_x[k_, 1], st_x[k_, 2], st_x[k_, 3]
        comp = lab_x[y_:y_ + h_, x_:x_ + w_] == k_
        a = xv_[y_:y_ + h_, x_:x_ + w_]; b = br_[y_:y_ + h_, x_:x_ + w_]
        e = (a - a[comp].mean()) / (a[comp].std() + eps_) + (b - b[comp].mean()) / (b[comp].std() + eps_)
        N = int(2 ** np.ceil(np.log2(max(w_, h_) + 8)))
        z = np.zeros((N, N), np.float32)
        win = np.outer(np.hanning(h_ + 2)[1:-1], np.hanning(w_ + 2)[1:-1]).astype(np.float32)
        z[:h_, :w_] = np.where(comp, e - e[comp].mean(), 0) * win
        S_ = np.abs(np.fft.fftshift(np.fft.fft2(z)))
        fy, fx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(N, d=RES)), np.fft.fftshift(np.fft.fftfreq(N, d=RES)), indexing="ij")
        fr = np.hypot(fx, fy); ring = (fr >= 1 / 1.6) & (fr <= 1 / 0.6)
        i_pk = int(np.argmax(np.where(ring, S_, -1)))
        snr = float(S_.ravel()[i_pk] / (np.median(S_[ring]) + eps_))
        ky, kx = float(fy.ravel()[i_pk]), float(fx.ravel()[i_pk])            # cycles per metre along rows, cols
        rr, cc = np.nonzero(comp)
        ph = np.mod((rr * ky + cc * kx) * RES, 1.0)
        bins = np.minimum((ph * 20).astype(int), 19)
        prof = np.bincount(bins, weights=e[rr, cc], minlength=20) / np.maximum(np.bincount(bins, minlength=20), 1)
        bar_bins = prof >= (prof.min() + prof.max()) / 2
        is_bar = bar_bins[bins]
        contrast = float(e[rr, cc][is_bar].mean() - e[rr, cc][~is_bar].mean()) if is_bar.any() and (~is_bar).any() else 0.0
        ok = snr >= 6 and contrast >= 0.5
        fits.append({"area_m2": round(float(comp.sum()) * RES * RES, 1), "period_m": round(1 / max(np.hypot(kx, ky), eps_), 2), "snr": round(snr, 1),
                     "contrast_z": round(contrast, 2), "duty": round(float(bar_bins.mean()), 2), "used": bool(ok)})
        if not ok:
            continue
        sub = out[y_:y_ + h_, x_:x_ + w_]
        sub[comp & (sub == 3)] = 1
        bar = np.zeros_like(comp); bar[rr[is_bar], cc[is_bar]] = True
        sub[bar & (sub == 1)] = 3
    stats["crosswalk_fits"] = fits
if XWALK == "fft_tiles":
    eps_ = 1e-6
    obs_ = F["vobs"].astype(np.float32)
    xv_ = (vk[3] + F["vraw3"].astype(np.float32)) / (obs_ + eps_)
    br_ = F["vpf"].astype(np.float32) / (obs_ + eps_)
    x_any = (vk[3] > 0) | (F["vraw3"].astype(np.float32) > 0)
    cross = (cv2.morphologyEx(x_any.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0) & drv
    n_x, lab_x, st_x, _ = cv2.connectedComponentsWithStats(cross.astype(np.uint8), connectivity=8)
    WIN, STRIDE, NP = 45, 15, 256
    fy, fx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(NP, d=RES)), np.fft.fftshift(np.fft.fftfreq(NP, d=RES)), indexing="ij")
    fr = np.hypot(fx, fy); ring = (fr >= 1 / 1.6) & (fr <= 1 / 0.6)
    hann = np.outer(np.hanning(WIN + 2)[1:-1], np.hanning(WIN + 2)[1:-1]).astype(np.float32)
    n_win, n_used, cells_bar = 0, 0, 0
    for k_ in np.flatnonzero(st_x[:, cv2.CC_STAT_AREA] >= 200)[1:]:
        x_, y_, w_, h_ = st_x[k_, 0], st_x[k_, 1], st_x[k_, 2], st_x[k_, 3]
        comp = lab_x[y_:y_ + h_, x_:x_ + w_] == k_
        a = xv_[y_:y_ + h_, x_:x_ + w_]; b = br_[y_:y_ + h_, x_:x_ + w_]
        e = (a - a[comp].mean()) / (a[comp].std() + eps_) + (b - b[comp].mean()) / (b[comp].std() + eps_)
        vb = np.zeros(comp.shape, np.int16); vg = np.zeros(comp.shape, np.int16)
        for r0 in range(0, max(h_ - WIN, 0) + 1, STRIDE):
            for c0 in range(0, max(w_ - WIN, 0) + 1, STRIDE):
                cw = comp[r0:r0 + WIN, c0:c0 + WIN]
                if cw.shape != (WIN, WIN) or cw.mean() < 0.6:
                    continue
                n_win += 1
                ew = e[r0:r0 + WIN, c0:c0 + WIN]
                z = np.zeros((NP, NP), np.float32)
                z[:WIN, :WIN] = np.where(cw, ew - ew[cw].mean(), 0) * hann
                S_ = np.abs(np.fft.fftshift(np.fft.fft2(z)))
                i_pk = int(np.argmax(np.where(ring, S_, -1)))
                snr = float(S_.ravel()[i_pk] / (np.median(S_[ring]) + eps_))
                if snr < 5:
                    continue
                ky, kx = float(fy.ravel()[i_pk]), float(fx.ravel()[i_pk])
                rr, cc = np.nonzero(cw)
                ph = np.mod(((rr + r0) * ky + (cc + c0) * kx) * RES, 1.0)
                bins = np.minimum((ph * 20).astype(int), 19)
                prof = np.bincount(bins, weights=ew[rr, cc], minlength=20) / np.maximum(np.bincount(bins, minlength=20), 1)
                bar_bins = prof >= (prof.min() + prof.max()) / 2
                is_bar = bar_bins[bins]
                if not is_bar.any() or is_bar.all():
                    continue
                contrast = float(ew[rr, cc][is_bar].mean() - ew[rr, cc][~is_bar].mean())
                if contrast < 0.5:
                    continue
                n_used += 1
                vb[rr[is_bar] + r0, cc[is_bar] + c0] += 1; vg[rr[~is_bar] + r0, cc[~is_bar] + c0] += 1
        fitted = comp & ((vb + vg) > 0)
        bar = fitted & (vb > vg)
        sub = out[y_:y_ + h_, x_:x_ + w_]
        sub[fitted & (sub == 3)] = 1
        sub[bar & (sub == 1)] = 3
        cells_bar += int(bar.sum())
    stats["xwalk_windows"] = n_win; stats["xwalk_windows_fitted"] = n_used; stats["xwalk_bar_cells"] = cells_bar
if XWALK == "dirclose":
    eps_ = 1e-6
    XL = int(os.environ.get("XWALK_LEN", "7"))
    obs_ = F["vobs"].astype(np.float32)
    V = cv2.GaussianBlur((vk[3] + F["vraw3"].astype(np.float32)) / (obs_ + eps_), (0, 0), 1.0)
    x_any = (vk[3] > 0) | (F["vraw3"].astype(np.float32) > 0)
    cross = (cv2.morphologyEx(x_any.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0) & drv
    gr_ = cv2.Sobel(V, cv2.CV_32F, 0, 1, ksize=3); gc_ = cv2.Sobel(V, cv2.CV_32F, 1, 0, ksize=3)
    jrr = cv2.GaussianBlur(gr_ * gr_, (0, 0), 10.0); jcc = cv2.GaussianBlur(gc_ * gc_, (0, 0), 10.0); jrc = cv2.GaussianBlur(gr_ * gc_, (0, 0), 10.0)
    ang_s = 0.5 * np.arctan2(2 * jrc, jrr - jcc) + np.pi / 2                  # stripe direction, (cos, sin) in (row, col)
    bins = np.mod(np.round(ang_s / (np.pi / 8)).astype(int), 8)
    S0 = (out == 3).astype(np.uint8)
    res_m = S0.astype(bool).copy()
    for b_ in range(8):
        th_ = b_ * np.pi / 8
        ker = np.zeros((XL + 2, XL + 2), np.uint8); c0 = (XL + 1) / 2
        for t in np.linspace(-(XL - 1) / 2, (XL - 1) / 2, 4 * XL):
            ker[int(round(c0 + t * np.cos(th_))), int(round(c0 + t * np.sin(th_)))] = 1
        closed = cv2.morphologyEx(S0, cv2.MORPH_CLOSE, ker) > 0
        res_m |= closed & (bins == b_)
    add = res_m & cross & (out == 1)
    out[add] = 3
    stats["xwalk_dirclose_cells_added"] = int(add.sum())
if LINE_EDGE_GAP > 0 or LINE_MIN_LEN_M > 0:
    line = out == 2
    removed = 0
    if LINE_EDGE_GAP > 0:
        nondrv = seen & np.isin(out, (0, 7))
        dnd = cv2.distanceTransform((~nondrv).astype(np.uint8), cv2.DIST_L2, 3)
        near = line & (dnd <= LINE_EDGE_GAP + 0.5)
        out[near] = 1; removed += int(near.sum())
    if LINE_MIN_LEN_M > 0:
        n_l, lab_l, st_l, _ = cv2.connectedComponentsWithStats((out == 2).astype(np.uint8), connectivity=8)
        for k_ in range(1, n_l):
            x_, y_, w_, h_ = st_l[k_, 0], st_l[k_, 1], st_l[k_, 2], st_l[k_, 3]
            sub = out[y_:y_ + h_, x_:x_ + w_]; comp = lab_l[y_:y_ + h_, x_:x_ + w_] == k_
            if np.hypot(w_, h_) * RES < LINE_MIN_LEN_M:                          # cannot reach L even diagonally
                sub[comp] = 1; removed += int(comp.sum()); continue
            ys_, xs_ = np.nonzero(comp)
            if len(ys_) < 3:
                sub[comp] = 1; removed += len(ys_); continue
            (_, _), (rw, rh), _ = cv2.minAreaRect(np.c_[xs_, ys_].astype(np.float32))
            if max(rw, rh) * RES < LINE_MIN_LEN_M:
                sub[comp] = 1; removed += int(comp.sum())
    stats["line_cells_removed"] = removed
# ---- edges exactly as renderer v5d, with EDGE_OBS option; paint never overwritten by an observed edge when EDGE_OBS=near
wlk_c = cv2.morphologyEx(wlk.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
drv_c = cv2.morphologyEx(drv.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
touch = cv2.dilate(drv_c.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
e_boundary = wlk_c & touch
n_e, lab_e, st_e, _ = cv2.connectedComponentsWithStats(e_boundary.astype(np.uint8), connectivity=8)
e_boundary = np.isin(lab_e, np.flatnonzero(st_e[:, cv2.CC_STAT_AREA] >= 10)[1:]) if n_e > 1 else e_boundary
e_obs = (vnear > 0) & (vedge >= 0.25 * vnear)
if EDGE_OBS == "0":
    e_obs = np.zeros_like(e_obs)
elif EDGE_OBS == "near":
    nondrv = seen & np.isin(out, (0, 7))
    e_obs = e_obs & (cv2.dilate(nondrv.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~np.isin(out, (2, 3, 4, 6))
out[e_boundary | e_obs] = 5
# ---- self-checks that need no LiDAR
nondrv = seen & np.isin(out, (0, 7))
near_nd = cv2.dilate(nondrv.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
stats["edge_cells"] = int((out == 5).sum())
stats["edge_cells_with_no_non_drivable_within_0.2m"] = int(((out == 5) & ~near_nd).sum())
a2 = (out == 2); a5 = (out == 5)
stats["line_edge_4neighbour_contacts"] = int((a2[1:, :] & a5[:-1, :]).sum() + (a2[:-1, :] & a5[1:, :]).sum() + (a2[:, 1:] & a5[:, :-1]).sum() + (a2[:, :-1] & a5[:, 1:]).sum())
stats["world_cells_per_class"] = {str(k): int((out == k).sum()) for k in range(0, 8)}
np.savez_compressed(od / "worldmap.npz", cls=out, rng=W0["rng"], origin=W0["origin"], res=W0["res"], clip_sha12=W0["clip_sha12"], ver=W0["ver"],
                    composite=str(W0["composite"]) + "+compose", options=json.dumps({k: os.environ.get(k) for k in ("LINE_EDGE_GAP", "LINE_MIN_LEN_M", "WLK_ISLAND_M2", "EDGE_OBS", "XWALK")}))
(od / "render_v5_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
print(json.dumps(stats)); print("ZZCOMPOSE-DONEZZ")
