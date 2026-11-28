"""Crosswalk vs hatched area decided on the GROUND, not in the image (v3.1).

v3 turned stripe regions into hatched areas when SAM3's "diagonal stripes on road" prompt covered them -- an IMAGE
cue: a zebra seen obliquely by a rear camera looks diagonal. Measured on the night clip: 78 of 84 crosswalk -> hatched
decisions came from the rear views L2/R2, 0 from the front camera. On the ground the two are separable by geometry:
  zebra    >= 3 stripes, mutually parallel (axial circular std <= 15 deg), PERPENDICULAR to the band they form
           (angle to the region's long axis >= 60 deg, region elongation >= 1.5)
  hatched  stripes in two orientations (chevron: axial std > 25 deg) OR parallel stripes DIAGONAL to the region
           (20-55 deg)
  otherwise ambiguous -> the caller falls back (front camera: the diagonal-stripes prompt; other views: crosswalk)
Stripes shorter than 0.8 m on the ground or with < 12 lifted samples are ignored.
"""
import numpy as np
import sam3_paint as P


def local_directions(xy, cell=1.0, min_pts=8, min_elong=1.8):
    """Axial directions of paint inside 1 m ground cells (stripes that touch a boundary line and merge into one
    component still give their own direction locally), weighted by the cell's point count."""
    key = np.floor(xy / cell).astype(np.int64)
    k = key[:, 0] * 1_000_003 + key[:, 1]
    order = np.argsort(k, kind="stable"); k, p = k[order], xy[order]
    starts = np.r_[0, np.flatnonzero(np.diff(k)) + 1]; ends = np.r_[starts[1:], len(k)]
    dirs, wts = [], []
    for s_, e_ in zip(starts, ends):
        if e_ - s_ < min_pts:
            continue
        c = p[s_:e_] - p[s_:e_].mean(axis=0)
        _, sv, vt = np.linalg.svd(c, full_matrices=False)
        if sv[0] < min_elong * max(sv[1], 1e-6):
            continue
        dirs.append(np.arctan2(vt[0, 1], vt[0, 0])); wts.append(e_ - s_)
    return np.asarray(dirs), np.asarray(wts, np.float64)


def decide2(stripe_union_mask, region_mask, K, R, t, grid, fb):
    """Mode of local stripe directions vs the region's long axis, both on the ground.
    zebra: dominant mode >= 60 deg to the region axis; hatched: 20-55 deg, or two strong modes >= 40 deg apart
    (chevron); region elongation >= 1.5 required; everything else ambiguous."""
    xy = P.lift(stripe_union_mask, K, R, t, grid, fb, stride=2)
    info = {"paint_pts": int(len(xy))}
    if len(xy) < 50:
        return "too_few", info
    dirs, w = local_directions(xy)
    info["cells"] = int(len(dirs))
    if len(dirs) < 3:
        return "too_few", info
    ang = np.degrees(dirs) % 180.0
    hist = np.array([w[(ang >= b) & (ang < b + 15)].sum() for b in range(0, 180, 15)])
    share = hist / hist.sum()
    b1 = int(np.argmax(share)); m1 = b1 * 15 + 7.5
    s2 = share.copy()
    for d in (-2, -1, 0, 1, 2):
        s2[(b1 + d) % 12] = 0
    b2 = int(np.argmax(s2)); m2 = b2 * 15 + 7.5
    rxy = P.lift(region_mask, K, R, t, grid, fb, stride=2)
    if len(rxy) < 30:
        return "ambiguous", info
    d, s, _ = _pca(rxy)
    elong = float(s[0] / max(s[1], 1e-6)); axis = float(np.degrees(np.arctan2(d[1], d[0])) % 180.0)
    delta1 = float(abs((m1 - axis + 90) % 180 - 90))
    sep = float(abs((m1 - m2 + 90) % 180 - 90))
    info.update({"mode1_share": round(float(share[b1]), 2), "mode2_share": round(float(s2[b2]), 2), "delta1_deg": round(delta1, 1),
                 "mode_sep_deg": round(sep, 1), "elongation": round(elong, 2)})
    if share[b1] >= 0.25 and s2[b2] >= 0.25 and sep >= 40:
        return "hatched", info
    if elong < 1.5 or share[b1] < 0.35:
        return "ambiguous", info
    if delta1 >= 60:
        return "zebra", info
    if 20 <= delta1 <= 55:
        return "hatched", info
    return "ambiguous", info


def _pca(xy):
    c = xy - xy.mean(axis=0)
    _, s, vt = np.linalg.svd(c, full_matrices=False)
    return vt[0], s, c


def decide(stripe_masks, region_mask, K, R, t, grid, fb):
    dirs = []
    for m in stripe_masks:
        xy = P.lift(m, K, R, t, grid, fb, stride=2)
        if len(xy) < 12:
            continue
        d, s, c = _pca(xy)
        proj = c @ d
        if proj.max() - proj.min() < 0.8:
            continue
        dirs.append(np.arctan2(d[1], d[0]))
    info = {"stripes": len(dirs)}
    if len(dirs) < 3:
        return "too_few", info
    a2 = 2 * np.asarray(dirs)
    Rbar = float(np.hypot(np.cos(a2).mean(), np.sin(a2).mean()))
    sigma = float(np.degrees(np.sqrt(-2 * np.log(max(Rbar, 1e-9))) / 2))
    theta_s = float(np.arctan2(np.sin(a2).mean(), np.cos(a2).mean()) / 2)
    xy = P.lift(region_mask, K, R, t, grid, fb, stride=2)
    info["sigma_deg"] = round(sigma, 1)
    if sigma > 25:
        return "hatched", info
    if len(xy) < 30:
        return "ambiguous", info
    d, s, _ = _pca(xy)
    elong = float(s[0] / max(s[1], 1e-6))
    delta = float(abs((np.degrees(theta_s - np.arctan2(d[1], d[0])) + 90) % 180 - 90))
    info.update({"delta_deg": round(delta, 1), "elongation": round(elong, 2)})
    if sigma <= 15 and elong >= 1.5 and delta >= 60:
        return "zebra", info
    if sigma <= 15 and elong >= 1.5 and 20 <= delta <= 55:
        return "hatched", info
    return "ambiguous", info
