"""Recompute the per-channel affine gain/bias on the CORRECT (wxyz) render.

The 0.485/0.446/0.434 gains that motivated the "trained per-frame ISP" hypothesis were
computed on out_xyzw/ -- the run with the quaternion layout the scene's own geometric
self-test REJECTED. Redo them on final/ (wxyz), full-frame and covered-pixels-only.
"""
import numpy as np

D = np.load("/home/nvidia/nurec_work/linear_dump.npz")
frames, render, alpha, ref = D["frames"], D["render"], D["alpha"], D["ref"]


def fit(img, r, mask=None):
    out = []
    for c in range(3):
        x = img[..., c].ravel()
        y = r[..., c].ravel()
        if mask is not None:
            m = mask.ravel()
            x, y = x[m], y[m]
        A = np.stack([x, np.ones_like(x)], 1)
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        out.append((float(sol[0]), float(sol[1])))
    return out


def psnr(a, b):
    m = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 999.0 if m <= 0 else float(10 * np.log10(1.0 / m))


print("STALE (from out_xyzw, WRONG quaternion layout): "
      "gains R 0.485 G 0.446 B 0.434, bias 0.167/0.166/0.137, PSNR 16.758, render_mean 0.2404")
print()
for i, fi in enumerate(frames):
    img = np.clip(render[i], 0, 1)
    r = ref[i]
    cov = alpha[i] >= 0.995
    gb_all = fit(img, r)
    gb_cov = fit(img, r, cov)
    print("frame %3d  render_mean=%.4f ref_mean=%.4f PSNR=%.3f  covered=%.1f%%"
          % (fi, img.mean(), r.mean(), psnr(img, r), 100 * cov.mean()))
    print("   FULL-FRAME  gain=%s  bias=%s"
          % ([round(g, 4) for g, _ in gb_all], [round(b, 4) for _, b in gb_all]))
    print("   COVERED-ONLY gain=%s  bias=%s"
          % ([round(g, 4) for g, _ in gb_cov], [round(b, 4) for _, b in gb_cov]))
    # what fraction of the full-frame residual is the uncovered (no-sky) region?
    unc = ~cov
    e = np.abs(img - r).mean(-1)
    print("   |err| covered=%.4f  uncovered=%.4f  uncovered share of total abs err=%.1f%%"
          % (e[cov].mean(), e[unc].mean(), 100 * e[unc].sum() / e.sum()))
