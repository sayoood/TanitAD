"""Census: what the 0.5 m cart grid does to each SAM3 class, measured on the 10 cm fine grid
of the SAME GT files (fine_codes [T,600,320], codes 0-7 class, 255 not seen; cart_frac is its
5x5 sub-sampled fraction). Read-only; CPU; every 10th frame of each eval GT file."""
import glob, json, sys, numpy as np
try:
    import psutil; print("host_free_GB", round(psutil.virtual_memory().available / 2**30, 1))
except Exception: pass
CH = ["seen-no-class", "drivable", "lane/road line", "crosswalk", "arrow/text",
      "non-drivable edge", "hatched", "sidewalk/verge"]
files = sorted(glob.glob("D:/refcv6_eval_kit/data/sam3_gt_eval_thor137/*.sam3mapgt.npz"))
fine_n = np.zeros(8, np.int64); fine_seen = 0
pres = np.zeros(8, np.int64); ge50 = np.zeros(8, np.int64); argm = np.zeros(8, np.int64); coarse_seen = 0
fine_in_ge50 = np.zeros(8, np.int64); fine_in_argm = np.zeros(8, np.int64)
frac_when_present = [[] for _ in range(8)]
x_band_fine = np.zeros((3, 8), np.int64)
nfr = 0; recon_bad = 0
for f in files:
    z = np.load(f)
    fc = z["fine_codes"]; cf = z["cart_frac"]
    idx = np.arange(0, fc.shape[0], 10)
    for t in idx:
        c = fc[t]                                   # [600,320]
        seen = c != 255
        fine_seen += int(seen.sum())
        fine_n += np.bincount(c[seen].ravel(), minlength=8)[:8]
        for b, (lo, hi) in enumerate(((0, 200), (200, 400), (400, 600))):
            cb = c[lo:hi]; sb = cb != 255
            x_band_fine[b] += np.bincount(cb[sb].ravel(), minlength=8)[:8]
        fr = cf[t].astype(np.float32) / 255.0       # [9,120,64]
        cseen = fr[8] < 0.5                         # the SEEN_SHARE_MIN rule: at least half seen
        coarse_seen += int(cseen.sum())
        am = fr[:8].argmax(0)
        # fine cells, block-reduced: which coarse cell each fine cell belongs to
        blk = c.reshape(120, 5, 64, 5).transpose(0, 2, 1, 3).reshape(120, 64, 25)
        for k in range(8):
            fk = fr[k]
            pres[k] += int(((fk > 0) & cseen).sum())
            ge50[k] += int(((fk >= 0.5) & cseen).sum())
            argm[k] += int(((am == k) & cseen).sum())
            nk = (blk == k).sum(-1)                 # fine cells of class k per coarse cell
            fine_in_ge50[k] += int(nk[(fk >= 0.5) & cseen].sum())
            fine_in_argm[k] += int(nk[(am == k) & cseen].sum())
            m = (fk > 0) & cseen
            if m.any() and len(frac_when_present[k]) < 200000:
                frac_when_present[k].extend(fk[m].tolist()[:20000])
        # consistency: cart_frac should equal the 5x5 block fraction of fine_codes (control)
        if t == idx[0]:
            rec = np.stack([(blk == k).mean(-1) for k in range(8)])
            recon_bad += int((np.abs(rec - fr[:8]) > 1.5 / 255).sum())
        nfr += 1
out = {"files": len(files), "frames": nfr, "fine_seen_cells": fine_seen, "coarse_seen_cells": coarse_seen,
       "control_cart_equals_fine_5x5_block_fraction_bad_cells": recon_bad, "classes": {}}
tot_fine_cls = fine_n.sum()
for k in range(8):
    fw = np.array(frac_when_present[k]) if frac_when_present[k] else np.array([0.0])
    out["classes"][CH[k]] = {
        "fine_share_of_seen_pct": round(100 * fine_n[k] / max(fine_seen, 1), 3),
        "coarse_present_any_pct": round(100 * pres[k] / max(coarse_seen, 1), 3),
        "coarse_frac_ge_0.5_pct": round(100 * ge50[k] / max(coarse_seen, 1), 3),
        "coarse_argmax_pct": round(100 * argm[k] / max(coarse_seen, 1), 3),
        "fine_cells_kept_by_ge50_rule_pct": round(100 * fine_in_ge50[k] / max(fine_n[k], 1), 1),
        "fine_cells_kept_by_argmax_pct": round(100 * fine_in_argm[k] / max(fine_n[k], 1), 1),
        "median_cell_fraction_when_present": round(float(np.median(fw)), 3),
        "fine_share_by_x_band_pct_0_20_40_60m": [round(100 * x_band_fine[b, k] / max(x_band_fine[b].sum(), 1), 3) for b in range(3)],
    }
json.dump(out, open("C:/Users/Admin/qland/work/refcv7/map_res_census.json", "w"), indent=1)
print(json.dumps(out, indent=1))
