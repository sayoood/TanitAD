"""p2_tol_jobs.py -- the light (no torch, no loader) per-window functions of the P2 pass: controls on real GT, one window's statistics,
and the banking of the per-arm tables.  Split out of ``p2_tol_pass.py`` so ``test_p2_e2e_synth.py`` can exercise the SAME code on synthetic logits."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import p2_tol_lib as T


def controls_job(gt):
    """C1' / C2' on ONE window's real GT (cheap).  Returns {'gt_as_pred': bool, 'one_class': bool, 'n_classes': int}."""
    p = np.where(gt == T.NOT_SEEN, 0, gt).astype(np.uint8)
    st = T.tol_stats(p, gt)
    ok1, ncls = True, 0
    for c in range(8):
        if st["gt"][c].sum() == 0:
            continue
        ncls += 1
        ok1 &= bool(np.array_equal(st["pred"][c], st["gt"][c]) and np.array_equal(st["inter"][c], st["gt"][c]))
        for k in T.KS:
            ok1 &= bool(np.array_equal(st[f"tpp{k}"][c], st["gt"][c]) and np.array_equal(st[f"tpg{k}"][c], st["gt"][c]))
    one = np.full(gt.shape, 1, np.uint8)
    st2 = T.tol_stats(one, gt)
    ok2 = True
    for c in range(8):
        if c == 1:
            continue
        for k in T.KS:
            ok2 &= int(st2[f"tpg{k}"][c].sum()) == 0 and int(st2[f"tpp{k}"][c].sum()) == 0
        ok2 &= int(st2["inter"][c].sum()) == 0
    return {"gt_as_pred": bool(ok1), "one_class": bool(ok2), "n_classes": ncls}


def window_job(s12, w0, gt, logits_np, rule, cw_np, run_controls):
    """Everything done with ONE window's logits (worker thread / unit test): banked-format stats, tolerant stats, census, controls."""
    from taniteval import map_hires_metrics as MH
    pred = MH.logits_to_codes(logits_np, rule=rule, class_weight=cw_np)
    st = MH.window_stats(pred, gt)                       # banked-format stats (+ EDT<0.2 m tp for classes 2,3,5)
    tol = T.tol_stats(pred, gt)
    edt_bad = 0
    for c in (2, 3, 5):                                  # C4: chessboard k=1 == EDT<0.2 m (integer-exact)
        edt_bad += int(not np.array_equal(st["tp_pred"][c], tol["tpp1"][c]))
        edt_bad += int(not np.array_equal(st["tp_gt"][c], tol["tpg1"][c]))
    base_bad = sum(int(not np.array_equal(st[a_], tol[b_]))
                   for a_, b_ in (("inter", "inter"), ("union", "union"), ("n_gt", "gt"), ("n_pred", "pred")))
    census = T.thickness_hist(gt)
    ctl = controls_job(gt) if run_controls else None
    return {"s12": s12, "ws": w0, "st": st, "tol": tol, "edt_bad": edt_bad, "base_bad": base_bad,
            "census": census, "ctl": ctl}


def bank_rows(rows, arm, out_dir, n_rows):
    """Write the banked-format WindowTable and the tolerant-count npz.  -> (table_path, tol_path)."""
    from taniteval import map_hires_metrics as MH
    out_dir = Path(out_dir)
    table = MH.WindowTable(arms=(arm,))
    table.n_rows = int(n_rows)
    for r in rows:
        dst = table.stats.setdefault(arm, {k: [] for k in MH.STATS})
        for k in MH.STATS:
            dst[k].append(r["st"][k])
        table.eid.append(r["s12"])
    tpath = out_dir / f"p2_{arm}_map_table.npz"
    table.save(str(tpath))
    z = {"eid": np.array([r["s12"] for r in rows]), "ws": np.array([r["ws"] for r in rows], np.int64),
         "n_rows": np.int64(table.n_rows)}
    for nm in T.stat_names():
        z[nm] = np.stack([r["tol"][nm] for r in rows])
    for key in ("thick", "lat", "lon"):
        z[f"census_{key}"] = np.sum([r["census"][key] for r in rows], axis=0)
    xpath = out_dir / f"p2_{arm}_tol.npz"
    np.savez_compressed(str(xpath), **z)
    return tpath, xpath
