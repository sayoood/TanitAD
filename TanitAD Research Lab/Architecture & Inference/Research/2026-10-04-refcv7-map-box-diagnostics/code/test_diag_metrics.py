"""Controls C3, C4, C6 of SPEC.md sec. 4 -- each must read a KNOWN value. Expected values are LITERALS.

Run: python -m pytest -q test_diag_metrics.py   (CPU only; no trainer import)
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_metrics as dm  # noqa: E402

H, W = 400, 60          # 2 bands of 200 rows


def _line_scene():
    """A 1-cell-wide lateral... no: a 1-cell-wide LONGITUDINAL line of class 5 (edge) at column 30, drivable
    to its left, sidewalk to its right; all cells supervised."""
    codes = torch.full((1, H, W), 7, dtype=torch.long)
    codes[:, :, :30] = 1
    codes[:, :, 30] = 5
    sup = torch.ones(1, H, W, dtype=torch.bool)
    return codes, sup


def _counts(pred_codes, gt_codes, sup):
    M = dm.band_matrix(H, "cpu")
    P = dm.onehot_masks(pred_codes, sup)
    G = dm.onehot_masks(gt_codes, sup)
    c = dm.tolerant_counts(P, G, M)
    return {k: v.sum(dim=(0, 2)) for k, v in c.items()}       # pooled over windows and bands -> [8]


def _summ(c, cls):
    return dm.tolerant_summary({k: v[cls] for k, v in c.items()})


def test_gt_as_prediction_reads_one():
    codes, sup = _line_scene()
    c = _counts(codes, codes, sup)
    for cls in (1, 5, 7):
        s = _summ(c, cls)
        for k in (0, 1, 2):
            assert s[f"IoU{k}"] == 1.0
        assert dm.iou(c["inter"][cls], c["pred"][cls], c["gt"][cls]) == 1.0


def test_empty_prediction_reads_zero_for_every_class():
    codes, sup = _line_scene()
    M = dm.band_matrix(H, "cpu")
    P = torch.zeros(1, 8, H, W, dtype=torch.bool)
    G = dm.onehot_masks(codes, sup)
    c = {k: v.sum(dim=(0, 2)) for k, v in dm.tolerant_counts(P, G, M).items()}
    for cls in (1, 5, 7):
        s = _summ(c, cls)
        assert s["IoU0"] == 0.0 and s["IoU1"] == 0.0 and s["IoU2"] == 0.0
        assert s["R0"] == 0.0 and s["P0"] is None
        assert dm.iou(c["inter"][cls], c["pred"][cls], c["gt"][cls]) == 0.0


def test_one_cell_shift_low_at_k0_one_at_k1():
    codes, sup = _line_scene()
    shifted = codes.clone()
    shifted[:, :, 30] = 1           # the line moves one cell right
    shifted[:, :, 31] = 5
    c = _counts(shifted, codes, sup)
    s = _summ(c, 5)
    assert s["IoU0"] < 0.1
    assert s["IoU0"] == 0.0         # a 1-cell line shifted by 1 has NO overlap
    assert s["IoU1"] == 1.0
    assert s["IoU2"] == 1.0


def test_iou0_equals_exact_iou():
    rng = np.random.default_rng(0)
    g = torch.from_numpy(rng.integers(0, 8, (2, H, W)))
    p = torch.from_numpy(rng.integers(0, 8, (2, H, W)))
    sup = torch.from_numpy(rng.random((2, H, W)) < 0.8)
    c = _counts(p, g, sup)
    for cls in range(8):
        s = _summ(c, cls)
        exact = dm.iou(c["inter"][cls], c["pred"][cls], c["gt"][cls])
        assert abs(s["IoU0"] - exact) < 1e-12


def test_hist_auroc_perfect_and_constant():
    pos = np.zeros(dm.HIST_NB)
    neg = np.zeros(dm.HIST_NB)
    pos[3000] = 50
    neg[1000] = 70
    assert dm.auroc_from_hist(neg, pos) == 1.0
    pos2 = np.zeros(dm.HIST_NB)
    neg2 = np.zeros(dm.HIST_NB)
    pos2[2000] = 50
    neg2[2000] = 70
    assert dm.auroc_from_hist(neg2, pos2) == 0.5
    i, best = dm.best_threshold(neg, pos)
    assert best == 1.0


def test_hist_threshold_iou_equals_direct_mask_iou():
    torch.manual_seed(0)
    z = torch.randn(2, 8, H, W) * 3
    g_codes = torch.randint(0, 8, (2, H, W))
    sup = torch.rand(2, H, W) < 0.9
    G = dm.onehot_masks(g_codes, sup)
    s = dm.class_logits(z)
    hist = dm.score_histograms(s, G, sup, 2).sum(dim=1).numpy()      # [8, 2, NB]
    edges = dm.hist_edges()
    for c in (2, 5):
        for i in (1500, 2000, 2300):
            tau = edges[i]
            mask = (s[:, c] >= float(tau)) & sup
            gt = G[:, c]
            inter = float((mask & gt).sum())
            direct = inter / float((mask | gt).sum())
            via, *_ = dm.iou_at_index(hist[c, 0], hist[c, 1], i)
            # bin edges are float; a cell exactly at an edge can fall either side -> tolerance 1e-3
            assert abs(via - direct) < 1e-3


def test_class_logits_matches_softmax():
    z = torch.randn(1, 8, 5, 5)
    p = torch.softmax(z, 1)
    s = dm.class_logits(z)
    assert torch.allclose(torch.sigmoid(s), p, atol=1e-6)


def test_fit_recovers_planted_shift_and_temperature():
    rng = np.random.default_rng(1)
    z = rng.normal(-3, 2, 200000)
    y = (rng.random(z.size) < dm.sigmoid(z + 2.5)).astype(float)       # planted shift +2.5
    a, b = dm.fit_affine(z, y, fit_a=False, fit_b=True)
    assert a == 1.0 and abs(b - 2.5) < 0.05
    y2 = (rng.random(z.size) < dm.sigmoid(z / 2.0)).astype(float)       # planted T = 2
    a2, b2 = dm.fit_affine(z, y2, fit_a=True, fit_b=False)
    assert b2 == 0.0 and abs(1.0 / a2 - 2.0) < 0.05


def test_ece_of_calibrated_sample_is_small_and_of_shifted_is_large():
    rng = np.random.default_rng(2)
    p = rng.random(200000)
    y = (rng.random(p.size) < p).astype(float)
    e, _ = dm.ece(p, y)
    assert e < 0.02
    e2, _ = dm.ece(p * 0.2, y)
    assert e2 > 0.3


def test_run_length_and_registration():
    codes, sup = _line_scene()
    codes[:, :, 30:33] = 5                                   # a 3-cell-wide edge line
    rl = dm.run_length_hist(codes.numpy(), sup.numpy(), 5, 2)
    assert rl[:, 3].sum() == H and rl.sum() == H
    drv = codes == 1
    bnd = dm.boundary(drv, sup)
    assert int(bnd.sum()) == H and bool(bnd[0, :, 29].all())
    edge = codes == 5
    rh = dm.registration_hist(edge, bnd, 2).sum(0)
    # cols 30, 31, 32 are 1, 2, 3 cells from the boundary column 29
    assert int(rh[1]) == H and int(rh[2]) == H and int(rh[3]) == H and int(rh[0]) == 0


def test_focal_inversion_reads_the_analytic_gate_belief():
    """slot_presence.gate_match_belief(0.5, 'focal') = 0.75 EXACTLY (alpha 0.25: pi*a = (1-pi)(1-a) at p = 1/2), so the
    inversion table must map p = 0.5 -> pi = 0.75. Needs the run's stack on sys.path; skipped (and SAID so) without it."""
    try:
        from tanitad.models.slot_presence import presence_optimum
    except Exception:            # noqa: BLE001
        import warnings
        warnings.warn("tanitad.models.slot_presence not importable -- focal inversion control NOT run here")
        return
    ps, pis = dm.focal_inversion_table(presence_optimum, n=801)
    assert abs(float(np.interp(0.5, ps, pis)) - 0.75) < 2e-3
    assert np.all(np.diff(ps) >= 0)
