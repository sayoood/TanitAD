"""Tests for the EDGE plane's thin-class decision rule (PREREG_M1_PEDGEFIT): the ``f_tol`` criterion of ``r9_map_fit_thresholds.py`` (``pedge`` fitted by the best
pooled F@0.3 m against GT class 5, E0d's rule) and the edge-class P / R / F @ 0.3 m BESIDE rows of ``r9_e1_eval.py`` / ``r9_e1_floor.py`` / ``r9_e1_verdict.py``.

* analytic hand cases (``_hand``'s docstring derives every level and every P / R / F by hand),
* the HISTOGRAM route of the fitter proven equal to the per-level brute force (explicit whole-raster dilation) on random rasters,
* a CONTROL: ``e0d_score``'s own ``dilate`` / ``gt_static`` / ``_edge_stats`` / ``fit_grid_threshold`` reproduced to 1e-12 on its THR_EDGE grid,
* MUTATIONS: each guard is also run against a deliberately broken implementation and must go RED (the guard returns False)."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_S = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_S))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r9_e1_eval as E                                   # noqa: E402
import r9_e1_floor as FL                                 # noqa: E402
import r9_e1_verdict as V                                # noqa: E402
import r9_map_fit_thresholds as F                        # noqa: E402

E0D_REL = Path("TanitAD Research Lab") / "Architecture & Inference" / "Research" / "2026-10-27-perception-redesign" / "e0" / "e0d_score.py"
EVAL_SPLIT = {"DEV": {"sha12": ["aaaaaaaaaaaa", "bbbbbbbbbbbb"]}, "HOLD": {"sha12": ["cccccccccccc", "dddddddddddd"]}}
FIT_CLIPS = ["a1b2c3d4e5f6", "0123456789ab"]


def _e0d():
    """e0d_score.py by path: the checkout the tests run in (the repo root above stack/), else the one ``tanitad`` is imported from. Never skipped."""
    import os
    import tanitad
    cands = [Path(os.environ["E0D_SCORE_PY"])] if os.environ.get("E0D_SCORE_PY") else []
    cands += [Path(__file__).resolve().parents[2] / E0D_REL, Path(tanitad.__file__).resolve().parents[2] / E0D_REL]
    for c in cands:
        if c.is_file():
            spec = importlib.util.spec_from_file_location("e0d_score_control", str(c))
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    raise AssertionError("e0d_score.py not found (tried %s): the control cannot run" % [str(c) for c in cands])


# ================================================================================================================= the hand world
def _hand(path=None, with_band1_edge=True):
    """A 40 x 30 raster (rows 0-19 = fit band 0-2 m, rows 20-39 = band 2-4 m; the fitter is called with --bands 0-2,2-4). Everything below is derived by hand.
    BAND 0:  GT edge (5) = row 10, cols 10-19 (10 cells) = value 200.  Rows 9 and 11, cols 10-19 are GT sidewalk (7, judged) = value 100.
             A GT-sidewalk blob (row 2, cols 25-27) = value 50 (judged, far from any edge); a GT-NOCLS blob (rows 3-4, cols 0-9: 20 cells) = value 150 (NOT judged: free);
             an UNSEEN strip (gt 255, rows 5-6, cols 28-29) = 50 (not judged).  Everything else gt 0 (nocls), value 0.
        t <= 50    flags 30 + 3 judged (+ 20 + 4 unjudged); judged AND within 0.3 m of the edge 30; recalled edge cells 10  -> P = 30/33, R = 1, F = 60/63
        51..100    judged flags = the line + its two sides (30 cells)                                                       -> P = 1, R = 1, F = 1
        101..200   judged flags = the 10 centre cells only                                                                  -> P = 1, R = 1, F = 1   (a TIE with 51..100)
        > 200      nothing                                                                                                  -> F = 0
        => f_tol level 51 (first level of the tie), F = 1, P = 1, R = 1.      If nocls counted as judged, 51..100 would be P = 30/50 and the optimum would move to 151.
           EXACT-CELL IoU of the same plane (flags incl. the nocls blob): 101..150 10/30; 151..200 10/10 = 1 -> IoU level 151.
    BAND 1:  GT edge (5) = row 25 AND row 30, cols 10-19 (20 cells).  The plane = value 40 on row 31 (one cell off the edge; GT sidewalk, judged), cols 10-19 (10 cells);
             GT-sidewalk (judged) rows 36-39, cols 0-9 (40 cells) = value 30.  Row 25 is 6 rows from row 31 (> 3): never recalled; row 30 needs the DILATION.
        t <= 30    flags 50, judged 50, hits 10 (row 31); recalled 10 of 20                                               -> P = 1/5, R = 1/2, F = 2/7
        31..40     flags the 10 line cells                                                                                -> P = 1,   R = 1/2, F = 2/3
        > 40       nothing                                                                                                -> F = 0
        => f_tol level 31, F = 2/3, P = 1, R = 1/2.  (Exact-cell IoU is 0 at every level: no flag sits ON a GT edge cell.)"""
    pe = np.zeros((40, 30), np.uint8)
    gt = np.zeros((40, 30), np.uint8)
    gt[10, 10:20] = 5
    pe[10, 10:20] = 200
    gt[9, 10:20] = gt[11, 10:20] = 7
    pe[9, 10:20] = pe[11, 10:20] = 100
    gt[2, 25:28] = 7
    pe[2, 25:28] = 50
    gt[3:5, 0:10] = 0
    pe[3:5, 0:10] = 150
    gt[5:7, 28:30] = 255
    pe[5:7, 28:30] = 50
    if with_band1_edge:
        gt[25, 10:20] = 5
        gt[30, 10:20] = 5
        gt[31, 10:20] = 7
        pe[31, 10:20] = 40
        gt[36:40, 0:10] = 7
        pe[36:40, 0:10] = 30
    if path is not None:
        np.savez_compressed(path, pdrv=np.zeros_like(pe), pedge=pe, psw=np.where(gt == 7, 200, 0).astype(np.uint8), code=np.zeros_like(pe), gt=gt)
    return pe, gt


def _simple(path):
    """A world where BOTH planes have an IoU cut (so a default psw+pedge fit succeeds): a sidewalk block (psw 200) and an edge line (pedge 180) in each band, clutter 40 / 30."""
    gt = np.zeros((40, 30), np.uint8)
    psw, pe = np.zeros((40, 30), np.uint8), np.zeros((40, 30), np.uint8)
    for r0 in (0, 20):
        gt[r0 + 2:r0 + 6, 2:11] = 7
        psw[r0 + 2:r0 + 6, 2:11] = 200
        gt[r0 + 12, 2:11] = 5
        pe[r0 + 12, 2:11] = 180
        psw[r0 + 15:r0 + 18, 15:25] = 40
        pe[r0 + 15:r0 + 18, 15:25] = 30
    np.savez_compressed(path, pdrv=np.zeros_like(pe), pedge=pe, psw=psw, code=np.zeros_like(pe), gt=gt)


def _dump(tmp_path, writer, n_files=2, name="dump"):
    """A FIT dump (<d>/rast/e*_w*.npz + dump.json) whose rasters are written by ``writer(path)``."""
    d = tmp_path / name
    (d / "rast").mkdir(parents=True)
    names = []
    for k in range(n_files):
        nm = "e%03d_w%04d.npz" % (k, 7)
        writer(d / "rast" / nm)
        names.append(nm)
    meta = {"split": {"set": "FIT", "n_clips": 2}, "raster_names_sha256": hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest(), "ckpt_md5": "ck1",
            "ckpt_kind": "ema_model_only", "step": 4000, "weights_are_ema": True, "config_md5": "cfg", "order_digest": "od", "SMOKE_RESTRICTED": False,
            "fit": {"k_per_ep": 8, "n_clips": 2, "clips_sha12": FIT_CLIPS, "rule": "r"}}
    (d / "dump.json").write_text(json.dumps(meta), encoding="utf-8")
    sp = tmp_path / "split.json"
    sp.write_text(json.dumps(EVAL_SPLIT), encoding="utf-8")
    return d, sp


def _hand_fit(tmp_path, name="thr.json", **kw):
    d, sp = _dump(tmp_path, _hand)
    return F.fit(d / "rast", sp, tmp_path / name, "T", bands_s="0-2,2-4", plane_criteria=kw.pop("plane_criteria", {"pedge": "f_tol"}), **kw), d, sp


def _hand_guard(tmp_path):
    """True iff the f_tol fit of the hand world gives exactly the hand-derived levels, F, P and R (the guard the MUTATIONS must turn False)."""
    try:
        obj, _, _ = _hand_fit(tmp_path)
    except F.FitRefused:                                            # a broken rule that finds no usable cut at all is RED too
        return False
    det = obj["provenance"]["fit_detail"]["pedge"]
    return (obj["planes"]["pedge"] == [51, 31] and det[0]["f_at_level"] == pytest.approx(1.0) and det[1]["f_at_level"] == pytest.approx(2 / 3)
            and det[0]["p_at_level"] == pytest.approx(1.0) and det[0]["r_at_level"] == pytest.approx(1.0)
            and det[1]["p_at_level"] == pytest.approx(1.0) and det[1]["r_at_level"] == pytest.approx(0.5)
            and det[0]["f_at_128"] == pytest.approx(1.0) and det[1]["f_at_128"] == pytest.approx(0.0)
            and det[0]["n_pos"] == 2 * 10 and det[1]["n_pos"] == 2 * 20)


def test_hand_world_levels_and_prf(tmp_path):
    assert _hand_guard(tmp_path) is True


def test_hand_world_curve_values_at_every_regime():
    pe, gt = _hand()
    h = F.edge_hists_arrays(pe, gt, [[0, 2], [2, 4]])
    c0, c1 = F.edge_curve(h[0]), F.edge_curve(h[1])
    # band 0: P = 30/33, R = 1, F = 60/63 at t <= 50; F = 1 on 51..200; 0 above
    assert c0["f"][1] == pytest.approx(60 / 63) and c0["p"][50] == pytest.approx(30 / 33) and c0["r"][50] == 1.0
    assert all(c0["f"][t] == 1.0 for t in (51, 100, 101, 200)) and all(c0["f"][t] == 0.0 for t in (201, 255))
    # band 1: F = 2/7 at t <= 30, 2/3 on 31..40, 0 above
    assert c1["f"][30] == pytest.approx(2 / 7) and c1["p"][30] == pytest.approx(0.2) and c1["r"][30] == 0.5
    assert c1["f"][31] == pytest.approx(2 / 3) and c1["f"][40] == pytest.approx(2 / 3) and c1["f"][41] == 0.0
    assert F.level_from_edge_hist(h[0])[0] == 51 and F.level_from_edge_hist(h[1])[0] == 31       # ties -> the FIRST level


def test_ties_take_the_lowest_level_and_a_flat_curve_is_level_one():
    h = np.zeros((3, 256))
    h[0, 200] = 7          # 7 GT edge cells whose neighbourhood max is 200
    h[1, 200] = 7          # and 7 judged flags at 200, all hits
    h[2, 200] = 7
    lv, c = F.level_from_edge_hist(h)
    assert lv == 1 and c["f"][1] == 1.0 and c["f"][200] == 1.0 and c["f"][201] == 0.0            # F = 1 on 1..200: the lowest of the tie, not 200


def test_a_band_with_no_gt_edge_is_refused_and_so_is_an_all_zero_f(tmp_path):
    with pytest.raises(F.FitRefused, match="no GT edge cell"):
        F.level_from_edge_hist(np.zeros((3, 256)))
    h = np.zeros((3, 256))
    h[0, 0] = 5            # GT edge cells exist, but no level recalls them ...
    h[1, 9] = 40           # ... and the flags are judged but never within 0.3 m of an edge
    with pytest.raises(F.FitRefused, match="F@0.3 m is 0 at every level"):
        F.level_from_edge_hist(h)

    def writer(p):                                                   # band 1 has no GT edge cell at all
        pe, gt = _hand(with_band1_edge=False)
        np.savez_compressed(p, pdrv=np.zeros_like(pe), pedge=pe, psw=np.where(gt == 7, 200, 0).astype(np.uint8), code=np.zeros_like(pe), gt=gt)
    d, sp = _dump(tmp_path, writer)
    with pytest.raises(F.FitRefused, match=r"plane pedge band 2-4 m: no GT edge cell"):
        F.fit(d / "rast", sp, tmp_path / "x.json", "T", planes=("pedge",), bands_s="0-2,2-4", plane_criteria={"pedge": "f_tol"})
    assert not (tmp_path / "x.json").exists()                          # nothing is written for a refused band


# ================================================================================================================= histogram == brute force
def _rand_raster(rng, h=130, w=70):
    cls = rng.choice([1, 2, 3, 4, 6, 7], size=(h, w))
    gt = np.where(rng.random((h, w)) < 0.35, cls, 0).astype(np.uint8)
    for _ in range(6):                                                   # thin GT edge lines
        r0, c0 = int(rng.integers(0, h - 12)), int(rng.integers(0, w - 12))
        if rng.random() < 0.5:
            gt[r0, c0:c0 + int(rng.integers(5, 12))] = 5
        else:
            gt[r0:r0 + int(rng.integers(5, 12)), c0] = 5
    gt[rng.random((h, w)) < 0.05] = 255
    pe = np.zeros((h, w), np.uint8)
    e = gt == 5
    pe[E.edge_dilate(e, 1)] = rng.integers(40, 256, size=int(E.edge_dilate(e, 1).sum()))          # a heatmap around the edges ...
    noise = rng.random((h, w)) < 0.12
    pe[noise] = np.maximum(pe[noise], rng.integers(1, 120, size=int(noise.sum())).astype(np.uint8))     # ... and clutter
    pe[rng.random((h, w)) < 0.01] = 255
    return pe, gt


def _brute_levels(rasters, bands):
    """Per band, for EVERY level 1..255: pooled counts by explicit per-level whole-raster dilation (E.edge_counts_brute), then e0d's arithmetic."""
    rows = [(lo * 10, hi * 10) for lo, hi in bands]
    cnt = np.zeros((256, len(bands), E.EDGE_TOL_N))
    for pe, gt in rasters:
        masks = E.edge_gt_masks(gt)
        for t in range(1, 256):
            cnt[t] += E.edge_counts_brute(pe >= t, masks, rows)
    return cnt


def _f_from(P, R):
    return 2 * P * R / (P + R) if (P is not None and R is not None and P + R > 0) else 0.0


def test_histogram_route_equals_per_level_brute_force_on_random_rasters():
    rng = np.random.default_rng(7)
    rasters = [_rand_raster(rng) for _ in range(3)]
    bands = [[0, 5], [5, 13]]
    hist = sum(F.edge_hists_arrays(pe, gt, bands) for pe, gt in rasters)
    cnt = _brute_levels(rasters, bands)
    for bi in range(len(bands)):
        for t in range(1, 256):
            assert F.edge_counts_at(hist[bi], t) == tuple(cnt[t, bi]), (bi, t)             # exact: nE, hitE, nXJ, hitX
        best, bf = None, -1.0
        for t in range(1, 256):                                                           # e0d's rule, level by level, in plain python
            nE, hitE, nXJ, hitX = cnt[t, bi]
            Fv = _f_from(hitX / nXJ if nXJ > 0 else None, hitE / nE if nE > 0 else None)
            if Fv > bf:
                best, bf = t, Fv
        lv, c = F.level_from_edge_hist(hist[bi])
        assert lv == best and c["f"][lv] == bf


def test_the_max_filter_identity_and_e0d_dilate():
    e0d = _e0d()
    rng = np.random.default_rng(3)
    for _ in range(5):
        m = rng.random((53, 41)) < 0.1
        for r in (0, 1, 3):
            assert np.array_equal(E.edge_dilate(m, r), e0d.dilate(m, r))
        p = (rng.random((53, 41)) * 256).astype(np.uint8) * (rng.random((53, 41)) < 0.3)
        for t in (1, 2, 17, 128, 255):
            assert np.array_equal(E.edge_maxfilter(p) >= t, E.edge_dilate(p >= t, 3))


# ================================================================================================================= the e0d control
def test_control_reproduces_e0d_edge_prf_and_threshold_choice_on_its_grid():
    e0d = _e0d()
    assert e0d.TOL_EDGE == E.EDGE_TOL_CELLS == 3 and e0d.NOCLS == E.EDGE_NOCLS and e0d.UNSEEN == E.EDGE_UNSEEN and e0d.EDGE == E.EDGE_CLASS
    rng = np.random.default_rng(11)
    rasters = [_rand_raster(rng, h=130, w=70) for _ in range(4)]
    edges = (0, 50, 130)                                                                    # rows: two "bands" covering the whole raster
    bands_m = [[0, 5], [5, 13]]
    thr = e0d.THR_EDGE
    stats = np.zeros((len(thr), 2, 5))
    for pe, gt in rasters:
        st = e0d.gt_static(gt, edges)
        for i, t in enumerate(thr):
            stats[i] += e0d._edge_stats(pe >= t, st)                                        # e0d's OWN counters: nE hitE nXJ hitX nX
    hist = sum(F.edge_hists_arrays(pe, gt, bands_m) for pe, gt in rasters)
    brute = _brute_levels(rasters, bands_m)
    for bi in range(2):
        c = F.edge_curve(hist[bi])
        for i, t in enumerate(thr):
            s = stats[i, bi]
            assert (c["n_e"], c["hit_e"][t], c["n_xj"][t], c["hit_x"][t]) == (s[0], s[1], s[2], s[3])          # the counts, exactly
            assert tuple(brute[t, bi]) == (s[0], s[1], s[2], s[3])
            R = s[1] / s[0] if s[0] > 0 else None
            P = s[3] / s[2] if s[2] > 0 else None
            assert abs(c["r"][t] - R) < 1e-12 and abs(c["f"][t] - e0d.f_from(P, R)) < 1e-12
            if P is not None:
                assert abs(c["p"][t] - P) < 1e-12
        # e0d's choice over ITS grid == the first-max of our curve restricted to that grid (when F > 0)
        t_e0d, F_e0d, P_e0d, R_e0d = e0d.fit_grid_threshold(stats, thr, (bi,))
        grid_f = [c["f"][t] for t in thr]
        assert F_e0d > 0 and t_e0d == thr[int(np.argmax(grid_f))] and abs(F_e0d - max(grid_f)) < 1e-12
        # and the unrestricted 1..255 optimum is at least as good as the grid's
        assert F.level_from_edge_hist(hist[bi])[1]["f"].max() >= F_e0d - 1e-12


def test_control_e0d_gt_static_masks_are_ours():
    e0d = _e0d()
    pe, gt = _rand_raster(np.random.default_rng(5))
    st = e0d.gt_static(gt, (0, 50, 130))
    e, edil, j = E.edge_gt_masks(gt)
    assert np.array_equal(e, st["E"]) and np.array_equal(edil, st["Edil"]) and np.array_equal(j, st["J"])


# ================================================================================================================= the criterion in the file
def test_pedge_f_tol_mixed_file_keeps_psw_on_iou_records_the_criterion_per_plane_and_the_loader_reads_it(tmp_path):
    mix, d, sp = _hand_fit(tmp_path, "mix.json")
    base = F.fit(d / "rast", sp, tmp_path / "base.json", "T", planes=("psw",), bands_s="0-2,2-4")        # (pedge by IoU has no cut on this world: no flag ON an edge cell)
    assert mix["planes"]["psw"] == base["planes"]["psw"] and mix["planes"]["pedge"] == [51, 31]
    assert mix["criterion"]["per_plane"]["pedge"]["criterion"] == "f_tol" and mix["criterion"]["per_plane"]["psw"]["criterion"] == "iou"
    assert mix["criterion"]["per_plane"]["pedge"]["mirrors"].startswith("e0d_score.py")
    back = E.load_map_thresholds(tmp_path / "mix.json")                                  # the evaluator's own loader, unchanged
    assert back.planes == mix["planes"] and back.level("pedge", 0, 2) == 51 and back.level("pedge", 2, 4) == 31 and back.level("psw", 0, 1) == base["planes"]["psw"][0]
    assert E.thresholds_binding(back, d / "rast") == {"checked": True, "ckpt_md5": "ck1"}


def test_default_call_is_byte_identical_and_adds_no_key(tmp_path):
    d, sp = _dump(tmp_path, _simple)                                # psw AND pedge both have an IoU cut here
    F.fit(d / "rast", sp, tmp_path / "a.json", "T", bands_s="0-2,2-4")
    F.fit(d / "rast", sp, tmp_path / "b.json", "T", bands_s="0-2,2-4", plane_criteria={"psw": "iou", "pedge": "iou"})
    F.fit(d / "rast", sp, tmp_path / "c.json", "T", bands_s="0-2,2-4", plane_criteria=[])
    ba = (tmp_path / "a.json").read_bytes()
    assert ba == (tmp_path / "b.json").read_bytes() == (tmp_path / "c.json").read_bytes()
    o = json.loads(ba)
    assert set(o["criterion"]) == {"name", "population", "gt_classes", "valid_cells"} and o["criterion"]["name"] == F.CRITERION
    assert "per_plane" not in o["criterion"] and o["provenance"]["fit_detail"]["pedge"][0].keys() >= {"iou_at_level", "iou_at_128"}


def test_cli_flag_and_refusals(tmp_path, capsys):
    d, sp = _dump(tmp_path, _hand)
    common = ["fit", "--rast-dir", str(d / "rast"), "--dev-hold", str(sp), "--arm", "T", "--bands", "0-2,2-4"]
    assert F.main(common + ["--plane-criterion", "pedge=f_tol", "--out", str(tmp_path / "o.json")]) == 0
    assert json.loads((tmp_path / "o.json").read_text())["planes"]["pedge"] == [51, 31]
    for bad, why in (("psw=f_tol", "EDGE class"), ("pedge=nope", "criterion must be"), ("pdrv=iou", "not among the fitted planes"), ("pedge", "want plane=criterion")):
        assert F.main(common + ["--plane-criterion", bad, "--out", str(tmp_path / "bad.json")]) == 2
        assert why in capsys.readouterr().err, bad
    assert not (tmp_path / "bad.json").exists()


def test_corridor_population_counts_the_corridor_only_but_dilates_the_whole_raster():
    pe = np.zeros((40, 600), np.uint8)
    gt = np.zeros((40, 600), np.uint8)
    gt[10, 100:110] = 5                      # 10 edge cells OUTSIDE the corridor (cols 270-329)
    gt[10, 271:275] = 5                      # 4 edge cells INSIDE it, 1-4 cells from its left border
    pe[10, 268] = 90                         # a flag OUTSIDE the corridor, 3 cells from col 271: dilation (whole raster) recalls it
    h_all = F.edge_hists_arrays(pe, gt, [[0, 4]], "all")[0]
    h_cor = F.edge_hists_arrays(pe, gt, [[0, 4]], "corridor")[0]
    assert h_all[0].sum() == 14 and h_cor[0].sum() == 4
    assert h_cor[0][90] == 1 and h_cor[0][0] == 3                    # only col 271 sees the flag (|268 - 271| = 3); 272-274 are 4-6 away
    assert h_all[0][90] == 1 and h_all[0][0] == 13


# ================================================================================================================= MUTATIONS (each guard must go RED)
def test_MUTATION_tolerance_3_to_0_makes_the_guard_red(tmp_path, monkeypatch):
    (tmp_path / "ok").mkdir()
    (tmp_path / "bad").mkdir()
    assert _hand_guard(tmp_path / "ok") is True
    monkeypatch.setattr(E, "EDGE_TOL_CELLS", 0)
    assert _hand_guard(tmp_path / "bad") is False


def test_MUTATION_p_and_r_swapped_makes_the_guard_red(tmp_path, monkeypatch):
    orig = F.edge_curve

    def swapped(h):
        c = orig(h)
        return dict(c, p=c["r"], r=c["p"])
    monkeypatch.setattr(F, "edge_curve", swapped)
    # F = 2PR/(P+R) is symmetric: the LEVEL cannot move, so the guard that catches this is the recorded P and R (band 1: P 1, R 1/2)
    assert _hand_guard(tmp_path) is False


def test_MUTATION_dilation_on_the_wrong_mask_makes_the_guard_red(tmp_path, monkeypatch):
    real = E.edge_gt_masks

    def wrong(gt):                                                   # Edil taken from the JUDGED mask instead of the GT edge
        e, _edil, j = real(gt)
        return e, E.edge_dilate(j), j
    monkeypatch.setattr(E, "edge_gt_masks", wrong)
    assert _hand_guard(tmp_path) is False


def test_MUTATION_recall_without_dilating_the_flags_makes_the_guard_red(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "edge_maxfilter", lambda plane, r=None: np.asarray(plane))
    assert _hand_guard(tmp_path) is False


def test_MUTATION_nocls_counted_as_judged_makes_the_guard_red(tmp_path, monkeypatch):
    real = E.edge_gt_masks

    def valid_only(gt):
        e, edil, _j = real(gt)
        return e, edil, np.asarray(gt) != 255
    monkeypatch.setattr(E, "edge_gt_masks", valid_only)
    assert _hand_guard(tmp_path) is False                           # the "judged" definition is part of the rule (e0d_score.gt_static J)


def test_MUTATION_the_per_plane_criterion_ignored_pedge_fitted_by_iou_makes_the_guard_red(tmp_path, monkeypatch):
    d, sp = _dump(tmp_path, _hand)
    monkeypatch.setattr(F, "parse_plane_criteria", lambda spec, planes: {p: "iou" for p in planes})
    with pytest.raises(F.FitRefused, match="plane pedge band 2-4 m"):      # IoU on the plane: band 0 would take 151, band 1 has no flag ON a GT edge cell -> level 0 -> refused
        F.fit(d / "rast", sp, tmp_path / "m.json", "T", bands_s="0-2,2-4", plane_criteria=["pedge=f_tol"])
    monkeypatch.undo()
    pe, gt = _hand()
    h = F.raster_hists(d / "rast" / "e000_w0007.npz", ("pedge",), [[0, 2], [2, 4]])        # the IoU criterion's own histograms, band 0
    assert F.level_from_hist(h[0, 0, 0], h[0, 0, 1])[0] == 151 and F.level_from_edge_hist(F.edge_hists_arrays(pe, gt, [[0, 2]])[0])[0] == 51
    (tmp_path / "ok").mkdir()
    assert _hand_guard(tmp_path / "ok") is True                           # the real (un-mutated) call honours the flag
    monkeypatch.setattr(F, "parse_plane_criteria", lambda spec, planes: {p: "iou" for p in planes})
    (tmp_path / "bad").mkdir()
    assert _hand_guard(tmp_path / "bad") is False


def test_MUTATION_first_level_on_ties_replaced_by_last_makes_the_guard_red(tmp_path, monkeypatch):
    real = F.level_from_edge_hist

    def last(h):
        lv, c = real(h)
        fs = c["f"][1:]
        return int(len(fs) - 1 - np.argmax(fs[::-1])) + 1, c
    monkeypatch.setattr(F, "level_from_edge_hist", last)
    assert _hand_guard(tmp_path) is False
