"""r9_e1_eval.py's refcv9 M1 MAP PRIMARY (REFCV9_DESIGN 4.3 / 4.4), known answers on synthetic rasters.

M1's primary is "corridor sidewalk+edge precision 5-20 m (lift-valid)"; M2a / M2b / M3 / F1 reuse it, A1 guards on it and E2's GO rule reads
"K2 beats K0 on the map primary". The evaluator used to emit SIDEWALK-only precision (``nf_prec_*``: flag ``psw``, GT class 7, GT-seen cells,
no lift mask) and nothing else for the map. This pins the additive keys:

  * ``nf_swedge_prec_{0_5,5_10,10_20,20_40,5_20}``      UNION flag (psw | pedge) vs GT {5 edge, 7 sidewalk}, GT-seen cells (lift-INDEPENDENT);
  * ``nf_swedge_prec_lv_{0_5,5_10,10_20,20_40,5_20}``   the same on lift-valid cells (the shared, ARM-INDEPENDENT ``--lift-valid`` mask);
    ``*_5_20`` pools the 5-10 and 10-20 m COUNTS (never the mean of two precisions).

Layout of the fixture (corridor cols 270-329, every cell GT-seen drivable unless a block says otherwise; flags 255):
  band 0-5 m   H  gt1 psw  100 cells  lift-INVALID (the blind zone: a false positive nobody could have seen)   I  gt7 psw 10 cells
  band 5-10 m  A  gt5 pedge 20 (edge TP)   B  gt7 psw 30 (sw TP)   C  gt1 psw 20 (FP)   D  gt7 psw 25 lift-INVALID   E  gt1 pedge 40 lift-INVALID
               U  gt255 psw 15 (GT-UNSEEN: counts nowhere)
  band 10-20 m F  gt7 psw 10 (TP)   G  gt1 psw 90 (FP)
Each known answer differs from the one next to it, so a mutated definition cannot hide behind a coincidence. The RED arms run the SAME
checks against (a) the pooled hold's evaluator (``R9E1_POOLED_EVAL_PY``, skipped when unset) and (b) in-process mutations; both must fail.
"""
import importlib.util
import json
import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_S = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_S))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r9_e1_eval as E                                   # noqa: E402
import test_r9_e1_eval as TE                             # noqa: E402  (its pack fixtures, reused unchanged)

pytest.importorskip("scipy.ndimage")

H_, W_ = 1000, 600
C0 = 270                                                 # corridor col 0

#: name -> (row0, row1, col0, col1, gt class, flagged by "psw" | "pedge" | None, lift-valid)
BLOCKS = {
    "H": (0, 10, 270, 280, 1, "psw", False),
    "I": (20, 21, 270, 280, 7, "psw", True),
    "A": (50, 52, 270, 280, 5, "pedge", True),
    "B": (52, 55, 270, 280, 7, "psw", True),
    "C": (55, 57, 270, 280, 1, "psw", True),
    "D": (57, 62, 270, 275, 7, "psw", False),
    "E": (62, 66, 270, 280, 1, "pedge", False),
    "U": (70, 73, 270, 275, 255, "psw", True),
    "F": (100, 101, 270, 280, 7, "psw", True),
    "G": (102, 112, 270, 279, 1, "psw", True),
}


def _n(k):
    r0, r1, c0, c1 = BLOCKS[k][:4]
    return (r1 - r0) * (c1 - c0)


assert [_n(k) for k in "HIABCDEUFG"] == [100, 10, 20, 30, 20, 25, 40, 15, 10, 90]


def _layout(drop=(), only=None, edge_plane=True):
    gt = np.ones((H_, W_), np.uint8)
    psw, pedge = np.zeros((H_, W_), np.uint8), np.zeros((H_, W_), np.uint8)
    lv = np.ones(E.LV_SHAPE, bool)
    for k, (r0, r1, c0, c1, cls, by, valid) in BLOCKS.items():
        if k in drop or (only is not None and k not in only):
            continue
        gt[r0:r1, c0:c1] = cls
        if by == "psw":
            psw[r0:r1, c0:c1] = 255
        elif by == "pedge":
            pedge[r0:r1, c0:c1] = 255
        if not valid:
            lv[r0:r1, c0 - C0:c1 - C0] = False
    arrs = {"gt": gt, "psw": psw, "pdrv": np.zeros((H_, W_), np.uint8), "code": np.zeros((H_, W_), np.uint8)}
    if edge_plane:
        arrs["pedge"] = pedge
    return arrs, lv


def _unobs_plane(lv):
    un = np.zeros((H_, W_), np.uint8)
    un[:E.LV_ROWS, C0:C0 + E.LV_SHAPE[1]] = (~lv).astype(np.uint8)
    return un


def _write(d, name, arrs, unobs_of=None):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    a = dict(arrs)
    if unobs_of is not None:
        a["unobs"] = _unobs_plane(unobs_of)
    np.savez_compressed(d / name, **a)
    return str(d / name)


def _same(a, b):
    return set(a) == set(b) and all((a[k] != a[k] and b[k] != b[k]) or a[k] == b[k] for k in a)       # NaN == NaN here


def _stats(mod, files, lv):
    names, acc, _g, info = mod.nearfield_per_file_ex(files, lift_valid=lv)
    return mod.nearfield_stats([acc[i] for i in range(len(names))]), acc, info


# --------------------------------------------------------------------------------------------------- the known answers (shared with the RED arms)
def check_union_counts_flag_on_gt_edge_as_tp(mod, tmp):
    arrs, lvm = _layout()
    f = _write(tmp, "e000_w0000.npz", arrs)
    s, _acc, info = _stats(mod, [f], mod.LiftValid({0: lvm}, "test"))
    assert info["edge_plane"] == "all"
    # 5-10 m, lift-valid: flags A20+B30+C20 = 70, TP A+B = 50 (the edge flags on GT edge cells ARE true positives)
    assert s["nf_swedge_prec_lv_5_10"] == pytest.approx(50 / 70, abs=1e-12)
    # the historical figure is SIDEWALK only and must not have moved: psw flags B30+C20+D25 = 75, TP B+D = 55
    assert s["nf_prec_5_10"] == pytest.approx(55 / 75, abs=1e-12)
    assert s["nf_swedge_prec_5_10"] == pytest.approx(75 / 135, abs=1e-12)            # unmasked union: flags 135, TP A+B+D 75


def check_unobserved_cells_are_excluded(mod, tmp):
    arrs, lvm = _layout()
    f = _write(tmp, "e000_w0000.npz", arrs)
    s, _acc, _i = _stats(mod, [f], mod.LiftValid({0: lvm}, "test"))
    # H (100 psw flags on lift-INVALID drivable cells, band 0-5) is a blind-zone false positive: out of the lv key, in the unmasked one
    assert s["nf_swedge_prec_lv_0_5"] == pytest.approx(10 / 10, abs=1e-12)
    assert s["nf_swedge_prec_0_5"] == pytest.approx(10 / 110, abs=1e-12) and s["nf_prec_0_5"] == pytest.approx(10 / 110, abs=1e-12)
    # D (TP) and E (FP) are lift-invalid: both leave the band-5-10 lv figure, which is therefore NOT the unmasked one
    assert s["nf_swedge_prec_lv_5_10"] != pytest.approx(s["nf_swedge_prec_5_10"], abs=1e-3)
    assert s["nf_swedge_prec_lv_10_20"] == pytest.approx(10 / 100, abs=1e-12)
    assert s["nf_swedge_prec_10_20"] == pytest.approx(10 / 100, abs=1e-12)           # no lift-invalid cell in this band: lv == unmasked
    assert np.isnan(s["nf_swedge_prec_lv_20_40"])                                    # no flags: NaN, never 0


def check_pooled_5_20_is_summed_counts(mod, tmp):
    arrs, lvm = _layout()
    f = _write(tmp, "e000_w0000.npz", arrs)
    s, _acc, _i = _stats(mod, [f], mod.LiftValid({0: lvm}, "test"))
    assert s["nf_swedge_prec_lv_5_20"] == pytest.approx(60 / 170, abs=1e-12)         # (50 + 10) / (70 + 100)
    assert s["nf_swedge_prec_lv_5_20"] != pytest.approx((50 / 70 + 10 / 100) / 2, abs=1e-6)           # NOT the band mean 0.407
    assert s["nf_swedge_prec_5_20"] == pytest.approx(85 / 235, abs=1e-12)            # unmasked: (75 + 10) / (135 + 100)


def check_no_edge_plane_scores_sidewalk_flags_against_the_same_union(mod, tmp):
    # a raster with NO pedge plane: only the sidewalk flag exists; a psw flag on a GT-EDGE cell is still a TP of the union
    arrs, lvm = _layout(only=(), edge_plane=False)
    arrs["gt"][60:62, 270:276] = 5
    arrs["psw"][60:62, 270:276] = 255                                                # 12 flags on GT edge cells, 5-10 m
    f = _write(tmp, "e000_w0000.npz", arrs)
    s, acc, info = _stats(mod, [f], mod.LiftValid({0: lvm}, "test"))
    assert info["edge_plane"].startswith("none") and info["n_with_edge_plane"] == 0
    assert s["nf_swedge_prec_lv_5_10"] == pytest.approx(12 / 12, abs=1e-12)
    assert s["nf_prec_5_10"] == 0.0                                                   # the sidewalk-only reading calls the same 12 flags all false
    a2, lv2 = _layout(only=(), edge_plane=True)                                      # an all-zero pedge plane gives the identical union counts
    a2["gt"][60:62, 270:276] = 5
    a2["psw"][60:62, 270:276] = 255
    f2 = _write(Path(tmp) / "withedge", "e000_w0000.npz", a2)
    s2, acc2, _i2 = _stats(mod, [f2], mod.LiftValid({0: lv2}, "test"))
    assert np.array_equal(acc[:, :, 6:], acc2[:, :, 6:], equal_nan=True)


MUTABLE_CHECKS = (check_union_counts_flag_on_gt_edge_as_tp, check_unobserved_cells_are_excluded, check_pooled_5_20_is_summed_counts,
                  check_no_edge_plane_scores_sidewalk_flags_against_the_same_union)


@pytest.mark.parametrize("check", MUTABLE_CHECKS, ids=lambda c: c.__name__)
def test_known_answers(check, tmp_path):
    check(E, tmp_path)


# --------------------------------------------------------------------------------------------------- additive: nothing existing moves
def test_existing_keys_keep_their_values_and_the_legacy_block_shape_still_works(tmp_path):
    arrs, lvm = _layout()
    f = _write(tmp_path, "e000_w0000.npz", arrs)
    names, acc, gtcls, _i = E.nearfield_per_file_ex([f], lift_valid=E.LiftValid({0: lvm}, "t"))
    assert acc.shape == (1, len(E.NEARFIELD_BANDS), E.NF_COLS) and gtcls.shape == (1, 4, 9)
    assert E.nearfield_per_file([f])[1].shape == (1, 4, E.NF_COLS)                    # the 3-tuple wrapper callers unpack
    wide = E.nearfield_stats([acc[0]])
    legacy = E.nearfield_stats([acc[0][:, :E.NF_BASE_COLS]])                          # a 6-column block, as test_r9_e1_eval_pooled builds
    old = {k: v for k, v in wide.items() if not k.startswith("nf_swedge")}
    assert set(legacy) == set(old) and not any(k.startswith("nf_swedge") for k in legacy)
    for k, v in old.items():
        assert (np.isnan(v) and np.isnan(legacy[k])) or v == legacy[k], k
    tab = E.nearfield_table(acc, gtcls)
    assert tab["5-10"]["n_flags"] == 75.0 and tab["5-10"]["precision_exact"] == pytest.approx(55 / 75)
    assert tab["0-5"]["lift_invalid_frac_of_seen"] == pytest.approx(100 / 3000)       # H: 100 of the 50 x 60 = 3000 seen corridor cells


def test_without_a_mask_no_swedge_key_exists_and_no_lv_number_is_invented(tmp_path):
    arrs, _lv = _layout()
    f = _write(tmp_path, "e000_w0000.npz", arrs)
    names, acc, _g, info = E.nearfield_per_file_ex([f])
    assert info["lift_valid"] is None and np.isnan(acc[:, :, 8:11]).all() and np.isfinite(acc[:, :, :8]).all()
    s = E.nearfield_stats([acc[0]])
    assert not [k for k in s if k.startswith("nf_swedge")]                          # none: a run that never asked for the primary keeps its row set
    assert set(s) == set(E.nearfield_stats([acc[0][:, :E.NF_BASE_COLS]]))           # exactly the legacy key set (test_r9_e1_floor pins its row count)
    assert "lift_invalid_frac_of_seen" not in E.nearfield_table(acc, np.zeros((1, 4, 9)))["0-5"]


# --------------------------------------------------------------------------------------------------- the mask: one object, any arm, loud when it disagrees
def test_a_c0_style_raster_is_scored_on_a_mask_harvested_from_a_banded_arm(tmp_path):
    arrs, lvm = _layout()
    c0 = [_write(tmp_path / "c0", n, arrs) for n in ("e000_w0000.npz", "e000_w0001.npz", "e001_w0000.npz")]      # refcv8 head: NO unobs plane
    b = [_write(tmp_path / "banded", n, arrs, unobs_of=lvm) for n in ("e000_w0000.npz", "e000_w0001.npz", "e001_w0000.npz")]
    lv = E.load_lift_valid(tmp_path / "banded")
    assert sorted(lv.masks) == [0, 1] and np.array_equal(lv.masks[0], lvm) and "harvest" in lv.source
    s_c0, _a, i_c0 = _stats(E, c0, lv)
    s_b, _a2, i_b = _stats(E, b, lv)
    assert i_c0["n_unobs_checked"] == 0 and i_b["n_unobs_checked"] == 3             # a banded arm's own plane is CHECKED against the shared mask
    assert _same(s_c0, s_b) and s_c0["nf_swedge_prec_lv_5_20"] == pytest.approx(60 / 170, abs=1e-12)
    # sidecar round trip: same masks, same sha
    E.save_lift_valid(lv, tmp_path / "lv.npz")
    lv2 = E.load_lift_valid(tmp_path / "lv.npz")
    assert lv2.sha256 == lv.sha256 and lv2.source.startswith("sidecar:") and lv.record()["n_episodes"] == 2


def test_the_mask_refuses_rather_than_guesses(tmp_path):
    arrs, lvm = _layout()
    f = _write(tmp_path / "r", "e000_w0000.npz", arrs)
    with pytest.raises(E.LiftValidError, match="no lift-valid mask for episode e0"):         # an episode the mask does not hold
        E.nearfield_per_file_ex([f], lift_valid=E.LiftValid({7: lvm}, "t"))
    other = lvm.copy()
    other[60:70, 0:5] = ~other[60:70, 0:5]
    fb = _write(tmp_path / "r2", "e000_w0000.npz", arrs, unobs_of=other)                    # the arm's own plane contradicts the shared mask
    with pytest.raises(E.LiftValidError, match="disagrees"):
        E.nearfield_per_file_ex([fb], lift_valid=E.LiftValid({0: lvm}, "t"))
    _write(tmp_path / "h", "e000_w0000.npz", arrs, unobs_of=lvm)
    _write(tmp_path / "h", "e000_w0001.npz", arrs, unobs_of=other)                          # one episode, two windows, two masks: not a clip property
    with pytest.raises(E.LiftValidError, match="per-clip"):
        E.load_lift_valid(tmp_path / "h")
    with pytest.raises(E.LiftValidError, match="no raster"):                                  # a directory of rasters with no unobs plane
        E.load_lift_valid(tmp_path / "r")
    with pytest.raises(E.LiftValidError, match="shape"):
        E.LiftValid({0: np.ones((10, 10), bool)}, "t")
    with pytest.raises(E.LiftValidError, match="neither"):
        E.load_lift_valid(tmp_path / "absent.npz")


def test_a_banded_arms_unobs_equals_the_arm_independent_lift_mask_over_0_40_m():
    """WHY one mask serves every arm: the banded head's `unobs` is the lift's OWN validity resampled per band (near 0-10 m 10 cm, mid 10-40 m
    the 0.25 m cell, far >= 40 m 2 x 2 conservative) and refcv8 applies ``lift_valid_to_fine`` to the same `valid.any(dim=1)`. Over 0-40 m (the
    evaluator's bands) the two are bit-identical; only the far band (>= 40 m, not scored here) differs."""
    torch = pytest.importorskip("torch")
    MB = pytest.importorskip("tanitad.models.map_banded")
    MH = pytest.importorskip("tanitad.models.map_head_hires")
    MPL = pytest.importorskip("tanitad.eval.map_planes")
    g = torch.Generator().manual_seed(0)
    lv = torch.rand(2, 400, 240, generator=g) > 0.35                       # [B, X4, Y4]: 100 m x 60 m at 0.25 m
    geom = MB.band_geometry(100.0, 30.0)
    logits = [torch.zeros(2, MB.N_CHANNELS, gg.rows, gg.cols) for gg in geom]
    planes = MPL.planes_from_banded({"logits": logits, "unobs": MB.unobs_masks(lv, geom)}, x_max_m=100.0, y_half_m=30.0)
    fine = MH.lift_valid_to_fine(lv)                                        # refcv8's path
    assert tuple(planes.unobs.shape) == tuple(fine.shape) == (2, 1000, 600)
    assert torch.equal(planes.unobs[:, :E.LV_ROWS], ~fine[:, :E.LV_ROWS])
    # and the evaluator's stored form of that mask is what the harvest reads off the plane
    un = planes.unobs[0].to(torch.uint8).numpy()
    h = ~un[:E.LV_ROWS, E.CORRIDOR_COLS[0]:E.CORRIDOR_COLS[1]].astype(bool)
    assert np.array_equal(h, fine[0, :E.LV_ROWS, E.CORRIDOR_COLS[0]:E.CORRIDOR_COLS[1]].numpy())


# --------------------------------------------------------------------------------------------------- the CLI
def test_cli_emits_the_primary_with_the_mask_and_withholds_the_lv_keys_without_it(tmp_path):
    arrs, lvm = _layout()
    for n in ("e000_w0000.npz", "e001_w0000.npz"):
        _write(tmp_path / "rast", n, arrs)
        _write(tmp_path / "banded", n, arrs, unobs_of=lvm)
    (tmp_path / "a.pkl").write_bytes(pickle.dumps([{"sha12": s, "ws": w, "box3d": [pk]} for (s, w, pk) in TE._records()]))
    base = ["--packs", str(tmp_path / "a.pkl"), "--gate", "0.5", "--boot", "20", "--rast-dir", str(tmp_path / "rast")]
    o1, o2 = tmp_path / "with.json", tmp_path / "without.json"
    assert E.main(base + ["--lift-valid", str(tmp_path / "banded"), "--lift-valid-save", str(tmp_path / "side.npz"), "--out", str(o1)]) == 0
    assert E.main(base + ["--out", str(o2)]) == 0
    d1, d2 = (json.loads(o.read_text(encoding="utf-8")) for o in (o1, o2))
    nf1, nf2 = d1["nearfield"], d2["nearfield"]
    assert nf1["ci"]["nf_swedge_prec_lv_5_20"]["point"] == pytest.approx(60 / 170, abs=1e-12) and nf1["edge_plane"] == "all"
    assert nf1["lift_valid"]["sha256"] == E.load_lift_valid(tmp_path / "side.npz").sha256 and nf1["lift_valid"]["n_episodes"] == 2
    assert not [k for k in nf2["ci"] if k.startswith("nf_swedge")] and nf2["lift_valid"] is None
    assert len([k for k in nf1["ci"] if k.startswith("nf_swedge")]) == 10           # 4 bands + 5_20, unmasked and lift-valid
    for k in nf2["ci"]:                                                    # every key the run without a mask has, the run with one has with the SAME point
        a, b = nf1["ci"][k]["point"], nf2["ci"][k]["point"]
        assert (a != a and b != b) or a == b, k
    with pytest.raises(SystemExit, match="lift-valid mask refused"):      # a mask that misses an episode: the run stops
        bad = tmp_path / "bad.npz"
        E.save_lift_valid(E.LiftValid({0: lvm}, "t"), bad)
        E.main(base + ["--lift-valid", str(bad), "--out", str(tmp_path / "x.json")])


# --------------------------------------------------------------------------------------------------- RED: the same checks must FAIL on a wrong evaluator
def _mutants():
    """In-process regressions of the evaluator: each breaks exactly one of the four definitions."""
    def ns(**over):
        m = types.SimpleNamespace(**{k: getattr(E, k) for k in dir(E) if not k.startswith("__")})
        for k, v in over.items():
            setattr(m, k, v)
        return m

    def ignores_edge_plane(files, *a, **kw):
        return E.nearfield_per_file_ex(files, *a, **{**kw, "edge_key": "pedge_ABSENT"})

    def ignores_mask(files, *a, **kw):
        return E.nearfield_per_file_ex(files, *a, **{**kw, "lift_valid": None})

    def band_mean(units):
        s = E.nearfield_stats(units)
        if "nf_swedge_prec_lv_5_20" in s:
            s["nf_swedge_prec_lv_5_20"] = (s["nf_swedge_prec_lv_5_10"] + s["nf_swedge_prec_lv_10_20"]) / 2
        return s

    def sidewalk_only_gt(files, *a, **kw):
        names, acc, g, info = E.nearfield_per_file_ex(files, *a, **kw)
        acc = acc.copy()
        acc[:, :, 6], acc[:, :, 7] = acc[:, :, 0], acc[:, :, 1]               # the historical counts, relabelled
        return names, acc, g, info

    return {"edge_plane_ignored": ns(nearfield_per_file_ex=ignores_edge_plane),
            "mask_ignored": ns(nearfield_per_file_ex=ignores_mask),
            "pooled_as_band_mean": ns(nearfield_stats=band_mean),
            "historical_sidewalk_counts_relabelled": ns(nearfield_per_file_ex=sidewalk_only_gt)}


@pytest.mark.parametrize("name", sorted(_mutants()))
def test_RED_each_mutated_evaluator_fails_at_least_one_known_answer(name, tmp_path):
    mod = _mutants()[name]
    failed = []
    for i, check in enumerate(MUTABLE_CHECKS):
        try:
            check(mod, tmp_path / str(i))
        except (AssertionError, KeyError):
            failed.append(check.__name__)
    assert failed, "the mutation %r passed every known answer: the checks cannot see this regression" % name


def test_RED_the_pooled_holds_evaluator_fails_the_known_answers(tmp_path):
    """The brief's RED arm: the evaluator BEFORE this change (the pooled hold's r9_e1_eval.py) must fail these checks. Point R9E1_POOLED_EVAL_PY at it."""
    p = os.environ.get("R9E1_POOLED_EVAL_PY")
    if not p or not Path(p).is_file():
        pytest.skip("R9E1_POOLED_EVAL_PY not set (the in-process mutants above are the standing RED arm)")
    spec = importlib.util.spec_from_file_location("r9_e1_eval_pooled_hold", p)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    for i, check in enumerate(MUTABLE_CHECKS):
        with pytest.raises((AssertionError, AttributeError, KeyError, TypeError)):
            check(old, tmp_path / str(i))
