"""refcv9 C1 -- the SCORER ADAPTER (``tanitad/eval/map_planes.py``): ONE code path for refcv8's 8-way raster and refcv9's banded
channels. The pin that matters: refcv8's output scores IDENTICALLY through the adapter and through the old path (MAPSEL's
``own_planes``, held verbatim in ``scripts/r9_e1_dump.py``), plane for plane and through the pinned ``nearfield_prec`` reader."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import r9_e1_dump as D  # noqa: E402
import r9_e1_eval as E  # noqa: E402
from tanitad.eval import map_planes as MP  # noqa: E402
from tanitad.models import map_banded as MB  # noqa: E402
from tanitad.models import map_head_hires as H  # noqa: E402

W8 = torch.tensor([1.0, 0.5, 2.0, 3.0, 4.0, 5.0, 2.5, 0.7])


def _model(w=W8, rule="prior_corrected"):
    return SimpleNamespace(_map_hires_class_weight=w, _map_hires=SimpleNamespace(cfg=SimpleNamespace(decision_rule=rule)))


def test_refcv8_planes_are_byte_identical_to_the_old_path():
    g = torch.Generator().manual_seed(0)
    for seed in range(4):
        z = torch.randn(1, 8, 100, 60, generator=g) * 3.0
        old = D.own_planes(_model(), z)
        new = MP.to_u8(MP.planes_from_logits8(z, W8, "prior_corrected"))
        for a, b in zip(old, new[:4]):
            assert a.dtype == b.dtype and np.array_equal(a, b), seed
        assert new[4] is None                                   # no lift validity given: honestly absent, never all-observed
    zb = torch.randn(3, 8, 20, 12, generator=g) * 3.0           # a batch: every item equals the single-item original
    pl = MP.planes_from_logits8(zb, W8, "prior_corrected")
    for i in range(3):
        for a, b in zip(D.own_planes(_model(), zb[i:i + 1]), MP.to_u8(pl, i)[:4]):
            assert np.array_equal(a, b)


def test_refcv8_scores_identically_through_the_nearfield_reader(tmp_path):
    g = torch.Generator().manual_seed(1)
    ra, rb = tmp_path / "old", tmp_path / "new"
    for j in range(3):
        z = torch.randn(1, 8, 1000, 600, generator=g) * 3.0
        gt = torch.randint(0, 8, (1000, 600), generator=g, dtype=torch.uint8).numpy()
        gt[:, :40] = 255
        D.write_raster(ra, 1, 10 + 5 * j, D.own_planes(_model(), z), gt)
        D.write_raster(rb, 1, 10 + 5 * j, MP.to_u8(MP.planes_from_output({"map_hires_logits": z}, class_weight=W8,
                                                                         rule="prior_corrected"))[:4], gt)
    fa = sorted(str(p) for p in ra.glob("*.npz"))
    fb = sorted(str(p) for p in rb.glob("*.npz"))
    na, acca, ca = E.nearfield_per_file(fa)
    nb, accb, cb = E.nearfield_per_file(fb)
    assert na == nb and np.array_equal(acca, accb) and np.array_equal(ca, cb) and acca.sum() > 0
    assert MP.read_raster(fb[0])["unobs_source"] == "absent" and MP.read_raster(fb[0])["schema"] == "refcv8"


def test_refcv8_with_lift_validity_gets_the_same_unobs_the_branch_would_emit():
    z = torch.randn(1, 8, 1000, 600)
    lv = torch.ones(1, 400, 240, dtype=torch.bool)
    lv[0, :40, :100] = False
    pl = MP.planes_from_logits8(z, W8, "prior_corrected", lv)
    ref = MB.unobs_masks(lv, MB.band_geometry(100.0, 30.0))[0]            # the refcv9 near band's unobs for the same validity
    assert pl.unobs_source == "lift_valid" and torch.equal(pl.unobs[:, :100], ref)
    assert int(pl.unobs.sum()) == int(ref.sum()) > 0
    with pytest.raises(RuntimeError, match="prior_corrected"):
        MP.planes_from_logits8(z, W8, "raw")
    with pytest.raises(RuntimeError, match="absent"):
        MP.planes_from_logits8(z, None, "prior_corrected")


def _banded_out(drv_sidewalk_split=True):
    g = MB.band_geometry(100.0, 30.0)
    logits, unobs = [], []
    for b in g:
        z = torch.full((1, MB.N_CHANNELS, b.rows, b.cols), -30.0)
        z[:, 0, :, b.cols // 2:] = 30.0                         # drv on the left half (cols >= half), sdw on the other
        z[:, 2, :, :b.cols // 2] = 30.0
        logits.append(z)
        unobs.append(torch.zeros(1, b.rows, b.cols, dtype=torch.bool))
    unobs[0][:, :10] = True
    return {"logits": logits, "unobs": unobs}


def test_refcv9_planes_land_on_the_10cm_raster_with_unobs_and_a_derived_code():
    pl = MP.planes_from_output({"map_banded": _banded_out()}, x_max_m=100.0, y_half_m=30.0)
    assert pl.source == "refcv9" and pl.unobs_source == "head"
    assert tuple(pl.pdrv.shape) == (1, 1000, 600) and tuple(pl.unobs.shape) == (1, 1000, 600)
    assert float(pl.pdrv[0, 500, 450]) > 0.99 and float(pl.psw[0, 500, 450]) < 0.01
    assert float(pl.psw[0, 500, 100]) > 0.99
    assert int(pl.code[0, 500, 450]) == 1 and int(pl.code[0, 500, 100]) == 7 and int(pl.code[0, 500, 300]) in (1, 7)
    assert bool(pl.unobs[0, :10].all()) and not bool(pl.unobs[0, 10:].any())
    u8 = MP.to_u8(pl)
    assert u8[0].dtype == np.uint8 and u8[4].dtype == np.uint8 and u8[4][:10].all() and not u8[4][10:].any()
    with pytest.raises(ValueError, match="declared extent"):
        MP.planes_from_output({"map_banded": _banded_out()})
    with pytest.raises(ValueError, match="neither"):
        MP.planes_from_output({})


def test_refcv9_raster_is_read_by_the_same_reader_and_carries_its_unobs(tmp_path):
    pl = MP.planes_from_output({"map_banded": _banded_out()}, x_max_m=100.0, y_half_m=30.0)
    u8 = MP.to_u8(pl)
    p = tmp_path / "e001_w0010.npz"
    np.savez_compressed(p, pdrv=u8[0], pedge=u8[1], psw=u8[2], code=u8[3], unobs=u8[4])
    r = MP.read_raster(p)
    assert r["schema"] == "refcv9" and r["unobs_source"] == "raster" and r["unobs"].dtype == bool and r["gt"] is None
    assert r["psw"][500, 100] == 255


def test_RED_an_adapter_that_drops_the_class_prior_turns_the_old_path_identity_red(monkeypatch):
    real = MP.planes_from_logits8
    monkeypatch.setattr(MP, "planes_from_logits8", lambda z, cw, rule, lv=None: real(z, torch.ones(8), rule, lv))
    with pytest.raises(AssertionError):
        test_refcv8_planes_are_byte_identical_to_the_old_path()
