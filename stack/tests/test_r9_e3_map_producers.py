"""r9_e1_eval.py's two E3 producers, known answers on synthetic rasters (no real data, no HOLD split touched).

r9_e3_verdict.py reads ``nearfield.ci.nf_prec_lv_0_5`` (MAP-2, 0-5 m: SIDEWALK-ONLY precision on LIFT-VALID cells) and ``nearfield.unobs_flags``
{sidewalk, edge} (MAP-5, a structural identity) and reported both NOT-COMPUTABLE because nothing emitted them.

Fixture (corridor cols 270-329, band 0-5 m = rows 0-49; every cell GT-seen drivable gt1 unless a block says otherwise; flags 255):
  H  gt1  psw 100 cells  lift-INVALID (a blind-zone false positive)      I  gt7 psw 10 cells lift-valid (TP)
  J  gt1  psw   5 cells  lift-valid (FP)                                  K  gt7 psw 20 cells lift-INVALID (a TP the mask removes)
  unmasked nf_prec_0_5 = 30 / 135 ; lift-valid nf_prec_lv_0_5 = 10 / 15 (the two differ, and each differs from 1.0 and from 0.5)
The RED arms run the SAME checks against in-process mutants (mask dropped; observed cells scored); both must FAIL.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

_S = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_S))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r9_e1_eval as E                                   # noqa: E402
import test_r9_e1_eval as TE                             # noqa: E402  (its pack fixtures, reused unchanged)

_SHIM = Path(__file__).resolve().parents[1] / "ops" / "r9e1_read" / "shim_scipy"


def _real_scipy() -> bool:
    try:
        import scipy.ndimage                             # noqa: F401
    except ModuleNotFoundError:
        return False
    return not getattr(sys.modules.get("scipy.ndimage"), "IS_SHIM", False)


@pytest.fixture(autouse=True)
def _scipy_or_the_reads_own_shim(monkeypatch):
    """Thor's training venv has no scipy, so the read runs these producers on ``stack/ops/r9e1_read/shim_scipy`` (R9E1_FORCE_SCIPY_SHIM),
    which test_r9_e1_floor::test_scipy_shim_equals_scipy_binary_dilation pins to scipy bit for bit. Real scipy where it exists, else that
    shim -- per test, and purged afterwards so no later test file imports the shim as if it were scipy. (11-11: without this the 8
    producer tests failed outside the pinned list on Thor and stopped the e2 gate.)"""
    if _real_scipy():
        yield
        return
    assert (_SHIM / "scipy" / "ndimage" / "__init__.py").is_file(), _SHIM
    monkeypatch.syspath_prepend(str(_SHIM))
    yield
    for k in [k for k in sys.modules if k == "scipy" or k.startswith("scipy.")]:
        del sys.modules[k]

H_, W_ = 1000, 600
C0 = 270
BLOCKS = {  # name -> (row0, row1, col0, col1, gt class, lift-valid)
    "H": (0, 10, 270, 280, 1, False),
    "I": (20, 21, 270, 280, 7, True),
    "J": (22, 23, 270, 275, 1, True),
    "K": (30, 32, 270, 280, 7, False),
}


def _layout():
    gt = np.ones((H_, W_), np.uint8)
    psw = np.zeros((H_, W_), np.uint8)
    lv = np.ones(E.LV_SHAPE, bool)
    for r0, r1, c0, c1, cls, valid in BLOCKS.values():
        gt[r0:r1, c0:c1] = cls
        psw[r0:r1, c0:c1] = 255
        if not valid:
            lv[r0:r1, c0 - C0:c1 - C0] = False
    return {"gt": gt, "psw": psw, "pedge": np.zeros((H_, W_), np.uint8), "pdrv": np.zeros((H_, W_), np.uint8), "code": np.zeros((H_, W_), np.uint8)}, lv


def _unobs(lv):
    un = np.zeros((H_, W_), np.uint8)
    un[:E.LV_ROWS, C0:C0 + E.LV_SHAPE[1]] = (~lv).astype(np.uint8)
    return un


def _write(d, name, arrs, **extra):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / name, **dict(arrs, **extra))
    return str(d / name)


def _same(a, b):
    return set(a) == set(b) and all((a[k] != a[k] and b[k] != b[k]) or a[k] == b[k] for k in a)


def _stats(mod, files, lv, gates=True):
    names, acc, _g, info = (mod.nearfield_per_file_ex_gates if gates else mod.nearfield_per_file_ex_edge)(files, lift_valid=lv)
    return mod.nearfield_stats([acc[i] for i in range(len(names))]), acc, info


# --------------------------------------------------------------------------------------------------- MAP-2: nf_prec_lv_*
def check_lv_precision_known_answer(mod, tmp):
    arrs, lv = _layout()
    f = _write(tmp, "e000_w0000.npz", arrs, unobs=_unobs(lv))
    s, _, _ = _stats(mod, [f], mod.LiftValid({0: lv}, "t"))
    assert s["nf_prec_lv_0_5"] == pytest.approx(10 / 15, abs=1e-12)
    assert s["nf_prec_0_5"] == pytest.approx(30 / 135, abs=1e-12)           # the historical, unmasked key keeps its own value
    assert s["nf_prec_lv_5_10"] != s["nf_prec_lv_5_10"]                    # no flag in 5-10 m: 0 / 0 is NaN, not 1 and not 0


def test_lv_precision_known_answer(tmp_path):
    check_lv_precision_known_answer(E, tmp_path)


def test_lv_precision_is_withheld_without_a_mask(tmp_path):
    arrs, _ = _layout()
    f = _write(tmp_path, "e000_w0000.npz", arrs)
    s, acc, _ = _stats(E, [f], None)
    assert not [k for k in s if k.startswith("nf_prec_lv")] and np.isnan(acc[:, :, E.G_OFF:]).all()      # not measured is not 0


def test_lv_matches_the_mask_of_the_swedge_primary(tmp_path):
    """The mask is the SAME object as nf_swedge_prec_lv_*: on a sidewalk-only raster (no edge plane) the two keys are the one number."""
    arrs, lv = _layout()
    arrs.pop("pedge")
    f = _write(tmp_path, "e000_w0000.npz", arrs)
    s, _, _ = _stats(E, [f], E.LiftValid({0: lv}, "t"))
    for b in ("0_5", "5_10", "10_20", "20_40"):
        a, c = s["nf_prec_lv_" + b], s["nf_swedge_prec_lv_" + b]
        assert (a != a and c != c) or a == c, b


# --------------------------------------------------------------------------------------------------- MAP-5: unobs_flags
def _unobs_fixture(plant_psw=0, plant_pedge=0, plant_low=0, observed_flags=True, unobs_plane=True, edge_plane=True):
    gt = np.ones((H_, W_), np.uint8)
    psw, pedge, un = (np.zeros((H_, W_), np.uint8) for _ in range(3))
    un[0:50, 270:300] = 1                                                 # unobserved block
    if observed_flags:
        psw[60:70, 270:300] = 255                                         # flags on OBSERVED cells: must NOT be counted
        pedge[60:70, 270:300] = 255
    psw[0, 270:270 + plant_psw] = 255                                     # planted flags on unobs cells
    pedge[1, 270:270 + plant_pedge] = 255
    psw[2, 270:270 + plant_low] = 100                                     # below the level: not a flag
    a = {"gt": gt, "psw": psw}
    if edge_plane:
        a["pedge"] = pedge
    if unobs_plane:
        a["unobs"] = un
    return a


def check_unobs_known_values(mod, tmp):
    f0 = _write(tmp / "z", "e000_w0000.npz", _unobs_fixture(unobs_plane=False))
    assert mod.unobs_flags_per_file([f0])["sidewalk"] is None              # no unobs plane: not measured (None), never 0
    f1 = _write(tmp / "a", "e000_w0000.npz", _unobs_fixture(plant_low=7))
    r = mod.unobs_flags_per_file([f1])
    assert (r["sidewalk"], r["edge"], r["n_with_unobs"], r["n_unobs_cells"]) == (0, 0, 1, 50 * 30)         # zero flags on unobs cells: exactly 0
    f2 = _write(tmp / "b", "e000_w0000.npz", _unobs_fixture(plant_psw=1, plant_pedge=3))
    r = mod.unobs_flags_per_file([f2])
    assert (r["sidewalk"], r["edge"]) == (1, 3)                            # one planted sidewalk flag reads exactly 1, three edge flags exactly 3
    f3 = _write(tmp / "c", "e000_w0000.npz", _unobs_fixture(plant_psw=2, edge_plane=False))
    r = mod.unobs_flags_per_file([f3])
    assert (r["sidewalk"], r["edge"]) == (2, None)                         # no edge plane: edge not measured


def test_unobs_known_values(tmp_path):
    check_unobs_known_values(E, tmp_path)


def test_unobs_counts_sum_over_files_and_use_the_band_level(tmp_path):
    fa = _write(tmp_path / "a", "e000_w0000.npz", _unobs_fixture(plant_psw=2))
    fb = _write(tmp_path / "b", "e001_w0000.npz", _unobs_fixture(plant_psw=3, plant_pedge=1))
    assert (lambda r: (r["sidewalk"], r["edge"], r["n_files"]))(E.unobs_flags_per_file([fa, fb])) == (5, 1, 2)
    thr = type("T", (), {"level": staticmethod(lambda key, lo, hi: 200 if key == "psw" else 128)})()      # per-band level above the planted 100s
    fc = _write(tmp_path / "c", "e002_w0000.npz", _unobs_fixture(plant_low=4))
    arr = dict(np.load(fc))
    arr["psw"][2, 270:274] = 150
    fd = _write(tmp_path / "d", "e003_w0000.npz", arr)
    assert E.unobs_flags_per_file([fd])["sidewalk"] == 4                   # at the fixed 128 the 150s are flags ...
    assert E.unobs_flags_per_file([fd], thresholds=thr)["sidewalk"] == 0   # ... at the arm's level 200 they are not


# --------------------------------------------------------------------------------------------------- ADDITIVE: old keys byte-equal
def test_old_keys_unchanged_by_the_new_columns(tmp_path):
    arrs, lv = _layout()
    files = [_write(tmp_path, "e%03d_w0000.npz" % i, arrs, unobs=_unobs(lv)) for i in range(3)]
    lvm = E.LiftValid({i: lv for i in range(3)}, "t")
    off, a_off, _ = _stats(E, files, lvm, gates=False)
    on, a_on, _ = _stats(E, files, lvm, gates=True)
    new = {k for k in on if k.startswith("nf_prec_lv")}
    assert new == {"nf_prec_lv_0_5", "nf_prec_lv_5_10", "nf_prec_lv_10_20", "nf_prec_lv_20_40"}
    assert _same(off, {k: v for k, v in on.items() if k not in new})        # every old key, same value, nothing dropped
    assert np.array_equal(a_off, a_on[:, :, :a_off.shape[2]], equal_nan=True)
    d_off = E.nearfield_per_file_ex_edge(files, lift_valid=lvm)[3]
    d_on = E.nearfield_per_file_ex_gates(files, lift_valid=lvm)[3]
    assert {k: v for k, v in d_on.items() if k != "map_gates"} == d_off


def _records_pkl(tmp_path):
    (tmp_path / "a.pkl").write_bytes(pickle.dumps([{"sha12": s, "ws": w, "box3d": [pk]} for (s, w, pk) in TE._records()]))


def _cli(tmp_path, extra, name):
    base = ["--packs", str(tmp_path / "a.pkl"), "--gate", "0.5", "--boot", "20", "--rast-dir", str(tmp_path / "rast")]
    out = tmp_path / name
    assert E.main(base + extra + ["--out", str(out)]) == 0
    return json.loads(out.read_text(encoding="utf-8"))


def test_cli_emits_both_producers_additively(tmp_path):
    arrs, lv = _layout()
    for n in ("e000_w0000.npz", "e001_w0000.npz"):
        _write(tmp_path / "rast", n, arrs, unobs=_unobs(lv))
    _records_pkl(tmp_path)
    d = _cli(tmp_path, ["--lift-valid", str(tmp_path / "rast")], "with.json")
    ci = d["nearfield"]["ci"]
    assert ci["nf_prec_lv_0_5"]["point"] == pytest.approx(10 / 15, abs=1e-12)
    # the unobs plane of this arm marks exactly the invalid cells; the fixture's flags there (H, K) are MAP-5 violations by construction
    assert d["nearfield"]["unobs_flags"]["sidewalk"] == 2 * (100 + 20) and d["nearfield"]["unobs_flags"]["edge"] == 0
    assert d["nearfield"]["unobs_flags"]["n_with_unobs"] == 2
    d2 = _cli(tmp_path, [], "without.json")
    assert not [k for k in d2["nearfield"]["ci"] if k.startswith("nf_prec_lv")]          # no mask: no key
    for k in d2["nearfield"]["ci"]:                                                       # every old key keeps its point
        a, b = d["nearfield"]["ci"][k]["point"], d2["nearfield"]["ci"][k]["point"]
        assert (a != a and b != b) or a == b, k
    assert d2["nearfield"]["unobs_flags"] == d["nearfield"]["unobs_flags"]               # the identity needs no mask
    assert d["boxes"] == d2["boxes"]


def test_cli_without_an_unobs_plane_emits_no_unobs_flags(tmp_path):
    arrs, lv = _layout()
    _write(tmp_path / "rast", "e000_w0000.npz", arrs)
    _records_pkl(tmp_path)
    assert "unobs_flags" not in _cli(tmp_path, [], "o.json")["nearfield"]               # not measured is absent, never 0


def test_cli_refuses_a_mixed_unobs_set(tmp_path):
    arrs, lv = _layout()
    _write(tmp_path / "rast", "e000_w0000.npz", arrs, unobs=_unobs(lv))
    _write(tmp_path / "rast", "e001_w0000.npz", arrs)
    _records_pkl(tmp_path)
    with pytest.raises(SystemExit, match="one arm, one plane set"):
        _cli(tmp_path, [], "x.json")


# --------------------------------------------------------------------------------------------------- RED: the same checks must FAIL on a wrong evaluator
class _NoMask:
    """Mutant 1: the lift-valid mask is dropped (the sidewalk-only precision scored on every GT-seen cell)."""
    def __getattr__(self, k):
        return getattr(E, k)

    @staticmethod
    def nearfield_per_file_ex_gates(files, **kw):
        lvm = kw.get("lift_valid")
        n, a, c, i = E.nearfield_per_file_ex_gates(files, **kw)
        full = E.LiftValid({e: np.ones_like(m) for e, m in lvm.masks.items()}, "all-valid")
        ga = E.map_gates_per_file(files, 1, "psw", E.SIDEWALK_CLASS, full)[1]
        a = a.copy()
        a[:, :, E.G_OFF:] = ga
        return n, a, c, i


class _ScoreObserved:
    """Mutant 2: the identity is scored on OBSERVED cells (unobs inverted)."""
    def __getattr__(self, k):
        return getattr(E, k)

    @staticmethod
    def unobs_flags_per_file(files, **kw):
        import tempfile
        out = []
        with tempfile.TemporaryDirectory() as td:
            for j, f in enumerate(files):
                z = dict(np.load(f))
                if "unobs" in z:
                    z["unobs"] = (z["unobs"] == 0).astype(np.uint8)
                out.append(_write(td, "e%03d_w0000.npz" % j, z))
            return E.unobs_flags_per_file(out, **kw)


def test_red_dropping_the_lift_mask_fails_the_check(tmp_path):
    with pytest.raises(AssertionError):
        check_lv_precision_known_answer(_NoMask(), tmp_path)


def test_red_scoring_observed_cells_fails_the_check(tmp_path):
    with pytest.raises(AssertionError):
        check_unobs_known_values(_ScoreObserved(), tmp_path)


# --------------------------------------------------------------------------------------------------- ADDITIVE against the PRE-CHANGE evaluator (opt-in: needs its source)
def test_every_old_output_is_byte_equal_to_the_pre_change_evaluator(tmp_path):
    """Set ``R9E3_BASE_EVAL_PY`` to the tip's r9_e1_eval.py (``git show <tip>:stack/scripts/r9_e1_eval.py``): both CLIs run on one fixture and the JSON,
    minus the new keys, must be identical. Skipped when unset (the file is not in the tree once this lands)."""
    import importlib.util
    import os
    base_py = os.environ.get("R9E3_BASE_EVAL_PY")
    if not base_py:
        pytest.skip("R9E3_BASE_EVAL_PY not set")
    spec = importlib.util.spec_from_file_location("r9_e1_eval_base", base_py)
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    arrs, lv = _layout()
    for n in ("e000_w0000.npz", "e001_w0000.npz"):
        _write(tmp_path / "rast", n, arrs, unobs=_unobs(lv))
    _records_pkl(tmp_path)
    for extra in ([], ["--lift-valid", str(tmp_path / "rast")]):
        args = ["--packs", str(tmp_path / "a.pkl"), "--gate", "0.5", "--boot", "20", "--rast-dir", str(tmp_path / "rast")] + extra
        assert B.main(args + ["--out", str(tmp_path / "old.json")]) == 0 and E.main(args + ["--out", str(tmp_path / "new.json")]) == 0
        old, new = (json.loads((tmp_path / n).read_text(encoding="utf-8")) for n in ("old.json", "new.json"))
        nf = new["nearfield"]
        nf["ci"] = {k: v for k, v in nf["ci"].items() if not k.startswith("nf_prec_lv")}
        nf.pop("unobs_flags", None)
        nf["definition"].pop("unobs_flags", None)
        nf["definition"].pop("nf_prec_lv_*")
        assert json.dumps(old, sort_keys=True) == json.dumps(new, sort_keys=True), extra
