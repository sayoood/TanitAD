"""test_q5_c2.py -- C2 planted-positive tool, synthetic small dirs (CPU).
Run: C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q -p no:cacheprovider --basetemp F:/TanitAD-scratch/q5_416/build/c2/tmp/bt test_q5_c2.py
"""
import json
import os
import sys

import numpy as np
import pytest

os.environ["C2_FLOOR_GB"] = "0"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q5_c2_planted as P  # noqa: E402
import q5_read as R  # noqa: E402

D, N, NE = 16, 80, 60
TREE = os.environ.get("S0P_TREE", "F:/TanitAD-scratch/stopbal_read/tree")


def _u8(rng, n):
    import torch
    x = torch.from_numpy(rng.normal(size=(n, 40, D)).astype(np.float32))
    return x.to(torch.float8_e4m3fn).contiguous().view(torch.uint8).numpy()


def _mk(base, rng):
    ev_n = NE
    for s in ("S1", "S2"):
        d = os.path.join(base, "assembled", f"{s}_416")
        os.makedirs(d)
        y = (np.arange(N) % 4 == 0).astype(np.uint8)
        for k, a in {"E": rng.normal(size=(N, 18)).astype(np.float32), "y": y, "clip": np.array([f"c{i // 4}" for i in range(N)]), "ws": np.arange(N), "eid": np.arange(N) // 4,
                     "v40": _u8(rng, N), "v160": rng.integers(0, 120, size=(N, 160, D), dtype=np.uint8), "mean": rng.normal(size=(N, D)).astype(np.float16)}.items():
            np.save(os.path.join(d, k + ".npy"), a)
        json.dump({"k": 1}, open(os.path.join(d, "meta.json"), "w"))
        open(os.path.join(d, "sample.json"), "w").write("{}")
    y = (np.arange(ev_n) % 4 == 0).astype(np.uint8)
    np.savez(os.path.join(base, "assembled", "eval_416.npz"), ci=np.arange(ev_n), E=rng.normal(size=(ev_n, 18)).astype(np.float32), v40=_u8(rng, ev_n), v160=rng.integers(0, 120, size=(ev_n, 160, D), dtype=np.uint8),
             mean=rng.normal(size=(ev_n, D)).astype(np.float16), y=y, clip=np.array([f"e{i // 4}" for i in range(ev_n)]), ws=np.arange(ev_n), eid=np.arange(ev_n) // 4, meta=np.array(json.dumps({"m": 1})))
    return base


@pytest.fixture
def base(tmp_path):
    return _mk(str(tmp_path / "run" / "x"), np.random.default_rng(1))


def _proj(base, s, plant_on, amp=1.0):
    """AUROC of the score 'projection of the planted cell on the planted direction' (a fit-free proxy of what a linear probe can read)."""
    prov = P.make(base, ["S1"], amp=amp, plant_on=plant_on, log=lambda *_: None)
    dst = os.path.join(base, "c2", "assembled", "S1_416")
    dec, _ = P._fp8()
    v = np.load(os.path.join(dst, "v40.npy"))
    y = np.load(os.path.join(dst, "y.npy")).astype(bool)
    u = P.direction(D)
    sc = dec(v[:, prov["cell_index"]]) @ u
    return R._auroc(sc, y), prov


def test_planting_on_positives_only_separates(base):
    auc, prov = _proj(base, "S1", "pos")
    assert auc >= 0.99
    assert prov["samples"]["S1"]["touched_windows"] == prov["samples"]["S1"]["n_positive"] == N // 4


def test_planting_on_all_windows_does_not_pass_the_bar(base):
    auc, prov = _proj(base, "S1", "all")
    assert auc < 0.99 and abs(auc - 0.5) < 0.2
    assert prov["samples"]["S1"]["touched_windows"] == N


def test_untouched_arrays_are_hardlinks_and_v40_differs_only_in_the_cell(base):
    _, prov = _proj(base, "S1", "pos")
    src, dst = (os.path.join(base, w, "S1_416") for w in ("assembled", os.path.join("c2", "assembled")))
    for k in P.TRAIN_FILES:
        a, b = os.path.join(src, k + ".npy"), os.path.join(dst, k + ".npy")
        assert os.path.samefile(a, b) and open(a, "rb").read() == open(b, "rb").read(), k
    a, b = np.load(os.path.join(src, "v40.npy")), np.load(os.path.join(dst, "v40.npy"))
    y = np.load(os.path.join(src, "y.npy")).astype(bool)
    cell = prov["cell_index"]
    m = np.ones(a.shape, bool)
    m[y, cell] = False
    assert a.dtype == b.dtype and a.shape == b.shape and (a[m] == b[m]).all()
    assert (a[y, cell] != b[y, cell]).any()
    ev0, ev1 = np.load(os.path.join(base, "assembled", "eval_416.npz")), np.load(os.path.join(base, "c2", "assembled", "eval_416.npz"))
    for k in ev0.files:
        if k not in ("v40", "meta"):
            assert ev0[k].dtype == ev1[k].dtype and ev0[k].tobytes() == ev1[k].tobytes(), k
    assert not np.array_equal(ev0["v40"], ev1["v40"])


def test_zero_amplitude_is_the_identity(base):
    _proj(base, "S1", "pos", amp=0.0)
    a = open(os.path.join(base, "assembled", "S1_416", "v40.npy"), "rb").read()
    b = open(os.path.join(base, "c2", "assembled", "S1_416", "v40.npy"), "rb").read()
    assert a[-N * 40 * D:] == b[-N * 40 * D:]


def test_provenance_records_norm_cell_direction_counts(base):
    _, prov = _proj(base, "S1", "pos")
    assert prov["cell_index"] == 25 and prov["seed"] == 5 and prov["median_token_norm"] > 0
    assert abs(prov["vector_l2"] - prov["median_token_norm"]) / prov["median_token_norm"] < 1e-5
    assert len(prov["direction_unit_sha256_float64"]) == 64


def test_link_failure_never_copies(tmp_path, monkeypatch):
    a = tmp_path / "a"
    a.write_bytes(b"x")
    monkeypatch.setattr(os, "link", lambda *_: (_ for _ in ()).throw(OSError("no link")))
    with pytest.raises(OSError):
        P.link(str(a), str(tmp_path / "b"))
    assert not (tmp_path / "b").exists()


def test_disk_floor_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "FLOOR_GB", 1e9)
    with pytest.raises(RuntimeError, match="disk"):
        P.need_free(str(tmp_path), 1)


def _ci():
    if not os.path.isdir(TREE + "/taniteval"):
        pytest.skip("taniteval tree absent")
    sys.path.insert(0, TREE)
    sys.path.insert(0, TREE + "/taniteval")
    from taniteval import ci
    return ci


def _scores(y, rng, planted):
    e = rng.normal(size=len(y)) + 1.0 * y
    sc = {}
    for f in R.FAMILIES:
        base = rng.normal(size=len(y)) + 1.0 * y
        sc[f"E|{f}"] = e
        sc[f"V40@256p+E|{f}"] = base
        sc[f"V40@416+E|{f}"] = (rng.normal(size=len(y)) + 6.0 * y) if planted else base
    return sc


def test_decide_planted_gives_adds_40_and_zero_amplitude_gives_flat():
    ci = _ci()
    rng = np.random.default_rng(3)
    eid = np.repeat(np.arange(40), 12)
    y = (rng.random(len(eid)) < 0.1).astype(np.uint8)
    per = {s: _scores(y, rng, True) for s in ("S1", "S2")}
    code, _, _, aur = P.decide_from_scores(per, y, eid, ci, n_boot=200)
    assert code == "RES416-ADDS-40" and min(v for a in aur.values() for v in a.values()) >= 0.99
    per0 = {s: _scores(y, rng, False) for s in ("S1", "S2")}          # amp 0: V40@416 scores are the 256p scores
    code0, _, _, _ = P.decide_from_scores(per0, y, eid, ci, n_boot=200)
    assert code0 == "RES416-FLAT"


def test_planted_on_all_windows_scores_do_not_give_adds_40():
    """A plant on every window carries no label signal: the V40@416 scores are noise relative to y, so the read must not return ADDS-40."""
    ci = _ci()
    rng = np.random.default_rng(4)
    eid = np.repeat(np.arange(40), 12)
    y = (rng.random(len(eid)) < 0.1).astype(np.uint8)
    per = {}
    for s in ("S1", "S2"):
        sc = _scores(y, rng, False)
        for f in R.FAMILIES:
            sc[f"V40@416+E|{f}"] = rng.normal(size=len(y))
        per[s] = sc
    code, _, _, aur = P.decide_from_scores(per, y, eid, ci, n_boot=200)
    assert code != "RES416-ADDS-40" and min(v for a in aur.values() for v in a.values()) < 0.99
