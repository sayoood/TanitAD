"""test_q5.py -- AMENDMENT 7 code tests (CPU, synthetic): C1, C2, C7, C8 (code path), C10, the S0-P dir writer, the encoding= guard.
Run: C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_q5.py
"""
import importlib.util
import json
import os
import re
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q5_now_encode as Q  # noqa: E402
import q5_read as R  # noqa: E402

REPO = os.environ.get("Q5_REPO", "E:/Projects/TanitAD")
PEP = REPO + "/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini/code/pull_encode_pooled.py"
SHIP = REPO + "/stack/scripts/dinov3_fp8_encode_ship.py"
S0P = os.environ.get("Q5_S0P", "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/stopperc/s0p")
TREE = os.environ.get("S0P_TREE", "F:/TanitAD-scratch/stopbal_read/tree")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def azimuth_ok(spans, deg):
    return bool(np.allclose(spans, deg, atol=1e-9, rtol=0))


# ------------------------------------------------------------------------------------------------------------ C7
@pytest.mark.skipif(not os.path.exists(PEP), reason="pull_encode_pooled.py absent")
def test_c7_integer_cells_equal_pool_field():
    import torch
    pep = _load(PEP, "pep_for_q5")
    f = torch.randn(3, 640, 1024, generator=torch.Generator().manual_seed(0))
    for gh, gw in ((4, 10), (8, 20)):
        ref = pep.pool_field(f, gh, gw)
        got = Q.pool_pa(f, "256", gh, gw)
        assert got.shape == ref.shape
        assert float((got - ref).abs().max()) <= 1e-6


def test_c7_constant_field_exact_both_resolutions():
    import torch
    for res in ("256", "416"):
        n = Q.RES[res]["tokens"]
        for c in (0.3125, -2.5, 7.0):
            f = torch.full((2, n, 8), c)
            for gh, gw in ((4, 10), (8, 20)):
                assert torch.equal(Q.pool_pa(f, res, gh, gw), torch.full((2, gh * gw, 8), c))
            assert torch.equal(Q.pool_mean(f, res), torch.full((2, 8), c))


def test_c7_row_ramp_analytic_at_416():
    """A field whose value is the patch row index + 0.5 (the row centre): the 4x10 cell mean is the centre of the cell's row span, 0.2 + 6.4 (r + 0.5), to float tolerance."""
    import torch
    g = Q.RES["416"]
    f = (torch.arange(g["gh"], dtype=torch.float32) + 0.5).reshape(1, g["gh"], 1, 1).expand(1, g["gh"], g["gw"], 4).reshape(1, -1, 4).contiguous()
    p = Q.pool_pa(f, "416", 4, 10).reshape(4, 10, 4)
    want = []
    for r in range(4):                                         # independent expression: integral of the step field over the cell span / the span
        lo, hi = 0.2 + 6.4 * r, 0.2 + 6.4 * (r + 1)
        xs = np.linspace(lo, hi, 2_000_001)[:-1] + (hi - lo) / 4_000_000
        want.append(float(np.mean(np.floor(xs) + 0.5)))
    want = np.asarray(want)
    assert np.allclose(p[:, :, 0].numpy(), want[:, None], atol=1e-4)


def test_c7_azimuth_12_degrees_both_resolutions():
    for res in ("256", "416"):
        assert azimuth_ok(Q.cell_azimuth_spans(res, 10), 12.0)
        assert azimuth_ok(Q.cell_azimuth_spans(res, 20), 6.0)
    assert Q.RES["416"]["deg_per_patch"] == 1.875 and Q.RES["256"]["deg_per_patch"] == 3.0


def test_c7_pinhole_cell_mutation_fails_the_12_degree_check():
    for res in ("256", "416"):
        spans = Q.cell_azimuth_spans(res, 10, projection="pinhole")
        assert not azimuth_ok(spans, 12.0), "the check cannot see a pinhole cell map: it is blind"


# ------------------------------------------------------------------------------------------------------------ C8 (code path; the real model run is the pod smoke)
class _FakeModel:
    """Deterministic stand-in for DINOv3: 16-px patch means -> a fixed linear map to 1024-d, plus 5 special tokens.  Row-independent, so batch composition does not matter."""

    def __init__(self):
        import torch
        g = torch.Generator().manual_seed(1)
        self.w = torch.randn(3, 1024, generator=g)

    def __call__(self, pixel_values):
        import torch
        import torch.nn.functional as F
        p = F.avg_pool2d(pixel_values.float(), 16)                               # [B,3,h,w]
        tok = p.flatten(2).transpose(1, 2) @ self.w                              # [B,N,1024]
        spec = torch.zeros(tok.shape[0], 5, 1024)
        return type("O", (), {"last_hidden_state": torch.cat([spec, tok], 1)})()


def _v2ep(h, w, n, seed=0):
    import torch
    import torchvision
    rng = np.random.default_rng(seed)
    pngs = [torchvision.io.encode_png(torch.from_numpy(rng.integers(0, 256, (3, h, w), dtype=np.uint8)), compression_level=1) for _ in range(n)]
    return {"codec": "png", "jpeg_buf": torch.cat(pngs), "jpeg_len": torch.tensor([int(p.numel()) for p in pngs])}


def _stats():
    import torch
    return torch.zeros(1, 3, 1, 1) + 0.45, torch.zeros(1, 3, 1, 1) + 0.225


@pytest.mark.skipif(not os.path.exists(SHIP), reason="dinov3_fp8_encode_ship.py absent")
def test_c8_sibling_bitwise_equals_encode_episode_at_256():
    import torch
    ship = _load(SHIP, "ship_for_q5")
    o = _v2ep(256, 640, 38)
    mean, std = _stats()
    m = _FakeModel()
    a = ship.encode_episode(o, m, mean, std, 5, "cpu")
    b = Q.encode_episode_sib(o, m, mean, std, 5, "cpu", 640)
    assert a.shape == b.shape == (19, 640, 1024) and torch.equal(a, b)
    # sparse NOW-frame batching returns exactly the requested rows
    sp = Q.encode_frames(Q.decode_frames(o, [2 * 3, 2 * 11, 2 * 12]), m, mean, std, 5, "cpu", 640)
    assert torch.equal(sp, a[[3, 11, 12]])
    # a regression arm: the banked function refuses 416, the sibling accepts 1,664 tokens and refuses a wrong count
    o4 = _v2ep(416, 1024, 4)
    with pytest.raises(AssertionError):
        ship.encode_episode(o4, m, mean, std, 5, "cpu")
    assert Q.encode_episode_sib(o4, m, mean, std, 5, "cpu", 1664).shape == (2, 1664, 1024)
    with pytest.raises(AssertionError):
        Q.encode_episode_sib(o4, m, mean, std, 5, "cpu", 640)


# ------------------------------------------------------------------------------------------------------------ C1 / C2 (code path through the paired bootstrap) and C10
def _ci():
    if not os.path.isdir(TREE + "/taniteval"):
        pytest.skip("taniteval tree absent")
    sys.path.insert(0, TREE)
    sys.path.insert(0, TREE + "/taniteval")
    from taniteval import ci
    return ci


def _synthetic(n_ep=40, per=12, seed=3):
    rng = np.random.default_rng(seed)
    eid = np.repeat(np.arange(n_ep), per)
    y = (rng.random(len(eid)) < 0.1).astype(np.uint8)
    return y, eid, rng


def _scores(y, rng, boost416=0.0):
    e = rng.normal(size=len(y)) + 1.0 * y
    sc = {}
    for f in R.FAMILIES:
        sc[f"E|{f}"] = e
        for g in R.GRIDS:
            base = rng.normal(size=len(y)) + 1.0 * y
            sc[f"V{g}@256p+E|{f}"] = base
            sc[f"V{g}@416+E|{f}"] = base if boost416 == 0.0 else rng.normal(size=len(y)) + boost416 * y
    return sc


def test_c1_identity_gives_exact_zero_and_not_adds():
    ci = _ci()
    y, eid, rng = _synthetic()
    sc = _scores(y, rng, 0.0)                                  # "V416" IS the 256p array
    _, cells = R.bootstrap_sample(sc, y, eid, ci, n_boot=200)
    for g in R.GRIDS:
        for f in R.FAMILIES:
            r = cells[g][f]["R"]
            assert r["point"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0
    inp = {"S1": cells, "S2": cells}
    assert R.decide(inp)[0] == "RES416-FLAT"


def test_c2_planted_positive_gives_adds_40():
    ci = _ci()
    y, eid, rng = _synthetic()
    sc = _scores(y, rng, 6.0)                                  # a planted direction on positives only
    _, cells = R.bootstrap_sample(sc, y, eid, ci, n_boot=200)
    inp = {"S1": cells, "S2": cells}
    assert R.decide(inp)[0] == "RES416-ADDS-40"


def _cell(point_r, lo_r, point_a, lo_a):
    return {"R": {"point": point_r, "lo": lo_r}, "A": {"point": point_a, "lo": lo_a}}


def _blank():
    return {g: {f: _cell(0.0, -0.01, 0.0, -0.01) for f in R.FAMILIES} for g in R.GRIDS}


def _sample(g=None, f=None, yes=False):
    s = _blank()
    if yes:
        s[g][f] = _cell(0.03, 0.005, 0.05, 0.01)
    return s


def test_c10_decide_basic_outcomes():
    assert R.decide({"S1": _sample(40, "linear", True), "S2": _sample(40, "linear", True)})[0] == "RES416-ADDS-40"
    assert R.decide({"S1": _sample(160, "mlp", True), "S2": _sample(160, "mlp", True)})[0] == "RES416-ADDS-160"
    assert R.decide({"S1": _blank(), "S2": _blank()})[0] == "RES416-FLAT"
    # the R bar is >= +0.020 AND lo > 0
    c = _sample(); c[40]["linear"] = _cell(0.019, 0.005, 0.05, 0.01)
    assert R.decide({"S1": c, "S2": c})[0] == "RES416-FLAT"
    c = _sample(); c[40]["linear"] = _cell(0.03, 0.0, 0.05, 0.01)
    assert R.decide({"S1": c, "S2": c})[0] == "RES416-FLAT"


def test_c10_unpaired_reading_goes_red():
    """A constructed case where the PAIRED difference does not separate but the unpaired reading (probe CI lower bound vs comparator point) says yes.
    The production reader must say FLAT; swapping in the unpaired reader must change the answer, so a regression to it is caught."""
    cell = {"R": {"point": 0.004, "lo": -0.02, "probe_lo": 0.86, "comparator_point": 0.80}, "A": {"point": 0.05, "lo": 0.01}}
    inp = {"S1": {40: {"linear": cell, "mlp": cell}, 160: {"linear": cell, "mlp": cell}}}
    inp["S2"] = inp["S1"]
    assert R.decide(inp)[0] == "RES416-FLAT"
    assert R.decide(inp, reader_R=R.pred_R_unpaired)[0] == "RES416-ADDS-40"       # the wrong reader flips it: the test goes red if production ever uses it


def test_c10_dropping_A_flips_a_constructed_case():
    c = _sample(); c[40]["linear"] = _cell(0.03, 0.005, 0.01, -0.004)              # resolution adds, but not above E
    inp = {"S1": c, "S2": c}
    assert R.decide(inp)[0] == "RES416-FLAT"
    assert R.decide(inp, use_A=False)[0] == "RES416-ADDS-40"


def test_c10_one_sample_only_is_samples_disagree():
    inp = {"S1": _sample(40, "linear", True), "S2": _blank()}
    code, d = R.decide(inp)
    assert code == "SAMPLES-DISAGREE" and d["holds_at"] == ["S1"]
    # with S3: 2 of 3 is ADDS, 1 of 3 is FLAT (there is no fourth sample)
    assert R.decide({**inp, "S3": _sample(40, "linear", True)})[0] == "RES416-ADDS-40"
    assert R.decide({**inp, "S3": _blank()})[0] == "RES416-FLAT"


def test_c10_mixed_families_do_not_count():
    c = _sample(); c[40]["mlp"] = _cell(0.03, 0.005, 0.01, -0.004)                 # R from the MLP ...
    c[40]["linear"] = _cell(0.001, -0.01, 0.05, 0.01)                              # ... A from the linear model
    assert R.decide({"S1": c, "S2": c})[0] == "RES416-FLAT"


def test_c10_family_must_agree_across_samples():
    s1, s2 = _sample(40, "linear", True), _sample(40, "mlp", True)
    assert R.decide({"S1": s1, "S2": s2})[0] != "RES416-ADDS-40"


def test_controls_refuse_on_any_miss():
    good = {"c0_differing_bytes": 0, "c1_dauc": 0.0, "c1_ci": [0.0, 0.0], "c1_outcome": "RES416-FLAT", "c2_auroc": 0.995, "c2_outcome": "RES416-ADDS-40", "c3a_leak_auroc": 1.0,
            "c3b_limit_max_abs_diff": 0.0, "c3c_E_windows_changed": 5, "c3d_shared_clips": 0, "c3e_E_y_clip_ws_eid_bitwise": True, "c4_E_bitwise_equal_C0": True,
            "c5_real_auroc": 0.86, "c5_null_max": 0.55, "c6a_max_abs_diff": 0, "c6b_mae_255": 0.6, "c6c_bitwise": True, "c7_ok": True, "c8_bitwise": True,
            "c9_counts": {"eval_n": 2365, "eval_pos": 131, "eval_episodes": 139, "train_n": 16000, "train_pos": 4000}}
    assert R.evaluate_controls(good)[1]
    for k in good:
        bad = dict(good)
        bad.pop(k)
        assert not R.evaluate_controls(bad)[1], f"a missing {k} did not refuse the read"
    assert not R.evaluate_controls({**good, "c6b_mae_255": 1.2})[1]


# ------------------------------------------------------------------------------------------------------------ the S0-P dir writer
def _fake_src(tmp, n=7, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(tmp, exist_ok=True)
    np.save(f"{tmp}/E.npy", rng.normal(size=(n, 18)).astype(np.float32))
    np.save(f"{tmp}/y.npy", (rng.random(n) < 0.3).astype(np.uint8))
    np.save(f"{tmp}/clip.npy", np.asarray([f"c{i % 3}" for i in range(n)]).astype("U40"))
    np.save(f"{tmp}/ws.npy", rng.integers(0, 90, n).astype(np.int64))
    np.save(f"{tmp}/eid.npy", rng.integers(0, 10**9, n).astype(np.int64))
    np.save(f"{tmp}/v40.npy", np.zeros((n, 40, 1024), np.uint8))
    np.save(f"{tmp}/v160.npy", np.zeros((n, 160, 1024), np.uint8))
    np.save(f"{tmp}/mean.npy", np.zeros((n, 1024), np.float16))
    json.dump({"sample_sha256": "x", "n_sample": n}, open(f"{tmp}/meta.json", "w"))
    json.dump({"clip": [], "ws": [], "y": [], "sha256": "x"}, open(f"{tmp}/sample.json", "w"))
    return n


def test_writer_copies_bitwise_and_replaces_only_vision(tmp_path):
    src, out = str(tmp_path / "src"), str(tmp_path / "out")
    n = _fake_src(src)
    rng = np.random.default_rng(9)
    v40 = rng.integers(0, 255, (n, 40, 1024), dtype=np.uint8)
    v160 = rng.integers(0, 255, (n, 160, 1024), dtype=np.uint8)
    mean = rng.normal(size=(n, 1024)).astype(np.float16)
    Q.write_s0p_dir(src, out, v40, v160, mean, {"res": "416"})
    for k in Q.COPIED:
        assert open(f"{src}/{k}.npy", "rb").read() == open(f"{out}/{k}.npy", "rb").read(), k
    assert np.array_equal(np.load(f"{out}/v40.npy"), v40) and np.array_equal(np.load(f"{out}/mean.npy"), mean)
    assert json.load(open(f"{out}/meta.json"))["q5"]["res"] == "416"
    lines = open(f"{out}/MANIFEST.sha256").read().split("\n")
    assert any(l.endswith("  E.npy") for l in lines) and any(l.endswith("  v160.npy") for l in lines)
    with pytest.raises(AssertionError):                               # a wrong row count is refused
        Q.write_s0p_dir(src, str(tmp_path / "bad"), v40[:-1], v160[:-1], mean[:-1])
    with pytest.raises(AssertionError):                               # a wrong dtype is refused
        Q.write_s0p_dir(src, str(tmp_path / "bad2"), v40.astype(np.float32), v160, mean)


def test_writer_fields_equal_s0p_lib():
    p = S0P + "/s0p_lib.py"
    if not os.path.exists(p):
        pytest.skip("s0p_lib absent")
    sys.path.insert(0, S0P)
    import s0p_lib
    assert tuple(s0p_lib.FIELDS) == tuple(Q.FIELDS)


def test_npz_writer_roundtrip(tmp_path):
    n = 5
    src = str(tmp_path / "e.npz")
    meta = json.dumps({"a": 1})
    np.savez(src, ci=np.arange(n), E=np.ones((n, 18), np.float32), v40=np.zeros((n, 40, 1024), np.uint8), v160=np.zeros((n, 160, 1024), np.uint8),
             mean=np.zeros((n, 1024), np.float16), y=np.zeros(n, np.uint8), clip=np.asarray(["a"] * n).astype("U40"), ws=np.arange(n), eid=np.arange(n), meta=np.array(meta))
    out = str(tmp_path / "o.npz")
    Q.write_s0p_npz(src, out, np.ones((n, 40, 1024), np.uint8), np.ones((n, 160, 1024), np.uint8), np.ones((n, 1024), np.float16))
    a, b = np.load(src), np.load(out)
    for k in ("E", "y", "clip", "ws", "eid", "ci"):
        assert np.array_equal(a[k], b[k])
    assert int(b["v40"].min()) == 1


# ------------------------------------------------------------------------------------------------------------ hygiene
def test_every_subprocess_text_true_has_encoding():
    for fn in os.listdir(HERE):
        if fn.startswith("q5_") and fn.endswith(".py"):
            src = open(os.path.join(HERE, fn), encoding="utf-8").read()
            for m in re.finditer(r"subprocess\.run\((?:[^()]|\([^()]*\))*\)", src, re.S):
                call = m.group(0)
                if "text=True" in call:
                    assert "encoding=" in call, f"{fn}: subprocess.run(text=True) without encoding=: {call[:80]}"
