"""test_gapdecomp_m.py -- the pure layer of gapdecomp_m.py on planted data (CPU, a few seconds)."""
import importlib.util
import os
import shutil
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("gapdecomp_m", os.path.join(HERE, "gapdecomp_m.py"))
GM = importlib.util.module_from_spec(spec)
sys.modules["gapdecomp_m"] = GM
spec.loader.exec_module(GM)
GD = GM.GD


def synth(n_ep=40, per=20, seed=0):
    rng = np.random.default_rng(seed)
    eid = np.repeat(np.arange(n_ep), per)
    n = len(eid)
    sub = rng.integers(0, 4, size=n)
    return rng, eid, n, sub


def test_boundaries():
    assert list(GM.shape_index([0, 9.99, 10.0, 45.0, 45.01, -50.0])) == [0, 0, 1, 1, 2, 2]
    assert list(GM.speed_index([0.5, 2.99, 3.0, 7.99, 8.0, 14.99, 15.0, 30.0])) == [0, 0, 1, 1, 2, 2, 3, 3]
    with pytest.raises(ValueError):
        GM.speed_index([0.3])


def test_heading_change_known_answer():
    t = np.arange(1, 31) * 0.2
    v = 6.0
    straight = np.stack([v * t, 0 * t], -1)[None]
    th = np.deg2rad(90.0) * (t / t[-1])                                   # a path whose heading turns 90 degrees over the horizon
    arc = np.cumsum(np.stack([np.cos(th), np.sin(th)], -1) * v * 0.2, axis=0)[None]
    d, npv = GM.heading_change_deg(np.concatenate([straight, arc, -arc[:, :, :] * np.array([1, -1])]))
    assert abs(d[0]) < 1e-9 and npv[0] == 29
    assert 80.0 < d[1] < 91.0                                              # about 90 minus the first step's own heading
    assert -91.0 < d[2] < -80.0                                            # the mirrored path turns the other way


def test_partitions_cover_once_and_pool():
    rng, eid, n, sub = synth(40, 20)
    shape = rng.integers(0, 3, size=n)
    speed = sub
    mm = np.ones(n, bool)
    shape = np.where(speed == 0, 0, shape)                                  # shape 0 holds all of speed 0
    # starve cell (2, 3): 6 windows only
    keep = ~((shape == 2) & (speed == 3))
    keep[np.where((shape == 2) & (speed == 3))[0][:6]] = True
    mm = keep
    parts = GM.build_partitions(shape, speed, eid, mm)
    for pn, p in parts.items():
        lab = p["labels"]
        assert (lab[mm] >= 0).all() and (lab[~mm] == -1).all(), pn
        assert sum(int((lab == i).sum()) for i in range(len(p["names"]))) == int(mm.sum()), pn
    names = parts["cross"]["names"]
    assert "cross=TURN/3" not in names and "cross=TURN/>15" not in names             # the starved cell is not named on its own
    assert any(nm.endswith("/other") for nm in names)
    for nm, nmb, i in zip(names, parts["cross"]["nameable"], range(len(names))):
        if nmb:
            m = parts["cross"]["labels"] == i
            assert m.sum() >= GM.FLOOR_N and len(set(eid[m])) >= GM.FLOOR_EP


def test_decompose_identity_xx_zero_and_matches_landed_class():
    rng, eid, n, sub = synth(30, 25)
    a, b = rng.normal(1.0, 1.0, n), rng.normal(0.5, 1.0, n)
    a[rng.random(n) < 0.05] = np.nan
    mm = rng.random(n) < 0.8
    lab = np.where(mm, sub, -1)
    names = ["s0", "s1", "s2", "s3"]
    r, dr = GM.decompose_groups(a, b, eid, lab, names)
    assert r["_class_sum_identity_abs_dev"] < 1e-12
    assert r["_share_sum_point_abs_dev"] < 1e-12 and r["_share_sum_max_abs_dev_over_draws"] < 1e-9
    # the same windows as ONE class of the landed 4-class decompose: identical Delta_M, interval and contribution
    cls = np.where(mm, 3, 0).astype(np.int8)
    L = GD.decompose(a, b, eid, cls, "masked")
    lm = L["classes"]["M"]
    for mine, theirs in ((r["M"]["delta"], lm["delta"]), (r["M"]["lo"], lm["delta_lo"]), (r["M"]["hi"], lm["delta_hi"]), (r["M"]["contribution"], lm["contribution"]),
                         (r["M"]["contribution_lo"], lm["contribution_lo"]), (r["M"]["contribution_hi"], lm["contribution_hi"])):
        assert abs(mine - theirs) < 1e-12
    z, _ = GM.decompose_groups(a, a, eid, lab, names)
    assert z["M"]["delta"] == 0.0 and z["M"]["lo"] == 0.0 and z["M"]["hi"] == 0.0 and all(g["delta"] in (0.0, None) for g in z["groups"].values())


def _row(effect_by_sub, seed, n_ep=60, per=30):
    rng = np.random.default_rng(seed)
    eid = np.repeat(np.arange(n_ep), per)
    n = len(eid)
    sub = rng.choice(4, size=n, p=[0.40, 0.25, 0.20, 0.15])
    b = rng.normal(0.0, 1.0, n)
    a = b + np.asarray(effect_by_sub)[sub] + rng.normal(0.0, 0.5, n)
    lab = sub
    names = ["A", "B", "C", "D"]
    r, dr = GM.decompose_groups(a, b, eid, lab, names)
    return {"M": r["M"], "groups": {f"p|{k}": v for k, v in r["groups"].items()}}, {f"p|{k}": v for k, v in dr.items()}


def test_planted_effect_is_localised_and_uniform_effect_is_diffuse():
    R = {s: _row([0.0, 1.0, 0.0, 0.0], 10 + s)[0] for s in (0, 1)}
    D = {s: _row([0.0, 1.0, 0.0, 0.0], 10 + s)[1] for s in (0, 1)}
    v = GM.row_verdict(R, D)
    assert v["verdict"] == "LOCALISED" and v["named"] == "p|B", v                  # B holds 25 % of the windows and all of the effect
    Ru = {s: _row([1.0, 1.0, 1.0, 1.0], 20 + s)[0] for s in (0, 1)}
    vu = GM.row_verdict(Ru)
    assert vu["verdict"] == "DIFFUSE" and vu["named"] is None, vu
    Rz = {s: _row([0.0, 0.0, 0.0, 0.0], 30 + s)[0] for s in (0, 1)}
    vz = GM.row_verdict(Rz)
    assert vz["verdict"].startswith("NOT DEFINED"), vz                            # a difference that may be zero has no share


def test_a_large_subclass_cannot_be_named_and_a_thin_one_cannot_either():
    R = {s: _row([1.0, 0.0, 0.0, 0.0], 40 + s)[0] for s in (0, 1)}                  # the effect sits in A, 40 % of the windows (> 30 %)
    assert GM.row_verdict(R)["verdict"] == "DIFFUSE"
    g = dict(R[0]["groups"]["p|B"])
    g.update(n=GM.FLOOR_N - 1)
    g["support_ok"] = bool(g["n"] >= GM.FLOOR_N and g["n_episodes"] >= GM.FLOOR_EP)
    assert not GM.eligible(g)
    g2 = dict(R[0]["groups"]["p|B"], nameable=False)
    assert not GM.eligible(g2)


def test_zeroed_rows_empty_and_full_masks():
    rng, eid, n, sub = synth(30, 20)
    labels = ("ADE6", "S-POS@30")
    z = {"eid": eid}
    for i in range(2):
        z[f"a_{i}"], z[f"b_{i}"] = rng.normal(1, 1, n), rng.normal(0, 1, n)
    e = GM.zeroed_rows(GD, z, np.zeros(n, bool), labels)
    f = GM.zeroed_rows(GD, z, np.ones(n, bool), labels)
    for l in labels:
        assert abs(e[l]["delta_zeroed"] - e[l]["delta_total"]) < 1e-12
        assert f[l]["delta_zeroed"] == 0.0 and f[l]["lo"] == 0.0 and f[l]["hi"] == 0.0


def test_card_change_is_refused(tmp_path):
    d = tmp_path / "frozen"
    d.mkdir()
    shutil.copy(os.path.join(HERE, "FROZEN_SHA.txt"), d / "FROZEN_SHA.txt")
    with open(os.path.join(HERE, "PREREG_FROZEN.md"), "rb") as f:
        body = f.read()
    (d / "PREREG_FROZEN.md").write_bytes(body + b"\nan edit after data\n")
    assert GM.main(["--signed", str(tmp_path), "--out", str(tmp_path / "x.json"), "--frozen-dir", str(d)]) == 2
