"""Tests of soup_build.py on synthetic state dicts and synthetic checkpoints (no real weights, no rollout, no eval)."""
import collections
import json
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("SOUP_BUILD_DIR", os.path.dirname(HERE)))
import soup_build as SB  # noqa: E402


def sd_like(seed, scale=1.0, base=None, delta=0.0):
    g = torch.Generator().manual_seed(seed)
    out = collections.OrderedDict()
    for k, shp in (("a.w", (7, 5)), ("a.b", (5,)), ("b.w", (3, 11)), ("c", (13,))):
        t = torch.randn(shp, generator=g, dtype=torch.float32) * scale
        out[k] = (base[k] + delta * t) if base is not None else t
    return out


def ckpt(sd, **extra):
    c = {"model": sd, "opt": {"state": {0: 1}}, "cfg": {"kin_mode": "yaw", "w_acc": 49.6969, "hist": 4}, "grid": "pool4x10", "arm": "M2b",
         "p10_path": "p10.pt", "cfg_tuple_fields": {"wm_strides": [2, 2]}, "cons_mu": np.arange(4, dtype=np.float32), "cons_sd": np.ones(4, np.float32),
         "step": 3000, "seed": 0, "argv_sha": "x"}
    c.update(extra)
    return c


# ---- check_compatible / average_state
def test_average_is_the_float64_mean_rounded_once():
    a, b = sd_like(1), sd_like(2)
    s = SB.average_state([a, b])
    for k in a:
        ref = ((a[k].double() + b[k].double()) / 2).float()
        assert torch.equal(s[k], ref) and s[k].dtype == torch.float32


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_k_zero_bit_for_bit_including_awkward_values(n):
    sd = sd_like(3, scale=1e3)
    sd["d"] = torch.tensor([1e-40, -1e-42, 3.4e38, -3.3e38, 1 / 3, 0.1, -0.0, 7.000001], dtype=torch.float32)
    s = SB.average_state([sd] * n)
    assert all(torch.equal(s[k], sd[k]) for k in sd)
    assert SB.k_zero({"model": sd}, ns=(n,))["PASS"]


def test_k_zero_detects_a_drift():
    sd = sd_like(4)
    r = SB.k_zero({"model": sd})
    assert r["PASS"] and set(r["per_n"]) == {"2", "3", "4"}


def test_float32_accumulation_is_not_exact_float64_is():
    # the reason for float64: three equal float32 values summed in float32 do not always divide back exactly
    x = torch.randn(20000, generator=torch.Generator().manual_seed(7), dtype=torch.float32) * 1e3
    assert not torch.equal((x + x + x) / 3, x)
    assert torch.equal(((x.double() + x.double() + x.double()) / 3).float(), x)
    s = SB.average_state([collections.OrderedDict(v=x)] * 3)
    assert torch.equal(s["v"], x)


def test_refuses_key_order_shape_dtype():
    a = sd_like(1)
    b = collections.OrderedDict(reversed(list(sd_like(2).items())))
    with pytest.raises(SB.SoupRefusal, match="keys"):
        SB.check_compatible([a, b])
    c = sd_like(2)
    c["c"] = torch.zeros(14)
    with pytest.raises(SB.SoupRefusal, match="shape"):
        SB.check_compatible([a, c])
    d = sd_like(2)
    d["c"] = d["c"].double()
    with pytest.raises(SB.SoupRefusal, match="dtype"):
        SB.check_compatible([a, d])
    e = sd_like(2)
    del e["c"]
    with pytest.raises(SB.SoupRefusal, match="keys"):
        SB.check_compatible([a, e])


def test_nonfloat_equal_is_kept_unequal_is_refused():
    a, b = sd_like(1), sd_like(2)
    a["n"] = torch.tensor([3, 4], dtype=torch.int64)
    b["n"] = torch.tensor([3, 4], dtype=torch.int64)
    s = SB.average_state([a, b])
    assert torch.equal(s["n"], a["n"]) and s["n"].dtype == torch.int64
    b["n"] = torch.tensor([3, 5], dtype=torch.int64)
    with pytest.raises(SB.SoupRefusal, match="non-float"):
        SB.average_state([a, b])


def test_needs_two_members():
    with pytest.raises(SB.SoupRefusal):
        SB.check_compatible([sd_like(1)])


# ---- meta
def test_cfg_allowlist_and_must_equal():
    a, b = ckpt(sd_like(1)), ckpt(sd_like(2))
    b["cfg"] = dict(b["cfg"], launch_intent_w=1.58926)
    SB.check_meta([a, b])                                   # training-only key may differ
    c = ckpt(sd_like(2))
    c["cfg"] = dict(c["cfg"], hist=5)
    with pytest.raises(SB.SoupRefusal, match="cfg"):
        SB.check_meta([a, c])
    d = ckpt(sd_like(2))
    d["cfg"] = {k: v for k, v in d["cfg"].items() if k != "hist"}
    with pytest.raises(SB.SoupRefusal, match="cfg"):
        SB.check_meta([a, d])
    for key, val in (("grid", "pool8x20"), ("arm", "M1"), ("p10_path", "q.pt"), ("cfg_tuple_fields", {"wm_strides": [2, 1]}),
                     ("cons_mu", np.arange(4, dtype=np.float32) + 1e-6), ("cons_sd", np.ones(4, np.float64))):
        e = ckpt(sd_like(2), **{key: val})
        with pytest.raises(SB.SoupRefusal, match=key):
            SB.check_meta([a, e])
    f = ckpt(sd_like(2), seed=10, step=2999, argv_sha="y")  # member-specific entries may differ
    SB.check_meta([a, f])


def test_make_soup_ckpt_donor_entries_opt_none_and_member_meta():
    a, b = ckpt(sd_like(1), seed=0), ckpt(sd_like(2), seed=10)
    s = SB.average_state([a["model"], b["model"]])
    out = SB.make_soup_ckpt([a, b], 0, s, {"name": "S"})
    assert out["opt"] is None and out["model"] is s and out["seed"] == 0 and out["cfg"] == a["cfg"]
    assert [m["seed"] for m in out["soup"]["members_meta"]] == [0, 10] and out["soup"]["donor_index"] == 0
    assert a["opt"] is not None                                 # the donor is not mutated
    out1 = SB.make_soup_ckpt([a, b], 1, s, {"name": "S"})
    assert out1["seed"] == 10


# ---- K-MID
def test_k_mid_passes_on_the_mean_and_fails_on_a_perturbation():
    a, b = sd_like(1), sd_like(2)
    s = SB.average_state([a, b])
    r = SB.k_mid(s, [a, b])
    assert r["PASS"] and r["max_abs_vs_numpy_mean"] == 0.0 and r["elements_outside_member_range"] == 0
    t = collections.OrderedDict((k, v.clone()) for k, v in s.items())
    t["a.w"][0, 0] = torch.nextafter(t["a.w"][0, 0], torch.tensor(1e9))
    assert not SB.k_mid(t, [a, b])["PASS"]
    u = collections.OrderedDict((k, v.clone()) for k, v in a.items())   # a member is not the midpoint
    assert not SB.k_mid(u, [a, b])["PASS"]
    four = [sd_like(i) for i in range(4)]
    assert SB.k_mid(SB.average_state(four), four)["PASS"]
    assert not SB.k_mid(SB.average_state(four[:2]), four)["PASS"]


def test_k_mid_distance_identity_is_load_bearing():
    a, b = sd_like(1), sd_like(2)
    s = SB.average_state([a, b])
    r = SB.k_mid(s, [a, b])
    assert abs(r["d_soup_a"] - r["d_ab"] / 2) <= 1e-6 * r["d_ab"]


# ---- K-BASIN on a synthetic same-init family and a synthetic cross-init pair
def family():
    i0, i1 = sd_like(100), sd_like(200)
    f0 = {f"m{j}": sd_like(10 + j, base=i0, delta=0.05) for j in range(4)}
    f1 = {f"n{j}": sd_like(20 + j, base=i1, delta=0.05) for j in range(4)}
    return i0, i1, f0, f1


def test_basin_passes_same_init_soups():
    i0, i1, f0, f1 = family()
    r = SB.basin({"m0": f0["m0"], "m1": f0["m1"]}, i0, i1, f0)
    assert r["PASS"] and r["B1"] and r["B2"] and r["spread_ratio"] < 3
    r4 = SB.basin(f0, i0, i1, f0)
    assert r4["PASS"]


def test_basin_fails_cross_init_soup_and_each_clause_is_load_bearing():
    i0, i1, f0, f1 = family()
    mem = {"m0": f0["m0"], "n0": f1["n0"]}
    full = SB.basin(mem, i0, i1, f0)
    assert not full["PASS"] and not full["B1"] and not full["B2"]
    assert not SB.basin(mem, i0, i1, f0, clauses=("B2",))["PASS"]
    assert not SB.basin(mem, i0, i1, f0, clauses=("B1",))["PASS"]
    assert full["B1_rows"] == {"m0": True, "n0": False}


def test_basin_b2_alone_catches_a_far_member_that_b1_allows():
    # a member displaced far from the init but still nearer to it than to the other init: B1 passes, B2 must fail
    i0, i1, f0, f1 = family()
    far = sd_like(31, base=i0, delta=0.6)
    r = SB.basin({"m0": f0["m0"], "far": far}, i0, i1, f0)
    assert r["B1"] and not r["B2"] and not r["PASS"]
    assert SB.basin({"m0": f0["m0"], "far": far}, i0, i1, f0, clauses=("B1",))["PASS"]


def test_basin_reference_is_the_median_not_the_max():
    i0, i1, f0, f1 = family()
    refs = dict(f0)
    refs["m3"] = f1["n0"]                                  # one foreign reference member
    r = SB.basin({"m0": f0["m0"], "m1": f0["m1"]}, i0, i1, refs)
    rv = sorted(SB.dist(v, i0) for v in refs.values())
    assert abs(r["R_ft"] - (rv[1] + rv[2]) / 2) < 1e-9


def test_basin_factors_are_the_declared_ones():
    assert SB.B1_FACTOR == 2.0 and SB.B2_FACTOR == 3.0 and SB.CLAUSES == ("B1", "B2")


# ---- the file layer, end to end on synthetic checkpoints
def write_world(tmp_path, cross=False):
    i0, i1, f0, f1 = family()
    paths = {}

    def put(name, sd, **extra):
        p = tmp_path / f"{name}.pt"
        torch.save(ckpt(sd, **extra), p)
        paths[name] = (str(p).replace("\\", "/"), SB.sha256_file(p))

    put("I0", i0)
    put("I1", i1)
    for j, (k, v) in enumerate(f0.items()):
        put(k, v, seed=j)
    put("n0", f1["n0"])
    spec = {"inits": {"0": {"path": paths["I0"][0], "sha256": paths["I0"][1]}, "1": {"path": paths["I1"][0], "sha256": paths["I1"][1]}},
            "members": {k: {"path": p, "sha256": h} for k, (p, h) in paths.items() if not k.startswith("I")},
            "reference": {"0": ["m0", "m1", "m2", "m3"], "1": ["m0", "m1", "m2", "m3"]},
            "soups": {"S": {"seed": 0, "members": ["m0", "m1"], "donor": "m0", "run_dir": "run_S"},
                      "S4": {"seed": 0, "members": ["m0", "m1", "m2", "m3"], "donor": "m0", "run_dir": "run_S4"},
                      "X": {"seed": 0, "members": ["m0", "n0"], "donor": "m0", "run_dir": "run_X", "expect": "K-BASIN FAIL"},
                      "Xw": {"seed": 0, "members": ["m0", "n0"], "donor": "m0", "run_dir": "run_Xw"}}}
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps(spec))
    return spec, str(sp), paths


def test_build_writes_a_loadable_soup_and_its_record(tmp_path):
    spec, sp, paths = write_world(tmp_path)
    out = tmp_path / "out"
    rec = SB.build(spec, "S", str(out), log=lambda *a: None)
    ckp = out / "run_S" / "ckpt_last.pt"
    assert rec["output_sha256"] == SB.sha256_file(ckp)
    assert rec["member_sha256"] == {"m0": paths["m0"][1], "m1": paths["m1"][1]} and rec["donor"] == "m0"
    assert rec["builder_sha256"] == SB.builder_sha() and rec["K-BASIN"]["PASS"] and rec["K-MID"]["PASS"]
    ck = torch.load(ckp, map_location="cpu", weights_only=False)
    m0 = torch.load(paths["m0"][0], map_location="cpu", weights_only=False)
    m1 = torch.load(paths["m1"][0], map_location="cpu", weights_only=False)
    assert ck["opt"] is None and ck["cfg"] == m0["cfg"] and ck["seed"] == m0["seed"]
    for k in ck["model"]:
        assert torch.equal(ck["model"][k], ((m0["model"][k].double() + m1["model"][k].double()) / 2).float())
    on_disk = json.loads((out / "run_S" / "soup.json").read_text())
    assert on_disk["output_sha256"] == rec["output_sha256"]
    assert SB.verify(spec, "S", str(out))["PASS"]
    assert SB.main(["verify", "--spec", sp, "--name", "S", "--out-root", str(out)]) == 0


def test_build_refuses_overwrite_xinit_and_a_cross_init_soup(tmp_path):
    spec, sp, _ = write_world(tmp_path)
    out = tmp_path / "out"
    SB.build(spec, "S", str(out), log=lambda *a: None)
    with pytest.raises(SB.SoupRefusal, match="overwrite"):
        SB.build(spec, "S", str(out), log=lambda *a: None)
    with pytest.raises(SB.SoupRefusal, match="deliberate-regression"):
        SB.build(spec, "X", str(out), log=lambda *a: None)
    with pytest.raises(SB.SoupRefusal, match="K-BASIN FAILED"):
        SB.build(spec, "Xw", str(out), log=lambda *a: None)
    assert not (out / "run_Xw").exists()
    assert SB.main(["build", "--spec", sp, "--name", "Xw", "--out-root", str(out)]) == 2


def test_build_four_member_soup(tmp_path):
    spec, _, _ = write_world(tmp_path)
    rec = SB.build(spec, "S4", str(tmp_path / "out"), log=lambda *a: None)
    assert rec["weights"] == [0.25] * 4 and rec["K-MID"]["PASS"]


def test_pins_are_enforced(tmp_path):
    spec, _, paths = write_world(tmp_path)
    spec["members"]["m1"]["sha256"] = "0" * 64
    with pytest.raises(SB.SoupRefusal, match="pinned"):
        SB.build(spec, "S", str(tmp_path / "out"), log=lambda *a: None)


def test_verify_catches_tampering(tmp_path):
    spec, _, paths = write_world(tmp_path)
    out = tmp_path / "out"
    SB.build(spec, "S", str(out), log=lambda *a: None)
    ckp = out / "run_S" / "ckpt_last.pt"
    ck = torch.load(ckp, map_location="cpu", weights_only=False)
    ck["model"]["c"][0] += 1.0
    torch.save(ck, ckp)
    v = SB.verify(spec, "S", str(out))
    assert not v["PASS"] and any("output sha256" in p for p in v["problems"]) and any("K-MID" in p for p in v["problems"])
    rec = json.loads((out / "run_S" / "soup.json").read_text())
    rec["output_sha256"] = SB.sha256_file(ckp)
    (out / "run_S" / "soup.json").write_text(json.dumps(rec))
    v2 = SB.verify(spec, "S", str(out))
    assert not v2["PASS"] and any("K-MID" in p for p in v2["problems"])


def test_verify_catches_kept_opt_and_wrong_cfg(tmp_path):
    spec, _, _ = write_world(tmp_path)
    out = tmp_path / "out"
    SB.build(spec, "S", str(out), log=lambda *a: None)
    ckp = out / "run_S" / "ckpt_last.pt"
    rp = out / "run_S" / "soup.json"
    for mut in ("opt", "cfg", "grid"):
        ck = torch.load(ckp, map_location="cpu", weights_only=False)
        if mut == "opt":
            ck["opt"] = {"state": {}}
        elif mut == "cfg":
            ck["cfg"] = dict(ck["cfg"], hist=9)
        else:
            ck["grid"] = "pool8x20"
        p2 = tmp_path / f"t_{mut}"
        p2.mkdir()
        (p2 / "run_S").mkdir()
        torch.save(ck, p2 / "run_S" / "ckpt_last.pt")
        rec = json.loads(rp.read_text())
        rec["output_sha256"] = SB.sha256_file(p2 / "run_S" / "ckpt_last.pt")
        (p2 / "run_S" / "soup.json").write_text(json.dumps(rec))
        assert not SB.verify(spec, "S", str(p2))["PASS"], mut


def test_basin_cli_requires_the_regression_to_fail(tmp_path):
    spec, sp, _ = write_world(tmp_path)
    del spec["soups"]["Xw"]
    (tmp_path / "spec2.json").write_text(json.dumps(spec))
    outj = tmp_path / "basin.json"
    assert SB.main(["basin", "--spec", str(tmp_path / "spec2.json"), "--out", str(outj)]) == 0
    b = json.loads(outj.read_text())
    assert b["ALL_AS_REQUIRED"] and b["soups"]["X"]["regression_as_required"] and b["soups"]["S"]["full"]["PASS"]
    spec["soups"]["X"]["members"] = ["m0", "m2"]           # a same-init "regression" passes K-BASIN -> the CLI must report NOT AS REQUIRED
    (tmp_path / "spec3.json").write_text(json.dumps(spec))
    assert SB.main(["basin", "--spec", str(tmp_path / "spec3.json"), "--out", str(outj)]) == 2


def test_load_ckpt_hashes_before_and_after(tmp_path):
    p = tmp_path / "x.pt"
    torch.save(ckpt(sd_like(1)), p)
    ck, h = SB.load_ckpt(str(p), SB.sha256_file(p))
    assert h == SB.sha256_file(p)
    with pytest.raises(SB.SoupRefusal):
        SB.load_ckpt(str(p), "f" * 64)


# ---- Amendment A0: the +-inf bound buffers (wm.stack.prior_lo_k / prior_hi_k in every real member and init)
def with_inf(sd):
    out = collections.OrderedDict((k, v.clone()) for k, v in sd.items())
    out["wm.prior_lo"] = torch.full((4,), float("-inf"))
    out["wm.prior_hi"] = torch.tensor([float("inf"), float("inf"), 2.0, float("inf")])
    return out


def test_equal_infinite_buffers_are_copied_and_excluded_from_distances():
    a, b = with_inf(sd_like(1)), with_inf(sd_like(2))
    s = SB.average_state([a, b])
    assert torch.equal(s["wm.prior_lo"], a["wm.prior_lo"]) and torch.equal(s["wm.prior_hi"], a["wm.prior_hi"])
    assert np.isfinite(SB.dist(a, b)) and abs(SB.dist(a, b) - SB.dist(sd_like(1), sd_like(2))) < 1e-9
    assert SB.k_mid(s, [a, b])["PASS"]
    assert all(torch.equal(SB.average_state([a] * n)[k], a[k]) for n in (2, 3, 4) for k in a)


def test_unequal_infinite_buffers_are_refused():
    a, b = with_inf(sd_like(1)), with_inf(sd_like(2))
    b["wm.prior_hi"][2] = float("inf")                      # pattern differs
    with pytest.raises(SB.SoupRefusal, match="non-finite pattern"):
        SB.average_state([a, b])
    with pytest.raises(SB.SoupRefusal, match="non-finite pattern"):
        SB.dist(a, b)
    c = with_inf(sd_like(2))
    c["wm.prior_lo"][0] = float("inf")                      # same pattern, other sign
    with pytest.raises(SB.SoupRefusal, match="non-finite values"):
        SB.average_state([a, c])
    d = with_inf(sd_like(2))
    d["wm.prior_lo"][1] = float("nan")
    with pytest.raises(SB.SoupRefusal):
        SB.average_state([a, d])


def test_k_mid_checks_infinite_elements_explicitly():
    a, b = with_inf(sd_like(1)), with_inf(sd_like(2))
    s = SB.average_state([a, b])
    t = collections.OrderedDict((k, v.clone()) for k, v in s.items())
    t["wm.prior_lo"][0] = -1e30
    assert not SB.k_mid(t, [a, b])["PASS"]
    u = collections.OrderedDict((k, v.clone()) for k, v in s.items())
    u["wm.prior_hi"][0] = float("-inf")
    assert not SB.k_mid(u, [a, b])["PASS"]


def test_basin_with_infinite_buffers_is_finite_and_still_discriminates():
    i0, i1, f0, f1 = family()
    i0, i1 = with_inf(i0), with_inf(i1)
    f0 = {k: with_inf(v) for k, v in f0.items()}
    f1 = {k: with_inf(v) for k, v in f1.items()}
    ok = SB.basin({"m0": f0["m0"], "m1": f0["m1"]}, i0, i1, f0)
    bad = SB.basin({"m0": f0["m0"], "n0": f1["n0"]}, i0, i1, f0)
    assert ok["PASS"] and np.isfinite(ok["d_max"]) and np.isfinite(ok["R_ft"])
    assert not bad["PASS"] and not bad["B1"] and not bad["B2"] and np.isfinite(bad["d_max"])


def test_a_file_that_changes_while_loading_is_refused(tmp_path, monkeypatch):
    p = tmp_path / "x.pt"
    torch.save(ckpt(sd_like(1)), p)
    h = SB.sha256_file(p)
    real = torch.load

    def load_then_touch(path, *a, **k):
        out = real(path, *a, **k)
        with open(path, "ab") as f:
            f.write(b"\0")
        return out

    monkeypatch.setattr(SB.torch, "load", load_then_touch)
    with pytest.raises(SB.SoupRefusal, match="changed while"):
        SB.load_ckpt(str(p), h)


def test_k_zero_fails_when_the_average_drifts_or_changes_dtype(monkeypatch):
    sd = sd_like(5)
    real = SB.average_state

    def drift(sds):
        out = real(sds)
        out["c"] = out["c"].clone()
        out["c"][0] = torch.nextafter(out["c"][0], torch.tensor(1e9))
        return out

    monkeypatch.setattr(SB, "average_state", drift)
    assert not SB.k_zero({"model": sd})["PASS"]

    def widen(sds):
        return collections.OrderedDict((k, v.double()) for k, v in real(sds).items())

    monkeypatch.setattr(SB, "average_state", widen)
    assert not SB.k_zero({"model": sd})["PASS"]
