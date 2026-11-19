"""Tests of build_dose_pack.py: the pack equals an independent row-by-row computation of the label rule; refusals; the file and meta; the optional state-file layout check."""
import json

import numpy as np
import pytest

import build_dose_pack as BP  # noqa: E402
import dose_testlib as TL  # noqa: E402


def setup(tmp_path, n=8, shuffle=True):
    lay = TL.make_layout(n)
    join = TL.make_join(lay, shuffle=shuffle)
    lp, jp = tmp_path / "lay.npz", tmp_path / "join.npz"
    np.savez(lp, **lay)
    np.savez(jp, **join)
    return lay, join, TL.sh(lp), TL.sh(jp)


def lay_only(lay):
    return {k: lay[k] for k in ("clips", "clip_idx", "t", "kin")}


def test_pack_equals_the_row_by_row_rule():
    lay = TL.make_layout(8)
    join = TL.make_join(lay)
    P = BP.build_pack(lay_only(lay), join)
    gap, pres = TL.expected_pack(lay, join)
    assert np.array_equal(np.isnan(P["gap"]), np.isnan(gap)) and np.array_equal(P["gap"][~np.isnan(gap)], gap[~np.isnan(gap)])
    assert np.array_equal(P["pres"], pres)
    assert np.array_equal(P["v"], lay["kin"][:, 0].astype(np.float64))
    assert np.array_equal(P["clip_idx"], lay["clip_idx"].astype(np.int64)) and np.array_equal(P["t"], lay["t"].astype(np.int64))
    assert np.isnan(P["gap"]).any() and np.isfinite(P["gap"]).any() and (P["pres"] == 1).any() and (P["pres"] == 0).any()
    assert P["gap"].dtype == np.float64 and P["pres"].dtype == np.float64 and P["v"].dtype == np.float64


def test_gap_rule_boundaries():
    lay = TL.make_layout(1, frames=[6])
    join = {"clip": np.array(["c000"]), "off": np.array([0]), "n": np.array([6]), "lead_valid": np.array([1, 1, 1, 1, 0, 1], np.int8),
            "lead_gap": np.array([0.0, 60.0, 60.5, -0.5, 10.0, np.nan], np.float32)}
    P = BP.build_pack(lay_only(lay), join)
    assert P["gap"][0] == 0.0 and P["gap"][1] == 60.0 and np.isnan(P["gap"][2]) and np.isnan(P["gap"][3]) and np.isnan(P["gap"][4]) and np.isnan(P["gap"][5])
    assert list(P["pres"]) == [1, 1, 1, 1, 0, 1]


def test_join_order_does_not_matter():
    lay = TL.make_layout(6)
    a = BP.build_pack(lay_only(lay), TL.make_join(lay, shuffle=False))
    j2 = TL.make_join(lay, shuffle=True)
    b = BP.build_pack(lay_only(lay), j2)
    ga, pa = TL.expected_pack(lay, TL.make_join(lay, shuffle=False))
    gb, pb = TL.expected_pack(lay, j2)
    assert np.array_equal(a["pres"], pa) and np.array_equal(b["pres"], pb)


def test_clip_missing_from_the_join_is_refused():
    lay = TL.make_layout(5)
    join = TL.make_join(lay)
    keep = np.array([c != "c003" for c in join["clip"]])
    j2 = {k: (v if k in ("lead_valid", "lead_gap") else v[keep]) for k, v in join.items()}
    with pytest.raises(SystemExit) as e:
        BP.build_pack(lay_only(lay), j2)
    assert "not in the join" in str(e.value)


def test_frame_count_mismatch_with_the_join_is_refused():
    lay = TL.make_layout(4)
    join = TL.make_join(lay)
    j = int(np.flatnonzero(join["clip"] == "c001")[0])
    join["n"] = join["n"].copy()
    join["n"][j] += 1
    with pytest.raises(SystemExit) as e:
        BP.build_pack(lay_only(lay), join)
    assert "frame count" in str(e.value)


def test_rows_not_clip_by_clip_are_refused():
    lay = TL.make_layout(3)
    L = lay_only(lay)
    L["t"] = L["t"].copy()
    L["t"][2] = 7
    with pytest.raises(SystemExit):
        BP.build_pack(L, TL.make_join(lay))


def test_main_writes_the_pack_files_and_meta(tmp_path):
    lay, join, lp, jp = setup(tmp_path)
    out = TL.sh(tmp_path / "pack.npz")
    assert BP.main(["--layout", lp, "--join", jp, "--out", out]) == 0
    z = np.load(out, allow_pickle=False)
    assert set(z.files) == {"clips", "clip_idx", "t", "gap", "pres", "v", "meta"}
    meta = json.loads(str(z["meta"]))
    assert meta["n_rows"] == len(lay["t"]) and meta["n_clips"] == 8 and len(meta["join_sha256"]) == 64 and meta["states_checked"] is False
    gap, pres = TL.expected_pack(lay, join)
    assert np.array_equal(z["pres"], pres)


def test_main_refuses_an_existing_output_and_changes_nothing(tmp_path):
    lay, join, lp, jp = setup(tmp_path)
    out = tmp_path / "pack.npz"
    out.write_bytes(b"keep")
    assert BP.main(["--layout", lp, "--join", jp, "--out", TL.sh(out)]) == 2
    assert out.read_bytes() == b"keep"


def test_main_refuses_a_missing_input(tmp_path):
    assert BP.main(["--layout", TL.sh(tmp_path / "nope.npz"), "--join", TL.sh(tmp_path / "nope2.npz"), "--out", TL.sh(tmp_path / "o.npz")]) == 2


def test_states_check_passes_for_the_same_layout_and_fails_for_a_changed_t(tmp_path):
    lay, join, lp, jp = setup(tmp_path)
    Z = TL.make_states(lay)
    sp = tmp_path / "s.npz"
    TL.write_states(sp, lay, Z)
    assert BP.main(["--layout", lp, "--join", jp, "--out", TL.sh(tmp_path / "p1.npz"), "--states", TL.sh(sp)]) == 0
    bad = dict(lay)
    bad["t"] = lay["t"].copy()
    bad["t"][0] = 5
    sp2 = tmp_path / "s2.npz"
    TL.write_states(sp2, bad, Z)
    assert BP.main(["--layout", lp, "--join", jp, "--out", TL.sh(tmp_path / "p2.npz"), "--states", TL.sh(sp2)]) == 2
    assert not (tmp_path / "p2.npz").exists()


def test_states_check_fails_for_a_different_clip_order(tmp_path):
    lay, join, lp, jp = setup(tmp_path)
    bad = dict(lay)
    bad["clips"] = lay["clips"][::-1].copy()
    sp = tmp_path / "s.npz"
    TL.write_states(sp, bad, TL.make_states(lay))
    assert BP.main(["--layout", lp, "--join", jp, "--out", TL.sh(tmp_path / "p.npz"), "--states", TL.sh(sp)]) == 2
