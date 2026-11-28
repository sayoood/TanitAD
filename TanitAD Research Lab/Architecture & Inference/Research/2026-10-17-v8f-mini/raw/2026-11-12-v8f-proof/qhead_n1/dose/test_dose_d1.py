"""Tests of dose_d1_pack_check.py (control d1): equal packs pass; a missing row, an extra row, any value difference, a changed clip name, another join all fail."""
import json

import numpy as np

import build_dose_pack as BP  # noqa: E402
import dose_d1_pack_check as D1  # noqa: E402
import dose_testlib as TL  # noqa: E402

LO, HI = 2, 6


def packs(n=9):
    lay = TL.make_layout(n)
    join = TL.make_join(lay)
    P = BP.build_pack({k: lay[k] for k in ("clips", "clip_idx", "t", "kin")}, join)
    rows = np.flatnonzero((P["clip_idx"] >= LO) & (P["clip_idx"] < HI))
    B = {"clips": P["clips"][LO:HI], "clip_idx": P["clip_idx"][rows] - LO, "t": P["t"][rows], "gap": P["gap"][rows].copy(), "pres": P["pres"][rows].copy(), "v": P["v"][rows].copy()}
    return P, B


def meta(sha="s1"):
    return {"join_sha256": sha}


def run(P, B, sha_new="s1", sha_b="s1"):
    return D1.compare(P, B, LO, HI, meta(sha_new), meta(sha_b))


def test_equal_packs_pass_with_zero_differences():
    P, B = packs()
    R = run(P, B)
    assert R["pass"] is True and R["gap_max_abs_diff"] == 0.0 and R["pres_max_abs_diff"] == 0.0 and R["v_max_abs_diff"] == 0.0
    assert R["missing_rows"] == 0 and R["extra_rows"] == 0 and R["rows_new_in_range"] == R["rows_banked"] > 0


def test_rows_outside_the_range_do_not_matter():
    P, B = packs()
    P["gap"] = P["gap"].copy()
    P["gap"][P["clip_idx"] == 0] = 123.0
    assert run(P, B)["pass"] is True


def test_missing_row_fails():
    P, B = packs()
    drop = int(np.flatnonzero((P["clip_idx"] == 3) & (P["t"] == 2))[0])
    keep = np.arange(len(P["t"])) != drop
    P2 = {k: (v[keep] if k != "clips" else v) for k, v in P.items()}
    R = run(P2, B)
    assert R["pass"] is False and R["missing_rows"] == 1


def test_extra_row_fails():
    P, B = packs()
    B2 = {k: (v[1:] if k != "clips" else v) for k, v in B.items()}
    R = run(P, B2)
    assert R["pass"] is False and R["extra_rows"] == 1


def test_tiny_value_difference_in_gap_fails():
    P, B = packs()
    i = int(np.flatnonzero(np.isfinite(B["gap"]))[0])
    B["gap"][i] += 1e-9
    R = run(P, B)
    assert R["pass"] is False and R["gap_max_abs_diff"] > 0


def test_nan_pattern_difference_fails():
    P, B = packs()
    i = int(np.flatnonzero(np.isfinite(B["gap"]))[0])
    B["gap"][i] = np.nan
    R = run(P, B)
    assert R["pass"] is False and R["gap_nan_pattern_equal"] is False


def test_presence_and_speed_differences_fail():
    P, B = packs()
    B1 = dict(B)
    B1["pres"] = 1.0 - B["pres"]
    assert run(P, B1)["pass"] is False
    B2 = dict(B)
    B2["v"] = B["v"] + 1e-12
    assert run(P, B2)["pass"] is False


def test_clip_name_difference_fails():
    P, B = packs()
    B["clips"] = B["clips"].copy()
    B["clips"][1] = "other"
    R = run(P, B)
    assert R["pass"] is False and R["clip_names_equal"] is False


def test_different_join_sha_fails_and_missing_meta_fails():
    P, B = packs()
    assert run(P, B, sha_new="s1", sha_b="s2")["pass"] is False
    assert D1.compare(P, B, LO, HI, {}, {})["pass"] is False


def test_range_of_the_wrong_size_fails():
    P, B = packs()
    assert D1.compare(P, B, LO, HI + 1, meta(), meta())["pass"] is False


def test_main_exit_codes_and_json(tmp_path, capsys):
    P, B = packs()
    pn, pb = tmp_path / "new.npz", tmp_path / "ban.npz"
    np.savez(pn, **P, meta=json.dumps(meta()))
    np.savez(pb, **B, meta=json.dumps(meta()))
    out = tmp_path / "d1.json"
    assert D1.main(["--new-pack", TL.sh(pn), "--banked-pack", TL.sh(pb), "--lo", str(LO), "--hi", str(HI), "--out", TL.sh(out)]) == 0
    R = json.loads(out.read_text(encoding="utf-8"))
    assert R["pass"] is True and '"pass": true' in out.read_text(encoding="utf-8")
    B["gap"][np.flatnonzero(np.isfinite(B["gap"]))[0]] += 0.5
    np.savez(pb, **B, meta=json.dumps(meta()))
    assert D1.main(["--new-pack", TL.sh(pn), "--banked-pack", TL.sh(pb), "--lo", str(LO), "--hi", str(HI)]) == 2
    assert D1.main(["--new-pack", TL.sh(tmp_path / "nope.npz"), "--banked-pack", TL.sh(pb), "--lo", str(LO), "--hi", str(HI)]) == 2
