"""Tests of dose_d2_state_check.py (control d2): equal states pass; a difference above the bar fails; the one-clip shift fails (planted, field and exit code); streaming equals loading; records, structure, layout."""
import json

import numpy as np
import pytest

import dose_d2_state_check as D2  # noqa: E402
import dose_testlib as TL  # noqa: E402

LO, HI = 2, 6


def files(tmp_path, n=9, D=12, ckpt_new="a" * 64, ckpt_ban="a" * 64, scale=1.0, tweak=None):
    lay = TL.make_layout(n)
    Z = TL.make_states(lay, D=D, scale=scale)
    rows, sub = TL.banked_from_new(lay, Z, LO, HI)
    Zb = Z[rows].copy()
    if tweak:
        tweak(Zb)
    new, ban = tmp_path / "new.npz", tmp_path / "ban.npz"
    TL.write_states(new, lay, Z, ckpt=ckpt_new, lo=0, hi=n)
    bl = {"clips": sub["clips"], "clip_idx": sub["clip_idx"], "t": sub["t"], "kin": sub["kin"], "frames": sub["frames"]}
    TL.write_states(ban, bl, Zb, ckpt=ckpt_ban, lo=LO, hi=HI)
    return lay, Z, TL.sh(new), TL.sh(ban)


def run(new, ban, **kw):
    return D2.run(new, ban, LO, HI, kw.pop("rel_bar", 1e-4), **kw)


def test_equal_states_pass_and_the_planted_shift_fails(tmp_path):
    lay, Z, new, ban = files(tmp_path)
    R = run(new, ban)
    assert R["pass"] is True and R["values"]["max_abs_diff"] == 0.0 and R["planted_shift"]["fails"] is True and R["planted_shift"]["max_abs_diff"] > R["values"]["bar_abs"]
    assert R["structure"]["ok"] and R["records"]["ok"]


def test_a_difference_above_the_bar_fails(tmp_path):
    def tw(Zb):
        Zb[5, 3] = np.float16(float(Zb[5, 3]) + 0.05)
    lay, Z, new, ban = files(tmp_path, tweak=tw)
    R = run(new, ban)
    assert R["pass"] is False and R["values"]["ok"] is False and R["values"]["max_abs_diff"] > R["values"]["bar_abs"]


def test_a_difference_below_the_bar_passes(tmp_path):
    def tw(Zb):
        Zb[5, 3] = np.float16(0.0)
    lay, Z, new, ban = files(tmp_path)
    z = np.load(new, allow_pickle=False)
    # set that element tiny in the new file: difference 5e-5 against rms about 1
    Zn = z["Z"].copy()
    rows = np.flatnonzero((lay["clip_idx"] >= LO) & (lay["clip_idx"] < HI))
    Zn[rows[5], 3] = np.float16(5e-5)
    TL.write_states(new, lay, Zn, lo=0, hi=9)
    zb = np.load(ban, allow_pickle=False)
    Zb = zb["Z"].copy()
    Zb[5, 3] = np.float16(0.0)
    TL.write_states(ban, {"clips": lay["clips"][LO:HI], "clip_idx": zb["clip_idx"], "t": zb["t"]}, Zb, lo=LO, hi=HI)
    R = run(new, ban)
    assert R["pass"] is True and 0 < R["values"]["max_abs_diff"] <= R["values"]["bar_abs"]


def test_bar_is_relative_to_the_rms_of_the_state(tmp_path):
    def tw(Zb):
        Zb[4, 1] = np.float16(float(Zb[4, 1]) + 0.01)
    lay, Z, new, ban = files(tmp_path, scale=100.0, tweak=tw)         # rms about 100: bar 0.01 absolute; the change of 0.01 sits at the bar
    R = run(new, ban)
    assert R["values"]["bar_abs"] == pytest.approx(1e-4 * R["values"]["rms_state"])
    lay2, Z2, new2, ban2 = files(tmp_path / "b" if (tmp_path / "b").mkdir() is None else tmp_path, scale=1.0, tweak=tw)
    assert run(new2, ban2)["pass"] is False                              # rms about 1: the same change is far above the bar


def test_shifted_comparison_fails_by_value_not_by_structure(tmp_path):
    lay, Z, new, ban = files(tmp_path)
    R = run(new, ban)
    assert R["planted_shift"]["shift_clips"] == 1 and R["planted_shift"]["rows_compared"] > 0 and R["planted_shift"]["fails"] is True


def test_blind_comparison_is_detected_when_every_row_is_identical(tmp_path):
    lay = TL.make_layout(9)
    Z = np.ones((len(lay["t"]), 12), np.float16)
    rows, sub = TL.banked_from_new(lay, Z, LO, HI)
    new, ban = tmp_path / "new.npz", tmp_path / "ban.npz"
    TL.write_states(new, lay, Z, lo=0, hi=9)
    TL.write_states(ban, {"clips": sub["clips"], "clip_idx": sub["clip_idx"], "t": sub["t"]}, Z[rows], lo=LO, hi=HI)
    R = run(TL.sh(new), TL.sh(ban))
    assert R["pass"] is False and R["planted_shift"]["fails"] is False and "blind" in R["why"]


def test_checkpoint_sha_difference_fails_without_comparing_values(tmp_path):
    lay, Z, new, ban = files(tmp_path, ckpt_ban="b" * 64)
    R = run(new, ban)
    assert R["pass"] is False and R["records"]["ckpt_sha256_equal"] is False and "values" not in R


def test_a_shifted_clip_order_in_the_new_file_fails(tmp_path):
    lay = TL.make_layout(9)
    Z = TL.make_states(lay)
    rows, sub = TL.banked_from_new(lay, Z, LO, HI)
    # banked built from clips 3..6 of the new file: the claimed range 2..5 then differs in names and values
    rows2, sub2 = TL.banked_from_new(lay, Z, LO + 1, HI + 1)
    new, ban = tmp_path / "new.npz", tmp_path / "ban.npz"
    TL.write_states(new, lay, Z, lo=0, hi=9)
    TL.write_states(ban, {"clips": sub2["clips"], "clip_idx": sub2["clip_idx"], "t": sub2["t"]}, Z[rows2], lo=LO, hi=HI)
    R = run(TL.sh(new), TL.sh(ban))
    assert R["pass"] is False


def test_row_count_mismatch_fails(tmp_path):
    lay, Z, new, ban = files(tmp_path)
    zb = np.load(ban, allow_pickle=False)
    TL.write_states(ban, {"clips": zb["clips"], "clip_idx": zb["clip_idx"][:-1], "t": zb["t"][:-1]}, zb["Z"][:-1], lo=LO, hi=HI)
    R = run(new, ban)
    assert R["pass"] is False and R["structure"]["rows_equal"] is False


def test_new_record_must_start_at_position_zero_and_name_its_clip_count(tmp_path):
    lay, Z, new, ban = files(tmp_path)
    rec = json.load(open(new + ".json", encoding="utf-8"))
    rec["lo"] = 600
    json.dump(rec, open(new + ".json", "w", encoding="utf-8"))
    assert run(new, ban)["pass"] is False
    rec["lo"], rec["n_clips"] = 0, 3
    json.dump(rec, open(new + ".json", "w", encoding="utf-8"))
    assert run(new, ban)["pass"] is False


def test_missing_record_is_refused(tmp_path):
    lay, Z, new, ban = files(tmp_path)
    import os
    os.replace(ban + ".json", ban + ".json.moved")
    R = run(new, ban)
    assert R["pass"] is False and "refused" in R["why"]


def test_layout_option(tmp_path):
    lay, Z, new, ban = files(tmp_path)
    lp = tmp_path / "lay.npz"
    np.savez(lp, **lay)
    assert run(new, ban, layout=TL.sh(lp))["layout"]["ok"] is True
    bad = dict(lay)
    bad["t"] = lay["t"].copy()
    bad["t"][-1] += 1
    lp2 = tmp_path / "lay2.npz"
    np.savez(lp2, **bad)
    R = run(new, ban, layout=TL.sh(lp2))
    assert R["layout"]["ok"] is False and R["pass"] is False


def test_streaming_rows_equal_a_full_load_and_chunks_do_not_change_the_result(tmp_path, monkeypatch):
    lay, Z, new, ban = files(tmp_path)
    rs = D2.RowStream(new, "Z")
    full = np.load(new, allow_pickle=False)["Z"].astype(np.float64)
    got = np.concatenate([x for _, x in rs.rows(7, 31, chunk=5)])
    assert np.array_equal(got, full[7:31]) and rs.shape == full.shape
    rs.close()
    a = run(new, ban)
    monkeypatch.setattr(D2, "CHUNK", 3)
    rs2 = D2.RowStream(new, "Z")
    assert np.array_equal(np.concatenate([x for _, x in rs2.rows(0, len(full), chunk=3)]), full)
    rs2.close()
    b = run(new, ban)
    assert a["values"]["max_abs_diff"] == b["values"]["max_abs_diff"] and a["values"]["rms_state"] == pytest.approx(b["values"]["rms_state"])


def test_main_exit_codes_and_json_fields(tmp_path):
    lay, Z, new, ban = files(tmp_path)
    out = tmp_path / "d2.json"
    assert D2.main(["--new", new, "--banked", ban, "--lo", str(LO), "--hi", str(HI), "--out", TL.sh(out)]) == 0
    txt = out.read_text(encoding="utf-8")
    R = json.loads(txt)
    assert '"pass": true' in txt and R["planted_shift"]["fails"] is True and "max_abs_diff" in R["values"]
    lay, Z, new, ban = files(tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path, ckpt_ban="c" * 64)
    assert D2.main(["--new", new, "--banked", ban, "--lo", str(LO), "--hi", str(HI)]) == 2
    assert D2.main(["--new", TL.sh(tmp_path / "none.npz"), "--banked", ban, "--lo", str(LO), "--hi", str(HI)]) == 2
