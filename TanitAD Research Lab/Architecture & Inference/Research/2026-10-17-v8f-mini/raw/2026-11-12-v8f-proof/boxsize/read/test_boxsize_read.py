"""tests of read_boxsize.py on tiny synthetic data (card PREREG_V8F_PROOF_ADD124). No result, dump, token, box, image or video file of the project is opened."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import read_boxsize as BS  # noqa: E402

PRE = BS.PRE


def cli(argv):
    return BS.main(list(argv))


def run_controls_cli(tmp_path, ks="cuts", fault=None, capsys=None):
    argv, info = BS.write_tree(str(tmp_path / "w"), ks, fault)
    out = str(tmp_path / "CONTROLS.json")
    rc = cli(["--stage", "controls", "--out", out] + argv)
    txt = capsys.readouterr().out if capsys else ""
    return rc, out, argv, txt


# --------------------------------------------------------------------------------------------- encoding and the map
def test_decode_formula():
    w, h = BS.decode(16.0 * 0.4, -2.0 + 8.0 * (-0.5))
    assert abs(w - np.exp(1.6 + 0.4)) < 1e-12 and abs(h - np.exp(1.6 - 0.5)) < 1e-12
    w0, h0 = BS.decode(0.0, -2.0)                                       # u = 0 -> exp(1.6) degrees in both
    assert abs(w0 - np.exp(1.6)) < 1e-12 and abs(h0 - np.exp(1.6)) < 1e-12


def test_encode_decode_roundtrip_inside_limits():
    rng = np.random.RandomState(3)
    w, h = np.exp(rng.uniform(1.6 - 2, 1.6 + 2, 50)), np.exp(rng.uniform(1.6 - 2, 1.6 + 2, 50))
    a, b = BS.encode(w, h)
    w2, h2 = BS.decode(a, b)
    assert np.abs(w2 - w).max() < 1e-9 and np.abs(h2 - h).max() < 1e-9
    assert not BS.outside_limits(w).any()
    assert BS.outside_limits(np.array([0.5, 40.0, 1.0])).tolist() == [True, True, False]
    lo, hi = np.exp(-0.4), np.exp(3.6)
    assert abs(lo - 0.67) < 0.01 and abs(hi - 36.6) < 0.05


def test_map_g_hand_checked():
    ln = np.arange(6, dtype=np.float64)
    gap = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    G = BS.MapG(ln, ln, gap, k=3)
    # query at ln 2.1 in both: distances idx2 0.1, idx3 0.9, idx1 1.1 (in scaled units, same order) -> median(30, 40, 20) = 30
    assert G.predict(np.exp([2.1]), np.exp([2.1]))[0] == 30.0
    # query far below: nearest three are idx 0, 1, 2 -> median 20
    assert G.predict(np.exp([-3.0]), np.exp([-3.0]))[0] == 20.0
    # features are (ln h, ln w), each scaled by its own std: w is stretched, so only h separates the frames here
    lnw = np.zeros(6)
    G2 = BS.MapG(ln, lnw + np.array([0.0, 0.1, 0.0, 0.1, 0.0, 0.1]), gap, k=1)
    assert G2.predict(np.exp([0.0]), np.exp([4.0]))[0] == 50.0


def test_map_g_tie_goes_to_lower_train_index():
    lnh = np.array([0.0, 1.0, 5.0, 6.0, 1.0, 9.0])
    lnw = np.array([0.0, 1.0, 5.0, 6.0, 1.0, 9.0])
    gap = np.array([10.0, 11.0, 12.0, 13.0, 99.0, 14.0])              # frames 1 and 4 are the same point: the lower index (gap 11) wins at k = 1
    G = BS.MapG(lnh, lnw, gap, k=1)
    assert G.predict(np.exp([1.0]), np.exp([1.0]))[0] == 11.0
    G = BS.MapG(lnh[::-1], lnw[::-1], gap[::-1], k=1)                  # reversed order: the frame that now has the lower index wins
    assert G.predict(np.exp([1.0]), np.exp([1.0]))[0] == 99.0
    G = BS.MapG(lnh, lnw, gap, k=2)
    assert G.predict(np.exp([1.0]), np.exp([1.0]))[0] == 55.0          # k = 2: both coincident frames, median of 11 and 99


def test_map_g_not_finite_query_is_nan():
    G = BS.MapG(np.arange(30.0), np.arange(30.0), np.arange(30.0), k=5)
    out = G.predict(np.array([np.nan, np.inf, 0.0, 2.0]), np.array([2.0, 2.0, 2.0, 2.0]))
    assert np.isnan(out[:3]).all() and np.isfinite(out[3])


def test_build_map_filters_train_frames(tmp_path):
    n = 6
    d = dict(has_lead=np.array([1, 1, 1, 1, 1, 0], bool), gap_re=np.array([10, 20, np.nan, 40, 70, 90.0]), ang_w=np.array([2, 2, 2, 0, 3, 3.0]), ang_h=np.array([1, np.nan, 1, 1, 1, 1.0]),
             ctr_az=np.zeros(n), ctr_el=np.zeros(n))
    np.savez(str(tmp_path / "a.npz"), **d)
    G, info = BS.build_map(str(tmp_path), ["a", "b"])
    assert info["n_train_clips_missing_box"] == 1 and info["n_train_frames"] == 2                 # frames 0 and 4 only; no cut on the gap (70 m is kept)
    assert sorted(G.gap.tolist()) == [10.0, 70.0]


# --------------------------------------------------------------------------------------------- the controls
def test_clean_tree_passes_all_controls(tmp_path, capsys):
    rc, out, _, txt = run_controls_cli(tmp_path, capsys=capsys)
    assert rc == 0
    res = json.load(io.open(out, encoding="utf-8"))
    assert res["all_pass"] is True and sorted(res["controls"]) == sorted(BS.CONTROLS)
    lines = txt.splitlines()
    assert [ln for ln in lines if ln.startswith("s") and ln.endswith(("PASS", "FAIL")) and len(ln) == 7] == ["s%d PASS" % i for i in range(7)]
    assert lines[-1] == "ZZ READ-BOXSIZE controls all_pass True"


@pytest.mark.parametrize("fault", sorted(BS.CTL_FAULTS))
def test_each_control_fails_on_its_planted_fault(tmp_path, capsys, fault):
    rc, out, _, txt = run_controls_cli(tmp_path, fault=fault, capsys=capsys)
    assert rc == 2
    res = json.load(io.open(out, encoding="utf-8"))
    assert res["all_pass"] is False and res["controls"][fault]["pass"] is False
    assert "%s FAIL" % fault in txt.splitlines()
    assert txt.splitlines()[-1] == "ZZ READ-BOXSIZE controls all_pass False"
    assert fault in {k for k, v in res["controls"].items() if not v["pass"]}


def test_controls_stage_holds_no_gap_error_of_arm_s(tmp_path, capsys):
    rc, out, _, txt = run_controls_cli(tmp_path, ks="mild", capsys=capsys)
    P = BS.syn_payload("mild")
    body = io.open(out, encoding="utf-8").read()
    res = json.loads(body)
    for k in BS.ARM_S:
        for v in (P["E"][k], P["far"]["signed_single"][k]):
            for s in ("%.4f" % v, "%.3f" % v, repr(float(v))):
                assert s not in body and s not in txt
    for key in ("sums", "R", "iv", "bins", "far", "beside", "payload", "values", "withheld"):
        assert key not in res
    assert "S_0" not in json.dumps({k: v.get("E") for k, v in res["controls"].items() if isinstance(v, dict) and isinstance(v.get("E"), dict)})


def test_s4_clauses_pure():
    ok = {"n_train_frames": 100, "n_train_clips_missing_box": 0, "E_label": 1.0, "E_perm": 9.0, "E_const": 9.0}
    assert BS.ctl_s4(ok)["pass"]
    assert not BS.ctl_s4(dict(ok, E_label=2.3))["pass"]
    assert BS.ctl_s4(dict(ok, E_label=2.2))["pass"]
    assert not BS.ctl_s4(dict(ok, E_perm=8.5))["pass"]                # below 0.95 of the constant's
    assert BS.ctl_s4(dict(ok, E_perm=8.55))["pass"]
    assert not BS.ctl_s4(dict(ok, n_train_clips_missing_box=1))["pass"]
    assert not BS.ctl_s5({"E_sp": 8.0, "E_const": 9.0})["pass"] and BS.ctl_s5({"E_sp": 9.0, "E_const": 9.0})["pass"]


def test_s2_pure_clauses():
    rec = {"module_values_changed": [{"name": "load_box"}], "module_values_restored": True, "locator_fit_sha256": "a" * 64}
    recs = {k: dict(rec, seed=BS.FIT_SEED[k], permute_targets=(k == BS.PERM_FIT)) for k in BS.FITS}
    rt = {"max_abs_diff": 1e-13, "n_inside": 10}
    assert BS.ctl_s2(recs, rt)["pass"]
    assert not BS.ctl_s2(recs, dict(rt, max_abs_diff=1e-8))["pass"]
    bad = dict(recs)
    bad["S_1"] = dict(recs["S_1"], module_values_restored=False)
    assert not BS.ctl_s2(bad, rt)["pass"]
    bad["S_1"] = dict(recs["S_1"], locator_fit_sha256="b" * 64)
    assert not BS.ctl_s2(bad, rt)["pass"]
    bad["S_1"] = dict(recs["S_1"], module_values_changed=[])
    assert not BS.ctl_s2(bad, rt)["pass"]
    bad["SP_0"] = dict(recs["SP_0"], permute_targets=False)
    assert not BS.ctl_s2(bad, rt)["pass"]


# --------------------------------------------------------------------------------------------- the names stage
def test_names_refuse_without_a_passing_controls_file(tmp_path, capsys):
    argv, _ = BS.write_tree(str(tmp_path / "w"))
    res = str(tmp_path / "RESULT.json")
    assert cli(["--stage", "names", "--controls", str(tmp_path / "none.json"), "--out", res] + argv) == 3
    assert not os.path.exists(res)
    # a failing controls file
    rc, out, argv2, _ = run_controls_cli(tmp_path / "f", fault="s3", capsys=capsys)
    assert rc == 2
    assert cli(["--stage", "names", "--controls", out, "--out", res] + argv2) == 3
    assert "REFUSED" in capsys.readouterr().out and not os.path.exists(res)
    # a hand-edited file that claims all pass without the controls present
    fake = str(tmp_path / "fake.json")
    io.open(fake, "w", encoding="utf-8").write(json.dumps({"card": BS.CARD, "all_pass": True, "controls": {"s0": {"pass": True}}}))
    assert cli(["--stage", "names", "--controls", fake, "--out", res] + argv2) == 3
    assert not os.path.exists(res)


def test_names_refuse_when_an_input_changed_after_the_controls(tmp_path, capsys):
    rc, out, argv, _ = run_controls_cli(tmp_path, capsys=capsys)
    assert rc == 0
    pred = tmp_path / "w" / "fit_out" / "pred" / "LOC_s1.npz"
    z = dict(np.load(str(pred), allow_pickle=True))
    z["az_pred"] = z["az_pred"] + 0.5
    np.savez(str(pred), **z)
    res = str(tmp_path / "RESULT.json")
    assert cli(["--stage", "names", "--controls", out, "--out", res] + argv) == 3
    assert not os.path.exists(res)


@pytest.mark.parametrize("ks,ratio,far", [("cuts", "-CUTS-GAP-ERROR", "-FAR-LEADS-READ"), ("worse", "-WORSE", "-FAR-LEADS-STAY-SHORT"), ("sep", "-NOT-SEPARATED", "-FAR-LEADS-STAY-SHORT")])
def test_names_stage_reaches_each_ratio_name_and_both_far_names(tmp_path, capsys, ks, ratio, far):
    rc, out, argv, _ = run_controls_cli(tmp_path, ks=ks, capsys=capsys)
    assert rc == 0
    res, log = str(tmp_path / "RESULT.json"), str(tmp_path / "names.log")
    rc = cli(["--stage", "names", "--controls", out, "--out", res, "--log", log] + argv)
    txt = capsys.readouterr().out
    assert rc == 0
    r = json.load(io.open(res, encoding="utf-8"))
    assert r["names"]["RATIO"] == PRE + ratio and r["names"]["FAR"] == PRE + far and r["second_derivation_agrees"] is True
    body = io.open(log, encoding="utf-8").read()
    assert PRE + ratio in body and PRE + far in body and "ratio S / D" in body and "by label gap, arm S" in body and "size read error in per cent" in body
    assert "outside the limits" in body and "G on the label sizes" in body and "LONGITUDINAL" in body
    assert txt.splitlines()[-1] == "ZZ READ-BOXSIZE names rc 0"
    # a second read into the same files is refused
    assert cli(["--stage", "names", "--controls", out, "--out", res, "--log", log] + argv) == 2


def test_bar_needs_ratio_and_interval():
    P = BS.syn_payload("cuts")
    assert BS.derive_names(P)[0]["bar"] is True
    P2 = dict(P, R=0.9)
    assert BS.names_first(P2)["bar"] is False
    P3 = dict(P, iv=[0.5, 1.02])
    assert BS.names_first(P3)["bar"] is False and BS.names_first(P3)["RATIO"] == PRE + "-NOT-SEPARATED"


def test_name_rules_pure():
    assert BS.name_ratio([0.5, 0.99]) == PRE + "-CUTS-GAP-ERROR" and BS.name_ratio([1.01, 1.4]) == PRE + "-WORSE" and BS.name_ratio([0.9, 1.1]) == PRE + "-NOT-SEPARATED" and BS.name_ratio(None) is None
    iv_pos, iv_zero = [1.0, 2.0], [-0.5, 2.0]
    assert BS.name_far(-3.0, -6.0, iv_pos).endswith("FAR-LEADS-READ")           # exactly half is enough
    assert BS.name_far(-3.1, -6.0, iv_pos).endswith("STAY-SHORT")
    assert BS.name_far(-1.0, -6.0, iv_zero).endswith("STAY-SHORT") and BS.name_far(-1.0, -6.0, None).endswith("STAY-SHORT")
    assert BS.name_far(1.0, -6.0, [-3.0, -1.0]).endswith("FAR-LEADS-READ")


def test_two_derivations_agree_on_every_world():
    for ks in ("cuts", "mild", "worse", "sep"):
        n1, n2, agree = BS.derive_names(BS.syn_payload(ks))
        assert agree and n1 == n2


def test_selftest_flag(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli(["--selftest"]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "ZZ READ-BOXSIZE SELFTEST OK"


def test_cli_as_subprocess_prints_last_line(tmp_path):
    argv, _ = BS.write_tree(str(tmp_path / "w"))
    out = str(tmp_path / "C.json")
    p = subprocess.run([sys.executable, os.path.join(HERE, "read_boxsize.py"), "--stage", "controls", "--out", out] + argv, capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0 and p.stdout.splitlines()[-1] == "ZZ READ-BOXSIZE controls all_pass True"


def test_no_banned_words_in_the_kit_files():
    words = ["vo" + "id", "super" + "seded", "re" + "tracted", "for" + "merly", "cor" + "rection", "with" + "drawn", "no lon" + "ger", "closed " + "loop"]
    for name in os.listdir(HERE):
        if name.endswith((".py", ".sh", ".md", ".txt")):
            body = io.open(os.path.join(HERE, name), encoding="utf-8", errors="replace").read().lower()
            for w in words:
                assert w not in body, (name, w)
