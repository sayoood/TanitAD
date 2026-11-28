"""tests of read_telegap.py on tiny synthetic dumps (card PREREG_V8F_PROOF_ADD122). No result file of the project is opened."""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import read_telegap as TG  # noqa: E402

RL, _RPATH = TG.load_lookgap(TG.LOOKGAP_DEFAULT)
LK, _LSHA, _LPATH = RL.load_lookat(TG.LOOKAT_DEFAULT)
PRE = TG.PRE


def payload(kf="cuts", km="cuts", amp=0.2, zero=False):
    return TG.syn_payload(LK, RL, kf, km, amp=amp, zero=zero)


# --------------------------------------------------------------------------------------------- names
@pytest.mark.parametrize("kf,km,wf,wm", [
    ("cuts", "cuts", "-F-CUTS-GAP-ERROR", "-M-CUTS-GAP-ERROR"),
    ("cuts", "worse", "-F-CUTS-GAP-ERROR", "-M-WORSE"),
    ("worse", "cuts", "-F-WORSE", "-M-CUTS-GAP-ERROR"),
    ("sep", "sep", "-F-NOT-SEPARATED", "-M-NOT-SEPARATED"),
])
def test_name_branches_and_two_derivations_agree(kf, km, wf, wm):
    P = payload(kf, km)
    n1, n2, agree = TG.derive_names(P)
    assert agree
    assert n1["F"] == PRE + wf and n1["M"] == PRE + wm
    assert n2["F"] == n1["F"] and n2["M"] == n1["M"]


def test_far_leads_names():
    n1, _, agree = TG.derive_names(payload("cuts", "cuts"))
    assert agree and n1["FAR"] == PRE + "-FAR-LEADS-READ"
    for kf in ("worse", "mild", "sep"):
        n1, _, agree = TG.derive_names(payload(kf, "cuts"))
        assert agree and n1["FAR"] == PRE + "-FAR-LEADS-STAY-SHORT"


def test_far_rule_pieces():
    iv_pos, iv_zero = [1.0, 2.0], [-0.5, 2.0]
    assert TG.name_far(-1.0, -6.0, iv_pos).endswith("FAR-LEADS-READ")
    assert TG.name_far(-3.0, -6.0, iv_pos).endswith("FAR-LEADS-READ")            # exactly half is enough
    assert TG.name_far(-3.1, -6.0, iv_pos).endswith("STAY-SHORT")
    assert TG.name_far(-1.0, -6.0, iv_zero).endswith("STAY-SHORT")               # not separated
    assert TG.name_far(-1.0, -6.0, None).endswith("STAY-SHORT")
    assert TG.name_far(None, None, iv_pos).endswith("STAY-SHORT")
    assert TG.name_far(1.0, -6.0, [-3.0, -1.0]).endswith("FAR-LEADS-READ")       # size, not sign, and a separated interval below zero


def test_name_x_branches():
    assert TG.name_x("F", [0.5, 0.99]).endswith("F-CUTS-GAP-ERROR")
    assert TG.name_x("F", [0.5, 1.0]).endswith("F-NOT-SEPARATED")                # upper end exactly 1 is not below 1
    assert TG.name_x("M", [1.01, 1.5]).endswith("M-WORSE")
    assert TG.name_x("M", [1.0, 1.5]).endswith("M-NOT-SEPARATED")
    assert TG.name_x("M", None) is None


def test_bar_line():
    P = payload("cuts", "cuts")
    assert P["R"]["F"] <= 0.85 and P["iv"]["F"][1] < 1
    assert TG.derive_names(P)[0]["bar"] is True
    Pm = payload("mild", "cuts", amp=0.02)                                        # cuts the error (ratio 0.875) but misses the bar
    n1, _, agree = TG.derive_names(Pm)
    assert agree and n1["F"] == PRE + "-F-CUTS-GAP-ERROR" and 0.85 < Pm["R"]["F"] < 1 and n1["bar"] is False
    for kf in ("worse", "sep"):
        assert TG.derive_names(payload(kf, "cuts"))[0]["bar"] is False
    lines = TG.print_sections(P, TG.names_first(P))
    assert any(ln.startswith("bar met: yes") for ln in lines)
    lines = TG.print_sections(Pm, TG.names_first(Pm))
    assert any(ln.startswith("bar met: no") for ln in lines)


def test_no_interval_gives_no_name():
    P = payload(zero=True)
    assert P["iv"]["F"] is None and P["iv"]["M"] is None
    n1, n2, agree = TG.derive_names(P)
    assert agree and n1["F"] is None and n1["M"] is None and n1["bar"] is False


def test_second_derivation_detects_a_disagreement():
    P = payload("cuts", "cuts")
    P["iv"]["F"] = [0.5, 1.2]                                                     # the first derivation's input is tampered
    n1, n2, agree = TG.derive_names(P)
    assert not agree


# --------------------------------------------------------------------------------------------- frame set and values
def test_common_frame_set_and_t3_rows_before_it():
    J, dumps, bx = TG.syn_world()
    P = payload()
    present = dumps["D_0"]["lab_present"]
    n_present = int(present.sum())
    assert P["n_t3_rows"] == n_present                                            # every D fit is finite on present rows
    assert P["n_common"] == n_present - 4 - 1                                     # 4 tele_dropped rows of M_0, one NaN gap of F_1
    assert P["n_tele_dropped_on_t3_rows"] == 4
    # t3 values are computed on the t3 rows, before the restriction
    lab = dumps["D_0"]["lab_gap"]
    for k in TG.ARMS["D"]:
        want = np.abs(dumps[k]["gap"][present] - lab[present]).mean()
        assert P["t3"]["E"][k] == pytest.approx(want, rel=1e-12)
    # arm values come from the common set only
    com = present.copy()
    com[np.flatnonzero(present)[:4]] = False
    com[np.flatnonzero(present)[10]] = False
    want = np.abs(dumps["F_0"]["gap"][com] - lab[com]).mean()
    assert P["E"]["F_0"] == pytest.approx(want, rel=1e-12)


def test_common_rows_needs_tele_dropped_and_ignores_extra_dumps():
    J, dumps, bx = TG.syn_world()
    L = {k: RL.lay(LK, J, d) for k, d in dumps.items()}
    for k, d in dumps.items():
        L[k]["tele_dropped"] = LK.on_rows(J, d, "tele_dropped").astype(bool) if "tele_dropped" in d else None
    L["F_2"]["tele_dropped"] = None
    with pytest.raises(ValueError):
        TG.common_rows(L)
    L["F_2"]["tele_dropped"] = np.zeros(len(L["D_0"]["gap"]), bool)
    n0 = int(TG.common_rows(L).sum())
    L["D2_0"]["gap"][:] = np.nan                                                  # D2_0 and FP_0 are not among the nine
    L["FP_0"]["gap"][:] = np.nan
    assert int(TG.common_rows(L).sum()) == n0


def test_lead_beyond_60_is_outside_the_common_set():
    J, dumps, bx = TG.syn_world()
    i = int(np.flatnonzero(dumps["D_0"]["lab_present"])[20])
    for d in dumps.values():
        d["lab_gap"] = d["lab_gap"].copy()
        d["lab_gap"][i] = 75.0
    L = {k: RL.lay(LK, J, d) for k, d in dumps.items()}
    for k, d in dumps.items():
        L[k]["tele_dropped"] = LK.on_rows(J, d, "tele_dropped").astype(bool) if "tele_dropped" in d else None
    assert TG.t3_rows(L)[i] and not TG.common_rows(L)[i]


def test_ratio_is_mean_over_fits_and_interval_brackets_it():
    P = payload("cuts", "worse")
    for X in ("F", "M"):
        assert P["R"][X] == pytest.approx(np.mean([P["E"][k] for k in TG.ARMS[X]]) / np.mean([P["E"][k] for k in TG.ARMS["D"]]))
        lo, hi = P["iv"][X]
        assert lo <= P["R"][X] <= hi
    assert P["iv"]["F"][1] < 1 < P["iv"]["M"][0]


def test_bins_signed_and_short_share():
    P = payload("cuts", "cuts")
    d40 = P["bins"]["D"][3]
    f40 = P["bins"]["F"][3]
    assert d40["bin"] == "40-60 m" and d40["n"] > 0
    assert d40["signed"] < -4.0 and d40["short5"] > 0.4                           # D reads far leads short by about 6 m
    assert abs(f40["signed"]) < 2.5 and f40["short5"] < d40["short5"]
    assert sum(b["n"] for b in P["bins"]["D"]) == P["n_common"]


def test_inside_t8_t4_and_slope_read():
    J, dumps, bx = TG.syn_world()
    P = payload()
    com_n = P["inside"]["n_lead"]
    assert com_n == P["n_common"] == P["inside"]["n_with_box"]
    # recompute the shares on the common rows independently
    present = dumps["D_0"]["lab_present"].copy()
    present[np.flatnonzero(dumps["D_0"]["lab_present"])[:4]] = False
    present[np.flatnonzero(dumps["D_0"]["lab_present"])[10]] = False
    az, el = bx["az"][present], bx["el"][present]
    s8 = float(((np.abs(az) <= 4) & (np.abs(el + 2) <= 4)).mean())
    s4 = float(((np.abs(az) <= 2) & (np.abs(el + 1) <= 2)).mean())
    assert P["inside"]["share_T8"] == pytest.approx(s8) and P["inside"]["share_T4"] == pytest.approx(s4)
    assert P["inside"]["F_inside_T8"]["n"] + P["inside"]["F_outside_T8"]["n"] == com_n
    assert P["slope"]["n_windows"] == 8
    Pz = payload(zero=True)                                                       # reads equal the label: the correlation is 1
    for X in "DMF":
        assert Pz["slope"]["corr_with_label_closing_speed"][X] == pytest.approx(1.0)


def test_slopes_helper():
    com = np.ones(60, bool)
    y = np.arange(60) * 0.5
    out = TG.slopes(com, np.array([0]), np.array([60]), {"y": y})
    assert len(out["y"]) == 2 and np.allclose(out["y"], 0.5)
    com[30] = False
    out = TG.slopes(com, np.array([0]), np.array([60]), {"y": y})
    assert len(out["y"]) == 2                                                     # windows 0-24 and 31-55; row 30 is in none
    assert TG.corr(np.zeros(5), np.arange(5.0)) is None


# --------------------------------------------------------------------------------------------- controls, unit level
def good_t1(n=20):
    clips = ["c%02d" % i for i in range(n)]
    return {"clips": clips, "stage_rows": {c: {"status": "ok", "width": 1920, "height": 1080, "crc_ok": True} for c in clips}, "manifest": {c: {"n_frames": 100} for c in clips},
            "mask": {c: (100, 100) for c in clips}}


def test_t1_branches():
    ex = {"rows": 0, "clips": 20}
    assert TG.ctl_t1(good_t1(), ex)["pass"]
    f = good_t1()
    del f["stage_rows"]["c00"]                                                    # one clip without a tele clip is 5 per cent: allowed
    assert TG.ctl_t1(f, ex)["pass"]
    f = good_t1()
    for c in ("c00", "c01"):
        del f["stage_rows"][c]
    assert not TG.ctl_t1(f, ex)["pass"]
    f = good_t1()
    f["mask"]["c03"] = (100, 94)                                                  # 0.94 of the frames: counts as bad, one is allowed
    assert TG.ctl_t1(f, ex)["pass"]
    f["mask"]["c04"] = (100, 94)
    assert not TG.ctl_t1(f, ex)["pass"]
    f = good_t1()
    f["mask"]["c03"] = (100, 95)
    assert TG.ctl_t1(f, ex)["pass"]                                               # exactly 0.95 is enough
    f = good_t1()
    f["stage_rows"]["c05"]["height"] = 720
    assert not TG.ctl_t1(f, ex)["pass"]
    f = good_t1()
    f["stage_rows"]["c05"]["crc_ok"] = False
    assert not TG.ctl_t1(f, ex)["pass"]
    f = good_t1()
    f["manifest"]["c06"]["n_frames"] = 99                                         # the encode and the match file disagree
    f["manifest"]["c07"]["n_frames"] = 99
    assert not TG.ctl_t1(f, ex)["pass"]
    assert not TG.ctl_t1(good_t1(19), ex)["pass"]                                 # wrong number of pilot clips


def test_t2_branches():
    ok = {"pass": True, "share": 0.9, "n_frames": 200, "threshold": 0.9}
    assert TG.ctl_t2(ok)["pass"]
    assert not TG.ctl_t2(dict(ok, share=0.89))["pass"]
    assert not TG.ctl_t2(dict(ok, **{"pass": False}))["pass"]
    assert not TG.ctl_t2(dict(ok, threshold=0.5))["pass"]
    assert not TG.ctl_t2(dict(ok, n_frames=0))["pass"]
    assert not TG.ctl_t2(None)["pass"]


def test_t3_branches():
    P3 = {"n_rows": 3841, "E": dict(TG.REF_D)}
    assert TG.ctl_t3(P3)["pass"]
    assert TG.ctl_t3(dict(P3, E={"D_0": 3.9605, "D_1": 3.9135, "D_2": 4.1125}))["pass"]
    assert not TG.ctl_t3(dict(P3, E={"D_0": 3.962, "D_1": 3.913, "D_2": 4.112}))["pass"]
    assert not TG.ctl_t3(dict(P3, n_rows=3840))["pass"]


def good_rec():
    return {"arm": "D", "seed": 0, "n_win": 2, "n_tok": 130, "crop_fit_sha256": "aa", "changed_module_values": ["N_WIN", "N_TOK", "the token store", "sample builder"]}


def test_t4_branches():
    lo, hi = min(TG.BANKED_FIXED) - 0.15, max(TG.BANKED_FIXED) + 0.15
    P4 = {"n_rows": 3841, "n_finite": 3841, "E": 4.0}
    assert TG.ctl_t4(P4, good_rec())["pass"]
    assert TG.ctl_t4(dict(P4, E=lo + 1e-6), good_rec())["pass"]
    assert TG.ctl_t4(dict(P4, E=hi - 1e-6), good_rec())["pass"]
    assert not TG.ctl_t4(dict(P4, E=lo - 0.01), good_rec())["pass"]
    assert not TG.ctl_t4(dict(P4, E=hi + 0.01), good_rec())["pass"]
    assert not TG.ctl_t4(dict(P4, n_finite=3000), good_rec())["pass"]
    assert not TG.ctl_t4(P4, dict(good_rec(), changed_module_values=["N_WIN", "lr"]))["pass"]
    assert not TG.ctl_t4(P4, dict(good_rec(), seed=1))["pass"]
    assert not TG.ctl_t4(P4, dict(good_rec(), n_win=4))["pass"]
    assert not TG.ctl_t4(P4, None)["pass"]
    assert TG.ctl_t4(P4, dict(good_rec(), changed_module_values=[{"name": "N_TOK"}]))["pass"]


def test_t5_branches():
    base = {"n_rows": 100, "n_t3_rows": 100, "const": 30.0, "const_source": "x"}
    assert TG.ctl_t5(dict(base, E_fp=9.5, E_const=10.0))["pass"]                  # exactly 0.95 passes
    assert not TG.ctl_t5(dict(base, E_fp=9.4, E_const=10.0))["pass"]
    assert not TG.ctl_t5(dict(base, E_fp=11.0, E_const=10.0, n_rows=90))["pass"]
    assert not TG.ctl_t5(None)["pass"]


def test_t5_constant_sources():
    J, dumps, bx = TG.syn_world()
    L = {k: RL.lay(LK, J, d) for k, d in dumps.items()}
    for k, d in dumps.items():
        L[k]["tele_dropped"] = LK.on_rows(J, d, "tele_dropped").astype(bool) if "tele_dropped" in d else None
    box = {"has": np.ones(len(bx["az"]), bool), "az": bx["az"], "el": bx["el"]}
    P = TG.compute_values(LK, RL, J, L, box, 100)
    assert "median label gap" in P["t5"]["const_source"]
    lab = dumps["D_0"]["lab_gap"][dumps["D_0"]["lab_present"]]
    assert P["t5"]["const"] == pytest.approx(float(np.median(lab)))
    L["FP_0"]["const_gap"] = 33.0
    P = TG.compute_values(LK, RL, J, L, box, 100)
    assert P["t5"]["const"] == 33.0 and "const_gap" in P["t5"]["const_source"]


def test_t6_branches():
    t = {"text": "....\n18 passed in 1.0s\n", "mtime": 100.0, "before": 200.0}
    f = {"tests": {"enc": dict(t), "fit": dict(t)}, "n_staged": 20, "n_tok_files": 20}
    assert TG.ctl_t6(f)["pass"]
    assert not TG.ctl_t6(dict(f, tests={"enc": dict(t), "fit": dict(t, text="1 failed, 17 passed in 1s\n")}))["pass"]
    assert not TG.ctl_t6(dict(f, tests={"enc": dict(t), "fit": dict(t, text="18 passed, 1 error in 1s\n")}))["pass"]
    assert not TG.ctl_t6(dict(f, tests={"enc": dict(t), "fit": dict(t, text="all fine\n")}))["pass"]
    assert not TG.ctl_t6(dict(f, tests={"enc": dict(t), "fit": dict(t, mtime=300.0)}))["pass"]        # written after the start epoch
    assert not TG.ctl_t6(dict(f, tests={"enc": dict(t), "fit": dict(t, text=None)}))["pass"]
    assert not TG.ctl_t6(dict(f, tests={"enc": dict(t)}))["pass"]
    assert not TG.ctl_t6(dict(f, n_tok_files=19))["pass"]
    assert not TG.ctl_t6(dict(f, n_staged=0, n_tok_files=0))["pass"]


def test_t0_branches():
    h = "a" * 64
    rec = {"train_list_sha256": h, "eval_list_sha256": "b" * 64}
    assert TG.ctl_t0({"record": rec, "files": {"train_list": h, "eval_list": "b" * 64}})["pass"]
    assert TG.ctl_t0({"record": {"files": {"train_list": h, "eval_list": "b" * 64}}, "files": {"train_list": h, "eval_list": "b" * 64}})["pass"]
    assert not TG.ctl_t0({"record": rec, "files": {"train_list": h, "eval_list": "c" * 64}})["pass"]
    assert not TG.ctl_t0({"record": rec, "files": {"train_list": h}})["pass"]
    assert not TG.ctl_t0({"record": rec, "files": {"train_list": h, "eval_list": "b" * 64, "frames_list": h}})["pass"]
    assert not TG.ctl_t0({"record": None, "files": {}})["pass"]


# --------------------------------------------------------------------------------------------- controls on a tree
def test_clean_tree_passes_every_control(tmp_path):
    res = TG.run_controls(TG.write_tree(LK, RL, str(tmp_path / "clean")), LK, RL)
    assert res["all_pass"] is True and "payload" in res and "withheld" not in res, {k: v["failures"] for k, v in res["controls"].items() if not v["pass"]}
    assert sorted(res["controls"]) == sorted(TG.CONTROLS)


@pytest.mark.parametrize("fault", sorted(TG.CTL_FAULTS))
def test_each_planted_fault_flips_exactly_its_control(tmp_path, fault):
    res = TG.run_controls(TG.write_tree(LK, RL, str(tmp_path / fault), fault=fault), LK, RL)
    flipped = sorted(c for c, v in res["controls"].items() if not v["pass"])
    assert flipped == [fault], res["controls"][flipped[0]]["failures"] if flipped else "nothing flipped"
    assert res["all_pass"] is False and "withheld" in res and "payload" not in res


def test_missing_tele_dropped_fails_the_dump_controls(tmp_path):
    ns = TG.write_tree(LK, RL, str(tmp_path / "nodrop"))
    z = dict(np.load(ns.dump_paths["F_2"]))
    z.pop("tele_dropped")
    np.savez(ns.dump_paths["F_2"], **z)
    res = TG.run_controls(ns, LK, RL)
    assert not res["controls"]["t3"]["pass"] and "tele_dropped" in res["controls"]["t3"]["failures"][0]
    assert not res["all_pass"]


def test_missing_dump_fails_not_crashes(tmp_path):
    ns = TG.write_tree(LK, RL, str(tmp_path / "nodump"))
    ns.dump = [x for x in ns.dump if not x.startswith("FP_0=")]
    res = TG.run_controls(ns, LK, RL)
    assert not res["all_pass"] and not res["controls"]["t5"]["pass"]


# --------------------------------------------------------------------------------------------- the stages through main()
def argv_of(ns, extra):
    a = ["--lookgap-reader", TG.LOOKGAP_DEFAULT, "--eval-join", ns.eval_join, "--box-dir", ns.box_dir, "--list-record", ns.list_record, "--enc-manifest", ns.enc_manifest, "--all-clips", ns.all_clips,
         "--match-dir", ns.match_dir, "--t2-json", ns.t2_json, "--d2-fit-record", ns.d2_fit_record, "--n-tok-files", str(ns.n_tok_files), "--n-boot", "300",
         "--expect-rows", str(ns.expect["rows"]), "--expect-clips", "20", "--ref-d", ",".join("%s=%s" % kv for kv in ns.ref_d.items()), "--banked", ",".join(str(x) for x in ns.banked)]
    for flag, vals in (("--list-file", ns.list_file), ("--stage-record", ns.stage_record), ("--test-out", ns.test_out), ("--test-before", ns.test_before), ("--dump", ns.dump)):
        for v in vals:
            a += [flag, v]
    return a + extra


def test_controls_stage_prints_only_t_lines_and_names_stage_runs(tmp_path, capsys):
    ns = TG.write_tree(LK, RL, str(tmp_path / "cli"))
    cj = str(tmp_path / "cli" / "CONTROLS.json")
    assert TG.main(argv_of(ns, ["--stage", "controls", "--out", cj])) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[:7] == ["t%d PASS" % i for i in range(7)] and out[-1] == "ZZ READ-TELEGAP controls all_pass True" and len(out) == 8
    assert TG.main(argv_of(ns, ["--stage", "controls", "--out", cj])) == 2                  # never reads twice into one file
    capsys.readouterr()
    res = str(tmp_path / "cli" / "RESULT.json")
    assert TG.main(["--lookgap-reader", TG.LOOKGAP_DEFAULT, "--stage", "names", "--controls", cj, "--out", res]) == 0
    lines = capsys.readouterr().out.splitlines()
    text = "\n".join(lines)
    assert "name: " + PRE + "-F-CUTS-GAP-ERROR" in text and "bar met: yes" in text and "FAR-LEADS-READ" in text
    assert lines[-1] == "ZZ READ-TELEGAP names rc 0"
    assert os.path.isfile(res) and os.path.isfile(os.path.splitext(res)[0] + "_names.log")
    assert json.load(io.open(res, encoding="utf-8"))["second_derivation_agrees"] is True
    assert TG.main(["--lookgap-reader", TG.LOOKGAP_DEFAULT, "--stage", "names", "--controls", cj, "--out", res]) == 2


@pytest.mark.parametrize("fault", ["t2", "t5"])
def test_names_stage_refuses_after_a_failed_control(tmp_path, capsys, fault):
    ns = TG.write_tree(LK, RL, str(tmp_path / "fail"), fault=fault)
    cj = str(tmp_path / "fail" / "CONTROLS.json")
    TG.main(argv_of(ns, ["--stage", "controls", "--out", cj]))
    out = capsys.readouterr().out.splitlines()
    assert "%s FAIL" % fault in out and out[-1] == "ZZ READ-TELEGAP controls all_pass False"
    assert "payload" not in json.load(io.open(cj, encoding="utf-8")) and "withheld" in json.load(io.open(cj, encoding="utf-8"))
    res = str(tmp_path / "fail" / "RESULT.json")
    assert TG.main(["--lookgap-reader", TG.LOOKGAP_DEFAULT, "--stage", "names", "--controls", cj, "--out", res]) == 3
    text = capsys.readouterr().out
    assert "REFUSED" in text and "ratio" not in text and "TELEGAP-T0-F-" not in text
    assert not os.path.exists(res)


def test_names_stage_refuses_a_hand_edited_controls_file(tmp_path, capsys):
    ns = TG.write_tree(LK, RL, str(tmp_path / "edit"), fault="t6")
    cj = str(tmp_path / "edit" / "CONTROLS.json")
    TG.main(argv_of(ns, ["--stage", "controls", "--out", cj]))
    capsys.readouterr()
    d = json.load(io.open(cj, encoding="utf-8"))
    d["all_pass"] = True
    d["payload"] = d["withheld"]
    json.dump(d, io.open(cj, "w", encoding="utf-8"))
    assert TG.main(["--lookgap-reader", TG.LOOKGAP_DEFAULT, "--stage", "names", "--controls", cj, "--out", str(tmp_path / "edit" / "R.json")]) == 3


def test_no_interval_names_stage_rc5(tmp_path, capsys):
    P = payload(zero=True)
    cj = str(tmp_path / "ni.json")
    ctl = {k: {"pass": True, "failures": []} for k in TG.CONTROLS}
    json.dump({"card": TG.CARD, "controls": ctl, "all_pass": True, "payload": P}, io.open(cj, "w", encoding="utf-8"), default=float)
    assert TG.main(["--lookgap-reader", TG.LOOKGAP_DEFAULT, "--stage", "names", "--controls", cj, "--out", str(tmp_path / "ni_res.json")]) == 5


def test_selftest_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert TG.selftest(LK, RL) == 0
    assert "ZZ READ-TELEGAP SELFTEST OK" in capsys.readouterr().out


def test_shell_scripts_have_the_declared_markers():
    s = io.open(os.path.join(HERE, "podread_telegap.sh"), encoding="utf-8").read()
    for token in ("TELEGAP_FIT_DONE.txt", "ZZ PODREAD-TELEGAP", "TELEGAP_CONTROLS.json", "ALREADY-READ", "sed '/hf"):
        assert token in s or token.replace("sed '/hf", "") in s
    p = io.open(os.path.join(HERE, "pull_telegap.sh"), encoding="utf-8").read()
    for token in ("pod_add122", "tmp_pull_add122", "sed '/hf[_]/d'", "--exclude=\\\"*.npz\\\""):
        assert token in p


def test_shell_syntax():
    import shutil
    import subprocess
    sh = shutil.which("bash")
    if not sh:
        pytest.skip("no bash")
    for f in ("podread_telegap.sh", "pull_telegap.sh"):
        r = subprocess.run([sh, "-n", os.path.join(HERE, f).replace("\\", "/")], capture_output=True, text=True, encoding="utf-8", errors="replace")
        assert r.returncode == 0, r.stderr


def test_t4_takes_the_record_key_and_the_five_names_of_the_fit_tool():
    rec = {"arm": "D2", "seed": 0, "n_win": 2, "n_tok": 130, "crop_fit_sha256": "ab" * 32,
           "module_values_changed": [{"name": "ARMS", "before": "x", "after": "y"}, {"name": "TokenStore"}, {"name": "build_samples"}]}
    P4 = {"E": 4.0, "n_finite": 10, "n_rows": 10}
    assert TG.ctl_t4(P4, rec)["pass"] is True
    rec2 = dict(rec, module_values_changed=rec["module_values_changed"] + [{"name": "CARD_HPARAMS"}])
    r = TG.ctl_t4(P4, rec2)
    assert r["pass"] is False and any("CARD_HPARAMS" in x for x in r["failures"])


def test_t5_row_clause_of_note_n3_marked_rows_pass_unmarked_rows_fail():
    base = {"n_rows": 96, "n_t3_rows": 100, "n_marked_dropped": 4, "n_unmarked_not_finite": 0, "const": 30.0, "const_source": "x", "E_fp": 9.9, "E_const": 10.0}
    assert TG.ctl_t5(base)["pass"] is True
    r = TG.ctl_t5(dict(base, n_rows=95, n_unmarked_not_finite=1))
    assert r["pass"] is False and any("neither marked nor finite" in x for x in r["failures"])
    old = {"n_rows": 99, "n_t3_rows": 100, "const": 30.0, "const_source": "x", "E_fp": 9.9, "E_const": 10.0}
    assert TG.ctl_t5(old)["pass"] is False
