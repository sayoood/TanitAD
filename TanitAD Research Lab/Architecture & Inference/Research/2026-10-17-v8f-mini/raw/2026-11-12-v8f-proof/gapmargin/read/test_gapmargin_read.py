"""test_gapmargin_read.py -- card ADD117 (GAPMARGIN10K), reader kit (lead-written): the generated reader and driver against their generator, the tables, the names on
hand-made payloads (every branch of the rule of section 4), the first control on a record made by the gap-file tool, the rule-log control, the identity control, the
argument rule on lines in the real format, and the refusals. The source kit of card ADD114 is taken from ../../presattr/read and the gap-file tool from ../queue
(same relative places in the pod stage). No pod result file is opened."""
import argparse
import io
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "make_gapmargin_reader.py")
READER = os.path.join(HERE, "read_gapmargin10k.py")
DRIVER = os.path.join(HERE, "podread_gapmargin10k.sh")
BASH = shutil.which("bash")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "queue"))
import make_margin_gapfiles as MG  # noqa: E402
import read_gapmargin10k as RG  # noqa: E402

CKPT = "2fd4b747532f55b24523c578948cb713fb80fa0e698586c329dd23ba6e6df914"
LINE = ("arm %s form RR replan steps 10 rule every %s readout %s kappa 0.2 preroll 0 exec plan nav deadreckon grid eval_grid_windows.json model seed 0 reader seed 0 "
        "ckpt sha256 " + CKPT + " wrapper sha256 " + "28" * 32)
P = "GAPMARGIN10K-"


def rd(p):
    return io.open(p, encoding="utf-8", newline="").read()


def payload(ivs=None, u=None):
    """ivs {(S, pair, measure): (lo, hi)}; u {(S, arm): pooled unsafe level}; default: nothing separated, every arm at 0.05, the label form at 0.0365"""
    ivs, u = ivs or {}, u or {}
    st = {}
    for S in ("K0", "CONV"):
        pairs = {pn: {mk: {"interval": list(ivs.get((S, pn, mk), (-0.01, 0.01)))} for mk in "UCF"} for pn in RG.PAIRS}
        levels = {arm: {"C": 0.0, "U": u.get((S, arm), 0.0365 if arm == "RLc" else 0.05), "F": 0.2} for arm in RG.ARMS}
        st[S] = {"pairs": pairs, "levels": levels}
    return {"settings": st}


@pytest.mark.parametrize("kind,path", [("reader", READER), ("driver", DRIVER)])
def test_generated_file_is_what_the_generator_prints(kind, path):
    r = subprocess.run([sys.executable, GEN, "--print", kind], capture_output=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.replace("\r\n", "\n") == rd(path)
    assert "\r" not in rd(path)


def test_tables():
    assert RG.CARD == "PREREG_V8F_PROOF_ADD117" and RG.PRE == "GAPMARGIN10K" and RG.SETTINGS == ("K0", "CONV") and RG.MARGIN_ARMS == ("A90", "A80")
    assert RG.SPEC["K0"]["runs"] == {"A90": "RQ50KF5Hb20gA90", "A80": "RQ50KF5Hb20gA80", "GEc": "RQ50KF5Hb20gGEc", "RLc": "RLcKF5Hb20", "RR": "RQ50KF5Hb20"}
    assert RG.SPEC["CONV"]["runs"] == {"A90": "RQ50Hb20gA90", "A80": "RQ50Hb20gA80", "GEc": "RQ50Hb20gGEc", "RLc": "RLcHb20", "RR": "RQ50Hb20"}
    assert RG.SPEC["K0"]["every"] == 5 and RG.SPEC["K0"]["readout"] == "K" and RG.SPEC["CONV"]["every"] is None and RG.SPEC["CONV"]["readout"] == "pinned"
    assert all(RG.SPEC[S]["P"] == 0 and RG.SPEC[S]["grid"] == "full" and RG.SPEC[S]["grid_file"] == "pinned" for S in RG.SETTINGS)
    assert sorted(RG.new_keys()) == sorted("%s_%s_s%d" % (S, a, m) for S in ("K0", "CONV") for a in ("A90", "A80") for m in (0, 1))
    assert RG.PAIRS == {"A90-VS-GEc": ("A90", "GEc"), "A80-VS-GEc": ("A80", "GEc"), "A90-VS-RLc": ("A90", "RLc"), "A80-VS-RLc": ("A80", "RLc"), "GEc-VS-RLc": ("GEc", "RLc")}
    assert RG.EXPECTED["K0"] == {"GEc": (0.0000, 0.0498, 0.1651), "RR": (0.0296, 0.0738, 0.1682), "RLc": (0.0000, 0.0365, 0.1550)}
    assert RG.EXPECTED["CONV"] == {"GEc": (0.0016, 0.0569, 0.2329), "RR": (0.0280, 0.0783, 0.2593), "RLc": (0.0016, 0.0320, 0.2243)}
    assert RG.ref_arm("K0", "A90") == "GEc" and RG.ref_arm("CONV", "A80") == "GEc" and RG.CLOSE_TOL == 0.0045
    assert RG.DONE_RE.match("ZZ GAPMARGIN10K DONE 2026-11-22T16:00:00Z all steps done") and RG.CONTROLS == tuple("e%d" % i for i in range(8))


def test_names_when_nothing_is_separated():
    nm = RG.all_names(payload())
    assert len(nm) == 18
    for S in ("K0", "CONV"):
        assert nm[S + "_OVERALL"] == P + S + "-MARGIN-DOES-NOT-CLOSE-UNSAFE"
        for a in ("A90", "A80"):
            assert nm["%s_%s_LABEL" % (S, a)] == P + "%s-%s-UNSAFE-AT-LABEL-FORM" % (S, a)
            for mw in ("UNSAFE", "CONTACT", "FALLING-BEHIND"):
                assert nm["%s_%s_%s" % (S, a, mw)] == P + "%s-%s-%s-NOT-SEPARATED" % (S, a, mw)


def test_names_close_at_no_separated_price():
    pl = payload({("K0", "A80-VS-GEc", "U"): (-0.03, -0.002)}, {("K0", "A80"): 0.0400})
    nm = RG.all_names(pl)
    assert nm["K0_OVERALL"] == P + "K0-MARGIN-CLOSES-UNSAFE-AT-NO-SEPARATED-PRICE" and nm["K0_A80_UNSAFE"] == P + "K0-A80-LOWERS-UNSAFE"
    assert nm["CONV_OVERALL"] == P + "CONV-MARGIN-DOES-NOT-CLOSE-UNSAFE"
    assert RG.closing_arm(pl, "K0") == "A80" and RG.closing_arm(pl, "CONV") == "none"


def test_names_close_at_a_price_and_not_near_enough():
    iv = {("K0", "A80-VS-GEc", "U"): (-0.03, -0.002), ("K0", "A80-VS-GEc", "F"): (0.004, 0.03)}
    pl = payload(iv, {("K0", "A80"): 0.0400})
    nm = RG.all_names(pl)
    assert nm["K0_OVERALL"] == P + "K0-MARGIN-CLOSES-UNSAFE-AT-A-PRICE" and nm["K0_A80_FALLING-BEHIND"] == P + "K0-A80-RAISES-FALLING-BEHIND" and RG.closing_arm(pl, "K0") == "A80"
    far = payload(iv, {("K0", "A80"): 0.0411})                                      # lowers the unsafe rate but stays more than 0.0045 above the label form
    assert RG.all_names(far)["K0_OVERALL"] == P + "K0-MARGIN-DOES-NOT-CLOSE-UNSAFE" and RG.closing_arm(far, "K0") == "none"
    edge = payload(iv, {("K0", "A80"): 0.0365 + 0.0045})
    assert RG.all_names(edge)["K0_OVERALL"] == P + "K0-MARGIN-CLOSES-UNSAFE-AT-A-PRICE"
    not_lower = payload({("K0", "A80-VS-GEc", "U"): (-0.03, 0.0)}, {("K0", "A80"): 0.0300})  # near the label form but the interval reaches 0
    assert RG.all_names(not_lower)["K0_OVERALL"] == P + "K0-MARGIN-DOES-NOT-CLOSE-UNSAFE"


def test_names_two_closing_arms_and_the_named_arm():
    base = {("CONV", "A90-VS-GEc", "U"): (-0.02, -0.001), ("CONV", "A80-VS-GEc", "U"): (-0.03, -0.002)}
    lv = {("CONV", "A90"): 0.0380, ("CONV", "A80"): 0.0300}
    both_free = payload(base, lv)
    assert RG.all_names(both_free)["CONV_OVERALL"] == P + "CONV-MARGIN-CLOSES-UNSAFE-AT-NO-SEPARATED-PRICE" and RG.closing_arm(both_free, "CONV") == "A90"
    a90_pays = payload({**base, ("CONV", "A90-VS-GEc", "F"): (0.002, 0.02)}, lv)
    assert RG.all_names(a90_pays)["CONV_OVERALL"] == P + "CONV-MARGIN-CLOSES-UNSAFE-AT-NO-SEPARATED-PRICE" and RG.closing_arm(a90_pays, "CONV") == "A80"
    both_pay = payload({**base, ("CONV", "A90-VS-GEc", "F"): (0.002, 0.02), ("CONV", "A80-VS-GEc", "F"): (0.004, 0.03)}, lv)
    assert RG.all_names(both_pay)["CONV_OVERALL"] == P + "CONV-MARGIN-CLOSES-UNSAFE-AT-A-PRICE" and RG.closing_arm(both_pay, "CONV") == "A90"
    lowers_f = payload({**base, ("CONV", "A80-VS-GEc", "F"): (-0.02, -0.002), ("CONV", "A90-VS-GEc", "F"): (0.002, 0.02)}, lv)
    nm = RG.all_names(lowers_f)
    assert nm["CONV_OVERALL"] == P + "CONV-MARGIN-CLOSES-UNSAFE-AT-NO-SEPARATED-PRICE" and nm["CONV_A80_FALLING-BEHIND"] == P + "CONV-A80-LOWERS-FALLING-BEHIND"


def test_names_against_the_label_form_and_contact():
    pl = payload({("K0", "A90-VS-RLc", "U"): (0.002, 0.02), ("K0", "A80-VS-RLc", "U"): (-0.02, -0.002), ("CONV", "A90-VS-GEc", "C"): (0.001, 0.01), ("CONV", "A80-VS-GEc", "U"): (0.001, 0.01)})
    nm = RG.all_names(pl)
    assert nm["K0_A90_LABEL"] == P + "K0-A90-UNSAFE-ABOVE-LABEL-FORM" and nm["K0_A80_LABEL"] == P + "K0-A80-UNSAFE-BELOW-LABEL-FORM"
    assert nm["CONV_A90_CONTACT"] == P + "CONV-A90-RAISES-CONTACT" and nm["CONV_A80_UNSAFE"] == P + "CONV-A80-RAISES-UNSAFE"


def test_first_control_on_a_record_of_the_gap_file_tool(tmp_path):
    bp = str(tmp_path / "base.npz")
    np.savez(bp, **MG.synthetic_base())
    rec = MG.build(bp, str(tmp_path / "gf"), (0.90, 0.80))
    sb = RG.sha_file(bp)
    shas = {a: RG.sha_file(str(tmp_path / "gf" / ("VGAPLE_%s.npz" % a))) for a in ("A90", "A80")}
    r = RG.ctl_e0(rec, sb, shas)
    assert r["pass"] is False and len(r["failures"]) == 1 and "card ADD106" in r["failures"][0]          # a synthetic base is not the base file of the card
    assert set(r["values"]["n_rows_newly_in_range"]) == {"A90", "A80"} and r["values"]["n_rows_newly_in_range"]["A80"] > 0
    real = dict(rec, base=dict(rec["base"], sha256="55de2f7c" + "0" * 56))
    assert RG.ctl_e0(real, "55de2f7c" + "0" * 56, shas)["pass"] is True
    for tamper in (dict(real, pass_k0=False), dict(real, scale_one_gives_the_base_arrays=False), dict(real, files={}), dict(real, clauses={})):
        assert RG.ctl_e0(tamper, "55de2f7c" + "0" * 56, shas)["pass"] is False
    assert RG.ctl_e0(real, "55de2f7c" + "0" * 56, dict(shas, A80="1" * 64))["pass"] is False
    assert RG.ctl_e0(real, "55de2f7c" + "1" * 56, shas)["pass"] is False and RG.ctl_e0(None, None, None)["pass"] is False
    wrong_a = dict(real, files=dict(real["files"], VGAPLE_A90=dict(real["files"]["VGAPLE_A90"], a=0.85)))
    assert RG.ctl_e0(wrong_a, "55de2f7c" + "0" * 56, shas)["pass"] is False


def test_rule_log_control():
    good = {k: {"PASS": True, "failures": [], "share_of_rule_steps_with_a_lead": 0.3, "n_rows": 10, "n_lead_rows": 3} for k in RG.new_keys()}
    assert RG.ctl_e7(good)["pass"] is True and set(RG.ctl_e7(good)["values"]["rollout_shares"]) == set(RG.new_keys())
    one = dict(good)
    one["K0_A80_s1"] = {"PASS": False, "failures": ["12 lead rows differ"]}
    r = RG.ctl_e7(one)
    assert r["pass"] is False and "K0_A80_s1: 12 lead rows differ" in r["failures"]
    assert RG.ctl_e7({k: v for k, v in good.items() if k != "CONV_A90_s0"})["pass"] is False and RG.ctl_e7({})["pass"] is False


def test_rule_checks_without_the_scaled_files_fail_every_rollout(tmp_path):
    a = argparse.Namespace(margin_dir=str(tmp_path / "nothing"), pull=str(tmp_path))
    per = RG.rule_checks(a)
    assert sorted(per) == sorted(RG.new_keys()) and all(v["PASS"] is False and "no scaled gap file" in v["failures"][0] for v in per.values())
    assert RG.margin_file(a, "A90").replace("\\", "/").endswith("nothing/VGAPLE_A90.npz")


def test_identity_control_on_the_banked_levels():
    lv = {S: {arm: dict(zip("CUF", v)) for arm, v in RG.EXPECTED[S].items()} for S in RG.EXPECTED}
    assert RG.ctl_e3(lv)["pass"] is True
    lv["CONV"]["RR"]["F"] -= 0.0002
    assert RG.ctl_e3(lv)["pass"] is False and RG.ctl_e3({"K0": lv["K0"]})["pass"] is False


def test_argument_rule_on_lines_in_the_real_format():
    F = lambda new, ref: RG.d2_line_failures(RG.parse_run_line(new), RG.parse_run_line(ref))
    assert F(LINE % ("I10RQ50KF5Hb20gA90_s0", "5", "causal_g"), LINE % ("I10RQ50KF5Hb20gGEc_s0", "5", "causal_g")) == []
    assert F(LINE % ("I10RQ50Hb20gA80_s0", "10", "pinned"), LINE % ("I10RQ50Hb20gGEc_s0", "10", "pinned")) == []
    assert F(LINE % ("I10RQ50Hb20gA80_s0", "5", "pinned"), LINE % ("I10RQ50Hb20gGEc_s0", "10", "pinned"))
    assert F(LINE % ("I10RQ50Hb20gA80_s0", "10", "causal_g"), LINE % ("I10RQ50Hb20gGEc_s0", "10", "pinned"))
    assert F((LINE % ("I10RQ50Hb20gA80_s0", "10", "pinned")).replace("form RR", "form RLc"), LINE % ("I10RQ50Hb20gGEc_s0", "10", "pinned"))
    assert F((LINE % ("I10RQ50Hb20gA80_s0", "10", "pinned")).replace("model seed 0", "model seed 1"), LINE % ("I10RQ50Hb20gGEc_s0", "10", "pinned"))


def test_refusals():
    assert RG.main(["--selftest"]) == 2
    assert RG.main(["--stage", "names"]) == 2


def test_driver_text():
    t = rd(DRIVER)
    r = subprocess.run([BASH, "-n", DRIVER.replace("\\", "/")], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr
    assert "GE=$V/gapmargin; K=$GE/kit_read; OUT=$GE/read; LT=$GE/lead_tests; GFM=$GE/gapfiles; K114=$V/presattr/kit_read" in t
    assert '"$PY" "$K114/read_presattr10k.py" --selftest' in t and '"$PY" "$K/read_gapmargin10k.py" --stage controls' in t and '"$PY" "$K/read_gapmargin10k.py" --stage names' in t
    assert '--gap-file "$GAPF" --margin-dir "$GFM" --margin-record "$GFM/GAPMARGIN_GAPFILES_RECORD.json"' in t
    assert "for x in I10RQ50KF5Hb20gA90 I10RQ50KF5Hb20gA80 I10RQ50KF5Hb20gGEc I10RLcKF5Hb20 I10RQ50KF5Hb20 I10RQ50Hb20gA90 I10RQ50Hb20gA80 I10RQ50Hb20gGEc I10RLcHb20 I10RQ50Hb20 I10R0Hb20; do" in t
    assert '[ -f "$GE/GAPMARGIN10K_DONE.txt" ] || stop "QUEUE-NOT-DONE"' in t and '[ -e "$GE/GAPMARGIN10K_STOP.txt" ] && stop "QUEUE-STOPPED"' in t
    for bad in ("grid_p30", "PRESATTR10K", "JLE", "$PJ", "--la-file"):
        assert bad not in t
    code = "\n".join(x for x in t.splitlines() if not x.lstrip().startswith("#"))
    assert not re.search(r"(^|[;&|]\s*|\s)(rm|mv|rmdir)\s", code)


def test_wording_of_the_kit_files():
    parts = [("vo", "id"), ("super", "seded"), ("retr", "acted"), ("form", "erly"), ("correc", "tion"), ("with", "drawn"), ("no lon", "ger"), ("closed", " loop")]
    for fn in os.listdir(HERE):
        p = os.path.join(HERE, fn)
        if os.path.isfile(p) and fn.endswith((".py", ".sh", ".md")):
            low = rd(p).lower()
            for a, b in parts:
                assert (a + b) not in low, (fn, a + b)
