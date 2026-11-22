"""test_presattrc_read.py -- card ADD115 (PRESATTRC10K), reader kit (lead-written): the generated reader and driver against their generator, the tables of the one setting CONV,
the names on a small hand-made payload, the identity control on the banked levels of row 179, the argument rule of note N1 on lines in the real format of the convention,
the loader on a run with one rule log, and the name across settings. The source kit of card ADD114 is taken from ../../presattr/read (same relative place in the pod stage).
No pod result file is opened."""
import io
import itertools
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "make_presattrc_reader.py")
READER = os.path.join(HERE, "read_presattrc10k.py")
DRIVER = os.path.join(HERE, "podread_presattrc10k.sh")
BASH = shutil.which("bash")
sys.path.insert(0, HERE)
import across_name as AN  # noqa: E402
import read_presattrc10k as RC  # noqa: E402

CKPT = "2fd4b747532f55b24523c578948cb713fb80fa0e698586c329dd23ba6e6df914"
REF_RLC = "arm I10RLcHb20_s0 form RLc replan steps 10 rule every 10 readout pinned kappa 0.2 model seed 0 reader seed 0 ckpt sha256 %s wrapper sha256 %s" % (CKPT, "37" * 32)
NEW_PL = ("arm I10RLcHb20pLE_s0 form RLc replan steps 10 rule every 10 readout pinned kappa 0.2 preroll 0 exec plan nav deadreckon grid eval_grid_windows.json model seed 0 reader seed 0 "
          "ckpt sha256 %s wrapper sha256 %s" % (CKPT, "28" * 32))
CONTRASTS = ("PRESENCE-AT-LABEL-GAP", "PRESENCE-AT-CAMERA-GAP", "GAP-AT-LABEL-PRESENCE", "GAP-AT-CAMERA-PRESENCE")
PAIR_OF = {"PRESENCE-AT-LABEL-GAP": "LA-VS-RLc", "PRESENCE-AT-CAMERA-GAP": "GEc-VS-PL", "GAP-AT-LABEL-PRESENCE": "PL-VS-RLc", "GAP-AT-CAMERA-PRESENCE": "GEc-VS-LA"}


def rd(p):
    return io.open(p, encoding="utf-8", newline="").read()


def payload(ivs=None):
    ivs = ivs or {}
    pairs = {pn: {mk: {"interval": list(ivs.get((pn, mk), (-0.01, 0.01)))} for mk in "UCF"} for pn in list(PAIR_OF.values()) + ["GEc-VS-RLc"]}
    return {"settings": {"CONV": {"pairs": pairs}}}


@pytest.mark.parametrize("kind,path", [("reader", READER), ("driver", DRIVER)])
def test_generated_file_is_what_the_generator_prints(kind, path):
    r = subprocess.run([sys.executable, GEN, "--print", kind], capture_output=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.replace("\r\n", "\n") == rd(path)
    assert "\r" not in rd(path)


def test_tables_of_the_one_setting():
    assert RC.CARD == "PREREG_V8F_PROOF_ADD115" and RC.PRE == "PRESATTRC10K" and RC.SETTINGS == ("CONV",)
    sp = RC.SPEC["CONV"]
    assert sp["P"] == 0 and sp["readout"] == "pinned" and sp["every"] is None and sp["grid"] == "full" and sp["grid_file"] == "pinned"
    assert sp["runs"] == {"RLc": "RLcHb20", "LA": "RQ50Hb20aLA", "PL": "RLcHb20pLE", "GEc": "RQ50Hb20gGEc", "RR": "RQ50Hb20"}
    assert RC.NEW_ARMS == {"CONV": ("LA", "PL")}
    assert sorted(RC.new_keys()) == ["CONV_LA_s0", "CONV_LA_s1", "CONV_PL_s0", "CONV_PL_s1"]
    assert RC.EXPECTED == {"CONV": {"GEc": (0.0016, 0.0569, 0.2329), "RR": (0.0280, 0.0783, 0.2593), "RLc": (0.0016, 0.0320, 0.2243)}}
    assert RC.ref_arm("CONV", "LA") == "RR" and RC.ref_arm("CONV", "PL") == "RLc"
    assert RC.run_name("CONV", "PL") == "RLcHb20pLE" and RC.CONTROLS == tuple("e%d" % i for i in range(8))
    assert RC.DONE_RE.match("ZZ PRESATTRC10K DONE 2026-11-22T15:00:00Z all steps done") and not RC.DONE_RE.match("ZZ PRESATTR10K DONE 2026-11-22T15:00:00Z all steps done")


def test_names_not_separated_everywhere():
    nm = RC.all_names(payload())
    assert len(nm) == 13 and nm["CONV_OVERALL"] == "PRESATTRC10K-CONV-UNSAFE-NOT-ATTRIBUTED"
    for cn in CONTRASTS:
        for mw in ("UNSAFE", "CONTACT", "FALLING-BEHIND"):
            assert nm["CONV_%s_%s" % (cn, mw)] == "PRESATTRC10K-CONV-%s-%s-NOT-SEPARATED" % (cn, mw)


@pytest.mark.parametrize("raised", [c for r in range(1, 5) for c in itertools.combinations(CONTRASTS, r)])
def test_overall_name_for_every_set_of_raising_contrasts(raised):
    nm = RC.all_names(payload({(PAIR_OF[c], "U"): (0.004, 0.03) for c in raised}))
    pres = any(c.startswith("PRESENCE") for c in raised)
    gap = any(c.startswith("GAP") for c in raised)
    want = "BOTH-CARRY-UNSAFE" if pres and gap else ("PRESENCE-CARRIES-UNSAFE" if pres else "GAP-CARRIES-UNSAFE")
    assert nm["CONV_OVERALL"] == "PRESATTRC10K-CONV-" + want
    for c in CONTRASTS:
        assert nm["CONV_%s_UNSAFE" % c] == "PRESATTRC10K-CONV-%s-%s" % (c, "RAISES-UNSAFE" if c in raised else "UNSAFE-NOT-SEPARATED")


def test_a_lowering_contrast_does_not_carry_and_an_end_at_zero_is_not_separated():
    nm = RC.all_names(payload({("LA-VS-RLc", "U"): (-0.03, -0.004), ("GEc-VS-PL", "U"): (0.0, 0.02), ("PL-VS-RLc", "C"): (0.001, 0.02), ("GEc-VS-LA", "F"): (-0.02, -0.001)}))
    assert nm["CONV_PRESENCE-AT-LABEL-GAP_UNSAFE"] == "PRESATTRC10K-CONV-PRESENCE-AT-LABEL-GAP-LOWERS-UNSAFE"
    assert nm["CONV_PRESENCE-AT-CAMERA-GAP_UNSAFE"] == "PRESATTRC10K-CONV-PRESENCE-AT-CAMERA-GAP-UNSAFE-NOT-SEPARATED"
    assert nm["CONV_GAP-AT-LABEL-PRESENCE_CONTACT"] == "PRESATTRC10K-CONV-GAP-AT-LABEL-PRESENCE-RAISES-CONTACT"
    assert nm["CONV_GAP-AT-CAMERA-PRESENCE_FALLING-BEHIND"] == "PRESATTRC10K-CONV-GAP-AT-CAMERA-PRESENCE-LOWERS-FALLING-BEHIND"
    assert nm["CONV_OVERALL"] == "PRESATTRC10K-CONV-UNSAFE-NOT-ATTRIBUTED"


def test_identity_control_on_the_banked_levels():
    lv = {"CONV": {arm: dict(zip("CUF", v)) for arm, v in RC.EXPECTED["CONV"].items()}}
    assert RC.ctl_e3(lv)["pass"] is True
    lv["CONV"]["GEc"]["U"] += 0.0002
    r = RC.ctl_e3(lv)
    assert r["pass"] is False and "CONV GEc U" in r["failures"][0]
    assert RC.ctl_e3({"CONV": {}})["pass"] is False and RC.ctl_e3({})["pass"] is False


def test_argument_rule_on_lines_in_the_format_of_the_convention():
    P = RC.parse_run_line
    assert RC.d2_line_failures(P(NEW_PL), P(REF_RLC)) == []
    assert RC.d2_line_failures(P(NEW_PL.replace("rule every 10", "rule every 5")), P(REF_RLC))
    assert RC.d2_line_failures(P(NEW_PL.replace("readout pinned", "readout causal_g")), P(REF_RLC))
    assert RC.d2_line_failures(P(NEW_PL.replace("preroll 0", "preroll 30")), P(REF_RLC))
    assert RC.d2_line_failures(P(NEW_PL.replace("grid eval_grid_windows.json", "grid grid_p30.json")), P(REF_RLC))
    assert RC.d2_line_failures(P(NEW_PL.replace("model seed 0", "model seed 1")), P(REF_RLC))
    assert RC.d2_line_failures(P(NEW_PL.replace(CKPT, "ab" * 32)), P(REF_RLC))
    assert RC.d2_line_failures(P(NEW_PL.replace(" preroll 0 exec plan nav deadreckon grid eval_grid_windows.json", "")), P(REF_RLC))


def test_loader_takes_the_one_rule_log_of_a_run_of_the_convention(tmp_path):
    d = tmp_path / "I10RLcHb20pLE_s0" / "ticks"
    d.mkdir(parents=True)
    with pytest.raises(SystemExit):
        RC.load_logs_setting(str(tmp_path), "RLcHb20pLE", 0, "CONV")
    np.savez(str(d / "rule_log_I10RLcHb20pLE_s0.npz"), ci=np.array([0, 0, 1]), j=np.array([2, 3, 0]), present=np.array([1, 0, 1], bool), g_src=np.array([10.0, np.nan, 20.0]))
    logs = RC.load_logs_setting(str(tmp_path), "RLcHb20pLE", 0, "CONV")
    assert len(logs) == 1
    JL = {"off": np.array([0, 5]), "n": np.array([5, 4]), "lead_valid": np.array([0, 0, 1, 0, 0, 1, 0, 0, 0]), "lead_gap": np.array([np.nan, np.nan, 10.0, np.nan, np.nan, 20.0004, np.nan, np.nan, np.nan])}
    r = RC.pl_check(RC.rows4(logs), JL)
    assert r["PASS"] is True and r["n_lead_rows"] == 2 and r["n_rows"] == 3
    JL["lead_gap"][5] = 20.01
    assert RC.pl_check(RC.rows4(logs), JL)["PASS"] is False
    JL["lead_gap"][5] = 61.0
    assert RC.pl_check(RC.rows4(logs), JL)["n_lead_not_in_join"] == 1


def test_generated_reader_refuses_its_selftest():
    assert RC.main(["--selftest"]) == 2


def test_driver_text():
    t = rd(DRIVER)
    r = subprocess.run([BASH, "-n", DRIVER.replace("\\", "/")], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr
    assert "GE=$V/presattrc; K=$GE/kit_read; OUT=$GE/read; LT=$GE/lead_tests; PJ=$V/presattr/joins; K114=$V/presattr/kit_read" in t
    assert '"$PY" "$K114/read_presattr10k.py" --selftest' in t and '"$PY" "$K/read_presattrc10k.py" --stage controls' in t and '"$PY" "$K/read_presattrc10k.py" --stage names' in t
    assert "for x in I10RLcHb20 I10RQ50Hb20aLA I10RLcHb20pLE I10RQ50Hb20gGEc I10RQ50Hb20 I10R0Hb20; do" in t
    assert '[ -f "$GE/PRESATTRC10K_DONE.txt" ] || stop "QUEUE-NOT-DONE"' in t and '[ -e "$GE/PRESATTRC10K_STOP.txt" ] && stop "QUEUE-STOPPED"' in t
    for bad in ("KF5", "grid_p30", "PRESATTR10K"):
        assert bad not in t
    code = "\n".join(x for x in t.splitlines() if not x.lstrip().startswith("#"))
    assert not re.search(r"(^|[;&|]\s*|\s)(rm|mv|rmdir)\s", code)


def test_name_across_settings():
    C, K = AN.PRE_CONV, AN.PRE_K0
    na = "UNSAFE-NOT-ATTRIBUTED"
    for c in AN.KINDS:
        for k in AN.KINDS:
            got = AN.across(C + c, K + k)
            if c == na and k == na:
                want = "PRESATTRC10K-NEITHER-SETTING-ATTRIBUTED"
            elif c == na or k == na:
                want = "PRESATTRC10K-ONE-SETTING-NOT-ATTRIBUTED"
            else:
                want = "PRESATTRC10K-ATTRIBUTION-AS-AT-K0" if c == k else "PRESATTRC10K-ATTRIBUTION-NOT-AS-AT-K0"
            assert got == want, (c, k)
    for bad in ((K + na, K + na), (C + na, C + na), (C + "X", K + na), (None, K + na)):
        with pytest.raises(SystemExit):
            AN.across(*bad)


def test_name_across_settings_from_files(tmp_path, capsys):
    a, b = str(tmp_path / "c.json"), str(tmp_path / "k.json")
    json.dump({"card": "PREREG_V8F_PROOF_ADD115", "names": {"CONV_OVERALL": "PRESATTRC10K-CONV-PRESENCE-CARRIES-UNSAFE"}}, io.open(a, "w", encoding="utf-8"))
    json.dump({"card": "PREREG_V8F_PROOF_ADD114", "names": {"K0_OVERALL": "PRESATTR10K-K0-PRESENCE-CARRIES-UNSAFE", "K30_OVERALL": "PRESATTR10K-K30-UNSAFE-NOT-ATTRIBUTED"}}, io.open(b, "w", encoding="utf-8"))
    assert AN.main(["--conv-result", a, "--k0-result", b]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "ZZ ACROSS-NAME PRESATTRC10K-ATTRIBUTION-AS-AT-K0"
    with pytest.raises(SystemExit):
        AN.main(["--conv-result", b, "--k0-result", a])


def test_wording_of_the_kit_files():
    parts = [("vo", "id"), ("super", "seded"), ("retr", "acted"), ("form", "erly"), ("correc", "tion"), ("with", "drawn"), ("no lon", "ger"), ("closed", " loop")]
    for fn in os.listdir(HERE):
        p = os.path.join(HERE, fn)
        if os.path.isfile(p) and fn.endswith((".py", ".sh", ".md")):
            low = rd(p).lower()
            for a, b in parts:
                assert (a + b) not in low, (fn, a + b)
