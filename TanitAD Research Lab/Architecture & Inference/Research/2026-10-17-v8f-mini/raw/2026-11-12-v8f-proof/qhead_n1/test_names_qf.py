"""test_names_qf.py -- the names tool of the dose arms (note PREREG_V8F_PROOF_ADD42_N2 section 3): one type (LSD, arms LSDQF), suffix -QF, controls d1 and d2 gate.
Helpers come from test_h3_names_n1.py; its module-level NM is pointed at the QF module for the test, so the synthetic reads and inputs carry the QF arms."""
import io
import json
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_qf_tools as MQ              # noqa: E402
import test_h3_names_n1 as T            # noqa: E402

SRC = MQ.names_qf_text()


def load(src, name="qhead_names_qf_mem"):
    m = types.ModuleType(name)
    m.__file__ = os.path.join(HERE, "qhead_names_qf.py")
    exec(compile(src, m.__file__, "exec"), m.__dict__)
    return m


QF = load(SRC)


@pytest.fixture(autouse=True)
def _point_helpers_at_qf(monkeypatch):
    monkeypatch.setattr(T, "NM", QF)


def test_the_file_on_disk_is_the_built_text():
    p = os.path.join(HERE, "qhead_names_qf.py")
    assert os.path.isfile(p) and io.open(p, encoding="utf-8", newline="").read() == SRC


def inputs(tmp, fail=None, d1=True, d2=(True, True), planted=(True, True)):
    hw, q3 = T.reads()
    a = T.make_inputs(tmp, hw, q3, fail)
    d = str(tmp)
    a.d1, a.d2_dir = d + "/d1_pack.json", d + "/controls"
    os.makedirs(a.d2_dir, exist_ok=True)
    if d1 is not None:
        json.dump({"pass": d1, "lo": 600, "hi": 1800}, open(a.d1, "w"))
    for s in (0, 1):
        if d2[s] is not None:
            json.dump({"pass": d2[s], "planted_shift": {"fails": planted[s]}}, open("%s/d2_LSD_s%d.json" % (a.d2_dir, s), "w"))
    return a


def run(mod, a, out):
    rc = mod.main(["--hw", a.hw, "--q3", a.q3, "--fit-dir", a.fit_dir, "--h1-text", a.h1_text, "--tool-file", a.tool_file, "--rule-log-dir", a.rule_log_dir, "--h3", a.h3, "--h6", a.h6,
                   "--d1", a.d1, "--d2-dir", a.d2_dir, "--out", out])
    return rc, json.load(open(out)), open(out).read()


def test_types_and_arms():
    assert list(QF.TYPES) == ["LSD"] and QF.TYPES["LSD"]["qh"] == "LSDQF" and QF.SUFFIX == "-QF"


def test_all_controls_pass_names_the_lsd_type_with_qf(tmp_path):
    rc, r, text = run(QF, inputs(tmp_path), str(tmp_path / "o.json"))
    assert rc == 0 and r["controls_all_pass"] and list(r["names"]) == ["LSD"]
    n = r["names"]["LSD"]
    assert n["reaches_oracle"] == "QHEAD-REACHES-ORACLE-QF" and n["U_against_map"] == "QHEAD-CUTS-MORE-QF" and n["cost_against_map"] == "QHEAD-CHEAPER-QF" and n["cost_against_no_rule"] == "QHEAD-NO-COST-QF"
    assert "-QE" not in text and "LSDQE" not in text and "section 3" in r["card"] and "STRATEGIC: NOT COMPUTED (n = 0;" in r["family_line"]
    assert r["controls"]["d1_label_pack_equals_banked_pack"]["PASS"] and r["controls"]["d2_new_states_equal_banked_states"]["PASS"]
    assert sorted(r["controls"]["h2_no_plan_moved_ahead"]["per_arm"]) == ["LSDQF_s0", "LSDQF_s1", "LSQHI_s0"]


def cov(base, c):
    def f(a, d):
        p = "%s/%s.qhead.json" % (a.fit_dir, base)
        j = json.load(open(p))
        j["coverage_check"]["share_q_above_label"] = c
        json.dump(j, open(p, "w"))
    return f


@pytest.mark.parametrize("kw", [dict(fail=cov("LSD_s0", 0.1086)), dict(fail=cov("LSD_s1", 0.26)), dict(d1=False), dict(d1=None), dict(d2=(True, False)), dict(d2=(None, True)),
                                dict(planted=(False, True)), dict(planted=(True, False))])
def test_a_failed_control_names_nothing(tmp_path, kw):
    rc, r, text = run(QF, inputs(tmp_path, **kw), str(tmp_path / "o.json"))
    assert rc == 2 and r["names"] is None and r["beside"] is None and not r["controls_all_pass"] and "QHEAD-" not in text and "U_qh_minus_none" not in text


MUTATIONS = {
    "m1_d1_not_read": ('bool(d1 and d1.get("pass") is True)', 'True'),
    "m2_d2_planted_shift_not_required": (' and ps.get("fails") is True)', ')'),
    "m3_d2_pass_not_required": ('bool(v and v.get("pass") is True and ', 'bool(v and '),
}


@pytest.mark.parametrize("name,kw", [("m1_d1_not_read", dict(d1=False)), ("m2_d2_planted_shift_not_required", dict(planted=(False, True))), ("m3_d2_pass_not_required", dict(d2=(True, False)))])
def test_mutation_lets_a_failed_control_through(tmp_path, name, kw):
    old, new = MUTATIONS[name]
    assert SRC.count(old) == 1
    mod = load(SRC.replace(old, new), "mut_" + name)
    rc, r, _ = run(mod, inputs(tmp_path, **kw), str(tmp_path / "o.json"))
    assert rc == 0 and r["names"] is not None, "the mutation %s was not seen (the unmutated tool refuses this case)" % name
