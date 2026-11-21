"""test_gaprule.py -- tests of the kit of card ADD94 (GAPRULE10K): planted joins, wrapper j text, queue, reader controls and names, with the planted faults of control j4. Synthetic only; nothing runs on the pod.
Run: PYCAP_MB=3000 bash ops/pycap.sh -m pytest gaprule/test_gaprule.py -q -p no:cacheprovider --basetemp gaprule/tmp_pt/<new name>"""
import difflib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_wrapper_j as CW          # noqa: E402
import make_planted_joins as MPJ      # noqa: E402
import read_gaprule10k as RG          # noqa: E402

WRAP_J = HERE / "v8f_t1_eval_reader_rule_rzgfhqetj.py"
PARENT = HERE.parent / "shufclose" / "v8f_t1_eval_reader_rule_rzgfhqet.py"
QUEUE = HERE / "pod_gaprule10k_queue.sh"


# ------------------------------------------------------------------------------------------------ planted joins (j1 and its faults)
@pytest.fixture(scope="module")
def joins(tmp_path_factory):
    base = tmp_path_factory.mktemp("pj")
    J = MPJ.synthetic()
    return {"J": J, "clean": MPJ.build(J, str(base / "clean")), "base": base}


def test_clean_build_passes_j1(joins):
    assert joins["clean"]["j1"]["PASS"], joins["clean"]["j1"]
    for a in MPJ.ARMS[1:]:
        st = joins["clean"]["arms"][a[0]]["planted_error_before_limits"]
        assert abs(st["sd"] - a[3]) <= 1e-9 and abs(st["mean"]) <= 1e-9


def test_lag1_bands(joins):
    u = joins["clean"]["unit_noise"]
    assert abs(u["W"]["lag1"]) < 0.05 and 0.89 <= u["R"]["lag1"] <= 0.97 and u["C"]["lag1"] >= 0.999


@pytest.mark.parametrize("fault", ["no_normalise", "prop_as_abs", "other_field"])
def test_planted_fault_turns_j1_red(joins, fault):
    rec = MPJ.build(joins["J"], str(joins["base"] / ("f_" + fault)), faults=(fault,))
    assert not rec["j1"]["PASS"] and rec["j1"]["failures"]


def test_out_dir_refused(joins):
    with pytest.raises(SystemExit):
        MPJ.build(joins["J"], str(joins["base"] / "clean"))


def test_g0_equals_label_and_only_lead_gap_changes(joins):
    a = joins["clean"]["arms"]
    assert a["G0"]["n_frames_changed_in_file"] == 0 and all(x["other_keys_all_equal"] for x in a.values())
    assert a["GR1"]["n_frames_changed_in_file"] > 0


def test_invalid_frames_untouched(joins):
    J = joins["J"]
    with np.load(str(joins["base"] / "clean" / "GR2.npz"), allow_pickle=False) as z:
        g = z["lead_gap"]
    v = MPJ.valid_mask(J)
    assert np.array_equal(g[~v], np.asarray(J["lead_gap"])[~v], equal_nan=True) and g.dtype == np.asarray(J["lead_gap"]).dtype


def test_seeding_by_name_when_join_has_clip_array():
    J = MPJ.synthetic()
    assert MPJ.clip_names(J)[1].startswith("clip name")
    J2 = {k: v for k, v in J.items() if k != "clip"}
    assert MPJ.clip_names(J2)[1].startswith("clip index")


# ------------------------------------------------------------------------------------------------ wrapper j text (j2 static part, and a rule join passed to the scoring)
def test_wrapper_j_is_the_parent_plus_counted_edits():
    p, n = PARENT.read_text(encoding="utf-8").split("\n"), WRAP_J.read_text(encoding="utf-8").split("\n")
    d = [x for x in difflib.unified_diff(p, n, lineterm="", n=0) if x[:1] in "+-" and x[:3] not in ("+++", "---")]
    assert len(d) <= 40, len(d)
    assert [x for x in CW.static_facts(WRAP_J.read_text(encoding="utf-8")) if not x[1]] == []


def test_rule_join_passed_to_the_scoring_turns_check_red():
    src = WRAP_J.read_text(encoding="utf-8")
    bad1 = src.replace('    X.STATE["label"] = build_rule_label(', '    a.eval_join = mine.rule_join or a.eval_join\n    X.STATE["label"] = build_rule_label(')
    bad2 = src.replace('W.install(T, "mix", a, None, log, "cap-only")', 'W.install(T, "mix", a, T.D.load_join(mine.rule_join), log, "cap-only")')
    assert bad1 != src and bad2 != src
    assert any(not ok for _, ok in CW.static_facts(bad1)) and any(not ok for _, ok in CW.static_facts(bad2))


def test_check_wrapper_j_compare_and_selftest():
    assert subprocess.run([sys.executable, str(HERE / "check_wrapper_j.py"), "--selftest"], capture_output=True).returncode == 0
    assert CW.maxdiff({"x": np.zeros(3)}, {"x": np.zeros(3) + 1e-12})[0] > 0


# ------------------------------------------------------------------------------------------------ queue
def test_queue_dry_run_has_twelve_rule_join_commands():
    bash = next((c for c in ("C:/Program Files/Git/bin/bash.exe", "C:/Program Files/Git/usr/bin/bash.exe") if os.path.isfile(c)), "bash")
    q = str(QUEUE).replace("\\", "/")
    assert subprocess.run([bash, "-n", q]).returncode == 0
    out = subprocess.run([bash, q], env=dict(os.environ, DRY="1"), capture_output=True, text=True).stdout
    cmds = [x for x in out.splitlines() if x.startswith("CMD:")]
    assert len(cmds) == 12 and all("--rule-join" in c and "rzgfhqetj.py" in c and "--form RLc" in c and "--kappa-max 0.2" in c and "--preroll-steps 30" in c for c in cmds)
    assert all("--eval-join /workspace/v8proof/hierrep/pins/join_eval141.npz" in c for c in cmds)
    for arm in ("G0", "GR1", "GR2", "GW05", "GC10", "GC20"):
        assert sum(("/joins/%s.npz" % arm) in c for c in cmds) == 2
    text = QUEUE.read_text(encoding="utf-8")
    assert "28" in text and " rm " not in text and " mv " not in text and " kill " not in text


# ------------------------------------------------------------------------------------------------ reader: names, controls, binding
def iv(lo, hi):
    return [lo, hi]


def test_name_rules_on_planted_intervals():
    n = RG.names_of("GR1", iv(-0.05, -0.01), iv(0.0, 0.004), iv(-0.02, 0.03), 0.0954, 0.0138)      # H = 0.0408
    assert n["GR1_CONTACT"].endswith("-GR1-CUTS-CONTACT") and n["GR1_HALF"].endswith("-GR1-WITHIN-HALF-THE-GAP") and n["GR1_PRICE"].endswith("-GR1-FALLING-BEHIND-NOT-SEPARATED")
    assert RG.names_of("GR2", iv(-0.01, 0.0), iv(0.02, 0.0408), iv(0.0, 0.1), 0.0954, 0.0138)["GR2_HALF"].endswith("WITHIN-HALF-THE-GAP")      # at H: within
    assert RG.names_of("GR2", iv(-0.01, 0.0), iv(0.0409, 0.05), iv(0.0, 0.1), 0.0954, 0.0138)["GR2_HALF"].endswith("OUTSIDE-HALF-THE-GAP")
    assert RG.names_of("GC10", iv(-0.01, 0.0), iv(0.02, 0.05), iv(0.0, 0.1), 0.0954, 0.0138)["GC10_HALF"].endswith("HALF-THE-GAP-NOT-DECIDED")
    assert RG.names_of("GC10", iv(0.0, 0.01), iv(0.02, 0.05), iv(-0.1, -0.001), 0.0954, 0.0138)["GC10_CONTACT"].endswith("CONTACT-NOT-SHOWN")
    assert RG.names_of("GC10", iv(0.0, 0.01), iv(0.02, 0.05), iv(-0.1, -0.001), 0.0954, 0.0138)["GC10_PRICE"].endswith("FALLS-BEHIND-LESS")
    assert RG.names_of("GC10", iv(0.0, 0.01), iv(0.02, 0.05), iv(0.001, 0.1), 0.0954, 0.0138)["GC10_PRICE"].endswith("FALLS-BEHIND-MORE")


def payload():
    pr = {}
    for a in RG.PLANTED:
        pr[a + "-VS-RR"] = {k: {"interval": iv(-0.05, -0.01), "pooled": -0.03} for k in ("C", "U", "F")}
        pr[a + "-VS-RLc"] = {k: {"interval": iv(0.0, 0.004), "pooled": 0.002} for k in ("C", "U", "F")}
    return {"levels": {"RR": {"C": 0.0954}, "RLc": {"C": 0.0138}}, "pairs": pr, "families": {}}


def test_fifteen_names():
    assert len(RG.all_names(payload())) == 15


def test_j0_j3_j2_j4_controls():
    c = {s: {"C": np.array([0.1, np.nan, 0.3]), "U": np.zeros(3)} for s in RG.SEEDS}
    assert RG.ctl_j0(c, {s: {k: v.copy() for k, v in c[s].items()} for s in RG.SEEDS})["PASS"]
    d = {s: {k: v.copy() for k, v in c[s].items()} for s in RG.SEEDS}
    d[1]["C"][2] += 1e-12
    assert not RG.ctl_j0(c, d)["PASS"]
    assert RG.ctl_j3(0.09541, 0.0138)["PASS"] and not RG.ctl_j3(0.0960, 0.0138)["PASS"]
    assert RG.ctl_j2("x\nZZ CHECK-WRAPPER-J PASS\n", "cbf7063d867c9f4e a.py", "cbf7063d867c9f4e")["PASS"] and not RG.ctl_j2("ZZ CHECK-WRAPPER-J FAIL J2A\n", "cbf7063d867c9f4e", "cbf7063d867c9f4e")["PASS"]
    assert RG.ctl_j4("25 passed in 3s")["PASS"] and not RG.ctl_j4("2 failed, 3 passed")["PASS"]


def test_j1_through_the_reader_and_fault(joins):
    sha = None
    ok = RG.ctl_j1(joins["clean"], str(joins["base"] / "clean"), sha)
    assert ok["PASS"], ok
    bad = MPJ.build(joins["J"], str(joins["base"] / "rd_bad"), faults=("no_normalise",))
    assert not RG.ctl_j1(bad, str(joins["base"] / "rd_bad"), sha)["PASS"]


def test_name_written_with_a_control_failed_turns_red(tmp_path):
    ctl = RG.assemble_controls({"j0": {"PASS": True}, "j1": {"PASS": False}}, payload())
    assert ctl["controls_all_pass"] is False and "payload" not in ctl and "withheld" in ctl
    f = tmp_path / "controls.json"
    f.write_text(json.dumps(ctl), encoding="utf-8")
    out = tmp_path / "result.json"
    assert RG.main(["--stage", "names", "--controls", str(f), "--out", str(out)]) == 3 and not out.exists()
    forged = dict(ctl, controls_all_pass=True)                                         # an edited flag does not open the names: the controls themselves are re-checked
    f2 = tmp_path / "forged.json"
    f2.write_text(json.dumps(dict(forged, payload=payload())), encoding="utf-8")
    assert RG.main(["--stage", "names", "--controls", str(f2), "--out", str(tmp_path / "r2.json")]) == 3


def test_names_stage_binds_the_controls_file(tmp_path):
    ctl = RG.assemble_controls({"j0": {"PASS": True}, "j1": {"PASS": True}}, payload())
    f = tmp_path / "c.json"
    f.write_text(json.dumps(ctl), encoding="utf-8")
    out = tmp_path / "r.json"
    assert RG.main(["--stage", "names", "--controls", str(f), "--out", str(out)]) == 0
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["controls_sha256"] == RG.sha_file(str(f)) and len(r["names"]) == 15


def test_texts_hold_no_banned_words():
    bad = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop")
    for f in ("make_wrapper_j.py", "make_planted_joins.py", "make_gaprule_queue.py", "pod_gaprule10k_queue.sh", "check_wrapper_j.py", "make_read_gaprule.py", "read_gaprule10k.py", "test_gaprule.py",
              "v8f_t1_eval_reader_rule_rzgfhqetj.py", "GAPRULE_DEFS.md"):
        p = HERE / f
        if p.exists():
            t = p.read_text(encoding="utf-8").lower()
            hit = [w for w in bad if w in t]
            if f == "v8f_t1_eval_reader_rule_rzgfhqetj.py":
                continue                                      # generated text carries the parent's words
            assert not hit, (f, hit)
