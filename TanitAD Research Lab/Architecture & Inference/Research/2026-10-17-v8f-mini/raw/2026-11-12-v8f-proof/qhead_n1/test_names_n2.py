"""test_names_n2.py -- note PREREG_V8F_PROOF_ADD42_N2 section 2: control h5 gates per arm type. Helpers come from test_h3_names_n1.py (synthetic reads and inputs).
The mutation cases build a changed copy of qhead_names_n2.py in memory and require the per-type scenario to read differently."""
import io
import json
import os
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_names_n2 as MK              # noqa: E402
import test_h3_names_n1 as T            # noqa: E402

SRC_N1 = io.open(os.path.join(HERE, "qhead_names_n1.py"), encoding="utf-8", newline="").read()
SRC_N2 = MK.build(SRC_N1)


def load(src, name="qhead_names_n2_mem"):
    m = types.ModuleType(name)
    m.__file__ = os.path.join(HERE, "qhead_names_n2.py")
    exec(compile(src, m.__file__, "exec"), m.__dict__)
    return m


N2 = load(SRC_N2)


def test_the_file_on_disk_is_the_built_text():
    p = os.path.join(HERE, "qhead_names_n2.py")
    assert os.path.isfile(p) and io.open(p, encoding="utf-8", newline="").read() == SRC_N2


def set_cov(a, base, c):
    p = "%s/%s.qhead.json" % (a.fit_dir, base)
    j = json.load(open(p))
    j["coverage_check"]["share_q_above_label"] = c
    json.dump(j, open(p, "w"))


def run(mod, a, out):
    rc = mod.main(["--hw", a.hw, "--q3", a.q3, "--fit-dir", a.fit_dir, "--h1-text", a.h1_text, "--tool-file", a.tool_file, "--rule-log-dir", a.rule_log_dir, "--h3", a.h3, "--h6", a.h6, "--out", out])
    return rc, json.load(open(out)), open(out).read()


def lsd_out_scenario(mod, tmp, drop_lsd_reads=True):
    """LSD_s0 is below the band (the measured case 0.1086); the LSD rollouts do not exist and the readers were called without the LSD pairs."""
    hw, q3 = T.reads()
    if drop_lsd_reads:
        for k in [k for k in hw["pairs"] if k.startswith("LSDQE")]:
            del hw["pairs"][k]
        for k in [k for k in q3["pairs"] if k.startswith("LSDQE")]:
            del q3["pairs"][k]
        del hw["levels"]["LSDQE"]

    def fail(a, d):
        set_cov(a, "LSD_s0", 0.1086)
        for s in (0, 1):                                    # what a missing rollout looks like: a rule log that would fail h2, an H3 file that would fail h3
            np.savez("%s/LSDQE_s%d/ticks/rule_log_LSDQE_s%d.npz" % (a.rule_log_dir, s, s), lift=np.array([1, 1]))
            json.dump({"PASS": False, "why": "not run"}, open("%s/H3_LSDQE_s%d.json" % (a.h3, s), "w"))
    a = T.make_inputs(tmp, hw, q3, fail)
    return run(mod, a, str(tmp) + "/o.json")


def lsd_out_reads_right(rc, r, text):
    try:
        h5 = r["controls"]["h5_check_coverage_in_0.15_0.25"]
        return bool(rc == 0 and r["controls_all_pass"] and h5["types_read"] == ["LS"] and h5["types_not_named"] == {"LSD": ["LSD_s0"]}
                    and r["names"]["LS"]["reaches_oracle"] == "QHEAD-REACHES-ORACLE-QE" and r["names"]["LSD"]["all"].startswith("NOT NAMED (h5")
                    and list(r["beside"].keys()) == ["LS"] and "LSDQE" not in json.dumps(r["beside"]) and "N2 section 2" in r["reading_rule"])
    except (KeyError, TypeError):
        return False


def test_all_inside_the_band_reads_both_types(tmp_path):
    hw, q3 = T.reads()
    rc, r, _ = run(N2, T.make_inputs(tmp_path, hw, q3), str(tmp_path / "o.json"))
    assert rc == 0 and r["controls"]["h5_check_coverage_in_0.15_0.25"]["types_read"] == ["LS", "LSD"] and r["controls"]["h5_check_coverage_in_0.15_0.25"]["types_not_named"] == {}
    assert r["names"]["LS"]["reaches_oracle"] == "QHEAD-REACHES-ORACLE-QE" and r["names"]["LSD"]["U_against_map"] == "QHEAD-CUTS-MORE-QE" and sorted(r["beside"]) == ["LS", "LSD"]


def test_type_with_a_checkpoint_outside_the_band_gets_no_name_and_no_number(tmp_path):
    assert lsd_out_reads_right(*lsd_out_scenario(N2, tmp_path))


def test_the_other_type_outside_the_band(tmp_path):
    hw, q3 = T.reads()
    a = T.make_inputs(tmp_path, hw, q3, lambda a, d: set_cov(a, "LS_s1", 0.31))
    rc, r, _ = run(N2, a, str(tmp_path / "o.json"))
    assert rc == 0 and r["names"]["LS"]["all"].startswith("NOT NAMED (h5") and "LS_s1" in r["names"]["LS"]["all"] and list(r["beside"].keys()) == ["LSD"]


def test_both_types_outside_the_band_names_nothing(tmp_path):
    hw, q3 = T.reads()

    def fail(a, d):
        set_cov(a, "LS_s0", 0.40)
        set_cov(a, "LSD_s1", 0.05)
    rc, r, text = run(N2, T.make_inputs(tmp_path, hw, q3, fail), str(tmp_path / "o.json"))
    assert rc == 2 and r["names"] is None and r["beside"] is None and not r["controls_all_pass"] and "QHEAD-" not in text and "U_qh_minus_none" not in text


@pytest.mark.parametrize("c,inside", [(0.15, True), (0.25, True), (0.1499, False), (0.2501, False), (None, False)])
def test_band_edges(tmp_path, c, inside):
    hw, q3 = T.reads()
    a = T.make_inputs(tmp_path, hw, q3, lambda a, d: set_cov(a, "LSD_s1", c))
    rc, r, _ = run(N2, a, str(tmp_path / "o.json"))
    assert rc == 0 and (("LSD" in r["controls"]["h5_check_coverage_in_0.15_0.25"]["types_read"]) is inside)


@pytest.mark.parametrize("which", ["h1", "h4", "h6", "h7", "h8_in_unread_type", "h2_read_type", "h3_read_type", "missing_fit_record"])
def test_the_other_controls_gate_every_name(tmp_path, which):
    hw, q3 = T.reads()

    def fail(a, d):
        set_cov(a, "LSD_s0", 0.1086)
        if which == "h1":
            open(a.h1_text, "w").write(open(a.h1_text).read().replace("differing=0", "differing=2"))
        elif which == "h4":
            j = json.load(open("%s/LS_s0.qhead.json" % a.fit_dir))
            j["files_read_sha256"]["/x/LS_s0_eval.npz"] = "b"
            json.dump(j, open("%s/LS_s0.qhead.json" % a.fit_dir, "w"))
        elif which == "h6":
            json.dump({"all_ok": False, "n_mutations": 9}, open(a.h6, "w"))
        elif which == "h7":
            h = json.load(open(a.hw))
            h["controls"]["ALL_PASS"] = False
            json.dump(h, open(a.hw, "w"))
        elif which == "h8_in_unread_type":
            j = json.load(open("%s/LSD_s0.qhead.json" % a.fit_dir))
            j["clip_index_range_of_rows_used"]["selection"] = [800, 1150]
            json.dump(j, open("%s/LSD_s0.qhead.json" % a.fit_dir, "w"))
        elif which == "h2_read_type":
            np.savez("%s/LSQE_s1/ticks/rule_log_LSQE_s1.npz" % a.rule_log_dir, lift=np.array([0, 1, 0]))
        elif which == "h3_read_type":
            json.dump({"PASS": False, "why": "x"}, open("%s/H3_LSQE_s0.json" % a.h3, "w"))
        elif which == "missing_fit_record":
            os.replace("%s/LSD_s1.qhead.json" % a.fit_dir, "%s/LSD_s1.qhead.json.moved" % a.fit_dir)
    rc, r, text = run(N2, T.make_inputs(tmp_path, hw, q3, fail), str(tmp_path / "o.json"))
    assert rc == 2 and r["names"] is None and r["beside"] is None and "QHEAD-" not in text


MUTATIONS = {
    "m1_every_type_is_read": ("    read_types = [ty for ty in TYPES if ty not in not_named]\n", "    read_types = list(TYPES)\n"),
    "m2_names_ignores_the_types": ("        if types is not None and ty not in types:\n            continue\n", ""),
    "m3_band_is_not_checked": ("            cov[base] = bool(c0 is not None and H5_LO <= c0 <= H5_HI)\n", "            cov[base] = bool(c0 is not None)\n"),
    "m4_reads_the_select_coverage": ('            c0 = ((f0 or {}).get("coverage_check") or {}).get("share_q_above_label")\n', '            c0 = ((f0 or {}).get("coverage_select") or {}).get("share_q_above_label")\n'),
    "m5_one_checkpoint_inside_is_enough": ("    not_named = {ty: v for ty, v in not_named.items() if v}\n", "    not_named = {ty: v for ty, v in not_named.items() if len(v) == len(SEEDS)}\n"),
    "m6_h2_on_every_arm": ('    arms = ["%s_s%d" % (TYPES[ty]["qh"], s) for ty in read_types for s in SEEDS] + ["LSQHI_s0"]\n', '    arms = ["%s_s%d" % (TYPES[ty]["qh"], s) for ty in TYPES for s in SEEDS] + ["LSQHI_s0"]\n'),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_mutation_changes_the_read_of_the_measured_case(tmp_path, name):
    old, new = MUTATIONS[name]
    assert SRC_N2.count(old) == 1
    mod = load(SRC_N2.replace(old, new), "mut_" + name)
    try:
        res = lsd_out_scenario(mod, tmp_path, drop_lsd_reads=(name != "m2_names_ignores_the_types"))
        ok = lsd_out_reads_right(*res)
    except (KeyError, TypeError):
        ok = False
    assert not ok, "the mutation %s was not seen" % name
