"""pytest of the reader kit of card ADD110 (FALLATTR10K): synthetic data only (tiny worlds of 30 join rows). The model reader of card ADD106 is imported as a sibling module (dev-box copy
cropscale/read/read_cropscale10k.py when the pod path does not exist). One test per control with its planted faults, every name branch and the agreement of the two derivations, a4's time rule,
the refusal of the names stage, the wording scan and the path length."""
import io
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import read_fallattr10k as RE                  # noqa: E402

RE.bind()
BANNED = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop")
ARMS = RE.ARMS


@pytest.fixture()
def world(tmp_path):
    return RE.make_world(str(tmp_path / "w"))


def ev(w):
    a, rec, J, m, wsha, jsha = w
    return RE.compute_controls(a, rec, J, m, wsha, jsha)


# ------------------------------------------------------------------------------------------------ selftest and controls
def test_selftest_ok(capsys):
    assert RE.selftest() == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "ZZ READ-FALLATTR10K SELFTEST OK"


def test_clean_world_all_pass(world):
    c = ev(world)
    assert sorted(c) == sorted(RE.CONTROLS)
    assert all(v["pass"] is True for v in c.values()), {k: v["failures"] for k, v in c.items()}


@pytest.mark.parametrize("fault", RE.CONTROLS)
def test_planted_fault_flips_exactly_that_control(tmp_path, fault):
    c = ev(RE.make_world(str(tmp_path / "f"), fault))
    assert {k: v["pass"] for k, v in c.items()} == {k: (k != fault) for k in RE.CONTROLS}
    assert c[fault]["failures"]


def test_model_constants_agree():
    M = RE.MD
    assert (M.WRAPPER_SHA, M.CHECK_LAST, M.V0_RUN) == (RE.WRAPPER_SHA, RE.CHECK_LAST, RE.V0_RUN)
    assert M.BANKED["RR"] == RE.REFS["RR"] and M.BANKED["RLc"] == RE.REFS["RLc"] and M.RUN["GEc"] == RE.REFS["GEc"]


def a0_args(w):
    a, rec, J, m, wsha, jsha = w
    files = {x: RE.gapfile_path(a.gap_dir, x) for x in ARMS}
    bp = rec["base"]["path"]
    return rec, files, J, jsha, RE.JOIN_NAME, RE.load_npz(bp), RE.sha_file(bp)


def test_a0_clean_and_carried_counts(world):
    r = RE.ctl_a0(*a0_args(world))
    assert r["pass"], r["failures"]
    v = r["values"]
    assert v["n_rows_carrying_label_gap_LA"] == 21
    assert v["n_rows_carrying_label_gap_LN"] + v["n_rows_carrying_label_gap_LM"] + v["n_rows_carrying_label_gap_LF"] == 21
    assert (v["n_rows_carrying_label_gap_LN"], v["n_rows_carrying_label_gap_LM"], v["n_rows_carrying_label_gap_LF"]) == (9, 6, 6)


def mutate_file(w, arm, fn):
    a = w[0]
    p = RE.gapfile_path(a.gap_dir, arm)
    z = RE.load_npz(p)
    fn(z)
    np.savez(p, **z)


def test_a0_record_clause_fails(world):
    rec, *rest = a0_args(world)
    rec["clauses"]["c2"]["pass"] = False
    r = RE.ctl_a0(rec, *rest)
    assert not r["pass"] and any("c2" in x for x in r["failures"])


def test_a0_record_without_clauses_fails(world):
    rec, *rest = a0_args(world)
    rec["clauses"] = {}
    assert not RE.ctl_a0(rec, *rest)["pass"]


def test_a0_file_hash_mismatch(world):
    def f(z):
        z["gap"] = z["gap"].copy()
        z["gap"][0] += 1.0
    mutate_file(world, "LN", f)
    r = RE.ctl_a0(*a0_args(world))
    assert not r["pass"] and any("sha256" in x for x in r["failures"])


def test_a0_composition_fault_with_matching_hash(world):
    """a file changed on one row and its record hash updated: only the composition clause can catch it"""
    def f(z):
        z["gap"] = z["gap"].copy()
        z["gap"][np.flatnonzero(z["has_out"])[5]] += 0.25
    mutate_file(world, "LF", f)
    rec, *rest = a0_args(world)
    rec["files"]["LF"]["sha256"] = RE.sha_file(RE.gapfile_path(world[0].gap_dir, "LF"))
    r = RE.ctl_a0(rec, *rest)
    assert not r["pass"] and any("composition" in x for x in r["failures"])


def test_a0_carried_rows_by_range(world):
    """each range file carries the label gap only inside its own range, so the three sets are disjoint and LA carries all label rows"""
    a, rec, J, m, wsha, jsha = world
    zn = RE.load_npz(RE.gapfile_path(a.gap_dir, "LN"))
    za = RE.load_npz(RE.gapfile_path(a.gap_dir, "LA"))
    has, gap, _, _ = RE.file_on_join(J, zn)
    has2, gap2, _, _ = RE.file_on_join(J, za)
    assert int(RE.carried_rows(J, "LN", has, gap).sum()) == 9 and int(RE.carried_rows(J, "LA", has2, gap2).sum()) == 21
    assert not RE.carried_rows(J, "LM", has, gap).any()
    assert RE.carried_rows(J, "LA", has, gap).sum() == 9                  # the LN file carries only its own range among all label rows


def test_a0_join_sha_and_name(world):
    rec, files, J, jsha, jname, base, bsha = a0_args(world)
    assert not RE.ctl_a0(rec, files, J, "0" * 64, jname, base, bsha)["pass"]
    assert not RE.ctl_a0(rec, files, J, jsha, "other.npz", base, bsha)["pass"]


def test_a0_base_clauses(world):
    rec, files, J, jsha, jname, base, bsha = a0_args(world)
    assert not RE.ctl_a0(rec, files, J, jsha, jname, base, "0" * 64)["pass"]
    assert not RE.ctl_a0(rec, files, J, jsha, jname, None, None)["pass"]
    rec["base"]["named_in_record_add106"] = False
    assert not RE.ctl_a0(rec, files, J, jsha, jname, base, bsha)["pass"]


def test_a0_duplicate_key_fails(world):
    def f(z):
        z["frame"] = z["frame"].copy()
        z["frame"][1] = z["frame"][0]
        z["clip_index"] = z["clip_index"].copy()
        z["clip_index"][1] = z["clip_index"][0]
    mutate_file(world, "LA", f)
    assert not RE.ctl_a0(*a0_args(world))["pass"]


def test_a0_label_row_rule():
    J = {"clip": np.array(["c"]), "off": np.array([0]), "n": np.array([6]), "lead_gap": np.array([10.0, 60.0, 60.5, -1.0, np.nan, 30.0]), "lead_valid": np.array([1, 1, 1, 1, 1, 0])}
    _, lab = RE.label_rows(J)
    assert lab.tolist() == [True, True, False, False, False, False]


def test_a1_clauses():
    ok = {"%s_s%d" % (x, m): RE.WRAPPER_SHA + "0" * 48 for x in ARMS for m in (0, 1)}
    chk = "x\n" + RE.CHECK_LAST + "\n"
    assert RE.ctl_a1(RE.WRAPPER_SHA, chk, RE.WRAPPER_SHA, ok)["pass"]
    assert not RE.ctl_a1("0" * 16, chk, RE.WRAPPER_SHA, ok)["pass"]
    assert not RE.ctl_a1(RE.WRAPPER_SHA, "x\nZZ CHECK-WRAPPER-V FAIL\n", RE.WRAPPER_SHA, ok)["pass"]
    assert not RE.ctl_a1(RE.WRAPPER_SHA, chk, "abc", ok)["pass"]
    bad = dict(ok)
    bad["LA_s0"] = "f" * 64
    assert not RE.ctl_a1(RE.WRAPPER_SHA, chk, RE.WRAPPER_SHA, bad)["pass"]
    bad = dict(ok)
    bad["LA_s0"] = None
    assert not RE.ctl_a1(RE.WRAPPER_SHA, chk, RE.WRAPPER_SHA, bad)["pass"]
    assert not RE.ctl_a1(RE.WRAPPER_SHA, chk, RE.WRAPPER_SHA, {k: v for k, v in list(ok.items())[:7]})["pass"]


def rewrite_logs(w, run, m, fn):
    a = w[0]
    d = os.path.join(a.pull, "I10%s_s%d" % (run, m), "ticks")
    for nm in ("rule_log_I10%s_s%d.npz", "rule_log_mid_I10%s_s%d.npz"):
        p = os.path.join(d, nm % (run, m))
        z = RE.load_npz(p)
        fn(z)
        np.savez(p, **z)


def test_a2_clean_has_shares(world):
    a, rec, J, *_ = world
    per = RE.a2_from_files(a, rec, J)
    r = RE.ctl_a2(per)
    assert r["pass"], r["failures"]
    sh = r["values"]["rollout_shares"]
    assert sorted(sh) == RE.rollout_keys()
    assert sh["LA_s0"]["share_of_lead_steps_with_label_gap"] == pytest.approx(21 / 30.0)
    assert sh["LN_s0"]["share_of_lead_steps_with_label_gap"] == pytest.approx(9 / 30.0)
    assert sh["LA_s0"]["share_of_lead_steps_with_nonzero_closing"] == pytest.approx(0.5)


def test_a2_closing_zero_fails(world):
    a, rec, J, *_ = world
    rewrite_logs(world, RE.RUN["LM"], 1, lambda z: z.update(c_src=np.zeros_like(z["c_src"])))
    r = RE.ctl_a2(RE.a2_from_files(a, rec, J))
    assert not r["pass"] and any("closing" in x for x in r["failures"])


def test_a2_no_output_step_must_equal_v0(world):
    a, rec, J, *_ = world
    def f(z):
        z["g_src"] = z["g_src"].copy()
        z["g_src"][np.flatnonzero((z["j"] == 3))] += 1.0           # frame 3 is a step with no output in the base file: V0 value expected
    rewrite_logs(world, RE.RUN["LF"], 0, f)
    r = RE.ctl_a2(RE.a2_from_files(a, rec, J))
    assert not r["pass"]


def test_a2_meta_names_file_and_switch(world):
    a, rec, J, *_ = world
    rewrite_logs(world, RE.RUN["LN"], 0, lambda z: z.update(vis_gap_file_sha256=np.array("0" * 64)))
    assert not RE.ctl_a2(RE.a2_from_files(a, rec, J))["pass"]
    w2 = RE.make_world(str(Path(a.pull).parent.parent / "w2"))
    rewrite_logs(w2, RE.RUN["LN"], 1, lambda z: z.update(vis_closing=np.array(0)))
    assert not RE.ctl_a2(RE.a2_from_files(w2[0], w2[1], w2[2]))["pass"]


def test_a2_missing_rollout_fails(world):
    a, rec, J, *_ = world
    per = RE.a2_from_files(a, rec, J)
    per.pop("LF_s1")
    assert not RE.ctl_a2(per)["pass"]
    shutil.rmtree(os.path.join(a.pull, "I10%s_s0" % RE.RUN["LA"]))
    per = RE.a2_from_files(a, rec, J)
    assert not RE.ctl_a2(per)["pass"]


def test_a3_each_value():
    lv = {k: dict(v) for k, v in RE.EXPECTED_LEVELS.items()}
    assert RE.ctl_a3(lv)["pass"]
    lv["RR"]["U"] += 0.00005
    assert RE.ctl_a3(lv)["pass"]                                       # inside 1e-4
    lv["RR"]["U"] += 0.0001
    assert not RE.ctl_a3(lv)["pass"]
    lv = {k: dict(v) for k, v in RE.EXPECTED_LEVELS.items()}
    lv["RLc"]["F"] = 0.4908 + 0.0002
    assert not RE.ctl_a3(lv)["pass"]
    assert not RE.ctl_a3({"GEc": lv["GEc"]})["pass"]
    lv["GEc"]["C"] = float("nan")
    assert not RE.ctl_a3(lv)["pass"]


def test_a4_time_rule():
    g, r = "...\n== 12 passed in 1.0s ==\n", "...\n== 30 passed in 2.0s ==\n\n"
    assert RE.ctl_a4(g, r, 100, 200, 300)["pass"]
    assert RE.ctl_a4(g, r, 200, 100, 300)["pass"]
    assert not RE.ctl_a4(g, r, 100, 300, 300)["pass"]                  # equal to the start: not older
    assert not RE.ctl_a4(g, r, 100, 301, 300)["pass"]
    assert not RE.ctl_a4(g, r, 301, 100, 300)["pass"]
    assert not RE.ctl_a4(g, r, None, 100, 300)["pass"]
    assert not RE.ctl_a4(g, r, 100, 200, None)["pass"]
    assert RE.ctl_a4(g, r, "100", "200", "1970-01-01T00:05:00Z")["pass"]
    assert not RE.ctl_a4(g, r, "100", "400", "1970-01-01T00:05:00Z")["pass"]


def test_a4_text_rule():
    ok = "== 5 passed in 1s =="
    assert not RE.ctl_a4("== 2 failed, 5 passed in 1s ==", ok, 1, 1, 9)["pass"]
    assert not RE.ctl_a4(ok, "== 5 passed, 1 error in 1s ==", 1, 1, 9)["pass"]
    assert not RE.ctl_a4("", ok, 1, 1, 9)["pass"]
    assert not RE.ctl_a4(ok, "== no tests ran in 0.1s ==", 1, 1, 9)["pass"]
    assert not RE.ctl_a4(ok, "== 0 passed in 0.1s ==", 1, 1, 9)["pass"]
    assert not RE.ctl_a4("== 5 passed ==\nnote: something", ok, 1, 1, 9)["pass"]          # the last non-empty line decides


def test_a5_parts():
    assert RE.ctl_a5(True, True, True)["pass"]
    assert not RE.ctl_a5(False, True, True)["pass"]
    assert not RE.ctl_a5(True, False, True)["pass"]
    assert not RE.ctl_a5(True, True, None)["pass"]


# ------------------------------------------------------------------------------------------------ names
LOW, HIGH, MID0 = [-0.05, -0.01], [0.01, 0.05], [-0.02, 0.02]


@pytest.mark.parametrize("iv,word", [(LOW, "LOWERS"), (HIGH, "RAISES"), (MID0, "NOT-SEPARATED"), ([-0.02, 0.0], "NOT-SEPARATED"), ([0.0, 0.02], "NOT-SEPARATED")])
def test_per_arm_measure_names(iv, word):
    for x in ARMS:
        for m, label in RE.MEASURES:
            kw = {"c": {x: iv}} if m == "C" else ({"u": {x: iv}} if m == "U" else {})
            pl = RE.mk_payload({x: iv}, **kw) if m == "F" else RE.mk_payload(**kw)
            nm = RE.all_names(pl)["%s_%s" % (x, label)]
            want = ("FALLATTR10K-%s-%s-%s" % (x, word, label)) if word != "NOT-SEPARATED" else ("FALLATTR10K-%s-%s-NOT-SEPARATED" % (x, label))
            assert nm == want


@pytest.mark.parametrize("la_gec,la_rlc,whole", [
    (LOW, MID0, "-GAP-CARRIES-THE-PRICE"), (LOW, LOW, "-GAP-CARRIES-THE-PRICE"), (LOW, [-0.03, 0.0], "-GAP-CARRIES-THE-PRICE"), (LOW, HIGH, "-GAP-CARRIES-PART"),
    (MID0, HIGH, "-PRICE-NOT-IN-THE-GAP"), (HIGH, HIGH, "-PRICE-NOT-IN-THE-GAP"), (MID0, MID0, "-NOT-DECIDED"), (HIGH, LOW, "-NOT-DECIDED"), (MID0, LOW, "-NOT-DECIDED"),
    ([-0.02, 0.0], HIGH, "-PRICE-NOT-IN-THE-GAP")])
def test_whole_branches(la_gec, la_rlc, whole):
    pl = RE.mk_payload({"LA": la_gec}, {"LA": la_rlc})
    assert RE.all_names(pl)["WHOLE"] == "FALLATTR10K" + whole
    assert RE.derive_all(pl)["WHOLE"] == "FALLATTR10K" + whole


def test_range_branches():
    def rng(sel):
        return RE.all_names(RE.mk_payload({x: LOW for x in sel}))["RANGE"]
    assert rng([]) == ["FALLATTR10K-NO-RANGE-SEPARATED"]
    assert rng(["LN"]) == ["FALLATTR10K-PRICE-IN-NEAR"]
    assert rng(["LM"]) == ["FALLATTR10K-PRICE-IN-MIDDLE"]
    assert rng(["LF"]) == ["FALLATTR10K-PRICE-IN-FAR"]
    assert rng(["LN", "LF"]) == ["FALLATTR10K-PRICE-IN-NEAR", "FALLATTR10K-PRICE-IN-FAR"]
    assert rng(["LN", "LM", "LF"]) == ["FALLATTR10K-PRICE-IN-NEAR", "FALLATTR10K-PRICE-IN-MIDDLE", "FALLATTR10K-PRICE-IN-FAR"]
    assert RE.all_names(RE.mk_payload({"LA": LOW}))["RANGE"] == ["FALLATTR10K-NO-RANGE-SEPARATED"]      # LA is not a range arm
    assert RE.all_names(RE.mk_payload({"LN": HIGH}))["RANGE"] == ["FALLATTR10K-NO-RANGE-SEPARATED"]


def test_two_derivations_agree_on_random_intervals():
    rnd = random.Random(11)
    pool = [LOW, HIGH, MID0, [-0.02, 0.0], [0.0, 0.02], [-0.03, -0.0001], [0.0001, 0.03]]
    for _ in range(300):
        pick = lambda: {x: rnd.choice(pool) for x in ARMS}              # noqa: E731
        pl = RE.mk_payload(pick(), pick(), pick(), pick())
        assert RE.names_of(pl) == RE.derive_all(pl)
        assert RE.all_names(pl) == RE.names_of(pl)


def test_disagreement_is_refused(monkeypatch):
    pl = RE.mk_payload({"LA": LOW}, {"LA": LOW})
    monkeypatch.setattr(RE, "derive_all", lambda p: dict(RE.names_of(p), WHOLE="FALLATTR10K-NOT-DECIDED"))
    with pytest.raises(SystemExit):
        RE.all_names(pl)


# ------------------------------------------------------------------------------------------------ stages
def full_payload():
    pl = RE.mk_payload({"LA": LOW, "LF": LOW}, {"LA": MID0})
    lv = {k: {"C": 0.01, "U": 0.05, "F": 0.5} for k in list(ARMS) + ["GEc", "RR", "RLc"]}
    pl.update({"levels": lv, "levels_per_seed": {k: {c: [0.1, 0.2] for c in "CUF"} for k in lv}, "families": {}, "strategic": {"pairs": {}},
               "gapfile_counts_from_record": {x: {"n_label_rows": 5, "n_source_changes": 2} for x in ARMS}, "control_values": {"a2": {"rollout_shares": {"LA_s0": {"n_rows": 3}}}}})
    return pl


def controls_file(path, all_pass=True, drop=None, flip=None):
    ctl = {k: {"pass": True, "failures": []} for k in RE.CONTROLS}
    if drop:
        ctl.pop(drop)
    if flip:
        ctl[flip]["pass"] = False
    res = {"card": RE.CARD, "controls": ctl, "all_pass": all_pass, "payload": full_payload()}
    Path(path).write_text(json.dumps(res), encoding="utf-8")


def test_names_stage_refuses_each_missing_or_failed_control(tmp_path, capsys):
    for i, k in enumerate(RE.CONTROLS):
        for kind in ("flip", "drop"):
            cf = tmp_path / ("c_%s_%s.json" % (k, kind))
            controls_file(cf, all_pass=(kind == "drop"), **{kind: k})
            out = tmp_path / ("r_%s_%s.json" % (k, kind))
            rc = RE.main(["--stage", "names", "--controls", str(cf), "--out", str(out)])
            assert rc == 3 and not out.exists()
    capsys.readouterr()


def test_names_stage_writes_result_and_log(tmp_path, capsys):
    cf, out = tmp_path / "c.json", tmp_path / "r.json"
    controls_file(cf)
    assert RE.main(["--stage", "names", "--controls", str(cf), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["controls_sha256"] == RE.sha_file(str(cf))
    assert res["names"]["WHOLE"] == "FALLATTR10K-GAP-CARRIES-THE-PRICE" and "FALLATTR10K-PRICE-IN-FAR" in res["names"]["RANGE"]
    log = (tmp_path / "r_names.log").read_text(encoding="utf-8")
    assert log.strip() == printed.strip()
    lines = log.splitlines()
    assert lines[0].startswith("TIER T1-kin") and lines[-1] == "ZZ READ-FALLATTR10K names done"
    assert any(x.startswith("FALLATTR10K-LA-LOWERS-FALLING-BEHIND") for x in lines)
    assert any(x.startswith("level GEc pooled") for x in lines) and any(x.startswith("LM-VS-RR:") for x in lines) and any(x.startswith("gap file LF:") for x in lines)
    assert any(x.startswith("rollout LA_s0:") for x in lines)
    assert RE.main(["--stage", "names", "--controls", str(cf), "--out", str(out)]) == 2      # a second read into the same file is refused


def test_assemble_controls_withholds_on_fail():
    ok = {k: {"pass": True, "failures": [], "values": {"x": 1}} for k in RE.CONTROLS}
    r = RE.assemble_controls(ok, {"levels": {}}, {"h": 1})
    assert r["all_pass"] and "payload" in r and "withheld" not in r and r["payload"]["control_values"]["a0"] == {"x": 1}
    ok["a3"]["pass"] = False
    r = RE.assemble_controls(ok, {"levels": {}}, {"h": 1})
    assert not r["all_pass"] and "withheld" in r and "payload" not in r
    assert sorted(r["controls"]) == sorted(RE.CONTROLS) and "values" not in r["controls"]["a0"]


def test_stdout_of_controls_has_only_the_allowed_lines(tmp_path, monkeypatch, capsys):
    res = RE.assemble_controls({k: {"pass": k != "a4", "failures": ["x"]} for k in RE.CONTROLS}, {}, {})
    monkeypatch.setattr(RE, "run_controls", lambda a: res)
    out = tmp_path / "c.json"
    assert RE.main(["--stage", "controls", "--out", str(out)]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == ["a0 PASS", "a1 PASS", "a2 PASS", "a3 PASS", "a4 FAIL", "a5 PASS", "ZZ READ-FALLATTR10K controls all_pass False"]
    assert "withheld" in json.loads(out.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ files: wording, paths, scripts
def kit_files():
    return [p for p in HERE.iterdir() if p.is_file() and p.suffix in (".py", ".sh", ".md")]


def test_wording_scan_and_path_length():
    names = {p.name for p in kit_files()}
    assert {"read_fallattr10k.py", "podread_fallattr10k.sh", "pull_fallattr.sh", "push_fallattr_read.sh", "test_fallattr_read.py", "FALLATTR_READ_DEFS.md"} <= names
    for p in kit_files():
        low = p.read_text(encoding="utf-8").lower()
        for w in BANNED:
            assert w not in low, (p.name, w)
        rel = "fallattr/read/" + p.name
        assert len(rel) <= 200 and p.stem.lower() not in ("aux", "con", "prn", "nul")


def test_scripts_parse_and_hold_their_rules():
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("no bash")
    for nm in ("podread_fallattr10k.sh", "pull_fallattr.sh", "push_fallattr_read.sh"):
        r = subprocess.run([bash, "-n", str(HERE / nm)], capture_output=True, encoding="utf-8")
        assert r.returncode == 0, (nm, r.stderr)
    pod = (HERE / "podread_fallattr10k.sh").read_text(encoding="utf-8")
    for must in ("FALLATTR10K_DONE.txt", "FALLATTR10K_STOP.txt", "FALLATTR10K_START.txt", "stat -c %Y", "date -d", "--selftest", "-ge 20", "ALREADY-READ", "aLA aLN aLM aLF gGEc vV0", "CUDA_VISIBLE_DEVICES=-1",
                "CS=$V/cropscale", "MODEL=$CS/kit_read/read_cropscale10k.py"):
        assert must in pod, must
    for banned_cmd in ("rm ", "mv ", "kill", "pkill"):
        assert banned_cmd not in pod, banned_cmd
    pull = (HERE / "pull_fallattr.sh").read_text(encoding="utf-8")
    assert "sed '/hf_/d'" in pull and "*.npz|*.pt) continue" in pull and "pod_add110" in pull and "sha256sum" in pull
    push = (HERE / "push_fallattr_read.sh").read_text(encoding="utf-8")
    assert "set -C" in push and "sha256sum" in push and "kit_read" in push and "lead_pytest.out" in push


def test_model_is_imported_not_edited():
    src = (HERE / "read_fallattr10k.py").read_text(encoding="utf-8")
    assert "spec_from_file_location" in src and "def w2_one" not in src and "def lead_multi" not in src
    for fn in ("rows_of", "load_logs", "lead_multi", "w2_one", "gapfile_ref", "meta_of", "run_file_sha", "ctl_g8"):
        assert hasattr(RE.MD, fn)
        assert "MD." + fn in src
