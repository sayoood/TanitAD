"""PREREG_V8F_STRAT1 launch gate: the token and the runner (``stack/scripts/strat1_check.py``). A fake tree, fake data and a fake trainer (it records its argv): no model, no GPU.

Asserted: an untampered setup PASSES and the trainer starts under EXACTLY the argv on disk; a tampered argv byte, a changed data file (join, nav npz, a bank shard, the cache manifest), a changed trainer or tree file, a missing
token, an edited token body, a wrong commit and a G-SUITE result with a regression each make the runner REFUSE (return code 2, the trainer never starts); a token cannot be written with a check that is not PASS.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import strat1_check as C  # noqa: E402

COMMIT = "b9250158fc6c41aefac50a3e7459d46a2e5c2f83"
FAKE_TRAINER = """import json, sys
a = sys.argv[1:]
out = a[a.index("--out") + 1]
import os
os.makedirs(out, exist_ok=True)
json.dump(a, open(os.path.join(out, "ran_argv.json"), "w"))
"""


def _w(p, data):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(data)
    return p


def _junit(d, results):
    os.makedirs(d, exist_ok=True)
    cases = "".join(f'<testcase classname="t.mod" name="{n}">' + ("" if s == "pass" else "<failure/>") + "</testcase>" for n, s in results.items())
    open(os.path.join(d, "shard0.xml"), "w").write(f"<testsuites><testsuite>{cases}</testsuite></testsuites>")


@pytest.fixture
def rig(tmp_path):
    root = str(tmp_path / "tree")
    _w(os.path.join(root, "stack", "scripts", "v8f_mini_train.py"), FAKE_TRAINER.encode())
    _w(os.path.join(root, "stack", "tanitad", "m.py"), b"x = 1\n")
    _w(os.path.join(root, "taniteval", "e.py"), b"y = 2\n")
    inp = str(tmp_path / "inputs")
    cache = str(tmp_path / "cache")
    _w(os.path.join(cache, "train", "manifest.jsonl"), b'{"clip": "a", "n_frames": 3}\n')
    _w(os.path.join(cache, "eval", "manifest.jsonl"), b'{"clip": "b", "n_frames": 3}\n')
    paths = {f: _w(os.path.join(inp, f.strip("-") + ".bin"), (f * 5).encode()) for f in C.DATA_FLAGS if f not in ("--nav-text-npz", "--ctx-labels")}
    nav = _w(os.path.join(inp, "strat1", "navtext_nav.npz"), b"navbytes")
    rep = {sp: {"G-CLOCK-NAV_pass": True, "G-CLOCK-NAV_regression_arm_plus_travelled": {"pass": False}} for sp in ("train", "eval")}
    rep["navtext_nav_sha256"] = C.sha_file(nav)
    nav_report = os.path.join(inp, "strat1", "navtext_nav_report.json")
    json.dump(rep, open(nav_report, "w"))
    ctx = _w(os.path.join(inp, "ctx.npz"), b"ctxbytes")
    bank = os.path.join(inp, "bank")
    _w(os.path.join(bank, "bank.json"), b'{"b": 1}')
    _w(os.path.join(bank, "shard_000of004.npz"), b"shard0")
    out = str(tmp_path / "runs" / "S1_s0" / "train")
    toks = ["--arm", "M2b", "--seed", "0", "--out", out, "--cache-dir", cache, "--w-acc", "49.6969"]
    for f, p in paths.items():
        toks += [f, p]
    toks += ["--dagger-bank", bank, "--dagger-bank-sha256", C.sha_file(os.path.join(bank, "bank.json")),
             "--nav-text-npz", nav, "--nav-text-sha256", C.sha_file(nav), "--ctx-labels", ctx, "--ctx-labels-sha256", C.sha_file(ctx)]
    _junit(str(tmp_path / "jb"), {"a": "pass", "b": "pass", "c": "fail"})
    _junit(str(tmp_path / "jn"), {"a": "pass", "b": "pass", "c": "fail", "d": "pass"})
    suite = str(tmp_path / "gsuite_result.json")
    json.dump(C.suite_compare(str(tmp_path / "jb"), str(tmp_path / "jn"), COMMIT, "cdbff971"), open(suite, "w"))
    runs = str(tmp_path / "runs")
    rd = os.path.join(runs, "S1_s0")
    os.makedirs(os.path.join(rd, "gate"))
    json.dump(toks, open(os.path.join(rd, "argv.json"), "w"))
    sc = _w(os.path.join(rd, "gate", "selfcheck.json"), b'{"declared_vs_built": "PASS"}')
    b, bad = C.binding("S1", toks, root, COMMIT, suite, nav_report=nav_report)
    assert bad == []
    checks = {c: {"pass": True} for c in C.REQUIRED_CHECKS}
    C.write_token(os.path.join(rd, "gate", "gate_token.json"), b, checks, C.sha_file(sc))
    a = argparse.Namespace(runs=runs, inputs=inp, root=root, commit=COMMIT, suite=suite, trainer="", py=sys.executable, gpu="")
    return dict(a=a, rd=rd, toks=toks, root=root, out=out, paths=paths, nav=nav, bank=bank, cache=cache, suite=suite, nav_report=nav_report, tmp=tmp_path)


def _refused(rig):
    rc = C.run_arm("S1_s0", "S1", rig["a"])
    assert rc == 2
    assert not os.path.exists(os.path.join(rig["out"], "ran_argv.json")), "the trainer started although the runner refused"


def test_untampered_passes_and_trainer_gets_exact_argv(rig):
    assert C.token_problems(os.path.join(rig["rd"], "gate", "gate_token.json"), "S1", rig["toks"], rig["root"], COMMIT, rig["suite"], nav_report=rig["nav_report"]) == []
    assert C.run_arm("S1_s0", "S1", rig["a"]) == 0
    assert json.load(open(os.path.join(rig["out"], "ran_argv.json"))) == rig["toks"]


def test_tampered_argv_byte_refuses(rig):
    p = os.path.join(rig["rd"], "argv.json")
    s = open(p).read()
    assert "49.6969" in s
    open(p, "w").write(s.replace("49.6969", "49.6960"))
    _refused(rig)


def test_tampered_argv_other_flag_refuses(rig):
    p = os.path.join(rig["rd"], "argv.json")
    t = json.load(open(p))
    t[t.index("--seed") + 1] = "1"
    json.dump(t, open(p, "w"))
    _refused(rig)


@pytest.mark.parametrize("which", ["join", "nav", "shard", "manifest", "bankjson"])
def test_changed_data_file_refuses(rig, which):
    p = {"join": rig["paths"]["--join-train"], "nav": rig["nav"], "shard": os.path.join(rig["bank"], "shard_000of004.npz"),
         "manifest": os.path.join(rig["cache"], "train", "manifest.jsonl"), "bankjson": os.path.join(rig["bank"], "bank.json")}[which]
    with open(p, "ab") as f:
        f.write(b"\x00")
    _refused(rig)


def test_missing_token_refuses(rig):
    os.remove(os.path.join(rig["rd"], "gate", "gate_token.json"))
    _refused(rig)


def test_edited_token_body_refuses(rig):
    p = os.path.join(rig["rd"], "gate", "gate_token.json")
    t = json.load(open(p))
    t["commit"] = "0" * 40
    json.dump(t, open(p, "w"))
    _refused(rig)


def test_hand_written_token_refuses(rig):
    p = os.path.join(rig["rd"], "gate", "gate_token.json")
    t = json.load(open(p))
    t["written_by"] = "me"
    t["body_sha256"] = C.body_sha(t)
    json.dump(t, open(p, "w"))
    _refused(rig)


def test_wrong_commit_refuses(rig):
    rig["a"].commit = "1" * 40
    _refused(rig)


def test_trainer_or_tree_change_refuses(rig):
    with open(os.path.join(rig["root"], "taniteval", "e.py"), "ab") as f:
        f.write(b"z = 3\n")
    _refused(rig)


def test_trainer_change_refuses(rig):
    with open(os.path.join(rig["root"], "stack", "scripts", "v8f_mini_train.py"), "ab") as f:
        f.write(b"# edited\n")
    _refused(rig)


def test_token_cannot_be_written_with_a_failed_check(rig):
    b, _ = C.binding("S1", rig["toks"], rig["root"], COMMIT, rig["suite"])
    checks = {c: {"pass": True} for c in C.REQUIRED_CHECKS}
    checks["G-EVAL"] = {"pass": False}
    with pytest.raises(ValueError):
        C.write_token(os.path.join(rig["rd"], "gate", "x.json"), b, checks, "0")
    assert not os.path.exists(os.path.join(rig["rd"], "gate", "x.json"))


def test_suite_regression_blocks_the_binding(rig):
    _junit(str(rig["tmp"] / "jn2"), {"a": "fail", "b": "pass", "c": "fail"})
    r = C.suite_compare(str(rig["tmp"] / "jb"), str(rig["tmp"] / "jn2"), COMMIT, "cdbff971")
    assert r["n_regressions"] == 1 and r["regressions"] == ["t.mod::a"] and r["fail_on_both"] == ["t.mod::c"]
    p = str(rig["tmp"] / "gs2.json")
    json.dump(r, open(p, "w"))
    _, bad = C.binding("S1", rig["toks"], rig["root"], COMMIT, p)
    assert any("regressions" in x for x in bad)


def test_nav_report_failure_refuses(rig):
    r = json.load(open(rig["nav_report"]))
    r["train"]["G-CLOCK-NAV_pass"] = False
    json.dump(r, open(rig["nav_report"], "w"))
    _refused(rig)


def test_suite_hand_edited_result_is_caught(rig):
    r = json.load(open(rig["suite"]))
    r["written_by"] = "me"
    json.dump(r, open(rig["suite"], "w"))
    _refused(rig)
