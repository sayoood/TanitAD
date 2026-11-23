"""Tests of read_dscale_eqp.py and podread_dscale_eqp.sh (card PREREG_V8F_PROOF_ADD125). Synthetic data only: tiny worlds, no real state or result file is opened.
Evidence class: MEASURED (synthetic).

What is synthetic and what is injected: the files the controls read (three done files, state records, batch logs, train logs, precheck records, the evaluation-id arrays of the state files, the pytest output
file, the landed run script as a stand-in) are written to disk in the formats the imported reader loads. The per-probe read of the banked chain (tokprobe_read.read_arm) is replaced by `read_probe`, an
injectable function of the reader that returns the same (R, out, ctrl) triple from synthetic gap outputs; the pack and context build is replaced the same way through `build_ctx`. The reader of card ADD123
is imported unchanged (a test asserts its sha256 before and after). Helper builders of that card's tests are reused through an import of its test file.

Run: PYCAP_MB=3000 bash F:/TanitAD-scratch/v8proof/ops/pycap.sh <python> -m pytest test_read_dscale_eqp.py -q -p no:cacheprovider --basetemp <folder>
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = Path(__file__).resolve().parent
OLD_DIR = Path(os.environ.get("D2X_READ_DIR") or (HERE.parent.parent / "add123" / "read"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(OLD_DIR))
import numpy as np
import pytest
import read_dscale_eqp as RE
import test_read_dscale2x as T2

D, N, RP, S0F = RE.D, RE.N, RE.RP, T2.S0F
SEEDS = RE.SEEDS
SCRIPT = HERE / "podread_dscale_eqp.sh"
READER_FILE = Path(RE.READER_PATH)
READER_SHA_AT_IMPORT = hashlib.sha256(READER_FILE.read_bytes()).hexdigest()
B = 400
TOOL_SHA = RE.TOOL_SHA_PREFIX + "0" * 48
must_fail, _assert, wjson, wlog = T2.must_fail, T2._assert, T2.wjson, T2.wlog


def sha_for(*parts):
    return format(zlib.crc32("".join(str(p) for p in parts).encode()), "08x") * 8


# ------------------------------------------------------------------------------------------------ the synthetic world
def make_preds(ctx, variant):
    """label -> gap_m for every probe of the probe table (10 banked, 4 of the run of card ADD123, 4 new). variant: gap | r2only | worse | flat."""
    y = ctx["G15"][:, S0F]
    n = len(y)
    P = {}
    for label, sp in RE.probe_specs().items():
        if sp["sh"]:
            P[label] = np.random.default_rng(zlib.crc32(label.encode())).uniform(10, 60, n)
    for k in SEEDS:
        P[RE.lab("U", k)] = T2.noise(ctx, RE.lab("U", k), 8.0)
        P[RE.lab("Q", k)] = T2.noise(ctx, RE.lab("Q", k), 4.0)
        if variant == "r2only":
            r = np.random.default_rng(zlib.crc32(("tail%d" % k).encode()))
            z, tail = r.normal(size=n), r.random(n) < 0.04
            err = z * np.where(tail, 25.0, 5.0)
            P[RE.lab("C", k)] = y + err
            P[RE.lab("E", k)] = y + err * np.where(tail, 0.7, 1.0)
            continue
        P[RE.lab("C", k)] = T2.noise(ctx, RE.lab("C", k), 6.0)
        if variant == "gap":
            P[RE.lab("E", k)] = T2.noise(ctx, RE.lab("E", k), 2.0)
        elif variant == "worse":
            P[RE.lab("E", k)] = T2.noise(ctx, RE.lab("E", k), 9.0)
        else:
            P[RE.lab("E", k)] = P[RE.lab("C", k)].copy()
    return P


def good_precheck_eqp():
    p = T2.good_precheck()
    p["checks"]["counts"] = {"pass": True, "union_manifest_clips": 14512, "corpus_ids": 4512}
    return p


def build_world(tmp, variant="gap", ctx=None, skip_state=()):
    ctx = ctx or T2.make_ctx()
    w = {"ctx": ctx, "preds": make_preds(ctx, variant), "auroc": {}, "tmp": Path(tmp)}
    root = Path(tmp)
    br, bi, qr, qi, nr, ni = (root / "banked_out", root / "banked_out" / "states", root / "q_out", root / "q_out" / "states", root / "new_out", root / "new_out" / "states")
    for d in (br, bi, qr, qi, nr, ni):
        d.mkdir(parents=True, exist_ok=True)
    lines = []
    for label in RE.BANKED_LABELS:
        folder, arm, steps, sh, corpus = D.ARM_MAP[label]
        sha = sha_for("b", folder, arm)
        T2.write_state(bi, folder, arm, steps, sha)
        lines.append("ckpt %s/%s %s" % (folder, arm, sha))
    (br / "DSCALE_LEAD_DONE.txt").write_text("DSCALE_LEAD_DONE 2026-11-20T00:00:00Z\n" + "\n".join(lines) + "\n", encoding="utf-8")
    hashes = {"1": "aa", "2": "bb", "3": "cc"}
    wjson(qr / "runQ" / "r1_batches.json", {"sampler_chunk": 64, "total_steps": 38328, "hashes": hashes})
    wjson(nr / "runQ" / "r1_batches.json", {"sampler_chunk": 64, "total_steps": 61568, "hashes": hashes})
    qdone, ndone = ["DSCALE_2X_DONE 2026-11-23T00:00:00Z"], ["DSCALE_2X_DONE 2026-11-24T00:00:00Z", "tok_probe_u.py sha256: " + "1" * 64, "tok_probe_u_ck.py sha256: " + TOOL_SHA,
                                                         "dscale_precheck.py sha256: " + "2" * 64, "dscale_clockcheck.py sha256: " + "3" * 64]
    for k in SEEDS:
        arm = "TK16_s%d" % k
        shq, she = sha_for("q", arm), sha_for("e", arm)
        T2.write_state(qi, "runQ", arm, 38328, shq)
        qdone += ["ckpt_step runQ/%s 19164 %s" % (arm, sha_for("q19", arm)), "ckpt runQ/%s/ckpt_last %s" % (arm, shq)]
        wlog(qr / "runQ" / arm / "train_log.jsonl", 38328)
        if arm not in skip_state:
            T2.write_state(ni, "runQ", arm, 61568, she)
        ndone += ["ckpt runQ/%s/ckpt_step19164 %s" % (arm, sha_for("e19", arm)), "ckpt runQ/%s/ckpt_step28746 %s" % (arm, sha_for("e28", arm)), "ckpt runQ/%s/ckpt_last %s" % (arm, she)]
        wlog(nr / "runQ" / arm / "train_log.jsonl", 61568)
    (qr / "DSCALE_2X_DONE.txt").write_text("\n".join(qdone) + "\n", encoding="utf-8")
    (nr / "DSCALE_2X_DONE.txt").write_text("\n".join(ndone) + "\n", encoding="utf-8")
    wjson(nr / "runQ" / "CK_RECORD.json", {"ck_steps": [19164, 28746], "files": {}, "tok_probe_u_sha256": "1" * 64, "tool_sha256": TOOL_SHA})
    wjson(nr / "runQ" / "TRAIN_DONE.json", {"done": True, "steps": 61568, "arms": ["TK16_s%d" % k for k in SEEDS], "p5": {"TK16_s%d" % k: {"last_step": 61568} for k in SEEDS}})
    pre = nr / "precheck"
    wjson(pre / "PRECHECK.json", good_precheck_eqp())
    wjson(pre / "PRECHECK_PLANTED.json", {"pass": False, "checks": {"d1_no_leak": {"pass": False, "planted": True}}})
    wjson(pre / "CLOCK.json", T2.good_clock())
    wjson(pre / "CLOCK_PLANTED.json", T2.good_clock(1))
    wjson(pre / "EVALSET.json", T2.good_evalset())
    wjson(pre / "EVALSET_PLANTED.json", T2.good_evalset(True))
    tout = root / "helper_pytest.out"
    tout.write_text("...........\n11 passed in 3.21s\n", encoding="utf-8")
    old = time.time() - 3600
    os.utime(tout, (old, old))
    script = root / "landed" / "pod_dscale_lead_2x.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("#!/bin/bash\necho the landed run script stand-in\n", encoding="utf-8")
    w.update({"banked_run": br, "banked_in": bi, "q_run": qr, "q_in": qi, "new_run": nr, "new_in": ni, "test_out": tout, "out": nr / "read", "script": script})
    return w


def argv(w, mode, out=None, before=None):
    return [mode, "--banked-run-dir", str(w["banked_run"]), "--banked-in-dir", str(w["banked_in"]), "--banked-done", str(w["banked_run"] / "DSCALE_LEAD_DONE.txt"),
            "--q-run-dir", str(w["q_run"]), "--q-in-dir", str(w["q_in"]), "--q-done", str(w["q_run"] / "DSCALE_2X_DONE.txt"),
            "--new-run-dir", str(w["new_run"]), "--new-in-dir", str(w["new_in"]), "--new-done", str(w["new_run"] / "DSCALE_2X_DONE.txt"), "--pack-dir", str(w["tmp"]),
            "--out-dir", str(out or w["out"]), "--test-out", str(w["test_out"]), "--test-before", str(before or time.time()), "--B", str(B)]


@pytest.fixture
def run(monkeypatch, capsys, tmp_path):
    def go(w, mode, **kw):
        monkeypatch.setattr(RE, "build_ctx", lambda a: w["ctx"])
        monkeypatch.setattr(RE, "read_probe", T2.fake_reader(w))
        monkeypatch.setattr(RE, "RUN_SCRIPT_DEFAULT", str(w["script"]))
        monkeypatch.setattr(RE, "RUN_SCRIPT_SHA_PREFIX", hashlib.sha256(Path(w["script"]).read_bytes()).hexdigest()[:16])
        capsys.readouterr()
        rc = RE.main(argv(w, mode, **kw))
        return rc, capsys.readouterr().out
    return go


def controls_json(w):
    return json.loads(Path(RE.X.latest_path(str(w["out"]), RE.CONTROLS_FILE)).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ tests
def test_second_instance_of_the_imported_reader_carries_the_new_step_count_and_the_first_is_untouched():
    assert RE.X is not T2.R2 and RE.X.__name__ == "read_dscale2x_eqp"
    assert RE.X.STEPS_Q == 61568 and RE.X.NEW_STEPS == (61568,) and T2.R2.STEPS_Q == 38328 and T2.R2.NEW_STEPS == (19164, 28746, 38328)
    assert RE.X.D is T2.R2.D
    must_fail(lambda: _assert(RE.X.STEPS_Q == 38328))


def test_a_changed_copy_of_the_imported_reader_is_refused_at_import(tmp_path):
    bad = tmp_path / "read_dscale2x.py"
    bad.write_bytes(READER_FILE.read_bytes() + b"\n# one more line\n")
    code = "import sys; sys.path.insert(0, %r); import read_dscale_eqp" % str(HERE)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", cwd=os.environ.get("EQP_TEST_CWD") or str(tmp_path), env={**os.environ, "D2X_READER": str(bad)})
    assert r.returncode != 0 and "REFUSED" in r.stderr
    ok = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", cwd=os.environ.get("EQP_TEST_CWD") or str(tmp_path), env={**os.environ, "D2X_READER": str(READER_FILE)})
    assert ok.returncode == 0, ok.stderr[-400:]


def test_the_landed_files_of_card_ADD123_carry_the_pinned_prefixes_where_this_box_holds_them():
    run_dir = HERE.parent.parent / "add123" / "run"
    if not (run_dir / "pod_dscale_lead_2x.sh").exists():
        pytest.skip("the landed run folder is not on this box")
    assert hashlib.sha256((run_dir / "pod_dscale_lead_2x.sh").read_bytes()).hexdigest().startswith(RE.RUN_SCRIPT_SHA_PREFIX)
    assert hashlib.sha256((run_dir / "tok_probe_u_ck.py").read_bytes()).hexdigest().startswith(RE.TOOL_SHA_PREFIX)
    assert READER_SHA_AT_IMPORT.startswith(RE.READER_SHA_PREFIX)


def test_probe_table_is_the_card_map():
    sp = RE.probe_specs()
    assert len(sp) == 10 + 4 + 4 and "P16U2_s0" not in sp
    for label in RE.BANKED_LABELS:
        assert (sp[label]["folder"], sp[label]["arm"], sp[label]["steps"], sp[label]["sh"]) == D.ARM_MAP[label][:4]
    assert sp["P16Q_s2"]["side"] == "q" and sp["P16Q_s2"]["folder"] == "runQ" and sp["P16Q_s2"]["steps"] == 38328
    assert sp["P16E_s3"]["side"] == "new" and sp["P16E_s3"]["folder"] == "runQ" and sp["P16E_s3"]["steps"] == 61568 and sp["P16E_s3"]["arm"] == "TK16_s3"
    must_fail(lambda: _assert(sp["P16E_s0"]["steps"] == 38328))


def test_clean_world_passes_every_control_and_prints_one_line_each(tmp_path, run):
    w = build_world(tmp_path)
    rc, out = run(w, "controls")
    assert rc == 0, out
    lines = out.strip().splitlines()
    assert lines[:7] == ["ctl e%d PASS" % i for i in range(7)] and lines[-1] == "ZZ DSCALE-EQP CONTROLS ALL-PASS" and lines[7].startswith("beside e1 batch sequence common 3 equal 3")
    cj = controls_json(w)
    assert cj["all_pass"] is True and set(cj["controls"]) == set(RE.CONTROL_NAMES) and all(isinstance(v["pass"], bool) for v in cj["controls"].values())
    assert set(cj["controls"]["e4"]["beside"]) == {"TK16_s%d" % k for k in SEEDS}
    assert "last_step_is_61568" in json.dumps(cj["controls"]["e4"]) and "38328" not in json.dumps(cj["controls"]["e4"]["per_probe"])


def _mut_e0_tool_line(w):
    p = w["new_run"] / "DSCALE_2X_DONE.txt"
    p.write_text(p.read_text(encoding="utf-8").replace("tok_probe_u_ck.py sha256: " + RE.TOOL_SHA_PREFIX, "tok_probe_u_ck.py sha256: 0000000000000000"), encoding="utf-8")


def _mut_e0_ck_record_tool(w):
    wjson(w["new_run"] / "runQ" / "CK_RECORD.json", {"ck_steps": [19164, 28746], "tool_sha256": "f" * 64})


def _mut_e0_script_file(w):
    w["script"].write_text("#!/bin/bash\necho a changed run script\n", encoding="utf-8")        # the prefix the run fixture pins was taken before the change? no: it is taken per call, so plant it in the done file


def _mut_e0_script_line(w):
    p = w["new_run"] / "DSCALE_2X_DONE.txt"
    p.write_text(p.read_text(encoding="utf-8") + "pod_dscale_lead_2x.sh sha256: " + "9" * 64 + "\n", encoding="utf-8")


def _mut_e0_counts(w):
    p = good_precheck_eqp()
    p["checks"]["counts"]["union_manifest_clips"] = 14000
    wjson(w["new_run"] / "precheck" / "PRECHECK.json", p)


def _mut_e0_precheck_planted(w):
    wjson(w["new_run"] / "precheck" / "CLOCK_PLANTED.json", T2.good_clock(0))


def _mut_e0_train_done_steps(w):
    wjson(w["new_run"] / "runQ" / "TRAIN_DONE.json", {"done": True, "steps": 61568, "arms": ["TK16_s%d" % k for k in SEEDS],
                                                       "p5": {"TK16_s%d" % k: {"last_step": 61000 if k == 2 else 61568} for k in SEEDS}})


def _mut_e0_q_done_hash(w):
    p = w["q_run"] / "DSCALE_2X_DONE.txt"
    p.write_text(p.read_text(encoding="utf-8").replace("ckpt runQ/TK16_s1/ckpt_last ", "ckpt runQ/TK16_s1/ckpt_last 00"), encoding="utf-8")


def _mut_e0_banked_done(w):
    p = w["banked_run"] / "DSCALE_LEAD_DONE.txt"
    p.write_text(p.read_text(encoding="utf-8").replace("ckpt runC/TK16_s1 ", "ckpt runC/TK16_s1 00"), encoding="utf-8")


def _mut_e1_first_record(w):
    wlog(w["new_run"] / "runQ" / "TK16_s1" / "train_log.jsonl", 61568, step1_bce=0.7010)


def _mut_e1_first_record_gap_term(w):
    wlog(w["new_run"] / "runQ" / "TK16_s3" / "train_log.jsonl", 61568, step1_gap=0.0613)


def _mut_e2_dead_e(w):
    w["preds"]["P16E_s1"] = np.random.default_rng(5).uniform(10, 60, len(w["ctx"]["G15"]))


def _mut_e2_auroc(w):
    w["auroc"]["P16C_s2"] = 0.60


def _mut_e2_auroc_e(w):
    w["auroc"]["P16E_s0"] = 0.84


def _mut_e3_alive_permuted(w):
    w["preds"]["P16USH_s0"] = T2.noise(w["ctx"], "live", 1.0)


def _mut_e3_auroc(w):
    w["auroc"]["P16CSH_s0"] = 0.55


def _mut_e4_log_short(w):
    wlog(w["new_run"] / "runQ" / "TK16_s2" / "train_log.jsonl", 61568, stop_at=50000)


def _mut_e4_state_step(w):
    p = w["new_in"] / "runQ" / "TK16_s1_eval.npz.json"
    r = json.loads(p.read_text(encoding="utf-8"))
    r["step"] = 38328
    wjson(p, r)


def _mut_e4_state_hash(w):
    p = w["new_in"] / "runQ" / "TK16_s0_eval.npz.json"
    r = json.loads(p.read_text(encoding="utf-8"))
    r["ckpt_sha256"] = "00" + r["ckpt_sha256"][2:]
    wjson(p, r)


def _mut_e4_no_gap_in_a_full_batch(w):
    p = w["new_run"] / "runQ" / "TK16_s3" / "train_log.jsonl"
    lines = p.read_text(encoding="utf-8").splitlines()
    r = json.loads(lines[40])
    r["n_gap"] = 0.0
    lines[40] = json.dumps(r)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mut_e4_header_total(w):
    p = w["new_run"] / "runQ" / "TK16_s0" / "train_log.jsonl"
    lines = p.read_text(encoding="utf-8").splitlines()
    h = json.loads(lines[0])
    h["total"] = 38328
    p.write_text("\n".join([json.dumps(h)] + lines[1:]) + "\n", encoding="utf-8")


def _mut_e5_tests_fail(w):
    w["test_out"].write_text("....F....\n1 failed, 10 passed in 3.2s\n", encoding="utf-8")
    old = time.time() - 3600
    os.utime(w["test_out"], (old, old))


def _mut_e5_tests_newer(w):
    os.utime(w["test_out"], (time.time() + 1000, time.time() + 1000))


def _mut_e6_ids(w):
    ids = list(T2.EVAL_IDS)
    ids[3], ids[4] = ids[4], ids[3]
    T2.write_state(w["q_in"], "runQ", "TK16_s2", 38328, sha_for("q", "TK16_s2"), ids=ids)


def _mut_e6_ids_new(w):
    T2.write_state(w["new_in"], "runQ", "TK16_s0", 61568, sha_for("e", "TK16_s0"), ids=T2.EVAL_IDS[:-1] + ["clipXXX"])


MUTATIONS = [("e0", _mut_e0_tool_line), ("e0", _mut_e0_ck_record_tool), ("e0", _mut_e0_script_line), ("e0", _mut_e0_counts), ("e0", _mut_e0_precheck_planted), ("e0", _mut_e0_train_done_steps),
             ("e0", _mut_e0_q_done_hash), ("e0", _mut_e0_banked_done), ("e1", _mut_e1_first_record), ("e1", _mut_e1_first_record_gap_term), ("e2", _mut_e2_dead_e), ("e2", _mut_e2_auroc),
             ("e2", _mut_e2_auroc_e), ("e3", _mut_e3_alive_permuted), ("e3", _mut_e3_auroc), ("e4", _mut_e4_log_short), ("e4", _mut_e4_state_step), ("e4", _mut_e4_state_hash),
             ("e4", _mut_e4_no_gap_in_a_full_batch), ("e4", _mut_e4_header_total), ("e5", _mut_e5_tests_fail), ("e5", _mut_e5_tests_newer), ("e6", _mut_e6_ids), ("e6", _mut_e6_ids_new)]


BOTH = {"_mut_e4_state_hash": ["e0", "e4"]}          # a state record whose hash differs from the done file breaks the input control and the completeness control alike


@pytest.mark.parametrize("name,mut", MUTATIONS, ids=[m[1].__name__ for m in MUTATIONS])
def test_each_control_fails_on_one_planted_fault_exit_2_and_names_it(tmp_path, run, name, mut):
    w = build_world(tmp_path)
    mut(w)
    rc, out = run(w, "controls")
    assert rc == 2, out
    assert out.strip().splitlines()[-1] == "ZZ DSCALE-EQP CONTROLS FAILED", out
    fails = [ln for ln in out.splitlines() if ln.startswith("ctl ") and "FAIL" in ln]
    want = BOTH.get(mut.__name__, [name])
    assert [ln.split()[1] for ln in fails] == want and all(len(ln) > len("ctl e0 FAIL ") for ln in fails), out
    assert controls_json(w)["all_pass"] is False and controls_json(w)["failed"] == want
    must_fail(lambda: _assert(rc == 0))


def test_a_changed_run_script_file_fails_e0_when_the_done_file_carries_no_script_line(tmp_path, monkeypatch, capsys):
    w = build_world(tmp_path)
    good_prefix = hashlib.sha256(Path(w["script"]).read_bytes()).hexdigest()[:16]
    monkeypatch.setattr(RE, "RUN_SCRIPT_DEFAULT", str(w["script"]))
    monkeypatch.setattr(RE, "RUN_SCRIPT_SHA_PREFIX", good_prefix)
    a = RE.parse(argv(w, "controls"))
    specs = RE.probe_specs()
    assert RE.e0_inputs(a, specs)["pass"] is True
    _mut_e0_script_file(w)
    r = RE.e0_inputs(a, specs)
    assert r["pass"] is False and r["hashes"]["run_script_is_the_landed_one"] is False
    r2 = RE.e0_inputs(a, specs, run_script_path=str(tmp_path / "nowhere.sh"))
    assert r2["pass"] is False


def test_missing_state_of_a_new_arm_fails_e4_and_names_nothing(tmp_path, run):
    w = build_world(tmp_path, skip_state=("TK16_s2",))
    rc, out = run(w, "controls")
    assert rc == 2 and "ctl e4 FAIL" in out and "ZZ DSCALE-EQP CONTROLS FAILED" in out
    assert "NAME" not in out


def test_a_full_batch_sequence_difference_is_printed_and_not_judged(tmp_path, run):
    w = build_world(tmp_path)
    wjson(w["new_run"] / "runQ" / "r1_batches.json", {"sampler_chunk": 64, "total_steps": 61568, "hashes": {"1": "aa", "2": "bb", "3": "XX"}})
    rc, out = run(w, "controls")
    assert rc == 0 and "beside e1 batch sequence common 3 equal 2" in out
    b = controls_json(w)["controls"]["e1"]["beside"]["batch_sequence_first_38328_steps_printed_not_judged"]
    assert b["first_differing_entry"] == "3"


def test_a_first_record_within_1e_4_passes_e1(tmp_path, run):
    w = build_world(tmp_path)
    wlog(w["new_run"] / "runQ" / "TK16_s0" / "train_log.jsonl", 61568, step1_bce=0.70005)
    rc, out = run(w, "controls")
    assert rc == 0, out


def test_controls_mode_writes_and_prints_no_measure_and_no_name(tmp_path, run):
    w = build_world(tmp_path)
    rc, out = run(w, "controls")
    assert rc == 0
    text = Path(RE.X.latest_path(str(w["out"]), RE.CONTROLS_FILE)).read_text(encoding="utf-8")
    assert all(re.match(r"^(ctl e\d (PASS|FAIL .*)|beside e1 .*|ZZ DSCALE-EQP .*)$", ln) for ln in out.strip().splitlines())
    for bad in ("LEAD-READ", "SCALES", "R2-ONLY", '"E"', '"T"', "two_level", "per_seed", "mean_gap_mae", "ci95", "mae_ratio", "upper_end"):
        assert bad not in text and bad not in out, bad
    for label in ("P16E_s0", "P16E_s1", "P16C_s0"):
        s = D.summ_one(w["ctx"], {"gap_m": w["preds"][label]})
        assert repr(s["M2"]["r2"])[:9] not in text and repr(s["MAE"]["mae_m"])[:9] not in text
    assert not any(f.name.startswith(("DSCALE_EQP_RESULT", "DSCALE_EQP_VERDICT")) for f in w["out"].iterdir())
    must_fail(lambda: _assert("ci95" in text))


def test_a_second_controls_call_writes_a_new_file_and_overwrites_nothing(tmp_path, run):
    w = build_world(tmp_path)
    run(w, "controls")
    first = (w["out"] / RE.CONTROLS_FILE).read_bytes()
    run(w, "controls")
    assert (w["out"] / RE.CONTROLS_FILE).read_bytes() == first and (w["out"] / "DSCALE_EQP_CONTROLS_2.json").exists()


def test_names_refuses_without_a_passing_controls_file_and_after_an_input_changed_and_on_a_missing_input(tmp_path, run):
    w = build_world(tmp_path)
    rc, out = run(w, "names")
    assert rc == 3 and "REFUSED" in out and not (w["out"] / RE.RESULT_FILE).exists()
    _mut_e2_dead_e(w)
    assert run(w, "controls")[0] == 2
    rc, out = run(w, "names")
    assert rc == 3 and "NAME" not in out and not (w["out"] / RE.RESULT_FILE).exists()
    w2 = build_world(tmp_path / "w2")
    assert run(w2, "controls")[0] == 0
    p = w2["new_in"] / "runQ" / "TK16_s0_eval.npz.json"
    r = json.loads(p.read_text(encoding="utf-8"))
    r["note"] = "changed after the controls"
    wjson(p, r)
    rc, out = run(w2, "names")
    assert rc == 3 and not (w2["out"] / RE.RESULT_FILE).exists()
    w3 = build_world(tmp_path / "w3")
    assert run(w3, "controls")[0] == 0
    assert run(w3, "names")[0] == 0 and (w3["out"] / RE.RESULT_FILE).exists() and (w3["out"] / RE.VERDICT_FILE).exists()
    w4 = build_world(tmp_path / "w4")
    argv4 = argv(w4, "controls")
    argv4[argv4.index("--q-done") + 1] = str(tmp_path / "w4" / "nothing_here.txt")
    assert RE.main(argv4) == 3


@pytest.mark.parametrize("variant,expected", [("gap", "LEAD-READ-EQP-SCALES-GAP"), ("r2only", "LEAD-READ-EQP-R2-ONLY"), ("worse", "LEAD-READ-EQP-WORSE"), ("flat", "LEAD-READ-EQP-FLAT")])
def test_each_of_the_four_names_is_reached_by_planted_data(tmp_path, run, variant, expected):
    w = build_world(tmp_path, variant)
    assert run(w, "controls")[0] == 0
    rc, out = run(w, "names")
    assert rc == 0 and out.strip().splitlines()[-1] == "ZZ DSCALE-EQP NAME " + expected, out
    res = json.loads((w["out"] / RE.RESULT_FILE).read_text(encoding="utf-8"))
    n, M = res["names"], res["measures"]
    assert n["name"] == expected and set(M) >= {"E", "T", "beside"} and set(res) >= {"names", "measures", "probes", "input_hashes"} and len(res["probes"]) == 18
    assert set(M["E"]) == {"M2", "MAE"} and set(M["T"]) == {"M2", "MAE"} and {"Q_minus_U", "S_Q_minus_C_card_ADD123", "known_value_beside"} <= set(M["beside"])
    md = (w["out"] / RE.VERDICT_FILE).read_text(encoding="utf-8")
    assert expected in md and "T0" in md and "STRATEGIC" in md and "the gain the read leaves open" in md
    assert ("%+.4f" % M["E"]["M2"]["ci95"][1]) in md
    e = M["E"]["M2"]
    if variant == "gap":
        assert e["ci95"][0] > 0 and e["n_pos"] == 4 and n["gap_error_bar_met"] and n["mae_ratio_E_to_C"] < 0.90
    if variant == "r2only":
        assert e["ci95"][0] > 0 and e["n_pos"] >= 3 and not n["gap_error_bar_met"] and n["mae_ratio_E_to_C"] > 0.90, (e, n)
    if variant == "worse":
        assert e["ci95"][1] < 0 and e["n_neg"] == 4
    if variant == "flat":
        assert e["mean"] == 0.0 and e["ci95"] == [0.0, 0.0] and M["T"]["M2"]["mean"] != 0.0
    must_fail(lambda: _assert(n["name"] == "LEAD-READ-EQP-FLAT" and variant != "flat"))


def test_decide_clauses_individually():
    def tl(mean, lo, hi, npos=4, nneg=0):
        return {"mean": mean, "ci95": [lo, hi], "n_pos": npos, "n_neg": nneg}
    E = {"M2": tl(0.05, 0.01, 0.09)}
    assert RE.decide(E, {"E": 2.0, "C": 3.0})["name"] == RE.NAMES[0]
    assert RE.decide(E, {"E": 2.7, "C": 3.0})["name"] == RE.NAMES[0]                                              # at the bar
    assert RE.decide(E, {"E": 2.71, "C": 3.0})["name"] == RE.NAMES[1]                                             # just above the bar
    assert RE.decide({"M2": tl(0.05, 0.01, 0.09, 3, 1)}, {"E": 2.0, "C": 3.0})["name"] == RE.NAMES[0]             # three of four
    assert RE.decide({"M2": tl(0.05, 0.01, 0.09, 2, 2)}, {"E": 2.0, "C": 3.0})["name"] == RE.NAMES[3]             # two of four
    assert RE.decide({"M2": tl(0.05, -0.01, 0.09)}, {"E": 2.0, "C": 3.0})["name"] == RE.NAMES[3]                  # interval touches 0
    assert RE.decide({"M2": tl(-0.05, -0.09, -0.01, 0, 4)}, {"E": 3.0, "C": 3.0})["name"] == RE.NAMES[2]
    assert RE.decide({"M2": tl(-0.05, -0.09, -0.01, 1, 3)}, {"E": 3.0, "C": 3.0})["name"] == RE.NAMES[2]          # three of four below
    assert RE.decide({"M2": tl(-0.05, -0.09, -0.01, 2, 2)}, {"E": 3.0, "C": 3.0})["name"] == RE.NAMES[3]
    assert RE.decide({"M2": tl(-0.05, -0.09, 0.01, 0, 4)}, {"E": 3.0, "C": 3.0})["name"] == RE.NAMES[3]
    assert RE.decide(E, {"E": 2.0, "C": 3.0})["e_upper_end_gap_r2"] == 0.09
    must_fail(lambda: _assert(RE.decide(E, {"E": 2.8, "C": 3.0})["name"] == RE.NAMES[0]))


def test_identity_the_P16Q_values_equal_those_the_imported_reader_computes_itself(tmp_path, monkeypatch, capsys):
    wi = T2.build_world(tmp_path / "imp", "gap")
    monkeypatch.setattr(T2.R2, "build_ctx", lambda a: wi["ctx"])
    monkeypatch.setattr(T2.R2, "read_probe", T2.fake_reader(wi))
    assert T2.R2.main(T2.argv(wi, "controls")) == 0
    assert T2.R2.main(T2.argv(wi, "names")) == 0
    imp = json.loads((wi["out"] / T2.R2.RESULT_FILE).read_text(encoding="utf-8"))
    w = build_world(tmp_path / "eqp", "gap", ctx=wi["ctx"])
    for label in list(w["preds"]):
        if label[:5] in ("P16C_", "P16U_") or label.endswith("SH_s0"):
            w["preds"][label] = wi["preds"][label]
    for k in SEEDS:
        w["preds"]["P16Q_s%d" % k] = wi["preds"]["P16Q_s%d@38328" % k]
    monkeypatch.setattr(RE, "build_ctx", lambda a: w["ctx"])
    monkeypatch.setattr(RE, "read_probe", T2.fake_reader(w))
    monkeypatch.setattr(RE, "RUN_SCRIPT_DEFAULT", str(w["script"]))
    monkeypatch.setattr(RE, "RUN_SCRIPT_SHA_PREFIX", hashlib.sha256(Path(w["script"]).read_bytes()).hexdigest()[:16])
    assert RE.main(argv(w, "controls")) == 0
    assert RE.main(argv(w, "names")) == 0
    mine = json.loads((w["out"] / RE.RESULT_FILE).read_text(encoding="utf-8"))
    for k in SEEDS:
        a, b = imp["probes"]["P16Q_s%d@38328" % k], mine["probes"]["P16Q_s%d" % k]
        assert a["M2"]["r2"] == b["M2"]["r2"] and a["MAE"]["mae_m"] == b["MAE"]["mae_m"] and a["M2"]["r2_ci95"] == b["M2"]["r2_ci95"]
    sm, im = mine["measures"]["beside"]["S_Q_minus_C_card_ADD123"], imp["measures"]
    assert sm["M2"]["mean"] == im["S"]["M2"]["mean"] and sm["M2"]["ci95"] == im["S"]["M2"]["ci95"] and sm["M2"]["per_seed"] == im["S"]["M2"]["per_seed"]
    assert sm["MAE"]["mean"] == im["S"]["MAE"]["mean"] and sm["MAE"]["ci95"] == im["S"]["MAE"]["ci95"]
    qu = mine["measures"]["beside"]["Q_minus_U"]
    assert qu["M2"]["mean"] == im["beside"]["Q_minus_banked_U_M2"]["mean"] and qu["M2"]["ci95"] == im["beside"]["Q_minus_banked_U_M2"]["ci95"]
    must_fail(lambda: _assert(mine["probes"]["P16Q_s0"]["M2"]["r2"] == mine["probes"]["P16E_s0"]["M2"]["r2"]))


def test_driver_text_never_removes_moves_or_kills_and_refuses_bad_calls(tmp_path):
    txt = SCRIPT.read_text(encoding="utf-8")
    code = "\n".join(ln for ln in txt.splitlines() if not ln.lstrip().startswith("#"))
    for word in ("rm ", "rm\t", "mv ", "kill", "rmdir", "unlink", "ssh", "scp"):
        assert word not in code, word
    must_fail(lambda: _assert("rm " in code))
    bash = shutil.which("bash")
    assert bash
    d = tmp_path / "drv"
    d.mkdir()
    shutil.copy(str(SCRIPT), str(d / "podread_dscale_eqp.sh"))
    shutil.copy(str(HERE / "read_dscale_eqp.py"), str(d / "read_dscale_eqp.py"))
    sp = str(d / "podread_dscale_eqp.sh")
    env = {**os.environ, "PY": sys.executable, "DSCALE_LEAD_KIT_DIR": str(tmp_path / "nokit")}

    def call(*args):
        return subprocess.run([bash, sp, *args], capture_output=True, text=True, encoding="utf-8", env=env)
    r = call("bogus")
    assert r.returncode == 3 and "STOP" in r.stdout
    r = call("controls")
    assert r.returncode == 3 and "NO-READ-ARGS-FILE" in r.stdout
    none = tmp_path / "none"
    (d / "read_args.txt").write_text("--new-run-dir\n%s\n--new-done\n%s\n" % (none.as_posix(), (none / "DSCALE_2X_DONE.txt").as_posix()), encoding="utf-8")
    r = call("controls")
    assert r.returncode == 3 and "NO-DONE-MARKER" in r.stdout
    assert not none.exists()                                                                       # nothing was created before the done file was found
    none.mkdir()
    (none / "DSCALE_2X_DONE.txt").write_text("DSCALE_2X_DONE x\n", encoding="utf-8")
    r = call("names")
    assert r.returncode == 3 and "NO-BANKED-READER" in r.stdout and (none / "read").is_dir()
    (d / "read_args.txt").write_text("--new-run-dir\n%s\n--new-done\n%s\n--out-dir\nx\n" % (none.as_posix(), (none / "DSCALE_2X_DONE.txt").as_posix()), encoding="utf-8")
    r = call("controls")
    assert r.returncode == 3 and "--out-dir is set by this script" in r.stdout
    assert "--out-dir" in txt and "_${n}.out" in txt and "ZZ POD-EQP $MODE DONE rc $rc" in txt and "--test-before" in txt


def test_none_of_the_banned_words_occur_in_the_kit_files():
    parts = [("vo", "id"), ("super", "seded"), ("retr", "acted"), ("form", "erly"), ("corre", "ction"), ("with", "drawn"), ("no lo", "nger"), ("closed", " loop")]
    for f in ("read_dscale_eqp.py", "podread_dscale_eqp.sh", "test_read_dscale_eqp.py", "DSCALE_EQP_READ_DEFS.md"):
        p = HERE / f
        if not p.exists():
            continue
        t = p.read_text(encoding="utf-8").lower()
        for a, b in parts:
            assert (a + b) not in t, (f, a + b)


def test_the_imported_reader_file_is_byte_unchanged_after_the_tests():
    assert hashlib.sha256(READER_FILE.read_bytes()).hexdigest() == READER_SHA_AT_IMPORT
    assert READER_SHA_AT_IMPORT.startswith(RE.READER_SHA_PREFIX)
