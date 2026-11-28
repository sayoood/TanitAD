"""Tests of the CL read chain's gates: code/cl_precheck.py (pulled-file controls), the reader's production controls on FAKE rollout outputs (the positive path, which no real rollout exercises before the pull),
and the chain script's table against the reader's flags.

Run:  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -m pytest -q tests/test_cl_chain_gates.py --basetemp F:/TanitAD-scratch/motion_c/cl_read/pt/basetemp_gates        (CPU, ~1 minute; basetemp parent must exist)
The precheck fixture builds a SYNTHETIC run dir on F: from the level-ON smoke checkpoint (raw/SMOKE_LEVEL_ON.json's NZ checkpoint: cfg level_head on, w_level 0.1, e_level non-zero) and the real C0 argv; it is NOT a CL checkpoint
and nothing here reads, writes or fakes a file in E:/Projects/TanitAD-artifacts/v8f-recover/.
"""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
ART = "E:/Projects/TanitAD-artifacts"
NZ = "F:/TanitAD-scratch/motion_c/cl_read/smoke/lvl/ck_NZ/ckpt_last.pt"
ZERO = "F:/TanitAD-scratch/motion_c/cl_read/smoke/lvl/ck_ZERO/ckpt_last.pt"
BASH = "C:/Program Files/Git/bin/bash.exe"
RDB_TEST = "F:/TanitAD-scratch/motion_c/cl_read/pt/rdb_gates"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load("cl_precheck_under_test", os.environ.get("CL_PRECHECK_PATH", f"{PK}/code/cl_precheck.py"))
C = _load("cl_read_gates_under_test", os.environ.get("CL_READ_PATH", f"{PK}/code/cl_read.py"))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


@pytest.fixture(scope="module")
def rdb():
    """A synthetic pulled-run tree: both seeds carry the level-ON smoke checkpoint, C0's real argv + the two level tokens, a SHA256SUMS line and the box-2 marker."""
    assert os.path.isfile(NZ), "run code/smoke_level_on.py first (it builds the level-ON smoke checkpoints)"
    shutil.rmtree(RDB_TEST, ignore_errors=True)
    os.makedirs(f"{RDB_TEST}/cl_markers")
    digest = sha(NZ)
    for s in (0, 1):
        c0 = json.load(open(f"{ART}/v8f-recover/M2b_s{s}_recft_C0/argv.json", encoding="utf-8"))
        os.makedirs(f"{RDB_TEST}/M2b_s{s}_recft_C0")
        json.dump(c0, open(f"{RDB_TEST}/M2b_s{s}_recft_C0/argv.json", "w", encoding="utf-8"))
        d = f"{RDB_TEST}/M2b_s{s}_recft_CL"
        os.makedirs(d)
        shutil.copyfile(NZ, f"{d}/ckpt_last.pt")
        open(f"{d}/SHA256SUMS.txt", "w", encoding="utf-8", newline="\n").write(f"{digest} *ckpt_last.pt\n")
        json.dump({"argv": c0["argv"] + ["--level-head", "on", "--w-level", "0.1"], "sha": "x"}, open(f"{d}/argv.json", "w", encoding="utf-8"))
        json.dump({"tag": f"CL_{s}", "evidence": {"ckpt_sha256": digest}}, open(f"{RDB_TEST}/cl_markers/RUN_CL_{s}.done", "w", encoding="utf-8"))
    open(f"{RDB_TEST}/FROZEN_W.txt", "w", encoding="utf-8", newline="\n").write("# the W frozen before launch\nW=0.1\n")
    return RDB_TEST


def pre(rdb, tmp_path, w=None):
    raw = tmp_path / "raw"
    rc = P.main(["--rdb", rdb, "--frozen-w", w or f"{rdb}/FROZEN_W.txt", "--raw", str(raw)])
    return rc, json.load(open(raw / "CL_PRECHECK.json", encoding="utf-8"))


class edit:
    """Edit a small file of the synthetic tree for one test and restore it afterwards."""
    def __init__(self, path, fn):
        self.path, self.fn = path, fn

    def __enter__(self):
        self.old = open(self.path, "rb").read()
        open(self.path, "wb").write(self.fn(self.old.decode("utf-8")).encode("utf-8"))

    def __exit__(self, *a):
        open(self.path, "wb").write(self.old)


# ------------------------------------------------------------------------------------------------ precheck
def test_precheck_passes_on_a_consistent_tree_and_reports_the_trained_lever(rdb, tmp_path):
    rc, r = pre(rdb, tmp_path)
    assert rc == 0 and r["PASS"] is True and r["failures"] == [] and r["w"] == "0.1"
    for k in ("s0", "s1"):
        assert r["seeds"][k]["cfg_level_head"] == "on" and r["seeds"][k]["e_level_abs_sum"] > 0 and r["seeds"][k]["e_level_all_zero"] is False
        assert r["seeds"][k]["argv_audit"]["diff"]["added"] == {"--level-head": "on", "--w-level": "0.1"}


def test_precheck_refuses_an_absent_checkpoint(rdb, tmp_path):
    p = f"{rdb}/M2b_s1_recft_CL/ckpt_last.pt"
    os.rename(p, p + ".away")
    try:
        rc, r = pre(rdb, tmp_path)
    finally:
        os.rename(p + ".away", p)
    assert rc == 5 and r["PASS"] is False and any("ckpt_last.pt is absent" in f for f in r["failures"])


def test_precheck_refuses_a_missing_marker(rdb, tmp_path):
    p = f"{rdb}/cl_markers/RUN_CL_0.done"
    os.rename(p, p + ".away")
    try:
        rc, r = pre(rdb, tmp_path)
    finally:
        os.rename(p + ".away", p)
    assert rc == 5 and any("marker" in f and "absent" in f for f in r["failures"])


def test_precheck_refuses_sha_mismatch_with_sums_or_marker(rdb, tmp_path):
    with edit(f"{rdb}/M2b_s0_recft_CL/SHA256SUMS.txt", lambda t: "0" * 64 + t[64:]):
        rc, r = pre(rdb, tmp_path)
    assert rc == 8 and any("SHA256SUMS.txt" in f for f in r["failures"])
    with edit(f"{rdb}/cl_markers/RUN_CL_1.done", lambda t: t.replace(sha(NZ), "f" * 64)):
        rc, r = pre(rdb, tmp_path)
    assert rc == 8 and any("box-2 marker" in f for f in r["failures"])
    with edit(f"{rdb}/cl_markers/RUN_CL_1.done", lambda t: json.dumps({"evidence": {}})):
        rc, r = pre(rdb, tmp_path)
    assert rc == 8 and any("no evidence.ckpt_sha256" in f for f in r["failures"])


def test_precheck_argv_audit_is_exact(rdb, tmp_path):
    with edit(f"{rdb}/M2b_s0_recft_CL/argv.json", lambda t: t.replace('"--lr", "1e-4"', '"--lr", "2e-4"')):           # a changed C0 token
        rc, r = pre(rdb, tmp_path)
    assert rc == 8 and r["seeds"]["s0"]["argv_audit"]["nothing_changed"] is False
    with edit(f"{rdb}/M2b_s0_recft_CL/argv.json", lambda t: t.replace('"--level-head", "on"', '"--level-head", "on", "--extra-flag", "1"')):   # an extra token
        rc, r = pre(rdb, tmp_path)
    assert rc == 8 and r["seeds"]["s0"]["argv_audit"]["added_exactly_the_two_level_tokens"] is False
    with edit(f"{rdb}/M2b_s1_recft_CL/argv.json", lambda t: t.replace('"--w-level", "0.1"', '"--w-level", "0.2"')):    # W is not the frozen W
        rc, r = pre(rdb, tmp_path)
    assert rc == 8 and r["seeds"]["s1"]["argv_audit"]["w_level_is_the_frozen_W"] is False


def test_precheck_refuses_a_missing_or_bad_frozen_W(rdb, tmp_path):
    bad = tmp_path / "bad_w.txt"
    bad.write_text("# comment\n-3\n", encoding="utf-8")
    rc, r = pre(rdb, tmp_path, w=str(bad))
    assert rc == 8 and any("frozen W" in f for f in r["failures"])
    rc, r = pre(rdb, tmp_path, w=str(tmp_path / "absent.txt"))
    assert rc == 8 and any("frozen W" in f for f in r["failures"])
    ok = tmp_path / "w.txt"
    ok.write_text("0.1\n", encoding="utf-8")
    assert pre(rdb, tmp_path, w=str(ok))[0] == 0                                                                        # the bare number form is accepted
    assert P.read_frozen_w(str(ok)) == "0.1"


def test_precheck_refuses_a_checkpoint_whose_cfg_has_no_level_head(rdb, tmp_path):
    c0 = f"{ART}/v8f-recover/M2b_s0_recft_C0/ckpt_last.pt"                                                                # the real C0: level_head off
    d = f"{rdb}/M2b_s0_recft_CL"
    shutil.copyfile(f"{d}/ckpt_last.pt", f"{d}/ckpt_last.pt.keep")
    try:
        shutil.copyfile(c0, f"{d}/ckpt_last.pt")
        digest = sha(f"{d}/ckpt_last.pt")
        with edit(f"{d}/SHA256SUMS.txt", lambda t: f"{digest} *ckpt_last.pt\n"), edit(f"{rdb}/cl_markers/RUN_CL_0.done", lambda t: json.dumps({"evidence": {"ckpt_sha256": digest}})):
            rc, r = pre(rdb, tmp_path)
    finally:
        os.replace(f"{d}/ckpt_last.pt.keep", f"{d}/ckpt_last.pt")
    assert rc == 8 and any("level_head" in f and "not 'on'" in f for f in r["failures"]) and any("lacks level-head keys" in f for f in r["failures"])


def test_precheck_reports_a_vacuous_head_without_refusing(rdb, tmp_path):
    d = f"{rdb}/M2b_s0_recft_CL"
    shutil.copyfile(f"{d}/ckpt_last.pt", f"{d}/ckpt_last.pt.keep")
    try:
        shutil.copyfile(ZERO, f"{d}/ckpt_last.pt")                                                                         # level head built, e_level exactly zero: the lever never reached O
        digest = sha(f"{d}/ckpt_last.pt")
        with edit(f"{d}/SHA256SUMS.txt", lambda t: f"{digest} *ckpt_last.pt\n"), edit(f"{rdb}/cl_markers/RUN_CL_0.done", lambda t: json.dumps({"evidence": {"ckpt_sha256": digest}})):
            rc, r = pre(rdb, tmp_path)
    finally:
        os.replace(f"{d}/ckpt_last.pt.keep", f"{d}/ckpt_last.pt")
    assert rc == 0 and r["seeds"]["s0"]["e_level_all_zero"] is True and r["seeds"]["s0"]["e_level_abs_sum"] == 0.0     # REPORTED (the reader's headline says so), not a gate


# ------------------------------------------------------------------------------------------------ the reader's production controls on FAKE rollout outputs
def fake_outputs(root, ckpt=("a" * 64, "b" * 64)):
    out, raw = root / "out", root / "raw"
    out.mkdir()
    raw.mkdir()
    for s in (0, 1):
        for role, (scene, feas) in C.NEW_ROLES.items():
            tag = f"{role}_s{s}"
            (out / tag).mkdir()
            flags = C.expected_flags(role)
            json.dump({"tag": tag, "flags": flags + (" --freeze-card-ref x" if scene == "hold_now" else ""), "ckpt": f"E:/x/v8f-recover/M2b_s{s}_recft_CL/ckpt_last.pt", "ckpt_sha256": ckpt[s],
                       "tool_sha256": C.TOOL_SHA, "threads": 12}, open(out / f"done_{tag}.txt", "w", encoding="utf-8"))
            h = {"scene": scene, "smooth": {"mode": "sg72"}, "n_windows": 2399, "kin_mode": "yaw"}
            if feas != "off":
                h["feasible"] = {"mode": feas}
            json.dump({"t1v8f": h}, open(out / tag / f"t1v8f_{tag}.json", "w", encoding="utf-8"))
    json.dump({"PASS": True, "max_plan_diff_m": 0.0, "logged_first64_16t_vs_banked_C0_s0": {"tool_sha256": C.TOOL_SHA}}, open(raw / "KA_NEW.json", "w", encoding="utf-8"))
    json.dump({"PASS": True, "w": "0.1", "seeds": {f"s{s}": {"ckpt_sha256": ckpt[s], "e_level_abs_sum": 1.0} for s in (0, 1)}}, open(raw / "CL_PRECHECK.json", "w", encoding="utf-8"))
    return str(out), str(raw)


@pytest.fixture(scope="module", autouse=True)
def heavy():
    C.init()


def test_reader_controls_pass_on_consistent_outputs(tmp_path):
    out, raw = fake_outputs(tmp_path)
    ctl = C.run_controls(out, raw)
    assert all(v["PASS"] for v in ctl.values()), {k: v for k, v in ctl.items() if not v["PASS"]}
    assert len([k for k in ctl if k.startswith("rollout_")]) == 10


@pytest.mark.parametrize("what", ["flags", "scene_header", "feasible_header", "tool_sha", "n_windows", "ckpt_two_shas", "wrong_run_dir", "ka_new", "precheck_ckpt", "precheck_fail", "missing_marker"])
def test_reader_controls_refuse_each_inconsistency(tmp_path, what):
    out, raw = fake_outputs(tmp_path)
    o = lambda tag: os.path.join(out, tag)
    rj = lambda p: json.load(open(p, encoding="utf-8"))
    wj = lambda p, d: json.dump(d, open(p, "w", encoding="utf-8"))
    if what == "flags":
        d = rj(o("FZCL_s1/../done_FZCL_s1.txt")); d["flags"] = "--smooth sg72 --scene logged"; wj(o("done_FZCL_s1.txt"), d); bad = "rollout_FZCL_s1"
    elif what == "scene_header":
        p = o("RICL_s0/t1v8f_RICL_s0.json"); d = rj(p); d["t1v8f"]["scene"] = "logged"; wj(p, d); bad = "rollout_RICL_s0"
    elif what == "feasible_header":
        p = o("CLC_s1/t1v8f_CLC_s1.json"); d = rj(p); d["t1v8f"].pop("feasible"); wj(p, d); bad = "rollout_CLC_s1"
    elif what == "tool_sha":
        d = rj(o("done_CL_s0.txt")); d["tool_sha256"] = "0" * 64; wj(o("done_CL_s0.txt"), d); bad = "rollout_CL_s0"
    elif what == "n_windows":
        p = o("RICLC_s0/t1v8f_RICLC_s0.json"); d = rj(p); d["t1v8f"]["n_windows"] = 64; wj(p, d); bad = "rollout_RICLC_s0"
    elif what == "ckpt_two_shas":
        d = rj(o("done_CLC_s0.txt")); d["ckpt_sha256"] = "c" * 64; wj(o("done_CLC_s0.txt"), d); bad = "one_checkpoint_per_seed"
    elif what == "wrong_run_dir":
        d = rj(o("done_CL_s1.txt")); d["ckpt"] = "E:/x/v8f-recover/M2b_s1_recft_C0/ckpt_last.pt"; wj(o("done_CL_s1.txt"), d); bad = "rollout_CL_s1"
    elif what == "ka_new":
        d = rj(os.path.join(raw, "KA_NEW.json")); d["PASS"] = False; wj(os.path.join(raw, "KA_NEW.json"), d); bad = "KA_NEW"
    elif what == "precheck_ckpt":
        d = rj(os.path.join(raw, "CL_PRECHECK.json")); d["seeds"]["s1"]["ckpt_sha256"] = "d" * 64; wj(os.path.join(raw, "CL_PRECHECK.json"), d); bad = "CL_PRECHECK"
    elif what == "precheck_fail":
        d = rj(os.path.join(raw, "CL_PRECHECK.json")); d["PASS"] = False; wj(os.path.join(raw, "CL_PRECHECK.json"), d); bad = "CL_PRECHECK"
    else:
        os.remove(o("done_RICLC_s1.txt")); bad = "rollout_RICLC_s1"
    ctl = C.run_controls(out, raw)
    assert ctl[bad]["PASS"] is False, (what, ctl[bad])
    assert sum(1 for v in ctl.values() if not v["PASS"]) >= 1


# ------------------------------------------------------------------------------------------------ the chain script
def test_chain_table_matches_the_readers_flags_and_priority():
    r = subprocess.run([BASH, f"{PK}/cl_read_chain.sh", "--dry-run"], capture_output=True, text=True)
    assert r.returncode == 0
    table = [l.split("\t") for l in r.stdout.splitlines() if l.count("\t") == 3]
    assert [t[0] for t in table] == ["RICL_s0", "RICL_s1", "CL_s0", "CL_s1", "RICLC_s0", "RICLC_s1", "CLC_s0", "CLC_s1", "FZCL_s0", "FZCL_s1"]     # the deciding contrast first
    for tag, seed, scene, feas in table:
        role = tag.rsplit("_s", 1)[0]
        assert seed == tag[-1] and (scene, feas) == C.NEW_ROLES[role], tag
    src = open(f"{PK}/cl_read_chain.sh", encoding="utf-8").read()
    assert 'local flags="--smooth sg72 --scene $scene"' in src and '[ "$feas" != "off" ] && flags="$flags --feasible $feas"' in src and "--freeze-card-ref" in src
    assert "--replan-steps 10 --nav deadreckon --schedule periodic --limit causal --dump-ticks" in src and "--dump-heads --device cpu" in src
    assert "ckpt_last.pt" in src and "M2b_s${2}_recft_CL" in src and "cl_markers" in open(f"{PK}/code/cl_precheck.py", encoding="utf-8").read()
    assert "CLRO.done" in src and "CLRO.fail" in src and "CLREAD.done" in src


def test_chain_refuses_without_pins_or_checkpoints(tmp_path):
    """The chain run for real REFUSES before any rollout: with PINS.txt absent (gate 0) it exits 2 and writes raw/CLRO.fail; the marker is moved aside afterwards so the package carries no stale one."""
    if os.path.exists(f"{PK}/PINS.txt"):
        pytest.skip("PINS.txt exists: the refusal at gate 1 (absent checkpoints) is exercised by the packaging run (logs/refusal_chain.out)")
    r = subprocess.run([BASH, f"{PK}/cl_read_chain.sh"], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    fail = f"{PK}/raw/CLRO.fail"
    assert r.returncode == 2 and os.path.isfile(fail) and "PINS.txt is absent" in open(fail, encoding="utf-8").read()
    shutil.move(fail, f"{PK}/logs/refusal_pins_absent.CLRO.fail")
