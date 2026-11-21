"""Tests of the CLR read chain's gates: code/cl_precheck.py (pulled-file controls), the reader's production controls on FAKE rollout outputs (the positive path, which no real rollout exercises before the pull),
and the chain script's table against the reader's flags.

Run:  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -m pytest -q tests/test_cl_chain_gates.py --basetemp F:/TanitAD-scratch/motion_c/cl_read/pt/basetemp_gates        (CPU, ~1 minute; basetemp parent must exist)
The precheck fixture builds a SYNTHETIC run dir on F: from the level-ON smoke checkpoint (raw/SMOKE_LEVEL_ON.json's NZ checkpoint: cfg level_head on, w_level 0.1, e_level non-zero) and the real C0 argv; it is NOT a CLR checkpoint
and nothing here reads, writes or fakes a file in E:/Projects/TanitAD-artifacts/v8f-recover/.
"""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
ART = "E:/Projects/TanitAD-artifacts"
NZ = "F:/TanitAD-scratch/motion_c/cl_read/smoke/lvl/ck_NZ/ckpt_last.pt"
ZERO = "F:/TanitAD-scratch/motion_c/cl_read/smoke/lvl/ck_ZERO/ckpt_last.pt"
BASH = "C:/Program Files/Git/bin/bash.exe"
RDB_TEST = "F:/TanitAD-scratch/motion_c/clr_read/pt/rdb_gates"
A1 = "E:/Projects/TanitAD-artifacts/dagland/motion_c/cl_readout/code"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load("cl_precheck_under_test", os.environ.get("CLR_PRECHECK_PATH", f"{PK}/code/clr_precheck.py"))
C = _load("cl_read_gates_under_test", os.environ.get("CLR_READ_PATH", f"{PK}/code/clr_read.py"))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


@pytest.fixture(scope="module")
def rdb():
    """A synthetic pulled-run tree: both seeds carry the level-ON smoke checkpoint as CLR (RAPR-P01's real argv + the two level tokens, a SHA256SUMS line, the box-2 marker with the recovery evidence) and a level-OFF
    copy of it as RAPRP01 (the attribution base).  It is NOT a CLR checkpoint and nothing here reads, writes or fakes a file in E:/Projects/TanitAD-artifacts/v8f-recover/."""
    import torch
    assert os.path.isfile(NZ), "run cl_read/code/smoke_level_on.py first (it builds the level-ON smoke checkpoints)"
    shutil.rmtree(RDB_TEST, ignore_errors=True)
    os.makedirs(f"{RDB_TEST}/clr_markers")
    digest = sha(NZ)
    off = f"{RDB_TEST}/_base_level_off.pt"
    ck = torch.load(NZ, map_location="cpu", weights_only=False)
    ck["cfg"] = dict(ck["cfg"], level_head="off")
    torch.save(ck, off)
    del ck
    off_digest = sha(off)
    for s in (0, 1):
        base = json.load(open(f"{ART}/v8f-recover/M2b_s{s}_recft_RAPRP01/argv.json", encoding="utf-8"))
        b = f"{RDB_TEST}/M2b_s{s}_recft_RAPRP01"
        os.makedirs(b)
        json.dump(base, open(f"{b}/argv.json", "w", encoding="utf-8"))
        shutil.copyfile(off, f"{b}/ckpt_last.pt")
        open(f"{b}/SHA256SUMS.txt", "w", encoding="utf-8", newline="\n").write(f"{off_digest} *ckpt_last.pt\n")
        d = f"{RDB_TEST}/M2b_s{s}_recft_CLR"
        os.makedirs(d)
        shutil.copyfile(NZ, f"{d}/ckpt_last.pt")
        open(f"{d}/SHA256SUMS.txt", "w", encoding="utf-8", newline="\n").write(f"{digest} *ckpt_last.pt\n")
        json.dump({"argv": base["argv"] + ["--level-head", "on", "--w-level", "0.1"], "sha": "x"}, open(f"{d}/argv.json", "w", encoding="utf-8"))
        json.dump({"tag": f"CLR_{s}", "evidence": {"ckpt_sha256": digest, "recover_n": 123, "level_head": "on", "w_level": 0.1}}, open(f"{RDB_TEST}/clr_markers/RUN_CLR_{s}.done", "w", encoding="utf-8"))
    open(f"{RDB_TEST}/FROZEN_W.txt", "w", encoding="utf-8", newline="\n").write("# the W frozen before launch\nW=0.1\n")
    return RDB_TEST


def pre(rdb, tmp_path, w=None):
    raw = tmp_path / "raw"
    rc = P.main(["--rdb", rdb, "--frozen-w", w or f"{rdb}/FROZEN_W.txt", "--raw", str(raw)])
    return rc, json.load(open(raw / "CLR_PRECHECK.json", encoding="utf-8"))


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
    p = f"{rdb}/M2b_s1_recft_CLR/ckpt_last.pt"
    os.rename(p, p + ".away")
    try:
        rc, r = pre(rdb, tmp_path)
    finally:
        os.rename(p + ".away", p)
    assert rc == 5 and r["PASS"] is False and any("ckpt_last.pt is absent" in f for f in r["failures"])


def test_precheck_refuses_a_missing_marker(rdb, tmp_path):
    p = f"{rdb}/clr_markers/RUN_CLR_0.done"
    os.rename(p, p + ".away")
    try:
        rc, r = pre(rdb, tmp_path)
    finally:
        os.rename(p + ".away", p)
    assert rc == 5 and any("marker" in f and "absent" in f for f in r["failures"])


def test_precheck_refuses_sha_mismatch_with_sums_or_marker(rdb, tmp_path):
    with edit(f"{rdb}/M2b_s0_recft_CLR/SHA256SUMS.txt", lambda t: "0" * 64 + t[64:]):
        rc, r = pre(rdb, tmp_path)
    assert rc == 8 and any("SHA256SUMS.txt" in f for f in r["failures"])
    with edit(f"{rdb}/clr_markers/RUN_CLR_1.done", lambda t: t.replace(sha(NZ), "f" * 64)):
        rc, r = pre(rdb, tmp_path)
    assert rc == 8 and any("box-2 marker" in f for f in r["failures"])
    with edit(f"{rdb}/clr_markers/RUN_CLR_1.done", lambda t: json.dumps({"evidence": {}})):
        rc, r = pre(rdb, tmp_path)
    assert rc == 8 and any("no evidence.ckpt_sha256" in f for f in r["failures"])


def test_precheck_argv_audit_is_exact(rdb, tmp_path):
    with edit(f"{rdb}/M2b_s0_recft_CLR/argv.json", lambda t: t.replace('"--lr", "1e-4"', '"--lr", "2e-4"')):           # a changed C0 token
        rc, r = pre(rdb, tmp_path)
    assert rc == 8 and r["seeds"]["s0"]["argv_audit"]["nothing_changed"] is False
    with edit(f"{rdb}/M2b_s0_recft_CLR/argv.json", lambda t: t.replace('"--level-head", "on"', '"--level-head", "on", "--extra-flag", "1"')):   # an extra token
        rc, r = pre(rdb, tmp_path)
    assert rc == 8 and r["seeds"]["s0"]["argv_audit"]["added_exactly_the_two_level_tokens"] is False
    with edit(f"{rdb}/M2b_s1_recft_CLR/argv.json", lambda t: t.replace('"--w-level", "0.1"', '"--w-level", "0.2"')):    # W is not the frozen W
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
    d = f"{rdb}/M2b_s0_recft_CLR"
    shutil.copyfile(f"{d}/ckpt_last.pt", f"{d}/ckpt_last.pt.keep")
    try:
        shutil.copyfile(c0, f"{d}/ckpt_last.pt")
        digest = sha(f"{d}/ckpt_last.pt")
        with edit(f"{d}/SHA256SUMS.txt", lambda t: f"{digest} *ckpt_last.pt\n"), edit(f"{rdb}/clr_markers/RUN_CLR_0.done", lambda t: json.dumps({"evidence": {"ckpt_sha256": digest, "recover_n": 123, "level_head": "on", "w_level": 0.1}})):
            rc, r = pre(rdb, tmp_path)
    finally:
        os.replace(f"{d}/ckpt_last.pt.keep", f"{d}/ckpt_last.pt")
    assert rc == 8 and any("level_head" in f and "not 'on'" in f for f in r["failures"]) and any("lacks level-head keys" in f for f in r["failures"])


def test_precheck_reports_a_vacuous_head_without_refusing(rdb, tmp_path):
    d = f"{rdb}/M2b_s0_recft_CLR"
    shutil.copyfile(f"{d}/ckpt_last.pt", f"{d}/ckpt_last.pt.keep")
    try:
        shutil.copyfile(ZERO, f"{d}/ckpt_last.pt")                                                                         # level head built, e_level exactly zero: the lever never reached O
        digest = sha(f"{d}/ckpt_last.pt")
        with edit(f"{d}/SHA256SUMS.txt", lambda t: f"{digest} *ckpt_last.pt\n"), edit(f"{rdb}/clr_markers/RUN_CLR_0.done", lambda t: json.dumps({"evidence": {"ckpt_sha256": digest, "recover_n": 123, "level_head": "on", "w_level": 0.1}})):
            rc, r = pre(rdb, tmp_path)
    finally:
        os.replace(f"{d}/ckpt_last.pt.keep", f"{d}/ckpt_last.pt")
    assert rc == 0 and r["seeds"]["s0"]["e_level_all_zero"] is True and r["seeds"]["s0"]["e_level_abs_sum"] == 0.0     # REPORTED (the reader's headline says so), not a gate


# ------------------------------------------------------------------------------------------------ the reader's production controls on FAKE rollout outputs
def test_precheck_refuses_a_marker_without_recovery_evidence(rdb, tmp_path):
    mk = f"{rdb}/clr_markers/RUN_CLR_1.done"
    for ev in ({"recover_n": 0}, {"recover_n": None}, {"recover_n": True}, {"level_head": "off"}, {"w_level": 0.2}):
        with edit(mk, lambda t, ev=ev: json.dumps({**json.loads(t), "evidence": {**json.loads(t)["evidence"], **ev}})):
            rc, r = pre(rdb, tmp_path)
        assert rc == 8 and any("marker evidence" in f for f in r["failures"]), ev


def test_precheck_refuses_an_argv_without_the_recovery_lever(rdb, tmp_path):
    """The base is C0's argv (no --recover-* token at all) and the CLR argv = it + the level tokens: every other audit check holds, the recovery check alone refuses."""
    c0 = json.load(open(f"{ART}/v8f-recover/M2b_s0_recft_C0/argv.json", encoding="utf-8"))
    with edit(f"{rdb}/M2b_s0_recft_RAPRP01/argv.json", lambda t: json.dumps(c0)), edit(f"{rdb}/M2b_s0_recft_CLR/argv.json", lambda t: json.dumps({"argv": c0["argv"] + ["--level-head", "on", "--w-level", "0.1"]})):
        rc, r = pre(rdb, tmp_path)
    a = r["seeds"]["s0"]["argv_audit"]
    assert rc == 8 and a["recovery_lever_is_RAPRP01s"] is False and a["nothing_changed"] and a["nothing_removed"] and a["added_exactly_the_two_level_tokens"]


def test_precheck_checks_the_attribution_base_checkpoint(rdb, tmp_path):
    rc, r = pre(rdb, tmp_path)
    s0 = r["seeds"]["s0"]
    assert rc == 0 and s0["raprp01_ckpt_sha256"] == s0["raprp01_sha256sums_line"] and s0["raprp01_cfg_level_head"] == "off"
    with edit(f"{rdb}/M2b_s0_recft_RAPRP01/SHA256SUMS.txt", lambda t: "0" * 64 + " *ckpt_last.pt\n"):
        rc, r = pre(rdb, tmp_path)
    assert rc == 8 and any("RAPRP01 ckpt sha256" in f for f in r["failures"])
    d = f"{rdb}/M2b_s1_recft_RAPRP01"
    os.replace(f"{d}/ckpt_last.pt", f"{d}/ckpt_last.pt.keep")
    try:
        shutil.copyfile(NZ, f"{d}/ckpt_last.pt")                                # a level-ON checkpoint as the base: refused
        with edit(f"{d}/SHA256SUMS.txt", lambda t: f"{sha(NZ)} *ckpt_last.pt\n"):
            rc, r = pre(rdb, tmp_path)
        assert rc == 8 and any("level_head on" in f for f in r["failures"])
        os.remove(f"{d}/ckpt_last.pt")                                          # an absent base: exit 5
        rc, r = pre(rdb, tmp_path)
        assert rc == 5 and any("attribution base" in f for f in r["failures"])
    finally:
        if os.path.exists(f"{d}/ckpt_last.pt"):
            os.remove(f"{d}/ckpt_last.pt")
        os.replace(f"{d}/ckpt_last.pt.keep", f"{d}/ckpt_last.pt")


def test_generated_files_equal_their_generators():
    for name in ("make_clr_from_cl.py", "make_clr_precheck.py", "make_clr_readout.py", "make_clr_tests.py"):
        r = subprocess.run([sys.executable, "-I", f"{PK}/code/{name}", "--check"], capture_output=True, text=True)
        assert r.returncode == 0, (name, r.stdout, r.stderr)


def test_readout_clone_differs_from_A1_only_in_the_five_path_lines():
    a = open(f"{A1}/cl_readout.py", encoding="utf-8", newline="").read().splitlines()
    b = open(f"{PK}/code/clr_readout.py", encoding="utf-8", newline="").read().splitlines()
    changed = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(a) == len(b) and len(changed) == 5 and all(("CLR" in y or "clr_read" in y) for _, y in changed)


def fake_outputs(root, ckpt=("a" * 64, "b" * 64), base_ckpt=("e" * 64, "f" * 64)):
    out, raw = root / "out", root / "raw"
    out.mkdir()
    raw.mkdir()
    for s in (0, 1):
        for role, (scene, feas) in C.NEW_ROLES.items():
            tag = f"{role}_s{s}"
            (out / tag).mkdir()
            flags = C.expected_flags(role)
            json.dump({"tag": tag, "flags": flags + (" --freeze-card-ref x" if scene == "hold_now" else ""), "ckpt": f"E:/x/v8f-recover/M2b_s{s}_recft_{C.RUN_OF[role]}/ckpt_last.pt", "ckpt_sha256": ckpt[s] if C.RUN_OF[role] == "CLR" else base_ckpt[s],
                       "tool_sha256": C.TOOL_SHA, "threads": 12}, open(out / f"done_{tag}.txt", "w", encoding="utf-8"))
            h = {"scene": scene, "smooth": {"mode": "sg72"}, "n_windows": 2399, "kin_mode": "yaw"}
            if feas != "off":
                h["feasible"] = {"mode": feas}
            json.dump({"t1v8f": h}, open(out / tag / f"t1v8f_{tag}.json", "w", encoding="utf-8"))
    json.dump({"PASS": True, "max_plan_diff_m": 0.0, "logged_first64_16t_vs_banked_C0_s0": {"tool_sha256": C.TOOL_SHA}}, open(raw / "KA_NEW.json", "w", encoding="utf-8"))
    json.dump({"PASS": True, "w": "0.1", "seeds": {f"s{s}": {"ckpt_sha256": ckpt[s], "raprp01_ckpt_sha256": base_ckpt[s], "e_level_abs_sum": 1.0} for s in (0, 1)}}, open(raw / "CLR_PRECHECK.json", "w", encoding="utf-8"))
    return str(out), str(raw)


@pytest.fixture(scope="module", autouse=True)
def heavy():
    C.init()


def test_reader_controls_pass_on_consistent_outputs(tmp_path, monkeypatch):
    out, raw = fake_outputs(tmp_path)
    monkeypatch.setattr(C, "CL_RAW", raw)
    ctl = C.run_controls(out, raw)
    assert all(v["PASS"] for v in ctl.values()), {k: v for k, v in ctl.items() if not v["PASS"]}
    assert len([k for k in ctl if k.startswith("rollout_")]) == 12


@pytest.mark.parametrize("what", ["flags", "scene_header", "feasible_header", "tool_sha", "n_windows", "ckpt_two_shas", "wrong_run_dir", "ka_new", "precheck_ckpt", "precheck_fail", "missing_marker", "cl_not_clr_run", "attribution_base_ckpt", "attribution_ckpt_vs_precheck"])
def test_reader_controls_refuse_each_inconsistency(tmp_path, what, monkeypatch):
    out, raw = fake_outputs(tmp_path)
    monkeypatch.setattr(C, "CL_RAW", raw)
    o = lambda tag: os.path.join(out, tag)
    rj = lambda p: json.load(open(p, encoding="utf-8"))
    wj = lambda p, d: json.dump(d, open(p, "w", encoding="utf-8"))
    if what == "flags":
        d = rj(o("FZCLR_s1/../done_FZCLR_s1.txt")); d["flags"] = "--smooth sg72 --scene logged"; wj(o("done_FZCLR_s1.txt"), d); bad = "rollout_FZCLR_s1"
    elif what == "scene_header":
        p = o("RICLR_s0/t1v8f_RICLR_s0.json"); d = rj(p); d["t1v8f"]["scene"] = "logged"; wj(p, d); bad = "rollout_RICLR_s0"
    elif what == "feasible_header":
        p = o("CLRC_s1/t1v8f_CLRC_s1.json"); d = rj(p); d["t1v8f"].pop("feasible"); wj(p, d); bad = "rollout_CLRC_s1"
    elif what == "tool_sha":
        d = rj(o("done_CLR_s0.txt")); d["tool_sha256"] = "0" * 64; wj(o("done_CLR_s0.txt"), d); bad = "rollout_CLR_s0"
    elif what == "n_windows":
        p = o("RICLRC_s0/t1v8f_RICLRC_s0.json"); d = rj(p); d["t1v8f"]["n_windows"] = 64; wj(p, d); bad = "rollout_RICLRC_s0"
    elif what == "ckpt_two_shas":
        d = rj(o("done_CLRC_s0.txt")); d["ckpt_sha256"] = "c" * 64; wj(o("done_CLRC_s0.txt"), d); bad = "one_checkpoint_per_seed"
    elif what == "wrong_run_dir":
        d = rj(o("done_CLR_s1.txt")); d["ckpt"] = "E:/x/v8f-recover/M2b_s1_recft_C0/ckpt_last.pt"; wj(o("done_CLR_s1.txt"), d); bad = "rollout_CLR_s1"
    elif what == "ka_new":
        d = rj(os.path.join(raw, "KA_NEW.json")); d["PASS"] = False; wj(os.path.join(raw, "KA_NEW.json"), d); bad = "KA_NEW"
    elif what == "precheck_ckpt":
        d = rj(os.path.join(raw, "CLR_PRECHECK.json")); d["seeds"]["s1"]["ckpt_sha256"] = "d" * 64; wj(os.path.join(raw, "CLR_PRECHECK.json"), d); bad = "CLR_PRECHECK"
    elif what == "precheck_fail":
        d = rj(os.path.join(raw, "CLR_PRECHECK.json")); d["PASS"] = False; wj(os.path.join(raw, "CLR_PRECHECK.json"), d); bad = "CLR_PRECHECK"
    elif what == "cl_not_clr_run":                  # CL's checkpoint path contains CLR's as a prefix: the control must still refuse it
        d = rj(o("done_CLR_s1.txt")); d["ckpt"] = "E:/x/v8f-recover/M2b_s1_recft_CL/ckpt_last.pt"; wj(o("done_CLR_s1.txt"), d); bad = "rollout_CLR_s1"
    elif what == "attribution_base_ckpt":           # RIRAPRP01 rolled from the CLR checkpoint instead of RAPR-P01's
        d = rj(o("done_RIRAPRP01_s0.txt")); d["ckpt"] = "E:/x/v8f-recover/M2b_s0_recft_CLR/ckpt_last.pt"; wj(o("done_RIRAPRP01_s0.txt"), d); bad = "rollout_RIRAPRP01_s0"
    elif what == "attribution_ckpt_vs_precheck":    # the rolled RAPR-P01 checkpoint is not the one the pre-check hashed
        d = rj(os.path.join(raw, "CLR_PRECHECK.json")); d["seeds"]["s0"]["raprp01_ckpt_sha256"] = "9" * 64; wj(os.path.join(raw, "CLR_PRECHECK.json"), d); bad = "CLR_PRECHECK"
    else:
        os.remove(o("done_RICLRC_s1.txt")); bad = "rollout_RICLRC_s1"
    ctl = C.run_controls(out, raw)
    assert ctl[bad]["PASS"] is False, (what, ctl[bad])
    assert sum(1 for v in ctl.values() if not v["PASS"]) >= 1


# ------------------------------------------------------------------------------------------------ the chain script
def test_production_main_runs_end_to_end_on_pre_extracted_arrays(tmp_path, monkeypatch):
    """The production branch of main (controls -> extract -> read_all -> CLR_READ.json + RESULTS.md) on FAKE rollout outputs whose arrays are pre-placed (the extractor returns an existing file): every new role is a copy
    of the base it must equal (RICLR := RIC0 ..., the attribution base RIRAPRP01 := RIC0), so the result is the null read."""
    out, raw = fake_outputs(tmp_path)
    monkeypatch.setattr(C, "CL_RAW", raw)
    arr = tmp_path / "arrays"
    arr.mkdir()
    base = {"CLR": "C0", "CLRC": "C0C", "FZCLR": "FZC0", "RICLR": "RIC0", "RICLRC": "RIC0C", "RIRAPRP01": "RIC0"}
    for s in (0, 1):
        for role, b in base.items():
            shutil.copyfile(f"{C.REUSED[b]}/{b}_s{s}.npz", arr / f"{role}_s{s}.npz")
    md = tmp_path / "RESULTS.md"
    rc = C.main(["--mode", "production", "--out-root", out, "--raw", raw, "--arr-dir", str(arr), "--results-md", str(md), "--work-dir", str(tmp_path / "w")])
    assert rc == 0
    res = json.load(open(f"{raw}/CLR_READ.json", encoding="utf-8"))
    assert "refused" not in res and all(v["PASS"] for v in res["controls"].values()) and res["outcome"]["label"] == "P-FAILS-ATTR-NOT-SEPARATED-BETTER"
    assert md.exists() and "**Outcome: P-FAILS-ATTR-NOT-SEPARATED-BETTER**" in open(md, encoding="utf-8").read()
    for k in ("s0", "s1"):
        assert all(r["delta"] == 0.0 for r in res["seeds"][k]["RICLR_minus_RIC0"].values()) and all(r["delta"] == 0.0 for r in res["seeds"][k]["RICLR_minus_RIRAPRP01"].values())


def test_chain_table_matches_the_readers_flags_and_priority():
    r = subprocess.run([BASH, f"{PK}/clr_read_chain.sh", "--dry-run"], capture_output=True, text=True)
    assert r.returncode == 0
    table = [l.split("\t") for l in r.stdout.splitlines() if l.count("\t") == 4]
    assert [t[0] for t in table] == ["RICLR_s0", "RICLR_s1", "RICLRC_s0", "RICLRC_s1", "RIRAPRP01_s0", "RIRAPRP01_s1", "CLR_s0", "CLR_s1", "CLRC_s0", "CLRC_s1", "FZCLR_s0", "FZCLR_s1"]     # the card's priority order
    for tag, seed, scene, feas, run in table:
        role = tag.rsplit("_s", 1)[0]
        assert seed == tag[-1] and (scene, feas) == C.NEW_ROLES[role] and run == C.RUN_OF[role], tag
    src = open(f"{PK}/clr_read_chain.sh", encoding="utf-8").read()
    assert 'local flags="--smooth sg72 --scene $scene"' in src and '[ "$feas" != "off" ] && flags="$flags --feasible $feas"' in src and "--freeze-card-ref" in src
    assert "--replan-steps 10 --nav deadreckon --schedule periodic --limit causal --dump-ticks" in src and "--dump-heads --device cpu" in src
    assert '[ "$scene" = "reindexed" ] && th=$TH_RI' in src and "TH_LOGGED=${CLR_TH_LOGGED:-16}" in src and "TH_RI=${CLR_TH_RI:-12}" in src     # reindexed (incl. RIRAPRP01) at 12, logged / hold_now at 16
    assert "ckpt_last.pt" in src and "M2b_s${2}_recft_${5}" in src and "clr_markers" in open(f"{PK}/code/clr_precheck.py", encoding="utf-8").read()
    assert "CLRRO.done" in src and "CLRRO.fail" in src and "CLREAD.done" in src and "CLREAD.fail" in src and "CLRREADOUT.done" in src and "clr_readout.py" in src and "clr_wiring.py" in src


def test_chain_refuses_without_pins_or_checkpoints(tmp_path):
    """The chain run for real REFUSES before any rollout: with PINS.txt absent (gate 0) it exits 2 and writes raw/CLRRO.fail; the marker is moved aside afterwards so the package carries no stale one."""
    if os.path.exists(f"{PK}/PINS.txt"):
        pytest.skip("PINS.txt exists: the refusal at gate 1 (absent checkpoints) is exercised by the packaging run (logs/refusal_chain.out)")
    r = subprocess.run([BASH, f"{PK}/clr_read_chain.sh"], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    fail = f"{PK}/raw/CLRRO.fail"
    assert r.returncode == 2 and os.path.isfile(fail) and "PINS.txt is absent" in open(fail, encoding="utf-8").read()
    shutil.move(fail, f"{PK}/logs/refusal_pins_absent.CLRRO.fail")
