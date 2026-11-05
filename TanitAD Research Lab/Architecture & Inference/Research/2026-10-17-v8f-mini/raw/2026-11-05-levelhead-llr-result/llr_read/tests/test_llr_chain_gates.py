"""test_llr_chain_gates.py -- the pre-check (llr_precheck.py) and the chain's gates for arm LLR, on a SYNTHETIC rdb (F:/.../pt/rdb: the synthetic LLR-shaped checkpoint of make_synthetic_llr.py + CLR's REAL argv, markers, checkpoints).
Every refusal has a test; every test is a mutation target (run_mutations_llr.py).  Env overrides: LLR_PRECHECK_PATH, LLR_READ_PATH.   Run with --basetemp on F: (the checkpoints are 88 MB; hardlinks are tried first)."""
import copy
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
DL = f"{ART}/dagland/motion_c"
RDB_REAL = f"{ART}/v8f-recover"
SYN = "F:/TanitAD-scratch/motion_c/llr_read/pt/synth/M2b_s0_recft_LLR/ckpt_last.pt"
BASH = "C:/Program Files/Git/bin/bash.exe"
TIP = open(f"{DL}/llr_launch/landed_tip.txt", encoding="utf-8").read().split()[0]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P = _load("llr_precheck_under_test", os.environ.get("LLR_PRECHECK_PATH", f"{PK}/code/llr_precheck.py"))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


LEAD = ["--lead-head", "on", "--w-lead", "3.362", "--aux-cond", "token"]


@pytest.fixture()
def rdb(tmp_path):
    assert os.path.isfile(SYN), "run code/make_synthetic_llr.py first"
    r = tmp_path / "rdb"
    h = sha(SYN)
    for s in (0, 1):
        for run in ("LLR", "CLR"):
            os.makedirs(r / f"M2b_s{s}_recft_{run}")
        link(SYN, r / f"M2b_s{s}_recft_LLR" / "ckpt_last.pt")
        (r / f"M2b_s{s}_recft_LLR" / "SHA256SUMS.txt").write_text(f"{h}  ckpt_last.pt\n", encoding="utf-8")
        link(f"{RDB_REAL}/M2b_s{s}_recft_CLR/ckpt_last.pt", r / f"M2b_s{s}_recft_CLR" / "ckpt_last.pt")
        for f in ("SHA256SUMS.txt", "argv.json"):
            shutil.copy2(f"{RDB_REAL}/M2b_s{s}_recft_CLR/{f}", r / f"M2b_s{s}_recft_CLR" / f)
        ca = json.load(open(f"{RDB_REAL}/M2b_s{s}_recft_CLR/argv.json", encoding="utf-8"))["argv"]
        la = [(x.replace("_recft_CLR", "_recft_LLR") if x.startswith("D:") else x) for x in ca] + LEAD
        (r / f"M2b_s{s}_recft_LLR" / "argv.json").write_text(json.dumps({"argv": la, "sha": "x"}), encoding="utf-8")
        (r / f"M2b_s{s}_recft_LLR" / "summary.json").write_text(json.dumps({"recover_n": 120, "level_head": "on", "w_level": 0.2027}), encoding="utf-8")
    os.makedirs(r / "llr_markers")
    os.makedirs(r / "clr_markers")
    for s in (0, 1):
        (r / "llr_markers" / f"RUN_LLR_{s}.done").write_text(json.dumps({"evidence": {"ckpt_sha256": h, "recover_n": 120, "lead_head": "on", "aux_cond": "token"}}), encoding="utf-8")
        (r / "llr_markers" / f"gate_token_LLR_{s}.json").write_text(json.dumps({"binding": {"commit": TIP, "w_lead": 3.362, "aux_cond": "token"}}), encoding="utf-8")
        shutil.copy2(f"{RDB_REAL}/clr_markers/RUN_CLR_{s}.done", r / "clr_markers" / f"RUN_CLR_{s}.done")
    (tmp_path / "FROZEN_LLR.txt").write_text("WIRING=token\n# x\nW_LEAD=3.362\n", encoding="utf-8")
    return r


def pre(rdb, tmp_path, frozen=None, **kw):
    args = ["--rdb", str(rdb), "--frozen-llr", str(frozen or tmp_path / "FROZEN_LLR.txt"), "--landed-tip", kw.pop("tip", f"{DL}/llr_launch/landed_tip.txt"), "--raw", str(tmp_path / "raw")] + kw.pop("extra", [])
    rc = P.main(args)
    return rc, json.load(open(tmp_path / "raw" / "LLR_PRECHECK.json", encoding="utf-8"))


def fails(res, needle):
    return any(needle in f for f in res["failures"])


def edit_json(p, fn):
    d = json.load(open(p, encoding="utf-8"))
    fn(d)
    open(p, "w", encoding="utf-8").write(json.dumps(d))


def test_precheck_passes_on_a_consistent_tree_and_reports_the_trained_lever(rdb, tmp_path):
    rc, res = pre(rdb, tmp_path)
    assert rc == 0 and res["PASS"] and not res["failures"], res["failures"]
    s0 = res["seeds"]["s0"]
    assert s0["argv_audit"]["PASS"] and s0["cfg_aux_cond"] == "token" and s0["keys"]["e_aux"] == 2 and s0["keys"]["e_level"] == 0 and s0["keys"]["e_lead"] == 0
    assert len(s0["e_aux_abs_sum_per_input_column_level_presence_gap"]) == 3 and s0["e_aux_abs_sum"] > 0 and s0["lead_head_abs_sum"] > 0
    assert s0["clr_ckpt_sha256"] == json.load(open(f"{DL}/clr_read/raw/CLR_PRECHECK.json", encoding="utf-8"))["seeds"]["s0"]["ckpt_sha256"]
    assert res["w_lead"] == "3.362" and res["aux_cond"] == "token" and res["landed_tip"] == TIP


def test_precheck_refuses_an_absent_checkpoint_exit_5(rdb, tmp_path):
    os.remove(rdb / "M2b_s1_recft_LLR" / "ckpt_last.pt")
    rc, res = pre(rdb, tmp_path)
    assert rc == 5 and not res["PASS"] and fails(res, "ckpt_last.pt is absent")


def test_precheck_refuses_a_missing_marker_or_gate_token_exit_5(rdb, tmp_path):
    os.remove(rdb / "llr_markers" / "RUN_LLR_0.done")
    rc, res = pre(rdb, tmp_path)
    assert rc == 5 and fails(res, "marker") and fails(res, "RUN_LLR_0.done")
    os.remove(rdb / "llr_markers" / "gate_token_LLR_1.json")
    rc, res = pre(rdb, tmp_path)
    assert rc == 5 and fails(res, "launch-gate token")


def test_precheck_refuses_sha_mismatch_with_sums_or_marker(rdb, tmp_path):
    (rdb / "M2b_s0_recft_LLR" / "SHA256SUMS.txt").write_text("0" * 64 + "  ckpt_last.pt\n", encoding="utf-8")
    rc, res = pre(rdb, tmp_path)
    assert rc == 8 and fails(res, "SHA256SUMS.txt")


def test_precheck_refuses_a_marker_sha_that_differs(rdb, tmp_path):
    edit_json(rdb / "llr_markers" / "RUN_LLR_1.done", lambda d: d["evidence"].__setitem__("ckpt_sha256", "1" * 64))
    rc, res = pre(rdb, tmp_path)
    assert rc == 8 and fails(res, "box-2 marker")


def _argv_edit(rdb, s, fn):
    edit_json(rdb / f"M2b_s{s}_recft_LLR" / "argv.json", lambda d: fn(d["argv"]))


def test_precheck_argv_audit_is_exact(rdb, tmp_path):
    for name, fn in (("lead token removed", lambda a: a.__delitem__(slice(-2, None))), ("w_lead changed", lambda a: a.__setitem__(a.index("--w-lead") + 1, "1.0")),
                     ("wiring add", lambda a: a.__setitem__(a.index("--aux-cond") + 1, "add")), ("extra token", lambda a: a.extend(["--w-acc", "1"])),
                     ("level token changed", lambda a: a.__setitem__(a.index("--w-level") + 1, "0.1")), ("recovery removed", lambda a: a.__delitem__(slice(a.index("--recover-p"), a.index("--recover-p") + 2))),
                     ("seed wrong", lambda a: a.__setitem__(a.index("--seed") + 1, "1")), ("lead head off", lambda a: a.__setitem__(a.index("--lead-head") + 1, "off"))):
        d = rdb.parent / f"v_{abs(hash(name))}"
        shutil.copytree(rdb, d, copy_function=lambda s_, t_: link(s_, t_) if s_.endswith(".pt") else shutil.copy2(s_, t_))
        _argv_edit(d, 0, fn)
        rc, res = pre(d, tmp_path)
        assert rc == 8 and fails(res, "argv audit failed"), name
        shutil.rmtree(d)


@pytest.mark.parametrize("body", ["", "WIRING=token\n", "W_LEAD=3.362\n", "WIRING=maybe\nW_LEAD=3.362\n", "WIRING=token\nW_LEAD=-1\n", "WIRING=token\nW_LEAD=abc\n"])
def test_precheck_refuses_a_missing_or_bad_frozen_file(rdb, tmp_path, body):
    (tmp_path / "bad.txt").write_text(body, encoding="utf-8")
    rc, res = pre(rdb, tmp_path, frozen=tmp_path / "bad.txt")
    assert rc == 8 and fails(res, "FROZEN_LLR")


def test_precheck_refuses_a_wiring_the_checkpoint_was_not_built_with(rdb, tmp_path):
    (tmp_path / "add.txt").write_text("WIRING=add\nW_LEAD=3.362\n", encoding="utf-8")
    rc, res = pre(rdb, tmp_path, frozen=tmp_path / "add.txt")
    assert rc == 8 and fails(res, "cfg aux_cond") and fails(res, "argv audit failed")


def test_precheck_refuses_a_checkpoint_without_the_lead_head(rdb, tmp_path):
    os.remove(rdb / "M2b_s0_recft_LLR" / "ckpt_last.pt")
    link(f"{RDB_REAL}/M2b_s0_recft_CLR/ckpt_last.pt", rdb / "M2b_s0_recft_LLR" / "ckpt_last.pt")           # a level-only checkpoint in the LLR slot
    h = sha(rdb / "M2b_s0_recft_LLR" / "ckpt_last.pt")
    (rdb / "M2b_s0_recft_LLR" / "SHA256SUMS.txt").write_text(f"{h}  ckpt_last.pt\n", encoding="utf-8")
    edit_json(rdb / "llr_markers" / "RUN_LLR_0.done", lambda d: d["evidence"].__setitem__("ckpt_sha256", h))
    rc, res = pre(rdb, tmp_path)
    assert rc == 8 and fails(res, "cfg lead_head") and fails(res, "lacks lvl_head.*") and fails(res, "needs e_aux")


def test_precheck_refuses_a_marker_or_summary_without_recovery_evidence(rdb, tmp_path):
    edit_json(rdb / "llr_markers" / "RUN_LLR_0.done", lambda d: d["evidence"].__setitem__("recover_n", 0))
    edit_json(rdb / "M2b_s0_recft_LLR" / "summary.json", lambda d: d.__setitem__("recover_n", 0))
    rc, res = pre(rdb, tmp_path)
    assert rc == 8 and fails(res, "recover_n")


def test_precheck_refuses_a_marker_key_that_disagrees(rdb, tmp_path):
    edit_json(rdb / "llr_markers" / "RUN_LLR_1.done", lambda d: d["evidence"].__setitem__("aux_cond", "add"))
    rc, res = pre(rdb, tmp_path)
    assert rc == 8 and fails(res, "aux_cond 'add'")


def test_precheck_refuses_a_gate_token_bound_to_another_commit_or_wiring(rdb, tmp_path):
    edit_json(rdb / "llr_markers" / "gate_token_LLR_0.json", lambda d: d["binding"].__setitem__("commit", "f" * 40))
    rc, res = pre(rdb, tmp_path)
    assert rc == 8 and fails(res, "gate token commit")
    edit_json(rdb / "llr_markers" / "gate_token_LLR_0.json", lambda d: d["binding"].update(commit=TIP, aux_cond="add"))
    rc, res = pre(rdb, tmp_path)
    assert rc == 8 and fails(res, "gate token w_lead / aux_cond")


def test_precheck_checks_that_the_base_is_the_checkpoint_the_banked_arrays_came_from(rdb, tmp_path):
    pc = json.load(open(f"{DL}/clr_read/raw/CLR_PRECHECK.json", encoding="utf-8"))
    pc["seeds"]["s0"]["ckpt_sha256"] = "2" * 64
    (tmp_path / "clr_pc.json").write_text(json.dumps(pc), encoding="utf-8")
    rc, res = pre(rdb, tmp_path, extra=["--clr-precheck", str(tmp_path / "clr_pc.json")])
    assert rc == 8 and fails(res, "banked CLR arrays came from another file")
    os.remove(rdb / "M2b_s1_recft_CLR" / "ckpt_last.pt")
    rc, res = pre(rdb, tmp_path)
    assert rc == 5 and fails(res, "attribution base")


def test_precheck_refuses_a_clr_checkpoint_that_has_the_lead_head(rdb, tmp_path):
    os.remove(rdb / "M2b_s0_recft_CLR" / "ckpt_last.pt")
    link(SYN, rdb / "M2b_s0_recft_CLR" / "ckpt_last.pt")
    h = sha(SYN)
    (rdb / "M2b_s0_recft_CLR" / "SHA256SUMS.txt").write_text(f"{h}  ckpt_last.pt\n", encoding="utf-8")
    rc, res = pre(rdb, tmp_path)
    assert rc == 8 and fails(res, "attribution base CLR must be level on, lead off")


def test_generated_files_equal_their_generators():
    py = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
    for g in ("make_llr_from_clr.py", "make_llr_chain.py"):
        r = subprocess.run([py, "-I", f"{PK}/code/{g}", "--check"], capture_output=True, text=True)
        assert r.returncode == 0, (g, r.stdout, r.stderr)


def test_chain_table_matches_the_readers_flags_and_priority():
    C = _load("llr_read_gates_under_test", os.environ.get("LLR_READ_PATH", f"{PK}/code/llr_read.py"))
    r = subprocess.run([BASH, "-c", f"bash {PK}/llr_read_chain.sh --dry-run"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows = [l.split("\t") for l in r.stdout.splitlines() if l.count("\t") == 4]
    assert len(rows) == 10 and [x[0] for x in rows[:4]] == ["RILLR_s0", "RILLR_s1", "RILLRC_s0", "RILLRC_s1"], rows
    assert {x[0].rsplit("_s", 1)[0] for x in rows} == set(C.NEW_ROLES) and all(x[4] == "LLR" for x in rows)
    for tag, seed, scene, feas, run in rows:
        role = tag.rsplit("_s", 1)[0]
        assert (scene, feas) == C.NEW_ROLES[role] and int(seed) == int(tag[-1]) and C.RUN_OF[role] == run
    assert "RAPRP01" not in r.stdout and "no attribution-base rollout" in open(f"{PK}/llr_read_chain.sh", encoding="utf-8").read().lower().replace("no attribution-base", "no attribution-base")


def test_chain_refuses_without_pins_and_the_shell_text_names_the_gates(tmp_path):
    env = dict(os.environ, LLR_PK=str(tmp_path).replace("\\", "/"), LLR_PKW=str(tmp_path).replace("\\", "/"))
    r = subprocess.run([BASH, "-c", f"bash {PK}/llr_read_chain.sh"], capture_output=True, text=True, env=env)
    assert r.returncode == 2, (r.returncode, r.stdout[-400:], r.stderr[-400:])
    assert "PINS.txt is absent" in open(tmp_path / "raw" / "LLRRO.fail", encoding="utf-8").read()
    t = open(f"{PK}/llr_read_chain.sh", encoding="utf-8").read()
    for need in ("llr_precheck.py", "ka_new2", "KA_NEW2.json", "ka-clr", "ZZLLR-GATE0", "ZZLLR-GATE1", "ZZLLR-GATE2", "PRE_LLR_s0_ri_kbox", "PRE_LLR_s1_hold", "PRE_LLR_s0_logged", "llr_readout.py", "ten LLR-read rollouts", "LLRRO.done", "LLREAD.done", "LLRREADOUT.done"):
        assert need in t, need
    assert "RAPRP01" not in t and "RIRAPRP01" not in t and "cl_read/tree" not in t.replace("motion_c/llr_read/tree", "")
