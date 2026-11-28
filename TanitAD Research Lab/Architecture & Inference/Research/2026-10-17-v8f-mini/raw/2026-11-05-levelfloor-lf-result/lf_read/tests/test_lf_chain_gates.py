"""test_lf_chain_gates.py -- the pre-check (lf_precheck.py) and the chain's gates for arms LF050 / LF075: floor-key layouts, absent / mismatching checkpoints, the dry-run (16 rollouts in priority order, no marker written) and the
chain's refusals on a COPY of the package (pins, patched-tool pin).  Env overrides: LF_PRECHECK_PATH, LF_CHAIN_PATH."""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
BASH = "C:/Program Files/Git/bin/bash.exe"
CHAIN = os.environ.get("LF_CHAIN_PATH", f"{PK}/lf_read_chain.sh")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P = _load("lf_precheck_under_test", os.environ.get("LF_PRECHECK_PATH", f"{PK}/code/lf_precheck.py"))
LFR = _load("lf_read_for_gates", f"{PK}/code/lf_read.py")


def test_floor_info_layouts_read_the_same():
    sha = "ab" * 32
    flat = {"t1v8f": {"level_floor": "on", "floor_kappa": 0.5, "floor_ckpt": "p", "floor_ckpt_sha256": sha}}
    nested = {"level_floor": {"mode": "on", "kappa": 0.5, "ckpt": "p", "sha256": sha}}
    ckdict = {"level_floor": "on", "floor_kappa": 0.5, "floor_ckpt": {"path": "p", "sha256": sha}}
    for d in (flat, nested, ckdict):
        assert P.floor_info(d) == {"level_floor": "on", "floor_kappa": 0.5, "floor_ckpt": "p", "floor_ckpt_sha256": sha}
    assert P.floor_info({"t1v8f": {}}) == {"level_floor": None, "floor_kappa": None, "floor_ckpt": None, "floor_ckpt_sha256": None}


def test_precheck_real_checkpoints_pass():
    r = P.precheck()
    assert r["PASS"] is True and not r["absent"] and not r["failures"]


def test_absent_checkpoint_is_reported_absent_not_failed(tmp_path):
    r = P.precheck(rdb=str(tmp_path))
    assert r["PASS"] is False and len(r["absent"]) == 4


def test_checkpoint_sha_mismatch_fails(tmp_path):
    for s in (0, 1):
        for run in ("C0", "LLR"):
            d = tmp_path / f"M2b_s{s}_recft_{run}"
            os.makedirs(d)
            (d / "ckpt_last.pt").write_bytes(b"not the banked checkpoint")
            (d / "SHA256SUMS.txt").write_text(hashlib.sha256(b"not the banked checkpoint").hexdigest() + " *ckpt_last.pt\n", encoding="utf-8")
    r = P.precheck(rdb=str(tmp_path))
    assert r["PASS"] is False and not r["absent"] and len(r["failures"]) >= 4


def test_rollout_check_refuses_a_wrong_kappa_and_a_wrong_llr_sha(tmp_path):
    def mk(kap, sha):
        d = tmp_path / "out" / "RILF050_s0"
        os.makedirs(d, exist_ok=True)
        ck = "E:/Projects/TanitAD-artifacts/v8f-recover/M2b_s0_recft_LLR/ckpt_last.pt"
        json.dump({"t1v8f": {"level_floor": "on", "floor_kappa": kap, "floor_ckpt": ck, "floor_ckpt_sha256": sha}}, open(d / "t1v8f_RILF050_s0.json", "w", encoding="utf-8"))
        json.dump({"ckpt_sha256": P.C0_SHA[0]}, open(tmp_path / "out" / "done_RILF050_s0.txt", "w", encoding="utf-8"))
        return P.check_rollout(str(tmp_path / "out"), "RILF050_s0")[0]
    good = P.llr_sha_of(0)
    assert all(mk(0.5, good).values())
    assert mk(0.75, good)["floor_kappa_is_the_dose"] is False
    assert mk(0.5, "0" * 64)["floor_ckpt_sha256_is_the_LLR_sha"] is False


def test_dry_run_lists_sixteen_rollouts_in_priority_order_and_writes_no_marker():
    before = sorted(os.listdir(f"{PK}/raw"))
    r = subprocess.run([BASH, CHAIN, "--dry-run"], capture_output=True, text=True, cwd=PK)
    assert r.returncode == 0, r.stderr
    rows = [l.split("\t") for l in r.stdout.splitlines() if l.count("\t") == 5]
    want = [[tag, str(s), sc, fe, d, LFR.KAPPA_FLAG[d]] for tag, _b, d, s, sc, fe in LFR.rollouts()]
    assert rows == want
    assert sorted(os.listdir(f"{PK}/raw")) == before
    assert "card sha256 977a785e (want 977a785e)" in r.stdout


def _copy_pkg(tmp):
    d = tmp / "pk"
    for sub in ("code", "tests"):
        shutil.copytree(f"{PK}/{sub}", d / sub, ignore=shutil.ignore_patterns("__pycache__"))
    for f in ("lf_read_chain.sh", "lf_read_chain.cmd"):
        shutil.copy2(CHAIN if f.endswith(".sh") else f"{PK}/{f}", d / f)
    for sub in ("logs", "raw"):
        os.makedirs(d / sub)
    lines = []
    for root, _, files in os.walk(d):
        for f in files:
            p = os.path.join(root, f).replace("\\", "/")
            rel = p[len(str(d).replace("\\", "/")) + 1:]
            if rel.startswith(("code/", "tests/")) or rel.startswith("lf_read_chain"):
                lines.append(f"{rel} sha256 {hashlib.sha256(open(p, 'rb').read()).hexdigest()}")
    (d / "PINS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return d


def _run_chain(d, bp):
    env = dict(os.environ, LF_PK=str(d).replace("\\", "/"), LF_PKW=str(d).replace("\\", "/"), LF_BUILD_PINS_FILE=bp)
    r = subprocess.run([BASH, str(d / "lf_read_chain.sh")], capture_output=True, text=True, env=env, cwd=str(d))
    fail = open(d / "raw" / "LFRO.fail", encoding="utf-8").read() if (d / "raw" / "LFRO.fail").exists() else ""
    return r.returncode, fail


def test_chain_refuses_without_the_patched_tool_pin(tmp_path):
    d = _copy_pkg(tmp_path)
    rc, fail = _run_chain(d, str(tmp_path / "absent_PINS.txt"))
    assert rc == 2 and "lf_build/PINS.txt is absent" in fail
    (tmp_path / "keyless.txt").write_text("# nothing\n", encoding="utf-8")
    rc, fail = _run_chain(d, str(tmp_path / "keyless.txt").replace("\\", "/"))
    assert rc == 2 and "no patched_tool key" in fail


def test_chain_refuses_a_tool_that_differs_from_the_pinned_patched_tool(tmp_path):
    tool = "F:/TanitAD-scratch/motion_c/lf_read/tree/stack/scripts/v8f_t1_eval_ri.py"
    if not os.path.isfile(tool):
        pytest.skip("the tree's tool is not built yet")
    d = _copy_pkg(tmp_path)
    (tmp_path / "wrong.txt").write_text("patched_tool sha256 " + "c" * 64 + "\n", encoding="utf-8")
    rc, fail = _run_chain(d, str(tmp_path / "wrong.txt").replace("\\", "/"))
    assert rc == 2 and "tree tool sha" in fail and "patched_tool" in fail
