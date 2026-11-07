"""The stage-5 identity property THROUGH code_lc/ (AGENT LC): C0's own checkpoint, run as arm C0ID by `code_lc/t2v8f_stage5_cell.py` with a `frozen_dinov3` manifest, reproduces stage 4's C0C_s0 plans and extra bit for
bit -- the same property `tests/test_t2v8f_stage5.py::check_identity` proves for code/. THREE fresh processes on the same kit cell (scene 7c72937c, T2): stage 4's cell (C0C_s0), code/'s stage-5 cell (C0ID_s0), code_lc's
stage-5 cell (C0ID_s0). Fixture: the existing stage-4 rehearsal fixture (`tests/test_t2v8f_stage4.fixture4`, imported READ-ONLY, built into this agent's scratch on E:).
RED arm: code_lc's cell pointed at C0's seed-1 checkpoint for seed 0 must NOT equal stage 4's plans.
Run from t2v8f/:  CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -m pytest tests_lc/test_t2v8f_lc_identity.py -q
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["T2V8F_S2_SCRATCH"] = "E:/Projects/TanitAD-artifacts/scratch_lc_t2v8f/fixtures"
os.environ["T2V8F_S1_SCRATCH"] = "E:/Projects/TanitAD-artifacts/scratch_lc_t2v8f/fixtures_s1"
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
T2 = os.path.dirname(HERE)
sys.dont_write_bytecode = True
for _p in (f"{T2}/code", f"{T2}/tests"):                        # the READ-ONLY stage fixtures live in tests/; they import code/ modules
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402

import t2v8f_stage2 as D  # noqa: E402
import test_t2v8f_stage1 as T1  # noqa: E402
import test_t2v8f_stage2 as T  # noqa: E402
import test_t2v8f_stage4 as T4  # noqa: E402

PY = T1.PY
_SEQ = itertools.count(1)
ENV = "T2V8F_S5_MANIFEST"


def cell(fx, f, sc, script, arm, env_extra, out_root):
    out = f"{out_root}/id_{arm}_{os.path.basename(os.path.dirname(script))}_{next(_SEQ)}"
    os.makedirs(out)
    env = T.cell_env(fx)
    env["T2V8F_RI_TOOL"] = f"{fx['pkg']}/refs/tool/v8f_t1_eval_ri.py"
    env.update(env_extra)
    p = subprocess.run([PY, "-I", "-u", script] + D.cell_argv(f, sc, "T2", arm, out), env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stdout[-500:] + p.stderr[-1500:]
    return [json.loads(x) for x in open(f"{out}/records.jsonl", encoding="utf-8")]


def check_identity_through_code_lc(mut=None):
    f4 = T4.fixture4()
    fx = f4["fx"]
    s2 = T.new_scenes2(fx, "lcid")
    f, sc = T.kit_cell(fx, s2)
    man = {"arm5": {"name": "C0ID", "encoder": {"kind": "frozen_dinov3"}, "ckpt_file": {"C0ID_s0": "C0_s0.pt", "C0ID_s1": "C0_s1.pt"}}}
    mp = f"{f4['root']}/MANIFEST5_lcid_{time.strftime('%H%M%S')}.json"
    json.dump(man, open(mp, "w", encoding="utf-8", newline="\n"))
    man_lc = json.loads(json.dumps(man))
    if mut == "wrong_ckpt":
        man_lc["arm5"]["ckpt_file"]["C0ID_s0"] = "C0_s1.pt"
    mp_lc = f"{f4['root']}/MANIFEST5_lcid_lc_{time.strftime('%H%M%S')}.json"
    json.dump(man_lc, open(mp_lc, "w", encoding="utf-8", newline="\n"))
    root = f4["root"]
    c4 = cell(fx, f, sc, f"{fx['pkg']}/code/t2v8f_stage4_cell.py", "C0C_s0", {}, root)                    # stage 4's cell: the reference
    c5 = cell(fx, f, sc, f"{T2}/code/t2v8f_stage5_cell.py", "C0ID_s0", {ENV: mp}, root)                  # code/'s stage-5 cell
    cl = cell(fx, f, sc, f"{T2}/code_lc/t2v8f_stage5_cell.py", "C0ID_s0", {ENV: mp_lc}, root)            # code_lc's stage-5 cell (frozen manifest)
    assert len(c4) == len(c5) == len(cl) == 2 and sorted(c4[0]) == sorted(cl[0]) == sorted(c5[0])
    for a, b, c in zip(c4, c5, cl):
        assert a["f0"] == b["f0"] == c["f0"]
        assert a["plans"] == c["plans"], "code_lc with C0's own checkpoint does not give stage 4's C0C plan"
        assert b["plans"] == c["plans"] and b["extra"] == c["extra"], "code_lc's frozen path differs from code/'s"
        assert a["extra"] == c["extra"] and a["arm"] == "C0C_s0" and c["arm"] == "C0ID_s0"
    return {"records": len(cl)}


def test_identity_through_code_lc():
    check_identity_through_code_lc()


def test_red_identity_wrong_ckpt():
    with pytest.raises(AssertionError):
        check_identity_through_code_lc("wrong_ckpt")
