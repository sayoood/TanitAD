"""test_constclose.py -- tests of the tool kit of card ADD80 (CONSTCLOSE10K): the three tools with a selftest (make_const_files.py, check_constclose.py, read_constclose10k.py) and direct cases on the pure
functions of the reader. CC_TOOL_DIR (default: this folder) names the folder of the tools under test; the mutation runner points it at a mutated copy.
  PYCAP_MB=3000 bash ops/pycap.sh -m pytest -q -p no:cacheprovider constclose/test_constclose.py --basetemp constclose/tmp_pt/x
"""
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.environ.get("CC_TOOL_DIR") or HERE
ROOT = os.path.dirname(HERE)
ENV = dict(os.environ, CLOSEST_DIR=os.environ.get("CLOSEST_DIR") or os.path.join(ROOT, "closeread"), CC_ESTREAD_DIR=os.environ.get("CC_ESTREAD_DIR") or os.path.join(ROOT, "estclose_read"),
           ECR_INTEG_DIR=os.environ.get("ECR_INTEG_DIR") or os.path.join(ROOT, "integ10k"), PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1")


def selftest(tool):
    r = subprocess.run([sys.executable, os.path.join(TOOLS, tool), "--selftest"], env=ENV, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1500, cwd=TOOLS)
    return r.returncode, r.stdout + r.stderr


@pytest.mark.parametrize("tool,last", [("make_const_files.py", "ZZ CONSTFILES-SELFTEST rc 0"), ("check_constclose.py", "ZZ SELFTEST-CHECK-CONSTCLOSE ok: True"), ("read_constclose10k.py", "ZZ SELFTEST-CONSTCLOSE10K ok: True")])
def test_selftest_green(tool, last):
    rc, out = selftest(tool)
    assert rc == 0, out[-1500:]
    assert out.strip().splitlines()[-1].startswith(last), out[-400:]
    assert "False" not in [l.split()[-1] for l in out.splitlines() if l.startswith("   ")], out[-1500:]


def test_reader_pure_functions():
    sys.path.insert(0, TOOLS)
    for k, v in ENV.items():
        os.environ.setdefault(k, v)
    import read_constclose10k as RC
    assert RC.VALUES == ("C10", "C20", "C40") and RC.CVAL["C40"] == 4.0
    pr = RC.pairs_of(True)
    assert len(pr) == 15 and pr["CONST-VS-EST-L25-C40"] == ("RRe_C40", "RRe_L25")
    assert RC.cname("CONST-VS-RR-C10", "FRESH", "U", "LOWER") == "CONSTCLOSE10K-CONST-VS-RR-C10-FRESH-U-LOWER"
    ns = {"C": "NOT-SEPARATED", "U": "NOT-SEPARATED", "F": "NOT-SEPARATED", "fam": {f: "NOT-SEPARATED" for f in RC.FAMS}}
    words = {st: {p: ns for p in pr} for st in RC.SETTING_ORDER}
    names, why = RC.all_names(words, {"AGE": True, "FRESH": True}, {"AGE": True, "FRESH": True}, True)
    assert names["VERDICT_C20"] == "CONSTCLOSE10K-C20-NOT-SUPPORTED" and names["VERDICT_N10"] == "CONSTCLOSE10K-N10-NOT-BEYOND-A-CONSTANT"
    names2, why2 = RC.all_names({st: {p: ns for p in RC.pairs_of(False)} for st in RC.SETTING_ORDER}, {"AGE": True, "FRESH": True}, {"AGE": True, "FRESH": True}, False, "no --estclose-json was given")
    assert "VERDICT_N10" not in names2 and "E_VERDICTS" in why2
