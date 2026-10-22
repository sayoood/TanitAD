"""score_navtest_refe.py, UNCHANGED, with only its two path constants re-pointed at the E: drive and W3's harness taken
from a drive-repointed scratch copy (w3e/code: D:/ -> E:/ by sed, nothing else). Usage = score_navtest_refe's own args."""
import importlib.util
import os
import sys

PKG = "E:/Projects/TanitAD/data/refe_navtest/_p6gaps_ctlB"
HERE = "C:/Users/Admin/par18/p6gaps/code/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval/harness"
spec = importlib.util.spec_from_file_location("score_navtest_refe", f"{PKG}/eval/score_navtest_refe.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.W3_CODE = os.path.join(HERE, "w3e", "code").replace("\\", "/")
m.W3_EXPORT = m.W3_EXPORT.replace("D:/", "E:/")
m.PKG = PKG
sys.exit(m.main(sys.argv[1:]))
