"""score_navtest_refe.py, UNCHANGED, with only its two path constants re-pointed at the E: drive and W3's harness taken
from a drive-repointed scratch copy (w3e/code: D:/ -> E:/ by sed, nothing else). Usage = score_navtest_refe's own args."""
import importlib.util
import os
import sys

PKG = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("score_navtest_refe", f"{PKG}/eval/score_navtest_refe.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.W3_CODE = os.path.join(HERE, "w3e", "code").replace("\\", "/")
m.W3_EXPORT = m.W3_EXPORT.replace("D:/", "E:/")
m.PKG = PKG
sys.exit(m.main(sys.argv[1:]))
