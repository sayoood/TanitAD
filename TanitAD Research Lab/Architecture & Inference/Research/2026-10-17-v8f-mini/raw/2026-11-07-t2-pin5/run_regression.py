"""The existing dominance, G2, R-SCN, scene-exclusion and arm tests of tests/ (frozen, unmodified) run against the PIN-5 module: `read_t2_dominance` is bound to pin5/read_t2_dominance_pin5.py before they import it.
Run from t2v8f/:  CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I pin5/run_regression.py <basetemp>"""
import importlib.util
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
T = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")).replace("\\", "/")
sys.path.insert(0, f"{T}/code")
sys.path.insert(0, f"{T}/tests")
spec = importlib.util.spec_from_file_location("read_t2_dominance", f"{T}/pin5/read_t2_dominance_pin5.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sys.modules["read_t2_dominance"] = mod
import pytest  # noqa: E402

files = [f"{T}/tests/{n}.py" for n in ("test_t2v8f_dominance", "test_t2v8f_dominance_g2", "test_t2v8f_dominance_rscn", "test_t2v8f_dominance_q", "test_scene_exclusions")]
sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", "--basetemp", sys.argv[1]] + files))
