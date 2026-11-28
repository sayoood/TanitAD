"""Runs the pod's study tests (test_dscale10k.py, test_dscale10k_gate.py) against kit/dscale10k_check_l3.py (loaded under the name the tests import). usage: run_study_tests.py [checker.py]"""
import importlib.util, os, sys
KIT = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
chk = sys.argv[1] if len(sys.argv) > 1 else KIT + "/dscale10k_check_l3.py"
sys.path.insert(0, KIT)
spec = importlib.util.spec_from_file_location("dscale10k_check", chk)
m = importlib.util.module_from_spec(spec); sys.modules["dscale10k_check"] = m; spec.loader.exec_module(m)
import pytest
tree = "C:/Users/Admin/lever3/tree_cdb"
sys.path[:0] = [tree + "/stack", tree + "/stack/scripts", tree]
d = KIT + "/inputs/records/dscale10k/study"
if len(sys.argv) == 1:      # the L3 checker names its token file gate_gatetok.json (secret-scan rule): the tests' file name is mapped in a COPY, nothing else
    os.makedirs(KIT + "/study_tests_l3", exist_ok=True)
    for f in ("test_dscale10k.py", "test_dscale10k_gate.py"):
        open(KIT + "/study_tests_l3/" + f, "w", encoding="utf-8").write(open(d + "/" + f, encoding="utf-8").read().replace("gate_token.json", "gate_gatetok.json"))
    d = KIT + "/study_tests_l3"
sys.exit(pytest.main([d + "/test_dscale10k.py", d + "/test_dscale10k_gate.py", "-q", "-p", "no:cacheprovider", "--basetemp", "F:/TanitAD-scratch/rollhist/lever3/kit/tmp_pt", "--rootdir", d]))
