"""pod_tests_q.py -- runs test_imag_plan_q.py once with the plain-python runner (mutlists/common/minipytest.py) and prints each test with ok / the end of its error. ON THE POD, CPU, one process."""
import os
import sys
import tempfile

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("DISTRULE_TREE", "/workspace/v8proof/tree_leadsup")
sys.path.insert(0, "/workspace/v8proof/mutlists/common")
for p in ("/workspace/v8proof/tree_leadsup/stack", "/workspace/v8proof/tree_leadsup/stack/scripts"):
    sys.path.insert(0, p)
import minipytest as mp  # noqa: E402

here = os.environ.get("IPQ_SRC", "/workspace/v8proof/imagplan_q")
tmp = os.environ.get("IPQ_TMP", "/workspace/v8proof/imagplan_q/mut/tmp_run")
os.makedirs(tmp, exist_ok=True)
r = mp.run_module(here + "/test_imag_plan_q.py", tmp_root=tmp)
for k, v in r.items():
    print(("OK   " if v["ok"] else "FAIL ") + k[:70], "" if v["ok"] else v["err"][-700:].replace("\n", " | "))
print("TESTS", len(r), "failed", sum(1 for v in r.values() if not v["ok"]))
