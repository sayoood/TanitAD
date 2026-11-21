#!/usr/bin/env python3
"""Run the refcv8 NavSim package's OWN K0 / KD (+ premises) controls on Thor CUDA WITHOUT pytest (the tanitad-edge venv has
no pytest and is not modified).  The test module ``tests/test_model_seam8.py`` is imported UNCHANGED; only the two pytest
DECORATORS it uses at import (``pytest.mark.skipif`` as a module value, ``pytest.fixture(scope=...)``) are stubbed as
identities.  The skip condition is ASSERTED false here (a skipped test must never read as a pass).

    R8_TEST_CKPT=<ckpt> R8_TEST_CONFIG=<config> R8_TEST_DEVICE=cuda python r8ns_cuda_controls.py <pkg_dir> <out.json>
"""
import json
import os
import sys
import time
import traceback
import types

pkg, out = sys.argv[1], sys.argv[2]
stub = types.ModuleType("pytest")


class _Mark:
    def skipif(self, cond, reason=""):
        return {"skipif": bool(cond), "reason": reason}


def _fixture(*a, **k):
    if a and callable(a[0]):
        return a[0]
    return lambda f: f


stub.mark = _Mark()
stub.fixture = _fixture
sys.modules["pytest"] = stub
sys.path.insert(0, os.path.join(pkg, "tests"))
rec = {"ckpt": os.environ.get("R8_TEST_CKPT"), "config": os.environ.get("R8_TEST_CONFIG"),
       "device": os.environ.get("R8_TEST_DEVICE"), "tests": {}, "label": "VALIDATION ONLY (Thor CUDA controls)"}
t0 = time.time()
import test_model_seam8 as T  # noqa: E402

rec["skipif"] = T.pytestmark
if T.pytestmark["skipif"]:
    rec["verdict"] = "SKIPPED_IS_NOT_A_PASS"
    json.dump(rec, open(out, "w"), indent=1)
    sys.exit(2)
import torch  # noqa: E402

w = T.world()
rec["world_s"] = round(time.time() - t0, 1)
rec["seam_checks"] = w["meta"]["seam_checks"]
rec["precision"] = w["meta"]["precision"]
for name in ("test_premises_all_hold", "test_K0_same_seed_is_bit_identical", "test_KD_exact_dedup_matches_the_native_path"):
    try:
        getattr(T, name)(w)
        rec["tests"][name] = "PASS"
    except AssertionError as e:
        rec["tests"][name] = f"FAIL: {e!r}"[:500]
    except Exception:                                                     # noqa: BLE001
        rec["tests"][name] = "ERROR: " + traceback.format_exc()[-800:]
rec["cuda_max_memory_allocated_gb"] = (round(torch.cuda.max_memory_allocated() / 1e9, 3)
                                       if torch.cuda.is_available() else None)
rec["torch"] = torch.__version__
rec["K0_pass"] = rec["tests"]["test_K0_same_seed_is_bit_identical"] == "PASS"
rec["KD_pass"] = rec["tests"]["test_KD_exact_dedup_matches_the_native_path"] == "PASS"
rec["premises_pass"] = rec["tests"]["test_premises_all_hold"] == "PASS"
rec["wall_s"] = round(time.time() - t0, 1)
json.dump(rec, open(out, "w"), indent=1, default=str)
print(json.dumps({k: rec[k] for k in ("K0_pass", "KD_pass", "premises_pass", "precision", "cuda_max_memory_allocated_gb",
                                       "wall_s")}))
