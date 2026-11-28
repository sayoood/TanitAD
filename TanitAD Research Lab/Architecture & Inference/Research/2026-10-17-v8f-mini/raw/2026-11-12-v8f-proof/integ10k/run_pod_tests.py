"""run_pod_tests.py -- runs test_integ10k.py on the pod CPU with minipytest (the pod has no pytest). Usage: python run_pod_tests.py BUILD_DIR TMP_DIR"""
import json
import os
import sys

b, tmp = sys.argv[1], sys.argv[2]
os.makedirs(tmp, exist_ok=True)
os.environ.setdefault("DISTRULE_DIR", b)
sys.path.insert(0, b)
import minipytest as mp  # noqa: E402

r = mp.run_module(os.path.join(b, "test_integ10k.py"), tmp_root=tmp)
bad = {k: v["err"][:300] for k, v in r.items() if not v["ok"]}
print(json.dumps({"n": len(r), "ok": sum(1 for v in r.values() if v["ok"]), "bad": bad}))
try:
    import torch
    import tok_reader as TR
    import numpy as np
    x = np.random.default_rng(0).normal(size=(50, 1024)).astype(np.float32)
    a = TR._gelu(x)
    e = 0.5 * x * (1.0 + TR._erf(x / np.sqrt(2.0)))
    print("torch gelu path vs numpy erf path max abs diff", float(np.abs(a - e).max()), "torch", torch.__version__)
except Exception as ex:  # noqa: BLE001
    print("gelu comparison not run:", ex)
