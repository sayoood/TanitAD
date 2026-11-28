"""fit_stub.py -- stands in for s0p_fit.py in the chain's dry run (S0P_FIT_SCRIPT, honoured only with S0P_DRY=1).  It does load the pulled shards with the REAL s0p_fit.load_any and records what the chain passed.  Test only."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import s0p_fit as F  # noqa: E402

a = sys.argv[1:]
assert a[0] == "all"
kv = {a[j]: a[j + 1] for j in range(1, len(a), 2)}
tr = F.load_any(kv["--train"])
rec = {"argv": a, "n_train": int(len(tr["y"])), "shards": tr["meta"].get("shards"), "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"), "omp": os.environ.get("OMP_NUM_THREADS"), "busy_exists": os.path.exists(os.environ.get("S0P_BUSY_DEV", "")), "v160_shape": list(tr["v160"].shape)}
os.makedirs(kv["--work"], exist_ok=True)
json.dump(rec, open(kv["--out"], "w", encoding="utf-8"))
print("fit stub ok", rec["n_train"])
