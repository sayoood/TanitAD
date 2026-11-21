"""extract_dag_cells.py -- reduce the four DAgger T1-kin runs (DAG / DAG-0 x seed 0 / 1, causal_heads) to the per-window, per-tick cells the video draws.

NOT a re-implementation: it imports the programme's own reader `t1hd_read.py` from the PINNED tool tree (the one `dag_read.py` used, `PINS.txt`) and calls its own
`load_ticks` + `analyze_run`, so the matched route row r', the label class of T's lateral / longitudinal head, the GT route checkpoints of S and S's position error
are the reader's, byte for byte.  CPU only, read-only on every input, ONE process, about 30 s per run.

  python -I extract_dag_cells.py            -> data/dag_cells_<ARM>_s<seed>.npz  (4 files)  +  data/dag_cells_manifest.json (input shas)

Output keys (N = 2399 windows in the dump order, 3 ticks = +0 s, +2 s, +4 s):
  lat_true lat_pred lon_true lon_pred [N,3] int   (true = -1: unscored)
  rT rS dT dS [N,3]                               (matched route row of T / S and the distance to it)
  gt_w0 [N,3,3,3] gt_ok [N,3,3]                   (S's GT route checkpoints at 30 / 50 / 80 m, in the window-start frame; scored flag)
  s_pos_err [N,3,3] ctrv_pos_err [N,3,3]          (S / CTRV position error at 30 / 50 / 80 m; NaN = unscored)
  crawl [N,3]
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
TREE = "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a"
RAW = f"{TREE}/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
T1HD = f"{RAW}/2026-10-26-t1hdread/t1hd_read.py"
ART = "E:/Projects/TanitAD-artifacts/v8f-dagger-t1"
INP = "E:/Projects/TanitAD-artifacts/sfresh_inputs"
NS = types.SimpleNamespace(eval_cache=f"{INP}/cache", tracks=f"{INP}/inputs/native_tracks.pt", eval_grid=f"{RAW}/eval_grid_windows.json", eval_join=f"{RAW}/join_eval141.npz")
RUNS = [("DAG", 0), ("DAG", 1), ("DAG0", 0), ("DAG0", 1)]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    import numpy as np
    os.environ["TANITEVAL_STACK_OVERRIDE"] = os.path.join(TREE, "stack")
    d = os.path.dirname(T1HD)
    sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location("t1hd_read", T1HD)
    T = importlib.util.module_from_spec(spec)
    sys.modules["t1hd_read"] = T
    spec.loader.exec_module(T)
    T.setup_imports(TREE)
    ctx = T.Ctx(NS)
    man = {"t1hd_read.py": sha(T1HD), "tree": TREE, "inputs": {k: (sha(v) if os.path.isfile(v) else v) for k, v in vars(NS).items()}, "runs": {}}
    for arm, sd in RUNS:
        tag = f"{arm}_s{sd}_m10_causal_heads"
        tp, dd = f"{ART}/ticks/ticks_{tag}.npz", f"{ART}/dump_{tag}"
        tt = T.load_ticks(tp)
        R = T.analyze_run(ctx, tt, NS, tag, 0, dd)
        out = {"lat_true": R["cells"]["lat"]["true"], "lat_pred": R["cells"]["lat"]["pred"], "lon_true": R["cells"]["lon"]["true"], "lon_pred": R["cells"]["lon"]["pred"],
               "rT": R["rT"], "rS": R["rS"], "dT": R["dT"], "dS": R["dS"], "gt_w0": R["gt_w0"], "gt_ok": R["gt_ok"], "crawl": R["crawl"]}
        out["s_pos_err"] = np.asarray(R["err"]["S"]["pos_err_m"], np.float64)
        out["ctrv_pos_err"] = np.asarray(R["err"]["CTRV"]["pos_err_m"], np.float64)
        np.savez_compressed(f"{HERE}/data/dag_cells_{arm}_s{sd}.npz", **{k: np.asarray(v) for k, v in out.items()})
        man["runs"][f"{arm}_s{sd}"] = {"ticks_sha256": sha(tp), "n_windows": int(R["N"]), "keys": sorted(out)}
        print(f"{tag}: lat scored {int((out['lat_true'] >= 0).sum())}, lon scored {int((out['lon_true'] >= 0).sum())}", flush=True)
    json.dump(man, open(f"{HERE}/data/dag_cells_manifest.json", "w"), indent=1)
    print("ZZEXTRACT-DONE", flush=True)


if __name__ == "__main__":
    main()
