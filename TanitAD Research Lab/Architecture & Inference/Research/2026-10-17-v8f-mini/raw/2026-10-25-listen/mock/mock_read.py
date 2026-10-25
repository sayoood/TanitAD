import json, sys, os
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import numpy as np, torch
torch.set_num_threads(2)
import listen as LM, listen_read as LR
from tanitad.data import v8f_mini_data as D
from tanitad.models import v8f_mini as M
root = HERE / "mock" / "syn"
w = D.make_synthetic_world(root, n_train=4, n_eval=3, T=56, seed=0, d_feat=64)
import json as js
grid = js.load(open(root / "eval_grid_windows.json")); ej = D.load_join(root / "join_eval.npz")
ej["limit"] = np.eye(4, dtype=np.float32)[np.arange(len(ej["clip"])) % 4]
X = D.eval_context(grid, root / "eval_kin.npz", ej, kin_mode="steer")
store = D.ClipStore(root, "eval", lru=8); wins = D.eval_grid_windows(root / "eval_grid_windows.json", ej)
ds = D.V8FMiniWindows(store, ej, wins, hmax=10)
def mk(seed):
    cfg = M.V8FMiniConfig(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, d_feat=64)
    mu, sd = M.cons_stats(ej); m = M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if not n.startswith("prior."): p.add_(torch.randn(p.shape, generator=g) * 0.3)
    return m.eval()
recs = {}
for key, (s, d) in {"s0": (0, 0), "s1": (1, 0), "s0_d1": (0, 1)}.items():
    m = mk(s); perm = LM.donor_perm(len(wins), X["eid"], d)
    res = LM.collect(m, ds, wins, X["gt"], perm, batch=6)
    recs[key] = LR.run_record(res, X, B=60)
an = LR.analyse(recs)
json.dump({"analysis": an, "grid": recs}, open(HERE / "mock" / "READ_mock.json", "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(an["outcome"]["headline"], an["qnav"]["headline"])
