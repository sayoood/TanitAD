"""Peak-commit probe for Phase B's `hyps` step: a REAL-SIZE random-init v8F-mini (default config, d=256) on a synthetic world, batch 32, 4 torch threads.
No real data, no checkpoint written. Prints peak private commit (psutil peak_pagefile) after each stage."""
import os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tempfile, psutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import khyp, torch
from khyp import D, M
torch.set_num_threads(4)
pr = psutil.Process()
g = lambda t: print(f"{t:28s} peak_commit {pr.memory_info().peak_pagefile/2**30:.2f} GB  commit_now {pr.memory_info().pagefile/2**30:.2f}  rss {pr.memory_info().rss/2**30:.2f}", flush=True)
g("imports")
root = tempfile.mkdtemp(dir=str(khyp.OUT / "pt_tmp"))
w = D.make_synthetic_world(root, n_train=3, n_eval=1, T=60, seed=0)
J = D.load_join(os.path.join(root, "join_train.npz"))
mu, sd = M.cons_stats(J)
model = M.V8FMini(M.V8FMiniConfig(), M.P10Prior.synthetic(0), mu, sd).eval()
print("params", sum(p.numel() for p in model.parameters()))
g("model built")
store = D.ClipStore(root, "train", lru=3)
wins = D.train_windows(store, J, w["names"]["train"])
ds = D.V8FMiniWindows(store, J, wins, hmax=10)
BS = int(sys.argv[1]) if len(sys.argv) > 1 else 32
for b0 in (0, BS, 2 * BS):
    items = []
    for i in range(b0, b0 + BS):
        ci, t, _, _ = wins[i]; a_t = i % 3
        items.append(ds.item(ci, t, a_t, a_t + (i // 3) % 5))
    d = khyp.enumerate_batch(model, D.collate(items))
    g(f"batch of {BS} x 9 forwards #{b0 // BS}")
import shutil; shutil.rmtree(root, ignore_errors=True)
