"""CPU-only check of the Stage 0 partial-save path.  A dump interrupted and restored must write the same files, bit for bit, as one that never stopped.  The RED arm drops a row at restore and must differ."""
import importlib.util, os, sys, tempfile
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np, torch
R = "F:/tanitad_scratch/stage0_20261108/code/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe"
sys.path.insert(0, R); os.chdir(R)
sp = importlib.util.spec_from_file_location("sf", R + "/scorer_finetune.py")
m = importlib.util.module_from_spec(sp); sys.modules["sf"] = m; sp.loader.exec_module(m)
rng = np.random.default_rng(0)
keys = [(f"log{i%3}", f"tok{i}", i * 2, i % 2) for i in range(20)]
sx = [torch.tensor(rng.normal(size=(20, 64, 7)), dtype=torch.float32) for _ in range(3)]
def feed(d, a, b):
    for s in range(a, b, 4):
        e = min(s + 4, b)
        d.add(keys[s:e], [10075] * (e - s), sx[0][s:e], sx[1][s:e], sx[2][s:e])
def files(dir_):
    out = {}
    for arm in m.ScoreDump.ARMS:
        with np.load(os.path.join(dir_, f"scores_step0_{arm}.npz"), allow_pickle=False) as z:
            out[arm] = {k: z[k].copy() for k in z.files}
    return out
def same(x, y):
    return all(set(x[a]) == set(y[a]) and all(x[a][k].dtype == y[a][k].dtype and np.array_equal(x[a][k], y[a][k]) for k in x[a]) for a in x)
with tempfile.TemporaryDirectory(dir="F:/tanitad_scratch/stage0_20261108") as t:
    A, B, C = (os.path.join(t, n) for n in "ABC"); P = os.path.join(t, "partial.npz")
    d = m.ScoreDump(); feed(d, 0, 20); d.write(A, "step0")
    d = m.ScoreDump(); feed(d, 0, 12); d.save_partial(P)
    assert not os.path.exists(P + ".tmp.npz"), "temp file left behind"
    d = m.ScoreDump(); n = d.load_partial(P); assert n == 12 and d.keys == keys[:12], (n, d.keys[:2])
    feed(d, 12, 20); d.write(B, "step0")
    print("restored equals uninterrupted:", same(files(A), files(B)))
    assert same(files(A), files(B)), "restored dump differs"
    d = m.ScoreDump(); d.load_partial(P); d.keys = d.keys[:-1]; d.ckpts = d.ckpts[:-1]
    d.chunks = {a: [d.chunks[a][0][:-1]] for a in d.ARMS}; feed(d, 12, 20); d.write(C, "step0")
    red = same(files(A), files(C))
    print("RED arm (one row dropped) equals uninterrupted:", red)
    assert not red, "RED arm did not differ: the check is blind"
print("ZZTEST_OK")
