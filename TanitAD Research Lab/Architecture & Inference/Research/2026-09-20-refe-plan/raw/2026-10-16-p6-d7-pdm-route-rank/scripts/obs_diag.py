"""Why does a part rebuilt with the same code differ in BYTES, and in canon(obs), from the existing one?  Read-only: loads two files, never writes the cache.
usage: obs_diag.py <fileA> <fileB>     (the D9 parts files <log>/<token>[__lane_rank_K].pkl.xz)"""
import sys, os, lzma, pickle, json
sys.path.insert(0, "/workspace/ops/d7fix/pkg/refe"); sys.path.insert(0, "/workspace/ops/d7fix/pkg/code")
os.chdir("/workspace/ops/d7fix/pkg/refe")
import numpy as np
import online_labeller as OL, p6_prep as PP, navsim_pdm_targets as PT
PT._setup()
fa, fb = sys.argv[1], sys.argv[2]
ra, rb = lzma.decompress(open(fa, "rb").read()), lzma.decompress(open(fb, "rb").read())
print("FILES", fa[-60:], "|", fb[-60:])
print("pickle bytes: A", len(ra), "B", len(rb), "equal", ra == rb)
a, b = OL.load_parts(fa), OL.load_parts(fb)
a2 = OL.load_parts(fa)
names = ("traj", "ego", "obs", "centerline", "lanes", "dam")
print("canon equal per part (A vs B):", {n: PP.canon(x) == PP.canon(y) for n, x, y in zip(names, a, b)})
print("canon(obs) of the SAME file loaded twice equal:", PP.canon(a[2]) == PP.canon(a2[2]))
print("type(obs):", type(a[2]).__name__)


def first_diff(x, y, path="obs", depth=0):
    cx, cy = PP.canon(x), PP.canon(y)
    if cx == cy:
        return None
    if depth > 5:
        return (path, cx[:200], cy[:200])
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)) and len(x) == len(y):
        for i, (p, q) in enumerate(zip(x, y)):
            d = first_diff(p, q, f"{path}[{i}]", depth + 1)
            if d:
                return d
    if isinstance(x, dict) and isinstance(y, dict) and set(map(repr, x)) == set(map(repr, y)):
        for k in sorted(x, key=repr):
            d = first_diff(x[k], y[k], f"{path}[{k!r}]", depth + 1)
            if d:
                return d
    if hasattr(x, "__dict__") and hasattr(y, "__dict__") and type(x) is type(y):
        return first_diff(vars(x), vars(y), path + "." + type(x).__name__, depth + 1) or (path + ".<same vars, different canon>", cx[:200], cy[:200])
    return (path, cx[:200], cy[:200])


d = first_diff(a[2], b[2])
print("FIRST_DIFF_IN_OBS", json.dumps(d)[:1200])
# the functional contract (D9 exactness): score_plans(A) == score_plans(B) on random proposals, EVERY array
rng = np.random.default_rng(0)
eq = 0
for _ in range(8):
    props = np.stack([np.stack([np.linspace(0.5, 30.0, 20) * (0.8 + 0.4 * rng.random()), rng.normal(0, 1.0, 20).cumsum() * 0.1, np.zeros(20)], -1) for _ in range(8)])
    plans8 = PT.executed_plans8(props, repair=True)
    s1, s2 = PT.score_plans(a, plans8), PT.score_plans(b, plans8)
    eq += int(all(np.array_equal(s1[k], s2[k]) for k in s1))
print(f"ZZSCORE_EQUAL {eq}/8 random proposal sets (every array np.array_equal)")
