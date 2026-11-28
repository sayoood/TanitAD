"""Paired episode-cluster bootstrap between two refcv3_arm dumps that share a
window grid — the estimator the doctrine requires for an arm-vs-arm delta.

⛔ The two dumps MUST be on the same grid or the pairing is a fiction, so the
window origins (`ws`) and episode ids are asserted equal BEFORE anything is
computed, and the script refuses rather than aligning them.

⛔ `overlapping_holdout_se` is forbidden; this uses
`taniteval.ci.paired_episode_cluster_bootstrap` only.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/workspace/TanitAD/stack")
sys.path.insert(0, "/workspace/TanitAD/taniteval")
from taniteval.ci import paired_episode_cluster_bootstrap   # noqa: E402

A_DIR, B_DIR = sys.argv[1], sys.argv[2]
ARM = sys.argv[3] if len(sys.argv) > 3 else "os"
OUT = sys.argv[4] if len(sys.argv) > 4 else None


def load(d):
    ade, eid, ws = [], [], []
    for f in sorted(glob.glob(os.path.join(d, "ep*.npz"))):
        z = np.load(f, allow_pickle=True)
        if ARM not in z.files:
            sys.exit("arm %r not in %s (have %s)" % (ARM, f, sorted(z.files)))
        g = np.asarray(z["g"], dtype=np.float64)        # [N, K, 2] GT
        p = np.asarray(z[ARM], dtype=np.float64)        # [N, K, 2] arm
        ade.append(np.linalg.norm(p - g, axis=-1).mean(axis=1))   # [N]
        eid.append(np.full(g.shape[0], int(np.asarray(z["eid"]).ravel()[0])))
        ws.append(np.asarray(z["ws"]).ravel())
    return (np.concatenate(ade), np.concatenate(eid), np.concatenate(ws))


a, ea, wa = load(A_DIR)
b, eb, wb = load(B_DIR)

# ⛔ the pairing assertions, before any number
assert a.shape == b.shape, "shape mismatch %s vs %s" % (a.shape, b.shape)
assert np.array_equal(ea, eb), "episode ids differ — not the same windows"
assert np.array_equal(wa, wb), "window origins differ — not the same grid"
print("PAIRING OK: n_windows=%d  n_episodes=%d  grid identical"
      % (a.size, len(set(ea.tolist()))))

res = paired_episode_cluster_bootstrap(a, b, ea, n_boot=2000, seed=0)
print("A (%s) mean = %.4f m" % (os.path.basename(A_DIR), a.mean()))
print("B (%s) mean = %.4f m" % (os.path.basename(B_DIR), b.mean()))
print("paired B - A: %s" % json.dumps(
    {k: (round(v, 6) if isinstance(v, float) else v) for k, v in res.items()}))

# ⛔ CONTROL: A paired against ITSELF must read exactly 0 with a zero-width
# interval and separated False. If it does not, the estimator is wrong and no
# number above is admissible.
ctrl = paired_episode_cluster_bootstrap(a, a, ea, n_boot=200, seed=0)
print("CONTROL A-vs-A: delta=%.10f lo=%.10f hi=%.10f separated=%s (must be 0,0,0,False)"
      % (ctrl.get("delta", float("nan")), ctrl.get("lo", float("nan")),
         ctrl.get("hi", float("nan")), ctrl.get("separated")))
n_changed = int((a != b).sum())
print("windows whose ADE differs at all: %d/%d = %.2f%%"
      % (n_changed, a.size, 100.0 * n_changed / a.size))

if OUT:
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"A": A_DIR, "B": B_DIR, "arm": ARM,
                   "n_windows": int(a.size),
                   "n_episodes": len(set(ea.tolist())),
                   "A_mean": float(a.mean()), "B_mean": float(b.mean()),
                   "paired_B_minus_A": res,
                   "control_A_vs_A": ctrl,
                   "n_windows_changed": n_changed,
                   "estimator": "paired_episode_cluster_bootstrap "
                                "(taniteval/ci.py), n_boot 2000, seed 0"},
                  fh, indent=1, default=float)
    print("wrote %s" % OUT)
