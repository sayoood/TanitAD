"""collect_speed.py -- the logged ego speed of every cached frame of the eval-141 windows, for the reads of the EGOV arms (card PREREG_V8F_PROOF_ADD19, note N1).  CPU.
  python collect_speed.py
Writes SPEED_WINDOWS.npz in this folder: V [N, 16] float32 (m/s, the batch field `wm_speed` of the target tree, read through `wm_frame_speed`), eid [N], t [N],
A_raw [N, 16, 2] (the raw action channels of the same items, for the alignment control of the reader).  The order is the order of `open_windows()`, the same
order as the rows of latents/<ARM>.pt.  It does not depend on any arm or checkpoint."""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, "/")
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import latvis_gw as T  # noqa: E402

T.assert_tree()
from tanitad.models import v8f_mini as M  # noqa: E402

ds, wins, eid = T.open_windows()
N = len(wins)
V = np.zeros((N, 16), np.float32)
A = np.zeros((N, 16, 2), np.float32)
t = np.zeros(N, np.int64)
for i in range(N):
    it = ds.item(ds.windows[i][0], ds.windows[i][1])
    V[i] = np.asarray(M.wm_frame_speed(it), np.float32)
    A[i] = np.asarray(it["act"], np.float32)
    t[i] = int(it["t"])
    if i % 400 == 0:
        print("speed", i, "of", N, flush=True)
out = HERE + "/SPEED_WINDOWS.npz"
np.savez(out, V=V, A_raw=A, eid=np.asarray(eid).astype(str), t=t, scale=np.float32(M.WM_SPEED_SCALE))
print("ZZ SPEED DONE", N, "windows; speed min %.2f max %.2f mean %.2f m/s; finite %s" % (float(V.min()), float(V.max()), float(V.mean()), bool(np.isfinite(V).all())))
