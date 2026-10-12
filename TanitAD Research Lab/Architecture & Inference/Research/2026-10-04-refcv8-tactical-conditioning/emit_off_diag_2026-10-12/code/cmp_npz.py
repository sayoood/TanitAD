"""Compare two ladder-eval npz files on their shared leading windows: identity of the window list, max |dtraj|,
and how many windows' plans differ by more than 1 cm anywhere."""
import sys
import numpy as np
a, b = np.load(sys.argv[1], allow_pickle=False), np.load(sys.argv[2], allow_pickle=False)
n = min(len(a["traj"]), len(b["traj"]))
same_w = bool(np.array_equal(a["win_sha12"][:n], b["win_sha12"][:n]) and np.array_equal(a["win_t"][:n], b["win_t"][:n]))
d = np.abs(a["traj"][:n] - b["traj"][:n])
per = d.reshape(n, -1).max(1)
print(f"n={n} same_windows={same_w} max|dtraj|={d.max():.6f} m  windows_differing>1cm={int((per > 0.01).sum())}/{n}  "
      f"median_per_window_max={np.median(per):.4f}")
for k in ("v_fed", "nav", "v0"):
    if k in a.files and k in b.files:
        print(f"  {k}: equal={bool(np.allclose(a[k][:n], b[k][:n], equal_nan=True))}")
