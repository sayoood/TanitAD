"""Where does the ego-path-on-road metric lose between two world maps? For every frame, the ego's own path within +-30 m
(the MAP_B evidence) is looked up in both maps; the codes of the other map at points the first map puts on-road are
tallied (255 not seen, 0 seen background, 1-7 classes), with the along-path distance from the frame."""
import json, sys
from pathlib import Path
import numpy as np

a_p, b_p, npz_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
A = dict(np.load(a_p, allow_pickle=True)); B = dict(np.load(b_p, allow_pickle=True))
Ts = [np.array(np.load(f, allow_pickle=True)["T_world_rig"]) for f in sorted(npz_dir.glob("[0-9][0-9][0-9].npz"))]
P = np.array([T[:2, 3] for T in Ts])
# dense path: the frame positions, linearly interpolated at 0.2 m
seg = np.hypot(*np.diff(P, axis=0).T); s = np.r_[0, np.cumsum(seg)]
ss = np.arange(0, s[-1], 0.2); path = np.c_[np.interp(ss, s, P[:, 0]), np.interp(ss, s, P[:, 1])]


def look(M, xy):
    cls, (x0, y0), res = M["cls"], M["origin"], float(M["res"])
    i = np.floor((xy[:, 0] - x0) / res).astype(int); j = np.floor((xy[:, 1] - y0) / res).astype(int)
    ok = (i >= 0) & (i < cls.shape[0]) & (j >= 0) & (j < cls.shape[1])
    out = np.full(len(xy), 255, np.uint8); out[ok] = cls[i[ok], j[ok]]
    return out


ca, cb = look(A, path), look(B, path)
on_a = np.isin(ca, (1, 2, 3, 4, 6)); on_b = np.isin(cb, (1, 2, 3, 4, 6))
lost = on_a & ~on_b
tally = {str(int(k)): int((cb[lost] == k).sum()) for k in np.unique(cb[lost])}
runs = np.flatnonzero(np.diff(np.r_[0, lost.astype(int), 0]))
spans = [(round(float(ss[a]), 1), round(float(ss[b - 1]), 1)) for a, b in zip(runs[::2], runs[1::2])]
print(json.dumps({"path_points": int(len(path)), "on_road_first": int(on_a.sum()), "on_road_second": int(on_b.sum()),
                  "lost_points": int(lost.sum()), "second_map_code_at_lost": tally, "lost_spans_along_path_m": spans[:20]}))
