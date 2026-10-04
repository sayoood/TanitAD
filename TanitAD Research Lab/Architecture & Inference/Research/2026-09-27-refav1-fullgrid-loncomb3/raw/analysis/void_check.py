"""VOID check for the full-grid SPEC s6: floors must be bit-identical between A1 and A2 (and vs the shipped dump)."""
import glob, os, sys
import numpy as np
R = "C:/Users/Admin/refav1_fullgrid"
PKG = ("D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/"
       "2026-09-27-refav1-fullgrid-loncomb3/raw/full141_floor_exploratory/shipped_full_dump")
A, B = f"{R}/dump_loncomb3_s0", f"{R}/dump_loncomb3_s1"
fa = sorted(os.path.basename(p) for p in glob.glob(f"{A}/ep*.npz"))
fb = sorted(os.path.basename(p) for p in glob.glob(f"{B}/ep*.npz"))
print("episodes", len(fa), len(fb), "same names:", fa == fb)
keys = ["g", "v0", "ha", "ha0", "ha0_ext", "ol", "ws", "eid", "clip_index"]
bad = {k: 0 for k in keys}; ncl_diff = 0; nw = 0
for f in fa:
    za, zb = np.load(f"{A}/{f}"), np.load(f"{B}/{f}")
    for k in keys:
        if not np.array_equal(za[k], zb[k]):
            bad[k] += 1
    nw += za["cl"].shape[0]
    ncl_diff += int((np.abs(za["cl"] - zb["cl"]).reshape(za["cl"].shape[0], -1).max(1) > 0).sum())
print("windows", nw, "| floor/input keys differing (episodes):", bad)
print("cl differs between seeds in", ncl_diff, "of", nw, "windows (must be > 0: the seeds really differ)")
# shipped dump (Thor 2026-09-04, older stack): pairable only if g/ha/ha0/ws are bit-identical
sf = sorted(glob.glob(f"{PKG}/*.npz")) or sorted(glob.glob(f"{PKG}/**/*.npz", recursive=True))
print("shipped dump files:", len(sf), (os.path.basename(sf[0]) if sf else None))
if sf:
    z = np.load(sf[0]); print("shipped keys:", z.files)
    same = {k: 0 for k in ["g", "v0", "ha", "ha0", "ws", "eid", "clip_index"]}; tot = 0; maxdiff = {k: 0.0 for k in same}
    for p in sf:
        f = os.path.basename(p)
        if not os.path.exists(f"{A}/{f}"):
            continue
        zs, za = np.load(p), np.load(f"{A}/{f}"); tot += 1
        for k in same:
            if k in zs.files and zs[k].shape == za[k].shape:
                if np.array_equal(zs[k], za[k]):
                    same[k] += 1
                elif zs[k].dtype.kind == "f":
                    maxdiff[k] = max(maxdiff[k], float(np.abs(zs[k] - za[k]).max()))
    print("shipped vs A1, episodes compared:", tot, "| bit-identical per key:", same, "| max abs diff:", maxdiff)
