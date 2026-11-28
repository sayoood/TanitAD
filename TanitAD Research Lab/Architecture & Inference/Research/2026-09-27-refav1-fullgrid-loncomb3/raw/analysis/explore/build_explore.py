"""EXPLORATORY (post-verdict, zero GPU) synthetic dumps on the full grid -- every candidate built is reported.

Question: on the SAME damped path, does the planner's LONGITUDINAL profile beat the kinematic floors'?
`kd_<src>` = the damped floor's path (0.5*ha0 + 0.5*ha0_ext), re-timed so that its along-track distance at each
step equals <src>'s own travelled distance. Lateral geometry is therefore identical across every kd_* arm; only
the longitudinal profile differs. Controls: kd_self (damp50 re-timed by itself) must reproduce damp50.
Also: `shipped` = the 2026-09-04 Thor shipped-cost plan on the identical windows (floors bit-identical, see
void_check), and `ens2` = the mean of the two inference seeds.
"""
import glob, os
import numpy as np

R = "C:/Users/Admin/refav1_fullgrid"
A1, A2 = f"{R}/dump_loncomb3_s0", f"{R}/dump_loncomb3_s1"
SHIP = ("D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/"
        "2026-09-27-refav1-fullgrid-loncomb3/raw/full141_floor_exploratory/shipped_full_dump")
OUT = f"{R}/analysis/explore"


def arclen(p):                      # p [10,2] positions at k=1..10, origin at k=0
    P = np.concatenate([np.zeros((1, 2), p.dtype), p], 0)
    seg = np.diff(P, axis=0)
    L = np.linalg.norm(seg, axis=-1)
    return P, seg, L, np.concatenate([[0.0], np.cumsum(L)])


def retime(path, lon_src):
    P, seg, L, S = arclen(path.astype(np.float64))
    s_src = arclen(lon_src.astype(np.float64))[3][1:]
    out = np.zeros((len(s_src), 2))
    nz = np.nonzero(L > 1e-9)[0]
    for i, s in enumerate(s_src):
        if S[-1] < 1e-6 or len(nz) == 0:          # degenerate damped path: straight ahead (+x = heading at t0)
            out[i] = (s, 0.0); continue
        if s <= S[-1]:
            j = int(np.clip(np.searchsorted(S, s, side="right") - 1, 0, len(L) - 1))
            while L[j] <= 1e-9 and j + 1 < len(L):
                j += 1
            out[i] = P[j] + (s - S[j]) / max(L[j], 1e-12) * seg[j] if L[j] > 1e-9 else P[j]
        else:
            jj = nz[-1]
            out[i] = P[-1] + (s - S[-1]) * seg[jj] / L[jj]
    return out.astype(np.float32)


def main():
    names = sorted(os.path.basename(p) for p in glob.glob(f"{A1}/ep*.npz"))
    arms = ["kd_cl0", "kd_cl1", "kd_x", "kd_0", "kd_self", "shipped", "ens2"]
    for a in arms:
        os.makedirs(f"{OUT}/dump_{a}", exist_ok=True)
    worst_self = 0.0
    for f in names:
        z1, z2, zs = np.load(f"{A1}/{f}"), np.load(f"{A2}/{f}"), np.load(f"{SHIP}/{f}")
        base = {k: z1[k] for k in z1.files}
        damp = 0.5 * z1["ha0"] + 0.5 * z1["ha0_ext"]
        n = damp.shape[0]
        src = {"kd_cl0": z1["cl"], "kd_cl1": z2["cl"], "kd_x": z1["ha0_ext"], "kd_0": z1["ha0"], "kd_self": damp}
        for a, s in src.items():
            cl = np.stack([retime(damp[w], s[w]) for w in range(n)])
            if a == "kd_self":
                worst_self = max(worst_self, float(np.abs(cl - damp).max()))
            np.savez(f"{OUT}/dump_{a}/{f}", **{**base, "cl": cl})
        assert np.array_equal(zs["ha"], z1["ha"]) and np.array_equal(zs["ws"], z1["ws"])
        np.savez(f"{OUT}/dump_shipped/{f}", **{**base, "cl": zs["cl"]})
        np.savez(f"{OUT}/dump_ens2/{f}", **{**base, "cl": 0.5 * z1["cl"] + 0.5 * z2["cl"]})
    print("episodes", len(names), "| kd_self max |retimed - damp50| =", worst_self, "(control: must be ~0)")


if __name__ == "__main__":
    main()
