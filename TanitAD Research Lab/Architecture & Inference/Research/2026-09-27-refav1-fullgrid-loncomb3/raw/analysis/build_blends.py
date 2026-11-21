"""The zero-GPU blend / damped-floor dumps scored in pd_fullgrid (SPEC A1 s8, A2 s9, A3 s10) -- banked builder.

b50_sX = 0.5*loncomb3_sX + 0.5*ha0_ext   (A1, the planner blend)
damp50 = 0.5*ha0 + 0.5*ha0_ext           (A2, the planner-free damped floor)
dampha = 0.5*ha0 + 0.5*ha                (A3, the damped hold-action floor)
w0     = ha0_ext                         (the w = 0 known-value control; must read exactly 0 vs ha0_ext)
Every other key is copied from the A1 dump (the floors are bit-identical between A1 and A2, see void_check).
"""
import glob, os, sys
import numpy as np

R = "C:/Users/Admin/refav1_fullgrid"
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{R}/analysis"


def main():
    names = sorted(os.path.basename(p) for p in glob.glob(f"{R}/dump_loncomb3_s0/ep*.npz"))
    for f in names:
        z0, z1 = np.load(f"{R}/dump_loncomb3_s0/{f}"), np.load(f"{R}/dump_loncomb3_s1/{f}")
        base = {k: z0[k] for k in z0.files}
        arms = {"b50_s0": 0.5 * z0["cl"] + 0.5 * z0["ha0_ext"],
                "b50_s1": 0.5 * z1["cl"] + 0.5 * z0["ha0_ext"],
                "damp50": 0.5 * z0["ha0"] + 0.5 * z0["ha0_ext"],
                "dampha": 0.5 * z0["ha0"] + 0.5 * z0["ha"],
                "w0": z0["ha0_ext"]}
        for a, cl in arms.items():
            os.makedirs(f"{OUT}/dump_{a}", exist_ok=True)
            np.savez(f"{OUT}/dump_{a}/{f}", **{**base, "cl": cl.astype(np.float32)})
    print("episodes", len(names), "->", OUT)


if __name__ == "__main__":
    main()
