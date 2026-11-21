"""make_standin.py -- STAND-IN artifacts for the dry run of the NAVRB2 read (a code check, never a result): every R arm (and its zero twin) is presented as a COPY of the banked arm it is paired with
(N1R_s* <- N1_s*, S1R_s* <- S1_s*), dumps, ticks and T0 passes alike.  R - banked then reads exactly 0.0 everywhere and real - zero reads exactly 0.0, so the pipeline's flow, files and controls are exercised.
Usage: python make_standin.py ROOT [--perturb-zero ARM]   (--perturb-zero rewrites ONE window of that arm's zero twin: the Z-EVAL deliberate-regression arm; it must go RED)"""
import glob
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import navrb2_lib as NL  # noqa: E402


def build(root, perturb_zero=None):
    t1, t0 = f"{root}/t1", f"{root}/out/t0"
    os.makedirs(f"{t1}/ticks", exist_ok=True)
    os.makedirs(t0, exist_ok=True)
    for arm in NL.R_ARMS:
        b = NL.BANKED_OF[arm]
        for z in ("", NL.Z):
            src = f"{NL.BANKED_DUMPS}/dump_STRAT1-{b}_m10_causal_heads"
            dst = f"{t1}/dump_NAVRB2-{arm}{z}_m10_causal_heads"
            if not os.path.isdir(dst):
                shutil.copytree(src, dst)
        for z in ("",):
            shutil.copyfile(f"{NL.BANKED_DUMPS}/ticks/ticks_STRAT1-{b}_m10_causal_heads.npz", f"{t1}/ticks/ticks_NAVRB2-{arm}_m10_causal_heads.npz")
        for v in ("real", "donor", "zero") + (("k4",) if arm.startswith("S1R") else ()):
            shutil.copyfile(f"{NL.BANKED_T0}/eval_STRAT1-{b}_{v}.npz", f"{t0}/eval_NAVRB2-{arm}_{v}.npz")
    if perturb_zero:
        import numpy as np
        d = f"{t1}/dump_NAVRB2-{perturb_zero}{NL.Z}_m10_causal_heads"
        f = sorted(glob.glob(d + "/ep*.npz"))[0]
        z = dict(np.load(f))
        z["cl"] = z["cl"].copy()
        z["cl"].reshape(-1)[0] += 1.0            # one value of one window: any zero-history window it belongs to breaks Z-EVAL
        np.savez(f, **z)
    return root


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[sys.argv.index("--perturb-zero") + 1] if "--perturb-zero" in sys.argv else None)
    print("stand-in built", sys.argv[1])
