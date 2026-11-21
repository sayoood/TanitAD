"""DECN post-hoc diagnostic (Master Mind, 2026-11-09, written AFTER control 3 failed; it decides nothing and changes no bar). The ten MLP rows that read above 0.05 on one permutation are refitted on 8 fresh permutations each (seeds 200..207), to see the spread of the control itself. Same `mlp_read_c`, same MLP seed 0, same epoch cap, same held-out set. CPU. Writes null_shuffled.json."""
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

import decn_lib as DN
import decinfo_lib as DL

ROWS = [("R0", "M2b_s0", "z1", "N1053", 1, "speed", 0.058), ("R0", "M2b_s0", "z1", "N1053", 4, "speed", 0.102), ("R0", "M2b_s0", "z1", "N194", 1, "yaw", 0.090), ("R0", "M2b_s1", "z1", "N194", 1, "yaw", 0.112),
        ("R1", "R35_s0", "z1", "N1053", 4, "speed", 0.061), ("R1", "R35_s0", "z1", "N194", 1, "yaw", 0.122), ("R1", "R35_s1", "z1", "N194", 1, "yaw", 0.158), ("R1", "S35_s0", "z1", "N194", 1, "yaw", 0.101),
        ("R2", "M2b_s0", "pooled", "N1053", 1, "speed", 0.052), ("R2", "M2b_s1", "pooled", "N194", 4, "yaw", 0.068)]
SEEDS = range(200, 208)
KCLIPS = {"N194": 141, "N1053": 1000}
OUT = os.path.join(HERE, "null_shuffled.json")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    c141, ce = DN.clip_lists()
    order = list(c141["clips"]) + list(ce["clips"])
    T = DN.load_tune_npz()
    out = []
    for rung, tag, kind, N, m, tgt, first in ROWS:
        DL.wait_ram()
        k = KCLIPS[N]
        key = f"{kind}_{tag}"
        X_tr, ci_tr = DN.load_train_arrays(order, k, key)
        vs, ys = [], []
        for c in order[:k]:
            z = np.load(f"{DN.LAT}/{c}.npz")
            vs.append(z["v"]); ys.append(z["yaw"])
        D, X, _ = DN.assemble(T, key, X_tr, ci_tr, order, k, (np.concatenate(vs), np.concatenate(ys)))
        y = DL.targets(D)[tgt]
        Xs = DL.stack_ch(X, m)
        yS = y[:, m - 1:]
        vals = []
        for s in SEEDS:
            ysh = DN.derange_gen(D, y, seed=s)[:, m - 1:]
            r, e = DN.mlp_read_c(D, Xs, yS, ysh, DN.CAP_BIG)
            vals.append(float(r))
            log(rung, tag, N, f"m={m}", tgt, "perm", s, f"{r:+.3f} ep {e['epochs']}")
        rec = dict(rung=rung, tag=tag, N=N, m=m, target=tgt, first_run_shuffled=first, perm_seeds=list(SEEDS), r2_shuffled=vals, mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)), max=float(np.max(vals)), n_above_0p05=int(sum(v > 0.05 for v in vals)))
        out.append(rec)
        json.dump(dict(tier="T0", note="post-hoc diagnostic of DECN control 3; decides nothing", rows=out), open(OUT, "w", encoding="utf-8"), indent=1)
        log("row done", rung, tag, N, m, tgt, f"mean {rec['mean']:+.3f} sd {rec['sd']:.3f} max {rec['max']:+.3f} above {rec['n_above_0p05']}/8")
        del D, X, X_tr, Xs
    log("complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
