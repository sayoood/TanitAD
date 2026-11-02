#!/usr/bin/env python3
"""k1_diag_compare.py -- SMOOTH-T1 K1 diagnostic: pairwise deviation between the dump/ticks of DAG-OACC s0 rollouts that differ only in tool patch or thread count.  Supplementary to the frozen card (K1 as written is read in smooth_t1_read.py)."""
import glob, json, os, sys
import numpy as np
sys.dont_write_bytecode = True
O = "F:/TanitAD-scratch/smooth_t1"
B = "E:/Projects/TanitAD-artifacts/v8f-dagoacc-t1"
RUNS = {"banked(16thr,unpatched)": (f"{B}/dump_DAGOACC_s0_m10_causal_heads", f"{B}/ticks/ticks_DAGOACC_s0_m10_causal_heads.npz"),
        "K1OFF(1thr,patched,off)": (f"{O}/out/K1OFF_DAGOACC_s0/dump_K1OFF_DAGOACC_s0", f"{O}/out/K1OFF_DAGOACC_s0/ticks/ticks_K1OFF_DAGOACC_s0.npz"),
        "K1U(1thr,unpatched)": (f"{O}/out_diag/K1U_DAGOACC_s0/dump_K1U_DAGOACC_s0", f"{O}/out_diag/K1U_DAGOACC_s0/ticks/ticks_K1U_DAGOACC_s0.npz"),
        "K1P3(3thr,patched,off)": (f"{O}/out_diag/K1P3_DAGOACC_s0/dump_K1P3_DAGOACC_s0", f"{O}/out_diag/K1P3_DAGOACC_s0/ticks/ticks_K1P3_DAGOACC_s0.npz")}
def load(d, t):
    zs = [np.load(f) for f in sorted(glob.glob(os.path.join(d, "ep*.npz")))]
    dump = {k: np.concatenate([z[k] for z in zs]) for k in zs[0].files}
    tk = {k: v for k, v in dict(np.load(t, allow_pickle=True)).items() if k not in ("tag", "ckpt")}
    return dump, tk
def cmp(a, b):
    out = {}
    for k in sorted(set(a) & set(b)):
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape:
            out[k] = "SHAPE"; continue
        if x.dtype.kind in "fc":
            if np.array_equal(x, y, equal_nan=True):
                out[k] = 0.0; continue
            d = np.abs(np.where(np.isnan(x) & np.isnan(y), 0.0, x.astype(np.float64) - y.astype(np.float64)))
            out[k] = float(d.max())
        else:
            out[k] = 0.0 if np.array_equal(x, y) else "DIFF"
    return out
D = {k: load(*v) for k, v in RUNS.items()}
names = list(D)
res = {}
for i, a in enumerate(names):
    for b in names[i + 1:]:
        dd, tt = cmp(D[a][0], D[b][0]), cmp(D[a][1], D[b][1])
        num = lambda m: max([v for v in m.values() if isinstance(v, float)] or [0.0])
        res[f"{a} vs {b}"] = {"dump_max_abs_dev": num(dd), "dump_by_key": {k: v for k, v in dd.items() if v != 0.0}, "ticks_max_abs_dev": num(tt), "ticks_by_key": {k: v for k, v in tt.items() if v != 0.0},
                              "bit_identical": bool(num(dd) == 0.0 and num(tt) == 0.0 and all(v == 0.0 for v in list(dd.values()) + list(tt.values())))}
        print(f"{a} vs {b}: dump max {num(dd):.3e}  ticks max {num(tt):.3e}  bit_identical={res[f'{a} vs {b}']['bit_identical']}")
json.dump(res, open(sys.argv[1], "w", encoding="utf-8", newline="\n"), indent=1)
