import numpy as np, json, hashlib
B="F:/TanitAD-scratch/hwm_plan/x"
for s in ("s0","s1"):
    a=np.load(f"{B}/smokeR35_{s}_a/plans_hwm_oracle_s0.npz"); b=np.load(f"{B}/smokeR35_{s}_b/plans_hwm_oracle_s0.npz")
    f=np.load(f"{B}/smokeR35_{s}_a/plans_flat_oracle_s0.npz")
    print(s, "keys", list(a.keys()))
    for k in a.files:
        if k=="wall_s": continue
        fin=bool(np.isfinite(a[k]).all()) if a[k].dtype.kind=="f" else None
        print("  ",k,a[k].shape,"finite",fin,"repeat_bit_equal",bool(np.array_equal(a[k],b[k])), "flat finite", bool(np.isfinite(f[k]).all()) if k in f.files and f[k].dtype.kind=="f" else None)
    m=json.loads(str(np.load(f"{B}/smokeR35_{s}_a/windows.npz")["meta"])); print("  geom",m["geometry"] if isinstance(m["geometry"],str) else m["geometry"])
