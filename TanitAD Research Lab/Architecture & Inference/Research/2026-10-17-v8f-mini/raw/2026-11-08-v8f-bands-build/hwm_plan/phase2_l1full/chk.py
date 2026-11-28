import numpy as np, json
B="F:/TanitAD-scratch/hwm_plan/x"
for s in ("s0","s1"):
    a=np.load(f"{B}/smokeL1F_{s}_a/plans_hwm_oracle_s0.npz"); b=np.load(f"{B}/smokeL1F_{s}_b/plans_hwm_oracle_s0.npz")
    print(s, a.files)
    for k in a.files:
        if k=="wall_s": continue
        print(" ",k,a[k].shape,"finite",bool(np.isfinite(a[k]).all()) if a[k].dtype.kind=="f" else None,"bit_equal",bool(np.array_equal(a[k],b[k])))
    g=json.loads(str(np.load(f"{B}/smokeL1F_{s}_a/windows.npz")["meta"]))["geometry"]; print(" geom",g)
