import numpy as np
for p in ["C:/Users/Admin/rr_scratch/out/C0C_s0/ticks/ticks_C0C_s0.npz"]:
    z=np.load(p,allow_pickle=True)
    for k in z.files:
        a=z[k]; print(k,a.shape,a.dtype, (a.ravel()[:3] if a.size else ''))
