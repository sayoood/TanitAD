import numpy as np, os, sys, subprocess
d=sys.argv[1]; delta=float(sys.argv[2]); os.makedirs(d,exist_ok=True)
rng=np.random.default_rng(1); n=2399; eid=np.repeat(np.arange(40),n//40+1)[:n]
for s in (0,1):
    base=rng.gamma(4,0.5,n)
    for arm,off in (("B0",0.0),("RH",0.1),("KI",0.1+delta),("RK",0.1)):
        np.savez(f"{d}/eval_{arm}_s{s}.npz",ade6=base+off+rng.normal(0,0.01,n),eid=eid)
