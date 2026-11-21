import sys
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
import numpy as np
import l2_common as C
T = C.load_train_tree(C.TRAIN_TREE30)
print({k: (v.shape if hasattr(v,'shape') else v) for k,v in T.items() if k not in ('logits',)})
print("compliant", T["compliant"].sum(), "v0 diff extras", float(np.abs(T["v0"]-T["v0_extras"]).max()))
gt = C.traj_class(T["g"], T["v0"], T["dt"])
for c in C.CLS: print(c, int((gt==c).sum()), int(((gt==c)&T["compliant"]).sum()))
L6 = C.lon6(T["logits"]); am = L6.argmax(1)
names = np.array(C.LON6)
print("head argmax counts (train, S6 windows)", {n:int((am==i).sum()) for i,n in enumerate(C.LON6)})
# label vs traj class
lab = T["lon_v7"]
print("label counts", {int(k):int(v) for k,v in zip(*np.unique(lab, return_counts=True))})
print("traj class by label:")
for l in (0,1,3,4,5,7,-100):
    m = lab==l
    print("  label", l, "n", int(m.sum()), {c:int(((gt==c)&m).sum()) for c in C.CLS})
print("head argmax (rows) by traj class (cols decel/cruise/accel), train in-sample")
for i,n in enumerate(C.LON6):
    m = am==i
    print("  ", n, int(m.sum()), {c:int(((gt==c)&m).sum()) for c in C.CLS})
dec = np.isin(am,[2,3,4])
c = T["compliant"]
print("TRAIN in-sample head-decel rate on GT-cruise compliant:", dec[(gt=='cruise')&c].mean(), " head-nondecel on GT-decel:", (~dec)[(gt=='decel')&c].mean())
