import sys
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
import numpy as np, glob
import l2_common as C
D = C.TRAIN_TREE30 / "dump_s0"
T = C.load_train_dump(D)
m = T["valid_s6"]
print("rows", T["n_rows"], "S6-valid", int(m.sum()), "compliant&valid", int((m & T["compliant"]).sum()))
# control: equals the panel-based derivation (s6_s0/ep*.npz g,v0) window for window
gcls, wsl = [], []
for f in sorted(glob.glob(str(C.TRAIN_TREE30 / "s6_s0" / "ep*.npz"))):
    a = np.load(f)
    gcls.append(C.plan_class(a["g"].astype(np.float64), a["v0"].astype(np.float64), 1.0)); wsl.append(a["ws"])
gcls = np.concatenate(gcls); wsl = np.concatenate(wsl)
sel = np.where(m)[0]
print("panel windows", len(wsl), "dump S6-valid", len(sel), "ws equal", bool(np.array_equal(T["ws"][sel], wsl)), "class equal", bool(np.array_equal(T["gt"][sel], gcls)))
print({c: int(((T['gt'] == c) & m & T['compliant']).sum()) for c in C.CLS})
