import os, sys, json, shutil, numpy as np
sys.path.insert(0, "F:/TanitAD-scratch/q5_416/build/c2"); sys.path.insert(0, "F:/TanitAD-scratch/q5_416/build")
import q5_c2_planted as P
R = "F:/TanitAD-scratch/q5_416/run"; B = "F:/TanitAD-scratch/q5_416/build/c2/tmp/smoke_base"
n = 200
d = B + "/assembled/S1_416"; os.makedirs(d)
s = R + "/assembled/S1_416"
y0 = np.load(s + "/y.npy"); idx = np.r_[np.nonzero(y0)[0][:60], np.nonzero(~y0.astype(bool))[0][:140]]
for k in ("E","y","clip","ws","eid","v40","v160","mean"):
    np.save(f"{d}/{k}.npy", np.load(f"{s}/{k}.npy", mmap_mode="r")[idx])
shutil.copyfile(s + "/meta.json", d + "/meta.json")
z = np.load(R + "/assembled/eval_416.npz"); ye = z["y"]; ie = np.r_[np.nonzero(ye)[0][:40], np.nonzero(ye==0)[0][:160]]
np.savez(B + "/assembled/eval_416.npz", **{k: (z[k][ie] if z[k].ndim else z[k]) for k in z.files})
P.make(B, ["S1"])
c = B + "/c2/assembled/S1_416"
for k in P.TRAIN_FILES: assert os.path.samefile(f"{d}/{k}.npy", f"{c}/{k}.npy"), k
a, b = np.load(d + "/v40.npy"), np.load(c + "/v40.npy"); y = np.load(d + "/y.npy").astype(bool)
print("v40", a.shape, a.dtype, b.shape, b.dtype)
m = np.ones(a.shape, bool); m[y, 25] = False
assert (a[m] == b[m]).all() and (a[y, 25] != b[y, 25]).any()
ev0, ev1 = np.load(B+"/assembled/eval_416.npz"), np.load(B+"/c2/assembled/eval_416.npz")
for k in ev0.files:
    if k not in ("v40","meta"): assert ev0[k].tobytes()==ev1[k].tobytes() and ev0[k].dtype==ev1[k].dtype, k
print("eval v40", ev1["v40"].shape, ev1["v40"].dtype, "changed windows", int((ev0["v40"]!=ev1["v40"]).reshape(len(ie),-1).any(1).sum()), "pos", int(ev0["y"].sum()))
print(json.load(open(B+"/c2/c2_make.json"))["median_token_norm"], "SMOKE OK")
