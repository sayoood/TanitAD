"""Verify the banked refav1-21109 plan facts BY CONTENT from the dump sidecars."""
import glob, json, os, sys
import numpy as np

D = r"C:\Users\Admin\tanitad-bench\refav1_eval_21109\full_dump"
man = json.load(open(os.path.join(D, "manifest.json")))
names = man["plan_source_names"]
print("plan_source_names:", names)

files = sorted(glob.glob(os.path.join(D, "decisions", "ep*.npz")))
print("n decision files:", len(files))
z = np.load(files[0])
print("sidecar keys:", sorted(z.files))

ctrl, src, cost, eids = [], [], [], []
arm = None
for f in files:
    z = np.load(f)
    if arm is None:
        cands = [k for k in z.files if k.endswith("_controls")]
        arm = cands[0][: -len("_controls")]
        print("arm key:", arm, "cands:", cands)
    ctrl.append(z[f"{arm}_controls"])
    src.append(z[f"plan_source_{arm}"])
    cost.append(z[f"plan_cost_{arm}"])
    eids.append(np.full(z[f"plan_source_{arm}"].shape[0], int(os.path.basename(f)[2:5])))

C = np.concatenate(ctrl, 0)          # [N, H, 2] (a, kappa)
S = np.concatenate(src, 0)
CO = np.concatenate(cost, 0)
E = np.concatenate(eids, 0)
N = C.shape[0]
print(f"\nN windows = {N}, H = {C.shape[1]}, A = {C.shape[2]}, clusters = {len(np.unique(E))}")

kap = C[..., 1]
acc = C[..., 0]
kzero = (kap == 0.0).all(-1)
print(f"kappa identically 0 on {kzero.sum()}/{N} = {kzero.mean()*100:.2f}%")
print(f"max |kappa| over all windows/steps: {np.abs(kap).max():.6g}")
acc_const = (acc.max(-1) - acc.min(-1) == 0.0)
print(f"accel constant in time on {acc_const.sum()}/{N} = {acc_const.mean()*100:.2f}%")

# distinct plans, bit-exact
rows = C.reshape(N, -1)
uniq, inv, cnt = np.unique(rows, axis=0, return_inverse=True, return_counts=True)
print(f"\nDISTINCT PLANS (bit-exact): {uniq.shape[0]}")
order = np.argsort(-cnt)
for j in order:
    u = uniq[j].reshape(C.shape[1], 2)
    print(f"  n={cnt[j]:4d}  a={np.unique(u[:,0])}  kappa={np.unique(u[:,1])}")

print("\nplan_source histogram (LABEL):")
for i, nm in enumerate(names):
    c = int((S == i).sum())
    if c:
        print(f"  {nm:24s} {c:4d}  ({c/N*100:.2f}%)")
cem_i = names.index("cem")
print(f"baseline_won_frac AS REPORTED (label-based) = {float((S != cem_i).mean()):.4f}")

# CONTENT-based: is the plan bit-identical to an injected baseline?
H = C.shape[1]
zero = np.zeros((H, 2), dtype=C.dtype)
dec = zero.copy(); dec[:, 0] = -1.5
is_zero = np.all(C == zero, axis=(1, 2))
is_dec = np.all(C == dec, axis=(1, 2))
print(f"\nCONTENT: plan == zeros (cv/hold_v0)   : {is_zero.sum()}/{N}")
print(f"CONTENT: plan == decel_1.5 (-1.5, 0)  : {is_dec.sum()}/{N}")
triv = is_zero | is_dec
print(f"CONTENT: plan is a TRIVIAL BASELINE   : {triv.sum()}/{N} = {triv.mean():.4f}")
print("\nlabel vs content cross-tab:")
for i, nm in enumerate(names):
    m = S == i
    if m.sum() == 0:
        continue
    print(f"  {nm:24s} n={int(m.sum()):4d}  zeros={int(is_zero[m].sum()):4d} decel={int(is_dec[m].sum()):4d} other={int((~triv[m]).sum()):4d}")

print(f"\nplan cost: min={CO.min():.10g} max={CO.max():.10g} mean={CO.mean():.10g}")
print(f"plan cost distinct values: {len(np.unique(CO))}")
u2, c2 = np.unique(np.round(CO, 12), return_counts=True)
for v, c in sorted(zip(u2, c2), key=lambda t: -t[1])[:8]:
    print(f"   cost={v:.12g}  n={c}")

# goal / neval
z = np.load(files[0])
for k in sorted(z.files):
    if k.startswith("goal") or k.startswith("plan_neval") or k.startswith("plan_agree"):
        print("sample key", k, z[k].dtype, z[k].shape, z[k][:4])
