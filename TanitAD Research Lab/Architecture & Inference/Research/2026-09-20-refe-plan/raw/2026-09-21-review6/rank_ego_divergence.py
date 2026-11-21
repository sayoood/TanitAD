"""If two ranks are two CLOSED-LOOP rollouts of the same log, their ego is in a DIFFERENT PLACE at
step i -- but both rows carry the SAME camera frames. Quantify the disagreement.
Expectation written as a LITERAL: identical frames imply identical observed scene; if the two rows'
ego speed at step i differs by more than sensor noise, at least one row pairs a real image with an
ego that was never there."""
import json, numpy as np, collections
B = r"D:/Projects/TanitAD/data/refe_targets_4cam"
r0 = {(r["token"], r["step"]): r for r in map(json.loads, open(B+"/targets_rank0.jsonl", encoding="utf-8"))}
r1 = {(r["token"], r["step"]): r for r in map(json.loads, open(B+"/targets_rank1.jsonl", encoding="utf-8"))}
ks = sorted(set(r0) & set(r1))
print(f"paired keys: {len(ks)}")
img_same = sum(1 for k in ks if r0[k]["image"] == r1[k]["image"])
print(f"image list IDENTICAL on {img_same}/{len(ks)} keys")
sp = np.array([[r0[k]["ego"][6], r1[k]["ego"][6]] for k in ks])
d_sp = np.abs(sp[:,0]-sp[:,1])
print(f"ego SPEED |rank0 - rank1|: mean {d_sp.mean():.4f} m/s  median {np.median(d_sp):.4f}  max {d_sp.max():.4f}")
print(f"  frames where speed differs by >0.5 m/s: {(d_sp>0.5).sum()} / {len(ks)} ({100*(d_sp>0.5).mean():.1f} %)")
print(f"  frames where speed differs by >2.0 m/s: {(d_sp>2.0).sum()} ({100*(d_sp>2.0).mean():.1f} %)")
# integrate |speed difference| * dt to bound the along-track separation after k steps within a log
# traj endpoint separation (the supervision target itself)
t0 = np.array([r0[k]["traj"] for k in ks]); t1 = np.array([r1[k]["traj"] for k in ks])
ep = np.linalg.norm(t0[:,-1,:2]-t1[:,-1,:2], axis=-1)
print(f"4 s trajectory ENDPOINT separation between ranks: mean {ep.mean():.3f} m  median {np.median(ep):.3f}  max {ep.max():.3f}")
g0 = np.array([r0[k]["goal"] for k in ks]); g1 = np.array([r1[k]["goal"] for k in ks])
gd = np.linalg.norm((g0-g1).reshape(len(ks),-1,2), axis=-1).max(-1)
print(f"goal separation: median {np.median(gd):.3f} m  identical(<1e-9) on {(gd<1e-9).sum()} keys")
# per-log drift: does the speed gap GROW with step (a closed-loop divergence signature)?
bylog = collections.defaultdict(list)
for k in ks: bylog[r0[k]["log_name"]].append((r0[k]["step"], abs(r0[k]["ego"][6]-r1[k]["ego"][6])))
print("\nspeed gap vs step, per log (first quarter -> last quarter):")
for lg, v in sorted(bylog.items()):
    v.sort(); a = np.array([x[1] for x in v]); q = len(a)//4
    print(f"  {lg[:38]:38s} n={len(a):4d}  first25% {a[:q].mean():6.3f}  last25% {a[-q:].mean():6.3f} m/s")
