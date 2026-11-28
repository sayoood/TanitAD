"""MM 2026-11-16, read-only: the single TB-5 row. Its RJ5 traj (feed bank) step lengths, its traj tangent vs label yaw, and whether it is a standstill row."""
import glob, json, math
import numpy as np
W = "/workspace/data/p5c_label_a32"
row = None
for fp in sorted(glob.glob(W + "/out/labels_s*of15.jsonl")):
    for ln in open(fp, encoding="utf-8"):
        if '"p5r_S2_yaw_branch": false' in ln:
            row = json.loads(ln); break
    if row: break
k = (row.get("log"), row.get("token"), row.get("rank"))
print("row rank", k[2], "token", str(k[1])[:12], "traj_sha", str(row.get("traj_sha256"))[:12], "fields", sorted(row)[:40])
traj = None
for fp in glob.glob(W + "/feed/bank/targets_rank%d.jsonl" % k[2]):
    for ln in open(fp, encoding="utf-8"):
        if k[1] in ln:
            r = json.loads(ln)
            if r.get("token") == k[1] and r.get("log", r.get("log_name")) == k[0]:
                traj = np.asarray(r["traj"], float); break
print("traj shape", None if traj is None else traj.shape)
if traj is not None:
    xy = traj[:, :2]
    steps = np.linalg.norm(np.diff(np.vstack([[0, 0], xy]), axis=0), axis=1)
    print("step lengths m: min %.4f max %.4f total %.3f; steps < 0.05 m: %d of %d" % (steps.min(), steps.max(), steps.sum(), int((steps < 0.05).sum()), len(steps)))
    if traj.shape[1] > 2:
        print("traj yaw first/last %.3f %.3f; min %.3f max %.3f" % (traj[0, 2], traj[-1, 2], traj[:, 2].min(), traj[:, 2].max()))
for key in ("p5r", "tc20s"):
    if key in row:
        lab = np.asarray(row[key], float)
        if lab.ndim == 2 and lab.shape[1] >= 3:
            print(key, "label yaw min %.3f max %.3f; xy end (%.2f, %.2f)" % (lab[:, 2].min(), lab[:, 2].max(), lab[-1, 0], lab[-1, 1]))
            if traj is not None and traj.shape[1] > 2:
                dy = (lab[:, 2] - traj[:lab.shape[0], 2])
                print(key, "max |label yaw - traj yaw| %.3f at step %d" % (np.abs(dy).max(), int(np.abs(dy).argmax())))
print("ZZTB5_PROBE_END")
