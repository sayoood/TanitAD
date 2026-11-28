"""MM 2026-11-16, read-only: (1) TB-0r Tc20s distribution, pod RJ5-pass labels vs the dev-box RJ5 labels (rj5_labels_local);
(2) the TB-5 row (yaw branch >= pi / S2 false): rank, source, values."""
import glob, gzip, json, math
import numpy as np
W = "/workspace/data/p5c_label_a32"


def rows(files):
    for fp in files:
        op = gzip.open if fp.endswith(".gz") else open
        with op(fp, "rt", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    yield json.loads(ln)


def key(r):
    return (r.get("log") or r.get("log_name"), r.get("token"), int(r.get("rank", 0)), r.get("traj_sha256"))


pod = {key(r): r for r in rows(sorted(glob.glob(W + "/out/labels_s*of15.jsonl"))) if "tc20s" in r and "error" not in r}
dev = {key(r): r for r in rows(["/workspace/data/p5c_ship/rj5_labels_local.jsonl.gz"]) if "tc20s" in r and "error" not in r}
common = [k for k in dev if k in pod]
d = np.array([float(np.abs(np.asarray(pod[k]["tc20s"], float) - np.asarray(dev[k]["tc20s"], float)).max()) for k in common])
print("TB-0r tc20s: dev", len(dev), "pod", len(pod), "common", len(common))
for t in (0.0, 1e-12, 1e-10, 5e-10, 1e-9, 2e-9):
    print("  rows > %g: %d" % (t, int((d > t).sum())))
print("  p50 %.3e p99 %.3e p99.9 %.3e max %.3e" % tuple(np.quantile(d, [0.5, 0.99, 0.999, 1.0])))
# TB-5: the S2-false / yaw-branch row, over the pod RJ5-pass outputs
bad = [r for r in rows(sorted(glob.glob(W + "/out/labels_s*of15.jsonl"))) if r.get("p5r_S2_yaw_branch") is False]
print("TB-5 rows with p5r_S2_yaw_branch False:", len(bad))
for r in bad[:3]:
    print("  rank", r.get("rank"), "log", (r.get("log") or "")[:40], "p5r_accepted", r.get("p5r_accepted"), "S1", r.get("p5r_S1_finite"), "S5", r.get("p5r_S5_lambda_reached"),
          "E_maxxy", r.get("p5r_E_maxxy"), "calls", r.get("p5r_calls"), "keys_with_yaw", [k for k in r if "yaw" in k.lower()][:6])
print("ZZPROBE2_END")
