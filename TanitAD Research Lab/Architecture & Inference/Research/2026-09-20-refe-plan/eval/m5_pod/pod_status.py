"""READ-ONLY pod probe for measure 5's effectiveness run (streamed over ssh stdin; writes NOTHING on the pod).

Prints ONE JSON line: the live trainer's recent step times (from the tail of train.log), the lr the live run logged at
snapshot 015's step (4933), the pixel root's top-level layout, and the sizes of the bank and calibration files.
    ssh -o BatchMode=yes -i ~/.ssh/tanitad_pod -p <port> root@<host> 'nice -n 19 python3 -' < pod_status.py
"""
import json
import os
import re
import time

RUN = "/workspace/data/refe_runs/vitl16_navtrain10_grow_tau0.3"
BANK = "/workspace/data/refe_navtrain/train_grow"
PIX = "/workspace/data/navtrain_pixels"
SNAP_STEP = 4933
out = {"at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def tail_lines(path, nbytes=400_000):
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - nbytes))
        return f.read().decode("utf-8", "replace").splitlines()


try:
    pat = re.compile(r"^\s+step\s+(\d+)\s+traj_L1.*?\s([\d.]+)s\s*$")
    pts = [(int(m.group(1)), float(m.group(2))) for m in (pat.match(l) for l in tail_lines(RUN + "/train.log")) if m]
    pts = pts[-12:]
    d = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(pts, pts[1:]) if b[0] > a[0]]
    out["last_step"] = pts[-1][0] if pts else None
    out["step_s_recent"] = [round(x, 2) for x in d]
    out["step_s_median"] = round(sorted(d)[len(d) // 2], 2) if d else None
    out["train_log_age_s"] = round(time.time() - os.path.getmtime(RUN + "/train.log"), 1)
except Exception as exc:
    out["train_log_error"] = repr(exc)[:200]
try:
    lr_at, best = None, None
    with open(RUN + "/metrics.jsonl", encoding="utf-8") as f:
        for line in f:
            if '"lr"' not in line:
                continue
            r = json.loads(line)
            if "step" in r and "lr" in r:
                dist = abs(int(r["step"]) - SNAP_STEP)
                if best is None or dist < best:
                    best, lr_at = dist, (int(r["step"]), r["lr"], r.get("epoch"))
    out["lr_near_snapshot_015"] = lr_at
except Exception as exc:
    out["metrics_error"] = repr(exc)[:200]
try:
    top = sorted(os.listdir(PIX))
    out["pix_top_n"] = len(top)
    out["pix_top_head"] = top[:12]
    out["pix_zips"] = sum(1 for n in top if n.endswith(".zip"))
except Exception as exc:
    out["pix_error"] = repr(exc)[:200]
for k, p in (("bank_rank0_bytes", BANK + "/targets_rank0.jsonl"), ("calib_bytes", BANK + "/calib_table.json")):
    try:
        out[k] = os.path.getsize(p)
    except OSError as exc:
        out[k] = repr(exc)[:120]
print(json.dumps(out))
