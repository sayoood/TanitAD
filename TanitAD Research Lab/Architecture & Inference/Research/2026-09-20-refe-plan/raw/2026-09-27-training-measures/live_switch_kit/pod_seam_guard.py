# READ-ONLY seam-guard check over the FULL pod bank for `--yaw-loss plain` (measures.check_plain_yaw_targets' rule:
# every target heading must satisfy |h| <= 3.0 rad, else the patched trainer REFUSES -- at start, and at every --grow
# epoch rebuild). Checks EVERY row of every train_grow/targets_rank*.jsonl the trainer's TargetBank globs (a superset
# of the rows it trains on). Stdlib only; writes NOTHING; reads throttled. Run with the script on STDIN:
#   ssh ... "nice -n 19 ionice -c3 python3 -B -u - " < pod_seam_guard.py
# Prints one JSON line and ZZSEAMGUARD_PASS / ZZSEAMGUARD_FAIL.
import glob
import json
import os
import sys
import time

BANK = "/workspace/data/refe_navtrain/train_grow"
LIMIT = 3.0                                   # measures.PLAIN_YAW_MAX_TARGET
RATE = 20e6                                   # bytes/s read throttle
t0, nread = time.time(), 0
res = {"bank": BANK, "limit_rad": LIMIT, "files": {}, "worst": []}
gmax = 0.0
for p in sorted(glob.glob(os.path.join(BANK, "targets_rank*.jsonl"))):
    size = os.path.getsize(p)
    rows = bad = over = 0
    fmax = 0.0
    with open(p, "rb") as f:
        pos, pend = 0, b""
        while pos < size:
            chunk = f.read(min(8 << 20, size - pos))
            if not chunk:
                break
            pos += len(chunk)
            nread += len(chunk)
            ahead = nread / RATE - (time.time() - t0)
            if ahead > 0:
                time.sleep(ahead)
            data = pend + chunk
            lines = data.split(b"\n")
            pend = lines.pop()                # an incomplete last line is not a row yet
            for line in lines:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    m = max(abs(float(q[2])) for q in r["traj"])
                except Exception:
                    bad += 1
                    continue
                rows += 1
                fmax = max(fmax, m)
                if m > LIMIT:
                    over += 1
                    if len(res["worst"]) < 10:
                        res["worst"].append([os.path.basename(p), r.get("log_name"), r.get("token"), r.get("step"),
                                             r.get("rank"), round(m, 4)])
    gmax = max(gmax, fmax)
    res["files"][os.path.basename(p)] = {"bytes": size, "rows": rows, "unparseable": bad, "max_abs_heading": round(fmax, 4),
                                         "rows_over_limit": over}
    sys.stderr.write("%s %s: %d rows, max |h| %.4f, over %d\n" % (time.strftime("%H:%M:%S"), os.path.basename(p), rows,
                                                                  fmax, over))
res["max_abs_heading"] = round(gmax, 4)
res["rows_over_limit"] = sum(v["rows_over_limit"] for v in res["files"].values())
res["seconds"] = round(time.time() - t0, 1)
res["pass"] = res["rows_over_limit"] == 0 and bool(res["files"])
print(json.dumps(res))
print("ZZSEAMGUARD_PASS" if res["pass"] else "ZZSEAMGUARD_FAIL")
