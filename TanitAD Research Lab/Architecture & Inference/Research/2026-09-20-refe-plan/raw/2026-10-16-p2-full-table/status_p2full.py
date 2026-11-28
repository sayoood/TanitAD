"""One-screen status of the P2 full-table job (no model turns, read-only).
    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I status_p2full.py
Prints: markers, driver pid alive?, heartbeat age, stage/state, chunks dumped, columns scored, rate, ETA, the GPU lock holder, D: free space, failures."""
import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock"


def alive(pid):
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True)
    return f'"{pid}"' in r.stdout.decode("utf-8", "replace")


def main():
    for m in ("P2FULL_DONE", "P2FULL_FAILED"):
        p = os.path.join(HERE, m)
        if os.path.exists(p):
            print(f"MARKER {m}:", open(p, encoding="utf-8").read()[:1500])
    sp = os.path.join(HERE, "STATUS.json")
    if not os.path.exists(sp):
        print("no STATUS.json: the job was never started from this directory")
        return
    s = json.load(open(sp, encoding="utf-8"))
    pid = s.get("pid")
    hb = s.get("heartbeat") or s.get("updated")
    age = None
    try:
        age = round(time.time() - time.mktime(time.strptime(hb[:19], "%Y-%m-%dT%H:%M:%S")))
    except Exception:  # noqa: BLE001
        pass
    print(f"driver pid {pid} alive={alive(pid) if pid else None}; heartbeat {hb} ({age} s ago; > 180 s with the process alive is a hang)")
    print("stage:", s.get("stage"), "|", str(s.get("state"))[:240])
    if s.get("guard"):
        print("GUARD:", s["guard"][:240])
    pr = s.get("progress") or {}
    print("progress:", pr)
    if pr.get("score_cols_per_min"):
        left = pr["score_total"] - pr["score_pass"]
        print(f"scoring ETA ~{left / pr['score_cols_per_min'] / 60:.1f} h at {pr['score_cols_per_min']} cols/min (columns vary 14-175 s; the rate is a running mean)")
    for k in sorted(k for k in s if k.startswith("gate_c")):
        print(k, s[k])
    if s.get("score_failed_cols"):
        print("FAILED COLUMNS:", s["score_failed_cols"][:20])
    if s.get("abort_reason") or s.get("result"):
        print("result/abort:", s.get("result"), "|", s.get("abort_reason"))
    try:
        print("GPU lock:", open(LOCK, encoding="utf-8").read()[:200])
    except OSError:
        print("GPU lock: (free)")
    try:
        import shutil
        print("D: free GiB:", round(shutil.disk_usage("D:/").free / 2 ** 30, 1), "| C: free GiB:", round(shutil.disk_usage("C:/").free / 2 ** 30, 1))
    except OSError:
        pass


if __name__ == "__main__":
    main()
