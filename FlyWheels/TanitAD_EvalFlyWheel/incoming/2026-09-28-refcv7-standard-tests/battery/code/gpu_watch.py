"""Collision watch for the dev-box GPU while a milestone chain holds the lock (Master Mind 2026-09-28:
"report any such collision").

    python gpu_watch.py --out D:/refcv7_eval_kit/chain/gpu_watch_5000.jsonl --stop-file <path> [--every 60]

Every `--every` seconds: the python compute apps on the card (`nvidia-smi`), each with its command line
(`Win32_Process`, read-only). A process whose command line is not one of this battery's own GPU stages is
written as a COLLISION row (its pid, its job hint, the lock holder at that moment). Nothing is ever
killed or signalled. Stops when `--stop-file` exists. Rows are JSON lines; the chain summarises them.
"""
import argparse
import csv
import io
import json
import os
import subprocess
import time

OURS = ("2026-09-28-refcv7-standard-tests\\battery", "2026-09-28-refcv7-standard-tests/battery",
        "2026-09-23-refcv6-standard-tests\\battery", "2026-09-23-refcv6-standard-tests/battery")
LOCK = os.environ.get("REFCV7_GPU_LOCK", "C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock")


def _q(args, timeout=60):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:                                        # noqa: BLE001 -- a watch never dies
        return ""


def snapshot():
    apps = []
    for row in csv.reader(io.StringIO(_q(["nvidia-smi", "--query-compute-apps=pid,process_name",
                                          "--format=csv,noheader"]))):
        if len(row) >= 2 and "python" in row[1].lower():
            try:
                apps.append(int(row[0].strip()))
            except ValueError:
                pass
    out = []
    for pid in apps:
        cmd = _q(["powershell.exe", "-NoProfile", "-Command",
                  f"(Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\").CommandLine"]).strip()
        out.append({"pid": pid, "ours": any(o in cmd for o in OURS),
                    "cmd_head": cmd[:200]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stop-file", required=True)
    ap.add_argument("--every", type=int, default=60)
    a = ap.parse_args()
    n = 0
    while not os.path.exists(a.stop_file):
        snap = snapshot()
        foreign = [x for x in snap if not x["ours"]]
        try:
            lock = open(LOCK, encoding="utf-8").read()[:300]
        except OSError:
            lock = None
        if foreign or n % 30 == 0:
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                    "COLLISION": bool(foreign), "foreign": foreign,
                                    "n_python_compute": len(snap), "lock": lock}) + "\n")
        n += 1
        time.sleep(a.every)


if __name__ == "__main__":
    main()
