"""Run ONE command under the gate chain's RAM floor (the same floor as run_suite_bounded.py):
start after 3 consecutive samples 30 s apart read >= 7.5 GB available, kill it below 6.5 GB.

    python gated_cmd.py <log> -- <cmd> [args...]
"""
import subprocess
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0].rsplit("\\", 1)[0])
from run_suite_bounded import (FLOOR_NOTE, KILL_GB, MAX_WAIT_S, START_GAP_S,  # noqa: E402
                               START_GB, START_SAMPLES, free_gb)


def main() -> int:
    log = sys.argv[1]
    cmd = sys.argv[sys.argv.index("--") + 1:]
    w0, ok = time.time(), 0
    while ok < START_SAMPLES and time.time() - w0 < MAX_WAIT_S:
        ok = ok + 1 if free_gb() >= START_GB else 0
        if ok < START_SAMPLES:
            time.sleep(START_GAP_S)
    if ok < START_SAMPLES:
        # ⛔ the wait budget ran out: do NOT launch (the first version started anyway)
        with open(log, "w", encoding="utf-8") as fh:
            fh.write(f"# {FLOOR_NOTE}\n# cmd: {cmd}\n# NOT RUN: RAM_WAIT_TIMEOUT "
                     f"({MAX_WAIT_S // 3600} h, last {free_gb():.2f} GiB)\n"
                     f"# exit None RAM_WAIT_TIMEOUT\n")
        print(f"gated_cmd NOT RUN status=RAM_WAIT_TIMEOUT log={log}", flush=True)
        return 8
    with open(log, "w", encoding="utf-8") as fh:
        fh.write(f"# {FLOOR_NOTE}\n# cmd: {cmd}\n")
        fh.flush()
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        status = None
        while p.poll() is None:
            time.sleep(2.0)
            if free_gb() < KILL_GB:
                p.kill()
                p.wait()
                status = "RAM_ABORT"
        fh.write(f"\n# exit {p.returncode} {status or ''}\n")
    print(f"gated_cmd exit={p.returncode} status={status or 'OK'} log={log}", flush=True)
    return 0 if status is None else 9


if __name__ == "__main__":
    sys.exit(main())
