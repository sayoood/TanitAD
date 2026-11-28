"""labels_first.py -- PI ruling 2026-11-08 ("Labels go first"): the Qwen frame-label pass goes ahead of the other sessions on the dev box.

Takes the shared dev-box GPU lock the moment it is free (2 s poll, O_EXCL), waits for the previous job to leave the GPU, and holds the lock across
the prompt test and all of stage B, so no other job can slip in between. It never forces, overwrites or removes a lock it does not own.

Sequence: take lock -> GPU idle -> audit_prompts.py (skipped if its result exists) -> marker LF_PROMPTS_DONE -> wait for marker GO_B (the Master Mind
writes the adopted questions into extract_context.VLM_Q, then creates it) -> chain_full.py --stage B -> release.
The children see STRAT_LOCK_HELD=1 (gpu_lock.acquire and release become no-ops) and STRAT_MIN_AVAIL_GB.

usage: labels_first.py [--min-avail-gb 4.5] [--budget 300] [--lock PATH --dry --max-wait-s N]   (--dry: take and release the lock only; for the tests)
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gpu_lock as L  # noqa: E402

OUT = "F:/strat_out"
STOP = OUT + "/STOP"
DONE_MARK, GO_MARK = HERE + "/LF_PROMPTS_DONE", HERE + "/GO_B"
RESULT = HERE + "/AUDIT_PROMPTS_RESULT.json"


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def take(lock, max_wait_s):
    """O_EXCL create as soon as the file is absent. Returns the body written, or None on timeout or STOP. Touches nothing while another job holds it."""
    t0 = time.time()
    body = json.dumps({"pid": os.getpid(), "owner": L.OWNER, "job": "labels_first", "t": time.time()})
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if os.path.exists(STOP) or time.time() - t0 > max_wait_s:
                return None
            time.sleep(2)
            continue
        os.write(fd, body.encode())
        os.close(fd)
        time.sleep(4)
        if open(lock, encoding="utf-8").read() != body:   # defensive: somebody replaced it
            log("lock content is not ours after taking it; leaving it alone")
            return None
        log(f"lock taken after {time.time() - t0:.0f} s")
        return body


def give_back(lock, body):
    try:
        if open(lock, encoding="utf-8").read() == body:
            os.remove(lock)
            log("lock released")
    except OSError as e:
        log(f"release: {e!r}")


def prompt_test(env):
    for attempt in range(1, 4):
        if os.path.exists(RESULT):
            return True
        r = subprocess.run([L.PY, HERE + "/audit_prompts.py"], stdin=subprocess.DEVNULL, env=env)
        log(f"audit_prompts attempt {attempt} rc={r.returncode}")
        if r.returncode == 0 and os.path.exists(RESULT):
            return True
        time.sleep(120)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-avail-gb", type=float, default=4.5)
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--lock", default=L.LOCK)
    ap.add_argument("--max-wait-s", type=float, default=24 * 3600)
    ap.add_argument("--go-wait-s", type=float, default=2 * 3600)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    body = take(a.lock, a.max_wait_s)
    if body is None:
        log("lock not taken; exiting without touching it")
        return 3
    try:
        if a.dry:
            return 0
        env = dict(os.environ, STRAT_LOCK_HELD="1", STRAT_MIN_AVAIL_GB=str(a.min_avail_gb), PYTHONPATH="F:/strat_models/pylibs", HF_HUB_OFFLINE="1")
        log("waiting for the previous job to leave the GPU")
        r = subprocess.run([L.PY, L.WAIT_IDLE], stdin=subprocess.DEVNULL, timeout=4 * 3600)
        if r.returncode != 0:
            log(f"wait_idle rc={r.returncode}")
            return 4
        if not prompt_test(env):
            open(OUT + "/ZZ_LF_FAILED", "w").write("audit_prompts failed three times\n")
            return 5
        open(DONE_MARK, "w").write(time.strftime("%H:%M:%S") + "\n")
        log("prompt test done; waiting for GO_B")
        t0 = time.time()
        while not os.path.exists(GO_MARK):
            if os.path.exists(STOP) or time.time() - t0 > a.go_wait_s:
                open(OUT + "/ZZ_LF_FAILED", "w").write("GO_B not given in time\n")
                return 6
            time.sleep(5)
        r = subprocess.run([L.PY, HERE + "/chain_full.py", "--stage", "B", "--clips", "splits/qwen_clips.json", "--budget", str(a.budget)],
                           stdin=subprocess.DEVNULL, env=env)
        log(f"chain_full stage B rc={r.returncode}")
        return r.returncode
    finally:
        give_back(a.lock, body)


if __name__ == "__main__":
    sys.exit(main())
