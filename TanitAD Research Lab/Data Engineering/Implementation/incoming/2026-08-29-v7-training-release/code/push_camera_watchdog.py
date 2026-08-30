"""Drive push_camera_batched.py to completion through a stalling network path.

MEASURED twice on this repo: a long-lived HF upload goes to 0 B/s with no client
timeout and no error — the 1.97 GB tar did it at ~55 % and ~81 %, and the 61.6 GB
camera bank did it after staging ~1.2 GB (0.0 MB of process IO over 25 s).

Sampling process IO is what distinguishes "slow" from "hung"; the remote listing
cannot, because a batch commits only when it finishes.

⛔ Kills by EXPLICIT PID. Never `pkill -f push_camera` — that matches this
watchdog's own command line and kills the wrong thing (the documented self-match
trap).
"""
import subprocess
import sys
import time
from pathlib import Path

PY = r"C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
SCRIPT = r"C:\Users\Admin\tanitad-wt\_s2build\release\push_camera_batched.py"
LOG = Path(r"C:\Users\Admin\tanitad-wt\_s2build\release\camera_push_rounds.log")
MAX_ROUNDS = 40
SAMPLE_S = 30
STALL_MB = 2.0
STALL_LIMIT = 3


def io_of(pid: int) -> float:
    """Total transfer bytes for the process tree root, in MB. -1 if unreadable."""
    r = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"$p=Get-Process -Id {pid} -EA SilentlyContinue; if($p){{"
         f"($p.ReadTransferCount+$p.WriteTransferCount+$p.OtherTransferCount)}}"],
        capture_output=True, text=True)
    try:
        return int(r.stdout.strip().splitlines()[-1]) / 1e6
    except Exception:                                       # noqa: BLE001
        return -1.0


def say(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    sys.stdout.flush()
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


for rnd in range(1, MAX_ROUNDS + 1):
    say(f"ROUND {rnd}: launching batched push")
    outp = LOG.parent / f"camera_push_round{rnd}.out"
    with open(outp, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            [PY, SCRIPT], stdout=fh, stderr=subprocess.STDOUT,
            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8",
                 "PYTHONUNBUFFERED": "1"})
        stall, last = 0, io_of(proc.pid)
        while proc.poll() is None:
            time.sleep(SAMPLE_S)
            cur = io_of(proc.pid)
            moved = (cur - last) if (cur >= 0 and last >= 0) else 1e9
            last = cur
            if moved < STALL_MB:
                stall += 1
                say(f"  stall {stall}/{STALL_LIMIT} ({moved:.1f} MB in {SAMPLE_S}s)")
            else:
                stall = 0
            if stall >= STALL_LIMIT:
                say("  STALLED — killing by explicit PID and relaunching "
                    "(finished batches are already committed and will be skipped)")
                proc.kill()
                proc.wait()
                break
    txt = outp.read_text(encoding="utf-8", errors="replace")
    if "CAMERA PUSH VERIFIED" in txt:
        say("SUCCESS — every file present on the remote at the right size")
        for ln in txt.splitlines()[-3:]:
            say("  " + ln.strip()[:120])
        sys.exit(0)
    if "QUOTA/BILLING SIGNAL" in txt:
        say("⛔ QUOTA SIGNAL — stopping. Spend is the PI's decision, not retried.")
        sys.exit(3)
    if "Keys.txt unreadable" in txt:
        say("round lost to a G: outage (token), not the upload — waiting 120 s")
        time.sleep(120)
    for ln in txt.splitlines()[-2:]:
        if ln.strip():
            say("  last: " + ln.strip()[:120])
say("GAVE UP after MAX_ROUNDS — escalate")
sys.exit(2)
