"""Chain-wide RAM watchdog for the GT-validation jobs (the coordinator's CPU-job rule, 2026-09-26):
if box-free RAM drops below 4.0 GB, kill every job of this chain -- selected by interpreter + script
name (+ the package's unique --out-dir for the renderer), explicit PIDs only -- and write ABORTED.
Rescans every 2 s (the chain starts jobs after the watchdog), and exits when STOP appears."""
import sys
import time
from pathlib import Path

import psutil

FLOOR_GB = 4.0
WORK = Path("C:/Users/Admin/qland/work/gtval")
LOG = WORK / "ram_watchdog.log"
STOP, ABORTED = WORK / "WATCHDOG_STOP", WORK / "ABORTED"


def log(msg):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(time.strftime("%H:%M:%S ") + msg + "\n")


def mine():
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            c = p.info["cmdline"] or []
            if not (p.info["name"] or "").lower().startswith("python"):
                continue
            if any("extract_eval_joins.py" in x for x in c) or (
                    any("render_refcv6_gt_validation.py" in x for x in c)
                    and any("2026-09-26-refcv6-gt-validation" in x for x in c)):
                out.append(p)
        except Exception:
            pass
    return out


log(f"armed; floor {FLOOR_GB} GB; avail {psutil.virtual_memory().available / 2 ** 30:.2f} GB")
min_av, seen = 1e9, set()
while not STOP.exists():
    av = psutil.virtual_memory().available / 2 ** 30
    min_av = min(min_av, av)
    procs = mine()
    new = {p.pid for p in procs} - seen
    if new:
        log(f"watching new pids {sorted(new)}")
        seen |= new
    if av < FLOOR_GB and procs:
        for p in procs:
            try:
                p.kill()
                log(f"ABORT: available {av:.2f} GB < {FLOOR_GB} GB -- killed pid {p.pid}")
            except Exception as e:
                log(f"ABORT: kill pid {p.pid} failed: {e!r}")
        ABORTED.write_text(f"{time.strftime('%H:%M:%S')} avail {av:.2f} GB", encoding="utf-8")
    time.sleep(2)
log(f"stopped; min available seen {min_av:.2f} GB")
