"""Stamp ``paused`` into the A12 early record (the Master Mind's SIGSTOP 11:31:59 Berlin for the
box head; resumed as recorded by watch_resume.sh). The record's s/step of the arm that was
running (s8_detached) includes the pause -- informative only, and now says so."""
import datetime as dt
import hashlib
import json
from pathlib import Path

D = Path("/home/nvidia/nb2r2_cef9/gmo_a12")
rec_p = D / "g_map_overfit_A12.EARLY_NONBINDING.json"
rec = json.loads(rec_p.read_text(encoding="utf-8"))
pause_txt = (D / "PAUSED_BY_MASTER_MIND.txt").read_text(encoding="utf-8")
res_txt = (D / "RESUMED_AT.txt").read_text(encoding="utf-8").strip()
# the pause log is Berlin local (CEST, UTC+2); the watcher writes UTC
hhmmss = pause_txt.split("paused at ")[1].split()[0]
paused_utc = dt.datetime.fromisoformat(f"2026-09-27T{hhmmss}+02:00").astimezone(dt.timezone.utc)
resumed_utc = dt.datetime.fromisoformat(res_txt.split()[1].replace("Z", "+00:00"))
paused_s = (resumed_utc - paused_utc).total_seconds()
rec["paused"] = {"by": "the Master Mind (SIGSTOP, the PI's request: box head first)",
                 "pid": 3653915, "paused_utc": paused_utc.isoformat(),
                 "resumed_utc": resumed_utc.isoformat(), "paused_s": round(paused_s, 1),
                 "arm_running_when_paused": "s8_detached (informative)",
                 "note": "s8_detached's s_per_step and the run's wall_s include the pause"}
rec_p.write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
print(json.dumps(rec["paused"]), hashlib.md5(rec_p.read_bytes()).hexdigest())
