"""Start condition for a detached milestone chain (2026-10-04, after `ZZCHAINNOLOCK50400ZZ`).

    python final_waiter.py check --disk D:/ --min-disk-gb 10 --min-commit-gb 6 --out <status.json>

Answers ONE question per call: may the chain START now? GO iff ALL hold:
  * the dev-box GPU lock file is ABSENT (`gpu_lock.read_lock() is None`) -- a held lock (the NavSim
    runner's, until ~12:00 Berlin on 2026-10-04) is never touched, broken or waited out by force;
  * free disk on the drive the G0 batch cache is written to is >= --min-disk-gb (brief: the disk
    batch cache writes ~4.5 GB; never start it below 10 GB free; D: is a `subst` of E:);
  * free COMMIT (Win32_OperatingSystem.FreeVirtualMemory -- the counter that decides whether an
    allocation succeeds) is >= --min-commit-gb.
⛔ FAIL CLOSED: any probe that cannot be read is a NO-GO with the error named, never a pass.
The verdict is the JSON `--out` (`go`, `why`, every probe value); assert on it, never on `$?`.
The chain itself still waits for commit and takes the lock through `gpu_lock.acquire` (which now
RETRIES a failed nvidia-smi probe instead of dying on it).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def decide(*, lock, free_disk_gb, free_commit_gb, min_disk_gb: float, min_commit_gb: float) -> dict:
    """The PURE decision (tested with literals)."""
    why = []
    if lock is not None:
        why.append(f"lock held/unreadable: {lock}")
    if not isinstance(free_disk_gb, (int, float)):
        why.append("free disk could not be read")
    elif free_disk_gb < min_disk_gb:
        why.append(f"free disk {free_disk_gb:.2f} GB < {min_disk_gb} GB")
    if not isinstance(free_commit_gb, (int, float)) or free_commit_gb < 0:
        why.append("free commit could not be read")
    elif free_commit_gb < min_commit_gb:
        why.append(f"free commit {free_commit_gb:.2f} GB < {min_commit_gb} GB")
    return {"go": not why, "why": why}


def probe(disk: str) -> dict:
    import gpu_lock
    from wait_commit import free_commit_gb
    out = {"t": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    try:
        out["lock"] = gpu_lock.read_lock()
    except Exception as exc:                            # noqa: BLE001 -- fail closed
        out["lock"] = {"unreadable": f"{type(exc).__name__}: {exc}"}
    try:
        out["free_disk_gb"] = round(shutil.disk_usage(disk).free / 2 ** 30, 2)
    except Exception as exc:                            # noqa: BLE001 -- fail closed
        out["free_disk_gb"], out["disk_error"] = None, f"{type(exc).__name__}: {exc}"
    try:
        fc, tc, fp = free_commit_gb()
        out.update(free_commit_gb=fc, commit_limit_gb=tc, free_phys_gb=fp)
    except Exception as exc:                            # noqa: BLE001 -- fail closed
        out["free_commit_gb"], out["commit_error"] = None, f"{type(exc).__name__}: {exc}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check",))
    ap.add_argument("--disk", default="D:/")
    ap.add_argument("--min-disk-gb", type=float, default=10.0)
    ap.add_argument("--min-commit-gb", type=float, default=6.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = probe(a.disk)
    d = decide(lock=p.get("lock"), free_disk_gb=p.get("free_disk_gb"),
               free_commit_gb=p.get("free_commit_gb"), min_disk_gb=a.min_disk_gb,
               min_commit_gb=a.min_commit_gb)
    rec = {**p, **d, "disk": a.disk, "min_disk_gb": a.min_disk_gb, "min_commit_gb": a.min_commit_gb}
    tmp = Path(a.out + ".part")
    tmp.write_text(json.dumps(rec, default=str), encoding="utf-8")
    tmp.replace(a.out)
    print(json.dumps(rec, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
