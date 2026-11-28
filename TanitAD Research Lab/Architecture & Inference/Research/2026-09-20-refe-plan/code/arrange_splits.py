"""Put the extracted nuPlan DBs where THEIR runner actually reads them.

MEASURED 2026-09-20 by parsing the local file headers of the partially downloaded archive (no need to
wait for 100%): the nuPlan v1.1 zips extract to

    data/cache/<split>/*.db          e.g. data/cache/test/, data/cache/mini/

but run_driverl_nuplan_eval.sh:192-194 reads

    ${NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/trainval     (val14_nr, val14_r)
    ${NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/test         (test14hard_*, test14random_*)

So extraction alone leaves the runner with "No log files found!" -- the same symptom the exFAT
junction produced in Stage 0, from a different cause.

D: IS exFAT: os.symlink and os.link both fail (WinError 1), and an NTFS junction from C: into D:
lists EMPTY. The only workable move is a same-volume RENAME, which is instant regardless of size.

Usage: python arrange_splits.py [--apply]      (default is a dry run)
"""
from __future__ import annotations
import json, os, shutil, sys

DATA = os.environ.get("DZ11_DATA_ROOT", "D:/Projects/TanitAD/data/nuplan")
CACHE = os.path.join(DATA, "data", "cache")
# source split dir -> the directory name their runner expects under nuplan-v1.1/splits/
TARGET = {"test": "test", "val": "trainval", "trainval": "trainval"}
SKIP = {"mini"}          # Stage 0 already moved mini to dblinks/driverl_val14


def gb(path: str) -> float:
    t = 0
    for r, _, fs in os.walk(path):
        for f in fs:
            try:
                t += os.path.getsize(os.path.join(r, f))
            except OSError:
                pass
    return t / 2**30


def main(apply: bool) -> None:
    if not os.path.isdir(CACHE):
        print(f"nothing to arrange: {CACHE} does not exist"); return
    out = []
    for name in sorted(os.listdir(CACHE)):
        src = os.path.join(CACHE, name)
        if not os.path.isdir(src):
            continue
        if name in SKIP:
            print(f"  SKIP  {name} (Stage 0 owns it)"); continue
        if name not in TARGET:
            print(f"  SKIP  {name}: no mapping -- add it to TARGET deliberately, do not guess"); continue
        dst = os.path.join(DATA, "nuplan-v1.1", "splits", TARGET[name])
        n = len([f for f in os.listdir(src) if f.endswith(".db")])
        size = gb(src)
        if os.path.isdir(dst) and os.listdir(dst):
            print(f"  REFUSE {name} -> {dst}: target exists and is NOT empty "
                  f"({len(os.listdir(dst))} entries). Resolve by hand.")
            out.append({"split": name, "action": "refused_nonempty_target", "dst": dst})
            continue
        print(f"  {'MOVE ' if apply else 'WOULD'} {src}  ->  {dst}   ({n} DBs, {size:.1f} GB)")
        if apply:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.isdir(dst):
                os.rmdir(dst)
            os.rename(src, dst)          # same volume => instant, and exFAT-safe (no links involved)
            n2 = len([f for f in os.listdir(dst) if f.endswith(".db")])
            assert n2 == n, f"post-move count {n2} != {n}"
            print(f"         verified {n2} DBs at the target")
        out.append({"split": name, "src": src, "dst": dst, "n_db": n, "gb": round(size, 1),
                    "applied": bool(apply)})
    if apply:
        try:
            os.rmdir(CACHE); os.rmdir(os.path.dirname(CACHE))
            print(f"  removed the now-empty {CACHE}")
        except OSError:
            pass
        json.dump(out, open(os.path.join(DATA, "arrange_splits.json"), "w"), indent=1)
    print("\nARRANGE_DONE" if apply else "\nDRY RUN -- rerun with --apply")


if __name__ == "__main__":
    main("--apply" in sys.argv)
