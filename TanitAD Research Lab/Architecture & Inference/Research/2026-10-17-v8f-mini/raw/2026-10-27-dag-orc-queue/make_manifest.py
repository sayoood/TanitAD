#!/usr/bin/env python3
"""Write MANIFEST.txt: the sha256 and size of every file of this package (except __pycache__ and MANIFEST.txt itself), sorted by path. `python make_manifest.py --verify` re-hashes the files against MANIFEST.txt (rc 0 = equal)."""
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "MANIFEST.txt"


def files():
    return sorted(p for p in HERE.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name != "MANIFEST.txt")


def rows():
    return [(hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_size, p.relative_to(HERE).as_posix()) for p in files()]


if __name__ == "__main__":
    if "--verify" in sys.argv:
        have = {ln.split("  ", 2)[2]: ln.split("  ", 2)[0] for ln in OUT.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")}
        now = {r[2]: r[0] for r in rows()}
        diff = sorted(set(have) ^ set(now)) + sorted(k for k in set(have) & set(now) if have[k] != now[k])
        print("MANIFEST", "OK" if not diff else f"DIFFERS: {diff[:10]}", f"({len(now)} files)")
        sys.exit(0 if not diff else 1)
    r = rows()
    OUT.write_text("# sha256  bytes  path (relative to dag_orc_chain/)\n" + "\n".join(f"{h}  {n}  {p}" for h, n, p in r) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(r)} files)")
