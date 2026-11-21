"""PRIVGEOM-CLEAN code pins: print the sha256 table of every pinned file (markdown), write PINS.json, or `--check` against PINS.json.  The gate token binds these hashes; fit / analyze refuse on a mismatch.
    python pgc_pins.py            # print + write PINS.json
    python pgc_pins.py --check    # exit 1 if any pinned file differs from PINS.json
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
CODE = ("pg_geom.py", "pg_extract.py", "pg_lib.py", "test_pg_geom.py", "test_pg_lib.py", "pgc_lib.py", "pgc_admit.py", "pgc_run.py", "pgc_coverage.py", "pgc_design.py", "pgc_calibrate_plant.py",
        "pgc_glive.py", "pgc_admit_seen.py", "pgc_cost.py", "pgc_rehearsal_identity.py", "build_package.py", "pgc_pins.py", "test_pgc_lib.py", "test_pgc_admit.py", "finish_clean.sh", "launch.ps1", "rehearsal_chain.sh",
        "pgc_seal.py", "test_pgc_seal.py", "pgc_read_chain.py", "pgc_seal_proofs.py")
REF = ("fastneg_lib.py",)
FROZEN = {"pg_geom.py": "5b78a694f40b6b2b4030fdeb3bf5be2c65b349c8887f9c11c78d0dfd8d9df7de", "pg_extract.py": "64a148906329f8aeba5dc3d87d30aba9f2e91d90389ea77c0169614e1bade23b",
          "pg_lib.py": "61bfe59692987de6e43cf9d2585c2992192a4b9dca6d513688bffc5c77dd7178", "test_pg_geom.py": "3cf48766adc17b4c1ac07bb1b043e3798ffa2a680226196f124fc69f56406b64",
          "test_pg_lib.py": "fd1d408aed2d870f86c9a02a556b32ce0f54708042c25823666c425f06b89cfa", "fastneg_lib.py": "0c0b399b37a47ee3e17f9d5484bc7f9521ff7dbe6bea37e8c7a38042668667ca"}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def table() -> dict:
    out = {n: sha(f"{HERE}/{n}") for n in CODE if os.path.exists(f"{HERE}/{n}")}
    out.update({f"ref/{n}": sha(f"{ROOT}/ref/{n}") for n in REF})
    return out


def main() -> int:
    t = table()
    bad_frozen = [n for n, h in FROZEN.items() if t.get(n, t.get("ref/" + n)) != h]
    if "--check" in sys.argv:
        pins = json.load(open(f"{HERE}/PINS.json", encoding="utf-8"))["sha256"]
        diff = [n for n in sorted(set(pins) | set(t)) if pins.get(n) != t.get(n)]
        print("PINS CHECK:", "OK" if not diff and not bad_frozen else f"MISMATCH {diff} frozen-bad {bad_frozen}")
        return 0 if not diff and not bad_frozen else 1
    json.dump({"note": "FROZEN = PRIVGEOM-SAFE files unchanged (their pinned sha256); everything else is new for PRIVGEOM-CLEAN", "frozen_ok": not bad_frozen, "frozen_expected": FROZEN, "sha256": t}, open(f"{HERE}/PINS.json", "w", encoding="utf-8", newline="\n"), indent=1)
    print("| file | frozen? | sha256 |\n|---|---|---|")
    for n, h in t.items():
        print(f"| `{n}` | {'FROZEN (PRIVGEOM-SAFE)' if n in FROZEN or n.replace('ref/', '') in FROZEN else 'new'} | `{h}` |")
    print("frozen files unchanged:", not bad_frozen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
