#!/usr/bin/env python3
"""Tests for refe/p7_ship_list.py (PKG-SHIP-COMPLETE): the file set the refe_p7 G-SUITE reads, derived from the suite itself, checked against a package directory.

Run: python refe/test_p7_ship_list.py   -> ZZP7SHIP_TEST_PASS <n>.   Never prints ZZTEST_SKIP.
  1. a synthetic package (a launch_gate.py with a one-script suite, the script reading `eval/arms.json` and `raw/x/PREREG.md` through `PKG / "eval" / ...` chains): `derive` finds both (static),
     the traced run finds an `open()` the static scan cannot see, `check` is OK on the complete package, and REFUSES (rc 1, ZZP7SHIP_BAD) on each deliberately missing / changed file;
  2. a STALE list (a suite script changed since the list was derived) REFUSES; a CRLF-only difference is accepted; a missing shipped root / SHIP_MANIFEST.json REFUSES;
  3. files made AFTER the derive (the list inside the package, its trace sidecar, SHIP_MANIFEST.json regenerated at ship time) are presence-only: check is a fixed point;
  4. the INCIDENT: on the real package (the tree this file sits in), the four files P6 lost are in the derived list, and removing any one of them from a copy of the package makes `check` REFUSE.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE / "p7_ship_list.py"
NPASS, FAILS = 0, []


def ok(c, name):
    global NPASS
    if c:
        NPASS += 1
    else:
        FAILS.append(name)
        print("  FAIL:", name, flush=True)


def run(*a):
    r = subprocess.run([sys.executable, str(TOOL), *map(str, a)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, r.stdout + r.stderr


def build_pkg(root: Path):
    (root / "refe").mkdir(parents=True)
    for d in ("code", "splits", "eval", "raw/x"):
        (root / d).mkdir(parents=True)
    (root / "code" / "c.sh").write_text("echo c\n"); (root / "splits" / "s.yaml").write_text("a: 1\n")
    (root / "eval" / "arms.json").write_text('{"a": 1}\n'); (root / "raw" / "x" / "PREREG.md").write_text("frozen\n"); (root / "raw" / "x" / "dyn.txt").write_text("dynamic\n")
    (root / "SHIP_MANIFEST.json").write_text("{}")
    (root / "refe" / "helper.py").write_text("X = 1\n")
    (root / "refe" / "launch_gate.py").write_text('PROFILES = {"refe_p7": {"suite": [{"script": "test_a.py", "token": "ZZA_PASS"}]}}\n')
    (root / "refe" / "test_a.py").write_text(
        'import json, sys\nfrom pathlib import Path\nimport helper\nHERE = Path(__file__).resolve().parent\nPKG = HERE.parent\n'
        'json.loads((PKG / "eval" / "arms.json").read_text())\n(PKG / "raw" / "x" / "PREREG.md").read_bytes()\n'
        'name = "dyn" + ".txt"\nopen(str(PKG / "raw" / "x" / name)).read()\nprint("ZZA_PASS 1")\n')


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="p7ship_"))
    try:
        pkg = tmp / "pkg"; build_pkg(pkg)
        lst = tmp / "list.json"
        rc, out = run("derive", "--pkg", pkg, "--out", lst)
        d = json.loads(lst.read_text())
        ok(rc == 0 and "eval/arms.json" in d["files"] and "raw/x/PREREG.md" in d["files"], f"static derive finds the two Div-chain reads ({out.strip()[:100]})")
        ok("raw/x/dyn.txt" not in d["files"], "control: the static scan does NOT see a name built at runtime (that is what the traced run is for)")
        ok(d["files"]["eval/arms.json"]["outside_shipped_roots"] is True and "refe/helper.py" in d["scripts"], "outside-roots flag set; the imported local module is in the closure")
        rc, out = run("derive", "--pkg", pkg, "--out", lst, "--trace")
        d = json.loads(lst.read_text())
        ok(rc == 0 and "raw/x/dyn.txt" in d["files"] and "trace" in d["files"]["raw/x/dyn.txt"]["via"], f"the TRACED run finds the open() the static scan cannot see ({out.strip()[-120:]})")
        ok("static" in d["files"]["eval/arms.json"]["via"] and "trace" in d["files"]["eval/arms.json"]["via"], "a read seen by both is marked both")
        rc, out = run("check", "--pkg", pkg, "--list", lst)
        ok(rc == 0 and "ZZP7SHIP_OK" in out, f"check: the complete package is OK ({out.strip()[:80]})")
        # ---- each deliberately missing file REFUSES
        for rel in ("eval/arms.json", "raw/x/PREREG.md", "raw/x/dyn.txt", "SHIP_MANIFEST.json", "refe/helper.py", "refe/test_a.py"):
            cp = tmp / "cp"; shutil.rmtree(cp, ignore_errors=True); shutil.copytree(pkg, cp)
            (cp / rel).unlink()
            rc, out = run("check", "--pkg", cp, "--list", lst)
            ok(rc == 1 and "ZZP7SHIP_BAD" in out and "missing=1" in out, f"check REFUSES with {rel} missing ({out.strip()[:90]})")
        cp = tmp / "cp"; shutil.rmtree(cp, ignore_errors=True); shutil.copytree(pkg, cp); shutil.rmtree(cp / "splits")
        rc, out = run("check", "--pkg", cp, "--list", lst)
        ok(rc == 1 and "splits/" in out, "check REFUSES with a shipped root missing")
        # ---- changed bytes, CRLF-only, stale script
        shutil.rmtree(cp); shutil.copytree(pkg, cp); (cp / "eval" / "arms.json").write_text('{"a": 2}\n')
        rc, out = run("check", "--pkg", cp, "--list", lst)
        ok(rc == 1 and "differs=1" in out, "check REFUSES a listed file whose bytes differ")
        shutil.rmtree(cp); shutil.copytree(pkg, cp); (cp / "raw" / "x" / "PREREG.md").write_bytes(b"frozen\r\n")
        rc, out = run("check", "--pkg", cp, "--list", lst)
        ok(rc == 0, "a CRLF-only difference is accepted (an LF blob checked out with autocrlf)")
        shutil.rmtree(cp); shutil.copytree(pkg, cp); (cp / "refe" / "test_a.py").write_text((cp / "refe" / "test_a.py").read_text() + "# landed later\n")
        rc, out = run("check", "--pkg", cp, "--list", lst)
        ok(rc == 1 and "stale_scripts=1" in out, "check REFUSES a STALE list (a suite script changed since the list was derived)")
        ok(run("check", "--pkg", pkg, "--list", tmp / "nope.json")[0] == 1 and run("derive", "--pkg", pkg, "--out", tmp / "l2.json", "--trace-file", tmp / "nope.json")[0] != 0, "an unreadable list / trace file REFUSES")
        # a stale trace is refused
        tf = tmp / "t.json"; tf.write_text(json.dumps({"closure_digest": "0" * 64, "opened": {}}))
        rc, out = run("derive", "--pkg", pkg, "--out", tmp / "l3.json", "--trace-file", tf)
        ok(rc == 1 and "ZZP7SHIP_BAD" in out, "--trace-file taken at another closure is refused")

        # ---- files made AFTER the derive (2026-11-06): the list INSIDE the package and a suite that reads SHIP_MANIFEST.json and globs refe/ (as
        # test_launch_gate.py does).  Pinning them made `check` REFUSE every real package: the list pinned its own previous bytes and the dev box's manifest.
        pk2 = tmp / "pkg2"; build_pkg(pk2)
        (pk2 / "refe" / "test_a.py").write_text((pk2 / "refe" / "test_a.py").read_text().replace('print("ZZA_PASS 1")',
            '(PKG / "SHIP_MANIFEST.json").read_text()\nfor f in sorted((PKG / "refe").iterdir()):\n    if f.is_file(): f.read_bytes()\nprint("ZZA_PASS 1")'))
        l2 = pk2 / "refe" / "p7_ship_list.json"
        run("derive", "--pkg", pk2, "--out", l2, "--trace")                     # the first derive writes the list + its trace sidecar into refe/
        rc, out = run("derive", "--pkg", pk2, "--out", l2, "--trace")           # the re-derive TRACES both (the tree reader globs refe/)
        d2 = json.loads(l2.read_text())
        ok(rc == 0 and "SHIP_MANIFEST.json" in str(json.loads((pk2 / "refe" / "p7_ship_list.json.trace.json").read_text())["opened"]),
           "control: the traced suite really opened SHIP_MANIFEST.json (else the next check proves nothing)")
        ok(not ({"refe/p7_ship_list.json", "refe/p7_ship_list.json.trace.json", "SHIP_MANIFEST.json"} & set(d2["files"])),
           f"files made after the derive (the list, its trace sidecar, SHIP_MANIFEST.json) are not pinned ({sorted(set(d2['files']) & {'refe/p7_ship_list.json', 'refe/p7_ship_list.json.trace.json', 'SHIP_MANIFEST.json'})})")
        rc, out = run("check", "--pkg", pk2, "--list", l2)
        ok(rc == 0 and "ZZP7SHIP_OK" in out, f"check is OK on the package the list was derived in, the list inside it (a fixed point) ({out.strip()[:100]})")
        (pk2 / "SHIP_MANIFEST.json").write_text('{"commit": "the landed one"}')
        rc, out = run("check", "--pkg", pk2, "--list", l2)
        ok(rc == 0, f"check is OK after SHIP_MANIFEST.json is regenerated at ship time ({out.strip()[:100]})")
        (pk2 / "SHIP_MANIFEST.json").unlink()
        rc, out = run("check", "--pkg", pk2, "--list", l2)
        ok(rc == 1 and "SHIP_MANIFEST.json" in out, "control: SHIP_MANIFEST.json is still REQUIRED (presence)")

        # ---- the INCIDENT, on the real package
        real = HERE.parent
        if (real / "eval" / "rls1b_arms.json").is_file():
            rl = tmp / "real.json"
            rc, out = run("derive", "--pkg", real, "--out", rl)
            rd = json.loads(rl.read_text())
            four = ["eval/rls1b_arms.json", "eval/rls1d_arms.json", "eval/rls1e_arms.json", "raw/2026-10-18-r0e-pick/PREREG_R0E_PICK.md"]
            ok(rc == 0 and all(f in rd["files"] for f in four), f"the four files P6 lost are in the derived list ({[f for f in four if f not in rd['files']]})")
            ok(rd["profile"] == "refe_p7" and "test_p7_integration.py" in " ".join(rd["suite"]), "the derived suite is refe_p7's")
        else:
            ok(False, "real package not found next to refe/ (run this test inside the full package tree)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if FAILS:
        print(f"ZZP7SHIP_TEST_FAIL {len(FAILS)} failing, {NPASS} passing")
        return 1
    print(f"ZZP7SHIP_TEST_PASS {NPASS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
