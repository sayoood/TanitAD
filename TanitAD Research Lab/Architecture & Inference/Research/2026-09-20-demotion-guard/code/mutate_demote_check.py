"""Mutation proof for demote_check.py — and a DEMONSTRATION that the old predicate is blind.

⛔ A guard that cannot fail is worse than none, because it licenses the mistake it appears to
prevent. So this does not assert that `superset_check` misses demotion; it RUNS the real
`superset_check` on a real demotion and shows it returns OK.

Arms:
  CONTROL      an honest rewrite, figure still live        -> both checks PASS   (no false positive)
  M1 DEMOTE    the live figure moved into a retraction line-> superset PASSES, demote FAILS
  M2 DELETE    the figure removed entirely                 -> superset FAILS    (old guard still works)
  M3 WAIVED    M1 plus an allowdemote entry                -> demote PASSES     (disclosure works)
  M4 CODESPAN  a live `code span` demoted                  -> demote FAILS      (not numbers only)
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

Q = pathlib.Path("C:/Users/Admin/qland")
ENV = dict(os.environ)
ENV["GIT_DIR"] = "C:/Users/Admin/tanitad-push/.git"

TIP = """# Pad drop census

| half | delivered | dropped |
|---|---|---|
| halfA | 3.9894 | 13.91 % |
| halfB | 4.6235 | 30.36 % |

The instrument is `prebuild_p3_targets.census` and the pad is 32.
"""

CLEAN = """# Pad drop census

| half | delivered | dropped |
|---|---|---|
| halfA | 3.9894 | 13.91 % |
| halfB | 4.6235 | 30.36 % |

Re-banked 2026-09-20. The instrument is `prebuild_p3_targets.census` and the pad is 32.
"""

DEMOTED = """# Pad drop census

| half | delivered | dropped |
|---|---|---|
| halfA | 3.9894 | 0.05 % |
| halfB | 4.6235 | 30.36 % |

RETRACTED: the halfA figure was wrong; it previously read 13.91 % and is superseded.
The instrument is `prebuild_p3_targets.census` and the pad is 32.
"""

DELETED = """# Pad drop census

| half | delivered | dropped |
|---|---|---|
| halfA | 3.9894 | 0.05 % |
| halfB | 4.6235 | 30.36 % |

The instrument is `prebuild_p3_targets.census` and the pad is 32.
"""

CODE_DEMOTED = """# Pad drop census

| half | delivered | dropped |
|---|---|---|
| halfA | 3.9894 | 13.91 % |
| halfB | 4.6235 | 30.36 % |

SUPERSEDED: we no longer use `prebuild_p3_targets.census`. The pad is 32.
"""


def blob(text: str) -> str:
    p = pathlib.Path(tempfile.mkdtemp()) / "tip.md"
    p.write_bytes(text.encode("utf-8"))
    r = subprocess.run(["git", "hash-object", "-w", "--no-filters", "--", str(p)],
                       env=ENV, capture_output=True)
    sha = r.stdout.decode().strip()
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha), f"bad sha {sha!r}"
    return sha


def run(script: str, sha: str, text: str, allow: str | None):
    d = pathlib.Path(tempfile.mkdtemp())
    f = d / "new.md"
    f.write_bytes(text.encode("utf-8"))
    args = [sys.executable, str(Q / script), sha, str(f)]
    if allow is not None:
        a = d / "w.allowdemote"
        a.write_text(allow, encoding="utf-8")
        args.append(str(a))
    r = subprocess.run(args, capture_output=True, cwd=str(Q))
    return r.returncode, (r.stdout + r.stderr).decode("utf-8", "replace")


def main() -> int:
    sha = blob(TIP)
    rows, bad = [], []

    def arm(name, text, allow, want_sup, want_dem):
        sc, so = run("superset_check.py", sha, text, None)
        dc, do = run("demote_check.py", sha, text, allow)
        ok = (sc == want_sup) and (dc == want_dem)
        rows.append({"arm": name, "superset_rc": sc, "demote_rc": dc,
                     "want": [want_sup, want_dem], "ok": ok})
        print(f"  {'OK  ' if ok else 'FAIL'} {name:<12} superset rc={sc} (want {want_sup})   "
              f"demote rc={dc} (want {want_dem})")
        if not ok:
            bad.append((name, sc, dc, so.strip()[-160:], do.strip()[-160:]))
        return so, do

    print("mutation arms:")
    arm("CONTROL", CLEAN, None, 0, 0)
    _m1s, m1 = arm("M1_DEMOTE", DEMOTED, None, 0, 1)      # <- superset MUST pass: that is the point
    arm("M2_DELETE", DELETED, None, 1, 0)
    arm("M3_WAIVED", DEMOTED, "13.91  # ruled superseded, disclosed\n", 0, 0)
    arm("M4_CODESPAN", CODE_DEMOTED, None, 0, 1)

    print("\n⛔ THE DEMONSTRATION, not an assertion -- superset_check on the M1 demotion:")
    _sup_rc, sup_out = run("superset_check.py", sha, DEMOTED, None)
    assert _sup_rc == 0, "the demonstration requires superset_check to PASS the demotion"
    for ln in sup_out.strip().splitlines():
        print(f"    {ln}")
    print("  => the OLD predicate returns OK on a figure moved into a retraction line.")
    print("\n  demote_check on the same input:")
    for ln in m1.strip().splitlines():
        print(f"    {ln}")

    rec = {"_what": "mutation proof for demote_check + demonstration that superset_check is blind",
           "arms": rows, "all_caught": not bad}
    (Q / "mutation_proof_demote_check.json").write_text(json.dumps(rec, indent=1),
                                                        encoding="utf-8")
    if bad:
        print("\nZZMUTATION-FAIL", json.dumps(bad[:2], indent=1))
        return 1
    print("\nZZMUTATION-OK -- 5/5 arms as specified, control green, old predicate demonstrated blind")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
