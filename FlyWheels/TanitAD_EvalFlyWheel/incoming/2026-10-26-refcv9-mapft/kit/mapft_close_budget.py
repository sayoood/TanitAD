#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mapft_close_budget.py -- the MAPFT-BUDGET gate closure, as a REVIEWABLE PATCH TOOL (it edits a checkout you name).

WHY THIS EXISTS (MEASURED 2026-10-26 by reading launch_gate.py at 70211d6):
  PROFILES["refcv8-mapft"] carries open item MAPFT-BUDGET.  `finalize()` turns ANY open item into verdict INCOMPLETE
  ("a profile with OPEN ITEMS is not a finished launch definition: never a PASS", launch_gate.py:~7724), and
  `verify_token` refuses a token whose verdict is not PASS (:~7870).  So at 70211d6 NO argv can ever be launched under this
  profile: the item has to be closed by an edit to launch_gate.py -- which the item's own `closes_when` says.
  That edit is the Master Mind's / PI's to make (a launch_gate.py edit is flagged "Security Weaken" by the safety
  classifier and needs the PI's explicit go -- memory nt-gate-bind-reuse-and-kit-defects).  This tool makes it small,
  exact, mechanical and TESTED so the MM can review a diff instead of writing one at 19:00.

TWO MODES (the MM picks; --mode is required):
  open-only   `open_items=()` only.  The budget flags stay UN-pinned by the profile (as they are today); the token still
              binds the FULL argv by content, so a different N is refused at launch.  N-INDEPENDENT => the launch commit
              C0 can be made BEFORE the smoke, the dev-box G-SUITE-PINNED + the launch tree can be staged early, and the
              argv file lives OUTSIDE the tree.  RECOMMENDED.
  literals    `open_items=()` AND --steps/--warmup/--ema-decay/--eval-every written into required_values from
              mapft_argv_full.json (the item's literal `closes_when`).  N-DEPENDENT => the commit can only follow the
              smoke, and the Thor suite + dev-box pinned suite + tree ship all wait for N.  Strictest.

  python mapft_close_budget.py --repo <checkout root> --mode open-only|literals [--argv mapft_argv_full.json] \
         [--dry-run] [--note "free text recorded in the comment"]

It edits ONLY  stack/scripts/launch_gate.py  and  stack/tests/test_map_only_ft.py  (the test that pins the OPEN state and
whose fixture uses placeholder budget values).  Every replacement is anchored on an exact string that must occur EXACTLY
ONCE or the tool refuses and writes nothing.  Newline style (LF / CRLF) is preserved.  It never runs git.
Prints a unified diff; markers ZZMAPFT-CLOSE-{OK,REFUSED}ZZ.
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

GATE_REL = "stack/scripts/launch_gate.py"
TEST_REL = "stack/tests/test_map_only_ft.py"
BUDGET_FLAGS = ("--steps", "--warmup", "--ema-decay", "--eval-every")


def die(msg):
    print("REFUSED:", msg)
    print("ZZMAPFT-CLOSE-REFUSEDZZ")
    sys.exit(2)


def read_text(p: Path):
    raw = p.read_bytes()
    nl = "\r\n" if b"\r\n" in raw else "\n"
    return raw.decode("utf-8").replace("\r\n", "\n"), nl


def once(text: str, anchor: str, what: str) -> int:
    n = text.count(anchor)
    if n != 1:
        die(f"{what}: anchor occurs {n} times (need exactly 1): {anchor[:90]!r}")
    return text.index(anchor)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--mode", required=True, choices=("open-only", "literals"))
    ap.add_argument("--argv", default=None, help="mapft_argv_full.json (required for --mode literals)")
    ap.add_argument("--note", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    repo = Path(a.repo)
    gp, tp = repo / GATE_REL, repo / TEST_REL
    for p in (gp, tp):
        if not p.is_file():
            die(f"{p} not found")
    g, gnl = read_text(gp)
    t, tnl = read_text(tp)
    lit = {}
    prov = ""
    if a.mode == "literals":
        if not a.argv:
            die("--mode literals needs --argv mapft_argv_full.json")
        d = json.loads(Path(a.argv).read_text(encoding="utf-8"))
        av = d["argv"] if isinstance(d, dict) else d
        for f in BUDGET_FLAGS:
            idx = [i for i, x in enumerate(av) if x == f]
            if len(idx) != 1 or idx[0] + 1 >= len(av) or av[idx[0] + 1].startswith("--"):
                die(f"{f} must appear exactly once with one value in {a.argv}")
            lit[f] = av[idx[0] + 1]
        b = d.get("budget", {}) if isinstance(d, dict) else {}
        prov = f"N={lit['--steps']} from s_step={b.get('s_step')} smoke_wall_s={b.get('smoke_wall_s')}"

    # ---- launch_gate.py -------------------------------------------------------------------------------------------
    a1 = "    open_items=_MAPFT_OPEN_ITEMS,\n"
    once(g, a1, "launch_gate.py open_items line")
    why = (f"    # MAPFT-BUDGET CLOSED ({a.mode}): PREREG_MAPFT s1 (sealed f8774a7b) fixes the budget BY RULE from the\n"
           f"    # measured 50-step smoke (N = min(10000, floor(0.85*(21600 - smoke_wall_s - 1800)/s_step)), N < 1000 -> STOP);\n"
           f"    # the token binds the FULL argv by content, so the N that launches is the N that was gated.\n"
           + (f"    # literals pinned below: {prov}\n" if a.mode == "literals" else
              "    # (open-only: the budget flags stay unpinned by the profile; the argv sha256 in the token is the pin)\n")
           + (f"    # {a.note}\n" if a.note else "")
           + "    open_items=(),\n")
    g2 = g.replace(a1, why)
    if a.mode == "literals":
        a2 = ('                         **{"--conflict-detector": "off"}),\n'
              '    required_values_why=dict({k: v for k, v in (_R8.get("required_values_why") or {}).items()\n'
              '                              if k not in set(_MAPFT_B1) | set(_MAPFT_BUDGET)}, **{\n'
              '        "--conflict-detector": "PI D-R9-MAPHEAD-PRECISION: no trunk gradient exists to measure"}),\n')
        once(g2, a2, "launch_gate.py required_values/_why block")
        vals = ", ".join(f'"{f}": "{lit[f]}"' for f in BUDGET_FLAGS)
        new2 = ('                         **{"--conflict-detector": "off", ' + vals + '}),\n'
                '    required_values_why=dict({k: v for k, v in (_R8.get("required_values_why") or {}).items()\n'
                '                              if k not in set(_MAPFT_B1) | set(_MAPFT_BUDGET)}, **{\n'
                '        "--conflict-detector": "PI D-R9-MAPHEAD-PRECISION: no trunk gradient exists to measure",\n'
                + "".join(f'        "{f}": "MAPFT-BUDGET closed: PREREG_MAPFT s1 rule applied to the measured smoke ({prov})",\n'
                          for f in BUDGET_FLAGS)
                + '    }),\n')
        g2 = g2.replace(a2, new2)

    # ---- test_map_only_ft.py --------------------------------------------------------------------------------------
    t1 = '    assert [it["id"] for it in MAPFT["open_items"]] == ["MAPFT-BUDGET"]\n'
    once(t, t1, "test open_items assertion")
    t2 = t.replace(t1, '    assert MAPFT["open_items"] == ()          # MAPFT-BUDGET CLOSED (mapft_close_budget.py, mode %s)\n' % a.mode)
    if a.mode == "literals":
        t3 = ('    for f in ("--steps", "--warmup", "--ema-decay", "--eval-every", "--b1-guard-ref-ap2m", "--b1-guard-step"):\n'
              '        assert f not in MAPFT["required_values"], f\n')
        once(t2, t3, "test required_values loop")
        t2 = t2.replace(t3, '    for f in ("--b1-guard-ref-ap2m", "--b1-guard-step"):\n'
                            '        assert f not in MAPFT["required_values"], f\n'
                            + "".join(f'    assert MAPFT["required_values"]["{f}"] == "{lit[f]}"\n' for f in BUDGET_FLAGS))
        t4 = '                 "--steps": ["3000"], "--warmup": ["100"], "--ema-decay": ["0.999"]}.items():\n'
        once(t2, t4, "test fixture budget values")
        t2 = t2.replace(t4, f'                 "--steps": ["{lit["--steps"]}"], "--warmup": ["{lit["--warmup"]}"], '
                            f'"--ema-decay": ["{lit["--ema-decay"]}"], "--eval-every": ["{lit["--eval-every"]}"]}}.items():\n')

    for rel, old, new in ((GATE_REL, g, g2), (TEST_REL, t, t2)):
        diff = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), rel, rel + " (closed)", n=2))
        print(diff)
    if a.dry_run:
        print("DRY-RUN: nothing written")
        print("ZZMAPFT-CLOSE-OKZZ")
        return 0
    gp.write_bytes(g2.replace("\n", gnl).encode("utf-8"))
    tp.write_bytes(t2.replace("\n", tnl).encode("utf-8"))
    print(f"WROTE {gp} and {tp}")
    print("ZZMAPFT-CLOSE-OKZZ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
