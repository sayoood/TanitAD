"""Driver for the refcv4b hierarchy panel — PREREG_REFCV4B_HIERARCHY_EVAL.md §3.

Written while refcv4b was still training, so that the decisions taken on 2026-09-05 are
executed rather than remembered at 3 a.m.

⭐ **THE PANEL IS 9 ROLLS, NOT 12.** `--no-navshuf` / `--no-navzero` SKIP their arms, i.e.
`os_navzero` and `os_navshuf` are rolled by the FULL invocation **by default**, and
`--with-navflip` rides on it too. So one FULL roll yields **4 of the 12 registered arms**
(`FULL`, `os_navzero`, `os_navshuf`, `navflip`) plus the `ha` / `ha0` / `ha0_ext` floors;
the remaining 8 are the eval-time ablations. A driver that rolls twelve times pays ~50 %
more GPU for the same table.

⛔ **ORDER IS LOAD-BEARING: FULL RUNS FIRST.** `gstr_shuffle` reads a banked FULL dump
(`--gstr-bank`) and permutes it, so it cannot run before FULL exists. This is item E4 of
the Master Mind's carried list.

⛔ **NOTHING IS RE-PAID.** A roll whose record already parses is skipped. A roll whose DUMP
exists but whose record does not is recovered with `--analyze-only`, never re-rolled —
MEASURED 2026-08-11, an arm rolled 40 episodes / 6,844 windows and died in `analyze()`, and
`T1_EXIT=NO_ARMS_PRODUCED` read as a total failure when it was a complete run with a broken
last step.

⛔ **A record counts as produced only if it PARSES and carries no `_defects`.** A family that
read UNAVAILABLE because our own code raised is a DEFECT, not a refusal
(`refcv3_arm.DEFECT_EXCEPTIONS`), and the panel must not be assembled over one.

Usage::

    python taniteval/tools/run_hierarchy_panel.py \
        --ckpt <ckpt.pt> --episodes <v2 cache> --labels <v72.jsonl.gz> \
        --workdir <panel dir> [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ARM = os.path.join(HERE, "refcv3_arm.py")
PREFLIGHT = os.path.join(HERE, "panel_preflight.py")

#: The 8 eval-time ablations of §3 as corrected by ERRATUM-1. `frames_blind` keeps its own
#: named flag because the panel's VALIDITY rests on it: a gate never shown to FAIL an
#: image-blind arm certifies nothing.
ABLATIONS = [
    ("gstr_zero", ["--ablate", "gstr_zero"]),
    ("gstr_shuffle", ["--ablate", "gstr_shuffle"]),          # + --gstr-bank, added below
    ("e7_off", ["--ablate", "e7_off"]),
    ("e9_off", ["--ablate", "e9_off"]),
    ("h19_off", ["--ablate", "h19_off"]),
    ("ego_zero", ["--ablate", "ego_zero"]),
    ("sel_refined", ["--ablate", "sel_refined"]),
    ("frames_blind", ["--ablate-frames"]),
]


def collect_defects(rec):
    """Every `_defects` list in the record, top level AND one level down.

    ⛔ The writer puts them INSIDE the arm block — `refcv3_arm.py` appends to
    `rec["refcv3"]["_defects"]` while its own comment claims they are "collected at
    the top level".  A reader that checks only `rec["_defects"]` therefore returns
    ok=True for a record carrying a real defect, which is how the STRATEGIC family
    stayed silently absent.  Scanning both levels fixes the gate for the records that
    ALREADY EXIST, not merely for future ones.
    """
    found = list(rec.get("_defects") or [])
    for value in rec.values():
        if isinstance(value, dict):
            found.extend(value.get("_defects") or [])
    return found


def record_ok(path: str):
    """(ok, why). ⛔ Existence is not production: it must parse and be defect-free."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False, "absent or empty"
    try:
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
    except Exception as ex:                                     # noqa: BLE001
        return False, f"unparseable ({type(ex).__name__})"
    defects = collect_defects(rec)
    if defects:
        return False, f"carries {len(defects)} DEFECT(s): {defects[:2]}"
    return True, "parses, no defects"


def run(cmd, dry: bool) -> int:
    print("    $ " + " ".join(cmd), flush=True)
    if dry:
        return 0
    return subprocess.run(cmd).returncode


def roll(name, extra, a, dry: bool) -> bool:
    dump = os.path.join(a.workdir, f"dump_{name}")
    out = os.path.join(a.workdir, f"rec_{name}.json")

    ok, why = record_ok(out)
    if ok:
        print(f"  [SKIP  ] {name}: record already {why}", flush=True)
        return True

    base = [sys.executable, ARM, "--ckpt", a.ckpt, "--episodes", a.episodes,
            "--labels", a.labels, "--arm", name, "--out", out, "--dump-dir", dump]
    if a.episodes_n:
        base += ["--episodes-n", str(a.episodes_n)]

    #: the dump survived a failed analysis -> recover it, never re-roll.
    if os.path.isdir(dump) and any(f.startswith("ep") for f in os.listdir(dump)):
        print(f"  [RECOVER] {name}: dump exists, re-analysing at ZERO GPU", flush=True)
        rc = run(base + ["--analyze-only", dump] + extra, dry)
    else:
        print(f"  [ROLL  ] {name}", flush=True)
        rc = run(base + extra, dry)

    if dry:
        return True
    ok, why = record_ok(out)
    if rc != 0 or not ok:
        print(f"  [FAIL  ] {name}: exit {rc}, record {why}", flush=True)
        return False
    print(f"  [OK    ] {name}: {why}", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--episodes-n", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="⛔ only for a resumed run whose preflight already passed")
    a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    t0 = time.time()

    # ---- 0. the cheap parts, before any GPU ------------------------------------
    if not a.skip_preflight:
        print("[0/2] preflight (zero GPU)", flush=True)
        full_dump = os.path.join(a.workdir, "dump_FULL")
        pf = [sys.executable, PREFLIGHT, "--ckpt", a.ckpt, "--labels", a.labels]
        if os.path.isdir(full_dump):
            pf += ["--dump-dir", full_dump]
        if run(pf, a.dry_run) != 0 and not a.dry_run:
            print("STOP: PREFLIGHT FAILED -- not launching. An INCONCLUSIVE check is not a pass.")
            return 1

    # ---- 1. FULL first: it yields 4 arms + the floors + the gstr bank ----------
    print("\n[1/2] FULL -- also produces os_navzero, os_navshuf, navflip and the floors",
          flush=True)
    if not roll("FULL", ["--with-navflip"], a, a.dry_run):
        print("STOP: FULL FAILED -- every ablation is scored against it and gstr_shuffle "
              "reads its dump. Stopping rather than producing a partial panel.")
        return 1

    # ---- 2. the eight eval-time ablations -------------------------------------
    print("\n[2/2] the eight ablations", flush=True)
    full_dump = os.path.join(a.workdir, "dump_FULL")
    failed = []
    for name, extra in ABLATIONS:
        if name == "gstr_shuffle":
            extra = extra + ["--gstr-bank", full_dump]
        if not roll(name, extra, a, a.dry_run):
            failed.append(name)

    mins = (time.time() - t0) / 60.0
    print(f"\npanel finished in {mins:.1f} min")
    if failed:
        print(f"STOP: {len(failed)} arm(s) FAILED: {failed}")
        print("   Their dumps persist -- re-run this driver; completed arms are skipped "
              "and dumped-but-unanalysed arms recover at zero GPU.")
        return 1
    print("OK: 12 registered arms present (4 from FULL + 8 ablations).")
    print("NOT DONE HERE: the paired episode-cluster bootstrap across "
          "arms, the four metric families per arm, and the echo gate. This driver produces "
          "the records; it does not adjudicate them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
