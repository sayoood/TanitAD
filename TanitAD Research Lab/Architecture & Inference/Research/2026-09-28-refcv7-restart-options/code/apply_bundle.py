#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""refcv7 restart options -- apply a chosen OPTION to a CLEAN tree in one step, blob-asserted.

⛔ Run it ONLY at a restart the PI has decided, on a tree extracted from the tip with
`core.autocrlf=false` (every base blob is checked; any drift REFUSES before a byte is written).
Every closure-touching file below voids G-MAP-OVERFIT and G-BOX-OVERFIT: the binding RUNS must be
re-done on the resulting tree (RESTART_OPTIONS.md sec. 2).

Options (RESTART_OPTIONS.md sec. 1):
  R4  (recommended) the bundle: freeze + lever 2 + getstate + I3 + conflict 50 argv
  R4c the bundle WITHOUT the cadence change (freeze + lever 2 + getstate + I3)
  R2  conflict 50 argv only
  R3  lever 2 only
  R1  freeze only (+ its converter and test)

What each component writes (tree-relative):
  freeze   stack/tanitad/refs/refc_v3.py, stack/tanitad/train/declared_vs_built.py (apply_freeze.py)
           + NEW stack/scripts/refcv7_ckpt_freeze_convert.py, stack/tests/test_refcv7_admitted_freeze.py
  lever2   stack/scripts/refc_v3_train.py (step-cost package apply_lever2.py, 203b437f -> fd874293)
  getstate stack/tanitad/data/v2_dataset.py (+ NEW stack/tests/test_v2_cache_pickle_state.py)
  i3       stack/tanitad/eval/refcv6_loader.py, stack/scripts/g_box_overfit.py,
           stack/tests/test_g_box_overfit.py (+ NEW stack/tests/test_refcv6_loader_stamped_queries.py)
  conflict50  stack/ops/runs.d/refcv7-r101-s0.argv.json := the conflict50 variant (argv sha e46ad3eb...)
              + stack/tests/test_launch_gate.py: its FINAL-argv pin `_FINAL_ARGV_SHA256` 6402d33d -> e46ad3eb
              (MEASURED: without it that pin goes RED on the R4 tree -- a G-SUITE regression)

Usage: apply_bundle.py <tree-root> --option R4 [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPCOST = HERE.parents[1] / "2026-09-27-refcv7-step-cost" / "code"
OPTIONS = {"R4": ("freeze", "lever2", "getstate", "i3", "conflict50"),
           "R4c": ("freeze", "lever2", "getstate", "i3"),
           "R2": ("conflict50",), "R3": ("lever2",), "R1": ("freeze",)}
NEW_FILES = {  # component -> [(package source, tree destination, expected blob)]
    "freeze": [("freeze/stack/scripts/refcv7_ckpt_freeze_convert.py",
                "stack/scripts/refcv7_ckpt_freeze_convert.py", None),
               ("freeze/stack/tests/test_refcv7_admitted_freeze.py",
                "stack/tests/test_refcv7_admitted_freeze.py", None)],
    "getstate": [("getstate/stack/tests/test_v2_cache_pickle_state.py",
                  "stack/tests/test_v2_cache_pickle_state.py", None)],
    "i3": [("i3/stack/tests/test_refcv6_loader_stamped_queries.py",
            "stack/tests/test_refcv6_loader_stamped_queries.py", None)],
}
ARGV_REL = "stack/ops/runs.d/refcv7-r101-s0.argv.json"
ARGV_BASE_BLOB = None      # checked by content instead: the canonical gate sha below
ARGV_CANONICAL_SHA = "6402d33de75b7f1c6dbdeb9aeedd46a00a82e7325eec420fa179f366213bd5cd"
ARGV_CONFLICT50_SHA = "e46ad3eb4099264754ba97846c0f3f5e8447ce61e21124dc980e2eb64464e9c8"
#: the launch gate's test pins the canonical argv sha: it moves WITH the argv
TLG_REL = "stack/tests/test_launch_gate.py"
TLG_BASE = "374dd42173a1c6a7c17e7e4896293a5010e28895"
TLG_OLD = f'_FINAL_ARGV_SHA256 = "{ARGV_CANONICAL_SHA}"\n'
TLG_NEW = ("#: refcv7 RESTART (restart-options package 2026-09-28): the FINAL list with "
           "--conflict-every\n"
           "#: 10 -> 50 (an instrument cadence; training numerics bit-identical). Launch list: "
           "6402d33d.\n"
           f'_FINAL_ARGV_SHA256 = "{ARGV_CONFLICT50_SHA}"\n')


def git_blob(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\x00" % len(b) + b).hexdigest()


def gate_sha(argv: list[str]) -> str:
    return hashlib.sha256(json.dumps(argv, ensure_ascii=False, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def run(cmd: list[str]) -> None:
    r = subprocess.run([sys.executable] + cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"[bundle] REFUSED by {Path(cmd[0]).name} (rc {r.returncode})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tree")
    ap.add_argument("--option", required=True, choices=sorted(OPTIONS))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.tree)
    comps = OPTIONS[a.option]
    chk = ["--check"] if a.check else []
    # 1) every base first (--check pass of every applier), THEN write: no half-patched tree
    plan = []
    if "freeze" in comps:
        plan.append([str(HERE / "freeze" / "apply_freeze.py"), str(root)])
    if "getstate" in comps:
        plan.append([str(HERE / "getstate" / "apply_getstate_fix.py"), str(root)])
    if "i3" in comps:
        plan.append([str(HERE / "i3" / "apply_i3_patch.py"), str(root)])
    for cmd in plan:
        run(cmd + ["--check"])
    if "lever2" in comps:
        run([str(STEPCOST / "apply_lever2.py"), str(root / "stack/scripts/refc_v3_train.py")])
    argv_doc = None
    if "conflict50" in comps:
        cur = json.loads((root / ARGV_REL).read_text(encoding="utf-8"))
        if gate_sha(list(cur["argv"])) != ARGV_CANONICAL_SHA:
            raise SystemExit(f"[bundle] {ARGV_REL} is not the launch argv (sha "
                             f"{gate_sha(list(cur['argv']))[:12]}) -- REFUSED")
        argv_doc = json.loads((HERE / "argv" / "refcv7-r101-s0.conflict50.argv.json")
                              .read_text(encoding="utf-8"))
        if gate_sha(list(argv_doc["argv"])) != ARGV_CONFLICT50_SHA:
            raise SystemExit("[bundle] the conflict50 variant does not hash to its pin -- REFUSED")
        print(f"[bundle] argv {ARGV_CANONICAL_SHA[:12]} -> {ARGV_CONFLICT50_SHA[:12]} "
              f"(--conflict-every 10 -> 50)")
        tlg = (root / TLG_REL).read_bytes()
        if git_blob(tlg) != TLG_BASE or tlg.decode("utf-8").count(TLG_OLD) != 1:
            raise SystemExit(f"[bundle] {TLG_REL} is not the launch blob {TLG_BASE[:12]} -- REFUSED")
        print(f"[bundle] {TLG_REL}: _FINAL_ARGV_SHA256 {ARGV_CANONICAL_SHA[:12]} -> "
              f"{ARGV_CONFLICT50_SHA[:12]}")
    for c in comps:
        for src, dst, _b in NEW_FILES.get(c, []):
            if (root / dst).exists():
                raise SystemExit(f"[bundle] {dst} already exists in the tree -- REFUSED")
            print(f"[bundle] NEW {dst} <- {src} ({git_blob((HERE / src).read_bytes())[:12]})")
    if a.check:
        print(f"[bundle] CHECK OK for {a.option}: {', '.join(comps)} (nothing written)")
        return 0
    # 2) write
    for cmd in plan:
        run(cmd)
    if "lever2" in comps:
        run([str(STEPCOST / "apply_lever2.py"), str(root / "stack/scripts/refc_v3_train.py"),
             "--write"])
    if argv_doc is not None:
        # the CANONICAL object keeps every descriptive field; only `argv` (what the gate binds)
        # changes, and the change is recorded beside it
        cur = json.loads((root / ARGV_REL).read_text(encoding="utf-8"))
        cur["argv"] = list(argv_doc["argv"])
        cur["restart_changes_vs_launch_fec3a0d"] = argv_doc["changes_vs_launch"]
        (root / ARGV_REL).write_text(json.dumps(cur, indent=1, ensure_ascii=False) + "\n",
                                     encoding="utf-8")
        back = json.loads((root / ARGV_REL).read_text(encoding="utf-8"))["argv"]
        if gate_sha(list(back)) != ARGV_CONFLICT50_SHA:
            raise SystemExit("[bundle] the written argv does not hash to the conflict50 pin")
        t = (root / TLG_REL).read_bytes().decode("utf-8").replace(TLG_OLD, TLG_NEW)
        (root / TLG_REL).write_bytes(t.encode("utf-8"))
    for c in comps:
        for src, dst, _b in NEW_FILES.get(c, []):
            (root / dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(HERE / src, root / dst)
    print(f"[bundle] APPLIED {a.option}: {', '.join(comps)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
