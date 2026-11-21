#!/usr/bin/env python3
"""G3 guard (P6-READ-GAPS, 2026-10-22): the scoring harness stage2_sft.py runs is the BANKED copy under eval/harness/, and every file of it is sha256-pinned.

WHY. stage2_sft.py used to take score_e.py, w3e/ and ev6e/families6.py from a previous session's %TEMP% scratchpad
(.../a085b71c-.../scratchpad). A temp cleanup already deleted the eval venv once; if the scratchpad goes, `families()` returns None and the verdict
silently becomes NOT PROVEN, and a scoring step that cannot import W3 dies AFTER the 4.8 h seam. The files are now banked in the repo and this
module REFUSES to let a read start on anything but the pinned bytes.

WHAT IS PINNED (eval/harness/HARNESS_PINS.json): every *.py under eval/harness/ by sha256 of its LF-normalised bytes (a core.autocrlf checkout turns
LF into CRLF; the normalisation makes the pin survive that and nothing else: any other change of a byte is refused).
Also the CALLER stage2 reaches through score_e.py (eval/score_navtest_refe.py): it must be one of the two known variants (pre- or post-NT-FIX1).
Origin of each banked file is recorded in the pins ("scratchpad_byte_identical", "scratchpad_plus_one_line", "repo_w3_original").

    python eval/harness_pins.py --verify            # exit 0 only if every pin holds
    python eval/harness_pins.py --build             # (re)write HARNESS_PINS.json from the files on disk (a deliberate act; review the diff)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "harness"
PINS_FILE = HARNESS / "HARNESS_PINS.json"
CALLER = HERE / "score_navtest_refe.py"


class HarnessPinError(SystemExit):
    """raised (as a SystemExit with a message) when the banked harness is not the pinned one"""


def sha256_lf(path) -> str:
    b = Path(path).read_bytes()
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


def _py_files(root: Path):
    out = []
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        out.append(p.relative_to(root).as_posix())
    return out


def load_pins(pins_file: Path = PINS_FILE) -> dict:
    return json.loads(Path(pins_file).read_text(encoding="utf-8"))


def verify(root: Path = HARNESS, pins: dict | None = None, caller: Path | None = CALLER) -> dict:
    """returns {"ok", "n_files", "mismatch", "missing", "extra", "caller", "pins_file_sha256"}; never raises on a mismatch"""
    root = Path(root)
    pins_file = root / "HARNESS_PINS.json"
    if pins is None:
        if not pins_file.exists():
            return {"ok": False, "n_files": 0, "mismatch": [], "missing": [], "extra": [], "caller": None,
                    "error": f"{pins_file} is absent: the harness is not banked here"}
        pins = load_pins(pins_file)
    want = pins["files"]
    mismatch, missing = [], []
    for rel, rec in want.items():
        p = root / rel
        if not p.is_file():
            missing.append(rel)
            continue
        got = sha256_lf(p)
        if got != rec["sha256_lf"]:
            mismatch.append({"file": rel, "got": got, "pinned": rec["sha256_lf"]})
    extra = [r for r in _py_files(root) if r not in want]          # an unpinned module could shadow a pinned one
    res = {"n_files": len(want), "mismatch": mismatch, "missing": missing, "extra": extra,
           "pins_file_sha256": (hashlib.sha256(pins_file.read_bytes().replace(b"\r\n", b"\n")).hexdigest() if pins_file.exists() else None)}
    cv = None
    if caller is not None:
        variants = pins.get("caller_variants", {})
        cp = Path(caller)
        if not cp.exists():
            cv = {"ok": False, "why": f"{cp} absent"}
        else:
            got = sha256_lf(cp)
            name = next((n for n, h in variants.items() if h == got), None)
            cv = {"ok": name is not None, "sha256_lf": got, "variant": name, "path": str(cp),
                  "why": None if name else "an unknown score_navtest_refe.py: the scoring path changed after the pins were written; review it and add its sha256_lf to caller_variants"}
    res["caller"] = cv
    res["ok"] = bool(not mismatch and not missing and not extra and (cv is None or cv["ok"]))
    return res


def require(root: Path = HARNESS, pins: dict | None = None, caller: Path | None = CALLER) -> dict:
    """the gate stage2_sft.py calls FIRST: SystemExit (message names every offending file) unless the banked harness equals its pins"""
    r = verify(root, pins, caller)
    if not r["ok"]:
        lines = ["⛔ HARNESS PIN GUARD: the banked scoring harness is not the pinned one; NOTHING was started."]
        if r.get("error"):
            lines.append("  " + r["error"])
        lines += [f"  missing:  {m}" for m in r.get("missing", [])]
        lines += [f"  CHANGED:  {m['file']} sha256_lf {m['got'][:16]}... != pinned {m['pinned'][:16]}..." for m in r.get("mismatch", [])]
        lines += [f"  UNPINNED: {e}" for e in r.get("extra", [])]
        if r.get("caller") and not r["caller"]["ok"]:
            lines.append(f"  CALLER:   {r['caller']['why']} ({r['caller'].get('sha256_lf')})")
        raise HarnessPinError("\n".join(lines))
    return r


def build(root: Path = HARNESS, origins: dict | None = None) -> dict:
    root = Path(root)
    prev = load_pins(root / "HARNESS_PINS.json") if (root / "HARNESS_PINS.json").exists() else {}
    files = {}
    for rel in _py_files(root):
        old = prev.get("files", {}).get(rel, {})
        files[rel] = {"sha256_lf": sha256_lf(root / rel), "bytes": (root / rel).stat().st_size,
                      **{k: v for k, v in old.items() if k not in ("sha256_lf", "bytes")}}
        if origins and rel in origins:
            files[rel].update(origins[rel])
    out = {"what": "sha256 (LF-normalised) of every file of the banked scoring harness read by eval/stage2_sft.py (P6-READ-GAPS G3)",
           "files": files, "caller_variants": prev.get("caller_variants", {})}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--root", default=str(HARNESS))
    a = ap.parse_args(argv)
    if a.build:
        out = build(Path(a.root))
        (Path(a.root) / "HARNESS_PINS.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {Path(a.root) / 'HARNESS_PINS.json'}: {len(out['files'])} files")
        return 0
    r = verify(Path(a.root))
    print(json.dumps({k: v for k, v in r.items()}, indent=1))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
