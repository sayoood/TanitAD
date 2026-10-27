#!/usr/bin/env python3
"""Render box2_ship/chain_v8f_dagorc.sh from chain_v8f_dagorc.sh.in. EVERY pinned value is COMPUTED from a banked artifact, never retyped:

  DAG_s<S> checkpoint / argv.json / summary.json   v8f-dagger/M2b_s<S>_dag/SHA256SUMS.txt   (cross-checked against the DAG-OACC r2 chain's own pins: two independent copies must agree)
  DAG2-0_s<S> argv.json / summary.json             v8f-dagger2/M2b_s<S>_dag20/SHA256SUMS.txt
  the sourced lib and checks helper                sha256 of the INSTALLED C:/Users/Admin/box2_ship/oaf_lib.sh and v8f_oaf_chain_checks.py
  BUILD_REF                                        the full sha of 1d4f4fbb in the mirror (the commit DAG2-0 was trained at)
  EXPORT                                           export_closure_dagorc.json (files + data)

    python make_chain.py            writes box2_ship/chain_v8f_dagorc.sh (LF)
    python make_chain.py --check    regenerates in memory and compares with the file on disk (rc 0 = equal)
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHIP = HERE / "box2_ship"
DAG_RUNS = Path("E:/Projects/TanitAD-artifacts/v8f-dagger")
DAG2_RUNS = Path("E:/Projects/TanitAD-artifacts/v8f-dagger2")
INSTALLED = Path("C:/Users/Admin/box2_ship")
MIRROR = "C:/Users/Admin/tanitad-push"


def sums(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        h, n = line.split(" ", 1)
        out[n.strip().lstrip("*")] = h
    return out


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render() -> str:
    t = (SHIP / "chain_v8f_dagorc.sh.in").read_text(encoding="utf-8")
    sub = {}
    for s in (0, 1):
        d = sums(DAG_RUNS / f"M2b_s{s}_dag" / "SHA256SUMS.txt")
        sub[f"CKPT_SHA{s}"], sub[f"ARGV_SHA{s}"], sub[f"SUMM_SHA{s}"] = d["ckpt_last.pt"], d["argv.json"], d["summary.json"]
        d2 = sums(DAG2_RUNS / f"M2b_s{s}_dag20" / "SHA256SUMS.txt")
        sub[f"D20_ARGV_SHA{s}"], sub[f"D20_SUMM_SHA{s}"] = d2["argv.json"], d2["summary.json"]
        # the banked files themselves must hash to the SHA256SUMS values (argv / summary are small; the 87 MB checkpoint is hashed by the chain on box 2)
        assert sha(DAG_RUNS / f"M2b_s{s}_dag" / "argv.json") == sub[f"ARGV_SHA{s}"] and sha(DAG_RUNS / f"M2b_s{s}_dag" / "summary.json") == sub[f"SUMM_SHA{s}"], f"DAG_s{s}: banked files != SHA256SUMS.txt"
        assert sha(DAG2_RUNS / f"M2b_s{s}_dag20" / "argv.json") == sub[f"D20_ARGV_SHA{s}"] and sha(DAG2_RUNS / f"M2b_s{s}_dag20" / "summary.json") == sub[f"D20_SUMM_SHA{s}"], f"DAG2-0_s{s}: banked files != SHA256SUMS.txt"
    # cross-check against the DAG-OACC r2 chain's independent copy of the DAG pins
    oacc = (INSTALLED / "chain_v8f_dagoacc_r2.sh").read_text(encoding="utf-8")
    for var, key in (("CKPT_SHA", "CKPT_SHA"), ("ARGV_SHA", "ARGV_SHA"), ("SUMM_SHA", "SUMM_SHA")):
        m = re.search(rf"^{var}=\(([0-9a-f]{{64}}) ([0-9a-f]{{64}})\)", oacc, re.M)
        assert m and (m.group(1), m.group(2)) == (sub[f"{key}0"], sub[f"{key}1"]), f"{var} disagrees with the DAG-OACC r2 chain's pins"
    sub["OAF_LIB_SHA"] = sha(INSTALLED / "oaf_lib.sh")
    sub["OAF_CHECKS_SHA"] = sha(INSTALLED / "v8f_oaf_chain_checks.py")
    sub["BUILD_REF"] = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*", "rev-parse", "1d4f4fbb^{commit}"], capture_output=True, text=True, check=True).stdout.strip()
    assert re.fullmatch(r"[0-9a-f]{40}", sub["BUILD_REF"])
    ec = json.loads((HERE / "export_closure_dagorc.json").read_text(encoding="utf-8"))
    paths = sorted(set(ec["files"]) | set(ec["data"]))
    assert not any("'" in p for p in paths)
    sub["EXPORT"] = "\n".join(f"  '{p}'" for p in paths)
    for k, v in sub.items():
        assert t.count(f"@{k}@") >= 1, f"placeholder @{k}@ not in the template"
        t = t.replace(f"@{k}@", v)
    left = re.findall(r"@[A-Z0-9_]+@", t)
    assert not left, f"unfilled placeholders {left}"
    return t


def main() -> int:
    text = render()
    out = SHIP / "chain_v8f_dagorc.sh"
    if "--check" in sys.argv:
        ok = out.exists() and out.read_bytes() == text.encode("utf-8")
        print("chain_v8f_dagorc.sh == a fresh regeneration:", ok)
        return 0 if ok else 1
    out.write_bytes(text.encode("utf-8"))
    print(f"wrote {out} sha256 {hashlib.sha256(text.encode()).hexdigest()} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
