#!/usr/bin/env python3
"""PREREG_V8F_ROLLHIST section 4.1, the chain checker's C-checks for the ROLLHIST stage-B pair (CPU, no model, no data): G-NAME and G-PAIR.

  G-NAME  the RH arm's argv NAMES both declaration flags (``--dagger-bank-readout causal_g1x``, ``--dagger-bank-kappa 0.2``) and the bank sha256 (``--dagger-bank-sha256``, 64 hex; equal to the
          bank's when ``--bank-sha256`` is given). CLAUDE.md: a flag whose default disables it must be named in the arm's argv, or the arm did not test it.
  G-PAIR  the B0 control's argv differs from RH's in EXACTLY the DAgger flags (``--dagger-p``, ``--dagger-bank``, ``--dagger-bank-sha256``), the two declaration flags and ``--out``; ``--dagger-seed`` may
          differ or not (the card's "four DAgger flags" count it; a B0 that names the same seed is still the same arm). Any other flag that differs (a lever moved, a path, a step count) is a problem.
          B0 names no non-default declaration (they are refused at ``--dagger-p 0``).

The stage-A/B tokens are minted by the chain checker (``gate_block3/block3_files/block3_prep.sh``, ``chk_arm3.sh mint``, ``study/dscale10k_check.py``): those are STUDY files outside ``stack/``, so this is a
separate stack script. Plug-in: ``import rollhist_pair_check as P; problems = P.pair_problems(rh_argv, b0_argv, bank_sha256=SHA)`` next to ``dscale10k_check.dagger_problems``; mint the token only on ``[]``.
CLI: ``python rollhist_pair_check.py --rh RH.argv.json --b0 B0.argv.json [--bank-sha256 SHA]`` (argv files: ``{"argv": [...]}`` or a bare list); exit 0 = PASS, 1 = problems printed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import os  # noqa: E402

_STACK = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _STACK not in sys.path:
    sys.path.insert(0, _STACK)
from tanitad.data.v8f_dagger import readout_token  # noqa: E402     the ONE spelling of the readout

RH_READOUT = readout_token("causal_g", 1.0, True)                  # the card's RH arm: causal_g1x (readout causal_g, gain 1, executed path)
DECL = {"--dagger-bank-readout": RH_READOUT, "--dagger-bank-kappa": "0.2"}
DECL_DEFAULT = {"--dagger-bank-readout": "pinned", "--dagger-bank-kappa": "none"}
REQUIRED_DIFF = {"--dagger-p", "--dagger-bank", "--dagger-bank-sha256", "--dagger-bank-readout", "--dagger-bank-kappa", "--out"}
OPTIONAL_DIFF = {"--dagger-seed"}
HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


def parse(argv) -> dict:
    """argv tokens -> {flag: value | True | [values]}; a flag given twice keeps the last value (argparse's rule)."""
    out, toks, i = {}, [str(t) for t in argv], 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            if "=" in t:
                k, v = t.split("=", 1)
                out[k] = v
                i += 1
                continue
            vals, j = [], i + 1
            while j < len(toks) and not (toks[j].startswith("--") and not re.match(r"^--?\d", toks[j])):
                vals.append(toks[j])
                j += 1
            out[t] = True if not vals else (vals[0] if len(vals) == 1 else vals)
            i = j
        else:
            i += 1
    return out


def name_problems(rh_argv, bank_sha256=None) -> list:
    a, bad = parse(rh_argv), []
    for k, want in DECL.items():
        if k not in a:
            bad.append(f"RH argv does not name {k} (a flag whose default disables it must be named): the arm would not test the declaration")
        elif a[k] != want:
            bad.append(f"RH argv names {k} {a[k]!r}, the RH bank is {want!r}")
    s = a.get("--dagger-bank-sha256")
    if s is None or s is True:
        bad.append("RH argv does not name --dagger-bank-sha256")
    elif not HEX64.match(str(s)) and not str(s).startswith("@"):
        bad.append(f"RH --dagger-bank-sha256 {s!r} is not 64 hex digits")
    elif bank_sha256 and str(s).lower() != str(bank_sha256).lower():
        bad.append(f"RH --dagger-bank-sha256 {str(s)[:12]} is not the bank's {str(bank_sha256)[:12]}")
    if "--dagger-bank" not in a:
        bad.append("RH argv does not name --dagger-bank")
    p = a.get("--dagger-p")
    try:
        if p is None or float(p) <= 0.0:
            bad.append(f"RH --dagger-p is {p!r}: the swap must be on")
    except (TypeError, ValueError):
        bad.append(f"RH --dagger-p {p!r} is not a number")
    return bad


def pair_problems(rh_argv, b0_argv, bank_sha256=None) -> list:
    bad = name_problems(rh_argv, bank_sha256)
    r, b = parse(rh_argv), parse(b0_argv)
    diff = {k for k in set(r) | set(b) if r.get(k) != b.get(k)}
    missing = sorted(REQUIRED_DIFF - diff)
    extra = sorted(diff - REQUIRED_DIFF - OPTIONAL_DIFF)
    if missing:
        bad.append(f"B0 does not differ from RH in {missing}: the pair is not the pair")
    if extra:
        bad.append(f"B0 differs from RH in {extra} beyond the DAgger flags, the two declarations and --out: a lever moved, the contrast is not attributable")
    for k, dflt in DECL_DEFAULT.items():
        if b.get(k, dflt) != dflt:
            bad.append(f"B0 names {k} {b[k]!r}: a declaration at --dagger-p 0 is refused by the trainer")
    try:
        if float(b.get("--dagger-p", 0.0)) != 0.0:
            bad.append(f"B0 --dagger-p is {b.get('--dagger-p')!r}: the control must name --dagger-p 0")
    except (TypeError, ValueError):
        bad.append(f"B0 --dagger-p {b.get('--dagger-p')!r} is not a number")
    if "--dagger-p" not in b:
        bad.append("B0 does not NAME --dagger-p (DAG-0 names only --dagger-p 0)")
    if r.get("--out") == b.get("--out"):
        bad.append("B0 and RH write to the same --out")
    return bad


def _load(p):
    o = json.load(open(p, encoding="utf-8"))
    return o["argv"] if isinstance(o, dict) else o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rh", required=True)
    ap.add_argument("--b0", required=True)
    ap.add_argument("--bank-sha256", default=None)
    a = ap.parse_args(argv)
    bad = pair_problems(_load(a.rh), _load(a.b0), a.bank_sha256)
    for m in bad:
        print("PROBLEM", m)
    print("ZZ ROLLHIST-PAIR", "PASS" if not bad else "FAIL")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
