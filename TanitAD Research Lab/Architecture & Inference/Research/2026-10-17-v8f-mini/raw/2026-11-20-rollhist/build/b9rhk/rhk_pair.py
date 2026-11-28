#!/usr/bin/env python3
"""rhk_pair.py --rh RHK.argv.json --b0 B0.argv.json --bank-sha256 SHA   -- the STOCK pair check (tree stack/scripts/rollhist_pair_check.py, unchanged, not patched) on an RHK argv.
That file hard-codes the declaration causal_g1x. This wrapper (1) REQUIRES the RHK argv to declare exactly causal_g0x, (2) swaps that one token for the stock spelling in a COPY, (3) runs the stock pair_problems on the copy,
so every other stock rule (bank sha named, DAgger flags the only difference to B0, B0 p = 0, no shared --out) runs as written. The pin of the K token itself is the study checker's (C-ARGV, C-RH-BANKGEN).
Last line `ZZ ROLLHIST-PAIR PASS` (the string rh_gate_record.py reads) or `ZZ ROLLHIST-PAIR FAIL`."""
import argparse
import json
import os
import sys

TREE = os.environ.get("RH_TREE", "/root/rollhist/tree_cdb")
for p in (TREE + "/stack", TREE + "/stack/scripts"):
    if p not in sys.path:
        sys.path.insert(0, p)
import rollhist_pair_check as P  # noqa: E402

TOK_K, TOK_STOCK = "causal_g0x", "causal_g1x"


def load(p):
    o = json.load(open(p, encoding="utf-8"))
    return [str(x) for x in (o["argv"] if isinstance(o, dict) else o)]


def rhk_problems(rh, b0, bank_sha=None):
    if P.RH_READOUT != TOK_STOCK:
        return [f"the stock pair check declares {P.RH_READOUT!r}, expected {TOK_STOCK!r}"]
    if "--dagger-bank-readout" not in rh or rh[rh.index("--dagger-bank-readout") + 1] != TOK_K:
        return [f"the RHK argv does not declare --dagger-bank-readout {TOK_K!r}"]
    cp = list(rh)
    cp[cp.index("--dagger-bank-readout") + 1] = TOK_STOCK
    return list(P.pair_problems(cp, b0, bank_sha256=bank_sha))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rh", required=True)
    ap.add_argument("--b0", required=True)
    ap.add_argument("--bank-sha256", default=None)
    a = ap.parse_args(argv)
    bad = rhk_problems(load(a.rh), load(a.b0), a.bank_sha256)
    for m in bad:
        print("PROBLEM", m)
    print("ZZ ROLLHIST-PAIR", "PASS" if not bad else "FAIL", f"(readout {TOK_K}; stock checker on a token-swapped copy)")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
