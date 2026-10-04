#!/usr/bin/env python3
"""Thor-vs-export AgentInput fingerprint audit (any numpy python).

Reads a ``make_ulp_reference.py`` dump made ON THOR (AgentInput + fingerprint per token) and the seam's
recorded fingerprints (``seam_*.npz['fingerprint']``, computed on the dev box at export time), and
lists every token whose Thor fingerprint differs -- i.e. every token E2/W3's byte-exact guard would
refuse on Thor. Writes those tokens in W3 token-file format so ``make_ulp_reference.py`` can be run
ON THE DEV BOX for exactly them (the ULP reference). The AgentInput depends only on the scene, never
on the arm, so one reference per split serves every arm and every checkpoint step (checked here:
``--seam`` may be given several times and their fingerprint tables must agree).

    python fp_audit.py --thor agentinput_thor.json --seam seamA.npz [--seam seamB.npz] \
        --tokens tokens.json --out audit.json --mismatch-tokens mismatch_tokens.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thor", required=True)
    ap.add_argument("--seam", action="append", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mismatch-tokens", required=True)
    a = ap.parse_args()
    thor = json.load(open(a.thor, encoding="utf-8"))["tokens"]
    tok_doc = json.load(open(a.tokens, encoding="utf-8"))
    tables = []
    for s in a.seam:
        z = np.load(s, allow_pickle=False)
        tables.append({str(t): str(f) for t, f in zip(z["token"], z["fingerprint"])})
    base = tables[0]
    seams_agree = all(all(tb.get(t) == f for t, f in base.items() if t in tb) for tb in tables[1:])
    want = [t for t in tok_doc["tokens"]]
    missing_thor = [t for t in want if t not in thor]
    missing_seam = [t for t in want if t not in base]
    mism = [t for t in want if t in thor and t in base and thor[t]["fingerprint"] != base[t]]
    rep = {"n_tokens": len(want), "n_thor": len(want) - len(missing_thor), "n_seam": len(want) - len(missing_seam),
           "n_fingerprint_equal": len(want) - len(missing_thor) - len(missing_seam) - len(mism),
           "n_fingerprint_differ": len(mism), "frac_differ": (len(mism) / max(1, len(want))),
           "missing_thor": missing_thor[:50], "missing_seam": missing_seam[:50],
           "seams": a.seam, "seam_fingerprint_tables_agree": seams_agree, "mismatch_tokens": mism}
    json.dump(rep, open(a.out, "w", encoding="utf-8"), indent=1)
    tl = tok_doc.get("token_log", {})
    json.dump({"rule": "tokens whose Thor AgentInput fingerprint differs from the seam export",
               "tokens": mism, "token_log": {t: tl[t] for t in mism if t in tl}},
              open(a.mismatch_tokens, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: rep[k] for k in ("n_tokens", "n_fingerprint_equal", "n_fingerprint_differ",
                                          "frac_differ", "seam_fingerprint_tables_agree")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
