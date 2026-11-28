#!/usr/bin/env python3
"""Gate for the refcv7 LEGAL row (R7_VMAXOFF): may Thor score this split's seam? (any python)

    python vmaxoff_gate.py --legal-dir <…/step50400/vmaxoff_legal> --split navtest [--out gate.json]

GO only if the dev-box orchestrator (``run_vmaxoff_legal7.py``) has written ``LEGAL_ROW_MANIFEST.json``
with, for this split: ``legal_verification.ok`` True (its ``verify_legal`` re-derived the LEGAL claim
from the banked rows), EVERY ``protocol_matches_R7_A1`` flag True (device, precision, exact_dedup,
ckpt md5, tree commit, KPR PASS, cv-standin count), no ``blocked`` key, and the seam npz present.
Prints/writes the seam sha256 so the copy on Thor can be checked against it. Exit 0 = GO, 3 = WAIT
(not decided yet), 4 = REFUSE (decided and failed) -- and the JSON says which; never read the code alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legal-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    man_p = os.path.join(a.legal_dir, "LEGAL_ROW_MANIFEST.json")
    seam = os.path.join(a.legal_dir, f"bridge_{a.split}", "seam_R7_VMAXOFF.npz")
    rep = {"split": a.split, "manifest": man_p, "seam": seam}
    try:
        man = json.load(open(man_p, encoding="utf-8"))
        rec = man["splits"][a.split]
    except (OSError, ValueError, KeyError) as e:
        rep.update(decision="WAIT", why=f"manifest/split not written yet: {e!r}")
    else:
        rep["manifest_record"] = rec
        why = []
        if rec.get("blocked"):
            why.append(f"blocked: {rec['blocked']}")
        lv = rec.get("legal_verification")
        pm = rec.get("protocol_matches_R7_A1")
        if lv is None or pm is None:
            rep.update(decision="WAIT" if not why else "REFUSE", why=why or ["verification not yet recorded"])
        else:
            if not lv.get("ok"):
                why.append(f"legal_verification failed: {lv.get('why')}")
            bad = [k for k, v in pm.items() if v is not True]
            if bad:
                why.append(f"protocol mismatches vs R7_A1: {bad}")
            if not os.path.exists(seam):
                why.append("seam npz missing")
            rep.update(decision="GO" if not why else "REFUSE", why=why)
            if not why:
                rep["seam_sha256"] = hashlib.sha256(open(seam, "rb").read()).hexdigest()
                rep["step"] = man.get("step")
                rep["ckpt_md5"] = man.get("ckpt_md5")
    if a.out:
        json.dump(rep, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps({k: rep.get(k) for k in ("split", "decision", "why", "seam_sha256")}, default=str))
    return {"GO": 0, "WAIT": 3, "REFUSE": 4}[rep["decision"]]


if __name__ == "__main__":
    raise SystemExit(main())
