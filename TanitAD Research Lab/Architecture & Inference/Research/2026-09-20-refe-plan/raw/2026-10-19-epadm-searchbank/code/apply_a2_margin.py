"""Re-read a frozen-gate `admission.json` at the REGISTERED A2 margin.  The frozen module's A2-r1 bar is the constant 1.00 (`A2_R1_RATIO_BAR`); the registration (MM_RULING_EPADM_R1.md section 2) takes 1.05, and no CLI flag
sets it.  This script changes nothing in the module: it applies the module's own `a2_head_ok_r1(ci, bar=margin)` to the A2 intervals the module already wrote, and recombines the module's own `admission_label`.
    python apply_a2_margin.py admission_c1.json [--margin 1.05] [--out admission_c1_margin105.json]
A1 verdicts, the controls and the C-NULL void flag are taken from the module's output unchanged."""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import epadm_searchbank as SB                                                                # noqa: E402


def apply(adm: dict, margin: float = 1.05) -> dict:
    A = SB.gate()
    if adm.get("schema") != A.SCHEMA or adm.get("rules") != "r1":
        raise SB.BankError(f"not an ep-admission-v1 'r1' output: schema {adm.get('schema')!r} rules {adm.get('rules')!r}")
    cids = list(adm["per_candidate"])
    a2 = {}
    for cid in cids:
        cis = adm["per_candidate"][cid]["A2_reliability"]
        a2[cid] = "ADMIT-A2" if all(A.a2_head_ok_r1(cis[h], margin) for h in A.A2_HEADS) else "FAIL-A2"
    a1 = [adm["verdict"]["A1"][c] for c in cids]
    void_null = any(n.get("void", False) for n in adm["null"].values())
    label = A.admission_label(a1, [a2[c] for c in cids], bool(adm["controls_ok"]), void_null)
    return {"margin": margin, "A1": dict(zip(cids, a1)), "A2_at_margin": a2, "A2_at_module_default_1.00": adm["verdict"]["A2"], "label": label, "module_label_at_1.00": adm["verdict"]["label"],
            "note": "A1 / controls / C-NULL unchanged from the module's output; only the A2 bar is re-applied with the module's own a2_head_ok_r1."}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("admission_json")
    ap.add_argument("--margin", type=float, default=1.05)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = apply(json.load(open(a.admission_json, encoding="utf-8")), a.margin)
    txt = json.dumps(res, indent=1, sort_keys=True)
    if a.out:
        open(a.out, "w", encoding="utf-8", newline="\n").write(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
