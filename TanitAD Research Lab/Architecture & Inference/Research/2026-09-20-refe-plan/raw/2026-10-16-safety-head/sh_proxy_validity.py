#!/usr/bin/env python3
"""PREREG_SAFEHEAD.md s.6: phead control (d)'s rule applied to TTC, NC and DAC, with NO new compute, from the banked phead proxy_validity.json and the real RL-val B - A of
raw/2026-10-15-rls1-adam-dose/ab_vs_base.json.  Rule (phead's, per term): P2 B - A negative at >= 2 of the 3 R0 states (u250/500/1000) AND |P2 B - A| < 0.5 at every C0 floor state,
AND (added here, because the rule is only meaningful against a measured trade) the real RL-val B - A of that term exists and has the proxy's sign at >= 2 of 3 R0 states.
A term without a real reference is UNVALIDATED.  -> proxy_validity_safety.json"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
PV = json.load(open(os.path.join(HERE, "..", "2026-10-16-phead-epo-feasibility", "proxy_validity.json")))["states"]
REAL = json.load(open(os.path.join(HERE, "..", "2026-10-15-rls1-adam-dose", "ab_vs_base.json")))["rows"]
out = {"rule": __doc__, "terms": {}}
for nm in ("TTC", "NC", "DAC"):
    r0 = [PV[f"r0_u{u}"][f"P2_pi_{nm}_B_minus_A_x100"]["mean"] for u in (250, 500, 1000)]
    c0 = [PV[f"c0_u{u}"][f"P2_pi_{nm}_B_minus_A_x100"]["mean"] for u in (250, 500, 1000)]
    real = [REAL[f"r0_u{u}"]["B_minus_A"].get(nm, {}).get("mean") for u in (250, 500, 1000)]
    has_real = all(x is not None for x in real)
    sign_agree = sum(1 for p, q in zip(r0, real) if q is not None and (p < 0) == (q < 0)) if has_real else 0
    ok = sum(x < 0 for x in r0) >= 2 and all(abs(x) < 0.5 for x in c0) and has_real and sign_agree >= 2
    out["terms"][nm] = {"P2_R0_B_minus_A_x100": r0, "P2_C0_floor_x100": c0, "real_RLval_R0_B_minus_A_x100": real, "real_reference_exists": has_real,
                        "sign_agreement_of_3": sign_agree, "status": "VALIDATED" if ok else "UNVALIDATED"}
json.dump(out, open(os.path.join(HERE, "proxy_validity_safety.json"), "w"), indent=1)
print({k: (v["status"], [round(x, 2) for x in v["P2_R0_B_minus_A_x100"]], v["real_RLval_R0_B_minus_A_x100"]) for k, v in out["terms"].items()})
