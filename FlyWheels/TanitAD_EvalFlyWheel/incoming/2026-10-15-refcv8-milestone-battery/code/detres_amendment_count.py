"""detres_amendment_count.py -- the EXACT count behind a proposed numerics-floor tolerance for DETECTION AP cells (draft amendment D-R8-1).

Reads  (a) the step-5,000 G0 (`g0.json`, verdict_A7.terms: in-run value, replay mean, class, support n for every judged term) and
       (b) `detection_residual_probe.json` (per-arm `ap_cells`: EVERY AP cell the G0 judged, recomputed in each numerics-only arm)
and, for the rule  tol_k = max(0.02, 2/n_k, K * phi_k)  with phi_k = rms over numerics-only arms of (v_arm,k - v_A0,k), K = 3 (SPEC section 2's own
convention for a measured floor; A6_K), reports per cell: registered tolerance, phi, the proposed tolerance, and whether the verdict changes.
No judgement is made here that G0 passes: this prints the counts the Master Mind needs to register (or refuse) the amendment.

    python detres_amendment_count.py --g0 E:/.../g0.json --probe <detection_residual_probe.json> --out <json> [--min-n 30] [--max-n 1e9] [--k 3]
Evidence class: MEASURED (arms) + arithmetic.  Needs no GPU.
"""
from __future__ import annotations

import argparse
import json
import math

NUMERICS_ARMS = ("A1",)             # + every "A2:<seed>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=float, default=3.0)
    ap.add_argument("--min-n", type=float, default=30.0)
    ap.add_argument("--max-n", type=float, default=float("inf"), help="apply the floor only to cells with support n < max-n (the 'low-count' scope)")
    a = ap.parse_args()
    g = json.load(open(a.g0, encoding="utf-8"))
    T = g["verdict_A7"]["terms"]
    P = json.load(open(a.probe, encoding="utf-8"))
    arms = P["arms"]
    a0 = (arms.get("A0") or {}).get("ap_cells")
    if not a0 or arms["A0"].get("status") != "DONE":
        raise SystemExit("A0 ap_cells missing: the probe has not run A0")
    used = [k for k, v in arms.items() if v.get("status") == "DONE" and (k in NUMERICS_ARMS or k.startswith("A2:")) and v.get("ap_cells")]
    if not used:
        raise SystemExit("no numerics-only arm with ap_cells")
    cells = []
    for k, r in T.items():
        if r.get("cls") != "DETECTION" or "_det_ap" not in k or r.get("support_n") is None:
            continue
        n = float(r["support_n"])
        if n < a.min_n:
            continue
        devs = [(arms[u]["ap_cells"].get(k), a0.get(k)) for u in used]
        devs = [(x, y) for x, y in devs if x is not None and y is not None]
        phi = math.sqrt(sum((x - y) ** 2 for x, y in devs) / len(devs)) if devs else None
        reg = max(0.02, 2.0 / n)
        in_scope = n < a.max_n
        prop = max(reg, a.k * phi) if (phi is not None and in_scope) else reg
        dev = abs(float(r["mean"]) - float(r["inrun"]))
        cells.append({"key": k, "n": n, "inrun": r["inrun"], "replay": r["mean"], "dev": round(dev, 5), "tol_registered": round(reg, 5),
                      "phi_rms": (None if phi is None else round(phi, 5)), "tol_proposed": round(prop, 5), "in_scope": in_scope,
                      "ok_registered": dev <= reg, "ok_proposed": dev <= prop,
                      "widened": prop > reg + 1e-12})
    out = {"tool": "detres_amendment_count.py", "rule": f"tol = max(0.02, 2/n, {a.k:g} * phi), phi = rms over numerics-only arms {used} of (v_arm - v_A0); cells with {a.min_n:g} <= n < {a.max_n:g}",
           "n_cells_detection_support_ge_min": len(cells), "n_arms_for_phi": len(used),
           "n_ok_registered": sum(c["ok_registered"] for c in cells), "n_ok_proposed": sum(c["ok_proposed"] for c in cells),
           "n_newly_ok": [c["key"] for c in cells if c["ok_proposed"] and not c["ok_registered"]],
           "n_newly_failing": [c["key"] for c in cells if c["ok_registered"] and not c["ok_proposed"]],
           "n_widened": sum(c["widened"] for c in cells),
           "cells_widened": [c for c in cells if c["widened"]][:60],
           "target_cell": next((c for c in cells if c["key"] == "eval_box3d_det_ap2_automobile_0_20"), None),
           "phi_rms_by_arm_count": len(used), "evidence": "MEASURED (probe arms) + arithmetic; phi from only the arms listed (few draws: crude)"}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    print("DETRES_AMEND_COUNT", out["n_cells_detection_support_ge_min"], out["n_ok_registered"], out["n_ok_proposed"], out["n_newly_ok"], out["n_widened"])


if __name__ == "__main__":
    main()
