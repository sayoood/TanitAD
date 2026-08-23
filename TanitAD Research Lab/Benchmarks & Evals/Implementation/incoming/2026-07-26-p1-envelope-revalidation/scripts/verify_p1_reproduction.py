"""Verify, point by point, that the extended sweep REPRODUCES P1 on its own grid.

The pre-registration makes this a validity gate, not a nicety:

    "The 0..12 deg points must reproduce P1's committed values or the run is void."

If the shared grid points reproduce, the extension beyond 12 deg is a strict
continuation of P1's own curve rather than a new experiment that merely
resembles it. Compares `artifacts/yawext_12ep.json` (ours, `tanitad-eval`,
stack @ 0f93b98) against the COMMITTED
`…/2026-07-23-lower-ood-closedloop-source/lowood_flagship_ci.json` (P1, pod1).

No number here is transcribed by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[5]
_INC = _REPO / "TanitAD Research Hub" / "Benchmarks & Eval" / "Implementation" / "incoming"

P1 = _INC / "2026-07-23-lower-ood-closedloop-source" / "lowood_flagship_ci.json"
OURS = _HERE.parent / "artifacts" / "yawext_12ep.json"

TOL = 5e-5           # 4-dp agreement; both artifacts round to 4 dp


def rows(d, kind):
    return {float(r["amount"]): r for r in d["conditions"].get(kind, [])}


def main():
    p1 = json.loads(P1.read_text(encoding="utf-8"))
    ours = json.loads(OURS.read_text(encoding="utf-8"))

    out = {"_what": "point-by-point reproduction of P1 on its own grid",
           "_p1_artifact": str(P1.relative_to(_REPO)),
           "_ours_artifact": str(OURS.relative_to(_REPO)),
           "_tolerance": TOL, "checks": [], }

    def chk(label, got, want):
        ok = abs(float(got) - float(want)) <= TOL
        out["checks"].append({"quantity": label, "ours": round(float(got), 6),
                              "p1_committed": round(float(want), 6), "match": ok})
        return ok

    chk("baseline_real_frames.mean", ours["baseline_real_frames"]["mean"],
        p1["baseline_real_frames"]["mean"])
    chk("baseline_real_frames.lo", ours["baseline_real_frames"]["lo"],
        p1["baseline_real_frames"]["lo"])
    chk("baseline_real_frames.hi", ours["baseline_real_frames"]["hi"],
        p1["baseline_real_frames"]["hi"])
    chk("baseline.n_windows", ours["baseline_real_frames"]["n_windows"],
        p1["baseline_real_frames"]["n_windows"])
    chk("baseline.n_episodes", ours["baseline_real_frames"]["n_episodes"],
        p1["baseline_real_frames"]["n_episodes"])

    for kind in ("yaw", "lat"):
        a, b = rows(ours, kind), rows(p1, kind)
        for amt in sorted(set(a) & set(b)):
            chk(f"{kind}[{amt:g}].ade2s", a[amt]["ade2s_ci"]["mean"],
                b[amt]["ade2s_ci"]["mean"])
            chk(f"{kind}[{amt:g}].paired_delta",
                a[amt]["paired_vs_baseline"]["delta"],
                b[amt]["paired_vs_baseline"]["delta"])
            chk(f"{kind}[{amt:g}].paired_lo", a[amt]["paired_vs_baseline"]["lo"],
                b[amt]["paired_vs_baseline"]["lo"])
            chk(f"{kind}[{amt:g}].paired_hi", a[amt]["paired_vs_baseline"]["hi"],
                b[amt]["paired_vs_baseline"]["hi"])
            same_sep = (bool(a[amt]["paired_vs_baseline"]["separated"])
                        == bool(b[amt]["paired_vs_baseline"]["separated"]))
            out["checks"].append({"quantity": f"{kind}[{amt:g}].separated",
                                  "ours": a[amt]["paired_vs_baseline"]["separated"],
                                  "p1_committed": b[amt]["paired_vs_baseline"]["separated"],
                                  "match": same_sep})

    n = len(out["checks"])
    bad = [c for c in out["checks"] if not c["match"]]
    out["n_checks"] = n
    out["n_mismatch"] = len(bad)
    out["mismatches"] = bad
    out["VERDICT"] = ("REPRODUCED — the extension is a strict continuation of "
                      "P1's own curve" if not bad else
                      "MISMATCH — per the pre-registration this run is VOID")

    dest = _HERE.parent / "artifacts" / "p1_reproduction_check.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {dest}")
    print(f"\n{n} checks against the committed P1 artifact, {len(bad)} mismatch")
    print(out["VERDICT"])
    for c in bad:
        print("  MISMATCH", c)


if __name__ == "__main__":
    main()
