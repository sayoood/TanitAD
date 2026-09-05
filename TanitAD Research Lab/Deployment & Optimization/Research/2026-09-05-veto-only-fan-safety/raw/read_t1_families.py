#!/usr/bin/env python3
"""P4 readout: the FOUR FAMILIES at T1, per family, never pooled — and read against the
SEED-REPLICATE floor, because a separated CI from one training run is necessary and not
sufficient (`H-ESTIM-SEED-1`).

Consumes `paired_openloop.py` records (`--out` JSONs) for the SAME arm at two seeds against
the same base, and emits one table plus the guard verdicts.

  lever(m, K)   the paired delta for seed K, base vs arm, over the shared `ha0` floor
  floor(m)      |lever_s0 - lever_s1|   -- the rig's own run-to-run noise on that metric
  QUOTABLE(m)   both separated AND same sign AND min(|lever|) > floor(m)

⛔ ADE alone is an INCOMPLETE result (binding, 2026-08-02): ADE/FDE are printed BESIDE the
four families, never as "the result".
⚠️ The `TAC_declared_*` and `STR_route_*` rows are STRUCTURAL ZEROS, not "no harm": the RL
trains `core.decoder` only, and `core.maneuver` / `core.route` / the goal heads / `conf_head`
are in `forbidden_prefixes` and frozen. Identical inputs through frozen heads give identical
outputs — an identity, not an estimate (the `H-ECHO-4` class). They are labelled as such.
"""
import argparse
import json
import os

FAMILIES = ("ADE", "LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")
STRUCTURAL_PREFIXES = ("TAC_declared_", "STR_route_")


def J(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s0", required=True, help="paired_openloop record, seed 0")
    ap.add_argument("--s1", default=None, help="paired_openloop record, seed 1 (the replicate)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    r0 = J(a.s0)
    r1 = J(a.s1) if a.s1 and os.path.isfile(a.s1) else None
    out = {"_tool": "read_t1_families.py",
           "_tier": "T1 — self-action OPEN loop (PI ruling 2026-09-02); never a closed-loop claim",
           "_evidence_class": "MEASURED (ours)",
           "s0_record": a.s0, "s1_record": a.s1,
           "replicate_present": r1 is not None,
           "void_s0": r0.get("void"), "void_s1": (r1 or {}).get("void"),
           "gates_s0": r0.get("gates"), "power_s0": r0.get("power"),
           "intersection_s0": r0.get("intersection"), "estimator": r0.get("estimator"),
           "families": {}}
    if r1 is None:
        out["_LIMIT"] = ("NO REPLICATE: every row below is a ONE-SEED result and is "
                         "NECESSARY-NOT-SUFFICIENT under H-ESTIM-SEED-1. Re-run with --s1.")

    print(f"\n=== T1 FOUR FAMILIES (self-action OPEN loop) — void_s0={r0.get('void')} "
          f"replicate={'yes' if r1 else 'NO'} ===")
    for fam in FAMILIES:
        f0 = (r0.get("families") or {}).get(fam) or {}
        f1 = (r1.get("families") or {}).get(fam) if r1 else {}
        m0 = f0.get("metrics") or {}
        m1 = (f1 or {}).get("metrics") or {}
        out["families"][fam] = {}
        print(f"\n--- {fam} ---")
        print(f"  {'metric':32s} {'d_s0':>10s} {'d_s1':>10s} {'seedfl':>9s} {'sep0':>5s} "
              f"{'sep1':>5s}  verdict")
        for k in sorted(m0):
            d0 = m0[k]
            d1 = m1.get(k)
            structural = k.startswith(STRUCTURAL_PREFIXES)
            lev0 = float(d0.get("delta", float("nan")))
            lev1 = float(d1.get("delta", float("nan"))) if d1 else float("nan")
            fl = abs(lev0 - lev1) if d1 else float("nan")
            both = bool(d0.get("separated") and (d1 or {}).get("separated"))
            same = (lev0 * lev1) > 0 if d1 else False
            clears = (min(abs(lev0), abs(lev1)) > fl) if d1 else False
            if structural and lev0 == 0.0:
                v = "STRUCTURAL ZERO (frozen head — identity, not an estimate)"
            elif d1 is None:
                v = "ONE SEED — necessary, not sufficient" if d0.get("separated") else "ns"
            elif both and same and clears:
                v = "QUOTABLE"
            elif both and same:
                v = "WITHIN-NOISE (separated but under the seed-replicate floor)"
            else:
                v = "ns"
            out["families"][fam][k] = {"lever_s0": d0, "lever_s1": d1,
                                       "seed_replicate_floor": fl, "structural": structural,
                                       "verdict": v}
            print(f"  {k:32s} {lev0:+10.4f} {lev1:+10.4f} {fl:9.4f} "
                  f"{str(bool(d0.get('separated'))):>5s} "
                  f"{str(bool((d1 or {}).get('separated'))):>5s}  {v}")
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n[t1] -> {a.out}")
    if r1 is None:
        print("[t1] ⚠️ NO REPLICATE — every row is one-seed and necessary-not-sufficient.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
