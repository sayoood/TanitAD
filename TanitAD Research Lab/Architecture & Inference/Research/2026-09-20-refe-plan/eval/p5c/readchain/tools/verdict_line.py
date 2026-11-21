#!/usr/bin/env python3
"""The one-line verdict of a P5-C read, plus the C0r trigger (PREREG_P5C section 6.3), computed from p5c_read.py's JSON only.

  verdict_line.py --read read.json --strategic strategic_family_C1_C0.json --out verdict.txt

C0r trigger (card 6.3, committed): C0r trains iff the primary verdict is SEPARATED-SINGLE, i.e. P1-a (CI-lo(d) > 0), P1-b (d > B = 0.7031), G1, G2 and
CTRL-1 all pass, AND the PI approves (+28-29 L40-h, ESTIMATED). Every other verdict: no replicate can turn it into LEVER.
Nothing is decided here; this only reads the clauses the read already computed. REHEARSAL reads print as such and NEVER trigger.
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--read", required=True)
    ap.add_argument("--strategic", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = json.load(open(a.read, encoding="utf-8"))
    v = r["verdict"]
    name = v["verdict"]
    reh = r.get("REHEARSAL")
    core = name.split(":", 1)[-1]
    cl = v.get("clauses", {})
    d = r["d_headline"]["d_tracked"]
    dt = r["d_headline"].get("d_transparent") or {}
    trig = (core == "SEPARATED-SINGLE") and not reh
    all_clauses = bool(cl) and cl.get("P1-a") and cl.get("P1-b") and cl.get("G1") == "PASS" and cl.get("G2") == "PASS" and cl.get("CTRL-1")
    if trig != bool(all_clauses and not reh):
        print("INCONSISTENT: verdict and clauses disagree; read.json is not to be quoted", file=sys.stderr)
        return 2
    strat = "strategic: not supplied"
    if a.strategic:
        try:
            s = json.load(open(a.strategic, encoding="utf-8"))
            strat = "strategic: " + ("INVALID (not quotable)" if "INVALID" in json.dumps(s.get("controls", s))[:4000].upper() else "reported, no bar") + f" ({a.strategic})"
        except Exception as e:  # noqa: BLE001
            strat = f"strategic: unreadable ({type(e).__name__})"
    line = (f"P5C-READ verdict={name} qualifier={r['qualifier']['code']} n={r['n_tokens']} logs={r['n_logs']} "
            f"d_tracked={d['point']:.4f} [{d['lo']:.4f}, {d['hi']:.4f}] B=0.7031 "
            f"d_transparent={(dt.get('point') if dt else float('nan')):.4f} "
            f"P1a={cl.get('P1-a')} P1b={cl.get('P1-b')} G1={cl.get('G1')} G2={cl.get('G2')} CTRL1={cl.get('CTRL-1')} "
            f"C0r_TRIGGERED={'YES (needs the PI go)' if trig else 'NO'} | {strat}")
    if reh:
        line += f" | REHEARSAL: {reh.get('label')} (no P5-C claim)"
    open(a.out, "w", encoding="utf-8", newline="\n").write(line + "\n")
    print("ZZP5C_VERDICT", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
