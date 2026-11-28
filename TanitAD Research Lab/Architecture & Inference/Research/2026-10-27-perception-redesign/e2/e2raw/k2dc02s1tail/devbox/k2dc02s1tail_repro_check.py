"""k2dc02s1tail_repro_check.py -- PREREG_K2DC02S1_2026-11-20.md sec. 2 item 6, the REPRODUCTION CHECK. Adapted from k2dc02tail_repro_check.py (K2DC02-TAIL AMENDMENT 1 sec. 2.5).

The comparators of this read are the PINNED, existing K2DC02-tail rolls (K0t, K2DCt); none is re-rolled. One reproduction roll was made in the K2DC02s1 tail environment: K0t at inference seed 0 ONLY.
Two families files, both from `r9_e2_families.py --mode families` at inference seed 0 (A = the pinned K2DCt dump, B = a K0t dump):
  pinned-dir  families_s0.json   A = pinned K2DCt, B = PINNED K0t          ("pinned K2DCt - K0t")
  reroll-dir  families_s0.json   A = pinned K2DCt, B = RE-ROLLED K0t       ("pinned K2DCt - re-rolled K0t")
  GATE   LAT_cross_mae_m: |reroll delta - pinned delta| <= 1 mm (0.001 m) at seed 0. Outside the band, or the row missing => REPRO-VOID: no dose claim is made (PREREG sec. 2 item 6, sec. 6 item 4).
  REPORT every other row of LONGITUDINAL / LATERAL / TACTICAL beside it: pinned, reroll, difference, and whether both agree on 'separated'. Those rows decide nothing.
Both values are READ from the two families files (computed by the same tool on the same A), nothing is typed here.

usage:  python k2dc02s1tail_repro_check.py --pinned-dir <dir> --reroll-dir <dir> [--out <json>]
        python k2dc02s1tail_repro_check.py --selftest <dir holding a families_s0.json to use as both reads>
EXIT: 0 REPRO-PASS | 5 REPRO-VOID | 4 REFUSED (unreadable input).   Selftest EXIT: 0 every control behaved | 3 a control failed."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

GATE_FAMILY, GATE_ROW, BAND = "LATERAL", "LAT_cross_mae_m", 0.001
EPS = 1e-9                      # the band is inclusive; float noise must not flip an exact 1.0 mm
FAMS = ("LONGITUDINAL", "LATERAL", "TACTICAL")
SEEDS = ("0",)                  # the reproduction roll is inference seed 0 only


def _load(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SystemExit(f"[repro] REFUSED: cannot read {p}: {e}")


def family_rows(fam_docs: dict) -> dict:
    """{seed: {row: record}} from families_s{seed}.json documents (rows.<FAMILY>.<row>)."""
    out = {}
    for s, d in fam_docs.items():
        out[s] = {}
        for fam in FAMS:
            for rid, v in ((d.get("rows") or {}).get(fam) or {}).items():
                if isinstance(v, dict) and "delta" in v:
                    out[s][rid] = {**v, "_family": fam}
    return out


def compare(pinned: dict, rer: dict) -> dict:
    gate, table = {}, []
    for s in SEEDS:
        b, r = pinned.get(s, {}), rer.get(s, {})
        g = {"pinned": None, "reroll": None, "diff_m": None, "within_1mm": False, "why": ""}
        if GATE_ROW not in b:
            g["why"] = f"the pinned families_s{s}.json has no {GATE_ROW}"
        elif GATE_ROW not in r:
            g["why"] = f"the re-rolled families_s{s}.json has no {GATE_ROW}"
        else:
            g["pinned"], g["reroll"] = float(b[GATE_ROW]["delta"]), float(r[GATE_ROW]["delta"])
            g["diff_m"] = g["reroll"] - g["pinned"]
            g["within_1mm"] = abs(g["diff_m"]) <= BAND + EPS
            g["why"] = "within the +-1 mm band" if g["within_1mm"] else f"|diff| {abs(g['diff_m']) * 1000:.3f} mm > 1 mm"
        gate[s] = g
        for rid in sorted(set(b) | set(r), key=lambda x: (b.get(x) or r.get(x))["_family"] + x):
            br, rr = b.get(rid), r.get(rid)
            row = {"seed": s, "family": (br or rr)["_family"], "row": rid, "pinned": None if br is None else float(br["delta"]),
                   "reroll": None if rr is None else float(rr["delta"])}
            row["diff"] = None if br is None or rr is None else row["reroll"] - row["pinned"]
            row["separated_agree"] = None if br is None or rr is None else bool(br.get("separated")) == bool(rr.get("separated"))
            row["n"] = [None if br is None else br.get("n"), None if rr is None else rr.get("n")]
            table.append(row)
    ok = all(g["within_1mm"] for g in gate.values())
    return {"verdict": "REPRO-PASS" if ok else "REPRO-VOID", "gate": gate, "rows": table,
            "rule": f"PREREG_K2DC02S1 sec. 2 item 6: {GATE_FAMILY}.{GATE_ROW}, pinned K2DCt - re-rolled K0t vs pinned K2DCt - K0t, within +-{BAND * 1000:.0f} mm at inference seed 0, else REPRO-VOID"}


def render(res: dict) -> str:
    L = [f"K2DC02S1TAIL-REPRO {res['verdict']}", res["rule"], ""]
    for s, g in res["gate"].items():
        L.append(f"  GATE seed {s}: {GATE_ROW} pinned {g['pinned']!r} re-roll {g['reroll']!r} diff {g['diff_m']!r} -> {'OK' if g['within_1mm'] else 'OUT'} ({g['why']})")
    L += ["", "every row (decides nothing but the gate row):", "| seed | family | row | pinned | re-roll | diff | separated agree | n pinned/re-roll |", "|---|---|---|---|---|---|---|---|"]
    f = lambda x: "-" if x is None else f"{x:+.6f}"   # noqa: E731
    for r in res["rows"]:
        L.append(f"| {r['seed']} | {r['family']} | {r['row']} | {f(r['pinned'])} | {f(r['reroll'])} | {f(r['diff'])} | {r['separated_agree']} | {r['n'][0]}/{r['n'][1]} |")
    nr = sum(1 for r in res["rows"] if r["diff"] is None)
    L.append(f"rows present in only one read: {nr}")
    return "\n".join(L)


def run(pinned_dir, reroll_dir):
    pin = {s: _load(Path(pinned_dir) / f"families_s{s}.json") for s in SEEDS}
    rer = {s: _load(Path(reroll_dir) / f"families_s{s}.json") for s in SEEDS}
    return compare(family_rows(pin), family_rows(rer))


def selftest(read_dir) -> int:
    """Controls on one real families_s0.json used as both reads."""
    docs = {s: _load(Path(read_dir) / f"families_s{s}.json") for s in SEEDS}
    bad = []

    def shifted(delta):
        d = copy.deepcopy(docs)
        d["0"]["rows"][GATE_FAMILY][GATE_ROW]["delta"] += delta
        return d

    base = compare(family_rows(docs), family_rows(docs))
    diffs = [r["diff"] for r in base["rows"] if r["diff"] is not None]
    print(f"[selftest] identity: verdict {base['verdict']}, {len(diffs)} rows compared")
    if base["verdict"] != "REPRO-PASS" or not diffs or any(x != 0.0 for x in diffs):
        bad.append("identity control must PASS with every delta exactly 0.0")
    for dl, want in ((0.002, "REPRO-VOID"), (-0.002, "REPRO-VOID"), (0.0009, "REPRO-PASS"), (-0.0009, "REPRO-PASS"), (0.0011, "REPRO-VOID")):
        v = compare(family_rows(docs), family_rows(shifted(dl)))["verdict"]
        print(f"[selftest] {GATE_ROW} shifted {dl * 1000:+.1f} mm: {v} (expected {want})")
        if v != want:
            bad.append(f"shift {dl}: {v} != {want}")
    d = copy.deepcopy(docs)
    del d["0"]["rows"][GATE_FAMILY][GATE_ROW]
    v = compare(family_rows(docs), family_rows(d))["verdict"]
    print(f"[selftest] gate row removed: {v} (expected REPRO-VOID)")
    if v != "REPRO-VOID":
        bad.append("a missing gate row did not void the read")
    v = compare(family_rows(docs), family_rows({}))["verdict"]
    print(f"[selftest] seed 0 file absent: {v} (expected REPRO-VOID)")
    if v != "REPRO-VOID":
        bad.append("a missing seed did not void the read")
    print("[selftest] " + ("ALL CONTROLS BEHAVED" if not bad else "FAILED: " + "; ".join(bad)))
    return 0 if not bad else 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pinned-dir")
    ap.add_argument("--reroll-dir")
    ap.add_argument("--out")
    ap.add_argument("--selftest")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest(a.selftest)
    if not (a.pinned_dir and a.reroll_dir):
        raise SystemExit("[repro] REFUSED: need --pinned-dir and --reroll-dir (or --selftest)")
    res = run(a.pinned_dir, a.reroll_dir)
    print(render(res))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    return 0 if res["verdict"] == "REPRO-PASS" else 5


if __name__ == "__main__":
    sys.exit(main())
