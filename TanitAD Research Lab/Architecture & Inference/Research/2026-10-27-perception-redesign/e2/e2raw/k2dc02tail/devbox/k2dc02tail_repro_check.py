"""k2dc02tail_repro_check.py -- PREREG_K2DC02TAIL AMENDMENT 1 sec. 2.5, the REPRODUCTION CHECK (MM 4a3d05 build agent, 2026-11-20).

The comparator planner rolls were RE-ROLLED (cmp/E2_K0_pf = K0t, cmp_k2dc/E2_K2_pf = K2DCt). Families on that pair (K2DCt - K0t, r9_e2_families.py, one file per inference seed)
must reproduce E2-TAIL's banked K2DCt - K0t read:
  GATE   LAT_cross_mae_m: |re-roll delta - banked delta| <= 1 mm (0.001 m) at EACH seed. Outside the band, or a row / seed missing => TAIL-VOID (PREREG sec. 5.5): no dose claim is made.
  REPORT every other row of LONGITUDINAL / LATERAL / TACTICAL beside it: banked, re-roll, difference, and whether both reads agree on 'separated'. Those rows decide nothing.
The banked values are READ from the banked E2_VERDICT.json (four_families[seed][family].rows[row].delta); nothing is typed here.

usage:  python k2dc02tail_repro_check.py --reroll-dir <dir holding families_s0.json, families_s1.json> --banked-verdict <e2tail stageB/read/E2_VERDICT.json> [--out <json>]
        python k2dc02tail_repro_check.py --selftest <e2tail stageB/read dir holding E2_VERDICT.json + families_s0/1.json>
EXIT: 0 REPRO-PASS | 5 TAIL-VOID | 4 REFUSED (unreadable input).   Selftest EXIT: 0 every control behaved | 3 a control failed."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

GATE_FAMILY, GATE_ROW, BAND = "LATERAL", "LAT_cross_mae_m", 0.001
EPS = 1e-9                      # the band is inclusive; float noise must not flip an exact 1.0 mm
FAMS = ("LONGITUDINAL", "LATERAL", "TACTICAL")
SEEDS = ("0", "1")


def _load(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SystemExit(f"[repro] REFUSED: cannot read {p}: {e}")


def banked_rows(verdict: dict) -> dict:
    """{seed: {row: record}} from a banked E2_VERDICT.json."""
    out = {}
    for s, fams in verdict["four_families"].items():
        out[s] = {}
        for fam in FAMS:
            b = fams.get(fam)
            if isinstance(b, dict) and isinstance(b.get("rows"), dict):
                for rid, v in b["rows"].items():
                    if isinstance(v, dict) and "delta" in v:
                        out[s][rid] = {**v, "_family": fam}
    return out


def reroll_rows(fam_docs: dict) -> dict:
    """{seed: {row: record}} from families_s{seed}.json documents (rows.<FAMILY>.<row>)."""
    out = {}
    for s, d in fam_docs.items():
        out[s] = {}
        for fam in FAMS:
            for rid, v in ((d.get("rows") or {}).get(fam) or {}).items():
                if isinstance(v, dict) and "delta" in v:
                    out[s][rid] = {**v, "_family": fam}
    return out


def compare(banked: dict, rer: dict) -> dict:
    gate, table = {}, []
    for s in SEEDS:
        b, r = banked.get(s, {}), rer.get(s, {})
        g = {"banked": None, "reroll": None, "diff_m": None, "within_1mm": False, "why": ""}
        if GATE_ROW not in b:
            g["why"] = f"banked E2_VERDICT.json has no {GATE_ROW} at seed {s}"
        elif GATE_ROW not in r:
            g["why"] = f"re-rolled families_s{s}.json has no {GATE_ROW}"
        else:
            g["banked"], g["reroll"] = float(b[GATE_ROW]["delta"]), float(r[GATE_ROW]["delta"])
            g["diff_m"] = g["reroll"] - g["banked"]
            g["within_1mm"] = abs(g["diff_m"]) <= BAND + EPS
            g["why"] = "within the +-1 mm band" if g["within_1mm"] else f"|diff| {abs(g['diff_m']) * 1000:.3f} mm > 1 mm"
        gate[s] = g
        for rid in sorted(set(b) | set(r), key=lambda x: (b.get(x) or r.get(x))["_family"] + x):
            br, rr = b.get(rid), r.get(rid)
            row = {"seed": s, "family": (br or rr)["_family"], "row": rid, "banked": None if br is None else float(br["delta"]),
                   "reroll": None if rr is None else float(rr["delta"])}
            row["diff"] = None if br is None or rr is None else row["reroll"] - row["banked"]
            row["separated_agree"] = None if br is None or rr is None else bool(br.get("separated")) == bool(rr.get("separated"))
            row["n"] = [None if br is None else br.get("n"), None if rr is None else rr.get("n")]
            table.append(row)
    ok = all(g["within_1mm"] for g in gate.values())
    return {"verdict": "REPRO-PASS" if ok else "TAIL-VOID", "gate": gate, "rows": table,
            "rule": f"AMENDMENT1 sec. 2.5: {GATE_FAMILY}.{GATE_ROW} K2DCt - K0t re-roll vs banked within +-{BAND * 1000:.0f} mm at each seed, else TAIL-VOID"}


def render(res: dict) -> str:
    L = [f"K2DC02TAIL-REPRO {res['verdict']}", res["rule"], ""]
    for s, g in res["gate"].items():
        L.append(f"  GATE seed {s}: {GATE_ROW} banked {g['banked']!r} re-roll {g['reroll']!r} diff {g['diff_m']!r} -> {'OK' if g['within_1mm'] else 'OUT'} ({g['why']})")
    L += ["", "every row (decides nothing but the gate row):", "| seed | family | row | banked | re-roll | diff | separated agree | n banked/re-roll |", "|---|---|---|---|---|---|---|---|"]
    f = lambda x: "-" if x is None else f"{x:+.6f}"   # noqa: E731
    for r in res["rows"]:
        L.append(f"| {r['seed']} | {r['family']} | {r['row']} | {f(r['banked'])} | {f(r['reroll'])} | {f(r['diff'])} | {r['separated_agree']} | {r['n'][0]}/{r['n'][1]} |")
    nr = sum(1 for r in res["rows"] if r["diff"] is None)
    L.append(f"rows present in only one read: {nr}")
    return "\n".join(L)


def run(reroll_dir, verdict_path):
    docs = {s: _load(Path(reroll_dir) / f"families_s{s}.json") for s in SEEDS}
    return compare(banked_rows(_load(verdict_path)), reroll_rows(docs))


def selftest(read_dir) -> int:
    """Controls on E2-TAIL's OWN banked files (the banked families as both 're-roll' and 'banked')."""
    rd = Path(read_dir)
    verdict = _load(rd / "E2_VERDICT.json")
    docs = {s: _load(rd / f"families_s{s}.json") for s in SEEDS}
    bad = []

    def shifted(seed, delta):
        d = copy.deepcopy(docs)
        d[seed]["rows"][GATE_FAMILY][GATE_ROW]["delta"] += delta
        return d

    base = compare(banked_rows(verdict), reroll_rows(docs))
    diffs = [r["diff"] for r in base["rows"] if r["diff"] is not None]
    print(f"[selftest] identity: verdict {base['verdict']}, {len(diffs)} rows compared, max |diff| {max(abs(x) for x in diffs) if diffs else None!r}, rows in one read only {sum(1 for r in base['rows'] if r['diff'] is None)}")
    if base["verdict"] != "REPRO-PASS" or not diffs or any(x != 0.0 for x in diffs):
        bad.append("identity control must PASS with every delta exactly 0.0")
    if any(g["reroll"] != g["banked"] for g in base["gate"].values()):
        bad.append("identity gate values differ")
    for seed in SEEDS:
        for dl, want in ((0.002, "TAIL-VOID"), (-0.002, "TAIL-VOID"), (0.0009, "REPRO-PASS"), (-0.0009, "REPRO-PASS"), (0.0011, "TAIL-VOID")):
            v = compare(banked_rows(verdict), reroll_rows(shifted(seed, dl)))["verdict"]
            print(f"[selftest] {GATE_ROW} seed {seed} shifted {dl * 1000:+.1f} mm: {v} (expected {want})")
            if v != want:
                bad.append(f"seed {seed} shift {dl}: {v} != {want}")
    # a missing gate row, and a missing seed file, must be TAIL-VOID (never a silent pass)
    d = copy.deepcopy(docs)
    del d["1"]["rows"][GATE_FAMILY][GATE_ROW]
    v = compare(banked_rows(verdict), reroll_rows(d))["verdict"]
    print(f"[selftest] gate row removed at seed 1: {v} (expected TAIL-VOID)")
    if v != "TAIL-VOID":
        bad.append("a missing gate row did not void the read")
    v = compare(banked_rows(verdict), reroll_rows({"0": docs["0"]}))["verdict"]
    print(f"[selftest] seed 1 file absent: {v} (expected TAIL-VOID)")
    if v != "TAIL-VOID":
        bad.append("a missing seed did not void the read")
    print("[selftest] " + ("ALL CONTROLS BEHAVED" if not bad else "FAILED: " + "; ".join(bad)))
    return 0 if not bad else 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reroll-dir")
    ap.add_argument("--banked-verdict")
    ap.add_argument("--out")
    ap.add_argument("--selftest")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest(a.selftest)
    if not (a.reroll_dir and a.banked_verdict):
        raise SystemExit("[repro] REFUSED: need --reroll-dir and --banked-verdict (or --selftest)")
    res = run(a.reroll_dir, a.banked_verdict)
    print(render(res))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    return 0 if res["verdict"] == "REPRO-PASS" else 5


if __name__ == "__main__":
    sys.exit(main())
