"""k2dc02s1tail_combined.py -- PREREG_K2DC02S1_2026-11-20.md sec. 4 (the floor-admissibility test) and sec. 5 (the combined dose reading), as code. Nothing here sets a constant of any earlier PREREG.

Inputs are two outputs of `r9_e2_dose_verdict.py --arm K2DC02s1` (DOSE_VERDICT.json), differing ONLY in the comparator that sits in K2's slot:
  --dose-dir   K2DC02s1 vs K0 and K2DC   (tail: K2DC02s1t vs K0t and K2DCt;  EMA: K2DC02s1 vs K0 and K2DC)   -> the dose rule of sec. 3
  --floor-dir  K2DC02s1 vs K0 and K2DC02 (tail: ... K2DC02t;                 EMA: ... K2DC02)               -> the training-run FLOOR of sec. 4 (identical flags, so its rows are noise, never a lever)
Per inference seed s, sec. 4: a dose effect is claimable only if |dose K2DC02s1 - K2 LAT_cross_mae_m| > |floor K2DC02s1 - K2DC02 LAT_cross_mae_m| at that seed (strict; a tie does not clear).

--mode tail   sec. 5, fixed:
    DOSE-REPLICATED  the tail outcome is SUPPORTED AND it clears the floor at BOTH inference seeds
    DOSE-ELIMINATED  the tail outcome is REFUTED
    DOSE-UNRESOLVED  anything else, including a SUPPORTED that does not clear the floor
  A read that is VOID decides nothing (sec. 2: DOSE-VOID, TAIL-VOID, REPRO-VOID; sec. 6 item 4): the reading is then `VOID` and names them. DOSE-VOID is read from the dose tool's own outcome
  (an outcome that carries `UNCHECKED-VOID` is a share condition that was not read, and is VOID as well); --tail-void / --repro-void are passed by the caller that ran those checks.
--mode ema    the EMA outcome is REPORTED BESIDE the tail's and is not a sec. 5 reading (sec. 3: the TAIL outcome decides): its reading is `EMA-REPORTED-BESIDE`, with the same floor test written out.
The floor tool outcome (INCONCLUSIVE / REFUTED / ...) is a floor reading, never a verdict about a lever: it is stored under `tool_label_not_a_verdict`.

usage: python k2dc02s1tail_combined.py --mode tail|ema --dose-dir <dir> --floor-dir <dir> --out-dir <dir> [--tail-void] [--repro-void]
EXIT: 0 a sec. 5 reading was made (tail) / the EMA block was written | 5 VOID | 4 REFUSED (unreadable or inconsistent input)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CROSS = "LAT_cross_mae_m"
SEEDS = ("0", "1")
ARM = "K2DC02s1"
FAMS = ("LONGITUDINAL", "LATERAL", "TACTICAL")
KNOWN = ("SUPPORTED", "REFUTED", "INCONCLUSIVE", "DOSE-VOID")


class Refused(Exception):
    pass


def base_outcome(doc: dict) -> str:
    o = str(doc.get("outcome", ""))
    return o.split(" ")[0]


def _cross(doc: dict, s: str) -> float:
    try:
        return float(doc["seeds"][s]["dc_vs_k2"][CROSS]["delta"])
    except (KeyError, TypeError, ValueError) as e:
        raise Refused(f"no seeds.{s}.dc_vs_k2.{CROSS}.delta in the dose document ({e!r})")


def check_arm(doc: dict, what: str) -> None:
    a = (doc.get("arm") or {}).get("arm")
    if a != ARM:
        raise Refused(f"{what}: the document's arm is {a!r}, not {ARM!r} (it was not made by `r9_e2_dose_verdict.py --arm {ARM}`)")


def floor_admissibility(dose: dict, floor: dict) -> dict:
    """sec. 4 per seed; `clears_both` only when every seed clears."""
    out = {}
    for s in SEEDS:
        d, f = _cross(dose, s), _cross(floor, s)
        out[s] = {"dose_delta_m": d, "floor_delta_m": f, "abs_dose_m": abs(d), "abs_floor_m": abs(f), "clears": abs(d) > abs(f)}
    out["clears_both"] = all(out[s]["clears"] for s in SEEDS)
    return out


def combine(dose: dict, floor: dict, mode: str = "tail", tail_void: bool = False, repro_void: bool = False) -> dict:
    check_arm(dose, "dose document")
    check_arm(floor, "floor document")
    fa = floor_admissibility(dose, floor)
    o = base_outcome(dose)
    voids = []
    if o not in KNOWN:
        raise Refused(f"unknown dose outcome {dose.get('outcome')!r}")
    if o == "DOSE-VOID":
        voids.append("DOSE-VOID")
    if "UNCHECKED-VOID" in str(dose.get("outcome", "")):
        voids.append("DOSE-VOID (the in-training share condition was not read)")
    if tail_void:
        voids.append("TAIL-VOID")
    if repro_void:
        voids.append("REPRO-VOID")
    if mode == "ema":
        reading = "VOID" if voids else "EMA-REPORTED-BESIDE"
    elif voids:
        reading = "VOID"
    elif o == "SUPPORTED" and fa["clears_both"]:
        reading = "DOSE-REPLICATED"
    elif o == "REFUTED":
        reading = "DOSE-ELIMINATED"
    else:
        reading = "DOSE-UNRESOLVED"
    why = {"DOSE-REPLICATED": "the outcome is SUPPORTED and it clears the training-run floor at both inference seeds",
           "DOSE-ELIMINATED": "the outcome is REFUTED",
           "DOSE-UNRESOLVED": ("the outcome is SUPPORTED but does NOT clear the floor at both inference seeds" if o == "SUPPORTED" else f"the outcome is {o}"),
           "EMA-REPORTED-BESIDE": f"EMA outcome {o}; the TAIL outcome decides the dose question (PREREG sec. 3)",
           "VOID": "no dose claim is made: " + "; ".join(voids)}[reading]
    return {"mode": mode, "reading": reading, "why": why, "outcome": dose.get("outcome"), "outcome_base": o, "voids": voids,
            "floor_admissibility": fa, "arm": ARM,
            "rule": "PREREG_K2DC02S1_2026-11-20.md sec. 5 (tail) / sec. 4 (floor): REPLICATED = SUPPORTED and clears the floor at both seeds; ELIMINATED = REFUTED; UNRESOLVED = anything else; a VOID read decides nothing"}


def floor_block(floor: dict) -> dict:
    """Every row of the floor, per seed: the training-run noise on this rig. Kept for H-ESTIM-SEED-1."""
    blk = {"note": "FLOOR READING (K2DC02s1 - K2DC02, identical flags). Never a verdict about a lever. Comparator noise is assumed to be of the same size as the candidate's (PREREG sec. 7).",
           "tool_label_not_a_verdict": floor.get("outcome"), "seeds": {}}
    for s in SEEDS:
        n = floor["seeds"][s]
        blk["seeds"][s] = {"dc_vs_k2": n.get("dc_vs_k2"), "dc_vs_k2_by_family": n.get("dc_vs_k2_by_family"), "jitter_dlag": n.get("jitter_dlag_dc_vs_k2")}
    return blk


def _f(r, k):
    v = r.get(k) if isinstance(r, dict) else None
    return "-" if v is None else f"{float(v):+.5f}"


def render(res: dict, floor: dict) -> str:
    L = [f"# K2DC02s1 {res['mode']} reading: **{res['reading']}**", "", res["why"], "", f"Rule: {res['rule']}", "",
         f"Dose-tool outcome (PREREG sec. 3): **{res['outcome']}**. Voids: {', '.join(res['voids']) or 'none'}.", "",
         "## Floor admissibility (sec. 4), LAT_cross_mae_m, metres", "", "| seed | K2DC02s1 - K2 (dose) | K2DC02s1 - K2DC02 (floor) | abs(dose) > abs(floor) |", "|---|---|---|---|"]
    for s in SEEDS:
        a = res["floor_admissibility"][s]
        L.append(f"| {s} | {a['dose_delta_m']:+.5f} | {a['floor_delta_m']:+.5f} | {'yes' if a['clears'] else 'NO'} |")
    L += ["", f"Clears at both seeds: **{'yes' if res['floor_admissibility']['clears_both'] else 'no'}**.", "",
          "## The floor, every row (K2DC02s1 - K2DC02; training-run noise; never a lever verdict)", "",
          f"Tool label for this comparison (not a verdict): `{floor.get('outcome')}`.", ""]
    for s in SEEDS:
        n = floor["seeds"][s]
        L += [f"### seed {s}", "", "| family | row | delta | lo | hi | separated |", "|---|---|---|---|---|---|"]
        for fam in FAMS:
            for rid, r in ((n.get("dc_vs_k2_by_family") or {}).get(fam) or {}).items():
                L.append(f"| {fam} | {rid} | {_f(r, 'delta')} | {_f(r, 'lo')} | {_f(r, 'hi')} | {'*' if r.get('separated') else ''} |")
        L += ["| STRATEGIC | - | PAUSED/see strategic/ | | | |", ""]
    return "\n".join(L) + "\n"


def _load(p: Path) -> dict:
    try:
        return json.loads((p / "DOSE_VERDICT.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refused(f"cannot read {p / 'DOSE_VERDICT.json'}: {e}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("tail", "ema"), required=True)
    ap.add_argument("--dose-dir", required=True)
    ap.add_argument("--floor-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tail-void", action="store_true")
    ap.add_argument("--repro-void", action="store_true")
    a = ap.parse_args(argv)
    try:
        dose, floor = _load(Path(a.dose_dir)), _load(Path(a.floor_dir))
        res = combine(dose, floor, a.mode, a.tail_void, a.repro_void)
    except Refused as e:
        print(f"[k2dc02s1-combined] REFUSED: {e}", file=sys.stderr)
        return 4
    res["floor"] = floor_block(floor)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "COMBINED_READING.json").write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8")
    (out / "COMBINED_READING.md").write_text(render(res, floor), encoding="utf-8")
    print(f"[k2dc02s1-combined] {res['mode']} reading {res['reading']} ({res['why']}) -> {out / 'COMBINED_READING.json'}")
    return 5 if res["reading"] == "VOID" else 0


if __name__ == "__main__":
    sys.exit(main())
