"""n1dwprobe_outcome.py -- the committed outcome logic of PROBE-N1D-FEAT (PREREG_V8F_N1DW.md sec 10, sha256 59f2b7f9...). Pure functions, no I/O, no numpy needed.

A *contrast* is a dict {delta, lo, hi, separated} from the paired episode-cluster bootstrap on per-window out-of-fold correctness (A - B on the same windows).
BAR = 0.0383 (the rig floor of the roundabout cells), card text: "Bar 0.0383".

Per UNIT = (N1D seed, probe seed) the four label conditions, tree probe, exactly as the card's table commits them:
  PROBE-LOST-AT-TOKEN      F-NAV  - F-TEXT  separated BELOW 0 by at least BAR            (delta <= -BAR and hi < 0)
  PROBE-LOST-AFTER-TOKEN   NOT(LOST-AT-TOKEN) and  F-PLAN - F-NAV separated below 0 by at least BAR
  PROBE-PRESENT-UNUSED     F-PLAN NOT separated below F-TEXT by BAR                     (NOT (delta <= -BAR and hi < 0) for F-PLAN - F-TEXT)
  PROBE-VISION             F-PLAN+TEXT - F-PLAN, OR F-VIS - F-TEXT, separated ABOVE 0 by at least BAR   (delta >= BAR and lo > 0)
More than one label can hold; all are printed. The lever order follows the table's rows.
Across units: a label HOLDS only if its condition holds in EVERY unit (both N1D seeds x both probe seeds); a label that holds in some units but not all is printed per unit and carries
"one-seed" (the card's wording, whichever axis the split is on); a label that holds nowhere is NOT-HELD. A missing contrast (None) in a unit makes that unit INCOMPLETE and no label
is claimed for it (it never counts as a hold)."""
from __future__ import annotations

BAR = 0.0383
LABELS = ("PROBE-LOST-AT-TOKEN", "PROBE-LOST-AFTER-TOKEN", "PROBE-PRESENT-UNUSED", "PROBE-VISION")
NEXT_LEVER = {
    "PROBE-LOST-AT-TOKEN": "the token path: the channel enters O too weakly. The --nav-rb-lr-mult dose (30) or a 2-layer e_nav_rb (PI go)",
    "PROBE-LOST-AFTER-TOKEN": "the information reaches the token but not the plan head. A route-decision auxiliary head on O's plan feature (trained on the kinematic label; labels may use ego), PI go",
    "PROBE-PRESENT-UNUSED": "the decoder / rollout: a T0-against-T1 read of N1D on D1-HO (zero training) first, then the planner-side lever it points to",
    "PROBE-VISION": "vision carries roundabout information the plan path lacks. Only this label un-parks N1D8 (interpolated position-table init, PI go)",
}
# contrast keys the logic needs (each "A-B" = acc(A) - acc(B))
NEED = ("NAV-TEXT", "PLAN-NAV", "PLAN-TEXT", "PLANTEXT-PLAN", "VIS-TEXT")


def below(c, bar=BAR):
    """separated below 0 by at least bar"""
    return bool(c["delta"] <= -bar and c["hi"] < 0)


def above(c, bar=BAR):
    """separated above 0 by at least bar"""
    return bool(c["delta"] >= bar and c["lo"] > 0)


def unit_labels(con, bar=BAR):
    """con: dict contrast-name -> contrast dict. Returns the list of labels holding in this unit, or None when a needed contrast is missing."""
    if any(con.get(k) is None for k in NEED):
        return None
    out = []
    at_token = below(con["NAV-TEXT"], bar)
    if at_token:
        out.append("PROBE-LOST-AT-TOKEN")
    if (not at_token) and below(con["PLAN-NAV"], bar):
        out.append("PROBE-LOST-AFTER-TOKEN")
    if not below(con["PLAN-TEXT"], bar):
        out.append("PROBE-PRESENT-UNUSED")
    if above(con["PLANTEXT-PLAN"], bar) or above(con["VIS-TEXT"], bar):
        out.append("PROBE-VISION")
    return out


def combine(units, bar=BAR):
    """units: dict unit_name -> {contrast dict}. Returns {label: {"status": HOLDS|ONE-SEED|NOT-HELD|INCOMPLETE, "units_holding": [...], "units_not": [...], "units_incomplete": [...]}} plus per-unit label lists."""
    per_unit = {u: unit_labels(c, bar) for u, c in units.items()}
    res = {}
    for lab in LABELS:
        hold = [u for u, ls in per_unit.items() if ls is not None and lab in ls]
        no = [u for u, ls in per_unit.items() if ls is not None and lab not in ls]
        inc = [u for u, ls in per_unit.items() if ls is None]
        if inc:
            st = "INCOMPLETE"
        elif hold and not no:
            st = "HOLDS"
        elif hold:
            st = "ONE-SEED"
        else:
            st = "NOT-HELD"
        res[lab] = {"status": st, "units_holding": hold, "units_not": no, "units_incomplete": inc}
    return {"labels": res, "per_unit": per_unit}


def headline(comb):
    """the labels that HOLD at every unit, in the card's row order; 'ONE-SEED' labels listed apart"""
    holds = [l for l in LABELS if comb["labels"][l]["status"] == "HOLDS"]
    one = [l for l in LABELS if comb["labels"][l]["status"] == "ONE-SEED"]
    return {"holds": holds, "one_seed": one, "next_lever_after_RBW_NO_GAIN": [NEXT_LEVER[l] for l in holds]}
