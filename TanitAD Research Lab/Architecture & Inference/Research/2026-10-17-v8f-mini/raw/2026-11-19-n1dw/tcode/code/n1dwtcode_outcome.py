"""n1dwtcode_outcome.py -- the committed outcome logic of PREREG_N1D_TCODE sec 4 (sha256 4cec3917...). Pure functions.
Unit = one N1D seed (tree probe decides; the logistic probe is evaluated by the same rules and printed beside). A contrast is {delta, lo, hi, separated} of A - B (paired episode-cluster bootstrap).
  CODE-CARRIES        F-CODE   - F-TEXT separated above 0 by at least BAR (delta >= BAR and lo > 0)
  T-HAS-CODE-LACKS    F-TTRUNK - F-TEXT separated above 0 by at least BAR, and NOT CODE-CARRIES
  T-LACKS             F-TTRUNK - F-TEXT NOT separated above 0 by at least BAR
  NEITHER             headline only: none of the three holds at both seeds
A label HOLDS if it holds at every unit (both N1D seeds); holding at only some = ONE-SEED (printed per seed); nowhere = NOT-HELD."""
BAR = 0.0383
LABELS = ("CODE-CARRIES", "T-HAS-CODE-LACKS", "T-LACKS")
NEXT_LEVER = {
    "CODE-CARRIES": "T hands O the decision and O's decoder does not turn it into the track. The lever is O-side: N1DW (in build); beyond it, a direct code-to-decoder path (the code conditioning head_o, zero-init)",
    "T-HAS-CODE-LACKS": "T forms the decision but its vocabulary does not pass it on. The lever is tactical: a roundabout-exit head in T whose softmax is appended to the code (the PI's tactical band, up to 6 s plus vocabulary)",
    "T-LACKS": "the decision forms only in O. The lever is the T0T1 reading's: an auxiliary route-decision head on O's plan feature",
    "NEITHER": "printed with every number; the lever is chosen with the coordinator",
}
NEED = ("CODE-TEXT", "TTRUNK-TEXT")


def above(c, bar=BAR):
    return bool(c["delta"] >= bar and c["lo"] > 0)


def unit_labels(con, bar=BAR):
    if any(con.get(k) is None for k in NEED):
        return None
    code_up, trunk_up = above(con["CODE-TEXT"], bar), above(con["TTRUNK-TEXT"], bar)
    out = []
    if code_up:
        out.append("CODE-CARRIES")
    if trunk_up and not code_up:
        out.append("T-HAS-CODE-LACKS")
    if not trunk_up:
        out.append("T-LACKS")
    return out


def combine(units, bar=BAR):
    per_unit = {u: unit_labels(c, bar) for u, c in units.items()}
    res = {}
    for lab in LABELS:
        hold = [u for u, ls in per_unit.items() if ls is not None and lab in ls]
        no = [u for u, ls in per_unit.items() if ls is not None and lab not in ls]
        inc = [u for u, ls in per_unit.items() if ls is None]
        st = "INCOMPLETE" if inc else ("HOLDS" if hold and not no else ("ONE-SEED" if hold else "NOT-HELD"))
        res[lab] = {"status": st, "units_holding": hold, "units_not": no, "units_incomplete": inc}
    return {"labels": res, "per_unit": per_unit}


def headline(comb):
    holds = [l for l in LABELS if comb["labels"][l]["status"] == "HOLDS"]
    one = [l for l in LABELS if comb["labels"][l]["status"] == "ONE-SEED"]
    inc = any(comb["labels"][l]["status"] == "INCOMPLETE" for l in LABELS)
    outcome = "INCOMPLETE" if inc else (holds[0] if len(holds) == 1 else (" + ".join(holds) if holds else "NEITHER"))
    return {"outcome": outcome, "holds": holds, "one_seed": one, "next_lever": [NEXT_LEVER[l] for l in holds] or ([NEXT_LEVER["NEITHER"]] if outcome == "NEITHER" else [])}
