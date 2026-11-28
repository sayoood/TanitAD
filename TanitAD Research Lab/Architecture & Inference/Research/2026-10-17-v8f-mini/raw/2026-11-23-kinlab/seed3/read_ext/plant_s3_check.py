"""plant_s3_check.py -- planted check of the SEED3 name (s3_name.py) and of mech_s3.py.  Synthetic inputs only; no rollout, no checkpoint.
Drives s3_name.decide_name through each of the five names, the precedence cases and the -STRADDLE case, then the CLI end to end (RESULT_SEED3.json), then mech_s3.py on synthetic T0 evals.
Usage: python plant_s3_check.py <tmp dir>   -> prints one line per case and 'ALL PASS True|False'; exit 0 only if all pass."""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s3_name as N  # noqa: E402

LEVER_OK = {k: True for k in N.LEVER_KEYS}
CELLS = {"LONGITUDINAL": ["LONGITUDINAL:speed_mae_mps", "LONGITUDINAL:along_mae_m", "HEADWAY:U"], "LATERAL": ["LATERAL:heading_mae_deg", "LATERAL:yaw_rate_mae_degps"], "TACTICAL": ["TACTICAL:ade_tac_m"]}


def cell(v, strad=None):
    d = {"verdict": v, "D": 0.0, "interval": [0.0, 0.0], "F": 0.1, "straddles_floor": bool(strad)}
    d["straddle_sign_verdict"] = strad
    return d


def p1_of(fam_verdicts, straddle=None):
    """fam_verdicts {family: 'BETTER'|'WORSE'|'NOT-SEPARATED'}: every cell of the family reads that; straddle {cell: 'BETTER'|'WORSE'} marks a NOT-SEPARATED cell that straddles its floor (sign-resolved reading given)."""
    straddle = straddle or {}
    cells, fams = {}, {}
    for f, cl in CELLS.items():
        vs = []
        for c in cl:
            v = fam_verdicts[f]
            cells[c] = cell("NOT-SEPARATED" if c in straddle else v, straddle.get(c))
            vs.append(cells[c]["verdict"])
        fams[f] = N.family_of(vs)
    st = []
    for m, arc in enumerate((30.0, 50.0, 80.0)):
        c = cell(fam_verdicts["STRATEGIC"])
        c["arc_m"] = arc
        st.append(c)
    cells["STRATEGIC:pos_tick2"] = st
    fams["STRATEGIC"] = N.family_of([c["verdict"] for c in st])
    return {"A": "RH_R0C10XHb20", "B": "B0_R0C10XHb20", "families": fams, "cells": cells}


NS = {"LONGITUDINAL": "NOT-SEPARATED", "LATERAL": "NOT-SEPARATED", "TACTICAL": "NOT-SEPARATED", "STRATEGIC": "NOT-SEPARATED"}
PROV = {"RH": True, "KI": True}
G_OK, G_BAD = {"point": 0.05, "lo": 0.0, "hi": 0.1174}, {"point": 0.09, "lo": 0.06, "hi": 0.1175}   # the bar is inclusive: hi == 0.1174 holds, 0.1175 fails
CASES = []


def case(label, got, want, extra=True):
    ok = bool(got == want and extra)
    CASES.append(ok)
    print(("PASS " if ok else "FAIL ") + label + " -> " + str(got) + ("" if got == want else "   (wanted " + str(want) + ")"), flush=True)


def main():
    tmp = sys.argv[1]
    os.makedirs(tmp, exist_ok=True)
    better = dict(NS, LATERAL="BETTER", STRATEGIC="BETTER")
    case("REPLICATES: G2 holds at the bar (0.1174 inclusive), two families BETTER, none WORSE", N.decide_name(LEVER_OK, PROV, G_OK, p1_of(better))["name"], "SEED3-REPLICATES")
    case("NO-EFFECT: G2 holds, no family BETTER or WORSE", N.decide_name(LEVER_OK, PROV, G_OK, p1_of(NS))["name"], "SEED3-NO-EFFECT")
    case("HARMS: G2 holds, one family WORSE (and another BETTER)", N.decide_name(LEVER_OK, PROV, G_OK, p1_of(dict(better, TACTICAL="WORSE")))["name"], "SEED3-HARMS")
    case("T0-FAILS: upper bound 0.1175 > 0.1174", N.decide_name(LEVER_OK, PROV, G_BAD, p1_of(better))["name"], "SEED3-T0-FAILS")
    case("BLOCKED: one lever-acts check false", N.decide_name(dict(LEVER_OK, zerograd_pass=False), PROV, G_OK, p1_of(better))["name"], "SEED3-BLOCKED")
    case("BLOCKED: lever-acts record absent", N.decide_name(None, PROV, G_OK, p1_of(better))["name"], "SEED3-BLOCKED")
    case("BLOCKED: provenance (checkpoint not sha-equal) false", N.decide_name(LEVER_OK, {"RH": True, "KI": False}, G_OK, p1_of(better))["name"], "SEED3-BLOCKED")
    case("precedence BLOCKED over T0-FAILS", N.decide_name(dict(LEVER_OK, dagger_rows_sum_gt_0=False), PROV, G_BAD, p1_of(NS))["name"], "SEED3-BLOCKED")
    case("precedence T0-FAILS over HARMS", N.decide_name(LEVER_OK, PROV, G_BAD, p1_of(dict(NS, LATERAL="WORSE")))["name"], "SEED3-T0-FAILS")
    case("precedence HARMS over REPLICATES", N.decide_name(LEVER_OK, PROV, G_OK, p1_of(dict(better, STRATEGIC="WORSE")))["name"], "SEED3-HARMS")
    case("G2 file under another bar -> BLOCKED", N.decide_name(LEVER_OK, PROV, G_OK, p1_of(better), 0.0705)["name"], "SEED3-BLOCKED")
    case("G2 missing -> NOT-DECIDABLE (outside the table, no made-up name)", N.decide_name(LEVER_OK, PROV, None, p1_of(better))["name"], "NOT-DECIDABLE")
    # -STRADDLE: the single LATERAL cell pair straddles; as read the family is NOT-SEPARATED (NO-EFFECT), by sign every LATERAL cell is BETTER -> REPLICATES
    lat = {"LATERAL:heading_mae_deg": "BETTER", "LATERAL:yaw_rate_mae_degps": "BETTER"}
    r = N.decide_name(LEVER_OK, PROV, G_OK, p1_of(NS, lat))
    case("STRADDLE: as read NO-EFFECT, by sign REPLICATES -> name carries -STRADDLE, both printed", r["name"], "SEED3-NO-EFFECT-STRADDLE", r.get("both_names") == ["SEED3-NO-EFFECT", "SEED3-REPLICATES"])
    rw = N.decide_name(LEVER_OK, PROV, G_OK, p1_of(NS, {"TACTICAL:ade_tac_m": "WORSE"}))
    case("STRADDLE: as read NO-EFFECT, by sign HARMS", rw["name"], "SEED3-NO-EFFECT-STRADDLE", rw.get("both_names") == ["SEED3-NO-EFFECT", "SEED3-HARMS"])
    rs = N.decide_name(LEVER_OK, PROV, G_OK, p1_of(better, {"LONGITUDINAL:speed_mae_mps": "BETTER"}))
    case("a straddling cell that does NOT change the name adds no suffix", rs["name"], "SEED3-REPLICATES", "both_names" not in rs)
    rb = N.decide_name(LEVER_OK, PROV, G_BAD, p1_of(NS, lat))
    case("T0-FAILS is never suffixed", rb["name"], "SEED3-T0-FAILS")
    # MIXED family (a cell BETTER, a cell not): neither BETTER nor WORSE; printed
    mixed = p1_of(NS)
    mixed["cells"]["LATERAL:heading_mae_deg"]["verdict"] = "BETTER"
    mixed["families"]["LATERAL"] = N.family_of([mixed["cells"][c]["verdict"] for c in CELLS["LATERAL"]])
    rm = N.decide_name(LEVER_OK, PROV, G_OK, mixed)
    case("MIXED family is not BETTER: NO-EFFECT by the family rule, MIXED printed", rm["name"], "SEED3-NO-EFFECT", rm["mixed_families_printed"] == ["LATERAL"])
    # CLI end to end
    d = os.path.join(tmp, "cli")
    os.makedirs(d, exist_ok=True)
    rh = {"provenance": {"ok": True}, "blocked": {"blocked": False}, "pairs": {"P1": p1_of(better)}, "P": {"P2": {"pass": True}, "P3": {"pass": True}, "P4": {"pass": False}}, "floors_file": {"path": "x", "sha256": "y"}}
    ki = {"provenance": {"ok": True}, "blocked": {"blocked": False}, "pairs": {"P1": p1_of(NS)}}
    g = {"F_T0_m": 0.1174, "per_seed": {"2": dict(G_OK, status="HOLDS", beside_status="FAILS")}}
    files = {"rh.json": rh, "ki.json": ki, "g_rh.json": g, "g_ki.json": g, "lever.json": LEVER_OK}
    for k, v in files.items():
        json.dump(v, open(f"{d}/{k}", "w"))
    out = f"{d}/RESULT_SEED3.json"
    rc = subprocess.run([sys.executable, "-I", HERE + "/s3_name.py", "--rh", f"{d}/rh.json", "--ki", f"{d}/ki.json", "--gt0-rh", f"{d}/g_rh.json", "--gt0-ki", f"{d}/g_ki.json",
                         "--lever", f"{d}/lever.json", "--out", out], capture_output=True, text=True)
    res = json.load(open(out)) if os.path.exists(out) else {}
    case("CLI writes RESULT_SEED3.json with the name and every input", res.get("name"), "SEED3-REPLICATES",
         rc.returncode == 0 and res.get("inputs", {}).get("G2", {}).get("beside_holds") is False and res.get("inputs", {}).get("P1_s2", {}).get("cells") is not None
         and "three_seed_pooled_reads" in res.get("beside_printed_no_bar", {}))
    # mech_s3.py on synthetic evals
    e = os.path.join(tmp, "t0")
    os.makedirs(e, exist_ok=True)
    rng = np.random.default_rng(1)
    n = 2399
    eid = np.repeat(np.arange(40), n // 40 + 1)[:n]
    base = rng.gamma(4, 0.5, n)
    for arm, off in (("B0", 0.0), ("RH", 0.1), ("KI", 0.0)):
        np.savez(f"{e}/eval_{arm}_s2.npz", ade6=base + off + rng.normal(0, 0.01, n), eid=eid)
    env = dict(os.environ, MECH_T0_DIR=e, CUDA_VISIBLE_DEVICES="-1")
    mo = os.path.join(tmp, "MECH4_S3.json")
    rr = subprocess.run([sys.executable, "-I", HERE + "/mech_s3.py", HERE, mo], capture_output=True, text=True, env=env)
    m = json.load(open(mo)) if os.path.exists(mo) else {}
    case("mech_s3: planted KI_s2 - RH_s2 = -0.1 reads below zero (MECH4 s2); RH_s2 - B0_s2 = +0.1 holds 0.1174, fails 0.0705",
         m.get("MECH4_s2", {}).get("below_zero"), True,
         rr.returncode == 0 and m["beside"]["RH_s2_minus_B0_s2"]["upper_le_0p1174"] is True and m["beside"]["RH_s2_minus_B0_s2"]["upper_le_0p0705"] is False and len(m["MECH4_three_seed_list"]["points"]) == 3)
    if rr.returncode != 0:
        print(rr.stderr[-600:])
    print("ALL PASS", all(CASES), f"({sum(CASES)}/{len(CASES)})")
    return 0 if all(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
