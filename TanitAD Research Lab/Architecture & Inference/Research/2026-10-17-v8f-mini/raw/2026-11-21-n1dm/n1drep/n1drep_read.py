"""n1drep_read.py -- MM 2026-11-22, card PREREG_V8F_N1DREP (FROZEN, sha256 0c641197), sections 2-4. CPU, zero training.

Reads the replicate N1DREP_s0 against N1D_s0 on eval-141 under T1-kin / executor hb0.2, with the N1DM read's own machinery (n1dm_read141.py:
geom, cells from 'cl', model-free arrays checked against the C0_s0 reference, 'fl' against the hb0.2 N1D_s0 floor; S.pair_rows with the banked F2).
  1. N1DREP-IDENTICAL test: every array of every episode file of the replicate dump equals the banked N1D_s0 hb0.2 dump.
  2. K-REDUMP: N1D_s0 dumped again in the same session/tree/env must equal the banked dump array for array; if not, the re-dump is the reference.
  3. F_REP per cell = |delta| of the paired row N1DREP_s0 - N1D_s0 (24 cells); outcome on ade / along_mae / speed_mae (operative 0-2 s):
     cell c exceeds when N1DM_s - N1D_s > max(F2_c, F_REP_c) at both seeds (deltas and F2 from RESULT_N1DM_EVAL141.json, unchanged).
Usage: python n1drep_read.py --rep DUMP --redump DUMP --out RESULT.json [--banked DUMP] [--guard RESULT_N1DM_EVAL141.json] [--n-boot 2000]
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

EV = "F:/TanitAD-scratch/n1dc/read_n1dm/eval141"
N2R = "F:/TanitAD-scratch/strat1/navrb2/read"
SR = "F:/TanitAD-scratch/strat1/read"
for _p in (EV, N2R, SR, SR + "/theory/obd/read", SR + "/theory/otband/read"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CARD = "PREREG_V8F_N1DREP.md sha256 0c6411976409dffd713e22ed7def580e42ee3ddb570c8fb4619a507625e9d551"
THREE = ("operative_0_2s:ade_m", "operative_0_2s:along_mae_m", "operative_0_2s:speed_mae_mps")
PAIRS = ("N1DM_s0-N1D_s0", "N1DM_s1-N1D_s1")


def compare_dumps(a, b):
    """every .npz file of dump b exists in a with the same keys and np.array_equal arrays"""
    fa, fb = sorted(f for f in os.listdir(a) if f.endswith(".npz")), sorted(f for f in os.listdir(b) if f.endswith(".npz"))
    out = {"n_files_a": len(fa), "n_files_b": len(fb), "n_arrays": 0, "n_unequal": 0, "unequal": [], "max_abs_diff": 0.0}
    if fa != fb:
        out["file_lists_differ"] = True
        out["unequal"].append("file lists differ")
        out["n_unequal"] += 1
        return out
    for f in fb:
        with np.load(os.path.join(a, f)) as x, np.load(os.path.join(b, f)) as y:
            if sorted(x.files) != sorted(y.files):
                out["n_unequal"] += 1
                out["unequal"].append(f"{f}: keys differ")
                continue
            for k in y.files:
                out["n_arrays"] += 1
                u, v = x[k], y[k]
                if u.shape != v.shape or not np.array_equal(u, v):
                    out["n_unequal"] += 1
                    if len(out["unequal"]) < 20:
                        out["unequal"].append(f"{f}:{k}")
                    if u.shape == v.shape and u.dtype.kind in "fiub":
                        out["max_abs_diff"] = max(out["max_abs_diff"], float(np.nanmax(np.abs(u.astype(np.float64) - v.astype(np.float64)))) if u.size else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", required=True)
    ap.add_argument("--redump", required=True)
    ap.add_argument("--banked", default=EV + "/dumps141_hb02/dump_N1DM-N1D_s0_m10_causal_heads")
    ap.add_argument("--guard", default=EV + "/out/RESULT_N1DM_EVAL141.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--label", default="real", help="'real' for the card's read; anything else marks a code check that is never a result")
    a = ap.parse_args()
    t0 = time.time()
    import navrb2_lib as NL
    import navrb2_read as N2
    import strat1_lib as L
    import strat1_read as S
    import obd_read as OR
    import obd_geom as G
    CI = L.load_ci()
    rep = {"tool": "n1drep_read.py", "card": CARD, "label": a.label, "tier": "T1-kin/hb0.2 (executor of record), eval-141, 141 episodes",
           "evidence_class": "MEASURED (ours; dumps, zero training)" if a.label == "real" else "CODE CHECK (not a result)",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap B={a.n_boot} seed 0 ({CI.SOURCE}); answers 'another draw of EPISODES'; "
                        "the replicate itself answers 'another TRAINING run' at fixed seed, init and code; inference deterministic",
           "inputs": {"rep": a.rep, "redump": a.redump, "banked": a.banked, "guard": a.guard,
                      "guard_sha256": hashlib.sha256(open(a.guard, "rb").read()).hexdigest()},
           "controls": {}, "problems": []}
    prob = rep["problems"]
    ident = compare_dumps(a.rep, a.banked)
    redump = compare_dumps(a.redump, a.banked)
    rep["controls"]["IDENTICAL_test"] = ident
    rep["controls"]["K-REDUMP"] = dict(redump, **{"pass": redump["n_unequal"] == 0 and redump["n_arrays"] > 0})
    identical = ident["n_unequal"] == 0 and ident["n_arrays"] > 0
    geom = G.build()
    bk = lambda x: f"{SR}/t1/dump_STRAT1-{x}_m10_causal_heads"  # noqa: E731
    dg = S.load_dump_n(bk("C0_s0"), 141)
    J = geom.join(dg)
    glat, glon = geom.labels(dg["g"])
    e141 = dg["eid"]
    fl_hb = OR.load_dump(a.banked)["fl"]
    cells, same = {}, {}
    for name, d in (("N1D_s0", a.banked), ("N1DREP_s0", a.rep), ("N1D_s0_redump", a.redump)):
        x = OR.load_dump(d)
        bad = [k for k in OR.SAME if not np.array_equal(x[k], fl_hb if k == "fl" else dg[k])]
        same[name] = {"pass": not bad, "differs": bad}
        if bad:
            prob.append(f"{name}: model-free arrays differ from the reference: {bad}")
        cells[name] = OR.to_cells(geom.arm_arrays(x["cl"], dg["g"], J, dg["eid"]), glat, glon)
    rep["controls"]["K-SAME"] = same
    c0rep = json.load(open(NL.C0REP_RESULT, encoding="utf-8"))
    F2 = N2.floor2(c0rep["floors"]["F"], c0rep["floors"]["F_rep"])
    ref = "N1D_s0" if rep["controls"]["K-REDUMP"]["pass"] else "N1D_s0_redump"
    rep["reference_for_F_REP"] = ref
    k1 = S.pair_rows(CI, cells[ref], cells[ref], e141, F2, 200)
    k1_bad = {c: v.get("delta") for c, v in k1.items() if v.get("delta") != 0.0}
    rep["controls"]["K1_self_pair_exactly_0"] = {"pass": not k1_bad, "nonzero": k1_bad}
    if k1_bad:
        prob.append("K1 self-pair not exactly 0")
    rows = S.slim(S.pair_rows(CI, cells["N1DREP_s0"], cells[ref], e141, F2, a.n_boot))
    rep["N1DREP_s0-N1D_s0"] = rows
    f_rep = {c: abs(float(v["delta"])) for c, v in rows.items() if v.get("delta") is not None}
    rep["F_REP"] = f_rep
    g = json.load(open(a.guard, encoding="utf-8"))["guard_eval141_hb02_N1DM-N1D"]
    beside, exceeds = {}, {}
    for c in sorted(f_rep):
        row = {}
        for p in PAIRS:
            gc = g[p].get(c)
            if not gc or gc.get("delta") is None:
                continue
            fl = max(float(gc["floor"]), f_rep[c])
            row[p] = {"delta": gc["delta"], "F2": gc["floor"], "F_REP": f_rep[c], "floor_max": fl, "exceeds": bool(gc["delta"] > fl)}
        beside[c] = row
        if c in THREE:
            exceeds[c] = len(row) == 2 and all(r["exceeds"] for r in row.values())
    rep["N1DM-N1D_beside_floor_max_F2_F_REP"] = beside
    rep["three_cells_exceed_at_both_seeds"] = exceeds
    n_ex = sum(exceeds.values())
    if prob:
        outcome = "PROBLEMS (no outcome)"
    elif len(exceeds) != 3:
        outcome = "PROBLEMS (a deciding cell is missing)"
    elif identical:
        outcome = "N1DREP-IDENTICAL"
    elif n_ex == 3:
        outcome = "N1DREP-COST-HOLDS"
    elif n_ex == 2:
        outcome = "N1DREP-COST-MIXED"
    else:
        outcome = "N1DREP-COST-IN-NOISE"
    rep["outcome"] = outcome if a.label == "real" else f"CODE CHECK ({outcome})"
    rep["families"] = {"LONGITUDINAL / LATERAL / TACTICAL": "F_REP on all 24 guard cells (operative 0-2 s, tactical 2-6 s, TAC-LAT, TAC-LON)",
                       "STRATEGIC": "not measured by this card: the guard carries no strategic cell on this line (card section 3)"}
    rep["wall_s"] = round(time.time() - t0, 1)
    json.dump(rep, open(a.out, "w", encoding="utf-8"), indent=1)
    print("F_REP three:", {c: round(f_rep.get(c, float("nan")), 5) for c in THREE})
    print("exceeds:", exceeds, "identical:", identical, "K-REDUMP:", rep["controls"]["K-REDUMP"]["pass"], "problems:", prob)
    print("ZZ N1DREP READ", rep["outcome"])
    return 0 if not prob else 1


if __name__ == "__main__":
    sys.exit(main())
