"""replay_n1dw.py -- REPLAY of n1dc_ho_read.py: N1DC slots read the N1DW{w} hb0.2 dumps; the 'N1DC - N1D' rows must equal the D1_HO_beside_N1DW-N1D rows of RESULT_N1DW_D1HO_HB.json
(delta exactly, lo/hi to 4 dp, bar to 1e-12), 'N1DC - N1R' = D1_HO_deciding_N1DW-N1R, Z-CHANNEL = Z_USE_HO_real_minus_zero (zero twin in both), R1 cells = R1_LON_LAT_HO_beside_no_bar, and K-RE must pass.
Usage: python -I replay_n1dw.py [--mutant] --out REPLAY_N1DW.json     (reads the result JSON of each weight from --res-dir/out_replay_w{2,4}[_mutant])
A wrapper must reach its post-processing: SystemExit from the reader is caught.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import n1dc_ho_read as R  # noqa: E402


TOL_CONT = 1e-12      # continuous R1 cells (metre / degree means): the dev box and the pod differ in the last ulp (MEASURED: 1e-16); accuracy rows stay EXACT


def eqrow(live, ban, with_bar=True, tol=0.0):
    ok = bool((live["delta"] == ban["delta"] if tol == 0.0 else abs(live["delta"] - ban["delta"]) <= tol) and round(live["lo"], 4) == round(ban["lo"], 4) and round(live["hi"], 4) == round(ban["hi"], 4)
              and (not with_bar or abs(live["bar"] - ban["bar"]) <= 1e-12) and live["separated"] == ban["separated"])
    return {"equal": ok, "tol": tol, "live": [live["delta"], live["lo"], live["hi"]], "banked": [ban["delta"], ban["lo"], ban["hi"]]}


def compare(rep, banked, w):
    hb = rep["executors"]["hb"]
    out = {"weight": w, "rows": {}, "pass": True}
    for s in (0, 1):
        n, b = f"N1DC_s{s}", f"N1DW{w}_s{s}"
        pairs = {f"N1DC-N1D {n}": (hb["D1_HO_deciding_N1DC-N1D"][n], banked["D1_HO_beside_N1DW-N1D"][b], True),
                 f"N1DC-N1R {n}": (hb["D1_HO_beside_N1DC-N1R"][n], banked["D1_HO_deciding_N1DW-N1R"][b], True),
                 f"Z-CHANNEL {n}": (hb["Z_CHANNEL_D_zero_twin_real_minus_zero"][n]["TAC-LAT"], banked["Z_USE_HO_real_minus_zero"][b], False)}
        for ref, rn in (("N1DC-N1R", "N1DW-N1R"), ("N1DC-N1D", "N1DW-N1D")):
            for c in R.CELLS:
                lv, bk = hb["R1_LON_LAT_HO_beside_no_bar"][n][ref][c], banked["R1_LON_LAT_HO_beside_no_bar"][b][rn][c]
                if lv["status"] != "OK" or bk["status"] != "OK":
                    out["rows"][f"R1 {ref} {n} {c}"] = {"equal": lv["status"] == bk["status"], "status": [lv["status"], bk["status"]]}
                else:
                    pairs[f"R1 {ref} {n} {c}"] = (lv, bk, False)
        for k, (lv, bk, wb) in pairs.items():
            out["rows"][k] = eqrow(lv, bk, wb, TOL_CONT if (k.startswith("R1") and not k.endswith(("TAC-LAT", "TAC-LON"))) else 0.0)
    k_re = hb["K-RE"]["pass"] and rep["executors"]["old"]["K-RE"]["pass"]
    out["K-RE_hb_and_old"] = bool(k_re)
    out["controls"] = {"K1": hb["K1_self_pair_N1R_s0"]["pass"], "K-CONST": hb["K-CONST"]["pass"], "K-T0_N1D": hb["K-T0_N1D_rederived"]["pass"]}
    out["n_rows"] = len(out["rows"])
    out["n_unequal"] = sum(1 for v in out["rows"].values() if not v["equal"])
    out["pass"] = bool(out["n_unequal"] == 0 and k_re and all(out["controls"].values()))
    out["reader_status"] = rep["status"]
    return out


def one(a, w):
    if a.recompare:                                                          # re-run only the comparison on the saved reader result
        d = f"{a.res_dir}/out_replay_w{w}" + ("_mutant" if a.mutant else "")
        return compare(json.load(open(d + "/RESULT_N1DC_D1HO.json")), R.load_banked(R.parser().parse_args([])) ["hb"], w)
    args = R.parser().parse_args(["--replay-w", str(w), "--out", f"{a.res_dir}/out_replay_w{w}" + ("_mutant" if a.mutant else "")] + (["--mutant"] if a.mutant else []))
    try:
        rep, rc = R.run(args)
    except SystemExit as e:                                                  # an inner exit must not kill the post-processing
        rep, rc = None, ("SystemExit", str(e))
    if rep is None:
        return {"weight": w, "pass": False, "error": rc}
    banked = R.load_banked(args)["hb"]
    return compare(rep, banked, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutant", action="store_true")
    ap.add_argument("--recompare", action="store_true", help="compare the saved out_replay_w* results again without re-reading")
    ap.add_argument("--weights", nargs="+", type=int, default=[2, 4])
    ap.add_argument("--res-dir", default=HERE)
    ap.add_argument("--out", default=HERE + "/REPLAY_N1DW.json")
    a = ap.parse_args()
    res = {w: one(a, w) for w in a.weights}
    allpass = all(r["pass"] for r in res.values())
    doc = {"what": "REPLAY of n1dc_ho_read.py on the N1DW hb0.2 dumps vs RESULT_N1DW_D1HO_HB.json (sha256 54565e44)", "mutant": a.mutant, "by_weight": {f"w{w}": r for w, r in res.items()},
           "verdict": ("MUTANT-RED (expected: the replay failed)" if (a.mutant and not allpass) else "MUTANT-NOT-DETECTED (BAD)" if a.mutant else "REPLAY-GREEN" if allpass else "REPLAY-RED")}
    json.dump(doc, open(a.out, "w"), indent=1, default=float)
    print(doc["verdict"], {f"w{w}": (r["pass"], r.get("n_unequal"), r.get("n_rows")) for w, r in res.items()})
    return 0 if ((allpass and not a.mutant) or (a.mutant and not allpass)) else 1


if __name__ == "__main__":
    sys.exit(main())
