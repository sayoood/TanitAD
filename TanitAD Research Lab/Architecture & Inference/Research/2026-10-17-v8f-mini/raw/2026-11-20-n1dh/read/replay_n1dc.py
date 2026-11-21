"""replay_n1dc.py -- REPLAY of n1dh_ho_read.py: the N1DH slots read the N1DC dumps (--replay-n1dc).  Every 'N1DH - N1D' row must equal the banked 'N1DC - N1D' row of RESULT_N1DC_D1HO.json
(delta EXACTLY, lo / hi to 4 dp, bar to 1e-12, separated equal), 'N1DH - N1R' = 'N1DC - N1R', the zero twin / Z-USE rows, the 26 R1 cells (LON / LAT) against N1R and N1D, T0 N1DH - N1D, for BOTH executors,
and K-RE must pass (hb and old).  Beside rows 'N1DH - N1DC' must read exactly 0 (the same dumps).  --mutant shifts N1DH_s0 TAC-LAT by one window: the comparison must go RED.
Usage: python -I replay_n1dc.py [--mutant] [--recompare] --out REPLAY_N1DC.json      A wrapper must reach its post-processing: SystemExit from the reader is caught."""
import argparse
import json
import sys

HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import n1dh_ho_read as R  # noqa: E402

BANKED_RESULT = "F:/TanitAD-scratch/n1dc/read/out_real/RESULT_N1DC_D1HO.json"     # the N1DC real read of record
TOL_CONT = 1e-12     # continuous R1 cells; accuracy rows stay EXACT


def eqrow(live, ban, with_bar=True, tol=0.0):
    ok = bool((live["delta"] == ban["delta"] if tol == 0.0 else abs(live["delta"] - ban["delta"]) <= tol) and round(live["lo"], 4) == round(ban["lo"], 4) and round(live["hi"], 4) == round(ban["hi"], 4)
              and (not with_bar or abs(live["bar"] - ban["bar"]) <= 1e-12) and live["separated"] == ban["separated"])
    return {"equal": ok, "tol": tol, "live": [live["delta"], live["lo"], live["hi"]], "banked": [ban["delta"], ban["lo"], ban["hi"]]}


def compare(rep, banked):
    out = {"rows": {}, "pass": True}
    exs = [e for e in ("hb", "old") if e in rep["executors"]]          # the mutant run reads hb only (the mutation lives there)
    out["executors_compared"] = exs
    for ex in exs:
        L, B = rep["executors"][ex], banked["executors"][ex]
        for s in (0, 1):
            h, c = f"N1DH_s{s}", f"N1DC_s{s}"
            pairs = {f"{ex} N1DH-N1D {h}": (L["D1_HO_deciding_N1DH-N1D"][h], B["D1_HO_deciding_N1DC-N1D"][c], True),
                     f"{ex} N1DH-N1R {h}": (L["D1_HO_beside_N1DH-N1R"][h], B["D1_HO_beside_N1DC-N1R"][c], True),
                     f"{ex} Z-CHANNEL {h}": (L["Z_CHANNEL_D_zero_twin_real_minus_zero"][h]["TAC-LAT"], B["Z_CHANNEL_D_zero_twin_real_minus_zero"][c]["TAC-LAT"], False),
                     f"{ex} Z-USE {h}": (L["Z_USE_skip_zeroed_real_minus_zuse"][h]["TAC-LAT"], B["Z_USE_skip_zeroed_real_minus_zuse"][c]["TAC-LAT"], False),
                     f"{ex} T0 N1DH-N1D s{s}": (L["T0_N1DH_minus_N1D"][f"s{s}"], B["T0_N1DC_minus_N1D"][f"s{s}"], False)}
            for ref in ("N1DH-N1R", "N1DH-N1D"):
                bref = ref.replace("N1DH", "N1DC")
                for cell in R.CELLS:
                    lv, bk = L["R1_LON_LAT_HO_beside_no_bar"][h][ref][cell], B["R1_LON_LAT_HO_beside_no_bar"][c][bref][cell]
                    if lv["status"] != "OK" or bk["status"] != "OK":
                        out["rows"][f"{ex} R1 {ref} {h} {cell}"] = {"equal": lv["status"] == bk["status"], "status": [lv["status"], bk["status"]]}
                    else:
                        pairs[f"{ex} R1 {ref} {h} {cell}"] = (lv, bk, False)
            for k, (lv, bk, wb) in pairs.items():
                out["rows"][k] = eqrow(lv, bk, wb, TOL_CONT if (" R1 " in k and not k.endswith(("TAC-LAT", "TAC-LON"))) else 0.0)
            z = L["D1_HO_beside_N1DH-N1DC"][h]                                  # same dumps: must read exactly 0
            out["rows"][f"{ex} beside N1DH-N1DC {h} exactly 0"] = {"equal": bool(z["delta"] == 0.0 and z["lo"] == 0.0 and z["hi"] == 0.0), "live": [z["delta"], z["lo"], z["hi"]], "banked": [0.0, 0.0, 0.0]}
    out["K-RE_hb_and_old"] = bool(all(rep["executors"][e]["K-RE"]["pass"] for e in exs))
    hb = rep["executors"]["hb"]
    out["controls"] = {"K1": hb["K1_self_pair_N1R_s0"]["pass"], "K-CONST": hb["K-CONST"]["pass"], "K-T0_N1D": hb["K-T0_N1D_rederived"]["pass"]}
    out["bar_equal"] = bool(abs(hb["floor"]["bar"] - banked["executors"]["hb"]["floor"]["bar"]) <= 1e-12)
    out["n_rows"] = len(out["rows"])
    out["n_unequal"] = sum(1 for v in out["rows"].values() if not v["equal"])
    out["unequal_rows"] = [k for k, v in out["rows"].items() if not v["equal"]][:12]
    out["pass"] = bool(out["n_unequal"] == 0 and out["K-RE_hb_and_old"] and all(out["controls"].values()) and out["bar_equal"])
    out["reader_status"] = rep["status"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutant", action="store_true")
    ap.add_argument("--recompare", action="store_true", help="compare the saved result again without re-reading")
    ap.add_argument("--res-dir", default=HERE)
    ap.add_argument("--out", default=HERE + "/REPLAY_N1DC.json")
    ap.add_argument("--reader-args", nargs="*", default=[])
    a = ap.parse_args()
    d = f"{a.res_dir}/out_replay" + ("_mutant" if a.mutant else "")
    banked = json.load(open(BANKED_RESULT, encoding="utf-8"))
    if a.recompare:
        rep = json.load(open(d + "/RESULT_N1DH_D1HO.json"))
        err = None
    else:
        args = R.parser().parse_args(["--replay-n1dc", "--out", d] + (["--mutant", "--execs", "hb"] if a.mutant else []) + a.reader_args)
        try:
            rep, rc = R.run(args)
            err = None
        except SystemExit as e:
            rep, err = None, ("SystemExit", str(e))
    res = compare(rep, banked) if rep is not None else {"pass": False, "error": err}
    doc = {"what": "REPLAY of n1dh_ho_read.py: N1DH slots read the N1DC dumps; rows vs RESULT_N1DC_D1HO.json of record", "banked_result": BANKED_RESULT, "mutant": a.mutant, **res}
    doc["verdict"] = ("MUTANT-RED (expected: the replay failed)" if (a.mutant and not res["pass"]) else "MUTANT-NOT-DETECTED (BAD)" if a.mutant else "REPLAY-GREEN" if res["pass"] else "REPLAY-RED")
    json.dump(doc, open(a.out, "w"), indent=1, default=float)
    print(doc["verdict"], {k: doc.get(k) for k in ("n_rows", "n_unequal", "K-RE_hb_and_old", "bar_equal", "unequal_rows")})
    return 0 if ((res["pass"] and not a.mutant) or (a.mutant and not res["pass"])) else 1


if __name__ == "__main__":
    sys.exit(main())
