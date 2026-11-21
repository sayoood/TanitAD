"""n1dw_ho_read_hb.py -- the D1-HO read of the N1DW card, unchanged, on the dumps made through the executor of record (T1-kin/hb0.2);
card PREREG_N1DW_HB_OVERLAP.md. It imports the banked reader /root/strat1/n1dw/read/code/n1dw_ho_read.py as a module and changes three things only:
  1. dpath -> the hb dumps (/root/strat1/n1dw/hb/out/hb0.2/<family>/t1[_zero]/<the same dump directory names>);
  2. kclip -> not re-read (a training-log property, executor-independent; the T1-kin/old read holds it);
  3. OUT -> /root/strat1/n1dw/hb/read.
Then it writes RESULT_N1DW_D1HO_HB.json: the reader's output, the executor stamp, K-HB0, K-RE marked as expected to differ (the banked
RESULT_NAVRBD.json was read under the old executor; the equality the executor change can keep is checked by K-HB0), the status recomputed
without the K-RE entry, and the deciding and beside rows side by side with the banked T1-kin/old read."""
import importlib.util
import json
import os
import sys

HB = "/root/strat1/n1dw/hb"
os.environ["N1DW_READ_OUT"] = HB + "/read"
os.environ.setdefault("N1DW_BANKED_RESULT", "/root/strat1/n1dw/read/code/RESULT_NAVRBD.json")
SRC = "/root/strat1/n1dw/read/code/n1dw_ho_read.py"
OLD = "/root/strat1/n1dw/read/RESULT_N1DW_D1HO.json"
spec = importlib.util.spec_from_file_location("n1dw_ho_read", SRC)
HO = importlib.util.module_from_spec(spec)
sys.modules["n1dw_ho_read"] = HO
spec.loader.exec_module(HO)

FAM_OF = lambda base: "n1r" if base.startswith("N1R") else ("n1dw" if base.startswith("N1DW") else "n1d")


def dpath_hb(arm):
    zero = arm.endswith("Z")
    base = arm[:-1] if zero else arm
    fam = FAM_OF(base)
    sub = "t1_zero" if zero else "t1"
    pre = "N1DW-HO-" if fam == "n1dw" else "NAVRBD-HO-"
    return f"{HB}/out/hb0.2/{fam}/{sub}/dump_{pre}{base}{'_zero' if zero else ''}_m10_causal_heads"


HO.dpath = dpath_hb
HO.kclip = lambda path: {"status": "not re-read under T1-kin/hb0.2 (training-log property, executor-independent); see the T1-kin/old read", "path": path}


def main():
    HO.main()
    rp = HO.OUT + "/RESULT_N1DW_D1HO.json"
    rep = json.load(open(rp, encoding="utf-8"))
    rep["executor"] = "T1-kin/hb0.2 (stack/scripts/v8f_t1_eval_hb.py --kappa-max 0.2, landed bcf267ce; PI ruling D16-D 2026-11-20)"
    rep["card_overlap"] = "PREREG_N1DW_HB_OVERLAP.md"
    kh = json.load(open(HB + "/K-HB0.json", encoding="utf-8"))
    rep.setdefault("controls", {})["K-HB0"] = kh
    if "K-RE" in rep.get("controls", {}):
        rep["controls"]["K-RE"]["note"] = "EXPECTED TO DIFFER under T1-kin/hb0.2: RESULT_NAVRBD.json was read under T1-kin/old; K-HB0 checks the equality the executor change keeps"
    probs = [p for p in rep.get("problems", []) if not p.startswith("K-RE")]
    if not kh.get("pass"):
        probs.append("K-HB0 failed")
    rep["problems_without_K-RE"] = probs
    rep["status_hb"] = "VALID" if not probs else "INVALID (" + "; ".join(probs) + ")"
    rep.pop("K-RE_FAILED_STOP", None)
    old = json.load(open(OLD, encoding="utf-8"))
    side = {}
    for key in ("D1_HO_deciding_N1DW-N1R", "D1_HO_beside_N1DW-N1D"):
        side[key] = {arm: {"T1-kin/old": {k: old[key][arm].get(k) for k in ("delta", "lo", "hi", "separated", "gain", "bar")},
                           "T1-kin/hb0.2": {k: rep[key][arm].get(k) for k in ("delta", "lo", "hi", "separated", "gain", "bar")}} for arm in rep[key]}
    side["D1_HO_each_acc"] = {a: {"T1-kin/old": old["D1_HO_each_acc"].get(a), "T1-kin/hb0.2": v} for a, v in rep["D1_HO_each_acc"].items()}
    side["outcome_labels"] = {"T1-kin/old": old["outcome_D1HO_alone"]["labels"], "T1-kin/hb0.2": rep["outcome_D1HO_alone"]["labels"]}
    rep["overlap_old_vs_hb"] = side
    json.dump(rep, open(HO.OUT + "/RESULT_N1DW_D1HO_HB.json", "w"), indent=1, default=float)
    os.replace(rp, HO.OUT + "/RESULT_N1DW_D1HO_reader_raw_hb.json")
    print("hb read", rep["status_hb"])


if __name__ == "__main__":
    main()
