"""wise_control.py -- Step-3 control.  The WiSE decoder-only blend (DEC file) shares EVERY perception weight with EMA-48,852
(WISEFT_MOD2_BUILD.json: outside_set_is_theta8 = true), so its box slots and its 10 cm map must read IDENTICALLY on the same windows.
Compares the W subset pass (every 16th S2 window) against the EMA pass A on the same (sha12, ws) keys:
  BOX  every per-window pack array of `box3d` and `agent` (logit, xy, cls, matched, ...): bit-equal?  else max |diff|;
  MAP  the W WindowTable rows embed, in order, into A's WindowTable rows with bit-identical statistics (same eid, same inter/union/...).
CPU only.   python wise_control.py --a-dir <A_nosync dir> --w-dir <W_dec_sub dir> --out <json>
"""
import argparse, json, pickle, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-dir", required=True)
    ap.add_argument("--w-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from taniteval import map_hires_metrics as MH
    A = {(r["sha12"], int(r["ws"])): r for r in pickle.load(open(f"{a.a_dir}/perc_det.pkl", "rb"))}
    W = pickle.load(open(f"{a.w_dir}/perc_det.pkl", "rb"))
    out = {"tool": "wise_control.py", "n_w_windows": len(W)}
    miss = [(r["sha12"], r["ws"]) for r in W if (r["sha12"], int(r["ws"])) not in A]
    out["n_w_windows_absent_from_A"] = len(miss)
    keys = ("logit", "xy", "cls", "cls_corr", "matched", "exempt", "pair_err", "pair_size_err", "pair_z_err", "gt_xy", "gt_cls", "pos", "ign", "hidden")
    res = {}
    for head in ("box3d", "agent"):
        n = neq = 0
        mx = {k: 0.0 for k in ("logit", "xy", "pair_err")}
        bad = []
        for r in W:
            ra = A.get((r["sha12"], int(r["ws"])))
            if ra is None or not r.get(head) or not ra.get(head):
                continue
            pw, pa = r[head][0], ra[head][0]
            n += 1
            eq = all(np.array_equal(np.asarray(pw[k]), np.asarray(pa[k])) for k in keys if k in pw)
            if not eq:
                neq += 1
                for k in mx:
                    if k in pw and np.shape(pw[k]) == np.shape(pa[k]) and np.size(pw[k]):
                        mx[k] = max(mx[k], float(np.max(np.abs(np.asarray(pw[k], np.float64) - np.asarray(pa[k], np.float64)))))
                if len(bad) < 3:
                    bad.append([r["sha12"], int(r["ws"])])
        res[head] = {"n_windows_compared": n, "n_windows_bit_equal": n - neq, "n_windows_differ": neq, "max_abs_diff_when_differ": mx,
                     "examples_of_differing_windows": bad}
    out["box"] = res
    # MAP: order-preserving embedding with bit-identical stats
    TW = MH.WindowTable.load(f"{a.w_dir}/perc_map_table.npz")
    TA = MH.WindowTable.load(f"{a.a_dir}/perc_map_table.npz")
    arm = TW.arms[0]
    sw = {k: np.stack(v) for k, v in TW.stats[arm].items()}
    sa = {k: np.stack(v) for k, v in TA.stats[arm].items()}
    j = 0
    matched = 0
    first_fail = None
    for i in range(TW.n_windows):
        while j < TA.n_windows and not (TA.eid[j] == TW.eid[i] and all(np.array_equal(sa[k][j], sw[k][i]) for k in sw)):
            j += 1
        if j >= TA.n_windows:
            first_fail = i
            break
        matched += 1
        j += 1
    out["map"] = {"n_w_rows": TW.n_windows, "n_rows_embedded_bit_identical": matched, "first_unmatched_w_row": first_fail,
                  "pass": bool(matched == TW.n_windows)}
    out["pass"] = bool(out["map"]["pass"] and all(v["n_windows_differ"] == 0 for v in res.values()) and out["n_w_windows_absent_from_A"] == 0)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("pass", "n_w_windows", "map")}), json.dumps({h: (res[h]["n_windows_compared"], res[h]["n_windows_differ"]) for h in res}))


if __name__ == "__main__":
    sys.exit(main())
