#!/usr/bin/env python3
"""The identity controls of ADDENDUM_1 (IC1-IC6, D1-D3) that need the harness CSVs, written as files `fp_read.py` reads.  CPU only.

    python fp_checks.py expected-states            -> EXPECTED_STATES.json (sha256 of the C0A / R0B / decoy tensor sets, computed from the banked files BEFORE any run)
    python fp_checks.py all --work W               -> W/IC3_IC4.json, W/IC5.json, W/IC6.json, W/D1_D3.json and W/gates_summary.json (status VALID / IDENTITY-FAILED / INVALID per ADDENDUM_1)
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PK = "C:/fp/PK"
DATA = "E:/Projects/TanitAD/data/refe_navtest"
C3_TABLE = f"{PK}/raw/2026-10-11-c3-selection-headroom/table.npz"
STATES = {"C0A": ("C:/Users/Admin/rls1_adam_dl/rls1-c0/scorer_states/scorer_states_u1000.pt", "A"),
          "R0B": ("C:/Users/Admin/rls1_adam_dl/rls1-r0/scorer_states/scorer_states_u1000.pt", "B")}
DECOYS = {"C0B(decoy)": ("C:/Users/Admin/rls1_adam_dl/rls1-c0/scorer_states/scorer_states_u1000.pt", "B"),
          "R0A(decoy)": ("C:/Users/Admin/rls1_adam_dl/rls1-r0/scorer_states/scorer_states_u1000.pt", "A")}
SUBS = ("NC", "DAC", "EP", "TTC", "C", "DDC")
COLS = ("PDMS",) + SUBS
EXPECTED = os.path.join(os.path.dirname(HERE), "EXPECTED_STATES.json")


def cmd_expected(a):
    import torch
    import fp_forward as FF
    out = {}
    for nm, (path, key) in {**STATES, **DECOYS}.items():
        ss = torch.load(path, map_location="cpu", weights_only=False)
        out[nm] = {"file": os.path.basename(os.path.dirname(os.path.dirname(path))) + "/scorer_states_u1000.pt", "key": key, "update": int(ss["update"]), "sha256_tensors": FF.sha256_tensors(ss[key])}
    json.dump(out, open(EXPECTED, "w"), indent=1)
    print(json.dumps({k: v["sha256_tensors"][:16] for k, v in out.items()}))
    return 0


def load_s3():
    import fp_read
    return fp_read.load_s3()


def cmd_all(a):
    S3 = load_s3()
    work = a.work
    exp = json.load(gzip.open(S3.EXPORT, "rt", encoding="utf-8"))["tokens"]
    tl = {t: v["log_name"] for t, v in exp.items()}
    rep = json.load(open(f"{work}/picks_report.json"))
    banked = S3.pick_baseline("u400")
    bt = S3.load_table(banked["csv"])
    tabs = {p: S3.load_table(f"{DATA}/score/refe_fp_{p}/refe_fp_{p}.csv") for p in ("BASE", "C0A", "R0B", "M25")}
    seam = {p: np.load(f"{work}/seam_{p}.npz", allow_pickle=False) for p in tabs}
    zb = np.load(banked["seam"], allow_pickle=False)
    bidx = {str(t): i for i, t in enumerate(zb["token"])}
    out = {}
    # ---- IC3 + IC4
    ic3 = rep["IC3_BASE_vs_banked_u400"]
    toks = [str(t) for t in seam["BASE"]["token"]]
    bit_eq = [t for i, t in enumerate(toks) if np.array_equal(seam["BASE"]["poses"][i], zb["poses"][bidx[t]])]
    a_bad = [t for t in bit_eq if any(tabs["BASE"][t][c] != bt[t][c] for c in COLS)]
    tol_only = [t for i, t in enumerate(toks) if t not in set(bit_eq) and np.abs(seam["BASE"]["poses"][i] - zb["poses"][bidx[t]]).max() <= ic3["bar_tol_m"]]
    d_tol = np.array([abs(tabs["BASE"][t]["PDMS"] - bt[t]["PDMS"]) for t in tol_only]) if tol_only else np.zeros(0)
    base_mean = float(np.mean([tabs["BASE"][t]["PDMS"] for t in toks]))
    bank_mean = float(np.mean([bt[t]["PDMS"] for t in toks]))
    PE = S3.PE
    d_all = {t: tabs["BASE"][t]["PDMS"] - bt[t]["PDMS"] for t in toks}
    mu, lo, hi = PE.boot(d_all, {t: tl[t] for t in toks})
    ic4 = {"IC4b_paired_diff_BASEpod_minus_banked": {"mean": mu, "ci95": [lo, hi]}, "IC4a_bit_equal_pose_tokens": len(bit_eq), "IC4a_scoring_differently": len(a_bad), "IC4a_pass": len(a_bad) == 0,
           "IC4a_prime_tolerance_only_tokens": len(tol_only), "IC4a_prime_max_abs_dPDMS": float(d_tol.max()) if d_tol.size else 0.0,
           "IC4a_prime_mean_abs_dPDMS": float(d_tol.mean()) if d_tol.size else 0.0,
           "IC4b_BASE_pod_mean": base_mean, "IC4b_banked_u400_mean": bank_mean, "IC4b_banked_pinned_value": 88.2469, "IC4b_abs_diff_vs_banked": abs(base_mean - bank_mean),
           "IC4b_bar": "0 inside the paired 95 % CI AND |d| <= 0.25 (ADDENDUM_1 amendment note 2)", "IC4b_pass": bool(lo <= 0.0 <= hi and abs(base_mean - bank_mean) <= 0.25), "banked_mean_matches_pinned_to_4dp": round(bank_mean, 4) == 88.2469}
    out["IC3_IC4"] = {"IC3": ic3, "IC4": ic4}
    json.dump(out["IC3_IC4"], open(f"{work}/IC3_IC4.json", "w"), indent=1)
    # ---- IC5: forward arm (from the pod's first chunk) + harness arm (C0A subset scored as its own run)
    ic5 = {"forward": json.load(open(f"{work}/ic5_forward.json")) if os.path.exists(f"{work}/ic5_forward.json") else {"missing": True}}
    subp = [f for f in os.listdir(f"{DATA}/score") if f.startswith("refe_fp_C0A_sub")]
    if subp:
        lab = sorted(subp)[0]
        ts = S3.load_table(f"{DATA}/score/{lab}/{lab}.csv")
        diff = [t for t in ts if any(ts[t][c] != tabs["C0A"][t][c] for c in COLS)]
        ic5["harness"] = {"label": lab, "n": len(ts), "n_differing": len(diff), "bit_identical": len(diff) == 0}
    else:
        ic5["harness"] = {"missing": True}
    ic5["pass"] = bool(ic5["forward"].get("bit_identical") and ic5["harness"].get("bit_identical"))
    json.dump(ic5, open(f"{work}/IC5.json", "w"), indent=1)
    # ---- IC6: the C3 table
    c3 = np.load(C3_TABLE, allow_pickle=False)
    c3t = [str(t) for t in c3["token"]]
    ci = {t: i for i, t in enumerate(c3t)}
    pts = np.load(f"{work}/points.npz", allow_pickle=False)
    ptok = {str(t): i for i, t in enumerate(pts["token"])}
    ic6 = {}
    nb = nt = nbad = 0
    d_tol6 = []
    for p in ("BASE", "C0A", "R0B", "M25"):
        k = pts[f"pick_{p}"]
        sp = seam[p]
        for t in c3t:
            if t not in ptok:
                continue
            i = ptok[t]
            j = int(k[i])
            prop = c3["proposals"][ci[t], j]
            pose = sp["poses"][i]
            if np.array_equal(pose, prop):
                nb += 1
                tab = tabs[p][t]
                want = [100.0 * float(c3["pdms"][ci[t], j])] + [100.0 * float(c3["sub"][ci[t], j, q]) for q in range(6)]
                if any(tab[c] != w for c, w in zip(COLS, want)):
                    nbad += 1
            elif np.abs(pose - prop).max() <= 5e-3:
                nt += 1
                d_tol6.append(abs(tabs[p][t]["PDMS"] - 100.0 * float(c3["pdms"][ci[t], j])))
    d6 = np.array(d_tol6) if d_tol6 else np.zeros(0)
    tol_ok = (d6.size == 0) or (float((d6 <= 1.0).mean()) >= 0.99 and float(np.median(d6)) <= 0.05)
    ic6.update({"bit_equal_checked": nb, "bit_equal_mismatch": nbad, "tolerance_only": nt, "tolerance_only_frac_within_1": (float((d6 <= 1.0).mean()) if d6.size else None),
                "tolerance_only_median_abs_dPDMS": (float(np.median(d6)) if d6.size else None), "tolerance_only_max_abs_dPDMS": (float(d6.max()) if d6.size else None),
                "underpowered": (nb + nt) < 30, "pass": bool(nbad == 0 and tol_ok)})
    json.dump(ic6, open(f"{work}/IC6.json", "w"), indent=1)
    # ---- D1 / D2 / D3
    exp_states = json.load(open(EXPECTED))
    got = rep["state_info"]
    st_ok = all(got.get(nm, {}).get("sha256_tensors") == exp_states[nm]["sha256_tensors"] and got[nm]["update"] == 1000 for nm in ("C0A", "R0B"))
    swapped = {"C0A": got.get("R0B", {}).get("sha256_tensors"), "R0B": got.get("C0A", {}).get("sha256_tensors")}
    d2_teeth = all(swapped[nm] != exp_states[nm]["sha256_tensors"] for nm in swapped) and exp_states["C0A"]["sha256_tensors"] != exp_states["C0B(decoy)"]["sha256_tensors"] and exp_states["R0B"]["sha256_tensors"] != exp_states["R0A(decoy)"]["sha256_tensors"]
    import fp_picks
    AT = fp_picks.load_attrib()
    rng = np.random.default_rng(5)
    lg = rng.normal(size=(300, 64, 7))
    d1 = float((np.asarray(AT.picks(lg)) == np.asarray(AT.picks(lg[rng.permutation(300)]))).mean())
    d3p = f"{work}/D3.json"
    d3 = json.load(open(d3p)) if os.path.exists(d3p) else {"missing": True}
    dd = {"D1_shuffled_pick_agreement": d1, "D1_has_teeth": d1 < 0.5, "D2_state_fingerprints_match_expected": st_ok, "D2_has_teeth": d2_teeth, "D3": d3,
          "D3_has_teeth": bool(d3.get("frac_equal_with_swap", 1.0) < 0.5) if not d3.get("missing") else None}
    json.dump(dd, open(f"{work}/D1_D3.json", "w"), indent=1)
    # ---- the summary, status per ADDENDUM_1
    ic1 = json.load(open(f"{work}/IC1.json")) if os.path.exists(f"{work}/IC1.json") else {"pass": False, "missing": True}
    hard = {"IC1": bool(ic1.get("pass")), "IC2": rep["IC2_max_abs"] == 0.0, "IC4a": ic4["IC4a_pass"], "IC5": ic5["pass"], "IC6": ic6["pass"],
            "D1_teeth": dd["D1_has_teeth"], "D2_ok_and_teeth": bool(st_ok and d2_teeth), "D3_teeth": (dd["D3_has_teeth"] is True)}
    soft = {"IC3": ic3["pass"], "IC4b": ic4["IC4b_pass"]}
    status = "VALID" if all(hard.values()) and all(soft.values()) else ("INVALID" if not all(hard.values()) else "IDENTITY-FAILED")
    summ = {"status": status, "hard_gates": hard, "soft_gates": soft, "ic6_underpowered": ic6["underpowered"]}
    json.dump(summ, open(f"{work}/gates_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
    print("ZZCHECKS_" + status)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("expected-states", "all"))
    ap.add_argument("--work", default=".")
    a = ap.parse_args()
    return {"expected-states": cmd_expected, "all": cmd_all}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
