"""RI-C0 read (PREREG_RI_C0.md, frozen before any RI rollout): C0 under --scene reindexed against C0 on logged frames and P10.
Reuses the RECOVER-NEXT reader's functions, loaded BY FILE PATH (its pins are checked by heavy_init); no stack root on sys.path.
usage: python -I ri_read.py [--out-root F:/TanitAD-scratch/motion_c/ri/out] [--n-boot 2000]"""
import argparse
import importlib.util
import json
import os
import time

import numpy as np

DL = "E:/Projects/TanitAD-artifacts/dagland"
RR = f"{DL}/recover_next/read/code/recover_read.py"
REF_ARR = f"{DL}/recover_next/read/raw/arrays"
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")


def load_rr():
    spec = importlib.util.spec_from_file_location("recover_read", RR)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="F:/TanitAD-scratch/motion_c/ri/out")
    ap.add_argument("--work-dir", default="F:/TanitAD-scratch/motion_c/ri/work")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    rr = load_rr()
    rr.WORK = f"{a.work_dir}/rig"
    rr.heavy_init()
    GD, SR, M1 = rr.GD, rr.SR, rr.M1
    arr_dir = f"{HERE}/raw/arrays"
    res = {"card": "PREREG_RI_C0.md", "card_sha256": rr.sha256_file(f"{HERE}/PREREG_RI_C0.md"), "reader": RR, "reader_sha256": rr.sha256_file(RR),
           "tier": "T1-kin (logged cameras; reindexed = longitudinal-only approximate re-perception, not T2)",
           "estimator": "paired episode-cluster bootstrap B=%d seed 0 (gapdecomp.decompose); answers another draw of EPISODES only" % a.n_boot,
           "clock": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seeds": {}}
    sens = {}
    for s in rr.SEEDS:
        z = {}
        for t in (f"RIC0_s{s}", f"RIC0C_s{s}"):
            hdr = rr.run_header(a.out_root, t)
            assert hdr.get("scene") == "reindexed" or json.dumps(hdr).find("reindexed") >= 0, f"{t}: header does not say reindexed"
            z[t] = rr.read_arrays(rr.extract(t, s, a.out_root, arr_dir))
        for t in (f"C0_s{s}", f"C0C_s{s}", f"FZC0_s{s}"):
            z[t] = rr.read_arrays(f"{REF_ARR}/{t}.npz")
        ri, ric, c0, c0c, fz = (z[f"RIC0_s{s}"], z[f"RIC0C_s{s}"], z[f"C0_s{s}"], z[f"C0C_s{s}"], z[f"FZC0_s{s}"])
        for k, q in z.items():
            assert np.array_equal(q["eid"], c0["eid"]), f"{k}: windows differ from C0's"
        eid = c0["eid"]
        cls, _ = GD.classify(c0["v0"], c0["g"])
        blk = {}
        blk["RIC0_minus_C0"], _ = rr.rows_of(rr.Aof(ri), rr.Aof(c0), eid, cls, a.n_boot)
        blk["RIC0_minus_FZC0"], _ = rr.rows_of(rr.Aof(ri), rr.Aof(fz), eid, cls, a.n_boot)
        blk["RIC0_minus_P10"], _ = rr.rows_of(rr.Aof(ri), rr.Bof(ri), eid, cls, a.n_boot)
        blk["C0_minus_P10"], _ = rr.rows_of(rr.Aof(c0), rr.Bof(c0), eid, cls, a.n_boot)
        blk["RIC0C_minus_P10"], _ = rr.rows_of(rr.Aof(ric), rr.Bof(ric), eid, cls, a.n_boot)
        blk["C0C_minus_P10"], _ = rr.rows_of(rr.Aof(c0c), rr.Bof(c0c), eid, cls, a.n_boot)
        blk["P10_floor_RI_minus_logged_max_abs"] = float(max(np.nanmax(np.abs(np.asarray(u, float) - np.asarray(v, float))) for u, v in zip(rr.Bof(ri), rr.Bof(c0))))
        blk["readA_RIC0C"] = rr.read_a_block(SR, GD, blk["RIC0C_minus_P10"])
        blk["readA_C0C_logged"] = rr.read_a_block(SR, GD, blk["C0C_minus_P10"])
        ex_ri, ex_lg = blk["RIC0_minus_P10"][M1]["delta"], blk["C0_minus_P10"][M1]["delta"]
        r = ex_ri / ex_lg if ex_lg else float("nan")
        blk["m1_excess"] = {"RI": ex_ri, "logged": ex_lg, "ratio_RI_over_logged": r}
        sep = bool(blk["RIC0_minus_C0"][M1]["separated"])
        sens[s] = (sep, abs(r - 1) < 0.25)
        res["seeds"][f"s{s}"] = blk
        print(f"s{s} M1 RIC0-C0 {blk['RIC0_minus_C0'][M1]['delta']:+.4f} sep {sep} | excess RI {ex_ri:+.4f} logged {ex_lg:+.4f} r {r:.3f} | "
              f"ADE6 RIC0-C0 {blk['RIC0_minus_C0']['ADE6']['delta']:+.4f} | RIC0C DOMINANT {blk['readA_RIC0C']['DOMINANT']} {blk['readA_RIC0C']['failing_terms']}")
    if all(v[0] for v in sens.values()):
        out = "CLOCK-SENSITIVE"
    elif all((not v[0]) and v[1] for v in sens.values()):
        out = "CLOCK-INSENSITIVE"
    else:
        out = "MIXED"
    res["outcome"] = out
    os.makedirs(f"{HERE}/raw", exist_ok=True)
    with open(f"{HERE}/raw/RI_C0_READ.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, default=rr._jd)
    print("OUTCOME", out)


if __name__ == "__main__":
    main()
