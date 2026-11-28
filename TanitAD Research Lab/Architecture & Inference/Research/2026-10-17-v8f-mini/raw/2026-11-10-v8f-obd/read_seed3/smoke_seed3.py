"""smoke_seed3.py -- the INSTRUMENT CONTROL of obd_read_seed3.py: the full reader on the existing REAL OBD dumps mapped as stand-ins (SMOKE, never a result):
    OBD-CTL_s10 <- OBD-CTL_s0, OBD-BAND_s10 <- OBD-BAND_s0   (init M2b s0)      OBD-CTL_s11 <- OBD-CTL_s1, OBD-BAND_s11 <- OBD-BAND_s1   (init M2b s1)
    OBD-BAND_rep, OBD-BAND_s0, DAGOACC_s0/s1: the real dumps of the existing read.
Reads only F:/TanitAD-scratch/otband_read/t1 (never written); writes only into this folder.  Compared (SMOKE_COMPARE_SEED3.json):
  * the stand-in s10 / s11 cell tables equal the existing RESULT_OBD.json band.per_seed.0 / .1 cells on every one of the 24 cells (delta, lo, hi, floor, gain, cost: bit-equal,
    since the same function, estimator and seed read the same arrays);
  * the stand-in labels equal the existing labels (s10 = s0's BAND-ADDS, s11 = s1's BAND-COSTS), so the four labels read ADDS, COSTS, ADDS, COSTS and the combination is BAND-UNSETTLED;
  * floor G computed here equals floors.G of the existing result (the reader's own cross-check, zero problems).
Usage: python smoke_seed3.py   (CPU, 2 threads, B 2,000: about 1-2 minutes)"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
T1 = "F:/TanitAD-scratch/otband_read/t1"
EXISTING = os.path.join(os.path.dirname(HERE), "read", "RESULT_OBD.json")


def make_map():
    d = lambda t: f"{T1}/dump_{t}_m10_causal_heads"  # noqa: E731
    m = {"OBD-BAND_rep": d("OBD-BAND_rep"), "OBD-BAND_s0": d("OBD-BAND_s0")}
    for s in (0, 1):
        m[f"DAGOACC_s{s}"] = d(f"PARITY_DAGOACC_s{s}")
    for new, old in (("OBD-CTL_s10", "OBD-CTL_s0"), ("OBD-BAND_s10", "OBD-BAND_s0"), ("OBD-CTL_s11", "OBD-CTL_s1"), ("OBD-BAND_s11", "OBD-BAND_s1")):
        m[new] = d(old)
    return m


def main():
    mp = HERE + "/smoke_map_seed3.json"
    json.dump(make_map(), open(mp, "w"), indent=1)
    out = HERE + "/SMOKE_OBD_SEED3.json"
    if "--skip-run" not in sys.argv:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
                   PYTHONPATH="F:/TanitAD-scratch/otband_read/tree/stack;F:/TanitAD-scratch/otband_read/tree/taniteval")
        r = subprocess.run([PY, HERE + "/obd_read_seed3.py", "--smoke", "--map", mp, "--out", out, "--md", HERE + "/SMOKE_OBD_SEED3.md", "--n-boot", "2000"],
                           env=env, capture_output=True, encoding="utf-8")
        print(r.stdout[-1500:], r.stderr[-1500:])
        assert r.returncode == 0, "obd_read_seed3.py failed"
    res = json.load(open(out, encoding="utf-8"))
    ex = json.load(open(EXISTING, encoding="utf-8"))
    rows, bad = 0, []
    for new, old in (("s10", "0"), ("s11", "1")):
        mine, orig = res["band"]["per_seed"][new]["cells"], ex["band"]["per_seed"][old]["cells"]
        for c, v in orig.items():
            for k in ("delta", "lo", "hi", "floor", "gain", "cost", "separated"):
                rows += 1
                if mine[c].get(k) != v.get(k):
                    bad.append({"seed": new, "cell": c, "field": k, "mine": mine[c].get(k), "orig": v.get(k)})
    labels = res["band"]["labels"]
    want = {"s0": "BAND-ADDS", "s1": "BAND-COSTS", "s10": "BAND-ADDS", "s11": "BAND-COSTS"}
    summary = {"n_compared": rows, "n_mismatch": len(bad), "mismatches": bad[:20], "stand_in_cells_equal_existing": not bad, "labels": labels, "labels_as_expected": labels == want,
               "outcome": res["outcome"], "band_outcome": res["band"]["outcome"], "problems": res["controls"]["problems"], "floor_G_equals_existing": not any("floor G" in p for p in res["controls"]["problems"]),
               "notes": "SMOKE: stand-in dumps (existing OBD-CTL/BAND s0 and s1 as s10 and s11), never a result"}
    json.dump(summary, open(HERE + "/SMOKE_COMPARE_SEED3.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in summary.items() if k != "mismatches"}, indent=1, default=str))
    assert not bad and labels == want and not res["controls"]["problems"], "SMOKE comparison failed"


if __name__ == "__main__":
    main()
