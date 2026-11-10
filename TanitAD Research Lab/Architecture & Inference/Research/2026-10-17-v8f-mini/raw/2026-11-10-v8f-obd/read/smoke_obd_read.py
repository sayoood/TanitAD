"""smoke_obd_read.py -- the INSTRUMENT CONTROL of the OBD read: obd_read.py on the banked OTBAND dumps as stand-ins (tag mapping overridden by smoke_map.json), and a comparison of its
numbers with the OTBAND read's own, to the printed precision.  Needs the OTBAND dumps in F:/TanitAD-scratch/otband_read/t1 (CPU, 2 threads, B = 2000: about 5-10 minutes; run it detached).
  stand-ins:  OBD-CTL_s{k} <- OB-CTL_s{k};  OBD-BAND_s{k} <- OB-BAND_s{k};  OBD-BAND_rep <- OB-BAND_rep;  DAGOACC_s{k} <- PARITY_DAGOACC_s{k};
              OBD-NULL_s{k} <- PARITY_DAGOACC_s{k} (parity forced 'holds': NULL = DAG-OACC);  OBD-NULL_rep <- PARITY_DAGOACCREP_s0 (the DAG-ORC pair, seed 10 + a later trainer file).
  compared:   band question rows  BAND_minus_CTL  (delta, lo, hi, separated)   vs  out/READ_FULL.json  seeds.s{k}.bands.<band>.rows.<metric>.BAND_minus_CTL  + TACTICAL_plan_reads
              floor G                                                         vs  out/READ_FULL.json  replicate.rows_REP_minus_BAND_s0  (|delta|)
              detach rows     CTL_minus_DAGOACC (reference DAG-OACC)          vs  out/READ_FULL.json  CTL_minus_DAG
              floor F (NULL_rep - NULL_s0 = DAG-ORC pair)                     vs  out/DAG_FLOOR.json  seeds.s0.<band>.<metric>.REP_minus_DAGOACC (delta)
Pass = every compared number equals the OTBAND read's to its own printed digits (half a unit of the last printed decimal; the floor / delta point values here are UNROUNDED, reading
choice o1, so they are compared after rounding to the original's digits), every `separated` flag equal.  Usage: python smoke_obd_read.py [--skip-run]
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OTB = os.path.join(os.path.dirname(os.path.dirname(HERE)), "otband", "read")
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
T1 = "F:/TanitAD-scratch/otband_read/t1"


def make_map():
    d = lambda t: f"{T1}/dump_{t}_m10_causal_heads"  # noqa: E731
    m = {"DAGORC_s0": d("PARITY_DAGOACCREP_s0"), "OBD-NULL_rep": d("PARITY_DAGOACCREP_s0"), "OBD-BAND_rep": d("OB-BAND_rep")}
    for s in (0, 1):
        m[f"DAGOACC_s{s}"] = d(f"PARITY_DAGOACC_s{s}")
        m[f"OBD-NULL_s{s}"] = d(f"PARITY_DAGOACC_s{s}")
        m[f"OBD-CTL_s{s}"] = d(f"OB-CTL_s{s}")
        m[f"OBD-BAND_s{s}"] = d(f"OB-BAND_s{s}")
    return m


def ndec(x):
    s = repr(float(x))
    return len(s.split(".")[1]) if "." in s and "e" not in s else 8


def same(mine, orig):
    """mine equals orig at orig's printed digits"""
    return round(float(mine), ndec(orig)) == round(float(orig), ndec(orig)) or abs(float(mine) - float(orig)) <= 0.5 * 10 ** -ndec(orig) + 1e-12


def cmp_row(tag, mine, orig, out, floor_only=False):
    if orig.get("status") != "OK":
        out.append({"row": tag, "field": "status", "mine": mine.get("status"), "orig": orig.get("status"), "ok": mine.get("status") == orig.get("status")})
        return
    for k_m, k_o in (("delta_rounded", "delta"), ("lo", "lo"), ("hi", "hi")):
        ok = same(mine[k_m], orig[k_o])
        out.append({"row": tag, "field": k_o, "mine": mine[k_m], "orig": orig[k_o], "ok": bool(ok)})
    out.append({"row": tag, "field": "separated", "mine": mine["separated"], "orig": orig["separated"], "ok": mine["separated"] == orig["separated"]})


def compare(res, rf, dfl):
    rows = []
    for s in (0, 1):
        sd = rf["seeds"][f"s{s}"]
        for band in ("operative_0_2s", "tactical_2_6s"):
            for m, r in sd["bands"][band]["rows"].items():
                cell = f"{band}:{m}"
                cmp_row(f"band s{s} {cell} BAND-CTL", res["band"]["per_seed"][str(s)]["cells"][cell], r["BAND_minus_CTL"], rows)
                cmp_row(f"detach s{s} {cell} CTL-DAGOACC", res["detach"]["per_seed"][str(s)]["cells"][cell], r["CTL_minus_DAG"], rows)
        for nm, r in sd["TACTICAL_plan_reads"].items():
            cmp_row(f"band s{s} {nm} BAND-CTL", res["band"]["per_seed"][str(s)]["cells"][nm], r["BAND_minus_CTL"], rows)
    for band, mm in rf["replicate"]["rows_REP_minus_BAND_s0"].items():
        if band == "full_0_6s":
            continue
        for m, r in mm.items():
            if r.get("status") != "OK":
                continue
            mine, orig = res["floors"]["G"][f"{band}:{m}"], abs(r["delta"])
            rows.append({"row": f"floor G {band}:{m}", "field": "|delta|", "mine": mine, "orig": orig, "ok": bool(same(mine, orig))})
    for band in ("operative_0_2s", "tactical_2_6s"):
        for m, r in dfl["seeds"]["s0"][band].items():
            if not isinstance(r, dict) or "REP_minus_DAGOACC" not in r:
                continue
            mine, orig = res["floors"]["F"].get(f"{band}:{m}"), abs(r["REP_minus_DAGOACC"]["delta"])
            if mine is None:
                continue
            rows.append({"row": f"floor F {band}:{m} (= DAG-ORC pair)", "field": "|delta|", "mine": mine, "orig": orig, "ok": bool(same(mine, orig))})
    return rows


def main():
    mp = HERE + "/smoke_map.json"
    json.dump(make_map(), open(mp, "w"), indent=1)
    out = HERE + "/SMOKE_OBD_READ.json"
    if "--skip-run" not in sys.argv:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run([PY, HERE + "/obd_read.py", "--smoke", "--parity", "holds", "--map", mp, "--out", out, "--md", HERE + "/SMOKE_OBD_READ.md", "--n-boot", "2000"],
                           env=env, capture_output=True, encoding="utf-8")
        print(r.stdout[-1500:], r.stderr[-1500:])
        assert r.returncode == 0, "obd_read.py failed"
    res = json.load(open(out, encoding="utf-8"))
    rf = json.load(open(OTB + "/out/READ_FULL.json", encoding="utf-8"))
    dfl = json.load(open(OTB + "/out/DAG_FLOOR.json", encoding="utf-8"))
    rows = compare(res, rf, dfl)
    bad = [r for r in rows if not r["ok"]]
    summary = {"n_compared": len(rows), "n_mismatch": len(bad), "mismatches": bad[:40], "reproduces_otband_read": not bad,
               "smoke_outcome": res.get("outcome"), "problems": res["controls"]["problems"], "notes": "stand-ins, never a result; see the docstring for the mapping"}
    json.dump(summary, open(HERE + "/SMOKE_COMPARE.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in summary.items() if k != "mismatches"}, indent=1, default=str))
    print("first mismatches:", bad[:5])


if __name__ == "__main__":
    main()
