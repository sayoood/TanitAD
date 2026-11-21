#!/usr/bin/env python3
"""Break the R0 identity control down by source and by the BANK row's teacher_device / scenario_type (stdlib only, reads the same inputs as rj5_bank_join.py).
  python rj5_control_breakdown.py --bank <slim> --rows local=<dir> --rows A=<dir> --rows B=<dir> --out <json>
Also counts, per source, RJ5 rows that sit on bank rows whose teacher_device is not 'cpu' (exposure: their RJ5-vs-bank movement includes a provenance term)."""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rj5_bank_join as J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--rows", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meta = {}
    for r in J.read_jsonl(a.bank):
        if int(r.get("rank", 0)) == 0:
            meta[(r["log_name"], r["token"])] = (str(r.get("teacher_device")), r.get("scenario_type"))
    bank, _, _ = J.load_bank(a.bank)
    cells, expo, worst = {}, {}, {}
    for s in a.rows:
        src, d = s.split("=", 1)
        for fp in sorted(glob.glob(os.path.join(d, "rj5_*.jsonl"))):
            for r in J.read_jsonl(fp):
                if "traj" not in r or r.get("arm") not in ("R0", "RJ5"):
                    continue
                k = (r["log_name"], r["token"])
                dev = meta.get(k, ("?", "?"))[0]
                if r["arm"] == "RJ5":
                    expo[(src, dev)] = expo.get((src, dev), 0) + 1
                    continue
                dx = J.dxy(J.flat(r["traj"]), bank[k][0])
                c = cells.setdefault((src, dev), {"n": 0, "exact_zero": 0, "gt_1e-6": 0, "max_dxy_m": 0.0, "logs_nonzero": set()})
                c["n"] += 1
                c["exact_zero"] += dx == 0.0
                c["gt_1e-6"] += dx > 1e-6
                c["max_dxy_m"] = max(c["max_dxy_m"], dx)
                if dx > 1e-6:
                    c["logs_nonzero"].add(k[0])
    out = {"control_by_source_and_bank_teacher_device": {f"{s}|{d}": dict(v, logs_nonzero=len(v["logs_nonzero"])) for (s, d), v in sorted(cells.items())},
           "rj5_rows_by_source_and_bank_teacher_device": {f"{s}|{d}": n for (s, d), n in sorted(expo.items())}}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
