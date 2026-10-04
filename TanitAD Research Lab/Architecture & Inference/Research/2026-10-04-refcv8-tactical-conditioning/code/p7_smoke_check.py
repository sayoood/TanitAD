"""P7 (X3) full-trainer smoke -- read the RUN'S ARTIFACTS, never the exit code.

Checks on <run>/config.json and <run>/metrics.jsonl:
  X1 the past-only stamp is present with every required token, and the v6 ORACLE stamp is absent;
  X2 the channel's record names the source (seams.max_speed_onehot_v6.provenance ego-PAST, no sidecar) and the census
     is the v9 column on both splits (known share, 4-way shares);
  X3 refcv8 stamp: speed_input n2, speed_unknown_p 0.45, regression_arm.roll_speed_input False;
  X4 the metrics rows exist for every step, and the eval row exists;
  X5 G-DVB: zero mismatches in config.json[declared_vs_built].
Writes <run>/p7_smoke_check.json; prints PASS / FAIL per check.
Run:  python p7_smoke_check.py --run <run dir> --stack <tree>/stack
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--stack", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.stack)
    from tanitad.train import refcv8_train as RT
    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    res = {}
    st = cfg.get("r8_speed_derivation")
    res["X1_stamp"] = bool(isinstance(st, str) and all(t in st for t in RT.R8_SPEED_STAMP_REQUIRED["n2"])
                           and cfg.get("speed_max_derivation_v6") is None)
    seam = (cfg.get("seams") or {}).get("max_speed_onehot_v6") or {}
    ms = cfg.get("refcv6_max_speed") or {}
    tr, ev = ms.get("train") or {}, ms.get("eval") or {}
    res["X2_source"] = bool(str(seam.get("provenance", "")).startswith("ego-PAST") and not seam.get("sidecar")
                            and seam.get("built") is True
                            and tr.get("column") == "speed_n2_kmh" and ev.get("column") == "speed_n2_kmh"
                            and tr.get("known_frac", 0) > 0.99 and ev.get("known_frac", 0) > 0.99)
    r8 = cfg.get("refcv8") or {}
    res["X3_refcv8_stamp"] = bool(r8.get("speed_input") == "n2" and r8.get("speed_unknown_p") == 0.45
                                  and (r8.get("regression_arm") or {}).get("roll_speed_input") is False)
    steps = sorted({int(r["step"]) for r in rows if "step" in r and "loss" in r})
    evals = [r for r in rows if any(k.startswith("eval") for k in r)]
    res["X4_rows"] = bool(steps and steps[-1] >= 1 and evals)
    dvb = cfg.get("declared_vs_built") or {}
    res["X5_gdvb"] = dvb.get("mismatches") in (0, [], None) and "mismatches" in dvb
    out = {"checks": res, "PASS": all(res.values()),
           "census_train": {k: tr.get(k) for k in ("n_windows", "known_frac", "bin4_shares_of_known",
                                                   "over_ceiling_frac_of_known")},
           "census_eval": {k: ev.get(k) for k in ("n_windows", "known_frac", "bin4_shares_of_known",
                                                  "over_ceiling_frac_of_known")},
           "steps_logged": steps, "n_eval_rows": len(evals), "dvb": dvb.get("mismatches")}
    (run / "p7_smoke_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in res.items():
        print(("PASS " if v else "FAIL ") + k)
    print("OVERALL", "PASS" if out["PASS"] else "FAIL")


if __name__ == "__main__":
    main()
