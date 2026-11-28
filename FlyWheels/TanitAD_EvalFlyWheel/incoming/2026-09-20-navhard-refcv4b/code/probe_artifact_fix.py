#!/usr/bin/env python3
"""Rebuild every arm's TanitEval artifact from a COMPLETED run's banked scores with the (fixed) builder,
and run the criteria checker on each — BEFORE any re-derive run depends on it.

⭐ The discriminating control is the SAME artifact under the SAME registry, OLD builder vs NEW: the run's
banked ``artifacts/<arm>.json`` (old builder) must still read its violations under the current registry
(proving the checker is live and can go red), and the rebuilt one must read 0.

    python probe_artifact_fix.py --run <run dir> --out raw/artifact_fix_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "taniteval"))
sys.path.insert(0, str(REPO / "tools"))


def check(art_path: Path) -> dict:
    import criteria_check as cc
    art = json.loads(Path(art_path).read_text(encoding="utf-8"))
    reg = cc.load_registry() if hasattr(cc, "load_registry") else json.loads(
        (REPO / "products" / "P7-TanitEval" / "CRITERIA_REGISTRY.json").read_text(encoding="utf-8"))
    res = cc.check_artifact(art, reg) if hasattr(cc, "check_artifact") else None
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from taniteval.bench.navsim import artifacts as ART, benchmark as BM, profiles as P, summarize as S, \
        model_arms as MA
    import pandas as pd
    run = Path(a.run)
    prof = P.SPLITS["navhard_two_stage"]
    y = P.read_split_yaml(prof.name)
    S1, S2 = set(y["stage_one"]), set(y["stage_two"])
    out = {"run": str(run), "arms": {}}
    import subprocess
    tmp = Path(tempfile.mkdtemp(prefix="w7_artfix_"))
    for arm in ("CV", "STOP", "ECHO", "A1"):
        rdir = run / "raw" / arm
        hooks = BM._hooks(rdir, arm)
        hdr, raw = S.read_raw_rows(run / "scores" / f"{arm}.csv")
        log_of = {c["token"]: c["log_name"] for c in hooks if c.get("token") and c.get("log_name")}
        n_standin = 0
        if arm == "A1":
            n_standin = BM._seam_standin_rows(run / "raw" / "model" / "A1.npz")
        elif arm in ("STOP", "ECHO"):
            n_standin = BM._seam_standin_rows(run / "raw" / "seams" / f"{arm}.npz")
        head = None if n_standin else S.headline_value(raw, hdr)
        interval = BM._interval(prof, rdir / f"{arm}_final_scores_frame.csv", y["mapping"], log_of, head)
        spec = BM.FLOOR_SPECS[arm]["spec"] if arm != "A1" else BM._model_spec(MA.MODEL_ARMS["A1"])
        full_df = pd.read_csv(run / "scores" / f"{arm}.csv", index_col=0)
        ctl = {"C4_formula": S.c4(full_df, True)}
        art = ART.build_arm_artifact(arm=arm, spec=spec, split=prof.name, protocol=prof.protocol, raw_rows=raw,
                                     hooks=hooks, S1=S1, S2=S2, n_logs=prof.n_logs, interval=interval,
                                     n_standin=n_standin, controls=ctl)
        new_p = tmp / f"{arm}.new.json"
        new_p.write_text(json.dumps(art, default=ART._jd), encoding="utf-8")
        res = {}
        for tag, p in (("OLD_banked", run / "artifacts" / f"{arm}.json"), ("NEW_rebuilt", new_p)):
            # the SAME invocation the suite uses (contract.py::run_criteria): target + `--json <OUT>`
            js = tmp / f"{arm}.{tag}.criteria.json"
            r = subprocess.run([sys.executable, str(REPO / "tools" / "criteria_check.py"), str(p), "--json", str(js)],
                               capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(REPO))
            try:
                j = json.loads(js.read_text(encoding="utf-8"))
                arts = list((j.get("artifacts") or {}).values())
                nv = sum(int(x.get("n_violations", 0)) for x in arts)
                nw = sum(int(x.get("n_work_items", 0)) for x in arts)
                viol = [v.get("id") or v.get("criterion") or v.get("name") for x in arts
                        for v in (x.get("violations") or [])]
            except Exception as e:                                       # noqa: BLE001
                nv, nw, viol = None, None, [f"unparseable checker output: {type(e).__name__}", r.stdout[:200]]
            res[tag] = {"n_violations": nv, "n_work_items": nw, "violations": viol[:20]}
        res["estimator"] = {"cluster_unit": art["estimator"]["cluster_unit"],
                            "interval_status": art["estimator"]["interval"].get("status")}
        res["n_standin"] = n_standin
        res["refused_criteria"] = art.get("_w7_refused_criteria")
        out["arms"][arm] = res
        print(f"{arm:5s} standin={n_standin:<4d} OLD viol={res['OLD_banked']['n_violations']}  "
              f"NEW viol={res['NEW_rebuilt']['n_violations']} work={res['NEW_rebuilt']['n_work_items']}  "
              f"unit={res['estimator']['cluster_unit']!r} interval={res['estimator']['interval_status']}")
    ok = all(v["NEW_rebuilt"]["n_violations"] == 0 for v in out["arms"].values())
    live = all((v["OLD_banked"]["n_violations"] or 0) > 0 for v in out["arms"].values())
    out["verdict"] = ("PASS: every rebuilt artifact reads 0 violations, and the checker is LIVE (every old "
                      "artifact still reads >0 under the same registry)" if (ok and live) else
                      f"FAIL: new all-zero={ok}, checker-live={live}")
    Path(a.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(out["verdict"])
    return 0 if (ok and live) else 1


if __name__ == "__main__":
    sys.exit(main())
