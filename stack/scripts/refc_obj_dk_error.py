#!/usr/bin/env python3
"""E-OBJ-1 addendum — LONGITUDINAL distance-keeping as an **ERROR**, not as a behaviour statistic.

⚠️ WHY THIS EXISTS, AND WHY IT IS NOT COSMETIC. `lead_metrics.distance_keeping` returns
`headway_min_m` / `time_gap_min_s` / `min_ttc_s` — *behaviours*, not errors. For those quantities
**more is not better**: the reference is what the human driver actually did, and MEASURED on
`refc-base-30k` the GT keeps **28.8857 m** while the shipped selector keeps **28.5928 m**, so an arm
that increases headway moves toward GT only until it overshoots. A paired delta against the shipped
arm therefore says *"this arm behaves differently"*, never *"this arm is right"*.

This module computes the missing half: the **per-window absolute error against the GROUND-TRUTH
path's own distance-keeping**, paired episode-cluster bootstrapped, on the windows where both arms
and GT keep the lead in their own corridor. That is a quantity where "better" has a direction.

⛔ It reuses `refc_obj_probe`'s fitter, folds, survivor mask and features EXACTLY — a
re-implementation would let the addendum drift from the panel it annotates. 0 GPU, 0 inference.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import refc_obj_probe as M
import refc_s1_climbout_probe as S
import refc_sel_probe as P

#: the arms worth the extra fits: the incumbent, S1's marginal under both objectives, and the
#: widest deployable set. Named here so "the best arm" cannot be picked after seeing the numbers.
ARMS = [("B-both", "O-ce"), ("B-both", "O-softade"), ("B-both", "O-softce"),
        ("D-lon+scores", "O-softade"), ("G-all-t0", "O-softade")]
KEYS = ("headway_min_m", "time_gap_min_s", "min_ttc_s")


def run(bank: str, t0_bank: str, arm: str, lead_blocks: str, out_dir: str) -> dict:
    t0 = time.time()
    d, t0b, ctl = M._load(bank, t0_bank)
    eid = list(d["eid"])
    de_all = P.candidate_ade(d["fan"], d["gt"])
    keep, _ = S.survivor_mask(d)
    feats = S.build_features(d, keep)
    feats["emitted_t0_logits"] = S._zrow(t0b["emitted_t0_logits"].double(), keep)
    feats["emitted_logits"] = S._zrow(t0b["emitted_logits"].double(), keep)
    lead = M.load_lead(lead_blocks, d)
    from taniteval import four_families
    dt, prov = four_families.infer_dt({"wp_steps": list(d["wp_steps"]), "dt_s": 0.1})
    b = torch.arange(d["fan"].shape[0])

    gt = M._dk(d["gt"], lead, dt)
    ship = M._dk(d["fan"][b, d["sel"]], lead, dt)

    def err_vs_gt(x):
        return {k: np.abs(np.asarray(x[k], float) - np.asarray(gt[k], float)) for k in KEYS}

    e_ship = err_vs_gt(ship)
    rows = {"shipped": {k: {"n": int(np.isfinite(e_ship[k]).sum()),
                            "mean_abs_err": round(float(np.nanmean(e_ship[k])), 4)}
                        for k in KEYS}}
    for k, v in (("CV", d["cv"]), ("ORACLE-in-fan", d["fan"][b, de_all.argmin(1)])):
        e = err_vs_gt(M._dk(v, lead, dt))
        rows[k] = {}
        for m in KEYS:
            ok = np.isfinite(e[m]) & np.isfinite(e_ship[m])
            rows[k][m] = {
                "n": int(np.isfinite(e[m]).sum()),
                "mean_abs_err": round(float(np.nanmean(e[m])), 4),
                "paired_vs_shipped": (P._paired(e[m][ok], e_ship[m][ok],
                                                [x for x, o in zip(eid, ok) if o])
                                      if ok.sum() >= 2 else {"status": "NOT-COMPUTABLE",
                                                             "n_both": int(ok.sum())}),
            }
    for fname, obj in ARMS:
        sc, _meta = M.loeo(feats, M.FEATURE_SETS[fname], keep, de_all, eid, obj,
                           M.THRESHOLDS["tau_headline"])
        idx = S.argmax_over_survivors(sc, keep)
        e = err_vs_gt(M._dk(d["fan"][b, idx], lead, dt))
        rows[f"{fname}|{obj}"] = {}
        for m in KEYS:
            ok = np.isfinite(e[m]) & np.isfinite(e_ship[m])
            rows[f"{fname}|{obj}"][m] = {
                "n": int(np.isfinite(e[m]).sum()),
                "mean_abs_err": round(float(np.nanmean(e[m])), 4),
                "n_both": int(ok.sum()),
                "paired_vs_shipped": (P._paired(e[m][ok], e_ship[m][ok],
                                                [x for x, o in zip(eid, ok) if o])
                                      if ok.sum() >= 2 else {"status": "NOT-COMPUTABLE",
                                                             "n_both": int(ok.sum())}),
            }
    res = {
        "_what": ("LONGITUDINAL distance-keeping as an ERROR against the GT path's own "
                  "distance-keeping, paired episode-cluster bootstrapped. The behaviour "
                  "statistics live in obj_probe_*.json; this is the half where 'better' has a "
                  "direction."),
        "_evidence_class": "MEASURED",
        "arm": arm, "dt_s": dt, "dt_provenance": prov,
        "window_states": lead["counts"],
        "C-lead-alignment": lead["control"],
        "C-banks": ctl,
        "reference_GT_absolute": {k: {"n": gt["n"],
                                      "mean": gt.get(f"mean_{k.replace('_m', '').replace('_s', '')}")}
                                  for k in KEYS},
        "gt_behaviour": {"mean_headway_min_m": gt.get("mean_headway_min_m"),
                         "mean_time_gap_min_s": gt.get("mean_time_gap_min_s"),
                         "mean_min_ttc_s": gt.get("mean_min_ttc_s"),
                         "n": gt["n"], "n_closing": gt.get("n_closing")},
        "rows": rows,
        "denominator_note": ("an arm whose PREDICTED path leaves the corridor drops its own lead, "
                             "so each row's n differs and every paired contrast is on the "
                             "intersection with the shipped arm. NO_LABEL is never free flow."),
        "wall_s": round(time.time() - t0, 1),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"dk_error_{arm}.json").write_text(json.dumps(P._clean(res), indent=2),
                                              encoding="utf-8")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--t0-bank", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(P.PUBLISHED))
    ap.add_argument("--lead-blocks", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    r = run(a.bank, a.t0_bank, a.arm, a.lead_blocks, a.out)
    print(json.dumps({k: {m: v[m].get("paired_vs_shipped", v[m]) if isinstance(v[m], dict) else v[m]
                          for m in v} for k, v in r["rows"].items() if k != "shipped"},
                     indent=1)[:3000], flush=True)
    return 0


if __name__ == "__main__":                                        # pragma: no cover
    raise SystemExit(main())
