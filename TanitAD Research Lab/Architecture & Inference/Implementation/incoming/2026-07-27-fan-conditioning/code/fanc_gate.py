#!/usr/bin/env python3
"""S0 GATE + S1 PREMISE.

S0 -- reproduce every committed number this stream will lean on, from raw
artifacts, BEFORE quoting any new one. If a gate fails the stream stops.

S1 -- the descriptive premise: is the shipped fan ALREADY `v0`-conditioned?
Measured as a SLOPE (m/s of proposed speed per m/s of ego speed), not a
correlation -- the correlation on this data is -0.97 and is MISLEADING, because
the envelope is nearly constant and the tiny residual drift dominates the
normalised statistic. GT's slope on the identical windows, through the identical
code path, is the instrument's positive control.

Usage:  python fanc_gate.py
"""
from __future__ import annotations

import json

import numpy as np

from fanc_common import (OUT, T_HORIZON, ade, eid_str, load_refc_fan, load_v5,
                         mean_speed, r4)

ARMS = ("xl", "base", "small")


def main() -> None:
    res: dict = {"_stream": "2026-07-27-fan-conditioning",
                 "_estimator": "episode_cluster_bootstrap B=2000 (unit=episode)",
                 "_note": "S0 gate = committed numbers reproduced from raw "
                          "artifacts. S1 = is the shipped fan v0-conditioned?"}

    # ---------------------------------------------------------------- S0 gate
    gate: dict = {}
    for arm in ARMS:
        d = load_refc_fan(arm)
        fan = d["fan"].numpy()
        gt = d["gt"].numpy()
        sel = d["sel"].numpy()
        err = np.linalg.norm(fan - gt[:, None], axis=-1).mean(-1)   # [W,N]
        gate[f"refc_{arm}"] = {
            "n_windows": int(fan.shape[0]),
            "n_anchors": int(fan.shape[1]),
            "oracle_in_fan": r4(err.min(1).mean()),
            "as_trained": r4(err[np.arange(len(sel)), sel].mean()),
            "cv_own_ade": r4(ade(d["cv"].numpy(), gt).mean()),
        }
    v5 = load_v5("v1")
    fe = v5["fan_err4"].numpy()
    pk = {k: v.numpy() for k, v in v5["picks"].items()}
    w = np.arange(fe.shape[0])
    gate["v4_fan_v1_scorer"] = {
        "oracle_in_fan": r4(fe.min(1).mean()),
        "A0_as_trained": r4(fe[w, pk["A0_as_trained"]].mean()),
        "C2_wm_ref_proximity": r4(fe[w, pk["C2_wm_ref_proximity"]].mean()),
        "A4_imag_combo_oof": r4(fe[w, pk["A4_imag_combo_oof"]].mean()),
    }
    # window identity between the two artifact families
    gate["window_identity"] = {
        "v0_allclose_refcxl_vs_v5dump": bool(
            np.allclose(load_refc_fan("xl")["v0"].numpy(), v5["v0"].numpy())),
        "n_windows_match": int(fe.shape[0]) == int(load_refc_fan("xl")["fan"].shape[0]),
    }
    # expected values, hard-coded from the committed documents
    expect = {"refc_xl.oracle_in_fan": 0.1640, "refc_xl.as_trained": 0.4714,
              "refc_base.oracle_in_fan": 0.1914, "refc_small.oracle_in_fan": 0.2213,
              "v4_fan_v1_scorer.oracle_in_fan": 0.2505,
              "v4_fan_v1_scorer.A0_as_trained": 0.8563,
              "v4_fan_v1_scorer.C2_wm_ref_proximity": 0.5645}
    checks = {}
    for k, v in expect.items():
        a, b = k.split(".")
        got = gate[a][b]
        checks[k] = {"expected": v, "got": got, "ok": abs(got - v) <= 5e-4}
    gate["_checks"] = checks
    gate["_all_ok"] = all(c["ok"] for c in checks.values())
    res["S0_gate"] = gate
    print("S0 gate all_ok =", gate["_all_ok"])
    for k, c in checks.items():
        if not c["ok"]:
            print("  FAIL", k, c)

    # ------------------------------------------------------------- S1 premise
    prem: dict = {"_definition":
                  "slope = OLS d(candidate mean speed)/d(v0). A state-conditioned "
                  "proposal set has slope ~ +1 (GT's own value). 0 means the fan "
                  "ignores ego speed entirely."}
    for arm in ARMS:
        d = load_refc_fan(arm)
        fan = d["fan"].numpy()
        gt = d["gt"].numpy()
        v0 = d["v0"].numpy()
        s = mean_speed(fan)                       # [W,N]
        fm = s.mean(1)
        sl_fan = float(np.polyfit(v0, fm, 1)[0])
        sl_gt = float(np.polyfit(v0, mean_speed(gt), 1)[0])
        # usable fraction: candidates within +-2 m/s of the speed GT actually took
        usable = float(np.mean(np.abs(s - mean_speed(gt)[:, None]) <= 2.0))
        prem[f"refc_{arm}"] = {
            "n_anchors": int(fan.shape[1]),
            "slope_fan_speed_on_v0": r4(sl_fan),
            "slope_gt_speed_on_v0": r4(sl_gt),          # positive control
            "corr_fan_speed_v0": r4(np.corrcoef(v0, fm)[0, 1]),
            "fan_mean_speed_over_windows": r4(fm.mean()),
            "fan_mean_speed_sd_over_windows": r4(fm.std()),
            "v0_sd_over_windows": r4(v0.std()),
            "frac_candidates_within_2ms_of_gt": r4(usable),
            "effective_usable_candidates": r4(usable * fan.shape[1]),
        }
        # per-v0-quintile mis-centring
        q = np.quantile(v0, np.linspace(0, 1, 6))
        buckets = []
        for i in range(5):
            m = (v0 >= q[i]) & ((v0 < q[i + 1]) if i < 4 else (v0 <= q[i + 1]))
            buckets.append({
                "v0_lo": r4(q[i]), "v0_hi": r4(q[i + 1]), "n": int(m.sum()),
                "gt_mean_speed": r4(mean_speed(gt)[m].mean()),
                "fan_cand_mean_speed": r4(s[m].mean()),
                "miscentring_ms": r4(s[m].mean() - mean_speed(gt)[m].mean()),
                "frac_within_2ms": r4(np.mean(
                    np.abs(s[m] - mean_speed(gt)[m][:, None]) <= 2.0)),
            })
        prem[f"refc_{arm}"]["by_v0_quintile"] = buckets

    # v4's own fan, from the 902 kB along-track tensor staged by the clip stream
    try:
        import torch
        from fanc_common import V5 as V5DIR
        s4 = torch.load(V5DIR / "fan_last_along_v4.pt", map_location="cpu",
                        weights_only=False)
        s4 = (s4["fan_last_along"] if isinstance(s4, dict) else s4).numpy() / T_HORIZON
        v0 = v5["v0"].numpy()
        fm = s4.mean(1)
        prem["v4_fan"] = {
            "n_anchors": int(s4.shape[1]),
            "slope_fan_speed_on_v0": r4(np.polyfit(v0, fm, 1)[0]),
            "corr_fan_speed_v0": r4(np.corrcoef(v0, fm)[0, 1]),
            "fan_mean_speed_over_windows": r4(fm.mean()),
            "fan_mean_speed_sd_over_windows": r4(fm.std()),
        }
    except Exception as e:                                    # pragma: no cover
        prem["v4_fan"] = {"error": repr(e)}

    res["S1_premise"] = prem
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fanc_gate.json").write_text(json.dumps(res, indent=2))
    print("wrote", OUT / "fanc_gate.json")
    for arm in ARMS:
        p = prem[f"refc_{arm}"]
        print(f"  {arm:6s} N={p['n_anchors']:3d} slope_fan={p['slope_fan_speed_on_v0']:+.4f} "
              f"(GT {p['slope_gt_speed_on_v0']:+.4f})  usable={p['frac_candidates_within_2ms_of_gt']:.4f}")
    print("  v4_fan:", prem.get("v4_fan"))


if __name__ == "__main__":
    main()
