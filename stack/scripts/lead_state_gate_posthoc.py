"""POST-HOC diagnostics for the lead-state gate. NOT the gate, NOT a verdict.

The pre-registered gate (`lead_state_gate.py`) asks ONE question at ONE horizon and
its answer stands whatever this file says. These probes exist only to tell Sayed
*why* the answer came out the way it did, and whether a DIFFERENT question would
be worth pre-registering. Nothing here may be quoted as a pass.

Probes
------
1. HORIZON SWEEP. The gate's target is 2 s. A lead vehicle constrains the ego over
   3-6 s, so the horizon itself is a candidate explanation of a null. Same arms,
   same estimator, targets at 2/3/4/5/6 s.
2. SUBGROUPS where the lead *should* matter most: a close gap, a low TTC, an
   approach (closing > 0), a decelerating ego.
3. A DIFFERENT TARGET: the 2 s speed CHANGE (v(t+H) - v(t)), which strips the
   near-deterministic v*H term that dominates displacement.

Usage:
    python scripts/lead_state_gate_posthoc.py --out <dir with lead_gate_windows_h.parquet>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "taniteval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lead_state_gate import (DENS_COLS, EGO_COLS, LEAD_COLS, N_BOOT, SEED,
                             _fit_predict, _rel_reduction)

HORIZONS = (2, 3, 4, 5, 6)


def _split_fit(df, y, X_by_arm):
    tr = np.asarray(df["split"] == "train")
    ok = np.isfinite(y)
    te = (~tr) & ok
    tr = tr & ok
    pred = {a: _fit_predict(X[tr], y[tr], X[te], "gbm") for a, X in X_by_arm.items()}
    return te, pred


def _cell(df, y, X_by_arm, label):
    te, pred = _split_fit(df, y, X_by_arm)
    sub = df[te]
    eid = sub["clip_id"].to_numpy(str)
    yt = y[te]
    ae = {a: np.abs(pred[a] - yt) for a in pred}
    rel = _rel_reduction(ae["A"], ae["B"], eid, N_BOOT, SEED)
    rel_ctl = _rel_reduction(ae["A"], ae["B_shuf"], eid, N_BOOT, SEED)
    out = {"label": label, "n_windows": int(te.sum()),
           "n_episodes": int(len(np.unique(eid))),
           "mae_A": round(float(np.mean(ae["A"])), 4),
           "mae_B": round(float(np.mean(ae["B"])), 4),
           "mae_B_shuf": round(float(np.mean(ae["B_shuf"])), 4),
           "rel_reduction_B": rel, "rel_reduction_B_shuf_control": rel_ctl}
    print(f"  {label:38s} n={out['n_windows']:6d}/{out['n_episodes']:3d}ep  "
          f"MAE {out['mae_A']:.4f}->{out['mae_B']:.4f}  "
          f"rel {100*rel['point']:+.2f} % [{100*rel['lo']:+.2f},{100*rel['hi']:+.2f}]"
          f"  shufctl {100*rel_ctl['point']:+.2f} %", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    df = pd.read_parquet(out / "lead_gate_windows_h.parquet").reset_index(drop=True)

    ego = df[EGO_COLS].to_numpy(np.float64)
    lead = df[LEAD_COLS].to_numpy(np.float64)
    dens = df[DENS_COLS].to_numpy(np.float64)
    rng = np.random.default_rng(SEED)
    eps = df["clip_id"].to_numpy(str)
    uniq = np.unique(eps)
    order = rng.permutation(len(uniq))
    idx_by_ep = {u: np.flatnonzero(eps == u) for u in uniq}
    shuf = np.empty_like(lead)
    for i, u in enumerate(uniq):
        src = idx_by_ep[uniq[order[i]]]
        dst = idx_by_ep[u]
        shuf[dst] = lead[np.resize(src, len(dst))]
    arms = {"A": ego, "B": np.column_stack([ego, lead]),
            "B_shuf": np.column_stack([ego, shuf]),
            "Bplus": np.column_stack([ego, lead, dens])}

    res = {"WARNING": "POST-HOC / EXPLORATORY. Not the gate. Cannot make it a PASS."}

    print("\n[1] HORIZON SWEEP -- displacement target", flush=True)
    res["horizon_displacement"] = [
        _cell(df, df[f"y_long_h{h}"].to_numpy(np.float64), arms, f"disp @{h}s")
        for h in HORIZONS]

    print("\n[2] HORIZON SWEEP -- speed-change target", flush=True)
    res["horizon_dspeed"] = [
        _cell(df, df[f"dv_h{h}"].to_numpy(np.float64), arms, f"dspeed @{h}s")
        for h in HORIZONS]

    print("\n[3] SUBGROUPS at the pre-registered 2 s horizon", flush=True)
    y2 = df["y_long_h2"].to_numpy(np.float64)
    subs = {
        "lead present": df.lead_present.to_numpy() > 0.5,
        "lead gap < 25 m": (df.gap_m.to_numpy() < 25.0),
        "lead TTC < 6 s": (df.ttc_s.to_numpy() < 6.0),
        "closing > 0.5 m/s": (df.closing_ms.to_numpy() > 0.5),
        "ego decelerating ax<-0.5": df.ax.to_numpy() < -0.5,
        "lead present & v > 5 m/s": (df.lead_present.to_numpy() > 0.5)
                                    & (df.v.to_numpy() > 5.0),
    }
    res["subgroups_2s"] = []
    for name, m in subs.items():
        m = np.asarray(m) & np.isfinite(y2)
        if m.sum() < 500:
            print(f"  {name}: only {m.sum()} windows, skip", flush=True)
            continue
        d2 = df[m].reset_index(drop=True)
        a2 = {k: v[m] for k, v in arms.items()}
        res["subgroups_2s"].append(_cell(d2, y2[m], a2, name))

    print("\n[4] HORIZON SWEEP restricted to lead-present windows", flush=True)
    m = (df.lead_present.to_numpy() > 0.5)
    dm = df[m].reset_index(drop=True)
    am = {k: v[m] for k, v in arms.items()}
    res["horizon_leadpresent"] = [
        _cell(dm, dm[f"y_long_h{h}"].to_numpy(np.float64), am,
              f"lead-present disp @{h}s") for h in HORIZONS]

    p = out / "lead_gate_posthoc.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"\n[posthoc] -> {p}", flush=True)


if __name__ == "__main__":
    main()
