#!/usr/bin/env python3
"""WHERE a model arm's PDMS_v1 goes: the multiplicative gate, by speed, by log, and by lateral error.

    python code/decompose_gate.py --arm A1 --seam raw/bridge_navtest/seam_A1_ego_cmd.npz
    python code/decompose_gate.py --arm A1sub200 --seam raw/bridge_navtest/seam_A1_sub200.npz \
        --restrict-tokens raw/A1_sub200_tokens.json --tag sub200

PDMS_v1 = NC·DAC·(5·EP + 5·TTC + 2·C)/12, so a single non-compliant token scores ZERO. A mean that
loses to a do-nothing plan can hide an arm that wins most head-to-heads — this reads the shape:
  * W/T/L and the mean AND median paired delta against a floor (default STOP);
  * the NC·DAC gate: tokens zeroed by NC, by DAC, by either — for the arm and the floor;
  * PDMS on the tokens the arm does NOT zero, arm vs floor on those same tokens;
  * the DAC/NC failure rate by t0-speed band, and how many logs the failures span;
  * progress with BOTH columns NAMED — ``agent_m`` and ``pdm_closed_m``. ⛔ The hook stores
    ``progress_raw_m = [PDM-Closed, agent]`` (``navsim_v1_win.py:199``); W3 once quoted index 0
    as the model's plan (COMMS, third self-correction). No bare "progress" is ever emitted;
  * a direct LATERAL read per token, independent of the suite adapter: heading error and
    cross-track error at the 4 s horizon, model vs the logged human future, both ego-frame.

Evidence class of everything written: MEASURED on the named CSV/hooks/seam/export.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import statistics as st
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
RAW = PKG / "raw"
EXPORT = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
SPEEDS = ((0.0, 1.0), (1.0, 4.0), (4.0, 8.0), (8.0, 100.0))
Z = 1e-12


def load_csv(label: str) -> dict:
    p = RAW / label / f"{label}.csv"
    return {r["token"]: r for r in csv.DictReader(open(p, encoding="utf-8"))
            if r["token"] != "average" and r["valid"] in ("True", "true", "1")}


def load_hooks(label: str) -> dict:
    p = RAW / label / f"{label}_hooks.json"
    if not p.exists():
        return {}
    return {r["token"]: r for r in json.load(open(p, encoding="utf-8"))["pdm_score_calls"]
            if "token" in r}


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def lateral_at_horizon(model: np.ndarray, human: np.ndarray) -> dict:
    """Heading error and cross-track error at the LAST pose (4 s), ego frame, (x, y, heading).
    Cross-track = the model's final position expressed in the human's final pose frame, lateral
    component. Pure geometry; no adapter, no scorer."""
    mx, my, mh = (float(v) for v in model[-1])
    hx, hy, hh = (float(v) for v in human[-1])
    dx, dy = mx - hx, my - hy
    xtrack = -math.sin(hh) * dx + math.cos(hh) * dy
    return {"heading_err_deg_4s": abs(math.degrees(wrap(mh - hh))), "xtrack_m_4s": abs(xtrack),
            "fde_m_4s": math.hypot(dx, dy)}


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(st.median(xs), 4) if xs else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="label prefix: reads raw/<arm>_navtest/")
    ap.add_argument("--seam", required=True)
    ap.add_argument("--floor", default="STOP")
    ap.add_argument("--restrict-tokens", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    A, F = load_csv(f"{a.arm}_navtest"), load_csv(f"{a.floor}_navtest")
    hk = load_hooks(f"{a.arm}_navtest")
    toks = sorted(set(A) & set(F))
    if a.restrict_tokens:
        d = json.load(open(a.restrict_tokens, encoding="utf-8"))
        keep = set(d["tokens"] if isinstance(d, dict) else d)
        toks = [t for t in toks if t in keep]
    if not toks:
        raise SystemExit("⛔ no token scored by both the arm and the floor")
    doc = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    z = np.load(a.seam, allow_pickle=False)
    sidx = {str(t): i for i, t in enumerate(z["token"].tolist())}
    missing_seam = [t for t in toks if t not in sidx]
    if missing_seam:
        raise SystemExit(f"⛔ {len(missing_seam)} scored tokens are absent from the seam "
                         f"(first {missing_seam[:3]}) — refusing to join mismatched sets")

    def f(r, c):
        return float(r[c])

    def zeroed(r):
        return f(r, "no_at_fault_collisions") == 0.0 or f(r, "drivable_area_compliance") == 0.0

    delta = [f(A[t], "score") - f(F[t], "score") for t in toks]
    zA = {t for t in toks if zeroed(A[t])}
    zF = {t for t in toks if zeroed(F[t])}
    keep = [t for t in toks if t not in zA]
    lat = {}
    for t in toks:
        lat[t] = lateral_at_horizon(np.asarray(z["poses"][sidx[t]], np.float64),
                                    np.asarray(doc[t]["human_future_poses"], np.float64))
    dac0 = [t for t in toks if f(A[t], "drivable_area_compliance") == 0.0]
    nc0 = [t for t in toks if f(A[t], "no_at_fault_collisions") == 0.0]
    dac1 = [t for t in toks if t not in set(dac0)]

    def v0(t):
        h = hk.get(t)
        if h and h.get("v0_mps") is not None:
            return float(h["v0_mps"])
        vel = doc[t]["ego_statuses"][-1]["ego_velocity"]
        return float(math.hypot(*vel))

    bands = {}
    for lo, hi in SPEEDS:
        ts = [t for t in toks if lo <= v0(t) < hi]
        if not ts:
            continue
        bands[f"{lo:g}-{hi:g} m/s"] = {
            "n": len(ts),
            "DAC0_rate": round(sum(t in set(dac0) for t in ts) / len(ts), 4),
            "NC0_rate": round(sum(t in set(nc0) for t in ts) / len(ts), 4),
            "arm_PDMS_x100": round(100 * st.mean(f(A[t], "score") for t in ts), 4),
            "floor_PDMS_x100": round(100 * st.mean(f(F[t], "score") for t in ts), 4),
            "median_heading_err_deg_4s": med(lat[t]["heading_err_deg_4s"] for t in ts),
            "median_xtrack_m_4s": med(lat[t]["xtrack_m_4s"] for t in ts)}
    logs = Counter(doc[t]["log_name"] for t in dac0)

    def prog(ts, i):
        return med(hk[t]["progress_raw_m"][i] for t in ts if t in hk and "progress_raw_m" in hk[t])

    out = {
        "arm": a.arm, "floor": a.floor, "n": len(toks), "restricted": bool(a.restrict_tokens),
        "seam": str(Path(a.seam)), "evidence": "MEASURED",
        "stamps": {"tier": "T1-family", "loop": "OPEN", "background": "non-reactive (logged)"},
        "paired_vs_floor": {"mean_delta_x100": round(100 * st.mean(delta), 4),
                            "median_delta_x100": round(100 * st.median(delta), 4),
                            "W": sum(x > Z for x in delta), "T": sum(abs(x) <= Z for x in delta),
                            "L": sum(x < -Z for x in delta)},
        "gate": {"arm_mean_NCxDAC": round(st.mean(f(A[t], "no_at_fault_collisions") *
                                                  f(A[t], "drivable_area_compliance")
                                                  for t in toks), 4),
                 "floor_mean_NCxDAC": round(st.mean(f(F[t], "no_at_fault_collisions") *
                                                    f(F[t], "drivable_area_compliance")
                                                    for t in toks), 4),
                 "arm_zeroed": {"NC": len(nc0), "DAC": len(dac0), "either": len(zA)},
                 "floor_zeroed_either": len(zF)},
        "on_tokens_the_arm_does_not_zero": {
            "n": len(keep),
            "arm_PDMS_x100": round(100 * st.mean(f(A[t], "score") for t in keep), 4) if keep else None,
            "floor_PDMS_x100": round(100 * st.mean(f(F[t], "score") for t in keep), 4) if keep else None},
        "by_t0_speed": bands,
        "DAC0_log_spread": {"n_logs_with_DAC0": len(logs),
                            "n_logs_total": len({doc[t]["log_name"] for t in toks}),
                            "max_DAC0_in_one_log": max(logs.values()) if logs else 0},
        "progress_median_m": {
            "_columns": "progress_raw_m = [PDM-Closed, agent] (code/navsim_v1_win.py:199)",
            "DAC0": {"agent_m": prog(dac0, 1), "pdm_closed_m": prog(dac0, 0)},
            "DAC1": {"agent_m": prog(dac1, 1), "pdm_closed_m": prog(dac1, 0)}},
        "lateral_at_4s_vs_human": {
            "_definition": ("per token, ego frame: |heading(model[7]) − heading(human[7])| and the "
                            "model's final position's lateral offset in the human's final pose "
                            "frame. Computed here from the seam + export, independent of the suite "
                            "adapter"),
            "DAC0": {"n": len(dac0),
                     "median_heading_err_deg": med(lat[t]["heading_err_deg_4s"] for t in dac0),
                     "median_xtrack_m": med(lat[t]["xtrack_m_4s"] for t in dac0),
                     "median_fde_m": med(lat[t]["fde_m_4s"] for t in dac0)},
            "DAC1": {"n": len(dac1),
                     "median_heading_err_deg": med(lat[t]["heading_err_deg_4s"] for t in dac1),
                     "median_xtrack_m": med(lat[t]["xtrack_m_4s"] for t in dac1),
                     "median_fde_m": med(lat[t]["fde_m_4s"] for t in dac1)}},
    }
    p = RAW / f"gate_decomposition_{a.arm}{('_' + a.tag) if a.tag else ''}.json"
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1)[:3500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
