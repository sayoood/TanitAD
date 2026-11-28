"""e0h_read.py -- reads the E0h runs (F:/r9e0/e0h/runs/<tag>/MAPPATH_RESULT.json + SYNTH_RECORD.json) and writes E0H_RESULT.{json,md}.  No new measurement: every cell is read from a run's JSON.

    python -X utf8 e0h_read.py --runs F:/r9e0/e0h/runs --out-dir <dir>

Frozen readout (E0h_CARD.md):  rate(p) = the planner's S6c sidewalk rate under MAPPATH's cross-fitted soft cost ('MAPSOFT-G-XF', off_drivable) when its map is GT + calibrated false flags at corridor
precision p.  BASE = the native plan's rate (same file).  p_half = the smallest precision at which rate <= BASE / 2 (linear interpolation between the bracketing grid points);
p_MAP1 = the same for the MAP-1 bar 0.098.  CONTROLS (a failed control voids the dose readout): K-P1 the p = 1 run recovers >= 80 % of the banked GT-T1 gain (GT-T1 0.0219 vs BASE 0.1792);
K-ALL the constant-map run reads the BASE rate exactly (4 dp) with a [0, 0] paired interval; K-CAL realized corridor precision per band within 0.02 of p.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GT_T1_BANKED, BASE_BANKED = 0.0219, 0.1792          # MAPPATH VERDICT.md (S6c, progress-matched fan), INHERITED -- re-read from the p1 run's own GT-T1 / BASE below
P_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def interp_cross(ps, rates, level):
    """smallest p whose rate <= level, linear interpolation between the bracketing points; ps ascending.  None if never reached; ps[0] if already reached at the first point (a bound, not a value)."""
    for i, (p, r) in enumerate(zip(ps, rates)):
        if r <= level:
            if i == 0:
                return {"p": p, "kind": "at_or_below_first_grid_point"}
            p0, r0 = ps[i - 1], rates[i - 1]
            return {"p": p0 + (r0 - level) / (r0 - r) * (p - p0), "kind": "interpolated", "bracket": [p0, p]}
    return {"p": None, "kind": "not_reached_on_grid"}


def load(run):
    j = json.loads((Path(run) / "MAPPATH_RESULT.json").read_text(encoding="utf-8"))
    s = json.loads((Path(run) / "SYNTH_RECORD.json").read_text(encoding="utf-8"))
    return j, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    R = Path(a.runs)
    out = {"tool": "e0h_read.py", "tier": "T1 self-action open loop on logged eval139 frames (not driving performance); S6c; one training seed; one inference seed; estimator: paired episode-cluster "
           "bootstrap B 2000 seed 0 over EPISODES (another draw of episodes only; the false-flag placement seed is a separate draw, not resampled)"}
    rows = {}
    for d in sorted(R.iterdir()):
        if not (d / "MAPPATH_RESULT.json").exists():
            continue
        j, s = load(d)
        rt = j["rates_S6c"]
        rows[d.name] = {"mode": s["mode"], "p": s["p"], "stamp_m": s["stamp_m"], "rate_soft": rt["MAPSOFT-G-XF"]["off_drivable"], "rate_base": rt["BASE"]["off_drivable"],
                        "rate_gt_t1": rt["GT-T1"]["off_drivable"], "vs_BASE": j["cells"]["M1_sidewalk"],
                        "fits": {k: [v["r0"], v["lam_s"], v["lam_e"]] for k, v in j["fits"]["MAPSOFT-G-XF"].items()},
                        "realized_precision": {b: v["precision"] for b, v in s["realized_corridor_precision"].items()}}
    out["runs"] = rows
    base = next(iter(rows.values()))["rate_base"] if rows else None
    ctl = {}
    if "p1" in rows:
        g, r1 = rows["p1"]["rate_gt_t1"], rows["p1"]["rate_soft"]
        share = (base - r1) / (base - g) if base != g else None
        ctl["K_P1_recovers_80pct_of_GT_T1_gain"] = {"base": base, "gt_t1": g, "rate_p1": r1, "share": share, "pass": bool(share is not None and share >= 0.8)}
    if "all" in rows:
        ctl["K_ALL_constant_map_equals_BASE"] = {"rate_all": rows["all"]["rate_soft"], "base": base, "pass": bool(round(rows["all"]["rate_soft"], 4) == round(base, 4))}
    cal = {k: v["realized_precision"] for k, v in rows.items() if v["mode"] == "dose"}
    ctl["K_CAL_realized_within_0.02_of_p"] = {"pass": bool(all(abs(pr - rows[k]["p"]) <= 0.02 for k, v in cal.items() for pr in v.values() if pr is not None))}
    out["controls"] = ctl
    pts = sorted((v["p"], v["rate_soft"]) for k, v in rows.items() if v["mode"] == "dose" and v["stamp_m"] == 1.0)
    if pts and base is not None:
        ps, rs = [p for p, _ in pts], [r for _, r in pts]
        if "p1" in rows:
            ps.append(1.0)
            rs.append(rows["p1"]["rate_soft"])
        out["dose_curve"] = [{"p": p, "rate": r} for p, r in zip(ps, rs)]
        out["p_half"] = interp_cross(ps, rs, base / 2)
        out["p_MAP1_rate_le_0.098"] = interp_cross(ps, rs, 0.098)
    out["stamp_sensitivity_p0.70"] = {k: v["rate_soft"] for k, v in rows.items() if k.startswith("p0.70")}
    out["VOID"] = bool(any(not v["pass"] for v in ctl.values()))
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(a.out_dir) / "E0H_RESULT.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    L = ["# E0h result (mechanical, from the run JSONs)", "", out["tier"], "", f"BASE sidewalk rate {base}", "", "| run | p | stamp m | MAPSOFT-G-XF sidewalk | vs BASE (paired) | realized corridor precision |", "|---|---|---|---|---|---|"]
    for k, v in rows.items():
        c = v["vs_BASE"]
        L.append(f"| {k} | {v['p']} | {v['stamp_m']} | {v['rate_soft']:.4f} | {c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] | {json.dumps({b: (round(x, 3) if x else x) for b, x in v['realized_precision'].items()})} |")
    L += ["", f"controls: {json.dumps(ctl)}", f"p_half: {json.dumps(out.get('p_half'))}", f"p for MAP-1 (<= 0.098): {json.dumps(out.get('p_MAP1_rate_le_0.098'))}", f"VOID: {out['VOID']}"]
    (Path(a.out_dir) / "E0H_RESULT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[-4:]))


if __name__ == "__main__":
    main()
