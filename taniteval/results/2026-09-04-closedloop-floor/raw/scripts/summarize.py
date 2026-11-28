#!/usr/bin/env python3
"""Build FLOOR_SUMMARY.json + the RESULT tables from the banked cl_metrics output."""
import json, sys
from pathlib import Path

M = Path(sys.argv[1])           # metrics dir

# metric -> (family, higher_is_better, unit)
SPEC = [
    ("ADE",          "ade_0_2s",                  False, "m"),
    ("ADE",          "dist_to_gt_traj_m",         False, "m"),
    ("LONGITUDINAL", "abs_target_speed_err_ms",   False, "m/s"),
    ("LONGITUDINAL", "abs_executed_speed_err_ms", False, "m/s"),
    ("LONGITUDINAL", "along_track_ade_m",         False, "m"),
    ("LONGITUDINAL", "real_lead_time_gap_s",      True,  "s"),
    ("LATERAL",      "lateral_ade_m",             False, "m"),
    ("LATERAL",      "heading_err_rad",           False, "rad"),
    ("LATERAL",      "curvature_err_1pm",         False, "1/m"),
    ("LATERAL",      "yawrate_err_rads",          False, "rad/s"),
    ("LATERAL",      "cross_track_abs_m",         False, "m"),
    ("TACTICAL",     "manoeuvre_plan_eq_logged",  True,  "rate"),
    ("TACTICAL",     "manoeuvre_head_eq_logged",  True,  "rate"),
    ("STRATEGIC",    "route_head_eq_logged",      True,  "rate"),
    ("STRATEGIC",    "route_corridor_departure_rate", False, "rate"),
]
LEVEL_PATH = {
    "ade_0_2s": ("ADE", "ade_0_2s"),
    "dist_to_gt_traj_m": ("ADE", "dist_to_gt_traj_m"),
    "abs_target_speed_err_ms": ("LONGITUDINAL", "abs_target_speed_err_ms"),
    "abs_executed_speed_err_ms": (None, None),
    "along_track_ade_m": ("LONGITUDINAL", "along_track_ade_m"),
    "real_lead_time_gap_s": (None, None),
    "lateral_ade_m": ("LATERAL", "lateral_ade_m"),
    "heading_err_rad": ("LATERAL", "heading_err_rad"),
    "curvature_err_1pm": ("LATERAL", "curvature_err_1pm"),
    "yawrate_err_rads": ("LATERAL", "yawrate_err_rads"),
    "cross_track_abs_m": ("LATERAL", "cross_track_abs_m"),
    "manoeuvre_plan_eq_logged": ("TACTICAL", "manoeuvre_plan_eq_logged"),
    "manoeuvre_head_eq_logged": ("TACTICAL", "head_eq_logged"),
    "route_head_eq_logged": ("STRATEGIC", "route_head_eq_logged"),
    "route_corridor_departure_rate": ("STRATEGIC", "route_corridor_departure_rate"),
}

CONTRASTS = [
    ("V3_refcv3_vs_ha0",     "refcv3",      "closed", "2026-09-04 v3 panel"),
    ("V3_refcbase_vs_ha0",   "refc-base",   "closed", "2026-09-04 v3 panel"),
    ("V3_flagshipv1_vs_ha0", "flagship-v1", "closed", "2026-09-04 v3 panel"),
    ("HQ_flagshipv1_vs_ha0", "flagship-v1", "closed", "2026-08-03 HQ panel (banked)"),
    ("HQ_refcbase_vs_ha0",   "refc-base",   "closed", "2026-08-03 HQ panel (banked)"),
    ("FL_ha_vs_ha0",         "cl_ha",       "closed", "floor internal"),
    ("FL_haext_vs_ha0",      "cl_ha0_ext",  "closed", "floor internal"),
    ("OL_refcv3_vs_ha0",     "refcv3",      "open",   "2026-09-04 open-loop sweep"),
    ("OL_refcbase_vs_ha0",   "refc-base",   "open",   "2026-09-04 open-loop sweep"),
    ("OL_flagshipv1_vs_ha0", "flagship-v1", "open",   "2026-09-04 open-loop sweep"),
    ("OL_ha_vs_ha0",         "cl_ha",       "open",   "floor internal"),
    ("OL_haext_vs_ha0",      "cl_ha0_ext",  "open",   "floor internal"),
    # --- against the STRONGEST trivial floors -----------------------------------
    ("V3_refcv3_vs_haext",     "refcv3",      "closed", "2026-09-04 v3 panel"),
    ("V3_refcbase_vs_haext",   "refc-base",   "closed", "2026-09-04 v3 panel"),
    ("V3_flagshipv1_vs_haext", "flagship-v1", "closed", "2026-09-04 v3 panel"),
    ("V3_refcv3_vs_ha",        "refcv3",      "closed", "2026-09-04 v3 panel"),
    ("V3_refcbase_vs_ha",      "refc-base",   "closed", "2026-09-04 v3 panel"),
    ("OL_refcv3_vs_haext",     "refcv3",      "open",   "2026-09-04 open-loop sweep"),
    ("OL_refcbase_vs_haext",   "refc-base",   "open",   "2026-09-04 open-loop sweep"),
    ("OL_flagshipv1_vs_haext", "flagship-v1", "open",   "2026-09-04 open-loop sweep"),
    ("OL_refcv3_vs_ha",        "refcv3",      "open",   "2026-09-04 open-loop sweep"),
    ("OL_refcbase_vs_ha",      "refc-base",   "open",   "2026-09-04 open-loop sweep"),
]
CONTROLS = [("CTRL_repro_ha0", "DETERMINISM — cl_ha0 re-run with identical flags"),
            ("CTRL_morn_ha0",  "RENDER INDEPENDENCE — cl_ha0 under the MORNING render")]


def lvl(fams, metric):
    fam, key = LEVEL_PATH.get(metric, (None, None))
    if fam is None:
        return None
    b = fams.get(fam, {}).get(key)
    if not isinstance(b, dict):
        return None
    if "mean" not in b:
        return {"n": b.get("n", 0), "reason": b.get("reason") or b.get("note")}
    return {k: b[k] for k in ("mean", "lo", "hi", "n_used", "n_episodes") if k in b}


out = {"levels": {}, "margins": {}, "controls": {}, "provenance": {}}
for tag, arm, tier, src in CONTRASTS:
    p = M / f"{tag}.json"
    if not p.exists():
        out["margins"][tag] = {"MISSING": True}
        continue
    d = json.loads(p.read_text())
    A, B = d["arm_A"], d["arm_B"]
    key = f"{tier}:{A['name']}"
    out["levels"].setdefault(key, {"tier": tier, "arm": A["name"], "source": src,
                                   "n_windows": A["n_windows"],
                                   "n_clusters": A["n_clusters"],
                                   "metrics": {m: lvl(A["families"], m)
                                               for _, m, _, _ in SPEC}})
    fk = f"{tier}:{B['name']}"
    out["levels"].setdefault(fk, {"tier": tier, "arm": B["name"], "source": "FLOOR",
                                  "n_windows": B["n_windows"],
                                  "n_clusters": B["n_clusters"],
                                  "metrics": {m: lvl(B["families"], m)
                                              for _, m, _, _ in SPEC}})
    pr = d["paired_A_minus_B"]
    out["margins"][tag] = {
        "arm": A["name"], "floor": B["name"], "tier": tier, "source": src,
        "paired_n_windows": d["paired_n_windows"],
        "n_clusters": A["n_clusters"],
        "families": {fam: {} for fam, _, _, _ in SPEC}}
    for fam, m, hib, unit in SPEC:
        b = pr.get(m, {})
        n = b.get("n_used", b.get("n", 0))
        if not n:
            out["margins"][tag]["families"][fam][m] = {
                "n": 0, "reason": b.get("reason", "no jointly finite windows — the "
                                        "floor exposes no such head / no lead in frame"),
                "unit": unit, "higher_is_better": hib}
            continue
        sep = bool(b.get("separated"))
        verdict = ("floor WINS" if (sep and ((b["delta"] > 0) != hib)) else
                   "arm WINS" if sep else "indistinguishable")
        out["margins"][tag]["families"][fam][m] = {
            "delta": b["delta"], "lo": b["lo"], "hi": b["hi"],
            "separated": sep, "n": n, "unit": unit, "higher_is_better": hib,
            "verdict_vs_floor": verdict,
            "estimator": b.get("estimator")}

for tag, label in CONTROLS:
    p = M / f"{tag}.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    pr = d["paired_A_minus_B"]
    nz = {k: v for k, v in pr.items()
          if v.get("n_used") and (v.get("delta") != 0.0 or v.get("ci95", 0.0) != 0.0)}
    scored = [k for k, v in pr.items() if v.get("n_used")]
    out["controls"][tag] = {
        "label": label, "paired_n_windows": d["paired_n_windows"],
        "n_metrics_scored": len(scored),
        "expected": "every scored metric delta == 0.0 with a ZERO-WIDTH CI",
        "measured_n_nonzero": len(nz),
        "nonzero_entries": nz,
        "verdict": "PASS" if not nz else "FAIL"}

raw = M / "RAW_CONTROLS.json"
if raw.exists():
    out["raw_controls"] = json.loads(raw.read_text())

Path(sys.argv[2]).write_text(json.dumps(out, indent=2))
print("wrote", sys.argv[2])
for k, v in out["margins"].items():
    a = v.get("families", {}).get("ADE", {}).get("ade_0_2s", {})
    if "delta" in a:
        print(f"{k:24s} {v['tier']:6s} n={v['paired_n_windows']:3d} "
              f"ade {a['delta']:+.4f} [{a['lo']:+.4f},{a['hi']:+.4f}] "
              f"{a['verdict_vs_floor']}")
for k, v in out["controls"].items():
    print(f"CONTROL {k:18s} {v['verdict']} scored={v['n_metrics_scored']} "
          f"nonzero={v['measured_n_nonzero']} n={v['paired_n_windows']}")
