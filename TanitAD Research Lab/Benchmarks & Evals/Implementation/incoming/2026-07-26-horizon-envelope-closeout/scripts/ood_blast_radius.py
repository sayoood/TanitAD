#!/usr/bin/env python3
"""ood_blast_radius.py — what the VOID criterion decided, and what the real rule says.

THE VOID CRITERION (C13), STATED EXACTLY
----------------------------------------
``OODMap.ratio_arr`` = ``1 + clip((interp(|dlat|) - base)/base, 0, inf)
                         + clip((interp(|dyaw|) - base)/base, 0, inf)``
and ``np.interp`` **CLAMPS** at the envelope edge. The interpolated curves are
piecewise linear through the P1 sweep points, so the ratio's **supremum over ALL
possible inputs** is a constant that can be computed from the envelope JSON alone,
with no model, no rollout and no GPU:

    sup(ratio) = 1 + max(0, (max(lat_ade) - base)/base)
                   + max(0, (max(yaw_ade) - base)/base)

This script computes it, then re-adjudicates every committed artifact that carries
an OOD node, using ``taniteval.ood.readjudicate`` (the packaged implementation of
E1a's FULL disjunction) — no second implementation, so the two cannot drift.

WHAT IT REPORTS
---------------
1. the exact supremum, and whether the two LIVE constants
   (``e1b_eval.py:403`` and ``e1c_common.py:34``, both ``ood <= 1.30``) and
   ``ood.RATIO_EXTRAPOLATION_X`` (1.5) are **reachable at all**;
2. per artifact / per stratum: the verdict CLASS before and after, so a flip is
   visible as a flip and a re-wording is not mistaken for one.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[6]   # …/<repo>/TanitAD Research Hub/…
sys.path.insert(0, str(REPO / "taniteval"))
sys.path.insert(0, str(REPO / "stack"))

from taniteval import ood as OOD  # noqa: E402

ENVELOPE_JSON = (REPO / "TanitAD Research Hub" / "Benchmarks & Eval" /
                 "Implementation" / "incoming" /
                 "2026-07-23-lowood-lanekeeping-refc" / "lowood_flagship_ci.json")

LIVE_CONSTANTS = [
    {"file": "TanitAD Research Hub/Architecture & Inference/Implementation/"
             "incoming/2026-07-25-e1b-failure-gated-clsft/scripts/e1b_eval.py",
     "line": 403, "expr": "c_ood_in_band: bool(ood_ft <= 1.30 + 1e-9)",
     "quantity": "closed_loop_K185.overall.ood_peak.ft.mean",
     "threshold": 1.30, "sense": "<=", "role": "GUARDRAIL c — a FAIL turns a "
     "SUCCESS verdict into BOUND"},
    {"file": "TanitAD Research Hub/Architecture & Inference/Implementation/"
             "incoming/2026-07-26-e1c-heldout-gated-clsft/scripts/e1c_common.py",
     "line": 34, "expr": "OOD_BAND = 1.30  -> Gc_ood_in_band: ood_ft <= OOD_BAND",
     "quantity": "closed-loop OOD peak ratio (ft, overall)",
     "threshold": 1.30, "sense": "<=", "role": "gate Gc — 'must hold'"},
    {"file": "taniteval/taniteval/ood.py", "line": 77,
     "expr": "RATIO_EXTRAPOLATION_X = 1.5 -> criterion_1_ratio_over_1p5",
     "quantity": "ood_peak_ratio.mean", "threshold": 1.5, "sense": ">",
     "role": "clause 1 of E1a's disjunction"},
]


def supremum(env_json):
    d = json.loads(Path(env_json).read_text(encoding="utf-8"))
    base = d["baseline_real_frames"]["mean"]
    ly = np.array([r["ade2s_ci"]["mean"] for r in d["conditions"]["lat"]])
    yy = np.array([r["ade2s_ci"]["mean"] for r in d["conditions"]["yaw"]])
    lx = np.array([r["amount"] for r in d["conditions"]["lat"]])
    yx = np.array([r["amount"] for r in d["conditions"]["yaw"]])
    ex_l = max(0.0, float((ly.max() - base) / base))
    ex_y = max(0.0, float((yy.max() - base) / base))
    return {"envelope_json": str(env_json), "baseline_ade2s": base,
            "lat_amounts_m": lx.tolist(), "lat_ade2s": ly.tolist(),
            "yaw_amounts_deg": yx.tolist(), "yaw_ade2s": yy.tolist(),
            "lat_max_excess": round(ex_l, 6), "yaw_max_excess": round(ex_y, 6),
            "SUPREMUM_of_ratio_arr": round(1.0 + ex_l + ex_y, 6),
            "_derivation": ("ratio = 1 + clip((interp_lat - base)/base,0,inf) + "
                            "clip((interp_yaw - base)/base,0,inf); np.interp is "
                            "piecewise linear through the sweep points and CLAMPS "
                            "outside them, so sup(interp) = max(sweep y-values). "
                            "The bound is EXACT and holds for every input, "
                            "including |dlat| = 1e9 m.")}


def walk(node, path=""):
    """Yield every dict that carries an EXTRAPOLATION verdict or an ood_peak."""
    if isinstance(node, dict):
        if ("EXTRAPOLATION_VERDICT" in node
                or "EXTRAPOLATION_frac_windows_any_step_out_of_envelope" in node
                or "ood_peak_ratio" in node or "ood_peak" in node):
            yield path or "<root>", node
        for k, v in node.items():
            yield from walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, f"{path}[{i}]")


def normalise(node):
    """Some emitters store the ratio as ``ood_peak: {ft: {...}, base: {...}}``
    rather than ``ood_peak_ratio``. Map to the readjudicator's field names
    without mutating the source."""
    out = dict(node)
    if "ood_peak_ratio" not in out and isinstance(out.get("ood_peak"), dict):
        op = out["ood_peak"]
        cand = op.get("ft") or op.get("base") or op
        if isinstance(cand, dict) and "mean" in cand:
            out["ood_peak_ratio"] = cand
    return out


def main():
    sup = supremum(ENVELOPE_JSON)
    S = sup["SUPREMUM_of_ratio_arr"]
    res = {"_experiment": "C13 blast radius — the OOD ratio criterion's exact "
                          "supremum, and every committed artifact re-adjudicated",
           "_evidence_class": "MEASURED (ours; artifact = this JSON)",
           "_readjudicator": "taniteval.ood.readjudicate (E1a's FULL disjunction)",
           "supremum": sup, "live_constants": []}
    for c in LIVE_CONSTANTS:
        reachable = (S > c["threshold"]) if c["sense"] == ">" \
            else not (S <= c["threshold"])
        res["live_constants"].append({
            **c, "supremum": S,
            "margin_to_threshold": round(c["threshold"] - S, 6),
            "CAN_THE_TEST_EVER_CHANGE_ITS_ANSWER": bool(reachable),
            "verdict": ("VOID — the test is decided before the model runs"
                        if not reachable else "live")})

    files = sorted(
        p for p in (REPO / "TanitAD Research Hub").rglob("*.json")
        if "horizon-envelope-closeout" not in p.as_posix())
    rows = []
    for p in files:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                            # noqa: BLE001
            continue
        for path, node in walk(d):
            n = normalise(node)
            if n.get("EXTRAPOLATION_VERDICT") is None \
                    and n.get("EXTRAPOLATION_frac_windows_any_step_out_of_envelope") is None:
                continue
            try:
                ra = OOD.readjudicate(n, _path=f"{p.name}:{path}")
                err = None
            except OOD.EnvelopeVerdictError as e:
                ra, err = None, str(e)[:300]
            fw = n.get("EXTRAPOLATION_frac_windows_any_step_out_of_envelope")
            fs = n.get("EXTRAPOLATION_frac_steps_any")
            fl = n.get("EXTRAPOLATION_frac_steps_lat_over_3m")
            peak = n.get("ood_peak_ratio")
            pm = (peak.get("mean") if isinstance(peak, dict) else peak)
            rows.append({
                "file": p.relative_to(REPO).as_posix(), "node": path,
                "K": n.get("horizon_K"),
                "n_windows": n.get("n_windows"),
                "verdict_before": n.get("EXTRAPOLATION_VERDICT"),
                "class_before": OOD.verdict_class(n.get("EXTRAPOLATION_VERDICT")),
                "verdict_after": ra["EXTRAPOLATION_VERDICT"] if ra else None,
                "class_after": ra["_class_after"] if ra else None,
                "CLASS_FLIPPED": bool(ra["_class_changed"]) if ra else None,
                "readjudicator_raised": err,
                "frac_windows_out": fw, "frac_steps_any": fs,
                "frac_steps_lat_over_3m": fl,
                "ood_peak_ratio_mean": pm,
                "ratio_criterion_could_fire": (bool(pm is not None
                                                    and S > 1.5)),
                "old_1p30_test_would_say": (None if pm is None else
                                            ("IN BAND (pass)" if pm <= 1.30 + 1e-9
                                             else "OUT OF BAND (fail)")),
            })
    res["artifacts"] = rows
    res["summary"] = {
        "n_nodes": len(rows),
        "n_class_flipped": sum(1 for r in rows if r["CLASS_FLIPPED"]),
        "n_readjudicator_raised": sum(1 for r in rows
                                      if r["readjudicator_raised"]),
        "n_with_ratio": sum(1 for r in rows
                            if r["ood_peak_ratio_mean"] is not None),
        "n_old_1p30_would_pass": sum(
            1 for r in rows if r["old_1p30_test_would_say"] == "IN BAND (pass)"),
        "n_old_1p30_would_fail": sum(
            1 for r in rows
            if r["old_1p30_test_would_say"] == "OUT OF BAND (fail)"),
        "files_touched": sorted({r["file"] for r in rows}),
    }
    out = (Path(__file__).resolve().parent.parent / "artifacts" /
           "ood_blast_radius.json")
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"SUPREMUM of ratio_arr = {S}")
    for c in res["live_constants"]:
        print(f"  {c['file'].split('/')[-1]}:{c['line']}  "
              f"'{c['expr'][:48]}'  -> {c['verdict']}  "
              f"(margin {c['margin_to_threshold']:+.6f})")
    print(f"\nnodes re-adjudicated : {res['summary']['n_nodes']}")
    print(f"  class FLIPPED      : {res['summary']['n_class_flipped']}")
    print(f"  readjudicator RAISED: {res['summary']['n_readjudicator_raised']}")
    print(f"  old 1.30 test PASS : {res['summary']['n_old_1p30_would_pass']}"
          f" / FAIL: {res['summary']['n_old_1p30_would_fail']}")
    for r in rows:
        if r["CLASS_FLIPPED"] or r["readjudicator_raised"]:
            print(f"  * {r['file']}:{r['node']} K={r['K']} "
                  f"{r['class_before']} -> {r['class_after']} "
                  f"(winOUT {r['frac_windows_out']}, ratio "
                  f"{r['ood_peak_ratio_mean']})")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
