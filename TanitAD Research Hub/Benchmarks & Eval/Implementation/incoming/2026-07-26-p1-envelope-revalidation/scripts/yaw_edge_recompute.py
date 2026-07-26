"""Recompute the out-of-envelope fractions as a FUNCTION OF THE YAW EDGE.

Answers pre-registration §1.5: *if the LATERAL clause alone already fires at
K = 20, then widening the YAW envelope -- to any value, including infinity --
cannot make K = 20 a MEASUREMENT.*  That is arithmetic, not a prediction, and it
is settled here from the staged per-window dumps with NO GPU and NO model.

Inputs  : `2026-07-26-horizon-envelope-closeout/artifacts/perwindow_K*.pt`
          (lat / yaw are [n_windows, n_steps] peak-deviation traces; `hd2s` is the
          2 s net heading change that fixes the strata across horizons).
Rule    : the PACKAGED `taniteval.ood` / `taniteval.corridor` -- deliberately NOT
          a second implementation. `closedloop.py` warns that a second copy of
          the rule drifting from the first "is how overlapping_holdout_se
          survived"; `test_ood_guard.py` exists to stop exactly that.

Every number this emits is MEASURED (ours), and the artifact is
`artifacts/yaw_edge_recompute.json`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[5]
sys.path.insert(0, str(_REPO / "taniteval"))

from taniteval.corridor import (ENV_LAT_MAX, ENV_YAW_MAX,  # noqa: E402
                                JUNCTION_DEG, junction_mask)

CLOSEOUT = (_REPO / "TanitAD Research Hub" / "Benchmarks & Eval" / "Implementation"
            / "incoming" / "2026-07-26-horizon-envelope-closeout" / "artifacts")

# Candidate yaw edges. `inf` is the decisive one: it is the LIMIT of any possible
# widening of the yaw arm, so whatever survives at `inf` is unreachable by
# re-validating yaw at all.
YAW_EDGES = [12.0, 12.14, 15.0, 15.47, 17.88, 20.0, 25.0, 25.7, 26.41, 30.0,
             40.0, 50.0, 60.0, 90.0, 120.0, 180.0, float("inf")]
# 15.47 [12.14, 17.88] is the MEASURED usable edge (IDF = 0.5) and 26.41 the
# MEASURED information-destruction edge, both from `yaw_edge_analysis.py`.
# They are in the grid so the post-revalidation fractions are MEASURED rather
# than interpolated.
HORIZONS = [20, 60, 70, 185]


def strata_of(hd2s, speed):
    """clhorizon.py:399-402 -- the stratification, held fixed across horizons."""
    junc = junction_mask(hd2s, JUNCTION_DEG)
    long_ = (~junc) & (speed >= np.median(speed))
    return {"overall": np.ones(len(hd2s), bool), "junction": junc,
            "longitudinal": long_, "other": (~junc) & (~long_)}


def fractions(lat, yaw, lat_edge, yaw_edge):
    """`taniteval.ood.envelope_fractions`, generalised to a candidate edge.

    Decomposed so the two clauses can never be confused for one another --
    the whole finding exists because an undecomposed statistic hid the axis.
    """
    out_lat = lat > lat_edge
    out_yaw = yaw > yaw_edge
    out_any = out_lat | out_yaw
    w_lat = out_lat.any(1)
    w_yaw = out_yaw.any(1)
    return {
        "frac_steps_lat_over_edge": round(float(out_lat.mean()), 5),
        "frac_steps_yaw_over_edge": round(float(out_yaw.mean()), 5),
        "frac_steps_any": round(float(out_any.mean()), 5),
        "frac_windows_any_step_out_of_envelope": round(float(out_any.any(1).mean()), 4),
        "frac_windows_out_via_lat": round(float(w_lat.mean()), 4),
        "frac_windows_out_via_yaw": round(float(w_yaw.mean()), 4),
        "frac_windows_out_via_lat_ONLY": round(float((w_lat & ~w_yaw).mean()), 4),
        "frac_windows_out_via_yaw_ONLY": round(float((w_yaw & ~w_lat).mean()), 4),
        "frac_windows_out_via_BOTH": round(float((w_lat & w_yaw).mean()), 4),
    }


def verdict_of(frac_windows_out, frac_steps_any):
    """`ood._verdict_string` with clause 1 VOID (sup(ratio) = 1.298888 < 1.5)."""
    if frac_steps_any > 0.5 or frac_windows_out > 0.5:
        return "EXTRAPOLATION"
    if frac_windows_out > 0.0 or frac_steps_any > 0.0:
        return "PARTIAL_EXTRAPOLATION"
    return "MEASUREMENT"


def main():
    out = {
        "_what": "out-of-envelope fractions as a function of the YAW edge",
        "_evidence_class": "MEASURED (ours) -- recomputed from the staged "
                           "per-window dumps of the 2026-07-26 K-sweep",
        "_source_dumps": str(CLOSEOUT),
        "_rule": "taniteval.ood disjunction; clause 1 (ratio>1.5) is VOID "
                 "(sup(ratio_arr)=1.298888), so clause 2 carries every verdict",
        "_shipped_envelope": {"lat_max_m": ENV_LAT_MAX, "yaw_max_deg": ENV_YAW_MAX},
        "_junction_deg": JUNCTION_DEG,
        "horizons": {},
    }
    for K in HORIZONS:
        p = CLOSEOUT / f"perwindow_K{K}.pt"
        d = torch.load(str(p), map_location="cpu", weights_only=False)
        lat = np.asarray(d["lat"], dtype=np.float64)
        yaw = np.asarray(d["yaw"], dtype=np.float64)
        hd2s = np.asarray(d["hd2s"], dtype=np.float64)
        spd = np.asarray(d["speed"], dtype=np.float64)
        eid = list(d["eid"])
        st = strata_of(hd2s, spd)
        rec = {"n_windows": int(lat.shape[0]), "n_steps": int(lat.shape[1]),
               "n_clusters": int(len(set(eid))), "horizon_s": round(K * 0.1, 2),
               "strata": {}}
        for name, mask in st.items():
            ix = np.flatnonzero(mask)
            if ix.size < 2:
                rec["strata"][name] = {"n_windows": int(ix.size),
                                       "_note": "too small to adjudicate"}
                continue
            L, Y = lat[ix], yaw[ix]
            peak_lat = L.max(1)
            peak_yaw = Y.max(1)
            # The pre-registered question, in closed form.
            lat_only_at_inf = fractions(L, Y, ENV_LAT_MAX, float("inf"))
            s = {
                "n_windows": int(ix.size),
                "n_clusters": int(len(set(eid[i] for i in ix))),
                "peak_lat_m": {"p50": round(float(np.percentile(peak_lat, 50)), 4),
                               "p90": round(float(np.percentile(peak_lat, 90)), 4),
                               "max": round(float(peak_lat.max()), 4)},
                "peak_yaw_deg": {"p50": round(float(np.percentile(peak_yaw, 50)), 4),
                                 "p90": round(float(np.percentile(peak_yaw, 90)), 4),
                                 "max": round(float(peak_yaw.max()), 4)},
                # ---- THE DECISIVE NUMBER -------------------------------------
                "YAW_EDGE_INFINITY": {
                    "frac_windows_out": lat_only_at_inf[
                        "frac_windows_any_step_out_of_envelope"],
                    "verdict": verdict_of(
                        lat_only_at_inf["frac_windows_any_step_out_of_envelope"],
                        lat_only_at_inf["frac_steps_any"]),
                    "_meaning": "the LIMIT of widening the yaw arm. A non-zero "
                                "value here is UNREACHABLE by any yaw "
                                "re-validation, because it is the LATERAL "
                                "clause firing alone.",
                },
                # the yaw edge that would zero the YAW clause on its own
                "yaw_edge_needed_to_zero_yaw_clause_deg": round(
                    float(peak_yaw.max()), 4),
                "lat_edge_needed_to_zero_lat_clause_m": round(
                    float(peak_lat.max()), 4),
                "sweep": {},
            }
            for e in YAW_EDGES:
                f = fractions(L, Y, ENV_LAT_MAX, e)
                f["verdict"] = verdict_of(
                    f["frac_windows_any_step_out_of_envelope"], f["frac_steps_any"])
                s["sweep"]["inf" if np.isinf(e) else f"{e:g}"] = f
            rec["strata"][name] = s
        out["horizons"][str(K)] = rec

    dest = _HERE.parent / "artifacts" / "yaw_edge_recompute.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {dest}")

    # ---- the headline, printed so it cannot be missed ----------------------- #
    print("\n=== PRE-REGISTERED QUESTION (§1.5): can widening YAW alone rescue a horizon? ===")
    print(f"{'K':>4} {'stratum':<13} {'n_win':>6} "
          f"{'@12deg(shipped)':>16} {'@30deg':>9} {'@YAW=INF':>10}  verdict@INF")
    for K in HORIZONS:
        for name in ("overall", "junction", "longitudinal", "other"):
            s = out["horizons"][str(K)]["strata"].get(name, {})
            if "sweep" not in s:
                continue
            print(f"{K:>4} {name:<13} {s['n_windows']:>6} "
                  f"{s['sweep']['12']['frac_windows_any_step_out_of_envelope']:>16.4f} "
                  f"{s['sweep']['30']['frac_windows_any_step_out_of_envelope']:>9.4f} "
                  f"{s['YAW_EDGE_INFINITY']['frac_windows_out']:>10.4f}"
                  f"  {s['YAW_EDGE_INFINITY']['verdict']}")


if __name__ == "__main__":
    main()
