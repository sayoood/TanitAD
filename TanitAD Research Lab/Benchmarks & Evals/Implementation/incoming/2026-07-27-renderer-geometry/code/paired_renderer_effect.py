"""Is the corrected re-render LOAD-BEARING in the closed loop, or just plumbed?

A frame that is threaded through the call chain but never actually applied
passes every plumbing test in silence — the exact shape of "an orthogonality
instrument that sat unmerged for 10 days because nobody re-read the README".
So this measures it: the SAME planner, the SAME episodes, the SAME anchors, the
SAME horizon, differing ONLY in the re-render (projection-aware vs the shipped
266/128 pinhole homography), compared with the **paired episode-cluster
bootstrap** (``taniteval.ci.paired_episode_cluster_bootstrap``, B=2000, unit =
val episode). ⛔ ``overlapping_holdout_se`` appears nowhere.

Reported per stratum, and decomposed:

* **corridor departure** — a purely LATERAL (cross-track) criterion: the metric
  the gate registers, and the axis that ends a drive;
* **peak \\|cross-track\\|** and **mean \\|cross-track\\|** — the lateral magnitude
  behind that rate;
* **ADE@2s** — the LONGITUDINAL-dominated companion (MEASURED at 98.6 % of the
  squared-error energy in ``LATERAL_VS_LONGITUDINAL_ANALYSIS.md``), so a change
  that shows up on one axis and not the other is visible rather than pooled.

⚠️ The planner is a pixel-sensitive PROBE with no driving merit. Its numbers are
not a result about driving; the DELTA is a result about the renderer.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from taniteval import ci as _ci
from taniteval import corridor as _corr

CORRIDOR_M = _corr.CORRIDOR_HALFWIDTH_M


def _load(p):
    return torch.load(p, weights_only=False, map_location="cpu")


def _np(x):
    return x.numpy() if torch.is_tensor(x) else np.asarray(x)


def strata_of(pw):
    hd = _np(pw["hd2s"])
    spd = _np(pw["speed"])
    junc = _corr.junction_mask(hd, _corr.JUNCTION_DEG)
    lon = (~junc) & (spd >= np.median(spd))
    return {"overall": np.ones(len(hd), bool), "junction": junc,
            "longitudinal": lon, "other": (~junc) & (~lon)}


def components(pw):
    lat = np.abs(_np(pw["lat"]))                      # [n, K] cross-track
    return {
        "corridor_departure_rate": (lat.max(axis=1) > CORRIDOR_M).astype(float),
        "peak_abs_cross_track_m": lat.max(axis=1),
        "mean_abs_cross_track_m": lat.mean(axis=1),
        "ade_0_2s_closed_loop_m": _np(pw["ade2s"]),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corrected", required=True, help="per-window .pt, "
                                                       "projection-aware warp")
    ap.add_argument("--legacy", required=True, help="per-window .pt, shipped "
                                                    "266/128 pinhole warp")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    A, B = _load(a.corrected), _load(a.legacy)
    eidA, eidB = list(A["eid"]), list(B["eid"])
    t0A, t0B = _np(A["t0"]), _np(B["t0"])
    aligned = (eidA == eidB) and bool(np.array_equal(t0A, t0B))
    if not aligned:
        raise SystemExit(
            "[paired] the two runs do NOT share windows (eid/t0 differ). A "
            "paired bootstrap on unaligned windows is invalid; re-run both arms "
            "with identical --episodes/--stride/--K.")

    cA, cB = components(A), components(B)
    out = {
        "what": ("does the projection-aware re-render CHANGE the closed loop, "
                 "on matched windows? Same planner, same anchors, same K; only "
                 "the re-render differs."),
        "estimator": "paired_episode_cluster_bootstrap (taniteval.ci)",
        "n_boot": a.n_boot,
        "windows_aligned": aligned,
        "n_windows": len(eidA), "n_episodes": len(set(eidA)),
        "horizon_K": int(A["lat"].shape[1]),
        "corridor_primary_m": CORRIDOR_M,
        "warp_corrected": A.get("_warp"), "warp_legacy": B.get("_warp"),
        "axis_note": ("corridor departure and cross-track are LATERAL; "
                      "ade_0_2s is longitudinal-dominated (98.6 % of the "
                      "squared-error energy, LATERAL_VS_LONGITUDINAL_"
                      "ANALYSIS.md). They are reported separately, never pooled."),
        "by_stratum": {},
    }
    for name, mask in strata_of(A).items():
        ix = np.flatnonzero(mask)
        if len(ix) == 0:
            out["by_stratum"][name] = {"n_windows": 0,
                                       "_": "NOT MEASURED (empty stratum)"}
            continue
        eid = [eidA[i] for i in ix]
        node = {"n_windows": int(len(ix)), "n_episodes": len(set(eid))}
        for k in cA:
            node[k] = {
                "corrected": _ci.episode_cluster_bootstrap(
                    cA[k][ix], eid, n_boot=a.n_boot, seed=0),
                "legacy": _ci.episode_cluster_bootstrap(
                    cB[k][ix], eid, n_boot=a.n_boot, seed=0),
                "paired_delta_corrected_minus_legacy":
                    _ci.paired_episode_cluster_bootstrap(
                        cA[k][ix], cB[k][ix], eid, n_boot=a.n_boot, seed=0),
            }
        out["by_stratum"][name] = node

    ov = out["by_stratum"]["overall"]["corridor_departure_rate"]
    dl = ov["paired_delta_corrected_minus_legacy"]
    any_nonzero = any(
        float(node[k]["paired_delta_corrected_minus_legacy"]["delta"]) != 0.0
        for node in out["by_stratum"].values() if "n_windows" in node
        and node["n_windows"] for k in cA)
    out["headline"] = {
        "corridor_departure_rate_overall": ov,
        "paired_delta": dl["delta"], "ci": [dl["lo"], dl["hi"]],
        "separated": bool(dl["separated"]),
        "renderer_is_load_bearing": bool(any_nonzero),
        "_falsifier": ("a paired delta of exactly 0.0 across every stratum and "
                       "every component would mean the frame is threaded but "
                       "NOT applied — the failure this file exists to catch."),
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out["headline"], indent=2, default=str))
    print(f"[paired] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
