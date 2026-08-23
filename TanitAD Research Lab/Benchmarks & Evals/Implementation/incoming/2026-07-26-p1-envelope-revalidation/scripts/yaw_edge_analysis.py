"""Locate the YAW fidelity edge, with a cluster-bootstrap interval on the EDGE.

P1 never defined a fidelity criterion (§2.2: the grid simply stopped at 12 deg),
so one has to be defined here. The requirement from C13 is that it can FAIL and
that the estimator can reach the failing value.

    INFORMATION DESTRUCTION FRACTION
        IDF(psi) = [ADE(psi) - ADE(0)] / [ADE_floor - ADE(0)]

    where ADE_floor = min over the destroyed-observation controls -- the ADE the
    model achieves with NO scene information at all, falling back on the action
    sequence and speed channel.

        IDF = 0   the warp costs nothing
        IDF = 1   the warped frame is worth exactly as much as NO frame
        IDF > 1   the warped frame is WORSE than no frame (actively misleading)

⚠️ C13 GATE: IDF is **not bounded above by 1** -- the psi=90 deg control is
expected to exceed it -- so the criterion can fail and can be seen to fail. It is
also anchored at BOTH ends by measured quantities rather than by a chosen
threshold. Contrast P1's bare ADE, which has no scale and no failing value.

The interval on the EDGE (not merely on the curve) is a matched cluster
bootstrap: each replicate resamples EPISODES with replacement -- reusing
`taniteval.ci`'s own `_draws`, so it is the program's estimator, not a second
one -- recomputes the whole curve on that replicate, and locates the crossing by
linear interpolation. The percentile spread of those crossings IS the interval.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[5]
sys.path.insert(0, str(_REPO / "taniteval"))

from taniteval.ci import _draws, episode_index  # noqa: E402

DEAD = ("dead_black", "dead_noise", "dead_shuffle")
IDF_LEVELS = (0.25, 0.50, 1.00)


def crossing(xs, ys, level):
    """First psi at which the curve reaches `level`, by linear interpolation.

    FIRST crossing, deliberately: the ADE curve is NON-MONOTONE at large psi
    (it turns over once the model stops using the image), so a last-crossing or
    a monotone-inverse rule would report a larger, flattering edge.
    """
    for i in range(1, len(xs)):
        y0, y1 = ys[i - 1], ys[i]
        if (y0 < level <= y1) or (y0 >= level > y1):
            if y1 == y0:
                return float(xs[i])
            t = (level - y0) / (y1 - y0)
            return float(xs[i - 1] + t * (xs[i] - xs[i - 1]))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    summ = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    eid = list(z["eid"])
    uniq, idx_by_ep = episode_index(eid)

    yaws = sorted(float(k.split("__yaw")[1]) for k in z.files
                  if k.startswith("ade__yaw"))
    yaws_sweep = [y for y in yaws if y <= 30.0]     # 90 deg is a control, not a sweep point
    base = z["ade__baseline"]
    dead = {d: z[f"ade__{d}"] for d in DEAD if f"ade__{d}" in z.files}

    def m(v, sel=None):
        return float(v.mean() if sel is None else v[sel].mean())

    # ---- point estimates (FULL SET -- the bootstrap supplies only the interval)
    b0 = m(base)
    floor_by = {d: m(v) for d, v in dead.items()}
    floor_name = min(floor_by, key=floor_by.get)
    floor = floor_by[floor_name]
    rng = floor - b0

    curve = []
    for y in yaws:
        a = z[f"ade__yaw{y:g}"]
        curve.append({
            "psi_deg": y,
            "ade2s": round(m(a), 4),
            "delta_vs_baseline": round(m(a) - b0, 4),
            "IDF": round((m(a) - b0) / rng, 4),
            "pct_of_baseline": round(100.0 * (m(a) - b0) / b0, 2),
            "_beyond_p1_grid": y > 12.0,
        })

    # ---- the EDGE, with a matched cluster-bootstrap interval ---------------- #
    xs = np.array(yaws_sweep, dtype=float)
    boot_edges = {lv: [] for lv in IDF_LEVELS}
    boot_idf12, boot_idf25 = [], []
    for sel in _draws(uniq, idx_by_ep, args.n_boot, args.seed):
        bb = m(base, sel)
        fl = min(m(v, sel) for v in dead.values())
        r = fl - bb
        if r <= 0:
            continue
        ys = np.array([(m(z[f"ade__yaw{y:g}"], sel) - bb) / r for y in yaws_sweep])
        for lv in IDF_LEVELS:
            c = crossing(xs, ys, lv)
            if c is not None:
                boot_edges[lv].append(c)
        boot_idf12.append(float(ys[list(xs).index(12.0)]))
        if 25.0 in list(xs):
            boot_idf25.append(float(ys[list(xs).index(25.0)]))

    def ci(v):
        if not v:
            return None
        a = np.asarray(v, float)
        return {"mean": round(float(a.mean()), 3),
                "lo": round(float(np.percentile(a, 2.5)), 3),
                "hi": round(float(np.percentile(a, 97.5)), 3),
                "n_replicates_with_a_crossing": int(a.size),
                "frac_replicates_with_a_crossing": round(a.size / args.n_boot, 4)}

    edges = {}
    for lv in IDF_LEVELS:
        pt = crossing(xs, np.array([c["IDF"] for c in curve if c["psi_deg"] <= 30.0]), lv)
        edges[f"IDF_{lv:g}"] = {
            "point_estimate_deg": None if pt is None else round(pt, 2),
            "cluster_bootstrap_ci": ci(boot_edges[lv]),
            "_meaning": {
                0.25: "a quarter of the frame's usable information is gone",
                0.50: "HALF the usable information is gone",
                1.00: "the warped frame is worth NO MORE than no frame at all",
            }[lv]}

    # ---- the detection edge: P1's own implicit reading --------------------- #
    yrows = {float(r["amount"]): r for r in summ["conditions"]["yaw"]}
    ns = [a for a in sorted(yrows) if a <= 30
          and not yrows[a]["paired_vs_baseline"]["separated"]]
    sep = [a for a in sorted(yrows) if a <= 30
           and yrows[a]["paired_vs_baseline"]["separated"]]
    detect = {"largest_psi_NOT_separated_deg": max(ns) if ns else None,
              "smallest_psi_SEPARATED_deg": min(sep) if sep else None,
              "_edge_lies_in": [max(ns) if ns else None, min(sep) if sep else None],
              "_note": "grid-limited: the interval is the gap between adjacent "
                       "grid points, not a bootstrap interval",
              "_this_is_P1s_own_published_reading": True}

    out = {
        "_what": "the YAW fidelity edge, with a cluster-bootstrap interval on the EDGE",
        "_evidence_class": "MEASURED (ours)",
        "_source": {"perwindow": args.npz, "summary": args.summary_json},
        "_estimator": "matched episode-cluster bootstrap (taniteval.ci._draws), "
                      "B=%d, unit = val episode; the SAME resampled episodes are "
                      "used for every condition within a replicate" % args.n_boot,
        "_refused_estimator": "overlapping_holdout_se",
        "n_windows": int(base.size), "n_clusters": int(len(uniq)),
        "baseline_ade2s": round(b0, 4),
        "destroyed_observation_controls": {d: round(v, 4)
                                           for d, v in floor_by.items()},
        "information_floor": {"value": round(floor, 4), "which": floor_name,
                              "_def": "min over destroyed-observation controls = "
                                      "the ADE achievable with NO scene "
                                      "information"},
        "total_information_range": round(rng, 4),
        "_c13_gate": {
            "criterion": "IDF",
            "fails_when": "IDF >= 1 (the warped frame is worth no more than no "
                          "frame); IDF > 1 means actively misleading",
            "estimator_can_reach_it": True,
            "_demonstrated_by": "the psi=90 deg control, which is expected to "
                                "exceed 1.0 -- see curve_including_controls",
        },
        "curve_including_controls": curve,
        "IDF_at_shipped_12deg": ci(boot_idf12),
        "IDF_at_25deg": ci(boot_idf25),
        "EDGES": edges,
        "detection_edge_P1s_reading": detect,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}\n")
    print(f"n = {out['n_windows']} windows / {out['n_clusters']} clusters")
    print(f"baseline ADE@2s          = {b0:.4f}")
    for d, v in floor_by.items():
        print(f"  control {d:<14} = {v:.4f}")
    print(f"information floor        = {floor:.4f} ({floor_name})")
    print(f"total information range  = {rng:.4f}\n")
    print(f"{'psi':>6} {'ADE':>8} {'delta':>9} {'IDF':>8}")
    for c in curve:
        mark = "  <-- SHIPPED EDGE" if c["psi_deg"] == 12 else ""
        mark = "  <-- CONTROL (psi=90)" if c["psi_deg"] >= 90 else mark
        print(f"{c['psi_deg']:>6g} {c['ade2s']:>8.4f} {c['delta_vs_baseline']:>+9.4f} "
              f"{c['IDF']:>8.3f}{mark}")
    print("\n=== EDGES ===")
    for k, v in edges.items():
        c = v["cluster_bootstrap_ci"]
        s = (f"{v['point_estimate_deg']} deg  CI95 [{c['lo']}, {c['hi']}] "
             f"(crossing found in {c['frac_replicates_with_a_crossing']:.1%} of "
             f"replicates)") if c else f"{v['point_estimate_deg']} — NOT REACHED on the grid"
        print(f"  {k:<10} {s}")
    print(f"\ndetection edge (P1's own reading): between "
          f"{detect['largest_psi_NOT_separated_deg']} and "
          f"{detect['smallest_psi_SEPARATED_deg']} deg")


if __name__ == "__main__":
    main()
