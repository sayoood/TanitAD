#!/usr/bin/env python3
"""LONGITUDINAL distance-keeping for an OPEN-LOOP run — the family cl_metrics cannot see.

⛔ THE PROBLEM THIS SOLVES. `cl_metrics.py`'s distance-keeping block measures the gap
between the EGO and the annotated lead. In closed loop that is a policy quantity: the ego
is where the model drove it. In OPEN loop the ego is pinned to the logged pose, so
`headway_m`, `time_gap_s` and `ttc_s_when_closing` are properties of the LOG and come out
**bit-identical for every arm** — MEASURED: all four `real_lead_*` paired deltas are
exactly +0.0000 on both conditions of scene 00040136. Reporting that as "not separated"
would be reporting the log, not the arms.

⭐ WHAT IS POLICY-DEPENDENT IN OPEN LOOP is the gap **the plan would produce**. Project
the ego forward along the emitted plan to +2 s, put the lead where the annotation says it
will be at +2 s, and measure the gap the arm is steering into:

    plan_headway_2s  = lead_x(t+2s) − plan_x(2s) − EGO_LEN
    plan_time_gap_2s = plan_headway_2s / max(v_target, 0.1)
    plan_ttc_2s      = plan_headway_2s / (v_target − v_lead)      [only while CLOSING]

and the rates that actually matter for safety: how often the plan implies a time gap under
1 s / 0.5 s, and how often it implies **driving into the lead** (`plan_headway_2s <= 0`).

⚠️ **A DUPLICATE THAT MUST NOT BE COUNTED TWICE.** The *error* of this quantity against
the log is algebraically identical to the along-track error at 2 s:

    plan_headway_2s − gt_headway_2s  =  −(plan_x(2s) − gt_x(2s))

because the lead term cancels. So `headway_err_2s` is NOT an independent separation and is
emitted with that stated on it — the same defect the closed-loop panel caught when
`dist_to_gt_traj_m` turned out to be `abs(cross_track)` under a second name and one
measurement was published as two. **The LEVELS and the RATES above are the new
information; the error is not.**

⚠️ In-lane gate: the lead must be inside |y| < LEAD_HALF_W at the time it is selected, and
the projected gap is reported with the lead's projected |y| so a "collision" 13 m to the
side cannot be counted as one — the exact bug corrected in `cl_metrics` on 2026-08-03.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

DT = 0.1
H2S = 20                    # ticks to +2.0 s
EGO_LEN = 4.7
LEAD_HALF_W = 2.0
LEAD_MAX_X = 80.0


def rows_for(path, tracks, keep):
    import cl_metrics as C
    d = json.loads(Path(path).read_text())
    gt = d["gt"]
    C._CLIP_SPAN_S[0] = max(1e-3, (gt[-1]["ts_us"] - gt[0]["ts_us"]) / 1e6)
    xy, gyaw, gv, gtp = C.gt_arrays(gt)
    N = len(gt)
    ts0, ts1 = gt[0]["ts_us"], gt[-1]["ts_us"]
    span = max(ts1 - ts0, 1.0)
    out, eids = [], []
    for rec in d["rollouts"]:
        for st in rec["steps"]:
            i = int(st["i_gt"])
            if i + H2S >= N:
                continue                       # truncated: no +2 s ground truth
            e = st["ego"]
            frac = float(np.clip((st["t_us"] - ts0) / span, 0.0, 1.0))
            frac2 = float(np.clip((st["t_us"] + 2e6 - ts0) / span, 0.0, 1.0))
            acts = C.actors_at(tracks, frac, e, keep=keep, dfrac=0.005)
            cand = [(a["rig"][0], n) for n, a in enumerate(acts)
                    if a["rig"][0] > 0 and abs(a["rig"][1]) < LEAD_HALF_W
                    and a["rig"][0] < LEAD_MAX_X]
            r = {"k": int(st["k"]), "eid": int(rec["start_frame"]), "lead": False}
            if cand:
                _, li = min(cand)
                idx = acts[li]["idx"]
                # the SAME track, two seconds later
                fut = C.actors_at(tracks, frac2, e, keep={idx}, dfrac=0.005)
                if fut:
                    lx, ly = fut[0]["rig"][0], fut[0]["rig"][1]
                    v_lead = fut[0].get("v")
                    plan_x = float(np.asarray(st["plan"], float)[3, 0])
                    gt_x = float(C.ego_frame(xy[i + H2S] - xy[i], gyaw[i])[0])
                    ph = lx - plan_x - EGO_LEN
                    gh = lx - gt_x - EGO_LEN
                    vt = float(st["v_target"])
                    tg = ph / max(vt, 0.1)
                    ttc = None
                    if v_lead is not None:
                        closing = vt - v_lead
                        if closing > 0.1 and ph > 0:
                            ttc = ph / closing
                    r.update({
                        "lead": True, "lead_idx": int(idx),
                        "plan_headway_2s": ph, "gt_headway_2s": gh,
                        "headway_err_2s_DUPLICATE_OF_ALONG_TRACK": ph - gh,
                        "plan_time_gap_2s": tg,
                        "plan_ttc_2s": (ttc if ttc is not None else np.nan),
                        "plan_tg_below_1s": float(tg < 1.0),
                        "plan_tg_below_0_5s": float(tg < 0.5),
                        "plan_lead_abs_y_2s": abs(ly),
                        "plan_inlane_2s": float(abs(ly) <= LEAD_HALF_W),
                        "plan_would_hit_UNGATED": float(ph <= 0.0),
                        "plan_would_hit_inlane": float(ph <= 0.0 and abs(ly) <= LEAD_HALF_W),
                        "v_target": vt, "v_lead": (v_lead if v_lead is not None else np.nan),
                    })
            out.append(r)
            eids.append(int(rec["start_frame"]))
    return d, out, eids


def ci(vals, eid, reduce="mean"):
    from taniteval.ci import episode_cluster_bootstrap
    v = np.asarray(vals, float)
    ok = np.isfinite(v)
    if ok.sum() == 0:
        return {"n": 0, "reason": "no finite values"}
    r = episode_cluster_bootstrap(v[ok], list(np.asarray(eid)[ok]), reduce=reduce)
    r["n_used"] = int(ok.sum())
    r["n_total"] = int(v.size)
    return r


def paired(a, b, eid):
    from taniteval.ci import paired_episode_cluster_bootstrap
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() == 0:
        return {"n": 0, "reason": "no jointly finite windows"}
    r = paired_episode_cluster_bootstrap(a[ok], b[ok], list(np.asarray(eid)[ok]))
    r["n_used"] = int(ok.sum())
    return r


FIELDS = ("plan_headway_2s", "plan_time_gap_2s", "plan_ttc_2s", "plan_tg_below_1s",
          "plan_tg_below_0_5s", "plan_would_hit_inlane", "plan_would_hit_UNGATED",
          "headway_err_2s_DUPLICATE_OF_ALONG_TRACK")


def arm_block(d, rows, eids):
    lead = [(r, e) for r, e in zip(rows, eids) if r["lead"]]
    n_lead, n_all = len(lead), len(rows)
    if not lead:
        return {"n": 0, "n_windows": n_all,
                "reason": "NOT MEASURABLE — no annotated agent was in-lane ahead of the "
                          "logged ego on any window of this clip. Stated per family with "
                          "the denominator rather than dropped."}
    lr = [r for r, _ in lead]
    le = [e for _, e in lead]
    out = {"n_windows_total": n_all, "n_windows_with_lead": n_lead,
           "lead_present_rate": round(n_lead / n_all, 4)}
    for f in FIELDS:
        out[f] = ci([r.get(f, np.nan) for r in lr], le)
    out["plan_min_headway_2s"] = ci([r["plan_headway_2s"] for r in lr], le,
                                    reduce=lambda v: float(np.min(v)))
    fires = [r for r in lr if r["plan_would_hit_UNGATED"] == 1.0]
    true_in = [r for r in fires if r["plan_would_hit_inlane"] == 1.0]
    out["plan_would_hit_precision"] = {
        "n_fires": len(fires), "n_true_inlane": len(true_in),
        "precision": (round(len(true_in) / len(fires), 4) if fires else None),
        "median_abs_y_of_fires_m": (round(float(np.median(
            [r["plan_lead_abs_y_2s"] for r in fires])), 3) if fires else None),
        "note": "precision of the UNGATED plan_headway<=0 test against the in-lane truth "
                f"(|y| <= {LEAD_HALF_W} m). Quote `plan_would_hit_inlane`."}
    out["note"] = (
        "OPEN-LOOP distance-keeping: the gap the PLAN would produce at +2 s against the "
        "annotated lead's own +2 s position, from the LOGGED ego pose. This is the "
        "policy-dependent part of the family; the ego-to-lead gap itself is a property of "
        "the log in open loop and is identical across arms by construction. "
        "⚠️ `headway_err_2s_DUPLICATE_OF_ALONG_TRACK` is algebraically −(along-track error "
        "at 2 s) — the lead term cancels — and must NOT be counted as an independent "
        "separation.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", default=None)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--renderable-from", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gsplat_renderer import ActorTracks
    for lbl, val in (("--tracks", args.tracks), ("--renderable-from", args.renderable_from)):
        if val and not Path(val).exists():
            raise FileNotFoundError(f"{lbl}={val!r} does not exist. Refusing to score.")
    tracks = ActorTracks(args.tracks)
    keep = None
    if args.renderable_from:
        am = json.loads(Path(args.renderable_from).read_text())
        keep = {int(x["best_track"]) for x in am["per_track"] if x["accepted"]}

    dA, rA, eA = rows_for(args.a, tracks, keep)
    res = {"arm_A": {"name": dA["arm"], "condition": dA["condition"],
                     "LONGITUDINAL_distance_keeping": arm_block(dA, rA, eA)},
           "renderable_restricted": (None if keep is None
                                     else {"source": args.renderable_from,
                                           "n_tracks": len(keep)}),
           "estimator_note": ("episode-cluster bootstrap over the DISJOINT SEGMENTS of "
                              "one clip used as clusters. Not 40 independent episodes."),
           "within_sim_note": ("WITHIN-SIM RELATIVE. REF-C open-loop ADE is 1.5157 on "
                               "these reconstructions vs 0.4728 on real footage "
                               "(3.21x OOD). Orderings survive; absolute rates do not.")}
    if args.b:
        dB, rB, eB = rows_for(args.b, tracks, keep)
        res["arm_B"] = {"name": dB["arm"], "condition": dB["condition"],
                        "LONGITUDINAL_distance_keeping": arm_block(dB, rB, eB)}
        KA = {(r["eid"], r["k"]): r for r in rA if r["lead"]}
        KB = {(r["eid"], r["k"]): r for r in rB if r["lead"]}
        common = sorted(set(KA) & set(KB))
        pe = [c[0] for c in common]
        res["paired_A_minus_B"] = {
            f: paired([KA[c].get(f, np.nan) for c in common],
                      [KB[c].get(f, np.nan) for c in common], pe) for f in FIELDS}
        res["paired_n_windows"] = len(common)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2)[:4000])


if __name__ == "__main__":
    main()
