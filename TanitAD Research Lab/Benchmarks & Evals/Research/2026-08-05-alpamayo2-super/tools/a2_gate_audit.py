"""⛔ THE 0.15 rad DIRECTION GATE IS MIS-SCALED FOR 2 s WINDOWS — audit, both arms.

⭐ HOW THIS WAS FOUND, because the route matters. Alpamayo's declared lateral manoeuvre
scored kappa 0.1488 against the path it drove — which reads as "the declaration is
decorative". Before reporting that, the obvious confound was tested: its own
Chain-of-Causation says things like *"Nudge left to pass the parked SUV"*, and a NUDGE is
a lateral offset far below a 0.15 rad net-heading turn. So the low kappa might be a
THRESHOLD MISMATCH rather than incoherence.

MEASURED: the median |net yaw| over 2 s on these clips is **0.0227 rad** — our gate is
**6.6x the typical turn**, so essentially every window classifies as "straight" and the
statistic is measuring the gate, not the model.

⛔ AND IT CONTAMINATES OUR OWN NUMBERS, WHICH IS THE POINT. `hierarchy.DIR_YAW_RAD =
0.15` produced our arm's executed-manoeuvre kappa (0.4968), its declared-vs-driven kappa
(0.3432) and the "0 of 2 left turns" in the four-family block. If the gate is 6.6x too
coarse, all three are gate artifacts. This sweeps BOTH arms across gates so the
sensitivity is visible instead of a single number being quoted as if it were stable.

⚠️ SECOND DEFECT, same family as the curvature-at-standstill trap: net yaw is
MEANINGLESS when the ego does not move. A stopped window's tangent flips freely and
produced a net yaw of **pi** in the raw pass. Steps below `MIN_DS_MPS * dt` are excluded
here, exactly as `four_families._seq_geometry` does.
"""
from __future__ import annotations

import argparse
import json
import lzma
import math

import numpy as np

MIN_DS_MPS = 0.5
DT = 0.1
LAT2DIR = {"Go Straight": 1, "Steer Left": 0, "Steer Right": 2,
           "Sharp Steer Left": 0, "Sharp Steer Right": 2,
           "Slight Steer Left": 0, "Slight Steer Right": 2}
MAN2DIR = {0: 1, 1: 0, 2: 2, 3: 1, 4: 1}       # our 5-way softmax -> {L,S,R}
DIRNAME = {0: "left", 1: "straight", 2: "right"}


def net_yaw(path: np.ndarray, dt: float = DT) -> np.ndarray:
    """[n,H,2] -> net heading change over the horizon, radians (+ = left).

    ⛔ Steps whose displacement is below ``MIN_DS_MPS * dt`` are DROPPED from the sum.
    A stopped or crawling ego has no path tangent, and including those steps let a
    single jitter contribute a full pi to the total — MEASURED in the raw pass."""
    p = np.concatenate([np.zeros((path.shape[0], 1, 2)), path], axis=1)
    d = p[:, 1:] - p[:, :-1]
    ds = np.linalg.norm(d, axis=-1)
    h = np.arctan2(d[..., 1], d[..., 0])
    dh = (h[:, 1:] - h[:, :-1] + math.pi) % (2 * math.pi) - math.pi
    valid = (ds[:, 1:] > MIN_DS_MPS * dt) & (ds[:, :-1] > MIN_DS_MPS * dt)
    return np.where(valid, dh, 0.0).sum(axis=1)


def cls(ny: np.ndarray, gate: float) -> np.ndarray:
    return np.where(ny > gate, 0, np.where(ny < -gate, 2, 1))


def kappa(x, y):
    labs = sorted(set(x.tolist()) | set(y.tolist()))
    po = float((x == y).mean())
    pe = sum((x == c).mean() * (y == c).mean() for c in labs)
    return None if abs(1 - pe) < 1e-9 else round((po - pe) / (1 - pe), 4)


GATES = (0.15, 0.10, 0.06, 0.04, 0.03, 0.02, 0.01)


def sweep(decl, ny_driven, ny_gt):
    """kappa of a DECLARED class against the driven path and against GT, per gate."""
    return {f"{g:.2f}": {
        "vs_driven_agreement": round(float((decl == cls(ny_driven, g)).mean()), 4),
        "vs_driven_kappa": kappa(decl, cls(ny_driven, g)),
        "vs_gt_kappa": kappa(decl, cls(ny_gt, g)),
        "n_gt_turns": int((cls(ny_gt, g) != 1).sum()),
    } for g in GATES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-parsed", required=True)
    ap.add_argument("--alpamayo-traj", required=True)
    ap.add_argument("--alpamayo-gt", required=True)
    ap.add_argument("--flagship-json", required=True)
    ap.add_argument("--four-families", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    idx_all = json.load(open(a.four_families))["clip_indices"]
    A = {int(k): np.asarray(v) for k, v in
         json.loads(lzma.open(a.alpamayo_traj).read()).items()}
    AG = {int(k): np.asarray(v) for k, v in
          json.load(open(a.alpamayo_gt))["gt_xy_by_index"].items()}
    FS = {r["i"]: r for r in json.load(open(a.flagship_json)) if "ade_2s_m" in r}
    MP = json.load(open(a.meta_parsed))["per_clip"]

    out = {
        "_evidence_class": "MEASURED (ours)",
        "_why": ("the 0.15 rad direction gate (hierarchy.DIR_YAW_RAD) decides every "
                 "TACTICAL number in the four-family block. This measures whether the "
                 "numbers are stable under it or produced by it."),
        "_min_ds_gate": (f"steps below {MIN_DS_MPS} m/s excluded from the net-yaw sum; "
                         "a stopped ego has no tangent and produced a net yaw of pi in "
                         "the ungated pass"),
        "gates_swept": list(GATES),
    }

    # --- how big are the turns actually? the fact the gate has to be judged against ---
    ny_gt_fs = net_yaw(np.stack([np.asarray(FS[i]["gt"])[:20] for i in idx_all]))
    out["turn_magnitude"] = {
        "n": len(idx_all),
        "median_abs_net_yaw_rad": round(float(np.median(np.abs(ny_gt_fs))), 4),
        "p90_abs_net_yaw_rad": round(float(np.percentile(np.abs(ny_gt_fs), 90)), 4),
        "max_abs_net_yaw_rad": round(float(np.abs(ny_gt_fs).max()), 4),
        "frac_above_0p15": round(float((np.abs(ny_gt_fs) > 0.15).mean()), 4),
        "_read": ("this is the HUMAN's own net yaw over 2 s. If the gate exceeds the "
                  "bulk of it, every window is 'straight' by construction."),
    }

    # --- ALPAMAYO: declared lateral axis vs its own driven path -------------------
    ai = [i for i in idx_all
          if str(i) in MP and MP[str(i)]["lateral"] in LAT2DIR and i in A and i in AG]
    if ai:
        decl = np.array([LAT2DIR[MP[str(i)]["lateral"]] for i in ai])
        nyd = net_yaw(np.stack([A[i][:20] for i in ai]))
        nyg = net_yaw(np.stack([AG[i][:20, :2] for i in ai]))
        out["alpamayo_declared_lateral"] = {"n": len(ai), "sweep": sweep(decl, nyd, nyg)}
        turn = decl != 1
        out["alpamayo_declared_lateral"]["sign_only_on_declared_turns"] = {
            "n": int(turn.sum()),
            "agreement": round(float((decl[turn] == np.where(nyd[turn] > 0, 0, 2)).mean()), 4),
            "_read": ("magnitude discarded entirely: does the driven path lean the "
                      "declared way AT ALL? This is the gate-free read."),
        }

    # --- OURS: declared 5-way head vs its own driven path -------------------------
    fi = [i for i in idx_all if FS[i].get("man") is not None]
    if fi:
        decl = np.array([MAN2DIR[int(FS[i]["man"])] for i in fi])
        nyd = net_yaw(np.stack([np.asarray(FS[i]["pred"])[:20] for i in fi]))
        nyg = net_yaw(np.stack([np.asarray(FS[i]["gt"])[:20] for i in fi]))
        out["flagship_declared_maneuver"] = {"n": len(fi), "sweep": sweep(decl, nyd, nyg)}
        turn = decl != 1
        out["flagship_declared_maneuver"]["sign_only_on_declared_turns"] = {
            "n": int(turn.sum()),
            "agreement": round(float((decl[turn] == np.where(nyd[turn] > 0, 0, 2)).mean()), 4)
            if turn.any() else None,
        }
        # executed-manoeuvre agreement (both sides trajectory-derived), per gate
        out["flagship_executed_vs_gt"] = {f"{g:.2f}": {
            "accuracy": round(float((cls(nyd, g) == cls(nyg, g)).mean()), 4),
            "kappa": kappa(cls(nyd, g), cls(nyg, g))} for g in GATES}
        aP = np.stack([A[i][:20] for i in fi if i in A])
        aG = np.stack([AG[i][:20, :2] for i in fi if i in AG])
        nyad, nyag = net_yaw(aP), net_yaw(aG)
        out["alpamayo_executed_vs_gt"] = {f"{g:.2f}": {
            "accuracy": round(float((cls(nyad, g) == cls(nyag, g)).mean()), 4),
            "kappa": kappa(cls(nyad, g), cls(nyag, g))} for g in GATES}

    json.dump(out, open(a.out, "w"), indent=1)
    tm = out["turn_magnitude"]
    print(f"HUMAN net yaw over 2 s: median {tm['median_abs_net_yaw_rad']} · "
          f"p90 {tm['p90_abs_net_yaw_rad']} rad · only "
          f"{tm['frac_above_0p15']:.1%} of windows exceed the 0.15 gate\n")
    print(f"{'gate':>6} | {'ALPAMAYO declared k':>20} | {'FLAGSHIP declared k':>20} "
          f"| {'ALP exec k':>11} | {'FS exec k':>10}")
    for g in GATES:
        k = f"{g:.2f}"
        print(f"{g:>6.2f} | "
              f"{str(out['alpamayo_declared_lateral']['sweep'][k]['vs_driven_kappa']):>20} | "
              f"{str(out['flagship_declared_maneuver']['sweep'][k]['vs_driven_kappa']):>20} | "
              f"{str(out['alpamayo_executed_vs_gt'][k]['kappa']):>11} | "
              f"{str(out['flagship_executed_vs_gt'][k]['kappa']):>10}")
    print(f"\nsign-only (gate-free) on declared turns: "
          f"Alpamayo {out['alpamayo_declared_lateral']['sign_only_on_declared_turns']['agreement']} "
          f"(n={out['alpamayo_declared_lateral']['sign_only_on_declared_turns']['n']}) · "
          f"flagship {out['flagship_declared_maneuver']['sign_only_on_declared_turns']['agreement']} "
          f"(n={out['flagship_declared_maneuver']['sign_only_on_declared_turns']['n']})")
    print(f"\n[out] {a.out}")


if __name__ == "__main__":
    main()
