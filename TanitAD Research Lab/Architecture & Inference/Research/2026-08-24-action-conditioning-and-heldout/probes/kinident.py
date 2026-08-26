"""IS `action -> delta-yaw` A FINDING, OR A KINEMATIC IDENTITY? (PI question, 2026-08-26)

⭐ THE QUESTION. After tonight's retractions the ONLY surviving "the action carries
information about the ego's future" claims are BOTH LATERAL and BOTH steering->yaw:
    action -> dyaw        t 4.57
    action -> yaw-rate    t 5.09
The PI points out that steering MUST correspond to yaw-rate. If so, these are not
two findings — they are ONE ALGEBRAIC IDENTITY measured twice, and the programme
has no evidence that its action channel carries ego information beyond kinematics.

THE PROVENANCE (physicalai.py:604-632, read directly):
    actions[:,0] = steer = atan(wheelbase * curvature)   <- from `curvature`
    actions[:,1] = accel = ax                            <- measured longitudinal
    poses[:,2]   = yaw   = quaternion_yaw(qx,qy,qz,qw)   <- a DIFFERENT sensor path
    poses[:,3]   = v     = hypot(vx, vy)
Kinematically yaw_rate = v * curvature, and curvature = tan(steer)/L. So
    dyaw(t -> t+k)  ~=  v_t * tan(steer_t)/L * k*dt
is a CLOSED FORM requiring no learning at all.

⭐⭐ THE TEST: compare the CLOSED FORM against the learned probe (+0.5638).
  closed-form r ~= probe r   -> the "learned" relation IS the identity. The probe
                               discovered kinematics we already knew, and the claim
                               carries no information about the world model.
  probe r >> closed-form r   -> the probe found something the identity does not
                               explain, and the claim survives as a finding.

⚠️ ALSO REPORTED: r(steer_t, instantaneous yaw-rate). If that is ~1, the two
channels are the same quantity in different units and `action -> yaw-rate` (t 5.09)
is a TAUTOLOGY, not a measurement.

CPU only, no model, no GPU. Same 20 held-out clips as every panel tonight.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import torch

SP = pathlib.Path(__file__).resolve().parent
LEAD = pathlib.Path(os.environ.get(
    "SPD_CORPUS", str(SP / "sp2/cache/physicalai-val130-heldout")))
OUT = pathlib.Path(os.environ.get("SPD_OUT", str(SP / "kinident.json")))
N_CLIPS, F, K, W = 20, 100, 4, 6
WHEELBASE = 2.9          # the legacy constant physicalai.py defaults to
DT = 0.1                 # 100 ms egomotion clock


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def r(x, y):
    x = np.asarray(x, float) - np.mean(x)
    y = np.asarray(y, float) - np.mean(y)
    return float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-12))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    clips = sorted(LEAD.glob("*.v2ep.pt"))[:N_CLIPS]
    S, A, V, YR, DY, KIN, KINR = [], [], [], [], [], [], []
    for c in clips:
        d = torch.load(c, map_location="cpu", weights_only=False)
        act = np.asarray(d["actions"], dtype=np.float64)
        pos = np.asarray(d["poses"], dtype=np.float64)
        yaw, v = pos[:, 2], pos[:, 3]
        n = min(len(act), len(yaw), F) - K - 1
        if n < 25:
            continue
        i = np.arange(n)
        steer, accel = act[i, 0], act[i, 1]
        vt = v[i]
        yr = wrap(yaw[i + 1] - yaw[i]) / DT                    # rad/s, measured
        dy = wrap(yaw[i + K] - yaw[i])                         # the panel's target
        kappa = np.tan(steer) / WHEELBASE                      # curvature from steer
        kin = vt * kappa * K * DT                              # CLOSED-FORM dyaw
        kinr = vt * kappa                                      # CLOSED-FORM yaw-rate
        S.append(steer); A.append(accel); V.append(vt)
        YR.append(yr); DY.append(dy); KIN.append(kin); KINR.append(kinr)
    S, A, V = map(np.concatenate, (S, A, V))
    YR, DY, KIN, KINR = map(np.concatenate, (YR, DY, KIN, KINR))

    print(f"\n  IS `action -> delta-yaw` A FINDING OR A KINEMATIC IDENTITY?")
    print(f"  n = {len(S)} frames, {len(clips)} held-out clips, wheelbase {WHEELBASE} m\n")
    rows = [
        ("steer_t  vs  yaw-rate_t (measured)", r(S, YR),
         "~1 => the two channels ARE the same quantity"),
        ("CLOSED FORM v*tan(steer)/L  vs  yaw-rate_t", r(KINR, YR),
         "the identity, with no learning"),
        ("steer_t  vs  dyaw(t->t+4)", r(S, DY), "the panel's raw linear version"),
        ("CLOSED FORM v*tan(steer)/L*k*dt  vs  dyaw", r(KIN, DY),
         "*** the comparison that decides it ***"),
    ]
    print(f"  {'relation':<46}{'r':>9}   note")
    print("  " + "-" * 92)
    for nm, val, note in rows:
        print(f"  {nm:<46}{val:>+9.4f}   {note}")

    probe = 0.5638            # the RFF probe's true-minus-shuffled r, E-DEC-50
    cf = r(KIN, DY)
    print(f"\n  the LEARNED probe scored           {probe:+.4f}")
    print(f"  the CLOSED FORM scores             {cf:+.4f}")
    gap = probe - cf
    print(f"  the probe's excess over kinematics  {gap:+.4f}")
    if abs(gap) < 0.10:
        v = ("THE RELATION IS THE IDENTITY. The probe recovered textbook kinematics, "
             "not information about the world model. `action -> dyaw` must NOT be "
             "quoted as evidence that the action channel carries ego information.")
    elif gap > 0.10:
        v = ("the probe EXCEEDS the closed form by "
             f"{gap:+.4f} — there is something beyond the identity, and it is that "
             "excess, not the raw r, which is the finding.")
    else:
        v = ("the closed form BEATS the learned probe — the probe is a lossy "
             "estimate of a relation we can compute exactly.")
    print(f"\n  => {v}\n")
    rep = {"_evidence_class": "MEASURED (ours; dev-box, CPU)",
           "eval_tier": "T0-DIAGNOSTIC", "split": "HELD-OUT", "n_frames": int(len(S)),
           "wheelbase_m": WHEELBASE, "dt_s": DT, "k": K,
           "r_steer_vs_yawrate": round(r(S, YR), 4),
           "r_closedform_vs_yawrate": round(r(KINR, YR), 4),
           "r_steer_vs_dyaw": round(r(S, DY), 4),
           "r_closedform_vs_dyaw": round(cf, 4),
           "learned_probe_r": probe,
           "probe_excess_over_kinematics": round(gap, 4),
           "verdict": v}
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
