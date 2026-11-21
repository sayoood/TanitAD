#!/usr/bin/env python3
"""Is the entry-clamp arm's residual a PROJECTION leak or a SCORER artifact on a stop?

MEASURED: `project_feasible(mu=0.70)` leaves 0/51200 candidates violating `envelope`;
adding `clamp_entry=True` leaves 2/51200, both in ONE window whose v0 is 0.000.

⛔ THE DISCRIMINATOR IS A SECOND, INDEPENDENT RECOVERY -- not the same one run twice.
`rewards.kinematics` (which `fan_safety.score_paths` uses) takes
`heading = atan2(dy, dx)` with NO guard, so a ZERO-LENGTH step reads heading 0 rather
than "undefined", and the step before it appears to have turned by the whole previous
heading. `models.kinematic.unicycle_controls_from_path` guards exactly this case --
"WHERE THE EGO IS NOT MOVING, CURVATURE IS UNDETERMINED -- NOT LARGE" -- and is the
programme's other implementation of the same idea.

If the guarded recovery reads the same paths as feasible, the residual is the SCORER's
unguarded heading on a stopped step, and the projection is exact wherever the path moves.
If BOTH read a violation, it is a projection leak and must be fixed. Reported either way.
ASCII-only output.
"""
import json
import os
import sys

import numpy as np
import torch

_REPO = os.environ.get("TANITAD_REPO") or "C:/Users/Admin/refcv4b_repo"
sys.path.insert(0, os.path.join(_REPO, "stack"))
sys.path.insert(0, os.path.join(_REPO, "taniteval"))
sys.path.insert(0, os.path.join(_REPO, "taniteval", "tools"))

from tanitad.refs import feasible_decode as FD                      # noqa: E402
from tanitad.models.kinematic import unicycle_controls_from_path    # noqa: E402
import fan_safety as FS                                             # noqa: E402

NPZ = "C:/Users/Admin/kingate/raw/kingate_bank_drawA_v1_selscore.npz"
OUT = "C:/Users/Admin/feasdec/raw/stopped_step_residual.json"

z = np.load(NPZ)
raw5 = FS.with_origin(torch.from_numpy(z["fan8"]).float()[..., :4, :])
v0 = torch.from_numpy(z["v0"]).float()
W, N = raw5.shape[0], raw5.shape[1]
rep = {"_tool": "2026-09-05-feasible-decode/raw/stopped_step_residual.py",
       "_evidence_class": "MEASURED (ours)", "npz": NPZ, "n_candidates": W * N,
       "arms": {}}

for tag, kw in (("mu0.70", dict(mu=FD.MU_KAMM)),
                ("mu0.70+entry", dict(mu=FD.MU_KAMM, clamp_entry=True))):
    q = FD.project_feasible(raw5, v0, **kw)
    sc = FS.score_paths(q, v0, None)
    env = sc["envelope"]
    sp = FD.recover_controls(q.double())[0]
    stopped = (sp < 1e-6).any(dim=-1)
    n_env = int(env.sum())
    r = {"config": {k: (None if v is None else float(v) if isinstance(v, (int, float))
                        else bool(v)) for k, v in kw.items()},
         "envelope_rate_is_exactly_zero": bool(float(env.float().mean()) == 0.0),
         "n_violating": n_env, "n_candidates": W * N,
         "n_with_a_zero_length_step": int(stopped.sum()),
         "violators_that_contain_a_stop": int((env & stopped).sum())}
    if n_env:
        wi, ci = torch.nonzero(env, as_tuple=True)
        # the SECOND, GUARDED recovery -- a different mechanism, not a re-run
        g = unicycle_controls_from_path(q[wi, ci].double(), dt=FD.DT_S)
        r["guarded_recovery"] = {
            "tool": "models.kinematic.unicycle_controls_from_path (guards a non-moving "
                    "step: curvature UNDETERMINED, not large)",
            "max_abs_accel": float(g[..., 0].abs().max()),
            "max_abs_curvature": float(g[..., 1].abs().max()),
            "within_box": bool(float(g[..., 0].abs().max()) <= FD.A_MAX_MPS2 + 1e-4
                               and float(g[..., 1].abs().max())
                               <= FD.KAPPA_MAX_1PM + 1e-4)}
        r["offenders"] = [{"window": int(a), "candidate": int(b), "v0_mps": float(v0[a]),
                           "speeds_mps": [round(x, 4) for x in
                                          sp[a, b].numpy().tolist()]}
                          for a, b in zip(wi.tolist(), ci.tolist())]
        r["VERDICT"] = ("SCORER ARTIFACT on a stopped step"
                        if (r["violators_that_contain_a_stop"] == n_env
                            and r["guarded_recovery"]["within_box"])
                        else "PROJECTION LEAK -- must be fixed")
    else:
        r["VERDICT"] = "EXACT -- no residual"
    rep["arms"][tag] = r
    print("%-14s exact_zero=%-5s n_violating=%d/%d  with_a_stop=%d  -> %s"
          % (tag, r["envelope_rate_is_exactly_zero"], n_env, W * N,
             r["violators_that_contain_a_stop"], r["VERDICT"]))
    if n_env:
        print("               guarded recovery: max|a| %.4f  max|kappa| %.4f  "
              "within box = %s"
              % (r["guarded_recovery"]["max_abs_accel"],
                 r["guarded_recovery"]["max_abs_curvature"],
                 r["guarded_recovery"]["within_box"]))
        for o in r["offenders"]:
            print("               offender w=%d c=%d v0=%.3f speeds=%s"
                  % (o["window"], o["candidate"], o["v0_mps"], o["speeds_mps"]))

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(rep, fh, indent=1)
print("[stop] wrote %s" % OUT)
