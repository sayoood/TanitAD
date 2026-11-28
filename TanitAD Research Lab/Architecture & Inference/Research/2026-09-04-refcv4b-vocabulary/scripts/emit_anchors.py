"""Emit the refcv4-b v0-CONDITIONED anchor artifact.

The file carries BOTH halves of the vocabulary:
  * `controls` [N, 2] = (accel m/s^2, curvature 1/m), constant over the horizon —
    THE vocabulary; the decoder rolls it per window from that window's v0;
  * `anchors`  [N, S, 2] = the same family rolled at `ref_speed_ms`, which is
    what the decoder installs in its `anchors` buffer (the checkpoint-visible
    artifact a torch.equal content check compares) and what a withheld
    ego-dropout row decodes.

⛔ ASSERTED HERE AND PRINTED: (a=0, kappa=0) is present EXACTLY. An even-count
linspace omits it and the family then reads 1.2768 m oracle-in-vocabulary
against 0.2610 -- a 4.9x artifact that looks exactly like a resolution finding.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import torch

STACK = r"C:\Users\Admin\run_refcv4v\repo\stack"
sys.path.insert(0, STACK)
from tanitad.refs.refa_v1_plan import unicycle_paths            # noqa: E402
from tanitad.refs.refc_v3 import V3_HORIZONS                    # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
DT = 0.1
H = max(V3_HORIZONS)
SLOTS = [k - 1 for k in V3_HORIZONS]
REF_SPEED = 10.0

NA, NK = 13, 9
A_LO, A_HI, K_LIM = -4.0, 3.0, 0.06

a_g = np.linspace(A_LO, A_HI, NA)
a_g = np.clip(a_g - a_g[np.abs(a_g).argmin()], A_LO, A_HI)
k_g = np.linspace(-K_LIM, K_LIM, NK)
assert np.any(a_g == 0.0), "accel grid omits 0.0 EXACTLY: %s" % a_g
assert np.any(k_g == 0.0), "curvature grid omits 0.0 EXACTLY: %s" % k_g
print("ASSERTION  accel grid contains 0.0 EXACTLY : True   %s" %
      np.array2string(a_g, precision=4, max_line_width=200))
print("ASSERTION  curvature grid contains 0.0 EXACTLY: True   %s" %
      np.array2string(k_g, precision=4, max_line_width=200))

aa, kk = np.meshgrid(a_g, k_g, indexing="ij")
ctrl = np.stack([aa.ravel(), kk.ravel()], -1)
N = len(ctrl)
zi = int(((ctrl[:, 0] == 0) & (ctrl[:, 1] == 0)).nonzero()[0][0])
print("ASSERTION  {a=0, kappa=0} present at index %d of %d" % (zi, N))

C = torch.from_numpy(ctrl).float()
paths = unicycle_paths(C[:, None, :].expand(-1, H, -1).contiguous(),
                       torch.tensor(REF_SPEED), DT,
                       action_units="kappa")[:, SLOTS].contiguous()
t = torch.tensor([k * DT for k in V3_HORIZONS])
cv = torch.stack([REF_SPEED * t, torch.zeros_like(t)], -1)
err = (paths[zi] - cv).abs().max().item()
print("CONTROL    {a=0,kappa=0} @ %.1f m/s vs (v0*t, 0): max abs err %.3e m"
      % (REF_SPEED, err))
assert err < 1e-4, err
straight = (C[:, 1] == 0)
print("CONTROL    kappa==0 anchors: %d, max |y| = %.3e m"
      % (int(straight.sum()), paths[straight, :, 1].abs().max().item()))
assert paths[straight, :, 1].abs().max().item() < 1e-6

art = {"anchors": paths, "controls": C}
p = os.path.join(OUT, "refc_anchors_6s_v0cond_%d.pt" % N)
torch.save(art, p)
sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
meta = {
    "artifact_kind": "tanitad.refc_anchor_vocabulary",
    "variant": "v0-CONDITIONED constant-(accel, curvature)",
    "n_anchors": N, "horizons_steps": list(V3_HORIZONS), "dt_s": DT,
    "ref_speed_ms": REF_SPEED,
    "accel_grid_ms2": [float(x) for x in a_g],
    "curvature_grid_inv_m": [float(x) for x in k_g],
    "straight_ahead_control_index": zi,
    "asserted_zero_present": {"accel": True, "curvature": True},
    "integrator": "tanitad.refs.refa_v1_plan.unicycle_paths"
                  " (action_units='kappa') -> models.kinematic.rollout_unicycle",
    "gate": {"surface": "taniteval/results/refcv3-40284-openloop-dump.tar.gz",
             "windows": 4823, "episodes": 141,
             "oracle_in_vocabulary_ade_0_2s_m": 0.2610,
             "along_mae_m": 0.1473, "lat_mae_m": 0.1701,
             "bar_ha_m": 0.2996,
             "paired_delta_vs_ha": [-0.0387, -0.0652, -0.0104],
             "verdict": "BEATS ha (separated)",
             "estimator": "paired episode-cluster bootstrap, n_boot 2000, "
                          "seed 0, cluster = episode"},
    "file_sha256": sha,
    "anchors_sha256": hashlib.sha256(
        paths.numpy().tobytes()).hexdigest(),
    "controls_sha256": hashlib.sha256(C.numpy().tobytes()).hexdigest(),
}
json.dump(meta, open(p + ".json", "w"), indent=1)
print("\nwrote %s  (%d bytes)\n  file sha256 %s" % (p, os.path.getsize(p), sha))
print("  anchors  %s  controls %s" % (tuple(paths.shape), tuple(C.shape)))
