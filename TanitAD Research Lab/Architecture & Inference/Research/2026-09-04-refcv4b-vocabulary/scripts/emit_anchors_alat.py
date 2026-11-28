"""Emit the refcv4b v0-CONDITIONED, SPEED-CLAMPED anchor artifact.

The vocabulary is `controls [N, 2] = (a_lon m/s^2, a_lat m/s^2)`. Curvature is
DERIVED per window as `a_lat / max(v0, V_FLOOR)^2`, clamped to KAPPA_CAP -- so
the control space IS the Kamm-circle space and a bound on the grid is a bound on
the friction circle.

⛔ ASSERTED AND PRINTED: `(a_lon = 0, a_lat = 0)` is present EXACTLY, and
`a_lat = 0` <=> `kappa = 0` exactly, so the pinned straight-ahead control
survives the reparameterisation.

⛔ THE REFERENCE-SPEED ROLL USES THE SAME DERIVATION as the decoder, or the
checkpoint-visible `anchors` buffer would not be the family a withheld
ego-dropout row actually decodes.

⭐ SELF-DESCRIBING ARTIFACT (2026-09-05). The `.pt` this writes carries
`control_units="alat"`, `horizon_s`, `dt`, `ref_speed_ms`, `kappa_cap`,
`alat_v_floor` and a provenance stamp IN the file
(`tanitad.refs.anchor_meta.build_anchor_artifact`). MEASURED 2026-09-04 on the
LIVE artifact this script's previous version built (`anchors.pt`, sha256
`e86cf507…e8fb`): it held exactly `anchors` + `controls`, and read as CURVATURE
the same bytes gave 396 g at 36 m/s with 104/117 over mu = 0.7, against the
true 0.31 g and 0/117. The run was never ambiguous (config.json['argv'] carries
`--anchor-control-units alat`); the standalone file was. ⛔ The live
`/workspace/experiments/refcv4b-b1-v72-40k/anchors.pt` is NOT rewritten -- the
run is training against it and loads it through the explicit override; this
version is for the NEXT build.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import torch

# the stack this script imports: an already-importable `tanitad` (PYTHONPATH,
# the venv) WINS; only if none is importable do we guess -- the repo's own
# stack/ first, then the dev-box rig the 2026-09-04 build ran from. The guess
# must never shadow an explicit PYTHONPATH (the rig's stack predates
# tanitad.refs.anchor_meta and would fail one line later).
_HERE = os.path.dirname(os.path.abspath(__file__))
try:
    import tanitad                                                # noqa: F401
except ImportError:
    for _cand in (os.path.normpath(os.path.join(_HERE, *([".."] * 5), "stack")),
                  r"C:\Users\Admin\run_refcv4v\repo\stack"):
        if os.path.isdir(os.path.join(_cand, "tanitad")):
            sys.path.insert(0, _cand)
            break
import tanitad                                                    # noqa: E402
from tanitad.refs import anchor_meta                            # noqa: E402
print("stack: %s" % os.path.dirname(os.path.abspath(tanitad.__file__)))
from tanitad.refs.refa_v1_plan import unicycle_paths            # noqa: E402
from tanitad.refs.refc_v3 import V3_HORIZONS                    # noqa: E402

OUT = os.path.join(_HERE, "out")
os.makedirs(OUT, exist_ok=True)
DT, H = 0.1, max(V3_HORIZONS)
SLOTS = [k - 1 for k in V3_HORIZONS]
REF_SPEED, V_FLOOR, KAPPA_CAP = 10.0, 4.0, 0.12

NA, NK = 13, 9
A_LO, A_HI = -4.0, 3.0
A_LAT_MAX = 3.0          # m/s^2 ~ 0.31 g -- the standard COMFORTABLE lateral
                         # bound for a passenger vehicle. Chosen on physics; the
                         # sweep {3.0, 4.0, 6.0} shows EVERY value clears the
                         # gate separated, so the verdict does not depend on it.

a_g = np.linspace(A_LO, A_HI, NA)
# ⛔ RE-CENTRED so that 0.0 is a NODE. np.linspace(-4, 3, 13) has step 7/12 and
# its nodes are ..., -0.5, +0.0833, ... -- an odd count does NOT put zero on an
# asymmetric range (RETRACTION_LOG 2026-09-05). The assert below is the check.
a_g = np.clip(a_g - a_g[np.abs(a_g).argmin()], A_LO, A_HI)
c_g = np.linspace(-1.0, 1.0, NK) * A_LAT_MAX
assert np.any(a_g == 0.0), "a_lon grid omits 0.0 EXACTLY: %s" % a_g
assert np.any(c_g == 0.0), "a_lat grid omits 0.0 EXACTLY: %s" % c_g
print("ASSERTION  a_lon grid contains 0.0 EXACTLY : True   %s"
      % np.array2string(a_g, precision=4, max_line_width=200))
print("ASSERTION  a_lat grid contains 0.0 EXACTLY : True   %s"
      % np.array2string(c_g, precision=4, max_line_width=200))

aa, cc = np.meshgrid(a_g, c_g, indexing="ij")
ctrl = np.stack([aa.ravel(), cc.ravel()], -1)
N = len(ctrl)
zi = int(((ctrl[:, 0] == 0) & (ctrl[:, 1] == 0)).nonzero()[0][0])
print("ASSERTION  {a_lon=0, a_lat=0} present at index %d of %d" % (zi, N))

C = torch.from_numpy(ctrl).float()
# the reference-speed roll, by the SAME derivation the decoder uses
kap = np.clip(ctrl[:, 1] / max(REF_SPEED, V_FLOOR) ** 2, -KAPPA_CAP, KAPPA_CAP)
ref_ctrl = torch.from_numpy(np.stack([ctrl[:, 0], kap], -1)).float()
paths = unicycle_paths(ref_ctrl[:, None, :].expand(-1, H, -1).contiguous(),
                       torch.tensor(REF_SPEED), DT,
                       action_units="kappa")[:, SLOTS].contiguous()

t = torch.tensor([k * DT for k in V3_HORIZONS])
cv = torch.stack([REF_SPEED * t, torch.zeros_like(t)], -1)
err = (paths[zi] - cv).abs().max().item()
print("CONTROL    {a_lon=0, a_lat=0} @ %.1f m/s vs (v0*t, 0): max abs err %.3e m"
      % (REF_SPEED, err))
assert err < 1e-4, err
straight = (C[:, 1] == 0)
print("CONTROL    a_lat==0 anchors: %d, max |y| = %.3e m  (a_lat=0 <=> kappa=0)"
      % (int(straight.sum()), paths[straight, :, 1].abs().max().item()))
assert paths[straight, :, 1].abs().max().item() < 1e-6

extra = {
    "artifact_kind": "tanitad.refc_anchor_vocabulary",
    "variant": "v0-CONDITIONED, SPEED-CLAMPED constant-(a_lon, a_lat)",
    "a_lon_grid_ms2": [float(x) for x in a_g],
    "a_lat_grid_ms2": [float(x) for x in c_g],
    "a_lat_max_ms2": A_LAT_MAX,
    "a_lat_max_rationale": "3.0 m/s^2 ~ 0.31 g, the standard COMFORTABLE "
                           "lateral bound for a passenger vehicle. Chosen on "
                           "physics, not on the score: the sweep {3.0, 4.0, "
                           "6.0} clears the gate SEPARATED at every value "
                           "(-0.1009 / -0.0900 / -0.0701), so the gate verdict "
                           "does not depend on this choice.",
    "kappa_derivation": "kappa = clamp(a_lat / max(v0, alat_v_floor)^2, "
                        "-kappa_cap, +kappa_cap), per window, in the decoder's "
                        "`roll_bank`",
    "straight_ahead_control_index": zi,
    "asserted_zero_present": {"a_lon": True, "a_lat": True},
    "integrator": "tanitad.refs.refa_v1_plan.unicycle_paths"
                  " (action_units='kappa') -> models.kinematic.rollout_unicycle",
    "gate": {"surface": "taniteval/results/refcv3-40284-openloop-dump.tar.gz",
             "windows": 4823, "episodes": 141,
             "oracle_in_vocabulary_ade_0_2s_m": 0.1987,
             "along_mae_m": 0.1432, "lat_mae_m": 0.1037,
             "bar_ha_m": 0.2996,
             "paired_delta_vs_ha": [-0.1009, -0.1213, -0.0813],
             "verdict": "BEATS ha (separated)",
             "estimator": "paired episode-cluster bootstrap, n_boot 2000, "
                          "seed 0, cluster = episode",
             "_scope": "MEASURED 2026-09-04 on the 2026-09-04 build of this "
                       "grid (sha256 e86cf507...e8fb); a re-emitted file "
                       "reproduces the grid, not the measurement -- re-run "
                       "the gate before quoting it for a new file"},
    "kamm": {"over_mu_0.7_at_v0_27.27_ms": 0, "peak_g_at_v0_27.27_ms": 0.68,
             "over_mu_0.7_at_v0_10.09_ms": 12, "peak_g_at_v0_10.09_ms": 1.23,
             "flat_kappa_family_for_scale": {"over_mu_0.7_at_27.27": 104,
                                             "peak_g": 3.96}},
}
art = anchor_meta.build_anchor_artifact(
    paths, C, control_units="alat", horizons=V3_HORIZONS, dt=DT,
    ref_speed_ms=REF_SPEED, kappa_cap=KAPPA_CAP, alat_v_floor=V_FLOOR,
    builder=__file__, extra=extra)
p = os.path.join(OUT, "refc_anchors_6s_v0cond_alat_%d.pt" % N)
torch.save(art, p)
sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
meta = {k: v for k, v in art.items() if k not in ("anchors", "controls")}
meta["file_sha256"] = sha
json.dump(meta, open(p + ".json", "w"), indent=1)
print("\nwrote %s  (%d bytes)\n  file sha256 %s" % (p, os.path.getsize(p), sha))
print("  anchors %s  controls %s (a_lon, a_lat)  units=%s  horizon_s=%s"
      % (tuple(paths.shape), tuple(C.shape), art["control_units"],
         art["horizon_s"]))
# read-back through the consumer's own resolver: no override needed any more
_rb = anchor_meta.read_anchor_artifact(p)
assert _rb.control_units == "alat" and _rb.control_units_source == "file", _rb
print("  read-back: %s" % anchor_meta.describe(_rb))
