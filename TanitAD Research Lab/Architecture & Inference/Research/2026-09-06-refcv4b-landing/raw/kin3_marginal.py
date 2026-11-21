"""Measure the EVAL-SET kin3 class marginal, and the no-information CE floor it
implies, from a refcv3_arm dump -- so the claim "the longitudinal head is worse
than its own class prior" is read against the marginal of the split it was
SCORED on, not against the trainer's TRAIN-side EMA buffer.

⛔ WHY THIS EXISTS. `core.lat_log_prior` / `core.lon_log_prior` are an EMA
maintained by `core.update_tactical_prior`, and `refc_v3_train.py` gates that
call on `model.training` -- so the buffer is a TRAIN statistic. Quoting its
entropy as the floor for an EVAL cross-entropy is a scope error of exactly the
`df` / `step_s` family. This closes it with the eval split's own marginal.

The labels are recomputed with the TRAINER'S OWN call --
`refc_tactical.window_factored_labels(pose_last, fut_ext[:, :20])`
(`refc_v3_train.py:578`) -- from `pose_last` and `gt_future_ext`, both of which
the decisions sidecar already carries. No model, no GPU, no re-roll.

⚠️ `lat_label` / `lon_label` IN THE SIDECAR ARE NOT THESE. They are the 8-wide
v7.2 z_tac labels with -100 = IGNORE outside the record's +-2 s band (verified:
values reach 7 and -100). The core's aux heads are kin3 and their labels are
never dumped, which is why they must be recomputed here.
"""
import glob
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/TanitAD/stack")
from tanitad.refs import refc_tactical as tac          # noqa: E402

dump = sys.argv[1]
out_json = sys.argv[2] if len(sys.argv) > 2 else None

files = sorted(glob.glob(os.path.join(dump, "decisions", "ep*.npz")))
if not files:
    sys.exit("no decisions/ep*.npz under %s" % dump)

lat_all, lon_all, n_ep = [], [], 0
horizon = int(getattr(tac, "LABEL_HORIZON", 20))
for f in files:
    d = np.load(f, allow_pickle=True)
    if "pose_last" not in d.files or "gt_future_ext" not in d.files:
        continue
    pl = torch.from_numpy(np.asarray(d["pose_last"], dtype=np.float32))
    fu = torch.from_numpy(np.asarray(d["gt_future_ext"], dtype=np.float32))
    if fu.shape[1] < horizon:
        continue
    lat, lon = tac.window_factored_labels(pl, fu[:, :horizon])
    lat_all.append(lat.numpy())
    lon_all.append(lon.numpy())
    n_ep += 1

lat = np.concatenate(lat_all)
lon = np.concatenate(lon_all)
N = int(lat.shape[0])
NL, NN = int(tac.N_LAT), int(tac.N_LON)


def marginal(x, k):
    c = np.bincount(x, minlength=k).astype(np.float64)
    return c / max(c.sum(), 1.0), c.astype(int).tolist()


def entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


pl_, cl_ = marginal(lat, NL)
pn_, cn_ = marginal(lon, NN)

res = {
    "dump": dump,
    "n_windows": N,
    "n_episodes": n_ep,
    "label_horizon_steps": horizon,
    "n_lat": NL, "n_lon": NN,
    "source": ("refc_tactical.window_factored_labels(pose_last, "
               "gt_future_ext[:, :LABEL_HORIZON]) -- the trainer's own call, "
               "refc_v3_train.py:578"),
    "lateral": {"marginal": [round(float(x), 6) for x in pl_],
                "counts": cl_,
                "prior_predictor_ce_nats": round(entropy(pl_), 6)},
    "longitudinal": {"marginal": [round(float(x), 6) for x in pn_],
                     "counts": cn_,
                     "prior_predictor_ce_nats": round(entropy(pn_), 6)},
    "uniform_ce_nats": round(math.log(NL), 6),
}

print("n_windows=%d  n_episodes=%d  horizon=%d  (n_lat=%d n_lon=%d)"
      % (N, n_ep, horizon, NL, NN))
print("uniform CE (ln k)            = %.4f nats" % math.log(NL))
print("LATERAL      marginal %s  counts %s" % (np.round(pl_, 4).tolist(), cl_))
print("             prior-predictor CE = %.4f nats" % entropy(pl_))
print("LONGITUDINAL marginal %s  counts %s" % (np.round(pn_, 4).tolist(), cn_))
print("             prior-predictor CE = %.4f nats" % entropy(pn_))
# CONTROL: the marginals must be proper distributions and the counts must sum
# to N -- a silently empty read would otherwise print a plausible uniform.
assert abs(pl_.sum() - 1.0) < 1e-9 and abs(pn_.sum() - 1.0) < 1e-9
assert sum(cl_) == N and sum(cn_) == N, "counts do not sum to n_windows"
print("CONTROL ok: both marginals sum to 1 and counts sum to n_windows=%d" % N)

if out_json:
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("wrote %s" % out_json)
