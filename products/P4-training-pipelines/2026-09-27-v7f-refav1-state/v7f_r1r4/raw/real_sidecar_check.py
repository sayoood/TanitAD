"""R1 on REAL labels (dev box, read-only): the local v8 EVAL label blob -> the refcv6 sidecar
(built with the UNCHANGED `scripts/build_refcv6_speed_max_window.py --omit-clip-id` into a
scratch dir, NOT banked) -> the trainer's `build_vmax_join` (with stand-in episodes carrying the
sidecar's own stable ids -- the v7F train corpus is not on this box) -> the model-side one-hot
and the cap limit. Aggregates only; no identifier is printed.
Usage: python real_sidecar_check.py <stack_root> <label_blob> <sidecar>"""
import collections
import hashlib
import json
import sys

root, blob, side = sys.argv[1:4]
sys.path.insert(0, root)
sys.path.insert(0, root + "/scripts")
import torch  # noqa: E402

import train_v6_staged as T  # noqa: E402
from tanitad.models.plan_speed_cap import plan_limit_from_vhi  # noqa: E402
from tanitad.refs.refcv6_max_speed import (SpeedMaxStampError, limit_ms_of_bin,  # noqa: E402
                                           speed_max_bin_tensor, speed_max_onehot)

md5 = hashlib.md5(open(blob, "rb").read()).hexdigest()
rows = [json.loads(x) for x in open(side, encoding="utf-8") if x.strip()]


class Ep:
    def __init__(self, sid):
        self.episode_id = sid


eps = [Ep(int(r["sid"])) for r in rows]
index = [(i, 0) for i in range(len(eps))]
j = T.build_vmax_join(side, eps, index, label_md5=md5)
v, ok = j["v"], j["ok"]
lim = plan_limit_from_vhi(v, ok)
idx, over = speed_max_bin_tensor(torch.where(ok > 0.5, v, torch.zeros_like(v)))
oh = speed_max_onehot(idx, ok > 0.5)
side_bin = torch.tensor([int(r.get("bin", -1) if r.get("bin") is not None else -1) for r in rows])
agree = bool(torch.equal(idx[ok > 0.5], side_bin[ok > 0.5]))
lim_from_side = limit_ms_of_bin(side_bin.clamp_min(0))
refused = None
try:
    T.build_vmax_join(side, eps, index, label_md5="0" * 32)
except SpeedMaxStampError as e:
    refused = "REFUSED (md5 mismatch): " + str(e)[:90]
out = {
    "label_blob_md5": md5, "n_sidecar_rows": len(rows),
    "join_report": {k: j["report"][k] for k in ("n_rows", "n_valid", "n_over_ceiling",
                                                "n_episodes_fed", "n_windows_fed",
                                                "window_ceiling_frac", "label_md5")},
    "limit_kmh_census": dict(sorted(collections.Counter(
        [round(float(x) * 3.6) for x in lim if torch.isfinite(x)]).items())),
    "n_no_limit_rows": int((~torch.isfinite(lim)).sum()),
    "onehot_row_sums": dict(collections.Counter([int(x) for x in oh.sum(-1)])),
    "model_side_bin_equals_sidecar_bin_on_every_valid_row": agree,
    "cap_limit_equals_limit_of_sidecar_bin": bool(torch.allclose(
        lim[ok > 0.5], lim_from_side[ok > 0.5].to(lim.dtype))),
    "n_over_ceiling_model_side": int(over[ok > 0.5].sum()),
    "wrong_blob_md5": refused,
    "_evidence_class": "MEASURED (dev box, the local v8 EVAL blob; not the train sidecar)"}
print(json.dumps(out, indent=1))
