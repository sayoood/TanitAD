"""Independent sanity check of the ADD clips' manifest poses (the jump rule is world-frame, so it needs them)."""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, 'C:/Users/Admin/tanitad-data/add203/repo/stack')
from tanitad.data import join_label_hygiene as H  # noqa: E402

R = 'C:/Users/Admin/qland/hold_jdm4572/raw/'
old = torch.load(R + 'old_train4369_v2manifest.pt', map_location='cpu', weights_only=False)
new = torch.load('C:/Users/Admin/tanitad-data/add203/in/train4572_v2manifest.pt', map_location='cpu', weights_only=False)
cen = json.load(open(R + 'out/join_clip_census.json'))['per_clip_sha12']
oidx = {c: i for i, c in enumerate(old['clip_id'])}
# (1) old clips' poses in the new manifest are bit-identical to the old manifest
same = n = 0
for i, c in enumerate(new['clip_id']):
    if c in oidx:
        n += 1
        same += int(torch.equal(new['poses'][i], old['poses'][oidx[c]]))
print('old clips compared', n, 'poses torch.equal', same)


def stats(P):
    P = P.numpy().astype(np.float64)
    d = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
    return float(np.median(d)), float(d.max()), len(P)


add = [(c, i) for i, c in enumerate(new['clip_id']) if c not in oidx]
add_j = [(c, i) for c, i in add if H.clip_key(c) in cen]
res = {}
for tag, lst, man in (('ADD203', add, new), ('ADD58_joined', add_j, new),
                      ('OLD4369', [(c, i) for c, i in oidx.items()], old)):
    s = np.array([stats(man['poses'][i]) for c, i in lst])
    res[tag] = dict(n=len(lst), median_step_m_p50=float(np.median(s[:, 0])), max_step_m_max=float(s[:, 1].max()),
                    max_step_m_p99=float(np.percentile(s[:, 1], 99)),
                    n_clips_max_step_gt_5m=int((s[:, 1] > 5.0).sum()), T_min_med_max=[int(s[:, 2].min()), float(np.median(s[:, 2])), int(s[:, 2].max())],
                    nonfinite=int(sum(0 if torch.isfinite(man['poses'][i]).all() else 1 for c, i in lst)))
print(json.dumps(res, indent=1))
json.dump(dict(old_pose_equal=dict(n=n, same=same), stats=res), open(R + 'out/pose_sanity.json', 'w'), indent=1)
