"""Characterise the ego-footprint boxes of the 3 ADD clips that carry them (read-only pass over the 3-D join)."""
import collections
import hashlib
import json
import lzma
import math
import statistics
import sys

import torch

sys.path.insert(0, 'C:/Users/Admin/tanitad-data/add203/repo/stack')
from tanitad.data import join_label_hygiene as H  # noqa: E402

R = 'C:/Users/Admin/qland/hold_jdm4572/raw/'
new_ids = list(torch.load('C:/Users/Admin/tanitad-data/add203/in/train4572_v2manifest.pt', map_location='cpu',
                          weights_only=False)['clip_id'])
want = {'5ed957d47a41', '70928506fcc6', '9dee20474b44'}
cid2k = {c: H.clip_key(c) for c in new_ids if H.clip_key(c) in want}
assert len(cid2k) == 3, cid2k
res = {k: dict(boxes=[], frames=set(), n_records=0, n_agents=[]) for k in want}
with lzma.open(R + 'b1_train_plus_eval_agents_3d.jsonl.xz', 'rt', encoding='utf-8') as fh:
    for line in fh:
        cid = line[12:48]
        if cid not in cid2k:
            continue
        r = json.loads(line)
        k = cid2k[cid]
        res[k]['n_records'] += 1
        res[k]['n_agents'].append(len(r['agents']))
        for a in r['agents']:
            if H.is_ego_footprint(a['cx'], a['cy']):
                res[k]['boxes'].append((r['frame_idx'], a['cx'], a['cy'], a['l'], a['w'], a['cls'], a['track_id'],
                                        a.get('occ')))
                res[k]['frames'].add(r['frame_idx'])
out = {}
for k, v in res.items():
    b = v['boxes']
    d = [math.hypot(x[1] - 1.4, x[2]) for x in b]
    out[k] = dict(n_records=v['n_records'], n_ego_boxes=len(b), n_frames=len(v['frames']),
                  within_0p5_of_1p4_0=sum(1 for x in d if x < 0.5),
                  cx_min_med_max=[min(x[1] for x in b), statistics.median(x[1] for x in b), max(x[1] for x in b)],
                  cy_min_med_max=[min(x[2] for x in b), statistics.median(x[2] for x in b), max(x[2] for x in b)],
                  l_med=statistics.median(x[3] for x in b), w_med=statistics.median(x[4] for x in b),
                  cls=dict(collections.Counter(x[5] for x in b)),
                  n_distinct_track_ids=len({x[6] for x in b}), occ=dict(collections.Counter(x[7] for x in b)),
                  agents_per_record_med=statistics.median(v['n_agents']))
json.dump(out, open(R + 'out/add_ego_boxes.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
tot = sum(o['n_ego_boxes'] for o in out.values())
w = sum(o['within_0p5_of_1p4_0'] for o in out.values())
print('TOTAL', tot, w, w / tot)
