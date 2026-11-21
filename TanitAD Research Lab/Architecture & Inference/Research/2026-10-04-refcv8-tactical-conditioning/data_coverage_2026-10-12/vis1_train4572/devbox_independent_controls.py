#!/usr/bin/env python3
"""Independent dev-box controls for the merged VIS-1 sidecar (different host, different numpy, different decode path
from Thor's vis1_train4572_controls.py, which walks clip_frame_ptr/frame_ptr per clip).

(a2) ROW-LEVEL: expand every row to (clip sha12, frame_f, row, track, n_full, n_img, n_vis, vis_rows, cx, cy) through
     ``frame_clip`` (NOT through clip_frame_ptr), restrict the NEW arrays to the OLD clip set, and require equality with the
     OLD expansion element-wise (order included). Same for the frame table.
(c2) the clip set vs the two dev manifests (md5-identical copies of the Thor ones), via the trainer's own
     ``VIS1Sidecar.require_clips`` from the DEV tree's vis1.py.
(d2) ADD-clip distribution check: per-clip mean vis_frac, ADD vs OLD, and the 5 joined ADD clips with zero stored rows.
"""
import hashlib, json, sys
import numpy as np
import torch
sys.path.insert(0, sys.argv[4])               # <dev tree>/stack
from tanitad.data import vis1 as V
old_p, new_p, man_dir = sys.argv[1], sys.argv[2], sys.argv[3]
def sha12(s): return hashlib.sha256(str(s).encode()).hexdigest()[:12]
zo, zn = np.load(old_p, allow_pickle=False), np.load(new_p, allow_pickle=False)
def expand(z):
    cl = z['clip_sha12']
    fr_sha = cl[z['frame_clip']]
    cnt = np.diff(z['frame_ptr'])
    row_sha = np.repeat(fr_sha, cnt); row_f = np.repeat(z['frame_f'], cnt)
    return fr_sha, row_sha, row_f
fo, ro, rfo = expand(zo); fn, rn, rfn = expand(zn)
old_set = set(zo['clip_sha12'].tolist())
mf = np.isin(fn, zo['clip_sha12']); mr = np.isin(rn, zo['clip_sha12'])
res = {}
ok = True
for k in ('frame_f', 'frame_n_zh', 'frame_n_valid'):
    ok &= bool(np.array_equal(zo[k], zn[k][mf]))
ok &= bool(np.array_equal(fo, fn[mf]))
rowkeys = ('row', 'track', 'n_full', 'n_img', 'n_vis', 'vis_rows', 'cx', 'cy')
for k in rowkeys:
    ok &= bool(zo[k].dtype == zn[k].dtype and np.array_equal(zo[k], zn[k][mr]))
ok &= bool(np.array_equal(ro, rn[mr]) and np.array_equal(rfo, rfn[mr]))
res['a2_row_level_old_rows_equal'] = {'old_frames': int(len(fo)), 'new_frames_in_old_clips': int(mf.sum()),
                                      'old_rows': int(len(ro)), 'new_rows_in_old_clips': int(mr.sum()), 'PASS': ok}
# old clip metadata equal
cm = {s: i for i, s in enumerate(zn['clip_sha12'].tolist())}
idx = np.array([cm[s] for s in zo['clip_sha12'].tolist()])
res['a2_old_clip_meta_equal'] = bool(np.array_equal(zo['clip_mask_ftheta'], zn['clip_mask_ftheta'][idx]) and
                                     np.array_equal(zo['clip_split'], zn['clip_split'][idx]))
# (c2)
mt = torch.load(f'{man_dir}/train4572_v2manifest.pt', map_location='cpu', weights_only=False)
me = torch.load(f'{man_dir}/eval139_v2manifest.pt', map_location='cpu', weights_only=False)
ids = [sha12(c) for c in mt['clip_id']] + [sha12(c) for c in me['clip_id']]
sc = V.VIS1Sidecar(new_p)
try:
    sc.require_clips(ids, where='dev-box independent control'); res['c2_require_clips'] = f'no raise ({len(ids)} ids, {sc.n_clips} sidecar clips)'
except SystemExit as e:
    res['c2_require_clips'] = 'RAISED ' + str(e)[:200]
res['c2_set_equal'] = bool(set(ids) == set(zn['clip_sha12'].tolist()) and len(set(ids)) == len(ids) == 4711)
# (d2)
add = sorted(set(zn['clip_sha12'].tolist()) - old_set)
res['n_add'] = len(add)
cfp, fp = zn['clip_frame_ptr'], zn['frame_ptr']
def clip_vf(s):
    ci = cm[s]; f0, f1 = int(cfp[ci]), int(cfp[ci + 1]); r0, r1 = int(fp[f0]), int(fp[f1])
    nf, nv = zn['n_full'][r0:r1].astype(np.int64), zn['n_vis'][r0:r1].astype(np.int64)
    m = nf > 0
    return f1 - f0, r1 - r0, (float((nv[m] / nf[m]).mean()) if m.any() else None), int(zn['frame_n_valid'][f0:f1].sum()), int(zn['frame_n_zh'][f0:f1].sum())
A = [clip_vf(s) for s in add]
joined = [(s, a) for s, a in zip(add, A) if a[0] > 0]
O = [clip_vf(s) for s in zo['clip_sha12'].tolist() if zn['clip_split'][cm[s]] == 'train']
def med(v): v = [x for x in v if x is not None]; return float(np.median(v)), float(np.mean(v)), len(v)
res['d2_per_clip_mean_vis_frac(median,mean,n)'] = {'ADD_joined': med([a[2] for _, a in joined]), 'OLD_train': med([o[2] for o in O if o[0] > 0])}
res['d2_rows_per_frame'] = {'ADD_joined': float(sum(a[1] for _, a in joined) / sum(a[0] for _, a in joined)),
                            'OLD_train': float(sum(o[1] for o in O) / sum(o[0] for o in O))}
res['d2_joined_ADD_with_zero_stored_rows'] = [{'sha12': s, 'frames': a[0], 'valid_rows_total': a[3], 'rows_with_3d_label_total': a[4]}
                                              for s, a in joined if a[1] == 0]
res['d2_frames_per_clip_ADD_joined_sorted_head'] = sorted(a[0] for _, a in joined)[:5]
res['d2_OLD_frames_per_clip_min'] = int(min(o[0] for o in O if o[0] > 0))
res['d2_zero_frame_clips_in_new'] = int((np.diff(zn['clip_frame_ptr']) == 0).sum())
res['d2_zero_frame_all_mask_true_split_train'] = bool(all(zn['clip_mask_ftheta'][cm[s]] and zn['clip_split'][cm[s]] == 'train'
                                                         for s in add if np.diff(zn['clip_frame_ptr'])[cm[s]] == 0))
print(json.dumps(res, indent=1))
