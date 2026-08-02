"""STREAM C probes on pod2 — all read-only w.r.t. project data.

P1  slice refusal: is the w120 256x640 cylindrical cache re-derivable to the
    256px SQUARE geometry by a centred sub-frame slice?
P2  parity guard: does a 40-episode PREFIX of the raw val epcache pass
    assert_val_cache (the registered canonical TanitEval deployment)?
P3  episode identity: bit-exact pose/action match between the raw epcache
    ep_%05d.pt and the w120 v2 <clip_id>.v2ep.pt, which recovers the
    ep_index -> clip_id map the epcache does not store.
"""
import glob
import json
import os
import sys
import time

import torch

OUT = {}
EPC = '/workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11'
W120 = '/workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl'

# ---------------------------------------------------------------- P1 slice --
from tanitad.data.calib import CanonicalFrame, CANONICAL_256, subframe_slice, F_REF

wf = sorted(glob.glob(os.path.join(W120, '*.v2ep.pt')))[0]
wd = torch.load(wf, map_location='cpu', weights_only=False, mmap=True)
stored = CanonicalFrame.from_dict(wd['frame'])
p1 = {'w120_stored_frame': wd['frame'], 'w120_tag': stored.tag(),
      'w120_codec': str(wd['codec']),
      'target_CANONICAL_256': CANONICAL_256.to_dict(),
      'target_tag': CANONICAL_256.tag(), 'F_REF': F_REF}
try:
    rs, cs = subframe_slice(stored, CANONICAL_256)
    p1['w120cyl_to_256sq'] = {'result': 'SLICE OK', 'rows': [rs.start, rs.stop],
                              'cols': [cs.start, cs.stop]}
except Exception as e:
    p1['w120cyl_to_256sq'] = {'result': 'REFUSED', 'exc': type(e).__name__,
                              'msg': str(e)}
# CONTROL: the slice machinery works when f_ref+projection DO match
same = CanonicalFrame(height=256, width=256, f_ref=float(wd['frame']['f_ref']),
                      projection='cylindrical')
try:
    rs, cs = subframe_slice(stored, same)
    p1['CONTROL_w120cyl_to_256x256cyl'] = {'result': 'SLICE OK',
                                           'rows': [rs.start, rs.stop],
                                           'cols': [cs.start, cs.stop]}
except Exception as e:
    p1['CONTROL_w120cyl_to_256x256cyl'] = {'result': 'REFUSED',
                                           'exc': type(e).__name__, 'msg': str(e)}
OUT['P1_slice_refusal'] = p1

# --------------------------------------------------------------- P2 parity --
from tanitad.data import parity

p2 = {'deployments': parity.val_deployments()}
try:
    p2['full_600'] = parity.assert_val_cache(EPC, label='P2-full')
except SystemExit as e:
    p2['full_600'] = {'REFUSED': str(e)}
# 40-episode PREFIX in a scratch dir of symlinks (no project data touched)
probe = '/tmp/val40_probe/physicalai-val-0c5f7dac3b11'
os.makedirs(probe, exist_ok=True)
for i in range(40):
    fn = 'ep_%05d.pt' % i
    dst = os.path.join(probe, fn)
    if not os.path.islink(dst):
        os.symlink(os.path.join(EPC, fn), dst)
try:
    p2['prefix_40'] = parity.assert_val_cache(probe, label='P2-prefix40')
except SystemExit as e:
    p2['prefix_40'] = {'REFUSED': str(e)}
# NEGATIVE control: 39 episodes must NOT be a registered deployment
probe39 = '/tmp/val39_probe/physicalai-val-0c5f7dac3b11'
os.makedirs(probe39, exist_ok=True)
for i in range(39):
    fn = 'ep_%05d.pt' % i
    dst = os.path.join(probe39, fn)
    if not os.path.islink(dst):
        os.symlink(os.path.join(EPC, fn), dst)
try:
    p2['NEGCTRL_prefix_39'] = parity.assert_val_cache(probe39, label='P2-neg39')
except SystemExit as e:
    p2['NEGCTRL_prefix_39'] = {'REFUSED': str(e)[:400]}
OUT['P2_parity_guard'] = p2

# ------------------------------------------------------------- P3 identity --
t0 = time.time()
wfiles = sorted(glob.glob(os.path.join(W120, '*.v2ep.pt')))
index = {}
for f in wfiles:
    d = torch.load(f, map_location='cpu', weights_only=False, mmap=True)
    k = int(d['n_stack']) - 1
    po = d['poses'][k:].float().contiguous()
    key = (tuple(round(float(x), 6) for x in po[0].tolist()), int(po.shape[0]))
    index.setdefault(key, []).append((str(d['clip_id']), po.clone()))
p3 = {'n_w120_clips': len(wfiles), 'index_build_s': round(time.time() - t0, 1),
      'n_distinct_keys': len(index)}

matches, unmatched = [], []
N_ID = 40
for i in range(N_ID):
    ep = torch.load(os.path.join(EPC, 'ep_%05d.pt' % i), map_location='cpu',
                    weights_only=False, mmap=True)
    po = ep['poses'].float().contiguous()
    key = (tuple(round(float(x), 6) for x in po[0].tolist()), int(po.shape[0]))
    cands = index.get(key, [])
    hit = None
    for cid, wpo in cands:
        if wpo.shape == po.shape:
            md = float((wpo - po).abs().max())
            if md == 0.0:
                hit = (cid, md, 'BIT_EXACT')
                break
            if md < 1e-5 and hit is None:
                hit = (cid, md, 'TOL_1e-5')
    if hit:
        matches.append({'ep_index': i, 'ep_uid': 'ep_%05d.pt' % i,
                        'clip_id': hit[0], 'max_abs_pose_diff': hit[1],
                        'grade': hit[2], 'T': int(po.shape[0])})
    else:
        unmatched.append({'ep_index': i, 'T': int(po.shape[0]),
                          'n_candidates': len(cands)})
p3['n_matched'] = len(matches)
p3['n_unmatched'] = len(unmatched)
p3['n_distinct_clip_ids'] = len({m['clip_id'] for m in matches})
p3['grades'] = {g: sum(1 for m in matches if m['grade'] == g)
                for g in {m['grade'] for m in matches}}
p3['matches'] = matches
p3['unmatched'] = unmatched
OUT['P3_episode_identity'] = p3

print(json.dumps(OUT, indent=1, default=str))
