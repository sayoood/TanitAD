"""Controls for refcv8_join_label_defects_train4572.json (every figure printed here goes into RESULT.md)."""
import copy
import hashlib
import json
import re
import sys

import torch

sys.path.insert(0, 'C:/Users/Admin/tanitad-data/add203/repo/stack')
from tanitad.data import join_label_hygiene as H  # noqa: E402

R = 'C:/Users/Admin/qland/hold_jdm4572/raw/'
OLD_MAN = R + 'old_train4369_v2manifest.pt'
NEW_MAN = 'C:/Users/Admin/tanitad-data/add203/in/train4572_v2manifest.pt'
EV_MAN = 'C:/Users/Admin/tanitad-data/add203/in/eval139_v2manifest.pt'
OLD_BLOB = R + 'branch_old_config.json'          # git show agent/arch-inf-20260803:... (LF blob)
OLD_REPRO = R + 'out/old_repro.json'
NEW = R + 'out/new_train4572.json'
CENSUS = R + 'out/join_clip_census.json'
META3D = 'C:/Users/Admin/tanitad-data/add203/in/join3d.meta.json'


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def ids(p):
    return list(torch.load(p, map_location='cpu', weights_only=False)['clip_id'])


old_ids, new_ids, ev_ids = ids(OLD_MAN), ids(NEW_MAN), ids(EV_MAN)
S_old, S_new, S_ev = set(old_ids), set(new_ids), set(ev_ids)
add = S_new - S_old
out = {}
out['manifest_counts'] = dict(old_train=len(old_ids), new_train=len(new_ids), eval139=len(ev_ids),
                              old_unique=len(S_old), new_unique=len(S_new), add=len(add),
                              old_subset_of_new=S_old <= S_new, eval_in_new_train=len(S_ev & S_new),
                              eval_in_old_train=len(S_ev & S_old), add_in_eval=len(add & S_ev))
k_old = {H.clip_key(c) for c in S_old}
k_new = {H.clip_key(c) for c in S_new}
k_ev = {H.clip_key(c) for c in S_ev}
k_add = {H.clip_key(c) for c in add}
out['sha12_collisions'] = dict(old=len(S_old) - len(k_old), new=len(S_new) - len(k_new), ev=len(S_ev) - len(k_ev))

# ---- control 0: tool + inputs reproduce the OLD file ---------------------------------------------------------------
a = json.load(open(OLD_BLOB, encoding='utf-8'))
b = json.load(open(OLD_REPRO, encoding='utf-8'))


def strip(d):
    d = copy.deepcopy(d)
    d['provenance'].pop('wall_s', None)
    return d


def norm_bytes(p):
    t = open(p, 'rb').read().decode('utf-8')
    return re.sub(r'"wall_s": [0-9.]+', '"wall_s": X', t)


c0 = dict(
    ego_equal=a['ego_footprint'] == b['ego_footprint'],
    jumps_equal=a['track_jumps'] == b['track_jumps'],
    schema_equal=a['schema'] == b['schema'],
    provenance_minus_wall_equal=strip(a)['provenance'] == strip(b)['provenance'],
    whole_minus_wall_equal=strip(a) == strip(b),
    old_wall_s=a['provenance']['wall_s'], repro_wall_s=b['provenance']['wall_s'],
    old_blob_md5=md5(OLD_BLOB), repro_md5=md5(OLD_REPRO),
    bytes_equal_after_wall_s_normalised=norm_bytes(OLD_BLOB) == norm_bytes(OLD_REPRO))
out['control0_old_reproduction'] = c0

# ---- control A: every old clip's entries unchanged -----------------------------------------------------------------
new = json.load(open(NEW, encoding='utf-8'))
oe, ov = a['ego_footprint']['frames'], a['track_jumps']['events']
ne_, nv = new['ego_footprint']['frames'], new['track_jumps']['events']


def cmp(keys):
    res = dict(n_compared=len(keys), ego_differ=0, ego_with_entry=0, jumps_differ=0, jumps_with_entry=0)
    for k in sorted(keys):
        if k in oe or k in ne_:
            res['ego_with_entry'] += 1
        if k in ov or k in nv:
            res['jumps_with_entry'] += 1
        if oe.get(k) != ne_.get(k):
            res['ego_differ'] += 1
        if ov.get(k) != nv.get(k):
            res['jumps_differ'] += 1
    return res


out['controlA_old_train_clips'] = cmp(k_old)
out['controlA_eval139_clips'] = cmp(k_ev)
out['controlA_old_train_plus_eval'] = cmp(k_old | k_ev)
out['controlA_every_old_key'] = dict(
    ego_old_keys=len(oe), ego_old_keys_equal_in_new=sum(1 for k in oe if ne_.get(k) == oe[k]),
    jump_old_keys=len(ov), jump_old_keys_equal_in_new=sum(1 for k in ov if nv.get(k) == ov[k]),
    old_ego_event_frames=sum(len(v) for v in oe.values()), old_jump_events=sum(len(v) for v in ov.values()))
extra_e = set(ne_) - set(oe)
extra_v = set(nv) - set(ov)
out['new_keys_not_in_old'] = dict(ego=len(extra_e), jumps=len(extra_v),
                                  ego_outside_add=len(extra_e - k_add), jumps_outside_add=len(extra_v - k_add),
                                  ego_in_eval=len(extra_e & k_ev), jumps_in_eval=len(extra_v & k_ev),
                                  ego_in_old_train=len(extra_e & k_old), jumps_in_old_train=len(extra_v & k_old))

# ---- ADD clips in the join -----------------------------------------------------------------------------------------
cen = json.load(open(CENSUS))
pc = cen['per_clip_sha12']
add_joined = sorted(k for k in k_add if k in pc)
add_joined_nonempty = sorted(k for k in add_joined if pc[k]['lines_nonempty_agents'] > 0)
meta = json.load(open(META3D))['per_clip']
meta_add = sorted(k for k in k_add if k in meta)
out['add_in_join'] = dict(
    census_n_lines=cen['n_lines'], census_n_clips=cen['n_clips'], census_bad_prefix=cen['bad_prefix_lines'],
    add_total=len(k_add), add_in_join_census=len(add_joined),
    add_in_join_nonempty_agents=len(add_joined_nonempty),
    add_in_join_train3d_meta=len(meta_add),
    census_vs_meta_agree=add_joined == meta_add,
    old_train_in_join=len([k for k in k_old if k in pc]), eval_in_join=len([k for k in k_ev if k in pc]),
    join_clips_in_none_of_old_add_eval=len([k for k in pc if k not in k_old and k not in k_add and k not in k_ev]),
    join_lines_add=sum(pc[k]['lines'] for k in add_joined),
    join_lines_total_for_old_plus_eval=sum(pc[k]['lines'] for k in pc if k in k_old or k in k_ev))
rows = []
for k in add_joined:
    e = ne_.get(k, [])
    v = nv.get(k, [])
    rows.append(dict(sha12=k, join_lines=pc[k]['lines'], lines_nonempty=pc[k]['lines_nonempty_agents'],
                     ego_frames=len(e), jump_events=len(v), ego_frame_list=e, jump_event_list=v))
out['add58_rows'] = rows
out['add58_totals'] = dict(clips=len(rows), clips_with_ego=sum(1 for r in rows if r['ego_frames']),
                           ego_frames=sum(r['ego_frames'] for r in rows),
                           clips_with_jumps=sum(1 for r in rows if r['jump_events']),
                           jump_events=sum(r['jump_events'] for r in rows))
out['add_entries_outside_join'] = len([k for k in list(ne_) + list(nv) if k in k_add and k not in pc])

# ---- loader --------------------------------------------------------------------------------------------------------
mo = H.JoinDefectMasks.load(OLD_BLOB)
mn = H.JoinDefectMasks.load(NEW)
so, sn = mo.stats(), mn.stats()
out['loader'] = dict(
    old=so, new=sn,
    delta=dict(n_ego_frames=sn['n_ego_frames'] - so['n_ego_frames'], n_ego_clips=sn['n_ego_clips'] - so['n_ego_clips'],
               n_track_events=sn['n_track_events'] - so['n_track_events'],
               n_track_event_clips=sn['n_track_event_clips'] - so['n_track_event_clips']),
    old_clip_ego_sets_equal=all(mn.ego_frames.get(k) == v for k, v in mo.ego_frames.items()),
    old_clip_event_sets_equal=all(mn.track_events.get(k) == v for k, v in mo.track_events.items()),
    rate_records_old=len(mo._rate_records), rate_records_new=len(mn._rate_records),
    rate_records_old_equal_in_new=all(mn._rate_records.get(k) == v for k, v in mo._rate_records.items()))
oh, nh = a, new
out['header'] = dict(
    old=dict(n_boxes=oh['ego_footprint']['n_boxes'], n_clips=oh['ego_footprint']['n_clips'],
             n_event_frames_all=oh['track_jumps']['n_event_frames_all'],
             pose_glitch=oh['track_jumps']['n_pose_glitch_like_excluded'],
             n_track_events_listed=oh['track_jumps']['n_track_events_listed'],
             n_lines=oh['provenance']['n_lines'], n_records=oh['provenance']['n_records']),
    new=dict(n_boxes=nh['ego_footprint']['n_boxes'], n_clips=nh['ego_footprint']['n_clips'],
             n_event_frames_all=nh['track_jumps']['n_event_frames_all'],
             pose_glitch=nh['track_jumps']['n_pose_glitch_like_excluded'],
             n_track_events_listed=nh['track_jumps']['n_track_events_listed'],
             n_lines=nh['provenance']['n_lines'], n_records=nh['provenance']['n_records']),
    rules_equal=(oh['ego_footprint']['rule'] == nh['ego_footprint']['rule']
                 and oh['track_jumps']['rule'] == nh['track_jumps']['rule']),
    new_provenance_join_md5=nh['provenance']['join_md5'], new_provenance_manifests=nh['provenance']['manifests'],
    new_class_counts=nh['provenance']['d3_aggregates']['ego_footprint']['class_counts'],
    old_class_counts=oh['provenance']['d3_aggregates']['ego_footprint']['class_counts'],
    new_frac_within_0p5=nh['provenance']['d3_aggregates']['ego_footprint']['frac_within_0.5m_of_(1.4,0)'],
    new_wall_s=nh['provenance']['wall_s'])
out['raw_id_leak_check'] = dict(non_sha12_keys=[k for k in list(ne_) + list(nv) if len(k) != 12])
raw = open(NEW, 'rb').read()
out['new_file'] = dict(path=NEW, md5=hashlib.md5(raw).hexdigest(), bytes=len(raw),
                       sha256=hashlib.sha256(raw).hexdigest(), has_cr=(b'\r' in raw))
json.dump(out, open(R + 'out/controls.json', 'w'), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k != 'add58_rows'}, indent=1, default=str))
