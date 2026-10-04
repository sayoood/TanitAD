import json, numpy as np
from pathlib import Path
base = Path('.')
out = {}
# cross-environment mirror: Thor vs dev box, eval139
a = np.load('raw/thor_eval139/windows_eval139.npz'); b = np.load('raw/devbox_eval/windows_eval139.npz')
diff = [k for k in a.files if not np.array_equal(a[k], b[k], equal_nan=(a[k].dtype.kind == 'f'))]
out['X_thor_vs_devbox_eval139'] = {'n_columns': len(a.files), 'n_windows': int(len(a['t'])), 'columns_not_bit_identical': diff,
    'note': 'same code, different OS / torch / data copies (Thor /home/nvidia/data vs D:/refcv6_eval_kit); every column bit-identical'}
tr = json.load(open('raw/analysis/numbers_train.json')); ev = json.load(open('raw/analysis/numbers_eval139.json'))
rt = json.load(open('raw/thor_train/record_train.json', encoding='utf-8')); re_ = json.load(open('raw/thor_eval139/record_eval139.json', encoding='utf-8'))
cfg = json.load(open('D:/refcv7_eval_kit/ckpt/config.json', encoding='utf-8'))
ag = cfg['agent_join_stats']
out['B_config_json_stamps'] = {
  'train': {'n_windows': (tr['n_windows'], 746946), 'agent_labelled': (tr['control_config']['agent_labelled'], ag['train']['n_windows_labelled']),
            'agent_clear': (tr['control_config']['agent_labelled_clear'], ag['train']['n_windows_labelled_clear']),
            'map_ok': (tr['control_config']['map_ok'], cfg['map_hires']['map_fine_stats']['train']['states']['ok']),
            'ceiling_fed': (tr['control_config']['ceiling_fed'], cfg['refcv6_max_speed']['train']['n_windows_fed']),
            'nav_clips_follow_left_right': (tr['control_config']['nav_clip_counts'], {k: cfg['nav_from_v7_stats']['train'][k] for k in ('follow', 'left', 'right')}),
            'n_clips': (tr['n_clips'], 4369),
            'g3_unverified_clock_clips': (rt['build']['clock']['g3']['n_unverified_clips'], cfg['label_clock']['train']['g3']['n_unverified_clips']),
            'builder_census_n_windows_labelled': (rt['build']['agent_join']['n_windows_labelled'], ag['train']['n_windows_labelled'])},
  'eval': {'n_windows': (ev['n_windows'], 23772), 'agent_labelled': (ev['control_config']['agent_labelled'], ag['eval']['n_windows_labelled']),
           'agent_clear': (ev['control_config']['agent_labelled_clear'], ag['eval']['n_windows_labelled_clear']),
           'map_ok': (ev['control_config']['map_ok'], cfg['map_hires']['map_fine_stats']['eval']['states']['ok']),
           'nav_clips_follow_left_right': (ev['control_config']['nav_clip_counts'], {k: cfg['nav_from_v7_stats']['eval'][k] for k in ('follow', 'left', 'right')})}}
def allmatch(d):
    return all((v[0] == v[1]) for v in d.values())
out['B_config_json_stamps']['train_all_equal'] = allmatch(out['B_config_json_stamps']['train'])
out['B_config_json_stamps']['eval_all_equal'] = allmatch(out['B_config_json_stamps']['eval'])
out['C_in_band_literal'] = {'train': tr['control_in_band_literal'], 'eval': ev['control_in_band_literal']}
out['A_route_following_package'] = ev['control_route_following_grid']
out['D_pose_vs_label_sign_and_units'] = {'train': rt['pose_controls'], 'eval': re_['pose_controls']}
out['F_vhi_vs_pose_vmax_at_anchor'] = {'train': tr['control_vhi_vs_pose_vmax_at_anchor'], 'eval': ev['control_vhi_vs_pose_vmax_at_anchor']}
out['C2_direct_equals_getitem'] = {'train': rt['C2_direct_equals_getitem'], 'eval': re_['C2_direct_equals_getitem']}
out['K_training_log'] = json.load(open('raw/analysis/controls_vs_training_log.json'))
json.dump(out, open('raw/analysis/controls_summary.json', 'w', encoding='utf-8'), indent=1, default=str)
print(json.dumps({k: (v if k in ('X_thor_vs_devbox_eval139',) else '...') for k, v in out.items()}, indent=1, default=str))
print(out['B_config_json_stamps']['train_all_equal'], out['B_config_json_stamps']['eval_all_equal'])
for k, v in out['B_config_json_stamps']['train'].items(): print('train', k, v)
for k, v in out['B_config_json_stamps']['eval'].items(): print('eval', k, v)
