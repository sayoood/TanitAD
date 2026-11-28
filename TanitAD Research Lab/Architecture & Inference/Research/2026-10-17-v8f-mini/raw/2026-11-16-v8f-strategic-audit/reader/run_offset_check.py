"""Deliverable 1 check: which table step belongs to an ACTSENS window's NOW (audit guess: now + 3, UNVERIFIED)."""
import sys, json
sys.path.insert(0, 'F:/TanitAD-scratch/strategic/build/src')
import numpy as np
import strategic_text as ST
tab = ST.load_table('F:/TanitAD-scratch/datascale/navtok/navtok_table.npz')
z = np.load('F:/TanitAD-scratch/actsens/read/out/NULL_s3/win/windows.npz', allow_pickle=True)
res = ST.verify_offset(tab, 'F:/TanitAD-scratch/datascale/navtok/raw/join_eval141.npz', 'E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/native_tracks.pt', z['eid'], z['now'], z['v0'])
a = ST.attach(tab, z['eid'], z['now'])
res['attach_at_NOW_OFFSET'] = {'offset': ST.NOW_OFFSET, 'n_windows': int(len(z['eid'])), 'text_ok': int(a['text_ok'].sum()), 'flag_counts': {int(k): int((a['flag'] == k).sum()) for k in np.unique(a['flag'])}}
res['evidence'] = 'MEASURED: windows.npz (NULL_s3), join_eval141.npz now_s, native_tracks.pt speed column, navtok_table.npz sha256 2836369e'
json.dump(res, open('F:/TanitAD-scratch/strategic/build/out/STRATEGIC_TEXT_OFFSET_CHECK.json', 'w'), indent=1)
print(json.dumps({k: res[k] for k in ('best_offset', 'by_offset', 'attach_at_NOW_OFFSET')}, indent=1))
print({k: round(v.get('frac_within_0p05', -1), 3) for k, v in res['shift_scan_at_offset0'].items()})
