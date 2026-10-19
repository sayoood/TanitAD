import json, collections
d = json.load(open('C:/Users/Admin/par18/s0probe/probe_s0_raw.json'))['logs']
full = [l for l, r in d.items() if r.get('tokens_all4cams_resolved') == r['n_tokens']]
part = [l for l, r in d.items() if 0 < r.get('tokens_all4cams_resolved', 0) < r['n_tokens']]
none = [l for l, r in d.items() if r.get('tokens_all4cams_resolved', 0) == 0]
print('full', len(full), sum(d[l]['n_tokens'] for l in full))
print('partial', len(part), [(l, d[l]['tokens_all4cams_resolved'], d[l]['n_tokens']) for l in part])
print('none', len(none), sum(d[l]['n_tokens'] for l in none))
# roots used by full logs
c = collections.Counter()
for l in full:
    for k, v in d[l]['tokens_via_root'].items():
        c[k] += v
print(c)
# roots present for 'none' logs
c2 = collections.Counter()
for l in none:
    for k in d[l]['roots_present']:
        c2[k] += 1
print('roots with any file for none-logs', c2)
# for 'none' logs: n_needed_frames, missing frames
nf = sum(d[l]['n_needed_frames'] for l in none)
mf = sum(d[l]['n_missing_frames'] for l in none)
print('none-logs needed frames', nf, 'missing', mf, 'known-size missing', sum(d[l]['missing_frames_with_known_size'] for l in none), sum(d[l]['missing_frames_bytes_known'] for l in none))
# history of the none logs
c3 = collections.Counter(d[l]['hist'].get('unit', '?')[0] for l in none)
print('S0 unit class of none-logs', c3)
c4 = collections.Counter(d[l]['hist'].get('unit', '?')[0] for l in full)
print('S0 unit class of full-logs', c4)
