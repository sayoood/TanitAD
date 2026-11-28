import json, sys, re, collections
sys.path.insert(0, 'C:/Users/Admin/par18/s0probe')
from common_s0 import *
logs, tok_log, by = s0()
L = set(logs)
raw = json.load(open('C:/Users/Admin/par18/s0probe/probe_s0_raw.json'))['logs']
none_logs = {l for l, r in raw.items() if r.get('tokens_all4cams_resolved', 0) == 0}
lines = [l.rstrip('\n') for l in open('D:/Projects/TanitAD/data/openscene_probe/members.txt', encoding='utf-8')]
print('members lines', len(lines), lines[:3])
mem = collections.defaultdict(set)   # (log) -> set of "CAM/hash.jpg"
for l in lines:
    p = l.strip().replace(chr(92), '/').split('/')
    # find the log segment
    for i, seg in enumerate(p):
        if re.match(r'^20\d\d\.\d\d\.\d\d\.\d\d\.\d\d\.\d\d_veh-', seg):
            if i + 2 < len(p) and p[i + 2].endswith('.jpg'):
                mem[seg].add(p[i + 1] + '/' + p[i + 2])
            break
print('logs in shard', len(mem), 'in S0', len(set(mem) & L), 'in S0-missing-76', len(set(mem) & none_logs))
# needed frames of those logs: use the S0 staged manifest as the list of needed frames for none logs
B = SR + '/run/batches'
need = collections.defaultdict(set)
import os
for u in os.listdir(B):
    pf = os.path.join(B, u, 'copy_manifest_frames.tsv')
    if os.path.exists(pf):
        for l in open(pf, encoding='utf-8'):
            q = l.rstrip('\n').split('\t')
            if len(q) >= 2:
                lg, cam, fn = q[0].split('/')
                need[lg].add(cam + '/' + fn)
tot_need = 0
tot_have = 0
per = {}
for lg in set(mem) & none_logs:
    h = len(need[lg] & mem[lg])
    per[lg] = {'staged_frames_needed': len(need[lg]), 'in_shard': h}
    tot_need += len(need[lg])
    tot_have += h
print('for none-logs in shard: needed (S0 manifest)', tot_need, 'present in shard', tot_have)
json.dump({'shard': 'D:/Projects/TanitAD/data/openscene_probe/navtrain_current_25.tgz', 'members_lines': len(lines), 'logs_in_shard': len(mem), 'S0_logs_in_shard': sorted(set(mem) & L),
           'S0_missing_logs_in_shard': sorted(set(mem) & none_logs), 'needed_vs_present': per, 'needed_total': tot_need, 'present_total': tot_have}, open('C:/Users/Admin/par18/s0probe/tgz_cov.json', 'w'), indent=0)
