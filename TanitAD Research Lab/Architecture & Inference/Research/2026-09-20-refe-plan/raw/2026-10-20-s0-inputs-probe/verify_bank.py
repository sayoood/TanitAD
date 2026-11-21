import os, sys, json, hashlib, glob, collections
import numpy as np
sys.path.insert(0, 'C:/Users/Admin/par18/s0probe')
from common_s0 import *
logs, tok_log, by = s0()
L = set(logs)
BK = SR + '/work/bank'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


man = json.load(open(BK + '/MANIFEST.sha256.json'))
print('manifest keys', list(man.keys())[:8])
table = man.get('table') or man.get('files') or {}
names = sorted(os.listdir(BK))
bad = []
chk = 0
for n in names:
    if n.endswith('.npz') and n in table:
        chk += 1
        if sha(os.path.join(BK, n)) != table[n]:
            bad.append(n)
print('manifest entries', len(table), 'npz checked', chk, 'sha mismatches', bad)
# counts
fan_tok = 0
sets_n = 0
picks_ok = 0
logs_seen = set()
fan_tokens = set()
for n in sorted(glob.glob(BK + '/*.fan.npz')):
    z = np.load(n, allow_pickle=False)
    fan_tok += len(z['token'])
    fan_tokens |= set(map(str, z['token']))
    assert z['logits'].shape[1:] == (64, 6) and z['pick'].shape[0] == z['token'].shape[0]
    assert z['proposals'].shape[0] == 0
    picks_ok += int(z['pick'].shape[0])
bank_keys = {}
for n in sorted(glob.glob(BK + '/*.sets.npz')):
    z = np.load(n, allow_pickle=False)
    lg = z['log']
    tk = z['token']
    st = z['step']
    rk = z['rank']
    lab = z['labels6']
    for i in range(len(lg)):
        bank_keys[(str(lg[i]), str(tk[i]), int(st[i]), int(rk[i]))] = (n, i)
    sets_n += len(lg)
    logs_seen |= set(map(str, lg))
print('fan tokens', fan_tok, 'distinct', len(fan_tokens), 'in S0', len(fan_tokens & set(tok_log)), '| sets', sets_n, 'distinct keys', len(bank_keys), 'logs', len(logs_seen), 'logs in S0', len(logs_seen & L))
# stored sets (trajectories)
tot_sets = 0
join = 0
same_lab = 0
for lg in logs:
    p = SR + '/work/sets/%s.npz' % lg
    z = np.load(p, allow_pickle=False)
    tot_sets += len(z['token'])
    tr = z['traj3']
    assert tr.shape[1:] == (64, 20, 3), tr.shape
    for i in range(len(z['token'])):
        k = (lg, str(z['token'][i]), int(z['step'][i]), int(z['rank'][i]))
        if k in bank_keys:
            join += 1
print('work/sets stores', len(logs), 'sets', tot_sets, 'joined by key to bank', join)
# ckpt
ck = 'D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt'
print('ckpt exists', os.path.exists(ck), os.path.getsize(ck) if os.path.exists(ck) else None)
ckh = sha(ck) if os.path.exists(ck) else None
print('ckpt sha256', ckh)
json.dump({'manifest_entries': len(table), 'npz_checked': chk, 'sha_mismatch': bad, 'fan_tokens': fan_tok, 'fan_tokens_in_S0': len(fan_tokens & set(tok_log)), 'sets_rows': sets_n, 'distinct_set_keys': len(bank_keys),
           'bank_logs': len(logs_seen), 'work_sets_stores': len(logs), 'work_sets_rows': tot_sets, 'work_sets_joined_to_bank': join, 'ckpt_sha256': ckh,
           'ckpt_bytes': os.path.getsize(ck) if os.path.exists(ck) else None}, open('C:/Users/Admin/par18/s0probe/verify_bank.json', 'w'), indent=1)
