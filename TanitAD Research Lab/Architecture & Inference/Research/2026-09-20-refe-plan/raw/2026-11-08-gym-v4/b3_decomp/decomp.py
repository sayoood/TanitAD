"""B3 decomposition from stored cycle records (no rerun). Reads raw/run_main/states.jsonl.gz.
Per state, rung, cycle: v, progress, qprod, per-term q, lead gap, active flags. Writes b3_decomp.json."""
import gzip, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'raw', 'run_main', 'states.jsonl.gz')
KEEP = (3, 11, 17)
out = {}
for l in gzip.open(SRC, 'rt'):
    d = json.loads(l)
    if d['idx'] not in KEEP: continue
    s = {'state': d['state'], 'limit': d['limit'], 'v0': d['v0'], 'rungs': {}}
    for rk, R in d['ladder'].items():
        cyc = []
        for c in R['logs']:
            cyc.append({k: c.get(k) for k in ('cycle','hard','reward','progress','v','a','qprod','q','lead_gap','conf_ttc','progress_legacy')})
        s['rungs'][rk] = {'vt': R['vt'], 'ret': R['ret'], 'cycles': cyc}
    out[str(d['idx'])] = s
json.dump(out, open(os.path.join(HERE, 'b3_decomp.json'), 'w'), indent=1)
TERMS = ['q_free','q_gap','q_stall','q_yield','q_comf','q_osc','q_lane','q_cons']
for i, s in out.items():
    print('=== state', i, s['state']['sampler'], 'limit %.2f v0 %.2f' % (s['limit'], s['v0']))
    for rk, R in s['rungs'].items():
        cy = R['cycles']
        tot = sum(c['reward'] for c in cy)
        prog = sum(c['progress'] for c in cy)
        mq = {t: sum(c['q'][t] for c in cy)/len(cy) for t in TERMS}
        # reward-weighted loss attribution: sum progress*(1-qprod)
        print(' %-9s vt=%.2f ret=%.3f n=%d sumprog=%.3f meanv=%.2f hard=%s | ' % (rk, R['vt'], tot, len(cy), prog, sum(c['v'] for c in cy)/len(cy), [c['hard'] for c in cy if c['hard']]) +
              ' '.join('%s=%.2f' % (t[2:], mq[t]) for t in TERMS if mq[t] < 0.999))
