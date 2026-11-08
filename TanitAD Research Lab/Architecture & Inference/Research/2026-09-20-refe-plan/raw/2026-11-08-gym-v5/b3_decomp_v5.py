"""M-GYM-5 B3 decomposition on STORED cycle records (CPU, no simulation). Adapted from pilot_v4/b3_decomp/attrib.py.
Run with an ABSOLUTE path to the dirs holding states.jsonl.gz. Output: raw/b3_decomp_v5.json"""
import gzip, json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRCS = [os.path.join(HERE, 'raw', 'run_main', 'states.jsonl.gz'), os.path.join(HERE, 'raw', 'ladder_20_160', 'states.jsonl.gz')]
TERMS = ['q_free','q_gap','q_stall','q_yield','q_comf','q_osc','q_lane','q_cons']
RUNGS = ['crawl1.5','f0.3','f0.5','f0.7','f0.9','f1.0']
FLOOR0 = 0.20
res = json.load(open(os.path.join(HERE, 'raw', 'results_v5.json')))
fail = [e['idx'] for e in res['B3']['eligible_detail'] if not e['known']]
S = {}
for p in SRCS:
    for l in gzip.open(p, 'rt', encoding='utf-8'):
        d = json.loads(l)
        if d['idx'] in fail: S[d['idx']] = d
def ret_floor(R, fl):
    tot = 0.0
    for c in R['logs']:
        if c['hard']: continue
        qf, qp = c['q']['q_free'], c['qprod']
        if abs(qf - FLOOR0) < 1e-9 and fl != FLOOR0: qp = qp / qf * fl
        tot += c['progress'] * qp
    return tot
def hardfree(R): return not any(c['hard'] for c in R['logs'])
out = {}
for i in sorted(S):
    d = S[i]; L = d['ladder']; lim = d['limit']
    best = max(L['f0.9']['ret'], L['f1.0']['ret']); pbest = max(sum(c['progress'] for c in L[k]['logs']) for k in ('f0.9','f1.0'))
    o = {'limit': lim, 'v0': d['v0'], 'best': best, 'rungs': {}, 'failing_conditions': []}
    for rk in RUNGS:
        R = L[rk]; n = len(R['logs']); loss = {t: 0.0 for t in TERMS}; atfloor = 0; active = {t: 0 for t in TERMS}
        for c in R['logs']:
            if c['hard']: continue
            if abs(c['q']['q_free'] - FLOOR0) < 1e-9: atfloor += 1
            for t in TERMS: active[t] += bool(c['active'][t])
            lq = -sum(math.log(max(c['q'][t], 1e-9)) for t in TERMS)
            if lq <= 0: continue
            tl = c['progress'] * (1 - c['qprod'])
            for t in TERMS: loss[t] += tl * (-math.log(max(c['q'][t], 1e-9))) / lq
        prog = sum(c['progress'] for c in R['logs'])
        o['rungs'][rk] = {'ret': R['ret'], 'hard_free': hardfree(R), 'ret_over_best': R['ret']/best, 'prog_over_best': prog/pbest,
            'n_cycles': n, 'cycles_q_free_at_floor': atfloor, 'loss_by_term': {t: round(v,4) for t, v in loss.items() if v > 0.005},
            'cycles_active': {t: v for t, v in active.items() if v}}
        if rk != 'f0.9' and rk != 'f1.0' and hardfree(R):
            if R['ret'] >= 0.85*best: o['failing_conditions'].append(f'{rk} ret/best={R["ret"]/best:.3f} >= 0.85')
            if rk == 'crawl1.5' and R['ret'] > 0.10*best: o['failing_conditions'].append(f'crawl ret/best={R["ret"]/best:.3f} > 0.10')
    o['floor_cf'] = {}
    for fl in (0.20, 0.15, 0.10, 0.05, 0.0):
        r = {rk: ret_floor(L[rk], fl) for rk in RUNGS}; b = max(r['f0.9'], r['f1.0'])
        ok = all(r[k]/b < 0.85 for k in ('crawl1.5','f0.3','f0.5','f0.7') if hardfree(L[k])) and (r['crawl1.5']/b <= 0.10 if hardfree(L['crawl1.5']) else True)
        o['floor_cf'][str(fl)] = {'crawl_over_best': r['crawl1.5']/b, 'f0.7_over_best': r['f0.7']/b, 'would_be_known': ok}
    c = o['rungs']['crawl1.5']
    o['crawl_dominant_loss_term'] = max(c['loss_by_term'], key=c['loss_by_term'].get) if c['loss_by_term'] else None
    o['crawl_prog_over_best'] = c['prog_over_best']
    out[i] = o
json.dump(out, open(os.path.join(HERE, 'raw', 'b3_decomp_v5.json'), 'w'), indent=1)
for i, o in out.items():
    c = o['rungs']['crawl1.5']
    print(i, 'lim %.1f v0 %.1f' % (o['limit'], o['v0']), '| crawl/best %.3f prog/best %.3f atfloor %d/%d loss %s' % (c['ret_over_best'], c['prog_over_best'], c['cycles_q_free_at_floor'], c['n_cycles'], c['loss_by_term']))
    print('   fails:', o['failing_conditions'], '| floor0.15 known:', o['floor_cf']['0.15']['would_be_known'], 'floor0.0 known:', o['floor_cf']['0.0']['would_be_known'], 'f0.7/best %.3f' % o['rungs']['f0.7']['ret_over_best'])
