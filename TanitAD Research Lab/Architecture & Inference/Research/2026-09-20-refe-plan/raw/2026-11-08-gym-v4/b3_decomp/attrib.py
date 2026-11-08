"""B3 attribution + exact floor counterfactual from STORED cycle records (CPU, no simulation).
Floor transform is exact for scripted ladder policies: trajectories do not depend on the reward, and a cycle's
q_free changes only if it sat at the floor (q_free == FLOOR0). Slope/knee unchanged."""
import gzip, json, os, math
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'raw', 'run_main', 'states.jsonl.gz')
TERMS = ['q_free','q_gap','q_stall','q_yield','q_comf','q_osc','q_lane','q_cons']
FLOOR0 = 0.3
RUNGS = ['crawl1.5','f0.3','f0.5','f0.7','f0.9','f1.0']
S = {}
for l in gzip.open(SRC, 'rt'):
    d = json.loads(l); S[d['idx']] = d

def ret_floor(R, fl):
    tot = 0.0
    for c in R['logs']:
        if c['hard']: continue
        qf = c['q']['q_free']
        qp = c['qprod']
        if abs(qf - FLOOR0) < 1e-9 and fl != FLOOR0:
            qp = qp / qf * fl
        tot += c['progress'] * qp
    return tot

out = {'attribution': {}, 'floor_cf': {}, 'stall_cycles': {}, 'all20_floor_effect': {}}
# 1. attribution: share of each cycle's (1-qprod) loss by log share, plus progress shortfall vs the 1.0x rung
for i in (3, 11, 17):
    d = S[i]; best = d['ladder']['f1.0']
    pbest = sum(c['progress'] for c in best['logs']); rbest = best['ret']
    out['attribution'][i] = {'limit': d['limit'], 'v0': d['v0'], 'best_ret': rbest, 'best_prog': pbest, 'rungs': {}}
    for rk in RUNGS:
        R = d['ladder'][rk]; loss = {t: 0.0 for t in TERMS}; prog = sum(c['progress'] for c in R['logs'])
        for c in R['logs']:
            if c['hard']: continue
            lq = -sum(math.log(max(c['q'][t],1e-9)) for t in TERMS)
            if lq <= 0: continue
            tot_loss = c['progress'] * (1 - c['qprod'])
            for t in TERMS:
                loss[t] += tot_loss * (-math.log(max(c['q'][t],1e-9))) / lq
        # cycles where q_free is inapplicable because a conflict agent exists although v < 0.9*limit
        fa = [c['cycle'] for c in R['logs'] if (not c['q']['q_free'] < 1) and c['conf_ttc'] is not None and c['lead_gap'] is None and c['v'] < 0.9*d['limit']]
        out['attribution'][i]['rungs'][rk] = {'ret': R['ret'], 'sum_prog': prog, 'prog_over_best': prog/pbest,
            'ret_over_best': R['ret']/rbest, 'loss_by_term': loss, 'qfree_off_by_conflict_while_slow_cycles': fa}
# 2. floor counterfactual
FL = [0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
for i in (3, 11, 17):
    d = S[i]; out['floor_cf'][i] = {}
    for fl in FL:
        r = {rk: ret_floor(d['ladder'][rk], fl) for rk in RUNGS}
        b = max(r['f0.9'], r['f1.0'])
        out['floor_cf'][i][str(fl)] = {'rets': r, 'crawl_over_best': r['crawl1.5']/b,
            'slow_over_best': {k: r[k]/b for k in ('crawl1.5','f0.3','f0.5','f0.7')},
            'd_f0.9': r['f0.9']/d['ladder']['f0.9']['ret']-1, 'd_f1.0': r['f1.0']/d['ladder']['f1.0']['ret']-1}
# 3. q_stall cycles, all rungs, all 20 states
for i, d in S.items():
    for rk in RUNGS:
        cs = [c['cycle'] for c in d['ladder'][rk]['logs'] if c['active']['q_stall']]
        if cs: out['stall_cycles'][f'{i}:{rk}'] = cs
# 4. effect of each floor on 0.9x/1.0x rungs over all 20 states
for fl in FL[1:]:
    ds = []
    for i, d in S.items():
        for rk in ('f0.9','f1.0'):
            R = d['ladder'][rk]
            if any(c['hard'] for c in R['logs']): continue
            ds.append(ret_floor(R, fl)/R['ret']-1)
    out['all20_floor_effect'][str(fl)] = {'n_hardfree_cells': len(ds), 'min': min(ds), 'mean': sum(ds)/len(ds)}
json.dump(out, open(os.path.join(HERE, 'attrib.json'), 'w'), indent=1)
for i in (3, 11, 17):
    print('state', i)
    for rk in RUNGS:
        a = out['attribution'][i]['rungs'][rk]
        print('  %-9s ret/best=%.3f prog/best=%.3f loss: %s qfree_off_slow=%s' % (rk, a['ret_over_best'], a['prog_over_best'],
              {t[2:]: round(v,2) for t, v in a['loss_by_term'].items() if v > 0.005}, a['qfree_off_by_conflict_while_slow_cycles']))
    for fl in FL:
        f = out['floor_cf'][i][str(fl)]
        print('  floor %.2f crawl/best %.3f slow %s d0.9 %.4f d1.0 %.4f' % (fl, f['crawl_over_best'], {k[:5]: round(v,3) for k, v in f['slow_over_best'].items()}, f['d_f0.9'], f['d_f1.0']))
print('stall', out['stall_cycles'])
print(json.dumps(out['all20_floor_effect']))
