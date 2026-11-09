"""REWARD-2 counterfactuals on STORED REWARD-1 cycle records (CPU only, no simulation, no GPU). Reads pilot_v5/raw only.
Output: pilot_v6/reward2_counterfactuals.json. Run: python reward2_counterfactuals.py  (paths derived from this file)."""
import gzip, json, os
def _clip(x, lo, hi): return max(lo, min(hi, x))
def _mean(xs): xs = list(xs); return sum(xs) / len(xs)
HERE = os.path.dirname(os.path.abspath(__file__)); V5 = os.path.join(os.path.dirname(HERE), 'pilot_v5', 'raw')
RUNGS = ['crawl1.5', 'f0.3', 'f0.5', 'f0.7', 'f0.9', 'f1.0']
DT, CYC = 0.1, 5; CT = DT * CYC          # cycle 0.5 s
S = {}
for sub in ('run_main', 'ladder_20_160'):
    for l in gzip.open(os.path.join(V5, sub, 'states.jsonl.gz'), 'rt', encoding='utf-8'):
        d = json.loads(l); S[d['idx']] = d
res = json.load(open(os.path.join(V5, 'results_v5.json')))
ELIG = [e['idx'] for e in res['B3']['eligible_detail']]
KNOWN = {e['idx']: e['known'] for e in res['B3']['eligible_detail']}
FAIL = sorted(i for i in ELIG if not KNOWN[i]); OK = sorted(i for i in ELIG if KNOWN[i])
assert len(ELIG) == 39 and len(FAIL) == 10 and len(OK) == 29


def hard_any(R): return any(c['hard'] for c in R['logs'])


def known_from(rets, hards):
    """gym_env.ladder_known (lines 958-970): slow hard-free rungs < 0.85 best AND crawl <= 0.10 best (crawl regardless of hard)."""
    best = max(rets['f0.9'], rets['f1.0'])
    slow = [rets[n] for n in ('crawl1.5', 'f0.3', 'f0.5', 'f0.7') if not hards[n]]
    return bool(best > 0 and all(r < 0.85 * best for r in slow) and rets['crawl1.5'] <= 0.10 * best)


def rets_of(d, w=None):
    out = {}
    for rk in RUNGS:
        R = d['ladder'][rk]; tot = 0.0
        for i, c in enumerate(R['logs']):
            tot += c['reward'] * (1.0 if w is None else w(d, rk, i, c))
        out[rk] = tot
    return out


def hards_of(d): return {rk: hard_any(d['ladder'][rk]) for rk in RUNGS}


for i in ELIG:   # sanity: reproduces the stored verdicts exactly
    assert known_from(rets_of(S[i]), hards_of(S[i])) == KNOWN[i], i
    assert abs(rets_of(S[i])['crawl1.5'] - S[i]['ladder']['crawl1.5']['ret']) < 1e-9
out = {'n_eligible': 39, 'fail_idx': FAIL, 'known_idx': OK}


def free_geom(c): return c['lead_gap'] is None and c['conf_ttc'] is None


# ---------------- Q1
q1 = {}
d = S[108]; norm = d['limit'] * CT; rows = {}
for rk in RUNGS:
    cum = 0.0; r = []
    for c in d['ladder'][rk]['logs']:
        cum += c['progress_legacy'] * norm
        r.append((c['cycle'], round(cum, 2), free_geom(c), bool(c['active']['q_free'])))
    rows[rk] = r
q1['state108_cycle_table_(cycle,cum_dist_m,no_lead_no_conf,q_free_applicable)'] = rows
inact = {rk: [x[1] for x in rows[rk] if x[2] and not x[3]] for rk in RUNGS}
act = {rk: [x[1] for x in rows[rk] if x[2] and x[3]] for rk in RUNGS}
q1['state108_cumdist_range_(min,max,n)_free_geometry_but_inactive'] = {k: (min(v), max(v), len(v)) if v else None for k, v in inact.items()}
q1['state108_cumdist_range_(min,max,n)_free_geometry_and_active'] = {k: (min(v), max(v), len(v)) if v else None for k, v in act.items()}
q1['state108_any_lead_or_conf_on_any_rung'] = any(not free_geom(c) for rk in RUNGS for c in d['ladder'][rk]['logs'])
gen = {}
for i, dd in S.items():
    if not dd.get('ladder') or 'crawl1.5' not in dd['ladder']: continue
    n = sum(1 for c in dd['ladder']['crawl1.5']['logs'] if free_geom(c) and not c['active']['q_free'] and not c['hard'])
    if n: gen[i] = (n, len(dd['ladder']['crawl1.5']['logs']))
q1['n_states_with_ladder_total'] = sum(1 for dd in S.values() if dd.get('ladder') and 'crawl1.5' in dd['ladder'])
q1['states_crawl_free_geometry_cycles_with_q_free_not_applicable_(n_cycles,total)'] = {str(k): v for k, v in sorted(gen.items())}
q1['of_the_39_eligible'] = {str(i): gen[i] for i in ELIG if i in gen}
out['Q1'] = q1

# ---------------- Q2
table = {}
for i in ELIG:
    d = S[i]
    table[i] = dict(known=KNOWN[i], v0=d['v0'], limit=d['limit'], n_obj0=d['n_obj0'], sampler=d['state'].get('sampler'),
                    agent_at_cycle0_any_rung=any(not free_geom(d['ladder'][rk]['logs'][0]) for rk in RUNGS if d['ladder'][rk]['logs']),
                    agent_in_window_crawl=any(not free_geom(c) for c in d['ladder']['crawl1.5']['logs']),
                    agent_in_window_any_rung=any(not free_geom(c) for rk in RUNGS for c in d['ladder'][rk]['logs']))
out['Q2_state_table'] = {str(k): v for k, v in table.items()}
defs = {}


def tally(name, keep):
    kk = [i for i in ELIG if keep(table[i])]
    defs[name] = dict(n_kept=len(kk), kept_known=sum(KNOWN[i] for i in kk), kept_not_known=sum(not KNOWN[i] for i in kk),
                      dropped_known=sum(KNOWN[i] for i in ELIG if i not in kk), dropped_not_known=sum(not KNOWN[i] for i in ELIG if i not in kk),
                      not_known_kept_idx=[i for i in kk if not KNOWN[i]], not_known_dropped_idx=[i for i in FAIL if i not in kk])


TAUS = (1.0, 1.5, 3.0, 5.0)
for tau in TAUS: tally(f'v0>={tau}', lambda t, tau=tau: t['v0'] >= tau)
for an in ('agent_at_cycle0_any_rung', 'agent_in_window_crawl', 'agent_in_window_any_rung'):
    tally(f'no:{an}', lambda t, an=an: not t[an])
    for tau in TAUS: tally(f'v0>={tau} & no:{an}', lambda t, an=an, tau=tau: t['v0'] >= tau and not t[an])
out['Q2_defs'] = defs


# ---------------- Q3
def ds_ref(v0, t0, a_nom, vmax):
    v0 = max(v0, 0.0); tacc = (vmax - v0) / a_nom if vmax > v0 else 0.0

    def s(t):
        tt = min(t, tacc); return v0 * tt + 0.5 * a_nom * tt * tt + (vmax * (t - tt) if t > tt else 0.0)
    return s(t0 + CT) - s(t0)


def make_w(kind, gamma, a_nom=2.0):
    def w(d, rk, i, c):
        lim = d['limit']; ds = c['progress_legacy'] * lim * CT          # progress_legacy = ds/(limit*0.5), clip [0,1.5] (gym_env.py:596)
        ref = ds_ref(d['v0'], i * CT, a_nom, lim) if kind == 'reach' else 0.9 * lim * CT
        return min(1.0, ds / ref) ** gamma if ref > 1e-9 else 1.0
    return w


forms = {'P1 reach a=2.0 gamma=1': make_w('reach', 1, 2.0), 'P2 reach a=2.0 gamma=2': make_w('reach', 2, 2.0),
         'P3 reach a=1.0 gamma=1': make_w('reach', 1, 1.0), 'P4 flat 0.9*limit gamma=1': make_w('flat', 1),
         'P5 flat 0.9*limit gamma=2': make_w('flat', 2)}
q3 = {}
for name, w in forms.items():
    nk = {}; ret = {}; cb = {}
    for i in ELIG:
        d = S[i]; r0 = rets_of(d); r1 = rets_of(d, w); h = hards_of(d)
        nk[i] = known_from(r1, h); b1 = max(r1['f0.9'], r1['f1.0'])
        cb[i] = dict(crawl_over_best=r1['crawl1.5'] / b1, f0_3=r1['f0.3'] / b1, f0_5=r1['f0.5'] / b1, f0_7=r1['f0.7'] / b1)
        ret[i] = dict(f0_9=r1['f0.9'] / r0['f0.9'], f1_0=r1['f1.0'] / r0['f1.0'])
    rf = [i for i in FAIL if nk[i]]; rg = [i for i in OK if not nk[i]]
    st = lambda k, sel: dict(mean=float(_mean([ret[i][k] for i in sel])), min=float(min(ret[i][k] for i in sel)), max=float(max(ret[i][k] for i in sel)))
    rest = [i for i in ELIG if S[i]['v0'] < 1.0]
    q3[name] = dict(fixed_of_10=len(rf), fixed_idx=rf, regressions_of_29=len(rg), regression_idx=rg,
                    still_failing_idx=[i for i in FAIL if not nk[i]], total_known_after=sum(nk.values()),
                    retention_f0_9_all39=st('f0_9', ELIG), retention_f1_0_all39=st('f1_0', ELIG),
                    retention_f0_9_launch_from_rest=st('f0_9', rest), retention_f1_0_launch_from_rest=st('f1_0', rest), n_launch_from_rest=len(rest),
                    per_state_fail={str(i): cb[i] for i in FAIL}, per_state_regressions={str(i): cb[i] for i in rg})
out['Q3'] = q3
out['Q3_baseline_fail'] = {}
for i in FAIL:
    r0 = rets_of(S[i]); b = max(r0['f0.9'], r0['f1.0'])
    out['Q3_baseline_fail'][str(i)] = dict(crawl_over_best=r0['crawl1.5'] / b, f0_3=r0['f0.3'] / b, f0_5=r0['f0.5'] / b, f0_7=r0['f0.7'] / b)


# ---------------- Q4: kinematics with the rung definition (gym_env.py:749-765): accel<=2, decel<=3, 0.1 s steps, 17 cycles, q=1
def kin_progress(v0, vt, limit):
    vs = v0; tot = 0.0; norm = limit * CT; v00 = max(v0, 0.0)
    for cyc in range(17):
        ds = 0.0
        for _ in range(CYC):
            vs = vs + _clip(vt - vs, -3.0 * DT, 2.0 * DT); ds += vs * DT

        def s(t):
            tt = min(t, v00 / 2.0); return v00 * tt - 0.5 * 2.0 * tt * tt     # brake baseline 2.0 (gym_env.py:461-466)
        ref = s(cyc * CT + CT) - s(cyc * CT)
        tot += _clip((ds - ref) / norm, 0.0, 1.5)
    return tot


q4 = {}
for i in ELIG:
    d = S[i]; lim = d['limit']; v0 = d['v0']
    pc = kin_progress(v0, 1.5, lim); pb = max(kin_progress(v0, 0.9 * lim, lim), kin_progress(v0, lim, lim))
    ec = sum(c['progress'] for c in d['ladder']['crawl1.5']['logs'])
    eb = max(sum(c['progress'] for c in d['ladder'][k]['logs']) for k in ('f0.9', 'f1.0'))
    q4[str(i)] = dict(known=KNOWN[i], v0=v0, limit=lim, kin_crawl_over_best=pc / pb, stored_prog_crawl_over_best=ec / eb)
out['Q4_per_state'] = q4
out['Q4_launch_from_rest_ratio_vs_limit'] = {str(L): kin_progress(0.0, 1.5, L) / max(kin_progress(0.0, 0.9 * L, L), kin_progress(0.0, L, L))
                                             for L in (4.47, 8.0, 11.18, 13.41, 15.0, 17.0, 20.0, 30.0)}

# ---------------- Q1b: if q_free were applicable on every no-lead/no-conflict cycle (vstar := lane limit, an approximation: true vstar can only be lower)
def w_q1b(d, rk, i, c):
    if free_geom(c) and not c['active']['q_free'] and not c['hard']:
        L = d['limit']; vbar = c['progress_legacy'] * L          # ds/0.5 s
        return _clip(1.0 - 0.7 * (0.9 * L - vbar) / (0.4 * L), 0.20, 1.0)
    return 1.0
nk = {}; per = {}
for i in ELIG:
    d = S[i]; r0 = rets_of(d); r1 = rets_of(d, w_q1b); h = hards_of(d); nk[i] = known_from(r1, h); b = max(r1['f0.9'], r1['f1.0'])
    per[i] = dict(crawl_over_best=r1['crawl1.5'] / b, f0_7=r1['f0.7'] / b, f1_0_retention=r1['f1.0'] / r0['f1.0'], f0_9_retention=r1['f0.9'] / r0['f0.9'])
out['Q1b_q_free_applicable_everywhere_vstar_eq_limit'] = dict(
    fixed_of_10=[i for i in FAIL if nk[i]], regressions_of_29=[i for i in OK if not nk[i]], total_known_after=sum(nk.values()),
    state_108=per[108], state_20=per[20], changed_states=[i for i in ELIG if abs(per[i]['f1_0_retention'] - 1) > 1e-12 or abs(per[i]['crawl_over_best'] - 1) < 0])
json.dump(out, open(os.path.join(HERE, 'reward2_counterfactuals.json'), 'w'), indent=1, default=float)
print('done')
