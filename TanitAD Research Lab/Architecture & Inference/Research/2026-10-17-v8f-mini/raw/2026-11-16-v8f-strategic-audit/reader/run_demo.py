"""End-to-end STRATEGIC demonstration on the banked ACTSENS windows (CPU, read-only inputs). T0 HWM-ORACLE plans: a demonstration of the instrument, not a decision.
Pipeline = the production path: hwm_eval.metrics_of (attaches the nav-text row), S_ADH / S_ADH_DEG / S_FALSE on the plan and on the logged path, hwm_eval.contrast (taniteval/ci.py paired episode-cluster bootstrap, B = 2000, seed 0)."""
import sys, json, types, os
sys.path.insert(0, 'F:/TanitAD-scratch/strategic/build/src')
import numpy as np
import hwm_eval as E
import hwm_plan as H
import strategic_cells as S
import strategic_text as ST

R = 'F:/TanitAD-scratch/actsens/read/out'
OUT = 'F:/TanitAD-scratch/strategic/build/out'
B = 2000
CHASSIS = {'C7_R35_s0': 0, 'IDM_s0': 0, 'DIV_s0': 0, 'NULL_s2': 0, 'IDM_s1': 1, 'DIV_s1': 1, 'NULL_s3': 1}
NULLREF = {0: 'NULL_s2', 1: 'NULL_s3'}
PLANNERS = ('hwm_oracle', 'flat_oracle')

tab = ST.load_table('F:/TanitAD-scratch/datascale/navtok/navtok_table.npz')
W = {a: np.load(f'{R}/{a}/win/windows.npz', allow_pickle=True) for a in CHASSIS}
ref = W['NULL_s3']
for a, z in W.items():                                            # the windows are the same 994 for every arm (C6 control), asserted here
    assert np.array_equal(z['eid'], ref['eid']) and np.array_equal(z['now'], ref['now']) and np.array_equal(z['A'], ref['A']) and np.array_equal(z['v0'], ref['v0'], equal_nan=True), a
eid = ref['eid'].astype(str)
ws = types.SimpleNamespace(eid=eid, now=ref['now'].astype(int), A=ref['A'].astype(float), v0=ref['v0'].astype(float), n=len(eid), nav_table=tab)
plans = {a: {p: np.load(f'{R}/{a}/win/plans_{p}_s0.npz')['plan_raw'].astype(float) for p in PLANNERS} for a in CHASSIS}


def stat(x):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    return dict(n_windows=int(ok.sum()), n_episodes=int(len(set(eid[ok]))), rate=float(x[ok].mean()) if ok.any() else None)


def cstat(a, b, direction):
    r = E.contrast(a, b, eid, B, direction)
    return {k: (float(r[k]) if isinstance(r.get(k), (float, np.floating)) else r.get(k)) for k in ('delta', 'lo', 'hi', 'label', 'n_windows', 'n_episodes', 'reason') if k in r}


def metrics(arm_plans, text=None):
    if text is not None:
        ws.text = text
    return E.metrics_of(arm_plans, ws)


# primary read (production path: metrics_of attaches the text row at ST.NOW_OFFSET)
ws.text = None
M0 = E.metrics_of({'hold': H.hold_plan(ws.A[:, 0], 30), 'straight': np.zeros((ws.n, 30, 2)), 'hwm_oracle': plans['NULL_s3']['hwm_oracle']}, ws)         # also attaches ws.text
text0 = ws.text
M = {a: E.metrics_of({p: plans[a][p] for p in PLANNERS}, ws) for a in CHASSIS}
F = 'full6s'
demo = {'tier': 'T0 HWM-ORACLE diagnostic: the goal that conditioned each plan is the encoded true future (never a driving number); demonstration of the instrument, not a decision',
        'evidence': 'MEASURED on the banked ACTSENS read (F:/TanitAD-scratch/actsens/read/out/<arm>/win), navtok_table.npz sha256 prefix 2836369e, tip hwm_eval build in strategic/build/src',
        'hypotheses': {'evidence': 'HYPOTHESIS', 'COMPLETE_M': S.COMPLETE_M, 'theta_deg': S.THETA_DEG, 'false_margin_m': S.FALSE_MARGIN_M, 'note': 'pre-register before any decisive read'},
        'text': {'now_offset': ST.NOW_OFFSET, 'n_windows': ws.n, 'n_episodes': int(len(set(eid))), 'n_text_ok': int(text0['text_ok'].sum()), 'flags': {str(int(k)): int((text0['flag'] == k).sum()) for k in np.unique(text0['flag'])}},
        'estimator': {'name': 'paired episode-cluster bootstrap (taniteval/ci.py), B = 2000, seed 0',
                      'answers': 'would another draw of EPISODES say this? (models held fixed). It does NOT answer: would another TRAINING run say this (H-ESTIM-SEED-1: a lever claim needs a replicate arm), and would another INFERENCE seed say this (the oracle planner samples; only inference seed 0 is read here)'},
        'control_straight': {'s_adh': stat(M0['straight'][F]['s_adh']), 's_adh_exact_zero': bool(np.all(M0['straight'][F]['s_adh'][np.isfinite(M0['straight'][F]['s_adh'])] == 0.0)), 's_adh_deg_max_abs': float(np.nanmax(np.abs(M0['straight'][F]['s_adh_deg'])))},
        'hold_baseline': {'s_adh': stat(M0['hold'][F]['s_adh']), 'note': 'HOLD repeats the NOW action INCLUDING the yaw rate, so on windows where the ego is already turning it is a turning plan: it is a baseline, not a zero control. The exact-0.0 control is the straight plan'},
        'logged_path': {'s_adh_ceiling': stat(M0['hold'][F]['_s_adh_l']), 's_false_logged_rate': stat(M0['hold'][F]['_s_false_l'])},
        'arms': {}}
for a, k in CHASSIS.items():
    d = {'chassis': k, 'null_ref': NULLREF[k], 'planners': {}}
    for p in PLANNERS:
        m = M[a][p][F]
        n = M[NULLREF[k]][p][F]
        e = {'s_adh': stat(m['s_adh']), 's_adh_deg': stat(m['s_adh_deg']), 's_false': stat(m['s_false']),
             'vs_logged': {'s_adh': cstat(m['s_adh'], m['_s_adh_l'], 'higher'), 's_adh_deg': cstat(m['s_adh_deg'], m['_s_adh_deg_l'], 'higher'), 's_false': cstat(m['s_false'], m['_s_false_l'], 'neutral')}}
        e['s_adh_deg']['mean_deg'] = float(np.nanmean(m['s_adh_deg'])) if np.isfinite(m['s_adh_deg']).any() else None
        if a != NULLREF[k]:
            e['vs_null'] = {'s_adh': cstat(m['s_adh'], n['s_adh'], 'higher'), 's_adh_deg': cstat(m['s_adh_deg'], n['s_adh_deg'], 'higher'), 's_false': cstat(m['s_false'], n['s_false'], 'neutral')}
        d['planners'][p] = e
    demo['arms'][a] = d

# sensitivity: COMPLETE_M 10 / 20 / 30 (HYPOTHESIS), and the audit's unverified table step now + 3 as a labelled extra
sens = {}
cm0 = S.COMPLETE_M
for off in (ST.NOW_OFFSET, 3):
    t = ST.attach(tab, eid, ws.now, offset=off)
    for cm in (10.0, 20.0, 30.0):
        S.COMPLETE_M = cm
        key = f'COMPLETE_M={cm:g}, table_step now+{off}' + (' (audit guess, not verified)' if off == 3 else ' (verified)')
        row = {}
        ws.text = t
        for a in CHASSIS:
            m = E.metrics_of({'hwm_oracle': plans[a]['hwm_oracle']}, ws)['hwm_oracle'][F]
            row[a] = {'hwm_oracle_s_adh': stat(m['s_adh']), 'logged_ceiling': stat(m['_s_adh_l'])}
        sens[key] = row
S.COMPLETE_M = cm0
demo['sensitivity_COMPLETE_M'] = sens
json.dump(demo, open(f'{OUT}/STRATEGIC_ACTSENS_DEMO.json', 'w'), indent=1, default=float)

# one full production report (build_report) for NULL_s3 as the end-to-end proof of the row structure
ws.text = text0
Mr = {'hold': M0['hold'], 'hwm_oracle': M['NULL_s3']['hwm_oracle'], 'flat_oracle': M['NULL_s3']['flat_oracle']}
rep = E.build_report(Mr, ws, n_boot=B)
md_rep = E.to_markdown(rep)
sub = dict(strategic=rep['strategic'], rows=[r for r in rep['rows'] if r['family'] == 'STRATEGIC'], not_computed={k: v for k, v in rep['not_computed'].items() if k.startswith('STRATEGIC.')})
json.dump(sub, open(f'{OUT}/STRATEGIC_ACTSENS_REPORT_NULL_s3_rows.json', 'w'), indent=1, default=float)
assert 'PAUSED' not in md_rep

# markdown
f = lambda s_: 'n/a' if s_['rate'] is None else f"{s_['rate']:.3f} (n {s_['n_windows']}, ep {s_['n_episodes']})"
cf = lambda c: 'n/a (%s)' % c.get('reason', 'n = 0') if c.get('n_windows', 0) == 0 else f"{c['delta']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] n {c['n_windows']}, ep {c['n_episodes']}, {c['label']}"
L = ['# STRATEGIC family on the banked ACTSENS windows: a demonstration of the instrument', '',
     f"Tier: {demo['tier']}.", f"Evidence: {demo['evidence']}.", '',
     f"Windows {demo['text']['n_windows']}, episodes {demo['text']['n_episodes']}; nav-text rows attached at table step now + {ST.NOW_OFFSET} (MEASURED, STRATEGIC_TEXT_OFFSET_CHECK.json): text_ok {demo['text']['n_text_ok']}, flags {demo['text']['flags']}.",
     f"COMPLETE_M {S.COMPLETE_M:g} m, theta 25 deg (SLIGHT 10 deg), false-turn margin 30 m: all HYPOTHESIS, to be pre-registered before any decisive read.", '',
     f"Estimator: {demo['estimator']['name']}. It answers: {demo['estimator']['answers']}.", '',
     f"Control, straight plan (zero action): S_ADH {f(demo['control_straight']['s_adh'])}, reads exactly 0.0 on every scorable window: {demo['control_straight']['s_adh_exact_zero']}; max |signed degrees| {demo['control_straight']['s_adh_deg_max_abs']:.3f}.",
     f"HOLD baseline (the NOW action repeated, yaw rate included, so it turns where the ego is turning): S_ADH {f(demo['hold_baseline']['s_adh'])}. It is a baseline, not a zero control.",
     f"Logged path through the same cells: S_ADH ceiling {f(demo['logged_path']['s_adh_ceiling'])}; S_FALSE logged-path rate {f(demo['logged_path']['s_false_logged_rate'])}. S_FALSE is read only as plan minus logged, never as an absolute.", '',
     '## Per arm (inference seed 0, HWM-ORACLE and FLAT-ORACLE plans)', '',
     '| arm | planner | S_ADH rate | S_ADH minus logged | S_ADH minus NULL | S_ADH_DEG mean deg | S_FALSE rate | S_FALSE minus logged | S_FALSE minus NULL |', '|---|---|---|---|---|---|---|---|---|']
for a, d in demo['arms'].items():
    for p, e in d['planners'].items():
        vn = e.get('vs_null')
        md_deg = e['s_adh_deg']['mean_deg']
        L.append(f"| {a} | {p} | {f(e['s_adh'])} | {cf(e['vs_logged']['s_adh'])} | {cf(vn['s_adh']) if vn else 'reference'} | {'n/a' if md_deg is None else f'{md_deg:+.1f}'} | {f(e['s_false'])} | {cf(e['vs_logged']['s_false'])} | {cf(vn['s_false']) if vn else 'reference'} |")
L += ['', 'NULL is the chassis-matched NULL arm (NULL_s2 for chassis 0, NULL_s3 for chassis 1). A "none" label means the interval contains 0; a separated interval here is necessary and not sufficient for any claim about a lever (H-ESTIM-SEED-1).', '',
      '## Sensitivity of the S_ADH cell to COMPLETE_M and to the table step (HWM-ORACLE plan; each cell n windows / episodes)', '',
      '| setting | ' + ' | '.join(CHASSIS) + ' |', '|---|' + '---|' * len(CHASSIS)]
for key, row in sens.items():
    L.append(f'| {key} | ' + ' | '.join(f"{row[a]['hwm_oracle_s_adh']['rate'] if row[a]['hwm_oracle_s_adh']['rate'] is not None else float('nan'):.3f} ({row[a]['hwm_oracle_s_adh']['n_windows']}/{row[a]['hwm_oracle_s_adh']['n_episodes']}; ceiling {row[a]['logged_ceiling']['rate'] if row[a]['logged_ceiling']['rate'] is not None else float('nan'):.2f})" for a in CHASSIS) + ' |')
L += ['', 'COMPLETE_M 10, 20 and 30 are HYPOTHESIS values; the rows show how far the cell moves with the allowance and with the table step (the now + 3 row is the audit\'s unverified guess, kept to show its effect).', '',
      '## Not computed (n/a entries carried by the report)', '']
for k, v in rep['not_computed'].items():
    if k.startswith('STRATEGIC.'):
        L.append(f'* {k}: {v}')
open(f'{OUT}/STRATEGIC_ACTSENS_DEMO.md', 'w', encoding='utf-8').write('\n'.join(L) + '\n')
assert demo['control_straight']['s_adh_exact_zero'] and demo['control_straight']['s_adh']['n_windows'] > 0
print(json.dumps({'text': demo['text'], 'control_straight': demo['control_straight'], 'hold_baseline': demo['hold_baseline'], 'logged': demo['logged_path']}, indent=1))
for a, d in demo['arms'].items():
    e = d['planners']['hwm_oracle']
    print(a, 's_adh', e['s_adh'], 'vs_logged', e['vs_logged']['s_adh'], 'vs_null', e.get('vs_null', {}).get('s_adh'))
