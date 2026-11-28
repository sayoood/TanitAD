"""MM 2026-11-22: how much PDMS does LQR tracking cost the pick, vs the same plan scored as drawn?
Reads the banked P7 field proposal tables (navtest, 64 proposals). All quantities are true sub-scores:
'tr' = tracked (LQR, the T1 harness), 'tp' = transparent (plan scored as drawn), 'tp20_pick' = the pick's
transparent row as the harness records it. Descriptive only; no CI here (paired log bootstrap if it is quoted)."""
import json, sys
import numpy as np

W = {'ep': 5.0, 'ttc': 5.0, 'c': 2.0}
ROOT = 'E:/Projects/TanitAD-artifacts/field_read/propscore/'

def pdms(t, i):
    return t['nc'][i] * t['dac'][i] * (W['ep'] * t['ep'][i] + W['ttc'] * t['ttc'][i] + W['c'] * t['c'][i]) / 12.0

out = {}
for arm in ('C', 'F'):
    rows = []
    for s in (0, 1):
        with open(ROOT + 'full_%s_s%dof2.jsonl' % (arm, s)) as fh:
            for line in fh:
                rows.append(json.loads(line))
    n = len(rows)
    ctrl = max(abs(pdms(r['tr'], r['pick']) - r['tr']['pdms'][r['pick']]) for r in rows)
    ctrl_tp20 = max(abs(r['tp']['pdms'][r['pick']] - r['tp20_pick']['pdms']) for r in rows)
    pick_tr = np.array([r['tr']['pdms'][r['pick']] for r in rows])
    pick_tp = np.array([r['tp']['pdms'][r['pick']] for r in rows])
    best_tr = np.array([max(r['tr']['pdms']) for r in rows])
    best_tp = np.array([max(r['tp']['pdms']) for r in rows])
    # the tracked score of the proposal that is best AS DRAWN (what a plan-level oracle would submit)
    best_tp_then_tr = np.array([r['tr']['pdms'][int(np.argmax(r['tp']['pdms']))] for r in rows])
    # per sub-score, pick, tracked minus transparent
    sub = {k: float(np.mean([r['tr'][k][r['pick']] - r['tp'][k][r['pick']] for r in rows])) for k in ('nc', 'dac', 'ep', 'ttc', 'c')}
    worse = int(np.sum(pick_tr < pick_tp - 1e-9)); better = int(np.sum(pick_tr > pick_tp + 1e-9))
    out[arm] = dict(n=n, ctrl_formula_max_err=ctrl, ctrl_tp_vs_tp20_max_err=ctrl_tp20,
                    pick_tracked=100 * pick_tr.mean(), pick_transparent=100 * pick_tp.mean(),
                    pick_track_cost=100 * (pick_tp.mean() - pick_tr.mean()),
                    best_tracked=100 * best_tr.mean(), best_transparent=100 * best_tp.mean(),
                    best_as_drawn_then_tracked=100 * best_tp_then_tr.mean(),
                    pick_sub_tracked_minus_transparent=sub,
                    scenes_tracking_hurts_pick=worse, scenes_tracking_helps_pick=better)
    print(arm, json.dumps(out[arm], indent=1))
json.dump(out, open(sys.argv[1], 'w'), indent=1)
