"""MM 2026-11-22: where does the scorer's pick lose PDMS against the best of its own 64 proposals? (F and C fans, navtest, tracked = 'tr').
MEASURED from E:/Projects/TanitAD-artifacts/field_read/propscore/full_{F,C}_s{0,1}of2.jsonl. best_of_64 is a label-side bound (T0-class),
never a driving result; the pick is the T1 tracked score. Control: the stored per-proposal pdms must be reproduced from its subscores.
Attribution (exact, per scene, sums to the gap): gap = best - pick. If the pick's multiplier (nc*dac[*ddc]) is below the best's, the whole
gap is a GATE loss, split nc / dac / ddc by which term failed. Otherwise the multipliers are equal and gap = mult*sum_k w_k*(best_k-pick_k)/12
over k in (ep, ttc, c), which is split exactly by term (a negative term = the pick is better on that term)."""
import json, sys, glob
import numpy as np
R = 'E:/Projects/TanitAD-artifacts/field_read/propscore/'
W = {'ep': 5.0, 'ttc': 5.0, 'c': 2.0}
out = {}
for arm in ('F', 'C'):
    rows = []
    for f in sorted(glob.glob(R + 'full_%s_s*of2.jsonl' % arm)):
        rows += [json.loads(l) for l in open(f)]
    toks = [r['token'] for r in rows]
    assert len(set(toks)) == len(toks), 'duplicate tokens'
    # control: which formula reproduces the stored pdms?
    errs = {}
    for name, use_ddc in (('v1', False), ('v1+ddc', True)):
        e = 0.0
        for r in rows:
            t = {k: np.asarray(v, float) for k, v in r['tr'].items()}
            mult = t['nc'] * t['dac'] * (t['ddc'] if use_ddc else 1.0)
            p = mult * (W['ep'] * t['ep'] + W['ttc'] * t['ttc'] + W['c'] * t['c']) / 12.0
            e = max(e, float(np.abs(p - t['pdms']).max()))
        errs[name] = e
    form = min(errs, key=errs.get)
    assert errs[form] < 1e-6, ('no formula reproduces pdms', errs)
    use_ddc = form == 'v1+ddc'
    n = len(rows)
    acc = {k: 0.0 for k in ('gate_nc', 'gate_dac', 'gate_ddc', 'ep', 'ttc', 'c')}
    cnt = {'scenes': n, 'gap_pos': 0, 'gate_scenes': 0, 'weighted_scenes': 0, 'pick_is_best': 0}
    pick_s, best_s, rand_s = [], [], []
    ep_short = []  # weighted-loss scenes: pick's ep deficit
    for r in rows:
        t = {k: np.asarray(v, float) for k, v in r['tr'].items()}
        mult = t['nc'] * t['dac'] * (t['ddc'] if use_ddc else 1.0)
        pd = t['pdms']; i = r['pick']; b = int(np.argmax(pd))
        pick_s.append(pd[i]); best_s.append(pd[b]); rand_s.append(pd.mean())
        gap = pd[b] - pd[i]
        if gap <= 1e-12:
            cnt['pick_is_best'] += 1; continue
        cnt['gap_pos'] += 1
        if mult[i] < mult[b] - 1e-12:
            cnt['gate_scenes'] += 1
            fails = [k for k in ('nc', 'dac', 'ddc') if (k != 'ddc' or use_ddc) and t[k][i] < t[k][b] - 1e-12]
            for k in fails:
                acc['gate_' + k] += gap / len(fails)
        else:
            cnt['weighted_scenes'] += 1
            parts = {k: mult[i] * W[k] * (t[k][b] - t[k][i]) / 12.0 for k in W}
            assert abs(sum(parts.values()) - gap) < 1e-9
            for k, v in parts.items():
                acc[k] += v
            ep_short.append(t['ep'][b] - t['ep'][i])
    gap_tot = (np.sum(best_s) - np.sum(pick_s)) / n * 100
    attr = {k: v / n * 100 for k, v in acc.items()}
    assert abs(sum(attr.values()) - gap_tot) < 1e-6, (sum(attr.values()), gap_tot)
    out[arm] = {'formula': form, 'formula_max_abs_err': errs, 'counts': cnt,
                'pick_pdms_x100': float(np.mean(pick_s) * 100), 'best_of_64_x100': float(np.mean(best_s) * 100),
                'random_mean_x100': float(np.mean(rand_s) * 100), 'gap_points': float(gap_tot),
                'gap_points_by_term': attr,
                'gap_share_by_term': {k: v / gap_tot for k, v in attr.items()},
                'weighted_scenes_median_ep_deficit': float(np.median(ep_short)) if ep_short else None}
json.dump(out, open(sys.argv[1], 'w'), indent=1)
for arm, o in out.items():
    print(arm, 'formula', o['formula'], 'n', o['counts']['scenes'], 'pick %.2f best %.2f gap %.2f' % (o['pick_pdms_x100'], o['best_of_64_x100'], o['gap_points']))
    print('  counts', o['counts'])
    print('  by term', {k: round(v, 3) for k, v in o['gap_points_by_term'].items()})
