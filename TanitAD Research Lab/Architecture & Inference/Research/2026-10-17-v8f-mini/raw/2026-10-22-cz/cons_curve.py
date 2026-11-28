"""T's constraint loss over M1/M2b training (2k-step bins), and the M2b argv; a zero-GPU diagnostic for the next lever."""
import json, sys
R = 'C:/Users/Admin/box2_ship/logs/v8f_mini/runs/'
for t in ('M1_s0', 'M1_s1', 'M2b_s0', 'M2b_s1'):
    rows = [json.loads(l) for l in open(R + t + '/train_log.jsonl', encoding='utf-8') if l.strip()]
    tr = [r for r in rows if 'cons' in r and 'step' in r]
    out = []
    for a in range(0, 20000, 2000):
        v = [r['cons'] for r in tr if a <= r['step'] < a + 2000]
        out.append(sum(v) / len(v) if v else float('nan'))
    pg = [r['p_gt'] for r in tr if 'p_gt' in r]
    print(t, 'n', len(tr), 'cons per 2k bin:', ' '.join(f'{x:.3f}' for x in out), '| p_gt end', f'{pg[-1]:.2f}')
    other = [r for r in rows if not ('cons' in r and 'step' in r)]
    print('  non-step rows', len(other), [sorted(r.keys())[:12] for r in other[-2:]])
a = json.load(open(R + 'M2b_s0/argv.json', encoding='utf-8')); print('argv', a)
s = json.load(open(R + 'M2b_s0/selfcheck.json', encoding='utf-8')); print('selfcheck keys', list(s.keys())[:30])
