"""Agent recompute of P7's MARGINAL PDMS interval (the read itself wrote only paired intervals).
Estimator: log-cluster bootstrap over 136 navtest logs, 10,000 resamples, 95 % percentile,
numpy default_rng(20260927). NOT the pinned proxy_eval.boot resample stream: point equals the harness
91.0724; the interval may differ from a pinned-stream value in the 2nd decimal. Run: python marginal_p7_recompute.py
Inputs: score/refe_stage2_p7_full.csv (this folder) and navtest_full_tokens.json (E:/Projects/TanitAD/data/refe_navtest/)."""
import csv, json, sys, numpy as np
here = sys.argv[1] if len(sys.argv) > 1 else '.'
tok = sys.argv[2] if len(sys.argv) > 2 else 'E:/Projects/TanitAD/data/refe_navtest/navtest_full_tokens.json'
r = [x for x in csv.DictReader(open(f'{here}/score/refe_stage2_p7_full.csv')) if x['token'] != 'average']
tl = json.load(open(tok))['token_log']
logs = sorted({tl[x['token']] for x in r}); li = {l: i for i, l in enumerate(logs)}
g = np.array([li[tl[x['token']]] for x in r]); s = np.array([float(x['score']) for x in r]) * 100
sums = np.bincount(g, s); cnt = np.bincount(g); rng = np.random.default_rng(20260927); L = len(logs)
b = [sums[i].sum() / cnt[i].sum() for i in (rng.integers(0, L, L) for _ in range(10000))]
print(json.dumps({'n': len(r), 'n_logs': L, 'pdms_x100': float(s.mean()), 'ci95': [float(v) for v in np.percentile(b, [2.5, 97.5])]}))
