"""Measures the largest difference between dist_rule.distance_rule and the d8 mutant (unchanged steps not copied back) on random plans, float32 and float64 plans."""
import sys, types
import numpy as np
src = open('src/dist_rule.py', encoding='utf-8').read()
assert src.count('pts[keep] = p64[b][keep]') == 1
def mk(text):
    m = types.ModuleType('m'); exec(compile(text, 'm', 'exec'), m.__dict__); return m
A, Bm = mk(src), mk(src.replace('pts[keep] = p64[b][keep]', 'pts[keep] = pts[keep]'))
rng = np.random.default_rng(0)
worst = {'float32': 0.0, 'float64': 0.0}; nkeep = 0; nchg = {'float32': 0, 'float64': 0}; nrule = {'float32': 0, 'float64': 0}; trials = 0
for dt in ('float32', 'float64'):
    for _ in range(300):
        n = 8
        v = rng.uniform(2, 15, n)
        ang = np.cumsum(rng.normal(0, 0.03, (n, 30)), 1)
        step = (v[:, None] * 0.2)
        plan = np.cumsum(np.stack([step * np.cos(ang), step * np.sin(ang)], -1), 1).astype(dt)
        g = rng.uniform(5, 50, n)
        vL = rng.uniform(0, 8, n)
        args = (plan, v, np.ones(n, bool), g, g, vL)
        a, b = A.distance_rule(*args), Bm.distance_rule(*args)
        d = np.abs(a.plan.astype(np.float64) - b.plan.astype(np.float64)).max()
        worst[dt] = max(worst[dt], float(d)); nchg[dt] += int((a.plan != b.plan).any(axis=(1, 2)).sum()); trials += n; nrule[dt] += int(a.changed.sum())
print('worst abs diff per plan dtype', worst, 'windows differing', nchg, 'windows per dtype', 2400, 'windows changed by the rule', nrule)
