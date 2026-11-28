"""Synthetic smoke test of run_latsem2b's code paths (random data, no model, no label of the programme): draw_task, fit_real, eval_*, analyse. Prints shapes and the share of
rows the two-part rule declares separated on pure noise (a sanity check of the machinery, not a LATSEM2B statistic)."""
import sys, os
sys.dont_write_bytecode = True
from types import SimpleNamespace
import numpy as np
import run_latsem2b as R

rng = np.random.default_rng(7)
ne, per = 30, 24
beid = np.repeat(np.array([f"ep{i:02d}" for i in range(ne)]), per); nb = len(beid)
uniq = np.unique(beid); pos = np.searchsorted(uniq, beid)
fold = np.array([i % 2 for i in range(ne)])[pos]
Wt = np.stack([np.bincount(rng.integers(0, ne, ne), minlength=ne) for _ in range(300)]).astype(float)
Y = {}
Y["lead_present"] = (rng.random(nb) < 0.4).astype(float)
gap = np.where(Y["lead_present"] == 1, rng.random(nb) * 40, np.nan); Y["lead_gap"] = gap
Y["lead_closing"] = np.where(np.isfinite(gap), rng.normal(size=nb), np.nan)
Y["agents30"] = rng.integers(0, 9, nb).astype(float); Y["nearest"] = rng.random(nb) * 50
Y["lon7"] = rng.integers(0, 7, nb).astype(float); Y["turn"] = (rng.random(nb) < 0.2).astype(float)
Y["lead_closing_uncapped"] = Y["lead_closing"].copy(); Y["agents30_view"] = rng.integers(0, 5, nb).astype(float)
Y["lon7"][rng.random(nb) < 0.05] = np.nan
speed = np.where(rng.random(nb) < 0.99, rng.random(nb) * 12, np.nan)
val = {q: np.isfinite(Y[q]) for q in R.ALLQ}
bases = {}
for q in R.ALLQ:
    for k in (0, 1):
        fit = val[q] & (fold != k); bases[(q, k)] = float(Y[q][fit].mean()) if R.KIND[q] == "cont" else R.fit_probe_majority(Y[q][fit])
n = 150; widx = rng.choice(nb - 16, n); qe = beid[widx]; qpos = np.searchsorted(uniq, qe); qfold = np.array([i % 2 for i in range(ne)])[qpos]
Yw = {q: np.stack([Y[q][np.minimum(w + np.arange(16), nb - 1)] for w in widx]) for q in R.ALLQ}
m3 = Yw["lead_present"][:, 0] == 1
d = 16
ctx = SimpleNamespace(name="synth", beid=beid, fold=fold, uniq=uniq, Wt=Wt, pos=pos, Y=Y, val=val, bases=bases, speed=speed, qe=qe, qpos=qpos, qfold=qfold, Yw=Yw, m3=m3,
                      nep3=int(len(np.unique(qe[m3]))), BZ={t: {l: rng.normal(size=(nb, d)) for l in (1, 2, 3)} for t in R.TAGS},
                      Z0={t: {l: rng.normal(size=(n, d)) for l in (1, 2, 3)} for t in R.TAGS}, ZQ={t: {(l, k): rng.normal(size=(n, d)) for l, ks in R.KS.items() for k in ks} for t in R.TAGS},
                      CF={t: {(nm, l): rng.normal(size=(n, d)) for nm in ("accel", "brake", "brake_twice") for l in (1, 2, 3)} for t in R.TAGS})
R.set_ctx(ctx)
print("selftest", R.selftest() if hasattr(R, "selftest") else None)
real = {"S1": {}, "S2": {}, "S3": {}}
for tag in R.TAGS:
    P, Pg, Pr, corr = R.fit_real(tag)
    real["S1"].update(R.eval_s1(tag, P)); real["S2"].update(R.eval_s2(tag, P)); real["S3"].update(R.eval_s3(tag, Pg, Pr, with_identical=True))
    print(tag, "corr", corr)
print("real rows", len(real["S1"]), len(real["S2"]), len(real["S3"]))
print("identical S3 point", [v["raw_identical"]["point"] for v in real["S3"].values()], [v["partial_identical"]["point"] for v in real["S3"].values()])
st = [v for k, v in real["S2"].items() if k.endswith("/start")]
print("start rows", len(st))
ref = [R.draw_task((s, "ref")) for s in range(1, 13)]; cal = [R.draw_task((s, "cal")) for s in range(1001, 1006)]
print("draw time", ref[0]["t"], "keys", len(ref[0]["S1"]), len(ref[0]["S2"]), len(ref[0]["S3raw"]), len(ref[0]["S3part"]))
# real rows with S2 decision keys only
real2 = {"S1": real["S1"], "S2": {k: v for k, v in real["S2"].items() if k.endswith("/shrunk") or (not "/L1/" in k and k.endswith("/raw"))}, "S3": real["S3"]}
a = R.analyse(real2, ref, cal)
print({f: (v["rows"], round(v["share_two_part"], 3), round(v["share_bootstrap_only"], 3)) for f, v in a["calibration"].items()}, a["calibration_pass"])
print("distinct", a["distinct_label_assignments"]["min_distinct"], "of", a["distinct_label_assignments"]["draws"])
print("S1 real separated two-part", sum(v["separated_two_part"] for v in a["S1"].values()), "of", len(a["S1"]), "bootstrap only", sum(v["separated_bootstrap_only"] for v in a["S1"].values()))
