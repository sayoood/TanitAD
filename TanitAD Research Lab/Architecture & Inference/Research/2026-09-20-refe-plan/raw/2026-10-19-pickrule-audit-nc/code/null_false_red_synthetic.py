"""SYNTHETIC false-fire rate of the interval rule `Q1 lower CI > 0.5` (prereg s8 C-NULL, RETRACTION_LOG R51).  No real data.  Run before sealing; output in raw/logs/null_false_red_synthetic.log."""
import sys, numpy as np, importlib.util
sys.path.insert(0, "C:/Users/Admin/par18/pickrule/ref/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-fastneg-census/code")
import fastneg_lib as L
draws = L.boot_draws(117, 10000, 20260927)
rng = np.random.default_rng(12345)
out = {}
for nA in (60, 100, 150):
    for pi in (0.10, 0.30, 0.50):
        fireL = fireU = 0
        R = 1000
        for _ in range(R):
            logs = rng.integers(0, 117, size=nA)
            den = np.bincount(logs, minlength=117).astype(float)
            num = np.array([rng.binomial(int(d), pi) for d in den], float)
            ci = L.ratio_ci(num, den, draws)
            fireL += ci["lo"] > 0.5
            fireU += ci["hi"] < 0.5
        out[(nA, pi)] = (fireL / R, fireU / R)
        print(nA, pi, "P(lo>0.5)=%.3f  P(hi<0.5)=%.3f" % out[(nA, pi)], flush=True)
