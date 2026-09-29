"""Red-team check (2026-09-29): under PERFECT exchangeability (i.i.d. continuous scores), what share of
split-conformal calibration draws gives test coverage below nominal?  Theory: coverage | calibration ~
Beta(k, n+1-k), k = ceil((1-alpha)(n+1)); simulated here with numpy only (no scipy on this box).
Evidence class: MEASURED (this script, seed 0, 20,000 draws per cell)."""
import json, math, numpy as np
rng = np.random.default_rng(0); out = []
for alpha in (0.05, 0.10, 0.20):
    for n in (100, 300, 1000, 10000):
        k = math.ceil((1 - alpha) * (n + 1))
        cov = rng.beta(k, n + 1 - k, size=20000)          # exact law of the conditional coverage
        out.append({"alpha": alpha, "n_calib": n, "mean_coverage": round(float(cov.mean()), 4),
                    "sd": round(float(cov.std()), 4),
                    "share_below_nominal": round(float((cov < 1 - alpha).mean()), 3)})
print(json.dumps(out, indent=1))
