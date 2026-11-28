"""POST-HOC descriptive diagnostics on the stored replicates of the C1-geometry simulation (written AFTER the bars were read; they change no verdict and no bar).
    python posthoc_diag.py --workdir W --out posthoc_diagnostics.json"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [c - h, c + h]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    z = np.load(os.path.join(a.workdir, "replicates.npz"))
    a2, dm = z["CAL__a2"], z["CAL__dm"]
    n = len(a2)
    out = {"label": "POST-HOC, descriptive; no bar, rule or verdict is changed by anything here", "n_cal": n}
    out["NC_ratio_undefined"] = {"seed_evaluations_nan": int(np.isnan(a2[:, :, 0, 1]).sum()), "of": int(a2.shape[0] * a2.shape[1]), "replicates_with_any_nan": int(np.isnan(a2[:, :, 0, 1]).any(axis=1).sum()),
                                 "share_of_replicates": float(np.isnan(a2[:, :, 0, 1]).any(axis=1).mean()), "meaning": "no NC-unsafe pick among the drawn tokens: observed / predicted is 0 / 0; the frozen module fails closed (a refusal)"}
    for t in (1.00, 1.05, 1.10):
        k = int((a2[:, :, :, 1] > t).any(axis=(1, 2)).sum())
        out[f"A2_false_refusal_NaN_NOT_counted_margin_{t:.2f}"] = {"k": k, "n": n, "rate": k / n, "wilson95": wilson(k, n), "per_head_k": [int((a2[:, :, j, 1] > t).any(axis=1).sum()) for j in range(3)]}
    for rho in ("1.25", "1.5"):
        d = z[f"PLANT_A2_DAC_{rho}__a2"][:, :, 1, :]
        out[f"DAC_power_x{rho}"] = {f"margin_{t:.2f}": float((d[:, :, 1] > t).any(axis=1).mean()) for t in (1.00, 1.05, 1.10)}
    out["A1_false_refusal_by_conjunct"] = {"lower_ci_gt_0_any_seed": int((dm[:, :, 1] > 0).any(axis=1).sum()), "point_gt_0.05_any_seed": int((dm[:, :, 0] > 0.05).any(axis=1).sum()), "n": n}
    for d in ("0.135", "0.05"):
        x = z[f"PLANT_EP_{d}__dm"]
        out[f"A1_power_plant_{d}"] = float(((x[:, :, 1] > 0) | (x[:, :, 0] > 0.05)).any(axis=1).mean())
    rng_distinct = [len(set(np.random.default_rng([20261202, r]).integers(0, 24, 24).tolist())) for r in range(n)]
    out["generator_A_distinct_logs_per_outer_draw"] = {"mean": float(np.mean(rng_distinct)), "of": 24, "note": "same 63 % as R1's 74 / 117; the inner bootstrap treats duplicates as independent clusters"}
    out["CAL_D_M"] = {"point_sd_by_seed": dm[:, :, 0].std(axis=0, ddof=1).tolist(), "mean_half_width_by_seed": ((dm[:, :, 2] - dm[:, :, 1]) / 2).mean(axis=0).tolist()}
    json.dump(out, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
