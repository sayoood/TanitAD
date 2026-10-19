"""NC-FASTCAL POST-HOC, DESCRIPTIVE diagnostic (written AFTER result.json existed; no authority; no threshold or bin is chosen from its output).

Question it answers: at the SELECTED picks, is the NC row overconfident (winner's curse) or is the NC-unsafe excess the honest residual risk the row already states?  For each arm's picks:
the predicted unsafe mass  sum(1 - sigmoid(NC logit))  against the observed unsafe mass  sum(1 - NC label)  (labels 0 / 0.5 / 1 read as probabilities of safe), overall with a paired log-cluster
bootstrap interval on their ratio (same 10,000 draws), and by fixed bins of the raw NC sigmoid ([0, .5), [.5, .9), [.9, .97), [.97, .99), [.99, 1]; fixed here before the run).
Reads the same arrays and the calibrated column that `ncfastcal_run.py analyze` wrote.  CPU only.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ncfastcal_lib as L                                                                 # noqa: E402
import ncfastcal_run as R                                                                 # noqa: E402

BINS = (0.0, 0.5, 0.9, 0.97, 0.99, 1.0000001)


def main() -> int:
    L.require_prereg()
    P, F = L.load_pickrule(), L.load_fastneg()
    z = np.load(R.ARRAYS, allow_pickle=False)
    z6, lab, log_idx = z["z6"], z["lab"], z["log_idx"].astype(np.int64)
    res = np.load(f"{R.RUN}/results/nc_cal_oof.npz")
    nc_cal = res["nc_cal_primary"].astype(np.float64)
    S = P.Stats(log_idx, F.boot_draws(L.N_LOGS, L.BOOT_B, L.BOOT_SEED), F)
    pk_p = P.rule_pick(z6)
    nc_lab = lab[..., L.NC].astype(np.float64)
    arms = {"P2": (pk_p, False)}
    for sd in L.SEEDS:
        ep = z[f"ep_OOF_s{sd}"]
        arms[f"U_s{sd}"] = (P.rule_pick(L.composite(z6, ep=ep)), False)
        arms[f"C_s{sd}"] = (P.rule_pick(L.composite(z6, ep=ep, nc=nc_cal)), True)
    out = {"note": "POST-HOC, DESCRIPTIVE, no authority; bins fixed before the run", "bins": list(BINS[:-1]) + [1.0], "arms": {}}
    for name, (pk, cal) in arms.items():
        sig = P.at(P.sigmoid64(nc_cal if cal else z6[..., L.NC]), pk)
        sig_raw = P.at(P.sigmoid64(z6[..., L.NC]), pk)
        obs = 1.0 - P.at(nc_lab, pk)
        pred = 1.0 - sig
        row = {"n": int(len(pk)), "predicted_unsafe_mass": float(pred.sum()), "observed_unsafe_mass": float(obs.sum()), "ratio_observed_over_predicted": S.ratio(obs, pred),
               "nc_unsafe_picks_label0": int((P.at(nc_lab, pk) == 0.0).sum()), "bins": {}}
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            m = (sig_raw >= lo) & (sig_raw < hi)
            row["bins"][f"[{lo:g},{min(hi, 1.0):g})"] = {"n": int(m.sum()), "mean_raw_sigmoid": float(sig_raw[m].mean()) if m.any() else None, "observed_unsafe_mass": float(obs[m].sum()),
                                                        "predicted_unsafe_mass_raw": float((1.0 - sig_raw[m]).sum()), "nc_unsafe_picks_label0": int((P.at(nc_lab, pk)[m] == 0.0).sum())}
        out["arms"][name] = row
    p = f"{R.RUN}/results/posthoc_diag.json"
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(P.jsonable(out), indent=1, sort_keys=True))
    for name, row in out["arms"].items():
        r = row["ratio_observed_over_predicted"]
        print(f"{name}: n {row['n']}, label-0 picks {row['nc_unsafe_picks_label0']}, predicted mass {row['predicted_unsafe_mass']:.1f}, observed mass {row['observed_unsafe_mass']:.1f}, ratio obs/pred {r['point']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]")
        for b, v in row["bins"].items():
            print(f"    {b}: n {v['n']}, observed mass {v['observed_unsafe_mass']:.1f}, predicted(raw) {v['predicted_unsafe_mass_raw']:.1f}, label-0 picks {v['nc_unsafe_picks_label0']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
