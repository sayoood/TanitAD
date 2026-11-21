"""P7 (WM_PHYSICS_PROOF): fan uncertainty calibration — spread vs realised error.

Runs on /workspace/x0_fan_dump.npz (OLD v5f fan, 881 windows, full 256 candidates,
f16). Spread measures: (a) selector entropy over softmax(scores); (b) prob-weighted
endpoint dispersion (metres). Realised error: ADE of the SELECTED candidate (sel_idx
in dump). Spearman rho + permutation p (10k). No episode ids in this dump -> NO
episode-cluster CI; flagged, registry-grade rerun binds to the v5.8f eval windows.
CPU-only, ~seconds.
"""
import json

import numpy as np


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))


def main():
    d = np.load("/workspace/x0_fan_dump.npz")
    fan = d["fan"].astype(np.float32)          # [N,256,20,2]
    sc = d["scores"].astype(np.float32)        # [N,256]
    gt = d["gt"].astype(np.float32)            # [N,20,2]
    sel = d["sel_idx"].astype(np.int64)
    N = fan.shape[0]
    p = np.exp(sc - sc.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)             # [N]
    ep = fan[:, :, -1, :]                                        # endpoints [N,256,2]
    mu = (p[..., None] * ep).sum(1, keepdims=True)
    disp = np.sqrt((p * ((ep - mu) ** 2).sum(-1)).sum(1))        # weighted std, m
    err = np.linalg.norm(fan[np.arange(N), sel] - gt, axis=-1).mean(-1)

    rng = np.random.default_rng(0)
    res = {}
    for name, x in (("selector_entropy", ent), ("endpoint_dispersion_m", disp)):
        rho = spearman(x, err)
        perm = np.array([spearman(rng.permutation(x), err) for _ in range(10000)])
        res[name] = {"spearman_rho": round(rho, 4),
                     "perm_p_two_sided": float((np.abs(perm) >= abs(rho)).mean()),
                     "n": int(N)}
    out = {
        "probe": "P7 fan uncertainty calibration (WM_PHYSICS_PROOF)",
        "arm": "v5f-30k ORIGINAL fan (x0_fan_dump)", "tier": "T0-diagnostic",
        "gate": "PRE-REGISTERED: Spearman rho >= 0.3 with interval excluding 0",
        "results": res,
        "spread_stats": {"entropy_mean": float(ent.mean()),
                         "dispersion_m_mean": float(disp.mean()),
                         "err_m_mean": float(err.mean())},
        "_caveat": ("dump carries no episode ids -> permutation p only, NOT the "
                    "episode-cluster CI; registry-grade P7 reruns on the v5.8f "
                    "eval windows which persist eid"),
        "_evidence_class": "MEASURED (ours; artifact = this JSON)"}
    json.dump(out, open("/workspace/p7_calibration.json", "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)
    print("P7_DONE", flush=True)


if __name__ == "__main__":
    main()
