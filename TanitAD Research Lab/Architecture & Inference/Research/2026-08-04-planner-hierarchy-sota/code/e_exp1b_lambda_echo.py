"""E-EXP-1b (⚠️ POST-HOC / EXPLORATORY — NOT pre-registered, decides nothing).

E-EXP-1 measured a CEILING for a per-window longitudinal scale lambda on the selected
trajectory. This asks the obvious next question cheaply: how much of that ceiling is a
`v0` ECHO? Same class of test as the nav-echo defect (flagship's route head is an exact
bijection of the nav it is fed) and the sitclf leak.

Arms, all fitted LEAVE-ONE-EPISODE-OUT so every number is out-of-episode:
  MAJORITY  - predict the modal lambda* of the 39 training episodes (no features at all)
  V0-ECHO   - predict lambda* from a v0 decile lookup (ego speed ONLY)
  GEOM      - v0 decile x along-distance decile of the SELECTED trajectory
Scored as REALIZED ADE after applying the predicted lambda, vs the shipped ade_sel.
"""
import sys, os, json
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "taniteval"))
from taniteval.ci import paired_episode_cluster_bootstrap  # noqa: E402

LAMBDAS = np.array([0.92, 0.96, 1.00, 1.04, 1.08])


def ade(t, g):
    return np.linalg.norm(t - g, axis=-1).mean(axis=-1)


def loeo_lookup(key, lstar, err, eids):
    """key (W,) int bucket. Fit modal lambda* per bucket on 39 eps, apply to the 40th."""
    W = len(lstar)
    out = np.empty(W, dtype=int)
    for ep in sorted(set(eids)):
        m = eids == ep
        tr = ~m
        # per-bucket best lambda by TOTAL error on the training episodes (not modal
        # class: minimising realised error is the decision we actually care about)
        table = {}
        for b in np.unique(key):
            sub = tr & (key == b)
            table[b] = int(err[:, sub].sum(axis=1).argmin()) if sub.any() else \
                int(err[:, tr].sum(axis=1).argmin())
        glob = int(err[:, tr].sum(axis=1).argmin())
        out[m] = [table.get(b, glob) for b in key[m]]
    return out


def run(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    fan = d["fan"].double().numpy(); gt = d["gt"].double().numpy()
    sel = d["sel"].numpy(); eids = np.array(list(d["eid"])); v0 = d["v0"].double().numpy()
    W = len(gt)
    fsel = fan[np.arange(W), sel]
    err = np.stack([ade(fsel * L, gt) for L in LAMBDAS])      # (5, W)
    ade_sel = err[LAMBDAS.tolist().index(1.00)]
    lstar = err.argmin(axis=0)

    def dec(x):
        return np.digitize(x, np.quantile(x, np.arange(0.1, 1.0, 0.1)))
    keys = {"MAJORITY": np.zeros(W, dtype=int),
            "V0_ECHO": dec(v0),
            "GEOM": dec(v0) * 10 + dec(fsel[:, -1, 0])}
    B = lambda a, b: paired_episode_cluster_bootstrap(a, b, list(eids), n_boot=2000, seed=0)
    out = {"arm": os.path.basename(path), "n_windows": W,
           "ade_sel": float(ade_sel.mean()),
           "ceiling_oracle_lambda": float(err.min(axis=0).mean()),
           "lambda_star_hist": {str(L): int((lstar == i).sum())
                                for i, L in enumerate(LAMBDAS)}}
    for name, k in keys.items():
        pred = loeo_lookup(k, lstar, err, eids)
        realized = err[pred, np.arange(W)]
        out[name] = {"ade": float(realized.mean()),
                     "vs_sel": B(ade_sel, realized),
                     "acc_vs_lambda_star": float((pred == lstar).mean())}
    return out


if __name__ == "__main__":
    res = [run(p) for p in sys.argv[1:-1]]
    json.dump(res, open(sys.argv[-1], "w"), indent=2)
    for r in res:
        print(f"\n=== {r['arm']}  sel {r['ade_sel']:.4f}  oracle-lambda ceiling "
              f"{r['ceiling_oracle_lambda']:.4f}")
        print("  lambda* histogram:", r["lambda_star_hist"])
        for a in ("MAJORITY", "V0_ECHO", "GEOM"):
            v = r[a]["vs_sel"]
            print(f"  {a:9s} ade {r[a]['ade']:.4f}  recovered {v['delta']:+.4f} "
                  f"[{v['lo']:+.4f},{v['hi']:+.4f}] sep={v['separated']}  "
                  f"acc={r[a]['acc_vs_lambda_star']:.3f}")
