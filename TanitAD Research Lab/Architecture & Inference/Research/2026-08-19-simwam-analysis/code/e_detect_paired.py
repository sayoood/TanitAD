"""E-DETECT-1 — PAIRED deltas against the two floors.

⛔ WHY THIS FILE IS NOT OPTIONAL. `dino_pooled` (AP 0.1416 [0.1275, 0.1550]) and
`prior` (AP 0.1242 [0.1123, 0.1365]) have OVERLAPPING marginal intervals. Reading
that overlap as "no difference" is wrong, and reading the point gap as "a
difference" is equally wrong. Both arms are scored on the SAME 5,617 rows and the
SAME 130 episodes, so the admissible statistic is the PAIRED episode-cluster
bootstrap of the DELTA — resample episodes once per replicate and evaluate BOTH
arms on that same resample.

⚠️ `CLAUDE.md`, verbatim: "for two arms on the same windows use the PAIRED
version, never a combination in quadrature". Two marginal CIs are a combination
in quadrature wearing a chart.

Comparisons run against BOTH floors, because they refute different things:
  vs `prior`  — did the head need the features at all?
  vs `pixel`  — did the TRAINED encoder add anything over the raw patches it
                was handed?

TIER: T0-DIAGNOSTIC.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
          r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
          r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
sys.path.insert(0, str(SP))
import e_detect as E      # noqa: E402
import e_detect_prep as P  # noqa: E402
import e_trunk2_probe as T  # noqa: E402

N_BOOT = 2000             # deltas are cheap; use the registry's usual depth


def prior_pred(occ: np.ndarray, folds) -> np.ndarray:
    """The closed-form floor, rebuilt per fold from TRAIN rows only."""
    pr = np.zeros_like(occ, dtype=np.float32)
    for k, te in enumerate(folds):
        tr = np.concatenate([folds[j] for j in range(len(folds)) if j != k])
        pr[te] = occ[tr].mean(0)[None, :]
    return pr


def paired(a: np.ndarray, b: np.ndarray, occ: np.ndarray,
           rows_by_ep: dict[str, np.ndarray], rng) -> dict:
    """Paired episode-cluster bootstrap of (metric[a] - metric[b])."""
    names = list(rows_by_ep)
    pt = {"d_ap": E.average_precision(occ.ravel(), a.ravel())
          - E.average_precision(occ.ravel(), b.ravel()),
          "d_auc": E.auroc(occ.ravel(), a.ravel())
          - E.auroc(occ.ravel(), b.ravel())}
    da, du = [], []
    for _ in range(N_BOOT):
        pick = rng.choice(len(names), len(names), replace=True)
        idx = np.concatenate([rows_by_ep[names[j]] for j in pick])
        y = occ[idx].ravel()
        sa, sb = a[idx].ravel(), b[idx].ravel()
        da.append(E.average_precision(y, sa) - E.average_precision(y, sb))
        du.append(E.auroc(y, sa) - E.auroc(y, sb))
    out = {}
    for k, v in (("d_ap", da), ("d_auc", du)):
        lo, hi = np.nanpercentile(v, [2.5, 97.5])
        out[k] = round(pt[k], 4)
        out[f"{k}_ci95"] = [round(float(lo), 4), round(float(hi), 4)]
        # the sign question, answered directly rather than by eyeballing an
        # interval: how often does the resample put the arm BELOW the floor?
        out[f"{k}_p_le0"] = round(float(np.mean(np.asarray(v) <= 0)), 4)
    return out


def main() -> None:
    keys = [tuple(k) for k in
            json.loads((P.FEAT / "keys.json").read_text(encoding="utf-8"))]
    ep = [k[0] for k in keys]
    occ = np.load(P.OUT / "occ.npy")
    folds = T.episode_folds(ep)
    tmp: dict[str, list[int]] = {}
    for i, e in enumerate(ep):
        tmp.setdefault(e, []).append(i)
    rows_by_ep = {k: np.array(v) for k, v in tmp.items()}

    preds = {"prior": prior_pred(occ, folds)}
    for f in sorted(SP.glob("e_detect_pred_*.npy")):
        preds[f.stem.replace("e_detect_pred_", "")] = np.load(f)
    print("arms available:", ", ".join(preds))

    floors = [f for f in ("prior", "pixel") if f in preds]
    out = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC",
           "estimator": "PAIRED episode-cluster bootstrap of the delta, "
                        f"{N_BOOT} replicates over {len(rows_by_ep)} episodes; "
                        "both arms evaluated on the SAME resample",
           "n_boot": N_BOOT, "floors": floors, "deltas": {}}

    for arm, pa in preds.items():
        for fl in floors:
            if arm == fl:
                continue
            rng = np.random.default_rng(E.SEED)
            d = paired(pa, preds[fl], occ, rows_by_ep, rng)
            out["deltas"][f"{arm}_vs_{fl}"] = d
            sig = ("ABOVE" if d["d_ap_ci95"][0] > 0 else
                   "BELOW" if d["d_ap_ci95"][1] < 0 else "indistinguishable")
            print(f"  {arm:<14} vs {fl:<7} dAP {d['d_ap']:+.4f} "
                  f"{d['d_ap_ci95']}  dAUC {d['d_auc']:+.4f} "
                  f"{d['d_auc_ci95']}  -> {sig}", flush=True)
            (SP / "e_detect_paired.json").write_text(
                json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n-> {SP / 'e_detect_paired.json'}")


if __name__ == "__main__":
    main()
