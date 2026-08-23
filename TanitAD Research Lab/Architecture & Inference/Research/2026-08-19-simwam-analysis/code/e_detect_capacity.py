"""E-DETECT-1C — is the HEAD too big for 130 clips? Test it, do not infer it.

⛔ THE CLAIM UNDER TEST. Five grid arms scored at or below the closed-form
`prior`, and `e_detect_variance.py` showed why: every head — RAW PIXELS INCLUDED
— learns the prior's spatial shape (corr +0.91..+0.98) and then adds
frame-specific variation that does not survive an episode-disjoint split. That is
overfitting, and it makes the whole comparison a statement about the head rather
than about any trunk.

⭐ THAT CLAIM HAS A DIRECT TEST, and it costs minutes: SHRINK THE HEAD AND WATCH
THE OUT-OF-FOLD AP. If overfitting is the binding constraint, AP RISES as
capacity falls, peaks, and then falls again once the head is too small to express
the task. If AP is flat or falls monotonically, overfitting is NOT the story and
the nulls need a different explanation.

Run on the 16-token arms only (~20 s/fold), so the primary sweep keeps the GPU
essentially to itself:

  v6_cells      the deployed latent — scored 0.0888, BELOW prior 0.1242
  dino_pooled   the only arm above prior (0.1416); its curve says whether the
                capacity that hurts v6_cells also caps the arm that works

⚠️ THIS IS A DIAGNOSTIC SWEEP, NOT AN ARM. Picking the winning capacity and then
quoting its AP as "the trunk's score" would be selection on the eval fold — the
winner's-curse shape SEL-1 refuses. The output is the SHAPE of the curve; any
capacity adopted from it must be re-run on every arm identically before a single
number is quoted.

TIER: T0-DIAGNOSTIC.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

SP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
          r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
          r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
sys.path.insert(0, str(SP))
import e_detect as E        # noqa: E402
import e_detect_prep as P   # noqa: E402
import e_trunk2_probe as T  # noqa: E402

#: (d_model, layers, epochs) — capacity falls left to right, then epochs fall.
#: The last rung tests whether the head is being given too much TIME as well as
#: too much width, since both buy overfitting.
LADDER = [
    ("d192_l2_e30", 192, 2, 30),          # the incumbent, for reference
    ("d96_l2_e30", 96, 2, 30),
    ("d48_l2_e30", 48, 2, 30),
    ("d48_l1_e30", 48, 1, 30),
    ("d24_l1_e30", 24, 1, 30),
    ("d48_l1_e10", 48, 1, 10),
]
ARMS = ("v6_cells", "dino_pooled")


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
    base = float(occ.mean())
    pos_w = (1 - base) / base

    pr = np.zeros_like(occ, np.float32)
    for k, te in enumerate(folds):
        tr = np.concatenate([folds[j] for j in range(len(folds)) if j != k])
        pr[te] = occ[tr].mean(0)[None, :]
    prior_ap = E.average_precision(occ.ravel(), pr.ravel())
    print(f"  prior AP {prior_ap:.4f}   base rate {base:.4f}\n")

    out = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC",
           "question": "does out-of-fold AP RISE as head capacity falls? "
                       "(if yes, the head was the binding constraint)",
           "warning": "DIAGNOSTIC SWEEP, NOT AN ARM. Adopting the winning rung "
                      "and quoting its AP would be selection on the eval fold.",
           "prior_ap": round(prior_ap, 4), "base_rate": round(base, 6),
           "ladder": [], "arms": {}}

    for arm in ARMS:
        f, n_tok, d_in, _ = E.ARMS[arm]
        if not (SP / "sp2" / f).exists():
            print(f"  [skip] {arm}"); continue
        X = E.load_arm(arm)
        out["arms"][arm] = {}
        for name, d, layers, epochs in LADDER:
            E.D_MODEL, E.EPOCHS = d, epochs
            _orig = E.OccHead

            class Head(_orig):                       # noqa: N801
                def __init__(self, d_in_, n_tok_, *a, **kw):
                    kw.pop("d", None); kw.pop("layers", None)
                    super().__init__(d_in_, n_tok_, *a, d=d, layers=layers, **kw)

            E.OccHead = Head
            t0 = time.time()
            pred = np.zeros_like(occ, np.float32)
            taps = []
            try:
                for k, te in enumerate(folds):
                    tr = np.concatenate([folds[j] for j in range(len(folds))
                                         if j != k])
                    pred[te], tap = E.run_fold(X, occ, tr, te, n_tok, d_in,
                                               pos_w)
                    taps.append(tap)
                ap = E.average_precision(occ.ravel(), pred.ravel())
                auc = E.auroc(occ.ravel(), pred.ravel())
                params = sum(p.numel() for p in Head(d_in, n_tok).parameters())
            finally:
                E.OccHead = _orig
            rec = {"d_model": d, "layers": layers, "epochs": epochs,
                   "params": int(params), "ap": round(ap, 4),
                   "auc": round(auc, 4),
                   "ap_minus_prior": round(ap - prior_ap, 4),
                   "train_ap_mean": round(float(np.mean(taps)), 4),
                   "overfit_gap": round(float(np.mean(taps)) - ap, 4),
                   "s": round(time.time() - t0, 1)}
            out["arms"][arm][name] = rec
            if name not in [x["name"] for x in out["ladder"]]:
                out["ladder"].append({"name": name, "d_model": d,
                                      "layers": layers, "epochs": epochs})
            print(f"  {arm:<12} {name:<12} params {params:>8,}  "
                  f"AP {ap:.4f} ({ap - prior_ap:+.4f} vs prior)  "
                  f"train_ap {np.mean(taps):.4f}  gap {rec['overfit_gap']:+.4f}"
                  f"  ({rec['s']:.0f}s)", flush=True)
            (SP / "e_detect_capacity.json").write_text(
                json.dumps(out, indent=1), encoding="utf-8")
        del X
        gc.collect()
        print()
    print(f"-> {SP / 'e_detect_capacity.json'}")


if __name__ == "__main__":
    main()
