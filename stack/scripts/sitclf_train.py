#!/usr/bin/env python3
"""Promoted trainer for the situation classifier — the DEPLOYABLE, VISION-ONLY arm.

WHY THIS EXISTS
---------------
The situation classifier has been studied for weeks through *probes* — ridge/PCA
readouts fitted inside analysis scripts and thrown away. Every result was about a
readout nobody could ship. ``tanitad.eval.sitclf`` already carries the head
(:class:`~tanitad.eval.sitclf.CausalSitHead`) and the fit recipe
(:func:`~tanitad.eval.sitclf.train_sit_head`), but there was **no CLI that trains the
deployable arm end to end and emits a checkpoint**, so no arm could be promoted.

The PI's standing priority is verbatim: *"We must improve the results of scenario
classification, since it's crucial for the tactical layer and thus for the program."*
A lever nothing can train is not a lever.

⛔ THE BINDING RULING THIS SCRIPT ENFORCES IN CODE
--------------------------------------------------
Sayed, 2026-08-03, verbatim: *"for ground truth data of scenario classification you can
use both ego and other label, for inference only vision."*

| stage | what may be used |
|---|---|
| label derivation (offline) | ego state, other agents, maps, future poses — anything |
| **inference** | ⛔ **VISION ONLY** |

So the deployable arm is **``head_img``**. This script **refuses** to build an arm whose
inference-time inputs include the ego block, and it refuses *loudly* rather than warning:
``--ego-at-inference`` does not exist and ``--features ego`` is rejected with a non-zero
exit. That is deliberate — the ego-only and image+ego arms score BETTER on the banked
numbers (``head_ego`` 0.0697 > ``head_img_ego`` 0.0525 > ``head_img`` 0.0376), which makes
the inadmissible arm the tempting one. A rule that is only a comment loses that argument.

⚠️ **Why the ranking is not a reason to reopen ego.** The situation labels are derived from
**ego dynamics** (``tanitad/data/situations.py``). A classifier *given* ego state at
inference is partly reading the label's own source. MEASURED clarification: the head's
window ``[t-0.7 s, t]`` and the label's evidence window ``[onset, onset+4 s]`` with
``onset > t`` are **disjoint**, so this is *same-source privileged access*, not a
future-information leak — but it is still a privileged channel that will not exist at
deployment. "Vision scores worse" is a finding about *how much* and *why*, never a reason
to ship the ego arm.

⚠️ **And motion is not an appearance shortcut here — that was MEASURED, not assumed.** The
still-frame control (same encoder, last RGB frame replicated 3×, motion deleted) costs
**~70 % of the skill** on ``intersection`` (recovery 0.297 / 0.303 / 0.316, all separated).
Reported as an **upper bound**: the frozen trunk never saw a degenerate stack.

WHAT THIS SCRIPT DOES NOT DECIDE
--------------------------------
The window and the anticipation horizon. ``WIN = 8`` (0.8 s) is the deployed constant and
the default here. MEASURED 2026-08-03: ``intersection`` and ``lane_change`` have **opposite
temporal signatures** — intersection skill DECAYS with the horizon (AP-lift +0.982 → +0.378
over 1→5 s) while lane_change RISES and separates only at 5 s — and the programme forces one
window and one horizon onto both. ``--win`` is exposed so a per-situation setting can be
trained once that study reports; it is **not** defaulted to anything new here, because
changing it silently would make this arm incomparable to every banked number.

CAPACITY DISCIPLINE
-------------------
The parameter count is printed and written into ``config.json`` before training starts.
This programme's history: an input lever once cost **+272,001** params before its own
capacity control caught it; the accepted ones were **+897**, **+385**, **+128**. A trainer
that does not report its own size cannot be held to that.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from tanitad.eval import sitclf

#: Inference-time feature sets this trainer will build. ``ego`` and ``img_ego`` are
#: deliberately ABSENT — see the module docstring. They are not omitted by oversight,
#: so they are listed here as refusals rather than silently unsupported.
ADMISSIBLE_FEATURES = ("img",)
REFUSED_FEATURES = {
    "ego": "ego state at inference — the PI's ruling is VISION ONLY",
    "img_ego": "image+ego at inference is still ego at inference",
    "fused": "score-level fusion of image and ego is still ego at inference",
}


def _fail(msg: str) -> "None":
    print(f"[sitclf-train] REFUSED: {msg}", file=sys.stderr)
    raise SystemExit(2)


def load_substrate(path: Path) -> dict:
    """Load a banked substrate ``.npz`` and return its arrays.

    Required keys: ``img`` [N, D] image features, ``y`` [N, S] labels, ``valid``
    [N, S] per-situation validity, ``clip_id`` [N]. ``ego`` may be present — it is
    NOT loaded into the model input, only reported, so that a substrate built for
    the label side can be reused without becoming an inference input by accident.
    """
    if not path.exists():
        _fail(f"substrate not found: {path}")
    z = np.load(path, allow_pickle=False)
    missing = [k for k in ("img", "y", "valid", "clip_id") if k not in z]
    if missing:
        _fail(f"substrate {path} is missing required keys {missing}; has {list(z.keys())}")
    out = {k: z[k] for k in ("img", "y", "valid", "clip_id")}
    out["has_ego_block"] = bool("ego" in z)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Train the deployable VISION-ONLY situation classifier.")
    ap.add_argument("--substrate", type=Path, required=True,
                    help="banked .npz with img / y / valid / clip_id")
    ap.add_argument("--out", type=Path, required=True,
                    help="output directory for ckpt.pt + config.json")
    ap.add_argument("--features", default="img", choices=sorted(ADMISSIBLE_FEATURES),
                    help="inference-time feature set. Only 'img' is admissible; "
                         "ego/img_ego/fused are REFUSED by the PI's vision-only ruling.")
    ap.add_argument("--win", type=int, default=8,
                    help="causal window in frames (8 = 0.8 s, the DEPLOYED constant). "
                         "Exposed for the per-situation-horizon study; changing it makes "
                         "this arm incomparable to every banked number, so say so if you do.")
    ap.add_argument("--width", type=int, default=128,
                    help="head width d. Must be a multiple of 4 — CausalSitHead fixes "
                         "4 attention heads and refuses other widths.")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--pos-weight", type=float, default=20.0)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--folds", type=int, default=2,
                    help="episode-CLUSTER-disjoint folds. Splitting on rows would put "
                         "frames of one clip on both sides and leak.")
    ap.add_argument("--holdout-fold", type=int, default=0,
                    help="fold held out of training and scored")
    args = ap.parse_args(argv)

    # --- the ruling, enforced before anything else happens -------------------
    if args.features in REFUSED_FEATURES:
        _fail(f"--features {args.features}: {REFUSED_FEATURES[args.features]}")
    if args.features not in ADMISSIBLE_FEATURES:
        _fail(f"--features {args.features} is not admissible; "
              f"only {ADMISSIBLE_FEATURES} may reach inference")
    if args.width % sitclf.HEAD_N_HEADS:
        _fail(f"--width {args.width} is not a multiple of {sitclf.HEAD_N_HEADS}; "
              f"CausalSitHead fixes {sitclf.HEAD_N_HEADS} attention heads and "
              f"nn.TransformerEncoderLayer refuses embed_dim % num_heads != 0")
    if not 0 <= args.holdout_fold < args.folds:
        _fail(f"--holdout-fold {args.holdout_fold} out of range for --folds {args.folds}")

    sub = load_substrate(args.substrate)
    X_all, Y, V, clip = sub["img"], sub["y"], sub["valid"], sub["clip_id"]
    in_dim = int(X_all.shape[1])

    starts, ends = sitclf.clip_runs(clip)
    # `causal_window` returns a FULL-LENGTH [N, win*C] array plus an [N] validity
    # mask — rows without a complete in-clip history are zero-filled, not dropped.
    # So the mask must be applied to the features too. Filtering only the labels
    # silently trains row t's history against row t'’s label for every row after
    # the first invalid one; it raised IndexError here only because the synthetic
    # fixture happened to have a size mismatch. On a real substrate whose invalid
    # rows chanced to match the fold size it would have trained on shifted labels
    # and reported a plausible number.
    Xw_full, keep = sitclf.causal_window(X_all, starts, ends, args.win)
    Xw = Xw_full[keep]
    Y, V, clip_w = Y[keep], V[keep], clip[keep]
    if Xw.shape[0] != Y.shape[0] or Xw.shape[0] != clip_w.shape[0]:
        _fail(f"row-count mismatch after windowing: X {Xw.shape[0]}, "
              f"Y {Y.shape[0]}, clip {clip_w.shape[0]}")
    if Xw.shape[0] == 0:
        _fail(f"--win {args.win} leaves no complete in-clip windows "
              f"(shortest clip may be < {args.win} frames)")

    folds = sitclf.cluster_folds(clip_w, n_folds=args.folds, seed=args.seed)
    tr, te = folds != args.holdout_fold, folds == args.holdout_fold
    if not tr.any() or not te.any():
        _fail(f"fold {args.holdout_fold} leaves an empty split "
              f"(train {int(tr.sum())}, held-out {int(te.sum())})")

    n_params = sitclf.head_param_count(in_dim, args.win, args.width, n_out=Y.shape[1])
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = {
        "arm": f"sitclf_head_{args.features}",
        "deployable": True,
        "inference_inputs": ["image_features"],
        # ⭐ Machine-readable provenance, mirroring RefCModel.goal_provenance(). The
        # ruling is checkable by a script, not only by reading this file.
        "ego_at_inference": False,
        "substrate_has_ego_block": sub["has_ego_block"],
        "ruling": "Sayed 2026-08-03 — labels may use ego; inference is VISION ONLY",
        "situations": list(sitclf.SITUATIONS),
        "win": args.win, "win_seconds": args.win / 10.0,
        "in_dim": in_dim, "width": args.width, "params": n_params,
        "epochs": args.epochs, "batch": args.batch, "lr": args.lr, "wd": args.wd,
        "pos_weight": args.pos_weight, "dropout": args.dropout, "seed": args.seed,
        "folds": args.folds, "holdout_fold": args.holdout_fold,
        "n_windows_total": int(Xw.shape[0]),
        "n_train": int(tr.sum()), "n_heldout": int(te.sum()),
        "n_clusters_train": int(np.unique(clip_w[tr]).size),
        "n_clusters_heldout": int(np.unique(clip_w[te]).size),
        "substrate": str(args.substrate),
    }
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2))

    print(f"[sitclf-train] arm={cfg['arm']} VISION-ONLY (ego_at_inference=False)")
    print(f"[sitclf-train] in_dim={in_dim} win={args.win} ({args.win/10:.1f}s) "
          f"width={args.width} params={n_params:,}")
    print(f"[sitclf-train] windows={Xw.shape[0]:,}  train={int(tr.sum()):,} "
          f"({cfg['n_clusters_train']} clusters)  heldout={int(te.sum()):,} "
          f"({cfg['n_clusters_heldout']} clusters), cluster-disjoint")

    t0 = time.time()
    model = sitclf.train_sit_head(
        Xw[tr], Y[tr], V[tr], win=args.win, in_dim=in_dim, epochs=args.epochs,
        pos_weight=args.pos_weight, d=args.width, batch=args.batch, lr=args.lr,
        wd=args.wd, seed=args.seed, device=args.device, log=print)

    scores = sitclf.predict_sit_head(model, Xw[te], in_dim, device=args.device)
    torch.save({"state_dict": model.state_dict(), "config": cfg},
               args.out / "ckpt.pt")
    np.savez_compressed(args.out / "heldout_scores.npz",
                        scores=scores, y=Y[te], valid=V[te], clip_id=clip_w[te])

    # ⛔ Deliberately NOT scored here. AP and its interval belong to the eval
    # harness with the paired episode-cluster bootstrap; a trainer that prints its
    # own headline metric is how "v1.6 is best-in-program" reached a report from a
    # TRAINER LOG that ran ~10 % optimistic against eval_*.py. This emits the raw
    # held-out scores and stops.
    print(f"[sitclf-train] done in {time.time()-t0:.1f}s -> {args.out}")
    print("[sitclf-train] held-out SCORES written; NOT scored here. Run the eval "
          "harness for AP + the paired episode-cluster bootstrap interval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
