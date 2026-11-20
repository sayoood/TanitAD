"""PREREG_V8F_N1DH control K-LIM (MM review item 1): the route head reads O's plan feature h, and h reads ``e_lim(limit)``. In TRAINING ``limit`` is the clip's LOGGED speed bucket, built from the ego's realised speed over
[t0 + 2 s, t0 + 6 s] (row H-V8F-CAUSALBUCKET-1): future information that correlates with a turn. At T1-kin inference the bucket is CAUSAL (``ROLLOUT["limit"] == "causal"``, v8f_dagger.py). The own-code mask does not remove this path.

K-LIM reads, on ONE collated batch, the head's accuracy (argmax of ``route_logits`` against the kinematic label ``route_y``, on the rows that carry a label) with ``limit`` = the logged bucket, = the causal bucket, = off (zeros, ``limit_ok``
0: the ``--limit off`` format). A large logged-minus-causal gap names the shortcut. REPORT ONLY: no bar, nothing in the trainer or the gate reads it.

The causal bucket here is ``speed_max_bin`` of the measured ego speed AT NOW (``kin3[:, 0]``, which reads nothing after NOW). The evaluator's own rule takes the max over the last 21 rows (4 s, ``v8f_t1_eval.causal_bucket``); a collated batch carries
only the speed at NOW, so this is the one-row form of that rule (it is <= the 4 s max, so it can only move a row DOWN a bucket). State this beside any number.

Usage:  python n1dh_klim.py --ckpt CKPT --batch-pt BATCH.pt --out OUT.json        (BATCH.pt = torch.save of a collated batch with ``route_y`` / ``route_ok``, e.g. from the trainer's probe batch)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in (os.path.join(REPO, "stack"), os.path.join(REPO, "taniteval"), os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

VARIANTS = ("logged", "causal", "off")


def causal_limit(batch: dict):
    """The one-row causal bucket ``(limit [B, 4] float32 one-hot, limit_ok [B] bool)`` from the measured ego speed at NOW (``kin3[:, 0]``); a row with no finite speed is ``off`` (zeros, False)."""
    from tanitad.refs.refcv6_max_speed import speed_max_bin
    v = batch["kin3"][:, 0].float().numpy()
    oh = np.zeros((v.shape[0], 4), np.float32)
    ok = np.isfinite(v)
    for i in np.where(ok)[0]:
        oh[i, speed_max_bin(float(max(v[i], 0.0)))[0]] = 1.0
    return torch.as_tensor(oh), torch.as_tensor(ok)


def with_limit(batch: dict, variant: str) -> dict:
    b = dict(batch)
    if variant == "logged":
        return b
    if variant == "causal":
        b["limit"], b["limit_ok"] = causal_limit(batch)
    elif variant == "off":
        b["limit"], b["limit_ok"] = torch.zeros_like(batch["limit"]), torch.zeros_like(batch["limit_ok"])
    else:
        raise ValueError(variant)
    return b


def klim(model, batch: dict) -> dict:
    """The K-LIM record on one batch: per variant, the route head's argmax accuracy on the labelled rows, its mean max-posterior and the plan's L1 distance to the logged-limit plan."""
    if getattr(model, "route_head", None) is None:
        raise ValueError("K-LIM needs a model built with --aux-route cond")
    ok = batch["route_ok"].bool()
    y = batch["route_y"].long()
    model.eval()
    res, base_plan = {}, None
    with torch.no_grad():
        for v in VARIANTS:
            out = model.forward_infer(with_limit(batch, v))
            pr = torch.softmax(out["route_logits"].float(), -1)
            acc = float((pr.argmax(-1) == y)[ok].float().mean()) if bool(ok.any()) else None
            plan = out["traj30"].float()
            base_plan = plan if base_plan is None else base_plan
            res[v] = {"acc": acc, "mean_max_posterior": float(pr.max(-1).values[ok].mean()) if bool(ok.any()) else None, "plan_l1_to_logged": float((plan - base_plan).abs().mean()),
                      "argmax_class_counts": [int(((pr.argmax(-1) == c) & ok).sum()) for c in range(pr.shape[1])]}
    n = int(ok.sum())
    chance = float(max(np.bincount(y[ok].numpy(), minlength=3))) / n if n else None
    res["n_labelled_rows"], res["majority_share"] = n, chance
    res["gap_logged_minus_causal"] = None if res["logged"]["acc"] is None else res["logged"]["acc"] - res["causal"]["acc"]
    res["gap_logged_minus_off"] = None if res["logged"]["acc"] is None else res["logged"]["acc"] - res["off"]["acc"]
    res["note"] = ("REPORT ONLY (PREREG_V8F_N1DH K-LIM). causal = speed_max_bin of the measured speed at NOW (the one-row form of the evaluator's 4 s rule). "
                   "A large logged-minus-causal gap names the logged-bucket shortcut; a near-zero gap does not clear the 4 s form.")
    return res


def load_model(ckpt_path: str):
    """The evaluators' build path: ``V8FMiniConfig(**ck["cfg"])`` -> V8FMini (the prior and the constraint statistics are state-dict entries, overwritten by the strict load)."""
    from tanitad.models import v8f_mini as M
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior.synthetic(0), torch.zeros(M.N_CONS), torch.ones(M.N_CONS))
    model.load_state_dict(ck["model"])
    return model


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="a checkpoint of an --aux-route cond arm")
    ap.add_argument("--batch-pt", required=True, help="torch.save of ONE collated batch that carries route_y / route_ok (the label stage is v8f_mini_train.route_stage)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    batch = torch.load(a.batch_pt, map_location="cpu", weights_only=False)
    if "route_y" not in batch:
        import v8f_mini_train as TR
        batch["route_y"], batch["route_ok"] = TR.route_stage(batch)
    res = klim(load_model(a.ckpt), batch)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: res[k] for k in ("n_labelled_rows", "majority_share", "gap_logged_minus_causal", "gap_logged_minus_off")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
