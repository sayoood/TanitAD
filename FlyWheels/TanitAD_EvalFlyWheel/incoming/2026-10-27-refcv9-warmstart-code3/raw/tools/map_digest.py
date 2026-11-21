"""map_digest.py -- the flags-absent bit identity of the MAP rig (test_map_only_ft's tiny REAL RefCV3Model with the 10 cm
branch: the path stepid_digest's refcv8 rig does not build) and of the `--r8-train-only-map` path, pristine vs
modified. Same digest / report format as stepid_digest.py (its --compare reads these files).

  arms: map_rig_step      -- flags absent: compute_losses_v3 (the 10 cm term included), backward, ONE AdamW step;
        r8_freeze_2steps  -- `--r8-train-only-map`: apply_freeze, the trainer's build_optimizer + recipe assert,
                             WeightEMA(exclude=ema_exclude), TWO steps; digests both steps' losses, the grads, the
                             post-step state, the EMA export and the freeze record.
  --mutate fp  : the DISCRIMINATING control -- `_w_map_fp = 0.5` on the map rig (the digest must see the term);
  --mutate r9  : `--r9-train-only planner,map` instead of the r8 freeze (the digest must see a different freeze).

  python -X utf8 map_digest.py --tree <tree> --out <json> [--mutate fp|r9]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _digest(t) -> str:
    import torch
    t = t.detach().cpu().reshape(-1)
    d = torch.empty(t.shape, dtype=t.dtype)
    d.copy_(t)
    h = hashlib.sha256()
    h.update(str(t.dtype).encode() + str(tuple(t.shape)).encode())
    h.update(d.view(torch.uint8).numpy().tobytes() if d.numel() else b"")
    return h.hexdigest()


def _losses(acc, L, pre):
    import torch
    for k in sorted(L, key=str):
        v = L[k]
        if torch.is_tensor(v):
            acc[f"{pre}.{k}"] = _digest(v)
        elif isinstance(v, (int, float, bool, str)) or v is None:
            acc[f"{pre}.{k}"] = repr(v)


def run(tree: Path, mutate: str | None) -> dict:
    stack = tree / "stack"
    for p in (str(stack), str(stack / "scripts"), str(stack / "tests")):
        sys.path.insert(0, p)
    import torch
    import tanitad
    import refc_v3_train as T
    import test_map_only_ft as R
    from tanitad.train import map_only_ft as M
    from tanitad.train import weight_ema as W
    torch.set_num_threads(1)
    res = {"tanitad_file": tanitad.__file__, "torch": torch.__version__, "mutate": mutate, "arms": {}}

    # ---- 1. the map rig, flags absent: one step ------------------------------------------------------------------- #
    model, cfg = R._build()
    if mutate == "fp":
        model._w_map_fp = 0.5
    batch = R._batch(cfg)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    opt.zero_grad(set_to_none=True)
    torch.manual_seed(5)
    L = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
    L["loss"].backward()
    acc = {}
    _losses(acc, L, "loss")
    for k, p in model.named_parameters():
        if p.grad is not None:
            acc[f"grad.{k}"] = _digest(p.grad)
    opt.step()
    st = {k: _digest(v) for k, v in model.state_dict().items()}
    res["arms"]["map_rig_step"] = {"out": acc, "state": st, "n_out": len(acc), "n_state": len(st),
                                   "loss": float(L["loss"].detach())}

    # ---- 2. --r8-train-only-map: the freeze, the trainer's optimizer, the EMA, two steps -------------------------- #
    model, cfg = R._build()
    args = R._args()
    if mutate == "r9":
        args = R._args(r8_train_only_map=False, r9_train_only="planner,map")
        rec = M.apply_train_only(model, ("planner", "map"))
        ema = W.WeightEMA(model, 0.9, exclude=M.ema_exclude_for(rec["modules"]))
    else:
        rec = M.apply_freeze(model)
        ema = W.WeightEMA(model, 0.9, exclude=M.ema_exclude)
    opt = T.build_optimizer(model, args)
    T.assert_optimizer_matches_recipe(opt, args, model=model)
    batch = R._batch(cfg)
    model.train()
    acc = {"freeze_record": hashlib.sha256(json.dumps(rec, sort_keys=True, default=str).encode()).hexdigest(),
           "opt_groups": repr([(g.get("name"), g["lr"], g["weight_decay"], len(g["params"])) for g in opt.param_groups])}
    for i in range(2):
        torch.manual_seed(5 + i)
        L = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
        _losses(acc, L, f"loss{i}")
        opt.zero_grad(set_to_none=True)
        L["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        for k, p in model.named_parameters():
            if p.grad is not None:
                acc[f"grad{i}.{k}"] = _digest(p.grad)
        opt.step()
        ema.update(model)
    for k, v in ema.model_state_dict(model).items():
        acc[f"ema.{k}"] = _digest(v)
    st = {k: _digest(v) for k, v in model.state_dict().items()}
    res["arms"]["r8_freeze_2steps"] = {"out": acc, "state": st, "n_out": len(acc), "n_state": len(st),
                                       "loss": float(L["loss"].detach())}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mutate", default=None, choices=("fp", "r9"))
    a = ap.parse_args()
    r = run(Path(a.tree), a.mutate)
    Path(a.out).write_text(json.dumps(r, indent=1), encoding="utf-8")
    print("tanitad from", r["tanitad_file"], {k: (v["n_out"], v["n_state"], v["loss"]) for k, v in r["arms"].items()})
