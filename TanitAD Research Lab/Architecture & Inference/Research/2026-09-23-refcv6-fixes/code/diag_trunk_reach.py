"""Which loss terms reach the shared trunk, on the REAL refcv6 launch configuration?

Runs the real `train()` with the argv a Thor smoke recorded in its config.json (overriding only
--out / --steps / --batch / --workers / eval off / conflict detector ON). At the first call of
`_conflict_terms` -- i.e. on the first real batch, before any backward -- it takes
`torch.autograd.grad(term, trunk_params, allow_unused=True)` for EVERY scalar loss term that
requires grad and prints: trunk tensors reached (non-None), tensors with a non-zero gradient,
the gradient's L2 norm, and whether it is finite. Then it stops the run.

    python diag_trunk_reach.py <smoke config.json> <out dir>
"""
import json
import sys

import torch

sys.argv, cfg_path, out_dir = sys.argv[:1], sys.argv[1], sys.argv[2]
import refc_v3_train as T  # noqa: E402

argv = json.load(open(cfg_path, encoding="utf-8"))["argv"]


def _drop(av, flag, n=1):
    out, i = [], 0
    while i < len(av):
        if av[i] == flag:
            i += 1 + n
            continue
        out.append(av[i])
        i += 1
    return out


for f, n in (("--out", 1), ("--steps", 1), ("--batch", 1), ("--workers", 1),
             ("--eval-cache", 1), ("--eval-labels", 1), ("--eval-every", 1),
             ("--eval-batches", 1), ("--speed-max-sidecar-v6-eval", 1),
             ("--conflict-detector", 1)):
    argv = _drop(argv, f, n)
argv += ["--out", out_dir, "--steps", "2", "--batch", "1", "--workers", "0",
         "--conflict-detector", "on"]


class _Stop(Exception):
    pass


real_terms = T._conflict_terms


_calls = {'n': 0}


def spy(model, losses):
    _calls['n'] += 1
    if _calls['n'] == 1:
        # step 1: skip the detector (its controls are what we are diagnosing) and let
        # the optimizer take one real step, so the zero-init control head moves.
        return (None, None)
    print('DIAG measuring at call', _calls['n'], '(after one optimizer step)', flush=True)
    params = [(n, p) for n, p in model.named_parameters()
              if n.startswith("core.encoder.") and p.requires_grad]
    print("DIAG trunk tensors:", len(params),
          "params:", sum(p.numel() for _, p in params), flush=True)
    for key, t in sorted(losses.items()):
        if not (torch.is_tensor(t) and t.ndim == 0 and t.requires_grad):
            continue
        g = torch.autograd.grad(t, [p for _, p in params], retain_graph=True,
                                allow_unused=True)
        reached = sum(1 for x in g if x is not None)
        nonzero = sum(1 for x in g if x is not None and bool(x.abs().sum() > 0))
        sq = sum(float(x.double().pow(2).sum()) for x in g if x is not None)
        finite = all(bool(torch.isfinite(x).all()) for x in g if x is not None)
        print("DIAG term %-24s value %-12.5g reached %3d/%d nonzero %3d norm %.6g finite %s"
              % (key, float(t.detach()), reached, len(params), nonzero, sq ** 0.5, finite),
              flush=True)
    lt, la = real_terms(model, losses)
    print("DIAG conflict plan-term is None:", lt is None, "| aux is None:", la is None, flush=True)
    raise _Stop()


T._conflict_terms = spy
try:
    T.train(T.build_parser().parse_args(argv))
except _Stop:
    print("DIAG_DONE", flush=True)
