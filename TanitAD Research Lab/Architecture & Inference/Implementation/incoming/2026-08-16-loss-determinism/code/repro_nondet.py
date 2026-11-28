"""BEFORE-measurement: reproduce v6_loss_step non-determinism and quantify PER TERM.

Also DYNAMICALLY enumerates every global-RNG consumer in the loss path
(C74: an unverified *enumeration* is the failure mode, so this does not
spot-check a grep list — it instruments torch itself and records tracebacks).
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import torch

_STACK = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack")
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))

from tanitad.config import EncoderConfig, PredictorConfig, ReadoutConfig  # noqa
from tanitad.models.v6 import V6Config, V6Stack  # noqa
import train_v6_staged as T  # noqa

STAGES = ("S-W", "S-T", "S-S", "S-J")


def _sub_cfgs():
    return dict(
        encoder=EncoderConfig(in_channels=9, image_size=64, image_width=64,
                              patch_size=16, d_model=32, depth=1, n_heads=2),
        readout=ReadoutConfig(grid=2, d_readout=8),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=2,
                                  action_dim=3))


def build(seed=0):
    torch.manual_seed(seed)
    cfg = V6Config(d_tac=32, d_str=16, adapter_hidden=32, f_hidden_tac=32,
                   f_hidden_str=32, f_blocks=1, aux_hidden=16, sigreg_slices=8,
                   plan_steps=6, dt=0.1, op_band_s=(0.0, 0.2),
                   tac_band_s=(0.2, 0.6), hz_op=10.0, hz_tac=2.0, hz_str=0.5,
                   d_plan_feat=16, emission_hidden=16, d_goal_embed=128,
                   n_candidates=8, **_sub_cfgs())
    return V6Stack(cfg)


def mk_batch(stack):
    b = T.synthetic_train_batch(stack, batch=2, k=4, seed=7)
    b["gt_wp"] = torch.zeros(2, 2, 2)
    return b


# --------------------------------------------------------------------------
# dynamic enumeration of global-RNG consumers
# --------------------------------------------------------------------------
_HITS: dict[str, dict] = {}
_ACTIVE = [False]

_FUNCS = ["randn", "rand", "randint", "randperm", "normal", "bernoulli",
          "multinomial", "randn_like", "rand_like", "randint_like", "poisson"]
_METHODS = ["normal_", "uniform_", "random_", "bernoulli_", "exponential_",
            "cauchy_", "log_normal_", "geometric_"]


def _install():
    orig = {}
    for name in _FUNCS:
        f = getattr(torch, name, None)
        if f is None:
            continue
        orig[("torch", name)] = f

        def wrap(f=f, name=name):
            def inner(*a, **kw):
                if _ACTIVE[0] and kw.get("generator") is None:
                    _rec(f"torch.{name}")
                return f(*a, **kw)
            return inner
        setattr(torch, name, wrap())
    for name in _METHODS:
        f = getattr(torch.Tensor, name, None)
        if f is None:
            continue
        orig[("Tensor", name)] = f

        def wrapm(f=f, name=name):
            def inner(self, *a, **kw):
                if _ACTIVE[0] and kw.get("generator") is None:
                    _rec(f"Tensor.{name}")
                return f(self, *a, **kw)
            return inner
        setattr(torch.Tensor, name, wrapm())
    # dropout: functional + the module's functional call
    import torch.nn.functional as F
    for name in ("dropout", "dropout1d", "dropout2d", "dropout3d",
                 "alpha_dropout", "feature_alpha_dropout"):
        f = getattr(F, name, None)
        if f is None:
            continue
        orig[("F", name)] = f

        def wrapd(f=f, name=name):
            def inner(*a, **kw):
                p = kw.get("p", a[1] if len(a) > 1 else 0.5)
                tr = kw.get("training", a[2] if len(a) > 2 else True)
                if _ACTIVE[0] and tr and p and float(p) > 0.0:
                    _rec(f"F.{name}(p={p})")
                return f(*a, **kw)
            return inner
        setattr(F, name, wrapd())
    return orig


def _rec(what):
    st = traceback.extract_stack()[:-2]
    # the deepest frame inside our repo
    ours = [fr for fr in st if "TanitAD" in fr.filename.replace("\\", "/")]
    site = "?"
    if ours:
        fr = ours[-1]
        rel = fr.filename.replace("\\", "/").split("TanitAD/")[-1]
        site = f"{rel}:{fr.lineno}  ({fr.name})  |  {fr.line}"
    key = f"{what} @ {site}"
    _HITS.setdefault(key, {"what": what, "site": site, "n": 0})["n"] += 1


def main():
    _install()
    report = {"per_stage": {}, "rng_sites": None,
              "torch": torch.__version__}

    for stage in STAGES:
        stack = build()
        stack.train()                    # the regime the PI measured
        batch = mk_batch(stack)
        kw = dict(stage=stage, o1_k=2, o5_k=2, weights=T.V6LossWeights())

        # --- global RNG-state watcher: airtight, enumeration-free ----------
        torch.manual_seed(123)
        st0 = torch.random.get_rng_state().clone()
        _ACTIVE[0] = True
        _HITS.clear()
        _ = T.v6_loss_step(stack, batch, generator=torch.Generator()
                           .manual_seed(11), **kw)
        _ACTIVE[0] = False
        st1 = torch.random.get_rng_state().clone()
        consumed = not torch.equal(st0, st1)
        sites = dict(_HITS)

        # --- two identical calls, same generator seed ---------------------
        def call():
            return T.v6_loss_step(stack, batch,
                                  generator=torch.Generator().manual_seed(11),
                                  **kw)

        a = call()
        b = call()
        terms = sorted(set(a["log"]["terms"]) | set(b["log"]["terms"]))
        per_term = {}
        for t in terms:
            ta, tb = a[t], b[t]
            va, vb = float(ta), float(tb)
            eq = bool(torch.equal(ta, tb))
            rel = abs(va - vb) / max(abs(va), 1e-30)
            per_term[t] = {"a": va, "b": vb, "bit_equal": eq,
                           "abs_delta": abs(va - vb), "rel_pct": 100.0 * rel}
        la, lb = float(a["loss"]), float(b["loss"])
        report["per_stage"][stage] = {
            "global_rng_consumed": consumed,
            "rng_sites": sites,
            "total": {"a": la, "b": lb, "bit_equal":
                      bool(torch.equal(a["loss"], b["loss"])),
                      "abs_delta": abs(la - lb),
                      "rel_pct": 100.0 * abs(la - lb) / max(abs(la), 1e-30)},
            "terms": per_term,
        }

    print(json.dumps(report, indent=1))
    out = Path(__file__).with_name("repro_before.json")
    out.write_text(json.dumps(report, indent=1))
    print("\nWROTE", out)


if __name__ == "__main__":
    main()
