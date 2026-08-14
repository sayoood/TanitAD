"""Run the P-battery's frozen-trunk probes against a **v6** checkpoint.

⛔ WHY THIS EXISTS. MEASURED 2026-08-14 on pod4: the P-battery could not read a
v6 checkpoint, and the blocker was ARCHITECTURAL, not a key name. Four runs each
fixed one surface assumption and exposed the next; the last one settles it —
even a checkpoint in perfect v5 ``{"model": ...}`` layout failed with
``KeyError: 'predictor.act_emb.0.weight'``, a v5 ``WorldModel`` PARAMETER name.
The probes were building a v5 model and inferring ``action_dim`` from v5
parameter names. v6's module tree is ``predictor_op.*``.

⭐ THE PORT IS THIN, AND THAT IS THE POINT. ``collect_grid`` and ``p8_latents``
need exactly four things from the trunk:

    world.encode_window(seq) -> [B, T, S]
    world.predictor          -> driven by rollout_transitions as predictor(s, a)[1]
    world.state_dim          -> int
    world.parameters() / named_parameters()

``V6Stack`` already satisfies the first two natively: it *has* ``encode_window``,
and ``OperativePredictor.forward(states, actions, intent=None)`` returns
``dict[int, Tensor]``, so ``rollout_transitions``' ``predictor(ws, wa)[1]``
selects the 1-step head exactly as it does for v5. The v6 predictor is built
with ``action_dim=3`` (v6.py:718) — the SAME 3-channel lifted format
``lift_actions3`` produces for v5 — so the action contract needs no translation
either. What was missing was only a construction path and the adapter below.

⚠️ STRICT LOAD, ALWAYS. The reconstruction goes through the run's own
``config["args"]`` and ``build_stack_from_args``, i.e. the trainer's exact
wiring path, so ``load_state_dict`` stays ``strict=True``. Never relax it: a
non-strict load leaves probe tensors random-initialised and emits numbers that
LOOK like results — the failure ``ckpt_compat``'s docstring was written to
prevent, and the one this file must not reintroduce.

⚠️ THIS IS A WM DIAGNOSTIC (T0-family). Nothing produced through this trunk is
a driving-performance claim; see ``EVAL_DOCTRINE.md``.
"""
from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import torch
from torch import Tensor


def _ensure_scripts() -> None:
    """Make ``stack/scripts`` importable — the ``models/v6.py:_ensure_scripts``
    precedent (``tanitad/eval/…`` -> ``parents[2]`` == the stack root)."""
    sp = str(Path(__file__).resolve().parents[2] / "scripts")
    if sp not in sys.path:
        sys.path.insert(0, sp)


def is_v6_checkpoint(ck) -> bool:
    """True for the v6 staged-trainer layout ``{"stack": sd, "config": …}``.

    Keyed on ``"stack"`` rather than on parameter names so the check is cheap
    and cannot be fooled by a v5 ckpt that happens to carry v6-ish tensors.
    """
    return isinstance(ck, dict) and "stack" in ck and "model" not in ck


def _run_args(ck, ckpt_path=None) -> dict:
    """The run's CLI args, from the ckpt's own ``config`` or the ``config.json``
    the trainer banks beside it.

    ⛔ Both sources are the RUN's, never a default. A v6 stack rebuilt from
    stock defaults would differ from the trained one and the strict load would
    refuse — which is the correct outcome, but a confusing one to debug. Failing
    here with an explicit message is better than failing in ``load_state_dict``.
    """
    cfg = ck.get("config") if isinstance(ck, dict) else None
    if not (isinstance(cfg, dict) and cfg.get("args")):
        side = Path(ckpt_path).parent / "config.json" if ckpt_path else None
        if side is not None and side.is_file():
            cfg = json.loads(side.read_text(encoding="utf-8"))
    if not (isinstance(cfg, dict) and isinstance(cfg.get("args"), dict)):
        raise SystemExit(
            "[v6-probe] checkpoint carries no run config and no config.json "
            "beside it — cannot rebuild the exact architecture. The v6 trainer "
            "banks both; a checkpoint that travelled without its config is a "
            "refused restart by design (POD_HANDOVER §2)."
        )
    return dict(cfg["args"])


class V6ProbeTrunk:
    """Adapter presenting a :class:`V6Stack` through the v5 trunk interface.

    Deliberately NOT an ``nn.Module`` subclass: it must not add parameters, and
    the probes' ``assert not any(p.requires_grad …)`` / ``module_md5`` should see
    the wrapped stack's own parameters, unchanged.
    """

    def __init__(self, stack):
        self.stack = stack
        #: `rollout_transitions(predictor, …)` calls `predictor(s, a)[1]`.
        self.predictor = stack.predictor_op
        #: the geometry firewall's width — the single source of the state dim.
        self.state_dim = int(stack.cfg.d_op)

    def encode_window(self, frames: Tensor) -> Tensor:
        return self.stack.encode_window(frames)

    def parameters(self, recurse: bool = True):
        return self.stack.parameters(recurse)

    def named_parameters(self, *a, **kw):
        return self.stack.named_parameters(*a, **kw)

    def eval(self):
        self.stack.eval()
        return self

    def to(self, *a, **kw):
        self.stack.to(*a, **kw)
        return self


def load_v6_from_ck(ck, device, *, ckpt_path=None) -> tuple[V6ProbeTrunk, int]:
    """Rebuild the exact trained :class:`V6Stack`, freeze it, wrap it.

    Returns ``(trunk, step)`` — the same shape as the v5
    ``load_v1_from_ck(...)[0], [2]`` pair the probes already consume.
    """
    _ensure_scripts()
    from train_v6_staged import build_stack_from_args  # noqa: PLC0415

    args = _run_args(ck, ckpt_path)
    stack = build_stack_from_args(Namespace(**args))
    stack.load_state_dict(ck["stack"], strict=True)      # STRICT — see module doc
    stack.to(device).eval()
    for p in stack.parameters():
        p.requires_grad_(False)
    return V6ProbeTrunk(stack), int(ck.get("step", -1))


def load_trunk_auto(ck, device, *, ckpt_path=None, frame=None):
    """v6 checkpoint -> :func:`load_v6_from_ck`; anything else -> the v5 path.

    Lets a single probe script read both generations without a flag, so a v6
    run cannot be silently scored through a v5 construction path.
    """
    if is_v6_checkpoint(ck):
        trunk, step = load_v6_from_ck(ck, device, ckpt_path=ckpt_path)
        return trunk, None, step
    _ensure_scripts()
    from eval_flagship_v4 import load_v1_from_ck  # noqa: PLC0415

    return load_v1_from_ck(ck, device, frame=frame)
