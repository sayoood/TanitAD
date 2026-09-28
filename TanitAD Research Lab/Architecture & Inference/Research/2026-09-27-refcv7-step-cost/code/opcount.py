#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DEVICE-INDEPENDENT structure of one training step, per component: kernel-launching aten ops,
FLOPs, bytes touched and device->host SYNCS -- forward, backward and checkpoint recompute.

Why this exists: the dev-box GPU was held by another session (deadline 04:27), so this runs the
REAL trainer step on the CPU. The COUNTS are properties of the program, not of the device: the same
step launches the same kernels and hits the same `.item()` / `.cpu()` syncs on Thor. Times are not
reported from here (CPU time says nothing about Thor); a roofline + launch-overhead model turns the
counts into an ESTIMATED Thor time in `summarise_opcount.py`.

Attribution:
  * forward ops -> the innermost active scope (module forward hooks at depth <= D, plus the named
    function wrappers from step_profiler.phase_targets);
  * backward ops -> the scope that CREATED the autograd node being executed
    (`torch._C._current_autograd_node()`), tagged by walking each scope's outputs' graph at scope
    exit (innermost first, so a node belongs to the innermost scope that produced it);
  * ops that run inside backward UNDER a module hook = gradient-checkpoint RECOMPUTE.
Views/metadata ops (no kernel) are excluded from the kernel count; `aten._local_scalar_dense`
(`.item()`, `int(t)`, `bool(t)`) and Python-level `.cpu()` / `.tolist()` / `.numpy()` are counted as
host syncs (on a CUDA device each one drains the queue).
"""
from __future__ import annotations

import functools
import json
import threading
from collections import defaultdict
from pathlib import Path

_TLS = threading.local()


def _stack():
    s = getattr(_TLS, "stack", None)
    if s is None:
        s = _TLS.stack = []
    return s


class State:
    def __init__(self):
        self.active = False
        self.in_backward = 0
        self.node_scope = {}                 # autograd Node -> scope
        self.tally = defaultdict(lambda: defaultdict(float))
        self.step_tallies = []
        self.cur_step = -1


ST = State()

_VIEWISH = {"view", "_unsafe_view", "expand", "permute", "transpose", "t", "slice", "select",
            "as_strided", "alias", "unsqueeze", "squeeze", "detach", "split", "split_with_sizes",
            "unbind", "chunk", "narrow", "diagonal", "unfold", "view_as_real", "view_as_complex",
            "_reshape_alias", "lift_fresh", "_to_copy_noop", "empty", "empty_strided",
            "empty_like", "sym_size", "sym_stride", "sym_numel", "size", "stride", "numel",
            "dim", "is_contiguous", "storage_offset", "sym_storage_offset", "_nested_tensor_size",
            "clone_noop", "set_", "resize_", "_has_compatible_shallow_copy_type", "is_same_size",
            "_version"}


def _nbytes(x):
    import torch
    n = 0
    if isinstance(x, torch.Tensor):
        try:
            return x.numel() * x.element_size()
        except Exception:                                   # noqa: BLE001
            return 0
    if isinstance(x, (list, tuple)):
        for y in x:
            n += _nbytes(y)
    elif isinstance(x, dict):
        for y in x.values():
            n += _nbytes(y)
    return n


def _scope_now():
    """(scope, phase). In a backward: grad mode ENABLED = gradient-checkpoint RECOMPUTE (torch's
    non-reentrant recompute runs under enable_grad; backward ops run with grad disabled), scoped by
    the innermost active module; otherwise a BACKWARD op, scoped by the scope that CREATED the
    autograd node being executed. Ops under the conflict probe carry its prefix."""
    import torch
    st = _stack()
    if ST.in_backward:
        pre = ""
        for s_ in st:
            if s_.startswith("fn:conflict"):
                pre = s_ + "|"
                break
        if torch.is_grad_enabled():
            return pre + (st[-1] if st else "<top>") + "@recompute", "recompute"
        try:
            node = torch._C._current_autograd_node()
        except Exception:                                   # noqa: BLE001
            node = None
        sc = ST.node_scope.get(node) if node is not None else None
        return pre + (sc or "<untagged>") + "@bwd", "bwd"
    return (st[-1] if st else "<top>"), "fwd"


def make_mode():
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils.flop_counter import flop_registry

    class OpCounter(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            out = func(*args, **kwargs)
            if not ST.active:
                return out
            name = func._overloadpacket.__name__
            scope, phase = _scope_now()
            t = ST.tally[scope]
            if name == "_local_scalar_dense":
                t["sync_item"] += 1
                return out
            if name in _VIEWISH or name.startswith("_foreach_") and False:
                t["views"] += 1
                return out
            t["kernels"] += 1
            if name.startswith("_foreach_"):
                t["foreach_kernels"] += 1
            fr = flop_registry.get(func._overloadpacket)
            if fr is not None:
                try:
                    t["flops"] += float(fr(*args, **kwargs, out_val=out))
                except Exception:                           # noqa: BLE001
                    t["flop_err"] += 1
            t["bytes"] += _nbytes(args) + _nbytes(out)
            t["op:" + name] += 1
            return out
    return OpCounter


def _tag_outputs(obj, scope, seen_limit=200000):
    import torch
    roots = []

    def collect(x):
        if isinstance(x, torch.Tensor):
            if x.grad_fn is not None:
                roots.append(x.grad_fn)
        elif isinstance(x, (list, tuple)):
            for y in x:
                collect(y)
        elif isinstance(x, dict):
            for y in x.values():
                collect(y)
    collect(obj)
    todo, n = list(roots), 0
    while todo and n < seen_limit:
        nd = todo.pop()
        if nd is None or nd in ST.node_scope:
            continue
        if type(nd).__name__ == "AccumulateGrad":
            continue
        ST.node_scope[nd] = scope
        n += 1
        for nxt, _ in nd.next_functions:
            if nxt is not None and nxt not in ST.node_scope:
                todo.append(nxt)


def scoped_fn(fn, scope):
    @functools.wraps(fn)
    def w(*a, **k):
        st = _stack()
        st.append(scope)
        try:
            r = fn(*a, **k)
        finally:
            st.pop()
        if ST.active and not ST.in_backward:
            _tag_outputs(r, scope)
        return r
    return w


def install_module_scopes(model, max_depth: int):
    import torch
    n = 0
    for name, mod in model.named_modules():
        if not name or name.count(".") > max_depth:
            continue
        if isinstance(mod, (torch.nn.ModuleList, torch.nn.ModuleDict)):
            continue

        def pre(m, inp, _n=name):
            st = _stack()
            m.__dict__.setdefault("_oc_depth", []).append(len(st))
            st.append("mod:" + _n)

        def post(m, inp, out, _n=name):
            st = _stack()
            d = m.__dict__.get("_oc_depth") or []
            n0 = d.pop() if d else max(len(st) - 1, 0)
            del st[n0:]                # also drops scopes an early-stopped recompute left open
            if ST.active and not ST.in_backward:
                _tag_outputs(out, "mod:" + _n)
        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)
        n += 1
    return n


def install(T, phase_targets, out: Path, hook_depth: int, measure_steps: set):
    import torch
    mode_cls = make_mode()
    for obj, attr, name, _s in phase_targets(T):
        f = getattr(obj, attr, None)
        if f is not None:
            setattr(obj, attr, scoped_fn(f, "fn:" + name))
    T.compute_losses_v3 = scoped_fn(T.compute_losses_v3, "phase:fwd_loss")
    orig_bw = torch.Tensor.backward

    def backward(self, *a, **k):
        ST.in_backward += 1
        n0 = len(_stack())
        try:
            return orig_bw(self, *a, **k)
        finally:
            ST.in_backward -= 1
            del _stack()[n0:]          # checkpoint EARLY-STOP aborts a recompute mid-module
    torch.Tensor.backward = backward
    orig_grad = torch.autograd.grad

    def agrad(*a, **k):
        ST.in_backward += 1
        n0 = len(_stack())
        try:
            return orig_grad(*a, **k)
        finally:
            ST.in_backward -= 1
            del _stack()[n0:]
    torch.autograd.grad = agrad

    def count_py(name):
        orig = getattr(torch.Tensor, name)

        @functools.wraps(orig)
        def w(self, *a, **k):
            if ST.active:
                sc, _ = _scope_now()
                ST.tally[sc]["sync_" + name] += 1
            return orig(self, *a, **k)
        setattr(torch.Tensor, name, w)
    for nm in ("cpu", "tolist", "numpy"):
        count_py(nm)

    orig_bo = T.build_optimizer

    def build_optimizer(model, args):
        install_module_scopes(model, hook_depth)
        return orig_bo(model, args)
    T.build_optimizer = build_optimizer

    mode = mode_cls()
    mode.__enter__()
    orig_next = T.next_train_batch

    def next_train_batch(it, dl, sampler, dpos):
        # close the previous step's tally
        if ST.active:
            ST.step_tallies.append({"step": ST.cur_step,
                                    "tally": {k: dict(v) for k, v in ST.tally.items()}})
            _dump(out)
        ST.tally = defaultdict(lambda: defaultdict(float))
        ST.node_scope = {}
        r = orig_next(it, dl, sampler, dpos)
        ST.cur_step += 1
        ST.active = ST.cur_step in measure_steps
        return r
    T.next_train_batch = next_train_batch

    def finish():
        if ST.active:
            ST.step_tallies.append({"step": ST.cur_step,
                                    "tally": {k: dict(v) for k, v in ST.tally.items()},
                                    "note": "final step: includes the end-of-run save/log"})
            ST.active = False
        try:
            mode.__exit__(None, None, None)
        except Exception:                                   # noqa: BLE001
            pass
        _dump(out)
    return finish


def _dump(out: Path):
    (out / "opcount_steps.json").write_text(json.dumps(ST.step_tallies, indent=1),
                                            encoding="utf-8")
