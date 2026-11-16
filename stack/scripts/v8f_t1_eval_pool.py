#!/usr/bin/env python3
"""v8f_t1_eval_pool.py -- PREREG_V8F_WEATHERHEAD form (a): the T1 evaluator with the pooled detected-information head in the tick dump.

``v8f_t1_eval.py`` stays byte-equal to its pinned bytes. ``v8f_t1_speedsub.py`` pins its sha256 (``TOOL_SHA256`` below), and an edit
of the shared tool regresses that study's suite: MEASURED 2026-11-16, 28 regressions on the pod G-SUITE while the flag lived in the
tool (all in ``tests/test_v8f_t1_speedsub.py``). This wrapper imports the tool unchanged and adds exactly three things:

  1. ``run --ctx-pool off|on`` (default off). It is REFUSED when it differs from the checkpoint's cfg: a flag that names a head the
     checkpoint lacks, or a cfg that carries a head the flag drops, would silently skip (or invent) the head.
  2. With ``--ctx-pool on``, ``ctx_pool_logits`` joins the head keys the planner keeps from the SAME ``forward_infer`` call that
     produced the plan (never a second call).
  3. With ``--ctx-pool on --dump-heads``, the tick recorder adds one ``ctx_pool_logits`` row [41] per tick.

With ``--ctx-pool off`` nothing of the tool is patched: the run is the tool's, with the same argv, bit for bit.
The patches name two of the tool's internals (``HEAD_KEYS`` and ``TickRecorder``), so the wrapper REFUSES to run when the imported
tool's sha256 differs from ``TOOL_SHA256``. Every patch is undone when the call returns (or raises).

usage:  python v8f_t1_eval_pool.py run --ckpt CK ... --ctx-pool on --dump-ticks DIR --dump-heads      (any other argv: the tool's)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import v8f_t1_eval as T  # noqa: E402

TOOL_SHA256 = "180044eb28246ba94a18366ed337f061ec659d6667496c0f88a4a325a0d65c93"      # = v8f_t1_speedsub.TOOL_SHA256 (the tool at a4357e79)
POOL_KEY = "ctx_pool_logits"
CTX_POOL_HELP = ("PREREG_V8F_WEATHERHEAD: on = the checkpoint carries the attention-pooled detected-information head and the dump writes "
                 "ctx_pool_logits; must equal the checkpoint's cfg (a mismatch is refused)")


def tool_sha256(path=None):
    with open(path or T.__file__, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class PoolTickRecorder(T.TickRecorder):
    """The tool's recorder, plus one row per tick: ``ctx_pool_logits`` of the ``forward_infer`` call that produced the tick's plan."""

    def __init__(self, planner, store, J, heads: bool = False, nav: bool = False, ctx: bool = False):
        super().__init__(planner, store, J, heads=heads, nav=nav, ctx=ctx)
        self.pool = bool(heads)                                           # rides with --dump-heads, as ctx_logits does in the tool
        if self.pool:
            self.rows[POOL_KEY] = []

    def __call__(self, n, batch, plan, states):
        super().__call__(n, batch, plan, states)
        if self.pool:
            hd = self.planner.heads
            for b in range(len(states)):
                self.rows[POOL_KEY].append(hd[POOL_KEY][b])


def build_parser(base=None):
    """The tool's parser, with ``--ctx-pool`` added to the ``run`` mode."""
    p = (base or T.build_parser)()
    sub = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction))
    sub.choices["run"].add_argument("--ctx-pool", choices=("off", "on"), default="off", help=CTX_POOL_HELP)
    return p


def checkpoint_ctx_pool(ckpt):
    """The ``ctx_pool`` of the checkpoint's cfg, read through the model's own config class (as the tool reads the cfg)."""
    import torch
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    return T.M.V8FMiniConfig(**ck["cfg"]).ctx_pool


def make_cmd_run(tool_cmd_run):
    def cmd_run(a):
        have = checkpoint_ctx_pool(a.ckpt)
        if a.ctx_pool != have:
            raise SystemExit(f"REFUSED: --ctx-pool {a.ctx_pool} but the checkpoint's cfg has ctx_pool={have!r}: the evaluator would silently skip (or invent) the pooled head")
        if a.ctx_pool != "on":
            return tool_cmd_run(a)
        keys, rec = T.HEAD_KEYS, T.TickRecorder
        T.HEAD_KEYS, T.TickRecorder = keys + (POOL_KEY,), PoolTickRecorder
        try:
            return tool_cmd_run(a)
        finally:
            T.HEAD_KEYS, T.TickRecorder = keys, rec
    return cmd_run


def main(argv=None):
    got = tool_sha256()
    if got != TOOL_SHA256:
        raise SystemExit(f"REFUSED: the T1 tool {T.__file__} has sha256 {got}, the pin is {TOOL_SHA256}: this wrapper patches HEAD_KEYS and TickRecorder of THAT tool")
    bp, cr = T.build_parser, T.cmd_run
    T.build_parser, T.cmd_run = (lambda: build_parser(bp)), make_cmd_run(cr)
    try:
        return T.main(argv)
    finally:
        T.build_parser, T.cmd_run = bp, cr


if __name__ == "__main__":
    sys.exit(main())
