#!/usr/bin/env python3
"""v8f_t1_eval_rb.py -- PREREG_V8F_NAVRB2: the T1 evaluator for a checkpoint trained with the ROUNDABOUT channel ``nav_rb`` (``--nav-rb on``).

``v8f_t1_eval.py`` stays byte-equal to its pinned bytes (``TOOL_SHA256`` = the tool at a4357e79 / 4dedac45; a refusal when it differs; every patch undone in ``finally``). Added, in ONE call (the pattern of ``v8f_t1_eval_far.py``):

  1. ``run --nav-rb off|on`` (default off), REFUSED when it differs from the checkpoint's cfg ``nav_rb``; ``on`` needs ``--nav-rb-npz`` + ``--nav-rb-sha256`` (the table, its EVAL split, row-aligned, re-hashed); ``off`` forbids them.
     ``--nav-rb-variant follow|real|zero`` (default follow): ``follow`` moves TOGETHER with ``--nav-variant`` (real / donor / zero, the tool's donor rule: the donor window's rows ending at the donor's NOW);
     ``real`` keeps the channel real beside a moved nav; ``zero`` = the Z-EVAL / Z-USE control (the channel all 0.0, the nav as ``--nav-variant``).
  2. ``nav_rb`` rows <= t0 are the logged (or donor / zero) rows; rows AFTER t0 are HELD from t0 (the token rule: the roundabout state is a text-nav token with no own-arc term, so dead-reckoning keeps it as at NOW). A ``--nav deadreckon`` run
     never reads a logged ``nav_rb`` row after t0 (the item builder's logged read is overwritten); a ``--nav logged`` run keeps the logged rows. The batch carries the key ``nav_rb`` (``MODEL_KEYS`` and ``build_item`` patched).
  3. With ``--dump-nav`` and the channel on, the tick recorder adds ``nav_rb_in`` [N, 1] (the value fed at NOW).
With the flag off nothing of the tool is patched: the run is the tool's, with the same argv, bit for bit.

usage:  python v8f_t1_eval_rb.py run --ckpt CK ... --nav-rb on --nav-rb-npz T --nav-rb-sha256 S   (any other argv: the tool's)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import v8f_t1_eval as T  # noqa: E402

TOOL_SHA256 = "180044eb28246ba94a18366ed337f061ec659d6667496c0f88a4a325a0d65c93"      # = v8f_t1_eval_far.TOOL_SHA256 (the tool at a4357e79 / 4dedac45)
RB_KEY = "nav_rb"
RB_VARIANTS = ("follow", "real", "zero")


def tool_sha256(path=None):
    with open(path or T.__file__, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def add_flags(p):
    p.add_argument("--nav-rb", choices=("off", "on"), default="off", help="must equal the checkpoint's cfg nav_rb (a mismatch is refused)")
    p.add_argument("--nav-rb-npz", default="")
    p.add_argument("--nav-rb-sha256", default="")
    p.add_argument("--nav-rb-variant", choices=RB_VARIANTS, default="follow")


def build_parser(base=None):
    p = (base or T.build_parser)()
    sub = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction))
    add_flags(sub.choices["run"])
    return p


def refusals(a, cfg):
    bad = []
    if a.nav_rb != cfg.nav_rb:
        bad.append(f"--nav-rb {a.nav_rb} but the checkpoint's cfg has nav_rb={cfg.nav_rb!r}: the evaluator would silently skip (or invent) the roundabout channel")
    if a.nav_rb == "on" and not (a.nav_rb_npz and a.nav_rb_sha256):
        bad.append("--nav-rb on needs --nav-rb-npz and --nav-rb-sha256 (the eval split of the nav_rb table)")
    if a.nav_rb == "off" and (a.nav_rb_npz or a.nav_rb_sha256 or a.nav_rb_variant != "follow"):
        bad.append("--nav-rb off forbids --nav-rb-npz / --nav-rb-sha256 / --nav-rb-variant: they name a channel the checkpoint never saw")
    return bad


def rb_rows(J, ci, t0, variant, donor, run_variant):
    """The channel rows [n, 1] a window starts from: the tool's ``nav_rows_variant`` rule pointed at the nav_rb table (real / zero / donor). ``variant`` is the run's --nav-rb-variant, ``run_variant`` the run's --nav-variant."""
    v = run_variant if variant == "follow" else variant
    if v == "donor" and donor is None:
        raise ValueError("a donor nav_rb needs the run's donor windows")
    return T.nav_rows_variant({**J, "nav": J[RB_KEY]}, ci, t0, v, donor)


class RBWindowState(T.WindowState):
    """The tool's window state plus the nav_rb rows: rows <= t0 as given, rows after t0 HELD from t0 in dead-reckoning mode (a logged run keeps the given rows)."""

    def __init__(self, base, track, nav_rows, ci, t0, nav_mode="deadreckon", rb_rows_=None):
        super().__init__(base, track, nav_rows, ci, t0, nav_mode)
        self.nav_rb = None if rb_rows_ is None else np.array(rb_rows_, np.float32, copy=True)
        if self.nav_rb is not None:
            self.nav_rb0 = self.nav_rb[t0].copy()
            if nav_mode == "deadreckon":
                self.nav_rb[t0 + 1:] = self.nav_rb0


def make_states_factory(tool_make_states, a):
    def make_states(store, tracks, J, wins, nav_mode, variant="real", donors=None):
        if a.nav_rb != "on":
            return tool_make_states(store, tracks, J, wins, nav_mode, variant, donors)
        out = []
        for i, (ci, t0, _, _) in enumerate(wins):
            clip = str(J["clip"][ci])
            dn = None if donors is None else donors[i]
            rows = T.nav_rows_variant(J, ci, t0, variant, dn)
            out.append(RBWindowState(store.get(clip), tracks[clip], rows, ci, t0, nav_mode, rb_rows(J, ci, t0, a.nav_rb_variant, dn, variant)))
        return out
    return make_states


def build_item_factory(tool_build_item):
    def build_item(ds, proxy, st, tick, a_t, a_s, scene, limit, stats=None):
        it = tool_build_item(ds, proxy, st, tick, a_t, a_s, scene, limit, stats)
        if getattr(st, "nav_rb", None) is not None:
            idc = np.clip(np.arange(st.j - ds.hmax + 1, st.j + 1), 0, st.T - 1)
            it[RB_KEY] = st.nav_rb[idc].astype(np.float32)           # overwrites the item builder's logged read (which would see the logged rows after t0)
        return it
    return build_item


class RBTickRecorder(T.TickRecorder):
    """The tool's recorder, plus per tick ``nav_rb_in`` [1] with ``--dump-nav``."""

    rbs = False

    def __init__(self, planner, store, J, heads=False, nav=False, ctx=False):
        super().__init__(planner, store, J, heads=heads, nav=nav, ctx=ctx)
        self.rb_on = bool(nav) and type(self).rbs
        if self.rb_on:
            self.rows["nav_rb_in"] = []

    def __call__(self, n, batch, plan, states):
        super().__call__(n, batch, plan, states)
        if self.rb_on:
            for b in range(len(states)):
                self.rows["nav_rb_in"].append(batch[RB_KEY][b, -1].float().cpu().numpy())


def make_cmd_run(tool_cmd_run):
    def cmd_run(a):
        import torch
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        cfg = T.M.V8FMiniConfig(**ck["cfg"])
        bad = refusals(a, cfg)
        if bad:
            raise SystemExit("REFUSED: " + "; ".join(bad))
        if a.nav_rb == "off":
            return tool_cmd_run(a)
        keys0 = {"MODEL_KEYS": T.MODEL_KEYS, "TickRecorder": T.TickRecorder, "make_states": T.make_states, "build_item": T.build_item, "load_join": T.D.load_join}
        T.MODEL_KEYS = frozenset(T.MODEL_KEYS) | {RB_KEY}
        T.TickRecorder = type("RBRec", (RBTickRecorder,), {"rbs": True})
        T.make_states = make_states_factory(keys0["make_states"], a)
        T.build_item = build_item_factory(keys0["build_item"])

        def load_join(path):
            J = keys0["load_join"](path)
            J[RB_KEY], _ = T.D.nav_rb_arrays(J, a.nav_rb_npz, a.nav_rb_sha256, "eval", "real")
            return J
        T.D.load_join = load_join
        try:
            return tool_cmd_run(a)
        finally:
            for k, v in keys0.items():
                if k == "load_join":
                    T.D.load_join = v
                else:
                    setattr(T, k, v)
    return cmd_run


def main(argv=None):
    got = tool_sha256()
    if got != TOOL_SHA256:
        raise SystemExit(f"REFUSED: the T1 tool {T.__file__} has sha256 {got}, the pin is {TOOL_SHA256}: this wrapper patches MODEL_KEYS, TickRecorder, make_states and build_item of THAT tool")
    bp, cr = T.build_parser, T.cmd_run
    T.build_parser, T.cmd_run = (lambda: build_parser(bp)), make_cmd_run(cr)
    try:
        return T.main(argv)
    finally:
        T.build_parser, T.cmd_run = bp, cr


if __name__ == "__main__":
    sys.exit(main())
