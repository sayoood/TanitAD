#!/usr/bin/env python3
"""v8f_t1_eval_rbd.py -- PREREG_V8F_NAVRBD: the T1 evaluator for a checkpoint trained with the 15-column ROUNDABOUT channel (``--nav-rb on --nav-rb-feat exitdist``). A study wrapper over ``v8f_t1_eval_rb.py`` (NAVRB2).

``v8f_t1_eval.py`` (sha256 180044eb) and ``v8f_mini_eval.py`` (sha256 008f48b7) stay byte-untouched (both pins are checked, a refusal when either differs). Added over the NAVRB2 wrapper:

  1. ``--nav-rb-feat state|exitdist`` (default state), REFUSED unless it equals the checkpoint's cfg ``nav_rb_feat`` (a cfg that dropped the feat reads ``state``, so a 15-column checkpoint with a dropped feat is refused) AND, when the
     checkpoint carries ``e_nav_rb.weight``, unless the weight's width equals the feat's width (G-GEN).
  2. ``--nav-rb-variant follow|real|zero|shuffle``: the three NAVRB2 variants plus ``shuffle`` (the loader's seeded clip-level derangement of the eval split, the deliberate-regression read).
  3. ``RBDWindowState``: in dead-reckoning mode c0-c12 and c14 are HELD from t0 (the NAVRB2 rule), and per tick ``c13 = clip(c13_t0 - arc[j] / 100, 0, 3)`` with the base class's own ``self.arc`` (the nav row's own rule,
     ``v8f_t1_eval.py`` ``_finish``: ``row[3] - arc / 100``). A ``--nav logged`` run keeps the logged rows. With a 1-column channel (feat state) nothing is changed: the NAVRB2 state.
With ``--nav-rb off`` nothing of the tool is patched: the run is the tool's, with the same argv, bit for bit.

usage:  python v8f_t1_eval_rbd.py run --ckpt CK ... --nav-rb on --nav-rb-feat exitdist --nav-rb-npz T --nav-rb-sha256 S   (any other argv: the tool's)
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

import v8f_t1_eval_rb as R  # noqa: E402

T = R.T
RB_KEY = R.RB_KEY
RBD_VARIANTS = ("follow", "real", "zero", "shuffle")
RB_FEATS = ("state", "exitdist")
FEAT_WIDTH = {"state": 1, "exitdist": 15}
C13 = 13                                                     # the distance column (dist / 100 in [0, 3]; card section 2)
MINI_EVAL_SHA256 = "008f48b79fd11ca553d118d8e29ece6aea29d1dd013653ac23b1dc7fe2dad558"


def add_flags(p):
    p.add_argument("--nav-rb-feat", choices=RB_FEATS, default="state", help="must equal the checkpoint's cfg nav_rb_feat (a mismatch is refused)")


def build_parser(base=None):
    p = R.build_parser(base)
    run = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction)).choices["run"]
    for act in run._actions:
        if act.dest == "nav_rb_variant":
            act.choices = RBD_VARIANTS
    add_flags(run)
    return p


def ckpt_width(ck):
    """The width of ``e_nav_rb.weight`` in the checkpoint, or None when it carries none."""
    for key in ("model", "state_dict"):
        sd = ck.get(key) if isinstance(ck, dict) else None
        if isinstance(sd, dict) and "e_nav_rb.weight" in sd:
            return int(sd["e_nav_rb.weight"].shape[1])
    return None


def refusals(a, cfg, ck=None):
    bad = list(R.refusals(a, cfg))
    cfg_feat = getattr(cfg, "nav_rb_feat", "state")
    if a.nav_rb_feat != cfg_feat:                                                              # the feat check (G-GEN)
        bad.append(f"--nav-rb-feat {a.nav_rb_feat} but the checkpoint's cfg has nav_rb_feat={cfg_feat!r}: the evaluator would feed a channel of the wrong width")
    w = None if ck is None else ckpt_width(ck)
    if w is not None and w != FEAT_WIDTH[a.nav_rb_feat]:
        bad.append(f"the checkpoint's e_nav_rb.weight is {w} columns wide but --nav-rb-feat {a.nav_rb_feat} means {FEAT_WIDTH[a.nav_rb_feat]}")
    return bad


class RBDWindowState(R.RBWindowState):
    """The NAVRB2 state (nav_rb rows after t0 HELD in dead-reckoning mode) plus the distance column dead-reckoned from the OWN executed arc."""

    def _finish(self, j_old, j_new):
        super()._finish(j_old, j_new)
        if self.nav_rb is None or self.nav_mode != "deadreckon" or self.nav_rb.shape[1] != FEAT_WIDTH["exitdist"]:
            return
        for j in range(j_old + 1, j_new + 1):
            self.nav_rb[j, C13] = np.float32(np.clip(self.nav_rb0[C13] - self.arc[j] / 100.0, 0.0, 3.0))


def make_states_factory(tool_make_states, a):
    def make_states(store, tracks, J, wins, nav_mode, variant="real", donors=None):
        if a.nav_rb != "on":
            return tool_make_states(store, tracks, J, wins, nav_mode, variant, donors)
        out = []
        rbv = "real" if a.nav_rb_variant == "shuffle" else a.nav_rb_variant                     # shuffle: J[nav_rb] already IS the derangement (load_join)
        for i, (ci, t0, _, _) in enumerate(wins):
            clip = str(J["clip"][ci])
            dn = None if donors is None else donors[i]
            rows = T.nav_rows_variant(J, ci, t0, variant, dn)
            out.append(RBDWindowState(store.get(clip), tracks[clip], rows, ci, t0, nav_mode, R.rb_rows(J, ci, t0, rbv, dn, variant)))
        return out
    return make_states


def make_cmd_run(tool_cmd_run):
    def cmd_run(a):
        import torch
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        cfg = T.M.V8FMiniConfig(**ck["cfg"])
        bad = refusals(a, cfg, ck)
        if bad:
            raise SystemExit("REFUSED: " + "; ".join(bad))
        if a.nav_rb == "off":
            return tool_cmd_run(a)
        keys0 = {"MODEL_KEYS": T.MODEL_KEYS, "TickRecorder": T.TickRecorder, "make_states": T.make_states, "build_item": T.build_item, "load_join": T.D.load_join}
        T.MODEL_KEYS = frozenset(T.MODEL_KEYS) | {RB_KEY}
        T.TickRecorder = type("RBRec", (R.RBTickRecorder,), {"rbs": True})
        T.make_states = make_states_factory(keys0["make_states"], a)
        T.build_item = R.build_item_factory(keys0["build_item"])

        def load_join(path):
            J = keys0["load_join"](path)
            J[RB_KEY], _ = T.D.nav_rb_arrays(J, a.nav_rb_npz, a.nav_rb_sha256, "eval", "shuffle" if a.nav_rb_variant == "shuffle" else "real", a.nav_rb_feat)
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
    got = R.tool_sha256()
    if got != R.TOOL_SHA256:
        raise SystemExit(f"REFUSED: the T1 tool {T.__file__} has sha256 {got}, the pin is {R.TOOL_SHA256}")
    bp, cr = T.build_parser, T.cmd_run
    T.build_parser, T.cmd_run = (lambda: build_parser(bp)), make_cmd_run(cr)
    try:
        return T.main(argv)
    finally:
        T.build_parser, T.cmd_run = bp, cr


if __name__ == "__main__":
    sys.exit(main())
