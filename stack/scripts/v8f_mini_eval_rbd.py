#!/usr/bin/env python3
"""v8f_mini_eval_rbd.py -- PREREG_V8F_NAVRBD: the T0 evaluator (teacher-forced, STRATEGIC / detected-information cells) for a checkpoint trained with the 15-column ROUNDABOUT channel. The counterpart of
``v8f_mini_eval_rb.py`` (NAVRB2); ``v8f_mini_eval.py`` stays byte-equal to its pinned bytes (sha256 008f48b7; a refusal when it differs; the patched ``D.load_join`` is restored in ``finally``).

The tool has no argparse hook, so this wrapper takes its own flags out of ``argv`` and passes the rest to the tool unchanged:

  ``--nav-rb off|on`` (default off; REFUSED when it differs from the checkpoint's cfg ``nav_rb``), ``--nav-rb-feat state|exitdist`` (default state; REFUSED unless it equals the checkpoint's cfg ``nav_rb_feat`` and the width of
  its ``e_nav_rb.weight``), ``--nav-rb-npz`` / ``--nav-rb-sha256`` (the table's EVAL split, re-hashed; needed with ``on``, forbidden with ``off``),
  ``--nav-rb-variant follow|real|zero|shuffle`` (default follow: ``--nav-withheld`` zeroes the channel with the nav; ``zero`` = the Z-EVAL / Z-USE control; ``real`` = as logged; ``shuffle`` = the loader's clip-level derangement).
At T0 the item builder reads the channel rows of the window's own frames only (<= NOW), so no row after NOW enters and there is no dead reckoning. ``--nav-shuffle`` (a donor nav) is REFUSED with ``--nav-rb-variant follow``
(as in the NAVRB2 wrapper). With ``--nav-rb off`` nothing of the tool is patched.
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

import v8f_mini_eval_rb as R0  # noqa: E402

E = R0.E
RBD_VARIANTS = ("follow", "real", "zero", "shuffle")
RB_FEATS = ("state", "exitdist")
FEAT_WIDTH = {"state": 1, "exitdist": 15}


def rb_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--nav-rb", choices=("off", "on"), default="off")
    p.add_argument("--nav-rb-feat", choices=RB_FEATS, default="state")
    p.add_argument("--nav-rb-npz", default="")
    p.add_argument("--nav-rb-sha256", default="")
    p.add_argument("--nav-rb-variant", choices=RBD_VARIANTS, default="follow")
    return p


def ckpt_width(ck):
    for key in ("model", "state_dict"):
        sd = ck.get(key) if isinstance(ck, dict) else None
        if isinstance(sd, dict) and "e_nav_rb.weight" in sd:
            return int(sd["e_nav_rb.weight"].shape[1])
    return None


def refusals(r, cfg, a, ck=None):
    bad = list(R0.refusals(r, cfg, a))
    cfg_feat = getattr(cfg, "nav_rb_feat", "state")
    if r.nav_rb_feat != cfg_feat:                                                              # the feat check (G-GEN)
        bad.append(f"--nav-rb-feat {r.nav_rb_feat} but the checkpoint's cfg has nav_rb_feat={cfg_feat!r}: the evaluator would feed a channel of the wrong width")
    w = None if ck is None else ckpt_width(ck)
    if w is not None and w != FEAT_WIDTH[r.nav_rb_feat]:
        bad.append(f"the checkpoint's e_nav_rb.weight is {w} columns wide but --nav-rb-feat {r.nav_rb_feat} means {FEAT_WIDTH[r.nav_rb_feat]}")
    return bad


def channel(J, r, a):
    """The eval-split channel [N, W]: ``zero`` (or ``follow`` under --nav-withheld) = all 0.0; ``shuffle`` = the derangement; else as logged."""
    rb, _ = E.D.nav_rb_arrays(J, r.nav_rb_npz, r.nav_rb_sha256, "eval", "shuffle" if r.nav_rb_variant == "shuffle" else "real", r.nav_rb_feat)
    if r.nav_rb_variant == "zero" or (r.nav_rb_variant == "follow" and a.nav_withheld):
        rb = np.zeros_like(rb)
    return rb


def make_cmd_run(tool_cmd_run, r):
    def cmd_run(a):
        import torch
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        cfg = __import__("tanitad.models.v8f_mini", fromlist=["x"]).V8FMiniConfig(**ck["cfg"])
        bad = refusals(r, cfg, a, ck)
        if bad:
            raise SystemExit("REFUSED: " + "; ".join(bad))
        if r.nav_rb == "off":
            return tool_cmd_run(a)
        load0 = E.D.load_join

        def load_join(path):
            J = load0(path)
            J["nav_rb"] = channel(J, r, a)
            return J
        E.D.load_join = load_join
        try:
            return tool_cmd_run(a)
        finally:
            E.D.load_join = load0
    return cmd_run


def main(argv=None):
    got = R0.tool_sha256()
    if got != R0.TOOL_SHA256:
        raise SystemExit(f"REFUSED: the T0 tool {E.__file__} has sha256 {got}, the pin is {R0.TOOL_SHA256}")
    r, rest = rb_parser().parse_known_args(sys.argv[1:] if argv is None else argv)
    cr = E.cmd_run
    E.cmd_run = make_cmd_run(cr, r)
    try:
        return E.main(rest)
    finally:
        E.cmd_run = cr


if __name__ == "__main__":
    sys.exit(main())
