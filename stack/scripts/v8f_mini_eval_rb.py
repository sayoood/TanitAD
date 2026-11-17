#!/usr/bin/env python3
"""v8f_mini_eval_rb.py -- PREREG_V8F_NAVRB2: the T0 evaluator (teacher-forced, STRATEGIC / detected-information cells) for a checkpoint trained with the ROUNDABOUT channel ``nav_rb``.

``v8f_mini_eval.py`` stays byte-equal to its pinned bytes (``TOOL_SHA256``; a refusal when it differs; the patched ``D.load_join`` is restored in ``finally``). The tool has no argparse hook, so this wrapper takes its own flags out of ``argv`` and passes
the rest to the tool unchanged:

  ``--nav-rb off|on`` (default off; REFUSED when it differs from the checkpoint's cfg ``nav_rb``), ``--nav-rb-npz`` / ``--nav-rb-sha256`` (the table's EVAL split, re-hashed; needed with ``on``, forbidden with ``off``),
  ``--nav-rb-variant follow|real|zero`` (default follow: ``--nav-withheld`` zeroes the channel with the nav; ``zero`` = the Z-EVAL / Z-USE control; ``real`` = the channel as logged).
At T0 the item builder reads the nav_rb rows of the window's own frames only (<= NOW), as it reads nav, so no row after NOW enters. ``--nav-shuffle`` (a donor nav) is REFUSED with ``--nav-rb-variant follow``: the tool swaps only the nav
array, so a donor nav beside an own channel is a mixed state; run the donor read at T1 (``v8f_t1_eval_rb.py --nav-variant donor``, which swaps both), or name ``--nav-rb-variant real|zero`` to say the channel stays as it is on purpose.
With ``--nav-rb off`` nothing of the tool is patched: the run is the tool's, with the same argv, bit for bit.
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

import v8f_mini_eval as E  # noqa: E402

TOOL_SHA256 = "008f48b79fd11ca553d118d8e29ece6aea29d1dd013653ac23b1dc7fe2dad558"      # v8f_mini_eval.py at 4dedac45
RB_VARIANTS = ("follow", "real", "zero")


def tool_sha256(path=None):
    with open(path or E.__file__, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def rb_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--nav-rb", choices=("off", "on"), default="off")
    p.add_argument("--nav-rb-npz", default="")
    p.add_argument("--nav-rb-sha256", default="")
    p.add_argument("--nav-rb-variant", choices=RB_VARIANTS, default="follow")
    return p


def refusals(r, cfg, a):
    bad = []
    if r.nav_rb != cfg.nav_rb:
        bad.append(f"--nav-rb {r.nav_rb} but the checkpoint's cfg has nav_rb={cfg.nav_rb!r}: the evaluator would silently skip (or invent) the roundabout channel")
    if r.nav_rb == "on" and not (r.nav_rb_npz and r.nav_rb_sha256):
        bad.append("--nav-rb on needs --nav-rb-npz and --nav-rb-sha256 (the eval split of the nav_rb table)")
    if r.nav_rb == "off" and (r.nav_rb_npz or r.nav_rb_sha256 or r.nav_rb_variant != "follow"):
        bad.append("--nav-rb off forbids --nav-rb-npz / --nav-rb-sha256 / --nav-rb-variant: they name a channel the checkpoint never saw")
    if r.nav_rb == "on" and a.nav_shuffle and r.nav_rb_variant == "follow":
        bad.append("--nav-shuffle with --nav-rb-variant follow: the tool swaps only the nav array, so the channel would stay the window's own beside a donor nav; use v8f_t1_eval_rb.py --nav-variant donor, or name --nav-rb-variant real|zero")
    return bad


def channel(J, r, a):
    """The eval-split channel [N, 1] for the run's variant: ``zero`` (or ``follow`` under --nav-withheld) = all 0.0; else as logged."""
    rb, _ = E.D.nav_rb_arrays(J, r.nav_rb_npz, r.nav_rb_sha256, "eval", "real")
    if r.nav_rb_variant == "zero" or (r.nav_rb_variant == "follow" and a.nav_withheld):
        rb = np.zeros_like(rb)
    return rb


def make_cmd_run(tool_cmd_run, r):
    def cmd_run(a):
        import torch
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        cfg = E.M.V8FMiniConfig(**ck["cfg"]) if hasattr(E, "M") else __import__("tanitad.models.v8f_mini", fromlist=["x"]).V8FMiniConfig(**ck["cfg"])
        bad = refusals(r, cfg, a)
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
    got = tool_sha256()
    if got != TOOL_SHA256:
        raise SystemExit(f"REFUSED: the T0 tool {E.__file__} has sha256 {got}, the pin is {TOOL_SHA256}: this wrapper patches D.load_join of THAT tool")
    r, rest = rb_parser().parse_known_args(sys.argv[1:] if argv is None else argv)
    cr = E.cmd_run
    E.cmd_run = make_cmd_run(cr, r)
    try:
        return E.main(rest)
    finally:
        E.cmd_run = cr


if __name__ == "__main__":
    sys.exit(main())
