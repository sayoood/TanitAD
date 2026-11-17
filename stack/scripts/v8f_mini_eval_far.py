#!/usr/bin/env python3
"""v8f_mini_eval_far.py -- PREREG_V8F_CKPT120: the T0 evaluator with the far output (S's 120 m checkpoint ``ckpt_far``) and S's route-text channel ``nav_s`` in the window batch.

``v8f_mini_eval.py`` stays byte-equal to its pinned bytes (``TOOL_SHA256`` below; the wrapper REFUSES to run when the imported tool differs, and a test pins the same). This wrapper imports the tool unchanged and adds, inside ONE call
(every patch is undone in ``finally``):

  1. ``run --far-out off|on`` and ``--nav-strat off|on`` (default off), each REFUSED when it disagrees with the checkpoint's cfg (``s_far`` / ``nav_strat``): a flag that names a head the checkpoint lacks, or a cfg that carries a head the
     flag drops, would silently skip (or invent) the output. ``--nav-strat on`` needs ``--nav-strat-npz`` + ``--nav-strat-sha256`` (the EVAL split, row-aligned to the eval join, re-hashed); ``off`` forbids them.
  2. ``nav_s`` joins the eval join (``D.load_join`` patched), so the window batch carries the key ``nav_s`` exactly as the trainer's loader does (``V8FMiniWindows.item``). Variants, moved TOGETHER with the tool's own nav variant by default
     (``--nav-s-variant follow``): ``--nav-shuffle`` => nav_s of the SAME donor window (``donor_indices`` over the run's window list, seed 0); ``--nav-withheld`` => nav_s all zeros (the trained no-row state). ``real`` keeps the real nav_s
     whatever the tool does to nav; ``zero`` zeroes nav_s alone.
  3. ``forward_infer`` is wrapped to keep ``ckpt_far`` [N, 3] (x, y, psi) and the fed ``nav_s`` row at NOW from the SAME call that produced the window's outputs (never a second call). Written to ``eval_<tag>_far.npz`` beside the tool's
     ``eval_<tag>.npz`` with the window index arrays and the variant records. With ``--far-out off --nav-strat off`` nothing of the tool is patched: the run is the tool's, bit for bit.

The far point is written RAW (S's tick frame, like the tool's ``ckpt``) as ``ckpt_far`` and re-expressed in the NOW frame by ``transform_ckpt`` from the batch's own ``pose_s`` / ``pose_now`` as ``ckpt_far_now`` (the two are equal with
``--ages fresh``; the STRAT1 / CKPT120 read is T0 at fresh ages).
Scope refusals (one forward per window is what makes the capture exact): ``--tac`` other than pred, ``--plan beta``.
usage:  python v8f_mini_eval_far.py run --ckpt CK ... --far-out on [--nav-strat on --nav-strat-npz T --nav-strat-sha256 S]      (any other argv: the tool's)
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
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402

TOOL_SHA256 = "008f48b79fd11ca553d118d8e29ece6aea29d1dd013653ac23b1dc7fe2dad558"      # v8f_mini_eval.py at a4357e79
NAV_S_VARIANTS = ("follow", "real", "zero")


def tool_sha256(path=None):
    with open(path or E.__file__, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def own_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--far-out", choices=("off", "on"), default="off")
    p.add_argument("--nav-strat", choices=("off", "on"), default="off")
    p.add_argument("--nav-strat-npz", default="")
    p.add_argument("--nav-strat-sha256", default="")
    p.add_argument("--nav-s-variant", choices=NAV_S_VARIANTS, default="follow")
    return p


def refusals(own, cfg, tool_a):
    """The problems of a (flags, checkpoint cfg, tool args) triple. [] = admissible."""
    bad = []
    if own.far_out != cfg.s_far:
        bad.append(f"--far-out {own.far_out} but the checkpoint's cfg has s_far={cfg.s_far!r}: the evaluator would silently skip (or invent) the far head")
    if own.nav_strat != cfg.nav_strat:
        bad.append(f"--nav-strat {own.nav_strat} but the checkpoint's cfg has nav_strat={cfg.nav_strat!r}: the evaluator would silently skip (or invent) the nav_s token")
    if own.nav_strat == "on" and not (own.nav_strat_npz and own.nav_strat_sha256):
        bad.append("--nav-strat on needs --nav-strat-npz and --nav-strat-sha256 (the eval split of the nav_s table)")
    if own.nav_strat == "off" and (own.nav_strat_npz or own.nav_strat_sha256 or own.nav_s_variant != "follow"):
        bad.append("--nav-strat off forbids --nav-strat-npz / --nav-strat-sha256 / --nav-s-variant: they name a channel the checkpoint never saw")
    if getattr(tool_a, "tac", "pred") != "pred" or getattr(tool_a, "plan", "") == "beta":
        bad.append("--tac shuffle|oracle|absmask and --plan beta call the model more than once per window; the far capture needs exactly one forward_infer per window")
    return bad


def nav_s_variant_of(own, tool_a):
    """The variant the run feeds: follow = the tool's own nav variant (zero when --nav-withheld, donor when --nav-shuffle, else real)."""
    if own.nav_s_variant != "follow":
        return own.nav_s_variant
    return "zero" if getattr(tool_a, "nav_withheld", False) else ("donor" if getattr(tool_a, "nav_shuffle", False) else "real")


def make_cmd_run(tool_cmd_run, own):
    def cmd_run(a):
        import torch
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        cfg = M.V8FMiniConfig(**ck["cfg"])
        bad = refusals(own, cfg, a)
        if bad:
            raise SystemExit("REFUSED: " + "; ".join(bad))
        if own.far_out == "off" and own.nav_strat == "off":
            return tool_cmd_run(a)
        variant = nav_s_variant_of(own, a)
        rows = {"far": [], "nav_s_now": [], "pose_s": [], "pose_now": []}
        state = {"windows": None}
        orig = {"load_join": D.load_join, "item": D.V8FMiniWindows.item, "init": D.V8FMiniWindows.__init__, "fi": M.V8FMini.forward_infer}

        def load_join(path):
            J = orig["load_join"](path)
            if own.nav_strat == "on" and "nav_s" not in J:
                ns, rec = D.nav_strat_arrays(J, own.nav_strat_npz, own.nav_strat_sha256, "eval", "real")
                J["nav_s"] = np.zeros_like(ns) if variant == "zero" else ns
                state["rec"] = rec
            return J

        def init(self, *args, **kw):
            orig["init"](self, *args, **kw)
            state["windows"] = self.windows
            self._far_perm = None

        def item(self, ci, t, a_t, a_s):
            it = orig["item"](self, ci, t, a_t, a_s)
            if own.nav_strat == "on" and variant == "donor":
                if self._far_perm is None:                              # the tool's donor rule: episode-disjoint, seed 0, over the run's window list
                    from tanitad.models.hjepa import donor_indices
                    w = self.windows
                    self._far_perm = donor_indices(len(w), episode_ids=torch.as_tensor(np.asarray([x[0] for x in w])), seed=0).numpy()
                    self._far_pos = {(int(x[0]), int(x[1])): i for i, x in enumerate(w)}
                i = self._far_pos[(int(ci), int(t))]
                dci, dt_ = self.windows[int(self._far_perm[i])][0], self.windows[int(self._far_perm[i])][1]
                it["nav_s"] = orig["item"](self, dci, dt_, a_t, a_s)["nav_s"]
            return it

        def forward_infer(self, batch):
            out = orig["fi"](self, batch)
            if own.far_out == "on":
                rows["far"].append(out["ckpt_far"].float().cpu().numpy())
                rows["pose_s"].append(batch["pose_s"].float().cpu()), rows["pose_now"].append(batch["pose_now"].float().cpu())
            if own.nav_strat == "on":
                rows["nav_s_now"].append(batch["nav_s"][:, -1].float().cpu().numpy())
            return out
        D.load_join, D.V8FMiniWindows.__init__, D.V8FMiniWindows.item, M.V8FMini.forward_infer = load_join, init, item, forward_infer
        try:
            rc = tool_cmd_run(a)
        finally:
            D.load_join, D.V8FMiniWindows.__init__, D.V8FMiniWindows.item, M.V8FMini.forward_infer = orig["load_join"], orig["init"], orig["item"], orig["fi"]
        import torch
        w = state["windows"]
        out = {"win_ci": np.asarray([x[0] for x in w], np.int64), "win_t": np.asarray([x[1] for x in w], np.int64), "far_out": np.asarray(own.far_out), "nav_strat": np.asarray(own.nav_strat), "nav_s_variant": np.asarray(variant)}
        if own.far_out == "on":
            out["ckpt_far"] = np.concatenate(rows["far"])
            far_t = torch.as_tensor(out["ckpt_far"])[:, None, :]
            out["ckpt_far_now"] = M.transform_ckpt(far_t, torch.cat(rows["pose_s"]), torch.cat(rows["pose_now"]))[:, 0].numpy()
            if len(out["ckpt_far"]) != len(w):
                raise SystemExit(f"REFUSED: {len(out['ckpt_far'])} far rows for {len(w)} windows: the capture is not one forward per window")
        if own.nav_strat == "on":
            out["nav_s_now"] = np.concatenate(rows["nav_s_now"])
            out["nav_strat_sha256"] = np.asarray(own.nav_strat_sha256)
        os.makedirs(a.out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(a.out_dir, f"eval_{a.tag}_far.npz"), **out)
        return rc
    return cmd_run


def main(argv=None):
    got = tool_sha256()
    if got != TOOL_SHA256:
        raise SystemExit(f"REFUSED: the T0 tool {E.__file__} has sha256 {got}, the pin is {TOOL_SHA256}: this wrapper patches the loader and forward_infer of THAT tool's run")
    own, rest = own_parser().parse_known_args(argv)
    cr = E.cmd_run
    E.cmd_run = make_cmd_run(cr, own)
    try:
        return E.main(rest)
    finally:
        E.cmd_run = cr


if __name__ == "__main__":
    sys.exit(main())
