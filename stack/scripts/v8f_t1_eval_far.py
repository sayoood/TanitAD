#!/usr/bin/env python3
"""v8f_t1_eval_far.py -- PREREG_V8F_CKPT120: the T1 evaluator with the far output ``ckpt_far`` in the tick dump and S's route-text channel ``nav_s`` dead-reckoned like the tactical channel.

``v8f_t1_eval.py`` stays byte-equal to its pinned bytes (``TOOL_SHA256`` = the pin of ``v8f_t1_eval_pool.py`` and ``v8f_t1_speedsub.py``; a refusal when it differs; every patch undone in ``finally``). Added, in ONE call:

  1. ``run --far-out off|on`` and ``--nav-strat off|on`` (default off), each REFUSED when it differs from the checkpoint's cfg (``s_far`` / ``nav_strat``); ``--nav-strat on`` needs ``--nav-strat-npz`` + ``--nav-strat-sha256`` (eval split,
     row-aligned, re-hashed); ``off`` forbids them. ``--nav-s-variant follow|real`` (default follow): nav_s moves TOGETHER with ``--nav-variant`` (real / donor / zero, the tool's donor rule); ``real`` keeps nav_s real beside a moved nav.
  2. ``nav_s`` is DEAD-RECKONED exactly like ``nav``: rows <= t0 are the logged (or donor / zero) rows; later rows are rebuilt from the OWN executed arc: ``row[3] = clip(row[3]_at_t0 - arc / 100, 0, 3)`` when ``row[4] > 0`` (the countdown),
     the token and valid flag held from t0. A logged-nav run keeps the logged rows. The window batch carries the key ``nav_s`` (``MODEL_KEYS`` and ``build_item`` patched), read by S alone.
  3. With ``--far-out on --dump-heads``, the tick recorder adds ``s_far`` [N, 3] (the far point RAW in S's tick frame, from the SAME ``forward_infer`` call that produced the plan) and its NOW-frame and window-origin-frame forms
     ``s_far_now`` / ``s_far_w0`` (``transform_ckpt``, as the tool does for the 30 / 50 / 80 m checkpoints); with ``--dump-nav`` and nav_s on, ``nav_s_in`` [N, 5] (the vector fed at NOW).
With both flags off nothing of the tool is patched: the run is the tool's, with the same argv, bit for bit.

usage:  python v8f_t1_eval_far.py run --ckpt CK ... --far-out on --dump-ticks DIR --dump-heads      (any other argv: the tool's)
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

TOOL_SHA256 = "180044eb28246ba94a18366ed337f061ec659d6667496c0f88a4a325a0d65c93"      # = v8f_t1_eval_pool.TOOL_SHA256 = v8f_t1_speedsub.TOOL_SHA256 (the tool at a4357e79)
FAR_KEY = "ckpt_far"
NAV_S_KEY = "nav_s"
NAV_S_VARIANTS = ("follow", "real")


def tool_sha256(path=None):
    with open(path or T.__file__, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def add_flags(p):
    p.add_argument("--far-out", choices=("off", "on"), default="off", help="must equal the checkpoint's cfg s_far (a mismatch is refused)")
    p.add_argument("--nav-strat", choices=("off", "on"), default="off", help="must equal the checkpoint's cfg nav_strat (a mismatch is refused)")
    p.add_argument("--nav-strat-npz", default="")
    p.add_argument("--nav-strat-sha256", default="")
    p.add_argument("--nav-s-variant", choices=NAV_S_VARIANTS, default="follow")


def build_parser(base=None):
    p = (base or T.build_parser)()
    sub = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction))
    add_flags(sub.choices["run"])
    return p


def refusals(a, cfg):
    bad = []
    if a.far_out != cfg.s_far:
        bad.append(f"--far-out {a.far_out} but the checkpoint's cfg has s_far={cfg.s_far!r}: the evaluator would silently skip (or invent) the far head")
    if a.nav_strat != cfg.nav_strat:
        bad.append(f"--nav-strat {a.nav_strat} but the checkpoint's cfg has nav_strat={cfg.nav_strat!r}: the evaluator would silently skip (or invent) the nav_s token")
    if a.nav_strat == "on" and not (a.nav_strat_npz and a.nav_strat_sha256):
        bad.append("--nav-strat on needs --nav-strat-npz and --nav-strat-sha256 (the eval split of the nav_s table)")
    if a.nav_strat == "off" and (a.nav_strat_npz or a.nav_strat_sha256 or a.nav_s_variant != "follow"):
        bad.append("--nav-strat off forbids --nav-strat-npz / --nav-strat-sha256 / --nav-s-variant: they name a channel the checkpoint never saw")
    return bad


class FarWindowState(T.WindowState):
    """The tool's window state plus the nav_s rows, dead-reckoned by the SAME rule as ``nav`` (``_finish`` of the tool, one line per row)."""

    def __init__(self, base, track, nav_rows, ci, t0, nav_mode="deadreckon", nav_s_rows=None):
        super().__init__(base, track, nav_rows, ci, t0, nav_mode)
        self.nav_s = None if nav_s_rows is None else np.array(nav_s_rows, np.float32, copy=True)
        if self.nav_s is not None:
            self.nav_s0 = self.nav_s[t0].copy()
            if nav_mode == "deadreckon":
                self.nav_s[t0 + 1:] = np.nan

    def _finish(self, j_old, j_new):
        super()._finish(j_old, j_new)                                   # the tool's: kinematics, arc, nav (self.arc[j] is set for every j in (j_old, j_new])
        if self.nav_s is not None and self.nav_mode == "deadreckon":
            for j in range(j_old + 1, j_new + 1):
                row = self.nav_s0.copy()
                if row[4] > 0:
                    row[3] = np.clip(row[3] - self.arc[j] / 100.0, 0.0, 3.0)
                self.nav_s[j] = row


def make_states_factory(tool_make_states, a, J_holder):
    """The tool's ``make_states`` with FarWindowState and the nav_s rows of the run's variant: real (the table), zero, donor (the tool's donor windows ``donors[i] = (dci, dt)``, rows ending at the donor's NOW)."""
    def make_states(store, tracks, J, wins, nav_mode, variant="real", donors=None):
        if a.nav_strat != "on":
            return tool_make_states(store, tracks, J, wins, nav_mode, variant, donors)
        v = "real" if a.nav_s_variant == "real" else variant
        Jn = {**J, "nav": J[NAV_S_KEY]}                                 # nav_rows_variant reads J["nav"]: pointed at the nav_s table for the nav_s rows
        out = []
        for i, (ci, t0, _, _) in enumerate(wins):
            clip = str(J["clip"][ci])
            rows = T.nav_rows_variant(J, ci, t0, variant, None if donors is None else donors[i])
            srows = T.nav_rows_variant(Jn, ci, t0, v, None if donors is None else donors[i])
            out.append(FarWindowState(store.get(clip), tracks[clip], rows, ci, t0, nav_mode, srows))
        return out
    return make_states


def build_item_factory(tool_build_item):
    def build_item(ds, proxy, st, tick, a_t, a_s, scene, limit, stats=None):
        it = tool_build_item(ds, proxy, st, tick, a_t, a_s, scene, limit, stats)
        if getattr(st, "nav_s", None) is not None:
            idc = np.clip(np.arange(st.j - ds.hmax + 1, st.j + 1), 0, st.T - 1)
            it[NAV_S_KEY] = st.nav_s[idc].astype(np.float32)
        return it
    return build_item


class FarTickRecorder(T.TickRecorder):
    """The tool's recorder, plus per tick: ``s_far`` (raw, S's tick frame) from the ``forward_infer`` call that produced the plan, and ``nav_s_in`` with ``--dump-nav``."""

    far = False
    navs = False

    def __init__(self, planner, store, J, heads=False, nav=False, ctx=False):
        super().__init__(planner, store, J, heads=heads, nav=nav, ctx=ctx)
        self.far_on = bool(heads) and type(self).far
        self.navs_on = bool(nav) and type(self).navs
        if self.far_on:
            self.rows["s_far"] = []
        if self.navs_on:
            self.rows["nav_s_in"] = []

    def __call__(self, n, batch, plan, states):
        super().__call__(n, batch, plan, states)
        for b in range(len(states)):
            if self.far_on:
                self.rows["s_far"].append(self.planner.heads[FAR_KEY][b])
            if self.navs_on:
                self.rows["nav_s_in"].append(batch[NAV_S_KEY][b, -1].float().cpu().numpy())

    def save(self, path, extra=None):
        if self.far_on and len(self.rows["s_far"]) and "s_ckpt" in self.rows:
            import torch
            far = torch.as_tensor(np.asarray(self.rows["s_far"]))[:, None, :]
            ps = torch.as_tensor(np.asarray(self.rows["s_pose"]))
            now = torch.as_tensor(np.asarray(self.rows["pose"])[:, :3].astype(np.float32))
            w0 = torch.as_tensor(np.asarray(self.rows["pose_w0"]).astype(np.float32))
            extra = dict(extra or {})
            extra["s_far_now"] = T.M.transform_ckpt(far, ps, now)[:, 0].numpy()
            extra["s_far_w0"] = T.M.transform_ckpt(far, ps, w0)[:, 0].numpy()
        return super().save(path, extra)


def make_cmd_run(tool_cmd_run):
    def cmd_run(a):
        import torch
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        cfg = T.M.V8FMiniConfig(**ck["cfg"])
        bad = refusals(a, cfg)
        if bad:
            raise SystemExit("REFUSED: " + "; ".join(bad))
        if a.far_out == "off" and a.nav_strat == "off":
            return tool_cmd_run(a)
        keys0 = {"HEAD_KEYS": T.HEAD_KEYS, "MODEL_KEYS": T.MODEL_KEYS, "TickRecorder": T.TickRecorder, "make_states": T.make_states, "build_item": T.build_item, "load_join": T.D.load_join}
        if a.far_out == "on":
            T.HEAD_KEYS = T.HEAD_KEYS + (FAR_KEY,)
        if a.nav_strat == "on":
            T.MODEL_KEYS = frozenset(T.MODEL_KEYS) | {NAV_S_KEY}
        rec = type("FarRec", (FarTickRecorder,), {"far": a.far_out == "on", "navs": a.nav_strat == "on"})
        T.TickRecorder = rec
        T.make_states = make_states_factory(keys0["make_states"], a, None)
        T.build_item = build_item_factory(keys0["build_item"])

        def load_join(path):
            J = keys0["load_join"](path)
            if a.nav_strat == "on":
                J[NAV_S_KEY], _ = T.D.nav_strat_arrays(J, a.nav_strat_npz, a.nav_strat_sha256, "eval", "real")
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
        raise SystemExit(f"REFUSED: the T1 tool {T.__file__} has sha256 {got}, the pin is {TOOL_SHA256}: this wrapper patches HEAD_KEYS, MODEL_KEYS, TickRecorder, make_states and build_item of THAT tool")
    bp, cr = T.build_parser, T.cmd_run
    T.build_parser, T.cmd_run = (lambda: build_parser(bp)), make_cmd_run(cr)
    try:
        return T.main(argv)
    finally:
        T.build_parser, T.cmd_run = bp, cr


if __name__ == "__main__":
    sys.exit(main())
