#!/usr/bin/env python3
"""v8f_t1_eval_reader_rule_rzgfhfc.py -- card PREREG_V8F_PROOF_ADD144 (RULEFC10K): the wrapper rzgfh with the lead forecast of the card (FC).

Tier T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance).

  python v8f_t1_eval_reader_rule_rzgfhfc.py [--lead-forecast {cv,ca}] [--fc-zero] [--fc-log NPZ] <every argument of v8f_t1_eval_reader_rule_rzgfh.py>

--lead-forecast cv (default) installs nothing: the run is the run of rzgfh, bit for bit.
--lead-forecast ca: every call of dist_rule.distance_rule (the rule at the tick, X.MixPlanner, and every later look, rzgfh.mid_looks) is routed to
dist_rule_fc.distance_rule with the lead's measured change of speed a_L of the card section 2:
  a_L(f) = (v_L(f) - v_L(f - 5)) / (5 dt),   v_L(x) = max(0, logged own speed at x + label closing at x)   (the label form's v_L),
f the logged frame of the look (j at the tick, j + q at a later look; q = m_tick - m of the call). a_L = 0 where the label has no lead at f - 5, where
f - 5 is outside the clip, or where a value is not finite. The label reads only rows at or before f (LabelSource.read). The routed call asserts that its
own v_L(f) equals the caller's v_L on every window where the caller has a lead (a frame mismatch stops the run).
--fc-zero: a_L forced to 0 (control r1 of the card: the run must equal the run of record).
--fc-log NPZ: one row per window and call (ci, f, q, present, a_L, v_now, v_prev, ok_prev).
Rule forms RLb and RLc only (the label forms of the card). rzgfh, dist_rule.py and the pinned tool are imported and not edited.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FC_K = 5                                               # card section 2: the change of speed over the last 1.0 s = 5 steps of 0.2 s
STATE = {"on": False, "zero": False, "ctx": None, "orig": None, "M": 10,
         "log": {k: [] for k in ("ci", "f", "q", "present", "a_L", "v_now", "v_prev", "ok_prev")}}


def find_rzgfh():
    for c in (os.environ.get("RZGFH_DIR", ""), HERE, os.environ.get("DISTRULE_DIR", "")):
        if c and os.path.isfile(os.path.join(c, "v8f_t1_eval_reader_rule_rzgfh.py")):
            return os.path.normpath(c)
    raise SystemExit("REFUSED: v8f_t1_eval_reader_rule_rzgfh.py not found (looked in RZGFH_DIR, this folder, DISTRULE_DIR)")


def load_rzgfh():
    d = find_rzgfh()
    if d not in sys.path:
        sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location("v8f_t1_eval_reader_rule_rzgfh", os.path.join(d, "v8f_t1_eval_reader_rule_rzgfh.py"))
    Z = importlib.util.module_from_spec(spec)
    sys.modules["v8f_t1_eval_reader_rule_rzgfh"] = Z
    spec.loader.exec_module(Z)
    return Z


def find_fc():
    for c in (HERE, os.environ.get("DISTRULE_DIR", "")):
        if c and os.path.isfile(os.path.join(c, "dist_rule_fc.py")):
            if c not in sys.path:
                sys.path.insert(0, c)
            import dist_rule_fc as F
            return F
    raise SystemExit("REFUSED: dist_rule_fc.py not found beside this wrapper")


# ---------------------------------------------------------------- the lead's measured change of speed (pure; tested without the tool)
def v_label(label, logged_v, ci, x):
    """v_L at logged frame x of clip ci as the label form builds it: (ok, max(0, logged speed + label closing)). ok is False without a lead or a finite value."""
    if x < 0 or x >= len(logged_v):
        return False, float("nan")
    pres, g, c = label.read(int(ci), int(x))
    v = float(logged_v[int(x)])
    if not pres or not (np.isfinite(g) and np.isfinite(c) and np.isfinite(v)):
        return False, float("nan")
    return True, max(0.0, v + float(c))


def lead_accel(label, logged, ci, f, dt=0.2, k=FC_K):
    """-> a_L [B], v_now [B], v_prev [B], ok_now [B], ok_prev [B]. a_L = (v_now - v_prev) / (k dt) where both are ok, else 0. Only rows at or before f are read."""
    B = len(ci)
    a, vn, vp = np.zeros(B), np.full(B, np.nan), np.full(B, np.nan)
    okn, okp = np.zeros(B, bool), np.zeros(B, bool)
    for b in range(B):
        okn[b], vn[b] = v_label(label, logged[b], ci[b], f[b])
        okp[b], vp[b] = v_label(label, logged[b], ci[b], f[b] - k)
        if okn[b] and okp[b]:
            a[b] = (vn[b] - vp[b]) / (k * dt)
    return a, vn, vp, okn, okp


def routed(F, label):
    """The replacement of dist_rule.distance_rule while --lead-forecast ca is on."""
    def distance_rule(plan, v0, present, g_src, g0, v_L, m=10, **kw):
        ctx = STATE["ctx"]
        if ctx is None:
            raise RuntimeError("FC: distance_rule called outside a bound planner call")
        ci, j, logged = ctx
        q = int(STATE["M"]) - int(m)
        if not (0 <= q < int(STATE["M"])):
            raise RuntimeError("FC: look q=%d outside the replan interval" % q)
        f = j + q
        a_L, vn, vp, okn, okp = lead_accel(label, logged, ci, f)
        pres = np.asarray(present, bool)
        vL = np.asarray(v_L, np.float64)
        if (pres & ~okn).any() or not np.array_equal(vn[pres], vL[pres]):
            raise RuntimeError("FC: v_L of the look differs from the caller's v_L (frame or source mismatch) at q=%d" % q)
        if STATE["zero"]:
            a_L = np.zeros_like(a_L)
        lg = STATE["log"]
        for key, val in (("ci", ci), ("f", f), ("q", np.full(len(ci), q, np.int64)), ("present", pres), ("a_L", a_L), ("v_now", vn), ("v_prev", vp), ("ok_prev", okp)):
            lg[key].append(np.asarray(val))
        return F.distance_rule(plan, v0, present, g_src, g0, v_L, m=m, a_L=a_L, **kw)
    return distance_rule


def save_fc_log(path):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in STATE["log"].items()}
    with open(path, "wb") as fh:
        np.savez(fh, **cols, fc_k=np.asarray(FC_K), zero=np.asarray(bool(STATE["zero"])), M=np.asarray(int(STATE["M"])))
    print("wrote", path, flush=True)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--lead-forecast", choices=("cv", "ca"), default="cv")
    ap.add_argument("--fc-zero", action="store_true")
    ap.add_argument("--fc-log", default="")
    mine, rest = ap.parse_known_args(argv)
    Z = load_rzgfh()
    if mine.lead_forecast == "cv":
        if mine.fc_zero or mine.fc_log:
            raise SystemExit("REFUSED: --fc-zero / --fc-log go with --lead-forecast ca")
        return Z.main(rest)
    if not mine.fc_log:
        raise SystemExit("REFUSED: --lead-forecast ca needs --fc-log (the card logs a_L)")
    form = Z.arg_value(rest, "--form")
    if form not in ("RLb", "RLc"):
        raise SystemExit("REFUSED: --lead-forecast ca is built for the label forms RLb and RLc only")
    rp = Z.arg_value(rest, "--replan-steps")
    STATE["M"] = int(rp) if rp is not None else 10
    F = find_fc()
    R = Z.R
    X = Z.X
    STATE["zero"] = bool(mine.fc_zero)
    parent = Z.ReaderPlanner

    class FCPlanner(parent):
        def __call__(self, batch):
            st = self.states
            if st is None:
                raise RuntimeError("FCPlanner not bound to the states of this batch")
            STATE["ctx"] = (np.array([s.ci for s in st], np.int64), np.array([s.j for s in st], np.int64), [s.logged_v for s in st])
            try:
                return parent.__call__(self, batch)
            finally:
                STATE["ctx"] = None

    orig_dr = R.distance_rule
    Z.ReaderPlanner = FCPlanner

    class LazyLabel:                                   # X.STATE["label"] is set inside Z.main, after this point
        def read(self, ci, j):
            return X.STATE["label"].read(ci, j)

    R.distance_rule = routed(F, LazyLabel())
    STATE["on"] = True
    try:
        return Z.main(rest)
    finally:
        R.distance_rule = orig_dr
        Z.ReaderPlanner = parent
        STATE["on"] = False
        save_fc_log(mine.fc_log)


if __name__ == "__main__":
    sys.exit(main())
