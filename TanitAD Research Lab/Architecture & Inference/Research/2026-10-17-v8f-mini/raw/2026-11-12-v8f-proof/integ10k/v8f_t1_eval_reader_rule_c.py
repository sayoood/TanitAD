#!/usr/bin/env python
"""v8f_t1_eval_reader_rule_c.py -- written by make_close10k.py from v8f_t1_eval_reader_rule.py for card PREREG_V8F_PROOF_ADD48 (CLOSE10K): two more forms,
RLc (labels: presence, gap and closing speed) and RRc (reader presence and gap through the map, closing speed from the labels where the label has one, else 0;
a diagnostic, not admissible at inference). The five forms of card ADD46 are unchanged. Header of the original:
v8f_t1_eval_reader_rule.py -- card PREREG_V8F_PROOF_ADD46 (INTEG10K-PRE), part 1: the study wrapper around the pinned T1-kin tool v8f_t1_eval.py of the tree of commit cdbff971.

  python v8f_t1_eval_reader_rule.py --tree TREE --form {R0,RLb,RR,RRid,RR0} [--reader CKPT --qmap JSON] [--rule-log NPZ] run <the argv of the pinned tool's `run`>

Forms of the card (the rule is the banked distance rule of cards ADD26/ADD33 in its form WITHOUT the closing speed, floor form cap-only as the arms of ADD33/ADD42, parameters unchanged):
  R0    nothing installed: the pinned tool's own run.
  RLb   the rule fed by labels (presence, gap from the label join; closing speed 0).
  RR    the rule fed by the reader: presence = reader probability >= 0.5 and a full 5-frame history; gap g_used = min(g_reader, Q(g_reader)), Q the monotone map (--qmap).
  RRid  as RR with the reader's gap replaced by 1e9 m (the rule is inactive above 60 m: the plan is returned unchanged).
  RR0   as RR with the reader's gap replaced by 0 m.
The reader reads batch["feats"][:, -5:], the cached encoder tokens of cache steps j-4 .. j at the read tick j (frames at or before the tick only), and nothing else; no model weight is read.
The pinned tool is imported from --tree by file path and is NOT edited; the rule code (dist_rule.py, v8f_t1_eval_rule.py, v8f_t1_eval_rule_mix.py of the distrule folder) is imported, not edited.
A side file `<rule-log minus .npz>_rd.npz` records the reader's presence, gap, gap used and the history flag row for row with the rule log.
Tier T1-kin (kinematic rollout on the model's own actions; the lead does not react); never driving performance.
STRATEGIC: not read by this wrapper (see note N1 of the card: T0 at the first tick, unchanged by construction for an inference-time rule).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FORMS = ("R0", "RLb", "RR", "RRid", "RR0", "RLc", "RRc")
P_MIN = 0.5
GAP_PLANT = {"RRid": 1e9, "RR0": 0.0}
PINNED_SHA256 = "61e140edb9da01acbbba4eef81e74bbb68b57e2ff31842e05958759446d97397"


def find_distrule():
    cands = [os.environ.get("DISTRULE_DIR", ""), os.path.join(HERE, "distrule"), os.path.join(HERE, "..", "distrule"), HERE]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "v8f_t1_eval_rule_mix.py")):
            return os.path.normpath(c)
    raise SystemExit("REFUSED: the distrule folder (v8f_t1_eval_rule_mix.py, dist_rule.py, v8f_t1_eval_rule.py) was not found; set DISTRULE_DIR")


for _p in (find_distrule(), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import qcal_map as QM  # noqa: E402
import v8f_t1_eval_rule_mix as X  # noqa: E402

W = X.W
R = W.R
_RULE_PLANNER = W.RulePlanner          # the original class, kept before main() points W.RulePlanner at ReaderPlanner
STATE = {"form": "R0", "reader": None, "xs": None, "ys": None, "cols": {k: [] for k in ("p", "g_read", "g_used", "ok")}}


# ---------------------------------------------------------------- the pure feed (tested without the tool)
def q_raw(g, xs, ys):
    """Q(g) of the monotone map without the min with g (the form of qcal_map.apply_map before its min); NaN stays NaN."""
    g = np.asarray(g, np.float64)
    out = np.array(g, copy=True)
    fin = np.isfinite(g)
    gg = g[fin]
    q = np.interp(gg, xs, ys)
    lo = gg < xs[0]
    q[lo] = ys[0] * gg[lo] / xs[0]
    hi = gg > xs[-1]
    q[hi] = gg[hi] - (xs[-1] - ys[-1])
    out[fin] = q
    return out


def feed(form, p, g, ok, xs=None, ys=None, p_min=P_MIN):
    """-> (present [B] bool, g_used [B] float64) of the reader forms. g is the reader's gap read; ok the full-history flag."""
    p = np.asarray(p, np.float64)
    g = np.asarray(g, np.float64)
    present = (p >= p_min) & np.asarray(ok, bool)
    if form in GAP_PLANT:
        used = np.full(g.shape, GAP_PLANT[form], np.float64)
    elif form in ("RR", "RRc"):
        used = np.minimum(g, q_raw(g, xs, ys))                    # g_used = min(g_reader, Q(g_reader)); NaN stays NaN
    else:
        raise ValueError("no reader feed for form " + form)
    return present, used


def head_values_reader(model, feats, frame_ok=None, hist=None):
    """Replaces W.head_values: p [B], gap [B], closing [B] (zero: the rule form without the closing speed), ok [B]. Takes tokens only; `model` is not read."""
    rd = STATE["reader"]
    f = feats[:, -rd.hist:]
    p, g = rd.read(f)
    ok = np.ones(len(p), bool) if frame_ok is None else frame_ok[:, -rd.hist:].all(-1).cpu().numpy()
    present, used = feed(STATE["form"], p, g, ok, STATE["xs"], STATE["ys"])
    for k, v in (("p", p), ("g_read", g), ("g_used", used), ("ok", ok)):
        STATE["cols"][k].append(np.asarray(v))
    # presence is encoded through p: the mix planner tests p >= W.HEAD_P_MIN and ok; a reader below 0.5 stays below
    return p, used.astype(np.float32), np.zeros(len(p), np.float32), ok


def head_values_none(model, feats, frame_ok=None, hist=None):
    n = len(feats)
    return np.zeros(n, np.float32), np.full(n, np.nan, np.float32), np.zeros(n, np.float32), np.ones(n, bool)


class ReaderPlanner(X.MixPlanner):
    """MixPlanner without the lead-head requirement: the reader uses no model weight."""

    def __init__(self, base, model, mode, m, fk, label=None, log=None, floor="card"):
        _RULE_PLANNER.__init__(self, base, model, mode, m, fk, label=label, log=log, floor=floor)
        if self.source == "mix":
            self.label = X.STATE["label"]


def arg_value(argv, flag):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--form", choices=FORMS, default=os.environ.get("INTEG_FORM", "R0"))
    ap.add_argument("--reader", default=os.environ.get("INTEG_READER", ""))
    ap.add_argument("--qmap", default=os.environ.get("INTEG_QMAP", ""))
    ap.add_argument("--rule-log", default="")
    mine, rest = ap.parse_known_args(argv)
    T, path = W.load_pinned(mine.tree)
    a = T.build_parser().parse_args(rest)
    if a.mode != "run":
        raise SystemExit("REFUSED: this wrapper applies to the `run` mode of the pinned tool only")
    pin = W.sha256_file(path)
    if pin != PINNED_SHA256:
        raise SystemExit("REFUSED: pinned tool sha256 %s != %s" % (pin, PINNED_SHA256))
    f = mine.form
    if f == "R0":
        if mine.rule_log or mine.reader or mine.qmap:
            raise SystemExit("REFUSED: R0 installs nothing; --rule-log/--reader/--qmap would write or read nothing")
        return T.cmd_run(a)
    if f in GAP_PLANT or f in ("RR", "RRc"):
        if not mine.reader or not os.path.isfile(mine.reader):
            raise SystemExit("REFUSED: form %s needs --reader CKPT" % f)
        if f in ("RR", "RRc"):
            if not mine.qmap or not os.path.isfile(mine.qmap):
                raise SystemExit("REFUSED: form RR needs --qmap JSON")
            STATE["xs"], STATE["ys"] = QM.load_map(mine.qmap)
            u = QM.unit_check(STATE["xs"], STATE["ys"])
            if not u["pass"]:
                raise SystemExit("REFUSED: the map fails its unit check")
        import tok_reader as TR
        STATE["reader"] = TR.TokReader.from_ckpt(mine.reader)
        W.head_values = head_values_reader
        mix = ("head", "head", "label" if f == "RRc" else "zero")
    elif f in ("RLb", "RLc"):
        W.head_values = head_values_none
        mix = ("label", "label", "label" if f == "RLc" else "zero")
    STATE["form"] = f
    W.HEAD_P_MIN = P_MIN
    X.MIX.update(presence=mix[0], gap=mix[1], closing=mix[2])
    W.MODE_CFG["mix"] = ("mix", {})
    W.RulePlanner = ReaderPlanner
    X.STATE["label"] = W.LabelSource(T.D.load_join(a.eval_join))
    log = W.RuleLog()
    made, restore = W.install(T, "mix", a, None, log, "cap-only")
    try:
        rc = T.cmd_run(a)
    finally:
        restore()
        if mine.rule_log:
            meta = {"mode": "mix", "source": "mix", "form": f, "mix": json.dumps(X.MIX, sort_keys=True), "mix_counts": json.dumps(X.COUNT, sort_keys=True), "floor": "cap-only",
                    "D": R.D_M, "TAU": R.TAU_S, "B_MAX": R.B_MAX, "G_MAX": R.G_MAX_M, "replan_steps": a.replan_steps, "pinned_path": path, "pinned_sha256": pin, "ckpt": a.ckpt, "tag": a.tag,
                    "eval_join": a.eval_join, "join_sha256": W.sha256_file(a.eval_join) if os.path.isfile(a.eval_join) else "", "reader": mine.reader,
                    "reader_sha256": W.sha256_file(mine.reader) if mine.reader else "", "qmap": mine.qmap, "qmap_sha256": W.sha256_file(mine.qmap) if mine.qmap else "", "mode_kwargs": "{}"}
            log.save(mine.rule_log, meta)
            print("wrote", mine.rule_log, flush=True)
            if STATE["cols"]["p"]:
                side = (mine.rule_log[:-4] if mine.rule_log.endswith(".npz") else mine.rule_log) + "_rd.npz"
                cols = {k: np.concatenate(v) for k, v in STATE["cols"].items()}
                with open(side, "wb") as fh:
                    np.savez(fh, **cols, form=np.asarray(f))
                print("wrote", side, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
