"""Shared test helpers of the INTEG-PRE tools: the path of the sources under test (IP_SRC, default this folder) and a builder of a synthetic world for read_ip1.py (numpy only)."""
import copy
import hashlib
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("IP_SRC") or HERE
if SRC not in sys.path:
    sys.path.insert(0, SRC)

C, K = "RLAm", "RLFT"
CODE_FIT_TOOL = "57e1242713e0b42eab05cf142b8d152fbb87db94354290ddd3f275f06292036b"


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def cellj(d_s, lo, hi):
    D = float(np.mean(d_s))
    return {"d_s": list(d_s), "D": D, "A_lo": lo, "A_hi": hi, "A_excludes_0": bool(lo > 0 or hi < 0)}


def pairs_all():
    import read_ip1 as R
    return R.pairs_of_card()


# default planted differences (U, per base 0..3); a card-conforming world: the cut, the share, no interference
DEFAULT = {
    ("IRQ_RLAm", "IR0_RLFT"): ([-0.06, -0.05, -0.07, -0.04], -0.08, -0.03),
    ("IRQ_RLAm", "IR0_RLAm"): ([-0.05, -0.05, -0.06, -0.03], -0.07, -0.03),
    ("IRQ_RLAm", "IRL_RLAm"): ([0.01, 0.0, 0.01, 0.02], -0.01, 0.03),
    ("IRQ_RLAm", "IRQ_RLFT"): ([0.0, -0.01, 0.01, 0.0], -0.02, 0.02),
    ("IR0_RLAm", "IR0_RLFT"): ([-0.01, 0.0, 0.0, -0.01], -0.03, 0.01),
    ("IRL_RLAm", "IR0_RLAm"): ([-0.08, -0.07, -0.09, -0.06], -0.10, -0.05),
    ("IRL_RLFT", "IR0_RLFT"): ([-0.08, -0.07, -0.09, -0.06], -0.10, -0.05),
    ("IR0_RLAm", "IPox_RLAm"): ([0.0, 0.0, 0.0, 0.0], -1e-12, 1e-12),
    ("IR0_RLFT", "IPox_RLFT"): ([0.0, 0.0, 0.0, 0.0], -1e-12, 1e-12),
}


def build(tmp, bases=(0, 1, 2, 3), over=None, cost=None, h5_false=(), c1_break=None, c3_ok=True, hw_all_pass=True, q3_all_pass=True, counts=(2399, 642, 56), hw_missing=None):
    """Writes small/, pull/, plan input and the two reader results under tmp; returns the paths. over = {(a, b): (d_s, lo, hi)} replaces planted cells."""
    tmp = str(tmp)
    small, pull = os.path.join(tmp, "small"), os.path.join(tmp, "pull")
    os.makedirs(small, exist_ok=True)
    os.makedirs(pull, exist_ok=True)
    cells = dict(DEFAULT)
    cells.update(over or {})
    all_bases = (0, 1, 2, 3)
    arms_txt = os.path.join(tmp, "arms.txt")
    with io.open(arms_txt, "w", encoding="utf-8") as fh:
        for k in all_bases:
            for a in (C, K):
                fh.write("IRQ_%s_s%d q %s_s%d %d 0.5\n" % (a, k, a, k, k))
    for k in bases:
        for a in (C, K):
            b = "%s_s%d" % (a, k)
            headbytes = ("head-%s" % b).encode()
            open(os.path.join(small, "heads_%s.qhead.npz" % b), "wb").write(headbytes)
            ck = sha_bytes(("ckpt-%s" % b).encode())
            rec_ck = ck if c1_break != ("ckpt", b) else sha_bytes(b"another checkpoint")
            hs = sha_bytes(headbytes)
            rec = {"card": "PREREG_V8F_PROOF_ADD42_N1", "arm": b, "clip_sets": {"check": [3810, 4572], "fit": [0, 3048], "select": [3048, 3810]}, H5_KEY: b not in h5_false,
                   "coverage_check": {"share_q_above_label": 0.19}, "head_file_sha256": {"/x/%s.qhead.npz" % b: hs}, "states_record": {"ckpt_sha256": rec_ck},
                   "tool_sha256": {"qhead_fit_n1.py": CODE_FIT_TOOL if c1_break != ("form", b) else "0" * 64}}
            json.dump(rec, open(os.path.join(small, "heads_%s.qhead.json" % b), "w"))
            if b in h5_false:
                open(os.path.join(small, "run_MISSING_%s.txt" % b), "w").write("MISSING\n")
            for t in ("IR0", "IRL", "IRQ"):
                d = os.path.join(pull, "%s_%s" % (t, b))
                os.makedirs(os.path.join(d, "ticks"), exist_ok=True)
                open(os.path.join(d, "PULLED.txt"), "w").write("tar_sha256 %s ckpt_sha256 %s pulled x\n" % ("a" * 64, ck))
            run_sha = hs if c1_break != ("runhead", b) else sha_bytes(b"head of another base")
            np.savez(os.path.join(pull, "IRQ_%s" % b, "ticks", "rule_log_IRQ_%s.npz" % b), qhead_sha256=np.asarray(run_sha), mode=np.asarray("head"))
    with io.open(os.path.join(small, "run_integpre_queue.log"), "w", encoding="utf-8") as fh:
        if c3_ok == "lie":
            fh.write("[t] integpre: c3 the identity form (q_head = g_head, P 0.5) against the head-fed rule on RLAm_s0: CMP files=141 differing=3 \n[t] integpre: control c3 passed\n")
        elif c3_ok:
            fh.write("[t] integpre: c3 the identity form (q_head = g_head, P 0.5) against the head-fed rule on RLAm_s0: CMP files=141 differing=0 \n[t] integpre: control c3 passed\n")
        else:
            fh.write("[t] integpre: c3 the identity form (q_head = g_head, P 0.5) against the head-fed rule on RLAm_s0: CMP files=141 differing=3 \n")
    pairs = {}
    q3pairs = {}
    for (a, b), (d_s, lo, hi) in cells.items():
        pk = "%s_minus_%s" % (a, b)
        d_s = list(d_s)[:len(bases)]
        pairs[pk] = {"cells": {"U": cellj(d_s, lo, hi), "C1": cellj([0.01] * len(d_s), -0.01, 0.02)}, "seeds_read": list(bases)}
        q3pairs[pk] = {"Q3": {"name": "x", "cost_families": list(cost or []), "gain_families": [], "final": True}, "beside": {"ade6_m|all": cellj([0.1] * len(d_s), -0.2, 0.4)}}
    hw = {"pairs": pairs, "counts": {"n_windows_total": counts[0], "n_lead_windows": counts[1], "n_episodes_with_lead": counts[2]}, "controls": {"ALL_PASS": hw_all_pass},
          "missing": [] if hw_missing is None else hw_missing}
    q3 = {"pairs": q3pairs, "controls": {"ALL_PASS": q3_all_pass}}
    hwp, q3p = os.path.join(tmp, "hw.json"), os.path.join(tmp, "q3.json")
    json.dump(hw, open(hwp, "w"))
    json.dump(q3, open(q3p, "w"))
    return {"small": small, "pull": pull, "arms": arms_txt, "hw": hwp, "q3": q3p, "tmp": tmp}


H5_KEY = "h5_check_coverage_in_0.15_0.25"


def run_read(paths):
    import read_ip1 as R
    plan_p = os.path.join(paths["tmp"], "plan.json")
    R.plan(paths["small"], paths["pull"], paths["arms"], plan_p)
    return R.read(plan_p, paths["hw"], paths["q3"], paths["small"], paths["pull"], os.path.join(paths["tmp"], "o.json"), os.path.join(paths["tmp"], "o.md"))
