#!/usr/bin/env python3
"""Tests for SAFEOBJ (raw/2026-10-17-safety-objective/PREREG_SAFEOBJ.md + SPEC_SAFEOBJ.md + SPEC_ADDENDUM.md): the constrained-EPO lever `--epo-con g5` of refe/scorer_finetune.py, its gate G13,
the profile `refe_rls1d` (R0d, R1d, R1o) and the chain wiring.  Run:  PYTHONIOENCODING=utf-8 python refe/test_rl_safeobj.py   ->   ZZTEST_PASS <n>   (prints the mutation table).
CPU only, no pod, no checkpoint.  It never runs the G-SUITE that contains it (the suite's membership is checked statically).

THE LEVER.  Arm B pays L_con = sum_k lambda_k (c^k_A - eps - c^k_B) on TTC and NC, c^k_X = mean over sets of E_{pi_X}[y^k] (pi_X = the EPO selection policy of arm X on the SAME micro-batch, arm A DETACHED),
and the duals follow ONE projected ascent step per optimiser update.  Registered values eps 0 / eta 4 / lambda_max 10 / lambda0 0.  The switch `--epo-con` defaults to off, which is every arm before SAFEOBJ.

WHAT IS PROVEN (CLAUDE.md "built, tested, and unreachable from its caller": a mechanism that WORKS is not a mechanism that RUNS):
  P  PINNED / ADDITIVE.  Reversing the patch's edits (SAFEOBJ_FILE_REVERSALS) on the patched scorer_finetune.py gives the R0c-tip trainer EXACTLY (git blob 7c8071ea) and, through test_rl_r0c's chain, the
     pre-R0c trainer (d34d5cf6).  The only functions whose text differs are the named ones; the ONE edit inside a pinned closure (main.train_losses) is the registered block; rl_reward.py is byte-unchanged.
  U  UNIT, on the REAL closures exec'd from main()'s source.  I1: lambda = 0 (called directly: the CLI refuses it as inert) gives loss and gradient BIT-IDENTICAL to --epo-con off.  I2: the fixed penalty
     R' = R - 0.25 (1 - y_TTC) has the gradient of EPO plus L_con at lambda_TTC 0.25 (<= 1e-6 relative).  I3: the gradient at lambda 0.5 is EXACTLY half the one at 1.  I4: the dual arithmetic is exact (0.3125,
     floor 0.0, cap lambda_max).  C1: arm A's gradient from L_con is EXACTLY 0.0 while A's own BCE gradient reads non-zero.  The columns are TTC=3 and NC=0 and nothing else.
  R  REACHABILITY, static (AST) and runtime: argparse -> namespace -> EpoDual -> train_losses -> epo_con_term -> lB -> ((lA + lB) / accum).backward(); micro-batch parts -> dual.step -> lambda ->
     the next update's term; the dual step sits between o.step() and `upd += 1`; the rows, the states file and the checkpoint meta carry the dual.
  E  END TO END through the REAL main() (test_rl_epo.smoke: CPU stubs for the GPU parts only): preflight G1-G13 PASS, the data event records the BUILT facts, EpoDual.step is called once per update with every
     micro-batch and lambda follows the closed form, arm A is bit-identical to the constraint-off run, B differs, `--epo-con off` (named or default) is bit-identical to the R0c-tip trainer, and every
     regression arm (the two env switches) makes G13 FAIL.
  G  THE GATE: refe_rls1d accepts exactly R0d / R1d / R1o, refuses an omitted / default / wrong flag, seed or arm; every older profile refuses --epo-con g5; G-DVB reads the BUILT block; G13 is re-judged
     from raw fields and each raw field has a negative twin.
  CH THE CHAIN (rls1_checks.py / rls1_pod.sh): R1o's K3 is OBSERVE-ONLY; R1o waits for both safeobj_S verdicts; a kill stops the chain.
Every mutation below edits SOURCE (each snippet must occur EXACTLY once, else the test errors) or sets a regression switch, and must turn >= 1 check RED.
"""
from __future__ import annotations

import argparse
import ast as _ast
import contextlib
import copy
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_rl_s1b as S1B  # noqa: E402  (the R0 segment pins, load / segments / git_blob / check, and the segment-level SAFEOBJ reversal)
SF_PATH, LG_PATH, GC_PATH, RR_PATH, RC_PATH, POD_PATH = (os.path.join(HERE, f) for f in ("scorer_finetune.py", "launch_gate.py", "gate_checks.py", "rl_reward.py", "rls1_checks.py", "rls1_pod.sh"))

R0C_TIP_BLOB = "7c8071eacf609a4ba8e04c925fc8abff4e940d88"          # refe/scorer_finetune.py at the R0c tip = the trainer BEFORE this patch
R0_ARGV_SHA256 = "24451b621d515066bf8febefaf66bea2c7a345b48bf7a89c9443e2d848d8eef6"          # R0's banked token argv (HF Sayood/tanitad-refe-rls1-r0 final/PASS_a120418a870f.json)
CHANGED_FUNCTIONS = {"build_parser", "main", "main.train_losses", "main.save_full"}                      # functions whose text differs from the R0c-tip trainer
NEW_FUNCTIONS = {"epo_con_cols", "epo_con_reference", "epo_con_reference_built", "epo_con_term", "epo_con_dual_reach", "epo_con_ckpt_roundtrip", "epo_con_refusals",
                 "g13_epo_con_live_and_identity", "_g13_dual_ok", "main.con_probe"}
NEW_FLAGS = ("--epo-con", "--epo-con-eps", "--epo-con-lr", "--epo-con-max", "--epo-con-lam0", "--epo-con-temp")          # (PICKCON added the sixth) the parser additions over the R0c tip (test_rl_r0c's "only new flag" pin allows exactly these on top of --epo-ref)
UNTOUCHED = ("epo_refusals", "epo_loss", "epo_policy", "epo_ramp", "epo_reference", "epo_reference_built", "epo_reward", "_rl_terms_record", "load_rl_terms", "main.epo_term", "main.losses",
             "main.losses6", "main.sel_score", "main.lane_bce", "module_grad_abs", "g11_epo_live_and_identity")

# <<<GEN SAFEOBJ_FILE_REVERSALS
SAFEOBJ_FILE_REVERSALS = (
    ('    log({"event": "done", "updates": upd, "model_A": pA, "model_B": pB, "hours": (time.time() - t0) / 3600, **({"epo_con": dual.state_dict()} if dual.active else {})})\n',
     '    log({"event": "done", "updates": upd, "model_A": pA, "model_B": pB, "hours": (time.time() - t0) / 3600})\n'),
    ('        meta["n_score_components"] = n_out                    # the planner sizes its score head from this (ckpt_io)\n        if dual.active:                                       # SAFEOBJ: the dual state at this save (an audit; any future resume must restore it)\n            meta["epo_con"] = dual.state_dict()\n',
     '        meta["n_score_components"] = n_out                    # the planner sizes its score head from this (ckpt_io)\n'),
    ('                            **({"AUX": aux.state_dict()} if aux is not None else {}),\n                            **({"EPO_CON": dual.state_dict()} if dual.active else {})},\n',
     '                            **({"AUX": aux.state_dict()} if aux is not None else {})},\n'),
    ('                if dual.active:\n                    nc_ = max(con_log["n"], 1)\n                    cm_ = lambda k_: con_log.get(k_, 0.0) / nc_              # noqa: E731  (window mean of a per-micro-batch part)\n                    row.update({"con": cm_("con"), "con_lam_ttc": dual.lam["ttc"], "con_lam_nc": dual.lam["nc"], "con_g_ttc": cm_("g_ttc"), "con_g_nc": cm_("g_nc"),\n                                "con_cA_ttc": cm_("cA_ttc"), "con_cB_ttc": cm_("cB_ttc"), "con_cA_nc": cm_("cA_nc"), "con_cB_nc": cm_("cB_nc"),\n                                "con_p1_gap_ttc": cm_("p1gap_ttc"), "con_p1_gap_nc": cm_("p1gap_nc"), "con_hinge_ttc": cm_("hinge_ttc"), "con_hinge_nc": cm_("hinge_nc"),\n                                "con_cap_hits": dual.cap_hits, "con_lam_pos_updates": dual.pos_updates,\n                                "con_lam_max_ttc": dual.lam_seen_max["ttc"], "con_lam_max_nc": dual.lam_seen_max["nc"]})\n                    con_log = {"n": 0}\n                if aux is not None:\n                    na = max(aux_acc["n"], 1)\n',
     '                if aux is not None:\n                    na = max(aux_acc["n"], 1)\n'),
    ('            if dual.active:                                  # SAFEOBJ: ONE projected dual step per optimiser update, after o.step() and before `upd += 1` (SPEC 1.4)\n                dual.step(con_acc)\n                for k_, v_ in con_acc.items():\n                    con_log[k_] = con_log.get(k_, 0.0) + v_\n                con_acc = {"n": 0}\n            upd += 1\n            if upd % 25 == 0:\n',
     '            upd += 1\n            if upd % 25 == 0:\n'),
    ('            if dual.active:                                  # SAFEOBJ: this micro-batch\'s parts (g, c_A, c_B, ...) -> the dual step and the logged means\n                for k_, v_ in con_last.items():\n                    con_acc[k_] = con_acc.get(k_, 0.0) + v_\n                con_acc["n"] += 1\n            micro += 1\n            if micro % a.accum:\n                continue\n',
     '            micro += 1\n            if micro % a.accum:\n                continue\n'),
    ('    aux_acc = {"bce": 0.0, "huber": 0.0, "n": 0}              # --aux-drv only\n    con_acc = {"n": 0}                                        # --epo-con only: the constraint\'s parts summed over THIS update\'s micro-batches (EpoDual.step reads it) ...\n    con_log = {"n": 0}                                        # ... and over the updates since the last logged row\n',
     '    aux_acc = {"bce": 0.0, "huber": 0.0, "n": 0}              # --aux-drv only\n'),
    ('and not any(os.environ.get(k_) == "1" for k_ in AUX_MUTATE_ENVS + EPO_CON_MUTATE_ENVS):\n',
     'and not any(os.environ.get(k_) == "1" for k_ in AUX_MUTATE_ENVS):\n'),
    ('            gates["G11_epo_live_and_identity"] = g11\n        if dual.active:                                      # G13 (SAFEOBJ): L_con is live on every B module, A untouched, zero controls, step-0 identity, dual reach, checkpoint block\n            gates["G13_epo_con_live_and_identity"] = g13_epo_con_live_and_identity(\n                model, armB, lambda mode: con_probe(batch, prow, mode), lambda: epo_con_dual_reach(dual), lambda: epo_con_ckpt_roundtrip(dual),\n                kl_tol=(1e-2 if (a.lane_head and not native7) else 0.0))\n',
     '            gates["G11_epo_live_and_identity"] = g11\n'),
    ('    def con_probe(batch, rows, mode):\n        """G13: the constraint term ALONE, as a DIFFERENCE of two train_losses calls on ONE fresh forward (the run\'s own assembly: deleting `lB = lB + ct` leaves a zero difference).  mode \'on\' = the\n        duals at (1, 1); \'lam0\' = at (0, 0) (graph live, factor 0); \'off\' = the block skipped.  The EPO term is built with an EXPLICIT weight 0.0 in both calls, so it cancels.\n        -> (the B-side term, arm A\'s BCE loss of the same forward, the constraint\'s batch parts)"""\n        fw_ = fwd(batch)\n        s0, sA_, sB_, tg_ = fw_\n        keep_comps, keep_lam = dual.comps, dual.lam\n        try:\n            dual.comps = ()\n            ref = train_losses(s0, sA_, sB_, tg_, rows, 0, lam_pg=0.0)\n            if mode != "off":\n                dual.comps, dual.lam = keep_comps, {c: (1.0 if mode == "on" else 0.0) for c in keep_comps}\n            con_last.clear()\n            on = train_losses(s0, sA_, sB_, tg_, rows, 0, lam_pg=0.0)\n        finally:\n            dual.comps, dual.lam = keep_comps, keep_lam\n        return on[1] - ref[1], on[0], dict(con_last)\n\n    @torch.no_grad()\n    def evaluate(tag, limit=None):\n',
     '    @torch.no_grad()\n    def evaluate(tag, limit=None):\n'),
    ('            lB = lB + ep\n            if dual.active:                                  # SAFEOBJ: the constraint, arm B ONLY (arm A is its detached reference); the dual weights are Python floats\n                ct = epo_con_term(sxB, sxA, tg, sel_score, a.epo_temp, a.epo_con_eps, dual.lam, con_last)\n                lB = lB + ct\n        return lA, lB, ln, ep\n',
     '            lB = lB + ep\n        return lA, lB, ln, ep\n'),
    ('                       "terms_train": st_tr, "terms_heldout": st_ho, "native7": native7}\n        info["epo_con"] = dual.built_facts()                 # SAFEOBJ: BUILT facts of the constraint for G-DVB (every --epo run writes the block, an OFF run included)\n',
     '                       "terms_train": st_tr, "terms_heldout": st_ho, "native7": native7}\n'),
    ("    # SAFEOBJ: the constraint's dual object (--epo-con g5; `comps` () = OFF, and then nothing below it runs).  `con_last` receives the latest epo_con_term call's batch parts (as epo_last does).\n    dual = EpoDual(EPO_CON_TERMS.get(a.epo_con, ()), a.epo_con_eps, a.epo_con_lr, a.epo_con_max, a.epo_con_lam0)\n    con_last: dict = {}\n    lv: dict = {}\n    for k_ in (key(ds.rows[i]) for i in tr_idx):\n",
     '    lv: dict = {}\n    for k_ in (key(ds.rows[i]) for i in tr_idx):\n'),
    ("    bad_cli = epo_refusals(a, ap) + epo_con_refusals(a, ap)           # RL-S1: a command line that names EPO levers inconsistently never starts (SAFEOBJ: epo_con_refusals is epo_refusals' sibling)\n",
     '    bad_cli = epo_refusals(a, ap)                            # RL-S1: a command line that names EPO levers inconsistently never starts\n'),
    ('    # SAFEOBJ (raw/2026-10-17-safety-objective/SPEC_SAFEOBJ.md 2): constrained EPO.  The switch\'s default DISABLES it; the numeric defaults are the registered values and enable nothing.\n    ap.add_argument("--epo-con", default=EPO_CON_DEFAULT, choices=EPO_CON_MODES,\n                    help="SAFEOBJ: off = no constraint (default; every arm before SAFEOBJ), g5 = constrain TTC and NC against arm A (needs --epo): L_con = sum_k lambda_k (c^k_A - eps - c^k_B), lambda by dual ascent")\n    ap.add_argument("--epo-con-eps", type=float, default=0.0, help="SAFEOBJ: slack eps of the constraint, in probability units (>= 0)")\n    ap.add_argument("--epo-con-lr", type=float, default=4.0, help="SAFEOBJ: the dual step eta per unit violation per optimiser update (>= 0)")\n    ap.add_argument("--epo-con-max", type=float, default=10.0, help="SAFEOBJ: the cap lambda_max of the duals (> 0)")\n    ap.add_argument("--epo-con-lam0", type=float, default=0.0, help="SAFEOBJ: the initial dual lambda_0, the same for both terms (in [0, --epo-con-max])")\n    ap.add_argument("--dump-scores", default="", help="RL-S1: DIR; after every held-out evaluation write the base / A / B scorers\' per-proposal logits as "\n',
     '    ap.add_argument("--dump-scores", default="", help="RL-S1: DIR; after every held-out evaluation write the base / A / B scorers\' per-proposal logits as "\n'),
    ('\n\n# ================================================================================================ SAFEOBJ: constrained EPO (--epo-con g5; default OFF)\n# raw/2026-10-17-safety-objective/PREREG_SAFEOBJ.md + SPEC_SAFEOBJ.md (+ SPEC_ADDENDUM.md).  Arm B only.  L_con = sum_k lambda_k * (c^k_A - eps - c^k_B) is added to arm B\'s loss inside\n# `train_losses`; arm A (detached) is the reference; the duals are updated ONCE per optimiser update by EpoDual.step.  Nothing here runs unless `--epo --epo-con g5` is on.\nEPO_CON_MODES = ("off", "g5")\nEPO_CON_DEFAULT = "off"            # = every arm before SAFEOBJ, bit for bit; R0d / R1d / R1o NAME the flag (a default that reproduces R0 must not carry the experiment)\nEPO_CON_DESTS = ("epo_con", "epo_con_eps", "epo_con_lr", "epo_con_max", "epo_con_lam0")\nEPO_CON_TERMS = {"g5": ("ttc", "nc")}          # mode -> the PDM sub-scores constrained: g5 = "TTC and NC may not fall by > 0.5 versus control" (rl_val_eval.py g5)\nEPO_CON_MUTATE_ENVS = ("REFE_RL_MUTATE_CON_DETACH", "REFE_RL_MUTATE_CON_ATTACH_A")      # deliberate regressions: each must turn G13 RED\n\n\ndef epo_con_cols(comps):\n    """The label columns of the constrained terms, READ from rl_reward.COMPONENTS (nc, dac, ep, ttc, c, ddc) by name -- never typed (a typed column would silently constrain EP)."""\n    return tuple(rl_reward.COMPONENTS.index(c) for c in comps)\n\n\ndef epo_con_reference(sxA):\n    """Arm A\'s logits as the constraint\'s REFERENCE: DETACHED, so no gradient of L_con can reach arm A (the control stays BCE-only).  REFE_RL_MUTATE_CON_ATTACH_A=1 keeps the graph (G13 / C1 must go RED)."""\n    if os.environ.get("REFE_RL_MUTATE_CON_ATTACH_A") == "1":\n        return sxA\n    return sxA.detach()\n\n\ndef epo_con_reference_built():\n    """What epo_con_reference ACTUALLY returns, read by calling it on a sentinel -> "armA_detached" / "armA_attached" / "other".  This is what the data event\'s epo_con.ref records for G-DVB."""\n    sA = torch.ones(1, requires_grad=True)\n    out = epo_con_reference(sA)\n    if out is sA:\n        return "armA_attached"\n    if (not out.requires_grad) and torch.equal(out, sA.detach()):\n        return "armA_detached"\n    return "other"\n\n\ndef epo_con_term(sxB, sxA, tg, rule_fn, T, eps, lam, parts=None):\n    """L_con = sum_k lam[k] * (c^k_A - eps - c^k_B)  (SPEC_SAFEOBJ 1.2), float32.  sxB / sxA [b, M, C] = arm B\'s / arm A\'s logits of the SAME micro-batch (sxA is detached HERE, by\n    epo_con_reference, which is the one guard G13 reads); tg [b, M, >= 6] the PDM labels in rl_reward.COMPONENTS order; rule_fn = the base\'s selection rule (main\'s sel_score); T = --epo-temp.\n    c^k_X = mean over sets of sum_i pi^X_i y^k_i with pi^X = epo_policy(sxX) -- the very policy of the EPO term.  lam = {component: float} (constant within one optimiser update).\n    `parts`, if given, receives the batch\'s detached floats: con (the term), and per component g_ / cA_ / cB_ (the violation and the two outcomes), hinge_ (the share of sets whose own\n    violation is > 0) and p1gap_ (the label of B\'s argmax pick minus A\'s: P1, negative = B picks the worse one)."""\n    comps = tuple(lam)\n    if not comps:\n        raise ValueError("epo_con_term needs at least one constrained component")\n    sxA_ref = epo_con_reference(sxA)\n    if os.environ.get("REFE_RL_MUTATE_CON_DETACH") == "1":          # deliberate regression for G13: c_B loses its graph\n        sxB = sxB.detach()\n    piB = epo_policy(sxB.float(), rule_fn, T).exp()\n    piA = epo_policy(sxA_ref.float(), rule_fn, T).exp()\n    y = tg.detach().float()\n    L = None\n    for comp, col in zip(comps, epo_con_cols(comps)):\n        yk = y[..., col]\n        cA_s, cB_s = (piA * yk).sum(1), (piB * yk).sum(1)\n        g = cA_s.mean() - eps - cB_s.mean()\n        t_ = float(lam[comp]) * g\n        L = t_ if L is None else L + t_\n        if parts is not None:\n            with torch.no_grad():\n                ia = rule_fn(torch.sigmoid(sxA_ref.detach().float())).argmax(1, keepdim=True)\n                ib = rule_fn(torch.sigmoid(sxB.detach().float())).argmax(1, keepdim=True)\n                parts["g_" + comp], parts["cA_" + comp], parts["cB_" + comp] = float(g.detach()), float(cA_s.detach().mean()), float(cB_s.detach().mean())\n                parts["hinge_" + comp] = float(((cA_s.detach() - eps - cB_s.detach()) > 0).float().mean())\n                parts["p1gap_" + comp] = float((yk.gather(1, ib) - yk.gather(1, ia)).mean())\n    if parts is not None and L is not None:\n        parts["con"] = float(L.detach())\n    return L\n\n\nclass EpoDual:\n    """The dual variables of --epo-con and their projected ascent (SPEC_SAFEOBJ 1.4): lambda_k <- min(lam_max, max(0, lambda_k + eta * gbar_k)), gbar_k = the mean over the update\'s\n    micro-batches of the detached violation g^k.  `comps` = () is the OFF object (`active` False: train_losses never calls epo_con_term).  lambda used during update u is the value BEFORE\n    u\'s step; `step` runs once per optimiser update, after o.step() and before `upd += 1`."""\n\n    def __init__(self, comps, eps, lr, lam_max, lam0):\n        self.comps, self.eps, self.lr, self.lam_max, self.lam0 = tuple(comps), float(eps), float(lr), float(lam_max), float(lam0)\n        self.reset()\n\n    def reset(self):\n        self.lam = {c: self.lam0 for c in self.comps}\n        self.updates = self.cap_hits = self.pos_updates = 0\n        self.lam_seen_max = {c: self.lam0 for c in self.comps}\n\n    @property\n    def active(self):\n        return bool(self.comps)\n\n    def step(self, acc):\n        """acc = {"n": micro-batches, "g_<comp>": the SUM over them of the detached violation}.  An empty accumulator is a wiring bug and raises."""\n        n = int(acc.get("n", 0))\n        if n <= 0:\n            raise ValueError("EpoDual.step: no micro-batch was accumulated for this update (the constraint\'s per-batch parts never reached the dual)")\n        if any(v > 0.0 for v in self.lam.values()):\n            self.pos_updates += 1\n        for c in self.comps:\n            raw = self.lam[c] + self.lr * (float(acc["g_" + c]) / n)\n            self.lam[c] = min(self.lam_max, max(0.0, raw))\n            self.lam_seen_max[c] = max(self.lam_seen_max[c], self.lam[c])\n            if raw >= self.lam_max:\n                self.cap_hits += 1\n        self.updates += 1\n\n    def built_mode(self):\n        for m, c in EPO_CON_TERMS.items():\n            if tuple(c) == self.comps:\n                return m\n        return "other" if self.comps else "off"\n\n    def built_facts(self):\n        """BUILT facts for G-DVB (the data event\'s `epo_con` block): read off this object and the selectors that run, never echoed from the namespace."""\n        return {"mode": self.built_mode(), "comps": list(self.comps), "cols": list(epo_con_cols(self.comps)), "eps": self.eps, "lr": self.lr, "max": self.lam_max,\n                "lam0": self.lam0, "ref": epo_con_reference_built()}\n\n    def state_dict(self):\n        return {"lam": dict(self.lam), "update": self.updates, "cap_hits": self.cap_hits, "pos_updates": self.pos_updates, "lam_seen_max": dict(self.lam_seen_max)}\n\n    def load_state_dict(self, sd):\n        self.lam, self.updates, self.cap_hits = dict(sd["lam"]), int(sd["update"]), int(sd["cap_hits"])\n        self.pos_updates, self.lam_seen_max = int(sd["pos_updates"]), dict(sd["lam_seen_max"])\n\n\ndef epo_con_dual_reach(dual):\n    """G13 "dual reach": SYNTHETIC sequences through a COPY of the run\'s own EpoDual.step (the run\'s lambda is never touched).  (a) ten steps of a violation of 2^-7 -> lambda0 + 10 * eta * 2^-7\n    (0.3125 exactly at the registered eta 4, lambda0 0); (b) then three steps of a violation of -1 floor lambda at 0.0 exactly; (c) then a violation of 1e6 caps it at lam_max exactly."""\n    d = copy.deepcopy(dual)\n    d.reset()\n    viol, steps = 2.0 ** -7, 10\n\n    def feed(g, n_steps):\n        for _ in range(n_steps):\n            d.step({"n": 8, **{"g_" + c: 8 * g for c in d.comps}})\n        return dict(d.lam)\n    got = feed(viol, steps)\n    floor_got = feed(-1.0, 3)\n    cap_got = feed(1e6, 2)\n    return {"comps": list(d.comps), "eta": dual.lr, "lam0": dual.lam0, "lam_max": dual.lam_max, "viol": viol, "steps": steps, "got": got, "floor_got": floor_got, "cap_got": cap_got}\n\n\ndef epo_con_ckpt_roundtrip(dual):\n    """G13 "checkpoint": the EPO_CON block (EpoDual.state_dict) survives torch.save / torch.load and load_state_dict BIT-EQUAL, on non-trivial values (so a block that dropped a field cannot pass)."""\n    import io\n    d = copy.deepcopy(dual)\n    d.lam = {c: 0.3125 + i / 3.0 for i, c in enumerate(d.comps)}\n    d.updates, d.cap_hits, d.pos_updates, d.lam_seen_max = 11, 7, 5, {c: 1.0 + i for i, c in enumerate(d.comps)}\n    buf = io.BytesIO()\n    torch.save({"EPO_CON": d.state_dict()}, buf)\n    buf.seek(0)\n    back = torch.load(buf, weights_only=False)["EPO_CON"]\n    e = copy.deepcopy(dual)\n    e.load_state_dict(back)\n    return {"keys": sorted(back), "block_equal": bool(back == d.state_dict()), "restored_equal": bool(e.state_dict() == d.state_dict())}\n\n\ndef epo_con_refusals(a, ap):\n    """Every reason the --epo-con levers must not start, as messages ([] = fine).  epo_refusals stays byte-unchanged (pinned): this is its sibling, called beside it (main(), the launch gate).\n    `ap` is build_parser() (its defaults say which levers were NAMED)."""\n    bad = []\n    con_named = [d for d in EPO_CON_DESTS if getattr(a, d) != ap.get_default(d)]\n    if not a.epo:\n        if con_named:\n            bad.append(f"--epo-con levers {con_named} are set but --epo is not: they would be silently ignored")\n        return bad\n    if a.epo_con not in EPO_CON_MODES:\n        bad.append(f"--epo-con {a.epo_con!r} is not one of {EPO_CON_MODES}")\n        return bad\n    if a.epo_con == "off":\n        if con_named:\n            bad.append(f"--epo-con-* levers {con_named} are set but --epo-con is off: their weights would belong to a term that is OFF")\n        return bad\n    if not a.epo_con_eps >= 0.0:\n        bad.append("--epo-con-eps must be >= 0")\n    if not a.epo_con_lr >= 0.0:\n        bad.append("--epo-con-lr must be >= 0")\n    if not a.epo_con_max > 0.0:\n        bad.append("--epo-con-max must be > 0")\n    if not 0.0 <= a.epo_con_lam0 <= a.epo_con_max:\n        bad.append("--epo-con-lam0 must be in [0, --epo-con-max]")\n    if a.epo_con_lr == 0.0 and a.epo_con_lam0 == 0.0:\n        bad.append("--epo-con g5 with --epo-con-lr 0 AND --epo-con-lam0 0 is INERT by construction (lambda is 0 at every update): a wired mechanism that can never act")\n    return bad\n\n\ndef g13_epo_con_live_and_identity(armA, armB, probe, dual_probe, ckpt_probe, kl_tol, names=SCORER):\n    """G13 (SAFEOBJ preflight), one dict of RAW FIELDS that the launch gate re-judges (gate_checks.judge_g13): it never trusts this `ok`.\n    probe(mode) -> (the constraint term ALONE as a DIFFERENCE of two `train_losses` calls on one fresh forward -- the run\'s own assembly, so deleting `lB = lB + ct` there leaves it\n    identically zero --, arm A\'s BCE loss of that forward, the constraint\'s batch parts) for mode \'on\' (lambda = (1, 1)), \'lam0\' (lambda = (0, 0): graph live, factor 0) and \'off\'\n    (the block skipped).  An L_con-ONLY backward must give a finite non-zero gradient on EVERY scorer module of arm B (\'on\'); the two controls must read EXACTLY 0.0 (a zero from a broken\n    probe looks the same); arm A must read EXACTLY 0.0 from it while A\'s own BCE gradient (same probe) reads non-zero; at step 0 the two policies coincide (c_A == c_B, so g = -eps)."""\n    reads, readsA, vals, infos, a_ctrl = {}, {}, {}, {}, None\n    modes = ("on", "lam0", "off")\n    for mode in modes:\n        armA.zero_grad(set_to_none=True)\n        armB.zero_grad(set_to_none=True)\n        term, loss_a, info = probe(mode)\n        vals[mode] = float(term.detach())\n        infos[mode] = info\n        if bool(term.requires_grad):\n            term.backward(retain_graph=True)\n        reads[mode], readsA[mode] = module_grad_abs(armB, names), module_grad_abs(armA, names)\n        if mode == "on":\n            armA.zero_grad(set_to_none=True)\n            if loss_a.requires_grad:\n                loss_a.backward()\n            a_ctrl = module_grad_abs(armA, names)\n    armA.zero_grad(set_to_none=True)\n    armB.zero_grad(set_to_none=True)\n    on_reads = reads["on"]\n    live = {m: bool(on_reads[m] > 0.0 and math.isfinite(on_reads[m])) for m in names}\n    zero_ok = vals["lam0"] == 0.0 and vals["off"] == 0.0 and all(v == 0.0 for r in ("lam0", "off") for v in reads[r].values())\n    a_zero = all(v == 0.0 for r in ("on", "lam0", "off") for v in readsA[r].values())\n    a_ctrl_live = bool(a_ctrl) and all(math.isfinite(v) and v > 0.0 for v in a_ctrl.values())\n    on = infos["on"]\n    comps = sorted(k[3:] for k in on if k.startswith("cA_"))\n    gap = {c: abs(on["cA_" + c] - on["cB_" + c]) for c in comps}\n    ident_ok = bool(gap) and all(math.isfinite(v) and v <= kl_tol for v in gap.values())\n    dual_reach, ckpt = dual_probe(), ckpt_probe()\n    mutate = [k for k in EPO_CON_MUTATE_ENVS if os.environ.get(k) == "1"]\n    dual_ok = _g13_dual_ok(dual_reach)\n    ckpt_ok = bool(ckpt.get("block_equal") and ckpt.get("restored_equal"))\n    return {"ok": bool(all(live.values()) and zero_ok and a_zero and a_ctrl_live and ident_ok and dual_ok and ckpt_ok and not mutate), "per_module_on": reads["on"],\n            "per_module_lam0": reads["lam0"], "per_module_off": reads["off"], "A_per_module_on": readsA["on"], "A_per_module_lam0": readsA["lam0"], "A_per_module_off": readsA["off"],\n            "A_bce_control": a_ctrl, "live": live, "controls_exactly_zero": bool(zero_ok), "A_untouched": bool(a_zero), "A_control_reads_nonzero": bool(a_ctrl_live),\n            "value_on": vals["on"], "value_lam0": vals["lam0"], "value_off": vals["off"], "c_gap": gap, "c_gap_tol": kl_tol, "identity_ok": bool(ident_ok), "comps": comps,\n            "dual_reach": dual_reach, "dual_ok": bool(dual_ok), "checkpoint": ckpt, "mutate": mutate}\n\n\ndef _g13_dual_ok(d):\n    """the probe\'s own reading of its synthetic dual sequences (gate_checks.judge_g13 re-derives the expected values from the raw fields; this is only the trainer-side `ok`)"""\n    exp = d["lam0"]\n    for _ in range(d["steps"]):\n        exp = min(d["lam_max"], max(0.0, exp + d["eta"] * d["viol"]))\n    return bool(d["comps"] and all(d["got"][c] == exp and d["floor_got"][c] == 0.0 and d["cap_got"][c] == d["lam_max"] for c in d["comps"]))\n\n\ndef epo_loss(sxB, sx_ref, R, rule_fn, T, beta, lam_pg, parts=None):\n',
     '\n\ndef epo_loss(sxB, sx_ref, R, rule_fn, T, beta, lam_pg, parts=None):\n'),
    ("  --epo-con       SAFEOBJ (raw/2026-10-17-safety-objective/{PREREG,SPEC}_SAFEOBJ.md): CONSTRAINED EPO.  off (the default = every arm before SAFEOBJ, bit for bit) | g5 = arm B also pays\n                  L_con = sum_k lambda_k * (c^k_A - eps - c^k_B),  k in {TTC, NC},  c^k_X = mean over sets of E_{pi_X}[y^k]  (pi_X = the EPO selection policy of arm X on the SAME\n                  micro-batch; arm A's is DETACHED), with ONE projected dual-ascent step per optimiser update: lambda_k <- clip(lambda_k + eta * mean g^k, 0, lambda_max).  Needs --epo.\n  --epo-con-eps / --epo-con-lr (eta) / --epo-con-max (lambda_max) / --epo-con-lam0 (lambda_0): the slack, the dual step, the cap and the start (registered 0 / 4 / 10 / 0; they enable nothing:\n                  only `--epo-con g5` does).  The profile (refe_rls1d) requires all five NAMED on R0d / R1d and `--epo-con off` NAMED on R1o.  The preflight adds gate G13.\n  --dump-scores DIR  OBSERVATIONAL, default off: after every held-out evaluation write the base / A / B scorers' per-proposal logits of every held-out set as\n",
     "  --dump-scores DIR  OBSERVATIONAL, default off: after every held-out evaluation write the base / A / B scorers' per-proposal logits of every held-out set as\n"),
)
# >>>GEN SAFEOBJ_FILE_REVERSALS


def reverse_edits(src: str) -> str:
    """the patched trainer -> the R0c-tip trainer (every edit reversed; each must occur exactly once).  PICKCON (the constraint's own temperature, test_rl_pickcon.py) is the OUTERMOST layer: its
    edits are reversed FIRST (S1B.pickcon_reverse_file -> the 0a5958a trainer, the one SAFEOBJ landed), then SAFEOBJ's own."""
    src = S1B.pickcon_reverse_file(src)
    for new, old in SAFEOBJ_FILE_REVERSALS:
        assert src.count(new) == 1, f"a patch edit occurs {src.count(new)} times in the trainer (must be exactly 1): {new[:70]!r}"
        src = src.replace(new, old)
    return src


D = "/workspace/data"
TERMS = f"--epo --epo-terms-train {D}/rls1/terms_train --epo-terms-heldout {D}/rls1/terms_val"
CON_ON = "--epo-con g5 --epo-con-eps 0 --epo-con-lr 4 --epo-con-max 10 --epo-con-lam0 0"


def arm_str(arm: str, seed: int, epo: str = "") -> str:
    ad = f"{D}/rls1/arms/{arm}"
    return (f"--ckpt {D}/rls1/base/model_sft4_B_u400.pt --backbone vitl16 --targets {D}/refe_navtrain/train_grow --images {D}/navtrain_pixels --calib {D}/refe_navtrain/train_grow/calib_table.json "
            f"--onpolicy {D}/rls1/sets_train --heldout {D}/rls1/sets_val --heldout-logs {D}/rls1/rl_val_logs.txt --out {ad} --pdm-labels-train {D}/rls1/pdm_train "
            f"--pdm-labels-heldout {D}/rls1/pdm_val --require-pdm --b-mode listnet --lam 0 --lr 1e-4 --max-updates 1000 --epochs 4 --workers 3 --seed {seed} --eval-every 250 "
            f"--dump-scores {ad}/scores --lane-head --teacher-lane-train {D}/rls1/tlane_train --teacher-lane-heldout {D}/rls1/tlane_heldout" + (f" {epo}" if epo else ""))


ARM_ARGS = {"C0": arm_str("C0", 0), "C1": arm_str("C1", 1), "R0": arm_str("R0", 0, TERMS), "R1": arm_str("R1", 1, TERMS), "R2": arm_str("R2", 0, TERMS + " --epo-k --epo-l"),
            "R0b": arm_str("R0b", 0, TERMS + " --epo-lam 0.25"), "R1b": arm_str("R1b", 1, TERMS + " --epo-lam 0.25"), "R0c": arm_str("R0c", 0, TERMS + " --epo-ref armA"),
            "R0d": arm_str("R0d", 0, TERMS + " " + CON_ON), "R1d": arm_str("R1d", 1, TERMS + " " + CON_ON), "R1o": arm_str("R1o", 1, TERMS + " --epo-con off")}


def check(bad, name, fn):
    S1B.check(bad, name, fn)


def bits(*xs) -> tuple:
    return tuple(torch.as_tensor(x).detach().cpu().double().numpy().tobytes() for x in xs)


@contextlib.contextmanager
def env(**kw):
    keep = {k: os.environ.get(k) for k in kw}
    os.environ.update({k: v for k, v in kw.items() if v is not None})
    for k, v in kw.items():
        if v is None:
            os.environ.pop(k, None)
    try:
        yield
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def rule_v1(p):                                   # an INDEPENDENT statement of the base's rule: navsim_v1 over the six components, x the lane output when present
    v = p[..., 0] * p[..., 1] * (5.0 * p[..., 2] + 5.0 * p[..., 3] + 2.0 * p[..., 4]) / 12.0
    return v * p[..., 6] if p.shape[-1] == 7 else v


# the five flags of one constrained run of the real main() (the smoke's own argv has the rest); lambda0 0.5 makes lambda MOVE whatever the sign of the violation
SMOKE_ON = ("--epo-con", "g5", "--epo-con-eps", "0", "--epo-con-lr", "4", "--epo-con-max", "10", "--epo-con-lam0", "0")
SMOKE_ON_L = ("--epo-con", "g5", "--epo-con-eps", "0", "--epo-con-lr", "4", "--epo-con-max", "10", "--epo-con-lam0", "0.5")
SEG_NEW = S1B.redo_pickcon("main.train_losses", S1B.SAFEOBJ_REVERSALS["main.train_losses"][0][0]) if hasattr(S1B, "SAFEOBJ_REVERSALS") else ""          # the registered block, spelled as the PICKCON trainer carries it (dual.temp)


# ================================================================================================ part SF: the trainer
def suite_sf(sf_src: str, tag: str, tip_src: str) -> list:
    import carvefeed_layer as CL
    sf_src = CL.peel(sf_src)                         # B8 (raw/2026-10-18-r0e-carvefeed) sits on top of this layer and has its own suite (test_carve_feed.py): this suite is about the trainer BEFORE it
    import test_rl_epo as TE
    import rl_reward as RW
    bad: list = []
    C = TE.Ctx(sf_src, f"sf_so_{tag}")
    M = C.mod
    seg, seg_tip = S1B.segments(sf_src), S1B.segments(tip_src)
    sha = lambda s: hashlib.sha256(s.encode()).hexdigest()          # noqa: E731

    # ---------------------------------------------------------------- P: pinned / purely additive
    check(bad, "P_reversal_is_the_R0c_tip_trainer", lambda: S1B.git_blob(reverse_edits(sf_src).encode("utf-8")) == R0C_TIP_BLOB)

    def chain_to_r0():
        import test_rl_r0c as R0C                                    # lazy: R0C imports this module's reverse_edits at the top, so the import must not run at module load
        s = tip_src
        for new, old in R0C.R0C_FILE_REVERSALS:
            assert s.count(new) == 1
            s = s.replace(new, old)
        return S1B.git_blob(s.encode("utf-8")) == R0C.R0_TRAINER_BLOB
    check(bad, "P_the_R0c_tip_still_reverses_to_the_pre_R0c_trainer_d34d5cf6", chain_to_r0)
    differ = {k for k in set(seg) | set(seg_tip) if seg.get(k) != seg_tip.get(k)}
    check(bad, "P_only_the_named_functions_differ_from_the_R0c_tip_trainer", lambda: differ == CHANGED_FUNCTIONS | NEW_FUNCTIONS)
    check(bad, "P_the_EPO_core_and_epo_refusals_are_byte_unchanged", lambda: all(seg[k] == seg_tip[k] for k in UNTOUCHED))
    check(bad, "P_train_losses_gained_exactly_the_registered_block", lambda: bool(SEG_NEW) and seg["main.train_losses"].count(SEG_NEW) == 1
          and seg["main.train_losses"].replace(SEG_NEW, "") == seg_tip["main.train_losses"])
    check(bad, "P_R0_pin_main.train_losses_through_both_reversals", lambda: sha(S1B.undo_r0c("main.train_losses", S1B.undo_safeobj("main.train_losses", seg["main.train_losses"]))) == S1B.R0_SEGMENTS["main.train_losses"])
    check(bad, "P_reward_module_untouched", lambda: S1B.git_blob(open(RR_PATH, "rb").read().replace(b"\r\n", b"\n")) == S1B.R0_REWARD_GIT_BLOB)
    check(bad, "P_default_is_off_and_the_numeric_defaults_are_the_registered_values", lambda: M.EPO_CON_DEFAULT == "off" and M.EPO_CON_MODES == ("off", "g5")
          and M.build_parser().get_default("epo_con") == "off" and M.build_parser().get_default("epo_con_eps") == 0.0 and M.build_parser().get_default("epo_con_lr") == 4.0
          and M.build_parser().get_default("epo_con_max") == 10.0 and M.build_parser().get_default("epo_con_lam0") == 0.0)

    def other_actions_unchanged():
        old = TE.load(tip_src, f"sf_so_old_{tag}")
        new_p, old_p = M.build_parser(), old.build_parser()
        nm = {tuple(x.option_strings): x for x in new_p._actions}
        for x in old_p._actions:
            y = nm.get(tuple(x.option_strings))
            if y is None or type(x) is not type(y) or any(getattr(x, at) != getattr(y, at) for at in ("dest", "default", "type", "choices", "required", "nargs", "const", "help")):
                return False
        return {o[0] for o in set(nm) - {tuple(x.option_strings) for x in old_p._actions}} == {"--epo-con", "--epo-con-eps", "--epo-con-lr", "--epo-con-max", "--epo-con-lam0", "--epo-con-temp"}
    check(bad, "P_every_other_flag_of_the_parser_unchanged_and_the_five_epo_con_flags_and_PICKCON_s_epo_con_temp_are_the_only_new_ones", other_actions_unchanged)

    # ---------------------------------------------------------------- helpers on the REAL closures
    DS = type("DS", (), {})()
    DS.rows = [{"log_name": f"L{i}", "token": f"t{i}", "step": i, "rank": 0} for i in range(6)]
    rngd = np.random.default_rng(9)
    TERMS_D = {(f"L{i}", f"t{i}", i, 0): {"d": rngd.random(8).astype(np.float32)} for i in range(4)}
    ROWS = torch.tensor([3, 1, 0, 2])
    CL = ["lane_bce", "sel_score", "losses", "losses6", "epo_term", "train_losses"]
    SC = ("score_q_mlp", "score_dec", "score_head")

    def mk(lane, dual=None, names=CL, fwd=None, **kw):
        base = dict(lam=0.0, epo=True, epo_lam=1.0, epo_ramp=2, epo_temp=0.1, epo_beta=0.1, accum=2, epo_ref="base", epo_con="g5", epo_con_eps=0.0, epo_con_lr=4.0, epo_con_max=10.0, epo_con_lam0=0.0, epo_con_temp=None)
        base.update(kw)
        an = TE.a_ns("listnet", lane, **base)
        d = dual if dual is not None else M.EpoDual(("ttc", "nc"), 0.0, 4.0, 10.0, 0.0, temp=0.1)
        cl: dict = {}
        return an, C.ns(names, a=an, compw=torch.tensor(TE.COMPW), rcfg=RW.RewardCfg(), terms_train=TERMS_D, ds=DS, epo_last={}, fwd=fwd, dual=d, con_last=cl), d, cl

    def fresh(n_out):
        A, B = TE.Tiny(0, n_out), TE.Tiny(0, n_out)
        base = copy.deepcopy(A)
        for p in base.parameters():
            p.requires_grad_(False)
        return A, B, base

    def moved(n_out):
        """B updated twice by AdamW on its own BCE (a different micro-batch each), so A (= the start) != B: c_A != c_B and the violation is not zero"""
        an, ns, _d, _cl = mk(n_out == 7)
        A, B, base = fresh(n_out)
        data = TE.micro_data(n_out, 3, 5)
        opt = torch.optim.AdamW(list(B.parameters()), lr=0.05, weight_decay=0.0)
        for k in (0, 1):
            opt.zero_grad()
            ns["losses"](A(data[k][0], data[k][1]).detach(), B(data[k][0], data[k][1]), data[k][2], 0.0)[1].backward()
            opt.step()
        return A, B, base, data[2]

    def con_only(lane, state, lam, parts=None, tg_edit=None, eps=0.0):
        """the REAL epo_con_term on (B, A) of `state` -> (loss, grads w.r.t. B, grads w.r.t. A or None)"""
        A, B, base, (cxy, sctx, tg) = state
        an, ns, _d, _cl = mk(lane)
        tg = tg if tg_edit is None else tg_edit(tg.clone())
        for q in list(A.parameters()) + list(B.parameters()):
            q.grad = None
        loss = M.epo_con_term(B(cxy, sctx), A(cxy, sctx), tg, ns["sel_score"], 0.1, eps, lam, parts)
        loss.backward()
        return loss, [p.grad.clone() for p in B.parameters()], [p.grad for p in A.parameters()]

    # ---------------------------------------------------------------- U: I1 lambda = 0 is --epo-con off, bit for bit
    def i1(lane, apart):
        n_out = 7 if lane else 6
        out = {}
        for variant in ("lam0", "off"):
            if apart:
                A, B, base, (cxy, sctx, tg) = moved(n_out)
            else:
                A, B, base = fresh(n_out)
                cxy, sctx, tg = TE.micro_data(n_out, 1, 1)[0]
            d = M.EpoDual(("ttc", "nc") if variant == "lam0" else (), 0.0, 0.0, 10.0, 0.0, temp=0.1)
            an, ns, d, cl = mk(lane, dual=d)
            with torch.no_grad():
                sx0 = base(cxy, sctx)
            lA, lB, ln, ep = ns["train_losses"](sx0, A(cxy, sctx), B(cxy, sctx), tg, ROWS, 5)
            (lA + lB).backward()
            out[variant] = (bits(lA, lB, ln, ep), [p.grad.clone() for p in B.parameters()], [p.grad.clone() for p in A.parameters()], dict(cl))
        a_, b_ = out["lam0"], out["off"]
        return (a_[0] == b_[0] and all(torch.equal(x, y) for x, y in zip(a_[1], b_[1])) and all(torch.equal(x, y) for x, y in zip(a_[2], b_[2]))
                and "g_ttc" in a_[3] and not b_[3] and all(float(g.abs().max()) > 0.0 for g in a_[1]))
    for lane in (False, True):
        for apart in (False, True):
            check(bad, f"U_I1_lambda_0_is_bit_identical_to_off_{'lane' if lane else '6col'}_{'apart' if apart else 'step0'}", lambda lane=lane, apart=apart: i1(lane, apart))

    # ---------------------------------------------------------------- U: I2 the fixed penalty (b) is this mechanism at lambda_TTC = 0.25
    def i2(lane, lam_ttc):
        n_out = 7 if lane else 6
        A, B, base, (cxy, sctx, tg) = moved(n_out)
        an, ns, _d, _cl = mk(lane)
        sel = ns["sel_score"]
        with torch.no_grad():
            sx0, sxA = base(cxy, sctx), A(cxy, sctx)
        R = torch.rand(tg.shape[0], tg.shape[1], generator=torch.Generator().manual_seed(4))
        y = tg[..., RW.COMPONENTS.index("ttc")]

        def grads(fn):
            B.zero_grad(set_to_none=True)
            fn(B(cxy, sctx)).backward()
            return [p.grad.clone() for p in B.parameters()]
        gb = grads(lambda sxB: M.epo_loss(sxB, sx0, R - 0.25 * (1.0 - y), sel, 0.1, 0.1, 1.0))
        ga = grads(lambda sxB: M.epo_loss(sxB, sx0, R, sel, 0.1, 0.1, 1.0) + M.epo_con_term(sxB, sxA, tg, sel, 0.1, 0.0, {"ttc": lam_ttc, "nc": 0.0}))
        den = max(float(g.abs().max()) for g in gb)
        return max(float((x - y_).abs().max()) for x, y_ in zip(ga, gb)) / den
    for lane in (False, True):
        check(bad, f"U_I2_R_prime_gradient_equals_EPO_plus_L_con_at_lambda_TTC_0.25_{'lane' if lane else '6col'}", lambda lane=lane: i2(lane, 0.25) <= 1e-6)
    check(bad, "U_I2_twin_the_identity_is_not_vacuous_a_wrong_lambda_breaks_it", lambda: i2(False, 0.5) > 1e-3)

    # ---------------------------------------------------------------- U: I3 linear dependence on lambda, both components live and independent
    def i3(lane):
        st = moved(7 if lane else 6)
        _l1, g1, _ = con_only(lane, st, {"ttc": 1.0, "nc": 1.0})
        _lh, gh, _ = con_only(lane, st, {"ttc": 0.5, "nc": 0.5})
        _lt, gt, _ = con_only(lane, st, {"ttc": 1.0, "nc": 0.0})
        _ln, gn, _ = con_only(lane, st, {"ttc": 0.0, "nc": 1.0})
        return (all(torch.equal(h, g * 0.5) for h, g in zip(gh, g1)) and all(float(g.abs().max()) > 0.0 for g in g1) and all(float(g.abs().max()) > 0.0 for g in gt)
                and all(float(g.abs().max()) > 0.0 for g in gn) and any(not torch.equal(x, y) for x, y in zip(gt, gn)))
    check(bad, "U_I3_gradient_at_lambda_0.5_is_EXACTLY_half_of_lambda_1_and_each_component_is_live_6col", lambda: i3(False))
    check(bad, "U_I3_gradient_at_lambda_0.5_is_EXACTLY_half_of_lambda_1_and_each_component_is_live_lane", lambda: i3(True))

    def zero_lambda():
        st = moved(6)
        loss, g, gA = con_only(False, st, {"ttc": 0.0, "nc": 0.0})
        return float(loss) == 0.0 and all(float(x.abs().max()) == 0.0 for x in g) and all(x is None for x in gA)
    check(bad, "U_lambda_0_control_reads_EXACTLY_0.0_value_and_gradient", zero_lambda)

    # ---------------------------------------------------------------- U: the sign, the columns, the temperature policy
    def direction():
        st = moved(6)
        A, B, base, (cxy, sctx, tg) = st
        an, ns, _d, _cl = mk(False)
        p0, p1 = {}, {}
        loss, g, _ = con_only(False, st, {"ttc": 1.0, "nc": 0.0}, p0)
        with torch.no_grad():
            for p, gr in zip(B.parameters(), g):
                p.sub_(1e-2 * gr)
            L1 = M.epo_con_term(B(cxy, sctx), A(cxy, sctx), tg, ns["sel_score"], 0.1, 0.0, {"ttc": 1.0, "nc": 0.0}, p1)
        return float(L1) < float(loss) and p1["cB_ttc"] > p0["cB_ttc"] and abs(p1["cA_ttc"] - p0["cA_ttc"]) == 0.0 and float(loss) == p0["g_ttc"] and "g_nc" in p0
    check(bad, "U_sign_a_step_on_L_con_RAISES_B_s_expected_TTC_and_lowers_the_term", direction)

    def columns():
        st = moved(6)
        p0, pe, pt, pn = {}, {}, {}, {}
        con_only(False, st, {"ttc": 1.0, "nc": 1.0}, p0)
        con_only(False, st, {"ttc": 1.0, "nc": 1.0}, pe, tg_edit=lambda t: (t.__setitem__((Ellipsis, 2), torch.rand_like(t[..., 2])), t.__setitem__((Ellipsis, 1), torch.rand_like(t[..., 1])),
                                                                          t.__setitem__((Ellipsis, 4), torch.rand_like(t[..., 4])), t)[-1])
        con_only(False, st, {"ttc": 1.0, "nc": 1.0}, pt, tg_edit=lambda t: (t.__setitem__((Ellipsis, 3), 1.0 - t[..., 3]), t)[-1])
        con_only(False, st, {"ttc": 1.0, "nc": 1.0}, pn, tg_edit=lambda t: (t.__setitem__((Ellipsis, 0), 1.0 - t[..., 0]), t)[-1])
        same = lambda p, q, ks: all(p[k] == q[k] for k in ks)          # noqa: E731
        return (same(p0, pe, ("cB_ttc", "cB_nc", "con")) and p0["cB_ttc"] != pt["cB_ttc"] and p0["cB_nc"] == pt["cB_nc"] and p0["cB_nc"] != pn["cB_nc"] and p0["cB_ttc"] == pn["cB_ttc"]
                and M.epo_con_cols(("ttc", "nc")) == (3, 0) and M.epo_con_cols(("ttc", "nc")) == (RW.COMPONENTS.index("ttc"), RW.COMPONENTS.index("nc")))
    check(bad, "U_columns_TTC_is_label_column_3_and_NC_is_column_0_and_EP_DAC_C_are_not_read", columns)
    check(bad, "U_the_columns_are_READ_from_rl_reward_COMPONENTS_not_typed", lambda: "rl_reward.COMPONENTS.index(c)" in _ast.unparse(next(n for n in C.tree.body if isinstance(n, _ast.FunctionDef) and n.name == "epo_con_cols")))

    def term_values():
        st = moved(6)
        A, B, base, (cxy, sctx, tg) = st
        an, ns, _d, _cl = mk(False)
        parts: dict = {}
        loss = M.epo_con_term(B(cxy, sctx), A(cxy, sctx), tg, rule_v1, 0.1, 0.0, {"ttc": 0.7, "nc": 0.3}, parts)
        with torch.no_grad():
            piB = torch.softmax(torch.log(rule_v1(torch.sigmoid(B(cxy, sctx))) + 1e-6) / 0.1, 1)
            piA = torch.softmax(torch.log(rule_v1(torch.sigmoid(A(cxy, sctx))) + 1e-6) / 0.1, 1)
            gt = (piA * tg[..., 3]).sum(1).mean() - (piB * tg[..., 3]).sum(1).mean()
            gn = (piA * tg[..., 0]).sum(1).mean() - (piB * tg[..., 0]).sum(1).mean()
        want = 0.7 * float(gt) + 0.3 * float(gn)
        return abs(float(loss) - want) < 1e-6 and abs(parts["g_ttc"] - float(gt)) < 1e-6 and abs(parts["g_nc"] - float(gn)) < 1e-6 and float(loss) != 0.0 and parts["con"] == float(loss)
    check(bad, "U_term_equals_an_independent_statement_of_the_math_L_con_and_its_logged_parts", term_values)

    # HARDENING (build 2026-10-17, after the sweep found M16 caught by the PIN only): eps = 0 in every registered arm, so no behavioural check read the slack at all.  This one does, at a NON-zero eps:
    # (a) through epo_con_term the violation of each component and the term shift by EXACTLY eps (and sum_k lambda_k * eps); (b) through the REAL train_losses (call site `a.epo_con_eps`) B's loss shifts by
    # eps * sum_k lambda_k (lambda0 0.5 on both => 0.07) and the logged violation by eps.  A call site that passes a literal 0.0, or a term that drops eps, both go RED here.
    def eps_shift():
        A, B, base, (cxy, sctx, tg) = moved(6)
        lam = {"ttc": 0.7, "nc": 0.3}
        p0, p1 = {}, {}
        l0 = M.epo_con_term(B(cxy, sctx), A(cxy, sctx), tg, rule_v1, 0.1, 0.0, lam, p0)
        l1 = M.epo_con_term(B(cxy, sctx), A(cxy, sctx), tg, rule_v1, 0.1, 0.07, lam, p1)
        direct = (float(l0) != float(l1) and abs(float(l0) - float(l1) - 0.07) < 1e-6 and abs(p0["g_ttc"] - p1["g_ttc"] - 0.07) < 1e-6 and abs(p0["g_nc"] - p1["g_nc"] - 0.07) < 1e-6)
        outs = []
        for e in (0.0, 0.07):
            an_, ns_, d_, cl_ = mk(False, dual=M.EpoDual(("ttc", "nc"), e, 4.0, 10.0, 0.5, temp=0.1), epo_con_eps=e)
            with torch.no_grad():
                sx0_ = base(cxy, sctx)
            lA_, lB_, ln_, ep_ = ns_["train_losses"](sx0_, A(cxy, sctx), B(cxy, sctx), tg, ROWS, 5)
            outs.append((float(lB_), dict(cl_)))
        through = (abs(outs[0][0] - outs[1][0] - 0.07) < 1e-5 and abs(outs[0][1]["g_ttc"] - outs[1][1]["g_ttc"] - 0.07) < 1e-6 and abs(outs[0][1]["g_nc"] - outs[1][1]["g_nc"] - 0.07) < 1e-6)
        return direct and through
    check(bad, "U_eps_shifts_the_violation_and_the_term_by_exactly_eps_through_epo_con_term_and_through_train_losses", eps_shift)

    # ---------------------------------------------------------------- U: C1 arm A is untouched; the term is live through the REAL train_losses
    def c1(lane):
        n_out = 7 if lane else 6
        A, B, base, (cxy, sctx, tg) = moved(n_out)
        an, ns, d, cl = mk(lane)
        d.lam = {"ttc": 1.0, "nc": 1.0}
        with torch.no_grad():
            sx0 = base(cxy, sctx)
        sxA, sxB = A(cxy, sctx), B(cxy, sctx)
        on = ns["train_losses"](sx0, sxA, sxB, tg, ROWS, 5, lam_pg=0.0)
        keep = d.comps
        d.comps = ()
        ref = ns["train_losses"](sx0, sxA, sxB, tg, ROWS, 5, lam_pg=0.0)
        d.comps = keep
        term = on[1] - ref[1]
        no_path = torch.autograd.grad(term, list(A.parameters()), retain_graph=True, allow_unused=True)
        term.backward(retain_graph=True)
        gA, gB = M.module_grad_abs(A), M.module_grad_abs(B)
        for q in A.parameters():
            q.grad = None
        on[0].backward()                                                                      # the probe's own control: it CAN read non-zero on A
        ctrl = M.module_grad_abs(A)
        return (all(g is None for g in no_path) and all(v == 0.0 for v in gA.values()) and all(v > 0.0 and np.isfinite(v) for v in gB.values()) and all(v > 0.0 for v in ctrl.values())
                and float(on[0]) == float(ref[0]) and float(term) != 0.0)
    check(bad, "U_C1_through_train_losses_A_gets_EXACTLY_0_from_L_con_B_is_live_on_every_module_A_BCE_control_reads_nonzero_6col", lambda: c1(False))
    check(bad, "U_C1_through_train_losses_A_gets_EXACTLY_0_from_L_con_B_is_live_on_every_module_A_BCE_control_reads_nonzero_lane", lambda: c1(True))

    def con_adds_exactly_ct_to_B_only():
        A, B, base, (cxy, sctx, tg) = moved(6)
        an, ns, d, cl = mk(False)
        d.lam = {"ttc": 0.5, "nc": 2.0}
        with torch.no_grad():
            sx0 = base(cxy, sctx)
        sxA, sxB = A(cxy, sctx), B(cxy, sctx)
        on = ns["train_losses"](sx0, sxA, sxB, tg, ROWS, 5)
        d.comps = ()
        off = ns["train_losses"](sx0, sxA, sxB, tg, ROWS, 5)
        want = M.epo_con_term(sxB, sxA, tg, ns["sel_score"], 0.1, 0.0, {"ttc": 0.5, "nc": 2.0})
        return (torch.equal(on[0], off[0]) and torch.equal(on[2], off[2]) and torch.equal(on[3], off[3]) and torch.equal(on[1], off[1] + want) and float(want) != 0.0)
    check(bad, "U_the_constraint_adds_exactly_L_con_to_lB_and_changes_nothing_else", con_adds_exactly_ct_to_B_only)

    def off_ignores_lam_and_skips_the_term():
        A, B, base, (cxy, sctx, tg) = moved(6)
        an, ns, d, cl = mk(False, dual=M.EpoDual((), 0.0, 4.0, 10.0, 0.0))
        with torch.no_grad():
            sx0 = base(cxy, sctx)
        ns["train_losses"](sx0, A(cxy, sctx), B(cxy, sctx), tg, ROWS, 5)
        return not cl and not d.active
    check(bad, "U_an_OFF_dual_never_calls_the_term", off_ignores_lam_and_skips_the_term)

    # ---------------------------------------------------------------- U: I4 the dual
    def dual_math():
        d = M.EpoDual(("ttc", "nc"), 0.0, 4.0, 10.0, 0.0, temp=0.1)
        for _ in range(10):
            d.step({"n": 8, "g_ttc": 8 * 2.0 ** -7, "g_nc": 8 * 2.0 ** -7})
        reach = d.lam == {"ttc": 0.3125, "nc": 0.3125} and d.updates == 10 and d.pos_updates == 9 and d.cap_hits == 0
        d.step({"n": 4, "g_ttc": -4.0, "g_nc": 4.0 * 1e6})
        edge = d.lam == {"ttc": 0.0, "nc": 10.0} and d.cap_hits == 1 and d.lam_seen_max == {"ttc": 0.3125, "nc": 10.0}
        mean = M.EpoDual(("ttc",), 0.0, 1.0, 10.0, 0.0)
        mean.step({"n": 2, "g_ttc": 1.0})
        e = M.EpoDual(("ttc",), 0.0, 4.0, 10.0, 0.0)
        try:
            e.step({"n": 0})
            raised = False
        except ValueError:
            raised = True
        return reach and edge and mean.lam == {"ttc": 0.5} and raised
    check(bad, "U_I4_the_dual_arithmetic_is_exact_0.3125_floor_0.0_cap_lam_max_and_the_mean_over_micro_batches", dual_math)

    def dual_state():
        d = M.EpoDual(("ttc", "nc"), 0.0, 4.0, 10.0, 0.25)
        d.step({"n": 2, "g_ttc": 0.3, "g_nc": -0.1})
        e = M.EpoDual(("ttc", "nc"), 0.0, 4.0, 10.0, 0.25)
        e.load_state_dict(d.state_dict())
        bf = d.built_facts()
        return (e.state_dict() == d.state_dict() and set(bf) == {"mode", "comps", "cols", "eps", "lr", "max", "lam0", "ref", "temp"} and bf["mode"] == "g5" and bf["comps"] == ["ttc", "nc"]
                and bf["cols"] == [3, 0] and bf["lam0"] == 0.25 and bf["ref"] == "armA_detached" and M.EpoDual((), 0.0, 4.0, 10.0, 0.0).built_facts()["mode"] == "off"
                and {"lam", "update", "cap_hits"} <= set(d.state_dict()))
    check(bad, "U_the_dual_state_round_trips_and_the_BUILT_facts_are_read_off_the_object", dual_state)
    check(bad, "U_built_probe_reads_armA_detached", lambda: M.epo_con_reference_built() == "armA_detached")

    # ---------------------------------------------------------------- U: G13 on the REAL con_probe closure with tiny CPU scorers
    def g13_unit(lane):
        n_out = 7 if lane else 6
        A, B, base = fresh(n_out)
        cxy, sctx, tg = TE.micro_data(n_out, 1, 1)[0]
        with torch.no_grad():
            sx0 = base(cxy, sctx)
        fwd = lambda batch: (sx0, A(cxy, sctx), B(cxy, sctx), tg)          # noqa: E731  (a FRESH forward per call, as the real fwd)
        an, ns, d, cl = mk(lane, names=CL + ["con_probe"], fwd=fwd)
        g = M.g13_epo_con_live_and_identity(A, B, lambda mode: ns["con_probe"](None, ROWS, mode), lambda: M.epo_con_dual_reach(d), lambda: M.epo_con_ckpt_roundtrip(d), kl_tol=0.0, names=SC, temp=d.temp)
        return g, d
    for lane in (False, True):
        def g13_green(lane=lane):
            g, d = g13_unit(lane)
            return (g["ok"] is True and all(g["live"].values()) and g["controls_exactly_zero"] and g["A_untouched"] and g["A_control_reads_nonzero"] and g["identity_ok"]
                    and g["dual_ok"] and g["checkpoint"]["block_equal"] and not g["mutate"] and g["value_lam0"] == 0.0 and g["value_off"] == 0.0 and d.lam == {"ttc": 0.0, "nc": 0.0})
        check(bad, f"U_G13_green_on_the_real_probe_{'lane' if lane else '6col'}", g13_green)

    def g13_restores_the_run_state():
        g, d = g13_unit(False)
        return d.comps == ("ttc", "nc") and d.lam == {"ttc": 0.0, "nc": 0.0} and d.updates == 0 and d.active and d.temp == 0.1          # PICKCON: the T2 probe restores the dual's temperature too
    check(bad, "U_the_G13_probe_leaves_the_runs_dual_exactly_as_it_found_it", g13_restores_the_run_state)

    # ---------------------------------------------------------------- refusals and the parser
    REQ = ["--ckpt", "x", "--targets", "x", "--images", "x", "--calib", "x", "--onpolicy", "x", "--heldout", "x", "--heldout-logs", "x", "--out", "x"]
    EPO_OK = ["--epo", "--pdm-labels-train", "d", "--require-pdm", "--b-mode", "listnet", "--lam", "0", "--epo-terms-train", "t"]

    def refusals(extra, base=None):
        ap = M.build_parser()
        a = ap.parse_args(REQ + (EPO_OK if base is None else base) + extra)
        return M.epo_con_refusals(a, ap), M.epo_refusals(a, ap)
    ok_ = lambda extra, base=None: refusals(extra, base)[0] == []          # noqa: E731
    check(bad, "U_refusal_a_complete_g5_argv_is_clean_and_so_is_off_named_and_the_default", lambda: ok_(CON_ON.split()) and ok_(["--epo-con", "off"]) and ok_([]) and ok_(["--epo-con", "g5", "--epo-con-lr", "0", "--epo-con-lam0", "0.5"]))
    check(bad, "U_refusal_g5_without_epo_is_refused", lambda: len(refusals(["--epo-con", "g5"], base=[])[0]) == 1 and "--epo is not" in refusals(["--epo-con", "g5"], base=[])[0][0])
    check(bad, "U_refusal_a_con_weight_without_epo_is_refused", lambda: any("epo_con_eps" in m for m in refusals(["--epo-con-eps", "0.1"], base=[])[0]))
    check(bad, "U_refusal_a_con_weight_while_off_is_refused_because_the_term_is_OFF", lambda: any("--epo-con is off" in m for m in refusals(["--epo-con", "off", "--epo-con-lr", "2"])[0])
          and any("--epo-con is off" in m for m in refusals(["--epo-con-lam0", "0.5"])[0]))
    check(bad, "U_refusal_g5_with_eta_0_and_lambda0_0_is_INERT_by_construction", lambda: any("INERT" in m for m in refusals(["--epo-con", "g5", "--epo-con-lr", "0"])[0])
          and not any("INERT" in m for m in refusals(["--epo-con", "g5", "--epo-con-lr", "0", "--epo-con-lam0", "1"])[0]))
    check(bad, "U_refusal_out_of_range_values", lambda: any("eps" in m for m in refusals(["--epo-con", "g5", "--epo-con-eps", "-0.1"])[0]) and any("lr" in m for m in refusals(["--epo-con", "g5", "--epo-con-lr", "-1"])[0])
          and any("max" in m for m in refusals(["--epo-con", "g5", "--epo-con-max", "0"])[0]) and any("lam0" in m for m in refusals(["--epo-con", "g5", "--epo-con-lam0", "11"])[0])
          and any("lam0" in m for m in refusals(["--epo-con", "g5", "--epo-con-lam0", "-1"])[0]) and any("eps" in m for m in refusals(["--epo-con", "g5", "--epo-con-eps", "nan"])[0]))
    check(bad, "U_epo_refusals_is_blind_to_the_new_levers_it_stays_byte_unchanged", lambda: refusals(CON_ON.split())[1] == [])

    def parser_rejects():
        try:
            M.build_parser().parse_args(REQ + ["--epo-con", "g6"])
        except SystemExit as e:
            return e.code == 2
        return False
    with contextlib.redirect_stderr(io.StringIO()):
        check(bad, "U_parser_rejects_a_mode_outside_the_set", parser_rejects)

    # ---------------------------------------------------------------- R (static): argparse -> main -> dual -> train_losses -> epo_con_term -> lB -> backward; the loop; the rows
    msrc = _ast.unparse(C.main)
    tl = _ast.unparse(C.nested["train_losses"])
    loop = next(n for n in _ast.walk(C.main) if isinstance(n, _ast.For) and _ast.unparse(n.iter) == "dl")
    stmts = [_ast.unparse(s) for s in loop.body]

    def idx(pred):
        return next((i for i, s in enumerate(stmts) if pred(s)), None)
    check(bad, "R_hop_parser_flags", lambda: all(any(x.dest == d_ and x.option_strings == ["--" + d_.replace("_", "-")] for x in M.build_parser()._actions) for d_ in M.EPO_CON_DESTS)
          and tuple(next(x for x in M.build_parser()._actions if x.dest == "epo_con").choices) == ("off", "g5"))
    check(bad, "R_hop_main_calls_epo_con_refusals_beside_epo_refusals_before_touching_disk", lambda: "bad_cli = epo_refusals(a, ap) + epo_con_refusals(a, ap)" in msrc
          and msrc.index("epo_con_refusals(a, ap)") < msrc.index("os.makedirs(a.out"))
    check(bad, "R_hop_main_builds_the_dual_from_the_five_flags", lambda: "dual = EpoDual(EPO_CON_TERMS.get(a.epo_con, ()), a.epo_con_eps, a.epo_con_lr, a.epo_con_max, a.epo_con_lam0," in msrc)
    check(bad, "R_hop_the_data_event_records_the_BUILT_facts_of_the_dual", lambda: "info['epo_con'] = dual.built_facts()" in msrc)

    def tl_block():
        fn = C.nested["train_losses"]
        epo_if = next(s for s in fn.body if isinstance(s, _ast.If) and _ast.unparse(s.test) == "a.epo")
        con_if = next((s for s in epo_if.body if isinstance(s, _ast.If) and _ast.unparse(s.test) == "dual.active"), None)
        return (con_if is not None and [_ast.unparse(s) for s in con_if.body] == ["ct = epo_con_term(sxB, sxA, tg, sel_score, dual.temp, a.epo_con_eps, dual.lam, con_last)", "lB = lB + ct"]
                and "lA = lA + " not in tl and "return (lA, lB, ln, ep)" in tl)
    check(bad, "R_hop_train_losses_adds_the_term_to_lB_inside_the_epo_branch_with_sxA_and_the_dual_s_lambda", tl_block)
    check(bad, "R_hop_the_loop_backwards_lA_plus_lB", lambda: "((lA + lB) / a.accum).backward()" in msrc)
    check(bad, "R_hop_the_loop_accumulates_the_micro_batch_parts_for_the_dual", lambda: idx(lambda s: "con_acc[k_] = con_acc.get(k_, 0.0) + v_" in s and "con_acc['n'] += 1" in s and "if dual.active" in s) is not None
          and idx(lambda s: "con_acc[k_]" in s) < idx(lambda s: s.startswith("micro += 1")))

    def order():
        i_opt = idx(lambda s: s.startswith("for o, ps in"))
        i_step = idx(lambda s: "dual.step(con_acc)" in s)
        i_upd = idx(lambda s: s == "upd += 1")
        return None not in (i_opt, i_step, i_upd) and i_opt < i_step < i_upd
    check(bad, "R_hop_dual_step_sits_between_the_optimiser_step_and_upd_plus_1", order)
    check(bad, "R_hop_the_rows_carry_the_dual_and_the_constraint_s_outcomes", lambda: all(f"'{k}'" in msrc for k in (
        "con", "con_lam_ttc", "con_lam_nc", "con_g_ttc", "con_g_nc", "con_cA_ttc", "con_cB_ttc", "con_cA_nc", "con_cB_nc", "con_p1_gap_ttc", "con_cap_hits", "con_hinge_ttc", "con_hinge_nc")))
    check(bad, "R_hop_states_file_and_checkpoint_meta_carry_the_dual", lambda: "'EPO_CON': dual.state_dict()" in msrc and "meta['epo_con'] = dual.state_dict()" in msrc)
    check(bad, "R_hop_G13_is_in_the_preflight_and_runs_the_run_s_own_assembly", lambda: "gates['G13_epo_con_live_and_identity'] = g13_epo_con_live_and_identity(model, armB, lambda mode: con_probe(batch, prow, mode)" in msrc
          and "train_losses(s0, sA_, sB_, tg_, rows, 0, lam_pg=0.0)" in _ast.unparse(C.nested["con_probe"]))
    check(bad, "R_hop_PASS_file_is_never_written_under_a_regression_switch", lambda: "AUX_MUTATE_ENVS + EPO_CON_MUTATE_ENVS" in msrc and M.EPO_CON_MUTATE_ENVS == ("REFE_RL_MUTATE_CON_DETACH", "REFE_RL_MUTATE_CON_ATTACH_A"))

    # ---------------------------------------------------------------- E: the REAL main() end to end
    def rec_hook():
        rec = {"steps": []}

        def hook(sm):
            orig = sm.EpoDual.step

            def w(self, acc):
                before = dict(self.lam)
                orig(self, acc)
                rec["steps"].append({"acc": dict(acc), "before": before, "after": dict(self.lam), "eta": self.lr, "max": self.lam_max})
            sm.EpoDual.step = w
        return rec, hook

    def norm(rows):
        drop = {"at", "s_per_update", "hours", "s_per_microbatch", "est_hours_per_epoch", "gpu_mem_gb", "model_A", "model_B"}
        out = []
        for r in rows:
            r = {k: v for k, v in r.items() if k not in drop}
            if r.get("event") == "data":
                r = {k: v for k, v in r.items() if k != "epo_con"}
            out.append(json.dumps(r, sort_keys=True))
        return out

    def full_equal(x, y):
        return x is not None and y is not None and set(x) == set(y) and all(torch.equal(x[k], y[k]) for k in x)

    runs, recs, kept = {}, {}, {}
    with tempfile.TemporaryDirectory() as td:
        jobs = [("on", sf_src, SMOKE_ON, True, 1.0), ("on_l", sf_src, SMOKE_ON_L, True, 1.0), ("off_named", sf_src, ("--epo-con", "off"), True, 1.0), ("off_default", sf_src, (), True, 1.0),
                ("tip", tip_src, (), False, 1.0)]
        for nm, src_, extra, hooked, lam_ in jobs:
            d_ = os.path.join(td, nm)
            os.makedirs(d_)
            rc, hk = rec_hook()
            try:
                runs[nm] = TE.smoke(src_, d_, epo=True, epo_lam=lam_, native7=True, terms=True, tag=f"so_{tag}_{nm}", extra=extra, hook=hk if hooked else None)
                for fn in ("scorer_states_last.pt", "model_sft_B.pt"):
                    p_ = os.path.join(d_, "out", fn)
                    kept[(nm, fn)] = torch.load(p_, weights_only=False) if os.path.exists(p_) else None
            except BaseException as e:                                          # noqa: BLE001 -- a run that raises is RED, and said so
                runs[nm] = {"error": f"{type(e).__name__}: {str(e)[:80]}", "pre": None, "run": None}
                bad.append(f"E_run_raised[{nm}]: {runs[nm]['error']}")
            recs[nm] = rc
        d6 = os.path.join(td, "on6")
        os.makedirs(d6)
        try:
            runs["on6"] = TE.smoke(sf_src, d6, epo=True, native7=False, terms=True, tag=f"so_{tag}_on6", extra=SMOKE_ON)
        except BaseException as e:                                              # noqa: BLE001
            runs["on6"] = {"error": str(e)[:80], "pre": None, "run": None}
            bad.append(f"E_run_raised[on6]: {runs['on6']['error']}")
        for nm, kw, extra in (("refuse_noepo", dict(epo=False, native7=True, terms=False), SMOKE_ON), ("refuse_inert", dict(epo=True, native7=True, terms=True), ("--epo-con", "g5", "--epo-con-lr", "0")),
                              ("refuse_off_weight", dict(epo=True, native7=True, terms=True), ("--epo-con", "off", "--epo-con-lr", "2"))):
            dn = os.path.join(td, nm)
            os.makedirs(dn)
            try:
                runs[nm] = TE.smoke(sf_src, dn, tag=f"so_{tag}_{nm}", extra=extra, **kw)
            except BaseException as e:                                          # noqa: BLE001
                runs[nm] = {"error": str(e)[:80], "pre": None, "run": None}
                bad.append(f"E_run_raised[{nm}]: {runs[nm]['error']}")
        # the regression arms: each deliberate regression must make the PREFLIGHT fail (no PASS file, rc 3)
        for nm, envn in (("reg_detach", "REFE_RL_MUTATE_CON_DETACH"), ("reg_attachA", "REFE_RL_MUTATE_CON_ATTACH_A")):
            dr = os.path.join(td, nm)
            os.makedirs(dr)
            try:
                with env(**{envn: "1"}):
                    runs[nm] = TE.smoke(sf_src, dr, epo=True, native7=True, terms=True, tag=f"so_{tag}_{nm}", extra=SMOKE_ON)
            except BaseException as e:                                          # noqa: BLE001
                runs[nm] = {"error": str(e)[:80], "pre": None, "run": None}
                bad.append(f"E_run_raised[{nm}]: {runs[nm]['error']}")

    R = runs

    def ok_run(r):
        return r.get("pre") == 0 and r.get("run") == 0 and r.get("A") is not None and r.get("B") is not None and r.get("pass_json") is not None
    data_of = lambda r: [x for x in r["rows"]["sft.jsonl"] if x.get("event") == "data"]          # noqa: E731
    pre_data_of = lambda r: [x for x in r["rows"]["preflight.jsonl"] if x.get("event") == "data"]          # noqa: E731
    trains = lambda r: [x for x in r["rows"]["sft.jsonl"] if x.get("event") == "train"]          # noqa: E731
    G13K = "G13_epo_con_live_and_identity"
    for nm in ("on", "on_l", "off_named", "off_default"):
        check(bad, f"E_runs_to_completion_preflight_PASS[{nm}]", lambda nm=nm: ok_run(R[nm]) and "G11_epo_live_and_identity" in R[nm]["pass_json"]["gates"])
    check(bad, "E_runs_to_completion_preflight_PASS[on6]", lambda: ok_run(R["on6"]))
    check(bad, "E_G13_is_in_the_preflight_of_a_constrained_arm_and_ONLY_there", lambda: all(G13K in R[nm]["pass_json"]["gates"] for nm in ("on", "on_l", "on6"))
          and all(G13K not in R[nm]["pass_json"]["gates"] for nm in ("off_named", "off_default", "tip")))
    ns_on = argparse.Namespace(epo_con="g5", epo_con_eps=0.0, epo_con_lr=4.0, epo_con_max=10.0, epo_con_lam0=0.0, epo_temp=0.1, epo_con_temp=None)
    ns_on_l = argparse.Namespace(**{**vars(ns_on), "epo_con_lam0": 0.5})
    import gate_checks as GC0
    check(bad, "E_G13_raw_fields_pass_the_gate_s_own_re_judging", lambda: GC0.judge_g13(R["on"]["pass_json"]["gates"][G13K], ns_on) == [] and GC0.judge_g13(R["on_l"]["pass_json"]["gates"][G13K], ns_on_l) == []
          and GC0.judge_g13(R["on6"]["pass_json"]["gates"][G13K], ns_on) == [])
    check(bad, "E_G13_on_a_real_preflight_every_module_live_A_untouched_controls_exactly_zero", lambda: (lambda g: all(v > 0.0 for v in g["per_module_on"].values()) and set(g["per_module_on"]) == set(SC)
          and all(v == 0.0 for v in g["A_per_module_on"].values()) and all(v > 0.0 for v in g["A_bce_control"].values()) and g["value_lam0"] == 0.0 and g["value_off"] == 0.0
          and g["dual_reach"]["got"] == {"ttc": 0.3125, "nc": 0.3125} and g["identity_ok"] and g["mutate"] == [])(R["on"]["pass_json"]["gates"][G13K]))
    want_facts = {"mode": "g5", "comps": ["ttc", "nc"], "cols": [3, 0], "eps": 0.0, "lr": 4.0, "max": 10.0, "lam0": 0.0, "ref": "armA_detached", "temp": 0.1}
    check(bad, "E_data_event_records_the_BUILT_facts_in_the_preflight_log_and_the_run_log", lambda: data_of(R["on"])[0]["epo_con"] == want_facts and pre_data_of(R["on"])[0]["epo_con"] == want_facts
          and data_of(R["on_l"])[0]["epo_con"] == {**want_facts, "lam0": 0.5} and data_of(R["off_named"])[0]["epo_con"] == {"mode": "off", "comps": [], "cols": [], "eps": 0.0, "lr": 4.0, "max": 10.0,
                                                                                                                              "lam0": 0.0, "ref": "armA_detached", "temp": 0.1}
          and data_of(R["off_default"])[0]["epo_con"]["mode"] == "off" and "epo_con" not in data_of(R["tip"])[0])
    check(bad, "E_G_DVB_reads_the_real_data_event_PASS_and_FAIL_when_the_built_side_differs", lambda: (lambda GC, blk: all(r["ok"] for r in GC.scorer_dvb_rows(ns_on_epo(), {"epo": {}, "epo_con": blk, "b_mode": "listnet", "b_compw": "1,1,1,1,1,1"},
                                                                                                                             {"epo_con_block": True}) if r["dest"].startswith("epo_con"))
                                                                                       and not all(r["ok"] for r in GC.scorer_dvb_rows(ns_on_epo(), {"epo": {}, "epo_con": {**blk, "cols": [2, 0]}, "b_mode": "listnet", "b_compw": "1,1,1,1,1,1"},
                                                                                                                                       {"epo_con_block": True}) if r["dest"].startswith("epo_con")))(GC0, data_of(R["on"])[0]["epo_con"]))

    def steps_of(nm):
        return [s for s in recs[nm]["steps"] if s["acc"].get("n") == 2]          # the loop's calls (accum 2); the preflight's synthetic dual probe feeds n = 8
    check(bad, "E_dual_step_is_called_once_per_optimiser_update_with_every_micro_batch", lambda: len(steps_of("on_l")) == 25 and all(s["acc"]["n"] == 2 for s in steps_of("on_l")))

    def closed_form():
        for s in steps_of("on_l"):
            for c in ("ttc", "nc"):
                raw = s["before"][c] + s["eta"] * (s["acc"]["g_" + c] / s["acc"]["n"])
                if s["after"][c] != min(s["max"], max(0.0, raw)):
                    return False
        return True
    check(bad, "E_lambda_follows_the_closed_form_from_the_logged_violations", lambda: closed_form() and len(steps_of("on_l")) > 0)
    check(bad, "E_lambda_is_threaded_the_next_update_starts_from_the_last_one", lambda: all(a_["after"] == b_["before"] for a_, b_ in zip(steps_of("on_l")[:-1], steps_of("on_l")[1:])))
    check(bad, "E_lambda_MOVES_off_lambda0_so_the_dual_step_reaches_the_term", lambda: any(s["after"] != s["before"] for s in steps_of("on_l")) and steps_of("on_l")[-1]["after"] != {"ttc": 0.5, "nc": 0.5})
    need = ("con", "con_lam_ttc", "con_lam_nc", "con_g_ttc", "con_g_nc", "con_cA_ttc", "con_cB_ttc", "con_cA_nc", "con_cB_nc", "con_p1_gap_ttc", "con_cap_hits", "con_hinge_ttc", "con_hinge_nc")
    check(bad, "E_train_rows_carry_the_con_columns_on_a_constrained_arm", lambda: bool(trains(R["on_l"])) and all(all(k in r for k in need) for r in trains(R["on_l"])))
    check(bad, "E_train_rows_carry_NO_con_column_off_their_presence_would_be_a_FAIL", lambda: all(not any(k.startswith("con") for k in r) for nm in ("off_named", "off_default", "tip") for r in trains(R[nm]))
          and bool(trains(R["off_named"])))
    check(bad, "E_the_logged_lambda_is_the_dual_s_after_the_last_update", lambda: trains(R["on_l"])[-1]["con_lam_ttc"] == steps_of("on_l")[-1]["after"]["ttc"]
          and trains(R["on_l"])[-1]["con_lam_nc"] == steps_of("on_l")[-1]["after"]["nc"])
    check(bad, "E_the_states_file_and_the_checkpoint_meta_hold_the_dual", lambda: kept[("on_l", "scorer_states_last.pt")]["EPO_CON"]["update"] == 25 and set(kept[("on_l", "scorer_states_last.pt")]["EPO_CON"]["lam"]) == {"ttc", "nc"}
          and kept[("on_l", "scorer_states_last.pt")]["EPO_CON"]["lam"] == steps_of("on_l")[-1]["after"] and kept[("on_l", "model_sft_B.pt")]["meta"]["epo_con"]["update"] == 25
          and "EPO_CON" not in kept[("off_named", "scorer_states_last.pt")] and "epo_con" not in kept[("off_named", "model_sft_B.pt")]["meta"])
    check(bad, "E_the_constraint_never_reaches_the_control_arm_A_is_bit_identical_to_the_off_run", lambda: full_equal(R["on_l"]["A"], R["off_named"]["A"]) and full_equal(R["on"]["A"], R["off_named"]["A"]))
    check(bad, "E_the_constraint_moves_B", lambda: not full_equal(R["on_l"]["B"], R["off_named"]["B"]) and not TE.scorer_equal(R["on_l"]["A"], R["on_l"]["B"]))
    check(bad, "E_train_rows_bceA_equal_between_the_constrained_and_the_off_run", lambda: [r["bceA"] for r in trains(R["on_l"])] == [r["bceA"] for r in trains(R["off_named"])]
          and [r["lossB"] for r in trains(R["on_l"])] != [r["lossB"] for r in trains(R["off_named"])])
    check(bad, "E_off_named_equals_off_default_bit_for_bit", lambda: ok_run(R["off_named"]) and ok_run(R["off_default"]) and full_equal(R["off_named"]["A"], R["off_default"]["A"])
          and full_equal(R["off_named"]["B"], R["off_default"]["B"]) and norm(R["off_named"]["rows"]["sft.jsonl"]) == norm(R["off_default"]["rows"]["sft.jsonl"]))
    check(bad, "E_off_is_bit_identical_to_the_R0c_tip_trainer_weights_and_every_logged_number", lambda: ok_run(R["tip"]) and full_equal(R["tip"]["A"], R["off_default"]["A"]) and full_equal(R["tip"]["B"], R["off_default"]["B"])
          and norm(R["tip"]["rows"]["sft.jsonl"]) == norm(R["off_default"]["rows"]["sft.jsonl"]) and norm(R["tip"]["rows"]["preflight.jsonl"]) == norm(R["off_default"]["rows"]["preflight.jsonl"])
          and R["tip"]["pass_json"]["gates"] == R["off_default"]["pass_json"]["gates"])
    for nm in ("refuse_noepo", "refuse_inert", "refuse_off_weight"):
        check(bad, f"E_refuses_before_touching_disk[{nm}]", lambda nm=nm: R[nm].get("pre") == 4 and R[nm].get("run") == 4 and "REFUSED" in R[nm]["out_pre"] and not R[nm]["out_exists"])
    for nm in ("reg_detach", "reg_attachA"):
        check(bad, f"E_regression_arm_makes_G13_FAIL_and_no_PASS_file[{nm}]", lambda nm=nm: R[nm].get("pre") == 3 and R[nm].get("pass_json") is None and R[nm].get("run") == 4
              and (lambda g: g is not None and g["ok"] is False and (any(v == 0.0 for v in g["per_module_on"].values()) if nm == "reg_detach" else any(v != 0.0 for v in g["A_per_module_on"].values()))
                   and bool(g["mutate"]))(next((r_ for r_ in R[nm]["rows"]["preflight.jsonl"] if r_.get("event") == "preflight"), {}).get("gates", {}).get(G13K)))
    return bad


def ns_on_epo():
    return argparse.Namespace(epo=True, epo_temp=0.1, epo_beta=0.1, epo_ramp=200, epo_lam=1.0, epo_lam_d=0.15, epo_w_prog=0.5, epo_k=False, epo_lam_k=0.05, epo_l=False, epo_lam_l=0.05, epo_ref="base",
                              b_mode="listnet", b_compw="1,1,1,1,1,1", aux_drv=False, epo_con="g5", epo_con_eps=0.0, epo_con_lr=4.0, epo_con_max=10.0, epo_con_lam0=0.0)


# ================================================================================================ part G: the gate (refe_rls1d) and G-DVB
def suite_g(lg_src: str, gc_src: str, tag: str, good: dict) -> list:
    """launch_gate imports gate_checks LAZILY (`import gate_checks as C` inside each function), so the module under test must BE sys.modules["gate_checks"] for the whole suite, or a mutation of
    gate_checks would be invisible to every consumer check (found 2026-10-17: G8 stayed green on the consumer checks until this swap)."""
    GC = S1B.load(gc_src, f"gc_so_{tag}", GC_PATH)
    real = sys.modules.get("gate_checks")
    sys.modules["gate_checks"] = GC
    try:
        return _suite_g(lg_src, GC, tag, good)
    finally:
        if real is not None:
            sys.modules["gate_checks"] = real
        else:
            sys.modules.pop("gate_checks", None)


def _suite_g(lg_src: str, GC, tag: str, good: dict) -> list:
    bad: list = []
    L = S1B.load(lg_src, f"lg_so_{tag}", LG_PATH)
    import scorer_finetune as S
    P, Pb, Pc, Pd = (L.PROFILES[k] for k in ("refe_rls1", "refe_rls1b", "refe_rls1c", "refe_rls1d"))
    A = {k: L.parse_args_string(v) for k, v in ARM_ARGS.items()}
    sha = lambda a: hashlib.sha256(json.dumps(a, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()          # noqa: E731

    def argv(arm, extra=""):
        return A[arm] + (extra.split() if extra else [])

    def drop(a, flag, nvals=1):
        i = a.index(flag)
        return a[:i] + a[i + 1 + nvals:]

    def refs(prof, arm, a):
        return L.profile_refusals({**prof, "declared_arm": arm} if arm else prof, a)

    def red(prof, arm, a, frag):
        return any(frag in r for r in refs(prof, arm, a))
    check(bad, "A_R0_argv_template_is_the_banked_R0_token_argv", lambda: sha(A["R0"]) == R0_ARGV_SHA256)

    r0, r0d, r1d, r1o = A["R0"], A["R0d"], A["R1d"], A["R1o"]
    def base_for(arm, seed):
        t = [x.replace("/arms/R0", f"/arms/{arm}") for x in r0]
        t[t.index("--seed") + 1] = str(seed)
        return t
    check(bad, "A_R0d_is_R0_s_argv_with_its_own_arm_paths_plus_the_five_flags_and_nothing_else", lambda: r0d == base_for("R0d", 0) + CON_ON.split())
    check(bad, "A_R1d_is_R0d_with_seed_1_and_its_own_arm_paths_only", lambda: r1d == base_for("R1d", 1) + CON_ON.split() and len(r1d) == len(r0d))
    check(bad, "A_R1o_is_R1d_s_base_with_the_five_flags_replaced_by_epo_con_off", lambda: r1o == base_for("R1o", 1) + ["--epo-con", "off"])
    # ---- the new profile accepts exactly its three arms
    check(bad, "G_clean_R0d", lambda: refs(Pd, "R0d", r0d) == [])
    check(bad, "G_clean_R1d", lambda: refs(Pd, "R1d", r1d) == [])
    check(bad, "G_clean_R1o", lambda: refs(Pd, "R1o", r1o) == [])
    for fl in ("--epo-con", "--epo-con-eps", "--epo-con-lr", "--epo-con-max", "--epo-con-lam0"):
        check(bad, f"G_R0d_{fl}_omitted_is_refused_and_says_NAMED", lambda fl=fl: red(Pd, "R0d", drop(r0d, fl), f"requires {fl} NAMED"))
        check(bad, f"G_R1d_{fl}_omitted_is_refused_and_says_NAMED", lambda fl=fl: red(Pd, "R1d", drop(r1d, fl), f"requires {fl} NAMED"))
    check(bad, "G_R0d_with_the_flag_omitted_altogether_also_says_the_registered_value_g5", lambda: red(Pd, "R0d", r0, "requires --epo-con g5") and red(Pd, "R0d", r0, "the parser DEFAULT"))
    check(bad, "G_R0d_with_off_named_is_refused", lambda: red(Pd, "R0d", argv("R0", "--epo-con off"), "requires --epo-con g5") and red(Pd, "R0d", argv("R0", "--epo-con off"), "requires --epo-con-eps NAMED"))
    for flag, val, want in (("--epo-con-eps", "0.01", "--epo-con-eps 0.0"), ("--epo-con-lr", "2", "--epo-con-lr 4.0"), ("--epo-con-max", "5", "--epo-con-max 10.0"), ("--epo-con-lam0", "0.5", "--epo-con-lam0 0.0")):
        check(bad, f"G_R0d_{flag}_value_is_registered", lambda flag=flag, val=val, want=want: red(Pd, "R0d", drop(r0d, flag) + [flag, val], f"requires {want}"))
    check(bad, "G_R0d_seed_1_and_R1d_seed_0_and_R1o_seed_0_are_refused", lambda: red(Pd, "R0d", drop(r0d, "--seed") + ["--seed", "1"], "registered with --seed 0")
          and red(Pd, "R1d", drop(r1d, "--seed") + ["--seed", "0"], "registered with --seed 1") and red(Pd, "R1o", drop(r1o, "--seed") + ["--seed", "0"], "registered with --seed 1"))
    check(bad, "G_R1o_with_g5_is_refused", lambda: red(Pd, "R1o", drop(r1o, "--epo-con") + ["--epo-con", "g5"], "requires --epo-con off"))
    check(bad, "G_R1o_without_epo_con_named_is_refused", lambda: red(Pd, "R1o", drop(r1o, "--epo-con"), "requires --epo-con NAMED"))
    check(bad, "G_R1o_with_a_con_weight_while_off_is_refused_by_the_trainer_s_own_rule", lambda: red(Pd, "R1o", r1o + ["--epo-con-lr", "2"], "--epo-con is off"))
    check(bad, "G_g5_with_eta_0_and_lambda0_0_is_refused_as_inert", lambda: red(Pd, "R0d", drop(r0d, "--epo-con-lr") + ["--epo-con-lr", "0"], "INERT"))
    check(bad, "G_the_other_levers_stay_registered", lambda: red(Pd, "R0d", argv("R0d", "--epo-lam 0.25"), "--epo-lam 1.0") and red(Pd, "R0d", argv("R0d", "--epo-ref armA"), "--epo-ref base")
          and red(Pd, "R0d", argv("R0d", "--epo-k --epo-l"), "CORE reward"))
    check(bad, "G_a_control_arm_with_epo_con_must_not_carry_it", lambda: red(Pd, None, argv("C0", "--epo-con off"), "must NOT carry") and red(P, "C0", argv("C0", "--epo-con off"), "must NOT carry"))
    check(bad, "G_C0_is_not_an_arm_of_refe_rls1d", lambda: red(Pd, "C0", A["C0"], "unknown arm"))
    # ---- every older profile keeps accepting what it accepted and now refuses any --epo-con but off
    check(bad, "G_rls1_rls1b_rls1c_accept_exactly_what_they_accepted", lambda: all(refs(P, a, A[a]) == [] for a in ("C0", "C1", "R0", "R1", "R2")) and all(refs(Pb, a, A[a]) == [] for a in ("R0b", "R1b"))
          and refs(Pb, "C1", A["C1"]) == [] and refs(Pc, "R0c", A["R0c"]) == [])
    check(bad, "G_older_profiles_accept_epo_con_off_named", lambda: refs(P, "R0", argv("R0", "--epo-con off")) == [] and refs(Pb, "R0b", argv("R0b", "--epo-con off")) == [] and refs(Pc, "R0c", argv("R0c", "--epo-con off")) == [])
    check(bad, "G_older_profiles_refuse_epo_con_g5", lambda: red(P, "R0", argv("R0", CON_ON), "requires --epo-con off") and red(P, "R1", argv("R1", CON_ON), "requires --epo-con off")
          and red(Pb, "R0b", argv("R0b", CON_ON), "requires --epo-con off") and red(Pc, "R0c", argv("R0c", CON_ON), "requires --epo-con off") and red(P, "R2", argv("R2", CON_ON), "requires --epo-con off"))
    check(bad, "G_the_older_profiles_do_not_accept_the_new_arms", lambda: red(P, "R0", r0d, "requires --epo-con off") and red(Pc, "R0c", r0d, "requires --epo-con off"))
    check(bad, "G_rls1d_does_not_accept_the_older_arms", lambda: red(Pd, "R0d", r0, "requires --epo-con NAMED") and red(Pd, "R0d", A["R0c"], "requires --epo-con NAMED"))
    # ---- the profile itself
    check(bad, "G_rls1d_is_rls1_with_its_own_arm_table_registry_must_name_terms_and_suite", lambda: all(Pd[k] == P[k] for k in ("checks", "data_flags", "forbidden_flags", "required_flags", "trainer", "parser_module",
                                                                                                                                 "trainer_src", "namespace_funcs", "runner", "arm_rules", "live_terms"))
          and Pd["arm_class"] == {"R0d": "R", "R1d": "R", "R1o": "R"} and Pd["arm_seeds"] == {"R0d": 0, "R1d": 1, "R1o": 1}
          and Pd["arm_epo_registered"] == {"R0d": {"epo_con": "g5", "epo_con_eps": 0.0, "epo_con_lr": 4.0, "epo_con_max": 10.0, "epo_con_lam0": 0.0}, "R1d": {"epo_con": "g5", "epo_con_eps": 0.0, "epo_con_lr": 4.0,
                                                                                                                                                          "epo_con_max": 10.0, "epo_con_lam0": 0.0},
                                           "R1o": {"epo_con": "off"}}
          and Pd["arm_must_name"]["R1o"] == ("--epo-con",) and len(Pd["arm_must_name"]["R0d"]) == 5 and "epo_registered" not in Pd and Pd["epo_con_block"] is True
          and all(S.build_parser().get_default(k) == v for k, v in Pd["arm_epo_registered"]["R0d"].items() if k != "epo_con"))
    check(bad, "G_the_older_profiles_keep_their_constants_unchanged", lambda: L.RLS1_EPO_REGISTERED == {"epo_temp": 0.1, "epo_beta": 0.1, "epo_ramp": 200, "epo_lam": 1.0, "epo_lam_d": 0.15, "epo_w_prog": 0.5, "epo_ref": "base"}
          and "arm_epo_registered" not in P and "epo_con_block" not in P)
    check(bad, "G_suite_has_the_safeobj_test_and_the_whole_older_suite", lambda: [e["script"] for e in Pd["suite"]] == ["test_rl_reward.py", "test_rl_epo.py", "test_launch_gate.py", "test_rl_s1b.py", "test_rl_r0c.py", "test_rl_safeobj.py"]
          and all(e["token"] in open(os.path.join(HERE, e["script"]), encoding="utf-8").read() for e in Pd["suite"]) and Pd["suite"][:5] == Pc["suite"])
    check(bad, "G_live_terms_a_constrained_arm_must_emit_the_con_columns_and_R1o_must_not", lambda: L.rls1_refusals({**Pd, "declared_arm": "R0d"}, r0d, S, S.build_parser())[1] == L.RLS1_EPO_TERMS + L.RLS1D_CON_TERMS
          and L.rls1_refusals({**Pd, "declared_arm": "R1o"}, r1o, S, S.build_parser())[1] == L.RLS1_EPO_TERMS and "con_lam_ttc" in L.RLS1D_CON_TERMS
          and L.rls1_refusals({**Pb, "declared_arm": "R0b"}, A["R0b"], S, S.build_parser())[1] == L.RLS1_EPO_TERMS)
    check(bad, "G_aux_s1_forbids_every_epo_flag_including_the_five_new_ones", lambda: set(o for x in S.build_parser()._actions for o in x.option_strings if o == "--epo" or o.startswith("--epo-")) == set(L.AUXS1_EPO_FLAGS)
          and set(L.AUXS1_EPO_FLAGS) <= set(L.AUXS1_FORBIDDEN) and set(L.RLS1D_CON_FLAGS) <= set(L.AUXS1_EPO_FLAGS))

    def pf_err(profile, arm):
        try:
            L._profile_for(argparse.Namespace(profile=profile, arm_name=arm), minting=True)
        except SystemExit as e:
            return str(e)
        return ""
    check(bad, "G_cli_arms_R0d_R1d_R1o_belong_to_refe_rls1d_only", lambda: all(pf_err("refe_rls1d", a) == "" for a in ("R0d", "R1d", "R1o")) and "is not an arm of refe_rls1d" in pf_err("refe_rls1d", "R0")
          and all("is not an arm of refe_rls1" in pf_err(p, "R0d") for p in ("refe_rls1", "refe_rls1b", "refe_rls1c")) and "mints for ONE named arm" in pf_err("refe_rls1d", None))

    def cli(*a):
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                rc = L.main(list(a))
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 2
        return rc, buf.getvalue()
    check(bad, "G_cli_refusals_R0d_clean_R0_refused_and_the_arm_name_is_a_valid_choice", lambda: cli("refusals", "--profile", "refe_rls1d", "--arm-name", "R0d", "--args", " ".join(r0d))[0] == 0
          and cli("refusals", "--profile", "refe_rls1d", "--arm-name", "R0d", "--args", " ".join(r0))[0] == 5 and cli("refusals", "--profile", "refe_rls1d", "--arm-name", "R1o", "--args", " ".join(r1o))[0] == 0)
    # ---- G-DVB: the argv's constraint is compared with what the trainer BUILT
    nsd = S.build_parser().parse_args(r0d)
    nso = S.build_parser().parse_args(r1o)
    BLK_ON = S.EpoDual(("ttc", "nc"), 0.0, 4.0, 10.0, 0.0, temp=0.1).built_facts()
    BLK_OFF = S.EpoDual((), 0.0, 4.0, 10.0, 0.0, temp=0.1).built_facts()

    def data(ns, blk, with_block=True):
        e = {k: getattr(ns, "epo_" + k) for k in GC.SCORER_EPO_KEYS if k != "ref"}
        e["ref"] = "base"
        d = {"event": "data", "b_mode": ns.b_mode, "b_compw": ns.b_compw, "epo": e}
        if with_block:
            d["epo_con"] = blk
        return d
    pdr = {**Pd, "declared_arm": "R0d"}
    pdo = {**Pd, "declared_arm": "R1o"}
    p0 = {**P, "declared_arm": "R0"}
    ev = lambda a, d, prof: GC.check_dvb_scorer(a, {"data": d}, prof)          # noqa: E731
    check(bad, "G_dvb_PASS_R0d_built_g5", lambda: ev(r0d, data(nsd, BLK_ON), pdr)["status"] == "PASS")
    check(bad, "G_dvb_PASS_R1o_built_off", lambda: ev(r1o, data(nso, BLK_OFF), pdo)["status"] == "PASS")
    check(bad, "G_dvb_FAIL_argv_g5_built_off", lambda: ev(r0d, data(nsd, BLK_OFF), pdr)["status"] == "FAIL" and any("--epo-con: declared 'g5' != built 'off'" in r for r in ev(r0d, data(nsd, BLK_OFF), pdr)["reasons"]))
    check(bad, "G_dvb_FAIL_argv_off_built_g5", lambda: ev(r1o, data(nso, BLK_ON), pdo)["status"] == "FAIL")
    check(bad, "G_dvb_FAIL_built_cols_EP_instead_of_TTC", lambda: ev(r0d, data(nsd, {**BLK_ON, "cols": [2, 0]}), pdr)["status"] == "FAIL"
          and any("--epo-con: declared [3, 0] != built [2, 0]" in r for r in ev(r0d, data(nsd, {**BLK_ON, "cols": [2, 0]}), pdr)["reasons"]))
    check(bad, "G_dvb_FAIL_built_comps_wrong", lambda: ev(r0d, data(nsd, {**BLK_ON, "comps": ["ep", "nc"]}), pdr)["status"] == "FAIL")
    check(bad, "G_dvb_FAIL_built_reference_attached", lambda: ev(r0d, data(nsd, {**BLK_ON, "ref": "armA_attached"}), pdr)["status"] == "FAIL")
    for k, v in (("eps", 0.01), ("lr", 2.0), ("max", 5.0), ("lam0", 0.5), ("temp", 0.05)):          # PICKCON added temp
        check(bad, f"G_dvb_FAIL_built_{k}_differs_from_the_argv", lambda k=k, v=v: ev(r0d, data(nsd, {**BLK_ON, k: v}), pdr)["status"] == "FAIL"
              and any(f"--epo-con-{k}: declared" in r for r in ev(r0d, data(nsd, {**BLK_ON, k: v}), pdr)["reasons"]))
    check(bad, "G_dvb_FAIL_the_block_missing_on_rls1d_even_for_an_off_arm", lambda: ev(r1o, data(nso, None, with_block=False), pdo)["status"] == "FAIL" and ev(r0d, data(nsd, None, with_block=False), pdr)["status"] == "FAIL")
    check(bad, "G_dvb_legacy_trainer_without_the_block_still_passes_on_the_older_profiles_when_off_and_fails_when_g5", lambda: ev(r0, data(S.build_parser().parse_args(r0), None, with_block=False), p0)["status"] == "PASS"
          and ev(argv("R0", CON_ON), data(S.build_parser().parse_args(argv("R0", CON_ON)), None, with_block=False), p0)["status"] == "FAIL")
    check(bad, "G_dvb_the_older_profiles_read_a_real_off_block", lambda: ev(r0, data(S.build_parser().parse_args(r0), BLK_OFF), p0)["status"] == "PASS"
          and ev(r0, data(S.build_parser().parse_args(r0), BLK_ON), p0)["status"] == "FAIL")

    def flag_without_entry():
        for fl in ("epo_con", "epo_con_eps", "epo_con_lr", "epo_con_max", "epo_con_lam0", "epo_con_temp"):          # PICKCON added epo_con_temp
            kept = GC.DVB_KINDS_SCORER.pop(fl)
            try:
                e = ev(r0d, data(nsd, BLK_ON), pdr)
                if not (e["status"] == "FAIL" and any("NO declared-vs-built entry" in r and fl in r for r in e["reasons"])):
                    return False
            finally:
                GC.DVB_KINDS_SCORER[fl] = kept
        return True
    check(bad, "G_a_flag_without_a_G_DVB_entry_is_refused_by_the_gate_itself_all_five", flag_without_entry)
    check(bad, "G_dvb_registry_classifies_the_five_flags_as_built_and_observes_them", lambda: all(GC.DVB_KINDS_SCORER.get(k) == "built" for k in ("epo_con", "epo_con_eps", "epo_con_lr", "epo_con_max", "epo_con_lam0", "epo_con_temp"))
          and GC.SCORER_EPO_CON_KEYS == ("mode", "comps", "cols", "eps", "lr", "max", "lam0", "ref", "temp") and GC.SCORER_EPO_CON_COMPS == {"g5": ("ttc", "nc")})

    def built_without_observer():
        for k in ("eps", "lr", "max", "lam0", "temp"):          # (PICKCON added temp) the numeric rows have a dest of their own; mode / comps / cols / ref share `epo_con`, so dropping ONE of those leaves the dest observed
            keep = GC.SCORER_EPO_CON_KEYS
            GC.SCORER_EPO_CON_KEYS = tuple(x for x in keep if x != k)
            try:
                e = ev(r0d, data(nsd, BLK_ON), pdr)
                if not (e["status"] == "FAIL" and any("nothing observes it" in r for r in e["reasons"])):
                    return False
            finally:
                GC.SCORER_EPO_CON_KEYS = keep
        return True
    check(bad, "G_a_built_flag_nothing_observes_is_refused", built_without_observer)
    # ---- G13 is obliged iff --epo AND --epo-con != off
    check(bad, "G_G13_is_required_by_a_constrained_argv_and_by_nothing_else", lambda: "G13" in GC.scorer_required_gates(nsd) and "G13" not in GC.scorer_required_gates(nso) and "G13" not in GC.scorer_required_gates(S.build_parser().parse_args(r0))
          and "G13" not in GC.scorer_required_gates(S.build_parser().parse_args(A["C0"])) and GC.SCORER_GATES["G13"] == "epo_con_live_and_identity")
    # ---- judge_g13: re-judged from raw fields, each with a negative twin
    g13 = good["g13"]
    nsj = argparse.Namespace(epo_con="g5", epo_con_eps=0.0, epo_con_lr=4.0, epo_con_max=10.0, epo_con_lam0=0.0, epo_temp=0.1, epo_con_temp=None)
    check(bad, "G_judge_g13_accepts_a_real_preflight_s_G13", lambda: GC.judge_g13(g13, nsj) == [])

    def twin(edit, frag):
        d = copy.deepcopy(g13)
        edit(d)
        return any(frag in r for r in GC.judge_g13(d, nsj))
    twins = {
        "ok_false": (lambda d: d.update(ok=False), "ok != true"), "mutate": (lambda d: d.update(mutate=["REFE_RL_MUTATE_CON_DETACH"]), "regression switch"),
        "B_module_dead": (lambda d: d["per_module_on"].update(score_head=0.0), "non-zero on every scorer module"), "B_module_missing": (lambda d: d["per_module_on"].pop("score_head"), "non-zero on every scorer module"),
        "B_nan": (lambda d: d["per_module_on"].update(score_dec=float("nan")), "non-zero on every scorer module"),
        "lam0_control_nonzero": (lambda d: d["per_module_lam0"].update(score_dec=1e-12), "zero controls"), "off_control_nonzero": (lambda d: d["per_module_off"].update(score_q_mlp=1e-12), "zero controls"),
        "value_lam0_nonzero": (lambda d: d.update(value_lam0=1e-9), "zero controls"), "controls_missing": (lambda d: d.update(per_module_off={}), "zero controls"),
        "A_touched": (lambda d: d["A_per_module_on"].update(score_head=1e-9), "arm A received a gradient"), "A_touched_lam0": (lambda d: d["A_per_module_lam0"].update(score_head=1e-9), "arm A received a gradient"),
        "A_control_dead": (lambda d: d["A_bce_control"].update(score_head=0.0), "arm A's own BCE gradient"), "A_control_missing": (lambda d: d.update(A_bce_control={}), "arm A's own BCE gradient"),
        "identity_gap": (lambda d: d["c_gap"].update(ttc=0.5), "step-0 identity"), "identity_comp_missing": (lambda d: d["c_gap"].pop("nc"), "step-0 identity"),
        "dual_got": (lambda d: d["dual_reach"]["got"].update(ttc=0.3), "does not reach lambda"), "dual_floor": (lambda d: d["dual_reach"]["floor_got"].update(nc=0.1), "floor lambda"),
        "dual_cap": (lambda d: d["dual_reach"]["cap_got"].update(nc=9.9), "cap lambda"), "dual_wrong_eta": (lambda d: d["dual_reach"].update(eta=2.0), "dual probe ran with eta"),
        "dual_wrong_lam0": (lambda d: d["dual_reach"].update(lam0=0.5), "dual probe ran with lam0"), "dual_comps": (lambda d: d["dual_reach"].update(comps=["ttc"]), "dual probe covers"),
        "dual_missing": (lambda d: d.update(dual_reach={}), "missing or malformed"),
        "ckpt_unequal": (lambda d: d["checkpoint"].update(block_equal=False), "round-trip"), "ckpt_fields": (lambda d: d["checkpoint"].update(keys=["lam"]), "round-trip"),
    }
    for nm, (edit, frag) in twins.items():
        check(bad, f"G_judge_g13_twin_{nm}", lambda edit=edit, frag=frag: twin(edit, frag))
    check(bad, "G_judge_g13_declares_the_registered_literal_0.3125", lambda: GC.judge_g13(g13, nsj) == [] and g13["dual_reach"]["got"]["ttc"] == 0.3125)
    # ---- the consumer (G-PREFLIGHT) obliges, forbids and re-judges
    pre = good["pre_rows"]

    def consume(argv_, prof, rows):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "preflight.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                f.write("".join(json.dumps(r) + "\n" for r in rows))
            return L.consume_scorer_preflight(p, argv_, prof, arm="R0d")

    def without_g13(rows):
        out = copy.deepcopy(rows)
        for r in out:
            if r.get("event") == "preflight":
                r["gates"].pop(G13K, None)
        return out
    G13K = "G13_epo_con_live_and_identity"

    def with_edit(rows, edit):
        out = copy.deepcopy(rows)
        for r in out:
            if r.get("event") == "preflight":
                edit(r["gates"][G13K])
        return out
    check(bad, "G_consumer_PASS_R0d_with_a_real_G13", lambda: consume(r0d, Pd, pre)["status"] == "PASS")
    check(bad, "G_consumer_FAIL_R0d_without_G13_says_ABSENT", lambda: (lambda v: v["status"] == "FAIL" and any("ABSENT" in r and "G13" in r for r in v["reasons"]))(consume(r0d, Pd, without_g13(pre))))
    check(bad, "G_consumer_FAIL_R1o_with_a_stray_G13_is_another_configuration", lambda: (lambda v: v["status"] == "FAIL" and any("do not oblige" in r.replace("does not oblige", "do not oblige") for r in v["reasons"]))(consume(r1o, Pd, pre)))
    check(bad, "G_consumer_PASS_R1o_without_G13", lambda: consume(r1o, Pd, without_g13(pre))["status"] == "PASS")
    check(bad, "G_consumer_re_judges_G13_from_raw_fields_not_from_its_ok", lambda: (lambda v: v["status"] == "FAIL" and any("L_con-only backward" in r for r in v["reasons"]))(
        consume(r0d, Pd, with_edit(pre, lambda g: g["per_module_on"].update(score_head=0.0)))))
    check(bad, "G_consumer_FAIL_a_G13_produced_under_a_regression_switch", lambda: consume(r0d, Pd, with_edit(pre, lambda g: g.update(mutate=["REFE_RL_MUTATE_CON_ATTACH_A"])))["status"] == "FAIL")
    return bad


# ================================================================================================ part CH: the chain (rls1_checks.py, rls1_pod.sh)
def _fn_body(text: str, name: str) -> str:
    m = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", text, re.S | re.M)
    return m.group(0) if m else ""


def _posix(p: str) -> str:
    return str(p).replace("\\", "/")


def _bash_args(pod_src: str, tmp: str, arm: str) -> str:
    """the chain's OWN arm_args function (extracted verbatim, with the header assignments it needs) RUN under bash -> the argv string of `arm`"""
    hdr = "\n".join(l for l in pod_src.split("\n") if re.match(r"^[A-Z0-9_]+=", l) and not l.startswith("export") and "$(" not in l and "FINAL_SHA256" not in l and not l.startswith("NSH"))
    sc = os.path.join(tmp, f"aa_{arm}.sh")
    with open(sc, "w", encoding="utf-8", newline="\n") as f:
        f.write(hdr + "\n" + _fn_body(pod_src, "arm_args") + '\narm_args "$1"\n')
    bash = shutil.which("bash")
    assert bash, "bash is not on PATH (Git Bash on the dev box): the chain's argv cannot be RUN"
    r = subprocess.run([bash, _posix(sc), arm], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, (arm, r.stderr[-200:])
    return r.stdout.strip()


def suite_ch(rc_src: str, pod_src: str, tag: str) -> list:
    import inspect
    import gate_token as GT
    bad: list = []
    RC = S1B.load(rc_src, f"rc_so_{tag}", RC_PATH)
    LGM = S1B.load(open(LG_PATH, encoding="utf-8").read(), f"lg_so_ch_{tag}", LG_PATH)
    AJ = json.load(open(os.path.join(HERE, "..", "eval", "rls1d_arms.json"), encoding="utf-8"))

    # ---------------------------------------------------------------- static: registration, profile, argv (RUN under bash), the files the pod needs
    reg = re.search(r'^REGISTERED_ARMS="([^"]*)"', pod_src, re.M)
    check(bad, "CH_registered_arms_are_the_older_eight_plus_R0d_R1d_R1o", lambda: reg is not None and reg.group(1).split() == ["C0", "C1", "R0", "R1", "R2", "R0b", "R1b", "R0c", "R0d", "R1d", "R1o"])
    ap_body = _fn_body(pod_src, "arm_profile")
    check(bad, "CH_profile_R0d_R1d_R1o_are_judged_under_refe_rls1d_and_R0c_under_refe_rls1c", lambda: "    R0d|R1d|R1o) echo refe_rls1d ;;\n" in ap_body and "    R0c) echo refe_rls1c ;;\n" in ap_body)
    check(bad, "CH_arms_json_is_derived_from_R0_s_banked_token_and_orders_the_three_arms", lambda: AJ["order"] == ["R0d", "R1d", "R1o"] and AJ["profile"] == "refe_rls1d"
          and AJ["derived_from"]["argv_sha256"] == R0_ARGV_SHA256 and AJ["self_check"]["equals_R0_banked_token"] is True and AJ["self_check"]["R0_argv_rebuilt_sha256"] == R0_ARGV_SHA256
          and AJ["budget"]["total_a40_hours"] == 10.5 and AJ["budget"]["a40_arms"] == 3)
    with tempfile.TemporaryDirectory() as tmp:
        got = {}
        for arm in ("R0", "R0d", "R1d", "R1o"):
            try:
                got[arm] = _bash_args(pod_src, tmp, arm).split()
            except Exception as e:                                              # noqa: BLE001
                got[arm] = None
                bad.append(f"CH_arm_args_run_under_bash[{arm}]: {type(e).__name__}: {str(e)[:80]}")
        with open(os.path.join(tmp, "pod.sh"), "w", encoding="utf-8", newline="\n") as f:
            f.write(pod_src)
        syntax_ok = subprocess.run([shutil.which("bash") or "bash", "-n", _posix(os.path.join(tmp, "pod.sh"))], capture_output=True).returncode == 0
    check(bad, "CH_the_chain_parses_under_bash_n", lambda: syntax_ok)
    check(bad, "CH_argv_R0_is_still_the_banked_token", lambda: got["R0"] is not None and GT.argv_sha256(got["R0"]) == R0_ARGV_SHA256)
    for arm in ("R0d", "R1d", "R1o"):
        check(bad, f"CH_argv_{arm}_equals_eval_rls1d_arms_json_token_for_token_and_its_sha256", lambda arm=arm: got[arm] == AJ["arms"][arm]["argv"] and GT.argv_sha256(got[arm]) == AJ["arms"][arm]["argv_sha256"])
        check(bad, f"CH_argv_{arm}_is_the_argv_the_gate_tests_use", lambda arm=arm: got[arm] == LGM.parse_args_string(ARM_ARGS[arm]))
        check(bad, f"CH_argv_{arm}_is_clean_under_the_real_refe_rls1d_profile", lambda arm=arm: LGM.profile_refusals({**LGM.PROFILES["refe_rls1d"], "declared_arm": arm}, got[arm]) == [])
    check(bad, "CH_R0d_and_R1d_carry_the_five_flags_R1o_names_off_and_only_the_epo_tail_differs_from_R0", lambda: got["R0d"][-10:] == CON_ON.split() and got["R1d"][-10:] == CON_ON.split() and got["R1o"][-2:] == ["--epo-con", "off"]
          and got["R0d"][:-10] == [t.replace("/arms/R0", "/arms/R0d") for t in got["R0"]] and got["R1o"][:-2] == [t.replace("/arms/R1d", "/arms/R1o") for t in got["R1d"][:-10]])
    # the arm loop
    i_generic, own = pod_src.find('if [ -z "$RD" ] ||'), 'case "$ARM" in R0d|R1d|R1o) REFDIR="$ARMDIR/scores";; esac'
    check(bad, "CH_own_arm_A_is_the_monitor_and_replay_reference_after_the_generic_pick", lambda: i_generic > 0 and pod_src.count(own) == 1 and pod_src.find(own) > i_generic)
    check(bad, "CH_the_seed_matched_control_is_named_the_SECOND_control_before_the_generic_pick", lambda: pod_src.count('case "$ARM" in R0d) RD=C0;; R1d|R1o) RD=C1;; esac') == 1
          and 0 < pod_src.find('case "$ARM" in R0d) RD=C0;; R1d|R1o) RD=C1;; esac') < i_generic)
    mon = re.search(r'rls1_checks\.py monitor .*?monitor\.log', pod_src, re.S)
    rep = re.search(r'rls1_checks\.py replay .*?replay\.log', pod_src, re.S)
    check(bad, "CH_M9_the_monitor_line_carries_OBS_so_R1o_s_K3_is_observe_only", lambda: mon is not None and '${OBS:-}' in mon.group(0))
    check(bad, "CH_the_replay_line_carries_OBS_too_a_retroactive_K3_must_not_kill_R1o", lambda: rep is not None and '${OBS:-}' in rep.group(0))
    check(bad, "CH_OBS_is_observe_only_for_R1o_alone", lambda: pod_src.count('OBS="--observe-only"') == 1 and '  OBS=""\n  if [ "$ARM" = R1o ]; then\n    OBS="--observe-only"' in pod_src)
    gate_i = pod_src.find('safeobj-gate --dirs "$R/arms/R0d" "$R/arms/R1d"')
    check(bad, "CH_R1o_waits_for_both_S_verdicts_before_any_directory_preflight_or_GPU_work", lambda: gate_i > 0 and pod_src.count("safeobj-gate --dirs") == 1
          and '|| refuse "R1o runs only if R0d and R1d both read bar S = PASS' in pod_src and gate_i < pod_src.find('mkdir -p "$ARMDIR" "$LG"') and gate_i < pod_src.find("scorer_finetune.py $ARGS --preflight"))
    check(bad, "CH_the_precheck_ships_test_rl_safeobj_the_refe_rls1d_suite_runs_it_on_the_pod", lambda: "refe/test_rl_r0c.py refe/test_rl_safeobj.py \\\n" in pod_src)
    check(bad, "CH_the_precheck_ships_the_S_reader_and_the_two_tests_that_pin_the_S_gate_a_missing_one_is_refused_before_the_queue",
          lambda: " eval/safeobj_read.py eval/test_safeobj_read.py eval/test_safeobj_gate.py SHIP_MANIFEST.json \\\n" in pod_src)
    check(bad, "CH_a_missing_second_control_is_ANNOUNCED_not_refused", lambda: re.search(r'elif \[ "\$ARM" = R0d \] \|\| \[ "\$ARM" = R1d \] \|\| \[ "\$ARM" = R1o \]; then\n        log "ZZRLS1_SAFEOBJ_NO_SECOND_CONTROL', pod_src) is not None)
    check(bad, "CH_a_killed_arm_still_stops_the_chain_no_exemption_for_R0d_R1d_R1o", lambda: pod_src.count('\n  [ -f "$S/arm_${ARM}_killed" ] && refuse "arm $ARM was KILLED') == 1
          and pod_src.count('\n  [ -f "$S/arm_${ARM}_killed" ] && stage_done "egress_$ARM" && refuse') == 1 and not re.search(r'"\$ARM" != R(0d|1d|1o)', pod_src))

    # ---------------------------------------------------------------- behaviour: rls1_checks.py
    rec = lambda t, g3=True, g5=True, k4=False, k5=False: {"tag": t, "g3_ok": g3, "g5_ok": g5, "k4_breach": k4, "k5_breach": k5}          # noqa: E731
    one = [rec("u250", g3=False)]
    check(bad, "CH_judge_K3_kills_by_default", lambda: RC.judge("R1o", rec("u500", g3=False), one)[0] == "K3" and RC.judge("R0d", rec("u500", g5=False), one)[0] == "K3")
    check(bad, "CH_judge_observe_only_is_off_by_default_and_never_returns_K3_when_on", lambda: inspect.signature(RC.judge).parameters["observe_only"].default is False
          and RC.judge("R1o", rec("u500", g3=False), one, True, True)[0] is None and RC.judge("R1o", rec("u500", g5=False), one, True, True)[0] is None)
    check(bad, "CH_judge_observe_only_keeps_K4_and_K5_exactly", lambda: RC.judge("R1o", rec("u500", k4=True), [rec("u250", k4=True)], True, True)[0] == "K4"
          and RC.judge("R2", rec("u250", k5=True), [], True, True)[0] == "K5" and RC.judge("R1o", rec("u250", k5=True), [], True, True)[0] is None)

    def run_mon(arm, events, recs, observe=False, now=None):
        with tempfile.TemporaryDirectory() as td:
            od = os.path.join(td, "o")
            os.makedirs(od)
            with open(os.path.join(od, "sft.jsonl"), "w", encoding="utf-8") as f:
                f.write("".join(json.dumps(e) + "\n" for e in events))
            killed, seq, clock = [], [True, False], (iter(now) if now else None)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = RC.monitor(arm, od, lambda t, files: recs[t], 99, lambda p: seq.pop(0) if len(seq) > 1 else seq[0], killed.append, sleep_fn=lambda s: None,
                                now_fn=(lambda: next(clock)) if clock else (lambda: 0.0), start=0.0, observe_only=observe)
            kp, cp = os.path.join(od, "KILLED.json"), os.path.join(od, "kill_checks.jsonl")
            code = json.load(open(kp, encoding="utf-8"))["code"] if os.path.exists(kp) else None
            rows = [json.loads(x) for x in open(cp, encoding="utf-8").read().splitlines()] if os.path.exists(cp) else []
            return rc, killed, code, buf.getvalue(), rows
    dump = lambda t: {"event": "score_dump", "tag": t, "files": {"B": "b", "A": "a", "base": "z"}}          # noqa: E731
    ev = [dump("step0"), dump("update250"), dump("update500"), dump("update750")]
    ok0 = {"step0": rec("step0")}
    k3 = {**ok0, "update250": rec("update250", g3=False), "update500": rec("update500", g3=False), "update750": rec("update750", g3=False)}
    k4 = {**ok0, "update250": rec("update250", k4=True), "update500": rec("update500", k4=True), "update750": rec("update750")}
    check(bad, "CH_monitor_R0d_two_consecutive_g3_g5_breaches_are_a_K3_KILL_as_ever", lambda: (lambda r: r[0] == 2 and r[1] == [99] and r[2] == "K3" and "ZZRLS1_K3_OBSERVED" not in r[3])(run_mon("R0d", ev, k3)))
    check(bad, "CH_monitor_R1o_without_the_flag_would_be_killed_which_is_what_M9_guards", lambda: run_mon("R1o", ev, k3)[2] == "K3")
    check(bad, "CH_monitor_R1o_observe_only_NEVER_kills_on_K3_logs_it_and_keeps_every_check_row", lambda: (lambda r: r[0] == 0 and r[1] == [] and r[2] is None and r[3].count("ZZRLS1_K3_OBSERVED R1o update500") == 1
                                                                                                           and r[3].count("ZZRLS1_K3_OBSERVED R1o update750") == 1 and len(r[4]) == 3 and all(x["kill"] is None for x in r[4]))(run_mon("R1o", ev, k3, observe=True)))
    check(bad, "CH_monitor_R1o_observe_only_still_dies_on_K4_two_consecutive_RL_val_PDMS_breaches", lambda: (lambda r: r[0] == 2 and r[2] == "K4")(run_mon("R1o", ev, k4, observe=True)))
    check(bad, "CH_monitor_R1o_observe_only_still_dies_on_K2_a_non_finite_loss", lambda: (lambda r: r[0] == 2 and r[2] == "K2")(
        run_mon("R1o", [{"event": "train", "update": 25, "bceA": 0.3, "lossB": float("nan")}], {}, observe=True)))
    check(bad, "CH_monitor_R1o_observe_only_still_dies_on_K6_hours", lambda: (lambda r: r[0] == 2 and r[2] == "K6")(run_mon("R1o", [dump("step0")], ok0, observe=True, now=[7 * 3600.0] * 4)))
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "sft.jsonl"), "w", encoding="utf-8") as f:
            f.write("".join(json.dumps(e) + "\n" for e in ev))
        check(bad, "CH_replay_R1o_observe_only_does_not_retro_kill_on_K3_and_marks_the_records", lambda: (lambda r: r[0] is None and r[2] and all(x.get("observe_only") is True for x in r[2]))(RC.replay("R1o", td, lambda t, f: k3[t], observe_only=True)))
        check(bad, "CH_replay_without_the_flag_finds_the_K3", lambda: RC.replay("R1o", td, lambda t, f: k3[t])[0] == "K3" and RC.replay("R0d", td, lambda t, f: k3[t], observe_only=False)[0] == "K3")
        check(bad, "CH_replay_R1o_observe_only_still_finds_a_K4", lambda: RC.replay("R1o", td, lambda t, f: k4[t], observe_only=True)[0] == "K4")

    def cli(*a):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = RC.main(list(a))
        return rc, buf.getvalue()
    def cargs(cmd, arm):
        return [cmd, "--arm", arm, "--out", "o", "--refdir", "r", "--pdm", "p", "--terms", "t", "--sets", "s", "--tsv", "v", *(["--pid", "1"] if cmd == "monitor" else []), "--observe-only"]
    check(bad, "CH_cli_observe_only_is_refused_on_every_arm_but_R1o_monitor_and_replay", lambda: all(
        (lambda r: r[0] == 1 and r[1].startswith("ZZRLS1_OBSERVE_ONLY_REFUSED"))(cli(*cargs(c, a))) for a in ("R0", "R0d", "R1d", "R0c", "C0") for c in ("monitor", "replay")))
    # ---- the S gate.  SAFEOBJ round 3: it opens only on a GENUINE, unmodified reader record, so the PASS / FAIL / NOT_READ fixtures below are arm directories made by the REAL producers with the record
    # written by the REAL reader (eval/test_safeobj_gate.py: genuine / copy_arm); a hand-written `{"S": "PASS"}` -- what the gate used to open on -- is now one of the refused cases.
    # (The full battery of mutations of the genuineness check lives in eval/test_safeobj_gate.py.)
    sys.path.insert(0, os.path.join(HERE, "..", "eval"))
    import test_safeobj_gate as TG
    with tempfile.TemporaryDirectory() as td:
        def own(label, kind, arm):                                          # a private copy of the genuine arm directory (its basename IS the arm)
            return str(TG.copy_arm(kind, arm, os.path.join(td, label)))

        def rewrite(d, fn):                                                 # edit the record in d (the reader's file, json) with fn(record) and write it back unstamped
            p = os.path.join(d, "safeobj_S.json")
            with open(p, encoding="utf-8") as f:
                rec = json.load(f)
            out = fn(rec)
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                json.dump(out, f, indent=1, sort_keys=True)
            return d
        restamp = lambda rec: {**rec, "record_sha256": TG.oracle_digest(rec)}                      # noqa: E731  (what a forger who knows the canonicalisation writes)
        d_pass, d_pass2 = own("p", "PASS", "R0d"), own("p2", "PASS", "R1d")
        d_fail, d_nr = own("f", "FAIL", "R1d"), own("nr", "NOT_READ", "R1d")
        d_none = os.path.join(td, "n")
        os.makedirs(d_none)
        d_junk = os.path.join(td, "j")
        os.makedirs(d_junk)
        with open(os.path.join(d_junk, "safeobj_S.json"), "w", encoding="utf-8") as f:
            f.write("{not json")
        d_weak = rewrite(own("w", "PASS", "R1d"), lambda r: restamp({**r, "S": "PASS "}))        # an almost-PASS whose stamp is right: the exactness of S is the only barrier
        d_hand = rewrite(own("h", "FAIL", "R1d"), lambda r: {"S": "PASS"})                         # hand-written
        d_edit = rewrite(own("e", "FAIL", "R1d"), lambda r: {**r, "S": "PASS"})                    # a genuine FAIL with S edited to PASS, nothing else touched
        d_stale = own("s", "PASS", "R1d")                                                          # a genuine PASS whose input changed after the read
        with open(os.path.join(d_stale, "kill_replay.json"), "ab") as f:
            f.write(b" ")
        check(bad, "CH_safeobj_gate_passes_only_when_every_constrained_arm_reads_S_PASS", lambda: RC.safeobj_gate([d_pass, d_pass2])[0] is True)
        check(bad, "CH_safeobj_gate_refuses_a_FAIL_a_NOT_READ_a_missing_file_junk_and_an_almost_PASS", lambda: all(RC.safeobj_gate([d_pass, x])[0] is False for x in (d_fail, d_nr, d_none, d_junk, d_weak))
              and all(RC.safeobj_gate([x, d_pass])[0] is False for x in (d_fail, d_nr, d_none, d_junk, d_weak)))
        d_forged = rewrite(own("fg", "FAIL", "R1d"), lambda r: restamp({**r, "S": "PASS", "breaches": []}))          # a forger who re-implements the canonicalisation: tool, stamp and inputs are right, the verdict is a lie
        d_late = own("late", "PASS", "R1d")                                                                           # a genuine PASS read, and then a K3 marker appears in the arm directory
        with open(os.path.join(d_late, "KILLED.json"), "w", encoding="utf-8") as f_:
            json.dump({"arm": "R1d", "code": "K3", "why": "two consecutive breaches (this marker appeared after the read)"}, f_)
        check(bad, "CH_safeobj_gate_refuses_a_forger_who_recomputes_the_stamp_and_a_KILLED_marker_that_appears_after_the_read_each_as_stale", lambda: all(
            RC.safeobj_gate([d_pass, x])[0] is False and RC.safeobj_gate([x, d_pass])[0] is False and "record is stale or not the reader's output" in RC.safeobj_gate([d_pass, x])[1] for x in (d_forged, d_late)))
        check(bad, "CH_safeobj_gate_refuses_a_hand_written_PASS_an_edited_record_and_a_record_whose_input_changed_each_for_its_reason", lambda: all(
            RC.safeobj_gate([d_pass, x])[0] is False and RC.safeobj_gate([x, d_pass])[0] is False and why in RC.safeobj_gate([d_pass, x])[1] for x, why in (
                (d_hand, "tool is"), (d_edit, "record_sha256 does not recompute"), (d_stale, "input 'kill_replay.json' changed since the read"))))
        check(bad, "CH_safeobj_gate_cli_prints_the_marker_the_chain_greps_and_the_exit_code", lambda: (lambda a, b: a[0] == 0 and a[1].startswith("ZZRLS1_SAFEOBJ_S_PASS") and b[0] == 1 and b[1].startswith("ZZRLS1_SAFEOBJ_S_REFUSED")
                                                                                                         and "safeobj_S.json" in b[1])(cli("safeobj-gate", "--dirs", d_pass, d_pass2), cli("safeobj-gate", "--dirs", d_pass, d_fail)))
    return bad


# the chain mutations (name, part CH, edits, env)
CH_MUTATIONS = [
    ("M9 R1o's K3 monitor without --observe-only", "CH", [(POD_PATH, '--pid "$TPID" ${OBS:-} > "$ARMDIR/monitor.log"', '--pid "$TPID" > "$ARMDIR/monitor.log"')], None),
    ("CH1 the replay forgets --observe-only", "CH", [(POD_PATH, '--tsv "$VAL_TSV" ${OBS:-} > "$ARMDIR/replay.log"', '--tsv "$VAL_TSV" > "$ARMDIR/replay.log"')], None),
    ("CH2 OBS is set for every arm", "CH", [(POD_PATH, '  OBS=""\n  if [ "$ARM" = R1o ]; then\n    OBS="--observe-only"', '  OBS="--observe-only"\n  if [ "$ARM" = R1o ]; then\n    :')], None),
    ("CH3 R1o's S gate is not enforced", "CH", [(POD_PATH, '|| refuse "R1o runs only if R0d and R1d both read bar S = PASS: $(tail', '|| echo "R1o runs only if R0d and R1d both read bar S = PASS: $(tail')], None),
    ("CH4 R0d is read against C0's dumps when they are there", "CH", [(POD_PATH, '  case "$ARM" in R0d|R1d|R1o) REFDIR="$ARMDIR/scores";; esac', "  :")], None),
    ("CH5 R1o is not registered", "CH", [(POD_PATH, 'REGISTERED_ARMS="C0 C1 R0 R1 R2 R0b R1b R0c R0d R1d R1o"', 'REGISTERED_ARMS="C0 C1 R0 R1 R2 R0b R1b R0c R0d R1d"')], None),
    ("CH6 R1o's argv names --epo-con g5", "CH", [(POD_PATH, '--epo-terms-heldout $TERMS_VA --epo-con off" ;;', '--epo-terms-heldout $TERMS_VA --epo-con g5" ;;')], None),
    ("CH7 a killed R0d does not stop the chain", "CH", [(POD_PATH, '\n  [ -f "$S/arm_${ARM}_killed" ] && refuse "arm $ARM was KILLED', '\n  [ "$ARM" != R0d ] && [ -f "$S/arm_${ARM}_killed" ] && refuse "arm $ARM was KILLED')], None),
    ("CH8 judge ignores observe_only", "CH", [(RC_PATH, "if k3_enabled and not observe_only and prev is not None", "if k3_enabled and prev is not None")], None),
    ("CH9 --observe-only is accepted on any arm", "CH", [(RC_PATH, 'and a.observe_only and a.arm != "R1o":', "and a.observe_only and False:")], None),
    ("CH10 safeobj_gate accepts anything but FAIL", "CH", [(RC_PATH, '        if s != "PASS":\n', '        if s == "FAIL":\n')], None),
    ("CH11 replay ignores observe_only", "CH", [(RC_PATH, "code, why = judge(arm, rec, hist, k3_enabled, observe_only)\n        hist.append(rec)", "code, why = judge(arm, rec, hist, k3_enabled)\n        hist.append(rec)")], None),
    ("CH12 the monitor does not log the observed K3", "CH", [(RC_PATH, '                    say("ZZRLS1_K3_OBSERVED", arm, r["tag"], would_why)\n', "                    pass\n")], None),
    ("CH13 R0d's argv loses --epo-con-lam0", "CH", [(POD_PATH, '    R0d) seed=0; epo="--epo --epo-terms-train $TERMS_TR --epo-terms-heldout $TERMS_VA --epo-con g5 --epo-con-eps 0 --epo-con-lr 4 --epo-con-max 10 --epo-con-lam0 0" ;;',
                                                          '    R0d) seed=0; epo="--epo --epo-terms-train $TERMS_TR --epo-terms-heldout $TERMS_VA --epo-con g5 --epo-con-eps 0 --epo-con-lr 4 --epo-con-max 10" ;;')], None),
    ("CH14 the pod precheck does not ship test_rl_safeobj.py", "CH", [(POD_PATH, "refe/test_rl_r0c.py refe/test_rl_safeobj.py \\\n", "refe/test_rl_r0c.py \\\n")], None),
    ("CH15 safeobj_gate does not verify the record's genuineness", "CH", [(RC_PATH, "            why = safeobj_not_genuine(d, rec)\n", '            why = ""\n')], None),
    ("CH16 safeobj_gate does not re-read the arm directory", "CH", [(RC_PATH, "            diff = SO.differing_fields(rec, SO.read_arm(d))\n", "            diff = []\n")], None),
    ("CH17 the pod precheck does not ship the S-reader and its two tests", "CH", [(POD_PATH, " eval/safeobj_read.py eval/test_safeobj_read.py eval/test_safeobj_gate.py SHIP_MANIFEST.json \\\n", " SHIP_MANIFEST.json \\\n")], None),
]
CH_EXPECT = {
    "M9 R1o's K3 monitor without --observe-only": ["CH_M9_the_monitor_line_carries_OBS_so_R1o_s_K3_is_observe_only"],
    "CH1 the replay forgets --observe-only": ["CH_the_replay_line_carries_OBS_too_a_retroactive_K3_must_not_kill_R1o"],
    "CH2 OBS is set for every arm": ["CH_OBS_is_observe_only_for_R1o_alone"],
    "CH3 R1o's S gate is not enforced": ["CH_R1o_waits_for_both_S_verdicts_before_any_directory_preflight_or_GPU_work"],
    "CH4 R0d is read against C0's dumps when they are there": ["CH_own_arm_A_is_the_monitor_and_replay_reference_after_the_generic_pick"],
    "CH5 R1o is not registered": ["CH_registered_arms_are_the_older_eight_plus_R0d_R1d_R1o"],
    "CH6 R1o's argv names --epo-con g5": ["CH_argv_R1o_equals_eval_rls1d_arms_json_token_for_token_and_its_sha256"],
    "CH7 a killed R0d does not stop the chain": ["CH_a_killed_arm_still_stops_the_chain_no_exemption_for_R0d_R1d_R1o"],
    "CH8 judge ignores observe_only": ["CH_monitor_R1o_observe_only_NEVER_kills_on_K3_logs_it_and_keeps_every_check_row", "CH_judge_observe_only_is_off_by_default_and_never_returns_K3_when_on"],
    "CH9 --observe-only is accepted on any arm": ["CH_cli_observe_only_is_refused_on_every_arm_but_R1o_monitor_and_replay"],
    "CH10 safeobj_gate accepts anything but FAIL": ["CH_safeobj_gate_refuses_a_FAIL_a_NOT_READ_a_missing_file_junk_and_an_almost_PASS"],
    "CH11 replay ignores observe_only": ["CH_replay_R1o_observe_only_does_not_retro_kill_on_K3_and_marks_the_records"],
    "CH12 the monitor does not log the observed K3": ["CH_monitor_R1o_observe_only_NEVER_kills_on_K3_logs_it_and_keeps_every_check_row"],
    "CH13 R0d's argv loses --epo-con-lam0": ["CH_argv_R0d_equals_eval_rls1d_arms_json_token_for_token_and_its_sha256"],
    "CH14 the pod precheck does not ship test_rl_safeobj.py": ["CH_the_precheck_ships_test_rl_safeobj_the_refe_rls1d_suite_runs_it_on_the_pod"],
    "CH15 safeobj_gate does not verify the record's genuineness": ["CH_safeobj_gate_refuses_a_hand_written_PASS_an_edited_record_and_a_record_whose_input_changed_each_for_its_reason"],
    "CH16 safeobj_gate does not re-read the arm directory": ["CH_safeobj_gate_refuses_a_forger_who_recomputes_the_stamp_and_a_KILLED_marker_that_appears_after_the_read_each_as_stale"],
    "CH17 the pod precheck does not ship the S-reader and its two tests": ["CH_the_precheck_ships_the_S_reader_and_the_two_tests_that_pin_the_S_gate_a_missing_one_is_refused_before_the_queue"],
}


# (name, part, edits, env) -- each snippet must occur EXACTLY once in its file
MUTATIONS = [
    # ---- the trainer: the two regression switches (set, not edited)
    ("M1 REFE_RL_MUTATE_CON_DETACH=1 (c_B computed from sxB.detach())", "SF", [], {"REFE_RL_MUTATE_CON_DETACH": "1"}),
    ("M2 REFE_RL_MUTATE_CON_ATTACH_A=1 (arm A is not detached)", "SF", [], {"REFE_RL_MUTATE_CON_ATTACH_A": "1"}),
    # ---- the trainer: source edits
    ("M3 delete `lB = lB + ct` in train_losses", "SF", [(SF_PATH, "                lB = lB + ct\n", "                pass\n")], None),
    ("M4 delete the dual.step(...) call in the loop", "SF", [(SF_PATH, "                dual.step(con_acc)\n", "                pass\n")], None),
    ("M5 the switch defaults to g5 instead of off", "SF", [(SF_PATH, 'EPO_CON_DEFAULT = "off" ', 'EPO_CON_DEFAULT = "g5" ')], None),
    ("M6 column 2 (EP) instead of 3 (TTC)", "SF", [(SF_PATH, "    return tuple(rl_reward.COMPONENTS.index(c) for c in comps)\n", '    return tuple(2 if c == "ttc" else rl_reward.COMPONENTS.index(c) for c in comps)\n')], None),
    ("M10 the sign of the violation flipped", "SF", [(SF_PATH, "        g = cA_s.mean() - eps - cB_s.mean()\n", "        g = cB_s.mean() - eps - cA_s.mean()\n")], None),
    ("M11 the term also added to arm A", "SF", [(SF_PATH, "                lB = lB + ct\n", "                lB = lB + ct\n                lA = lA + ct\n")], None),
    ("M12 the floor at 0.0 removed", "SF", [(SF_PATH, "            self.lam[c] = min(self.lam_max, max(0.0, raw))\n", "            self.lam[c] = min(self.lam_max, raw)\n")], None),
    ("M13 the cap at lam_max removed", "SF", [(SF_PATH, "            self.lam[c] = min(self.lam_max, max(0.0, raw))\n", "            self.lam[c] = max(0.0, raw)\n")], None),
    ("M14 the dual sums the micro-batches instead of averaging", "SF", [(SF_PATH, '            raw = self.lam[c] + self.lr * (float(acc["g_" + c]) / n)\n', '            raw = self.lam[c] + self.lr * float(acc["g_" + c])\n')], None),
    ("M15 lambda ignored by the term", "SF", [(SF_PATH, "        t_ = float(lam[comp]) * g\n", "        t_ = 1.0 * g\n")], None),
    ("M16 eps ignored by the term", "SF", [(SF_PATH, "        g = cA_s.mean() - eps - cB_s.mean()\n", "        g = cA_s.mean() - cB_s.mean()\n")], None),
    ("M27 the call site passes a literal 0.0 for the slack", "SF", [(SF_PATH, "dual.temp, a.epo_con_eps, dual.lam, con_last)", "dual.temp, 0.0, dual.lam, con_last)")], None),
    ("M17 the loop never accumulates the parts", "SF", [(SF_PATH, '                    con_acc[k_] = con_acc.get(k_, 0.0) + v_\n', "                    pass\n")], None),
    ("M18 the dual reads the wrong flag (eps and eta swapped in main)", "SF", [(SF_PATH, "a.epo_con_eps, a.epo_con_lr, a.epo_con_max, a.epo_con_lam0,\n", "a.epo_con_lr, a.epo_con_eps, a.epo_con_max, a.epo_con_lam0,\n")], None),
    ("M19 the data event ECHOES the reference and the selector is broken", "SF", [(SF_PATH, '"lam0": self.lam0, "ref": epo_con_reference_built(), "temp": self.temp}', '"lam0": self.lam0, "ref": "armA_detached", "temp": self.temp}'),
                                                                                    (SF_PATH, "    return sxA.detach()\n\n\ndef epo_con_reference_built", "    return sxA\n\n\ndef epo_con_reference_built")], None),
    ("M20 the PASS file is written under a regression switch", "SF", [(SF_PATH, "AUX_MUTATE_ENVS + EPO_CON_MUTATE_ENVS):", "AUX_MUTATE_ENVS):")], None),
    ("M21 a con column dropped from the rows", "SF", [(SF_PATH, '"con_lam_ttc": dual.lam["ttc"], ', "")], None),
    ("M22 the states file forgets the dual", "SF", [(SF_PATH, '                            **({"EPO_CON": dual.state_dict()} if dual.active else {})},\n', "                            },\n")], None),
    ("M23 main does not call epo_con_refusals", "SF", [(SF_PATH, "bad_cli = epo_refusals(a, ap) + epo_con_refusals(a, ap)", "bad_cli = epo_refusals(a, ap)")], None),
    ("M24 G13 left out of the preflight", "SF", [(SF_PATH, '            gates["G13_epo_con_live_and_identity"] = g13_epo_con_live_and_identity(', '            _unused = g13_epo_con_live_and_identity(')], None),
    ("M25 the inert configuration is no longer refused", "SF", [(SF_PATH, "    if a.epo_con_lr == 0.0 and a.epo_con_lam0 == 0.0:\n", "    if False:\n")], None),
    ("M26 a con weight while off is no longer refused", "SF", [(SF_PATH, "    if a.epo_con == \"off\":\n        if con_named:\n", "    if a.epo_con == \"off\":\n        if False:\n")], None),
    ("M28 the dual probe of G13 reports a wrong closed form (floor not applied)", "SF", [(SF_PATH, "    floor_got = feed(-1.0, 3)\n", "    floor_got = feed(1.0, 3)\n")], None),
    ("M29 the dual's state_dict forgets cap_hits", "SF", [(SF_PATH, '"update": self.updates, "cap_hits": self.cap_hits, "pos_updates"', '"update": self.updates, "pos_updates"')], None),
    # ---- the gate
    ("M7 drop epo_con from DVB_KINDS_SCORER", "G", [(GC_PATH, '    "epo_con": "built", "epo_con_eps": "built",', '    "epo_con_eps": "built",')], None),
    ("M8 the refe_rls1d suite without test_rl_safeobj.py", "G", [(LG_PATH, 'RLS1D_SUITE = RLS1C_SUITE + [{"script": "test_rl_safeobj.py", "token": "ZZTEST_PASS"}]', "RLS1D_SUITE = RLS1C_SUITE")], None),
    ("G1 R1o not required to NAME --epo-con", "G", [(LG_PATH, '"R1o": ("--epo-con",)}', '"R1o": ()}')], None),
    ("G2 R0d registers lambda_max 5", "G", [(LG_PATH, '"epo_con_max": 10.0, "epo_con_lam0": 0.0}', '"epo_con_max": 5.0, "epo_con_lam0": 0.0}')], None),
    ("G3 the per-arm registry is ignored", "G", [(LG_PATH, '**prof.get("arm_epo_registered", {}).get(declared, {})}', "}")], None),
    ("G4 the per-arm must-name list is ignored", "G", [(LG_PATH, '        for f in prof.get("arm_must_name", {}).get(declared, ()):', "        for f in ():")], None),
    ("G5 older profiles no longer register --epo-con off", "G", [(LG_PATH, 'reg = {"epo_con": "off", **reg,', "reg = {**reg,")], None),
    ("G6 epo_con_refusals not called by the gate", "G", [(LG_PATH, " + mod.epo_con_refusals(ns, parser)]", "]")], None),
    ("G7 G13 not re-judged by the consumer", "G", [(LG_PATH, "        reasons += C.judge_g13(g13, ns)\n", "        pass\n")], None),
    ("G8 G13 not obliged by --epo-con", "G", [(GC_PATH, '    if ns.epo and getattr(ns, "epo_con", "off") != "off":\n        req.append("G13")\n', "")], None),
    ("G9 the aux profile no longer forbids the new flags", "G", [(LG_PATH, '"--epo-l", "--epo-lam-l", "--epo-con", "--epo-con-eps", "--epo-con-lr", "--epo-con-max", "--epo-con-lam0",', '"--epo-l", "--epo-lam-l",')], None),
    ("G10 the con rows of G-DVB removed", "G", [(GC_PATH, '                add("epo_con_" + k, getattr(ns, "epo_con_" + k), blk.get(k), f"data event: epo_con.{k}" + why)\n', "                pass\n")], None),
    ("G11 judge_g13 trusts the probe's own ok", "G", [(GC_PATH, "    on, lam0, off = (g13.get(k) or {} for k in (\"per_module_on\", \"per_module_lam0\", \"per_module_off\"))\n    if set(on) != mods", "    return r\n    on, lam0, off = (g13.get(k) or {} for k in (\"per_module_on\", \"per_module_lam0\", \"per_module_off\"))\n    if set(on) != mods")], None),
    ("G12 judge_g13 does not check the arm A reads", "G", [(GC_PATH, "    if any(set(d) != mods or any(x != 0.0 for x in d.values()) for d in a_reads):\n", "    if False:\n")], None),
    ("G13 the legacy tolerance applies to rls1d too", "G", [(GC_PATH, 'legacy = blk is None and mode == "off" and not (profile or {}).get("epo_con_block")', 'legacy = blk is None and mode == "off"')], None),
    ("G14 the dual-reach closed form is not checked", "G", [(GC_PATH, "            if (dr.get(\"got\") or {}).get(c) != exp:\n", "            if False:\n")], None),
]  + CH_MUTATIONS


def apply_edits(srcs: dict, edits: list, name: str) -> dict:
    out = dict(srcs)
    for path, old, new in edits:
        cnt = out[path].count(old)
        assert cnt == 1, f"mutation {name}: the snippet occurs {cnt} times in {os.path.basename(path)} (must be exactly 1) -- a vacuous mutation proves nothing: {old[:60]!r}"
        out[path] = out[path].replace(old, new)
    return out


# What a mutation must do beyond "something goes RED": named checks that MUST go red.  These turn the prose claims of SPEC_SAFEOBJ 6 into machine-checked ones.
EXPECT = {
    "M1 REFE_RL_MUTATE_CON_DETACH=1 (c_B computed from sxB.detach())": ["E_runs_to_completion_preflight_PASS[on]", "U_G13_green_on_the_real_probe_6col"],
    "M2 REFE_RL_MUTATE_CON_ATTACH_A=1 (arm A is not detached)": ["U_C1_through_train_losses_A_gets_EXACTLY_0_from_L_con_B_is_live_on_every_module_A_BCE_control_reads_nonzero_6col", "E_runs_to_completion_preflight_PASS[on]",
                                                                  "E_the_constraint_never_reaches_the_control_arm_A_is_bit_identical_to_the_off_run"],
    "M3 delete `lB = lB + ct` in train_losses": ["U_G13_green_on_the_real_probe_6col", "U_the_constraint_adds_exactly_L_con_to_lB_and_changes_nothing_else", "E_runs_to_completion_preflight_PASS[on]",
                                                  "R_hop_train_losses_adds_the_term_to_lB_inside_the_epo_branch_with_sxA_and_the_dual_s_lambda"],
    "M4 delete the dual.step(...) call in the loop": ["R_hop_dual_step_sits_between_the_optimiser_step_and_upd_plus_1", "E_dual_step_is_called_once_per_optimiser_update_with_every_micro_batch", "E_lambda_MOVES_off_lambda0_so_the_dual_step_reaches_the_term"],
    "M5 the switch defaults to g5 instead of off": ["P_default_is_off_and_the_numeric_defaults_are_the_registered_values", "E_off_named_equals_off_default_bit_for_bit", "E_G13_is_in_the_preflight_of_a_constrained_arm_and_ONLY_there"],
    "M6 column 2 (EP) instead of 3 (TTC)": ["U_columns_TTC_is_label_column_3_and_NC_is_column_0_and_EP_DAC_C_are_not_read", "E_data_event_records_the_BUILT_facts_in_the_preflight_log_and_the_run_log"],
    "M10 the sign of the violation flipped": ["U_sign_a_step_on_L_con_RAISES_B_s_expected_TTC_and_lowers_the_term", "U_term_equals_an_independent_statement_of_the_math_L_con_and_its_logged_parts"],
    "M11 the term also added to arm A": ["R_hop_train_losses_adds_the_term_to_lB_inside_the_epo_branch_with_sxA_and_the_dual_s_lambda", "U_the_constraint_adds_exactly_L_con_to_lB_and_changes_nothing_else"],
    "M12 the floor at 0.0 removed": ["U_I4_the_dual_arithmetic_is_exact_0.3125_floor_0.0_cap_lam_max_and_the_mean_over_micro_batches", "E_runs_to_completion_preflight_PASS[on]"],
    "M13 the cap at lam_max removed": ["U_I4_the_dual_arithmetic_is_exact_0.3125_floor_0.0_cap_lam_max_and_the_mean_over_micro_batches", "E_runs_to_completion_preflight_PASS[on]"],
    "M14 the dual sums the micro-batches instead of averaging": ["U_I4_the_dual_arithmetic_is_exact_0.3125_floor_0.0_cap_lam_max_and_the_mean_over_micro_batches", "E_lambda_follows_the_closed_form_from_the_logged_violations"],
    "M15 lambda ignored by the term": ["U_I3_gradient_at_lambda_0.5_is_EXACTLY_half_of_lambda_1_and_each_component_is_live_6col", "U_lambda_0_control_reads_EXACTLY_0.0_value_and_gradient"],
    "M16 eps ignored by the term": ["U_eps_shifts_the_violation_and_the_term_by_exactly_eps_through_epo_con_term_and_through_train_losses"],
    "M27 the call site passes a literal 0.0 for the slack": ["U_eps_shifts_the_violation_and_the_term_by_exactly_eps_through_epo_con_term_and_through_train_losses"],
    "M17 the loop never accumulates the parts": ["E_dual_step_is_called_once_per_optimiser_update_with_every_micro_batch", "R_hop_the_loop_accumulates_the_micro_batch_parts_for_the_dual"],
    "M18 the dual reads the wrong flag (eps and eta swapped in main)": ["E_data_event_records_the_BUILT_facts_in_the_preflight_log_and_the_run_log", "R_hop_main_builds_the_dual_from_the_five_flags"],
    "M19 the data event ECHOES the reference and the selector is broken": ["U_C1_through_train_losses_A_gets_EXACTLY_0_from_L_con_B_is_live_on_every_module_A_BCE_control_reads_nonzero_6col"],
    "M20 the PASS file is written under a regression switch": ["R_hop_PASS_file_is_never_written_under_a_regression_switch"],
    "M21 a con column dropped from the rows": ["R_hop_the_rows_carry_the_dual_and_the_constraint_s_outcomes", "E_train_rows_carry_the_con_columns_on_a_constrained_arm"],
    "M22 the states file forgets the dual": ["R_hop_states_file_and_checkpoint_meta_carry_the_dual", "E_the_states_file_and_the_checkpoint_meta_hold_the_dual"],
    "M23 main does not call epo_con_refusals": ["R_hop_main_calls_epo_con_refusals_beside_epo_refusals_before_touching_disk", "E_refuses_before_touching_disk[refuse_inert]"],
    "M24 G13 left out of the preflight": ["E_G13_is_in_the_preflight_of_a_constrained_arm_and_ONLY_there"],
    "M25 the inert configuration is no longer refused": ["U_refusal_g5_with_eta_0_and_lambda0_0_is_INERT_by_construction", "E_refuses_before_touching_disk[refuse_inert]"],
    "M26 a con weight while off is no longer refused": ["U_refusal_a_con_weight_while_off_is_refused_because_the_term_is_OFF", "E_refuses_before_touching_disk[refuse_off_weight]"],
    "M28 the dual probe of G13 reports a wrong closed form (floor not applied)": ["E_G13_raw_fields_pass_the_gate_s_own_re_judging"],
    "M29 the dual's state_dict forgets cap_hits": ["U_the_dual_state_round_trips_and_the_BUILT_facts_are_read_off_the_object", "E_runs_to_completion_preflight_PASS[on]"],
    "M7 drop epo_con from DVB_KINDS_SCORER": ["G_a_flag_without_a_G_DVB_entry_is_refused_by_the_gate_itself_all_five", "G_dvb_registry_classifies_the_five_flags_as_built_and_observes_them"],
    "M8 the refe_rls1d suite without test_rl_safeobj.py": ["G_suite_has_the_safeobj_test_and_the_whole_older_suite"],
    "G1 R1o not required to NAME --epo-con": ["G_R1o_without_epo_con_named_is_refused"],
    "G5 older profiles no longer register --epo-con off": ["G_older_profiles_refuse_epo_con_g5"],
    "G7 G13 not re-judged by the consumer": ["G_consumer_re_judges_G13_from_raw_fields_not_from_its_ok"],
    "G8 G13 not obliged by --epo-con": ["G_G13_is_required_by_a_constrained_argv_and_by_nothing_else", "G_consumer_FAIL_R0d_without_G13_says_ABSENT"],
    "G11 judge_g13 trusts the probe's own ok": ["G_judge_g13_twin_B_module_dead", "G_judge_g13_twin_A_touched"],
    "G13 the legacy tolerance applies to rls1d too": ["G_dvb_FAIL_the_block_missing_on_rls1d_even_for_an_off_arm"],
}
EXPECT.update(CH_EXPECT)


def build_good(sf_src: str, tip_src: str) -> dict:
    """one REAL preflight of the unmutated trainer (registered values, native-7 lane head): the G13 dict and the preflight rows the gate-side checks start from"""
    import test_rl_epo as TE
    with tempfile.TemporaryDirectory() as td:
        r = TE.smoke(sf_src, td, epo=True, native7=True, terms=True, tag="so_good", extra=SMOKE_ON)
    assert r["pre"] == 0 and r["pass_json"] is not None, f"the unmutated trainer's constrained preflight did not pass: {r.get('out_pre', '')[-300:]}"
    return {"g13": r["pass_json"]["gates"]["G13_epo_con_live_and_identity"], "pre_rows": r["rows"]["preflight.jsonl"]}


def main() -> int:
    srcs = {p: open(p, encoding="utf-8").read() for p in (SF_PATH, LG_PATH, GC_PATH)}
    import carvefeed_layer as CL
    srcs[SF_PATH] = CL.peel(srcs[SF_PATH])           # B8 (raw/2026-10-18-r0e-carvefeed): this suite is about the trainer BEFORE the carve feed, and the M3-M15 snippets are written against that text (M4's `dual.step(con_acc)` is no longer a line of the B8 trainer): peel ONCE here, so every mutation edits the pre-B8 text
    tip_src = reverse_edits(srcs[SF_PATH])
    assert S1B.git_blob(tip_src.encode("utf-8")) == R0C_TIP_BLOB, "reversing the patch's edits does not give the R0c-tip trainer: the trainer moved -- regenerate SAFEOBJ_FILE_REVERSALS"
    good = build_good(srcs[SF_PATH], tip_src)
    chain_srcs = {p: open(p, encoding="utf-8").read() for p in (RC_PATH, POD_PATH)}
    real = suite_sf(srcs[SF_PATH], "real", tip_src) + suite_g(srcs[LG_PATH], srcs[GC_PATH], "real", good) + suite_ch(chain_srcs[RC_PATH], chain_srcs[POD_PATH], "real")
    assert not real, f"the unmutated sources FAIL their own checks: {real}"
    n, problems = 1, []
    print(f"{'mutation':92s} {'part':4s} caught-by (failing checks)")
    for i, (name, part, edits, envs) in enumerate(MUTATIONS):
        m = apply_edits({**srcs, **chain_srcs}, edits, name)
        try:
            with env(**(envs or {})):
                if part == "SF":
                    _ast.parse(m[SF_PATH])
                    failing = suite_sf(m[SF_PATH], f"m{i}", tip_src)
                elif part == "G":
                    failing = suite_g(m[LG_PATH], m[GC_PATH], f"m{i}", good)
                else:
                    failing = suite_ch(m[RC_PATH], m[POD_PATH], f"m{i}")
        except (Exception, SystemExit) as e:                            # noqa: BLE001 -- the mutated source breaks the suite's own setup: RED, and said so
            failing = [f"suite setup raised {type(e).__name__}: {str(e)[:60]}"]
        if not failing:
            problems.append(f"mutation {name} stayed GREEN: the suite cannot see it")
            continue
        behavioural = [f for f in failing if not f.startswith("P_")]                    # the pins fire on ANY edit of the patched text; the table shows what else saw it
        for c in EXPECT.get(name, ()):
            if c not in failing:
                problems.append(f"mutation {name}: the check {c} must go RED and stayed green (RED: {failing[:8]})")
        print(f"{name[:92]:92s} {part:4s} {', '.join((behavioural or failing)[:4])}" + ("" if behavioural else "   [PIN ONLY]"), flush=True)
        n += 1
    if problems:
        print("\n".join("PROBLEM " + p_ for p_ in problems))
        print(f"ZZTEST_FAIL {len(problems)} problem(s)")
        return 1
    print(f"ZZTEST_PASS {n}  (1 unmutated suite + {len(MUTATIONS)} RED mutations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
