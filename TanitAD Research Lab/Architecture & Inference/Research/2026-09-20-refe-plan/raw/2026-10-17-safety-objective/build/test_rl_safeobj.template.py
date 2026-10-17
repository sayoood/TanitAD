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
NEW_FLAGS = ("--epo-con", "--epo-con-eps", "--epo-con-lr", "--epo-con-max", "--epo-con-lam0")          # the parser additions over the R0c tip (test_rl_r0c's "only new flag" pin allows exactly these on top of --epo-ref)
UNTOUCHED = ("epo_refusals", "epo_loss", "epo_policy", "epo_ramp", "epo_reference", "epo_reference_built", "epo_reward", "_rl_terms_record", "load_rl_terms", "main.epo_term", "main.losses",
             "main.losses6", "main.sel_score", "main.lane_bce", "module_grad_abs", "g11_epo_live_and_identity")

# <<<GEN SAFEOBJ_FILE_REVERSALS
SAFEOBJ_FILE_REVERSALS = ()
# >>>GEN SAFEOBJ_FILE_REVERSALS


def reverse_edits(src: str) -> str:
    """the patched trainer -> the R0c-tip trainer (every edit reversed; each must occur exactly once)"""
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
SEG_NEW = S1B.SAFEOBJ_REVERSALS["main.train_losses"][0][0] if hasattr(S1B, "SAFEOBJ_REVERSALS") else ""


# ================================================================================================ part SF: the trainer
def suite_sf(sf_src: str, tag: str, tip_src: str) -> list:
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
        return {o[0] for o in set(nm) - {tuple(x.option_strings) for x in old_p._actions}} == {"--epo-con", "--epo-con-eps", "--epo-con-lr", "--epo-con-max", "--epo-con-lam0"}
    check(bad, "P_every_other_flag_of_the_parser_unchanged_and_the_five_epo_con_flags_are_the_only_new_ones", other_actions_unchanged)

    # ---------------------------------------------------------------- helpers on the REAL closures
    DS = type("DS", (), {})()
    DS.rows = [{"log_name": f"L{i}", "token": f"t{i}", "step": i, "rank": 0} for i in range(6)]
    rngd = np.random.default_rng(9)
    TERMS_D = {(f"L{i}", f"t{i}", i, 0): {"d": rngd.random(8).astype(np.float32)} for i in range(4)}
    ROWS = torch.tensor([3, 1, 0, 2])
    CL = ["lane_bce", "sel_score", "losses", "losses6", "epo_term", "train_losses"]
    SC = ("score_q_mlp", "score_dec", "score_head")

    def mk(lane, dual=None, names=CL, fwd=None, **kw):
        base = dict(lam=0.0, epo=True, epo_lam=1.0, epo_ramp=2, epo_temp=0.1, epo_beta=0.1, accum=2, epo_ref="base", epo_con="g5", epo_con_eps=0.0, epo_con_lr=4.0, epo_con_max=10.0, epo_con_lam0=0.0)
        base.update(kw)
        an = TE.a_ns("listnet", lane, **base)
        d = dual if dual is not None else M.EpoDual(("ttc", "nc"), 0.0, 4.0, 10.0, 0.0)
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
            d = M.EpoDual(("ttc", "nc") if variant == "lam0" else (), 0.0, 0.0, 10.0, 0.0)
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
            an_, ns_, d_, cl_ = mk(False, dual=M.EpoDual(("ttc", "nc"), e, 4.0, 10.0, 0.5), epo_con_eps=e)
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
        d = M.EpoDual(("ttc", "nc"), 0.0, 4.0, 10.0, 0.0)
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
        return (e.state_dict() == d.state_dict() and set(bf) == {"mode", "comps", "cols", "eps", "lr", "max", "lam0", "ref"} and bf["mode"] == "g5" and bf["comps"] == ["ttc", "nc"]
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
        g = M.g13_epo_con_live_and_identity(A, B, lambda mode: ns["con_probe"](None, ROWS, mode), lambda: M.epo_con_dual_reach(d), lambda: M.epo_con_ckpt_roundtrip(d), kl_tol=0.0, names=SC)
        return g, d
    for lane in (False, True):
        def g13_green(lane=lane):
            g, d = g13_unit(lane)
            return (g["ok"] is True and all(g["live"].values()) and g["controls_exactly_zero"] and g["A_untouched"] and g["A_control_reads_nonzero"] and g["identity_ok"]
                    and g["dual_ok"] and g["checkpoint"]["block_equal"] and not g["mutate"] and g["value_lam0"] == 0.0 and g["value_off"] == 0.0 and d.lam == {"ttc": 0.0, "nc": 0.0})
        check(bad, f"U_G13_green_on_the_real_probe_{'lane' if lane else '6col'}", g13_green)

    def g13_restores_the_run_state():
        g, d = g13_unit(False)
        return d.comps == ("ttc", "nc") and d.lam == {"ttc": 0.0, "nc": 0.0} and d.updates == 0 and d.active
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
    check(bad, "R_hop_main_builds_the_dual_from_the_five_flags", lambda: "dual = EpoDual(EPO_CON_TERMS.get(a.epo_con, ()), a.epo_con_eps, a.epo_con_lr, a.epo_con_max, a.epo_con_lam0)" in msrc)
    check(bad, "R_hop_the_data_event_records_the_BUILT_facts_of_the_dual", lambda: "info['epo_con'] = dual.built_facts()" in msrc)

    def tl_block():
        fn = C.nested["train_losses"]
        epo_if = next(s for s in fn.body if isinstance(s, _ast.If) and _ast.unparse(s.test) == "a.epo")
        con_if = next((s for s in epo_if.body if isinstance(s, _ast.If) and _ast.unparse(s.test) == "dual.active"), None)
        return (con_if is not None and [_ast.unparse(s) for s in con_if.body] == ["ct = epo_con_term(sxB, sxA, tg, sel_score, a.epo_temp, a.epo_con_eps, dual.lam, con_last)", "lB = lB + ct"]
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
    ns_on = argparse.Namespace(epo_con="g5", epo_con_eps=0.0, epo_con_lr=4.0, epo_con_max=10.0, epo_con_lam0=0.0)
    ns_on_l = argparse.Namespace(**{**vars(ns_on), "epo_con_lam0": 0.5})
    import gate_checks as GC0
    check(bad, "E_G13_raw_fields_pass_the_gate_s_own_re_judging", lambda: GC0.judge_g13(R["on"]["pass_json"]["gates"][G13K], ns_on) == [] and GC0.judge_g13(R["on_l"]["pass_json"]["gates"][G13K], ns_on_l) == []
          and GC0.judge_g13(R["on6"]["pass_json"]["gates"][G13K], ns_on) == [])
    check(bad, "E_G13_on_a_real_preflight_every_module_live_A_untouched_controls_exactly_zero", lambda: (lambda g: all(v > 0.0 for v in g["per_module_on"].values()) and set(g["per_module_on"]) == set(SC)
          and all(v == 0.0 for v in g["A_per_module_on"].values()) and all(v > 0.0 for v in g["A_bce_control"].values()) and g["value_lam0"] == 0.0 and g["value_off"] == 0.0
          and g["dual_reach"]["got"] == {"ttc": 0.3125, "nc": 0.3125} and g["identity_ok"] and g["mutate"] == [])(R["on"]["pass_json"]["gates"][G13K]))
    want_facts = {"mode": "g5", "comps": ["ttc", "nc"], "cols": [3, 0], "eps": 0.0, "lr": 4.0, "max": 10.0, "lam0": 0.0, "ref": "armA_detached"}
    check(bad, "E_data_event_records_the_BUILT_facts_in_the_preflight_log_and_the_run_log", lambda: data_of(R["on"])[0]["epo_con"] == want_facts and pre_data_of(R["on"])[0]["epo_con"] == want_facts
          and data_of(R["on_l"])[0]["epo_con"] == {**want_facts, "lam0": 0.5} and data_of(R["off_named"])[0]["epo_con"] == {"mode": "off", "comps": [], "cols": [], "eps": 0.0, "lr": 4.0, "max": 10.0,
                                                                                                                              "lam0": 0.0, "ref": "armA_detached"}
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
    BLK_ON = S.EpoDual(("ttc", "nc"), 0.0, 4.0, 10.0, 0.0).built_facts()
    BLK_OFF = S.EpoDual((), 0.0, 4.0, 10.0, 0.0).built_facts()

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
    for k, v in (("eps", 0.01), ("lr", 2.0), ("max", 5.0), ("lam0", 0.5)):
        check(bad, f"G_dvb_FAIL_built_{k}_differs_from_the_argv", lambda k=k, v=v: ev(r0d, data(nsd, {**BLK_ON, k: v}), pdr)["status"] == "FAIL"
              and any(f"--epo-con-{k}: declared" in r for r in ev(r0d, data(nsd, {**BLK_ON, k: v}), pdr)["reasons"]))
    check(bad, "G_dvb_FAIL_the_block_missing_on_rls1d_even_for_an_off_arm", lambda: ev(r1o, data(nso, None, with_block=False), pdo)["status"] == "FAIL" and ev(r0d, data(nsd, None, with_block=False), pdr)["status"] == "FAIL")
    check(bad, "G_dvb_legacy_trainer_without_the_block_still_passes_on_the_older_profiles_when_off_and_fails_when_g5", lambda: ev(r0, data(S.build_parser().parse_args(r0), None, with_block=False), p0)["status"] == "PASS"
          and ev(argv("R0", CON_ON), data(S.build_parser().parse_args(argv("R0", CON_ON)), None, with_block=False), p0)["status"] == "FAIL")
    check(bad, "G_dvb_the_older_profiles_read_a_real_off_block", lambda: ev(r0, data(S.build_parser().parse_args(r0), BLK_OFF), p0)["status"] == "PASS"
          and ev(r0, data(S.build_parser().parse_args(r0), BLK_ON), p0)["status"] == "FAIL")

    def flag_without_entry():
        for fl in ("epo_con", "epo_con_eps", "epo_con_lr", "epo_con_max", "epo_con_lam0"):
            kept = GC.DVB_KINDS_SCORER.pop(fl)
            try:
                e = ev(r0d, data(nsd, BLK_ON), pdr)
                if not (e["status"] == "FAIL" and any("NO declared-vs-built entry" in r and fl in r for r in e["reasons"])):
                    return False
            finally:
                GC.DVB_KINDS_SCORER[fl] = kept
        return True
    check(bad, "G_a_flag_without_a_G_DVB_entry_is_refused_by_the_gate_itself_all_five", flag_without_entry)
    check(bad, "G_dvb_registry_classifies_the_five_flags_as_built_and_observes_them", lambda: all(GC.DVB_KINDS_SCORER.get(k) == "built" for k in ("epo_con", "epo_con_eps", "epo_con_lr", "epo_con_max", "epo_con_lam0"))
          and GC.SCORER_EPO_CON_KEYS == ("mode", "comps", "cols", "eps", "lr", "max", "lam0", "ref") and GC.SCORER_EPO_CON_COMPS == {"g5": ("ttc", "nc")})

    def built_without_observer():
        for k in ("eps", "lr", "max", "lam0"):          # the numeric rows have a dest of their own; mode / comps / cols / ref share `epo_con`, so dropping ONE of those leaves the dest observed
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
    nsj = argparse.Namespace(epo_con="g5", epo_con_eps=0.0, epo_con_lr=4.0, epo_con_max=10.0, epo_con_lam0=0.0)
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
@CHAIN_SUITE@


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
    ("M27 the call site passes a literal 0.0 for the slack", "SF", [(SF_PATH, "a.epo_temp, a.epo_con_eps, dual.lam, con_last)", "a.epo_temp, 0.0, dual.lam, con_last)")], None),
    ("M17 the loop never accumulates the parts", "SF", [(SF_PATH, '                    con_acc[k_] = con_acc.get(k_, 0.0) + v_\n', "                    pass\n")], None),
    ("M18 the dual reads the wrong flag (eps and eta swapped in main)", "SF", [(SF_PATH, "a.epo_con_eps, a.epo_con_lr, a.epo_con_max, a.epo_con_lam0)", "a.epo_con_lr, a.epo_con_eps, a.epo_con_max, a.epo_con_lam0)")], None),
    ("M19 the data event ECHOES the reference and the selector is broken", "SF", [(SF_PATH, '"lam0": self.lam0, "ref": epo_con_reference_built()}', '"lam0": self.lam0, "ref": "armA_detached"}'),
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
