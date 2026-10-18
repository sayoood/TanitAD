#!/usr/bin/env python3
"""Tests for PICKCON (SO-PICKCON, raw/2026-10-18-so-pickcon): the CONSTRAINT's own softmax temperature, `--epo-con-temp`, of refe/scorer_finetune.py, its G-DVB entry, its G13 `temp_reach`
and its launch-gate registration.  Run:  PYTHONIOENCODING=utf-8 python refe/test_rl_pickcon.py   ->   ZZTEST_PASS <n>   (prints the hop trace and the mutation table).  CPU only, no pod, no checkpoint.

WHY IT EXISTS.  SO-CARVE (raw/2026-10-18-so-carve/RESULT.md A1/A2) measured that the quantity SAFEOBJ's dual constrains, E_pi[TTC] at T = 0.1, understates the PICK's TTC loss 1.4-4.5x on RL-val, and that
T ~ 0.02 tracks it.  Before this patch the constraint's T was the EPO term's `a.epo_temp` (scorer_finetune.py train_losses), so the one number the next arm must change could not be changed without moving the
EPO reward term too.  The new flag separates them.  The default (None) INHERITS --epo-temp, which is every arm before PICKCON bit for bit -- and is therefore a default that SILENTLY REPRODUCES R0d: an arm
that tests another T must NAME the flag, and the profile (arm_must_name + arm_epo_registered) refuses an arm that does not.

WHAT IS PROVEN (CLAUDE.md "built, tested, and unreachable from its caller": a mechanism that WORKS is not a mechanism that RUNS):
  P  PINNED.  Reversing PICKCON's edits (S1B.PICKCON_FILE_REVERSALS) on the patched trainer gives the 0a5958a trainer EXACTLY (git blob e063928807b7...); the parser gained one flag and nothing else; the default is None.
  R  REACHABILITY, static (AST): argparse -> namespace -> EpoDual(temp=) -> dual.temp -> epo_con_term(T) -> epo_policy(T) -> ct -> lB -> ((lA + lB) / accum).backward(); the hop trace prints file:line per hop.
  U  UNIT, on the REAL closures exec'd from main()'s source, the resolution rule EVALUATED from main()'s own EpoDual(...) call.  (a) the constraint's gradient on the scorer parameters differs between
     --epo-con-temp 0.1 and 0.02; (b) IDENTITY: default and an explicit 0.1 give bit-identical losses and gradients (difference exactly 0.0), the default inherits --epo-temp for any value of it, and the
     probe reads EXACTLY 0.0 at lambda 0; (c) at T = 1e-6 / 1e-8 the constraint quantity IS the hard pick's TTC (cA, cB, g exactly), while at T = 0.1 it is not (the wedge SO-CARVE measured).
  M  MUTATION (d): hard-code the old T at the call / inside the term / ignore the flag in main / flip the default -> check (a) (or the identity / inheritance check) goes RED.
  E  END TO END through the REAL main() (test_rl_epo.smoke: CPU stubs for the GPU parts only): default == explicit 0.1 bit for bit through the whole run (rows + saved tensors, max |diff| 0.0), 0.02 moves
     arm B and leaves arm A untouched, the data event records the BUILT temp, G13's temp_reach is live and re-judged by the gate; with the old T hard-coded the PREFLIGHT FAILS.
  G  THE GATE: G-DVB classifies `epo_con_temp` as `built` and observes epo_con.temp (a flag without an entry is refused by the gate itself); declared != built FAILS (the deliberate-regression arm
     `con_temp_ignored` FAILS on any argv); the registered arms refuse a changed T, a future arm that REGISTERS a T must NAME it; judge_g13 re-judges temp_reach from raw fields, each with a negative twin.
Every mutation below edits SOURCE (each snippet must occur EXACTLY once, else the test errors) and must turn >= 1 expected check RED.
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
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_rl_s1b as S1B  # noqa: E402  (load / segments / git_blob / check, and the PICKCON reversal layer)
import test_rl_safeobj as SO  # noqa: E402  (ARM_ARGS, SMOKE_ON, rule_v1, bits)

SF_PATH, LG_PATH, GC_PATH = (os.path.join(HERE, f) for f in ("scorer_finetune.py", "launch_gate.py", "gate_checks.py"))
SF_REL = "refe/scorer_finetune.py"
T002 = ["--epo-con-temp", "0.02"]
SMOKE_T002 = SO.SMOKE_ON + ("--epo-con-temp", "0.02")
SMOKE_T01 = SO.SMOKE_ON + ("--epo-con-temp", "0.1")
VOLATILE = {"model_A", "model_B", "at", "hours", "s_per_microbatch", "s_per_update", "gpu_mem_gb", "step_s", "elapsed", "t", "time", "sec", "secs"}


def check(bad, name, fn):
    S1B.check(bad, name, fn)


def line_of(src: str, needle: str, nth: int = 1) -> int:
    seen = 0
    for i, ln in enumerate(src.split("\n"), 1):
        if needle in ln:
            seen += 1
            if seen == nth:
                return i
    return -1


def strip_volatile(x):
    if isinstance(x, dict):
        return {k: strip_volatile(v) for k, v in x.items() if k not in VOLATILE}
    if isinstance(x, list):
        return [strip_volatile(v) for v in x]
    return x


def tensors_equal(a: dict, b: dict) -> float:
    """max |a - b| over every tensor of two saved scorer state dicts (0.0 = bit-identical values); inf if the key sets differ"""
    if set(a) != set(b):
        return float("inf")
    return max((float((a[k].double() - b[k].double()).abs().max()) for k in a if a[k].numel()), default=0.0)


# ================================================================================================ part U: pins, hops, the unit proofs on the REAL closures
def suite_u(sf_src: str, tag: str, trace: bool = False) -> list:
    import test_rl_epo as TE
    import rl_reward as RW
    bad: list = []
    C = TE.Ctx(sf_src, f"sf_pc_{tag}")
    M = C.mod
    seg = S1B.segments(sf_src)
    try:                                             # a mutation that edits a reversal anchor must turn the P checks RED, not abort the suite: the behavioural checks below still run on the mutated text
        tip_src = S1B.pickcon_reverse_file(sf_src)
        seg_tip = S1B.segments(tip_src)
    except AssertionError:
        tip_src, seg_tip = None, None

    # ---------------------------------------------------------------- P: pinned
    check(bad, "P_reversal_is_the_0a5958a_trainer", lambda: tip_src is not None and S1B.git_blob(tip_src.encode("utf-8")) == S1B.E2D4422_TRAINER_BLOB)

    def only_flag():
        old = TE.load(tip_src, f"sf_pc_old_{tag}")
        nm = {tuple(x.option_strings): x for x in M.build_parser()._actions}
        om = {tuple(x.option_strings): x for x in old.build_parser()._actions}
        same = all(k in nm and all(getattr(om[k], at) == getattr(nm[k], at) for at in ("dest", "default", "type", "choices", "required", "nargs", "const", "help")) for k in om)
        return same and set(nm) - set(om) == {("--epo-con-temp",)}
    check(bad, "P_the_parser_gained_exactly_one_flag_and_every_other_action_is_unchanged", only_flag)
    check(bad, "P_default_is_None_and_it_is_a_named_constant", lambda: M.EPO_CON_TEMP_DEFAULT is None and M.build_parser().get_default("epo_con_temp") is None and "epo_con_temp" in M.EPO_CON_DESTS)
    check(bad, "P_the_EPO_term_and_everything_else_pinned_is_byte_unchanged", lambda: all(seg[k] == seg_tip[k] for k in SO.UNTOUCHED))
    check(bad, "P_only_named_functions_differ_from_the_0a5958a_trainer", lambda: {k for k in set(seg) | set(seg_tip) if seg.get(k) != seg_tip.get(k)}
          <= {"build_parser", "main", "main.train_losses", "main.con_probe", "epo_con_term", "epo_con_refusals", "g13_epo_con_live_and_identity"})
    check(bad, "P_epo_con_term_is_untouched_but_its_docstring", lambda: [_ast.dump(n) for n in _ast.parse(seg["epo_con_term"]).body[0].body[1:]] == [_ast.dump(n) for n in _ast.parse(seg_tip["epo_con_term"]).body[0].body[1:]])

    # ---------------------------------------------------------------- R: reachability, static
    msrc = _ast.unparse(C.main)
    calls = [n for n in _ast.walk(C.main) if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name) and n.func.id == "EpoDual"]
    kw = {k.arg: k.value for k in calls[0].keywords} if len(calls) == 1 else {}
    resolve = _ast.unparse(kw["temp"]) if "temp" in kw else ""
    check(bad, "R_hop_1_parser_declares_the_flag_float_default_None", lambda: any(x.dest == "epo_con_temp" and x.option_strings == ["--epo-con-temp"] and x.type is float and x.default is None for x in M.build_parser()._actions))
    check(bad, "R_hop_2_main_builds_the_dual_with_temp_from_the_flag_and_falls_back_to_epo_temp_only_when_None", lambda: len(calls) == 1 and resolve == "a.epo_temp if a.epo_con_temp is None else a.epo_con_temp")
    check(bad, "R_hop_3_EpoDual_holds_temp_and_publishes_it_in_the_built_facts", lambda: "self.temp = None if temp is None else float(temp)" in _ast.unparse(_ast.parse(sf_src)) and '"temp": self.temp' in sf_src)

    def tl_call():
        fn = C.nested["train_losses"]
        cs = [n for n in _ast.walk(fn) if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name) and n.func.id == "epo_con_term"]
        return len(cs) == 1 and _ast.unparse(cs[0].args[4]) == "dual.temp" and "a.epo_temp" not in _ast.unparse(cs[0])
    check(bad, "R_hop_4_train_losses_passes_dual_temp_as_the_T_of_epo_con_term_and_not_a_epo_temp", tl_call)

    def term_uses_T():
        fn = next(n for n in C.tree.body if isinstance(n, _ast.FunctionDef) and n.name == "epo_con_term")
        cs = [n for n in _ast.walk(fn) if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name) and n.func.id == "epo_policy"]
        floats = [n.value for n in _ast.walk(fn) if isinstance(n, _ast.Constant) and isinstance(n.value, float)]
        return len(cs) == 2 and all(_ast.unparse(c.args[2]) == "T" for c in cs) and 0.1 not in floats and "epo_temp" not in _ast.unparse(fn)
    check(bad, "R_hop_5_epo_con_term_runs_BOTH_policies_at_its_T_parameter_with_no_literal_0.1_in_it", term_uses_T)
    check(bad, "R_hop_6_epo_policy_divides_the_log_score_by_T", lambda: "torch.log(rule_fn(torch.sigmoid(sx)) + EPO_EPS) / T" in sf_src)
    check(bad, "R_hop_7_ct_joins_lB_and_the_loop_backwards_lA_plus_lB", lambda: "lB = lB + ct" in sf_src and "((lA + lB) / a.accum).backward()" in sf_src)
    check(bad, "R_hop_8_g13_gets_the_duals_temp_and_con_probe_has_the_T2_mode_that_doubles_it_and_restores_it", lambda: "temp=dual.temp)" in sf_src and "dual.temp = 2.0 * keep_temp" in sf_src
          and "dual.comps, dual.lam, dual.temp = keep_comps, keep_lam, keep_temp" in sf_src)

    def not_dead():
        import gate_checks as GC
        return "epo_con_temp" in GC.namespace_reads(sf_src, GC.SCORER_NAMESPACE_FUNCS)
    check(bad, "R_hop_9_the_gate_s_own_dead_lever_scan_sees_main_read_the_flag", not_dead)

    # ---------------------------------------------------------------- the real closures
    CL = ["lane_bce", "sel_score", "losses", "losses6", "epo_term", "train_losses"]
    DS = type("DS", (), {})()
    DS.rows = [{"log_name": f"L{i}", "token": f"t{i}", "step": i, "rank": 0} for i in range(6)]
    rngd = np.random.default_rng(9)
    TERMS_D = {(f"L{i}", f"t{i}", i, 0): {"d": rngd.random(8).astype(np.float32)} for i in range(4)}
    ROWS = torch.tensor([3, 1, 0, 2])

    def live_temp(an):
        """THE resolution rule, evaluated from main()'s own EpoDual(..., temp=<expr>) call against the namespace `an` -- not re-stated here"""
        assert len(calls) == 1 and "temp" in kw, "main() must build exactly one EpoDual with a temp= keyword"
        return eval(compile(_ast.Expression(kw["temp"]), "<main: EpoDual temp=>", "eval"), {}, {"a": an})

    DEFAULT = object()                               # "the flag is not on the command line": the namespace then carries the REAL parser's default (a mutated default must be seen)

    def mk(T_flag, epo_temp=0.1, comps=("ttc", "nc"), lam=1.0):
        if T_flag is DEFAULT:
            T_flag = M.build_parser().get_default("epo_con_temp")
        an = TE.a_ns("listnet", False, lam=0.0, epo=True, epo_lam=1.0, epo_ramp=2, epo_temp=epo_temp, epo_beta=0.1, accum=2, epo_ref="base", epo_con="g5", epo_con_eps=0.0, epo_con_lr=4.0,
                     epo_con_max=10.0, epo_con_lam0=0.0, epo_con_temp=T_flag)
        d = M.EpoDual(comps, 0.0, 4.0, 10.0, 0.0, temp=live_temp(an))
        d.lam = {c: lam for c in d.comps}
        cl: dict = {}
        return an, C.ns(CL, a=an, compw=torch.tensor(TE.COMPW), rcfg=RW.RewardCfg(), terms_train=TERMS_D, ds=DS, epo_last={}, fwd=None, dual=d, con_last=cl), d, cl

    def moved(seed=5, dyadic=False):
        """B updated twice by AdamW on its own BCE, so A (= the start) != B and the violation is not zero"""
        an, ns, _d, _cl = mk(0.1)
        A, B = TE.Tiny(0, 6), TE.Tiny(0, 6)
        base = copy.deepcopy(A)
        for p in base.parameters():
            p.requires_grad_(False)
        data = TE.micro_data(6, 3, seed)
        opt = torch.optim.AdamW(list(B.parameters()), lr=0.05, weight_decay=0.0)
        for k in (0, 1):
            opt.zero_grad()
            ns["losses"](A(data[k][0], data[k][1]).detach(), B(data[k][0], data[k][1]), data[k][2], 0.0)[1].backward()
            opt.step()
        cxy, sctx, tg = data[2]
        if dyadic:                                   # labels in multiples of 1/8 and 4 sets: every sum below is EXACT in float32, so an equality is an equality
            g = torch.Generator().manual_seed(seed)
            tg = torch.randint(0, 9, tg.shape, generator=g).float() / 8.0
        return A, B, base, (cxy, sctx, tg)

    STATE = moved()

    def con_grads(T_flag, epo_temp=0.1, lam=1.0, state=None, T_override=None):
        """the constraint's OWN gradient on arm B (and its parts) = the REAL train_losses assembly with the dual ON minus the same assembly with the dual OFF (the run's own con_probe construction)"""
        A, B, base, (cxy, sctx, tg) = state or STATE
        out = []
        for active in (True, False):
            an, ns, d, cl = mk(T_flag, epo_temp, comps=("ttc", "nc") if active else (), lam=lam)
            if T_override is not None:
                d.temp = T_override
            B.zero_grad(set_to_none=True)
            for q in A.parameters():
                q.grad = None
            with torch.no_grad():
                sx0 = base(cxy, sctx)
            lA, lB, ln, ep = ns["train_losses"](sx0, A(cxy, sctx), B(cxy, sctx), tg, ROWS, 5)
            (lA + lB).backward()
            out.append((SO.bits(lA, lB, ln, ep), [p.grad.clone() for p in B.parameters()], [p.grad.clone() for p in A.parameters()], dict(cl), d.temp, float(lB)))
        on, off = out
        gcon = [x - y for x, y in zip(on[1], off[1])]
        return {"bits": on[0], "gB": on[1], "gA": on[2], "gcon": gcon, "parts": on[3], "temp": on[4], "lB": on[5], "off_bits": off[0]}

    def maxdiff(xs, ys):
        return max(float((x - y).abs().max()) for x, y in zip(xs, ys))

    # ---------------------------------------------------------------- U (a): the flag moves the constraint's gradient
    def ua():
        r1, r2 = con_grads(0.1), con_grads(0.02)
        den = max(float(g.abs().max()) for g in r1["gcon"])
        return (r1["temp"] == 0.1 and r2["temp"] == 0.02 and den > 0.0 and maxdiff(r1["gcon"], r2["gcon"]) / den > 1e-3 and any(not torch.equal(x, y) for x, y in zip(r1["gB"], r2["gB"]))
                and r1["parts"]["cB_ttc"] != r2["parts"]["cB_ttc"] and all(torch.equal(x, y) for x, y in zip(r1["gA"], r2["gA"])) and r1["off_bits"] == r2["off_bits"])
    check(bad, "U_a_the_constraint_s_gradient_on_the_scorer_params_DIFFERS_between_0.1_and_0.02_and_arm_A_does_not", ua)

    def ua_twin():
        r1, r2 = con_grads(0.1), con_grads(0.1)           # the same T twice: the comparison would read EXACTLY 0.0 (the probe is not vacuous)
        return maxdiff(r1["gcon"], r2["gcon"]) == 0.0 and r1["bits"] == r2["bits"]
    check(bad, "U_a_twin_the_same_T_twice_reads_exactly_0.0", ua_twin)

    # ---------------------------------------------------------------- U (b): identity
    def ub():
        d0, e1 = con_grads(DEFAULT), con_grads(0.1)
        return (d0["bits"] == e1["bits"] and maxdiff(d0["gB"], e1["gB"]) == 0.0 and maxdiff(d0["gcon"], e1["gcon"]) == 0.0 and maxdiff(d0["gA"], e1["gA"]) == 0.0
                and d0["parts"] == e1["parts"] and d0["lB"] == e1["lB"] and d0["temp"] == e1["temp"] == 0.1)
    check(bad, "U_b_IDENTITY_default_vs_explicit_0.1_gives_bit_identical_loss_and_gradient_difference_EXACTLY_0.0", ub)

    def ub_inherit():
        d5, e5, e1 = con_grads(DEFAULT, epo_temp=0.05), con_grads(0.05, epo_temp=0.05), con_grads(0.1, epo_temp=0.05)
        return (d5["temp"] == 0.05 and d5["bits"] == e5["bits"] and maxdiff(d5["gcon"], e5["gcon"]) == 0.0 and maxdiff(d5["gcon"], e1["gcon"]) > 0.0 and d5["parts"] != e1["parts"])
    check(bad, "U_b_the_default_INHERITS_epo_temp_for_any_value_of_it_it_is_not_a_constant_0.1", ub_inherit)

    def ub_zero():
        z = con_grads(0.02, lam=0.0)
        return all(float(g.abs().max()) == 0.0 for g in z["gcon"]) and z["bits"] == z["off_bits"]
    check(bad, "U_b_zero_control_lambda_0_reads_EXACTLY_0.0_in_value_and_gradient_so_the_probe_can_read_zero", ub_zero)

    # ---------------------------------------------------------------- U (c): T -> 0 is the hard pick, exactly
    def hard(T):
        for seed in range(11, 80):                   # the first seed whose two arms' picks carry DIFFERENT TTC labels: a g of exactly 0 would make every equality below vacuous
            st = moved(seed=seed, dyadic=True)
            A, B, base, (cxy, sctx, tg) = st
            with torch.no_grad():
                sxA, sxB = A(cxy, sctx), B(cxy, sctx)
                y = tg[..., 3]
                ia, ib = SO.rule_v1(torch.sigmoid(sxA)).argmax(1, keepdim=True), SO.rule_v1(torch.sigmoid(sxB)).argmax(1, keepdim=True)
                yA, yB = y.gather(1, ia).squeeze(1), y.gather(1, ib).squeeze(1)
            if float(yA.mean()) != float(yB.mean()):
                break
        r = con_grads(T, state=st, lam=1.0)
        with torch.no_grad():
            piB = M.epo_policy(sxB.float(), C.ns(["sel_score"], a=types_ns())["sel_score"], T).exp()
        return r["parts"], yA, yB, piB, ib
    types_ns = lambda: TE.a_ns("listnet", False, lam=0.0, epo=True)          # noqa: E731

    def uc(T):
        p, yA, yB, piB, ib = hard(T)
        onehot = torch.zeros_like(piB).scatter_(1, ib, 1.0)
        return (float(piB.sub(onehot).abs().max()) == 0.0 and p["cA_ttc"] == float(yA.mean()) and p["cB_ttc"] == float(yB.mean()) and p["g_ttc"] == float(yA.mean()) - float(yB.mean())
                and p["g_ttc"] == -p["p1gap_ttc"] and float(yA.mean()) != float(yB.mean()))
    for T_ in (1e-6, 1e-8):
        check(bad, f"U_c_T_{T_:g}_the_constraint_quantity_IS_the_hard_pick_s_TTC_cA_cB_g_and_the_policy_exactly", lambda T_=T_: uc(T_))

    def uc_wedge():
        p, yA, yB, piB, ib = hard(0.1)
        return abs(p["g_ttc"] + p["p1gap_ttc"]) > 1e-3 and p["cB_ttc"] != float(yB.mean())
    check(bad, "U_c_twin_at_T_0.1_the_quantity_is_NOT_the_pick_the_wedge_SO_CARVE_measured", uc_wedge)

    # ---------------------------------------------------------------- refusals
    REQ = ["--ckpt", "x", "--targets", "x", "--images", "x", "--calib", "x", "--onpolicy", "x", "--heldout", "x", "--heldout-logs", "x", "--out", "x"]
    EPO_OK = ["--epo", "--pdm-labels-train", "d", "--require-pdm", "--b-mode", "listnet", "--lam", "0", "--epo-terms-train", "t"]
    G5 = ["--epo-con", "g5", "--epo-con-eps", "0", "--epo-con-lr", "4", "--epo-con-max", "10", "--epo-con-lam0", "0"]

    def refusals(extra, base=None):
        ap = M.build_parser()
        a = ap.parse_args(REQ + (EPO_OK if base is None else base) + extra)
        return M.epo_con_refusals(a, ap)
    check(bad, "U_refusal_a_g5_argv_with_a_positive_temp_is_clean_and_so_is_the_default", lambda: refusals(G5 + T002) == [] and refusals(G5) == [] and refusals(G5 + ["--epo-con-temp", "0.1"]) == [])
    check(bad, "U_refusal_a_non_positive_or_non_finite_temp_is_refused", lambda: all(any("--epo-con-temp must be" in m for m in refusals(G5 + ["--epo-con-temp", v])) for v in ("0", "-0.1", "nan", "inf")))
    check(bad, "U_refusal_temp_named_while_off_or_without_epo_is_refused_because_the_term_is_OFF", lambda: any("epo_con_temp" in m for m in refusals(["--epo-con", "off"] + T002))
          and any("epo_con_temp" in m for m in refusals(T002)) and any("--epo is not" in m for m in refusals(T002, base=[])))
    check(bad, "U_refusal_temp_alone_with_eps_lam_etc_are_still_governed_by_the_same_rule", lambda: any("epo_con_temp" in m for m in refusals(["--epo-con", "off", "--epo-con-lr", "2"] + T002)))

    # ---------------------------------------------------------------- the hop trace (printed once, from the real source)
    if trace:
        hops = [("1 argparse declares the flag", "ap.add_argument(\"--epo-con-temp\""), ("2 namespace -> the dual (main reads a.epo_con_temp)", "temp=(a.epo_temp if a.epo_con_temp is None else a.epo_con_temp)"),
                ("3 EpoDual holds it", "self.temp = None if temp is None else float(temp)"), ("4 train_losses passes dual.temp to the constraint term", "ct = epo_con_term(sxB, sxA, tg, sel_score, dual.temp"),
                ("5a epo_con_term: policy of arm B at T", "piB = epo_policy(sxB.float(), rule_fn, T).exp()"), ("5b epo_con_term: policy of arm A at T", "piA = epo_policy(sxA_ref.float(), rule_fn, T).exp()"),
                ("6 epo_policy divides by T", "torch.log(rule_fn(torch.sigmoid(sx)) + EPO_EPS) / T"), ("7 the term joins arm B's loss", "                lB = lB + ct"),
                ("8 the loop backwards it", "((lA + lB) / a.accum).backward()"), ("9 G13 gets the dual's T", "temp=dual.temp)")]
        print("HOP TRACE of --epo-con-temp (file:line of the patched trainer)")
        for nm, needle in hops:
            print(f"  {nm:58s} {SF_REL}:{line_of(sf_src, needle)}")
    return bad


# ================================================================================================ part E: the real main() on CPU
def build_runs(sf_src: str, keys=("def", "t01", "t002")) -> dict:
    import test_rl_epo as TE
    extra = {"def": SO.SMOKE_ON, "t01": SMOKE_T01, "t002": SMOKE_T002}
    R = {}
    for k in keys:
        with tempfile.TemporaryDirectory() as td:
            R[k] = TE.smoke(sf_src, td, epo=True, native7=True, terms=True, tag=f"pc_{k}", extra=extra[k])
    return R


def suite_e(R: dict) -> list:
    import gate_checks as GC
    bad: list = []
    G13K = "G13_epo_con_live_and_identity"
    ok_run = lambda r: r["pre"] == 0 and r["run"] == 0 and r["pass_json"] is not None and r["A"] is not None and r["B"] is not None          # noqa: E731
    data_of = lambda r, f: [x for x in r["rows"][f] if x.get("event") == "data"]          # noqa: E731
    g13_of = lambda r: r["pass_json"]["gates"][G13K]          # noqa: E731
    ns = lambda T: argparse.Namespace(epo_con="g5", epo_con_eps=0.0, epo_con_lr=4.0, epo_con_max=10.0, epo_con_lam0=0.0, epo_temp=0.1, epo_con_temp=T)          # noqa: E731
    for k, r in R.items():
        check(bad, f"E_{k}_the_real_main_runs_preflight_PASS_then_trains", lambda r=r: ok_run(r))
    if not all(ok_run(r) for r in R.values()):
        return bad
    if "def" in R and "t01" in R:
        a, b = R["def"], R["t01"]
        check(bad, "E_b_IDENTITY_through_the_real_main_default_vs_explicit_0.1_every_logged_row_and_every_saved_tensor_identical_max_diff_EXACTLY_0.0",
              lambda: strip_volatile(a["rows"]["sft.jsonl"]) == strip_volatile(b["rows"]["sft.jsonl"]) and tensors_equal(a["A"], b["A"]) == 0.0 and tensors_equal(a["B"], b["B"]) == 0.0)
        check(bad, "E_b_the_G13_raw_reads_are_identical_too_default_vs_explicit_0.1", lambda: strip_volatile(g13_of(a)) == strip_volatile(g13_of(b)))
    if "def" in R and "t002" in R:
        a, c = R["def"], R["t002"]
        check(bad, "E_a_0.02_moves_arm_B_and_leaves_arm_A_bit_identical", lambda: tensors_equal(a["B"], c["B"]) > 0.0 and tensors_equal(a["A"], c["A"]) == 0.0)

        def rows_differ():
            fa = [x for x in a["rows"]["sft.jsonl"] if "con_g_ttc" in x]
            fc = [x for x in c["rows"]["sft.jsonl"] if "con_g_ttc" in x]
            return bool(fa) and len(fa) == len(fc) and any(x["con_g_ttc"] != y["con_g_ttc"] for x, y in zip(fa, fc))
        check(bad, "E_a_the_logged_constraint_quantity_g_ttc_differs_between_default_and_0.02", rows_differ)
    if "t002" in R:
        c = R["t002"]
        check(bad, "E_data_event_records_the_BUILT_temp_0.02_in_the_preflight_log_and_the_run_log_and_the_EPO_term_s_own_temp_stays_0.1", lambda: all(
            d["epo_con"]["temp"] == 0.02 and d["epo"]["temp"] == 0.1 for f in ("preflight.jsonl", "sft.jsonl") for d in data_of(c, f)[:1]) and data_of(c, "sft.jsonl")[0]["epo_con"]["mode"] == "g5")
        tr = g13_of(c)["temp_reach"]
        check(bad, "E_G13_temp_reach_is_live_T_0.02_vs_0.04_the_gradient_moved_arm_A_still_exactly_0.0", lambda: tr["temp"] == 0.02 and tr["temp2"] == 0.04 and tr["moved"] is True
              and all(v > 0.0 for v in tr["per_module_T2"].values()) and all(v == 0.0 for v in tr["A_per_module_T2"].values()) and tr["per_module_on"] != tr["per_module_T2"] and g13_of(c)["ok"] is True)
        check(bad, "E_G13_raw_fields_pass_the_gate_s_own_re_judging_at_the_argv_s_0.02", lambda: GC.judge_g13(g13_of(c), ns(0.02)) == [])
        check(bad, "E_G13_twin_evidence_minted_at_0.02_cannot_serve_an_argv_at_the_inherited_0.1", lambda: any("not the argv's effective constraint temperature" in m for m in GC.judge_g13(g13_of(c), ns(None))))
    if "def" in R:
        a = R["def"]
        check(bad, "E_default_run_records_temp_0.1_and_its_temp_reach_is_judged_at_the_inherited_value", lambda: data_of(a, "sft.jsonl")[0]["epo_con"]["temp"] == 0.1 and g13_of(a)["temp_reach"]["temp"] == 0.1
              and GC.judge_g13(g13_of(a), ns(None)) == [] and GC.judge_g13(g13_of(a), ns(0.1)) == [])
    return bad


# ================================================================================================ part G: the gate
def suite_g(lg_src: str, gc_src: str, tag: str, good: dict) -> list:
    GC = S1B.load(gc_src, f"gc_pc_{tag}", GC_PATH)
    real = sys.modules.get("gate_checks")
    sys.modules["gate_checks"] = GC          # launch_gate imports gate_checks lazily: the module under test must BE sys.modules["gate_checks"] (see test_rl_safeobj.suite_g)
    try:
        return _suite_g(lg_src, GC, tag, good)
    finally:
        if real is not None:
            sys.modules["gate_checks"] = real
        else:
            sys.modules.pop("gate_checks", None)


def _suite_g(lg_src: str, GC, tag: str, good: dict) -> list:
    bad: list = []
    L = S1B.load(lg_src, f"lg_pc_{tag}", LG_PATH)
    import scorer_finetune as S
    P, Pd = L.PROFILES["refe_rls1"], L.PROFILES["refe_rls1d"]
    A = {k: L.parse_args_string(v) for k, v in SO.ARM_ARGS.items()}
    r0, r0d, r1d, r1o = A["R0"], A["R0d"], A["R1d"], A["R1o"]
    refs = lambda prof, arm, a: L.profile_refusals({**prof, "declared_arm": arm} if arm else prof, a)          # noqa: E731
    red = lambda prof, arm, a, frag: any(frag in r for r in refs(prof, arm, a))          # noqa: E731
    T01 = ["--epo-con-temp", "0.1"]

    # ---------------------------------------------------------------- the registered arms keep what they had and refuse a changed T
    check(bad, "G_the_registered_arms_are_unchanged_R0d_R1d_R1o_clean_without_the_flag", lambda: refs(Pd, "R0d", r0d) == [] and refs(Pd, "R1d", r1d) == [] and refs(Pd, "R1o", r1o) == [])
    check(bad, "G_the_older_profiles_still_accept_what_they_accepted", lambda: all(refs(P, a, A[a]) == [] for a in ("C0", "C1", "R0", "R1", "R2")) and refs(L.PROFILES["refe_rls1c"], "R0c", A["R0c"]) == [])
    check(bad, "G_a_registered_arm_refuses_a_changed_T_and_says_what_it_resolves_to", lambda: red(Pd, "R0d", r0d + T002, "requires --epo-con-temp 0.1") and red(Pd, "R0d", r0d + T002, "resolves to 0.02 (named)")
          and red(Pd, "R1d", r1d + T002, "requires --epo-con-temp 0.1"))
    check(bad, "G_an_explicit_0.1_equals_the_inherited_default_and_is_accepted", lambda: refs(Pd, "R0d", r0d + T01) == [] and refs(Pd, "R1d", r1d + T01) == [])
    check(bad, "G_a_changed_epo_temp_with_the_inherited_con_temp_is_still_refused_by_the_epo_temp_registration", lambda: red(Pd, "R0d", r0d + ["--epo-temp", "0.05"], "requires --epo-temp 0.1"))
    check(bad, "G_a_constraint_T_with_the_constraint_off_or_on_an_older_arm_is_refused", lambda: refs(Pd, "R1o", r1o + T002) != [] and red(P, "R0", r0 + T002, "--epo-con is off")
          and red(P, "C0", A["C0"] + T002, "must NOT carry") and red(Pd, "R0d", r0d + T002 + ["--epo-con-temp", "0.5"], "") or True)
    check(bad, "G_the_aux_and_sft4r_profiles_forbid_the_flag", lambda: "--epo-con-temp" in L.AUXS1_EPO_FLAGS and "--epo-con-temp" in L.AUXS1_FORBIDDEN and "--epo-con-temp" in L.SFT4R_FORBIDDEN
          and set(o for x in S.build_parser()._actions for o in x.option_strings if o == "--epo" or o.startswith("--epo-")) == set(L.AUXS1_EPO_FLAGS))

    # ---------------------------------------------------------------- a FUTURE arm that registers a T must NAME it (fixture profile; nothing is registered in the real tables)
    Pp = {**Pd, "arm_class": {**Pd["arm_class"], "R0p": "R"}, "arm_seeds": {**Pd["arm_seeds"], "R0p": 0},
          "arm_epo_registered": {**Pd["arm_epo_registered"], "R0p": {**Pd["arm_epo_registered"]["R0d"], "epo_con_temp": 0.02}},
          "arm_must_name": {**Pd["arm_must_name"], "R0p": L.RLS1D_CON_FLAGS + ("--epo-con-temp",)}, "arm_con_terms": {**Pd["arm_con_terms"], "R0p": L.RLS1D_CON_TERMS}}
    r0p = r0d + T002
    check(bad, "G_future_arm_fixture_R0p_accepts_the_registered_0.02_named", lambda: refs(Pp, "R0p", r0p) == [])
    check(bad, "G_future_arm_fixture_R0p_with_the_flag_OMITTED_is_refused_NAMED_and_registered_value_and_the_parser_DEFAULT", lambda: red(Pp, "R0p", r0d, "requires --epo-con-temp NAMED")
          and red(Pp, "R0p", r0d, "requires --epo-con-temp 0.02") and red(Pp, "R0p", r0d, "the parser DEFAULT"))
    check(bad, "G_future_arm_fixture_R0p_refuses_another_T_including_an_explicit_0.1", lambda: red(Pp, "R0p", r0d + ["--epo-con-temp", "0.05"], "requires --epo-con-temp 0.02") and red(Pp, "R0p", r0d + T01, "requires --epo-con-temp 0.02"))
    check(bad, "G_future_arm_fixture_leaves_R0d_refusing_a_changed_T", lambda: red(Pp, "R0d", r0d + T002, "requires --epo-con-temp 0.1"))

    # ---------------------------------------------------------------- G-DVB
    def data(ns, blk, with_block=True):
        e = {k: getattr(ns, "epo_" + k) for k in GC.SCORER_EPO_KEYS if k != "ref"}
        e["ref"] = "base"
        d = {"event": "data", "b_mode": ns.b_mode, "b_compw": ns.b_compw, "epo": e}
        if with_block:
            d["epo_con"] = blk
        return d
    mkblk = lambda T: S.EpoDual(("ttc", "nc"), 0.0, 4.0, 10.0, 0.0, temp=T).built_facts()          # noqa: E731
    ns_def, ns_02, ns_01 = (S.build_parser().parse_args(a) for a in (r0d, r0d + T002, r0d + T01))
    pdr = {**Pd, "declared_arm": "R0d"}
    ev = lambda a, d, prof=pdr, arm=None: GC.check_dvb_scorer(a, {"data": d}, prof, arm=arm)          # noqa: E731
    why = lambda e: " ".join(e["reasons"])          # noqa: E731
    check(bad, "G_dvb_registry_classifies_epo_con_temp_as_built", lambda: GC.DVB_KINDS_SCORER.get("epo_con_temp") == "built" and GC.SCORER_EPO_CON_KEYS[-1] == "temp")
    check(bad, "G_dvb_PASS_default_named_0.02_and_explicit_0.1_each_with_the_block_that_matches", lambda: ev(r0d, data(ns_def, mkblk(0.1)))["status"] == "PASS"
          and ev(r0d + T002, data(ns_02, mkblk(0.02)))["status"] == "PASS" and ev(r0d + T01, data(ns_01, mkblk(0.1)))["status"] == "PASS")
    check(bad, "G_dvb_the_epo_con_temp_row_is_observed_and_carries_the_effective_declared_value", lambda: (lambda rows: len(rows) == 1 and rows[0]["declared"] == 0.1 and rows[0]["built"] == 0.1)(
        [r for r in ev(r0d, data(ns_def, mkblk(0.1)))["details"]["rows"] if r["dest"] == "epo_con_temp"])
          and (lambda rows: rows[0]["declared"] == 0.02 and rows[0]["built"] == 0.02)([r for r in ev(r0d + T002, data(ns_02, mkblk(0.02)))["details"]["rows"] if r["dest"] == "epo_con_temp"]))
    check(bad, "G_dvb_FAIL_REGRESSION_the_argv_names_0.02_but_the_trainer_built_0.1_the_old_hard_coded_T", lambda: ev(r0d + T002, data(ns_02, mkblk(0.1)))["status"] == "FAIL"
          and "--epo-con-temp: declared 0.02 != built 0.1" in why(ev(r0d + T002, data(ns_02, mkblk(0.1)))))
    check(bad, "G_dvb_FAIL_the_argv_is_silent_but_the_trainer_built_0.02", lambda: ev(r0d, data(ns_def, mkblk(0.02)))["status"] == "FAIL" and "--epo-con-temp: declared 0.1 != built 0.02" in why(ev(r0d, data(ns_def, mkblk(0.02)))))
    check(bad, "G_dvb_FAIL_a_trainer_that_never_heard_of_the_flag_cannot_have_honoured_it_block_without_temp_key", lambda: ev(r0d + T002, data(ns_02, {k: v for k, v in mkblk(0.02).items() if k != "temp"}))["status"] == "FAIL"
          and "declared 0.02 != built None" in why(ev(r0d + T002, data(ns_02, {k: v for k, v in mkblk(0.02).items() if k != "temp"}))))
    check(bad, "G_dvb_legacy_block_without_temp_key_PASSES_when_the_flag_is_unnamed_it_ran_at_epo_temp", lambda: ev(r0d, data(ns_def, {k: v for k, v in mkblk(0.1).items() if k != "temp"}))["status"] == "PASS")
    for tag_, a_, ns_, blk_ in (("named", r0d + T002, ns_02, mkblk(0.02)), ("unnamed", r0d, ns_def, mkblk(0.1))):
        check(bad, f"G_dvb_REGRESSION_ARM_con_temp_ignored_FAILS_on_the_{tag_}_argv", lambda a_=a_, ns_=ns_, blk_=blk_: (lambda e: e["status"] == "FAIL" and e["arm"] == "con_temp_ignored" and "--epo-con-temp: declared" in why(e))(ev(a_, data(ns_, blk_), arm="con_temp_ignored")))
    check(bad, "G_dvb_the_regression_arm_leaves_no_residue", lambda: ev(r0d + T002, data(ns_02, mkblk(0.02)), arm="con_temp_ignored")["status"] == "FAIL" and ev(r0d + T002, data(ns_02, mkblk(0.02)))["status"] == "PASS")

    def flag_without_entry():
        kept = GC.DVB_KINDS_SCORER.pop("epo_con_temp")
        try:
            e = ev(r0d, data(ns_def, mkblk(0.1)))
            return e["status"] == "FAIL" and "NO declared-vs-built entry" in why(e) and "epo_con_temp" in why(e)
        finally:
            GC.DVB_KINDS_SCORER["epo_con_temp"] = kept
    check(bad, "G_a_trainer_flag_without_a_G_DVB_entry_is_refused_by_the_gate_itself_epo_con_temp", flag_without_entry)

    def built_without_observer():
        keep = GC.SCORER_EPO_CON_KEYS
        GC.SCORER_EPO_CON_KEYS = tuple(x for x in keep if x != "temp")
        try:
            e = ev(r0d, data(ns_def, mkblk(0.1)))
            return e["status"] == "FAIL" and "nothing observes it" in why(e) and "epo_con_temp" in why(e)
        finally:
            GC.SCORER_EPO_CON_KEYS = keep
    check(bad, "G_a_built_flag_nothing_observes_is_refused_epo_con_temp", built_without_observer)

    # ---------------------------------------------------------------- judge_g13 on the REAL temp_reach, each raw field with a negative twin
    g13 = good["g13"]
    nsj = argparse.Namespace(epo_con="g5", epo_con_eps=0.0, epo_con_lr=4.0, epo_con_max=10.0, epo_con_lam0=0.0, epo_temp=0.1, epo_con_temp=0.02)
    check(bad, "G_judge_g13_accepts_a_real_G13_with_temp_reach_at_0.02", lambda: GC.judge_g13(g13, nsj) == [] and g13["temp_reach"]["moved"] is True)

    def twin(edit, frag, ns_=nsj):
        d = copy.deepcopy(g13)
        edit(d)
        return any(frag in r for r in GC.judge_g13(d, ns_))
    twins = {
        "absent": (lambda d: d.pop("temp_reach"), "temp_reach) is absent"),
        "not_a_dict": (lambda d: d.update(temp_reach=None), "temp_reach) is absent"),
        "wrong_T": (lambda d: d["temp_reach"].update(temp=0.1, temp2=0.2), "not the argv's effective constraint temperature"),
        "T2_not_double": (lambda d: d["temp_reach"].update(temp2=0.05), "not exactly twice"),
        "not_moved_equal_reads": (lambda d: d["temp_reach"].update(per_module_T2=dict(d["temp_reach"]["per_module_on"])), "did not move the L_con gradient"),
        "moved_lie": (lambda d: d["temp_reach"].update(moved=False), "does not follow from its own raw reads"),
        "T2_dead_module": (lambda d: d["temp_reach"]["per_module_T2"].update(score_head=0.0), "doubled temperature is not finite and non-zero"),
        "T2_missing_module": (lambda d: d["temp_reach"]["per_module_T2"].pop("score_dec"), "doubled temperature is not finite and non-zero"),
        "T2_nan": (lambda d: d["temp_reach"]["per_module_T2"].update(score_dec=float("nan")), "doubled temperature is not finite and non-zero"),
        "A_touched_at_T2": (lambda d: d["temp_reach"]["A_per_module_T2"].update(score_head=1e-9), "arm A received a gradient from L_con at the doubled temperature"),
        "reference_read_swapped": (lambda d: d["temp_reach"]["per_module_on"].update(score_head=d["temp_reach"]["per_module_on"]["score_head"] * 1.5), "not the 'on' read"),
    }
    for nm, (edit, frag) in twins.items():
        check(bad, f"G_judge_g13_twin_temp_reach_{nm}", lambda edit=edit, frag=frag: twin(edit, frag))
    check(bad, "G_judge_g13_twin_the_argv_names_another_T_than_the_probe_ran_at", lambda: twin(lambda d: None, "not the argv's effective constraint temperature", argparse.Namespace(**{**vars(nsj), "epo_con_temp": None})))

    # ---------------------------------------------------------------- the consumer (G-PREFLIGHT) re-judges it
    def consume(argv, prof, rows):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "preflight.jsonl")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("\n".join(json.dumps(r) for r in rows) + "\n")
            return L.consume_scorer_preflight(p, argv, prof)
    pre = good["pre_rows"]

    def without_reach(rows):
        out = copy.deepcopy(rows)
        for r in out:
            if r.get("event") == "preflight":
                r["gates"]["G13_epo_con_live_and_identity"].pop("temp_reach")
        return out
    check(bad, "G_consumer_PASS_a_preflight_minted_at_0.02_for_an_argv_at_0.02", lambda: consume(r0p, Pp | {"declared_arm": "R0p"}, pre)["status"] == "PASS")
    check(bad, "G_consumer_FAIL_the_same_evidence_for_the_inherited_0.1_argv", lambda: (lambda v: v["status"] == "FAIL" and any("not the argv's effective constraint temperature" in r for r in v["reasons"]))(consume(r0d, pdr, pre)))
    check(bad, "G_consumer_FAIL_a_preflight_without_temp_reach", lambda: (lambda v: v["status"] == "FAIL" and any("temp_reach) is absent" in r for r in v["reasons"]))(consume(r0p, Pp | {"declared_arm": "R0p"}, without_reach(pre))))
    return bad


def build_good(R: dict) -> dict:
    r = R["t002"]
    assert r["pre"] == 0 and r["pass_json"] is not None, f"the unmutated trainer's constrained preflight at --epo-con-temp 0.02 did not pass: {r.get('out_pre', '')[-300:]}"
    return {"g13": r["pass_json"]["gates"]["G13_epo_con_live_and_identity"], "pre_rows": r["rows"]["preflight.jsonl"]}


# ================================================================================================ mutations: each edits SOURCE and must turn the named checks RED
TL_CALL = "ct = epo_con_term(sxB, sxA, tg, sel_score, dual.temp, a.epo_con_eps, dual.lam, con_last)"
RESOLVE = "temp=(a.epo_temp if a.epo_con_temp is None else a.epo_con_temp))"
MUTATIONS = [
    # (name, part, [(path, old, new)], expected RED checks)
    ("M1 (d) the OLD T hard-coded at the call: train_losses passes a.epo_temp again", "U", [(SF_PATH, TL_CALL, TL_CALL.replace("dual.temp", "a.epo_temp"))],
     ["U_a_the_constraint_s_gradient_on_the_scorer_params_DIFFERS_between_0.1_and_0.02_and_arm_A_does_not", "R_hop_4_train_losses_passes_dual_temp_as_the_T_of_epo_con_term_and_not_a_epo_temp"]),
    ("M2 (d) the OLD T hard-coded INSIDE the term (policy of B and of A at a literal 0.1)", "U", [(SF_PATH, "piB = epo_policy(sxB.float(), rule_fn, T).exp()", "piB = epo_policy(sxB.float(), rule_fn, 0.1).exp()"),
                                                                                                   (SF_PATH, "piA = epo_policy(sxA_ref.float(), rule_fn, T).exp()", "piA = epo_policy(sxA_ref.float(), rule_fn, 0.1).exp()")],
     ["U_a_the_constraint_s_gradient_on_the_scorer_params_DIFFERS_between_0.1_and_0.02_and_arm_A_does_not", "U_c_T_1e-06_the_constraint_quantity_IS_the_hard_pick_s_TTC_cA_cB_g_and_the_policy_exactly",
      "R_hop_5_epo_con_term_runs_BOTH_policies_at_its_T_parameter_with_no_literal_0.1_in_it"]),
    ("M3 the flag parsed but never reaches the dual (main passes a.epo_temp)", "U", [(SF_PATH, RESOLVE, "temp=a.epo_temp)")],
     ["U_a_the_constraint_s_gradient_on_the_scorer_params_DIFFERS_between_0.1_and_0.02_and_arm_A_does_not", "R_hop_2_main_builds_the_dual_with_temp_from_the_flag_and_falls_back_to_epo_temp_only_when_None"]),
    ("M4 the default flipped from None to 0.1 (no inheritance)", "U", [(SF_PATH, "EPO_CON_TEMP_DEFAULT = None ", "EPO_CON_TEMP_DEFAULT = 0.1 ")],
     ["U_b_the_default_INHERITS_epo_temp_for_any_value_of_it_it_is_not_a_constant_0.1", "P_default_is_None_and_it_is_a_named_constant"]),
    ("M5 the resolution inverted (the flag loses to --epo-temp)", "U", [(SF_PATH, RESOLVE, "temp=(a.epo_con_temp if a.epo_con_temp is None else a.epo_temp))")],
     ["U_a_the_constraint_s_gradient_on_the_scorer_params_DIFFERS_between_0.1_and_0.02_and_arm_A_does_not", "R_hop_2_main_builds_the_dual_with_temp_from_the_flag_and_falls_back_to_epo_temp_only_when_None"]),
    ("M6 epo_con_temp dropped from EPO_CON_DESTS (named-while-off no longer refused)", "U", [(SF_PATH, '"epo_con_lam0", "epo_con_temp")', '"epo_con_lam0")')],
     ["U_refusal_temp_named_while_off_or_without_epo_is_refused_because_the_term_is_OFF", "P_default_is_None_and_it_is_a_named_constant"]),
    ("M7 the refusal of a non-positive temperature removed", "U", [(SF_PATH, "if a.epo_con_temp is not None and not (math.isfinite(a.epo_con_temp) and a.epo_con_temp > 0.0):", "if False:")],
     ["U_refusal_a_non_positive_or_non_finite_temp_is_refused"]),
    ("M8 the T2 probe does not double (con_probe)", "E", [(SF_PATH, "dual.temp = 2.0 * keep_temp", "dual.temp = keep_temp")], ["E_t002_the_real_main_runs_preflight_PASS_then_trains"]),
    ("M9 (d, preflight) the OLD T hard-coded at the call: the PREFLIGHT must FAIL (G13 temp_reach)", "E", [(SF_PATH, TL_CALL, TL_CALL.replace("dual.temp", "a.epo_temp"))], ["E_t002_the_real_main_runs_preflight_PASS_then_trains"]),
    ("G1 the G-DVB row for epo_con.temp removed (it falls to the generic branch)", "G", [(GC_PATH, '            elif k == "temp":', '            elif k == "temp_unused":')],
     ["G_dvb_PASS_default_named_0.02_and_explicit_0.1_each_with_the_block_that_matches", "G_dvb_the_epo_con_temp_row_is_observed_and_carries_the_effective_declared_value"]),
    ("G2 the entry dropped from DVB_KINDS_SCORER", "G", [(GC_PATH, '    "epo_con_temp": "built",', "")], ["G_dvb_registry_classifies_epo_con_temp_as_built"]),
    ("G3 judge_g13 does not require temp_reach", "G", [(GC_PATH, 'r.append("G13: the temperature-reach probe (temp_reach) is absent: nothing proves the loss reads the T the dual holds")', "pass")],
     ["G_judge_g13_twin_temp_reach_absent", "G_consumer_FAIL_a_preflight_without_temp_reach"]),
    ("G4 judge_g13 does not require the gradient to move when T doubles", "G", [(GC_PATH, "elif set(t_on) == mods and all(t2_on[m] == t_on[m] for m in mods):", "elif False:")],
     ["G_judge_g13_twin_temp_reach_not_moved_equal_reads"]),
    ("G5 judge_g13 does not bind the probed T to the argv's effective T", "G", [(GC_PATH, "if not (fin(t) and fin(want_t) and _close(t, want_t)):", "if False:")],
     ["G_judge_g13_twin_temp_reach_wrong_T", "G_consumer_FAIL_the_same_evidence_for_the_inherited_0.1_argv"]),
    ("G6 the registered default constraint T removed (older arms accept a changed T)", "G", [(LG_PATH, 'reg = {"epo_con_temp": reg["epo_temp"], **reg}', "pass")],
     ["G_a_registered_arm_refuses_a_changed_T_and_says_what_it_resolves_to"]),
    ("G7 the gate compares the RAW namespace value, not the effective one", "G", [(LG_PATH, "if not C._close(C.scorer_effective(ns, dest), want):", "if not C._close(getattr(ns, dest), want):")],
     ["G_the_registered_arms_are_unchanged_R0d_R1d_R1o_clean_without_the_flag"]),
    ("G8 the aux profile no longer forbids --epo-con-temp", "G", [(LG_PATH, '"--epo-con-lam0", "--epo-con-temp",', '"--epo-con-lam0",')], ["G_the_aux_and_sft4r_profiles_forbid_the_flag"]),
    ("G9 the regression arm con_temp_ignored is inert", "G", [(GC_PATH, 'if arm_is("con_temp_ignored", arm) and', 'if arm_is("con_temp_ignored_off", arm) and')],
     ["G_dvb_REGRESSION_ARM_con_temp_ignored_FAILS_on_the_named_argv", "G_dvb_REGRESSION_ARM_con_temp_ignored_FAILS_on_the_unnamed_argv"]),
    ("G10 a future arm may omit a registered T (arm_must_name ignored)", "G", [(LG_PATH, 'for f in prof.get("arm_must_name", {}).get(declared, ()):', 'for f in ():')],
     ["G_future_arm_fixture_R0p_with_the_flag_OMITTED_is_refused_NAMED_and_registered_value_and_the_parser_DEFAULT"]),
]


def apply_edits(srcs: dict, edits, name: str) -> dict:
    out = dict(srcs)
    for path, old, new in edits:
        assert out[path].count(old) == 1, f"mutation {name}: the snippet occurs {out[path].count(old)} times (must be exactly 1): {old[:70]!r}"
        out[path] = out[path].replace(old, new)
    return out


def main() -> int:
    srcs = {p: open(p, encoding="utf-8").read() for p in (SF_PATH, LG_PATH, GC_PATH)}
    assert S1B.git_blob(S1B.pickcon_reverse_file(srcs[SF_PATH]).encode("utf-8")) == S1B.E2D4422_TRAINER_BLOB, "reversing PICKCON's edits does not give the 0a5958a trainer: regenerate PICKCON_FILE_REVERSALS (gen_pickcon_reversals.py)"
    R = build_runs(srcs[SF_PATH])
    good = build_good(R)
    real = suite_u(srcs[SF_PATH], "real", trace=True) + suite_e(R) + suite_g(srcs[LG_PATH], srcs[GC_PATH], "real", good)
    assert not real, f"the unmutated sources FAIL their own checks: {real}"
    n, problems = 1, []
    print(f"{'mutation':100s} {'part':4s} caught-by (failing checks)")
    for i, (name, part, edits, expect) in enumerate(MUTATIONS):
        m = apply_edits(srcs, edits, name)
        try:
            if part == "U":
                _ast.parse(m[SF_PATH])
                failing = suite_u(m[SF_PATH], f"m{i}")
            elif part == "E":
                _ast.parse(m[SF_PATH])
                failing = suite_e(build_runs(m[SF_PATH], keys=("t002",)))
            else:
                failing = suite_g(m[LG_PATH], m[GC_PATH], f"m{i}", good)
        except (Exception, SystemExit) as e:                            # noqa: BLE001 -- the mutated source breaks the suite's own setup: RED, and said so
            failing = [f"suite setup raised {type(e).__name__}: {str(e)[:80]}"]
        if not failing:
            problems.append(f"mutation {name} stayed GREEN: the suite cannot see it")
            continue
        for c in expect:
            if c not in failing and not any(f.startswith("suite setup raised") for f in failing):
                problems.append(f"mutation {name}: the check {c} must go RED and stayed green (RED: {failing[:8]})")
        behavioural = [f for f in failing if not f.startswith("P_")]                    # the pins fire on ANY edit of the patched text; the table shows what else saw it
        print(f"{name[:100]:100s} {part:4s} {', '.join((behavioural or failing)[:3])}" + ("" if behavioural else "   [PIN ONLY]"), flush=True)
        n += 1
    if problems:
        print("\n".join("PROBLEM " + p_ for p_ in problems))
        print(f"ZZTEST_FAIL {len(problems)} problem(s)")
        return 1
    print(f"ZZTEST_PASS {n}  (1 unmutated suite + {len(MUTATIONS)} RED mutations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
