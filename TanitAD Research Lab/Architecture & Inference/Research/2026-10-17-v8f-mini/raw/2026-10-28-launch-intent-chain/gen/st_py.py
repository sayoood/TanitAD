#!/usr/bin/env python3
"""Self-test, python layer: the one-lever gate asserter (A; RA = the same cases for A_rep,s against REP_s), the run-record / K-SHARE / K-MASK checks (B, K; RB / RK for A_rep), the one-lever command (OL), the REP licence (C), the predecessor verdict (P).

Every CASE is a world with a registered expectation (PASS, or a REFUSAL whose message contains a given text). The positive cases are tested first against the ORIGINAL modules (tests). Every MUTANT edits ONE line of a module's text; it is KILLED iff some probe case
no longer gives its expected outcome (the worlds are always BUILT with the ORIGINAL asserter, so a mutant cannot make its own world consistent).
"""
from __future__ import annotations

import copy
import json
import re
import shutil
import sys
from pathlib import Path

from st_common import *                              # noqa: F401,F403
import st_common as S

G0, K0 = S.mods()
LINES = {}


# ---------------------------------------------------------------------------------------------------------------------------------------- argv mutators
def with_val(flag, val):
    def f(a):
        a = list(a)
        a[a.index(flag) + 1] = val
        return a
    return f


def add(*toks):
    return lambda a: list(a) + list(toks)


def drop_last(n):
    return lambda a: list(a)[:-n]


def move_flag_before_wacc(a):
    a = list(a)
    i = a.index("--w-acc")
    return a[:i] + ["--launch-intent-w", "1.58926"] + a[i:-2]


def set_out(path):
    return with_val("--out", path)


def sc_set(path, v):
    return S.edit(path, v)


def sc_del(path):
    return S.edit(path, delete=True)


def all3(f):
    """A line mutation applied to the planned training argv (the gate line and the pin follow it in the world)."""
    return dict(line_mutate=f)


A_CASES = [
    ("A00_pass_s0", dict(seed=0), None), ("A00_pass_s1", dict(seed=1), None),
    ("A00_out_free", dict(seed=0, line_mutate=set_out(r"D:\other\place\M2b_s0_x")), None),
    ("A_li_absent", dict(mutate=sc_del("levers.launch_intent_w")), "built levers.launch_intent_w"),
    ("A_w_zero", dict(mutate=sc_set("levers.launch_intent_w", 0.0)), "built levers.launch_intent_w"),
    ("A_w_B", dict(mutate=sc_set("levers.launch_intent_w", 0.726175)), "built levers.launch_intent_w"),
    ("A_w_bool", dict(mutate=sc_set("levers.launch_intent_w", True)), "built levers.launch_intent_w"),
    ("A_mask_v0", dict(mutate=sc_set("levers.launch_intent.v0_max", 7.0)), "registered mask definition"),
    ("A_mask_knots", dict(mutate=sc_set("levers.launch_intent.knots", [1, 30])), "registered mask definition"),
    ("A_probe_absent", dict(mutate=sc_del("launch_intent_probe")), "launch_intent_probe"),
    ("A_probe_fail", dict(mutate=sc_set("launch_intent_probe.pass", False)), "launch_intent_probe"),
    ("A_probe_off", dict(mutate=sc_set("launch_intent_probe.launch_intent", "off")), "launch_intent_probe"),
    ("A_term0", dict(mutate=sc_set("launch_intent_probe.term_at_zero_weight", 0.1)), "computed at weight 0.0"),
    ("A_contrib_zero", dict(mutate=sc_set("launch_intent_probe.contribution", 0.0)), "does not enter the loss at the declared weight"),
    ("A_contrib_10pct", dict(mutate=sc_set("launch_intent_probe.contribution", 1.3243833 * 1.1)), "does not enter the loss at the declared weight"),
    ("A_head0", dict(mutate=sc_set("launch_intent_probe.head_o_grad_diff", 0.0)), "O's head receives gradient"),
    ("A_ctrl_allfalse_grad", dict(mutate=sc_set("launch_intent_probe.control_allfalse_grad_diff", 1e-9)), "control control_allfalse_grad_diff"),
    ("A_ctrl_prior", dict(mutate=sc_set("launch_intent_probe.control_prior_grad_diff", 1e-9)), "control control_prior_grad_diff"),
    ("A_ctrl_repeat", dict(mutate=sc_set("launch_intent_probe.control_repeat_grad_diff", 1e-9)), "control control_repeat_grad_diff"),
    ("A_allfalse_term", dict(mutate=sc_set("launch_intent_probe.control_allfalse_term", 0.5)), "ALL-FALSE mask gives term"),
    ("A_wm_ctrl", dict(mutate=sc_set("launch_intent_probe.control_wm_grad_diff", 0.01)), "world-model auxiliary"),
    ("A_natural0", dict(mutate=sc_set("launch_intent_probe.natural_contribution", 0.0)), "the natural mask names"),
    ("A_ego_p", dict(mutate=sc_set("levers.ego_perturb.ego_perturb_p", 0.3)), "built ego_perturb_p"),
    ("A_compose_true", dict(mutate=sc_set("levers.ego_perturb_with_dagger", True)), "built ego_perturb_with_dagger"),
    ("A_dagger_probe_missing", dict(mutate=sc_del("dagger_probe")), "dagger_probe"),
    ("A_acc_probe_missing", dict(mutate=sc_del("acc_probe")), "acc_probe"),
    ("A_init_sha", dict(over=dict(banked_init_sha="0" * 64)), "does not start from M2b"),
    ("A_bank_sha", dict(over=dict(bank_sha="1" * 64)), "bank sha256"),
    ("A_extra_wtraj", all3(add("--w-traj", "2.0")), "sets --w-traj"),
    ("A_extra_unknown", all3(add("--foo", "1")), "ONE-LEVER"),
    ("A_law_flag", all3(add("--ego-perturb-vscale", "0.5", "1.0")), "names --ego-perturb-vscale"),
    ("A_compose_flag", all3(add("--ego-perturb-with-dagger")), "names --ego-perturb-with-dagger"),
    ("A_W_B", all3(with_val("--launch-intent-w", "0.726175")), "arm B"),
    ("A_W_HI", all3(with_val("--launch-intent-w", "4.17852")), "arm HI"),
    ("A_W_respelled", all3(with_val("--launch-intent-w", "1.589260")), "does not carry --launch-intent-w 1.58926 exactly once"),
    ("A_W_off_by_one", all3(with_val("--launch-intent-w", "1.58927")), "does not carry --launch-intent-w 1.58926 exactly once"),
    ("A_flag_missing", all3(drop_last(2)), "does not carry --launch-intent-w"),
    ("A_flag_twice", all3(add("--launch-intent-w", "1.58926")), "exactly once"),
    ("A_flag_moved", all3(move_flag_before_wacc), "ONE-LEVER"),
    ("A_steps", all3(with_val("--steps", "3001")), "does not carry --steps 3000"),
    ("A_wacc_line", all3(with_val("--w-acc", "49.6970")), "does not carry --w-acc"),
    ("A_dagger_p_line", all3(with_val("--dagger-p", "0.2")), "does not carry --dagger-p 0.1"),
    ("A_ego_p_line", all3(with_val("--ego-perturb-p", "0.3")), "does not carry --ego-perturb-p 0"),
    ("A_pin_differs", dict(pin_mutate=drop_last(2)), "THE PIN IS NOT"),
    ("A_gate_differs", dict(gate_mutate=lambda g: g[:-3] + ["--selfcheck-only"]), "gate argv is not the training argv"),
    ("A_argv_sha", dict(sha_override="0" * 16), "selfcheck's argv_sha differs"),
    ("A_ref_missing", dict(over=dict(ref_argv_json="")), "required"),
    # ---- the PAIR rule: A_s is checked against DAG-OACC_s (trainer seed s), A_rep,s against REP_s (trainer seed 10 + s); a swapped reference or a swapped trainer seed is refused
    ("A_ref_is_rep", dict(seed=0, over=dict(ref_argv_json=str(S.REPDIR / "M2b_s0_dagoaccorcrep" / "argv.json"))), "the reference argv must carry"),
    ("A_seed_as_rep", dict(seed=0, line_mutate=with_val("--seed", "10")), "does not carry --seed 0 exactly once"),
    ("A_dagger_seed_trainer", dict(seed=0, line_mutate=with_val("--dagger-seed", "10")), "does not carry --dagger-seed 0"),
]
A_CASES_REP = [("R" + cid, {**kw, "arm": "launchintentrep"}, ex) for cid, kw, ex in A_CASES if not cid.startswith(("A_ref_is_rep", "A_seed_as_rep", "A_dagger_seed_trainer"))] + [
    ("RA_ref_is_oacc", dict(seed=0, arm="launchintentrep", over=dict(ref_argv_json=str(S.OACC / "M2b_s0_dagoacc" / "argv.json"))), "the reference argv must carry"),
    ("RA_ref_is_oacc_s1", dict(seed=1, arm="launchintentrep", over=dict(ref_argv_json=str(S.OACC / "M2b_s1_dagoacc" / "argv.json"))), "the reference argv must carry"),
    ("RA_seed_as_A", dict(seed=0, arm="launchintentrep", line_mutate=with_val("--seed", "0")), "does not carry --seed 10 exactly once"),
    ("RA_seed_other_pair", dict(seed=0, arm="launchintentrep", line_mutate=with_val("--seed", "11")), "does not carry --seed 10 exactly once"),
    ("RA_dagger_seed_trainer", dict(seed=0, arm="launchintentrep", line_mutate=with_val("--dagger-seed", "10")), "does not carry --dagger-seed 0"),
]
A_ALL = A_CASES + A_CASES_REP
LIMIT = {}


def eval_a(Gm, case, tag):
    cid, kw, expect = case
    kw = dict(kw)
    seed = kw.pop("seed", 0)
    root = S.fresh(f"A/{tag}_{cid}")
    d = S.gate_world(G0, root, seed, **kw)      # kw may carry arm=launchintentrep
    try:
        bad = S.run_gate(Gm, d)
    except Exception as e:                                              # noqa: BLE001  a crashing checker is a refusal of the wrong kind: it must not satisfy an expectation
        return False, [f"EXC {type(e).__name__}: {e}"]
    ok = (bad == []) if expect is None else any(expect in b for b in bad)
    return ok, bad


# ---------------------------------------------------------------------------------------------------------------------------------------- run record (B) / K-SHARE (K)
def n_for_share(s, e=191203):
    a = 1.0 + LI_W
    return int(round(s * e / (a * (1.0 - s) + s)))


def mk_counts(s, e=191203):
    n = n_for_share(s, e)
    return n, e, K0.share_of(n, e, LI_W)


B_CASES = [
    ("B00_pass_s0", dict(seed=0), None), ("B00_pass_s1", dict(seed=1), None),
    ("B_stale_sm", dict(mut_sm=sc_del("levers.launch_intent_w")), "summary.json: levers.launch_intent_w is ABSENT"),
    ("B_stale_sc", dict(mut_sc=sc_del("levers.launch_intent_w")), "selfcheck.json: levers.launch_intent_w is ABSENT"),
    ("B_w_sm", dict(mut_sm=sc_set("levers.launch_intent_w", 1.58927)), "summary.json: levers.launch_intent_w 1.58927"),
    ("B_w_sc", dict(mut_sc=sc_set("levers.launch_intent_w", 0.726175)), "selfcheck.json: levers.launch_intent_w 0.726175"),
    ("B_w_int", dict(mut_sm=sc_set("levers.launch_intent_w", 2)), "summary.json: levers.launch_intent_w 2"),
    ("B_missing_n", dict(mut_sm=sc_del("launch_n")), "lacks"), ("B_missing_share", dict(mut_sm=sc_del("launch_weight_share")), "lacks"),
    ("B_missing_rows", dict(mut_sm=sc_del("launch_rows")), "lacks"), ("B_missing_top_w", dict(mut_sm=sc_del("launch_intent_w")), "lacks"),
    ("B_argv_json", dict(mut_aj=lambda aj: aj.__setitem__("argv", aj["argv"][:-2])), "argv.json is not the planned"),
    ("B_pin", dict(mut_pin=add("--x")), "pinned training argv"),
    ("B_done", dict(mut_sm=sc_set("done", False)), "done is not true"), ("B_steps", dict(mut_sm=sc_set("steps", 2999)), "summary.json steps"),
    ("B_seed", dict(mut_sm=sc_set("seed", 5)), "arm / seed"), ("B_init", dict(over=dict(banked_init_sha="0" * 64)), "K-INIT"),
    ("B_ego_p", dict(mut_sm=sc_set("levers.ego_perturb.ego_perturb_p", 0.3)), "built ego_perturb ego_perturb_p"),
    ("B_compose", dict(mut_sm=sc_set("levers.ego_perturb_with_dagger", True)), "ego_perturb_with_dagger"),
    ("B_dagger_p", dict(mut_sm=sc_set("levers.dagger_p", 0.2)), "built dagger_p"), ("B_bank", dict(over=dict(bank_sha="1" * 64)), "dagger_bank_sha256"),
    ("B_wacc", dict(mut_sm=sc_set("levers.w_acc", 50.0)), "built w_acc"), ("B_sha16", dict(mut_sm=sc_set("argv_sha", "0" * 16)), "summary.json argv_sha"),
    ("B_sc_fail", dict(mut_sc=sc_set("declared_vs_built", "FAIL")), "not a PASS"), ("B_mask", dict(mut_sm=sc_set("levers.launch_intent.v0_max", 7.0)), "levers.launch_intent.v0_max"),
    ("B_seed_as_rep", dict(mut_sm=sc_set("seed", 10)), "arm / seed"),
]
B_CASES_REP = [("R" + cid, {**kw, "arm": "launchintentrep"}, ex) for cid, kw, ex in B_CASES if cid not in ("B_seed", "B_seed_as_rep")] + [
    ("RB_seed_as_A", dict(arm="launchintentrep", mut_sm=sc_set("seed", 0)), "arm / seed"), ("RB_seed_other", dict(arm="launchintentrep", mut_sm=sc_set("seed", 11)), "arm / seed"),
    ("RB_planned_is_A", dict(arm="launchintentrep", mut_planned=with_val("--seed", "0")), "does not carry --seed 10 exactly once"),
    ("B_planned_is_rep", dict(arm="launchintent", mut_planned=with_val("--seed", "10")), "does not carry --seed 0 exactly once")]
B_ALL = B_CASES + B_CASES_REP


def eval_b(Km, case, tag):
    cid, kw, expect = case
    kw = dict(kw)
    seed = kw.pop("seed", 0)
    root = S.fresh(f"B/{tag}_{cid}")
    d = S.record_world(G0, K0, root, seed, **kw)      # kw may carry arm=launchintentrep
    try:
        bad = S.quiet(Km.run_record, S.NS(d))
    except Exception as e:                                              # noqa: BLE001
        return False, [f"EXC {type(e).__name__}: {e}"]
    return ((bad == []) if expect is None else any(expect in b for b in bad)), bad


KS_CASES = [
    ("K00_pass", dict(), None),
    ("K_share_in_lo", dict(counts=mk_counts(0.3136)), None), ("K_share_in_hi", dict(counts=mk_counts(0.3530)), None),
    ("K_share_low", dict(counts=mk_counts(0.3128)), "K-SHARE"), ("K_share_high", dict(counts=mk_counts(0.3538)), "K-SHARE"),
    ("K_rows_ne", dict(mut=lambda sm: sm.__setitem__("launch_rows", sm["launch_eligible"] + 1)), "launch_rows"),
    ("K_share_rec", dict(mut=lambda sm: sm.__setitem__("launch_weight_share", 0.30)), "own launch_weight_share"),
    ("K_n0", dict(counts=(0, 191203, 0.0)), "saw no launch row"),
    ("K_window", dict(over=dict(window_sha="0" * 64)), "K-MASK: train_window_list_sha256"),
    ("K_missing_n", dict(mut=sc_del("launch_n")), "lacks"),
    ("K_w_wrong", dict(mut=lambda sm: sm.__setitem__("launch_intent_w", 1.58927)), "launch_intent_w summary"),
]


KS_CASES_REP = [("R" + cid, {**kw, "arm": "launchintentrep"}, ex) for cid, kw, ex in KS_CASES]
KS_ALL = KS_CASES + KS_CASES_REP


def eval_ks(Km, case, tag):
    cid, kw, expect = case
    root = S.fresh(f"K/{tag}_{cid}")
    mut = kw.get("mut")
    counts = kw.get("counts")
    arm = kw.get("arm", "launchintent")

    def m(sm):
        if counts:
            S.sm_with_counts(K0, counts[0], counts[1], share=counts[2])(sm)
        if mut:
            mut(sm)
    d = S.record_world(G0, K0, root, 0, mut_sm=m, over=kw.get("over"), arm=arm)
    try:
        bad = Km.kshare_problems(S.load(d["summary"]), d["window_sha"])
    except Exception as e:                                              # noqa: BLE001
        return False, [f"EXC {type(e).__name__}: {e}"]
    return ((bad == []) if expect is None else any(expect in b for b in bad)), bad


# ---------------------------------------------------------------------------------------------------------------------------------------- the one-lever command (OL): a printed line against its pair control's banked argv
OL_CASES = [
    ("OL_A_pass_s0", "launchintent", 0, None, None, 0), ("OL_A_pass_s1", "launchintent", 1, None, None, 0),
    ("OL_R_pass_s0", "launchintentrep", 0, None, None, 0), ("OL_R_pass_s1", "launchintentrep", 1, None, None, 0),
    ("OL_A_vs_rep_ref", "launchintent", 0, None, "rep", 1), ("OL_R_vs_oacc_ref", "launchintentrep", 0, None, "oacc", 1),
    ("OL_R_line_checked_as_A", "launchintentrep", 0, None, "asA", 1),
    ("OL_A_extra", "launchintent", 0, add("--w-traj", "2.0"), None, 1), ("OL_R_extra", "launchintentrep", 0, add("--w-traj", "2.0"), None, 1),
    ("OL_R_W_B", "launchintentrep", 1, with_val("--launch-intent-w", "0.726175"), None, 1),
]


def eval_ol(Km, case, tag):
    cid, arm, seed, mutf, refkind, want_rc = case
    root = S.fresh(f"OL/{tag}_{cid}")
    train = S.arm_argv(G0, seed, arm=arm)
    if mutf:
        train = mutf(list(train))
    S.put(root / "line.txt", S.line(train))
    ref = {None: S.ctrl_dir(arm, seed), "rep": S.REPDIR / f"M2b_s{seed}_dagoaccorcrep", "oacc": S.OACC / f"M2b_s{seed}_dagoacc", "asA": S.ctrl_dir(arm, seed)}[refkind] / "argv.json"
    use_arm = "launchintent" if refkind == "asA" else arm
    try:
        rc = S.quiet(Km.one_lever, S.NS(dict(ref_argv_json=str(ref), line=str(root / "line.txt"), seed=str(seed), arm=use_arm)))
    except Exception as e:                                              # noqa: BLE001
        return False, f"EXC {type(e).__name__}: {e}"
    return rc == want_rc, f"rc={rc}"


# ---------------------------------------------------------------------------------------------------------------------------------------- REP licence (C)
def rm(files_key, path, v=None, delete=False):
    ks = path.split(".")

    def f(files):
        cur = files[files_key]
        for k in ks[:-1]:
            cur = cur[k]
        if delete:
            cur.pop(ks[-1], None)
        else:
            cur[ks[-1]] = v
    return f


def set_argv_tok(flag, val):
    def f(files):
        a = files["argv"]["argv"]
        a[a.index(flag) + 1] = val
    return f


REP_LOG_PASS = "\n".join([f"DAG-ORC (DAG-OACC base) run record: PASS | dagoaccorcrep s{s} | trainer seed {10 + s} | x" for s in (0, 1)]) + "\n"
C_CASES = [
    ("C00_pass_s0", 0, None, "", None), ("C00_pass_s1", 1, None, "", None), ("C00_pass_log", 0, None, REP_LOG_PASS, None),
    ("C_swap", 0, rm("summary", "dagger_n", 30000), "", "swap share"), ("C_ego", 0, rm("summary", "ego_perturb_n", 5), "", "ego_perturb_n"),
    ("C_compose", 0, rm("summary", "levers.ego_perturb_with_dagger", True), "", "composition flag"),
    ("C_stream_absent", 0, rm("summary", "levers.ego_perturb_stream", delete=True), "", "ego_perturb_stream"),
    ("C_stream_one", 0, rm("selfcheck", "levers.ego_perturb_stream", 1), "", "ego_perturb_stream"),
    ("C_stream_bool", 0, rm("summary", "levers.ego_perturb_stream", False), "", "ego_perturb_stream"),
    ("C_launch", 0, rm("summary", "levers.launch_intent_w", 1.58926), "", "carries a launch lever"),
    ("C_argv", 0, set_argv_tok("--steps", "3001"), "", "not DAG-OACC's banked argv"),
    ("C_tok_init", 0, rm("token", "init_ckpt_sha256", "0" * 64), "", "REP token init_ckpt_sha256"),
    ("C_tok_bank", 0, rm("token", "bank_sha256", "1" * 64), "", "REP token bank_sha256"),
    ("C_tok_wacc", 0, rm("token", "w_acc", 49.0), "", "REP token w_acc"),
    ("C_tok_excl", 0, rm("token", "excluded_sha256", "2" * 64), "", "REP token excluded_sha256"),
    ("C_tok_data", 0, rm("token", "binding.data_sha256", "3" * 64), "", "binding.data_sha256"),
    ("C_tok_verdict", 0, rm("token", "verdict", "FAIL"), "", "not a PASS"),
    ("C_window", 0, rm("summary", "train_window_list_sha256", "4" * 64), "", "REP summary train_window_list_sha256"),
    ("C_done", 0, rm("summary", "done", False), "", "did not finish"), ("C_seed", 0, rm("summary", "seed", 5), "", "arm / seed"),
    ("C_log_no_pass", 0, None, "10-28 05:19 something else\n", "run record: PASS"), ("C_log_missing", 0, None, "MISSING", "chain log"),
    ("C_log_fail", 0, None, REP_LOG_PASS + "DAG-ORC (DAG-OACC base) K-REACH: FAIL | dagoaccorcrep | swap x\n", "carries a FAIL for REP"),
]


def eval_c(Km, case, tag):
    cid, seed, mut, log, expect = case
    root = S.fresh(f"C/{tag}_{cid}")
    w = S.rep_world(root, seed, mut)
    logp = ""
    if log == "MISSING":
        logp = str(root / "no_such_chain.out")
    elif log:
        logp = str(root / "r2.out")
        Path(logp).write_bytes(log.encode())
    try:
        bad = S.quiet(Km.rep_licence, S.NS(S.rep_args(w, seed, log=logp)))
    except Exception as e:                                              # noqa: BLE001
        return False, [f"EXC {type(e).__name__}: {e}"]
    return ((bad == []) if expect is None else any(expect in b for b in bad)), bad


# ---------------------------------------------------------------------------------------------------------------------------------------- predecessor verdict (P)
def oacc_done_line():
    p = S.C.R2 / "fixtures" / "oacc_r2_chain_tail.out"
    if not p.exists():
        return "ZZV8FMINI-DONE 10-27 21:51 tip aeb544683b1c6b70e29f241fe7133d4290a1b4a5: PREREG_V8F_DAG_OACC (r2): w_acc 49.6969 (C-O3 PASS), both gates passed"
    return [x for x in p.read_text(encoding="utf-8", errors="replace").split("\n") if x.startswith("ZZV8FMINI-DONE")][-1]


P_CASES = [
    ("P_go", lambda: S.r2_log(), False, "GO"), ("P_busy", lambda: S.r2_log(), True, "BUSY"), ("P_wait", lambda: S.r2_log(done=False), False, "WAIT"), ("P_empty", lambda: "", False, "WAIT"),
    ("P_stop", lambda: S.r2_log(done=False, stop="ZZV8FMINI-STOP K-REACH something"), False, "HALT"),
    ("P_stop_then_done", lambda: S.r2_log(stop="ZZV8FMINI-STOP reached-control fail M2b_s0_dagoaccorc"), False, "HALT"),
    ("P_guard", lambda: S.r2_log(guard=True), False, "HALT"), ("P_floor", lambda: S.r2_log(floor=True), False, "HALT"),
    ("P_ctrl", lambda: S.r2_log(fail_line="ZZDAGOACCORC-CONTROL-FAIL M2b_s0_dagoaccorc: x"), False, "HALT"),
    ("P_after", lambda: S.r2_log(extra_after="10-28 09:00 something after the end\n"), False, "HALT"),
    ("P_two_done", lambda: S.r2_log(two_done=True), False, "HALT"),
    ("P_foreign_done", lambda: S.r2_log(done_line=oacc_done_line()), False, "HALT"),
    ("P_missing_pull", lambda: S.r2_log(drop_pull="M2b_s1_dagoaccorcrep"), False, "HALT"),
    ("P_missing_rr", lambda: S.r2_log(drop_rr="M2b_s0_dagoaccorc"), False, "HALT"), ("P_missing_kreach", lambda: S.r2_log(drop_kreach="dagoaccorcrep"), False, "HALT"),
    ("P_fail_line", lambda: S.r2_log(fail_line="DAG-ORC (DAG-OACC base) K-REACH: FAIL | dagoaccorc | x"), False, "HALT"),
    ("P_dated_echo", lambda: S.r2_log().replace("waiting for the r1 chain (x)", "waiting for ^ZZV8FMINI-STOP"), False, "GO"),
]


def eval_p(Km, case):
    cid, mk, busy, want = case
    try:
        v, why = Km.pred_verdict_li(mk(), busy)
    except Exception as e:                                              # noqa: BLE001
        return False, f"EXC {e}"
    return v == want, f"{v} {why}"


# ---------------------------------------------------------------------------------------------------------------------------------------- the mutant table
def gsub(old, new, n=1):
    return ("G", old, new, n)


def ksub(old, new, n=1):
    return ("K", old, new, n)


MUTANTS_PY = [
    # ---- the gate asserter (probes: A cases)
    ("G01 launch_problems never called", gsub("    bad += launch_problems(sc, LI_W)\n", ""), ["A_li_absent", "A_probe_absent", "A_ctrl_prior"]),
    ("G02 ego_off_problems never called", gsub("    bad += ego_off_problems(sc)\n", ""), ["A_ego_p", "A_compose_true"]),
    ("G03 swap_problems never called", gsub("    bad += swap_problems(sc, arm, SWAP_P, seed, a)\n", ""), ["A_dagger_probe_missing"]),
    ("G04 wacc_problems never called", gsub("        bad += wacc_problems(sc, WA)\n", "        pass\n"), ["A_acc_probe_missing"]),
    ("G05 the registered W is 1.58927", gsub('LI_W_STR = "1.58926"', 'LI_W_STR = "1.58927"'), ["A00_pass_s0"]),
    ("G06 B / HI weights no longer named", gsub('LI_W_REFUSED = {"0.726175": "arm B (share 1/4)", "4.17852": "arm HI (NEXT-LAUNCH only)"}', "LI_W_REFUSED = {}"), ["A_W_B", "A_W_HI"]),
    ("G07 the exact-argv comparison is skipped", gsub("                if drop(line) != exp_argv:", "                if False:"), ["A_extra_unknown", "A_flag_moved"]),
    ("G08 expected_from_ref adds no flag", gsub("    return r + [LI_FLAG, LI_W_STR]", "    return r"), ["A00_pass_s0"]),
    ("G09 expected_from_ref puts the flag first", gsub("    return r + [LI_FLAG, LI_W_STR]", "    return [LI_FLAG, LI_W_STR] + r"), ["A00_pass_s0"]),
    ("G10 --ref-argv-json is optional", gsub("    if not a.ref_argv_json:", "    if False:"), ["A_ref_missing"]),
    ("G11 the prior's control is not required", gsub(', "control_prior_grad_diff")      # each EXACTLY 0.0', ")      # each EXACTLY 0.0"), ["A_ctrl_prior"]),
    ("G12 the all-false term control is skipped", gsub('    if pr.get("control_allfalse_term") != 0.0:', "    if False:"), ["A_allfalse_term"]),
    ("G13 head_o's gradient may be 0", gsub('pr["head_o_grad_diff"] > 0.0', 'pr["head_o_grad_diff"] >= 0.0'), ["A_head0"]),
    ("G14 the contribution tolerance is 50 %", gsub("abs(c - want) <= 2e-3 * want", "abs(c - want) <= 0.5 * want"), ["A_contrib_10pct"]),
    ("G15 the mask's knots are 1-30", gsub('"knots": [1, 12]}', '"knots": [1, 30]}'), ["A00_pass_s0"]),
    ("G16 the mask definition is not compared", gsub("            if li.get(k) != w:", "            if False:"), ["A_mask_v0", "A_mask_knots"]),
    ("G17 a term at weight 0 is allowed", gsub('    if pr.get("term_at_zero_weight") is not None:', "    if False:"), ["A_term0"]),
    ("G18 the ego perturbation may be on", gsub('    if ep.get("ego_perturb_p") != 0.0:', "    if False:"), ["A_ego_p"]),
    ("G19 the composition flag may be built", gsub("    if wd not in (None, False):", "    if False:"), ["A_compose_true"]),
    ("G20 law / composition flags in the lines are not named", gsub("        for k in EGO_LAW + (COMPOSE,):", "        for k in ():"), ["A_law_flag", "A_compose_flag"]),
    ("G21 the pin may differ from the planned line", gsub("    if pin != tl:", "    if False:"), ["A_pin_differs"]),
    ("G22 the gate's argv_sha is not compared", gsub('    if sc.get("argv_sha") != hashlib.sha256(" ".join(gl).encode()).hexdigest()[:16]:\n        bad.append("the gate ran an argv other than the queue\'s printed gate line (the selfcheck\'s argv_sha differs)")\n    # ---- THE ONE-LEVER',
                                                    '    if False:\n        bad.append("x")\n    # ---- THE ONE-LEVER'), ["A_argv_sha"]),
    ("G23 the probe's pass flag is not read", gsub('or pr.get("pass") is not True or pr.get("launch_intent") == "off"', 'or pr.get("launch_intent") == "off"'), ["A_probe_fail"]),
    ("G24 a probe record of an OFF lever is accepted", gsub(' or pr.get("launch_intent") == "off":', ":"), ["A_probe_off"]),
    ("G25 the natural mask's wire is not checked", gsub("    elif n > 0 and not (", "    elif False and not ("), ["A_natural0"]),
    ("G26 the world-model control is skipped", gsub('\n    if pr.get("control_wm_grad_diff") not in (None, 0.0):', "\n    if False:"), ["A_wm_ctrl"]),
    ("G27 the built W is not compared", gsub("    if type(got) is not float or got != W:", "    if False:"), ["A_w_zero", "A_w_B", "A_w_bool"]),
    ("G28 the token records W 0.0", gsub('"launch_intent_w": LI_W, "ego_perturb_p": 0.0', '"launch_intent_w": 0.0, "ego_perturb_p": 0.0'), ["TOKEN"]),
    ("G29 the token binds the gate argv, not the pinned one", gsub('"argv_sha256": argv_sha256(t_argv)', '"argv_sha256": argv_sha256(g_argv)'), ["TOKEN"]),
    ("G30 the dose check on the lines is skipped", gsub("            if fl.get(k) != [[w[0]]]:", "            if False:"), ["A_W_respelled", "A_steps"]),
    ("G31 the init checkpoint is not compared", gsub("    if ic.get(\"sha256\") != a.banked_init_sha:", "    if False:"), ["A_init_sha"]),
    # ---- A_rep,s: the PAIR rule (probes: RA / OL / RB cases)
    ("G32 A_rep's trainer seed is the init seed (offset 0)", gsub('ARMS = {"launchintent": 0, "launchintentrep": 10}', 'ARMS = {"launchintent": 0, "launchintentrep": 0}'), ["RA00_pass_s0"]),
    ("G33 expected_from_ref ignores the arm's seed offset", gsub("    ts = seed + ARMS[arm]\n    r = drop(ref)", "    ts = seed\n    r = drop(ref)"), ["RA00_pass_s0", "RA00_pass_s1"]),
    ("G34 the reference's trainer seed is not checked (a swapped control is accepted)", gsub('    if r[kp + 1] != "0" or r[ks + 1] != str(ts):', '    if r[kp + 1] != "0":'), ["A_ref_is_rep", "RA_ref_is_oacc"]),
    ("G35 the lines' --seed is the init seed for every arm", gsub('"--seed": [str(tseed)]', '"--seed": [str(seed)]'), ["RA00_pass_s0", "A_seed_as_rep"]),
    ("G36 the swap config knows only arm A", gsub('"init_name": "M2b"} for a in ARMS}', '"init_name": "M2b"} for a in ("launchintent",)}'), ["RA00_pass_s0"]),
    ("G37 the token records the init seed as the trainer seed", gsub('"trainer_seed": int(a.seed) + ARMS[a.arm],', '"trainer_seed": int(a.seed),'), ["TOKEN_REP"]),
    ("G38 a token is written without the PREREG / reader pins", gsub('        if not re.fullmatch(r"[0-9a-f]{64}", v or ""):', "        if False:"), ["TOKEN_NOPIN"]),
    ("G39 the token does not record the PREREG pin", gsub('"prereg_sha256": a.prereg_sha, ', ""), ["TOKEN"]),
    ("K74 the run record does not read the planned argv's own --seed", ksub('    if fl_seed != [[str(tseed)]]:', "    if False:"), ["RB_planned_is_A", "B_planned_is_rep"]),
    ("K70 the run record compares the init seed with the summary's seed for A_rep", ksub('    if (sm.get("arm"), sm.get("seed")) != ("M2b", tseed):', '    if (sm.get("arm"), sm.get("seed")) != ("M2b", seed):'), ["RB00_pass_s0", "RB_seed_as_A"]),
    ("K71 the run record's trainer seed ignores the arm", ksub("    tseed = seed + ARMS[arm]", "    tseed = seed + 0 * ARMS[arm]"), ["RB00_pass_s0"]),
    ("K72 the one-lever command ignores the arm (A_rep checked as A)", ksub("G.expected_from_ref(ref, int(a.seed), a.arm)", "G.expected_from_ref(ref, int(a.seed))"), ["OL_R_pass_s0"]),
    ("K73 the one-lever command never refuses", ksub("    return 0 if ok else 1\n\n\ndef bands_cmd", "    return 0\n\n\ndef bands_cmd"), ["OL_A_extra", "OL_R_W_B"]),
    # ---- the checks module: run-record (probes: B cases)
    ("K01 summary's lever key not required", ksub("    bad += lever_record_problems(sm, \"summary.json\")\n", ""), ["B_stale_sm", "B_w_sm"]),
    ("K02 selfcheck's lever key not required", ksub("    bad += lever_record_problems(sc, \"selfcheck.json\")\n", ""), ["B_stale_sc", "B_w_sc"]),
    ("K03 an absent lever key passes the record test", ksub('    if not isinstance(lv, dict) or "launch_intent_w" not in lv:', "    if not isinstance(lv, dict):"), ["B_stale_sm"]),
    ("K04 a record without the launch keys passes", ksub("    if missing:\n        bad.append(", "    if False:\n        bad.append("), ["B_missing_n", "B_missing_rows"]),
    ("K05 argv.json is not compared with the plan", ksub('    if aj.get("argv") != planned:', "    if False:"), ["B_argv_json"]),
    ("K06 the pin is not compared with the plan", ksub("    if pinned != planned:", "    if False:"), ["B_pin"]),
    ("K07 an unfinished run passes", ksub('    if sm.get("done") is not True:\n', "    if False:\n"), ["B_done"]),
    ("K08 any step count passes", ksub('    if sm.get("steps") != steps:', "    if False:"), ["B_steps"]),
    ("K09 the run's seed is not compared", ksub('    if (sm.get("arm"), sm.get("seed")) != ("M2b", tseed):', "    if False:"), ["B_seed", "RB_seed_as_A"]),
    ("K10 K-INIT is not compared", ksub('    if ic.get("sha256") != a.banked_init_sha:', "    if False:"), ["B_init"]),
    ("K11 a perturbation may be built", ksub("        if ep.get(k) != w:", "        if False:"), ["B_ego_p"]),
    ("K12 the composition flag may be built", ksub("    if wd not in (False, None):\n        bad.append(f\"summary.json built", "    if False:\n        bad.append(f\"summary.json built"), ["B_compose"]),
    ("K13 the swap probability is not compared", ksub('    if lv.get("dagger_p") != G.SWAP_P:', "    if False:"), ["B_dagger_p"]),
    ("K14 the bank is not compared", ksub('    if lv.get("dagger_bank_sha256") != a.bank_sha.lower():', "    if False:"), ["B_bank"]),
    ("K15 w_acc is not compared", ksub('    if lv.get("w_acc") != WA:', "    if False:"), ["B_wacc"]),
    ("K16 the three argv_sha16 are not compared", ksub("        if v != sha16:", "        if False:"), ["B_sha16"]),
    ("K17 the run's own selfcheck may fail", ksub('    if sc.get("declared_vs_built") != "PASS":\n        bad.append("the run\'s own selfcheck.json is not a PASS")', '    if False:\n        bad.append("x")'), ["B_sc_fail"]),
    ("K18 the built mask definition is not compared", ksub("        for k, w in G.LI_MASK.items():\n            if li.get(k) != w:", "        for k, w in G.LI_MASK.items():\n            if False:"), ["B_mask"]),
    # ---- K-SHARE / K-MASK (probes: K cases)
    ("K20 launch_rows may differ from launch_eligible", ksub("    if r != e:\n", "    if False:\n"), ["K_rows_ne"]),
    ("K21 the K-SHARE band is +- 0.5", ksub("SHARE_TARGET, SHARE_TOL = 1.0 / 3.0, 0.02", "SHARE_TARGET, SHARE_TOL = 1.0 / 3.0, 0.5"), ["K_share_low", "K_share_high"]),
    ("K22 the K-SHARE band is +- 0.0001", ksub("SHARE_TARGET, SHARE_TOL = 1.0 / 3.0, 0.02", "SHARE_TARGET, SHARE_TOL = 1.0 / 3.0, 0.0001"), ["K00_pass", "K_share_in_lo", "K_share_in_hi"]),
    ("K23 the record's own share is not compared", ksub("or abs(rec - sh) > 1e-9", "or False"), ["K_share_rec"]),
    ("K24 a run with no launch row passes the count test", ksub("    if not (e > 0 and 0 < n <= e):", "    if False:"), ["K_n0"]),
    ("K25 the window list is not compared", ksub('    if sm.get("train_window_list_sha256") != window_sha:', "    if False:"), ["K_window"]),
    ("K26 a record lacking a launch key passes the K-SHARE test", ksub("    if missing:\n        return [", "    if False:\n        return ["), ["K_missing_n"]),
    ("K27 the top-level W is not compared", ksub("    if type(lw) is not float or lw != LI_W or sm.get(\"launch_intent_w\") != LI_W or (sm.get(\"levers\") or {}).get(\"launch_intent_w\") != LI_W:", "    if False:"), ["K_w_wrong"]),
    # ---- REP licence (probes: C cases)
    ("K30 the REP swap share is not banded", ksub("    if not ok or abs(n / e - SWAP_BAND[0]) > SWAP_BAND[1] + 1e-12:", "    if not ok:"), ["C_swap"]),
    ("K31 REP's perturbed count may be non-zero", ksub("    if not ok or n != 0:", "    if not ok:"), ["C_ego"]),
    ("K32 REP's composition flag may be built", ksub('    if findk(sm, "ego_perturb_with_dagger") not in (False, None):', "    if False:"), ["C_compose"]),
    ("K33 REP's stream is not read", ksub("        if type(st) is not int or st != 0:", "        if False:"), ["C_stream_absent", "C_stream_one", "C_stream_bool"]),
    ("K34 REP may carry the launch lever", ksub("    if lw not in (None, 0.0) or any(", "    if any("), ["C_launch"]),
    ("K35 REP's argv is not compared with DAG-OACC's", ksub('    if not ra or G.drop(ra, names=("--out", "--seed")) != G.drop(oa, names=("--out", "--seed")):', "    if not ra:"), ["C_argv"]),
    ("K36 REP's token fields are not compared with DAG-OACC's", ksub("        if tok.get(k) is None or tok.get(k) != otok.get(k):", "        if False:"), ["C_tok_init", "C_tok_bank", "C_tok_wacc", "C_tok_excl"]),
    ("K37 REP's token data hash is not compared", ksub('    if (tok.get("binding") or {}).get("data_sha256") != (otok.get("binding") or {}).get("data_sha256"):', "    if False:"), ["C_tok_data"]),
    ("K38 REP's window list is not compared with DAG-OACC's", ksub('    for k in ("train_window_list_sha256", "train_sha256", "excluded_sha256", "n_windows"):', "    for k in ():"), ["C_window"]),
    ("K39 an unfinished REP passes", ksub('    if sm.get("done") is not True or sm.get("steps") != 3000:', "    if False:"), ["C_done"]),
    ("K40 REP's token / selfcheck need not PASS", ksub('    if sc.get("declared_vs_built") != "PASS" or tok.get("verdict") != "PASS":', "    if False:"), ["C_tok_verdict"]),
    ("K41 REP's run-record line in the chain log is not required", ksub('        if not re.search(rf"^DAG-ORC \\(DAG-OACC base\\) run record: PASS \\| {run} \\|", txt, re.M):', "        if False:"), ["C_log_no_pass"]),
    ("K42 REP's seed is not compared", ksub('    if (sm.get("arm"), sm.get("seed")) != ("M2b", 10 + seed):', "    if False:"), ["C_seed"]),
    ("K43 a missing chain log passes the REP licence", ksub("    if a.log and not os.path.isfile(a.log):", "    if False:"), ["C_log_missing"]),
    # ---- the predecessor verdict (probes: P cases)
    ("K50 a guard stop is ignored", ksub('    if guard:\n        return "HALT"', '    if False:\n        return "HALT"'), ["P_guard"]),
    ("K51 a STOP is ignored", ksub('    if stop:\n        return "HALT"', '    if False:\n        return "HALT"'), ["P_stop", "P_stop_then_done"]),
    ("K52 text after the DONE marker is ignored", ksub("    if not body or body[-1] != done[0]:", "    if False:"), ["P_after"]),
    ("K53 any chain's DONE marker is accepted", ksub("    if not DONE_RE.match(done[0]):", "    if False:"), ["P_foreign_done"]),
    ("K54 the four pulls / records are not required", ksub("    for run in R2_RUNS:\n        if not re.search(", "    for run in ():\n        if not re.search("), ["P_missing_pull", "P_missing_rr", "P_missing_kreach"]),
    ("K55 a second DONE marker is accepted", ksub("    if len(done) != 1:", "    if False:"), ["P_two_done"]),
    ("K56 a held BOX2_BUSY is ignored", ksub('    if busy_exists:\n        return "BUSY"', '    if False:\n        return "BUSY"'), ["P_busy"]),
    ("K57 a log with no end marker is not a WAIT", ksub('    if not done:\n        return "WAIT"', '    if False:\n        return "WAIT"'), ["P_wait", "P_empty"]),
    ("K58 named stops are ignored", ksub("    if floor or ctrl:", "    if False:"), ["P_floor", "P_ctrl"]),
    ("K59 a FAIL line behind a DONE is ignored", ksub('    if re.search(r"^DAG-ORC \\(DAG-OACC base\\) (run record|K-REACH): FAIL", txt, re.M):', "    if False:"), ["P_fail_line"]),
    ("K61 a dated STOP echo counts as a STOP (not anchored)", ksub('stop = [x for x in lines if x.startswith("ZZV8FMINI-STOP")]', 'stop = [x for x in lines if "ZZV8FMINI-STOP" in x]'), ["P_dated_echo"]),
    ("K60 markers are not anchored", ksub('done = [x for x in lines if x.startswith("ZZV8FMINI-DONE")]', 'done = [x for x in lines if "ZZV8FMINI-DONE" in x]'), ["P_dated_echo_done"]),
]


def apply_mut(spec):
    which, old, new, n = spec
    path = S.SHIP / ("gate_launchintent_assert.py" if which == "G" else "v8f_launchintent_chain_checks.py")
    t = path.read_text(encoding="utf-8")
    assert t.count(old) == n, (spec[1][:80], t.count(old))
    t2 = t.replace(old, new)
    return S.mods(gtext=t2) if which == "G" else S.mods(ktext=t2)


CASE_KIND = {}
for c in A_ALL:
    CASE_KIND[c[0]] = ("A", c)
for c in B_ALL:
    CASE_KIND[c[0]] = ("B", c)
for c in KS_ALL:
    CASE_KIND[c[0]] = ("K", c)
for c in OL_CASES:
    CASE_KIND[c[0]] = ("OL", c)
for c in C_CASES:
    CASE_KIND[c[0]] = ("C", c)
for c in P_CASES:
    CASE_KIND[c[0]] = ("P", c)
CASE_KIND["P_dated_echo_done"] = ("P", ("P_dated_echo_done", lambda: S.r2_log(done=False).replace("waiting for the r1 chain (x)", "10-28 03:48 waiting for ZZV8FMINI-DONE"), False, "WAIT"))


def probe(mods_pair, cid, tag):
    kind, case = CASE_KIND[cid]
    Gm, Km = mods_pair
    if kind == "A":
        return eval_a(Gm, case, tag)
    if kind == "B":
        return eval_b(Km, case, tag)
    if kind == "K":
        return eval_ks(Km, case, tag)
    if kind == "OL":
        return eval_ol(Km, case, tag)
    if kind == "C":
        return eval_c(Km, case, tag)
    if kind == "P":
        return eval_p(Km, case)
    raise KeyError(cid)


PIN_P, PIN_R = "c" * 64, "d" * 64


def token_ok(Gm, tag, arm="launchintent", prereg=PIN_P, reader=PIN_R, want_written=True) -> bool:
    root = S.fresh(f"A/TOKEN_{tag}")
    d = S.gate_world(G0, root, 0, arm=arm)
    run = f"M2b_s0_{arm}"
    d.update(mode="token", commit="a" * 40, queue_sha="b" * 64, key_sha_json=json.dumps({"x": "1"}), run=run, prereg_sha=prereg, reader_sha=reader, out=str(root / "tok.json"))
    try:
        rc = S.quiet(lambda: (Gm.write_token(S.NS(d))))
    except Exception:                                                   # noqa: BLE001
        return False
    if not want_written:
        return rc != 0 and not Path(d["out"]).exists()
    if rc != 0:
        return False
    tok = S.load(d["out"])
    pin = Gm.argv_of(open(d["pin_line"], encoding="utf-8").read())
    return (tok["arm"] == arm and tok["launch_intent_w"] == LI_W and tok["verdict"] == "PASS" and tok["binding"]["argv_sha256"] == Gm.argv_sha256(pin) and tok["run"] == run
            and tok.get("trainer_seed") == (10 if arm == "launchintentrep" else 0) and tok.get("init_seed") == 0 and tok.get("prereg_sha256") == prereg and tok.get("reader_sha256") == reader
            and tok["binding"]["commit"] == "a" * 40 and tok["init_ckpt_sha256"] == d["banked_init_sha"] and tok["bank_sha256"] == d["bank_sha"] and tok["w_acc"] == 49.6969 and tok["prereg"] == "PREREG_V8F_LAUNCH_INTENT")


def run_tests():
    shutil.rmtree(S.SCR / "A", ignore_errors=True)
    for c in A_ALL:
        ok, bad = eval_a(G0, c, "orig")
        S.T(f"asserter {c[0]}: " + ("PASS" if c[2] is None else f"refused ({c[2]})"), ok, bad)
    S.T("asserter token (A_0): written on a PASS gate with the registered fields (arm, W, commit, pinned argv sha256, init, bank, w_acc, trainer seed 0, the PREREG and reader pins)", token_ok(G0, "orig"))
    S.T("asserter token (A_rep,0): trainer seed 10, init seed 0, run name M2b_s0_launchintentrep, the same pins", token_ok(G0, "orig_rep", arm="launchintentrep"))
    S.T("asserter token: a placeholder or empty PREREG pin gets NO token (and no file)", token_ok(G0, "nopin_p", prereg="", want_written=False))
    S.T("asserter token: a placeholder reader pin gets NO token (and no file)", token_ok(G0, "nopin_r", reader="<<UNSET>>", want_written=False))
    try:
        S.quiet(G0.main, ["check", "--selfcheck", "x", "--gate-line", "x", "--train-line", "x", "--pin-line", "x", "--arm", "dagoaccorc", "--seed", "0", "--banked-init-sha", "x", "--bank-sha", "x", "--bank-json", "x", "--w-acc", "1"])
        refused = False
    except SystemExit as e:
        refused = e.code == 2
    S.T("asserter: a sibling chain's arm name (dagoaccorc) is not accepted", refused)
    ref = copy.deepcopy(S.load(S.OACC / "M2b_s0_dagoacc" / "argv.json")["argv"])
    for nm, r in (("the launch flag", ref + ["--launch-intent-w", "1.58926"]), ("the composition flag", ref + ["--ego-perturb-with-dagger"]), ("a law flag", ref + ["--ego-perturb-vmax", "8.0"])):
        try:
            G0.expected_from_ref(r, 0)
            ok = False
        except ValueError:
            ok = True
        S.T(f"asserter: a reference argv that already carries {nm} is refused", ok)
    for c in B_ALL:
        ok, bad = eval_b(K0, c, "orig")
        S.T(f"run-record {c[0]}: " + ("PASS" if c[2] is None else f"refused ({c[2]})"), ok, bad)
    for c in KS_ALL:
        ok, bad = eval_ks(K0, c, "orig")
        S.T(f"K-SHARE/K-MASK {c[0]}: " + ("PASS" if c[2] is None else f"refused ({c[2]})"), ok, bad)
    for c in OL_CASES:
        ok, why = eval_ol(K0, c, "orig")
        S.T(f"one-lever command {c[0]}: " + ("ONE-LEVER (rc 0)" if c[5] == 0 else "NOT-ONE-LEVER (rc 1)"), ok, why)
    # ---- the pair rule on the REAL banked records (not fixtures)
    for s in (0, 1):
        rv, ov = S.load(S.REPDIR / f"M2b_s{s}_dagoaccorcrep" / "argv.json")["argv"], S.load(S.OACC / f"M2b_s{s}_dagoacc" / "argv.json")["argv"]
        o_at_rep = list(ov)
        o_at_rep[o_at_rep.index("--seed") + 1] = str(10 + s)
        S.T(f"pair s{s}: REP_s's banked argv == DAG-OACC_s's at --seed {10 + s}, apart from --out only (so A_rep,s = DAG-OACC_s's argv at --seed {10 + s} + the lever)", G0.drop(rv) == G0.drop(o_at_rep), "")
        S.T(f"pair s{s}: the registered argv of A_rep,s built from REP_s == built from DAG-OACC_s at --seed {10 + s}, ending `--launch-intent-w 1.58926`", G0.expected_from_ref(rv, s, "launchintentrep") == G0.drop(o_at_rep) + ["--launch-intent-w", "1.58926"])
        S.T(f"pair s{s}: A_s and A_rep,s differ in --seed (and --out) alone", [i for i, (x, y) in enumerate(zip(G0.expected_from_ref(ov, s, "launchintent"), G0.expected_from_ref(rv, s, "launchintentrep"))) if x != y] == [G0.drop(ov).index("--seed") + 1])
        rsm, osm = S.load(S.REPDIR / f"M2b_s{s}_dagoaccorcrep" / "summary.json"), S.load(S.OACC / f"M2b_s{s}_dagoacc" / "summary.json")
        S.T(f"pair s{s}: REP_s's banked summary shares DAG-OACC_s's data, parameter groups and window list (the gate comparator for A_rep,s)",
            all(rsm[k] == osm[k] for k in ("train_sha256", "train_window_list_sha256", "n_windows")) and rsm["levers"]["params"] == osm["levers"]["params"] and rsm["levers"]["n_params_trainable"] == osm["levers"]["n_params_trainable"]
            and rsm["levers"]["w_acc"] == osm["levers"]["w_acc"] and rsm["seed"] == 10 + s and osm["seed"] == s,
            [k for k in ("train_sha256", "train_window_list_sha256", "n_windows") if rsm[k] != osm[k]])
    for c in C_CASES:
        ok, bad = eval_c(K0, c, "orig")
        S.T(f"REP licence {c[0]}: " + ("PASS" if c[4] is None else f"refused ({c[4]})"), ok, bad)
    root = S.fresh("C/nofile")
    bad = S.quiet(K0.rep_licence, S.NS(S.rep_args({"rep_dir": str(root / "nope_rep"), "oacc_dir": str(root / "nope_oacc")}, 0)))
    S.T("REP licence: a missing REP pull is a FAIL (REP must be trained anew)", any("is missing" in b for b in bad))
    for c in P_CASES + [CASE_KIND["P_dated_echo_done"][1]]:
        ok, why = eval_p(K0, c)
        S.T(f"predecessor verdict {c[0]}: {c[3]}", ok, why)
    # the arithmetic the PREREG registers, recomputed
    S.T("W_A recomputed from the measured f by PREREG section 3's rule is 1.58926; W_B 0.726175; W_HI 4.17852", (S.C.w_from_rule(1, 3, S.C.f_measured()), S.C.w_from_rule(1, 4, S.C.f_measured()), S.C.w_from_rule(1, 2, S.C.f_measured())) == ("1.58926", "0.726175", "4.17852"))
    b = K0.bands(LI_W)
    S.T("K-SHARE band is [0.3133, 0.3533] and the measured f gives share 1/3 inside it", abs(b["share"][0] - 0.31333) < 1e-4 and abs(b["share"][1] - 0.35333) < 1e-4 and abs(b["share_at_f"] - 1 / 3) < 1e-6)
    src = (S.C.STAGE / "stack/scripts/v8f_mini_train.py").read_text(encoding="utf-8")
    m = re.search(r"^def launch_weight_share\(.*?(?=^def )", src, re.M | re.S)
    ns = {}
    exec("from __future__ import annotations\n" + m.group(0), ns)
    diffs = [abs(ns["launch_weight_share"](n, e, LI_W) - K0.share_of(n, e, LI_W)) for n, e in ((30727, 191203), (30674, 191396), (1, 10), (5, 5))]
    S.T("share_of equals the STAGED trainer's launch_weight_share (extracted from its source) at four count pairs", max(diffs) == 0.0, diffs)
    S.T("the helpers extracted from the r2 asserter are byte-identical to it (argv_of flags argv_sha256 drop finite wacc_problems swap_problems)",
        all(re.search(rf"^def {n}\(.*?(?=^def |\Z)", (S.C.R2SHIP / 'gate_dagoaccorc_r2_assert.py').read_text(encoding='utf-8'), re.M | re.S).group(0).rstrip("\n") + "\n" in (S.SHIP / 'gate_launchintent_assert.py').read_text(encoding='utf-8')
            for n in ("argv_of", "flags", "argv_sha256", "drop", "finite", "wacc_problems", "swap_problems")))


def run_mutants():
    for name, spec, probes in MUTANTS_PY:
        try:
            pair = apply_mut(spec)
        except AssertionError as e:
            S.M(name, False, f"mutation did not apply: {e}")
            continue
        killed, why = False, "no probe flipped"
        for cid in probes:
            if cid.startswith("TOKEN"):
                ok = {"TOKEN": lambda: token_ok(pair[0], "mut"), "TOKEN_REP": lambda: token_ok(pair[0], "mut_rep", arm="launchintentrep"),
                      "TOKEN_NOPIN": lambda: token_ok(pair[0], "mut_nopin", prereg="", want_written=False)}[cid]()
                if not ok:
                    killed, why = True, f"{cid}: the token behaviour changed"
                    break
                continue
            ok, bad = probe(pair, cid, "mut")
            if not ok:
                killed, why = True, f"{cid}: {str(bad)[:160]}"
                break
        S.M(name + (f"  [by {why}]" if killed else ""), killed, why)
