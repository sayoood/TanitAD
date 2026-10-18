#!/usr/bin/env python3
"""Tests for B8 (raw/2026-10-18-r0e-carvefeed): the CARVE FEED of the SAFEOBJ dual, `--epo-con-feed {train,carve}` of refe/scorer_finetune.py.  Run:
    PYTHONIOENCODING=utf-8 python refe/test_carve_feed.py   ->   ZZTEST_PASS <n>   (prints the hop trace and the mutation table).  CPU only, no pod, no checkpoint.

WHY IT EXISTS.  The SAFEOBJ dual read only `con_acc`, summed from the TRAINING micro-batches (scorer_finetune.py, the loop).  SO-CARVE (raw/2026-10-18-so-carve) established that a carve of RL-train cut from R0's
data is in-sample for the policy (R0 consumed every RL-train log), so a carve is only a held-out feed if the carve's logs are REMOVED from training.  The committed next arm of outcome row P-O4-PICK feeds the
dual from such a carve.  This patch builds the feed; the default (`train`) is every arm before B8, bit for bit.

WHAT IS PROVEN (CLAUDE.md "built, tested, and unreachable from its caller": a mechanism that WORKS is not a mechanism that RUNS):
  P  PINNED.  Peeling the B8 layer (carvefeed_layer.CARVEFEED_FILE_REVERSALS) off the patched trainer gives the 821915a tip blob EXACTLY (869b1862); the parser gained exactly four flags; the set of functions that
     differ from the tip is exactly the named set; the defaults are train / '' / '' / 1.
  R  REACHABILITY, static (AST): argparse -> main (carve split of tr_idx, cdl) -> carve_stream -> carve_feed_acc (no_grad, fwd + the loop's own epo_con_term at the SAME T / eps / lam) -> the loop (before o.step())
     -> dual.step(carve_acc) -> EpoDual.step reads acc["g_<c>"] -> dual.lam (Python floats) -> train_losses' epo_con_term on the TRAINING micro-batches -> lB -> backward.  The hop trace prints file:line per hop.
  U  UNIT, on the REAL closure exec'd from main()'s source: the carve accumulator carries NO graph (no_grad inside fwd, no .grad on either arm after the call, floats only); the zero control (B == A) reads EXACTLY 0.0;
     a moved B reads != 0; --epo-carve-micro 3 sums three micro-batches in order; a micro-batch row outside the carve logs raises.  The refusals (feed/levers/sha/micro/aux) are clean on the good argv and fire on the bad.
  E  END TO END through the REAL main() (test_rl_epo.smoke: CPU stubs for the GPU parts only), with a recorder on epo_con_term / EpoDual.step:
       E-id   default == explicit `--epo-con-feed train` == the PRE-B8 trainer (peeled): every logged row and every saved tensor identical, max |diff| EXACTLY 0.0; no `epo_con_carve` block in the data event.
       E-feed in the carve run the dual's accumulator is the CARVE's (n == micro; g == the no-grad carve call's g, exactly), every no-grad call scores carve-log sets only, every training call scores NON-carve sets only;
              in the control run (same training stream, feed = train) the accumulator is the two training calls' sum.
       E-attr carve vs control (carve logs removed from training in BOTH, so the training stream is identical): arm A is bit-identical (0.0) and arm B differs; the lambda trajectory differs.  The feed is isolated
              from the data removal, which is the attribution the draft prereg R0e-pick 2.3 says the carve lever needs.
       E-refuse the real main() refuses (rc 4, before any training) a wrong sha, a list that overlaps --heldout-logs, a list naming logs that are not RL-train logs.
Every mutation below edits SOURCE (each snippet must occur EXACTLY once, else the test errors) and must turn >= 1 expected check RED.
"""
from __future__ import annotations

import ast as _ast
import copy
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import carvefeed_layer as CL  # noqa: E402
import test_rl_epo as TE  # noqa: E402
import test_rl_s1b as S1B  # noqa: E402
import test_rl_safeobj as SO  # noqa: E402

SF_PATH = os.path.join(HERE, "scorer_finetune.py")
CLG_PATH = os.path.join(HERE, "carve_logs.py")
SOCARVE_PATH = os.path.join(HERE, "..", "raw", "2026-10-18-so-carve", "so_carve_analysis.py")
SF_REL = "refe/scorer_finetune.py"
CARVE_LOGS = "T3 T7 T11 T19 T21\n"                       # 5 of the stub's 25 RL-train logs (4 sets each) = 20 sets
HO_EXT = "H0 H1 H2 H3 T3 T7 T11 T19 T21\n"               # the control: the carve logs removed from training through --heldout-logs (same training index, same stream), feed = train
EXTRA = SO.SMOKE_ON + ("--epochs", "2")                  # 2 epochs so that 25 updates are reached with the carve removed (80 sets / (2 x 2) = 20 updates per epoch)
NEW_FLAGS = {"--epo-con-feed", "--epo-carve-logs", "--epo-carve-sha256", "--epo-carve-micro"}
ALLOWED_DIFF = {"build_parser", "main", "epo_con_refusals", "epo_carve_refusals", "main.carve_stream", "main.carve_feed_acc"}
VOLATILE = {"model_A", "model_B", "at", "hours", "s_per_microbatch", "s_per_update", "gpu_mem_gb", "step_s", "elapsed", "t", "time", "sec", "secs"}
LOG_OF = {np.asarray(v[2])[:, :6].astype(np.float32).tobytes(): k[0] for k, v in TE.StubBank("train", 8, 4).by.items()}          # a set's PDM labels -> its log


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
    if set(a) != set(b):
        return float("inf")
    return max((float((a[k].double() - b[k].double()).abs().max()) for k in a if a[k].numel()), default=0.0)


# ================================================================================================ the real main() with a recorder
def run_smoke(src: str, tag: str, extra=(), carve=CARVE_LOGS, sha=None, ho_ext=False, logs_file=None) -> dict:
    """-> the smoke result + {"rec": {"calls", "steps"}} or {"exc": ...}.  `carve` = the list's text (None = the default feed); sha defaults to its true sha256."""
    rec = {"phase": 0, "calls": [], "steps": []}

    def hook(sm):
        o_main, o_term, o_step = sm.main, sm.epo_con_term, sm.EpoDual.step

        def main_w():
            rec["phase"] += 1
            return o_main()

        def term_w(sxB, sxA, tg, rule_fn, T, eps, lam, parts=None):
            out = o_term(sxB, sxA, tg, rule_fn, T, eps, lam, parts)
            t_ = tg.detach().cpu().numpy()
            rec["calls"].append({"phase": rec["phase"], "grad": torch.is_grad_enabled(), "logs": [LOG_OF.get(t_[i, :, :6].astype(np.float32).tobytes()) for i in range(t_.shape[0])],
                                 "parts": dict(parts) if parts is not None else {}})
            return out

        def step_w(self, acc):
            if "cA_ttc" in acc:
                rec["steps"].append({"phase": rec["phase"], "acc": dict(acc)})
            return o_step(self, acc)
        sm.main, sm.epo_con_term, sm.EpoDual.step = main_w, term_w, step_w
    with tempfile.TemporaryDirectory() as td:
        ex = list(extra)
        if carve is not None:
            p = os.path.join(td, "carve.txt")
            open(p, "wb").write(carve.encode())
            ex += ["--epo-con-feed", "carve", "--epo-carve-logs", p, "--epo-carve-sha256", sha or hashlib.sha256(carve.encode()).hexdigest()]
        if ho_ext:
            p2 = os.path.join(td, "ho_ext.txt")
            open(p2, "wb").write(HO_EXT.encode())
            ex += ["--heldout-logs", p2]
        try:
            r = TE.smoke(src, td, epo=True, native7=True, terms=True, tag=tag, extra=tuple(ex), hook=hook)
        except Exception as e:                                  # noqa: BLE001 -- a mutated trainer that raises is a RED run, and says so
            return {"exc": f"{type(e).__name__}: {str(e)[:120]}", "rec": rec}
    r["rec"] = rec
    return r


def ok_run(r) -> bool:
    return "exc" not in r and r["pre"] == 0 and r["run"] == 0 and r["pass_json"] is not None and r["A"] is not None and r["B"] is not None


def build_refs(sf_src: str) -> dict:
    """the reference runs that do not depend on a mutation: the PRE-B8 trainer (peeled) on the default argv"""
    return {"peeled": run_smoke(CL.peel(sf_src), "cf_peeled", EXTRA, carve=None)}


# ================================================================================================ the suite
def suite(sf_src: str, tag: str, refs: dict, trace: bool = False) -> list:
    bad: list = []
    C = TE.Ctx(sf_src, f"sf_cf_{tag}")
    M = C.mod
    try:
        tip_src = CL.peel(sf_src)
        seg, seg_tip = S1B.segments(sf_src), S1B.segments(tip_src)
    except AssertionError:
        tip_src = seg = seg_tip = None
    msrc = _ast.unparse(C.main)

    # ---------------------------------------------------------------- P: pinned
    check(bad, "P_peeling_the_B8_layer_gives_the_821915a_trainer_blob_869b1862", lambda: tip_src is not None and CL.has_layer(sf_src) and S1B.git_blob(tip_src.encode("utf-8")) == CL.TIP_TRAINER_BLOB)

    def only_flags():
        old = TE.load(tip_src, f"sf_cf_old_{tag}")
        nm = {tuple(x.option_strings): x for x in M.build_parser()._actions}
        om = {tuple(x.option_strings): x for x in old.build_parser()._actions}
        same = all(k in nm and all(getattr(om[k], at) == getattr(nm[k], at) for at in ("dest", "default", "type", "choices", "required", "nargs", "const", "help")) for k in om)
        return same and {k[0] for k in set(nm) - set(om)} == NEW_FLAGS
    check(bad, "P_the_parser_gained_exactly_four_flags_and_every_other_action_is_unchanged", only_flags)
    check(bad, "P_defaults_are_train_empty_empty_one_and_named_constants", lambda: M.EPO_CON_FEED_DEFAULT == "train" and M.EPO_CON_FEEDS == ("train", "carve") and M.build_parser().get_default("epo_con_feed") == "train"
          and M.build_parser().get_default("epo_carve_logs") == "" and M.build_parser().get_default("epo_carve_sha256") == "" and M.build_parser().get_default("epo_carve_micro") == 1
          and set(M.EPO_CARVE_DESTS) == {"epo_carve_logs", "epo_carve_sha256", "epo_carve_micro"})
    check(bad, "P_exactly_the_named_functions_differ_from_the_tip_and_the_training_loss_assembly_is_untouched",
          lambda: {k for k in set(seg) | set(seg_tip) if seg.get(k) != seg_tip.get(k)} == ALLOWED_DIFF and all(seg[k] == seg_tip[k] for k in ("epo_con_term", "EpoDual") if k in seg))

    # ---------------------------------------------------------------- R: reachability, static
    ft = C.nested["carve_feed_acc"] if "carve_feed_acc" in C.nested else None
    calls_named = lambda node, nm: [n for n in _ast.walk(node) if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name) and n.func.id == nm]          # noqa: E731
    check(bad, "R_hop_1_parser_declares_the_switch_with_choices_train_carve_default_train", lambda: any(x.dest == "epo_con_feed" and x.option_strings == ["--epo-con-feed"] and tuple(x.choices) == ("train", "carve")
                                                                                               and x.default == "train" for x in M.build_parser()._actions))
    check(bad, "R_hop_2_main_cuts_the_carve_logs_out_of_tr_idx_and_builds_the_carve_loader_from_carve_idx_only_under_the_switch", lambda: 'if a.epo_con_feed == "carve":' in sf_src
          and "tr_idx = [i for i in tr_idx if ds.rows[i].get(\"log_name\") not in carve_logs]" in sf_src and "cdl = DataLoader(make_train_set(ds, carve_idx, a.epo)" in sf_src
          and "carve_idx = [i for i in tr_idx if ds.rows[i].get(\"log_name\") in carve_logs]" in sf_src)
    check(bad, "R_hop_3_the_carve_stream_iterates_cdl_and_main_binds_carve_it_only_when_cdl_exists", lambda: "for b_ in cdl:" in sf_src and "carve_it = carve_stream() if cdl is not None else None" in sf_src)

    def acc_hops():
        wi = [n for n in _ast.walk(ft) if isinstance(n, _ast.With) and "torch.no_grad()" in _ast.unparse(n.items[0].context_expr)]
        inner = calls_named(wi[0], "fwd") + calls_named(wi[0], "epo_con_term") if wi else []
        tl = calls_named(C.nested["train_losses"], "epo_con_term")
        cc = calls_named(ft, "epo_con_term")
        return (len(wi) == 1 and len(inner) == 2 and len(cc) == 1 and len(tl) == 1
                and [_ast.unparse(x) for x in cc[0].args[3:7]] == [_ast.unparse(x) for x in tl[0].args[3:7]] == ["sel_score", "dual.temp", "a.epo_con_eps", "dual.lam"])
    check(bad, "R_hop_4_carve_feed_acc_scores_fwd_and_epo_con_term_inside_no_grad_with_the_SAME_rule_T_eps_lam_as_the_training_call", acc_hops)
    check(bad, "R_hop_5_the_loop_calls_carve_feed_acc_inside_the_update_before_o_step_on_the_same_weights", lambda: 0 < line_of(sf_src, "if micro % a.accum:") < line_of(sf_src, "carve_acc = carve_feed_acc() if carve_it is not None else None")
          < line_of(sf_src, "for o, ps in ((optA, paramsA), (optB, paramsB)):"))

    def one_step_call():
        cs = [n for n in _ast.walk(C.main) if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Attribute) and n.func.attr == "step" and _ast.unparse(n.func.value) == "dual"]
        return len(cs) == 1 and _ast.unparse(cs[0].args[0]) == "con_acc if carve_acc is None else carve_acc"
    check(bad, "R_hop_6_dual_step_is_fed_the_carve_accumulator_when_the_switch_is_on_and_con_acc_otherwise", one_step_call)
    check(bad, "R_hop_7_EpoDual_step_reads_the_accumulator_s_g_into_lambda", lambda: 'raw = self.lam[c] + self.lr * (float(acc["g_" + c]) / n)' in sf_src and "self.lam[c] = min(self.lam_max, max(0.0, raw))" in sf_src)
    check(bad, "R_hop_8_lambda_reaches_the_training_loss_train_losses_passes_dual_lam_ct_joins_lB_and_the_loop_backwards_it", lambda: "lB = lB + ct" in sf_src and "((lA + lB) / a.accum).backward()" in sf_src
          and "ct = epo_con_term(sxB, sxA, tg, sel_score, dual.temp, a.epo_con_eps, dual.lam, con_last)" in sf_src)

    def not_dead():
        import gate_checks as GC
        return NEW_FLAGS_DESTS <= GC.namespace_reads(sf_src, GC.SCORER_NAMESPACE_FUNCS)
    NEW_FLAGS_DESTS = {"epo_con_feed", "epo_carve_logs", "epo_carve_sha256", "epo_carve_micro"}
    check(bad, "R_hop_9_the_gate_s_own_dead_lever_scan_sees_main_read_all_four_flags", not_dead)

    # ---------------------------------------------------------------- U: the real closure
    def mk_u(micro=1, move=True, stray=False, n_batches=None):
        A, B = TE.Tiny(0, 6), TE.Tiny(0, 6)
        if move:
            g = torch.Generator().manual_seed(11)
            with torch.no_grad():
                for p in B.parameters():
                    p.add_(torch.randn(p.shape, generator=g) * 0.4)
        data = TE.micro_data(6, 3, 5)
        seen = {"grad": [], "k": 0}

        def fwd(cb):
            k = seen["k"]
            seen["k"] += 1
            cxy, sctx, tg = data[k % 3]
            seen["grad"].append(torch.is_grad_enabled())
            return torch.zeros(1), A(cxy, sctx), B(cxy, sctx), tg
        ds = types.SimpleNamespace(rows=[{"log_name": ("X9" if (stray and i == 2) else "C1")} for i in range(4)])
        batch = tuple(torch.zeros(1) for _ in range(9)) + (torch.tensor([0, 1, 2, 3]),)
        an = TE.a_ns("listnet", False, lam=0.0, epo=True, epo_carve_micro=micro, epo_con_eps=0.0)
        d = M.EpoDual(("ttc", "nc"), 0.0, 4.0, 10.0, 0.0, temp=0.1)
        d.lam = {c: 1.0 for c in d.comps}
        ns = C.ns(["sel_score", "carve_feed_acc"], a=an, ds=ds, carve_logs={"C1"}, carve_it=iter([batch] * (n_batches or micro)), fwd=fwd, dual=d)
        return A, B, data, seen, ns, d

    def direct(A, B, data, ns, d, ks):
        """the expected accumulator, INDEPENDENTLY: epo_con_term called straight on the same arrays, summed in order"""
        acc = {"n": 0}
        with torch.no_grad():
            for k in ks:
                cxy, sctx, tg = data[k % 3]
                pr: dict = {}
                M.epo_con_term(B(cxy, sctx), A(cxy, sctx), tg, ns["sel_score"], d.temp, 0.0, d.lam, pr)
                for kk, v in pr.items():
                    acc[kk] = acc.get(kk, 0.0) + v
                acc["n"] += 1
        return acc

    def u1():
        A, B, data, seen, ns, d = mk_u()
        acc = ns["carve_feed_acc"]()
        exp = direct(A, B, data, ns, d, [0])
        return (acc["n"] == 1 and seen["grad"] == [False] and acc == exp and acc["g_ttc"] != 0.0 and all(isinstance(v, (float, int)) and not torch.is_tensor(v) for v in acc.values())
                and all(p.grad is None for p in list(A.parameters()) + list(B.parameters())))
    check(bad, "U_1_the_carve_accumulator_carries_NO_graph_fwd_runs_with_grad_disabled_no_param_grad_floats_only_and_equals_an_independent_epo_con_term_call", u1)

    def u2():
        A, B, data, seen, ns, d = mk_u(move=False)
        acc = ns["carve_feed_acc"]()
        return acc["g_ttc"] == 0.0 and acc["g_nc"] == 0.0 and acc["n"] == 1
    check(bad, "U_2_ZERO_CONTROL_B_equals_A_reads_EXACTLY_0.0_so_the_probe_can_read_zero_and_U_1_s_nonzero_is_B_s_doing", u2)

    def u3():
        A, B, data, seen, ns, d = mk_u(micro=3, n_batches=3)
        acc = ns["carve_feed_acc"]()
        exp = direct(A, B, data, ns, d, [0, 1, 2])
        return acc["n"] == 3 and acc == exp and seen["k"] == 3 and acc["g_ttc"] != direct(A, B, data, ns, d, [0])["g_ttc"]
    check(bad, "U_3_epo_carve_micro_3_scores_three_micro_batches_and_sums_them_in_order", u3)

    def u4():
        A, B, data, seen, ns, d = mk_u(stray=True)
        try:
            ns["carve_feed_acc"]()
        except RuntimeError as e:
            return "not in the carve logs" in str(e)
        return False
    check(bad, "U_4_a_carve_micro_batch_row_outside_the_carve_logs_raises", u4)

    REQ = ["--ckpt", "x", "--targets", "x", "--images", "x", "--calib", "x", "--onpolicy", "x", "--heldout", "x", "--heldout-logs", "x", "--out", "x"]
    EPO_OK = ["--epo", "--pdm-labels-train", "d", "--require-pdm", "--b-mode", "listnet", "--lam", "0", "--epo-terms-train", "t"]
    G5 = ["--epo-con", "g5", "--epo-con-eps", "0", "--epo-con-lr", "4", "--epo-con-max", "10", "--epo-con-lam0", "0"]
    CV = ["--epo-con-feed", "carve", "--epo-carve-logs", "c.txt", "--epo-carve-sha256", "ab" * 32]

    def refusals(extra, base=None):
        ap = M.build_parser()
        return M.epo_con_refusals(ap.parse_args(REQ + (EPO_OK if base is None else base) + extra), ap)
    check(bad, "U_refusal_the_default_feed_and_a_complete_carve_argv_are_clean", lambda: refusals(G5) == [] and refusals(G5 + ["--epo-con-feed", "train"]) == [] and refusals(G5 + CV) == [] and refusals(G5 + CV + ["--epo-carve-micro", "4"]) == [])
    check(bad, "U_refusal_carve_levers_named_while_the_feed_is_train_or_off_are_refused", lambda: all(any("silently ignored" in m for m in refusals(G5 + x)) for x in (["--epo-carve-logs", "c.txt"], ["--epo-carve-sha256", "ab" * 32], ["--epo-carve-micro", "4"]))
          and any("--epo-carve" in m for m in refusals(["--epo-con", "off", "--epo-carve-micro", "4"])))
    check(bad, "U_refusal_carve_without_the_dual_or_without_epo_is_refused", lambda: any("needs --epo and --epo-con g5" in m for m in refusals(["--epo-con", "off"] + CV[:2] + CV[2:])) and any("needs --epo and --epo-con g5" in m for m in refusals(CV, base=[])))
    check(bad, "U_refusal_carve_without_logs_or_with_a_malformed_sha_or_a_bad_micro_is_refused", lambda: any("--epo-carve-logs" in m for m in refusals(G5 + ["--epo-con-feed", "carve", "--epo-carve-sha256", "ab" * 32]))
          and any("64 lowercase hex" in m for m in refusals(G5 + CV[:4] + ["--epo-carve-sha256", "XYZ"])) and any("64 lowercase hex" in m for m in refusals(G5 + CV[:4]))
          and any(">= 1" in m for m in refusals(G5 + CV + ["--epo-carve-micro", "0"])))
    check(bad, "U_refusal_carve_with_the_aux_head_is_refused", lambda: any("--aux-drv" in m for m in refusals(G5 + CV + ["--aux-drv"])))

    # ---------------------------------------------------------------- E: the real main()
    R = {k: refs["peeled"] if k == "peeled" else None for k in ("peeled",)}
    R["dflt"], R["expl"] = run_smoke(sf_src, f"cf_d_{tag}", EXTRA, carve=None), run_smoke(sf_src, f"cf_e_{tag}", EXTRA + ("--epo-con-feed", "train"), carve=None)
    R["ctrl"], R["carve"] = run_smoke(sf_src, f"cf_c_{tag}", EXTRA, carve=None, ho_ext=True), run_smoke(sf_src, f"cf_k_{tag}", EXTRA)
    for k, r in R.items():
        check(bad, f"E_{k}_the_real_main_runs_preflight_PASS_then_trains", lambda r=r: ok_run(r))
    data_of = lambda r: [x for x in r["rows"]["sft.jsonl"] if x.get("event") == "data"][0]          # noqa: E731
    trains = lambda r, ph=2: [c for c in r["rec"]["calls"] if c["phase"] == ph and c["grad"]]          # noqa: E731
    nograd = lambda r, ph=2: [c for c in r["rec"]["calls"] if c["phase"] == ph and not c["grad"]]          # noqa: E731
    steps = lambda r: [s for s in r["rec"]["steps"] if s["phase"] == 2]          # noqa: E731
    if ok_run(R["peeled"]) and ok_run(R["dflt"]):
        a, b = R["peeled"], R["dflt"]
        check(bad, "E_id_DEFAULT_vs_the_PRE_B8_trainer_every_logged_row_and_saved_tensor_identical_max_diff_EXACTLY_0.0",
              lambda: strip_volatile(a["rows"]["sft.jsonl"]) == strip_volatile(b["rows"]["sft.jsonl"]) and strip_volatile(a["rows"]["preflight.jsonl"]) == strip_volatile(b["rows"]["preflight.jsonl"])
              and tensors_equal(a["A"], b["A"]) == 0.0 and tensors_equal(a["B"], b["B"]) == 0.0 and strip_volatile(a["pass_json"]["gates"]) == strip_volatile(b["pass_json"]["gates"]))
        check(bad, "E_id_the_default_data_event_has_no_epo_con_carve_block_and_G7_has_no_carve_field", lambda: "epo_con_carve" not in data_of(b) and "carve_logs" not in b["pass_json"]["gates"]["G7_heldout_disjoint"])
        if ok_run(R["expl"]):
            e = R["expl"]
            check(bad, "E_id_EXPLICIT_feed_train_vs_default_identical_max_diff_EXACTLY_0.0", lambda: strip_volatile(e["rows"]["sft.jsonl"]) == strip_volatile(b["rows"]["sft.jsonl"])
                  and tensors_equal(e["A"], b["A"]) == 0.0 and tensors_equal(e["B"], b["B"]) == 0.0)
        check(bad, "E_id_in_the_default_run_the_dual_s_accumulator_is_the_two_training_calls_sum_exactly",
              lambda: len(steps(b)) == 25 and all(s["acc"]["n"] == 2 and s["acc"]["g_ttc"] == 0.0 + t1["parts"]["g_ttc"] + t2["parts"]["g_ttc"] and s["acc"]["g_nc"] == 0.0 + t1["parts"]["g_nc"] + t2["parts"]["g_nc"]
                                                      for s, (t1, t2) in zip(steps(b), zip(trains(b)[0::2], trains(b)[1::2]))))
    if ok_run(R["carve"]):
        k = R["carve"]
        d = data_of(k)
        cl = {"T3", "T7", "T11", "T19", "T21"}
        sha = hashlib.sha256(CARVE_LOGS.encode()).hexdigest()
        check(bad, "E_feed_the_data_event_records_the_BUILT_carve_and_the_training_index_lost_exactly_the_carve_sets", lambda: d.get("epo_con_carve") == {"feed": "carve", "logs": 5, "sets": 20, "sha256": sha, "micro": 1, "train_rows_in_carve_logs": 0,
                                                                                                                         "carve_rows_in_heldout_logs": 0} and d["train_sets"] == 80)
        g7 = k["pass_json"]["gates"]["G7_heldout_disjoint"]
        check(bad, "E_feed_G7_carries_the_carve_disjointness_and_a_real_probe_of_the_feed", lambda: g7["ok"] is True and g7["train_rows_in_carve_logs"] == 0 and g7["carve_rows_in_heldout_logs"] == 0 and g7["carve_sets"] == 20
              and g7["carve_probe_n"] == 1 and all(np.isfinite(v) for v in g7["carve_probe_g"].values()))
        check(bad, "E_feed_every_NO_GRAD_call_scores_carve_log_sets_only_and_every_TRAINING_call_scores_non_carve_sets_only", lambda: len(nograd(k)) == 25 and len(trains(k)) == 50
              and all(set(c["logs"]) <= cl and c["logs"] for c in nograd(k)) and all(not (set(c["logs"]) & cl) and None not in c["logs"] for c in trains(k)))
        check(bad, "E_feed_the_dual_s_accumulator_IS_the_carve_call_n_equals_micro_and_g_exactly_the_no_grad_call_s_g",
              lambda: len(steps(k)) == 25 and all(s["acc"]["n"] == 1 and s["acc"]["g_ttc"] == 0.0 + c["parts"]["g_ttc"] and s["acc"]["g_nc"] == 0.0 + c["parts"]["g_nc"] for s, c in zip(steps(k), nograd(k))))
        check(bad, "E_feed_the_carve_accumulator_is_NOT_the_training_calls_sum_the_feed_changed_something", lambda: any(s["acc"]["g_ttc"] != 0.0 + t1["parts"]["g_ttc"] + t2["parts"]["g_ttc"]
                                                                                                                    for s, (t1, t2) in zip(steps(k), zip(trains(k)[0::2], trains(k)[1::2]))))
        rows = [x for x in k["rows"]["sft.jsonl"] if x.get("event") == "train"]
        check(bad, "E_feed_the_log_carries_both_the_in_batch_g_and_the_carve_g", lambda: rows and all(f in rows[-1] for f in ("con_g_ttc", "carve_g_ttc", "carve_g_nc", "carve_n", "carve_cA_ttc")) and rows[-1]["carve_n"] == 25)
        if ok_run(R["ctrl"]):
            c = R["ctrl"]
            check(bad, "E_attr_the_control_trains_on_the_IDENTICAL_stream_and_its_dual_reads_the_training_calls", lambda: [x["logs"] for x in trains(c)] == [x["logs"] for x in trains(k)] and len(steps(c)) == 25
                  and all(s["acc"]["n"] == 2 and s["acc"]["g_ttc"] == 0.0 + t1["parts"]["g_ttc"] + t2["parts"]["g_ttc"] for s, (t1, t2) in zip(steps(c), zip(trains(c)[0::2], trains(c)[1::2]))))
            check(bad, "E_attr_carve_feed_vs_train_feed_on_the_same_stream_arm_A_bit_identical_arm_B_differs", lambda: tensors_equal(c["A"], k["A"]) == 0.0 and tensors_equal(c["B"], k["B"]) > 0.0)
            lam = lambda r: [x["con_lam_ttc"] for x in r["rows"]["sft.jsonl"] if "con_lam_ttc" in x]          # noqa: E731
            check(bad, "E_attr_the_lambda_trajectory_differs_between_the_carve_feed_and_the_train_feed", lambda: lam(c) and len(lam(c)) == len(lam(k)) and lam(c) != lam(k))
    import carve_logs as CLG
    built = CLG.build([f"T{i}" for i in range(25)], ["H0", "H1", "H2", "H3"], [], m=5)
    R3 = run_smoke(sf_src, f"cf_t_{tag}", EXTRA, carve=built["text"])
    check(bad, "E_tool_a_carve_list_BUILT_BY_carve_logs_py_(SO_CARVE_s_rule)_is_accepted_by_the_real_main_and_its_sha256_is_the_one_the_trainer_recorded",
          lambda: ok_run(R3) and data_of(R3)["epo_con_carve"]["sha256"] == built["sha256"] and data_of(R3)["epo_con_carve"]["logs"] == 5 and data_of(R3)["epo_con_carve"]["train_rows_in_carve_logs"] == 0)
    # refusals through the real main() (no training happens)
    R2 = {"sha": run_smoke(sf_src, f"cf_r1_{tag}", EXTRA, sha="0" * 64), "overlap": run_smoke(sf_src, f"cf_r2_{tag}", EXTRA, carve="T3 H1\n"),
          "unknown": run_smoke(sf_src, f"cf_r3_{tag}", EXTRA, carve="T3 T999\n"), "empty": run_smoke(sf_src, f"cf_r4_{tag}", EXTRA, carve="\n"),
          "tiny": run_smoke(sf_src, f"cf_r5_{tag}", EXTRA + ("--batch", "32"), carve=CARVE_LOGS)}
    check(bad, "E_refuse_a_carve_list_whose_sha256_is_not_the_frozen_one_rc_4_before_any_training", lambda: R2["sha"]["pre"] == 4 and "not the frozen --epo-carve-sha256" in R2["sha"]["out_pre"] and not R2["sha"]["rows"]["sft.jsonl"])
    check(bad, "E_refuse_a_carve_list_that_overlaps_the_RL_val_logs_rc_4_before_any_training", lambda: R2["overlap"]["pre"] == 4 and "RL-val logs" in R2["overlap"]["out_pre"] and not R2["overlap"]["rows"]["sft.jsonl"])
    check(bad, "E_refuse_a_carve_list_naming_logs_that_are_not_RL_train_logs_rc_4", lambda: R2["unknown"]["pre"] == 4 and "not logs of the RL-train index" in R2["unknown"]["out_pre"])
    check(bad, "E_refuse_an_empty_carve_list_and_a_carve_smaller_than_one_batch", lambda: R2["empty"]["pre"] == 4 and "carve list is empty" in R2["empty"]["out_pre"] and R2["tiny"]["pre"] == 4 and "carve sets" in R2["tiny"]["out_pre"])

    # ---------------------------------------------------------------- the hop trace (printed once, from the real source)
    if trace:
        hops = [("1 argparse declares the switch", 'ap.add_argument("--epo-con-feed"'), ("2a main cuts the carve logs out of tr_idx", "tr_idx = [i for i in tr_idx if ds.rows[i].get(\"log_name\") not in carve_logs]"),
                ("2b main builds the carve loader", "cdl = DataLoader(make_train_set(ds, carve_idx, a.epo)"), ("3 the carve stream", "for b_ in cdl:"),
                ("4a carve_feed_acc: no_grad scope", NOGRAD), ("4b carve_feed_acc: the loop's fwd", "_c0, cA_, cB_, ct_ = fwd(cb_)"),
                ("4c carve_feed_acc: the loop's epo_con_term", "epo_con_term(cB_, cA_, ct_, sel_score, dual.temp, a.epo_con_eps, dual.lam, cl_)"),
                ("5 the loop scores the carve before o.step()", "carve_acc = carve_feed_acc() if carve_it is not None else None"), ("6 dual.step takes the carve accumulator", "dual.step(con_acc if carve_acc is None else carve_acc)"),
                ("7 EpoDual.step: g -> lambda", 'raw = self.lam[c] + self.lr * (float(acc["g_" + c]) / n)'), ("8a lambda enters the TRAINING loss", "ct = epo_con_term(sxB, sxA, tg, sel_score, dual.temp, a.epo_con_eps, dual.lam, con_last)"),
                ("8b the term joins arm B's loss", "                lB = lB + ct"), ("8c the loop backwards it", "((lA + lB) / a.accum).backward()"), ("9 G7 re-reads the disjointness + probes the feed", "cp_ = carve_feed_acc()")]
        print("HOP TRACE of --epo-con-feed carve (file:line of the patched trainer)")
        for nm, needle in hops:
            at = sf_src.find(needle)
            ln_ = (sf_src[:at].count(chr(10)) + 1) if at >= 0 else -1
            print(f"  {nm:58s} {SF_REL}:{ln_}")
    return bad


# ================================================================================================ part C: the carve list builder (carve_logs.py)
def suite_c(src: str, tag: str) -> list:
    bad: list = []
    m = types.ModuleType(f"clg_{tag}")
    m.__file__ = CLG_PATH
    sys.modules[m.__name__] = m
    exec(compile(src, CLG_PATH, "exec"), m.__dict__)
    pool = [f"log_{i:03d}" for i in range(166)]
    shuffled = pool[::-1]

    def parity():
        spec = importlib.util.spec_from_file_location("so_carve_analysis_cf", SOCARVE_PATH)
        sc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sc)
        want = sorted(pool[int(i)] for i in sc.carve_draws(len(pool), 13, n=1)[0])
        return m.draw(shuffled, 13) == want and len(want) == 13 and (m.SEED, m.M_PRIMARY) == (sc.SEED, sc.M_PRIMARY) == (20260927, 13)
    check(bad, "C_1_the_draw_IS_SO_CARVE_s_first_carve_draw_over_the_sorted_unique_pool_independent_of_the_input_order", parity)

    def built():
        r = m.build(shuffled, ["V1", "V2"], ["N1"], 13, counts={x: 2 for x in pool})
        r2 = m.build(pool, ["V1", "V2"], ["N1"], 13)
        return (r["text"] == r2["text"] and r["logs"] == r2["logs"] and r["text"].split() == r["logs"] and r["text"].endswith("\n") and r["sha256"] == hashlib.sha256(r["text"].encode()).hexdigest()
                and r["manifest"]["carve_sets"] == 26 and r["manifest"]["pool_sets"] == 332 and len(r["logs"]) == 13 and m.build(pool, ["V1"], [], 13, seed=1)["logs"] != r["logs"])
    check(bad, "C_2_build_is_deterministic_sorted_newline_terminated_with_the_sha256_of_its_bytes_and_a_manifest", built)

    def refuses(fn):
        try:
            fn()
        except ValueError:
            return True
        return False
    check(bad, "C_3_a_pool_that_contains_an_RL_val_log_is_refused", lambda: refuses(lambda: m.build(pool + ["V1"], ["V1"], [], 13)))
    check(bad, "C_4_a_carve_that_contains_a_navtest_log_is_refused", lambda: refuses(lambda: m.build(pool, ["V1"], m.draw(pool, 13), 13)))
    check(bad, "C_5_m_outside_1_to_n_minus_1_is_refused", lambda: refuses(lambda: m.build(pool, [], [], 0)) and refuses(lambda: m.build(pool, [], [], 166)) and not refuses(lambda: m.build(pool, [], [], 165)))

    def cli():
        with tempfile.TemporaryDirectory() as td:
            f = lambda n, t: (open(os.path.join(td, n), "w").write(t), os.path.join(td, n))[1]          # noqa: E731
            tl, vl, out = f("t.txt", " ".join(pool)), f("v.txt", "V1 V2"), os.path.join(td, "carve.txt")
            rc = m.main(["--train-logs", tl, "--rlval-logs", vl, "--out", out])
            body = open(out, "rb").read()
            ok1 = rc == 0 and open(out + ".sha256").read().strip() == hashlib.sha256(body).hexdigest() and json.load(open(out + ".manifest.json"))["sha256"] == hashlib.sha256(body).hexdigest()
            rc2 = m.main(["--train-logs", f("t2.txt", " ".join(pool + ["V1"])), "--rlval-logs", vl, "--out", os.path.join(td, "x.txt")])
            return ok1 and rc2 == 4 and not os.path.exists(os.path.join(td, "x.txt"))
    check(bad, "C_6_the_cli_writes_the_list_its_sha256_and_a_manifest_and_a_refusal_exits_4_writing_nothing", cli)
    return bad


# ================================================================================================ mutations: each edits SOURCE and must turn the named checks RED
STEP = "dual.step(con_acc if carve_acc is None else carve_acc)"
CDL = "cdl = DataLoader(make_train_set(ds, carve_idx, a.epo)"
EXCL = 'tr_idx = [i for i in tr_idx if ds.rows[i].get("log_name") not in carve_logs]'
NOGRAD = "        with torch.no_grad():\n            for _ in range(a.epo_carve_micro):"
GUARD = "                if stray:\n"
MUTATIONS = [
    # (name, [(old, new)], expected RED checks)
    ("M1 the feed is ignored: dual.step(con_acc) again", [(STEP, "dual.step(con_acc)")],
     ["E_feed_the_dual_s_accumulator_IS_the_carve_call_n_equals_micro_and_g_exactly_the_no_grad_call_s_g", "E_attr_the_lambda_trajectory_differs_between_the_carve_feed_and_the_train_feed", "R_hop_6_dual_step_is_fed_the_carve_accumulator_when_the_switch_is_on_and_con_acc_otherwise"]),
    ("M2a the carve loader reads TRAINING sets (the stray-row guard stays in)", [(CDL, "cdl = DataLoader(make_train_set(ds, tr_idx, a.epo)")],
     ["E_carve_the_real_main_runs_preflight_PASS_then_trains", "R_hop_2_main_cuts_the_carve_logs_out_of_tr_idx_and_builds_the_carve_loader_from_carve_idx_only_under_the_switch"]),
    ("M2b the carve loader reads TRAINING sets AND the stray-row guard is removed", [(CDL, "cdl = DataLoader(make_train_set(ds, tr_idx, a.epo)"), (GUARD, "                if False:\n")],
     ["E_feed_every_NO_GRAD_call_scores_carve_log_sets_only_and_every_TRAINING_call_scores_non_carve_sets_only", "U_4_a_carve_micro_batch_row_outside_the_carve_logs_raises"]),
    ("M3 the carve logs are NOT removed from the training index (G7 must FAIL the preflight)", [(EXCL, "tr_idx = tr_idx")], ["E_carve_the_real_main_runs_preflight_PASS_then_trains"]),
    ("M3b ... AND G7's carve disjointness term removed (the run trains on the carve; the per-call log census sees it)", [(EXCL, "tr_idx = tr_idx"), (" and carve_train_in == 0 and carve_in_ho == 0", "")],
     ["E_feed_every_NO_GRAD_call_scores_carve_log_sets_only_and_every_TRAINING_call_scores_non_carve_sets_only"]),
    ("M4 the carve forward keeps its graph (no torch.no_grad)", [(NOGRAD, "        if True:\n            for _ in range(a.epo_carve_micro):")],
     ["U_1_the_carve_accumulator_carries_NO_graph_fwd_runs_with_grad_disabled_no_param_grad_floats_only_and_equals_an_independent_epo_con_term_call", "R_hop_4_carve_feed_acc_scores_fwd_and_epo_con_term_inside_no_grad_with_the_SAME_rule_T_eps_lam_as_the_training_call"]),
    ("M5 the default flipped to carve", [('EPO_CON_FEED_DEFAULT = "train" ', 'EPO_CON_FEED_DEFAULT = "carve" ')],
     ["P_defaults_are_train_empty_empty_one_and_named_constants", "E_dflt_the_real_main_runs_preflight_PASS_then_trains"]),
    ("M6 the RL-val overlap refusal removed", [("        if carve_logs & ho_logs:\n", "        if False:\n")], ["E_refuse_a_carve_list_that_overlaps_the_RL_val_logs_rc_4_before_any_training"]),
    ("M7 the frozen-sha256 check removed", [("        if carve_sha != a.epo_carve_sha256:\n", "        if False:\n")], ["E_refuse_a_carve_list_whose_sha256_is_not_the_frozen_one_rc_4_before_any_training"]),
    ("M8 --epo-carve-micro ignored (always one micro-batch)", [("            for _ in range(a.epo_carve_micro):", "            for _ in range(1):")], ["U_3_epo_carve_micro_3_scores_three_micro_batches_and_sums_them_in_order"]),
    ("M9 the carve levers are not wired into epo_con_refusals (the gate and main never see them)", [("    bad += epo_carve_refusals(a, ap)                  # B8: the carve feed's own levers (all conditions are inside)\n", "")],
     ["U_refusal_carve_levers_named_while_the_feed_is_train_or_off_are_refused", "U_refusal_carve_without_logs_or_with_a_malformed_sha_or_a_bad_micro_is_refused"]),
    ("M10 the unknown-log refusal removed (a typo'd carve silently shrinks)", [("        if carve_logs - carve_pool - ho_logs:\n", "        if False:\n")], ["E_refuse_a_carve_list_naming_logs_that_are_not_RL_train_logs_rc_4"]),
]


def apply_edits(src: str, edits, name: str) -> str:
    for old, new in edits:
        assert src.count(old) == 1, f"mutation {name}: the snippet occurs {src.count(old)} times (must be exactly 1): {old[:70]!r}"
        src = src.replace(old, new)
    return src


CLG_MUTATIONS = [
    ("C-M1 the draw's seed changed", [("SEED, M_PRIMARY = 20260927, 13", "SEED, M_PRIMARY = 20260928, 13")], ["C_1_the_draw_IS_SO_CARVE_s_first_carve_draw_over_the_sorted_unique_pool_independent_of_the_input_order"]),
    ("C-M2 the pool is not sorted before the draw (input order leaks in)", [("    logs = sorted(set(pool_logs))\n", "    logs = list(dict.fromkeys(pool_logs))\n")], ["C_1_the_draw_IS_SO_CARVE_s_first_carve_draw_over_the_sorted_unique_pool_independent_of_the_input_order"]),
    ("C-M3 the RL-val pool overlap refusal removed", [("    if pool & val:\n", "    if False:\n")], ["C_3_a_pool_that_contains_an_RL_val_log_is_refused"]),
    ("C-M4 the navtest refusal removed", [("    if set(carve) & nav:\n", "    if False:\n")], ["C_4_a_carve_that_contains_a_navtest_log_is_refused"]),
]


def main() -> int:
    src = open(SF_PATH, encoding="utf-8").read()
    clg_src = open(CLG_PATH, encoding="utf-8").read()
    assert S1B.git_blob(CL.peel(src).encode("utf-8")) == CL.TIP_TRAINER_BLOB, "peeling the B8 layer does not give the 821915a trainer: regenerate CARVEFEED_FILE_REVERSALS (gen_layer.py)"
    refs = build_refs(src)
    assert ok_run(refs["peeled"]), f"the PRE-B8 trainer's smoke run failed: {refs['peeled'].get('exc')}"
    real = suite(src, "real", refs, trace=True)
    assert not real, f"the unmutated source FAILS its own checks: {real}"
    real_c = suite_c(clg_src, "real")
    assert not real_c, f"the unmutated carve_logs.py FAILS its own checks: {real_c}"
    n, problems = 2, []
    print(f"{'mutation':100s} caught-by (failing checks)")
    for name, edits, expect in CLG_MUTATIONS:
        failing = suite_c(apply_edits(clg_src, edits, name), "m")
        if not failing or any(c not in failing for c in expect):
            problems.append(f"mutation {name}: must turn {expect} RED, got {failing}")
            continue
        print(f"{name[:100]:100s} {', '.join(failing[:3])}", flush=True)
        n += 1
    for i, (name, edits, expect) in enumerate(MUTATIONS):
        m = apply_edits(src, edits, name)
        try:
            _ast.parse(m)
            failing = suite(m, f"m{i}", refs)
        except (Exception, SystemExit) as e:                            # noqa: BLE001 -- the mutated source breaks the suite's own setup: RED, and said so
            failing = [f"suite setup raised {type(e).__name__}: {str(e)[:80]}"]
        if not failing:
            problems.append(f"mutation {name} stayed GREEN: the suite cannot see it")
            continue
        for c in expect:
            if c not in failing and not any(f.startswith("suite setup raised") for f in failing):
                problems.append(f"mutation {name}: the check {c} must go RED and stayed green (RED: {failing[:8]})")
        behavioural = [f for f in failing if not f.startswith("P_")]
        print(f"{name[:100]:100s} {', '.join((behavioural or failing)[:3])}" + ("" if behavioural else "   [PIN ONLY]"), flush=True)
        n += 1
    if problems:
        print("\n".join("PROBLEM " + p_ for p_ in problems))
        print(f"ZZTEST_FAIL {len(problems)} problem(s)")
        return 1
    print(f"ZZTEST_PASS {n}  (2 unmutated suites + {len(MUTATIONS) + len(CLG_MUTATIONS)} RED mutations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
