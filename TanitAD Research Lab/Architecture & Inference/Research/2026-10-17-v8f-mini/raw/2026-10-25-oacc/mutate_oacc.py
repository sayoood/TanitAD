"""Mutation proof for PREREG_V8F_OACC's instruments: every mutation must turn stack/tests/test_v8f_oacc.py RED, by an ASSERTION; the unmutated tree must be GREEN before and after, and every file is
restored byte-identical.

    python mutate_oacc.py <repo root> <python exe> [--check] [--list] [--only <substring of a mutant's name or file>] [--from N --to M] [--idx I,J,K] [--no-baseline] [--tmp <dir>]

    --check   only verify that every anchor occurs exactly once (in the ORIGINAL bytes), then exit
    --only    run only the mutants whose name or file contains the substring (the baseline and the restore still run)
    --from/--to  run only mutants N .. M-1 of `--list` (the baseline and the restore still run): the harness is long, so it can be sharded -- each shard on its OWN copy of the tree
    --idx     run exactly these mutants (a comma list of `--list` indices): how the second proof shards the list into cost-balanced workers, each ONE process with ONE baseline and ONE restore
    --no-baseline  DEVELOPMENT ONLY: skip the baseline run (the banked proof never uses it: its baseline and restore are part of the claim; the banner it prints says so)
    --tmp     the pytest basetemp parent (default C:/Users/Admin/pt_oacc_mut; a SHORT path: the queue tests start `cmd /c` on paths under it and cmd refuses > 260 characters)

The mutants live in the model (the term, its second difference, the origin knot, the GT, the mask, the line that adds it to the total, the weight, the config, the built lever), the trainer (the flag, make_config,
G-DVB, G-LIVE's expected terms, the acc probe and every one of its checks, the refusal), the rule / fit script (the 0.5, the ratio, the rounding, the cross-check, the refusals), the reader (the committed bars,
the outcome table, the conditional seed 1, the dominance row, every refusal of check_x_run), the queue (CRLF, edited as BYTES) and the chain (the predecessor predicate included, the s1 branch, every assertion
of its gate / rule blocks). Each mutation is one (or, where stated, two) exact substring(s) replaced once; a mutation whose anchor is not found exactly once aborts the run.

RED BY AN ASSERTION: the first failure of a RED run is read: `E   assert ...`, `AssertionError` and `Failed: DID NOT RAISE` count as assertions. A first failure that is any other exception (a TypeError, a
KeyError, a ZeroDivisionError) is reported as `RED(exception: <type>)` and the group is re-run WITHOUT -x to see whether any test fails by assertion too; a mutant that is RED by exception ONLY is listed in the
summary line, and a collection / syntax error is never accepted as a RED.

THE DISK: runs on ONE copy of the tree per shard, with its pytest temp dir wiped before every run, and every synthetic world is removed by the tests that make it.

SPEED: a mutant is first run against the group of tests that own its file (`-k`), with `-x`; a RED there is a RED of the suite. A mutant that survives its group is re-run against the WHOLE file before it is
called SURVIVED, so a too-narrow selector can never hide a live mutant, only cost time.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
PY = sys.argv[2]
CHECK = "--check" in sys.argv
LIST = "--list" in sys.argv
ONLY = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
FROM = int(sys.argv[sys.argv.index("--from") + 1]) if "--from" in sys.argv else 0
TO = int(sys.argv[sys.argv.index("--to") + 1]) if "--to" in sys.argv else 10 ** 6
IDX = {int(x) for x in sys.argv[sys.argv.index("--idx") + 1].split(",")} if "--idx" in sys.argv else None
NO_BASE = "--no-baseline" in sys.argv
TMP = Path(sys.argv[sys.argv.index("--tmp") + 1]) if "--tmp" in sys.argv else Path("C:/Users/Admin/pt_oacc_mut")
PKG = ROOT / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
CODE = PKG / "code"
MODEL = ROOT / "stack" / "tanitad" / "models" / "v8f_mini.py"
TRAIN = ROOT / "stack" / "scripts" / "v8f_mini_train.py"
EVAL = ROOT / "stack" / "scripts" / "v8f_mini_eval_oacc.py"
READER = CODE / "v8f_oacc_read.py"
QUEUE = CODE / "v8f_mini_box2_queue.cmd"
CHAIN = CODE / "box2_ops" / "chain_v8f_oacc.sh"
TEST = "stack/tests/test_v8f_oacc.py"

_RD = "outcome_partition or next_texts or dominance_row or dom_row or dk_flags or keep_row or seed_pass or check_x_run or end_to_end or check_inputs_reads or oacc_reader"
_QU = "queue_oacc or guard_pattern"
_CH = "chain_oacc or predecessor or wait_loop or every_stop"
_RL = "rule_ or offline_route or model_route or hx_fit"
SEL = {
    "model": f"not ({_RD} or {_QU} or {_CH} or {_RL})",
    "rule": _RL,
    "reader": _RD,
    "queue": _QU,
    "chain": _CH,
}
NL = "\r\n"          # the queue is CRLF: its multi-line anchors say so


def M(path, name, a, b, group):
    return (path, name, [(a, b)], group)


def MM(path, name, edits, group):
    return (path, name, edits, group)


def off(path, name, anchor, group, repl="False"):
    """`<indent>if <cond>:` -> `<indent>if False:`  (the anchor is the whole condition text, from `if` to the colon)."""
    kw = "elif" if anchor.startswith("elif") else "if"
    return M(path, name, anchor, f"{kw} {repl}:", group)


TERM = 'T["acc"] = accel_term(out["traj30"], gt, gok)'
LINE = 'total = total + cfg.w_acc * T["acc"] if cfg.w_acc > 0.0 else total'
_E0 = 'set "E0="%PY%" "%EVAL%" run --ckpt "%OUT%\\ckpt_last.pt" --tag %TG% %EC% %ER% --project --families"'
_TG = 'set "TG=M2b-oacc_s%~2"'

MUT = [
    # =========================================================================================================== the model: the term, the line that adds it, the weight, the config, the built lever
    M(MODEL, "the line that adds the term to the total is deleted", LINE, "total = total", "model"),
    M(MODEL, "the weight is ignored (the term enters at 1.0)", 'total = total + cfg.w_acc * T["acc"] if', 'total = total + T["acc"] if', "model"),
    M(MODEL, "the weight is applied twice", 'total = total + cfg.w_acc * T["acc"] if', 'total = total + cfg.w_acc * cfg.w_acc * T["acc"] if', "model"),
    M(MODEL, "the term enters detached (no gradient)", 'total = total + cfg.w_acc * T["acc"] if', 'total = total + cfg.w_acc * T["acc"].detach() if', "model"),
    M(MODEL, "the term never enters (the condition is inverted)", "if cfg.w_acc > 0.0 else total", "if cfg.w_acc < 0.0 else total", "model"),
    M(MODEL, "the term enters with the opposite sign", 'total = total + cfg.w_acc * T["acc"] if', 'total = total - cfg.w_acc * T["acc"] if', "model"),
    M(MODEL, "the term is computed on a detached plan", TERM, 'T["acc"] = accel_term(out["traj30"].detach(), gt, gok)', "model"),
    M(MODEL, "the term is computed against the prior's plan, not the GT", TERM, 'T["acc"] = accel_term(out["traj30"], out["traj30"].detach() * 0.0, gok)', "model"),
    M(MODEL, "the second difference is a first difference", "return P[:, 2:] - 2.0 * P[:, 1:-1] + P[:, :-2]", "return P[:, 1:] - P[:, :-1]", "model"),
    M(MODEL, "the second difference has the wrong centre coefficient", "return P[:, 2:] - 2.0 * P[:, 1:-1] + P[:, :-2]", "return P[:, 2:] - 1.0 * P[:, 1:-1] + P[:, :-2]", "model"),
    M(MODEL, "the origin knot is not prepended", "P = torch.cat([xy.new_zeros(xy.shape[0], 1, 2), xy], 1)", "P = xy", "model"),
    M(MODEL, "the origin knot is prepended at the wrong place (a constant, not the ego at NOW)", "P = torch.cat([xy.new_zeros(xy.shape[0], 1, 2), xy], 1)", "P = torch.cat([xy[:, :1], xy], 1)", "model"),
    M(MODEL, "the GT's second difference is dropped", "d2 = second_diff_02(traj30) - second_diff_02(gt.to(traj30.dtype))", "d2 = second_diff_02(traj30)", "model"),
    M(MODEL, "the GT's second difference is added, not subtracted", "d2 = second_diff_02(traj30) - second_diff_02(gt.to(traj30.dtype))", "d2 = second_diff_02(traj30) + second_diff_02(gt.to(traj30.dtype))", "model"),
    M(MODEL, "the GT-valid mask is dropped", "return (sq.mean(1) * g).sum() / g.sum().clamp_min(1.0)", "return sq.mean(1).mean()", "model"),
    M(MODEL, "the mean over the GT-valid windows is a sum", "return (sq.mean(1) * g).sum() / g.sum().clamp_min(1.0)", "return (sq.mean(1) * g).sum()", "model"),
    M(MODEL, "the mean over the 29 centres is a sum", "return (sq.mean(1) * g).sum() / g.sum().clamp_min(1.0)", "return (sq.sum(1) * g).sum() / g.sum().clamp_min(1.0)", "model"),
    M(MODEL, "the norm over the two axes is a mean", "sq = (d2 ** 2).sum(-1) ", "sq = (d2 ** 2).mean(-1) ", "model"),
    M(MODEL, "the term is L1, not squared", "sq = (d2 ** 2).sum(-1) ", "sq = d2.abs().sum(-1) ", "model"),
    M(MODEL, "the ADE / FDE spelling changed (B's loss is no longer B's)", "d = torch.sqrt(((traj30 - gt) ** 2).sum(-1) + 1e-9)", "d = torch.sqrt(((traj30 - gt) ** 2).sum(-1) + 1e-6)", "model"),
    M(MODEL, "the FDE is taken at the wrong knot", "(d[:, -1] * gok).sum()", "(d[:, -2] * gok).sum()", "model"),
    M(MODEL, "built_levers does not record w_acc", '"w_acc": cfg.w_acc,', "", "model"),
    M(MODEL, "built_levers echoes 0.0 whatever was built", '"w_acc": cfg.w_acc,', '"w_acc": 0.0,', "model"),
    off(MODEL, "a negative or non-finite weight is accepted by the config", "if not (math.isfinite(self.w_acc) and self.w_acc >= 0.0):", "model"),
    M(MODEL, "the config default is 0.5", "    w_acc: float = 0.0 ", "    w_acc: float = 0.5 ", "model"),
    # =========================================================================================================== the trainer: flag, make_config, G-DVB, G-LIVE, the acc probe
    M(TRAIN, "the flag has no declared-vs-built entry", '"w_acc": "cfg:w_acc",', "", "model"),
    M(TRAIN, "make_config drops the lever", "o_interface=a.o_interface, w_acc=a.w_acc)", "o_interface=a.o_interface)", "model"),
    off(TRAIN, "declared-vs-built does not compare the built lever", 'if built.get("w_acc") != a.w_acc:', "model"),
    M(TRAIN, "declared-vs-built compares the lever with itself", 'if built.get("w_acc") != a.w_acc:', 'if built.get("w_acc") != built.get("w_acc"):', "model"),
    off(TRAIN, "declared-vs-built does not refuse a negative weight", "if not (math.isfinite(a.w_acc) and a.w_acc >= 0.0):", "model"),
    M(TRAIN, "the flag's default is 0.5", 'ap.add_argument("--w-acc", dest="w_acc", type=float, default=0.0,', 'ap.add_argument("--w-acc", dest="w_acc", type=float, default=0.5,', "model"),
    M(TRAIN, "the flag is stored under another name", 'ap.add_argument("--w-acc", dest="w_acc", type=float', 'ap.add_argument("--w-acc", dest="w_acc_x", type=float', "model"),
    MM(TRAIN, "declared-vs-built compares the lever neither in the lever check nor in the cfg check",
       [('if built.get("w_acc") != a.w_acc:', "if False:"), ('"w_pen_self", "w_wm", "detach_upper", "kin_mode", "w_acc"):', '"w_pen_self", "w_wm", "detach_upper", "kin_mode"):')], "model"),
    M(TRAIN, "live_check does not expect the acc term",
      'opt.step()\n    expect = {"ckpt", "goal", "lat", "lon", "cons", "ade", "fde", "acc", "total"}', 'opt.step()\n    expect = {"ckpt", "goal", "lat", "lon", "cons", "ade", "fde", "total"}', "model"),
    M(TRAIN, "live_check_lora does not expect the acc term",
      'enc.load_lora_state_dict(sd0)\n        for p in lora:\n            p.grad = None\n    expect = {"ckpt", "goal", "lat", "lon", "cons", "ade", "fde", "acc", "total"}',
      'enc.load_lora_state_dict(sd0)\n        for p in lora:\n            p.grad = None\n    expect = {"ckpt", "goal", "lat", "lon", "cons", "ade", "fde", "total"}', "model"),
    M(TRAIN, "the acc probe is faked (never reads the built loss)", "accp = acc_probe(model, probe, cfg)",
      'accp = {"w_acc_declared": cfg.w_acc, "w_probe": cfg.w_acc, "acc": 1.0, "acc_at_zero_weight": 1.0, "contribution": 0.0, "w_x_acc": 0.0, "head_o_grad_diff": 0.0, "read_o_grad_diff": 0.0, '
      '"control_repeat_total_diff": 0.0, "control_repeat_grad_diff": 0.0, "control_prior_grad_diff": 0.0, "control_wm_grad_diff": 0.0, "pass": True}', "model"),
    M(TRAIN, "the acc probe's refusal does not stop the run",
      'except AssertionError as e:\n        raise SystemExit(f"ZZDVB-REFUSEDZZ declared-vs-built: {e}")\n    adm = M.assert_no_situation_classifier_input',
      'except AssertionError as e:\n        accp = {"pass": False, "error": str(e)}\n    adm = M.assert_no_situation_classifier_input', "model"),
    M(TRAIN, "selfcheck.json omits the acc probe", '"acc_probe": accp, ', "", "model"),
    M(TRAIN, "the acc probe runs at the wrong weight (not the declared one)", "w = w_decl if w_decl > 0.0 else 1.0", "w = 1.0", "model"),
    off(TRAIN, "the acc probe does not require a positive term", "if not math.isfinite(acc_w) or not acc_w > 0.0:", "model"),
    off(TRAIN, "the acc probe does not require the logged term to be weight-independent", "if acc_w != acc_0:", "model"),
    off(TRAIN, "the acc probe does not require its own determinism", "if res[\"control_repeat_total_diff\"] != 0.0 or res[\"control_repeat_grad_diff\"] != 0.0:", "model"),
    off(TRAIN, "the acc probe does not require float32 visibility", "if not want > 64.0 * eps32 * max(abs(t_0), 1.0):", "model"),
    off(TRAIN, "the acc probe does not check the total moved by w x the term", "if abs(contribution - want) > rtol * want + 4.0 * eps32 * max(abs(t_w), abs(t_0)):", "model"),
    off(TRAIN, "the acc probe does not require a gradient in O's head", 'if not res["head_o_grad_diff"] > 0.0:', "model"),
    off(TRAIN, "the acc probe does not check the frozen prior's control", 'if res["control_prior_grad_diff"] != 0.0:', "model"),
    off(TRAIN, "the acc probe does not check the world-model control", 'if res["control_wm_grad_diff"] not in (None, 0.0):', "model"),
    M(TRAIN, "the acc probe measures the contribution on one total only", "contribution = t_w - t_0", "contribution = t_w - t_w", "model"),
    M(TRAIN, "the acc probe compares against the wrong product", "want = w * acc_w", "want = acc_w", "model"),
    M(TRAIN, "the acc probe's head gradient reads the prior's block", 'dsum(g_w, g_0, "head_o.")', 'dsum(g_w, g_0, "head_o_none.")', "model"),
    M(TRAIN, "the acc probe's prior control reads nothing", 'float((g_w["__prior__"] - g_0["__prior__"]).abs().sum())', "0.0", "model"),
    # =========================================================================================================== the rule / fit script
    M(EVAL, "the rule's fraction is 1.0", "FRAC = 0.5 ", "FRAC = 1.0 ", "rule"),
    M(EVAL, "the rule's ratio is inverted", "raw = frac * L_traj / L_acc", "raw = frac * L_acc / L_traj", "rule"),
    M(EVAL, "the rule rounds to 3 digits", "SIG_DIGITS = 6 ", "SIG_DIGITS = 3 ", "rule"),
    M(EVAL, "the rule does not round", "return float(f\"{x:.{n}g}\")", "return x", "rule"),
    M(EVAL, "L_traj has no FDE term", "float(w_traj) * (ade / 3.0 + 0.5 * fde / 6.0)", "float(w_traj) * (ade / 3.0)", "rule"),
    M(EVAL, "L_traj weighs the FDE at 1.0", "float(w_traj) * (ade / 3.0 + 0.5 * fde / 6.0)", "float(w_traj) * (ade / 3.0 + fde / 6.0)", "rule"),
    M(EVAL, "L_traj ignores w_traj", "float(w_traj) * (ade / 3.0 + 0.5 * fde / 6.0)", "(ade / 3.0 + 0.5 * fde / 6.0)", "rule"),
    M(EVAL, "the rule's L_acc is the first difference's", "acc = M.accel_term(t, g, k)", "acc = float(((t[:, 1:] - t[:, :-1]) - (g[:, 1:] - g[:, :-1])).pow(2).sum(-1).mean())", "rule"),
    off(EVAL, "the rule divides by anything", "if not (math.isfinite(L_traj) and math.isfinite(L_acc) and L_traj > 0.0 and L_acc > 0.0):", "rule"),
    M(EVAL, "the cross-check tolerance is 1.0", "CROSS_RTOL = 1e-3 ", "CROSS_RTOL = 1.0 ", "rule"),
    M(EVAL, "the cross-check ignores the window identity",
      'rec["banked_crosscheck"]["pass"] = bool(max(rel.values()) <= CROSS_RTOL and rec["banked_crosscheck"]["same_windows"])', 'rec["banked_crosscheck"]["pass"] = bool(max(rel.values()) <= CROSS_RTOL)', "rule"),
    off(EVAL, "a failed cross-check does not refuse", "if not rec[\"banked_crosscheck\"][\"pass\"]:", "rule"),
    off(EVAL, "the rule accepts an X checkpoint", "if float(cfg.w_acc) != 0.0:", "rule"),
    off(EVAL, "the checkpoint's sha256 is not checked", "if got != expect_sha:", "rule"),
    off(EVAL, "a fit window without a 6 s future is accepted", 'if not O["gt_ok"].all():', "rule"),
    M(EVAL, "the HX fit uses the model's plan as the prior", 'R = H.fit_weights(O["traj30"], O["p30"], O["gt"])', 'R = H.fit_weights(O["traj30"], O["traj30"], O["gt"])', "rule"),
    M(EVAL, "the HX fit scores against the wrong target", 'R = H.fit_weights(O["traj30"], O["p30"], O["gt"])', 'R = H.fit_weights(O["traj30"], O["p30"], O["p30"])', "rule"),
    M(EVAL, "the HX record does not carry the checkpoint's sha256", '"ckpt_sha256": sha256_file(a.ckpt), "w_acc_built"', '"ckpt_sha256": "0" * 64, "w_acc_built"', "rule"),
    M(EVAL, "the HX record does not carry the built weight", '"w_acc_built": float(cfg.w_acc)', '"w_acc_built": 0.0', "rule"),
    M(EVAL, "w_acc.txt is not the repr of the number", "f.write(f\"{rec['w_acc']!r}\\n\")", "f.write(f\"{rec['w_acc']:.12f}\\n\")", "rule"),
    # =========================================================================================================== the reader: bars, outcome table, conditional seed 1, dominance, every refusal
    M(READER, "the KEEP bar is F, not F/2", '"bar_hi_le": F / 2, "pass": bool(r["hi"] <= F / 2)', '"bar_hi_le": F, "pass": bool(r["hi"] <= F)', "reader"),
    M(READER, "the KEEP bar is the printed 0.060", '"bar_hi_le": F / 2, "pass": bool(r["hi"] <= F / 2)', '"bar_hi_le": PRINTED_KEEP_BAR, "pass": bool(r["hi"] <= PRINTED_KEEP_BAR)', "reader"),
    M(READER, "KEEP reads the point, not the upper bound", '"pass": bool(r["hi"] <= F / 2)', '"pass": bool(r["mean"] <= F / 2)', "reader"),
    M(READER, "the seed's KEEP reads the point", 'bool(keep["hi"] <= keep["bar_hi_le"])', 'bool(keep["mean"] <= keep["bar_hi_le"])', "reader"),
    M(READER, "the seed's KEEP is strict", 'bool(keep["hi"] <= keep["bar_hi_le"])', 'bool(keep["hi"] < keep["bar_hi_le"])', "reader"),
    M(READER, "ACC needs a positive difference", 'acc["mean"] < 0 and acc["separated"]', 'acc["mean"] > 0 and acc["separated"]', "reader"),
    M(READER, "ACC needs no separation", 'acc["mean"] < 0 and acc["separated"]', 'acc["mean"] < 0', "reader"),
    M(READER, "ACC passes at zero", 'acc["mean"] < 0 and acc["separated"]', 'acc["mean"] <= 0 and acc["separated"]', "reader"),
    M(READER, "the DOMINANCE flag of the seed is not read", '"DOM": bool(dom["pass"])', '"DOM": True', "reader"),
    M(READER, "the outcome: NO EFFECT reads COSTS POSITION", 'if not all(v["ACC"] for v in states.values()):\n        return "NO EFFECT", False', 'if not all(v["ACC"] for v in states.values()):\n        return "COSTS POSITION", False', "reader"),
    M(READER, "the outcome: COSTS POSITION reads NO EFFECT", 'if not all(v["KEEP"] for v in states.values()):\n        return "COSTS POSITION", False', 'if not all(v["KEEP"] for v in states.values()):\n        return "NO EFFECT", False', "reader"),
    M(READER, "the outcome: ACC at ONE seed is enough", 'if not all(v["ACC"] for v in states.values()):', 'if not any(v["ACC"] for v in states.values()):', "reader"),
    M(READER, "the outcome: KEEP at ONE seed is enough", 'if not all(v["KEEP"] for v in states.values()):', 'if not any(v["KEEP"] for v in states.values()):', "reader"),
    M(READER, "the outcome: seed 0 alone is an outcome, not pending", 'return "PENDING SEED 1", True', 'return "PENDING SEED 1", False', "reader"),
    M(READER, "the outcome: seed 0 alone reads SMOOTHER", 'return "PENDING SEED 1", True', 'return "SMOOTHER, NOT DOMINANT", False', "reader"),
    off(READER, "the outcome: a seed-0-only read is not pending", "if sorted(states) == [0]:", "reader"),
    M(READER, "the outcome: DOMINATES needs DOMINANCE at one seed only", 'if all(v["DOM"] for v in states.values()):', 'if any(v["DOM"] for v in states.values()):', "reader"),
    M(READER, "the outcome: DOMINATES and SMOOTHER swapped", 'return "DOMINATES ITS PRIOR", False\n    return "SMOOTHER, NOT DOMINANT", False', 'return "SMOOTHER, NOT DOMINANT", False\n    return "DOMINATES ITS PRIOR", False', "reader"),
    off(READER, "the outcome accepts any seed set", "if sorted(states) not in ([0], [0, 1]):", "reader"),
    off(READER, "the outcome accepts a non-boolean state", 'if set(v) != {"ACC", "KEEP", "DOM"} or not all(isinstance(x, bool) for x in v.values()):', "reader"),
    M(READER, "NEXT_TEXT: DOMINATES loses its caveat", "never as state of the art. Next: the same row against FLAT", "Next: the same row against FLAT", "reader"),
    M(READER, "NEXT_TEXT: COSTS POSITION allows a re-tune", "the weight rule is not re-tuned post hoc", "the weight rule is re-tuned", "reader"),
    M(READER, "NEXT_TEXT: NO EFFECT names another next step", "next, an O-capacity read", "next, a bigger weight", "reader"),
    M(READER, "NEXT_TEXT: SMOOTHER names another next step", "next, the longitudinal decision (`PREREG_V8F_KHYP`)", "next, the lateral decision", "reader"),
    M(READER, "DOM_BETTER lacks the accel row", '"accel MAE": "fw_lon_accel_mae", ', "", "reader"),
    M(READER, "DOM_BETTER lacks the curvature row", ', "curvature MAE": "fw_lat_curvature_mae_1pm"}', "}", "reader"),
    M(READER, "DOM_NOT_WORSE lacks the yaw-rate row", '"yaw-rate MAE": "fw_lat_yaw_rate_mae_degps", ', "", "reader"),
    M(READER, "DOM_NOT_WORSE lacks the longitudinal decision agreement", ',\n                 "plan-derived longitudinal decision agreement": "fw_tac_lon_acc"}', "}", "reader"),
    M(READER, "dom_row: the better rows need only not-worse", 'ok_better &= bool(r["better"])', 'ok_better &= not bool(r["worse"])', "reader"),
    M(READER, "dom_row: the not-worse rows are not checked", '"need": "not separated-worse"}\n        ok_nw &= not bool(r["worse"])\n    r = cons_viol_any', '"need": "not separated-worse"}\n        ok_nw &= True\n    r = cons_viol_any', "reader"),
    M(READER, "dom_row: consistency is not checked", 'rows["CONSISTENCY(a) \'any\'"] = {"mean": r["mean"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"], "better": r["better"], "worse": r["worse"], "need": "not separated-worse"}\n    ok_nw &= not bool(r["worse"])',
      'rows["CONSISTENCY(a) \'any\'"] = {"mean": r["mean"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"], "better": r["better"], "worse": r["worse"], "need": "not separated-worse"}\n    ok_nw &= True', "reader"),
    M(READER, "dom_row: distance keeping is not checked", "ok_nw &= not f[\"worse\"]", "ok_nw &= True", "reader"),
    M(READER, "dom_row: the pass ignores the not-worse rows", 'bool(ok_better and ok_nw)', 'bool(ok_better)', "reader"),
    M(READER, "dom_row: the pass ignores the better rows", 'bool(ok_better and ok_nw)', 'bool(ok_nw)', "reader"),
    M(READER, "dk_flags: headway is higher-is-worse", '("headway min (m)", dk_pair["dk"]["headway_min_m"], False)', '("headway min (m)", dk_pair["dk"]["headway_min_m"], True)', "reader"),
    M(READER, "dk_flags: the at-risk share is lower-is-worse", '("LEAD-VIOL at-risk share", dk_pair["lead_viol"]["at_risk"], True)', '("LEAD-VIOL at-risk share", dk_pair["lead_viol"]["at_risk"], False)', "reader"),
    M(READER, "dk_flags: worse needs no separation", '"worse": bool(sep and ((d > 0) if hi_is_worse else (d < 0)))', '"worse": bool(((d > 0) if hi_is_worse else (d < 0)))', "reader"),
    M(READER, "the lower-is-better table lacks the tactical agreements", 'LOWER_IS_BETTER = {"tac_lat_acc": False, "tac_lon_acc": False}', "LOWER_IS_BETTER = {}", "reader"),
    M(READER, "the bootstrap B defaults to 1,000", "B_BOOT = 2000", "B_BOOT = 1000", "reader"),
    M(READER, "the committed window count is 2,400", "EVAL_WINDOWS, EVAL_EPISODES = 2399, 141", "EVAL_WINDOWS, EVAL_EPISODES = 2400, 141", "reader"),
    M(READER, "F may be anything within 0.05", "F_PRINTED, F_TOL = 0.1207, 5e-4", "F_PRINTED, F_TOL = 0.1207, 5e-2", "reader"),
    M(READER, "P10's ADE6 may be anything within 0.5", "P10_ADE6, P10_TOL = 2.4870, 1e-3", "P10_ADE6, P10_TOL = 2.4870, 5e-1", "reader"),
    M(READER, "the banked s0 checkpoint sha256 is wrong in the reader", '"M2b_s0": "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8"', '"M2b_s0": "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f9"', "reader"),
    off(READER, "F is not checked", "if abs(F - F_PRINTED) > F_TOL:", "reader"),
    off(READER, "P10's ADE6 is not checked", "if abs(p10_ade6 - P10_ADE6) > P10_TOL:", "reader"),
    M(READER, "the eval grid size is not checked in check-inputs", 'if (n, ne) != (a.n_windows, a.n_episodes):\n        refuse(f"the eval grid holds', 'if False:\n        refuse(f"the eval grid holds', "reader"),
    M(READER, "the paired ACC row is B minus X", 'acc_row = BE.paired_row(pwX["fw_lon_accel_mae"], pwB["fw_lon_accel_mae"], eid, nb)', 'acc_row = BE.paired_row(pwB["fw_lon_accel_mae"], pwX["fw_lon_accel_mae"], eid, nb)', "reader"),
    M(READER, "the ACC row reads the speed MAE", 'acc_row = BE.paired_row(pwX["fw_lon_accel_mae"], pwB["fw_lon_accel_mae"], eid, nb)', 'acc_row = BE.paired_row(pwX["fw_lon_speed_mae"], pwB["fw_lon_speed_mae"], eid, nb)', "reader"),
    M(READER, "the KEEP row reads ADE2", 'keep = keep_row(BE.paired_row(pwX["ade6"], pwB["ade6"], eid, nb), F)', 'keep = keep_row(BE.paired_row(pwX["ade2"], pwB["ade2"], eid, nb), F)', "reader"),
    M(READER, "the KEEP row is B minus X", 'keep = keep_row(BE.paired_row(pwX["ade6"], pwB["ade6"], eid, nb), F)', 'keep = keep_row(BE.paired_row(pwB["ade6"], pwX["ade6"], eid, nb), F)', "reader"),
    M(READER, "HX is X (no re-weighting)", "HX, HX1, HX0 = H.apply_w(P, tauX, w), H.apply_w(P, tauX, 1.0), H.apply_w(P, tauX, 0.0)", "HX, HX1, HX0 = tauX, H.apply_w(P, tauX, 1.0), H.apply_w(P, tauX, 0.0)", "reader"),
    M(READER, "the DOMINANCE row reads HX minus B", "p_hx_p10 = paired_rows(I, pwHX, pwP, eid, nb)", "p_hx_p10 = paired_rows(I, pwHX, pwB, eid, nb)", "reader"),
    M(READER, "the DOMINANCE row reads X minus P10", "p_hx_p10 = paired_rows(I, pwHX, pwP, eid, nb)", "p_hx_p10 = paired_rows(I, pwX, pwP, eid, nb)", "reader"),
    M(READER, "the distance-keeping flags read X minus P10", 'dkf = dk_flags(dkread["pairs"][f"HX_s{s}__minus__P10"])', 'dkf = dk_flags(dkread["pairs"][f"X_s{s}__minus__P10"])', "reader"),
    M(READER, "consistency reads X, not HX", 'cons_hx_p10 = BE.paired_row(viol["HX"]["viol_any"].astype(float), viol["P10"]["viol_any"].astype(float), eid, nb)', 'cons_hx_p10 = BE.paired_row(viol["X"]["viol_any"].astype(float), viol["P10"]["viol_any"].astype(float), eid, nb)', "reader"),
    off(READER, "the identity controls are not enforced", 'if not ctl["all_pass"]:', "reader"),
    off(READER, "HX at w = 1 is not required to reproduce X's own block", "if d1:", "reader"),
    off(READER, "the eval's interface is not checked", 'if (bjX.get("o_interface") or "full") != "full":', "reader"),
    off(READER, "the eval's checkpoint path is not checked", 'if len(parts) < 2 or parts[-2] != x_dir(s) or parts[-1] != "ckpt_last.pt":', "reader"),
    off(READER, "the evals' grid identity is not checked", 'if not np.array_equal(np.asarray(zX["eid"]), np.asarray(eid)) or not np.array_equal(np.asarray(zB["eid"]), np.asarray(eid)):', "reader"),
    M(READER, "the evals' window count is not checked", 'if (n, ne) != (a.n_windows, a.n_episodes):\n        refuse(f"the evals hold', 'if False:\n        refuse(f"the evals hold', "reader"),
    off(READER, "the HX weights' checkpoint is not checked", "if W.get(\"ckpt_sha256\") != sha256_file(ck_path):", "reader"),
    off(READER, "the HX weights' built w_acc is not checked", 'if not _same(W.get("w_acc_built"), w_acc):', "reader"),
    off(READER, "the HX weights need not be monotone", "if not (w.shape == (30, 2) and ((w >= 0) & (w <= 1)).all() and (np.diff(w, axis=0) >= -1e-12).all()):", "reader"),
    off(READER, "a missing tactical / strategic family is not refused", "if not blk.get(fam):", "reader"),
    off(READER, "a missing HX family is not refused", 'if not (resHX.get("families") or {}).get(fam):', "reader"),
    off(READER, "the checkpoint's sha256 chain is not checked (trained vs after evals)", "if not (a1 and a2) or a1 != a2:", "reader"),
    off(READER, "the pulled checkpoint's sha256 is not checked", "if a1 != loc:", "reader"),
    off(READER, "B's banked checkpoint sha256 is not checked", "if got.get(tag) != want:", "reader"),
    off(READER, "the checkpoint's own cfg is not checked", "if not _same(wa, w_acc) or ash != sm.get(\"argv_sha\") or step != sm.get(\"steps\") or arm != \"M2b\" or sd != s:", "reader"),
    off(READER, "a run that is not done is read", 'if sm.get("done") is not True:', "reader"),
    off(READER, "another arm / seed is read", 'if sm.get("arm") != "M2b" or sm.get("seed") != s:', "reader"),
    off(READER, "another clip count is read", 'if sm.get("n_clips") != n_train:', "reader"),
    off(READER, "a run that did not withhold the holdout is read", 'if holdout is not None and sm.get("excluded_sha256") != holdout:', "reader"),
    off(READER, "another clip list / step count / window set is read", "if sm.get(k) != rsm.get(k):", "reader"),
    off(READER, "a run without the PASS token is read", 'if sc.get("declared_vs_built") != "PASS":', "reader"),
    off(READER, "G-LIVE need not pass", 'if live.get("pass") is not True or live.get("control_prior_grad_sum") != 0.0:', "reader"),
    off(READER, "a dead weight is accepted", 'if live.get("dead_params") != [] or live.get("declared_dead") != []:', "reader"),
    off(READER, "the term need not be in the loss dict", 'if "acc" not in (live.get("terms") or {}):', "reader"),
    off(READER, "the acc probe need not pass", "if not isinstance(pr, dict) or pr.get(\"pass\") is not True:", "reader"),
    off(READER, "the acc probe may have run for another weight", 'if pr.get("w_acc_declared") != w_acc:', "reader"),
    off(READER, "O's head may have received no gradient", 'if not (isinstance(pr.get("head_o_grad_diff"), (int, float)) and pr["head_o_grad_diff"] > 0.0):', "reader"),
    off(READER, "an acc probe control may be non-zero", 'if pr.get("control_prior_grad_diff") != 0.0 or pr.get("control_repeat_grad_diff") != 0.0 or pr.get("control_repeat_total_diff") != 0.0:', "reader"),
    off(READER, "the total need not have moved by w x the term", 'if not abs(float(pr.get("contribution", 0.0)) - float(pr.get("w_x_acc", 1.0))) <= 1e-3 * abs(float(pr.get("w_x_acc", 1.0))) + 1e-5:', "reader"),
    off(READER, "the selfcheck need not be this run's", 'if sc.get("argv_sha") != sm.get("argv_sha"):', "reader"),
    off(READER, "another arm on the argv is read", 'if fl.get("--arm") != "M2b":', "reader"),
    off(READER, "the lever need not be named once with the rule's value", "if argv.count(LEVER_FLAG) != 1 or not _same(fl.get(LEVER_FLAG), w_acc):", "reader"),
    off(READER, "B's argv may name the lever", "if LEVER_FLAG in rfl:", "reader"),
    off(READER, "a second lever on the argv is accepted", "if fx != fb:", "reader"),
    off(READER, "the built lever need not be recorded", 'if lg.get("w_acc") is None:', "reader"),
    off(READER, "other built levers may differ from B's", "if diff:", "reader"),
    off(READER, "loss weights may differ from B's (Amendment 2)", "if w_bad:", "reader"),
    off(READER, "the parameter count may differ from B's", 'if not pb or pb != px or lg.get("n_params_trainable") != lb.get("n_params_trainable"):', "reader"),
    off(READER, "a non-positive or unrecorded w_acc record is accepted", 'if not (isinstance(w_acc, (int, float)) and w_acc > 0 and wj.get("schema") == "v8f-oacc-wacc-v1"):', "reader"),
    M(READER, "the s1_required key is not written", '"s1_required": bool(s1), ', "", "reader"),
    M(READER, "s1_required is always true", '"s1_required": bool(s1), ', '"s1_required": True, ', "reader"),
    M(READER, "the reading carries no outcome", 'reading = (f"PREREG_V8F_OACC: {oc}; w_acc {w_acc};', 'reading = (f"PREREG_V8F_OACC: ; w_acc {w_acc};', "reader"),
    M(READER, "a pending read quotes a next step", 'nxt = None if oc == "PENDING SEED 1" else f"{oc}: {NEXT_TEXT[oc]}."', 'nxt = f"{oc}: {NEXT_TEXT.get(oc)}."', "reader"),
    M(READER, "the strategic family is not read", '"strategic": {"status": "READ (switched on for v8F: PI-V8F-2026-10-17, R72)", "B": bjB["raw"]["strategic"],', '"strategic": {"status": "READ (switched on for v8F: PI-V8F-2026-10-17, R72)", "B": None,', "reader"),
    M(READER, "the strategic family is marked identical by construction", '"identical_by_construction": False,', '"identical_by_construction": True,', "reader"),
    M(READER, "the reader labels the strategic family with the forbidden word", '"READ (switched on for v8F: PI-V8F-2026-10-17, R72)"', '"PAUSED (R72)"', "reader"),
    # =========================================================================================================== the queue (CRLF, edited as BYTES)
    M(QUEUE, "gateoacc does not pass --w-acc", '--arm M2b --w-acc %WACC% --out "%RUNSM%\\gate_M2b-oacc"', '--arm M2b --out "%RUNSM%\\gate_M2b-oacc"', "queue"),
    M(QUEUE, "gateoacc hard-codes a weight", '--arm M2b --w-acc %WACC% --out "%RUNSM%\\gate_M2b-oacc"', '--arm M2b --w-acc 1.0 --out "%RUNSM%\\gate_M2b-oacc"', "queue"),
    M(QUEUE, "gateoacc trains (no --selfcheck-only)", '--out "%RUNSM%\\gate_M2b-oacc" %A% --selfcheck-only"', '--out "%RUNSM%\\gate_M2b-oacc" %A%"', "queue"),
    M(QUEUE, "trainoacc does not pass --w-acc", "--grid pool4x10 --w-acc %WACC% --out", "--grid pool4x10 --out", "queue"),
    M(QUEUE, "trainoacc hard-codes a weight", "--grid pool4x10 --w-acc %WACC% --out", "--grid pool4x10 --w-acc 1.0 --out", "queue"),
    M(QUEUE, "trainoacc trains another arm", '"%TRAIN%" --arm M2b --seed %~2 --grid pool4x10 --w-acc', '"%TRAIN%" --arm M1 --seed %~2 --grid pool4x10 --w-acc', "queue"),
    M(QUEUE, "trainoacc moves a second lever", "--grid pool4x10 --w-acc %WACC% --out", "--grid pool4x10 --hist 10 --w-acc %WACC% --out", "queue"),
    M(QUEUE, "trainoacc runs in B's directory", f'if "%~2"=="" (echo REFUSED: trainoacc needs a SEED & exit /b 2){NL}set "OUT=%RUNSM%\\M2b_s%~2_oacc"', f'if "%~2"=="" (echo REFUSED: trainoacc needs a SEED & exit /b 2){NL}set "OUT=%RUNSM%\\M2b_s%~2"', "queue"),
    M(QUEUE, "trainoacc does not skip a finished run", 'if exist "%OUT%\\summary.json" (echo [skip] M2b_s%~2_oacc & exit /b 0)', "rem", "queue"),
    M(QUEUE, "trainoacc needs no seed", 'if "%~2"=="" (echo REFUSED: trainoacc needs a SEED & exit /b 2)', "rem", "queue"),
    M(QUEUE, "a failed training does not stop the stage", "(echo [FAIL] train M2b_s%~2_oacc - see %OUT%\\train.log & exit /b 1)", "(echo [FAIL] train M2b_s%~2_oacc - see %OUT%\\train.log & exit /b 0)", "queue"),
    M(QUEUE, "a training without a summary is accepted", "(echo [FAIL] train M2b_s%~2_oacc exited 0 without summary.json & exit /b 1)", "(echo [FAIL] train M2b_s%~2_oacc exited 0 without summary.json & exit /b 0)", "queue"),
    M(QUEUE, "a failed gate does not stop the stage", "(echo [FAIL] gate M2b-oacc - see %RUNSM%\\gate_M2b-oacc.log & exit /b 1)", "(echo [FAIL] gate M2b-oacc - see %RUNSM%\\gate_M2b-oacc.log & exit /b 0)", "queue"),
    M(QUEUE, "no value is accepted (no refusal without w_acc)", 'if not defined WACC (echo REFUSED: no w_acc - run the waccoacc stage first & exit /b 2)', "rem", "queue"),
    M(QUEUE, "the file wins over the env", 'if not defined WACC if exist "%OD%\\w_acc.txt" set /p WACC=<"%OD%\\w_acc.txt"', 'if exist "%OD%\\w_acc.txt" set /p WACC=<"%OD%\\w_acc.txt"', "queue"),
    M(QUEUE, "the file is not read", 'if not defined WACC if exist "%OD%\\w_acc.txt" set /p WACC=<"%OD%\\w_acc.txt"', "rem", "queue"),
    M(QUEUE, "the dry run does not print the placeholder", 'if not defined WACC if defined V8F_DRYRUN set "WACC=WACC_FROM_RULE"', "rem", "queue"),
    M(QUEUE, "evaloacc drops --families", f'{_TG}{NL}{_E0}', f'{_TG}{NL}{_E0.replace(" --project --families", " --project")}', "queue"),
    M(QUEUE, "evaloacc scores B's checkpoint", f'{_TG}{NL}{_E0}', f'{_TG}{NL}{_E0.replace(chr(34) + "%OUT%" + chr(92) + "ckpt_last.pt" + chr(34), chr(34) + "%RUNSM%" + chr(92) + "M2b_s%~2" + chr(92) + "ckpt_last.pt" + chr(34))}', "queue"),
    M(QUEUE, "evaloacc writes B's tag", f'{_TG}{NL}{_E0}', f'{_TG}{NL}{_E0.replace("--tag %TG%", "--tag M2b_s%~2")}', "queue"),
    M(QUEUE, "evaloacc runs an oracle", f'{_TG}{NL}{_E0}', f'{_TG}{NL}{_E0.replace(" --project --families", " --project --families --tac oracle")}', "queue"),
    M(QUEUE, "evaloacc does not refuse an unfinished run", f'(echo %E0% & exit /b 0){NL}if not exist "%OUT%\\summary.json" (echo REFUSED: %TG% has not finished training & exit /b 2)', f'(echo %E0% & exit /b 0){NL}rem', "queue"),
    M(QUEUE, "evaloacc does not skip a finished eval", f'(echo %E0% & exit /b 0){NL}if not exist "%OUT%\\summary.json" (echo REFUSED: %TG% has not finished training & exit /b 2){NL}if exist "%EV%\\eval_%TG%.npz" (echo [skip] eval %TG% & exit /b 0)',
      f'(echo %E0% & exit /b 0){NL}if not exist "%OUT%\\summary.json" (echo REFUSED: %TG% has not finished training & exit /b 2){NL}rem', "queue"),
    M(QUEUE, "the dry run does not print the eval", f'set "OUT=%RUNSM%\\M2b_s%~2_oacc"{NL}{_TG}{NL}{_E0}{NL}if defined V8F_DRYRUN (echo %E0% & exit /b 0)', f'set "OUT=%RUNSM%\\M2b_s%~2_oacc"{NL}{_TG}{NL}{_E0}{NL}if defined V8F_DRYRUN (exit /b 0)', "queue"),
    M(QUEUE, "waccoacc reads X's checkpoint", 'wacc --ckpt "%RUNSM%\\M2b_s0\\ckpt_last.pt"', 'wacc --ckpt "%RUNSM%\\M2b_s0_oacc\\ckpt_last.pt"', "queue"),
    M(QUEUE, "waccoacc reads B's seed 1", 'wacc --ckpt "%RUNSM%\\M2b_s0\\ckpt_last.pt"', 'wacc --ckpt "%RUNSM%\\M2b_s1\\ckpt_last.pt"', "queue"),
    M(QUEUE, "waccoacc is skipped on one file", 'if exist "%OD%\\w_acc.json" if exist "%OD%\\w_acc.txt" (echo [skip] w_acc & exit /b 0)', 'if exist "%OD%\\w_acc.json" (echo [skip] w_acc & exit /b 0)', "queue"),
    M(QUEUE, "a failed rule does not stop the stage", "(echo [FAIL] w_acc rule - see %OD%\\waccoacc.log & exit /b 1)", "(echo [FAIL] w_acc rule - see %OD%\\waccoacc.log & exit /b 0)", "queue"),
    M(QUEUE, "a rule without its number is accepted", 'if not exist "%OD%\\w_acc.txt" (echo [FAIL] w_acc rule exited 0 without w_acc.txt & exit /b 1)', "rem", "queue"),
    M(QUEUE, "a failed copy of HBLEND's clip list is not refused", 'copy /y "%HB%\\fit_clips.json" "%OD%\\fit_clips.json" >nul || (echo REFUSED: HBLEND\'s fit_clips.json did not copy & exit /b 2)', 'copy /y "%HB%\\fit_clips.json" "%OD%\\fit_clips.json" >nul', "queue"),
    M(QUEUE, "the rule's dry run prints nothing", "if defined V8F_DRYRUN (echo %WX% & exit /b 0)", "if defined V8F_DRYRUN (exit /b 0)", "queue"),
    M(QUEUE, "the rule has no cross-check file", '--banked-fit-windows "%OD%\\fit_windows_M2b_s0.npz" ', "", "queue"),
    M(QUEUE, "fitoacc reads B's checkpoint", 'hx --ckpt "%OUT%\\ckpt_last.pt"', 'hx --ckpt "%RUNSM%\\M2b_s%~2\\ckpt_last.pt"', "queue"),
    M(QUEUE, "fitoacc writes B's tag", '--tag M2b-oacc_s%~2 --out "%OD%\\weights_M2b-oacc_s%~2.json"', '--tag M2b_s%~2 --out "%OD%\\weights_M2b-oacc_s%~2.json"', "queue"),
    M(QUEUE, "fitoacc does not refuse an unfinished run", 'if not exist "%OUT%\\summary.json" (echo REFUSED: M2b-oacc_s%~2 has not finished training & exit /b 2)', "rem", "queue"),
    M(QUEUE, "fitoacc does not skip a finished fit", 'if exist "%OD%\\weights_M2b-oacc_s%~2.json" (echo [skip] HX weights M2b-oacc_s%~2 & exit /b 0)', "rem", "queue"),
    M(QUEUE, "a failed fit does not stop the stage", "(echo [FAIL] HX fit M2b-oacc_s%~2 - see %OD%\\fitoacc_s%~2.log & exit /b 1)", "(echo [FAIL] HX fit M2b-oacc_s%~2 - see %OD%\\fitoacc_s%~2.log & exit /b 0)", "queue"),
    M(QUEUE, "fitoacc does not need the clip list", 'if not exist "%OD%\\fit_clips.json" (echo REFUSED: %OD%\\fit_clips.json is missing - run waccoacc first & exit /b 2)', "rem", "queue"),
    M(QUEUE, "no waccoacc dispatch", 'if /i "%~1"=="waccoacc" goto :waccoacc', "rem no waccoacc", "queue"),
    M(QUEUE, "no gateoacc dispatch", 'if /i "%~1"=="gateoacc" goto :gateoacc', "rem no gateoacc", "queue"),
    M(QUEUE, "no trainoacc dispatch", 'if /i "%~1"=="trainoacc" goto :trainoacc', "rem no trainoacc", "queue"),
    M(QUEUE, "no evaloacc dispatch", 'if /i "%~1"=="evaloacc" goto :evaloacc', "rem no evaloacc", "queue"),
    M(QUEUE, "no fitoacc dispatch", 'if /i "%~1"=="fitoacc" goto :fitoacc', "rem no fitoacc", "queue"),
    M(QUEUE, "an old stage is edited (a blank line after evalflat_4)", f":evalflat_4{NL}exit /b 0{NL}:waccoacc", f":evalflat_4{NL}exit /b 0{NL}{NL}:waccoacc", "queue"),
    M(QUEUE, "an old stage mentions the lever", f":evalflat_4{NL}exit /b 0{NL}:waccoacc", f":evalflat_4{NL}rem --w-acc{NL}exit /b 0{NL}:waccoacc", "queue"),
    M(QUEUE, "the queue ends with a bare LF", f"echo [ok] HX weights M2b-oacc_s%~2{NL}exit /b 0{NL}", "echo [ok] HX weights M2b-oacc_s%~2\nexit /b 0\n", "queue"),
    # =========================================================================================================== the chain (the predecessor predicate included)
    M(CHAIN, "box 2's M2b_s0 checkpoint sha256 is wrong in the chain", "CKPT_SHA=(601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8 ", "CKPT_SHA=(601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f9 ", "chain"),
    M(CHAIN, "box 2's M2b_s1 checkpoint sha256 is wrong in the chain", " 6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6)", " 6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e7)", "chain"),
    M(CHAIN, "the chain waits for another predecessor", "AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_flat.out}", "AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_kind.out}", "chain"),
    M(CHAIN, "the chain reads another predecessor's guard log", "AFTER_GUARD=${AFTER_GUARD:-$L/v8fflat_guard.out}", "AFTER_GUARD=${AFTER_GUARD:-$L/v8fkind_guard.out}", "chain"),
    M(CHAIN, "the predicate does not look for DONE", '  if grep -a -q "^$AFTER_DONE" "$out" 2>/dev/null; then echo "PROCEED the predecessor ended:', '  if false; then echo "PROCEED the predecessor ended:', "chain"),
    M(CHAIN, "the predicate holds on a DONE", """tr -d '\\r' | cut -c1-160)"; return 0; fi""", """tr -d '\\r' | cut -c1-160)"; return 1; fi""", "chain"),
    M(CHAIN, "the predicate proceeds on a stop it has never seen", 'echo "HOLD a stop that is not recognised as verdict-, gate- or run-level (fail closed): $msg"; return 1', 'echo "PROCEED a stop that is not recognised: $msg"; return 0', "chain"),
    M(CHAIN, "the predicate ignores box-level evidence", "ev=$(grep -a -h -E '^(ZZGUARD-STOP|ZZCRASH|ZZUNREACHABLE|ZZTIMEOUT)'", "ev=$(grep -a -h -E '^(ZZNEVERPRINTED)'", "chain"),
    M(CHAIN, "the predicate does not read the guard log", '${guard:+"$guard"} 2>/dev/null | head -1', "2>/dev/null | head -1", "chain"),
    M(CHAIN, "the predicate takes a clean exit for a failed run", "rc=[1-9][0-9]*$'", "rc=[0-9]*$'", "chain"),
    M(CHAIN, "the predicate takes any job for a FLAT job", "RUN_LEVEL_RE='^v8fflat[0-9]{8}_(gateflat|t_M2b_s[01]_flat|e_M2b_s[01]_flat) rc=", "RUN_LEVEL_RE='^[A-Za-z0-9_.-]+ rc=", "chain"),
    M(CHAIN, "the predicate takes FLAT's refresh job for a run-level stop", "(gateflat|t_M2b_s[01]_flat|e_M2b_s[01]_flat) rc=", "(refresh|gateflat|t_M2b_s[01]_flat|e_M2b_s[01]_flat) rc=", "chain"),
    M(CHAIN, "the predicate takes a wait result for a run-level stop", "rc=[1-9][0-9]*$'", "(rc|wait)=[0-9]*.*$'", "chain"),
    M(CHAIN, "the predicate does not know FLAT's verdict-level stop", "VERDICT_LEVEL_RE='^flat read$'", "VERDICT_LEVEL_RE='^NEVER$'", "chain"),
    M(CHAIN, "the predicate takes any read for a verdict-level stop", "VERDICT_LEVEL_RE='^flat read$'", "VERDICT_LEVEL_RE='read'", "chain"),
    M(CHAIN, "the predicate does not know FLAT's gate-level stop", "GATE_LEVEL_RE='^gate FLAT assertions$'", "GATE_LEVEL_RE='^NEVER$'", "chain"),
    M(CHAIN, "the predicate takes any gate for a gate-level stop", "GATE_LEVEL_RE='^gate FLAT assertions$'", "GATE_LEVEL_RE='assertions'", "chain"),
    M(CHAIN, "the predicate does not know the pre-flight stops", '  if [[ $msg =~ $PREFLIGHT_LEVEL_RE ]]; then', '  if false; then', "chain"),
    M(CHAIN, "the predicate takes every text for a pre-flight stop", 'PREFLIGHT_LEVEL_RE="^(archive|', 'PREFLIGHT_LEVEL_RE="^(.*|archive|', "chain"),
    M(CHAIN, "the predicate takes FLAT's code-refresh failure for proceedable", '  if [[ $msg =~ ^v8fflat[0-9]{8}_refresh\\ rc= ]]; then echo "HOLD box-level', '  if [[ $msg =~ ^v8fflat[0-9]{8}_refresh\\ rc= ]]; then echo "PROCEED box-level', "chain"),
    M(CHAIN, "a held chain does not stop", """if [ $r = 1 ]; then echo "ZZV8FMINI-STOP the predecessor's end does not clear the box: $d"; exit 1; fi""", "if [ $r = 1 ]; then break; fi", "chain"),
    M(CHAIN, "the busy marker is not taken atomically", '( set -C; echo "v8F-mini PREREG_V8F_OACC', '( echo "v8F-mini PREREG_V8F_OACC', "chain"),
    M(CHAIN, "the rule stage queues another stage", "queue waccoacc $FIT_POLLS waccoacc", "queue waccoacc $FIT_POLLS gateoacc", "chain"),
    M(CHAIN, "the gate stage queues another stage", "queue gateoacc 120 gateoacc", "queue gateoacc 120 trainoacc 0", "chain"),
    M(CHAIN, "a failed check-inputs does not stop the chain", '|| stop "the reader cannot use the M2b side (check-inputs)"', "|| true", "chain"),
    M(CHAIN, "a failed rule assertion does not stop the chain", """<<'EOF' || stop "w_acc rule assertions\"""", "<<'EOF' || true", "chain"),
    M(CHAIN, "a failed gate assertion does not stop the chain", """<<'EOF' || stop "gate OACC assertions\"""", "<<'EOF' || true", "chain"),
    M(CHAIN, "a failed token write does not stop the chain", """<<'EOF' || stop "write the gate token\"""", "<<'EOF' || true", "chain"),
    M(CHAIN, "the checkpoint is not re-hashed after the evals", "  ckptsha_x $s after_evals\n", "  true\n", "chain"),
    M(CHAIN, "B's checkpoints are not asserted", 'rm -f $CK\nckptsha_b "before the rule"\n', "rm -f $CK\n", "chain"),
    M(CHAIN, "box 2's M2b checkpoint may differ from the banked one", '[ "$g" = "${CKPT_SHA[$s]}" ] || stop', '[ "$g" = "${CKPT_SHA[$s]}" ] || true #', "chain"),
    M(CHAIN, "the eval is not pulled", 'for x in json npz log; do pull "eval/eval_$tg.$x" "$E/eval"; done', "true", "chain"),
    M(CHAIN, "the HX weights are not pulled", 'pull "oacc/weights_$tg.json" "$E/w"', "true", "chain"),
    M(CHAIN, "the rule files are not pulled", """$SCP "tanitad-box2:$R/oacc/w_acc.json" "$E/w_acc.json" && $SCP "tanitad-box2:$R/oacc/w_acc.txt" "$E/w_acc.txt" || stop "pull the w_acc rule files\"""", "true", "chain"),
    M(CHAIN, "the rule's value is not the file's", """WACC=$(tr -d '\\r\\n' < $E/w_acc.txt); export WACC""", "WACC=0.5; export WACC", "chain"),
    M(CHAIN, "seed 1 is never run", "runone 1\n", "true\n", "chain"),
    M(CHAIN, "seed 1 is unconditional", """if [ "$(jget oacc_read_s0.json s1_required)" != "True" ]; then""", "if false; then", "chain"),
    M(CHAIN, "seed 1's condition is inverted", """if [ "$(jget oacc_read_s0.json s1_required)" != "True" ]; then""", """if [ "$(jget oacc_read_s0.json s1_required)" = "True" ]; then""", "chain"),
    M(CHAIN, "seed 1's condition reads another key", """jget oacc_read_s0.json s1_required""", """jget oacc_read_s0.json OUTCOME""", "chain"),
    M(CHAIN, "the seed-0 read is not taken", "oaccread 0 oacc_read_s0.json\n", "true\n", "chain"),
    M(CHAIN, "the final read is a one-seed read", "oaccread 0,1 oacc_read.json", "oaccread 0 oacc_read.json", "chain"),
    M(CHAIN, "stop does not release the busy marker", 'echo "" > $CUR; rm -f $B/BOX2_BUSY.txt; exit 1; }', 'echo "" > $CUR; exit 1; }', "chain"),
    M(CHAIN, "finish does not release the busy marker", '  rm -f $B/BOX2_BUSY.txt\n  echo "ZZV8FMINI-DONE', '  echo "ZZV8FMINI-DONE', "chain"),
    M(CHAIN, "the training wait is shorter than the guard cap", "TRAIN_POLLS=1320;", "TRAIN_POLLS=600;", "chain"),
    M(CHAIN, "the step count asserted is not B's", "M2B_STEPS=19164", "M2B_STEPS=19000", "chain"),
    M(CHAIN, "the window count asserted is not B's", "M2B_WINDOWS=306566 ", "M2B_WINDOWS=306000 ", "chain"),
    M(CHAIN, "the fit-set size asserted is not HBLEND's", "FIT_WINDOWS=9582 ", "FIT_WINDOWS=9000 ", "chain"),
    M(CHAIN, "the reader is not given the checkpoint sha json", "--wacc-json $4 --ckpt-sha $5 --seeds $6", "--wacc-json $4 --seeds $6", "chain"),
    M(CHAIN, "the reader is not given the rule's record", "--weights-dir $3 --wacc-json $4 --ckpt-sha $5", "--weights-dir $3 --ckpt-sha $5", "chain"),
    M(CHAIN, "the guard's markers are not checked", """grep -qF "'^(ZZV8FMINI|" $B/v8f_guard.sh || stop "v8f_guard.sh does not know this chain's markers (anchored): it would never exit\"""", "true", "chain"),
    M(CHAIN, "the tip is not checked for the lever", """| grep -q '"w_acc": cfg.w_acc' || stop "the tip's built_levers does not record w_acc\"""", "| true", "chain"),
    M(CHAIN, "the tip is not checked for the loss line", """| grep -q 'total = total + cfg.w_acc \\* T\\["acc"\\] if cfg.w_acc > 0.0 else total' || stop "the tip's v8f_mini.py has no acceleration line in the loss total\"""", "| true", "chain"),
    M(CHAIN, "the tip is not checked for the probe", """| grep -q 'acc_probe(model, probe, cfg)' || stop "the tip's trainer does not run the acc probe in its in-process gate\"""", "| true", "chain"),
    M(CHAIN, "the gate does not assert the built lever", 'if lv.get("w_acc") != w: bad.append(', "if False: bad.append(", "chain"),
    M(CHAIN, "the gate does not assert B's interface", 'if lv.get("o_interface", "full") != "full": bad.append(', "if False: bad.append(", "chain"),
    M(CHAIN, "the gate does not assert the code mix", 'if lv.get("code_mix", "whole") != "whole": bad.append(', "if False: bad.append(", "chain"),
    M(CHAIN, "the gate does not assert parameter parity", 'if lv["n_params_trainable"] != bl["n_params_trainable"]: bad.append(', "if False: bad.append(", "chain"),
    M(CHAIN, "the gate does not assert the probe passes", 'if not isinstance(pr, dict) or pr.get("pass") is not True: bad.append(', "if False: bad.append(", "chain"),
    M(CHAIN, "the gate does not assert the probe's weight", 'if pr.get("w_acc_declared") != w or pr.get("w_probe") != w: bad.append(', "if False: bad.append(", "chain"),
    M(CHAIN, "the gate does not assert O's head gradient", 'if not pr.get("head_o_grad_diff", 0.0) > 0.0: bad.append(', "if False: bad.append(", "chain"),
    M(CHAIN, "the gate does not assert the contribution", 'if abs(pr.get("contribution", 0.0) - pr.get("w_x_acc", 1.0)) > 1e-3 * pr.get("w_x_acc", 1.0) + 1e-5: bad.append(', "if False: bad.append(", "chain"),
    M(CHAIN, "the gate does not assert the dead set", 'if live.get("dead_params") != [] or live.get("declared_dead") != []: bad.append(', "if False: bad.append(", "chain"),
    M(CHAIN, "the gate does not assert the term is in the loss dict", """if "acc" not in live.get("terms", {}): bad.append(""", "if False: bad.append(", "chain"),
    M(CHAIN, "the gate does not assert the built p_gt schedule", 'if (lv.get("p_gt_start"), lv.get("p_gt_end")) != (1.0, 0.2): bad.append(', "if False: bad.append(", "chain"),
    M(CHAIN, "the gate does not assert the queue line carries the lever", 'if toks.count("--w-acc") != 1 or float(toks[toks.index("--w-acc") + 1]) != w: bad.append(', "if False: bad.append(", "chain"),
    M(CHAIN, "the gate does not assert B's window set", """if i.get("n_windows_all_clips") != b.get("n_windows"): bad.append(""", "if False: bad.append(", "chain"),
    M(CHAIN, "the gate does not assert the step count", 'if sc["steps_total"] != steps: bad.append(', "if False: bad.append(", "chain"),
    M(CHAIN, "the rule assertions do not check the cross-check", 'if cc.get("pass") is not True: bad.append(', "if False: bad.append(", "chain"),
    M(CHAIN, "the rule assertions do not check B's checkpoint", """if (r.get("source") or {}).get("ckpt_sha256") != b0: bad.append(""", "if False: bad.append(", "chain"),
    M(CHAIN, "the rule assertions do not check the fit-set size", """if r.get("n_windows") != nfit: bad.append(""", "if False: bad.append(", "chain"),
    M(CHAIN, "the rule assertions do not check the ratio", """if abs(r.get("check_w_acc_x_L_acc_over_L_traj", 0) - 0.5) > 1e-4: bad.append(""", "if False: bad.append(", "chain"),
    M(CHAIN, "the rule assertions do not check the txt", """if float(txt) != r.get("w_acc") or repr(r.get("w_acc")) != txt: bad.append(""", "if False: bad.append(", "chain"),
    M(CHAIN, "the token carries a zero weight", '"w_acc": float(sys.argv[6]),', '"w_acc": 0.0,', "chain"),
    # =========================================================================================================== added with the second proof (run 2): each clause of the reader's checks that the first list named only as a whole
    M(READER, "the checkpoint's own cfg: the argv_sha is not compared", ' or ash != sm.get("argv_sha") or step != sm.get("steps")', ' or step != sm.get("steps")', "reader"),
    M(READER, "the checkpoint's own cfg: the step is not compared", ' or step != sm.get("steps") or arm != "M2b"', ' or arm != "M2b"', "reader"),
    M(READER, "the checkpoint's own cfg: the arm is not compared", ' or arm != "M2b" or sd != s:', ' or sd != s:', "reader"),
    M(READER, "the checkpoint's own cfg: the seed is not compared", ' or arm != "M2b" or sd != s:', ' or arm != "M2b":', "reader"),
    M(READER, "the checkpoint's own cfg: the built w_acc is not compared", 'if not _same(wa, w_acc) or ash != sm.get("argv_sha")', 'if ash != sm.get("argv_sha")', "reader"),
    M(READER, "the HX weights' shape is not checked", "if not (w.shape == (30, 2) and ((w >= 0) & (w <= 1)).all()", "if not (((w >= 0) & (w <= 1)).all()", "reader"),
    M(READER, "the HX weights' upper bound is not checked", "((w >= 0) & (w <= 1)).all() and (np.diff", "(w >= 0).all() and (np.diff", "reader"),
    M(READER, "the HX weights' lower bound is not checked", "((w >= 0) & (w <= 1)).all() and (np.diff", "(w <= 1).all() and (np.diff", "reader"),
    M(READER, "the HX weights' axes are swapped", 'w = np.stack([np.asarray(W["w_along_by_step"], np.float64), np.asarray(W["w_cross_by_step"], np.float64)], 1)',
      'w = np.stack([np.asarray(W["w_cross_by_step"], np.float64), np.asarray(W["w_along_by_step"], np.float64)], 1)', "reader"),
    M(READER, "consistency is read against B, not P10", 'viol["HX"]["viol_any"].astype(float), viol["P10"]["viol_any"].astype(float), eid, nb)', 'viol["HX"]["viol_any"].astype(float), viol["B"]["viol_any"].astype(float), eid, nb)', "reader"),
    M(READER, "the distance-keeping pair is HX minus B", 'dkf = dk_flags(dkread["pairs"][f"HX_s{s}__minus__P10"])', 'dkf = dk_flags(dkread["pairs"][f"HX_s{s}__minus__B_s{s}"])', "reader"),
    M(READER, "the lead block is read at another horizon", "k=30, dt=0.2, n_boot=a.dk_n_boot", "k=20, dt=0.2, n_boot=a.dk_n_boot", "reader"),
    M(READER, "the distance-keeping bootstrap ignores --dk-n-boot", "n_boot=a.dk_n_boot, seed=0, saved_ade6", "n_boot=7, seed=0, saved_ade6", "reader"),
    M(READER, "the paired bootstrap ignores --n-boot", "    nb = a.n_boot\n", "    nb = 7\n", "reader"),
    M(READER, "HX is X at w = 1 only (the weights are ignored for the scored plan)", "HX, HX1, HX0 = H.apply_w(P, tauX, w), H.apply_w(P, tauX, 1.0), H.apply_w(P, tauX, 0.0)", "HX, HX1, HX0 = H.apply_w(P, tauX, 0.5), H.apply_w(P, tauX, 1.0), H.apply_w(P, tauX, 0.0)", "reader"),
    M(READER, "the ACC row is taken at the other seed's B", 'acc_row = BE.paired_row(pwX["fw_lon_accel_mae"], pwB["fw_lon_accel_mae"], eid, nb)', 'acc_row = BE.paired_row(pwX["fw_lon_accel_mae"], H.arm_pw(P10, gt, BE)["fw_lon_accel_mae"], eid, nb)', "reader"),
    M(READER, "the output record's F is not the exact one", '"F": F, "keep_bar": F / 2, "keep_bar_printed"', '"F": 0.1207, "keep_bar": 0.1207 / 2, "keep_bar_printed"', "reader"),
    M(READER, "the output record's checkpoint sha256 is not the measured one", '"checkpoint_sha256": sha,', '"checkpoint_sha256": {},', "reader"),
]

if LIST:
    for i, (p, n, e, g) in enumerate(MUT):
        print(i, p.name, "|", n)
    sys.exit(0)

orig = {p: p.read_bytes() for p in {m[0] for m in MUT}}
bad_anchor = 0
for path, name, edits, g in MUT:
    for a, b in edits:
        a_b = a.encode("utf-8")
        if path is QUEUE or path.suffix == ".cmd":
            a_b = a_b.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n") if b"\r\n" not in a_b else a_b
        n = orig[path].count(a_b)
        if n != 1:
            bad_anchor += 1
            print(f"ANCHOR FOUND {n}x (need exactly 1): [{path.name}] {name}\n    {a!r}", flush=True)
if bad_anchor:
    sys.exit(3)
print(f"{len(MUT)} mutants over {len(orig)} files, each anchor found exactly once", flush=True)
if CHECK:
    sys.exit(0)

BASE = TMP / f"mut{FROM}_{TO if TO < 10 ** 6 else 'end'}"
BASE.parent.mkdir(exist_ok=True, parents=True)
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2"}


def first_failure(out):
    """'assert' | 'exception:<Type>' | 'collection' | 'internal' | '' -- the kind of the FIRST failing test's first E line."""
    if "INTERNALERROR" in out:
        return "internal"
    if "ERROR collecting" in out or "SyntaxError" in out.split("FAILURES")[0]:
        return "collection"
    i = out.find("FAILURES")
    if i < 0:
        return ""
    m = re.search(r"^E\s+(.*)$", out[i:], re.M)
    if not m:
        return ""
    e = m.group(1).strip()
    if e.startswith(("assert", "AssertionError", "Failed:")):
        return "assert"
    return "exception:" + e.split(":")[0].split(" ")[0]


def run(sel=None, x=True):
    cmd = [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", str(BASE), TEST]
    if x:
        cmd[3:3] = ["-x"]
    if sel:
        cmd[-1:-1] = ["-k", sel]
    shutil.rmtree(BASE, ignore_errors=True)
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=ENV, timeout=7200)
    shutil.rmtree(BASE, ignore_errors=True)
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.startswith(("FAILED", "ERROR")) or " passed" in ln or " failed" in ln or " error" in ln]
    tail = " | ".join(ln[:170] for ln in (lines[-2:] if lines else [(r.stdout.strip().splitlines() or [r.stderr.strip()[-170:]])[-1]]))
    return r.returncode, tail, r.stdout


def restore():
    for p, b in orig.items():
        p.write_bytes(b)


if NO_BASE:
    print("baseline: SKIPPED (--no-baseline: development run, not a banked proof)", flush=True)
else:
    rc, tail, _ = run()
    print("baseline:", rc, tail, flush=True)
    assert rc == 0, "the unmutated tree must be green"
red, survived, by_exc, ran = 0, [], [], 0
try:
    for i, (path, name, edits, group) in enumerate(MUT):
        if i < FROM or i >= TO or (ONLY and ONLY not in name and ONLY not in path.name) or (IDX is not None and i not in IDX):
            continue
        ran += 1
        src = orig[path]
        for a, b in edits:
            a_b, b_b = a.encode("utf-8"), b.encode("utf-8")
            if path.suffix == ".cmd" and b"\r\n" not in a_b:
                a_b = a_b.replace(b"\n", b"\r\n")
                b_b = b_b.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n") if b"\r\n" not in b_b and name != "the queue ends with a bare LF" else b_b
            src = src.replace(a_b, b_b, 1)
        path.write_bytes(src)
        kind, where = "", ""
        try:
            sel = SEL.get(group)
            rc, tail, out = run(sel)
            where = f"[{group} group]" if sel else ""
            if rc == 0 and sel:
                rc, tail, out = run()                                   # survived its group: the whole file decides
                where = "[whole file]"
            if rc != 0:
                kind = first_failure(out)
                if kind.startswith(("exception", "collection", "internal")) or not kind:
                    rc2, tail2, out2 = run(sel if where != "[whole file]" else None, x=False)       # any test failing by ASSERTION too?
                    sections = re.split(r"\n_{3,} .* _{3,}\n", out2)[1:]
                    n_assert = sum(1 for s in sections if re.search(r"^E\s+(assert|AssertionError|Failed:)", s, re.M))
                    if kind.startswith("collection") or kind == "internal":
                        n_assert = 0
                    kind = f"{kind}; {n_assert} of {len(sections)} failing tests fail by assertion"
        finally:
            path.write_bytes(orig[path])
        red += rc != 0
        if rc == 0:
            survived.append(name)
        elif not kind.startswith("assert"):
            mm = re.search(r"; (\d+) of", kind)
            if not mm or int(mm.group(1)) == 0:
                by_exc.append(name)
        print(f"{'RED     ' if rc else 'SURVIVED'} #{i} [{path.name}] {name} {where}: ({kind}) {tail}", flush=True)
finally:
    restore()
rc, tail, _ = run()
print("restored:", rc, tail, "| byte-identical:", all(p.read_bytes() == b for p, b in orig.items()))
print(f"{red}/{ran} RED" + (f"  SURVIVED: {survived}" if survived else "") + (f"  RED BY EXCEPTION ONLY (no test fails by assertion): {by_exc}" if by_exc else ""))
sys.exit(0 if red == ran and rc == 0 and not by_exc else 1)
