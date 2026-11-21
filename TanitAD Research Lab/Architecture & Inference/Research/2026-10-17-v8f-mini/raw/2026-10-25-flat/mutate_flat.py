"""Mutation proof for PREREG_V8F_FLAT's instruments: every mutation must turn stack/tests/test_v8f_flat.py RED; the unmutated tree must be GREEN before and after, and every file is
restored byte-identical.

    python mutate_flat.py <repo root> <python exe> [--check] [--only <substring of a mutant's name or file>] [--list] [--from N --to M]

    --check   only verify that every anchor occurs exactly once (in the ORIGINAL bytes), then exit
    --only    run only the mutants whose name or file contains the substring (the baseline and the restore still run)
    --from/--to  run only mutants N .. M-1 of `--list` (the baseline and the restore still run): the harness is long, so it can be sharded

The mutants live in the model (the cut, its three halves, the config, the built lever), the trainer (the flag, make_config, G-DVB, G-LIVE's dead set, the o_interface probe), the evaluator
(the recorded interface), the reader (the committed bars, the outcome table, every refusal, the identity read), the queue (CRLF, edited as BYTES) and the chain (the predecessor
predicate included). Each mutation is one (or, where stated, two) exact substring(s) replaced once; a mutation whose anchor is not found exactly once aborts the run.

THE DISK: the first PREREG_V8F_MIXO harness filled C: at mutant 9 (ENOSPC). This one runs on ONE copy of the tree, with its pytest temp dir (BASE) wiped before every run, and
test_v8f_flat.py removes each synthetic world when its test ends.

SPEED: a mutant is first run against the group of tests that own its file (`-k`), with `-x`; a RED there is a RED of the suite. A mutant that survives its group is re-run against the WHOLE
file before it is called SURVIVED, so a too-narrow selector can never hide a live mutant, only cost time.
"""
import os
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
PKG = ROOT / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
CODE = PKG / "code"
MODEL = ROOT / "stack" / "tanitad" / "models" / "v8f_mini.py"
TRAIN = ROOT / "stack" / "scripts" / "v8f_mini_train.py"
EVAL = ROOT / "stack" / "scripts" / "v8f_mini_eval.py"
READER = CODE / "v8f_flat_read.py"
QUEUE = CODE / "v8f_mini_box2_queue.cmd"
CHAIN = CODE / "box2_ops" / "chain_v8f_flat.sh"
TEST = "stack/tests/test_v8f_flat.py"

_RD = "reader or check_inputs or strata or flat_identity or known_values or gate_token or committed or seed_state or verdict_function"
_CH = "chain_flat or predecessor or wait_loop or every_stop"
SEL = {
    "model": f"not ({_RD} or queue_flat or {_CH})",
    "reader": _RD,
    "queue": "queue_flat",
    "chain": _CH,
}
NL = "\r\n"          # the queue is CRLF: its multi-line anchors say so
_TG = 'set "TG=M2b-flat_s%~2"'
_E = {0: 'set "E0="%PY%" "%EVAL%" run --ckpt "%OUT%\\ckpt_last.pt" --tag %TG% %EC% %ER% --project --families"',
      1: 'set "E1="%PY%" "%EVAL%" run --ckpt "%OUT%\\ckpt_last.pt" --tag %TG%-tacabs --tac absmask %EC% %ER%"',
      2: 'set "E2="%PY%" "%EVAL%" run --ckpt "%OUT%\\ckpt_last.pt" --tag %TG%-tacorc --tac oracle %EC% %ER%"',
      3: 'set "E3="%PY%" "%EVAL%" run --ckpt "%OUT%\\ckpt_last.pt" --tag %TG%-tacorc-all-latcons --tac oracle --tac-groups all-latcons --tac-alpha 1 %EC% %ER%"'}


def _evblock(upto):
    """evalflat's own lines from its (unique) TG line down to E<upto>: the eval strings are the same text in other arms' stages, so a mutant anchors on THIS stage's TG line."""
    return NL.join([_TG] + [_E[k] for k in range(upto + 1)])


_DRY_TAIL = f"echo %E3% & exit /b 0){NL}if not exist \"%OUT%\\summary.json\" (echo REFUSED: %TG% has not finished training & exit /b 2){NL}if exist \"%EV%\\eval_%TG%.npz\" (echo [skip] eval %TG% & goto :evalflat_1)"


def M(path, name, a, b, group):
    return (path, name, [(a, b)], group)


def MM(path, name, edits, group):
    return (path, name, edits, group)


def off(path, name, anchor, group, repl="False"):
    """`<indent>if <cond>:` -> `<indent>if False:`  (the anchor is the whole condition text, from `if` to the colon)."""
    kw = "elif" if anchor.startswith("elif") else "if"
    return M(path, name, anchor, f"{kw} {repl}:", group)


MUT = [
    # =========================================================================================================== the model: the cut, its three halves, the config, the built lever
    off(MODEL, "the cut never runs (cut == full)", 'if cfg.o_interface == "cut":', "model"),
    M(MODEL, "the cut runs under full (and not under cut)", 'if cfg.o_interface == "cut":', 'if cfg.o_interface == "full":', "model"),
    M(MODEL, "the checkpoint half is deleted: O still reads S's checkpoints", "ck_for_o = self.ck_nom[None].expand(B, -1, -1)", "ck_for_o = ck_for_o", "model"),
    M(MODEL, "the ok half is deleted: O reads the nominal arcs but is told they are present", "ck_ok = torch.zeros_like(ck_ok)", "ck_ok = ck_ok", "model"),
    M(MODEL, "the code half is deleted: O still reads T's code", "code = torch.zeros_like(code)", "code = code", "model"),
    M(MODEL, "the cut hands O ones, not zeros", "code = torch.zeros_like(code)", "code = torch.ones_like(code)", "model"),
    M(MODEL, "the cut hands O a nominal arc that is not the model's", "ck_for_o = self.ck_nom[None].expand(B, -1, -1)", "ck_for_o = (self.ck_nom[None] * 1.01).expand(B, -1, -1)", "model"),
    M(MODEL, "the cut hands O the ok flag set", "ck_ok = torch.zeros_like(ck_ok)", "ck_ok = torch.ones_like(ck_ok)", "model"),
    M(MODEL, "out['code'] stops recording the code OFFERED to the interface", 'out["code"] = code', 'out["code"] = code * 0', "model"),
    M(MODEL, "the config default is cut", '    o_interface: str = "full"', '    o_interface: str = "cut"', "model"),
    M(MODEL, "the modes lack cut", 'O_INTERFACE_MODES = ("full", "cut")', 'O_INTERFACE_MODES = ("full",)', "model"),
    off(MODEL, "an unknown o_interface is accepted by the config", "if self.o_interface not in O_INTERFACE_MODES:", "model"),
    M(MODEL, "built_levers does not record o_interface", '"o_interface": cfg.o_interface,', "", "model"),
    M(MODEL, "built_levers echoes 'full' whatever was built", '"o_interface": cfg.o_interface,', '"o_interface": "full",', "model"),
    # =========================================================================================================== the trainer: flag, make_config, G-DVB, G-LIVE's dead set, the probe
    M(TRAIN, "the flag has no declared-vs-built entry", '"o_interface": "lever:o_interface",', "", "model"),
    M(TRAIN, "make_config drops the lever", 'code_mix=a.code_mix, o_interface=a.o_interface)', "code_mix=a.code_mix)", "model"),
    off(TRAIN, "declared-vs-built does not compare the built o_interface", 'if built.get("o_interface") != a.o_interface:', "model"),
    M(TRAIN, "the flag's default is cut", 'choices=M.O_INTERFACE_MODES, default="full",', 'choices=M.O_INTERFACE_MODES, default="cut",', "model"),
    M(TRAIN, "the flag accepts any value", "choices=M.O_INTERFACE_MODES, default=", "choices=None, default=", "model"),
    M(TRAIN, "live_check does not declare e_code.weight dead under cut", 'not cfg.consistency)\n    declared_dead = [n for n in acc if cfg.wm == "a" and ".idm_heads." in n]\n    if cfg.o_interface == "cut":',
      'not cfg.consistency)\n    declared_dead = [n for n in acc if cfg.wm == "a" and ".idm_heads." in n]\n    if False:', "model"),
    M(TRAIN, "live_check_lora does not declare e_code.weight dead under cut", 'missing = sorted(expect - set(terms_seen))\n    declared_dead = [n for n in acc if cfg.wm == "a" and ".idm_heads." in n]\n    if cfg.o_interface == "cut":',
      'missing = sorted(expect - set(terms_seen))\n    declared_dead = [n for n in acc if cfg.wm == "a" and ".idm_heads." in n]\n    if False:', "model"),
    M(TRAIN, "live_check does not require the zero (the dead weight may be alive)", "declared_dead_alive = sorted(n for n in declared_dead if acc[n] != 0.0)", "declared_dead_alive = []", "model"),
    M(TRAIN, "the probe is faked (never reads what O is handed)", "        ocut = o_interface_probe(model, probe, cfg)",
      '        ocut = {"o_interface": cfg.o_interface, "shift": 2.5, "n_windows": 1, "perturbations": {}, "pass": True}  #', "model"),
    M(TRAIN, "the probe's refusal does not stop the run", '    except AssertionError as e:\n        raise SystemExit(f"ZZDVB-REFUSEDZZ declared-vs-built: {e}")\n    adm = M.assert_no_situation_classifier_input',
      '    except AssertionError as e:\n        ocut = {"pass": False, "error": str(e)}\n    adm = M.assert_no_situation_classifier_input', "model"),
    M(TRAIN, "selfcheck.json omits the probe", '"o_interface_probe": ocut, ', "", "model"),
    off(TRAIN, "the probe does not refuse a cut whose traj60 moves", "if d_traj != 0.0:", "model"),
    off(TRAIN, "the probe does not refuse a full whose traj60 does not move (it cannot see a change)", "if not d_traj > 0.0:", "model"),
    off(TRAIN, "the probe does not refuse a cut that hands O a non-constant", 'if row["o_read_code_abs_max"] != 0.0 or row["o_read_ck_minus_nominal_abs_max"] != 0.0 or row["o_read_ck_ok_abs_max"] != 0.0:', "model"),
    off(TRAIN, "the probe does not require T to still read S", 'if not row["t_reads_s_max_abs_dgoal_logits"] > 0.0:', "model"),
    off(TRAIN, "the probe does not check the override reached the interface", 'if ov is not None and not torch.equal(out["code"], ov.to(out["code"].dtype)):', "model"),
    off(TRAIN, "the probe does not check the checkpoint shift moved S", 'if row["max_abs_d_ckpt"] == 0.0:', "model"),
    M(TRAIN, "the probe leaves the residual head at zero (blind)", "        w.copy_(torch.randn(w.shape, generator=g) * 0.05)", "        w.copy_(torch.zeros(w.shape))", "model"),
    M(TRAIN, "the probe perturbs no random code", '"random_code": (torch.rand(own.shape, generator=g).to(own.dtype), False), ', "", "model"),
    M(TRAIN, "the probe shifts the checkpoints by nothing", "def o_interface_probe(model, batch, cfg, shift: float = 2.5)", "def o_interface_probe(model, batch, cfg, shift: float = 0.0)", "model"),
    # =========================================================================================================== the evaluator
    M(EVAL, "the eval does not record the interface it built", '"tac": a.tac, "o_interface": cfg.o_interface}', '"tac": a.tac}', "model"),
    # =========================================================================================================== the reader: bars, outcome table, identity, refusals
    M(READER, "the 2F bar is 0.2400", "BAR_2F = 0.2415 ", "BAR_2F = 0.2400 ", "reader"),
    M(READER, "the bootstrap B defaults to 1,000", "\nB = 2000\n", "\nB = 1000\n", "reader"),
    M(READER, "LO needs no separation", 'if pt <= -BAR_2F and p["separated"] and p["hi"] < 0:', 'if pt <= -BAR_2F and p["hi"] < 0:', "reader"),
    M(READER, "LO needs no interval below 0", 'if pt <= -BAR_2F and p["separated"] and p["hi"] < 0:', 'if pt <= -BAR_2F and p["separated"]:', "reader"),
    M(READER, "HI needs no separation", 'if pt >= BAR_2F and p["separated"] and p["lo"] > 0:', 'if pt >= BAR_2F and p["lo"] > 0:', "reader"),
    M(READER, "HI needs no interval above 0", 'if pt >= BAR_2F and p["separated"] and p["lo"] > 0:', 'if pt >= BAR_2F and p["separated"]:', "reader"),
    M(READER, "a seed exactly at -2F does not clear", 'if pt <= -BAR_2F and', 'if pt < -BAR_2F and', "reader"),
    M(READER, "a seed exactly at +2F does not clear", 'if pt >= BAR_2F and', 'if pt > BAR_2F and', "reader"),
    M(READER, "the sign is flipped: LO is a positive Delta", 'if pt <= -BAR_2F and', 'if pt >= BAR_2F and', "reader"),
    # the nine-cell outcome table (PREREG_V8F_FLAT Amendment 1): one mutant per class of mis-mapped cell, plus the hard-error guards
    M(READER, "verdict: (H, M) beats flat on one LO seed", '("H", "M"): "NOT ESTABLISHED"', '("H", "M"): "HIERARCHY BEATS FLAT"', "reader"),
    M(READER, "verdict: (F, M) is FLAT BETTER on one HI seed", '("F", "M"): "FLAT LEANING"', '("F", "M"): "FLAT BETTER"', "reader"),
    M(READER, "verdict: (F, M) falls back to NOT ESTABLISHED (the pre-amendment reading)", '("F", "M"): "FLAT LEANING"', '("F", "M"): "NOT ESTABLISHED"', "reader"),
    M(READER, "verdict: (M, F) is FLAT BETTER on one HI seed", '("M", "F"): "FLAT LEANING"', '("M", "F"): "FLAT BETTER"', "reader"),
    M(READER, "verdict: SEEDS DISAGREE reads NOT ESTABLISHED", '("H", "F"): "SEEDS DISAGREE"', '("H", "F"): "NOT ESTABLISHED"', "reader"),
    M(READER, "verdict: (M, M) reads FLAT LEANING", '("M", "M"): "NOT ESTABLISHED"}', '("M", "M"): "FLAT LEANING"}', "reader"),
    M(READER, "verdict: two outcomes swapped ((H, H) and (F, F))", '("H", "H"): "HIERARCHY BEATS FLAT",\n              ("F", "F"): "FLAT BETTER",',
      '("H", "H"): "FLAT BETTER",\n              ("F", "F"): "HIERARCHY BEATS FLAT",', "reader"),
    M(READER, "verdict: the H / F letters are swapped", 'HFM = {"LO": "H", "HI": "F", "MID": "M"}', 'HFM = {"LO": "F", "HI": "H", "MID": "M"}', "reader"),
    M(READER, "verdict: a pair missing from the table is not a hard error", "    if key not in OUTCOME_OF:\n", "    if False:\n", "reader"),
    M(READER, "verdict: an unknown seed state is not a hard error", " or any(x not in HFM for x in states)", "", "reader"),
    M(READER, "Delta is FLAT - B (the sign of the table flipped)", "row = PGT.contrast(pairfn, Bm, A, pw, ctx, a.n_boot)", "row = PGT.contrast(pairfn, A, Bm, pw, ctx, a.n_boot)", "reader"),
    off(READER, "a one-seed read is allowed", "if not os.path.isdir(os.path.join(a.runs_dir, x_dir(s))):", "reader"),
    M(READER, "the identity is not enforced", 'if not (np.array_equal(Z["orc_ade6"], Z["ade6"])', 'if False and not (np.array_equal(Z["orc_ade6"], Z["ade6"])', "reader"),
    M(READER, "the identity ignores the oracle's plan", ' and np.array_equal(Z["orc_pred"], Z["pred"])', "", "reader"),
    M(READER, "the identity ignores the all-latcons plan", ' and np.array_equal(Z["lat_pred"], Z["pred"])', "", "reader"),
    off(READER, "the identity read need not have changed the code", 'if m.get("mode") != "oracle" or not (isinstance(m.get("frac_code_changed"), (int, float)) and m["frac_code_changed"] > 0.0):', "reader"),
    M(READER, "the launch-gate token is not checked", "        check_gate_token(tag, sm, sc, bad)", "        pass", "reader"),
    off(READER, "the gate token does not require the dead set", 'if live.get("declared_dead") != ["e_code.weight"] or live.get("dead_params") != ["e_code.weight"]:', "reader"),
    off(READER, "the gate token does not require a passing probe", 'if not isinstance(pr, dict) or pr.get("pass") is not True:', "reader"),
    off(READER, "the gate token does not read the probe's values", 'if v.get("max_abs_dtraj60") != 0.0 or v.get("o_read_code_abs_max") != 0.0 or v.get("o_read_ck_minus_nominal_abs_max") != 0.0 or v.get("o_read_ck_ok_abs_max") != 0.0:', "reader"),
    off(READER, "the gate token does not require T to read S", 'if not (isinstance(sh.get("t_reads_s_max_abs_dgoal_logits"), (int, float)) and sh["t_reads_s_max_abs_dgoal_logits"] > 0.0):', "reader"),
    off(READER, "the argv need not carry the lever exactly once", "if argv.count(LEVER_FLAG) != 1 or fl.get(LEVER_FLAG) != LEVER_VAL:", "reader"),
    off(READER, "B's argv may name the lever", "if LEVER_FLAG in rfl:", "reader"),
    off(READER, "the argv may differ from B's besides the lever", "if fx != fb:", "reader"),
    off(READER, "the built o_interface may be anything", "elif lg.get(\"o_interface\") != LEVER_VAL:", "reader"),
    off(READER, "the built code_mix may be anything", 'if lg.get("code_mix", "whole") != "whole":', "reader"),
    off(READER, "the built p_gt schedule may be anything", "elif not PGT._same_float(lg[key], want):", "reader"),
    off(READER, "the built hmax may be anything", 'if lg.get("hmax") is not None and lg["hmax"] != hm_b:', "reader"),
    M(READER, "the built levers may differ from B's", "diff = sorted(k for k in set(lg) & set(lb) if k not in _SKIP_LEVERS_X and lg[k] != lb[k])", "diff = []", "reader"),
    M(READER, "the built loss weights may differ from B's", 'diff += [f"w.{k}" for k in w_diff(lg.get("w") or {}, lb.get("w") or {}, rfl)]', "pass", "reader"),
    off(READER, "B may have built the cut", '''if (rsm.get("levers") or {}).get("o_interface", "full") != "full":''', "reader"),
    off(READER, "a parameter group may differ from B's", "if px.get(k, 0) != pb.get(k, 0):", "reader"),
    off(READER, "the parameter total may differ from B's", 'if (lv_x.get("n_params_trainable") or 0) != (lv_b.get("n_params_trainable") or 0):', "reader"),
    off(READER, "the checkpoint may be built full", 'if ck["o_interface"] != LEVER_VAL:', "reader"),
    off(READER, "the checkpoint may be built with another code_mix", 'if ck["code_mix"] not in (None, "whole"):', "reader"),
    off(READER, "the checkpoint may be built with another hmax", 'if ck["hmax"] != hm_b:', "reader"),
    off(READER, "the checkpoint may be built on another grid", 'if ck["grid"] != B_GRID or ck["n_tok"] != N_TOK_B or ck["pos_rows"] != N_TOK_B:', "reader"),
    off(READER, "the checkpoint may come from another argv", 'if ck["argv_sha"] != sm.get("argv_sha"):', "reader"),
    off(READER, "the checkpoint may be at another step", 'if ck["step"] != sm.get("steps"):', "reader"),
    off(READER, "the step count may differ from B's", 'if sm.get("steps") != rsm.get("steps"):', "reader"),
    off(READER, "the window set may differ from B's", 'if sm.get("n_windows") != rsm.get("n_windows"):', "reader"),
    off(READER, "the run may not have withheld the holdout", 'if sm.get("excluded_sha256") != PGT.HOLDOUT_SHA:', "reader"),
    off(READER, "the run may train on another clip list", 'if sm.get("train_sha256") != rsm.get("train_sha256"):', "reader"),
    off(READER, "the run need not be done", 'if sm.get("done") is not True:', "reader"),
    off(READER, "B's checkpoint need not be the banked file", "if have != want:", "reader"),
    off(READER, "the scored checkpoint need not be the trained one", "if not (a1 and a2) or a1 != a2:", "reader"),
    off(READER, "the pulled checkpoint need not be the measured one", "if a1 != loc:", "reader"),
    off(READER, "an eval may have scored another checkpoint", 'if len(parts) < 2 or parts[-2] != x_dir(s) or parts[-1] != "ckpt_last.pt":', "reader"),
    off(READER, "an eval may have built the full interface", 'if js.get("o_interface") != LEVER_VAL:', "reader"),
    off(READER, "B's eval may have built the cut", 'if os.path.exists(p) and _jload(p).get("o_interface", "full") != "full":', "reader"),
    off(READER, "the grid's window and episode counts are not checked", "if (n, ne) != (a.n_windows, a.n_episodes):", "reader"),
    off(READER, "the arms may be on different grids", "if not np.array_equal(np.asarray(eid_x), np.asarray(eid)):", "reader"),
    off(READER, "a missing decision family is dropped", "if not ej.get(fam):", "reader"),
    M(READER, "the known values are not checked", "known = PGT.known_values(base_st, F, (0, 1))", "known = {}", "reader"),
    M(READER, "the identity control is not run", "idc = identity_control_b(pairfn, {s: Zb[(ARM_B, s)] for s in seeds}, pw, ctx, a.n_boot, WC, seeds, famfn)", "idc = {}", "reader"),
    off(READER, "a stratum under 10 episodes gets an interval", "if blk[\"n_episodes\"] < KR.MIN_STRATUM_EPISODES:", "reader"),
    M(READER, "the strata sign is flipped", '"point": float(b.mean() - x.mean()), "interval"', '"point": float(x.mean() - b.mean()), "interval"', "reader"),
    M(READER, "the committed next step of NOT ESTABLISHED is altered", "the interface is the defect. Run the lever MIXO names", "the interface is not the defect. Run the lever MIXO names", "reader"),
    M(READER, "the committed meaning of FLAT BETTER is altered", "at own codes the interface HURTS", "at own codes the interface HELPS", "reader"),
    M(READER, "the arm name is wrong", 'ARM_X = "M2b-flat"', 'ARM_X = "M2b-flt"', "reader"),
    M(READER, "an eval suffix is dropped", 'X_SUFFIXES = ("", "-tacabs", "-tacorc", "-tacorc-all-latcons")', 'X_SUFFIXES = ("", "-tacabs", "-tacorc")', "reader"),
    M(READER, "B's seed-0 checkpoint sha is wrong in the reader", "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8", "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f9", "reader"),
    M(READER, "B's schedule is wrong in the reader", "B_P_GT = (1.0, 0.2) ", "B_P_GT = (1.0, 0.0) ", "reader"),
    # =========================================================================================================== the queue
    M(QUEUE, "gateflat does not pass --o-interface cut", '--arm M2b --o-interface cut --out "%RUNSM%\\gate_M2b-flat"', '--arm M2b --out "%RUNSM%\\gate_M2b-flat"', "queue"),
    M(QUEUE, "gateflat says full", '--arm M2b --o-interface cut --out "%RUNSM%\\gate_M2b-flat"', '--arm M2b --o-interface full --out "%RUNSM%\\gate_M2b-flat"', "queue"),
    M(QUEUE, "trainflat says full", "--grid pool4x10 --o-interface cut --out", "--grid pool4x10 --o-interface full --out", "queue"),
    M(QUEUE, "trainflat trains another arm", '"%TRAIN%" --arm M2b --seed %~2 --grid pool4x10 --o-interface cut', '"%TRAIN%" --arm M1 --seed %~2 --grid pool4x10 --o-interface cut', "queue"),
    M(QUEUE, "trainflat moves a second lever", "--grid pool4x10 --o-interface cut --out", "--grid pool4x10 --hist 10 --o-interface cut --out", "queue"),
    M(QUEUE, "the all-latcons oracle drops its alpha", _evblock(3), _evblock(3).replace(" --tac-alpha 1 %EC%", " %EC%"), "queue"),
    M(QUEUE, "the GT-code oracle scores B's checkpoint", _evblock(2), _evblock(2).replace('--ckpt "%OUT%\\ckpt_last.pt" --tag %TG%-tacorc --tac oracle', '--ckpt "%RUNSM%\\M2b_s%~2\\ckpt_last.pt" --tag %TG%-tacorc --tac oracle'), "queue"),
    M(QUEUE, "the absmask eval is an oracle", _evblock(1), _evblock(1).replace("--tac absmask", "--tac oracle"), "queue"),
    M(QUEUE, "the standard eval drops --families", _evblock(0), _evblock(0).replace(" --project --families", " --project"), "queue"),
    M(QUEUE, "the dry run does not print the all-latcons eval", _DRY_TAIL, _DRY_TAIL.replace("echo %E3% & exit /b 0)", "exit /b 0)"), "queue"),
    M(QUEUE, "the eval stage does not refuse an unfinished run", _DRY_TAIL, _DRY_TAIL.replace('if not exist "%OUT%\\summary.json" (echo REFUSED: %TG% has not finished training & exit /b 2)', "rem"), "queue"),
    M(QUEUE, "a failed eval does not stop the stage", f"(echo [FAIL] eval %TG%-tacorc-all-latcons - see %EV%\\eval_%TG%-tacorc-all-latcons.log & exit /b 1){NL}echo [ok] eval %TG%-tacorc-all-latcons{NL}:evalflat_4",
      f"(echo [FAIL] eval %TG%-tacorc-all-latcons - see %EV%\\eval_%TG%-tacorc-all-latcons.log & exit /b 0){NL}echo [ok] eval %TG%-tacorc-all-latcons{NL}:evalflat_4", "queue"),
    M(QUEUE, "no gateflat dispatch", 'if /i "%~1"=="gateflat" goto :gateflat', "rem no gateflat", "queue"),
    M(QUEUE, "no trainflat dispatch", 'if /i "%~1"=="trainflat" goto :trainflat', "rem no trainflat", "queue"),
    M(QUEUE, "no evalflat dispatch", 'if /i "%~1"=="evalflat" goto :evalflat', "rem no evalflat", "queue"),
    M(QUEUE, "trainflat runs in B's directory", f'if "%~2"=="" (echo REFUSED: trainflat needs a SEED & exit /b 2){NL}set "OUT=%RUNSM%\\M2b_s%~2_flat"',
      f'if "%~2"=="" (echo REFUSED: trainflat needs a SEED & exit /b 2){NL}set "OUT=%RUNSM%\\M2b_s%~2"', "queue"),
    M(QUEUE, "trainflat does not skip a finished run", 'if exist "%OUT%\\summary.json" (echo [skip] M2b_s%~2_flat & exit /b 0)', "rem", "queue"),
    M(QUEUE, "a failed training does not stop the stage", "(echo [FAIL] train M2b_s%~2_flat - see %OUT%\\train.log & exit /b 1)", "(echo [FAIL] train M2b_s%~2_flat - see %OUT%\\train.log & exit /b 0)", "queue"),
    M(QUEUE, "a failed gate does not stop the stage", "(echo [FAIL] gate M2b-flat - see %RUNSM%\\gate_M2b-flat.log & exit /b 1)", "(echo [FAIL] gate M2b-flat - see %RUNSM%\\gate_M2b-flat.log & exit /b 0)", "queue"),
    M(QUEUE, "an existing eval is not skipped (absmask)", 'if exist "%EV%\\eval_%TG%-tacabs.npz" (echo [skip] eval %TG%-tacabs & goto :evalflat_2)', "rem", "queue"),
    M(QUEUE, "the skip chain jumps over the all-latcons eval", "(echo [skip] eval %TG%-tacorc & goto :evalflat_3)", "(echo [skip] eval %TG%-tacorc & goto :evalflat_4)", "queue"),
    M(QUEUE, "an old stage is edited (a blank line after evalmixo_7)", f":evalmixo_7{NL}exit /b 0{NL}:gateflat", f":evalmixo_7{NL}exit /b 0{NL}{NL}:gateflat", "queue"),
    M(QUEUE, "the queue ends with a bare LF", f":evalflat_4{NL}exit /b 0", ":evalflat_4\nexit /b 0", "queue"),
    # =========================================================================================================== the chain (the predecessor predicate included)
    M(CHAIN, "box 2's M2b_s0 checkpoint sha256 is wrong in the chain", "CKPT_SHA=(601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8 ", "CKPT_SHA=(601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f9 ", "chain"),
    M(CHAIN, "box 2's M2b_s1 checkpoint sha256 is wrong in the chain", " 6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6)", " 6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e7)", "chain"),
    M(CHAIN, "the chain waits for another predecessor", "AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_mixo.out}", "AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_kind.out}", "chain"),
    M(CHAIN, "the chain reads another predecessor's guard log", "AFTER_GUARD=${AFTER_GUARD:-$L/v8fmixo_guard.out}", "AFTER_GUARD=${AFTER_GUARD:-$L/v8fkind_guard.out}", "chain"),
    M(CHAIN, "the predicate does not look for DONE", 'if grep -a -q "^$AFTER_DONE" "$out" 2>/dev/null; then', "if false; then", "chain"),
    M(CHAIN, "the predicate holds on a DONE", 'echo "PROCEED the predecessor ended: $(grep -a "^$AFTER_DONE" "$out" | tail -1 | tr -d \'\\r\' | cut -c1-160)"; return 0; fi',
      'echo "HOLD the predecessor ended: $(grep -a "^$AFTER_DONE" "$out" | tail -1 | tr -d \'\\r\' | cut -c1-160)"; return 1; fi', "chain"),
    M(CHAIN, "the predicate proceeds on a stop it has never seen", 'echo "HOLD a stop that is not recognised as verdict- or run-level (fail closed): $msg"; return 1', 'echo "PROCEED a stop that is not recognised: $msg"; return 0', "chain"),
    M(CHAIN, "the predicate ignores box-level evidence", "ev=$(grep -a -h -E '^(ZZGUARD-STOP|ZZCRASH|ZZUNREACHABLE|ZZTIMEOUT)'", "ev=$(grep -a -h -E '^(ZZNEVERPRINTED)'", "chain"),
    M(CHAIN, "the predicate does not read the guard log", "${guard:+\"$guard\"} 2>/dev/null | head -1", "2>/dev/null | head -1", "chain"),
    M(CHAIN, "the predicate takes a clean exit for a failed run", "RUN_LEVEL_RE='^[A-Za-z0-9_.-]+ rc=[1-9][0-9]*$'", "RUN_LEVEL_RE='^[A-Za-z0-9_.-]+ rc=[0-9]*$'", "chain"),
    M(CHAIN, "the predicate takes any wait result for a run-level stop", "RUN_LEVEL_RE='^[A-Za-z0-9_.-]+ rc=[1-9][0-9]*$'", "RUN_LEVEL_RE='^.* (rc|wait)=[0-9]+.*$'", "chain"),
    M(CHAIN, "the predicate does not know the verdict-level stop", "VERDICT_LEVEL_RE='^[a-z0-9_]+ read( [A-Za-z0-9_]+)?$'", "VERDICT_LEVEL_RE='^NEVER$'", "chain"),
    M(CHAIN, "the predicate does not know the gate-level stop", "GATE_LEVEL_RE='^gate [A-Za-z0-9_]+ assertions$'", "GATE_LEVEL_RE='^NEVER$'", "chain"),
    M(CHAIN, "the predicate proceeds on a stop that merely mentions a read", "VERDICT_LEVEL_RE='^[a-z0-9_]+ read( [A-Za-z0-9_]+)?$'", "VERDICT_LEVEL_RE='read'", "chain"),
    M(CHAIN, "a held chain does not stop", 'if [ $r = 1 ]; then echo "ZZV8FMINI-STOP the predecessor\'s end does not clear the box: $d"; exit 1; fi', "if [ $r = 1 ]; then break; fi", "chain"),
    M(CHAIN, "the busy marker is not taken atomically", '( set -C; echo "v8F-mini PREREG_V8F_FLAT', '( echo "v8F-mini PREREG_V8F_FLAT', "chain"),
    M(CHAIN, "the gate stage queues another stage", "queue gateflat 120 gateflat", "queue gateflat 120 trainflat 0", "chain"),
    M(CHAIN, "a failed check-inputs does not stop the chain", '|| stop "the reader cannot use the M2b side (check-inputs)"', "|| true", "chain"),
    M(CHAIN, "a failed gate assertion does not stop the chain", "<<'EOF' || stop \"gate FLAT assertions\"", "<<'EOF' || true", "chain"),
    M(CHAIN, "a failed token write does not stop the chain", "<<'EOF' || stop \"write the gate token\"", "<<'EOF' || true", "chain"),
    M(CHAIN, "the checkpoint is not re-hashed after the evals", "  ckptsha_x $s after_evals\n", "  true\n", "chain"),
    M(CHAIN, "B's checkpoints are not asserted", 'rm -f $CK\nckptsha_b "before the gate"\n', "rm -f $CK\n", "chain"),
    M(CHAIN, "box 2's M2b checkpoint may differ from the banked one", '[ "$g" = "${CKPT_SHA[$s]}" ] || stop', '[ "$g" = "${CKPT_SHA[$s]}" ] || true #', "chain"),
    M(CHAIN, "the standard eval is not pulled", 'for f in "" $X_SUFFIXES; do', "for f in $X_SUFFIXES; do", "chain"),
    M(CHAIN, "the GT-code oracle is not pulled", 'X_SUFFIXES="-tacabs -tacorc -tacorc-all-latcons"', 'X_SUFFIXES="-tacabs -tacorc-all-latcons"', "chain"),
    M(CHAIN, "seed 1 is not run", "runone 1\n", "true\n", "chain"),
    M(CHAIN, "seed 1 is conditional", "runone 1\n", "[ -f /nonexistent ] && runone 1\n", "chain"),
    M(CHAIN, "the read runs before seed 1", "runone 0\n", "runone 0\nflatread flat_read.json\n", "chain"),
    M(CHAIN, "stop does not release the busy marker", 'echo "" > $CUR; rm -f $B/BOX2_BUSY.txt; exit 1; }', 'echo "" > $CUR; exit 1; }', "chain"),
    M(CHAIN, "finish does not release the busy marker", '  rm -f $B/BOX2_BUSY.txt\n  echo "ZZV8FMINI-DONE', '  echo "ZZV8FMINI-DONE', "chain"),
    M(CHAIN, "the training wait is shorter than the guard cap", "TRAIN_POLLS=1320;", "TRAIN_POLLS=600;", "chain"),
    M(CHAIN, "the step count asserted is not B's", "M2B_STEPS=19164", "M2B_STEPS=19000", "chain"),
    M(CHAIN, "the window count asserted is not B's", "M2B_WINDOWS=306566 ", "M2B_WINDOWS=306000 ", "chain"),
    M(CHAIN, "the reader is not given the checkpoint sha json", 'echo "--eval-dir $1 --runs-dir $2 --ckpt-sha $3 --out $4"', 'echo "--eval-dir $1 --runs-dir $2 --out $4"', "chain"),
    M(CHAIN, "the guard's markers are not checked", "grep -qF \"'^(ZZV8FMINI|\" $B/v8f_guard.sh || stop \"v8f_guard.sh does not know this chain's markers (anchored): it would never exit\"", "true", "chain"),
    M(CHAIN, "the tip is not checked for the lever", "| grep -q '\"o_interface\": cfg.o_interface' || stop \"the tip's built_levers does not record o_interface\"", "| true", "chain"),
    M(CHAIN, "the gate does not assert the lever", 'if lv.get("o_interface") != "cut": bad.append(', "if False: bad.append(", "chain"),
    M(CHAIN, "the gate does not assert parameter parity", 'if lv["n_params_trainable"] != bl["n_params_trainable"]: bad.append(', "if False: bad.append(", "chain"),
    M(CHAIN, "the gate does not assert the probe", 'if not isinstance(pr, dict) or pr.get("pass") is not True: bad.append(', "if False: bad.append(", "chain"),
    M(CHAIN, "the gate does not assert the dead set", 'if sc["live"].get("declared_dead") != ["e_code.weight"] or sc["live"].get("dead_params") != ["e_code.weight"]: bad.append(', "if False: bad.append(", "chain"),
    M(CHAIN, "the gate does not assert the built p_gt schedule", 'if (lv.get("p_gt_start"), lv.get("p_gt_end")) != (1.0, 0.2): bad.append(', "if False: bad.append(", "chain"),
    M(CHAIN, "the gate does not assert the queue line carries the lever", 'if toks.count("--o-interface") != 1 or toks[toks.index("--o-interface") + 1] != "cut": bad.append(', "if False: bad.append(", "chain"),
]

if LIST:
    for i, (p, n, e, g) in enumerate(MUT):
        print(i, p.name, "|", n)
    sys.exit(0)

orig = {p: p.read_bytes() for p in {m[0] for m in MUT}}
bad_anchor = 0
for path, name, edits, g in MUT:
    for a, b in edits:
        n = orig[path].count(a.encode("utf-8"))
        if n != 1:
            bad_anchor += 1
            print(f"ANCHOR FOUND {n}x (need exactly 1): [{path.name}] {name}\n    {a!r}", flush=True)
if bad_anchor:
    sys.exit(3)
print(f"{len(MUT)} mutants over {len(orig)} files, each anchor found exactly once", flush=True)
if CHECK:
    sys.exit(0)

BASE = Path("C:/Users/Admin/pt_flat_mut") / "mut"
BASE.parent.mkdir(exist_ok=True)
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4"}


def run(sel=None):
    cmd = [PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--basetemp", str(BASE), TEST]
    if sel:
        cmd[-1:-1] = ["-k", sel]
    shutil.rmtree(BASE, ignore_errors=True)
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=ENV, timeout=7200)
    shutil.rmtree(BASE, ignore_errors=True)
    lines = [x for x in r.stdout.strip().splitlines() if x.startswith(("FAILED", "ERROR")) or " passed" in x or " failed" in x or " error" in x]
    tail = " | ".join(x[:170] for x in (lines[-2:] if lines else [(r.stdout.strip().splitlines() or [r.stderr.strip()[-170:]])[-1]]))
    return r.returncode, tail


def restore():
    for p, b in orig.items():
        p.write_bytes(b)


rc, tail = run()
print("baseline:", rc, tail, flush=True)
assert rc == 0, "the unmutated tree must be green"
red, survived, ran = 0, [], 0
try:
    for i, (path, name, edits, group) in enumerate(MUT):
        if i < FROM or i >= TO or (ONLY and ONLY not in name and ONLY not in path.name):
            continue
        ran += 1
        src = orig[path]
        for a, b in edits:
            src = src.replace(a.encode("utf-8"), b.encode("utf-8"), 1)
        path.write_bytes(src)
        try:
            rc, tail = run(SEL.get(group))
            where = SEL.get(group) and f"[{group} group]"
            if rc == 0 and SEL.get(group):
                rc, tail = run()                                   # survived its group: the whole file decides
                where = "[whole file]"
        finally:
            path.write_bytes(orig[path])
        red += rc != 0
        if rc == 0:
            survived.append(name)
        print(f"{'RED     ' if rc else 'SURVIVED'} #{i} [{path.name}] {name} {where or ''}: {tail}", flush=True)
finally:
    restore()
rc, tail = run()
print("restored:", rc, tail, "| byte-identical:", all(p.read_bytes() == b for p, b in orig.items()))
print(f"{red}/{ran} RED" + (f"  SURVIVED: {survived}" if survived else ""))
sys.exit(0 if red == ran and rc == 0 else 1)
