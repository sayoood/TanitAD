"""Mutation proof for PREREG_V8F_MIXO's instruments: every mutation must turn stack/tests/test_v8f_mixo.py RED; the unmutated tree must be GREEN before and after, and every file is
restored byte-identical.

    python mutate_mixo.py <repo root> <python exe> [--check] [--only <substring of a mutant's name>] [--list]

    --check   only verify that every anchor occurs exactly once (in the ORIGINAL bytes), then exit
    --only    run only the mutants whose name or file contains the substring (the baseline and the restore still run)

The mutants live in the model (the unit partition, the mask, the branch, whole's RNG stream, the built lever), the trainer (the flag, make_config, G-DVB, the probe), the reader (the
committed bars, every refusal, the dose chain, the decision table), the queue (CRLF, edited as BYTES) and the chain. Each mutation is one (or, where stated, two) exact substring(s)
replaced once; a mutation whose anchor is not found exactly once aborts the run.

SPEED: a mutant is first run against the group of tests that own its file (`-k`), with `-x`; a RED there is a RED of the suite. A mutant that survives its group is re-run against the WHOLE
file before it is called SURVIVED, so a too-narrow selector can never hide a live mutant, only cost time.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
PY = sys.argv[2]
CHECK = "--check" in sys.argv
LIST = "--list" in sys.argv
ONLY = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
PKG = ROOT / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
CODE = PKG / "code"
MODEL = ROOT / "stack" / "tanitad" / "models" / "v8f_mini.py"
TRAIN = ROOT / "stack" / "scripts" / "v8f_mini_train.py"
EVAL = ROOT / "stack" / "scripts" / "v8f_mini_eval.py"
READER = CODE / "v8f_mixo_read.py"
QUEUE = CODE / "v8f_mini_box2_queue.cmd"
CHAIN = CODE / "box2_ops" / "chain_v8f_mixo.sh"
TEST = "stack/tests/test_v8f_mixo.py"

SEL = {
    "model": "not (reader or queue or chain or check_inputs or dose or harm or known_values or committed)",
    "reader": "reader or dose or harm or check_inputs or known_values or committed",
    "queue": "queue",
    "chain": "chain",
}
NL = "\r\n"          # the queue is CRLF: its multi-line anchors say so


def M(path, name, a, b, group):
    return (path, name, [(a, b)], group)


def MM(path, name, edits, group):
    return (path, name, edits, group)


def off(path, name, anchor, group, repl="False"):
    """`<indent>if <cond>:` -> `<indent>if False:`  (the anchor is the whole `if ...:` line text without its newline)."""
    ind = anchor[:len(anchor) - len(anchor.lstrip())]
    kw = "elif" if anchor.lstrip().startswith("elif") else "if"
    return M(path, name, anchor, f"{ind}{kw} {repl}:", group)


MUT = [
    # =========================================================================================================== the model: the mask, the branch, whole's stream, the lever
    M(MODEL, "the mixing line is deleted: kind = O always reads T's own code", '                code = torch.where(self._code_mix_mask(b["lab_t"]["has"], p_gt, gen), self._code_gt(b["lab_t"]), code)\n',
      "                pass\n", "model"),
    M(MODEL, "the mixing is reversed: O reads T's own code where the draw says GT", 'p_gt, gen), self._code_gt(b["lab_t"]), code)', 'p_gt, gen), code, self._code_gt(b["lab_t"]))', "model"),
    M(MODEL, "kind draws ONE switch per sample (whole's granularity under the kind flag)", "draw = torch.rand(has.shape[0], N_CODE_UNITS, generator=gen, device=has.device) < p_gt",
      "draw = (torch.rand(has.shape[0], 1, generator=gen, device=has.device) < p_gt).expand(has.shape[0], N_CODE_UNITS)", "model"),
    M(MODEL, "kind draws ONE switch per unit for the whole batch (samples not independent)", "draw = torch.rand(has.shape[0], N_CODE_UNITS, generator=gen, device=has.device) < p_gt",
      "draw = (torch.rand(1, N_CODE_UNITS, generator=gen, device=has.device) < p_gt).expand(has.shape[0], N_CODE_UNITS)", "model"),
    M(MODEL, "kind is not gated by has: an unlabelled window may read the GT code", "return (draw & has[:, None])[:, unit]", "return draw[:, unit]", "model"),
    M(MODEL, "kind's p is inverted (the draw is > p_gt)", "generator=gen, device=has.device) < p_gt", "generator=gen, device=has.device) > p_gt", "model"),
    M(MODEL, "kind ignores p_gt (a fixed 0.5)", "generator=gen, device=has.device) < p_gt", "generator=gen, device=has.device) < 0.5", "model"),
    M(MODEL, "CODE_UNIT_OF is reversed: each index reads another unit's switch", "for i in range(CODE_DIM))\n\n\nLIMITS_MS", "for i in range(CODE_DIM))[::-1]\n\n\nLIMITS_MS", "model"),
    MM(MODEL, "a HOLE in the unit partition with the import guard weakened (the partition test must catch it)",
       [('CODE_MIX_UNITS = {"decisions": tuple(i for n in CODE_COMPOSITES["decisions"] for i in CODE_GROUPS[n]), **CODE_KINDS}',
         'CODE_MIX_UNITS = {"decisions": tuple(i for n in CODE_COMPOSITES["decisions"] for i in CODE_GROUPS[n])[1:], **CODE_KINDS}'),
        ("if sorted(i for idx in CODE_MIX_UNITS.values() for i in idx) != list(range(CODE_DIM)):", "if False:"),
        ("CODE_UNIT_OF = tuple(next(u for u, idx in enumerate(CODE_MIX_UNITS.values()) if i in idx) for i in range(CODE_DIM))",
         "CODE_UNIT_OF = tuple(next((u for u, idx in enumerate(CODE_MIX_UNITS.values()) if i in idx), 0) for i in range(CODE_DIM))")], "model"),
    MM(MODEL, "an OVERLAP in the unit partition with the import guard weakened",
       [('CODE_MIX_UNITS = {"decisions": tuple(i for n in CODE_COMPOSITES["decisions"] for i in CODE_GROUPS[n]), **CODE_KINDS}',
         'CODE_MIX_UNITS = {"decisions": tuple(i for n in CODE_COMPOSITES["decisions"] for i in CODE_GROUPS[n]) + (34,), **CODE_KINDS}'),
        ("if sorted(i for idx in CODE_MIX_UNITS.values() for i in idx) != list(range(CODE_DIM)):", "if False:")], "model"),
    M(MODEL, "the import guard is removed from a HOLE", 'CODE_MIX_UNITS = {"decisions": tuple(i for n in CODE_COMPOSITES["decisions"] for i in CODE_GROUPS[n]), **CODE_KINDS}\nN_CODE_UNITS',
      'CODE_MIX_UNITS = {"decisions": tuple(i for n in CODE_COMPOSITES["decisions"] for i in CODE_GROUPS[n])[1:], **CODE_KINDS}\nN_CODE_UNITS', "model"),
    M(MODEL, "whole draws an extra number first (whole's RNG stream moves)", '                use_t = (torch.rand(B, generator=gen, device=dev) < p_gt) & b["lab_t"]["has"]',
      '                _extra = torch.rand(1, generator=gen, device=dev)\n                use_t = (torch.rand(B, generator=gen, device=dev) < p_gt) & b["lab_t"]["has"]', "model"),
    M(MODEL, "whole's p is halved", '                use_t = (torch.rand(B, generator=gen, device=dev) < p_gt) & b["lab_t"]["has"]',
      '                use_t = (torch.rand(B, generator=gen, device=dev) < p_gt / 2) & b["lab_t"]["has"]', "model"),
    M(MODEL, "whole is not gated by has", '                use_t = (torch.rand(B, generator=gen, device=dev) < p_gt) & b["lab_t"]["has"]',
      "                use_t = (torch.rand(B, generator=gen, device=dev) < p_gt)", "model"),
    M(MODEL, "the branch is on the wrong value (whole runs the kind mask)", 'if cfg.code_mix == "kind":', 'if cfg.code_mix == "whole":', "model"),
    M(MODEL, "the config default is kind", '    code_mix: str = "whole"', '    code_mix: str = "kind"', "model"),
    M(MODEL, "built_levers does not record code_mix", '"cons_absent": cfg.cons_absent, "code_mix": cfg.code_mix,', '"cons_absent": cfg.cons_absent,', "model"),
    M(MODEL, "built_levers echoes 'whole' whatever was built", '"code_mix": cfg.code_mix,', '"code_mix": "whole",', "model"),
    M(MODEL, "the modes lack kind", 'CODE_MIX_MODES = ("whole", "kind")', 'CODE_MIX_MODES = ("whole",)', "model"),
    off(MODEL, "an unknown code_mix is accepted by the config", "        if self.code_mix not in CODE_MIX_MODES:", "model"),
    M(EVAL, "the evaluator grows a code_mix path", "import argparse\n", "import argparse  # code_mix\n", "model"),
    # =========================================================================================================== the trainer: flag, make_config, G-DVB, the probe
    M(TRAIN, "the flag has no declared-vs-built entry", '    "code_mix": "lever:code_mix",', "   ", "model"),
    M(TRAIN, "make_config drops the lever", 'cons_absent=lv["cons_absent"], code_mix=a.code_mix)', 'cons_absent=lv["cons_absent"])', "model"),
    off(TRAIN, "declared-vs-built does not compare the built code_mix", '    if built.get("code_mix") != a.code_mix:', "model"),
    off(TRAIN, "kind on a constant p_gt is not refused", '    if a.code_mix == "kind" and built["p_gt_start"] == built["p_gt_end"] and built["p_gt_start"] in (0.0, 1.0):', "model"),
    M(TRAIN, "the flag's default is kind", 'choices=M.CODE_MIX_MODES, default="whole",', 'choices=M.CODE_MIX_MODES, default="kind",', "model"),
    M(TRAIN, "the flag accepts any value", "choices=M.CODE_MIX_MODES, default=", "choices=None, default=", "model"),
    off(TRAIN, "the probe does not refuse a kind that read no partly-GT code", "        if n_partial == 0:", "model"),
    off(TRAIN, "the probe does not refuse a partly-GT code under whole", "    elif n_partial:", "model"),
    off(TRAIN, "the probe does not refuse a cell that is neither source", "        if other.any():", "model"),
    off(TRAIN, "the probe does not refuse a GT read on an unlabelled window", "        if (is_gt & ~has[:, None]).any():", "model"),
    off(TRAIN, "the probe does not refuse a unit whose two sources agree", "        if (same & has[:, None]).any():", "model"),
    off(TRAIN, "the probe does not refuse a unit share far from p_gt", "        if any(abs(r - p_gt) > 0.25 for r in rate) and n_seen >= 64:", "model", repl="False"),
    M(TRAIN, "the probe is faked (never reads the code O receives)", "        mix = code_mix_probe(model, probe, cfg)",
      '        mix = {"code_mix": cfg.code_mix, "pass": True, "n_partial": 1, "n_draws_samples": 1}', "model"),
    M(TRAIN, "the probe's refusal does not stop the run", '        raise SystemExit(f"ZZDVB-REFUSEDZZ declared-vs-built: {e}")', '        mix = {"code_mix": cfg.code_mix, "pass": False, "error": str(e)}', "model"),
    M(TRAIN, "selfcheck.json omits the probe", '"code_mix_probe": mix, ', "", "model"),
    # =========================================================================================================== the reader: bars, decision table
    M(READER, "the 2F bar is 0.2400", "BAR_2F = 0.2415 ", "BAR_2F = 0.2400 ", "reader"),
    M(READER, "the rise bar is 0.08", "RISE_BAR = 0.10 ", "RISE_BAR = 0.08 ", "reader"),
    M(READER, "B's committed r(0.5) at seed 1 is 0.25", "R_B_COMMITTED = {0: 0.36, 1: 0.21}", "R_B_COMMITTED = {0: 0.36, 1: 0.25}", "reader"),
    M(READER, "B's committed r(0.5) at seed 0 is 0.30", "R_B_COMMITTED = {0: 0.36, 1: 0.21}", "R_B_COMMITTED = {0: 0.30, 1: 0.21}", "reader"),
    M(READER, "the r tolerance is huge", "R_B_TOL = 0.005 ", "R_B_TOL = 0.5 ", "reader"),
    M(READER, "the bootstrap B defaults to 1,000", "\nB = 2000\n", "\nB = 1000\n", "reader"),
    M(READER, "the PRIMARY need not be separated", 'p["point"] <= -BAR_2F and p["separated"] and p["point"] < 0', 'p["point"] <= -BAR_2F and p["point"] < 0', "reader"),
    M(READER, "a PRIMARY exactly at -2F does not clear", 'p["point"] <= -BAR_2F and', 'p["point"] < -BAR_2F and', "reader"),
    M(READER, "a rise exactly at the bar does not clear", "rise >= RISE_BAR", "rise > RISE_BAR", "reader"),
    M(READER, "the rise is read against B's MEASURED r, not the committed one", 'else round(xs["r"] - rb, 12)', 'else round(xs["r"] - bs["r"], 12)', "reader"),
    M(READER, "an undefined r_X reads 0.0 (a fall, not 'no rise')", 'float(dh["point"] / d1["point"]) if ok else None', 'float(dh["point"] / d1["point"]) if ok else 0.0', "reader"),
    M(READER, "r from the 4-dp rounded deltas", 'float(dh["point"] / d1["point"]) if ok else None', 'float(dh["delta"] / d1["delta"]) if ok else None', "reader"),
    M(READER, "r(0.5) is taken against the full oracle, not T's own code", 'dh = PGT.paired(pf, dose["half"], dose["pred"], eid, n_boot)', 'dh = PGT.paired(pf, dose["half"], dose["full"], eid, n_boot)', "reader"),
    M(READER, "d_cons(1) is taken against the half oracle", 'd1 = PGT.paired(pf, dose["full"], dose["pred"], eid, n_boot)', 'd1 = PGT.paired(pf, dose["full"], dose["half"], eid, n_boot)', "reader"),
    M(READER, "r is defined where the full oracle does not help", 'ok = d1["point"] < 0', 'ok = d1["point"] != 0', "reader"),
    M(READER, "verdict: CONTINUE always (a seed-0 miss does not stop)", "        if passed[0]:\n            return \"CONTINUE\", None, \"n/a\"", "        if True:\n            return \"CONTINUE\", None, \"n/a\"", "reader"),
    M(READER, "verdict: the final PASS needs only seed 0", '    if all(passed.values()):\n        return None, "PASS", "n/a"', '    if passed[0]:\n        return None, "PASS", "n/a"', "reader"),
    M(READER, "verdict: the final PASS needs either seed", '    if all(passed.values()):\n        return None, "PASS", "n/a"', '    if any(passed.values()):\n        return None, "PASS", "n/a"', "reader"),
    M(READER, "verdict: a miss is O-LEARNED if ANY missed seed learned", 'learned = all(rows[f"s{s}"]["judge"]["dm"]["learned"] for s in missed)', 'learned = any(rows[f"s{s}"]["judge"]["dm"]["learned"] for s in missed)', "reader"),
    M(READER, "verdict: the seed-0 split is swapped", '"FAIL-O-LEARNED" if rows["s0"]["judge"]["dm"]["learned"] else "FAIL-O-DID-NOT-LEARN"', '"FAIL-O-DID-NOT-LEARN" if rows["s0"]["judge"]["dm"]["learned"] else "FAIL-O-LEARNED"', "reader"),
    M(READER, "verdict: the extension cell is not labelled", '("defined" if missed == [0] else "extension")', '"defined"', "reader"),
    off(READER, "the extension caveat is not appended", '    if cell == "extension":', "reader"),
    M(READER, "stage s0 reads both seeds", 'seeds = (0,) if a.stage == "s0" else (0, 1)', "seeds = (0, 1)", "reader"),
    off(READER, "a final read after a seed-0 miss is allowed", '    if a.stage == "final" and not rows["s0"]["judge"]["passed"]:', "reader"),
    M(READER, "a next-step text is altered (O-DID-NOT-LEARN)", "(an architecture change: PI go)\"}", "(an architecture change: PI no)\"}", "reader"),
    M(READER, "a next-step text is altered (PASS)", '"PASS": "X becomes the base; KIND is re-read on X"', '"PASS": "X becomes the base"', "reader"),
    # ---- the refusals (reader)
    off(READER, "the family check is dropped", "    if miss:", "reader"),
    off(READER, "the tacpred reproduction guard is dropped", "    if dev > TOL_REPRO:", "reader"),
    off(READER, "the B r(0.5) reproduction is dropped (read)", '    if bs["r"] is None or abs(bs["r"] - rb) > R_B_TOL:', "reader"),
    off(READER, "the B r(0.5) reproduction is dropped (check-inputs)", '        if st["r"] is None or abs(st["r"] - R_B_COMMITTED[s]) > R_B_TOL:', "reader"),
    MM(READER, "the dose identity cells are not enforced (read)",
       [("            if not (v[\"delta\"] == 0.0 and v[\"lo\"] == 0.0 and v[\"hi\"] == 0.0 and v[\"point\"] == 0.0):\n                refuse(f\"IDENTITY CONTROL FAILED: B_s{s} - B_s{s} reads {v['delta']} [{v['lo']}, {v['hi']}] on {name}",
         "            if False:\n                refuse(f\"IDENTITY CONTROL FAILED: B_s{s} - B_s{s} reads {v['delta']} [{v['lo']}, {v['hi']}] on {name}")], "reader"),
    MM(READER, "the dose identity cells are not enforced (check-inputs)",
       [("            if not (v[\"delta\"] == 0.0 and v[\"lo\"] == 0.0 and v[\"hi\"] == 0.0 and v[\"point\"] == 0.0):\n                refuse(f\"IDENTITY CONTROL FAILED: B_s{s} - B_s{s} reads {v['delta']} [{v['lo']}, {v['hi']}] on dose_{name}",
         "            if False:\n                refuse(f\"IDENTITY CONTROL FAILED: B_s{s} - B_s{s} reads {v['delta']} [{v['lo']}, {v['hi']}] on dose_{name}")], "reader"),
    off(READER, "the checkpoint sha256 of B is not compared", "        if have != want:", "reader"),
    off(READER, "X's trained == after-evals sha256 is not compared", "        if not (a1 and a2) or a1 != a2:", "reader"),
    off(READER, "the pulled checkpoint is not compared to the measured sha256", "        if a1 != loc:", "reader"),
    M(READER, "the checkpoint sha256 json is not read", "    sha = check_ckpt_sha(a.ckpt_sha, a.runs_dir, seeds)", "    sha = {}", "reader"),
    off(READER, "the scored checkpoint of an eval is not checked", '            if len(parts) < 2 or parts[-2] != x_dir(s) or parts[-1] != "ckpt_last.pt":', "reader"),
    M(READER, "the eval ckpt check is not called", "    check_eval_ckpts(a.eval_dir, seeds)", "    pass", "reader"),
    off(READER, "the committed window / episode counts are not checked", "    if (n, ne) != (a.n_windows, a.n_episodes):", "reader"),
    off(READER, "the X eval grid is not compared to B's", "    if not np.array_equal(np.asarray(eid_x), np.asarray(eid)):", "reader"),
    off(READER, "a dose file's grid is not compared", '        if not np.array_equal(np.asarray(z["eid"]), np.asarray(eid)):', "reader"),
    off(READER, "a non-finite dose ade6 is accepted", "        if not np.isfinite(ade).all():", "reader"),
    M(READER, "the dose manipulation's groups are not checked", '(m.get("mode") == "oracle" and m.get("groups") == want_groups and m.get("n_code_dims_replaced") == 26 and a_ok)',
      '(m.get("mode") == "oracle" and m.get("n_code_dims_replaced") == 26 and a_ok)', "reader"),
    M(READER, "the dose manipulation's alpha is not checked", "    a_ok = (alpha_optional and alpha is None) or (alpha is not None and abs(float(alpha) - want_alpha) <= 1e-9)", "    a_ok = True", "reader"),
    off(READER, "the tacpred file may carry a manipulation", '            if js.get("tac", "pred") != "pred" or js.get("tac_manipulation"):', "reader"),
    off(READER, "the oracle arms may disagree about the windows without a v9 row", "    if len(fbs) == 2 and not np.array_equal(fbs[0], fbs[1]):", "reader"),
    off(READER, "the K = 10 lead windows are not compared", '    if not np.array_equal(np.asarray(pw["is_lead"], bool), ctx["lead"]):', "reader"),
    M(READER, "the GT-code check is not run", "    PGT.check_gt_codes(Zrel, seeds, WC)", "    pass", "reader"),
    M(READER, "the train log is not required", 'sm.get("steps"), p0, p1, True)', 'sm.get("steps"), p0, p1, False)', "reader"),
    off(READER, "the run need not be done", '        if sm.get("done") is not True:', "reader"),
    off(READER, "the step count may differ from B's", '        if sm.get("steps") != rsm.get("steps"):', "reader"),
    off(READER, "the window count may differ from B's", '        if sm.get("n_windows") != rsm.get("n_windows"):', "reader"),
    off(READER, "the clip list may differ from B's", '        if sm.get("train_sha256") != rsm.get("train_sha256"):', "reader"),
    off(READER, "the RA-1 holdout need not be withheld", '        if sm.get("excluded_sha256") != PGT.HOLDOUT_SHA:', "reader"),
    off(READER, "X's argv may differ from B's besides the lever", "        if fx != fb:", "reader"),
    off(READER, "the lever need be named exactly once", "        if argv.count(LEVER_FLAG) != 1 or fl.get(LEVER_FLAG) != LEVER_VAL:", "reader"),
    off(READER, "B's argv may name the lever", "        if LEVER_FLAG in rfl:", "reader"),
    off(READER, "B need not be a whole-code run", '        if (rsm.get("levers") or {}).get("code_mix", "whole") != "whole":', "reader"),
    off(READER, "the built code_mix need not be kind (record)", '            elif lg.get("code_mix") != LEVER_VAL:', "reader"),
    off(READER, "the built p_gt may differ from B's", "                elif not PGT._same_float(lg[key], want):", "reader"),
    M(READER, "the built levers may differ from B's", 'diff = sorted(k for k in set(lg) & set(lb) if k not in _SKIP_LEVERS_X and lg[k] != lb[k])', "diff = []", "reader"),
    off(READER, "the parameter groups may differ from B's", "            if px.get(k, 0) != pb.get(k, 0):", "reader"),
    off(READER, "the trainable parameter count may differ from B's", '    if (lv_x.get("n_params_trainable") or 0) != (lv_b.get("n_params_trainable") or 0):', "reader"),
    off(READER, "the checkpoint need not be built with kind", '        if ck["code_mix"] != LEVER_VAL:', "reader"),
    off(READER, "the checkpoint's hmax may differ from B's", '        if ck["hmax"] != hm_b:', "reader"),
    off(READER, "the checkpoint grid may differ from B's", '        if ck["grid"] != B_GRID or ck["n_tok"] != N_TOK_B or ck["pos_rows"] != N_TOK_B:', "reader"),
    off(READER, "the checkpoint may be from another argv", '        if ck["argv_sha"] != sm.get("argv_sha"):', "reader"),
    off(READER, "the checkpoint may be at another step", '        if ck["step"] != sm.get("steps"):', "reader"),
    # ---- the gate token (reader)
    off(READER, "the PASS token need not carry declared_vs_built PASS", '    if sc.get("declared_vs_built") != "PASS":', "reader"),
    off(READER, "the PASS token need not carry G-LIVE with a zero control", '    if lv_live.get("pass") is not True or lv_live.get("control_prior_grad_sum") != 0.0:', "reader"),
    M(READER, "the PASS token need not carry a passing probe", '    if not isinstance(pr, dict) or pr.get("pass") is not True:', "    if not isinstance(pr, dict):", "reader"),
    off(READER, "the probe need not have been run for kind", "        if pr.get(\"code_mix\") != LEVER_VAL:", "reader"),
    off(READER, "the probe need not have seen a partly-GT code", '        if not (isinstance(pr.get("n_partial"), int) and pr["n_partial"] > 0):', "reader"),
    off(READER, "the selfcheck need not be this run's", '    if sc.get("argv_sha") != sm.get("argv_sha"):', "reader"),
    # =========================================================================================================== the queue (CRLF: edited as bytes)
    M(QUEUE, "trainmixo does not pass --code-mix kind", '--grid pool4x10 --code-mix kind --out "%OUT%" %A%"', '--grid pool4x10 --out "%OUT%" %A%"', "queue"),
    M(QUEUE, "gatemixo does not pass --code-mix kind", '--arm M2b --code-mix kind --out "%RUNSM%\\gate_M2b-mixo"', '--arm M2b --out "%RUNSM%\\gate_M2b-mixo"', "queue"),
    M(QUEUE, "gatemixo says whole", '--arm M2b --code-mix kind --out "%RUNSM%\\gate_M2b-mixo"', '--arm M2b --code-mix whole --out "%RUNSM%\\gate_M2b-mixo"', "queue"),
    M(QUEUE, "trainmixo says whole", "--grid pool4x10 --code-mix kind --out", "--grid pool4x10 --code-mix whole --out", "queue"),
    M(QUEUE, "trainmixo trains another arm", '"%TRAIN%" --arm M2b --seed %~2 --grid pool4x10 --code-mix kind', '"%TRAIN%" --arm M1 --seed %~2 --grid pool4x10 --code-mix kind', "queue"),
    M(QUEUE, "the alpha-0.50 eval runs at alpha 0.75", "--tag %TG%-tacorc-constraints-a50 --tac oracle --tac-groups constraints --tac-alpha 0.50", "--tag %TG%-tacorc-constraints-a50 --tac oracle --tac-groups constraints --tac-alpha 0.75", "queue"),
    M(QUEUE, "the alpha-1 constraints eval runs at alpha 0.5", "--tag %TG%-tacorc-constraints --tac oracle --tac-groups constraints --tac-alpha 1 ", "--tag %TG%-tacorc-constraints --tac oracle --tac-groups constraints --tac-alpha 0.5 ", "queue"),
    M(QUEUE, "the tacpred eval is an oracle", "--tag %TG%-tacpred --tac pred ", "--tag %TG%-tacpred --tac oracle ", "queue"),
    M(QUEUE, "the constraints oracle scores B's checkpoint", '--ckpt "%OUT%\\ckpt_last.pt" --tag %TG%-tacorc-constraints --tac oracle', '--ckpt "%RUNSM%\\M2b_s%~2\\ckpt_last.pt" --tag %TG%-tacorc-constraints --tac oracle', "queue"),
    M(QUEUE, "the all-latcons oracle drops its alpha", f"--tac-groups all-latcons --tac-alpha 1 %EC% %ER%\"{NL}set \"E4=", f"--tac-groups all-latcons %EC% %ER%\"{NL}set \"E4=", "queue"),
    M(QUEUE, "the standard eval drops --families", f"set \"TG=M2b-mixo_s%~2\"{NL}set \"E0=\"%PY%\" \"%EVAL%\" run --ckpt \"%OUT%\\ckpt_last.pt\" --tag %TG% %EC% %ER% --project --families\"",
      f"set \"TG=M2b-mixo_s%~2\"{NL}set \"E0=\"%PY%\" \"%EVAL%\" run --ckpt \"%OUT%\\ckpt_last.pt\" --tag %TG% %EC% %ER% --project\"", "queue"),
    M(QUEUE, "the dry run does not print the tacpred eval", "echo %E3% & echo %E4% & echo %E5%", "echo %E3% & echo %E5%", "queue"),
    M(QUEUE, "no evalmixo dispatch", 'if /i "%~1"=="evalmixo" goto :evalmixo', "rem no evalmixo", "queue"),
    M(QUEUE, "no trainmixo dispatch", 'if /i "%~1"=="trainmixo" goto :trainmixo', "rem no trainmixo", "queue"),
    M(QUEUE, "no gatemixo dispatch", 'if /i "%~1"=="gatemixo" goto :gatemixo', "rem no gatemixo", "queue"),
    M(QUEUE, "trainmixo runs in B's directory", f'if "%~2"=="" (echo REFUSED: trainmixo needs a SEED & exit /b 2){NL}set "OUT=%RUNSM%\\M2b_s%~2_mixo"',
      f'if "%~2"=="" (echo REFUSED: trainmixo needs a SEED & exit /b 2){NL}set "OUT=%RUNSM%\\M2b_s%~2"', "queue"),
    M(QUEUE, "trainmixo does not skip a finished run", 'if exist "%OUT%\\summary.json" (echo [skip] M2b_s%~2_mixo & exit /b 0)', "rem", "queue"),
    M(QUEUE, "a failed training does not stop the stage", "(echo [FAIL] train M2b_s%~2_mixo - see %OUT%\\train.log & exit /b 1)", "(echo [FAIL] train M2b_s%~2_mixo - see %OUT%\\train.log & exit /b 0)", "queue"),
    M(QUEUE, "a failed eval does not stop the stage", "(echo [FAIL] eval %TG%-tacorc-constraints-a50 - see %EV%\\eval_%TG%-tacorc-constraints-a50.log & exit /b 1)",
      "(echo [FAIL] eval %TG%-tacorc-constraints-a50 - see %EV%\\eval_%TG%-tacorc-constraints-a50.log & exit /b 0)", "queue"),
    M(QUEUE, "the eval stage does not refuse an unfinished run", f"echo %E6% & exit /b 0){NL}if not exist \"%OUT%\\summary.json\" (echo REFUSED: %TG% has not finished training & exit /b 2)",
      f"echo %E6% & exit /b 0){NL}rem", "queue"),
    M(QUEUE, "an existing eval is not skipped (tacpred)", 'if exist "%EV%\\eval_%TG%-tacpred.npz" (echo [skip] eval %TG%-tacpred & goto :evalmixo_5)', "rem", "queue"),
    M(QUEUE, "the skip chain jumps over the alpha-0.50 eval", "(echo [skip] eval %TG%-tacorc-constraints & goto :evalmixo_6)", "(echo [skip] eval %TG%-tacorc-constraints & goto :evalmixo_7)", "queue"),
    M(QUEUE, "an old stage is edited", f"echo [ok] compare{NL}exit /b 0", f"echo [ok] compared{NL}exit /b 0", "queue"),
    M(QUEUE, "the queue ends with a bare LF", f":evalmixo_7{NL}exit /b 0", ":evalmixo_7\nexit /b 0", "queue"),
    # =========================================================================================================== the chain
    M(CHAIN, "box 2's M2b_s0 checkpoint sha256 is wrong in the chain", "CKPT_SHA=(601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8 ", "CKPT_SHA=(601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f9 ", "chain"),
    M(CHAIN, "box 2's M2b_s1 checkpoint sha256 is wrong in the chain", " 6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6)", " 6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e7)", "chain"),
    M(CHAIN, "the chain waits for another predecessor", "AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_kind.out}", "AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_lora_ar2.out}", "chain"),
    M(CHAIN, "the busy marker is not taken atomically", '( set -C; echo "v8F-mini PREREG_V8F_MIXO', '( echo "v8F-mini PREREG_V8F_MIXO', "chain"),
    M(CHAIN, "the gate stage queues another stage", "queue gatemixo 120 gatemixo", "queue gatemixo 120 trainmixo 0", "chain"),
    M(CHAIN, "a failed check-inputs does not stop the chain", '|| stop "the reader cannot use the M2b side (check-inputs)"', "|| true", "chain"),
    M(CHAIN, "a failed gate assertion does not stop the chain", "<<'EOF' || stop \"gate X assertions\"", "<<'EOF' || true", "chain"),
    M(CHAIN, "a failed token write does not stop the chain", "<<'EOF' || stop \"write the gate token\"", "<<'EOF' || true", "chain"),
    M(CHAIN, "the checkpoint is not re-hashed after the evals", "  ckptsha_x $s after_evals\n", "  true\n", "chain"),
    M(CHAIN, "B's checkpoints are not asserted", 'rm -f $CK\nckptsha_b "before the gate"\n', "rm -f $CK\n", "chain"),
    M(CHAIN, "box 2's M2b checkpoint may differ from the banked one", '[ "$g" = "${CKPT_SHA[$s]}" ] || stop', '[ "$g" = "${CKPT_SHA[$s]}" ] || true #', "chain"),
    M(CHAIN, "a banked dose file is not staged (alpha 0.50)", "eval_M2b_s$s-tacorc-constraints-a50.npz eval_M2b_s$s-tacorc-constraints-a50.json; do", "eval_M2b_s$s-tacorc-constraints.npz eval_M2b_s$s-tacorc-constraints.json; do", "chain"),
    M(CHAIN, "the standard eval is not pulled", 'for f in "" $X_SUFFIXES; do', "for f in $X_SUFFIXES; do", "chain"),
    M(CHAIN, "the tacpred eval is not pulled", 'X_SUFFIXES="-tacabs -tacorc -tacorc-all-latcons -tacpred -tacorc-constraints -tacorc-constraints-a50"', 'X_SUFFIXES="-tacabs -tacorc -tacorc-all-latcons -tacorc-constraints -tacorc-constraints-a50"', "chain"),
    M(CHAIN, "seed 1 is trained after a seed-0 STOP", '[ "$(jget mixo_read_s0.json DECISION)" = CONTINUE ] || finish "$M1 -- seed 1 not trained (committed): $(jget mixo_read_s0.json next)"',
      '[ "$(jget mixo_read_s0.json DECISION)" = CONTINUE ] || true', "chain"),
    M(CHAIN, "the final read is run at stage s0", "mixoread final mixo_read.json", "mixoread s0 mixo_read.json", "chain"),
    M(CHAIN, "stop does not release the busy marker", 'echo "" > $CUR; rm -f $B/BOX2_BUSY.txt; exit 1; }', 'echo "" > $CUR; exit 1; }', "chain"),
    M(CHAIN, "finish does not release the busy marker", '  rm -f $B/BOX2_BUSY.txt\n  echo "ZZV8FMINI-DONE', '  echo "ZZV8FMINI-DONE', "chain"),
    M(CHAIN, "the training wait is shorter than the guard cap", "TRAIN_POLLS=1320;", "TRAIN_POLLS=600;", "chain"),
    M(CHAIN, "the step count asserted is not B's", "M2B_STEPS=19164", "M2B_STEPS=19000", "chain"),
    M(CHAIN, "the window count asserted is not B's", "M2B_WINDOWS=306566 ", "M2B_WINDOWS=306000 ", "chain"),
    M(CHAIN, "the reader is not given the checkpoint sha json", 'echo "--eval-dir $1 --runs-dir $2 --ckpt-sha $3 --out $4"', 'echo "--eval-dir $1 --runs-dir $2 --out $4"', "chain"),
    M(CHAIN, "the guard's markers are not checked", "grep -qF \"'^(ZZV8FMINI|\" $B/v8f_guard.sh || stop \"v8f_guard.sh does not know this chain's markers (anchored): it would never exit\"", "true", "chain"),
    M(CHAIN, "the remote tip is not checked for the lever", "git -C $MIRROR show \"$T:stack/tanitad/models/v8f_mini.py\" | grep -q '\"code_mix\": cfg.code_mix' || stop \"the tip's built_levers does not record code_mix\"", "true", "chain"),
    M(CHAIN, "the queue's gatemixo line is not asserted to carry the lever", 'if toks.count("--code-mix") != 1 or toks[toks.index("--code-mix") + 1] != "kind": bad.append(', "if False: bad.append(", "chain"),
    M(CHAIN, "the lever's parameter parity is not asserted", 'if lv["n_params_trainable"] != bl["n_params_trainable"]: bad.append(', "if False: bad.append(", "chain"),
    M(CHAIN, "the probe's partly-GT read is not asserted", 'if not (isinstance(pr.get("n_partial"), int) and pr["n_partial"] > 0): bad.append(f"the code_mix probe saw no PARTLY-GT code', 'if False: bad.append(f"the code_mix probe saw no PARTLY-GT code', "chain"),
    M(CHAIN, "the built p_gt schedule is not asserted", "if (lv.get(\"p_gt_start\"), lv.get(\"p_gt_end\")) != (1.0, 0.2): bad.append(", "if False: bad.append(", "chain"),
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

BASE = Path("C:/Users/Admin/pt_mixo") / "mut"
BASE.parent.mkdir(exist_ok=True)
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}


def run(sel=None):
    cmd = [PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--basetemp", str(BASE), TEST]
    if sel:
        cmd[-1:-1] = ["-k", sel]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=ENV, timeout=3600)
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
    for path, name, edits, group in MUT:
        if ONLY and ONLY not in name and ONLY not in path.name:
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
        print(f"{'RED     ' if rc else 'SURVIVED'} [{path.name}] {name} {where or ''}: {tail}", flush=True)
finally:
    restore()
rc, tail = run()
print("restored:", rc, tail, "| byte-identical:", all(p.read_bytes() == b for p, b in orig.items()))
print(f"{red}/{ran} RED" + (f"  SURVIVED: {survived}" if survived else ""))
sys.exit(0 if red == ran and rc == 0 else 1)
