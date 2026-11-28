"""Mutation proof for PREREG_V8F_GRID's instruments: every mutation must turn stack/tests/test_v8f_grid.py RED; the unmutated tree must be
GREEN before and after, and every mutated file is restored byte-identical (in a `finally`, then verified). Deletes nothing.

  python mutate_grid.py <repo root> <python exe> [--set reach|reader|queue|chain|cache|all] [--check-anchors] [--part k/n]

  reach   the lever's path: make_config / the gate / ClipStore / the checkpoint / the model's position table / the eval's grid read, and the two
          "silent scoring" mutants (a model that tolerates a feature/position-table size mismatch, a model that drops 120 of the 160 tokens)
  reader  code/v8f_grid_read.py: the base rule, the verdict logic, the PRIMARY / M-LAT bars, the refusals, the identity control
  queue   code/v8f_mini_box2_queue.cmd (:gridbase / gategrid / traingrid / evalgrid and their dispatch lines)
  chain   code/box2_ops/chain_v8f_grid.sh (order, timeouts, the gate-assertion heredoc, the pulls, the markers)
  cache   code/check_grid_cache.py
Anchors must occur EXACTLY ONCE (asserted); a mutant that is semantically equivalent to the original is not listed."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
PY = sys.argv[2]
WHICH = sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else "all"
MODEL = ROOT / "stack" / "tanitad" / "models" / "v8f_mini.py"
TRAIN = ROOT / "stack" / "scripts" / "v8f_mini_train.py"
EVAL = ROOT / "stack" / "scripts" / "v8f_mini_eval.py"
DATA = ROOT / "stack" / "tanitad" / "data" / "v8f_mini_data.py"
CODE = ROOT / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini" / "code"
READ, QUEUE, CHAIN, CACHE = CODE / "v8f_grid_read.py", CODE / "v8f_mini_box2_queue.cmd", CODE / "box2_ops" / "chain_v8f_grid.sh", CODE / "check_grid_cache.py"
TEST = "stack/tests/test_v8f_grid.py"

REACH = [
    # ---- argv -> make_config -> the gate
    (TRAIN, "make_config ignores --grid (always 40 tokens)", 'n_tok=40 if a.grid == "pool4x10" else 160,', "n_tok=40,"),
    (TRAIN, "make_config builds 80 tokens at 8x20", 'n_tok=40 if a.grid == "pool4x10" else 160,', 'n_tok=40 if a.grid == "pool4x10" else 80,'),
    (TRAIN, "the gate does not compare the built n_tok with the declared grid", '    if model.cfg.n_tok != (40 if a.grid == "pool4x10" else 160):\n', "    if False:\n"),
    # ---- the features the model trains on
    (TRAIN, "ClipStore is handed the 4x10 grid whatever --grid says", 'store = D.ClipStore(cache, "train", lru=lru, grid=a.grid,', 'store = D.ClipStore(cache, "train", lru=lru, grid="pool4x10",'),
    (DATA, "ClipStore ignores the grid it is given", "self.lru, self.grid, self._c = int(lru), grid, {}", 'self.lru, self.grid, self._c = int(lru), "pool4x10", {}'),
    (DATA, "ClipStore (yaw branch) reads pool4x10 whatever the grid", '"feats": o[self.grid].to(torch.float16).numpy(), "pose": pg, "kin3": kin3,',
     '"feats": o["pool4x10"].to(torch.float16).numpy(), "pose": pg, "kin3": kin3,'),
    (DATA, "ClipStore (steer branch) reads pool4x10 whatever the grid", '"feats": o[self.grid].to(torch.float16).numpy(), "pose": pg, "kin3": o["kin3"]',
     '"feats": o["pool4x10"].to(torch.float16).numpy(), "pose": pg, "kin3": o["kin3"]'),
    (TRAIN, "the pipeline sizing ignores the token count", 'J["n"].max() * cfg.n_tok * 1024 * 2', 'J["n"].max() * 40 * 1024 * 2'),
    # ---- the built model
    (MODEL, "the position table has 40 rows whatever n_tok is", "self.pos = nn.Parameter(torch.randn(cfg.n_tok, d) * 0.02)", "self.pos = nn.Parameter(torch.randn(40, d) * 0.02)"),
    (MODEL, "the position table is in no optimiser group", "[self.adapter, self.pos, self.time]", "[self.adapter, self.time]"),
    (MODEL, "SILENT: the model drops 120 of the 160 tokens (rows 40-159 never train)", 'adapt = self.adapter(b["feats"].float()) + self.pos[None, None]',
     'adapt = (self.adapter(b["feats"].float()) + self.pos[None, None])[:, :, :40]'),
    (MODEL, "SILENT: the model tolerates a feature / position-table size mismatch (the eval would score a wrong pair)",
     'adapt = self.adapter(b["feats"].float()) + self.pos[None, None]', 'adapt = self.adapter(b["feats"].float()) + self.pos[None, None][:, :, :b["feats"].shape[2]]'),
    # ---- the checkpoint and the eval
    (TRAIN, "the checkpoint records pool4x10 whatever was trained", '"grid": a.grid, "arm": a.arm,', '"grid": "pool4x10", "arm": a.arm,'),
    (EVAL, "the eval reads pool4x10 whatever the checkpoint says", 'grid=ck.get("grid", "pool4x10")', 'grid="pool4x10"'),
    (EVAL, "the eval's default for a checkpoint without a grid key is 8x20", 'grid=ck.get("grid", "pool4x10")', 'grid=ck.get("grid", "pool8x20")'),
]
R_ = READ
READER = [
    # ---- the base rule
    (R_, "base: SUCCESS picks M2b (swapped)", 'base = "M2b-pgt" if v == "SUCCESS" else "M2b"', 'base = "M2b" if v == "SUCCESS" else "M2b-pgt"'),
    (R_, "base: anything but FAIL-NO-MECHANISM picks M2b-pgt", 'base = "M2b-pgt" if v == "SUCCESS" else "M2b"', 'base = "M2b-pgt" if v != "FAIL-NO-MECHANISM" else "M2b"'),
    (R_, "base: a seed-0 STOP picks M2b-pgt", 'return {"base": "M2b", "why": f"PREREG_V8F_PGT seed-0 STOP', 'return {"base": "M2b-pgt", "why": f"PREREG_V8F_PGT seed-0 STOP'),
    (R_, "refusal off: no PGT read at all", "    if final is None and s0 is None:\n", "    if False:\n"),
    (R_, "refusal off: a PGT read of the wrong schema / stage", 'if r is not None and (r.get("schema") != PGT_SCHEMA or r.get("stage") != stage):', "if False:"),
    (R_, "refusal off: a seed-0 STOP beside a final read", 'if s0 is not None and s0.get("DECISION") != "CONTINUE":', "if False:"),
    (R_, "refusal off: an unknown PGT verdict", 'if v not in ("SUCCESS", "FAIL-MECHANISM-OK", "FAIL-NO-MECHANISM"):', "if False:"),
    (R_, "refusal off: --base disagrees with the PGT verdict", '    if base != dec["base"]:\n', "    if False:\n"),
    (R_, "the committed dose of M2b-pgt is another", '"M2b-pgt": (0.5, 0.0)}', '"M2b-pgt": (0.5, 0.2)}'),
    (R_, "the committed dose of M2b is another", 'DOSE = {"M2b": (1.0, 0.2)', 'DOSE = {"M2b": (1.0, 0.0)'),
    (R_, "the token counts are another", "N_TOK_B, N_TOK_G = 40, 160", "N_TOK_B, N_TOK_G = 40, 80"),
    # ---- the decision table
    (R_, "final: SUCCESS needs only ONE passing seed", 'ok = all(rows[f"s{s}"]["judge"]["passed"] for s in seeds)', 'ok = any(rows[f"s{s}"]["judge"]["passed"] for s in seeds)'),
    (R_, "mechanism read as ANY seed instead of every seed", 'mech_all = all(rows[f"s{s}"]["judge"]["mlat"]["met"] for s in seeds)',
     'mech_all = any(rows[f"s{s}"]["judge"]["mlat"]["met"] for s in seeds)'),
    (R_, "mechanism is always met", 'mech_all = all(rows[f"s{s}"]["judge"]["mlat"]["met"] for s in seeds)', "mech_all = True"),
    (R_, "final: the two FAIL labels swapped", '("SUCCESS" if ok else ("FAIL-MECHANISM-OK" if mech_all else "FAIL-NO-MECHANISM"))',
     '("SUCCESS" if ok else ("FAIL-NO-MECHANISM" if mech_all else "FAIL-MECHANISM-OK"))'),
    (R_, "s0: the two FAIL labels swapped", '(None if go else ("FAIL-MECHANISM-OK" if mech_all else "FAIL-NO-MECHANISM"))',
     '(None if go else ("FAIL-NO-MECHANISM" if mech_all else "FAIL-MECHANISM-OK"))'),
    (R_, "s0: CONTINUE on a failed seed 0", 'go = rows["s0"]["judge"]["passed"]', 'go = not rows["s0"]["judge"]["passed"]'),
    (R_, "final: SUCCESS declared whatever the seeds read", 'return None, ("SUCCESS" if ok else', 'return None, ("SUCCESS" if True else'),
    (R_, "stage s0 reads both seeds", 'seeds = (0,) if a.stage == "s0" else (0, 1)', "seeds = (0, 1)"),
    # ---- the committed bars
    (R_, "PRIMARY bar is F, not 2F", 'abs(p["point"]) > 2 * F)', 'abs(p["point"]) > F)'),
    (R_, "PRIMARY bar has no size floor", 'abs(p["point"]) > 2 * F)', 'abs(p["point"]) > 0)'),
    (R_, "PRIMARY bar is 3F", 'abs(p["point"]) > 2 * F)', 'abs(p["point"]) > 3 * F)'),
    (R_, "PRIMARY does not need a separated interval", 'return {"passed": bool(PGT._is_neg_sep(p) and abs(p["point"]) > 2 * F),', 'return {"passed": bool(p["point"] < 0 and abs(p["point"]) > 2 * F),'),
    (R_, "the judge reads F as zero", 'row["judge"] = judge(row, F)', 'row["judge"] = judge(row, 0.0)'),
    (R_, "F is the B-vs-G gap, not B's own seed floor",
     '    PGT.check_gt_codes(Zrel, seeds, WC)\n    F = abs(float(Zb[(base, 0)]["ade6"].mean()) - float(Zb[(base, 1)]["ade6"].mean()))',
     '    PGT.check_gt_codes(Zrel, seeds, WC)\n    F = abs(float(Zb[(base, 0)]["ade6"].mean()) - float(Zg[(G, 0)]["ade6"].mean()))'),
    (R_, "M-LAT bar is 10 %", "MLAT_BAR = 0.125", "MLAT_BAR = 0.10"),
    (R_, "M-LAT bar is 15 %", "MLAT_BAR = 0.125", "MLAT_BAR = 0.15"),
    (R_, "M-LAT bar is gone", "MLAT_BAR = 0.125", "MLAT_BAR = 0.0"),
    (R_, "M-LAT bar is exclusive", 'ml["fall"] >= MLAT_BAR)', 'ml["fall"] > MLAT_BAR)'),
    (R_, "M-LAT does not need a separated MSE delta", 'bool(PGT._is_neg_sep(d) and ml["fall"] >= MLAT_BAR)', 'bool(ml["fall"] >= MLAT_BAR)'),
    (R_, "M-LAT is separation OR size", 'bool(PGT._is_neg_sep(d) and ml["fall"] >= MLAT_BAR)', 'bool(PGT._is_neg_sep(d) or ml["fall"] >= MLAT_BAR)'),
    (R_, "M-LAT fall has the wrong sign", "    fall = 1.0 - ra / rb\n", "    fall = ra / rb - 1.0\n"),
    (R_, "M-LAT fall is relative to G, not B", "    fall = 1.0 - ra / rb\n", "    fall = 1.0 - rb / ra\n"),
    (R_, "M-LAT window-weighted fall has the wrong sign", "    fall_win = 1.0 - win_a / win_b\n", "    fall_win = 1.0 - win_b / win_a\n"),
    (R_, "M-LAT never reports the definitions disagreeing", '"definitions_agree": bool((fall >= MLAT_BAR) == (fall_win >= MLAT_BAR))', '"definitions_agree": True'),
    (R_, "M-LAT contrast sign flipped", "d = PGT.paired(pf, ma[ok], mb[ok], eid[ok], n_boot)", "d = PGT.paired(pf, mb[ok], ma[ok], eid[ok], n_boot)"),
    (R_, "M-LAT reads the lonspeed slots", 'cols = list(WC.GROUPS["latcons"])', 'cols = list(WC.GROUPS["lonspeed"])'),
    (R_, "M-LAT scores absent slots as zero-error truth", "    pres = has[:, None] & (g_cons != 0.0)\n", "    pres = has[:, None] & (g_cons == g_cons)\n"),
    (R_, "refusal off: M-LAT present-slot sets differ", "if not ok.any() or na != nb or not np.array_equal(np.isfinite(ma), ok):", "if not ok.any():"),
    (R_, "refusal off: no latcons RMSE to measure", "    if ra is None or rb is None or not rb > 0:\n", "    if False:\n"),
    (R_, "PRIMARY contrast sign flipped", "row = PGT.contrast(pairfn, A, Bm, pw, ctx, a.n_boot)", "row = PGT.contrast(pairfn, Bm, A, pw, ctx, a.n_boot)"),
    (R_, "G and B swapped at each seed", "A, Bm = Zg[(G, s)], Zb[(base, s)]", "A, Bm = Zb[(base, s)], Zg[(G, s)]"),
    # ---- the run checks (load_g_runs): one lever, built as declared
    (R_, "refusal off: a G run is not done", '        if sm.get("done") is not True:\n', "        if False:\n"),
    (R_, "refusal off: G's n_clips", '        if sm.get("n_clips") != PGT.N_TRAIN:\n', "        if False:\n"),
    (R_, "refusal off: G did not withhold the RA-1 holdout", '        if sm.get("excluded_sha256") != PGT.HOLDOUT_SHA:\n', "        if False:\n"),
    (R_, "refusal off: G trained on another clip list", '        if sm.get("train_sha256") != rsm.get("train_sha256"):\n', "        if False:\n"),
    (R_, "refusal off: G trained another number of steps", '        if sm.get("steps") != rsm.get("steps"):\n', "        if False:\n"),
    (R_, "refusal off: G's launch gate is not PASS", '        if sc.get("declared_vs_built") != "PASS":\n', "        if False:\n"),
    (R_, "refusal off: selfcheck.json is another run's", '        if sc.get("argv_sha") != sm.get("argv_sha"):\n', "        if False:\n"),
    (R_, "refusal off: G's argv --arm", '        if fl.get("--arm") != "M2b":\n', "        if False:\n"),
    (R_, "refusal off: G's argv --grid", "        if fl.get(GRID_FLAG) != GRID_VAL:\n", "        if False:\n"),
    (R_, "refusal off: B's argv --grid", "        if rfl.get(GRID_FLAG, B_GRID_VAL) != B_GRID_VAL:\n", "        if False:\n"),
    (R_, "refusal off: G's argv differs from B's in a second flag", "        if fg != fb:\n", "        if False:\n"),
    (R_, "refusal off: G records no built p_gt", "            if b0 is None or b1 is None:\n", "            if False:\n"),
    (R_, "refusal off: G built another p_gt", "            elif not (PGT._same_float(b0, p0) and PGT._same_float(b1, p1)):\n", "            elif False:\n"),
    (R_, "refusal off: a built lever other than the grid differs", "            if diff:\n", "            if False:\n"),
    (R_, "built levers: weights are exempt from the comparison", '_SKIP_LEVERS = ("params", "n_params_trainable", "residual_zero_init")', '_SKIP_LEVERS = ("params", "n_params_trainable", "residual_zero_init", "w", "wm")'),
    (R_, "refusal off: the checkpoint's grid", '        if ck["grid"] != GRID_VAL:\n', "        if False:\n"),
    (R_, "refusal off: the checkpoint's built n_tok", '        if ck["n_tok"] != N_TOK_G:\n', "        if False:\n"),
    (R_, "refusal off: the checkpoint's position-table rows", '        if ck["pos_rows"] != N_TOK_G:\n', "        if False:\n"),
    (R_, "refusal off: the checkpoint's argv_sha", '        if ck["argv_sha"] != sm.get("argv_sha"):\n', "        if False:\n"),
    (R_, "refusal off: the checkpoint's step", '        if ck["step"] != sm.get("steps"):\n', "        if False:\n"),
    (R_, "refusal off: the checkpoint's arm / seed", '        if ck["arm"] != "M2b" or ck["seed"] != s:\n', "        if False:\n"),
    (R_, "G's train log is optional", "p0, p1, True)", "p0, p1, False)"),
    (R_, "check_params: the expected delta is another", "    want = (N_TOK_G - N_TOK_B) * d\n", "    want = (N_TOK_G - N_TOK_B) * d // 2\n"),
    (R_, "check_params: per-group deltas unchecked", '            if dd != (want if k == "adapter" else 0):\n', "            if False:\n"),
    (R_, "check_params: the total delta unchecked", "    if dt != want:\n", "    if False:\n"),
    (R_, "refusal off: a G eval scored another checkpoint", '            if len(parts) < 2 or parts[-2] != g_dir(base, s) or parts[-1] != "ckpt_last.pt":', "            if False:"),
    (R_, "refusal off: G's evals on another eval grid", "    if not np.array_equal(np.asarray(eid_g), np.asarray(eid)):\n", "    if False:\n"),
    (R_, "the GT constraint blocks are not compared (G vs B)", "    PGT.check_gt_codes(Zrel, seeds, WC)\n    F = abs(", "    F = abs("),
    (R_, "refusal off: B's evals are not the ones the PGT verdict read", "            if abs(got - float(rec[base])) > 1e-9:\n", "            if False:\n"),
    # ---- the controls
    (R_, "identity control: M-LAT cell unchecked", '        for name, v in cells.items():\n            if not (v["delta"] == 0.0 and v["lo"] == 0.0 and v["hi"] == 0.0 and v["point"] == 0.0):\n',
     '        for name, v in cells.items():\n            if False:\n'),
    (R_, "identity control: the latcons fall unchecked", '        if row["fall"] != 0.0 or row["fall_window"] != 0.0:\n', "        if False:\n"),
    (R_, "known values are never checked", 'known = PGT.known_values(base_st, F, (0, 1)) if base == "M2b" else {"checked": False, "why": "B is M2b-pgt: the PREREG',
     'known = PGT.known_values(base_st, F, (0, 1)) if base == "NONE" else {"checked": False, "why": "B is M2b-pgt: the PREREG'),
    (R_, "check-inputs never checks the known values", 'known = PGT.known_values(base_st, F, (0, 1)) if base == "M2b" else {"checked": False, "why": "B is M2b-pgt"}',
     'known = PGT.known_values(base_st, F, (0, 1)) if base == "NONE" else {"checked": False, "why": "B is M2b-pgt"}'),
    (R_, "check-inputs does not compare the banked GT codes", "    PGT.check_gt_codes(Zrel, (0, 1), WC)\n", "    pass\n"),
]
Q_, C_, K_ = QUEUE, CHAIN, CACHE
QUEUEM = [
    # ---- :gridbase, the one place the dose is read
    (Q_, "gridbase: M2b-pgt carries no dose", 'if /i "%GBASE%"=="M2b-pgt" set "GLEV= --p-gt-start %PGTS% --p-gt-end %PGTE%"\n', "rem\n"),
    (Q_, "gridbase: the dose end is M2b's 0.2", 'set "GLEV= --p-gt-start %PGTS% --p-gt-end %PGTE%"', 'set "GLEV= --p-gt-start %PGTS% --p-gt-end 0.2"'),
    (Q_, "gridbase: the dose is a literal copy", 'set "GLEV= --p-gt-start %PGTS% --p-gt-end %PGTE%"', 'set "GLEV= --p-gt-start 0.5 --p-gt-end %PGTE%"'),
    (Q_, "gridbase: the dose endpoints swapped", 'set "GLEV= --p-gt-start %PGTS% --p-gt-end %PGTE%"', 'set "GLEV= --p-gt-start %PGTE% --p-gt-end %PGTS%"'),
    (Q_, "gridbase: any BASE is accepted", 'if /i not "%GBASE%"=="M2b" if /i not "%GBASE%"=="M2b-pgt" (echo REFUSED: BASE must be M2b or M2b-pgt, got "%GBASE%" & exit /b 2)\n', "rem\n"),
    (Q_, "the dose start moved", 'set "PGTS=0.5"', 'set "PGTS=0.6"'),
    # ---- gategrid
    (Q_, "gategrid: still 4x10", '--arm M2b%GLEV% --grid pool8x20 --out "%RUNSM%\\gate_%GBASE%-8x20"', '--arm M2b%GLEV% --grid pool4x10 --out "%RUNSM%\\gate_%GBASE%-8x20"'),
    (Q_, "gategrid: no --grid at all", ' --grid pool8x20 --out "%RUNSM%\\gate_%GBASE%-8x20"', ' --out "%RUNSM%\\gate_%GBASE%-8x20"'),
    (Q_, "gategrid: drops the base's lever", '--arm M2b%GLEV% --grid pool8x20 --out "%RUNSM%\\gate_%GBASE%-8x20"', '--arm M2b --grid pool8x20 --out "%RUNSM%\\gate_%GBASE%-8x20"'),
    (Q_, "gategrid: not a selfcheck-only run", ' %A% --selfcheck-only"\nif defined V8F_DRYRUN (echo %GG%', ' %A%"\nif defined V8F_DRYRUN (echo %GG%'),
    (Q_, "gategrid: writes into B's gate directory", '--out "%RUNSM%\\gate_%GBASE%-8x20" %A%', '--out "%RUNSM%\\gate_%GBASE%" %A%'),
    (Q_, "gategrid: runs a different string than it prints", '%GG% > "%RUNSM%\\gate_%GBASE%-8x20.log" 2>&1',
     '"%PY%" "%TRAIN%" --arm M2b --grid pool8x20 --out "%RUNSM%\\gate_%GBASE%-8x20" %A% --selfcheck-only > "%RUNSM%\\gate_%GBASE%-8x20.log" 2>&1'),
    (Q_, "gategrid: does not notice a failing gate", 'if errorlevel 1 (echo [FAIL] gate %GBASE%-8x20 - see %RUNSM%\\gate_%GBASE%-8x20.log & exit /b 1)\n', "rem\n"),
    (Q_, "gategrid: reads BASE from the wrong position", 'call :gridbase %~2 || exit /b 2\n', 'call :gridbase %~3 || exit /b 2\n'),
    # ---- traingrid
    (Q_, "traingrid: still 4x10", '--seed %~2 --grid pool8x20 --out "%OUT%" %A%"', '--seed %~2 --grid pool4x10 --out "%OUT%" %A%"'),
    (Q_, "traingrid: no --grid at all", ' --seed %~2 --grid pool8x20 --out "%OUT%" %A%"', ' --seed %~2 --out "%OUT%" %A%"'),
    (Q_, "traingrid: drops the base's lever", 'set "TGR="%PY%" "%TRAIN%" --arm M2b%GLEV% --seed', 'set "TGR="%PY%" "%TRAIN%" --arm M2b --seed'),
    (Q_, "traingrid: no seed on the line", '--arm M2b%GLEV% --seed %~2 --grid pool8x20 --out "%OUT%"', '--arm M2b%GLEV% --grid pool8x20 --out "%OUT%"'),
    (Q_, "traingrid: trains into B's own run directory", 'call :gridbase %~3 || exit /b 2\nset "OUT=%RUNSM%\\%GBASE%_s%~2_8x20"\nif exist',
     'call :gridbase %~3 || exit /b 2\nset "OUT=%RUNSM%\\%GBASE%_s%~2"\nif exist'),
    (Q_, "traingrid: a finished run is re-trained", 'if exist "%OUT%\\summary.json" (echo [skip] %GBASE%_s%~2_8x20 & exit /b 0)\n', "rem\n"),
    (Q_, "traingrid: runs a different string than it prints", '%TGR% >> "%OUT%\\train.log" 2>&1',
     '"%PY%" "%TRAIN%" --arm M2b --seed %~2 --grid pool8x20 --out "%OUT%" %A% >> "%OUT%\\train.log" 2>&1'),
    (Q_, "traingrid: does not notice a failing trainer", 'if errorlevel 1 (echo [FAIL] train %GBASE%_s%~2_8x20 - see %OUT%\\train.log & exit /b 1)\n', "rem\n"),
    (Q_, "traingrid: does not notice a missing summary.json", 'if not exist "%OUT%\\summary.json" (echo [FAIL] train %GBASE%_s%~2_8x20 exited 0 without summary.json & exit /b 1)\n', "rem\n"),
    (Q_, "traingrid: accepts no seed", 'if "%~2"=="" (echo REFUSED: traingrid needs a SEED and a BASE & exit /b 2)\n', "rem\n"),
    (Q_, "evalgrid: accepts no seed", 'if "%~2"=="" (echo REFUSED: evalgrid needs a SEED and a BASE & exit /b 2)\n', "rem\n", ":evalgrid"),
    (Q_, "traingrid: BASE not validated", 'call :gridbase %~3 || exit /b 2\nset "OUT=%RUNSM%\\%GBASE%_s%~2_8x20"\nif exist',
     'rem\nset "GBASE=%~3"\nset "OUT=%RUNSM%\\%GBASE%_s%~2_8x20"\nif exist'),
    # ---- evalgrid
    (Q_, "evalgrid E0: no --project --families", ' --tag %TG% %EC% %ER% --project --families"', ' --tag %TG% %EC% %ER%"', ":evalgrid"),
    (Q_, "evalgrid E0: scores B's checkpoint", 'set "E0="%PY%" "%EVAL%" run --ckpt "%OUT%\\ckpt_last.pt"', 'set "E0="%PY%" "%EVAL%" run --ckpt "%RUNSM%\\%GBASE%_s%~2\\ckpt_last.pt"', ":evalgrid"),
    (Q_, "evalgrid E1: not an absmask eval", ' --tag %TG%-tacabs --tac absmask %EC% %ER%"', ' --tag %TG%-tacabs %EC% %ER%"', ":evalgrid"),
    (Q_, "evalgrid E2: absmask instead of the GT oracle", ' --tag %TG%-tacorc --tac oracle %EC% %ER%"', ' --tag %TG%-tacorc --tac absmask %EC% %ER%"', ":evalgrid"),
    (Q_, "evalgrid E2: scores B's checkpoint", 'set "E2="%PY%" "%EVAL%" run --ckpt "%OUT%\\ckpt_last.pt"', 'set "E2="%PY%" "%EVAL%" run --ckpt "%RUNSM%\\%GBASE%_s%~2\\ckpt_last.pt"', ":evalgrid"),
    (Q_, "evalgrid E3: not the all-latcons oracle", ' --tac oracle --tac-groups all-latcons --tac-alpha 1 %EC% %ER%"', ' --tac oracle --tac-alpha 1 %EC% %ER%"', ":evalgrid"),
    (Q_, "evalgrid E3: alpha not 1", ' --tac-groups all-latcons --tac-alpha 1 %EC% %ER%"', ' --tac-groups all-latcons %EC% %ER%"', ":evalgrid"),
    (Q_, "evalgrid E3: B's tag", ' --tag %TG%-tacorc-all-latcons --tac oracle', ' --tag %TG%-tacorc-all-latcons-b --tac oracle', ":evalgrid"),
    (Q_, "evalgrid: dry run omits E3", "echo %E2% & echo %E3% & exit /b 0)", "echo %E2% & exit /b 0)", ":evalgrid"),
    (Q_, "evalgrid: evaluates an unfinished run", 'if not exist "%OUT%\\summary.json" (echo REFUSED: %TG% has not finished training & exit /b 2)\n', "rem\n", ":evalgrid"),
    (Q_, "evalgrid: absmask skip looks at the wrong file", 'if exist "%EV%\\eval_%TG%-tacabs.npz" (echo [skip] eval %TG%-tacabs & goto :evalgrid_2)',
     'if exist "%EV%\\eval_%TG%.npz" (echo [skip] eval %TG%-tacabs & goto :evalgrid_2)', ":evalgrid"),
    (Q_, "evalgrid: does not notice a failing oracle eval", 'if errorlevel 1 (echo [FAIL] eval %TG%-tacorc - see %EV%\\eval_%TG%-tacorc.log & exit /b 1)\n', "rem\n", ":evalgrid"),
    (Q_, "evalgrid: does not notice a failing all-latcons eval", 'if errorlevel 1 (echo [FAIL] eval %TG%-tacorc-all-latcons - see %EV%\\eval_%TG%-tacorc-all-latcons.log & exit /b 1)\n', "rem\n", ":evalgrid"),
    (Q_, "evalgrid: evaluates B's run directory", 'call :gridbase %~3 || exit /b 2\nset "OUT=%RUNSM%\\%GBASE%_s%~2_8x20"\nset "TG=', 'call :gridbase %~3 || exit /b 2\nset "OUT=%RUNSM%\\%GBASE%_s%~2"\nset "TG=', ":evalgrid"),
    # ---- the dispatch
    (Q_, "gategrid is not dispatched", 'if /i "%~1"=="gategrid" goto :gategrid\n', "rem\n"),
    (Q_, "traingrid is not dispatched", 'if /i "%~1"=="traingrid" goto :traingrid\n', "rem\n"),
    (Q_, "evalgrid is not dispatched", 'if /i "%~1"=="evalgrid" goto :evalgrid\n', "rem\n"),
]
CHAINM = [
    (C_, "the dry run omits evalgrid 1 M2b-pgt", ' "evalgrid 0 M2b-pgt" "evalgrid 1 M2b-pgt"; do', ' "evalgrid 0 M2b-pgt"; do'),
    (C_, "the dry run omits traingrid 1 M2b", ' "traingrid 0 M2b" "traingrid 1 M2b" ', ' "traingrid 0 M2b" '),
    (C_, "the STOP marker is renamed", 'stop() { echo "ZZV8FMINI-STOP $*"', 'stop() { echo "ZZV8FMINI-HALT $*"'),
    (C_, "the DONE marker is renamed", 'echo "ZZV8FMINI-DONE $(date', 'echo "ZZV8FMINI-OK $(date'),
    (C_, "the busy marker is not honoured", "[ -f $B/BOX2_BUSY.txt ] && {", "[ -f $B/BOX2_BUSY_X.txt ] && {"),
    (C_, "the training wait is 5 h (a guarded 10 h run outlasts it)", "TRAIN_POLLS=1320; EVAL_POLLS=600", "TRAIN_POLLS=600; EVAL_POLLS=600"),
    (C_, "the predecessor is the wrong chain", "AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_pgt.out}", "AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_wc.out}"),
    (C_, "the wait for the predecessor is skipped", 'until grep -q "^$AFTER_DONE" "$AFTER_OUT" 2>/dev/null; do', "until true; do"),
    (C_, "the base is hard-coded to M2b", "BASE=$(jget base.json base)\n", "BASE=M2b\n"),
    (C_, "the base is not validated", 'case "$BASE" in M2b|M2b-pgt) ;; *) stop "the decided base \'$BASE\' is neither M2b nor M2b-pgt";; esac\n', "true\n"),
    (C_, "check-inputs failure is ignored", '|| stop "the reader cannot use the $BASE side (check-inputs)"', "|| true"),
    (C_, "the cache check failure is ignored", '|| stop "box 2\'s cache does not carry 8x20 for every clip (check_grid_cache)"', "|| true"),
    (C_, "the cache check expects another clip count", "--n-train 4572", "--n-train 4000"),
    (C_, "the train job drops the base", 'queue "t_$g" $TRAIN_POLLS traingrid $s $BASE', 'queue "t_$g" $TRAIN_POLLS traingrid $s'),
    (C_, "the eval job drops the base", 'queue "e_$g" $EVAL_POLLS evalgrid $s $BASE', 'queue "e_$g" $EVAL_POLLS evalgrid $s'),
    (C_, "the gate job is M2b whatever the base", "queue gategrid 120 gategrid $BASE", "queue gategrid 120 gategrid M2b"),
    (C_, "a seed-0 STOP is not a committed end", '[ "$(jget grid_read_s0.json DECISION)" = CONTINUE ] || finish', "true || finish"),
    (C_, "the checkpoint is not pulled", "train_log.jsonl train.log ckpt_last.pt; do pull", "train_log.jsonl train.log; do pull"),
    (C_, "the all-latcons eval is not pulled", 'for f in "" -tacabs -tacorc -tacorc-all-latcons; do for x in json npz log; do pull', 'for f in "" -tacabs -tacorc; do for x in json npz log; do pull'),
    (C_, "the P10 reproduction guard file is not staged", 'cp "$L/v8f_mini/eval/floors_perwindow.npz" $E/eval/ || stop', 'cp "$L/v8f_mini/eval/floors_perwindow.npz" $E/eval/ || true; false || stop'),
    (C_, "gate assertion off: the parameter total", 'if lv["n_params_trainable"] != bl["n_params_trainable"] + D: bad.append(', "if False: bad.append("),
    (C_, "gate assertion off: the per-group delta", 'if pg.get(k, 0) - pb.get(k, 0) != (D if k == "adapter" else 0): bad.append(', "if False: bad.append("),
    (C_, "gate assertion: the position table is 128 wide", "D = (160 - 40) * 256", "D = (160 - 40) * 128"),
    (C_, "gate assertion: M2b's schedule is checked against the queue's dose", 'want = (p0, p1) if base == "M2b-pgt" else (1.0, 0.2)', "want = (p0, p1)"),
    (C_, "gate assertion off: the weights besides w_tac / w_cons", 'for k in ("w_pen", "w_pen_self", "w_wm"):', 'for k in ("w_pen", "w_pen_self"):'),
    (C_, "gate assertion off: w_tac / w_cons", 'if (w.get("w_tac"), w.get("w_cons")) != (1.0, 0.5): bad.append(', "if False: bad.append("),
    (C_, "gate assertion off: the step count", 'if sc["steps_total"] != steps: bad.append(', "if False: bad.append("),
    (C_, "gate assertion off: B's clip list", 'if i["train_sha256"] != b["train_sha256"]: bad.append(', "if False: bad.append("),
    (C_, "gate assertion off: the holdout", 'if i["excluded_sha256"] != hold: bad.append(', "if False: bad.append("),
    (C_, "gate assertion off: the clip counts", 'if (i["n_clips"], i["n_excluded"], i["n_excluded_declared"], i["excluded_in_train"]) != (4512, 60, 60, 0): bad.append(', "if False: bad.append("),
    (C_, "gate assertion off: kin_mode", 'if i["kin_mode"] != "yaw": bad.append("kin_mode")', "pass"),
    (C_, "gate assertion off: the levers", 'if (lv["wm"], lv["navonly"], lv["consistency"], lv.get("cons_absent")) != ("b", False, True, "mask"): bad.append(', "if False: bad.append("),
    (C_, "gate assertion off: the live check", 'if not sc["live"].get("pass") or sc["live"].get("control_prior_grad_sum") != 0.0: bad.append(', "if False: bad.append("),
    (C_, "gate assertion off: declared_vs_built", 'if sc.get("declared_vs_built") != "PASS": bad.append("declared_vs_built")', "pass"),
    (C_, "gate assertion off: the recorded pipeline sizing", 'if not i.get("lru"): bad.append(', "if False: bad.append("),
]
CACHEM = [
    (K_, "cache: 8x20 has 80 tokens", 'WANT = {"pool4x10": ("4x10", 40), "pool8x20": ("8x20", 160)}', 'WANT = {"pool4x10": ("4x10", 40), "pool8x20": ("8x20", 80)}'),
    (K_, "cache: 8x20 not required in grids", "        if g not in grids:\n", "        if False:\n"),
    (K_, "cache: shapes not compared", "        elif list(shapes[arr]) != [n, n_tok, D_FEAT]:\n", "        elif False:\n"),
    (K_, "cache: the FIRST manifest line wins", "        p = row_problems(rows[c][-1])\n", "        p = row_problems(rows[c][0])\n"),
    (K_, "cache: the train clip count unchecked", "        if len(tr) != a.n_train:\n", "        if False:\n"),
    (K_, "cache: every eval clip is required, not the eval grid's", 'check_split("eval", ev, order)', 'check_split("eval", ev)'),
    (K_, "cache: a row without a grids list passes", "    if not isinstance(grids, list):\n", "    if False:\n"),
    (K_, "cache: a clip with no row passes", "        if c not in rows:\n", "        if False:\n"),
    (K_, "cache: an empty manifest passes", "    if not rows:\n", "    if False:\n"),
    (K_, "cache: an empty eval grid passes", "        if not order:\n", "        if False:\n"),
]
GROUPS = {"reach": REACH, "reader": READER, "queue": QUEUEM, "chain": CHAINM, "cache": CACHEM}
MUT = [m for g in GROUPS.values() for m in g] if WHICH == "all" else GROUPS[WHICH]
if "--part" in sys.argv:                                   # `--part 2/3`: every 3rd mutant starting at the 2nd
    _k, _n = (int(x) for x in sys.argv[sys.argv.index("--part") + 1].split("/"))
    MUT = MUT[_k - 1::_n]
if "--only" in sys.argv:                                   # `--only "a||b"`: the mutants whose name contains any of the substrings (the round-2 re-run of the survivors)
    _keys = sys.argv[sys.argv.index("--only") + 1].split("||")
    MUT = [m for m in MUT if any(k in m[1] for k in _keys)]
OWN = {"reach": "not (reader or decide or mlat or judge or check_inputs or known_values or queue_grid or chain_grid or cache_check or dose_arms)",
       "reader": "reader or decide or mlat or judge or check_inputs or known_values",
       "queue": "queue_grid", "chain": "chain_grid", "cache": "cache_check"}
KSEL = {id(m): k for g, L in GROUPS.items() for k in [OWN[g]] for m in L}      # a mutant is tried on its own group's tests first


def run(k=None):
    cmd = [PY, "-m", "pytest", "-q", "-x", "-rf", "-p", "no:cacheprovider", TEST] + (["-k", k] if k else [])
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1",
                            "PYTHONPATH": os.pathsep.join([str(ROOT / "stack"), str(ROOT / "taniteval")])})
    lines = r.stdout.strip().splitlines() or ["<no output>"]
    failed = [x.split("::", 1)[1].split(" - ")[0].strip() for x in lines if x.startswith("FAILED ") and "::" in x]
    return r.returncode, lines[-1] + (f"  [{', '.join(failed)}]" if failed else "")


def apply(m, b):
    """-> the mutated bytes. The anchor must occur EXACTLY ONCE: in the whole file, or -- when the mutant names a 5th element, a `:label` -- in that label's stage only
    (the evalgrid stage is the evalpgt stage's twin, so its lines are not unique file-wide)."""
    crlf = b"\r\n" in b                                                      # .cmd files are CRLF on the tip; every other file is LF
    fix = (lambda t: t.replace(b"\n", b"\r\n")) if crlf else (lambda t: t)
    o, n = fix(m[2].encode()), fix(m[3].encode())
    lo, hi = 0, len(b)
    if len(m) > 4:
        lab = fix(("\n" + m[4] + "\n").encode())
        assert b.count(lab) == 1, f"{m[1]}: stage label {m[4]} found {b.count(lab)} times"
        lo = b.index(lab)
        hi = b.index(fix(b"\nexit /b 0\n"), lo)
    assert b[lo:hi].count(o) == 1, f"{m[1]}: anchor found {b[lo:hi].count(o)} times in {m[0].name}" + (f" ({m[4]})" if len(m) > 4 else "")
    assert o != n, f"{m[1]}: the mutation changes nothing"
    return b[:lo] + b[lo:hi].replace(o, n) + b[hi:]


orig = {p: p.read_bytes() for p in {m[0] for m in MUT}}
for m in MUT:
    apply(m, orig[m[0]])
print(f"anchors OK: {len(MUT)} mutants, every anchor occurs exactly once", flush=True)
if "--check-anchors" in sys.argv:
    sys.exit(0)
rc, last = run()
print(f"BASELINE rc={rc} {last}", flush=True)
assert rc == 0, "the unmutated tree must be GREEN"
red = 0
survivors = []
for m in MUT:
    path, name = m[0], m[1]
    b = orig[path]
    path.write_bytes(apply(m, b))
    try:
        k = KSEL.get(id(m))
        rc, last = run(k)
        scope = f"[-k group] "
        if rc == 0 and k:                                  # survived its own group's tests: the whole file decides
            rc, last = run()
            scope = "[whole file] "
    finally:
        path.write_bytes(b)
    red += rc != 0
    if rc == 0:
        survivors.append(name)
    print(f"{'RED  ' if rc != 0 else 'GREEN'} {path.name}: {name} -> {scope}{last}", flush=True)
assert all(p.read_bytes() == orig[p] for p in orig), "a file was not restored"
rc, last = run()
print(f"AFTER rc={rc} {last}")
print(f"{red} / {len(MUT)} mutations RED" + (f"; SURVIVORS: {survivors}" if survivors else ""))
sys.exit(0 if red == len(MUT) and rc == 0 else 1)
