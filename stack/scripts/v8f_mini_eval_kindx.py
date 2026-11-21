#!/usr/bin/env python3
"""v8f_mini_eval_kindx.py -- the box-2 / chain helper of PREREG_V8F_KINDX's SEED-1 CONFIRMATION (chain_v8f_kindx_s1.sh).

WHY THIS FILE IS NAMED `v8f_mini_eval_*`: `box2_ops/v8f_guard.sh` stops "our" python on box 2 by matching `v8f_mini_train|v8f_mini_eval|...` on the command line. This driver runs
the ten KINDX oracle evals in ONE process, so its name carries `v8f_mini_eval`: the thermal / time guard that is already running can kill it with no change to the guard (a new
name would need a WIRING patch to a shared file, as v8f_kscore did). A test pins that the guard's regex matches this driver's command line.

SUBCOMMANDS
  evals        (box 2)  X = M2b-mixo_s<seed>: the ten KINDX evals, each through the tip's own `v8f_mini_eval.py run` (imported, unmodified): own | six single kinds | union | all | none.
                        `none` is the IDENTITY ORACLE (GT in no kind): `v8f_mini_eval` has no empty group spec, so the ONE patch is `v8f_mini.code_group_index('none') -> []`
                        (with an empty selection the eval's own code does `code = where(sel, code, code0)` = code0 exactly), the SAME patch kindx_eval.py used at s0. The checkpoint's
                        sha256 must equal --expect-sha before the first eval and after the last; an eval whose npz AND json exist is skipped (resume).
  evals-argv   (dev box) the ten `v8f_mini_eval.py run` argvs the job's process will run, parsed back OUT OF the job's own command line (box 2's paths); the chain's --dry-run prints them.
  cmdline / mkjob       (dev box) the ONE command line a box-2 job runs / the job .cmd (CRLF) around it, in the HIST/KSCORE recipe: the chain's --dry-run prints exactly this line.
  assert-argv           the training argv of X at seed 1 differs from MIXO's seed-0 argv in EXACTLY the seed (and the output directory that carries it). Two queue lines or two argv.json.
  assert-gate           the launch-gate assertions of chain_v8f_mixo.sh (a port of its heredoc, tested against the banked s0 gate selfcheck) on a gate's selfcheck.json; used for the
                        pre-launch gate AND for the run's own in-process selfcheck.
  write-token           the gate token record (written by the chain, never by hand).
Evidence tier of everything it feeds: T0 (GT-conditioned oracles) -- a diagnostic of how O consumes T's code, never a driving claim.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys

KINDS = ("speed", "angle", "tte", "timegap", "geom", "leadgap")
#: the order the evals run in: the controls' arms first (own, none, all, union), then the six single kinds -- exactly the order of kindx_eval's phases at s0
EVAL_NAMES = ("own", "none", "all", "union") + KINDS
TAGBASE = "M2b-mixo_s{seed}-kindx"
HOLDOUT_SHA = "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9"
#: box 2's layout (the queue's defaults: REPO / PY / CACHEFULL / V9 / RUNSM); KINDX_B2_<KEY> overrides one (tests only)
BOX2 = {"REPO": "D:\\tanitad\\repo", "PY": "D:\\tanitad\\venv\\Scripts\\python.exe", "RUNSM": "D:\\tanitad\\runs\\v8f_mini_full", "CACHE": "D:\\tanitad\\cache_full",
        "V9": "D:\\tanitad\\data\\v9"}


def tag_of(name: str, seed: int) -> str:
    """The eval tag: KIND's `only-` prefix for the single kinds, as kindx_eval.py named them at s0."""
    return f"{TAGBASE.format(seed=seed)}-" + (f"only-{name}" if name in KINDS else name)


def groups_of(name: str):
    """The `--tac-groups` spec of an oracle arm (None for T's own code)."""
    if name == "own":
        return None
    if name in KINDS:
        return name
    if name == "union":
        return "constraints"
    if name in ("all", "none"):
        return name
    raise SystemExit(f"unknown KINDX eval name {name!r}")


def eval_argv(name: str, ckpt: str, tag: str, paths: dict) -> list:
    """`v8f_mini_eval.py run ...` for one KINDX eval: kindx_eval.argv_for's flags with box 2's paths (a test pins the flag set equal to KIND's queue lines, `kindown` / `kindorc`)."""
    common = ["run", "--ckpt", ckpt, "--tag", tag,
              "--v9-eval", paths["v9_eval"], "--lead-eval", paths["lead_eval"], "--speed-eval", paths["speed_eval"], "--kin-mode", "yaw", "--p10", paths["p10"],
              "--out-dir", paths["eval_dir"], "--eval-cache", paths["cache"], "--tracks", paths["tracks"]]
    g = groups_of(name)
    if g is None:
        return common + ["--tac", "pred", "--project", "--families"]
    return common + ["--tac", "oracle", "--tac-groups", g, "--tac-alpha", "1"]


def paths_from(runs_dir: str, cache: str, v9: str, p10: str, tracks: str, eval_dir: str) -> dict:
    j = lambda d, f: d.rstrip("\\/") + ("\\" if "\\" in d else "/") + f                      # noqa: E731
    return {"v9_eval": j(v9, "v9_labels_eval139.npz"), "lead_eval": j(v9, "lead_eval139.npz"), "speed_eval": j(v9, "refcv6_speed_max_v8_eval.jsonl"), "p10": p10, "eval_dir": eval_dir,
            "cache": cache, "tracks": tracks}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ evals (box 2)
def install_none(M):
    """The ONE patch of the identity oracle: `code_group_index('none') -> []`. Returns the original (a test restores it)."""
    orig = M.code_group_index
    M.code_group_index = lambda spec: [] if spec == "none" else orig(spec)
    return orig


def run_evals(a, M=None, E=None, log=print):
    """The ten evals of X in this process. M = the tip's `tanitad.models.v8f_mini`, E = its `v8f_mini_eval` (both injectable for the tests)."""
    run_dir = a.run_dir
    ckpt = run_dir.rstrip("\\/") + ("\\" if "\\" in run_dir else "/") + "ckpt_last.pt"
    if not os.path.exists(ckpt):
        raise SystemExit(f"REFUSED: {ckpt} does not exist (X has not finished training)")
    got = sha256_file(ckpt)
    if got != a.expect_sha:
        raise SystemExit(f"REFUSED: checkpoint sha256 {got} != the one the chain measured after training ({a.expect_sha})")
    if M is None:
        here = os.path.dirname(os.path.abspath(__file__))
        for p in (os.path.join(here, "..", "..", "taniteval"), os.path.join(here, "..")):
            if os.path.abspath(p) not in sys.path:
                sys.path.insert(0, os.path.abspath(p))
        from tanitad.models import v8f_mini as M          # noqa: N812
        spec = importlib.util.spec_from_file_location("v8f_mini_eval_for_kindx", os.path.join(here, "v8f_mini_eval.py"))
        E = importlib.util.module_from_spec(spec)         # noqa: N806
        spec.loader.exec_module(E)
    orig = install_none(M)
    paths = paths_from(a.runs_dir, a.cache_dir, a.v9_dir, a.p10, a.tracks, a.eval_dir)
    n_done = 0
    try:
        for name in EVAL_NAMES:
            tag = tag_of(name, a.seed)
            npz, js = os.path.join(a.eval_dir, f"eval_{tag}.npz"), os.path.join(a.eval_dir, f"eval_{tag}.json")
            if os.path.exists(npz) and os.path.exists(js):
                log(f"[skip] eval {tag}")
                continue
            argv = eval_argv(name, ckpt, tag, paths)
            log("EVAL " + " ".join(argv))
            rc = E.main(argv)
            if rc not in (0, None):
                raise SystemExit(f"FAIL: eval {tag} returned rc {rc}")
            if not (os.path.exists(npz) and os.path.exists(js)):
                raise SystemExit(f"FAIL: eval {tag} returned 0 without writing its npz and json")
            log(f"[ok] eval {tag}")
            n_done += 1
    finally:
        M.code_group_index = orig
    after = sha256_file(ckpt)
    if after != a.expect_sha:
        raise SystemExit(f"FAIL: the checkpoint changed during the evals: {after} != {a.expect_sha}")
    log(f"KINDX evals done: {n_done} run, {len(EVAL_NAMES) - n_done} skipped; checkpoint sha256 unchanged {after[:16]}")
    return 0


def cmd_evals(a):
    if a.dry_run:                                                                         # prints, imports no torch, touches no file
        paths = paths_from(a.runs_dir, a.cache_dir, a.v9_dir, a.p10, a.tracks, a.eval_dir)
        ck = a.run_dir.rstrip("\\/") + ("\\" if "\\" in a.run_dir else "/") + "ckpt_last.pt"
        for name in EVAL_NAMES:
            print("v8f_mini_eval.py " + " ".join(eval_argv(name, ck, tag_of(name, a.seed), paths)))
        return 0
    return run_evals(a)


# ------------------------------------------------------------------------------------------------ the job (dev box)
def _env(env=None):
    e = dict(BOX2)
    for k, v in (env or os.environ).items():
        if k.startswith("KINDX_B2_") and v:
            e[k[len("KINDX_B2_"):]] = v
    return e


def box2_cmdline(stage_args: list, env=None) -> str:
    """The ONE command string the box-2 job runs for `evals --seed S --expect-sha H` (the chain's --dry-run prints exactly this; the job .cmd embeds exactly this)."""
    e = _env(env)
    if not stage_args or stage_args[0] != "evals":
        raise SystemExit(f"cmdline: unknown stage {stage_args[:1]}")
    b = e["RUNSM"]
    seed = None
    rest = list(stage_args[1:])
    if "--seed" in rest:
        seed = rest[rest.index("--seed") + 1]
    if seed is None or "--expect-sha" not in rest:
        raise SystemExit("cmdline evals needs --seed S --expect-sha H")
    paths = (f'--run-dir "{b}\\M2b_s{seed}_mixo" --runs-dir "{b}" --eval-dir "{b}\\eval" --cache-dir "{e["CACHE"]}" --v9-dir "{e["V9"]}" '
             f'--p10 "{b}\\p10\\p10_prior.pt" --tracks "{e["CACHE"]}\\native_tracks.pt"')
    return " ".join([f'"{e["PY"]}"', f'"{e["REPO"]}\\stack\\scripts\\v8f_mini_eval_kindx.py"', "evals", *rest, paths])


def mkjob_text(name: str, stage_args: list, env=None) -> str:
    """The box-2 job .cmd (CRLF): env.cmd (safe mode: 4 threads, AVX2, TEMP), PYTHONPATH as the queue sets it, ONE %RUNSAFE% python call, the ZZJOB-END marker with the rc."""
    e = _env(env)
    BS = chr(92)
    lines = ["@echo off", "setlocal enabledelayedexpansion", "call D:" + BS + "tanitad" + BS + "env.cmd", f'set "REPO={e["REPO"]}"',
             'set "PYTHONPATH=%REPO%' + BS + 'stack;%REPO%' + BS + 'taniteval"', 'set "PYTHONIOENCODING=utf-8"',
             f"echo ===== V8FKINDX {name} {' '.join(stage_args)} %DATE% %TIME%", "%RUNSAFE% " + box2_cmdline(stage_args, env), "set RC=!ERRORLEVEL!",
             "echo ===== ZZJOB-END %DATE% %TIME% rc=!RC!"]
    return "\r\n".join(lines) + "\r\n"


# ------------------------------------------------------------------------------------------------ the argv assertion
def tokens(s: str) -> list:
    """A queue line (quotes kept around a path with spaces) -> its tokens, quotes stripped."""
    return [t.strip('"') for t in re.findall(r'"[^"]*"|\S+', s.strip())]


def seed_only_violations(a: list, b: list, seed_a: int = 0, seed_b: int = 1) -> list:
    """The two training argvs (X at seed_a and at seed_b) differ in EXACTLY the seed -- and in the output directory that carries it in its name. [] = they do."""
    bad = []
    if len(a) != len(b):
        return [f"argv lengths differ: {len(a)} vs {len(b)}"]
    for flag, want in (("--seed", (str(seed_a), str(seed_b))),):
        for nm, v, w in (("base", a, want[0]), ("other", b, want[1])):
            if v.count(flag) != 1:
                bad.append(f"the {nm} argv carries {flag} {v.count(flag)} times")
            elif v[v.index(flag) + 1] != w:
                bad.append(f"the {nm} argv has {flag} {v[v.index(flag) + 1]}, expected {w}")
    for nm, v in (("base", a), ("other", b)):
        if v.count("--code-mix") != 1 or v[v.index("--code-mix") + 1] != "kind":
            bad.append(f"the {nm} argv does not carry `--code-mix kind` exactly once")
        if v.count("--arm") != 1 or v[v.index("--arm") + 1] != "M2b":
            bad.append(f"the {nm} argv is not `--arm M2b`")
    if bad:
        return bad
    i_seed, i_out = a.index("--seed") + 1, (a.index("--out") + 1 if "--out" in a else None)
    if i_out is None or "--out" not in b or b.index("--out") + 1 != i_out:
        return ["the --out flag is missing or moved"]
    for nm, v, sd in (("base", a, seed_a), ("other", b, seed_b)):
        if not v[i_out].rstrip("\\/").endswith(f"M2b_s{sd}_mixo"):
            bad.append(f"the {nm} argv's --out {v[i_out]!r} is not the M2b_s{sd}_mixo directory (a seed-{seed_b} run must not write into X_s{seed_a}'s)")
    if bad:
        return bad
    for i, (x, y) in enumerate(zip(a, b)):
        if x == y:
            continue
        if i == i_seed:
            continue
        if i == i_out:
            if re.sub(rf"M2b_s{seed_a}_mixo(?=[\\/]?$)", f"M2b_s{seed_b}_mixo", x) == y and x.rstrip("\\/").endswith(f"M2b_s{seed_a}_mixo"):
                continue
            bad.append(f"--out differs by more than the seed: {x!r} vs {y!r}")
            continue
        bad.append(f"token {i} differs: {x!r} vs {y!r} (the argv may differ in the seed only)")
    return bad


def load_argv(path_or_line: str) -> list:
    """An argv.json (`{"argv": [...], "sha": ...}`) or a file holding one queue line."""
    with open(path_or_line, encoding="utf-8") as f:
        txt = f.read()
    if path_or_line.lower().endswith(".json"):
        return list(json.loads(txt)["argv"])
    lines = [x for x in txt.splitlines() if ".py" in x]
    if len(lines) != 1:
        raise SystemExit(f"{path_or_line}: expected exactly one queue line naming a .py, found {len(lines)}")
    return tokens(lines[0])


# ------------------------------------------------------------------------------------------------ the gate assertions (a port of chain_v8f_mixo.sh's heredoc)
def gate_violations(sc: dict, b: dict, *, holdout_sha: str = HOLDOUT_SHA, steps: int = 19164, windows: int = 306566, line: str | None = None, seed: int | None = None,
                    require_o_interface: bool = True) -> list:
    """The assertions chain_v8f_mixo.sh makes on X's gate selfcheck, verbatim in substance. `b` = the banked B (M2b) summary.json the comparison is against. `line` = the queue's printed
    gate line (the pre-launch gate only). [] = the gate passes. A missing key is a violation, never a crash."""
    bad = []
    try:
        i, lv, bl = sc["info"], sc["levers"], b["levers"]
        if sc.get("declared_vs_built") != "PASS":
            bad.append("declared_vs_built")
        live = sc.get("live") or {}
        if not live.get("pass") or live.get("control_prior_grad_sum") != 0.0:
            bad.append(f"live {live.get('pass')} / control {live.get('control_prior_grad_sum')}")
        pr = sc.get("code_mix_probe")
        if not isinstance(pr, dict) or pr.get("pass") is not True:
            bad.append(f"code_mix_probe {pr}")
        else:
            if pr.get("code_mix") != "kind":
                bad.append(f"the code_mix probe was run for {pr.get('code_mix')!r}")
            if not (isinstance(pr.get("n_partial"), int) and pr["n_partial"] > 0):
                bad.append(f"the code_mix probe saw no PARTLY-GT code ({pr.get('n_partial')}): the mix did not reach the code O reads")
        if (i["n_clips"], i["n_excluded"], i["n_excluded_declared"], i["excluded_in_train"]) != (4512, 60, 60, 0):
            bad.append("clips / holdout counts")
        if i["excluded_sha256"] != holdout_sha:
            bad.append("the excluded set is not the RA-1 holdout")
        if i["train_sha256"] != b["train_sha256"]:
            bad.append(f"TRAIN list {i['train_sha256'][:12]} != B's {b['train_sha256'][:12]}")
        if i["kin_mode"] != "yaw":
            bad.append("kin_mode")
        if (lv["wm"], lv["navonly"], lv["consistency"], lv.get("cons_absent")) != ("b", False, True, "mask"):
            bad.append(f"levers {lv}")
        if lv.get("code_mix") != "kind":
            bad.append(f"built code_mix {lv.get('code_mix')!r} != 'kind'")
        if require_o_interface:
            if lv.get("o_interface") != "full":
                bad.append(f"built o_interface {lv.get('o_interface')!r} != 'full' (X does not cut O's interface)")
            oi = sc.get("o_interface_probe")
            if not isinstance(oi, dict) or oi.get("pass") is not True or oi.get("o_interface") != "full":
                bad.append(f"o_interface_probe {oi if not isinstance(oi, dict) else {k: oi.get(k) for k in ('pass', 'o_interface')}}: the trainer's probe of what O is handed did not pass for `full`")
        if (lv.get("p_gt_start"), lv.get("p_gt_end")) != (1.0, 0.2):
            bad.append(f"built p_gt {lv.get('p_gt_start')} -> {lv.get('p_gt_end')}, expected 1.0 -> 0.2")
        w = lv.get("w") or {}
        for k in ("w_pen", "w_pen_self", "w_wm"):
            if w.get(k) != bl["w"][k]:
                bad.append(f"{k} {w.get(k)} != B's {bl['w'][k]}")
        if (w.get("w_tac"), w.get("w_cons")) != (1.0, 0.5):
            bad.append(f"w_tac / w_cons {w.get('w_tac')} / {w.get('w_cons')} != B's 1.0 / 0.5 (a second lever moved)")
        if i.get("n_windows_all_clips") != b.get("n_windows"):
            bad.append(f"the window set is not B's: n_windows_all_clips {i.get('n_windows_all_clips')} != B's {b.get('n_windows')}")
        if i.get("n_windows_all_clips") != windows:
            bad.append(f"n_windows_all_clips {i.get('n_windows_all_clips')} != the PREREG's {windows}")
        if sc["steps_total"] != steps:
            bad.append(f"steps {sc['steps_total']} != B's {steps}")
        if lv.get("hist") != 4:
            bad.append(f"built hist {lv.get('hist')} != 4 (B's)")
        if lv.get("hmax") != 16:
            bad.append(f"built hmax {lv.get('hmax')} != 16 (the arm's auto value, B's)")
        if lv.get("window_start") != 3:
            bad.append(f"built window_start {lv.get('window_start')} != 3 (B's effective first window, hist - 1)")
        if (i.get("window_start"), i.get("window_t_min"), i.get("ds_window_start")) != (3, 3, 3):
            bad.append(f"window start (asked / first t of the built window list / dataset clamp origin) {(i.get('window_start'), i.get('window_t_min'), i.get('ds_window_start'))} != (3, 3, 3)")
        pg, pb = lv["params"], bl["params"]
        if lv["n_params_trainable"] != bl["n_params_trainable"]:
            bad.append(f"n_params_trainable {lv['n_params_trainable']} != B's {bl['n_params_trainable']}")
        for k in sorted(set(pg) | set(pb)):
            if pg.get(k, 0) != pb.get(k, 0):
                bad.append(f"parameter group {k}: {pg.get(k)} vs B's {pb.get(k)}")
        if line is not None:
            toks = line.split()
            if toks.count("--code-mix") != 1 or toks[toks.index("--code-mix") + 1] != "kind":
                bad.append("the queue's gatemixo line does not carry --code-mix kind exactly once")
            if toks.count("--arm") != 1 or toks[toks.index("--arm") + 1] != "M2b":
                bad.append("the queue's gatemixo line is not --arm M2b")
            for fl in ("--hist", "--hmax", "--window-start", "--p-gt-start", "--p-gt-end", "--grid", "--w-tac", "--w-cons"):
                if fl in toks:
                    bad.append(f"the queue's gatemixo line sets {fl}: a second lever beside --code-mix")
        if not i.get("lru"):
            bad.append("info.lru is not recorded in the selfcheck (the pipeline sizing must be on the record)")
    except (KeyError, TypeError) as e:
        bad.append(f"the selfcheck / summary lacks {e!r}")
    return bad


#: info keys that are not part of "the same model and data": the pipeline sizing (free RAM) and the argv hash (it carries the seed)
_INFO_VOLATILE = ("lru", "argv_sha")
#: what the tip may have ADDED to the built levers since MIXO's s0 gate was banked, with the only value X may have for it
_LEVERS_ADDED_SINCE_S0 = {"o_interface": "full"}


def same_model_violations(sc: dict, base: dict) -> list:
    """X's built model + data at the new tip equals MIXO's banked s0 gate: every banked lever / parameter group / data fact is equal, and the only additions are the levers a later PREREG
    appended at their DEFAULT (o_interface = full). [] = same model (the seed is the only argv difference, and it is not in the selfcheck)."""
    bad = []
    try:
        lv, bl = sc["levers"], base["levers"]
        for k in bl:
            if k not in lv:
                bad.append(f"levers.{k} is no longer recorded")
            elif lv[k] != bl[k]:
                bad.append(f"levers.{k}: {lv[k]!r} != MIXO s0's {bl[k]!r}")
        for k in sorted(set(lv) - set(bl)):
            if k not in _LEVERS_ADDED_SINCE_S0:
                bad.append(f"levers.{k} is new and not a known default-valued addition")
            elif lv[k] != _LEVERS_ADDED_SINCE_S0[k]:
                bad.append(f"levers.{k} = {lv[k]!r}, expected the default {_LEVERS_ADDED_SINCE_S0[k]!r}")
        for k in sorted(set(sc["info"]) | set(base["info"])):
            if k not in _INFO_VOLATILE and sc["info"].get(k) != base["info"].get(k):
                bad.append(f"info.{k}: {sc['info'].get(k)!r} != MIXO s0's {base['info'].get(k)!r}")
        if sc.get("steps_total") != base.get("steps_total"):
            bad.append(f"steps_total {sc.get('steps_total')} != MIXO s0's {base.get('steps_total')}")
    except (KeyError, TypeError) as e:
        bad.append(f"the selfcheck lacks {e!r}")
    return bad


def run_violations(run_dir: str, b_summary: dict, base_argv_json: str, seed: int = 1, base_selfcheck: dict | None = None) -> list:
    """The finished run's OWN records (pulled back): its in-process gate selfcheck passes the same assertions, its summary says arm M2b / this seed / 19,164 steps, and its argv differs
    from MIXO's seed-0 argv in the seed only."""
    def jl(n):
        with open(os.path.join(run_dir, n), encoding="utf-8") as f:
            return json.load(f)
    bad = []
    try:
        sc, sm, ar = jl("selfcheck.json"), jl("summary.json"), jl("argv.json")
    except (OSError, ValueError) as e:
        return [f"cannot read the run's records: {e}"]
    bad += [f"selfcheck: {x}" for x in gate_violations(sc, b_summary)]
    if base_selfcheck is not None:
        bad += [f"selfcheck vs MIXO s0: {x}" for x in same_model_violations(sc, base_selfcheck)]
    if (sm.get("arm"), sm.get("seed"), sm.get("steps"), sm.get("done")) != ("M2b", seed, 19164, True):
        bad.append(f"summary says arm/seed/steps/done {(sm.get('arm'), sm.get('seed'), sm.get('steps'), sm.get('done'))}, expected ('M2b', {seed}, 19164, True)")
    if sm.get("train_sha256") != b_summary.get("train_sha256") or sm.get("n_windows") != b_summary.get("n_windows"):
        bad.append("the run's train list / window count are not B's")
    if not (sm.get("argv_sha") == ar.get("sha") == sc.get("argv_sha")) or not sm.get("argv_sha"):
        bad.append(f"argv_sha disagrees between summary / argv.json / selfcheck: {sm.get('argv_sha')} / {ar.get('sha')} / {sc.get('argv_sha')}")
    base = load_argv(base_argv_json)
    bad += [f"argv: {x}" for x in seed_only_violations(base, list(ar.get("argv") or []), 0, seed)]
    return bad


def make_token(sc_path: str, out_path: str, commit: str, queue_sha: str, key_files: dict, prereg: str = "PREREG_V8F_KINDX (seed-1 confirmation of MIXO's X)") -> dict:
    with open(sc_path, "rb") as f:
        raw = f.read()
    sc = json.loads(raw)
    tok = {"prereg": prereg, "gate": "PASS", "commit": commit, "queue_sha256": queue_sha, "selfcheck_sha256": hashlib.sha256(raw).hexdigest(), "argv_sha": sc.get("argv_sha"),
           "key_files_sha256": key_files, "train_sha256": sc["info"]["train_sha256"], "excluded_sha256": sc["info"]["excluded_sha256"],
           "v9_md5": {"v9_labels_train.npz": "83e30bd3791f0767707eade5f4671759", "v9_labels_eval139.npz": "6b5c7f207cffc3b7eb3cd527fd433599"}}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tok, f, indent=1)
    return tok


# ------------------------------------------------------------------------------------------------ CLI
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("evals")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--expect-sha", required=True, help="sha256 of X's ckpt_last.pt as the chain measured it after training")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--v9-dir", required=True)
    p.add_argument("--p10", required=True)
    p.add_argument("--tracks", required=True)
    p.add_argument("--dry-run", action="store_true", help="print the ten `v8f_mini_eval.py run` argvs and exit (no torch, no file)")
    p = sub.add_parser("evals-argv")
    p.add_argument("--seed", type=int, default=1)
    p = sub.add_parser("cmdline")
    p.add_argument("stage_args", nargs=argparse.REMAINDER)
    p = sub.add_parser("mkjob")
    p.add_argument("name")
    p.add_argument("stage_args", nargs=argparse.REMAINDER)
    p = sub.add_parser("assert-argv")
    p.add_argument("--base", required=True, help="MIXO's seed-0 training argv: an argv.json or a file holding the queue's `trainmixo 0` line")
    p.add_argument("--other", required=True, help="the seed-1 argv, same forms")
    p.add_argument("--seeds", type=int, nargs=2, default=(0, 1))
    p = sub.add_parser("assert-gate")
    p.add_argument("--selfcheck", required=True)
    p.add_argument("--b-summary", required=True, help="the banked M2b summary.json the gate is compared with")
    p.add_argument("--line", default=None, help="the queue's printed gate line (the pre-launch gate only)")
    p.add_argument("--base-selfcheck", default=None, help="MIXO's banked s0 gate selfcheck.json: X's built model must equal it (plus o_interface = full)")
    p.add_argument("--pre-flat", action="store_true", help="the selfcheck predates PREREG_V8F_FLAT (no o_interface record): a TEST input; the chain never passes it")
    p.add_argument("--holdout", default=HOLDOUT_SHA)
    p.add_argument("--steps", type=int, default=19164)
    p.add_argument("--windows", type=int, default=306566)
    p = sub.add_parser("assert-run")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--b-summary", required=True)
    p.add_argument("--base-argv", required=True, help="MIXO's banked seed-0 argv.json")
    p.add_argument("--base-selfcheck", default=None, help="MIXO's banked s0 gate selfcheck.json (the built model must equal it, plus o_interface = full)")
    p.add_argument("--seed", type=int, default=1)
    p = sub.add_parser("write-token")
    p.add_argument("--selfcheck", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--queue-sha", required=True)
    p.add_argument("--key-files", required=True, help="json object {path: sha256}")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.mode == "evals":
        return cmd_evals(a)
    if a.mode == "evals-argv":
        toks = tokens(box2_cmdline(["evals", "--seed", str(a.seed), "--expect-sha", "X_sha256_placeholder"]))           # [python, script, "evals", args...]: the job's own line, parsed back
        b = build_parser().parse_args(toks[2:])
        b.dry_run = True
        return cmd_evals(b)
    if a.mode == "cmdline":
        print(box2_cmdline(a.stage_args))
        return 0
    if a.mode == "mkjob":
        sys.stdout.buffer.write(mkjob_text(a.name, a.stage_args).encode("utf-8"))        # bytes: a text-mode stdout would turn CRLF into CRCRLF on Windows
        return 0
    if a.mode == "assert-argv":
        bad = seed_only_violations(load_argv(a.base), load_argv(a.other), *a.seeds)
        print("argv diff: " + ("exactly the seed (and the output directory that carries it)" if not bad else "; ".join(bad)))
        return 0 if not bad else 2
    if a.mode in ("assert-gate", "assert-run"):
        with open(a.b_summary, encoding="utf-8") as f:
            b = json.load(f)
        if a.mode == "assert-gate":
            with open(a.selfcheck, encoding="utf-8") as f:
                sc = json.load(f)
            bad = gate_violations(sc, b, holdout_sha=a.holdout, steps=a.steps, windows=a.windows, line=a.line, require_o_interface=not a.pre_flat)
            if a.base_selfcheck:
                with open(a.base_selfcheck, encoding="utf-8") as f:
                    bad += [f"vs MIXO s0: {x}" for x in same_model_violations(sc, json.load(f))]
            i, lv = sc.get("info", {}), sc.get("levers", {})
            print("X gate:", "PASS" if not bad else bad, "| clips", i.get("n_clips"), "steps", sc.get("steps_total"), "windows", i.get("n_windows_all_clips"), "| built code_mix", lv.get("code_mix"),
                  "| built o_interface", lv.get("o_interface"), "| built p_gt", lv.get("p_gt_start"), "->", lv.get("p_gt_end"), "| probe partial", (sc.get("code_mix_probe") or {}).get("n_partial"), "of",
                  (sc.get("code_mix_probe") or {}).get("n_draws_samples"), "| params", lv.get("n_params_trainable"), "(B", b.get("levers", {}).get("n_params_trainable"), ") | lru", i.get("lru"))
        else:
            base_sc = None
            if a.base_selfcheck:
                with open(a.base_selfcheck, encoding="utf-8") as f:
                    base_sc = json.load(f)
            bad = run_violations(a.run_dir, b, a.base_argv, a.seed, base_sc)
            print(f"X s{a.seed} run records:", "PASS (in-process gate, summary and argv: the seed is the only difference from MIXO's s0)" if not bad else bad)
        return 0 if not bad else 2
    tok = make_token(a.selfcheck, a.out, a.commit, a.queue_sha, json.loads(a.key_files))
    print("gate token written for commit", tok["commit"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
