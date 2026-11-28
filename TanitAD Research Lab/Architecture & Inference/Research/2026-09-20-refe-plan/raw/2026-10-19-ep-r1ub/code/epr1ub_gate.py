#!/usr/bin/env python3
r"""EP-R1ub launch gate: mints `epr1ub_gate_token.json` ONLY when every check passes; `verify` is what `launch_epr1ub.ps1` runs before it starts the fit.

WHY (PREREG_EPR1UB.md s6; CLAUDE.md "NO TRAINING LAUNCH WITHOUT A LAUNCH-GATE PASS", PI 2026-09-26).  EP-R1ub is a training run (257 parameters on cached features, 9 configs x 2 seeds x 5 folds); the rule has no size
exemption.  The token is bound to {the landed commit of the PARENT code, sha256 of the exact argv, sha256 of the S0 bank manifest, sha256 of epr1.py AND of the new files, the f10 identity reference, the other inputs}.
Nothing is hand-written: the token is HMAC-signed (same schema `tanitad.launch_gate/1`, same canonical JSON, same key file as the landed EP-R1 gate, whose primitives are IMPORTED from its verified bytes).

STATED DEVIATION (PREREG_EPR1UB s11).  The new files (`epr1ub.py`, `epr1ub_gate.py`, `test_epr1ub.py`) are STAGED, not landed: the agent that writes them never commits.  G-COMMIT therefore (a) binds the
LANDED parents (epr1.py, epr1_gate.py, zero_tail.py, test_epr1.py, epr1_dev.py, template pins) to their blobs at the landed commit, and (b) records the sha256 of the staged files in the token with `landed: false`.
At landing the blobs must equal those hashes.  Nothing is imported or run from unverified bytes: the parents are checked against their blobs and the staged files against the hash recorded in the token.

CHECKS (each has a deliberate-regression arm in test_epr1ub_gate.py that must make the gate FAIL)
  G-COMMIT  the landed parents equal their blobs at the commit; the staged files exist and are hashed (their `landed` flag is reported).
  G-DVB     declared-vs-built: every argv lever equals PREREG_EPR1UB; the BUILT objects (Options, grid_ub(), CfgUB.sha256, module constants, the selection function on canned tables) equal the registered ones.
            Every new lever is NAMED in the argv: --lr-grid, --u-grid, --anchor, --criterion.  A default is not a declaration.
  G-PATH    the dev-gate / navtest guard refuses (the landed check, run against the UB options, which add the read-only f10 dir).
  G-BANK    the landed check: S0_DONE, every bank unit's sha256 equals the manifest, coverage of the selected logs.
  G-LIVE    the landed G-LIVE (C1/C6 on one real unit) AND the UB probes on the same unit, re-implemented here (independent of `reachability_check_ub`): the u term is non-zero on the EP row and EXACTLY 0.0 on
            the other rows and the features; exactly 0.0 with no unsafe entry; bit-identical when the labels or the stored SAFE logits change; changes when the stored UNSAFE logits change; u = 0 is the landed
            loss bit for bit; the fit path moves the row by lr at 1e-3, 3e-3, 1e-2 and by exactly 0.0 at lr = 0; u reaches the optimiser.  No concordance, no label statistic.
  G-SUITE   `test_epr1.py` (>= 46 tests) and `test_epr1ub.py` (>= 39 tests) in full form on a clean export (git archive of the landed code dir at the commit + the hash-recorded staged files): OK, 0 skips.

COMMANDS (exact; run on the dev box)
    mint   : C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe -I C:\Users\Admin\epr1ub\code\epr1ub_gate.py --fraction 10 --commit <40-hex landed commit of the parents>   [--stage c2]
    verify : ... epr1ub_gate.py verify --fraction 10 [--stage c2]        (launch_epr1ub.ps1 does this for you)
    argv   : ... epr1ub_gate.py argv --fraction 10 [--stage c2]          (prints the exact command, JSON)
CPU ONLY.  Never touches the GPU lock.  Never reads the dev-gate or navtest tables.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT_GATE_SHA256 = "5bfe2cf95a7edbfd6ca94ae9f871c0f2bee6afd5e702642e7758246e25a17d6c"      # the LANDED epr1_gate.py (CODE_SHA256.json)
PARENT_EPR1_SHA256 = "a751c63eb8e38343a7ec1286114a158046ce4ec13390b2eea0a4b1d2237c38d8"


def _load_landed_gate():
    p = os.path.join(HERE, "epr1_gate.py")
    with open(p, "rb") as f:
        src = f.read()
    sha = hashlib.sha256(src).hexdigest()
    if sha != PARENT_GATE_SHA256:
        raise RuntimeError(f"epr1_gate.py in {HERE} has sha256 {sha}, the landed pin is {PARENT_GATE_SHA256}: refusing to import an unverified gate")
    cur = sys.modules.get("epr1_gate")
    if cur is not None:
        # the sha256 above covers the file at `p`; a module already registered under this name is reused ONLY if it is that file (epr1ub.py's `_load_parent` makes the same check for epr1.py).
        # Before EPR1UB-GATE-CACHE the registered module was returned whatever file it came from, so a decoy `epr1_gate` (the HMAC primitives!) imported earlier in the process was never looked at.
        if os.path.normcase(os.path.abspath(getattr(cur, "__file__", "") or "")) == os.path.normcase(os.path.abspath(p)):
            return cur
        raise RuntimeError(f"a different 'epr1_gate' module is already imported ({getattr(cur, '__file__', None)!r}), not the verified {p}: refusing to use it")
    mod = types.ModuleType("epr1_gate")
    mod.__file__ = p
    sys.modules["epr1_gate"] = mod
    exec(compile(src, p, "exec", dont_inherit=True), mod.__dict__)
    return mod


EG = _load_landed_gate()

GATE_SCHEMA, TOKEN_KIND = EG.GATE_SCHEMA, EG.TOKEN_KIND
PROFILE = "ep-r1ub"
PROFILE_SYNTHETIC = "ep-r1ub-synthetic"
TOKEN_NAME = "epr1ub_gate_token.json"
FAIL_NAME = "epr1ub_gate_FAIL.json"
CHECKS = EG.CHECKS
STAGES = ("run", "c2")

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# REGISTERED values (PREREG_EPR1UB.md).  Hard-coded HERE, independent of epr1ub.py, so the gate compares the built object against the registration and not against itself.
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
REG_LR_GRID = (1e-3, 3e-3, 1e-2)
REG_U_GRID = (0.03, 0.1, 0.3)
REG_ANCHOR = 0.1
REG_UNSAFE_TARGET = 0.0
REG_UNSAFE_NORM = "mean_over_unsafe_entries"
REG_CRITERION = "feasible_oof_dpdms"
REG_SEL_BOUND = 0.0
REG_SEL_UPPER_PCT = 97.5
REG_SEL_SEEDS_RULE = "all"
REG_SEL_OBJECTIVE = "mean_over_seeds_oof_pick_dpdms_x100"
REG_SEL_TIE = "first_in_grid_order"
REG_IDENT = (1e-3, 0.0)
REG_F10_GRID_ID = "lr0.001_anchor0.1"
REG_F10_RESULT_SHA256 = "eb573404bfd02bb8243cbb18cba954e2136bdcd0e3874cc21fb93e647e142463"
REG_F10_ROW_SHA256 = {"0": "f775753f0817d53607b0044e9d33f6d2c5f5b01b73a8d20567d6394ca393fa95", "1": "636f6270f52ef462fb6a1a56a990aa8c683f3c30a3394091b77d10f439852fa8"}
REG_LR_STEP_TOL = 0.01
REG_FRACTION = 10                                  # PREREG_EPR1UB s2: f = 10 % (R7: "same epochs, folds, seeds, fit logs and f = 10 %")
REG_F10_DIR = "C:/Users/Admin/epr1/run/f10"
REG_MIN_TESTS = {"test_epr1.py": 46, "test_epr1ub.py": 39}
REQUIRED_EXPLICIT = ("--fraction", "--out-dir", "--threads", "--epochs", "--batch-sets", "--seeds", "--lr-grid", "--u-grid", "--anchor", "--criterion")
PATH_FLAGS = EG.PATH_FLAGS + ("--f10-dir",)

DEFAULT_PYTHON = EG.DEFAULT_PYTHON
DEFAULT_CODE_DIR = r"C:\Users\Admin\epr1ub\code"
DEFAULT_RUN_ROOT = r"C:\Users\Admin\epr1ub\run"
DEFAULT_GATE_ROOT = r"C:\Users\Admin\epr1ub\gate"
DEFAULT_GIT_DIR = EG.DEFAULT_GIT_DIR
DEFAULT_REPO_CODE_PATH = EG.DEFAULT_REPO_CODE_PATH           # the LANDED parents' directory
LANDED_FILES = ("epr1.py", "epr1_gate.py", "zero_tail.py", "test_epr1.py", "epr1_dev.py", "template_pins.json", "template_pins_make.py")
STAGED_FILES = ("epr1ub.py", "epr1ub_gate.py", "test_epr1ub.py")
BOUND_CODE_FILES = ("epr1.py", "epr1_gate.py", "epr1ub.py", "epr1ub_gate.py")
UB_MODULE_FILE = "epr1ub.py"


class GateError(RuntimeError):
    """A condition under which the gate cannot certify anything."""


_res = EG._res
_norm = EG._norm
sha256_bytes, sha256_file, canonical_json, argv_sha256 = EG.sha256_bytes, EG.sha256_file, EG.canonical_json, EG.argv_sha256
join_path = EG.join_path


def build_argv(*, python: str, code_dir: str, stage: str, fraction: int, out_dir: str, threads: int, epochs: int, batch_sets: int, seeds: str, lr_grid: str, u_grid: str, anchor: str,
               criterion: str, extra=()) -> list:
    """THE exact command.  Every lever is NAMED (a flag whose default happens to be right is still declared).  launch_epr1ub.ps1 runs exactly this list."""
    return [python, "-I", join_path(code_dir, UB_MODULE_FILE), stage, "--fraction", str(fraction), "--out-dir", out_dir, "--threads", str(threads), "--epochs", str(epochs), "--batch-sets",
            str(batch_sets), "--seeds", seeds, "--lr-grid", lr_grid, "--u-grid", u_grid, "--anchor", anchor, "--criterion", criterion, *list(extra)]


@dataclasses.dataclass
class GateConfig:
    commit: str = ""
    fraction: int = 10
    stage: str = "run"
    python: str = DEFAULT_PYTHON
    code_dir: str = DEFAULT_CODE_DIR
    out_dir: str = ""
    threads: int = 8
    epochs: int = EG.REG_EPOCHS
    batch_sets: int = EG.REG_BATCH_SETS
    seeds: str = "0,1"
    lr_grid: str = "0.001,0.003,0.01"
    u_grid: str = "0.03,0.1,0.3"
    anchor: str = "0.1"
    criterion: str = REG_CRITERION
    replacements_file: str | None = None
    git_dir: str = DEFAULT_GIT_DIR
    repo_code_path: str = DEFAULT_REPO_CODE_PATH
    token_path: str = ""
    key_file: str | None = None
    suite_timeout_s: int = 3000
    suite_tests: tuple = (("test_epr1.py", REG_MIN_TESTS["test_epr1.py"]), ("test_epr1ub.py", REG_MIN_TESTS["test_epr1ub.py"]))

    def resolved_out_dir(self) -> str:
        return self.out_dir or f"{DEFAULT_RUN_ROOT}\\f{self.fraction}"

    def resolved_token_path(self) -> str:
        return self.token_path or os.path.join(DEFAULT_GATE_ROOT, f"f{self.fraction}_{self.stage}", TOKEN_NAME)

    def extra_cli(self) -> list:
        return ["--replacements-file", self.replacements_file] if self.replacements_file else []

    def argv(self) -> list:
        return build_argv(python=self.python, code_dir=self.code_dir, stage=self.stage, fraction=self.fraction, out_dir=self.resolved_out_dir(), threads=self.threads, epochs=self.epochs,
                          batch_sets=self.batch_sets, seeds=self.seeds, lr_grid=self.lr_grid, u_grid=self.u_grid, anchor=self.anchor, criterion=self.criterion, extra=self.extra_cli())


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# G-COMMIT (parents landed, new files staged and hashed)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def check_commit(*, commit: str, git_dir: str, repo_code_path: str, code_dir: str, landed=LANDED_FILES, staged=STAGED_FILES) -> dict:
    reasons, det = [], {"commit": commit, "landed_files": {}, "staged_files": {}}
    try:
        if not EG.git_is_commit(git_dir, commit):
            return _res("FAIL", [f"{commit!r} is not a 40-hex commit object in {git_dir}"], det)
        for f in landed:
            p = os.path.join(code_dir, f)
            if not os.path.isfile(p):
                reasons.append(f"{f} is not deployed in {code_dir}")
                continue
            disk = sha256_file(p)
            blob = EG.git_blob(git_dir, commit, f"{repo_code_path}/{f}")
            if blob is None:
                reasons.append(f"{f} is not in commit {commit[:12]} (a landed parent must be)")
            elif sha256_bytes(blob) != disk:
                reasons.append(f"{f} differs from its blob at {commit[:12]} (disk {disk[:12]}, blob {sha256_bytes(blob)[:12]})")
            det["landed_files"][f] = disk
        if det["landed_files"].get("epr1.py") not in (None, PARENT_EPR1_SHA256):
            reasons.append(f"epr1.py sha256 {det['landed_files'].get('epr1.py')} is not the pin {PARENT_EPR1_SHA256}")
        for f in staged:
            p = os.path.join(code_dir, f)
            if not os.path.isfile(p):
                reasons.append(f"staged file {f} is not in {code_dir}")
                continue
            disk = sha256_file(p)
            blob = EG.git_blob(git_dir, commit, f"{repo_code_path}/{f}")
            det["staged_files"][f] = {"sha256": disk, "landed": bool(blob is not None and sha256_bytes(blob) == disk)}
    except Exception as e:                                       # noqa: BLE001 -- a refusal, never a pass
        reasons.append(f"{type(e).__name__}: {str(e)[:300]}")
    return _res("FAIL" if reasons else "PASS", reasons, det)


_LOAD_SEQ = itertools.count(1)                 # every load gets its own module name; nothing is ever looked up by it


def load_verified_ub(code_dir: str, expect_sha256: str | None = None):
    """epr1ub.py executed from bytes that were read once and hashed (no check-then-use gap).  Its own `_load_parent` then verifies epr1.py against the pin.

    NOT CACHED, ON PURPOSE (BACKLOG EPR1UB-GATE-CACHE; `D-REFE-EPR1UB-1`).  What a call certifies is epr1ub.py AND the epr1.py beside it, but the cache that used to be here was keyed on the sha256 of epr1ub.py alone:
    after one clean load, a second code_dir holding a byte-identical epr1ub.py and a TAMPERED epr1.py hit the cache, and `_load_parent` (the only place epr1.py is checked) never ran.  The same hole opens when
    epr1.py is edited in the SAME directory after a clean load.  A cache key has to name every input of the verification (the ub bytes, each parent file `_load_parent` reads, the directory it reads them from), and the
    next input added to `_load_parent` would be left out of the key again.  A repeat load costs ~12 ms (MEASURED 2026-10-19, median of 20, range 9.9-13.7 ms; the first load of a process is 7.5 s, which is the torch import
    and is not repeated), so every call re-reads and re-hashes instead: there is no key to get wrong.
    Each load registers its own module name (dataclasses with `from __future__ import annotations` look the module up in sys.modules while the class bodies run), and a load that raises removes it again."""
    p = os.path.join(code_dir, UB_MODULE_FILE)
    src = Path(p).read_bytes()
    sha = sha256_bytes(src)
    if expect_sha256 is not None and sha != expect_sha256:
        raise GateError(f"{UB_MODULE_FILE} sha256 {sha} != the recorded {expect_sha256}")
    name = f"epr1ub__gate_{sha[:12]}_{next(_LOAD_SEQ)}"
    while name in sys.modules:                     # another instance of this gate (a test mutant of it) may already hold the name
        name = f"epr1ub__gate_{sha[:12]}_{next(_LOAD_SEQ)}"
    mod = types.ModuleType(name)
    mod.__file__ = p
    sys.modules[name] = mod
    try:
        exec(compile(src, p, "exec", dont_inherit=True), mod.__dict__)
    except BaseException:
        sys.modules.pop(name, None)                # a refused or failed load leaves nothing behind
        raise
    return mod, sha


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# G-DVB
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def _canned_row(i, hi, d):
    return {"id": f"c{i}", "per_seed": {str(s): {"pooled": 0.9, "boot": {"delta_unsafe_rate": {"hi": hi[s], "lo": hi[s] - 0.01, "point": hi[s] - 0.005}, "delta_pdms_x100": {"point": d[s]}}} for s in (0, 1)}}


def selection_behaviour(mod) -> list:
    """The BUILT selection function on canned tables with known answers (independent of the module's own tests).  -> reasons."""
    out = []

    def chk(name, got, want):
        if got != want:
            out.append(f"select_cstar: {name}: got {got!r}, registered behaviour {want!r}")
    S = (0, 1)
    nan = float("nan")
    chk("infeasible-highest-not-chosen", mod.select_cstar([_canned_row(0, [0.01, 0.01], [5, 5]), _canned_row(1, [-0.001, -0.002], [0.1, 0.2])], S)["id"], "c1")
    chk("both-seeds-required", mod.select_cstar([_canned_row(0, [-0.01, 0.002], [9, 9]), _canned_row(1, [-0.001, -0.001], [0.5, 0.5])], S)["id"], "c1")
    chk("hi==0.0-feasible", mod.select_cstar([_canned_row(0, [0.0, 0.0], [1, 1])], S)["status"], "FEASIBLE")
    chk("hi>0-infeasible", mod.select_cstar([_canned_row(0, [1e-12, 0.0], [1, 1])], S)["status"], "NONE-FEASIBLE")
    chk("nan-infeasible", mod.select_cstar([_canned_row(0, [nan, 0.0], [1, 1])], S)["status"], "NONE-FEASIBLE")
    chk("ties-first", mod.select_cstar([_canned_row(0, [-1, -1], [0.3, 0.3]), _canned_row(1, [-1, -1], [0.3, 0.3])], S)["id"], "c0")
    chk("mean-over-seeds", mod.select_cstar([_canned_row(0, [-1, -1], [1.0, -1.0]), _canned_row(1, [-1, -1], [0.1, 0.1])], S)["id"], "c1")
    chk("argmax", mod.select_cstar([_canned_row(0, [-1, -1], [0.1, 0.1]), _canned_row(1, [-1, -1], [0.4, 0.2]), _canned_row(2, [-1, -1], [0.2, 0.2])], S)["id"], "c1")
    chk("negative-dpdms-still-chosen", mod.select_cstar([_canned_row(0, [-1, -1], [-0.5, -0.5])], S)["id"], "c0")
    r = _canned_row(0, [0.01, 0.01], [9, 9])
    for s in ("0", "1"):
        r["per_seed"][s]["boot"]["delta_unsafe_rate"]["lo"] = -0.5
    chk("upper-bound-not-lower", mod.select_cstar([r, _canned_row(1, [-1, -1], [0.1, 0.1])], S)["id"], "c1")
    chk("none-feasible-has-no-cstar", (lambda x: (x["status"], x["id"], x["index"]))(mod.select_cstar([_canned_row(0, [0.5, 0.5], [1, 1])], S)), ("NONE-FEASIBLE", None, None))
    return out


def check_dvb(modub, argv: list, *, stage: str, strict: bool = True) -> tuple:
    """-> (check dict, Options or None).  Inspects the BUILT objects, not the argv text."""
    reasons, det = [], {}
    opts = None
    E = modub.E
    try:
        if len(argv) < 4 or argv[1] != "-I" or os.path.basename(str(argv[2]).replace("\\", "/")) != UB_MODULE_FILE:
            return _res("FAIL", [f"argv must start [python, '-I', <...>/{UB_MODULE_FILE}, {stage}]: got {argv[:4]}"], det), None
        cli = argv[3:]
        if cli[0] != stage:
            reasons.append(f"sub-command {cli[0]!r} != the stage being gated {stage!r}")
        pairs, problems = EG._cli_flags(cli[1:])
        reasons += problems
        names = [f for f, _ in pairs]
        for f in sorted({n for n in names if names.count(n) > 1}):
            reasons.append(f"flag {f} is given more than once (argparse would silently keep the last)")
        parser = modub.build_parser()
        known = {s for a in parser._actions for s in a.option_strings}
        for f in names:
            if f not in known:
                reasons.append(f"unknown flag {f}")
        for f in REQUIRED_EXPLICIT:
            if f not in names:
                reasons.append(f"lever {f} is not NAMED in the argv (a default is not a declaration)")
        err = io.StringIO()
        try:
            import contextlib
            with contextlib.redirect_stderr(err):
                ns = parser.parse_args(cli)
        except SystemExit:
            reasons.append(f"epr1ub.py's own parser refuses this argv: {err.getvalue().strip().splitlines()[-1] if err.getvalue().strip() else '(no message)'}")
            return _res("FAIL", reasons, det), None
        opts = modub.options_from_args(ns)
        built = {"epochs": opts.epochs, "batch_sets": opts.batch_sets, "seeds": tuple(opts.seeds), "criterion": opts.criterion, "lr_grid": tuple(opts.lr_grid), "u_grid": tuple(opts.u_grid),
                 "anchor": opts.anchor, "fraction": opts.fraction}
        want = {"epochs": EG.REG_EPOCHS, "batch_sets": EG.REG_BATCH_SETS, "seeds": EG.REG_SEEDS, "criterion": REG_CRITERION, "lr_grid": REG_LR_GRID, "u_grid": REG_U_GRID, "anchor": REG_ANCHOR}
        for k, v in want.items():
            if built[k] != v:
                reasons.append(f"lever {k} = {built[k]!r}, registered {v!r}")
        if built["fraction"] != REG_FRACTION:
            reasons.append(f"fraction {built['fraction']!r} != the registered f = {REG_FRACTION} %")
        if not (isinstance(opts.threads, int) and opts.threads >= 1):
            reasons.append(f"threads {opts.threads!r} must be a positive integer")
        det["levers"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in built.items()}
        # --- the grid the module BUILDS
        cfgs = opts.grid()
        want_grid = [(lr, u) for lr in REG_LR_GRID for u in REG_U_GRID]
        got_grid = [(c.lr, c.u) for c in cfgs]
        det["n_configs_built"] = len(cfgs)
        if len(cfgs) != 9:
            reasons.append(f"the built grid has {len(cfgs)} configurations, registered 9 (3 lr x 3 u)")
        if got_grid != want_grid:
            reasons.append(f"the built grid {got_grid} != the registered {want_grid} (order is part of the registration: ties go to the first in grid order)")
        for c in cfgs:
            if (c.k, c.pair_w, c.epochs, c.batch_sets, c.anchor) != (EG.REG_K, EG.REG_PAIR_W, EG.REG_EPOCHS, EG.REG_BATCH_SETS, REG_ANCHOR):
                reasons.append(f"built config {c.id()}: (k, pair_w, epochs, batch_sets, anchor) = {(c.k, c.pair_w, c.epochs, c.batch_sets, c.anchor)}")
            payload = {"lr": c.lr, "anchor": REG_ANCHOR, "epochs": EG.REG_EPOCHS, "batch_sets": EG.REG_BATCH_SETS, "k": EG.REG_K, "pair_w": EG.REG_PAIR_W, "u": c.u, "optimizer": "adam",
                       "betas": list(EG.REG_ADAM_BETAS), "eps": EG.REG_ADAM_EPS, "weight_decay": 0.0, "row": EG.REG_EP_COL, "target": EG.REG_TARGET_ID, "safe_min": EG.REG_SAFE_MIN,
                       "unsafe_anchor_target": REG_UNSAFE_TARGET, "unsafe_anchor_norm": REG_UNSAFE_NORM}
            if c.sha256() != sha256_bytes(json.dumps(payload, sort_keys=True, indent=1, ensure_ascii=True).encode("utf-8")):
                reasons.append(f"built config {c.id()}: its config sha256 does not equal the registered payload (Adam {EG.REG_ADAM_BETAS}, unsafe target {REG_UNSAFE_TARGET}, norm {REG_UNSAFE_NORM})")
            if c.parent_cfg().sha256() != E.Cfg(lr=c.lr, anchor=REG_ANCHOR, epochs=EG.REG_EPOCHS, batch_sets=EG.REG_BATCH_SETS).sha256():
                reasons.append(f"built config {c.id()}: parent_cfg() is not the landed Cfg the frozen dev applier reconstructs")
        # --- module constants: epr1ub.py, then the landed epr1.py it imports
        ub_consts = {"UB_GRID_LR": REG_LR_GRID, "UB_GRID_U": REG_U_GRID, "UB_ANCHOR": REG_ANCHOR, "UNSAFE_TARGET": REG_UNSAFE_TARGET, "UNSAFE_NORM": REG_UNSAFE_NORM, "CRITERION_REG": REG_CRITERION,
                     "SEL_UPPER_PCT": REG_SEL_UPPER_PCT, "SEL_BOUND": REG_SEL_BOUND, "SEL_SEEDS_RULE": REG_SEL_SEEDS_RULE, "SEL_OBJECTIVE": REG_SEL_OBJECTIVE, "SEL_TIE": REG_SEL_TIE,
                     "IDENT_LR": REG_IDENT[0], "IDENT_U": REG_IDENT[1], "F10_GRID_ID": REG_F10_GRID_ID, "F10_RESULT_SHA256": REG_F10_RESULT_SHA256, "F10_ROW_SHA256": REG_F10_ROW_SHA256,
                     "PARENT_EPR1_SHA256": PARENT_EPR1_SHA256, "LR_STEP_TOL": REG_LR_STEP_TOL}
        for k, v in ub_consts.items():
            if getattr(modub, k, "<absent>") != v:
                reasons.append(f"epr1ub.{k} = {getattr(modub, k, '<absent>')!r}, registered {v!r}")
        if tuple(getattr(modub, "CRITERIA", ())) != (REG_CRITERION, "pooled_oof"):
            reasons.append(f"epr1ub.CRITERIA = {getattr(modub, 'CRITERIA', None)!r}")
        e_consts = {"K": EG.REG_K, "EP_COL": EG.REG_EP_COL, "NC_COL": 0, "DAC_COL": 1, "TTC_COL": 3, "C_COL": 4, "SAFE_MIN": EG.REG_SAFE_MIN, "TARGET_ID": EG.REG_TARGET_ID, "C2_SHUFFLE_SEED": EG.REG_C2_SEED,
                    "N_FOLDS": EG.REG_N_FOLDS, "FOLD_SALT": EG.REG_FOLD_SALT, "RULE": EG.REG_RULE, "ADAM_BETAS": EG.REG_ADAM_BETAS, "ADAM_EPS": EG.REG_ADAM_EPS, "SEEDS": EG.REG_SEEDS,
                    "BOOT_B": EG.REG_BOOT_B, "BOOT_SEED": EG.REG_BOOT_SEED, "N_FIT_LOGS": EG.REG_N_FIT_LOGS, "MIN_LOGS_PER_FOLD": EG.REG_MIN_LOGS_PER_FOLD, "RANKED_LOGS_SHA256": EG.REG_RANKED_SHA256,
                    "ROW_CONSISTENCY_TOL": EG.REG_ROW_TOL, "DEFAULT_EPOCHS": EG.REG_EPOCHS, "DEFAULT_BATCH_SETS": EG.REG_BATCH_SETS}
        for k, v in e_consts.items():
            if getattr(E, k, "<absent>") != v:
                reasons.append(f"epr1.{k} = {getattr(E, k, '<absent>')!r}, registered {v!r}")
        if tuple(getattr(E, "LABEL_ORDER", ())) != EG.REG_LABEL_ORDER:
            reasons.append(f"epr1.LABEL_ORDER = {getattr(E, 'LABEL_ORDER', None)!r}")
        # --- the BUILT selection function (behaviour, not constants)
        reasons += selection_behaviour(modub)
        if strict:
            for k, v in EG.REG_PATHS.items():
                if _norm(getattr(opts, k)) != _norm(v):
                    reasons.append(f"{k} = {getattr(opts, k)!r}, registered {v!r}")
            if _norm(opts.f10_dir) != _norm(REG_F10_DIR):
                reasons.append(f"f10_dir = {opts.f10_dir!r}, registered {REG_F10_DIR!r}")
            if opts.pins_overridden():
                reasons.append(f"pins overridden: {opts.pins_overridden()}")
            for k, v in (("ranked_sha256", EG.REG_RANKED_SHA256), ("expected_n_logs", EG.REG_N_FIT_LOGS), ("min_logs_per_fold", EG.REG_MIN_LOGS_PER_FOLD), ("row_tol", EG.REG_ROW_TOL)):
                if getattr(opts, k) != v:
                    reasons.append(f"pin {k} = {getattr(opts, k)!r}, registered {v!r}")
            if opts.ident_ref is not None:
                reasons.append("ident_ref is injected (API-only): the identity arm must read the registered f10 files")
        det["strict"] = bool(strict)
    except Exception as e:                                       # noqa: BLE001
        reasons.append(f"{type(e).__name__}: {str(e)[:300]}")
    return _res("FAIL" if reasons else "PASS", reasons, det), opts


def check_path_ub(modub, opts, argv: list) -> dict:
    """The landed G-PATH against the UB options, plus the paths it does not know: the read-only f10 dir must not name a dev-gate / navtest token and the guard must still refuse one inside it."""
    base = EG.check_path(modub.E, opts, argv)
    reasons, det = list(base["reasons"]), dict(base["details"])
    try:
        for p in (opts.f10_dir, os.path.join(HERE, UB_MODULE_FILE)):
            hit = modub.E.PathGuard.forbidden(p)
            if hit:
                reasons.append(f"a path of the UB launch names a dev-gate / navtest token {hit}: {p!r}")
        guard = opts.guard()
        guard.check_read(os.path.join(opts.f10_dir, "epr1_result.json"))
        try:
            guard.check_read(os.path.join(opts.f10_dir, "x_navtest_y.json"))
            reasons.append("the guard did NOT refuse a navtest-named file inside the f10 dir")
        except modub.E.PathGuardError:
            pass
        try:
            guard.check_write(os.path.join(opts.f10_dir, "epr1_result.json"))
            reasons.append("the guard allowed a WRITE into the read-only f10 dir")
        except modub.E.PathGuardError:
            pass
        det["f10_dir_read_only_and_guarded"] = True
    except Exception as e:                                       # noqa: BLE001
        reasons.append(f"{type(e).__name__}: {str(e)[:300]}")
    return _res("FAIL" if reasons else "PASS", reasons, det)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# G-LIVE (the landed one, plus the UB probes re-implemented here)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def check_live_ub(modub, opts, aux: dict) -> dict:
    """The landed G-LIVE on one real unit, then the UB probes on the same unit (independent of `reachability_check_ub`; that check's own verdict is recorded as an extra)."""
    import numpy as np
    import torch
    E = modub.E
    base = EG.check_live(E, opts, aux)
    reasons, det = list(base["reasons"]), {"parent_g_live": base["status"], "parent_details": base["details"]}
    tmp = None
    try:
        torch.set_num_threads(max(1, min(int(opts.threads), 4)))
        guard = opts.guard()
        selected, _sha, _rep = EG.selected_rows(E, guard, opts)
        rank_of = {r["log"]: r["rank"] for r in selected}
        unit_logs = aux.get("unit_logs") or EG.unit_log_index(E, guard, opts.bank_dir, [n for n in E._io_listdir(guard, opts.bank_dir) if n.endswith(".sets.npz")])
        best = None
        for n in sorted(unit_logs):
            rk = [rank_of[lg] for lg in unit_logs[n] if lg in rank_of]
            if rk and (best is None or min(rk) < best[0]):
                best = (min(rk), n)
        if best is None:
            return _res("FAIL", reasons + ["no bank unit holds a selected fit log"], det)
        unit = best[1]
        table, _d = EG._unit_table(json.loads(E._io_read_bytes(guard, opts.manifest).decode("utf-8")))
        tmp = tempfile.mkdtemp(prefix="epr1ub_live_")
        shutil.copyfile(os.path.join(opts.bank_dir, unit), os.path.join(tmp, unit))
        sha = sha256_file(os.path.join(tmp, unit))
        if unit not in table or sha != table[unit]:
            return _res("FAIL", reasons + [f"unit {unit}: sha256 {sha[:16]} is not the manifest's"], det)
        with open(os.path.join(tmp, "MANIFEST.sha256.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"files": {unit: {"path": unit, "sha256": sha}}}, f)
        opts_live = dataclasses.replace(opts, bank_dir=tmp, manifest=os.path.join(tmp, "MANIFEST.sha256.json"))
        glive = opts_live.guard()
        in_unit = set(unit_logs[unit])
        sel_unit = [r for r in selected if r["log"] in in_unit]
        W6, b6 = E.load_p2_row(glive, opts.p2_ckpt)
        data = E.load_fit_data(glive, opts_live, sel_unit, {unit: sha}, W6, b6)
        batch = E._probe_batch(data, opts.reach_n_sets)
        s0, y0, z0 = batch["safe"], batch["y"], batch["z6"]
        if int((~s0).sum()) == 0 or int(s0.sum()) == 0:
            return _res("FAIL", reasons + ["the probe batch needs both safe and unsafe proposals"], det)
        det.update({"unit": unit, "n_sets_in_unit": int(data.n_sets), "batch_sets": int(len(batch["feat"]))})
        a_w, p_w, u_pos = REG_ANCHOR, EG.REG_PAIR_W, 0.1

        def probe(pw, aw, u, y=None, safe=None, z6=None, loss_fn=None):
            loss_fn = loss_fn or modub.total_loss_ub
            head = E.RowHead(W6, b6, probe=True, perturb=True)
            feat = batch["feat"].clone().float().requires_grad_(True)
            z = E.forward_z6(head, feat, batch["z6"] if z6 is None else z6)
            loss, _p, _a = loss_fn(z, batch["y"] if y is None else y, batch["safe"] if safe is None else safe, batch["top8"], aw, pw, u)
            if loss.requires_grad:
                loss.backward()
            g = lambda t: 0.0 if t is None else float(t.abs().sum())               # noqa: E731
            gw = head.row_w.grad if head.row_w.grad is not None else torch.zeros_like(head.row_w)
            gb = head.row_b.grad if head.row_b.grad is not None else torch.zeros_like(head.row_b)
            return {"fin": bool(torch.isfinite(loss)), "ep_w": g(head.row_w.grad), "ep_b": g(head.row_b.grad), "fr_w": g(head.frozen_W6.grad), "fr_b": g(head.frozen_b6.grad),
                    "feat": g(feat.grad), "_gw": gw.detach().clone(), "_gb": gb.detach().clone()}
        same = lambda a, b: bool(torch.equal(a["_gw"], b["_gw"]) and torch.equal(a["_gb"], b["_gb"]))      # noqa: E731
        full, uonly, zero = probe(p_w, a_w, u_pos), probe(0.0, 0.0, 1.0), probe(0.0, 0.0, 0.0)
        for name, r in (("full", full), ("u_only", uonly)):
            if not r["fin"]:
                reasons.append(f"{name}: the loss is not finite")
            if not (np.isfinite(r["ep_w"]) and r["ep_w"] > 0.0 and np.isfinite(r["ep_b"]) and r["ep_b"] > 0.0):
                reasons.append(f"{name}: gradient on the EP row is ({r['ep_w']!r}, {r['ep_b']!r}); must be finite and > 0 (the u term does not reach the EP row)")
        for name, r in (("full", full), ("u_only", uonly), ("zero_control", zero)):
            if r["fr_w"] != 0.0 or r["fr_b"] != 0.0 or r["feat"] != 0.0:
                reasons.append(f"{name}: a NON-EP row or the features received a gradient (w {r['fr_w']!r}, b {r['fr_b']!r}, feat {r['feat']!r}); must be EXACTLY 0.0")
        if zero["ep_w"] != 0.0 or zero["ep_b"] != 0.0:
            reasons.append("zero_control: weights (0, 0, 0) still gave an EP-row gradient: the probe is not reading the loss")

        def leaky(z6, y, safe, idx8, aw, pw, u):                  # positive control: a loss reading the TTC column MUST show a non-zero frozen-row gradient
            loss, p, a = modub.total_loss_ub(z6, y, safe, idx8, aw, pw, u)
            return loss + 0.1 * z6[..., 3].pow(2).mean(), p, a
        leak = probe(p_w, a_w, u_pos, loss_fn=leaky)
        if not leak["fr_w"] > 0.0:
            reasons.append("positive control: a loss that reads another column showed NO frozen-row gradient, so a zero reading above would mean nothing")
        # u = 0 is the landed loss, bit for bit
        ref = E._probe_grads(W6, b6, batch, E.total_loss, p_w, a_w)
        u0 = probe(p_w, a_w, 0.0)
        ident = bool(torch.equal(u0["_gw"], ref["_gw"]) and torch.equal(u0["_gb"], ref["_gb"]))
        if not ident:
            reasons.append("u = 0: the gradient is NOT bit-identical to the landed total_loss")
        if same(full, u0):
            reasons.append("positive control: u > 0 did not change the full-loss gradient (the weight is not wired)")
        gen = torch.Generator().manual_seed(77)
        lab_inv = same(probe(0.0, 0.0, 1.0, y=torch.rand(y0.shape, generator=gen)), uonly)
        if not lab_inv:
            reasons.append("u-term label invariance: changing every EP label changed the u-only gradient (the target is not the constant 0)")
        zs = z0.clone()
        zs[..., E.EP_COL] = torch.where(s0, torch.randn(z0[..., E.EP_COL].shape, generator=gen), z0[..., E.EP_COL])
        safe_inv = same(probe(0.0, 0.0, 1.0, z6=zs), uonly)
        if not safe_inv:
            reasons.append("u-term mask: changing the stored EP logits of SAFE proposals changed the u-only gradient (the term acts on safe proposals)")
        zu = z0.clone()
        zu[..., E.EP_COL] = torch.where(~s0, torch.randn(z0[..., E.EP_COL].shape, generator=gen) * 3.0, z0[..., E.EP_COL])
        unsafe_sens = not same(probe(0.0, 0.0, 1.0, z6=zu), uonly)
        if not unsafe_sens:
            reasons.append("positive control: changing the stored EP logits of UNSAFE proposals did NOT change the u-only gradient")
        nou = probe(0.0, 0.0, 1.0, safe=torch.ones_like(s0))
        if nou["ep_w"] != 0.0 or nou["ep_b"] != 0.0:
            reasons.append(f"u term with no unsafe entry: EP-row gradient ({nou['ep_w']!r}, {nou['ep_b']!r}) is not EXACTLY 0.0")
        au_f, au_u = probe(p_w, a_w, u_pos, safe=torch.zeros_like(s0)), probe(0.0, 0.0, u_pos, safe=torch.zeros_like(s0))
        if not (same(au_f, au_u) and au_u["ep_w"] > 0.0):
            reasons.append("all-unsafe batch: the full-loss gradient is not bit-identical to the u-only gradient (a safe term still acts on unsafe proposals) or the u-only gradient is zero")
        zt = z0[..., E.EP_COL]
        verr = float((modub.unsafe_anchor_term(z0, s0) - torch.nn.functional.softplus(zt[~s0]).mean()).abs())
        if not verr <= 1e-6:
            reasons.append(f"the u term differs from mean softplus(z_ep) over the unsafe entries by {verr!r} (wrong target, mask or denominator)")
        # the real fit path
        nb = int(len(batch["feat"]))
        mini = E.FitData(log_names=["probe"], log_rank=np.zeros(1, np.int64), log_fold=np.zeros(1, np.int64), log_idx=np.zeros(nb, np.int64), fold=np.zeros(nb, np.int64), feat=batch["feat"], z6=z0,
                         y=y0, safe=s0, pdms=torch.zeros_like(y0), top8=batch["top8"], provenance={})
        p2w, p2b = np.asarray(W6)[E.EP_COL], np.asarray(b6)[E.EP_COL]

        def one(lr, u):
            c = modub.CfgUB(lr=lr, anchor=REG_ANCHOR, epochs=1, batch_sets=nb, u=u)
            return modub.fit_ub(mini, np.arange(nb), c, 0, W6, b6)
        steps = {}
        for lr in REG_LR_GRID + (0.0,):
            w, b = one(lr, u_pos)
            moved = float(max(np.abs(w - p2w).max(), np.abs(b - p2b).max()))
            steps[f"{lr:g}"] = moved
            if lr == 0.0 and moved != 0.0:
                reasons.append(f"lr = 0 moved the row by {moved!r}: must be EXACTLY 0.0")
            if lr > 0.0 and not (lr * (1 - REG_LR_STEP_TOL) <= moved <= lr * (1 + REG_LR_STEP_TOL)):
                reasons.append(f"lr {lr:g}: one Adam step moved the row by {moved:.6g}, expected {lr:g} +- {100 * REG_LR_STEP_TOL:g} % (the fit path does not use this lr)")
        wu, _ = one(1e-3, u_pos)
        w0, _ = one(1e-3, 0.0)
        u_reaches = not np.array_equal(wu, w0)
        if not u_reaches:
            reasons.append("the fitted row at u > 0 equals the row at u = 0 through fit_ub: the u term does not reach the optimiser")
        own = "GREEN"
        try:
            modub.reachability_check_ub(W6, b6, batch, modub.CfgUB(lr=1e-3, anchor=REG_ANCHOR, epochs=2, batch_sets=32, u=0.1))
        except Exception as e:                                   # noqa: BLE001
            own = f"RED: {str(e)[:200]}"
            reasons.append(f"epr1ub.reachability_check_ub is {own}")
        det.update({"u_only_nonzero": bool(uonly["ep_w"] > 0 and uonly["ep_b"] > 0), "other_rows_and_features_exactly_zero": all(r["fr_w"] == 0.0 and r["fr_b"] == 0.0 and r["feat"] == 0.0 for r in (full, uonly, zero)),
                    "zero_control_exactly_zero": zero["ep_w"] == 0.0 and zero["ep_b"] == 0.0, "leak_control_reads_nonzero": bool(leak["fr_w"] > 0.0), "u0_bit_identical_to_landed": ident,
                    "label_invariant": lab_inv, "safe_logit_invariant": safe_inv, "unsafe_logit_sensitive": unsafe_sens, "no_unsafe_entry_exactly_zero": bool(nou["ep_w"] == 0.0 and nou["ep_b"] == 0.0),
                    "term_value_abs_err": verr, "one_step_move_by_lr": steps, "u_reaches_optimiser": bool(u_reaches), "module_reachability_check_ub": own, "concordance_computed": False})
    except Exception as e:                                       # noqa: BLE001
        reasons.append(f"{type(e).__name__}: {str(e)[:300]}")
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    return _res("FAIL" if reasons else "PASS", reasons, det)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# G-SUITE
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def check_suite(*, commit: str, git_dir: str, repo_code_path: str, code_dir: str, python: str, suite_tests=None, timeout_s: int = 3000, staged=STAGED_FILES, staged_hashes=None) -> dict:
    """Each suite in full form (`python -I <test> -v`, CPU only, 2 threads, one process at a time) on a clean export: the landed code dir at `commit` + the staged files (hash-checked against the
    token's record).  OK, no skips, at least the registered number of tests."""
    reasons, det = [], {"suites": {}}
    suite_tests = suite_tests or (("test_epr1.py", REG_MIN_TESTS["test_epr1.py"]), ("test_epr1ub.py", REG_MIN_TESTS["test_epr1ub.py"]))
    tmp = tempfile.mkdtemp(prefix="epr1ub_suite_")
    try:
        files = EG.export_code_dir(git_dir, commit, repo_code_path, tmp)
        det["n_exported_landed_files"] = len(files)
        for f in staged:
            src = os.path.join(code_dir, f)
            if not os.path.isfile(src):
                reasons.append(f"staged {f} missing from {code_dir}")
                continue
            if staged_hashes and staged_hashes.get(f) and sha256_file(src) != staged_hashes[f]:
                reasons.append(f"staged {f} changed since G-COMMIT hashed it")
            shutil.copyfile(src, os.path.join(tmp, f))
        py = python if python and os.path.exists(python) else sys.executable
        env = EG._clean_env()
        env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
        for test_file, floor in suite_tests:
            if not os.path.isfile(os.path.join(tmp, test_file)):
                reasons.append(f"{test_file} is not in the clean export")
                continue
            t0 = time.time()
            try:
                cp = subprocess.run([py, "-I", test_file, "-v"], cwd=tmp, env=env, capture_output=True, timeout=timeout_s)
            except subprocess.TimeoutExpired:
                reasons.append(f"{test_file} did not finish in {timeout_s} s")
                continue
            out = (cp.stdout + b"\n" + cp.stderr).decode("utf-8", "replace")
            m = re.findall(r"^Ran (\d+) tests? in ([\d.]+)s", out, re.M)
            n = int(m[-1][0]) if m else 0
            lines = [ln for ln in out.strip().splitlines() if ln.strip()]
            status = lines[-1].strip() if lines else ""
            det["suites"][test_file] = {"returncode": cp.returncode, "n_tests": n, "floor": floor, "status_line": status, "seconds": round(time.time() - t0, 1),
                                        "sha256": sha256_file(os.path.join(tmp, test_file))}
            if cp.returncode != 0:
                reasons.append(f"{test_file} exited {cp.returncode}: {status[:80]}; tail: {' | '.join(ln.strip()[:100] for ln in lines[-6:])}")
            if status != "OK":
                reasons.append(f"{test_file}: the final line is {status[:80]!r}, not 'OK' (failures, errors, skips and expected-failures are all refused)")
            if n < floor:
                reasons.append(f"{test_file}: only {n} tests ran, the floor is {floor} (tests deleted or not collected)")
        det["interpreter"] = py
    except Exception as e:                                       # noqa: BLE001
        reasons.append(f"{type(e).__name__}: {str(e)[:300]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return _res("FAIL" if reasons else "PASS", reasons, det)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# orchestration, mint, verify
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def _not_run(why: str) -> dict:
    return _res("NOT-RUN", [why])


def run_checks(cfg: GateConfig, *, overrides: dict | None = None, log=print) -> dict:
    argv = cfg.argv()
    checks: dict = {}
    modub = opts = None
    aux: dict = {}
    mod_sha = None
    checks["G-COMMIT"] = check_commit(commit=cfg.commit, git_dir=cfg.git_dir, repo_code_path=cfg.repo_code_path, code_dir=cfg.code_dir)
    log(f"  G-COMMIT {checks['G-COMMIT']['status']}")
    if checks["G-COMMIT"]["status"] == "PASS":
        try:
            modub, mod_sha = load_verified_ub(cfg.code_dir, checks["G-COMMIT"]["details"]["staged_files"][UB_MODULE_FILE]["sha256"])
        except Exception as e:                                   # noqa: BLE001
            checks["G-COMMIT"] = _res("FAIL", [f"cannot load epr1ub.py from the verified bytes: {type(e).__name__}: {str(e)[:200]}"])
    if modub is None:
        for c in ("G-DVB", "G-PATH", "G-BANK", "G-LIVE"):
            checks[c] = _not_run("epr1ub.py / epr1.py are not verified (G-COMMIT)")
    else:
        checks["G-DVB"], opts = check_dvb(modub, argv, stage=cfg.stage, strict=overrides is None)
        log(f"  G-DVB    {checks['G-DVB']['status']}")
        if opts is None:
            for c in ("G-PATH", "G-BANK", "G-LIVE"):
                checks[c] = _not_run("the argv does not parse (G-DVB)")
        else:
            if overrides:
                opts = dataclasses.replace(opts, **overrides)
            checks["G-PATH"] = check_path_ub(modub, opts, argv)
            log(f"  G-PATH   {checks['G-PATH']['status']}")
            checks["G-BANK"], aux = EG.check_bank(modub.E, opts)
            log(f"  G-BANK   {checks['G-BANK']['status']}")
            checks["G-LIVE"] = check_live_ub(modub, opts, aux) if checks["G-BANK"]["status"] == "PASS" else _not_run("G-BANK is not PASS: no unit is verified to read")
            log(f"  G-LIVE   {checks['G-LIVE']['status']}")
    if checks["G-COMMIT"]["status"] == "PASS":
        staged_hashes = {f: v["sha256"] for f, v in checks["G-COMMIT"]["details"]["staged_files"].items()}
        checks["G-SUITE"] = check_suite(commit=cfg.commit, git_dir=cfg.git_dir, repo_code_path=cfg.repo_code_path, code_dir=cfg.code_dir, python=cfg.python, suite_tests=cfg.suite_tests,
                                        timeout_s=cfg.suite_timeout_s, staged_hashes=staged_hashes)
    else:
        checks["G-SUITE"] = _not_run("the commit is not verified (G-COMMIT)")
    log(f"  G-SUITE  {checks['G-SUITE']['status']}")
    return {"checks": checks, "mod_sha": mod_sha, "opts": opts, "aux": aux, "argv": argv}


def _demote(path: Path, tag: str) -> None:
    if path.exists():
        (path.parent / "superseded").mkdir(parents=True, exist_ok=True)
        os.replace(path, path.parent / "superseded" / f"{path.stem}.{tag}{path.suffix}")


def assemble_token(cfg: GateConfig, run: dict, key: bytes, *, synthetic: bool) -> dict:
    checks, argv = run["checks"], run["argv"]
    reasons = [f"{c}: {r}" for c in CHECKS for r in checks[c]["reasons"]] + [f"{c}: {checks[c]['status']}" for c in CHECKS if checks[c]["status"] not in ("PASS", "FAIL")]
    verdict = "PASS" if all(checks[c]["status"] == "PASS" for c in CHECKS) else "FAIL"
    profile = PROFILE_SYNTHETIC if synthetic else PROFILE
    if verdict == "PASS":
        opts = run["opts"]
        paths = {"bank_manifest": opts.manifest, "ranked_fit_logs": opts.ranked_logs, "p2_ckpt": opts.p2_ckpt, "s0_marker": opts.s0_marker,
                 "f10_result": os.path.join(opts.f10_dir, "epr1_result.json"), "f10_row_seed0": os.path.join(opts.f10_dir, "ep_row_seed0.npy"), "f10_row_seed1": os.path.join(opts.f10_dir, "ep_row_seed1.npy")}
        if opts.replacements_file:
            paths["replacements_file"] = opts.replacements_file
        if cfg.stage == "c2":
            paths["epr1_result"] = os.path.join(cfg.resolved_out_dir(), "epr1_result.json")
            paths["epr1ub_result"] = os.path.join(cfg.resolved_out_dir(), "epr1ub_result.json")
        data = {k: {"path": str(v), **EG.fingerprint_file(v)} for k, v in sorted(paths.items())}
        code_files = {f: sha256_file(os.path.join(cfg.code_dir, f)) for f in BOUND_CODE_FILES}
        cd = checks["G-COMMIT"]["details"]
        binding = {"profile": profile, "commit": cfg.commit, "tree_sha256": EG.tree_digest(code_files), "argv_sha256": argv_sha256(argv), "argv": argv, "stage": cfg.stage, "fraction": cfg.fraction,
                   "trainer": UB_MODULE_FILE, "epr1_sha256": code_files["epr1.py"], "epr1ub_sha256": code_files["epr1ub.py"], "bank_manifest_sha256": data["bank_manifest"]["sha256"],
                   "code_files": code_files, "landed_files": cd["landed_files"], "staged_files": cd["staged_files"], "repo_code_path": cfg.repo_code_path,
                   "deviation": "the staged files are NOT landed (the writing agent never commits): the parents are bound to their blobs at the commit, the staged files to their sha256 (landed: false)",
                   "data": data, "data_sha256": sha256_bytes(canonical_json(data))}
    else:
        binding = {"profile": profile, "commit": cfg.commit, "argv_sha256": argv_sha256(argv), "argv": argv, "stage": cfg.stage, "fraction": cfg.fraction}
    token = {"schema": GATE_SCHEMA, "kind": TOKEN_KIND, "verdict": verdict, "binding": binding, "checks": checks, "reasons": reasons, "gate_sha256": sha256_file(__file__),
             "created_utc": EG.utc_now(), "key_id": EG.key_id(key)}
    token["hmac_sha256"] = EG.sign(token, key)
    return token


def mint(cfg: GateConfig, *, overrides: dict | None = None, log=print) -> dict:
    """Run every check and write `epr1ub_gate_token.json` ONLY if all PASS (else a signed `epr1ub_gate_FAIL.json` and no token)."""
    log(f"EP-R1ub launch gate: stage {cfg.stage}, fraction {cfg.fraction}, commit {cfg.commit[:12]}")
    key = EG.load_key(cfg.key_file, create=True, forbid_under=[cfg.code_dir])
    run = run_checks(cfg, overrides=overrides, log=log)
    token = assemble_token(cfg, run, key, synthetic=bool(overrides))
    tok_path = Path(cfg.resolved_token_path())
    stamp = EG.utc_now().replace(":", "")
    fail_path = tok_path.parent / FAIL_NAME
    if token["verdict"] == "PASS":
        _demote(tok_path, stamp)
        _demote(fail_path, stamp)
        EG.write_json(tok_path, token)
        log(f"PASS: wrote {tok_path}")
        return {"verdict": "PASS", "token_path": str(tok_path), "token": token, "checks": run["checks"], "reasons": []}
    _demote(tok_path, stamp)                                     # a stale PASS must not survive a failing run
    EG.write_json(fail_path, token)
    log(f"FAIL: no token minted; report {fail_path}")
    for r in token["reasons"]:
        log(f"  - {r[:400]}")
    return {"verdict": "FAIL", "token_path": None, "token": token, "checks": run["checks"], "reasons": token["reasons"]}


def verify_token(token_path, argv: list, *, code_dir: str, key: bytes, git_dir: str | None = DEFAULT_GIT_DIR, profile: str = PROFILE) -> tuple:
    """THE LAUNCHER'S SIDE.  Every field the token binds is RE-MEASURED now; any difference refuses.  -> (ok, reasons, info)."""
    reasons, info = [], {"token": str(token_path)}
    try:
        tok = EG.read_json(token_path)
    except Exception as e:                                       # noqa: BLE001
        return False, [f"token unreadable: {type(e).__name__}: {str(e)[:160]}"], info
    if tok.get("kind") != TOKEN_KIND or tok.get("schema") != GATE_SCHEMA:
        reasons.append("not a launch-gate token")
    if not EG.signature_ok(tok, key):
        reasons.append(f"HMAC does not verify: edited, or signed with another key (token key_id {tok.get('key_id')}, this host's {EG.key_id(key)})")
    if tok.get("verdict") != "PASS":
        reasons.append(f"token verdict is {tok.get('verdict')!r}, not PASS")
    b = tok.get("binding") or {}
    info["binding"] = {k: b.get(k) for k in ("profile", "commit", "argv_sha256", "epr1_sha256", "epr1ub_sha256", "bank_manifest_sha256", "stage", "fraction")}
    if b.get("profile") != profile:
        reasons.append(f"token profile {b.get('profile')!r} != the launch profile {profile!r}")
    a_sha = argv_sha256(argv)
    info["argv_sha256"] = a_sha
    if b.get("argv_sha256") != a_sha or b.get("argv") != argv:
        changed = [f"{x} -> {y}" for x, y in zip(b.get("argv") or [], argv) if x != y][:5]
        reasons.append(f"argv differs from the gated argv (sha {a_sha[:12]} vs {str(b.get('argv_sha256'))[:12]}; {changed or 'length differs'})")
    code = b.get("code_files") or {}
    now, n_before = {}, len(reasons)
    for f in BOUND_CODE_FILES:
        p = os.path.join(code_dir, f)
        now[f] = sha256_file(p) if os.path.isfile(p) else None
        if now[f] is None or now[f] != code.get(f):
            reasons.append(f"{f} in {code_dir} is {str(now[f])[:12]}, the gated file is {str(code.get(f))[:12]}")
    if len(reasons) == n_before and EG.tree_digest(now) != b.get("tree_sha256"):
        reasons.append("code tree digest differs from the gated tree")
    if b.get("epr1_sha256") != now.get("epr1.py") or b.get("epr1ub_sha256") != now.get(UB_MODULE_FILE):
        reasons.append("epr1.py / epr1ub.py sha256 differ from the token's")
    for f, ent in (b.get("staged_files") or {}).items():
        p = os.path.join(code_dir, f)
        if f != UB_MODULE_FILE and (not os.path.isfile(p) or sha256_file(p) != ent.get("sha256")):
            reasons.append(f"staged {f} changed since the gate hashed it")
    data = b.get("data") or {}
    if sha256_bytes(canonical_json(data)) != b.get("data_sha256"):
        reasons.append("data_sha256 does not match the bound data block")
    d_bad = []
    for k, want in sorted(data.items()):
        got = EG.fingerprint_file(want.get("path", ""))
        if EG.fp_key(got) != EG.fp_key(want):
            d_bad.append(k)
    if d_bad:
        reasons.append(f"inputs differ from the gated inputs on {d_bad} (bank manifest / ranked logs / P2 checkpoint / S0 marker / f10 identity reference / the stage-1 result)")
    if (data.get("bank_manifest") or {}).get("sha256") != b.get("bank_manifest_sha256"):
        reasons.append("bank_manifest_sha256 is not the bound manifest's sha256")
    commit = b.get("commit")
    if git_dir is not None:
        try:
            if not EG.git_is_commit(git_dir, str(commit)):
                reasons.append(f"commit {str(commit)[:12]} is not a commit object in {git_dir}")
            else:
                for f, disk in (b.get("landed_files") or {}).items():
                    blob = EG.git_blob(git_dir, commit, f"{b.get('repo_code_path')}/{f}")
                    p = os.path.join(code_dir, f)
                    if blob is None or sha256_bytes(blob) != disk or (os.path.isfile(p) and sha256_file(p) != disk):
                        reasons.append(f"landed {f} no longer equals its blob at {str(commit)[:12]}")
        except Exception as e:                                   # noqa: BLE001
            reasons.append(f"cannot re-check the commit blobs: {type(e).__name__}: {str(e)[:160]}")
    info["created_utc"] = tok.get("created_utc")
    return not reasons, reasons, info


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="epr1ub_gate.py", description="EP-R1ub launch gate (mint / verify / argv). CPU only.")
    ap.add_argument("cmd", nargs="?", choices=("mint", "verify", "argv"), default="mint")
    ap.add_argument("--fraction", type=int, required=True, choices=EG.REG_FRACTIONS)
    ap.add_argument("--stage", choices=STAGES, default="run")
    ap.add_argument("--commit", default="", help="the landed commit (40-hex) the deployed PARENT files must equal; required for mint")
    ap.add_argument("--python", default=DEFAULT_PYTHON)
    ap.add_argument("--code-dir", default=DEFAULT_CODE_DIR)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=EG.REG_EPOCHS)
    ap.add_argument("--batch-sets", type=int, default=EG.REG_BATCH_SETS)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--lr-grid", default="0.001,0.003,0.01")
    ap.add_argument("--u-grid", default="0.03,0.1,0.3")
    ap.add_argument("--anchor", default="0.1")
    ap.add_argument("--criterion", default=REG_CRITERION)
    ap.add_argument("--replacements-file", default=None)
    ap.add_argument("--git-dir", default=DEFAULT_GIT_DIR)
    ap.add_argument("--repo-code-path", default=DEFAULT_REPO_CODE_PATH)
    ap.add_argument("--token", default="", help="default: C:\\Users\\Admin\\epr1ub\\gate\\f<fraction>_<stage>\\epr1ub_gate_token.json")
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--suite-timeout-s", type=int, default=3000)
    return ap


def config_from_args(a: argparse.Namespace) -> GateConfig:
    return GateConfig(commit=a.commit, fraction=a.fraction, stage=a.stage, python=a.python, code_dir=a.code_dir, out_dir=a.out_dir, threads=a.threads, epochs=a.epochs, batch_sets=a.batch_sets,
                      seeds=a.seeds, lr_grid=a.lr_grid, u_grid=a.u_grid, anchor=a.anchor, criterion=a.criterion, replacements_file=a.replacements_file, git_dir=a.git_dir,
                      repo_code_path=a.repo_code_path, token_path=a.token, key_file=a.key_file, suite_timeout_s=a.suite_timeout_s)


def main(argv=None, *, _test_overrides: dict | None = None, _out=print) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("mint", "verify", "argv", "-h", "--help"):
        argv = ["mint"] + argv
    a = make_parser().parse_args(argv)
    cfg = config_from_args(a)
    if a.cmd == "argv":
        _out(json.dumps({"argv": cfg.argv()}))
        return 0
    if a.cmd == "verify":
        try:
            key = EG.load_key(cfg.key_file, create=False)
            ok, reasons, info = verify_token(cfg.resolved_token_path(), cfg.argv(), code_dir=cfg.code_dir, key=key, git_dir=cfg.git_dir, profile=PROFILE_SYNTHETIC if _test_overrides else PROFILE)
        except Exception as e:                                   # noqa: BLE001 -- a refusal, never a pass
            ok, reasons, info = False, [f"{type(e).__name__}: {str(e)[:300]}"], {}
        _out(json.dumps({"ok": bool(ok), "reasons": reasons, "argv": cfg.argv(), "token": cfg.resolved_token_path(), "info": info}))
        return 0 if ok else 3
    if not re.fullmatch(r"[0-9a-f]{40}", cfg.commit or ""):
        _out("REFUSED: --commit must be the 40-hex landed commit of the parents")
        return 2
    res = mint(cfg, overrides=_test_overrides, log=_out)
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
