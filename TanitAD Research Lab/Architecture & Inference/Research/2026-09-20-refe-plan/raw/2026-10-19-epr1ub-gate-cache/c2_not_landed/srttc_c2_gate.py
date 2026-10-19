#!/usr/bin/env python3
"""SR-TTC C2 launch gate (stage `c2`): mints `c2_gate_token.json` ONLY when every check passes; `verify` is what the enforcing entry `srttc_c2_run.py` runs before it fits anything.  CPU only.

WHY (MM_RULINGS_HEADROWS.md R5; CLAUDE.md "NO TRAINING LAUNCH WITHOUT A LAUNCH-GATE PASS", PI 2026-09-26): SR-TTC's C2 negative control (PREREG_SELRANK_ARM.md:71) is a shuffled-label head FIT at c*,
i.e. a training run, and the rule has no size exemption.  `srttc_r1_gate.py` (landed, frozen) mints stages `run` and `lc` only; EP-R1's gate has a `c2` stage, SR-TTC's did not, so the landed
`srttc_c2.py` could only be started with `--no-gate-stage-ack` ("the Master Mind rules").  R5 is that ruling: C2 needs a gate token exactly as EP-R1's `--stage c2` does, and `--no-gate-stage-ack` is NOT accepted.
This file is a NEW file; `srttc_r1_gate.py`, `srttc_r1.py` and `srttc_c2.py` are untouched (landed bytes).  It re-uses the frozen gate's primitives and its tested bank / path / real-unit checks (loaded
from the verified bytes, see `_load_frozen_gate`) and adds what is specific to C2.

THE TOKEN binds {landed commit, sha256 of the exact C2 argv, sha256 of the S0 bank manifest, sha256 of srttc_c2.py / srttc_r1.py / srttc_r1_gate.py (each equal to its blob at the commit), the sha256 of
this gate and of the enforcing entry (not landed yet: recorded as `landed: false`, re-measured at every verify), sha256 of r1_result.json (c* is THE registered one), the other inputs C2 reads}.
Nothing is hand-written: the token is HMAC-signed (same schema `tanitad.launch_gate/1`, same key file as the programme's gate).  Profile `srttc-c2`: an R1 `run` or `lc` token can never open C2 and the
reverse (the dev gate reads the R1 `run` token; this one is a different profile and stage).

CHECKS (each has a deliberate-regression arm in test_srttc_c2_gate.py that must make the gate FAIL)
  G-COMMIT  srttc_c2.py, srttc_r1.py and srttc_r1_gate.py equal their git blobs at the commit (a file that is not landed, or edited since, REFUSES); the two NEW files (this gate, the entry) are bound by
            sha256 and disclosed as unlanded, or checked against their blob if the commit has them.  test_srttc_r1.py and test_srttc_dev.py must be at the commit (G-SUITE has something to run).
  G-DVB     declared-vs-built.  (a) every lever of the C2 argv equals the registered one AND the object srttc_r1.py BUILDS (Options, grid(), Cfg.sha256, module constants) equals the registered one: this IS the
            frozen gate's check_dvb, applied to the same argv with the entry name translated; (b) `--no-gate-stage-ack` and the `lc` sub-command are refused; (c) the C2 object srttc_c2.py BUILDS:
            C2_SHUFFLE_SEED = 20261018 (the value srttc_dev.py records, GAP G1: unregistered), RESULT_NAME / ROW_FMT = the files srttc_dev.py reads, and its `R` is THE verified srttc_r1 module;
            (d) c* is declared-vs-built: r1_result.json has the REGISTERED sha256 (an argument of the mint, the same one the dev gate is given), schema/stage/fraction/params/seeds equal, c* is one of the
            registered 9 and `Cfg(...).sha256()` rebuilt from (lr, anchor, epochs, batch-sets) equals the recorded config sha256, `code_sha256` equals the srttc_r1.py being gated, no pin overridden;
            (e) no C2 output exists yet in the out dir (the rows are written once).
  G-PATH    the frozen gate's check_path (dev-gate / navtest refusal, the guard still allows a legitimate path and refuses one outside its allow-list) + the real entry path names no forbidden token.
  G-BANK    the frozen gate's check_bank (S0_DONE, no S0_FAILED, every unit's sha256 equals the manifest, logs_without_any_token <= 5 %, every selected log in the bank, R1's own manifest reader accepts it)
            + r1_result.json was fitted on THIS bank (its bank_manifest_sha256 equals the manifest now).
  G-LIVE    (1) the frozen gate's check_live on the first fit-set unit (R1's fit path reaches the TTC row; EXACTLY 0.0 everywhere else; zero control 0.0; a leaky loss reads non-zero), AND (2) C2's own reach,
            on the same real unit: the shuffle moves a non-zero fraction of labels, keeps every set's label multiset, is deterministic in its seed and DEPENDS on the seed, leaves the real labels and the
            frozen top-8 membership untouched; and THROUGH srttc_c2.fit_c2_rows -> srttc_r1.fit_row the loss receives the SHUFFLED labels (a recording loss sees them, not the real ones), the TTC row moves,
            and the TTC-row gradient on the shuffled labels differs from the gradient on the real labels (a control that must read non-zero) while two evaluations on the same labels are bit-identical (a control
            that must read EXACTLY 0.0).  This is the "built, tested, unreachable from its caller" class: the label flag must reach the loss, not just exist in the library.  No concordance, no dev/navtest table.
  G-SUITE   on a clean `git archive` export of the code dir at the commit: test_srttc_r1.py in full (>= 33 tests, 0 failures, 0 skips) AND the C2 tests of test_srttc_dev.py (`C2Fit`, >= 5 tests).
            CPU only, one process at a time.  The gate's OWN tests (test_srttc_c2_gate.py) are run by the author and banked; they are not part of this check.

COMMANDS (exact; the landed files + the two new ones deployed to the code dir, run on the dev box after R1 has written r1_result.json)
    mint   : C:\\Users\\Admin\\venvs\\driverl-eval\\Scripts\\python.exe -I <code>\\srttc_c2_gate.py --fraction 10 --commit <40-hex landed commit> --r1-result-sha256 <registered sha256 of r1_result.json>
    verify : ... srttc_c2_gate.py verify --fraction 10            (srttc_c2_run.py does this for itself before anything is fitted)
    argv   : ... srttc_c2_gate.py argv --fraction 10               (prints the exact command, JSON)
    run    : C:\\Users\\Admin\\venvs\\driverl-eval\\Scripts\\python.exe -I <code>\\srttc_c2_run.py run --fraction 10 --out-dir ... (the exact `argv` output)

NEVER touches the GPU lock.  Never reads the dev-gate or navtest tables (every file access is behind srttc_r1's own PathGuard).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
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

GATE_SCHEMA = "tanitad.launch_gate/1"
TOKEN_KIND = "launch-gate-token"
PROFILE = "srttc-c2"
PROFILE_SYNTHETIC = "srttc-c2-synthetic"          # a token minted with test overrides can NEVER verify for the real profile
STAGE = "c2"
TOKEN_NAME = "c2_gate_token.json"
FAIL_NAME = "c2_gate_FAIL.json"
ATTEST_NAME = "r1_c2_attestation.json"
CHECKS = ("G-COMMIT", "G-DVB", "G-PATH", "G-BANK", "G-LIVE", "G-SUITE")
RUN_ENTRY = "srttc_c2_run.py"
LANDED_FILES = ("srttc_c2.py", "srttc_r1.py", "srttc_r1_gate.py")           # each must equal its blob at the commit
NEW_FILES = ("srttc_c2_gate.py", RUN_ENTRY)                                   # bound by sha256; blob-checked only if the commit already has them
BOUND_FILES = LANDED_FILES + NEW_FILES

# registered / recorded values, hard-coded HERE (independent of srttc_c2.py, so the gate compares the built object against the record and not against itself)
REG_C2_SHUFFLE_SEED = 20261018                    # srttc_dev.py C2_SHUFFLE_SEED (GAP G1: UNREGISTERED for SR-TTC, EP-R1's value)
REG_C2_RESULT = "r1_c2_result.json"               # srttc_dev.py C2_RESULT
REG_C2_ROW_FMT = "ttc_row_c2shuffle_seed{sd}.npy"  # srttc_dev.py C2_ROW_FMT
REG_R1_SCHEMA = "srttc-r1-v1"
SUITE_REAL = (("test_srttc_r1.py", (), 33), ("test_srttc_dev.py", ("C2Fit",), 5))      # (file, unittest selectors, minimum tests)
DEFAULT_PYTHON = r"C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe"
DEFAULT_CODE_DIR = r"C:\Users\Admin\srttc_r1\code"           # where the landed R1 files are deployed; the 2 new files + srttc_c2.py go next to them (or pass --code-dir)
DEFAULT_GATE_ROOT = r"C:\Users\Admin\srttc_r1\gate"
DEFAULT_RUN_ROOT = r"C:\Users\Admin\srttc_r1\run"
DEFAULT_GIT_DIR = "C:/Users/Admin/tanitad-push/.git"
DEFAULT_REPO_CODE_PATH = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-18-sel-rank/code"


class GateError(RuntimeError):
    """A condition under which the gate cannot certify anything."""


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the frozen gate, loaded from the bytes that are hashed (read once, hashed, exec'd from those same bytes: no check-then-use gap)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def _load_frozen_gate(code_dir: str = HERE):
    p = os.path.join(code_dir, "srttc_r1_gate.py")
    src = Path(p).read_bytes()
    sha = hashlib.sha256(src).hexdigest()
    name = "srttc_r1_gate__c2dep_" + sha[:12]
    if name in sys.modules:
        return sys.modules[name], sha
    mod = types.ModuleType(name)
    mod.__file__ = p
    sys.modules[name] = mod                      # dataclasses (with `from __future__ import annotations`) looks the module up here
    try:
        exec(compile(src, p, "exec", dont_inherit=True), mod.__dict__)
    except BaseException:
        sys.modules.pop(name, None)              # poison guard: _load_frozen_gate -- the lookup above trusts sys.modules, so a half-built module must not stay there for the next caller
        raise
    return mod, sha


G, FROZEN_GATE_SHA = _load_frozen_gate()
sha256_bytes, sha256_file, canonical_json, argv_sha256 = G.sha256_bytes, G.sha256_file, G.canonical_json, G.argv_sha256
_res = G._res


def exec_verified_module(code_dir: str, filename: str, modname: str, expect_sha256: str | None = None):
    """Exec `filename` from the exact bytes that are hashed and register it as `modname__c2g_<sha>`.  -> (module, sha256).  A module of that name that already exists is the same bytes (the name carries the sha)."""
    p = os.path.join(code_dir, filename)
    src = Path(p).read_bytes()
    sha = sha256_bytes(src)
    if expect_sha256 is not None and sha != expect_sha256:
        raise GateError(f"{filename} sha256 {sha} != the verified {expect_sha256}")
    name = f"{modname}__c2g_{sha[:12]}"
    if name in sys.modules:
        return sys.modules[name], sha
    mod = types.ModuleType(name)
    mod.__file__ = p
    sys.modules[name] = mod                      # dataclasses (with `from __future__ import annotations`) looks the module up here
    try:
        exec(compile(src, p, "exec", dont_inherit=True), mod.__dict__)
    except BaseException:
        sys.modules.pop(name, None)              # poison guard: exec_verified_module -- a load that raised is not served to the next caller as if it had been verified and loaded
        raise
    return mod, sha


def load_verified_pair(code_dir: str, expect: dict) -> tuple:
    """-> (srttc_r1 module, srttc_c2 module, shas).  srttc_c2.py does `import srttc_r1 as R`: while it is exec'd, `srttc_r1` resolves to THE verified module (restored afterwards), so the C2 fit runs on the bytes
    that were hashed and not on whatever file the import path would find."""
    mod, sha_r1 = exec_verified_module(code_dir, "srttc_r1.py", "srttc_r1", expect.get("srttc_r1.py"))
    p = os.path.join(code_dir, "srttc_c2.py")
    src = Path(p).read_bytes()
    sha_c2 = sha256_bytes(src)
    if expect.get("srttc_c2.py") is not None and sha_c2 != expect["srttc_c2.py"]:
        raise GateError(f"srttc_c2.py sha256 {sha_c2} != the verified {expect['srttc_c2.py']}")
    name = f"srttc_c2__c2g_{sha_c2[:12]}_{sha_r1[:12]}"
    if name in sys.modules:
        return mod, sys.modules[name], {"srttc_r1.py": sha_r1, "srttc_c2.py": sha_c2}
    c2mod = types.ModuleType(name)
    c2mod.__file__ = p
    sys.modules[name] = c2mod
    prev = sys.modules.get("srttc_r1")
    sys.modules["srttc_r1"] = mod
    try:
        exec(compile(src, p, "exec", dont_inherit=True), c2mod.__dict__)
    except BaseException:
        sys.modules.pop(name, None)              # poison guard: load_verified_pair -- the same, for the C2 module (its name already carries both shas, so a half-built one would be a cache hit)
        raise
    finally:
        if prev is None:
            sys.modules.pop("srttc_r1", None)
        else:
            sys.modules["srttc_r1"] = prev
    return mod, c2mod, {"srttc_r1.py": sha_r1, "srttc_c2.py": sha_c2}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# configuration and the exact command
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def build_argv(*, python: str, code_dir: str, fraction: int, out_dir: str, threads: int, epochs: int, batch_sets: int, seeds: str, criterion: str, both_rule: str, extra=()) -> list:
    """THE exact C2 command.  Every lever is NAMED (a flag whose default happens to be right is still declared)."""
    return [python, "-I", G.join_path(code_dir, RUN_ENTRY), "run", "--fraction", str(fraction), "--out-dir", out_dir, "--threads", str(threads), "--epochs", str(epochs),
            "--batch-sets", str(batch_sets), "--seeds", seeds, "--criterion", criterion, "--both-rule", both_rule, *list(extra)]


@dataclasses.dataclass
class C2Config:
    commit: str = ""
    fraction: int = 10
    python: str = DEFAULT_PYTHON
    code_dir: str = DEFAULT_CODE_DIR
    out_dir: str = ""
    threads: int = 8
    epochs: int = G.REG_EPOCHS
    batch_sets: int = G.REG_BATCH_SETS
    seeds: str = "0,1"
    criterion: str = G.REG_CRITERION
    both_rule: str = G.REG_BOTH_RULE
    replacements_file: str | None = None
    r1_result_sha256: str = ""                      # the REGISTERED sha256 of r1_result.json (written down before any dev read)
    git_dir: str = DEFAULT_GIT_DIR
    repo_code_path: str = DEFAULT_REPO_CODE_PATH
    token_path: str = ""
    key_file: str | None = None
    suite_timeout_s: int = 1800
    suite_specs: tuple = SUITE_REAL
    extra_flags: tuple = ()                         # test hook only (the CLI cannot reach it): extra argv tokens, to prove a refusal

    def resolved_out_dir(self) -> str:
        return self.out_dir or f"{DEFAULT_RUN_ROOT}\\f{self.fraction}"

    def resolved_token_path(self) -> str:
        return self.token_path or default_token_path(self.fraction)

    def argv(self) -> list:
        extra = (["--replacements-file", self.replacements_file] if self.replacements_file else []) + list(self.extra_flags)
        return build_argv(python=self.python, code_dir=self.code_dir, fraction=self.fraction, out_dir=self.resolved_out_dir(), threads=self.threads, epochs=self.epochs,
                          batch_sets=self.batch_sets, seeds=self.seeds, criterion=self.criterion, both_rule=self.both_rule, extra=extra)


def default_token_path(fraction: int) -> str:
    return os.path.join(DEFAULT_GATE_ROOT, f"f{fraction}_{STAGE}", TOKEN_NAME)


def translate_argv(argv: list) -> list:
    """The C2 command with its entry name replaced by srttc_r1.py: the SAME flag set, so the frozen gate's declared-vs-built check applies unchanged."""
    p = str(argv[2])
    i = max(p.rfind("\\"), p.rfind("/"))
    return [argv[0], "-I", p[: i + 1] + "srttc_r1.py", *argv[3:]]


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# G-COMMIT
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def check_commit(*, commit: str, git_dir: str, repo_code_path: str, code_dir: str, suite_files=(), frozen_gate_sha: str | None = None) -> dict:
    reasons, det = [], {"commit": commit, "files": {}, "landed": {}}
    try:
        if not G.git_is_commit(git_dir, commit):
            return _res("FAIL", [f"{commit!r} is not a 40-hex commit object in {git_dir}"], det)
        for f in BOUND_FILES:
            p = os.path.join(code_dir, f)
            if not os.path.isfile(p):
                reasons.append(f"{f} is not deployed in {code_dir}")
                continue
            disk = sha256_file(p)
            blob = G.git_blob(git_dir, commit, f"{repo_code_path}/{f}")
            det["files"][f] = disk
            if f in LANDED_FILES:
                det["landed"][f] = True
                if blob is None:
                    reasons.append(f"{f} is not in commit {commit[:12]} (not landed)")
                elif sha256_bytes(blob) != disk:
                    reasons.append(f"{f} differs from its blob at {commit[:12]} (disk {disk[:12]}, blob {sha256_bytes(blob)[:12]})")
            else:                                                 # a NEW file: unlanded is disclosed; a landed copy that differs is not
                det["landed"][f] = blob is not None
                if blob is not None and sha256_bytes(blob) != disk:
                    reasons.append(f"{f} is in commit {commit[:12]} but differs from the deployed file (disk {disk[:12]}, blob {sha256_bytes(blob)[:12]})")
        if frozen_gate_sha is not None and det["files"].get("srttc_r1_gate.py") not in (None, frozen_gate_sha):
            reasons.append(f"the frozen gate this process imported ({frozen_gate_sha[:12]}) is not the deployed srttc_r1_gate.py ({str(det['files'].get('srttc_r1_gate.py'))[:12]}): it changed after import, or the gate runs from another directory")
        for f in suite_files:
            blob = G.git_blob(git_dir, commit, f"{repo_code_path}/{f}")
            if blob is None:
                reasons.append(f"{f} is not in commit {commit[:12]} (G-SUITE has nothing to run)")
            else:
                det.setdefault("test_blobs", {})[f] = sha256_bytes(blob)
    except Exception as e:                                       # noqa: BLE001 -- a refusal, never a pass
        reasons.append(f"{type(e).__name__}: {str(e)[:300]}")
    return _res("FAIL" if reasons else "PASS", reasons, det)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# G-DVB
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def load_r1_result(mod, opts) -> tuple:
    """-> (doc, sha256, path).  Read through srttc_r1's own PathGuard (the dev-gate / navtest tables are unreachable)."""
    guard = opts.guard()
    path = os.path.join(opts.resolved_out_dir(), "r1_result.json")
    raw = mod._io_read_bytes(guard, path)
    return json.loads(raw.decode("utf-8")), sha256_bytes(raw), path


def check_dvb(mod, c2mod, argv: list, *, code_dir: str, r1_sha256: str, mod_sha: str, strict: bool = True) -> tuple:
    """-> (check dict, Options or None).  Inspects the BUILT objects, not the argv text."""
    reasons, det = [], {}
    opts = None
    try:
        if len(argv) < 5 or argv[1] != "-I" or os.path.basename(str(argv[2]).replace("\\", "/")) != RUN_ENTRY:
            return _res("FAIL", [f"argv must start [python, '-I', <...>/{RUN_ENTRY}, run]: got {argv[:4]}"], det), None
        if argv[3] != "run":
            reasons.append(f"sub-command {argv[3]!r} != 'run' (C2 has no learning-curve stage; the lc sub-command is refused)")
        if any(str(t).split("=")[0] == "--no-gate-stage-ack" for t in argv):
            reasons.append("--no-gate-stage-ack is NOT accepted (MM R5): a shuffled-label C2 fit is a fit and needs a gate token")
        if os.path.normcase(os.path.dirname(os.path.abspath(str(argv[2])))) != os.path.normcase(os.path.abspath(code_dir)):
            reasons.append(f"the entry {argv[2]!r} is not in the gated code dir {code_dir!r}")
        frozen_res, opts = G.check_dvb(mod, translate_argv(argv), stage="run", strict=strict)
        reasons += list(frozen_res["reasons"])
        det.update(frozen_res["details"])
        # --- the C2 object srttc_c2.py BUILDS
        if getattr(c2mod, "C2_SHUFFLE_SEED", None) != REG_C2_SHUFFLE_SEED:
            reasons.append(f"srttc_c2.C2_SHUFFLE_SEED = {getattr(c2mod, 'C2_SHUFFLE_SEED', '<absent>')!r}, recorded {REG_C2_SHUFFLE_SEED} (srttc_dev.py C2_SHUFFLE_SEED)")
        if getattr(c2mod, "RESULT_NAME", None) != REG_C2_RESULT:
            reasons.append(f"srttc_c2.RESULT_NAME = {getattr(c2mod, 'RESULT_NAME', '<absent>')!r}, the dev gate reads {REG_C2_RESULT!r}")
        if getattr(c2mod, "ROW_FMT", None) != REG_C2_ROW_FMT:
            reasons.append(f"srttc_c2.ROW_FMT = {getattr(c2mod, 'ROW_FMT', '<absent>')!r}, the dev gate reads {REG_C2_ROW_FMT!r}")
        for fn in ("run_c2", "fit_c2_rows", "shuffle_labels_within_set"):
            if not callable(getattr(c2mod, fn, None)):
                reasons.append(f"srttc_c2.{fn} is not callable")
        if getattr(c2mod, "R", None) is not mod:
            reasons.append("srttc_c2.R is not THE verified srttc_r1 module the gate checked (it would fit with other code)")
        det["c2_shuffle_seed"] = getattr(c2mod, "C2_SHUFFLE_SEED", None)
        # --- c*: declared-vs-built
        if opts is not None:
            if not re.fullmatch(r"[0-9a-f]{64}", r1_sha256 or ""):
                reasons.append("the registered sha256 of r1_result.json was not given (--r1-result-sha256): c* is not pinned")
            try:
                doc, sha, path = load_r1_result(mod, opts)
            except Exception as e:                               # noqa: BLE001
                doc = None
                reasons.append(f"r1_result.json unreadable in {opts.resolved_out_dir()}: {type(e).__name__}: {str(e)[:200]}")
            if doc is not None:
                det["r1_result_sha256"] = sha
                if r1_sha256 and sha != r1_sha256:
                    reasons.append(f"r1_result.json sha256 {sha[:16]} != the REGISTERED {r1_sha256[:16]}: C2 would be fitted at a c* nobody registered")
                if doc.get("schema") != REG_R1_SCHEMA or doc.get("stage") != "R1":
                    reasons.append(f"r1_result.json schema/stage {doc.get('schema')}/{doc.get('stage')} != {REG_R1_SCHEMA}/R1")
                if doc.get("fraction_pct") != opts.fraction:
                    reasons.append(f"r1_result.json is for f = {doc.get('fraction_pct')} %, not {opts.fraction} %")
                p = doc.get("params") or {}
                if (p.get("epochs"), p.get("batch_sets"), [int(s) for s in p.get("seeds", [])], p.get("criterion")) != (opts.epochs, opts.batch_sets, list(opts.seeds), opts.criterion):
                    reasons.append(f"r1_result.json params (epochs, batch_sets, seeds, criterion) = {(p.get('epochs'), p.get('batch_sets'), p.get('seeds'), p.get('criterion'))} differ from this argv")
                cs = doc.get("c_star") or {}
                if (cs.get("lr"), cs.get("anchor")) not in {(lr, a) for lr in G.REG_GRID_LR for a in G.REG_GRID_ANCHOR}:
                    reasons.append(f"c* (lr {cs.get('lr')}, anchor {cs.get('anchor')}) is not one of the registered 9 configurations")
                else:
                    rebuilt = mod.Cfg(lr=cs["lr"], anchor=cs["anchor"], epochs=opts.epochs, batch_sets=opts.batch_sets).sha256()
                    if rebuilt != cs.get("config_sha256"):
                        reasons.append(f"c* rebuilt from (lr, anchor, epochs, batch-sets) has config sha256 {rebuilt[:16]}, r1_result.json records {str(cs.get('config_sha256'))[:16]}")
                    det["c_star"] = {"id": cs.get("id"), "lr": cs["lr"], "anchor": cs["anchor"], "config_sha256": rebuilt}
                if doc.get("code_sha256") != mod_sha:
                    reasons.append(f"r1_result.json was written by srttc_r1.py {str(doc.get('code_sha256'))[:12]}, the code being gated is {mod_sha[:12]}")
                if strict and (doc.get("inputs") or {}).get("pins_overridden"):      # strict only: a synthetic world necessarily overrides pins (the frozen gate's pin check is strict-only too)
                    reasons.append(f"the R1 fit overrode registered pins {doc['inputs']['pins_overridden']}")
                rep = (doc.get("inputs") or {}).get("replacements") or {}
                used = bool(opts.replacements_file)
                if used != bool(rep.get("dropped") or rep.get("added")):
                    reasons.append(f"replacements: R1 recorded {rep or 'none'}, this argv {'passes' if used else 'does not pass'} --replacements-file: the selected logs would differ")
            # --- C2 is written ONCE
            out = opts.resolved_out_dir()
            for fn in [REG_C2_RESULT, ATTEST_NAME] + [REG_C2_ROW_FMT.format(sd=sd) for sd in opts.seeds]:
                if os.path.exists(os.path.join(out, fn)):
                    reasons.append(f"{os.path.join(out, fn)} already exists: the C2 rows are written once")
    except Exception as e:                                       # noqa: BLE001
        reasons.append(f"{type(e).__name__}: {str(e)[:300]}")
    return _res("FAIL" if reasons else "PASS", reasons, det), opts


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# G-PATH / G-BANK: the frozen, tested checks, plus what is specific to C2
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def check_path(mod, opts, argv: list) -> dict:
    res = G.check_path(mod, opts, translate_argv(argv))
    reasons, det = list(res["reasons"]), dict(res["details"])
    try:
        for p in (argv[0], argv[2]):
            hit = mod.PathGuard.forbidden(p)
            if hit:
                reasons.append(f"the C2 launch names a dev-gate / navtest token {hit}: {p!r}")
    except Exception as e:                                       # noqa: BLE001
        reasons.append(f"{type(e).__name__}: {str(e)[:300]}")
    return _res("FAIL" if reasons else "PASS", reasons, det)


def check_bank(mod, opts) -> tuple:
    res, aux = G.check_bank(mod, opts)
    reasons, det = list(res["reasons"]), dict(res["details"])
    try:
        doc, sha, _p = load_r1_result(mod, opts)
        man = aux.get("manifest_sha256")
        det["r1_result_bank_manifest_sha256"] = (doc.get("inputs") or {}).get("bank_manifest_sha256")
        if man and (doc.get("inputs") or {}).get("bank_manifest_sha256") != man:
            reasons.append(f"r1_result.json was fitted on bank manifest {str(det['r1_result_bank_manifest_sha256'])[:16]}, the manifest now is {man[:16]}: C2 must use the SAME bank")
    except Exception as e:                                       # noqa: BLE001
        reasons.append(f"r1_result.json cannot be compared with the bank: {type(e).__name__}: {str(e)[:200]}")
    return _res("FAIL" if reasons else "PASS", reasons, det), aux


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# G-LIVE
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def _live_unit(mod, opts, aux: dict, tmp: str):
    """The first fit-set unit in key order, copied, hashed against the manifest, loaded through R1's own loader.  -> (data, W6, b6, unit, sha) ; raises GateError."""
    guard = opts.guard()
    selected, _sha, _rep = G.selected_rows(mod, guard, opts)
    rank_of = {r["log"]: r["rank"] for r in selected}
    unit_logs = aux.get("unit_logs") or G.unit_log_index(mod, guard, opts.bank_dir, [n for n in mod._io_listdir(guard, opts.bank_dir) if n.endswith(".sets.npz")])
    best = None
    for n in sorted(unit_logs):
        rk = [rank_of[lg] for lg in unit_logs[n] if lg in rank_of]
        if rk and (best is None or min(rk) < best[0]):
            best = (min(rk), n)
    if best is None:
        raise GateError("no bank unit holds a selected fit log")
    unit = best[1]
    raw = mod._io_read_bytes(guard, opts.manifest)
    table, _d = G._unit_table(json.loads(raw.decode("utf-8")))
    if unit not in table:
        raise GateError(f"unit {unit} is not in the manifest")
    shutil.copyfile(os.path.join(opts.bank_dir, unit), os.path.join(tmp, unit))
    sha = sha256_file(os.path.join(tmp, unit))
    if sha != table[unit]:
        raise GateError(f"unit {unit}: sha256 {sha[:16]} != manifest {table[unit][:16]}")
    with open(os.path.join(tmp, "MANIFEST.sha256.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"files": {unit: {"path": unit, "sha256": sha}}}, f)
    opts_live = dataclasses.replace(opts, bank_dir=tmp, manifest=os.path.join(tmp, "MANIFEST.sha256.json"))
    glive = opts_live.guard()
    sel_unit = [r for r in selected if r["log"] in set(unit_logs[unit])]
    W6, b6 = mod.load_p2_row(glive, opts.p2_ckpt)
    data = mod.load_fit_data(glive, opts_live, sel_unit, {unit: sha}, W6, b6)
    return data, W6, b6, unit, sha


def check_live(mod, c2mod, opts, aux: dict) -> dict:
    import numpy as np
    import torch
    base = G.check_live(mod, opts, aux)                          # (1) R1's fit path reaches the TTC row, exactly 0.0 elsewhere, controls
    reasons, det = list(base["reasons"]), dict(base["details"])
    tmp = None
    saved_loss = getattr(mod, "total_loss", None)
    try:
        torch.set_num_threads(max(1, min(int(opts.threads), 4)))
        tmp = tempfile.mkdtemp(prefix="srttc_c2live_")
        data, W6, b6, unit, _sha = _live_unit(mod, opts, aux, tmp)
        y_real = data.y3.clone()
        top8_real, feat_real = data.top8.clone(), data.feat.clone()
        seed = int(getattr(c2mod, "C2_SHUFFLE_SEED", REG_C2_SHUFFLE_SEED))
        # --- (2a) the shuffle itself
        ys = c2mod.shuffle_labels_within_set(data.y3, seed)
        ys2 = c2mod.shuffle_labels_within_set(data.y3, seed)
        yo = c2mod.shuffle_labels_within_set(data.y3, seed + 1)
        moved = float((ys.numpy() != y_real.numpy()).mean()) if tuple(ys.shape) == tuple(y_real.shape) else float("nan")
        if tuple(ys.shape) != tuple(y_real.shape):
            reasons.append(f"the shuffle changed the label tensor shape {tuple(y_real.shape)} -> {tuple(ys.shape)}")
        if not (np.isfinite(moved) and moved > 0.0):
            reasons.append(f"the shuffle moved {moved!r} of the labels: C2 would test nothing")
        if tuple(ys.shape) == tuple(y_real.shape):
            if not np.array_equal(np.sort(ys.numpy(), axis=1), np.sort(y_real.numpy(), axis=1)):
                reasons.append("the shuffle did not keep each set's label multiset (it mixed labels ACROSS sets, or invented values)")
            if not torch.equal(ys, ys2):
                reasons.append("the shuffle is not deterministic in its seed")
            if torch.equal(ys, yo):
                reasons.append("the shuffle does not depend on its seed (seed and seed + 1 give the same labels)")
        if not torch.equal(data.y3, y_real):
            reasons.append("shuffle_labels_within_set modified the REAL label tensor in place (R1's fit data would be corrupted)")
        # --- (2b) THROUGH srttc_c2.fit_c2_rows -> srttc_r1.fit_row: the loss receives the SHUFFLED labels
        batch = mod._probe_batch(data, opts.reach_n_sets)
        nb = int(len(batch["feat"]))
        mini = mod.FitData(log_names=["probe"], log_rank=np.zeros(1, np.int64), log_fold=np.zeros(1, np.int64), log_idx=np.zeros(nb, np.int64), fold=np.zeros(nb, np.int64),
                           feat=batch["feat"], z6=batch["z6"], y3=batch["y3"], top8=batch["top8"], provenance={})
        y_mini_real, feat_mini, top8_mini = mini.y3.clone(), mini.feat.clone(), mini.top8.clone()
        y_mini_shuf = c2mod.shuffle_labels_within_set(mini.y3, seed)
        seen = []
        real_total = saved_loss

        def recorder(z6, y3, idx8, aw, pw=1.0):
            seen.append(y3.detach().clone())
            return real_total(z6, y3, idx8, aw, pw)

        mod.total_loss = recorder
        try:
            star = mod.Cfg(lr=mod.GRID_LR[-1], anchor=mod.GRID_ANCHOR[1], epochs=1, batch_sets=nb)
            rows, y_used = c2mod.fit_c2_rows(mini, star, (0,), W6, b6, seed)
        finally:
            mod.total_loss = saved_loss
        n_calls = len(seen)
        srt = lambda a: a[np.lexsort(a.T[::-1])]   # noqa: E731
        got_shuffled = got_real = False
        if n_calls != 1:
            reasons.append(f"srttc_c2.fit_c2_rows -> fit_row called the loss {n_calls} times for one epoch of one batch (expected exactly 1): the C2 fit does not run R1's loss path")
        else:
            got = seen[0].numpy()
            got_shuffled = bool(np.array_equal(srt(got), srt(y_mini_shuf.numpy())))
            got_real = bool(np.array_equal(srt(got), srt(y_mini_real.numpy())))
            if not got_shuffled:
                reasons.append("the loss did not receive the SHUFFLED labels (the labels the optimiser trained on are not the shuffle of the real ones)")
            if got_real:
                reasons.append("the loss received the REAL labels in the C2 fit: C2 would reproduce the real row and test nothing")
        w, b = rows["0"]
        w_moved = float(np.abs(np.asarray(w) - np.asarray(W6)[mod.TTC_COL]).max())
        if not (np.isfinite(w_moved) and w_moved > 0.0):
            reasons.append(f"the C2 fit moved the TTC row by {w_moved!r}: the shuffled-label loss is not wired to the optimiser")
        if not torch.equal(mini.y3, y_mini_real):
            reasons.append("fit_c2_rows modified the REAL label tensor of the data it was given")
        if not (torch.equal(mini.top8, top8_mini) and torch.equal(mini.feat, feat_mini)):
            reasons.append("fit_c2_rows changed the frozen top-8 membership or the features")
        if not torch.equal(y_used, y_mini_shuf):
            reasons.append("the labels fit_c2_rows reports having used are not the deterministic shuffle")
        # --- (2c) gradient controls on the same batch: shuffled != real (must read non-zero), same labels twice == identical (must read EXACTLY 0.0)
        def grad_of(y):
            head = mod.RowHead(W6, b6, probe=True, perturb=True)
            z6 = mod.forward_z6(head, batch["feat"].clone().float(), batch["z6"])
            loss, _p, _a = saved_loss(z6, y, batch["top8"], mod.GRID_ANCHOR[1], 1.0)
            loss.backward()
            return torch.cat([head.row_w.grad.flatten(), head.row_b.grad.flatten()]).clone()

        g_real, g_real2, g_shuf = grad_of(y_mini_real), grad_of(y_mini_real), grad_of(y_mini_shuf)
        d_pos = float((g_shuf - g_real).abs().sum())
        d_zero = float((g_real2 - g_real).abs().sum())
        if not (np.isfinite(d_pos) and d_pos > 0.0):
            reasons.append(f"the TTC-row gradient on the shuffled labels equals the gradient on the real labels (|diff| = {d_pos!r}): the labels do not reach the loss")
        if d_zero != 0.0:
            reasons.append(f"control: two evaluations on the SAME labels differ by {d_zero!r}; must be EXACTLY 0.0 (the probe is not deterministic)")
        det.update({"c2_unit": unit, "c2_fraction_of_labels_moved": moved, "c2_shuffle_keeps_label_multiset": not any("multiset" in r for r in reasons), "c2_loss_received_shuffled_labels": bool(got_shuffled and not got_real),
                    "c2_loss_calls": n_calls, "c2_fit_moved_ttc_row": bool(w_moved > 0.0), "c2_grad_shuffled_vs_real_l1": d_pos, "c2_grad_same_labels_l1_must_be_0": d_zero,
                    "c2_batch_sets": nb, "concordance_computed": False})
    except Exception as e:                                       # noqa: BLE001
        reasons.append(f"{type(e).__name__}: {str(e)[:300]}")
    finally:
        if saved_loss is not None:
            mod.total_loss = saved_loss
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    return _res("FAIL" if reasons else "PASS", reasons, det)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# G-SUITE
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def check_suite(*, commit: str, git_dir: str, repo_code_path: str, python: str, specs, timeout_s: int = 1800) -> dict:
    """Each (test file, selectors, minimum) in full form on a clean export at `commit`: `python -I <file> -v <selectors>` with cwd = the export.  CPU only, 2 threads, ONE process at a time."""
    reasons, det = [], {"runs": []}
    tmp = tempfile.mkdtemp(prefix="srttc_c2suite_")
    try:
        files = G.export_code_dir(git_dir, commit, repo_code_path, tmp)
        py = python if python and os.path.exists(python) else sys.executable
        env = G._clean_env()
        env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
        for test_file, selectors, min_tests in specs:
            if test_file not in files:
                reasons.append(f"{test_file} is not in the export of {commit[:12]}")
                continue
            blob = G.git_blob(git_dir, commit, f"{repo_code_path}/{test_file}")
            if blob is None or sha256_file(os.path.join(tmp, test_file)) != sha256_bytes(blob):
                reasons.append(f"the exported {test_file} differs from its blob")
                continue
            t0 = time.time()
            try:
                cp = subprocess.run([py, "-I", test_file, "-v", *selectors], cwd=tmp, env=env, capture_output=True, timeout=timeout_s)
            except subprocess.TimeoutExpired:
                reasons.append(f"{test_file} did not finish in {timeout_s} s")
                continue
            out = (cp.stdout + b"\n" + cp.stderr).decode("utf-8", "replace")
            m = re.findall(r"^Ran (\d+) tests? in ([\d.]+)s", out, re.M)
            n = int(m[-1][0]) if m else 0
            lines = [ln for ln in out.strip().splitlines() if ln.strip()]
            status = lines[-1].strip() if lines else ""
            det["runs"].append({"test_file": test_file, "selectors": list(selectors), "returncode": cp.returncode, "n_tests": n, "status_line": status, "seconds": round(time.time() - t0, 1),
                                "test_file_sha256": sha256_bytes(blob), "interpreter": py})
            if cp.returncode != 0:
                reasons.append(f"{test_file} {list(selectors)} exited {cp.returncode}: {status[:80]}; tail: {' | '.join(ln.strip()[:100] for ln in lines[-6:])}")
            if status != "OK":
                reasons.append(f"{test_file}: the final line is {status[:80]!r}, not 'OK' (failures, errors, skips and expected-failures are all refused)")
            if n < min_tests:
                reasons.append(f"{test_file} {list(selectors)}: only {n} tests ran, the floor is {min_tests} (tests deleted or not collected)")
        if not specs:
            reasons.append("no suite was specified")
    except Exception as e:                                       # noqa: BLE001
        reasons.append(f"{type(e).__name__}: {str(e)[:300]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return _res("FAIL" if reasons else "PASS", reasons, det)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# orchestration, mint
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def _not_run(why: str) -> dict:
    return _res("NOT-RUN", [why])


def run_checks(cfg: C2Config, *, overrides: dict | None = None, log=print) -> dict:
    """Run every check; -> {checks, mod_sha, opts, aux, argv, c2_sha}.  `overrides` (Options fields) is the API-level test hook: the CLI cannot reach it."""
    argv = cfg.argv()
    checks: dict = {}
    mod = c2mod = opts = None
    mod_sha = c2_sha = None
    aux: dict = {}
    suite_files = tuple(s[0] for s in cfg.suite_specs)
    checks["G-COMMIT"] = check_commit(commit=cfg.commit, git_dir=cfg.git_dir, repo_code_path=cfg.repo_code_path, code_dir=cfg.code_dir, suite_files=suite_files, frozen_gate_sha=FROZEN_GATE_SHA)
    log(f"  G-COMMIT {checks['G-COMMIT']['status']}")
    if checks["G-COMMIT"]["status"] == "PASS":
        try:
            files = checks["G-COMMIT"]["details"]["files"]
            mod, c2mod, shas = load_verified_pair(cfg.code_dir, files)
            mod_sha, c2_sha = shas["srttc_r1.py"], shas["srttc_c2.py"]
        except Exception as e:                                   # noqa: BLE001
            checks["G-COMMIT"] = _res("FAIL", [f"cannot load srttc_r1.py / srttc_c2.py from the verified bytes: {type(e).__name__}: {str(e)[:200]}"])
            mod = None
    if mod is None:
        for c in ("G-DVB", "G-PATH", "G-BANK", "G-LIVE"):
            checks[c] = _not_run("srttc_c2.py / srttc_r1.py are not verified against their blobs (G-COMMIT)")
    else:
        checks["G-DVB"], opts = check_dvb(mod, c2mod, argv, code_dir=cfg.code_dir, r1_sha256=cfg.r1_result_sha256, mod_sha=mod_sha, strict=overrides is None)
        log(f"  G-DVB    {checks['G-DVB']['status']}")
        if opts is None:
            for c in ("G-PATH", "G-BANK", "G-LIVE"):
                checks[c] = _not_run("the argv does not parse (G-DVB)")
        else:
            if overrides:
                opts = dataclasses.replace(opts, **overrides)
            checks["G-PATH"] = check_path(mod, opts, argv)
            log(f"  G-PATH   {checks['G-PATH']['status']}")
            checks["G-BANK"], aux = check_bank(mod, opts)
            log(f"  G-BANK   {checks['G-BANK']['status']}")
            if checks["G-BANK"]["status"] == "PASS":
                checks["G-LIVE"] = check_live(mod, c2mod, opts, aux)
            else:
                checks["G-LIVE"] = _not_run("G-BANK is not PASS: no unit is verified to read")
            log(f"  G-LIVE   {checks['G-LIVE']['status']}")
    if checks["G-COMMIT"]["status"] == "PASS":
        checks["G-SUITE"] = check_suite(commit=cfg.commit, git_dir=cfg.git_dir, repo_code_path=cfg.repo_code_path, python=cfg.python, specs=cfg.suite_specs, timeout_s=cfg.suite_timeout_s)
    else:
        checks["G-SUITE"] = _not_run("the commit is not verified (G-COMMIT)")
    log(f"  G-SUITE  {checks['G-SUITE']['status']}")
    return {"checks": checks, "mod_sha": mod_sha, "c2_sha": c2_sha, "opts": opts, "aux": aux, "argv": argv}


def mint(cfg: C2Config, *, overrides: dict | None = None, log=print) -> dict:
    """Run every check and write `c2_gate_token.json` ONLY if all PASS (else a signed `c2_gate_FAIL.json` and no token).  -> {verdict, token_path, token, checks, reasons}."""
    log(f"SR-TTC C2 launch gate: stage {STAGE}, fraction {cfg.fraction}, commit {cfg.commit[:12]}")
    key = G.load_key(cfg.key_file, create=True, forbid_under=[cfg.code_dir])
    run = run_checks(cfg, overrides=overrides, log=log)
    checks, argv = run["checks"], run["argv"]
    reasons = [f"{c}: {r}" for c in CHECKS for r in checks[c]["reasons"]] + [f"{c}: {checks[c]['status']}" for c in CHECKS if checks[c]["status"] not in ("PASS", "FAIL")]
    verdict = "PASS" if all(checks[c]["status"] == "PASS" for c in CHECKS) else "FAIL"
    tok_path = Path(cfg.resolved_token_path())
    stamp = G.utc_now().replace(":", "")
    profile = PROFILE_SYNTHETIC if overrides else PROFILE
    if verdict == "PASS":
        opts = run["opts"]
        r1_path = os.path.join(opts.resolved_out_dir(), "r1_result.json")
        paths = {"bank_manifest": opts.manifest, "ranked_fit_logs": opts.ranked_logs, "p2_ckpt": opts.p2_ckpt, "s0_marker": opts.s0_marker, "r1_result": r1_path}
        if opts.replacements_file:
            paths["replacements_file"] = opts.replacements_file
        data = {k: {"path": str(v), **G.fingerprint_file(v)} for k, v in sorted(paths.items())}
        det = checks["G-COMMIT"]["details"]
        code_files = {f: sha256_file(os.path.join(cfg.code_dir, f)) for f in BOUND_FILES}
        binding = {"profile": profile, "commit": cfg.commit, "tree_sha256": G.tree_digest(code_files), "argv_sha256": argv_sha256(argv), "argv": argv, "stage": STAGE,
                   "fraction": cfg.fraction, "trainer": "srttc_c2_run.py -> srttc_c2.run_c2", "srttc_c2_sha256": code_files["srttc_c2.py"], "srttc_r1_sha256": code_files["srttc_r1.py"],
                   "bank_manifest_sha256": data["bank_manifest"]["sha256"], "r1_result_sha256": data["r1_result"]["sha256"], "c_star": checks["G-DVB"]["details"].get("c_star"),
                   "shuffle_seed": checks["G-DVB"]["details"].get("c2_shuffle_seed"), "code_files": code_files, "landed": det.get("landed"), "repo_code_path": cfg.repo_code_path,
                   "suite_test_blobs": det.get("test_blobs", {}), "data": data, "data_sha256": sha256_bytes(canonical_json(data))}
    else:
        binding = {"profile": profile, "commit": cfg.commit, "argv_sha256": argv_sha256(argv), "argv": argv, "stage": STAGE, "fraction": cfg.fraction}
    token = {"schema": GATE_SCHEMA, "kind": TOKEN_KIND, "verdict": verdict, "binding": binding, "checks": checks, "reasons": reasons,
             "gate_sha256": sha256_file(os.path.join(HERE, "srttc_c2_gate.py")), "created_utc": G.utc_now(), "key_id": G.key_id(key)}
    token["hmac_sha256"] = G.sign(token, key)
    fail_path = tok_path.parent / FAIL_NAME
    if verdict == "PASS":
        G._demote(tok_path, stamp)
        G._demote(fail_path, stamp)
        G.write_json(tok_path, token)
        log(f"PASS: wrote {tok_path}")
        return {"verdict": verdict, "token_path": str(tok_path), "token": token, "checks": checks, "reasons": []}
    G._demote(tok_path, stamp)                                   # a stale PASS must not survive a failing run
    G.write_json(fail_path, token)
    log(f"FAIL: no token minted; report {fail_path}")
    for r in reasons:
        log(f"  - {r[:400]}")
    return {"verdict": verdict, "token_path": None, "token": token, "checks": checks, "reasons": reasons}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# verify (the enforcing entry's side): every field the token binds is RE-MEASURED now
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def _argv_equal(token_argv, argv) -> bool:
    """Exact list equality, except the interpreter (element 0: same program name) and the entry script (element 2: same normalised path), which the shell may spell differently."""
    if not isinstance(token_argv, list) or len(token_argv) != len(argv):
        return False
    for i, (a, b) in enumerate(zip(token_argv, argv)):
        if i == 0:                                                # the interpreter: the venv shim and the real interpreter may spell the path differently; the program must be the same
            if os.path.basename(str(a).replace("\\", "/")).lower() != os.path.basename(str(b).replace("\\", "/")).lower():
                return False
        elif i == 2:
            if os.path.normcase(os.path.normpath(str(a))) != os.path.normcase(os.path.normpath(str(b))):
                return False
        elif a != b:
            return False
    return True


def verify_token(token_path, argv: list, *, code_dir: str, key: bytes, git_dir: str | None = DEFAULT_GIT_DIR, profile: str = PROFILE) -> tuple:
    """-> (ok, reasons, info).  Any difference refuses."""
    reasons, info = [], {"token": str(token_path)}
    try:
        tok = G.read_json(token_path)
    except Exception as e:                                       # noqa: BLE001
        return False, [f"token unreadable: {type(e).__name__}: {str(e)[:160]}"], info
    if tok.get("kind") != TOKEN_KIND or tok.get("schema") != GATE_SCHEMA:
        reasons.append("not a launch-gate token")
    if not G.signature_ok(tok, key):
        reasons.append(f"HMAC does not verify: edited, or signed with another key (token key_id {tok.get('key_id')}, this host's {G.key_id(key)})")
    if tok.get("verdict") != "PASS":
        reasons.append(f"token verdict is {tok.get('verdict')!r}, not PASS")
    b = tok.get("binding") or {}
    info["binding"] = {k: b.get(k) for k in ("profile", "commit", "argv_sha256", "srttc_c2_sha256", "srttc_r1_sha256", "bank_manifest_sha256", "r1_result_sha256", "stage", "fraction")}
    if b.get("profile") != profile:
        reasons.append(f"token profile {b.get('profile')!r} != the launch profile {profile!r}")
    if b.get("stage") != STAGE:
        reasons.append(f"token stage {b.get('stage')!r} != {STAGE!r} (an R1 run / lc token cannot open C2)")
    info["argv_sha256"] = argv_sha256(argv)
    if not _argv_equal(b.get("argv"), argv):
        changed = [f"{x} -> {y}" for x, y in zip(b.get("argv") or [], argv) if x != y][:5]
        reasons.append(f"argv differs from the gated argv (sha {info['argv_sha256'][:12]} vs {str(b.get('argv_sha256'))[:12]}; {changed or 'length differs'})")
    code = b.get("code_files") or {}
    now, n_before = {}, len(reasons)
    for f in BOUND_FILES:
        p = os.path.join(code_dir, f)
        now[f] = sha256_file(p) if os.path.isfile(p) else None
        if now[f] is None or now[f] != code.get(f):
            reasons.append(f"{f} in {code_dir} is {str(now[f])[:12]}, the gated file is {str(code.get(f))[:12]}")
    if len(reasons) == n_before and G.tree_digest(now) != b.get("tree_sha256"):
        reasons.append("code tree digest differs from the gated tree")
    for key_, f in (("srttc_c2_sha256", "srttc_c2.py"), ("srttc_r1_sha256", "srttc_r1.py")):
        if b.get(key_) != now.get(f):
            reasons.append(f"{f} sha256 {str(now.get(f))[:12]} != the token's {str(b.get(key_))[:12]}")
    data = b.get("data") or {}
    if sha256_bytes(canonical_json(data)) != b.get("data_sha256"):
        reasons.append("data_sha256 does not match the bound data block")
    d_bad = [k for k, want in sorted(data.items()) if G.fp_key(G.fingerprint_file(want.get("path", ""))) != G.fp_key(want)]
    if d_bad:
        reasons.append(f"inputs differ from the gated inputs on {d_bad} (bank manifest / ranked logs / P2 checkpoint / S0 marker / r1_result.json)")
    if (data.get("bank_manifest") or {}).get("sha256") != b.get("bank_manifest_sha256"):
        reasons.append("bank_manifest_sha256 is not the bound manifest's sha256")
    if (data.get("r1_result") or {}).get("sha256") != b.get("r1_result_sha256"):
        reasons.append("r1_result_sha256 is not the bound r1_result.json's sha256")
    commit = b.get("commit")
    if git_dir is not None:
        try:
            if not G.git_is_commit(git_dir, str(commit)):
                reasons.append(f"commit {str(commit)[:12]} is not a commit object in {git_dir}")
            else:
                landed = b.get("landed") or {}
                for f in BOUND_FILES:
                    if f in LANDED_FILES or landed.get(f):
                        blob = G.git_blob(git_dir, commit, f"{b.get('repo_code_path')}/{f}")
                        if blob is None or sha256_bytes(blob) != now.get(f):
                            reasons.append(f"{f} no longer equals its blob at {str(commit)[:12]}")
        except Exception as e:                                   # noqa: BLE001
            reasons.append(f"cannot re-check the commit blobs: {type(e).__name__}: {str(e)[:160]}")
    info["created_utc"] = tok.get("created_utc")
    info["token_sha256"] = sha256_file(token_path)
    info["token_hmac_sha256"] = tok.get("hmac_sha256")
    info["key_id"] = tok.get("key_id")
    info["commit"] = commit
    info["code_files"] = now
    info["unlanded_files"] = sorted(f for f in NEW_FILES if not (b.get("landed") or {}).get(f))
    return not reasons, reasons, info


def write_attestation(path: str, *, info: dict, argv: list, out_dir: str, result: dict, mod=None, guard=None) -> str:
    """The sidecar `r1_c2_attestation.json` (exclusive create): which token (sha256 + HMAC) licensed THIS C2 result and which rows it wrote.  srttc_c2.py's own result file cannot carry it (landed)."""
    res_path = os.path.join(out_dir, REG_C2_RESULT)
    rows = {sd: {"file": ent["file"], "sha256": ent["sha256"]} for sd, ent in (result.get("final_rows") or {}).items()}
    doc = {"schema": "srttc-c2-attestation-v1", "stage": STAGE, "token_sha256": info.get("token_sha256"), "token_hmac_sha256": info.get("token_hmac_sha256"), "key_id": info.get("key_id"),
           "commit": info.get("commit"), "unlanded_files": info.get("unlanded_files"), "argv_sha256": argv_sha256(argv), "r1_result_sha256": (info.get("binding") or {}).get("r1_result_sha256"),
           "bank_manifest_sha256": (info.get("binding") or {}).get("bank_manifest_sha256"), "c2_result_file": REG_C2_RESULT, "c2_result_sha256": sha256_file(res_path), "rows": rows,
           "written": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    blob = json.dumps(doc, indent=1, sort_keys=True).encode("utf-8")
    if mod is not None and guard is not None:
        mod._io_write_bytes(guard, path, blob, exclusive=True)
    else:
        with open(path, "xb") as f:
            f.write(blob)
    return sha256_bytes(blob)


def verify_attestation(run_dir: str, *, key: bytes, token_path: str | None = None, profile: str = PROFILE) -> tuple:
    """Post-hoc, before the dev read: the C2 result on disk is the one a PASS token licensed.  -> (ok, reasons, doc)."""
    reasons = []
    try:
        att = G.read_json(os.path.join(run_dir, ATTEST_NAME))
        res_path = os.path.join(run_dir, REG_C2_RESULT)
        if att.get("c2_result_sha256") != sha256_file(res_path):
            reasons.append("r1_c2_result.json differs from the attested one")
        c2 = G.read_json(res_path)
        for sd, ent in (att.get("rows") or {}).items():
            if sha256_file(os.path.join(run_dir, ent["file"])) != ent["sha256"]:
                reasons.append(f"seed {sd}: the C2 row file differs from the attested one")
            if (c2.get("final_rows") or {}).get(sd, {}).get("sha256") != ent["sha256"]:
                reasons.append(f"seed {sd}: the result file does not list the attested row")
        tp = token_path or default_token_path(int(c2.get("fraction_pct", 0)))
        tok = G.read_json(tp)
        if not G.signature_ok(tok, key) or tok.get("verdict") != "PASS" or (tok.get("binding") or {}).get("profile") != profile:
            reasons.append("the attested token does not verify (HMAC / verdict / profile)")
        if sha256_file(tp) != att.get("token_sha256"):
            reasons.append("the token on disk is not the one the attestation names")
        if (tok.get("binding") or {}).get("r1_result_sha256") != att.get("r1_result_sha256"):
            reasons.append("token and attestation bind different r1_result.json")
        if sha256_file(os.path.join(run_dir, "r1_result.json")) != att.get("r1_result_sha256"):
            reasons.append("r1_result.json changed since the attestation")
        return not reasons, reasons, att
    except Exception as e:                                       # noqa: BLE001
        return False, [f"{type(e).__name__}: {str(e)[:200]}"], {}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the enforcing entry: srttc_c2_run.py calls this.  It REFUSES, before it imports or fits anything, unless a PASS token matches the exact command it is running.
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def run_main(argv: list, entry_file: str, *, _token_path: str | None = None, _key_file: str | None = None, _git_dir: str | None = None, _profile: str | None = None,
             _opts_overrides: dict | None = None, _out=print) -> int:
    """The SR-TTC C2 fit, behind the launch gate.  0 = C2 rows written + attested;  2 = usage / `--no-gate-stage-ack`;  3 = refused by the gate;  1 = the fit refused to produce rows.
    The `_*` keyword arguments are API-only test hooks: the command line cannot reach them (a hook-minted token has the synthetic profile and never verifies for the real one)."""
    argv = list(argv)
    if any(str(t).split("=")[0] == "--no-gate-stage-ack" for t in argv):
        _out("REFUSED: --no-gate-stage-ack is NOT accepted (Master Mind R5, MM_RULINGS_HEADROWS.md). A shuffled-label C2 fit is a fit and needs a launch-gate PASS token (srttc_c2_gate.py).")
        return 2
    pairs, problems = G._cli_flags(argv[1:])
    fr = dict(pairs).get("--fraction")
    if not argv or argv[0] != "run" or problems or fr is None or not str(fr).isdigit() or int(fr) not in G.REG_FRACTIONS:
        _out(f"REFUSED: usage: {RUN_ENTRY} run --fraction {{10,25,50,100}} --out-dir D ... (the exact `srttc_c2_gate.py argv` output); got {argv[:3]} {problems[:1]}")
        return 2
    code_dir = os.path.dirname(os.path.abspath(entry_file))
    self_argv = [sys.executable, "-I", os.path.abspath(entry_file)] + argv
    token_path = _token_path or default_token_path(int(fr))
    try:
        key = G.load_key(_key_file, create=False)
        ok, reasons, info = verify_token(token_path, self_argv, code_dir=code_dir, key=key, git_dir=_git_dir or DEFAULT_GIT_DIR, profile=_profile or PROFILE)
    except Exception as e:                                       # noqa: BLE001 -- a refusal, never a pass
        ok, reasons, info = False, [f"{type(e).__name__}: {str(e)[:300]}"], {}
    if not ok:
        _out("REFUSED by the launch gate (no C2 fit was started): " + " | ".join(reasons))
        return 3
    mod, c2mod, _shas = load_verified_pair(code_dir, info["code_files"])
    opts = mod.options_from_args(mod.build_parser().parse_args(argv))
    if _opts_overrides:
        opts = dataclasses.replace(opts, **_opts_overrides)
    try:
        result = c2mod.run_c2(opts)
    except c2mod.C2Error as e:
        _out(f"C2 REFUSED TO PRODUCE ROWS (nothing written): {e}")
        return 1
    out_dir = opts.resolved_out_dir()
    att = write_attestation(os.path.join(out_dir, ATTEST_NAME), info=info, argv=self_argv, out_dir=out_dir, result=result, mod=mod, guard=opts.guard())
    _out(f"C2 done: {os.path.join(out_dir, REG_C2_RESULT)} sha256 {sha256_file(os.path.join(out_dir, REG_C2_RESULT))}; attestation {att}")
    return 0


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="srttc_c2_gate.py", description="SR-TTC C2 launch gate (mint / verify / argv). CPU only.")
    ap.add_argument("cmd", nargs="?", choices=("mint", "verify", "argv"), default="mint")
    ap.add_argument("--fraction", type=int, required=True, choices=G.REG_FRACTIONS)
    ap.add_argument("--commit", default="", help="the landed commit (40-hex) the deployed code files must equal; required for mint")
    ap.add_argument("--r1-result-sha256", default="", help="the REGISTERED sha256 of r1_result.json (required for mint)")
    ap.add_argument("--python", default=DEFAULT_PYTHON)
    ap.add_argument("--code-dir", default=DEFAULT_CODE_DIR)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=G.REG_EPOCHS)
    ap.add_argument("--batch-sets", type=int, default=G.REG_BATCH_SETS)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--criterion", default=G.REG_CRITERION)
    ap.add_argument("--both-rule", default=G.REG_BOTH_RULE)
    ap.add_argument("--replacements-file", default=None)
    ap.add_argument("--git-dir", default=DEFAULT_GIT_DIR)
    ap.add_argument("--repo-code-path", default=DEFAULT_REPO_CODE_PATH)
    ap.add_argument("--token", default="", help=f"default: {DEFAULT_GATE_ROOT}\\f<fraction>_c2\\{TOKEN_NAME}")
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--suite-timeout-s", type=int, default=1800)
    return ap


def config_from_args(a: argparse.Namespace) -> C2Config:
    return C2Config(commit=a.commit, fraction=a.fraction, python=a.python, code_dir=a.code_dir, out_dir=a.out_dir, threads=a.threads, epochs=a.epochs, batch_sets=a.batch_sets,
                    seeds=a.seeds, criterion=a.criterion, both_rule=a.both_rule, replacements_file=a.replacements_file, r1_result_sha256=a.r1_result_sha256, git_dir=a.git_dir,
                    repo_code_path=a.repo_code_path, token_path=a.token, key_file=a.key_file, suite_timeout_s=a.suite_timeout_s)


def main(argv=None, *, _test_overrides: dict | None = None, _test_cfg: dict | None = None, _out=print) -> int:
    """`_test_overrides` / `_test_cfg` are API-only hooks (the command line cannot reach them): Options fields for a synthetic world, and C2Config fields for a fake export."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("mint", "verify", "argv", "-h", "--help"):
        argv = ["mint"] + argv
    a = make_parser().parse_args(argv)
    cfg = config_from_args(a)
    for k, v in (_test_cfg or {}).items():
        setattr(cfg, k, v)
    if a.cmd == "argv":
        _out(json.dumps({"argv": cfg.argv()}))
        return 0
    if a.cmd == "verify":
        try:
            key = G.load_key(cfg.key_file, create=False)
            ok, reasons, info = verify_token(cfg.resolved_token_path(), cfg.argv(), code_dir=cfg.code_dir, key=key, git_dir=cfg.git_dir,
                                             profile=PROFILE_SYNTHETIC if _test_overrides else PROFILE)
        except Exception as e:                                   # noqa: BLE001 -- a refusal, never a pass
            ok, reasons, info = False, [f"{type(e).__name__}: {str(e)[:300]}"], {}
        _out(json.dumps({"ok": bool(ok), "reasons": reasons, "argv": cfg.argv(), "token": cfg.resolved_token_path(), "info": info}))
        return 0 if ok else 3
    if not re.fullmatch(r"[0-9a-f]{40}", cfg.commit or ""):
        _out("REFUSED: --commit must be the 40-hex landed commit")
        return 2
    if not re.fullmatch(r"[0-9a-f]{64}", cfg.r1_result_sha256 or ""):
        _out("REFUSED: --r1-result-sha256 must be the registered 64-hex sha256 of r1_result.json")
        return 2
    res = mint(cfg, overrides=_test_overrides, log=_out)
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
