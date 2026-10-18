#!/usr/bin/env python3
"""SR-TTC R1 launch gate: mints `r1_gate_token.json` ONLY when every check passes; `verify` is what `launch_r1.ps1` runs before it starts R1.

WHY (MM_DECISIONS_R1.md row 11; CLAUDE.md "NO TRAINING LAUNCH WITHOUT A LAUNCH-GATE PASS", PI 2026-09-26): R1 is a training run (it fits 257 parameters on cached features); the rule has no size
exemption.  The token is bound to {landed commit, sha256 of the exact R1 argv, sha256 of the S0 bank manifest, sha256 of srttc_r1.py, plus the other inputs R1 reads}.  Nothing is hand-written: the
token is HMAC-signed (same schema `tanitad.launch_gate/1`, same canonical JSON, same key file as the programme's gate and `refe/gate_token.py`; known-answer vectors in test_srttc_r1_gate.py pin the equality).

CHECKS (each has a deliberate-regression arm in test_srttc_r1_gate.py that must make it FAIL)
  G-COMMIT  precondition: the deployed code files equal their git blobs at the commit (a file that is not landed, or edited since, REFUSES).  Nothing is imported or run from unverified bytes.
  G-DVB     declared-vs-built: every argv lever equals MM_DECISIONS_R1 rows 1-10 AND the object srttc_r1.py BUILDS (Options, grid(), Cfg.sha256, module constants) equals the registered one: the
            registered 9 = lr {1e-4,3e-4,1e-3} x anchor {0.03,0.1,0.3}, K = 8, epochs 10, batch-sets 256, seeds 0,1, criterion pooled_oof, both-rule flat, fraction in {10,25,50,100}.
            The argv must NAME every lever (a default is not a declaration) and every path/pin must equal the registered one.
  G-LIVE    ONE real optimiser step on the first fit-set unit in key order, through R1's own loader and fit path: finite loss; non-zero gradient on the TTC row (full, pair-only, anchor-only);
            EXACTLY 0.0 on every other row of score_head and on the features; the zero-weight control reads EXACTLY 0.0; a built-in leaky loss reads NON-zero (the probe can read a leak).
            Computes NO concordance and records no label statistic (booleans and unit identity only).
  G-BANK    S0_DONE exists (and no S0_FAILED); every bank unit's sha256 equals the S0 manifest (and every manifest unit exists); `logs_without_any_token` <= 5 % of the S0 logs; every selected
            log has a set in the bank (else R1 stops and needs a --replacements-file); R1's OWN manifest reader accepts the manifest (else R1 would crash at start).
  G-PATH    the dev-gate / navtest guard refuses: every forbidden token is refused for read and write, a legitimate bank path is allowed, and no path in the argv names a forbidden token.
  G-SUITE   test_srttc_r1.py in full form (`python -I test_srttc_r1.py -v`, CPU only, one process) on a clean `git archive` export of the code dir at the commit: 0 failures, 0 skips, >= 33 tests.

COMMANDS (exact; run on the dev box after S0 is done and the landed files are deployed to C:\\Users\\Admin\\srttc_r1\\code)
    mint   : C:\\Users\\Admin\\venvs\\driverl-eval\\Scripts\\python.exe -I C:\\Users\\Admin\\srttc_r1\\code\\srttc_r1_gate.py --fraction 10 --commit <40-hex landed commit>
    verify : C:\\Users\\Admin\\venvs\\driverl-eval\\Scripts\\python.exe -I C:\\Users\\Admin\\srttc_r1\\code\\srttc_r1_gate.py verify --fraction 10     (launch_r1.ps1 does this for you)
    argv   : ... srttc_r1_gate.py argv --fraction 10                                                                                               (prints the exact command, JSON)
  add `--stage lc` for the A2.4 learning-curve read (a separate token: it is a training run too).

CPU ONLY.  Never touches the GPU lock.  Never reads the dev-gate or navtest tables (every R1 file access is behind R1's own PathGuard).  Stdlib + the srttc_r1.py it gates (numpy, torch).
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import hmac
import io
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import types
from pathlib import Path

GATE_SCHEMA = "tanitad.launch_gate/1"             # == refe/gate_token.py == the programme's generic gate
TOKEN_KIND = "launch-gate-token"
PROFILE = "srttc-r1"
PROFILE_SYNTHETIC = "srttc-r1-synthetic"          # a token minted with test overrides can NEVER verify for the real profile
KEY_ENV = "TANITAD_GATE_KEY_FILE"
TOKEN_NAME = "r1_gate_token.json"
FAIL_NAME = "r1_gate_FAIL.json"
CHECKS = ("G-COMMIT", "G-DVB", "G-PATH", "G-BANK", "G-LIVE", "G-SUITE")

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# REGISTERED values.  Hard-coded HERE, independent of srttc_r1.py, so the gate compares the built object against the registration and not against itself.
# Sources: MM_DECISIONS_R1.md rows 1-10; PREREG_SELRANK_ARM.md section 2 (grid, K); amendment2 A2.1-A2.4.
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
REG_EPOCHS = 10                                    # row 1
REG_BATCH_SETS = 256                               # row 1
REG_SEEDS = (0, 1)                                 # rows 2-3
REG_CRITERION = "pooled_oof"                       # row 2
REG_BOTH_RULE = "flat"                             # row 6
REG_FRACTIONS = (10, 25, 50, 100)                  # row 10 (f0 = 10 %, f_max = 100 %) + A2.1 item 4
REG_GRID_LR = (1e-4, 3e-4, 1e-3)                   # PREREG section 2
REG_GRID_ANCHOR = (0.03, 0.1, 0.3)                 # PREREG section 2
REG_K = 8
REG_PAIR_W = 1.0                                   # row 5
REG_ADAM_BETAS = (0.9, 0.999)                      # row 1
REG_ADAM_EPS = 1e-8
REG_TTC_COL = 3
REG_LABEL_ORDER = ("nc", "dac", "ep", "ttc", "c", "ddc")
REG_N_FOLDS = 5
REG_FOLD_SALT = "SR-TTC/A2/v1"
REG_RULE = "navsim_v1"
REG_N_FIT_LOGS = 1167
REG_MIN_LOGS_PER_FOLD = 20
REG_ROW_TOL = 1e-2
REG_RANKED_SHA256 = "a61275037a3d2b06903dc7e1ad0f6f99fb740f668a6ec7488fdef6fe901ce9b1"
REG_LC_DELTA = 0.010
REG_LC_PREFIXES = (25, 50, 100)
REG_BOOT_B = 10000
REG_BOOT_SEED = 20260927
REG_MAX_EMPTY_LOG_FRACTION = 0.05                  # row 11: logs_without_any_token <= 5 %
REG_MIN_SUITE_TESTS = 33                           # test_srttc_r1.py at 58bd819 has 33 tests; fewer means tests were deleted
REG_PATHS = {"bank_dir": "C:/Users/Admin/srttc_s0/work/bank", "manifest": "C:/Users/Admin/srttc_s0/work/bank/MANIFEST.sha256.json",
             "ranked_logs": "C:/Users/Admin/srttc_s0/run/plan/ranked_fit_logs.tsv", "s0_marker": "C:/Users/Admin/srttc_s0/run/S0_DONE",
             "p2_ckpt": "D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt"}
REQUIRED_FORBIDDEN_TOKENS = ("heldout", "p2ho", "table_p2", "navtest", "gate_table", "dev_gate", "devgate", "dev-gate")
REQUIRED_EXPLICIT = ("--fraction", "--out-dir", "--threads", "--epochs", "--batch-sets", "--seeds", "--criterion", "--both-rule")
PATH_FLAGS = ("--bank-dir", "--manifest", "--ranked-logs", "--p2-ckpt", "--out-dir", "--s0-marker", "--replacements-file")

DEFAULT_PYTHON = r"C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe"
DEFAULT_CODE_DIR = r"C:\Users\Admin\srttc_r1\code"
DEFAULT_RUN_ROOT = r"C:\Users\Admin\srttc_r1\run"
DEFAULT_GATE_ROOT = r"C:\Users\Admin\srttc_r1\gate"
DEFAULT_GIT_DIR = "C:/Users/Admin/tanitad-push/.git"
DEFAULT_REPO_CODE_PATH = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-18-sel-rank/code"
BOUND_CODE_FILES = ("srttc_r1.py", "srttc_r1_gate.py")          # run from the deployed code dir; each must equal its blob at the commit
SUITE_TEST_FILE = "test_srttc_r1.py"
GATE_TEST_FILE = "test_srttc_r1_gate.py"


class GateError(RuntimeError):
    """A condition under which the gate cannot certify anything."""


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# primitives: byte-identical re-implementations of refe/gate_token.py (pinned by known-answer vectors in test_srttc_r1_gate.py)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    buf = bytearray(chunk)
    mv = memoryview(buf)
    with open(p, "rb", buffering=0) as fh:
        while True:
            n = fh.readinto(mv)
            if not n:
                break
            h.update(mv[:n])
    return h.hexdigest()


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")


def argv_sha256(argv: list) -> str:
    if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
        raise GateError("argv must be a list of strings")
    return sha256_bytes(json.dumps(argv, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def write_json(path, obj) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str), encoding="utf-8", newline="\n")
    os.replace(tmp, p)
    return p


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def default_key_file() -> Path:
    env = os.environ.get(KEY_ENV)
    return Path(env) if env else Path.home() / ".config" / "tanitad" / "launch_gate.key"


def load_key(path=None, *, create: bool = True, forbid_under=()) -> bytes:
    """32 random bytes, hex on disk, created on first use.  Refused inside a code tree: a key that can be shipped is not a key."""
    p = Path(path) if path else default_key_file()
    rp = os.path.normcase(str(p.resolve()))
    for t in forbid_under:
        rt = os.path.normcase(str(Path(t).resolve()))
        if rp == rt or rp.startswith(rt + os.sep):
            raise GateError(f"refusing a key file inside the code tree {t}: {p}")
    if not p.exists():
        if not create:
            raise GateError(f"no gate key at {p} (mint on this host first)")
        p.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="ascii") as fh:
            fh.write(secrets.token_hex(32) + "\n")
    txt = p.read_text(encoding="ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", txt):
        raise GateError(f"gate key {p} is not 64 hex chars")
    return bytes.fromhex(txt)


def key_id(key: bytes) -> str:
    return sha256_bytes(b"tanitad-launch-gate-key-id:" + key)[:16]


def sign(body: dict, key: bytes) -> str:
    return hmac.new(key, canonical_json(body), hashlib.sha256).hexdigest()


def signature_ok(token: dict, key: bytes) -> bool:
    body = {k: v for k, v in token.items() if k != "hmac_sha256"}
    got = str(token.get("hmac_sha256", ""))
    return len(got) == 64 and hmac.compare_digest(sign(body, key), got)


def tree_digest(man: dict) -> str:
    return sha256_bytes(canonical_json(man))


def fingerprint_file(path) -> dict:
    """== refe/gate_token.fingerprint for a file; `missing` otherwise (a directory is not an input here)."""
    p = Path(path)
    if p.is_file():
        return {"kind": "file", "bytes": p.stat().st_size, "sha256": sha256_file(p), "method": "sha256-v1"}
    return {"kind": "missing"}


def fp_key(fp: dict) -> str:
    if fp.get("kind") == "file":
        return f"file:{fp.get('sha256')}"
    return f"{fp.get('kind')}:{fp.get('sha256', fp.get('why', ''))}"


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# git (read-only): blobs at a commit, and a clean export of the code dir
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def _git_exe() -> str:
    exe = shutil.which("git")
    if exe:
        return exe
    for c in (r"C:\Program Files\Git\cmd\git.exe", r"C:\Program Files\Git\bin\git.exe"):
        if os.path.exists(c):
            return c
    raise GateError("git is not on PATH")


def _clean_env() -> dict:
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_") and not k.startswith("PYTHON")}


def _git(git_dir: str, *args, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run([_git_exe(), "-c", "core.autocrlf=false", "--git-dir", git_dir, *args], capture_output=True, env=_clean_env(), timeout=timeout)


def git_is_commit(git_dir: str, commit: str) -> bool:
    if not re.fullmatch(r"[0-9a-f]{40}", commit or ""):
        return False
    r = _git(git_dir, "cat-file", "-t", commit)
    return r.returncode == 0 and r.stdout.decode().strip() == "commit"


def git_blob(git_dir: str, commit: str, path: str):
    r = _git(git_dir, "cat-file", "blob", f"{commit}:{path}")
    return r.stdout if r.returncode == 0 else None


def export_code_dir(git_dir: str, commit: str, repo_code_path: str, dest: str) -> list:
    """`git archive` of the code dir at `commit` extracted flat into `dest` (prefix stripped).  Refuses a path escape.  -> relative names written."""
    r = _git(git_dir, "archive", "--format=tar", commit, "--", repo_code_path, timeout=300)
    if r.returncode != 0:
        raise GateError(f"git archive {commit[:12]} failed: {r.stderr.decode('utf-8', 'replace')[:200]}")
    prefix = repo_code_path.strip("/") + "/"
    out = []
    root = os.path.realpath(dest)
    with tarfile.open(fileobj=io.BytesIO(r.stdout)) as tar:
        for m in tar:
            if not m.isfile() or not m.name.startswith(prefix):
                continue
            rel = m.name[len(prefix):]
            tgt = os.path.realpath(os.path.join(dest, rel))
            if os.path.commonpath([root, tgt]) != root:
                raise GateError(f"archive member {m.name!r} escapes the export dir")
            os.makedirs(os.path.dirname(tgt), exist_ok=True)
            with open(tgt, "wb") as f:
                f.write(tar.extractfile(m).read())
            out.append(rel)
    return out


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the gated code: loaded from the VERIFIED bytes (read once, hashed, then exec'd from those same bytes: no check-then-use gap)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
_MODULES: dict = {}


def load_verified_module(code_dir: str, expect_sha256: str | None = None):
    p = os.path.join(code_dir, "srttc_r1.py")
    src = Path(p).read_bytes()
    sha = sha256_bytes(src)
    if expect_sha256 is not None and sha != expect_sha256:
        raise GateError(f"srttc_r1.py sha256 {sha} != the verified {expect_sha256}")
    name = f"srttc_r1__gate_{sha[:12]}"
    if name in _MODULES:
        return _MODULES[name], sha
    mod = types.ModuleType(name)
    mod.__file__ = p
    sys.modules[name] = mod                      # dataclasses (with `from __future__ import annotations`) looks the module up here
    exec(compile(src, p, "exec", dont_inherit=True), mod.__dict__)
    _MODULES[name] = mod
    return mod, sha


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# configuration and the exact command
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def join_path(d: str, name: str) -> str:
    sep = "\\" if "\\" in d else "/"
    return d.rstrip("\\/") + sep + name


def build_argv(*, python: str, code_dir: str, stage: str, fraction: int, out_dir: str, threads: int, epochs: int, batch_sets: int, seeds: str, criterion: str, both_rule: str,
               extra=()) -> list:
    """THE exact R1 command.  Every lever is NAMED (a flag whose default happens to be right is still declared).  launch_r1.ps1 runs exactly this list."""
    return [python, "-I", join_path(code_dir, "srttc_r1.py"), stage, "--fraction", str(fraction), "--out-dir", out_dir, "--threads", str(threads), "--epochs", str(epochs),
            "--batch-sets", str(batch_sets), "--seeds", seeds, "--criterion", criterion, "--both-rule", both_rule, *list(extra)]


@dataclasses.dataclass
class GateConfig:
    commit: str = ""
    fraction: int = 10
    stage: str = "run"
    python: str = DEFAULT_PYTHON
    code_dir: str = DEFAULT_CODE_DIR
    out_dir: str = ""
    threads: int = 8
    epochs: int = REG_EPOCHS
    batch_sets: int = REG_BATCH_SETS
    seeds: str = "0,1"
    criterion: str = REG_CRITERION
    both_rule: str = REG_BOTH_RULE
    replacements_file: str | None = None
    git_dir: str = DEFAULT_GIT_DIR
    repo_code_path: str = DEFAULT_REPO_CODE_PATH
    token_path: str = ""
    key_file: str | None = None
    suite_timeout_s: int = 1800
    suite_test_file: str = SUITE_TEST_FILE
    suite_min_tests: int = REG_MIN_SUITE_TESTS

    def resolved_out_dir(self) -> str:
        return self.out_dir or f"{DEFAULT_RUN_ROOT}\\f{self.fraction}"

    def resolved_token_path(self) -> str:
        return self.token_path or os.path.join(DEFAULT_GATE_ROOT, f"f{self.fraction}_{self.stage}", TOKEN_NAME)

    def extra_cli(self) -> list:
        return ["--replacements-file", self.replacements_file] if self.replacements_file else []

    def argv(self) -> list:
        return build_argv(python=self.python, code_dir=self.code_dir, stage=self.stage, fraction=self.fraction, out_dir=self.resolved_out_dir(), threads=self.threads,
                          epochs=self.epochs, batch_sets=self.batch_sets, seeds=self.seeds, criterion=self.criterion, both_rule=self.both_rule, extra=self.extra_cli())


def _res(status: str, reasons=None, details=None) -> dict:
    return {"status": status, "reasons": list(reasons or []), "details": details or {}}


def _norm(p) -> str:
    return os.path.normpath(str(p)).replace("\\", "/").lower()


def _cli_flags(cli: list) -> tuple:
    """-> ([(flag, value)], [problems]).  Every srttc_r1.py flag takes exactly one value."""
    pairs, problems, i = [], [], 0
    while i < len(cli):
        t = cli[i]
        if not t.startswith("--"):
            problems.append(f"unexpected token {t!r}")
            i += 1
            continue
        if i + 1 >= len(cli) or cli[i + 1].startswith("--"):
            problems.append(f"flag {t} has no value")
            i += 1
            continue
        pairs.append((t, cli[i + 1]))
        i += 2
    return pairs, problems


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# G-COMMIT
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def check_commit(*, commit: str, git_dir: str, repo_code_path: str, code_dir: str, test_files=(SUITE_TEST_FILE,)) -> dict:
    reasons, det = [], {"commit": commit, "files": {}}
    try:
        if not git_is_commit(git_dir, commit):
            return _res("FAIL", [f"{commit!r} is not a 40-hex commit object in {git_dir}"], det)
        for f in BOUND_CODE_FILES:
            p = os.path.join(code_dir, f)
            if not os.path.isfile(p):
                reasons.append(f"{f} is not deployed in {code_dir}")
                continue
            disk = sha256_file(p)
            blob = git_blob(git_dir, commit, f"{repo_code_path}/{f}")
            if blob is None:
                reasons.append(f"{f} is not in commit {commit[:12]} (not landed)")
            elif sha256_bytes(blob) != disk:
                reasons.append(f"{f} differs from its blob at {commit[:12]} (disk {disk[:12]}, blob {sha256_bytes(blob)[:12]})")
            det["files"][f] = disk
        for f in test_files:
            blob = git_blob(git_dir, commit, f"{repo_code_path}/{f}")
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
def check_dvb(mod, argv: list, *, stage: str, strict: bool = True) -> tuple:
    """-> (check dict, Options or None).  Inspects the BUILT objects (parsed Options, grid(), Cfg.sha256, module constants), not the argv text."""
    reasons, det = [], {}
    opts = None
    try:
        if len(argv) < 4 or argv[1] != "-I" or os.path.basename(str(argv[2]).replace("\\", "/")) != "srttc_r1.py":
            reasons.append(f"argv must start [python, '-I', <...>/srttc_r1.py, {stage}]: got {argv[:4]}")
            return _res("FAIL", reasons, det), None
        cli = argv[3:]
        if cli[0] != stage:
            reasons.append(f"sub-command {cli[0]!r} != the stage being gated {stage!r}")
        pairs, problems = _cli_flags(cli[1:])
        reasons += problems
        names = [f for f, _ in pairs]
        for f in sorted({n for n in names if names.count(n) > 1}):
            reasons.append(f"flag {f} is given more than once (argparse would silently keep the last)")
        parser = mod.build_parser()
        known = {s for a in parser._actions for s in a.option_strings}
        for f in names:
            if f not in known:
                reasons.append(f"unknown flag {f}")
        for f in REQUIRED_EXPLICIT:
            if f not in names:
                reasons.append(f"lever {f} is not NAMED in the argv (a default is not a declaration)")
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                ns = parser.parse_args(cli)
        except SystemExit:
            reasons.append(f"srttc_r1.py's own parser refuses this argv: {err.getvalue().strip().splitlines()[-1] if err.getvalue().strip() else '(no message)'}")
            return _res("FAIL", reasons, det), None
        opts = mod.options_from_args(ns)
        # --- the levers, read from the BUILT Options
        built = {"epochs": opts.epochs, "batch_sets": opts.batch_sets, "seeds": tuple(opts.seeds), "criterion": opts.criterion, "both_rule": opts.both_rule, "fraction": opts.fraction}
        want = {"epochs": REG_EPOCHS, "batch_sets": REG_BATCH_SETS, "seeds": REG_SEEDS, "criterion": REG_CRITERION, "both_rule": REG_BOTH_RULE}
        for k, v in want.items():
            if built[k] != v:
                reasons.append(f"lever {k} = {built[k]!r}, registered {v!r}")
        if built["fraction"] not in REG_FRACTIONS:
            reasons.append(f"fraction {built['fraction']!r} is not a registered nested size {REG_FRACTIONS}")
        if not (isinstance(opts.threads, int) and opts.threads >= 1):
            reasons.append(f"threads {opts.threads!r} must be a positive integer")
        det["levers"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in built.items()}
        # --- the grid R1 BUILDS
        cfgs = mod.grid(opts.epochs, opts.batch_sets)
        want_grid = [(lr, a) for lr in REG_GRID_LR for a in REG_GRID_ANCHOR]
        got_grid = [(c.lr, c.anchor) for c in cfgs]
        det["n_configs_built"] = len(cfgs)
        if len(cfgs) != 9:
            reasons.append(f"the built grid has {len(cfgs)} configurations, registered 9 (3 lr x 3 anchor)")
        if got_grid != want_grid:
            reasons.append(f"the built grid {got_grid} != the registered {want_grid} (order is part of the registration: ties go to the first in grid order)")
        for c in cfgs:
            if (c.k, c.pair_w, c.epochs, c.batch_sets) != (REG_K, REG_PAIR_W, REG_EPOCHS, REG_BATCH_SETS):
                reasons.append(f"built config {c.id()}: (k, pair_w, epochs, batch_sets) = {(c.k, c.pair_w, c.epochs, c.batch_sets)}, registered {(REG_K, REG_PAIR_W, REG_EPOCHS, REG_BATCH_SETS)}")
            payload = {**dataclasses.asdict(c), "optimizer": "adam", "betas": list(REG_ADAM_BETAS), "eps": REG_ADAM_EPS, "weight_decay": 0.0, "row": REG_TTC_COL}
            if c.sha256() != sha256_bytes(json.dumps(payload, sort_keys=True, indent=1, ensure_ascii=True).encode("utf-8")):
                reasons.append(f"built config {c.id()}: its config sha256 does not equal the registered optimiser/row payload (Adam {REG_ADAM_BETAS}, eps {REG_ADAM_EPS}, wd 0.0, row {REG_TTC_COL})")
        # --- module constants the run depends on
        consts = {"GRID_LR": REG_GRID_LR, "GRID_ANCHOR": REG_GRID_ANCHOR, "K": REG_K, "TTC_COL": REG_TTC_COL, "N_FOLDS": REG_N_FOLDS, "FOLD_SALT": REG_FOLD_SALT, "RULE": REG_RULE,
                  "ADAM_BETAS": REG_ADAM_BETAS, "ADAM_EPS": REG_ADAM_EPS, "SEEDS": REG_SEEDS, "LC_DELTA_FLAT": REG_LC_DELTA, "LC_PREFIXES": REG_LC_PREFIXES, "BOOT_B": REG_BOOT_B,
                  "BOOT_SEED": REG_BOOT_SEED, "N_FIT_LOGS": REG_N_FIT_LOGS, "MIN_LOGS_PER_FOLD": REG_MIN_LOGS_PER_FOLD, "RANKED_LOGS_SHA256": REG_RANKED_SHA256,
                  "ROW_CONSISTENCY_TOL": REG_ROW_TOL, "DEFAULT_EPOCHS": REG_EPOCHS, "DEFAULT_BATCH_SETS": REG_BATCH_SETS}
        for k, v in consts.items():
            if getattr(mod, k, "<absent>") != v:
                reasons.append(f"srttc_r1.{k} = {getattr(mod, k, '<absent>')!r}, registered {v!r}")
        if tuple(getattr(mod, "LABEL_ORDER", ())) != REG_LABEL_ORDER:
            reasons.append(f"srttc_r1.LABEL_ORDER = {getattr(mod, 'LABEL_ORDER', None)!r}, registered {REG_LABEL_ORDER!r}")
        # --- paths and pins (strict only; the API-level test overrides are what lets a synthetic world through, and they mint a synthetic-profile token)
        if strict:
            for k, v in REG_PATHS.items():
                if _norm(getattr(opts, k)) != _norm(v):
                    reasons.append(f"{k} = {getattr(opts, k)!r}, registered {v!r}")
            if opts.pins_overridden():
                reasons.append(f"pins overridden: {opts.pins_overridden()}")
            for k, v in (("ranked_sha256", REG_RANKED_SHA256), ("expected_n_logs", REG_N_FIT_LOGS), ("min_logs_per_fold", REG_MIN_LOGS_PER_FOLD), ("row_tol", REG_ROW_TOL)):
                if getattr(opts, k) != v:
                    reasons.append(f"pin {k} = {getattr(opts, k)!r}, registered {v!r}")
        det["strict"] = bool(strict)
    except Exception as e:                                       # noqa: BLE001
        reasons.append(f"{type(e).__name__}: {str(e)[:300]}")
    return _res("FAIL" if reasons else "PASS", reasons, det), opts


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# G-PATH
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def check_path(mod, opts, argv: list) -> dict:
    reasons, det = [], {}
    try:
        for t in REQUIRED_FORBIDDEN_TOKENS:
            if t not in mod.FORBIDDEN_TOKENS:
                reasons.append(f"the guard's forbidden-token list lost {t!r}")
        paths = [argv[0], argv[2]]
        cli = argv[3:]
        for f, v in _cli_flags(cli[1:])[0]:
            if f in PATH_FLAGS:
                paths.append(v)
        paths += [opts.bank_dir, opts.manifest, opts.ranked_logs, opts.p2_ckpt, opts.resolved_out_dir()] + ([opts.s0_marker] if opts.s0_marker else [])
        for p in paths:
            hit = mod.PathGuard.forbidden(p)
            if hit:
                reasons.append(f"a path in this launch names a dev-gate / navtest token {hit}: {p!r}")
        guard = opts.guard()
        out_probe = os.path.join(opts.resolved_out_dir(), "r1_result.json")
        n_refused = 0
        for tok in mod.FORBIDDEN_TOKENS:
            rp = os.path.join(opts.bank_dir, f"x_{tok}_y.sets.npz")
            try:
                guard.check_read(rp)
                reasons.append(f"the guard did NOT refuse a read of {rp!r} (token {tok!r})")
            except mod.PathGuardError as e:
                if "dev-gate / navtest table token" in str(e):
                    n_refused += 1
                else:
                    reasons.append(f"the guard refused {tok!r} for the wrong reason: {str(e)[:120]}")
            try:
                guard.check_write(os.path.join(opts.resolved_out_dir(), f"x_{tok}_y.json"))
                reasons.append(f"the guard did NOT refuse a write of a path naming {tok!r}")
            except mod.PathGuardError:
                pass
        try:                                                      # controls: legitimate paths are ALLOWED (a guard that refuses everything proves nothing)
            guard.check_read(os.path.join(opts.bank_dir, "L0123456789.sets.npz"))
            guard.check_read(opts.manifest)
            guard.check_write(out_probe)
        except mod.PathGuardError as e:
            reasons.append(f"control: the guard refused a LEGITIMATE path: {str(e)[:160]}")
        try:
            guard.check_read(os.path.join(tempfile.gettempdir(), "srttc_gate_outside_probe.bin"))
            reasons.append("the guard did NOT refuse a read outside its allow-list")
        except mod.PathGuardError:
            pass
        det.update({"n_tokens": len(mod.FORBIDDEN_TOKENS), "n_tokens_refused_for_the_token": n_refused, "n_paths_scanned": len(paths)})
    except Exception as e:                                       # noqa: BLE001
        reasons.append(f"{type(e).__name__}: {str(e)[:300]}")
    return _res("FAIL" if reasons else "PASS", reasons, det)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# G-BANK
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def _unit_table(doc: dict) -> tuple:
    table, dups = {}, []
    for rel, ent in doc.get("files", {}).items():
        base = os.path.basename(str(ent.get("path", rel)).replace("\\", "/"))
        if base.endswith(".sets.npz"):
            if base in table and table[base] != ent["sha256"]:
                dups.append(base)
            table[base] = ent["sha256"]
    return table, dups


def _named_shas(doc: dict, name: str) -> set:
    return {ent["sha256"] for rel, ent in doc.get("files", {}).items() if os.path.basename(str(ent.get("path", rel)).replace("\\", "/")) == name}


def _safe_read(mod, path: str) -> bytes:
    hit = mod.PathGuard.forbidden(path)
    if hit:
        raise GateError(f"refused to read {path!r}: names {hit}")
    with open(path, "rb") as f:
        return f.read()


def unit_log_index(mod, guard, bank_dir: str, units: list) -> dict:
    """{unit: sorted unique log names}.  Reads the small `log` member of each unit only."""
    out = {}
    for n in units:
        z = mod._io_npz(guard, os.path.join(bank_dir, n))
        try:
            out[n] = sorted({str(x) for x in z["log"]})
        finally:
            z.close()
    return out


def selected_rows(mod, guard, opts) -> tuple:
    rows, ranked_sha = mod.load_ranked_logs(guard, opts.ranked_logs, opts.ranked_sha256, opts.expected_n_logs)
    repl = json.loads(mod._io_read_bytes(guard, opts.replacements_file).decode("utf-8")) if opts.replacements_file else None
    selected, rep = mod.select_logs(rows, opts.fraction, repl)
    return selected, ranked_sha, rep


def check_bank(mod, opts) -> tuple:
    """-> (check dict, aux).  aux = {manifest_sha256, verified_units, unit_logs} for G-LIVE and the token."""
    reasons, det, aux = [], {}, {}
    try:
        guard = opts.guard()
        marker = opts.s0_marker
        if not marker or not mod._io_exists(guard, marker):
            return _res("FAIL", [f"{marker} does not exist: S0 is not done, R1 does not start"], det), aux
        failed = os.path.join(os.path.dirname(marker), "S0_FAILED")
        if os.path.exists(failed):
            reasons.append(f"{failed} exists next to S0_DONE (a failed S0 must not be launched on)")
        raw = mod._io_read_bytes(guard, opts.manifest)
        man_sha = sha256_bytes(raw)
        doc = json.loads(raw.decode("utf-8"))
        table, dups = _unit_table(doc)
        aux["manifest_sha256"] = man_sha
        det.update({"manifest_sha256": man_sha, "n_manifest_units": len(table)})
        for d in dups:
            reasons.append(f"the manifest lists unit {d} twice with different sha256")
        # R1's OWN reader must accept this manifest, or R1 crashes at start (built and tested on a synthetic manifest, unreachable on the real one)
        try:
            mod.load_manifest(guard, opts.manifest)
            det["r1_reads_manifest"] = True
        except Exception as e:                                   # noqa: BLE001
            det["r1_reads_manifest"] = False
            reasons.append(f"R1's own load_manifest REJECTS this manifest, so run_r1 would crash in _setup before any fit: {type(e).__name__}: {str(e)[:200]}")
        units = [n for n in mod._io_listdir(guard, opts.bank_dir) if n.endswith(".sets.npz")]
        if not units:
            reasons.append(f"no *.sets.npz in {opts.bank_dir}")
        for n in units:
            if n not in table:
                reasons.append(f"unit {n} is on disk but not in the manifest")
        for n in sorted(table):
            if n not in units:
                reasons.append(f"unit {n} is in the manifest but not on disk")
        verified, bad, nbytes = {}, [], 0
        for n in units:
            if n not in table:
                continue
            p = os.path.join(opts.bank_dir, n)
            sha = mod._io_sha256(guard, p)
            nbytes += os.path.getsize(p)
            if sha != table[n]:
                bad.append(n)
                reasons.append(f"unit {n}: sha256 {sha[:16]} != manifest {table[n][:16]}")
            else:
                verified[n] = sha
        det.update({"n_units_on_disk": len(units), "n_units_verified": len(verified), "n_units_sha_mismatch": len(bad), "bytes_hashed": nbytes})
        aux["verified_units"] = verified
        # S0 summary: logs_without_any_token <= 5 %
        summ_path = os.path.join(os.path.dirname(marker), "S0_SUMMARY.json")
        if not os.path.exists(summ_path):
            reasons.append(f"{summ_path} does not exist (the A2.1 item 7 coverage report)")
        else:
            sraw = _safe_read(mod, summ_path)
            sdoc = json.loads(sraw.decode("utf-8"))
            listed = _named_shas(doc, "S0_SUMMARY.json")
            if sha256_bytes(sraw) not in listed:
                reasons.append(f"S0_SUMMARY.json sha256 {sha256_bytes(sraw)[:16]} is not the manifest's {sorted(x[:16] for x in listed) or '(not listed)'}")
            n_logs, n_empty = int(sdoc["n_logs"]), int(sdoc["n_logs_without_any_token"])
            if n_logs <= 0:
                reasons.append("S0_SUMMARY.json n_logs <= 0")
            else:
                frac = n_empty / n_logs
                det.update({"s0_n_logs": n_logs, "s0_logs_without_any_token": n_empty, "s0_logs_without_any_token_fraction": round(frac, 6), "s0_PARTIAL": bool(sdoc.get("PARTIAL"))})
                if frac > REG_MAX_EMPTY_LOG_FRACTION:
                    reasons.append(f"logs_without_any_token {n_empty} of {n_logs} = {100 * frac:.2f} % > {100 * REG_MAX_EMPTY_LOG_FRACTION:g} %")
            if len(sdoc.get("logs_without_any_token", [])) != n_empty:
                reasons.append("S0_SUMMARY.json: logs_without_any_token list length != n_logs_without_any_token")
            if sdoc.get("replacement_rule_exceeded"):
                reasons.append("S0_SUMMARY.json: replacement_rule_exceeded is true")
        # coverage of the SELECTED logs by the bank, and the A2.2 fold floor
        selected, ranked_sha, rep = selected_rows(mod, guard, opts)
        per_fold = [sum(1 for r in selected if mod.fold_of_rank(r["rank"]) == k) for k in range(REG_N_FOLDS)]
        det.update({"n_selected_logs": len(selected), "logs_per_fold": per_fold, "ranked_fit_logs_sha256": ranked_sha, "replacements": {"dropped": len(rep["dropped"]), "added": len(rep["added"])}})
        if min(per_fold) < opts.min_logs_per_fold:
            reasons.append(f"A2.2: a fold has {min(per_fold)} logs < the minimum {opts.min_logs_per_fold}")
        unit_logs = unit_log_index(mod, guard, opts.bank_dir, sorted(verified))
        aux["unit_logs"] = unit_logs
        in_bank = {lg for v in unit_logs.values() for lg in v}
        missing = [r["log"] for r in selected if r["log"] not in in_bank]
        det["n_selected_logs_missing_from_bank"] = len(missing)
        if missing:
            reasons.append(f"{len(missing)} selected log(s) have no set in the bank (R1 stops with BankError; supply an explicit --replacements-file, A2.1 item 7): e.g. {missing[:2]}")
    except Exception as e:                                       # noqa: BLE001
        reasons.append(f"{type(e).__name__}: {str(e)[:300]}")
    return _res("FAIL" if reasons else "PASS", reasons, det), aux


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# G-LIVE
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def _gabs(t) -> float:
    return 0.0 if t is None else float(t.abs().sum())


def check_live(mod, opts, aux: dict, *, tmp_root: str | None = None) -> dict:
    """ONE optimiser step on the first fit-set unit in key order, through R1's own loader (`load_fit_data`) and fit path (`fit_row`).  No concordance, no label statistic."""
    import numpy as np
    import torch
    reasons, det = [], {}
    tmp = None
    try:
        torch.set_num_threads(max(1, min(int(opts.threads), 4)))
        guard = opts.guard()
        selected, _sha, _rep = selected_rows(mod, guard, opts)
        rank_of = {r["log"]: r["rank"] for r in selected}
        unit_logs = aux.get("unit_logs") or unit_log_index(mod, guard, opts.bank_dir, [n for n in mod._io_listdir(guard, opts.bank_dir) if n.endswith(".sets.npz")])
        best = None
        for n in sorted(unit_logs):
            rk = [rank_of[lg] for lg in unit_logs[n] if lg in rank_of]
            if rk and (best is None or min(rk) < best[0]):
                best = (min(rk), n)
        if best is None:
            return _res("FAIL", ["no bank unit holds a selected fit log"], det)
        first_rank, unit = best
        det.update({"unit": unit, "first_key_rank": first_rank})
        raw = mod._io_read_bytes(guard, opts.manifest)
        table, _dups = _unit_table(json.loads(raw.decode("utf-8")))
        if unit not in table:
            return _res("FAIL", [f"unit {unit} is not in the manifest"], det)
        # a private one-unit bank: the COPY is hashed against the manifest, so what R1's loader reads is verified bytes
        tmp = tempfile.mkdtemp(prefix="srttc_live_", dir=tmp_root)
        shutil.copyfile(os.path.join(opts.bank_dir, unit), os.path.join(tmp, unit))
        sha = sha256_file(os.path.join(tmp, unit))
        if sha != table[unit]:
            return _res("FAIL", [f"unit {unit}: sha256 {sha[:16]} != manifest {table[unit][:16]}"], det)
        with open(os.path.join(tmp, "MANIFEST.sha256.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"files": {unit: {"path": unit, "sha256": sha}}}, f)
        opts_live = dataclasses.replace(opts, bank_dir=tmp, manifest=os.path.join(tmp, "MANIFEST.sha256.json"))
        glive = opts_live.guard()
        in_unit = set(unit_logs[unit])
        sel_unit = [r for r in selected if r["log"] in in_unit]
        W6, b6 = mod.load_p2_row(glive, opts.p2_ckpt)
        data = mod.load_fit_data(glive, opts_live, sel_unit, {unit: sha}, W6, b6)       # shapes, finiteness, label range, frozen top8, P2-row == stored logits
        det.update({"n_sets_in_unit": int(data.n_sets), "n_selected_logs_in_unit": len(sel_unit), "p2_row_reproduces_stored_logits": True})
        batch = mod._probe_batch(data, opts.reach_n_sets)
        nb = int(len(batch["feat"]))
        anchor = mod.GRID_ANCHOR[1]
        pair_w = 1.0

        def probe(pw, aw, loss_fn=None):
            loss_fn = loss_fn or mod.total_loss
            head = mod.RowHead(W6, b6, probe=True, perturb=True)
            feat = batch["feat"].clone().float().requires_grad_(True)
            z6 = mod.forward_z6(head, feat, batch["z6"])
            loss, _p, _a = loss_fn(z6, batch["y3"], batch["top8"], aw, pw)
            fin = bool(torch.isfinite(loss))
            if loss.requires_grad:                              # a detached TTC row leaves the loss with no graph: that is a ZERO gradient (reported below), not a crash
                loss.backward()
            return {"finite": fin, "ttc_w": _gabs(head.row_w.grad), "ttc_b": _gabs(head.row_b.grad), "frozen_w": _gabs(head.frozen_W6.grad), "frozen_b": _gabs(head.frozen_b6.grad),
                    "features": _gabs(feat.grad)}

        full, pair, anch, zero = probe(pair_w, anchor), probe(pair_w, 0.0), probe(0.0, 1.0), probe(0.0, 0.0)

        def leaky(z6, y3, idx8, aw, pw=1.0):                    # positive control: a loss that reads the EP column MUST show a non-zero frozen-row gradient
            loss, p, a = mod.total_loss(z6, y3, idx8, aw, pw)
            return loss + 0.1 * z6[..., 2].pow(2).mean(), p, a

        leak = probe(pair_w, anchor, leaky)
        for name, r in (("full", full), ("pair_only", pair), ("anchor_only", anch)):
            if not r["finite"]:
                reasons.append(f"{name}: the loss is not finite")
            for k in ("ttc_w", "ttc_b"):
                if not (np.isfinite(r[k]) and r[k] > 0.0):
                    reasons.append(f"{name}: gradient on the TTC row ({k}) is {r[k]!r}: must be finite and > 0 (the loss does not reach the TTC row)")
        for name, r in (("full", full), ("pair_only", pair), ("anchor_only", anch), ("zero_control", zero)):
            if r["frozen_w"] != 0.0 or r["frozen_b"] != 0.0:
                reasons.append(f"{name}: a NON-TTC row of score_head received a gradient (w {r['frozen_w']!r}, b {r['frozen_b']!r}); must be EXACTLY 0.0")
            if r["features"] != 0.0:
                reasons.append(f"{name}: the features received a gradient ({r['features']!r}); the trunk must get EXACTLY 0.0")
        if zero["ttc_w"] != 0.0 or zero["ttc_b"] != 0.0:
            reasons.append(f"zero_control: weights (0, 0) still gave a TTC-row gradient ({zero['ttc_w']!r}, {zero['ttc_b']!r}); the probe is not reading the loss")
        if not (leak["frozen_w"] > 0.0):
            reasons.append("positive control: a loss that reads another column showed NO frozen-row gradient, so a zero reading above would mean nothing")
        # the real fit path: ONE optimiser step (epochs 1, one batch) must move the TTC row
        mini = mod.FitData(log_names=["probe"], log_rank=np.zeros(1, np.int64), log_fold=np.zeros(1, np.int64), log_idx=np.zeros(nb, np.int64), fold=np.zeros(nb, np.int64),
                           feat=batch["feat"], z6=batch["z6"], y3=batch["y3"], top8=batch["top8"], provenance={})
        moved = float("nan")
        try:
            w, b = mod.fit_row(mini, np.arange(nb), mod.Cfg(lr=mod.GRID_LR[-1], anchor=anchor, epochs=1, batch_sets=nb), 0, W6, b6)
            moved = float(np.abs(np.asarray(w) - np.asarray(W6)[mod.TTC_COL]).max())
            if np.asarray(w).shape != (mod.FEAT_DIM,) or np.asarray(b).shape != (1,):
                reasons.append(f"the fitted row has shape {np.asarray(w).shape}/{np.asarray(b).shape}, expected ({mod.FEAT_DIM},)/(1,)")
        except Exception as e:                                   # noqa: BLE001
            reasons.append(f"the real fit path (fit_row) raised on one batch: {type(e).__name__}: {str(e)[:160]}")
        if not (np.isfinite(moved) and moved > 0.0):
            reasons.append(f"one optimiser step at lr {mod.GRID_LR[-1]:g} moved the TTC row by {moved!r}: the loss is not wired to the optimiser")
        det.update({"loss_finite": bool(full["finite"] and pair["finite"] and anch["finite"]),
                    "ttc_row_gradient_nonzero": {"full": full["ttc_w"] > 0 and full["ttc_b"] > 0, "pair_only": pair["ttc_w"] > 0 and pair["ttc_b"] > 0, "anchor_only": anch["ttc_w"] > 0 and anch["ttc_b"] > 0},
                    "other_rows_and_features_exactly_zero": all(r["frozen_w"] == 0.0 and r["frozen_b"] == 0.0 and r["features"] == 0.0 for r in (full, pair, anch, zero)),
                    "zero_control_exactly_zero": zero["ttc_w"] == 0.0 and zero["ttc_b"] == 0.0, "leak_control_reads_nonzero": bool(leak["frozen_w"] > 0.0),
                    "one_step_moved_the_ttc_row": bool(moved > 0.0), "batch_sets": nb, "concordance_computed": False})
    except Exception as e:                                       # noqa: BLE001
        reasons.append(f"{type(e).__name__}: {str(e)[:300]}")
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    return _res("FAIL" if reasons else "PASS", reasons, det)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# G-SUITE
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def check_suite(*, commit: str, git_dir: str, repo_code_path: str, python: str, test_file: str = SUITE_TEST_FILE, min_tests: int = REG_MIN_SUITE_TESTS, timeout_s: int = 1800) -> dict:
    """The tests in full form on a clean export at `commit`: `python -I test_srttc_r1.py -v` with cwd = the export.  CPU only (CUDA hidden), 2 threads, ONE process."""
    reasons, det = [], {}
    tmp = tempfile.mkdtemp(prefix="srttc_suite_")
    try:
        files = export_code_dir(git_dir, commit, repo_code_path, tmp)
        if test_file not in files:
            return _res("FAIL", [f"{test_file} is not in the export of {commit[:12]}"], det)
        blob = git_blob(git_dir, commit, f"{repo_code_path}/{test_file}")
        if blob is None or sha256_file(os.path.join(tmp, test_file)) != sha256_bytes(blob):
            return _res("FAIL", [f"the exported {test_file} differs from its blob"], det)
        py = python if python and os.path.exists(python) else sys.executable
        env = _clean_env()
        env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
        t0 = time.time()
        try:
            cp = subprocess.run([py, "-I", test_file, "-v"], cwd=tmp, env=env, capture_output=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return _res("FAIL", [f"{test_file} did not finish in {timeout_s} s"], det)
        out = (cp.stdout + b"\n" + cp.stderr).decode("utf-8", "replace")
        m = re.findall(r"^Ran (\d+) tests? in ([\d.]+)s", out, re.M)
        n = int(m[-1][0]) if m else 0
        lines = [ln for ln in out.strip().splitlines() if ln.strip()]
        status = lines[-1].strip() if lines else ""
        det.update({"returncode": cp.returncode, "n_tests": n, "status_line": status, "seconds": round(time.time() - t0, 1), "test_file_sha256": sha256_bytes(blob), "interpreter": py})
        if cp.returncode != 0:
            reasons.append(f"{test_file} exited {cp.returncode}: {status[:80]}; tail: {' | '.join(ln.strip()[:100] for ln in lines[-6:])}")
        if status != "OK":
            reasons.append(f"the final line is {status[:80]!r}, not 'OK' (failures, errors, skips and expected-failures are all refused)")
        if n < min_tests:
            reasons.append(f"only {n} tests ran, the floor is {min_tests} (tests deleted or not collected)")
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
    """Run every check; -> {checks, mod_sha, opts, aux, argv}.  `overrides` (Options fields) is the API-level test hook: the CLI cannot reach it."""
    argv = cfg.argv()
    checks: dict = {}
    mod = mod_sha = opts = None
    aux: dict = {}
    checks["G-COMMIT"] = check_commit(commit=cfg.commit, git_dir=cfg.git_dir, repo_code_path=cfg.repo_code_path, code_dir=cfg.code_dir,
                                      test_files=(cfg.suite_test_file,))
    log(f"  G-COMMIT {checks['G-COMMIT']['status']}")
    if checks["G-COMMIT"]["status"] == "PASS":
        try:
            mod, mod_sha = load_verified_module(cfg.code_dir, checks["G-COMMIT"]["details"]["files"]["srttc_r1.py"])
        except Exception as e:                                   # noqa: BLE001
            checks["G-COMMIT"] = _res("FAIL", [f"cannot load srttc_r1.py from the verified bytes: {type(e).__name__}: {str(e)[:200]}"])
    if mod is None:
        for c in ("G-DVB", "G-PATH", "G-BANK", "G-LIVE"):
            checks[c] = _not_run("srttc_r1.py is not verified against its blob (G-COMMIT)")
    else:
        checks["G-DVB"], opts = check_dvb(mod, argv, stage=cfg.stage, strict=overrides is None)
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
                checks["G-LIVE"] = check_live(mod, opts, aux)
            else:
                checks["G-LIVE"] = _not_run("G-BANK is not PASS: no unit is verified to read")
            log(f"  G-LIVE   {checks['G-LIVE']['status']}")
    if checks["G-COMMIT"]["status"] == "PASS":
        checks["G-SUITE"] = check_suite(commit=cfg.commit, git_dir=cfg.git_dir, repo_code_path=cfg.repo_code_path, python=cfg.python, test_file=cfg.suite_test_file,
                                        min_tests=cfg.suite_min_tests, timeout_s=cfg.suite_timeout_s)
    else:
        checks["G-SUITE"] = _not_run("the commit is not verified (G-COMMIT)")
    log(f"  G-SUITE  {checks['G-SUITE']['status']}")
    return {"checks": checks, "mod_sha": mod_sha, "opts": opts, "aux": aux, "argv": argv}


def _demote(path: Path, tag: str) -> None:
    if path.exists():
        (path.parent / "superseded").mkdir(parents=True, exist_ok=True)
        os.replace(path, path.parent / "superseded" / f"{path.stem}.{tag}{path.suffix}")


def mint(cfg: GateConfig, *, overrides: dict | None = None, log=print) -> dict:
    """Run every check and write `r1_gate_token.json` ONLY if all PASS (else a signed `r1_gate_FAIL.json` and no token).  -> {verdict, token_path, token, checks, reasons}."""
    log(f"SR-TTC R1 launch gate: stage {cfg.stage}, fraction {cfg.fraction}, commit {cfg.commit[:12]}")
    key = load_key(cfg.key_file, create=True, forbid_under=[cfg.code_dir])
    run = run_checks(cfg, overrides=overrides, log=log)
    checks, argv = run["checks"], run["argv"]
    reasons = [f"{c}: {r}" for c in CHECKS for r in checks[c]["reasons"]] + [f"{c}: {checks[c]['status']}" for c in CHECKS if checks[c]["status"] not in ("PASS", "FAIL")]
    verdict = "PASS" if all(checks[c]["status"] == "PASS" for c in CHECKS) else "FAIL"
    tok_path = Path(cfg.resolved_token_path())
    stamp = utc_now().replace(":", "")
    if verdict == "PASS":
        opts = run["opts"]
        paths = {"bank_manifest": opts.manifest, "ranked_fit_logs": opts.ranked_logs, "p2_ckpt": opts.p2_ckpt, "s0_marker": opts.s0_marker}
        if opts.replacements_file:
            paths["replacements_file"] = opts.replacements_file
        data = {k: {"path": str(v), **fingerprint_file(v)} for k, v in sorted(paths.items())}
        code_files = {f: sha256_file(os.path.join(cfg.code_dir, f)) for f in BOUND_CODE_FILES}
        tsha = tree_digest(code_files)
        binding = {"profile": PROFILE_SYNTHETIC if overrides else PROFILE, "commit": cfg.commit, "tree_sha256": tsha, "argv_sha256": argv_sha256(argv), "argv": argv, "stage": cfg.stage,
                   "fraction": cfg.fraction, "trainer": "srttc_r1.py", "srttc_r1_sha256": code_files["srttc_r1.py"], "bank_manifest_sha256": data["bank_manifest"]["sha256"],
                   "code_files": code_files, "repo_code_path": cfg.repo_code_path, "suite_test_blobs": checks["G-COMMIT"]["details"].get("test_blobs", {}), "data": data,
                   "data_sha256": sha256_bytes(canonical_json(data))}
    else:
        binding = {"profile": PROFILE_SYNTHETIC if overrides else PROFILE, "commit": cfg.commit, "argv_sha256": argv_sha256(argv), "argv": argv, "stage": cfg.stage, "fraction": cfg.fraction}
    token = {"schema": GATE_SCHEMA, "kind": TOKEN_KIND, "verdict": verdict, "binding": binding, "checks": checks, "reasons": reasons,
             "gate_sha256": sha256_file(__file__), "created_utc": utc_now(), "key_id": key_id(key)}
    token["hmac_sha256"] = sign(token, key)
    fail_path = tok_path.parent / FAIL_NAME
    if verdict == "PASS":
        _demote(tok_path, stamp)
        _demote(fail_path, stamp)
        write_json(tok_path, token)
        log(f"PASS: wrote {tok_path}")
        return {"verdict": verdict, "token_path": str(tok_path), "token": token, "checks": checks, "reasons": []}
    _demote(tok_path, stamp)                                     # a stale PASS must not survive a failing run
    write_json(fail_path, token)
    log(f"FAIL: no token minted; report {fail_path}")
    for r in reasons:
        log(f"  - {r[:400]}")
    return {"verdict": verdict, "token_path": None, "token": token, "checks": checks, "reasons": reasons}


def verify_token(token_path, argv: list, *, code_dir: str, key: bytes, git_dir: str | None = DEFAULT_GIT_DIR, profile: str = PROFILE) -> tuple:
    """THE LAUNCHER'S SIDE.  Every field the token binds is RE-MEASURED now; any difference refuses.  -> (ok, reasons, info)."""
    reasons, info = [], {"token": str(token_path)}
    try:
        tok = read_json(token_path)
    except Exception as e:                                       # noqa: BLE001
        return False, [f"token unreadable: {type(e).__name__}: {str(e)[:160]}"], info
    if tok.get("kind") != TOKEN_KIND or tok.get("schema") != GATE_SCHEMA:
        reasons.append("not a launch-gate token")
    if not signature_ok(tok, key):
        reasons.append(f"HMAC does not verify: edited, or signed with another key (token key_id {tok.get('key_id')}, this host's {key_id(key)})")
    if tok.get("verdict") != "PASS":
        reasons.append(f"token verdict is {tok.get('verdict')!r}, not PASS")
    b = tok.get("binding") or {}
    info["binding"] = {k: b.get(k) for k in ("profile", "commit", "argv_sha256", "srttc_r1_sha256", "bank_manifest_sha256", "stage", "fraction")}
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
    if len(reasons) == n_before and tree_digest(now) != b.get("tree_sha256"):
        reasons.append("code tree digest differs from the gated tree")
    if b.get("srttc_r1_sha256") != now.get("srttc_r1.py"):
        reasons.append(f"srttc_r1.py sha256 {str(now.get('srttc_r1.py'))[:12]} != the token's {str(b.get('srttc_r1_sha256'))[:12]}")
    data = b.get("data") or {}
    if sha256_bytes(canonical_json(data)) != b.get("data_sha256"):
        reasons.append("data_sha256 does not match the bound data block")
    d_bad = []
    for k, want in sorted(data.items()):
        got = fingerprint_file(want.get("path", ""))
        if fp_key(got) != fp_key(want):
            d_bad.append(k)
    if d_bad:
        reasons.append(f"inputs differ from the gated inputs on {d_bad} (bank manifest / ranked logs / P2 checkpoint / S0 marker)")
    if (data.get("bank_manifest") or {}).get("sha256") != b.get("bank_manifest_sha256"):
        reasons.append("bank_manifest_sha256 is not the bound manifest's sha256")
    commit = b.get("commit")
    if git_dir is not None:
        try:
            if not git_is_commit(git_dir, str(commit)):
                reasons.append(f"commit {str(commit)[:12]} is not a commit object in {git_dir}")
            else:
                for f in BOUND_CODE_FILES:
                    blob = git_blob(git_dir, commit, f"{b.get('repo_code_path')}/{f}")
                    if blob is None or sha256_bytes(blob) != now.get(f):
                        reasons.append(f"{f} no longer equals its blob at {str(commit)[:12]}")
        except Exception as e:                                   # noqa: BLE001
            reasons.append(f"cannot re-check the commit blobs: {type(e).__name__}: {str(e)[:160]}")
    info["created_utc"] = tok.get("created_utc")
    return not reasons, reasons, info


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="srttc_r1_gate.py", description="SR-TTC R1 launch gate (mint / verify / argv). CPU only.")
    ap.add_argument("cmd", nargs="?", choices=("mint", "verify", "argv"), default="mint")
    ap.add_argument("--fraction", type=int, required=True, choices=REG_FRACTIONS)
    ap.add_argument("--stage", choices=("run", "lc"), default="run")
    ap.add_argument("--commit", default="", help="the landed commit (40-hex) the deployed code files must equal; required for mint")
    ap.add_argument("--python", default=DEFAULT_PYTHON)
    ap.add_argument("--code-dir", default=DEFAULT_CODE_DIR)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=REG_EPOCHS)
    ap.add_argument("--batch-sets", type=int, default=REG_BATCH_SETS)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--criterion", default=REG_CRITERION)
    ap.add_argument("--both-rule", default=REG_BOTH_RULE)
    ap.add_argument("--replacements-file", default=None)
    ap.add_argument("--git-dir", default=DEFAULT_GIT_DIR)
    ap.add_argument("--repo-code-path", default=DEFAULT_REPO_CODE_PATH)
    ap.add_argument("--token", default="", help="default: C:\\Users\\Admin\\srttc_r1\\gate\\f<fraction>_<stage>\\r1_gate_token.json")
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--suite-timeout-s", type=int, default=1800)
    return ap


def config_from_args(a: argparse.Namespace) -> GateConfig:
    return GateConfig(commit=a.commit, fraction=a.fraction, stage=a.stage, python=a.python, code_dir=a.code_dir, out_dir=a.out_dir, threads=a.threads, epochs=a.epochs,
                      batch_sets=a.batch_sets, seeds=a.seeds, criterion=a.criterion, both_rule=a.both_rule, replacements_file=a.replacements_file, git_dir=a.git_dir,
                      repo_code_path=a.repo_code_path, token_path=a.token, key_file=a.key_file, suite_timeout_s=a.suite_timeout_s)


def main(argv=None, *, _test_overrides: dict | None = None, _test_suite: dict | None = None, _out=print) -> int:
    """`_test_overrides` / `_test_suite` are API-only hooks (the command line cannot reach them): Options fields for a synthetic world, and the suite floor/test file for a fake export."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("mint", "verify", "argv", "-h", "--help"):
        argv = ["mint"] + argv
    a = make_parser().parse_args(argv)
    cfg = config_from_args(a)
    for k, v in (_test_suite or {}).items():
        setattr(cfg, k, v)
    if a.cmd == "argv":
        _out(json.dumps({"argv": cfg.argv()}))
        return 0
    if a.cmd == "verify":
        try:
            key = load_key(cfg.key_file, create=False)
            ok, reasons, info = verify_token(cfg.resolved_token_path(), cfg.argv(), code_dir=cfg.code_dir, key=key, git_dir=cfg.git_dir,
                                             profile=PROFILE_SYNTHETIC if _test_overrides else PROFILE)
        except Exception as e:                                   # noqa: BLE001 -- a refusal, never a pass
            ok, reasons, info = False, [f"{type(e).__name__}: {str(e)[:300]}"], {}
        _out(json.dumps({"ok": bool(ok), "reasons": reasons, "argv": cfg.argv(), "token": cfg.resolved_token_path(), "info": info}))
        return 0 if ok else 3
    if not re.fullmatch(r"[0-9a-f]{40}", cfg.commit or ""):
        _out("REFUSED: --commit must be the 40-hex landed commit")
        return 2
    res = mint(cfg, overrides=_test_overrides, log=_out)
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
