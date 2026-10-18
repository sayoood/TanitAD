#!/usr/bin/env python3
"""P6-HN-S: the DEV-BOX CHAIN of the screen.  (raw/2026-10-18-p6hn-prereg/PREREG_P6HN_SCREEN.md sections 4-5, 11, 13; SCREEN_NOTES.md checklist item 8; MM_DECISIONS_SCREEN.md "Integration still owed".)

    python refe/hnscreen_devbox.py run  --config chain_config.json --root <state dir> [--stop-after S2]
    python refe/hnscreen_devbox.py plan --config chain_config.json --root <state dir>        # stage order, DONE state, missing named parameters; writes nothing
    python refe/hnscreen_devbox.py skeleton --out chain_config.json                           # every named parameter, the unknown ones null

STAGE ORDER (SCREEN_NOTES.md "Order, with stop points"; each stage has a DONE marker, a HANDOFF, and is skipped on resume only if its outputs RE-HASH to the marker):
    S0  stage the inputs (a named command; the chain never touches the pod)
    S1  bank build, SY (hn_bank.py yield), I1, I2          ->  STOP: SCREEN-INVALID (a gate failed; checked FIRST, section 11 row 0) / SCREEN-SPARSE (SY < 0.10, row 1)
    S2  trunk-context cache (units, the LANDED lock taken by the driver per unit), I6, G15 (GPU, behind the lease)   ->  STOP: SCREEN-INVALID
    T<s>  mint the `refe_hnscreen` token of pass s: trainer --preflight (GPU), `launch_gate.py run`, `launch_gate.py verify`
    P<s>  the paired pass of seed s: `launch_gate.py verify` IMMEDIATELY before the trainer starts, then the trainer behind the lease, then the post-pass gates (I4 / I5 / D1)
    S4  the gates file, the offsets scoring (M3), `hn_screen_families.py --per-set-out`, `hn_screen_read.py --families-per-set`
    VERDICT  the first matching section-11 row; row None = rule E fired: seeds 3 and 4 are added (ONCE) and the chain re-plans
    NEXT  the verdict row's next lever: the configured commands for that row, and NEXT_LEVER.md (Rule Zero: a verdict is never the end of a turn)

THE FOUR GUARANTEES THIS FILE EXISTS FOR (each has a RED mutation in refe/test_hnscreen_chain.py):
  1. NO TRAINER WITHOUT A TOKEN.  `verify_pass_token` runs `launch_gate.py verify` on the EXACT argv list the trainer is about to receive (the same file, `--args-file`), after the lease is held and
     immediately before `Popen`; it needs rc 0 AND the `ZZGATE-MATCHZZ` marker.  No token / rc != 0 / no marker = the trainer is NOT started and the chain ends rc 5.
  2. THE GPU IS TAKEN ONLY THROUGH THE LANDED LOCK.  `eval/aux_navtest_forward.py` `gpu_lock` (wait_idle.py + the O_EXCL devbox_gpu.lock + the memory.used re-check), imported, not copied.
  3. ANOTHER JOB'S LOCK IS NEVER REMOVED.  The chain only YIELDS: while any lock file or the chain_active marker exists it polls and waits (bounded; then it ends rc 75 with a HANDOFF, resumable).
     ⚠ the landed `acquire_gpu_lock` itself removes a lock whose holder pid it cannot see once the file is > 600 s old, and the NavSim/battery tools write an MSYS pid Windows cannot see
     (`battery/code/gpu_lock.py`: "pid not alive is not evidence").  So the chain calls it ONLY when the lock file was absent a moment ago, and caps its wait_idle at 300 s: the lock it could then meet
     is younger than 300 s and the removal branch is unreachable.  `release` is the landed pid-matched one: it removes only a lock whose pid is this process's.
  4. RESUME BY CONTENT.  A DONE marker records the sha256 of every declared output (small files; the cache is bound by its own manifest) and a fingerprint of (config, stage spec, upstream markers).
     A stage is skipped only if the fingerprint is equal AND every output still hashes to the marker; anything upstream that re-ran invalidates what depends on it.

NAMED PARAMETERS, NOT GUESSES (AMBIGUITIES in RESULT.md): the commands that have no CLI yet (S0 staging, the bank build, I1/I2, the cache driver, I6/G15, the offsets scorer, the families inputs) and the profile's argv
order are CONFIG; the chain refuses (rc 4) before running anything if one is missing.  Children never see HF_TOKEN.  This file writes no pod address, port, id or key path anywhere.

EXIT CODES: 0 finished (verdict row 2-9 written, NEXT done) | 1 a command failed / crashed | 2 SCREEN-INVALID | 3 SCREEN-SPARSE | 4 config / missing named parameter / lever not exactly once
            5 token refused (trainer NOT started) | 6 low disk or RAM | 7 another chain holds this root | 75 yielded (GPU held by another job past max_wait_s, or STOP file): relaunch resumes.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

TOOL = "hnscreen_devbox/1"
HERE = os.path.dirname(os.path.abspath(__file__))
PKG_DEFAULT = os.path.dirname(HERE)
OWNER = "hnscreen_devbox"
CHAIN_MARKER = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active"
MATCH = "ZZGATE-MATCHZZ"
PREFLIGHT_PASS = "ZZSFT_PREFLIGHT_PASS"
CACHE_DONE, CACHE_PARTIAL = "ZZCTX_DONE", "ZZCTX_PARTIAL"
RULE = "navsim_v1_lane"
SEEDS, EXT_SEEDS = (0, 1, 2), (3, 4)
SCHEMA = "hnscreen_devbox_marker/1"
WAIT_IDLE_CAP_S = 300.0        # < 600: the landed acquire's stale-removal branch (dead holder, lock older than 600 s) can never be reached, see the module docstring

RC_OK, RC_FAILED, RC_INVALID, RC_SPARSE, RC_CONFIG, RC_REFUSED, RC_RESOURCE, RC_LOCKED, RC_YIELD = 0, 1, 2, 3, 4, 5, 6, 7, 75

# section 11 rows the CHAIN itself decides (rows 2-9 come from hn_screen_read's outcome)
ROW0 = ("SCREEN-INVALID", "not decided", "fix the failed control, re-run that stage; nothing is read")
ROW1 = ("SCREEN-SPARSE", "NO-GO (no supply)", "the draft's SPARSE lever: pair-contrast on the fan's own discordant twins, under its own prereg, on this rig")
GATE_IDS = ("I1", "I2", "I3", "I4", "I5", "I6", "D1", "G14", "G15", "launch_gate")
PER_PASS_GATES = ("I4", "I5", "D1")

DEFAULTS = {
    "python": None, "pkg": None,
    "gpu": {"lock": None, "chain_marker": CHAIN_MARKER, "wait_idle": "default", "idle_timeout_s": 240.0, "poll_s": 30.0, "max_wait_s": 6 * 3600.0, "settle_s": 4.0},
    "trainer": {"cmd": None, "argv_template": None, "seeds": list(SEEDS), "extension_seeds": list(EXT_SEEDS),
                "score_files": ["scores_update1000_A.npz", "scores_update1000_B.npz"], "attempt_artifacts": ["scores", "sft.jsonl", "sft.log"]},
    "gate": {"cmd": None, "profile": "refe_hnscreen", "key_file": None, "arm_name_fmt": None, "token_check_names": {"G14": "G14", "G15": "G15"}},
    "paths": {"bank_train": None, "bank_val": None, "cache_dir": None},
    "sy": {"partition_id": None, "fan_fingerprint": None, "label_version": 5},
    "commands": {"s0_stage": None, "s1_build": None, "s1_i1": None, "s1_i2": None, "s1_i3": None, "s2_cache": None, "s2_i6": None, "s2_g15": None,
                 "post_pass": [], "s4_offsets": None, "s4_prep": None, "next_lever": {}},
    "cache": {"budget_s": 1800, "max_rounds": 200},
    "s4": {"pdm_dir": None, "rl_val_scenes": None, "base_scores": None, "sets_dir": None, "c0_scores": None, "taniteval_dir": None},
    "guards": {"min_free_gb": 5.0, "min_free_ram_mb": 6000, "ram_wait_s": 600.0, "paths": []},
}
REQUIRED = ("trainer.argv_template", "paths.bank_train", "paths.bank_val", "paths.cache_dir", "commands.s0_stage", "commands.s1_build", "commands.s1_i1", "commands.s1_i2", "commands.s2_cache",
            "commands.s2_i6", "commands.s2_g15", "commands.s4_offsets", "commands.s4_prep", "s4.pdm_dir", "s4.rl_val_scenes", "s4.base_scores")


class ChainStop(Exception):
    """end the chain with `rc`; the reason goes into the HANDOFF"""

    def __init__(self, rc, reason, status="STOPPED"):
        super().__init__(reason)
        self.rc, self.reason, self.status = rc, reason, status


class Yielded(ChainStop):
    def __init__(self, reason):
        super().__init__(RC_YIELD, reason, "YIELDED")


class GpuBusy(RuntimeError):
    """the landed lock could not be taken right now (held, GPU not idle, memory.used too high after taking it)"""


class Replan(Exception):
    """rule E fired: the seed list grew"""


# ====================================================================================================================== small utilities
def now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def norm(p):
    return os.path.normpath(str(p)).replace("\\", "/")


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def atomic_write(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def jdump(obj, path):
    atomic_write(path, json.dumps(obj, indent=1, sort_keys=True, default=str))


def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ev_ok(path):
    """gate evidence: a json object whose `ok` is exactly True (a missing, torn or non-bool file is NOT ok)"""
    try:
        d = jload(path)
    except (OSError, ValueError):
        return False
    return isinstance(d, dict) and d.get("ok") is True


def _merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


_PH = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def subst(tokens, mapping):
    """replace {name} in every token; an unknown name is a CONFIG error, never left in an argv"""
    def rep(mo, t):
        k = mo.group(1)
        if k not in mapping:
            raise ChainStop(RC_CONFIG, f"unknown placeholder {{{k}}} in {t!r}", "CONFIG")
        return str(mapping[k])
    return [_PH.sub(lambda mo, t=str(t): rep(mo, t), str(t)) for t in tokens]


def pid_alive(pid):
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if sys.platform == "win32":
        r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True)      # NEVER os.kill(pid, 0) on Windows: it terminates the process
        return f'"{pid}"' in r.stdout.decode("utf-8", "replace")
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def free_gb(path):
    import shutil
    try:
        return shutil.disk_usage(path).free / 2 ** 30
    except OSError:
        return 0.0


def free_ram_mb():
    if sys.platform != "win32":
        try:
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // (1024 * 1024)
        except (ValueError, OSError, AttributeError):
            return 10 ** 9
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
    s = MS()
    s.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return int(s.ullAvailPhys // (1024 * 1024))


# ====================================================================================================================== config
class Cfg:
    def __init__(self, doc, text_sha):
        self.d = _merge(DEFAULTS, doc)
        self.sha = text_sha
        self.d["python"] = self.d["python"] or sys.executable
        self.d["pkg"] = norm(self.d["pkg"] or PKG_DEFAULT)
        py, pkg = self.d["python"], self.d["pkg"]
        if self.d["trainer"]["cmd"] is None:
            self.d["trainer"]["cmd"] = [py, f"{pkg}/refe/scorer_finetune.py"]
        if self.d["gate"]["cmd"] is None:
            self.d["gate"]["cmd"] = [py, f"{pkg}/refe/launch_gate.py"]

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            raw = f.read()
        return cls(json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest())

    def get(self, dotted, default=None):
        cur = self.d
        for k in dotted.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def missing(self):
        return [k for k in REQUIRED if self.get(k) in (None, "", [])]

    def redacted(self):
        d = json.loads(json.dumps(self.d))
        if d["gate"].get("key_file"):
            d["gate"]["key_file"] = "<set>"
        return d


# ====================================================================================================================== DONE markers (content-bound)
class Marks:
    def __init__(self, root):
        self.dir = os.path.join(root, "stage")

    def path(self, sid):
        return os.path.join(self.dir, f"{sid}.done.json")

    def read(self, sid):
        try:
            return jload(self.path(sid))
        except (OSError, ValueError):
            return None

    def digest(self, sid):
        p = self.path(sid)
        return sha256_file(p) if os.path.isfile(p) else None

    def is_done(self, sid, fp):
        m = self.read(sid)
        if not m or m.get("schema") != SCHEMA or m.get("status") != "DONE" or m.get("fp") != fp:
            return False
        for p, sha in m.get("outputs", {}).items():
            if not os.path.isfile(p) or sha256_file(p) != sha:                      # CONTENT, never presence
                return False
        return True

    def write(self, sid, fp, outputs, extra=None):
        outs = {}
        for p in outputs:
            if not os.path.isfile(p):
                raise ChainStop(RC_FAILED, f"stage {sid} did not produce its declared output {p}", "FAILED")
            outs[norm(p)] = sha256_file(p)
        jdump({"schema": SCHEMA, "stage": sid, "status": "DONE", "fp": fp, "utc": now_utc(), "outputs": outs, "extra": extra or {}}, self.path(sid))


# ====================================================================================================================== running commands
class Runner:
    """one child at a time, output appended to a log; returns (rc, the text this run appended).  HF_TOKEN is never passed on."""

    def __init__(self, env=None):
        self.env = env

    def run(self, argv, log, cwd=None):
        env = dict(os.environ if self.env is None else self.env)
        env.pop("HF_TOKEN", None)
        env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
        os.makedirs(os.path.dirname(os.path.abspath(log)), exist_ok=True)
        start = os.path.getsize(log) if os.path.exists(log) else 0
        with open(log, "ab") as f:
            f.write(f"\n--- {now_utc()} {' '.join(str(a) for a in argv[:3])} ...\n".encode("utf-8"))
            f.flush()
            rc = subprocess.call([str(a) for a in argv], cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
        with open(log, "rb") as f:
            f.seek(start)
            text = f.read(1 << 20).decode("utf-8", "replace")
        return rc, text


# ====================================================================================================================== the GPU door
@contextlib.contextmanager
def landed_gpu_lock(pkg, owner, **kw):
    """the dev-box GPU lock of eval/aux_navtest_forward.py (`gpu_lock`: wait_idle.py + the O_EXCL devbox_gpu.lock, memory.used re-checked), imported from the package, not copied; the owner label of the lock body is
    this tool's for the call.  An acquisition failure is re-raised as this module's GpuBusy; its release is the landed pid-matched one."""
    ev = os.path.join(pkg, "eval")
    if ev not in sys.path:
        sys.path.insert(0, ev)
    import aux_navtest_forward as ANF
    with contextlib.ExitStack() as st:
        old = ANF.OWNER
        ANF.OWNER = owner
        try:
            body = st.enter_context(ANF.gpu_lock(enabled=True, **kw))
        except ANF.GpuBusy as e:
            raise GpuBusy(str(e)) from e
        finally:
            ANF.OWNER = old
        yield body


def peek(path):
    try:
        with open(path, encoding="utf-8") as f:
            body = f.read(300)
        return f"{body.strip()[:200]} (age {time.time() - os.path.getmtime(path):.0f} s)"
    except OSError:
        return "(unreadable)"


class GpuDoor:
    def __init__(self, *, lock, marker, landed, landed_kw=None, poll_s=30.0, max_wait_s=6 * 3600.0, clock=time.time, sleep=time.sleep, log=print, stop=lambda: False):
        self.lock, self.marker, self.landed = lock, marker, landed
        self.landed_kw = dict(landed_kw or {})
        self.poll_s, self.max_wait_s, self.clock, self.sleep, self.log, self.stop = poll_s, max_wait_s, clock, sleep, log, stop

    def foreign(self):
        """what holds the GPU, or None.  ANY lock file and the chain_active marker count as another job's: this chain holds its own lock only inside lease(), and a leftover is reported, never reused or cleared."""
        for p, what in ((self.lock, "lock"), (self.marker, "chain_active")):
            if p and os.path.exists(p):
                return f"{what} {peek(p)}"
        return None

    def _wait_poll(self, deadline):
        if self.stop():
            raise Yielded("a STOP file asked the chain to end")
        if self.clock() >= deadline:
            raise Yielded(f"the GPU stayed with another job for {self.max_wait_s:.0f} s; relaunch the chain to continue")
        self.sleep(self.poll_s)

    def yield_until_free(self, deadline):
        n = 0
        while True:
            f = self.foreign()
            if f is None:
                return
            n += 1
            self.log(f"ZZHNCHAIN_YIELD poll {n}: {f} -- NOT removed")
            self._wait_poll(deadline)

    @contextlib.contextmanager
    def lease(self, job):
        deadline = self.clock() + self.max_wait_s
        with contextlib.ExitStack() as st:
            while True:
                self.yield_until_free(deadline)
                try:
                    st.enter_context(self.landed(owner=f"{OWNER}:{job}", **self.landed_kw))
                    break
                except GpuBusy as e:
                    self.log(f"ZZHNCHAIN_GPUBUSY {str(e)[:200]}")
                    self._wait_poll(deadline)
            self.log(f"ZZHNCHAIN_LEASE {job} acquired")
            yield
        self.log(f"ZZHNCHAIN_LEASE {job} released")


# ====================================================================================================================== the chain
class Chain:
    def __init__(self, cfg, root, *, door=None, runner=None, guard_gb=free_gb, guard_ram=free_ram_mb, clock=time.time, sleep=time.sleep, printer=print):
        self.cfg, self.root = cfg, norm(root)
        self.marks = Marks(self.root)
        self.runner = runner or Runner()
        self.guard_gb, self.guard_ram, self.clock, self.sleep, self.printer = guard_gb, guard_ram, clock, sleep, printer
        self.py, self.pkg = cfg.get("python"), cfg.get("pkg")
        self.dirs = {k: f"{self.root}/{k}" for k in ("s0", "s1", "s2", "s4", "gates", "passes", "handoffs", "logs")}
        self._door = door
        self.stage_order_run = []

    @property
    def door(self):
        if self._door is None:                                  # lazy: `plan` / `skeleton` never import the torch-heavy landed lock module
            self._door = self._default_door()
        return self._door

    # ------------------------------------------------------------------ paths
    def _default_door(self):
        g = self.cfg.get("gpu")
        lock, wait_idle = g.get("lock"), g.get("wait_idle")
        if lock is None or wait_idle == "default":
            if os.path.join(self.pkg, "eval") not in sys.path:
                sys.path.insert(0, os.path.join(self.pkg, "eval"))
            import aux_navtest_forward as ANF
            lock = lock or ANF.LOCK
            wait_idle = ANF.WAIT_IDLE if wait_idle == "default" else wait_idle
        kw = {"lock": lock, "wait_idle": wait_idle, "wait_timeout_s": min(float(g["idle_timeout_s"]), WAIT_IDLE_CAP_S), "settle_s": float(g["settle_s"])}
        return GpuDoor(lock=lock, marker=g.get("chain_marker"), landed=lambda **k: landed_gpu_lock(self.pkg, **k), landed_kw=kw, poll_s=float(g["poll_s"]),
                       max_wait_s=float(g["max_wait_s"]), clock=self.clock, sleep=self.sleep, log=self.log, stop=self.stop_requested)

    @property
    def chain_log(self):
        return f"{self.root}/chain.log"

    def log(self, msg):
        line = f"{now_utc()} {msg}"
        self.printer(line)
        os.makedirs(self.root, exist_ok=True)
        with open(self.chain_log, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def stop_requested(self):
        return os.path.exists(f"{self.root}/STOP")

    def gate_path(self, gid, seed=None):
        return f"{self.dirs['gates']}/{gid}.json" if seed is None else f"{self.dirs['gates']}/{gid}.s{seed}.json"

    def pass_dir(self, seed):
        return f"{self.dirs['passes']}/s{seed}"

    def pass_out(self, seed):
        return f"{self.pass_dir(seed)}/out"

    def pass_scores(self, seed):
        return f"{self.pass_out(seed)}/scores"

    @property
    def sy_path(self):
        return f"{self.dirs['s1']}/sy.json"

    @property
    def rule_e_path(self):
        return f"{self.root}/rule_e.json"

    def seeds(self):
        base = [int(s) for s in self.cfg.get("trainer.seeds")]
        return base + ([int(s) for s in self.cfg.get("trainer.extension_seeds")] if os.path.exists(self.rule_e_path) else [])

    def ph(self, **extra):
        m = {k: v for k, v in (self.cfg.get("paths") or {}).items() if isinstance(v, str)}
        m.update({"root": self.root, "pkg": self.pkg, "python": self.py, "gpu_lock": self.door.lock or "", "budget_s": self.cfg.get("cache.budget_s"),
                  "gates": self.dirs["gates"], "passes": self.dirs["passes"], "s0": self.dirs["s0"], "s1": self.dirs["s1"], "s2": self.dirs["s2"], "s4": self.dirs["s4"],
                  "sy": self.sy_path, "arms": ",".join(["base"] + [f"{a}{s}" for s in self.seeds() for a in ("A", "B")])})
        m.update(extra)
        return m

    def cmd(self, name, **extra):
        argv = self.cfg.get(f"commands.{name}")
        if not argv:
            raise ChainStop(RC_CONFIG, f"named parameter commands.{name} is not set", "CONFIG")
        return subst(argv, self.ph(**extra))

    # ------------------------------------------------------------------ the trainer argv (the ONE list that is token-verified AND launched)
    def pass_argv(self, seed):
        out, scores = self.pass_out(seed), self.pass_scores(seed)
        argv = subst(self.cfg.get("trainer.argv_template"), self.ph(seed=seed, out=out, scores=scores))
        lever_issues = self.lever_problems(argv, seed, out, scores)
        if lever_issues:
            raise ChainStop(RC_CONFIG, "the trainer argv is not the registered one: " + "; ".join(lever_issues), "CONFIG")
        return argv

    def lever_problems(self, argv, seed, out, scores):
        """PREREG 4.2: --seed, --out, --dump-scores, --hn-bank, --ctx-cache each EXACTLY once, carrying this pass's values (the profile fails any moved, added, dropped or doubled token too)"""
        want = {"--seed": str(seed), "--out": out, "--dump-scores": scores, "--hn-bank": self.cfg.get("paths.bank_train"), "--ctx-cache": self.cfg.get("paths.cache_dir")}
        bad = []
        for flag, val in want.items():
            n = argv.count(flag)
            if n != 1:
                bad.append(f"{flag} occurs {n} times (exactly once)")
                continue
            got = argv[argv.index(flag) + 1] if argv.index(flag) + 1 < len(argv) else None
            same = got == val if flag == "--seed" else (got is not None and norm(got) == norm(val))
            if not same:
                bad.append(f"{flag} carries {got!r}, this pass needs {val!r}")
        for flag in ("--epo", "--aux-drv"):
            if flag in argv:
                bad.append(f"{flag} with --hn-bank is refused (PREREG 4.4)")
        if "--lam" in argv and argv[argv.index("--lam") + 1] not in ("0", "0.0"):
            bad.append("--lam must be 0 (PREREG 4.4)")
        return bad

    # ------------------------------------------------------------------ running a command
    def guard(self, where):
        g = self.cfg.get("guards")
        for p in (g.get("paths") or []) + [self.root]:
            free = self.guard_gb(p)
            if free < float(g["min_free_gb"]):
                raise ChainStop(RC_RESOURCE, f"low disk before {where}: {p} has {free:.1f} GiB free < {g['min_free_gb']}", "STOPPED")
        t0 = self.clock()
        while self.guard_ram() < int(g["min_free_ram_mb"]):
            if self.clock() - t0 >= float(g["ram_wait_s"]):
                raise ChainStop(RC_RESOURCE, f"free RAM stayed below {g['min_free_ram_mb']} MiB for {g['ram_wait_s']} s before {where}", "STOPPED")
            self.sleep(15)

    def exec(self, argv, log_name, *, lock=None, where=None):
        """run one command; `lock` = a lease name (the chain holds the landed GPU lock around it) or None (CPU, or a driver that takes the lock itself per unit)"""
        where = where or log_name
        self.guard(where)
        log = f"{self.dirs['logs']}/{log_name}.log"
        self.log(f"ZZHNCHAIN_CMD {where}: {' '.join(str(a) for a in argv[:4])} ...")
        with (self.door.lease(lock) if lock else contextlib.nullcontext()):
            rc, text = self.runner.run(argv, log)
        return rc, text

    def must(self, argv, log_name, *, lock=None, marker=None):
        rc, text = self.exec(argv, log_name, lock=lock)
        if rc != 0:
            raise ChainStop(RC_FAILED, f"{log_name} exited {rc} (see logs/{log_name}.log)", "FAILED")
        if marker and marker not in text:
            raise ChainStop(RC_FAILED, f"{log_name} exited 0 but printed no {marker} (logs/{log_name}.log)", "FAILED")
        return text

    def gate_cmd(self, name, gid, seed=None, lock=None, **extra):
        """a command that must leave `gates/<gid>[.s<seed>].json`; the previous evidence is renamed away first, so a crash can never leave an old verdict standing"""
        ev = self.gate_path(gid, seed)
        if os.path.exists(ev):
            os.replace(ev, f"{ev}.prev{int(self.clock())}")
        os.makedirs(self.dirs["gates"], exist_ok=True)
        rc, _ = self.exec(self.cmd(name, **extra), f"{gid}{'' if seed is None else f'_s{seed}'}", lock=lock)
        if not os.path.isfile(ev):
            raise ChainStop(RC_FAILED, f"{name} (rc {rc}) wrote no evidence file for gate {gid}", "FAILED")
        return ev

    # ------------------------------------------------------------------ the token (guarantee 1)
    def token_file(self, seed):
        toks = sorted(glob.glob(f"{self.pass_dir(seed)}/lg/PASS_*.json"))
        return toks[-1] if toks else None

    def verify_pass_token(self, seed, where):
        """`launch_gate.py verify` on the argv list of THIS pass.  Refuses (rc 5) without a token, without rc 0, or without the MATCH marker."""
        tok = self.token_file(seed)
        if tok is None:
            raise ChainStop(RC_REFUSED, f"no PASS token for pass s{seed} ({where}): the trainer is NOT started", "REFUSED")
        gate = self.cfg.get("gate")
        argv_file = f"{self.pass_dir(seed)}/argv.json"
        jdump(self.pass_argv(seed), argv_file)
        cmd = gate["cmd"] + ["verify", "--profile", gate["profile"], "--token", tok, "--args-file", argv_file, "--pkg", self.pkg, "--json", f"{self.pass_dir(seed)}/token_verify.json"]
        if gate.get("key_file"):
            cmd += ["--key-file", gate["key_file"]]
        r = self.runner.run(cmd, f"{self.dirs['logs']}/token_verify_s{seed}.log")
        ok = r[0] == 0 and MATCH in r[1]
        if not ok:
            raise ChainStop(RC_REFUSED, f"the launch-gate token of pass s{seed} does not verify at {where} (rc {r[0]}, marker {'seen' if MATCH in r[1] else 'NOT seen'}): the trainer is NOT started", "REFUSED")
        self.log(f"ZZHNCHAIN_TOKEN_OK s{seed} {where} sha256 {sha256_file(tok)[:12]}")
        return tok, jload(argv_file)

    # ------------------------------------------------------------------ stage bodies: each returns (outputs, extra)
    def stage_S0(self):
        self.must(self.cmd("s0_stage"), "S0_stage")
        return [f"{self.dirs['s0']}/staged_manifest.json"], {}

    def stage_S1(self):
        os.makedirs(self.dirs["s1"], exist_ok=True)
        self.must(self.cmd("s1_build"), "S1_build")
        bank = self.cfg.get("paths.bank_train")
        sy = self.cfg.get("sy")
        argv = [self.py, f"{self.pkg}/refe/hn_bank.py", "yield", "--bank", bank, "--label-version", str(sy["label_version"]), "--out", self.sy_path]
        if sy.get("partition_id"):
            argv += ["--partition-id", sy["partition_id"]]
        if sy.get("fan_fingerprint"):
            argv += ["--fan-fingerprint", sy["fan_fingerprint"]]
        self.must(argv, "S1_sy")
        self.gate_cmd("s1_i1", "I1")
        self.gate_cmd("s1_i2", "I2")
        if self.cfg.get("commands.s1_i3"):
            self.gate_cmd("s1_i3", "I3")
        outs = [bank, self.cfg.get("paths.bank_val"), self.sy_path, self.gate_path("I1"), self.gate_path("I2")]
        return outs + ([self.gate_path("I3")] if self.cfg.get("commands.s1_i3") else []), {}

    def stage_S2(self):
        cache = self.cfg.get("paths.cache_dir")
        for rnd in range(1, int(self.cfg.get("cache.max_rounds")) + 1):
            if self.stop_requested():
                raise Yielded("a STOP file asked the chain to end between cache rounds")
            text = self.must(self.cmd("s2_cache"), "S2_cache", lock=None)             # the driver takes the landed lock itself, once per unit: the chain must NOT hold it
            if CACHE_DONE in text:
                break
            if CACHE_PARTIAL not in text:
                raise ChainStop(RC_FAILED, f"the cache driver printed neither {CACHE_DONE} nor {CACHE_PARTIAL} (logs/S2_cache.log)", "FAILED")
            self.log(f"ZZHNCHAIN_CACHE_ROUND {rnd} partial: the lock was released between units")
        else:
            raise ChainStop(RC_FAILED, f"the cache was still partial after {self.cfg.get('cache.max_rounds')} rounds", "FAILED")
        self.gate_cmd("s2_i6", "I6", lock="S2_gates")
        self.gate_cmd("s2_g15", "G15", lock="S2_gates")
        return [f"{cache}/manifest.json", self.gate_path("I6"), self.gate_path("G15")], {}

    def stage_T(self, seed):
        """mint the token of pass `seed`: the trainer's own --preflight (GPU), the gate run (consumes its preflight.jsonl), then the token is VERIFIED"""
        pd, out = self.pass_dir(seed), self.pass_out(seed)
        os.makedirs(out, exist_ok=True)
        argv = self.pass_argv(seed)
        jdump(argv, f"{pd}/argv.json")
        text = self.must(self.cfg.get("trainer.cmd") + argv + ["--preflight"], f"T{seed}_preflight", lock=f"T{seed}_preflight")
        if PREFLIGHT_PASS not in text:
            raise ChainStop(RC_FAILED, f"the trainer's --preflight printed no {PREFLIGHT_PASS} (logs/T{seed}_preflight.log)", "FAILED")
        lg = f"{pd}/lg"
        if os.path.isdir(lg) and os.listdir(lg):
            os.replace(lg, f"{lg}.attempt{int(self.clock())}")                           # evidence is archived, never deleted
        os.makedirs(lg, exist_ok=True)
        gate = self.cfg.get("gate")
        cmd = gate["cmd"] + ["run", "--profile", gate["profile"], "--pkg", self.pkg, "--args-file", f"{pd}/argv.json", "--out-dir", lg,
                             "--gate-preflight-json", f"{out}/preflight.jsonl"]
        if gate.get("arm_name_fmt"):
            cmd += ["--arm-name", gate["arm_name_fmt"].replace("{seed}", str(seed))]
        if gate.get("key_file"):
            cmd += ["--key-file", gate["key_file"]]
        rc, _ = self.exec(cmd, f"T{seed}_gate_run")
        tok = self.token_file(seed)
        if rc != 0 or tok is None:
            raise ChainStop(RC_REFUSED, f"the launch gate did not mint a PASS token for pass s{seed} (rc {rc}; logs/T{seed}_gate_run.log)", "REFUSED")
        self.verify_pass_token(seed, where="mint")
        return [tok, f"{pd}/argv.json"], {"token_sha256": sha256_file(tok)}

    def stage_P(self, seed):
        pd, out = self.pass_dir(seed), self.pass_out(seed)
        tr = self.cfg.get("trainer")
        for a in tr["attempt_artifacts"]:                                                # the trainer has no resume: an interrupted attempt is archived, never appended to
            p = f"{out}/{a}"
            if os.path.exists(p):
                os.replace(p, f"{p}.attempt{int(self.clock())}")
        with self.door.lease(f"P{seed}"):
            argv = self.verify_pass_token(seed, where="start")[1]                        # the trainer gets the list the gate just verified, read back from the file it verified
            rc, _ = self.exec(tr["cmd"] + argv, f"P{seed}_trainer")
            if rc != 0:
                raise ChainStop(RC_FAILED, f"the trainer of pass s{seed} exited {rc} (logs/P{seed}_trainer.log); the lock was released, a relaunch retries the pass", "FAILED")
            for item in self.cfg.get("commands.post_pass") or []:
                gid = item.get("gate")
                if gid:
                    ev = self.gate_path(gid, seed)
                    if os.path.exists(ev):
                        os.replace(ev, f"{ev}.prev{int(self.clock())}")
                rc, _ = self.exec(subst(item["argv"], self.ph(seed=seed, out=out, scores=self.pass_scores(seed))), f"P{seed}_{item['name']}")
                if gid and not os.path.isfile(self.gate_path(gid, seed)):
                    raise ChainStop(RC_FAILED, f"post-pass command {item['name']} (rc {rc}) wrote no evidence for gate {gid}", "FAILED")
        outs = [f"{self.pass_scores(seed)}/{f}" for f in tr["score_files"]]
        outs += [self.gate_path(i["gate"], seed) for i in self.cfg.get("commands.post_pass") or [] if i.get("gate")]
        return outs, {"token_verified": True, "token_sha256": sha256_file(self.token_file(seed))}

    # ------------------------------------------------------------------ the gates file
    def token_checks(self, seed, name):
        try:
            return jload(self.token_file(seed))["checks"][name]["status"] == "PASS"
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def assemble_gates(self, seeds):
        g, names = {}, self.cfg.get("gate.token_check_names")
        for gid in ("I1", "I2", "I3", "I6"):
            g[gid] = ev_ok(self.gate_path(gid))
        for gid in PER_PASS_GATES:
            g[gid] = all(ev_ok(self.gate_path(gid, s)) for s in seeds)
        g["G14"] = all(self.token_checks(s, names.get("G14", "G14")) for s in seeds)
        g["G15"] = ev_ok(self.gate_path("G15")) and all(self.token_checks(s, names.get("G15", "G15")) for s in seeds)
        g["launch_gate"] = all((self.marks.read(f"P{s}") or {}).get("extra", {}).get("token_verified") is True for s in seeds)
        return {k: bool(g[k]) for k in GATE_IDS}

    @property
    def per_set(self):
        return f"{self.dirs['s4']}/families_per_set.npz"

    def families_argv(self):
        s4 = self.cfg.get("s4")
        a = [self.py, f"{self.pkg}/eval/hn_screen_families.py", "--inputs", f"{self.dirs['s4']}/families_inputs.npz", "--picks", f"{self.dirs['s4']}/picks.npz",
             "--out", f"{self.dirs['s4']}/families.json", "--per-set-out", self.per_set]
        return a + (["--taniteval-dir", s4["taniteval_dir"]] if s4.get("taniteval_dir") else [])

    def read_argv(self, seeds):
        s4 = self.cfg.get("s4")
        scores = [f"base={s4['base_scores']}"] + [f"{a}{s}={self.pass_scores(s)}/{self.cfg.get('trainer.score_files')[i]}" for s in seeds for i, a in enumerate(("A", "B"))]
        if s4.get("c0_scores"):
            scores.append(f"C0={s4['c0_scores']}")
        scores9 = [f"{n}={self.dirs['s4']}/scores9_{n}.npz" for n in ["base"] + [f"{a}{s}" for s in seeds for a in ("A", "B")]]
        a = [self.py, f"{self.pkg}/eval/hn_screen_read.py", "--scores", *scores, "--pdm", s4["pdm_dir"], "--rl-val-scenes", s4["rl_val_scenes"], "--gates", f"{self.dirs['s4']}/gates.json",
             "--sy", self.sy_path, "--bank9", f"{self.dirs['s4']}/bank9.npz", "--scores9", *scores9, "--families", f"{self.dirs['s4']}/families.json",
             "--families-per-set", self.per_set, "--rule", RULE, "--out", f"{self.dirs['s4']}/read_result.json"]
        return a + (["--sets", s4["sets_dir"]] if s4.get("sets_dir") else [])

    def stage_S4(self, seeds):
        os.makedirs(self.dirs["s4"], exist_ok=True)
        gates = self.assemble_gates(seeds)
        jdump(gates, f"{self.dirs['s4']}/gates.json")
        bad = sorted(k for k, v in gates.items() if not v)
        if bad:                                                                             # section 5: any gate failure = SCREEN-INVALID, nothing is read
            raise ChainStop(RC_INVALID, f"gates not PASS before the read: {bad}", "INVALID")
        self.must(self.cmd("s4_offsets"), "S4_offsets", lock="S4_offsets")
        self.must(self.cmd("s4_prep"), "S4_prep")
        self.must(self.families_argv(), "S4_families", marker="ZZHNFAM DONE")
        self.must(self.read_argv(seeds), "S4_read")
        d = self.dirs["s4"]
        return [f"{d}/gates.json", f"{d}/families.json", self.per_set, f"{d}/read_result.json"], {"seeds": list(seeds)}

    def stage_VERDICT(self, seeds):
        res = jload(f"{self.dirs['s4']}/read_result.json")
        o = res.get("outcome") or {}
        if o.get("row") is None:                                                            # PENDING (rule E): the pre-committed extension, once
            if os.path.exists(self.rule_e_path):
                raise ChainStop(RC_INVALID, "rule E already fired and the read is still PENDING: the screen has no further extension", "INVALID")
            jdump({"fired_utc": now_utc(), "reason": o.get("reason"), "seeds_added": self.cfg.get("trainer.extension_seeds")}, self.rule_e_path)
            raise Replan()
        if int(o["row"]) == 0:
            raise ChainStop(RC_INVALID, f"hn_screen_read returned SCREEN-INVALID: {o.get('reason')}", "INVALID")
        out = {"row": int(o["row"]), "name": o.get("name"), "full_arm": o.get("full_arm"), "next_lever": o.get("next_lever"), "rig_drift": o.get("rig_drift"), "source": "hn_screen_read"}
        jdump(out, f"{self.root}/outcome.json")
        return [f"{self.root}/outcome.json"], {"row": out["row"]}

    def stage_NEXT(self):
        o = jload(f"{self.root}/outcome.json")
        ran = []
        for item in (self.cfg.get("commands.next_lever") or {}).get(str(o["row"]), []):
            self.must(subst(item["argv"], self.ph(row=o["row"])), f"NEXT_{item['name']}", lock=item.get("lock"))
            ran.append(item["name"])
        txt = (f"# NEXT LEVER (section 11 row {o['row']}: {o['name']})\n\nfull P6-HN arm: {o.get('full_arm')}\n\nnext lever, executed in the same run as the verdict: {o.get('next_lever')}\n\n"
               f"configured commands run by the chain: {ran or 'none configured for this row -- the lever above is for the next agent, from the HANDOFF'}\n")
        atomic_write(f"{self.root}/NEXT_LEVER.md", txt)
        return [f"{self.root}/NEXT_LEVER.md"], {"ran": ran}

    # ------------------------------------------------------------------ stage table, fingerprints, markers, handoffs
    def stage_list(self, seeds):
        return ["S0", "S1", "S2"] + [f"T{s}" for s in seeds] + [f"P{s}" for s in seeds] + ["S4", "VERDICT", "NEXT"]

    def deps(self, sid, seeds):
        if sid == "S0":
            return []
        if sid in ("S1", "S2"):
            return ["S0" if sid == "S1" else "S1"]
        if sid[0] == "T":
            return ["S2"]
        if sid[0] == "P":
            return ["T" + sid[1:]]
        if sid == "S4":
            return ["S1"] + [f"P{s}" for s in seeds]
        return {"VERDICT": ["S4"], "NEXT": ["VERDICT"]}[sid]

    def fingerprint(self, sid, seeds):
        d = {"tool": TOOL, "cfg": self.cfg.sha, "stage": sid, "deps": {x: self.marks.digest(x) for x in self.deps(sid, seeds)}}
        if sid in ("S4", "VERDICT"):
            d["seeds"] = list(seeds)
        return sha256_text(json.dumps(d, sort_keys=True))

    def body_of(self, sid, seeds):
        fixed = {"S0": self.stage_S0, "S1": self.stage_S1, "S2": self.stage_S2, "S4": lambda: self.stage_S4(seeds), "VERDICT": lambda: self.stage_VERDICT(seeds), "NEXT": self.stage_NEXT}
        if sid in fixed:
            return fixed[sid]
        return (lambda: self.stage_T(int(sid[1:]))) if sid[0] == "T" else (lambda: self.stage_P(int(sid[1:])))

    def run_stage(self, sid, seeds):
        body = self.body_of(sid, seeds)
        fp = self.fingerprint(sid, seeds)
        if self.marks.is_done(sid, fp):
            self.log(f"ZZHNCHAIN_SKIP {sid} (DONE marker, fingerprint equal, outputs re-hash)")
        else:
            self.log(f"ZZHNCHAIN_STAGE_START {sid}")
            self.stage_order_run.append(sid)
            self.status(sid, "RUNNING")
            outputs, extra = body()
            self.marks.write(sid, fp, outputs, extra)
            self.log(f"ZZHNCHAIN_STAGE_DONE {sid}")
            self.handoff(sid, "DONE", seeds=seeds)
        return self.stop_points(sid, seeds)

    def stop_points(self, sid, seeds):
        """evaluated after a stage whether it ran or was skipped, so a relaunch ends at the same place.  Gates first (row 0), then SY (row 1): the first matching row wins."""
        if sid == "S1":
            bad = [g for g in ("I1", "I2") if not ev_ok(self.gate_path(g))]
            if bad:
                return self.terminal(0, f"gate(s) {bad} failed or left no PASS evidence", sid)
            try:
                sparse = jload(self.sy_path).get("sparse")
            except (OSError, ValueError):
                sparse = None
            if sparse is None or not isinstance(sparse, bool):
                raise ChainStop(RC_FAILED, f"{self.sy_path} carries no boolean `sparse`", "FAILED")
            if sparse is True:
                return self.terminal(1, "SY (share of RL-train scenes with a hard negative) is below 0.10: no cache is built and no pass runs", sid)
        if sid == "S2":
            bad = [g for g in ("I6", "G15") if not ev_ok(self.gate_path(g))]
            if bad:
                return self.terminal(0, f"gate(s) {bad} failed: nothing is trained", sid)
        return None

    def write_outcome(self, row, reason):
        name, full_arm, lever = ROW0 if row == 0 else ROW1
        jdump({"row": row, "name": name, "full_arm": full_arm, "next_lever": lever, "reason": reason, "source": "hnscreen_devbox"}, f"{self.root}/outcome.json")
        atomic_write(f"{self.root}/NEXT_LEVER.md", f"# NEXT LEVER (section 11 row {row}: {name})\n\nreason: {reason}\n\nfull P6-HN arm: {full_arm}\n\nnext lever: {lever}\n")
        self.log(f"ZZHNCHAIN_OUTCOME row {row} {name}: {reason}")
        return lever

    def terminal(self, row, reason, sid):
        lever = self.write_outcome(row, reason)
        self.handoff(sid, "TERMINAL " + (ROW0 if row == 0 else ROW1)[0], reason=reason, next_step=lever)
        return RC_INVALID if row == 0 else RC_SPARSE

    def status(self, sid, state, **kw):
        jdump({"tool": TOOL, "stage": sid, "state": state, "utc": now_utc(), **kw}, f"{self.root}/STATUS.json")

    def handoff(self, sid, status, *, seeds=None, reason=None, next_step=None):
        seeds = seeds or self.seeds()
        order = self.stage_list(seeds)
        done = [s for s in order if (self.marks.read(s) or {}).get("status") == "DONE"]
        pending = [s for s in order if s not in done]
        m = self.marks.read(sid) or {}
        arts = [f"- {p} (sha256 {h[:12]})" for p, h in list(m.get("outputs", {}).items())[:12]]
        nxt = next_step or (f"stage {pending[0]}" if pending else "nothing: the chain is complete")
        resume = f"{self.py} {HERE.replace(chr(92), '/')}/hnscreen_devbox.py run --config <the SAME config file> --root {self.root}"
        txt = (f"# HANDOFF {sid}: {status}\n\nchain {TOOL}, {now_utc()}, state dir {self.root}\n\nDONE: {', '.join(done) or 'none'}\nPENDING: {', '.join(pending) or 'none'}\n"
               + (f"\nREASON: {reason}\n" if reason else "") + (f"\nARTIFACTS of {sid}:\n" + "\n".join(arts) + "\n" if arts else "")
               + f"\nNEXT: {nxt}\nRESUME (safe after any interruption; DONE stages are skipped by content): {resume}\n"
               f"STOP FILE: create {self.root}/STOP to end the chain between commands.  Never remove another job's GPU lock: the chain only yields.\n")
        atomic_write(f"{self.dirs['handoffs']}/HANDOFF_{sid}.md", txt[:6000])
        atomic_write(f"{self.root}/HANDOFF.md", txt[:6000])

    # ------------------------------------------------------------------ run
    def preflight(self):
        miss = self.cfg.missing()
        if miss:
            raise ChainStop(RC_CONFIG, f"named parameters not set: {miss}", "CONFIG")
        os.makedirs(self.root, exist_ok=True)
        sha_p = f"{self.root}/config.sha256"
        if os.path.exists(sha_p):
            if open(sha_p, encoding="utf-8").read().strip() != self.cfg.sha:
                raise ChainStop(RC_CONFIG, "this state dir was started with a different config: the argv, the data and the stage commands are frozen for the whole screen (a changed config is a new screen)", "CONFIG")
        else:
            atomic_write(sha_p, self.cfg.sha + "\n")
            jdump(self.cfg.redacted(), f"{self.root}/config.frozen.json")
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)
        for s in self.seeds() + [int(x) for x in self.cfg.get("trainer.extension_seeds")]:      # every argv must be registered BEFORE anything runs, including rule E's
            self.pass_argv(s)

    @contextlib.contextmanager
    def instance(self):
        p = f"{self.root}/chain.pid"
        os.makedirs(self.root, exist_ok=True)
        for attempt in (1, 2):
            try:
                fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                try:
                    other = int(open(p, encoding="utf-8").read().strip())
                except (OSError, ValueError):
                    other = -1
                if attempt == 2 or pid_alive(other):
                    raise ChainStop(RC_LOCKED, f"another chain (pid {other}) holds {p}", "REFUSED")
                os.replace(p, f"{p}.stale{int(self.clock())}")
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        try:
            yield
        finally:
            with contextlib.suppress(OSError):
                os.remove(p)

    def run(self, stop_after=None):
        try:
            with self.instance():
                self.preflight()
                for _ in range(3):
                    seeds = self.seeds()
                    try:
                        for sid in self.stage_list(seeds):
                            rc = self.run_stage(sid, seeds)
                            if rc is not None:
                                return rc
                            if stop_after == sid:
                                self.log(f"ZZHNCHAIN_STOP_AFTER {sid}")
                                return RC_OK
                    except Replan:
                        self.log("ZZHNCHAIN_RULE_E fired: seeds 3 and 4 are added (once) and the chain re-plans")
                        continue
                    o = jload(f"{self.root}/outcome.json")
                    self.log(f"ZZHNCHAIN_FINISHED row {o['row']} {o['name']}")
                    return RC_OK
                raise ChainStop(RC_INVALID, "the plan did not settle after rule E", "INVALID")
        except ChainStop as e:
            self.log(f"ZZHNCHAIN_{e.status} rc {e.rc}: {e.reason}")
            if e.rc == RC_INVALID:
                self.write_outcome(0, e.reason)
            self.status("-", e.status, rc=e.rc, reason=e.reason)
            with contextlib.suppress(OSError):
                self.handoff("STOPPED", f"{e.status} (rc {e.rc})", reason=e.reason)
            return e.rc
        except Exception as e:                                                              # noqa: BLE001 -- a crash is recorded and resumable, never silent
            self.log(f"ZZHNCHAIN_CRASHED {type(e).__name__}: {str(e)[:300]}")
            with contextlib.suppress(OSError):
                self.handoff("STOPPED", "CRASHED", reason=f"{type(e).__name__}: {str(e)[:300]}")
            return RC_FAILED


def skeleton():
    d = json.loads(json.dumps(DEFAULTS))
    d["trainer"]["argv_template"] = ["--ckpt", "<C1 argv, token for token>", "--seed", "{seed}", "--out", "{out}", "--dump-scores", "{scores}", "--hn-bank", "{bank_train}", "--ctx-cache", "{cache_dir}"]
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("run", "plan"):
        p = sub.add_parser(n)
        p.add_argument("--config", required=True)
        p.add_argument("--root", required=True)
        if n == "run":
            p.add_argument("--stop-after", default=None)
    p = sub.add_parser("skeleton")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "skeleton":
        jdump(skeleton(), a.out)
        print(f"wrote {a.out}: every key that is null and listed in REQUIRED must be set before `run`")
        return 0
    chain = Chain(Cfg.load(a.config), a.root)
    if a.cmd == "plan":
        seeds = chain.seeds()
        print("missing named parameters:", chain.cfg.missing() or "none")
        for sid in chain.stage_list(seeds):
            print(f"  {sid:8s} {'DONE' if chain.marks.is_done(sid, chain.fingerprint(sid, seeds)) else 'pending'}")
        return 0
    return chain.run(stop_after=a.stop_after)


if __name__ == "__main__":
    sys.exit(main())
