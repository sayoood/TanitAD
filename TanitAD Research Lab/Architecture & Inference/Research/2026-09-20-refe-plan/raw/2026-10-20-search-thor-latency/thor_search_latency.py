#!/usr/bin/env python3
"""SEARCH-THOR-LATENCY: per-token latency of Step B's P2HEAD CEM search on the deployment target (AMENDMENT_C2_DRAFT section 7, MM_RULING_C2AMEND DQ3 / DQ4).

WHAT IT TIMES, per token, at batch 1, on the scorer modules of P2-s0 (`model_final.pt`, sha `ea31db22...`) with the banked post-trunk `visual_ctx` rows of carve B:
  (a) P2's PICK-ONLY SCORING  = one set-scoring of the 64-proposal fan + aggregate + argmax.  NARROW BY CONSTRUCTION: the banked ctx is the trunk's OUTPUT, so the trunk (frames -> ctx) and the
      proposal generator are NOT in (a).  Every ratio below is therefore "search vs the scorer-only pick"; against a full-forward baseline the ratio is smaller.  The absolute `added_ms` is the safe quantity.
  (b) the full search, config C8 (`stepB_config.json`: K=5, M=64, 8 elites)  = 14 scorer calls / 896 rows per token.
  (c) the full search, config O4 (`stepB_config_O4.json`: K=5, M=256, 32 elites, widen 2.0) = 38 scorer calls / 3392 rows per token.
The search is Step B's own `stepB.search_token` -> `tts_cem.cem_search`, UNCHANGED (L1 pins verified), ONE search seed (the deployed configuration).  No pass / fail (DQ4): it reports ms and ratios.

TIMED REGION: from "fan and ctx on device" (`bind(n)` is OUTSIDE it) to "final pick returned", bracketed by a device synchronise.  It includes the scorer closure's H2D of the [K,20,3] set and D2H of the [K,6] logits,
and the host numpy parts (`composite`, `aggregate`, argsort).  It excludes: `bind`, the untimed reference runs, the identity comparison, JSON writing, the trunk, the proposal generator.
WARM-UP (stated, excluded): the first `--warmup` tokens (default 30) are run through all three arms once, discarded, and counted in no statistic.  They are the first tokens of the timed list; the 200 timed tokens are
the first 200 sorted tokens of carve B.  The untimed reference phase before the warm-up already ran every token twice per arm.
THE LABELS ARE NEVER READ: the bundle is opened lazily and only `token`, `traj_raw`, `pick`, `logits` are indexed (`sub`, `pdms`, `valid` are label arrays; indexing one raises LabelAccess).

CONTROLS (a RED control means the numbers are NOT to be quoted; the JSON is still written, exit code 6):
  identity     the timed path returns exactly the untimed reference's output on EVERY timed token (bit for bit: the search's raw / exec / theta / moved / history; (a)'s logits and pick)
  determinism  the untimed reference run twice agrees (a nondeterministic device would make `identity` unattributable)
  call_counts  scorer calls and rows per token equal the config's formula (K + 1 + elites calls; K*M + 64 + elites*64 rows), on every timed token
  n0_pick      `k_iters = 0` returns the pick bit for bit with exactly one scorer call
  timer_floor  the no-op timing floor (the same bracket around an empty callable) is far below what is measured
  sees_work    p50(a) < p50(b) < p50(c): the timer sees the extra calls
INFORMATIVE (no gate): a no-op-scorer search (isolates the host numpy cost), the per-call time vs the population size, the pick agreement with the banked pick, the parity of the base-fan logits with the banked logits.

REFUSALS: Thor busy (`preflight`), missing device fields (`check_device_fields`), unpinned code / config / checkpoint / ctx / bundle / carve (no flag overrides a pin), a read-only-command allowlist (never bare
`jetson_clocks`, which SETS clocks).  Thor memory probes (`free`, `tegrastats`, `mem_get_info`, VmRSS) are NEVER read: they lie there (TRAPS.md); the busy test uses GPU utilisation, the process table and the Thor GPU flock.

    python thor_search_latency.py preflight --target thor --expect-hostname thor6 --env-dir <dir>
    python thor_search_latency.py run --target thor --expect-hostname thor6 --env-dir <dir> --bundle <bundle.npz> --ctx <ctx.npy> --ckpt <model_final.pt> --refe-dir <dir with model.py> \
        --c8 <stepB_config.json> --o4 <stepB_config_O4.json> --autocast bf16 --out-dir <dir>
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = "tanitad.thor_search_latency/1"

# ------------------------------------------------------------------------------------------------------------------------------ the pins (no CLI flag overrides any of them)
PINS = {
    "ckpt_sha256": "ea31db2296210b6a35dbf32ad95f1813edfa259ab330002b3ca7564aac727e51",        # P2-s0 model_final.pt (PREREG_L2_STEPB.md PINS block)
    "c8_config_sha256": "299c6e63dd3016100149f88b6ca498529d79f8a5fbcfc6845753c8121d62c441",   # stepB_config.json
    "o4_config_sha256": "1ebd719617e8978a03cea0254edd6f531504c1d070792d066736130ceb929013",   # stepB_config_O4.json
    "carve_b_tokens_sha256": "0629811ff5d05879eb62abc5726005ccc4924f03539423003503a099ba29577c",  # stepB.SEALED_CARVE_SHA256 (checked by stepB.check_data_clause)
    "bundle_sha256": "580b6cc395aa9074c75eb7a62fe99e8b493abd9ea5671be1a5765d8ae4fd673f",       # carve-B bundle.npz (chain log ZZSTEPB_DATA_CLAUSE_OK, bundle=580b6cc395aa)
    "ctx_sha256": "c8c287cf6bdb6b1b91785d6251b1027897cc16d11d13857b524f1731a23bd03b",         # carve-B ctx.npy (ctx.json, 1,572,864,128 bytes)
}
STEPB_CODE_PINS = {   # LF sha256 of the SEALED Step B files this script stands on (PREREG_L2_STEPB.md PINS block); L1's four files are verified by stepB.verify_l1_code()
    "stepB.py": "74e60d3de6f16aa1bcb3a9503cba54d81bc6e79fa55e4fe6ecb9efc60e390ea8",
    "l2_scorer.py": "8584bb1e91a04f247e79552910651330d2e4dff1f3c7248bf8c8a0fdb97be286",
}
MODEL_PY_SHA256 = "e6011cd83c846dfbd9756301ae731f4b84541b7b8e07a139a2c2c2b663a8a46d"        # refe/model.py (LF), the file real_scorer executes
ARMS = ("a", "b", "c")
ARM_LABEL = {"a": "P2 pick-only scoring", "b": "search C8 (M=64)", "c": "search O4 (M=256)"}
BUNDLE_ALLOWED = ("token", "traj_raw", "pick", "logits")
BUNDLE_LABEL_KEYS = ("sub", "pdms", "valid")                                                  # label arrays of the bundle: NEVER indexed
GPU_UTIL_MAX_PCT = 5.0                                                                        # AMENDMENT section 7: no other GPU job (utilisation < 5 % before start)
GPU_UTIL_SAMPLES = 5
MAX_ENV_AGE_S = 1800
THOR_GPU_LOCK = "/home/nvidia/refcv7_post/thor_gpu.lock"                                      # flock; cleanup_thor_A.sh (2026-10-04) refuses while it is held
EXIT = {"ok": 0, "usage": 2, "thor_busy": 3, "device_fields": 4, "pin": 5, "controls_red": 6}

# the programme's own definitions of "a job is running": tools/fleet_probe.py JOB_RE / EXCLUDE_RE, plus supervisors, plus cleanup_thor_A.sh's "a python job of the user is alive"
JOB_RE = re.compile(r"(train[_a-z0-9]*\.py|scripts/train|vlm_[a-z_]+\.py|taniteval\.runner|refa_train|refb_train|refc_train|supervise_run|sup_[a-z0-9_]+\.sh|torchrun)", re.I)
EXCLUDE_RE = re.compile(r"(jupyter|ipykernel|/bin/ssh|sshd|fleet_probe)", re.I)
PY_RE = re.compile(r"(^|/)(python[0-9.]*|torchrun)$")


class Refusal(SystemExit):
    """a refusal carries its exit code; nothing here is caught and continued"""

    def __init__(self, code: int, msg: str):
        super().__init__(f"REFUSED[{code}]: {msg}")
        self.exit_code = code
        self.msg = msg


class LabelAccess(RuntimeError):
    """the bundle's label arrays (sub / pdms / valid) must never be read by a latency instrument"""


class ForbiddenCommand(RuntimeError):
    """a shell command outside the read-only allowlist"""


class DeviceFieldsMissing(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------------------------------------ hashing and the Step B / L1 code
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_lf(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def tokens_sha256(tokens) -> str:
    return hashlib.sha256("\n".join(str(t) for t in tokens).encode()).hexdigest()


def _first_dir(marker: str, cands) -> str | None:
    for c in cands:
        if c and os.path.exists(os.path.join(c, marker)):
            return os.path.abspath(c)
    return None


def locate_code() -> dict:
    """Step B's code dir (stepB.py) and L1's (tts_cem.py): env STEPB_CODE / L1_CODE, this directory (the flat layout used on Thor), the repo layout, the dev-box run copies"""
    rel_stepb = os.path.normpath(os.path.join(HERE, "..", "2026-10-19-l2-stepB", "code"))
    rel_l1 = os.path.normpath(os.path.join(HERE, "..", "..", "..", "2026-10-18-ep-rl-lit", "l1_stepA"))
    stepb = _first_dir("stepB.py", [os.environ.get("STEPB_CODE"), HERE, rel_stepb, "C:/Users/Admin/par18/l2stepb/code"])
    l1 = _first_dir("tts_cem.py", [os.environ.get("L1_CODE"), HERE, rel_l1, "C:/Users/Admin/par18/l1run/code"])
    return {"stepB_dir": stepb, "l1_dir": l1}


_SB = None


def stepb():
    """import the sealed Step B module (which imports L1's stepA / tts_cem unchanged) after pinning its files; lazy so the pure helpers stay importable"""
    global _SB
    if _SB is not None:
        return _SB
    d = locate_code()
    if not d["stepB_dir"] or not d["l1_dir"]:
        raise Refusal(EXIT["pin"], f"Step B / L1 code not found ({d}): set STEPB_CODE and L1_CODE (no silent fallback)")
    os.environ["L1_CODE"] = d["l1_dir"]
    verify_stepb_files(d["stepB_dir"])
    for p in (d["stepB_dir"], d["l1_dir"]):
        if p not in sys.path:
            sys.path.insert(0, p)
    import stepB as SB
    SB.verify_l1_code(d["l1_dir"])
    _SB = SB
    return SB


def verify_stepb_files(code_dir: str, pins: dict | None = None) -> dict:
    """the SEALED Step B files are the landed ones, byte for byte (LF).  A latency taken on edited search machinery certifies nothing."""
    pins = STEPB_CODE_PINS if pins is None else pins
    got, bad = {}, {}
    for fn, want in pins.items():
        p = os.path.join(code_dir, fn)
        got[fn] = sha256_lf(p) if os.path.exists(p) else None
        if got[fn] != want:                                                                    # @GUARD:CODEPIN
            bad[fn] = f"{str(got[fn])[:12]} != {want[:12]}"
    if bad:
        raise Refusal(EXIT["pin"], f"Step B code is not the sealed code: {bad}")
    return got


# ------------------------------------------------------------------------------------------------------------------------------ percentile maths (own implementation = numpy's 'linear')
def percentile(samples, q: float) -> float:
    x = sorted(float(v) for v in samples)
    if not x:
        raise ValueError("percentile of an empty sample")
    if not all(math.isfinite(v) for v in x):                                                  # @GUARD:FINITE
        raise ValueError("percentile of a non-finite sample")
    if not (0.0 <= q <= 100.0):
        raise ValueError(f"q={q} outside [0, 100]")
    pos = (len(x) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return x[lo] + (x[hi] - x[lo]) * (pos - lo)


def summarize(ms) -> dict:
    x = [float(v) for v in ms]
    return {"n": len(x), "mean": sum(x) / len(x), "p50": percentile(x, 50), "p95": percentile(x, 95), "p99": percentile(x, 99), "min": min(x), "max": max(x)}


def ratios(num: dict, den: dict) -> dict:
    """p50 / p95 / p99 of `num` over the same quantile of `den` (quantile-wise ratio, not a paired ratio)"""
    out = {}
    for k in ("p50", "p95", "p99"):
        if not den[k] > 0.0:                                                                   # @GUARD:RATIODEN
            raise ValueError(f"ratio denominator {k}={den[k]} is not positive")
        out[k] = num[k] / den[k]
    return out


# ------------------------------------------------------------------------------------------------------------------------------ read-only commands, device state, the Thor-busy preflight
READONLY_ALLOW = (
    ("nvpmodel", "-q"),
    ("jetson_clocks", "--show"),                                  # --show only: bare `jetson_clocks` SETS the clocks
    ("nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"),
    ("nvidia-smi", "--query-gpu=temperature.gpu,power.draw,utilization.gpu", "--format=csv,noheader"),
    ("nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"),
    ("ps", "-eo", "pid,ppid,user,etimes,args", "--no-headers"),
)


def run_readonly(argv, timeout: float = 15.0) -> tuple[int, str]:
    """the ONLY subprocess entry point: an argv not on the allowlist is refused; a missing binary returns (127, '')"""
    t = tuple(argv)
    if t not in READONLY_ALLOW:                                                                # @GUARD:READONLY
        raise ForbiddenCommand(f"{t!r} is not a read-only command on the allowlist")
    try:
        p = subprocess.run(list(t), capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout
    except (FileNotFoundError, PermissionError):
        return 127, ""
    except subprocess.TimeoutExpired:
        return 124, ""


def _rd(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return None


def read_env_capture(env_dir: str | None, name: str, now: float, max_age_s: float = MAX_ENV_AGE_S) -> dict | None:
    """a pre-captured read-only status (e.g. `sudo jetson_clocks --show > <env_dir>/jetson_clocks_show.txt`): used only when the live read is impossible; refused when stale"""
    if not env_dir:
        return None
    p = os.path.join(env_dir, name)
    if not os.path.isfile(p):
        return None
    age = now - os.path.getmtime(p)
    txt = _rd(p)
    if not txt:
        return None
    if age > max_age_s:   # @GUARD:STALE
        return {"text": None, "source": f"file:{p}", "stale_age_s": age}
    return {"text": txt, "source": f"file:{p}", "age_s": age, "sha256": sha256_file(p)}


def read_status(argv, capture_name: str, env_dir: str | None, now: float, runner=run_readonly) -> dict:
    rc, txt = runner(argv)
    if rc == 0 and txt.strip():
        return {"text": txt.strip(), "source": "live"}
    cap = read_env_capture(env_dir, capture_name, now)
    if cap and cap.get("text"):
        return cap
    return {"text": None, "source": "unavailable" if not cap else cap["source"], "live_rc": rc, **({"stale_age_s": cap["stale_age_s"]} if cap and "stale_age_s" in cap else {})}


def read_devfreq(root: str = "/sys/class/devfreq") -> dict:
    """GPU / EMC / bwmgr clocks from sysfs (the programme's own thor_clocks_experiment.sh snapshot); readable without root"""
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "*") + os.sep)):
        out[os.path.basename(f.rstrip("/\\"))] = {k: _rd(os.path.join(f, fn)) for k, fn in (("cur", "cur_freq"), ("min", "min_freq"), ("max", "max_freq"), ("gov", "governor"))}
    return out


def read_thermal(root: str = "/sys/devices/virtual/thermal") -> dict:
    out = {}
    for z in sorted(glob.glob(os.path.join(root, "thermal_zone*"))):
        t = _rd(os.path.join(z, "type"))
        v = _rd(os.path.join(z, "temp"))
        if t is not None and v is not None:
            out[t] = v
    return out


def parse_clocks(show_text: str | None) -> dict:
    """BEST-EFFORT parse of `jetson_clocks --show` (format UNVERIFIED on Thor; the raw text is the record)"""
    if not show_text:
        return {}
    m = re.search(r"GPU[^\n]*?MaxFreq=(\d+)[^\n]*?CurrentFreq=(\d+)", show_text)
    return {"gpu_max_hz": int(m.group(1)), "gpu_cur_hz": int(m.group(2)), "gpu_at_max": m.group(1) == m.group(2)} if m else {}


def device_state(target: str, torch_mod, env_dir: str | None, now: float, runner=run_readonly, sysfs=None) -> dict:
    """everything that identifies the device, its clocks and its power mode.  The values are recorded RAW; `check_device_fields` decides whether the record is complete."""
    sysfs = sysfs or {}
    env = {"schema": SCHEMA, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)), "target": target, "hostname": platform.node(), "platform": platform.platform(),
           "machine": platform.machine(), "cpu_name": platform.processor() or None, "python": sys.version.split()[0], "numpy": np.__version__,
           "torch": getattr(torch_mod, "__version__", None), "torch_cuda": getattr(getattr(torch_mod, "version", None), "cuda", None),
           "torch_threads": int(torch_mod.get_num_threads()) if torch_mod is not None else None, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if torch_mod is not None and torch_mod.cuda.is_available():
        env["cuda_device_name"] = torch_mod.cuda.get_device_name(0)
        env["cuda_capability"] = list(torch_mod.cuda.get_device_capability(0))
        env["cuda_matmul_tf32"] = bool(torch_mod.backends.cuda.matmul.allow_tf32)
        env["cudnn_tf32"] = bool(torch_mod.backends.cudnn.allow_tf32)
        env["float32_matmul_precision"] = torch_mod.get_float32_matmul_precision()
    if target == "thor":
        nv = read_status(("nvpmodel", "-q"), "nvpmodel_q.txt", env_dir, now, runner)
        jc = read_status(("jetson_clocks", "--show"), "jetson_clocks_show.txt", env_dir, now, runner)
        rc, smi = runner(("nvidia-smi", "--query-gpu=temperature.gpu,power.draw,utilization.gpu", "--format=csv,noheader"))
        env["nvpmodel_q"], env["nvpmodel_q_source"] = nv["text"], nv["source"]
        env["jetson_clocks_show"], env["jetson_clocks_show_source"] = jc["text"], jc["source"]
        env["jetson_clocks_parsed"] = parse_clocks(jc["text"])
        env["devfreq"] = read_devfreq(**sysfs.get("devfreq", {}))
        env["thermal"] = read_thermal(**sysfs.get("thermal", {}))
        env["smi_temp_power_util"] = smi.strip() if rc == 0 else None
    return env


def required_fields(target: str) -> tuple:
    base = ("hostname", "platform", "machine", "python", "numpy", "torch", "torch_threads")
    if target == "thor":
        return base + ("torch_cuda", "cuda_device_name", "cuda_capability", "nvpmodel_q", "jetson_clocks_show", "devfreq", "thermal")
    return base + ("cpu_name",)


def check_device_fields(env: dict, target: str) -> None:
    """REFUSE (DeviceFieldsMissing) when any required device / clock / power-mode field is absent or empty: a latency without its device, clocks and power mode is not quotable"""
    missing = [k for k in required_fields(target) if env.get(k) in (None, "", {}, [])]
    if missing:                                                                                # @GUARD:DEVFIELDS
        raise DeviceFieldsMissing(f"device fields missing for target {target!r}: {missing}")


def parse_ps(text: str) -> list[dict]:
    """`ps -eo pid,ppid,user,etimes,args --no-headers` -> rows"""
    rows = []
    for ln in text.splitlines():
        parts = ln.split(None, 4)
        if len(parts) < 5 or not parts[0].isdigit():
            continue
        try:
            rows.append({"pid": int(parts[0]), "ppid": int(parts[1]), "user": parts[2], "etimes": int(parts[3]), "args": parts[4]})
        except ValueError:
            continue
    return rows


def ancestors(rows: list[dict], pid: int) -> set:
    by = {r["pid"]: r for r in rows}
    out, cur = {pid}, pid
    while cur in by and by[cur]["ppid"] not in out and by[cur]["ppid"] > 0:
        cur = by[cur]["ppid"]
        out.add(cur)
    return out


def busy_processes(rows: list[dict], self_pid: int, me: str | None) -> list[dict]:
    """processes that mean the box is busy: a trainer / supervisor (fleet_probe JOB_RE), or any other python / torchrun of this user (cleanup_thor_A.sh).  Matched IN-PROCESS on parsed rows
    (never `pgrep -f`, which self-matches the caller's own command line); this process and its ancestors are excluded."""
    mine = ancestors(rows, self_pid)
    out = []
    for r in rows:
        if r["pid"] in mine or EXCLUDE_RE.search(r["args"]):
            continue
        exe = r["args"].split()[0]
        if JOB_RE.search(r["args"]) or (PY_RE.search(exe) and (me is None or r["user"] == me)):
            out.append({k: r[k] for k in ("pid", "user", "etimes", "args")})
    return out


def check_gpu_lock(path: str, hold: bool = True, flock_mod=None) -> dict:
    """the Thor GPU flock (`cleanup_thor_A.sh`): HELD -> an eval is running -> busy.  When free and `hold`, we keep it for the whole run so nothing else starts under our timing.
    An absent lock file is NOT evidence of idle (absence at one location): it is recorded and the other checks decide."""
    rec = {"path": path, "exists": os.path.exists(path), "held_by_other": False, "held_by_us": False, "checked": True}
    if not rec["exists"]:
        return rec
    try:
        fl = flock_mod
        if fl is None:
            import fcntl as fl
        fd = os.open(path, os.O_RDONLY)
    except (ImportError, OSError) as e:
        rec.update(checked=False, error=f"{type(e).__name__}: {e}")
        return rec
    try:
        fl.flock(fd, fl.LOCK_EX | fl.LOCK_NB)
    except (BlockingIOError, OSError):
        os.close(fd)
        rec["held_by_other"] = True
        return rec
    if hold:
        rec["held_by_us"] = True
        rec["_fd"] = fd
    else:
        fl.flock(fd, fl.LOCK_UN)
        os.close(fd)
    return rec


def parse_utils(text: str) -> list[float] | None:
    vals = []
    for ln in text.splitlines():
        s = ln.strip().rstrip("%").strip()
        if not s:
            continue
        try:
            vals.append(float(s))
        except ValueError:
            return None                                                                        # "[N/A]" or garbage: unreadable, which is NOT idle
    return vals or None


def preflight_thor(*, runner=run_readonly, sleep=time.sleep, self_pid: int | None = None, me: str | None = None, gpu_lock: str = THOR_GPU_LOCK, hold_lock: bool = True,
                   expect_hostname: str | None = None, hostname: str | None = None, flock_mod=None, samples: int = GPU_UTIL_SAMPLES) -> dict:
    """REFUSE-TO-START test: is anything else using this Thor?  Four independent reads, all must say idle; an unreadable instrument is a FAILED check, never a pass.
    Memory probes are not used (free / tegrastats / mem_get_info / VmRSS lie on Thor: TRAPS.md)."""
    reasons, ev = [], {}
    me = me or _me()                                                                           # the user's own python jobs count (cleanup_thor_A.sh); root's system python daemons do not
    host = hostname if hostname is not None else platform.node()
    ev["hostname"] = host
    if not expect_hostname:   # @GUARD:NOHOST
        reasons.append("no --expect-hostname given: a run is bound to a named box")
    elif host != expect_hostname:   # @GUARD:HOST
        reasons.append(f"hostname {host!r} != expected {expect_hostname!r}")
    lock = check_gpu_lock(gpu_lock, hold=hold_lock, flock_mod=flock_mod)
    ev["gpu_lock"] = {k: v for k, v in lock.items() if k != "_fd"}
    if lock.get("held_by_other"):   # @GUARD:LOCK
        reasons.append(f"GPU lock {gpu_lock} is HELD: an eval / job owns the GPU")
    if lock.get("checked") is False:
        reasons.append(f"GPU lock {gpu_lock} exists but could not be tested ({lock.get('error')})")
    rc, ps = runner(("ps", "-eo", "pid,ppid,user,etimes,args", "--no-headers"))
    rows = parse_ps(ps) if rc == 0 else []
    if rc != 0 or not rows:   # @GUARD:PSREAD
        reasons.append("process table unreadable (ps failed or empty): cannot establish that no trainer runs")
    busy = busy_processes(rows, self_pid if self_pid is not None else os.getpid(), me)
    ev["busy_processes"] = busy
    ev["n_processes_seen"] = len(rows)
    if busy:   # @GUARD:BUSY
        reasons.append(f"{len(busy)} busy process(es): " + "; ".join(f"pid {b['pid']} {b['args'][:80]}" for b in busy[:5]))
    utils = []
    for i in range(samples):
        rc, t = runner(("nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"))
        v = parse_utils(t) if rc == 0 else None
        if v is None:   # @GUARD:UTILREAD
            reasons.append("nvidia-smi utilisation unreadable: cannot establish that the GPU is idle")
            utils = None
            break
        utils.extend(v)
        if i + 1 < samples:
            sleep(1.0)
    ev["gpu_util_pct_samples"] = utils
    if utils is not None and max(utils) >= GPU_UTIL_MAX_PCT:                                   # @GUARD:UTIL
        reasons.append(f"GPU utilisation {max(utils)} % >= {GPU_UTIL_MAX_PCT} % over {samples} samples")
    rc, apps = runner(("nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"))
    app_lines = [ln for ln in apps.splitlines() if ln.strip()] if rc == 0 else None
    ev["compute_apps"] = app_lines
    if app_lines:   # @GUARD:APPS
        reasons.append(f"nvidia-smi lists {len(app_lines)} compute app(s): {app_lines[:3]}")
    ev["memory_probes"] = "NOT USED (Thor free / tegrastats / mem_get_info / VmRSS are not evidence: TRAPS.md)"
    return {"ok": not reasons, "reasons": reasons, "evidence": ev, "_lock": lock}


# ------------------------------------------------------------------------------------------------------------------------------ inputs: the carve-B fan, never a label
def load_fan(path: str, n: int, keys=BUNDLE_ALLOWED) -> dict:
    """lazily open the bundle and index ONLY the non-label arrays (`np.load` of an npz reads a key only when it is indexed)"""
    bad = [k for k in keys if k in BUNDLE_LABEL_KEYS]
    if bad:                                                                                    # @GUARD:NOLABEL
        raise LabelAccess(f"refusing to read label array(s) {bad} from {path!r}")
    z = np.load(path, allow_pickle=False)
    try:
        missing = [k for k in keys if k not in z.files]
        if missing:
            raise Refusal(EXIT["pin"], f"bundle {path!r} lacks {missing}")
        full_tokens = [str(t) for t in np.asarray(z["token"])]
        out = {"token_all": full_tokens}
        for k in keys:
            if k == "token":
                continue
            out[k] = np.asarray(z[k])[:n]
        out["token"] = full_tokens[:n]
    finally:
        z.close()
    return out


def check_tokens_sorted(toks) -> None:
    """'the first N sorted tokens of carve B' is only well defined if the bundle's own order is the sorted order"""
    if list(toks) != sorted(toks):                                                             # @GUARD:SORTED
        raise Refusal(EXIT["pin"], "the bundle's tokens are not sorted: 'the first N sorted tokens' would be ambiguous")


def make_sync(device: str, torch_mod):
    if str(device).startswith("cuda"):
        return torch_mod.cuda.synchronize
    return lambda: None


# ------------------------------------------------------------------------------------------------------------------------------ the timed wrapper and the arms
class CountingScorer:
    """wraps a score_fn: counts calls and rows and accumulates the wall time of each call (device-synchronise bracketed).  It returns the underlying function's output OBJECT unchanged and never touches its input."""

    def __init__(self, fn, clock_ns, sync):
        self.fn, self.clock_ns, self.sync = fn, clock_ns, sync
        self.reset()

    def reset(self):
        self.calls, self.rows, self.ns = 0, 0, 0

    def __call__(self, S):
        self.sync()
        t0 = self.clock_ns()
        out = self.fn(S)
        self.sync()
        self.ns += self.clock_ns() - t0
        self.calls += 1
        self.rows += int(np.shape(S)[0])
        return out


def pick_only(score_fn, fan, A) -> dict:
    """(a): P2's own pick = argmax of the scorer aggregate over the 64-proposal fan (the same expression Step B uses for `arm_argmax`)"""
    logits = np.asarray(score_fn(np.array(fan, copy=True)), dtype=np.float64)
    return {"logits": logits, "pick": int(np.argmax(A.aggregate(logits)))}   # @GUARD:PICKONLY


def expected_counts(cfg, n_fan: int) -> dict:
    """scorer calls and rows per token for one search, from the config alone: K population sets of M, the base fan once, and (if K > 0) one re-score of the fan per elite"""
    k = int(cfg.k_iters)
    elites = int(cfg.n_elite) if k > 0 else 0
    return {"calls": k + 1 + elites, "rows": k * int(cfg.m_pop) + n_fan + elites * n_fan}


def outputs_identical(x, y) -> bool:
    """exact, bit-for-bit equality of two arm outputs (arrays by bytes + dtype + shape, floats by ==, containers recursively)"""
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        x, y = np.asarray(x), np.asarray(y)
        return x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
    if isinstance(x, dict):
        return isinstance(y, dict) and x.keys() == y.keys() and all(outputs_identical(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return isinstance(y, (list, tuple)) and len(x) == len(y) and all(outputs_identical(p, q) for p, q in zip(x, y))
    return bool(x == y) and type(x) is type(y)


def time_one(run, counter, clock_ns, sync):
    """one timed execution of an arm: (output, total_ns, scorer_ns, calls, rows)"""
    counter.reset()
    sync()
    t0 = clock_ns()
    out = run(counter)
    sync()
    t1 = clock_ns()
    return out, t1 - t0, counter.ns, counter.calls, counter.rows


def noop_timer_floor(clock_ns, sync, reps: int = 5000) -> dict:
    """the same bracket as `time_one` around an EMPTY callable: the resolution and overhead of the instrument itself (ns)"""
    s = []
    f = lambda: None  # noqa: E731
    for _ in range(reps):
        sync()
        t0 = clock_ns()
        f()
        sync()
        s.append(clock_ns() - t0)
    return {"reps": reps, "ns": summarize(s)}


class StubScorer:
    """a no-op scorer (zeros): the search around it costs only the host numpy work"""

    def __call__(self, S):
        return np.zeros((int(np.shape(S)[0]), 6), np.float64)


def measure(score_fn, bind, fans, picks, cfg_b, cfg_c, *, seed: int, warmup: int, repeats: int, clock_ns=time.perf_counter_ns, sync=lambda: None, wrap=CountingScorer, SB=None,
            progress=None, banked_logits=None, n_parity: int = 3) -> dict:
    """the whole measurement on `n = len(fans)` tokens.  Phases: (1) UNTIMED reference runs of every arm, twice; (2) warm-up (discarded); (3) `repeats` timed passes, arms rotated per token;
    (4) the no-op floor, the no-op-scorer search, the n0 control.  Returns raw samples and the controls; `report()` turns it into the JSON."""
    SB = SB or stepb()
    A = SB.A
    n = len(fans)
    n_fan = int(np.shape(fans[0])[0])
    cfgs = {"b": cfg_b, "c": cfg_c}

    def runner_for(arm, i):
        if arm == "a":
            return lambda sf: pick_only(sf, fans[i], A)
        return lambda sf: SB.search_token(sf, fans[i], int(picks[i]), int(seed), cfgs[arm])

    # (1) untimed references, bare score_fn.  Twice, to read the device's own determinism floor.
    ref1, ref2 = {a: [] for a in ARMS}, {a: [] for a in ARMS}
    for i in range(n):
        bind(i)
        for a in ARMS:
            ref1[a].append(runner_for(a, i)(score_fn))
            ref2[a].append(runner_for(a, i)(score_fn))
    nondet = {a: [i for i in range(n) if not outputs_identical(ref1[a][i], ref2[a][i])] for a in ARMS}
    if progress:
        progress("REF_DONE", n)

    # (2) warm-up: all three arms, discarded
    counter = wrap(score_fn, clock_ns, sync)
    n_warm = min(int(warmup), n)   # @GUARD:WARMUP
    for i in range(n_warm):
        bind(i)
        for a in ARMS:
            time_one(runner_for(a, i), counter, clock_ns, sync)
    if progress:
        progress("WARMUP_DONE", n_warm)

    # (3) timed passes
    lat = {a: [[0.0] * n for _ in range(repeats)] for a in ARMS}
    scorer_ms = {a: [[0.0] * n for _ in range(repeats)] for a in ARMS}
    calls = {a: [] for a in ARMS}
    rows = {a: [] for a in ARMS}
    mism = {a: [] for a in ARMS}
    order = list(ARMS)
    for r in range(repeats):
        for i in range(n):
            bind(i)
            k = (i + r) % len(order)   # @GUARD:ROTATE
            for a in order[k:] + order[:k]:
                out, tot_ns, sc_ns, c, rw = time_one(runner_for(a, i), counter, clock_ns, sync)
                lat[a][r][i], scorer_ms[a][r][i] = tot_ns / 1e6, sc_ns / 1e6
                calls[a].append(c)
                rows[a].append(rw)
                same = outputs_identical(ref1[a][i], out)                                      # @GUARD:IDENTITY
                if not same:
                    mism[a].append({"repeat": r, "token_index": i})
        if progress:
            progress("PASS_DONE", r)

    # (4) floors and side controls
    floor = noop_timer_floor(clock_ns, sync)
    stub = wrap(StubScorer(), clock_ns, sync)
    noop_search = {}
    for a in ("b", "c"):
        s_ms, s_calls = [], []
        for i in range(n):
            _, tot, _, c, _ = time_one(lambda sf, i=i, a=a: SB.search_token(sf, fans[i], int(picks[i]), int(seed), cfgs[a]), stub, clock_ns, sync)
            s_ms.append(tot / 1e6)
            s_calls.append(c)
        noop_search[a] = {"ms": summarize(s_ms), "calls_per_token": sorted(set(s_calls))}
    cfg0 = dataclasses.replace(cfg_b, k_iters=0)
    n0 = []
    for i in range(min(5, n)):
        bind(i)
        out, _, _, c, _ = time_one(lambda sf, i=i: SB.search_token(sf, fans[i], int(picks[i]), int(seed), cfg0), counter, clock_ns, sync)
        pick_raw = np.asarray(fans[i][int(picks[i])], np.float32)
        n0.append(bool(   # @GUARD:N0
            out["moved"] is False and out["raw"].tobytes() == pick_raw.tobytes() and out["exec"].tobytes() == A.repair_last_heading_np(pick_raw).tobytes() and c == 1))

    # sensitivity: per-call time vs population size (informative: a launch-bound device may be flat)
    bind(0)
    scaling = {}
    for m in (64, 256):
        S = np.array(np.repeat(np.asarray(fans[0], np.float32)[:1], m, axis=0), copy=True) if m != n_fan else np.array(fans[0], copy=True)
        ts = []
        for _ in range(15):
            _, tot, _, _, _ = time_one(lambda sf, S=S: sf(S), counter, clock_ns, sync)
            ts.append(tot / 1e6)
        scaling[str(m)] = summarize(ts[5:])

    parity = None
    if banked_logits is not None and n_parity > 0:
        d = []
        for i in range(min(n_parity, n)):
            d.append(float(np.max(np.abs(np.asarray(ref1["a"][i]["logits"], np.float64) - np.asarray(banked_logits[i], np.float64)))))
        parity = {"tokens": len(d), "max_abs_logit_diff_vs_banked": max(d), "per_token": d, "note": "INFORMATIVE: banked logits come from a different device / precision; no tolerance is gated"}

    return {"n": n, "n_fan": n_fan, "warmup_tokens": n_warm, "repeats": repeats, "seed": int(seed), "lat_ms": lat, "scorer_ms": scorer_ms, "calls": calls, "rows": rows, "mismatches": mism,
            "nondeterministic_reference_tokens": nondet, "noop_floor": floor, "noop_search": noop_search, "n0": n0, "scaling_per_call_ms": scaling, "parity": parity,
            "pick_agreement": {a: float(np.mean([ref1[a][i]["pick"] == int(picks[i]) for i in range(n)])) for a in ("a",)},
            "moved_fraction": {a: float(np.mean([bool(ref1[a][i]["moved"]) for i in range(n)])) for a in ("b", "c")},
            "expected": {"a": {"calls": 1, "rows": n_fan}, "b": expected_counts(cfg_b, n_fan), "c": expected_counts(cfg_c, n_fan)}}


def flatten(per_repeat) -> list[float]:
    return [v for rep in per_repeat for v in rep]


def report(m: dict, cfg_b, cfg_c) -> dict:
    """raw samples -> p50 / p95 / p99 per arm, the ratios to (a), the stage split and the controls"""
    arms = {}
    for a in ARMS:
        tot = flatten(m["lat_ms"][a])
        sc = flatten(m["scorer_ms"][a])
        cpu = [t - s for t, s in zip(tot, sc)]
        arms[a] = {"label": ARM_LABEL[a], "ms": summarize(tot), "per_repeat_ms": [summarize(rep) for rep in m["lat_ms"][a]], "scorer_calls_ms": summarize(sc), "host_rest_ms": summarize(cpu),
                   "scorer_calls_per_token": sorted(set(m["calls"][a])), "scorer_rows_per_token": sorted(set(m["rows"][a])), "expected": m["expected"][a], "samples_ms": m["lat_ms"][a]}
    a_s = arms["a"]["ms"]
    for a in ("b", "c"):
        arms[a]["ratio_to_a"] = ratios(arms[a]["ms"], a_s)
        arms[a]["added_ms_vs_a"] = {k: arms[a]["ms"][k] - a_s[k] for k in ("p50", "p95", "p99")}
    arms["c"]["ratio_to_b"] = ratios(arms["c"]["ms"], arms["b"]["ms"])
    ctl = {}
    for a in ARMS:
        ctl[f"identity_{a}"] = {"ok": len(m["mismatches"][a]) == 0, "mismatched_samples": len(m["mismatches"][a]), "of": m["n"] * m["repeats"], "first": m["mismatches"][a][:3]}
        ctl[f"determinism_{a}"] = {"ok": len(m["nondeterministic_reference_tokens"][a]) == 0, "nondeterministic_tokens": len(m["nondeterministic_reference_tokens"][a])}   # @GUARD:DETERMINISM
        exp = m["expected"][a]
        bad = sum(1 for c, r in zip(m["calls"][a], m["rows"][a]) if c != exp["calls"] or r != exp["rows"])   # @GUARD:COUNTS
        ctl[f"call_counts_{a}"] = {"ok": bad == 0, "samples_off": bad, "expected": exp, "observed_calls": sorted(set(m["calls"][a])), "observed_rows": sorted(set(m["rows"][a]))}
    ctl["n0_pick"] = {"ok": bool(m["n0"]) and all(m["n0"]), "tokens": len(m["n0"])}
    fl = m["noop_floor"]["ns"]["p99"] / 1e6
    ctl["timer_floor"] = {"ok": fl <= 0.05 * a_s["p50"],  # @GUARD:FLOOR
                          "floor_p99_ms": fl, "a_p50_ms": a_s["p50"], "rule": "floor p99 <= 5 % of p50(a)"}
    ctl["sees_work"] = {"ok": bool(a_s["p50"] < arms["b"]["ms"]["p50"] < arms["c"]["ms"]["p50"]),  # @GUARD:SEESWORK
                        "p50_a_b_c_ms": [a_s["p50"], arms["b"]["ms"]["p50"], arms["c"]["ms"]["p50"]]}
    ctl_ok = all(v["ok"] for v in ctl.values())   # @GUARD:CONJ
    sc = m["scaling_per_call_ms"]
    rep = {"arms": arms, "controls": ctl, "controls_ok": ctl_ok,
           "informative": {"noop_scorer_search_ms": m["noop_search"], "noop_timer_floor_ns": m["noop_floor"], "per_call_ms_by_population": sc,
                           "row_scaling_visible_256_vs_64": sc["256"]["p50"] / sc["64"]["p50"] if sc["64"]["p50"] > 0 else None, "pick_agreement_with_banked_pick": m["pick_agreement"],
                           "search_moved_fraction": m["moved_fraction"], "scorer_parity_vs_banked_logits": m["parity"]}}
    return rep


STATEMENTS = (
    "NO PASS / FAIL (MM_RULING_C2AMEND DQ4): this reports ms per token and ratios. An absolute latency budget is an open architecture decision; until it exists there is NO claim of deployability.",
    "Evidence class: MEASURED on the device named in `device`. A record whose target is not `thor` is NOT a Thor number.",
    "(a) is the scorer-only pick (one set-scoring + aggregate + argmax) on the banked post-trunk ctx; the trunk and the proposal generator are NOT included. ratio_to_a is vs that narrow baseline; added_ms is the safe quantity.",
    "ONE search seed (the first of the config's three) = the deployed configuration. Tokens: the first N sorted tokens of carve B; the fan and ctx are banked inputs; no label array is read.",
    "Warm-up tokens are excluded from every statistic; the untimed reference phase (two runs of every arm on every token) precedes the warm-up.",
    "Excluded from timing: bind(n) (ctx H2D), reference runs, identity comparison, JSON writing, trunk, proposal generation. Included: set H2D, forward, logits D2H, host numpy (composite, aggregate, argsort).",
    "Python GC is left enabled (part of the deployed cost); the percentiles are over repeats x tokens samples (linear interpolation).",
)


def build_record(*, target, env, pins_seen, inputs, cfgs, rep, args_echo, preflight, preflight_end, thermal_end) -> dict:
    rec = {"schema": SCHEMA, "no_pass_fail": True, "verdict": None, "is_thor_number": bool(target == "thor" and rep["controls_ok"] and (preflight or {}).get("ok")),
           "evidence_class": "MEASURED (ours, this device)" if rep["controls_ok"] else "CONTROLS RED: not quotable", "statements": list(STATEMENTS), "device": env, "args": args_echo,
           "pins_seen": pins_seen, "inputs": inputs, "configs": cfgs, "preflight": preflight, "preflight_end": preflight_end, "thermal_end": thermal_end}
    rec.update(rep)
    return rec


def summary_lines(rec: dict) -> list[str]:
    out = []
    for a in ARMS:
        s = rec["arms"][a]
        ms = s["ms"]
        line = f"{a} {s['label']:<24} p50 {ms['p50']:9.3f}  p95 {ms['p95']:9.3f}  p99 {ms['p99']:9.3f} ms  (n={ms['n']}, calls/token {s['scorer_calls_per_token']}, rows/token {s['scorer_rows_per_token']})"
        if "ratio_to_a" in s:
            r = s["ratio_to_a"]
            line += f"  ratio to (a) p50/p95/p99 {r['p50']:.2f}/{r['p95']:.2f}/{r['p99']:.2f}"
        out.append(line)
    out.append(f"controls_ok {rec['controls_ok']}  device {rec['device'].get('cuda_device_name') or rec['device'].get('cpu_name')}  target {rec['device']['target']}  is_thor_number {rec['is_thor_number']}")
    return out


# ------------------------------------------------------------------------------------------------------------------------------ CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="thor_search_latency.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("preflight", "run"):
        p = sub.add_parser(name)
        p.add_argument("--target", required=True, choices=("thor", "devbox-cpu"))
        p.add_argument("--expect-hostname", default=None)
        p.add_argument("--env-dir", default=None, help="dir with pre-captured read-only status files (nvpmodel_q.txt, jetson_clocks_show.txt), used only when the live read is impossible")
        p.add_argument("--gpu-lock", default=THOR_GPU_LOCK)
    r = sub.choices["run"]
    r.add_argument("--bundle", required=True)
    r.add_argument("--ctx", required=True)
    r.add_argument("--ckpt", required=True)
    r.add_argument("--refe-dir", required=True)
    r.add_argument("--c8", required=True)
    r.add_argument("--o4", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--autocast", required=True, choices=("none", "bf16"))
    r.add_argument("--n-tokens", type=int, default=200)
    r.add_argument("--warmup", type=int, default=30)
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--seed-index", type=int, default=0)
    return ap


def _emit(*a):
    print(*a, flush=True)


def _me() -> str | None:
    """the invoking user (a non-login shell may not set $USER): the user's own python jobs count as 'busy'"""
    import getpass
    try:
        return os.environ.get("USER") or os.environ.get("USERNAME") or getpass.getuser()
    except Exception:
        return None


def apply_end_scan(rep: dict, ps_text: str, ps_rc: int, self_pid: int, me: str | None = None) -> dict:
    """after the timed passes: any trainer / eval / python job of this user that appeared DURING the run contaminates the timing -> a RED control (the preflight cannot see the future)"""
    rows = parse_ps(ps_text) if ps_rc == 0 else []
    foreign = busy_processes(rows, self_pid, me or _me())
    if foreign:                                                                                # @GUARD:ENDSCAN
        rep["controls"]["no_foreign_job_at_end"] = {"ok": False, "processes": foreign}
        rep["controls_ok"] = False
    return {"foreign_busy_processes_at_end": foreign, "ps_readable": bool(rows)}


def check_n_tokens(n: int, total: int) -> None:
    if not (1 <= int(n) <= int(total)):                                                        # @GUARD:NTOK
        raise Refusal(EXIT["usage"], f"--n-tokens {n} outside 1..{total}")


def finalize(rec: dict, out_dir: str, stamp: str) -> int:
    """write the record (ALWAYS, also when a control is RED), print the summary, return the exit code: 0 only if every control is green"""
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"THOR_SEARCH_LATENCY_{stamp}.json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
    for ln in summary_lines(rec):
        _emit("ZZTHORLAT_SUMMARY", ln)
    _emit("ZZTHORLAT_RESULT", out)
    _emit("ZZTHORLAT_CONTROLS", "OK" if rec["controls_ok"] else "RED " + str([k for k, v in rec["controls"].items() if not v["ok"]]))
    return EXIT["ok"] if rec["controls_ok"] else EXIT["controls_red"]                          # @GUARD:EXITCODE


def _check_pin(name: str, got: str, want: str):
    if got != want:                                                                            # @GUARD:PIN
        raise Refusal(EXIT["pin"], f"{name} sha256 {got[:12]} != pinned {want[:12]} (no flag overrides a pin)")


def cmd_preflight(a, *, runner=run_readonly, torch_mod=None, **kw) -> int:
    if a.target != "thor":
        _emit("ZZTHORLAT_PREFLIGHT_NA target devbox-cpu: nothing shared to protect; the record will say NOT A THOR NUMBER")
        return EXIT["ok"]
    pf = preflight_thor(runner=runner, expect_hostname=a.expect_hostname, gpu_lock=a.gpu_lock, hold_lock=False, **kw)
    pf.pop("_lock", None)
    _emit("ZZTHORLAT_PREFLIGHT", json.dumps({"ok": pf["ok"], "reasons": pf["reasons"], "evidence": pf["evidence"]}))
    if not pf["ok"]:
        _emit("ZZTHORLAT_STOP THOR_BUSY_OR_UNVERIFIED")
        return EXIT["thor_busy"]
    _emit("ZZTHORLAT_PREFLIGHT_OK")
    return EXIT["ok"]


def cmd_run(a, *, runner=run_readonly, **kw) -> int:
    if a.target == "thor" and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        raise Refusal(EXIT["usage"], "target thor with CUDA_VISIBLE_DEVICES=-1")
    if a.target == "devbox-cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise Refusal(EXIT["usage"], "target devbox-cpu is CPU only: set CUDA_VISIBLE_DEVICES=-1")
    now = time.time()
    # 1. busy-test BEFORE anything is loaded or the CUDA context is created
    pf = None
    if a.target == "thor":
        pf = preflight_thor(runner=runner, expect_hostname=a.expect_hostname, gpu_lock=a.gpu_lock, hold_lock=True, **kw)
        if not pf["ok"]:
            _emit("ZZTHORLAT_PREFLIGHT", json.dumps({"ok": False, "reasons": pf["reasons"]}))
            _emit("ZZTHORLAT_STOP THOR_BUSY_OR_UNVERIFIED")
            return EXIT["thor_busy"]
        _emit("ZZTHORLAT_PREFLIGHT_OK")
    import torch
    torch.set_num_threads(int(a.threads))
    device = "cuda" if a.target == "thor" else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise Refusal(EXIT["device_fields"], "target thor but torch.cuda.is_available() is False")
    # 2. device fields
    env = device_state(a.target, torch, a.env_dir, now, runner=runner)
    try:
        check_device_fields(env, a.target)
    except DeviceFieldsMissing as e:
        os.makedirs(a.out_dir, exist_ok=True)
        with open(os.path.join(a.out_dir, "REFUSED_DEVICE_FIELDS.json"), "w", encoding="utf-8") as f:
            json.dump({"schema": SCHEMA, "status": "REFUSED", "reason": str(e), "device": env}, f, indent=1)
        _emit("ZZTHORLAT_STOP DEVICE_FIELDS_MISSING", str(e))
        return EXIT["device_fields"]
    _emit("ZZTHORLAT_ENV_OK", env.get("cuda_device_name") or env.get("cpu_name"))
    # 3. pins: code, configs, checkpoint, bundle, ctx, carve, model.py
    SB = stepb()
    code_sha = {"stepB_dir": locate_code()["stepB_dir"], "l1_dir": locate_code()["l1_dir"], "this_file": sha256_lf(os.path.abspath(__file__))}
    pins_seen = {"stepB_files": verify_stepb_files(code_sha["stepB_dir"]), "l1_files": SB.verify_l1_code(code_sha["l1_dir"]), "code_dirs": code_sha}
    cfg_meta, loaded = {}, {}
    for key, path, want in (("b", a.c8, PINS["c8_config_sha256"]), ("c", a.o4, PINS["o4_config_sha256"])):
        got = sha256_lf(path)
        _check_pin(f"config {key}", got, want)
        loaded[key] = SB.StepBConfig.from_json(path)
        SB.validate_config(loaded[key])
        cfg_meta[key] = {"path": os.path.abspath(path), "sha256": got, **loaded[key].as_json()}
    cfg_b, cfg_c = loaded["b"], loaded["c"]
    model_py = os.path.join(a.refe_dir, "model.py")
    if not os.path.isfile(model_py):
        raise Refusal(EXIT["pin"], f"{model_py} not found")
    _check_pin("refe/model.py", sha256_lf(model_py), MODEL_PY_SHA256)
    ck_sha = sha256_file(a.ckpt)
    _check_pin("checkpoint", ck_sha, PINS["ckpt_sha256"])
    bundle_sha = sha256_file(a.bundle)
    _check_pin("bundle", bundle_sha, PINS["bundle_sha256"])
    ctx_sha = sha256_file(a.ctx)
    _check_pin("ctx.npy", ctx_sha, PINS["ctx_sha256"])
    fan = load_fan(a.bundle, int(a.n_tokens))
    toks_all = fan["token_all"]
    check_tokens_sorted(toks_all)
    SB.check_data_clause({"token": np.array(toks_all)})                                        # the sealed carve B (raises SystemExit on a mismatch)
    n = int(a.n_tokens)
    check_n_tokens(n, len(toks_all))
    _emit("ZZTHORLAT_PINS_OK", f"ckpt={ck_sha[:12]} bundle={bundle_sha[:12]} ctx={ctx_sha[:12]} carve={PINS['carve_b_tokens_sha256'][:12]} n={n}")
    # 4. the scorer: real_scorer.make_score_fn UNCHANGED (head = None: plain P2)
    import real_scorer
    score_fn = real_scorer.make_score_fn(a.ckpt, a.ctx, np.array(toks_all), device=device, refe_dir=a.refe_dir, autocast=a.autocast)
    sync = make_sync(device, torch)
    seed = int(cfg_b.seeds[int(a.seed_index)])

    def progress(kind, v):
        _emit(f"ZZTHORLAT_{kind}", v)

    env["thermal_start_read"] = read_thermal() if a.target == "thor" else {}
    t_wall = time.time()
    m = measure(score_fn, score_fn.bind, [fan["traj_raw"][i] for i in range(n)], [int(p) for p in fan["pick"][:n]], cfg_b, cfg_c, seed=seed, warmup=a.warmup, repeats=a.repeats, sync=sync, SB=SB,
                progress=progress, banked_logits=fan["logits"])
    wall_s = time.time() - t_wall
    rep = report(m, cfg_b, cfg_c)
    pf_end = None
    thermal_end = None
    if a.target == "thor":
        rc, ps = runner(("ps", "-eo", "pid,ppid,user,etimes,args", "--no-headers"))
        pf_end = apply_end_scan(rep, ps, rc, kw.get("self_pid") or os.getpid(), kw.get("me"))
        thermal_end = {"thermal": read_thermal(), "devfreq": read_devfreq()}
    inputs = {"bundle": {"path": os.path.abspath(a.bundle), "sha256": bundle_sha, "keys_read": list(BUNDLE_ALLOWED), "label_keys_never_read": list(BUNDLE_LABEL_KEYS)},
              "ctx": {"path": os.path.abspath(a.ctx), "sha256": ctx_sha}, "checkpoint": {"path": os.path.abspath(a.ckpt), "sha256": ck_sha},
              "refe_model_py_sha256": sha256_lf(model_py), "carve_b_tokens_sha256": PINS["carve_b_tokens_sha256"], "n_tokens": n, "timed_tokens_sha256": tokens_sha256(fan["token"]),
              "first_token": fan["token"][0], "last_token": fan["token"][-1], "search_seed": seed, "autocast": a.autocast, "scorer_info": {k: v for k, v in score_fn.info.items() if k != "ckpt"}}
    args_echo = {k: v for k, v in vars(a).items()}
    rec = build_record(target=a.target, env=env, pins_seen=pins_seen, inputs=inputs, cfgs=cfg_meta, rep=rep, args_echo=args_echo, preflight=(None if pf is None else {"ok": pf["ok"], "reasons": pf["reasons"], "evidence": pf["evidence"]}),
                       preflight_end=pf_end, thermal_end=thermal_end)
    rec["wall_seconds"] = wall_s
    rc = finalize(rec, a.out_dir, f"{a.target}_{a.autocast}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(now))}")
    if pf is not None and pf.get("_lock", {}).get("_fd") is not None:
        os.close(pf["_lock"]["_fd"])
    return rc


def main(argv=None, **kw) -> int:
    a = build_parser().parse_args(argv)
    try:
        return cmd_preflight(a, **kw) if a.cmd == "preflight" else cmd_run(a, **kw)
    except Refusal as e:
        _emit("ZZTHORLAT_STOP", e.msg)
        return e.exit_code
    except SystemExit as e:                                                                    # a refusal raised by Step B's own checks (config rule, data clause, L1 pins)
        _emit("ZZTHORLAT_STOP", str(e.code))
        return EXIT["pin"]


if __name__ == "__main__":
    sys.exit(main())
