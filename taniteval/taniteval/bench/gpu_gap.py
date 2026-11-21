"""The GPU-GAP LAUNCHER — our-model inference touches the GPU only in a gap.

PI 2026-09-19: *"wait for a gap in the gpu"*. BUILD_PLAN.md §1: GPU use only through this
launcher — **no training process alive AND GPU memory used < 1 GB, re-checked every minute; it
backs off the moment either changes.**

    requested "cpu"   never probes, never touches CUDA.
    requested "auto"  ONE probe: gap -> "cuda" (watchdog armed), else "cpu".   (the default)
    requested "cuda"  WAITS for a gap (probe every ``interval_s``) up to ``max_wait_s``, then
                      falls back to "cpu" with a recorded GAP_WAIT_TIMEOUT event.

While on "cuda" a watchdog thread re-probes every ``interval_s``; the moment a training process
appears or FOREIGN memory reaches the limit it sets the back-off flag. The inference loop polls
:meth:`GpuGapLauncher.should_back_off` between scenes and moves to CPU for the rest of the run
(never flaps back). Every probe and event is recorded in :meth:`record` -> ``bench_run.json``
``device.gpu_gap``.

⚠️ WHAT IS MEASURABLE ON THIS BOX (MEASURED 2026-09-19): Windows/WDDM ``nvidia-smi
--query-compute-apps`` reports ``used_memory = [N/A]`` for every process, so per-process GPU
memory is NOT observable — only the DEVICE total (``--query-gpu=memory.used``). While we hold the
card, foreign usage is therefore ``total - own``, with ``own`` = the memory we added since the
pre-acquire reading (the caller reports it via :meth:`mark_own_usage`). An underestimate of
``own`` reads as foreign load and backs off — the conservative direction.

⚠️ "training process" = a live Python process whose command line names a script matching
:data:`TRAINING_SCRIPT_RE` (``refc_v3_train.py``, ``train_v6_staged.py``, ``*train*.py``,
``torchrun``) other than this process tree. A false positive only keeps us on CPU.
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time

DEFAULT_MEM_LIMIT_MIB = 1024          # "< 1 GB"
DEFAULT_INTERVAL_S = 60.0             # "re-checked every minute"
DEFAULT_MAX_WAIT_S = 8 * 3600.0

#: Env override for the foreign-GPU-memory limit, in MiB.
#:
#: ⛔ WHY THIS EXISTS. ``DEFAULT_MEM_LIMIT_MIB`` was hardcoded at 1024 and **unreachable from every
#: caller**: ``limit_mib`` was threaded correctly through :func:`gap_verdict`, :func:`device_policy`
#: and :class:`GpuGapLauncher`, yet both construction sites (``contract.py``, ``navsim/benchmark.py``)
#: omitted it and the CLI had no flag — the "built, tested, and unreachable from its caller" class in
#: ``CLAUDE.md``, which has now surfaced six times in this programme.
#:
#: MEASURED 2026-09-20/26 on the dev box: with **no training process alive at all**, the Windows/WDDM
#: desktop (explorer, Chrome, VS Code, the NVIDIA overlay) holds **1,175–5,812 MiB**, so a 1,024 MiB
#: limit can NEVER be satisfied — W7 sampled 0 gaps in 12 samples and every model arm silently fell
#: back to CPU. The PI's own instruction for this box (relayed 2026-09-26) is to gate at
#: **4,300 MiB used with >= 8 GB free host RAM**, which the hardcoded limit made unexpressible.
#:
#: ⚠️ Raising the limit does NOT weaken the standing invariant: a live training process still refuses
#: regardless of this value (see :func:`gap_verdict`). This bounds only FOREIGN GPU MEMORY.
ENV_MEM_LIMIT = "TANITAD_GPU_MEM_LIMIT_MIB"


def resolve_mem_limit_mib(limit_mib=None) -> float:
    """The limit actually in force: explicit argument > ``TANITAD_GPU_MEM_LIMIT_MIB`` > 1024.

    ⛔ Resolved HERE, at the impure boundary, and never inside :func:`gap_verdict` — that predicate
    stays pure and literal-tested, and its callers hand it an already-resolved number.
    A malformed or non-positive env value is IGNORED (falling back to the default) rather than
    silently disabling the gate: a gate that reads ``limit=0`` would refuse forever, and one that
    reads ``limit=inf`` would never refuse at all.
    """
    if limit_mib is not None:
        return float(limit_mib)
    raw = os.environ.get(ENV_MEM_LIMIT)
    if raw:
        try:
            v = float(raw)
            if v > 0 and v != float("inf"):
                return v
        except (TypeError, ValueError):
            pass
    return float(DEFAULT_MEM_LIMIT_MIB)

#: a training job, by the SCRIPT it runs (basename contains "train"), or a torch launcher
TRAINING_SCRIPT_RE = re.compile(r"(?:^|[\\/\s])[\w.\-]*train[\w.\-]*\.py\b|\btorchrun\b|torch\.distributed\.run",
                                re.IGNORECASE)


def is_training_cmdline(cmdline) -> bool:
    s = " ".join(cmdline) if isinstance(cmdline, (list, tuple)) else str(cmdline or "")
    return bool(TRAINING_SCRIPT_RE.search(s))


def gap_verdict(training_pids, mem_used_mib, *, limit_mib: float = DEFAULT_MEM_LIMIT_MIB,
                cuda_available: bool = True) -> dict:
    """The PURE predicate (unit-tested with literals). ``mem_used_mib`` = device memory used by
    everyone except us (None = unknown -> no gap)."""
    reasons = []
    if not cuda_available:
        reasons.append("CUDA not available in this interpreter")
    if training_pids:
        reasons.append(f"training process alive: pids {sorted(training_pids)}")
    if mem_used_mib is None:
        reasons.append("GPU memory used is UNKNOWN (nvidia-smi gave no reading) — no gap is assumed")
    elif not (mem_used_mib < limit_mib):
        reasons.append(f"GPU memory used {mem_used_mib:.0f} MiB >= limit {limit_mib:.0f} MiB")
    return {"gap": not reasons, "reasons": reasons, "training_pids": sorted(training_pids or []),
            "mem_used_mib": mem_used_mib, "limit_mib": limit_mib}


def probe_training_pids(exclude_pids=()) -> list:
    import psutil
    me = {os.getpid()}
    try:
        me |= {p.pid for p in psutil.Process().children(recursive=True)}
        par = psutil.Process().parent()
        while par is not None:          # our own launcher chain is never "a training process"
            me.add(par.pid)
            par = par.parent()
    except Exception:                                                  # noqa: BLE001
        pass
    me |= set(exclude_pids)
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["pid"] in me:
                continue
            name = (p.info["name"] or "").lower()
            if "python" not in name and "torchrun" not in name:
                continue
            if is_training_cmdline(p.info["cmdline"] or []):
                out.append(p.info["pid"])
        except Exception:                                              # noqa: BLE001
            continue
    return out


def training_processes(exclude_pids=()) -> list:
    """``[{pid, name, cmdline}]`` for every live training process — the SAME definition as
    :func:`probe_training_pids` (W3, 2026-09-20: *"keep that definition in one place"*; it is why a
    launcher named ``p_runner.py`` reads as no-trainer while ``refc_v3_train.py`` matches)."""
    import psutil
    out = []
    for pid in probe_training_pids(exclude_pids):
        rec = {"pid": pid, "name": None, "cmdline": None}
        try:
            pr = psutil.Process(pid)
            rec["name"] = pr.name()
            rec["cmdline"] = " ".join(pr.cmdline())[:400]
        except Exception:                                              # noqa: BLE001
            rec["cmdline"] = "UNREADABLE (the process ended, or access was denied)"
        out.append(rec)
    return out


def probe_gpu_mem_used_mib() -> float | None:
    """Device-total memory used (MiB), summed over GPUs; None if nvidia-smi cannot answer."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        vals = [float(x) for x in r.stdout.split() if re.fullmatch(r"\d+(\.\d+)?", x)]
        return sum(vals) if (r.returncode == 0 and vals) else None
    except Exception:                                                  # noqa: BLE001
        return None


def default_probe() -> tuple:
    return probe_training_pids(), probe_gpu_mem_used_mib()


def default_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:                                                  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# THE SHARED DEVICE GATE (orchestrator arbitration, 2026-09-20)                #
# --------------------------------------------------------------------------- #
OVERRIDE_FLAG = "--accept-training-box-load"
#: ONE implementation for the whole suite (W1's benchmarks AND W3's / W6's plugins):
#:   --device auto | cuda : model arms WAIT for a GPU gap -> REFUSE now, with the queue command
#:   --device cpu (explicit), no training alive : ALLOWED — the operator has chosen the hours
#:   --device cpu (explicit), training alive    : REFUSED unless OVERRIDE_FLAG is passed, and the
#:                                                flag AND the accepted training PIDs are RECORDED
#: (the standing invariant is "never add load to a box that is training"; an exception that is not
#: in the manifest is invisible in six weeks)
DECISIONS = ("CUDA", "CPU", "REFUSE")


def device_policy(requested: str, *, accept_training_box_load: bool = False, probe=default_probe,
                  cuda_available=default_cuda_available, limit_mib=None,
                  queue_hint: str = "", processes=None) -> dict:
    """The decision for OUR-MODEL INFERENCE on this box. Pure w.r.t. ``probe`` — unit-tested with
    literals. Returns ``{decision, reason, training_pids, mem_used_mib, record}``; ``record`` is what
    ``bench_run.json`` must carry (including any accepted override).

    ``limit_mib=None`` resolves through :func:`resolve_mem_limit_mib` (explicit > env > 1024), so the
    limit in force is always the one recorded in ``record["limit_mib"]`` — never an assumed default.
    """
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError(f"device must be auto|cpu|cuda, got {requested!r}")
    limit_mib = resolve_mem_limit_mib(limit_mib)
    pids, used = probe()
    v = gap_verdict(pids, used, limit_mib=limit_mib, cuda_available=cuda_available())
    rec = {"requested": requested, "training_pids": sorted(pids or []), "mem_used_mib": used,
           "limit_mib": limit_mib, "override_flag": OVERRIDE_FLAG,
           "override_used": bool(accept_training_box_load and pids and requested == "cpu"),
           "override_passed_as": (f"CLI flag {OVERRIDE_FLAG}" if accept_training_box_load else None),
           "training_processes": (processes if processes is not None else
                                  (training_processes() if pids else [])),
           "training_process_definition": "taniteval.bench.gpu_gap.TRAINING_SCRIPT_RE (the ONE definition)",
           "rule": ("auto|cuda: model arms WAIT for a GPU gap; cpu: allowed only when no training process is "
                    f"alive, or with {OVERRIDE_FLAG} (recorded)")}
    if requested in ("auto", "cuda"):
        if v["gap"]:
            return {"decision": "CUDA", "reason": "a GPU gap is open", "training_pids": rec["training_pids"],
                    "mem_used_mib": used, "record": rec}
        return {"decision": "REFUSE", "training_pids": rec["training_pids"], "mem_used_mib": used, "record": rec,
                "reason": ("model arms WAIT for a GPU gap on --device " + requested + ": " + "; ".join(v["reasons"])
                           + (f". Queue it: {queue_hint}" if queue_hint else "")
                           + f". To run on CPU instead, pass --device cpu explicitly"
                           + (f" (and {OVERRIDE_FLAG}, since a training process is alive)" if pids else ""))}
    if not pids:
        return {"decision": "CPU", "reason": "explicit --device cpu and no training process is alive",
                "training_pids": [], "mem_used_mib": used, "record": rec}
    if accept_training_box_load:
        return {"decision": "CPU", "training_pids": rec["training_pids"], "mem_used_mib": used, "record": rec,
                "reason": (f"explicit --device cpu WITH {OVERRIDE_FLAG}: accepting CPU load beside training "
                           f"process(es) {rec['training_pids']} — recorded in bench_run.json")}
    return {"decision": "REFUSE", "training_pids": rec["training_pids"], "mem_used_mib": used, "record": rec,
            "reason": (f"--device cpu while training process(es) {rec['training_pids']} are alive: REFUSED. "
                       f"The standing invariant is 'never add load to a box that is training'. Pass "
                       f"{OVERRIDE_FLAG} to accept it (the flag and those PIDs are then recorded)"
                       + (f", or queue it: {queue_hint}" if queue_hint else ""))}


#: ⭐ THE NAME W3's plugin probes for (`device_gate`), so it switches to the shared gate with no
#: edit (W3, 2026-09-20). `device_policy` stays the canonical name; this is the same object.
device_gate = device_policy


class GpuGapLauncher:
    def __init__(self, requested: str = "auto", *, limit_mib=None,
                 interval_s: float = DEFAULT_INTERVAL_S, max_wait_s: float = DEFAULT_MAX_WAIT_S,
                 probe=default_probe, cuda_available=default_cuda_available,
                 sleep=time.sleep, clock=time.time, log=print):
        if requested not in ("auto", "cpu", "cuda"):
            raise ValueError(f"device must be auto|cpu|cuda, got {requested!r}")
        limit_mib = resolve_mem_limit_mib(limit_mib)
        self.requested, self.limit_mib, self.interval_s = requested, float(limit_mib), float(interval_s)
        self.max_wait_s = float(max_wait_s)
        self._probe, self._cuda_available, self._sleep, self._clock, self._log = (
            probe, cuda_available, sleep, clock, log)
        self.device = None
        self.events: list = []
        self.used_before_mib = None
        self.own_mib = 0.0
        self._backoff = threading.Event()
        self._stop = threading.Event()
        self._thread = None

    # ------------------------------------------------------------------ #
    def _event(self, kind: str, **kw):
        ev = {"t": round(self._clock(), 3), "event": kind, **kw}
        self.events.append(ev)
        self._log(f"[gpu-gap] {kind} {kw}")
        return ev

    def check(self, *, own_mib: float = 0.0) -> dict:
        pids, used = self._probe()
        foreign = None if used is None else max(0.0, used - own_mib)
        v = gap_verdict(pids, foreign, limit_mib=self.limit_mib, cuda_available=self._cuda_available())
        v["mem_used_total_mib"] = used
        return v

    def acquire(self) -> str:
        """Decide the device ONCE for this run (idempotent)."""
        if self.device is not None:
            return self.device
        if self.requested == "cpu":
            self._event("CPU_REQUESTED")
            self.device = "cpu"
            return self.device
        t0 = self._clock()
        while True:
            v = self.check()
            if v["gap"]:
                self.used_before_mib = v["mem_used_total_mib"]
                self._event("GAP_ACQUIRED", mem_used_mib=v["mem_used_total_mib"], limit_mib=self.limit_mib)
                self.device = "cuda"
                self._arm_watchdog()
                return self.device
            if self.requested == "auto":
                self._event("NO_GAP_CPU", reasons=v["reasons"])
                self.device = "cpu"
                return self.device
            if self._clock() - t0 >= self.max_wait_s:
                self._event("GAP_WAIT_TIMEOUT", waited_s=round(self._clock() - t0, 1), reasons=v["reasons"])
                self.device = "cpu"
                return self.device
            self._event("WAITING_FOR_GAP", reasons=v["reasons"])
            self._sleep(self.interval_s)

    def mark_own_usage(self, mem_used_now_mib: float | None = None):
        """Call after the model is on the GPU: everything above the pre-acquire reading is ours."""
        if self.device != "cuda":
            return
        if mem_used_now_mib is None:
            _, mem_used_now_mib = self._probe()
        if mem_used_now_mib is not None and self.used_before_mib is not None:
            self.own_mib = max(0.0, mem_used_now_mib - self.used_before_mib)
        self._event("OWN_USAGE_MARKED", own_mib=self.own_mib)

    def _arm_watchdog(self):
        def loop():
            while not self._stop.wait(self.interval_s):
                v = self.check(own_mib=self.own_mib)
                if not v["gap"]:
                    self._event("BACK_OFF", reasons=v["reasons"], mem_used_total_mib=v["mem_used_total_mib"])
                    self._backoff.set()
                    return
        self._thread = threading.Thread(target=loop, name="gpu-gap-watchdog", daemon=True)
        self._thread.start()

    def should_back_off(self) -> bool:
        return self._backoff.is_set()

    def backed_off(self):
        """The caller has moved to CPU; stay there for the rest of the run."""
        if self.device == "cuda":
            self.device = "cpu"
            self._event("MOVED_TO_CPU")
        self.stop()

    def stop(self):
        self._stop.set()

    def record(self) -> dict:
        used = "cuda" if any(e["event"] == "GAP_ACQUIRED" for e in self.events) else "cpu"
        if used == "cuda" and any(e["event"] == "MOVED_TO_CPU" for e in self.events):
            used = "mixed"
        return {"requested": self.requested, "used": used, "limit_mib": self.limit_mib,
                "interval_s": self.interval_s, "max_wait_s": self.max_wait_s,
                "rule": ("GPU only when no training process is alive AND device memory used < "
                         f"{self.limit_mib:.0f} MiB; re-checked every {self.interval_s:.0f} s; back off at once "
                         "(PI 2026-09-19 'wait for a gap in the gpu')"),
                "events": self.events}


def main(argv=None) -> int:
    """``python -m taniteval.bench gpu-gap`` — print the current verdict (probes only)."""
    import json
    pids, used = default_probe()
    v = gap_verdict(pids, used, cuda_available=True)
    v["note"] = "CUDA availability not probed here (no torch import); the launcher checks it."
    print(json.dumps(v, indent=1))
    return 0
