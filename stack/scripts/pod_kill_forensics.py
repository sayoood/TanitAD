#!/usr/bin/env python3
"""Why did that process die? — the counters that actually move, in one shot.

Written 2026-08-04 after the v5f ``rc=137`` cost three unnecessary restarts of
the programme's headline training job, twice in opposite directions:

1. First a **container-OOM** diagnosis from ``memory.usage_in_bytes`` at 98-100 %
   of the cap. That counter includes **reclaimable page cache** — on this pod it
   reads 74 % of the cap *with nothing running at all* — so it is not a pressure
   signal. Retracted as ``R-2026-08-03-mem``.
2. Then the **refutation** of that diagnosis from ``memory.failcnt == 0``. That
   is the right *kind* of counter (it only moves on the event you care about) but
   on this cgroup it is **structurally frozen at zero** and can never move — see
   :func:`live_failcnt`. The refutation was therefore as unfounded as the claim.

Both errors are the same error: **a counter was read without establishing what it
is able to say.** This module answers that question first and reports the number
second.

What it collects
----------------
* memory: the limit, the **unreclaimable** footprint (``rss + shmem`` — the only
  part a reclaim pass cannot give back), and *the failcnt that is live for this
  cgroup* rather than the one that is conventionally quoted.
* the OOM-kill counter, **with the container-start time that bounds it** — it is
  monotonic within one container and resets when the container is recreated, so
  it is meaningless without that window (``oom_kill`` read 6 then 0 on pod2 and
  was quoted as history; that was retracted).
* CPU-quota throttling measured as a **delta over a window**, not a lifetime
  total — a 25-hour total looks alarming and says nothing about now.
* GPU utilisation over **>= 10 samples**, because a single sample of an
  input-bound trainer is worthless (measured here: 21 % to 100 % within 20 s).
* per-process RSS split into anon / file / shmem, so worker cost and transport
  cost are separable.
* whether ``dmesg`` is even **readable** — in an unprivileged container it is
  not, and "no OOM line in dmesg" then means nothing at all.

Usage
-----
    python3 pod_kill_forensics.py --json /tmp/forensics.json
    python3 pod_kill_forensics.py --pids 30599,30649 --gpu-samples 15
    python3 pod_kill_forensics.py --explain-rc 137

The parsing and verdict logic is pure and lives in the functions below so it can
be tested without a pod (``stack/tests/test_pod_kill_forensics.py``).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time

# --------------------------------------------------------------------------
# exit codes
# --------------------------------------------------------------------------

# 128+N is the shell's encoding of "died on signal N". The three that actually
# show up on a training pod, and what each one licenses you to conclude.
_SIGNALS = {
    2: ("SIGINT", "interactive interrupt — a human, or a Ctrl-C reaching the process group"),
    9: ("SIGKILL", "UNCATCHABLE: the kernel OOM killer, or an external `kill -9`. "
                   "Never a Python exception — a Python error exits 1 with a traceback"),
    11: ("SIGSEGV", "native crash (C extension / driver), not a memory-limit event"),
    15: ("SIGTERM", "polite termination — a supervisor, an operator, or container shutdown"),
}


def decode_exit_code(rc: int) -> dict:
    """What an exit code does and does NOT license you to conclude.

    The distinction that matters: ``rc=137`` is SIGKILL and is therefore
    **never** a CUDA OOM. ``torch.OutOfMemoryError`` is an exception; it exits
    **1** and leaves a traceback. Conflating the two sent the v5f diagnosis in
    the wrong direction twice.
    """
    rc = int(rc)
    if rc == 0:
        return {"rc": rc, "kind": "clean", "signal": None, "name": None,
                "meaning": "exited normally"}
    if rc == 1:
        return {"rc": rc, "kind": "exception", "signal": None, "name": None,
                "meaning": "Python-level failure — EXPECT A TRACEBACK on stderr. "
                           "CUDA OOM lands here, NOT on 137"}
    if 128 < rc < 192:
        sig = rc - 128
        name, meaning = _SIGNALS.get(sig, (f"SIG{sig}", "see `kill -l`"))
        return {"rc": rc, "kind": "signal", "signal": sig, "name": name,
                "meaning": meaning}
    return {"rc": rc, "kind": "error", "signal": None, "name": None,
            "meaning": "non-zero application exit"}


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def parse_kv(text: str) -> dict[str, int]:
    """Parse the ``key value`` files (``memory.stat``, ``cpu.stat``, …).

    Non-integer values are skipped rather than raising: these files gain fields
    across kernel versions and a probe must never die on an unfamiliar one.
    """
    out: dict[str, int] = {}
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            out[parts[0]] = int(parts[1])
        except ValueError:
            continue
    return out


def _read(path: str) -> str | None:
    try:
        with open(path, "r") as fh:
            return fh.read().strip()
    except OSError:
        return None


def _read_int(path: str) -> int | None:
    raw = _read(path)
    if raw is None:
        return None
    try:
        return int(raw.split()[0])
    except (ValueError, IndexError):
        return None


# --------------------------------------------------------------------------
# THE load-bearing one: which failcnt can move?
# --------------------------------------------------------------------------

def live_failcnt(mem_limit: int | None, memsw_limit: int | None,
                 swap_total: int = 0) -> dict:
    """Which of ``memory.failcnt`` / ``memory.memsw.failcnt`` is able to move.

    In cgroup v1 ``try_charge()`` charges the **memsw** counter FIRST and only
    then ``memory``. ``page_counter_try_charge`` increments ``failcnt`` on
    whichever counter it exceeded. So when swap accounting is on and
    ``memsw.limit <= memory.limit`` — the ordinary Docker/RunPod configuration,
    where the two are set equal and there is no swap — the memsw charge fails at
    exactly the point the memory charge would have, memsw absorbs every failure,
    and **``memory.failcnt`` is pinned at 0 for the life of the container no
    matter how hard the cap is hit.**

    MEASURED on ``tanitad-new`` 2026-08-04, which is what this function exists
    to stop anyone re-deriving: ``memory.failcnt`` **0** while
    ``memory.memsw.failcnt`` **28,908,911** and ``memory.max_usage_in_bytes``
    exactly equal to ``memory.limit_in_bytes``. A cgroup whose peak usage equals
    its limit has certainly hit the limit; ``failcnt 0`` alongside that is only
    explicable by the memsw-first charge path.

    ⇒ **"failcnt is 0, therefore the cap was never hit" is UNSOUND** unless this
    check says ``memory`` is the live counter.
    """
    if mem_limit is None:
        return {"live": "unknown", "frozen": [],
                "reason": "no memory limit readable — cgroup v2, or not containerised"}
    if memsw_limit is None:
        return {"live": "memory.failcnt", "frozen": [],
                "reason": "no memsw accounting — memory.failcnt is the live counter"}
    if memsw_limit <= mem_limit and swap_total == 0:
        return {
            "live": "memory.memsw.failcnt",
            "frozen": ["memory.failcnt"],
            "reason": (
                f"memsw limit ({memsw_limit}) <= memory limit ({mem_limit}) and swap is 0, "
                "so the memsw counter is charged first and absorbs every failure: "
                "memory.failcnt is STRUCTURALLY FROZEN AT 0 and proves nothing"),
        }
    return {"live": "memory.failcnt", "frozen": [],
            "reason": f"memsw limit ({memsw_limit}) exceeds memory limit ({mem_limit}) — "
                      "swap headroom exists, so memory.failcnt can move"}


def unreclaimable_bytes(memstat: dict[str, int]) -> int:
    """The part of the cgroup's charge a reclaim pass **cannot** give back.

    ``rss`` (anonymous) plus ``shmem`` (tmpfs / DataLoader IPC segments, which
    are counted inside ``cache`` but are only evictable to swap — and there is
    no swap). Page cache is excluded precisely because it is reclaimable: it is
    what made ``usage_in_bytes`` read 98-100 % on an idle box.

    **This, against the limit, is the headroom number.** ``usage_in_bytes`` is not.
    """
    return int(memstat.get("rss", 0)) + int(memstat.get("shmem", 0))


def memory_headroom(memstat: dict[str, int], limit: int | None) -> dict:
    """Unreclaimable footprint vs the cap, with the reclaimable part named."""
    unrec = unreclaimable_bytes(memstat)
    cache = int(memstat.get("cache", 0))
    out = {
        "unreclaimable_bytes": unrec,
        "unreclaimable_gb": round(unrec / 2**30, 2),
        "reclaimable_cache_gb": round(max(0, cache - int(memstat.get("shmem", 0))) / 2**30, 2),
        "limit_gb": None, "headroom_gb": None, "unreclaimable_pct_of_limit": None,
    }
    if limit:
        out["limit_gb"] = round(limit / 2**30, 2)
        out["headroom_gb"] = round((limit - unrec) / 2**30, 2)
        out["unreclaimable_pct_of_limit"] = round(100.0 * unrec / limit, 1)
    return out


def oom_window(oom_control: dict[str, int], container_start_epoch: float | None,
               now: float | None = None) -> dict:
    """``oom_kill`` **with the window that makes it quotable**.

    The counter is monotonic within one container and **resets when the
    container is recreated**. It read 6 and later 0 on pod2 and was quoted as
    "killed six times"; that was retracted. Reporting it without the window is
    the same error, so this function refuses to hand back a bare count.
    """
    now = time.time() if now is None else now
    kills = int(oom_control.get("oom_kill", 0)) if oom_control else 0
    out = {
        "oom_kill": kills,
        "under_oom": int(oom_control.get("under_oom", 0)) if oom_control else None,
        "oom_kill_disable": int(oom_control.get("oom_kill_disable", 0)) if oom_control else None,
        "window_start_epoch": container_start_epoch,
        "window_hours": None,
        "quotable": False,
        "caveat": ("counts tasks in this cgroup killed by ANY OOM killer (this cgroup's "
                   "or the host's global one) — it does NOT by itself say which"),
    }
    if container_start_epoch:
        out["window_hours"] = round((now - container_start_epoch) / 3600.0, 2)
        out["quotable"] = True
        out["statement"] = (
            f"{kills} OOM kill(s) in this container since it started "
            f"{out['window_hours']} h ago — NOT a lifetime history of the pod")
    else:
        out["statement"] = ("container start time unknown ⇒ the counter has no window "
                            "and MUST NOT be quoted as a count of kills")
    return out


# --------------------------------------------------------------------------
# the v2-cache DataLoader memory model
# --------------------------------------------------------------------------

def lru_ram_bytes(workers: int, lru_size: int, mean_payload_bytes: int,
                  n_cache_dirs: int = 1) -> int:
    """RAM held by :class:`V2CompressedCache` LRUs across the DataLoader.

    The LRU is per-PROCESS and dropped on pickling (``__getstate__``), so every
    worker fills its own and the main process holds one more:
    ``(workers + 1) * lru_size * mean_payload`` per cache dir.

    ⚠️ The class docstring says "~2-4 MB/clip". **MEASURED 2026-08-04 on
    ``physicalai-train-e438721ae894-w120-256x640cyl``: mean 33.4 MB/clip**
    (n=40) — the 256x640 lossless-PNG caches are 8-17x the figure that estimate
    was written for. Using the docstring's number under-budgets
    ``--workers 8 --v2-lru 64`` by ~16 GB, which is how a config that cannot fit
    looked affordable.
    """
    return int(max(0, workers) + 1) * int(lru_size) * int(mean_payload_bytes) * int(n_cache_dirs)


def inflight_sample_bytes(observed_shmem_bytes: int, workers: int,
                          batch: int, prefetch_factor: int = 2) -> float:
    """Per-in-flight-sample transport cost, back-solved from an observed shmem.

    A DataLoader holds ``workers * prefetch_factor * batch`` samples in flight,
    delivered to the parent through shared memory. Dividing the parent's
    observed ``RssShmem`` by that count gives a per-sample figure that can be
    re-projected onto a candidate config. Returns 0.0 when nothing is in flight.
    """
    n = int(workers) * int(prefetch_factor) * int(batch)
    return (float(observed_shmem_bytes) / n) if n > 0 else 0.0


def project_config(base_unreclaimable: int, *, from_workers: int, from_batch: int,
                   from_lru: int, to_workers: int, to_batch: int, to_lru: int,
                   mean_payload_bytes: int, observed_shmem_bytes: int,
                   per_worker_anon_bytes: int, prefetch_factor: int = 2) -> dict:
    """Project the unreclaimable footprint of a candidate DataLoader config.

    Every term is linear in a quantity that was measured on the running config,
    which is the only reason this is worth more than a guess — and it is still
    an EXTRAPOLATION, so it is labelled as one in the output.
    """
    d_lru = (lru_ram_bytes(to_workers, to_lru, mean_payload_bytes)
             - lru_ram_bytes(from_workers, from_lru, mean_payload_bytes))
    per_sample = inflight_sample_bytes(observed_shmem_bytes, from_workers,
                                       from_batch, prefetch_factor)
    d_shmem = per_sample * prefetch_factor * (to_workers * to_batch
                                              - from_workers * from_batch)
    d_workers = int(per_worker_anon_bytes) * (int(to_workers) - int(from_workers))
    total = int(base_unreclaimable + d_lru + d_shmem + d_workers)
    return {
        "evidence_class": "ESTIMATED (linear extrapolation of MEASURED per-unit costs)",
        "delta_lru_gb": round(d_lru / 2**30, 2),
        "delta_shmem_gb": round(d_shmem / 2**30, 2),
        "delta_worker_anon_gb": round(d_workers / 2**30, 2),
        "projected_unreclaimable_gb": round(total / 2**30, 2),
        "projected_unreclaimable_bytes": total,
    }


def summarize_samples(vals) -> dict:
    """median/mean/min/max — a single GPU sample of an input-bound job is noise."""
    xs = sorted(float(v) for v in vals)
    if not xs:
        return {"n": 0, "median": None, "mean": None, "min": None, "max": None}
    n = len(xs)
    median = xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0
    return {"n": n, "median": round(median, 2), "mean": round(sum(xs) / n, 2),
            "min": xs[0], "max": xs[-1]}


# --------------------------------------------------------------------------
# collection (needs a real box)
# --------------------------------------------------------------------------

CG = "/sys/fs/cgroup/memory"


def collect_memory() -> dict:
    limit = _read_int(f"{CG}/memory.limit_in_bytes")
    memsw = _read_int(f"{CG}/memory.memsw.limit_in_bytes")
    memstat = parse_kv(_read(f"{CG}/memory.stat") or "")
    oomc = parse_kv(_read(f"{CG}/memory.oom_control") or "")
    swap_total = int(memstat.get("swap", 0))
    which = live_failcnt(limit, memsw, swap_total)
    counters = {
        "memory.failcnt": _read_int(f"{CG}/memory.failcnt"),
        "memory.memsw.failcnt": _read_int(f"{CG}/memory.memsw.failcnt"),
        "memory.kmem.failcnt": _read_int(f"{CG}/memory.kmem.failcnt"),
    }
    return {
        "limit_bytes": limit, "memsw_limit_bytes": memsw,
        "usage_bytes": _read_int(f"{CG}/memory.usage_in_bytes"),
        "max_usage_bytes": _read_int(f"{CG}/memory.max_usage_in_bytes"),
        "failcnt_counters": counters,
        "which_failcnt_is_live": which,
        "live_failcnt_value": counters.get(which["live"]),
        "headroom": memory_headroom(memstat, limit),
        "memory_stat": {k: memstat.get(k) for k in
                        ("rss", "cache", "shmem", "swap", "mapped_file",
                         "active_anon", "inactive_anon", "active_file", "inactive_file")},
        "oom": oom_window(oomc, container_start_epoch()),
        "note": ("usage_in_bytes INCLUDES reclaimable page cache and is NOT a pressure "
                 "signal — read headroom.unreclaimable_gb against limit_gb instead"),
    }


def container_start_epoch() -> float | None:
    """Epoch seconds at which PID 1 started — bounds every monotonic counter."""
    try:
        with open("/proc/uptime") as fh:
            host_uptime = float(fh.read().split()[0])
        with open("/proc/1/stat") as fh:
            starttime_ticks = float(fh.read().rsplit(")", 1)[1].split()[19])
        hz = os.sysconf("SC_CLK_TCK")
        return time.time() - (host_uptime - starttime_ticks / hz)
    except (OSError, IndexError, ValueError, AttributeError):
        return None


def collect_cpu(window_s: float = 5.0) -> dict:
    """Throttling as a DELTA. A lifetime total says nothing about now."""
    path = "/sys/fs/cgroup/cpu,cpuacct/cpu.stat"
    quota = _read_int("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us")
    period = _read_int("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us")
    a = parse_kv(_read(path) or "")
    time.sleep(window_s)
    b = parse_kv(_read(path) or "")
    d_per = b.get("nr_periods", 0) - a.get("nr_periods", 0)
    d_thr = b.get("nr_throttled", 0) - a.get("nr_throttled", 0)
    return {
        "effective_cpus": (round(quota / period, 2) if quota and period and quota > 0 else None),
        "window_s": window_s,
        "delta_nr_periods": d_per, "delta_nr_throttled": d_thr,
        "delta_throttled_time_s": round(
            (b.get("throttled_time", 0) - a.get("throttled_time", 0)) / 1e9, 3),
        "throttled_period_pct": (round(100.0 * d_thr / d_per, 2) if d_per else None),
        "lifetime_nr_throttled": b.get("nr_throttled"),
    }


def collect_gpu(samples: int = 12, interval: float = 1.0) -> dict:
    if not shutil.which("nvidia-smi"):
        return {"available": False, "reason": "nvidia-smi not on PATH"}
    utils, mems = [], []
    for _ in range(max(1, samples)):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20, check=False).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            break
        row = out.splitlines()[0].split(",") if out else []
        if len(row) >= 2:
            try:
                utils.append(float(row[0])); mems.append(float(row[1]))
            except ValueError:
                pass
        time.sleep(interval)
    return {"available": True, "utilization_pct": summarize_samples(utils),
            "memory_used_mib": summarize_samples(mems),
            "note": "a single sample of an input-bound trainer is meaningless; >=10 required"}


def collect_processes(pids: list[int]) -> list[dict]:
    """Per-process RSS split so worker cost and IPC transport are separable."""
    rows = []
    for pid in pids:
        status = _read(f"/proc/{pid}/status")
        if status is None:
            rows.append({"pid": pid, "alive": False})
            continue
        kv = {}
        for line in status.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                parts = v.split()
                if parts and parts[0].isdigit():
                    kv[k] = int(parts[0])
        cmd = (_read(f"/proc/{pid}/cmdline") or "").replace("\x00", " ").strip()
        rows.append({
            "pid": pid, "alive": True,
            "rss_kb": kv.get("VmRSS"), "anon_kb": kv.get("RssAnon"),
            "file_kb": kv.get("RssFile"), "shmem_kb": kv.get("RssShmem"),
            "threads": kv.get("Threads"), "cmdline": cmd[:400],
        })
    return rows


def collect_kernel_log_access() -> dict:
    """Can we see the kernel's OOM report at all?

    ⚠️ In an unprivileged container ``dmesg`` fails with EPERM. "No OOM line in
    dmesg" is then **not evidence of no OOM** — it is evidence of no dmesg.
    Absence found at one location is not absence.
    """
    res = {"dmesg_readable": False, "kmsg_readable": os.access("/dev/kmsg", os.R_OK),
           "detail": ""}
    if shutil.which("dmesg"):
        try:
            p = subprocess.run(["dmesg"], capture_output=True, text=True,
                               timeout=15, check=False)
            res["dmesg_readable"] = p.returncode == 0 and bool(p.stdout)
            res["detail"] = (p.stderr or "").strip()[:200]
        except (OSError, subprocess.SubprocessError) as exc:
            res["detail"] = str(exc)[:200]
    else:
        res["detail"] = "dmesg not installed"
    if not res["dmesg_readable"] and not res["kmsg_readable"]:
        res["warning"] = ("kernel log NOT readable from here — an OOM kill CANNOT be "
                          "confirmed or excluded via dmesg. Use the cgroup oom counter "
                          "and the provider console (RunPod surfaces an OOM banner).")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pids", default="", help="comma-separated PIDs to profile")
    ap.add_argument("--gpu-samples", type=int, default=12)
    ap.add_argument("--cpu-window", type=float, default=5.0)
    ap.add_argument("--json", default="", help="write the full report here")
    ap.add_argument("--explain-rc", type=int, default=None,
                    help="decode one exit code and exit")
    a = ap.parse_args(argv)

    if a.explain_rc is not None:
        print(json.dumps(decode_exit_code(a.explain_rc), indent=2))
        return 0

    pids = [int(p) for p in a.pids.split(",") if p.strip().isdigit()]
    report = {
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename if hasattr(os, "uname") else "?",
        "memory": collect_memory(),
        "kernel_log": collect_kernel_log_access(),
        "cpu": collect_cpu(a.cpu_window),
        "gpu": collect_gpu(a.gpu_samples),
        "processes": collect_processes(pids),
    }
    txt = json.dumps(report, indent=2, default=str)
    if a.json:
        with open(a.json, "w") as fh:
            fh.write(txt + "\n")
    print(txt)

    mem = report["memory"]
    hr = mem.get("headroom", {})
    print("\n--- READ THIS, NOT usage_in_bytes ---")
    print(f"unreclaimable (rss+shmem): {hr.get('unreclaimable_gb')} GB of "
          f"{hr.get('limit_gb')} GB  => headroom {hr.get('headroom_gb')} GB")
    print(f"live failcnt: {mem['which_failcnt_is_live']['live']} = "
          f"{mem.get('live_failcnt_value')}  ({mem['which_failcnt_is_live']['reason']})")
    print(f"OOM: {mem['oom']['statement']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
