"""Logged, guarded range reads for the unseen-stratum fetch (S2 + S4).

Every request is appended to logs/requests_<stage>.jsonl: url, range, bytes, http status, attempt.
Guards (all binding per BRIEF.md):
  * C: free space < 5.5 GB  -> refuse to start / stop mid-way (DiskGuardError)
  * at most 4 parallel range requests (module semaphore)
  * only the one host+prefix the brief approved
  * a range request that the server answers with 200 (whole object) is aborted by --max-filesize
"""
from __future__ import annotations
import json, os, shutil, subprocess, threading, time

HOST_PREFIX = "https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1/"
DISK_ROOT = "C:/"
MIN_FREE = int(5.5e9)
MAX_PARALLEL = 4
LOGDIR = "C:/Users/Admin/par18/unseen_s2s4/logs"

_sem = threading.BoundedSemaphore(MAX_PARALLEL)
_lock = threading.Lock()
_stage = "misc"
_stats = {"requests": 0, "bytes": 0, "retries": 0, "failures": 0}


class DiskGuardError(RuntimeError):
    pass


def set_stage(name: str) -> None:
    global _stage
    _stage = name
    os.makedirs(LOGDIR, exist_ok=True)


def stats() -> dict:
    return dict(_stats)


def disk_guard(where: str = "") -> int:
    free = shutil.disk_usage(DISK_ROOT).free
    if free < MIN_FREE:
        raise DiskGuardError(f"C: free {free/1e9:.3f} GB < 5.5 GB guard {where}")
    return free


def _log(rec: dict) -> None:
    rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with _lock:
        with open(os.path.join(LOGDIR, f"requests_{_stage}.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


def _check_url(url: str) -> None:
    if not url.startswith(HOST_PREFIX):
        raise RuntimeError(f"URL outside the approved prefix: {url}")


_CURL = ["curl", "--ssl-no-revoke", "-s", "--retry", "0", "--connect-timeout", "30", "--max-time", "600"]


def rng(url: str, a: int, b: int, tries: int = 6) -> bytes:
    """Range read [a, b] inclusive. Logs every attempt. Raises after `tries` failures."""
    _check_url(url)
    want = b - a + 1
    last = None
    for attempt in range(1, tries + 1):
        disk_guard(f"before range {a}-{b}")
        t0 = time.time()
        with _sem:
            p = subprocess.run(
                _CURL + ["--max-filesize", str(want + 4096), "-r", f"{a}-{b}",
                         "-w", "\n%{http_code}", url],
                capture_output=True)
        dt = time.time() - t0
        out = p.stdout
        code, body = None, b""
        if p.returncode == 0 and b"\n" in out:
            body, _, tail = out.rpartition(b"\n")
            try:
                code = int(tail.strip())
            except ValueError:
                code = None
        ok = (p.returncode == 0 and code == 206 and len(body) == want)
        with _lock:
            _stats["requests"] += 1
            if ok:
                _stats["bytes"] += len(body)
            else:
                _stats["retries"] += 1
        _log({"url": url, "range": f"{a}-{b}", "want": want, "bytes": len(body), "status": code,
              "curl_rc": p.returncode, "attempt": attempt, "secs": round(dt, 3), "ok": ok})
        if ok:
            return body
        last = f"rc={p.returncode} status={code} got={len(body)} want={want} err={p.stderr[:120]!r}"
        time.sleep(min(60, 2 ** attempt))
    with _lock:
        _stats["failures"] += 1
    raise RuntimeError(f"range {a}-{b} of {url} failed after {tries} tries: {last}")


def head_size(url: str, tries: int = 6) -> int:
    _check_url(url)
    last = None
    for attempt in range(1, tries + 1):
        disk_guard("before HEAD")
        t0 = time.time()
        with _sem:
            p = subprocess.run(_CURL + ["-I", url], capture_output=True, text=True)
        dt = time.time() - t0
        size, code = None, None
        for line in p.stdout.splitlines():
            if line.startswith("HTTP/"):
                try:
                    code = int(line.split()[1])
                except Exception:
                    pass
            if line.lower().startswith("content-length"):
                size = int(line.split(":")[1].strip())
        ok = (p.returncode == 0 and code == 200 and size is not None)
        with _lock:
            _stats["requests"] += 1
        _log({"url": url, "range": "HEAD", "bytes": 0, "status": code, "content_length": size,
              "curl_rc": p.returncode, "attempt": attempt, "secs": round(dt, 3), "ok": ok})
        if ok:
            return size
        last = f"rc={p.returncode} status={code}"
        if code in (403, 404):
            raise RuntimeError(f"HEAD {url}: HTTP {code}")
        time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"HEAD {url} failed: {last}")
