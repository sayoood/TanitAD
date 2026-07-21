#!/usr/bin/env python3
"""fleet_probe — discovery-based liveness probe for the TanitAD compute fleet.

Why this exists
---------------
The 6-hourly fleet monitor (`.claude/skills/fleet-status/SKILL.md`) has now
missed a dead trainer **four times**. The root cause is structural, not a bug
in any one check: every check greps a **hardcoded target** —

    grep -o '"step": [0-9]*' /workspace/experiments/p0-sB01-realmix.log
    pgrep -fc "train_worldmode[l]"
    grep ... /workspace/experiments/arm_base.log   # and arm_kstep.log

Those three names belong to runs that ended weeks ago. When the run is renamed
(and every arm since has been), the grep matches nothing, prints nothing, and
the monitor reports **no anomaly**. On 2026-07-20 05:01 UTC two of four GPUs
were idle behind dead trainers and the 04:55 probe called the fleet fine.

The structural fix is one rule, enforced everywhere below:

    **Absence of evidence is an ALARM, not an all-clear.**

So this probe:
  * **discovers** trainers from the live process table (full cmdline), it does
    not look for a name somebody typed into a skill file months ago;
  * **discovers** logs by mtime under the run's own ``--out`` dir and the
    shell redirect captured in the launcher's cmdline, not by filename;
  * starts every host at verdict ``UNKNOWN`` and requires *positive evidence*
    to reach GREEN — a host that answers but shows nothing is AMBER;
  * cross-checks the GPU against the process table, which is what actually
    catches "dead trainer, hot/idle GPU": memory resident with no owner, or a
    training-role host at 0 % with nobody running;
  * measures disk headroom with a real ``dd`` write, because ``df`` on these
    pods reports the multi-TB MooseFS cluster and hides the per-pod quota
    (CLAUDE.md, traps preflight);
  * remembers the last observed step per run so **"step not advancing"** is
    detectable across probes — a futex-deadlocked trainer is alive in ``ps``
    and its log is silent, which no single-shot check can see.

Safety: this probe is strictly read-only apart from one ~100 MB scratch file it
writes and deletes for the ``dd`` test. It never uses ``pgrep -f``/``pkill -f``
(those self-match the probing ssh command — CLAUDE.md).

Usage
-----
    python tools/fleet_probe.py                     # table, all hosts
    python tools/fleet_probe.py --json              # machine-readable
    python tools/fleet_probe.py --hosts pod1 pod2   # subset
    python tools/fleet_probe.py --no-dd             # skip the disk write test

Exit codes: ``0`` all GREEN, ``1`` any AMBER, ``2`` any RED.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

# --------------------------------------------------------------------------
# fleet definition. `role` decides what "healthy" means: a `train` host with no
# trainer is RED; a `burst` host with no trainer is GREEN-idle by design.
# --------------------------------------------------------------------------
FLEET: dict[str, dict] = {
    "pod1": {"ssh": "tanitad-pod", "role": "train",
             "note": "A6000 — flagship arm"},
    "pod2": {"ssh": "tanitad-pod2", "role": "train",
             "note": "A40 — arm slot"},
    "pod3": {"ssh": "tanitad-pod3", "role": "train",
             "note": "A40 — REF/VLM slot"},
    "eval": {"ssh": "tanitad-eval", "role": "burst",
             "note": "A40 — TanitEval, idle by design between jobs"},
}

# A process is a *trainer/job* if its cmdline matches this and not EXCLUDE.
# Deliberately broad: the whole point is not to depend on one run's name.
JOB_RE = re.compile(
    r"(train[_a-z0-9]*\.py|scripts/train|vlm_[a-z_]+\.py|"
    r"taniteval\.runner|refa_train|refb_train|refc_train)", re.I)
EXCLUDE_RE = re.compile(r"(jupyter|ipykernel|/bin/ssh|fleet_probe)", re.I)

# Thresholds. Each is a policy choice, so each is named and overridable.
GPU_MEM_ORPHAN_MB = 512      # >this MiB resident with no job = orphaned memory
LOG_STALE_S = 900            # 15 min without a log write while a job is alive
STEP_STALL_S = 1800          # 30 min of identical step across probes
DISK_MIN_MBPS = 5.0          # a healthy pod writes far above this
DISK_HEADROOM_MB = 2048      # dd must be able to place this much

STATE_DEFAULT = Path(__file__).with_name(".fleet_probe_state.json")

# --------------------------------------------------------------------------
# the remote payload. One ssh round-trip per host, sectioned plain text.
# --------------------------------------------------------------------------
REMOTE = r"""
echo '##NOW'; date +%s
echo '##HOST'; hostname
echo '##GPU'; nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo NA
echo '##PS'; ps -eo pid,ppid,etimes,rss,args --no-headers 2>/dev/null | tr -s ' ' | sed 's/^ //'
echo '##LOGS'
# Per-directory, hard-timeboxed, recent-only. A single `find /workspace -maxdepth 3`
# took >90 s on the MooseFS-backed pods (measured 2026-07-21) and timed out the
# whole probe on exactly the two hosts that were actually training — i.e. the
# naive form is blind precisely where it matters.
for d in /workspace/experiments /root /tmp /workspace; do
  [ -d "$d" ] && timeout 8 find "$d" -maxdepth 2 -type f -mmin -2880 \
    \( -name '*.log' -o -name '*.out' -o -name 'metrics.json' \) \
    -printf '%T@|%s|%p\n' 2>/dev/null
done | sort -rn | head -60
echo '##DD'
if [ "$DO_DD" = "1" ]; then
  D=/workspace/.fleet_probe_dd
  T0=$(date +%s.%N)
  if dd if=/dev/zero of=$D bs=1M count=100 conv=fsync 2>/dev/null 1>/dev/null; then
    T1=$(date +%s.%N); echo "OK $(echo "$T1 $T0" | awk '{printf "%.3f", $1-$2}')"
  else
    echo "FAIL dd-could-not-write-100MB"
  fi
  rm -f $D
else
  echo SKIP
fi
echo '##END'
"""


# --------------------------------------------------------------------------
@dataclass
class Job:
    """One discovered training/eval run on a host."""
    key: str                       # the --out value, or the script name
    script: str
    pids: list[int] = field(default_factory=list)
    etimes: int = 0                # oldest member's age (s)
    log: str | None = None
    log_age_s: float | None = None
    step: int | None = None


@dataclass
class HostReport:
    name: str
    ssh: str
    role: str
    reachable: bool = False
    error: str | None = None
    hostname: str | None = None
    gpus: list[dict] = field(default_factory=list)
    jobs: list[Job] = field(default_factory=list)
    disk_mbps: float | None = None
    disk_note: str | None = None
    verdict: str = "UNKNOWN"
    findings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
def _lf(script: str) -> bytes:
    """Remote bash payloads must be LF-only, and must be fed as **bytes**.

    Two Windows traps, both of which turn every `fi` into `fi\\r` and make bash
    die with the misleading "unexpected end of file":
    (1) a CRLF checkout of this file, (2) ``subprocess(text=True)``, which
    wraps stdin in a TextIOWrapper that translates ``\\n`` -> ``os.linesep``
    on write. Encoding here defeats both."""
    return script.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def ssh_client() -> str:
    """Which ``ssh`` binary to drive.

    Measured 2026-07-21, and it cost an hour: the **MSYS/git-bash**
    ``C:\\Program Files\\Git\\usr\\bin\\ssh.exe``, driven from a native-Windows
    Python over ``subprocess`` pipes, **deadlocks intermittently** — the remote
    payload completes in ~2 s when run from a shell, but the pipe never sees
    EOF and the probe reports "ssh timeout after 90 s". It reproduced 100 % on
    the two hosts that were *training* and 0 % on the two idle ones, which
    reads exactly like a fleet outage and is not one. The native
    ``C:\\Windows\\System32\\OpenSSH\\ssh.exe`` ran the identical payload on all
    four hosts in 0.7–2.5 s. So: prefer native OpenSSH on win32.
    """
    if sys.platform == "win32":
        native = Path(r"C:\Windows\System32\OpenSSH\ssh.exe")
        if native.exists():
            return str(native)
    return "ssh"


def run_remote(ssh_alias: str, do_dd: bool, timeout: int,
               ssh_bin: str | None = None) -> tuple[bool, str]:
    cmd = [ssh_bin or ssh_client(), "-o", "ConnectTimeout=12",
           "-o", "BatchMode=yes",
           ssh_alias, f"DO_DD={'1' if do_dd else '0'} bash -s"]
    try:
        # CRLF must never reach the remote bash: a Windows checkout turns every
        # `fi` into `fi\r`, and bash then dies with "unexpected end of file".
        p = subprocess.run(cmd, input=_lf(REMOTE), capture_output=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"ssh timeout after {timeout}s"
    except OSError as exc:                                # pragma: no cover
        return False, f"ssh could not start: {exc}"
    out = p.stdout.decode("utf-8", "replace")
    err = p.stderr.decode("utf-8", "replace")
    if p.returncode != 0 or "##END" not in out:
        tail = (err or out).strip().splitlines()[-1:] or ["no output"]
        return False, f"ssh rc={p.returncode}: {tail[0][:160]}"
    return True, out


def split_sections(payload: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    cur = None
    for line in payload.splitlines():
        if line.startswith("##"):
            cur = line[2:].strip()
            out.setdefault(cur, [])
        elif cur is not None and line.strip():
            out[cur].append(line.rstrip())
    return out


# --------------------------------------------------------------------------
def parse_ps(lines: list[str]) -> list[dict]:
    procs = []
    for ln in lines:
        parts = ln.split(" ", 4)
        if len(parts) < 5:
            continue
        try:
            procs.append({"pid": int(parts[0]), "ppid": int(parts[1]),
                          "etimes": int(parts[2]), "rss_kb": int(parts[3]),
                          "args": parts[4]})
        except ValueError:
            continue
    return procs


def _out_key(args: str) -> str | None:
    """The run identity is its ``--out`` dir — stable across the worker fan-out
    and across renames of the script."""
    m = re.search(r"--out[= ]+(\S+)", args)
    return m.group(1) if m else None


def _redirect_target(args: str) -> str | None:
    """`bash -c '... > /tmp/foo.out 2>&1'` — the launcher records where the
    stdout log actually went. This is how a log is found without guessing."""
    m = re.search(r">\s*(/\S+\.(?:out|log))", args)
    return m.group(1) if m else None


def _ancestor_redirect(proc: dict, by_pid: dict[int, dict]) -> str | None:
    """Walk the ppid chain for a launcher that recorded where stdout went.

    Jobs are started as ``bash -c '... nohup python3 x.py ... > /root/foo.log
    2>&1 &'``. The python process itself carries no trace of its log, but an
    ancestor bash does — so the log is *discoverable* rather than guessed.
    (Measured: this is exactly why pod3's VLM job bound no log on the first
    live run.)"""
    seen, cur = set(), proc
    for _ in range(8):
        tgt = _redirect_target(cur["args"])
        if tgt:
            return tgt
        nxt = by_pid.get(cur["ppid"])
        if nxt is None or nxt["pid"] in seen:
            return None
        seen.add(nxt["pid"])
        cur = nxt
    return None


def discover_jobs(procs: list[dict]) -> list[Job]:
    """Group job processes by run identity. Worker processes share the parent's
    cmdline, so grouping by ``--out`` collapses a fan-out to one run."""
    groups: dict[str, Job] = {}
    by_pid = {p["pid"]: p for p in procs}
    for p in procs:
        args = p["args"]
        if EXCLUDE_RE.search(args) or not JOB_RE.search(args):
            continue
        key = _out_key(args) or JOB_RE.search(args).group(1)
        script = (re.search(r"(\S+\.py)", args) or [None, args[:40]])[1] \
            if re.search(r"(\S+\.py)", args) else args[:40]
        job = groups.setdefault(key, Job(key=key, script=script))
        job.pids.append(p["pid"])
        job.etimes = max(job.etimes, p["etimes"])
        if job.log is None:
            job.log = _ancestor_redirect(p, by_pid)
    return list(groups.values())


def parse_logs(lines: list[str]) -> list[tuple[float, int, str]]:
    out = []
    for ln in lines:
        try:
            mt, size, path = ln.split("|", 2)
            out.append((float(mt), int(size), path))
        except ValueError:
            continue
    return out


STEP_RE = re.compile(r'"step"\s*:\s*(\d+)')


def attach_logs(jobs: list[Job], logs: list[tuple[float, int, str]],
                now: float) -> None:
    """Bind each run to its newest own log. Candidate order:
    (1) the redirect target from the launcher cmdline, (2) any log inside the
    ``--out`` dir, (3) any log whose name contains the out-dir basename.
    A run with no candidate keeps ``log=None`` — which is an AMBER finding,
    never a silent pass."""
    for job in jobs:
        cands = []
        if job.log:
            cands += [(mt, sz, p) for mt, sz, p in logs if p == job.log]
        if job.key.startswith("/"):
            base = job.key.rstrip("/").split("/")[-1]
            cands += [(mt, sz, p) for mt, sz, p in logs
                      if p.startswith(job.key.rstrip("/") + "/") or base in p]
        if not cands:
            job.log = None
            continue
        mt, _sz, path = max(cands, key=lambda c: c[0])
        job.log, job.log_age_s = path, max(0.0, now - mt)


def fetch_steps(ssh_alias: str, jobs: list[Job], timeout: int,
                ssh_bin: str | None = None) -> None:
    """One extra round-trip: tail each bound log for the last ``"step": N``.
    Kept separate so the main payload stays a single fixed script."""
    paths = [j.log for j in jobs if j.log]
    if not paths:
        return
    script = "; ".join(
        f"echo '##L {shlex.quote(p)}'; tail -c 20000 {shlex.quote(p)} "
        f"2>/dev/null | grep -o '\"step\"[ ]*:[ ]*[0-9]*' | tail -1"
        for p in paths)
    try:
        p = subprocess.run([ssh_bin or ssh_client(), "-o", "ConnectTimeout=12",
                            "-o", "BatchMode=yes", ssh_alias, "bash -s"],
                           input=_lf(script), capture_output=True,
                           timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):           # pragma: no cover
        return
    cur = None
    found: dict[str, int] = {}
    for ln in p.stdout.decode("utf-8", "replace").splitlines():
        if ln.startswith("##L "):
            cur = ln[4:].strip().strip("'")
        elif cur:
            m = STEP_RE.search(ln)
            if m:
                found[cur] = int(m.group(1))
    for j in jobs:
        if j.log in found:
            j.step = found[j.log]


# --------------------------------------------------------------------------
def judge(rep: HostReport, prev: dict, now: float,
          probe_gap_s: float | None) -> None:
    """Assign the verdict. Starts UNKNOWN; only positive evidence reaches GREEN.

    RED   — a GPU that should be working is not, or a job is alive but frozen,
            or the disk cannot take a write.
    AMBER — the host answered but the evidence is incomplete (no log bound,
            no GPU readable). This is the case the old monitor scored GREEN.
    """
    if not rep.reachable:
        rep.verdict = "RED"
        rep.findings.append(f"UNREACHABLE: {rep.error}")
        return

    busy_mem = sum(g["mem_used"] for g in rep.gpus) if rep.gpus else 0
    max_util = max((g["util"] for g in rep.gpus), default=None)
    njobs = len(rep.jobs)

    if not rep.gpus:
        rep.findings.append("AMBER NO_GPU_READOUT: nvidia-smi gave nothing")

    if njobs == 0:
        if busy_mem > GPU_MEM_ORPHAN_MB:
            rep.findings.append(
                f"RED ORPHANED_GPU_MEMORY: {busy_mem} MiB resident with no "
                f"job process — the classic dead-trainer signature")
        elif rep.role == "train":
            rep.findings.append(
                f"RED GPU_IDLE_NO_TRAINER: role=train, util={max_util}%, "
                f"{busy_mem} MiB — the slot is paid for and doing nothing")
        else:
            rep.findings.append("OK idle burst host (no job expected)")
    else:
        keys = {j.key for j in rep.jobs}
        if len(keys) > 1:
            rep.findings.append(
                f"AMBER MULTIPLE_RUNS: {len(keys)} distinct --out on one host "
                f"({', '.join(sorted(keys))}) — check this is intended")
        for j in rep.jobs:
            if j.log is None:
                rep.findings.append(
                    f"AMBER NO_LOG_BOUND: {j.script} (pid {min(j.pids)}) is "
                    f"running but no log could be discovered — liveness is "
                    f"UNVERIFIED, not fine")
                continue
            if j.log_age_s is not None and j.log_age_s > LOG_STALE_S:
                rep.findings.append(
                    f"RED LOG_STALE: {j.log} last written "
                    f"{j.log_age_s/60:.1f} min ago while pid {min(j.pids)} "
                    f"is alive — frozen (futex/deadlock class)")
            key = f"{rep.name}:{j.key}"
            was = prev.get(key)
            if j.step is not None and was is not None and \
                    was.get("step") == j.step and probe_gap_s and \
                    probe_gap_s > STEP_STALL_S:
                rep.findings.append(
                    f"RED STEP_NOT_ADVANCING: {j.key} still at step {j.step} "
                    f"after {probe_gap_s/60:.0f} min")
            if j.step is None:
                rep.findings.append(
                    f"AMBER NO_STEP: could not read a step from {j.log}")
        if max_util is not None and max_util == 0 and busy_mem == 0 and njobs:
            rep.findings.append(
                "RED JOB_WITHOUT_GPU: a job process exists but the GPU is "
                "completely cold — it is not the training you think it is")

    if rep.disk_mbps is not None and rep.disk_mbps < DISK_MIN_MBPS:
        rep.findings.append(
            f"RED DISK_SLOW_OR_FULL: {rep.disk_mbps:.1f} MB/s on a real "
            f"100 MB write")
    elif rep.disk_note == "FAIL":
        rep.findings.append(
            "RED DISK_FULL: could not write 100 MB to /workspace (quota — "
            "df would have shown the cluster and lied)")

    reds = [f for f in rep.findings if f.startswith("RED")]
    ambers = [f for f in rep.findings if f.startswith("AMBER")]
    rep.verdict = "RED" if reds else ("AMBER" if ambers else "GREEN")


# --------------------------------------------------------------------------
def probe_host(name: str, spec: dict, do_dd: bool, timeout: int,
               with_steps: bool = True,
               ssh_bin: str | None = None) -> HostReport:
    rep = HostReport(name=name, ssh=spec["ssh"], role=spec["role"])
    ok, payload = run_remote(spec["ssh"], do_dd, timeout, ssh_bin)
    if not ok:
        rep.error = payload
        return rep
    rep.reachable = True
    sec = split_sections(payload)
    now = float(sec.get("NOW", [str(time.time())])[0])
    rep.hostname = (sec.get("HOST") or [None])[0]

    for ln in sec.get("GPU", []):
        if ln.strip() == "NA":
            continue
        f = [x.strip() for x in ln.split(",")]
        if len(f) >= 5:
            try:
                rep.gpus.append({"index": int(f[0]), "name": f[1],
                                 "util": int(f[2]), "mem_used": int(f[3]),
                                 "mem_total": int(f[4])})
            except ValueError:
                pass

    rep.jobs = discover_jobs(parse_ps(sec.get("PS", [])))
    attach_logs(rep.jobs, parse_logs(sec.get("LOGS", [])), now)
    if with_steps:
        fetch_steps(spec["ssh"], rep.jobs, timeout, ssh_bin)

    dd = (sec.get("DD") or ["SKIP"])[0].split()
    if dd[0] == "OK" and len(dd) > 1:
        try:
            secs = float(dd[1])
            rep.disk_mbps = 100.0 / secs if secs > 0 else None
            rep.disk_note = "OK"
        except ValueError:
            rep.disk_note = "OK"
    elif dd[0] == "FAIL":
        rep.disk_note = "FAIL"
    return rep


def load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(path: Path, reports: list[HostReport], now: float) -> None:
    state = {"_probed_at": now}
    for rep in reports:
        for j in rep.jobs:
            state[f"{rep.name}:{j.key}"] = {"step": j.step, "at": now}
    try:
        path.write_text(json.dumps(state, indent=1), encoding="utf-8")
    except OSError:                                        # pragma: no cover
        pass


def render(reports: list[HostReport]) -> str:
    ico = {"GREEN": "OK  ", "AMBER": "WARN", "RED": "RED ", "UNKNOWN": "????"}
    out = ["", "TanitAD fleet probe — discovery-based (no hardcoded targets)",
           "-" * 78,
           f"{'host':6} {'v':5} {'gpu':>16}  jobs / evidence", "-" * 78]
    for r in reports:
        gpu = ("-" if not r.gpus else
               ", ".join(f"{g['util']}% {g['mem_used']}MiB" for g in r.gpus))
        head = f"{r.name:6} {ico[r.verdict]:5} {gpu:>16}  "
        if not r.reachable:
            out.append(head + f"({r.error})")
            continue
        if not r.jobs:
            out.append(head + "no job process discovered")
        for i, j in enumerate(r.jobs):
            age = "n/a" if j.log_age_s is None else f"{j.log_age_s:.0f}s"
            out.append((head if i == 0 else " " * len(head)) +
                       f"{Path(j.key).name} pid={min(j.pids)} n={len(j.pids)} "
                       f"step={j.step} log_age={age}")
        if r.disk_mbps is not None:
            out.append(" " * len(head) + f"disk write {r.disk_mbps:.0f} MB/s")
        for f in r.findings:
            out.append(" " * 12 + f)
    worst = ("RED" if any(r.verdict == "RED" for r in reports)
             else "AMBER" if any(r.verdict == "AMBER" for r in reports)
             else "GREEN")
    out += ["-" * 78, f"FLEET: {worst}", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hosts", nargs="*", default=list(FLEET))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-dd", action="store_true",
                    help="skip the 100 MB disk write test")
    ap.add_argument("--no-steps", action="store_true",
                    help="skip the second round-trip that reads log steps")
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--ssh", default=None,
                    help="ssh binary to use (default: native OpenSSH on "
                         "win32 — see ssh_client())")
    ap.add_argument("--state", type=Path, default=STATE_DEFAULT)
    a = ap.parse_args(argv)

    prev = load_state(a.state)
    now = time.time()
    gap = now - prev["_probed_at"] if "_probed_at" in prev else None

    reports = []
    for name in a.hosts:
        if name not in FLEET:
            print(f"unknown host {name!r}; known: {', '.join(FLEET)}",
                  file=sys.stderr)
            return 2
        rep = probe_host(name, FLEET[name], not a.no_dd, a.timeout,
                         with_steps=not a.no_steps, ssh_bin=a.ssh)
        judge(rep, prev, now, gap)
        reports.append(rep)

    save_state(a.state, reports, now)
    if a.json:
        print(json.dumps({"probed_at": now, "probe_gap_s": gap,
                          "hosts": [asdict(r) for r in reports]}, indent=1))
    else:
        print(render(reports))
    if any(r.verdict == "RED" for r in reports):
        return 2
    return 1 if any(r.verdict == "AMBER" for r in reports) else 0


if __name__ == "__main__":                                 # pragma: no cover
    raise SystemExit(main())
