#!/usr/bin/env python3
"""BEHAVIOUR tests for the RL-S1 queue and the fail-closed kill monitor (2026-10-14).  No pod, no GPU, no network.  Companion of refe/test_rls1_chain.py (static properties + mutations).

  A. THE QUEUE: `log`, `refuse`, `gpu_busy`, `lock_held`, `chain_past`, `ahead`, `queue_wait` extracted VERBATIM from refe/rls1_pod.sh and run under bash with REAL advisory locks (a perl `flock`
     shim on the inherited fd, so a probe fails while another process holds the lock and succeeds after it closes it) and "other chains" that are real processes writing real chain.log markers.
     One sandbox per interleaving: S1 alive; the P2 seed replicate still WAITING (RL-S1 goes first, the replicate waits on it) / already PAST its wait (RL-S1 waits and does NOT hold the lock the
     replicate re-checks: no deadlock) / passing its wait INSIDE the settle window (RL-S1 yields the lock); a stale marker from an earlier attempt; P6 past / not past its wait; the GPU busy,
     idle, unanswerable; two chains started at once.  The invariant sampled every 0.25 s: RL-S1 never holds chain.lock while S1 is alive.
  B2. RL-S1b: the chain's argv for C1 / R0b / R1b equals eval/rls1b_arms.json token for token and the real refe_rls1b gate accepts it; the whole arm loop run for "R0b C1 R1b": order, control
     (R0b vs C0's banked dumps, R1b vs C1's), profile, every monitor READY; a K3 kill (or a K-MON) of R0b STOPS the chain with its evidence banked; a missing control is refused.
  B. FAIL CLOSED: the chain's own stage 9c (launch, supervise_arm, the rc bookkeeping) extracted verbatim and run against a fake trainer / monitor / watcher: a healthy run; the monitor SIGKILLed
     before READY (the 2026-10-14 R0 failure: rc 137, empty log), SIGKILLed after READY, alive but never READY; a monitor that applies a REAL rule (its KILLED.json is kept); every dead-monitor
     case ends in KILLED.json code K-MON, the trainer's process group dead, the arm marked killed.
  C. THE CHECKER'S MEMORY: load_scores_light == rl_val_eval.load_scores byte for byte AND flat where the original grows as N x N; load_path_cache == rl_val_eval.load_paths; the monitor prints
     ZZRLS1_MONITOR_UP before it loads anything and READY after, and a monitor that dies while loading has still said it started.
  D. MUTATIONS that must go RED: chain.lock held from the start of the queue / never re-checked once held / a waiting replicate counted as ahead / S1 not ahead / the GPU never polled /
     the fail-closed branch removed / the READY timeout removed / the original loader back.

Run: python refe/test_rls1_queue.py   -> ZZTEST_PASS <n>   (needs bash + perl on PATH; Git Bash on the dev box).  A skipped check prints ZZTEST_SKIP and is NOT counted.
"""
from __future__ import annotations

import io
import json
import contextlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKGROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PKGROOT / "eval"))
import rls1_checks as RC  # noqa: E402
from test_rls1_chain import fn_any, fn_body, posix, write_sh  # noqa: E402

CHAIN = HERE / "rls1_pod.sh"
BASH = shutil.which("bash")

PERL_FLOCK = r"""#!/usr/bin/perl
# flock(1) on an inherited fd: flock [-n] [-s|-x] FD   (a REAL advisory lock: held while any process holds the open file description)
use Fcntl qw(:flock);
my ($nb, $sh) = (0, 0); my @a;
foreach (@ARGV) { if ($_ eq "-n") { $nb = 1 } elsif ($_ eq "-s") { $sh = 1 } elsif ($_ eq "-x") { } else { push @a, $_ } }
my $fd = $a[0];
open(my $fh, "+<&=", $fd) or open($fh, ">&=", $fd) or die "flock: $fd: Bad file descriptor\n";
exit(flock($fh, ($sh ? LOCK_SH : LOCK_EX) | ($nb ? LOCK_NB : 0)) ? 0 : 1);
"""
NVSMI = '#!/bin/bash\n[ -e "$FAKE_GPU_BROKEN" ] && exit 9\n[ -s "$FAKE_GPU_FILE" ] && cat "$FAKE_GPU_FILE"\nexit 0\n'
PS_SHIM = '#!/bin/bash\nif [ "$1" = -o ] && [ "$2" = "pgid=" ]; then cat "/proc/$3/pgid" 2> /dev/null || /bin/ps -o pgid= "$3"; else exec /bin/ps "$@"; fi\n'


def perl_ok():
    return shutil.which("perl") is not None and BASH is not None


def shims(tmp):
    b = Path(tmp) / "bin"
    b.mkdir(parents=True, exist_ok=True)
    write_sh(b / "flock", PERL_FLOCK)
    write_sh(b / "nvidia-smi", NVSMI)
    write_sh(b / "ps", PS_SHIM)
    return b


def helpers(text):
    return "\n".join(re.findall(r"^(?:log|refuse)\(\) .*$", text, re.M))


# ------------------------------------------------------------------------------------------------------------------------------- A. the queue
class Queue:
    """One sandbox: a DATA dir, the shims, the chain's own queue functions, other chains as real processes."""

    def __init__(self, text, tmp, name, env=None):
        self.d = Path(tmp) / name
        shutil.rmtree(self.d, ignore_errors=True)
        self.data = self.d / "data"
        self.R = self.data / "rls1"
        self.R.mkdir(parents=True)
        self.bin = shims(self.d)
        self.gpu = self.d / "gpu.txt"
        self.gpu.write_text("")
        self.env = {**os.environ, "PATH": posix(self.bin) + os.pathsep + os.environ["PATH"], "FAKE_GPU_FILE": posix(self.gpu), "FAKE_GPU_BROKEN": posix(self.d / "no_such_file"),
                    "RLS1_QUEUE_POLL_S": "1", "RLS1_IDLE_POLL_S": "1", "RLS1_IDLE_POLLS": "1", "RLS1_SETTLE_S": "1", "RLS1_WAIT_MAX_S": "40", **(env or {})}
        self.text = text
        self.procs = []
        fns = "\n".join(fn_any(text, n) for n in ("gpu_busy", "lock_held", "chain_past", "ahead", "queue_wait"))
        self.script = self.d / "queue.sh"
        write_sh(self.script, f'set -u\nDATA={posix(self.data)}; R=$DATA/rls1; P2=$DATA/refe_p2; S1=$DATA/s1aux; P2REP=$DATA/refe_p2rep; P6D=$DATA/refe_p6; L=$R/chain.log\n'
                              f'{helpers(text)}\n{fns}\n'
                              'exec 199>"$R/chain.queue"\nflock -n 199 || { echo INSTANCE_REFUSED > "$R/queue.rc"; exit 9; }\n'
                              'queue_wait; rc=$?\necho "$rc $(date +%s.%N)" > "$R/queue.rc"\nsleep "${HOLD:-0}"\n')
        self.samples = self.d / "samples.txt"

    def start(self, hold=40):
        p = subprocess.Popen([BASH, posix(self.script)], env={**self.env, "HOLD": str(hold)}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.procs.append(p)
        return p

    def other(self, name, body, keep=True):
        """another chain: takes <data>/<name>/chain.lock for real, then runs `body` (bash).  Helper fns: mark <text> appends a timestamped line to its chain.log."""
        d = self.data / name
        d.mkdir(parents=True, exist_ok=True)
        sc = self.d / f"other_{name}.sh"
        write_sh(sc, f'D={posix(d)}\nmark() {{ echo "$(date -u +%FT%T) $*" >> "$D/chain.log"; }}\nprobe_rls1() {{ ( flock -n 9 ) 9<> {posix(self.R)}/chain.lock; }}\n'
                     f'exec 8>"$D/chain.lock"\nflock -n 8 || exit 9\n{body}\n')
        p = subprocess.Popen([BASH, posix(sc)], env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.procs.append(p)
        return p

    def observe(self, seconds, s1="s1aux"):
        """a bash sampler: every 0.25 s whether S1's lock and OUR chain.lock are held (a probe fails = held)."""
        sc = self.d / "observe.sh"
        write_sh(sc, f'end=$((SECONDS + {seconds}))\nwhile [ $SECONDS -lt $end ]; do\n  a=0; ( flock -n 9 ) 9<> {posix(self.data)}/{s1}/chain.lock 2> /dev/null || a=1\n'
                     f'  o=0; ( flock -n 9 ) 9<> {posix(self.R)}/chain.lock 2> /dev/null || o=1\n  echo "s1=$a ours=$o" >> {posix(self.samples)}\n  sleep 0.25\ndone\n')
        (self.data / s1).mkdir(parents=True, exist_ok=True)
        (self.data / s1 / "chain.lock").touch()
        self.R.joinpath("chain.lock").touch()
        p = subprocess.Popen([BASH, posix(sc)], env=self.env)
        self.procs.append(p)
        return p

    def rc(self, timeout=60):
        f = self.R / "queue.rc"
        t0 = time.time()
        while time.time() - t0 < timeout:
            if f.exists() and f.read_text().strip():
                return f.read_text().split()
            time.sleep(0.2)
        return None

    def ours_held(self):
        return subprocess.run([BASH, "-c", f'( flock -n 9 ) 9<> {posix(self.R)}/chain.lock'], env=self.env).returncode != 0

    def log(self):
        f = self.R / "chain.log"
        return f.read_text() if f.exists() else ""

    def close(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
        for p in self.procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass


def queue_scenarios(text, tmp, only=None):
    """-> number of assertions.  `only` = a set of scenario names (the mutation runs use a subset)."""
    n = 0
    want = lambda k: only is None or k in only                                                            # noqa: E731

    def invariant(q):
        rows = q.samples.read_text().split() if q.samples.exists() else []
        assert rows, "the sampler wrote nothing"
        bad = [r for r in rows if r == "s1=1" or r == "ours=1"]
        both = [i for i in range(0, len(rows) - 1, 2) if rows[i] == "s1=1" and rows[i + 1] == "ours=1"]
        assert not both, f"RL-S1 held chain.lock while S1 was alive in {len(both)} samples"
        return len(rows)

    if want("s1"):                                                                                         # S1 alive for 4 s: RL-S1 holds NOTHING it reads, then takes it
        q = Queue(text, tmp, "q_s1")
        try:
            q.other("s1aux", "mark ZZS1AUX_START; sleep 4; mark ZZS1AUX_DONE")
            q.observe(14)
            time.sleep(1.0)
            t_start = time.time()
            q.start()
            r = q.rc()
            assert r and r[0] == "0", (r, q.log())
            assert time.time() - t_start >= 3.5, "returned before S1 was over"
            lg = q.log()
            assert "ZZRLS1_QUEUED the S1 chain holds" in lg and "ZZRLS1_QUEUE_LOCK" in lg and lg.index("ZZRLS1_QUEUED") < lg.index("ZZRLS1_QUEUE_LOCK"), lg
            time.sleep(0.5)
            assert invariant(q) > 10 and q.ours_held(), "after the queue cleared RL-S1 must HOLD chain.lock"; n += 3
        finally:
            q.close()
    if want("replicate_waiting"):                                                                          # the replicate is still in its first wait: RL-S1 goes first, the replicate waits on it
        q = Queue(text, tmp, "q_rep_wait")
        try:
            rp = q.other("refe_p2rep", "mark ZZP2REP_START chain; mark 'ZZP2REP_WAITING upstream'; sleep 25")
            time.sleep(1.0)
            q.start()
            r = q.rc(30)
            assert r and r[0] == "0" and rp.poll() is None, (r, q.log())
            assert q.ours_held(), "the replicate (still waiting) reads chain.lock as 'RL-S1 holds the pod'"
            assert "ZZRLS1_QUEUED" not in q.log() and "ZZRLS1_QUEUE_YIELD" not in q.log(), q.log(); n += 3
        finally:
            q.close()
    if want("replicate_past"):                                                                             # past its wait: RL-S1 waits, holds nothing, the replicate's 2nd wait_clear is not blocked
        q = Queue(text, tmp, "q_rep_past")
        try:
            body = ('mark ZZP2REP_START chain; mark "ZZP2REP_WAIT_OK (upstream over, GPU idle)"; sleep 2\n'
                    'for i in $(seq 1 40); do probe_rls1 && break; sleep 0.5; done\n'                         # P2REP's wait_clear before train: the RL-S1 lock must be FREE
                    'probe_rls1 && mark ZZP2REP_TRAIN_START || mark ZZP2REP_DEADLOCK\nsleep 3\ndate +%s.%N > "$D/exited"\n')
            rp = q.other("refe_p2rep", body)
            time.sleep(1.0)
            q.start()
            r = q.rc(45)
            assert r and r[0] == "0", (r, q.log())
            rlog = (q.data / "refe_p2rep" / "chain.log").read_text()
            assert "ZZP2REP_TRAIN_START" in rlog and "DEADLOCK" not in rlog, rlog
            assert float(r[1]) >= float((q.data / "refe_p2rep" / "exited").read_text()) - 0.05, "RL-S1 returned while the replicate was still training"
            assert "ZZRLS1_QUEUED the P2 seed replicate is past its wait" in q.log() and q.ours_held(); n += 4
        finally:
            q.close()
    if want("replicate_inside_settle"):                                                                    # RL-S1 took the lock; the replicate passes its wait INSIDE the settle window
        q = Queue(text, tmp, "q_rep_race", {"RLS1_SETTLE_S": "5"})
        try:
            body = ('mark ZZP2REP_START chain; sleep 2.5; mark "ZZP2REP_WAIT_OK (upstream over, GPU idle)"; sleep 1.5\n'
                    'for i in $(seq 1 40); do probe_rls1 && break; sleep 0.5; done\n'
                    'probe_rls1 && mark ZZP2REP_TRAIN_START || mark ZZP2REP_DEADLOCK\nsleep 2\ndate +%s.%N > "$D/exited"\n')
            q.other("refe_p2rep", body)
            time.sleep(1.0)
            q.start()
            r = q.rc(60)
            assert r and r[0] == "0", (r, q.log())
            lg = q.log()
            assert "ZZRLS1_QUEUE_LOCK" in lg and "ZZRLS1_QUEUE_YIELD chain.lock released: the P2 seed replicate is past its wait" in lg, lg
            assert lg.count("ZZRLS1_QUEUE_LOCK") >= 2, "after the replicate the lock must be taken again"
            rlog = (q.data / "refe_p2rep" / "chain.log").read_text()
            assert "ZZP2REP_TRAIN_START" in rlog and "DEADLOCK" not in rlog, rlog; n += 4
        finally:
            q.close()
    if want("both"):                                                                                       # S1 alive AND a replicate waiting: S1 first, then RL-S1, the replicate still waiting
        q = Queue(text, tmp, "q_both")
        try:
            q.other("s1aux", "mark ZZS1AUX_START; sleep 3; mark ZZS1AUX_DONE")
            rp = q.other("refe_p2rep", "mark ZZP2REP_START chain; sleep 30")
            q.observe(12)
            time.sleep(1.0)
            q.start()
            r = q.rc(30)
            assert r and r[0] == "0" and rp.poll() is None and q.ours_held(), (r, q.log())
            invariant(q); n += 3
        finally:
            q.close()
    if want("stale_marker"):                                                                               # a WAIT_OK from an EARLIER attempt does not count; a dead chain's does not either
        q = Queue(text, tmp, "q_stale")
        try:
            (q.data / "refe_p2rep").mkdir(parents=True)
            (q.data / "refe_p2rep" / "chain.log").write_text("t ZZP2REP_START chain\nt ZZP2REP_WAIT_OK x\nt ZZP2REP_START chain\nt ZZP2REP_WAITING y\n")
            q.other("refe_p2rep", "sleep 25")
            time.sleep(1.0)
            q.start()
            r = q.rc(30)
            assert r and r[0] == "0" and "ZZRLS1_QUEUED" not in q.log(), (r, q.log()); n += 2          # the CURRENT attempt is still waiting: RL-S1 goes first
        finally:
            q.close()
        q = Queue(text, tmp, "q_dead")
        try:
            (q.data / "refe_p2rep").mkdir(parents=True)
            (q.data / "refe_p2rep" / "chain.log").write_text("t ZZP2REP_START chain\nt ZZP2REP_WAIT_OK x\n")
            (q.data / "refe_p2rep" / "chain.lock").touch()                                                  # the file is there, nobody holds it
            q.start()
            r = q.rc(30)
            assert r and r[0] == "0" and "ZZRLS1_QUEUED" not in q.log(), (r, q.log()); n += 2
        finally:
            q.close()
    if want("p6"):
        for name, body, waits in (("past", 'mark "chain start (p6_pod.sh md5 abc)"; mark "ZZP6_WAIT_OK (no other chain holds the pod)"; sleep 5', True),
                                  ("not_past", 'mark "chain start (p6_pod.sh md5 abc)"; mark "ZZP6_WAITING x"; sleep 20', False)):
            q = Queue(text, tmp, f"q_p6_{name}")
            try:
                rp = q.other("refe_p6", body)
                time.sleep(1.0)
                q.start()
                r = q.rc(30)
                assert r and r[0] == "0", (r, q.log())
                assert ("ZZRLS1_QUEUED P6 is past its wait" in q.log()) == waits and (rp.poll() is not None) == waits, (name, q.log()); n += 2
            finally:
                q.close()
    if want("gpu"):
        q = Queue(text, tmp, "q_gpu", {"RLS1_IDLE_POLLS": "3", "RLS1_SETTLE_S": "0"})                  # busy for ~4 s, then idle: needs 3 consecutive idle polls
        try:
            q.gpu.write_text("4242\n")
            t0 = time.time()
            q.start()
            time.sleep(4)
            q.gpu.write_text("")
            r = q.rc(30)
            assert r and r[0] == "0" and time.time() - t0 >= 4 + 1.5, (r, time.time() - t0)           # at least 2 more idle polls after it cleared
            n += 1
        finally:
            q.close()
        q = Queue(text, tmp, "q_gpu_broken", {"RLS1_WAIT_MAX_S": "5", "RLS1_SETTLE_S": "0"})            # nvidia-smi cannot answer: busy, forever; the wait is bounded and REFUSES
        try:
            (q.d / "broken").write_text("")
            q.env["FAKE_GPU_BROKEN"] = posix(q.d / "broken")
            q.start()
            time.sleep(9)
            assert "REFUSED: waited" in q.log() and "GPU" in q.log() and not (q.R / "queue.rc").exists(), q.log(); n += 1
        finally:
            q.close()
    if want("instance"):                                                                                    # a second chain while the first is queued: refused by chain.queue
        q = Queue(text, tmp, "q_inst")
        try:
            q.other("s1aux", "sleep 20")
            time.sleep(1.0)
            q.start()
            time.sleep(2.5)
            p2 = q.start()
            p2.wait(timeout=20)
            assert p2.returncode == 9 and (q.R / "queue.rc").read_text().strip().startswith("INSTANCE_REFUSED"), (p2.returncode, q.log()); n += 1
        finally:
            q.close()
    return n


# ------------------------------------------------------------------------------------------------------------------------------- B. fail closed
FAKEPY = r"""#!/bin/bash
# a stand-in for the venv python: routes the scripts the arm stages launch
script="$(basename "$1")"; shift
args=("$@")
arg() { local i; for ((i = 0; i < ${#args[@]}; i++)); do [ "${args[$i]}" = "$1" ] && { echo "${args[$((i + 1))]}"; return; }; done; }
echo "$script $*" >> "${FAKE_CALLS:-/dev/null}"
ad="$(arg --out)"; ad="${ad:-${ARMDIR:-.}}"
case "$script" in
  scorer_finetune.py)
    case " $* " in *" --preflight "*) echo "ZZSFT_PREFLIGHT_PASS"; exit 0 ;; esac
    mkdir -p "$ad/scores"; echo $$ > "$ad/trainer.pid"
    for k in A B base; do touch "$ad/scores/scores_update1000_$k.npz"; done; touch "$ad/model_sft_A.pt" "$ad/model_sft_B.pt"
    case "${FAKE_TRAINER:-finish}" in
      finish) sleep 4; echo "ZZSFT_DONE 1000 updates" ;;
      long) sleep 120 & echo $! > "$ad/trainer.child"; wait ;;
    esac ;;
  launch_gate.py)
    case "$1" in run) d="$(arg --out-dir)"; mkdir -p "$d"; echo '{}' > "$d/PASS_fake.json" ;; esac; exit 0 ;;
  hf_egress.py)
    case "$1" in
      watch) halt="$(arg --halt-pid)"; while kill -0 "$halt" 2> /dev/null; do sleep 0.4; done; echo "ZZEGRESS_WATCH_END ok=True" ;;
      push) echo "$(date -u +%FT%T)Z ZZEGRESS_OK $(arg --tag) 1 files 1 B" >> "$(arg --chain-log)" ;;
    esac ;;
  rl_val_eval.py) echo "ZZRLVAL_OK" ;;
  rls1_checks.py)
    case "$1" in
      updates-check) echo "ZZRLS1_UPDATES_OK 1000 updates"; exit 0 ;;
      replay) echo "ZZRLS1_REPLAY_CLEAN $(arg --arm)"; exit 0 ;;
    esac
    echo $$ > "$ad/monitor.pid"
    tp="$(arg --pid)"; marm="$(arg --arm)"; mode_var="FAKE_MONITOR_$marm"; mode="${!mode_var:-${FAKE_MONITOR:-ok}}"
    case "$mode" in
      ok) echo "ZZRLS1_MONITOR_UP $marm pid=$$"; sleep 1; echo "ZZRLS1_MONITOR_READY rss_mb=50"; while kill -0 "$tp" 2> /dev/null; do sleep 0.4; done; exit 0 ;;
      die_early) echo "ZZRLS1_MONITOR_UP $marm pid=$$"; sleep 1; kill -KILL $$ ;;
      die_after_ready) echo "ZZRLS1_MONITOR_UP $marm pid=$$"; echo "ZZRLS1_MONITOR_READY rss_mb=50"; sleep 2; kill -KILL $$ ;;
      no_ready) echo "ZZRLS1_MONITOR_UP $marm pid=$$"; sleep 120 ;;
      kill_failed) echo "ZZRLS1_MONITOR_UP $marm pid=$$"; echo "ZZRLS1_MONITOR_READY rss_mb=50"; sleep 1.5
                 echo '{"arm": "'$marm'", "code": "K3", "why": "g3/g5 bound breached", "at": "t"}' > "$ad/KILLED.json"; exit 2 ;;
      kill_rule) echo "ZZRLS1_MONITOR_UP $marm pid=$$"; echo "ZZRLS1_MONITOR_READY rss_mb=50"; sleep 1.5
                 echo '{"arm": "'$marm'", "code": "K3", "why": "g3/g5 bound breached", "at": "t"}' > "$ad/KILLED.json"; kill -TERM -- "-$tp"; sleep 1; exit 2 ;;
    esac ;;
esac
"""


def kmon_run(text, tmp, trainer, monitor, extra_env=None, name=None, timeout=40):
    """Run the chain's stage 9c (extracted verbatim) for arm R0 against the fakes.  -> dict of what happened."""
    d = Path(tmp) / (name or f"k_{trainer}_{monitor}")
    shutil.rmtree(d, ignore_errors=True)
    w = d / "w"
    (w / "pkg" / "refe").mkdir(parents=True)
    R = w / "data" / "rls1"
    ad = R / "arms" / "R0"
    (R / "stage").mkdir(parents=True)
    ad.mkdir(parents=True)
    b = shims(d)
    write_sh(b / "fakepy", FAKEPY)
    blk = re.search(r"(?s)  # ---- 9c\..*?(?=  # ---- 9d\.)", text)
    assert blk, "stage 9c not found"
    fns = "\n".join(fn_any(text, n) for n in ("pid_alive", "stop_trainer", "supervise_arm"))
    sc = d / "stage9c.sh"
    write_sh(sc, f'set -u\nR={posix(R)}; S=$R/stage; L=$R/chain.log; ARM=R0; ARMDIR={posix(ad)}; LG=$R/lg/R0; ARGS="--x 1"; PKG={posix(w / "pkg")}; PY={posix(b / "fakepy")}\n'
                 f'OUT_REPO=Sayood/x; REFDIR=$ARMDIR/scores; PDM_VA=p; TERMS_VA=t; SETS_VA=s; VAL_TSV=v; export ARMDIR\n'
                 f'{helpers(text)}\n' + "\n".join(re.findall(r"^(?:stage_done|stage_mark)\(\) .*$", text, re.M)) + f'\n{fns}\n'
                 'token_ok() { return 0; }\ngpu_idle_wait() { return 0; }\nTICKPID=\ncd "$PKG/refe"\n' + blk.group(0) + 'echo BLOCK_END\n')
    env = {**os.environ, "PATH": posix(b) + os.pathsep + os.environ["PATH"], "FAKE_TRAINER": trainer, "FAKE_MONITOR": monitor, "RLS1_SUP_POLL_S": "1", "RLS1_LAUNCH_SETTLE_S": "1",
           "RLS1_STOP_GRACE_S": "5", "RLS1_MON_READY_S": "60", "HF_TOKEN": "hf_notarealtoken", **(extra_env or {})}
    t0 = time.time()
    try:
        r = subprocess.run([BASH, posix(sc)], capture_output=True, text=True, env=env, timeout=timeout)
        rc, out = r.returncode, r.stdout + r.stderr
        timed_out = False
    except subprocess.TimeoutExpired as e:
        rc, out, timed_out = -9, str(e), True
    res = {"rc": rc, "out": out, "seconds": time.time() - t0, "timed_out": timed_out, "log": (R / "chain.log").read_text() if (R / "chain.log").exists() else "",
           "killed": json.loads((ad / "KILLED.json").read_text()) if (ad / "KILLED.json").exists() else None,
           "stages": sorted(p.name for p in (R / "stage").iterdir()), "marker": (R / "stage" / "arm_R0_killed").read_text() if (R / "stage" / "arm_R0_killed").exists() else ""}

    def alive(f):
        if not (ad / f).exists():
            return False
        pid = (ad / f).read_text().strip()
        return subprocess.run([BASH, "-c", f'kill -0 {pid} 2> /dev/null && [ "$(sed -n "s/^.*) \\(.\\).*$/\\1/p" /proc/{pid}/stat 2> /dev/null)" != Z ]']).returncode == 0
    res["trainer_alive"], res["trainer_child_alive"], res["monitor_alive"] = alive("trainer.pid"), alive("trainer.child"), alive("monitor.pid")
    if (ad / "trainer.pid").exists():
        subprocess.run([BASH, "-c", f"kill -KILL -- -{(ad / 'trainer.pid').read_text().strip()} 2> /dev/null"])
    for f in ("trainer.child", "monitor.pid"):                                                              # never leave a sleeper behind a failed assertion
        if (ad / f).exists():
            subprocess.run([BASH, "-c", f"kill -KILL {(ad / f).read_text().strip()} 2> /dev/null"])
    return res


def kmon_scenarios(text, tmp, only=None):
    n = 0
    want = lambda k: only is None or k in only                                                              # noqa: E731
    if want("healthy"):
        r = kmon_run(text, tmp, "finish", "ok")
        assert r["rc"] == 0 and "BLOCK_END" in r["out"] and r["killed"] is None and "train_R0.ok" in r["stages"] and "arm_R0_killed" not in r["stages"], r
        assert "kill monitor READY" in r["log"] and "trainer rc 0, watcher rc 0, monitor rc 0" in r["log"], r["log"]; n += 3
    if want("die_early"):                                                                                    # the 2026-10-14 failure: SIGKILL before the first line, the trainer alive
        r = kmon_run(text, tmp, "long", "die_early")
        assert not r["timed_out"] and r["killed"] and r["killed"]["code"] == "K-MON" and "monitor died rc 137" in r["killed"]["why"], r
        assert r["killed"]["why"].startswith("kill rules unenforceable: monitor died rc"), r["killed"]
        assert not r["trainer_alive"] and not r["trainer_child_alive"], "the trainer's process group must be dead"
        assert "arm_R0_killed" in r["stages"] and "K-MON" in r["marker"] and "train_R0.ok" not in r["stages"], r
        assert "ZZRLS1_KILL R0 K-MON" in r["log"] and "ZZRLS1_ARM_KILLED R0" in r["log"] and r["seconds"] < 30, r["log"]; n += 6
    if want("die_after_ready"):
        r = kmon_run(text, tmp, "long", "die_after_ready")
        assert not r["timed_out"] and r["killed"] and r["killed"]["code"] == "K-MON" and "monitor died rc 137" in r["killed"]["why"], r
        assert "kill monitor READY" in r["log"] and not r["trainer_alive"] and not r["trainer_child_alive"], r; n += 3
    if want("no_ready"):
        r = kmon_run(text, tmp, "long", "no_ready", {"RLS1_MON_READY_S": "4"})
        assert not r["timed_out"] and r["killed"] and r["killed"]["code"] == "K-MON" and "monitor not READY within 4s" in r["killed"]["why"], r
        assert not r["trainer_alive"] and not r["trainer_child_alive"] and not r["monitor_alive"], "trainer AND the hung monitor must be dead"; n += 3
    if want("real_rule"):                                                                                    # the monitor's own K3 stands; K-MON never overwrites it
        r = kmon_run(text, tmp, "long", "kill_rule")
        assert not r["timed_out"] and r["killed"] and r["killed"]["code"] == "K3" and "K-MON" not in json.dumps(r["killed"]), r
        assert "arm_R0_killed" in r["stages"] and "K3" in r["marker"] and not r["trainer_alive"], r; n += 3
    if want("kill_failed"):                                                                                  # the monitor decided K3 but its kill did not land: the chain stops the trainer, the K3 stands
        r = kmon_run(text, tmp, "long", "kill_failed")
        assert not r["timed_out"] and r["killed"] and r["killed"]["code"] == "K3" and "K-MON" not in json.dumps(r["killed"]), r
        assert not r["trainer_alive"] and not r["trainer_child_alive"] and "arm_R0_killed" in r["stages"] and "K3" in r["marker"], r; n += 3
    return n


# ------------------------------------------------------------------------------------------------------------------------------- B2. RL-S1b: argv, profile, order, the R0b stop
def armloop_run(text, tmp, name, arms, env=None, prepare=None, timeout=150):
    """The chain's whole arm loop (stage 9, from `export OMP_NUM_THREADS=4` to its end) extracted verbatim and run for `arms` against the fakes.  -> dict."""
    d = Path(tmp) / name
    shutil.rmtree(d, ignore_errors=True)
    w = d / "w"
    (w / "refe-plan-rls1" / "refe").mkdir(parents=True)
    R = w / "data" / "rls1"
    (R / "stage").mkdir(parents=True)
    b = shims(d)
    write_sh(b / "fakepy", FAKEPY)
    hdr = "\n".join(ln for ln in text[:text.index('mkdir -p "$R" "$S"')].split("\n") if re.match(r"^[A-Z0-9_]+=", ln) and not ln.startswith("export") and "$(" not in ln and "FINAL_SHA256" not in ln
                    and not ln.startswith(("NSH", "PY="))).replace("WORK=/workspace", f"WORK={posix(w)}")
    loop = text[text.index("export OMP_NUM_THREADS=4\n"):]
    assert loop.count("export OMP_NUM_THREADS=4\n") == 1
    fns = "\n".join(fn_any(text, n) for n in ("arm_args", "arm_profile", "arm_registered", "token_ok", "pid_alive", "stop_trainer", "supervise_arm"))
    sc = d / "armloop.sh"
    write_sh(sc, f'set -u\n{hdr}\nPY={posix(b / "fakepy")}; NTRAIN=16000; TICKPID=; export PKG\n{helpers(text)}\n'
                 + "\n".join(re.findall(r"^(?:stage_done|stage_mark|stop_after)\(\) .*$", text, re.M)) + f'\n{fns}\ngpu_idle_wait() {{ return 0; }}\ncd "$PKG/refe"\n{loop}')
    if prepare:
        prepare(R)
    e = {**os.environ, "PATH": posix(b) + os.pathsep + os.environ["PATH"], "FAKE_CALLS": posix(d / "calls.log"), "RLS1_SUP_POLL_S": "1", "RLS1_LAUNCH_SETTLE_S": "1", "RLS1_STOP_GRACE_S": "5",
         "RLS1_MON_READY_S": "60", "HF_TOKEN": "hf_notarealtoken", "RLS1_ARMS": arms, **(env or {})}
    try:
        r = subprocess.run([BASH, posix(sc)], capture_output=True, text=True, env=e, timeout=timeout)
        rc, out = r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired as ex:
        rc, out = -9, str(ex)
    for pf in R.glob("arms/*/trainer.pid"):                                                                  # never leave a sleeper behind a failed assertion
        subprocess.run([BASH, "-c", f"kill -KILL -- -{pf.read_text().strip()} 2> /dev/null"])
    log = (R / "chain.log").read_text() if (R / "chain.log").exists() else ""
    calls = (d / "calls.log").read_text() if (d / "calls.log").exists() else ""
    return {"rc": rc, "out": out, "log": log, "calls": calls, "R": R, "done": re.findall(r"ZZRLS1_ARM_DONE (\S+)", log),
            "arms_dirs": sorted(p.name for p in (R / "arms").iterdir()) if (R / "arms").exists() else []}


def rls1b_checks(text, tmp, only=None):
    import shlex
    import launch_gate as LGM
    from test_rls1_chain import arm_args_runner
    n = 0
    want = lambda k: only is None or k in only                                                              # noqa: E731
    arms_json = json.loads((PKGROOT / "eval" / "rls1b_arms.json").read_text(encoding="utf-8"))["arms"]
    if want("argv"):
        for arm in ("C1", "R0b", "R1b"):                                                                    # the chain's argv IS the other agent's, token for token, and the REAL refe_rls1b gate accepts it
            got = arm_args_runner(text, Path(tmp), arm)
            assert shlex.split(got) == arms_json[arm]["argv"], (arm, got)
            prof = {**LGM.PROFILES["refe_rls1b"], "declared_arm": arm}
            assert LGM.profile_refusals(prof, LGM.parse_args_string(got)) == [], (arm, LGM.profile_refusals(prof, LGM.parse_args_string(got))); n += 2
        bad = arm_args_runner(text, Path(tmp), "R0b").replace("--epo-lam 0.25", "--epo-lam 0.5")
        assert LGM.profile_refusals({**LGM.PROFILES["refe_rls1b"], "declared_arm": "R0b"}, LGM.parse_args_string(bad)); n += 1     # the gate would refuse another dose
        pr = Path(tmp) / "ap.sh"
        write_sh(pr, fn_any(text, "arm_profile") + '\narm_profile "$1"\n')
        for arm, arms, want_p in (("R0b", "", "refe_rls1b"), ("R1b", "", "refe_rls1b"), ("C1", "R0b C1 R1b", "refe_rls1b"), ("C1", "C0 R0 C1 R1 R2", "refe_rls1"), ("C1", "", "refe_rls1"),
                                  ("C0", "R0b C1 R1b", "refe_rls1"), ("R0", "", "refe_rls1")):
            assert subprocess.run([BASH, posix(pr), arm], capture_output=True, text=True, env={**os.environ, "RLS1_ARMS": arms}).stdout.strip() == want_p, (arm, arms); n += 1

    def ctrl(R):                                                                                            # C0's banked dumps stay on the pod
        (R / "arms" / "C0" / "scores").mkdir(parents=True)
        (R / "arms" / "C0" / "scores" / "scores_update1000_A.npz").write_text("c0")
    if want("order"):                                                                                       # R0b (vs C0) -> C1 -> R1b (vs C1), each under refe_rls1b, every monitor READY
        r = armloop_run(text, tmp, "al_order", "R0b C1 R1b", prepare=ctrl)
        assert r["rc"] == 0 and r["done"] == ["R0b", "C1", "R1b"] and "ZZRLS1_DONE arms: R0b C1 R1b" in r["log"], (r["rc"], r["done"], r["log"][-1500:], r["out"][-500:])
        ctl = {a: re.search(rf"arm {a}: trainer pid/pgid .*\(control: ([^)]*)\)", r["log"]).group(1) for a in ("R0b", "C1", "R1b")}
        assert ctl["R0b"].endswith("/arms/C0/scores") and ctl["R1b"].endswith("/arms/C1/scores") and ctl["C1"].endswith("/arms/C1/scores"), ctl           # R0b vs C0, R1b vs C1
        runs = re.findall(r"launch_gate\.py run --profile (\S+) --arm-name (\S+)", r["calls"])
        assert runs == [("refe_rls1b", "R0b"), ("refe_rls1b", "C1"), ("refe_rls1b", "R1b")], runs
        assert all(f"arm {a}: kill monitor READY" in r["log"] for a in ("R0b", "C1", "R1b")); n += 6
    if want("r0b_kill"):                                                                                    # a K3 kill of R0b STOPS the chain: no C1, no R1b, the Master Mind decides
        r = armloop_run(text, tmp, "al_r0b_kill", "R0b C1 R1b", env={"FAKE_TRAINER": "long", "FAKE_MONITOR_R0b": "kill_rule"}, prepare=ctrl)
        assert r["rc"] == 1 and r["done"] == [] and "ZZRLS1_ARM_KILLED R0b" in r["log"] and "REFUSED: arm R0b was KILLED" in r["log"] and "Master Mind decides" in r["log"], (r["rc"], r["log"][-1500:])
        assert r["arms_dirs"] == ["C0", "R0b"] and not re.search(r"arm (C1|R1b):", r["log"]) and "ZZRLS1_DONE" not in r["log"], (r["arms_dirs"], r["log"])
        assert (r["R"] / "stage" / "egress_R0b.ok").exists() and "K3" in (r["R"] / "stage" / "arm_R0b_killed").read_text(), "the killed arm's evidence is banked first"; n += 4
    if want("r0b_kmon"):                                                                                    # the R0b monitor is SIGKILLed (the 2026-10-14 failure): K-MON, the chain stops there too
        r = armloop_run(text, tmp, "al_r0b_kmon", "R0b C1 R1b", env={"FAKE_TRAINER": "long", "FAKE_MONITOR_R0b": "die_early"}, prepare=ctrl)
        assert r["rc"] == 1 and "K-MON" in (r["R"] / "stage" / "arm_R0b_killed").read_text() and r["arms_dirs"] == ["C0", "R0b"] and "ZZRLS1_DONE" not in r["log"], (r["rc"], r["log"][-1200:]); n += 3
    if want("no_control"):                                                                                  # R1b without C1's dumps is REFUSED, never read against its own arm A
        r = armloop_run(text, tmp, "al_noctl", "R1b")
        assert r["rc"] == 1 and "RL-S1b arm R1b is read against C1" in r["log"] and "launch_gate.py run" not in r["calls"], (r["rc"], r["log"]); n += 1
    return n


# ------------------------------------------------------------------------------------------------------------------------------- C. the checker's memory
def synth_scores(path, n, m=8, c=7, dup=False, space="logit"):
    import numpy as np
    rng = np.random.default_rng(7)
    keys = [(f"L{i // 4}", f"t{i:05d}", 0, 0) for i in range(n)]
    if dup:
        keys[-1] = keys[0]
    np.savez(path, log=np.asarray([k[0] for k in keys]), token=np.asarray([k[1] for k in keys]), step=np.asarray([k[2] for k in keys], np.int64),
             rank=np.asarray([k[3] for k in keys], np.int64), ckpt_step=np.asarray([10075] * n, np.int64), scores=rng.normal(size=(n, m, c)), space=np.asarray(space))


def peak_alloc(fn, *a):
    import tracemalloc
    tracemalloc.start()
    out = fn(*a)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, peak


def loader_checks(tmp, RCm=None):
    """-> number of assertions.  RCm = the module under test (the real one, or a mutated copy: then the MEMORY assertion must fail)."""
    import numpy as np
    import rl_val_eval as RV
    RCm = RCm or RC
    n = 0
    base = Path(tmp) / "ld"
    base.mkdir(parents=True, exist_ok=True)
    f = base / "s.npz"
    synth_scores(f, 240, m=64)
    old, p_old = peak_alloc(RV.load_scores, str(f))
    new, p_new = peak_alloc(RCm.load_scores_light, str(f))
    assert list(old) == list(new) and all(old[k]["ckpt"] == new[k]["ckpt"] and old[k]["space"] == new[k]["space"] and old[k]["scores"].dtype == new[k]["scores"].dtype
                                          and old[k]["scores"].tobytes() == new[k]["scores"].tobytes() for k in old); n += 1                  # same dict, same bytes
    assert p_old > 100 * (f.stat().st_size / 4), (p_old, f.stat().st_size)                                                                   # the original retains N full copies (N x N)
    assert p_new < 4 * f.stat().st_size and p_new * 20 < p_old, (p_new, p_old); n += 1                                                       # the light loader: ONE read
    synth_scores(base / "d.npz", 10, dup=True)
    for fn in (RV.load_scores, RCm.load_scores_light):                                                                                       # a duplicated key is an error in both
        try:
            fn(str(base / "d.npz"))
            raise AssertionError("duplicate key accepted")
        except ValueError as e:
            assert "duplicate scores" in str(e)
    synth_scores(base / "sp.npz", 4, space="weird")
    for fn in (RV.load_scores, RCm.load_scores_light):
        try:
            fn(str(base / "sp.npz"))
            raise AssertionError("bad space accepted")
        except ValueError as e:
            assert "space must be" in str(e)
    n += 2
    # the path-length cache == rl_val_eval.load_paths for the wanted keys (stale ckpt, duplicate line, unwanted key)
    sd = base / "sets"
    sd.mkdir(exist_ok=True)

    def line(log, tok, ck, scale):
        traj = [[[float(scale * (t + 1)), 0.0] for t in range(5)] for _ in range(3)]
        return json.dumps({"kind": "onpolicy_set", "log_name": log, "token": tok, "step": 0, "rank": 0, "ckpt_step": ck, "traj": traj, "yaw": [[0.0] * 5] * 3}) + "\n"
    (sd / "onpolicy_rlfan_a.jsonl").write_text(line("L0", "t1", 5, 1.0) + line("L0", "t2", 4, 2.0) + line("L1", "t3", 5, 3.0) + line("L0", "t1", 5, 4.0), encoding="utf-8")
    keys = [("L0", "t1", 0, 0), ("L0", "t2", 0, 0), ("L1", "t3", 0, 0), ("L9", "t9", 0, 0)]
    ck = {k: 5 for k in keys}
    want = RV.load_paths(str(sd), keys, ck)
    cache = RCm.load_path_cache(str(sd))
    got = {k: cache[(k, ck[k])] for k in keys if (k, ck[k]) in cache}
    assert set(got) == set(want) == {keys[0], keys[2]} and all(np.array_equal(got[k], want[k]) for k in want); n += 1                        # t2 is another ckpt's set, t9 not dumped, t1 -> the LAST line
    return n


def monitor_cli_checks(tmp):
    n = 0
    src = (HERE / "rls1_checks.py").read_text(encoding="utf-8")
    main_src = src[src.index("def main(argv=None):"):]
    assert main_src.index('say("ZZRLS1_MONITOR_UP"') < main_src.index("RlValChecker(a.arm") < main_src.index('say("ZZRLS1_MONITOR_READY"'); n += 1      # UP before anything is loaded, READY after
    od = Path(tmp) / "mcli"
    od.mkdir(exist_ok=True)
    dead_pid = "2147483000"                                                                                                                      # not a process: the loop ends at once
    common = ["--out", str(od), "--refdir", str(od), "--pdm", str(od), "--terms", str(od), "--sets", str(od)]
    r = subprocess.run([sys.executable, "-I", str(HERE / "rls1_checks.py"), "monitor", "--arm", "C0", *common, "--tsv", str(od / "none.tsv"), "--pid", dead_pid, "--poll", "0.1"],
                       capture_output=True, text=True, timeout=60)
    lines = r.stdout.splitlines()
    assert r.returncode == 0 and lines[0].startswith("ZZRLS1_MONITOR_UP C0 pid=") and re.fullmatch(r"ZZRLS1_MONITOR_READY rss_mb=-?\d+", lines[1]), (r.returncode, lines, r.stderr[-300:]); n += 1
    r = subprocess.run([sys.executable, "-I", str(HERE / "rls1_checks.py"), "monitor", "--arm", "R0", *common, "--tsv", str(od / "none.tsv"), "--pid", dead_pid, "--poll", "0.1"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and r.stdout.startswith("ZZRLS1_MONITOR_UP R0") and "ZZRLS1_MONITOR_READY" not in r.stdout, (r.returncode, r.stdout, r.stderr[-300:]); n += 1   # dies while loading: it still said UP
    cur, peak = RC._rss_mb()
    assert (cur, peak) == (-1, -1) or (cur > 0 and peak >= cur); n += 1
    buf = io.StringIO()                                                                                                                          # every check prints its peak RSS
    sft = od / "sft.jsonl"
    sft.write_text("".join(json.dumps(e) + "\n" for e in ({"event": "score_dump", "tag": "step0", "files": {}}, {"event": "score_dump", "tag": "update250", "files": {}})), encoding="utf-8")
    recs = {t: {"tag": t, "g3_ok": True, "g5_ok": True, "k4_breach": False, "k5_breach": False} for t in ("step0", "update250")}
    with contextlib.redirect_stdout(buf):
        RC.monitor("R0", str(od), lambda tag, files: recs[tag], 99, lambda p: False, lambda p: None, sleep_fn=lambda s: None, now_fn=lambda: 0.0, start=0.0)
    out = buf.getvalue()
    assert re.search(r"ZZRLS1_MONITOR_RSS R0 step0 rss_mb=-?\d+ peak_mb=-?\d+", out) and re.search(r"ZZRLS1_MONITOR_RSS R0 update250 rss_mb=", out), out; n += 1
    return n


def module_mutation(old, new, tag):
    src = (HERE / "rls1_checks.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, (tag, src.count(old))
    m = types.ModuleType("rls1_checks_mut")
    m.__file__ = str(HERE / "rls1_checks.py")
    sys.modules["rls1_checks_mut"] = m
    exec(compile(src.replace(old, new), m.__file__, "exec"), m.__dict__)
    return m


def must_fail(fn, name):
    try:
        fn()
    except Exception:                                                                                         # noqa: BLE001  (any failure IS the suite seeing the mutation)
        return 1
    raise AssertionError(f"mutation '{name}' stayed GREEN")


def main() -> int:
    if not perl_ok():
        print("ZZTEST_SKIP the queue and K-MON behaviour tests need bash and perl on PATH")
        return 0
    text = CHAIN.read_text(encoding="utf-8")
    n = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # ---- A, B: the real interleavings and the fail-closed cases
        n += queue_scenarios(text, tmp)
        n += kmon_scenarios(text, tmp)
        n += rls1b_checks(text, tmp)
        # ---- C: the checker's memory and its start-up markers
        n += loader_checks(tmp)
        n += monitor_cli_checks(tmp)
        # ---- D: mutations that must go RED (each runs the scenario that exists for it)
        sw = lambda a, b: (text.replace(a, b) if text.count(a) == 1 else (_ for _ in ()).throw(AssertionError(f"mutation anchor x{text.count(a)}: {a[:60]}")))  # noqa: E731
        muts = [
            ("chain.lock is held from the start of the queue and never given back",
             sw("local t0=$SECONDS held=0", 'exec 200>"$R/chain.lock"; flock -n 200; local t0=$SECONDS held=1').replace('if [ "$held" = 1 ]; then exec 200>&-; held=0;', 'if false; then exec 200>&-; held=0;'),
             {"s1", "replicate_past"}),
            ("the lock is never re-checked once held", sw('if why="$(ahead)"; then', 'if [ "$held" = 0 ] && why="$(ahead)"; then'), {"replicate_inside_settle"}),
            ("a WAITING replicate counts as ahead", sw('chain_past "$P2REP" "ZZP2REP_START" "ZZP2REP_WAIT_OK"', 'lock_held "$P2REP"'), {"replicate_waiting"}),
            ("S1 is not ahead", sw('  if lock_held "$S1"; then echo "the S1 chain holds $S1/chain.lock"; return 0; fi\n', ""), {"s1"}),
            ("the past-wait test ignores the attempt boundary", sw("index($0, s) { ok = 0 } ", ""), {"stale_marker"}),
            ("the GPU is never polled", sw("if gpu_busy; then idle=0; why=", "if false; then idle=0; why="), {"gpu"}),
        ]
        for name, mut, only in muts:
            n += must_fail(lambda mut=mut, only=only: queue_scenarios(mut, tmp, only), name)
        kmuts = [
            ("the fail-closed branch is removed", sw('if pid_alive "$tp"; then why="kill rules unenforceable: monitor died rc $MON_RC"; fi', ":"), {"die_early"}),
            ("the READY timeout is removed", sw('[ $((SECONDS - t0)) -ge "$ready_s" ]', "false"), {"no_ready"}),
            ("K-MON overwrites the monitor's own kill", sw('[ -f "$ad/KILLED.json" ] || printf', "printf"), {"kill_failed"}),
        ]
        for name, mut, only in kmuts:
            n += must_fail(lambda mut=mut, only=only: kmon_scenarios(mut, tmp, only), name)
        bmuts = [
            ("a killed R0b does not stop the chain", sw('  [ -f "$S/arm_${ARM}_killed" ] && refuse "arm $ARM was KILLED ($(cat "$S/arm_${ARM}_killed")): its evidence is banked; the Master Mind decides before any further arm"\n', ""), {"r0b_kill"}),
            ("R0b is read against the wrong control", sw("RD=C${ARM:1:1};;", "RD=C1;;"), {"order"}),
            ("a missing RL-S1b control is replaced by the arm's own A", sw('if [ "$PROFILE" = refe_rls1b ] && [ "$KIND" = B ]', 'if false && [ "$KIND" = B ]'), {"no_control"}),
            ("the RL-S1b arms are judged under the original profile", sw("R0b|R1b) echo refe_rls1b", "R0b|R1b) echo refe_rls1"), {"order"}),
        ]
        for name, mut, only in bmuts:
            n += must_fail(lambda mut=mut, only=only: rls1b_checks(mut, tmp, only), name)
        n += must_fail(lambda: loader_checks(tmp, module_mutation("    if not str(path).endswith(\".npz\"):", "    if True:", "original loader back")), "the original loader is back")
    print(f"ZZTEST_PASS {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
