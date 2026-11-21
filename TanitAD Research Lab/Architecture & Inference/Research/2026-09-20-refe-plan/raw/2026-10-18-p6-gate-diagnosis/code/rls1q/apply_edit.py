"""Build the corrected refe/test_rls1_queue.py from the c2d4042 blob (exact anchors, each must match once; LF kept)."""
import sys

src = open("base_test_rls1_queue.py", "rb").read().decode("utf-8")
assert "\r" not in src


def rep(old, new):
    global src
    assert src.count(old) == 1, (src.count(old), old[:80])
    src = src.replace(old, new)


# 1. other(): return only once the fake chain HOLDS its lock (optionally after a simulated slow start)
rep('''    def other(self, name, body, keep=True):
        """another chain: takes <data>/<name>/chain.lock for real, then runs `body` (bash).  Helper fns: mark <text> appends a timestamped line to its chain.log."""
        d = self.data / name
        d.mkdir(parents=True, exist_ok=True)
        sc = self.d / f"other_{name}.sh"
        write_sh(sc, f'D={posix(d)}\\nmark() {{ echo "$(date -u +%FT%T) $*" >> "$D/chain.log"; }}\\nprobe_rls1() {{ ( flock -n 9 ) 9<> {posix(self.R)}/chain.lock; }}\\n'
                     f'exec 8>"$D/chain.lock"\\nflock -n 8 || exit 9\\n{body}\\n')
        p = subprocess.Popen([BASH, posix(sc)], env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.procs.append(p)
        return p
''', '''    def other(self, name, body, keep=True, start_delay=0.0):
        """another chain: takes <data>/<name>/chain.lock for real, then runs `body` (bash).  Helper fns: mark <text> appends a timestamped line to its chain.log.
        Returns only once that chain HOLDS its lock (2026-10-18): every scenario's clock starts there, never at the Popen.  Before, a bash start slower than the scenario's 1 s head start (a
        loaded box) ran the queue against a chain that was not alive yet, and `s1` failed "returned before S1 was over" on a tree whose queue code was byte-identical to a green one
        (raw/2026-10-18-p6-gate-diagnosis/RESULT.md section 6).  `start_delay` simulates that slow start (scenario s1_slow_start)."""
        d = self.data / name
        d.mkdir(parents=True, exist_ok=True)
        sc = self.d / f"other_{name}.sh"
        held = self.d / f"other_{name}.held"                                                                 # OUTSIDE data/: no chain function reads it
        if held.exists():
            held.unlink()
        write_sh(sc, f'D={posix(d)}\\nmark() {{ echo "$(date -u +%FT%T) $*" >> "$D/chain.log"; }}\\nprobe_rls1() {{ ( flock -n 9 ) 9<> {posix(self.R)}/chain.lock; }}\\n'
                     + (f'sleep {start_delay}\\n' if start_delay else '') + f'exec 8>"$D/chain.lock"\\nflock -n 8 || exit 9\\n: > "{posix(held)}"\\n{body}\\n')
        p = subprocess.Popen([BASH, posix(sc)], env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.procs.append(p)
        t0 = time.time()
        while SANDBOX["handshake"] and not held.exists():
            if p.poll() is not None:
                assert held.exists(), f"SANDBOX, not a queue result: the fake {name} chain exited rc {p.returncode} before it held {posix(d)}/chain.lock"
                break
            assert time.time() - t0 < 60, f"SANDBOX, not a queue result: the fake {name} chain did not hold its chain.lock within 60 s"
            time.sleep(0.05)
        return p
''')

# 2. observe(): OUR lock first, then S1's liveness from its OWN log (never a probe of S1's lock)
rep('''    def observe(self, seconds, s1="s1aux"):
        """a bash sampler: every 0.25 s whether S1's lock and OUR chain.lock are held (a probe fails = held)."""
        sc = self.d / "observe.sh"
        write_sh(sc, f'end=$((SECONDS + {seconds}))\\nwhile [ $SECONDS -lt $end ]; do\\n  a=0; ( flock -n 9 ) 9<> {posix(self.data)}/{s1}/chain.lock 2> /dev/null || a=1\\n'
                     f'  o=0; ( flock -n 9 ) 9<> {posix(self.R)}/chain.lock 2> /dev/null || o=1\\n  echo "s1=$a ours=$o" >> {posix(self.samples)}\\n  sleep 0.25\\ndone\\n')
''', '''    def observe(self, seconds, s1="s1aux"):
        """a bash sampler: every 0.25 s whether OUR chain.lock is held (a probe fails = held), THEN whether S1 is alive.
        S1's witness is its OWN log, never a probe of its lock, and it is read SECOND (2026-10-18).  A probe of S1's lock also reads "held" when it collides with the queue's own ahead() probe
        of that lock, and an S1-then-ours order lets RL-S1 take its lock between the two reads just after S1 exits.  Either way a row could read "s1=1 ours=1" with no overlap; one did, on the
        pristine base tree (raw/2026-10-18-p6-gate-diagnosis/RESULT.md section 6).  S1 holds its lock from before this sampler starts (other() returns only then) until after it logs
        ZZS1AUX_DONE.  So "START logged, no DONE yet" at the second read means S1 held its lock at the first read too, and a row "s1=1 ours=1" is a REAL overlap."""
        sc = self.d / "observe.sh"
        lg = f"{posix(self.data)}/{s1}/chain.log"
        write_sh(sc, f'end=$((SECONDS + {seconds}))\\nwhile [ $SECONDS -lt $end ]; do\\n  o=0; ( flock -n 9 ) 9<> {posix(self.R)}/chain.lock 2> /dev/null || o=1\\n'
                     f'  a=0; grep -q ZZS1AUX_START "{lg}" 2> /dev/null && ! grep -q ZZS1AUX_DONE "{lg}" 2> /dev/null && a=1\\n'
                     f'  echo "s1=$a ours=$o" >> {posix(self.samples)}\\n  sleep 0.25\\ndone\\n')
''')

# 3. the s1 scenario, and the same scenario with a fake S1 that needs 4 s to reach its lock
rep('''    if want("s1"):                                                                                         # S1 alive for 4 s: RL-S1 holds NOTHING it reads, then takes it
        q = Queue(text, tmp, "q_s1")
        try:
            q.other("s1aux", "mark ZZS1AUX_START; sleep 4; mark ZZS1AUX_DONE")
''', '''    for scen, delay in (("s1", 0.0), ("s1_slow_start", 4.0)):                                             # S1 alive for 4 s: RL-S1 holds NOTHING it reads, then takes it
        if not want(scen):                                                                                 # s1_slow_start: the same, from a fake S1 whose bash needs 4 s to reach its lock
            continue
        q = Queue(text, tmp, "q_" + scen)
        try:
            q.other("s1aux", "mark ZZS1AUX_START; sleep 4; mark ZZS1AUX_DONE", start_delay=delay)
''')

# 4. the sandbox switch, next to the other module constants
rep('''CHAIN = HERE / "rls1_pod.sh"
''', '''CHAIN = HERE / "rls1_pod.sh"
SANDBOX = {"handshake": True}        # Queue.other() waits until the fake chain HOLDS its lock; False only in the mutation that proves s1_slow_start needs it
''')

# 5. the mutation: without the handshake, s1_slow_start must go RED
rep('''        for name, mut, only in muts:
            n += must_fail(lambda mut=mut, only=only: queue_scenarios(mut, tmp, only), name)
''', '''        for name, mut, only in muts:
            n += must_fail(lambda mut=mut, only=only: queue_scenarios(mut, tmp, only), name)
        SANDBOX["handshake"] = False                                                                         # the sandbox's own precondition: without it a slow fake S1 is not alive yet
        try:
            n += must_fail(lambda: queue_scenarios(text, tmp, {"s1_slow_start"}), "the sandbox starts the queue before the fake S1 holds its lock")
        finally:
            SANDBOX["handshake"] = True
''')
# 6. a fake chain that DIED is the sandbox's failure, never the queue's
rep('''        self.procs = []
''', '''        self.procs = []
        self.others = []                                                                                     # (name, Popen) of the fake chains, for sandbox_alive()
''')
rep('''        self.procs.append(p)
        t0 = time.time()
        while SANDBOX["handshake"]''', '''        self.procs.append(p)
        self.others.append((name, p))
        t0 = time.time()
        while SANDBOX["handshake"]''')
rep('''    def rc(self, timeout=60):
        f = self.R / "queue.rc"
        t0 = time.time()
        while time.time() - t0 < timeout:
            if f.exists() and f.read_text().strip():
                return f.read_text().split()
            time.sleep(0.2)
        return None
''', '''    def rc(self, timeout=60):
        f = self.R / "queue.rc"
        t0 = time.time()
        while time.time() - t0 < timeout:
            if f.exists() and f.read_text().strip():
                self.sandbox_alive()
                return f.read_text().split()
            time.sleep(0.2)
        self.sandbox_alive()
        return None

    def sandbox_alive(self):
        """A fake chain that DIED (rc != 0) before the queue answered invalidates the scenario.  It is reported as the SANDBOX's failure and never read as the queue's (2026-10-18: a fake
        chain's bash died rc 3221225773 = 0xC000012D STATUS_COMMITMENT_LIMIT on a dev box whose commit headroom and C: were exhausted)."""
        dead = [(nm, p.returncode) for nm, p in self.others if p.poll() is not None and p.returncode != 0]
        assert not dead, f"SANDBOX, not a queue result: fake chain(s) died {dead}"
''')
open(sys.argv[1], "wb").write(src.encode("utf-8"))
print("written", sys.argv[1])
