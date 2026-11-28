"""Rebuild test_p2rep_chain.py's wait_clear harness exactly (its own helpers) and run it under bash -x, to see where it waits on the pod."""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, "/workspace/refe-plan-p2rep/refe")
os.chdir("/workspace/refe-plan-p2rep/refe")
import test_p2rep_chain as T  # noqa: E402

text = T.CHAIN.read_text(encoding="utf-8")
tmp = Path(tempfile.mkdtemp(prefix="wcprobe_"))
bindir = tmp / "bin"
bindir.mkdir()
T.write_sh(bindir / "flock", T.FLOCK_SHIM)
T.write_sh(bindir / "nv.sh", T.NV_SHIM)
data = tmp / "data"
p2, rl, s1 = data / "refe_p2", data / "rls1", data / "s1aux"
for d in (p2, rl, s1):
    d.mkdir(parents=True)
logf = tmp / "wc.log"
harness = tmp / "wc.sh"
T.write_sh(harness, f'P2="{T.posix(p2)}"; RLS1="{T.posix(rl)}"; S1="{T.posix(s1)}"; L="{T.posix(logf)}"\n' + T.one_line_fn(text, "log")
           + "\nSTAGE=wait\nrefuse() { log \"ZZP2REP_REFUSED stage=$STAGE: $*\"; exit 1; }\n"
           + T.fn_body(text, "gpu_busy") + T.fn_body(text, "upstream") + T.fn_body(text, "wait_clear") + "wait_clear\necho CLEAR\n")
(s1 / "chain.log").write_text("2026-10-14T00:00:00 ZZS1AUX_DONE arms: N-s0\n")
logf.write_text("")
(tmp / "nvc").write_text("0")
e = dict(os.environ)
e.update({"PATH": T.posix(bindir) + os.pathsep + os.environ["PATH"], "P2REP_NVSMI": T.posix(bindir / "nv.sh"), "NVCOUNT": T.posix(tmp / "nvc"),
          "P2REP_POLL_S": "0", "P2REP_IDLE_POLLS": "3"})
print("harness:", harness)
print(harness.read_text()[:3000])
t0 = time.time()
try:
    r = subprocess.run(["bash", "-x", str(harness)], capture_output=True, text=True, env=e, timeout=20)
    print("rc", r.returncode, "in %.1fs" % (time.time() - t0), "stdout", r.stdout[-300:])
    print("TRACE head:\n", r.stderr[:2500])
except subprocess.TimeoutExpired as x:
    err = x.stderr.decode() if isinstance(x.stderr, bytes) else (x.stderr or "")
    print("TIMEOUT after 20 s; trace head:\n", err[:2500], "\n...trace tail:\n", err[-1500:])
print("nvc", (tmp / "nvc").read_text(), "log", logf.read_text()[:500])
