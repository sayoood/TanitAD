"""Does the test's 0.25 s observer probe make a fake chain's `flock -n 8 || exit 9` fail?  Same shim, same probe line as Queue.observe()."""
import os, subprocess, sys, tempfile, time
from pathlib import Path
HERE = Path(sys.argv[1]); N = int(sys.argv[2])
sys.path.insert(0, str(HERE)); os.chdir(HERE)
import test_rls1_queue as T
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp); b = T.shims(tmp); lk = tmp / "s1aux" / "chain.lock"; lk.parent.mkdir(); lk.touch()
    env = {**os.environ, "PATH": T.posix(b) + os.pathsep + os.environ["PATH"]}
    obs = tmp / "observe.sh"
    T.write_sh(obs, f'end=$((SECONDS + 600))\nwhile [ $SECONDS -lt $end ]; do\n  a=0; ( flock -n 9 ) 9<> {T.posix(lk)} 2> /dev/null || a=1\n  sleep 0.25\ndone\n')
    con = tmp / "contend.sh"     # the fake chain's own acquire, verbatim form: exec 8>"$D/chain.lock"; flock -n 8 || exit 9
    T.write_sh(con, f'ok=0; bad=0\nfor i in $(seq 1 {N}); do\n  ( exec 8>"{T.posix(lk)}"; flock -n 8 || exit 9 ) && ok=$((ok+1)) || bad=$((bad+1))\n  sleep 0.$((RANDOM % 9 + 1))\ndone\necho "ZZCOLLIDE n={N} acquired=$ok refused=$bad"\n')
    base = subprocess.run([T.BASH, T.posix(con)], env=env, capture_output=True, text=True, timeout=250).stdout.strip()
    po = subprocess.Popen([T.BASH, T.posix(obs)], env=env)
    try:
        time.sleep(1.0)
        withobs = subprocess.run([T.BASH, T.posix(con)], env=env, capture_output=True, text=True, timeout=250).stdout.strip()
    finally:
        po.kill(); po.wait(10)
    print("control (no observer):", base); print("with the observer probing:", withobs)
