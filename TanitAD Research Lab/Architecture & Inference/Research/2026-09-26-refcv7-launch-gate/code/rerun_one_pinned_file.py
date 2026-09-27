"""Re-run ONE pinned test file in EXACTLY G-SUITE-PINNED's environment (`launch_gate.pinned_env`), on
the full-commit archive the job already built (its throwaway repo asserted at the commit), verbose,
with its own timeout -- the diagnosis of a file that did not complete, without re-running the 32 that
did.

    python rerun_one_pinned_file.py <gate tree> <archive dir> <commit> <sub> <file> <out dir> [timeout_s]
"""
import json
import subprocess
import sys
import time
from pathlib import Path

tree, dest, commit, sub, f, out = (Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4],
                                   sys.argv[5], Path(sys.argv[6]))
timeout = int(sys.argv[7]) if len(sys.argv) > 7 else 900
sys.path.insert(0, str(tree / "stack" / "scripts"))
import launch_gate as LG  # noqa: E402

out.mkdir(parents=True, exist_ok=True)
head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True,
                      text=True).stdout.strip()
assert head == commit, f"the throwaway repo's HEAD is {head!r}, not {commit}"
env = LG.pinned_env(dest, {"omp": 4, "cpu_only": True}, dict(__import__("os").environ))
log = out / (f.replace("/", "__") + ".log")
t0 = time.time()
try:
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        p = subprocess.run([sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider", "-rfEs",
                            "--durations=15", f], cwd=str(dest / sub), env=env, stdout=fh,
                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=timeout)
    rc = p.returncode
except subprocess.TimeoutExpired:
    rc = "timeout"
res = {"file": f"{sub}/{f}", "rc": rc, "elapsed_s": round(time.time() - t0, 1), "head": head,
       "env_git": sorted(k for k in env if k.startswith("GIT_")),
       "secret_scan_full": env.get("SECRET_SCAN_FULL"), "log": str(log)}
(out / "rerun.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
print(json.dumps(res))
