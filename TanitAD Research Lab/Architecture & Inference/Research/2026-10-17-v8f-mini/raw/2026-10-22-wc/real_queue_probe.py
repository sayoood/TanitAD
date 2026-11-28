"""One-off proof (not in the suite): the REAL gatewc / trainwc lines, run against a fake python that records its argv."""
import os, subprocess, sys
F, Q = sys.argv[1], sys.argv[2]
open(os.path.join(F, "fakepy.cmd"), "w").write('@echo off\r\necho ARGS %*>> "' + os.path.join(F, "args.txt") + '"\r\nexit /b 0\r\n')
env = dict(os.environ, REPO=os.path.join(F, "repo"), PY=os.path.join(F, "fakepy.cmd"), RUNSM=os.path.join(F, "runsm"),
           CACHEFULL=os.path.join(F, "cf"), V9=os.path.join(F, "v9"))
env.pop("V8F_DRYRUN", None)
for args in ("trainwc 0", "gatewc"):
    r = subprocess.run(f'cmd /c ""{Q}" {args}"', env=env, capture_output=True, text=True, timeout=60)
    print(args, "rc", r.returncode, r.stdout.strip()[-300:], r.stderr.strip()[-300:])
print(open(os.path.join(F, "args.txt")).read())
