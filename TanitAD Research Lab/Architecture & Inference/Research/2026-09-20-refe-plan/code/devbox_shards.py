"""Run N shards of a REFe data-prep builder on the WINDOWS dev box, with NO console anywhere.

⛔ WHY (2026-09-23 20:40 local): all four detached dev-box scorer shards died in the same second with
`forrtl: error (200): program aborting due to window-CLOSE event` -- the Intel Fortran runtime inside
numpy/MKL installs a console-control handler and aborts the process when its console window gets a
close event. They had been launched through WMI (which survives the Claude session) but via bash.exe,
which owns a console. Here the supervisor runs under pythonw.exe (no console) and every child is
created with CREATE_NO_WINDOW: there is no console to close. Launch this file via WMI
Win32_Process.Create so it also sits outside the app's process tree.

  pythonw devbox_shards.py --name aug_dev --shards 10 --cwd <refe dir> --log-dir <dir> \
      --env-file env.json -- <python.exe> augment_search.py --bank B --out D/aug_s@I@ --resume ...

Each shard runs the command with `@I@` replaced by its index and `--log-shard i/N` appended;
a failed shard is restarted (bounded); `<log-dir>/<name>.status` is rewritten every minute and
`<name>.done` appears only when EVERY shard exited 0.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

CREATE_NO_WINDOW = 0x08000000


def main() -> int:
    if "--" not in sys.argv:
        print("usage: devbox_shards.py [opts] -- <command ...>")
        return 2
    cut = sys.argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--cwd", required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--env-file", default=None, help="JSON {KEY: VALUE} added to the environment")
    ap.add_argument("--max-restarts", type=int, default=20)
    a = ap.parse_args(sys.argv[1:cut])
    cmd = sys.argv[cut + 1:]
    os.makedirs(a.log_dir, exist_ok=True)
    env = dict(os.environ)
    if a.env_file:
        env.update(json.load(open(a.env_file, encoding="utf-8")))
    env.update({"OMP_NUM_THREADS": str(a.threads), "MKL_NUM_THREADS": str(a.threads),
                "OPENBLAS_NUM_THREADS": str(a.threads), "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1"})
    status_p = os.path.join(a.log_dir, f"{a.name}.status")

    def start(i):
        args = [x.replace("@I@", str(i)) for x in cmd] + ["--log-shard", f"{i}/{a.shards}"]
        log = open(os.path.join(a.log_dir, f"{a.name}_s{i}.log"), "a", encoding="utf-8")
        log.write(f"\n=== start {time.strftime('%Y-%m-%dT%H:%M:%S')} {' '.join(args)}\n")
        log.flush()
        return subprocess.Popen(args, cwd=a.cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW), log

    procs = {i: start(i) for i in range(a.shards)}
    restarts = {i: 0 for i in range(a.shards)}
    finished = {}
    while len(finished) < a.shards:
        time.sleep(60)
        for i, (p, log) in list(procs.items()):
            rc = p.poll()
            if rc is None:
                continue
            log.close()
            del procs[i]
            if rc == 0:
                finished[i] = 0
            elif restarts[i] < a.max_restarts:
                restarts[i] += 1
                time.sleep(15)
                procs[i] = start(i)
            else:
                finished[i] = rc               # gave up: recorded, and no .done is written
        with open(status_p, "w", encoding="utf-8") as f:
            json.dump({"name": a.name, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "running": sorted(procs), "finished": finished, "restarts": restarts}, f)
    ok = all(rc == 0 for rc in finished.values())
    if ok:
        open(os.path.join(a.log_dir, f"{a.name}.done"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%S"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
