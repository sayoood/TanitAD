"""Push the dev box's bank shards to the pod, where code/grow_assemble.py folds them into the --grow bank.

Each local shard file `<src>/<dir>/<file>` goes to `<dst>/<dir>_<tag>/<file>` on the pod -- a SUFFIX,
so `r0_s0_DEV10` still matches the pod's `r0_s*` globs: the assembler folds it in AND the pod's own
rank-0 shards count it in their shared resume set (a log the dev box finished is never redone). As a
temp name then an atomic `mv` -- the pod's assembler may read the destination at any moment and
must never see half a file (a torn tail is harmless to it, a torn MIDDLE would not be). Only files
whose size changed since the last push are sent (the uplink is 1.2 MB/s, MEASURED 2026-09-19).
Only COMPLETE lines are sent: the local writer may be mid-row, so the file is cut at its last
newline before upload.

  python devbox_upload.py --src D:/Projects/TanitAD/data/refe_navtrain10 --glob "r0_s*/targets_rank0.jsonl" \
      --dst /workspace/data/refe_navtrain --tag DEV10 --every-min 30
Prints ZZUPLOAD_OK <files sent> <bytes> per pass; stops when <src>/UPLOAD_STOP exists.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time

SSH = ["-i", os.path.expanduser("~/.ssh/tanitad_pod"), "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
# ⛔ NATIVE Windows OpenSSH, never git-bash's MSYS ssh: MSYS ssh.exe DEADLOCKS under Python subprocess
# pipes against busy pods (memory: python-ssh-to-pods-on-devbox, MEASURED 2026-07-21).
_WIN = r"C:\Windows\System32\OpenSSH"
SSH_EXE = os.path.join(_WIN, "ssh.exe") if os.path.exists(os.path.join(_WIN, "ssh.exe")) else "ssh"
SCP_EXE = os.path.join(_WIN, "scp.exe") if os.path.exists(os.path.join(_WIN, "scp.exe")) else "scp"


def run(cmd, timeout=900):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    return r.returncode, (r.stdout + r.stderr)[-400:]


def one_pass(a, state) -> tuple[int, int]:
    sent = nbytes = 0
    for f in sorted(glob.glob(os.path.join(a.src, a.glob))):
        rel_dir = os.path.basename(os.path.dirname(f))
        with open(f, "rb") as fh:
            buf = fh.read()
        cut = buf.rfind(b"\n")
        if cut < 0:
            continue
        body = buf[:cut + 1]
        key = f"{rel_dir}/{os.path.basename(f)}"
        if state.get(key) == len(body):
            continue
        dst_dir = f"{a.dst}/{rel_dir}_{a.tag}"
        dst = f"{dst_dir}/{os.path.basename(f)}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
            tmp.write(body)
            tpath = tmp.name
        try:
            rc, out = run([SSH_EXE] + SSH + ["-p", a.port, a.host, f"mkdir -p '{dst_dir}'"], 120)
            if rc != 0:
                print(f"  mkdir failed for {dst_dir}: {out}", flush=True)
                continue
            rc, out = run([SCP_EXE, "-q"] + SSH + ["-P", a.port, tpath, f"{a.host}:{dst}.part"])
            if rc != 0:
                print(f"  scp failed for {key}: {out}", flush=True)
                continue
            # the size check happens ON THE POD before the rename: a short transfer never goes live
            rc, out = run([SSH_EXE] + SSH + ["-p", a.port, a.host,
                           f"s=$(stat -c %s '{dst}.part'); [ \"$s\" = '{len(body)}' ] && mv -f '{dst}.part' '{dst}' "
                           f"&& echo MOVED || echo SIZE_MISMATCH $s"], 120)
            if "MOVED" not in out:
                print(f"  NOT moved {key}: {out.strip()}", flush=True)
                continue
            state[key] = len(body)
            sent += 1
            nbytes += len(body)
        finally:
            os.unlink(tpath)
    return sent, nbytes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--glob", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--tag", default="DEV")
    ap.add_argument("--host", default="root@69.30.85.13")
    ap.add_argument("--port", default="22175")
    ap.add_argument("--every-min", type=float, default=0.0)
    a = ap.parse_args()
    # ⛔ MEASURED 2026-09-24: launched from git-bash, `--dst /workspace/...` arrived as
    # `C:/Program Files/Git/workspace/...` (MSYS path conversion) and 11 files landed in
    # /root/C:/Program Files/... on the pod while every transfer reported MOVED. Refuse such a dst.
    if not a.dst.startswith("/") or ":" in a.dst:
        print(f"  REFUSING --dst {a.dst!r}: not a POSIX path (MSYS path conversion? launch natively "
              f"or with MSYS_NO_PATHCONV=1)")
        return 2
    # one state file per (tag, glob): several uploader loops run side by side (rank 0, aug, scorer)
    # and must never write the same JSON concurrently. The rank-0 loop keeps its original name.
    import hashlib
    g = a.glob.replace("\\", "/")
    suffix = "" if g == "r0_s*/targets_rank0.jsonl" else "_" + hashlib.sha1(g.encode()).hexdigest()[:8]
    state_p = os.path.join(a.src, f".upload_state_{a.tag}{suffix}.json")
    state = json.load(open(state_p)) if os.path.exists(state_p) else {}
    while True:
        t = time.time()
        sent, nbytes = one_pass(a, state)
        json.dump(state, open(state_p, "w"))
        print(f"ZZUPLOAD_OK {time.strftime('%Y-%m-%dT%H:%M:%S')} files {sent} bytes {nbytes:,} "
              f"{time.time() - t:.0f}s", flush=True)
        if a.every_min <= 0 or os.path.exists(os.path.join(a.src, "UPLOAD_STOP")):
            return 0
        time.sleep(a.every_min * 60.0)


if __name__ == "__main__":
    sys.exit(main())
