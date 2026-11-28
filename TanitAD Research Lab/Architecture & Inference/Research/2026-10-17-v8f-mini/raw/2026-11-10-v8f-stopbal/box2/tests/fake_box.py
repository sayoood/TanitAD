"""Dry-run stand-in for box 2 (dev box only). Called by ssh_stub.sh / sftp_stub.sh; it never opens a network connection and never imports a client library.
  ssh  <ssh options> tanitad-box2 <command string>   -> emulates exactly the command forms stopbal_common.sh sends (free GB, nvidia-smi, tasklist, Test-Path, type, certutil, New-Item, launch.ps1)
  sftp -b - tanitad-box2 (batch on stdin)            -> put <local> /<remote>   |   get -r /<remote> <local>
The fake box is the folder FAKE_BOX (D:/... maps to <FAKE_BOX>/D/...). Control files in it: free_gb.txt, py_count.txt, gpu_mib.txt. Every call is appended to <FAKE_BOX>/transport.log.
launch.ps1 runs the REAL job .cmd on Windows cmd.exe, synchronously, so the markers exist when the dev-box poll looks."""
import hashlib
import os
import re
import shutil
import subprocess
import sys

BOX = os.environ["FAKE_BOX"]


def fp(p):
    p = p.strip().strip('"').replace("\\", "/")
    m = re.match(r"^/?([A-Za-z]):/?(.*)$", p)
    return os.path.join(BOX, m.group(1), *[x for x in m.group(2).split("/") if x]) if m else p


def log(s):
    with open(os.path.join(BOX, "transport.log"), "a", encoding="utf-8") as f:
        f.write(s + "\n")


def ssh(cmd):
    log("ssh " + cmd[:200])
    m = re.search(r"Get-PSDrive D", cmd)
    if m:
        f = os.path.join(BOX, "free_gb.txt")
        print(open(f).read().strip() if os.path.exists(f) else "100")
        return 0
    if cmd.startswith("nvidia-smi"):
        f = os.path.join(BOX, "gpu_mib.txt")
        print(open(f).read().strip() if os.path.exists(f) else "0")
        return 0
    if cmd.startswith("tasklist"):
        f = os.path.join(BOX, "py_count.txt")
        n = int(open(f).read().strip()) if os.path.exists(f) else 0
        print("\n".join(["python.exe  1234 Console 1 100 K"] * n) if n else "INFO: No tasks are running which match the specified criteria.")
        return 0
    m = re.search(r"Test-Path (\S+?)\"?$", cmd)
    if m:
        print("True" if os.path.exists(fp(m.group(1))) else "False")
        return 0
    m = re.search(r"New-Item -ItemType Directory -Force (\S+)", cmd)
    if m:
        os.makedirs(fp(m.group(1)), exist_ok=True)
        return 0
    m = re.match(r"type (.+)$", cmd)
    if m:
        p = fp(m.group(1))
        if not os.path.isfile(p):
            return 1
        sys.stdout.write(open(p, encoding="utf-8", errors="replace").read())
        return 0
    m = re.match(r"certutil -hashfile (.+) SHA256$", cmd)
    if m:
        p = fp(m.group(1))
        if not os.path.isfile(p):
            print("CertUtil: -hashfile command FAILED")
            return 1
        print("SHA256 hash of file:\n" + hashlib.sha256(open(p, "rb").read()).hexdigest().upper() + "\nCertUtil: -hashfile command completed successfully.")
        return 0
    m = re.search(r"launch\.ps1 -Job (\S+) -Log (\S+)", cmd)
    if m:
        job, lg = fp(m.group(1)), fp(m.group(2))
        os.makedirs(os.path.dirname(lg), exist_ok=True)
        env = dict(os.environ, MSYS_NO_PATHCONV="1")
        with open(lg, "w") as out:
            subprocess.run(["cmd.exe", "/c", job.replace("/", "\\")], stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, env=env)
        return 0
    log("UNKNOWN COMMAND " + cmd[:200])
    print("fake box: unknown command", file=sys.stderr)
    return 2


def sftp():
    rc = 0
    for line in sys.stdin.read().splitlines():
        line = line.strip()
        if not line:
            continue
        log("sftp " + line[:200])
        a = line.split(" ")
        if a[0] == "put" and len(a) == 3:
            dst = fp(a[2])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.environ.get("FAKE_CORRUPT_PUT") and os.path.basename(dst) == os.environ["FAKE_CORRUPT_PUT"]:
                open(dst, "wb").write(open(a[1], "rb").read() + b"x")
            else:
                shutil.copyfile(a[1], dst)
        elif a[0] == "get" and a[1] == "-r" and len(a) == 4:
            src = fp(a[2])
            if not os.path.exists(src):
                rc = 1
                continue
            os.makedirs(os.path.dirname(os.path.abspath(a[3])) or ".", exist_ok=True)
            if os.path.isdir(src):
                shutil.copytree(src, a[3], dirs_exist_ok=True)
            else:
                shutil.copyfile(src, a[3])
        else:
            rc = 2
    return rc


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "sftp":
        sys.exit(sftp())
    sys.exit(ssh(sys.argv[-1]))
