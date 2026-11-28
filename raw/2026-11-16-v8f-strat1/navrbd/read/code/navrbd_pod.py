"""navrbd_pod.py -- the pod-side steps of the NAVRBD chain: sha256-verified pulls and the wait for the pod's FINAL line.  Reads and pulls only, under /root/strat1/navrbd/ and /root/strat1/logs/rbdpod.log.
The pod address is read at run time from the box's podenv.sh (sourced inside bash) and is never written anywhere.
Sub-commands:
  pull-ckpts   runs/N1D_s{0,1}/train/ckpt_last.pt -> read/ckpts/N1D_s{0,1}/ckpt_last.pt, sha256 equal to the pod's own sha256sum (a BANK_MANIFEST.json is written beside it: tag, pod sha256, commit)
  wait-end     poll rbdpod.log every --poll seconds (default 300), cap --cap-h hours (default 6) for the FINAL line `ZZ RBDPOD ALL END ZZ main: <main sentinel>; zero dumps DONE|FAIL` (MM ruling 2026-11-18: the
               DONE / FAIL line of the main script is written BEFORE the zero-variant dumps and is NOT the end); the line is printed (the chain log records it).  Exit 1 on the cap.
  pull-dumps   every dump_NAVRBD-HO-* dir under out/t1 and out/t1_zero -> read/dumps/ (flat), each file sha256-verified; a dir already complete locally is skipped
Usage: python navrbd_pod.py {pull-ckpts|wait-end|pull-dumps}"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CK = os.environ.get("NAVRBD_CKPTS", HERE + "/ckpts")
DUMPS = os.environ.get("NAVRBD_D_HO", HERE + "/dumps")
PODENV = os.environ.get("NAVRBD_PODENV", "C:/Users/Admin/lorapod/podenv.sh")
PODROOT = "/root/strat1/navrbd"
POD_LOG = "/root/strat1/logs/rbdpod.log"
COMMIT = "0c3b374f"
ARMS = ("N1D_s0", "N1D_s1")
DUMP_DIRS = ("out/t1", "out/t1_zero")
ALL_END = re.compile(r"^ZZ RBDPOD ALL END ZZ main: (?P<main>.*); zero dumps (?P<zero>DONE|FAIL)\s*$")


def bash_exe():
    for p in ("C:/Program Files/Git/usr/bin/bash.exe", shutil_which("bash")):
        if p and os.path.isfile(p):
            return p
    raise RuntimeError("no Git bash")


def shutil_which(n):
    return shutil.which(n)


def pod(cmd, binary=False, stdout=None, timeout=3600):
    """run `cmd` on the pod through podenv.sh's pssh; returns CompletedProcess.  The address stays inside the shell."""
    script = f'source "{PODENV}"; pssh {cmd}'
    kw = dict(stdout=stdout or subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if not binary:
        kw["encoding"] = "utf-8"
    return subprocess.run([bash_exe(), "-c", script], **kw)


def last_line(text):
    lines = [x for x in text.splitlines() if x.strip()]
    return lines[-1] if lines else ""


def parse_all_end(text):
    """-> (main sentinel, 'DONE'|'FAIL') when the LAST non-empty line is the ALL END line, else None.  Pure.  A DONE / FAIL line of the main script is NOT the end."""
    m = ALL_END.match(last_line(text))
    return (m.group("main"), m.group("zero")) if m else None


def parse_sums(text):
    """sha256sum output -> {relative path: sha}.  Pure."""
    out = {}
    for ln in text.splitlines():
        m = re.match(r"^([0-9a-f]{64})\s+\*?(.+?)\s*$", ln)
        if m:
            out[m.group(2).lstrip("./")] = m.group(1)
    return out


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def verify_tree(local_root, sums):
    """-> list of problems: every remote file must exist locally with the same sha256.  Pure given files."""
    bad = []
    for rel, want in sums.items():
        p = os.path.join(local_root, rel)
        if not os.path.isfile(p):
            bad.append(f"missing {rel}")
        elif sha256_file(p) != want:
            bad.append(f"sha256 differs {rel}")
    if not sums:
        bad.append("no files listed")
    return bad


def wait_end(read_log, poll=300, cap_s=6 * 3600, sleep=time.sleep, clock=time.time):
    """polls read_log() -> text until parse_all_end; returns (main, zero, line) or None at the cap"""
    t0 = clock()
    while True:
        txt = read_log()
        r = parse_all_end(txt)
        if r:
            return r[0], r[1], last_line(txt)
        if clock() - t0 >= cap_s:
            return None
        sleep(poll)


def pod_log_text():
    r = pod(f"tail -n 40 {POD_LOG}", timeout=120)
    return r.stdout if r.returncode == 0 else ""


def pull_one_ckpt(arm, ck_root=None):
    ck_root = ck_root or CK
    rp = f"{PODROOT}/runs/{arm}/train/ckpt_last.pt"
    r = pod(f"sha256sum {rp}", timeout=900)
    sums = parse_sums(r.stdout.replace(rp, os.path.basename(rp)))
    want = sums.get("ckpt_last.pt")
    if r.returncode or not want:
        raise RuntimeError(f"{arm}: pod sha256sum failed")
    d = f"{ck_root}/{arm}"
    dst = f"{d}/ckpt_last.pt"
    if os.path.isfile(dst) and sha256_file(dst) == want:
        print("skip (verified)", dst, flush=True)
    else:
        os.makedirs(d, exist_ok=True)
        with open(dst + ".part", "wb") as f:
            r = pod(f"cat {rp}", binary=True, stdout=f, timeout=7200)
        if r.returncode:
            raise RuntimeError(f"{arm}: pull failed rc {r.returncode}")
        got = sha256_file(dst + ".part")
        if got != want:
            raise RuntimeError(f"{arm}: sha256 {got[:12]} != pod {want[:12]}")
        os.replace(dst + ".part", dst)
        print("pulled", dst, want[:12], flush=True)
    json.dump({"tag": arm, "commit": COMMIT, "source": "pod runs/<arm>/train/ckpt_last.pt, sha256 equal to the pod's sha256sum",
               "files": {"ckpt_last.pt": {"sha256": want}}}, open(f"{d}/BANK_MANIFEST.json", "w", encoding="utf-8"), indent=1)
    return want


def list_dumps():
    """-> [(remote dir under PODROOT, dump name)]"""
    out = []
    for d in DUMP_DIRS:
        r = pod(f"ls -1 {PODROOT}/{d}", timeout=300)
        if r.returncode:
            continue
        out += [(d, n.strip()) for n in r.stdout.splitlines() if re.match(r"^dump_NAVRBD-HO-\S+$", n.strip())]
    return out


def pull_dump(d, name, root=None):
    root = root or DUMPS
    cmd = f"'cd {PODROOT}/{d} && find {name} -type f -print0 | sort -z | xargs -0 sha256sum'"
    r = pod(cmd, timeout=900)
    sums = parse_sums(r.stdout)
    if r.returncode or not sums:
        raise RuntimeError(f"{name}: pod sha256 listing failed")
    if os.path.isdir(f"{root}/{name}") and not verify_tree(root, sums):
        print("skip (verified)", name, flush=True)
        return len(sums)
    os.makedirs(root, exist_ok=True)
    tarp = f"{root}/{name}.tar.part"
    with open(tarp, "wb") as f:
        r = pod(f"'cd {PODROOT}/{d} && tar cf - {name}'", binary=True, stdout=f, timeout=3600)
    if r.returncode:
        raise RuntimeError(f"{name}: tar pull rc {r.returncode}")
    subprocess.run(["tar", "xf", tarp, "-C", root], check=True)
    os.remove(tarp)
    bad = verify_tree(root, sums)
    if bad:
        raise RuntimeError(f"{name}: {bad[:3]}")
    print("pulled", name, len(sums), "files", flush=True)
    return len(sums)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "pull-ckpts":
        for a in ARMS:
            pull_one_ckpt(a)
    elif cmd == "wait-end":
        poll = int(os.environ.get("NAVRBD_POLL_S", "300"))
        cap = float(os.environ.get("NAVRBD_CAP_H", "6")) * 3600
        r = wait_end(pod_log_text, poll, cap)
        if r is None:
            print("CAP: no ZZ RBDPOD ALL END ZZ line within the cap", flush=True)
            sys.exit(1)
        print("POD FINAL LINE:", r[2], flush=True)
    elif cmd == "pull-dumps":
        ds = list_dumps()
        print("dump dirs on the pod:", [n for _, n in ds], flush=True)
        for d, n in ds:
            pull_dump(d, n)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
