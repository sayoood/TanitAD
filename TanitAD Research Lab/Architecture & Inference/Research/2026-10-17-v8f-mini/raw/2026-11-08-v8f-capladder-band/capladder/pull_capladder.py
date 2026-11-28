"""CAPLADDER frame pull (IMPL_CHOICES section 4): the 141 eval-141 source episodes from box 2 with `tar -cf -` over ssh, batches of 8, each file checked against
box 2's own _manifest_sha256.jsonl (sha256 + bytes). Box 2 gets a read-only tar and a `type` of the small manifest; nothing else. Destination F:/TanitAD-scratch/capladder_frames/.
Resume-safe: a verified file is kept; pull_done.json is written when all 141 verify."""
import hashlib, json, os, shutil, subprocess, sys, time
sys.dont_write_bytecode = True
DST = "F:/TanitAD-scratch/capladder_frames"
SRC = r"D:\tanitad\data\eval141-256rb"
SSH = r"C:\Windows\System32\OpenSSH\ssh.exe" if os.path.exists(r"C:\Windows\System32\OpenSSH\ssh.exe") else "ssh"
TAR = r"C:\Windows\System32\tar.exe" if os.path.exists(r"C:\Windows\System32\tar.exe") else "tar"
BATCH = 8


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def ssh(cmd, out=None, timeout=1800):
    return subprocess.run([SSH, "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15", "-o", "BatchMode=yes", "tanitad-box2", cmd], stdout=out or subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)


def main():
    os.makedirs(DST, exist_ok=True)
    if os.path.exists(f"{DST}/pull_done.json"):
        print("[pull] already done", open(f"{DST}/pull_done.json").read(), flush=True)
        return 0
    t0 = time.time()
    man = f"{DST}/_manifest_sha256.jsonl"
    for _ in range(3):
        r = ssh(rf"type {SRC}\_manifest_sha256.jsonl")
        if r.returncode == 0 and len(r.stdout) > 1000:
            open(man, "wb").write(r.stdout)
            break
        time.sleep(5)
    M = [json.loads(l) for l in open(man, encoding="utf-8") if l.strip()]
    assert len(M) == 141, len(M)
    total = sum(m["bytes"] for m in M)
    print(f"[pull] manifest {len(M)} episodes, {total:,} bytes ({total / 2**30:.2f} GiB); ssh={SSH}", flush=True)

    def ok(m):
        p = f"{DST}/{m['clip']}.v2ep.pt"
        return os.path.exists(p) and os.path.getsize(p) == m["bytes"] and sha(p) == m["sha256"]
    todo = [m for m in M if not ok(m)]
    print(f"[pull] {len(M) - len(todo)} verified already, {len(todo)} to pull", flush=True)
    for b0 in range(0, len(todo), BATCH):
        batch = todo[b0:b0 + BATCH]
        names = [f"{m['clip']}.v2ep.pt" for m in batch]
        for attempt in range(4):
            tmp = f"{DST}/_batch.tar"
            try:
                with open(tmp, "wb") as fo:
                    r = ssh(f"tar -cf - -C {SRC} " + " ".join(names), out=fo)
                if r.returncode != 0:
                    raise RuntimeError(f"ssh rc {r.returncode}: {r.stderr[-200:]!r}")
                rr = subprocess.run([TAR, "-xf", tmp, "-C", DST], capture_output=True)
                if rr.returncode != 0:
                    raise RuntimeError(f"tar x rc {rr.returncode}: {rr.stderr[-200:]!r}")
                bad = [m["clip"] for m in batch if not ok(m)]
                if bad:
                    raise RuntimeError(f"verify failed: {bad}")
                break
            except Exception as e:
                print(f"[pull] batch {b0 // BATCH} attempt {attempt + 1} failed: {e}", flush=True)
                time.sleep(10 * (attempt + 1))
        else:
            print("[pull] GIVING UP on a batch", flush=True)
            return 1
        os.path.exists(tmp) and os.remove(tmp)
        print(f"[pull] {min(b0 + BATCH, len(todo))}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    n = sum(1 for m in M if ok(m))
    assert n == 141, n
    json.dump({"episodes": n, "bytes": total, "seconds": round(time.time() - t0, 1)}, open(f"{DST}/pull_done.json", "w"))
    print(f"[pull] done: {n} episodes verified by sha256, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
