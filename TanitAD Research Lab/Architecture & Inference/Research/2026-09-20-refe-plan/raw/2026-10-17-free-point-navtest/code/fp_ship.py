#!/usr/bin/env python3
"""STAGE B of the free-point navtest read (PLAN.md): ship the navtest frame zips to the pod in log-chunks, keep the pod footprint small, pull the outputs.

The uplink dev box -> pod is the bottleneck (MEASURED 2026-10-17: 1.25-1.29 MB/s, a cap on the link not on a flow), 9.7 GB of frames = 2.1 h.  The pod's volume is 97 % full
and a bank build is writing to it, so at most TWO chunks are ever on the pod (one being processed, one arriving) and nothing is uploaded when free space would drop below 6 GB.

The pod address is NOT in this file (the package lands in a public repo).  It is read from  <work>/../pod_ssh.txt  : one line, the ssh command prefix, e.g.
    ssh -o BatchMode=yes -o ConnectTimeout=25 -i <key> -p <port> root@<host>

    python fp_ship.py plan     --work W                 -> W/ship_plan.json (chunks of logs, <= 1.3 GB of zips each)
    python fp_ship.py prep-pod --work W                 -> ships code, inputs_full.json, calib.npz, the two scorer states; checks the u400 md5 and the model code hashes on the pod
    python fp_ship.py run      --work W                 -> uploads chunks (gated by pod space and backlog) and pulls finished outputs into W/out; ends when every chunk is pulled
    python fp_ship.py status   --work W
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time

FRAMES = "E:/Projects/TanitAD/data/refe_navtest/frames"
REMOTE = "/workspace/ops/freept"
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
STATES = {"c0_u1000.pt": "C:/Users/Admin/rls1_adam_dl/rls1-c0/scorer_states/scorer_states_u1000.pt",
          "r0_u1000.pt": "C:/Users/Admin/rls1_adam_dl/rls1-r0/scorer_states/scorer_states_u1000.pt"}
MODEL_HASHES = {"model.py": "e6011cd83c846dfb", "ckpt_io.py": "240de3486a1ebe30", "load_dinov3.py": "92b9c308694366dc", "calib_table.py": "da4e834f5cf93f8d"}
REMOTE_CODE = "/workspace/refe-plan-rls1/refe"
REMOTE_CKPT = "/workspace/data/rls1/base/model_sft4_B_u400.pt"
CHUNK_BYTES = int(1.3e9)
MIN_FREE_GB = 6.0
MAX_BACKLOG = 1                      # chunks that are ready on the pod but not done, before another upload starts


def ssh_prefix(work):
    p = os.path.join(os.path.dirname(os.path.abspath(work)), "pod_ssh.txt")
    return shlex.split(open(p).read().strip())


def ssh(work, cmd, stdin=None, check=True, timeout=600):
    r = subprocess.run(ssh_prefix(work) + [cmd], stdin=stdin, capture_output=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"ssh failed rc {r.returncode}: {r.stderr.decode(errors='replace')[:400]}")
    return r.stdout.decode(errors="replace")


def put_file(work, local, remote):
    with open(local, "rb") as f:
        ssh(work, f"cat > {shlex.quote(remote)}.tmp && mv {shlex.quote(remote)}.tmp {shlex.quote(remote)}", stdin=f, timeout=3600)


def md5_local(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def cmd_plan(a):
    plan_in = json.load(open(f"{a.work}/plan.json"))
    logs = sorted(plan_in["tokens_per_log"])
    size = {l: sum(os.path.getsize(f"{FRAMES}/{l}_{c}.zip") for c in ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")) for l in logs}
    chunks, cur, cb = [], [], 0
    for l in logs:
        if cur and cb + size[l] > CHUNK_BYTES:
            chunks.append(cur)
            cur, cb = [], 0
        cur.append(l)
        cb += size[l]
    if cur:
        chunks.append(cur)
    out = {"chunk_bytes": CHUNK_BYTES, "total_bytes": sum(size.values()),
           "chunks": [{"name": f"chunk_{i:03d}", "logs": c, "bytes": sum(size[l] for l in c), "tokens": sum(plan_in["tokens_per_log"][l] for l in c)} for i, c in enumerate(chunks)]}
    json.dump(out, open(f"{a.work}/ship_plan.json", "w"), indent=0)
    print(f"{len(chunks)} chunks, {out['total_bytes'] / 1e9:.2f} GB, max chunk {max(c['bytes'] for c in out['chunks']) / 1e9:.2f} GB, {out['total_bytes'] / 1.29e6 / 3600:.1f} h at 1.29 MB/s")
    return 0


def cmd_prep(a):
    ssh(a.work, f"mkdir -p {REMOTE}/in {REMOTE}/out {REMOTE}/states")
    for f in ("fp_forward.py",):
        put_file(a.work, os.path.join(CODE_DIR, f), f"{REMOTE}/{f}")
    for f in ("inputs_full.json", "calib.npz"):
        put_file(a.work, f"{a.work}/{f}", f"{REMOTE}/{f}")
    for nm, p in STATES.items():
        put_file(a.work, p, f"{REMOTE}/states/{nm}")
    # verify by CONTENT on the pod
    out = ssh(a.work, f"cd {REMOTE}; md5sum inputs_full.json calib.npz states/*.pt fp_forward.py; cd {REMOTE_CODE}; sha256sum model.py ckpt_io.py load_dinov3.py calib_table.py | cut -c1-16,65-; "
                      f"md5sum {REMOTE_CKPT} | cut -c1-12; df -BG /workspace | tail -1")
    print(out)
    loc = {f: md5_local(f"{a.work}/{f}") for f in ("inputs_full.json", "calib.npz")}
    loc.update({f"states/{nm}": md5_local(p) for nm, p in STATES.items()})
    loc["fp_forward.py"] = md5_local(os.path.join(CODE_DIR, "fp_forward.py"))
    bad = [k for k, v in loc.items() if f"{v}  {k}" not in out]
    for f, h in MODEL_HASHES.items():
        if f"{h}" not in out:
            bad.append(f"pod {f} hash")
    if "8e382eebb35b" not in out:
        bad.append("pod u400 md5")
    if bad:
        print("PREP_FAILED", bad)
        return 2
    print("ZZSHIP_PREP_OK")
    return 0


def pod_state(work):
    out = ssh(work, f"cd {REMOTE}; ls in out 2>/dev/null; df -BG /workspace | tail -1 | awk '{{print $4}}'")
    free = float(out.strip().splitlines()[-1].rstrip("G"))
    names = out.split()
    ready = {n[:-6] for n in names if n.startswith("chunk_") and n.endswith(".ready")}
    done = {n[:-5] for n in names if n.startswith("chunk_") and n.endswith(".done")}
    return free, ready, done


def upload_chunk(work, ch):
    name = ch["name"]
    ssh(work, f"rm -rf {REMOTE}/in/{name}.part {REMOTE}/in/{name}; mkdir -p {REMOTE}/in/{name}.part")
    files = [f"{l}_{c}.zip" for l in ch["logs"] for c in ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")]
    tar = subprocess.Popen(["tar", "cf", "-", "-C", FRAMES] + files, stdout=subprocess.PIPE)
    r = subprocess.run(ssh_prefix(work) + [f"tar xf - -C {REMOTE}/in/{name}.part"], stdin=tar.stdout, capture_output=True)
    tar.stdout.close()
    tar.wait()
    if r.returncode != 0 or tar.returncode != 0:
        raise RuntimeError(f"upload of {name} failed: tar {tar.returncode} ssh {r.returncode} {r.stderr.decode(errors='replace')[:300]}")
    n = ssh(work, f"ls {REMOTE}/in/{name}.part | wc -l").strip()
    if int(n) != len(files):
        raise RuntimeError(f"{name}: {n} files on the pod, expected {len(files)}")
    sz = int(ssh(work, f"du -sb {REMOTE}/in/{name}.part | cut -f1").strip())
    if sz < ch["bytes"] * 0.999:
        raise RuntimeError(f"{name}: {sz} bytes on the pod, expected {ch['bytes']}")
    ssh(work, f"mv {REMOTE}/in/{name}.part {REMOTE}/in/{name}; printf '%s\\n' {' '.join(shlex.quote(l) for l in ch['logs'])} > {REMOTE}/in/{name}.logs; touch {REMOTE}/in/{name}.ready")


def pull_chunk(work, name):
    os.makedirs(f"{work}/out", exist_ok=True)
    h = ssh(work, f"cat {REMOTE}/out/{name}.sha256").strip()
    with open(f"{work}/out/{name}.npz.part", "wb") as f:
        r = subprocess.run(ssh_prefix(work) + [f"cat {REMOTE}/out/{name}.npz"], stdout=f)
    if r.returncode != 0:
        raise RuntimeError(f"pull {name} failed")
    got = hashlib.sha256(open(f"{work}/out/{name}.npz.part", "rb").read()).hexdigest()
    if got != h:
        raise RuntimeError(f"pull {name}: sha256 {got[:12]} != pod {h[:12]}")
    os.replace(f"{work}/out/{name}.npz.part", f"{work}/out/{name}.npz")


def cmd_run(a):
    plan = json.load(open(f"{a.work}/ship_plan.json"))
    chunks = plan["chunks"]
    pulled = {c["name"] for c in chunks if os.path.exists(f"{a.work}/out/{c['name']}.npz")}
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)  # noqa: E731
    while len(pulled) < len(chunks):
        free, ready, done = pod_state(a.work)
        for n in sorted(done - pulled):
            pull_chunk(a.work, n)
            pulled.add(n)
            log(f"pulled {n} ({len(pulled)}/{len(chunks)})")
        backlog = len(ready - done)
        nxt = next((c for c in chunks if c["name"] not in ready and c["name"] not in done and c["name"] not in pulled), None)
        if nxt is None:
            time.sleep(30)
            continue
        if backlog > MAX_BACKLOG or free - nxt["bytes"] / 1e9 < MIN_FREE_GB:
            log(f"holding {nxt['name']}: backlog {backlog}, pod free {free:.0f} GB")
            time.sleep(60)
            continue
        t0 = time.time()
        try:
            upload_chunk(a.work, nxt)
        except Exception as e:                                    # a dropped ssh costs one chunk's progress, never the run
            log(f"upload of {nxt['name']} FAILED ({str(e)[:200]}); retrying in 60 s")
            time.sleep(60)
            continue
        log(f"uploaded {nxt['name']} {nxt['bytes'] / 1e9:.2f} GB in {time.time() - t0:.0f} s ({nxt['bytes'] / 1e6 / (time.time() - t0):.2f} MB/s)")
    ssh(a.work, f"touch {REMOTE}/in/ALL_DONE")
    log("ZZSHIP_ALL_PULLED")
    return 0


def cmd_status(a):
    plan = json.load(open(f"{a.work}/ship_plan.json"))
    free, ready, done = pod_state(a.work)
    pulled = [c["name"] for c in plan["chunks"] if os.path.exists(f"{a.work}/out/{c['name']}.npz")]
    print(json.dumps({"chunks": len(plan["chunks"]), "ready_on_pod": sorted(ready - done), "done_on_pod": len(done), "pulled": len(pulled), "pod_free_gb": free}))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("plan", "prep-pod", "run", "status"))
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    return {"plan": cmd_plan, "prep-pod": cmd_prep, "run": cmd_run, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
