#!/usr/bin/env python3
"""Measure 5 effectiveness run (eval/PREREG_MEASURE5.md §3a): the pod gate and the frame pull. POD SIDE IS READ-ONLY.

Every pod action is a Python script GENERATED here with its inputs embedded, streamed as ssh STDIN into
`nice -n 19 [ionice -c3] python3 -` -- it reads files and prints to STDOUT; it writes nothing on the pod. ssh runs with
FILE handles, not pipes (the MSYS-ssh pipe deadlock; memory python-ssh-to-pods-on-devbox), via Windows OpenSSH.

  gate    the pre-registered row gate: for the drawn samples (+ a reserve in the same seeded order) the pod bank
          (/workspace/data/refe_navtrain/train_grow/targets_rank0.jsonl) must hold the SAME key and teacher trajectory
          (sha256 of both, as values). Rows that fail are replaced by the next in the permutation order (declared).
          Also returns the pod's calibration for the logs (compared with the local build) and the trainer's step time.
          -> <out>/gate.json, <out>/samples_gated.json
  pull    the 4 camera frames of every gated sample, CHUNKED: the pod script finds each frame the way FrameStore does
          (<root>/<ref>, else <root>/<top-dir>/<ref>) and streams a tar to stdout, ending with a manifest (sha256 per
          frame + the trainer's step time at the chunk's start and end). The driver extracts to LOOSE files
          <out>/pixels/<log>/<CAM>/<id>.jpg, checks sha256 + decode per frame, and PAUSES if the trainer's median step
          time exceeds --max-step-s (the coordinator's bar: 61.5 + 2 = 63.5 s).
  verify  every frame of every gated sample decodes (cv2) and matches its manifest sha256; counts and bytes.

    python eval/m5_transfer.py gate  --out D:/Projects/TanitAD/data/refe_m5
    python eval/m5_transfer.py pull  --out D:/Projects/TanitAD/data/refe_m5 [--chunk 100] [--max-chunks N]
    python eval/m5_transfer.py verify --out D:/Projects/TanitAD/data/refe_m5
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time

HOST, PORT, KEY = "root@<redacted-ip>", "22175", "C:/Users/Admin/.ssh/tanitad_pod"
SSH = "C:/Windows/System32/OpenSSH/ssh.exe"
POD_BANK = "/workspace/data/refe_navtrain/train_grow/targets_rank0.jsonl"
POD_CALIB = "/workspace/data/refe_navtrain/train_grow/calib_table.json"
POD_PIX = "/workspace/data/navtrain_pixels"
POD_LOG = "/workspace/data/refe_runs/vitl16_navtrain10_grow_tau0.3/train.log"
RESERVE = 1.25                     # rows gated per needed row, in the seeded order (replacements for failures)

# ------------------------------------------------------------------------------------------ pod-side script bodies
POD_COMMON = r'''
import json, os, re, sys, time, hashlib
LOG = %(log)r
def step_times(nbytes=300000):
    try:
        with open(LOG, "rb") as f:
            f.seek(0, os.SEEK_END); size = f.tell(); f.seek(max(0, size - nbytes))
            lines = f.read().decode("utf-8", "replace").splitlines()
        pat = re.compile(r"^\s+step\s+(\d+)\s+traj_L1.*?\s([\d.]+)s\s*$")
        pts = [(int(m.group(1)), float(m.group(2))) for m in (pat.match(l) for l in lines) if m][-11:]
        d = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(pts, pts[1:]) if b[0] > a[0]]
        return {"last_step": pts[-1][0] if pts else None, "step_s": [round(x, 2) for x in d],
                "median": round(sorted(d)[len(d) // 2], 2) if d else None}
    except Exception as exc:
        return {"error": repr(exc)[:160]}
'''

POD_GATE = POD_COMMON + r'''
WANT = set(json.loads(%(want)r))
LOGS = set(json.loads(%(logs)r))
t0 = time.time()
out = {"step_start": step_times(), "rows": {}}
tok_re = re.compile(r'"token":\s*"([0-9a-f]+)"')
n = 0
with open(%(bank)r, encoding="utf-8") as f:
    for line in f:
        n += 1
        m = tok_re.search(line)
        if not m or m.group(1) not in WANT:
            continue
        r = json.loads(line)
        key = "%%s|%%s|%%d|%%d" %% (r["log_name"], r["token"], int(r["step"]), int(r.get("rank", 0)))
        blob = json.dumps([r["log_name"], r["token"], int(r["step"]), int(r.get("rank", 0)), r["traj"]],
                          separators=(",", ":"))
        out["rows"].setdefault(key, []).append({
            "row_sha": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
            "ego_sha": hashlib.sha256(json.dumps(r.get("ego")).encode()).hexdigest(),
            "goal_sha": hashlib.sha256(json.dumps(r.get("goal")).encode()).hexdigest(),
            "image": r.get("image")})
out["bank_lines"] = n
try:
    cal = json.load(open(%(calib)r, encoding="utf-8"))
    out["calib"] = {lg: cal["logs"].get(lg) for lg in LOGS}
    out["calib_fields"] = cal.get("fields")
except Exception as exc:
    out["calib_error"] = repr(exc)[:160]
out["step_end"] = step_times()
out["seconds"] = round(time.time() - t0, 1)
sys.stdout.write(json.dumps(out) + "\n")
'''

POD_PULL = POD_COMMON + r'''
import tarfile, io
PIX = %(pix)r
REFS = json.loads(%(refs)r)          # [[log_name, ref], ...]
t0 = time.time()
man = {"step_start": step_times(), "frames": {}, "missing": []}
tops = [d for d in sorted(os.listdir(PIX)) if os.path.isdir(os.path.join(PIX, d))]
where = {}
tar = tarfile.open(fileobj=sys.stdout.buffer, mode="w|")
for log, ref in REFS:
    p = os.path.join(PIX, *ref.split("/"))
    if not os.path.isfile(p):
        if log not in where:
            where[log] = [d for d in tops if os.path.isdir(os.path.join(PIX, d, log))]
        p = None
        for d in where[log]:
            q = os.path.join(PIX, d, *ref.split("/"))
            if os.path.isfile(q):
                p = q
                break
    if p is None:
        man["missing"].append(ref)
        continue
    with open(p, "rb") as f:
        b = f.read()
    ti = tarfile.TarInfo(ref)
    ti.size = len(b)
    ti.mtime = 0
    tar.addfile(ti, io.BytesIO(b))
    man["frames"][ref] = {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b), "src": p[len(PIX) + 1:]}
man["step_end"] = step_times()
man["seconds"] = round(time.time() - t0, 1)
mb = json.dumps(man).encode("utf-8")
ti = tarfile.TarInfo("_manifest.json")
ti.size = len(mb)
tar.addfile(ti, io.BytesIO(mb))
tar.close()
sys.stdout.buffer.flush()
'''


def run_pod(script_text: str, remote_cmd: str, out_path: str, log_path: str, timeout: int = 1800) -> int:
    """Run a generated script on the pod: script = stdin FILE, stdout -> FILE, stderr -> FILE."""
    sp = out_path + ".script.py"
    with open(sp, "w", encoding="utf-8", newline="\n") as f:
        f.write(script_text)
    with open(sp, "rb") as fi, open(out_path, "wb") as fo, open(log_path, "wb") as fe:
        p = subprocess.run([SSH, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30",
                            "-i", KEY, "-p", PORT, HOST, remote_cmd], stdin=fi, stdout=fo, stderr=fe,
                           timeout=timeout)
    return p.returncode


# ------------------------------------------------------------------------------------------ stages
def gate(a) -> int:
    S = json.load(open(os.path.join(a.out, "samples.json"), encoding="utf-8"))
    cand = {}
    for name, n in (("train", len(S["train"])), ("val", len(S["val"]))):
        cand[name] = S[f"order_{name}"][:int(n * RESERVE) + 20]
    want = sorted({x["token"] for v in cand.values() for x in v})
    logs = sorted({x["log_name"] for v in cand.values() for x in v})
    text = POD_GATE % {"log": POD_LOG, "want": json.dumps(want), "logs": json.dumps(logs), "bank": POD_BANK,
                       "calib": POD_CALIB}
    outp = os.path.join(a.out, "gate_pod.json")
    t0 = time.time()
    rc = run_pod(text, "nice -n 19 python3 -", outp, outp + ".stderr", timeout=900)
    raw = open(outp, "rb").read().decode("utf-8", "replace").strip().splitlines()
    if rc != 0 or not raw:
        print(f"ZZM5_GATE_FAIL ssh rc {rc}; stderr: {open(outp + '.stderr', 'rb').read()[-600:]!r}")
        return 1
    pod = json.loads(raw[-1])
    res = {"pod_bank": POD_BANK, "bank_lines_scanned": pod["bank_lines"], "pod_seconds": pod["seconds"],
           "step_start": pod["step_start"], "step_end": pod["step_end"], "ssh_wall_s": round(time.time() - t0, 1)}
    gated, stats = {}, {}
    for name, need in (("train", len(S["train"])), ("val", len(S["val"]))):
        ok, fail = [], []
        for x in cand[name]:
            hits = pod["rows"].get(x["key"], [])
            if any(h["row_sha"] == x["row_sha"] for h in hits):
                h = next(h for h in hits if h["row_sha"] == x["row_sha"])
                ok.append(dict(x, pod_image=h["image"], pod_ego_sha=h["ego_sha"], pod_goal_sha=h["goal_sha"]))
            else:
                fail.append({"key": x["key"], "pod_rows": len(hits)})
            if len(ok) == need:
                break
        drawn = {x["key"] for x in S[name]}
        stats[name] = {"need": need, "gated": len(ok), "failed_before_filled": len(fail),
                       "failures": fail[:20], "replaced_from_reserve": sum(1 for x in ok if x["key"] not in drawn),
                       "drawn_that_passed": sum(1 for x in ok if x["key"] in drawn)}
        gated[name] = ok
    res["split"] = stats
    # the calibration cross-check: the local build must equal the pod's for every gated log
    local = json.load(open(os.path.join(a.out, "calib_table.json"), encoding="utf-8"))
    glogs = sorted({x["log_name"] for v in gated.values() for x in v})
    cams = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")
    same = sum(1 for lg in glogs if pod.get("calib", {}).get(lg) is not None and local["logs"].get(lg) is not None
               and all(pod["calib"][lg].get(c) == local["logs"][lg].get(c) for c in cams))
    res["calib"] = {"logs": len(glogs), "local_equals_pod": same, "pod_fields": pod.get("calib_fields")}
    ok = all(s["gated"] == s["need"] for s in stats.values()) and same == len(glogs)
    json.dump(res, open(os.path.join(a.out, "gate.json"), "w", encoding="utf-8"), indent=1)
    json.dump({"train": gated["train"], "val": gated["val"], "seed": S["seed"], "source": S["source"],
               "source_sha256": S["source_sha256"]},
              open(os.path.join(a.out, "samples_gated.json"), "w", encoding="utf-8"), indent=0)
    print(json.dumps({k: v for k, v in res.items() if k != "split"}))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "failures"} for k, v in stats.items()}))
    print(f"ZZM5_GATE_{'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


def rows_by_key(src: str, keys: set) -> dict:
    out = {}
    with open(src, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            k = f"{r['log_name']}|{r['token']}|{int(r['step'])}|{int(r.get('rank', 0))}"
            if k in keys:
                out[k] = r
    return out


def pull(a) -> int:
    G = json.load(open(os.path.join(a.out, "samples_gated.json"), encoding="utf-8"))
    samples = G["train"] + G["val"]
    rows = rows_by_key(G["source"], {x["key"] for x in samples})
    refs = []
    for x in samples:
        r = rows[x["key"]]
        if r["image"] != x["pod_image"]:
            print(f"ZZM5_PULL_FAIL image paths differ from the pod's for {x['key']}")
            return 1
        refs += [(r["log_name"], ref) for ref in r["image"]]
    pix = os.path.join(a.out, "pixels")
    tdir = os.path.join(a.out, "pull")
    os.makedirs(pix, exist_ok=True)
    os.makedirs(tdir, exist_ok=True)
    manp = os.path.join(a.out, "frames_manifest.json")
    M = json.load(open(manp, encoding="utf-8")) if os.path.exists(manp) else {"frames": {}, "chunks": []}
    todo = [rf for rf in refs if rf[1] not in M["frames"]]
    print(f"  {len(refs):,} frames for {len(samples):,} samples; {len(todo):,} still to pull", flush=True)
    import cv2
    import numpy as np
    n_chunks = 0
    for c0 in range(0, len(todo), a.chunk * 4):
        part = todo[c0:c0 + a.chunk * 4]
        text = POD_PULL % {"log": POD_LOG, "pix": POD_PIX, "refs": json.dumps(part)}
        tp = os.path.join(tdir, f"chunk_{len(M['chunks']):04d}.tar")
        t0 = time.time()
        rc = run_pod(text, "nice -n 19 ionice -c3 python3 -", tp, tp + ".stderr", timeout=1800)
        wall = time.time() - t0
        try:
            tf = tarfile.open(tp, mode="r:")
            members = tf.getmembers()
        except Exception as exc:
            print(f"ZZM5_PULL_FAIL chunk tar unreadable (ssh rc {rc}): {exc}; stderr "
                  f"{open(tp + '.stderr', 'rb').read()[-400:]!r}")
            return 1
        mani = [m for m in members if m.name == "_manifest.json"]
        if rc != 0 or not mani:
            print(f"ZZM5_PULL_FAIL ssh rc {rc}, manifest {'present' if mani else 'ABSENT'}")
            return 1
        man = json.loads(tf.extractfile(mani[0]).read())
        got = bad = 0
        for m in members:
            if m.name == "_manifest.json":
                continue
            b = tf.extractfile(m).read()
            meta = man["frames"].get(m.name)
            okh = meta is not None and hashlib.sha256(b).hexdigest() == meta["sha256"]
            img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            if not okh or img is None:
                bad += 1
                continue
            dst = os.path.join(pix, *m.name.split("/"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst + ".tmp", "wb") as f:
                f.write(b)
            os.replace(dst + ".tmp", dst)
            M["frames"][m.name] = {"sha256": meta["sha256"], "bytes": meta["bytes"], "shape": list(img.shape)}
            got += 1
        tf.close()
        med = (man.get("step_end") or {}).get("median")
        M["chunks"].append({"tar": os.path.basename(tp), "asked": len(part), "got": got, "bad": bad,
                            "missing": man["missing"], "bytes": os.path.getsize(tp), "wall_s": round(wall, 1),
                            "MBps": round(os.path.getsize(tp) / 1e6 / max(wall, 1e-3), 2),
                            "step_start": man.get("step_start"), "step_end": man.get("step_end")})
        json.dump(M, open(manp + ".tmp", "w", encoding="utf-8"))
        os.replace(manp + ".tmp", manp)
        os.remove(tp)
        n_chunks += 1
        print(f"  chunk {len(M['chunks']) - 1}: {got}/{len(part)} frames ({bad} bad, {len(man['missing'])} missing), "
              f"{os.path.getsize(tp) / 1e6 if os.path.exists(tp) else M['chunks'][-1]['bytes'] / 1e6:.1f} MB in "
              f"{wall:.0f} s ({M['chunks'][-1]['MBps']} MB/s); trainer median step {med} s "
              f"(recent {(man.get('step_end') or {}).get('step_s')})", flush=True)
        if bad or man["missing"]:
            print("ZZM5_PULL_FAIL bad or missing frames -- see frames_manifest.json")
            return 1
        if med is not None and med > a.max_step_s:
            print(f"ZZM5_PULL_PAUSED trainer median step {med} s > {a.max_step_s} s -- tell the coordinator")
            return 3
        if a.max_chunks and n_chunks >= a.max_chunks:
            break
    left = sum(1 for rf in refs if rf[1] not in M["frames"])
    print(f"ZZM5_PULL_{'OK' if not left else 'PARTIAL'} {len(refs) - left:,}/{len(refs):,} frames")
    return 0


def verify(a) -> int:
    import cv2
    import numpy as np
    G = json.load(open(os.path.join(a.out, "samples_gated.json"), encoding="utf-8"))
    M = json.load(open(os.path.join(a.out, "frames_manifest.json"), encoding="utf-8"))
    rows = rows_by_key(G["source"], {x["key"] for x in G["train"] + G["val"]})
    pix = os.path.join(a.out, "pixels")
    n = nbytes = bad = 0
    shapes = {}
    for x in G["train"] + G["val"]:
        for ref in rows[x["key"]]["image"]:
            p = os.path.join(pix, *ref.split("/"))
            b = open(p, "rb").read() if os.path.exists(p) else b""
            img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) if b else None
            ok = img is not None and hashlib.sha256(b).hexdigest() == M["frames"].get(ref, {}).get("sha256")
            bad += int(not ok)
            n += 1
            nbytes += len(b)
            if img is not None:
                shapes[str(img.shape)] = shapes.get(str(img.shape), 0) + 1
    res = {"frames": n, "bytes": nbytes, "bad_or_missing": bad, "shapes": shapes,
           "samples": len(G["train"]) + len(G["val"])}
    json.dump(res, open(os.path.join(a.out, "frames_verify.json"), "w"), indent=1)
    print(json.dumps(res))
    print(f"ZZM5_VERIFY_{'OK' if bad == 0 and n == 4 * res['samples'] else 'FAIL'}")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("gate", "pull", "verify"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=100, help="samples per pull chunk (4 frames each)")
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--max-step-s", type=float, default=63.5, help="pause above this trainer median step time")
    a = ap.parse_args()
    return {"gate": gate, "pull": pull, "verify": verify}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
