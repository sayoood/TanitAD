#!/usr/bin/env python3
"""READ-ONLY pod extraction for the dev-box proxy: a self-contained pod script, run over ssh, that streams a TAR
to stdout and writes NOTHING on the pod. Two phases, so no byte moves before the subset is fixed:

  phase A  emit-index    keys + byte offsets of every on-policy set line (`onpolicy/sets/onpolicy_*.jsonl`), the keys
                         of train_grow's target files, and the small files the proxy needs (the run's config.json and
                         metrics.jsonl, train_grow/calib_table.json). ~tens of MB. The manifest is then finalized
                         (`proxy_manifest.py finalize --index`): picks absent from train_grow are replaced.
  phase B  emit-extract  the 4 camera JPEGs of every manifest frame (found by BASENAME under the pixel root, as the
                         trainer's FrameStore does) and exactly the set lines of the manifest's (frame, rank) keys, read
                         by offset, re-parsed and key-checked on the pod, gzipped in <= 64 MB parts.
Both tars end with a DONE member carrying every member's sha256: a stream that broke off has no DONE and `verify`
refuses it (a truncated artifact must never read as a complete one).

POD SIDE: python3 stdlib only, run as `nice -n 19 ionice -c3 python3 -B -u -` with the script on STDIN (no file,
no .pyc on the pod), every read throttled to --max-mb-s. Growing files are read up to their size at stat time,
complete lines only. ⚠️ ionice has no effect on a FUSE/network filesystem; the in-script throttle is the guard.

    python refe/proxy_pod_extract.py emit-index   --out <pod_index.py>
    python refe/proxy_pod_extract.py emit-extract --manifest <final manifest> --index <pod_index.tar> --out <pod_extract.py>
    python refe/proxy_pod_extract.py print-command --script <pod_*.py>        # the ssh line; NOT run by this tool
    python refe/proxy_pod_extract.py verify <tar> [--manifest <m>]
    python refe/proxy_pod_extract.py selftest                                 # a fake pod tree, end to end, locally
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zlib
from pathlib import Path

POD_DEFAULTS = {"sets_dir": "/workspace/data/refe_navtrain/onpolicy/sets",
                "train_grow": "/workspace/data/refe_navtrain/train_grow",
                "run_dir": "/workspace/data/refe_runs/vitl16_navtrain10_grow_tau0.3",
                "pixels": "/workspace/data/navtrain_pixels",
                "max_mb_s": 20.0}

# ------------------------------------------------------------------------------------------------ the POD script
POD_CODE = r'''
import base64, gzip, hashlib, io, json, os, sys, tarfile, time, zlib
CFG = json.loads(zlib.decompress(base64.b64decode(CFG_B64)).decode("utf-8"))
RATE = float(CFG["max_mb_s"]) * 1e6
T0 = time.time()
ST = {"read": 0, "members": 0}
OUT = sys.stdout.buffer
TAR = tarfile.open(fileobj=OUT, mode="w|", format=tarfile.PAX_FORMAT)
DIG = {}


def log(m):
    sys.stderr.write(time.strftime("%H:%M:%S ") + m + "\n")
    sys.stderr.flush()


def throttle(n):
    ST["read"] += n
    ahead = ST["read"] / RATE - (time.time() - T0)
    if ahead > 0:
        time.sleep(ahead)


class Reader:
    def __init__(self, f):
        self.f, self.h = f, hashlib.sha256()

    def read(self, n=-1):
        b = self.f.read(n)
        self.h.update(b)
        throttle(len(b))
        return b


def add_bytes(name, data):
    ti = tarfile.TarInfo(name)
    ti.size, ti.mtime, ti.mode = len(data), int(time.time()), 0o644
    TAR.addfile(ti, io.BytesIO(data))
    ST["members"] += 1
    DIG[name] = [hashlib.sha256(data).hexdigest(), len(data)]


def add_path(name, path):
    st = os.stat(path)
    ti = tarfile.TarInfo(name)
    ti.size, ti.mtime, ti.mode = st.st_size, int(st.st_mtime), 0o644
    with open(path, "rb") as f:
        r = Reader(f)
        TAR.addfile(ti, r)
    ST["members"] += 1
    DIG[name] = [r.h.hexdigest(), st.st_size]


def lines(path, size):
    """(offset, bytes without the newline) of every COMPLETE line within the first `size` bytes"""
    with open(path, "rb") as f:
        pos, pend, pend_off = 0, b"", 0
        while pos < size:
            chunk = f.read(min(8 << 20, size - pos))
            if not chunk:
                break
            throttle(len(chunk))
            pos += len(chunk)
            data = pend + chunk
            start = 0
            while True:
                nl = data.find(b"\n", start)
                if nl < 0:
                    break
                yield pend_off + start, data[start:nl]
                start = nl + 1
            pend_off += start
            pend = data[start:]


def small_files(done):
    for name, path in (("run/config.json", os.path.join(CFG["run_dir"], "config.json")),
                       ("run/metrics.jsonl", os.path.join(CFG["run_dir"], "metrics.jsonl")),
                       ("bank/calib_table.json", os.path.join(CFG["train_grow"], "calib_table.json"))):
        if os.path.isfile(path):
            add_path(name, path)
        else:
            done.setdefault("missing_small_files", []).append(path)


def index_mode(done):
    sd = CFG["sets_dir"]
    rows = ["file\toffset\tnbytes\tlog_name\ttoken\tstep\trank\tckpt_step\tlabel_version\tok"]
    finfo = ["file\tsize\tmtime\tlines\tbad"]
    files = sorted(fn for fn in os.listdir(sd) if fn.startswith("onpolicy_") and fn.endswith(".jsonl"))
    for fn in files:
        p = os.path.join(sd, fn)
        st = os.stat(p)
        nl = nb = 0
        for off, line in lines(p, st.st_size):
            nl += 1
            try:
                r = json.loads(line)
                if r.get("kind") != "onpolicy_set":
                    raise ValueError("kind")
                rows.append("\t".join(str(x) for x in (fn, off, len(line) + 1, r["log_name"], r.get("token", ""),
                                                        int(r["step"]), int(r.get("rank", 0)), int(r["ckpt_step"]),
                                                        int(r.get("label_version", 1)), 1)))
            except Exception:
                nb += 1
                rows.append("\t".join(str(x) for x in (fn, off, len(line) + 1, "", "", -1, -1, -1, -1, 0)))
        finfo.append("\t".join(str(x) for x in (fn, st.st_size, int(st.st_mtime), nl, nb)))
        log("set file %s: %d lines (%d bad), %.1f MB" % (fn, nl, nb, st.st_size / 1e6))
    add_bytes("index/sets_files.tsv", ("\n".join(finfo) + "\n").encode("utf-8"))
    add_bytes("index/sets_index.tsv", ("\n".join(rows) + "\n").encode("utf-8"))
    krows = ["file\toffset\tnbytes\tlog_name\ttoken\tstep\trank"]
    tinfo = ["file\tsize\tmtime\tlines\tbad"]
    tg = CFG["train_grow"]
    for fn in sorted(os.listdir(tg)):
        if not (fn.startswith("targets_rank") and fn.endswith(".jsonl")):
            continue
        p = os.path.join(tg, fn)
        st = os.stat(p)
        nl = nb = 0
        for off, line in lines(p, st.st_size):
            nl += 1
            try:
                r = json.loads(line)
                krows.append("\t".join(str(x) for x in (fn, off, len(line) + 1, r["log_name"], r.get("token", ""),
                                                         int(r.get("step", 0)), int(r.get("rank", 0)))))
            except Exception:
                nb += 1
        tinfo.append("\t".join(str(x) for x in (fn, st.st_size, int(st.st_mtime), nl, nb)))
        log("train_grow %s: %d rows (%d bad)" % (fn, nl, nb))
    add_bytes("index/train_grow_files.tsv", ("\n".join(tinfo) + "\n").encode("utf-8"))
    add_bytes("index/train_grow_keys.tsv", ("\n".join(krows) + "\n").encode("utf-8"))
    small_files(done)
    done.update({"set_files": len(files), "set_lines": len(rows) - 1, "train_grow_rows": len(krows) - 1})


def extract_mode(done):
    sel = CFG["selection"]
    need = {}
    for ref in sel["images"]:
        need.setdefault(os.path.basename(ref), ref)
    found = {}
    n_walk = 0
    for dp, _dn, fns in os.walk(CFG["pixels"]):
        for fn in fns:
            n_walk += 1
            if fn in need and fn not in found:
                found[fn] = os.path.join(dp, fn)
    log("pixel walk: %d files seen, %d of %d needed found" % (n_walk, len(found), len(need)))
    for b in sorted(need):
        if b in found:
            add_path("pixels/" + need[b], found[b])
    by_file = {}
    for fn, off, nb, key in sel["set_ranges"]:
        by_file.setdefault(fn, []).append((int(off), int(nb), key))
    part, buf, acc, bad, n_ok = 0, [], 0, [], 0
    for fn in sorted(by_file):
        p = os.path.join(CFG["sets_dir"], fn)
        with open(p, "rb") as f:
            for off, nb, key in sorted(by_file[fn]):
                f.seek(off)
                line = f.read(nb)
                throttle(len(line))
                try:
                    r = json.loads(line)
                    ok = (line.endswith(b"\n") and r.get("kind") == "onpolicy_set" and
                          [r["log_name"], r.get("token", ""), int(r["step"]), int(r.get("rank", 0))] == list(key))
                except Exception:
                    ok = False
                if not ok:
                    bad.append([fn, off])
                    continue
                buf.append(line)
                n_ok += 1
                acc += len(line)
                if acc >= (64 << 20):
                    add_bytes("onpolicy/part_%04d.jsonl.gz" % part, gzip.compress(b"".join(buf), 6))
                    part, buf, acc = part + 1, [], 0
    if buf:
        add_bytes("onpolicy/part_%04d.jsonl.gz" % part, gzip.compress(b"".join(buf), 6))
    done.update({"images_needed": len(need), "images_found": len(found),
                 "missing_images": sorted(need[b] for b in need if b not in found),
                 "pixel_files_walked": n_walk, "set_lines_ok": n_ok, "bad_set_lines": bad})


done = {"mode": CFG["mode"], "cfg": {k: v for k, v in CFG.items() if k != "selection"},
        "host": os.uname().nodename if hasattr(os, "uname") else "?", "python": sys.version.split()[0],
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
if CFG["mode"] == "index":
    index_mode(done)
else:
    extract_mode(done)
done.update({"bytes_read": ST["read"], "seconds": round(time.time() - T0, 1), "members": ST["members"],
             "digests": DIG, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
data = json.dumps(done).encode("utf-8")
ti = tarfile.TarInfo("%s/DONE.json" % CFG["mode"])
ti.size, ti.mtime, ti.mode = len(data), int(time.time()), 0o644
TAR.addfile(ti, io.BytesIO(data))
TAR.close()
OUT.flush()
log("DONE %s: %d members, %.1f MB read, %.0f s" % (CFG["mode"], ST["members"], ST["read"] / 1e6, time.time() - T0))
'''


def emit(cfg: dict, out: str) -> str:
    blob = base64.b64encode(zlib.compress(json.dumps(cfg).encode("utf-8"), 9)).decode("ascii")
    text = (f"# GENERATED by refe/proxy_pod_extract.py -- mode {cfg['mode']}; run with the script on STDIN:\n"
            f"#   nice -n 19 ionice -c3 python3 -B -u -     (writes NOTHING on the pod; the tar goes to stdout)\n"
            f"CFG_B64 = {blob!r}\n" + POD_CODE)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------------ local side
def read_tar(path) -> dict:
    """{member name: bytes} of a (possibly truncated) tar; a truncated one raises or lacks DONE."""
    out = {}
    with tarfile.open(path, "r|") as t:
        for m in t:
            if m.isfile():
                out[m.name] = t.extractfile(m).read()
    return out


def verify(tar_path, manifest=None) -> dict:
    res = {"tar": str(tar_path), "ok": False}
    try:
        mem = read_tar(tar_path)
    except (tarfile.TarError, EOFError, OSError) as e:
        res["error"] = f"unreadable tar: {type(e).__name__}: {e}"
        return res
    done_name = next((n for n in mem if n.endswith("/DONE.json")), None)
    if done_name is None:
        res["error"] = "no DONE.json -- the stream broke off; a truncated tar is not an artifact"
        return res
    done = json.loads(mem[done_name])
    bad = [n for n, (sha, size) in done["digests"].items()
           if n not in mem or hashlib.sha256(mem[n]).hexdigest() != sha or len(mem[n]) != size]
    res.update({"mode": done["mode"], "members": len(mem), "digest_mismatch": bad[:10], "n_digest_mismatch": len(bad),
                "seconds": done.get("seconds"), "bytes_read": done.get("bytes_read")})
    if done["mode"] == "index":
        idx = mem.get("index/sets_index.tsv", b"").decode("utf-8").splitlines()
        files = mem.get("index/sets_files.tsv", b"").decode("utf-8").splitlines()
        n_lines = sum(int(l.split("\t")[3]) for l in files[1:]) if len(files) > 1 else 0
        res.update({"set_files": len(files) - 1, "set_lines": len(idx) - 1, "set_lines_by_files": n_lines,
                    "missing_small_files": done.get("missing_small_files", [])})
        res["ok"] = not bad and len(idx) - 1 == n_lines
    else:
        n_lines, keys = 0, set()
        for n, b in mem.items():
            if n.startswith("onpolicy/"):
                for line in gzip.decompress(b).decode("utf-8").splitlines():
                    r = json.loads(line)
                    keys.add((r["log_name"], r["token"], int(r["step"]), int(r["rank"])))
                    n_lines += 1
        imgs = [n for n in mem if n.startswith("pixels/")]
        res.update({"images": len(imgs), "missing_images": len(done.get("missing_images", [])),
                    "set_lines": n_lines, "set_keys": len(keys), "bad_set_lines": len(done.get("bad_set_lines", []))})
        ok = not bad and n_lines == done.get("set_lines_ok")
        if manifest is not None:
            m = json.load(open(manifest, encoding="utf-8"))
            want = {i for f in m["frames"] for i in f["images"]}
            got = {n[len("pixels/"):] for n in imgs}
            res["manifest_images_missing"] = len(want - got)
            res["unexpected_images"] = len(got - want)
            ok = ok and not (got - want) and len(want - got) == len(done.get("missing_images", []))
        res["ok"] = ok
    return res


def print_command(script) -> str:
    return ("C:\\Windows\\System32\\OpenSSH\\ssh.exe -i %USERPROFILE%\\.ssh\\tanitad_pod -p <PORT> -o BatchMode=yes "
            "-o ConnectTimeout=20 root@<IP> \"nice -n 19 ionice -c3 python3 -B -u -\" "
            f"< {script} > {Path(script).with_suffix('.tar')} 2> {Path(script).with_suffix('.stderr.log')}"
            "\n  (cmd.exe syntax: stdin/stdout redirection is binary-safe there; PowerShell has no '<'. Use the pod's "
            "DIRECT port, never the ssh.runpod.io proxy, which cannot carry a byte stream.)")


def selection_from(manifest, index_tar, whole=False) -> dict:
    m = json.load(open(manifest, encoding="utf-8"))
    if not str(m.get("status", "")).startswith("final"):
        raise SystemExit("REFUSING: the manifest is not final -- run proxy_manifest.py finalize --index first")
    idx = read_tar(index_tar)
    rows = idx["index/sets_index.tsv"].decode("utf-8").splitlines()[1:]
    want = {(f["log_name"], f["token"], int(f["step"]), int(rk)) for f in m["frames"] for rk in f["ranks"]}
    ranges = []
    for line in rows:
        fn, off, nb, lg, tok, st, rk, ck, lv, ok = line.split("\t")
        if ok == "1" and (lg, tok, int(st), int(rk)) in want:
            ranges.append([fn, int(off), int(nb), [lg, tok, int(st), int(rk)]])
    images = [i for f in m["frames"] for i in f["images"]]
    if len({os.path.basename(i) for i in images}) != len(images):
        raise SystemExit("REFUSING: two manifest images share a basename -- the pod lookup is by basename")
    return {"images": images, "set_ranges": ranges}


def selftest() -> int:
    """A fake pod tree, end to end, with the REAL generated scripts; every check has a mutation that must fail."""
    tmp = Path(tempfile.mkdtemp(prefix="podx_selftest_"))
    pod = tmp / "workspace"
    cfg = {"sets_dir": str(pod / "sets"), "train_grow": str(pod / "train_grow"), "run_dir": str(pod / "run"),
           "pixels": str(pod / "pixels"), "max_mb_s": 50.0}
    for d in ("sets", "train_grow", "run", "pixels/navtrain_current_1"):
        (pod / d).mkdir(parents=True)
    import random
    rng = random.Random(0)
    keys = [("log%02d" % (i % 3), "tok%04d" % i, 0) for i in range(12)]
    imgs = {}
    for lg, tok, st in keys:
        refs = []
        for cam in ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"):
            ref = f"{lg}/{cam}/{hashlib.sha1((tok + cam).encode()).hexdigest()[:16]}.jpg"
            refs.append(ref)
            if (tok, cam) != ("tok0005", "CAM_R0"):                   # one image MISSING on the pod
                p = pod / "pixels" / "navtrain_current_1" / ref
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(bytes(rng.getrandbits(8) for _ in range(3000 + 17 * len(refs))))
        imgs[(lg, tok, st)] = refs
    with open(pod / "train_grow" / "targets_rank0.jsonl", "w", newline="\n") as f:
        for k in keys:
            f.write(json.dumps({"log_name": k[0], "token": k[1], "step": 0, "rank": 0, "image": imgs[k]}) + "\n")
    (pod / "train_grow" / "calib_table.json").write_text('{"x": 1}')
    (pod / "run" / "config.json").write_text('{"argv": ["train.py"]}')
    (pod / "run" / "metrics.jsonl").write_text('{"event": "bank", "epoch": 13}\n')

    def set_line(k, rank, ck, lv, backfill=False):
        d = {"kind": "onpolicy_set", "label_version": lv, "log_name": k[0], "token": k[1], "step": 0, "rank": rank,
             "ckpt_step": ck, "traj": [[[1.0, 2.0]] * 3] * 2, "yaw": [[0.1] * 3] * 2, "targets": [{"a": 1}] * 2}
        if backfill:
            d.pop("label_version")
            d["label_version"] = lv                                    # at the END, as a backfill writes it
        return json.dumps(d)
    with open(pod / "sets" / "onpolicy_r0_w0.jsonl", "w", newline="\n") as f:
        for k in keys[:8]:
            f.write(set_line(k, 0, 4000, 3) + "\n")
        f.write("{not json\n")                                        # an unparseable line
        f.write(set_line(keys[8], 0, 4100, 3)[:40])                   # a TORN tail: not a line
    with open(pod / "sets" / "onpolicy_r0_backfill3.jsonl", "w", newline="\n") as f:
        for k in keys[:4]:
            f.write(set_line(k, 0, 3000, 3, backfill=True) + "\n")
    before = sorted((str(p.relative_to(pod)), p.stat().st_size, p.stat().st_mtime_ns) for p in pod.rglob("*"))
    checks = {}

    def run(script, out):
        with open(script, "rb") as si, open(out, "wb") as so:
            p = subprocess.run([sys.executable, "-B", "-u", "-"], stdin=si, stdout=so, stderr=subprocess.PIPE,
                               cwd=str(tmp))
        return p.returncode, p.stderr.decode("utf-8", "replace")
    emit(dict(cfg, mode="index"), str(tmp / "pod_index.py"))
    rc, err = run(tmp / "pod_index.py", tmp / "pod_index.tar")
    v = verify(tmp / "pod_index.tar")
    idx = read_tar(tmp / "pod_index.tar")
    rows = idx["index/sets_index.tsv"].decode().splitlines()[1:]
    lv_end = [r for r in rows if r.split("\t")[0] == "onpolicy_r0_backfill3.jsonl" and r.split("\t")[8] == "3"]
    checks["index: runs, verifies, counts"] = (rc == 0 and v["ok"] and v["set_lines"] == 13 and
                                               sum(r.endswith("\t0") for r in rows) == 1 and len(lv_end) == 4,
                                               {"rc": rc, "set_lines": v.get("set_lines"), "backfill_label_version_found":
                                                len(lv_end), "bad_lines": sum(r.endswith("\t0") for r in rows)})
    # the CONTRACT with proxy_manifest.py: its reader must accept what the real pod index script wrote.
    # (MEASURED 2026-09-27: it looked for INDEX_DONE.json while the pod script writes index/DONE.json, and would have
    # refused the real phase-A tar AFTER the transfer; caught by a synthetic pre-flight.)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import proxy_manifest as PM
    try:
        li = PM.load_index(tmp / "pod_index.tar")
        contract = (li["done"] is not None and len(li["train_grow"]) == 12 and
                    sum(len(v) for v in li["sets"].values()) == 12)
        cinfo = {"train_grow_keys": len(li["train_grow"]), "set_lines_ok": sum(len(v) for v in li["sets"].values())}
    except SystemExit as e:
        contract, cinfo = False, {"refused": str(e)[:80]}
    checks["contract: proxy_manifest reads the real index tar"] = (contract, cinfo)
    # the offsets are the real byte offsets of the lines on the pod
    srcb = (pod / "sets" / "onpolicy_r0_w0.jsonl").read_bytes()
    offs_ok = all(srcb[int(r.split("\t")[1]):int(r.split("\t")[1]) + int(r.split("\t")[2])].endswith(b"\n")
                  for r in rows if r.split("\t")[0] == "onpolicy_r0_w0.jsonl")
    checks["index: byte offsets land on line ends"] = (offs_ok, {})
    # a final manifest for 5 frames (one with the missing image), then phase B
    man = {"status": "final", "frames": [{"log_name": k[0], "token": k[1], "step": 0, "images": imgs[k],
                                          "ranks": {"0": {}}} for k in keys[3:8]]}
    (tmp / "man.json").write_text(json.dumps(man))
    sel = selection_from(tmp / "man.json", tmp / "pod_index.tar")
    emit(dict(cfg, mode="extract", selection=sel), str(tmp / "pod_extract.py"))
    rc, err = run(tmp / "pod_extract.py", tmp / "pod_extract.tar")
    v = verify(tmp / "pod_extract.tar", tmp / "man.json")
    ex = read_tar(tmp / "pod_extract.tar")
    same_imgs = all(ex["pixels/" + ref] == (pod / "pixels" / "navtrain_current_1" / ref).read_bytes()
                    for k in keys[3:8] for ref in imgs[k] if ("pixels/" + ref) in ex)
    got_lines = [l for n, b in ex.items() if n.startswith("onpolicy/") for l in gzip.decompress(b).splitlines()]
    src_lines = [l for p in ("onpolicy_r0_w0.jsonl", "onpolicy_r0_backfill3.jsonl")
                 for l in (pod / "sets" / p).read_bytes().splitlines()
                 if l.startswith(b"{\"kind") and any(f'"{k[1]}"'.encode() in l for k in keys[3:8])]
    checks["extract: runs, verifies, bytes identical"] = (
        rc == 0 and v["ok"] and v["images"] == 19 and v["missing_images"] == 1 and same_imgs
        and sorted(got_lines) == sorted(src_lines),
        {"rc": rc, "images": v.get("images"), "missing": v.get("missing_images"), "set_lines": len(got_lines),
         "expected_set_lines": len(src_lines)})
    after = sorted((str(p.relative_to(pod)), p.stat().st_size, p.stat().st_mtime_ns) for p in pod.rglob("*"))
    checks["nothing written on the pod tree"] = (before == after, {"entries": len(after)})
    # MUTATIONS that must be caught
    b = (tmp / "pod_extract.tar").read_bytes()
    (tmp / "trunc.tar").write_bytes(b[:len(b) // 2])
    checks["MUTATION truncated stream -> refused"] = (not verify(tmp / "trunc.tar")["ok"], {})
    ref0 = imgs[keys[3]][0]
    pos = b.find(ex["pixels/" + ref0][:64])
    flipped = bytearray(b)
    flipped[pos + 10] ^= 0xFF
    (tmp / "flip.tar").write_bytes(bytes(flipped))
    vf = verify(tmp / "flip.tar")
    checks["MUTATION one flipped image byte -> refused"] = (not vf["ok"] and vf.get("n_digest_mismatch") == 1, {})
    sel_bad = dict(sel)
    sel_bad["set_ranges"] = [[fn, off + 1, nb, key] for fn, off, nb, key in sel["set_ranges"][:1]] + sel["set_ranges"][1:]
    emit(dict(cfg, mode="extract", selection=sel_bad), str(tmp / "pod_extract_bad.py"))
    rc, err = run(tmp / "pod_extract_bad.py", tmp / "pod_extract_bad.tar")
    done_bad = json.loads(read_tar(tmp / "pod_extract_bad.tar")["extract/DONE.json"])
    checks["MUTATION shifted offset -> line rejected on the pod"] = (len(done_bad["bad_set_lines"]) == 1, {})
    ok = all(c[0] for c in checks.values())
    for name, (good, vals) in checks.items():
        print(f"  [{'PASS' if good else 'FAIL'}] {name}  {json.dumps(vals)}")
    shutil.rmtree(tmp, ignore_errors=True)
    print("ZZPODX_SELFTEST_OK" if ok else "ZZPODX_SELFTEST_FAIL")
    return 0 if ok else 1


def unpack(tars, root, sets_out) -> dict:
    """phase-B tars -> `<root>/<log>_<CAM>.zip` containers (ZIP_STORED, member = the frame's basename: exactly what
    train.FrameStore reads) + the on-policy parts copied to `sets_out`. Appends across tars; a member already in its
    container is skipped (idempotent). exFAT's 1 MiB clusters make 40,000 loose JPEGs cost ~40 GB; containers do not."""
    import zipfile
    root, sets_out = Path(root), Path(sets_out)
    root.mkdir(parents=True, exist_ok=True)
    sets_out.mkdir(parents=True, exist_ok=True)
    n_img = n_skip = n_sets = 0
    for tp in tars:
        with tarfile.open(tp, "r") as t:                               # random access: group members by container
            groups: dict = {}
            for m in t.getmembers():
                if m.isfile() and m.name.startswith("pixels/"):
                    ref = m.name[len("pixels/"):]
                    log, cam, base = ref.split("/")[0], ref.split("/")[1], os.path.basename(ref)
                    groups.setdefault(f"{log}_{cam}.zip", []).append((base, m))
                elif m.isfile() and m.name.startswith("onpolicy/"):
                    dst = sets_out / f"{Path(tp).stem}_{Path(m.name).name}"
                    if not dst.exists():
                        dst.write_bytes(t.extractfile(m).read())
                    n_sets += 1
            for zname, members in sorted(groups.items()):
                zp = root / zname
                have = set()
                if zp.exists():
                    with zipfile.ZipFile(zp) as z:
                        have = set(z.namelist())
                with zipfile.ZipFile(zp, "a", compression=zipfile.ZIP_STORED) as z:
                    for base, m in members:
                        if base in have:
                            n_skip += 1
                            continue
                        z.writestr(base, t.extractfile(m).read())
                        n_img += 1
    return {"images_written": n_img, "images_already_present": n_skip, "set_parts": n_sets,
            "containers": len(list(root.glob("*.zip")))}


def check_resolvable(manifest, root) -> dict:
    """every manifest image through the TRAINER's own FrameStore (resolvable + read + JPEG magic)"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train as T
    m = json.load(open(manifest, encoding="utf-8"))
    fs = T.FrameStore(str(root))
    bad, n = [], 0
    for f in m["frames"]:
        for ref in f["images"]:
            n += 1
            b = fs.read(ref, f["log_name"])
            if b is None or b[:2] != b"\xff\xd8":
                bad.append(ref)
    return {"images": n, "unreadable_or_not_jpeg": len(bad), "examples": bad[:3]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("emit-index", "emit-extract", "print-command", "verify", "selftest", "unpack",
                                    "check"))
    ap.add_argument("--tars", default="", help="unpack: comma-separated phase-B tars")
    ap.add_argument("--root", default=None, help="unpack/check: the pixel root (zip containers)")
    ap.add_argument("--sets-out", default=None, help="unpack: where the on-policy parts go")
    ap.add_argument("--out")
    ap.add_argument("--manifest")
    ap.add_argument("--index")
    ap.add_argument("--script")
    ap.add_argument("--tar")
    ap.add_argument("--max-mb-s", type=float, default=POD_DEFAULTS["max_mb_s"])
    for k in ("sets_dir", "train_grow", "run_dir", "pixels"):
        ap.add_argument("--" + k.replace("_", "-"), default=POD_DEFAULTS[k])
    a = ap.parse_args()
    cfg = {"sets_dir": a.sets_dir, "train_grow": a.train_grow, "run_dir": a.run_dir, "pixels": a.pixels,
           "max_mb_s": a.max_mb_s}
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "unpack":
        res = unpack([t for t in a.tars.split(",") if t], a.root, a.sets_out)
        print(json.dumps(res))
        return 0
    if a.cmd == "check":
        res = check_resolvable(a.manifest, a.root)
        print(json.dumps(res))
        return 0 if res["unreadable_or_not_jpeg"] == 0 else 1
    if a.cmd == "emit-index":
        sha = emit(dict(cfg, mode="index"), a.out)
        print(f"  wrote {a.out} (sha256 {sha[:16]}); run it with:\n  {print_command(a.out)}")
        return 0
    if a.cmd == "emit-extract":
        sel = selection_from(a.manifest, a.index)
        sha = emit(dict(cfg, mode="extract", selection=sel), a.out)
        print(f"  wrote {a.out} (sha256 {sha[:16]}): {len(sel['images']):,} images, {len(sel['set_ranges']):,} set "
              f"lines ({sum(r[2] for r in sel['set_ranges']) / 1e9:.2f} GB before gzip); run it with:\n  "
              f"{print_command(a.out)}")
        return 0
    if a.cmd == "print-command":
        print(print_command(a.script))
        return 0
    res = verify(a.tar, a.manifest)
    print(json.dumps(res, indent=1))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
