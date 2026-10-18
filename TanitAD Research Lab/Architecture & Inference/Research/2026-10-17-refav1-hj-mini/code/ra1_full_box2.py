"""RA-1 FULL CORPUS on box 2 (../PREREG_RA1_FULL.md §Data): build the native-256 v2ep of the 3,972 train clips not yet built,
pool-encode them into the full-corpus cache root, and delete each chunk's intermediates once its encode is verified -- so the
bus-powered D: never holds the whole v2ep (~137 GB).

Run as ONE box-2 job under SAFE MODE (`%RUNSAFE%` + env.cmd: children inherit below-normal priority, affinity 0xFF, AVX2):
    %RUNSAFE% %PY% -u ra1_full_box2.py [--chunk 400] [--workers 4] [--only-control] [--dry-run]

⛔ A SEPARATE CACHE ROOT (`cache_full`), seeded with a copy of the 600-clip root. The RA-1 trainer takes EVERY clip in its cache's
manifest minus the holdout (the queue passes no --n-clips), so encoding the new clips into `cache` would make every later 540-clip
stage silently train on 4,512.

Order, and why:
  0. CONTROL (known value): rebuild 2 clips the dev box already built (S4b). The dev box rebuilt both from the full stage
     BYTE-IDENTICAL to its own S4b build (MEASURED 2026-10-18), so a difference here is box 2's. PASS = sha identical, or the decoded
     frames/actions/poses bit-equal (disclosed); anything else => exit 10, nothing else runs.
  1. seed `cache_full` with `cache`'s train + eval splits and native_tracks.pt (once; C1 of the final verify re-checks every md5).
  2. per chunk, resumable (state file), CPU and GPU NEVER overlap (the box-2 crash hypothesis is sustained CPU load alongside GPU):
     a. sha256 of every chunk mp4 against HF's own ledger (camera_sha256.json)  -> a torn transfer stops the run (exit 12)
     b. build    stack/scripts/pod_build_b1_epcache.py, UNCHANGED, CPU only, --workers 4 (1 thread each)
     c. encode   pull_encode_pooled.py --only-file <chunk>, the 600-clip encode's flags (grids 4x10,8x20, local source, keep-src),
                 up to 3 tries (it resumes; a user-mode native crash in torch_cpu.dll was measured on this box)
     d. verify   every chunk clip is in the manifest dir, and verify_cache.py --allow-partial passes (C1 all files, C2 on 40, C3, C4)
     e. delete   the chunk's v2ep, mp4 and timestamps -- only after d.
  3. final verify_cache.py over ALL 4,572 (--n 0: C4 no missing clip; --sample 0: C2 on every file).
Exit codes: 0 done; 10 control mismatch; 11 disk guard; 12 mp4 sha mismatch; 13 build failed; 14 encode failed; 15 verify failed."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

TAN = Path(os.environ.get("RA1FULL_ROOT", "D:/tanitad"))
REPO = TAN / "repo"
PKG = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-refav1-hj-mini" / "code"
STAGE = TAN / "data" / "trainall_stage"
CAM = STAGE / "r0" / "camera_front_wide"
V2EP = TAN / "data" / "trainrest-256"
REF600 = TAN / "data" / "train600-256"
CTL_OUT = TAN / "data" / "_ctl256"
SRC_CACHE = TAN / "cache"
CACHE = TAN / "cache_full"
STATE = TAN / "data" / "ra1_full_state.json"
MIN_FREE_GB = float(os.environ.get("RA1FULL_MIN_FREE_GB", "35"))   # box 2 D: had 179.8 GB free; the build peaks at ~64 GB more


def log(msg):
    print(f"[ra1full {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def free_gb(path=TAN):
    return shutil.disk_usage(str(path)).free / 2 ** 30


def run(cmd, env_extra, dry):
    log("RUN " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    if dry:
        return 0
    env = dict(os.environ, **env_extra)
    return subprocess.run(cmd, env=env, cwd=str(PKG)).returncode


def label_flags():
    """The encoder's label/listing flags, exactly env.cmd's LABELFLAGS (the 600-clip encode's)."""
    lf = os.environ.get("LABELFLAGS", "")
    if not lf:
        raise SystemExit("LABELFLAGS is not set: run under `call D:/tanitad/env.cmd`")
    return lf.split()


def builder_env():
    return {"CUDA_VISIBLE_DEVICES": "-1", "TANITAD_STACK": str(REPO / "stack"), "PYTHONPATH": str(REPO / "stack"),
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "V2_TORCH_THREADS": "1", "PAI_DECODE_THREADS": "1"}


def build(clips_file, out, workers, dry):
    out.mkdir(parents=True, exist_ok=True)
    return run([sys.executable, "-u", str(REPO / "stack" / "scripts" / "pod_build_b1_epcache.py"), "--out", str(out), "--root", str(STAGE),
                "--workers", str(workers), "--clips", str(clips_file)], builder_env(), dry)


def encode_cmd(chunk_file):
    return [sys.executable, "-u", "pull_encode_pooled.py", "--split", "train", "--n", "0", "--only-file", str(chunk_file),
            "--grids", "4x10,8x20", "--source", "local", "--src-dir", str(V2EP), "--src-geom", "256", "--keep-src",
            "--prefetch-workers", "0"] + label_flags() + ["--out", str(CACHE), "--device", "cuda"]


def verify_cmd(sample, partial):
    return ([sys.executable, "-u", "verify_cache.py", "--split", "train", "--n", "0", "--grids", "4x10,8x20", "--sample", str(sample),
             "--redo", "0"] + (["--allow-partial"] if partial else []) + label_flags() + ["--out", str(CACHE), "--device", "cuda"])


def same_content(p, q):
    """Decoded frames + actions + poses bit-equal (what reaches the encoder). Used only when the file bytes differ."""
    sys.path.insert(0, str(REPO / "stack" / "scripts"))
    import torch
    import v2_compressed as V
    x, y = V.load_compressed(str(p)), V.load_compressed(str(q))
    return all(torch.equal(getattr(x, k), getattr(y, k)) for k in ("frames", "actions", "poses"))


def control(a):
    """PREREG_RA1_FULL §C0. PASS = sha identical; PASS-CONTENT = bytes differ but the decoded frames/actions/poses are bit-equal
    (disclosed); anything else = STOP."""
    exp = json.load(open(STAGE / "ctl_expected.json", encoding="utf-8"))     # {clip: sha256 of the dev box's S4b v2ep}
    f = STAGE / "ctl_clips.txt"
    f.write_text("\n".join(exp) + "\n", encoding="utf-8")
    if build(f, CTL_OUT, min(a.workers, len(exp)), a.dry_run) != 0:
        log("ZZCONTROL-BUILD-FAILED"); return 13
    if a.dry_run:
        return 0
    res = {}
    for c, want_sha in exp.items():
        p, ref = CTL_OUT / f"{c}.v2ep.pt", REF600 / f"{c}.v2ep.pt"
        r = dict(expected=want_sha, got=sha256(p) if p.exists() else None, ref=sha256(ref) if ref.exists() else None)
        r["verdict"] = ("IDENTICAL" if r["got"] == want_sha else
                        "NO-OUTPUT" if r["got"] is None else
                        "REF-CORRUPT" if r["ref"] != want_sha else
                        "CONTENT-IDENTICAL" if same_content(p, ref) else "MISMATCH")
        res[c] = r
    ok = all(r["verdict"] in ("IDENTICAL", "CONTENT-IDENTICAL") for r in res.values())
    json.dump(dict(clips=res, ok=ok), open(STAGE / "ctl_result.json", "w", encoding="utf-8"), indent=1)
    log(f"control {[(c[:8], r['verdict']) for c, r in res.items()]}")
    if not ok:
        log("ZZCONTROL-MISMATCH"); return 10
    log(f"ZZCONTROL-OK {len(exp)} clips")
    return 0


def seed_cache(dry):
    """cache_full = a copy of cache's train + eval splits and native_tracks.pt (never the encoder's scratch)."""
    if (CACHE / "train" / "manifest.jsonl").exists():
        return
    log(f"seeding {CACHE} from {SRC_CACHE}")
    if dry:
        return
    ign = shutil.ignore_patterns(".tmp")
    for split in ("train", "eval"):
        shutil.copytree(SRC_CACHE / split, CACHE / split, ignore=ign, dirs_exist_ok=True)
    shutil.copy2(SRC_CACHE / "native_tracks.pt", CACHE / "native_tracks.pt")
    n = {s: len(list((CACHE / s / "clips").glob("*.pt"))) for s in ("train", "eval")}
    log(f"seeded {n}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only-control", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    rest = [l.strip() for l in open(STAGE / "clips_rest.txt", encoding="utf-8") if l.strip()]
    ledger = json.load(open(STAGE / "camera_sha256.json", encoding="utf-8"))           # {clip: {"sha256": ..., "bytes": ...}}
    st = json.load(open(STATE, encoding="utf-8")) if STATE.exists() else {"control": None, "done": []}
    log(f"rest={len(rest)} chunk={a.chunk} workers={a.workers} done_chunks={len(st['done'])} free={free_gb():.1f} GB")
    if st["control"] != "OK":
        rc = control(a)
        if rc:
            return rc
        if not a.dry_run:
            st["control"] = "OK"; json.dump(st, open(STATE, "w", encoding="utf-8"), indent=1)
    if a.only_control:
        return 0
    seed_cache(a.dry_run)
    chunks = [rest[i:i + a.chunk] for i in range(0, len(rest), a.chunk)]
    for k, ch in enumerate(chunks):
        if k in st["done"]:
            continue
        if free_gb() < MIN_FREE_GB:
            log(f"ZZDISK-GUARD free {free_gb():.1f} GB < {MIN_FREE_GB}"); return 11
        t0 = time.time()
        cf = STAGE / f"chunk_{k:02d}.txt"
        cf.write_text("\n".join(ch) + "\n", encoding="utf-8")
        todo = [c for c in ch if not (CACHE / "train" / "clips" / f"{c}.pt").exists()]
        bad = [] if a.dry_run else [c for c in todo if sha256(CAM / f"{c}.mp4") != ledger[c]["sha256"]]
        if bad:
            log(f"ZZMP4-SHA-MISMATCH chunk {k}: {bad[:10]} ({len(bad)})"); return 12
        log(f"chunk {k + 1}/{len(chunks)}: {len(ch)} clips, {len(todo)} not yet cached; mp4 sha ok")
        if todo:
            if build(cf, V2EP, a.workers, a.dry_run) != 0:
                log(f"ZZBUILD-FAILED chunk {k}"); return 13
            missing = [] if a.dry_run else [c for c in todo if not (V2EP / f"{c}.v2ep.pt").exists()]
            if missing:
                log(f"ZZBUILD-INCOMPLETE chunk {k}: {missing[:10]} ({len(missing)})"); return 13
            rc = 1
            for t in range(1, 4):
                rc = run(encode_cmd(cf), {}, a.dry_run)
                log(f"encode chunk {k} try {t} rc={rc}")
                if rc == 0:
                    break
            if rc != 0:
                log(f"ZZENCODE-FAILED chunk {k}"); return 14
        miss = [] if a.dry_run else [c for c in ch if not (CACHE / "train" / "clips" / f"{c}.pt").exists()]
        if miss:
            log(f"ZZVERIFY-MISSING chunk {k}: {miss[:10]} ({len(miss)})"); return 15
        if run(verify_cmd(40, True), {}, a.dry_run) != 0:
            log(f"ZZVERIFY-FAILED chunk {k}"); return 15
        if not a.dry_run:
            for c in ch:                                            # intermediates go only AFTER the verified encode
                for p in (V2EP / f"{c}.v2ep.pt", CAM / f"{c}.mp4", CAM / f"{c}.timestamps.parquet"):
                    p.unlink(missing_ok=True)
            st["done"].append(k); json.dump(st, open(STATE, "w", encoding="utf-8"), indent=1)
        log(f"ZZCHUNK-DONE {k + 1}/{len(chunks)} in {time.time() - t0:.0f}s, free {free_gb():.1f} GB")
    if run(verify_cmd(0, False), {}, a.dry_run) != 0:
        log("ZZFINAL-VERIFY-FAILED"); return 15
    log("ZZRA1FULL-DONE all 4,572 train clips cached in " + str(CACHE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
