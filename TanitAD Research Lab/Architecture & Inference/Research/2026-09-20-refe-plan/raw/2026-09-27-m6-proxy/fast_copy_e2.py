#!/usr/bin/env python3
"""The ONE NVMe copy the coordinator authorised at 22:34 for the AMENDMENT-1 extension arms only (P0e2 P1e2 W0e2 W1e2).
Guards (coordinator): refuse unless C: keeps >= 30 GiB AFTER the copy; while the copy exists, every 15 s, C: < 15 GiB
-> stop every proxy_train.py reading the copy, delete the copy, record it (the chain then reads D:). Verification as
before: bytes read from D: == the manifest's sha256 == the copy read back; then `proxy_cache.py verify` on the copy;
the record is APPENDED to fast_copy.json. Writes COPY_VERIFIED beside the copy when (and only when) it is usable.
The chain deletes the copy after the extension's last reader; this guard exits when the copy is gone."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[2]
ROOT = PKG / "raw" / "2026-09-27-m6-proxy"
TC = Path("D:/Projects/TanitAD/data/refe_proxy/cache_train_ep015")
TCF = Path("C:/Users/Admin/refe_proxy_fast/cache_train_ep015")
MARK = TCF.parent / "COPY_VERIFIED"
FLOOR_AFTER, EMERGENCY = 30.0, 15.0
PY = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
LOG = ROOT / "chain.log"


def log(m):
    line = f"{time.strftime('%H:%M:%S')} COPY-E2: {m}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def free():
    return shutil.disk_usage("C:/").free / 2 ** 30


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()


def copy_hashed(src, dst):
    h = hashlib.sha256()
    tmp = dst.with_name(dst.name + ".part")
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        for b in iter(lambda: fi.read(1 << 23), b""):
            h.update(b)
            fo.write(b)
        fo.flush()
        os.fsync(fo.fileno())
    os.replace(tmp, dst)
    return h.hexdigest()


def record(key, val):
    fj = ROOT / "fast_copy.json"
    d = json.load(open(fj)) if fj.exists() else {}
    d[key] = val
    json.dump(d, open(fj, "w"), indent=1)


def emergency(why):
    before = free()
    subprocess.run(["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Process | Where-Object { "
                    "$_.CommandLine -match 'proxy_train.py' -and $_.CommandLine -match 'refe_proxy_fast' } | "
                    "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"],
                   capture_output=True)
    time.sleep(5)
    for k in range(20):
        shutil.rmtree(TCF.parent, ignore_errors=True)
        if not TCF.parent.exists():
            break
        time.sleep(3)
    log(f"EMERGENCY ({why}): readers stopped, copy deleted, C: free {before:.2f} -> {free():.2f} GiB; arms read D:")
    record(f"e2_emergency_{time.strftime('%H%M%S')}", {"why": why, "c_free_gib_before_delete": round(before, 2),
                                                       "c_free_gib_after_delete": round(free(), 2),
                                                       "still_exists": TCF.parent.exists()})


def main():
    mf = json.load(open(TC / "manifest.json"))
    files = [(sh[k], sh[f"{k}_sha256"]) for sh in mf["shards"] for k in ("scene", "visual")]
    total = sum((TC / n).stat().st_size for n, _ in files) / 2 ** 30
    f0 = free()
    if f0 - total < FLOOR_AFTER:
        log(f"REFUSED: C: {f0:.2f} GiB - {total:.2f} GiB copy < {FLOOR_AFTER} GiB floor; the extension reads D:")
        record(f"e2_refused_{time.strftime('%H%M%S')}", {"c_free_gib": round(f0, 2), "copy_gib": round(total, 2)})
        return 0
    TCF.mkdir(parents=True, exist_ok=True)
    log(f"start: C: free {f0:.2f} GiB, copy {total:.2f} GiB, floor after {FLOOR_AFTER}, emergency {EMERGENCY}")
    t0, rec, nbytes = time.time(), [], 0
    for name, want in files:
        src, dst = TC / name, TCF / name
        if free() - src.stat().st_size / 2 ** 30 < FLOOR_AFTER:
            emergency(f"{name} would take C: below the {FLOOR_AFTER} GiB floor")
            return 0
        s_ = copy_hashed(src, dst)
        g_ = sha(dst)
        rec.append([name, want, s_, g_])
        if not (s_ == want == g_):
            emergency(f"{name} MISMATCH source {s_[:12]} manifest {want[:12]} copy {g_[:12]}")
            return 0
        nbytes += dst.stat().st_size
        if free() < EMERGENCY:
            emergency(f"C: {free():.2f} GiB < {EMERGENCY} after {name}")
            return 0
    for name in ("frames.jsonl", "rows.jsonl", "manifest.json"):
        s_ = copy_hashed(TC / name, TCF / name)
        g_ = sha(TCF / name)
        rec.append([name, None, s_, g_])
        if s_ != g_:
            emergency(f"{name} MISMATCH")
            return 0
    v = subprocess.run([PY, str(PKG / "refe" / "proxy_cache.py"), "verify", "--out", str(TCF)], capture_output=True,
                       text=True)
    if v.returncode != 0:
        emergency("verify of the copy failed")
        return 0
    f1 = free()
    record(f"copy_e2_{time.strftime('%H%M%S')}", {
        "path": str(TCF), "source": str(TC), "for": "AMENDMENT 1 extension arms only (P0e2 P1e2 W0e2 W1e2)",
        "authorised": "coordinator 2026-09-27 22:34 (floor >= 30 GiB after, emergency < 15 GiB)",
        "files_manifest_source_copy_sha256": rec, "bytes_copied": nbytes, "seconds": round(time.time() - t0, 1),
        "c_free_gib_before": round(f0, 2), "c_free_gib_after": round(f1, 2), "verify_rc": v.returncode,
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    json.dump({"verified_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "files": len(rec)}, open(MARK, "w"))
    log(f"VERIFIED: {len(rec)} files, {nbytes / 1e9:.2f} GB in {time.time() - t0:.0f} s, C: free {f1:.2f} GiB; "
        f"guard on (every 15 s, C: < {EMERGENCY} GiB deletes it)")
    while TCF.parent.exists():
        if free() < EMERGENCY:
            emergency(f"C: {free():.2f} GiB < {EMERGENCY}")
            return 0
        time.sleep(15)
    log("the copy is gone (deleted after its last reader) -- guard off")
    return 0


if __name__ == "__main__":
    sys.exit(main())
