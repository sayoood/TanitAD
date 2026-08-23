"""Is the 22 GB local `_pod_backup` REALLY redundant with HuggingFace?

The drumbeat item says "delete 22.6 GB of _pod_backup ckpts (verify by size on HF
first)".  Size is NOT verification — two different 30k checkpoints of the same
architecture have byte-identical SIZES (measured: v2 and v3enc are both exactly
3,416,xxx,xxx B, and v4.1 / v4.2 likewise).  Deleting on a size match would
destroy an irreplaceable checkpoint while the log said "verified".

⛔ These pods are GONE.  `_pod_backup` is the ONLY copy.  So the check is by
CONTENT: git-LFS stores each remote file's sha256 in its metadata, so a local
sha256 can be compared against every file we own on the Hub with ZERO download.

A file is SAFE-TO-DELETE only if its sha256 is found on the Hub.
Anything else is reported UNSAFE and is NOT deleted by this script — it deletes
nothing at all; it only produces the evidence.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import truststore

truststore.inject_into_ssl()
from huggingface_hub import HfApi  # noqa: E402

SPD = Path(__file__).resolve().parent
BACKUP = Path(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\_pod_backup")
OUT = SPD / "hf_reconcile.json"


def sha256_file(p: Path, chunk: int = 8 << 20) -> str:
    """Streamed, so a 3.4 GB checkpoint never lands in RAM.

    The G: mount transiently raises OSError (errno 22) — retry rather than
    reporting a false 'unreadable', which would read as 'no local copy'.
    """
    for attempt in range(5):
        try:
            h = hashlib.sha256()
            with p.open("rb") as f:
                while True:
                    b = f.read(chunk)
                    if not b:
                        break
                    h.update(b)
            return h.hexdigest()
        except OSError as e:
            if attempt == 4:
                raise
            print(f"      (G: hiccup {e.errno}, retry {attempt + 1}/4)", flush=True)
            time.sleep(4)
    raise AssertionError("unreachable")


def main() -> int:
    tok = None
    for line in Path(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\Keys.txt").read_text(
        encoding="utf-8", errors="ignore").splitlines():
        for w in line.split():
            if w.startswith("hf_"):
                tok = w.strip().strip('"\'')
                break
        if tok:
            break
    assert tok, "no hf_ token in Keys.txt"
    api = HfApi(token=tok)

    # ---- 1. every sha256 we already hold on the Hub --------------------------
    print("  indexing the Hub (sha256 of every file we own) ...", flush=True)
    remote: dict[str, list[str]] = {}
    n_repos = 0
    for lister, info_fn, kind in ((api.list_models, api.model_info, "model"),
                                  (api.list_datasets, api.dataset_info, "dataset")):
        for r in lister(author="Sayood"):
            try:
                info = info_fn(r.id, files_metadata=True)
            except Exception as e:  # a repo we cannot read is not evidence of absence
                print(f"    ! {r.id}: {type(e).__name__}", flush=True)
                continue
            n_repos += 1
            for s in info.siblings:
                oid = getattr(getattr(s, "lfs", None), "sha256", None) or (
                    (s.lfs or {}).get("sha256") if isinstance(getattr(s, "lfs", None), dict) else None)
                if oid:
                    remote.setdefault(oid, []).append(f"{kind}:{r.id}/{s.rfilename}")
    print(f"  {len(remote)} LFS objects across {n_repos} repos\n", flush=True)

    # ---- 2. local backup, hashed --------------------------------------------
    files = sorted((p for p in BACKUP.rglob("*") if p.is_file() and p.stat().st_size > 100 << 20),
                   key=lambda p: -p.stat().st_size)
    print(f"  hashing {len(files)} local files > 100 MB "
          f"({sum(p.stat().st_size for p in files) / 1e9:.1f} GB) ...\n", flush=True)
    rows, safe_bytes, unsafe_bytes = [], 0, 0
    for i, p in enumerate(files, 1):
        sz = p.stat().st_size
        t0 = time.time()
        try:
            d = sha256_file(p)
        except OSError as e:
            rows.append({"path": str(p.relative_to(BACKUP)), "bytes": sz,
                         "sha256": None, "status": f"UNREADABLE errno={e.errno}", "remote": []})
            print(f"    [{i}/{len(files)}] {p.name:<42} UNREADABLE errno={e.errno}", flush=True)
            unsafe_bytes += sz
            continue
        hits = remote.get(d, [])
        status = "SAFE-TO-DELETE" if hits else "UNSAFE — ONLY COPY"
        (safe_bytes := safe_bytes) if hits else None
        if hits:
            safe_bytes += sz
        else:
            unsafe_bytes += sz
        rows.append({"path": str(p.relative_to(BACKUP)), "bytes": sz, "sha256": d,
                     "status": status, "remote": hits})
        print(f"    [{i}/{len(files)}] {p.name:<42} {sz / 1e9:6.3f} GB  {status}"
              f"{('  <- ' + hits[0]) if hits else ''}   ({time.time() - t0:.0f}s)", flush=True)

    rep = {"_evidence_class": "MEASURED (ours; sha256 of local bytes vs HF git-LFS oid)",
           "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "backup_root": str(BACKUP), "n_files": len(rows),
           "safe_gb": round(safe_bytes / 1e9, 3), "unsafe_gb": round(unsafe_bytes / 1e9, 3),
           "verdict": ("NOTHING may be deleted — see unsafe list" if unsafe_bytes else
                       "every file is content-verified on the Hub"),
           "files": rows}
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"\n  SAFE (content-verified on the Hub): {safe_bytes / 1e9:.2f} GB")
    print(f"  UNSAFE (only local copy)          : {unsafe_bytes / 1e9:.2f} GB")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
