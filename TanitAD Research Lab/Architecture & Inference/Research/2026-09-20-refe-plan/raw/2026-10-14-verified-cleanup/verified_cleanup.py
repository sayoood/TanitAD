"""Delete a pod-local file ONLY when the PI-approved rule holds (2026-10-14, "Delete verified copies"): the HF remote listing proves an
identical copy -- same size AND same LFS sha256 -- of that exact file.  Anything not proven stays, and is reported.

    python verified_cleanup.py --repo Sayood/x --dir /workspace/data/... [--repo ...] [--exclude <subpath>]... [--min-mb 10] [--delete]

Default is a DRY RUN.  Each decision is one JSON line in --log (append).  The token comes from the HF_TOKEN environment variable and is never printed.
Only files >= --min-mb are considered (LFS-sized; small files do not matter for disk and carry no LFS sha256)."""
import argparse
import hashlib
import json
import os
import sys
import time


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def remote_index(repos, token):
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    idx = {}
    for spec in repos:
        rtype, repo = spec.split(":", 1) if ":" in spec else ("model", spec)     # "dataset:Sayood/x" for a dataset repo
        for e in api.list_repo_tree(repo, repo_type=rtype, recursive=True, expand=True):
            lfs = getattr(e, "lfs", None)
            if lfs is None:
                continue                                  # a folder, or a non-LFS small file: never a proof here
            sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            size = lfs.get("size") if isinstance(lfs, dict) else getattr(lfs, "size", None)
            if sha and size:
                idx.setdefault((os.path.basename(e.path), int(size)), []).append({"repo": repo, "path": e.path, "sha256": sha})
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append", required=True)
    ap.add_argument("--dir", action="append", required=True)
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--min-mb", type=float, default=10.0)
    ap.add_argument("--delete", action="store_true")
    ap.add_argument("--log", default="/workspace/data/verified_cleanup.jsonl")
    a = ap.parse_args()
    token = os.environ.get("HF_TOKEN") or sys.exit("HF_TOKEN is not set in the environment")
    idx = remote_index(a.repo, token)
    print(f"ZZCLEAN_REMOTE {sum(len(v) for v in idx.values())} LFS files in {len(a.repo)} repo(s)", flush=True)
    freed = kept = 0
    with open(a.log, "a", encoding="utf-8") as log:
        for d in a.dir:
            for root, dirs, files in os.walk(d):
                dirs[:] = [x for x in dirs if not any(os.path.join(root, x).rstrip("/").endswith(ex.rstrip("/")) for ex in a.exclude)]
                for fn in sorted(files):
                    p = os.path.join(root, fn)
                    if os.path.islink(p) or any(p.endswith(ex) for ex in a.exclude):
                        continue
                    size = os.path.getsize(p)
                    if size < a.min_mb * 1e6:
                        continue
                    cands = idx.get((fn, size), [])
                    rec = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "path": p, "size": size, "dry_run": not a.delete}
                    if not cands:
                        rec.update(verdict="KEEP", why="no HF file with this name and size")
                    else:
                        local = sha256(p)
                        hit = [c for c in cands if c["sha256"] == local]
                        rec["sha256"] = local
                        if not hit:
                            rec.update(verdict="KEEP", why="name+size match on HF but sha256 differs", remote=cands[:3])
                        else:
                            rec.update(verdict="DELETE" if a.delete else "WOULD_DELETE", proof=hit[0])
                            if a.delete:
                                os.remove(p)
                    if rec["verdict"] in ("DELETE", "WOULD_DELETE"):
                        freed += size
                    else:
                        kept += size
                    log.write(json.dumps(rec) + "\n"); log.flush()
                    print(f"{rec['verdict']:13s} {size/1e9:7.3f} GB  {p}  {rec.get('proof', {}).get('repo', '') or rec.get('why', '')}", flush=True)
    print(f"ZZCLEAN_{'DONE' if a.delete else 'DRY'} freed_or_would {freed/1e9:.3f} GB  kept {kept/1e9:.3f} GB", flush=True)


if __name__ == "__main__":
    sys.exit(main())
