#!/usr/bin/env python3
"""Report code that exists on a POD but not in git — "a pod is not storage".

The 2026-07-20 audit found the program's dominant failure mode was good work
stranded outside the repo: REF-B v2's architecture and the ENTIRE TanitEval
harness each lived on a single pod disk, with no copy in git and no backup.
Losing either pod would have destroyed a top-3 arm and the evidence base for
every headline number we have published.

This script makes that condition detectable instead of discoverable-by-audit.
For every candidate source file on each pod it asks two questions:

  * does a file with this BASENAME exist anywhere in the repo?  -> if not,
    the file is POD_ONLY: it exists in exactly one place on earth.
  * does the CONTENT match the repo copy?                       -> if not,
    it is DRIFTED: the pod is running something we cannot rebuild.

Exit code is non-zero when anything POD_ONLY is found, so it can gate a
nightly job.

Usage:
    python3 stack/scripts/pod_git_drift.py                    # all known pods
    python3 stack/scripts/pod_git_drift.py --pods tanitad-pod tanitad-eval
    python3 stack/scripts/pod_git_drift.py --json report.json

Read-only: it only runs `find`/`sha256sum` over ssh and never writes to a pod.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_PODS = ["tanitad-pod", "tanitad-pod2", "tanitad-pod3", "tanitad-eval"]

# Where humans and agents actually stash things on a pod.
SEARCH_ROOTS = ["/root", "/workspace"]
SUFFIXES = (".py", ".sh")

# Noise that is never a deliverable.
EXCLUDE_PARTS = (
    "__pycache__", "site-packages", "dist-packages", "/.git/", "/node_modules/",
    "/.cache/", "/venv/", "/.venv/", "/miniconda", "/anaconda",
)

# Verdicts
POD_ONLY = "POD_ONLY"      # no file of this name in the repo at all — rescue it
DRIFTED = "DRIFTED"        # name exists but content differs — pod runs something else
IN_GIT = "IN_GIT"          # byte-identical to a repo copy


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def repo_index(repo_root: str | Path) -> dict:
    """Index the repo's tracked-ish source files by content hash and basename.

    Returns {"by_hash": {sha: [paths]}, "by_name": {basename: {sha, ...}}}.
    Pure function over the filesystem so the comparison logic stays testable.
    """
    root = Path(repo_root)
    by_hash: dict[str, list[str]] = defaultdict(list)
    by_name: dict[str, set[str]] = defaultdict(set)

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in SUFFIXES:
            continue
        posix = path.as_posix()
        if any(part in posix for part in EXCLUDE_PARTS):
            continue
        try:
            digest = sha256_bytes(path.read_bytes())
        except OSError:
            continue
        rel = path.relative_to(root).as_posix()
        by_hash[digest].append(rel)
        by_name[path.name].add(digest)

    return {"by_hash": dict(by_hash), "by_name": {k: v for k, v in by_name.items()}}


def classify(pod_files: list[tuple[str, str]], index: dict) -> list[dict]:
    """Classify (sha256, pod_path) pairs against a repo index.

    This is the heart of the check and is deliberately free of ssh/IO so it can
    be unit-tested directly.
    """
    by_hash = index["by_hash"]
    by_name = index["by_name"]
    out = []
    for digest, pod_path in pod_files:
        name = pod_path.rsplit("/", 1)[-1]
        if digest in by_hash:
            verdict, note = IN_GIT, by_hash[digest][0]
        elif name in by_name:
            verdict, note = DRIFTED, f"{name} exists in repo with different content"
        else:
            verdict, note = POD_ONLY, "no file of this name anywhere in the repo"
        out.append({"path": pod_path, "sha256": digest,
                    "verdict": verdict, "note": note})
    return out


def scan_pod(host: str, timeout: int = 180) -> list[tuple[str, str]]:
    """ssh into `host` and hash candidate source files. Read-only."""
    names = " -o ".join(f"-name '*{s}'" for s in SUFFIXES)
    roots = " ".join(SEARCH_ROOTS)
    # -size -2M keeps us to source, not data blobs; maxdepth 3 covers the
    # stash spots without walking 300GB of episode cache.
    remote = (
        f"find {roots} -maxdepth 3 -type f \\( {names} \\) -size -2M 2>/dev/null "
        f"| grep -vE '__pycache__|site-packages|dist-packages|/\\.git/|/\\.cache/' "
        f"| head -800 | xargs -r sha256sum 2>/dev/null"
    )
    try:
        res = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", host, remote],
            capture_output=True, text=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(f"  !! {host}: unreachable ({type(exc).__name__})", file=sys.stderr)
        return []
    if res.returncode != 0 and not res.stdout.strip():
        print(f"  !! {host}: ssh failed: {res.stderr.strip()[:120]}", file=sys.stderr)
        return []

    files = []
    for line in res.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and len(parts[0]) == 64:
            files.append((parts[0], parts[1]))
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pods", nargs="*", default=DEFAULT_PODS)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument("--show-drifted", action="store_true",
                    help="also list DRIFTED files (default: only counts)")
    args = ap.parse_args(argv)

    print(f"[drift] indexing repo at {args.repo} ...")
    index = repo_index(args.repo)
    print(f"[drift] {len(index['by_hash'])} distinct source blobs, "
          f"{len(index['by_name'])} distinct filenames")

    report, pod_only_total = {}, 0
    for host in args.pods:
        print(f"\n=== {host} ===")
        found = classify(scan_pod(host), index)
        report[host] = found
        if not found:
            print("  (no candidate files / unreachable)")
            continue
        counts = defaultdict(int)
        for f in found:
            counts[f["verdict"]] += 1
        print(f"  {counts[IN_GIT]} in git · {counts[DRIFTED]} drifted · "
              f"{counts[POD_ONLY]} POD-ONLY")
        for f in found:
            if f["verdict"] == POD_ONLY:
                pod_only_total += 1
                print(f"    POD_ONLY  {f['path']}")
            elif f["verdict"] == DRIFTED and args.show_drifted:
                print(f"    DRIFTED   {f['path']}")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[drift] wrote {args.json}")

    print(f"\n[drift] TOTAL POD-ONLY FILES: {pod_only_total}")
    if pod_only_total:
        print("[drift] These exist in exactly one place. Rescue them into git.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
