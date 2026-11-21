#!/usr/bin/env python3
"""Run a command, record its PEAK RSS and its exit code, exit with its exit code.

    python peak.py <tag> <logs dir> -- <command ...>

Writes ``<logs dir>/PEAK_<tag>`` = ``{"tag", "rc", "peak_rss_gb", "wall_s", "cmd"}``. The peak is
``getrusage(RUSAGE_CHILDREN).ru_maxrss`` (kB on Linux) -- the largest resident set any waited-for descendant reached.
On Thor this is CPU memory only (these steps run with CUDA_VISIBLE_DEVICES=""), where RSS is the right probe.
Used inside a kernel memory cap (``systemd-run --user --scope -p MemoryMax=...``): the cap PREVENTS a runaway, this
records how close a step came to it (2026-10-21 22:50: an uncapped gather reached 122.9 GB and took Thor into a
global OOM)."""
import json
import resource
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    tag, logs = sys.argv[1], Path(sys.argv[2])
    if sys.argv[3] != "--":
        raise SystemExit("usage: peak.py <tag> <logs dir> -- <command ...>")
    cmd = sys.argv[4:]
    t0 = time.time()
    rc = subprocess.call(cmd)
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    rec = {"tag": tag, "rc": rc, "peak_rss_gb": round(ru.ru_maxrss / 1024 ** 2, 3),
           "wall_s": round(time.time() - t0, 1), "cmd": cmd}
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"PEAK_{tag}").write_text(json.dumps(rec) + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
