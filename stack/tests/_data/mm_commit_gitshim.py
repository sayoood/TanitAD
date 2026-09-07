"""A git that fails ONE subcommand the way the G: mount fails it.

MM_SHIM_FAIL_ADD=1   -> `git add ...` exits 3221225478 (0xC0000006,
                        STATUS_IN_PAGE_ERROR) with EMPTY STDERR, which is
                        exactly how it presented on 2026-09-07: an exit code
                        with nothing to string-match on.
MM_SHIM_EMPTY_DIFF=1 -> `git diff --cached --name-only <ref>` prints NOTHING and
                        exits 0, which is how the mount answered for a commit
                        that really changed 146 files.
Everything else is passed to the real git untouched.
"""
import os
import subprocess
import sys

args = sys.argv[1:]
sub = next((a for a in args if not a.startswith("-")), "")

if sub == "add" and os.environ.get("MM_SHIM_FAIL_ADD"):
    sys.exit(3221225478)
if sub == "diff" and "--cached" in args and os.environ.get("MM_SHIM_EMPTY_DIFF"):
    sys.exit(0)

sys.exit(subprocess.run(["git", *args]).returncode)
