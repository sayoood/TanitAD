"""Count HF-token-shaped strings per file (the VALUES are never printed). With --redact, replace each with hf_REDACTED
in place. Unreadable files are COUNTED and retried, never silently skipped (a scan that could not read a file is not
a clean scan).  usage: python token_scan.py [--redact] <root> [<root> ...]"""
import re
import sys
import time
from pathlib import Path

PAT = re.compile(rb"hf_[A-Za-z0-9]{20,}")
SUFFIXES = {".log", ".json", ".txt", ".md", ".out", ".py", ".sh", ".jsonl"}


def main():
    args = sys.argv[1:]
    redact = "--redact" in args
    roots = [Path(a) for a in args if a != "--redact"]
    hits, unread = {}, []
    files = []
    for r in roots:
        for p in r.rglob("*"):
            if p.suffix in SUFFIXES and "__pycache__" not in p.parts:
                files.append(p)
    for p in files:
        data = None
        for attempt in range(5):
            try:
                if not p.is_file() or p.stat().st_size > 50_000_000:
                    data = b""
                    break
                data = p.read_bytes()
                break
            except OSError:
                time.sleep(0.5 * (attempt + 1))
        if data is None:
            unread.append(str(p))
            continue
        n = len(PAT.findall(data))
        if n:
            hits[str(p)] = n
            if redact:
                p.write_bytes(PAT.sub(b"hf_REDACTED", data))
    for k, v in sorted(hits.items()):
        print(v, k)
    print(f"scanned {len(files)} files; token-shaped strings in {len(hits)}; unreadable {len(unread)}"
          + (" (REDACTED in place)" if redact and hits else ""))
    for u in unread:
        print("UNREADABLE", u)
    return 1 if unread else 0


if __name__ == "__main__":
    sys.exit(main())
