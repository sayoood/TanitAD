"""scrub_clip_ids.py -- enforce the clip-id rule on a directory tree: no banked file may carry a full clip id or an 8+ hex prefix of one.

  python scrub_clip_ids.py --root <dir> [--root <dir2> ...]  [--known-ids <txt>]  [--check-only]  [--json <report.json>]

How it works (stdlib + numpy only; CPU):
  1. LEARN the ids: every full UUID found in any text file under the roots (manifest.json carries `clip_id`), plus the lines of --known-ids if given
     (the pool list; it is READ, never written out).
  2. REWRITE (unless --check-only) each text file (.json .log .out .err .txt .md .sh .py .jsonl): every hex-dash token that is a PREFIX (>= 8 chars) of a
     learned id becomes `clip-sha12-<sha256(id)[:12]>` -- the same key TRAIN139_WINDOWS.json uses -- so a reader can still join by sha12.
  3. CHECK (always, AFTER the rewrite, on the bytes re-read from disk): (a) no UUID-shaped token and no >= 8-char prefix of a learned id remains in any text file;
     (b) no string/object array in any .npz holds one (npz are zip-compressed; they are opened, not grepped).
  Exit 0 and ZZL7CSCRUBOKZZ iff the post-rewrite check is clean AND at least one file was scanned (an empty scan is INCONCLUSIVE, not clean).
  Exit 3 and ZZL7CSCRUBBADZZ otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np

TEXT_EXT = (".json", ".log", ".out", ".err", ".txt", ".md", ".sh", ".py", ".jsonl", ".csv")
UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
HEXDASH = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{0,12})*")


def sha12(x: str) -> str:
    return hashlib.sha256(x.encode()).hexdigest()[:12]


def _files(roots):
    for r in roots:
        for d, _, fs in os.walk(r):
            for f in fs:
                yield os.path.join(d, f)


def _read(p):
    with open(p, "rb") as fh:
        return fh.read().decode("utf-8", "ignore")


def learn(roots, known_file=None):
    ids = set()
    for p in _files(roots):
        if p.lower().endswith(TEXT_EXT):
            ids.update(m.lower() for m in UUID.findall(_read(p)))
    if known_file:
        for ln in open(known_file, encoding="utf-8"):
            ln = ln.strip().lower()
            if UUID.fullmatch(ln):
                ids.add(ln)
    return ids


def _prefix_hit(tok: str, ids):
    t = tok.lower()
    if len(t) < 8:
        return None
    for i in ids:
        if i.startswith(t):
            return i
    return None


def rewrite_text(txt: str, ids) -> tuple[str, int]:
    n = 0

    def sub(m):
        nonlocal n
        i = _prefix_hit(m.group(0).rstrip("-"), ids)
        if i is None:
            return m.group(0)
        n += 1
        return "clip-sha12-" + sha12(i) + ("-" if m.group(0).endswith("-") else "")
    return HEXDASH.sub(sub, txt), n


def check(roots, ids):
    bad, scanned, npz_scanned = [], 0, 0
    for p in _files(roots):
        pl = p.lower()
        if pl.endswith(TEXT_EXT):
            scanned += 1
            t = _read(p)
            if UUID.search(t):
                bad.append((p, "uuid"))
                continue
            for m in HEXDASH.finditer(t):
                if _prefix_hit(m.group(0).rstrip("-"), ids):
                    bad.append((p, "id-prefix"))
                    break
        elif pl.endswith(".npz"):
            npz_scanned += 1
            try:
                with np.load(p, allow_pickle=True) as z:
                    for k in z.files:
                        a = z[k]
                        if a.dtype.kind in "USO":
                            for v in a.ravel().tolist():
                                s = v.decode("utf-8", "ignore") if isinstance(v, bytes) else str(v)
                                if UUID.search(s) or any(_prefix_hit(m.group(0).rstrip("-"), ids) for m in HEXDASH.finditer(s)):
                                    bad.append((p + "::" + k, "npz-string"))
                                    raise StopIteration
            except StopIteration:
                pass
            except Exception as e:                       # an unreadable npz is not clean
                bad.append((p, "unreadable-npz:" + type(e).__name__))
    return bad, scanned, npz_scanned


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--known-ids", default=None)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    ids = learn(a.root, a.known_ids)
    n_rew = n_files = 0
    if not a.check_only:
        for p in _files(a.root):
            if p.lower().endswith(TEXT_EXT):
                t = _read(p)
                t2, n = rewrite_text(t, ids)
                if n:
                    with open(p, "w", encoding="utf-8", newline="") as fh:
                        fh.write(t2)
                    n_rew += n
                    n_files += 1
    bad, scanned, npz_scanned = check(a.root, ids)
    ok = (not bad) and scanned > 0
    rep = {"ids_learned": len(ids), "tokens_rewritten": n_rew, "files_rewritten": n_files, "text_files_scanned": scanned, "npz_scanned": npz_scanned,
           "residual": [list(b) for b in bad[:20]], "n_residual": len(bad), "CLEAN": bool(ok)}
    if a.json:
        json.dump(rep, open(a.json, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "residual"}), "ZZL7CSCRUBOKZZ" if ok else "ZZL7CSCRUBBADZZ", flush=True)
    for b in bad[:10]:
        print("  residual:", b[1], os.path.basename(b[0]), flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
