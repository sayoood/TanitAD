#!/usr/bin/env python3
"""A13 — independent re-derivation of the drift set in Thor's /home/nvidia/TanitAD checkout.

Second probe, deliberately NOT pod_currency_audit.py: a different enumeration, a different
framing, a different history walk.  CLAUDE.md: repeated samples through ONE broken channel
are one sample; a second *probe* means a different mechanism.

Direction evidence, in order of strength:
  1. pod bytes == blob at some commit in this path's history  -> THOR_MATCHES_OLD_COMMIT / BEHIND
  2. pod bytes == current worktree bytes (uncommitted)        -> THOR_MATCHES_DIRTY_WORKTREE
  3. pod bytes match nothing we ever committed                -> DIVERGED (thor-local edit)
mtime is corroboration only, never the settling fact.

Read-only on both sides.  Never writes to Thor.  Never runs git on Thor.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time

# Disjoint-token protocol: the markers EMITTED share no substring with anything greped
# client-side, and the payload is framed by its own md5, so a PTY that echoes this script
# back cannot forge a match.
POD_SCAN = r"""
set -u
cd /home/nvidia/TanitAD 2>/dev/null || { echo "QQFATALQQ"; exit 3; }
out=$(mktemp)
find . -type f \
  -not -path "./.git/*" \
  -not -path "*/__pycache__/*" \
  -not -path "./_pod_backup/*" \
  -not -path "*/.pytest_cache/*" \
  -not -path "*/.mypy_cache/*" \
  -not -path "*/node_modules/*" \
  -size -2049k \
  \( -name "*.py" -o -name "*.sh" -o -name "*.md" -o -name "*.yaml" -o -name "*.yml" \
     -o -name "*.toml" -o -name "*.cfg" -o -name "*.ini" -o -name "*.patch" \
     -o -name "*.diff" -o -name "*.sql" -o -name "*.bash" -o -name "*.json" \
     -o -name "*.jsonl" -o -name "*.txt" -o -name "*.csv" -o -name "*.tsv" \
     -o -name "*.html" -o -name "*.svg" -o -name "*.gitignore" \) \
  -printf '%P\n' | LC_ALL=C sort | while IFS= read -r f; do
    r=$(md5sum "$f" 2>/dev/null | cut -c1-32)
    n=$(tr -d '\r' < "$f" 2>/dev/null | md5sum | cut -c1-32)
    s=$(stat -c '%s' "$f" 2>/dev/null)
    m=$(stat -c '%Y' "$f" 2>/dev/null)
    printf '%s\t%s\t%s\t%s\t%s\n' "${r:-NA}" "${n:-NA}" "${s:-0}" "${m:-0}" "$f" >> "$out"
done
L=$(wc -l < "$out")
B=$(gzip -9c "$out" | base64 -w0)
echo "QQLINES${L}QQ"
echo "QQB64LEN${#B}QQ"
echo "QQB64MD5$(printf '%s' "$B" | md5sum | cut -c1-32)QQ"
echo "QQPAYLOADSTARTQQ"
printf '%s\n' "$B" | fold -w 200
echo "QQPAYLOADENDQQ"
rm -f "$out"
"""


def md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def lf(b: bytes) -> bytes:
    return b.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def git(args, repo, binary=False, stdin=None, timeout=1800):
    p = subprocess.run(["git", "-C", repo] + args, input=stdin,
                       capture_output=True, timeout=timeout)
    if p.returncode != 0 and stdin is None:
        raise RuntimeError("git %s -> rc=%d %s" % (" ".join(args[:3]), p.returncode,
                                                   p.stderr.decode("utf-8", "replace")[-300:]))
    return p.stdout if binary else p.stdout.decode("utf-8", "replace")


def pod_scan(host: str, timeout: int):
    env = dict(os.environ)
    env["MSYS_NO_PATHCONV"] = "1"
    env["MSYS2_ARG_CONV_EXCL"] = "*"
    # ssh -n ALWAYS (stdin closed).  The script is carried in argv as base64 so no stdin is
    # needed at all -- which also means nothing can eat the caller's stdin, and MSYS cannot
    # path-mangle a base64 blob the way it mangled `SCAN_ROOT=/home/...`.
    b64 = base64.b64encode(POD_SCAN.encode()).decode()
    remote = "echo %s | base64 -d | bash" % b64
    cmd = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25", host, remote]
    p = subprocess.run(cmd, capture_output=True, timeout=timeout, env=env)
    out = p.stdout.decode("utf-8", "replace")
    if p.returncode != 0 or "QQPAYLOADSTARTQQ" not in out:
        raise RuntimeError("pod scan failed rc=%d out=%s err=%s"
                           % (p.returncode, out[-400:],
                              p.stderr.decode("utf-8", "replace")[-400:]))

    def marker(name):
        tag = "QQ" + name
        for line in out.splitlines():
            s = line.strip()
            if s.startswith(tag) and s.endswith("QQ"):
                return s[len(tag):-2]
        raise RuntimeError("marker %s absent -- truncated pull" % name)

    n_lines, b64md5, b64len = int(marker("LINES")), marker("B64MD5"), int(marker("B64LEN"))
    body, grab = [], False
    for line in out.splitlines():
        s = line.strip()
        if s == "QQPAYLOADSTARTQQ":
            grab = True
            continue
        if s == "QQPAYLOADENDQQ":
            grab = False
            continue
        if grab:
            body.append(s)
    b64 = "".join(body)
    if len(b64) != b64len:
        raise RuntimeError("payload %d chars != expected %d -- TRUNCATED" % (len(b64), b64len))
    if md5(b64.encode()) != b64md5:
        raise RuntimeError("payload md5 mismatch -- CORRUPT PULL")
    tsv = gzip.decompress(base64.b64decode(b64)).decode("utf-8", "replace")
    rows = [ln for ln in tsv.split("\n") if ln]
    if len(rows) != n_lines:
        raise RuntimeError("%d rows != expected %d" % (len(rows), n_lines))
    res = {}
    for ln in rows:
        parts = ln.split("\t")
        if len(parts) < 5:
            continue
        res["\t".join(parts[4:]).replace("\\", "/")] = {
            "md5": parts[0], "md5_lf": parts[1],
            "size": int(parts[2]) if parts[2].isdigit() else -1,
            "mtime": int(parts[3]) if parts[3].isdigit() else -1,
        }
    return res


CHUNK = 200


def batch_check(repo, specs):
    """POSITIVE presence assertion per path.  {spec: sha or None}.

    ⛔ MEASURED on this G: mount 2026-09-07: ``git cat-file --batch-check`` returned ONE line
    for a TWO-line query and exited 127, then returned all four for a four-line query.  A
    SHORT result is indistinguishable from a real answer, so every chunk is length-checked,
    retried, and finally falls back to per-path ``rev-parse`` -- never silently accepted.
    """
    out, fallbacks = {}, 0
    for i in range(0, len(specs), CHUNK):
        chunk = specs[i:i + CHUNK]
        lines = None
        for _ in range(5):
            p = subprocess.run(["git", "-C", repo, "cat-file", "--batch-check"],
                               input="".join(s + "\n" for s in chunk).encode(),
                               capture_output=True, timeout=900)
            got = p.stdout.decode("utf-8", "replace").splitlines()
            if p.returncode == 0 and len(got) == len(chunk):
                lines = got
                break
        if lines is None:                                  # positive per-path fallback
            fallbacks += len(chunk)
            for s in chunk:
                r = subprocess.run(["git", "-C", repo, "rev-parse", s],
                                   capture_output=True, timeout=300)
                sha = r.stdout.decode().strip()
                out[s] = sha if (r.returncode == 0 and len(sha) == 40) else None
            continue
        for spec, line in zip(chunk, lines):
            parts = line.split()
            out[spec] = parts[0] if len(parts) >= 2 and parts[1] == "blob" else None
    if fallbacks:
        sys.stderr.write("  [batch-check] %d specs needed the per-path fallback\n" % fallbacks)
    return out


def blob_md5_lf(repo, shas):
    """{sha: md5_lf}.  Chunked + length-checked for the same reason as batch_check."""
    if not shas:
        return {}
    uniq = sorted(set(shas))
    out = {}
    for i in range(0, len(uniq), CHUNK):
        chunk = uniq[i:i + CHUNK]
        got = None
        for _ in range(5):
            p = subprocess.run(["git", "-C", repo, "cat-file", "--batch"],
                               input="".join(s + "\n" for s in chunk).encode(),
                               capture_output=True, timeout=900)
            if p.returncode != 0:
                continue
            buf, off, parsed = p.stdout, 0, {}
            ok = True
            for sha in chunk:
                nl = buf.find(b"\n", off)
                if nl < 0:
                    ok = False
                    break
                hdr = buf[off:nl].split()
                if len(hdr) < 3 or not hdr[2].isdigit():
                    ok = False
                    break
                size = int(hdr[2])
                body = buf[nl + 1:nl + 1 + size]
                if len(body) != size:
                    ok = False
                    break
                parsed[sha] = md5(lf(body))
                off = nl + 1 + size + 1
            if ok and len(parsed) == len(chunk):
                got = parsed
                break
        if got is None:                                    # positive per-sha fallback
            got = {}
            for sha in chunk:
                r = subprocess.run(["git", "-C", repo, "cat-file", "blob", sha],
                                   capture_output=True, timeout=300)
                if r.returncode == 0:
                    got[sha] = md5(lf(r.stdout))
        out.update(got)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="tanitad-thor-wifi")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--pod-cache", default=None,
                    help="reuse a saved pod scan instead of re-hitting the box")
    a = ap.parse_args()

    t0 = time.time()
    if a.pod_cache and os.path.exists(a.pod_cache):
        pod = json.load(open(a.pod_cache, encoding="utf-8"))
        sys.stderr.write("pod scan: %d files from cache %s\n" % (len(pod), a.pod_cache))
    else:
        pod = pod_scan(a.host, a.timeout)
        sys.stderr.write("pod scan: %d files in %.1fs\n" % (len(pod), time.time() - t0))
        if a.pod_cache:
            json.dump(pod, open(a.pod_cache, "w", encoding="utf-8"))

    ref_sha = git(["rev-parse", a.ref], a.repo).strip()
    paths = sorted(pod)
    present = batch_check(a.repo, ["%s:%s" % (a.ref, p) for p in paths])
    ref_blob = {p: present.get("%s:%s" % (a.ref, p)) for p in paths}
    md5s = blob_md5_lf(a.repo, [v for v in ref_blob.values() if v])
    sys.stderr.write("ref blobs resolved: %d of %d present\n"
                     % (sum(1 for v in ref_blob.values() if v), len(paths)))

    rows, mismatched = [], []
    for p in paths:
        rec = {"path": p, "pod_md5_lf": pod[p]["md5_lf"], "pod_size": pod[p]["size"],
               "pod_mtime": pod[p]["mtime"]}
        sha = ref_blob[p]
        if not sha:
            rec["state"] = "THOR_ONLY"
            rows.append(rec)
            continue
        rec["ref_blob"] = sha
        rec["ref_md5_lf"] = md5s.get(sha, "?")
        if rec["ref_md5_lf"] == rec["pod_md5_lf"]:
            rec["state"] = "IDENTICAL"
            if pod[p]["md5"] != rec["ref_md5_lf"]:
                rec["eol_note"] = "matches after CRLF normalisation"
        else:
            rec["state"] = "MISMATCH"
            mismatched.append(p)
        rows.append(rec)
    sys.stderr.write("MISMATCH: %d\n" % len(mismatched))

    # ---- direction, for the mismatched only -------------------------------------------
    by_path = {r["path"]: r for r in rows}
    for i, p in enumerate(mismatched, 1):
        rec = by_path[p]
        # (a) worktree
        wt = os.path.join(a.repo, p)
        try:
            with open(wt, "rb") as fh:
                rec["worktree_md5_lf"] = md5(lf(fh.read()))
        except Exception as exc:
            rec["worktree_md5_lf"] = None
            rec["worktree_error"] = str(exc)[:200]
        # (b) history walk -- every commit that touched this path, on ANY ref
        try:
            log = git(["log", "--all", "--format=%H", "--", p], a.repo, timeout=900)
            commits = [c for c in log.split() if len(c) == 40]
        except Exception as exc:
            rec["history"] = {"error": str(exc)[:200]}
            commits = []
        rec["history_n_commits"] = len(commits)
        hit = None
        if commits:
            specs = ["%s:%s" % (c, p) for c in commits]
            chk = batch_check(a.repo, specs)
            cshas = [chk[s] for s in specs]
            cm = blob_md5_lf(a.repo, [s for s in cshas if s])
            for depth, (c, s) in enumerate(zip(commits, cshas)):
                if s and cm.get(s) == rec["pod_md5_lf"]:
                    hit = {"commit": c, "depth_from_tip_of_path_history": depth}
                    if depth > 0:
                        sup = commits[depth - 1]
                        hit["superseded_by"] = sup
                        hit["superseded_by_subject"] = git(
                            ["log", "-1", "--format=%ci|%s", sup], a.repo).strip()[:220]
                    break
        if hit:
            hit["subject"] = git(["log", "-1", "--format=%ci|%s", hit["commit"]],
                                 a.repo).strip()[:220]
            # Is that commit an ANCESTOR of the ref?  If yes, Thor holds a version the repo
            # has since superseded => the REPO is authoritative.  If no, the bytes are a
            # committed version that is not on this line -- report it, do not guess.
            anc = subprocess.run(["git", "-C", a.repo, "merge-base", "--is-ancestor",
                                  hit["commit"], a.ref], capture_output=True)
            hit["is_ancestor_of_ref"] = (anc.returncode == 0)
            # Which commit superseded it?  The newest commit on this path that is an ancestor.
            rec["history_hit"] = hit
            rec["state"] = "THOR_MATCHES_OLD_COMMIT"
        elif rec.get("worktree_md5_lf") == rec["pod_md5_lf"]:
            rec["state"] = "THOR_MATCHES_DIRTY_WORKTREE"
        elif commits:
            rec["state"] = "DIVERGED_THOR_LOCAL_EDIT"
        else:
            rec["state"] = "INCONCLUSIVE_HISTORY_PROBE_FAILED"
        sys.stderr.write("  [%d/%d] %s -> %s\n" % (i, len(mismatched), p, rec["state"]))

    result = {"host": a.host, "pod_root": "/home/nvidia/TanitAD", "ref": a.ref,
              "ref_sha": ref_sha, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                 time.gmtime()),
              "inclusion_rule": {
                  "suffixes": [".py", ".sh", ".md", ".yaml", ".yml", ".toml", ".cfg", ".ini",
                               ".patch", ".diff", ".sql", ".bash", ".json", ".jsonl", ".txt",
                               ".csv", ".tsv", ".html", ".svg", ".gitignore"],
                  "max_bytes": 2049 * 1024,
                  "excluded": [".git/", "__pycache__/", "_pod_backup/", ".pytest_cache/",
                               ".mypy_cache/", "node_modules/"],
                  "note": "A CENSUS IS A CLAIM ABOUT ITS FILTER until the filter is stated."},
              "n_scanned": len(pod), "rows": rows}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
