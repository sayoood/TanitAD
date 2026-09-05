#!/usr/bin/env python3
"""pod_currency_audit.py -- is the BOX running the REPO?  (repo -> box, by CONTENT)

Why this exists
---------------
Two runs were burned by pod code that silently predated the repo:

* refcv3 trained 40,284 steps on a synthetic anchor fallback because its launch script
  never passed ``--anchors`` and the load was SILENT.
* 2026-09-04: ``refc_tactical.py`` / ``refc_v3_train.py`` were STALE ON THE POD -- a width
  assertion committed as ``c37fa68`` was simply not there, which is exactly why the aborted
  refcv4 run did not crash on an 8-wide tactical vocabulary being indexed by a 3-wide
  contract.

``CLAUDE.md`` already records the general form ("a pod's ``stack/`` checkout drifts silently
and a launch from it resurrects fixed bugs"), and ``AGENT_OPERATING_STANDARD.md`` records the
third rung of the ladder:

    presence proves transfer, md5 proves bytes, a successful import proves loading --
    NONE OF THEM PROVES CURRENCY.

This tool proves currency.  It is the converse of ``pod_git_drift.py`` (box -> repo, "what
lives only on a box?") and a sibling of ``launch_closure_audit.py`` (which audits one launch
command's import closure).  Neither of those can answer "is every file on the box the version
that is in git", because a repo file that is missing or stale on the box is invisible to a
box->repo scan.

What it does
------------
1. Runs ONE ssh command that md5s every file under the pod subtree and returns the table
   gzip+base64 framed with its own md5, so a truncated or PTY-mangled pull FAILS LOUDLY
   instead of silently producing a short table.  (The GOTTY PTY drops ~5 of every 14 lines
   on bulk pulls -- MEASURED 2026-08-12.)
2. md5s the corresponding blobs at a git ref (default ``HEAD``) -- the REF, not the worktree,
   because the worktree may carry uncommitted edits, which are reported as a THIRD state.
3. Classifies every path and, for every mismatch, walks that path's git history to answer the
   question md5 alone cannot: is the pod BEHIND (its bytes equal an ancestor commit's blob)
   or DIVERGED (its bytes match nothing we ever committed)?

Direction is never guessed.  A mismatch that matches no known revision is reported as
POD-DIVERGED, not as "ahead".

States
------
==================== ==========================================================
IDENTICAL            pod bytes == ref blob bytes
POD-BEHIND           pod bytes == an ANCESTOR commit's blob for this path
                     (the report names the commit and how many commits back)
POD-AHEAD            pod bytes == a DESCENDANT commit's blob (ref is not the tip)
POD-DIVERGED         pod bytes match no revision of this path -- a pod-local edit
POD-MATCHES-DIRTY    pod bytes == the (uncommitted) WORKTREE file, not the ref blob
                     => the pod is running a change that was never committed
HISTORY-UNKNOWN      the bytes differ AND the history probe could not complete
                     (mount outage, or --no-history).  NOT a verdict -- it is the
                     absence of one, and it FAILS the audit by default so that a
                     probe which could not run is never scored as clean.
REPO-DIRTY           worktree != ref blob (reported alongside, informational)
POD-ONLY             exists on the pod, not at the ref
REPO-ONLY            exists at the ref, not on the pod
UNREADABLE           the mount refused the file (NOT the same as absent -- see
                     CLAUDE.md: "0 hits is a claim about the SEARCH, not the CONTENT")
==================== ==========================================================

Exit codes
----------
0  clean for the selected files
1  drift found in a class named by ``--fail-on``
   (default: ``pod-behind,pod-diverged,history-unknown``)
2  the audit itself could not be trusted (ssh failed, payload md5 mismatch, git unreachable)

``REPO-ONLY`` is NOT in the default ``--fail-on``: a box legitimately carries a subset of the
repo (tests, sibling arms, tooling it never runs).  Add it when auditing a box that is meant
to be a full mirror.  ⚠️ But REPO-ONLY is not automatically benign either -- MEASURED
2026-09-04 on ``tanitad-refcv3``, ``tanitad/data/v72_eval_clip_digests.json`` was REPO-ONLY,
and it is the DATA a safety oracle in ``parity.py`` loads.  Which is also why the default
``--ext .py`` is a trap of its own: **run ``--all-files`` at least once per box**, or a
sidecar whose absence disables a guard stays invisible.

Usage
-----
    python stack/scripts/pod_currency_audit.py --host tanitad-refcv3
    python stack/scripts/pod_currency_audit.py --host tanitad-refcv3 --ext .py \
        --json "TanitAD Research Lab/.../drift_map.json"
    python stack/scripts/pod_currency_audit.py --host thor --pod-root /home/nvidia/TanitAD/stack \
        --fail-on pod-behind

⛔ This tool NEVER writes to the pod and NEVER runs git on the pod.  ``git fetch`` on a pod
HANGS (no credentials) and a failed fetch followed by a checkout RESETS the tree to an ancient
commit, destroying shipped files.  Shipping is a separate, deliberate act.
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
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------------------
# The pod-side scanner.
#
# Disjoint-token protocol: the markers we EMIT (ZZLINES.../ZZB64MD5...) share no substring
# with anything we GREP for client-side, and the payload is framed by its own md5.  A PTY that
# echoes this script back cannot forge a matching md5.  (A polling monitor whose filter
# contains the pattern it searches for matches its own echoed command -- MEASURED three times
# in this programme, most recently 2026-08-12.)
# --------------------------------------------------------------------------------------
POD_SCAN_SH = r"""
set -u
cd "$SCAN_ROOT" 2>/dev/null || { echo "ZZFATALnorootZZ"; exit 3; }
out=$(mktemp)
bad=0
find . -type f -printf '%P\n' | LC_ALL=C sort | while IFS= read -r f; do
  r=$(md5sum "$f" 2>/dev/null | cut -c1-32)
  n=$(tr -d '\r' < "$f" 2>/dev/null | md5sum | cut -c1-32)
  s=$(stat -c %s "$f" 2>/dev/null)
  m=$(stat -c %Y "$f" 2>/dev/null)
  [ -z "$r" ] && r="UNREADABLE"
  printf '%s\t%s\t%s\t%s\t%s\n' "$r" "$n" "$s" "$m" "$f" >> "$out"
done
lines=$(wc -l < "$out")
b64=$(gzip -9c "$out" | base64 -w0)
h=$(printf '%s' "$b64" | md5sum | cut -c1-32)
rm -f "$out"
echo "ZZLINES${lines}ZZ"
echo "ZZB64MD5${h}ZZ"
echo "ZZB64LEN${#b64}ZZ"
echo "ZZPAYLOADSTART"
printf '%s\n' "$b64"
echo "ZZPAYLOADEND"
echo "ZZDONE${bad}ZZ"
"""


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _lf(b: bytes) -> bytes:
    return b.replace(b"\r\n", b"\n")


GIT_RETRIES = 40          # ~5 min of patience; the G: mount goes down for minutes
GIT_BACKOFF_CAP = 8.0


def run_git(args: List[str], repo: str, binary: bool = False,
            check: bool = True, retries: int = None):
    """Run git, retrying hard.

    The Google-Drive mount this repo lives on goes FULLY down -- reads and writes,
    ``Errno 22`` / ``Invalid request code`` -- for MINUTES at a time, and git then reports
    ``not a git repository`` or ``couldn't read .git/packed-refs``.  Those are mount
    failures wearing a git costume.  Four retries over 20 s is not enough: MEASURED
    2026-09-04, a full audit died on a flap that lasted longer than that.  A short retry
    budget turns a transient outage into a fabricated verdict, which is the failure this
    whole tool exists to prevent.
    """
    retries = GIT_RETRIES if retries is None else retries
    last = None
    for attempt in range(retries):
        try:
            p = subprocess.run(["git", "-C", repo] + args, capture_output=True, timeout=600)
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - mount pathology
            last = exc
            time.sleep(min(GIT_BACKOFF_CAP, 1.0 + 0.5 * attempt))
            continue
        if p.returncode == 0 or not check:
            return p.stdout if binary else p.stdout.decode("utf-8", "replace")
        last = RuntimeError(
            "git %s -> rc=%d: %s" % (" ".join(args[:3]), p.returncode,
                                     p.stderr.decode("utf-8", "replace")[:400]))
        time.sleep(min(GIT_BACKOFF_CAP, 1.0 + 0.5 * attempt))
    raise last if isinstance(last, Exception) else RuntimeError("git failed")


def run_git_stdin(args: List[str], repo: str, payload: bytes, timeout: int = 1800) -> bytes:
    """``git <args>`` fed `payload` on stdin, retried like :func:`run_git`.

    The first revision called ``cat-file --batch`` / ``--batch-check`` through a BARE
    ``subprocess.run``, so the patient retry in :func:`run_git` protected only SOME of the
    git surface.  MEASURED 2026-09-04: two audit runs died right here -- ``fatal: not a git
    repository`` -- while ``run_git`` itself was riding out the identical mount outage.
    A retry policy is only as good as its least-protected call site.
    """
    last = ""
    for attempt in range(GIT_RETRIES):
        try:
            p = subprocess.run(["git", "-C", repo] + args, input=payload,
                               capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            last = "timeout"
            time.sleep(min(GIT_BACKOFF_CAP, 1.0 + 0.5 * attempt))
            continue
        if p.returncode == 0:
            return p.stdout
        last = p.stderr.decode("utf-8", "replace")[:300]
        time.sleep(min(GIT_BACKOFF_CAP, 1.0 + 0.5 * attempt))
    raise RuntimeError("git %s (stdin) exhausted %d retries: %s"
                       % (" ".join(args[:3]), GIT_RETRIES, last))


def pod_scan(host: str, root: str, ssh_opts: List[str], timeout: int) -> Dict[str, dict]:
    """Return {relpath: {md5, md5_lf, size, mtime}} for every file under `root` on `host`."""
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25"] + ssh_opts + [
        host, "SCAN_ROOT=%s bash -s" % root]
    p = subprocess.run(cmd, input=POD_SCAN_SH.encode(), capture_output=True, timeout=timeout)
    out = p.stdout.decode("utf-8", "replace")
    if p.returncode != 0 or "ZZPAYLOADSTART" not in out:
        raise RuntimeError("pod scan failed (rc=%d): %s | %s"
                           % (p.returncode, out[-400:], p.stderr.decode("utf-8", "replace")[-400:]))

    def marker(name: str) -> str:
        tag = "ZZ" + name
        for line in out.splitlines():
            line = line.strip()
            if line.startswith(tag) and line.endswith("ZZ"):
                return line[len(tag):-2]
        raise RuntimeError("pod scan: marker %s absent (truncated pull?)" % name)

    exp_lines, exp_md5, exp_len = int(marker("LINES")), marker("B64MD5"), int(marker("B64LEN"))
    body, grab = [], False
    for line in out.splitlines():
        s = line.strip()
        if s == "ZZPAYLOADSTART":
            grab = True
            continue
        if s == "ZZPAYLOADEND":
            grab = False
            continue
        if grab:
            body.append(s)
    b64 = "".join(body)
    if len(b64) != exp_len:
        raise RuntimeError("pod scan: payload %d chars, expected %d -- TRUNCATED PULL"
                           % (len(b64), exp_len))
    if _md5(b64.encode()) != exp_md5:
        raise RuntimeError("pod scan: payload md5 %s != %s -- CORRUPT PULL"
                           % (_md5(b64.encode()), exp_md5))
    tsv = gzip.decompress(base64.b64decode(b64)).decode("utf-8", "replace")
    rows = [ln for ln in tsv.split("\n") if ln]
    if len(rows) != exp_lines:
        raise RuntimeError("pod scan: %d rows, expected %d" % (len(rows), exp_lines))
    res: Dict[str, dict] = {}
    for ln in rows:
        parts = ln.split("\t")
        if len(parts) < 5:
            continue
        r, n, s, m, path = parts[0], parts[1], parts[2], parts[3], "\t".join(parts[4:])
        res[path.replace("\\", "/")] = {
            "md5": r, "md5_lf": n,
            "size": int(s) if s.isdigit() else -1,
            "mtime": int(m) if m.isdigit() else -1,
        }
    return res


def ref_blobs(repo: str, ref: str, subtree: str) -> Dict[str, str]:
    """{relpath-under-subtree: blob-sha} at `ref`.

    ⛔ ``git ls-tree -r`` SILENTLY TRUNCATES on the G: mount, exits 0, and truncates
    CONSISTENTLY -- so repeating it looks like confirmation (MEASURED 2026-09-02, cost a
    wrongly reverted commit).  We therefore take the UNION of three enumerations and then
    make a POSITIVE assertion per path via ``cat-file --batch-check``, which is the only
    admissible evidence of presence on this mount.
    """
    cands = set()
    for _ in range(3):
        for ln in run_git(["ls-tree", "-r", ref, "--name-only", "--", subtree], repo).splitlines():
            if ln.strip():
                cands.add(ln.strip())
    for ln in run_git(["ls-files", "--", subtree], repo).splitlines():
        if ln.strip():
            cands.add(ln.strip())
    if not cands:
        raise RuntimeError("no candidate paths under %s at %s" % (subtree, ref))

    ordered = sorted(cands)
    query = "".join("%s:%s\n" % (ref, p) for p in ordered)
    raw = run_git_stdin(["cat-file", "--batch-check"], repo, query.encode(), timeout=900)
    lines = raw.decode("utf-8", "replace").splitlines()
    out: Dict[str, str] = {}
    for path, line in zip(ordered, lines):
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "blob":
            out[path] = parts[0]
    return out


def blob_md5s(repo: str, shas: List[str]) -> Dict[str, Tuple[str, str]]:
    """{blob-sha: (md5_raw, md5_lf)} via one streaming ``cat-file --batch``."""
    if not shas:
        return {}
    uniq = sorted(set(shas))
    buf = run_git_stdin(["cat-file", "--batch"], repo,
                        ("\n".join(uniq) + "\n").encode(), timeout=1800)
    pos, res = 0, {}
    for _ in uniq:
        nl = buf.find(b"\n", pos)
        if nl < 0:
            break
        header = buf[pos:nl].decode("utf-8", "replace").split()
        pos = nl + 1
        if len(header) < 3:
            continue
        sha, size = header[0], int(header[2])
        content = buf[pos:pos + size]
        pos += size + 1  # trailing newline
        res[sha] = (_md5(content), _md5(_lf(content)))
    return res


def worktree_md5(repo: str, relpath: str) -> Optional[Tuple[str, str]]:
    fp = os.path.join(repo, relpath)
    for attempt in range(4):
        try:
            with open(fp, "rb") as fh:
                data = fh.read()
            return _md5(data), _md5(_lf(data))
        except FileNotFoundError:
            return None
        except OSError:  # Errno 22 -- the mount is flapping, not the file missing
            time.sleep(1.5 * (attempt + 1))
    return ("UNREADABLE", "UNREADABLE")


def history_match(repo: str, relpath: str, want_lf: str, ref: str) -> dict:
    """Find the commit whose blob for `relpath` LF-matches `want_lf`.

    EVERY git call here is ``check=True``.  The first revision of this function used
    ``check=False``, so a G:-mount flap (``Invalid argument`` / ``Invalid request code``)
    returned an EMPTY log and the caller read that as "matches no revision" -- i.e. it
    printed POD-DIVERGED for files that may simply be POD-BEHIND.  MEASURED 2026-09-04: all
    14 mismatches came back ``hist=None`` while the mount was flapping, which is a fabricated
    verdict, not a measurement.  This is the CLAUDE.md rule in tool form: **an empty result is
    a claim about the SEARCH, not about the CONTENT**, unless the search is asserted to have
    succeeded.  A probe that cannot complete now RAISES, and the row reads HISTORY-UNKNOWN
    instead of silently reading as a verdict.

    There is deliberately NO ``-n`` cap either: a cap makes "no match" ambiguous between
    "diverged" and "older than the window" -- the same ambiguity in a different costume.
    """
    # ⭐ Walk the REF'S OWN history first, and only fall back to ``--all`` on a miss.
    # ``git log --all -- <path>`` walks every ref in the repository and, on the G: mount, a
    # single such call took MINUTES and stalled the whole audit (MEASURED 2026-09-04: the
    # process sat at 0.8 s CPU for 15 minutes, all of it inside one of these).  The question
    # that actually matters -- "is the box BEHIND?" -- is by definition about ANCESTORS of the
    # ref, so ``git log <ref>`` answers it, and it is the overwhelmingly common case.  ``--all``
    # is still tried afterwards, so a pod that is AHEAD or on a side branch is not misreported
    # as DIVERGED; it is just no longer paid for on every lookup.
    scopes = ([ref] if ref else []) + ["--all"]
    tried, commits, scope_used = [], [], None
    for scope in scopes:
        log = run_git(["log", scope, "--format=%H", "--", relpath], repo)
        found = [c for c in log.split() if len(c) == 40]
        tried.append(scope)
        if found:
            commits, scope_used = found, scope
            hit = _scan_commits(repo, relpath, want_lf, ref, commits)
            if hit is not None:
                return {"searched": len(commits), "scope": scope, "hit": hit}
    if not commits:
        return {"searched": 0, "scope": ",".join(tried), "hit": None}
    return {"searched": len(commits), "scope": scope_used, "hit": None}


def _scan_commits(repo: str, relpath: str, want_lf: str, ref: str, commits) -> Optional[dict]:
    """Return the hit dict for the first commit whose blob LF-matches, else ``None``."""
    # Resolve EVERY <commit>:<path> in ONE batch-check, not one rev-parse per commit.
    # A path with a long history (train_v6_staged.py has hundreds of commits) otherwise
    # costs 2 git processes per commit ACROSS A NETWORK MOUNT, and the audit stops being
    # something anyone will actually run before a launch -- which is the whole point of it.
    query = "".join("%s:%s\n" % (c, relpath) for c in commits)
    lines = run_git_stdin(["cat-file", "--batch-check"], repo,
                          query.encode(), timeout=900).decode("utf-8", "replace").splitlines()
    pairs = []
    for c, line in zip(commits, lines):
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "blob":
            pairs.append((c, parts[0]))
    md5s = blob_md5s(repo, [b for _, b in pairs])
    for commit, sha in pairs:
        got = md5s.get(sha)
        if got and got[1] == want_lf:
            # rc==1 means "not an ancestor" (a real answer); anything else is the mount.
            for _try in range(GIT_RETRIES):
                anc = subprocess.run(["git", "-C", repo, "merge-base", "--is-ancestor",
                                      commit, ref], capture_output=True)
                if anc.returncode in (0, 1):
                    break
                time.sleep(min(GIT_BACKOFF_CAP, 1.0 + 0.5 * _try))
            else:
                raise RuntimeError("merge-base --is-ancestor never returned a verdict")
            n_back = run_git(["rev-list", "--count", "%s..%s" % (commit, ref)], repo).strip()
            subj = run_git(["log", "-1", "--format=%h %ad %s", "--date=short", commit],
                           repo).strip()
            return {"commit": commit,
                    "direction": "BEHIND" if anc.returncode == 0 else "AHEAD_OR_SIDE",
                    "commits_since": int(n_back) if n_back.isdigit() else None,
                    "subject": subj}
    return None


def classify(repo: str, ref: str, subtree: str, pod: Dict[str, dict],
             exts: Optional[List[str]], skip_globs: List[str],
             deep_history: bool = True) -> dict:
    blobs = ref_blobs(repo, ref, subtree)
    prefix = subtree.rstrip("/") + "/"
    # pod paths are relative to the pod subtree root; repo paths carry the subtree prefix
    pod_rel = {p: v for p, v in pod.items()}
    repo_rel = {p[len(prefix):]: sha for p, sha in blobs.items() if p.startswith(prefix)}

    def selected(rel: str) -> bool:
        if any(seg in rel.split("/") for seg in skip_globs):
            return False
        return not exts or any(rel.endswith(e) for e in exts)

    all_rel = sorted({r for r in pod_rel if selected(r)} | {r for r in repo_rel if selected(r)})
    md5map = blob_md5s(repo, list(repo_rel.values()))

    rows = []
    for rel in all_rel:
        on_pod, in_ref = rel in pod_rel, rel in repo_rel
        rec = {"path": rel, "repo_path": prefix + rel}
        if on_pod and not in_ref:
            rec["state"] = "POD-ONLY"
            rec["pod_md5"] = pod_rel[rel]["md5"]
            rows.append(rec)
            continue
        if in_ref and not on_pod:
            rec["state"] = "REPO-ONLY"
            rows.append(rec)
            continue
        pm, pm_lf = pod_rel[rel]["md5"], pod_rel[rel]["md5_lf"]
        bm, bm_lf = md5map.get(repo_rel[rel], ("?", "?"))
        wt = worktree_md5(repo, prefix + rel)
        rec.update({"pod_md5": pm, "pod_md5_lf": pm_lf, "ref_md5": bm, "ref_md5_lf": bm_lf,
                    "pod_size": pod_rel[rel]["size"], "pod_mtime": pod_rel[rel]["mtime"]})
        if wt:
            rec["worktree_md5_lf"] = wt[1]
            rec["repo_dirty"] = (wt[1] != bm_lf)
        if pm == bm:
            rec["state"] = "IDENTICAL"
        elif pm_lf == bm_lf:
            rec["state"] = "IDENTICAL"          # line-ending only; not a currency defect
            rec["eol_only"] = True
        elif wt and pm_lf == wt[1]:
            rec["state"] = "POD-MATCHES-DIRTY"  # pod runs an UNCOMMITTED repo edit
        else:
            if not deep_history:
                rec["state"] = "HISTORY-UNKNOWN"
                rec["history"] = {"skipped": True}
            else:
                try:
                    res = history_match(repo, prefix + rel, pm_lf, ref)
                except Exception as exc:
                    # The probe FAILED. That is not evidence of divergence.
                    rec["state"] = "HISTORY-UNKNOWN"
                    rec["history"] = {"error": str(exc)[:300]}
                else:
                    hit = res["hit"]
                    if hit and hit["direction"] == "BEHIND":
                        rec["state"] = "POD-BEHIND"
                    elif hit:
                        rec["state"] = "POD-AHEAD"
                    else:
                        rec["state"] = "POD-DIVERGED"
                    rec["history"] = res
        rows.append(rec)
    return {"ref": ref, "ref_sha": run_git(["rev-parse", ref], repo).strip(),
            "subtree": subtree, "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", required=True, help="ssh alias of the box, e.g. tanitad-refcv3")
    ap.add_argument("--pod-root", default="/workspace/TanitAD/stack")
    ap.add_argument("--repo", default=None, help="repo root (default: git toplevel of cwd)")
    ap.add_argument("--subtree", default="stack", help="repo subtree matching --pod-root")
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--ext", action="append", default=None,
                    help="restrict to these suffixes (repeatable). Default: .py")
    ap.add_argument("--all-files", action="store_true", help="every file, not just --ext")
    ap.add_argument("--skip-dir", action="append",
                    default=["__pycache__", ".pytest_cache", ".git", "results", ".ipynb_checkpoints"])
    ap.add_argument("--json", default=None, help="write the raw drift map here")
    ap.add_argument("--fail-on", default="pod-behind,pod-diverged,history-unknown",
                    help="comma list of states (lowercased) that make the exit code 1. "
                         "HISTORY-UNKNOWN is fail-by-default on purpose: a probe that could "
                         "not run must never be scored as a clean result.")
    ap.add_argument("--no-history", action="store_true",
                    help="skip the git-history walk (faster; every mismatch reads DIVERGED)")
    ap.add_argument("--ssh-opt", action="append", default=[])
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--quiet-identical", action="store_true", default=True)
    a = ap.parse_args(argv)

    repo = a.repo or run_git(["rev-parse", "--show-toplevel"], os.getcwd()).strip()
    exts = None if a.all_files else (a.ext or [".py"])

    try:
        pod = pod_scan(a.host, a.pod_root, ["-o=%s" % o for o in a.ssh_opt], a.timeout)
    except Exception as exc:
        print("AUDIT UNTRUSTWORTHY: %s" % exc, file=sys.stderr)
        return 2

    result = classify(repo, a.ref, a.subtree, pod, exts, a.skip_dir,
                      deep_history=not a.no_history)
    result["host"], result["pod_root"] = a.host, a.pod_root

    counts: Dict[str, int] = {}
    for r in result["rows"]:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    result["counts"] = counts

    print("== pod currency audit ==")
    print("host=%s pod_root=%s" % (a.host, a.pod_root))
    print("repo=%s ref=%s (%s) subtree=%s" % (repo, a.ref, result["ref_sha"][:10], a.subtree))
    print("selected=%d files  (%s)" % (len(result["rows"]), ", ".join(exts) if exts else "ALL"))
    for st in sorted(counts):
        print("  %-20s %4d" % (st, counts[st]))
    print()
    for r in result["rows"]:
        if r["state"] == "IDENTICAL" and a.quiet_identical:
            continue
        extra = ""
        h = (r.get("history") or {}).get("hit")
        if h:
            extra = "  <- %s (%s commits back) %s" % (h["commit"][:10], h["commits_since"],
                                                      h["subject"][:70])
        elif (r.get("history") or {}).get("error"):
            extra = "  [probe failed: %s]" % r["history"]["error"][:80]
        if r.get("repo_dirty"):
            extra += "  [REPO-DIRTY too]"
        print("%-18s %s%s" % (r["state"], r["path"], extra))

    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, sort_keys=True)
        print("\nraw drift map -> %s" % a.json)

    bad = {s.strip().upper() for s in a.fail_on.split(",") if s.strip()}
    hits = sum(v for k, v in counts.items() if k in bad)
    if hits:
        print("\nFAIL: %d file(s) in %s -- the box is NOT running the repo." % (hits, sorted(bad)))
        return 1
    print("\nOK: box matches %s for the selected files." % a.ref)
    return 0


if __name__ == "__main__":
    sys.exit(main())
