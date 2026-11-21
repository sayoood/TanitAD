#!/usr/bin/env python3
"""A13 — FINAL adjudication table for Thor's /home/nvidia/TanitAD checkout.

Everything is a comparison of GIT BLOB SHAs, so no file content has to be read from the
G: mount (which truncates: `git log --name-only` over the 1,273-commit range returned 42
commits and exit 127 — MEASURED 2026-09-07, and therefore NOT used here).

⛔ THE CRLF CORRECTION.  A blob sha is over RAW bytes.  14 of the 4,684 files in the Thor
census carry CR (shipped from the Windows dev box), so for those the raw sha differs for
line endings alone.  Thor computed the LF-normalised blob sha for exactly those 14
(`tr -d '\r' | git hash-object --stdin`), and this table uses that as the effective blob.

⭐ CONTROL for "repo HEAD blobs are LF-clean" — n = 1,445, not a 3-file sample: 1,445 Thor
files are byte-identical to their repo HEAD blob, and every Thor file outside the 14 is LF.
A CRLF repo blob cannot be byte-equal to an LF file, so those 1,445 repo blobs are LF.

DIRECTION, strongest first:
  E1  effective Thor blob == the blob at Thor's checkout HEAD 30d6d60 (2026-08-15), a
      VERIFIED ancestor of repo HEAD 7084b2cf (git merge-base --is-ancestor, rc=0; the repo
      is 1,273 commits ahead).  => Thor holds a superseded COMMITTED version.
  E2  effective Thor blob == the blob at a named commit in 30d6d60..HEAD (per-path
      range-limited `git log`, which IS reliable on this mount).
  E3  effective Thor blob is / is not in the repo's object store at all, each probe
      bracketed by a same-breath control blob that must resolve.
"""
import collections
import json
import os
import subprocess
import sys

SP = os.path.dirname(os.path.abspath(__file__))
REPO = "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
REPO_HEAD = "7084b2cf50615bbf3fdae48ef4e82d8be43899d2"
THOR_HEAD = "30d6d601cb0589b1cbf2f6f3f99241da8c548af5"
CTRL = "697955681987e3878611f2edca07a41edf34581f"      # repo HEAD:README.md


def g(args, tries=3, timeout=600):
    for _ in range(tries):
        p = subprocess.run(["git", "-C", REPO] + args, capture_output=True, timeout=timeout)
        if p.returncode == 0:
            return p.stdout.decode("utf-8", "replace")
    return None


def present(sha):
    c1 = subprocess.run(["git", "-C", REPO, "cat-file", "-e", CTRL],
                        capture_output=True).returncode == 0
    r = subprocess.run(["git", "-C", REPO, "cat-file", "-e", sha],
                       capture_output=True).returncode == 0
    c2 = subprocess.run(["git", "-C", REPO, "cat-file", "-e", CTRL],
                        capture_output=True).returncode == 0
    return None if not (c1 and c2) else r


st1 = json.load(open(os.path.join(SP, "adjudicate_stage1.json"), encoding="utf-8"))
unt = json.load(open(os.path.join(SP, "thor_untracked.json"), encoding="utf-8"))
pod = json.load(open(os.path.join(SP, "pod_scan.json"), encoding="utf-8"))
th = json.load(open(os.path.join(SP, "thor_head.json"), encoding="utf-8"))
closure = set(l.strip() for l in open(os.path.join(SP, "closure_files.txt"),
                                      encoding="utf-8") if l.strip())
lfblob = {}
for ln in open(os.path.join(SP, "thor_lf_blobs.txt"), encoding="utf-8"):
    ln = ln.strip()
    if ln.startswith("ZZ") and ln.endswith("ZZ"):
        s, p = ln[2:-2].split("|", 1)
        lfblob[p] = s
repo_idx = {}
for ln in open(os.path.join(SP, "head_index.txt"), encoding="utf-8"):
    ln = ln.rstrip("\n")
    if not ln:
        continue
    m, p = ln.split("\t", 1)
    repo_idx[p] = m.split()[1]

# ---- assemble every path Thor holds that the repo also holds, tracked or not ----------
rows = []
for r in st1["rows"]:
    if r["state"] == "DRIFT":
        rows.append({"path": r["path"], "thor_raw_blob": r["thor_blob"],
                     "thor_head_blob": r["thor_head_blob"], "thor_tracked": True,
                     "repo_blob": r["repo_blob"], "thor_mtime": r.get("thor_mtime"),
                     "thor_size": r.get("thor_size")})
for p in sorted(unt):
    rb = repo_idx.get(p)
    if rb is not None and rb != unt[p]:
        rows.append({"path": p, "thor_raw_blob": unt[p], "thor_head_blob": None,
                     "thor_tracked": False, "repo_blob": rb,
                     "thor_mtime": pod.get(p, {}).get("mtime"),
                     "thor_size": pod.get(p, {}).get("size")})

print("raw-sha drift rows: %d" % len(rows))

final = []
for i, r in enumerate(rows, 1):
    p = r["path"]
    r["in_refav1_closure"] = p in closure
    r["thor_is_crlf"] = p in lfblob
    eff = lfblob.get(p, r["thor_raw_blob"])
    r["effective_thor_blob"] = eff
    if eff == r["repo_blob"]:
        r["verdict"] = "EOL_ONLY"
        r["evidence"] = ("Thor's CRLF file LF-normalises to blob %s == repo HEAD blob. "
                         "NOT a currency defect." % eff[:10])
        final.append(r)
        continue
    if r["thor_head_blob"] and eff == r["thor_head_blob"]:
        r["verdict"] = "REPO_NEWER"
        r["thor_matches_commit"] = THOR_HEAD
        r["evidence"] = ("E1: Thor's bytes ARE the blob at Thor's checkout HEAD %s "
                         "(2026-08-15), a VERIFIED ancestor of repo HEAD %s"
                         % (THOR_HEAD[:10], REPO_HEAD[:10]))
        final.append(r)
        continue
    # shipped/edited after the checkout -> name the commit via a per-path range log
    log = g(["log", "%s..%s" % (THOR_HEAD, REPO_HEAD), "--format=%H", "--", p], timeout=900)
    commits = [c for c in log.split() if len(c) == 40] if log is not None else []
    r["commits_in_range_touching_path"] = len(commits) if log is not None else None
    hit = None
    for depth, c in enumerate(commits):
        s = (g(["rev-parse", "%s:%s" % (c, p)], tries=2, timeout=120) or "").strip()
        if len(s) == 40 and s == eff:
            hit = {"commit": c, "depth_from_newest": depth,
                   "subject": (g(["log", "-1", "--format=%ci|%s", c]) or "").strip()[:180]}
            break
    if hit:
        r["verdict"] = "REPO_NEWER"
        r["thor_matches_commit"] = hit["commit"]
        r["history_hit"] = hit
        r["evidence"] = ("E2: Thor's bytes are the blob committed at %s (%s); %d later "
                         "commit(s) in the range changed this path"
                         % (hit["commit"][:10], hit["subject"][:70], hit["depth_from_newest"]))
    else:
        pres = present(eff)
        r["effective_blob_in_objectstore"] = pres
        if pres is None:
            r["verdict"] = "INCONCLUSIVE_PROBE_FAILED"
            r["evidence"] = "the same-breath control blob did not resolve -- NOT absence"
        elif pres:
            r["verdict"] = "REPO_NEWER_COMMIT_UNIDENTIFIED"
            r["evidence"] = ("E3: Thor's effective blob IS in the repo's object store "
                             "(control passed), so this content is committed; no commit in "
                             "%s..HEAD carries it AT THIS PATH (the per-path range log "
                             "listed %s)" % (THOR_HEAD[:10], r["commits_in_range_touching_path"]))
        else:
            r["verdict"] = "THOR_BYTES_NOT_IN_GIT"
            r["evidence"] = ("E3: Thor's effective blob %s is NOT in the repo's object store "
                             "(control passed) -- this content was never committed"
                             % eff[:10])
    final.append(r)
    sys.stderr.write("  [%d] %s -> %s\n" % (i, p, r["verdict"]))

real = [r for r in final if r["verdict"] != "EOL_ONLY"]
json.dump({"repo_head": REPO_HEAD, "thor_head": THOR_HEAD,
           "thor_head_is_ancestor_of_repo_head": True,
           "commits_repo_ahead": 1273,
           "closure_entry": "stack/scripts/refa_v1_train.py",
           "closure_pythonpath": "/home/nvidia/TanitAD/stack",
           "closure_n": len(closure),
           "n_raw_sha_drift": len(final),
           "n_eol_only": len(final) - len(real),
           "n_real_drift": len(real),
           "rows": final},
          open(os.path.join(SP, "A13_DRIFT_TABLE.json"), "w", encoding="utf-8"), indent=1)

print()
print("== VERDICTS ==")
for k, v in collections.Counter(r["verdict"] for r in final).most_common():
    print("  %4d %s" % (v, k))
inc = [r for r in final if r["in_refav1_closure"]]
print()
print("== rows inside the refav1 import closure: %d ==" % len(inc))
for r in sorted(inc, key=lambda x: (x["verdict"], x["path"])):
    print("  %-48s %-32s tracked=%s crlf=%s"
          % (r["path"], r["verdict"], r["thor_tracked"], r["thor_is_crlf"]))
print("wrote A13_DRIFT_TABLE.json")
