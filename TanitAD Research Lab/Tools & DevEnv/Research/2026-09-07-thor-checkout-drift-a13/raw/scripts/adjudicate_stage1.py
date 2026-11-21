#!/usr/bin/env python3
"""A13 — adjudicate Thor's /home/nvidia/TanitAD checkout against the repo, by BLOB SHA.

Why blob shas and not md5: the repo's HEAD blobs are LF-clean (MEASURED: `git cat-file blob`
for three sampled paths carries 0 CR bytes) and Thor's filesystem is LF, so a git blob sha is
directly comparable on both sides.  No content transfer, no CRLF normalisation, and the
comparison is exact rather than a digest of a digest.

Direction, in order of strength:
  1. Thor bytes == the blob at THOR'S OWN HEAD (30d6d60, 2026-08-15) and that commit is a
     verified ANCESTOR of the repo's HEAD  =>  Thor holds a superseded committed version.
     This is a POSITIVE identification of a named commit, not an inference from mtimes.
  2. Thor bytes != Thor's HEAD blob  =>  something was shipped/edited onto Thor after the
     checkout; walk the repo's history for that path to find which revision it is.
  3. No revision matches  =>  a Thor-local edit that exists nowhere in git.
"""
import json
import os
import subprocess
import sys

SP = os.path.dirname(os.path.abspath(__file__))
REPO = "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
REPO_HEAD = sys.argv[1] if len(sys.argv) > 1 else "7084b2cf50615bbf3fdae48ef4e82d8be43899d2"

th = json.load(open(os.path.join(SP, "thor_head.json"), encoding="utf-8"))
THOR_HEAD = th["thor_head"]
thor_tree = th["tree"]        # path -> blob sha at Thor's HEAD
thor_wt = th["worktree"]      # path -> blob sha of Thor's working-tree bytes

repo = {}
for ln in open(os.path.join(SP, "head_index.txt"), encoding="utf-8"):
    ln = ln.rstrip("\n")
    if not ln:
        continue
    meta, path = ln.split("\t", 1)
    parts = meta.split()
    if len(parts) >= 2:
        repo[path] = parts[1]

pod = json.load(open(os.path.join(SP, "pod_scan.json"), encoding="utf-8"))

print("repo HEAD  %s  paths %d" % (REPO_HEAD, len(repo)))
print("thor HEAD  %s  tracked %d" % (THOR_HEAD, len(thor_tree)))

rows = []
for p in sorted(thor_wt):
    tw, tt = thor_wt[p], thor_tree.get(p)
    rec = {"path": p, "thor_blob": tw, "thor_head_blob": tt,
           "thor_bytes_equal_thor_head": (tw == tt)}
    if p in pod:
        rec["thor_mtime"] = pod[p]["mtime"]
        rec["thor_size"] = pod[p]["size"]
    rb = repo.get(p)
    if rb is None:
        rec["state"] = "ABSENT_FROM_REPO_HEAD"
        rows.append(rec)
        continue
    rec["repo_blob"] = rb
    if rb == tw:
        rec["state"] = "IDENTICAL"
    else:
        rec["state"] = "DRIFT"
    rows.append(rec)

# repo files NOT on Thor at all (REPO-ONLY) -- the class pod_git_drift cannot see
thor_all = set(thor_wt) | set(pod)
repo_only = sorted(p for p in repo if p not in thor_all)

drift = [r for r in rows if r["state"] == "DRIFT"]
absent = [r for r in rows if r["state"] == "ABSENT_FROM_REPO_HEAD"]
print("IDENTICAL            %d" % sum(1 for r in rows if r["state"] == "IDENTICAL"))
print("DRIFT                %d" % len(drift))
print("ABSENT_FROM_REPO_HEAD %d" % len(absent))
print("REPO_ONLY (not on Thor at all) %d" % len(repo_only))

json.dump({"repo_head": REPO_HEAD, "thor_head": THOR_HEAD,
           "rows": rows, "repo_only": repo_only},
          open(os.path.join(SP, "adjudicate_stage1.json"), "w", encoding="utf-8"), indent=1)
print("wrote adjudicate_stage1.json")
