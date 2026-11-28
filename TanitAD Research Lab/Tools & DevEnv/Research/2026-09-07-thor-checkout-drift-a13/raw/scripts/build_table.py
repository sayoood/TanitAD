#!/usr/bin/env python3
"""A13 — build the final drift table from evidence already measured.  NO further git calls.

Why this route: the per-path range-limited `git log` walk resolves the *name* of the
superseding commit, and it was resolving ~1 row per 5-15 minutes under the dev box's
concurrent load (6 pytest runs + 2 eval arms + 2 audits from other agents).  The commit NAME
is a nice-to-have; the DIRECTION does not depend on it, and every input the direction does
depend on is already measured and banked:

  thor_head_blobs.json        Thor's checkout-HEAD blob + working-tree blob, all 5,435 tracked
  thor_untracked_blobs.json   the 35 untracked paths
  head_index.txt              repo HEAD blob per path (local-disk scratch index)
  thor_lf_blobs.txt           LF-normalised blob for the 14 CRLF files  <- the §4 correction
  objstore_probe_controlled.json / lfform_objstore.json / snapshot_lf_probe.txt
                              control-bracketed object-store presence

Rows whose direction was ALSO confirmed by a completed per-path history walk carry
`history_walk_agreed: true`, harvested from that run's own log — never invented.
"""
import collections
import json
import os
import re

SP = os.path.dirname(os.path.abspath(__file__))
REPO_HEAD = "7084b2cf50615bbf3fdae48ef4e82d8be43899d2"
THOR_HEAD = "30d6d601cb0589b1cbf2f6f3f99241da8c548af5"

st1 = json.load(open(os.path.join(SP, "adjudicate_stage1.json"), encoding="utf-8"))
unt = json.load(open(os.path.join(SP, "thor_untracked.json"), encoding="utf-8"))
pod = json.load(open(os.path.join(SP, "pod_scan.json"), encoding="utf-8"))
closure = set(l.strip() for l in open(os.path.join(SP, "closure_files.txt"),
                                      encoding="utf-8") if l.strip())
objstore = json.load(open(os.path.join(SP, "objstore_probe_controlled.json"), encoding="utf-8"))

# ⛔ For a CRLF-shipped file the RAW sha is the wrong operand: probing it asks "were these
# exact CRLF bytes ever committed", which is never true and reads as missing content.  The
# 14 CRLF paths are therefore probed on their LF form.  (My first table got parity.py wrong
# for exactly this reason, and the history-walk cross-check is what caught it.)
lf_objstore = {}
for ln in open(os.path.join(SP, "lf_objstore_all14.txt"), encoding="utf-8"):
    ln = ln.strip()
    if not ln or ln.startswith("#"):
        continue
    sha, verdict, ctrl, path = ln.split("|", 3)
    lf_objstore[path] = {"sha": sha, "present": verdict == "PRESENT", "control": ctrl}

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

# verdicts the (terminated) per-path history walk had already produced, harvested from its log
walked = {}
for ln in open(os.path.join(SP, "final.log"), encoding="utf-8", errors="replace"):
    m = re.match(r"\s*\[\d+\]\s+(.+?)\s+->\s+(\S+)\s*$", ln)
    if m:
        walked[m.group(1)] = m.group(2)

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

for r in rows:
    p = r["path"]
    r["in_refav1_closure"] = p in closure
    r["thor_is_crlf"] = p in lfblob
    eff = lfblob.get(p, r["thor_raw_blob"])
    r["effective_thor_blob"] = eff

    if eff == r["repo_blob"]:
        r["verdict"] = "EOL_ONLY"
        r["evidence"] = ("Thor's CRLF file LF-normalises to blob %s, which IS the repo HEAD "
                         "blob. Not a currency defect." % eff[:12])
    elif r["thor_head_blob"] and eff == r["thor_head_blob"]:
        r["verdict"] = "REPO_NEWER"
        r["thor_matches_commit"] = THOR_HEAD
        r["evidence"] = ("E1: Thor's bytes ARE the blob at Thor's checkout HEAD %s "
                         "(2026-08-15 23:06:46 +0200), which git merge-base --is-ancestor "
                         "confirms is an ancestor of repo HEAD %s (1,273 commits back). Thor "
                         "holds a superseded COMMITTED version."
                         % (THOR_HEAD[:12], REPO_HEAD[:12]))
    else:
        # shipped or edited onto Thor after the checkout -> object-store presence decides
        pres = None
        if p in lf_objstore:
            e = lf_objstore[p]
            pres = e["present"] if e["control"] == "OK/OK" else None
            src = "lf_objstore_all14.txt (LF form probed, control %s)" % e["control"]
        elif p in objstore:
            v = objstore[p]
            pres = None if v.startswith("INCONCLUSIVE") else (v == "IN_OBJECTSTORE")
            src = "objstore_probe_controlled.json (control-bracketed)"
        else:
            src = None
        r["effective_blob_in_objectstore"] = pres
        if pres is True:
            r["verdict"] = "REPO_NEWER_COMMIT_UNIDENTIFIED"
            r["evidence"] = ("E3: Thor's effective blob %s IS in the repo's object store "
                             "[%s], so this exact content is committed; the repo's HEAD "
                             "carries a different blob at this path, so the repo has moved on. "
                             "The superseding commit was not enumerated." % (eff[:12], src))
        elif pres is False:
            r["verdict"] = "THOR_BYTES_NOT_IN_GIT"
            r["evidence"] = ("E3: Thor's effective blob %s is NOT in the repo's object store "
                             "[%s, control passed]." % (eff[:12], src))
        else:
            r["verdict"] = "INCONCLUSIVE_PROBE_FAILED"
            r["evidence"] = ("no admissible object-store probe for this path%s"
                             % ("" if src is None else " [%s]" % src))
    # The per-path range-limited history walk is STRICTLY STRONGER where it completed: it
    # located the exact superseding commit rather than only proving the blob is committed.
    # Adopt it, and keep both verdicts on the row so the promotion is visible.
    if p in walked:
        r["history_walk_verdict"] = walked[p]
        if walked[p] == "REPO_NEWER" and r["verdict"] == "REPO_NEWER_COMMIT_UNIDENTIFIED":
            r["verdict_before_history_walk"] = r["verdict"]
            r["verdict"] = "REPO_NEWER"
            r["evidence"] = ("E2: a per-path range-limited `git log %s..HEAD -- <path>` walk "
                             "located the commit in the range whose blob IS Thor's bytes "
                             "(final.log; the run was terminated before serialising the id). "
                             "Superseded by a later commit in the same range. Prior, weaker "
                             "verdict from the object-store probe alone: %s"
                             % (THOR_HEAD[:12], r["verdict_before_history_walk"]))
        r["history_walk_agreed"] = (walked[p] == r["verdict"])

vc = collections.Counter(r["verdict"] for r in rows)
n_eol = vc.get("EOL_ONLY", 0)
json.dump({"repo_head": REPO_HEAD, "thor_head": THOR_HEAD,
           "thor_head_is_ancestor_of_repo_head": True,
           "commits_repo_ahead": 1273,
           "closure_entry": "stack/scripts/refa_v1_train.py",
           "closure_pythonpath": "/home/nvidia/TanitAD/stack",
           "closure_n": len(closure),
           "n_raw_sha_drift": len(rows),
           "n_eol_only": n_eol,
           "n_real_drift": len(rows) - n_eol,
           "verdict_counts": dict(vc),
           "note": __doc__,
           "rows": rows},
          open(os.path.join(SP, "A13_DRIFT_TABLE.json"), "w", encoding="utf-8"), indent=1)
for k, v in vc.most_common():
    print("  %4d %s" % (v, k))
agreed = [r for r in rows if "history_walk_agreed" in r]
print("history-walk cross-check: %d rows, agreed %d, disagreed %d"
      % (len(agreed), sum(1 for r in agreed if r["history_walk_agreed"]),
         sum(1 for r in agreed if not r["history_walk_agreed"])))
for r in agreed:
    if not r["history_walk_agreed"]:
        print("   DISAGREE %s: table=%s walk=%s"
              % (r["path"], r["verdict"], r["history_walk_verdict"]))
print("wrote A13_DRIFT_TABLE.json")
