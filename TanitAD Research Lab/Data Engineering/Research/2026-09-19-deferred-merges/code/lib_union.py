"""Key-level three-way union of TanitAD Research Lab/Library/library.json.

Sources: tip 2a88524 (T1), D:'s pre-sync backup 57e2944 (B, base d221843 = Bp), and
D:'s CURRENT worktree (W, today's LAB-RUN-016 on top of 37645fc = T0).

Field rule = kb_add._merge_entry's own rule for a re-banked paper:
  tags, cited_by -> sorted set union;  note -> append each ' | ' segment not already
  contained;  every other field (path, sha256, bytes, title, ...) is NEVER rewritten
  -- a disagreement there is reported as a CONFLICT and the tip's value kept.
Resurrection guard: a key present in the base Bp but ABSENT from the tip was deleted
by the tip; it is NOT re-added from B or W, only reported.
The write happens inside kb_add._IndexLock and through kb_add._save (same format).
"""
import json
import subprocess
import sys

R = "D:/Projects/TanitAD"
P = "TanitAD Research Lab/Library/library.json"
sys.path.insert(0, R + "/tools")
import kb_add  # noqa: E402


def ref(r):
    return json.loads(subprocess.run(["git", "-C", R, "show", "%s:%s" % (r, P)],
                                     capture_output=True).stdout.decode("utf-8"))["entries"]


Bp, B, T0, T1 = ref("d221843"), ref("57e2944"), ref("37645fc"), ref("2a88524")
SCALARS = ("kind", "title", "authors", "published", "abstract", "url", "path",
           "bytes", "sha256", "banked")
FILLABLE = ("title", "authors", "published", "abstract")
DRY = "--dry" in sys.argv


def merge_into(e, o, key, src, conflicts):
    for f in ("tags", "cited_by"):
        e[f] = sorted(set(e.get(f) or []) | set(o.get(f) or []))
    for seg in [s.strip() for s in (o.get("note") or "").split(" | ") if s.strip()]:
        if seg not in (e.get("note") or ""):
            e["note"] = ((e.get("note") or "") + " | " + seg).strip(" |")
    for f in SCALARS:
        if f not in o or o.get(f) == e.get(f):
            continue
        if f == "banked":                       # first banking is the provenance
            e[f] = min(x for x in (e.get(f), o.get(f)) if x)
        elif f in FILLABLE and not e.get(f) and o.get(f):
            e[f] = o[f]                         # fill an EMPTY field, never overwrite
        elif f in FILLABLE and e.get(f) and not o.get(f):
            pass                                # the other side is empty: keep ours
        else:
            conflicts.append((key, f, src))     # a real, non-empty disagreement


with kb_add._IndexLock(kb_add.INDEX_JSON):
    W = kb_add._load()["entries"]                      # re-read INSIDE the lock
    schema = kb_add._load().get("schema")
    out, conflicts, resurrect = {}, [], []
    order = list(T1) + [k for k in B if k not in T1] + [k for k in W if k not in T1 and k not in B]
    for k in order:
        deleted_by_tip = (k in Bp) and (k not in T1)
        if deleted_by_tip:
            resurrect.append((k, "B" if k in B else "", "W" if k in W else ""))
            continue
        srcs = [(n, s[k]) for n, s in (("T1", T1), ("B", B), ("W", W)) if k in s]
        e = json.loads(json.dumps(srcs[0][1]))
        for n, o in srcs[1:]:
            merge_into(e, o, k, n, conflicts)
        out[k] = e
    stats = {
        "tip": len(T1), "backup": len(B), "worktree": len(W), "merged": len(out),
        "only_in_backup": sum(1 for k in B if k not in T1 and k not in W),
        "only_in_worktree": sum(1 for k in W if k not in T1 and k not in B),
        "in_backup_and_worktree_not_tip": sum(1 for k in B if k in W and k not in T1),
        "tip_keys_missing_from_worktree": sum(1 for k in T1 if k not in W),
        "scalar_conflicts": len(conflicts), "resurrection_refused": len(resurrect),
    }
    print(json.dumps(stats, indent=1))
    for c in conflicts[:12]:
        print("  CONFLICT", c)
    for r in resurrect[:12]:
        print("  REFUSED-RESURRECTION", r)
    # superset assertions -- nothing any source carried may be lost, except refused keys
    for name, src in (("T1", T1), ("B", B), ("W", W)):
        for k, o in src.items():
            if k in out:
                assert set(o.get("tags") or []) <= set(out[k]["tags"]), (name, k, "tags")
                assert set(o.get("cited_by") or []) <= set(out[k]["cited_by"]), (name, k, "cited")
                for seg in [s.strip() for s in (o.get("note") or "").split(" | ") if s.strip()]:
                    assert seg in out[k]["note"], (name, k, "note")
    assert all(k in out for k in T1), "a tip key was dropped"
    print("superset assertions: PASS (every tag, citation and note segment of T1, B and W survives)")
    if not DRY:
        kb_add._save({"schema": schema, "entries": out})
        print("written through kb_add._save:", kb_add.INDEX_JSON)
