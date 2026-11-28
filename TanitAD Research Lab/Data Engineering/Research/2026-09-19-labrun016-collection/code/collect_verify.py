"""Verify the LAB-RUN-016 files staged in D: against the tip (read from the push mirror
by plumbing). Content is read from D:'s INDEX BLOBS -- the exact bytes a lander takes."""
import hashlib
import json
import re
import subprocess
import sys

D, M = "D:/Projects/TanitAD", "C:/Users/Admin/tanitad-push"
TIP = open("C:/Users/Admin/AppData/Local/Temp/claude/tip.txt").read().strip()
sys.path.insert(0, D + "/tools")
import kb_add  # noqa: E402  (for _pdf_truncated's exact rule)

U = re.compile(rb"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
CRED = re.compile(rb"hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{32,}|AKIA[0-9A-Z]{16}")
uni = json.load(open("C:/Users/Admin/AppData/Local/Temp/claude/clip_universe.json"))
IDS = set(uni["ids"])
DIG = set().union(*[set(v) for v in uni["digests"].values()])
PRE8 = {i[:8] for i in IDS}


def g(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True).stdout


land = json.load(open("C:/Users/Admin/AppData/Local/Temp/claude/landscape.json", encoding="utf-8"))
LAB = [r for r in land["rows"] if r["where"] == "index" and (
    r["path"].startswith("TanitAD Research Lab/") or r["path"].startswith("stack/scripts/nuplan_")
    or r["path"].startswith("stack/tests/test_nuplan_"))]
tip_lib = json.loads(g(M, "show", TIP + ":TanitAD Research Lab/Library/library.json"))["entries"]
by_path = {e["path"]: e for e in tip_lib.values()}
out, fails = [], 0
for r in LAB:
    p = r["path"]
    blob = g(D, "ls-files", "--stage", "--", p).decode().split()[1]
    data = g(D, "cat-file", "-p", blob)
    rec = {"path": p, "blob": blob, "bytes": len(data), "status": "differs" if r["on_tip"] else "new"}
    # scan: UUID-shaped strings are CLASSIFIED against the 306,152-clip universe
    uu = {m.group(0).decode().lower() for m in U.finditer(data)}
    rec["uuid_shaped"] = len(uu)
    rec["clip_ids"] = sorted(hashlib.sha256(u.encode()).hexdigest()[:12] for u in uu
                             if u in IDS or hashlib.sha256(u.encode()).hexdigest() in DIG)
    rec["credentials"] = len(CRED.findall(data))
    if not p.endswith(".pdf"):
        toks = set(re.findall(rb"(?<![0-9a-f])([0-9a-f]{8})(?![0-9a-f])", data))
        rec["clip_prefix_hits"] = sum(1 for t in toks if t.decode() in PRE8)
    if p.endswith(".pdf"):
        e = by_path.get(p)
        rec["in_tip_library"] = e is not None
        rec["sha256_match"] = bool(e) and hashlib.sha256(data).hexdigest() == e["sha256"]
        rec["bytes_match"] = bool(e) and str(len(data)) == str(e["bytes"])
        tail = data[-2048:]
        rec["pdf_complete"] = data[:5] == b"%PDF-" and b"%%EOF" in tail
        ok = rec["in_tip_library"] and rec["sha256_match"] and rec["pdf_complete"]
    elif r["on_tip"]:
        tip_txt = g(M, "show", "%s:%s" % (TIP, p)).replace(b"\r\n", b"\n").decode("utf-8", "replace")
        new_txt = data.replace(b"\r\n", b"\n").decode("utf-8", "replace")
        nl = set(new_txt.split("\n"))
        lost = [l for l in tip_txt.split("\n") if l.strip() and l not in nl]
        rec["tip_lines_absent"] = len(lost)
        rec["lines_added"] = len([l for l in new_txt.split("\n") if l not in set(tip_txt.split("\n"))])
        ok = not lost
    else:
        ok = True
    ok = ok and not rec["clip_ids"] and not rec["credentials"] and not rec.get("clip_prefix_hits")
    rec["pass"] = ok
    fails += not ok
    out.append(rec)

# the Library is complete only if every tip-library path absent from the tip TREE is here
tip_tree = set(g(M, "ls-tree", "-r", "--name-only", "-z", TIP).decode("utf-8").split("\0"))
needed = sorted(p for p in by_path if p not in tip_tree)
have = {r["path"] for r in out if r["path"].endswith(".pdf")}
summary = {
    "tip": TIP[:10], "lab_paths": len(out),
    "new": sum(r["status"] == "new" for r in out), "differs": sum(r["status"] == "differs" for r in out),
    "pdfs": len(have), "library_refs_missing_from_tip_tree": len(needed),
    "library_refs_missing_and_not_collected": len(set(needed) - have),
    "collected_pdfs_not_in_tip_library": len(have - set(by_path)),
    "fails": fails,
}
print(json.dumps(summary, indent=1))
for r in out:
    if not r["pass"] or r["status"] == "differs":
        print("  %-5s %-8s %-78s %s" % ("PASS" if r["pass"] else "FAIL", r["status"], r["path"][-78:],
                                        {k: r[k] for k in ("tip_lines_absent", "lines_added", "clip_ids",
                                                            "credentials", "clip_prefix_hits", "uuid_shaped")
                                         if k in r and r[k]}))
json.dump({"summary": summary, "files": out},
          open("C:/Users/Admin/AppData/Local/Temp/claude/collect_verify.json", "w"), indent=1)
