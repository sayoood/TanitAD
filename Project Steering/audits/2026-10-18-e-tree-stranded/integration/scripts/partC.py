"""Part C: the LANDING_READY entries absent from the tip (audit section 7), re-checked against a given tip."""
import json, csv, os, sys, subprocess, collections

MIRROR = "C:/Users/Admin/tanitad-push"
TIP = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "C:/Users/Admin/integ/work/C.json"
AUD = "C:/Users/Admin/estrand/out/Project Steering/audits/2026-10-18-e-tree-stranded/"
ER = "E:/Projects/TanitAD/"
BS = "\\"


def lp(path):
    fp = ER + path
    if len(fp) > 240:
        fp = BS * 2 + "?" + BS + fp.replace("/", BS)
    return fp


def git(*a):
    return subprocess.run(["git", "-C", MIRROR, "-c", "core.autocrlf=false", "--no-optional-locks", *a], capture_output=True).stdout


tree = set()
base = collections.defaultdict(list)
for rec in git("ls-tree", "-r", "-z", "--name-only", TIP).split(b"\0"):
    if rec:
        p = rec.decode("utf8", "surrogateescape")
        tree.add(p)
        base[p.rsplit("/", 1)[-1]].append(p)
xref = json.load(open("C:/Users/Admin/estrand/out/work/landing_ready_xref.json", encoding="utf8"))
s4 = {}
for row in csv.DictReader(open(AUD + "s4_list.csv", encoding="utf8", newline="")):
    s4[row["path"]] = row
cls = json.load(open(AUD + "classes.json", encoding="utf8"))["entries"]
land_paths = set(x[0] for x in json.load(open("C:/Users/Admin/estrand/out/paths.json", encoding="utf8")))
P3 = set(x[0] for x in json.load(open("C:/Users/Admin/estrand/out/paths_P3_optional.json", encoding="utf8")))
SCR = set(x[0] for x in json.load(open("C:/Users/Admin/estrand/out/paths_scrubbed.json", encoding="utf8")))

dirs = set()
for p in tree:
    parts = p.split("/")
    for k in range(1, len(parts)):
        dirs.add("/".join(parts[:k]))


def clean(p, man):
    s0 = p.strip()
    cands = [s0]
    for sep in ["  #", " | ", " (", "  ->", " -> ", "	", "  ", " md5 ", " -- ", " # "]:
        c = s0.split(sep)[0].strip()
        if c and c not in cands:
            cands.append(c)
    toks = s0.split(" ")
    for k in range(len(toks) - 1, 0, -1):
        c = " ".join(toks[:k]).strip().rstrip(",;.")
        if c and c not in cands:
            cands.append(c)
    md = man.rsplit("/", 1)[0]
    for c in list(cands):
        cands.append(md + "/" + c)
    for c in cands:
        c = c.rstrip("/")
        if c in tree:
            return c, "file-on-tip"
        if c in dirs:
            return c, "dir-on-tip"
    for c in cands:
        c = c.rstrip("/")
        if c and os.path.exists(lp(c)):
            return c, "in-E"
    return s0, "unresolved"


rows, per = [], []
for m in xref:
    ab = m.get("not_on_tip_paths") or []
    if not ab:
        continue
    cnt = collections.Counter()
    for p0 in ab:
        p, how = clean(p0, m["manifest"])
        inE = os.path.exists(lp(p)) if how in ("in-E", "unresolved") else False
        rec = {"manifest": m["manifest"], "path": p, "in_E": inE, "audit_entry": p0[:200] if p0 != p else None, "parse": how}
        if p0 != p:
            rec["parse_artifact"] = True
        if how == "dir-on-tip":
            rec["reason"] = "PARSE-ARTIFACT:DIR-ON-TIP"
        elif p in tree and p0 != p:
            rec["reason"] = "PARSE-ARTIFACT:FILE-ON-TIP"
        elif p in tree:
            rec["reason"] = "LANDED-SINCE-AUDIT"
        elif inE:
            r = s4.get(p)
            c = (cls.get(p) or {}).get("c")
            if r:
                rec["recommendation"] = r["recommendation"]
                rec["bytes"] = int(r["bytes"])
            if p in land_paths:
                rec["reason"] = "PENDING-LANDING (in audit paths.json, not yet on tip)"
            elif p in P3:
                rec["reason"] = "OPTIONAL-P3 (5-20 MB, separate commit)"
            elif p in SCR:
                rec["reason"] = "HOLD-SCRUB (redacted copy listed in paths_scrubbed.json)"
            elif r and r["recommendation"] == "BULK-ONLY":
                rec["reason"] = "BULK (> 20 MB or media/checkpoint; stays on D:/E:)"
            elif r and r["recommendation"] in ("DROP-scratch",):
                rec["reason"] = "DROP-scratch"
            elif r and r["recommendation"] == "GITIGNORED-DERIVED":
                rec["reason"] = "GITIGNORED-DERIVED"
            elif c in ("S4m", "S4h"):
                rec["reason"] = "MOVED (content on tip/history under another path)"
            elif c:
                rec["reason"] = f"IN-E-CLASS-{c}-NOT-IN-LAND-SET"
            else:
                rec["reason"] = "IN-E-UNCLASSIFIED(ignored by E .gitignore or not in population)"
        else:
            bn = p.rsplit("/", 1)[-1]
            alt = [q for q in base.get(bn, []) if q != p]
            # same package root = same first 4 path components
            root4 = "/".join(p.split("/")[:4])
            near = [q for q in alt if q.startswith(root4)]
            if near:
                rec["reason"] = "RENAMED/MOVED-WITHIN-PACKAGE (same basename on tip)"
                rec["on_tip_as"] = near[0]
            elif alt:
                rec["reason"] = "RENAMED/MOVED (same basename elsewhere on tip)"
                rec["on_tip_as"] = alt[0]
            else:
                rec["reason"] = "ABSENT-EVERYWHERE (not on E:, no same-basename file on the tip)"
        cnt[rec["reason"].split(" (")[0]] += 1
        rows.append(rec)
    per.append({"manifest": m["manifest"], "absent_at_audit": len(ab), "classes": dict(cnt)})
json.dump({"tip": TIP, "per_manifest": per, "rows": rows}, open(OUT, "w"), indent=1)
tot = collections.Counter(r["reason"].split(" (")[0] for r in rows)
print("tip", TIP[:8], "absent at audit", len(rows), dict(tot))
for p in per:
    print(p["absent_at_audit"], p["manifest"][-70:], p["classes"])
