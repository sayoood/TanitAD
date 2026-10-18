"""Part B: the 47 S2b CANDIDATE-EDITs: content verdict vs the tip + its history. Read-only on E:."""
import subprocess, json, re, sys, os, collections, hashlib

MIRROR = "C:/Users/Admin/tanitad-push"
TIP = sys.argv[1]
AUD = "C:/Users/Admin/estrand/out/Project Steering/audits/2026-10-18-e-tree-stranded/AUDIT.md"
ER = "E:/Projects/TanitAD/"
OUT = "C:/Users/Admin/integ/work/B.json"
TMP = "C:/Users/Admin/integ/work/tmpB"
os.makedirs(TMP, exist_ok=True)


def git(*a, check=True):
    r = subprocess.run(["git", "-C", MIRROR, "-c", "core.autocrlf=false", "--no-optional-locks", *a], capture_output=True)
    if check and r.returncode != 0:
        raise RuntimeError(r.stderr.decode("utf8", "replace"))
    return r.stdout


def blobid(b):
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def lines_of(b):
    t = b.decode("utf8", "replace").replace("\r\n", "\n")
    return [l.rstrip() for l in t.split("\n")]


cache = {}


def getblob(i):
    if i not in cache:
        cache[i] = git("cat-file", "blob", i)
    return cache[i]


def parse_hunks(diff):
    hunks, cur = [], None
    for l in diff.split("\n"):
        if l.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", l)
            cur = {"a_start": int(m.group(1)), "add": [], "rem": []}
            hunks.append(cur)
        elif cur is not None:
            if l.startswith("+") and not l.startswith("+++"):
                cur["add"].append(l[1:])
            elif l.startswith("-") and not l.startswith("---"):
                cur["rem"].append(l[1:])
    return hunks


txt = open(AUD, encoding="utf8").read()
sec = txt.split("## 4. S2b")[1].split("Verdict counts:")[0]
rows = []
for ln in sec.splitlines():
    if not ln.startswith("| ") or ln.startswith("| path") or ln.startswith("|---"):
        continue
    c = [x.strip() for x in ln.strip().strip("|").split(" | ")]
    rows.append({"path": c[0], "origin": c[1], "audit_verdict": c[2], "mtime_hint": c[3], "e_mtime": c[7], "tip_last_change": c[8]})
print(len(rows), "S2b rows")

res = []
for r in rows:
    p = r["path"]
    rec = dict(r)
    eb = open(ER + p, "rb").read()
    rec["e_blob"] = blobid(eb)
    rec["e_blob_lf"] = blobid(eb.replace(b"\r\n", b"\n"))
    tb_id = git("rev-parse", f"{TIP}:{p}", check=False).decode().strip()
    if not tb_id or len(tb_id) != 40:
        rec["verdict"] = "OBSOLETE"
        rec["note"] = "path no longer on the tip"
        res.append(rec)
        continue
    rec["tip_blob"] = tb_id
    tb = getblob(tb_id)
    rec["byte_equal_tip"] = rec["e_blob"] == tb_id
    rec["lf_equal_tip"] = rec["e_blob_lf"] == blobid(tb.replace(b"\r\n", b"\n"))
    # history (tip) and all refs
    hist = []
    out = git("log", "--format=C %H %ci %s", "--raw", "--no-renames", "-m", TIP, "--", p).decode("utf8", "replace")
    cur = None
    for l in out.splitlines():
        if l.startswith("C "):
            q = l.split(" ", 4)
            cur = (q[1], q[2] + " " + q[3], q[4] if len(q) > 4 else "")
        elif l.startswith(":"):
            f = l.split()
            hist.append({"commit": cur[0], "date": cur[1], "subj": cur[2], "new": f[3]})
    rec["hist_n"] = len(hist)
    rec["tip_last_commit"] = hist[0]["commit"][:8] if hist else None
    rec["tip_last_date"] = hist[0]["date"][:16] if hist else None
    rec["tip_last_subj"] = hist[0]["subj"][:120] if hist else None
    hit = [h for h in hist if h["new"] in (rec["e_blob"], rec["e_blob_lf"])]
    # LF-variant of historical blobs
    if not hit:
        for h in hist:
            if h["new"] != "0" * 40 and blobid(getblob(h["new"]).replace(b"\r\n", b"\n")) == rec["e_blob_lf"]:
                hit.append(h)
                break
    rec["e_blob_is_history"] = [hit[0]["commit"][:8], hit[0]["date"][:16], hit[0]["subj"][:100]] if hit else None
    el = lines_of(eb)
    tl = lines_of(tb)
    ce, ct = collections.Counter(el), collections.Counter(tl)
    only_e = sum((ce - ct).values())
    only_t = sum((ct - ce).values())
    only_e_nonblank = sum(v for k, v in (ce - ct).items() if k.strip())
    rec["e_only_lines"] = only_e
    rec["e_only_nonblank"] = only_e_nonblank
    rec["tip_only_lines"] = only_t
    # closest base among history
    best = None
    for h in hist:
        if h["new"] == "0" * 40:
            continue
        cb = collections.Counter(lines_of(getblob(h["new"])))
        d = sum(((ce - cb) + (cb - ce)).values())
        if best is None or d < best[0]:
            best = (d, h)
    if best:
        d, h = best
        rec["base"] = {"commit": h["commit"][:8], "date": h["date"][:16], "linediff_to_E": d, "subj": h["subj"][:100]}
        pb, pv = os.path.join(TMP, "b.txt"), os.path.join(TMP, "v.txt")
        open(pb, "w", encoding="utf8", newline="\n").write("\n".join(lines_of(getblob(h["new"]))))
        open(pv, "w", encoding="utf8", newline="\n").write("\n".join(el))
        dr = subprocess.run(["git", "diff", "--no-index", "--no-color", "--histogram", "-U0", pb, pv], capture_output=True)
        hunks = parse_hunks(dr.stdout.decode("utf8", "replace"))
        ttext = "\n".join(tl)
        hs = []
        for hk in hunks:
            a_txt, r_txt = "\n".join(hk["add"]), "\n".join(hk["rem"])
            a_in = bool(hk["add"]) and a_txt in ttext
            r_in = bool(hk["rem"]) and r_txt in ttext
            if not hk["add"]:
                st = "DEL-APPLIED" if not r_in else "DEL-MISSING"
            elif a_in and not r_in:
                st = "APPLIED"
            elif a_in and r_in:
                st = "BOTH-PRESENT"
            elif not a_in and r_in:
                st = "MISSING"
            else:
                st = "CHANGED"
            hs.append({"a_start": hk["a_start"], "n_add": len(hk["add"]), "n_rem": len(hk["rem"]), "status": st,
                       "add_head": hk["add"][:2], "rem_head": hk["rem"][:2]})
        rec["n_hunks"] = len(hs)
        rec["hunk_counts"] = dict(collections.Counter(x["status"] for x in hs))
        rec["hunks"] = hs
    res.append(rec)
    print(p[-60:], "| eq", rec["byte_equal_tip"], rec["lf_equal_tip"], "| hist", bool(rec["e_blob_is_history"]), "| E-only", only_e_nonblank,
          "tip-only", only_t, "|", rec.get("hunk_counts"), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
