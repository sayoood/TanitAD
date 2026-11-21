"""Part A2: hunk-containment analysis. For each variant: closest historical base, diff(base->variant) hunks,
then test each hunk's added/removed block against the tip text and against history."""
import subprocess, json, re, sys, os, collections, tempfile, hashlib

MIRROR = "C:/Users/Admin/tanitad-push"
TIP = sys.argv[1]
ER = "E:/Projects/TanitAD/"
A = json.load(open("C:/Users/Admin/integ/work/A.json"))
OUT = "C:/Users/Admin/integ/work/A2.json"
TMP = "C:/Users/Admin/integ/work/tmp"
os.makedirs(TMP, exist_ok=True)


def git(*a, inp=None, check=True):
    r = subprocess.run(["git", "-C", MIRROR, "-c", "core.autocrlf=false", "--no-optional-locks", *a],
                       capture_output=True, input=inp)
    if check and r.returncode != 0:
        raise RuntimeError(r.stderr.decode("utf8", "replace"))
    return r.stdout


def lines_of(b):
    t = b.decode("utf8", "replace").replace("\r\n", "\n")
    return [l.rstrip() for l in t.split("\n")]


blob_cache = {}


def getblob(i):
    if i not in blob_cache:
        blob_cache[i] = lines_of(git("cat-file", "blob", i))
    return blob_cache[i]


def parse_hunks(diff):
    hunks = []
    cur = None
    for l in diff.split("\n"):
        if l.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", l)
            cur = {"a_start": int(m.group(1)), "b_start": int(m.group(3)), "add": [], "rem": []}
            hunks.append(cur)
        elif cur is not None:
            if l.startswith("+") and not l.startswith("+++"):
                cur["add"].append(l[1:])
            elif l.startswith("-") and not l.startswith("---"):
                cur["rem"].append(l[1:])
    return hunks


results = []
for r in A:
    tgt = r["target"]
    vb = open(ER + r["variant"], "rb").read()
    vl = lines_of(vb)
    rec = {"variant": r["variant"], "target": tgt}
    if not r["tip_blob"]:
        rec["note"] = "target NEW on tip"
        results.append(rec)
        continue
    tl = getblob(r["tip_blob"])
    ttext = "\n".join(tl)
    # candidate bases: unique historical blobs on tip history (+ all-refs blobs)
    cands = collections.OrderedDict()
    for h in r["hist_tip"]:
        if h["new"] != "0" * 40:
            cands[h["new"]] = h
    cv = collections.Counter(vl)
    best = None
    for bid, h in cands.items():
        bl = getblob(bid)
        cb = collections.Counter(bl)
        d = sum(((cv - cb) + (cb - cv)).values())
        if best is None or d < best[0]:
            best = (d, bid, h)
    d, bid, h = best
    rec["base_blob"] = bid
    rec["base_commit"] = h["commit"]
    rec["base_date"] = h["date"]
    rec["base_subj"] = h["subj"][:140]
    rec["base_to_variant_linediff"] = d
    rec["tip_to_variant_linediff"] = sum(((cv - collections.Counter(tl)) + (collections.Counter(tl) - cv)).values())
    bl = getblob(bid)
    pb = os.path.join(TMP, "base.txt")
    pv = os.path.join(TMP, "var.txt")
    open(pb, "w", encoding="utf8", newline="\n").write("\n".join(bl))
    open(pv, "w", encoding="utf8", newline="\n").write("\n".join(vl))
    dr = subprocess.run(["git", "diff", "--no-index", "--no-color", "--histogram", "-U0", pb, pv], capture_output=True)
    diff = dr.stdout.decode("utf8", "replace")
    hunks = parse_hunks(diff)
    # history texts, oldest -> newest
    hist_old = list(reversed([x for x in r["hist_tip"] if x["new"] != "0" * 40]))
    out_h = []
    for hk in hunks:
        a_txt = "\n".join(hk["add"])
        r_txt = "\n".join(hk["rem"])
        a_in = bool(hk["add"]) and (a_txt in ttext)
        r_in = bool(hk["rem"]) and (r_txt in ttext)
        a_lines_present = (sum(1 for x in hk["add"] if x in set(tl)) / len(hk["add"])) if hk["add"] else None
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
        first = None
        if hk["add"] and a_in:
            for x in hist_old:
                if a_txt in "\n".join(getblob(x["new"])):
                    first = (x["commit"][:8], x["date"][:10])
                    break
        out_h.append({"a_start": hk["a_start"], "n_add": len(hk["add"]), "n_rem": len(hk["rem"]), "status": st,
                      "add_lines_present_frac": a_lines_present, "first_commit_with_add": first,
                      "add_head": hk["add"][:3], "rem_head": hk["rem"][:3]})
    rec["n_hunks"] = len(hunks)
    cnt = collections.Counter(h["status"] for h in out_h)
    rec["status_counts"] = dict(cnt)
    rec["hunks"] = out_h
    # persist diffs
    safe = hashlib.md5(r["variant"].encode()).hexdigest()[:8]
    open(os.path.join(TMP, f"base2var_{safe}.diff"), "w", encoding="utf8").write(diff)
    rec["diff_file"] = f"base2var_{safe}.diff"
    results.append(rec)
    print(r["variant"].split("/")[-5] + "|" + r["variant"].split("/")[-4], tgt.split("/")[-1], "base", h["commit"][:8], h["date"][:10],
          "d(base,var)", d, "d(tip,var)", rec["tip_to_variant_linediff"], "hunks", len(hunks), dict(cnt), flush=True)
json.dump(results, open(OUT, "w"), indent=1)
