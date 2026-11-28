"""Part A: classify the 29 S4 package-local code variants vs the tip. Read-only on E:."""
import subprocess, json, re, sys, os, hashlib, collections, difflib, tempfile

MIRROR = "C:/Users/Admin/tanitad-push"
TIP = sys.argv[1]
AUD = "C:/Users/Admin/estrand/out/Project Steering/audits/2026-10-18-e-tree-stranded/AUDIT.md"
ER = "E:/Projects/TanitAD/"
OUT = "C:/Users/Admin/integ/work/A.json"


def git(*a, check=True):
    r = subprocess.run(["git", "-C", MIRROR, "-c", "core.autocrlf=false", "--no-optional-locks", *a],
                       capture_output=True)
    if check and r.returncode != 0:
        raise RuntimeError(r.stderr.decode("utf8", "replace"))
    return r.stdout


def blobid(b):
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def norm(b):
    return b.replace(b"\r\n", b"\n")


# parse the table
txt = open(AUD, encoding="utf8").read()
sec = txt.split("### S4 package-local code that targets the repo")[1].split("### S4 decision-grade")[0]
rows = []
for ln in sec.splitlines():
    m = re.match(r"\| (.+?) \| (.+?) \| (.+?) \|$", ln)
    if m and m.group(1).endswith((".py", ".json")) and "/stack/" in m.group(1) or (m and "/taniteval/" in m.group(1)):
        rows.append((m.group(1), m.group(2), m.group(3)))
print(len(rows), "rows")

res = []
for pkg, target, tflag in rows:
    r = {"variant": pkg, "target": target, "audit_flag": tflag}
    vb = open(ER + pkg, "rb").read()
    r["variant_blob"] = blobid(vb)
    r["variant_blob_lf"] = blobid(norm(vb))
    r["variant_eol"] = "CRLF" if b"\r\n" in vb else "LF"
    # tip blob
    try:
        tb_id = git("rev-parse", f"{TIP}:{target}").decode().strip()
    except RuntimeError:
        tb_id = None
    r["tip_blob"] = tb_id
    if tb_id:
        tb = git("cat-file", "blob", tb_id)
        r["tip_eol"] = "CRLF" if b"\r\n" in tb else "LF"
        r["tip_blob_lf"] = blobid(norm(tb))
        r["byte_equal_tip"] = (r["variant_blob"] == tb_id)
        r["lf_equal_tip"] = (r["variant_blob_lf"] == r["tip_blob_lf"])
    # history on tip
    hist = []
    out = git("log", "--format=C %H %ci %s", "--raw", "--no-renames", "-m", TIP, "--", target).decode("utf8", "replace")
    cur = None
    for l in out.splitlines():
        if l.startswith("C "):
            p = l.split(" ", 4)
            cur = (p[1], p[2] + " " + p[3], p[4] if len(p) > 4 else "")
        elif l.startswith(":"):
            f = l.split()
            hist.append({"commit": cur[0], "date": cur[1], "subj": cur[2], "old": f[2], "new": f[3], "st": f[4]})
    r["hist_tip_n"] = len(hist)
    r["hist_tip"] = hist
    # history over all refs
    out = git("log", "--format=C %H", "--raw", "--no-renames", "-m", "--all", "--", target).decode("utf8", "replace")
    allb = collections.OrderedDict()
    for l in out.splitlines():
        if l.startswith(":"):
            f = l.split()
            allb[f[3]] = 1
    r["hist_all_blobs_n"] = len(allb)
    hit = [h for h in hist if h["new"] == r["variant_blob"]]
    r["variant_blob_in_tip_history"] = hit[0]["commit"] if hit else None
    r["variant_blob_in_any_history"] = (r["variant_blob"] in allb) or (r["variant_blob_lf"] in allb)
    hitlf = []
    for h in hist:
        if h["new"] != "0" * 40 and h["new"] not in ("0" * 40,):
            pass
    res.append(r)
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
for r in res:
    print(r["variant"].split("/")[-4:][0:1], r["target"], "tipblob", (r["tip_blob"] or "NEW")[:8],
          "var", r["variant_blob"][:8], "eqtip", r.get("byte_equal_tip"), r.get("lf_equal_tip"),
          "inTipHist", r["variant_blob_in_tip_history"], "inAnyHist", r["variant_blob_in_any_history"], "histn", r["hist_tip_n"])
