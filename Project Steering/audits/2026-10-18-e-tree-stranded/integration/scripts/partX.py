"""Extension: every `package path -> repo path` mapping declared in an E: LANDING_READY manifest, vs the tip."""
import subprocess, json, re, sys, os, glob, hashlib, collections

MIRROR = "C:/Users/Admin/tanitad-push"
TIP = sys.argv[1]
ER = "E:/Projects/TanitAD/"
OUT = "C:/Users/Admin/integ/work/X.json"
BS = "\\"


def git(*a, check=True, inp=None):
    r = subprocess.run(["git", "-C", MIRROR, "-c", "core.autocrlf=false", "--no-optional-locks", *a], capture_output=True, input=inp)
    if check and r.returncode != 0:
        raise RuntimeError(r.stderr.decode("utf8", "replace"))
    return r.stdout


def blobid(b):
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def lp(path):
    fp = (ER + path)
    if len(fp) > 240:
        fp = BS * 2 + "?" + BS + fp.replace("/", BS)
    return fp


# tip tree: path -> blob
tree = {}
for rec in git("ls-tree", "-r", "-z", TIP).split(b"\0"):
    if not rec:
        continue
    meta, path = rec.split(b"\t", 1)
    tree[path.decode("utf8", "surrogateescape")] = meta.split(b" ")[2].decode()
print("tip files", len(tree))

# manifests = every LANDING_READY*.txt on E: tree (walk only dirs under the known roots)
mans = [r["manifest"] for r in json.load(open("C:/Users/Admin/estrand/out/work/landing_ready_xref.json", encoding="utf8"))]
print("manifests", len(mans))

MAP = re.compile(r"^(\S.*?)\s+->\s+(\S.*)$")
HDR = re.compile(r"^#\s+(\S.*?): base (\S+) \| new (\S+)")
SUP = re.compile(r"SUPERSEDED", re.I)
maps = {}
for m in mans:
    try:
        txt = open(lp(m), "rb").read().decode("utf8", "replace").replace("\r", "")
    except Exception as ex:
        continue
    lines = txt.split("\n")
    base = new = None
    for i, ln in enumerate(lines):
        h = HDR.match(ln)
        if h:
            base, new = h.group(2), h.group(3)
            continue
        if ln.startswith("# SUPERSEDED") or (ln.startswith("#") and SUP.search(ln) and "->" in ln):
            continue
        mm = MAP.match(ln) if not ln.startswith("#") else None
        if mm:
            pkg, repo = mm.group(1).strip(), mm.group(2).strip()
            key = (pkg, repo)
            if key not in maps:
                maps[key] = {"pkg": pkg, "repo": repo, "manifests": [m], "decl_base": base, "decl_new": new}
            else:
                maps[key]["manifests"].append(m)
            base = new = None
print("mappings", len(maps))

# only code-ish repo targets (stack/, taniteval/, tools/, colab/, Project Steering/ code)
res = []
hist_cache = {}


def hist(path):
    if path not in hist_cache:
        out = git("log", "--format=C %H %ci", "--raw", "--no-renames", "-m", TIP, "--", path).decode("utf8", "replace")
        h, cur = [], None
        for l in out.splitlines():
            if l.startswith("C "):
                q = l.split(" ", 3)
                cur = (q[1], q[2])
            elif l.startswith(":"):
                f = l.split()
                h.append((cur[0], cur[1], f[3]))
        hist_cache[path] = h
    return hist_cache[path]


cnt = collections.Counter()
for key, r in sorted(maps.items()):
    pkg, repo = r["pkg"], r["repo"]
    rec = dict(r)
    pkgp = pkg
    if not os.path.exists(lp(pkgp)):
        alt = os.path.dirname(r["manifests"][0]) + "/" + pkg
        if os.path.exists(lp(alt)):
            pkgp = alt
            rec["pkg_resolved"] = alt
    pkg = pkgp
    if not re.match(r"^(stack|taniteval|tools|colab|Project Steering|data|Benchmarks|FlyWheels|TanitAD Research Lab|products|Media|Paper|docs)/", repo) or pkg.startswith(("*", "`", "C:")):
        rec["status"] = "NOT-A-MAPPING"
        res.append(rec)
        cnt[rec["status"]] += 1
        continue
    if not os.path.exists(lp(pkg)):
        rec["status"] = "PKG-FILE-GONE"
        res.append(rec)
        cnt[rec["status"]] += 1
        continue
    b = open(lp(pkg), "rb").read()
    rec["pkg_blob"] = blobid(b)
    rec["pkg_blob_lf"] = blobid(b.replace(b"\r\n", b"\n"))
    tb = tree.get(repo)
    rec["tip_blob"] = tb
    if pkg in tree:
        rec["pkg_on_tip"] = True
    if tb is None:
        rec["status"] = "NEW-NOT-ON-TIP"
    else:
        tdat = git("cat-file", "blob", tb)
        tlf = blobid(tdat.replace(b"\r\n", b"\n"))
        if tb == rec["pkg_blob"] or tlf == rec["pkg_blob_lf"]:
            rec["status"] = "APPLIED-EXACT"
        else:
            h = hist(repo)
            hit = [x for x in h if x[2] in (rec["pkg_blob"], rec["pkg_blob_lf"])]
            if not hit:
                for x in h:
                    if x[2] != "0" * 40 and blobid(git("cat-file", "blob", x[2]).replace(b"\r\n", b"\n")) == rec["pkg_blob_lf"]:
                        hit = [x]
                        break
            if hit:
                rec["status"] = "APPLIED-THEN-SUPERSEDED"
                rec["hist_commit"] = hit[0][0][:8]
                rec["hist_date"] = hit[0][1][:16]
            else:
                rec["status"] = "DIFFERS-NOT-IN-HISTORY"
                db = r["decl_base"]
                rec["tip_equals_declared_base"] = bool(db) and tb.startswith(db[:7]) if db and db != "NEW" else None
    cnt[rec["status"]] += 1
    res.append(rec)
json.dump(res, open(OUT, "w"), indent=1)
print(dict(cnt))
