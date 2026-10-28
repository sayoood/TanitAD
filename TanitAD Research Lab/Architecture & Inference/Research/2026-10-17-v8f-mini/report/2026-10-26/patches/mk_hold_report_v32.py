"""Hold: report v32 (7.24 SPEEDREAD NOT IN INPUT, 7.25 DAGGER2 HARMS, 7.26 DAG-ORC HARMS, the abstract and 8.1, 8.3, 8.4 rows 1w-1y) + its patch, checks and numbers audit.
python mk_hold_report_v32.py <live tip> [--out <hold dir>] [--regen]

It rebuilds both report files with patch_v32.py on <live tip> (blobs only, never a working tree) and stages them as mods-rw at their repo paths
(TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/report/2026-10-26/). The patch, the audit (audit_v32.py, audit_reviewed.py,
v32_NUMBERS_AUDIT.md), the checks (checks_v32.py, render_check.js, v32_checks.txt) and this builder go into stage/ under report/2026-10-26/patches/ as new files.
msg.txt ends with the Co-Authored-By line. The Master Mind lands the hold; this script only writes it, and writes nowhere else.

The audit and checks files are staged as the last run of audit_v32.py / checks_v32.py in this folder wrote them. The builder verifies that they were produced for the very
bytes it stages (the sha256 that v32_checks.txt and v32_NUMBERS_AUDIT.md record) and refuses otherwise. If the live tip moved since that run, pass --regen: the builder then
re-runs audit_v32.py and checks_v32.py on <live tip> against its own rebuild and stages those outputs (needs node and the local marked, as checks_v32.py does).
The default hold dir is C:/Users/Admin/qland/hold_report_v32/ (the builder refuses an existing dir)."""
import difflib
import hashlib
import os
import re
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
args = sys.argv[1:]
assert args and not args[0].startswith("--"), "usage: mk_hold_report_v32.py <live tip> [--out <hold dir>] [--regen]"
TIP = args[0]
HOLD = "C:/Users/Admin/qland/hold_report_v32/"
if "--out" in args:
    HOLD = args[args.index("--out") + 1].replace("\\", "/").rstrip("/") + "/"
REGEN = "--regen" in args
H = HOLD
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
RP = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/report/2026-10-26/"
assert not os.path.exists(H), H
FORBID = ("void", "superseded", "retracted", "formerly", "no longer", "correction", "withdrawn")
HFTOK = re.compile(rb"hf_[A-Za-z0-9]{8,}")
sha = lambda b: hashlib.sha256(b).hexdigest()
rd = lambda p: open(p, "rb").read()


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and len(H + rel) < 260, (rel, len(H + rel))
    assert not HFTOK.search(b), "hf_ token in " + rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


def run(cmd):
    r = subprocess.run(cmd, capture_output=True)
    assert r.returncode == 0, (cmd[1:3], r.stdout[-300:], r.stderr[-500:])
    return r.stdout.decode("utf-8", "replace")


full = subprocess.run(GIT + ["rev-parse", "--verify", TIP + "^{commit}"], capture_output=True)
assert full.returncode == 0, ("not a commit in the mirror", TIP)
FULL = full.stdout.decode().strip()
tpl = blob(RP + "page_template_v30.html")
old_md = blob(RP + "V8F_SCIENTIFIC_REPORT.md")
assert tpl is not None and old_md is not None and b"**Status:** research report, v31 of 2026-10-27" in old_md, "the live tip no longer carries the v31 report this patch builds on"
assert blob(RP + "patches/patch_v32.py") is None and blob(RP + "patches/mk_hold_report_v32.py") is None, "v32 patches are already at the tip"

# ---- rebuild on the live tip, in a scratch folder next to this script (removed at the end)
with tempfile.TemporaryDirectory(prefix="_v32hold_", dir=HERE) as WK:
    WK = WK.replace("\\", "/") + "/"
    print(run([sys.executable, HERE + "patch_v32.py", TIP, WK + "out"]).strip())
    md, html = rd(WK + "out/V8F_SCIENTIFIC_REPORT.md"), rd(WK + "out/v8f_report.html")
    assert b"\r" not in md and b"\r" not in html and ("results through repository tip `%s`" % FULL[:8]).encode() in md, "status line does not name the live tip"
    assert html == tpl.replace(b"%%MD%%", md), "html is not the template filled with the md"
    assert b"@@" not in md and b"DAGORC_RESULT" not in md
    added = [l for l in difflib.ndiff(old_md.decode("utf-8").splitlines(), md.decode("utf-8").splitlines()) if l.startswith("+ ")]
    for w in FORBID:
        hits = [l for l in added if re.search(r"\b" + w + r"\b", l.lower())]
        assert not hits, (w, hits[:2])
    here_md = HERE + "out/V8F_SCIENTIFIC_REPORT.md"
    same_as_here = os.path.exists(here_md) and rd(here_md) == md and rd(HERE + "out/v8f_report.html") == html
    print("rebuild on %s: %d bytes md, %d bytes html; identical to the out/ checked in this folder: %s" % (FULL[:8], len(md), len(html), same_as_here))

    if REGEN:
        aud, chk = WK + "v32_NUMBERS_AUDIT.md", WK + "v32_checks.txt"
        print(run([sys.executable, HERE + "audit_v32.py", TIP, WK + "out", aud]).strip())
        print(run([sys.executable, HERE + "checks_v32.py", TIP, WK + "out", chk, aud]).strip().splitlines()[-1])
        audit_b, checks_b = rd(aud), rd(chk)
    else:
        audit_b, checks_b = rd(HERE + "v32_NUMBERS_AUDIT.md"), rd(HERE + "v32_checks.txt")
    msha, hsha = sha(md), sha(html)
    assert (b"V8F_SCIENTIFIC_REPORT.md " + msha.encode()) in checks_b and (b"v8f_report.html " + hsha.encode()) in checks_b, \
        "v32_checks.txt was produced for other bytes than the rebuild: re-run checks_v32.py, or pass --regen"
    assert (b"`V8F_SCIENTIFIC_REPORT.md` " + msha.encode()) in audit_b, "v32_NUMBERS_AUDIT.md was produced for other bytes than the rebuild: re-run audit_v32.py, or pass --regen"
    assert f"source tip {FULL[:8]} = {FULL}".encode() in checks_b, "v32_checks.txt names another source tip"
    assert b"ZZSUPERSET-OKZZ" in checks_b and b"ZZDEMOTE-OKZZ" in checks_b and b"== patch_v32.py re-run on the v31 blobs reproduces both deliverables byte for byte: True" in checks_b

    put("mods-rw/" + RP + "V8F_SCIENTIFIC_REPORT.md", md)
    put("mods-rw/" + RP + "v8f_report.html", html)
    staged = [("patches/patch_v32.py", rd(HERE + "patch_v32.py")), ("patches/audit_v32.py", rd(HERE + "audit_v32.py")),
              ("patches/audit_reviewed.py", rd(HERE + "audit_reviewed.py")), ("patches/checks_v32.py", rd(HERE + "checks_v32.py")),
              ("patches/render_check.js", rd(HERE + "render_check.js")), ("patches/v32_NUMBERS_AUDIT.md", audit_b), ("patches/v32_checks.txt", checks_b),
              ("patches/mk_hold_report_v32.py", rd(HERE + "mk_hold_report_v32.py"))]
    for rel, b in staged:
        assert blob(RP + rel) is None and b"\r" not in b, rel
        put("stage/" + RP + rel, b)

MSG = ("Report v32: 7.24 SPEEDREAD (NOT IN INPUT: a ridge or an MLP read-out of T's five speed cells leaves ADE6 where it was, -0.0097 / +0.0022 m against a bar of -0.1207 m),"
       " 7.25 DAGGER2 (HARMS: the crawl rows did not move), 7.26 DAG-ORC (HARMS, arm A: R-ADE fails at s0 with ADE6 A - DAG-OACC +0.1434 m and R-T0 fails at both seeds,"
       " +0.1106 / +0.0722 m against +0.0604 m; the eight crawl cells are separated-better but reach 8.8 to 33.8 percent of their bars, which the replicate arm sets; A not adopted,"
       " B needs the PI's go; box 2 used 4.93 h by the logs against the ~3 h asked), the abstract, 8.1 (items 12-14), 8.3 and 8.4 (rows 1w-1y)."
       " Tables copied line for line from the three RESULT files. The T0 four families are NOT RUN and M-LAUNCH, the anti-echo block and M2 / L2 on the launch subset are NOT BUILT."
       " Numbers audit (with a JSON cross-check of the decisive DAG-ORC numbers), checks and patches are in patches/.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
for w in FORBID:
    assert w not in MSG.lower(), w
put("msg.txt", MSG.encode("utf-8"))

# ---- last checks on everything staged
bad, n = [], 0
for dp, _, fs in os.walk(H):
    for f in fs:
        p = os.path.join(dp, f)
        b = rd(p)
        n += 1
        if HFTOK.search(b):
            bad.append(("hf_", p))
        if p.endswith((".md", ".py", ".sh", ".html", ".js", ".txt")) and b"\r" in b:
            bad.append(("CR", p))
        if p.endswith((".pt", ".npz", ".npy", ".pyc")):
            bad.append(("binary", p))
assert not bad, bad
assert n == 11, n  # 2 report files, 8 patches files, msg.txt
print("hold written", H, n, "files;", len(added), "added lines; tip", FULL)
