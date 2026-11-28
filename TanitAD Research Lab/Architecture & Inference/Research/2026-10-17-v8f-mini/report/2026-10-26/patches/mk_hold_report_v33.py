"""Hold: report v33 (7.27 SPEEDSUB PASS, 7.28 SPEEDSUB-T1 NULL, 7.29 LAUNCH-INTENT HARMS with M-LAUNCH NOT MOVED, the abstract and 8.1, 8.3, 8.4 rows 1z-1ab) + its patch, fill, checks and numbers audit.
python mk_hold_report_v33.py <live tip> [--out <hold dir>] [--regen]

It rebuilds both report files with patch_v33.py (and the fill that mk_fill_v33.py generates from the repository blobs) on <live tip> (blobs only, never a working tree) and stages them
as mods-rw at their repo paths (TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/report/2026-10-26/). The patch, the fill generator and the fill (patch_v33.py,
mk_fill_v33.py, FILL.json), the audit (audit_v33.py, audit_reviewed_v33.py, v33_NUMBERS_AUDIT.md), the checks (checks_v33.py, render_check_v33.js, v33_checks.txt) and this builder go into stage/
under report/2026-10-26/patches/ as new files (the v32 files of that folder keep their names; nothing of v32 is overwritten). msg.txt ends with the Co-Authored-By line.
The Master Mind lands the hold; this script only writes it, and writes nowhere else. It does not run the lander and does not touch git.

The audit and checks files are staged as the last run of audit_v33.py / checks_v33.py in this folder wrote them. The builder verifies that they were produced for the very bytes it stages
(the sha256 that v33_checks.txt and v33_NUMBERS_AUDIT.md record) and refuses otherwise; it also requires that FILL.json equals what mk_fill_v33.py generates on <live tip>. If the live tip moved
since that run, pass --regen: the builder then regenerates the fill, re-runs audit_v33.py and checks_v33.py on <live tip> against its own rebuild, in a scratch copy of this folder, and stages those
outputs (needs node and the local marked, as checks_v33.py does; a stale line in audit_reviewed_v33.py stops it loudly).
The default hold dir is C:/Users/Admin/qland/hold_report_v33/ (the builder refuses an existing dir)."""
import difflib
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
args = sys.argv[1:]
assert args and not args[0].startswith("--"), "usage: mk_hold_report_v33.py <live tip> [--out <hold dir>] [--regen]"
TIP = args[0]
HOLD = "C:/Users/Admin/qland/hold_report_v33/"
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
SCRIPTS = ("patch_v33.py", "mk_fill_v33.py", "audit_v33.py", "audit_reviewed_v33.py", "checks_v33.py", "render_check_v33.js")


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
assert tpl is not None and old_md is not None and b"**Status:** research report, v32 of 2026-10-28" in old_md, "the live tip does not carry the v32 report this patch builds on"
for n in ("patch_v33.py", "mk_fill_v33.py", "FILL.json", "audit_v33.py", "audit_reviewed_v33.py", "checks_v33.py", "render_check_v33.js", "v33_NUMBERS_AUDIT.md", "v33_checks.txt", "mk_hold_report_v33.py"):
    assert blob(RP + "patches/" + n) is None, ("a v33 patches file is already at the tip", n)

# ---- rebuild on the live tip, in a scratch folder next to this script (removed at the end)
with tempfile.TemporaryDirectory(prefix="_v33hold_", dir=HERE) as WK:
    WK = WK.replace("\\", "/") + "/"
    # the fill is generated from the repository blobs; the one checked in this folder must be the same bytes unless --regen
    print(run([sys.executable, HERE + "mk_fill_v33.py", TIP, WK + "FILL.json"]).strip()[:160])
    fill_b = rd(WK + "FILL.json")
    if not REGEN:
        assert fill_b == rd(HERE + "FILL.json"), "FILL.json of this folder is not what mk_fill_v33.py generates on the live tip: pass --regen"
    print(run([sys.executable, HERE + "patch_v33.py", TIP, WK + "out", "--fill", WK + "FILL.json"]).strip())
    md, html = rd(WK + "out/V8F_SCIENTIFIC_REPORT.md"), rd(WK + "out/v8f_report.html")
    assert b"\r" not in md and b"\r" not in html and ("results through repository tip `%s`" % FULL[:8]).encode() in md, "status line does not name the live tip"
    assert html == tpl.replace(b"%%MD%%", md), "html is not the template filled with the md"
    assert b"@@LI_RESULT_" not in md and b"@@LI_RESULT_" not in html and b"DAGORC_RESULT" not in md
    added = [l for l in difflib.ndiff(old_md.decode("utf-8").splitlines(), md.decode("utf-8").splitlines()) if l.startswith("+ ")]
    for w in FORBID:
        hits = [l for l in added if re.search(r"\b" + w + r"\b", l.lower())]
        assert not hits, (w, hits[:2])
    here_md = HERE + "out/V8F_SCIENTIFIC_REPORT.md"
    same_as_here = os.path.exists(here_md) and rd(here_md) == md and rd(HERE + "out/v8f_report.html") == html
    print("rebuild on %s: %d bytes md, %d bytes html; identical to the out/ checked in this folder: %s" % (FULL[:8], len(md), len(html), same_as_here))

    if REGEN:
        SC = WK + "scripts/"
        os.makedirs(SC)
        for n in SCRIPTS:
            shutil.copyfile(HERE + n, SC + n)
        shutil.copyfile(WK + "FILL.json", SC + "FILL.json")
        aud, chk = WK + "v33_NUMBERS_AUDIT.md", WK + "v33_checks.txt"
        print(run([sys.executable, SC + "audit_v33.py", TIP, WK + "out", aud]).strip().splitlines()[-1])
        print(run([sys.executable, SC + "checks_v33.py", TIP, WK + "out", chk, aud]).strip().splitlines()[-1])
        audit_b, checks_b = rd(aud), rd(chk)
    else:
        audit_b, checks_b = rd(HERE + "v33_NUMBERS_AUDIT.md"), rd(HERE + "v33_checks.txt")
    msha, hsha = sha(md), sha(html)
    assert (b"V8F_SCIENTIFIC_REPORT.md " + msha.encode()) in checks_b and (b"v8f_report.html " + hsha.encode()) in checks_b, \
        "v33_checks.txt was produced for other bytes than the rebuild: re-run checks_v33.py, or pass --regen"
    assert (b"`V8F_SCIENTIFIC_REPORT.md` " + msha.encode()) in audit_b, "v33_NUMBERS_AUDIT.md was produced for other bytes than the rebuild: re-run audit_v33.py, or pass --regen"
    assert f"source tip {FULL[:8]} = {FULL}".encode() in checks_b, "v33_checks.txt names another source tip"
    assert b"ZZSUPERSET-OKZZ" in checks_b and b"ZZDEMOTE-OKZZ" in checks_b and b"mode of the checked deliverable: FINAL (no slot marker)" in checks_b
    assert b"== patch_v33.py re-run on the v32 blobs and FILL.json reproduces both deliverables byte for byte: True" in checks_b
    assert b"== mk_fill_v33.py re-run on the tip regenerates FILL.json byte for byte" in checks_b and b"the fill is generated from the repository blobs, not typed): True" in checks_b

    put("mods-rw/" + RP + "V8F_SCIENTIFIC_REPORT.md", md)
    put("mods-rw/" + RP + "v8f_report.html", html)
    staged = [("patches/patch_v33.py", rd(HERE + "patch_v33.py")), ("patches/mk_fill_v33.py", rd(HERE + "mk_fill_v33.py")), ("patches/FILL.json", fill_b),
              ("patches/audit_v33.py", rd(HERE + "audit_v33.py")), ("patches/audit_reviewed_v33.py", rd(HERE + "audit_reviewed_v33.py")),
              ("patches/checks_v33.py", rd(HERE + "checks_v33.py")), ("patches/render_check_v33.js", rd(HERE + "render_check_v33.js")),
              ("patches/v33_NUMBERS_AUDIT.md", audit_b), ("patches/v33_checks.txt", checks_b), ("patches/mk_hold_report_v33.py", rd(HERE + "mk_hold_report_v33.py"))]
    for rel, b in staged:
        assert blob(RP + rel) is None and b"\r" not in b, rel
        put("stage/" + RP + rel, b)

MSG = ("Report v33: 7.27 SPEEDSUB (PASS, a screening result on the declared open channel: s1 clears the bar by 0.0083 m only), 7.28 SPEEDSUB-T1 (NULL, label NULL-ONE: the causal read-out hits"
       " the bar at 1 of 4 runs, REP s0, a slow-T training draw), 7.29 LAUNCH-INTENT (HARMS, arm A, W 1.58926, not adopted: all 206 reader controls PASS, R-MOVE and R-CRAWL are false at both seeds,"
       " R-ADE fails at s0 in the replicate pair and at s1 in the main pair, and the separated ADE6 cells flip sign between the two same-seed pairs; M-LAUNCH reads NOT MOVED over 49 episodes /"
       " 327 launch windows, which routes section 11 to LAUNCH-INTENT-HI), the abstract, 8.1 (items 15-17), 8.3 and 8.4 (rows 1z-1ab). Tables copied line for line from the three RESULT files and the"
       " PREREG; the launch facts are MEASURED from the banked box-2 chain log. The T0 four families are NOT RUN; M-LAUNCH inside the reader, the anti-echo block and M2 / L2 on the launch subset are"
       " NOT BUILT. Committed next: B is registered after HARMS with its own PI go; HI is the section 11 branch, also with its own PI go; the PI is being asked which, if either, box 2 runs."
       " Numbers audit (with a JSON and log cross-check), the fill, checks and patches are in patches/.\n\n"
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
        if p.endswith((".md", ".py", ".sh", ".html", ".js", ".txt", ".json")) and b"\r" in b:
            bad.append(("CR", p))
        if p.endswith((".pt", ".npz", ".npy", ".pyc")):
            bad.append(("binary", p))
assert not bad, bad
assert n == 13, n  # 2 report files, 10 patches files, msg.txt
print("hold written", H, n, "files;", len(added), "added lines; tip", FULL)
