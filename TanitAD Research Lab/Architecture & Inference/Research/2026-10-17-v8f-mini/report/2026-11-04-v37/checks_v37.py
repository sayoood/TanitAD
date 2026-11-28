"""Independent checks of report v37 (the patch's own assertions have already passed to produce out/). Writes the checks file.
python -I checks_v37.py tip [out_dir [checks_path]]   (git, python; no network)"""
import difflib, hashlib, os, re, subprocess, sys, tempfile
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TIP = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"
OUT = (sys.argv[2] if len(sys.argv) > 2 else HERE + "out").rstrip("/\\") + "/"
CP = sys.argv[3] if len(sys.argv) > 3 else HERE + "v37_checks.txt"
M = "C:/Users/Admin/tanitad-push"
R = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
L, FAILS = [], []


def say(x=""):
    L.append(x)


def show(p, b=False):
    r = subprocess.run(["git", "-C", M, "-c", "safe.directory=*", "show", f"{TIP}:{p}"], capture_output=True)
    assert r.returncode == 0, p
    return r.stdout if b else r.stdout.decode("utf-8")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def chk(name, cond, extra=""):
    say(("PASS " if cond else "FAIL ") + name + ((" " + extra) if extra else ""))
    if not cond:
        FAILS.append(name)


norm = lambda t: re.sub(r"\s+", " ", t)
v36 = show(R + "report/2026-11-04-v36/V8F_SCIENTIFIC_REPORT.md")
s = open(OUT + "V8F_SCIENTIFIC_REPORT.md", encoding="utf-8", newline="").read()
h = open(OUT + "v8f_report.html", encoding="utf-8", newline="").read()
say("# report v37 checks, tip " + TIP)
say()
chk("v36 inputs are the landed deliverables (sha256)", sha(show(R + "report/2026-11-04-v36/V8F_SCIENTIFIC_REPORT.md", True)) == "1a1fef350809732fc8a7a787d34274b34fd6e6a03b3425387ae7a2aacaff327f"
    and sha(show(R + "report/2026-11-04-v36/v8f_report.html", True)) == "1cd74716c52211bdad8f5e97476b10752490197042505377baed45bad6b63ce7")
chk("html = page template with the v37 md", h == show(R + "report/2026-10-26/page_template_v30.html").replace("%%MD%%", s))
chk("LF only, no BOM", "\r" not in s and "\ufeff" not in s and "\r" not in h)
d = list(difflib.ndiff(v36.split("\n"), s.split("\n")))
rem = [l[2:] for l in d if l.startswith("- ")]
KEYS = ("**Status:** research report, v36 of 2026-11-04", "* **The M1 excess enters through the ego-history input, and the speed level is what the model lacks",
        "the training-side lever they point to (the recovery fine-tune of O) is not read in this version", "is read in a later section; no result of it is used here",
        "It is read in a later section of this report", "The lever the cards point to, the recovery fine-tune of O, is read in a later section", "| 1ao | **HISTSWAP-LF**",
        "13. **v35 reads and their limits.**", "14. **v36 reads and their limits.**", "(6) HISTSWAP and HISTSWAP-LF are read in §7.35; FRAMEFREEZE")
unexp = [l[:80] for l in rem if not any(k in l for k in KEYS)]
chk("removed lines from v36: only the edited lines (status, abstract 7.35 bullet and next, 7.35 pointers, the Update (v36) bullet, row 1ao, items 13 and 14, the 7.34 pointer)",
    not unexp and len(rem) <= 12, "(%d removed; unexpected %s)" % (len(rem), unexp[:2]))
i735, i7354, i736, i8 = s.index("### 7.35 "), s.index("**7.35.4 "), s.index("### 7.36 "), s.index("## 8. Discussion")
chk("order: 7.35 < 7.35.4 < 7.36 < section 8, one 7.36", i735 < i7354 < i736 < i8 and s.count("### 7.36 ") == 1)
sec = s[i7354:i8]
srcs = {k: show(R + p) for k, p in (("RF", "raw/2026-11-03-recover-read-result/RESULTS.md"), ("RT", "raw/2026-11-04-recover-next-rtl/RESULTS.md"),
                                    ("RP", "raw/2026-11-04-recover-next-rp01/RESULTS.md"), ("RA", "raw/2026-11-04-recover-next-rapr/RESULTS.md"),
                                    ("LP", "raw/2026-11-04-levelprobe/RESULTS.md"), ("LP2", "raw/2026-11-04-levelprobe/RESULTS2.md"))}
lines = set(x for v in srcs.values() for x in v.split("\n"))
tl = [l for l in sec.split("\n") if l.startswith("|")]
chk("every table line of 7.35.4 and 7.36 is a line of a source RESULTS file", tl and all(l in lines for l in tl), "(%d table lines)" % len(tl))
cards = [show(R + p) for p in ("raw/2026-11-02-recover-ft/PREREG_RECOVER_FT_FROZEN.md", "raw/2026-11-02-recover-read/PREREG_COMPOSITE_FROZEN.md", "raw/2026-11-03-recover-next/PREREG_RECOVER_NEXT_FROZEN.md",
                                  "raw/2026-11-04-recover-next-rtl/PREREG_RECOVER_NEXT_ADDENDUM_RTLRP01.md", "raw/2026-11-04-recover-next-rapr/PREREG_RECOVER_NEXT_ADDENDUM2_RAPR.md",
                                  "raw/2026-11-04-levelprobe/PREREG_LEVELPROBE.md", "raw/2026-11-04-levelprobe/PREREG_LEVELPROBE2.md")]
corp = norm(" ".join(list(srcs.values()) + cards + [show("Project Steering/GOALS_AND_CLAIMS.md"), v36]))
qs = [q.strip('"') for q in re.findall(r"(?<![*\w])\*(?!\*)([^*\n]{25,}?)(?<!\*)\*(?![*\w])", sec)]
chk("italic quotations of 25+ characters verbatim in a source (RESULTS, cards, register, v36)", qs and all(norm(q) in corp for q in qs), "(%d quotations)" % len(qs))
fams = [len(re.findall(f, sec)) for f in ("LONGITUDINAL|DK-VIOL|DK-HW|along-track|speed MAE|accel MAE", "LATERAL|curvature|yaw-rate|heading|cross-track", "TACTICAL|TAC-LAT|TAC-LON", "STRATEGIC|S-POS")]
chk("each of the four metric families named in 7.35.4 + 7.36 at least 3 times", all(c >= 3 for c in fams), str(fams))
chk("T1-kin and the estimator stated in 7.35.4 and in 7.36", all("T1-kin" in x and "B = 2,000" in x for x in (s[i7354:i736], s[i736:i8])))
chk("no 'closed loop' or driving wording in the added sections", not re.search(r"closed[- ]loop|driving performance|\bdrives\b", sec.lower()))
chk("no banned word in the added lines", not any(re.search(r"\b(" + "|".join(w[::-1] for w in ("diov", "dedesrepus", "detcarter", "ylremrof", "regnol on", "noitcerroc", "nwardhtiw")) + r")\b", l[2:], re.I) for l in d if l.startswith("+ ")))
i83 = s.index("### 8.3 ")
chk("8.3 has Update (v37) before Update (v36)", s.index("**Update (v37).**", i83) < s.index("**Update (v36).**", i83))
i81, i82 = s.index("### 8.1 "), s.index("### 8.2 ")
chk("8.1 ends with item 24", re.findall(r"^(\d+)\. \*\*", s[i81:i82], re.M)[-1] == "24")
i9, i10 = s.index("## 9. "), s.index("## 10. Reproducibility appendix")
chk("section 9 ends with item 15 (v37 reads and their limits)", re.findall(r"^(\d+)\. \*\*", s[i9:i10], re.M)[-1] == "15" and "15. **v37 reads and their limits.**" in s[i9:i10])
chk("8.4 has rows 1ap to 1as, once each, before row 2", all(s.count("| %s | " % r) == 1 for r in ("1ap", "1aq", "1ar", "1as")) and s.index("| 1as | ") < s.index("| 2 | **LoRA on DINOv3**"))
line = s.split("\n")[2]
chk("status line names v37 of 2026-11-04, the tip, 7.35 and 7.36; no 2026-11-05 anywhere", "v37 of 2026-11-04" in line and "`%s`" % TIP[:8] in line and "§7.36" in line and "2026-11-05" not in s)
chk("no stale pointer about the recovery reads", "is read in a later section" not in s and "are not read in v36" not in s and "not read in this version, and no claim about their states" not in s)
with tempfile.TemporaryDirectory() as td:
    r = subprocess.run([sys.executable, "-I", HERE + "patch_v37.py", TIP, td], capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1"))
    same = r.returncode == 0 and all(open(os.path.join(td, f), "rb").read() == open(OUT + f, "rb").read() for f in ("V8F_SCIENTIFIC_REPORT.md", "v8f_report.html"))
    chk("patch_v37.py re-run reproduces both deliverables byte for byte", same, r.stdout.strip()[-120:] + r.stderr.strip()[-200:])
say()
say("RESULT: %s (%d checks, %d failed)" % ("PASS" if not FAILS else "FAIL", sum(1 for x in L if x[:5] in ("PASS ", "FAIL ")), len(FAILS)))
open(CP, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print(L[-1])
sys.exit(1 if FAILS else 0)
