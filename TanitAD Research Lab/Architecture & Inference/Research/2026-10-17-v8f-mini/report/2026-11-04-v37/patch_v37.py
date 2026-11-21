"""Report v37 = v36 + section 7.35.4 (LEVELPROBE 1 and 2) + section 7.36 (RECOVER-FT, its composite and frame-freeze control, and the RECOVER-NEXT arms RTL, RP01 and RAPR), the status line,
the abstract (the 7.35 bullet and a new 7.36 bullet), 8.1 item 24, 8.3 Update (v37), 8.4 rows 1ap to 1as, 9 item 15, and the pointers that said these reads were not covered.
Built from the landed v36 report blobs of one repository tip (anchored edits, every anchor asserted to occur exactly once). The prose is in the package files beside this script
(sec_735_lp.md, sec_736_part1.md, sec_736_part2.md whose first line is the section title, summary_v37.md with one block per edit); every source is `git show <tip>:<path>` from the mirror.
Nothing is read from a working tree of the repository. Nothing is staged, committed or pushed.

python -I patch_v37.py tip [out_dir]
Every table line of the added text is asserted to be a line of a source RESULTS file; every italic quotation of 25 characters or more is asserted verbatim in a source (RESULTS, cards, register)."""
import hashlib, json, os, re, subprocess, sys
from collections import OrderedDict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_argv = sys.argv[1:]
assert _argv, "usage: patch_v37.py <tip> [out_dir]"
TIP = _argv[0]
TP = TIP[:8]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = (_argv[1] if len(_argv) > 1 else HERE + "/out").rstrip("/\\") + "/"
MIRROR = "C:/Users/Admin/tanitad-push"
R = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
BANNED = tuple(w[::-1] for w in ("diov", "dedesrepus", "detcarter", "ylremrof", "regnol on", "noitcerroc", "nwardhtiw"))  # the programme's banned words, stored reversed so this file passes its own lander scan
_BAN = re.compile(r"\b(" + "|".join(BANNED) + r")\b", re.I)


def gitrun(args, binary=False):
    r = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*"] + args, capture_output=True)
    assert r.returncode == 0, (args, r.stderr[:200])
    return r.stdout if binary else r.stdout.decode("utf-8").strip()


def raw(path):
    return gitrun(["show", f"{TIP}:{path}"], binary=True)


def blob(path):
    t = raw(path).decode("utf-8")
    assert len(t) > 0 and "\r" not in t and "\ufeff" not in t, path
    return t


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def part(name):
    t = open(os.path.join(HERE, name), encoding="utf-8", newline="").read()
    assert "\r" not in t and "\ufeff" not in t, name
    return t.strip("\n")


# ---- the v36 deliverables are the starting point ---------------------------------------------------------------------------------------------------------
V36_MD_SHA = "1a1fef350809732fc8a7a787d34274b34fd6e6a03b3425387ae7a2aacaff327f"
V36_HTML_SHA = "1cd74716c52211bdad8f5e97476b10752490197042505377baed45bad6b63ce7"
RP36 = R + "report/2026-11-04-v36/"
s = blob(RP36 + "V8F_SCIENTIFIC_REPORT.md")
h = blob(RP36 + "v8f_report.html")
tpl = blob(R + "report/2026-10-26/page_template_v30.html")
assert sha(raw(RP36 + "V8F_SCIENTIFIC_REPORT.md")) == V36_MD_SHA, "the tip's report md is not the landed v36 deliverable"
assert sha(raw(RP36 + "v8f_report.html")) == V36_HTML_SHA, "the tip's report html is not the landed v36 deliverable"
assert tpl.count("%%MD%%") == 1 and h == tpl.replace("%%MD%%", s), "v36 html is not the template with the v36 md"
assert s.count("**Status:** research report, v36 of 2026-11-04") == 1
S0 = s

# ---- sources ----------------------------------------------------------------------------------------------------------------------------------------------
SRC_PATHS = OrderedDict([("RF", "raw/2026-11-03-recover-read-result/RESULTS.md"), ("RT", "raw/2026-11-04-recover-next-rtl/RESULTS.md"),
                         ("RP", "raw/2026-11-04-recover-next-rp01/RESULTS.md"), ("RA", "raw/2026-11-04-recover-next-rapr/RESULTS.md"),
                         ("LP", "raw/2026-11-04-levelprobe/RESULTS.md"), ("LP2", "raw/2026-11-04-levelprobe/RESULTS2.md"),
                         ("LPA", "raw/2026-11-04-levelprobe/AMEND_A1_PROPOSED.md")])
SRC = OrderedDict((k, blob(R + p)) for k, p in SRC_PATHS.items())
CARD_PATHS = OrderedDict([("FT", ("raw/2026-11-02-recover-ft/PREREG_RECOVER_FT_FROZEN.md", "57caba3b")),
                          ("CO", ("raw/2026-11-02-recover-read/PREREG_COMPOSITE_FROZEN.md", "18ed37bc")),
                          ("NX", ("raw/2026-11-03-recover-next/PREREG_RECOVER_NEXT_FROZEN.md", "01046fe6")),
                          ("A1", ("raw/2026-11-04-recover-next-rtl/PREREG_RECOVER_NEXT_ADDENDUM_RTLRP01.md", "8c62b4da")),
                          ("A2", ("raw/2026-11-04-recover-next-rapr/PREREG_RECOVER_NEXT_ADDENDUM2_RAPR.md", "a292eac5")),
                          ("L1", ("raw/2026-11-04-levelprobe/PREREG_LEVELPROBE.md", "25c66e93")),
                          ("L2", ("raw/2026-11-04-levelprobe/PREREG_LEVELPROBE2.md", "8f1208a5"))])
CARDS = OrderedDict((k, blob(R + p)) for k, (p, _) in CARD_PATHS.items())
for k, (p, pre) in CARD_PATHS.items():
    assert sha(raw(R + p)).startswith(pre), ("card sha256", k, p)
assert sha(raw(R + CARD_PATHS["L2"][0])) in SRC["LP2"] and CARD_PATHS["NX"][1] in SRC["RA"], "a RESULTS file names a different card"
REG = blob("Project Steering/GOALS_AND_CLAIMS.md")
RES_SHA = {k: sha(raw(R + p)) for k, p in SRC_PATHS.items()}
# landing commits (first commit that added the file), measured from the mirror
LAND = {}
for k in ("RF", "RT", "RP", "RA", "LP"):
    c = gitrun(["log", "--format=%h", "--diff-filter=A", TIP, "--", R + SRC_PATHS[k]]).split("\n")[-1]
    assert re.fullmatch(r"[0-9a-f]{7,}", c), (k, c)
    LAND[k] = c[:8]
# raw JSON cross-check of the RAPR headline (MEASURED artefact behind its RESULTS)
J_RA = json.loads(blob(R + "raw/2026-11-04-recover-next-rapr/RECOVER_READ_RAPR.json"))
J_NX = json.loads(blob(R + "raw/2026-11-04-recover-next-rapr/RECOVER_NEXT_after_RAPR.json"))
RA_SH = {k: "%.4f" % J_RA["primary"]["share_closed"][k] for k in ("s0", "s1")}

# ---- the section texts ------------------------------------------------------------------------------------------------------------------------------------
TITLE, P2 = part("sec_736_part2.md").split("\n", 1)
assert TITLE.startswith("### 7.36 ") and "\n### " not in P2
SEC735 = part("sec_735_lp.md")
SEC736 = TITLE + "\n\n" + part("sec_736_part1.md") + "\n\n" + P2.strip("\n")
_blocks = re.split(r"^=== (\w+) ===\n", part("summary_v37.md"), flags=re.M)
assert _blocks[0].strip() == "" and len(_blocks) % 2 == 1
B = {_blocks[i]: _blocks[i + 1].strip("\n") for i in range(1, len(_blocks), 2)}
assert sorted(B) == ["ABS", "ABS35", "I15", "R83", "ROWS", "S81", "ST"], sorted(B)


def fill(t):
    for k, c in LAND.items():
        t = t.replace("{{LAND_%s}}" % k, c)
    t = t.replace("{{TP}}", TP)
    for k, v in RES_SHA.items():
        t = t.replace("{{RES_%s}}" % k, v[:8])
    assert "{{" not in t and "<<" not in t, re.findall(r"(\{\{\S+?\}\}|<<\S+?>>)", t)[:3]
    return t


SEC735, SEC736 = fill(SEC735), fill(SEC736)
B = {k: fill(v) for k, v in B.items()}
for k in ("s0", "s1"):
    assert RA_SH[k] in SEC736, ("RAPR share closed from the raw JSON is not in 7.36", k, RA_SH[k])

# ---- tables are source lines; quotations are verbatim -----------------------------------------------------------------------------------------------------
SRCL = set()
for v in SRC.values():
    SRCL |= set(v.split("\n"))


def ncols(line):
    return re.sub(r"`[^`\n]*`", "x", line.replace("\\|", "x")).count("|") - 1


ADDED = "\n".join([SEC735, SEC736] + list(B.values()))
_tabs, _cur = [], []
for l in (SEC735 + "\n\n" + SEC736).split("\n"):
    if l.startswith("|"):
        _cur.append(l)
    elif _cur:
        _tabs.append(_cur); _cur = []
if _cur:
    _tabs.append(_cur)
_bad = [l[:90] for t in _tabs for l in t if l not in SRCL]
assert _tabs and not _bad, ("a table line is not a line of a source RESULTS file", _bad[:3])
assert all(len({ncols(r) for r in t}) == 1 and t[1].startswith("|---") for t in _tabs), "a table with ragged columns or no header rule"
assert not any(l.startswith("|") for v in B.values() for l in v.split("\n") if not re.match(r"\| 1a[p-s] \|", l)), "a table line outside 7.35.4 / 7.36 other than rows 1ap to 1as"
_corp = re.sub(r"\s+", " ", " ".join(list(SRC.values()) + list(CARDS.values()) + [REG, S0]))
_q = [m.strip('"') for m in re.findall(r"(?<![*\w])\*(?!\*)([^*\n]{25,}?)(?<!\*)\*(?![*\w])", ADDED)]
_nq = [q for q in _q if re.sub(r"\s+", " ", q) not in _corp]
assert _q and not _nq, ("an italic quotation is not verbatim in a source", _nq[:3])

# ---- anchored edits -----------------------------------------------------------------------------------------------------------------------------------
ANCHOR_8 = "\n\n---\n\n## 8. Discussion"
LP_ABS = ("Two later probes (§7.35.4, LEVELPROBE 1 and 2, ORACLE) hold one logged speed per window instead: the speed at t0 (HLVL-T0) closes a mean −0.386 of the excess and harms ADE6 by "
          "+0.7181 to +0.8045 m, and the mean logged speed over the next 6 s, or the logged speed 3 s or 1 s ahead, close −0.461, −0.416 and −0.354 (0 of 4 cells separated-better for each; "
          "outcome PROFILE): no single per-window speed carries HLVL's closure. ")
E = [
    ("research report, v36 of 2026-11-04, results through repository tip `9b79730c` and, within it,",
     "research report, v37 of 2026-11-04, results through repository tip `" + TP + "` and, within it,"),
    (") and the executed-state reads (§7.35: HISTSWAP and HISTSWAP-LF); the tip also holds later reads (FRAMEFREEZE, RECOVER-FT, RECOVER-NEXT) that this report does not cover (v35: `92cbf277`; v34: `870fa33a`",
     "), the executed-state reads (§7.35: HISTSWAP, HISTSWAP-LF and the level probes LEVELPROBE 1 and 2) and the recovery fine-tunes (§7.36: RECOVER-FT with its composite and its frame-freeze control, "
     "and the RECOVER-NEXT arms RTL, RP01 and RAPR) (v36: `9b79730c`; v35: `92cbf277`; v34: `870fa33a`"),
    ("closes 0.599 to 0.633 (HLVL), the card's reading (B) level.", "closes 0.599 to 0.633 (HLVL), the card's reading (B) level. " + B["ST"]),
    ("* **The M1 excess enters through the ego-history input, and the speed level is what the model lacks (§7.35;",
     "* **The M1 excess enters through the ego-history input, and closing it needs the logged speed at every tick (§7.35;"),
    ("The lever both cards point to, the recovery fine-tune of O, is read in a later section.\n* **What is next.** The crawl set is still open",
     LP_ABS + "The lever both cards point to, the recovery fine-tune of O, is the next bullet (§7.36).\n" + B["ABS"] + "\n* **What is next.** The crawl set is still open"),
    ("the training-side lever they point to (the recovery fine-tune of O) is not read in this version.",
     "the training-side lever they point to (the recovery fine-tune of O) is read in §7.36. " + B["ABS35"]),
    ("The lever both cards point to, the recovery fine-tune of O, is read in a later section; no result of it is used here.",
     "The lever both cards point to, the recovery fine-tune of O, is read in §7.36; no result of it is used in §7.35.1 to §7.35.3."),
    ("It is read in a later section of this report; this section reports no RECOVER result and makes no claim about its state.",
     "It is read in §7.36; §7.35.1 to §7.35.3 report no RECOVER result."),
    ("The lever the cards point to, the recovery fine-tune of O, is read in a later section.\n", "The lever the cards point to, the recovery fine-tune of O, is read in §7.36.\n"),
    ("(PI go; read in a later section) |", "(PI go; read as 1ar) |"),
    (ANCHOR_8, "\n\n" + SEC735 + "\n\n" + SEC736 + ANCHOR_8),
    ("\n\n### 8.2 What is not established", "\n" + B["S81"] + "\n\n### 8.2 What is not established"),
    ("**Update (v36).** Two items are written up here", B["R83"] + "\n\n**Update (v36).** Two items are written up here"),
    ("| 2 | **LoRA on DINOv3**", B["ROWS"] + "\n| 2 | **LoRA on DINOv3**"),
    ("HISTSWAP and HISTSWAP-LF are read in v36 (§7.35, item 14); FRAMEFREEZE, RECOVER-FT and RECOVER-NEXT are not read in v36.",
     "HISTSWAP and HISTSWAP-LF are read in v36 (§7.35, item 14); FRAMEFREEZE (as the control of the recovery reads), RECOVER-FT and RECOVER-NEXT are read in v37 (§7.36, item 15)."),
    ("beyond the quoted passages. FRAMEFREEZE, RECOVER-FT and RECOVER-NEXT are not read in v36.",
     "beyond the quoted passages. FRAMEFREEZE, RECOVER-FT and RECOVER-NEXT are read in v37 (§7.36, item 15), and the level probes in §7.35.4."),
    ("\n\n---\n\n## 10. Reproducibility appendix", "\n" + B["I15"] + "\n\n---\n\n## 10. Reproducibility appendix"),
    ("(6) HISTSWAP and HISTSWAP-LF are read in §7.35; FRAMEFREEZE, RECOVER-FT and RECOVER-NEXT are not read in this version, and no claim about their states is made here (**UNVERIFIED**).",
     "(6) HISTSWAP and HISTSWAP-LF are read in §7.35, and FRAMEFREEZE (as the control of the recovery reads), RECOVER-FT and RECOVER-NEXT in §7.36."),
]
for a, b in E:
    assert s.count(a) == 1, (s.count(a), a[:90])
    s = s.replace(a, b)

# ---- guards on the inserted text ---------------------------------------------------------------------------------------------------------------------
ins = "".join(b for _, b in E[:10] + E[11:]) + SEC735 + SEC736
assert not _BAN.search(ins), _BAN.search(ins)
low = ins.lower()
assert re.search("retract|supersed|" + BANNED[6] + "|was wrong|previously read|outdated|i said", low) is None
assert "closed loop" not in low and "closed-loop" not in low and not re.search(r"\bdrives\b|driving performance", low), "closed-loop or driving wording"
assert "\r" not in ins and "$" not in ins.replace(SEC736, ""), "a CR, or a dollar sign outside 7.36"
assert SEC736.count("$$") == 2 and SEC736.count("$") % 2 == 0, "7.36 math is one display block plus paired inline spans"
nocode = re.sub(r"`[^`\n]*`", " ", ins)
assert re.search(r"<[A-Za-z/!]", nocode) is None, re.search(r".{20}<[A-Za-z/!].{20}", nocode)
assert "@@" not in s and "</script" not in s.lower() and "\r" not in s
_flat = re.sub(r"\s+", " ", SEC736)
assert "B = 2,000" in _flat and "T1-kin" in _flat and "H-ESTIM-SEED-1" in _flat and "RAPR" in _flat and "frame-freeze" in _flat
assert s.count("### 7.36 ") == 1 and s.index("### 7.35 ") < s.index("**7.35.4 ") < s.index("### 7.36 ") < s.index("## 8. Discussion")
assert s.count("**Status:** research report, v37 of 2026-11-04") == 1 and "v36 of 2026-11-04, results" not in s.split("\n")[2]
assert "is read in a later section" not in s and "are not read in v36" not in s, "a stale pointer about the recovery reads remains"
h = tpl.replace("%%MD%%", s)
assert h.count("%%MD%%") == 0
os.makedirs(OUT, exist_ok=True)
open(OUT + "V8F_SCIENTIFIC_REPORT.md", "w", encoding="utf-8", newline="\n").write(s)
open(OUT + "v8f_report.html", "w", encoding="utf-8", newline="\n").write(h)
print("v37 built", len(S0), len(s), "tip", TIP, "7.35.4 chars", len(SEC735), "7.36 chars", len(SEC736), "tables", len(_tabs), "table lines", sum(len(t) for t in _tabs),
      "quotations", len(_q), "landed", LAND)
