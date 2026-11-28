"""Report v34 pre-draft: 7.30 crawl diagnosis and the DAgger-bank pre-steps (zero training), 7.31 trigger probes 1 and 2, 7.32 SOUP (HARMS, not adopted), 7.33 fine-token probes 3 and 4
(marked slots only: the probes were still running), a new background section 5.4, related work (three paragraphs, library keys), the abstract, 8.1 (items 18-21), 8.3 (Update v34),
8.4 (rows 1ac-1af), 9 (item 12), 10 (one paragraph) and the version line. Built from the v33 report blobs of one repository tip, exactly as patch_v33.py built v33.
Nothing is read from a working tree. Nothing is staged, committed or pushed.

python patch_v34.py [tip [out_dir]] [--predraft] [--fill FT_FILL.json]
  tip         repository tip whose blobs are the inputs (default 7b1375964422351ba5f41a0b52da444fba8c5883); its first 8 hex digits go to the status line
  out_dir     folder for V8F_SCIENTIFIC_REPORT.md and v8f_report.html (default: out/ beside this script)
  --predraft  write the PRE-DRAFT: section 7.33 carries marked slots (the string at-at FT_RESULT_<NAME> at-at, one per slot, listed in SLOTS below); FT_FILL_TEMPLATE.json and
              FT_FILL_HELP.txt are written beside the report. Without this flag the patch writes a FINAL version only: it needs --fill FT_FILL.json (a JSON object with a non-empty text for every
              slot) and REFUSES (non-zero exit, nothing written) while a slot is missing, empty or unknown, or while a text still holds a slot marker. Both the md and the html are built
              by the same anchored edits and must agree (the html edited in place equals the template with the new md).

Inputs (all `git show <tip>:<path>` from the mirror, never the mirror's HEAD, never a working tree):
  the v33 V8F_SCIENTIFIC_REPORT.md, v8f_report.html and page_template_v30.html (asserted byte-identical to the v33 deliverables by sha256);
  crawl-diag RESULTS.md, bank-presteps regret_prestep.md / fitgap_m2b.md / recovery/RECOVERY_RESULT.md, trigger-probe and trigger-probe2 RESULTS.md, RESULT_V8F_SOUP.md (their tables are COPIED LINE FOR
  LINE by anchored extraction, each line asserted to be a line of the source and free of the banned words); the register rows of GOALS_AND_CLAIMS.md, the frozen-card sha files, PREREG_V8F_SOUP.md,
  SOUP_READ.json, the two 2026-10-28 literature reviews with their LIBRARY_KEYS.txt and TanitAD Research Lab/Library/library.json (every cited key and its PDF sha256 is checked).
Every insertion is anchored and its anchor is asserted to occur exactly once. Claims taken from the sources are asserted present in them (PHRASES)."""
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import OrderedDict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_argv = sys.argv[1:]
PREDRAFT = "--predraft" in _argv
FILLF = None
if "--fill" in _argv:
    _i = _argv.index("--fill")
    assert _i + 1 < len(_argv), "--fill needs a file"
    FILLF = _argv[_i + 1]
    del _argv[_i:_i + 2]
FT_PATHS = []
while "--ft-result" in _argv:
    _i = _argv.index("--ft-result")
    assert _i + 1 < len(_argv), "--ft-result needs a repo path"
    FT_PATHS.append(_argv[_i + 1])
    del _argv[_i:_i + 2]
_argv = [a for a in _argv if a != "--predraft"]
TIP = (_argv[0] if len(_argv) > 0 else "7b1375964422351ba5f41a0b52da444fba8c5883")
TP = TIP[:8]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = (_argv[1] if len(_argv) > 1 else HERE + "/out").rstrip("/\\") + "/"
MIRROR = "C:/Users/Admin/tanitad-push"
R = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
RP = R + "report/2026-10-26/"
RA = "TanitAD Research Lab/Architecture & Inference/Research/"
RD = "TanitAD Research Lab/Data Engineering/Research/"
BANNED = ("void", "superseded", "retracted", "formerly", "no longer", "correction", "withdrawn")

# ---- the outcome slots of 7.33 and its other sites: name -> what the fill text is ---------------------------------------------------------------------
SLOTS = OrderedDict([
    ("LINE", "status line (v34 adds ...): the clause that names the fine-token probes, with their outcome word, ending in a full stop"),
    ("ABSTRACT", "abstract: ONE bullet starting with '* **', the fine-token bullet (outcome word, the decisive numbers with tier, evidence class and estimator)"),
    ("TITLE733", "heading of 7.33 after 'Fine-token probes 3 and 4: ' (the outcome word and label), one line"),
    ("STATUS733", "7.33 first bullet: the text that follows the fixed lead '* **Status of this section (v34).**' (read date, RESULT or RESULTS file and landing commit, controls, outcome word)"),
    ("SETUP733", "7.33: the setup bullet (the PREREG and its commit, claim rows, token grids, populations, arms, the committed outcomes, frozen-before-data time, TIER and ESTIMATOR of the probes)"),
    ("OUTCOME733", "7.33: the outcome bullet(s) with the source's outcome table(s) copied line for line, the word, the committed next verbatim"),
    ("CONTROLS733", "7.33: the controls bullet (names and counts, each ran before its rows were read; the failed ones named)"),
    ("FAMILIES733", "7.33: the four families (LONGITUDINAL, LATERAL, TACTICAL, STRATEGIC), STRATEGIC not paused, per family what is read or NOT RUN / NOT BUILT / NOT COMPUTABLE with the reason"),
    ("NEXT733", "7.33: Next (Rule Zero): the committed next, every remaining lever named and what blocks or unblocks it"),
    ("PROVENANCE733", "7.33: provenance (files, sha256, commits, where the arrays live)"),
    ("R81", "8.1 item 21: the text after the fixed lead '21. ' (a bold claim, then the established / not established statement with tier and estimator)"),
    ("R83", "8.3 Update (v34): the text after the fixed lead '* ' of the fine-token bullet"),
    ("R84", "8.4 row 1af: the status cell, one line without a pipe character"),
])
FILL = {}
if not PREDRAFT:
    if not FILLF:
        sys.exit("patch_v34.py REFUSES to write a final version without --fill FT_FILL.json; every fine-token slot needs its text (" + ", ".join(SLOTS) + "). Use --predraft for the pre-draft.")
    if not os.path.exists(FILLF):
        sys.exit("patch_v34.py REFUSES: the fill file does not exist: " + FILLF)
    FILL = json.load(open(FILLF, encoding="utf-8"))
    _unk = [k for k in FILL if k not in SLOTS]
    _unf = [k for k in SLOTS if not (isinstance(FILL.get(k), str) and FILL[k].strip())]
    if _unk or _unf:
        sys.exit("patch_v34.py REFUSES to write a final version: unfilled slots %s, unknown keys %s" % (_unf, _unk))
    _left = [k for k, v in FILL.items() if "@@" in v]
    if _left:
        sys.exit("patch_v34.py REFUSES: the fill text of %s still holds an at-at marker" % _left)
USED = []


def site(name, pre):
    """The text at a slot site: the pre-draft sentence plus the visible marker, or the fill text of a final version."""
    assert name in SLOTS, name
    USED.append(name)
    if PREDRAFT:
        return pre + (" " if pre else "") + "@@FT_RESULT_" + name + "@@"
    return FILL[name].strip("\n")


def gitrun(args, binary=False):
    r = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*"] + args, capture_output=True)
    assert r.returncode == 0, (args, r.stderr[:200])
    return r.stdout if binary else r.stdout.decode("utf-8").strip()


def raw(path):
    return gitrun(["show", f"{TIP}:{path}"], binary=True)


def blob(path, strict=True):
    t = raw(path).decode("utf-8")
    assert len(t) > 0, path
    if strict:
        assert "\r" not in t and "\ufeff" not in t, path
    return t.replace("\r\n", "\n")


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


# ---- the v33 deliverables are the starting point: asserted byte-identical to what v33 delivered -------------------------------------------------------
V33_MD_SHA = "8533bb8457038373fa1648bfd4fe48c1f4e6941c78bb18ed6567b99b637b26e0"
V33_HTML_SHA = "8866c6fd9b030a3cb8759ad9ca05af516e2db1bb091ced63b9d1f113dfe30950"
s = blob(RP + "V8F_SCIENTIFIC_REPORT.md")
h = blob(RP + "v8f_report.html")
tpl = blob(RP + "page_template_v30.html")
assert sha(raw(RP + "V8F_SCIENTIFIC_REPORT.md")) == V33_MD_SHA, "the tip's report md is not the v33 deliverable"
assert sha(raw(RP + "v8f_report.html")) == V33_HTML_SHA, "the tip's report html is not the v33 deliverable"
assert tpl.count("%%MD%%") == 1 and h == tpl.replace("%%MD%%", s), "v33 html is not the template with the v33 md"
assert s.count("**Status:** research report, v33 of 2026-10-28") == 1

# ---- sources ---------------------------------------------------------------------------------------------------------------------------------------------
CR = blob(R + "raw/2026-10-28-crawl-diag/RESULTS.md")
CRP = blob(R + "raw/2026-10-28-crawl-diag/PREREG_FROZEN.md")
CRF = blob(R + "raw/2026-10-28-crawl-diag/FROZEN_SHA.txt")
CRRD = blob(R + "raw/2026-10-28-crawl-diag/READINGS.md")
CRRS = blob(R + "raw/2026-10-28-crawl-diag/READINGS_SHA.txt")
BM = blob(R + "raw/2026-10-28-bank-presteps/MANIFEST.md")
BR = blob(R + "raw/2026-10-28-bank-presteps/regret_prestep.md")
BF = blob(R + "raw/2026-10-28-bank-presteps/fitgap_m2b.md")
BC = blob(R + "raw/2026-10-28-bank-presteps/recovery/RECOVERY_RESULT.md")
BCARD = raw(R + "raw/2026-10-28-bank-presteps/FROZEN_CARDS.md")
BA1 = raw(R + "raw/2026-10-28-bank-presteps/FROZEN_AMENDMENT_1.md")
BA2 = raw(R + "raw/2026-10-28-bank-presteps/FROZEN_AMENDMENT_2.md")
T1 = blob(R + "raw/2026-10-28-trigger-probe/RESULTS.md")
T1P = raw(R + "raw/2026-10-28-trigger-probe/PREREG_FROZEN.md")
T1F = blob(R + "raw/2026-10-28-trigger-probe/FROZEN_SHA.txt")
T2 = blob(R + "raw/2026-10-29-trigger-probe2/RESULTS.md")
T2P = raw(R + "raw/2026-10-29-trigger-probe2/PREREG_FROZEN.md")
T2F = blob(R + "raw/2026-10-29-trigger-probe2/FROZEN_SHA.txt")
T2L = blob(R + "raw/2026-10-29-trigger-probe2/raw/POSTFREEZE_LOG.md")
SO = blob(R + "RESULT_V8F_SOUP.md")
SOP = raw("Project Steering/PREREG_V8F_SOUP.md")
SOPT = blob("Project Steering/PREREG_V8F_SOUP.md")
SOJ = raw(R + "raw/2026-10-29-soup-read/out/read/SOUP_READ.json")
REG = blob("Project Steering/GOALS_AND_CLAIMS.md")
REV_C = blob(RA + "2026-10-28-crawl-variance-lit/REVIEW.md")
REV_D = blob(RD + "2026-10-28-data-stages-lit/REVIEW.md")
KEYS_C = blob(RA + "2026-10-28-crawl-variance-lit/LIBRARY_KEYS.txt")
KEYS_D = blob(RD + "2026-10-28-data-stages-lit/LIBRARY_KEYS.txt")
LIB = json.loads(raw("TanitAD Research Lab/Library/library.json").decode("utf-8"))["entries"]

FT_RESULTS = [blob(_p) for _p in FT_PATHS]  # fine-token RESULT files at the tip, read only to check a FINAL version's fill texts (tables line for line, quotations verbatim)
# the landing commits named in the text: their author times are measured, not inherited
for _c, _t in (("c1c30fb7", "2026-10-28 23:04:22"), ("897b711b", "2026-10-29 00:32:44"), ("e03b348", "2026-10-29 02:00:06"), ("0c7b598", "2026-10-29 03:10:30"), ("5cc7d7b1", "2026-10-28 21:47:25")):
    _o = gitrun(["show", "-s", "--format=%ci", _c]).strip()
    assert _o.startswith(_t), (_c, _o)
# frozen cards: the bytes at the tip hash to the sha256 the packages recorded
for _b, _pre, _what in ((BCARD, "3202a62fbf7699005c60f7a6ae37fb544e4bf0a0f0564b7a9115ee7bf865fd5b", "FROZEN_CARDS.md"), (BA1, "4676bdb8d27c1095c942782889ab9a04932e5aa1cfe1fcda650285ca365b93c7", "FROZEN_AMENDMENT_1.md"),
                        (BA2, "17cce25fa237989efa589e299c491803377db5cb5265bc475d98bd15d8369f47", "FROZEN_AMENDMENT_2.md"), (T1P, "0eb1c25c9349295010993557c15bc1b3f38c1cc357fa4c63648174ad9e6bff97", "trigger-probe PREREG_FROZEN.md"),
                        (T2P, "d9af82be053463e37f295f5dc0b774a103e5858a7bddf2fd03a77a20739a40cb", "trigger-probe2 PREREG_FROZEN.md"), (SOP, "830d9c9143850628188f7d8489566e92f2a9cbdf050689331f1eaa7d3c78993c", "PREREG_V8F_SOUP.md"),
                        (SOJ, "dca10c07f5fabda9ddc9ea2725f425f53f1462bec8488d58ebc92ecd8d33de6a", "SOUP_READ.json")):
    assert sha(_b) == _pre, (_what, sha(_b))
# the crawl package's own stamps: PREREG_FROZEN.md and READINGS.md hash to what FROZEN_SHA.txt and READINGS_SHA.txt record
assert sha(raw(R + "raw/2026-10-28-crawl-diag/PREREG_FROZEN.md")) == "7f4601ec9035f30d6b94ba5e9c219f98ae4b1fec828f7cc04d154c1c8898a947" and "7f4601ec9035f30d6b94ba5e9c219f98ae4b1fec828f7cc04d154c1c8898a947" in CRF
assert sha(raw(R + "raw/2026-10-28-crawl-diag/READINGS.md")) == "a97423b613569b9e9b61ecd31e1b31ce9550e544df6356153349115dda174eca" and "a97423b613569b9e9b61ecd31e1b31ce9550e544df6356153349115dda174eca" in CRRS
assert "2026-10-28T21:21:51" in CRF and "2026-10-28T21:28:08" in CRRS and "written_before: any H-CRAWL-01/-02/-03 result" in CRRS
assert "2026-10-28T22:10:08" in T1F and "2026-10-28T23:11:02" in T2F


# ---- tables, copied line for line ------------------------------------------------------------------------------------------------------------------
def grab(text, anchor, nrows):
    assert text.count(anchor) == 1, (text.count(anchor), anchor[:70])
    lines = text[text.index(anchor):].split("\n")
    first = next(k for k, l in enumerate(lines) if k > 0 and l.startswith("|"))
    assert first <= 3, (anchor[:60], first)
    rows = []
    for l in lines[first:]:
        if not l.startswith("|"):
            break
        rows.append(l)
    assert len(rows) == nrows, (anchor[:60], len(rows), nrows)
    return rows


def ncols(line):
    body = re.sub(r"`[^`\n]*`", "x", line.replace("\\|", "x"))
    return body.count("|") - 1


TABLES = OrderedDict([
    ("CD_H1", (CR, "## 2. H-CRAWL-01 numbers (m/s; mean [95 % CI])", 6)),
    ("CD_SV", (CR, "**Counterfactual (closest faithful version).**", 7)),
    ("BK_REG1", (BR, "## Verdict (frozen R3): **MIDDLE BAND", 8)),
    ("BK_REG2", (BR, "Crawl state = own speed < 0.5 x logged speed at the matched row r'", 4)),
    ("BK_REC_C", (BC, "## (c) kept fraction (tick 2", 6)),
    ("BK_REC_R", (BC, "**Prediction NOT CONFIRMED** (frozen band 75-81 %)", 4)),
    ("BK_REC_E", (BC, "## (e) label-vs-human deviation", 6)),
    ("BK_FIT", (BF, "Controls PASS: True (my eval-side R2", 4)),
    ("TP1_ARMS", (T1, "## 2. All arms (AUC [95 % CI]; pooled out-of-fold", 11)),
    ("TP2_VERD", (T2, "## 1. Verdicts by the committed rules", 8)),
    ("TP2_NAV", (T2, "**What the nav is worth as a feature", 8)),
    ("TP2_ZNN", (T2, "TRAIN fold-mean AUC (linear [CI]; MLP in brackets):", 12)),
    ("TP2_V4", (T2, "TRAIN fold-mean AUC, linear (MLP):", 8)),
    ("TP2_STR", (T2, "P-STOP: GO rate (full pop.) TL_RED 0.445", 6)),
    ("SO_OUT", (SO, "Source: `pairs.<seed>.<main|rep>` and `OUTCOME` in the JSON.", 12)),
    ("SO_PS", (SO, "Per-seed booleans (`OUTCOME.per_seed`; PREREG section 10).", 4)),
    ("SO_ADE", (SO, "### 2.2 ADE6: the guard R-ADE and the adoption row R-KEEP (T1-kin)", 6)),
    ("SO_GRD", (SO, "R-T0 is at tier T0 (a point, no interval in the JSON;", 6)),
    ("SO_MOVE", (SO, "R-MOVE needs, in each pair, every crawl row separated-better", 18)),
    ("SO_SG", (SO, "SG on ADE6 (PREREG section 7): per window", 8)),
    ("SO_F2", (SO, "**Fact 2. At s0 the soup recovers most of the slow draw", 7)),
    ("SO_F2B", (SO, "**Fact 2b. The same holds on the four crawl rows at s0", 10)),
    ("SO_F41", (SO, "### 4.1 Separated cells per family, per seed, per pair", 7)),
    ("SO_LON", (SO, "### 4.2 LONGITUDINAL (speed, along-track, accel, distance keeping)", 8)),
    ("SO_LAT", (SO, "### 4.3 LATERAL (read first: the crawl rows are LATERAL plus M1)", 8)),
    ("SO_M1", (SO, "The M1 crawl share is a crawl row outside the LATERAL block of the JSON", 3)),
    ("SO_TAC", (SO, "### 4.4 TACTICAL", 4)),
    ("SO_STR", (SO, "### 4.5 STRATEGIC (not paused for v8F)", 5)),
    ("SO_COV", (SO, "`strategic_family` in the JSON reads", 7)),
    ("SO_BLK", (SO, "### 7.3 What is blocked, and on what (every remaining lever)", 8)),
    ("SO_PROV", (SO, "Evidence class MEASURED (ours). Every sha256 below is of the bytes at the tip", 16)),
])
TAB = {}
_BANRE = re.compile(r"\b(" + "|".join(BANNED) + r")\b", re.I)
for name, (src, anchor, n) in TABLES.items():
    TAB[name] = grab(src, anchor, n)
    assert all(r in src.split("\n") for r in TAB[name]), name
    assert len({ncols(r) for r in TAB[name]}) == 1, (name, sorted({ncols(r) for r in TAB[name]}))
    assert not any(_BANRE.search(r) for r in TAB[name]), ("a banned word in a table line", name)
    assert not any(re.search(r"closed[- ]loop|driving performance", r, re.I) for r in TAB[name]), ("a tier word in a table line", name)

# ---- claims taken from the sources, asserted present --------------------------------------------------------------------------------------------------
PHRASES = [
    # crawl diagnosis
    (CR, "Card outcome (a) does **not** hold for either run; (b) holds **on the point estimate only** for DAG-OACC s0; REP s0 is **between** the two committed outcomes."),
    (CR, "it reads **-1.034 [-1.298, -0.774]** = 118 % of DAG-OACC's gap (72 % of REP's)"), (CR, "The mean table reads -0.800 [-1.071, -0.529] = 91 % (56 % of REP's), inside the predicted share."),
    (CR, "The card's core claim, that a large part of the gap is selection plus regression to the conditional statistic, is supported by this read."),
    (CR, "306,566 train windows, window-list sha256 `e144adcd...8168` rebuilt and **matched**; 91,000 with v0 < 6 enter the table; 12 cells (4 speed buckets x 3 history classes), the smallest cell has 475 windows, no fallback."),
    (CR, "Nothing was tuned on the 327."), (CR, "In the stopped bucket the conditional **median** is the stay predictor (0.0 at tick 1 in the two stopped cells that hold 97.5 % of the stopped train windows)"),
    (CR, "on a stopped car they behave like a regression to the mixture."), (CR, "REP - DAG-OACC: all launch -0.556 [-0.699, -0.398]; stopped launch -0.318 [-0.580, -0.056]; moving launch **-0.611 [-0.778, -0.445]**; GT-stay -0.056 [-0.548, +0.345] (not separated)."),
    (CR, "gap **+0.516 [+0.177, +0.907]** on the 327, stopped launch +1.887, GT-stay **+3.554**; model - P10 = -1.391 [-1.669, -1.106] (DAG-OACC), -1.947 [-2.302, -1.593] (REP)."),
    (CR, "c1 median bias on the table's own cell = 4.4e-16"), (CR, "c3 shuffled-v_GT table moves the gap (-1.034 -> -1.223) and collapses the between-cell spread (2.12 -> 0.33)"), (CR, "c7 perfect predictor reads exactly 0.0: **PASS**. **c6 FAILED**"),
    (CR, "1,798 of 2,399 windows are bit-identical, 374 differ by more than 1e-4 m, the maximum is 6.5 mm at rollout step 30"), (CR, "the R2 route from the dump's own `fl` (no re-run) agrees to 4 dp."),
    (CR, "Falsifier 1 does **not** fire. The \"supported\" reading (point S_v >= 0.6 on both runs) is **met**, with lower CI bounds below 0.6."),
    (CR, "the card's prediction that the difference **tracks** the launch gap is **refuted by sign**"), (CR, "has the **smaller** S_v"),
    (CR, "scrambling them changes the plan by exactly 0.0"), (CR, "Stopped-start, 4 s: 0.456 [0.256, 0.595] / 0.359 [0.194, 0.479]."), (CR, "0.196 [-0.025, 0.350] / 0.134 [-0.058, 0.269]"),
    (CR, "**Speed set to 0** (positive control, moving windows): -3.56 [-3.86, -3.26] / -3.36 [-3.66, -3.07], S_v = 0.98 / 0.92."), (CR, "stopped-start 0.23 / 0.19, all 0.44 / 0.61, moving 0.49 / 0.71 m/s"),
    (CR, "For comparison a +0.5 m/s speed shift moves the output by ~0.42 / 0.37."), (CR, "**Controls (H-CRAWL-02): ALL PASS.**"), (CR, "bit-identical** to the untouched tool path"),
    (CR, "**Not evaluable.** No intermediate checkpoint of either run exists"), (CR, "`v8f_mini_train.py:190` `--ckpt-every` default 500 (not in either argv), :1651 `if step % a.ckpt_every == 0`, :1554 / :1582-1583 writes `ckpt_last.pt.tmp` then `os.replace` onto the **same** file."),
    (CR, "both `step = 3000`, optimiser state present, no EMA"), (CR, "the shared fine-tune start (parent M2b_s0 at its step 19,164)"), (CR, "sha256 `601cb435...c6f8`"), (CR, "**equal** to the `init_ckpt` sha both runs recorded"),
    (CR, "\"cost ~ 0\" is not true as written:** the trainer has no keep-last flag."), (CR, "the launch gate refuses a new flag without a G-DVB entry"), (CR, "**+0.020 [-0.101, +0.137]** at s1"),
    (CR, "single-seed-pair statements (`H-ESTIM-SEED-1`), not lever effects."), (CR, "One table specification (edges fixed in READINGS); the median-vs-mean choice changes magnitudes but not the per-run committed outcome."),
    (CRP, "(a) model gap − outcome-blind gap ≤ −0.4 m/s with CI excluding 0 **and** a non-positive GT-stay gap"), (CRP, "(b) the difference is within ±0.2 m/s"), (CRP, "`S_v ≤ 0.3` with upper CI < 0.5 on **both** runs"),
    (CRP, "Within-run SD ≥ 0.4 m/s ⇒ checkpoint noise dominates"), (CRP, "Source: dagland/research_crawl/HYPOTHESES.md (sha256 cd3e1d7089acae7eaf8522f074ded57ce2501b54b2323407ec9e79004d42ab65)"),
    (CRRD, "**S_v** = `sum_i dv_out,i / sum_i dv_in,i` (ratio of sums over a window subset"), (CRRD, "`v' = max(v + delta, 0)` on every row <= t0; delta in {-0.5, +0.5, +1.0} m/s"),
    (CRRD, "**Gap = mean over 0-based ticks {1, 2} of (v_own - v_GT), per window, then mean over windows**"),
    # bank pre-steps
    (BM, "613,132 records each"), (BR, "**MIDDLE BAND (report the multiplier; card neither dies nor is confirmed)**"), (BR, "Crawl state = own speed < 0.5 x logged speed at the matched row r' AND logged speed >= 3 m/s; usable = reason ok."),
    (BR, "amendment 1 sha256 `4676bdb8d27c1095c942782889ab9a04932e5aa1cfe1fcda650285ca365b93c7`"),
    (BR, "s0: crawl share over ALL records 6.13 %, over the 95930 off_lat/off_head records 8.79 %"), (BR, "The card's own prediction (INHERITED, card text): share < 3 % and ~450-600 crawl samples in ~15-19k draws."),
    (BR, "clip-cluster bootstrap (4,512 clips, B = 2000, seed 0"),
    (BC, "* (a) identity: **1000 / 1000** real windows reproduce the human label"), (BC, "the offset-ignoring arm FAILS the gate on 304 / 305 records (99.7 %); the no-clip / unchecked arm fails on 257 / 257 (100.0 %)."),
    (BC, "V1 s0: 0 of 18707; V1 s1: 0 of 24436; V2 s0: 0 of 14760; V2 s1: 0 of 18707."), (BC, "24 passed in 6.39s"), (BC, "3 passed in 0.13s"), (BC, "**Prediction NOT CONFIRMED** (frozen band 75-81 %): V1 reaches 59.5 % / 62.0 %."),
    (BC, "overall emitted 23.3 %, refused_infeasible 47.8 %, no_logged_future 25.8 %"), (BC, "labelable ceiling is 71.9 % / 70.6 % (s0 / s1)"),
    (BC, "the HUMAN labels themselves violate this envelope at 0.2 s resolution on 2.9 % / 2.6 % of sampled ok records (s0 / s1)"), (BC, "Envelope constants from `feasible_decode.py` (A_MAX 4.0, KAPPA_MAX 0.2, MU 0.7, 1 % margin)"),
    (BC, "(UNVERIFIED-as-'P10's')"), (BC, "partial-horizon recovery labels with a loss mask for the no-logged-future states (needs a trainer flag and a G-DVB entry, so box-2 work)"), (BC, "(i) a curvature-bounded planner (Dubins / clothoid with R_min = 5.05 m)"),
    (BF, "**PARTIAL verdict:** 2 of 6 checkpoints observed"), (BF, "Config mismatch (disclosed): the plain-M2b dumps ran `--limit logged`, the bank rollouts `--limit causal`."), (BF, "C_eval(logged) - C_eval(causal) = s0: +0.2992 m/s, s1: +0.3049 m/s"),
    (BF, "gives r = s0: 0.950 (FIT-LIMITED), s1: 1.047 (FIT-LIMITED)"), (BF, "~0.17 h on a 10 % clip subset"),
    (BF, "my eval-side R2 reproduces the M-LAUNCH instrument's own DAG-OACC R2 mean to < 1e-9 on 327 windows / 49 episodes"),
    # trigger probes
    (T1, "Full-population GO rate: **P-STOP 57.4 %** (10,948 / 19,070 windows, 859 clips), **P-MOVE 53.8 %** (38,670 / 71,930)."), (T1, "GO = the registered launch mask (`v6 >= v0 + 1`, chord over the last 0.2 s of the 6 s horizon)."),
    (T1, "**Evidence class: MEASURED** (artifacts in `raw/`). **Tier:** T0-class probe of representations on logged inputs, not driving performance."), (T1, "**P-STOP 0.4629 [0.4267, 0.49993] -- FAILED the committed \"CI covers 0.5\" by 7e-5.**"),
    (T1, "the pooled out-of-fold AUC is biased BELOW 0.5 on clip-clustered labels, because each fold's model carries an intercept fitted to the other folds' prevalence."),
    (T1, "so **the frozen pooled metric inflates dAUC(V+E - E) on P-STOP (+0.087 vs +0.054)**."), (T1, "(E pooled 0.593 vs per-fold mean 0.626)"), (T1, "C2 under it: 0.4831 [0.465, 0.502] PASS"),
    (T1, "test_foldmean_metric_unbiased_on_prevalence_shifted_null"), (T1, "V-alone minus E is +0.007 [-0.030, 0.041] (fold-mean)"),
    (T1, "nav + limit + E alone (N+E) reaches **0.719 pooled / 0.727 fold-mean on P-STOP (E: 0.593 / 0.626)**"), (T1, "P-STOP: n = 101 windows, 19 episodes, 61 GO: E 0.517 [0.41, 0.65], V+E 0.519 [0.29, 0.76], Z+E 0.655 [0.45, 0.85]"),
    (T1, "the TL strata are CLIP-level and VLM-derived (`vlm-cot`, disputed, untimed), so they are coarse; none enters the rule"), (T1, "**UNVERIFIED byte-for-byte against box 2's `D:\\tanitad\\cache_full`.**"),
    (T1, "**TL_RED +0.099** (E 0.609 -> V+E 0.708; Z+E 0.806), TL_GREEN **-0.058**"), (T1, "**lead-departs -0.019**"),
    (T1, "**But the ceiling is low and the evidence thin:**"), (T1, "the largest single source of GO predictability in Z is the oracle nav"),
    (T2, "`build_v9_labels.py:233-260` (`turn_table`): the turns are every sustained-yaw segment"), (T2, "**63.3 % when a turn lies ahead (70 % of windows) vs 43.8 % when none (30 %)**"),
    (T2, "P(turn ahead | GO) 77.0 % vs P(turn ahead | not GO) **60.1 %**"), (T2, "tip `build_v8_speed_max_labels.py:16-19, 138-146`"), (T2, "\"the maximum of the ego's OWN REALISED speed over [t0+2 s, +6 s]\" (provenance ego-future, oracle; quantised to 4 bins"),
    (T2, "So the trunk has TWO ego-future channels; the speed ceiling carries about twice the go/stay information of the nav token"), (T2, "no null-nav / nav-dropout path anywhere"),
    (T2, "**FAIL, as frozen: Z_nn+E on P-STOP, fold-mean 0.4543 [0.418, 0.491]: the CI misses 0.5.**"), (T2, "Z_nn+E P-STOP 0.5105 sd 0.0226 [0.470, 0.544]; with seed 2 the 11-draw mean is 0.505, seed 2 is the minimum of the 11 (a 1/11 event), and the true arm (0.760) exceeds the band maximum by 0.216."),
    (T2, "Scrambling the nav alone removes only 0.040 (P-STOP) / 0.032 (P-MOVE) of Z+E"), (T2, "Scrambling the speed ceiling as well removes a further 0.074 / 0.035."), (T2, "The contributions are not additive (interactions), so these are two readings of one table, not a decomposition."),
    (T2, "Z_nl+E - E +0.072 [-0.114, +0.227] / +0.019 [-0.040, +0.080]; Z+E - Z_nl+E +0.066 [-0.033, +0.151] / +0.076 [+0.014, +0.140]."), (T2, "the TRAIN residual (+0.06 / +0.08) is unconfirmed (and may be memorisation"),
    (T2, "**motion in the model's own 4 cached frames carries nothing the current frame does not**"), (T2, "V4+E - V+E -0.003 [-0.028, +0.015] / +0.006 [-0.009, +0.022]"), (T2, "the frames are 0.2 s apart, so t-3 is 0.6 s back"),
    (T2, "A loss-only stop/go head on this trunk (the previous probe's D1 lever) would mostly learn to read those oracle channels"), (T2, "The head is not the lever until the oracle inputs are out of the loop."),
    (T2, "**Motion is not the lever** (V4/Vdiff null)."), (T2, "The live perception lever is therefore targeted (light state, lead gap/TTC at higher resolution or an ROI crop), not temporal pooling."),
    (T2, "evaluate the deployed planner's GO windows with nav AND ceiling at their donor values"), (T2, "any GO-skill number measured with them in the input is optimistic by construction"),
    (T2, "One checkpoint, one seed: every Z statement is blind to training variance (H-ESTIM-SEED-1)."), (T2, "UNVERIFIED: that the Z_nn gain on TRAIN is not memorisation"),
    (T2, "Most of that diagnostic's N was the **`limit` / `limit_ok` input, which is NOT a posted speed limit**"),
    # SOUP
    (SO, "all **368 reader controls PASS** (368 of 368), reader wall 634.9 s, rc 0"), (SO, "promoted at `e03b348` (2026-10-29 02:00:06; sha256 `830d9c91…`"), (SO, "banked at `0c7b598` (2026-10-29 03:10:30)"),
    (SO, "Weights are averaged in float64 and rounded once to float32; the soups share the init M2b_s with their members."), (SO, "Control of the main pair: DAG-OACC_s. Control of the replicate pair: A_s (PREREG section 0 item 1)."),
    (SO, "The registered rule is that H must hold at both seeds, so the s1 pass does not change the word (PREREG section 10). Nothing below softens that."), (SO, "R-MOVE is false in all four pairs and R-CRAWL is false in all four pairs."),
    (SO, "Of the 16 crawl cells X − Y, 0 are separated-better and 9 are separated-worse (8 of them at s0"), (SO, "Read A reads **MARGIN, NOT DOMINANT** for SOUP, SOUP' and SOUP4 at both seeds"),
    (SO, "Five of six cells are separated-better; SOUP at s1 is not separated. Routing: **FLAT**, so the committed next is SOUP-K."), (SO, "BARRIER = SG separated-worse (> 0) in at least one of the four decisive soup-seed cells"),
    (SO, "the soup carrying part of the slow member's crawl"), (SO, "A soup that is better than REP_s and worse than DAG-OACC_s is not an improvement of the base (PREREG section 12)."),
    (SO, "the PREREG's first-order prediction for SOUP_0 − DAG-OACC_0 was about +0.20 m (ESTIMATED, section 0 item 3: half of +0.4025)"), (SO, "averaging recovers 50.8 to 76.9 % of that excess"),
    (SO, "at 23.1 to 49.2 % of the way from the good to the slow member against 50 % for a linear interpolation of the errors"), (SO, "so there is no draw-specific crawl for the average to remove."),
    (SO, "Both controls, DAG-OACC and A, are separated-worse than P10 on the four crawl rows at both seeds"), (SO, "**The lever the evidence points to is perception.**"),
    (SO, "**Proposed next experiment, PROPOSED and not registered: a finer-token DINOv3 probe**"), (SO, "It has no card, no committed outcomes and no cost estimate; none is invented."),
    (SO, "SOUP-K**, K = 4 pure-recipe draws per seed"), (SO, "**Status: not launched; blocked on the PI go (box-2 time).**"), (SO, "**A soup effect is the effect of averaging these two draws.** N = 2 per decisive soup"),
    (SO, "**SG is a mean-of-errors statistic.** SG < 0 says the average beats the mean of its members' errors. It does not say the soup beats the better member (Fact 2)."), (SO, "**R-T0 is a dev-box CPU T0 reading** on both arms of the pair; it is not box 2's standard eval."),
    (SO, "**The words 'perception is the lever' are a reading of diagnostics**"), (SO, "*\"T1-kin goal-head F1 and head accuracy are NOT COMPUTABLE here"), (SO, "the anti-echo block is NOT BUILT (the DAG-ORC rig carries none)."),
    (SO, "**The reader's provenance label.**"), (SO, "**The launcher was edited and restored.**"), (SO, "**A deletion.** The SOUP finishing agent deleted its own test directories in the session scratchpad (INHERITED from the brief)."),
    (SO, "**A register label.** The register row `H-V8F-SOUP-1` calls the -0.0125 cell \"TAC-LAT macro-F1\"."), (SO, "**What `check_result.py` is and is not.**"), (SO, "Twenty-eight steps ran, each at attempt 1"),
    (SO, "*\"ZZSOUP-DONE 28/28 steps, reader rc 0\"*"), (SO, "The 28 steps sum to 2074 s of step wall"),
    (SO, "STRATEGIC is not paused for v8F (PI ruling 2026-10-17); the JSON carries `paused: false`"),
    (SOPT, "**Status: PRE-REGISTERED, NOT RUN (promoted by the Master Mind 2026-10-29, from the draft sha256 4caafe79…"), (SOPT, "θ_soup[k] = (1/N) Σ_i θ_i[k] for every float tensor k of `ckpt[\"model\"]`, computed in float64 and rounded once to the members' dtype (float32)."),
    (SOPT, "HYPOTHESIS here: part of the crawl-row error of one draw is draw-specific, and an average of two draws keeps the shared part and halves the draw-specific part."),
    (SOPT, "prediction for SOUP_0 − DAG-OACC_0 on ADE6 is therefore about +0.20 m, worse (ESTIMATED"),
    (SOPT, "The soup checkpoint is the **donor member's checkpoint with `model` replaced by the average and `opt` set to `None`**"), (SOPT, "**The donor is the member the soup is read against**"),
    # register rows
    (REG, "**H-CRAWL-01** (2026-10-28) |"), (REG, "**H-CRAWL-02** (2026-10-28) |"), (REG, "**H-CRAWL-03** (2026-10-28) |"), (REG, "**H-ST-REGRET-PRE** (2026-10-28) |"), (REG, "**H-ST-RECOVERYLABEL-1** (2026-10-28) |"),
    (REG, "**H-DD-FITGAP-M2B** (2026-10-28) |"), (REG, "**H-CRAWL-06-PROBE** (2026-10-28) |"), (REG, "**H-CRAWL-06-PROBE2** (2026-10-29) |"), (REG, "**H-V8F-SOUP-1** (2026-10-29) |"),
    (REG, "cards frozen at 21:21 before any data (PREREG_FROZEN.md sha256 7f4601ec…)"), (REG, "cards frozen at 22:05 before data (FROZEN_CARDS.md sha256 3202a62f…); Amendment 1 (labeller construction, FITGAP statistic) at 22:17, before the first bank read at 22:21"),
    (REG, "a post-hoc V2 (Amendment 2, 22:27, after a 3,000-record smoke of V1) is reported beside, never in the verdict."), (REG, "Reading: MIDDLE BAND, the seeds straddle the 3 % line; the card is neither killed nor confirmed, and weighting buys about 3x."),
    (REG, "MEASURED (P-MOVE thin D1 as frozen; P-STOP frozen reading NOT EVALUABLE, exploratory D1; Z advantage carried by the oracle nav)"),
    (REG, "MEASURED (V4 NO-MOTION; Z_nn VISION-THROUGH-TRUNK by the letter on P-MOVE, P-STOP not evaluable; post-freeze the trunk's edge is the speed-ceiling and nav oracles)"),
    (REG, "once both ego-future channels are scrambled the trunk's go/stay gain equals the frozen image features' own small gain and is not distinguishable from 0 on eval."),
    (REG, "The go/stay trigger is weakly present in what v8F sees, and absent for green lights and departing leads: the lever is perception, and a loss-only stop/go head on this trunk would learn the oracle channels."),
    (REG, "**Refined by H-CRAWL-06-PROBE2 (2026-10-29): most of the trunk's oracle edge is the speed-ceiling input (ego-future, declared, the PI's 2026-09-10 ruling); the nav token is the smaller part.**"),
    (REG, "MEASURED (HARMS) | `Project Steering/PREREG_V8F_SOUP.md` |"), (REG, "Committed next (section 11, FLAT): SOUP-K, K = 4 pure-recipe draws per seed, 4 box-2 runs, its own PI go."),
    (REG, "PARTIAL (2 of 6 checkpoints, both FIT-LIMITED; GEN-GAP unreachable)"), (REG, "NOT EVALUABLE (no intermediate checkpoints)"),
]
_missing = [ph[:110] for src, ph in PHRASES if src.count(ph) < 1]
assert not _missing, _missing

# ---- the library keys the related-work paragraphs cite: in library.json, with the PDF sha256 the reviews' LIBRARY_KEYS files record, and the claim is the review's own ---------
KEYROWS = {}
for _t in (KEYS_C, KEYS_D):
    for _l in _t.split("\n"):
        if _l and not _l.startswith("#"):
            _p = [x.strip() for x in _l.split("|")]
            KEYROWS[_p[1]] = _p[3]
CITE = OrderedDict([  # key -> (review text, anchor that is the review's own claim)
    ("1904.08980", (REV_C, "Names the **inertia problem**: low speed correlates with no acceleration [p.3]")),
    ("2312.03031", (REV_C, "avg L2 **6.16 m** vs 0.37 m; image corruption 0.44-0.46 m [2312.03031 p.5, Table 2]")),
    ("2306.07957", (REV_C, "L1 waypoint regression **interpolates multimodal future velocity**; which mode wins \"depends on the training run and dataset\" [p.5-6]")),
    ("2310.20650", (REV_C, "in 56 % of 50 inspected episodes the nearest-neighbour features of the attended region include a red light, i.e. a perception error [p.8, Table VII]")),
    ("2203.05482", (REV_C, "Greedy soup on CLIP ViT-B/32: ImageNet **81.03 vs 80.38** best individual (+0.7 pp, 72 hyperparameter-diverse ingredients). The **uniform soup was 79.97, below** the best individual: it helps only when all ingredients are good [2203.05482 p.6, Table 3 and takeaway 2].")),
    ("2008.11687", (REV_C, "Models fine-tuned from the same pretrained weights stay in one basin, close in parameter space [2008.11687 p.1]")),
    ("1912.05671", (REV_C, "Networks become stable to SGD noise early in training, then land in one linearly connected region [1912.05671 p.1]")),
    ("2205.09739", (REV_C, "Weight averaging then acts like ensembling, and it succeeds when the variance term dominates the error [2205.09739 p.1]")),
    ("2002.06305", (REV_C, "Same hyperparameters, different seeds give \"substantially different results\"; init and data order contribute comparably [2002.06305 p.1]")),
    ("2605.15120", (REV_C, "CLOVER on NAVSIM: 3 seeds, PDMS 94.488/94.502/94.492, spread <0.02 over ~12k scenes [2605.15120 p.30-31, Table 21]")),
    ("1011.0686", (REV_D, "supervised BC cost grows quadratically in horizon T; aggregation of on-policy data restores linear growth | [PUB 1011.0686 p3]")),
    ("2412.05334", (REV_D, "**7M model RMM 0.7702 > 102M model 0.7614**; its own BC base 0.7591")),
    ("2210.02174", (REV_D, "upsampling the scenes that fail closed loop: **up to ~35% fewer collisions** on some metrics")),
    ("2603.15185", (REV_D, "point-estimator planner's growth **decelerates after ~8,000 scenes**")),
    ("2504.04338", (REV_D, "closed-loop **MDBF improves only to 256 h, then plateaus at ~1 km**")),
])
for _k, (_rev, _anc) in CITE.items():
    _a = re.sub(r"\s+", " ", _anc)
    assert _a in re.sub(r"\s+", " ", _rev), ("the review does not carry this claim", _k, _a[:80])
    assert _k in LIB and LIB[_k]["sha256"] == KEYROWS[_k], ("library key / sha256 mismatch", _k)
assert "Direction: PI 2026-10-28" in REV_D and "No paper measures our quantity." in REV_C and "removes only the run-specific term, never a bias shared by all runs" in REV_C
assert "24.3% of tick-2 records (lateral > 3 m or heading > 30 deg) are discarded because the matched human label is invalid there" in REV_D
assert "the closest published form of our REGRET card" in REV_D and "I found **no driving-planner paper that measures weight averaging or EMA**" in REV_C
assert "so visited states stay near GT and the GT label stays valid" in REV_D

# ============================================================================================================================================================
# ---- 5.4 background -----------------------------------------------------------------------------------------------------------------------------------
# ============================================================================================================================================================
MATH_54 = r"""### 5.4 Background for §7.30–§7.33: a floor under selection, input sensitivity, weight averages, fold-mean AUC

*Definitions and short derivations for the kinds of read that §7.30–§7.33 use. The algebra below is DERIVED from the definitions (no data); the section named at the end of each paragraph measures the numbers.*

**(a) A floor under selection on the outcome (§7.30).** A window has inputs $x$, here a speed bucket $b(v_0)$ and a history class $h(\Delta v)$, so a cell is $c=(b,h)$ (12 cells), and a future outcome $y_n$, the ground-truth speed at replan tick $n\in\{1,2\}$. The *launch set* $L$ is chosen on the same future, $G=\mathbb 1[\,v_{6\,\mathrm s}\ge v_0+1\,]$. A predictor that sees only $c$ and was fitted on TRAIN returns the cell median $m_n(c)$ under an $L_1$ loss and the cell mean $\mu_n(c)$ under $L_2$. Write $p_c=\Pr(G=1\mid c)$ and $\mu_g(c)=\mathbb E[y_n\mid c,G=g]$. Then
$$
\mu(c)=p_c\,\mu_1(c)+(1-p_c)\,\mu_0(c),\qquad
\mu(c)-\mu_1(c)=-(1-p_c)\big(\mu_1(c)-\mu_0(c)\big),\qquad
\mu(c)-\mu_0(c)=p_c\big(\mu_1(c)-\mu_0(c)\big).
$$
When launching windows have the higher future speed, $\mu_1\ge\mu_0$, a predictor blind to the trigger reads a **negative** gap on the launch windows and a **positive** gap on the stay windows by selection alone; where $p_c<\tfrac12$ the median is the stay value and the launch deficit is read in full. The *outcome-blind floor* is $\mathrm{gap}_{\rm blind}=\frac1{|L|}\sum_{w\in L}\big(f(c_w)-y_w\big)$, and the diagnostic for a model $g$ is $D_g=\mathrm{gap}_g-\mathrm{gap}_{\rm blind}$, paired by window and read with the episode-cluster bootstrap. Two outcomes were committed before any data: $D_g\le-0.4$ m/s with the interval excluding 0 and a non-positive GT-stay gap (the model crawls beyond what its inputs allow), or the whole interval inside $(-0.2,+0.2)$ (the crawl is information-limited).

**(b) Input sensitivity $S_v$ (§7.30).** For a window set $W$ and a speed shift $\delta$, let the realised change of the input speed be $\Delta v^{\rm in}_i=v'_{0,i}-v_{0,i}$ with $v'=\max(v+\delta,0)$, and let $\Delta v^{\rm out}_i$ be the change of the 2 s output speed of the tick-0 plan. Pooled over $\delta\in\{-0.5,+0.5,+1.0\}$ m/s,
$$
S_v=\frac{\sum_i\Delta v^{\rm out}_i}{\sum_i\Delta v^{\rm in}_i}
$$
(a ratio of sums, equal to $\overline{\Delta v^{\rm out}}/\delta$ when nothing is clipped). $S_v=1$ means the output speed follows the input speed one for one and $S_v=0$ that it ignores it. The plan is the frozen prior plus a learned residual, so $S_v^{\rm plan}=S_v^{\rm prior}+S_v^{\rm res}$ exactly (one denominator): the prior's part and the residual's part are read off one table.

**(c) The soup gain (§7.32).** For members $\theta_1,\dots,\theta_K$ fine-tuned from one initialisation the uniform soup is $\bar\theta=\frac1K\sum_k\theta_k$ (float64, rounded once). For a metric $M$ (here ADE6, per window, then averaged) the soup gain is
$$
\mathrm{SG}=M(\bar\theta)-\frac1K\sum_{k=1}^K M(\theta_k).
$$
If $M$ is convex along the segment between the members, Jensen gives $\mathrm{SG}\le0$; for a quadratic $M(\theta)=M^*+\tfrac12(\theta-\theta^*)^\top H(\theta-\theta^*)$ and two members, $\mathrm{SG}=-\tfrac18(\theta_a-\theta_b)^\top H(\theta_a-\theta_b)$. A separated $\mathrm{SG}>0$ is a *barrier* (the metric lies above the chord at the midpoint: the linear-mode-connectivity picture of 1912.05671), and a separated $\mathrm{SG}<0$ is the averaging gain. The adoption test compares the soup with the *control*, the member it replaces. With a good member (the control, error $M_g$), a slow member ($M_s=M_g+\Delta$, $\Delta\ge0$) and $K=2$,
$$
M(\bar\theta)-M_g=\mathrm{SG}+\tfrac12\Delta,\qquad
\frac{M_s-M(\bar\theta)}{\Delta}=\tfrac12-\frac{\mathrm{SG}}{\Delta}.
$$
The soup's error relative to its control is the soup gain plus half the member spread, so a soup beats its control only if $\mathrm{SG}<-\Delta/2$, and the share of the spread it recovers is $\tfrac12-\mathrm{SG}/\Delta$. §7.32 measures both sides of this identity.

**(d) The mean of per-fold AUCs, and why the pooled out-of-fold AUC is biased below 0.5 on clip-clustered labels (§7.31).** Windows are split by episode into $K=5$ folds; the model for fold $k$ is fitted on the other folds and scores fold $k$. The statistic read is the mean of per-fold AUCs,
$$
\overline{\mathrm{AUC}}=\frac1K\sum_{k=1}^K\mathrm{AUC}_k,\qquad
\mathrm{AUC}_k=\Pr\big(s_i>s_j\mid y_i=1,\,y_j=0;\ i,j\in\text{fold }k\big),
$$
and the alternative pools all out-of-fold scores before ranking. Under a null (labels independent of the features) $\mathrm{AUC}_k$ has mean $\tfrac12$ whatever the model's intercept, because a constant added to every score of a fold does not change the within-fold ranks. The pooled statistic also ranks *across* folds, where the intercepts differ: the intercept of fold $k$'s model is the logit of the prevalence $\bar\pi_{-k}=(N\bar\pi-n_k\pi_k)/(N-n_k)$ of the *other* folds, which falls as the fold's own prevalence $\pi_k$ rises. For two folds of equal size $n$, let $e_{kl}$ be the probability excess over $\tfrac12$ that a positive of fold $k$ outranks a negative of fold $l$; its sign is that of the intercept difference, $\operatorname{sign}e_{kl}=\operatorname{sign}(\pi_l-\pi_k)$, and $e_{lk}=-e_{kl}$. The net excess of the cross-fold pairs is proportional to
$$
n^2\,e_{kl}\,\big[\pi_k(1-\pi_l)-\pi_l(1-\pi_k)\big]=n^2\,e_{kl}\,(\pi_k-\pi_l)<0\quad\text{whenever }\pi_k\ne\pi_l .
$$
With labels clustered by clip, fold prevalences disperse more than under independent labels, so the pooled AUC sits below $\tfrac12$ under a null and the per-fold mean does not. The paired difference $\Delta\mathrm{AUC}=\mathrm{AUC}(V{+}E)-\mathrm{AUC}(E)$ is taken on the same windows and the same resampled episodes. §7.31 reports both statistics.

**(e) The episode-cluster bootstrap and the three variances (§5.3, applied to §7.30–§7.33).** §5.3 gives $\mathrm{Var}(\hat\Delta)=V_\theta+V_\omega+V_E$. The bootstrap resamples $M$ episodes (clips, for the bank pre-steps) with the models and the labels held fixed, so it estimates only $V_E$; at a fixed per-episode spread its half-width scales as $1/\sqrt M$ (standard cluster-sampling arithmetic), so the 49-episode launch subset, the 15-episode stopped-start subset and the 10-episode GT-stay subset carry wider intervals than the 141-episode split. The other terms are answered by design, not by the interval:

| read | resampling unit | training runs ($V_\theta$) | inference runs ($V_\omega$) | what else the interval cannot see |
|---|---|---|---|---|
| §7.30 launch gaps and S_v | 49 / 15 / 10 episodes, models fixed | two runs are two training seeds of one recipe: every run contrast is one pair | none: the planner is deterministic | whether the second seed would repeat the first (s0 and s1 differ, §7.30) |
| §7.30 bank pre-steps | 4,512 clips, B = 2,000 | the two seeds of the bank only | none | the bank is one rollout of each generator |
| §7.31 probes | 849 / 2,140 clips, and eval-141 | one checkpoint (DAG-OACC s0) | none | the draw of a label permutation in a null control: ten further draws spread 0.5105 (sd 0.0226) where the first read 0.4543 |
| §7.32 soups | 141 episodes, paired | the replicate pair at each seed; N = 2 members per soup | none: a soup is a fixed set of weights | the member set (fixed in advance, no selection on eval-141) |

**(f) Bank statistics (§7.30).** A *crawl state* is a rollout state whose own speed is below half the logged speed at the matched route point while that logged speed is at least 3 m/s; the crawl share $s$ is the fraction of usable bank records in that set. The deficit-weighted draw mixes the uniform draw 50 : 50 with a draw proportional to $d_i=\max(0,\,v_{\rm GT}(r')-v_{\rm own})$, so
$$
M_{\rm def}=\frac{P_{\rm mix}(\text{crawl})}{P_{\rm unif}(\text{crawl})}=\tfrac12+\tfrac12\,\frac{P_d(\text{crawl})}{s},\qquad M_{\max}=\tfrac12+\frac{1}{2s},
$$
the ceiling putting all of $P_d$ on crawl states. The fit-gap ratio is $r=D_{\rm train}/D_{\rm eval}$ with $D=-C$ the launch-speed deficit (m/s, positive = slower than the human): $r\ge0.5$ is *fit-limited* (the deficit is as large where the model was trained as where it is evaluated) and $r\le0.25$ a generalisation gap."""

# ---- related work ------------------------------------------------------------------------------------------------------------------------------------
REL_WORK = """**Inertia, copycat and the launch crawl (v34).** *The three paragraphs added in v34 repeat the claims of the two 2026-10-28 literature reviews (`R_A/2026-10-28-crawl-variance-lit/REVIEW.md` and `R_D/2026-10-28-data-stages-lit/REVIEW.md`, with the pages they cite); their primaries are banked in the Library (landed `5cc7d7b1`) and cited here by library key. I did not re-read the PDFs, so detail beyond the review's own sentence is UNVERIFIED.* Codevilla et al. (1904.08980) name the inertia problem, low speed correlating with no acceleration (p.3), and add a speed-prediction branch to the shared backbone to reduce the dependency on input speed (p.4) **[P]**. Li et al. (2312.03031) report that VAD with its ego-velocity input set to zero collapses to a near-stationary plan (average L2 6.16 m against 0.37 m) while blank camera images move it little (0.44–0.46 m) (p.5, Table 2) **[P]**; the `H-CRAWL-02` counterfactual of §7.30 is the same style of input intervention applied to v8F-mini, and it finds the speed sensitivity in the frozen prior rather than in the learned residual. Jaeger et al. (2306.07957) report that L1 waypoint regression interpolates a multimodal future velocity and that which mode wins depends on the training run and the dataset (p.5–6) **[P]**: the mechanism behind the outcome-blind floor of §5.4 and §7.30. Cultrera et al. (2310.20650) report that the residual stalls of a stop/go-trained agent at green lights come with a red light among the nearest-neighbour features of the attended region in 56 % of 50 inspected episodes, which the review reads as a perception error (p.8, Table VII) **[P]**; the review adds that their stop/go output also conditions stage 2, which the goal-input rule of this programme excludes as built. The review found no paper that measures our quantity, the launch-speed gap in m/s: every published effect is a CARLA success rate, a nuPlan closed-loop score, an inertia-failure count or an ImageNet accuracy, so any conversion to m/s is ESTIMATED **[P]**.

**Weight averaging and run-to-run variance (v34).** Wortsman et al. (2203.05482) report for a greedy soup on CLIP ViT-B/32 an ImageNet accuracy of 81.03 against 80.38 for the best individual model (+0.7 pp, 72 hyperparameter-diverse ingredients), and a uniform soup of 79.97, below the best individual: it helps only when all ingredients are good (p.6, Table 3) **[P]**. Models fine-tuned from the same pretrained weights stay in one basin (2008.11687, p.1), networks become stable to SGD noise early in training and then land in one linearly connected region (1912.05671, p.1), and weight averaging then acts like ensembling and succeeds when the variance term dominates the error (2205.09739, p.1) **[P]**. The same hyperparameters with different seeds give substantially different results, and initialisation and data order contribute comparably (2002.06305, p.1) **[P]**. Aggregate benchmark metrics over about 12k scenes show a tiny seed spread (2605.15120: three seeds, PDMS 94.488 / 94.502 / 94.492, p.30–31, Table 21), narrow or interactive subsets do not, and the review found no driving-planner paper that measures weight averaging or EMA (a negative search result, not proof of absence) **[P]**. Averaging removes the run-specific term and never a bias shared by all runs (review, bottom line 5): this is the premise that §7.32 tests.

**On-policy supervision, label validity and data (v34).** The cost of supervised behaviour cloning grows quadratically in the horizon, and aggregating on-policy data restores linear growth (DAgger, 1011.0686, p.3) **[P]**. In closed-loop supervised fine-tuning of tokenised traffic models (2412.05334, p.1 and p.6) a 7M model reaches an RMM of 0.7702, above a 102M model at 0.7614 and its own behaviour-cloning base at 0.7591; the method keeps the ground-truth label valid by picking, among the top-K, the action nearest the ground truth so that visited states stay near it **[P]**. The data review reads this as the reason label validity decides the recovery-label work of §7.30 (a bank statistic of the review: 24.3 % of tick-2 records are discarded because the matched human label is invalid there). Upsampling the scenes that fail in closed loop (CW-ERM, 2210.02174, p.4) gives up to about 35 % fewer collisions on some metrics, which the review calls the closest published form of the regret-weighted draws that `H-ST-REGRET-PRE` pre-measures **[P]**. Point-estimate planners saturate with data early in closed-loop benchmarks: the growth of a point-estimator planner decelerates after about 8,000 scenes (2603.15185, p.6) and the mean distance between failures improves only to 256 h, then plateaus (2504.04338, p.8) **[P]**; whether v8F-mini is in that regime is not measured here."""
REL_WORK = REL_WORK.replace("R_A/", RA).replace("R_D/", RD)

# ============================================================================================================================================================
# ---- 7.30 crawl diagnosis and the DAgger-bank pre-steps ------------------------------------------------------------------------------------------------
# ============================================================================================================================================================
SEC_730 = """### 7.30 Crawl diagnosis (zero training): the smaller crawl sits at the outcome-blind floor on the point, the speed sensitivity belongs to the frozen prior, and the DAgger-bank pre-steps read a middle band

* **Status of this section (v34).** Six claim rows, all zero training on the dev box (CPU, `CUDA_VISIBLE_DEVICES=-1`; no box 2): `H-CRAWL-01` (the outcome-blind floor), `H-CRAWL-02` (the speed sensitivity `S_v`), `H-CRAWL-03` (within- against between-run variance) and the three pre-steps of the data and staging review, `H-ST-REGRET-PRE`, `H-ST-RECOVERYLABEL-1` and `H-DD-FITGAP-M2B`. The crawl package is `R/raw/2026-10-28-crawl-diag/` (landed `09ebfa3e`, 2026-10-28 21:58:17, **[M]** from the commit) and the bank package `R/raw/2026-10-28-bank-presteps/` (landed `aea53c4c`, 22:36:30, **[M]**); the rows are in `Project Steering/GOALS_AND_CLAIMS.md` at the tip. They test cards of the two 2026-10-28 literature passes (§3) and follow Rule Zero: LAUNCH-INTENT (§7.29), SPEEDSUB-T1 (§7.28) and DAG-ORC (§7.26) each left the crawl rows where they were, so the cheapest discriminating experiments are measurements of where the crawl comes from, run before any further training arm.
* **Cards before data [M].** Crawl package: the three cards were copied byte for byte from the crawl review's hypothesis file (sha256 `cd3e1d70…`, as the frozen card records it) and frozen first (`PREREG_FROZEN.md`, sha256 `7f4601ec…`, stamped 2026-10-28T21:21:51); the reading choices (table keys, statistic, verdict rule, controls) were written next (`READINGS.md`, sha256 `a97423b6…`, stamped 21:28:08 and marked as written before any `H-CRAWL` result). Bank package: the cards (`FROZEN_CARDS.md`, sha256 `3202a62f…`), Amendment 1 for the labeller construction and the FITGAP statistic (`4676bdb8…`), and a post-hoc Amendment 2 (`17cce25f…`, a second target ladder for the recovery labeller, written after a 3,000-record smoke of the first and reported beside, never in the verdict). The register gives 22:05 for the cards, 22:17 for Amendment 1 (before the first bank read at 22:21) and 22:27 for Amendment 2 (**[I]**, the register row; the three sha256 values are **[M]**, re-measured on the blobs). Neither package changed a threshold or a reading after data.
* **Tier and estimator [M].** The `H-CRAWL-01` gaps are T1-kin (the car's own executed state with the logged cameras: a self-action open loop). `H-CRAWL-02` is a first-tick inference call at tick 0 (T0-class: teacher-forced, one `forward_infer` call, no rollout; the counterfactual inputs are measurement tools and are never deployed). The three pre-steps read the DAgger banks (T1-kin rollout states on TRAIN) and, as a reference, the banked T1-kin dump of plain M2b on eval-141. Estimator: the (paired) episode-cluster bootstrap of `taniteval.ci`, B = 2,000, seed 0, over the episodes of each subset (the 327 launch windows are 49 episodes; the stopped-start subset is 61 windows / 15 episodes; the GT-stay set 32 windows / 10 episodes); the bank pre-steps resample clips (4,512). Each answers *"would another draw of EPISODES (or clips) say this?"* and nothing else. Two runs are two training seeds of one recipe (`H-ESTIM-SEED-1`) and the planner is deterministic, so no inference variance exists (§5.3, §5.4 (e)).
* **The four families, per family (STRATEGIC is reported, not paused, for v8F).** These six reads are measurements on subsets of windows and on the bank, not full evaluations. LONGITUDINAL: the launch-speed gap (mean of `v_own − v_GT` at ticks 1 and 2) and the 2 s output speed are the quantities read; along-track, acceleration and distance-keeping rows are not computed here. LATERAL: not computed here (the crawl's lateral rows are read in §7.26 and §7.29). TACTICAL: not computed here. STRATEGIC: not computed here, and not paused (PI ruling 2026-10-17). Nothing in this section is a statement about those three families.
* **`H-CRAWL-01`, the outcome-blind floor [M]** (`R/raw/2026-10-28-crawl-diag/RESULTS.md`, `CRAWL_DIAG.json`; T1-kin gaps in m/s, negative = slower than the human). The 327 launch windows are selected because the human launched, so a predictor blind to the trigger scores negative on them by construction (§5.4 (a)). The blind predictor is a table fitted on TRAIN only (306,566 windows, window-list sha256 `e144adcd…` rebuilt and matched; 91,000 with `v0` < 6 m/s enter it; 12 cells = 4 speed buckets × 3 history classes, the smallest cell 475 windows, no fallback; nothing was tuned on the 327), read as the cell median (primary) and the cell mean (beside). D is the model gap minus the blind gap, paired by window:

<<CD_H1>>

  * **Committed readings.** Outcome (a) (D ≤ −0.4 m/s with the interval excluding 0 and a non-positive GT-stay gap) fails for both runs: on the GT-stay windows (stopped, the human stays; 32 windows / 10 episodes) the models move **+1.272 [+1.005, +1.624]** and **+1.215 [+1.037, +1.366]** m/s. Outcome (b) (the whole interval inside ±0.2) holds **on the point only** for DAG-OACC s0 (D median +0.159 [−0.136, +0.468], mean table −0.075 [−0.377, +0.232]; the interval half-width is about 0.30). REP s0 sits between the two (D median −0.397 [−0.777, +0.010], mean table −0.631 [−1.008, −0.225]): **neither committed outcome**. The runs split by window class: in the moving launch windows (266 / 47) DAG-OACC reads −1.008 against a floor of −0.948 (D −0.060 [−0.430, +0.330]) while REP sits below it (D −0.670 [−1.130, −0.196]).
  * **Reading [M for the arithmetic, H for the inference].** The card's predicted floor (−0.3 to −0.8 m/s, 35–95 % of the gap) is not met at the median table, which reads −1.034 [−1.298, −0.774] m/s, 118 % of DAG-OACC's gap and 72 % of REP's; the mean table, −0.800 [−1.071, −0.529], is 91 % (56 % of REP's) and inside the predicted share. In the stopped bucket the conditional median is the stay value, so it reads the full launch deficit, while the conditional mean is the mixture: both models sit within +0.29 m/s of the mean table in the four stopped cells, which on a stopped car is a regression to the mixture of launching and staying. The smaller crawl is therefore what a predictor blind to the launch trigger shows, and the lever is trigger perception or the output representation, not a copycat fix (the card's wording for outcome (b)); the half-width of 0.30 keeps this a statement on the point. Where the 0.556 m/s run difference lives: REP − DAG-OACC is −0.556 [−0.699, −0.398] over all launch windows, −0.611 [−0.778, −0.445] over the moving ones, −0.318 [−0.580, −0.056] over the stopped ones and −0.056 [−0.548, +0.345] on the GT-stay windows (not separated).
  * **P10 and the controls.** P10, re-run with the same definitions, reads a gap of +0.516 [+0.177, +0.907] m/s on the 327 (stopped launch +1.887, GT-stay +3.554): faster than the human in launch and stay windows alike, and not a conditional predictor (model − P10 = −1.391 [−1.669, −1.106] for DAG-OACC and −1.947 [−2.302, −1.593] for REP). Controls c1 to c5 and c7 PASS (median bias on the table's own cell 4.4e-16; the shuffled-`v_GT` table moves the gap, −1.034 to −1.223, and collapses the between-cell spread, 2.12 to 0.33; a perfect predictor reads exactly 0.0). **c6 FAILED**: the P10 re-run did not reproduce the dump's `fl` bit for bit (1,798 of 2,399 windows are bit-identical, 374 differ by more than 1e-4 m, the maximum is 6.5 mm at rollout step 30, growing smoothly: a float-level signature), so the P10 rows are measured with a caveat and do not enter the verdict; the independent route from the dump's own `fl` (no re-run) agrees to 4 dp. The registered readings were not changed after data.
* **`H-CRAWL-02`, the speed sensitivity `S_v` [M]** (first-tick inference at tick 0, T0-class; 61 stopped-start windows / 15 episodes, B = 2,000). At tick 0 the planner reads only the NOW row of its kinematic inputs (v, a, κ and products), the extrapolated `base` and the 4 s max-speed bucket; scrambling the older rows changes the plan by exactly 0.0 (a control). The counterfactual changes the speed column and rebuilds every derived input with the tool's own functions, so it is a speed-sensor counterfactual (odometry then disagrees with the speed):

<<CD_SV>>

  * **Committed readings.** Falsifier 1 (`S_v` ≤ 0.3 with the upper bound < 0.5 on both runs) does not fire, and the "supported" reading (point `S_v` ≥ 0.6 on both runs) is met, with lower bounds below 0.6 (DAG-OACC +0.736 [+0.593, +0.829], REP +0.628 [+0.501, +0.726]). Falsifier 2 does not fire (the run difference +0.108 [+0.053, +0.167] is separated), but the card's prediction that the difference tracks the launch gap is **refuted by sign**: the run with the worse launch gap, REP, has the smaller `S_v`. `S_v` falls with the horizon (stopped-start, 4 s: 0.456 [0.256, 0.595] / 0.359 [0.194, 0.479]; the 6 s end chord, secondary and not the registered horizon: 0.196 [−0.025, 0.350] / 0.134 [−0.058, 0.269]).
  * **Attribution [M for the table, H for the inference].** The frozen P10 prior alone gives +0.686 [+0.557, +0.768] on the stopped-start set, identical in both checkpoints, and the learned residual adds +0.050 [+0.015, +0.090] (DAG-OACC) and −0.059 [−0.121, +0.012] (REP): the speed sensitivity belongs to the frozen prior, and REP's lower `S_v` comes from a residual that counteracts it. Kinematic-token dropout or history truncation on the learned path (cards `H-CRAWL-04` and `-05` of the crawl review) has no learned shortcut to remove. Positive control: setting the input speed to 0 on the moving windows moves the 2 s output by −3.56 [−3.86, −3.26] / −3.36 [−3.66, −3.07] m/s (`S_v` 0.98 / 0.92). An image from another episode (a launch donor) moves the output by 0.23 / 0.19 m/s on stopped starts, against about 0.42 / 0.37 m/s for a +0.5 m/s speed shift, and by 0.49 / 0.71 m/s on moving windows (mean |change|). All 8 controls PASS; the shift of 0 through the rebuild path is bit-identical to the untouched path.
* **`H-CRAWL-03`, within- against between-run variance: NOT EVALUABLE [M]** (feasibility only; nothing was run). No intermediate checkpoint of either run exists, after two independent probes (by location and by name): each train directory holds one `ckpt_last.pt` (DAG-OACC s0 `67265ec1…`, REP s0 `b03fd6ef…`, both at `step` = 3000), because the trainer rewrites that single file every `--ckpt-every` 500 steps (`stack/scripts/v8f_mini_train.py:190`, `:1651`, `:1554` and `:1582-1583`, read from source). The shared fine-tune start exists (M2b s0 at its step 19,164, sha256 `601cb435…`, equal to the `init_ckpt` both runs recorded). The card's "cost about 0" does not hold as written: the trainer has no keep-last flag, so keeping the last five checkpoints needs a new flag (the launch gate refuses a flag without a G-DVB entry) or a copy sidecar on the box. Context (**[I]**, not re-measured here): the same-init REP − DAG-OACC launch difference is −0.556 [−0.699, −0.398] m/s at s0 and +0.020 [−0.101, +0.137] at s1.
* **`H-ST-REGRET-PRE`: MIDDLE BAND [M]** (T1-kin rollout states on TRAIN, the DAgger banks of M2b s0 / s1, 613,132 records each; clip-cluster bootstrap over 4,512 clips, B = 2,000). The card asks whether crawl states are rare enough in the bank that deficit-weighted draws concentrate the p = 0.1 own-state budget on them by at least 5 times. A crawl state is an own speed below half the logged speed at the matched route point with a logged speed of at least 3 m/s; the decision statistic is the pooled share of usable records (frozen: a share ≥ 15 % at both seeds kills the card; < 3 % at both seeds with M_def ≥ 5 confirms it; anything else is the middle band):

<<BK_REG1>>

<<BK_REG2>>

  The shares straddle the 3 % line (5.752 % [5.439, 6.062] at s0, 2.608 % [2.443, 2.782] at s1), so the card is neither killed nor confirmed. Tick 2 carries the crawl (9.575 % / 5.071 % against 3.147 % / 0.937 % at tick 1). The 50 : 50 deficit weighting buys 2.68 [2.60, 2.77] and 3.08 [2.99, 3.18] times, below the 5 times branch, against ceilings of 9.2 and 19.7 times if all weight fell on crawl states (§5.4 (f)); the expected crawl samples in the 14,688 / 14,839 own-state draws are 845 [799, 890] / 387 [363, 413]. The card's own prediction (**[I]**, card text) was a share below 3 % and about 450–600 crawl samples in 15–19k draws. Beside: the crawl share over all records is 6.13 % / 2.76 %.
* **`H-ST-RECOVERYLABEL-1`: labeller gates PASS, kept-fraction prediction NOT MET [M]** (a census of both full banks, 306,566 windows per tick, labels offline; T1-kin states on TRAIN; no sampling interval is quoted for a census). The question is whether a kinematically feasible recovery label onto the human path can be given to the off-route own states that the DAgger bank drops, lifting the usable tick-2 states from 53.9 % to the frozen 75–81 % band, with zero feasibility violations and an exact identity on on-route states. Gates: the identity branch reproduces the human label on 1000 / 1000 real windows with zero tolerance; the offset-ignoring arm (a deliberate regression) fails the gate on 304 / 305 real off-route records and the unclipped arm on 257 / 257; 0 violations among 18707 (s0) and 24436 (s1) emitted V1 labels (a refusal is not a violation); 24 labeller tests and 3 bootstrap-equivalence tests pass. Kept fraction, then refusals:

<<BK_REC_C>>

<<BK_REC_R>>

  The 75–81 % prediction is **not met**: V1 reaches 59.5 % / 62.0 % at tick 2 (V2, the post-hoc arc-lookahead variant reported beside, 58.2 % / 60.1 %). Of s0's 74368 tick-2 off-route records 25.8 % have no complete logged future at the matched point (the card's 78 % double-counts them) and 47.8 % are refused as infeasible, mostly below 3 m/s own speed or above 60° of heading error; with the no-logged-future records removed the labelable ceiling is 71.9 % / 70.6 %. Label deviation from the human path, emitted labels, world frame:

<<BK_REC_E>>

  The envelope constants (A_MAX 4.0, KAPPA_MAX 0.2) come from `feasible_decode.py`; whether `P10Prior` holds such a limit is UNVERIFIED in the source. The human labels themselves violate the envelope at 0.2 s resolution on 2.9 % / 2.6 % of sampled records, so "exactly 0" is a property of the recovery phase only. Named next, not run: a curvature-bounded (Dubins or clothoid) recovery for the low-speed and large-heading states, and loss-masked partial-horizon labels for the no-logged-future states (a trainer flag and a G-DVB entry, so box-2 work).
* **`H-DD-FITGAP-M2B`: PARTIAL, both observed checkpoints FIT-LIMITED [M]** (T1-kin; the TRAIN side is M2b's own rollout states, 49618 GT-launch windows in 1891 clips; the eval side is the banked T1-kin dump of plain M2b on the 327 windows / 49 episodes, a reference and never a selector; cluster bootstrap, B = 2,000; the eval-side statistic reproduces the M-LAUNCH instrument to < 1e-9). The card asks whether the launch-speed deficit on the model's own TRAIN launch windows is at least half its size on eval-141 (r ≥ 0.5, FIT-LIMITED: no data-amount or data-selection lever can clear a crawl bar) or at most a quarter (r ≤ 0.25, a generalisation gap), at ≥ 5 of 6 checkpoints; only M2b s0 / s1, the checkpoints the banks hold, can be read now:

<<BK_FIT>>

  Both are FIT-LIMITED. Disclosed mismatch: the plain-M2b dumps ran `--limit logged` and the bank rollouts `--limit causal`; shifted by the DAG arm's measured logged-to-causal change (+0.2992 / +0.3049 m/s) the ratios read 0.950 / 1.047 (**[E]**, beside, no verdict). With 2 of 6 observed, GEN-GAP is unreachable under the card's 5-of-6 rule, and FIT-LIMITED needs 3 of DAG-OACC s0 / s1 and REP s0 / s1 (about 0.17 h each on a 10 % clip subset, **[E]** from a 290 s log). Reading: M2b crawls on its own training windows as much as on eval, so at M2b the crawl is not a data-coverage gap; the scope is the two M2b checkpoints, not the checkpoints of the later arms.
* **Reading across the six (H; decides nothing).** (i) The smaller crawl sits at the floor that a trigger-blind predictor shows, on the point; (ii) the output speed follows the input speed through the frozen prior, with nothing learned to remove; (iii) at M2b the deficit is as large on training windows as on eval; (iv) crawl states are 2.6–5.8 % of the bank and deficit weighting buys about 3 times; (v) a feasible recovery label reaches about 60 % of the tick-2 states against a predicted 78 %. Together they say that more data or re-weighted data of the same kind has a small ceiling for the crawl, and they leave one question open that none of the six asks: whether the go/stay trigger is in what the model sees (§7.31, §7.33).
* **What this section cannot claim.** It is not a lever result: nothing was trained. Every run contrast is one pair of training runs (`H-ESTIM-SEED-1`), and the s0 and s1 pairs differ (REP − DAG-OACC −0.556 at s0 against +0.020 at s1, **[I]**), so "REP sits below the floor" and "`S_v` differs by 0.11" are single-pair statements. The subsets are small (stopped launch 61 windows / 15 episodes; GT-stay 32 / 10), the table has one specification (the median-against-mean choice changes magnitudes, not the per-run outcome), the `H-CRAWL-02` counterfactual changes the speed channel only and is one forward call, and the FITGAP result covers 2 of the 6 checkpoints the card needs.
* **Next (Rule Zero).** The refutations here are waypoints: the diagnosis points at the trigger and at perception, so §7.31 measures whether the trunk's features carry the go/stay information and §7.33 whether finer DINOv3 tokens do. Every remaining lever and what blocks it: `H-CRAWL-03` (a trainer option that keeps the last k checkpoints, with its G-DVB entry; about zero box-2 cost in the next fine-tune); the other four FITGAP checkpoints (about 0.17 h each on the dev box, **[E]**; not run in this version); the curvature-bounded recovery label (zero training to build and test; not built) and the loss-masked partial-horizon label (a trainer flag, so box 2 and a PI go); regret-weighted DAgger draws (`H-ST-REGRET`: not run; the pre-step measures only the achievable multiplier, about 3 times).
* **Provenance [M].** Crawl package `R/raw/2026-10-28-crawl-diag/` (`RESULTS.md`, `CRAWL_DIAG.json`, `PREREG_FROZEN.md`, `READINGS.md`, `crawl_diag.py`, `test_crawl_diag.py`); bank package `R/raw/2026-10-28-bank-presteps/` (`MANIFEST.md`, `regret_prestep.md` / `.json`, `fitgap_m2b.md` / `.json`, `recovery/RECOVERY_RESULT.md` / `recovery_result.json`, `recovery/labeller.py` and `labeller_v2.py` as a prototype outside `stack/`, `FROZEN_CARDS.md`, `FROZEN_AMENDMENT_1.md`, `FROZEN_AMENDMENT_2.md`); claim rows `H-CRAWL-01`, `H-CRAWL-02`, `H-CRAWL-03`, `H-ST-REGRET-PRE`, `H-ST-RECOVERYLABEL-1`, `H-DD-FITGAP-M2B` in `Project Steering/GOALS_AND_CLAIMS.md`. The per-window arrays of the crawl package (`raw/crawl01_arrays.npz`, `raw/crawl02_arrays.npz`) and of the bank package (`raw/bank_table_s0.npz` and `_s1.npz`, ESTIMATED to rebuild in about five minutes of dev-box CPU) are in the landed packages; the exFAT allocation of the extracted per-episode dumps in the bank package is disclosed in its MANIFEST."""

# ============================================================================================================================================================
# ---- 7.31 trigger probes 1 and 2 ---------------------------------------------------------------------------------------------------------------------
# ============================================================================================================================================================
SEC_731 = """### 7.31 Trigger probes 1 and 2 (zero training): the trunk's go/stay edge is carried by the nav and speed-ceiling oracles, and the model's own four frames carry no motion signal

* **Status of this section (v34).** Two claim rows: `H-CRAWL-06-PROBE` (probe 1; package `R/raw/2026-10-28-trigger-probe/`, landed `c1c30fb7`, 2026-10-28 23:04:22) and `H-CRAWL-06-PROBE2` (probe 2; `R/raw/2026-10-29-trigger-probe2/`, landed `897b711b`, 2026-10-29 00:32:44; landing times **[M]** from the commits). They are precursors of card `H-CRAWL-06` of the crawl review (a loss-only stop/go head on the trunk: an architecture change that needs the PI). Zero training of the planner, dev-box CPU. Each card was frozen before data: probe 1 `PREREG_FROZEN.md` sha256 `0eb1c25c…` (stamped 2026-10-28T22:10:08), probe 2 `d9af82be…` (stamped 2026-10-28T23:11:02) (**[M]**, re-measured on the blobs). The question both ask, after §7.30: where is the go/stay trigger information lost, if the crawl is partly a trigger that the model does not use?
* **Tier and estimator [M].** T0-class probes of representations on logged inputs: they measure what a probe can decode from stored features, never what the planner does; nothing here is a statement about driving. Estimator: episode (clip) cluster bootstrap, B = 2,000, seed 0, on the mean of per-fold AUCs with the pooled out-of-fold AUC beside (§5.4 (d)). It answers *"would another draw of EPISODES say this?"* only; every Z arm rests on one checkpoint (DAG-OACC s0, sha256 `67265ec1…`), so it is blind to training variance (`H-ESTIM-SEED-1`) and, for a null control built on one permutation of the labels, to the draw of that permutation (§5.4 (e)).
* **Setup [M].** Windows are TRAIN windows of the canonical 306,566. GO is the registered launch mask (`v6 >= v0 + 1`, chord over the last 0.2 s of the 6 s horizon), read in two populations: P-STOP (`v0` < 0.5 m/s; full-population GO rate 57.4 %, 10,948 / 19,070 windows, 859 clips) and P-MOVE (0.5 ≤ `v0` < 6 m/s; 53.8 %, 38,670 / 71,930); 6,000 GO and 6,000 not-GO windows per population (849 and 2,140 clips), 5 episode folds. Arms, each scored by a linear probe with an MLP beside: E (the 8 numbers O reads), V (PCA-64 of the frozen DINOv3 `pool4x10` features of the current frame), V+E, Z (PCA-64 of the trunk state h_O of DAG-OACC s0) and Z+E. dAUC is the paired difference in AUC on the same windows. Probe 1's rule: D1 = information present but unused (dAUC(V+E − E) ≥ 0.05 with the interval excluding 0, and AUC(Z+E) ≥ AUC(V+E) − 0.02); D2 = the trunk discards it; D3 = the features lack it (dAUC < 0.02 for the linear and the MLP probe). Probe 2's rules: VISION-THROUGH-TRUNK if dAUC(Z_nn+E − E) ≥ 0.05 with the nav neutralised, NAV-ONLY if < 0.02; MOTION if dAUC(max(V4, Vdiff)+E − V+E) ≥ 0.03, none if < 0.01.
* **The metric (a post-freeze choice for probe 1, the primary of probe 2) [M].** Probe 1 froze the pooled out-of-fold AUC. Its control C2 (episode-shuffled labels) on P-STOP read 0.4629 [0.4267, 0.49993] and missed the committed "interval covers 0.5" by 7e-5, so the frozen pooled reading of P-STOP is **NOT EVALUABLE**. The unit test `test_foldmean_metric_unbiased_on_prevalence_shifted_null` shows why: the pooled out-of-fold AUC is biased below 0.5 on clip-clustered labels, because each fold's model carries an intercept fitted to the other folds' prevalence (§5.4 (d)). The mean of per-fold AUCs gives C2 0.4831 [0.465, 0.502] (PASS). The pooled metric also deflates the 8-number E more than V+E (E pooled 0.593, per-fold mean 0.626), so it inflates dAUC(V+E − E) on P-STOP (+0.087 pooled against +0.054 per-fold mean). The per-fold mean is marked as a post-freeze choice for probe 1; probe 2 froze it as primary.
* **Probe 1 outcome [M]** (all arms, pooled out-of-fold AUC [95 % interval], as frozen; the per-fold means are in `raw/results_p*.json`):

<<TP1_ARMS>>

  * **Reading.** P-MOVE, as frozen, all controls PASS: AUC E 0.731, V+E 0.784, Z+E 0.880; dAUC(V+E − E) +0.053 [0.041, 0.066], so the rule reads D1, but with a nonlinear ego baseline (the MLP) the vision gain is +0.016 [0.007, 0.025]. P-STOP: the frozen pooled reading is NOT EVALUABLE (above); the per-fold-mean reading gives dAUC +0.054 [0.025, 0.082] and also reads D1 (exploratory). The image increment is thin: on P-STOP the current-frame features alone (0.631) do not out-predict the 8 ego numbers (V alone minus E +0.007 [−0.030, 0.041], per-fold mean). On eval-141 (probes fitted on TRAIN, no refit; the committed memorisation test is not triggered) P-STOP (101 windows / 19 episodes, 61 GO) reads E 0.517 [0.41, 0.65], V+E 0.519 [0.29, 0.76], Z+E 0.655 [0.45, 0.85], and P-MOVE E 0.708, V+E 0.748, Z+E 0.804: the P-STOP increment is neither confirmed nor refuted (wide intervals).
  * **Strata, coarse (clip-level, VLM-derived traffic-light labels, disputed and untimed; none enters the rule).** The image gain over E on P-STOP is +0.099 on TL_RED clips (E 0.609 to V+E 0.708; Z+E 0.806), −0.058 on TL_GREEN clips and −0.019 where the lead vehicle departs by 2 m within 2 s. So the frozen features carry something about stopped-at-a-red-light windows and nothing measurable about a departing lead or a green light.
  * **A confound found after the freeze (a diagnostic, not a verdict).** Z is not "V plus a better trunk": h_O also reads the nav token and the speed-limit input, and four frames. Nav + limit + E alone reaches 0.719 pooled / 0.727 per-fold mean on P-STOP (E 0.593 / 0.626), so most of Z's go/stay predictability does not come from the image.
* **Probe 2, verdicts by the committed rules (TRAIN, linear, fold-mean, per population) [M].** The Z_nn row must be read with the arm 2 bullet below before it is quoted: a second ego-future input, the speed ceiling, is still in the trunk when only the nav is scrambled:

<<TP2_VERD>>

* **Probe 2, arm 1: what the nav and the "speed limit" input encode [M].** The nav token is built from the turns of the whole recording on the ego's own future path (`stack/scripts/build_v9_labels.py:233-260`), so it can say that the ego will move at some point and never when. On P-STOP the GO rate is 63.3 % when a turn lies ahead (70 % of the windows) and 43.8 % when none does (30 %); P(turn ahead | GO) is 77.0 % against 60.1 % for not-GO windows. The larger ego-future channel is the `limit` / `limit_ok` input. It is not a posted speed limit: it is the maximum of the ego's own realised speed over [t0 + 2 s, t0 + 6 s], one value per clip, quantised to 4 bins (a declared oracle, provenance ego-future; `stack/scripts/build_v8_speed_max_labels.py:16-19, 138-146`; the PI's 2026-09-10 ruling), so probe 1's "nav" diagnostic was mostly this input:

<<TP2_NAV>>

  The trunk therefore has two ego-future channels, and the speed ceiling carries about twice the go/stay information of the nav token (L+E − E +0.072 against N5+E − E +0.036 on P-STOP). Neither is a vision signal.
* **Probe 2, arm 2: the trunk with the nav scrambled, and with nav and ceiling scrambled [M].** The model has no null-nav path (no null-nav or nav-dropout code in the model, data, trainer or evaluator), so the nav is swapped for a donor window's nav (Z_nn); Z_nl scrambles the nav and the `limit` input (post-freeze). TRAIN fold-mean AUC, linear (MLP in brackets):

<<TP2_ZNN>>

  * **Frozen readings.** P-MOVE: dAUC(Z_nn+E − E) +0.117 [0.105, 0.129], VISION-THROUGH-TRUNK by the letter. P-STOP: +0.134 [0.110, 0.155], but control C2 read 0.4543 [0.418, 0.491], so P-STOP is NOT EVALUABLE as frozen (FAILED-CONTROL). Post-freeze (reported beside, never replacing the frozen reading): ten further label draws (seeds 3–12) give Z_nn+E P-STOP 0.5105 (sd 0.0226) [0.470, 0.544]; seed 2 is the minimum of the 11 draws (a 1 in 11 event) and the true arm, 0.760, exceeds the band maximum by 0.216, so the miss is a draw artefact and not a leak.
  * **Why the committed label over-reads.** Scrambling the nav alone removes only +0.040 (P-STOP) and +0.032 (P-MOVE) of Z+E; scrambling the speed ceiling as well removes a further 0.074 / 0.035 (the contributions interact, so these are two readings of one table, not a decomposition). With both scrambled the P-STOP trunk residual, Z_nl+E − E +0.060 [0.035, 0.083], is the size of the frozen features' own gain (V+E − E +0.054): Z_nl+E − V+E = +0.006 [−0.016, +0.027] (MLP +0.000). On P-MOVE a small gain remains on TRAIN windows (Z_nl+E − V+E +0.029 [0.021, 0.037], MLP +0.027). On eval-141 (unseen, no refit, single fit) Z_nn+E − E is +0.123 [−0.083, +0.290] (P-STOP, n = 101) and +0.075 [+0.005, +0.148] (P-MOVE, n = 507 / 67 episodes), and the scrambled-both trunk does not clear E: Z_nl+E − E +0.072 [−0.114, +0.227] / +0.019 [−0.040, +0.080], so the TRAIN residual is unconfirmed and may be memorisation (the checkpoint trained on these windows and their labels). The donor nav also makes the input inconsistent for a trunk trained only on consistent navs, so Z_nn can under-state a nav-free trunk.
* **Probe 2, arm 3: motion cues from the model's own four frames [M].** The frames are 0.2 s apart, so t−3 is 0.6 s back. V4 is the four frames concatenated (PCA-64) and Vdiff the difference V_t − V_{t−3}:

<<TP2_V4>>

  Both arms are within 0.01 of V+E or below it: **the model's own four cached frames carry nothing the current frame does not**, on TRAIN in both populations, with the nonlinear probe (P-STOP MLP −0.015 [−0.025, −0.005], the frames do worse), and on eval-141 (V4+E − V+E −0.003 [−0.028, +0.015] / +0.006 [−0.009, +0.022]). Caveats: the best-of-two selection is a winner's curse (V4 won by 0.038 / 0.050 over Vdiff, so it is immaterial here), and the concatenation's PCA is dominated by static content, which is why Vdiff is the motion-pure test and it is also null against V+E. Strata (below): no stratum in which V4 beats V+E.

<<TP2_STR>>

  Where the frozen features add nothing (green light, a departing lead), the trunk without the nav and the ceiling adds nothing either, and the trunk's gain there (Z_nn) comes through the oracle channels.
* **Controls [M].** Probe 1 (9 tests pass): label shuffle (C1) covers 0.5 (V+E 0.4999 [0.490, 0.509] on P-STOP), the leak control (C3: the label appended) reads 1.0000, and C2 passes on P-MOVE (0.5023 [0.485, 0.518]) and misses on P-STOP as above. Probe 2 (14 tests pass): C1 passes for E, N5+E, Z_nn+E, Z_nl+E, V4+E and Vdiff+E in both populations (0.492–0.507), C3 reads 1.0000 for the same arms, C2 passes for E, Z_nn+E on P-MOVE (0.5065 [0.490, 0.524]) and V4+E, and fails as frozen for Z_nn+E on P-STOP (above); N1: a fresh forward with the true nav reproduces the banked Z (max |diff| 1.9e-6 over 128 windows) and the swap harness with the donor set to the window itself reproduces it, with the emitted plan changing by exactly 0.0; N2: for all 109 of 128 windows whose donor nav differs from their own, h_O changes (median relative L2 0.141) and so does the plan (median max |Δ traj60| 0.59 m), and windows whose donor equals their own are bit for bit unchanged; the old arms E, V+E and Z+E re-fitted reproduce probe 1's eval-141 AUC with |diff| 0.0.
* **Conclusion, as the register states it [M for the numbers, H for the reading].** The trunk's go/stay edge beyond ego is carried by two ego-future inputs, the speed ceiling (the larger) and the nav token (the smaller), and not by vision. With both scrambled, the P-STOP trunk equals the level of the frozen features and the eval-141 gain is not distinguishable from 0 (P-MOVE keeps a small gain on TRAIN windows that the eval-141 interval does not confirm). The model's own four frames carry no motion signal beyond the current frame. The image gain over ego is concentrated on red-light clips (+0.099) and absent for green lights and departing leads. The go/stay trigger is weakly present in what v8F sees, so the lever is perception, and a loss-only stop/go head on this trunk would learn to read the oracle channels (a head that would score on the logged-input T0 probe and buy no deployable skill).
* **The four families (STRATEGIC reported, not paused, for v8F).** These are representation probes, not evaluations of a plan: LONGITUDINAL, LATERAL, TACTICAL and STRATEGIC rows are NOT RUN here (no plan is scored), and the probes are not a statement about any of the four. The go/stay label is a longitudinal event (a launch); the strata that matter are light state and lead departure.
* **What this section cannot claim.** Every Z statement rests on one checkpoint (DAG-OACC s0) and on TRAIN windows that checkpoint was trained on; eval-141 P-STOP has 101 windows from 19 episodes; the traffic-light strata are clip-level and VLM-derived; the TRAIN cache and the rebuilt TRAIN join are **UNVERIFIED** byte for byte against box 2's copies (`D:\\tanitad\\cache_full`, `join_train.npz`); whether a nav-free trunk trained without the oracle inputs would behave like Z_nn or Z_nl is **UNVERIFIED**; the nav and the ceiling are ego-future channels on this corpus, so any go/stay skill measured with them in the input is optimistic by construction (the goal-input rule).
* **Next (Rule Zero).** Named, cheapest first, none run in this version. (1) Evaluate the deployed planner on the GO windows with the nav and the ceiling set to donor values (the Z_nl scramble at the planner output, plan error on the 327 launch windows): zero training, and the only experiment that gives the launch skill without ego-future inputs (the probes measure representations, not plans); it decides whether the crawl is a perception defect or an oracle-dependence defect. (2) A targeted perception lever (light state, lead gap or time-to-collision at higher resolution or from a region crop), not temporal pooling: §7.33 reads finer DINOv3 tokens. (3) A stop/go head is not the lever until the oracle inputs are out of the loop. Open PI decision, sharper than before: whether GO skill measured through the nav counts at all, since the nav is an oracle on this corpus.
* **Provenance [M].** Probe 1 `R/raw/2026-10-28-trigger-probe/` (`RESULTS.md`, `PREREG_FROZEN.md`, `FROZEN_SHA.txt`, `raw/results_pmove.json`, `raw/results_pstop.json`, `raw/results_strata.json`, `raw/results_nav_postfreeze.json`, `run_probe.py`, `decide.py`, `test_trigger_probe.py`); probe 2 `R/raw/2026-10-29-trigger-probe2/` (`RESULTS.md`, `PREREG_FROZEN.md`, `FROZEN_SHA.txt`, `raw/VERDICT.json`, `raw/POSTFREEZE_LOG.md`, `raw/tables2.txt`, `decide2.py`, `test_trigger_probe2.py`); claim rows `H-CRAWL-06-PROBE` and `H-CRAWL-06-PROBE2` in `Project Steering/GOALS_AND_CLAIMS.md` (the second refines the first)."""

# ============================================================================================================================================================
# ---- 7.32 SOUP ---------------------------------------------------------------------------------------------------------------------------------------
# ============================================================================================================================================================
SEC_732 = """### 7.32 SOUP: a uniform two-draw weight average is not adopted (HARMS); the soup gain is real, there is no barrier, and no crawl row moves

* **Status of this section (v34).** Read 2026-10-29, zero training: every member checkpoint is banked from the LAUNCH-INTENT and DAG-OACC runs. Claim row `H-V8F-SOUP-1`; `R/RESULT_V8F_SOUP.md`; pre-registration `Project Steering/PREREG_V8F_SOUP.md`, promoted at `e03b348` (2026-10-29 02:00:06, **[M]** from the commit; sha256 `830d9c91…`, **[M]** on the tip's blob; status line and section 0 item 5 changed at promotion, nothing else, **[I]** from the RESULT). The pre-registered reader `soup_read.py` (sha256 `de39aefe…`) ran with all **368 of 368** controls PASS (wall 634.9 s, rc 0); its output `SOUP_READ.json` (sha256 `dca10c07…`, **[M]** on the tip's blob) was banked at `0c7b598` (2026-10-29 03:10:30, **[M]** from the commit). The registered word is **HARMS**, the soup is not adopted, and the committed next is SOUP-K.
* **Tier and estimator [M].** T1-kin (self-action open loop: m = 10, causal speed bucket, dead-reckoned nav, logged scene, periodic replan, executor plan) unless a cell says T0; the R-T0 points and the slow-draw cells are T0 (teacher-forced own-code, dev-box CPU on both arms of a pair). Paired episode-cluster bootstrap, B = 2,000, seed 0, 141 episodes / 2,399 windows, per seed and per pair, on the same windows and matched cells. Intervals are 95 % (**[I]**: the convention of the LAUNCH-INTENT RESULT; the JSON stores the bounds only). The bootstrap answers *"would another draw of EPISODES say this?"* and nothing else; *"would another TRAINING RUN say this?"* is answered only by the replicate pair at each seed and by the second seed (`H-ESTIM-SEED-1`). The planner is deterministic and a soup is a fixed set of weights, so the third variance (inference) does not arise.
* **The lever and the arms [M].** A uniform average of the weights of two DAG-OACC-recipe O fine-tunes that share the init M2b_s, computed in float64 and rounded once to float32 (§5.4 (c)). SOUP_s = DAG-OACC_s + REP_s (donor DAG-OACC_s); SOUP'_s = A_s + A_rep,s (donor A_s; both members carry the launch-intent weight 1.58926); SOUP4_s = all four, beside. The control of the main pair is DAG-OACC_s, of the replicate pair A_s, so each soup is read against its trainer-seed-s member; a soup that is better than REP_s and worse than DAG-OACC_s is not an improvement of the base. The PREREG's first-order prediction for SOUP_0 − DAG-OACC_0 was about +0.20 m, worse (**[E]**, half of the member spread +0.4025 m).
* **The outcome table (T1-kin; X = the soup of the pair, Y = its control; bold = separated; B = separated-better, W = separated-worse).** The condition for HARMS, verbatim from PREREG section 10: *"controls pass; H fails at ≥ 1 seed"*. H fails at s0 (R-ADE in both pairs, R-TLAT in the main pair); the registered rule needs H at both seeds, so the s1 pass does not change the word:

<<SO_OUT>>

  Per-seed booleans (`OUTCOME.per_seed`; each holds at a seed only when it holds in both pairs of that seed):

<<SO_PS>>

  At s1 alone every guard and both adoption rows hold (H and AD true), the lever still moves no crawl row (MV false) and does not reach P10 on the crawl rows (DM false). The word is not taken seed by seed.
* **ADE6, the guard R-ADE and the adoption row R-KEEP (T1-kin) [M].** R-ADE needs X − Y not separated-worse; R-KEEP reads the same contrast's point against +0.0604 m (F/2):

<<SO_ADE>>

  R-ADE and R-KEEP fail in both pairs at s0 and hold in both at s1; at s1 the replicate pair is separated-better. The tactical guard and the dev-box T0 point (the plan-derived agreements beside the reader's booleans; they are not the statistic the guards test):

<<SO_GRD>>

* **The soup gain is real and there is no barrier [M].** SG on ADE6 is the soup's error minus the mean of its members' errors, per window (§5.4 (c)). SG < 0 separated says the average beats the average of its members' errors; SG > 0 separated is a barrier:

<<SO_SG>>

  Five of the six cells are separated-better (SOUP at s1 is not separated), no cell is a barrier, so the route is **FLAT** (the weights average at least as well as the metrics; the shortfall is the members' draws) and the committed next is SOUP-K. SG is a mean-of-errors statistic: it does not say the soup beats the better member.
* **At s0 the soup recovers most of the slow draw and still lands worse than the good one [M for the cells, DERIVED for the shares].** For a two-member soup that contains its control, soup − control = SG + Δ/2 with Δ = (slow member − control) (§5.4 (c)). At s0 SG −0.1242 plus Δ/2 = 0.2013 gives +0.0771 (the table's +0.0770; the one-unit difference is the rounding of the four-decimal cells), at s1 −0.0163 plus 0.0058 gives −0.0105 (the table's −0.0105; DERIVED). The recovered share of the member spread is 1/2 − SG/Δ = 80.9 % at s0 (DERIVED; the RESULT's table reads 80.9 % and 79.8 %):

<<SO_F2>>

  A linear interpolation of the errors puts a two-member soup at 50 % of the way; the measured point sits 19.1 % and 20.2 % of the way from the good to the slow member on ADE6, so the average does better than interpolation and is still separated-worse than the good draw. The same holds on the four crawl rows:

<<SO_F2B>>

  On every crawl row the soup lies between its members, at 23.1 to 49.2 % of the way from the good to the slow member, and is separated-worse than the good member on all four rows in both pairs (averaging recovers 50.8 to 76.9 % of the slow draw's excess). At s1 the members agree (REP_1 − DAG-OACC_1 ADE6 +0.0116 [−0.0531, +0.0765], not separated), so there is no draw-specific crawl for the average to remove and the soup sits with them.
* **The crawl rows: no row moves by its bar and none reaches P10 [M].** R-MOVE needs, in each pair, every crawl row separated-better by at least its bar; the bars are 2 × max(F, RF0) = 1.1070 (yaw-rate), 0.0526 (curvature), 0.0136 (L2 curvature, GT ≥ 5.2) and 0.0096 (M1 crawl share), unchanged from the LAUNCH-INTENT read. Improvement is − delta (all four rows are lower-is-better):

<<SO_MOVE>>

  0 of the 16 cells are separated-better and 0 clear their bar; 9 are separated-worse (8 at s0). R-CRAWL is false in all four pairs: all 16 X − P10 crawl cells are separated-worse, as are the control's Y − P10 cells in every pair (that is how the crawl set was fixed). Read A reads MARGIN, NOT DOMINANT for SOUP, SOUP' and SOUP4 at both seeds: D1 (ADE6 better than P10 by at least 2F) and D3 (each family has a separated-better row) hold, D2 fails on the same four crawl rows.
* **The four families at T1-kin (per family, per seed, per pair, never pooled; STRATEGIC reported, not paused for v8F).** Cells are X − Y with the estimator above. Separated cells per family:

<<SO_F41>>

  LONGITUDINAL (a lower value is better; 2,399 windows / 141 episodes except the distance-keeping rows):

<<SO_LON>>

  LATERAL, read first because the crawl rows are LATERAL plus M1:

<<SO_LAT>>

  The M1 crawl share is a crawl row outside the LATERAL block of the JSON:

<<SO_M1>>

  TACTICAL (the plan-derived agreements; a higher value is better):

<<SO_TAC>>

  STRATEGIC (S-POS error, lower is better; the cell is the difference of the two runs' contrasts against CTRV on the same windows):

<<SO_STR>>

  * **Reading.** At s0 the soup is worse than its control on LONGITUDINAL speed, along-track and acceleration, on five of six LATERAL rows and on one STRATEGIC row (main pair), and in the main pair its only separated-better cell is cross-track (−0.0333). At s1 the main pair shows no separated-better LONGITUDINAL or LATERAL row, and the replicate pair is separated-better on four LONGITUDINAL and two LATERAL rows; two STRATEGIC rows are separated-better in the main pair (S-POS@50 −0.4784, S-POS@80 −0.6932). The separated cells flip sign between seeds and between same-seed pairs, which is the rig's own run-to-run noise (`H-ESTIM-SEED-1`), so no family row is read as a lever effect.
  * **Coverage, per family, with the reason.**

<<SO_COV>>

* **Reading (HYPOTHESIS; decides nothing).** The crawl rows carry two parts. One is draw-specific: at s0 the slow draws are separated-worse than the good draws on all four crawl rows and averaging recovers 50.8 to 76.9 % of that excess, so the PREREG's hypothesis (part of the crawl-row error of one draw is draw-specific, and an average keeps the shared part and halves the draw-specific part) is supported in direction at s0. The other part is shared by every member: both controls are separated-worse than P10 on those rows at both seeds, and the soup, lying between its members, sits above the floor as well. Averaging, and by the same arithmetic more draws of the same recipe (SOUP-K), can shrink the draw-specific part and cannot move the shared one. Read together with §7.30 (the smaller crawl sits at the outcome-blind floor on the point, the speed sensitivity is the frozen prior's, the M2b checkpoints measured are FIT-LIMITED on TRAIN) and §7.31 (the trunk's go/stay edge is the declared oracle channels and the frozen features' increment is thin), the lever the evidence points to is **perception**. The words *"perception is the lever"* are a reading of diagnostics, each measured on other arms and checkpoints; they are not a result of this read.
* **Beside, deciding nothing [M].** The slow-draw speed cells at T0 (dev-box, one control per pair): at s0 the soup is slower than its control in both pairs (SOUP_0 − DAG-OACC_0: lon_v_target −0.6143 [−0.7063, −0.5211] m/s; SOUP'_0 − A_0: −0.6880 [−0.7841, −0.5974] m/s) and at s1 every cell is within 0.15 m/s of 0. The reader's `threshold_m_s` is 0.5 and its label is null in all four entries; the labelling rule is not re-read here (**UNVERIFIED**), so no label is claimed.
* **What this section cannot claim.** (1) A separated interval is necessary, not sufficient (`H-ESTIM-SEED-1`): a soup is two training draws inside one number, and with two seeds the read cannot say how often a seed like s0 occurs. (2) A soup effect is the effect of averaging these two draws: N = 2 per decisive soup, and nothing here says a soup of K > 2 behaves the same (SOUP4 is beside and overlaps both pairs). (3) The control is asymmetric by design (above). (4) The replicate soup carries the launch-intent lever on both members, so its result attributes nothing to the launch weight. (5) R-T0 is a dev-box CPU T0 reading on both arms of the pair, not box 2's standard eval. (6) The T0 four families are **NOT RUN** in this read (the reader carries only the R-T0 points and the slow-draw cells); the T1-kin goal-head F1 and head accuracy are **NOT COMPUTABLE** here (the paired tool reads eval dumps, not tick dumps); the anti-echo block is **NOT BUILT** (the DAG-ORC rig carries none); strategic decision and goal / route-setting quality has no row in the JSON, and whether another tool computes it for these arms is **UNVERIFIED**. (7) No bar, row, member set, donor rule, K-BASIN factor, routing variable or outcome changed after any soup row was seen (**[I]**, the RESULT's statement).
* **Disclosures (from the RESULT, section 9).** (a) The JSON's `provenance.prereg` reads *"PREREG_V8F_SOUP.md (DRAFT, 2026-10-28)"* while the sha256 it checked is the promoted file's; the JSON carries no control named PREREG-TEXT (its static controls are one boolean, `controls_PASS.STATIC`, true), and whether PREREG-TEXT sits inside it is **UNVERIFIED** here. (b) The launcher `soup_chain.cmd` differs from the shipped copy in exactly two PREREG lines, **[M]** from the two banked copies. (c) The register row calls the −0.0125 cell *"TAC-LAT macro-F1"*; in the JSON that cell is the TAC-LAT row, a plan-derived agreement, and the guard R-TLAT is a boolean whose macro-F1 value is not carried; this section follows the RESULT. (d) The per-window dumps and tick arrays are single copies under `F:/TanitAD-scratch/soup_t1_out/`, and the six soup checkpoints (29.0 MB each, `opt` None) are on E: only. (e) `check_result.py` re-finds every table cell and quoted string in the banked JSON and recomputes the decision from the cells; it does not recompute the cells from the per-window dumps or re-run the bootstrap, so a defect inside the estimator or the reader's heavy layer would not show there. (f) Two items are **[I]** from the RESULT's brief: the SOUP finishing agent deleted its own test directories in the session scratchpad (the rule is to move, not delete), and 46 old scratchpad folders (about 32 GB) were moved to F: before launch.
* **Next (Rule Zero).** Committed, as registered (route FLAT, PREREG section 11): **SOUP-K**, K = 4 pure-recipe draws per seed (DAG-OACC_s's argv at trainer seeds 20 + s and 30 + s added to DAG-OACC_s and REP_s), 4 new box-2 runs, about 2.8 h (**[E]**, from the LAUNCH-INTENT runs), read under the same table; it needs its own PI go and is not launched. The soup read shows why more draws of the same recipe are the weaker lever: they can shrink the draw-specific part of the crawl and cannot move the shared part. Every remaining lever and what blocks it (RESULT section 7.3):

<<SO_BLK>>

  The finer-token DINOv3 probe that the RESULT names PROPOSED, not registered, is read in §7.33.
* **Provenance [M].** The sha256 below are of the bytes at the RESULT's tip `0c7b598e`, as the RESULT records them (the paths are under `R/raw/` unless they start with `Project Steering/`); the audit re-measures the JSON, the PREREG and the RESULT on this report's tip. The chain started 02:19:45 and its last step ended 02:56:08; 28 steps ran, each at attempt 1, the reader started 02:56:28 and the chain closed at 03:07:03 with *"ZZSOUP-DONE 28/28 steps, reader rc 0"* (**[I]**, the RESULT's run record).

<<SO_PROV>>"""

# ============================================================================================================================================================
# ---- 7.33 fine-token probes 3 and 4 (marked slots in the pre-draft) --------------------------------------------------------------------------------------
# ============================================================================================================================================================
SEC_733 = ("### 7.33 Fine-token probes 3 and 4: " + site("TITLE733", "outcome not read in this version") + """

* **Status of this section (v34).** """ + site("STATUS733", "Probes 3 and 4 were still running when this pre-draft was built, so no probe 3 or probe 4 number is read in this version (**[I]**).") + """
* **Why these probes [M].** Probe 1 (§7.31) found that the frozen current-frame features add a thin go/stay increment over ego (P-MOVE dAUC(V+E − E) +0.053 [0.041, 0.066] as frozen, MLP +0.016 [0.007, 0.025]), concentrated on red-light clips (+0.099) and absent on green-light and lead-departure clips; probe 2 found no motion cue in the model's own four frames. Both read the pooled `pool4x10` features. The SOUP RESULT (§7.32) names the next step: **a finer-token DINOv3 probe** of the go/stay trigger, PROPOSED and not registered, whose question is whether DINOv3 tokens finer than the pooled `pool4x10` features carry the go/stay trigger beyond ego; it states *"Which clips, which probe family and which outcomes is not decided here."* and that it needs its own PREREG with both outcomes committed before any data. Probe 2's own next step points the same way: a targeted perception lever (light state, lead gap or time-to-collision at higher resolution or from a region crop), not temporal pooling. This section holds probes 3 and 4, the two probes that follow; their cards, tiers, estimators and outcomes are in the bullets below.
""" + site("SETUP733", "* **Setup.** The setup of probes 3 and 4 (cards, token grids, populations, arms, committed outcomes, tier and estimator) is not read in this version.") + "\n"
    + site("OUTCOME733", "* **Outcome.** Not read in this version.") + "\n"
    + site("CONTROLS733", "* **Controls.** Not read in this version.") + "\n"
    + site("FAMILIES733", "* **The four families (LONGITUDINAL, LATERAL, TACTICAL, STRATEGIC).** Not read in this version; STRATEGIC is reported, not paused, for v8F.") + """
* **What this section cannot claim (any probe of this family).** A probe of stored features measures what can be decoded from logged inputs: it is a statement about representations and says nothing about a plan. The nav token and the speed ceiling are ego-future channels on this corpus (§7.31), so a go/stay skill measured with them in the input is optimistic by construction. A probe fitted and scored on TRAIN windows of a checkpoint that trained on them may be memorisation, and eval-141 is the unseen read. Any further limit of probes 3 and 4 (checkpoints, windows, strata, tokens) is stated in the setup and outcome bullets above.
""" + site("NEXT733", "* **Next (Rule Zero).** The committed next of probes 3 and 4 is not read in this version.") + "\n"
    + site("PROVENANCE733", "* **Provenance.** Not read in this version."))

SEC = SEC_730 + "\n\n" + SEC_731 + "\n\n" + SEC_732 + "\n\n" + SEC_733
for name, rows in TAB.items():
    tok = "<<" + name + ">>"
    if tok in SEC:
        assert SEC.count(tok) == 1, tok
        SEC = SEC.replace(tok, "\n".join(rows))
_left_tok = re.findall(r"<<[A-Z0-9_]+>>", SEC)
assert not _left_tok, _left_tok
assert "<<" not in SEC and ">>" not in SEC.replace("→", "")
_unused = [n for n in TAB if ("<<" + n + ">>") not in (SEC_730 + SEC_731 + SEC_732 + SEC_733)]
assert not _unused, ("tables extracted and not used", _unused)

# ---- numbers quoted in the abstract, 8.x and the rows: each is present in a source blob ----------------------------------------------------------------------
NUMS = [
    (REG, "-1.034 [-1.298, -0.774]"), (REG, "-0.800 [-1.071, -0.529]"), (REG, "-0.875 [-1.143, -0.601]"), (REG, "+0.159 [-0.136, +0.468]"), (REG, "-0.075 [-0.377, +0.232]"),
    (REG, "-0.397 [-0.777, +0.010]"), (REG, "-0.631 [-1.008, -0.225]"), (REG, "+0.736 [+0.593, +0.829]"), (REG, "+0.686 [+0.557, +0.768]"), (REG, "+0.050 [+0.015, +0.090]"),
    (REG, "5.752 % [5.439, 6.062]"), (REG, "2.608 % [2.443, 2.782]"), (REG, "2.68 [2.60, 2.77]"), (REG, "3.08 [2.99, 3.18]"), (REG, "1.229 [0.976, 1.680]"), (REG, "1.612 [1.023, 4.416]"),
    (REG, "+0.053 [0.041, 0.066]"), (T2, "+0.006 [-0.016, +0.027]"),
    (CR, "-0.556 [-0.699, -0.398]"), (BC, "59.5 % / 62.0 %"), (SO, "+0.0770 [+0.0213, +0.1348]"), (SO, "+0.0959 [+0.0321, +0.1624]"), (SO, "-0.1242 [-0.1517, -0.0977]"),
    (SO, "Which clips, which probe family and which outcomes is not decided here."),
    (T1, "0.4629 [0.4267, 0.49993]"), (T2, "-0.003 [-0.028, +0.015]"), (T2, "+0.006 [-0.009, +0.022]"),
]
_nm = [x[:60] for src, x in NUMS if x not in src.replace("−", "-")]
assert not _nm, ("a number or quotation is not in its source", _nm)

# ---- the abstract, the status line and the edits -------------------------------------------------------------------------------------------------------------
OLD_ST = "research report, v33 of 2026-10-28, results through repository tip `8c1e9636` (v32: `ea5e3d3b`; v31:"
NEW_ST = "research report, v34 of 2026-10-29, results through repository tip `" + TP + "` (v33: `8c1e9636`; v32: `ea5e3d3b`; v31:"
OLD_ST2 = "the LAUNCH-INTENT read (§7.29, HARMS: arm A, W 1.58926, is not adopted; M-LAUNCH reads NOT MOVED)."
NEW_ST2 = (OLD_ST2 + " v34 adds the crawl diagnosis and the DAgger-bank pre-steps (§7.30, zero training), trigger probes 1 and 2 (§7.31), the SOUP read (§7.32, HARMS: the two-draw weight average is not adopted),"
           " a background section on the statistics those reads use (§5.4), three related-work paragraphs and " + site("LINE", "the fine-token probes 3 and 4 (§7.33, not read in this version)."))

_a0 = s.index("* **What is next.** The open gap is the crawl set (yaw-rate, curvature, L2 and M1 against P10 for DAG-OACC); the adopted base stays DAG-OACC.")
_a1 = s.index("\n\n---\n\n## 2. Introduction", _a0)
OLD_ABS = s[_a0:_a1]
assert OLD_ABS.rstrip().endswith("which waits on the PI."), OLD_ABS[-80:]
NEW_ABS = (
    "* **The launch crawl was diagnosed with zero training (§7.30; `H-CRAWL-01` to `-03`, `H-ST-REGRET-PRE`, `H-ST-RECOVERYLABEL-1`, `H-DD-FITGAP-M2B`; cards frozen before data, dev-box CPU, **[M]**).**"
    " On the 327 GT-launch windows (49 episodes; T1-kin; episode-cluster bootstrap, B = 2,000, seed 0) a predictor that is blind to the outcome and built on TRAIN reads a gap of the same order as the model's:"
    " floor median −1.034 [−1.298, −0.774] m/s (mean −0.800 [−1.071, −0.529]) against DAG-OACC s0's −0.875 [−1.143, −0.601], so the model minus the floor is +0.159 [−0.136, +0.468] (median) and −0.075"
    " [−0.377, +0.232] (mean): at the floor on the point (interval half-width 0.30). The slower run REP s0 sits below it (−0.397 [−0.777, +0.010] median, −0.631 [−1.008, −0.225] mean). The model's input-speed"
    " sensitivity S_v is the frozen prior's: +0.736 [+0.593, +0.829] for DAG-OACC, the prior alone +0.686 [+0.557, +0.768], the learned residual +0.050 [+0.015, +0.090]. Whether the run-to-run difference is"
    " between-run or within-run noise is NOT EVALUABLE (no intermediate checkpoint exists). The DAgger-bank pre-steps read a middle band: crawl states are 5.752 % [5.439, 6.062] (s0) and 2.608 % [2.443, 2.782]"
    " (s1) of the usable bank records, and weighting by deficit buys about 3 times (2.68 [2.60, 2.77] and 3.08 [2.99, 3.18], below the 5-times branch); a feasible recovery label keeps 53.85 to 59.51 % (s0)"
    " and 54.69 to 61.95 % (s1) of the tick-2 states against a predicted 75 to 81 % (NOT CONFIRMED); the fit-gap ratio is 1.229 [0.976, 1.680] and 1.612 [1.023, 4.416] on 2 of the card's 6 checkpoints (both"
    " FIT-LIMITED, PARTIAL).\n"
    "* **Trigger probes 1 and 2 (§7.31; T0-class probes of representations on logged inputs; episode-cluster bootstrap on the mean of per-fold AUCs, B = 2,000; one checkpoint, DAG-OACC s0; zero training, **[M]**).**"
    " The frozen current-frame features add a thin go/stay increment over ego on P-MOVE (dAUC(V+E − E) +0.053 [0.041, 0.066] as frozen, MLP +0.016 [0.007, 0.025]); the frozen pooled reading on P-STOP is NOT"
    " EVALUABLE (control C2 0.4629 [0.4267, 0.49993]), and the pooled out-of-fold AUC is biased below 0.5 on clip-clustered labels (§5.4 (d)). The trunk's go/stay edge beyond ego is carried by two ego-future inputs,"
    " the declared speed ceiling (the larger) and the nav token (the smaller): with both scrambled the P-STOP trunk equals the frozen features (Z_nl+E − V+E +0.006 [−0.016, +0.027]) and the unseen eval-141 gain is"
    " not distinguishable from 0. The model's own four frames add nothing (V4+E − V+E −0.003 [−0.028, +0.015] and +0.006 [−0.009, +0.022] on eval-141). The image gain over ego is concentrated on red-light clips"
    " (+0.099) and absent for green lights and departing leads. A loss-only stop/go head on this trunk would learn the oracle channels, so the lever is perception.\n"
    "* **SOUP reads HARMS: the uniform two-draw weight average is not adopted (§7.32; T1-kin; paired episode-cluster bootstrap, B = 2,000, 141 episodes, per seed and per pair; all 368 reader controls PASS; zero training, **[M]**).**"
    " The guards fail at s0 in both pairs (ADE6 SOUP_0 − DAG-OACC_0 +0.0770 [+0.0213, +0.1348] m, SOUP'_0 − A_0 +0.0959 [+0.0321, +0.1624] m, both separated-worse) and hold at s1; the registered rule needs them at both seeds."
    " R-MOVE and R-CRAWL are false in all four pairs: 0 of the 16 crawl cells are separated-better, 9 are separated-worse, and all 16 stay separated-worse against P10. The soup gain on ADE6 is real"
    " (SG −0.1242 [−0.1517, −0.0977] at s0; five of six cells separated-better) and no cell is a barrier, so the route is FLAT. At s0 the soup recovers 80.9 % of the slow draw's ADE6 excess"
    " and still lands worse than the good draw: averaging removes the draw-specific part of the crawl and keeps the part every member shares.\n"
    + site("ABSTRACT", "* **Fine-token probes 3 and 4 (§7.33).** Not read in this version: the probes were still running when this pre-draft was built.") + "\n"
    "* **What is next.** The crawl set is still open and the adopted base stays DAG-OACC. The zero-training reads (§7.30 to §7.32) say that more data or re-weighted data of the same kind has a small ceiling (deficit"
    " weighting about 3 times; a feasible recovery label reaches about 60 % of the tick-2 states), that the go/stay trigger is weakly present in the frozen features, and that averaging draws shrinks only the"
    " draw-specific part of the crawl, so the lever the evidence points to is perception (a **[H]** reading of diagnostics measured on other arms and checkpoints). Committed next, as registered: SOUP-K (4 box-2 runs,"
    " about 2.8 h, **[E]**) needs its own PI go. LAUNCH-INTENT's arm B and LAUNCH-INTENT-HI (v33) are carried from v33, each with its own PI go; their state is not re-read in v34 (**UNVERIFIED**). The history-aware"
    " causal read-out stays registered and unrun (§7.28), and the speed line that SPEEDREAD (`PREREG_V8F_SPEEDREAD`) asked about routes to perception (`PREREG_V8F_LORA` on the A40), which waits on the PI."
    " §7.33 reads the next perception step, the fine-token probes.")

UPD729 = ("* **Update (v34).** The crawl set that this lever aimed at was diagnosed afterwards with zero training (§7.30): the smaller launch crawl sits at the outcome-blind floor on the point, and the model's input-speed"
          " sensitivity is the frozen prior's. A uniform two-draw weight average of the DAG-OACC-recipe fine-tunes then did not move the crawl rows either (§7.32, HARMS). Neither read changes this section's word or its numbers.\n")

OLD_811 = "Both B (registered after HARMS) and HI (the §11 branch) need their own PI go.\n\n### 8.2 What is not established"
NEW_811 = ("Both B (registered after HARMS) and HI (the §11 branch) need their own PI go.\n"
           "18. **The smaller launch crawl sits at the floor of a predictor that is blind to the outcome, and the speed sensitivity belongs to the frozen prior** (§7.30; the gaps are T1-kin, the counterfactual is T0-class;"
           " episode-cluster bootstrap, B = 2,000, seed 0; 49 episodes / 327 launch windows). **Established (MEASURED):** for DAG-OACC s0 the model minus the outcome-blind floor is +0.159 [−0.136, +0.468] (median table)"
           " and −0.075 [−0.377, +0.232] (mean table) m/s, inside the pre-registered ±0.2 on the point only; the run-to-run launch difference REP − DAG-OACC is −0.556 [−0.699, −0.398] m/s; S_v is +0.736 and +0.628, of"
           " which the frozen prior alone is +0.686; the three DAgger-bank pre-steps read a middle band (deficit weighting about 3 times), a recovery label that keeps 59.5 / 62.0 % of the tick-2 states against a predicted"
           " 75 to 81 %, and FIT-LIMITED on 2 of 6 checkpoints. **Not established:** that the crawl is a trigger problem rather than a fit problem (these reads do not separate them), anything about the other four"
           " checkpoints of the fit-gap card, and whether the 0.556 m/s run difference is between-run or within-run noise (NOT EVALUABLE). One checkpoint pair, no replicate arm (`H-ESTIM-SEED-1`).\n"
           "19. **The trunk's go/stay edge beyond ego is carried by two ego-future inputs, and the frozen features add a thin increment** (§7.31; T0-class probes of representations on logged inputs; episode-cluster"
           " bootstrap on the mean of per-fold AUCs, B = 2,000; one checkpoint). **Established (MEASURED):** P-MOVE dAUC(V+E − E) +0.053 [0.041, 0.066] as frozen (MLP +0.016 [0.007, 0.025]); the frozen pooled P-STOP reading"
           " is NOT EVALUABLE (C2 0.4629 [0.4267, 0.49993]); with the nav and the speed ceiling both scrambled the P-STOP trunk residual is Z_nl+E − V+E +0.006 [−0.016, +0.027]; the model's own four frames add nothing."
           " **Not established:** that a nav-free trunk would behave like Z_nl (UNVERIFIED), any statement about a plan (representation probes only), and the TRAIN-window residual on unseen windows.\n"
           "20. **A uniform two-draw weight average does not move the crawl rows and is not adopted: HARMS** (§7.32; T1-kin, paired episode-cluster bootstrap, B = 2,000, 141 episodes, per seed and per pair; all 368 reader"
           " controls PASS). **Established (MEASURED):** R-ADE fails at s0 in both pairs (ADE6 +0.0770 [+0.0213, +0.1348] and +0.0959 [+0.0321, +0.1624] m) and holds at s1; R-MOVE and R-CRAWL are false in all four pairs"
           " (0 of 16 crawl cells separated-better, 9 separated-worse); the soup gain on ADE6 is separated-better in five of six cells with no barrier, so the route is FLAT; at s0 the soup recovers 80.9 % and 79.8 % of"
           " the slow draw's ADE6 excess and still lands worse than the good draw. **Not established:** an effect of averaging more than two draws (SOUP-K not run), anything at T0 across the four families (NOT RUN), and"
           " the T1-kin tactical goal-head F1 (NOT COMPUTABLE). N = 2 per decisive soup; the s0 / s1 difference is the slow draw.\n"
           "21. " + site("R81", "**Whether DINOv3 tokens finer than the pooled features carry the go/stay trigger is probed in §7.33** (probes 3 and 4). No probe 3 or probe 4 number is read in this version, so nothing is established about it.")
           + "\n\n### 8.2 What is not established")

OLD_83 = "### 8.3 The gap to \"dominant driving results\"\n\n**Update (v33).** Three items are written up here:"
NEW_83 = ("### 8.3 The gap to \"dominant driving results\"\n\n"
          "**Update (v34).** Four items are written up here, and the adopted base is unchanged (DAG-OACC):\n"
          "* **The launch crawl diagnosis (§7.30, zero training) reads:** the smaller crawl is at the outcome-blind floor on the point (model − floor +0.159 [−0.136, +0.468] / −0.075 [−0.377, +0.232] m/s, T1-kin, 327 launch"
          " windows), S_v +0.736 of which the frozen prior is +0.686, deficit weighting buys about 3 times and a recovery label keeps about 60 % of the tick-2 states. More data of the same kind has a small ceiling for the crawl;"
          " the question none of the six asks is whether the trigger is in what the model sees.\n"
          "* **Trigger probes 1 and 2 (§7.31, T0-class representation probes, one checkpoint) read:** the frozen features add a thin go/stay increment over ego (P-MOVE +0.053 [0.041, 0.066]), the trunk's edge is the declared"
          " speed-ceiling and nav oracles, and the model's own four frames add no motion cue. The lever is perception, not a stop/go head on this trunk.\n"
          "* **SOUP (§7.32) reads HARMS** (T1-kin; 368 of 368 controls PASS): the guards fail at s0 in both pairs (ADE6 +0.0770 / +0.0959 m, separated-worse), no crawl row moves by its bar, and all 16 X − P10 crawl cells stay"
          " separated-worse. The soup gain is real with no barrier (route FLAT). The crawl set is still open and the committed next, SOUP-K, needs the PI's go.\n"
          "* " + site("R83", "**Fine-token probes 3 and 4 (§7.33)** are not read in this version: the probes were still running when this pre-draft was built.")
          + "\n\n**Update (v33).** Three items are written up here:")

OLD_ROWS = "| 2 | **LoRA on DINOv3**"
NEW_ROWS = (
    "| 1ac | **CRAWL-DIAG and the DAgger-bank pre-steps** (zero training, dev-box CPU): `H-CRAWL-01` outcome-blind floor, `H-CRAWL-02` S_v, `H-CRAWL-03` intermediate checkpoints, `H-ST-REGRET-PRE`,"
    " `H-ST-RECOVERYLABEL-1`, `H-DD-FITGAP-M2B` | cards frozen before data (crawl `PREREG_FROZEN.md` `7f4601ec…`; bank `FROZEN_CARDS.md` `3202a62f…`, Amendments 1 and 2): (a) model − floor ≤ −0.4 m/s with CI excluding 0 and a"
    " non-positive GT-stay gap, (b) within ±0.2 m/s; S_v ≥ 0.6 supported, ≤ 0.3 with upper CI < 0.5 falsified; REGRET: crawl share < 3 % predicted, 5-times branch; RECOVERYLABEL: 75 to 81 % predicted; FITGAP:"
    " r ≥ 0.5 FIT-LIMITED, ≤ 0.25 GEN-GAP, 5 of 6 checkpoints | **Read** (§7.30, T1-kin and T0-class): (a) not met; (b) met on the point for DAG-OACC s0 only; S_v supported (+0.736 / +0.628); CRAWL-03 NOT EVALUABLE"
    " (no intermediate checkpoint); REGRET-PRE MIDDLE BAND (about 3 times); RECOVERYLABEL NOT CONFIRMED (59.5 / 62.0 %); FITGAP PARTIAL, 2 of 6 FIT-LIMITED. Next: 3 more checkpoints for FITGAP (about 0.17 h each"
    " on a 10 % clip subset, **[E]**); a trainer option that keeps the last k checkpoints (needs a G-DVB entry) |\n"
    "| 1ad | **Trigger probes 1 and 2** (zero training, dev-box CPU; precursors of `H-CRAWL-06`) | frozen first (probe 1 `0eb1c25c…`, probe 2 `d9af82be…`): D1 (information present, unused) if dAUC(V+E − E) ≥ 0.05 with CI excluding 0 and"
    " AUC(Z+E) ≥ AUC(V+E) − 0.02, D3 if < 0.02; VISION-THROUGH-TRUNK if dAUC(Z_nn+E − E) ≥ 0.05, NAV-ONLY if < 0.02; MOTION if dAUC(max(V4, Vdiff)+E − V+E) ≥ 0.03, none if < 0.01 | **Read** (§7.31, T0-class): P-MOVE thin D1 as"
    " frozen; P-STOP frozen reading NOT EVALUABLE; Z_nn VISION-THROUGH-TRUNK by the letter on P-MOVE, and post-freeze the edge is the speed-ceiling and nav oracles; V4 / Vdiff NO MOTION. Next: the deployed planner on the GO windows"
    " with nav and ceiling at donor values (zero training); a perception lever at higher resolution |\n"
    "| 1ae | **SOUP** (`H-V8F-SOUP-1`): uniform weight average of two DAG-OACC-recipe O fine-tunes (SOUP, SOUP', SOUP4 beside); zero training, dev-box CPU | `PREREG_V8F_SOUP` (`830d9c91…`, promoted `e03b348`): HARMS if controls pass and the"
    " guard set H (R-TLAT, R-TLON, R-ADE, R-T0, R-NEW) fails at one or more seeds; adoption R-KEEP (ADE6 point ≤ +0.0604 m) and R-NOBASE; crawl rows R-MOVE (every crawl row separated-better by at least its bar) and R-CRAWL; routing: a"
    " barrier (SG separated-worse) selects ENS2, otherwise FLAT selects SOUP-K | **HARMS** (§7.32; `RESULT_V8F_SOUP.md`, 368 of 368 controls PASS, T1-kin): H fails at s0 (ADE6 +0.0770 / +0.0959 m); R-MOVE and R-CRAWL false in all four"
    " pairs; SG separated-better in five of six cells, no barrier, route FLAT. Next: SOUP-K, K = 4 draws per seed (4 box-2 runs, about 2.8 h, **[E]**), needs its own PI go |\n"
    "| 1af | **Fine-token probes 3 and 4** (zero training, dev-box CPU) | the cards of probes 3 and 4 (§7.33) | " + site("R84", "**RUNNING** (§7.33; no probe 3 or probe 4 number is read in this version).") + " |\n"
    "| 2 | **LoRA on DINOv3**")

OLD_9 = "and the P10 floors.\n\n---\n\n## 10. Reproducibility appendix"
NEW_9 = ("and the P10 floors.\n"
         "12. **v34 reads and their limits.** The crawl diagnosis (§7.30) rests on one checkpoint pair (DAG-OACC s0 and REP s0) for the gaps and S_v, on two M2b checkpoints for the fit-gap ratio, and on banks built from them; the 327 launch"
         " windows are 49 episodes, so the intervals on the model-minus-floor difference are wide (half-width about 0.30 m/s). The trigger probes (§7.31) are representation probes on one checkpoint (DAG-OACC s0), fitted and scored on"
         " TRAIN windows that checkpoint trained on, and their TRAIN cache and rebuilt join are UNVERIFIED byte for byte against box 2's copies; the nav token and the speed ceiling are ego-future channels. The SOUP read (§7.32) averages two"
         " draws (N = 2 per decisive soup) at two seeds, one of which has a slow draw, and the bootstrap is blind to training variance. The fine-token probes (§7.33) carry the limits stated in that section."
         " The two literature reviews behind the v34 related-work paragraphs are read through their own sentences and page numbers; beyond them the descriptions are UNVERIFIED.\n\n---\n\n## 10. Reproducibility appendix")

OLD_10 = "\n\n**Figures.** `make_figures.py` in this folder"
NEW_10 = ("\n\n**Reads added in v34 (tip `" + TP + "`).** Crawl diagnosis `R/raw/2026-10-28-crawl-diag/` (cards `PREREG_FROZEN.md` `7f4601ec…`, `READINGS.md` `a97423b6…`, frozen 21:21:51 and 21:28:08 on 2026-10-28); DAgger-bank pre-steps"
          " `R/raw/2026-10-28-bank-presteps/` (`FROZEN_CARDS.md` `3202a62f…`, Amendment 1 `4676bdb8…`, Amendment 2 `17cce25f…`); trigger probe 1 `R/raw/2026-10-28-trigger-probe/` (`PREREG_FROZEN.md` `0eb1c25c…`, landed `c1c30fb7`)"
          " and probe 2 `R/raw/2026-10-29-trigger-probe2/` (`d9af82be…`, landed `897b711b`); SOUP `R/RESULT_V8F_SOUP.md`, `Project Steering/PREREG_V8F_SOUP.md` (`830d9c91…`, promoted `e03b348`) and `R/raw/2026-10-29-soup-read/out/read/SOUP_READ.json`"
          " (`dca10c07…`, banked `0c7b598`); the two literature reviews with their banked primaries (Library, landed `5cc7d7b1`). The build is `patch_v34.py` (anchored edits, every table copied line for line from its RESULT and asserted to be a"
          " source line), checked by `checks_v34.py` and re-measured by `audit_v34.py` (`v34_NUMBERS_AUDIT.md`).\n\n**Figures.** `make_figures.py` in this folder")

ANCHOR_4 = "\n\n---\n\n## 4. Method"
ANCHOR_6 = "\n\n---\n\n## 6. Experimental protocol"
ANCHOR_8 = "\n\n---\n\n## 8. Discussion"
OLD_729N = "* **Next (Rule Zero).** **Committed next after HARMS:**"
E = [(OLD_ST, NEW_ST), (OLD_ST2, NEW_ST2), (OLD_ABS, NEW_ABS), (ANCHOR_4, "\n\n" + REL_WORK + ANCHOR_4), (ANCHOR_6, "\n\n" + MATH_54 + ANCHOR_6),
     (OLD_729N, UPD729 + OLD_729N), (ANCHOR_8, "\n\n" + SEC.rstrip("\n") + ANCHOR_8), (OLD_811, NEW_811), (OLD_83, NEW_83), (OLD_ROWS, NEW_ROWS), (OLD_9, NEW_9), (OLD_10, NEW_10)]
for a, b in E:
    assert s.count(a) == 1, (s.count(a), a[:80])
    assert h.count(a) == 1, ("html", h.count(a), a[:80])
    s = s.replace(a, b)
    h = h.replace(a, b)

# ---- guards on the inserted text -----------------------------------------------------------------------------------------------------------------------------
ins = "\n".join(b for _, b in E)
low = ins.lower()
REL_LOW = REL_WORK.lower()
_cl_rel = [m.start() for m in re.finditer(r"closed[- ]loop", REL_LOW)]
for _p in _cl_rel:  # published closed-loop claims in the related-work paragraphs are tagged [P] in their own sentence; nowhere else may the words appear
    _sent = REL_LOW[max(0, _p - 160):_p + 160]
    assert "[p]" in REL_LOW[_p:_p + 400], ("a closed-loop mention in the related work without a [P] tag", _sent)
_ins_no_rel = low.replace(REL_LOW, "")
assert "closed loop" not in _ins_no_rel and "closed-loop" not in _ins_no_rel and "drives" not in low, "closed-loop or drives wording outside the related work"
for w in BANNED:
    _m = re.search(r"\b" + w + r"\b", low)
    assert _m is None, (w, low[max(0, _m.start() - 150):_m.end() + 100] if _m else "")
assert re.search(r"retract|supersed|withdrawn|was wrong|not the floor|previously read|outdated|i said", low) is None
_math_dollar = MATH_54.count("$")
assert "$" not in SEC and "$" not in REL_WORK and "\r" not in ins and _math_dollar > 0
nocode = re.sub(r"`[^`\n]*`", "", SEC + "\n" + REL_WORK + "\n" + NEW_ABS + NEW_811 + NEW_83 + NEW_ROWS + NEW_9 + NEW_10)
assert re.search(r"<[A-Za-z/!]", nocode) is None, re.search(r".{20}<[A-Za-z/!].{20}", nocode)
for nm in ("LINE", "TITLE733", "R84"):
    if not PREDRAFT:
        assert "\n" not in FILL[nm].strip() and (nm != "R84" or "|" not in FILL[nm]), ("single-line slot with a line break or a pipe", nm)
assert sorted(USED) == sorted(SLOTS), (sorted(USED), sorted(SLOTS))
_markers = re.findall(r"@@FT_RESULT_([A-Z0-9]+)@@", s)
if PREDRAFT:
    assert sorted(_markers) == sorted(SLOTS), (sorted(_markers))
    assert all(h.count("@@FT_RESULT_" + n + "@@") == 1 for n in SLOTS)
else:
    assert "@@" not in s and "@@FT_RESULT" not in h and not _markers, "a slot marker is left in a final version"
    for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"):
        assert fam in FILL["FAMILIES733"], ("the FAMILIES733 text names no " + fam)
    for stale in ("not read in this version", "Not read in this version", "pre-draft", "were still running", "RUNNING", "is not read in this version"):
        assert stale not in ins, ("a pre-draft sentence is left in the final text", stale)
    assert FILL["ABSTRACT"].lstrip().startswith("* **"), "the ABSTRACT text is one bullet"
    for _k, _v in FILL.items():
        assert not re.search(r"closed[- ]loop|driving performance|\bdrives\b", _v, re.I), ("closed-loop or driving-performance wording in a fill text", _k)
        for w in BANNED:
            assert re.search(r"\b" + w + r"\b", _v, re.I) is None, (w, _k)
    if FT_RESULTS:
        _res_lines = set("\n".join(FT_RESULTS).split("\n"))
        for _k, _v in FILL.items():
            for _l in _v.split("\n"):
                if _l.startswith("|"):
                    assert _l in _res_lines, ("a table line of the fill text is not a line of a fine-token RESULT", _k, _l[:90])
        _corpus = re.sub(r"\s+", " ", "\n".join(FT_RESULTS))
        for _k, _v in FILL.items():
            for _q in re.findall(r'\*"([^"\n]{25,}?)"\*', re.sub(r"\s*\n\s*", " ", _v)):
                assert re.sub(r"\s+", " ", _q) in _corpus, ("a quotation in the fill text is not verbatim in a fine-token RESULT", _k, _q[:90])
    else:
        print("NOTE: no --ft-result given; the fill text's table lines and quotations were NOT checked against a RESULT (the checks script must be run with the RESULT)")
assert "\r" not in s and "</script" not in s.lower()
for sec_no in ("### 7.30 ", "### 7.31 ", "### 7.32 ", "### 7.33 ", "### 5.4 "):
    assert s.count(sec_no) == 1, sec_no
assert s.index("### 5.3 ") < s.index("### 5.4 ") < s.index("## 6. Experimental protocol")
assert s.index("### 7.29 ") < s.index("### 7.30 ") < s.index("### 7.31 ") < s.index("### 7.32 ") < s.index("### 7.33 ") < s.index("## 8. Discussion")
for _sec, _txt in (("7.30", SEC_730), ("7.31", SEC_731), ("7.32", SEC_732)):
    _flat = re.sub(r"\s+", " ", _txt)
    for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"):
        assert fam in _flat, (_sec, fam)
    assert "B = 2,000" in _flat and ("T1-kin" in _flat or "T0" in _flat), _sec
assert s.count("**Status:** research report, v34 of 2026-10-29") == 1
assert tpl.replace("%%MD%%", s) == h, "html edited in place differs from template + md"

os.makedirs(OUT, exist_ok=True)
open(OUT + "V8F_SCIENTIFIC_REPORT.md", "w", encoding="utf-8", newline="\n").write(s)
open(OUT + "v8f_report.html", "w", encoding="utf-8", newline="\n").write(h)
if PREDRAFT:
    open(OUT + "FT_FILL_TEMPLATE.json", "w", encoding="utf-8", newline="\n").write(json.dumps(OrderedDict((k, "") for k in SLOTS), indent=1) + "\n")
    open(OUT + "FT_FILL_HELP.txt", "w", encoding="utf-8", newline="\n").write(
        "Slots of the v34 pre-draft (fine-token probes 3 and 4, section 7.33). Copy FT_FILL_TEMPLATE.json to FT_FILL.json, write one non-empty text per slot, then run:\n"
        "  python -I patch_v34.py <tip> <out_dir> --fill FT_FILL.json --ft-result <repo path of the RESULT at <tip>> [--ft-result <another>]\n"
        "A slot text REPLACES the pre-draft text at its site together with the marker. SETUP733 to PROVENANCE733 and ABSTRACT hold whole bullets: start each with '* ' (SETUP733 to PROVENANCE733: '* **Name.** ...');\n"
        "a table inside a bullet is a block at column 0 with a blank line before and after, and the text after it starts with two spaces. STATUS733, R81 and R83 follow a fixed lead (see below).\n"
        "TITLE733 follows the fixed lead 'Fine-token probes 3 and 4: ' in the 7.33 heading. LINE, TITLE733 and R84 are one line (R84 has no pipe character). Banned words, tier words and the closed-loop wording rule apply to the texts.\n"
        "Every table line of a fill text must be a line of a --ft-result RESULT and every italic quotation of 25 characters or more must be verbatim in it. Every number needs its evidence class, tier (T0 / T1-kin) and estimator;\n"
        "name per family what is NOT RUN / NOT BUILT / NOT COMPUTABLE with the reason. STRATEGIC is reported, not paused, for v8F.\n\n"
        + "\n".join("%-14s %s" % (k, v) for k, v in SLOTS.items()) + "\n")
print("report v34 built (%s)" % ("PRE-DRAFT, %d slots" % len(SLOTS) if PREDRAFT else "FINAL"), len(s), len(h), "tip", TIP)
