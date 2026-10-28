"""Report v33: SPEEDSUB (7.27, PASS), SPEEDSUB-T1 (7.28, NULL / NULL-ONE) and LAUNCH-INTENT (7.29: HARMS, arm A not adopted; M-LAUNCH NOT MOVED); the abstract,
8.1 (items 15-17), 8.3, 8.4 (rows 1z-1ab); the version line. Builds the v33 .md and .html from the v32 blobs of one repository tip. Nothing is read from a working tree.
The final build takes the LAUNCH-INTENT outcome from FILL.json (written by mk_fill_v33.py from RESULT_V8F_LAUNCH_INTENT.md, LAUNCH_INTENT_READ.json, MLAUNCH.json and the banked box-2
chain log at the same tip); --predraft rebuilds the earlier pre-draft with marked slots.

python patch_v33.py [tip [out_dir]] [--predraft] [--fill FILL.json]
  tip         repository tip whose blobs are the inputs (default 8c1e9636388dbb105f6b1853bd95dfdccea9eb93); its first 8 hex digits go to the status line
  out_dir     folder to write V8F_SCIENTIFIC_REPORT.md and v8f_report.html into (default: out/ beside this script with --predraft, else final/)
  --predraft  write the PRE-DRAFT: section 7.29 carries marked outcome slots (the string at-at LI_RESULT_<NAME> at-at, one per slot, listed in SLOTS below) and a
              LI_FILL_TEMPLATE.json is written beside the report. Without this flag the patch writes a FINAL version only: it needs --fill FILL.json, a JSON object that
              holds a non-empty text for every slot, and it REFUSES (non-zero exit, nothing written) while any slot is missing, empty or unknown, or while the text still holds a slot marker.
  --fill      the JSON file with the slot texts (a later run writes it from RESULT_V8F_LAUNCH_INTENT.md). The texts replace the pre-draft sentences together with their slot,
              so a final version carries no "running" or "not read" wording of the pre-draft. The same guards (banned words, tier words, anchors) run on them.

Inputs (all `git show <tip>:<path>` from the mirror, never the mirror's HEAD):
  the v32 V8F_SCIENTIFIC_REPORT.md, v8f_report.html and page_template_v30.html; RESULT_V8F_SPEEDSUB.md and RESULT_V8F_SPEEDSUB_T1.md and PREREG_V8F_LAUNCH_INTENT.md
  (their tables are COPIED LINE FOR LINE by anchored extraction, each line asserted to be a line of the source);
  PREREG_V8F_SPEEDSUB.md, PREREG_V8F_SPEEDSUB_T1.md (with Amendment 1), PREREG_V8F_DAG_ORC.md (Amendment 3), PREREG_V8F_LAUNCH_INTENT.md (as promoted at 77234276, sha256 469bac1f, and
  as amended by its Amendment 1 at e28516e8, sha256 6ae79f2f: the amended text must begin with the promoted bytes), RESULT_V8F_LAUNCH_INTENT.md, the banked box-2 chain log
  raw/2026-10-28-launch-intent-read/box2/chain_v8f_launchintent.out, R2C_NOTE.md (raw/2026-10-28-dag-orc-result/notes/), the launch-intent chain notes and the register rows
  H-V8F-SPEEDSUB-1, H-V8F-SPEEDSUB-T1-1 and H-V8F-LAUNCH-INTENT-1 of GOALS_AND_CLAIMS.md (every claim taken from them is asserted to be present in the source text, see PHRASES).
Every insertion is anchored, and its anchor is asserted to occur exactly once. The .html is produced twice, by the same anchored edits applied to the v32 .html and by
the template with the new .md, and the two must be equal. The three v32 report blobs (md, html, template) are asserted byte-identical at d8c9f7bf (the tip the first draft was built on) and at
the tip: nothing in the report folder moved between the two.
Every table line in a final fill text must be a line of RESULT_V8F_LAUNCH_INTENT.md (copied line for line); the launch facts are quoted from the banked chain log, not from a note outside the repository."""
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
_argv = [a for a in _argv if a != "--predraft"]
TIP = (_argv[0] if len(_argv) > 0 else "8c1e9636388dbb105f6b1853bd95dfdccea9eb93")
TP = TIP[:8]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = (_argv[1] if len(_argv) > 1 else HERE + ("/out" if PREDRAFT else "/final")).rstrip("/\\") + "/"
MIRROR = "C:/Users/Admin/tanitad-push"
R = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
RP = R + "report/2026-10-26/"

# ---- the outcome slots of 7.29 and its other sites: name -> what the fill text is -------------------------------------------------------------------
SLOTS = OrderedDict([
    ("LINE", "status line (v33 adds ...): the clause that names the LAUNCH-INTENT read, with its outcome word, ending in a full stop"),
    ("ABSTRACT", "abstract: TWO bullets, each starting with '* ': the LAUNCH-INTENT bullet (outcome word, the decisive numbers with tier and estimator) and the 'What is next' bullet that replaces the pre-draft one"),
    ("TITLE729", "heading of 7.29 after 'LAUNCH-INTENT: ' (the outcome word and label), one line"),
    ("STATUS729", "7.29 first bullet: the text that follows the fixed lead '* **Status of this section (v33).**' (read date, RESULT file and landing commit, controls count, outcome word)"),
    ("OUTCOME", "7.29: the outcome bullet(s) with the RESULT's outcome table(s) copied line for line, the word, the committed next verbatim"),
    ("CONTROLS", "7.29: the controls bullet (names and counts, each ran before its rows were read)"),
    ("RUNS", "7.29: the four runs (checkpoint sha256 prefixes, K-SHARE and K-MASK values, run times, box-2 hours against the ~2.8 h estimated, any stop)"),
    ("FLOORS", "7.29: the bars and the replicate floors RF beside, with the main and the replicate pair contrasts for all 19 rows"),
    ("CRAWL", "7.29: R-MOVE and R-CRAWL on the crawl set C, per pair, per seed, with the bars"),
    ("GUARDS", "7.29: the guards R-TLAT, R-TLON, R-ADE, R-T0, R-NEW and the adoption rows R-KEEP, R-NOBASE, per pair, per seed"),
    ("FAMILIES", "7.29: the four families (LONGITUDINAL, LATERAL, TACTICAL, STRATEGIC) per pair and seed, with NOT RUN / NOT BUILT / NOT COMPUTABLE per family and the reason"),
    ("MLAUNCH", "7.29: M-LAUNCH (or NOT BUILT, with the reason) and the slow-draw flag, and the NEXT-LAUNCH route they give"),
    ("NUMBERS", "7.29: what the numbers say (shares, ranges, counts computed from the tables, never a bare exponent)"),
    ("NEXT", "7.29: Next (Rule Zero): the committed next, every remaining lever named and what blocks or unblocks it"),
    ("PROVENANCE", "7.29: provenance (RESULT file, read JSON sha256, reader, run times, where the dumps live)"),
    ("R81", "8.1 item 17: the text after the fixed lead '17. ' (a bold claim, then the established / not established statement for LAUNCH-INTENT with tier and estimator)"),
    ("R83", "8.3 Update (v33): the text after the fixed lead '* ' of the LAUNCH-INTENT bullet; keep a sentence on the adopted base"),
    ("R84", "8.4 row 1ab: the status cell, one line without a pipe character"),
])
FILL = {}
if not PREDRAFT:
    if not FILLF:
        sys.exit("patch_v33.py REFUSES to write a final version without --fill FILL.json; every LAUNCH-INTENT slot needs its text (" + ", ".join(SLOTS) + "). Use --predraft for the pre-draft.")
    if not os.path.exists(FILLF):
        sys.exit("patch_v33.py REFUSES: the fill file does not exist: " + FILLF)
    FILL = json.load(open(FILLF, encoding="utf-8"))
    _unk = [k for k in FILL if k not in SLOTS]
    _unf = [k for k in SLOTS if not (isinstance(FILL.get(k), str) and FILL[k].strip())]
    if _unk or _unf:
        sys.exit("patch_v33.py REFUSES to write a final version: unfilled slots %s, unknown keys %s" % (_unf, _unk))
    _left = [k for k, v in FILL.items() if "@@" in v]
    if _left:
        sys.exit("patch_v33.py REFUSES: the fill text of %s still holds an at-at marker" % _left)
USED = []


def site(name, pre):
    """The text at a slot site: the pre-draft sentence plus the visible marker, or the fill text of a final version."""
    assert name in SLOTS, name
    USED.append(name)
    if PREDRAFT:
        return pre + (" " if pre else "") + "@@LI_RESULT_" + name + "@@"
    return FILL[name].strip("\n")


def blob(path, strict=True):
    r = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*", "show", f"{TIP}:{path}"], capture_output=True)
    assert r.returncode == 0 and len(r.stdout) > 0, (path, r.stderr[:200])
    t = r.stdout.decode("utf-8")
    if strict:
        assert "\r" not in t and "\ufeff" not in t, path
    return t.replace("\r\n", "\n")


FIRST_TIP = "d8c9f7bfc91798b6fafb5a13dd793771e2f79f6a"  # the tip the first draft (patch_v33 of 14:40) was built on


def gitrun(args, binary=False):
    r = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*"] + args, capture_output=True)
    assert r.returncode == 0, (args, r.stderr[:200])
    return r.stdout if binary else r.stdout.decode("utf-8").strip()


# the v32 report blobs are the starting point: they must be the very objects the first draft started from
for _f in ("V8F_SCIENTIFIC_REPORT.md", "v8f_report.html", "page_template_v30.html"):
    _a, _b = gitrun(["rev-parse", FIRST_TIP + ":" + RP + _f]), gitrun(["rev-parse", TIP + ":" + RP + _f])
    assert _a == _b, ("the v32 report blob moved between d8c9f7bf and the tip", _f, _a, _b)
V32_BLOBS = {f: gitrun(["rev-parse", TIP + ":" + RP + f]) for f in ("V8F_SCIENTIFIC_REPORT.md", "v8f_report.html", "page_template_v30.html")}

s = blob(RP + "V8F_SCIENTIFIC_REPORT.md")
h = blob(RP + "v8f_report.html")
tpl = blob(RP + "page_template_v30.html")
assert tpl.count("%%MD%%") == 1 and h == tpl.replace("%%MD%%", s), "v32 html is not the template with the v32 md"
assert s.count("**Status:** research report, v32 of 2026-10-28") == 1

SS = blob(R + "RESULT_V8F_SPEEDSUB.md")
T1 = blob(R + "RESULT_V8F_SPEEDSUB_T1.md")
PSS = blob("Project Steering/PREREG_V8F_SPEEDSUB.md")
PT1 = blob("Project Steering/PREREG_V8F_SPEEDSUB_T1.md")
PLI = blob("Project Steering/PREREG_V8F_LAUNCH_INTENT.md")  # as amended (Amendment 1 appended at e28516e8)
PLI0 = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*", "show", "77234276:Project Steering/PREREG_V8F_LAUNCH_INTENT.md"], capture_output=True).stdout.decode("utf-8")
PDO = blob("Project Steering/PREREG_V8F_DAG_ORC.md")
REG = blob("Project Steering/GOALS_AND_CLAIMS.md")
RES = blob(R + "RESULT_V8F_LAUNCH_INTENT.md")
CHAIN_RAW = gitrun(["show", f"{TIP}:{R}raw/2026-10-28-launch-intent-read/box2/chain_v8f_launchintent.out"], binary=True)  # a Windows log: CRLF line ends, hashed as banked
CHAIN = CHAIN_RAW.decode("utf-8").replace("\r\n", "\n")
R2C_RAW = gitrun(["show", f"{TIP}:{R}raw/2026-10-28-dag-orc-result/notes/R2C_NOTE.md"], binary=True)
R2C = R2C_RAW.decode("utf-8").replace("\r\n", "\n")
LIP = blob(R + "raw/2026-10-28-launch-intent-chain/PINS_FILLED.txt")
LIH = blob(R + "raw/2026-10-28-launch-intent-chain/HANDOFF.md")
LIN = blob(R + "raw/2026-10-28-launch-intent-v2/NOTES.md")
assert hashlib.sha256(PLI0.encode("utf-8")).hexdigest().startswith("469bac1f3918b31b"), "the PREREG blob at 77234276 is not the file the chain was pinned to"
assert hashlib.sha256(PLI.encode("utf-8")).hexdigest().startswith("6ae79f2f451868e8"), "the PREREG blob at the tip is not the amended text the read of record was pinned to"
assert PLI.startswith(PLI0) and PLI[len(PLI0):].lstrip("\n").startswith("## Amendment 1 (A1-2026-10-28)"),"the amended PREREG does not begin with the promoted bytes: it must only append Amendment 1"
for _txt, _pre, _what in ((RES, None, "RESULT"),):
    assert _txt.startswith("# RESULT_V8F_LAUNCH_INTENT: O's launch-sample along-track weight on DAG-OACC's recipe, aimed at the crawl rows. Outcome **HARMS** (arm A, W 1.58926, not adopted)"), "RESULT title"
for _raw, _pre, _what in ((CHAIN_RAW, "ddae7b9e3ac82f86", "box-2 chain log"), (R2C_RAW, "df02ff2f8b46ab05", "R2C_NOTE")):
    assert hashlib.sha256(_raw).hexdigest().startswith(_pre), (_what, "sha256 prefix")
for _rel, _pre in (("raw/2026-10-28-launch-intent-read/read_A1/LAUNCH_INTENT_READ.json", "e451e1a6ba44e1d3"), ("raw/2026-10-28-launch-intent-mlaunch/MLAUNCH.json", "f5ebb4b209a5072e"),
                   ("raw/2026-10-28-launch-intent-chain/PINS_FILLED.txt", "fa825de89d77e6d8")):
    _b = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*", "show", f"{TIP}:{R}{_rel}"], capture_output=True).stdout
    assert hashlib.sha256(_b).hexdigest().startswith(_pre), (_rel, "sha256 prefix does not match the blob at the tip")
    if _rel.endswith("LAUNCH_INTENT_READ.json"):
        LIR_J = json.loads(_b.decode("utf-8"))
    if _rel.endswith("MLAUNCH.json"):
        MLJ_J = json.loads(_b.decode("utf-8"))


# ---- tables, copied line for line -------------------------------------------------------------------------------------------
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


TABLES = {
    "SS_OUT": (SS, "## 1. The outcome, as the pre-registered table decides it", 12),
    "SS_COND": (SS, "The pre-registered conditions (PREREG section 5, REFUSED first, then HARMS, then PASS), restated against the rows.", 6),
    "SS_CTRL": (SS, "The reader's flags are in `SUBREAD.json` `controls`", 17),
    "SS_ATTR": (SS, "Mean hold-out RMSE over the five cells, train-SD units, lower is better:", 6),
    "SS_EGO": (SS, "### `ego` − `own` (the arm; point [95 % CI])", 33),
    "SS_GT": (SS, "### `gt` − `own` (T0 ceiling, a control only; point [95 % CI])", 33),
    "SS_GAPS": (SS, "The same gaps hold in all four family JSONs", 13),
    "SS_VIOL": (SS, "The rates equal the dumps' own `raw.consistency_a.*.rate_all` (asserted).", 7),
    "SS_CELLS": (SS, "`mlp_ego` = the arm's cells (identical at the two seeds", 7),
    "T1_OUT": (T1, "**Hit rule (PREREG section 6):**", 9),
    "T1_WALK": (T1, "**The table's walk (REFUSED first, then HARMS, then PASS, PARTIAL, NULL):**", 7),
    "T1_GUARDS": (T1, "### The guards, recomputed from the cells", 6),
    "T1_CTRL": (T1, "### Controls (each must read its known value before any row is read; any failure is REFUSED)", 12),
    "T1_ROWS": (T1, "Each cell is the paired episode-cluster bootstrap of the arm minus the same run's `identity` rollout", 23),
    "T1_CENSUS": (T1, "**Census of the table above (DERIVED; counted by `gen_result.py` over the 18 computed rows per run):**", 6),
    "T1_LONG": (T1, "`egoc − P10` (LONGITUDINAL):", 8),
    "T1_LAT": (T1, "`egoc − P10` (LATERAL):", 8),
    "T1_TAC": (T1, "`egoc − P10` (TACTICAL):", 4),
    "T1_SPOS": (T1, "S-POS@30 / 50 / 80 are the strategic head's position error **within a run**", 11),
    "T1_GAPS": (T1, "### Gaps, per family: what is not computed, with the reason", 7),
    "T1_REPS0": (T1, "At REP s0 `egoc` cuts ADE6 by", 12),
    "T1_FLOOR0": (T1, "**COLLAPSE** needs `g_id(r) > 0` and `g_c(r) ≤ 0.5 · g_id(r)` on **all three** rows", 5),
    "T1_FLOOR1": (T1, "so the part of the gap the substitution removes is REP s0's own repair net of the small CTRL s0 movement.", 5),
    "T1_HEALTHY": (T1, "## 5. The healthy draws: CTRL s0, CTRL s1 and REP s1 show no separated ADE6 change", 5),
    "T1_SHARE": (T1, "The P10 floor is the crawl reference. DAG-ORC's perturbation arm A closed 27.6 % to 79.9 % of the DAG-OACC crawl gap", 6),
    "T1_BRIDGE": (T1, "Lane: `--limit logged (declared OPEN channel", 14),
    "T1_BLOCKED": (T1, "### What is blocked, and on what (Rule Zero: every remaining lever)", 7),
    "LI_GAP": (PLI, "Against the whole gap DAG-OACC leaves to P10:", 6),
    "LI_BARS": (PLI, "**The bars, all 19 rows** (v2 bar = 2 × max(F, RF0); RF and the v1 bar are beside; units are the row's own):", 21),
    "LI_ARMS": (PLI, "Each pair differs in the lever alone: same init, seed, bank and argv, apart from `--out` and the lever.", 7),
    "LI_PRED": (PLI, "**The per-seed predicates, defined on both pairs of the seed**", 9),
    "LI_OUT": (PLI, "A guard \"fails at ≥ 1 seed\" when it fails at that seed in either pair.", 10),
    "LI_NEXT": (PLI, "**moved** = separated toward 0 in both pairs at both seeds", 4),
}
TAB = {}
for name, (src, anchor, n) in TABLES.items():
    TAB[name] = grab(src, anchor, n)
    assert all(r in src.split("\n") for r in TAB[name]), name
    assert len({ncols(r) for r in TAB[name]}) == 1, (name, sorted({ncols(r) for r in TAB[name]}))

# ---- claims taken from the sources, asserted present --------------------------------------------------------------------------
PHRASES = [
    # RESULT_V8F_SPEEDSUB.md
    (SS, "Outcome **PASS** (a screening result at T1-class open loop)"), (SS, "all **27 control flags PASS**"), (SS, "`ZZSPEEDSUB-DONE PASS`"), (SS, "(sha256 `1fe7fcb3…`)"),
    (SS, "Run 2026-10-28 07:59:08 to 08:17:43 (18 min 35 s; dev-box CPU, 4 threads"), (SS, "the pre-registered reader `subread` ran 08:05:36–08:05:38"), (SS, "No SPEEDSUB stage stopped."),
    (SS, "`numpy.random.default_rng(20261029).permutation(2399)`"), (SS, "bin plus `v0` recovers it at R² 0.9702"), (SS, "**there is no replicate arm in this experiment**"),
    (SS, "one fit (`readout.pt` sha256 `a205609a…`) serves both seeds"), (SS, "It is still the third read-out examined on these 141 episodes"),
    (SS, "mean fold-0 hold-out RMSE `mlp_ego` 0.2475 against `lin` 0.2724, `lin_ego` 0.2807, `mlp` 0.2920"), (SS, "**s1 clears the bar by 0.0083 m only**"),
    (SS, "**The two seeds' effects differ by 0.0604 m**"), (SS, "`ego` recovers 18 % and 10 % of what the GT cells would give"),
    (SS, "identical delta, lo and hi to 4 dp on all 18 values"), (SS, "s0 sha256 `c6a62876…`, s1 `68a0f335…`"),
    (SS, "18.7 % (s0) and 11.1 % (s1) of the `gt` ceiling on speed MAE, 18.4 % / 11.1 % on along-track MAE"), (SS, "but 92.7 % / 97.4 % on accel MAE"),
    (SS, "the ceiling moves accel MAE by only 0.05 and 0.03 m/s²"), (SS, "The three head macro-F1 rows read exactly +0.0000 [+0.0000, +0.0000] at both seeds"),
    (SS, "**STRATEGIC: exactly zero, by construction.**"), (SS, "**The 2 s horizon does not improve.**"), (SS, "**s0:** 11 of 49 cells separated"), (SS, "**s1:** 9 of 49 cells separated"),
    (SS, "at a nominal 5 % about 2 to 3 would separate with no effect"), (SS, "On the 81 stop windows `ego` is *worse* than `own` at both seeds (+0.3374 / +0.1762 m, point values)"),
    (SS, "LL-3's largest deviation is 4.99e-05 (bar 1e-4)"), (SS, "identity on 45 cells, reproduction of 119 point values per arm"),
    (SS, "(the hold-out of `fit.json`, 846 clips)"), (SS, "(sha256 `e89af4a0…`)"), (SS, "The read-out's skill survives without `nav` and does not survive without `limit`."),
    (SS, "Dropping `nav` costs 0.0156 SD"), (SS, "dropping `limit` costs 0.0328 SD"), (SS, "so **which input carries the ADE6 cut is not measured here**"),
    (SS, "cut `v_a_ms` by 40–59 % and moved ADE6 by about 0"), (SS, "so a lower cell RMSE was not a sufficient condition"), (SS, "`mlp_ego` is below `T_own` in all five cells at both seeds"),
    (SS, "*\"T1-kin zero-training substitution of the banked ego read-out on the DAG-OACC / DAG-ORC base, both seeds, under its own PREREG (PREREG_V8F_SPEEDSUB section 6)\"*"),
    (SS, "**That follow-up is pre-registered as `PREREG_V8F_SPEEDSUB_T1` (landed `842ec2ac`"), (SS, "`a95f4fb1…`"), (SS, "rederive_sub.py"),
    (SS, "It is the strongest reason for the next arm to carry a rule-violation guard beside ADE6 (section 6)."), (SS, "**Not a statement about DAG-OACC / DAG-ORC.**"),
    (SS, "T1-class open loop (X_s's own held S / T outputs; only the five speed cells O reads are replaced)"), (SS, "**Distance keeping (DK-VIOL / DK-HW / DK-TG) and the anti-echo block are not computed**"),
    (SS, "F = 0.1207 m is the cross-seed spread of M2b (INHERITED, `RESULT_V8F_GRID.md`"), (SS, "**Not run:** the `egoshuf` control has no family tables"),
    (SS, "the T1-kin lane uses `--limit causal`"), (SS, "It is not a causal-bucket result"),
    # PREREG_V8F_SPEEDSUB.md
    (PSS, "**Status: PRE-REGISTERED, NOT RUN (2026-10-28).**"), (PSS, "Claim row `H-V8F-SPEEDSUB-1`"), (PSS, "**F/2 = 0.06035 exactly.**"),
    (PSS, "`mlp_ego` is below `T_own` in every cell at both seeds."), (PSS, "(RMSE cut 59 % at s0 and 40 % at s1"),
    # RESULT_V8F_SPEEDSUB_T1.md
    (T1, "Outcome **NULL** (label NULL-ONE: one hit of four runs, at the slow-T draw REP s0)"), (T1, "**all 158 control flags PASS**"), (T1, "`ZZSPEEDSUB-T1-READ-DONE NULL NULL-ONE`"),
    (T1, "promoted at `842ec2ac` (09:00:08), before any T1-kin substitution rollout and before its read-out was fitted"), (T1, "landed at `4cec09e7` (10:17:23), before the smoke (10:21:44)"),
    (T1, "`readout_c.pt`, sha256 `799f65ed…`"), (T1, "9.412e-06 m/s against the 1e-5 tolerance"), (T1, "Chain run 2026-10-28 10:21:41 to 11:09:29 (47 min 48 s)"), (T1, "the reader's own wall 454.4 s"),
    (T1, "REP s0's T predicts all five speed cells 0.73 to 1.17 m/s slow at T0 (INHERITED, section 4)"), (T1, "the data at those runs do not exclude an ADE6 cut as large as the bar"),
    (T1, "At CTRL s1 the lower end (−0.0500) lies inside the bar"), (T1, "**So the speed-cell channel is not what limits healthy DAG-OACC draws.**"),
    (T1, "Its measured use is repairing slow-T training draws (REP s0, section 4)."), (T1, "**(a) stays registered and is not run now, because arithmetic predicts NULL.**"),
    (T1, "**The largest measured crawl lever is DAG-ORC's perturbation**"), (T1, "**The registered crawl route is LAUNCH-INTENT on box 2, blocked on the PI's go.**"),
    (T1, "**M1 crawl share moves at no run.**"), (T1, "`egoshufc − identity` ADE6 is separated-worse at all four runs (K-REG"), (T1, "**s0 verdict: NOT COLLAPSE**"),
    (T1, "falls from 0.4025 to 0.2309 m (a drop of 42.6 %)"), (T1, "G-VIOL needs 3 of the 4 runs and fires at 1"), (T1, "The P10 floor sits at 0.7999 on this measure"),
    (T1, "At the healthy CTRL seeds `egoc` closes at most 12.8 % of the gap on any crawl row"), (T1, "27 min 52 s for the twelve decisive rollouts"), (T1, "at most about 1.6 h of dev-box CPU in all"),
    (T1, "138 to 141 s per decisive rollout"), (T1, "**Amendment 1 precedes every rollout.**"), (T1, "**The fit was reused, not re-run, in the chain.**"),
    (T1, "**Ticks 1 and 2 read the model's own executed motion**"), (T1, "**The four runs are two seeds × two replicates of one recipe, not four recipes**"),
    (T1, "STRATEGIC is reported, not paused, for v8F."), (T1, "**The paired `egoc − identity` S-POS row is NOT COMPUTED**"), (T1, "| LONGITUDINAL | anti-echo block | **NOT BUILT** |"),
    (T1, "| TACTICAL | goal-head F1 at T1-kin | **NOT COMPUTABLE here** |"), (T1, "sha256 `64785f8a…`"), (T1, "K-SHA true at all 12 rollouts"),
    (T1, "CTRL s0 `67265ec1…`, CTRL s1 `90b01474…`, REP s0 `b03fd6ef…`, REP s1 `77853e56…`"), (T1, "/SPEEDSUB_T1_READ.json` (sha256 `b3b72325…`)"),
    (T1, "**CTRL s1 is the one healthy draw that moves, and it moves the wrong way on three rows:**"), (T1, "**The read-out with the most future information available moves ADE6 by +0.0008 and −0.0185 m at CTRL s0 and s1: not separated**"),
    (T1, "On DAG-OACC at T1-kin the same read-out in the same lane does not move ADE6."), 
    (T1, "**Sensitivity (DERIVED, same code).**"), (T1, "hits at 1 of 4 runs, not at both CTRL seeds: label **NULL-ONE**"), (T1, "**One read-out fit.**"),
    (T1, "the logged oracle bucket (TRAIN hold-out 0.2475 SD units, against 0.2793 for `mlp_ego_c`)"), (T1, "2,399 windows × 3 ticks"),
    # PREREG_V8F_SPEEDSUB_T1.md
    (PT1, "Promoted by the Master Mind (draft sha256 237240e3…) before any T1-kin substitution rollout and before `mlp_ego_c` is fitted."), (PT1, "Claim row `H-V8F-SPEEDSUB-T1-1` opens on promotion."),
    (PT1, "## Amendment 1 (`A1-2026-10-28`; appended before the smoke and before any rollout of this experiment)"), (PT1, "**G-VIOL: `VIOL-ANY` is not separated-worse at ≥ 3 of the 4 runs.**"),
    (PT1, "`speed_max_bin(max measured ego speed over rows [j−20, j])`"), (PT1, "The labels NULL-ONE (hits at 1–3 runs, not both CTRL seeds) and NULL-FLAT (no hit) describe and decide"),
    (PT1, "ESTIMATED prior:"), (PT1, "| **NULL** | anything else | next: section 8 (a), the history-aware read-out |"), (PT1, "PASS is not the expected outcome"),
    (PT1, "Beside (CTRL only, lowest priority, decides nothing): `identity_l` and `egol`"),
    # PREREG_V8F_DAG_ORC.md, Amendment 3
    (PDO, "## Amendment 3 (A3-2026-10-28): the PI's routing decision after the read."), (PDO, "*\"Yes, queue it\"*"), (PDO, "B (p = 0.1) is therefore **not trained and not read**."),
    (PDO, "A's outcome stays **HARMS**, read under the table as written."),
    # PREREG_V8F_LAUNCH_INTENT.md
    (PLI, "Promoted by the Master Mind from DRAFT v2 (sha256 `aec8156a…`)"), (PLI, "Claim row `H-V8F-LAUNCH-INTENT-1`, registered at promotion."), (PLI, "**Status: PRE-REGISTERED, NOT RUN (2026-10-28).**"),
    (PLI, "It is the `LAUNCH-INTENT` branch of `PREREG_V8F_DAG_ORC` §8 (NEXT-CRAWL), re-based on DAG-OACC"), (PLI, "Box-2 time: the PI's go, *\"Yes, queue it\"* (2026-10-28), for the four runs of §4"),
    (PLI, "up-weighted on GT-launch TRAIN samples (v0 < 6 m/s and GT v(6 s) ≥ v0 + 1 m/s)"), (PLI, "**GT-launch windows** | **49,618 → f = 0.161851**"),
    (PLI, "**Arm A, s = 1/3: 1 + W = (1 − f) / (2 f) ⇒ W_A = 1.58926.**"), (PLI, "**Arm B, s = 1/4: 1 + W = (1 − f) / (3 f) ⇒ W_B = 0.726175**"), (PLI, "**Arm HI (NEXT-LAUNCH only), s = 1/2: 1 + W = (1 − f) / f ⇒ W_HI = 4.17852.**"),
    (PLI, "The bar and arms were changed by the Master Mind before promotion, after DAG-ORC's floors and the REP s0 diagnosis were known, and before any LAUNCH-INTENT number existed. The change follows from arithmetic on banked numbers (section 1). It is not a reaction to a LAUNCH-INTENT outcome, which does not exist."),
    (PLI, "In v1 the bar was 2 × max(F, RF0, RF), with RF the maximum of two REP draws, and REP s0 set RF on 17 of 19 rows."),
    (PLI, "would have failed v1's R-MOVE at both seeds: the decisive condition was unreachable by construction"), (PLI, "Each v2 crawl bar is below the CTRL − P10 gap at both seeds."),
    (PLI, "| **bar_r** | **2 × max(F_r, RF0_r)**, one value per row, applied at both seeds and in both pairs, inclusive."),
    (PLI, "⚠ **Stricter in one way:** every guard must hold in both pairs of the seed. ⚠ **Looser in one way:** the size bar drops from 2 × RF to 2 × RF0"),
    (PLI, "524,288 cells; each matches exactly one row: REFUSED 262,144 · HARMS 261,888 · FIXED / ADOPTED 1 · FIXED / NOT ADOPTED 15 · PARTIAL / ADOPTED 3 · PARTIAL / NOT ADOPTED 45 · PARTIAL / ONE SEED 128 · NO EFFECT 64."),
    (PLI, "A lever effect is credited at a seed only when two independent training draws show it. Across both seeds that is four draws"), (PLI, "**New box-2 runs: A_0, A_1, A_rep,0, A_rep,1.** ESTIMATED ≈ 2.8 h"),
    (PLI, "The GT-launch windows are 16.2 % of the training windows (MEASURED, §3), so their along-track error is diluted about 5:1."), (PLI, "v0 −1.78 / −0.93 m/s at tick 1 in the launch windows"),
    (PLI, "The launch subset is label-side (future GT speed), which is admissible for labels; nothing new enters at inference."), (PLI, "The flag-off run is the tip's, bit for bit"),
    (PLI, "**Disclosed coupling.** DAG-ORC and LAUNCH-INTENT share the REP_s runs."), (PLI, "**STRATEGIC is not paused for v8F** (PI ruling 2026-10-17)"),
    (PLI, "**The lever acts on LONGITUDINAL (along-track) pressure, so the LONGITUDINAL family and the distance-keeping rows are read first.**"), (PLI, "* **It is not closed loop.** T1-kin is a self-action open loop."),
    (PLI, "**A one-seed R-MOVE is not an effect** (`H-ESTIM-SEED-1`)"), (PLI, "**LOOP-STATE** (zero training, dev-box CPU)"),
    (PLI, "**LAUNCH-INTENT-HI**: the same lever at s = 1/2 (W_HI = 4.17852)"), (PLI, "No read adopted another base, so the base is DAG-OACC."),
    (PLI, "(`MM_REDESIGN_2026-10-28.md` §3 item 2)"), (PLI, "every (X, X) contrast reads exactly 0.0"), (PLI, "**The domain D** of a pair is every Read-A row not separated-worse for Y − P10"),
    # launch-intent chain package and reader notes (in the repo at the tip)
    (LIP, "after PREREG_V8F_LAUNCH_INTENT was promoted and landed at 77234276 (ls-remote verified)"), (LIP, "--prereg-sha 469bac1f3918b31b875e7c06b637980f6acb194a69ded3294c180aec95b1f65a"),
    (LIP, "--reader-sha 5e950dcff2993653dffbe9b4f01ea17eb4d7b1721b7c41a669e0dc43c559fc35"), (LIP, "sha256 c3edb43179cf1a73aa6563462c22b00e46d448049b30469130d62856b07e3ba3"),
    (LIP, "LAUNCH_COMMIT stays b09ac89f (the tip must descend from it); trainer 0d708b3c..., model b337ccec... unchanged."),
    (LIH, "SELFTEST 390/390 tests pass; MUTATION 185/185 mutants killed, 0 survivors"), (LIH, "Nothing is installed, landed or launched"),
    (LIN, "The reader does not compute M-LAUNCH (below), so `next_launch` prints UNDECIDED."), (LIN, "**Not built, and listed in the result's `not_built` block (never a silent omission):** M-LAUNCH"),
    (LIN, "the anti-echo block, along-track / speed / L2 on the GT-launch subset and M2, and the T0 four families"),
    (LIN, "the T-head F1 and head accuracy are NOT COMPUTABLE from the paired tool's eval dumps and say so"),
    # the register rows
    (REG, "**PASS (MEASURED 2026-10-28, a screening result, no amendment; all 27 reader control flags PASS and the rows recomputed from the per-window dumps).**"),
    (REG, "**NULL (label NULL-ONE; MEASURED 2026-10-28, T1-kin self-action open loop, Amendment 1 in force; all 158 reader control flags PASS, no guard fires"),
    (REG, "| **H-V8F-LAUNCH-INTENT-1** (2026-10-28) |"), (REG, "**PRE-REGISTERED, NOT RUN (2026-10-28).** The LAUNCH-INTENT branch of `PREREG_V8F_DAG_ORC` section 8 (NEXT-CRAWL), routed here by its Amendment 3"),
    (REG, "reader built and self-tested (87 tests GREEN, 245 / 245 mutants killed, dry run reproduces DAG-ORC's banked tables with 0 differences), not run."),
    (REG, "| MEASURED (NULL) | `Project Steering/PREREG_V8F_SPEEDSUB_T1.md` |"), (REG, "| MEASURED (screening) | `Project Steering/PREREG_V8F_SPEEDSUB.md` |"),
    (REG, "| MEASURED (HARMS) | `Project Steering/PREREG_V8F_LAUNCH_INTENT.md` |"),
    (REG, "**M-LAUNCH NOT MOVED (MEASURED 2026-10-28, T1-kin, 327 GT-launch windows / 49 episodes, paired episode-cluster bootstrap, B = 2,000)"),
    (REG, "section 11 routes to LAUNCH-INTENT-HI (W 4.17852, its own PI go)"),
    (REG, "**HARMS (MEASURED 2026-10-28, T1-kin, Amendment 1 in force; all 206 controls PASS; arm A not adopted).**"),
    (REG, "Committed next (section 10): B (W 0.726175, 4 box-2 runs) with its own PI go"),
    (REG, "the separated ADE6 cells flip sign between the two same-seed pairs, the pattern the pair design exists to expose (`H-ESTIM-SEED-1`)"),
    (REG, "Beside: the training draw alone (REP − DAG-OACC s0) is −0.5560 [−0.6994, −0.3975] m/s."),
    # PREREG_V8F_LAUNCH_INTENT.md, Amendment 1 (appended at e28516e8, before any LAUNCH-INTENT row was read)
    (PLI, "## Amendment 1 (A1-2026-10-28): K-BUILD's data-module clause, before any LAUNCH-INTENT row is read"),
    (PLI, "**Written 2026-10-28 ~15:10 by the Master Mind.**"),
    (PLI, "stack/tanitad/data/v8f_mini_data.py` at the launch commit `d8c9f7bf` has sha256 `c117e1cf…751752`."),
    (PLI, "twelve hunks, one adding the module constant `EGO_PERTURB_COMPOSED_STREAM` and eleven inside `EgoPerturb`"),
    (PLI, "The reader changes in that clause alone: `BYTE_IDENTICAL_FILES` becomes the pin `LAUNCH_PINNED_FILES`."),
    (PLI, "No bar, row, outcome, predicate or other control changes."),
    (PLI, "The launch-gate tokens bind the PREREG as promoted (sha256 `469bac1f…`) and the reader as landed (`5e950dcf…`)."),
]
_missing = [ph[:110] for src, ph in PHRASES if src.count(ph) < 1]
assert not _missing, _missing

PHRASES2 = [
    (T1, "The pre-registered reader `speedsub_t1_read.py` (sha256 `9d3c2792…`)"), (T1, "committed at `266b470e` (11:12:21)"), (T1, "C-COVER 306,566 of 306,566 TRAIN windows carry a finite past (causal-bucket agreement with the logged bucket 76.7 %)"),
    (T1, "(INHERITED, T0 diagnostic; `raw/2026-10-28-speedsub-t1-prereg/rep_s0_diag/REP_S0_SPEED_BIAS.md`, sha256 `211601de…`)"), (T1, "`lon_v_target_ms` −1.1097 [−1.2814, −0.9345]"),
    (T1, "from 0.2885 to 0.3306"), (T1, "The two halves of the hit rule name the same single run"),
    (T1, "† A rounded bound of the cell is exactly 0.0000."), (T1, "Without G-VIOL the word is the same (NULL, NULL-ONE)"), (T1, "against the PREREG's ESTIMATE of about 305 s"),
    (T1, "It does **not** recompute the cells from the per-window dumps or re-run the bootstrap"), (T1, "C-FIT PASS at 9.412e-06 m/s against the 1e-5 tolerance"),
    (T1, "PREREG section 8 (b), the per-cell attribution at T1-kin"),
    (T1, "arm A is separated-better on every crawl row at both seeds"), (T1, "**Single copies.**"),
    (T1, "REP s1 and A sit with CTRL."), (T1, "The planner is deterministic, so there is no inference-seed term."),
    (PT1, "The causal fit loses the oracle part of `limit`"), (PT1, "`FINDING_1` measured 9.41e-6 on this path"),
    (PT1, "**Tier of every decisive row: T1-kin (self-action open loop: m = 10, causal bucket, dead-reckoned nav, logged scene, periodic, executor"),
]
_missing = [ph[:110] for src, ph in PHRASES2 if src.count(ph) < 1]
assert not _missing, _missing

PHRASES3 = [
    # RESULT_V8F_LAUNCH_INTENT.md
    (RES, "all **206 reader controls PASS**, wall 341.6 s"), (RES, "**HARMS.** Arm A is not adopted. It failed its pre-registered bar: the decisive R-MOVE is false at both seeds, and the guard R-ADE fails at both seeds."),
    (RES, "Nothing below softens that."), (RES, "No crawl row moves by its bar in both pairs at either seed."), (RES, "All 16 X − P10 crawl cells are separated-worse."),
    (RES, "2 of the 16 crawl cells are separated-better and 1 clears its bar: L2 curvature in the replicate pair at s0"), (RES, "Its main-pair partner reads -0.0075 [-0.0212, +0.0025] –."),
    (RES, "At each seed the separated ADE6 cells flip sign between the main pair and the replicate pair."), (RES, "R-KEEP reads the point against +0.0604 m and holds in all four pairs."),
    (RES, "**R-NEW and R-NOBASE hold because condition (1) fails, not because condition (3) does.**"),
    (RES, "No arm is labelled. The largest shift in magnitude over the 20 cells is 0.1627 m/s, and 13 of the 20 cell intervals exclude 0."),
    (RES, "**M-LAUNCH: NOT MOVED** (MEASURED 2026-10-28, `raw/2026-10-28-launch-intent-mlaunch/MLAUNCH.json`, sha256 `f5ebb4b2…`"), (RES, "Section 11 label: **not moved** (moved needs all four cells separated toward 0)."),
    (RES, "Route: **LAUNCH-INTENT-HI** (W_HI 4.17852, four runs, its own PI go)."), (RES, "about 16 times the lever's largest point"),
    (RES, "Both routes need a PI go: B, registered after HARMS (section 10), and HI, the branch M-LAUNCH selects (section 11). The PI is asked which, if either, box 2 runs."),
    (RES, "The A chain ran 13:54 to 16:48 on box 2 (174 min, 2.9 h; MEASURED, `chain_v8f_launchintent.out`)"), (RES, "The twelve steps sum to 1756 s."),
    (RES, "The stopped launch's last line, 15:07:07, reads *\"LOW RAM available 3040 MB < 4096 MB\"*"), (RES, "It was relaunched with a 2,048 MB guard."), (RES, "**The T1-kin chain was stopped once, on its own RAM guard.**"),
    (RES, "**Pin slip (dated note).**"), (RES, "They were filled ~13:50, before commit `d8c9f7bf` (13:51:52) and after the PREREG was promoted at `77234276` (13:49:00)."),
    (RES, "**Amendment 1 (`A1-2026-10-28`), before the read.**"), (RES, "The replicate pair is unaffected: A_rep and REP have the same data module."),
    (RES, "(204 cells, asserted by `check_result.py`)"), (RES, "B and HI are not read."), (RES, "at 17:01:36 (chain log), with one failed control, K-BUILD, and wrote nothing"),
    (RES, "A_0's three T1-kin steps ran from 15:13:04 to 15:22:14"), (RES, "First stamped line 13:54, last 16:48: 174 min, 2.9 h."), (RES, "A label inside the reader."),
    (RES, "The per-episode dumps and tick arrays are under `F:/TanitAD-scratch/launch_intent_t1_out/`"), (RES, "**Disclosed coupling.** DAG-ORC and LAUNCH-INTENT share the REP_s runs."),
    (RES, "It does **not** recompute the cells from the per-window dumps or re-run the bootstrap"), (RES, "STRATEGIC is **not paused** for v8F (PI ruling 2026-10-17)."),
    # the box-2 chain log (CRLF removed): the launch facts
    (CHAIN, "10-28 13:54 predecessor verdict: GO the DAG-ORC r2 chain ended DONE and released the box: ZZV8FMINI-DONE 10-28 07:44 tip 8c24823f2b1bd24332148ebe434c870bcf3662f8"),
    (CHAIN, "10-28 13:55 box 2 code = tip d8c9f7bfc91798b6fafb5a13dd793771e2f79f6a"), (CHAIN, "descends from the landed flag b09ac89f"), (CHAIN, "carries the reviewed trainer (sha256 0d708b3c"),
    (CHAIN, "ZZV8FMINI-DONE 10-28 16:48 tip d8c9f7bfc91798b6fafb5a13dd793771e2f79f6a"), (CHAIN, "B is not queued"),
    # R2C_NOTE.md (the Master Mind's incident note, in the repo)
    (R2C, "**Unaffected.** A s0 and REP s0 trained and evaluated before 05:56."), (R2C, "branch `agent/dagorc-r2-pin`, created at `8c24823f`"), (R2C, "A s1 and REP s1"),
]
_missing = [ph[:110] for src, ph in PHRASES3 if src.count(ph) < 1]
assert not _missing, _missing

# ---- 7.27 SPEEDSUB ------------------------------------------------------------------------------------------------------------
SEC_SS = """### 7.27 SPEEDSUB: an ego-only read-out put into O's five speed cells cuts ADE6 on M2b-mixo (PASS, a screening result)

* **Setup** (`PREREG_V8F_SPEEDSUB`, no amendment; promoted by the Master Mind and landed `94a47f4b` before any SPEEDSUB number; claim row `H-V8F-SPEEDSUB-1`).
  The question, left open by SPEEDREAD (§7.24): that read's beside table showed that the ego-only MLP `mlp_ego` has the lowest fold-0 hold-out RMSE of the four fitted
  read-outs and the lowest eval RMSE in all five cells, but it was never substituted. Does `mlp_ego`, put into the five speed cells O reads, cut O's own-code ADE6 by at
  least F/2 = 0.06035 m, separated, at both seeds?
  * **Base and arms.** X_s = `M2b-mixo_s` (the SPEEDCAL / KINDX / SPEEDREAD base; §7.24), **not** DAG-OACC. sha256 `c6a62876…` (s0) and `68a0f335…` (s1). No model is trained,
    and the read-out is the banked one (`readout.pt`, sha256 `a205609a…`, no re-fit). Modes per seed: `own`; `identity` (control S1); `gt` (the GT cells, a T0 ceiling and a
    control, never a result); `ego` (the arm); `egoshuf` (the same cells shuffled across the 2,399 windows by `numpy.random.default_rng(20261029).permutation(2399)`; the
    deliberate-regression arm, which must read separated-worse at both seeds).
  * **What `ego` reads.** The 22-dim ego block at the tick frame: the 8 P10 kinematic features, the 5 `nav` channels, the 4-wide speed-limit bucket `limit` and its validity
    `limit_ok` (1), and 4 history-validity flags. T reads the same block at inference. No pixel, label, pose, goal or situation-classifier output enters.
  * **Selection disclosure.** The arm was commissioned after SPEEDREAD's beside table (which holds eval-141 RMSE) was read. The selection rule was restated on TRAIN fold 0
    only, before any SPEEDSUB number (control S13: mean fold-0 hold-out RMSE `mlp_ego` 0.2475 against `lin` 0.2724, `lin_ego` 0.2807, `mlp` 0.2920). It is still the third
    read-out examined on these 141 episodes, so a PASS is a screening result.
  * **Tier and estimator.** The instrument's tier stamp is *T1-class open loop (X_s's own held S / T outputs; only the five speed cells O reads are replaced)*; `gt` is T0, a
    control. Paired episode-cluster bootstrap, B = 2,000, seed 0, 141 episodes / 2,399 windows per seed. It answers *"would another draw of EPISODES say this?"* only. There is
    **no replicate arm**: the two seeds are O's two training runs (n = 2), and one read-out fit serves both. The planner is deterministic. F = 0.1207 m is the cross-seed
    spread of M2b (**[I]**, `RESULT_V8F_GRID.md`).
  * **The declared open channel (`D-SPEEDBAND-EGO-ECHO-1`) is what the PASS rests on.** The PASS was read with the logged `limit` lane. That lane is the bucket of the ego's
    own realised future max speed, which bin plus `v0` recovers at R² 0.9702 (**[I]**, the register). By PI ruling it is admissible as a declared input and never as a
    training signal, and T reads it too. The read-out's skill depends on it (the attribution below), and that is why SPEEDSUB-T1 (§7.28) followed: the T1-kin lane feeds a
    causal bucket instead.
* **Outcome [M]** (`RESULT_V8F_SPEEDSUB.md`, landed `af2dee4a`; the reader `subread` ran 08:05:36–08:05:38 and printed `ZZSPEEDSUB-DONE PASS`; all **27 control flags PASS**;
  `SUBREAD.json` sha256 `1fe7fcb3…`; `read/REDERIVE.json` recomputes the three paired rows per seed from the per-window dumps and gets identical delta, lo and hi to 4 dp on all 18 values):

<<SS_OUT>>

  The pre-registered conditions (REFUSED first, then HARMS, then PASS), against the rows:

<<SS_COND>>

  The outcome is **PASS**: at both seeds `ego − own` is separated-better and its point lies at or below −0.06035 m. The means are the dumps' `raw.traj.ade6_m.mean` (4
  decimals), so a delta can differ from the difference of two printed means by 0.0001. CAPTURE is `ego` / (`gt` − `own`).
  ⚠ **How to read the PASS** (none of it moves the word). (1) It is a screening result by the PREREG's own terms: a statement about two O training runs and one read-out fit at
  T1-class open loop, not a lever claim for any trained arm and not a driving result. (2) s1 clears the bar by 0.0083 m only (−0.0686 against −0.06035); at both seeds the
  interval's upper end (−0.0548, −0.0232) lies above −0.06035, so the intervals do not exclude an effect smaller than the bar. (3) The two seeds' effects differ by 0.0604 m
  (−0.1290 against −0.0686): the only look this experiment has at training-run variance, a description and not an estimate. (4) `ego` recovers 18 % and 10 % of what the GT
  cells would give (CAPTURE 0.1801 / 0.1028), so O leaves most of the 0.7 m on the table.
* **Controls [M]** (each ran before its rows were read; 27 flags, all PASS; `egoshuf` reads separated-worse at both seeds):

<<SS_CTRL>>

  The family tools' own controls hold at both seeds: LL-1 (identity reads exactly 0.0 on every row), LL-2 (its ADE6 row equals `SUBREAD.json`), LL-3 (per-arm values equal the
  dump's own; largest deviation 4.99e-05 against a bar of 1e-4) and `pairfam` (identity on 45 cells, reproduction of 119 point values per arm).
* **Attribution of the read-out's skill [M]** (FINDING_1, banked at `842ec2ac` as `R/raw/2026-10-28-speedsub-t1-prereg/speedsub_t1_prep/FINDING_1_ATTRIBUTION.md`, sha256 `e89af4a0…`; the SPEEDSUB RESULT cites it at its
  dev-box path; TRAIN fold 0, 846 clips, not eval-141 and not ADE6; mean hold-out RMSE over the five cells in train-SD units, lower is better):

<<SS_ATTR>>

  The skill survives without `nav` and does not survive without `limit`: dropping `nav` costs 0.0156 SD, dropping `limit` costs 0.0328 SD, and the no-`limit` read-out (0.28032)
  sits above the best other banked read-out on this fold (`lin`, 0.2724). No reduced read-out was substituted, so **which input carries the ADE6 cut is not measured here**.
  **[H]** If the ADE6 cut follows the read-out's skill, it rests mostly on the declared open channel.
* **The four families [M]** (beside the decision; per family, per seed, never pooled). Tier: T1-class open loop; the `gt − own` table is the T0 ceiling and a control.
  Estimator: paired episode-cluster bootstrap, B = 2,000, seed 0, 141 episodes; one training run per seed and no replicate arm, so a separated cell is a pointer
  (`H-ESTIM-SEED-1`). **STRATEGIC is reported, not paused, for v8F.** B / W = separated-better / -worse, – = not separated, S+ / S− = separated on a row with no better
  direction (a bias or a ratio), bold = separated. Error rows are lower-better; accuracy and F1 rows are higher-better.

  **ego − own** (the arm; point [95 % CI]):

<<SS_EGO>>

  **gt − own** (T0 ceiling, a control only; point [95 % CI]):

<<SS_GT>>

  What is not computed, per family, with the reason and n:

<<SS_GAPS>>

* **What the rows say [M]** (beside; none of it moves the outcome). For `ego − own`, 11 of 49 cells are separated at s0 and 9 of 49 at s1, where at a nominal 5 % about 2 to 3
  would separate with no effect, so the separated cells are pointers.
  1. **LONGITUDINAL: the three registered rows move, at both seeds.** Speed MAE, along-track MAE and accel MAE are separated-better. The `ego` change is 18.7 % (s0) and 11.1 % (s1)
     of the `gt` ceiling on speed MAE, 18.4 % / 11.1 % on along-track MAE and 92.7 % / 97.4 % on accel MAE (the ceiling moves accel MAE by only 0.05 and 0.03 m/s²). Distance
     keeping and the anti-echo block are UNAVAILABLE (n = 0), so half of the LONGITUDINAL family is unmeasured for this arm: a work item, not a pass.
  2. **LATERAL: pointers only, none worse.** L3 yaw-rate MAE (GT ≥ 5.2 m/s) is separated-better at both seeds (−0.5435 / −0.3193 deg/s); cross-track MAE is not separated
     (+0.0020 / +0.0021 m).
  3. **TACTICAL: the decisions do not move; the goal point does.** TAC-LAT, TAC-LON and the 5-way manoeuvre accuracy are not separated at either seed. The goal-point error is
     separated-better at both seeds (−0.3943 / −0.1950 m). The three head macro-F1 rows read exactly +0.0000 [+0.0000, +0.0000] at both seeds, because the substitution acts on the
     code O reads, after T has run. The goal-point cells are read from O's plan, so they are not an independent confirmation of ADE6.
  4. **STRATEGIC: exactly zero, by construction.** All six S rows read +0.0000 [+0.0000, +0.0000] at both seeds, for `ego` and for `gt`: a property of the substitution site and
     not evidence about strategic quality.
  5. **The 2 s horizon does not improve.** ADE2 is not separated at either seed and its point is positive: +0.0080 [−0.0033, +0.0181] / +0.0039 [−0.0013, +0.0092] m. The gain is a
     long-horizon gain.
* **The read-out's own accuracy [M]** (beside; eval RMSE to the GT, m/s, on the eval-141 windows where the GT cell is present). `mlp_ego` is below `T_own` in all five cells at
  both seeds. SPEEDREAD's ridge was also lower than `T_own` in `v_a_ms` (RMSE cut 40–59 %) and moved ADE6 by about 0 (**[I]**, `PREREG_V8F_SPEEDSUB`), so a lower cell RMSE was not
  a sufficient condition; here it coincides with an ADE6 cut at both seeds. Which cell carries the cut is not known from this experiment.

<<SS_CELLS>>

* **Beside, post-hoc and not registered [M]: the plan's rule-violation rate rises under `ego`.** The harness scores every plan against constraints built from the GT (rules SPEED,
  STOP, TURN and LEAD); rate = the share of the 2,399 windows that break the rule, lower is better; same estimator. The block decides nothing, and it is the reason SPEEDSUB-T1
  carries a rule-violation guard (Amendment 1, §7.28):

<<SS_VIOL>>

  The rate of plans that break at least one rule is separated-worse at both seeds (+0.0217 [+0.0050, +0.0404] / +0.0192 [+0.0038, +0.0346]), carried by the LEAD rule at both seeds,
  while the `gt` cells move it the other way. On the 81 stop windows `ego` is worse than `own` at both seeds (ADE6 +0.3374 / +0.1762 m, point values with no interval, because the
  strata's window lists are not in the dumps).
* **What this read cannot claim.** (1) It is not a driving result: T1-class open loop, five cells replaced, and ADE6 at this tier is not a capability claim. (2) It is not a lever
  claim for any trained arm: two O training runs, one read-out fit and no replicate arm, and a separated interval from a one-seed arm is necessary and not sufficient
  (`H-ESTIM-SEED-1`). (3) It is not a statement about DAG-OACC: the base is M2b-mixo, and DAG-trained O has seen its own rollouts. (4) It is read with an open channel, so it is not
  a causal-bucket result. (5) The `egoshuf` control has no family run.
* **Next (Rule Zero).** The PASS leaves a committed next, verbatim from `SUBREAD.json` `next`: *"T1-kin zero-training substitution of the banked ego read-out on the DAG-OACC / DAG-ORC
  base, both seeds, under its own PREREG (PREREG_V8F_SPEEDSUB section 6)"*. Its prerequisites ran first: the attribution above, and the finding that the T1-kin lane feeds
  `--limit causal`, so the follow-up is a refitted read-out. It was pre-registered as `PREREG_V8F_SPEEDSUB_T1` (landed `842ec2ac`, before its read-out was fitted and before any T1-kin
  rollout) and is read in §7.28.
* **Provenance.** Dev-box CPU (`CUDA_VISIBLE_DEVICES=-1`, 4 threads), 2026-10-28 07:59:08 to 08:17:43 (18 min 35 s); no stage stopped and none re-ran. Instrument
  `stack/scripts/v8f_mini_eval_speedread.py`, sha256 `a95f4fb1…`. Banked in `R/raw/2026-10-28-speedsub-result/`. **Single copy on F: (dev-box scratch), not in the repo:** the per-mode
  `eval_*.npz` and `speedread_cells_*.npz` dumps and the run tree; if F: is lost, controls S5, S6 and S7b and the family tools cannot be reproduced.
"""

# ---- 7.28 SPEEDSUB-T1 ---------------------------------------------------------------------------------------------------------
SEC_T1 = """### 7.28 SPEEDSUB-T1: the causal ego read-out does not move healthy DAG-OACC draws at T1-kin (NULL, label NULL-ONE)

* **Setup** (`PREREG_V8F_SPEEDSUB_T1`, promoted and landed `842ec2ac` before any T1-kin substitution rollout and before its read-out was fitted; **Amendment 1**, `A1-2026-10-28`, landed
  `4cec09e7` at 10:17:23, before the smoke at 10:21:44 and before any rollout of this experiment; claim row `H-V8F-SPEEDSUB-T1-1`, with the floor rule `H-V8F-SPEEDSUB-T1-2` beside it).
  The question: SPEEDSUB's PASS (§7.27) was read with the logged `limit` lane, the declared channel that carries the ego's own future speed. The T1-kin lane that scores the adopted
  base feeds `--limit causal`, the bucket of the past 20 rows of measured ego speed. Does a causal read-out in O's five speed cells, at every replan tick, move DAG-OACC?
  * **Read-out and arms.** `mlp_ego_c` is the `mlp_ego` design refitted once on the causal bucket (sha256 `799f65ed…`, fitted before Amendment 1 and before any rollout; C-FIT passes
    at 9.412e-06 m/s against the tolerance 1e-5, a thin margin that is disclosed). X = DAG-OACC at four runs: CTRL s0 / s1 (`67265ec1…`, `90b01474…`) and the DAG-ORC replicates REP s0 /
    s1 (`b03fd6ef…`, `77853e56…`). Per run: `identity` (the paired base), `egoc` (the arm) and `egoshufc` (the cells permuted across the 2,399 windows, sha256 `64785f8a…`; the
    deliberate-regression arm, which must read separated-worse at all four runs: K-REG). Beside, CTRL only: the logged lane with the banked `mlp_ego`.
  * **Bar, hit and guards.** A hit at a run is `egoc − identity` ADE6 separated-better and at or below −0.06035 m (F/2). PASS needs a hit at all four runs. Guards: G-ADE (not
    separated-worse at any run), G-WORSE (no Read-A row separated-worse at 3 or more of the 4 runs), G-NEW (no row becomes separated-worse against P10 at both CTRL seeds) and, from
    Amendment 1, G-VIOL (`VIOL-ANY`, the executed path's rule-violation rate, not separated-worse at 3 or more of the 4 runs), added because SPEEDSUB's post-hoc block (§7.27) found
    that rate rising under `ego`. The order is REFUSED, HARMS, PASS, PARTIAL (hits at both CTRL seeds) and NULL (anything else; NULL-ONE and NULL-FLAT describe and decide nothing).
    The PREREG's ESTIMATED prior was that PASS is not the expected outcome, because the causal fit loses the oracle part of `limit`.
  * **Tier and estimator.** T1-kin: a self-action open loop (m = 10, causal bucket, dead-reckoned nav, logged scene, periodic, executor plan), not T2. Paired episode-cluster
    bootstrap, B = 2,000, seed 0, 141 episodes / 2,399 windows, per run. It answers *"would another draw of EPISODES say this?"*. The four runs (two seeds × two replicates of one
    recipe) carry *"would another TRAINING RUN say this?"* at n = 4 (`H-ESTIM-SEED-1`). The read-out is fitted once, so *"would another READ-OUT FIT say this?"* is not answered. The
    planner is deterministic, so there is no inference-seed term.
* **Outcome [M]** (`RESULT_V8F_SPEEDSUB_T1.md`, landed `37d78ed1`; read committed at `266b470e` (11:12:21) by the pre-registered reader `speedsub_t1_read.py` (sha256 `9d3c2792…`), which
  printed `ZZSPEEDSUB-T1-READ-DONE NULL NULL-ONE`; **all 158 control flags PASS**; output `raw/2026-10-28-speedsub-t1-read/SPEEDSUB_T1_READ.json`, sha256 `b3b72325…`; chain run 2026-10-28
  10:21:41 to 11:09:29 (47 min 48 s) on the dev-box CPU, the reader's own wall 454.4 s. The word is not copied: the RESULT's generator recomputes the hits, the four guards and the floor
  rule from the per-run cells and stops on any mismatch with the reader):

<<T1_OUT>>

  The table's walk (REFUSED first, then HARMS, then PASS, PARTIAL, NULL):

<<T1_WALK>>

  The guards, recomputed from the cells:

<<T1_GUARDS>>

  The outcome is **NULL**, label NULL-ONE: one hit of four runs, at the slow-T draw REP s0, and not at both CTRL seeds. Committed next, verbatim from the read: *"next: section 8 (a), the
  history-aware causal read-out"*. Without G-VIOL the word is the same, and under Amendment 1's reading of G-NEW it is the same. The two halves of the hit rule name the same single run
  (separated-better alone gives REP s0 only, and the bar alone gives REP s0 only).
  ⚠ **How to read the NULL** (none of it moves the word). NULL means the bar was not cleared at all four runs; it does not mean the effect is zero. At CTRL s0 and REP s1 the lower ends
  of the intervals (−0.0629, −0.0832) lie beyond −0.06035, so the data at those runs do not exclude an ADE6 cut as large as the bar; at CTRL s1 the lower end (−0.0500) lies inside the bar,
  so a bar-sized cut is excluded there at the 95 % level (the episode question). All three points are small and their intervals include 0.
* **Controls [M]** (each must read its known value before any row is read; any failure is REFUSED). The deliberate-regression arm reads its known value: `egoshufc − identity` ADE6 is
  +4.2928 [+3.4386, +5.1945] (CTRL s0), +3.9857 [+3.2389, +4.7695] (CTRL s1), +4.4851 [+3.5747, +5.4322] (REP s0) and +4.1886 [+3.3996, +5.0549] m (REP s1), separated-worse at all four
  runs, so the small `egoc` movements at the healthy draws are not a blind instrument:

<<T1_CTRL>>

  Before any rollout (**[I]**, MEASURED in the build, `raw/2026-10-28-speedsub-t1-build/evidence/fit_c.json`): C-FIT PASS at 9.412e-06 m/s against the 1e-5 tolerance; C-CAUSAL 0 mismatches
  against the tool's own `build_item --limit causal`; C-COVER 306,566 of 306,566 TRAIN windows carry a finite past (causal-bucket agreement with the logged bucket 76.7 %); C-DISJ eval ∩ fit
  clips = ∅; C-PARITY 4,512 clips / 306,566 windows.
* **The rows, `egoc − identity` per run, on every row [M]** (T1-kin; point [95 % CI]; B / W = separated-better / separated-worse, – = not separated; each cell is the paired
  episode-cluster bootstrap of the arm minus the same run's `identity` rollout on the same windows, B = 2,000, seed 0; error rows are lower-better, TAC rows higher-better; units are those of
  the DAG-ORC reader's rows, §7.26; † on REP s0 DK-VIOL: a rounded bound of the cell is exactly 0.0000, and the flag is the estimator's reading of the unrounded bound):

<<T1_ROWS>>

  The census of the table, per run (DERIVED; 18 computed rows per run):

<<T1_CENSUS>>

* **The four families [M]** (T1-kin; per run, per family, never pooled; **STRATEGIC is reported, not paused, for v8F**). The rows table above carries every family row against the run's
  own `identity`. Each family's standing against the P10 floor (`egoc − P10`, same estimator and windows):

  LONGITUDINAL (speed, along-track and accel MAE; DK-VIOL / DK-HW / DK-TG on the LEAD windows):

<<T1_LONG>>

  LATERAL:

<<T1_LAT>>

  TACTICAL:

<<T1_TAC>>

  STRATEGIC. S-POS@30 / 50 / 80 is the strategic head's position error within a run (S − CTRV, ticks ≥ 1; unpaired against CTRV; direction lower is better). **The paired
  `egoc − identity` S-POS row is NOT COMPUTED**, because the reader scores S against CTRV inside each rollout and does not pair the two rollouts' S errors window by window. A
  difference between the two columns below is therefore not a tested contrast, and STRATEGIC carries no across-arm statement in this read:

<<T1_SPOS>>

  What is not computed, per family, with the reason:

<<T1_GAPS>>

* **What the numbers say [M]** (beside the word; none of it moves it).
  1. **The one hit is the repair of a slow-T draw.** At REP s0 `egoc` cuts ADE6 by −0.1949 [−0.2619, −0.1313] m, 3.23 times the bar. Of the 18 computed rows, 8 are separated-better and 2
     separated-worse. The cost is rule violation: the share of windows on which the executed path breaks a SPEED / STOP / TURN / LEAD rule rises from 0.2885 to 0.3306 (`VIOL-ANY` +0.0421
     [+0.0246, +0.0612]), so G-VIOL, which needs 3 of the 4 runs, fires at 1. M1 crawl share does not separate (−0.0052 [−0.0115, +0.0007]): the repair is of the speed cells' draw, not
     of the crawl as such.

<<T1_REPS0>>

  Why REP s0 (**[I]**, T0 diagnostic; `raw/2026-10-28-speedsub-t1-prereg/rep_s0_diag/REP_S0_SPEED_BIAS.md`, sha256 `211601de…`): REP s0's T predicts every speed cell slow against CTRL s0
     on the same 2,399 windows, separated: `lon_v_target_ms` −1.1097 [−1.2814, −0.9345], `v_a_ms` −0.7315 [−0.8635, −0.5901], `v_end_ms` −1.0056 [−1.1436, −0.8686], `v_lo_ms` −1.1700
     [−1.3210, −1.0175] and `v_hi_ms` −0.9913 [−1.1383, −0.8487] m/s. REP s1 and A sit with CTRL. The diagnosis is the Master Mind's, written before this read; the substitution helps at the
     draw whose cells are most wrong and at no other run, which is what it predicts. It is **one** slow-T draw (n = 1).
  2. **The floor rule `H-V8F-SPEEDSUB-T1-2`: NOT COLLAPSE.** COLLAPSE needs `g_id(r) > 0` and `g_c(r) ≤ 0.5 · g_id(r)` on all three rows (g_id = REP s0 − CTRL s0 under `identity`, g_c = the
     same under `egoc`). The REP s0 − CTRL s0 gap on ADE6 falls from 0.4025 to 0.2309 m (a drop of 42.6 %), yaw-rate MAE and curvature MAE fall by less, and no row reaches half, so the floors
     stay DAG-ORC's (§7.26). At s0 (decisive, beside):

<<T1_FLOOR0>>

  At s1 (reported) the rule cannot collapse what is not there (REP s1 − CTRL s1 is about 0 under `identity`, and curvature MAE has g_id below 0):

<<T1_FLOOR1>>

  3. **The healthy draws show no separated ADE6 change.** CTRL s0 and REP s1 move no row of 18. CTRL s1 moves the wrong way on three rows, each at that one run (accel MAE +0.1598
     [+0.0743, +0.2611], yaw-rate MAE +1.3443 [+0.2401, +2.5747], L1 lateral-accel MAE +0.3627 [+0.1230, +0.6572]), so G-WORSE, which needs 3 runs per row, does not fire. **M1 crawl share moves at
     no run.**

<<T1_HEALTHY>>

  4. **How much of the crawl gap to P10 `egoc` closes** (DERIVED, in DAG-ORC's form: (identity − P10 − (egoc − P10)) / (identity − P10); negative means the gap widens; DAG-ORC's arm A for
     reference, **[I]**, §7.26). At the healthy CTRL seeds `egoc` closes at most 12.8 % of the gap on any crawl row and widens it on 3 of 8 cells, against 27.6 % at the least for DAG-ORC's
     perturbation. The two tables are not one experiment, so the comparison is of sizes and not a contrast:

<<T1_SHARE>>

  5. **Beside, deciding nothing: the logged-lane bridge.** The banked `mlp_ego` in the `--limit logged` lane (a declared open channel), CTRL only. The read-out with the most future information
     available moves ADE6 by +0.0008 and −0.0185 m at CTRL s0 and s1: not separated. On DAG-OACC at T1-kin the same read-out in the same lane does not move ADE6, where on M2b-mixo at T1-class
     open loop it did (§7.27). The change has four ingredients at once (the base, the tier, the rollout machinery and the checkpoint seeds), and no arm isolates one. **[H]** (an inference from the two flat reads) Because the
     best-informed read-out is also flat at CTRL, the causal read-out's flatness is not explained by the future information it lacks:

<<T1_BRIDGE>>

* **Next (Rule Zero).** The committed next, verbatim from the read: *"next: section 8 (a), the history-aware causal read-out"* (the four history frames of `p10f` plus the causal bucket and
  nav, under its own amendment with the same table; ESTIMATED at most about 1.6 h of dev-box CPU in all, no GPU and no spend). The Master Mind's ranking, labelled ESTIMATED / INFERENCE (not
  a measured outcome; **[I]**, `RESULT_V8F_SPEEDSUB_T1.md` §7):
  1. **(a) stays registered and is not run now, because arithmetic predicts NULL.** The oracle bucket, the read-out with the most future information (TRAIN hold-out 0.2475 SD units against
     0.2793 for `mlp_ego_c`), did not move healthy CTRL, and a history-aware causal read-out carries less future information than that oracle. **So the speed-cell channel is not what limits
     healthy DAG-OACC draws.** Its measured use is repairing slow-T training draws (REP s0).
  2. **The largest measured crawl lever is DAG-ORC's perturbation** (§7.26): arm A is separated-better on every crawl row at both seeds, while `egoc` at the same CTRL seeds moves M1 by −0.0007
     and +0.0021, not separated. The registered crawl route is LAUNCH-INTENT on box 2 (§7.29), which waited on the PI's go.
  3. **(a) runs if a later arm shows healthy-draw speed cells binding.** The RESULT's reading of binding, for the Master Mind to confirm: an arm whose healthy draws move ADE6 or a crawl row under
     a speed-cell substitution, separated and above the rig's replicate floor. PREREG section 8 (b), the per-cell attribution at T1-kin, is unblocked and not routed from NULL.

<<T1_BLOCKED>>

* **What this read cannot claim, and disclosures.** It is a self-action open loop, not T2 and not a driving result for the deployed car; the read-out is not built as a sub-head. There is one
  read-out fit. Ticks 1 and 2 read the model's own executed motion, outside the fit's distribution by construction. `egoshufc` is a global shuffle, a control with a committed reading and not an
  estimate of the read-out's information. REP s0 is a slow-T draw, so its hit partly measures repair of that draw, and the four runs are two seeds × two replicates of one recipe, not four
  recipes. Not computed or not built: S-POS across arms, the anti-echo block and goal-head F1 at T1-kin (work items). Amendment 1 precedes every rollout, and no guard fires under either reading
  of G-NEW, so the word does not depend on it. The fit was reused, not re-run, in the chain (made once at build time, with its sha256 pinned before the first rollout). The rollouts took 138 to
  141 s each against the PREREG's ESTIMATE of about 305 s; K-ID reads a maximum absolute difference of 0.0 against the banked rollouts, so the speed difference changes no number. The RESULT's
  re-derivation tests the decision logic and the consistency of the read JSON; it does **not** recompute the cells from the per-window dumps or re-run the bootstrap.
* **Provenance.** Dev-box CPU (`CUDA_VISIBLE_DEVICES=-1`, OMP / MKL 8, one rollout at a time), twelve decisive rollouts in 27 min 52 s, four beside rollouts. Banked in
  `R/raw/2026-10-28-speedsub-t1-read/` (the read JSON, the beside JSON `SPEEDSUB_T1_BESIDE.json`, pins, the runs manifest, the chain log) and `R/raw/2026-10-28-speedsub-t1-result/`.
  **Single copies.** The per-episode dumps, tick arrays and run trees are on `F:/TanitAD-scratch/speedsub_t1/out/` only; if F: is lost, the tables stay checkable against the read JSON but cannot
  be recomputed from the dumps.
"""

# ---- commits cited in the text: each exists, is an ancestor of the tip and carries its subject word ----------------------------
def commit_msg(sha):
    r = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*", "log", "-1", "--format=%H%n%s%n%b", sha], capture_output=True)
    assert r.returncode == 0, (sha, r.stderr[:200])
    return r.stdout.decode("utf-8")


for sha, word in (("94a47f4b", "SPEEDSUB"), ("842ec2ac", "PREREG_V8F_SPEEDSUB_T1"), ("af2dee4a", "RESULT_V8F_SPEEDSUB"), ("4cec09e7", "Amendment 1"), ("37d78ed1", "RESULT_V8F_SPEEDSUB_T1"),
                  ("266b470e", "SPEEDSUB-T1 read"), ("051259d4", "Amendment 3"), ("b09ac89f", "--launch-intent-w"), ("77234276", "PREREG_V8F_LAUNCH_INTENT"), ("d8c9f7bf", ":launchintentarm")):
    assert word in commit_msg(sha), (sha, word)
    anc = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*", "merge-base", "--is-ancestor", sha, TIP], capture_output=True)
    assert anc.returncode == 0, ("not an ancestor of the tip", sha)
assert "390 / 390 tests, 185 / 185 mutants killed" in commit_msg("d8c9f7bf")
assert "`train_window_list_sha256` equals DAG-OACC's (`e144adcd…8168`)" in PLI and "`launch_n / launch_eligible` lies within ±0.02 of f = 0.1619" in PLI
_cn = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*", "show", f"{TIP}:{R}raw/2026-10-28-launch-intent-chain/NOTES.md"], capture_output=True).stdout.decode("utf-8")
assert "`launch_rows == launch_eligible`" in _cn, "the chain's K-MASK extra is not in its NOTES"
# the two banked files the text names with a sha256 prefix: the blob at the tip must hash to it, and the file must be in the commit that banked it
for _rel, _pre, _commit in (("raw/2026-10-28-speedsub-t1-prereg/speedsub_t1_prep/FINDING_1_ATTRIBUTION.md", "e89af4a0", "842ec2ac"),
                            ("raw/2026-10-28-speedsub-t1-prereg/rep_s0_diag/REP_S0_SPEED_BIAS.md", "211601de", "842ec2ac")):
    _b = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*", "show", f"{TIP}:{R}{_rel}"], capture_output=True)
    assert _b.returncode == 0 and hashlib.sha256(_b.stdout).hexdigest().startswith(_pre), (_rel, "sha256 prefix does not match the blob at the tip")
    _e = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*", "cat-file", "-e", f"{_commit}:{R}{_rel}"], capture_output=True)
    assert _e.returncode == 0, (_rel, "not in", _commit)

# ---- 7.29 LAUNCH-INTENT -------------------------------------------------------------------------------------------------------
USED.append("TITLE729")
HEAD_729 = "### 7.29 LAUNCH-INTENT: PRE-REGISTERED, RUNNING" if PREDRAFT else "### 7.29 LAUNCH-INTENT: " + FILL["TITLE729"].strip()

SEC_LI_A = HEAD_729 + """

* **Status of this section (v33).** """ + site("STATUS729", """`PREREG_V8F_LAUNCH_INTENT` is pre-registered and its four box-2 runs are running (**[I]**, run state below). **No LAUNCH-INTENT checkpoint, rollout or row is read in
  this version.** The design, the Master Mind's redesign of the bar and arms before any data, the outcome table, the controls and the registered routes are committed in the bullets
  below. The outcome bullets after them are marked slots, which a later run fills from `RESULT_V8F_LAUNCH_INTENT.md` once it lands; the patch refuses to write a final version while one is empty.""") + """
* **Design** (`PREREG_V8F_LAUNCH_INTENT`, claim row `H-V8F-LAUNCH-INTENT-1`; promoted by the Master Mind from its draft v2 (sha256 `aec8156a…`) and landed `77234276` as PRE-REGISTERED,
  NOT RUN, before any LAUNCH-INTENT gate, run, rollout or row; the PREREG as landed has sha256 `469bac1f…`; its Amendment 1 (`A1-2026-10-28`, landed `e28516e8`, before any row was read)
  appends to those bytes and gives the amended text sha256 `6ae79f2f…`, see the controls below).
  * **Why it runs.** It is the LAUNCH-INTENT branch of `PREREG_V8F_DAG_ORC` §8 (NEXT-CRAWL), re-based on DAG-OACC. DAG-ORC read HARMS (§7.26). The PI answered the question whether box 2
    should run the redesigned LAUNCH-INTENT next and skip B with *"Yes, queue it"* (2026-10-28), recorded as `PREREG_V8F_DAG_ORC` Amendment 3 (`A3-2026-10-28`, landed `051259d4`): B
    (p = 0.1) is not trained and not read, A stays HARMS, and NEXT-CRAWL on DAG-OACC's states is LAUNCH-INTENT under its own PREREG. The rows it aims at are the crawl set C: DAG-OACC − P10
    is separated-worse at both seeds on yaw-rate MAE, curvature MAE, L2 and M1 (**[I]**, `PREREG_V8F_DAG_ORC` Amendment 1). The mechanism it targets is O's own slow launch intent, v0
    −1.78 / −0.93 m/s at tick 1 in the launch windows (**[I]**, `DIAGNOSIS.md` via `PREREG_V8F_DAG_ORC` §8). **[H]** The PREREG's hypothesis is that the 2-D ADE averages 30 knots over every
    window, so the launch windows' along-track error is diluted about 5:1.
  * **The lever.** One trainer flag, `--launch-intent-w` (default 0.0, off), landed `b09ac89f`: an additive term, W times the along-axis, knots 1–12 slice of ADE's own per-knot distance,
    restricted to GT-launch TRAIN samples (v0 < 6 m/s and GT v(6 s) ≥ v0 + 1 m/s). Knots 1–12 are the executed horizon at m = 10, and nothing else gets a gradient from the term. The subset
    is label-side (future GT speed is admissible for labels) and nothing new enters at inference. With the flag off, the run is the tip's, bit for bit.
  * **The dose.** f = 49,618 / 306,566 = 0.161851 of the TRAIN windows are GT-launch (**[M]**, the PREREG's `f_measure.json`). Arm A gives launch samples one third of the along-track weight,
    W_A = 1.58926. Arm B (one quarter, W_B = 0.726175) is registered and not trained; arm HI (one half, W_HI = 4.17852) is trained only from NEXT-LAUNCH, with its own PI go.
  * **The redesign before any data (disclosed).** The draft's bar was 2 × max(F, RF0, RF), with RF the larger of the two REP draws, and REP s0 (a slow-T training draw, §7.28) set RF on 17 of
    19 rows. Against the whole gap DAG-OACC leaves to P10 on a crawl row, that bar was 1.3 to 9.1 times the gap (**[M]**, arithmetic on the banked `P0_RF0.json`, `RF_REP.json` and
    `DAGORC_READ.json`; the v1 bar ÷ gap column):

<<LI_GAP>>

  A lever that removed the crawl completely would have failed the draft's R-MOVE at both seeds: the decisive condition was unreachable by construction. The disclosure in the PREREG, verbatim:
    *"The bar and arms were changed by the Master Mind before promotion, after DAG-ORC's floors and the REP s0 diagnosis were known, and before any LAUNCH-INTENT number existed. The change follows
    from arithmetic on banked numbers (section 1). It is not a reaction to a LAUNCH-INTENT outcome, which does not exist."* There are two changes. (1) **The bar is 2 × max(F, RF0)**, one value per
    row, inclusive; RF stays beside every row and decides nothing. Each v2 crawl bar is below the CTRL − P10 gap at both seeds. (2) **Two same-seed one-lever pairs per seed**, and each per-seed
    predicate holds only when it holds in both pairs. ⚠ Stricter in one way: every guard must hold in both pairs of the seed. ⚠ Looser in one way: the size bar drops from 2 × RF to 2 × RF0,
    which is why replication in both pairs is required. The 19 bars (v2 bar = 2 × max(F, RF0); RF and the v1 bar are beside; units are the row's own):

<<LI_BARS>>

  * **Arms** (box 2; each through the trainer's own launch gate; both seeds). The main pair is A_s − CTRL_s (CTRL_s = DAG-OACC_s, banked); the replicate pair is A_rep,s − REP_s (REP_s banked from
    DAG-ORC r2):

<<LI_ARMS>>

  REP is reused under a licence the reader checks (the run finished and passed its run record; its launch-gate token matches DAG-OACC's init, bank, `w_acc` and window list; the T1-kin dumps follow
    the PREREG's settings; the build term). REP provenance: the launch-gate tokens of both REP runs bind commit `8c24823f` (**[M]**, `raw/2026-10-28-dag-orc-result/runs/M2b_s{0,1}_dagoaccorcrep/launch_gate_token.json`,
    trainer seeds 10 and 11 in the run records), and the DAG-ORC r2 chain log that the LAUNCH-INTENT chain waited for ended in its DONE marker at tip `8c24823f` (**[M]**, `chain_v8f_launchintent.out`, 13:54). The timing is
    the Master Mind's incident note `R2C_NOTE.md` (**[I]**, repo path `raw/2026-10-28-dag-orc-result/notes/`, sha256 `df02ff2f…`; §7.26): REP s0 trained and was evaluated before 05:56, and REP s1 was trained after the r2c
    relaunch, which read the branch pinned at `8c24823f`. DAG-ORC and LAUNCH-INTENT share the REP runs; RF is not a bar term here, so an unusually large or small REP draw cannot loosen or block this read's bar.
  * **Tier and estimator.** The decisive rollouts are T1-kin, a self-action open loop (m = 10, causal bucket, dead-reckoned nav, logged scene, periodic, executor plan), not T2: A_s and A_rep,s at
    both seeds on the dev-box CPU, with CTRL, REP, DAG and P10 banked. R-T0 and the T0 four families are read on box 2's standard T0 eval (eval-141, ages cycle), a world-model diagnostic and not
    driving performance. Paired episode-cluster bootstrap, B = 2,000, seed 0, 141 episodes / 2,399 windows, per seed and per pair. Three variances are named: *another draw of EPISODES* (the
    bootstrap), *another TRAINING RUN* (the replicate pair at each seed; a lever effect is credited at a seed only when both pairs show it, which across both seeds is four draws) and *another
    INFERENCE RUN* (none: the planner is deterministic).
  * **Rows, guards and the per-seed predicates** (`PREREG_V8F_DAG_ORC` §4 with its Amendment 1, evaluated in each pair): R-CRAWL and R-MOVE are decisive; R-TLAT, R-TLON, R-ADE, R-T0 (T0 ADE6
    point ≤ +F/2 = 0.0604 m) and R-NEW are guards; R-KEEP (T1-kin ADE6 point ≤ +0.0604 m) and R-NOBASE decide adoption. A row **clears** when X − Y is separated-better and the improvement is at
    least the bar. It **fires** when X − P10 is separated-worse, Y − P10 is not, and X − Y is separated-worse by at least the bar. The domain D of a pair is every Read-A row not separated-worse for
    the pair's control against P10, with no exempt rows.

<<LI_PRED>>

  * **The outcome table** (committed before any LAUNCH-INTENT row; the order is REFUSED, then HARMS, then R-MOVE and R-CRAWL, adoption last):

<<LI_OUT>>

  The table is complete and exclusive: 524,288 cells, each matching exactly one row (REFUSED 262,144; HARMS 261,888; FIXED / ADOPTED 1; FIXED / NOT ADOPTED 15; PARTIAL / ADOPTED 3; PARTIAL / NOT
    ADOPTED 45; PARTIAL / ONE SEED 128; NO EFFECT 64), by `enumerate_cells_li.py` (**[I]**, `PREREG_V8F_LAUNCH_INTENT` §10; the counts equal DAG-ORC's own).
  * **NEXT-LAUNCH, registered before any row.** The routing variable is M-LAUNCH, X − Y of the mean (v_own − v_GT) at ticks 1–2 in the GT-launch windows, from the tick dumps; it is moved when it
    is separated toward 0 in both pairs at both seeds:

<<LI_NEXT>>

  If both branches are exhausted, the programme has no registered lever for the crawl rows on DAG-OACC (**[I]**, `PREREG_V8F_LAUNCH_INTENT` §11 names the blockers: a new diagnosis of the loop
    term, or a PI decision on an architecture change that gives O a launch-intent input).
  * **Controls (committed; any failure is REFUSED).** K-GATE (the trainer's own launch gate on `--launch-intent-w`, G-DVB and G-LIVE, for every trained arm); K-PAIR, one per pair (argv equal apart
    from `--out` and the lever; REP's argv equals DAG-OACC's apart from `--out` and `--seed`); K-INIT; K-SHARE (the realised launch share of the along-track weight in 1/3 ± 0.02 per trained arm,
    replacing DAG-ORC's K-REACH) and K-MASK (`train_window_list_sha256` equal to DAG-OACC's, `e144adcd…8168`; `launch_n / launch_eligible` within ±0.02 of f = 0.1619; the chain's own check also asserts `launch_rows` equal to `launch_eligible`, **[I]**, the chain `NOTES.md`); K-BUILD (the launch commit against `aeb54468`:
    byte-identical, or every changed hunk behind a default-off flag whose bit-identity test passes; Amendment 1 changes that clause alone: `v8f_mini_data.py` is pinned to DAG-ORC r2's build, `c117e1cf…`, with
    its twelve flag-gated hunks PASS); K-P0; K-RC (the reader reproduces DAG-ORC's banked tables with 0 differences); K-ZERO (every X − X
    reads exactly 0.0, and an all-False mask reproduces the unweighted run bit for bit); K-REP-LICENCE. The trainer's own sampler, simulated, puts A's realised share at 0.3315 / 0.3307 / 0.3256 /
    0.3281 for trainer seeds 0 / 1 / 10 / 11 (**[E]**, a simulation and not a training run).
  * **The four families** (per seed, per pair, per family, never pooled; **STRATEGIC is not paused for v8F**). `PREREG_V8F_DAG_ORC` §5 applies to each pair at T1-kin: LONGITUDINAL (speed,
    along-track and accel MAE; DK-VIOL / DK-HW / DK-TG on the lead windows; the anti-echo block), LATERAL (heading, yaw-rate, curvature, cross-track, L1 lateral-accel, L2), TACTICAL (TAC-LAT /
    TAC-LON, R-TLAT / R-TLON) and STRATEGIC (S-POS@30 / 50 / 80 against CTRV, within a run). The T0 four families are specified on box 2's T0 eval. The lever acts on longitudinal (along-track)
    pressure, so the LONGITUDINAL family and the distance-keeping rows are read first. What the reader of record computed and did not compute, per family, is in the families bullet of the read below.
  * **Launch and pins** (**[M]**, the banked box-2 chain log `chain_v8f_launchintent.out`, sha256 `ddae7b9e…`, under `raw/2026-10-28-launch-intent-read/box2/`; the queue stage `:launchintentarm` landed at `d8c9f7bf`).
    The chain's first stamped line is 13:54 on 2026-10-28: it finds the DAG-ORC r2 chain's DONE marker and the box released, and runs the box-2 refresh (ZZJOB-END 13:54:55, rc 0; box-2 code = tip `d8c9f7bf` at
    13:55). It then runs the four launch gates in turn (launched 13:57, 13:58, 14:00 and 14:01; ZZJOB-END 13:58:34, 14:00:02, 14:01:33 and 14:02:37). All four print `LAUNCH-INTENT gate: PASS` and write a gate
    token that binds commit `d8c9f7bf`. The first training (A_0) is launched at 14:02. Each run is pulled with its run record PASS and K-SHARE PASS at 14:44 (A_0), 15:25 (A_1), 16:06 (A_rep,0) and 16:48 (A_rep,1), and
    the chain ends in its DONE marker `ZZV8FMINI-DONE` at 16:48, which reads *"B is not queued"*. Box-2 time is 13:54 to 16:48, 174 min or 2.9 h. Pins: the chain log's tip line names the landed flag `b09ac89f` as an
    ancestor of `d8c9f7bf` and the reviewed trainer (sha256 `0d708b3c…`); PREREG sha256 `469bac1f…` and reader `launch_intent_read.py` sha256 `5e950dcf…` are the chain package's pins (`PINS_FILLED.txt`, whose stamp is
    disclosed under Provenance); the reader's 87 tests GREEN and 245 / 245 mutants killed on the promoted text are the register row's (**[I]**), and the chain's self-test passed 390 of 390 tests with 185 / 185
    mutants killed (**[I]**, `HANDOFF.md` of the chain package, and the message of `d8c9f7bf`). **[E]** At launch the PREREG estimated about 2.8 h of box 2 for the four runs (about 42 min each; DAG-OACC's two runs took 41
    and 42 min with their T0 evals), where DAG-ORC took 4.93 h by the logs against the ~3 h asked (§7.26), and about 37 min of dev-box CPU for the four T1-kin rollouts (from DAG-ORC's 37 min for A and REP); the measured
    figures are in the runs bullet of the read below.""" + (" The dev-box T1-kin chain and the reader run are not started in this version." if PREDRAFT else "") + "\n"

SEC_LI_B = "\n" + site("OUTCOME", "* **Outcome.** Not read in this version.") + "\n" + site("CONTROLS", "* **Controls.** Not read in this version.") + "\n" + \
    site("RUNS", "* **The four runs.** Not read in this version.") + "\n" + site("FLOORS", "* **Bars and floors.** Not read in this version.") + "\n" + \
    site("CRAWL", "* **The crawl set (R-MOVE, R-CRAWL).** Not read in this version.") + "\n" + site("GUARDS", "* **Guards and adoption rows.** Not read in this version.") + "\n" + \
    site("FAMILIES", "* **The four families (LONGITUDINAL, LATERAL, TACTICAL, STRATEGIC).** Not read in this version.") + "\n" + \
    site("MLAUNCH", "* **M-LAUNCH and the slow-draw flag.** Not read in this version.") + "\n" + site("NUMBERS", "* **What the numbers say.** Not read in this version.") + """
* **What this read cannot claim.** T1-kin is a self-action open loop (the lead follows its logged future and the cameras are logged), not T2; T0 is teacher-forced and is not driving performance. A one-seed R-MOVE is
  not an effect (`H-ESTIM-SEED-1`): only R-MOVE at both seeds, each in both pairs (four training draws), attributes a crawl change to the lever, and the four draws here disagree on ADE6 (see what the numbers say). It
  attributes to a weighting and not to a mechanism: the lever is one label-defined weight on one term, and the read does not say which of the 12 knots or which launch windows carried a cell. The mask uses future GT
  speed, which is admissible for labels, so a result says nothing about a deployable launch detector. f, the dose, K-SHARE and K-MASK are facts about the TRAIN windows of each run, not eval facts; the eval launch subset is
  a different, smaller set (327 windows / 49 episodes of 2,399 windows, **[M]**, `MLAUNCH.json`). B and HI are not read: nothing here says what a lower or a higher dose does. The T0 four families, the anti-echo block and the
  M2 / L2 launch-subset rows are not computed by the reader of record, and the T-head macro-F1 behind R-TLAT and R-TLON is not in its JSON (families bullet). No bar, row, floor formula, dose or routing changed after any
  LAUNCH-INTENT row was seen.
""" + site("NEXT", "* **Next (Rule Zero).** The committed next follows the outcome table and the NEXT-LAUNCH table above.") + "\n" + \
    site("PROVENANCE", "* **Provenance.** Not read in this version; the PREREG, the reader and the chain are in the repo (`R/raw/2026-10-28-launch-intent-v2/`, `R/raw/2026-10-28-launch-intent-chain/`).") + "\n"

SEC_LI = SEC_LI_A + SEC_LI_B

SEC = SEC_SS + "\n" + SEC_T1 + "\n" + SEC_LI
for name, rows in TAB.items():
    tok = "<<" + name + ">>"
    if tok in SEC:
        assert SEC.count(tok) == 1, tok
        SEC = SEC.replace(tok, "\n".join(rows))
_left_tok = re.findall(r"<<[A-Z0-9_]+>>", SEC)
assert not _left_tok, _left_tok
assert "<<" not in SEC and ">>" not in SEC.replace("→", "")
_unused = [n for n in TAB if n not in TABLES]
assert not _unused

# ---- edits --------------------------------------------------------------------------------------------------------------------
OLD_ST = "research report, v32 of 2026-10-28, results through repository tip `ea5e3d3b` (v31: `569a9d9b`; v30: `93aa1956`; branch"
NEW_ST = f"research report, v33 of 2026-10-28, results through repository tip `{TP}` (v32: `ea5e3d3b`; v31: `569a9d9b`; v30: `93aa1956`; branch"
OLD_ST2 = "v32 adds the SPEEDREAD read (§7.24), the DAgger iteration 2 read (§7.25) and the DAG-ORC read (§7.26, HARMS)."
NEW_ST2 = (OLD_ST2 + " v33 adds the SPEEDSUB read (§7.27, PASS, a screening result), the SPEEDSUB-T1 read (§7.28, NULL) and "
           + site("LINE", "the LAUNCH-INTENT pre-registration (§7.29, running; no LAUNCH-INTENT row is read in this version)."))

OLD_724 = "`PREREG_V8F_SPEEDSUB` registers it (promoted at `94a47f4b`, before any SPEEDSUB number; no SPEEDSUB number exists in this version)."
NEW_724 = ("`PREREG_V8F_SPEEDSUB` registers it (promoted at `94a47f4b`, before any SPEEDSUB number). It was read after this section was written: PASS at T1-class open loop, "
           "through a declared open channel (§7.27), and its T1-kin follow-up reads NULL (§7.28).")

OLD_726 = "* **Next (Rule Zero).**\n  * **A is not adopted, and the committed next is arm B, blocked on the PI's go.**"
NEW_726 = ("* **Next (Rule Zero).**\n"
           "  * **Update (v33): the routing changed after this read.** The PI answered the question whether box 2 should run the redesigned LAUNCH-INTENT next and skip B with *\"Yes, queue it\"*"
           " (2026-10-28), recorded as `PREREG_V8F_DAG_ORC` Amendment 3 (`A3-2026-10-28`, landed `051259d4`; **[I]**, read from the PREREG). B (p = 0.1) is not trained and not read, A stays HARMS,"
           " and NEXT-CRAWL is LAUNCH-INTENT on DAG-OACC's states under its own PREREG (§7.29). The two items below that name arm B as the committed next, and the LAUNCH-INTENT PREREG as a draft,"
           " describe the state at the read. LAUNCH-INTENT has since been read (§7.29, HARMS); the B that is registered after that read is LAUNCH-INTENT's own dose B (W 0.726175), not this section's B (p = 0.1).\n"
           "  * **A is not adopted, and the committed next is arm B, blocked on the PI's go.**")

OLD_NEXT = ("* **What is next.** The open gap is the crawl set (yaw-rate, curvature, L2 and M1 against P10 for DAG-OACC). DAG-ORC (§7.26) moved those rows against DAG-OACC but not past"
            " their bars. Its committed next is arm B (p = 0.1) under the same table, which needs the PI's go, and B would have to exceed A's crawl effect by 3.0–11.4 × at a third of the"
            " dose to clear the same bars; if it does not clear them, the registered branch is NEXT-CRAWL, LAUNCH-INTENT (its PREREG is a draft). The speed line that SPEEDREAD"
            " (`PREREG_V8F_SPEEDREAD`) asked about routes to perception (`PREREG_V8F_LORA` on the A40), which waits on the PI, and nothing on box 2 remains on it.")
NEW_NEXT = (
    "* **SPEEDSUB reads PASS, a screening result (§7.27).** The ego-only MLP read-out `mlp_ego` (SPEEDREAD's fit, no re-fit), put into the five speed cells O reads, cuts O's own-code ADE6 on"
    " M2b-mixo by −0.1290 [−0.2107, −0.0548] m at seed 0 and −0.0686 [−0.1151, −0.0232] m at seed 1, against a bar of −0.06035 m; the shuffled-cells control reads +6.7181 / +6.4923 m"
    " (separated-worse). Tier: T1-class open loop; paired episode-cluster bootstrap, B = 2,000, 141 episodes; two O training runs, one read-out fit, no replicate arm. The PASS was read with the"
    " declared future-speed channel `limit` (`D-SPEEDBAND-EGO-ECHO-1`): on TRAIN fold 0 the read-out's skill survives without `nav` and not without `limit` (mean hold-out RMSE 0.28032 against"
    " 0.24751 SD units for the full block).\n"
    "* **SPEEDSUB-T1 reads NULL, label NULL-ONE (§7.28).** At T1-kin on DAG-OACC, with the channel closed (a causal speed bucket), the refitted read-out `mlp_ego_c` hits the bar at 1 of 4 runs:"
    " ADE6 `egoc − identity` is −0.0233 [−0.0629, +0.0156] m (CTRL s0), −0.0076 [−0.0500, +0.0349] m (CTRL s1), −0.0333 [−0.0832, +0.0131] m (REP s1) and, the one hit, −0.1949 [−0.2619, −0.1313] m"
    " (REP s0, a slow-T training draw whose speed cells read 0.73 to 1.17 m/s slow). All 158 reader control flags PASS and no guard fires. The speed-cell channel does not bind on healthy DAG-OACC"
    " draws, and the logged-lane read-out is flat too (+0.0008 / −0.0185 m at CTRL). Paired episode-cluster bootstrap, B = 2,000, 141 episodes, per run; four training runs.\n"
    + site("ABSTRACT", "* **LAUNCH-INTENT is pre-registered (§7.29).** O's launch-sample along-track weight (GT-launch TRAIN samples hold one third of the along-track weight on knots 1–12, W = 1.58926; a label-side"
    " subset, nothing new at inference) is aimed at the open crawl set of DAG-OACC (yaw-rate, curvature, L2 and M1 against P10). The Master Mind redesigned its bar and arms before any LAUNCH-INTENT"
    " number existed: the draft bar 2 × max(F, RF0, RF) was 1.3 to 9.1 times the whole CTRL − P10 gap on the crawl rows, so no fix could have cleared it. The bar is 2 × max(F, RF0), with two"
    " same-seed one-lever pairs per seed, and a per-seed predicate holds only when it holds in both pairs (T1-kin; paired episode-cluster bootstrap, B = 2,000, 141 episodes). "
    "The four box-2 runs started on 2026-10-28 at 14:02 at commit `d8c9f7bf` and all four launch gates PASS (**[I]**). No LAUNCH-INTENT row is read in this version.\n"
    "* **What is next.** The open gap is the crawl set (yaw-rate, curvature, L2 and M1 against P10 for DAG-OACC). DAG-ORC (§7.26) moved those rows against DAG-OACC but not past their bars, and the PI"
    " routed past its arm B (`PREREG_V8F_DAG_ORC` Amendment 3), so the next lever is LAUNCH-INTENT (§7.29). The history-aware causal read-out stays registered and unrun, because the speed-cell"
    " channel does not limit healthy DAG-OACC draws (§7.28). The speed line that SPEEDREAD (`PREREG_V8F_SPEEDREAD`) asked about routes to perception (`PREREG_V8F_LORA` on the A40), which waits on the"
    " PI; no speed-line item is queued on box 2."))

OLD_811 = "DAG-OACC stays the base; arm B (p = 0.1) needs the PI's go.\n\n### 8.2 What is not established"
NEW_811 = ("DAG-OACC stays the base. (v33: the PI routed past arm B, `PREREG_V8F_DAG_ORC` Amendment 3; LAUNCH-INTENT is item 17.)\n"
           "15. **An ego-only read-out of T's five speed cells, put into O, cuts ADE6 on M2b-mixo through a declared open channel** (§7.27, SPEEDSUB; T1-class open loop, paired episode-cluster bootstrap, B = 2,000,"
           " 141 episodes; PASS, a screening result). ADE6 `ego − own` is −0.1290 [−0.2107, −0.0548] m at seed 0 and −0.0686 [−0.1151, −0.0232] m at seed 1 against a bar of −0.06035 m, and the"
           " shuffled-cells control reads +6.7181 / +6.4923 m (separated-worse). The read-out's skill rests mostly on the logged `limit` lane, which carries the ego's own future max speed"
           " (`D-SPEEDBAND-EGO-ECHO-1`; TRAIN fold 0: 0.28032 without it against 0.24751 with the full block). Two O training runs, one read-out fit, no replicate arm: this is not a lever claim, and not a"
           " statement about DAG-OACC.\n"
           "16. **With the channel closed, the same kind of read-out does not move healthy DAG-OACC draws** (§7.28, SPEEDSUB-T1; T1-kin, four training runs, paired episode-cluster bootstrap, B = 2,000, 141 episodes)."
           " The outcome is NULL, label NULL-ONE: ADE6 `egoc − identity` hits the bar (−0.06035 m, separated) at 1 of 4 runs, the slow-T draw REP s0 (−0.1949 [−0.2619, −0.1313] m); CTRL s0 −0.0233,"
           " CTRL s1 −0.0076 and REP s1 −0.0333 m are not separated, and M1 moves at no run. The logged-lane read-out is flat at CTRL too (+0.0008 / −0.0185 m). The speed-cell channel does not limit"
           " healthy DAG-OACC draws; its measured use is repairing a slow-T draw.\n"
           "17. " + site("R81", "**O's launch-sample along-track weight is pre-registered against the crawl set** (§7.29, LAUNCH-INTENT; T1-kin, paired episode-cluster bootstrap, B = 2,000, 141 episodes)."
                       " Four box-2 runs started on 2026-10-28 (**[I]**); no LAUNCH-INTENT row is read in this version, so nothing is established about the lever.")
           + "\n\n### 8.2 What is not established")

OLD_83 = "### 8.3 The gap to \"dominant driving results\"\n\n**Update (v32).**"
NEW_83 = ("### 8.3 The gap to \"dominant driving results\"\n\n"
          "**Update (v33).** Three items are written up here:\n"
          "* **SPEEDSUB (§7.27) reads PASS, a screening result** (T1-class open loop, on M2b-mixo): an ego-only read-out in O's five speed cells cuts ADE6 by −0.1290 [−0.2107, −0.0548] / −0.0686"
          " [−0.1151, −0.0232] m against −0.06035 m. The PASS rests on the declared future-speed channel `limit` (`D-SPEEDBAND-EGO-ECHO-1`); it is not a driving result, and not a statement about DAG-OACC.\n"
          "* **SPEEDSUB-T1 (§7.28) reads NULL, label NULL-ONE** (T1-kin, four DAG-OACC runs): with the channel closed (a causal bucket) the read-out hits only at the slow-T draw REP s0"
          " (−0.1949 [−0.2619, −0.1313] m). The healthy draws do not move and M1 moves at no run, so the speed-cell channel does not limit healthy DAG-OACC draws. The crawl set (yaw-rate, curvature, L2"
          " and M1 against P10) is still open.\n"
          "* " + site("R83", "**LAUNCH-INTENT (§7.29) is pre-registered** against the crawl set, with a bar the Master Mind redesigned before any data (2 × max(F, RF0) in place of 2 × max(F, RF0, RF)) and two same-seed"
                    " one-lever pairs per seed. The adopted base is unchanged (DAG-OACC) in this version. Its four box-2 runs started on 2026-10-28 at 14:02 (**[I]**); no LAUNCH-INTENT row is read in this version.")
          + "\n\n**Update (v32).**")

OLD_1W = "Next: arm B (p = 0.1) under the same table, which needs the PI's go; then NEXT-CRAWL, LAUNCH-INTENT |"
NEW_1W = ("Next: arm B (p = 0.1) under the same table, which needs the PI's go; then NEXT-CRAWL, LAUNCH-INTENT."
          " Update (v33): the PI routed past B (`PREREG_V8F_DAG_ORC` Amendment 3), and NEXT-CRAWL is LAUNCH-INTENT (row 1ab) |")
OLD_ROWS = "Next: DAG-ORC (row 1w) |\n| 2 | **LoRA on DINOv3**"
NEW_ROWS = ("Next: DAG-ORC (row 1w) |\n"
            "| 1z | **SPEEDSUB**: `mlp_ego` (SPEEDREAD's fit, no re-fit) put into the five speed cells O reads, on M2b-mixo; zero training, dev-box CPU | `PREREG_V8F_SPEEDSUB` (claim row `H-V8F-SPEEDSUB-1`): PASS"
            " = ADE6 `ego − own` separated-better and at or below −F/2 = −0.06035 m at both seeds; HARMS if separated-worse at either seed; REFUSED if a control fails or `egoshuf` is not separated-worse | **PASS,"
            " a screening result** (§7.27; `RESULT_V8F_SPEEDSUB.md`, all 27 control flags PASS, T1-class open loop): −0.1290 [−0.2107, −0.0548] / −0.0686 [−0.1151, −0.0232] m; `egoshuf` +6.7181 / +6.4923 m."
            " Read with the declared open channel `limit` (`D-SPEEDBAND-EGO-ECHO-1`); two O training runs, one read-out fit, no replicate arm. Next: SPEEDSUB-T1 (row 1aa) |\n"
            "| 1aa | **SPEEDSUB-T1**: the causal read-out `mlp_ego_c` (the `mlp_ego` design refitted on the causal speed bucket) in O's five speed cells at every replan tick, DAG-OACC at four runs (CTRL s0 / s1,"
            " REP s0 / s1); zero training, dev-box CPU | `PREREG_V8F_SPEEDSUB_T1` (Amendment 1, `A1-2026-10-28`): a hit is ADE6 `egoc − identity` separated-better and at or below −0.06035 m; PASS = a hit at all"
            " four runs and no guard (G-ADE, G-WORSE, G-NEW, G-VIOL); PARTIAL = hits at both CTRL seeds; NULL = anything else | **NULL, label NULL-ONE** (§7.28; `RESULT_V8F_SPEEDSUB_T1.md`, all 158 control"
            " flags PASS, no guard fires, T1-kin): one hit of four, REP s0 −0.1949 [−0.2619, −0.1313] m (a slow-T draw; VIOL-ANY +0.0421 separated-worse there); CTRL s0 −0.0233, CTRL s1 −0.0076, REP s1 −0.0333 m,"
            " not separated; M1 moves at no run. Floor rule `H-V8F-SPEEDSUB-T1-2`: NOT COLLAPSE (ADE6 gap 0.4025 → 0.2309 m). Next: section 8 (a), the history-aware read-out, ranked down and not run; the crawl lever"
            " is LAUNCH-INTENT (row 1ab) |\n"
            "| 1ab | **LAUNCH-INTENT**: O's launch-sample along-track weight on DAG-OACC's recipe (`--launch-intent-w 1.58926`: GT-launch TRAIN samples hold one third of the along-track weight on knots 1–12), two"
            " same-seed one-lever pairs per seed (A_s − DAG-OACC_s and A_rep,s − REP_s); four box-2 runs | `PREREG_V8F_LAUNCH_INTENT` (claim row `H-V8F-LAUNCH-INTENT-1`): bar 2 × max(F, RF0) (crawl rows: yaw-rate"
            " 1.1070, curvature 0.0526, L2 0.0136, M1 0.0096), RF beside; each per-seed predicate holds only when it holds in both pairs; the DAG-ORC outcome table over the per-seed booleans (524,288 cells) | "
            + site("R84", "**PRE-REGISTERED, RUNNING** (§7.29; promoted `77234276` as PRE-REGISTERED, NOT RUN, before any LAUNCH-INTENT number; four box-2 runs started 2026-10-28 14:02 at `d8c9f7bf`, **[I]**)."
                          " No LAUNCH-INTENT row is read in this version.") + " |\n"
            "| 2 | **LoRA on DINOv3**")

ANCHOR_8 = "\n\n---\n\n## 8. Discussion"
assert s.count(ANCHOR_8) == 1
E = [(OLD_ST, NEW_ST), (OLD_ST2, NEW_ST2), (OLD_724, NEW_724), (OLD_726, NEW_726), (OLD_NEXT, NEW_NEXT),
     (ANCHOR_8, "\n\n" + SEC.rstrip("\n") + ANCHOR_8),
     (OLD_811, NEW_811), (OLD_83, NEW_83), (OLD_1W, NEW_1W), (OLD_ROWS, NEW_ROWS)]
for a, b in E:
    assert s.count(a) == 1, (s.count(a), a[:80])
    assert h.count(a) == 1, ("html", h.count(a), a[:80])
    s = s.replace(a, b)
    h = h.replace(a, b)

# ---- guards on the inserted text ----------------------------------------------------------------------------------------------
ins = "\n".join(b for _, b in E)
low = ins.lower()
assert "closed loop" not in low and "closed-loop" not in low and "drives" not in low
for w in ("void", "superseded", "retracted", "formerly", "no longer", "correction", "withdrawn"):
    assert re.search(r"\b" + w + r"\b", low) is None, w
assert re.search(r"retract|supersed|withdrawn|was wrong|not the floor|previously read|outdated|i said", low) is None
assert "$" not in SEC and "\r" not in ins
nocode = re.sub(r"`[^`\n]*`", "", SEC)
assert re.search(r"<[A-Za-z/!]", nocode) is None, re.search(r"<[A-Za-z/!]", nocode)
for nm in ("LINE", "TITLE729", "R84"):
    if not PREDRAFT:
        assert "\n" not in FILL[nm].strip() and (nm != "R84" or "|" not in FILL[nm]), ("single-line slot with a line break or a pipe", nm)
assert sorted(USED) == sorted(SLOTS), (sorted(USED), sorted(SLOTS))
_markers = re.findall(r"@@LI_RESULT_([A-Z0-9]+)@@", s)
if PREDRAFT:
    assert sorted(_markers) == sorted(n for n in SLOTS if n != "TITLE729"), (sorted(_markers))
    assert all(h.count("@@LI_RESULT_" + n + "@@") == 1 for n in SLOTS if n != "TITLE729")
else:
    assert "@@" not in s and "@@LI_RESULT" not in h and not _markers, "a slot marker is left in a final version"
    assert "RUNNING" not in SEC_LI_A.split("\n")[0], "the final heading still says RUNNING"
    for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"):
        assert fam in FILL["FAMILIES"], ("the FAMILIES text names no " + fam)
    for stale in ("is not read in this version", "are not read in this version", "Not read in this version", "pre-draft", "not started in this version", "No LAUNCH-INTENT checkpoint",
                  "no LAUNCH-INTENT row is read", "PRE-REGISTERED, RUNNING"):
        assert stale not in ins, ("a pre-draft sentence is left in the final text", stale)
    assert FILL["ABSTRACT"].count("\n* ") + (1 if FILL["ABSTRACT"].lstrip().startswith("* ") else 0) >= 2, "the ABSTRACT text must hold the LAUNCH-INTENT bullet and the 'What is next' bullet"
    assert "What is next" in FILL["ABSTRACT"], "the ABSTRACT text must carry the 'What is next' bullet"
    # the RESULT's tables are COPIED line for line: every table line of a fill text is a line of the RESULT
    _res_lines = set(RES.split("\n"))
    for _k, _v in FILL.items():
        for _l in _v.split("\n"):
            if _l.startswith("|"):
                assert _l in _res_lines, ("a table line of the fill text is not a line of RESULT_V8F_LAUNCH_INTENT.md", _k, _l[:90])
    # every italic quotation of 25 characters or more must be verbatim in a banked source (the RESULT, the two JSON files, the chain log, the PREREG or the R2C note)
    _corpus = re.sub(r"\s+", " ", "\n".join([RES, json.dumps(LIR_J, ensure_ascii=False), json.dumps(MLJ_J, ensure_ascii=False), CHAIN, PLI, R2C]))
    for _k, _v in FILL.items():
        for _q in re.findall(r'\*"([^"\n]{25,}?)"\*', re.sub(r"\s*\n\s*", " ", _v)):
            assert re.sub(r"\s+", " ", _q) in _corpus, ("a quotation in the fill text is not verbatim in a banked source", _k, _q[:90])
    # the launch facts are the chain log's: no note outside the repository is quoted
    for _k, _v in FILL.items():
        assert "MM_STATE" not in _v and "state note" not in _v, ("a launch fact is attributed to a note outside the repository", _k)
for stale in ("no SPEEDSUB number exists in this version",):
    assert stale not in s and stale not in h, stale
assert s.count("### 7.27 ") == 1 and s.count("### 7.28 ") == 1 and s.count("### 7.29 ") == 1
assert s.index("### 7.26 ") < s.index("### 7.27 ") < s.index("### 7.28 ") < s.index("### 7.29 ") < s.index("## 8. Discussion")
for sec, extra in ((SEC_SS, "STRATEGIC is reported, not paused, for v8F"), (SEC_T1, "STRATEGIC is reported, not paused, for v8F"), (SEC_LI_A, "STRATEGIC is not paused for v8F")):
    assert extra in sec
    for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"):
        assert fam in sec or fam in "\n".join(TAB["SS_EGO"] + TAB["T1_ROWS"]), fam
for sec, words in ((SEC_SS, ("T1-class open loop", "B = 2,000", "141 episodes")), (SEC_T1, ("T1-kin", "B = 2,000", "141 episodes")), (SEC_LI_A, ("T1-kin", "B = 2,000", "141 episodes"))):
    flat = re.sub(r"\s+", " ", sec)
    assert all(w in flat for w in words), words
assert "\r" not in s and "</script" not in s.lower()
assert s.count("**Status:** research report, v33 of 2026-10-28") == 1
# the html: the same edits on the v32 html must equal the template filled with the new md
assert tpl.replace("%%MD%%", s) == h, "html edited in place differs from template + md"

os.makedirs(OUT, exist_ok=True)
open(OUT + "V8F_SCIENTIFIC_REPORT.md", "w", encoding="utf-8", newline="\n").write(s)
open(OUT + "v8f_report.html", "w", encoding="utf-8", newline="\n").write(h)
if PREDRAFT:
    open(OUT + "LI_FILL_TEMPLATE.json", "w", encoding="utf-8", newline="\n").write(json.dumps(OrderedDict((k, "") for k in SLOTS), indent=1) + "\n")
    open(OUT + "LI_FILL_HELP.txt", "w", encoding="utf-8", newline="\n").write(
        "Slots of the v33 pre-draft. Copy LI_FILL_TEMPLATE.json to FILL.json, write one non-empty text per slot, then run:\n"
        "  python patch_v33.py <tip> <out_dir> --fill FILL.json\n"
        "A slot text REPLACES the pre-draft text at its site together with the marker. Slots OUTCOME to PROVENANCE and ABSTRACT hold whole bullets: start each with '* ' (OUTCOME to PROVENANCE: '* **Name.** ...');\n"
        "a table inside a bullet is a block at column 0 with a blank line before and after, and the text after it starts with two spaces. STATUS729, R81 and R83 follow a fixed lead (see below).\n"
        "TITLE729 has no marker in the pre-draft: it replaces 'PRE-REGISTERED, RUNNING' in the 7.29 heading.\n"
        "LINE, TITLE729 and R84 are one line (R84 has no pipe character). Banned words, tier words and the closed-loop wording rule apply to the texts. Every number needs its evidence class,\n"
        "tier (T0 / T1-kin) and estimator; name per family what is NOT RUN / NOT BUILT / NOT COMPUTABLE with the reason.\n\n"
        + "\n".join("%-11s %s" % (k, v) for k, v in SLOTS.items()) + "\n")
print("report v33 built (%s)" % ("PRE-DRAFT, %d slots" % (len(SLOTS) - 1) if PREDRAFT else "FINAL"), len(s), len(h), "tip", TIP)
