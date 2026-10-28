"""Numbers audit for report v33 (mechanical; mirrors the v32 audit).
python audit_v33.py [tip [out_dir [audit_path]]]  ->  writes v33_NUMBERS_AUDIT.md next to this script (or to audit_path).
out_dir is the folder holding V8F_SCIENTIFIC_REPORT.md (default: out/ beside this script, else final/, else the folder above it).

Method: the v33 md is diffed against the v32 md (the blob at <tip>); every added or changed line is tokenised for numbers (sign kept, the Unicode minus read as '-',
thousands commas removed, single digits 0-9 unsigned excluded as enumerations). Each distinct signed token is looked up line by line in the source files below, in its
PLAIN form and in its comma-formatted form (a number such as 306566 is also searched as 306,566, and 306,566 as 306566, because the sources print both). A = found with the
same sign in a source line; B = found only with a different or absent sign; C = found only in the v32 report; D = found nowhere. Every B, C and D token must carry a hand
review in audit_reviewed_v33.py, or this script fails (a new unexplained token cannot pass).

The pre-draft carries marked slots (the string at-at LI_RESULT_<NAME> at-at). The audit treats the slot markers as text without numbers, and reports in section I
which slots are still open: a final version has none.

Every source is a blob at the tip, read with git show. Nothing outside the repository is read: the launch facts of 7.29 (13:54, the gates, the pulls, the DONE marker at 16:48,
box-2 time) are found in the banked box-2 chain log raw/2026-10-28-launch-intent-read/box2/chain_v8f_launchintent.out, and the REP provenance in R2C_NOTE.md
(raw/2026-10-28-dag-orc-result/notes/) and the REP launch-gate tokens. The audit can be re-run on any machine that has the mirror. Section H recomputes the decisive and the
derived LAUNCH-INTENT numbers from LAUNCH_INTENT_READ.json and MLAUNCH.json; section J re-finds the RESULT's log-derived disclosures in the banked T1-kin chain logs."""
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import OrderedDict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TIP = sys.argv[1] if len(sys.argv) > 1 else "8c1e9636388dbb105f6b1853bd95dfdccea9eb93"
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"


def _out_dir():
    if len(sys.argv) > 2:
        return sys.argv[2]
    for d in (HERE + "out", HERE + "final"):
        if os.path.exists(d + "/V8F_SCIENTIFIC_REPORT.md"):
            return d
    up = os.path.dirname(HERE.rstrip("/"))
    return up if os.path.exists(up + "/V8F_SCIENTIFIC_REPORT.md") else HERE + "out"


OUT = os.path.abspath(_out_dir()).replace("\\", "/").rstrip("/") + "/"
AUDIT_PATH = sys.argv[3] if len(sys.argv) > 3 else HERE + "v33_NUMBERS_AUDIT.md"
MIRROR = "C:/Users/Admin/tanitad-push"
R = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
RP = R + "report/2026-10-26/"


def git(args):
    r = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*"] + args, capture_output=True)
    assert r.returncode == 0, (args, r.stderr[:200])
    return r.stdout


def blob(path):
    return git(["show", f"{TIP}:{path}"]).decode("utf-8").replace("\r\n", "\n")


def blob_at(rev, path):
    return git(["show", f"{rev}:{path}"]).decode("utf-8").replace("\r\n", "\n")


REGROWS = "\n".join(l for l in blob("Project Steering/GOALS_AND_CLAIMS.md").split("\n")
                    if re.match(r"\| \*\*(D-SPEEDBAND-EGO-ECHO-1|H-V8F-SPEEDSUB-1|H-V8F-SPEEDSUB-T1-1|H-V8F-SPEEDSUB-T1-2|H-V8F-LAUNCH-INTENT-1|H-V8F-DAG-ORC-1)\*\*", l))
assert REGROWS.count("\n") >= 4, "the register rows were not found"
COMMITS = ["94a47f4b", "842ec2ac", "af2dee4a", "4cec09e7", "37d78ed1", "266b470e", "051259d4", "b09ac89f", "77234276", "d8c9f7bf", "ea5e3d3b", "8c24823f", "e28516e8", "d7dc9f11", "8c1e9636"]
SOURCES = OrderedDict([
    ("RESULT_V8F_SPEEDSUB.md", blob(R + "RESULT_V8F_SPEEDSUB.md")),
    ("RESULT_V8F_SPEEDSUB_T1.md", blob(R + "RESULT_V8F_SPEEDSUB_T1.md")),
    ("PREREG_V8F_SPEEDSUB.md", blob("Project Steering/PREREG_V8F_SPEEDSUB.md")),
    ("PREREG_V8F_SPEEDSUB_T1.md", blob("Project Steering/PREREG_V8F_SPEEDSUB_T1.md")),
    ("PREREG_V8F_LAUNCH_INTENT.md", blob("Project Steering/PREREG_V8F_LAUNCH_INTENT.md")),
    ("PREREG_V8F_DAG_ORC.md", blob("Project Steering/PREREG_V8F_DAG_ORC.md")),
    ("RESULT_V8F_DAG_ORC.md", blob(R + "RESULT_V8F_DAG_ORC.md")),
    ("GOALS_AND_CLAIMS.md (register rows)", REGROWS),
    ("SUBREAD.json", blob(R + "raw/2026-10-28-speedsub-result/read/SUBREAD.json")),
    ("SPEEDSUB_T1_READ.json", blob(R + "raw/2026-10-28-speedsub-t1-read/SPEEDSUB_T1_READ.json")),
    ("SPEEDSUB_T1_BESIDE.json", blob(R + "raw/2026-10-28-speedsub-t1-read/SPEEDSUB_T1_BESIDE.json")),
    ("REP_S0_SPEED_BIAS.md", blob(R + "raw/2026-10-28-speedsub-t1-prereg/rep_s0_diag/REP_S0_SPEED_BIAS.md")),
    ("FINDING_1_ATTRIBUTION.md", blob(R + "raw/2026-10-28-speedsub-t1-prereg/speedsub_t1_prep/FINDING_1_ATTRIBUTION.md")),
    ("launch-intent-flag/RESULTS.md", blob(R + "raw/2026-10-28-launch-intent-flag/RESULTS.md")),
    ("launch-intent-flag/MM_REDESIGN_2026-10-28.md", blob(R + "raw/2026-10-28-launch-intent-flag/MM_REDESIGN_2026-10-28.md")),
    ("launch-intent-flag/f_measure.json", blob(R + "raw/2026-10-28-launch-intent-flag/f_measure.json")),
    ("launch-intent-v2/NOTES.md", blob(R + "raw/2026-10-28-launch-intent-v2/NOTES.md")),
    ("launch-intent-v2/enumerate_cells_li.out", blob(R + "raw/2026-10-28-launch-intent-v2/enumerate_cells_li.out")),
    ("launch-intent-chain/HANDOFF.md", blob(R + "raw/2026-10-28-launch-intent-chain/HANDOFF.md")),
    ("launch-intent-chain/NOTES.md", blob(R + "raw/2026-10-28-launch-intent-chain/NOTES.md")),
    ("launch-intent-chain/PINS_FILLED.txt", blob(R + "raw/2026-10-28-launch-intent-chain/PINS_FILLED.txt")),
    ("launch-intent-chain/LAUNCH.txt", blob(R + "raw/2026-10-28-launch-intent-chain/LAUNCH.txt")),
    ("DAGORC_READ.json", blob(R + "raw/2026-10-28-dag-orc-result/read/DAGORC_READ.json")),
    ("RF_REP.json", blob(R + "raw/2026-10-28-dag-orc-result/read/RF_REP.json")),
    ("P0_RF0.json (rebase)", blob(R + "raw/2026-10-28-dag-orc-build/rebase/read/P0_RF0.json")),
    ("git log (cited commits)", git(["log", "--no-walk=sorted", "--format=%h %ad %s%n%b", "--date=iso"] + COMMITS).decode("utf-8")),
    ("RESULT_V8F_LAUNCH_INTENT.md", blob(R + "RESULT_V8F_LAUNCH_INTENT.md")),
    ("LAUNCH_INTENT_READ.json", blob(R + "raw/2026-10-28-launch-intent-read/read_A1/LAUNCH_INTENT_READ.json")),
    ("MLAUNCH.json", blob(R + "raw/2026-10-28-launch-intent-mlaunch/MLAUNCH.json")),
    ("MLAUNCH NOTES.md", blob(R + "raw/2026-10-28-launch-intent-mlaunch/NOTES.md")),
    ("chain_v8f_launchintent.out", blob(R + "raw/2026-10-28-launch-intent-read/box2/chain_v8f_launchintent.out")),
    ("R2C_NOTE.md", blob(R + "raw/2026-10-28-dag-orc-result/notes/R2C_NOTE.md")),
    ("lit1_logs/chain.out", blob(R + "raw/2026-10-28-launch-intent-read/lit1_logs/chain.out")),
    ("lit1_logs/chain.out.prev_1028151251", blob(R + "raw/2026-10-28-launch-intent-read/lit1_logs/chain.out.prev_1028151251")),
    ("lit1_logs/launch_intent_read_REFUSED.out", blob(R + "raw/2026-10-28-launch-intent-read/lit1_logs/launch_intent_read_REFUSED.out")),
    ("read_A1/read_A1.out", blob(R + "raw/2026-10-28-launch-intent-read/read_A1/read_A1.out")),
    ("check_result_filled.out", blob(R + "raw/2026-10-28-launch-intent-result/check_result_filled.out")),
    ("REP launch-gate tokens (s0, s1)", blob(R + "raw/2026-10-28-dag-orc-result/runs/M2b_s0_dagoaccorcrep/launch_gate_token.json") + "\n"
     + blob(R + "raw/2026-10-28-dag-orc-result/runs/M2b_s1_dagoaccorcrep/launch_gate_token.json")),
    ("REP run records (s0, s1)", blob(R + "raw/2026-10-28-dag-orc-result/runs/M2b_s0_dagoaccorcrep/summary.json") + "\n" + blob(R + "raw/2026-10-28-dag-orc-result/runs/M2b_s1_dagoaccorcrep/summary.json")),
])
PLI_PROMOTED = blob_at("77234276", "Project Steering/PREREG_V8F_LAUNCH_INTENT.md")
READER_AT_PROMOTION = git(["show", f"77234276:{R}raw/2026-10-28-launch-intent-v2/read/launch_intent_read.py"])
V32 = blob(RP + "V8F_SCIENTIFIC_REPORT.md")
V33 = open(OUT + "V8F_SCIENTIFIC_REPORT.md", encoding="utf-8", newline="").read()
assert "\r" not in V33

NUM = re.compile(r"(?<![\w.])[-+\u2212]?\d[\d,]*(?:\.\d+)?(?:[eE][-+\u2212]?\d+)?(?![\w])")


def norm(tok):
    return tok.replace("\u2212", "-").replace(",", "")


def toks(line):
    out = []
    for m in NUM.finditer(line):
        t = norm(m.group(0))
        if re.fullmatch(r"\d", t):
            continue
        out.append(t)
    return out


def unsigned(t):
    return t.lstrip("+-")


a, b = V32.split("\n"), V33.split("\n")
sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
added, removed = [], []
for op, i1, i2, j1, j2 in sm.get_opcodes():
    if op in ("replace", "insert"):
        added += [(j + 1, b[j]) for j in range(j1, j2)]
    if op in ("replace", "delete"):
        removed += [(i + 1, a[i]) for i in range(i1, i2)]

tok_lines = OrderedDict()
for ln, text in added:
    for t in toks(text):
        tok_lines.setdefault(t, []).append(ln)

# A source line holds a token in its plain form (306566) or in its comma form (306,566): the tokenizer removes commas, so both forms give the same key; the raw
# search below confirms the exact printed form that was found, and reports it.
src_idx = OrderedDict()
src_unsigned = {}
src_forms = {}  # token -> set of printed forms found in the sources
for name, text in SOURCES.items():
    for k, line in enumerate(text.split("\n"), 1):
        for m in NUM.finditer(line):
            t = norm(m.group(0))
            if re.fullmatch(r"\d", t):
                continue
            src_idx.setdefault(t, OrderedDict()).setdefault(name, []).append(k)
            src_unsigned.setdefault(unsigned(t), OrderedDict()).setdefault(name, []).append(k)
            src_forms.setdefault(t, set()).add(m.group(0).replace("\u2212", "-"))
v32_unsigned = {}
rem_set = {x for x in removed}
for k, line in enumerate(a, 1):
    if (k, line) in rem_set:
        continue
    for t in toks(line):
        v32_unsigned.setdefault(unsigned(t), []).append(k)


def span(d, cap=6):
    parts = []
    for name, lines in d.items():
        short = lines[:cap]
        parts.append(name + ":" + ",".join(str(x) for x in short) + ("..." if len(lines) > cap else ""))
    return "; ".join(parts)


A, B, C, D = [], [], [], []
for t, lns in tok_lines.items():
    if t in src_idx:
        A.append((t, lns, src_idx[t]))
    elif unsigned(t) in src_unsigned:
        B.append((t, lns, src_unsigned[unsigned(t)]))
    elif unsigned(t) in v32_unsigned:
        C.append((t, lns, v32_unsigned[unsigned(t)]))
    else:
        D.append((t, lns))

# ---- hand review: every B, C and D token is explained in audit_reviewed_v33.py; the script fails on an unexplained one ----------------
REVIEWED = {}
SCOPE_OK = {}
exec(open(HERE + "audit_reviewed_v33.py", encoding="utf-8").read()) if os.path.exists(HERE + "audit_reviewed_v33.py") else None
need = [t for t, *_ in B + C + D]
missing = [t for t in need if t not in REVIEWED]
if missing:
    print("UNREVIEWED TOKENS (%d):" % len(missing))
    for kind, lst in (("B", B), ("C", C), ("D", D)):
        for row in lst:
            if row[0] in missing:
                print(kind, row[0], "v33 lines", row[1][:5], "|", (span(row[2], 3) if kind == "B" else row[2][:5] if kind == "C" else ""))
    sys.exit(2)
stale = [t for t in REVIEWED if t not in need]
assert not stale, ("reviewed tokens that are not in B/C/D", stale)

# ---- section-scoped check: a number of 7.27 / 7.28 / 7.29 must be found in THAT section's own sources -----------------------------
SECTION_SOURCES = {
    "7.27": ["RESULT_V8F_SPEEDSUB.md", "PREREG_V8F_SPEEDSUB.md", "SUBREAD.json", "FINDING_1_ATTRIBUTION.md", "RESULT_V8F_SPEEDSUB_T1.md", "PREREG_V8F_SPEEDSUB_T1.md",
             "GOALS_AND_CLAIMS.md (register rows)", "git log (cited commits)"],
    "7.28": ["RESULT_V8F_SPEEDSUB_T1.md", "PREREG_V8F_SPEEDSUB_T1.md", "SPEEDSUB_T1_READ.json", "SPEEDSUB_T1_BESIDE.json", "REP_S0_SPEED_BIAS.md", "FINDING_1_ATTRIBUTION.md",
             "RESULT_V8F_SPEEDSUB.md", "GOALS_AND_CLAIMS.md (register rows)", "git log (cited commits)"],
    "7.29": ["PREREG_V8F_LAUNCH_INTENT.md", "PREREG_V8F_DAG_ORC.md", "RESULT_V8F_DAG_ORC.md", "GOALS_AND_CLAIMS.md (register rows)", "launch-intent-flag/RESULTS.md",
             "launch-intent-flag/MM_REDESIGN_2026-10-28.md", "launch-intent-flag/f_measure.json", "launch-intent-v2/NOTES.md", "launch-intent-v2/enumerate_cells_li.out",
             "launch-intent-chain/HANDOFF.md", "launch-intent-chain/NOTES.md", "launch-intent-chain/PINS_FILLED.txt", "launch-intent-chain/LAUNCH.txt", "DAGORC_READ.json", "RF_REP.json",
             "P0_RF0.json (rebase)", "git log (cited commits)", "RESULT_V8F_SPEEDSUB_T1.md", "RESULT_V8F_LAUNCH_INTENT.md", "LAUNCH_INTENT_READ.json", "MLAUNCH.json", "MLAUNCH NOTES.md",
             "chain_v8f_launchintent.out", "R2C_NOTE.md", "lit1_logs/chain.out", "lit1_logs/chain.out.prev_1028151251", "lit1_logs/launch_intent_read_REFUSED.out", "read_A1/read_A1.out",
             "check_result_filled.out", "REP launch-gate tokens (s0, s1)", "REP run records (s0, s1)"],
}
_i27, _i28, _i29 = V33.index("### 7.27 "), V33.index("### 7.28 "), V33.index("### 7.29 ")
_iend = V33.index("\n---\n\n## 8. Discussion")
SECTION_TEXT = {"7.27": V33[_i27:_i28], "7.28": V33[_i28:_i29], "7.29": V33[_i29:_iend]}
scope_rows, scope_missing = [], []
for sec, text in SECTION_TEXT.items():
    own = {}
    for name in SECTION_SOURCES[sec]:
        for t in {tt for line in SOURCES[name].split("\n") for tt in toks(line)}:
            own.setdefault(t, []).append(name)
    seen = set()
    for line in text.split("\n"):
        if line.startswith("|"):
            continue  # table lines are verbatim copies of source lines, asserted by the patch and re-tested in section I
        for t in toks(line):
            if t in seen:
                continue
            seen.add(t)
            if t not in own:
                scope_rows.append((sec, t))
                if (sec + ":" + t) not in SCOPE_OK:
                    scope_missing.append(sec + ":" + t)
if scope_missing:
    print("TOKENS NOT FOUND IN THEIR OWN SECTION'S SOURCES, UNREVIEWED (%d):" % len(scope_missing), scope_missing)
    sys.exit(3)
stale_scope = [k for k in SCOPE_OK if tuple(k.split(":", 1)) not in set(scope_rows)]
assert not stale_scope, stale_scope

# ---- claims taken from a PREREG, a note or a log (wording and constants), resolved to lines ------------------------------------------------
CLAIMS = [
    ("7.27: the question, the selection disclosure, F/2 = 0.06035, the arm `ego`, the shuffle control, tier T1-class open loop", "PREREG_V8F_SPEEDSUB.md",
     ["**F/2 = 0.06035 exactly.**", "`mlp_ego` is below `T_own` in every cell at both seeds.", "Claim row `H-V8F-SPEEDSUB-1`", "**Status: PRE-REGISTERED, NOT RUN (2026-10-28).**"]),
    ("7.27: the PASS, the 27 control flags, the reader line, the declared channel and its attribution, the rule-violation block, the follow-up", "RESULT_V8F_SPEEDSUB.md",
     ["all **27 control flags PASS**", "`ZZSPEEDSUB-DONE PASS`", "bin plus `v0` recovers it at R² 0.9702", "Dropping `nav` costs 0.0156 SD", "dropping `limit` costs 0.0328 SD",
      "**which input carries the ADE6 cut is not measured here**", "**there is no replicate arm in this experiment**", "It is still the third read-out examined on these 141 episodes",
      "**Not a statement about DAG-OACC / DAG-ORC.**", "On the 81 stop windows `ego` is *worse* than `own` at both seeds"]),
    ("7.27: the attribution table (TRAIN fold 0: full 0.24751, no nav 0.26313, no limit 0.28032, neither 0.30849)", "FINDING_1_ATTRIBUTION.md", ["0.24751", "0.26313", "0.28032", "0.30849"]),
    ("7.28: design, the hit rule, the guards (G-WORSE, G-NEW, G-ADE, G-VIOL), the outcome order, the causal bucket, the prior", "PREREG_V8F_SPEEDSUB_T1.md",
     ["Claim row `H-V8F-SPEEDSUB-T1-1` opens on promotion.", "## Amendment 1 (`A1-2026-10-28`; appended before the smoke and before any rollout of this experiment)",
      "**G-VIOL: `VIOL-ANY` is not separated-worse at ≥ 3 of the 4 runs.**", "`speed_max_bin(max measured ego speed over rows [j−20, j])`",
      "The labels NULL-ONE (hits at 1–3 runs, not both CTRL seeds) and NULL-FLAT (no hit) describe and decide", "PASS is not the expected outcome", "| **NULL** | anything else | next: section 8 (a), the history-aware read-out |"]),
    ("7.28: the NULL, the 158 control flags, the reader line, the floor rule NOT COLLAPSE, the ranking of the next steps, the single-copy dumps", "RESULT_V8F_SPEEDSUB_T1.md",
     ["**all 158 control flags PASS**", "`ZZSPEEDSUB-T1-READ-DONE NULL NULL-ONE`", "**s0 verdict: NOT COLLAPSE**", "falls from 0.4025 to 0.2309 m (a drop of 42.6 %)", "G-VIOL needs 3 of the 4 runs and fires at 1",
      "**(a) stays registered and is not run now, because arithmetic predicts NULL.**", "**The largest measured crawl lever is DAG-ORC's perturbation**", "**M1 crawl share moves at no run.**",
      "**The paired `egoc − identity` S-POS row is NOT COMPUTED**", "| LONGITUDINAL | anti-echo block | **NOT BUILT** |", "| TACTICAL | goal-head F1 at T1-kin | **NOT COMPUTABLE here** |"]),
    ("7.28: the REP s0 diagnosis (all five speed cells slow against CTRL s0; REP s1 and A sit with CTRL; the Master Mind's, written before the read)", "REP_S0_SPEED_BIAS.md",
     ["v_end_ms", "v_lo_ms", "v_hi_ms"]),
    ("7.26 update: the PI's routing, B not trained, A stays HARMS, NEXT-CRAWL = LAUNCH-INTENT", "PREREG_V8F_DAG_ORC.md",
     ["## Amendment 3 (A3-2026-10-28): the PI's routing decision after the read.", "*\"Yes, queue it\"*", "B (p = 0.1) is therefore **not trained and not read**.", "A's outcome stays **HARMS**, read under the table as written."]),
    ("7.29: status, promotion, the claim row, the dose (f = 0.161851, W_A 1.58926, W_B 0.726175, W_HI 4.17852), the lever, the mask, the redesign disclosure", "PREREG_V8F_LAUNCH_INTENT.md",
     ["Claim row `H-V8F-LAUNCH-INTENT-1`, registered at promotion.", "**Status: PRE-REGISTERED, NOT RUN (2026-10-28).**", "**GT-launch windows** | **49,618 → f = 0.161851**",
      "**Arm A, s = 1/3: 1 + W = (1 − f) / (2 f) ⇒ W_A = 1.58926.**", "**Arm B, s = 1/4: 1 + W = (1 − f) / (3 f) ⇒ W_B = 0.726175**",
      "The bar and arms were changed by the Master Mind before promotion, after DAG-ORC's floors and the REP s0 diagnosis were known, and before any LAUNCH-INTENT number existed.",
      "HYPOTHESIS: the 2-D ADE averages 30 knots over every window", "Knots 1–12 are τ = 0.2–2.4 s: the executed horizon at m = 10"]),
    ("7.29: K-MASK as registered (window-list sha256 equal to DAG-OACC's; launch_n / launch_eligible within 0.02 of f), the controls K-GATE, K-PAIR, K-INIT, K-SHARE, K-BUILD, K-P0, K-RC, K-ZERO, K-REP-LICENCE", "PREREG_V8F_LAUNCH_INTENT.md",
     ["| (new) K-MASK | per trained arm: `train_window_list_sha256` equals DAG-OACC's (`e144adcd…8168`)", "`launch_n / launch_eligible` lies within ±0.02 of f = 0.1619", "| K-GATE |", "| K-PAIR |", "| K-INIT |",
      "| K-BUILD |", "| K-P0 |", "| K-RC |", "| K-ZERO |", "| K-REP-LICENCE |", "ESTIMATED by simulating the trainer's own `ChunkedSampler` (3,000 steps, batch 64, `tools/measure_f.py`)"]),
    ("7.29: the chain's own K-MASK check also asserts launch_rows == launch_eligible (INHERITED from the chain notes)", "launch-intent-chain/NOTES.md", ["`launch_rows == launch_eligible`"]),
    ("7.29: the PREREG's reading choices confirmed at promotion: domain D of the replicate pair, M-LAUNCH moved only in both pairs at both seeds, no exempt rows", "PREREG_V8F_LAUNCH_INTENT.md",
     ["**CONFIRMED** by the Master Mind at promotion (2026-10-28): readings (a)–(e) as written.", "**The domain D** of a pair is every Read-A row not separated-worse for Y − P10 at s, with no exempt rows",
      "⚠ **Stricter in one way:** every guard must hold in both pairs of the seed.", "The seed's predicate holds only when it holds in both pairs (§10)."]),
    ("7.29: the launch commit, the pins, the reader's not-built items (M-LAUNCH, anti-echo, M2, T0 families), the chain, the REP licence", "launch-intent-v2/NOTES.md",
     ["M-LAUNCH", "anti-echo", "T0"]),
    ("7.29: the chain package (launch commit b09ac89f, trainer 0d708b3c, the self-test totals) and its pins", "launch-intent-chain/PINS_FILLED.txt", ["469bac1f", "5e950dcf", "b09ac89f", "0d708b3c"]),
    ("7.29: the launch facts, MEASURED from the banked box-2 chain log: first stamped line 13:54, the predecessor's DONE marker, the refresh, the four gates (launched, ZZJOB-END, PASS), A_0 launched 14:02, the four pulls with run record PASS and K-SHARE PASS, the DONE marker at 16:48 and 'B is not queued'",
     "chain_v8f_launchintent.out",
     ["10-28 13:54 waiting for the DAG-ORC r2 chain", "10-28 13:54 predecessor verdict: GO the DAG-ORC r2 chain ended DONE", "===== ZZJOB-END 28.10.2026 13:54:55,82 rc=0", "10-28 13:55 box 2 code = tip d8c9f7bfc91798b6fafb5a13dd793771e2f79f6a",
      "===== ZZJOB-END 28.10.2026 13:58:34,48 rc=0", "===== ZZJOB-END 28.10.2026 14:00:02,89 rc=0", "===== ZZJOB-END 28.10.2026 14:01:33,52 rc=0", "===== ZZJOB-END 28.10.2026 14:02:37,90 rc=0",
      "10-28 14:02 launched pid=23736", "10-28 14:44 M2b_s0_launchintent pulled to", "10-28 15:25 M2b_s1_launchintent pulled to", "10-28 16:06 M2b_s0_launchintentrep pulled to",
      "10-28 16:48 M2b_s1_launchintentrep pulled to", "ZZV8FMINI-DONE 10-28 16:48 tip d8c9f7bfc91798b6fafb5a13dd793771e2f79f6a", "B is not queued", "descends from the landed flag b09ac89f", "carries the reviewed trainer (sha256 0d708b3c"]),
    ("7.29: the REP provenance: the Master Mind's incident note (repo path), REP s0 and the other pre-05:56 runs, the pinned relaunch at 8c24823f", "R2C_NOTE.md",
     ["**Unaffected.** A s0 and REP s0 trained and evaluated before 05:56.", "branch `agent/dagorc-r2-pin`, created at `8c24823f`", "A s1 and REP s1"]),
    ("7.29: the REP launch-gate tokens bind commit 8c24823f at both seeds", "REP launch-gate tokens (s0, s1)", ['"run": "M2b_s0_dagoaccorcrep"', '"run": "M2b_s1_dagoaccorcrep"', '"commit": "8c24823f2b1bd24332148ebe434c870bcf3662f8"']),
    ("7.29: the REP run records: trainer seeds 10 and 11", "REP run records (s0, s1)", ['"seed": 10,', '"seed": 11,']),
    ("7.29: Amendment 1 (appended at e28516e8, sha256 6ae79f2f): K-BUILD's data-module clause, pinned to DAG-ORC r2's build", "PREREG_V8F_LAUNCH_INTENT.md",
     ["## Amendment 1 (A1-2026-10-28): K-BUILD's data-module clause, before any LAUNCH-INTENT row is read", "**Written 2026-10-28 ~15:10 by the Master Mind.**", "twelve hunks, one adding the module constant `EGO_PERTURB_COMPOSED_STREAM`",
      "No bar, row, outcome, predicate or other control changes.", "The launch-gate tokens bind the PREREG as promoted (sha256 `469bac1f…`) and the reader as landed (`5e950dcf…`)."]),
    ("7.29: the RESULT: word, controls, tier, estimator, run record, disclosures (Amendment 1, RAM-guard stop, pin slip, reader label, single copies, coupling), next, what it cannot claim", "RESULT_V8F_LAUNCH_INTENT.md",
     ["all **206 reader controls PASS**, wall 341.6 s", "**HARMS.** Arm A is not adopted.", "Nothing below softens that.", "2 of the 16 crawl cells are separated-better and 1 clears its bar", "All 16 X − P10 crawl cells are separated-worse.",
      "At each seed the separated ADE6 cells flip sign between the main pair and the replicate pair.", "**M-LAUNCH: NOT MOVED**", "Section 11 label: **not moved**", "Route: **LAUNCH-INTENT-HI**",
      "Both routes need a PI go: B, registered after HARMS (section 10), and HI, the branch M-LAUNCH selects (section 11). The PI is asked which, if either, box 2 runs.",
      "**Amendment 1 (`A1-2026-10-28`), before the read.**", "**The T1-kin chain was stopped once, on its own RAM guard.**", "**Pin slip (dated note).**",
      "They were filled ~13:50, before commit `d8c9f7bf` (13:51:52) and after the PREREG was promoted at `77234276` (13:49:00).", "**A label inside the reader.**", "**Single copies.**", "**Disclosed coupling.**",
      "It does **not** recompute the cells from the per-window dumps or re-run the bootstrap", "B and HI are not read."]),
    ("7.29: the RESULT's own checker: 133 checks, 0 failed", "check_result_filled.out", ["SUMMARY 133 checks passed, 0 failed", "ALL CHECKS PASS"]),
    ("7.29: M-LAUNCH build notes: the readings, the controls, the disclosed redesign of the controls, the post-hoc per-tick look, the beside M2 block", "MLAUNCH NOTES.md",
     ["all 9 control groups pass", "`M-LAUNCH = not moved`", "The first run of `mlaunch.py`", "Post-hoc look, NOT registered, not a finding", "Eight per-tick looks, no multiplicity control.",
      "Mixed in sign and cell: no M2 row is separated in the same direction in both pairs at the same seed."]),
    ("7.29: the T1-kin chain logs: the stopped launch (RAM guard), the relaunch, the REFUSED read of the chain", "lit1_logs/chain.out.prev_1028151251", ["15:07:07 LI_s0_m10_causal_heads: waiting, a guard is LOW: LOW RAM available 3040 MB < 4096 MB"]),
    ("7.29: the PI's go as recorded in the DAG-ORC PREREG; the redesign note of the Master Mind (bar 2 x max(F, RF0), four same-seed pairs)", "launch-intent-flag/MM_REDESIGN_2026-10-28.md", ["RF0", "bar"]),
    ("Register rows: H-V8F-SPEEDSUB-1, H-V8F-SPEEDSUB-T1-1 and H-V8F-LAUNCH-INTENT-1 exist and carry the words the report uses", "GOALS_AND_CLAIMS.md (register rows)",
     ["**H-V8F-SPEEDSUB-1**", "**H-V8F-SPEEDSUB-T1-1**", "**H-V8F-LAUNCH-INTENT-1**", "**D-SPEEDBAND-EGO-ECHO-1**"]),
    ("Commits named in the text exist and carry the subject word (94a47f4b SPEEDSUB, 842ec2ac PREREG_V8F_SPEEDSUB_T1, af2dee4a RESULT_V8F_SPEEDSUB, 4cec09e7 Amendment 1, 37d78ed1 RESULT_V8F_SPEEDSUB_T1, 051259d4 Amendment 3, b09ac89f the flag, 77234276 the LAUNCH-INTENT PREREG, d8c9f7bf the queue stage)", "git log (cited commits)",
     ["94a47f4", "842ec2a", "af2dee4", "4cec09e", "37d78ed", "266b470", "051259d", "b09ac89", "7723427", "d8c9f7b"]),
]
claim_rows = []
for what, src, phrases in CLAIMS:
    text = SOURCES[src].split("\n")
    refs = []
    for ph in phrases:
        hit = [k for k, line in enumerate(text, 1) if ph in line]
        assert hit, (src, ph)
        refs.append(f"{hit[0]}")
    claim_rows.append(f"| {what} | `{src}`:{','.join(refs)} |")

# ---- verbatim table lines ------------------------------------------------------------------------------------------------------
T_LINES = {"7.27": [l for l in SECTION_TEXT["7.27"].split("\n") if l.startswith("|")], "7.28": [l for l in SECTION_TEXT["7.28"].split("\n") if l.startswith("|")],
           "7.29": [l for l in SECTION_TEXT["7.29"].split("\n") if l.startswith("|")]}
res_lines = {"7.27": set(SOURCES["RESULT_V8F_SPEEDSUB.md"].split("\n")), "7.28": set(SOURCES["RESULT_V8F_SPEEDSUB_T1.md"].split("\n")),
             "7.29": set(SOURCES["PREREG_V8F_LAUNCH_INTENT.md"].split("\n")) | set(SOURCES["RESULT_V8F_LAUNCH_INTENT.md"].split("\n"))}
t_copy = {k: [l for l in v if l in res_lines[k]] for k, v in T_LINES.items()}
t_auth = {k: [l for l in v if l not in res_lines[k]] for k, v in T_LINES.items()}
assert len(T_LINES["7.27"]) == len(t_copy["7.27"]) and len(T_LINES["7.28"]) == len(t_copy["7.28"]), "a 7.27 / 7.28 table line is not a line of the RESULT"
# in 7.29 the committed tables are copied from the PREREG (gap, bars, arms, predicates, outcomes, next) and the read's tables from the RESULT: every line must be a line of one of the two,
# but the separator rows of two tables may coincide with each other, which is fine.
assert len(T_LINES["7.29"]) == len(t_copy["7.29"]), [l[:80] for l in t_auth["7.29"]]

# ---- JSON cross-check: the decisive and the derived numbers, recomputed from the banked JSON and found in the text -----------------------------------
from decimal import Decimal, ROUND_HALF_UP

SUB = json.loads(SOURCES["SUBREAD.json"])
T1J = json.loads(SOURCES["SPEEDSUB_T1_READ.json"])
BES = json.loads(SOURCES["SPEEDSUB_T1_BESIDE.json"])
DOJ = json.loads(SOURCES["DAGORC_READ.json"])
RFJ = json.loads(SOURCES["RF_REP.json"])
P0J = json.loads(SOURCES["P0_RF0.json (rebase)"])
FMJ = json.loads(SOURCES["launch-intent-flag/f_measure.json"])


def sg(x, nd=4):
    return ("+" if x >= 0 else "−") + ("%." + str(nd) + "f") % abs(x)


def cis(c):
    return "%s [%s, %s]" % (sg(c["delta"]), sg(c["lo"]), sg(c["hi"]))


def rhu(x, nd):
    return str(Decimal(repr(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def region(name):
    if name == "abstract":
        return V33[V33.index("## 1. Abstract"):V33.index("## 2. Introduction")]
    if name in ("7.27", "7.28", "7.29"):
        return SECTION_TEXT[name]
    if name == "7.26":
        return V33[V33.index("### 7.26 "):_i27]
    if name == "7.24":
        return V33[V33.index("### 7.24 "):V33.index("### 7.25 ")]
    if name == "8.1":
        return V33[V33.index("### 8.1 "):V33.index("### 8.2 ")]
    if name == "8.3":
        return V33[V33.index("### 8.3 "):V33.index("### 8.4 ")]
    if name.startswith("row "):
        return "\n".join(l for l in V33.split("\n") if l.startswith("| " + name[4:] + " |"))
    if name == "status":
        return V33.split("\n")[2]
    raise KeyError(name)


JSON_ROWS = []


def jchk(what, path, value, snippet, where):
    ok = [w for w in where if re.sub(r"\s+", " ", snippet) in re.sub(r"\s+", " ", region(w))]  # a sentence may wrap across lines of the md
    assert len(ok) == len(where), ("JSON cross-check failed", what, path, value, snippet, [w for w in where if w not in ok])
    JSON_ROWS.append((what, path, value, snippet, ", ".join(ok)))


# ---- SPEEDSUB (SUBREAD.json) ----
assert SUB["outcome"] == "PASS" and SUB["F_pass_m"] == 0.06035 and SUB["selected_readout"] == "mlp_ego" and SUB["readout_sha256"].startswith("a205609a")
ctl = SUB["controls"]
flat = []
for k, v in ctl.items():
    if isinstance(v, dict):
        flat += list(v.values()) if all(isinstance(x, bool) for x in v.values()) else [x for vv in v.values() for x in (vv.values() if isinstance(vv, dict) else [vv])]
    else:
        flat.append(v)
assert len(flat) == 27 and all(x is True for x in flat), (len(flat), flat)
jchk("outcome PASS, F/2 = 0.06035, mlp_ego, readout sha a205609a, 27 control flags all true", "outcome; F_pass_m; selected_readout; readout_sha256; controls (counted)", "PASS; 0.06035; mlp_ego; a205609a…; 27 true",
     "all **27 control flags PASS**", ["7.27"])
jchk("27 control flags in the 8.4 row", "controls (counted)", "27", "all 27 control flags PASS", ["row 1z"])
for sd, key in (("s0", "0"), ("s1", "1")):
    e = SUB["rows"][key]["ego"]
    assert e["hi"] < 0 and e["delta"] <= -SUB["F_pass_m"]
    jchk("ADE6 ego - own, %s: delta and 95 %% CI" % sd, 'rows["%s"]["ego"].{delta, lo, hi}' % key, cis(e), cis(e), ["7.27", "abstract", "8.1", "8.3", "row 1z"])
    sh = SUB["rows"][key]["egoshuf"]
    assert sh["lo"] > 0
    jchk("ADE6 egoshuf - own (deliberate-regression arm), %s: separated-worse" % sd, 'rows["%s"]["egoshuf"].delta' % key, sg(sh["delta"]), sg(sh["delta"]), ["7.27", "abstract", "8.1", "row 1z"])
    g = SUB["rows"][key]["gt"]
    jchk("ADE6 gt - own (T0 ceiling), %s" % sd, 'rows["%s"]["gt"].delta' % key, sg(g["delta"]), sg(g["delta"]), ["7.27"])
    cap = SUB["rows"][key]["capture"]
    assert abs(cap - SUB["rows"][key]["ego"]["delta"] / SUB["rows"][key]["gt"]["delta"]) < 5e-4
    jchk("CAPTURE = ego / gt, %s (4 dp)" % sd, 'rows["%s"].capture; ego.delta / gt.delta' % key, "%.4f" % cap, "%.4f" % cap, ["7.27"])
    assert SUB["rows"][key]["regression_arm_separated_worse"] is True
jchk("CAPTURE as a share of the GT cells' ceiling: 18 % and 10 %", "rows[<seed>].capture", "%d %% and %d %%" % (round(100 * SUB["rows"]["0"]["capture"]), round(100 * SUB["rows"]["1"]["capture"])), "18 % and 10 %", ["7.27"])
d1 = -SUB["rows"]["1"]["ego"]["delta"] - SUB["F_pass_m"]
jchk("s1 clears the bar by 0.0083 m (0.0686 - 0.06035 = 0.00825, half up)", 'rows["1"]["ego"].delta; F_pass_m', rhu(round(d1, 6), 4), "clears the bar by 0.0083 m only", ["7.27"])
assert rhu(round(d1, 6), 4) == "0.0083"
d2 = SUB["rows"]["0"]["ego"]["delta"] - SUB["rows"]["1"]["ego"]["delta"]
jchk("the two seeds' effects differ by 0.0604 m", 'rows[<seed>]["ego"].delta', "%.4f" % abs(d2), "differ by 0.0604 m", ["7.27"])
for sd, key in (("s0", "0"), ("s1", "1")):
    assert SUB["rows"][key]["ego"]["hi"] > -SUB["F_pass_m"]  # the interval's upper end lies above the bar (the PASS text says so)
jchk("the upper ends of the two intervals (-0.0548, -0.0232) lie above -0.06035", 'rows[<seed>]["ego"].hi; F_pass_m', "-0.0548, -0.0232 > -0.06035", "(−0.0548, −0.0232) lies above −0.06035", ["7.27"])

# ---- SPEEDSUB-T1 (SPEEDSUB_T1_READ.json, SPEEDSUB_T1_BESIDE.json) ----
assert T1J["OUTCOME"] == "NULL" and T1J["label"] == "NULL-ONE" and T1J["flags"]["n_hit"] == 1 and T1J["flags"]["hits"] == {"CTRL_s0": False, "CTRL_s1": False, "REP_s0": True, "REP_s1": False}
assert T1J["flags"]["controls_ok"] is True and T1J["flags"]["guards_fired"] == [] and len(T1J["controls"]) == 158 and all(T1J["controls"].values()) and T1J["bar"] == 0.06035
for g_, v in T1J["guards"].items():
    assert v["fires"] is False
assert T1J["guards"]["G-VIOL"]["runs"] == ["REP_s0"] and T1J["guards"]["G-VIOL"]["active"] is True
jchk("NULL, NULL-ONE, 1 hit of 4 (REP s0), 158 controls true, no guard fires, bar 0.06035", "OUTCOME; label; flags.hits; controls (158); guards.*.fires; bar", "NULL; NULL-ONE; REP_s0 only; 158; none; 0.06035",
     "**all 158 control flags PASS**", ["7.28"])
jchk("158 control flags in the 8.4 row", "controls (counted)", "158", "all 158 control flags PASS", ["row 1aa"])
jchk("NULL-ONE in the abstract, 8.1, 8.3, the 8.4 row", "label", "NULL-ONE", "NULL-ONE", ["abstract", "8.1", "8.3", "row 1aa"])
for run, nm in (("CTRL_s0", "CTRL s0"), ("CTRL_s1", "CTRL s1"), ("REP_s0", "REP s0"), ("REP_s1", "REP s1")):
    c = T1J["d_ADE6_egoc_minus_identity"][run]
    assert c["separated"] == (c["hi"] < 0) and T1J["flags"]["hits"][run] == (c["separated"] and c["delta"] <= -T1J["bar"])
    jchk("ADE6 egoc - identity, %s: delta and 95 %% CI" % nm, "d_ADE6_egoc_minus_identity.%s.{delta, lo, hi}" % run, cis(c), cis(c), ["7.28", "abstract"])
for run in ("CTRL_s0", "CTRL_s1", "REP_s1"):
    c = T1J["d_ADE6_egoc_minus_identity"][run]
    jchk("ADE6 egoc - identity, %s, as a point alone (8.1 item 16, 8.4 row 1aa)" % run, "d_ADE6_egoc_minus_identity.%s.delta" % run, sg(c["delta"]), sg(c["delta"]), ["8.1", "row 1aa"])
c = T1J["d_ADE6_egoc_minus_identity"]["REP_s0"]
jchk("the one hit as a multiple of the bar: 3.23 x", "d_ADE6_egoc_minus_identity.REP_s0.delta / bar", "%.2f" % (-c["delta"] / T1J["bar"]), "3.23 times the bar", ["7.28"])
fr = T1J["floor_rule_H_V8F_SPEEDSUB_T1_2"]["s0 (decisive, beside)"]
assert fr["verdict"] == "NOT COLLAPSE" and fr["g_id"]["ADE6"] == 0.4025 and fr["g_c"]["ADE6"] == 0.2309 and not any(fr["per_row"].values())
jchk("floor rule at s0: NOT COLLAPSE; ADE6 gap 0.4025 -> 0.2309 m; drop 42.6 %", "floor_rule_H_V8F_SPEEDSUB_T1_2[s0].{verdict, g_id, g_c}", "NOT COLLAPSE; 0.4025; 0.2309; %.1f %%" % (100 * (0.4025 - 0.2309) / 0.4025),
     "falls from 0.4025 to 0.2309 m (a drop of 42.6 %)", ["7.28"])
assert "%.1f" % (100 * (0.4025 - 0.2309) / 0.4025) == "42.6"
jchk("floor rule in the 8.4 row", "floor_rule_H_V8F_SPEEDSUB_T1_2", "0.4025 -> 0.2309", "ADE6 gap 0.4025 → 0.2309 m", ["row 1aa"])
TB = T1J["tables"]
rows18 = [r for r, v in TB["egoc-identity"]["CTRL_s0"].items() if v.get("status") is None]
assert len(rows18) == 18, len(rows18)


# direction: error rows are lower-better, TAC rows higher-better
def better_worse(run):
    b = w = 0
    for r in rows18:
        v = TB["egoc-identity"][run][r]
        if not v["separated"]:
            continue
        good = (v["delta"] > 0) if r.startswith("TAC") else (v["delta"] < 0)
        b += good
        w += (not good)
    return b, w


bw = {run: better_worse(run) for run in ("CTRL_s0", "CTRL_s1", "REP_s0", "REP_s1")}
assert bw["REP_s0"] == (8, 2) and bw["CTRL_s0"] == (0, 0) and bw["REP_s1"] == (0, 0) and bw["CTRL_s1"] == (0, 3), bw
jchk("REP s0: 8 of 18 rows separated-better, 2 separated-worse; CTRL s0 and REP s1: none; CTRL s1: 3 separated-worse", "tables[egoc-identity].<run>.<row>.{delta, separated}", str(bw),
     "Of the 18 computed rows, 8 are separated-better and 2", ["7.28"])
jchk("CTRL s0 and REP s1 move no row of 18; CTRL s1 moves the wrong way on three rows", "tables[egoc-identity].<run>.<row>.separated", "0 / 0 / 3 worse", "CTRL s0 and REP s1 move no row of 18", ["7.28"])
for r_, run_ in (("VIOL-ANY", "REP_s0"), ("M1 crawl share", "REP_s0"), ("accel MAE", "CTRL_s1"), ("yaw-rate MAE", "CTRL_s1"), ("L1 lateral-accel MAE", "CTRL_s1")):
    cc = TB["egoc-identity"][run_][r_]
    jchk("%s, %s, egoc - identity" % (r_, run_), "tables[egoc-identity].%s[%s].{delta, lo, hi}" % (run_, r_), cis(cc), cis(cc), ["7.28"])
for run_ in ("CTRL_s0", "CTRL_s1", "REP_s0", "REP_s1"):
    assert TB["egoc-identity"][run_]["M1 crawl share"]["separated"] is False
jchk("M1 crawl share is not separated at any of the four runs", "tables[egoc-identity].<run>[M1 crawl share].separated", "false x4", "M1 crawl share moves at no run", ["7.28"])
m1a, m1b = TB["egoc-identity"]["CTRL_s0"]["M1 crawl share"]["delta"], TB["egoc-identity"]["CTRL_s1"]["M1 crawl share"]["delta"]
jchk("M1 at CTRL s0 and s1: -0.0007 and +0.0021", "tables[egoc-identity].CTRL_s*[M1 crawl share].delta", "%s and %s" % (sg(m1a), sg(m1b)), "%s and %s" % (sg(m1a), sg(m1b)), ["7.28"])
for run_, nm_ in (("CTRL_s0", "CTRL s0"), ("CTRL_s1", "CTRL s1"), ("REP_s0", "REP s0"), ("REP_s1", "REP s1")):
    cc = TB["egoshufc-identity"][run_]["ADE6"]
    assert cc["separated"] and cc["delta"] > 0
    jchk("ADE6 egoshufc - identity (deliberate-regression arm), %s, separated-worse" % nm_, "tables[egoshufc-identity].%s[ADE6].{delta, lo, hi}" % run_, cis(cc), cis(cc), ["7.28"])
b0, b1 = BES["egol_minus_identity_l"]["CTRL_s0"]["ADE6"], BES["egol_minus_identity_l"]["CTRL_s1"]["ADE6"]
assert not b0["separated"] and not b1["separated"]
jchk("the logged-lane bridge, ADE6 at CTRL s0 and s1: +0.0008 and -0.0185, not separated", "SPEEDSUB_T1_BESIDE.egol_minus_identity_l.CTRL_s*.ADE6.{delta, separated}", "%s and %s" % (sg(b0["delta"]), sg(b1["delta"])),
     "+0.0008 and −0.0185 m", ["7.28"])
jchk("the bridge in the abstract and in 8.1 item 16", "SPEEDSUB_T1_BESIDE.egol_minus_identity_l.CTRL_s*.ADE6.delta", "+0.0008 / -0.0185", "+0.0008 / −0.0185 m", ["abstract", "8.1"])
# share of the CTRL gap to P10 that egoc closes on the crawl rows, at the two CTRL seeds
CR = ["yaw-rate MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share"]
shares = []
for run_ in ("CTRL_s0", "CTRL_s1"):
    for r in CR:
        i_p = TB["identity-P10"][run_][r]["delta"]
        e_p = TB["egoc-P10"][run_][r]["delta"]
        shares.append(100 * (i_p - e_p) / i_p)
assert "%.1f" % max(shares) == "12.8" and sum(1 for x in shares if x < 0) == 3, (max(shares), shares)
jchk("share of the identity - P10 crawl gap that egoc closes at CTRL s0 / s1: at most 12.8 %; widens it on 3 of 8 cells", "tables[egoc-P10], tables[identity-P10] ([crawl rows], CTRL_s0 / s1).delta",
     "max %.1f %%; negative on %d of 8" % (max(shares), sum(1 for x in shares if x < 0)), "closes at most 12.8 % of the gap on any crawl row and widens it on 3 of 8 cells", ["7.28"])

# ---- LAUNCH-INTENT (PREREG tables recomputed from the banked DAG-ORC JSON; the dose from f_measure.json) ----
RA, FL = DOJ["read_a"], DOJ["floors"]
CRN = {"yaw-rate MAE": "yaw-rate MAE", "curvature MAE": "curvature MAE", "L2 curvature MAE (GT>=5.2)": "L2 curvature MAE (GT ≥ 5.2)", "M1 crawl share": "M1 crawl share"}
ratios = []
for r, shown in CRN.items():
    f_, rf0, rf = FL["F"][r], P0J["rows"][r]["RF0"], RFJ["RF"][r]
    v1, v2 = 2 * max(f_, rf0, rf), 2 * max(f_, rf0)
    g0, g1 = RA["CTRL_P10"]["s0"][r]["delta"], RA["CTRL_P10"]["s1"][r]["delta"]
    row = [l for l in SECTION_TEXT["7.29"].split("\n") if l.startswith("| " + shown + " | ") and " / " in l][0]  # the gap table: its cells hold 's0 / s1' pairs
    cells = [c.strip() for c in row.strip("|").split("|")]
    exp = [shown, "%.4f" % v1, "%.4f / %.4f" % (g0, g1), "%.1f / %.1f" % (v1 / g0, v1 / g1), "**%.4f**" % v2, "%.2f / %.2f" % (v2 / g0, v2 / g1)]
    assert cells == exp, (cells, exp)
    ratios += [v1 / g0, v1 / g1]
    assert v2 < g0 and v2 < g1
assert "%.1f" % min(ratios) == "1.3" and "%.1f" % max(ratios) == "9.1"
JSON_ROWS.append(("the gap table of 7.29: v1 bar, CTRL - P10 at s0 / s1, v1 bar / gap, v2 bar, v2 bar / gap, for the four crawl rows, recomputed cell by cell (an assertion fails the script on a mismatch)",
                  "floors.F, P0_RF0.rows[r].RF0, RF_REP.RF[r], read_a.CTRL_P10.<seed>[r].delta", "min %.1f, max %.1f; every v2 bar below both gaps" % (min(ratios), max(ratios)), "| **1.1070** | 0.32 / 0.13 |", "7.29"))
assert "| **1.1070** | 0.32 / 0.13 |" in region("7.29")
jchk("v1 bar / gap ranges from 1.3 to 9.1 (min and max over the 8 crawl cells)", "(the same recomputation)", "%.1f to %.1f" % (min(ratios), max(ratios)), "1.3 to 9.1 times", ["7.29"])
for r in ["ADE6", "speed MAE", "along-track MAE", "accel MAE", "DK-VIOL", "DK-HW", "DK-TG", "heading MAE", "yaw-rate MAE", "curvature MAE", "cross-track MAE", "L1 lateral-accel MAE", "L2 curvature MAE (GT>=5.2)",
          "TAC-LAT", "TAC-LON", "S-POS@30", "S-POS@50", "S-POS@80", "M1 crawl share"]:
    f_, rf0, rf = FL["F"][r], P0J["rows"][r]["RF0"], RFJ["RF"][r]
    shown = "L2 curvature MAE (GT>=5.2)" if r.startswith("L2") else r
    row = [l for l in SECTION_TEXT["7.29"].split("\n") if l.startswith("| " + shown + " | ") and " / " not in l and l.count("|") == 7][0]  # the bar table
    cells = [c.strip() for c in row.strip("|").split("|")]
    exp = [shown, "%.4f" % f_, "%.4f" % rf0, "**%.4f**" % (2 * max(f_, rf0)), "%.4f" % rf, "%.4f" % (2 * max(f_, rf0, rf))]
    assert cells == exp, (cells, exp)
JSON_ROWS.append(("the 19-row bar table of 7.29: F, RF0, v2 bar = 2 x max(F, RF0), RF, v1 bar = 2 x max(F, RF0, RF), recomputed row by row", "floors.F, P0_RF0.rows[r].RF0, RF_REP.RF[r]", "19 rows equal", "| ADE6 | 0.1207 | 0.0138 | **0.2414** | 0.4025 | 0.8050 |", "7.29"))
assert "| ADE6 | 0.1207 | 0.0138 | **0.2414** | 0.4025 | 0.8050 |" in region("7.29")
RC = RA["REP_CTRL"]
ROWS19 = list(RA["A_CTRL"]["s0"].keys())
s0set = sum(1 for r in ROWS19 if abs(RC["s0"][r]["delta"]) >= abs(RC["s1"][r]["delta"]))
assert s0set == 17
assert "REP s0 (a slow-T training draw, §7.28) set RF on 17 of" in region("7.29")
JSON_ROWS.append(("REP s0 sets RF (the larger |REP - CTRL| of the two seeds) on 17 of 19 rows", "read_a.REP_CTRL.<seed>[<row>].delta", "%d of 19" % s0set, "set RF on 17 of", "7.29"))
f_ = FMJ["f"]
assert FMJ["n_launch"] == 49618 and FMJ["n_windows_all"] == 306566 and "%.6f" % f_ == "0.161851" and abs(f_ - 49618 / 306566) < 1e-12 and FMJ["mask_disagreements_torch_vs_numpy"] == 0
wa, wb, whi = (1 - f_) / (2 * f_) - 1, (1 - f_) / (3 * f_) - 1, (1 - f_) / f_ - 1
assert "%.5f" % wa == "1.58926" and "%.6f" % wb == "0.726175" and "%.5f" % whi == "4.17852"
jchk("f = 49,618 / 306,566 = 0.161851 (the GT-launch share of the TRAIN windows)", "f_measure.json.{n_launch, n_windows_all, f}", "%d / %d = %.6f" % (FMJ["n_launch"], FMJ["n_windows_all"], f_), "f = 49,618 / 306,566 = 0.161851", ["7.29"])
jchk("W_A = (1 - f) / (2 f) - 1 = 1.58926 (launch samples hold 1/3 of the along-track weight)", "f_measure.json.f", "%.5f" % wa, "W_A = 1.58926", ["7.29"])
jchk("W_B = (1 - f) / (3 f) - 1 = 0.726175 (1/4)", "f_measure.json.f", "%.6f" % wb, "W_B = 0.726175", ["7.29"])
jchk("W_HI = (1 - f) / f - 1 = 4.17852 (1/2)", "f_measure.json.f", "%.5f" % whi, "W_HI = 4.17852", ["7.29"])
jchk("W = 1.58926 in the abstract and in the 8.4 row 1ab", "f_measure.json.f", "1.58926", "1.58926", ["abstract", "row 1ab"])
cnt = [262144, 261888, 1, 15, 3, 45, 128, 64]
assert sum(cnt) == 524288 == 2 ** 19
jchk("the outcome table covers 524,288 = 2^19 cells: 262,144 + 261,888 + 1 + 15 + 3 + 45 + 128 + 64", "(arithmetic on the PREREG's counts)", str(sum(cnt)), "524,288 cells", ["7.29"])

# ======================================================================================================================================================================
# LAUNCH-INTENT, the read: LAUNCH_INTENT_READ.json, MLAUNCH.json, the banked box-2 chain log and the T1-kin chain logs. Every table line of 7.29 that carries a number of the read is rebuilt from the JSON
# (or from the log) and must equal the printed line; a mismatch fails the script.
# ======================================================================================================================================================================
from collections import Counter

LIR = json.loads(SOURCES["LAUNCH_INTENT_READ.json"])
MLJ = json.loads(SOURCES["MLAUNCH.json"])
CHL = SOURCES["chain_v8f_launchintent.out"]
T1C = SOURCES["lit1_logs/chain.out"]
T1P = SOURCES["lit1_logs/chain.out.prev_1028151251"]
RESULT_SRC = SOURCES["RESULT_V8F_LAUNCH_INTENT.md"]
S29 = SECTION_TEXT["7.29"]
T29 = set(l for l in S29.split("\n") if l.startswith("|"))
MN, EN = "\u2212", "\u2013"
raL, flL = LIR["read_a"], LIR["floors"]
ROWS19L = list(raL["A_CTRL"]["s0"].keys())
BARL = flL["bar"]
PL = {"main": "main (A " + MN + " CTRL)", "rep": "replicate (A_rep " + MN + " REP)"}
PAIRKEY = {"main": ("A_CTRL", "A_P10", "CTRL_P10"), "rep": ("Arep_REP", "Arep_P10", "REP_P10")}
LOG_ROWS = []
LI_TABLE_LINES = 0


def flag(c):
    return "B" if c["separated_better"] else ("W" if c["separated_worse"] else EN)


def fmtn(c):
    return "%+.4f [%+.4f, %+.4f]" % (c["delta"], c["lo"], c["hi"])


def fmtc(c):
    txt = fmtn(c)
    if c["separated_better"]:
        return "**%s B**" % txt
    if c["separated_worse"]:
        return "**%s W**" % txt
    return "%s %s" % (txt, EN)


def needl(line, what):
    global LI_TABLE_LINES
    assert line in T29, ("a table line of 7.29 differs from the line rebuilt from the JSON / log", what, line[:240])
    LI_TABLE_LINES += 1


def addrow(what, path, value, snippet, where):
    assert re.sub(r"\s+", " ", snippet) in re.sub(r"\s+", " ", region(where)), ("addrow snippet not found", what, snippet)
    JSON_ROWS.append((what, path, value, snippet, where))


def lchk(what, source, value, snippet, where):
    ok = [w for w in where if re.sub(r"\s+", " ", snippet) in re.sub(r"\s+", " ", region(w))]
    assert len(ok) == len(where), ("log cross-check failed", what, source, value, snippet, [w for w in where if w not in ok])
    LOG_ROWS.append((what, source, value, snippet, ", ".join(ok)))


def impv(c):
    return -c["delta"] if c["direction"] == "lower" else c["delta"]


# ---- the word, the controls, the flags ----
oc = LIR["OUTCOME"]
assert oc["OUTCOME"] == "HARMS" and oc["adopts"] is False and LIR["cascade"]["final"] == "HARMS" and oc["arm"] == "A" and oc["matching_rows_in_table_order"] == ["HARMS"]
assert len(LIR["controls_PASS"]) == 206 and all(v is True for v in LIR["controls_PASS"].values())
GR = Counter(k.split(":")[0] for k in LIR["controls_PASS"])
assert sum(GR.values()) == 206
jchk("HARMS, arm A not adopted, 206 controls all true", "OUTCOME.OUTCOME; OUTCOME.adopts; controls_PASS (counted)", "HARMS; False; 206 true", "all **206 reader controls PASS**", ["7.29"])
jchk("the control groups and their counts (sum 206)", "controls_PASS keys, grouped by the text before ':'", "%d groups, sum %d" % (len(GR), sum(GR.values())), ", ".join("%s %d" % (g, n) for g, n in GR.items()), ["7.29"])
jchk("the JSON condition verbatim", "OUTCOME.condition", oc["condition"][:60] + "...", oc["condition"], ["7.29"])
jchk("committed next verbatim", "OUTCOME.committed_next", oc["committed_next"][:60] + "...", oc["committed_next"], ["7.29"])
jchk("b_status verbatim", "cascade.b_status", LIR["cascade"]["b_status"][:60] + "...", LIR["cascade"]["b_status"], ["7.29"])
pv = LIR["provenance"]
assert pv["wall_s"] == 341.6 and pv["reader_sha256"].startswith("fb3d2d7d") and pv["rf0_sha256"].startswith("1608d70b") and pv["manifest_sha256"].startswith("98d2cd3d")
jchk("reader fb3d2d7d, RF0 1608d70b, manifest 98d2cd3d, wall 341.6 s", "provenance.{reader_sha256, rf0_sha256, manifest_sha256, wall_s}", "fb3d2d7d; 1608d70b; 98d2cd3d; 341.6", "Wall 341.6 s", ["7.29"])
assert hashlib.sha256(SOURCES["LAUNCH_INTENT_READ.json"].encode("utf-8")).hexdigest().startswith("e451e1a6")
assert hashlib.sha256(SOURCES["MLAUNCH.json"].encode("utf-8")).hexdigest().startswith("f5ebb4b2")

# ---- per-pair guards and the per-seed predicate table ----
PR = {sd: LIR["rows"]["A"][sd] for sd in ("s0", "s1")}
for sd in ("s0", "s1"):
    for pair in ("main", "rep"):
        blk = PR[sd]["pairs"][pair]
        assert blk["guards"]["R-TLAT"] and blk["guards"]["R-TLON"] and blk["guards"]["R-T0"] and blk["guards"]["R-NEW"] and blk["flags"]["R-KEEP"] and blk["flags"]["R-NOBASE"]
        assert blk["flags"]["R-MOVE"] is False and blk["flags"]["R-CRAWL"] is False
assert [(sd, pair) for sd in ("s0", "s1") for pair in ("main", "rep") if PR[sd]["pairs"][pair]["guards"]["R-ADE"] is False] == [("s0", "rep"), ("s1", "main")]
for sd in ("s0", "s1"):
    assert PR[sd]["guards"]["R-ADE"] is False and PR[sd]["flags"]["R-MOVE"] is False and PR[sd]["flags"]["R-CRAWL"] is False and PR[sd]["flags"]["R-KEEP"] is True and PR[sd]["flags"]["R-NOBASE"] is True


def held(b):
    return "held" if b else "FAILS"


cellsP = {(sd, pair): raL[PAIRKEY[pair][0]][sd]["ADE6"] for sd in ("s0", "s1") for pair in ("main", "rep")}
adel = [("%s %s" % (sd, pair), cellsP[(sd, pair)]["delta"]) for sd in ("s0", "s1") for pair in ("main", "rep")]
assert (min(abs(d) for _, d in adel), max(abs(d) for _, d in adel)) == (0.0294, 0.0483), adel
jchk("the four ADE6 points lie between 0.0294 and 0.0483 m", "read_a.{A_CTRL, Arep_REP}.<seed>[ADE6].delta", str(adel), "the four points lie between 0.0294 and 0.0483 m", ["7.29"])
cells_r = []
for sd in ("s0", "s1"):
    sc = []
    for pair in ("main", "rep"):
        c = cellsP[(sd, pair)]
        sc.append("%s: %+.4f %s" % (held(PR[sd]["pairs"][pair]["guards"]["R-ADE"]), c["delta"], flag(c)))
    cells_r += [sc[0], sc[1], "**%s**" % held(PR[sd]["guards"]["R-ADE"])]
needl("| R-ADE (T1-kin ADE6, X " + MN + " Y, not separated-worse) | " + " | ".join(cells_r) + " |", "R-ADE predicate row")
for nm_, key, fmt, word in (("R-NEW (no row of D fires)", "R-NEW_fires", "held: %d of %d rows", "held"), ("R-NOBASE", "R-NOBASE_violations", "true: %d of %d rows violate", "true")):
    cs = [fmt % (sum(1 for v in PR[sd]["pairs"][pair][key].values() if v), len(PR[sd]["pairs"][pair]["domain"])) for sd in ("s0", "s1") for pair in ("main", "rep")]
    needl("| %s | %s | %s | **%s** | %s | %s | **%s** |" % (nm_, cs[0], cs[1], word, cs[2], cs[3], word), "predicate row " + nm_)
for nm_, key, lab in (("R-MOVE (every crawl row clears its bar)", "R-MOVE_rows", "false: %d of 4 rows clear"), ("R-CRAWL (every crawl row of X " + MN + " P10 not separated-worse)", "R-CRAWL_rows", "false: %d of 4 rows")):
    cs = [lab % sum(1 for v in PR[sd]["pairs"][pair][key].values() if v) for sd in ("s0", "s1") for pair in ("main", "rep")]
    needl("| %s | %s | %s | **false** | %s | %s | **false** |" % (nm_, cs[0], cs[1], cs[2], cs[3]), "predicate row " + nm_)
cs = ["held: %+.4f" % PR[sd]["pairs"][pair]["T0_point"] for sd in ("s0", "s1") for pair in ("main", "rep")]
needl("| R-T0 (T0 own-code ADE6, X " + MN + " Y, point <= +0.0604 m) | %s | %s | **held** | %s | %s | **held** |" % (cs[0], cs[1], cs[2], cs[3]), "predicate row R-T0")
cs = ["true: %+.4f" % PR[sd]["pairs"][pair]["KEEP_point"] for sd in ("s0", "s1") for pair in ("main", "rep")]
needl("| R-KEEP (T1-kin ADE6, X " + MN + " Y, point <= +0.0604 m) | %s | %s | **true** | %s | %s | **true** |" % (cs[0], cs[1], cs[2], cs[3]), "predicate row R-KEEP")
needl("| controls (reader, 206) | PASS | PASS | PASS | PASS | PASS | PASS |", "predicate row controls")
for nm_ in ("R-TLAT", "R-TLON"):
    needl("| %s (T head macro-F1, ticks >= 1, X %s Y not separated-worse) | held | held | **held** | held | held | **held** |" % (nm_, MN), "predicate row " + nm_)
for sd in ("s0", "s1"):
    for pair in ("main", "rep"):
        t0, kp = PR[sd]["pairs"][pair]["T0_point"], PR[sd]["pairs"][pair]["KEEP_point"]
        assert t0 <= 0.0604 and kp <= 0.0604
        needl("| %s | %s | %+.4f | held | %+.4f | true |" % (sd, PL[pair], t0, kp), "2.6 " + sd + pair)
addrow("2.7 the per-seed predicate table (9 lines: controls, R-TLAT, R-TLON, R-ADE, R-T0, R-NEW, R-MOVE, R-CRAWL, R-KEEP, R-NOBASE) rebuilt from rows.A.<seed>.pairs.<pair>.{guards, flags, domain, R-NEW_fires, R-NOBASE_violations, R-MOVE_rows, R-CRAWL_rows, T0_point, KEEP_point}",
       "rows.A", "10 of 10 lines equal; R-ADE fails at s0 (replicate pair) and at s1 (main pair)", "| R-ADE (T1-kin ADE6, X " + MN + " Y, not separated-worse) | held: -0.0294 B | FAILS: +0.0431 W |", "7.29")
addrow("2.6 R-T0 and R-KEEP: the 4 points, all at most +0.0604", "rows.A.<seed>.pairs.<pair>.{T0_point, KEEP_point}", "4 of 4 lines equal", "| s0 | main (A " + MN + " CTRL) | +0.0021 | held | -0.0294 | true |", "7.29")

# ---- ADE6 (2.2), R-MOVE (2.3), R-CRAWL (2.4): rebuilt line by line ----
CRAWL4 = ["yaw-rate MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share"]
n_ade = n_move = n_crawl = 0
imps_main, imps_all, shares_all, shares_main = [], [], [], []
for sd in ("s0", "s1"):
    for pair in ("main", "rep"):
        k_xy, k_xp, k_yp = PAIRKEY[pair]
        c, px, py = raL[k_xy][sd]["ADE6"], raL[k_xp][sd]["ADE6"], raL[k_yp][sd]["ADE6"]
        needl("| %s | %s | ADE6 | %s | %.4f | %.4f | %s | %s | 2399/141 |" % (sd, PL[pair], fmtc(c), c["mean_arm"], c["mean_ref"], fmtc(px).replace("**", ""), fmtc(py).replace("**", "")), "2.2 " + sd + pair)
        n_ade += 1
        for r in CRAWL4:
            c = raL[k_xy][sd][r]
            imp = -c["delta"]
            ratio = 100 * imp / BARL[r]
            clears = c["separated_better"] and imp >= BARL[r]
            needl("| %s | %s | %s | %s | %+.4f | %.4f | %.1f %% | %s |" % (sd, PL[pair], r, fmtc(c), imp, BARL[r], ratio, "true" if clears else "false"), "2.3 " + sd + pair + r)
            n_move += 1
            imps_all.append(ratio)
            if pair == "main":
                imps_main.append(ratio)
            xp, yp = raL[k_xp][sd][r], raL[k_yp][sd][r]
            share = 100 * (yp["delta"] - xp["delta"]) / yp["delta"]
            assert xp["separated_worse"] and yp["separated_worse"]
            needl("| %s | %s | %s | %s | %s | false | %.1f %% |" % (sd, PL[pair], r, fmtc(xp), fmtc(yp), share), "2.4 " + sd + pair + r)
            n_crawl += 1
            shares_all.append(share)
            if pair == "main":
                shares_main.append(share)
assert (n_ade, n_move, n_crawl) == (4, 16, 16)
addrow("2.2 ADE6: the 4 lines (X " + MN + " Y, means, X " + MN + " P10, Y " + MN + " P10) rebuilt from read_a.{A_CTRL, Arep_REP, A_P10, CTRL_P10, Arep_P10, REP_P10}.<seed>[ADE6]", "read_a.<pair>.<seed>.ADE6", "4 of 4 lines equal",
       "| s0 | main (A " + MN + " CTRL) | ADE6 |", "7.29")
addrow("2.3 R-MOVE: the 16 crawl lines (X " + MN + " Y, improvement = -delta, bar, improvement / bar, clears) rebuilt cell by cell", "read_a.<pair>.<seed>[crawl row]; floors.bar", "16 of 16 lines equal",
       "| s0 | replicate (A_rep " + MN + " REP) | L2 curvature MAE (GT>=5.2) |", "7.29")
addrow("2.4 R-CRAWL: the 16 lines (X " + MN + " P10, Y " + MN + " P10, share of the gap closed from the point deltas) rebuilt cell by cell", "read_a.{A_P10, CTRL_P10, Arep_P10, REP_P10}.<seed>[crawl row]", "16 of 16 lines equal",
       "| s1 | main (A " + MN + " CTRL) | curvature MAE |", "7.29")
clr = [(sd, p_, r) for sd in ("s0", "s1") for p_ in ("main", "rep") for r in CRAWL4 if raL[PAIRKEY[p_][0]][sd][r]["separated_better"] and -raL[PAIRKEY[p_][0]][sd][r]["delta"] >= BARL[r]]
sepb = [(sd, p_, r) for sd in ("s0", "s1") for p_ in ("main", "rep") for r in CRAWL4 if raL[PAIRKEY[p_][0]][sd][r]["separated_better"]]
assert clr == [("s0", "rep", "L2 curvature MAE (GT>=5.2)")] and len(sepb) == 2
jchk("2 of the 16 crawl cells separated-better, 1 clears its bar", "read_a.<pair>.<seed>[crawl rows]", "2; 1", "2 of the 16 cells are separated-better and 1 clears its bar", ["7.29"])
assert sum(1 for x in shares_all if x > 0) == 9 and sum(1 for x in shares_all if x < 0) == 7
jchk("share of the Y - P10 gap closed: 9 positive, 7 negative", "(Y - P10 - (X - P10)) / (Y - P10) from the point deltas", "9 / 7", "positive in 9 of the 16 cells and negative in 7", ["7.29"])
assert ("%.1f" % min(shares_all), "%.1f" % max(shares_all), "%.1f" % min(shares_main), "%.1f" % max(shares_main)) == ("-30.2", "28.7", "-3.0", "28.7")
jchk("share range over all 16", "(the same)", "%.1f to %.1f" % (min(shares_all), max(shares_all)), "from " + MN + "30.2 % to +28.7 %", ["7.29"])
jchk("main-pair share range", "(the same)", "%.1f to %.1f" % (min(shares_main), max(shares_main)), "lies between " + MN + "3.0 % and +28.7 %", ["7.29"])
assert ("%.1f" % min(imps_main), "%.1f" % max(imps_main), "%.1f" % min(imps_all), "%.1f" % max(imps_all)) == ("-23.4", "73.6", "-130.9", "158.1")
jchk("improvement / bar range over the 8 main-pair cells and over all 16", "(2.3 recomputed)", "-23.4 to +73.6; -130.9 to +158.1",
     "from " + MN + "23.4 % to +73.6 % over the eight main-pair cells, from " + MN + "130.9 % to +158.1 % over all 16", ["7.29"])
for sd, p_, r in clr + [("s1", "main", "curvature MAE")]:
    c = raL[PAIRKEY[p_][0]][sd][r]
    needl("| %s | %s | %s | %s | %.1f %% | %s |" % (sd, PL[p_], r, fmtc(c), 100 * (-c["delta"]) / BARL[r], "true" if (sd, p_, r) in clr else "false"), "numbers (3) table " + sd + p_ + r)
c = raL["A_CTRL"]["s0"]["L2 curvature MAE (GT>=5.2)"]
jchk("the main-pair partner of the one clearing cell: L2 at s0 main, 55.1 % of the bar", "read_a.A_CTRL.s0[L2 curvature MAE (GT>=5.2)]", cis(c) + "; %.1f %%" % (100 * -c["delta"] / BARL["L2 curvature MAE (GT>=5.2)"]),
     "reads " + cis(c) + ", not separated, 55.1 %", ["7.29"])

# ---- guards: R-NEW / R-NOBASE (2.5) ----
for sd in ("s0", "s1"):
    for pair in ("main", "rep"):
        k_xy, k_xp, k_yp = PAIRKEY[pair]
        dom = PR[sd]["pairs"][pair]["domain"]
        assert len(dom) == 15 and not any(PR[sd]["pairs"][pair]["R-NEW_fires"].values()) and not any(PR[sd]["pairs"][pair]["R-NOBASE_violations"].values())
        assert all(not raL[k_yp][sd][r]["separated_worse"] for r in dom)
        swl = [r for r in dom if raL[k_xy][sd][r]["separated_worse"]]
        wbl = [r for r in swl if (raL[k_xy][sd][r]["delta"] if raL[k_xy][sd][r]["direction"] == "lower" else -raL[k_xy][sd][r]["delta"]) >= BARL[r]]
        fires = [r for r in wbl if raL[k_xp][sd][r]["separated_worse"]]
        assert fires == []
        ylab = ("CTRL" if pair == "main" else "REP") + " " + MN + " P10 not separated-worse on D"
        needl("| %s | %s | %s | 15 | %s | %s | %s |" % (sd, PL[pair], ylab, ", ".join(swl) or "none", ", ".join(wbl) or "none", "none"), "2.5 " + sd + pair)
addrow("2.5 R-NEW / R-NOBASE: domain D has 15 rows in all four pairs; the X " + MN + " Y separated-worse rows, those worsening by at least the bar, and those whose X " + MN + " P10 is separated-worse, rebuilt per pair",
       "rows.A.<seed>.pairs.<pair>.{domain, R-NEW_fires, R-NOBASE_violations}; read_a", "4 of 4 lines equal; no row fires, no row violates", "| s0 | main (A " + MN + " CTRL) | CTRL " + MN + " P10 not separated-worse on D | 15 |", "7.29")
for r_, sd_, pr_ in (("speed MAE", "s1", "main"), ("accel MAE", "s1", "main"), ("cross-track MAE", "s1", "main"), ("L1 lateral-accel MAE", "s1", "main"), ("along-track MAE", "s0", "rep")):
    c = raL[PAIRKEY[pr_][0]][sd_][r_]
    assert c["separated_worse"] and c["delta"] >= BARL[r_]
    assert not raL[PAIRKEY[pr_][1]][sd_][r_]["separated_worse"]
    snip = ("along-track MAE meets (3), " + cis(c)) if r_ == "along-track MAE" else ("%s %s" % (r_, cis(c)))
    jchk("R-NEW condition (3) met, condition (1) fails: %s, %s %s" % (r_, sd_, pr_), "read_a.%s.%s[%s]; read_a.%s.%s[%s]" % (PAIRKEY[pr_][0], sd_, r_, PAIRKEY[pr_][1], sd_, r_), cis(c), snip, ["7.29"])

# ---- the four families: census (3.1), X - Y (3.2-3.5), X - P10 (3.8) ----
FAM = OrderedDict([("LONGITUDINAL", ["speed MAE", "along-track MAE", "accel MAE", "DK-VIOL", "DK-HW", "DK-TG"]),
                   ("LATERAL", ["heading MAE", "yaw-rate MAE", "curvature MAE", "cross-track MAE", "L1 lateral-accel MAE", "L2 curvature MAE (GT>=5.2)"]),
                   ("TACTICAL", ["TAC-LAT", "TAC-LON"]), ("STRATEGIC", ["S-POS@30", "S-POS@50", "S-POS@80"])])
ORD4 = [("A_CTRL", "s0"), ("Arep_REP", "s0"), ("A_CTRL", "s1"), ("Arep_REP", "s1")]
n_cell = 0
for fam, rows in FAM.items():
    cen = []
    for k, sd in ORD4:
        cs = [raL[k][sd][r] for r in rows]
        cen.append("%d / %d / %d" % (sum(1 for c in cs if c["separated_better"]), sum(1 for c in cs if c["separated_worse"]), sum(1 for c in cs if not c["separated"])))
    needl("| %s (%d) | %s |" % (fam, len(rows), " | ".join(cen)), "3.1 " + fam)
    for r in rows:
        cs = [raL[k][sd][r] for k, sd in ORD4]
        needl("| %s | %s | %s | %s |" % (r, cs[0]["direction"], " | ".join(fmtc(c) for c in cs), " ; ".join(c["n"] for c in cs)), "3.2-3.5 " + r)
        n_cell += 4
assert n_cell == 68
n_p10 = 0
for sd in ("s0", "s1"):
    for r in ROWS19L:
        cs = [raL[k][sd][r] for k in ("A_P10", "CTRL_P10", "Arep_P10", "REP_P10")]
        needl("| %s | %s | %s |" % (r, cs[0]["direction"], " | ".join(fmtc(c) for c in cs)), "3.8 " + sd + r)
        n_p10 += 4
assert n_p10 == 152
addrow("3.1 census of separated cells per family (4 families x 4 pair-seed cells) recomputed from read_a", "read_a.{A_CTRL, Arep_REP}.<seed>[family rows].{separated_better, separated_worse, separated}", "4 of 4 lines equal (16 census cells)", "| LONGITUDINAL (6) | 2 / 0 / 4 |", "7.29")
addrow("3.2-3.5 the 17 family rows x 4 X " + MN + " Y cells (68 cells) with their n, rebuilt", "read_a.{A_CTRL, Arep_REP}.<seed>[row]", "17 of 17 lines equal", "| speed MAE | lower |", "7.29")
addrow("3.8 X " + MN + " P10 and Y " + MN + " P10 for 19 rows x 2 seeds x 4 columns (152 cells), rebuilt", "read_a.{A_P10, CTRL_P10, Arep_P10, REP_P10}.<seed>[row]", "38 of 38 lines equal", "| ADE6 | lower | **-1.2305 [-1.4302, -1.0448] B** |", "7.29")
assert LIR["strategic_family"].startswith("NOT PAUSED for v8F")
assert set(LIR["not_built"]) == {"M-LAUNCH", "anti_echo_block", "M2_and_L2_launch_subset", "t0_four_families"} and LIR["next_launch"] is None
for k_, v_ in LIR["not_built"].items():
    if k_ == "t0_four_families":
        jchk("not_built.t0_four_families verbatim", "not_built.t0_four_families", v_[:50] + "...", v_.split(": NOT RUN")[0] + ": NOT RUN", ["7.29"])
    else:
        jchk("not_built.%s verbatim" % k_, "not_built." + k_, v_[:50] + "...", v_, ["7.29"])
jchk("next_launch is null", "next_launch", "null", "The JSON field `next_launch` is null", ["7.29"])
jchk("STRATEGIC not paused for v8F", "strategic_family", LIR["strategic_family"][:40], "STRATEGIC is not paused for v8F", ["7.29"])

# ---- the 76-cell census, the bar-reaching cells, RF / bar ----
cells76 = [raL[PAIRKEY[p_][0]][sd][r] for p_ in ("main", "rep") for sd in ("s0", "s1") for r in ROWS19L]
nb, nw = sum(1 for c in cells76 if c["separated_better"]), sum(1 for c in cells76 if c["separated_worse"])
assert (len(cells76), nb, nw, 76 - nb - nw) == (76, 16, 7, 53)
jchk("76 X - Y cells: 16 separated-better, 7 separated-worse, 53 not separated", "read_a.{A_CTRL, Arep_REP}.<seed>[19 rows]", "16 / 7 / 53", "**16 are separated-better, 7 separated-worse and 53 not separated.**", ["7.29"])
per = {}
for p_ in ("main", "rep"):
    for sd in ("s0", "s1"):
        cs = [raL[PAIRKEY[p_][0]][sd][r] for r in ROWS19L]
        per[(p_, sd)] = (sum(1 for c in cs if c["separated_better"]), sum(1 for c in cs if c["separated_worse"]), sum(1 for c in cs if not c["separated"]))
txt = "main s0 %d / %d / %d, replicate s0 %d / %d / %d, main s1 %d / %d / %d, replicate s1 %d / %d / %d" % (per[("main", "s0")] + per[("rep", "s0")] + per[("main", "s1")] + per[("rep", "s1")])
jchk("per pair and seed census of the 76 cells", "(the same)", txt, txt, ["7.29"])
ri = [(sd, p_, r) for p_ in ("main", "rep") for sd in ("s0", "s1") for r in ROWS19L if impv(raL[PAIRKEY[p_][0]][sd][r]) >= BARL[r]]
rw = [(sd, p_, r) for p_ in ("main", "rep") for sd in ("s0", "s1") for r in ROWS19L if -impv(raL[PAIRKEY[p_][0]][sd][r]) >= BARL[r]]
assert len(ri) == 3 and len(rw) == 6 and sum(1 for sd, p_, r in rw if raL[PAIRKEY[p_][0]][sd][r]["separated_worse"]) == 5, (ri, rw)
assert raL["A_CTRL"]["s1"]["DK-TG"]["delta"] == -0.0094 and raL["Arep_REP"]["s0"]["DK-TG"]["delta"] == -0.0245
jchk("3 cells reach the bar improving, 6 reach it worsening, 5 of those separated-worse", "(impv >= bar over the 76 cells)", "3; 6; 5", "Three cells reach the bar in the improving direction", ["7.29"])
jchk("DK-TG points", "read_a.A_CTRL.s1[DK-TG].delta; read_a.Arep_REP.s0[DK-TG].delta", "-0.0094; -0.0245", "(" + MN + "0.0094) and in the replicate pair at s0 (" + MN + "0.0245)", ["7.29"])
jchk("six cells reach the bar worsening, five separated-worse", "(impv >= bar over the 76 cells)", "6; 5", "Six cells reach it in the worsening direction and five of those are separated-worse", ["7.29"])
RFB = {r: flL["RF"][r] / BARL[r] for r in ROWS19L}
assert sum(1 for v in RFB.values() if v > 1) == 13 and "%.2f" % RFB["DK-VIOL"] == "0.17" and "%.2f" % RFB["yaw-rate MAE"] == "14.32"
jchk("RF / bar from 0.17 (DK-VIOL) to 14.32 (yaw-rate MAE), above 1 on 13 of 19 rows", "floors.RF / floors.bar", "0.17; 14.32; 13",
     "RF \u00f7 bar runs from 0.17 (DK-VIOL) to 14.32 (yaw-rate MAE) and exceeds 1 on 13 of the 19 rows", ["7.29"])
BETTER = {r: raL["A_CTRL"]["s0"][r]["direction"] for r in ROWS19L}
for r in ROWS19L:
    f_, rf0, rf = flL["F"][r], flL["RF0"][r], flL["RF"][r]
    assert abs(2 * max(f_, rf0) - BARL[r]) < 1e-9
    needl("| %s | %s | %.4f | %.4f | **%.4f** | %.4f | %.2f |" % (r, BETTER[r], f_, rf0, BARL[r], rf, RFB[r]), "2.1 " + r)
    pl = [l for l in SOURCES["PREREG_V8F_LAUNCH_INTENT.md"].split("\n") if l.startswith("| %s | " % r) and l.count("|") == 7 and "**" in l and " / " not in l][0]
    pc = [x.strip() for x in pl.strip("|").split("|")]
    assert pc[:5] == [r, "%.4f" % f_, "%.4f" % rf0, "**%.4f**" % BARL[r], "%.4f" % rf], (pc, r)
addrow("2.1 the 19-row floors table (F, RF0, bar = 2 x max(F, RF0), RF, RF / bar), rebuilt from floors.{F, RF0, bar, RF}; its F, RF0, bar and RF columns also equal the PREREG's committed 19-row table, cell by cell", "floors", "19 of 19 lines equal",
       "| ADE6 | lower | 0.1207 | 0.0138 | **0.2414** | 0.4025 | 1.67 |", "7.29")
op = []
for sd in ("s0", "s1"):
    a_, b_ = raL["A_CTRL"][sd], raL["Arep_REP"][sd]
    for r in ROWS19L:
        if a_[r]["separated"] and b_[r]["separated"] and (a_[r]["delta"] > 0) != (b_[r]["delta"] > 0):
            op.append((sd, r))
assert op == [("s0", "ADE6"), ("s1", "ADE6"), ("s1", "speed MAE"), ("s1", "accel MAE"), ("s1", "L1 lateral-accel MAE")], op
jchk("rows with separated, opposite-sign X - Y cells in the two pairs of a seed", "read_a.{A_CTRL, Arep_REP}.<seed>[row]", str(op), "at s1, speed MAE, accel MAE and L1 lateral-accel MAE (at s0 only ADE6)", ["7.29"])
for sd, key in (("s0", "0 / 10"), ("s1", "1 / 11")):
    a_, b_ = raL["A_CTRL"][sd]["ADE6"], raL["Arep_REP"][sd]["ADE6"]
    needl("| %s | %s | %s | %s |" % (sd, key, fmtc(a_), fmtc(b_)), "numbers (2) table " + sd)
c1, c2 = raL["A_CTRL"]["s1"]["accel MAE"], raL["Arep_REP"]["s1"]["accel MAE"]
jchk("accel MAE at s1: main and replicate", "read_a.{A_CTRL, Arep_REP}.s1[accel MAE]", cis(c1) + "; " + cis(c2), "accel MAE at s1 reads " + cis(c1) + " in the main pair and " + cis(c2) + " in the replicate pair", ["7.29"])

# ---- the slow-draw flag ----
SD = LIR["beside"]["A"]["slow_draw"]
cells20 = [(k, c) for k, v in SD.items() for c in v["cells"].values()]
assert len(cells20) == 20 and all(v["label"] is None for v in SD.values()) and all(v["threshold_m_s"] == 0.5 for v in SD.values())
mx = max(abs(c["mean"]) for _, c in cells20)
ex = sum(1 for _, c in cells20 if c["lo"] > 0 or c["hi"] < 0)
assert "%.4f" % mx == "0.1627" and ex == 13
jchk("slow-draw flag: no arm labelled; largest shift 0.1627 m/s; 13 of 20 cell intervals exclude 0", "beside.A.slow_draw", "none; 0.1627; 13", "the largest shift in magnitude over the 20 cells is 0.1627 m/s, and 13 of the 20 cell intervals exclude 0", ["7.29"])
SDL = {"main:A-CTRL:s0": "s0 main (A " + MN + " CTRL)", "rep:Arep-REP:s0": "s0 replicate (A_rep " + MN + " REP)", "main:A-CTRL:s1": "s1 main (A " + MN + " CTRL)", "rep:Arep-REP:s1": "s1 replicate (A_rep " + MN + " REP)"}
for k in ("main:A-CTRL:s0", "rep:Arep-REP:s0", "main:A-CTRL:s1", "rep:Arep-REP:s1"):
    cs = SD[k]["cells"]
    cc = " | ".join("%+.4f [%+.4f, %+.4f]" % (cs[n]["mean"], cs[n]["lo"], cs[n]["hi"]) for n in ("lon_v_target_ms", "v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms"))
    needl("| %s | %s | none |" % (SDL[k], cc), "slow-draw " + k)
addrow("slow-draw table: 4 arm-pair lines (5 cells each, mean [lo, hi], label none) rebuilt from beside.A.slow_draw", "beside.A.slow_draw", "4 of 4 lines equal", "| s0 main (A " + MN + " CTRL) | +0.0718 [+0.0195, +0.1256] |", "7.29")

# ---- M-LAUNCH ----
MC = MLJ["primary"]["cells"]
assert MLJ["M_LAUNCH"] == "not moved" and MLJ["launch_subset"]["n_windows"] == 327 and MLJ["launch_subset"]["n_episodes"] == 49 and MLJ["launch_subset"]["of_windows"] == 2399
for (pl_, sd, key) in (("main (A " + MN + " DAG-OACC)", "s0", "main_s0"), ("main (A " + MN + " DAG-OACC)", "s1", "main_s1"), ("replicate (A_rep " + MN + " REP)", "s0", "replicate_s0"), ("replicate (A_rep " + MN + " REP)", "s1", "replicate_s1")):
    x = MC[key]["x_minus_y"]
    assert x["separated"] is False and MC[key]["n_windows"] == 327 and MC[key]["n_episodes"] == 49
    needl("| %s | %s | %s | no |" % (pl_, sd, cis(x)), "M-LAUNCH " + key)
addrow("M-LAUNCH: the 4 cells (X " + MN + " Y [95 % CI], not separated) rebuilt from MLAUNCH.json primary.cells; 327 windows / 49 episodes in every cell", "primary.cells.<cell>.x_minus_y; launch_subset", "4 of 4 lines equal; M_LAUNCH = not moved",
       "| main (A " + MN + " DAG-OACC) | s0 | +0.0348 [" + MN + "0.0299, +0.0923] | no |", "7.29")
fl2 = MLJ["beside_training_seed_floor"]
r0, r1 = fl2["REP_minus_CTRL_s0"]["x_minus_y"], fl2["REP_minus_CTRL_s1"]["x_minus_y"]
big = max(abs(MC[k]["x_minus_y"]["delta"]) for k in MC)
assert r0["separated"] is True and r1["separated"] is False and round(abs(r0["delta"]) / big) == 16 and big == 0.0348
jchk("training draw alone, REP - DAG-OACC at s0, about 16 times the largest M-LAUNCH point (+0.0348)", "beside_training_seed_floor.REP_minus_CTRL_s0.x_minus_y", cis(r0) + "; ratio %.2f" % (abs(r0["delta"]) / big),
     "is " + cis(r0) + " m/s, about 16 times the lever's largest point (+0.0348)", ["7.29"])
jchk("REP - DAG-OACC at s1 in the numbers item", "beside_training_seed_floor.REP_minus_CTRL_s1.x_minus_y.delta", sg(r1["delta"]), "(" + sg(r1["delta"]) + " at s1)", ["7.29"])
mctl = MLJ["controls"]
grp = [k for k in mctl if k != "PASS"]
assert len(grp) == 9 and mctl["PASS"] is True and all(mctl[k]["PASS"] is True for k in grp)
jchk("M-LAUNCH build controls: 9 groups, all PASS", "controls.*.PASS", "9 of 9", "(all nine groups PASS)", ["7.29"])
M2 = MLJ["beside_M2"]["pairs"]
nm = {"along_mae_m": "along-track MAE", "speed_mae_mps": "speed MAE", "L2_curvature_mae_fast_1pm": "L2 curvature MAE"}
pn = {"main_s0": "main pair s0", "main_s1": "main pair s1", "replicate_s0": "replicate pair s0", "replicate_s1": "replicate pair s1"}
sepM2 = [(pk, rk, rv["x_minus_y"]) for pk, pv in M2.items() for rk, rv in pv["rows"].items() if rv["x_minus_y"]["separated"]]
assert len(sepM2) == 4 and sum(len(pv["rows"]) for pv in M2.values()) == 12
for pk, rk, x in sepM2:
    jchk("M2 beside, separated cell: %s %s" % (nm[rk], pk), "beside_M2.pairs.%s.rows.%s.x_minus_y" % (pk, rk), cis(x), "%s, %s %s (%s)" % (nm[rk], pn[pk], cis(x), "better" if x["delta"] < 0 else "worse"), ["7.29"])
ln = sorted({v["rows"]["L2_curvature_mae_fast_1pm"]["x_minus_y"]["n_windows"] for v in M2.values()})
assert ln == [253, 254, 255] and all(v["rows"]["L2_curvature_mae_fast_1pm"]["x_minus_y"]["n_episodes"] == 45 for v in M2.values())
jchk("M2 beside: 253 to 255 windows / 45 episodes for L2; 12 cells, 4 separated", "beside_M2.pairs.*.rows.L2_curvature_mae_fast_1pm.x_minus_y.{n_windows, n_episodes}", "253..255 / 45; 12; 4", "253 to 255 windows / 45 episodes for L2", ["7.29"])

# ---- the run record: K-SHARE, K-MASK, the gates, the box-2 times, the T1-kin steps ----
RUNS = [("A_0", "M2b_s0_launchintent", "0"), ("A_1", "M2b_s1_launchintent", "1"), ("A_rep,0", "M2b_s0_launchintentrep", "10"), ("A_rep,1", "M2b_s1_launchintentrep", "11")]
CL = CHL.split("\n")
ck = {}
for ln_ in CL:
    m = re.match(r"\d\d-\d\d \d\d:\d\d (\S+) checkpoint sha256 \(trained\) ([0-9a-f]{12})", ln_)
    if m:
        ck[m.group(1)] = m.group(2)
assert len(ck) == 4, ck
for lab, run, seed in RUNS:
    rr = [l for l in CL if l.startswith("LAUNCH-INTENT run record: PASS | %s |" % lab)]
    ks = [l for l in CL if l.startswith("LAUNCH-INTENT K-SHARE: PASS | %s |" % lab)]
    assert len(rr) == 1 and len(ks) == 1, (lab, len(rr), len(ks))
    a16 = re.search(r"argv sha16 ([0-9a-f]{16})", rr[0]).group(1)
    m = re.search(r"launch (\d+) / (\d+) = ([0-9.]+) .*weight share ([0-9.]+)", ks[0])
    nl_, rows_, frac, share = m.group(1), m.group(2), m.group(3), m.group(4)
    assert re.search(r"trainer seed (\d+)", rr[0]).group(1) == seed
    needl("| %s (%s) | %s | %s / %s | %s | %s | `%s` | `%s` |" % (lab, run, seed, nl_, rows_, frac, share, a16, ck[run]), "4.3 " + lab)
for lab, run, seed in RUNS:
    parts = run.split("_")
    gl = [l for l in CL if l.startswith("LAUNCH-INTENT gate: PASS | %s %s |" % (parts[2], parts[1]))]
    assert len(gl) == 2 and gl[0] == gl[1], (lab, len(gl))
    g0 = gl[0]
    sw = re.search(r"swap probe grad diff ([0-9.eE+-]+)", g0).group(1)
    lp = re.search(r"launch probe contribution ([0-9.eE+-]+)", g0).group(1)
    hg = re.search(r"head_o grad diff ([0-9.eE+-]+)", g0).group(1)
    pb = re.search(r"probe batch launch (\d+ / \d+)", g0).group(1)
    wv = re.search(r"launch W ([0-9.]+)", g0).group(1)
    ts = re.search(r"trainer seed (\d+)", g0).group(1)
    assert float(hg) > 0 and ts == seed
    needl("| %s | %s | %s | %s | %s | %s | %s |" % (lab, wv, ts, pb, lp, hg, sw), "4.4 " + lab)
addrow("4.3 K-SHARE / K-MASK / checkpoint prefix / argv sha16 for the 4 runs, rebuilt from the banked box-2 chain log (run record, K-SHARE and 'checkpoint sha256 (trained)' lines)", "chain_v8f_launchintent.out", "4 of 4 lines equal",
       "| A_0 (M2b_s0_launchintent) | 0 | 30727 / 191203 | 0.160704 | 0.331451 |", "7.29")
addrow("4.4 the four launch gates (W, trainer seed, probe batch, launch probe contribution, head_o grad diff, swap probe grad diff), rebuilt from the chain log's 'LAUNCH-INTENT gate: PASS' lines; head_o grad diff > 0 in all four", "chain_v8f_launchintent.out",
       "4 of 4 lines equal", "| A_0 | 1.58926 | 0 | 11 / 64 |", "7.29")
sh = sorted(float(re.search(r"weight share ([0-9.]+)", l).group(1)) for l in CL if l.startswith("LAUNCH-INTENT K-SHARE: PASS"))
assert ["%.4f" % x for x in sh] == ["0.3256", "0.3281", "0.3307", "0.3315"] and all(abs(x - 1 / 3) <= 0.02 for x in sh)
jchk("the four weight shares at four decimals equal the PREREG's ESTIMATED shares (0.3315, 0.3307, 0.3256, 0.3281); all inside 1/3 +- 0.02", "chain log K-SHARE lines", "0.3256 .. 0.3315", "(0.3315, 0.3307, 0.3256, 0.3281 at four decimals)", ["7.29"])
jchk("weight share range in 8.1", "chain log K-SHARE lines", "0.3256 .. 0.3315", "0.3256 to 0.3315 against 1/3", ["8.1"])


def stamp_min(hhmm):
    h_, m_ = hhmm.split(":")
    return int(h_) * 60 + int(m_)


stamps = [re.match(r"10-28 (\d\d:\d\d) ", l).group(1) for l in CL if re.match(r"10-28 \d\d:\d\d ", l)]
assert stamps[0] == "13:54" and stamps[-1] == "16:48", (stamps[0], stamps[-1])
span_min = stamp_min(stamps[-1]) - stamp_min(stamps[0])
assert span_min == 174 and "%.1f" % (span_min / 60) == "2.9"
assert [l for l in CL if l.startswith("ZZV8FMINI-DONE 10-28 16:48 tip d8c9f7bfc91798b6fafb5a13dd793771e2f79f6a:")]
jchk("box-2 time: first stamped line 13:54 to DONE marker 16:48 = 174 min = 2.9 h", "chain log: first and last 'MM-DD hh:mm' stamps", "%d min; %.1f h" % (span_min, span_min / 60), "**174 min or 2.9 h**", ["7.29"])
jchk("174 min in the abstract", "(the same)", "174 min", "174 min of box 2", ["abstract"])
job = {}
cur = None
for l in CL:
    m = re.match(r"10-28 (\d\d:\d\d) launched pid=\d+ rc=0 log=\S*?v8flaunchintent\d+_(\w+)\.log", l)
    if m:
        cur = m.group(2)
        job[cur] = [m.group(1), None]
        continue
    m = re.match(r"===== ZZJOB-END \d\d\.\d\d\.\d{4} (\d\d:\d\d:\d\d),\d+ rc=0", l)
    if m and cur and job[cur][1] is None:
        job[cur][1] = m.group(1)
for lab, run, seed in RUNS:
    t_, e_ = job["t_" + run], job["e_" + run]
    needl("| %s (%s) | %s | %s | %s | %s |" % (lab, run, t_[0], t_[1], e_[0], e_[1]), "9 box-2 " + lab)
addrow("9 box-2 run times: train launch minute, train ZZJOB-END, T0 eval launch minute, T0 eval ZZJOB-END for the 4 runs, rebuilt from the chain log", "chain_v8f_launchintent.out", "4 of 4 lines equal", "| A_0 (M2b_s0_launchintent) | 14:02 | 14:41:35 | 14:41 | 14:43:37 |", "7.29")
gj = [(job["gate_" + r][0], job["gate_" + r][1]) for _, r, _ in RUNS]
assert [g[0] for g in gj] == ["13:57", "13:58", "14:00", "14:01"] and [g[1] for g in gj] == ["13:58:34", "14:00:02", "14:01:33", "14:02:37"], gj
jchk("the four gates: launched 13:57 / 13:58 / 14:00 / 14:01, ZZJOB-END 13:58:34 / 14:00:02 / 14:01:33 / 14:02:37", "chain log: gate jobs", str(gj),
     "(launched 13:57, 13:58, 14:00 and 14:01; ZZJOB-END 13:58:34, 14:00:02, 14:01:33 and 14:02:37)", ["7.29"])
assert job["refresh"][0] == "13:54" and job["refresh"][1] == "13:54:55" and job["t_M2b_s0_launchintent"][0] == "14:02"
jchk("refresh ZZJOB-END 13:54:55; A_0 launched 14:02", "chain log: refresh, t_M2b_s0_launchintent", "13:54:55; 14:02", "(ZZJOB-END 13:54:55, rc 0; box-2 code = tip `d8c9f7bf` at", ["7.29"])
pulls = [re.match(r"10-28 (\d\d:\d\d) (\S+) pulled to", l) for l in CL]
pulls = [(m.group(2), m.group(1)) for m in pulls if m]
assert pulls == [("M2b_s0_launchintent", "14:44"), ("M2b_s1_launchintent", "15:25"), ("M2b_s0_launchintentrep", "16:06"), ("M2b_s1_launchintentrep", "16:48")], pulls
jchk("the pulls at 14:44 / 15:25 / 16:06 / 16:48", "chain log: 'pulled to' lines", str(pulls), "14:44 (A_0), 15:25 (A_1), 16:06 (A_rep,0) and 16:48 (A_rep,1)", ["7.29"])

# ---- the T1-kin chain steps ----
steps = {}
for l in T1C.split("\n"):
    m = re.match(r"10-28 \d\d:\d\d:\d\d end (\S+(?: eval \S+)?) rc=0 wall=(\d+)s", l)
    if m:
        steps[m.group(1)] = int(m.group(2))
assert len(steps) == 12 and sum(steps.values()) == 1756, (len(steps), sum(steps.values()), steps)
STEPKEY = [("A_0", "LI_s0_m10_causal_heads", "LI_s0_m10_logged_heads", "fresh-T0 eval M2b-launchintent_s0-fresh"), ("A_1", "LI_s1_m10_causal_heads", "LI_s1_m10_logged_heads", "fresh-T0 eval M2b-launchintent_s1-fresh"),
           ("A_rep,0", "LIREP_s0_m10_causal_heads", "LIREP_s0_m10_logged_heads", "fresh-T0 eval M2b-launchintentrep_s0-fresh"), ("A_rep,1", "LIREP_s1_m10_causal_heads", "LIREP_s1_m10_logged_heads", "fresh-T0 eval M2b-launchintentrep_s1-fresh")]
for lab, a_, b_, c_ in STEPKEY:
    needl("| %s | %d | %d | %d |" % (lab, steps[a_], steps[b_], steps[c_]), "T1-kin steps " + lab)
addrow("the 12 T1-kin chain steps (causal, logged, fresh T0 per run) rebuilt from the relaunch's chain.out; they sum to 1756 s = 29.3 min", "lit1_logs/chain.out", "4 of 4 lines equal; sum %d s = %.1f min" % (sum(steps.values()), sum(steps.values()) / 60),
       "| A_0 | 218 | 218 | 88 |", "7.29")
jchk("the twelve steps sum to 1756 s, 29.3 min of dev-box CPU", "lit1_logs/chain.out: 'end ... rc=0 wall=<n>s'", "%d s; %.1f min" % (sum(steps.values()), sum(steps.values()) / 60), "The twelve steps sum to 1756 s (**[M]**, arithmetic on the table), 29.3 min of dev-box CPU", ["7.29"])
jchk("12 steps in 1756 s in the Provenance bullet", "(the same)", "1756 s", "12 steps in 1756 s", ["7.29"])

# ---- section J: the log-derived disclosures and the commit times and pinned hashes, re-measured ----
glog = {}
for line in SOURCES["git log (cited commits)"].split("\n"):
    m = re.match(r"([0-9a-f]{7,12}) (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ([+-]\d{4}) ", line)
    if m:
        glog[m.group(1)[:7]] = (m.group(2), m.group(3))  # %h is 7 hex digits here; the text cites 8


def GLOG(h):
    return glog[h[:7]]


for h, t in (("77234276", "2026-10-28 13:49:00"), ("d8c9f7bf", "2026-10-28 13:51:52"), ("e28516e8", "2026-10-28 15:24:46"), ("d7dc9f11", "2026-10-28 17:11:44"), ("8c1e9636", "2026-10-28 17:43:02")):
    assert GLOG(h)[0] == t, (h, GLOG(h))
lchk("commit 77234276 (PREREG promoted) at 13:49:00 and commit d8c9f7bf (queue stage) at 13:51:52 bracket the pin fill", "git log --format=%ad", "%s ; %s" % (GLOG("77234276")[0], GLOG("d8c9f7bf")[0]),
     "before commit `d8c9f7bf` (13:51:52) and after the PREREG was promoted at `77234276` (13:49:00)", ["7.29"])
lchk("commit e28516e8 (Amendment 1) at 15:24:46", "git log --format=%ad", GLOG("e28516e8")[0], "It landed at `e28516e8` (15:24:46)", ["7.29"])
lchk("RESULT landed 8c1e9636 at 17:43:02; the read banked at d7dc9f11 at 17:11:44", "git log --format=%ad", "%s ; %s" % (GLOG("8c1e9636")[0], GLOG("d7dc9f11")[0]), "(landed `8c1e9636`, 17:43:02; the read banked at `d7dc9f11`, 17:11:44)", ["7.29"])
sha = lambda b: hashlib.sha256(b).hexdigest()
assert sha(PLI_PROMOTED.encode("utf-8")).startswith("469bac1f3918b31b") and sha(READER_AT_PROMOTION).startswith("5e950dcf")
amend = git(["show", f"{TIP}:Project Steering/PREREG_V8F_LAUNCH_INTENT.md"])
assert sha(amend).startswith("6ae79f2f451868e8") and sha(git(["show", "e28516e8:Project Steering/PREREG_V8F_LAUNCH_INTENT.md"])) == sha(amend) == sha(git(["show", "d7dc9f11:Project Steering/PREREG_V8F_LAUNCH_INTENT.md"]))
lchk("the PREREG blob and the reader blob at 77234276 hash to 469bac1f and 5e950dcf (the pins the chain carries)", "git show 77234276:<path> | sha256", "%s ; %s" % (sha(PLI_PROMOTED.encode("utf-8"))[:8], sha(READER_AT_PROMOTION)[:8]),
     "hash to the `469bac1f…` and `5e950dcf…` that the pins carry", ["7.29"])
lchk("the amended PREREG has sha256 6ae79f2f at e28516e8, d7dc9f11 and the tip", "git show <rev>:Project Steering/PREREG_V8F_LAUNCH_INTENT.md | sha256", sha(amend)[:8], "gives the amended text sha256 `6ae79f2f…`", ["7.29"])
pins = SOURCES["launch-intent-chain/PINS_FILLED.txt"]
assert "14:10" in pins
lchk("PINS_FILLED.txt carries the 14:10 stamp", "launch-intent-chain/PINS_FILLED.txt", [l for l in pins.split("\n") if "14:10" in l][0][:100], "says the Master Mind filled the chain pins at ~14:10", ["7.29"])
PRV = [l for l in S29.split("\n") if l.startswith("| ") and re.search(r"\| `[0-9a-f]{64}` \|$", l)]
n_blob = n_ck = 0
BASE = R + "raw/2026-10-28-launch-intent-read/"
for l in PRV:
    cells = [c.strip() for c in l.strip("|").split("|")]
    role, path, h = cells[0], cells[1].strip("`"), cells[2].strip("`")
    if path.startswith("E:/"):
        run = path.split("/")[-2]
        assert h[:12] == ck[run] and h in RESULT_SRC, (run, h)
        n_ck += 1
        continue
    if path.startswith("Project Steering/"):
        assert h == sha(amend)
        n_blob += 1
        continue
    rel = path.split(" at ")[0]
    full = (R + "raw/2026-10-28-launch-intent-chain/" + rel.split("/")[-1]) if rel.startswith("../") else (BASE + rel)
    assert sha(git(["show", f"d7dc9f11:{full}"])) == h, (role, full)
    n_blob += 1
assert n_blob == 15 and n_ck == 4, (n_blob, n_ck, len(PRV))
LOG_ROWS.append(("the provenance table of 7.29: %d repo rows, each sha256 equal to the sha256 of the blob at d7dc9f11 (the PREREG row: the blob at e28516e8, d7dc9f11 and the tip); %d checkpoint rows, each full sha256 begins with the 12-hex prefix of the chain log's 'checkpoint sha256 (trained)' line and occurs in the RESULT" % (n_blob, n_ck),
                 "git show d7dc9f11:<path> | sha256; chain_v8f_launchintent.out; RESULT_V8F_LAUNCH_INTENT.md", "%d of %d equal" % (n_blob + n_ck, len(PRV)), "| read of record (reader output) | `read_A1/LAUNCH_INTENT_READ.json` | `e451e1a6ba44e1d3", "7.29"))
assert "| read of record (reader output) | `read_A1/LAUNCH_INTENT_READ.json` | `e451e1a6ba44e1d3" in S29
last = [l for l in T1P.split("\n") if l.strip()][-1]
assert last.startswith("10-28 15:07:07 ") and "LOW RAM available 3040 MB < 4096 MB" in last
assert not [l for l in T1P.split("\n") if re.match(r"10-28 \d\d:\d\d:\d\d start ", l)]
lchk("the stopped launch's last line is stamped 15:07:07 and reads 'LOW RAM available 3040 MB < 4096 MB'; its log holds no 'start' line for a rollout", "lit1_logs/chain.out.prev_1028151251", last[:120], "the stopped launch's last line, 15:07:07, reads", ["7.29"])
t1c = [l for l in T1C.split("\n") if l.strip()]
assert t1c[0].startswith("10-28 15:12:51 === lit1_chain start")
first_start = [l for l in t1c if re.match(r"10-28 \d\d:\d\d:\d\d start ", l)][0]
assert first_start.startswith("10-28 15:13:04 start LI_s0_m10_causal_heads (attempt 1;")
att = [re.search(r"\(attempt (\d+);", l).group(1) for l in t1c if re.match(r"10-28 \d\d:\d\d:\d\d start (LI|LIREP)_", l)]
assert att == ["1"] * 8, att
lchk("the relaunch started at 15:12:51, its first rollout at 15:13:04, all eight rollouts show attempt 1", "lit1_logs/chain.out", "15:12:51; 15:13:04; attempts %s" % att, "The relaunch started at 15:12:51 and its first rollout at 15:13:04, all eight rollouts in its log show attempt 1", ["7.29"])
assert "template proof 1: rollout_causal.tmpl rendered with DAG-OACC s0's parameters == its banked rollout argv (32 tokens, byte for byte)" in T1C and "template proofs 2-4:" in T1C
assert "== all 12 steps done" in T1C
a0_end = [l for l in t1c if "end fresh-T0 eval M2b-launchintent_s0-fresh" in l][0]
assert a0_end.startswith("10-28 15:22:14 ")
lchk("A_0's three T1-kin steps ran from 15:13:04 to 15:22:14", "lit1_logs/chain.out", "15:13:04 .. 15:22:14", "A_0's three T1-kin steps ran from 15:13:04 to 15:22:14", ["7.29"])
assert "10-28 16:55:55 == reader:" in T1C and "10-28 17:01:36 reader rc=2: LAUNCH-INTENT READ REFUSED" in T1C
REF = SOURCES["lit1_logs/launch_intent_read_REFUSED.out"]
assert "LAUNCH-INTENT READ REFUSED: failed controls:" in REF and len(re.findall(r"^\s+- ", REF, flags=re.M)) == 1 and "  - K-BUILD [" in REF
lchk("the chain's own read ran 16:55:55 to 17:01:36 and REFUSED on one failed control, K-BUILD", "lit1_logs/chain.out; lit1_logs/launch_intent_read_REFUSED.out", "16:55:55 .. 17:01:36; 1 failed control: K-BUILD",
     "The chain's own read ran 16:55:55 to 17:01:36 and REFUSED", ["7.29"])
RA1 = SOURCES["read_A1/read_A1.out"]
ra1 = RA1.split("\n")
assert re.match(r"17:02:21 \[LI_s0_m10_causal_heads\] ticks loaded", [l for l in ra1 if re.match(r"\d\d:\d\d:\d\d \[", l)][0])
assert ra1[0].startswith("reader fb3d2d7d945b5420") and " prereg 6ae79f2f451868e8" in ra1[0]
assert "ZZLAUNCH-INTENT-READ-DONE HARMS" in RA1 and "ZZREAD-A1-RC 0" in RA1
lchk("the read of record: first console stamp 17:02:21; reader fb3d2d7d with PREREG 6ae79f2f on the console's first line; the console ends with ZZLAUNCH-INTENT-READ-DONE HARMS and ZZREAD-A1-RC 0", "read_A1/read_A1.out", "17:02:21; fb3d2d7d / 6ae79f2f; HARMS; rc 0",
     "before the read of record started (first console stamp 17:02:21)", ["7.29"])
lchk("the console's last lines", "read_A1/read_A1.out", "ZZLAUNCH-INTENT-READ-DONE HARMS; ZZREAD-A1-RC 0", "*\"ZZLAUNCH-INTENT-READ-DONE HARMS\"* and *\"ZZREAD-A1-RC 0\"*", ["7.29"])
CKR = SOURCES["check_result_filled.out"]
assert "SUMMARY 133 checks passed, 0 failed" in CKR
lchk("the RESULT's own checker: 133 checks, 0 failed", "check_result_filled.out", "SUMMARY 133 checks passed, 0 failed", "(133 checks, 0 failed; **[I]**, its output in the repo)", ["7.29"])
R2 = SOURCES["R2C_NOTE.md"]
assert "**Unaffected.** A s0 and REP s0 trained and evaluated before 05:56." in R2
lchk("REP s0 trained and was evaluated before 05:56; REP s1 after the r2c relaunch on the branch pinned at 8c24823f", "R2C_NOTE.md", "Unaffected. A s0 and REP s0 trained and evaluated before 05:56.", "REP s0 trained and was evaluated before 05:56", ["7.29"])
RT = SOURCES["REP launch-gate tokens (s0, s1)"]
assert RT.count('"commit": "8c24823f2b1bd24332148ebe434c870bcf3662f8"') == 2
RRC = SOURCES["REP run records (s0, s1)"]
assert re.findall(r'"seed": (\d+),', RRC)[:2] == ["10", "11"]
lchk("both REP launch-gate tokens bind commit 8c24823f; the run records carry trainer seeds 10 and 11", "REP launch-gate tokens; REP run records", "8c24823f x2; seeds 10, 11", "bind commit `8c24823f` (**[M]**", ["7.29"])

md = []
md.append("# NUMBERS_AUDIT: report v33 (v32 + section 7.27 SPEEDSUB, 7.28 SPEEDSUB-T1, 7.29 LAUNCH-INTENT)\n")
md.append("Audited deliverable (sha256): `V8F_SCIENTIFIC_REPORT.md` %s, built on source tip `%s`. The deliverable is the **final** version: section 7.29 carries the LAUNCH-INTENT read (HARMS) and no open slot (section I)."
          "\n" % (hashlib.sha256(open(OUT + "V8F_SCIENTIFIC_REPORT.md", "rb").read()).hexdigest(), TIP[:8]))
md.append("Method (mechanical, `audit_v33.py`): the v33 md was diffed against the v32 md on the tip (`%s`); every added or changed line was tokenised for numbers"
          " (sign kept, the Unicode minus read as `-`, thousands commas removed; a number is searched in its plain form and in its comma-formatted form at once, because the tokenizer drops the commas"
          " on both sides); each distinct signed token was looked up line by line in the source files below. `v33:N` are line numbers in `V8F_SCIENTIFIC_REPORT.md`. Sources are read from the blobs"
          " at the tip, except the one marked as outside the repo.\n" % TIP)
md.append("| source | role |\n|---|---|")
roles = {
    "RESULT_V8F_SPEEDSUB.md": "the SPEEDSUB result, landed `af2dee4a` (primary source for 7.27; its tables are copied line for line)",
    "RESULT_V8F_SPEEDSUB_T1.md": "the SPEEDSUB-T1 result, landed `37d78ed1` (primary source for 7.28; its tables are copied line for line)",
    "PREREG_V8F_SPEEDSUB.md": "the SPEEDSUB design (wording and constants, not results)",
    "PREREG_V8F_SPEEDSUB_T1.md": "the SPEEDSUB-T1 design with Amendment 1 (wording and constants, not results)",
    "PREREG_V8F_LAUNCH_INTENT.md": "the LAUNCH-INTENT PREREG (landed `77234276`; primary source for 7.29: design, redesign disclosure, bars, arms, predicates, outcome table, NEXT-LAUNCH; its tables are copied line for line)",
    "PREREG_V8F_DAG_ORC.md": "the DAG-ORC design and Amendment 3 (the PI's routing); wording and constants",
    "RESULT_V8F_DAG_ORC.md": "the DAG-ORC result (cost lines, crawl rows) for the 7.26 update and the cross-references",
    "GOALS_AND_CLAIMS.md (register rows)": "only the register rows `H-V8F-SPEEDSUB-1`, `H-V8F-SPEEDSUB-T1-1`, `H-V8F-SPEEDSUB-T1-2`, `H-V8F-LAUNCH-INTENT-1`, `H-V8F-DAG-ORC-1`, `D-SPEEDBAND-EGO-ECHO-1` (the whole register is not a source)",
    "SUBREAD.json": "the SPEEDSUB read JSON (`raw/2026-10-28-speedsub-result/read/`)",
    "SPEEDSUB_T1_READ.json": "the SPEEDSUB-T1 read JSON (`raw/2026-10-28-speedsub-t1-read/`)",
    "SPEEDSUB_T1_BESIDE.json": "the logged-lane (bridge) beside JSON of SPEEDSUB-T1",
    "REP_S0_SPEED_BIAS.md": "the Master Mind's diagnosis of REP s0 (sha256 `211601de…`)",
    "FINDING_1_ATTRIBUTION.md": "the attribution of the read-out's skill (sha256 `e89af4a0…`), banked at `842ec2ac`",
    "launch-intent-flag/RESULTS.md": "the lever's build results (tests, mutants, f)",
    "launch-intent-flag/MM_REDESIGN_2026-10-28.md": "the Master Mind's redesign note (bar and pairs), written before any LAUNCH-INTENT number",
    "launch-intent-flag/f_measure.json": "the measured GT-launch share f (49,618 of 306,566)",
    "launch-intent-v2/NOTES.md": "the reader and PREREG v2 notes (what the reader does not build)",
    "launch-intent-v2/enumerate_cells_li.out": "the enumeration of the 524,288 outcome cells",
    "launch-intent-chain/HANDOFF.md": "the box-2 chain handoff (self-test totals, pins)",
    "launch-intent-chain/NOTES.md": "the box-2 chain notes",
    "launch-intent-chain/PINS_FILLED.txt": "the pins of the chain (PREREG, reader, launch commit, trainer)",
    "launch-intent-chain/LAUNCH.txt": "the launch instructions of the chain",
    "DAGORC_READ.json": "the DAG-ORC read JSON (CTRL - P10 gaps, floors, REP - CTRL)",
    "RF_REP.json": "the replicate floors RF of DAG-ORC",
    "P0_RF0.json (rebase)": "the RF0 floors (DAG-0 - ORC-0), re-based file",
    "git log (cited commits)": "commit messages and dates of the commits named in the text",
    "RESULT_V8F_LAUNCH_INTENT.md": "the LAUNCH-INTENT result, landed `8c1e9636` (primary source for the read in 7.29; its tables are copied line for line)",
    "LAUNCH_INTENT_READ.json": "the read of record (reader output, sha256 `e451e1a6…`, banked at `d7dc9f11`): counts, ranges and shares are recomputed from it",
    "MLAUNCH.json": "the M-LAUNCH build (sha256 `f5ebb4b2…`): the four cells, the controls, the training-seed floor and the beside M2 block",
    "MLAUNCH NOTES.md": "the M-LAUNCH build notes (readings chosen, controls, design history, the post-hoc per-tick look)",
    "chain_v8f_launchintent.out": "the banked box-2 chain log (sha256 `ddae7b9e…`, CRLF read as LF): the launch times, the gates, the pulls, the DONE marker (MEASURED)",
    "R2C_NOTE.md": "the Master Mind's incident note on DAG-ORC r2 (repo path `raw/2026-10-28-dag-orc-result/notes/`, sha256 `df02ff2f…`): the REP provenance timing",
    "lit1_logs/chain.out": "the T1-kin chain log of the relaunch (dev box)",
    "lit1_logs/chain.out.prev_1028151251": "the T1-kin chain log of the stopped launch (RAM guard)",
    "lit1_logs/launch_intent_read_REFUSED.out": "the chain's own read with the launch-time reader (REFUSED on K-BUILD)",
    "read_A1/read_A1.out": "the console of the read of record",
    "check_result_filled.out": "the output of the RESULT's own checker (133 checks)",
    "REP launch-gate tokens (s0, s1)": "the launch-gate tokens of the two REP runs (they bind commit `8c24823f`)",
    "REP run records (s0, s1)": "the run records (`summary.json`) of the two REP runs (trainer seeds 10 and 11)",
}
for name in SOURCES:
    md.append(f"| `{name}` | {roles[name]} |")
md.append("\n**Totals.** %d added or changed lines (%d v32 lines replaced or removed); %d distinct signed numeric tokens (unsigned single digits 0-9 excluded as enumerations);"
          " %d found with the same sign in a source line (section A); %d found only with a different or absent sign (section B); %d found only in the v32 report (section C); %d found nowhere (section D)."
          " B, C and D are each reviewed by hand in E.\n" % (len(added), len(removed), len(tok_lines), len(A), len(B), len(C), len(D)))
md.append("⚠ **Limit of class A.** A token in A is a token that occurs somewhere in the source list, in a line the audit cannot tie to the claim; a coincidence cannot be told apart by this test."
          " Sections G (own-section sources), F (claims resolved to lines) and H (recomputation from the banked JSON, assertion on mismatch) are the checks that tie decisive and derived numbers to their source.\n")
md.append("## A. Tokens found with the same sign in a source (token -> v33 lines -> source:lines)\n")
md.append("| token | v33 lines | source lines |\n|---|---|---|")
for t, lns, sd in A:
    forms = sorted(src_forms.get(t, []))
    md.append("| `%s` | %s | %s |" % (t, ",".join(str(x) for x in lns[:5]) + ("..." if len(lns) > 5 else ""), span(sd, 4)))
md.append("\n## B. Found only with a different sign or none (sign is a property of the sentence, reviewed by hand in E)\n")
md.append("| token | v33 lines | unsigned form found at |\n|---|---|---|")
for t, lns, sd in B:
    md.append("| `%s` | %s | %s |" % (t, ",".join(str(x) for x in lns[:5]), span(sd, 3)))
md.append("\n## C. Found only in the v32 report (carried context, not a v33 source claim)\n")
md.append("| token | v33 lines | v32 lines |\n|---|---|---|")
for t, lns, v in C:
    md.append("| `%s` | %s | %s |" % (t, ",".join(str(x) for x in lns[:5]), ",".join(str(x) for x in v[:5])))
md.append("\n## D. Found nowhere (must be structural: section numbers, dates, identifiers, derived values)\n")
md.append("| token | v33 lines |\n|---|---|")
for t, lns in D:
    md.append("| `%s` | %s |" % (t, ",".join(str(x) for x in lns[:5])))
md.append("\n## E. Hand review of sections B, C and D\n")
md.append("| token | review |\n|---|---|")
for t in need:
    md.append("| `%s` | %s |" % (t, REVIEWED[t]))
md.append("\n## F. Claims taken from a PREREG, a note or a log (wording and constants), resolved to source lines\n")
md.append("| claim in the report | source:lines |\n|---|---|")
md.extend(claim_rows)
md.append("\nThe launch facts of 7.29 are MEASURED from the banked box-2 chain log (`chain_v8f_launchintent.out`): the first stamped line 13:54, the refresh, the four gates (launched 13:57 / 13:58 / 14:00 / 14:01,"
          " ZZJOB-END 13:58:34 / 14:00:02 / 14:01:33 / 14:02:37, each PASS), A_0 launched 14:02, the pulls at 14:44 / 15:25 / 16:06 / 16:48 and the DONE marker at 16:48 (section H recomputes the 174 min). Nothing outside the"
          " repository is read by this audit. The REP provenance is `R2C_NOTE.md` (a Master Mind note in the repo, tagged INHERITED in the text) plus the two REP launch-gate tokens, which bind commit `8c24823f` (MEASURED).")
md.append("⚠ **One source inconsistency, disclosed in the report and in `RESULT_V8F_LAUNCH_INTENT.md` section 8:** `PINS_FILLED.txt` in the chain package is stamped about 14:10, while the pins were filled about 13:50,"
          " before commit `d8c9f7bf` (13:51:52) and after the PREREG was promoted at `77234276` (13:49:00). The report quotes the stamp only with that disclosure; section J re-measures the commit times, the two pinned hashes and the log-derived times.\n")
md.append("## G. Section-scoped check (a number in the prose of 7.27 / 7.28 / 7.29 must occur in that section's own sources)\n")
md.append("Section 7.27 is checked against the SPEEDSUB result, PREREG, read JSON and the attribution file (and the T1 files it points forward to); 7.28 against the SPEEDSUB-T1 result, PREREG, read JSON, beside JSON,"
          " the REP s0 diagnosis and the register rows; 7.29 against the LAUNCH-INTENT PREREG, the DAG-ORC PREREG and result and read JSON, the LAUNCH-INTENT result, read JSON, M-LAUNCH JSON and notes, the box-2 chain log, the T1-kin chain logs,"
          " the REP tokens and run records, R2C_NOTE.md, the build, redesign, reader and chain notes and the commit messages."
          " Table lines are verbatim copies and are not re-tested here (section I). Section numbers are structural, and a section-scoped match for them is not evidence. %d tokens are not found in their own section's sources;"
          " each is reviewed:\n" % len(scope_rows))
md.append("| section:token | review |\n|---|---|")
for sec, tk in scope_rows:
    md.append("| `%s:%s` | %s |" % (sec, tk, SCOPE_OK[sec + ":" + tk]))
md.append("")
md.append("## H. JSON cross-check of the decisive and the derived numbers\n")
md.append("`SUBREAD.json`, `SPEEDSUB_T1_READ.json`, `SPEEDSUB_T1_BESIDE.json`, `DAGORC_READ.json`, `RF_REP.json`, `P0_RF0.json`, `f_measure.json`, `LAUNCH_INTENT_READ.json`, `MLAUNCH.json` and the banked box-2 chain log"
          " `chain_v8f_launchintent.out` (with the T1-kin chain log for the twelve steps) are read from the tip. Each row below is recomputed from the JSON or the log by"
          " `audit_v33.py` (an assertion fails the script on a mismatch) and the quoted snippet is found in the named part of v33. Derived values (shares, ratios, counts, sums, the dose, the box-2 span) are computed here, not copied."
          " For 7.29 every table line that carries a number of the read (the predicate table, the 76 cells, the crawl, guard, R-NEW, R-T0 and family tables, the 19-row floors table, the slow-draw table, the M-LAUNCH cells, the run record, the gates, the box-2 and dev-box times)"
          " is rebuilt from the JSON or log and must equal the printed line; %d such lines were rebuilt and matched (the rows marked as tables below summarise them).\n" % LI_TABLE_LINES)
md.append("| what | JSON key path | value in the JSON | snippet in v33 | found in |\n|---|---|---|---|---|")
for what, path, val, snip, where in JSON_ROWS:
    md.append("| %s | `%s` | %s | `%s` | %s |" % (what.replace("|", "/"), path.replace("|", "/"), str(val).replace("|", "/"), snip.replace("|", "/"), where))
md.append("")
jmd = []
jmd.append("## J. Log, commit and hash cross-check of the LAUNCH-INTENT disclosures (re-measured from the repository)\n")
jmd.append("Every row is read from a banked log, a commit, or a blob at the tip and the quoted snippet is found in the named part of v33 (an assertion fails the script on a mismatch). The times are the logs' own stamps and the commits' author dates;"
          " the Master Mind's brief is not a source here, which is why the 15:11 stop time and the 2,048 MB guard value of the T1-kin chain (not in any log) stay tagged INHERITED in the text.\n")
jmd.append("| what | source | value in the source | snippet in v33 | found in |\n|---|---|---|---|---|")
for what, path, val, snip, where in LOG_ROWS:
    jmd.append("| %s | `%s` | %s | `%s` | %s |" % (what.replace("|", "/"), path.replace("|", "/"), str(val).replace("|", "/"), snip.replace("|", "/")[:200], where))
jmd.append("")
forms_diff = []
for t in tok_lines:
    if t in src_forms and len(t.replace("-", "").replace("+", "").split(".")[0]) >= 4:
        forms_diff.append((t, sorted(src_forms[t])))
md.append("## I. Mechanical checks run\n")
md.append("* Every table line inside 7.27 (%d lines) is a line of `RESULT_V8F_SPEEDSUB.md`, inside 7.28 (%d lines) of `RESULT_V8F_SPEEDSUB_T1.md`, and inside 7.29 (%d lines) of `PREREG_V8F_LAUNCH_INTENT.md` or `RESULT_V8F_LAUNCH_INTENT.md`, byte for byte"
          " (`patch_v33.py` extracts each table by an anchored heading and asserts membership in the source)." % (len(T_LINES["7.27"]), len(T_LINES["7.28"]), len(T_LINES["7.29"])))
md.append("* Numbers of four or more integer digits (plain and comma forms are searched together): %d in the added lines, each found in a source in the form(s) listed here: %s." % (
    len(forms_diff), "; ".join("%s -> %s" % (t, "/".join(f)) for t, f in forms_diff[:40]) + (" ..." if len(forms_diff) > 40 else "")))
open_slots = re.findall(r"@@LI_RESULT_([A-Z0-9]+)@@", V33)
md.append("* Open outcome slots in this deliverable (a final version has none): %d: %s." % (len(open_slots), ", ".join(open_slots)))
md.append("* Banned words, closed-loop wording, tier stamps, section order, superset and demotion checks, the offline render and the rebuild: see `v33_checks.txt`.")
md.append("")
md.extend(jmd)
open(AUDIT_PATH, "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
print("audit written:", len(tok_lines), "tokens; A", len(A), "B", len(B), "C", len(C), "D", len(D), "| JSON rows", len(JSON_ROWS), "| claims", len(claim_rows))
