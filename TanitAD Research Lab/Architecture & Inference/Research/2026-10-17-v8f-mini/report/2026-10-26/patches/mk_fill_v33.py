"""Writes FILL.json for patch_v33.py: the LAUNCH-INTENT outcome (HARMS, arm A not adopted; M-LAUNCH NOT MOVED) as the 18 slot texts of section 7.29 and its other sites.
python mk_fill_v33.py [tip [out_file]]      (default tip 8c1e9636388dbb105f6b1853bd95dfdccea9eb93, default out_file FILL.json beside this script)

Sources, all read with `git show <tip>:<path>` from C:/Users/Admin/tanitad-push (never a working tree):
  RESULT_V8F_LAUNCH_INTENT.md          prose and tables; every table line written here is asserted to be a line of this file (copied line for line)
  raw/2026-10-28-launch-intent-read/read_A1/LAUNCH_INTENT_READ.json   the read of record: counts and ranges are computed from it, not typed
  raw/2026-10-28-launch-intent-mlaunch/MLAUNCH.json                    M-LAUNCH and its beside blocks
  raw/2026-10-28-launch-intent-read/box2/chain_v8f_launchintent.out    the box-2 chain log (launch times; none is taken from a note outside the repository)
The texts carry an evidence class, a tier and an estimator for every number; audit_v33.py recomputes the decisive and the derived numbers from the same JSON files and finds them in the built report.
Nothing here decides a word: the word is the reader's."""
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, OrderedDict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"
TIP = sys.argv[1] if len(sys.argv) > 1 else "8c1e9636388dbb105f6b1853bd95dfdccea9eb93"
OUTF = sys.argv[2] if len(sys.argv) > 2 else HERE + "FILL.json"
MIRROR = "C:/Users/Admin/tanitad-push"
R = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
NL = chr(10)


def gitshow(path):
    r = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*", "show", f"{TIP}:{path}"], capture_output=True)
    assert r.returncode == 0 and r.stdout, (path, r.stderr[:200])
    return r.stdout


RES = gitshow(R + "RESULT_V8F_LAUNCH_INTENT.md").decode("utf-8")
assert "\r" not in RES
RES_LINES = RES.split(NL)
RES_SET = set(RES_LINES)
LIR_B = gitshow(R + "raw/2026-10-28-launch-intent-read/read_A1/LAUNCH_INTENT_READ.json")
MLJ_B = gitshow(R + "raw/2026-10-28-launch-intent-mlaunch/MLAUNCH.json")
assert hashlib.sha256(LIR_B).hexdigest().startswith("e451e1a6ba44e1d3") and hashlib.sha256(MLJ_B).hexdigest().startswith("f5ebb4b209a5072e")
LIR = json.loads(LIR_B.decode("utf-8"))
MLJ = json.loads(MLJ_B.decode("utf-8"))
CHAIN = gitshow(R + "raw/2026-10-28-launch-intent-read/box2/chain_v8f_launchintent.out").decode("utf-8").replace(chr(13) + NL, NL)


def ncols(line):
    body = re.sub(r"`[^`\n]*`", "x", line.replace("\\|", "x"))
    return body.count("|") - 1


def tab(header_start, nrows, after=None):  # nrows None: every consecutive table line
    """The table of RESULT_V8F_LAUNCH_INTENT.md whose header line starts with header_start (after the heading `after`), nrows lines, each a line of the RESULT."""
    lo = 0
    if after is not None:
        assert RES.count(after) == 1, after
        lo = RES_LINES.index(after) if after in RES_SET else next(k for k, l in enumerate(RES_LINES) if l.startswith(after))
    k = next(k for k in range(lo, len(RES_LINES)) if RES_LINES[k].startswith(header_start))
    if nrows is None:
        nrows = 0
        while k + nrows < len(RES_LINES) and RES_LINES[k + nrows].startswith("|"):
            nrows += 1
    rows = RES_LINES[k:k + nrows]
    assert len(rows) == nrows and all(r.startswith("|") for r in rows), (header_start, len(rows))
    assert not (k + nrows < len(RES_LINES) and RES_LINES[k + nrows].startswith("|")), ("the table is longer than asked", header_start)
    assert all(r in RES_SET for r in rows)
    assert len({ncols(r) for r in rows}) == 1, (header_start, {ncols(r) for r in rows})
    return NL.join(rows)


def tab_sub(header_start, keep, nrows_all):
    """The provenance table of RESULT section 10 reduced to the header, the separator and the rows keep(cells) accepts: every line is still a line of the RESULT."""
    full = tab(header_start, nrows_all).split(NL)
    out = full[:2] + [r for r in full[2:] if keep([c.strip() for c in r.strip("|").split("|")])]
    assert len({ncols(r) for r in out}) == 1
    return NL.join(out)


# ---- wrapping: hard-wrap prose lines at ~190 columns without splitting a code span or an interval; table blocks are left alone --------------------------------------------
def wrap(par, indent, first_indent=None, width=186):
    first_indent = indent if first_indent is None else first_indent
    atom = re.sub(r"`[^`\n]*`", lambda m: m.group(0).replace(" ", chr(1)), par)
    atom = re.sub(r"\[[+\u2212-]?\d[^\[\]\n]{0,40}\]", lambda m: m.group(0).replace(" ", chr(1)), atom)
    atom = re.sub(r"\*\"[^\"\n]{0,60}\"\*", lambda m: m.group(0).replace(" ", chr(1)), atom)
    atom = re.sub(r"(?<=\d) %", chr(1) + "%", atom)
    words, lines, cur, started = atom.split(" "), [], first_indent, False
    for w in words:
        if not started:
            cur, started = cur + w, True
        elif len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = indent + w
        else:
            cur = cur + " " + w
    lines.append(cur)
    out = [l.replace(chr(1), " ") for l in lines]
    for l in out[1:]:
        assert not re.match(r"^\s*([*+\-]|\d+[.)])\s", l) and not l.lstrip().startswith(("|", "#", ">")), ("a wrapped line would open a block", l[:60])
    return NL.join(out)


def bullet(text, lead="* ", indent="  "):
    return wrap(text, indent, first_indent=lead)


def sub(text):
    return wrap(text, "    ", first_indent="  * ")


def para(text):
    return wrap(text, "  ", first_indent="  ")


def block(*parts):
    """A fill text: paragraphs and tables joined; a table is a block at column 0 with a blank line before and after."""
    out = []
    for p in parts:
        if isinstance(p, tuple):  # ("T", table text)
            out.append("")
            out.append(p[1])
            out.append("")
        else:
            out.append(p)
    return NL.join(out).replace(NL * 3, NL * 2).strip(NL)


T = lambda t: ("T", t)


def sg(x, nd=4):
    return ("+" if x >= 0 else "\u2212") + ("%." + str(nd) + "f") % abs(x)


def cis(c):
    return "%s [%s, %s]" % (sg(c["delta"]), sg(c["lo"]), sg(c["hi"]))


# ---- numbers computed from the JSON ------------------------------------------------------------------------------------------------------------------------------------------
ra, fl = LIR["read_a"], LIR["floors"]
ROWS = list(ra["A_CTRL"]["s0"].keys())
assert len(ROWS) == 19
BAR = fl["bar"]
CELLS = []
for pair, key in (("main", "A_CTRL"), ("rep", "Arep_REP")):
    for sd in ("s0", "s1"):
        for r in ROWS:
            CELLS.append((pair, sd, r, ra[key][sd][r]))
assert len(CELLS) == 76
cb = sum(1 for *_, c in CELLS if c["separated_better"])
cw = sum(1 for *_, c in CELLS if c["separated_worse"])
cn = 76 - cb - cw
assert (cb, cw, cn) == (16, 7, 53)
grp = {}
for pair in ("main", "rep"):
    for sd in ("s0", "s1"):
        g = [c for p, s_, r, c in CELLS if p == pair and s_ == sd]
        grp[(pair, sd)] = (sum(1 for c in g if c["separated_better"]), sum(1 for c in g if c["separated_worse"]), sum(1 for c in g if not c["separated"]))


def imp(c):
    return -c["delta"] if c["direction"] == "lower" else c["delta"]


reach_imp = [(p, s_, r, c) for p, s_, r, c in CELLS if imp(c) >= BAR[r]]
reach_wor = [(p, s_, r, c) for p, s_, r, c in CELLS if -imp(c) >= BAR[r]]
assert len(reach_imp) == 3 and sum(1 for *_, c in reach_imp if c["separated_better"]) == 1
assert len(reach_wor) == 6 and sum(1 for *_, c in reach_wor if c["separated_worse"]) == 5
RFB = {r: fl["RF"][r] / BAR[r] for r in ROWS}
rf_over = sum(1 for v in RFB.values() if v > 1)
assert rf_over == 13 and min(RFB, key=RFB.get) == "DK-VIOL" and max(RFB, key=RFB.get) == "yaw-rate MAE"
CRAWL = ["yaw-rate MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share"]
shares = []
for pair, ykey, xkey in (("main", "CTRL_P10", "A_P10"), ("rep", "REP_P10", "Arep_P10")):
    for sd in ("s0", "s1"):
        for r in CRAWL:
            y, x = ra[ykey][sd][r]["delta"], ra[xkey][sd][r]["delta"]
            shares.append((pair, 100 * (y - x) / y))
assert sum(1 for _, v in shares if v > 0) == 9 and sum(1 for _, v in shares if v < 0) == 7
sh_all = (min(v for _, v in shares), max(v for _, v in shares))
sh_main = (min(v for p, v in shares if p == "main"), max(v for p, v in shares if p == "main"))
assert ("%.1f" % sh_all[0], "%.1f" % sh_all[1]) == ("-30.2", "28.7") and ("%.1f" % sh_main[0], "%.1f" % sh_main[1]) == ("-3.0", "28.7")
GROUPS = Counter(k.split(":")[0] for k in LIR["controls_PASS"])
assert sum(GROUPS.values()) == 206 and all(LIR["controls_PASS"].values())
GROUPS_TXT = ", ".join("%s %d" % (g, n) for g, n in GROUPS.items())
M = MLJ["primary"]["cells"]
MCELLS = [M[k]["x_minus_y"] for k in ("main_s0", "main_s1", "replicate_s0", "replicate_s1")]
assert [m["separated"] for m in MCELLS] == [False] * 4 and MLJ["M_LAUNCH"] == "not moved"
ratio_floor = abs(MLJ["beside_training_seed_floor"]["REP_minus_CTRL_s0"]["x_minus_y"]["delta"]) / max(abs(m["delta"]) for m in MCELLS)
assert round(ratio_floor) == 16
M2 = MLJ["beside_M2"]["pairs"]
M2SEP = []
for pk, pv in M2.items():
    for rk, rv in pv["rows"].items():
        if rv["x_minus_y"]["separated"]:
            M2SEP.append((pk, rk, rv["x_minus_y"]))
assert len(M2SEP) == 4
KS = {"A_0": 0.331451, "A_1": 0.330729, "A_rep,0": 0.325586, "A_rep,1": 0.328093}

# ---- the tables, copied line for line ----------------------------------------------------------------------------------------------------------------------------------------
T_OUT = tab("| predicate (PREREG section 10)", 12)
T_FLOOR = tab("| row | better | F | RF0 |", 21)
T_ADE = tab("| seed | pair | row | X \u2212 Y | mean X |", 6)
T_MOVE = tab("| seed | pair | crawl row | X \u2212 Y | improvement | bar |", 18)
T_CRAWL = tab("| seed | pair | crawl row | X \u2212 P10 | Y \u2212 P10 |", 18)
T_NEW = tab("| seed | pair | control Y | rows in D |", 6)
T_T0 = tab("| seed | pair | T0 own-code ADE6, X \u2212 Y", 6)
T_CENSUS = tab("| family (rows) |", 6)
T_LONG = tab("| row | better | A \u2212 CTRL s0 |", 8, after="### 3.2 LONGITUDINAL (read first: the lever acts on along-track pressure)")
T_LAT = tab("| row | better | A \u2212 CTRL s0 |", 8, after="### 3.3 LATERAL")
T_TAC = tab("| row | better | A \u2212 CTRL s0 |", 4, after="### 3.4 TACTICAL")
T_STR = tab("| row | better | A \u2212 CTRL s0 |", 5, after="### 3.5 STRATEGIC")
T_T0FAM = tab("| family | what the JSON names |", 6)
T_P10_S0 = tab("| row | better | A \u2212 P10 s0 |", 21)
T_P10_S1 = tab("| row | better | A \u2212 P10 s1 |", 21)
T_SLOW = tab("| arm and pair (T0, m/s) |", 6)
T_SHARE = tab("| run | trainer seed | launch rows / rows |", 6)
T_GATE = tab("| run | W | trainer seed | probe batch launch rows |", 6)
T_FACT1 = tab("| seed | trainer seed (main pair / replicate pair) |", 4)
T_FACT2 = tab("| seed | pair | crawl row | X \u2212 Y | improvement / bar |", 4)
T_MLAUNCH = tab("| pair | seed | X \u2212 Y [95 % CI] | separated |", 6)
T_BOX2 = tab("| run | train: launch minute |", 6)
T_T1STEPS = tab("| run | causal rollout (s) |", 6)
T_PROV = tab_sub("| role | repo path", lambda c: not c[0].startswith("run record"), None)

# ---- the launch facts, found in the banked chain log ---------------------------------------------------------------------------------------------------------------------------
for needle in ("10-28 13:54 waiting for the DAG-ORC r2 chain", "10-28 13:54 predecessor verdict: GO the DAG-ORC r2 chain ended DONE", "===== ZZJOB-END 28.10.2026 13:54:55,82 rc=0",
               "10-28 13:55 box 2 code = tip d8c9f7bfc91798b6fafb5a13dd793771e2f79f6a", "===== ZZJOB-END 28.10.2026 13:58:34,48 rc=0", "===== ZZJOB-END 28.10.2026 14:00:02,89 rc=0",
               "===== ZZJOB-END 28.10.2026 14:01:33,52 rc=0", "===== ZZJOB-END 28.10.2026 14:02:37,90 rc=0", "10-28 14:02 launched pid=23736 rc=0 log=D:" + chr(92) + "tanitad" + chr(92) + "logs" + chr(92) + "v8flaunchintent10281354_t_M2b_s0_launchintent.log",
               "10-28 14:44 M2b_s0_launchintent pulled to", "10-28 15:25 M2b_s1_launchintent pulled to", "10-28 16:06 M2b_s0_launchintentrep pulled to", "10-28 16:48 M2b_s1_launchintentrep pulled to",
               "ZZV8FMINI-DONE 10-28 16:48 tip d8c9f7bfc91798b6fafb5a13dd793771e2f79f6a", "B is not queued"):
    assert needle in CHAIN, needle
assert CHAIN.count("LAUNCH-INTENT gate: PASS") == 8 and CHAIN.count("LAUNCH-INTENT run record: PASS") == 4 and CHAIN.count("LAUNCH-INTENT K-SHARE: PASS") == 4

# ======================================================================================================================================================================================
F = OrderedDict()
F["LINE"] = "the LAUNCH-INTENT read (\u00a77.29, HARMS: arm A, W 1.58926, is not adopted; M-LAUNCH reads NOT MOVED)."
F["TITLE729"] = "HARMS (arm A, W 1.58926, not adopted; M-LAUNCH NOT MOVED)"

F["ABSTRACT"] = NL.join([
    bullet("**LAUNCH-INTENT reads HARMS: arm A is not adopted (\u00a77.29).** O's launch-sample along-track weight (GT-launch TRAIN samples hold one third of the along-track weight on knots 1\u201312, W = 1.58926; a label-side "
           "subset, nothing new at inference) was trained in four box-2 runs (A and A_rep at two seeds; 174 min of box 2, **[M]**, the chain log) and read at T1-kin (paired episode-cluster bootstrap, B = 2,000, 141 episodes, per seed "
           "and per pair; all 206 reader controls PASS). R-MOVE is false at both seeds: 2 of the 16 crawl cells are separated-better and 1 clears its bar (L2 curvature, replicate pair, seed 0: "
           "\u22120.0215 [\u22120.0423, \u22120.0048] against 0.0136), no crawl row moves by its bar in both pairs at either seed, and all 16 X \u2212 P10 crawl cells stay separated-worse. R-ADE fails at both seeds, each in one pair "
           "(ADE6 A_rep \u2212 REP s0 +0.0431 [+0.0137, +0.0711] m; A \u2212 DAG-OACC s1 +0.0483 [+0.0105, +0.0884] m), while the other pair of each seed is separated the other way (\u22120.0294 [\u22120.0522, \u22120.0077] and "
           "\u22120.0341 [\u22120.0655, \u22120.0027] m): the separated ADE6 cells flip sign between the two same-seed pairs, which is what the pair design is for (`H-ESTIM-SEED-1`). R-TLAT, R-TLON, R-T0 and R-NEW hold. "
           "**M-LAUNCH reads NOT MOVED** (T1-kin, paired episode-cluster bootstrap, B = 2,000, over 49 episodes / 327 launch windows): X \u2212 Y +0.0348 [\u22120.0299, +0.0923] and +0.0282 [\u22120.0247, +0.0836] m/s (main pair, s0 and s1), "
           "+0.0243 [\u22120.0281, +0.0846] and \u22120.0146 [\u22120.0607, +0.0286] m/s (replicate pair), none separated, so `PREREG_V8F_LAUNCH_INTENT` \u00a711 routes to LAUNCH-INTENT-HI. "
           "The T0 four families are NOT RUN, the anti-echo block is NOT BUILT and the T-head F1 is NOT COMPUTABLE at T1-kin."),
    bullet("**What is next.** The open gap is the crawl set (yaw-rate, curvature, L2 and M1 against P10 for DAG-OACC); the adopted base stays DAG-OACC. The committed next after HARMS is LAUNCH-INTENT's arm B (W 0.726175, four box-2 runs), "
           "registered with its own PI go. LAUNCH-INTENT-HI (W 4.17852, four box-2 runs) is the \u00a711 branch that M-LAUNCH selects, also with its own PI go. The PI is being asked which, if either, box 2 runs. "
           "The history-aware causal read-out stays registered and unrun (\u00a77.28), and the speed line that SPEEDREAD (`PREREG_V8F_SPEEDREAD`) asked about routes to perception (`PREREG_V8F_LORA` on the A40), which waits on the PI."),
])

F["STATUS729"] = ("Read 2026-10-28 by the pre-registered reader of record `launch_intent_read.py` (sha256 `fb3d2d7d\u2026`, the Amendment 1 reader): outcome **HARMS** (arm A, W 1.58926, not adopted) with all **206 reader controls PASS** "
                  "(**[M]**, `LAUNCH_INTENT_READ.json`, sha256 `e451e1a6\u2026`, banked at `d7dc9f11`). `RESULT_V8F_LAUNCH_INTENT.md` landed at `8c1e9636` and carries the M-LAUNCH read, **NOT MOVED** (`MLAUNCH.json`, sha256 `f5ebb4b2\u2026`). "
                  "The design, the Master Mind's redesign of the bar and arms, the outcome table and the registered routes below were committed before any LAUNCH-INTENT row existed. The bullets after them are the read, with the "
                  "RESULT's tables copied line for line. Arm B and arm HI are not trained and not read.")

F["OUTCOME"] = block(
    bullet("**Outcome: HARMS, as registered [M].** Arm A (W 1.58926) is not adopted. It failed its pre-registered bar: the decisive R-MOVE is false at both seeds, and the guard R-ADE fails at both seeds, each in one pair. "
           "Nothing below softens that. Tier: T1-kin, a self-action open loop (m = 10, causal bucket, dead-reckoned nav, logged scene, periodic, executor plan; plain arms), not T2. The R-T0 points are T0 (teacher-forced own-code, "
           "eval-141, ages cycle: box 2's standard eval), a world-model diagnostic and not driving performance. Estimator: `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0, 141 episodes / 2,399 windows, "
           "per seed and per pair, paired, on the same windows and matched cells. It answers *\"would another draw of EPISODES say this?\"*; the question *\"would another training run say this?\"* is answered by the replicate pair "
           "at each seed (`H-ESTIM-SEED-1`), and the planner is deterministic, so no inference-seed question arises. Intervals are 95 % (**[I]**, the convention of `RESULT_V8F_DAG_ORC.md`; the JSON stores the bounds only; a T0 point carries "
           "no interval). X is A (main pair) or A_rep (replicate pair); Y is the pair's control, CTRL = DAG-OACC or REP. The registered table decides the word (`rows.A.<seed>.pairs.<pair>` and `rows.A.<seed>` in the JSON):"),
    T(T_OUT),
    para("The table's condition, verbatim from the JSON: *\"controls pass; a guard (R-TLAT, R-TLON, R-ADE, R-T0, R-NEW) fails at \u2265 1 seed\"*. A guard fails at a seed when it fails in either pair, so R-ADE fails at s0 and at s1; "
         "REFUSED is decided first and HARMS second, and the controls pass, so the R-MOVE and R-CRAWL flags do not change the word. The word is the reader's output (`matching_rows_in_table_order`: one row, HARMS). "
         "The committed next, verbatim from the JSON: *\"not adopted; read B under this table (train B first, with its own PI go).\"* The cascade reads `final` HARMS and `b_status` *\"PENDING: train B (4 runs, share 1/4) through "
         "its gate, with its own PI go, and read it under this same table\"*. \u26a0 Reported as written, beside the word: R-MOVE is false at both seeds and R-CRAWL is false at both seeds; the registered guard failure is R-ADE; "
         "R-TLAT, R-TLON, R-T0 and R-NEW hold, and R-KEEP and R-NOBASE hold."),
)

F["CONTROLS"] = block(
    bullet("**Controls: all 206 reader controls PASS [M].** The reader of record writes its JSON only when every control passes (the T1-kin chain's own read with the launch-time reader REFUSED on one control and wrote nothing, "
           "see Amendment 1 below). Wall 341.6 s. By group (`controls_PASS` in `LAUNCH_INTENT_READ.json`, every value true; names and counts): " + GROUPS_TXT + ". "
           "They include the committed K-GATE, K-PAIR, K-INIT, K-SHARE, K-MASK, K-BUILD, K-P0, K-ZERO (every X \u2212 X reads exactly 0.0) and K-REP-LICENCE (both seeds) of \u00a79 of the PREREG."),
    bullet("**Amendment 1 (`A1-2026-10-28`) was appended before any row was read, and K-BUILD is read by its registered rule.** The PREREG's K-BUILD row states a rule (against `aeb54468`, the commit DAG-OACC trained at, each of the three "
           "files is byte-identical, or every changed hunk sits behind a default-off flag whose bit-identity test passes) and then adds a fact, that `v8f_mini_data.py` is byte-identical. That fact had been measured against `d7208328`, not "
           "against `aeb54468`, and the reader coded it as a rule, so it refused the launch commit on the data module. The amendment pins the data module to DAG-ORC r2's build (sha256 `c117e1cf\u2026`) and lists its twelve flag-gated "
           "hunks with a PASSing bit-identity test. The reader changed in that clause alone; no bar, row, outcome, predicate or other control changed. What had been seen before it: controls only, namely the T1-kin chain's dry run "
           "(where K-BUILD REFUSES), A_0's run record and K-SHARE (both PASS) and the four launch-gate lines; the statement, and its time (~15:10), are the Amendment's own and the Master Mind's (**[I]**, RESULT \u00a78.1; a log cannot prove "
           "the negative that no metric line was read). It landed at `e28516e8` (15:24:46), before the read of record started (first console stamp 17:02:21); A_0's three T1-kin steps ran from 15:13:04 to 15:22:14, between the writing and "
           "the landing (**[I]**, RESULT \u00a78.1). The launch-gate tokens bind the PREREG as promoted (sha256 `469bac1f\u2026`) and the launch-time reader (sha256 `5e950dcf\u2026`); the reader of record (sha256 `fb3d2d7d\u2026`) ran with the "
           "amended PREREG (sha256 `6ae79f2f\u2026`), and both readers are on record. The T1-kin chain's own read with the launch-time reader REFUSED at 17:01:36 on K-BUILD alone and wrote nothing (**[M]**, the T1-kin chain log; banked as "
           "`lit1_logs/launch_intent_read_REFUSED.out`); the read of record ran the amended reader on the same manifest (sha256 `98d2cd3d\u2026`). The replicate pair is unaffected (A_rep and REP share the data module); the main pair "
           "carries DAG-ORC's flag-gated data hunks on the A side, as DAG-ORC's own main pair did."),
)

F["RUNS"] = block(
    bullet("**The four runs [M]** (box 2, launch commit `d8c9f7bf`, 13:51:52; the chain log `chain_v8f_launchintent.out`, sha256 `ddae7b9e\u2026`). Four launch gates PASS, four run records PASS, four K-SHARE PASS. "
           "A_s is DAG-OACC_s's training argv verbatim plus `--launch-intent-w` 1.58926 (one third of the along-track weight on the launch samples). A_rep,s is REP_s's argv (trainer seed 10 / 11) plus the same flag. CTRL_s = DAG-OACC_s "
           "and REP_s are banked, so no control was retrained. Checkpoint sha256 prefixes: A `deb8f28b\u2026` (s0) / `abe0fd24\u2026` (s1); A_rep `b86f7d43\u2026` / `2900ff73\u2026`. **Arm B (W 0.726175) was not trained and is not read.** "
           "The lever reached the data. The realised weight share is recomputed by the run record from `launch_n`, `launch_eligible` and the flag (K-SHARE: 1/3 \u00b1 0.02; K-MASK: launch-row fraction 0.1619 \u00b1 0.02); rows equal eligible "
           "rows in all four runs:"),
    T(T_SHARE),
    para("The four weight shares (0.3315, 0.3307, 0.3256, 0.3281 at four decimals) equal the PREREG's ESTIMATED shares for trainer seeds 0 / 1 / 10 / 11, simulated from the sampler stream. Each run went through its own launch gate "
         "before training (G-DVB, the swap probe, the acc probe, the launch probe, and the one-lever check against its pair control's banked argv); `head_o grad diff` is the gradient the launch term puts on O's head, non-zero in all four:"),
    T(T_GATE),
    para("Box-2 run times, from the chain log (the minute stamp of the `launched` line for each job, and the job's own ZZJOB-END stamp; training, then the standard T0 eval, then the pull and the run record):"),
    T(T_BOX2),
    para("Box-2 time is 13:54 to 16:48, **174 min or 2.9 h** (**[M]**, first stamped line to the DONE marker), against the PREREG's ESTIMATED \u2248 2.8 h. The dev-box T1-kin chain (CPU, one process at a time) ran twelve steps: the causal "
         "rollouts are rows, and the logged rollouts and fresh-T0 evals are controls (K2-logged, K4). Step wall times (s):"),
    T(T_T1STEPS),
    para("The twelve steps sum to 1756 s (**[M]**, arithmetic on the table), 29.3 min of dev-box CPU. The chain's own read ran 16:55:55 to 17:01:36 and REFUSED (Amendment 1 above). "
         "**One stop.** The T1-kin chain was stopped once, on its own RAM guard, at 15:11, still waiting on the guard and before any rollout: the stopped launch's last line, 15:07:07, reads *\"LOW RAM available 3040 MB < 4096 MB\"*, "
         "and its log holds no `start` line for a rollout (**[M]**, `lit1_logs/chain.out.prev_1028151251`). It was relaunched with a 2,048 MB guard. The 15:11 stop time and the 2,048 MB value are from the Master Mind's brief (**[I]**); "
         "the relaunch's log does not echo the guard value. The relaunch started at 15:12:51 and its first rollout at 15:13:04, all eight rollouts in its log show attempt 1, and the template proofs (rollout argv equal to the banked argv, "
         "token for token) pass in both launches (**[M]**, `lit1_logs/chain.out`). The guard gates dev-box memory; it is not a measurement gate."),
)

F["FLOORS"] = block(
    bullet("**Bars and floors [M]** (`floors` in the JSON; T1-kin). The bars read are the committed ones: `bar_r = 2 \u00d7 max(F_r, RF0_r)`, one value per row, applied at both seeds and in both pairs, inclusive. F is 0.1207 m on ADE6 and 0 on "
           "every other row; RF0 is the frozen same-seed cross-build floor (`P0_RF0.json`, sha256 `1608d70b\u2026`); RF, the larger of the two REP \u2212 DAG-OACC draws, is beside every row and is not a bar term. The F, RF0, bar and RF columns "
           "equal the 19-row table committed above, cell by cell (checked by the audit). RF is the training-run spread measured by REP \u2212 DAG-OACC, which the replicate pair carries per pair; RF \u00f7 bar runs from 0.17 (DK-VIOL) to 14.32 (yaw-rate MAE) and exceeds 1 on "
           "13 of the 19 rows. RF decides nothing here:"),
    T(T_FLOOR),
    para("The main and the replicate pair contrasts (X \u2212 Y) for all 19 rows are in the bullets below: ADE6 in the guards bullet, the four crawl rows in the crawl bullet, the 17 family rows in the families bullet. "
         "Counted over those 76 cells (19 rows \u00d7 4 pair-seed cells; **[M]**, from `read_a.A_CTRL` and `read_a.Arep_REP`): **%d are separated-better, %d separated-worse and %d not separated.** Per pair and seed (better / worse / not "
         "separated): main s0 %d / %d / %d, replicate s0 %d / %d / %d, main s1 %d / %d / %d, replicate s1 %d / %d / %d. Three cells reach the bar in the improving direction, and one of them is separated: DK-TG in the main pair at s1 "
         "(\u22120.0094) and in the replicate pair at s0 (\u22120.0245), neither separated, and L2 curvature in the replicate pair at s0 (separated). Six cells reach it in the worsening direction and five of those are separated-worse: speed MAE, "
         "accel MAE, cross-track MAE and L1 lateral-accel MAE in the main pair at s1, and along-track MAE in the replicate pair at s0 (L2 curvature in the replicate pair at s1 reaches it worsening, not separated)."
         % ((cb, cw, cn) + grp[("main", "s0")] + grp[("rep", "s0")] + grp[("main", "s1")] + grp[("rep", "s1")])),
)

F["CRAWL"] = block(
    bullet("**The crawl set (R-MOVE, R-CRAWL) [M]** (T1-kin). R-MOVE needs, in each pair, every crawl row (yaw-rate MAE, curvature MAE, L2 curvature MAE at GT \u2265 5.2, M1 crawl share) separated-better with an improvement of at least its bar; "
           "improvement is \u2212delta. **It is false at both seeds.** 2 of the 16 cells are separated-better and 1 clears its bar: L2 curvature in the replicate pair at s0, \u22120.0215 [\u22120.0423, \u22120.0048] against a bar of 0.0136 (158.1 % of it). "
           "Its main-pair partner reads \u22120.0075 [\u22120.0212, +0.0025], not separated, 55.1 % of the bar. No pair clears all four rows, and no main-pair cell clears. Cells read point [95 % CI] and a flag: B = separated-better, W = separated-worse, "
           "\u2013 = not separated:"),
    T(T_MOVE),
    para("R-CRAWL needs X \u2212 P10 not separated-worse on every crawl row (P10 is the kinematic floor). **It is false at both seeds**: all 16 X \u2212 P10 cells (4 rows \u00d7 2 pairs \u00d7 2 seeds) are separated-worse, and so are the Y \u2212 P10 "
         "cells in every pair, which is how the crawl set C was fixed. The share of the Y \u2212 P10 gap that the lever closes, (Y \u2212 P10 \u2212 (X \u2212 P10)) / (Y \u2212 P10) from the point deltas (**[M]**, arithmetic on the banked JSON), is "
         "positive in 9 of the 16 cells and negative in 7 (the gap widens), from %.1f %% to +%.1f %%; in the main pair it lies between %.1f %% and +%.1f %%:" % (sh_all[0], sh_all[1], sh_main[0], sh_main[1])).replace("-30.2", "\u221230.2").replace("-3.0", "\u22123.0"),
    T(T_CRAWL),
    para("A negative share means the gap to P10 widens."),
)

F["GUARDS"] = block(
    bullet("**Guards and adoption rows [M]** (T1-kin unless a heading says T0). R-TLAT, R-TLON, R-ADE, R-T0 and R-NEW are guards, and each fails at a seed when it fails in either pair; R-KEEP and R-NOBASE decide adoption. "
           "R-ADE (T1-kin ADE6, X \u2212 Y not separated-worse) fails in the replicate pair at s0 and in the main pair at s1. R-KEEP reads the point against +0.0604 m and holds in all four pairs; R-ADE and R-KEEP read one contrast, "
           "as the PREREG discloses:"),
    T(T_ADE),
    para("R-NEW fires on a row of D when three things hold: (1) X \u2212 P10 is separated-worse, (2) Y \u2212 P10 is not (the row is in D, which has no exempt rows), and (3) X \u2212 Y is separated-worse with a worsening of at least the bar. "
         "R-NOBASE reads (1) and (3). **Both hold because condition (1) fails, not because condition (3) does:**"),
    T(T_NEW),
    para("In the main pair at s1, four rows meet (3): speed MAE +0.1797 [+0.0998, +0.2726], accel MAE +0.7391 [+0.4366, +1.0952], cross-track MAE +0.0720 [+0.0403, +0.1109] and L1 lateral-accel MAE +1.4444 [+0.6003, +2.4872], all separated-worse. "
         "None of the four has X \u2212 P10 separated-worse. In the replicate pair at s0, along-track MAE meets (3), +0.0493 [+0.0170, +0.0801], and its X \u2212 P10 is not separated-worse either. The registered predicate is applied as written. "
         "R-T0 is at tier T0 (a point; the JSON stores no interval), and R-KEEP is T1-kin:"),
    T(T_T0),
    para("R-TLAT and R-TLON are booleans in the JSON and read *held* in all four pairs. The T-head macro-F1 values behind them are not carried in the JSON (UNVERIFIED as numbers), and the T-head F1 is NOT COMPUTABLE at T1-kin: the paired tool "
         "reads eval dumps, not tick dumps."),
)

F["FAMILIES"] = block(
    bullet("**The four families [M]** (T1-kin; per family, per seed, per pair, never pooled; STRATEGIC is not paused for v8F, PI ruling 2026-10-17). Each family block of the JSON carries the tier and estimator strings, and its leaves equal "
           "the `read_a` cells cell for cell (204 cells; **[I]**, asserted by the RESULT's `check_result.py`). Each table shows X \u2212 Y, the one-lever contrast of the pair; paired episode-cluster bootstrap, B = 2,000, 141 episodes. "
           "Separated cells per family, per seed, per pair (B / W / not separated; X \u2212 Y):"),
    T(T_CENSUS),
    para("**LONGITUDINAL** (read first: the lever acts on along-track pressure; DK-VIOL / DK-HW / DK-TG are on the lead windows, n = windows / episodes):"),
    T(T_LONG),
    para("Not built for this family: M-LAUNCH inside the reader (built separately, below), the anti-echo block and the M2 / L2 launch-subset rows. **LATERAL:**"),
    T(T_LAT),
    para("**TACTICAL** (TAC-LAT and TAC-LON are the plan-derived agreements, higher is better; the guards R-TLAT and R-TLON are in the guards bullet):"),
    T(T_TAC),
    para("The T1-kin goal-head F1 and head accuracy are **NOT COMPUTABLE** here: the paired tool reads eval dumps, not tick dumps (JSON note). **STRATEGIC** (S-POS@30 / 50 / 80, lower is better; the X \u2212 Y cell is the difference of the two "
         "runs' (S \u2212 CTRV, ticks \u2265 1) contrasts on the same windows):"),
    T(T_STR),
    para("Goal and route-setting quality is not a Read-A row; whether another tool computes it for these arms is UNVERIFIED here. **The T0 four families are NOT RUN** (box 2's standard eval):"),
    T(T_T0FAM),
    para("`not_built.t0_four_families` reads: *\"orc_families.py pointed at the arms and their controls (the T0 LONGITUDINAL / LATERAL / TACTICAL rows and distance keeping at T0): NOT RUN, a build item, never a silent omission\"*. "
         "The only T0 numbers in this read are the R-T0 points and the slow-draw cells; the inputs exist (each run's box-2 T0 eval JSON is in the run record, and its `.npz` is in the pulled run directory on E:). "
         "**Nothing is claimed for T0 LONGITUDINAL, LATERAL, TACTICAL or STRATEGIC.** Registered rows that are NOT BUILT or NOT COMPUTABLE, verbatim from the JSON `not_built` (the reader of record): `M-LAUNCH`: *\"X - Y of the mean (v_own - v_GT) "
         "at ticks 1-2 in the GT-launch windows, from the tick dumps, in each pair: NOT BUILT (routes NEXT-LAUNCH by hand)\"*; `anti_echo_block`: *\"longitudinal_claim_admissible, hold-v0, copy detector (RESULT_V8F_ORECOVER 3.1): NOT BUILT\"*; "
         "`M2_and_L2_launch_subset`: *\"along-track MAE, speed MAE and L2 on the GT-launch subset; M2: NOT BUILT\"*. By family: LONGITUDINAL carries all three; TACTICAL has the T-head F1 NOT COMPUTABLE and its R-TLAT / R-TLON macro-F1 values are not in "
         "the JSON; LATERAL and STRATEGIC have every registered Read-A row computed; all four families have the T0 rows NOT RUN. The JSON field `next_launch` is null (M-LAUNCH is not built in the reader; it is read separately, below). "
         "X \u2212 P10 and Y \u2212 P10 for every row (T1-kin). For the three S-POS rows the P10 contrast is the run's own S \u2212 CTRV, labelled in the JSON *\"ticks>=1, unpaired vs CTRV (DAG_READ.json)\"*: those cells are not paired across runs. **s0:**"),
    T(T_P10_S0),
    para("**s1:**"),
    T(T_P10_S1),
)

M2TXT = []
for pk, rk, x in M2SEP:
    nm = {"along_mae_m": "along-track MAE", "speed_mae_mps": "speed MAE", "L2_curvature_mae_fast_1pm": "L2 curvature MAE"}[rk]
    pn = {"main_s0": "main pair s0", "main_s1": "main pair s1", "replicate_s0": "replicate pair s0", "replicate_s1": "replicate pair s1"}[pk]
    good = x["delta"] < 0
    M2TXT.append("%s, %s %s (%s)" % (nm, pn, cis(x), "better" if good else "worse"))
F["MLAUNCH"] = block(
    bullet("**M-LAUNCH and the slow-draw flag [M].** The slow-draw flag (PREREG \u00a77; T0; the five speed cells; per-window paired difference, episode-cluster bootstrap, B = 2,000, seed 0) labels an arm a slow-T draw when all five cells "
           "are separated from the pair's control and each shift exceeds 0.5 m/s in magnitude. It decides nothing. **No arm is labelled**: the largest shift in magnitude over the 20 cells is 0.1627 m/s, and 13 of the 20 cell intervals exclude 0:"),
    T(T_SLOW),
    para("**M-LAUNCH reads NOT MOVED** (T1-kin, causal lane; paired episode-cluster bootstrap, B = 2,000, seed 0, over the 49 episodes of the 327 GT-launch windows, the `hmc_read.py` definition; `MLAUNCH.json`, sha256 `f5ebb4b2\u2026`, "
         "banked in `raw/2026-10-28-launch-intent-mlaunch/` at `8c1e9636`). The cell is the mean (v_own \u2212 v_GT) at ticks 1\u20132, X \u2212 Y, in m/s; a negative value is slower than the human:"),
    T(T_MLAUNCH),
    para("Section 11 label: **not moved** (moved needs all four cells separated toward 0). The route is **LAUNCH-INTENT-HI** (W_HI 4.17852, four runs, its own PI go). The controls of the M-LAUNCH build read their known values (all nine groups PASS): "
         "X = Y gives exactly 0.0; tick 0 gives exactly 0 in all four pairs; shuffled GT, the complement mask and a permuted Y move the number; a \u00b10.5 m/s injection reads back exactly. Beside, deciding nothing: the training draw alone "
         "(REP \u2212 DAG-OACC at s0) is \u22120.5560 [\u22120.6994, \u22120.3975] m/s, about 16 times the lever's largest point (+0.0348), so a separated cell here would have been necessary and not sufficient (`H-ESTIM-SEED-1`). The readings its builder had "
         "to choose (tick indexing, the v_GT rule, \"toward 0\" as the |X| \u2212 |Y| interval below 0, \"moved\" in all four cells) are in its `NOTES.md`; the Master Mind confirmed them (**[I]**). Its first control pass was badly posed and refused; the cell "
         "numbers did not change, and the redesign of those controls was made after the numbers were seen, which its `NOTES.md` records (**[I]**). The verdict stays not moved under the executed-chord speed reading, the sign reading of \"toward 0\", tick 1 alone "
         "and tick 2 alone (`NOTES.md`; **[I]**). At tick 1 alone the main pair is separated toward 0 at both seeds and the replicate pair is not: a post-hoc look that `NOTES.md` marks as not registered and not a finding, among eight per-tick looks "
         "without multiplicity control (**[I]**)."),
    bullet("**Beside, deciding nothing, not read by the RESULT: M2 in the M-LAUNCH package [M]** (`MLAUNCH.json` `beside_M2`). The RESULT lists the M2 / L2 launch-subset rows as NOT BUILT by the reader of record (above). The M-LAUNCH build "
           "computed a beside block of them: along-track MAE, speed MAE and L2 on the GT-launch subset, X \u2212 Y, over the executed 30-knot horizon (not ticks 1\u20132), T1-kin, the same paired episode-cluster bootstrap "
           "(B = 2,000, seed 0; 327 windows / 49 episodes, and 253 to 255 windows / 45 episodes for L2, which needs GT \u2265 5.2 m/s). Of its 12 cells, 4 are separated: " + "; ".join(M2TXT) + ". No M2 row is separated in the same direction in both "
           "pairs at the same seed, and each cell is one seed in one pair against the training-run spread above, so none is read as an effect."),
)

F["NUMBERS"] = block(
    bullet("**What the numbers say [M]** (computed from the tables above; MEASURED unless marked; T1-kin). (1) **The word is decided by the guard R-ADE, which asks only for 'not separated-worse' and carries no size bar.** R-ADE fails at both seeds, each in one pair; "
           "R-MOVE and R-CRAWL are also false at both seeds. The ADE6 cells are small: the four points lie between 0.0294 and 0.0483 m, all inside \u00b10.0604 m, which is why R-KEEP (a size bar on the point) holds in all four pairs. (2) **At each seed, the separated ADE6 cells flip sign between the main pair "
           "and the replicate pair.** The lever is the same in both pairs (W 1.58926); the two pairs of a seed differ in the trainer seed and the control, and the reader's K-PAIR control passes for both:"),
    T(T_FACT1),
    para("Beyond ADE6, the rows whose separated X \u2212 Y cells have opposite signs between the two pairs of the same seed are, at s1, speed MAE, accel MAE and L1 lateral-accel MAE (at s0 only ADE6); accel MAE at s1 reads +0.7391 [+0.4366, +1.0952] in the "
         "main pair and \u22120.2560 [\u22120.4105, \u22120.1222] in the replicate pair. (3) **No crawl row moved by its bar in both pairs at any seed.** The crawl cells that are separated-better:"),
    T(T_FACT2),
    para("The only cell that clears its bar is in the replicate pair at s0; its main-pair partner is not separated, and every main-pair crawl cell is below its bar (improvement \u00f7 bar from \u221223.4 % to +73.6 % over the eight main-pair cells, "
         "from \u2212130.9 % to +158.1 % over all 16). (4) **Over all 76 X \u2212 Y cells, 16 are separated-better, 7 separated-worse and 53 not separated** (floors bullet), and the cells that reach the bar in the worsening direction (6) outnumber those that reach it in "
         "the improving direction (3). (5) **M-LAUNCH moved by at most +0.0348 m/s, none separated,** against REP \u2212 DAG-OACC at s0 of \u22120.5560 m/s (+0.0199 at s1) on the same cell set. **Why the pattern is what the pair design exists to expose:** the episode-cluster bootstrap "
         "resamples EPISODES with the models held fixed (`H-ESTIM-SEED-1`), so it cannot answer *\"would another training run say this?\"*. The replicate pair asks that second question, and on ADE6 the answer is the opposite sign at each seed: a read on "
         "either pair alone would have called the lever better on ADE6 at one seed and worse at the other, depending on which pair was run. Each separated cell is right about its episodes and silent about its training run, which is why a separated cell from "
         "one pair is necessary and not sufficient, and why the registered predicates hold only when they hold in both pairs. The registered table does not use this reading: R-ADE counts as failed at a seed when it fails in either pair, so the word stays HARMS."),
)

F["NEXT"] = block(
    bullet("**Next (Rule Zero).** **Committed next after HARMS:** arm B, with its own PI go (PREREG \u00a710, verbatim from the JSON: *\"not adopted; read B under this table (train B first, with its own PI go).\"*). This B is LAUNCH-INTENT's own dose B, "
           "not DAG-ORC's B (p = 0.1): the same lever at a lower dose, W 0.726175, a launch-sample weight share of one quarter (K-SHARE band 1/4 \u00b1 0.02; ESTIMATED realised shares 0.2484 / 0.2478 / 0.2435 / 0.2456 at trainer seeds 0 / 1 / 10 / 11, "
           "PREREG \u00a73). It is four box-2 runs (B_0, B_1, B_rep,0, B_rep,1) in the same pair design, read under the same table with its own seeds and never pooled with A; B's HARMS routes to NEXT-LAUNCH. **The \u00a711 branch** is "
           "LAUNCH-INTENT-HI, the same lever at a launch-sample share of one half (W_HI 4.17852, four runs), which M-LAUNCH selects (NOT MOVED, above); it too needs its own PI go. **Both routes need a PI go: B, registered after HARMS, and HI, the branch "
           "M-LAUNCH selects. The PI is being asked which, if either, box 2 runs.** Cost (**[E]**, from a **[M]** span): the A chain ran 13:54 to 16:48 on box 2, 174 min or 2.9 h, against the PREREG's ESTIMATED 2.8 h; B is about the same. "
           "The chain's closing line reads *\"B is not queued\"*."),
    sub("**Every remaining lever and what blocks or unblocks it.** B (W 0.726175, four runs): committed next; blocked on the PI's go for box-2 time, and unblocked by that go plus a B chain on the pattern of the A chain (0 GPU to build; UNVERIFIED that "
        "nothing else is needed)."),
    sub("LAUNCH-INTENT-HI (W 4.17852, four runs): routed by M-LAUNCH NOT MOVED; blocked on the PI's go for box 2. LOOP-STATE (zero training, dev-box CPU; decomposes the crawl by plan index in X's tick dumps): the branch for M-LAUNCH moved, "
        "so it is not routed by this read; it would read the tick dumps, which are single copies on F:."),
    sub("T0 four families: NOT RUN; a build item (`orc_families.py` pointed at the four arms and their controls; inputs: each run's eval JSON in the repo and `.npz` on E:). The anti-echo block and the M2 / L2 launch-subset rows: NOT BUILT by the "
        "reader of record; build items (M2 has a beside block in the M-LAUNCH package, above)."),
    sub("If both NEXT-LAUNCH branches are exhausted, the programme has no registered lever for the crawl rows on DAG-OACC (PREREG \u00a711). The blockers then are the ones that section names: route extent and nav timing (eliminated as the lever by N0, "
        "**[I]**), the state-channel levers (DAgger iteration 2, **[I]**), and the zero-training post-map (it moves the crawl but failed KEEP at s0, **[I]**). What would unblock: a new diagnosis of the loop term, or a PI decision on an architecture "
        "change that gives O a launch-intent input."),
)

F["PROVENANCE"] = block(
    bullet("**Provenance [M].** `RESULT_V8F_LAUNCH_INTENT.md` (landed `8c1e9636`, 17:43:02; the read banked at `d7dc9f11`, 17:11:44), claim row `H-V8F-LAUNCH-INTENT-1`. The read of record is `read_A1/LAUNCH_INTENT_READ.json`, sha256 "
           "`e451e1a6\u2026` (reader `launch_intent_read.py` sha256 `fb3d2d7d\u2026`; PREREG as amended `6ae79f2f\u2026`; manifest `runs.json` sha256 `98d2cd3d\u2026`; RF0 `P0_RF0.json` sha256 `1608d70b\u2026`; wall 341.6 s); its console ends with the lines *\"ZZLAUNCH-INTENT-READ-DONE HARMS\"* and *\"ZZREAD-A1-RC 0\"*. M-LAUNCH is `MLAUNCH.json`, sha256 `f5ebb4b2\u2026`. Run times: box 2 13:54 to 16:48 (174 min), the dev-box T1-kin chain 12 steps in 1756 s, the read of record 341.6 s. The RESULT's checker "
           "`check_result.py` re-finds every quoted number (133 checks, 0 failed; **[I]**, its output in the repo), and does not recompute the cells from the per-window dumps or re-run the bootstrap (those dumps are single copies, below): a defect "
           "inside the estimator or the reader's heavy layer would not show there. Every sha256 of the RESULT's provenance table is of the bytes at `d7dc9f11`; the table, reduced to the read, the logs, the pins and the checkpoints "
           "(the 20 run-record rows are in the RESULT):"),
    T(T_PROV),
    bullet("**Disclosures carried from the RESULT (\u00a78) that are not stated elsewhere in this section.** (a) **Pin-fill time.** `PINS_FILLED.txt` of the chain package (`raw/2026-10-28-launch-intent-chain/`, sha256 `fa825de8\u2026`) says the Master Mind "
           "filled the chain pins at ~14:10. They were filled at ~13:50 (**[I]**, the Master Mind's brief), before commit `d8c9f7bf` (13:51:52) and after the PREREG was promoted at `77234276` (13:49:00); the bracket between the two commit times is "
           "**[M]**, from the commits, and the box-2 pins line is stamped 13:54 (**[M]**, the chain log). The pins are the promoted ones: the PREREG blob and the reader blob at `77234276` hash to the `469bac1f\u2026` and `5e950dcf\u2026` that the "
           "pins carry (**[M]**, re-measured by the audit). (b) **A label inside the reader.** The JSON's `provenance.prereg` reads *\"Project Steering/PREREG_V8F_LAUNCH_INTENT.md (DRAFT v2, MM_REDESIGN_2026-10-28)\"*; that is a string inside the reader, "
           "which ran with `--prereg-sha` equal to the amended text's sha256 `6ae79f2f\u2026`. (c) **Single copies.** The per-episode dumps and tick arrays are under `F:/TanitAD-scratch/launch_intent_t1_out/` (the chain's `--out-dir` and `--dump-ticks`); "
           "the repo holds the rollout JSONs under `t1/`, the fresh-T0 evals under `t0fresh/` and each run's record, and no tick arrays. M-LAUNCH reads tick dumps, so it needs F:. The checkpoints are on E: only. "
           "(d) **Disclosed coupling.** DAG-ORC and LAUNCH-INTENT share the REP_s runs: in DAG-ORC they set the floor RF, here they are the control of the replicate pair and RF is not a bar term, so neither read's result changes the other's bar."),
)

F["R81"] = (wrap("**O's launch-sample along-track weight does not move the crawl rows and fails the position guard: HARMS, arm A is not adopted** (\u00a77.29, LAUNCH-INTENT; T1-kin, paired episode-cluster bootstrap, B = 2,000, 141 episodes, per seed and "
                  "per pair; four training draws). R-MOVE is false at both seeds (1 of the 16 crawl cells clears its bar, L2 curvature in the replicate pair at s0, \u22120.0215 [\u22120.0423, \u22120.0048] against 0.0136; no main-pair cell clears), all 16 X \u2212 P10 "
                  "crawl cells stay separated-worse, and R-ADE fails at both seeds, each in one pair (ADE6 +0.0431 [+0.0137, +0.0711] and +0.0483 [+0.0105, +0.0884] m), while the other pair of each seed is separated the other way "
                  "(\u22120.0294 [\u22120.0522, \u22120.0077] and \u22120.0341 [\u22120.0655, \u22120.0027] m). **Established (MEASURED):** the registered word HARMS for arm A (W 1.58926); that the lever reached the data (the weight share on launch samples is "
                  "0.3256 to 0.3315 against 1/3 \u00b1 0.02); M-LAUNCH NOT MOVED in all four cells (T1-kin, 49 episodes / 327 launch windows), which routes \u00a711 to LAUNCH-INTENT-HI. **Not established:** an effect of the lever at any dose (one training "
                  "run per arm, pair and seed; the separated ADE6 cells flip sign between same-seed pairs; `H-ESTIM-SEED-1`), anything about B or HI (not read), and the T0 four families, the anti-echo block and the M2 / L2 launch-subset rows "
                  "(NOT RUN / NOT BUILT). Both B (registered after HARMS) and HI (the \u00a711 branch) need their own PI go.", "   ", first_indent="")).strip()

F["R83"] = wrap("**LAUNCH-INTENT (\u00a77.29) reads HARMS: arm A (W 1.58926) is not adopted.** On the crawl set (yaw-rate, curvature, L2 and M1 against P10) no row moves by its bar in both pairs at either seed (T1-kin; paired episode-cluster bootstrap, "
                "B = 2,000, 141 episodes; 1 of the 16 crawl cells clears its bar and none of the 8 main-pair cells), and R-ADE fails at both seeds, each in one pair; the separated ADE6 cells flip sign between the two same-seed pairs. M-LAUNCH reads "
                "NOT MOVED (49 episodes), which routes \u00a711 to LAUNCH-INTENT-HI. The adopted base is unchanged (DAG-OACC) and the crawl set is still open. The committed next after HARMS is LAUNCH-INTENT's arm B (W 0.726175) and the \u00a711 branch is "
                "LAUNCH-INTENT-HI (W 4.17852); each needs its own PI go, and the PI is being asked which, if either, box 2 runs.", "  ", first_indent="")

F["R84"] = ("**HARMS** (\u00a77.29; `RESULT_V8F_LAUNCH_INTENT.md`, landed `8c1e9636`; all 206 reader controls PASS; T1-kin, Amendment 1 in force): arm A not adopted. R-MOVE is false at both seeds (1 of 16 crawl cells clears its bar: L2 curvature, replicate pair, "
            "s0, \u22120.0215 against 0.0136); R-ADE fails at both seeds, each in one pair (+0.0431 and +0.0483 m), and the other pair of each seed is separated the other way. M-LAUNCH **NOT MOVED** (none of 4 cells separated; 49 episodes), which routes section 11 to "
            "LAUNCH-INTENT-HI. T0 four families NOT RUN; anti-echo block NOT BUILT. Next: B (W 0.726175) is registered after HARMS and HI (W 4.17852) is the section 11 branch, each with its own PI go; the PI is being asked which, if either, box 2 runs.")

# ---- guards on the texts ----------------------------------------------------------------------------------------------------------------------------------------------------
SLOT_ORDER = ["LINE", "ABSTRACT", "TITLE729", "STATUS729", "OUTCOME", "CONTROLS", "RUNS", "FLOORS", "CRAWL", "GUARDS", "FAMILIES", "MLAUNCH", "NUMBERS", "NEXT", "PROVENANCE", "R81", "R83", "R84"]
out = OrderedDict((k, F[k]) for k in SLOT_ORDER)
assert list(out) == SLOT_ORDER
for k, v in out.items():
    assert isinstance(v, str) and v.strip() and "@@" not in v and chr(13) not in v, k
    low = v.lower()
    for w in ("void", "superseded", "retracted", "formerly", "no longer", "correction", "withdrawn"):
        assert re.search(r"\b" + w + r"\b", low) is None, (k, w)
    assert "closed loop" not in low and "closed-loop" not in low and "drives" not in low and "$" not in v, k
    assert re.search(r"retract|supersed|withdrawn|was wrong|not the floor|previously read|outdated|i said", low) is None, k
    nocode = re.sub(r"`[^`\n]*`", "", v)
    assert re.search(r"<[A-Za-z/!]", nocode) is None, (k, re.search(r"<[A-Za-z/!]", nocode))
    for line in v.split(NL):
        if line.startswith("|"):
            assert line in RES_SET, (k, line[:80])
for k in ("LINE", "TITLE729", "R84"):
    assert NL not in out[k].strip(), k
assert "|" not in out["R84"]
assert out["ABSTRACT"].count(NL + "* ") + 1 == 2
json.dump(out, open(OUTF, "w", encoding="utf-8", newline=NL), ensure_ascii=False, indent=1)
open(OUTF, "a", encoding="utf-8", newline=NL).write(NL)
print("FILL written:", OUTF, {k: len(v) for k, v in out.items()})
