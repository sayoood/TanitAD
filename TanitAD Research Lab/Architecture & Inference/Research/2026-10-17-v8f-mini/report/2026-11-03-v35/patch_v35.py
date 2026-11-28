"""Report v35 = v34 + section 7.34 (the M1 crawl, located, and the decode levers of 2026-11-02), the status line, one abstract bullet, 8.1 item 22, 8.3 Update (v35), 8.4 rows 1ag to 1am and 9 item 13.
Built from the v34 report blobs of one repository tip (anchored edits, every anchor asserted to occur exactly once, md and html edited by the same replaces). Nothing is read from a working tree.
Nothing is staged, committed or pushed.

python -I patch_v35.py [tip [out_dir]]
Inputs, all `git show <tip>:<path>` from the mirror: the v34 V8F_SCIENTIFIC_REPORT.md and v8f_report.html (asserted byte-identical to the landed v34 deliverables by sha256), page_template_v30.html,
the RESULTS of GAPDECOMP, GAPDECOMP-M, SMOOTH-T1 (RESULTS_A1.md and RESULTS.md), trigger probe 5, KAPPA-T1 (RESULTS.md, GATE_RESULT.md) and FEAS2 / FEAS3 (RESULTS.md, RESULTS_FEAS3.md), the register.
Every table line of 7.34 is asserted to be a line of its source RESULTS; every italic quotation of 25 characters or more is asserted verbatim in a source."""
import hashlib
import os
import re
import subprocess
import sys
from collections import OrderedDict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_argv = sys.argv[1:]
TIP = _argv[0] if _argv else "92cbf277dd4091854ab4c20c590d3a58ad7d3047"
TP = TIP[:8]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = (_argv[1] if len(_argv) > 1 else HERE + "/out").rstrip("/\\") + "/"
MIRROR = "C:/Users/Admin/tanitad-push"
R = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
BANNED = ("void", "superseded", "retracted", "formerly", "no longer", "correction", "withdrawn")


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


# ---- the v34 deliverables are the starting point -------------------------------------------------------------------------------------------------------
V34_MD_SHA = "c90df7c3b722369131dceed4ead4dada181863dfc604d1264147d2e4f5530950"
V34_HTML_SHA = "4fb3d6422fcd83442144f52c060140971bead8a2341217ac9d5cd699db04a7b5"
RP34 = R + "report/2026-11-03/"
s = blob(RP34 + "V8F_SCIENTIFIC_REPORT.md")
h = blob(RP34 + "v8f_report.html")
tpl = blob(R + "report/2026-10-26/page_template_v30.html")
assert sha(raw(RP34 + "V8F_SCIENTIFIC_REPORT.md")) == V34_MD_SHA, "the tip's report md is not the landed v34 deliverable"
assert sha(raw(RP34 + "v8f_report.html")) == V34_HTML_SHA, "the tip's report html is not the landed v34 deliverable"
assert tpl.count("%%MD%%") == 1 and h == tpl.replace("%%MD%%", s), "v34 html is not the template with the v34 md"
assert s.count("**Status:** research report, v34 of 2026-11-03") == 1
S0 = s

# ---- sources --------------------------------------------------------------------------------------------------------------------------------------------
SRC_PATHS = OrderedDict([
    ("GD", "raw/2026-11-02-gapdecomp/RESULTS.md"), ("GDM", "raw/2026-11-02-gapdecomp-m/RESULTS.md"), ("SM", "raw/2026-11-02-smooth-t1/RESULTS_A1.md"), ("SM1", "raw/2026-11-02-smooth-t1/RESULTS.md"),
    ("P5", "raw/2026-11-02-trigger-probe5/RESULTS.md"), ("KP", "raw/2026-11-02-kappa-t1/RESULTS.md"), ("KPG", "raw/2026-11-02-kappa-t1/GATE_RESULT.md"),
    ("F2", "raw/2026-11-02-feas2-t1/RESULTS.md"), ("F3", "raw/2026-11-02-feas2-t1/RESULTS_FEAS3.md"),
])
SRC = OrderedDict((k, blob(R + p)) for k, p in SRC_PATHS.items())
REG = blob("Project Steering/GOALS_AND_CLAIMS.md")
CARDS = OrderedDict([("GD", "raw/2026-11-02-gapdecomp/PREREG_FROZEN.md"), ("GDM", "raw/2026-11-02-gapdecomp-m/PREREG_FROZEN.md"), ("SM", "raw/2026-11-02-smooth-t1/PREREG_FROZEN.md"),
                     ("SMA", "raw/2026-11-02-smooth-t1/AMENDMENT_1.md"), ("P5", "raw/2026-11-02-trigger-probe5/PREREG_FROZEN.md"), ("KP", "raw/2026-11-02-kappa-t1/PREREG_FROZEN.md"),
                     ("F2", "raw/2026-11-02-feas2-t1/PREREG_FROZEN.md"), ("F3", "raw/2026-11-02-feas2-t1/PREREG_FEAS3_FROZEN.md")])
CARD_SHA = {k: sha(raw(R + p)) for k, p in CARDS.items()}
for k, pre in (("GD", "d5c9caa8"), ("GDM", "1c5dbaf0"), ("SM", "4030d22f"), ("SMA", "717ff177"), ("P5", "791e8a62"), ("KP", "9564d568"), ("F2", "ecf9bb5d"), ("F3", "e817abcf")):
    assert CARD_SHA[k].startswith(pre), (k, CARD_SHA[k])
# landing commits and their author times, measured from the mirror
for _c, _t in (("665d6bd", "2026-11-02 14:15:15"), ("51fd683", "2026-11-02 14:30:56"), ("0a5709a", "2026-11-02 15:57:02"), ("5efeffb", "2026-11-02 16:26:15"), ("992d736", "2026-11-02 16:28:19"),
               ("8e1a40d", "2026-11-02 18:44:44")):
    assert gitrun(["show", "-s", "--format=%ci", _c]).startswith(_t), _c
for k, c in (("GD", "665d6bd"), ("GDM", "51fd683"), ("SM", "0a5709a"), ("SM1", "0a5709a"), ("P5", "5efeffb"), ("KP", "992d736"), ("KPG", "992d736"), ("F2", "8e1a40d"), ("F3", "8e1a40d")):
    assert gitrun(["log", "--format=%h", "-1", TIP, "--", R + SRC_PATHS[k]]).startswith(c[:7]), (k, c)
_BAN = re.compile(r"\b(" + "|".join(BANNED) + r")\b", re.I)


# ---- tables, copied line for line --------------------------------------------------------------------------------------------------------------------
def ncols(line):
    body = re.sub(r"`[^`\n]*`", "x", line.replace("\\|", "x"))
    return body.count("|") - 1


def blk(src, first, nth=0, n=None, keep=None, must=None):
    lines = SRC[src].split("\n")
    idx = [i for i, l in enumerate(lines) if l.startswith(first)]
    assert len(idx) > nth, (src, first[:60], len(idx), nth)
    rows = []
    for l in lines[idx[nth]:]:
        if not l.startswith("|"):
            break
        rows.append(l)
    if keep is not None:
        sel = rows[:2] + [r for r in rows[2:] if any(r.startswith(p) for p in keep)]
        assert len(sel) == 2 + sum(1 for r in rows[2:] if any(r.startswith(p) for p in keep)), (src, first[:40])
        rows = sel
    if n is not None:
        assert len(rows) == n, (src, first[:60], len(rows), n)
    if must is not None:
        assert any(must in r for r in rows), (src, first[:50], must)
    assert all(r in lines for r in rows) and len({ncols(r) for r in rows}) == 1, (src, first[:60])
    assert not any(_BAN.search(r) for r in rows), ("banned word in a table line", first[:60])
    assert not any(re.search(r"closed[- ]loop|driving performance", r, re.I) for r in rows), ("tier word in a table line", first[:60])
    return rows


T = OrderedDict()
T["GD_HEAD"] = blk("GD", "| row (DAG-OACC vs P10) |", n=7)
T["GD_CLS"] = blk("GD", "| class | windows | episodes | share of windows |", n=7)
T["GD_CTRL"] = blk("GD", "| control | result |", n=8)
T["GD_RATIO"] = blk("GD", "| row | DAG-OACC s0 | DAG-OACC s1 | REP s0 |", n=7)
T["GD_FAM"] = blk("GD", "| row | w_T | DAG-OACC s0 Delta / T-share |", n=21)
T["GDM_HEAD"] = blk("GDM", "| row (DAG-OACC vs P10, class M) |", n=7)
T["GDM_CTRL"] = blk("GDM", "| control | result |", n=8)
T["GDM_PLAN"] = blk("GDM", "| arm / seed | who | plan speed |", n=10)
T["GDM_6B"] = blk("GDM", "| zeroed | arm | seed | rows separated-worse | D2 |", n=10)
T["SM_HEAD"] = blk("SM", "| arm | seed 0 | seed 1 | outcome by the committed rule", n=4)
T["SM_RULE"] = blk("SM", "| arm | DOMINANT | HARMS (either seed) |", n=4)
T["SM_COND"] = blk("SM", "| arm | seed | D1 ADE6 X - P10 |", n=6)
T["SM_CTRL"] = blk("SM", "| control | measured | tolerance | result |", n=12)
T["SM_CRAWL"] = blk("SM", "| arm | seed | row | X - Y' | X - P10 |", keep=("| DAG-OACC | s", "| SOUP | s"), n=22)
T["SM_FAM"] = blk("SM", "| arm | seed | family (rows) | X - P10 | X - Y' |", keep=("| DAG-OACC |",), n=14)
T["P5_VERD"] = blk("P5", "| decision cell | dAUC (fold-mean) [CI] |", n=4)
T["P5_ARMS"] = blk("P5", "| arm | P-STOP | P-MOVE |", n=8)
T["P5_PAIR"] = blk("P5", "| contrast | P-STOP | P-MOVE |", n=9)
T["P5_CTRL"] = blk("P5", "| control | P-STOP | P-MOVE |", n=12)
T["KP_GATE"] = blk("KPG", "| run | E = curvature MAE X - P10 (1/m) |", n=6)
T["KP_OUT"] = blk("KP", "| arm | outcome | seed 0: D1 / D2 / D3 |", n=4)
T["KP_PRED"] = blk("KP", "| arm | outcome read | curvature MAE X - Y separated-better s0 / s1 |", n=4)
T["KP_KA1"] = blk("KP", "| run | dump bit-identical |", n=6)
T["KP_FAM"] = blk("KP", "| row | family | X - P10 | X - Y | Y - P10 |", nth=0, n=21, must="-1.2441 [-1.4510, -1.0498] SB")
T["F2_OUT"] = blk("F2", "| lever | arm | outcome | seed 0: D1 / D2 / D3 |", n=8)
T["F2_D2"] = blk("F2", "| lever | M1 crawl share | curvature MAE |", n=5)
T["F2_PRED"] = blk("F2", "| lever | DOMINANT | HARMS | MOVED | NO EFFECT |", n=5)
T["F2_M1"] = blk("F2", "| lever | cell | M1 X - Y | Y - P10 |", n=14)
T["F2_PRE"] = blk("F2", "| cell | pre-check kbox | pre-check abox |", n=6)
T["F2_FAM"] = blk("F2", "| row | family | X - P10 | X - Y | Y - P10 |", nth=0, n=21, must="-1.2326 [-1.4374, -1.0372] SB")
T["F3_OUT"] = blk("F3", "| lever | arm | outcome | seed 0: D1 / D2 / D3 |", n=8)
T["F3_D2"] = blk("F3", "| lever | M1 crawl share | curvature MAE |", n=5)
T["F3_HARMS"] = blk("F3", "| lever | ADE6 beyond guard | ADE6 |", n=5)
T["F3_PRED"] = blk("F3", "| lever | DOMINANT | HARMS | MOVED | NO EFFECT |", n=5)
T["F3_M1"] = blk("F3", "| lever | cell | M1 X - Y | Y - P10 |", n=14)
T["F3_SB"] = blk("F3", "| lever | cell | 5.2 row | 1 m/s |", n=14)
T["F3_FAM"] = blk("F3", "| row | family | X - P10 | X - Y | Y - P10 |", nth=6, n=21, must="-0.0004")

# ---- claims taken from the sources, asserted present ----------------------------------------------------------------------------------------------------
PHRASES = [
    ("GD", "no crawl row is TRIGGER-BOUND for DAG-OACC at both seeds, and removing the trigger windows' gap entirely does not clear D2 for DAG-OACC or SOUP at either seed"),
    ("GD", "the trigger windows carry 3.0 to 3.6 times their window share of the curvature gap, 5.4 to 6.4 times of L2 and 3.9 to 5.0 times of M1"),
    ("GD", "L2 curvature is **TRIGGER-BOUND at both seeds** (+0.747 [+0.517, +0.899], +0.768 [+0.585, +0.892])"),
    ("GD", "DAG-OACC s0: yaw-rate +4.94 [+3.02, +7.08], curvature +0.045 [+0.027, +0.064], L2 +0.010 [+0.004, +0.019], M1 +0.022 [+0.013, +0.033]"),
    ("GD", "Class O is 8 windows / 3 episodes"), ("GD", "Classes S and O hold **no** valid window for L2 and M1"), ("GD", "the point share there is negative because the GT-launch windows are better than P10 on that row"),
    ("GD", "(same arrays, minutes, CPU) by GT turning against straight and by v0 bucket, with a new frozen card and both outcomes committed"),
    ("GDM", "three of the five rows are LOCALISED to the 3 to 8 m/s windows (`speed=3-8`, 413 windows / 68 episodes, 20.3 % of M)"), ("GDM", "**Shape is not the axis.**"), ("GDM", "Path shape localises nothing."),
    ("GDM", "TURN holds +0.100 / +0.084 of yaw-rate and +0.062 / +0.077 of curvature with 0.074 of the windows"), ("GDM", "the excess is magnitude without a sign: jitter, not a steady bias"),
    ("GDM", "**9.0 % / 11.4 % of DAG-OACC windows have a plan step below 0.5 m/s, against 0.7 % for P10**"), ("GDM", "**zeroing every window with v0 < 8 m/s (861 of 2,399) still leaves the same four rows separated-worse**"),
    ("GDM", "the lateral excess in M lives in the turning windows"), ("GDM", "No lever is blocked: everything above is CPU on banked arrays and dumps."),
    ("GDM", "In the 3 to 8 m/s windows replace each DAG-OACC plan step below a floor by the previous step's speed and heading (a pure post-processing of the executed track)"),
    ("SM1", "The outcome is REFUSED by the card's own rule"), ("SM", "**Status: QUOTABLE ON Y' (banked-reference comparisons exceed the measured tolerance: K1p, K2p, K5_floor_clause;"),
    ("SM", "0 of 7197 ticks have a `plan` deviation above 1e-4 m"), ("SM", "as the next zero-training arm, wired the same way as a flag of the same tool, with its own pre-registration frozen before data."),
    ("SM", "the cheapest next step is S composed with the curvature-bounded decode"), ("SM", "The amendment moved the reference of the controls and the Y side after the 1-thread tables were seen"),
    ("SM", "HARMS tests 15 non-crawl rows x 2 seeds = 30 tests per arm"),
    ("REG", "what still blocks D2 is a near-uniform per-knot curvature overshoot over P10, which a within-plan smoother does not act on"),
    ("P5", "CROPS HELP iff dAUC >= 0.03 with CI lower > 0 in either population (P-STOP, P-MOVE); CROPS DO NOT HELP iff dAUC < 0.01 in both populations"),
    ("P5", "per-row relative L2 median **0.0264**"), ("P5", "relative L2 **median 0.0637, max 0.1228**"), ("P5", "so that item was not run: UNVERIFIED"), ("P5", "the 5-draw mean fold-mean AUC is 0.4962, inside the committed band [0.47, 0.53]"),
    ("P5", "The frozen DINOv3-L/16 caps the go/stay trigger at about V+E."), ("P5", "a fine-tuned or different encoder (the encoder is the part of the pipeline probes 1-5 could not vary), or a dedicated traffic-light-state input"),
    ("P5", "TL rows [0,336) cols [384,1536) = 21x72 = 1,512 tokens; LEAD rows [272,608) cols [480,1440) = 21x60 = 1,260 tokens"),
    ("KPG", "Decision: (a) s_HIGHK >= 0.5 at both DAG-OACC seeds."), ("KP", "as a further flag value of the same tool, to attribute the harm to the accel clamp or the curvature clamp"),
    ("KP", "s0: curvature 164%, L2 90%, M1 -10%, yaw-rate n/a (Y - P10 <= 0); s1: curvature 127%, L2 91%, M1 -13%, yaw-rate 410%"),
    ("KP", "HYPOTHESIS: the acceleration clamp (`a_max` 4.0, which holds a plan's speed down when it launches above the entry speed; GATE_RESULT.md tool smoke) drives M1, along-track and ADE6"),
    ("KP", "gate lever (a) at 55 %; given (a), **DOMINANT 10 %, MOVED 40 %, NO EFFECT 30 %, HARMS 20 %**"), ("KP", "READER_FIX_FLOOR.md"),
    ("F2", "6 of 6 lever x arm outcomes read HARMS (DOMINANT, MOVED and NO EFFECT: 0). D1 (ADE6 X - P10) holds at 12 of 12 cells, D3 at 12 of 12, and D2 fails at 12 of 12."),
    ("F2", "**MEASURED: M1 crawl share fails D2 at 12 of 12 cells** (the card gave 85 % that it fails under all three levers)"), ("F2", "the rollout M1 row above reads the executed rows, where the excess lives."),
    ("F2", "plan_s sat below the P10 prior's share in all four cells"), ("F2", "is aimed at the heading / S-POS harm (the curvature cap stops acting below 2 m/s)"),
    ("F2", "abox was predicted NO EFFECT (65 %) and read HARMS at both arms (25 %)"), ("F2", "the card's committed next arms do not act on abox's own harm, so abox is not carried into FEAS3"),
    ("F3", "6 of 6 lever x arm outcomes read **HARMS** (DOMINANT, MOVED and NO EFFECT: 0). D1 (ADE6 X - P10) holds at 12 of 12 cells, D3 at 12 of 12, and D2 fails at 12 of 12 (on M1 crawl share at 12 of 12)."),
    ("F3", "**The one MOVED condition hidden by precedence: lonfloor on SOUP.**"), ("F3", "(-0.0004; +0.0104 m)"), ("F3", "share of the excess closed 2.0 % to 6.6 % (mean 4.3 %)"), ("F3", "a null here is not evidence of safety"),
    ("F3", "the decode-time box family (box, kbox, abox, kabox, kbox2) is closed as a heading / S-POS lever (the clamp acts on executed steps the logged future truly turns on)"),
    ("F3", "12 of 12 identical (identical)"), ("F3", "the executed-state question was taken up by the separate package `../histswap`"), ("F3", "whether DAgger's TRAIN-window rollout inputs exist on disk is **UNVERIFIED**"),
    ("F3", "120 windows from 19 episodes, threshold 2.372 m/s^2"), ("F3", "(c) the lateral shrink `P + diag(1, alpha)(S(plan) - P)` with alpha chosen on TRAIN windows"),
    ("REG", "Seven decode levers since S leave M1 the D2 blocker; the cause sits upstream of the decode (H-HISTSWAP-1)."),
    ("REG", "the crawl is not a threshold artefact (CTRL s0, row definition Y 0.0713 vs P10 0.0230; at a 1 m/s margin 0.1786 vs 0.0480)"),
    ("REG", "some ticks' issued plans move by 105-227 m"), ("REG", "the banked baseline differs from a fresh run by up to 1.71e-02 m on the floor"),
]
_corp = {k: re.sub(r"\s+", " ", v) for k, v in SRC.items()}
_corp["REG"] = re.sub(r"\s+", " ", REG)
_nm = [(k, p[:70]) for k, p in PHRASES if re.sub(r"\s+", " ", p) not in _corp[k]]
assert not _nm, ("a claim phrase is not in its source", _nm)

# ---- the section text ------------------------------------------------------------------------------------------------------------------------------------
SEC_734 = """### 7.34 The M1 crawl, located, and the decode levers (2026-11-02): the excess sits in executed speed at v0 3 to 8 m/s, and no decode lever closes it

* **Status of this section (v35) [M].** Seven zero-training reads of 2026-11-02 on the eval-141 causal lane (2,399 windows, 141 episodes), each a card frozen before its own numbers and a RESULTS file landed on the branch: GAPDECOMP (`H-GAPDECOMP-1`), GAPDECOMP-M (`H-GAPDECOMP-M-1`), SMOOTH-T1 (`H-SMOOTH-T1-1`), trigger probe 5 (`H-CRAWL-06-PROBE5`), KAPPA-T1 (`H-KAPPA-T1-1`), FEAS2 (`H-FEAS2-T1-1`) and FEAS3 (`H-FEAS3-T1-1`). Each is read from its RESULTS file at the tip `{{TP}}` with `git show`, never from a working tree, and each number is **[M]** with that path. Tier: the six rollout reads are T1-kin (the model's own executed state, logged cameras, its own plan applied open loop in the scene; it says nothing about re-perception; no T2 number exists); probe 5 is a T0-class probe of representations on logged inputs. Estimator: the paired episode-cluster bootstrap, B = 2,000, seed 0 (`taniteval.ci` draw stream), which answers *"would another draw of EPISODES say this?"* and nothing about another training run (`H-ESTIM-SEED-1`: one training run per seed) or another inference run (the planner is deterministic). Arms in the six rollout reads: DAG-OACC (the adopted base, the headline arm) and SOUP (§7.32), each at s0 and s1, against the P10 floor (Read A) and, from SMOOTH-T1 on, against the same arm without the lever (X − Y).
* **Synthesis (zero training; read the setup of each study before the sentence that cites it).** After these reads, D2 of Read A against P10 is blocked by the M1 crawl share. M1 fails D2 at 4 of 4 arm-seed cells after the curvature box (KAPPA-T1), at 12 of 12 cells for FEAS2 and at 12 of 12 cells for FEAS3, while D1 (ADE6 better than P10) holds at 12 of 12 and D3 (a separated-better row per family) at 12 of 12 in each of the two. It is the only failing D2 row at every cell for kbox and kbox2lf (the other lever-arm cells add L2 at 1 of 4, or curvature, L2 and yaw-rate for abox and lonfloor). The excess does not live in the go/stay trigger windows (15.0 % of the windows): no DAG-OACC crawl row is TRIGGER-BOUND, M1 reads +0.606 / +0.545 (MIXED) and the DAG-OACC s0 gap stays separated-worse with the trigger windows' gap set to 0 (ESTIMATED). It lives in the moving windows at v0 3 to 8 m/s (413 windows, 20.3 % of class M), which carry +0.653 / +0.684 of DAG-OACC's M gap on yaw-rate, +0.934 / +0.915 on L2 curvature and +0.792 / +0.751 on M1; there the plan's mean |kappa| is 0.1845 / 0.2621 against P10's 0.0623 and the plan speed is only +0.398 / +0.241 m/s above GT (P10 +1.599), so the defect is curvature jitter and slow steps (9.0 / 11.4 % of the windows have a plan step below 0.5 m/s, against 0.7 % for P10), not a bias or a speed collapse. The M1 row reads the executed rows, and the plan-level crawl is not where the excess is: in the FEAS2 pre-check the plan sat below the P10 prior's share of crawl steps in all four cells. None of the decode levers closes it. S moves SOUP (MOVED) and leaves DAG-OACC NO EFFECT; the curvature box and all six later levers read HARMS (heading MAE X − Y +1.6946 / +1.3399 deg for the box at DAG-OACC; ADE6 X − Y +0.1283 / +0.1442 m for kbox2 against a +0.0604 m guard); the lever that moves M1 most, lonfloor, moves it separated-better at 4 of 4 cells and closes 2.0 to 6.6 % of its excess. Native-scale crops of the traffic-light and lead-vehicle bands do not lift the go/stay trigger above the whole-vector readout either (C+V+E − V+E −0.0231 [−0.0422, −0.0050] on P-STOP, T0-class). The register's own count is seven decode levers since S that leave M1 the D2 blocker, with the cause upstream of the decode (`H-HISTSWAP-1`; not read in this version).

**7.34.1 GAPDECOMP: the gap against P10 is not trigger-bound (`H-GAPDECOMP-1`).**

* **Status and setup [M].** Read 2026-11-02; `R/raw/2026-11-02-gapdecomp/RESULTS.md`, landed `665d6bd` (2026-11-02 14:15:15). Card `PREREG_FROZEN.md` sha256 `{{C:GD}}…`, frozen 2026-11-02T14:07:12+0100 before any class-level number. Question: how much of what blocks a DOMINANT read lives in the go/stay trigger windows, GT-launch (327 windows, 49 episodes) plus GT-stay (32), 359 of the 2,399 windows? Arms DAG-OACC (headline), REP and SOUP against P10, s0 and s1, from the SOUP read's per-window dumps. Committed outcomes (register row): a crawl row is TRIGGER-BOUND iff the trigger windows carry at least 0.7 of its X − P10 gap with a CI lower bound above 0.5 at both seeds, NOT TRIGGER-BOUND iff at most 0.3 with the CI upper bound below 0.5, otherwise MIXED. T1-kin; estimator as in the status bullet, class statistics recomputed in every draw.
* **Outcome [M] (the headline table; T-share = share of the row's gap carried by the trigger windows).** The classes first:

<<T:GD_CLS>>

  Then the verdict per row for DAG-OACC:

<<T:GD_HEAD>>

  No DAG-OACC crawl row is TRIGGER-BOUND at both seeds (verdicts: ADE6 and yaw-rate NOT TRIGGER-BOUND; curvature, L2 and M1 MIXED). For SOUP the L2 curvature row is TRIGGER-BOUND at both seeds (+0.747 [+0.517, +0.899], +0.768 [+0.585, +0.892]); for REP L2 reads MIXED (s0 +0.671 [+0.480, +0.808] misses 0.7). Beside, POST-HOC descriptive (not pre-registered, decides nothing), the T-share divided by the trigger windows' window share w_T:

<<T:GD_RATIO>>

  The trigger windows carry 3.0 to 3.6 times their window share of the curvature gap, 5.4 to 6.4 times of L2 and 3.9 to 5.0 times of M1: over-representation, not ownership, since about half of the curvature gap (T-share 0.42 to 0.50) and 39 to 53 % of M1 sit outside them. **[E]** With the trigger windows' gap set to 0 the four rows yaw-rate, curvature, L2 and M1 stay separated-worse for DAG-OACC and SOUP at both seeds (DAG-OACC s0: yaw-rate +4.94 [+3.02, +7.08], curvature +0.045 [+0.027, +0.064], L2 +0.010 [+0.004, +0.019], M1 +0.022 [+0.013, +0.033]): decomposition arithmetic, not a retrained model, so a trigger-only fix would not clear D2.
* **Controls [M].** All run before any class number was read:

<<T:GD_CTRL>>

* **The four families (per family, never pooled; STRATEGIC reported, not paused, for v8F).** The class split gives Delta and T-share on all 19 rows, descriptive, with a verdict only on the five crawl-linked rows (ADE6, yaw-rate, curvature, L2, M1). LONGITUDINAL (speed, along-track, accel MAE, DK-VIOL, DK-HW, DK-TG), LATERAL (heading, yaw-rate, curvature, cross-track, L1, L2), TACTICAL (TAC-LAT, TAC-LON) and STRATEGIC (S-POS@30, @50, @80) are read as descriptive shares only:

<<T:GD_FAM>>

  No family verdict is claimed. The T-share of the TACTICAL rows is +0.09 / +0.12 (TAC-LAT) and −0.36 / −0.39 (TAC-LON) for DAG-OACC; the tactical goal-head F1 is NOT COMPUTABLE at T1-kin (**INHERITED** from §7.32, not re-read). Classes S and O hold no valid window for L2 and M1, so the trigger set for those two rows is the GT-launch class alone; class O is 8 windows / 3 episodes.
* **Limits [M].** The two seeds differ by about as much as the intervals are wide (L2 T-share +0.771 against +0.660; yaw-rate −0.424 against +0.205), so a verdict that flips on one seed is a statement about the rig's run-to-run spread, not about a lever. The yaw-rate share is unstable (heavy-tailed MAE; the point share is negative at DAG-OACC s0 because the GT-launch windows are better than P10 on that row).
* **Next, as committed.** *"(same arrays, minutes, CPU) by GT turning against straight and by v0 bucket, with a new frozen card and both outcomes committed"*: GAPDECOMP-M below, with the hypothesis that the lateral excess lives in the turning windows.

**7.34.2 GAPDECOMP-M: the M gap is localised to v0 3 to 8 m/s, and path shape localises nothing (`H-GAPDECOMP-M-1`).**

* **Status and setup [M].** Read 2026-11-02; `R/raw/2026-11-02-gapdecomp-m/RESULTS.md`, landed `51fd683` (2026-11-02 14:30:56). Card sha256 `{{C:GDM}}…`, frozen 2026-11-02T14:19:53+0100 before any subclass number. Class M is the 2,032 moving windows (140 episodes) of GAPDECOMP; the subclasses are GT path shape (STRAIGHT, GENTLE, TURN) and v0 bucket. Committed outcome: a row is LOCALISED iff one subclass holding at most 30 % of M's windows carries at least 0.5 of the row's M gap with a CI lower bound above 0.3 at both seeds for DAG-OACC, DIFFUSE otherwise; 8 eligible candidates per row, no multiplicity adjustment (card). Five rows are decomposed: ADE6, yaw-rate, curvature, L2 curvature, M1. T1-kin; estimator as in the status bullet; the planner is deterministic. SOUP is a second arm on the same windows and the same P10, not an independent replication.
* **Outcome [M].** Share of DAG-OACC's class-M gap carried by `speed=3-8`:

<<T:GDM_HEAD>>

  Three of the five rows are LOCALISED to the 3 to 8 m/s windows (413 windows / 68 episodes, 20.3 % of M) at both seeds, and SOUP reads the same by the rule; curvature MAE (+0.473 at s0, a miss of the committed 0.5 by 0.027) and ADE6 are DIFFUSE. Path shape localises nothing: TURN holds +0.100 / +0.084 of the yaw-rate gap and +0.062 / +0.077 of curvature with 0.074 of the windows. The hypothesis written into GAPDECOMP's next-lever note, *"the lateral excess in M lives in the turning windows"*, is therefore not supported for the two lateral rows.
* **Beside the verdict [M].** Plan against GT inside `speed=3-8` (plan minus GT):

<<T:GDM_PLAN>>

  The plan is not slower than GT on average (P10 overshoots by +1.599); the plan's mean |kappa| is 0.1845 / 0.2621 against 0.0135 for GT, with a signed error near 0 (+0.0037 / −0.0020): *"the excess is magnitude without a sign: jitter, not a steady bias"*. POST-HOC (not in the card, decides nothing): 9.0 % / 11.4 % of DAG-OACC's 3 to 8 m/s windows have a plan step below 0.5 m/s, against 0.7 % for P10; the pooled v0 < 8 m/s view carries curvature +0.880 [+0.786, +0.950] / +0.893 [+0.815, +0.952] of the M gap and ADE6 only +0.219 / +0.208. **[E]** D2 arithmetic with more windows zeroed:

<<T:GDM_6B>>

  **[H]** (untested): the 3 to 8 m/s defect is an intermittent slow-dip tail in the plan, where heading noise over a short step becomes large curvature and yaw-rate and where the plan falls below the 5.2 m/s line that M1 and L2 count.
* **Controls [M].** Run before any verdict was read:

<<T:GDM_CTRL>>

* **The four families.** Rows read: ADE6 (PRIMARY), yaw-rate, curvature and L2 curvature (LATERAL), M1 (the crawl row). LONGITUDINAL, TACTICAL and STRATEGIC rows are NOT RUN as a split (the card decomposes five rows); only the D2 arithmetic above spans all 19 rows (ESTIMATED). STRATEGIC is reported, not paused, for v8F.
* **Next, as committed.** *"In the 3 to 8 m/s windows replace each DAG-OACC plan step below a floor by the previous step's speed and heading (a pure post-processing of the executed track)"* and recompute the four rows against P10 (the dip-floor oracle), and pre-register the pooled v0 < 8 m/s curvature hypothesis; *"No lever is blocked: everything above is CPU on banked arrays and dumps."*

**7.34.3 SMOOTH-T1: the SG(7, 2) smoother moves SOUP and leaves DAG-OACC NO EFFECT (`H-SMOOTH-T1-1`; QUOTABLE ON Y', not DOMINANT, no HARMS).**

* **Status [M].** Two reads, both landed `0a5709a` (2026-11-02 15:57:02): `R/raw/2026-11-02-smooth-t1/RESULTS.md` (the first read) and `RESULTS_A1.md` (the decision read, under `AMENDMENT_1.md`). The first read ran the rollouts at 1 thread against banked rollouts made at the default 16 threads, and the card's own controls refused it: *"The outcome is REFUSED by the card's own rule"* (K1, K2 and the floor clause of K5 fail). `AMENDMENT_1.md` (sha256 `{{C:SMA}}…`, control-only, frozen 2026-11-02T15:41:09+0100 before any 16-thread number was read, disclosed after the 1-thread tables had been seen) makes fresh unsmoothed runs at 16 threads, Y', the decision baseline. The status word is QUOTABLE ON Y' (the banked-reference comparisons exceed the measured tolerance; the decision contrast is X − Y').
* **Setup [M].** Card `PREREG_FROZEN.md` sha256 `{{C:SM}}…`, frozen 2026-11-02T14:40:42+0100 before any smoothed number. The lever is the pre-registered SG(7, 2) residual smoother S (`H-V8F-SMOOTH-1`, PASS at T0 on 2026-10-20) applied to the plan at every replan inside the rollout; arms DAG-OACC and SOUP at s0 and s1. Outcome rule, precedence DOMINANT > HARMS > MOVED > NO EFFECT: DOMINANT is Read A against P10 at both seeds; MOVED is every crawl row (yaw-rate MAE, curvature MAE, L2 curvature MAE, M1) separated-better against Y' at both seeds with ADE6 not worse than +0.0604 m; HARMS is a separated-worse non-crawl row (15 rows x 2 seeds = 30 tests per arm, the count printed beside the verdict). T1-kin; estimator as in the status bullet.
* **Outcome [M].**

<<T:SM_HEAD>>

<<T:SM_RULE>>

  Neither arm is DOMINANT and none HARMS: 0 of 19 rows are separated-worse against Y' in any run. ADE6 X − Y' is separated-better (−0.0544 / −0.0671 m for DAG-OACC, −0.0687 / −0.0558 m for SOUP). Read A against P10 passes D1 (ADE6 −1.2556 / −1.1903 / −1.1928 / −1.1895 m) and D3 and fails D2 at all four cells; the condition-by-condition table:

<<T:SM_COND>>

  The four crawl rows and ADE6, X − Y' (decides), X − P10 and the rig-noise column Y' − Y (banked):

<<T:SM_CRAWL>>

  **[I]** from the register row: the post-hoc seam diagnostic in the first read read what still blocks D2 as *"a near-uniform per-knot curvature overshoot over P10, which a within-plan smoother does not act on"*.
* **Controls [M].**

<<T:SM_CTRL>>

  K1', K2' and the K5' floor clause fail at the measured tolerance (the banked reference differs from a fresh run by up to 1.71e-02 m on the floor); K2'' passes, and Y' against Y'' (the same configuration run twice at 16 threads) is bit-identical, with 0 of 7,197 ticks differing above 1e-4 m, which is why the tolerances stand. The tolerance comes from one replicate pair.
* **The four families (STRATEGIC reported, not paused, for v8F).** Rows by flag (B separated-better, W separated-worse, last number not separated), DAG-OACC:

<<T:SM_FAM>>

  LONGITUDINAL (6 rows) reads 4 / 0 / 2 against P10 at s0 and 3 / 0 / 3 against Y'; LATERAL (6) reads 3 / 2 / 1 against P10 at s0 and 5 / 0 / 1 against Y'; TACTICAL (the plan-derived TAC-LAT and TAC-LON agreements, 2 rows) and STRATEGIC (S-POS@30, @50, @80) read all separated-better against P10 and against Y' at s0. The tactical goal-head F1 is NOT COMPUTABLE at T1-kin (**INHERITED** from §7.32).
* **Next, as committed.** For DAG-OACC NO EFFECT the card commits a curvature-bounded decode *"as the next zero-training arm, wired the same way as a flag of the same tool, with its own pre-registration frozen before data"*, and for SOUP MOVED *"the cheapest next step is S composed with the curvature-bounded decode"*: KAPPA-T1 below. The patched tool (`--smooth`, default off) is banked in the package, not in `stack/`. Limits: the amendment moved the reference of the controls and the Y side after the 1-thread tables were seen.

**7.34.4 Trigger probe 5: native-scale crops of the traffic-light and lead bands do not help (`H-CRAWL-06-PROBE5`; T0-class).**

* **Status and setup [M].** Read 2026-11-02; `R/raw/2026-11-02-trigger-probe5/RESULTS.md`, landed `5efeffb` (2026-11-02 16:26:15). Card `PREREG_FROZEN.md` sha256 `{{C:P5}}…`, frozen 2026-11-02T14:03:08+0100 before any crop feature, label statistic or probe output. Question: do DINOv3-L/16 features at native patch resolution on crops of the traffic-light band (rows [0,336) cols [384,1536), 21x72 = 1,512 tokens) and the lead-vehicle band (rows [272,608) cols [480,1440), 21x60 = 1,260 tokens), cut from the 768x1920 frame on box 2, carry more go/stay trigger than probe 1's whole-vector V+E? Same 12,000 + 12,000 TRAIN windows as probes 1 to 4 (P-STOP 849 clips, P-MOVE 2,140 clips), PCA-64 and an L2 logistic readout. Tier: T0-class probe of representations on logged inputs; it measures what a probe can decode from stored features, never what the planner does. Estimator: episode-cluster bootstrap on the mean of per-fold AUCs, B = 2,000, seed 0, paired; it has no seed spread (one PCA seed, one frozen network), so another training run and another PCA draw are not answered (`H-ESTIM-SEED-1`). The committed rule: *"CROPS HELP iff dAUC >= 0.03 with CI lower > 0 in either population (P-STOP, P-MOVE); CROPS DO NOT HELP iff dAUC < 0.01 in both populations"*, otherwise INCONCLUSIVE.
* **Outcome [M].**

<<T:P5_VERD>>

  Both cells are valid and below 0.01: **CROPS DO NOT HELP**. The arm table (fold-mean AUC) and the paired contrasts beside it:

<<T:P5_ARMS>>

<<T:P5_PAIR>>

  Each crop carries ego-independent signal (LEAD+E reaches V+E on P-MOVE, +0.053 over E) that the whole-frame V already carries; neither crop arm passes V+E and stacking the crops on V+E adds nothing. On P-STOP the 200-feature arm reads below V+E, and the contrast includes the cost of 128 more features, so it is not a statement that the crops carry anti-signal.
* **Controls and disclosures [M].** The align gate PASSES (per-row relative L2 median 0.0264 against a gate of 0.05); `controls_ok` is true in both populations:

<<T:P5_CTRL>>

  P-MOVE C2 LEAD+E reads 0.468 [0.446, 0.489] at seed 2, so seeds 3 to 6 were run and the 5-draw mean is 0.4962, inside the committed band [0.47, 0.53]. Disclosed: the crops travelled as JPEG q95 and the features move by a median of 6.4 % (max 0.1228), not gated; it is label-independent and can only add noise, which biases toward NOT HELP. The card lists a pixel test that no test in the test file implements, so that item was not run: **UNVERIFIED**. The grid edge C = 0.001 was selected in every outer fold for C+E and C+V+E.
* **The four families.** A representation probe, not an evaluation of a plan: LONGITUDINAL, LATERAL, TACTICAL and STRATEGIC rows are NOT RUN (no plan is scored), and the probe says nothing about any of the four. STRATEGIC is reported, not paused, for v8F.
* **Reading and next.** With probes 1 to 4, *"The frozen DINOv3-L/16 caps the go/stay trigger at about V+E."* (P-STOP 0.680, P-MOVE 0.785; ego alone 0.626 / 0.732), and GAPDECOMP places no DAG-OACC crawl row as TRIGGER-BOUND, so the trigger is not the lever for DOMINANT. What the RESULTS names as left: *"a fine-tuned or different encoder (the encoder is the part of the pipeline probes 1-5 could not vary), or a dedicated traffic-light-state input"*, each an architecture change and GPU work, hence a PI decision; none is started. Limits: one readout (PCA-64 plus L2 linear) on one frame, so a NOT HELP says this readout of this frozen encoder at 3x scale does not carry the trigger.

**7.34.5 KAPPA-T1: the curvature-bounded decode after S reads HARMS (`H-KAPPA-T1-1`).**

* **Status and setup [M].** Read 2026-11-02 16:19:43; `R/raw/2026-11-02-kappa-t1/RESULTS.md` and `GATE_RESULT.md`, landed `992d736` (2026-11-02 16:28:19). Card `PREREG_FROZEN.md` sha256 `{{C:KP}}…`, frozen 2026-11-02T15:44:58+0100 before any gate number or lever rollout. Part 1 is the G-KAPPA gate, an exact point decomposition of the curvature excess with no estimator, which picks the lever; Part 2 is the rollout: X = S plus `project_feasible(a_max = 4.0, kappa_max = 0.2, clamp_entry = False)` at every replan, Y = the fresh S rerun, DAG-OACC and SOUP at s0 and s1. Committed outcomes: DOMINANT; MOVED (the crawl rows separated-better against Y without HARMS: ADE6 X − Y not above +0.0604 m and none of the 15 non-crawl rows separated-worse); HARMS; NO EFFECT, precedence in that order. The extractor's 1e-2 m floor guard refused the first read; `READER_FIX_FLOOR.md` (written before any X number) applies the floor and window check against the fresh Y, which the card already defines as the baseline, with the outcome rule unchanged. T1-kin; estimator as in the status bullet.
* **The gate [M].** Lever (a) is selected where s_HIGHK is at least 0.5 at both DAG-OACC seeds:

<<T:KP_GATE>>

* **Outcome [M].**

<<T:KP_OUT>>

  **HARMS for DAG-OACC and SOUP at both seeds.** The box closes the curvature, L2 curvature and yaw-rate excess against Y and fails on position and heading rows: heading MAE X − Y +1.6946 / +1.3399 deg (DAG-OACC), +1.6201 / +1.4407 deg (SOUP); S-POS@30 / @50 / @80 separated-worse at s0 for both arms; at DAG-OACC s1 ADE6 X − Y +0.0606 m, above the +0.0604 m guard, with along-track as well. D2 against P10 fails only on M1 crawl share, at all four arm-seeds. The card's prediction (ESTIMATED, written before any number) and the share of each Y − P10 excess that X − Y closes:

<<T:KP_PRED>>

  M1 moves the wrong way (−10 % and −13 % of its excess at DAG-OACC), and some ticks' issued plans move by 105 to 227 m (register row).
* **Controls [M].** All five known answers PASS (KA1 Y reproduces `out_m16`, KA2 the issued plan is the box of S, KF the floor and window check against the fresh Y, K3, KB1); KA1:

<<T:KP_KA1>>

* **The four families (DAG-OACC s0; STRATEGIC reported, not paused, for v8F).**

<<T:KP_FAM>>

  LONGITUDINAL: speed MAE and accel MAE separated-better against Y, along-track not separated. LATERAL: heading MAE separated-worse (+1.6946 [+0.3667, +3.3224]), yaw-rate, curvature, L2 and L1 separated-better. TACTICAL: the plan-derived TAC-LAT agreement separated-better (+0.0142 [+0.0029, +0.0259]); the goal-head F1 is NOT COMPUTABLE at T1-kin (**INHERITED** from §7.32). STRATEGIC: S-POS@30 / @50 / @80 +0.4995 / +0.7836 / +1.0942, all separated-worse.
* **Reading and next.** **[H]** (the card's, not tested here): the acceleration clamp accounts for M1, along-track and ADE6 and the curvature clamp for curvature, yaw-rate, heading and S-POS; FEAS2 tests it. Committed next for HARMS: *"as a further flag value of the same tool, to attribute the harm to the accel clamp or the curvature clamp"*, the kappa-only projection (`a_max` = inf), with its own frozen card.

**7.34.6 FEAS2: the box split into its clamps (kbox, abox, kabox) reads HARMS for every lever on both bases (`H-FEAS2-T1-1`).**

* **Status and setup [M].** Read 2026-11-02 17:20:31; `R/raw/2026-11-02-feas2-t1/RESULTS.md`, landed `8e1a40d` (2026-11-02 18:44:44). Card `PREREG_FROZEN.md` sha256 `{{C:F2}}…`, frozen 2026-11-02T16:43:48+0100 before any X number; a per-tick pre-check on KAPPA-T1's seen ticks is disclosed in the card (it attributed the heading harm to the curvature clamp). Levers after S: kbox (kappa-only, a_max = inf), abox (accel-bounded longitudinal decode with entry clamp, aimed at M1) and kabox (both); DAG-OACC and SOUP at s0 and s1, Y = KAPPA-T1's fresh S run, 12 arms. Same outcome rule and guard as KAPPA-T1. T1-kin; estimator as in the status bullet.
* **Outcome [M].**

<<T:F2_OUT>>

  6 of 6 lever-arm outcomes read HARMS; D1 holds at 12 of 12 cells, D3 at 12 of 12, and D2 fails at 12 of 12. Failing D2 rows per lever (cells of 4):

<<T:F2_D2>>

  The card's probabilities (ESTIMATED) and M1:

<<T:F2_PRED>>

<<T:F2_M1>>

  M1 crawl share fails D2 at 12 of 12 cells (the card gave 85 % that it fails under all three levers) and is separated-better under no lever. kbox and kabox read HARMS as predicted; abox, predicted NO EFFECT at 65 %, reads HARMS at both arms (25 %), separated-worse on heading MAE and S-POS@30 / @50 / @80 X − Y at every cell, and closes no crawl row. kbox ADE6 X − Y is +0.0753 m at DAG-OACC s1, beyond the +0.0604 m guard.
* **The pre-check against the rollout [M].** Heading MAE proxy delta at the plan level against the rollout row (X − Y):

<<T:F2_PRE>>

  The pre-check located the heading harm at the curvature clamp and read abox as improving heading; the rollouts agree for kbox and kabox and not for abox. *"the rollout M1 row above reads the executed rows, where the excess lives."* At the plan level the crawl was already below the P10 prior's share in all four cells, so the executed state, not the issued plan, carries it. **[H]** (untested): the entry bound and the accel clamp perturb the executed first knots.
* **Controls [M].** All six control groups PASS (KY, KA1, KA2, KF, K3, KB1); KY: 8 of 8 reused-Y files hash-equal to the pins.
* **The four families (kbox, DAG-OACC s0; STRATEGIC reported, not paused, for v8F).**

<<T:F2_FAM>>

  Read per family at every cell in the RESULTS file (12 per-cell tables); the harms sit in LATERAL (heading), STRATEGIC (S-POS@30 / @50 / @80) and, at some cells, LONGITUDINAL (along-track) and ADE6, and the crawl row M1 is separated-worse against P10 at 12 of 12. The tactical goal-head F1 is NOT COMPUTABLE at T1-kin (**INHERITED** from §7.32).
* **Next, as committed.** kbox2 *"is aimed at the heading / S-POS harm (the curvature cap stops acting below 2 m/s)"*, lonfloor at M1, and their composition kbox2lf; abox is not carried forward, since *"the card's committed next arms do not act on abox's own harm, so abox is not carried into FEAS3"*. Limits as in the status bullet; adoption into any K-1 / WM-1 claim is not decided here.

**7.34.7 FEAS3: kbox2, lonfloor and kbox2lf read HARMS; lonfloor moves M1 separated-better and closes at most 6.6 % of its excess (`H-FEAS3-T1-1`).**

* **Status and setup [M].** Read 2026-11-02 18:21:23 (the re-run read); `R/raw/2026-11-02-feas2-t1/RESULTS_FEAS3.md`, landed `8e1a40d` (2026-11-02 18:44:44, with FEAS2). Card `PREREG_FEAS3_FROZEN.md` sha256 `{{C:F3}}…`, frozen 2026-11-02T17:43:04+0100 before any X number. Levers after S: kbox2 (the kappa clamp with min_speed 2.0), lonfloor (the executed step speed floored at the P10 prior's step speed minus 2 m/s) and kbox2lf (both); DAG-OACC and SOUP at s0 and s1, Y = KAPPA-T1's fresh S runs, reused (pins in `FROZEN_SHA_FEAS3.txt`). The first read stopped at rc 1 inside the card's beside checks on a stack-shadow guard; `READER_FIX_FEAS3.md` records an import-path-only fix (the operator file is byte-identical in both trees), and the 12 per-cell outcome lines of the re-run equal the first run's (12 of 12 identical). T1-kin; estimator as in the status bullet.
* **Outcome [M].**

<<T:F3_OUT>>

  6 of 6 lever-arm outcomes read HARMS; D1 holds at 12 of 12 cells, D3 at 12 of 12, and D2 fails at 12 of 12, on M1 crawl share at 12 of 12. Failing D2 rows and HARMS triggers per lever (cells of 4):

<<T:F3_D2>>

<<T:F3_HARMS>>

  kbox2 ADE6 X − Y is +0.1283 / +0.1442 m (DAG-OACC) and +0.1342 / +0.1287 m (SOUP), beyond the guard at 4 of 4 cells, with heading MAE and all three S-POS rows separated-worse. The one MOVED condition hidden by precedence is lonfloor on SOUP: at both seeds all four crawl rows are separated-better on X − Y and ADE6 X − Y is within the guard (−0.0004; +0.0104 m), and HARMS wins because a non-crawl row is separated-worse (accel MAE and cross-track MAE at s0; L1 lateral-accel MAE at s1). The card's probabilities (ESTIMATED) and M1:

<<T:F3_PRED>>

<<T:F3_M1>>

  M1 crawl share fails D2 at 12 of 12 cells and no arm is DOMINANT (0 of 6; the card gave 4 %). lonfloor, predicted NO EFFECT at 85 %, reads HARMS at both arms (the card gave 12 %); its M1 X − Y is separated-better at 4 of 4 cells and closes 2.0 to 6.6 % of the excess (mean 4.3 %).
* **Disclosure (i), the M1 margins [M].** The M1 row is the share of GT-fast pairs whose own pair speed is below 5.2 m/s. At a 1, 2 and 3 m/s margin instead, the effect is not a threshold artefact (CTRL s0: row definition Y 0.0713 against P10 0.0230; at a 1 m/s margin 0.1786 against 0.0480). Cells at which M1 X − Y is separated-better, per margin:

<<T:F3_SB>>

  The card's THRESHOLD-LOCAL reading, taken literally, holds at no cell; under the broader reading it holds at lonfloor DAG-OACC s1 and kbox2lf DAG-OACC s0; the outcome words do not change either way. **Disclosure (ii), the hard-braking windows:** 120 windows from 19 episodes (top 5 % of GT peak deceleration, threshold 2.372 m/s^2); with 19 episodes the intervals are wide, and *"a null here is not evidence of safety"*.
* **Controls [M].** All six control groups PASS (KY 8 of 8 files hash-equal to the pins, KA1, KA2, KF, K3, KB1).
* **The four families (lonfloor, SOUP s0, the cell where the MOVED condition holds; STRATEGIC reported, not paused, for v8F).**

<<T:F3_FAM>>

  The other eleven cell tables are in the RESULTS file. The tactical goal-head F1 is NOT COMPUTABLE at T1-kin (**INHERITED** from §7.32).
* **Next, as committed.** *"the decode-time box family (box, kbox, abox, kabox, kbox2) is closed as a heading / S-POS lever (the clamp acts on executed steps the logged future truly turns on)"*. Remaining levers and what blocks each: (c) the lateral shrink `P + diag(1, alpha)(S(plan) - P)` with alpha chosen on TRAIN windows is blocked because the tool rolls out eval-141 only (whether DAgger's TRAIN-window rollout inputs exist on disk is **UNVERIFIED**); the clamp as a loss term or output layer at training time is blocked on the PI's go and compute. The card committed no line for HARMS at lonfloor; the executed-state question was taken up by the separate package `../histswap`, which this version does not read.
* **What this section cannot claim.** (1) A separated interval is necessary, not sufficient, for a lever claim (`H-ESTIM-SEED-1`): every contrast here has one training run per seed and no replicate arm, and an effect smaller than the rig's run-to-run spread is not an effect. (2) The six rollout reads use the same 141 eval episodes, and each lever was chosen after the previous reads were seen (each card frozen before its own numbers, not independent of the earlier ones). (3) T1-kin applies the model's own plan to its own state with logged cameras; nothing here is a statement about re-perception, and no T2 number exists. (4) The P10 side is the banked one (SMOOTH-T1's reference choice); the fresh floor differs from it by up to 1.71e-02 m. (5) The GAPDECOMP shares are descriptive decompositions, not interventions. (6) HISTSWAP, HISTSWAP-LF, FRAMEFREEZE, RECOVER-FT and RECOVER-NEXT are read in v36; no claim about their states is made here (**UNVERIFIED** in v35).
* **Next (Rule Zero).** The committed arms of the seven reads are run, and none clears its bar; the remaining levers and their blockers are those named above: the dip-floor oracle and the pooled v0 < 8 m/s hypothesis (zero training, CPU, GAPDECOMP-M), the lateral shrink (needs TRAIN-window rollout inputs, UNVERIFIED), a training-time clamp (the PI's go and compute), and a perception change at the encoder or a traffic-light-state input (architecture, a PI decision). The executed-state question is the next read (v36).
* **Provenance [M].** All under `R/raw/` at the tip `{{TP}}`: `2026-11-02-gapdecomp/RESULTS.md` (sha256 `{{S:GD}}…`), `2026-11-02-gapdecomp-m/RESULTS.md` (`{{S:GDM}}…`), `2026-11-02-smooth-t1/RESULTS_A1.md` (`{{S:SM}}…`) and `RESULTS.md` (`{{S:SM1}}…`), `2026-11-02-trigger-probe5/RESULTS.md` (`{{S:P5}}…`), `2026-11-02-kappa-t1/RESULTS.md` (`{{S:KP}}…`) and `GATE_RESULT.md` (`{{S:KPG}}…`), `2026-11-02-feas2-t1/RESULTS.md` (`{{S:F2}}…`) and `RESULTS_FEAS3.md` (`{{S:F3}}…`); claim rows `H-GAPDECOMP-1`, `H-GAPDECOMP-M-1`, `H-SMOOTH-T1-1`, `H-CRAWL-06-PROBE5`, `H-KAPPA-T1-1`, `H-FEAS2-T1-1`, `H-FEAS3-T1-1` in `Project Steering/GOALS_AND_CLAIMS.md`. The build is `patch_v35.py` (anchored edits on the landed v34 blobs; every table copied line for line from its RESULTS and asserted to be a source line), checked by `checks_v35.py` and re-measured by `audit_v35.py`.
"""
SEC = SEC_734
SEC = SEC.replace("{{TP}}", TP)
for _k in CARD_SHA:
    SEC = SEC.replace("{{C:%s}}" % _k, CARD_SHA[_k][:8])
for _k in SRC:
    SEC = SEC.replace("{{S:%s}}" % _k, sha(SRC[_k])[:8])
for _k, _rows in T.items():
    assert SEC.count("<<T:%s>>" % _k) == 1, _k
    SEC = SEC.replace("<<T:%s>>" % _k, "\n".join(_rows))
assert "<<T:" not in SEC and "{{" not in SEC, re.findall(r"(<<T:\w+>>|\{\{\S+\}\})", SEC)[:3]
SEC = SEC.replace("\r", "")

# ---- the abstract, the status line and the edits -----------------------------------------------------------------------------------------------------------
OLD_ST = ("research report, v34 of 2026-11-03 (pre-draft 2026-10-29), built at repository tip `870fa33a`; results through the trigger probe 3 and 4 reads (probe 4 landed `70e2affe`); the reads of 2026-11-02 and 2026-11-03"
          " (probe 5 to RECOVER-NEXT) are not covered here and come in v35 (v33: `8c1e9636`;")
NEW_ST = ("research report, v35 of 2026-11-03, results through repository tip `" + TP + "` and, within it, through the 2026-11-02 crawl-diagnosis and decode-lever reads (§7.34: GAPDECOMP, GAPDECOMP-M, SMOOTH-T1, trigger probe 5,"
          " KAPPA-T1, FEAS2 and FEAS3); the later reads (HISTSWAP, HISTSWAP-LF, FRAMEFREEZE, RECOVER-FT, RECOVER-NEXT) are not covered here and come in v36 (v34: `870fa33a`, results through the trigger probe 3 and 4 reads; v33: `8c1e9636`;")
OLD_ST2 = "Probes 3 and 4 landed at `d4cf54d9` and `70e2affe`; the tip named above also holds later reads (trigger probe 5, landed `5efeffbe`, among them) that this report does not cover."
NEW_ST2 = ("Probes 3 and 4 landed at `d4cf54d9` and `70e2affe`. v35 adds the crawl located and the decode levers (§7.34, T1-kin, MEASURED, zero training): the gap against P10 is not trigger-bound for DAG-OACC (GAPDECOMP), the class-M gap is"
           " localised to v0 3 to 8 m/s (yaw-rate +0.653 / +0.684, L2 +0.934 / +0.915, M1 +0.792 / +0.751; GAPDECOMP-M), S moves SOUP and leaves DAG-OACC NO EFFECT (SMOOTH-T1), native-scale crops do not help the go/stay trigger (probe 5, T0-class),"
           " and the curvature box, kbox / abox / kabox and kbox2 / lonfloor / kbox2lf all read HARMS with M1 failing D2 at every cell (KAPPA-T1, FEAS2, FEAS3).")
OLD_ABS_ANCHOR = "* **What is next.** The crawl set is still open and the adopted base stays DAG-OACC."
NEW_ABS = (
    "* **The M1 crawl was located, and no decode lever closes it (§7.34; seven zero-training reads of 2026-11-02, T1-kin except probe 5, which is a T0-class probe; paired episode-cluster bootstrap, B = 2,000, seed 0, eval-141, DAG-OACC and SOUP at s0 and s1; **[M]**).**"
    " D1 and D3 of Read A against P10 hold in every cell of every lever read after S, and D2 fails on the M1 crawl share at 4 of 4 cells (curvature box), 12 of 12 (FEAS2) and 12 of 12 (FEAS3). The gap is not in the go/stay trigger windows (15.0 % of windows;"
    " no DAG-OACC crawl row is TRIGGER-BOUND, M1 +0.606 / +0.545 MIXED) but in the moving windows at v0 3 to 8 m/s (20.3 % of class M; yaw-rate +0.653 / +0.684, L2 +0.934 / +0.915, M1 +0.792 / +0.751 of the M gap), as curvature jitter and slow plan steps"
    " (plan mean |kappa| 0.1845 / 0.2621 against P10 0.0623; plan speed only +0.398 / +0.241 m/s above GT; 9.0 / 11.4 % of windows with a plan step below 0.5 m/s against 0.7 % for P10), and path shape localises nothing. The pre-registered smoother S moves"
    " SOUP and leaves DAG-OACC NO EFFECT (ADE6 X − Y' −0.0544 / −0.0671 m, 0 of 19 rows separated-worse); the curvature box reads HARMS (heading MAE X − Y +1.6946 / +1.3399 deg at DAG-OACC); kbox, abox and kabox read HARMS (6 of 6), and so do kbox2, lonfloor and kbox2lf (6 of 6;"
    " kbox2 ADE6 X − Y +0.1283 / +0.1442 m at DAG-OACC); lonfloor moves M1 separated-better at 4 of 4 cells and closes 2.0 to 6.6 % of its excess. Native-scale crops add nothing to the go/stay trigger (C+V+E − V+E −0.0231 [−0.0422, −0.0050] on P-STOP,"
    " −0.0028 [−0.0081, +0.0024] on P-MOVE). The cause is upstream of the decode; the executed-state reads (HISTSWAP and later) come in v36.\n")
OLD_811 = "\n\n### 8.2 What is not established"
NEW_811 = (
    "\n22. **The M1 crawl sits in the executed speed of the moving 3 to 8 m/s windows, and none of S, the kappa bound, kbox / abox / kabox or kbox2 / lonfloor / kbox2lf closes it** (§7.34; T1-kin, paired episode-cluster bootstrap, B = 2,000, seed 0, eval-141, one training run per seed;"
    " probe 5 is T0-class). **Established (MEASURED):** no DAG-OACC crawl row is TRIGGER-BOUND (M1 +0.606 / +0.545 MIXED) and with the trigger windows' gap set to 0 four rows stay separated-worse (ESTIMATED); the class-M gap is LOCALISED to v0 3 to 8 m/s"
    " (yaw-rate +0.653 / +0.684, L2 +0.934 / +0.915, M1 +0.792 / +0.751) and path shape localises nothing (TURN +0.100 / +0.084 of yaw-rate with 0.074 of the windows); S reads NO EFFECT at DAG-OACC and MOVED at SOUP with 0 of 19 rows separated-worse against Y';"
    " the curvature box, kbox / abox / kabox and kbox2 / lonfloor / kbox2lf read HARMS in 6 of 6 and 6 of 6 lever-arm outcomes and at both arms of the box, with M1 failing D2 at 4 of 4, 12 of 12 and 12 of 12 cells; crops of the two bands do not lift the trigger"
    " (C+V+E − V+E −0.0231 [−0.0422, −0.0050] on P-STOP). **Not established:** that a lever on the model's executed state or a training-time clamp would close M1 (the HISTSWAP reads come in v36; the lateral shrink and the training-time clamp are blocked, §7.34),"
    " that the 3 to 8 m/s dip tail is the cause (**[H]**, untested), and any statement about re-perception (T1-kin only; no T2 number exists). One training run per seed, no replicate arm (`H-ESTIM-SEED-1`); the same 141 episodes serve all six rollout reads."
    "\n\n### 8.2 What is not established")
OLD_83 = "**Update (v34).** Four items are written up here, and the adopted base is unchanged (DAG-OACC):"
NEW_83 = (
    "**Update (v35).** Three items are written up here, and the adopted base is unchanged (DAG-OACC):\n"
    "* **The crawl was located (§7.34, GAPDECOMP and GAPDECOMP-M, T1-kin):** not in the go/stay trigger windows (no DAG-OACC crawl row TRIGGER-BOUND; M1 +0.606 / +0.545) but at v0 3 to 8 m/s (yaw-rate +0.653 / +0.684, L2 +0.934 / +0.915, M1 +0.792 / +0.751 of the M gap),"
    " as curvature jitter (plan mean |kappa| 0.1845 / 0.2621 against P10 0.0623) and slow steps (9.0 / 11.4 % of windows against 0.7 %), with the plan speed +0.398 / +0.241 m/s above GT. Native-scale crops add nothing to the trigger (probe 5, T0-class).\n"
    "* **S and the decode levers (§7.34, SMOOTH-T1, KAPPA-T1, FEAS2, FEAS3, T1-kin):** S moves SOUP and leaves DAG-OACC NO EFFECT; the curvature box, kbox / abox / kabox and kbox2 / lonfloor / kbox2lf all read HARMS (6 of 6 and 6 of 6 outcomes for the last two sets),"
    " D1 and D3 hold everywhere and M1 fails D2 at every cell (4 of 4, 12 of 12, 12 of 12). The best M1 mover, lonfloor, closes 2.0 to 6.6 % of its excess.\n"
    "* **What is blocked:** the lateral shrink (needs TRAIN-window rollout inputs, UNVERIFIED), a training-time clamp (the PI's go and compute) and a perception change (architecture, a PI decision). The executed-state reads come in v36.\n\n"
    "**Update (v34).** Four items are written up here, and the adopted base is unchanged (DAG-OACC):")
OLD_ROWS = "| 2 | **LoRA on DINOv3**"
NEW_ROWS = (
    "| 1ag | **GAPDECOMP** (zero training, dev-box CPU; `H-GAPDECOMP-1`) | card `d5c9caa8…`, frozen 2026-11-02 14:07:12: a crawl row is TRIGGER-BOUND iff the trigger windows carry ≥ 0.7 of its X − P10 gap with CI lower > 0.5 at both seeds, NOT iff ≤ 0.3 with CI upper < 0.5, else MIXED |"
    " **Read** (§7.34.1, T1-kin, MEASURED): NOT TRIGGER-BOUND for DAG-OACC (no crawl row; curvature, L2, M1 MIXED); SOUP L2 TRIGGER-BOUND at both seeds. Next: the split of class M (read as 1ah) |\n"
    "| 1ah | **GAPDECOMP-M** (zero training; `H-GAPDECOMP-M-1`) | card `1c5dbaf0…`, frozen 2026-11-02 14:19:53: LOCALISED iff one subclass with ≤ 30 % of M's windows carries ≥ 0.5 of the row's M gap with CI lower > 0.3 at both seeds, else DIFFUSE |"
    " **Read** (§7.34.2, T1-kin): LOCALISED to v0 3 to 8 m/s on yaw-rate, L2, M1; curvature MAE and ADE6 DIFFUSE; path shape localises nothing. Next: the dip-floor oracle on the dumps, the pooled v0 < 8 m/s hypothesis (zero training, CPU) |\n"
    "| 1ai | **SMOOTH-T1** (zero training; `H-SMOOTH-T1-1`) | card `4030d22f…`, frozen 2026-11-02 14:40:42, Amendment 1 `717ff177…` (control-only): DOMINANT, or MOVED (every crawl row separated-better against Y' at both seeds, ADE6 ≤ +0.0604 m), HARMS, NO EFFECT |"
    " **Read** (§7.34.3, T1-kin, QUOTABLE ON Y'): SOUP MOVED, DAG-OACC NO EFFECT; not DOMINANT, 0 of 19 rows separated-worse; D2 fails on curvature, L2, M1. The first 1-thread read was REFUSED by the card's controls. Next: the curvature-bounded decode (read as 1ak) |\n"
    "| 1aj | **Trigger probe 5** (zero training, box-2 encode; `H-CRAWL-06-PROBE5`) | card `791e8a62…`, frozen 2026-11-02 14:03:08: CROPS HELP iff dAUC(C+V+E − V+E) ≥ 0.03 with CI lower > 0 in either population; DO NOT HELP iff < 0.01 in both |"
    " **Read** (§7.34.4, T0-class): CROPS DO NOT HELP (P-STOP −0.0231 [−0.0422, −0.0050], P-MOVE −0.0028 [−0.0081, +0.0024]), controls PASS. Next: a fine-tuned or different encoder or a traffic-light-state input (architecture, a PI decision) |\n"
    "| 1ak | **KAPPA-T1** (zero training; `H-KAPPA-T1-1`) | card `9564d568…`, frozen 2026-11-02 15:44:58: the box after S (a_max 4.0, kappa_max 0.2): DOMINANT, or MOVED against Y without HARMS (ADE6 ≤ +0.0604 m, none of 15 non-crawl rows separated-worse) |"
    " **Read** (§7.34.5, T1-kin): HARMS at both arms, both seeds (heading MAE X − Y +1.6946 / +1.3399 deg at DAG-OACC); D2 fails only on M1. Next: the kappa-only projection (read as 1al) |\n"
    "| 1al | **FEAS2** (zero training; `H-FEAS2-T1-1`) | card `ecf9bb5d…`, frozen 2026-11-02 16:43:48: kbox, abox, kabox after S; DOMINANT or MOVED without HARMS |"
    " **Read** (§7.34.6, T1-kin): HARMS 6 of 6; M1 fails D2 at 12 of 12; abox closes no crawl row. Next: kbox2, lonfloor and kbox2lf (read as 1am) |\n"
    "| 1am | **FEAS3** (zero training; `H-FEAS3-T1-1`) | card `e817abcf…`, frozen 2026-11-02 17:43:04: kbox2, lonfloor, kbox2lf after S; DOMINANT or MOVED without HARMS |"
    " **Read** (§7.34.7, T1-kin): HARMS 6 of 6 (kbox2 ADE6 X − Y +0.1283 / +0.1442 m at DAG-OACC); lonfloor M1 separated-better at 4 of 4 cells, 2.0 to 6.6 % of its excess; M1 fails D2 at 12 of 12; 0 of 6 DOMINANT."
    " Next: the executed-state question (HISTSWAP, v36); the lateral shrink and a training-time clamp are blocked (UNVERIFIED inputs; the PI's go) |\n"
    "| 2 | **LoRA on DINOv3**")
OLD_9 = "\n\n---\n\n## 10. Reproducibility appendix"
NEW_9 = (
    "\n13. **v35 reads and their limits.** The seven reads of §7.34 use one set of 141 eval episodes (2,399 windows) for six rollout reads, each lever chosen after the earlier reads were seen; one training run per seed (DAG-OACC and SOUP at s0 and s1), no replicate arm,"
    " and an interval that answers another draw of episodes only (`H-ESTIM-SEED-1`). The committed outcome rules of the cards are quoted from the register rows and the RESULTS files; the card texts were hashed (sha256) but not re-read in v35 (**UNVERIFIED** against the card text). SMOOTH-T1's decision baseline is the fresh unsmoothed run Y' under a control-only amendment written after the 1-thread tables had been seen, with tolerances from one replicate pair."
    " The GAPDECOMP shares are decompositions of a measured gap, not interventions, and the yaw-rate share is unstable (s0 −0.424 against s1 +0.205). Probe 5 is a T0-class readout (one PCA-64 and L2 linear readout on one frame of one frozen encoder) with a JPEG transport step"
    " (median 6.4 % feature change) that was not gated, and one card item (a pixel test) was not run (UNVERIFIED). The tactical goal-head F1 is not computed at T1-kin and the family reads of the rollout studies are the plan-derived rows. HISTSWAP, HISTSWAP-LF, FRAMEFREEZE,"
    " RECOVER-FT and RECOVER-NEXT are not read in v35 (v36)."
    + "\n\n---\n\n## 10. Reproducibility appendix")
ANCHOR_8 = "\n\n---\n\n## 8. Discussion"
E = [(OLD_ST, NEW_ST), (OLD_ST2, NEW_ST2), (OLD_ABS_ANCHOR, NEW_ABS + OLD_ABS_ANCHOR), (ANCHOR_8, "\n\n" + SEC.rstrip("\n") + ANCHOR_8), (OLD_811, NEW_811), (OLD_83, NEW_83), (OLD_ROWS, NEW_ROWS), (OLD_9, NEW_9)]
for a, b in E:
    assert s.count(a) == 1, (s.count(a), a[:80])
    assert h.count(a) == 1, ("html", h.count(a), a[:80])
    s = s.replace(a, b)
    h = h.replace(a, b)

# ---- guards on the inserted text ---------------------------------------------------------------------------------------------------------------------
ins = "\n".join(b for _, b in E)
low = ins.lower()
for w in BANNED:
    _m = re.search(r"\b" + w + r"\b", low)
    assert _m is None, (w, low[max(0, _m.start() - 150):_m.end() + 100] if _m else "")
assert re.search(r"retract|supersed|withdrawn|was wrong|not the floor|previously read|outdated|i said", low) is None
assert "closed loop" not in low and "closed-loop" not in low and not re.search(r"\bdrives\b|driving performance", low), "closed-loop or driving wording"
assert "$" not in ins and "\r" not in ins, "a dollar sign or a CR in the inserted text"
nocode = re.sub(r"`[^`\n]*`", "", ins)
assert re.search(r"<[A-Za-z/!]", nocode) is None, re.search(r".{20}<[A-Za-z/!].{20}", nocode)
assert "@@" not in s and "</script" not in s.lower() and "\r" not in s
_flat = re.sub(r"\s+", " ", SEC)
for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"):
    assert fam in _flat, fam
assert "B = 2,000" in _flat and "T1-kin" in _flat and "T0-class" in _flat and "Rule Zero" in _flat
assert not re.search(r"\bT2 number exists\b", _flat) or "no T2 number exists" in _flat
for _p in re.finditer(r"T1-kin[^.]{0,80}closed", _flat):
    raise AssertionError("T1-kin called closed")
_corpus = re.sub(r"\s+", " ", "\n".join(SRC.values()) + "\n" + REG)
for _q in re.findall(r'\*"([^"\n]{25,}?)"\*', re.sub(r"\s*\n\s*", " ", ins)):
    assert re.sub(r"\s+", " ", _q) in _corpus, ("a quotation is not verbatim in a source", _q[:90])
_srcl = set("\n".join(SRC.values()).split("\n"))
for _l in SEC.split("\n"):
    if _l.startswith("|"):
        assert _l in _srcl, ("a table line of 7.34 is not a source line", _l[:90])
assert s.count("### 7.34 ") == 1 and s.index("### 7.33 ") < s.index("### 7.34 ") < s.index("## 8. Discussion")
assert s.count("**Status:** research report, v35 of 2026-11-03") == 1
assert tpl.replace("%%MD%%", s) == h, "html edited in place differs from template + md"
os.makedirs(OUT, exist_ok=True)
open(OUT + "V8F_SCIENTIFIC_REPORT.md", "w", encoding="utf-8", newline="\n").write(s)
open(OUT + "v8f_report.html", "w", encoding="utf-8", newline="\n").write(h)
print("report v35 built", len(s), len(h), "tip", TIP, "section 7.34 chars", len(SEC), "tables", len(T), "table lines", sum(len(v) for v in T.values()))
