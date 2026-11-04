"""Report v36 = v35 + section 7.35 (HISTSWAP and HISTSWAP-LF: the executed-state reads), the status line, one abstract bullet, 8.1 item 23, 8.3 Update (v36), 8.4 rows 1an and 1ao and 9 item 14.
Built from the landed v35 report blobs of one repository tip (anchored edits, every anchor asserted to occur exactly once). Nothing is read from a working tree. Nothing is staged, committed or pushed.

python -I patch_v36.py [tip [out_dir]]
Inputs, all `git show <tip>:<path>` from the mirror: the v35 V8F_SCIENTIFIC_REPORT.md and v8f_report.html (asserted by sha256), page_template_v30.html, the RESULTS and the frozen cards of
HISTSWAP and HISTSWAP-LF, the two raw JSONs (cross-check of the headline shares), the register.
Every table line of 7.35 is asserted to be a line of its source RESULTS; every italic quotation of 25 characters or more is asserted verbatim in a source."""
import hashlib, json, os, re, subprocess, sys
from collections import OrderedDict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_argv = sys.argv[1:]
TIP = _argv[0] if _argv else "9b79730c77ccd8521a149126d0a293d629770d4b"
TP = TIP[:8]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = (_argv[1] if len(_argv) > 1 else HERE + "/out").rstrip("/\\") + "/"
MIRROR = "C:/Users/Admin/tanitad-push"
R = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
BANNED = ("void", "superseded", "retracted", "formerly", "no longer", "correction", "withdrawn")
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


# ---- the v35 deliverables are the starting point ---------------------------------------------------------------------------------------------------------
V35_MD_SHA = "f398ca7c81f1719f717a86cfaf2fb8c4dfdbefb1c69e3aabf22689f75cfda394"
V35_HTML_SHA = "d337a818ac9ef2af7d9704433f585422f8bcba064533e7cfd34e74ad7be7ad6c"
RP35 = R + "report/2026-11-03-v35/"
s = blob(RP35 + "V8F_SCIENTIFIC_REPORT.md")
h = blob(RP35 + "v8f_report.html")
tpl = blob(R + "report/2026-10-26/page_template_v30.html")
assert sha(raw(RP35 + "V8F_SCIENTIFIC_REPORT.md")) == V35_MD_SHA, "the tip's report md is not the landed v35 deliverable"
assert sha(raw(RP35 + "v8f_report.html")) == V35_HTML_SHA, "the tip's report html is not the landed v35 deliverable"
assert tpl.count("%%MD%%") == 1 and h == tpl.replace("%%MD%%", s), "v35 html is not the template with the v35 md"
assert s.count("**Status:** research report, v35 of 2026-11-03") == 1
S0 = s

# ---- sources ----------------------------------------------------------------------------------------------------------------------------------------------
SRC_PATHS = OrderedDict([("HS", "raw/2026-11-02-histswap/RESULTS.md"), ("HL", "raw/2026-11-02-histswap-lf/RESULTS.md")])
SRC = OrderedDict((k, blob(R + p)) for k, p in SRC_PATHS.items())
CARD_PATHS = OrderedDict([("HS", "raw/2026-11-02-histswap/PREREG_HISTSWAP_FROZEN.md"), ("HL", "raw/2026-11-02-histswap-lf/PREREG_HISTSWAP_LF_FROZEN.md")])
CARDS = OrderedDict((k, blob(R + p)) for k, p in CARD_PATHS.items())
REG = blob("Project Steering/GOALS_AND_CLAIMS.md")
CARD_SHA = {k: sha(raw(R + p)) for k, p in CARD_PATHS.items()}
assert CARD_SHA["HS"].startswith("7e91a7c7") and CARD_SHA["HL"].startswith("d98db980"), CARD_SHA
for k in SRC:
    assert CARD_SHA[k] in SRC[k], ("the RESULTS file names a different card sha256", k)
RES_SHA = {k: sha(raw(R + p)) for k, p in SRC_PATHS.items()}
# landing commits (first commit that added the file), measured from the mirror
LAND = {"HS": ("8e1a40d", "2026-11-02 18:44:44"), "HL": ("408cd81", "2026-11-02 19:29:45")}
for k, (c, t) in LAND.items():
    for p in (SRC_PATHS[k], CARD_PATHS[k]):
        assert gitrun(["log", "--format=%h", "--diff-filter=A", TIP, "--", R + p]).startswith(c[:7]), (k, p)
    assert gitrun(["show", "-s", "--format=%ci", c]).startswith(t), c
# raw JSON cross-check of the headline shares (MEASURED artefacts behind the RESULTS)
J_HS = json.loads(blob(R + "raw/2026-11-02-histswap/raw/HISTSWAP_T1.json"))
J_HL = json.loads(blob(R + "raw/2026-11-02-histswap-lf/raw/HISTSWAP_LF_T1.json"))
assert J_HS["outcome"]["outcome"] == "SUPPORTED" and J_HS["outcome"]["n_cells_separated_better"] == 4
assert round(J_HS["outcome"]["mean_share_closed"], 3) == 0.939 and round(J_HS["outcome"]["pooled_share_closed"], 3) == 0.937
assert J_HL["interpretation"]["reading"] == "B_LEVEL" and J_HL["m1_reading"]["HCV"]["n_cells_separated_better"] == 0 and J_HL["m1_reading"]["HLVL"]["n_cells_separated_better"] == 4
assert round(J_HL["m1_reading"]["HCV"]["mean_share_closed"], 3) == -0.425 and round(J_HL["m1_reading"]["HLVL"]["mean_share_closed"], 3) == 0.610
assert round(J_HL["m1_reading"]["HCV"]["pooled_share_closed"], 3) == -0.415 and round(J_HL["m1_reading"]["HLVL"]["pooled_share_closed"], 3) == 0.612


# ---- tables, copied line for line ---------------------------------------------------------------------------------------------------------------------
def ncols(line):
    body = re.sub(r"`[^`\n]*`", "x", line.replace("\\|", "x"))
    return body.count("|") - 1


def blk(src, first, nth=0, n=None, must=None):
    lines = SRC[src].split("\n")
    idx = [i for i, l in enumerate(lines) if l.startswith(first)]
    assert len(idx) > nth, (src, first[:60], len(idx), nth)
    rows = []
    for l in lines[idx[nth]:]:
        if not l.startswith("|"):
            break
        rows.append(l)
    if n is not None:
        assert len(rows) == n, (src, first[:60], len(rows), n)
    if must is not None:
        assert any(must in r for r in rows), (src, first[:50], must)
    assert all(r in lines for r in rows) and len({ncols(r) for r in rows}) == 1, (src, first[:60])
    assert not any(_BAN.search(r) for r in rows), ("banned word in a table line", first[:60])
    assert not any(re.search(r"closed[- ]loop|driving performance", r, re.I) for r in rows), ("tier word in a table line", first[:60])
    return rows


T = OrderedDict()
T["HS_HEAD"] = blk("HS", "| cell | M1 X - Y (X = oracle history) |", n=6)
T["HS_KS2"] = blk("HS", "| cell | v0_in = v0 at tick 0 (max abs) |", n=6)
T["HS_S0"] = blk("HS", "| row | family | X - P10 | X - Y | Y - P10 |", nth=0, n=21, must="-0.0476 [-0.0664, -0.0315] SB")
T["HS_S1"] = blk("HS", "| row | family | X - P10 | X - Y | Y - P10 |", nth=1, n=21, must="-0.0523 [-0.0739, -0.0338] SB")
T["HS_CLS"] = blk("HS", "| cell | contrast | row delta | L: n, delta, contribution |", n=14)
T["HS_SWAP"] = blk("HS", "| cell | swap_ticks | swap_bucket_differs | share |", n=6)
T["HL_HEAD"] = blk("HL", "| cell | arm | M1 X - Y | Y - P10 (the excess) |", n=10)
T["HL_LABEL"] = blk("HL", "| cell | arm | ADE6 X - Y (m) vs guard |", n=10)
T["HL_READA"] = blk("HL", "| cell | arm | D1 | D2 | D3 (L / Lat / T / S) |", n=10)
T["HL_S0"] = blk("HL", "| row | family | Y - P10 | HCV X - Y | HLVL X - Y |", nth=0, n=21, must="+0.0270 [+0.0121, +0.0432] SW")
T["HL_CLS"] = blk("HL", "| cell | contrast | row delta | L: n, delta, contribution |", n=14)
T["HL_PRIOR"] = blk("HL", "| cell | arm | P (Y, X, X - Y) m |", n=10)
T["HL_KC1"] = blk("HL", "| run | `--histswap` | max abs v0_in - v0, all ticks (m/s) |", n=10)
T["HL_SWAP"] = blk("HL", "| cell | HCV | HLVL |", n=6)
T["HL_PROB"] = blk("HL", "| quantity | card (ESTIMATED, written before any number) |", n=5)
T["HL_VS"] = blk("HL", "| cell | share closed: HISTSWAP / HLVL / HCV |", n=6)

# ---- claims taken from the sources, asserted present ---------------------------------------------------------------------------------------------------
PHRASES = [
    ("HS", "X also leaks the human's realised motion at ticks 1 and 2"),
    ("HS", "SUPPORTED = 4 of 4 better and mean share closed >= 0.5; NOT SUPPORTED = fewer than 3 of 4 better; MIXED otherwise."),
    ("HS", "SUPPORTED does not say which part of the logged history carries the effect"),
    ("HS", "the mean share of Y - P10's M1 excess closed is **0.939** (bar 0.5; pooled over the four cells 0.937)"),
    ("HS", "Launch-class windows are 12.1 %-12.1 % of the M1 row's windows (weight `w`) and carry 57 %-65 % of the Y - P10 excess and 58 %-68 % of the X - Y movement"),
    ("HS", "its `DOMINANT` flag is an oracle artefact and is never a claim about a deployable arm"),
    ("HS", "S-POS rows of X mix the logged poses the S head read with the executed pose the extractor frames them in"),
    ("HS", "dump bit-identical True (max abs dev 0.000e+00 m)"),
    ("HS", "32-window smoke `logs/smoke_histswap.out`: run when the tool was built; not re-run in this read."),
    ("HS", "the swap is not ceiling-limited"),
    ("HL", "The trend copy is not supported as the carrier"),
    ("HL", "nothing here shows that a model trained or run without the human's speed recovers any of the 60 %-63 %"),
    ("HL", "the movement of HLVL's plan end point comes through the prior's anchor, with O's delta moving against it"),
    ("HL", "a model that reads the human's speed from frame motion can pass M1 here and not in driving"),
    ("HL", "HCV removes all acceleration information, not only the trend"),
    ("HL", "HLVL alone cannot tell *O needs the level* from *the prior's anchor is the input speed*"),
    ("HL", "is untested (HTRND not run)"),
    ("HL", "whether the harm is the missing acceleration information or the off-distribution input, a history that is exactly constant at every row, is UNVERIFIED"),
    ("HL", "HLVL's HARMS label rests on the S-POS rows alone."),
    ("HL", "the frame-freeze control that separates this is pre-registered in `DRAFT_RECOVERY_CARD_v2.md` section 8, and it was **not run on any arm of this read**"),
    ("HL", "The split of the 0.33 between acceleration, trend and yaw rows is not measured here."),
    ("HL", "the level source (perception or training) named as its unknown"),
    ("HL", "the STRATEGIC family is PAUSED (ITEM 25)"),
    ("HL", "HLVL's and HISTSWAP's X - Y intervals overlap on M1 at 4 of 4 cells and on ADE6 at 0 of 4 cells"),
    ("HL", "Tick 0 of both arms differs from Y by construction"),
    ("HL", "Prior art on slowed-history training"),
]
_corp = {k: re.sub(r"\s+", " ", v) for k, v in list(SRC.items())}
_corp["REG"] = re.sub(r"\s+", " ", REG)
_corp["HS"] += " " + re.sub(r"\s+", " ", CARDS["HS"]); _corp["HL"] += " " + re.sub(r"\s+", " ", CARDS["HL"])
_nm = [(k, p[:70]) for k, p in PHRASES + [
    ("REG", "O lacks the true speed level once the executed speed lags"),
    ("REG", "Seven decode levers since S leave M1 the D2 blocker; the cause sits upstream of the decode (H-HISTSWAP-1)."),
    ("REG", "STRATEGIC family, active for v8F"),
] if re.sub(r"\s+", " ", p) not in _corp[k]]
assert not _nm, ("a claim phrase is not in its source", _nm)

# ---- the section text ---------------------------------------------------------------------------------------------------------------------------------
SEC_735 = """### 7.35 HISTSWAP and HISTSWAP-LF (2026-11-02): an oracle ego history closes 0.918 to 0.984 of the M1 excess, a constant-twist history at the executed speed does not, and the same history at the logged speed closes 0.599 to 0.633

* **Status of this section (v36) [M].** Two zero-training reads of 2026-11-02 on the eval-141 causal lane (2,399 windows, 141 episodes), each a card frozen before its own numbers and a RESULTS file landed on the branch: HISTSWAP (`H-HISTSWAP-1`) and HISTSWAP-LF (`H-HISTSWAP-LF-1`). Each is read from its RESULTS file at the tip `{{TP}}` with `git show`, never from a working tree, and each number is **[M]** with that path. Tier: both are T1-kin (the model's own executed state, logged cameras, its own plan applied open loop in the scene; it says nothing about re-perception; no T2 number exists). Estimator: the paired episode-cluster bootstrap, B = 2,000, seed 0 (`taniteval.ci` draw stream via `gapdecomp.decompose`), which answers *"would another draw of EPISODES say this?"* and nothing about another training run (`H-ESTIM-SEED-1`: one training run per seed) or another inference run (the planner is deterministic, bit-identity rerun KA1). Arms: DAG-OACC (the adopted base, the headline arm) and SOUP (§7.32), each at s0 and s1, against the same arm without the lever (X − Y, here Y = KAPPA-T1's fresh S runs, reused) and against the P10 floor (Read A). In the cards' tags DAG-OACC is `CTRL`.
* **Synthesis (zero training; read the setup of each study before the sentence that cites it).** After §7.34, D2 of Read A against P10 is blocked by the M1 crawl share (it fails at 12 of 12 cells for each of FEAS2 and FEAS3), and the register's count is seven decode levers since S that leave M1 the D2 blocker. The two reads below ask where in the rollout the excess enters. **HISTSWAP** replaces the network's ego-history input at every tick by the LOGGED ego history, with the plan still committed to the executed state: M1 crawl share X − Y is separated-better at 4 of 4 cells and closes 0.918 to 0.984 of the Y − P10 excess (mean 0.939; bar 0.5), H SUPPORTED. That arm is an ORACLE (the logged history carries the human's speed at the tick) and is not deployable. **HISTSWAP-LF** splits the closure with two constant-twist histories (speed and yaw rate held on every earlier row, acceleration 0). At the EXECUTED speed (HCV, deployable, no human information) M1 X − Y is separated-WORSE at 4 of 4 cells (+0.0201 to +0.0270; share closed −0.559 to −0.301) and ADE6 X − Y is +0.5867 to +0.6527 m against a +0.0604 m guard. At the LOGGED speed (HLVL, an oracle, level only) M1 X − Y is separated-better at 4 of 4 cells (−0.0422 to −0.0289; share closed 0.599 to 0.633) and ADE6 X − Y is −0.6165 to −0.5171 m. The card's committed reading is (B) LEVEL, and the outcome label by the KAPPA / FEAS rule is HARMS for both arms on both bases (HLVL on the S-POS rows alone). The reading is the card's, with its caveats: HLVL's closure is near-mechanical (the prior's anchor carries the plan's end point and O's delta moves against it), HCV removes all acceleration information and so leaves *trend alone* untested, and both are T1-kin diagnostics on one training run per seed. The lever both cards point to, the recovery fine-tune of O, is read in a later section; no result of it is used here.

**7.35.1 HISTSWAP: replacing the ego-history input by the logged history closes the M1 excess (`H-HISTSWAP-1`; ORACLE diagnostic).**

* **Status and setup [M].** Read 2026-11-02T18:31:45+0100; `R/raw/2026-11-02-histswap/RESULTS.md` (sha256 `{{HS_RES}}…`), landed `8e1a40d` (2026-11-02 18:44:44, with FEAS2 and FEAS3). Card `PREREG_HISTSWAP_FROZEN.md` sha256 `7e91a7c7…`, frozen 2026-11-02T17:57:30+0100 before any swap number. **Arm X** = Y's flags plus `--histswap logged`: at every tick the network's ego-history INPUT (`p10f`, `base`, `kin3`, `pose`, `pose_now`, `pose_t`, `pose_s` and the causal speed bucket, i.e. every ego-row-built key) is read from the LOGGED clip arrays at that tick, while the plan is still committed to the EXECUTED state. Unchanged: nav (dead-reckoned from the executed motion), camera, hold ages, S (`P + SG(plan - P)`, with P the prior the network computed in that call, so P now reads the logged history) and the P10 floor. Tick 0 is identical to Y by construction. **Y** = KAPPA-T1's fresh S runs `SREP_*` (reused, pinned), as FEAS2 reused them; DAG-OACC and SOUP at s0 and s1 give four rollouts `HSW_*`, plus one bit-identity rerun of a Y (`SREP4_DAGOACC_s0`, `--histswap off`). The M1 row is the crawl-share row of §7.34 (the card writes it as *ego speed < GT - 2 m/s*, `t1hd_read.py:422`; a share, lower is better).
* **Why this is an oracle and not a deployable arm [M].** The logged ego history after tick 0 is the human's. X's absolute rows are therefore never quoted as performance; only X − Y on M1 and the row structure are read. The card adds a limit on a positive reading: *"X also leaks the human's realised motion at ticks 1 and 2"*, so it cannot tell *the executed lag is the cause* from *any human-motion history fixes it*; a negative reading would not be subject to that leak. The card's own pre-check on seen Y data (`raw/PRECHECK_CEILING.json`) finds the slow GT-fast pairs of Y minus those of P10, by executed-step block [0-10) / [10-20) / [20-30), at DAG-OACC s0 −25 / +583 / +1158 and s1 +6 / +765 / +1414, SOUP s0 +24 / +934 / +1579 and s1 −6 / +716 / +1497: the excess starts after tick 0, so the swap (which cannot change tick 0) is not ceiling-limited, and the pattern does not discriminate the hypothesis from a plan that decays with the horizon.
* **The pre-registered bars, both outcomes committed before any number [M].** H (a HYPOTHESIS before the read): O follows its own slowed ego history, so that once the executed speed lags the history input says slow and O plans slow. Per cell, `better` = the M1 row of X − Y is separated and delta < 0; `share closed` = −(X − Y) / (Y − P10) on the M1 row. The reader's rule (`decide` in `code/histswap_t1_read.py`, six rule tests, one mutation RED): *"SUPPORTED = 4 of 4 better and mean share closed >= 0.5; NOT SUPPORTED = fewer than 3 of 4 better; MIXED otherwise."* The card's committed next step per outcome: SUPPORTED sends the recovery fine-tune of O (a draft card, not frozen) to the PI for a box-2 go together with the leak caveat and the leak-free separating arm as its unknown; NOT SUPPORTED names the next lever from the per-class split of the remaining M1 excess (launch-class share of 0.5 or more points to the trigger / encoder path, moving-class share of 0.5 or more to a longitudinal mode-commit head); MIXED runs the leak-free arm before any training go. The card's probabilities (ESTIMATED, written before any number) were P(SUPPORTED) 0.60, P(MIXED) 0.25 and P(NOT SUPPORTED) 0.15.
* **Controls [M].** All eight PASS before any row: KY (8 of 8 reused Y files hash-equal to the pins), KA1 (the fresh `--histswap off` rerun equals the reused Y: dump bit-identical True, max abs dev 0.000e+00 m), KS0 (tick 0 of every X equals Y's tick 0, max abs deviation 0.0e+00 on 2399 tick-0 rows per cell), KS1 (tick 1's executed pose and v0 equal Y's), KS2 (below), KF (windows and floor equal Y's), K3 (Y banked − P10 reproduces `SOUP_READ.json` on 76 of 76 rows) and KB1 (the P10 side equals the landed banked arrays, 8 of 8 runs). Tool tests `logs/pytest_histswap.log` (8 passed, three mutations RED) and `logs/pytest_read_rule.log` (6 passed); the 32-window smoke was run when the tool was built and not re-run in this read. KS2 shows the swap reached the network: `v0_in` is the speed the network read, `v0` the executed speed.

<<T:HS_KS2>>

* **Outcome [M].** The card's rule reads **SUPPORTED**: M1 crawl share X − Y is separated-better at 4 of 4 cells (both seeds on both bases), the mean share of Y − P10's M1 excess closed is 0.939 (bar 0.5; pooled over the four cells 0.937; raw `raw/HISTSWAP_T1.json`: `mean_share_closed` rounds to 0.939 and `pooled_share_closed` to 0.937). The card gave SUPPORTED 0.60. X is an ORACLE: the ADE6 X − Y column is a leak the card expected and is not performance.

<<T:HS_HEAD>>

* **Read A of X against P10, four metric families [M].** At all four cells D1, D2 and D3 are true (D3 with LONGITUDINAL, LATERAL, TACTICAL and STRATEGIC each true), no row of X − P10 is separated-worse and 16 of 19 rows are separated-better; the DOMINANT flag reads true and the RESULTS file marks it an oracle artefact that is never a claim about a deployable arm. On X − Y no row of the 19 is separated-worse at any cell, no non-crawl row is separated-worse, and all four crawl rows (yaw-rate MAE, curvature MAE, L2 curvature MAE, M1 crawl share) are separated-better at 4 of 4 cells. The DAG-OACC s0 and s1 cells in full (per family, never pooled; the SOUP cells are in the RESULTS file; STRATEGIC is reported, not paused, for v8F, PI ruling 2026-10-17; the S-POS rows of X mix the logged poses the S head read with the executed pose the extractor frames them in and are UNVERIFIED, outside the reading). Cells are `point [lo, hi] mark`; SB / SW = separated-better / -worse (direction-aware), ns = interval spans 0. DAG-OACC s0:

<<T:HS_S0>>

  DAG-OACC s1:

<<T:HS_S1>>

  LONGITUDINAL at DAG-OACC s0: speed MAE, along-track MAE, accel MAE, DK-HW and DK-TG are separated-better on X − Y and DK-VIOL is not separated (−0.0336 [−0.0790, +0.0018]); LATERAL: heading MAE, yaw-rate MAE, curvature MAE, cross-track MAE, L1 and L2 are all separated-better (curvature MAE −0.0510 [−0.0765, −0.0299]); TACTICAL: TAC-LAT and TAC-LON are separated-better (+0.0692 and +0.3005; the direction-aware mark is the RESULTS file's); STRATEGIC: S-POS@30 / 50 / 80 are separated-better (−1.9915, −2.8514, −3.7918 m). At DAG-OACC s1 heading MAE is the one LATERAL row that is not separated on X − Y (−0.7822 [−1.5407, +0.0641]). The tactical goal-head F1 is NOT COMPUTABLE at T1-kin (**INHERITED** from §7.32).
* **The M1 row by situation class [M] (descriptive, not a decision).** Classes are L launch, S stay, O other-stopped, M moving; the S and O classes carry no windows in the masked M1 pool at any cell. Launch-class windows are 12.1 % of the M1 row's windows and carry 57 % to 65 % of the Y − P10 excess and 58 % to 68 % of the X − Y movement; the moving class carries the rest, and X − Y is separated-better in both classes at all four cells. The split is by situation class, not by v0 band: whether the swap's closure within class M sits in the v0 3 to 8 m/s windows of §7.34.2 is not measured here (**UNVERIFIED**). The launch-class share is of the same order as the trigger-window share of the M1 gap in §7.34.1 (+0.606 / +0.545), but the two pools differ (launch class against the trigger windows, 15.0 % of all windows), so the figures are not interchangeable.

<<T:HS_CLS>>

* **Swap bookkeeping [M].** The swap includes the causal speed bucket (a function of the ego speed history); `swap_bucket_differs / swap_ticks` is 1.61 % to 1.68 % and the bucket's separate contribution to M1 is not attributed.

<<T:HS_SWAP>>

* **Reading and next.** The card's rule reads H SUPPORTED at 4 of 4 cells with the share closed 0.918 to 0.984: the executed-speed drift enters through the network's ego-history input, and with that input swapped the M1 excess against P10 is not separated from zero (X − P10 on M1 is +0.0017 [−0.0046, +0.0084] ns at DAG-OACC s0, ns at all four cells); this does not show that no other input contributes, because the swap also changes the prior P that S smooths toward. In the RESULTS file's words, *"SUPPORTED does not say which part of the logged history carries the effect"* (the human's realised speeds at ticks 1 and 2, the speed bucket, or the executed lag itself), and it does not say that a model trained or run without the human's history recovers any of the closure. **Next, as committed.** The recovery fine-tune of O goes to the PI for a box-2 go with the leak caveat, and the leak-free separating arm is named as its unknown: that arm is §7.35.2. STRATEGIC is reported, not paused, for v8F.

**7.35.2 HISTSWAP-LF: a constant-twist history at the executed speed (HCV) reads HARMS and does not close M1; at the logged speed (HLVL) it closes 0.599 to 0.633 (`H-HISTSWAP-LF-1`; reading (B) LEVEL).**

* **Status and setup [M].** Read 2026-11-02T19:15:12+0100; `R/raw/2026-11-02-histswap-lf/RESULTS.md` (sha256 `{{HL_RES}}…`), landed `408cd81` (2026-11-02 19:29:45). Card `PREREG_HISTSWAP_LF_FROZEN.md` sha256 `d98db980…`, frozen 2026-11-02T18:50:07+0100 before any HCV or HLVL number. The question: is HISTSWAP's closure *inertia* (O follows the TREND of its own executed history; a lag self-perpetuates) or *level* (O needs the human's speed LEVEL, which only the oracle supplies)? The card names prior art on slowed-history training (ORC, DAG2, DAG-ORC) as reading HARMS without clearing M1. Both arms replace the ego-history input at EVERY tick by a **constant-twist history**: speed v and yaw rate omega held on every earlier row, acceleration 0, anchored at the EXECUTED pose of the tick, built through the programme's one spelling (`yaw_kin_from_track`, `ego_channels_native`, `kin_features`, `kdx_base` on a synthetic native track). **HCV** (`--histswap cv_exec`, deployable): v = the EXECUTED measured speed at the tick, no trend and no human information. **HLVL** (`--histswap cv_logged`, ORACLE, level only): v = the LOGGED measured speed at the tick. Omega is the executed yaw rate in both, so HCV and HLVL differ in exactly one number, the speed. The channels replaced are exactly the set HISTSWAP's tool swaps; nav is not replaced, and the plan is still committed to the executed state. Pinned choices: omega is held, not zeroed; the tick row's own acceleration is also 0, so HCV removes all acceleration information (the trend and the current value); the causal bucket is rebuilt from the CV history. Tick 0 is not identical to Y (the CV history differs from the logged one at t0). Y = KAPPA-T1's fresh S runs, reused; eight rollouts `HCV_*` / `HLV_*` plus the bit-identity rerun `SREP5_DAGOACC_s0`.
* **The pre-registered bars, both outcomes committed before any number [M].** (1) Outcome rule per arm and base, the KAPPA / FEAS rule (`smooth_t1_read.arm_outcome`): precedence DOMINANT, HARMS, MOVED, NO EFFECT; HARMS at a cell is ADE6 X − Y > +0.0604 m or a non-crawl row of X − Y separated-worse (the base is HARMS if either seed is); MOVED is all four crawl rows separated-better and ADE6 X − Y ≤ +0.0604 m. (2) M1 reading per arm over the four cells: CLOSES = M1 X − Y separated-better at all 4 cells and mean share closed ≥ 0.5; NOT CLOSES = separated-better at fewer than 3 cells; else MIXED. (3) The committed interpretation: HCV CLOSES means (A) inertia, and HCV is a deployable inference lever; HCV does not close and HLVL CLOSES means (B) level, so the speed level must come from perception or training and not from the history form; neither CLOSES means (C) interaction, the trend-times-level structure of the full history. The card's probabilities (ESTIMATED): HCV CLOSES 0.10, MIXED 0.15, NOT CLOSES 0.75; HLVL CLOSES 0.80, MIXED 0.12, NOT CLOSES 0.08; reading (A) 0.10, (B) 0.70, (C) 0.20. The card warned that HLVL's closure is near-mechanical, so that **HCV is the discriminating arm**.
* **Controls [M].** All eight PASS before any row: KY (8 of 8), KA1 (the LF-tool `--histswap off` rerun equals the reused Y: dump and ticks bit-identical, max abs dev 0.000e+00; the planner is deterministic, so the thread-order floor is 0 on this rig), KC1 (the flag reached the run, below), KC2 (the CV history reached the network: max abs tick-0 plan_raw deviation 42.5 to 50.6 m over 2399 tick-0 rows per run, 8 of 8 runs), KC3 (tick 1's executed position equals the tick-0 pose composed with plan knot 10: max deviation 1.1e-13 m, share within tolerance 1.0000 at 8 of 8 runs), KF, K3 (76 of 76 rows) and KB1 (12 of 12 runs). Tool tests `logs/pytest_histswap_lf.log` (9 passed, three mutations RED), rule tests (7 passed, three mutations RED), the 32-window smoke SMOKELF OK, and `ZZHLF-ROLLOUTS-DONE nfail=0`.

<<T:HL_KC1>>

* **Outcome of the M1 reading [M].** The card's rule reads **B_LEVEL** (`hcv_closes` False, `hlvl_closes` True, `hlvl_discordant` False; raw `raw/HISTSWAP_LF_T1.json`). HCV: NOT CLOSES, M1 X − Y separated-WORSE at 4 of 4 cells (0 of 4 separated-better), mean share closed −0.425 (pooled −0.415; bar 0.5). HLVL: CLOSES, separated-better at 4 of 4 cells, mean share closed 0.610 (pooled 0.612). HISTSWAP's full logged history closed 0.939 on the same cells. The card gave HCV NOT CLOSES 0.75, HLVL CLOSES 0.80 and (B) 0.70. HCV's NOT CLOSES is not a null: it is separated-worse than Y at 4 of 4 cells and ADE6 is over the guard at 4 of 4, so the executed-speed constant-twist history is itself a harmful input.

<<T:HL_HEAD>>

* **The outcome label and the four metric families [M].** The label by the KAPPA / FEAS rule is HARMS for both arms on both bases. Cells: ADE6 X − Y against the +0.0604 m guard, crawl rows separated-better, and the non-crawl rows separated-worse on X − Y:

<<T:HL_LABEL>>

  **HCV** (per family, rows separated-worse at some cell): LONGITUDINAL DK-HW, DK-TG, DK-VIOL, along-track MAE and speed MAE; PRIMARY ADE6; STRATEGIC S-POS@30 / 50 / 80; TACTICAL TAC-LON; no LATERAL non-crawl row. At all four cells: ADE6, DK-VIOL, along-track MAE and the three S-POS rows. HCV's HARMS is not a result on the margin: ADE6 X − Y is over the guard at every cell (+0.5867 to +0.6527 m), and leaving out the UNVERIFIED S-POS rows the longitudinal rows still carry it. **HLVL**: ADE6 is within the guard at 4 of 4 cells, all four crawl rows are separated-better at 4 of 4 cells (MOVED holds at both bases), and the only non-crawl rows separated-worse are S-POS@30 / 50 / 80, so HLVL's HARMS label rests on the S-POS rows alone. Those rows are UNVERIFIED in the card (they mix the CV pose the S head read with the executed pose the extractor frames them in), and the label stands as the rule gave it. STRATEGIC is reported, not paused, for v8F (PI ruling 2026-10-17; the register row records the family as active for v8F); the family column of the RESULTS lines below carries the file's own tag *PAUSED, ITEM 25*, which this report does not adopt for v8F. Read A of X against P10:

<<T:HL_READA>>

  D1 and D3 hold and D2 fails at all 8 cells of both arms; no cell is DOMINANT. M1 crawl share keeps D2 failing for HLVL at every cell (X − P10 on M1 +0.0203 [+0.0104, +0.0314] SW at DAG-OACC s0, an excess reduced from Y's +0.0483 but not removed), and Y itself fails D2 on M1. All 19 rows at DAG-OACC s0 (the other seven cells are in the RESULTS file; per family, never pooled):

<<T:HL_S0>>

  At DAG-OACC s0 the LONGITUDINAL, LATERAL and TACTICAL pattern separates the arms: HCV is separated-worse on speed MAE (+0.1043), along-track MAE (+0.7059), DK-VIOL and DK-HW, while HLVL is separated-better on speed MAE (−0.5230) and along-track MAE (−0.5105); both are separated-better on heading MAE, yaw-rate MAE and curvature MAE; TAC-LON is separated-worse for HCV (−0.0383) and separated-better for HLVL (+0.2193) under the direction-aware marks; the S-POS rows are separated-worse for both (+0.9631 / +1.6025 / +2.2796 m for HCV, +0.9863 / +1.5608 / +2.1888 m for HLVL).
* **The M1 row by situation class [M] (descriptive).** HCV's harm sits in the launch class: its launch-class X − Y is +0.2552 to +0.3079, which is 0.81 to 1.14 of Y's own launch-class excess over P10, while the moving class improves slightly (−0.0150 to −0.0084). HLVL improves both classes (launch −0.1417 to −0.1008, moving −0.0285 to −0.0167). The weight of the launch class is 12.1 % of the row's windows and it carries 57 % to 65 % of Y − P10, 35 % to 49 % of HLVL's X − Y movement and 127 % to 155 % of HCV's.

<<T:HL_CLS>>

* **Prior anchor against O's delta [M] (descriptive; point means with no interval).** The plan is `P + O's delta` (P the prior the network computed in the same call, which reads the swapped history). Mean over windows of the per-tick 6 s along-track value, ticks ≥ 1:

<<T:HL_PRIOR>>

  For HLVL the prior's 6 s end point rises against Y at 4 of 4 cells (X − Y +0.528 to +3.472 m), O's own delta moves down at 4 of 4 (−0.916 to −0.014 m) and the plan's end point rises at 4 of 4 (+0.358 to +2.552 m): in the RESULTS file's words, *"the movement of HLVL's plan end point comes through the prior's anchor, with O's delta moving against it"*. This is consistent with the card's caution that HLVL's closure is near-mechanical; it is a consistency check and not a decomposition of the M1 closure (the block averages over all windows and ticks ≥ 1, while M1 lives in the launch-class windows). For HCV the signs are not uniform (the prior's end point moves up at 2 of 4 cells), so the mean 6 s end point does not carry HCV's M1 and ADE6 harm.
* **Against HISTSWAP's full logged history [M].** Same Y, same windows, same checkpoints; HISTSWAP is an oracle too.

<<T:HL_VS>>

  Mean share closed: HISTSWAP 0.939, HLVL 0.610, HCV −0.425; mean ADE6 X − Y −1.261 m, −0.576 m and +0.610 m. The level alone, as an oracle, recovers 65 % of what the full logged history recovers on M1 (0.610 / 0.939). HISTSWAP − HLVL, what the rest of the logged history adds on top of the speed level, is 0.329 of the share closed and −0.685 m of ADE6; it also carries the logged yaw and pose rows, which HLVL holds at the executed yaw rate, and *"The split of the 0.33 between acceleration, trend and yaw rows is not measured here."* The overlap of the X − Y intervals is a check and not an estimator of the difference (the arms were not bootstrapped against each other): they overlap on M1 at 4 of 4 cells and on ADE6 at 0 of 4.
* **Swap bookkeeping and the card's probabilities [M].** The rebuilt causal bucket differs from the executed-history bucket on 8.70 % to 9.28 % of ticks for HCV and 13.85 % to 13.98 % for HLVL; its separate contribution is not attributed.

<<T:HL_SWAP>>

<<T:HL_PROB>>

  One read is not a calibration.
* **Reading and next.** Reading (B) LEVEL by the card's committed rule; the register records *"O lacks the true speed level once the executed speed lags"*. The trend copy is not supported as the carrier: HCV, which holds speed and yaw rate constant and sets acceleration to 0, does not close M1 and is separated-worse, while HLVL differs from it in exactly one number, the speed, and closes 60 % to 63 % at the same four cells. **Next, as committed.** Under (B) the recovery fine-tune with re-expressed targets goes to the PI for a box-2 go with the level source (perception or training) named as its unknown, and no history-form lever is pursued; HARMS at either arm makes the named non-crawl rows (ADE6, DK-VIOL, along-track MAE at HCV) the target of the next lever before any other reading is used. STRATEGIC is reported, not paused, for v8F.

**7.35.3 What the two reads establish, what they cannot claim, and the lever.**

* **Established [M, T1-kin, oracle diagnostics, paired episode-cluster bootstrap B = 2,000 seed 0].** (1) Giving the network the logged ego history, and nothing else changed, closes 0.918 to 0.984 of the M1 excess over P10 at 4 of 4 cells with no row of X − Y separated-worse: the M1 excess enters through the ego-history input channel. (2) A history that is the same in form but built from the executed speed does not close it and is itself harmful (HCV: M1 separated-worse at 4 of 4 cells, ADE6 +0.5867 to +0.6527 m, longitudinal rows separated-worse). (3) The same history at the logged speed closes 0.599 to 0.633, with ADE6 within the guard at 4 of 4 cells. By the card's committed rule that reads (B) level: what the model needs from the history, and does not get from its own executed motion, is the speed level.
* **What this section cannot claim.** (1) Oracle swaps are diagnostics: X (HISTSWAP) and HLVL read the human's speed, no row of either is performance, and neither is a deployable fix. (2) *"nothing here shows that a model trained or run without the human's speed recovers any of the 60 %-63 %"*. (3) HLVL alone cannot tell that O needs the level from the prior's anchor being the input speed; the plan is `P + O's delta`, the executor tracks it, and the descriptive block shows the end point following P. HCV is the discriminating arm and it reads HARMS, which supports the card's reading (B) and does not measure O's own dependence on the level. (4) HCV removes all acceleration information, so *trend alone* is untested (HTRND, the card's route (C), was not run), and whether HCV's harm is the missing information or an off-distribution input (a history exactly constant at every row) is **UNVERIFIED**. (5) HISTSWAP leaks the human's realised motion at ticks 1 and 2, and the 0.329 that HISTSWAP adds over HLVL (acceleration, trend and logged yaw / pose rows) is not split. (6) T1-kin: the cameras are the logged ones and advance at the human's pace, so *"a model that reads the human's speed from frame motion can pass M1 here and not in driving"*; the frame-freeze control that separates this is not run on any arm of these two reads. (7) One training run per seed and no replicate arm (`H-ESTIM-SEED-1`): the four cells are the rig's replicates and not a variance estimate, X and Y share the checkpoint, the same 141 episodes serve every read of §7.34 and §7.35, and an interval from this bootstrap answers another draw of episodes only. (8) The S-POS rows of the swapped arms are UNVERIFIED, and HLVL's HARMS label rests on them alone. (9) The class split is by situation class, not by v0 band; the 3 to 8 m/s localisation of §7.34.2 is not re-measured here.
* **Next (Rule Zero).** Both committed arms are run and the committed line is the recovery fine-tune of O with re-expressed targets (RECOVER-FT), whose unknown is where the speed level comes from (perception or training) and whose read needs the pre-registered frame-freeze control because T1-kin's logged frames move at the human's pace. It is read in a later section of this report; this section reports no RECOVER result and makes no claim about its state. Blocked until then: any deployable form of the level (it needs the PI's box-2 go and compute), and the split of HISTSWAP − HLVL (a new zero-training arm with its own card).
* **Provenance [M].** Under `R/raw/` at the tip `{{TP}}`: `2026-11-02-histswap/RESULTS.md` (sha256 `{{HS_RES}}…`) and `PREREG_HISTSWAP_FROZEN.md` (`7e91a7c7…`), `2026-11-02-histswap-lf/RESULTS.md` (`{{HL_RES}}…`) and `PREREG_HISTSWAP_LF_FROZEN.md` (`d98db980…`), and the two raw JSONs `raw/HISTSWAP_T1.json` and `raw/HISTSWAP_LF_T1.json`; claim rows `H-HISTSWAP-1` and `H-HISTSWAP-LF-1` in `Project Steering/GOALS_AND_CLAIMS.md`. The build is `patch_v36.py` (anchored edits on the landed v35 blobs; every table copied line for line from its RESULTS and asserted to be a source line), checked by `checks_v36.py` and re-measured by `audit_v36.py`.
"""

SEC = SEC_735.replace("{{TP}}", TP).replace("{{HS_RES}}", RES_SHA["HS"][:8]).replace("{{HL_RES}}", RES_SHA["HL"][:8])
for _k, _rows in T.items():
    assert SEC.count("<<T:%s>>" % _k) == 1, _k
    SEC = SEC.replace("<<T:%s>>" % _k, "\n".join(_rows))
assert "<<T:" not in SEC and "{{" not in SEC, re.findall(r"(<<T:\w+>>|\{\{\S+\}\})", SEC)[:3]
SEC = SEC.replace("\r", "")

# ---- the abstract, the status line and the edits -------------------------------------------------------------------------------------------------------
OLD_ST_A = ("research report, v35 of 2026-11-03, results through repository tip `92cbf277` and, within it, through the 2026-11-02 crawl-diagnosis and decode-lever reads (§7.34: GAPDECOMP, GAPDECOMP-M, SMOOTH-T1, trigger probe 5, KAPPA-T1, FEAS2 and FEAS3); "
            "the later reads (HISTSWAP, HISTSWAP-LF, FRAMEFREEZE, RECOVER-FT, RECOVER-NEXT) are not covered here and come in v36 (v34: `870fa33a`")
NEW_ST_A = ("research report, v36 of 2026-11-04, results through repository tip `" + TP + "` and, within it, through the 2026-11-02 crawl-diagnosis and decode-lever reads (§7.34: GAPDECOMP, GAPDECOMP-M, SMOOTH-T1, trigger probe 5, KAPPA-T1, FEAS2 and FEAS3) "
            "and the executed-state reads (§7.35: HISTSWAP and HISTSWAP-LF); the tip also holds later reads (FRAMEFREEZE, RECOVER-FT, RECOVER-NEXT) that this report does not cover (v35: `92cbf277`; v34: `870fa33a`")
OLD_ST_B = "kbox2lf all read HARMS with M1 failing D2 at every cell (KAPPA-T1, FEAS2, FEAS3)."
NEW_ST_B = (OLD_ST_B + " v36 adds the executed-state reads (§7.35, T1-kin, MEASURED, zero training, ORACLE diagnostics): an oracle swap of the logged ego history closes 0.918 to 0.984 of the M1 excess at 4 of 4 cells (HISTSWAP, H SUPPORTED), "
            "a constant-twist history at the executed speed does not close it and reads HARMS (HISTSWAP-LF, HCV), and the same history at the logged speed closes 0.599 to 0.633 (HLVL), the card's reading (B) level.")

OLD_ABS_END = "The cause is upstream of the decode; the executed-state reads (HISTSWAP and later) come in v36."
NEW_ABS_END = "The cause is upstream of the decode; the executed-state reads are the next bullet (§7.35)."
OLD_ABS_ANCHOR = "* **What is next.** The crawl set is still open and the adopted base stays DAG-OACC."
NEW_ABS = (
    "* **The M1 excess enters through the ego-history input, and the speed level is what the model lacks (§7.35; two zero-training reads of 2026-11-02, T1-kin, ORACLE diagnostics except HCV; paired episode-cluster bootstrap, B = 2,000, seed 0, eval-141, DAG-OACC and SOUP at s0 and s1; **[M]**).** "
    "HISTSWAP replaces the network's ego-history input by the logged history at every tick, the plan staying committed to the executed state: M1 crawl share X − Y is separated-better at 4 of 4 cells (−0.0476 / −0.0523 at DAG-OACC, −0.0612 / −0.0537 at SOUP) and closes 0.918 to 0.984 of the Y − P10 excess (mean 0.939; bar 0.5), H SUPPORTED, with D1, D2 and D3 of Read A true at all four cells and no row of X − Y separated-worse; X is an oracle and its absolute rows are not performance. "
    "HISTSWAP-LF splits the closure with a constant-twist history: at the executed speed (HCV, deployable) M1 X − Y is separated-WORSE at 4 of 4 cells (+0.0201 to +0.0270; share closed −0.559 to −0.301; ADE6 X − Y +0.5867 to +0.6527 m against a +0.0604 m guard), at the logged speed (HLVL, an oracle, level only) it is separated-better at 4 of 4 (−0.0422 to −0.0289; share closed 0.599 to 0.633; ADE6 X − Y −0.6165 to −0.5171 m). "
    "The card's reading is (B) level, and the outcome label is HARMS for both arms on both bases (HLVL on the S-POS rows alone). HLVL's closure is near-mechanical (the prior's anchor carries the plan's end point and O's delta moves against it) and HCV removes all acceleration information, so trend alone is untested. "
    "The lever both cards point to, the recovery fine-tune of O, is read in a later section.\n")
OLD_ABS_NEXT_END = "§7.33 reads the next perception step, the fine-token probes."
NEW_ABS_NEXT_END = OLD_ABS_NEXT_END + " §7.34 and §7.35 move the T1-kin evidence from the decode to the speed level the model is given; the training-side lever they point to (the recovery fine-tune of O) is not read in this version."

OLD_811 = "\n\n### 8.2 What is not established"
NEW_811 = (
    "\n23. **The M1 excess enters through the ego-history input, and the card's reading is that the model lacks the speed level; the oracle swaps are diagnostics** (§7.35; T1-kin, paired episode-cluster bootstrap, B = 2,000, seed 0, eval-141, one training run per seed). "
    "**Established (MEASURED):** replacing the ego-history input by the logged history closes 0.918 to 0.984 of the Y − P10 M1 excess at 4 of 4 cells (mean 0.939), H SUPPORTED, with D1, D2 and D3 of Read A true and no row of X − Y separated-worse; a constant-twist history at the executed speed (HCV) is separated-worse on M1 at 4 of 4 cells (share closed −0.559 to −0.301) with ADE6 X − Y +0.5867 to +0.6527 m; the same history at the logged speed (HLVL) closes 0.599 to 0.633 with ADE6 X − Y within the guard; the card's reading is (B) level, outcome HARMS for both arms (HLVL on the S-POS rows alone, UNVERIFIED rows). "
    "**Not established:** that a model without the human's speed recovers any of the closure (both closing arms are oracles), that O depends on the level rather than the prior's anchor following the input speed (HLVL's closure is near-mechanical), that trend alone does not help (HCV removes all acceleration information; HTRND not run), the split of the 0.329 that the full logged history adds over HLVL, the frame-freeze control (not run on any arm), and anything about re-perception (T1-kin only; no T2 number exists). One training run per seed, no replicate arm (`H-ESTIM-SEED-1`); the same 141 episodes serve every read of §7.34 and §7.35.\n"
    "\n### 8.2 What is not established")
OLD_I22 = "(the HISTSWAP reads come in v36; the lateral shrink and the training-time clamp are blocked, §7.34)"
NEW_I22 = "(the executed-state reads are §7.35; the lateral shrink and the training-time clamp are blocked, §7.34)"

OLD_83 = "**Update (v35).** Three items are written up here, and the adopted base is unchanged (DAG-OACC):"
NEW_83 = (
    "**Update (v36).** Two items are written up here, and the adopted base is unchanged (DAG-OACC):\n"
    "* **The excess enters through the ego-history input (§7.35, HISTSWAP, T1-kin, oracle):** replacing that input by the logged history closes 0.918 to 0.984 of the M1 excess at 4 of 4 cells (mean 0.939; M1 X − Y −0.0476 / −0.0523 at DAG-OACC), H SUPPORTED, with D1, D2 and D3 true at every cell. X is an oracle; its absolute rows (ADE6 X − Y −1.2325 to −1.2971 m) are not performance.\n"
    "* **The leak-free split (§7.35, HISTSWAP-LF, T1-kin):** a constant-twist history at the executed speed (HCV, deployable) does not close M1 and reads HARMS (M1 +0.0201 to +0.0270, ADE6 +0.5867 to +0.6527 m); at the logged speed (HLVL, oracle) it closes 0.599 to 0.633. The card's reading is (B) level. The dominance bar stays unmet: D2 fails on M1 at all 8 cells of both arms. The lever the cards point to, the recovery fine-tune of O, is read in a later section.\n\n"
    "**Update (v35).** Three items are written up here, and the adopted base is unchanged (DAG-OACC):")
OLD_83B = "The executed-state reads come in v36."
NEW_83B = "The executed-state reads are in §7.35."

OLD_1AM = "Next: the executed-state question (HISTSWAP, v36); the lateral shrink"
NEW_1AM = "Next: the executed-state question (HISTSWAP, read as 1an); the lateral shrink"
OLD_ROWS = "| 2 | **LoRA on DINOv3**"
NEW_ROWS = (
    "| 1an | **HISTSWAP** (zero training, ORACLE; `H-HISTSWAP-1`) | card `7e91a7c7…`, frozen 2026-11-02T17:57:30+0100: SUPPORTED iff M1 X − Y is separated-better at 4 of 4 cells and the mean share of the Y − P10 excess closed is ≥ 0.5; NOT SUPPORTED iff separated-better at fewer than 3 of 4 cells; else MIXED | "
    "**SUPPORTED** (§7.35.1, T1-kin, MEASURED; X is an oracle, not performance): M1 X − Y −0.0476 / −0.0523 (DAG-OACC), −0.0612 / −0.0537 (SOUP), 4 of 4 separated-better; share closed 0.918 to 0.984, mean 0.939; ADE6 X − Y −1.2325 to −1.2971 m (leak); controls PASS. Next: the leak-free split (read as 1ao) |\n"
    "| 1ao | **HISTSWAP-LF** (zero training; `H-HISTSWAP-LF-1`): constant-twist history at the executed speed (HCV, deployable) and at the logged speed (HLVL, oracle, level only) | card `d98db980…`, frozen 2026-11-02T18:50:07+0100: HCV CLOSES ⇒ (A) inertia and a deployable lever; HCV not and HLVL CLOSES ⇒ (B) level (the speed level must come from perception or training); neither ⇒ (C) interaction (HTRND); CLOSES = 4 of 4 separated-better and mean share closed ≥ 0.5 | "
    "**Reading (B) level** (§7.35.2, T1-kin, MEASURED): HCV NOT CLOSES, M1 separated-worse at 4 of 4 cells (+0.0201 to +0.0270), share closed −0.559 to −0.301, ADE6 +0.5867 to +0.6527 m; HLVL CLOSES, 0.599 to 0.633, ADE6 −0.6165 to −0.5171 m; outcome HARMS for both arms (HLVL on the S-POS rows alone); controls PASS. Next: the recovery fine-tune of O with re-expressed targets (PI go; read in a later section) |\n"
    + OLD_ROWS)

OLD_9 = "\n\n---\n\n## 10. Reproducibility appendix"
NEW_9 = (
    "\n14. **v36 reads and their limits.** The two reads of §7.35 use the same 141 eval episodes (2,399 windows) as §7.34, one training run per seed (DAG-OACC and SOUP at s0 and s1), no replicate arm, and an interval that answers another draw of episodes only (`H-ESTIM-SEED-1`); X and Y share a checkpoint, so no training-seed variance sits inside X − Y. "
    "Both closing arms (HISTSWAP and HLVL) are oracles that read the human's speed, and HISTSWAP also leaks the human's realised motion at ticks 1 and 2. HLVL's closure is near-mechanical: the plan is `P + O's delta` and the prior's anchor follows the input speed. HCV removes all acceleration information, so *trend alone* is untested (HTRND not run), and whether its harm is the missing information or an off-distribution input is UNVERIFIED. "
    "The class split of the M1 row is by situation class and not by v0 band. The S-POS rows of the swapped arms are UNVERIFIED, and HLVL's HARMS label rests on them alone. T1-kin applies the model's own plan to its own state with logged cameras that advance at the human's pace; the frame-freeze control was not run on any arm of these reads. "
    "The card texts were hashed (sha256) and their rules quoted from the RESULTS files and the register; the card texts were not re-read line by line in v36 beyond the quoted passages. FRAMEFREEZE, RECOVER-FT and RECOVER-NEXT are not read in v36.\n"
    + OLD_9)
ANCHOR_8 = "\n\n---\n\n## 8. Discussion"
OLD_I13 = "HISTSWAP, HISTSWAP-LF, FRAMEFREEZE, RECOVER-FT and RECOVER-NEXT are not read in v35 (v36)."
NEW_I13 = "HISTSWAP and HISTSWAP-LF are read in v36 (§7.35, item 14); FRAMEFREEZE, RECOVER-FT and RECOVER-NEXT are not read in v36."
# pointer edits inside §7.34 (they said the HISTSWAP reads were not read in that version)
P1 = ("(`H-HISTSWAP-1`; not read in this version)", "(`H-HISTSWAP-1`; read in §7.35)")
P2 = ("the executed-state question was taken up by the separate package `../histswap`, which this version does not read.", "the executed-state question was taken up by the separate package `../histswap`, which is read in §7.35.")
P3 = ("(6) HISTSWAP, HISTSWAP-LF, FRAMEFREEZE, RECOVER-FT and RECOVER-NEXT are read in v36; no claim about their states is made here (**UNVERIFIED** in v35).",
      "(6) HISTSWAP and HISTSWAP-LF are read in §7.35; FRAMEFREEZE, RECOVER-FT and RECOVER-NEXT are not read in this version, and no claim about their states is made here (**UNVERIFIED**).")
P4 = ("The executed-state question is the next read (v36).", "The executed-state question is read in §7.35.")

E = [(OLD_ST_A, NEW_ST_A), (OLD_ST_B, NEW_ST_B), (OLD_ABS_END, NEW_ABS_END), (OLD_ABS_ANCHOR, NEW_ABS + OLD_ABS_ANCHOR), (OLD_ABS_NEXT_END, NEW_ABS_NEXT_END),
     (ANCHOR_8, "\n\n" + SEC.rstrip("\n") + ANCHOR_8), (OLD_811, NEW_811), (OLD_I22, NEW_I22), (OLD_83, NEW_83), (OLD_83B, NEW_83B), (OLD_1AM, NEW_1AM), (OLD_ROWS, NEW_ROWS), (OLD_I13, NEW_I13), (OLD_9, NEW_9),
     P1, P2, P3, P4]
for a, b in E:
    assert s.count(a) == 1, (s.count(a), a[:90])
    s = s.replace(a, b)

# ---- guards on the inserted text ---------------------------------------------------------------------------------------------------------------------
ins = SEC + NEW_ABS + NEW_811 + NEW_83 + NEW_ROWS + NEW_9 + NEW_ST_A + NEW_ST_B + NEW_I13 + "".join(b for _, b in (P1, P2, P3, P4))
assert not _BAN.search(ins), _BAN.search(ins)
low = ins.lower()
assert re.search(r"retract|supersed|withdrawn|was wrong|previously read|outdated|i said", low) is None
assert "closed loop" not in low and "closed-loop" not in low and not re.search(r"\bdrives\b|driving performance", low), "closed-loop or driving wording"
assert "$" not in ins and "\r" not in ins, "a dollar sign or a CR in the inserted text"
nocode = re.sub(r"`[^`\n]*`", " ", ins)
assert re.search(r"<[A-Za-z/!]", nocode) is None, re.search(r".{20}<[A-Za-z/!].{20}", nocode)
assert "@@" not in s and "</script" not in s.lower() and "\r" not in s
_flat = re.sub(r"\s+", " ", SEC)
assert "B = 2,000" in _flat and "T1-kin" in _flat and "ORACLE" in _flat and "H-ESTIM-SEED-1" in _flat
assert s.count("### 7.35 ") == 1 and s.index("### 7.34 ") < s.index("### 7.35 ") < s.index("## 8. Discussion")
assert s.count("**Status:** research report, v36 of 2026-11-04") == 1 and "v35 of 2026-11-03, results" not in s.split("\n")[2]
h = tpl.replace("%%MD%%", s)
assert h.count("%%MD%%") == 0
os.makedirs(OUT, exist_ok=True)
open(OUT + "V8F_SCIENTIFIC_REPORT.md", "w", encoding="utf-8", newline="\n").write(s)
open(OUT + "v8f_report.html", "w", encoding="utf-8", newline="\n").write(h)
print("v36 built", len(S0), len(s), "tip", TIP, "section 7.35 chars", len(SEC), "tables", len(T), "table lines", sum(len(v) for v in T.values()))
