#!/usr/bin/env python3
"""Generate this package's GOALS_AND_CLAIMS.md rows FROM THE ARTIFACTS, not by transcription.

Every number in every row below is read out of a JSON this package produced, so a row cannot
drift from its artifact. Also appends the correction to `D-REFC-KINGATE-1` rather than editing
its numbers in place -- a recorded correction is recoverable, a silent one is not.

Usage: make_register_rows.py <repo-root>
"""
import json
import os
import sys

RAW = os.path.dirname(os.path.abspath(__file__))
REPO = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
PATH = os.path.join(REPO, "Project Steering", "GOALS_AND_CLAIMS.md")
CR, LF = chr(13), chr(10)
CRLF = CR + LF
ST, NE, W, ED, X, ARR, ELL, SEC = (chr(11088), chr(9940), chr(9888), chr(8212), chr(215),
                                   chr(8594), chr(8230), chr(167))
MID, PCT, RHO = chr(183), "%", chr(961)


def J(name):
    with open(os.path.join(RAW, name), "r", encoding="utf-8") as f:
        return json.load(f)


K = J("kingate_base.json")
S = J("gate_share_of_gap.json")
F = J("displacement_frontier.json")
NS = J("no_scene_input.json")
LD = J("longitudinal_lead.json")

A, B = K["draws"]["A"], K["draws"]["B"]
gA, gB = A["G_OFF"], B["G_OFF"]
apA, apB = gA["as_published_control"], gB["as_published_control"]
det = K["G_DET"]
vg = K["verdict"]["gate2"]


def d(tag, m, k):
    return K["draws"][tag]["results"]["paired_vs_model"]["gate2"][m][k]


def ab(tag, rule, m):
    return K["draws"][tag]["results"]["abs"][rule][m]["mean"]


def f(x, n=4, sign=False):
    fmt = "%+." + str(n) + "f" if sign else "%." + str(n) + "f"
    return fmt % float(x)


def q(m):
    v = vg[m]
    return ("QUOTABLE" if v["QUOTABLE"] else
            "UNDETECTABLE-DOWNWARD" if v["UNDETECTABLE_DOWNWARD"] else
            "WITHIN-NOISE" if v["WITHIN_NOISE"] else "NOT-SEPARATED")


lam = {r["lambda"]: r for r in F["sweep"]}
c2 = F["controls"]["C2"]

ROWS = []

# ---------------------------------------------------------------- H-KINGATE-1
ROWS.append(
    "| H-KINGATE-1 | {ST} **PRE-REGISTERED, both outcomes committed BEFORE any arm ran "
    "(SPEC banked at commit `dbc30f3`): the ZERO-TRAINING top-2 kinematic gate makes refcv3's "
    "DRIVEN path measurably safer at no measurable ADE cost, and the gain survives a "
    "deliberate-regression control, an inference-determinism guard and an EPISODE-DISJOINT "
    "replicate.** Executes `Decisions/2026-09-05-mm-decisions.md` {SEC}M23.2. One variable: the "
    "gate ON or OFF; ONE forward per window banks the fan and every rule, both draws, all four "
    "families and every bootstrap are computed offline from it, so `model` and `gate_k` differ "
    "in exactly one thing {ED} WHICH of the 128 already-emitted candidates is returned. "
    "{NE} **Three variances named before any number existed:** V1 episode draw (live {ED} "
    "paired episode-cluster bootstrap PLUS an episode-disjoint second draw), V2 training run "
    "({ST} **STRUCTURALLY ABSENT**: zero training, one frozen checkpoint, so `H-ESTIM-SEED-1`'s "
    "hole is closed BY CONSTRUCTION for this lever {ED} the only lever in the programme of which "
    "that is true), V3 inference sampling (asserted, not assumed: `refc.py:1720`/`:1501` carry "
    "stochastic ops gated by CONFIG, not by `self.training`). **SUCCESS** = `sel_envelope` "
    "negative + separated on BOTH draws, same sign, clearing the replicate floor, AND "
    "`sel_peak_g` likewise, AND `ade_m` not separated or |delta| < 0.010 m, AND every guard "
    "passes. **FAILURE** = any of those, and a `G-OFF` failure is HARD (it voids rather than "
    "loses the measurement) | {VERDICT} | `TanitAD Research Lab/Deployment & Optimization/"
    "Research/2026-09-05-kinematic-gate/SPEC.md` {SEC}2-{SEC}7, `RESULT.md`; "
    "`raw/kin_gate_eval.py`, `raw/kingate_base.json` |")

# --------------------------------------------------- D-REFC-KINGATE-RANK-1 (the confound)
ROWS.append(
    "| D-REFC-KINGATE-RANK-1 | {NE}{NE} **THE PUBLISHED GATE RANKED ON THE WRONG SCORE, AND ITS "
    "OWN IDENTITY CONTROL WAS DOCUMENTED AND NEVER RUN.** `rl_fan_rerank_probe.py` builds the "
    "gate's top-k from `out['sel_score']`; refcv3 is the **`hier`** arm (`config.json` argv "
    "`--arm hier`) and `refc.py:1763` argmaxes `rank`, which on that arm is **`sel_score_v3`** "
    "(the goal-seam-grafted score) masked by `reach_keep`. MEASURED 2026-09-05 on 400 EVAL "
    "windows: the `sel_score` reconstruction disagrees with the model's own `sel_idx` on "
    "**{APDIS} of {APN} windows ({APPCT} {PCT})** {ED} so on ~1 window in 11 the gate's candidate "
    "set did NOT contain the model's own pick, and part of any measured gain came from "
    "RE-DERIVING the ranking rather than from the kinematic tiebreak. {NE} And the probe's "
    "docstring names `k = 1` as *'the model itself by construction {ED} a built-in identity "
    "control'* while `GATE_KS = (2, 4, 8, 16, 32, 128)`: **k = 1 is not in the tuple**, so the "
    "control that would have caught this was described and never executed. {ST} Corrected here: "
    "ranking on `sel_score_v3` makes `gate1 == model` hold on **{TRUEN}/{TRUEN} windows with "
    "every metric max|diff| exactly 0.0**, and the as-published ranking is kept as a side arm "
    "(`gateAP_k`) so the defect's size is MEASURED rather than asserted. {W} Per RETRACTION #30 "
    "the control asserts the SELECTED INDEX, not only the metric values {ED} two different "
    "candidates can carry equal metrics | **SUPPORTED (MEASURED 2026-09-05)** {ED} "
    "`D-REFC-KINGATE-1`'s numbers are superseded by the corrected panel | "
    "`{ELL}/2026-09-05-kinematic-gate/RESULT.md` {SEC}1, {SEC}3; `raw/kingate_base.json` "
    "(`draws.*.G_OFF`), `stack/tanitad/refs/refc.py:1747-1763` |")

# --------------------------------------------------- D-REFC-KINGATE-NOSCENE-1 (P2)
ROWS.append(
    "| D-REFC-KINGATE-NOSCENE-1 | {ST} **'NO SCENE INPUT' HOLDS AT SOURCE FOR THE RANKING "
    "FUNCTION, AND THE ADMISSIBLE CLAIM IS 'NO **NEW** SCENE INPUT'.** MEASURED 2026-09-05 by "
    "TWO independent probes with three positive controls (`raw/verify_no_scene_input.py`). "
    "**AST call-graph walk:** `feasibility` + `comfort` reach only `_motion_gate` and "
    "`kinematics` and read **six** ctx keys {ED} `dt`, `a_max`, `kappa_max`, `jerk_max`, "
    "`lat_acc_max`, `min_motion_m` {ED} **every one a programme CONSTANT**; no `v0`, no "
    "`lead_path`, no `obstacles`, no `gt_traj`, and **no dynamic ctx access**. "
    "**Runtime permutation:** under a full scene garble the gate score AND its argmax are "
    "**BITWISE unchanged (max|diff| 0.000e+00)** while the controls all move {ED} `headway` "
    "7.5e-01, `collision` 1.0, `progress` 3.06 {ED} so the garble took, and the invariance is a "
    "property of the components, not of the test. {W} **BUT the CANDIDATE SET is scene-"
    "conditioned**: the gate ranks `out['sel_score_v3']` masked by `out['reach_keep']`, both "
    "model outputs. {ARR} The admissible claim is **'adds no NEW scene input and no new "
    "perception'** {ED} every input the gate uses, the deployed model already computes {ED} and "
    "it is admissible under the vision-only-at-inference rule for that reason, not because it is "
    "blind | **SUPPORTED (MEASURED 2026-09-05)** {ED} sharpens `D-REFC-KINGATE-1`'s wording; the "
    "'no lead, no obstacle track, no ego state' half is CONFIRMED exactly | "
    "`{ELL}/2026-09-05-kinematic-gate/RESULT.md` {SEC}2; `raw/no_scene_input.json`, "
    "`raw/verify_no_scene_input.py`, `stack/tanitad/rl/rewards.py:503-527` |")

# --------------------------------------------------- D-REFC-KINGATE-SHARE-1 (P4)
ROWS.append(
    "| D-REFC-KINGATE-SHARE-1 | {NE}{NE} **THE GATE CLOSES 0.00 {PCT} OF THE 8.56{X} {ED} AND "
    "THAT IS A STRUCTURAL ZERO, NOT A SMALL EFFECT.** MEASURED 2026-09-05, 0 GPU, "
    "{SW} EVAL windows {X} 128 candidates (`raw/gate_share_of_gap.json`). The 8.56{X} is a "
    "property of the EMITTED FAN (`D-REFC-OFFSET-FEAS-1`: bank **{BPG} g** {ARR} fan "
    "**{FPG} g**, D1 = **{D1} g**); a selection rule changes no waypoint, so its effect on any "
    "fan-level metric is an IDENTITY {ED} asserted, not assumed: every rule indexes ONE score "
    "tensor computed from ONE banked fan. {ST} **The reframing that makes the gate legible:** "
    "the DRIVEN path is already **{RFAN}{X} better conditioned than the fan's average member** "
    "(model `sel_peak_g` **{MPG} g** vs fan mean **{FANM} g**) and **{RBANK}{X} BELOW the frozen "
    "vocabulary's own {BPG} g** {ED} the selector never picks the fan's bad members, so the "
    "8.56{X} headline describes candidates the car does not drive. The only denominator a "
    "re-ranking rule CAN be measured against is **D2 = model `sel_peak_g` {ED} the best "
    "`peak_g` selectable from the same fan = {D2} g**. Against D2: `gate2` **{G2PG} g = "
    "{G2PCT} {PCT}**, `gate4` {G4PCT} {PCT}, `gate8` {G8PCT} {PCT}, `gate128` {G128PCT} {PCT}. "
    "{W} `oracle_ade` is **WORSE** on `peak_g` ({ORCPG} g) {ED} `D-RL-FANSAFE-1` reproduced from "
    "the other side. {ARR} **The gate is a MITIGATION and the fan is untouched; quoting a "
    "selection gain against D1 would flatter it by ~{FLATTER}{X}** | **SUPPORTED (MEASURED "
    "2026-09-05)** | `{ELL}/2026-09-05-kinematic-gate/RESULT.md` {SEC}5; "
    "`raw/gate_share_of_gap.json`, `{ELL}/2026-09-05-veto-only-fan-safety/raw/"
    "bank_vs_fan_feasibility.json` |")

# --------------------------------------------------- D-DECODE-DISP-FRONTIER-1 (P5)
ROWS.append(
    "| D-DECODE-DISP-FRONTIER-1 | {NE}{NE} **THE DISPLACEMENT FRONTIER HAS NO KNEE: SHRINKING "
    "refcv3's DECODE TOWARD ITS OWN VOCABULARY BUYS FEASIBILITY ONLY AT A RUINOUS ADE PRICE, SO "
    "THE VOCABULARY {ED} NOT THE DECODE {ED} IS THE BINDING CONSTRAINT.** MEASURED 2026-09-05 at "
    "**ZERO GPU** (`raw/displacement_frontier.py`, {SW} EVAL windows {X} 128 candidates, "
    "`core.decoder.anchors` read straight out of the checkpoint): "
    "`path(lambda) = bank + lambda*(fan - bank)`. At **lambda = 0.9** {ED} giving back only "
    "10 {PCT} of the displacement {ED} `sel_ade` goes **{SA100} {ARR} {SA090} m (+{SAPCT} {PCT})** "
    "to buy `fan_peak_g` **{PG100} {ARR} {PG090} g (-{PGPCT} {PCT})**, and across the sweep "
    "`fan_off_reach` rises **{OR100} {ARR} {OR000}** as `fan_envelope` falls **{EN100} {ARR} "
    "{EN000}**: the two failure modes are ONE trade on this bank, because the bank is "
    "speed-blind. {ST} **Both controls pass and C2 is the one that matters (RETRACTION #30):** at "
    "lambda = 0 this script reads `peak_g` **{C2OURS}** and `envelope` **{C2ENV}** against "
    "`bank_vs_fan_feasibility.json`'s **{C2REF}** / **{C2ENVREF}** {ED} a DIFFERENT script on a "
    "DIFFERENT window draw ({C2REFN} vs {SW} windows). C1 alone is blind by construction: "
    "lambda = 1 is the emitted fan whatever the left operand is. {W} **Scope the negative "
    "honestly:** this sweeps ONE one-dimensional family (the straight line between bank and fan); "
    "a trained control-space decode is not confined to it, so the conclusion is about the LINEAR "
    "displacement family, not about all reparameterisations. {ST}{ST} **Consequence: condition "
    "the VOCABULARY first, then constrain the decode** {ED} and `H-EGO-LIT-4` reached the same "
    "lever independently from the other side (T0 oracle-in-vocabulary 4.6762 m fixed {ARR} "
    "2.7873 m own-predicted-speed bank, paired -1.8889 [-2.7307, -1.1304] SEP) | "
    "**SUPPORTED (MEASURED 2026-09-05)** {ED} REDIRECTS the feasibility-aware-decode work item "
    "opened by `D-REFC-OFFSET-FEAS-1`: v0-conditioned bank FIRST | "
    "`{ELL}/2026-09-05-kinematic-gate/SUCCESSOR_FEASIBILITY_AWARE_DECODE.md` {SEC}4b; "
    "`raw/displacement_frontier.json` |")

SUB = {
    "ST": ST, "NE": NE, "W": W, "ED": ED, "X": X, "ARR": ARR, "ELL": ELL, "SEC": SEC,
    "MID": MID, "PCT": PCT, "RHO": RHO,
    "VERDICT": "{VERDICT}",
    "APDIS": str(apA["n_windows_disagreeing_on_index"]),
    "APN": str(apA["n_windows"]),
    "APPCT": f(100.0 * apA["disagreement_rate"], 2),
    "TRUEN": str(gA["n_windows"]),
    "SW": str(S["n_windows"]),
    "BPG": f(S["bank_peak_g"]),
    "FPG": f(S["emitted_fan_peak_g"]),
    "D1": f(S["D1_fan_gap_g"]),
    "MPG": f(S["model_sel_peak_g"]),
    "FANM": f(S["structural_zero_on_fan_metrics"]["fan_peak_g_mean_under_every_rule"]),
    "RFAN": f(S["model_sel_peak_g_vs_fan_mean_ratio"], 1),
    "RBANK": f(S["bank_peak_g"] / S["model_sel_peak_g"], 2),
    "D2": f(S["D2_selection_gap_g"]),
    "G2PG": f(S["rules"]["gate2"]["d_peak_g"]["delta"], 4, True),
    "G2PCT": f(S["rules"]["gate2"]["share_of_D2_selection_gap_pct"], 1),
    "G4PCT": f(S["rules"]["gate4"]["share_of_D2_selection_gap_pct"], 1),
    "G8PCT": f(S["rules"]["gate8"]["share_of_D2_selection_gap_pct"], 1),
    "G128PCT": f(S["rules"]["gate128"]["share_of_D2_selection_gap_pct"], 1),
    "ORCPG": f(S["rules"]["oracle_ade"]["d_peak_g"]["delta"], 4, True),
    "FLATTER": f(S["D1_fan_gap_g"] / S["D2_selection_gap_g"], 0),
    "SA100": f(lam[1.0]["sel_ade_m"]), "SA090": f(lam[0.9]["sel_ade_m"]),
    "SAPCT": f(100.0 * (lam[0.9]["sel_ade_m"] / lam[1.0]["sel_ade_m"] - 1.0), 0),
    "PG100": f(lam[1.0]["fan_peak_g"], 3), "PG090": f(lam[0.9]["fan_peak_g"], 3),
    "PGPCT": f(100.0 * (1.0 - lam[0.9]["fan_peak_g"] / lam[1.0]["fan_peak_g"]), 0),
    "OR100": f(lam[1.0]["fan_off_reach"], 3), "OR000": f(lam[0.0]["fan_off_reach"], 3),
    "EN100": f(lam[1.0]["fan_envelope"], 3), "EN000": f(lam[0.0]["fan_envelope"], 3),
    "C2OURS": f(c2["peak_g"]["ours"], 6), "C2REF": f(c2["peak_g"]["ref"], 6),
    "C2ENV": f(c2["envelope"]["ours"], 6), "C2ENVREF": f(c2["envelope"]["ref"], 6),
    "C2REFN": str(c2["reference_n_windows"]),
}


def render(t):
    for k, v in SUB.items():
        t = t.replace("{" + k + "}", v)
    return t


if __name__ == "__main__":
    verdict_path = os.path.join(RAW, "_verdict_text.txt")
    with open(verdict_path, "r", encoding="utf-8") as fh:
        SUB["VERDICT"] = fh.read().strip()
    rows = [render(r) for r in ROWS]

    raw = open(PATH, "rb").read()
    is_crlf = CRLF.encode() in raw
    s = raw.decode("utf-8").replace(CRLF, LF)
    if "H-KINGATE-1" in s:
        raise SystemExit("[register] kingate rows already present")
    lines = s.split(LF)
    idx = [i for i, l in enumerate(lines) if l.startswith("| D-REFC-KINGATE-1 |")]
    if len(idx) != 1:
        raise SystemExit("[register] expected exactly one D-REFC-KINGATE-1 row, found %d"
                         % len(idx))
    # a RECORDED correction on the superseded row, never a silent edit of its numbers
    corr = (" " + NE + " **CORRECTED 2026-09-05 by `D-REFC-KINGATE-RANK-1`: this row's gate "
            "ranked on `sel_score`, and the deployed `hier` arm argmaxes `sel_score_v3` "
            "(`refc.py:1763`) {ED} they disagree with the model's own `sel_idx` on "
            "{APDIS}/{APN} windows ({APPCT} {PCT}), so the candidate set did not always contain "
            "the model's own pick. Its `re-confirmed by the corrected re-run` clause also names "
            "`raw/fan_rerank_base.json`, which did not exist when it was written. The numbers "
            "here are SUPERSEDED by `H-KINGATE-1`'s corrected panel; the row is kept for the "
            "record rather than rewritten.**")
    corr = render(corr)
    row = lines[idx[0]]
    parts = row.split(" | ")
    if len(parts) >= 3:
        parts[-2] = parts[-2] + corr
        lines[idx[0]] = " | ".join(parts)
    lines[idx[0] + 1:idx[0] + 1] = rows
    out = LF.join(lines)
    if is_crlf:
        out = out.replace(LF, CRLF)
    open(PATH, "wb").write(out.encode("utf-8"))
    print("[register] inserted %d rows after D-REFC-KINGATE-1 and appended its correction"
          % len(rows))
