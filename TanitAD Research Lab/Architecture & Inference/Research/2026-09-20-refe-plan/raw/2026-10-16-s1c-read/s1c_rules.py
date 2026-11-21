#!/usr/bin/env python3
"""s1c_rules.py -- the DECISION RULES of eval/PREREG_S1C.md section 4 and eval/PREREG_AUXVETO.md section 6 (+ amendment A1), applied MECHANICALLY.  stdlib only (no numpy): it runs in any venv.

Nothing here measures anything.  It takes the already-measured reads and returns the prereg's row, with the row's committed NEXT lever QUOTED VERBATIM from the prereg file itself
(the table cell is parsed at run time, never retyped).  Every place where the prereg's words admit two readings is in QUESTIONS.md, resolved BEFORE any S1c number existed.

PREREG_S1C section 4 (read in this order; O3 before O4, as the table lists them):
    INVALID   C1 fails (step0 held-out IoU != S1 H1-s0's 0.1586791569036157 +-1e-3) or the chain's run-check refused or the arm's evidence contradicts its label
              [QUESTIONS Q2: validity is evaluated FIRST, because 'no S1c read' and O3's own clause 'and C1 holds' both presuppose it]
    O1        P1_C3 >= 0.80 and P1_NAT >= 0.80
    O2        Delta_C3 and Delta_NAT both separated and > 0   (and not O1, i.e. P1_C3 < 0.80 or P1_NAT < 0.80)
    O3        Delta_C3 or Delta_NAT not separated  (not separated := the paired CI includes 0 or |Delta| <= F)
    O4        Delta_C3 or Delta_NAT separated and < 0
  separated := the paired CI excludes 0 AND |Delta| > F;  F = 0.035 (C3), 0.023 (NAT) (section 3, MEASURED from S1's three H1 seeds).
  If O3 and O4 BOTH match (one population adverse, the other not separated) the row is O3 (table order) and the overlap is FLAGGED with O4's committed next lever beside it.
  K2 itself: if P1_C3 and P1_NAT disagree on K2, the C3 read governs (section 4 last line); both numbers are reported with their population named.
PREREG_AUXVETO section 6, for V-S1C in V-PRED's role (amendment A1: 'read by the same section 6 decision list, as its own row'), V-GT by the same estimator:
    AV-GO  lo > 0 and point >= 0.30 | AV-PROMISING  lo > 0 and point < 0.30 | AV-HARM  hi < 0 | AV-CEILING-ONLY  not separated and V-GT lo > 0 | AV-NULL  otherwise.
"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

K2_BAR = 0.80
F_C3, F_NAT = 0.035, 0.023                               # PREREG_S1C section 3: the largest pairwise |difference| between S1's three H1 seeds (MEASURED)
C1_STEP0_IOU, C1_TOL = 0.1586791569036157, 1e-3          # PREREG_S1C section 2, C1
AV_GO_POINT = 0.30                                       # PREREG_AUXVETO section 6

# the prereg files in force.  Full-text sha256 of the LF-normalised bytes at the landing of the S1c registration (10d4fae); any dated amendment is APPENDED as a '## Amendment' heading
# and is listed (verbatim heading) in the result; the BODY above the first amendment must still hash to the pin.
PREREG_S1C_BODY_SHA256 = "317584a3f7e82ee8cb7fbffc0e44ff1bf4245988fa3f2fa7b8feb28ab004416a"
# PREREG_AUXVETO: the body above '## Amendment A1' normalised exactly as the read of 2026-10-16 saw it (rstrip + one newline, CRLF form) is `ef9f17fa26a9d818...`, the prefix auxveto_read.py asserted;
# the A1 section itself is pinned separately so a later edit to it is seen.
PREREG_AUXVETO_BODY_CRLF_PREFIX = "ef9f17fa26a9d81817a7a992d2577eebae4f382eda3ed549980f2c079d23a518"
PREREG_AUXVETO_A1_SHA256 = "2b26ccadd2095711e6189f41224aaa38fbe3c586b5641aff19b3c1c84e870d4b"


# ================================================================================================================ reading the prereg files
def read_lf(path) -> str:
    return Path(path).read_bytes().decode("utf-8").replace("\r\n", "\n")


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def split_amendments(text: str):
    """-> (body, [amendment sections]).  An amendment starts at a line beginning '## Amendment'."""
    idx = [m.start() for m in re.finditer(r"^## Amendment", text, re.M)]
    if not idx:
        return text, []
    return text[:idx[0]], [text[a:b] for a, b in zip(idx, idx[1:] + [len(text)])]


def guard_prereg_s1c(path) -> dict:
    """Refuse (ValueError) unless the S1c prereg's body is the registered one.  Returns the provenance incl. any amendment headings (verbatim)."""
    text = read_lf(path)
    body, am = split_amendments(text)
    got = sha256_text(body.rstrip(chr(10)) + chr(10))                                  # one trailing newline: an appended amendment may be preceded by a blank line
    if got != PREREG_S1C_BODY_SHA256:
        raise ValueError(f"REFUSED: {path} body sha256 {got[:16]} != the registered {PREREG_S1C_BODY_SHA256[:16]} (a change after landing needs a dated AMENDMENT, appended)")
    return {"path": str(path), "body_sha256": got, "full_sha256": sha256_text(text), "amendments": [a.splitlines()[0] for a in am]}


def guard_prereg_auxveto(path) -> dict:
    text = read_lf(path)
    body, am = split_amendments(text)
    norm = (body.rstrip("\n") + "\n").replace("\n", "\r\n")                      # the form whose sha256 auxveto_read.py asserted
    got = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    if got != PREREG_AUXVETO_BODY_CRLF_PREFIX:
        raise ValueError(f"REFUSED: {path} body (above the first amendment) sha256 {got[:16]} != the prereg the AUXVETO read was built on {PREREG_AUXVETO_BODY_CRLF_PREFIX[:16]}")
    a1 = [a for a in am if a.startswith("## Amendment A1")]
    if len(a1) != 1 or sha256_text(a1[0]) != PREREG_AUXVETO_A1_SHA256:
        raise ValueError(f"REFUSED: {path}: amendment A1 (arm V-S1C) is missing or edited (sha256 {sha256_text(a1[0])[:16] if a1 else None} != {PREREG_AUXVETO_A1_SHA256[:16]})")
    return {"path": str(path), "body_sha256_crlf_form": got, "a1_sha256": sha256_text(a1[0]), "full_sha256": sha256_text(text), "amendments": [a.splitlines()[0] for a in am]}


def table_row(text: str, label_prefix: str) -> list:
    """The RAW (unstripped) cells of the markdown table row whose first cell starts with `**<label_prefix>`.  A literal '|' inside a cell (e.g. '|Delta|') splits it, so callers index from BOTH ends
    and re-join with '|' (raw cells keep their original spacing, so the re-join is byte-faithful)."""
    for line in text.splitlines():
        if line.startswith(f"| **{label_prefix}"):
            return line.strip().strip("|").split("|")
    raise ValueError(f"no table row starting '| **{label_prefix}'")


def s1c_row_cells(text: str, label_prefix: str) -> dict:
    """PREREG_S1C section 4 row -> {label, rule, meaning, next, blocks}: five logical cells; the rule cell may contain '|' characters (O3: '|Delta| <= F'), so it is the join of the middle."""
    c = table_row(text, label_prefix)
    if len(c) < 5:
        raise ValueError(f"row {label_prefix}: {len(c)} cells, want >= 5")
    return {"label": c[0].strip().strip("*"), "rule": "|".join(c[1:-3]).strip(), "meaning": c[-3].strip(), "next": c[-2].strip(), "blocks": c[-1].strip()}


def auxveto_row_cells(text: str, label_prefix: str) -> dict:
    c = table_row(text, label_prefix)
    if len(c) != 4:
        raise ValueError(f"row {label_prefix}: {len(c)} cells, want 4")
    return {"label": c[0].strip().strip("*"), "condition": c[1].strip(), "reading": c[2].strip(), "next": c[3].strip()}


S1C_LABELS = {"O1": "O1 K2 CLEARED", "O2": "O2 PARTIAL", "O3": "O3 NO GAIN", "O4": "O4 WORSE", "INVALID": "INVALID"}


# ================================================================================================================ PREREG_S1C section 4
def separated(d: dict, F: float) -> str:
    """'pos' | 'neg' | 'none'.  separated := the paired CI excludes 0 AND |Delta| > F (both strict)."""
    est, (lo, hi) = d["est"], d["ci95"]
    if not (math.isfinite(est) and math.isfinite(lo) and math.isfinite(hi)):
        return "none"
    if lo > 0 and est > F:
        return "pos"
    if hi < 0 and -est > F:
        return "neg"
    return "none"


def c1_check(step0_iou: float, expected: float = C1_STEP0_IOU, tol: float = C1_TOL) -> dict:
    d = abs(step0_iou - expected)
    return {"ok": bool(d <= tol), "step0_iou": step0_iou, "expected": expected, "abs_diff": d, "tol": tol}


def decide_s1c(m: dict, prereg_text: str) -> dict:
    """m: {c1_ok, runcheck_ok, evidence_ok, p1_c3, p1_nat, d_c3:{est,ci95}, d_nat:{est,ci95}} -> the row, its conditions as evaluated, the overlap flag and the row's cells QUOTED from the prereg."""
    valid = bool(m["c1_ok"] and m["runcheck_ok"] and m["evidence_ok"])
    k2_c3 = m["p1_c3"] >= K2_BAR
    k2_nat = m["p1_nat"] >= K2_BAR
    s_c3 = separated(m["d_c3"], F_C3)
    s_nat = separated(m["d_nat"], F_NAT)
    conds = {
        "valid (C1 holds, run-check ok, evidence consistent)": valid,
        "O1: P1_C3 >= 0.80 and P1_NAT >= 0.80": bool(k2_c3 and k2_nat),
        "O2: Delta_C3 and Delta_NAT both separated and > 0 (and not O1)": bool(s_c3 == "pos" and s_nat == "pos" and not (k2_c3 and k2_nat)),
        "O3: Delta_C3 or Delta_NAT not separated": bool(s_c3 == "none" or s_nat == "none"),
        "O4: Delta_C3 or Delta_NAT separated and < 0": bool(s_c3 == "neg" or s_nat == "neg"),
    }
    if not valid:
        row = "INVALID"
    elif conds["O1: P1_C3 >= 0.80 and P1_NAT >= 0.80"]:
        row = "O1"
    elif conds["O2: Delta_C3 and Delta_NAT both separated and > 0 (and not O1)"]:
        row = "O2"
    elif conds["O3: Delta_C3 or Delta_NAT not separated"]:
        row = "O3"
    elif conds["O4: Delta_C3 or Delta_NAT separated and < 0"]:
        row = "O4"
    else:                                                  # unreachable: after O1/O2 fail, each population is pos/neg/none and every case lands in O3/O4
        raise AssertionError(f"no row matched: {s_c3=} {s_nat=} {k2_c3=} {k2_nat=}")
    overlap = []
    if row == "O3" and conds["O4: Delta_C3 or Delta_NAT separated and < 0"]:
        overlap.append("O4")
    cells = s1c_row_cells(prereg_text, S1C_LABELS[row])
    out = {"row": row, "label": S1C_LABELS[row], "separated": {"C3": s_c3, "NAT": s_nat}, "F": {"C3": F_C3, "NAT": F_NAT}, "conditions": conds,
           "k2": {"bar": K2_BAR, "C3": {"p1": m["p1_c3"], "cleared": bool(k2_c3)}, "NAT": {"p1": m["p1_nat"], "cleared": bool(k2_nat)},
                  "governing_verdict_C3": "K2 CLEARED" if k2_c3 else "K2 FIRES", "populations_disagree": bool(k2_c3 != k2_nat)},
           "remaining_gap_to_0.80": {"C3": K2_BAR - m["p1_c3"], "NAT": K2_BAR - m["p1_nat"], "worse_population": "C3" if m["p1_c3"] <= m["p1_nat"] else "NAT"},
           "overlap": overlap, "quoted_row": cells, "stamp": "separated, single seed, not yet a lever claim" if row in ("O2", "O3", "O4") else None}
    if overlap:
        out["overlap_rows_quoted"] = {o: s1c_row_cells(prereg_text, S1C_LABELS[o]) for o in overlap}
    if row == "O1":
        out["stamp"] = "K2 verdict is an absolute read of one trained head; 'the budget is the lever' still needs the replicate (PREREG_S1C section 5)"
    return out


# ================================================================================================================ PREREG_AUXVETO section 6 (V-S1C in V-PRED's role)
def decide_veto(vpred: dict, vgt: dict) -> str:
    """Identical to auxveto_read.decide (cross-checked by test_s1c_autoread over a grid).  vpred / vgt: {point, lo, hi} of the stratified Delta PDMS x100, unrounded."""
    lo, hi, pt = vpred["lo"], vpred["hi"], vpred["point"]
    if lo > 0 and pt >= AV_GO_POINT:
        return "AV-GO"
    if lo > 0 and pt < AV_GO_POINT:
        return "AV-PROMISING"
    if hi < 0:
        return "AV-HARM"
    if vgt["lo"] > 0:
        return "AV-CEILING-ONLY"
    return "AV-NULL"


def veto_row(vs1c: dict, vgt: dict, prereg_text: str) -> dict:
    row = decide_veto(vs1c, vgt)
    cells = auxveto_row_cells(prereg_text, row)
    conds = {
        "AV-GO: V-PRED lower bound > 0 AND point >= +0.30": bool(vs1c["lo"] > 0 and vs1c["point"] >= AV_GO_POINT),
        "AV-PROMISING: V-PRED lower bound > 0 AND point < +0.30": bool(vs1c["lo"] > 0 and vs1c["point"] < AV_GO_POINT),
        "AV-HARM: V-PRED upper bound < 0": bool(vs1c["hi"] < 0),
        "AV-CEILING-ONLY: V-PRED not separated AND V-GT lower bound > 0": bool(vs1c["lo"] <= 0 <= vs1c["hi"] and vgt["lo"] > 0),
        "AV-NULL: V-PRED not separated AND V-GT not separated above 0": bool(vs1c["lo"] <= 0 <= vs1c["hi"] and not vgt["lo"] > 0),
    }
    gt_branch = None
    if row == "AV-HARM":                                  # the row's own next arm reads V-GT: 'if the GT ceiling is positive, the lever is raster quality (S1c), else AV-NULL's next arm'
        gt_pos = bool(vgt["lo"] > 0)
        gt_branch = {"V_GT_lower_bound_gt_0": gt_pos, "reading": "GT ceiling positive (V-GT lower bound > 0): the lever is raster quality (S1c)" if gt_pos
                     else "GT ceiling not positive: AV-NULL's next arm", "AV-NULL_next_arm_quoted": None if gt_pos else auxveto_row_cells(prereg_text, "AV-NULL")["next"]}
    return {"row": row, "conditions": conds, "gt_branch": gt_branch, "quoted_row": cells, "V-S1C": vs1c, "V-GT": vgt,
            "stamp": "separated, single head seed, not yet a lever claim" if row in ("AV-GO", "AV-PROMISING", "AV-HARM") else None}
