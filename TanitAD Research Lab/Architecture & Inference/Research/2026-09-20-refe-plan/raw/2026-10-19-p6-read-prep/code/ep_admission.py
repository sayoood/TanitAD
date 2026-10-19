"""EP-ADMISSION: the candidate-agnostic EP-reliability admission gate (DRAFT_PREREG_EP_ADMISSION.md, Master Mind 2026-10-19) as code.  CPU only; numpy + the landed EP-RELIAB-PICKS library.

STATUS OF THE CONSTANTS: the draft is NOT sealed.  Every bar below is the draft's, copied literally and marked DRAFT; the Master Mind seals them (or amends them) against a NAMED candidate BEFORE that
candidate's selection-tier bank is read.  A run without `--prereg` is `authority: NONE` (a control or smoke run) and its verdict must not be quoted as a gate result.

WHAT IT DOES (draft sections A and B), for a candidate C against P2, both on the SAME S0 sets (15,808 sets, 117 logs):
  A1  EP reliability at C's picks:  D_M = M_p - M_e (predicted minus realised EP MASS, the rank-free mass form of EP-RELIAB-PICKS; C's own EP row at P2's picks and at C's picks).
      ADMIT-A1 iff the upper CI of D_M <= A1_BAR (+0.02, EP mass units).
  A2  NC / DAC / TTC reliability at C's picks: observed / predicted UNSAFE (deficit-form) mass, C's own rows.  ADMIT-A2 iff EVERY CI overlaps [A2_LO, A2_HI] = [0.8, 1.25].
  A3  realised dPDMS x100 vs P2 on the bank (+ the NC / DAC unsafe-pick-rate differences), REPORTED, not gated.
  B   the pessimistic-EP pick rule  EP_used = EP_slot + lam * (EP_row - EP_slot), out of fold: RECOVERS iff dPDMS lower CI >= 0 AND the NC / DAC unsafe-rate differences have upper CI <= +0.001 (all banks).
  Controls (the EP-RELIAB set, gating): C-KEYS / C-LABELS (the bank is on P2's sets and labels), C-ID (P2 as the candidate reads exactly 0), C-REACH (planted shifts move Delta exactly), C-POWER
  (a planted +0.10 over-prediction must FAIL A1 and -0.10 must ADMIT, on the label-independent neutral bank of Amendment 1), C-NULL (picks shuffled within token, 5 replicates, VOID iff >= 2 fire) and E-NULL
  (200 further replicates, descriptive: the empirical false-FAIL rate of A1), and the branch-table / RED-mutation check of every decision function in this file.

NOTHING of the estimator is re-implemented: the paired log-cluster bootstrap (10,000, seed 20260927), the gain / mass-form / scale-gate / attribution / reliability / reach / neutral-label / shuffle code is
IMPORTED from the landed `epreliab_lib` (sha256-pinned below) and, through it, the pinned `pickrule_lib` / `fastneg_lib`.  This file adds only: the bank loader and key alignment, the draft's decision
functions, the pessimistic rule, the admission-flavoured controls and the CLI.

Estimator discipline (CLAUDE.md): the interval answers "would another draw of LOGS say this?" and nothing about another training run of the candidate (`H-ESTIM-SEED-1`) or another inference run.
Tier: SELECTION, SEEN scenes (`D-L1-G0-1`), label-lookup PDMS; not T1.  A PASS is a NECESSARY condition for a navtest look, never evidence of a navtest gain.

    python ep_admission.py run --p2-bank <S0 bank dir> --cand-bank <candidate bank dir> --ranked-logs <ranked_fit_logs.tsv> --prereg <sealed prereg> --candidate-ckpt-sha256 <sha> --out admission.json
    python ep_admission.py selftest-real --p2-bank <S0 bank dir> --ranked-logs <tsv> --arrays-npz <pickrule arrays.npz> --landed-result <ep-reliab result.json> --out selftest_real.json
    python ep_admission.py branch-table          # the decision functions' branch table + the RED-mutation report
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import importlib.util
import json
import math
import os
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = "ep-admission-v1"

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# constants.  DRAFT = copied literally from DRAFT_PREREG_EP_ADMISSION.md; the Master Mind seals or amends them against a named candidate before its bank is read.
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
EPRELIAB_LIB_SHA256 = "72f493fd150ef5165b5476d270138088046c6172216a207497174969e9f74878"      # the landed code/epreliab_lib.py (EP-RELIAB-PICKS), imported by path after this check
A1_BAR = 0.02              # DRAFT (A1): ADMIT-A1 iff the upper CI of D_M <= +0.02 (EP mass units)
A2_LO, A2_HI = 0.8, 1.25   # DRAFT (A2): ADMIT-A2 iff every CI overlaps [0.8, 1.25]
A2_HEADS = ("NC", "DAC", "TTC")
B_UNSAFE_UB = 0.001        # DRAFT (B): the NC / DAC unsafe-rate difference upper CI <= +0.001
PLANT = 0.10               # C-POWER planted shift (EP-RELIAB Amendment 1)
NULL_C_REPS, NULL_E_REPS, NULL_VOID_FIRES = 5, 200, 2    # EP-RELIAB prereg section 7
N_LOGS, BOOT_B, BOOT_SEED = 117, 10000, 20260927
SELECT_PCT = 10            # S0 is the 10 % log subsample (ranked_fit_logs.tsv first_fraction <= 0.10)
N_FOLDS = 5                # fold = rank mod 5 (epr1 A2.2)
RANKED_LOGS_SHA256 = "a61275037a3d2b06903dc7e1ad0f6f99fb740f668a6ec7488fdef6fe901ce9b1"      # the registered S0 ranked_fit_logs.tsv (epr1.RANKED_LOGS_SHA256)
LABEL_ORDER = ["nc", "dac", "ep", "ttc", "c", "ddc"]
RULE = "navsim_v1"
NC, DAC, EP, TTC, CMF, DDC = 0, 1, 2, 3, 4, 5
HEAD_COL = {"NC": NC, "DAC": DAC, "EP": EP, "TTC": TTC}
MIN_AVAIL_GB = 3.0


class AdmissionError(RuntimeError):
    """An integrity check, a pin or a precondition failed (the run refuses; no verdict is written)."""


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the landed library, imported by path after its sha256 is checked
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _epreliab_candidates(explicit: str | None) -> list:
    envp = os.environ.get("EPRELIAB_CODE_DIR")
    out = [explicit, os.path.join(envp, "epreliab_lib.py") if envp else None,
           os.path.join(HERE, "..", "..", "2026-10-19-ep-reliab-picks", "code", "epreliab_lib.py"),     # repo layout: raw/2026-10-19-p6-read-prep/code -> raw/2026-10-19-ep-reliab-picks/code
           "C:/Users/Admin/par18/epreliab/code/epreliab_lib.py"]                                           # dev-box staging copy of the same package
    return [c for c in out if c]


def load_epreliab(explicit: str | None = None):
    """Import the landed epreliab_lib by path; refuse a copy whose sha256 differs from the pin (a tampered or edited copy would change the estimator under every verdict)."""
    if "epreliab_lib" in sys.modules and getattr(sys.modules["epreliab_lib"], "_ADMISSION_PINNED", None) == EPRELIAB_LIB_SHA256:
        return sys.modules["epreliab_lib"]
    tried = []
    for c in _epreliab_candidates(explicit):
        if os.path.exists(c):
            sha = sha256_file(c)
            if sha != EPRELIAB_LIB_SHA256:
                raise AdmissionError(f"epreliab_lib at {c} has sha256 {sha} != the pinned {EPRELIAB_LIB_SHA256}")
            d = os.path.dirname(os.path.abspath(c))
            if d not in sys.path:
                sys.path.insert(0, d)
            spec = importlib.util.spec_from_file_location("epreliab_lib", c)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["epreliab_lib"] = mod
            spec.loader.exec_module(mod)
            mod._ADMISSION_PINNED = EPRELIAB_LIB_SHA256
            return mod
        tried.append(c)
    raise AdmissionError(f"epreliab_lib.py not found; probed: {tried}")


_LIBS: dict = {}


def libs():
    """(L = epreliab_lib, P = pickrule_lib, F = fastneg_lib), loaded once, hash-pinned by the landed library's own checks."""
    if not _LIBS:
        L = load_epreliab()
        _LIBS.update(L=L, P=L.load_pickrule(), F=L.load_fastneg())
    return _LIBS["L"], _LIBS["P"], _LIBS["F"]


def jsonable(o):
    if isinstance(o, dict):
        return {(str(k) if not isinstance(k, str) else k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the draft's decision functions (pure; `branch_table_failures` is the table every mutation in DECISION_MUTATIONS must break)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def a1_verdict(ci: dict, bar: float = A1_BAR) -> str:
    """DRAFT (A1): ADMIT-A1 iff the UPPER CI of D_M <= the bar (closed).  A NaN bound compares False, so it FAILS (fail closed)."""
    if ci["hi"] <= bar:
        return "ADMIT-A1"
    return "FAIL-A1"


def a2_head_ok(ci: dict, lo: float = A2_LO, hi: float = A2_HI) -> bool:
    """DRAFT (A2), one head: the CI of observed / predicted unsafe mass OVERLAPS [lo, hi] (closed).  A NaN bound is not ok."""
    return bool(ci["lo"] <= hi and ci["hi"] >= lo)


def a2_verdict(cis: dict, heads: tuple = A2_HEADS) -> str:
    """DRAFT (A2): ADMIT-A2 iff EVERY head (NC, DAC, TTC) is ok; a missing head is not ok."""
    if len(heads) > 0 and all((h in cis) and a2_head_ok(cis[h]) for h in heads):
        return "ADMIT-A2"
    return "FAIL-A2"


def a1_form(on_scale: bool, mode: str) -> str:
    """Which statistic carries A1: 'mass' (D_M, the draft's literal text) always; 'auto' = the EP-RELIAB rule (ON-SCALE -> Delta, OFF-SCALE -> the rank-free mass form D_M)."""
    if mode == "auto" and on_scale:
        return "DELTA"
    return "MASS"


def b_recovers(dpdms_lo: list, nc_hi: list, dac_hi: list, ub: float = B_UNSAFE_UB) -> bool:
    """DRAFT (B): RECOVERS iff, for EVERY candidate bank, the dPDMS lower CI >= 0 AND the NC and DAC unsafe-rate-difference upper CIs <= +0.001."""
    n = len(dpdms_lo)
    if n == 0 or len(nc_hi) != n or len(dac_hi) != n:
        return False
    return bool(all(lo >= 0.0 for lo in dpdms_lo) and all(h <= ub for h in nc_hi) and all(h <= ub for h in dac_hi))


def null_fires_a1(verdict: str) -> bool:
    """C-NULL / E-NULL: a replicate FIRES iff A1 would FAIL on the shuffled-pick bank (a false refusal of a candidate whose picks carry no information)."""
    return bool(verdict != "ADMIT-A1")


def null_void(fires: list) -> bool:
    """EP-RELIAB section 7: VOID iff the null fires on 2 or more of its 5 replicates."""
    return bool(sum(1 for f in fires if f) >= NULL_VOID_FIRES)


def admission_label(a1: list, a2: list, controls_ok: bool, null_is_void: bool) -> str:
    """The final label.  A failing gating control or a firing C-NULL VOIDs the read (no verdict); otherwise every candidate bank must pass BOTH A1 and A2."""
    if not controls_ok:
        return "VOID (a gating control failed)"
    if null_is_void:
        return "VOID (C-NULL: A1 fired on >= 2 of 5 shuffled-pick replicates)"
    a1_ok = len(a1) > 0 and all(v == "ADMIT-A1" for v in a1)
    a2_ok = len(a2) > 0 and all(v == "ADMIT-A2" for v in a2)
    if a1_ok and a2_ok:
        return "ADMIT (A1 and A2 pass: the shared navtest look is permitted)"
    if (not a1_ok) and (not a2_ok):
        return "REFUSE-A1+A2"
    return "REFUSE-A1" if not a1_ok else "REFUSE-A2"


NEXT_ARM = {
    "ADMIT": "the candidate's shared navtest look (MM_RULINGS_HEADROWS.md R1-R9 rules). A1 / A2 are necessary conditions, not a gain.",
    "REFUSE-A1": "NO navtest look: the candidate's EP gain is not real on the selection tier. Attribute the cause with the (D) decomposition of EP-RELIAB-PICKS and return the lever to the trunk.",
    "REFUSE-A2": "NO navtest look: the safety rows are mis-calibrated at the candidate's picks. Register a calibration arm for THAT candidate, with reliability at the picks pre-registered.",
    "REFUSE-A1+A2": "NO navtest look: both fail. Attribute the EP gain (D), register the safety-row calibration arm for THAT candidate, return the lever to the trunk.",
    "VOID": "no verdict from this read; the failing control is reported and the instrument repaired first.",
}


def next_arm(label: str) -> str:
    if label.startswith("VOID"):
        return NEXT_ARM["VOID"]
    if label.startswith("ADMIT"):
        return NEXT_ARM["ADMIT"]
    return NEXT_ARM[label]


# (old, new) text mutations of THIS file's decision functions; EACH must make `branch_table_failures` non-empty (the RED arm of every branch)
DECISION_MUTATIONS = [
    ('if ci["hi"] <= bar:\n        return "ADMIT-A1"', 'if ci["hi"] < bar:\n        return "ADMIT-A1"'),
    ('if ci["hi"] <= bar:\n        return "ADMIT-A1"', 'if ci["hi"] >= bar:\n        return "ADMIT-A1"'),
    ('if ci["hi"] <= bar:\n        return "ADMIT-A1"', 'if ci["lo"] <= bar:\n        return "ADMIT-A1"'),
    ('if ci["hi"] <= bar:\n        return "ADMIT-A1"', 'if ci["point"] <= bar:\n        return "ADMIT-A1"'),
    ('if ci["hi"] <= bar:\n        return "ADMIT-A1"', 'if ci["hi"] <= 0.0:\n        return "ADMIT-A1"'),
    ('if ci["hi"] <= bar:\n        return "ADMIT-A1"', 'if not (ci["hi"] > bar):\n        return "ADMIT-A1"'),
    ('return bool(ci["lo"] <= hi and ci["hi"] >= lo)', 'return bool(ci["lo"] < hi and ci["hi"] >= lo)'),
    ('return bool(ci["lo"] <= hi and ci["hi"] >= lo)', 'return bool(ci["lo"] <= hi and ci["hi"] > lo)'),
    ('return bool(ci["lo"] <= hi and ci["hi"] >= lo)', 'return bool(ci["lo"] <= hi or ci["hi"] >= lo)'),
    ('return bool(ci["lo"] <= hi and ci["hi"] >= lo)', 'return bool(ci["lo"] >= lo and ci["hi"] <= hi)'),
    ('return bool(ci["lo"] <= hi and ci["hi"] >= lo)', 'return bool(ci["point"] <= hi and ci["point"] >= lo)'),
    ('if len(heads) > 0 and all((h in cis) and a2_head_ok(cis[h]) for h in heads):', 'if len(heads) > 0 and any((h in cis) and a2_head_ok(cis[h]) for h in heads):'),
    ('if len(heads) > 0 and all((h in cis) and a2_head_ok(cis[h]) for h in heads):', 'if len(heads) > 0 and all(a2_head_ok(cis[h]) for h in heads if h in cis):'),
    ('A2_HEADS = ("NC", "DAC", "TTC")', 'A2_HEADS = ("NC", "DAC")'),
    ('if mode == "auto" and on_scale:', 'if mode == "auto" and not on_scale:'),
    ('if mode == "auto" and on_scale:', 'if mode != "mass" and on_scale:'),
    ('return bool(all(lo >= 0.0 for lo in dpdms_lo) and all(h <= ub for h in nc_hi) and all(h <= ub for h in dac_hi))', 'return bool(all(lo > 0.0 for lo in dpdms_lo) and all(h <= ub for h in nc_hi) and all(h <= ub for h in dac_hi))'),
    ('return bool(all(lo >= 0.0 for lo in dpdms_lo) and all(h <= ub for h in nc_hi) and all(h <= ub for h in dac_hi))', 'return bool(any(lo >= 0.0 for lo in dpdms_lo) and all(h <= ub for h in nc_hi) and all(h <= ub for h in dac_hi))'),
    ('return bool(all(lo >= 0.0 for lo in dpdms_lo) and all(h <= ub for h in nc_hi) and all(h <= ub for h in dac_hi))', 'return bool(all(lo >= 0.0 for lo in dpdms_lo) and all(h < ub for h in nc_hi) and all(h <= ub for h in dac_hi))'),
    ('return bool(all(lo >= 0.0 for lo in dpdms_lo) and all(h <= ub for h in nc_hi) and all(h <= ub for h in dac_hi))', 'return bool(all(lo >= 0.0 for lo in dpdms_lo) and all(h <= ub for h in nc_hi))'),
    ('return bool(all(lo >= 0.0 for lo in dpdms_lo) and all(h <= ub for h in nc_hi) and all(h <= ub for h in dac_hi))', 'return bool(all(h <= ub for h in nc_hi) and all(h <= ub for h in dac_hi))'),
    ('return bool(verdict != "ADMIT-A1")', 'return bool(verdict == "ADMIT-A1")'),
    ('return bool(sum(1 for f in fires if f) >= NULL_VOID_FIRES)', 'return bool(sum(1 for f in fires if f) > NULL_VOID_FIRES)'),
    ('NULL_C_REPS, NULL_E_REPS, NULL_VOID_FIRES = 5, 200, 2', 'NULL_C_REPS, NULL_E_REPS, NULL_VOID_FIRES = 5, 200, 1'),
    ('if not controls_ok:\n        return "VOID (a gating control failed)"', 'if controls_ok:\n        return "VOID (a gating control failed)"'),
    ('if null_is_void:\n        return "VOID (C-NULL: A1 fired on >= 2 of 5 shuffled-pick replicates)"', 'if not null_is_void:\n        return "VOID (C-NULL: A1 fired on >= 2 of 5 shuffled-pick replicates)"'),
    ('a1_ok = len(a1) > 0 and all(v == "ADMIT-A1" for v in a1)', 'a1_ok = len(a1) > 0 and any(v == "ADMIT-A1" for v in a1)'),
    ('a2_ok = len(a2) > 0 and all(v == "ADMIT-A2" for v in a2)', 'a2_ok = len(a2) > 0 and any(v == "ADMIT-A2" for v in a2)'),
    ('if a1_ok and a2_ok:', 'if a1_ok or a2_ok:'),
    ('return "REFUSE-A1" if not a1_ok else "REFUSE-A2"', 'return "REFUSE-A2" if not a1_ok else "REFUSE-A1"'),
    ('"REFUSE-A1": "NO navtest look: the candidate\'s EP gain is not real', '"REFUSE-A1": "the candidate\'s shared navtest look: the candidate\'s EP gain is not real'),
    ('A1_BAR = 0.02 ', 'A1_BAR = 0.05 '),
    ('A2_LO, A2_HI = 0.8, 1.25 ', 'A2_LO, A2_HI = 0.7, 1.25 '),
    ('A2_LO, A2_HI = 0.8, 1.25 ', 'A2_LO, A2_HI = 0.8, 1.35 '),
    ('B_UNSAFE_UB = 0.001 ', 'B_UNSAFE_UB = 0.002 '),
]


def branch_table_failures(mod) -> list:
    """Every decision function on a synthetic branch table (boundary cases on every closed / open edge).  `mod` is this module or a MUTATED copy; returns the failing cases (empty = every branch right)."""
    bad = []

    def chk(name, got, want):
        if got != want:
            bad.append(f"{name}: got {got!r}, want {want!r}")
    ci = lambda lo, hi, pt=None: {"lo": lo, "hi": hi, "point": ((lo + hi) / 2 if pt is None else pt)}       # noqa: E731
    nan = float("nan")
    # --- A1
    chk("a1 hi = 0", mod.a1_verdict(ci(0.0, 0.0)), "ADMIT-A1")
    chk("a1 hi well below the bar", mod.a1_verdict(ci(-0.3, -0.1)), "ADMIT-A1")
    chk("a1 hi exactly the bar is closed", mod.a1_verdict(ci(-0.01, 0.02)), "ADMIT-A1")
    chk("a1 hi just above the bar", mod.a1_verdict(ci(-0.01, 0.0201)), "FAIL-A1")
    chk("a1 the EP-RELIAB interval", mod.a1_verdict(ci(0.111, 0.164)), "FAIL-A1")
    chk("a1 hi decides, not the point (point under the bar, hi above)", mod.a1_verdict(ci(-0.2, 0.05, -0.1)), "FAIL-A1")
    chk("a1 hi decides, not lo (lo under the bar, hi above)", mod.a1_verdict(ci(0.0, 0.5)), "FAIL-A1")
    chk("a1 hi decides, not 'hi <= 0' (0 < hi <= bar)", mod.a1_verdict(ci(-0.1, 0.01)), "ADMIT-A1")
    chk("a1 NaN fails closed", mod.a1_verdict(ci(nan, nan, nan)), "FAIL-A1")
    chk("a1 NaN hi only", mod.a1_verdict({"lo": -0.1, "hi": nan, "point": 0.0}), "FAIL-A1")
    chk("a1 bar is a parameter (admit)", mod.a1_verdict(ci(0.0, 0.05), 0.05), "ADMIT-A1")
    chk("a1 bar is a parameter (fail)", mod.a1_verdict(ci(0.0, 0.05), 0.04), "FAIL-A1")
    chk("a1 default bar is the draft's +0.02", mod.A1_BAR, 0.02)
    # --- A2 per head
    chk("a2 head containing 1", mod.a2_head_ok(ci(0.664, 1.168)), True)
    chk("a2 head all above the band", mod.a2_head_ok(ci(1.3, 1.6)), False)
    chk("a2 head all below the band", mod.a2_head_ok(ci(0.5, 0.79)), False)
    chk("a2 head touching the lower edge (closed)", mod.a2_head_ok(ci(0.7, 0.8)), True)
    chk("a2 head touching the upper edge (closed)", mod.a2_head_ok(ci(1.25, 1.4)), True)
    chk("a2 head inside the band", mod.a2_head_ok(ci(0.9, 1.1)), True)
    chk("a2 head wider than the band (overlap, not containment)", mod.a2_head_ok(ci(0.1, 5.0)), True)
    chk("a2 head straddling the lower edge only", mod.a2_head_ok(ci(0.6, 0.85)), True)
    chk("a2 head straddling the upper edge only", mod.a2_head_ok(ci(1.2, 1.9)), True)
    chk("a2 head point inside but CI far above is not the point rule", mod.a2_head_ok(ci(1.3, 1.5, 1.0)), False)
    chk("a2 head NaN", mod.a2_head_ok(ci(nan, nan)), False)
    # --- A2 verdict over heads
    good = {"NC": ci(0.664, 1.168), "DAC": ci(0.837, 1.084), "TTC": ci(0.762, 1.089), "EP": ci(6.0, 9.0)}
    chk("a2 verdict all three ok (EP is not a head of A2)", mod.a2_verdict(good), "ADMIT-A2")
    chk("a2 NC over-confident", mod.a2_verdict({**good, "NC": ci(1.3, 1.6)}), "FAIL-A2")
    chk("a2 DAC over-confident", mod.a2_verdict({**good, "DAC": ci(1.3, 1.6)}), "FAIL-A2")
    chk("a2 TTC over-confident", mod.a2_verdict({**good, "TTC": ci(1.3, 1.6)}), "FAIL-A2")
    chk("a2 TTC over-conservative (two-sided as drafted)", mod.a2_verdict({**good, "TTC": ci(0.4, 0.7)}), "FAIL-A2")
    chk("a2 TTC missing", mod.a2_verdict({k: v for k, v in good.items() if k != "TTC"}), "FAIL-A2")
    chk("a2 NaN head", mod.a2_verdict({**good, "DAC": ci(nan, nan)}), "FAIL-A2")
    chk("a2 empty", mod.a2_verdict({}), "FAIL-A2")
    chk("a2 heads are exactly NC, DAC, TTC", tuple(mod.A2_HEADS), ("NC", "DAC", "TTC"))
    chk("a2 band is the draft's [0.8, 1.25]", (mod.A2_LO, mod.A2_HI), (0.8, 1.25))
    # --- A1 form
    chk("form mass is literal even when on-scale", mod.a1_form(True, "mass"), "MASS")
    chk("form mass off-scale", mod.a1_form(False, "mass"), "MASS")
    chk("form auto on-scale -> Delta", mod.a1_form(True, "auto"), "DELTA")
    chk("form auto off-scale -> mass", mod.a1_form(False, "auto"), "MASS")
    chk("form: an unknown mode is the literal mass form", mod.a1_form(True, "bogus"), "MASS")
    # --- B
    chk("b all ok", mod.b_recovers([0.1, 0.2], [0.0005, 0.001], [0.0, 0.0009]), True)
    chk("b dPDMS lo exactly 0 is ok (>=)", mod.b_recovers([0.0], [0.0], [0.0]), True)
    chk("b dPDMS lo slightly negative", mod.b_recovers([-0.01], [0.0], [0.0]), False)
    chk("b NC hi exactly 0.001 is ok (<=)", mod.b_recovers([0.1], [0.001], [0.0]), True)
    chk("b NC hi above", mod.b_recovers([0.1], [0.0011], [0.0]), False)
    chk("b DAC hi exactly 0.001 is ok (<=)", mod.b_recovers([0.1], [0.0], [0.001]), True)
    chk("b DAC hi above", mod.b_recovers([0.1], [0.0], [0.0011]), False)
    chk("b one bank fails dPDMS (all, not any)", mod.b_recovers([0.1, -0.1], [0.0, 0.0], [0.0, 0.0]), False)
    chk("b one bank fails NC (all, not any)", mod.b_recovers([0.1, 0.1], [0.0, 0.01], [0.0, 0.0]), False)
    chk("b one bank fails DAC (all, not any)", mod.b_recovers([0.1, 0.1], [0.0, 0.0], [0.0, 0.01]), False)
    chk("b NaN is not ok", mod.b_recovers([nan], [0.0], [0.0]), False)
    chk("b NaN NC is not ok", mod.b_recovers([0.1], [nan], [0.0]), False)
    chk("b empty", mod.b_recovers([], [], []), False)
    chk("b ragged", mod.b_recovers([0.1, 0.1], [0.0], [0.0, 0.0]), False)
    chk("b draft unsafe bound", mod.B_UNSAFE_UB, 0.001)
    # --- null
    chk("null fires when A1 fails", mod.null_fires_a1("FAIL-A1"), True)
    chk("null silent when A1 admits", mod.null_fires_a1("ADMIT-A1"), False)
    chk("null void 2 of 5", mod.null_void([True, True, False, False, False]), True)
    chk("null not void 1 of 5", mod.null_void([True, False, False, False, False]), False)
    chk("null void 5 of 5", mod.null_void([True] * 5), True)
    chk("null not void 0 of 5", mod.null_void([False] * 5), False)
    chk("null constants", (mod.NULL_C_REPS, mod.NULL_E_REPS, mod.NULL_VOID_FIRES), (5, 200, 2))
    # --- the final label
    A, F_ = "ADMIT-A1", "FAIL-A1"
    B, G_ = "ADMIT-A2", "FAIL-A2"
    chk("label admit", mod.admission_label([A], [B], True, False).split(" (")[0], "ADMIT")
    chk("label admit, two banks", mod.admission_label([A, A], [B, B], True, False).split(" (")[0], "ADMIT")
    chk("label refuse a1", mod.admission_label([F_], [B], True, False), "REFUSE-A1")
    chk("label refuse a2", mod.admission_label([A], [G_], True, False), "REFUSE-A2")
    chk("label refuse both", mod.admission_label([F_], [G_], True, False), "REFUSE-A1+A2")
    chk("label one bank fails a1 (all, not any)", mod.admission_label([A, F_], [B, B], True, False), "REFUSE-A1")
    chk("label one bank fails a2 (all, not any)", mod.admission_label([A, A], [B, G_], True, False), "REFUSE-A2")
    chk("label a1 passes and a2 fails on different banks", mod.admission_label([A, F_], [G_, B], True, False), "REFUSE-A1+A2")
    chk("label control failed", mod.admission_label([A], [B], False, False).split(" (")[0], "VOID")
    chk("label control failed beats a refusal", mod.admission_label([F_], [G_], False, False), "VOID (a gating control failed)")
    chk("label null void", mod.admission_label([A], [B], True, True).split(" (")[0], "VOID")
    chk("label control failed beats null void", mod.admission_label([A], [B], False, True), "VOID (a gating control failed)")
    chk("label empty is not an admit", mod.admission_label([], [], True, False), "REFUSE-A1+A2")
    # --- the next arm
    chk("next admit", mod.next_arm("ADMIT (x)").startswith("the candidate's shared navtest look"), True)
    chk("next refuse a1 is no look", mod.next_arm("REFUSE-A1").startswith("NO navtest look"), True)
    chk("next refuse a1 is not the look text", mod.next_arm("REFUSE-A1").startswith("the candidate"), False)
    chk("next refuse a2 is the calibration arm", "calibration arm" in mod.next_arm("REFUSE-A2"), True)
    chk("next refuse a1 is not the calibration arm", "calibration arm" in mod.next_arm("REFUSE-A1"), False)
    chk("next void", mod.next_arm("VOID (x)").startswith("no verdict"), True)
    return bad


def mutated_copy(old: str, new: str):
    """A copy of THIS module's source with the FIRST occurrence of `old` replaced by `new` (the anchor must exist)."""
    path = os.path.join(HERE, "ep_admission.py")
    src = open(path, "r", encoding="utf-8").read()
    assert old in src, f"mutation anchor not found: {old!r}"
    ns = types.ModuleType("ep_admission_mut")
    ns.__file__ = path
    exec(compile(src.replace(old, new, 1), path, "exec"), ns.__dict__)
    return ns


def mutation_report() -> dict:
    """The unmutated branch table is empty and EVERY mutation of DECISION_MUTATIONS breaks it."""
    unmut = branch_table_failures(mutated_copy("SCHEMA = ", "SCHEMA = "))
    caught = []
    for old, new in DECISION_MUTATIONS:
        m = mutated_copy(old, new)
        caught.append({"mutation": f"{old[:70]!r} -> {new[:70]!r}", "caught": len(branch_table_failures(m)) > 0})
    return {"unmutated_failures": unmut, "n_mutations": len(caught), "n_caught": int(sum(c["caught"] for c in caught)), "uncaught": [c["mutation"] for c in caught if not c["caught"]]}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the bank: S0's unit format (`<unit>.sets.npz`), ranked_fit_logs.tsv, alignment by set key
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def read_ranked(path: str, pct: int = SELECT_PCT, expected_sha256: str | None = None) -> tuple:
    """-> (selected rows in key order [{rank, log, fold}], sha256).  Folds must equal rank mod 5 (epr1 A2.2); `expected_sha256` pins the registered S0 table."""
    raw = open(path, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and sha != expected_sha256:
        raise AdmissionError(f"ranked_fit_logs.tsv sha256 {sha} != the registered {expected_sha256}")
    lines = raw.decode("utf-8").splitlines()
    head = lines[0].split("\t")
    for need in ("rank", "log", "fold", "first_fraction"):
        if need not in head:
            raise AdmissionError(f"ranked_fit_logs.tsv has no column {need!r}: {head}")
    ix = {h: head.index(h) for h in head}
    rows = []
    for i, ln in enumerate(lines[1:]):
        c = ln.split("\t")
        rank, fold, ff = int(c[ix["rank"]]), int(c[ix["fold"]]), float(c[ix["first_fraction"]])
        if rank != i:
            raise AdmissionError(f"ranked_fit_logs.tsv: rank {rank} at row {i}")
        if fold != rank % N_FOLDS:
            raise AdmissionError(f"ranked_fit_logs.tsv: fold {fold} != rank {rank} mod {N_FOLDS}")
        if int(round(ff * 100)) <= pct:
            rows.append({"rank": rank, "log": c[ix["log"]], "fold": fold})
    if len({r["log"] for r in rows}) != len(rows):
        raise AdmissionError("ranked_fit_logs.tsv: duplicate log")
    return rows, sha


def read_manifest(bank_dir: str) -> dict | None:
    p = os.path.join(bank_dir, "MANIFEST.sha256.json")
    if not os.path.exists(p):
        return None
    doc = json.load(open(p, "r", encoding="utf-8"))
    table: dict = {}
    for rel, ent in doc["files"].items():
        base = os.path.basename(str(ent.get("path", rel)).replace("\\", "/"))
        if not base.endswith(".sets.npz"):
            continue
        if base in table and table[base] != ent["sha256"]:
            raise AdmissionError(f"manifest lists {base} twice with different sha256")
        table[base] = ent["sha256"]
    return {"table": table, "sha256": sha256_file(p)}


def load_bank_units(bank_dir: str, ranked_rows: list, *, require_manifest: bool = True, check_top8: bool = True) -> dict:
    """Assemble a bank from S0-format units: sorted unit names, rows in unit order, the selected logs only (the order `epr1.load_fit_data` / the pickrule `prepare` stage use).
    Every unit's sha256 is checked against `<bank_dir>/MANIFEST.sha256.json` BEFORE np.load (a bank without a manifest is refused unless `require_manifest=False`)."""
    sel_pos = {r["log"]: i for i, r in enumerate(ranked_rows)}
    units = sorted(n for n in os.listdir(bank_dir) if n.endswith(".sets.npz"))
    if not units:
        raise AdmissionError(f"no *.sets.npz in {bank_dir}")
    man = read_manifest(bank_dir)
    if man is None and require_manifest:
        raise AdmissionError(f"{bank_dir} has no MANIFEST.sha256.json (build one with `build_candidate_bank.py manifest`): refused, an unverified bank is not read")
    unit_sha: dict = {}
    parts: dict = {k: [] for k in ("log", "token", "step", "rank", "ckpt_step", "logits", "labels6", "top8")}
    meta = {"rule": set(), "ckpt": set()}
    for n in units:
        p = os.path.join(bank_dir, n)
        sha = sha256_file(p)
        if man is not None:
            if n not in man["table"]:
                raise AdmissionError(f"unit {n} is not in the manifest (refused: unverified)")
            if sha != man["table"][n]:
                raise AdmissionError(f"unit {n}: sha256 {sha} != manifest {man['table'][n]}")
        unit_sha[n] = sha
        z = np.load(p, allow_pickle=False)
        for key in ("log", "token", "step", "rank", "logits", "labels6", "label_order"):
            if key not in z.files:
                raise AdmissionError(f"unit {n}: missing array {key!r}")
        if [str(x) for x in z["label_order"]] != LABEL_ORDER:
            raise AdmissionError(f"unit {n}: label_order {[str(x) for x in z['label_order']]} != {LABEL_ORDER}")
        if "rule" in z.files:
            meta["rule"].add(str(z["rule"]))
        if "ckpt" in z.files:
            meta["ckpt"].add(str(z["ckpt"]))
        mask = np.array([str(x) in sel_pos for x in z["log"]], dtype=bool)
        if not mask.any():
            z.close()
            continue
        lo, lb = z["logits"][mask], z["labels6"][mask]
        if lo.shape[1:] != (64, 6) or lb.shape[1:] != (64, 6):
            raise AdmissionError(f"unit {n}: logits {lo.shape} / labels6 {lb.shape} are not [S, 64, 6]")
        if not (np.isfinite(lo).all() and np.isfinite(lb).all()):
            raise AdmissionError(f"unit {n}: non-finite logit or label")
        if lb.min() < 0.0 or lb.max() > 1.0:
            raise AdmissionError(f"unit {n}: a label outside [0, 1]")
        parts["log"].append(np.array([str(x) for x in z["log"][mask]]))
        parts["token"].append(np.array([str(x) for x in z["token"][mask]]))
        parts["step"].append(z["step"][mask].astype(np.int64))
        parts["rank"].append(z["rank"][mask].astype(np.int64))
        parts["ckpt_step"].append(z["ckpt_step"][mask].astype(np.int64) if "ckpt_step" in z.files else np.full(int(mask.sum()), -1, np.int64))
        parts["logits"].append(lo.astype(np.float32))
        parts["labels6"].append(lb.astype(np.float32))
        parts["top8"].append(z["top8"][mask].astype(np.int64) if "top8" in z.files else np.zeros((int(mask.sum()), 0), np.int64))
        z.close()
    if meta["rule"] and meta["rule"] != {RULE}:
        raise AdmissionError(f"bank rule {sorted(meta['rule'])} != {RULE!r}")
    if len(meta["ckpt"]) > 1:
        raise AdmissionError(f"bank units were scored by different checkpoints: {sorted(meta['ckpt'])}")
    cat = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    N = len(cat["log"])
    log_idx = np.array([sel_pos[x] for x in cat["log"]], np.int64)
    ranks = np.array([r["rank"] for r in ranked_rows], np.int64)
    if check_top8 and cat["top8"].shape[1] == 8:
        _check_top8(cat["logits"], cat["top8"])
    present = set(cat["log"].tolist())
    missing = [r["log"] for r in ranked_rows if r["log"] not in present]
    if missing:
        raise AdmissionError(f"{len(missing)} selected logs have no set in the bank, e.g. {missing[:3]}")
    keys = list(zip(cat["log"].tolist(), cat["token"].tolist(), cat["step"].tolist(), cat["rank"].tolist()))
    if len(set(keys)) != N:
        raise AdmissionError("duplicate set key (log, token, step, rank) in the bank")
    return {"z6": cat["logits"], "lab": cat["labels6"], "top8": cat["top8"], "keys": keys, "ckpt_step": cat["ckpt_step"], "log_idx": log_idx,
            "fold": (ranks[log_idx] % N_FOLDS).astype(np.int64), "log_names": [r["log"] for r in ranked_rows], "n_sets": N, "n_logs": len(ranked_rows),
            "ckpt": sorted(meta["ckpt"])[0] if meta["ckpt"] else None, "unit_sha256": unit_sha, "manifest_sha256": man["sha256"] if man else None, "bank_dir": bank_dir}


def _check_top8(z6: np.ndarray, top8: np.ndarray) -> None:
    """The stored top-8 must be the frozen rule applied to the stored logits (stable descending argsort of the float64 aggregate); mismatches are re-checked per set with the dump's own expression."""
    for s in range(0, len(z6), 4096):
        p = 1.0 / (1.0 + np.exp(-z6[s:s + 4096].astype(np.float64)))
        agg = p[..., 0] * p[..., 1] * (5 * p[..., 2] + 5 * p[..., 3] + 2 * p[..., 4]) / 12.0
        again = np.argsort(-agg, axis=1, kind="stable")[:, :8]
        for i in np.nonzero((again != top8[s:s + 4096]).any(axis=1))[0]:
            q = 1.0 / (1.0 + np.exp(-z6[s + i].astype(np.float64)))
            a1 = q[:, 0] * q[:, 1] * (5 * q[:, 2] + 5 * q[:, 3] + 2 * q[:, 4]) / 12.0
            if not np.array_equal(np.argsort(-a1, kind="stable")[:8], top8[s + i]):
                raise AdmissionError(f"stored top8 of set {s + int(i)} is not the frozen rule applied to its stored logits")


def align_candidate(p2: dict, cand: dict) -> dict:
    """Reorder the candidate bank to P2's set order BY KEY and prove it is on the same sets with the same labels (labels6 are the PDM labels of the stored trajectory: candidate-independent).
    -> the candidate dict with `z6` in P2's order plus the checks.  Refuses on any mismatch."""
    if cand["n_sets"] != p2["n_sets"]:
        raise AdmissionError(f"C-KEYS: the candidate bank has {cand['n_sets']} sets, P2's has {p2['n_sets']}")
    pos = {k: i for i, k in enumerate(p2["keys"])}
    try:
        order = np.array([pos[k] for k in cand["keys"]], np.int64)
    except KeyError as e:
        raise AdmissionError(f"C-KEYS: the candidate bank has a set key that P2's bank does not: {e}")
    if len(np.unique(order)) != p2["n_sets"]:
        raise AdmissionError("C-KEYS: the candidate's keys are not a bijection onto P2's")
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    z6 = cand["z6"][inv]                        # row i of the result = the candidate's row for P2's set i
    lab = cand["lab"][inv]
    if not np.array_equal(lab, p2["lab"]):
        raise AdmissionError("C-LABELS: the candidate bank's labels6 differ from P2's on the same keys (labels are candidate-independent: a different label source means a different bank)")
    ck_ok = bool(np.array_equal(cand["ckpt_step"][inv], p2["ckpt_step"]))
    if not ck_ok:
        raise AdmissionError("C-KEYS: the stored sets' ckpt_step differ between the two banks")
    return {**cand, "z6": z6, "lab": lab, "keys": [cand["keys"][j] for j in inv.tolist()], "ckpt_step": cand["ckpt_step"][inv],
            "checks": {"keys_bijection": True, "labels_bit_equal": True, "ckpt_step_equal": ck_ok, "n_sets": p2["n_sets"]}}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the analysis of ONE candidate bank against P2's (A1, A2, A3, scale; no decision is taken here)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def make_stats(log_idx: np.ndarray, n_logs: int, B: int = BOOT_B, seed: int = BOOT_SEED):
    L, P, F = libs()
    return P.Stats(np.asarray(log_idx, np.int64), F.boot_draws(n_logs, B, seed), F, n_logs=n_logs)


def analyze_candidate(z6_p2: np.ndarray, z6_c: np.ndarray, lab: np.ndarray, S) -> dict:
    """prereg-draft quantities for one candidate: the EP gain predicted vs realised at the picks and its mass form (A1), the safety-row reliability at the picks (A2), dPDMS and unsafe-rate differences (A3),
    the scale gate and the (D) attribution.  Identical in structure to `epreliab_lib.analyze_seed`, with the candidate's FULL row set (not an EP-column swap)."""
    L, P, F = libs()
    pk_p, pk_c = P.rule_pick(z6_p2), P.rule_pick(z6_c)
    sig = P.sigmoid64(z6_c[..., EP])
    y = lab[..., EP].astype(np.float64)
    t = L.pair_terms(sig, y, pk_p, pk_c)
    G = L.full_gain_stats(S, t)
    ch = t["changed"].astype(np.float64)
    nch = int(ch.sum())
    if nch > 0:
        G["D_M_changed_only"] = S.ratio_diff(t["y_p"] * ch, t["s_p"] * ch, t["y_e"] * ch, t["s_e"] * ch)
    pdms = P.pdms_from_labels(lab)
    a3 = P.sweep_one(z6_c, lab, pdms, pk_p, S, None)
    rel = L.reliability_stats(S, lab, z6_c, z6_p2, pk_p, pk_c)
    out = {"n_sets": int(len(pk_p)), "n_changed": nch, "frac_changed": float(nch / len(pk_p)), "G": G, "scale": {"slope": L.scale_slope(S, sig, y, L.safe_mask(lab))},
           "A2_reliability": {h: rel[h]["composite_picks"]["deficit_form_sum_1my_over_sum_1msigma"] for h in A2_HEADS},
           "reliability_full": rel,
           "A3": {"d_pdms_x100": a3["d_pdms_x100"], "d_nc_unsafe_rate": a3["d_nc_unsafe_rate"], "d_dac_unsafe_rate": a3["d_dac_unsafe_rate"], "pdms_x100": a3["pdms_x100"],
                  "n_changed_vs_p2": a3["n_changed_vs_p2"]},
           "attribution": L.attribution_stats(S, lab, pk_p, pk_c)}
    sum_se = float(t["s_e"].sum())
    G_half = float((G["D_M"]["hi"] - G["D_M"]["lo"]) / 2.0)
    out["A1_resolution"] = {"analytic_oc_at_this_banks_half_width": a1_analytic_oc(G_half), "D_M_half_width": G_half, "bar": A1_BAR, "half_width_exceeds_bar": bool(G_half > A1_BAR), "n_changed": nch,
                            "per_changed_pick_overprediction_that_puts_the_D_M_point_on_the_bar": (float(A1_BAR * sum_se / (nch * (1.0 - A1_BAR))) if nch > 0 else None),
                            "sum_sigma_at_candidate_picks": sum_se}
    return out


def a1_statistic(an: dict, form: str) -> dict:
    """The CI that carries A1 under `form` ('MASS' = D_M, 'DELTA' = the gain difference)."""
    return an["G"]["D_M"] if form == "MASS" else an["G"]["delta"]


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# (B) the pessimistic-EP pick rule, out of fold
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def pessimistic_picks(z6_p2: np.ndarray, z6_c: np.ndarray, lab: np.ndarray, fold: np.ndarray, S, *, slot_source: str = "label", lam_override: float | None = None) -> dict:
    """EP_used = EP_slot + lam * (EP_row - EP_slot), on the SIGMOID (probability) scale, fitted on the TRAINING folds only and applied to the held-out fold (k = 0..4, fold = rank mod 5 of the log).
      EP_slot[j]  per proposal index j: the mean EP LABEL of slot j over the training sets (`slot_source='label'`, the feature-free slot prior of HEADROW-PROBE) or the mean EP ROW (`'row'`);
      lam_k       the within-set calibration slope of the EP label on the EP row over the SAFE proposals of the training sets (`epreliab_lib.scale_slope`, restricted to the training sets).
    The pick is the landed rule on the candidate's rows with the EP column replaced by logit(EP_used) (clipped to [1e-9, 1 - 1e-9]).  `lam_override` (a plant) bypasses the fit.  [impl] choices: the probability
    scale (lam is a slope of probabilities), the slot-prior source, and no clipping of lam; the Master Mind seals them."""
    L, P, F = libs()
    N = len(z6_c)
    sig = P.sigmoid64(z6_c[..., EP])
    y = lab[..., EP].astype(np.float64)
    safe = L.safe_mask(lab)
    used = np.empty_like(sig)
    lam_by_fold, slot_by_fold = {}, {}
    n_clip = 0
    for k in range(N_FOLDS):
        te = fold == k
        tr = ~te
        if not te.any():
            continue
        if not tr.any():
            raise AdmissionError("pessimistic rule: a fold has no training sets")
        slot = (y[tr].mean(axis=0) if slot_source == "label" else sig[tr].mean(axis=0))
        lam = float(L.scale_slope(S, sig, y, safe & tr[:, None])["point"]) if lam_override is None else float(lam_override)
        used[te] = slot[None, :] + lam * (sig[te] - slot[None, :])
        lam_by_fold[k], slot_by_fold[k] = lam, slot
    lo_c, hi_c = 1e-9, 1.0 - 1e-9
    n_clip = int(((used < lo_c) | (used > hi_c)).sum())
    used = np.clip(used, lo_c, hi_c)
    zB = z6_c.astype(np.float64)
    zB[..., EP] = np.log(used / (1.0 - used))
    pk_p, pk_b = P.rule_pick(z6_p2), P.rule_pick(zB)
    pdms = P.pdms_from_labels(lab)
    sw = P.sweep_one(zB, lab, pdms, pk_p, S, None)
    return {"picks": pk_b, "z6_B": zB, "lam_by_fold": lam_by_fold, "lam_outside_unit_interval": bool(any((v < 0.0 or v > 1.0) for v in lam_by_fold.values())), "n_clipped_entries": n_clip,
            "slot_source": slot_source, "n_changed_vs_p2": int((pk_b != pk_p).sum()), "n_changed_vs_candidate": int((pk_b != P.rule_pick(z6_c)).sum()),
            "d_pdms_x100": sw["d_pdms_x100"], "d_nc_unsafe_rate": sw["d_nc_unsafe_rate"], "d_dac_unsafe_rate": sw["d_dac_unsafe_rate"]}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the controls
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def _all_zero(ci: dict) -> bool:
    return all(float(ci[k]) == 0.0 for k in ("point", "lo", "hi"))


def control_id(z6_p2: np.ndarray, lab: np.ndarray, S) -> dict:
    """C-ID: P2 as the candidate reads EXACTLY zero on every quantity (D_M, Delta, gains, dPDMS, unsafe-rate differences, attribution), n_changed = 0, and A1 ADMITs."""
    an = analyze_candidate(z6_p2, z6_p2, lab, S)
    G, A = an["G"], an["attribution"]
    cis = [G["delta"], G["pred_gain"], G["real_gain"], G["D_M"], an["A3"]["d_pdms_x100"], an["A3"]["d_nc_unsafe_rate"], an["A3"]["d_dac_unsafe_rate"], A["total_x100"]] + list(A["sequential_x100"].values()) \
        + list(A["shapley_x100"].values())
    ok = an["n_changed"] == 0 and all(_all_zero(c) for c in cis) and float(G["M_e"]["point"]) == float(G["M_p"]["point"]) and a1_verdict(G["D_M"]) == "ADMIT-A1"
    return {"pass": bool(ok), "n_changed": an["n_changed"], "n_intervals_checked": len(cis), "every_point_lo_hi_equals_0.0": bool(all(_all_zero(c) for c in cis)), "M_e_equals_M_p": bool(G["M_e"]["point"] == G["M_p"]["point"]),
            "a1_verdict": a1_verdict(G["D_M"]), "p2_own_reliability_at_its_own_picks": {h: an["A2_reliability"][h] for h in A2_HEADS}}


def control_reach(z6_p2: np.ndarray, z6_c: np.ndarray, lab: np.ndarray, S) -> dict:
    """C-REACH (EP-RELIAB): planted label / sigma shifts at the candidate's changed picks move Delta by exactly d * n_changed / N (all sets) and d (changed sets); NOT RUN when n_changed = 0."""
    L, P, F = libs()
    pk_p, pk_c = P.rule_pick(z6_p2), P.rule_pick(z6_c)
    if int((pk_p != pk_c).sum()) == 0:
        return {"pass": True, "status": "NOT RUN (n_changed = 0: the picks are identical to P2's)"}
    r = L.reach_check(S, P.sigmoid64(z6_c[..., EP]), lab[..., EP].astype(np.float64), pk_p, pk_c)
    return {"pass": bool(r["pass"]), "status": "RUN", "detail": r}


def control_power(z6_p2: np.ndarray, z6_c: np.ndarray, lab: np.ndarray, S, shift: float = PLANT, bar: float = A1_BAR) -> dict:
    """C-POWER for the admission bar, label-independent (EP-RELIAB Amendment 1): on the NEUTRAL bank (labels = the candidate's sigma at both picks, so D_M = 0 exactly), planting +shift in sigma at the
    candidate's changed picks must FAIL A1, -shift must ADMIT, 0 must ADMIT with an interval exactly [0, 0].  If the candidate changes too few picks for +shift to cross the bar, the read is VOID by resolution:
    ADMIT would then be the instrument's inability to see a 0.10 over-prediction, not evidence of reliability."""
    L, P, F = libs()
    pk_p, pk_c = P.rule_pick(z6_p2), P.rule_pick(z6_c)
    nch = int((pk_p != pk_c).sum())
    if nch == 0:
        return {"pass": True, "status": "NOT RUN (n_changed = 0: the picks are identical to P2's, there is nothing to gate)"}
    sig = P.sigmoid64(z6_c[..., EP])
    yn = L.neutral_labels(sig, lab[..., EP].astype(np.float64), pk_p, pk_c)
    cs = lambda s: L.core_stats(S, L.pair_terms(L.plant_sigma(sig, pk_p, pk_c, s), yn, pk_p, pk_c))["D_M"]     # noqa: E731
    up, dn, zero = cs(+shift), cs(-shift), cs(0.0)
    row = {"plus": {"ci": up, "a1": a1_verdict(up, bar)}, "minus": {"ci": dn, "a1": a1_verdict(dn, bar)}, "zero": {"ci": zero, "a1": a1_verdict(zero, bar)}, "n_changed": nch, "shift": shift}
    row["zero_is_exactly_0_0"] = bool(float(zero["point"]) == 0.0 and float(zero["lo"]) == 0.0 and float(zero["hi"]) == 0.0)
    row["pass"] = bool(row["plus"]["a1"] == "FAIL-A1" and row["minus"]["a1"] == "ADMIT-A1" and row["zero"]["a1"] == "ADMIT-A1" and row["zero_is_exactly_0_0"])
    row["status"] = "RUN" if row["pass"] else "FAILED (A1 cannot see a planted over-prediction of %.2f at this candidate's n_changed = %d: VOID by resolution)" % (shift, nch)
    return row


def run_null(z6_p2: np.ndarray, z6_c: np.ndarray, lab: np.ndarray, S, reps_c: int = NULL_C_REPS, reps_e: int = NULL_E_REPS) -> dict:
    """C-NULL / E-NULL: picks SHUFFLED WITHIN TOKEN (`epreliab_lib.shuffled_pair`, permutation seeds 20261019 + 7919 * rep); a replicate FIRES iff A1 would FAIL on it.  C-NULL = reps 0..4 (gating, VOID iff >= 2
    fire); E-NULL = the next `reps_e` (descriptive: the empirical false-FAIL rate of A1 on this candidate's pick geometry, the R51 / R52 number the draft asks for)."""
    L, P, F = libs()
    pk_p, pk_c = P.rule_pick(z6_p2), P.rule_pick(z6_c)
    sig, y = P.sigmoid64(z6_c[..., EP]), lab[..., EP].astype(np.float64)
    if int((pk_p != pk_c).sum()) == 0:
        return {"status": "NOT RUN (n_changed = 0)", "void": False, "c_null": {"fires": 0, "of": reps_c}}
    rows = []
    for rep in range(reps_c + reps_e):
        a, b = L.shuffled_pair(pk_p, pk_c, rep)
        cs = L.core_stats(S, L.pair_terms(sig, y, a, b))
        v = a1_verdict(cs["D_M"])
        rows.append({"rep": rep, "role": "C-NULL" if rep < reps_c else "E-NULL", "D_M": cs["D_M"], "a1": v, "fires": null_fires_a1(v)})
    c_rows, e_rows = rows[:reps_c], rows[reps_c:]
    fires_c = [r["fires"] for r in c_rows]
    out = {"status": "RUN", "void": null_void(fires_c), "c_null": {"replicates": c_rows, "fires": int(sum(fires_c)), "of": reps_c}}
    if e_rows:
        pts = np.array([r["D_M"]["point"] for r in e_rows])
        his = np.array([r["D_M"]["hi"] for r in e_rows])
        out["e_null"] = {"n": len(e_rows), "empirical_a1_false_fail_rate": float(np.mean([r["fires"] for r in e_rows])), "D_M_point_mean": float(pts.mean()), "D_M_point_sd": float(pts.std(ddof=1)),
                         "D_M_upper_ci_quantiles_min_5_50_95_max": [float(v) for v in np.quantile(his, [0.0, 0.05, 0.5, 0.95, 1.0])],
                         "p_at_least_2_of_5_fire_at_that_rate": float(1 - (1 - np.mean([r["fires"] for r in e_rows])) ** 5 - 5 * np.mean([r["fires"] for r in e_rows]) * (1 - np.mean([r["fires"] for r in e_rows])) ** 4)}
    return out


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def a1_analytic_oc(half_width: float, bar: float = A1_BAR, truths: tuple = (-0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.10)) -> dict:
    """What the A1 bar MEANS at a given D_M interval half-width h.  A1 admits iff point + h <= bar, so a bank whose interval has half-width h can ADMIT only a point <= bar - h.  With the point ~ N(truth, se^2),
    se = h / 1.96 (a normal approximation that treats the bootstrap half-width as fixed), P(FAIL-A1 | truth) = 1 - Phi((bar - h - truth) / se).  ESTIMATED.  Anchor h on a MEASURED value: EP-RELIAB-PICKS' EP-R1
    composite has h = 0.0263 / 0.0259 (seeds 0 / 1; `raw/results/result.json` D_M intervals) at 77.9 % changed picks, which puts the bar's break-even point at -0.006: a candidate whose TRUE D_M is 0 fails A1
    about two times in three.  (An i.i.d.-noise Monte Carlo was tried first and DROPPED: it gave h = 0.008, 3.4x too small, because the real label noise is clustered by log.)"""
    h = float(half_width)
    se = h / 1.959964
    rows = {}
    for d in truths:
        if se > 0:
            pf = 1.0 - _phi((bar - h - d) / se)
        else:
            pf = 1.0 if d > bar else 0.0
        rows[f"{d:+g}"] = {"true_D_M": d, "p_fail_a1": float(pf), "p_admit_a1": float(1.0 - pf)}
    return {"status": "ESTIMATED (normal approximation, bootstrap half-width treated as fixed)", "half_width": h, "bar": bar, "point_that_just_admits": float(bar - h), "by_truth": rows}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# sealing
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def check_prereg(path: str, cand_ckpt_sha256: str | None, cand_bank_manifest_sha256: list) -> dict:
    """The run is bound to a sealed EP-ADMISSION registration: the file's sha256 must equal its `.sha256` sidecar (written at sealing), the text must NAME the candidate checkpoint (full sha256) and every
    candidate bank manifest sha256.  This cannot prove the bank was unread at sealing; it proves the registration names THIS candidate and THIS bank and was hashed before the run."""
    if not os.path.exists(path):
        raise AdmissionError(f"the registration {path} does not exist")
    sha = sha256_file(path)
    side = os.path.splitext(path)[0] + ".sha256"
    if not os.path.exists(side):
        raise AdmissionError(f"no sealing sidecar {side} (the registration must be hashed when it is sealed)")
    want = open(side, "r", encoding="utf-8").read().split()[0].strip().lower()
    if want != sha:
        raise AdmissionError(f"{path} sha256 {sha} != the sealed {want}: the registration changed after sealing (write an amendment)")
    text = open(path, "r", encoding="utf-8", errors="replace").read().lower()
    if not cand_ckpt_sha256 or cand_ckpt_sha256.lower() not in text:
        raise AdmissionError("the registration does not name the candidate checkpoint sha256 (a registration must be sealed AGAINST a named candidate)")
    for m in cand_bank_manifest_sha256:
        if m and m.lower() not in text:
            raise AdmissionError(f"the registration does not name the candidate bank manifest sha256 {m[:16]}...")
    return {"prereg": os.path.abspath(path), "prereg_sha256": sha, "names_candidate_ckpt_sha256": cand_ckpt_sha256, "names_bank_manifest_sha256": cand_bank_manifest_sha256}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def run_admission(p2: dict, cands: dict, *, a1_form_mode: str = "mass", slot_source: str = "label", null_reps_c: int = NULL_C_REPS, null_reps_e: int = NULL_E_REPS, B: int = BOOT_B,
                  skip_controls_mutation: bool = False, log=print) -> dict:
    """The full read: integrity -> controls (C-ID, C-REACH, C-POWER, C-NULL, branch table) -> A1 / A2 / A3 / B per candidate bank -> the final label.  `cands` {id: aligned bank}."""
    L, P, F = libs()
    S = make_stats(p2["log_idx"], p2["n_logs"], B=B)
    lab = p2["lab"]
    controls: dict = {}
    t0 = datetime.datetime.now()
    log("controls: C-ID (P2 as the candidate)")
    controls["C-ID"] = control_id(p2["z6"], lab, S)
    if not skip_controls_mutation:
        mr = mutation_report()
        controls["C-DECISION"] = {"pass": bool(len(mr["unmutated_failures"]) == 0 and mr["n_caught"] == mr["n_mutations"]), **mr}
    per, a1_list, a2_list, b_lo, b_nc, b_dac, nulls = {}, [], [], [], [], [], {}
    scale_on_all = []
    for cid, c in cands.items():
        log(f"candidate {cid}: analysis")
        an = analyze_candidate(p2["z6"], c["z6"], lab, S)
        scale_on_all.append(bool(L.scale_on(an["scale"]["slope"])))
        per[cid] = an
        controls[f"C-REACH[{cid}]"] = control_reach(p2["z6"], c["z6"], lab, S)
        controls[f"C-POWER[{cid}]"] = control_power(p2["z6"], c["z6"], lab, S)
        log(f"candidate {cid}: C-NULL / E-NULL")
        nulls[cid] = run_null(p2["z6"], c["z6"], lab, S, null_reps_c, null_reps_e)
        controls[f"C-NULL[{cid}]"] = {"pass": bool(not nulls[cid]["void"]), "fires": nulls[cid]["c_null"]["fires"], "of": nulls[cid]["c_null"]["of"], "status": nulls[cid]["status"]}
    on_scale = bool(len(scale_on_all) > 0 and all(scale_on_all))
    form = a1_form(on_scale, a1_form_mode)
    for cid, an in per.items():
        ci1 = a1_statistic(an, form)
        v1 = a1_verdict(ci1)
        v2 = a2_verdict(an["A2_reliability"])
        an["A1"] = {"form": form, "statistic": ci1, "bar": A1_BAR, "verdict": v1, "other_form_statistic": a1_statistic(an, "DELTA" if form == "MASS" else "MASS")}
        an["A2"] = {"verdict": v2, "band": [A2_LO, A2_HI], "heads": {h: {"ci": an["A2_reliability"][h], "ok": a2_head_ok(an["A2_reliability"][h]),
                                                                           "side": ("ABOVE (observed unsafe mass exceeds predicted: an over-confident row)" if an["A2_reliability"][h]["lo"] > A2_HI else
                                                                                   "BELOW (predicted unsafe mass exceeds observed: an over-conservative row)" if an["A2_reliability"][h]["hi"] < A2_LO else "overlaps")}
                                                                       for h in A2_HEADS}}
        a1_list.append(v1)
        a2_list.append(v2)
        log(f"candidate {cid}: B (pessimistic EP)")
        c = cands[cid]
        bres = pessimistic_picks(p2["z6"], c["z6"], lab, p2["fold"], S, slot_source=slot_source)
        an["B"] = {k: v for k, v in bres.items() if k not in ("picks", "z6_B")}
        b_lo.append(float(bres["d_pdms_x100"]["lo"]))
        b_nc.append(float(bres["d_nc_unsafe_rate"]["hi"]))
        b_dac.append(float(bres["d_dac_unsafe_rate"]["hi"]))
        an["B"]["recovers_this_bank"] = bool(b_recovers([b_lo[-1]], [b_nc[-1]], [b_dac[-1]]))
    controls_ok = bool(all(v.get("pass", False) for v in controls.values()))
    void_null = bool(any(n.get("void", False) for n in nulls.values()))
    label = admission_label(a1_list, a2_list, controls_ok, void_null)
    warnings = []
    for cid, an in per.items():
        if an["n_changed"] == 0:
            warnings.append(f"{cid}: its picks are IDENTICAL to P2's on every set; A1 / A2 are vacuous and there is no navtest gain to look for")
        if an["A1_resolution"]["half_width_exceeds_bar"]:
            warnings.append(f"{cid}: the D_M half-width {an['A1_resolution']['D_M_half_width']:.4f} exceeds the A1 bar {A1_BAR}: the bar sits inside the interval noise, so a calibrated candidate can FAIL A1 by chance (see E-NULL)")
        if an["frac_changed"] < 0.05 and an["n_changed"] > 0:
            warnings.append(f"{cid}: only {an['frac_changed']:.3%} of the picks differ from P2's; D_M is diluted by the unchanged sets (see D_M_changed_only and C-POWER)")
        if an["B"]["lam_outside_unit_interval"]:
            warnings.append(f"{cid}: a fitted lam is outside [0, 1]; the 'pessimistic' rule is not a shrinkage on this bank")
    if form == "MASS" and not on_scale:
        warnings.append("the EP row is OFF-SCALE (slope CI outside [0.9, 1.1]): the rank-free mass form carries A1, as EP-RELIAB's rule would for this row too")
    return {"schema": SCHEMA, "warnings": warnings, "controls": controls, "controls_ok": controls_ok, "per_candidate": per, "a1_form_mode": a1_form_mode, "a1_form_used": form, "scale_on_all_banks": on_scale,
            "verdict": {"A1": dict(zip(cands, a1_list)), "A2": dict(zip(cands, a2_list)), "label": label, "next_arm": next_arm(label)},
            "B": {"recovers_all_banks": bool(b_recovers(b_lo, b_nc, b_dac)), "slot_source": slot_source, "per_bank_dpdms_x100_lo": b_lo, "per_bank_nc_unsafe_rate_diff_hi": b_nc, "per_bank_dac_unsafe_rate_diff_hi": b_dac},
            "null": {cid: {k: v for k, v in n.items()} for cid, n in nulls.items()}, "n_logs": p2["n_logs"], "n_sets": p2["n_sets"], "bootstrap": {"B": B, "seed": BOOT_SEED, "unit": "log", "percentiles": [2.5, 97.5]}}


def wait_for_ram(min_gb: float = MIN_AVAIL_GB) -> float:
    try:
        import psutil
    except ImportError:
        return float("nan")
    av = psutil.virtual_memory().available / 2 ** 30
    if av < min_gb:
        raise AdmissionError(f"available RAM {av:.2f} GB < {min_gb} GB")
    return av


def cmd_run(a) -> int:
    avail = wait_for_ram()
    L, P, F = libs()
    rows, rsha = read_ranked(a.ranked_logs, expected_sha256=(None if a.no_ranked_pin else RANKED_LOGS_SHA256))
    p2 = load_bank_units(a.p2_bank, rows)
    ids = a.cand_id or [os.path.basename(os.path.normpath(d)) for d in a.cand_bank]
    if len(ids) != len(a.cand_bank) or len(set(ids)) != len(ids):
        raise AdmissionError("--cand-id must be given once per --cand-bank and be unique")
    cands = {}
    for cid, d in zip(ids, a.cand_bank):
        c = load_bank_units(d, rows)
        cands[cid] = align_candidate(p2, c)
    seal = None
    if a.prereg:
        seal = check_prereg(a.prereg, a.candidate_ckpt_sha256, [c["manifest_sha256"] for c in cands.values()])
    elif not a.allow_unsealed:
        raise AdmissionError("no --prereg: a verdict needs the sealed EP-ADMISSION registration naming this candidate (use --allow-unsealed for a control / smoke run with NO authority)")
    res = run_admission(p2, cands, a1_form_mode=a.a1_form, slot_source=a.slot_source, null_reps_e=a.enull_reps)
    res.update({"created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "authority": ("SEALED REGISTRATION" if seal else "NONE (unsealed run: control / smoke only)"), "seal": seal,
                "inputs": {"p2_bank": a.p2_bank, "p2_unit_sha256": p2["unit_sha256"], "p2_manifest_sha256": p2["manifest_sha256"], "p2_ckpt": p2["ckpt"], "ranked_logs_sha256": rsha,
                           "candidates": {cid: {"bank": cands[cid]["bank_dir"], "ckpt": cands[cid]["ckpt"], "manifest_sha256": cands[cid]["manifest_sha256"], "unit_sha256": cands[cid]["unit_sha256"],
                                                "alignment": cands[cid]["checks"]} for cid in cands}},
                "code_sha256": {"ep_admission.py": sha256_file(os.path.abspath(__file__)), "epreliab_lib.py": EPRELIAB_LIB_SHA256}, "ram_available_gb": avail,
                "limits": ["SELECTION tier, SEEN scenes, label-lookup PDMS: T0-class, NOT a capability claim.",
                           "The interval answers 'another draw of LOGS' only (H-ESTIM-SEED-1): not another training run of the candidate, not another inference run.",
                           "P2 is in-sample on S0 (it biases the realised gain DOWN, toward OVER); whether the candidate is in-sample on S0 is a fact of ITS training set and must be stated in the registration.",
                           "A PASS is a NECESSARY condition for a navtest look, not evidence of a navtest gain."]})
    out = a.out
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(jsonable(res), indent=1, sort_keys=True))
    v = res["verdict"]
    print(f"ADMISSION {v['label']} | A1 {v['A1']} | A2 {v['A2']} | controls_ok {res['controls_ok']} | authority {res['authority']} | -> {out}")
    return 0 if res["controls_ok"] else 3


def cmd_selftest_real(a) -> int:
    """REAL-DATA REPLICATION (positive control of the gate): EP-RELIAB-PICKS' EP-led composite (EP-R1's c* row, out of fold) is fed through THIS tool as the candidate; every EP-RELIAB number must be reproduced
    (to 1e-9 against the landed result.json) and the gate must read A1 = FAIL (the known-bad candidate).  P2 as the candidate must read exactly 0 (C-ID).  Also proves the loader: the assembled P2 arrays equal
    the pickrule `arrays.npz` bit for bit."""
    L, P, F = libs()
    avail = wait_for_ram()
    rows, rsha = read_ranked(a.ranked_logs, expected_sha256=RANKED_LOGS_SHA256)
    p2 = load_bank_units(a.p2_bank, rows)
    z = np.load(a.arrays_npz, allow_pickle=False)
    loader = {k: bool(np.array_equal(v, z[k])) for k, v in (("z6", p2["z6"]), ("lab", p2["lab"]), ("top8", p2["top8"]), ("log_idx", p2["log_idx"]), ("fold", p2["fold"]))}
    loader["arrays_npz_sha256"] = sha256_file(a.arrays_npz)
    loader["pass"] = all(v for k, v in loader.items() if isinstance(v, bool))
    landed = json.load(open(a.landed_result, "r", encoding="utf-8"))
    S = make_stats(p2["log_idx"], p2["n_logs"])
    res = {"schema": SCHEMA + "-selftest-real", "loader_equals_pickrule_arrays": loader, "composites": {}}
    ok = loader["pass"]
    for s in (0, 1):
        zc = p2["z6"].copy()
        zc[..., EP] = z[f"ep_OOF_s{s}"]
        an = analyze_candidate(p2["z6"], zc, p2["lab"], S)
        ref = landed["per_seed"][str(s)]
        cmpd = {}
        def cmp(name, mine, theirs):
            d = max(abs(float(mine[k]) - float(theirs[k])) for k in ("point", "lo", "hi"))
            cmpd[name] = {"max_abs_diff": d, "pass": bool(d <= 1e-9)}
        cmp("D_M", an["G"]["D_M"], ref["G"]["D_M"])
        cmp("delta", an["G"]["delta"], ref["G"]["delta"])
        cmp("pred_gain", an["G"]["pred_gain"], ref["G"]["pred_gain"])
        cmp("real_gain", an["G"]["real_gain"], ref["G"]["real_gain"])
        cmp("scale_slope", an["scale"]["slope"], ref["scale"]["slope"])
        for h in ("NC", "DAC", "TTC"):
            cmp(f"reliability_{h}_deficit", an["A2_reliability"][h], ref["reliability"][h]["composite_picks"]["deficit_form_sum_1my_over_sum_1msigma"])
        cmp("attribution_total_x100", an["attribution"]["total_x100"], ref["attribution"]["total_x100"])
        cmpd["n_changed"] = {"mine": an["n_changed"], "landed": ref["n_changed"], "pass": bool(an["n_changed"] == ref["n_changed"])}
        v1 = a1_verdict(an["G"]["D_M"])
        v2 = a2_verdict(an["A2_reliability"])
        cmpd["gate_reads_the_known_bad_candidate_as_FAIL-A1"] = {"a1": v1, "pass": bool(v1 == "FAIL-A1")}
        ok = ok and all(v["pass"] for v in cmpd.values())
        res["composites"][f"EP-R1_c*_OOF_s{s}"] = {"vs_landed": cmpd, "A1": {"verdict": v1, "D_M": an["G"]["D_M"]}, "A2": {"verdict": v2, "heads": an["A2_reliability"]}, "A3": an["A3"],
                                                    "A1_resolution": an["A1_resolution"], "n_changed": an["n_changed"], "frac_changed": an["frac_changed"]}
    res["C-ID"] = control_id(p2["z6"], p2["lab"], S)
    ok = ok and res["C-ID"]["pass"]
    # the null on the known-bad candidate's pick geometry: how often would A1 FALSELY refuse a candidate whose picks carry no information?  (the R51 / R52 number the draft asks for)
    zc0 = p2["z6"].copy()
    zc0[..., EP] = z["ep_OOF_s0"]
    res["null_on_EP-R1_geometry_seed0"] = run_null(p2["z6"], zc0, p2["lab"], S)
    res["power_on_EP-R1_geometry_seed0"] = control_power(p2["z6"], zc0, p2["lab"], S)
    res["reach_on_EP-R1_geometry_seed0"] = control_reach(p2["z6"], zc0, p2["lab"], S)
    ok = ok and res["power_on_EP-R1_geometry_seed0"]["pass"] and res["reach_on_EP-R1_geometry_seed0"]["pass"]
    # SMOKE of (B) on the known candidate: the identity lam = 1 reproduces the candidate's own picks exactly; the fitted lam per fold is reported (NO authority; EP-R1 is not the admission candidate)
    bid = pessimistic_picks(p2["z6"], zc0, p2["lab"], p2["fold"], S, lam_override=1.0, slot_source="label")
    res["B_smoke_on_EP-R1_seed0"] = {"identity_lam1_n_changed_vs_candidate": bid["n_changed_vs_candidate"], "identity_pass": bool(bid["n_changed_vs_candidate"] == 0)}
    ok = ok and res["B_smoke_on_EP-R1_seed0"]["identity_pass"]
    bfit = pessimistic_picks(p2["z6"], zc0, p2["lab"], p2["fold"], S, slot_source="label")
    res["B_smoke_on_EP-R1_seed0"]["fitted"] = {k: v for k, v in bfit.items() if k not in ("picks", "z6_B")}
    res["A1_analytic_oc_at_EP-R1_half_width_seed0"] = a1_analytic_oc(res["composites"]["EP-R1_c*_OOF_s0"]["A1_resolution"]["D_M_half_width"])
    brow = pessimistic_picks(p2["z6"], zc0, p2["lab"], p2["fold"], S, slot_source="row")
    res["B_smoke_on_EP-R1_seed0"]["fitted_slot_source_row"] = {k: v for k, v in brow.items() if k not in ("picks", "z6_B")}
    res["pass"] = bool(ok)
    res["ram_available_gb"] = avail
    res["code_sha256"] = {"ep_admission.py": sha256_file(os.path.abspath(__file__)), "epreliab_lib.py": EPRELIAB_LIB_SHA256}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(jsonable(res), indent=1, sort_keys=True))
    print(f"SELFTEST_REAL pass={res['pass']} loader={loader['pass']} C-ID={res['C-ID']['pass']} -> {a.out}")
    return 0 if res["pass"] else 3


def cmd_oc(a) -> int:
    """Run BEFORE sealing: what does the A1 bar mean at a D_M interval half-width h?  Reads no bank and no label."""
    res = {"schema": SCHEMA + "-oc", "authority": "NONE (a pre-sealing planning aid)", "oc": a1_analytic_oc(a.half_width, a.bar)}
    txt = json.dumps(jsonable(res), indent=1, sort_keys=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(txt)
    print(txt)
    return 0


def cmd_branch_table(a) -> int:
    mr = mutation_report()
    print(json.dumps(jsonable(mr), indent=1))
    return 0 if (not mr["unmutated_failures"] and mr["n_caught"] == mr["n_mutations"]) else 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="admission read of a candidate bank (or several seeds) against P2")
    r.add_argument("--p2-bank", required=True)
    r.add_argument("--cand-bank", action="append", required=True, help="a candidate bank dir (S0 unit format + MANIFEST.sha256.json); repeat for several seeds")
    r.add_argument("--cand-id", action="append", default=None)
    r.add_argument("--ranked-logs", required=True)
    r.add_argument("--no-ranked-pin", action="store_true", help="do not pin ranked_fit_logs.tsv to the registered S0 sha256 (synthetic banks only)")
    r.add_argument("--prereg", default=None, help="the SEALED EP-ADMISSION registration naming this candidate (needs its .sha256 sidecar)")
    r.add_argument("--candidate-ckpt-sha256", default=None)
    r.add_argument("--allow-unsealed", action="store_true", help="run without a registration: authority NONE")
    r.add_argument("--a1-form", choices=("mass", "auto"), default="mass", help="mass = the draft's literal D_M; auto = EP-RELIAB's rule (ON-SCALE -> Delta, else D_M). The Master Mind seals this.")
    r.add_argument("--slot-source", choices=("label", "row"), default="label")
    r.add_argument("--enull-reps", type=int, default=NULL_E_REPS)
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_run)
    o = sub.add_parser("oc", help="PRE-SEALING: what the A1 bar means at a given D_M interval half-width (default: EP-R1's measured 0.0263); reads no bank")
    o.add_argument("--half-width", type=float, default=0.0263)
    o.add_argument("--bar", type=float, default=A1_BAR)
    o.add_argument("--out", default=None)
    o.set_defaults(fn=cmd_oc)
    t = sub.add_parser("selftest-real", help="replicate EP-RELIAB-PICKS' numbers through this tool (the gate must read the known-bad EP-led composite as FAIL-A1)")
    t.add_argument("--p2-bank", required=True)
    t.add_argument("--ranked-logs", required=True)
    t.add_argument("--arrays-npz", required=True)
    t.add_argument("--landed-result", required=True)
    t.add_argument("--out", required=True)
    t.set_defaults(fn=cmd_selftest_real)
    b = sub.add_parser("branch-table", help="the decision functions' branch table and RED-mutation report")
    b.set_defaults(fn=cmd_branch_table)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
