"""EP-RELIAB-PICKS library: the EP gain predicted vs realised at the composite's picks, the rank-free mass form, the exact PDMS attribution, reliability at picks, the scale gate, the registered
decision functions, the planted checks and the controls.

REGISTERED SOURCE: PREREG_EP_RELIAB_PICKS.md (sha256 PREREG_SHA256 below).  Every constant cites its section.  No stage of this package computes a quantity unless the prereg file (and every
amendment) still hashes to its sealed value (`require_prereg`).  CPU only; numpy plus the pinned landed modules (pickrule_lib, epr1, fastneg_lib).
"""
from __future__ import annotations

import glob
import hashlib
import importlib.util
import math
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = "ep-reliab-picks-v1"
PREREG_SHA256 = "e54067d4b16f598c1710064a7f01eafe825d19c01eadbe7124b86fa67e9a8e48"          # PREREG_EP_RELIAB_PICKS.md, sealed 2026-10-19T10:57:25+02:00 before any real-data stage
AMENDMENTS: dict = {"PREREG_EP_RELIAB_PICKS_AMENDMENT1.md": "9ed702999ae3e771392e76f75791cbc1c753a2a85c92c93f1254ebfa8be5a1ed"}      # Amendment 1 (C-POWER label-independent), sealed before any real-data stage
PICKRULE_LIB_SHA256 = "555c0a3468acd94adeeeb10f47527644a2b9d49722f554fed2da1fb29668eaeb"     # prereg s2
EPR1_SHA256 = "a751c63eb8e38343a7ec1286114a158046ce4ec13390b2eea0a4b1d2237c38d8"             # prereg s2
FASTNEG_LIB_SHA256 = "0c0b399b37a47ee3e17f9d5484bc7f9521ff7dbe6bea37e8c7a38042668667ca"      # prereg s2
PLANNER_SHA256 = "91437cd09362ff872a861c6166b3c00eee98ce9b98c4e85b6a95ca133e29e211"          # prereg s2
ARRAYS_SHA256 = "29130bcd4fda758e5f03bb84c550eb7d253197806528a5b67dd893d1df9824dc"           # prereg s2
REF = "C:/Users/Admin/par18/epreliab/ref"
PARENT = "C:/Users/Admin/par18"

SEEDS = (0, 1)                                   # prereg s2
N_LOGS, BOOT_B, BOOT_SEED = 117, 10000, 20260927   # prereg s2
NC, DAC, EP, TTC, CMF, DDC = 0, 1, 2, 3, 4, 5
TERMS = ("NC", "DAC", "EP", "TTC", "C")          # prereg s4 (D): the FIXED substitution order; label columns 0..4
SCALE_LO, SCALE_HI = 0.9, 1.1                    # prereg s5: ON-SCALE iff the slope CI lies inside [0.9, 1.1]
PLANT = 0.10                                     # prereg s7 C-POWER
NULL_C_REPS, NULL_E_REPS, NULL_SALT, NULL_SEED_BASE = 5, 200, 7919, 20261019   # prereg s7 C-NULL / E-NULL
NULL_VOID_FIRES = 2
MIN_AVAIL_GB = 3.0
TOL_ID_LANDED = 1e-9                             # prereg s7 C-ID-LANDED / C-ID-R53
TOL_ATTR_EXACT = 1e-12                           # prereg s7 C-ATTR
TOL_ATTR_LANDED_X100 = 1e-4                      # prereg s7 C-ATTR (float64 attribution vs the landed float32 PDMS)
TOL_REACH = 1e-9
WORDS = {"OVER": "EP-OVERPREDICTED", "UNDER": "EP-UNDERPREDICTED", "WITHIN": "EP-WITHIN-NOISE"}


class EpReliabError(RuntimeError):
    """An integrity check, a pin or a precondition failed."""


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# hashing, pins, pinned module loading
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def require_prereg(path: str | None = None) -> str:
    """Refuse unless the prereg file AND every amendment on disk still hash to their sealed values (a change comes only by a written amendment BEFORE the number it governs)."""
    p = path or os.path.join(HERE, "..", "PREREG_EP_RELIAB_PICKS.md")
    if not os.path.exists(p):
        raise EpReliabError(f"the sealed registration file {p} does not exist")
    sha = sha256_file(p)
    if sha != PREREG_SHA256:
        raise EpReliabError(f"{p} sha256 {sha} != the sealed {PREREG_SHA256}: refuse (write an amendment BEFORE the number it governs)")
    d = os.path.dirname(os.path.abspath(p))
    on_disk = sorted(os.path.basename(x) for x in glob.glob(os.path.join(d, "PREREG_EP_RELIAB_PICKS_AMENDMENT*.md")))
    for name in on_disk:
        if name not in AMENDMENTS:
            raise EpReliabError(f"amendment {name} exists on disk but is not registered in epreliab_lib.AMENDMENTS: refuse")
    for name, want in AMENDMENTS.items():
        f = os.path.join(d, name)
        if not os.path.exists(f):
            raise EpReliabError(f"registered amendment {name} does not exist")
        if sha256_file(f) != want:
            raise EpReliabError(f"amendment {name} sha256 != the sealed {want}")
    return PREREG_SHA256


def load_pinned(name: str, candidates: list, want_sha: str):
    """Import a landed module by path after checking its sha256 against the pin; a tampered or missing copy is refused."""
    if name in sys.modules and getattr(sys.modules[name], "_EPRELIAB_PINNED", None) == want_sha:
        return sys.modules[name]
    tried = []
    for c in candidates:
        if c and os.path.exists(c):
            sha = sha256_file(c)
            if sha != want_sha:
                raise EpReliabError(f"{name} at {c} has sha256 {sha} != the pinned {want_sha}")
            if os.path.dirname(os.path.abspath(c)) not in sys.path:
                sys.path.insert(0, os.path.dirname(os.path.abspath(c)))
            spec = importlib.util.spec_from_file_location(name, c)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            mod._EPRELIAB_PINNED = want_sha
            mod._PICKRULE_PINNED = want_sha          # the landed pickrule_lib.load_pinned cache key
            return mod
        tried.append(c)
    raise EpReliabError(f"{name} not found; probed: {tried}")


def _cands(rel: str, fname: str, explicit: str | None) -> list:
    return [explicit, os.path.join(HERE, "..", "..", rel, "code", fname), f"{REF}/{fname}"]


def load_pickrule(explicit: str | None = None):
    return load_pinned("pickrule_lib", _cands("2026-10-19-pickrule-audit-nc", "pickrule_lib.py", explicit), PICKRULE_LIB_SHA256)


def load_epr1(explicit: str | None = None):
    return load_pinned("epr1", _cands("2026-10-18-ep-r1", "epr1.py", explicit), EPR1_SHA256)


def load_fastneg(explicit: str | None = None):
    return load_pinned("fastneg_lib", _cands("2026-10-19-fastneg-census", "fastneg_lib.py", explicit), FASTNEG_LIB_SHA256)


def v1_w() -> tuple:
    """The PDMS weights.  SOURCE (prereg s4 D): `REFePlanner.V1_W = (5.0, 5.0, 2.0)` at refe/planner.py:520 (comment there: navsim @ 3e8291b pdm_scorer.py:38-42), imported from the pinned
    pickrule_lib.V1_W and equal bit for bit to epr1.pdms_np (control C-PDMS)."""
    return tuple(load_pickrule().V1_W)


def jsonable(o):
    if isinstance(o, dict):
        return {(str(k) if not isinstance(k, str) else k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the registered decision functions (pure; `branch_table_failures` is the table every mutation must break)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def stat_class(ci: dict) -> str:
    """prereg s6, one seed, one statistic (Delta or D_M): lower CI > 0 -> OVER; upper CI < 0 -> UNDER; else WITHIN (a NaN bound is WITHIN)."""
    if ci["lo"] > 0:
        return "OVER"
    if ci["hi"] < 0:
        return "UNDER"
    return "WITHIN"


def outcome_class(classes: list) -> str:
    """prereg s6: EP-OVERPREDICTED iff BOTH seeds read OVER; EP-UNDERPREDICTED iff BOTH read UNDER; otherwise EP-WITHIN-NOISE."""
    if len(classes) > 0 and all(c == "OVER" for c in classes):
        return "EP-OVERPREDICTED"
    if len(classes) > 0 and all(c == "UNDER" for c in classes):
        return "EP-UNDERPREDICTED"
    return "EP-WITHIN-NOISE"


def scale_on(ci: dict) -> bool:
    """prereg s5, one seed: ON-SCALE iff the within-set slope CI lies inside [0.9, 1.1] (closed bounds; a NaN bound is OFF)."""
    return bool(ci["lo"] >= 0.9 and ci["hi"] <= 1.1)


def scale_gate(cis: list) -> bool:
    """prereg s5: the row is ON-SCALE iff BOTH seeds are ON-SCALE."""
    return bool(len(cis) > 0 and all(scale_on(c) for c in cis))


def deciding_form(on_scale: bool) -> str:
    """prereg s6: ON-SCALE -> Delta; OFF-SCALE -> the rank-free mass form D_M."""
    return "DELTA" if on_scale else "MASS"


def dominant_term(contrib_points: list, total_point: float) -> str:
    """prereg s4 (D), one seed: a point dPDMS >= 0 reads NO-LOSS; else the term with the most negative contribution (the largest share of the loss); ties go to the fixed order NC, DAC, EP, TTC, C."""
    if not (total_point < 0):
        return "NO-LOSS"
    return TERMS[int(np.argmin(np.asarray(contrib_points, np.float64)))]


def dominant_agree(names: list) -> str:
    """prereg s4 (D): the arm names the dominant term iff both seeds name the same one, else SEED-DEPENDENT."""
    if len(names) > 0 and len(set(names)) == 1:
        return names[0]
    return "SEED-DEPENDENT"


def null_fires(outcome: str) -> bool:
    """prereg s7 C-NULL: a replicate FIRES iff its outcome is not WITHIN."""
    return bool(outcome != "EP-WITHIN-NOISE")


def null_void(fires: list) -> bool:
    """prereg s7 C-NULL: VOID iff the null fires on 2 or more of its 5 replicates."""
    return bool(sum(1 for f in fires if f) >= NULL_VOID_FIRES)


def final_label(outcome: str, controls_ok: bool, void_null: bool) -> str:
    """prereg s6: VOID if a gating control failed or the C-NULL fired on >= 2 of 5; else the outcome."""
    if not controls_ok:
        return "VOID (a gating control failed)"
    if void_null:
        return "VOID (C-NULL fired on >= 2 of 5 replicates)"
    return outcome


NEXT_ARM = {
    "EP-OVERPREDICTED": "EP-led head changes are CLOSED as a lever on these features; the EP lever moves to scene-specific EP (trunk / P6) and the Master Mind drafts a pessimistic-EP pick-rule registration (EP shrunk toward its slot prior)",
    "EP-WITHIN-NOISE": "the loss is in the rule's risk weighting: a rule-weight audit (do our planner rule's term weights match the PDMS formula the labels are scored with?)",
    "EP-UNDERPREDICTED": "the loss is in the rule's risk weighting: a rule-weight audit (do our planner rule's term weights match the PDMS formula the labels are scored with?)",
}


def next_arm(label: str) -> str:
    if label.startswith("VOID"):
        return "none from this read; the failing control is reported and the instrument repaired first"
    return NEXT_ARM[label]


def form_sensitive(outcome_deciding: str, outcome_other: str) -> bool:
    return bool(outcome_deciding != outcome_other)


# (old, new) text mutations of this file; EACH must make `branch_table_failures` non-empty (prereg s7 C-REACH-DECISION)
DECISION_MUTATIONS = [
    ('if ci["lo"] > 0:\n        return "OVER"', 'if ci["lo"] >= 0:\n        return "OVER"'),
    ('if ci["hi"] < 0:\n        return "UNDER"', 'if ci["hi"] <= 0:\n        return "UNDER"'),
    ('if ci["lo"] > 0:\n        return "OVER"', 'if ci["hi"] > 0:\n        return "OVER"'),
    ('if ci["hi"] < 0:\n        return "UNDER"', 'if ci["lo"] < 0:\n        return "UNDER"'),
    ('if len(classes) > 0 and all(c == "OVER" for c in classes):', 'if len(classes) > 0 and any(c == "OVER" for c in classes):'),
    ('if len(classes) > 0 and all(c == "UNDER" for c in classes):', 'if len(classes) > 0 and any(c == "UNDER" for c in classes):'),
    ('return bool(ci["lo"] >= 0.9 and ci["hi"] <= 1.1)', 'return bool(ci["lo"] > 0.9 and ci["hi"] <= 1.1)'),
    ('return bool(ci["lo"] >= 0.9 and ci["hi"] <= 1.1)', 'return bool(ci["lo"] >= 0.9 and ci["hi"] < 1.1)'),
    ('return bool(ci["lo"] >= 0.9 and ci["hi"] <= 1.1)', 'return bool(ci["lo"] >= 0.9 or ci["hi"] <= 1.1)'),
    ('return bool(ci["lo"] >= 0.9 and ci["hi"] <= 1.1)', 'return bool(ci["lo"] >= 0.8 and ci["hi"] <= 1.1)'),
    ('return bool(ci["lo"] >= 0.9 and ci["hi"] <= 1.1)', 'return bool(ci["lo"] >= 0.9 and ci["hi"] <= 1.2)'),
    ("return bool(len(cis) > 0 and all(scale_on(c) for c in cis))", "return bool(len(cis) > 0 and any(scale_on(c) for c in cis))"),
    ('return "DELTA" if on_scale else "MASS"', 'return "MASS" if on_scale else "DELTA"'),
    ("if not (total_point < 0):", "if not (total_point <= 0):"),
    ("return TERMS[int(np.argmin(np.asarray(contrib_points, np.float64)))]", "return TERMS[int(np.argmax(np.asarray(contrib_points, np.float64)))]"),
    ("if len(names) > 0 and len(set(names)) == 1:", "if len(names) > 0 and len(set(names)) >= 1:"),
    ("return bool(outcome != \"EP-WITHIN-NOISE\")", "return bool(outcome == \"EP-WITHIN-NOISE\")"),
    ("return bool(sum(1 for f in fires if f) >= NULL_VOID_FIRES)", "return bool(sum(1 for f in fires if f) > NULL_VOID_FIRES)"),
    ("if not controls_ok:\n        return \"VOID (a gating control failed)\"", "if controls_ok:\n        return \"VOID (a gating control failed)\""),
    ('"EP-WITHIN-NOISE": "the loss is in the rule\'s risk weighting', '"EP-WITHIN-NOISE": "EP-led head changes are CLOSED; the loss is in the rule\'s risk weighting'),
]


def branch_table_failures(mod) -> list:
    """Every decision function on a synthetic branch table.  `mod` is this module or a MUTATED copy; returns the failing cases (empty = every branch right)."""
    bad = []

    def chk(name, got, want):
        if got != want:
            bad.append(f"{name}: got {got!r}, want {want!r}")
    ci = lambda lo, hi: {"lo": lo, "hi": hi}                                                          # noqa: E731
    nan = float("nan")
    chk("class lo>0 -> OVER", mod.stat_class(ci(0.01, 0.2)), "OVER")
    chk("class hi<0 -> UNDER", mod.stat_class(ci(-0.2, -0.01)), "UNDER")
    chk("class straddle", mod.stat_class(ci(-0.1, 0.1)), "WITHIN")
    chk("class lo=0 is not >", mod.stat_class(ci(0.0, 0.3)), "WITHIN")
    chk("class hi=0 is not <", mod.stat_class(ci(-0.3, 0.0)), "WITHIN")
    chk("class nan bounds", mod.stat_class(ci(nan, nan)), "WITHIN")
    chk("class lo>0 but hi huge (lo decides)", mod.stat_class(ci(0.001, 9.0)), "OVER")
    chk("class hi<0 but lo very negative (hi decides)", mod.stat_class(ci(-9.0, -0.001)), "UNDER")
    chk("class both bounds positive vs lo only", mod.stat_class(ci(-0.5, 0.5)), "WITHIN")
    chk("outcome both OVER", mod.outcome_class(["OVER", "OVER"]), "EP-OVERPREDICTED")
    chk("outcome both UNDER", mod.outcome_class(["UNDER", "UNDER"]), "EP-UNDERPREDICTED")
    chk("outcome O/W", mod.outcome_class(["OVER", "WITHIN"]), "EP-WITHIN-NOISE")
    chk("outcome W/O", mod.outcome_class(["WITHIN", "OVER"]), "EP-WITHIN-NOISE")
    chk("outcome U/W", mod.outcome_class(["UNDER", "WITHIN"]), "EP-WITHIN-NOISE")
    chk("outcome O/U", mod.outcome_class(["OVER", "UNDER"]), "EP-WITHIN-NOISE")
    chk("outcome both WITHIN", mod.outcome_class(["WITHIN", "WITHIN"]), "EP-WITHIN-NOISE")
    chk("outcome empty", mod.outcome_class([]), "EP-WITHIN-NOISE")
    chk("scale inside", mod.scale_on(ci(0.95, 1.05)), True)
    chk("scale closed lower bound", mod.scale_on(ci(0.9, 1.0)), True)
    chk("scale closed upper bound", mod.scale_on(ci(1.0, 1.1)), True)
    chk("scale both closed bounds", mod.scale_on(ci(0.9, 1.1)), True)
    chk("scale lower below", mod.scale_on(ci(0.89, 1.0)), False)
    chk("scale upper above", mod.scale_on(ci(1.0, 1.11)), False)
    chk("scale both outside (and, not or)", mod.scale_on(ci(0.7, 1.3)), False)
    chk("scale only lower ok (and, not or)", mod.scale_on(ci(0.95, 1.5)), False)
    chk("scale only upper ok (and, not or)", mod.scale_on(ci(0.5, 1.05)), False)
    chk("scale lower just below the 0.8 constant", mod.scale_on(ci(0.85, 1.0)), False)
    chk("scale upper just below the 1.2 constant", mod.scale_on(ci(1.0, 1.15)), False)
    chk("scale nan", mod.scale_on(ci(nan, nan)), False)
    chk("gate both on", mod.scale_gate([ci(0.95, 1.05), ci(0.92, 1.08)]), True)
    chk("gate one off (all, not any)", mod.scale_gate([ci(0.95, 1.05), ci(0.5, 0.7)]), False)
    chk("gate other one off", mod.scale_gate([ci(0.5, 0.7), ci(0.95, 1.05)]), False)
    chk("gate both off", mod.scale_gate([ci(0.5, 0.7), ci(0.2, 0.3)]), False)
    chk("gate empty", mod.scale_gate([]), False)
    chk("form on", mod.deciding_form(True), "DELTA")
    chk("form off", mod.deciding_form(False), "MASS")
    chk("dominant NC", mod.dominant_term([-0.5, -0.1, 0.2, -0.2, 0.0], -0.6), "NC")
    chk("dominant TTC", mod.dominant_term([-0.1, -0.1, 0.2, -0.9, 0.0], -0.9), "TTC")
    chk("dominant C", mod.dominant_term([0.1, 0.0, 0.2, -0.1, -0.7], -0.5), "C")
    chk("dominant tie goes to the fixed order", mod.dominant_term([-0.3, -0.3, 0.1, 0.0, 0.0], -0.5), "NC")
    chk("dominant tie EP/TTC", mod.dominant_term([0.0, 0.0, -0.3, -0.3, 0.0], -0.6), "EP")
    chk("dominant NO-LOSS positive", mod.dominant_term([0.1, 0.0, 0.2, 0.0, 0.0], 0.3), "NO-LOSS")
    chk("dominant NO-LOSS exactly 0", mod.dominant_term([0.0, 0.0, 0.0, 0.0, 0.0], 0.0), "NO-LOSS")
    chk("dominant: a positive EP gain is not the dominant term (most negative wins)", mod.dominant_term([-0.5, -0.4, 0.9, -0.6, -0.1], -0.7), "TTC")
    chk("agree same", mod.dominant_agree(["TTC", "TTC"]), "TTC")
    chk("agree differ", mod.dominant_agree(["TTC", "DAC"]), "SEED-DEPENDENT")
    chk("agree NO-LOSS", mod.dominant_agree(["NO-LOSS", "NO-LOSS"]), "NO-LOSS")
    chk("agree empty", mod.dominant_agree([]), "SEED-DEPENDENT")
    chk("null fires OVER", mod.null_fires("EP-OVERPREDICTED"), True)
    chk("null fires UNDER", mod.null_fires("EP-UNDERPREDICTED"), True)
    chk("null silent WITHIN", mod.null_fires("EP-WITHIN-NOISE"), False)
    chk("null void 2 of 5", mod.null_void([True, True, False, False, False]), True)
    chk("null not void 1 of 5", mod.null_void([True, False, False, False, False]), False)
    chk("null void 5 of 5", mod.null_void([True] * 5), True)
    chk("null not void 0 of 5", mod.null_void([False] * 5), False)
    chk("final OVER", mod.final_label("EP-OVERPREDICTED", True, False), "EP-OVERPREDICTED")
    chk("final control failed", mod.final_label("EP-OVERPREDICTED", False, False).split(" (")[0], "VOID")
    chk("final null void", mod.final_label("EP-WITHIN-NOISE", True, True).split(" (")[0], "VOID")
    chk("final control failed beats null", mod.final_label("EP-OVERPREDICTED", False, True), "VOID (a gating control failed)")
    chk("final WITHIN passes through", mod.final_label("EP-WITHIN-NOISE", True, False), "EP-WITHIN-NOISE")
    chk("next OVER", mod.next_arm("EP-OVERPREDICTED").split(";")[0].split(" (")[0], "EP-led head changes are CLOSED as a lever on these features")
    chk("next WITHIN is the rule-weight audit", mod.next_arm("EP-WITHIN-NOISE").split(":")[0], "the loss is in the rule's risk weighting")
    chk("next WITHIN is not the closure", mod.next_arm("EP-WITHIN-NOISE").startswith("EP-led"), False)
    chk("next UNDER is the rule-weight audit", mod.next_arm("EP-UNDERPREDICTED").split(":")[0], "the loss is in the rule's risk weighting")
    chk("next VOID", mod.next_arm("VOID (a gating control failed)").split(";")[0], "none from this read")
    chk("form sensitive", mod.form_sensitive("EP-OVERPREDICTED", "EP-WITHIN-NOISE"), True)
    chk("form not sensitive", mod.form_sensitive("EP-WITHIN-NOISE", "EP-WITHIN-NOISE"), False)
    return bad


def mutated_copy(old: str, new: str):
    """A copy of THIS module's source with the FIRST occurrence of `old` replaced by `new` (the anchor must exist)."""
    path = os.path.join(HERE, "epreliab_lib.py")
    src = open(path, "r", encoding="utf-8").read()
    assert old in src, f"mutation anchor not found: {old!r}"
    ns = types.ModuleType("epreliab_lib_mut")
    ns.__file__ = path
    exec(compile(src.replace(old, new, 1), path, "exec"), ns.__dict__)
    return ns


def mutation_report() -> dict:
    """prereg s7 C-REACH-DECISION: the unmutated table is empty and EVERY mutation of DECISION_MUTATIONS breaks it."""
    unmut = branch_table_failures(sys.modules[__name__] if __name__ in sys.modules else _self())
    caught = []
    for old, new in DECISION_MUTATIONS:
        m = mutated_copy(old, new)
        caught.append({"mutation": f"{old[:60]!r} -> {new[:60]!r}", "caught": len(branch_table_failures(m)) > 0})
    return {"unmutated_failures": unmut, "n_mutations": len(caught), "n_caught": int(sum(c["caught"] for c in caught)), "uncaught": [c["mutation"] for c in caught if not c["caught"]]}


def _self():
    ns = types.ModuleType("epreliab_lib_self")
    ns.__file__ = os.path.join(HERE, "epreliab_lib.py")
    exec(compile(open(ns.__file__, "r", encoding="utf-8").read(), ns.__file__, "exec"), ns.__dict__)
    return ns


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the quantities (prereg s4)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def safe_mask(lab: np.ndarray) -> np.ndarray:
    """prereg s5: safe = NC >= 1 and DAC >= 1 (`epr1.safe_np`; a control compares them)."""
    return (lab[..., NC] >= 1.0) & (lab[..., DAC] >= 1.0)


def pair_terms(sig: np.ndarray, y: np.ndarray, pk_p: np.ndarray, pk_e: np.ndarray) -> dict:
    """Per-set vectors at the two picks.  sig [N, 64] the predicted EP (sigmoid scale, float64), y [N, 64] the EP label (float64); pk_p P2's pick, pk_e the composite's."""
    P = load_pickrule()
    s_e, s_p = P.at(sig, pk_e), P.at(sig, pk_p)
    y_e, y_p = P.at(y, pk_e), P.at(y, pk_p)
    return {"s_e": s_e, "s_p": s_p, "y_e": y_e, "y_p": y_p, "ghat": s_e - s_p, "g": y_e - y_p, "changed": (pk_e != pk_p)}


def core_stats(S, t: dict) -> dict:
    """The two decision statistics: Delta (the CI on mean(ghat - g), paired) and D_M = M_p - M_e (prereg s4 G, G-mass)."""
    return {"delta": S.mean(t["ghat"] - t["g"]), "D_M": S.ratio_diff(t["y_p"], t["s_p"], t["y_e"], t["s_e"])}


def full_gain_stats(S, t: dict) -> dict:
    out = core_stats(S, t)
    out["pred_gain"] = S.mean(t["ghat"])
    out["real_gain"] = S.mean(t["g"])
    out["M_e"] = S.ratio(t["y_e"], t["s_e"])
    out["M_p"] = S.ratio(t["y_p"], t["s_p"])
    ch = t["changed"].astype(np.float64)
    out["n_changed"] = int(ch.sum())
    if out["n_changed"] > 0:
        out["changed_only"] = {"delta": S.ratio((t["ghat"] - t["g"]) * ch, ch), "pred_gain": S.ratio(t["ghat"] * ch, ch), "real_gain": S.ratio(t["g"] * ch, ch)}
    return out


def scale_slope(S, sig: np.ndarray, y: np.ndarray, safe: np.ndarray) -> dict:
    """prereg s5: the within-set calibration slope of y on sigma over the SAFE proposals: demean both over each set's safe proposals, slope = sum(sc * yc) / sum(sc ** 2) as a ratio of per-log sums.
    Sets with fewer than 2 safe proposals contribute 0 to both sums."""
    m = safe.astype(np.float64)
    n = np.maximum(m.sum(axis=1), 1.0)[:, None]
    sc = (sig - (sig * m).sum(axis=1, keepdims=True) / n) * m
    yc = (y - (y * m).sum(axis=1, keepdims=True) / n) * m
    return S.ratio((sc * yc).sum(axis=1), (sc * sc).sum(axis=1))


def _pdms_terms(t: np.ndarray, w: tuple) -> np.ndarray:
    """t [N, 5] in TERMS order (NC, DAC, EP, TTC, C) -> float64 PDMS = NC * DAC * (w_ep EP + w_ttc TTC + w_c C) / W (the planner.py:558-561 expression, prereg s4 D)."""
    return t[:, 0] * t[:, 1] * (w[0] * t[:, 2] + w[1] * t[:, 3] + w[2] * t[:, 4]) / float(sum(w))


def attribute(lab_a: np.ndarray, lab_b: np.ndarray, w: tuple | None = None, order: tuple = (0, 1, 2, 3, 4)) -> tuple:
    """prereg s4 (D): counterfactual substitution of the five terms of pick B (the composite's) into pick A (P2's), in the fixed `order` (default NC, DAC, EP, TTC, C).
    lab_a, lab_b [N, >=5] label rows of the two picks (cast to float64).  -> (contrib [N, 5] in TERMS order, total [N] = PDMS(B) - PDMS(A) in float64)."""
    w = w or v1_w()
    A, B = np.asarray(lab_a, np.float64)[:, :5].copy(), np.asarray(lab_b, np.float64)[:, :5]
    cur = A.copy()
    prev = _pdms_terms(cur, w)
    contrib = np.zeros((len(A), 5), np.float64)
    for k in order:
        cur[:, k] = B[:, k]
        val = _pdms_terms(cur, w)
        contrib[:, k] = val - prev
        prev = val
    return contrib, _pdms_terms(B, w) - _pdms_terms(A, w)


def shapley(lab_a: np.ndarray, lab_b: np.ndarray, w: tuple | None = None) -> np.ndarray:
    """prereg s4 (D) sensitivity: the Shapley attribution over all 5! orders; coalition value v(S) = PDMS with the terms in S taken from B and the others from A.  -> [N, 5] in TERMS order."""
    w = w or v1_w()
    A, B = np.asarray(lab_a, np.float64)[:, :5], np.asarray(lab_b, np.float64)[:, :5]
    v = np.empty((32, len(A)), np.float64)
    for m in range(32):
        st = A.copy()
        for k in range(5):
            if (m >> k) & 1:
                st[:, k] = B[:, k]
        v[m] = _pdms_terms(st, w)
    phi = np.zeros((len(A), 5), np.float64)
    for k in range(5):
        for m in range(32):
            if (m >> k) & 1:
                continue
            s = bin(m).count("1")
            phi[:, k] += math.factorial(s) * math.factorial(4 - s) / math.factorial(5) * (v[m | (1 << k)] - v[m])
    return phi


def signed_ratio_ci(num_pl: np.ndarray, den_pl: np.ndarray, draws: np.ndarray) -> dict:
    """sum(num) / sum(den) over logs with the percentile log-cluster bootstrap interval; the denominator may be NEGATIVE (the landed ratio_ci requires a positive one); a draw with a zero denominator is NaN."""
    d = den_pl.sum()
    point = float(num_pl.sum() / d) if d != 0 else float("nan")
    bd, bn = den_pl[draws].sum(axis=1), num_pl[draws].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(bd != 0, bn / bd, np.nan)
    ok = ~np.isnan(v)
    if not ok.any():
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "n_nan": int(len(v))}
    return {"point": point, "lo": float(np.percentile(v[ok], 2.5)), "hi": float(np.percentile(v[ok], 97.5)), "n_nan": int((~ok).sum())}


def attribution_stats(S, lab: np.ndarray, pk_p: np.ndarray, pk_e: np.ndarray) -> dict:
    """prereg s4 (D) for one seed: sequential contributions, the Shapley sensitivity, shares of the loss, the dominant term."""
    P = load_pickrule()
    idx = np.arange(len(pk_p))
    A, B = lab[idx, pk_p], lab[idx, pk_e]
    contrib, total = attribute(A, B)
    phi = shapley(A, B)
    sc = lambda ci: P.scale_ci(ci, 100.0)                                                           # noqa: E731
    seq = {t: sc(S.mean(contrib[:, k])) for k, t in enumerate(TERMS)}
    shp = {t: sc(S.mean(phi[:, k])) for k, t in enumerate(TERMS)}
    tot = sc(S.mean(total))
    den = S.pl(total)
    shares = {t: signed_ratio_ci(S.pl(contrib[:, k]), den, S.draws) for k, t in enumerate(TERMS)}
    pts = [seq[t]["point"] for t in TERMS]
    dom = dominant_term(pts, tot["point"])
    # bootstrap frequency of each term being the largest share of the loss (descriptive)
    Cpl = np.stack([S.pl(contrib[:, k]) for k in range(5)], axis=1)
    sums = Cpl[S.draws].sum(axis=1)                                                                  # [B, 5]
    tot_b = sums.sum(axis=1)
    amin = np.argmin(sums, axis=1)
    freq = {t: float(np.mean((amin == k) & (tot_b < 0))) for k, t in enumerate(TERMS)}
    freq["NO-LOSS"] = float(np.mean(tot_b >= 0))
    return {"order": list(TERMS), "sequential_x100": seq, "shapley_x100": shp, "total_x100": tot, "share_of_loss": shares, "dominant_term": dom, "dominant_bootstrap_frequency": freq,
            "max_abs_seq_sum_minus_total_per_set": float(np.abs(contrib.sum(axis=1) - total).max()), "max_abs_shapley_sum_minus_total_per_set": float(np.abs(phi.sum(axis=1) - total).max()),
            "mean_total_x100_float64": float(100.0 * total.mean()), "n_sets": int(len(total))}


HEADS = (("NC", NC, "deficit"), ("DAC", DAC, "deficit"), ("EP", EP, "positive"), ("TTC", TTC, "deficit"))     # prereg s4 (R): the primary form per head


def reliability_stats(S, lab: np.ndarray, zE: np.ndarray, z6: np.ndarray, pk_p: np.ndarray, pk_e: np.ndarray) -> dict:
    """prereg s4 (R): observed / predicted mass at the composite's picks and at P2's picks, from the composite's rows; deficit form sum(1-y)/sum(1-sigma) and positive form sum(y)/sum(sigma).
    The primary form is the deficit (risk) form for NC, DAC, TTC and the positive form for EP; the paired difference is of the primary form (HEADS)."""
    P = load_pickrule()
    out = {}
    for name, col, _ in HEADS:
        y = lab[..., col].astype(np.float64)
        sg = P.sigmoid64(zE[..., col])
        row = {"primary": "positive" if name == "EP" else "deficit"}
        pair = {}
        for pname, pk in (("composite_picks", pk_e), ("P2_picks", pk_p)):
            ye, se = P.at(y, pk), P.at(sg, pk)
            row[pname] = {"positive_form_sum_y_over_sum_sigma": S.ratio(ye, se), "deficit_form_sum_1my_over_sum_1msigma": S.ratio(1.0 - ye, 1.0 - se),
                          "observed_mass_positive": float(ye.sum()), "predicted_mass_positive": float(se.sum()), "observed_mass_deficit": float((1.0 - ye).sum()), "predicted_mass_deficit": float((1.0 - se).sum())}
            pair[pname] = (ye, se)
        (ye_c, se_c), (ye_p, se_p) = pair["composite_picks"], pair["P2_picks"]
        if name == "EP":
            row["paired_composite_minus_P2_primary"] = S.ratio_diff(ye_c, se_c, ye_p, se_p)
            y0 = P.at(P.sigmoid64(z6[..., col]), pk_p)
            row["P2_own_row_at_P2_picks"] = {"positive_form_sum_y_over_sum_sigma": S.ratio(ye_p, y0), "deficit_form_sum_1my_over_sum_1msigma": S.ratio(1.0 - ye_p, 1.0 - y0)}
        else:
            row["paired_composite_minus_P2_primary"] = S.ratio_diff(1.0 - ye_c, 1.0 - se_c, 1.0 - ye_p, 1.0 - se_p)
        out[name] = row
    return out


def analyze_seed(arr: dict, ep_col: np.ndarray, S) -> dict:
    """Everything of prereg s4 and s5 for ONE composite (the EP column `ep_col` [N, 64]); no decision is taken here."""
    P = load_pickrule()
    z6, lab = arr["z6"], arr["lab"]
    zE = z6.copy()
    zE[..., EP] = ep_col
    pk_p, pk_e = P.rule_pick(z6), P.rule_pick(zE)
    sig = P.sigmoid64(zE[..., EP])
    y = lab[..., EP].astype(np.float64)
    t = pair_terms(sig, y, pk_p, pk_e)
    g = full_gain_stats(S, t)
    sig0 = P.sigmoid64(z6[..., EP])
    g["P2_own_row_pred_gain"] = S.mean(P.at(sig0, pk_e) - P.at(sig0, pk_p))
    return {"n_sets": int(len(pk_p)), "n_changed": int((pk_p != pk_e).sum()), "G": g, "scale": {"slope": scale_slope(S, sig, y, safe_mask(lab))},
            "attribution": attribution_stats(S, lab, pk_p, pk_e), "reliability": reliability_stats(S, lab, zE, z6, pk_p, pk_e)}


def decide(per_seed: dict) -> dict:
    """prereg s5 / s6 on the per-seed analyses {seed: analyze_seed(...)}."""
    seeds = sorted(per_seed)
    on = scale_gate([per_seed[s]["scale"]["slope"] for s in seeds])
    form = deciding_form(on)
    cls = {"DELTA": [stat_class(per_seed[s]["G"]["delta"]) for s in seeds], "MASS": [stat_class(per_seed[s]["G"]["D_M"]) for s in seeds]}
    outcomes = {f: outcome_class(cls[f]) for f in cls}
    other = "MASS" if form == "DELTA" else "DELTA"
    doms = [per_seed[s]["attribution"]["dominant_term"] for s in seeds]
    return {"seeds": seeds, "scale_on": [scale_on(per_seed[s]["scale"]["slope"]) for s in seeds], "scale_gate_on_scale": on, "deciding_form": form, "classes": cls, "outcomes": outcomes,
            "outcome": outcomes[form], "other_form": other, "other_form_outcome": outcomes[other], "form_sensitive": form_sensitive(outcomes[form], outcomes[other]),
            "dominant_term_by_seed": doms, "dominant_term": dominant_agree(doms)}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# planted inputs (prereg s7 C-REACH, C-POWER)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plant_sigma(sig: np.ndarray, pk_p: np.ndarray, pk_e: np.ndarray, s: float) -> np.ndarray:
    """+s in the predicted sigma at the composite's pick of every CHANGED set (probability space, not clipped)."""
    out = sig.copy()
    ch = np.nonzero(pk_p != pk_e)[0]
    out[ch, pk_e[ch]] += s
    return out


def plant_label(y: np.ndarray, pk_p: np.ndarray, pk_e: np.ndarray, d: float) -> np.ndarray:
    """-d in the EP label at the composite's pick of every CHANGED set."""
    out = y.copy()
    ch = np.nonzero(pk_p != pk_e)[0]
    out[ch, pk_e[ch]] -= d
    return out


def reach_check(S, sig: np.ndarray, y: np.ndarray, pk_p: np.ndarray, pk_e: np.ndarray, d: float = 0.07, s: float = 0.05) -> dict:
    """prereg s7 C-REACH on one bank: the planted label / sigma shifts move Delta by exactly +d * n_changed / N (all sets) and +d (changed sets), and leave the other gain unchanged."""
    N = len(pk_p)
    nch = int((pk_p != pk_e).sum())
    t0 = pair_terms(sig, y, pk_p, pk_e)
    d0 = float(core_stats(S, t0)["delta"]["point"])
    c0 = float(full_gain_stats(S, t0)["changed_only"]["delta"]["point"]) if nch > 0 else float("nan")
    t1 = pair_terms(sig, plant_label(y, pk_p, pk_e, d), pk_p, pk_e)
    f1 = full_gain_stats(S, t1)
    t2 = pair_terms(plant_sigma(sig, pk_p, pk_e, s), y, pk_p, pk_e)
    f2 = full_gain_stats(S, t2)
    frac = nch / N
    res = {"n_changed": nch, "N": N, "planted_label_d": d, "planted_sigma_s": s,
           "label_delta_move": float(f1["delta"]["point"]) - d0, "label_delta_move_expected": d * frac,
           "label_changed_only_delta_move": (float(f1["changed_only"]["delta"]["point"]) - c0) if nch > 0 else float("nan"), "label_changed_only_expected": d,
           "label_pred_gain_unchanged": abs(float(f1["pred_gain"]["point"]) - float(full_gain_stats(S, t0)["pred_gain"]["point"])) < 1e-12,
           "sigma_delta_move": float(f2["delta"]["point"]) - d0, "sigma_delta_move_expected": s * frac,
           "sigma_real_gain_unchanged": abs(float(f2["real_gain"]["point"]) - float(full_gain_stats(S, t0)["real_gain"]["point"])) < 1e-12}
    ok = (abs(res["label_delta_move"] - res["label_delta_move_expected"]) < TOL_REACH and abs(res["sigma_delta_move"] - res["sigma_delta_move_expected"]) < TOL_REACH
          and res["label_pred_gain_unchanged"] and res["sigma_real_gain_unchanged"])
    if nch > 0:
        ok = ok and abs(res["label_changed_only_delta_move"] - d) < TOL_REACH
    res["pass"] = bool(ok)
    return res


def neutral_labels(sig: np.ndarray, y: np.ndarray, pk_p: np.ndarray, pk_e: np.ndarray) -> np.ndarray:
    """Amendment 1: the label field equals the predicted sigma at the two picks of every set (so g = ghat per set and M_e = M_p = 1) and the real EP label elsewhere."""
    out = y.copy()
    idx = np.arange(len(pk_p))
    out[idx, pk_p] = sig[idx, pk_p]
    out[idx, pk_e] = sig[idx, pk_e]
    return out


def power_check(S, sig: np.ndarray, y: np.ndarray, pk_p: np.ndarray, pk_e: np.ndarray, shift: float = PLANT) -> dict:
    """Amendment 1 (C-POWER, label-independent) for ONE seed: on the NEUTRAL bank (unplanted Delta and D_M are exactly 0), for EACH form, +shift in sigma at the composite's picks of the changed sets must
    read OVER, -shift must read UNDER, and 0 must read WITHIN with an interval exactly [0, 0]."""
    yn = neutral_labels(sig, y, pk_p, pk_e)
    out, ok = {}, True
    for form, key in (("DELTA", "delta"), ("MASS", "D_M")):
        up = core_stats(S, pair_terms(plant_sigma(sig, pk_p, pk_e, +shift), yn, pk_p, pk_e))[key]
        dn = core_stats(S, pair_terms(plant_sigma(sig, pk_p, pk_e, -shift), yn, pk_p, pk_e))[key]
        zero = core_stats(S, pair_terms(plant_sigma(sig, pk_p, pk_e, 0.0), yn, pk_p, pk_e))[key]
        row = {"plus": {"ci": up, "class": stat_class(up)}, "minus": {"ci": dn, "class": stat_class(dn)}, "zero": {"ci": zero, "class": stat_class(zero)}}
        row["zero_is_exactly_0_0"] = bool(float(zero["point"]) == 0.0 and float(zero["lo"]) == 0.0 and float(zero["hi"]) == 0.0)
        row["pass"] = bool(row["plus"]["class"] == "OVER" and row["minus"]["class"] == "UNDER" and row["zero"]["class"] == "WITHIN" and row["zero_is_exactly_0_0"])
        ok = ok and row["pass"]
        out[form] = row
    out["pass"] = bool(ok)
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the null: picks shuffled within token (prereg s7 C-NULL / E-NULL)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def shuffled_pair(pk_p: np.ndarray, pk_e: np.ndarray, rep: int, n_prop: int = 64) -> tuple:
    """One permutation of the 64 proposal indices per set (inside the set, hence inside its token); BOTH picks go through the SAME permutation, so a_i = b_i is preserved."""
    N = len(pk_p)
    rng = np.random.default_rng(NULL_SEED_BASE + NULL_SALT * rep)
    perm = np.argsort(rng.random((N, n_prop)), axis=1)
    idx = np.arange(N)
    return perm[idx, pk_p], perm[idx, pk_e]


def run_null(arr: dict, ep_cols: dict, S, form: str, reps_c: int = NULL_C_REPS, reps_e: int = NULL_E_REPS, real_points: dict | None = None) -> dict:
    """prereg s7: C-NULL (reps 0..reps_c-1, authority) and E-NULL (the following reps_e, descriptive).  The deciding `form` is the one the REAL data's scale gate selected."""
    P = load_pickrule()
    z6, lab = arr["z6"], arr["lab"]
    pk_p = P.rule_pick(z6)
    y = lab[..., EP].astype(np.float64)
    seeds = sorted(ep_cols)
    sig, pk_e = {}, {}
    for s in seeds:
        zE = z6.copy()
        zE[..., EP] = ep_cols[s]
        sig[s] = P.sigmoid64(zE[..., EP])
        pk_e[s] = P.rule_pick(zE)
    key = {"DELTA": "delta", "MASS": "D_M"}
    reps = []
    for rep in range(reps_c + reps_e):
        row = {"rep": rep, "perm_seed": NULL_SEED_BASE + NULL_SALT * rep, "role": "C-NULL" if rep < reps_c else "E-NULL", "seeds": {}}
        cls = {"DELTA": [], "MASS": []}
        for s in seeds:
            a, b = shuffled_pair(pk_p, pk_e[s], rep)
            cs = core_stats(S, pair_terms(sig[s], y, a, b))
            row["seeds"][str(s)] = {"delta": cs["delta"], "D_M": cs["D_M"], "n_changed": int((a != b).sum())}
            for f in cls:
                cls[f].append(stat_class(cs[key[f]]))
        row["classes"] = cls
        row["outcomes"] = {f: outcome_class(cls[f]) for f in cls}
        row["fires"] = null_fires(row["outcomes"][form])
        row["fires_other_form"] = null_fires(row["outcomes"]["MASS" if form == "DELTA" else "DELTA"])
        reps.append(row)
    c_rows, e_rows = reps[:reps_c], reps[reps_c:]
    fires_c = [r["fires"] for r in c_rows]
    out = {"form": form, "c_null": {"replicates": c_rows, "fires": int(sum(fires_c)), "of": reps_c, "void": null_void(fires_c)}}
    if e_rows:
        alpha = float(np.mean([r["fires"] for r in e_rows]))
        e = {"n": len(e_rows), "empirical_any_fire_rate": alpha, "p_at_least_2_of_5_fire_at_that_rate": float(1 - (1 - alpha) ** 5 - 5 * alpha * (1 - alpha) ** 4), "per_seed": {}}
        for s in seeds:
            pts = np.array([r["seeds"][str(s)][key[form]]["point"] for r in e_rows])
            q = [float(v) for v in np.quantile(pts, [0.0, 0.025, 0.25, 0.5, 0.75, 0.975, 1.0])]
            ent = {"null_points_quantiles_min_2.5_25_50_75_97.5_max": q, "null_points_mean": float(pts.mean()), "null_points_sd": float(pts.std(ddof=1))}
            if real_points is not None and s in real_points:
                rp = float(real_points[s])
                ent["real_point"] = rp
                ent["empirical_two_sided_p_with_plus_one"] = float((1 + int((np.abs(pts) >= abs(rp)).sum())) / (1 + len(pts)))
                ent["real_point_in_null_sd_units"] = float(rp / pts.std(ddof=1)) if pts.std(ddof=1) > 0 else float("nan")
            e["per_seed"][str(s)] = ent
        e["other_form_any_fire_rate"] = float(np.mean([r["fires_other_form"] for r in e_rows]))
        out["e_null"] = e
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# synthetic planted bank (tests and the gate's planted checks; no real data)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def synthetic_bank(N: int = 600, n_logs: int = 12, seed: int = 3, frac_changed: float = 0.6) -> tuple:
    """P2 picks proposal 0 in every set; the composite picks proposal 1 on `frac_changed` of the sets and proposal 0 elsewhere.  -> (arr {z6, lab}, ep_col [N,64], log_idx [N], changed [N] bool)."""
    rng = np.random.default_rng(seed)
    z6 = (rng.normal(0.0, 1.0, (N, 64, 6)) - 3.0).astype(np.float32)
    z6[:, 0, :] = 4.0
    z6[:, 0, EP] = 2.0
    z6[:, 1, :] = 4.0
    z6[:, 1, EP] = -2.0
    lab = np.stack([rng.choice([0.0, 0.5, 1.0], size=(N, 64), p=[0.07, 0.05, 0.88]), rng.choice([0.0, 1.0], size=(N, 64), p=[0.05, 0.95]), rng.uniform(0.0, 1.0, (N, 64)),
                    rng.choice([0.0, 1.0], size=(N, 64), p=[0.1, 0.9]), rng.choice([0.0, 1.0], size=(N, 64), p=[0.1, 0.9]), np.ones((N, 64))], axis=-1).astype(np.float32)
    changed = np.zeros(N, bool)
    changed[rng.permutation(N)[:int(round(frac_changed * N))]] = True
    ep = z6[..., EP].copy()
    ep[changed, 1] = 8.0
    log_idx = rng.integers(0, n_logs, size=N)
    return {"z6": z6, "lab": lab}, ep, log_idx, changed


def planted_slope_bank(kind: str, N: int = 400, n_logs: int = 10, seed: int = 5) -> tuple:
    """Planted scale-gate banks: sigma = x (random in (0.05, 0.95)), label y = k * x + tiny noise (k by kind) over all-safe sets.  -> (sig, y, safe, log_idx, n_logs)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.05, 0.95, (N, 64))
    k = {"exact": 1.0, "ok_1.05": 1.05, "half": 0.5, "high_1.2": 1.2}[kind]
    y = k * x + rng.normal(0.0, 1e-4, x.shape)
    return x, y, np.ones((N, 64), bool), rng.integers(0, n_logs, size=N), n_logs
