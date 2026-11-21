#!/usr/bin/env python3
"""RED mutations for EVERY branch of the L2 outcome mapping (draft section 4).  `scenarios(mod)` is a scenario suite with the expected verdict of each; the ORIGINAL module must pass all of
them, and EVERY textual mutation of `outcome.py` below must fail at least one (a mapping branch no scenario can distinguish from its mutant is an unguarded branch).
Run:  python test_outcome.py   (exit 0 = all mutations were caught)   or   pytest test_outcome.py"""
from __future__ import annotations

import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "outcome.py")

NULL_OK = {0: [0.62, 0.66, 0.70, 0.68, 0.64], 1: [0.63, 0.65, 0.69, 0.66, 0.61]}      # max 0.70, none exceeds P2 0.685 + 0.05 = 0.735
P2 = 0.685


def R(a1=0.85, lo=0.78, a4=0.02, a5=0.01, infan=0.89, infan_h0=0.887, infan_p2=0.95):
    """infan_p2 defaults ABOVE every in-fan value used here: P2's in-fan is reported and NEVER gated, so a scenario that gated on it would mis-map"""
    return {"a1_point": a1, "a1_lo": lo, "a4_point": a4, "a5_point": a5, "infan": infan, "infan_h0": infan_h0, "infan_p2": infan_p2}


def both(**kw):
    return {0: R(**kw), 1: R(**kw)}


def scenarios(m):
    """-> list of (name, expected, got)"""
    f = m.l2_outcome
    out = []

    def chk(name, expected, **kw):
        kw.setdefault("cfeat_pass", True)
        kw.setdefault("p2_a1", P2)
        kw.setdefault("null_points", NULL_OK)
        got = f(**kw)["outcome"]
        out.append((name, expected, got))

    chk("admit_clear", "L2-ADMIT", reads=both())
    chk("admit_a1_point_exactly_gate", "L2-ADMIT", reads=both(a1=0.80, lo=0.75))
    chk("admit_a4_exactly_gate", "L2-ADMIT", reads=both(a4=0.10))
    chk("admit_a5_exactly_gate", "L2-ADMIT", reads=both(a5=0.05))
    chk("admit_infan_just_inside_margin", "L2-ADMIT", reads=both(infan=0.7301, infan_h0=0.75))
    chk("admit_infan_far_below_p2_but_at_h0", "L2-ADMIT", reads=both(infan=0.60, infan_h0=0.60, infan_p2=0.89))        # P2's in-fan is REPORTED, never gated
    chk("admit_infan_above_h0", "L2-ADMIT", reads=both(infan=0.91, infan_h0=0.80))
    chk("admit_infan_exactly_at_margin", "L2-ADMIT", reads=both(infan=0.75 - 0.02, infan_h0=0.75))                       # the bar is >= (the same float expression the module evaluates)
    chk("partial_a1_point_below_gate", "L2-PARTIAL", reads=both(a1=0.78, lo=0.74))
    chk("partial_a1_point_just_below_gate", "L2-PARTIAL", reads=both(a1=0.7999, lo=0.74))
    chk("partial_a4_over", "L2-PARTIAL", reads=both(a4=0.11))
    chk("partial_a5_over", "L2-PARTIAL", reads=both(a5=0.06))
    chk("partial_infan_degraded", "L2-PARTIAL", reads=both(infan=0.84, infan_h0=0.887))
    chk("partial_infan_just_outside_margin", "L2-PARTIAL", reads=both(infan=0.7299, infan_h0=0.75))
    chk("partial_infan_degraded_on_seed1_only", "L2-PARTIAL", reads={0: R(), 1: R(infan=0.80, infan_h0=0.887)})
    chk("infan_h0_is_read_per_seed", "L2-ADMIT", reads={0: R(infan=0.80, infan_h0=0.82), 1: R(infan=0.90, infan_h0=0.90)})
    chk("admit_needs_both_seeds_a1_gate", "L2-PARTIAL", reads={0: R(), 1: R(a1=0.79, lo=0.74)})
    chk("admit_needs_both_seeds_a4", "L2-PARTIAL", reads={0: R(), 1: R(a4=0.2)})
    chk("flat_one_seed_lower_ci_below_null", "L2-FLAT", reads={0: R(), 1: R(lo=0.69)})
    chk("flat_lower_ci_equals_null_max", "L2-FLAT", reads=both(lo=0.70))
    chk("flat_point_high_but_lower_ci_not", "L2-FLAT", reads=both(a1=0.95, lo=0.60))
    chk("flat_other_seed", "L2-FLAT", reads={0: R(lo=0.5), 1: R()})
    # null pool: seed-0 replicates reach 0.72, seed-1's only 0.60; the strict default compares seed 1 with 0.72
    npool = {0: [0.62, 0.72, 0.60, 0.61, 0.63], 1: [0.55, 0.60, 0.58, 0.57, 0.59]}
    chk("null_pool_all_is_strict", "L2-FLAT", reads={0: R(lo=0.80), 1: R(lo=0.65)}, null_points=npool)
    chk("null_pool_matched_seed_reading", "L2-ADMIT", reads={0: R(lo=0.80), 1: R(lo=0.65)}, null_points=npool, null_pool="matched_seed")
    # VOID
    chk("void_cfeat_overrides_admit", "VOID", reads=both(), cfeat_pass=False)
    chk("void_two_replicates_exceed", "VOID", reads=both(), null_points={0: [0.74, 0.75, 0.60, 0.61, 0.63], 1: NULL_OK[1]})
    chk("void_two_replicates_exceed_on_seed1", "VOID", reads=both(), null_points={0: NULL_OK[0], 1: [0.60, 0.61, 0.62, 0.80, 0.90]})
    chk("not_void_one_replicate_exceeds", "L2-ADMIT", reads=both(lo=0.95, a1=0.97), null_points={0: [0.74, 0.60, 0.60, 0.61, 0.63], 1: NULL_OK[1]})
    chk("not_void_exceed_by_less_than_gate", "L2-ADMIT", reads=both(lo=0.80, a1=0.9), null_points={0: [0.72, 0.725, 0.60, 0.61, 0.63], 1: NULL_OK[1]})
    chk("void_scope_any_seed_one_per_seed_is_not_void", "L2-FLAT", reads=both(lo=0.5), null_points={0: [0.74, 0.60, 0.60, 0.61, 0.63], 1: [0.74, 0.60, 0.60, 0.61, 0.63]})
    chk("void_scope_pooled_one_per_seed_is_void", "VOID", reads=both(), null_points={0: [0.74, 0.60, 0.60, 0.61, 0.63], 1: [0.74, 0.60, 0.60, 0.61, 0.63]}, void_scope="pooled")
    return out


MUTATIONS = [
    ("A1 gate 0.80 -> 0.70", 'A1_GATE = 0.80', 'A1_GATE = 0.70'),
    ("A1 point >= gate -> >", 'r["a1_point"] >= A1_GATE', 'r["a1_point"] > A1_GATE'),
    ("A1 point gate dropped", 'r["a1_point"] >= A1_GATE', 'True'),
    ("A4 <= gate -> <", 'r["a4_point"] <= A4_GATE', 'r["a4_point"] < A4_GATE'),
    ("A4 gate dropped", 'r["a4_point"] <= A4_GATE', 'True'),
    ("A5 <= gate -> <", 'r["a5_point"] <= A5_GATE', 'r["a5_point"] < A5_GATE'),
    ("A5 gate dropped", 'r["a5_point"] <= A5_GATE', 'True'),
    ("in-fan margin 0.02 -> 0.2", 'INFAN_MARGIN = 0.02', 'INFAN_MARGIN = 0.2'),
    ("in-fan term dropped", 'r["infan"] >= r["infan_h0"] - INFAN_MARGIN', 'True'),
    ("in-fan compared with P2 instead of H0 (P2 is reported, never gated)", 'r["infan"] >= r["infan_h0"] - INFAN_MARGIN', 'r["infan"] >= r["infan_p2"] - INFAN_MARGIN'),
    ("in-fan >= -> >", 'r["infan"] >= r["infan_h0"] - INFAN_MARGIN', 'r["infan"] > r["infan_h0"] - INFAN_MARGIN'),
    ("in-fan margin sign flipped (H0 + margin)", 'r["infan"] >= r["infan_h0"] - INFAN_MARGIN', 'r["infan"] >= r["infan_h0"] + INFAN_MARGIN'),
    ("lower CI replaced by the point", 'r["a1_lo"] > null_max', 'r["a1_point"] > null_max'),
    ("lower CI > null max -> >=", 'r["a1_lo"] > null_max', 'r["a1_lo"] >= null_max'),
    ("FLAT: both seeds -> either seed (all -> any)", 'lower_all = all(', 'lower_all = any('),
    ("FLAT branch removed", 'if not lower_all:', 'if False:'),
    ("ADMIT: both seeds -> any seed", 'admit = all(', 'admit = any('),
    ("ADMIT label swapped to PARTIAL", '"outcome": "L2-ADMIT", "why": "both', '"outcome": "L2-PARTIAL", "why": "both'),
    ("PARTIAL label swapped to ADMIT", '"outcome": "L2-PARTIAL", "why": "lower', '"outcome": "L2-ADMIT", "why": "lower'),
    ("VOID: C-FEAT clause dropped", 'if not cfeat_pass:', 'if False:'),
    ("VOID: count 2 -> 3", 'VOID_COUNT = 2', 'VOID_COUNT = 3'),
    ("VOID: exceed 0.05 -> 0.0", 'VOID_EXCEED = 0.05', 'VOID_EXCEED = 0.0'),
    ("VOID: exceed 0.05 -> 0.5", 'VOID_EXCEED = 0.05', 'VOID_EXCEED = 0.5'),
    ("VOID branch removed entirely", 'if why is not None:\n        return {"outcome": "VOID"', 'if False:\n        return {"outcome": "VOID"'),
    ("VOID scope any_seed -> pooled", 'if void_scope == "pooled":', 'if True:'),
    ("null pool all -> matched seed", 'null_max = pooled_max if null_pool == "all" else max(null_points[s])', 'null_max = max(null_points[s])'),
    ("null pool max -> min", 'pooled_max = max(p for s in null_points for p in null_points[s])', 'pooled_max = min(p for s in null_points for p in null_points[s])'),
]


def load(src):
    m = types.ModuleType("outcome_mut")
    exec(compile(src, "outcome_mut", "exec"), m.__dict__)
    return m


def failures(m):
    return [(n, e, g) for n, e, g in scenarios(m) if e != g]


def run():
    src = open(SRC, encoding="utf-8").read().replace("\r\n", "\n")
    base = failures(load(src))
    print(f"original module: {len(scenarios(load(src)))} scenarios, {len(base)} failures", flush=True)
    for b in base:
        print("   ORIGINAL FAILS:", b)
    bad = []
    for name, old, new in MUTATIONS:
        if src.count(old) != 1:
            bad.append((name, f"mutation anchor occurs {src.count(old)}x, expected 1"))
            print(f"  [BAD ANCHOR] {name}: {src.count(old)} occurrences")
            continue
        fl = failures(load(src.replace(old, new)))
        print(f"  [{'RED (caught)' if fl else 'GREEN (NOT CAUGHT)'}] {name}: {len(fl)} scenario(s) fail" + (f"  e.g. {fl[0][0]}" if fl else ""))
        if not fl:
            bad.append((name, "no scenario distinguishes this mutant"))
    ok = not base and not bad
    print("ZZOUTCOME_MUTATIONS", "OK" if ok else "FAIL", f"{len(MUTATIONS)} mutations")
    return 0 if ok else 1


def test_original_passes_all_scenarios():
    assert failures(load(open(SRC, encoding="utf-8").read().replace("\r\n", "\n"))) == []


def test_every_mutation_is_caught():
    src = open(SRC, encoding="utf-8").read().replace("\r\n", "\n")
    for name, old, new in MUTATIONS:
        assert src.count(old) == 1, name
        assert failures(load(src.replace(old, new))), f"mutation not caught: {name}"


if __name__ == "__main__":
    sys.exit(run())
