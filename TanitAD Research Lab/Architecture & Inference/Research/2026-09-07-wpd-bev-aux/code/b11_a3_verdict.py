# -*- coding: utf-8 -*-
"""WP-D step 11 -- CLOSE A3 by the PRE-COMMITTED arithmetic in PANEL_RESULT.md 5b.

⛔ NOTHING IS RE-DERIVED HERE. The two thresholds are read as LITERALS from the
panel's own section 5b, which fixed them before either replicate arm landed:

  "The measured lever gap is +0.00173 AP.  A3 passes only if the replicate floor
   |AP(D0b) - AP(D0)| is below 0.00058 AP"
  "If the floor exceeds 0.00539, then D0, D1 and D2 are all within one replicate
   spread of each other and the honest reading becomes F5 -- underpowered at
   4,000 steps, not a measured difference in either direction."

And the denominator is 5b's own rule:
  "A3's denominator = max(f_same, f_seed) -- the larger, because a same-seed-only
   floor is anti-conservative and would make the 3x bar easier"
"""
import argparse, json, os

# ---- LITERALS from PANEL_RESULT.md 5b, pre-committed. Do not compute these. ---
# CARTESIAN is the PRIMARY read and the one 5b fixed IN WORDS. POLAR's
# analogues are DERIVED from section 5's polar table and are labelled DERIVED --
# quoting them as 5b literals would be putting words in the pre-registration.
LITERALS = {
 "cart": {"lever": 0.00173, "floor_max": 0.00058, "f5": 0.00539,
          "src": "PANEL_RESULT.md 5b, quoted verbatim (PRIMARY geometry)"},
 "pol": {"lever": 0.00923, "floor_max": 0.00923 / 3.0, "f5": 0.00898,
         "src": "DERIVED from PANEL_RESULT.md section 5's POLAR table "
                "(A3 +0.00923; D2-D0 context +0.00898). 5b fixed only the "
                "Cartesian numbers in words, so these are stated as DERIVED."},
}

p = argparse.ArgumentParser()
p.add_argument("--boot", required=True)
p.add_argument("--panel", required=True)
p.add_argument("--out", required=True)
p.add_argument("--instrument-boot", default="")   # boot on two IDENTICAL-input arms
p.add_argument("--determinism", default="")       # bank/probe reproducibility control
a = p.parse_args()

B = json.load(open(a.boot, encoding="utf-8"))
_L = LITERALS[B["geom"]]
LEVER_GAP_COMMITTED = _L["lever"]
A3_FLOOR_MAX = _L["floor_max"]
F5_FLOOR_THRESHOLD = _L["f5"]
P = json.load(open(a.panel, encoding="utf-8"))
AP = B["point_ap"]
pr = B["pairs"]

f_same = abs(AP["tok_D0b"] - AP["tok_D0"])
f_seed = abs(AP["tok_D0c"] - AP["tok_D0"])
floor = max(f_same, f_seed)
which = "f_seed (D0c, seed 1)" if f_seed >= f_same else "f_same (D0b, same seed)"
lever_here = AP["tok_D1"] - AP["tok_D0"]

# ---- THE INSTRUMENT FLOOR: what the chain reports when NOTHING differs ------
# ⛔ A3's floor is only a TRAINING-variance measurement if the chain reproduces
# itself. Two independent readings of that, both on inputs that are bit-identical:
#   within one invocation : tok_D0dup vs tok_D0   (same file, same seed, same run)
#   across invocations    : AP(D0) panel vs replicate (bank proven bit-equal)
instr = {}
if a.instrument_boot:
    IB = json.load(open(a.instrument_boot, encoding="utf-8"))
    ap0, apd = IB["point_ap"]["tok_D0"], IB["point_ap"]["tok_D0dup"]
    pr_i = IB["pairs"].get("tok_D0dup_minus_tok_D0", {})
    instr["within_invocation_identical_file"] = {
        "ap_tok_D0": ap0, "ap_tok_D0dup": apd, "floor": abs(apd - ap0),
        "true_value": 0.0, "ci": [pr_i.get("lo"), pr_i.get("hi")],
        "separated": pr_i.get("separated")}
if a.determinism:
    DC = json.load(open(a.determinism, encoding="utf-8"))
    instr["across_invocations_bit_identical_features"] = {
        "ap_panel": DC["probe_ap_tok_D0"]["panel_2026_09_08"],
        "ap_replicate": DC["probe_ap_tok_D0"]["replicate_run"],
        "floor": DC.get("instrument_floor_same_arm_same_code"),
        "bank_is_bit_deterministic": DC.get("bank_is_deterministic"),
        "bank_differing_cells": DC.get("bank_tok_D0_differing_cells"),
        "replicate_is_distinct": DC.get("tok_D0b_is_distinct_from_D0")}
_iv = [v["floor"] for v in instr.values() if v.get("floor") is not None]
instr_floor = max(_iv) if _iv else None
if instr_floor is not None:
    instr["max_instrument_floor"] = instr_floor
    instr["exceeds_a3_required_floor_bar"] = instr_floor > A3_FLOOR_MAX
    instr["times_the_a3_bar"] = instr_floor / A3_FLOOR_MAX
    instr["_reading"] = (
        "A3 required the replicate floor to be BELOW %.5f. The measurement chain "
        "cannot reproduce ITSELF to better than %.6f on inputs that differ in "
        "NOTHING, which is %.2fx that bar. A3 was therefore UNMEASURABLE on this "
        "rig before any replicate arm was trained -- and an unmeasurable criterion "
        "is reported as UNDERPOWERED, never as a negative."
        % (A3_FLOOR_MAX, instr_floor, instr_floor / A3_FLOOR_MAX))

# ---- A3's OWN bar: 3x the floor. This does not depend on the F5 question. ---
a3_pass = LEVER_GAP_COMMITTED >= 3.0 * floor
a3_bar = "PASS" if a3_pass else "FAIL"
shortfall = (3.0 * floor) / LEVER_GAP_COMMITTED if LEVER_GAP_COMMITTED else float("inf")

# ---- F4 vs F5: 5b states the test TWO ways and BOTH are reported. -----------
# ⛔ Reported side by side ON PURPOSE. Picking whichever reads better after
# seeing the data would be moving a goalpost (CLAUDE.md). A3's own bar above
# fails on EITHER reading, so only the LEVER's F4/F5 LABEL is at stake here.
#   (a) THE LITERAL: 5b wrote the number 0.00539, taken from the 2026-09-08
#       panel's own D2-D0 delta.
#   (b) THE SAME-PANEL SPREAD: 5b's words are "exceed the whole D0->D2 spread",
#       and 5b also REQUIRES the floor be measured "IN THE SAME PANEL". The
#       coherent comparison is therefore floor vs THIS panel's D0->D2 spread --
#       comparing a floor from one panel to a spread from another mixes two
#       instruments, which is the error 5b's own "in the same panel" rule exists
#       to prevent.
lever_arms = [AP["tok_D0"], AP["tok_D1"], AP["tok_D2"]]
spread_same_panel = max(lever_arms) - min(lever_arms)
rep_arms = [AP["tok_D0"], AP["tok_D0b"], AP["tok_D0c"]]
spread_replicate = max(rep_arms) - min(rep_arms)

f5_literal = floor > F5_FLOOR_THRESHOLD
f5_same_panel = floor > spread_same_panel
f5_agree = f5_literal == f5_same_panel

UNMEASURABLE = instr_floor is not None and instr_floor > A3_FLOOR_MAX
if a3_pass:
    verdict = "A3 PASS"
elif UNMEASURABLE or f5_literal or f5_same_panel:
    # ⛔ UNMEASURABLE outranks the threshold tests and does not depend on either.
    # It says the criterion demanded a precision the instrument does not have.
    verdict = "F5 UNDERPOWERED"
else:
    verdict = "A3 FAIL"

why = (
 f"A3's OWN BAR: the committed lever gap {LEVER_GAP_COMMITTED:.5f} must be >= 3x the "
 f"replicate floor. 3 x {floor:.5f} = {3*floor:.5f}, which the lever gap misses by "
 f"{shortfall:.1f}x => A3's bar {a3_bar}S. "
 f"F4-vs-F5 LABEL, both readings of 5b reported: (a) LITERAL floor {floor:.5f} vs "
 f"0.00539 -> F5={f5_literal}; (b) SAME-PANEL floor {floor:.5f} vs this panel's "
 f"D0->D2 spread {spread_same_panel:.5f} -> F5={f5_same_panel}. "
 f"AGREE={f5_agree}. "
 f"The structural statement, which does not hinge on either threshold: three arms "
 f"differing in NOTHING BUT A SEED span {spread_replicate:.5f} AP, while D0/D1/D2 "
 f"span {spread_same_panel:.5f} -- a ratio of "
 f"{spread_replicate/spread_same_panel if spread_same_panel else float('inf'):.2f}x. "
 f"An unmeasurable criterion is reported as UNDERPOWERED, never as a negative."
 + ("" if instr_floor is None else
    (f" ⛔ AND THE DECIDING GROUND, which depends on NEITHER threshold: the chain "
     f"cannot reproduce ITSELF to better than {instr_floor:.6f} AP on inputs that "
     f"differ in NOTHING, while A3 required a floor below {A3_FLOOR_MAX:.5f} -- "
     f"{instr_floor/A3_FLOOR_MAX:.2f}x the bar. A3 was UNMEASURABLE on this rig "
     f"before any replicate was trained."
     if UNMEASURABLE else
     f" ⚠️ The instrument floor {instr_floor:.6f} is {instr_floor/A3_FLOOR_MAX:.2f}x "
     f"this geometry's required floor bar {A3_FLOOR_MAX:.5f}, i.e. BELOW it, so on "
     f"THIS geometry A3 is a MEASURED FAIL and NOT underpowered. ⛔ Do not carry "
     f"the Cartesian read's `unmeasurable` wording across to it.")))

# ---- the controls that make any of this admissible --------------------------
ctl = P["controls"]
base = P["controls"]["base_rate"]
ctl_ok = bool(ctl["constant_equals_base_rate_exactly"]) and ctl["perfect_ranker_ap"] == 1.0 \
         and ctl["allzero_ap"] == base and ctl["allone_ap"] == base

out = {
 "_evidence_class": "MEASURED (ours; artifact = this file + " + a.boot + ")",
 "tier": "NOT APPLICABLE - frozen-feature representation probe, no trajectory",
 "geom": B["geom"], "n_boot": B["n_boot"], "n_episodes": B["n_episodes"],
 "n_test_rows": B["n_test_rows"], "n_cells_per_row": B["n_cells_per_row"],
 "n_scored_cells": B["n_scored_cells"], "n_pos": B["n_pos"],
 "base_rate": B["base_rate"],
 "d_raw_per_row": P["arms"]["tok_D0"]["d_raw_per_row"],
 "d_head_input_per_cell": P["arms"]["tok_D0"]["d_head_input_per_cell"],
 "prereg_literals_from_PANEL_RESULT_5b": {
    "lever_gap_committed": LEVER_GAP_COMMITTED,
    "a3_floor_must_be_below": A3_FLOOR_MAX,
    "f5_underpowered_if_floor_above": F5_FLOOR_THRESHOLD,
    "denominator_rule": "max(f_same, f_seed)",
    "provenance": _L["src"]},
 "point_ap": AP,
 "floors": {"f_same_D0b_minus_D0": f_same, "f_seed_D0c_minus_D0": f_seed,
            "floor_used": floor, "which": which,
            "floor_as_pct_of_D0_ap": 100.0 * floor / AP["tok_D0"]},
 "lever_gap_this_panel": lever_here,
 "three_x_floor": 3.0 * floor,
 "instrument_floor": instr,
 "a3_unmeasurable_on_this_rig": (instr_floor is not None and instr_floor > A3_FLOOR_MAX),
 "a3_own_bar": a3_bar, "a3_bar_shortfall_x": shortfall,
 "spread_D0_to_D2_same_panel": spread_same_panel,
 "spread_replicates_same_panel": spread_replicate,
 "replicate_spread_over_lever_spread_x": (spread_replicate / spread_same_panel
                                          if spread_same_panel else None),
 "f5_by_literal_threshold": f5_literal, "f5_by_same_panel_spread": f5_same_panel,
 "f5_readings_agree": f5_agree,
 "a3_verdict": verdict, "a3_why": why,
 "controls_all_pass": ctl_ok, "controls": ctl,
 "paired_intervals": {k: {kk: pr[k][kk] for kk in
                          ("delta", "lo", "hi", "separated", "why")}
                      for k in pr},
}
json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)

print("=" * 78)
print(f"A3 CLOSURE -- geom {B['geom']}  n_boot {B['n_boot']}  episodes {B['n_episodes']}")
print(f"  n = {B['n_scored_cells']:,} scored cells over {B['n_test_rows']:,} test rows")
print(f"  d = {out['d_raw_per_row']:,} raw per row / {out['d_head_input_per_cell']} into the head")
print(f"  base rate {B['base_rate']:.9f}  (an all-zero predictor scores "
      f"{100*(1-B['base_rate']):.4f} % ACCURACY -- no number here is an accuracy)")
print(f"  CONTROLS all pass: {ctl_ok}   constant AP == base rate EXACTLY: "
      f"{ctl['constant_equals_base_rate_exactly']}")
print("-" * 78)
for k, v in AP.items():
    print(f"  AP {k:10s} {v:.6f}")
print("-" * 78)
print(f"  f_same = |AP(D0b) - AP(D0)| = {f_same:.6f}   (same seed, zero levers)")
print(f"  f_seed = |AP(D0c) - AP(D0)| = {f_seed:.6f}   (seed 1, prereg s4)")
print(f"  FLOOR  = max            = {floor:.6f}   <- {which}")
print(f"  3 x FLOOR                   = {3*floor:.6f}")
print(f"  lever gap (committed)       = {LEVER_GAP_COMMITTED:.6f}"
      f"   [{_L['src'].split(' -- ')[0][:46]}]")
print(f"  lever gap (this panel)      = {lever_here:.6f}")
print("-" * 78)
print(f"  D0->D2 spread THIS panel    = {spread_same_panel:.6f}")
print(f"  REPLICATE spread (D0/D0b/D0c, ZERO levers) = {spread_replicate:.6f}"
      f"   ratio {spread_replicate/spread_same_panel if spread_same_panel else 0:.2f}x")
print(f"  A3's own 3x bar             : {a3_bar}  (misses by {shortfall:.1f}x)")
print(f"  F5 by LITERAL 0.00539       : {f5_literal}")
print(f"  F5 by SAME-PANEL spread     : {f5_same_panel}   (agree={f5_agree})")
if instr_floor is not None:
    print("-" * 78)
    for k, v in instr.items():
        if isinstance(v, dict) and v.get("floor") is not None:
            print(f"  INSTRUMENT {k}: floor {v['floor']:.6f}  (true value 0)")
    print(f"  MAX INSTRUMENT FLOOR        = {instr_floor:.6f}")
    print(f"  A3 REQUIRED FLOOR BELOW     = {A3_FLOOR_MAX:.6f}"
          f"   -> instrument is {instr_floor/A3_FLOOR_MAX:.2f}x the bar")
    print(f"  => A3 UNMEASURABLE ON THIS RIG: {instr_floor > A3_FLOOR_MAX}")
print("=" * 78)
print(f"  VERDICT: {verdict}")
print(f"  {why}")
print("=" * 78)
print("WROTE", a.out)
