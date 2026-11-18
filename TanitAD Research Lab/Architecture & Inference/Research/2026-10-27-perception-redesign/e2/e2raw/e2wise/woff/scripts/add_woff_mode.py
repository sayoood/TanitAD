"""add_woff_mode.py: add mode 'woff' (WIRE-OFF, ADDENDUM 4, 81d409d9) to build_e2wise_hold.py. Every replacement count-asserted == 1.
usage: python -I add_woff_mode.py <build_e2wise_hold.py>"""
import sys

P = sys.argv[1]
s = open(P, "rb").read().decode("utf-8")
assert "woff" not in s, "already has woff"
R = [
    ('assert MODE in ("plan", "enc", "plan0", "both"), MODE',
     'assert MODE in ("plan", "enc", "plan0", "both", "woff"), MODE'),
    ('NEWB = MODE in ("plan0", "both")', 'NEWB = MODE in ("plan0", "both", "woff")'),
    ('PLANLIKE = MODE in ("plan", "plan0")      # perception copied bit-exactly -> Addendum 1 check applies',
     'PLANLIKE = MODE in ("plan", "plan0")      # weight_interp planner subsets (provenance carries perception_untouched_all_bit_equal)\n'
     'PERC_SAME = MODE in ("plan", "plan0", "woff")   # perception copied bit-exactly -> Addendum 1 check applies'),
    ('TAG = {"plan": "wise", "enc": "wenc", "plan0": "wise", "both": "wboth"}[MODE]',
     'TAG = {"plan": "wise", "enc": "wenc", "plan0": "wise", "both": "wboth", "woff": "wise"}[MODE]'),
    ('CK = {"plan0": "model_wise_plan_a000.pt"}.get(MODE, "model_wise_" + MODE + "_a050.pt")',
     'CK = {"plan0": "model_wise_plan_a000.pt", "woff": "model_wireoff.pt"}.get(MODE, "model_wise_" + MODE + "_a050.pt")'),
    ('"plan0": "PLAN-RESET", "both": "WISE-BOTH"}[MODE]', '"plan0": "PLAN-RESET", "both": "WISE-BOTH", "woff": "WIRE-OFF"}[MODE]'),
    ('          "both": "S_plan + S_enc (20 planner prefixes + core.encoder)"}[MODE]',
     '          "both": "S_plan + S_enc (20 planner prefixes + core.encoder)",\n'
     '          "woff": "the 8 planner-wire tensors of K2DC (6 box_gates + map_chan_proj weight/bias), set to 0; K0 has none"}[MODE]'),
    ('              "both": ["e2wboth_stageA.sh", "e2wboth_stageB_launch.sh", "e2wboth_weights.sh"]}[MODE]',
     '              "both": ["e2wboth_stageA.sh", "e2wboth_stageB_launch.sh", "e2wboth_weights.sh"],\n'
     '              "woff": ["e2woff_stageA.sh", "e2woff_stageB_launch.sh", "e2woff_weights.sh", "wire_off.py"]}[MODE]'),
    ('                        "make_wboth_kit.py", "gen_auto_wboth.py"]}[MODE] + ["param_census.py", "build_e2wise_hold.py"]',
     '                        "make_wboth_kit.py", "gen_auto_wboth.py"],\n'
     '               "woff": ["e2woff_s6c_pull_score.sh", "wait_e2woffB_then_s6c.sh", "auto_woff.sh", "after_wboth_then_woff.sh",\n'
     '                        "make_woff_kit.py", "add_woff_mode.py"]}[MODE] + ["param_census.py", "build_e2wise_hold.py"]'),
    ('    prov[arm] = {k: pv.get(k) for k in ("alpha", "n_interpolated", "n_copied_in_S", "n_untouched")}',
     '    if MODE == "woff":\n'
     '        assert pv.get("n_zeroed") == (8 if arm == "K2DC" else 0) and pv.get("all_others_bit_equal") is True, (arm, pv.get("n_zeroed"))\n'
     '    prov[arm] = {k: pv.get(k) for k in ("alpha", "n_interpolated", "n_copied_in_S", "n_untouched", "n_zeroed", "n_untouched_bit_equal", "zeroed")}'),
    ('if MODE in ("plan0", "both") and both_up:', 'if MODE in ("plan0", "both", "woff") and both_up:'),
    ('"plan0": "PLANNER-RESET (alpha 0)", "both": "PLANNER + TRUNK"}[MODE]',
     '"plan0": "PLANNER-RESET (alpha 0)", "both": "PLANNER + TRUNK", "woff": "WIRE-OFF (planner wires zeroed)"}[MODE]'),
    ('       "both": " + `../../PREREG_E2WISE_2026-11-17_ADDENDUM2.md` (landed 8153ee7d) + ADDENDUM3 (order, landed 93688696)"}[MODE]',
     '       "both": " + `../../PREREG_E2WISE_2026-11-17_ADDENDUM2.md` (landed 8153ee7d) + ADDENDUM3 (order, landed 93688696)",\n'
     '       "woff": " + `../../PREREG_E2WISE_2026-11-17_ADDENDUM4.md` (landed 81d409d9)"}[MODE]'),
    ('             if MODE == "plan0" else', '             if MODE in ("plan0", "woff") else'),
    ('r"\\*\\*(WISE-[A-Z0-9]+|PLAN-RESET) VERDICT"', 'r"\\*\\*(WISE-[A-Z0-9]+|PLAN-RESET|WIRE-OFF) VERDICT"'),
]
for old, new in R:
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)
# the plan0 perception-check text names PLAN-RESET: make it the mode label
old = ("). PLAN-RESET \"\n             \"copies `core.encoder`, `_perception` and `core.agent_head` bit-exactly; (d) decides the lever.\\n\"")
assert s.count(old) == 1, s.count(old)
s = s.replace(old, "). \" + LABEL + \" \"\n             \"copies `core.encoder`, `_perception` and `core.agent_head` bit-exactly; (d) decides the lever.\\n\"")
# woff outcomes: insert before the WISE-BOTH branch of the outcome block (the `else:` that starts the both branch)
anchor = ('else:\n    if st["d"] == "PASS" and st["a"] == "PASS":\n        outcome = ("Addendum 2 committed outcome 1 (WISE-BOTH')
assert s.count(anchor) == 1, s.count(anchor)
WO = '''elif MODE == "woff":
    LROWS = ["LAT_cross_mae_m", "LAT_cross_abs_final_m", "LAT_heading_mae_deg", "LAT_yaw_rate_mae_radps_valid"]
    vs_tail, notbelow = [], []
    for m in LROWS:
        q0, qt = fam["s0"]["LATERAL"].get(m), tfam["s0"]["LATERAL"].get(m)
        vs_tail.append(("LATERAL", m, qt, q0))
        if not q0["delta"] < qt["delta"]:
            notbelow.append(m)
    wrows = sorted({"%s/%s %s" % (F, m, s) for F, m in rowset for s in ("s0", "s1") if worse(fam[s][F].get(m))})
    if not b_equal or not same_map:
        outcome = ("ADDENDUM 1 CHECK FAILS (required by Addendum 4): the box points or the stage-A map points differ from E2-TAIL, so the "
                   "wires reached perception. This is a tool or read defect, not a result; diagnose before reading (d).")
    elif st["d"] == "PASS":
        outcome = ("Addendum 4 committed outcome 1 (WIRE-OFF passes (d) at both seeds): the lateral deficit is carried by the planner reading "
                   "the wires at inference. Next, pre-registered in Addendum 4: BOX-OFF (the 6 box_gates alone) to attribute the wire; the "
                   "training lever is the attributed wire, under its own prereg; launching it is the PI's call.")
        if st["c"] != "PASS" or st["e"] != "PASS":
            outcome += " (c)/(e) not PASS while (d) passes: a trade, reported as such, not a pass."
    elif not notbelow:
        outcome = ("Addendum 4 committed outcome 2 (WIRE-OFF fails (d), every lateral gap below its E2-TAIL value): the wires carry part of "
                   "the deficit. BOX-OFF still runs to attribute that part; the residual rows (" + "; ".join(wrows) + ") get their own prereg.")
    else:
        outcome = ("Addendum 4 committed outcome 3 (a lateral gap is not below its E2-TAIL value: " + ", ".join(notbelow) + "): the wires at "
                   "inference do not carry the deficit. It lives in what training did to the weights outside the wires (trunk / map "
                   "co-adaptation); the lower map dose (0.02 share target, hold_r9e2_k2dc02) and the planner L2-SP anchor "
                   "(hold_r9_l2sp_planner, unlanded) stay the candidate training levers, both the PI's decisions.")
    outcome += " Clause (b) under the PI's 2026-11-17 ruling: " + btxt + "."
'''
s = s.replace(anchor, WO + anchor)
# woff notes block: insert before `readme = (`
anchor2 = 'readme = ("# E2-WISE " + UP'
assert s.count(anchor2) == 1
NB = '''if MODE == "woff":
    zt = prov["K2DC"]["zeroed"] or {}
    notes = ("## Notes on this read\\n\\n"
             "* Weights (`wire_off.py`, provenance in `stageA/weights/`): K2DC n_zeroed %s (Addendum 4: 8), untouched bit-equal %s; K0 n_zeroed %s,\\n"
             "  a byte copy of its tail (md5 equal). Zeroed tensors and their tail values: %s.\\n"
             "* Clause (b) is judged under the PI's 2026-11-17 ruling (landed 50856a00): %s.\\n\\n"
             "## Lateral seed-0 gaps vs E2-TAIL (paired K2DC - K0, T1, 70 episodes; Addendum 4's secondary readout)\\n\\n"
             "| row | E2-TAIL | WIRE-OFF |\\n|---|---|---|\\n" % (
                 prov["K2DC"]["n_zeroed"], prov["K2DC"]["n_untouched_bit_equal"], prov["K0"]["n_zeroed"],
                 "; ".join("`%s` %s" % (n, ", ".join("%+.4f" % x for x in v["values"][:2]) + (" ..." if v["shape"] != [1] else ""))
                           for n, v in sorted(zt.items())), btxt)
             + "".join("| %s / `%s` | %s | %s |\\n" % (F, m, cell(a), cell(b)) for F, m, a, b in vs_tail) + "\\n")
'''
s = s.replace(anchor2, NB + anchor2)
# provenance-loop perception assert uses PLANLIKE (weight_interp field); the Addendum-1 equality check text uses PERC_SAME via the mode test above
open(P, "wb").write(s.encode("utf-8"))
print("PATCHED", P)
